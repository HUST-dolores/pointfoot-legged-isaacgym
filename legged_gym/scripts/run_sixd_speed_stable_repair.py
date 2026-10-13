#!/usr/bin/env python3
"""Repair the six-dimensional speed search using stable actual speed.

The original V90 objective penalized command-tracking error. This script
reclassifies the saved speed trials using the 10th percentile of actual
forward speed, with every reset trial assigned zero speed. It then adds two
GP-EI batches and formally repeats the corrected top three plus nominal.
"""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


SOURCE = Path(
    os.environ.get(
        "SPEED_STABLE_SOURCE",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_single_objective_v1_2026-07-26",
    )
)
OUTPUT = Path(
    os.environ.get(
        "SPEED_STABLE_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_speed_stable_repair_v1_2026-07-26",
    )
)
os.environ["SIXD_OUT"] = str(OUTPUT)
os.environ["PHYS_EVAL_SPEED_SUCCESS_MODE"] = "stable_actual"

import run_sixd_single_objective_search as base  # noqa: E402


TASK = "speed"
BO1_COUNT = int(os.environ.get("SPEED_STABLE_BO1", "10"))
BO2_COUNT = int(os.environ.get("SPEED_STABLE_BO2", "4"))
TRAINING_COMMAND_MAX_MPS = float(
    os.environ.get("SPEED_STABLE_TRAINING_COMMAND_MAX_MPS", "2.6")
)
SEARCH_LEVELS = (0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6)
FORMAL_LEVELS = tuple(
    float(value) for value in np.round(np.arange(0.5, 2.61, 0.1), 1)
)
base.SEARCH_LEVELS[TASK] = ",".join(
    f"{value:g}" for value in SEARCH_LEVELS
)
base.FORMAL_LEVELS[TASK] = ",".join(
    f"{value:.1f}" for value in FORMAL_LEVELS
)


def source_path(remote_path):
    path = Path(remote_path)
    try:
        relative = path.relative_to(SOURCE)
    except ValueError:
        try:
            run_root_id = path.parts.index(SOURCE.name)
        except ValueError:
            return path
        relative = Path(*path.parts[run_root_id + 1 :])
    return SOURCE / relative


def stable_actual_summary(result):
    raw = np.load(source_path(result["raw_path"]))
    level = raw["level"]
    failed = raw["failed"].astype(bool)
    actual_speed = np.maximum(raw["vx_mean"], 0.0)
    training_limit = float(
        result.get("summary", {}).get(
            "training_command_max_mps",
            TRAINING_COMMAND_MAX_MPS,
        )
    )
    rows = []
    for value in np.unique(level):
        if float(value) > training_limit + 1.0e-9:
            continue
        mask = level == value
        score = np.where(~failed[mask], actual_speed[mask], 0.0)
        rows.append(
            {
                "level": float(value),
                "trials": int(mask.sum()),
                "no_reset_rate": float(np.mean(~failed[mask])),
                "actual_speed_mean_mps": float(
                    np.mean(actual_speed[mask])
                ),
                "stable_actual_speed_q10_mps": float(
                    np.quantile(score, 0.10)
                ),
            }
        )
    if not rows:
        raise RuntimeError("no in-domain speed rows in source result")
    best = max(
        rows,
        key=lambda row: (
            row["stable_actual_speed_q10_mps"],
            -row["level"],
        ),
    )
    return {
        "metric_name": "V_stable_actual_90",
        "metric_value": best["stable_actual_speed_q10_mps"],
        "unit": "m/s",
        "success_mode": "stable_actual",
        "reliability_quantile": 0.10,
        "command_at_metric_mps": best["level"],
        "no_reset_rate_at_command": best["no_reset_rate"],
        "actual_speed_mean_at_command_mps": best[
            "actual_speed_mean_mps"
        ],
        "training_command_max_mps": training_limit,
        "in_domain_command_levels_mps": [
            row["level"] for row in rows
        ],
        "right_censored_by_training_command_domain": bool(
            np.isclose(best["level"], rows[-1]["level"])
        ),
        "per_level_reclassification": rows,
        "success_definition": {
            "failed_or_reset_trial_score_mps": 0.0,
            "nonfailed_trial_score": (
                "mean actual forward speed, clipped below at zero"
            ),
            "command_tracking_error_used": False,
            "roll_threshold_used": False,
            "metric": (
                "maximum 10th-percentile trial score over "
                "in-training-domain command levels"
            ),
        },
    }


def reclassify_source_records():
    records = []
    for original in base.read_json(SOURCE / "ALL_SEARCH_RECORDS.json"):
        if original["task"] != TASK:
            continue
        result = base.read_json(source_path(original["result_path"]))
        summary = stable_actual_summary(result)
        record = dict(original)
        record["source_phase"] = original["phase"]
        record["phase"] = "warm_reclassified_" + original["phase"]
        record["legacy_tracking_objective"] = original["objective"]
        record["objective"] = float(summary["metric_value"])
        record["summary"] = summary
        records.append(record)
    if len(records) != 48:
        raise RuntimeError(
            f"expected 48 source speed records, found {len(records)}"
        )
    return records


def evaluate(phase, designs, selected, fidelity="search", seeds=None):
    if seeds is None:
        seeds = (base.SEED,)
    return base.evaluate_design_stage(
        phase,
        {TASK: base.unique_designs(designs)},
        {TASK: selected},
        fidelity=fidelity,
        seeds=seeds,
    )


def aggregate_formal(records):
    groups = {}
    for record in records:
        groups.setdefault(record["design_id"], []).append(record)
    rows = []
    for design_id, values in groups.items():
        objective = np.asarray(
            [record["objective"] for record in values],
            dtype=np.float64,
        )
        rows.append(
            {
                "task": TASK,
                "design_id": design_id,
                "design": values[0]["design"],
                "design_vector": values[0]["design_vector"],
                "seed_count": len(values),
                "seeds": [record["seed"] for record in values],
                "objective_by_seed": objective.tolist(),
                "objective_mean": float(np.mean(objective)),
                "objective_std": float(np.std(objective, ddof=1)),
                "command_at_metric_by_seed": [
                    record["summary"]["command_at_metric_mps"]
                    for record in values
                ],
                "no_reset_rate_at_command_by_seed": [
                    record["summary"]["no_reset_rate_at_command"]
                    for record in values
                ],
                "right_censored_by_training_domain_by_seed": [
                    record["summary"][
                        "right_censored_by_training_command_domain"
                    ]
                    for record in values
                ],
                "motor_module_mass_kg": values[0][
                    "motor_module_mass_kg"
                ],
                "result_paths": [
                    record["result_path"] for record in values
                ],
                "summaries": [record["summary"] for record in values],
            }
        )
    rows.sort(
        key=lambda row: (
            -row["objective_mean"],
            row["objective_std"],
            row["motor_module_mass_kg"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {"speed": {"winner": rows[0], "finalists": rows}}


def write_final(final, search_records):
    base.atomic_json(OUTPUT / "FINAL_RESULTS.json", final)
    winner = final[TASK]["winner"]
    with (OUTPUT / "FINAL_RESULTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        row = {
            "task": TASK,
            **winner["design"],
            "objective_mean": winner["objective_mean"],
            "objective_std": winner["objective_std"],
            "seed_count": winner["seed_count"],
            "motor_module_mass_kg": winner["motor_module_mass_kg"],
        }
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    lines = [
        "# Corrected six-dimensional stable-actual-speed result",
        "",
        "Metric: maximum in-domain 10th percentile of actual mean forward "
        "speed, with every reset trial assigned zero speed. Command-tracking "
        "error does not enter the metric.",
        "",
        "| rank | thigh | shank | abad | hip | knee | wheel | "
        "V_stable_actual_90 mean +/- std (m/s) | commands by seed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in final[TASK]["finalists"]:
        design = row["design"]
        commands = "/".join(
            f"{value:g}" for value in row["command_at_metric_by_seed"]
        )
        lines.append(
            f"| {row['rank']} | {design['thigh']:.4f} | "
            f"{design['shank']:.4f} | {design['abad']:.4f} | "
            f"{design['hip']:.4f} | {design['knee']:.4f} | "
            f"{design['wheel']:.4f} | {row['objective_mean']:.6g} "
            f"+/- {row['objective_std']:.3g} | {commands} |"
        )
    lines.extend(
        [
            "",
            "Training-domain command maximum: "
            f"{TRAINING_COMMAND_MAX_MPS:g} m/s.",
            "A result whose selected command is 2.6 m/s is right-censored "
            "by the training command domain.",
            "Warm-start reclassified evaluations: 48.",
            f"New BO evaluations: {BO1_COUNT + BO2_COUNT}.",
            f"Total corrected search records: {len(search_records)}.",
            f"Formal seeds: {list(base.FORMAL_SEEDS)}.",
        ]
    )
    (OUTPUT / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def snapshot_sources():
    for relative in (
        "legged_gym/scripts/physical_capability_eval.py",
        "legged_gym/scripts/run_sixd_single_objective_search.py",
        "legged_gym/scripts/run_sixd_speed_stable_repair.py",
        "legged_gym/envs/base/base_task.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py",
        "legged_gym/utils/motor_cost.py",
    ):
        source = base.ROOT / relative
        target = OUTPUT / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "STARTED").write_text(base.now_iso() + "\n", encoding="ascii")
    (OUTPUT / "DONE").unlink(missing_ok=True)
    (OUTPUT / "EXIT_CODE").unlink(missing_ok=True)
    snapshot_sources()
    selected = base.read_json(SOURCE / "selected_generalists.json")[TASK]
    base.atomic_json(
        OUTPUT / "RUN_MANIFEST.json",
        {
            "protocol": "sixd_speed_stable_repair_v1_2026-07-26",
            "started_at": base.now_iso(),
            "source": str(SOURCE),
            "output": str(OUTPUT),
            "policy": {
                "run_name": selected["run_name"],
                "checkpoint": selected["checkpoint"],
            },
            "parameters": list(base.PARAMETERS),
            "bounds": {
                name: [float(low), float(high)]
                for name, low, high in zip(
                    base.PARAMETERS, base.LOWER, base.UPPER
                )
            },
            "metric": {
                "name": "V_stable_actual_90",
                "reset_trial_score_mps": 0.0,
                "actual_speed_quantile": 0.10,
                "command_tracking_error_used": False,
                "roll_threshold_used": False,
                "training_command_max_mps": TRAINING_COMMAND_MAX_MPS,
                "search_commands_mps": list(SEARCH_LEVELS),
                "formal_commands_mps": list(FORMAL_LEVELS),
            },
            "warm_reclassified_count": 48,
            "new_bo_batches": [BO1_COUNT, BO2_COUNT],
            "formal_seeds": list(base.FORMAL_SEEDS),
        },
    )

    warm = reclassify_source_records()
    base.save_records(OUTPUT / "WARM_RECLASSIFIED_RECORDS.json", warm)
    search_records = list(warm)
    gp_fits = []

    designs, fit = base.propose_expected_improvement(
        search_records, BO1_COUNT, base.SEED + 500
    )
    gp_fits.append({"phase": "stable_speed_bo1", "fit": fit})
    search_records.extend(
        evaluate("stable_speed_bo1_search", designs, selected)
    )
    base.save_records(OUTPUT / "SEARCH_AFTER_BO1.json", search_records)

    designs, fit = base.propose_expected_improvement(
        search_records, BO2_COUNT, base.SEED + 600
    )
    gp_fits.append({"phase": "stable_speed_bo2", "fit": fit})
    search_records.extend(
        evaluate("stable_speed_bo2_search", designs, selected)
    )
    base.save_records(OUTPUT / "ALL_SEARCH_RECORDS.json", search_records)
    base.atomic_json(OUTPUT / "GP_FITS.json", gp_fits)

    ranked = sorted(
        search_records,
        key=lambda record: (
            -record["objective"],
            record["motor_module_mass_kg"],
        ),
    )
    formal_designs = [
        np.asarray(record["design_vector"], dtype=np.float64)
        for record in ranked[:3]
    ]
    formal_designs = base.unique_designs(
        formal_designs + [base.NOMINAL]
    )
    formal_records = evaluate(
        "stable_speed_formal_repeats",
        formal_designs,
        selected,
        fidelity="formal",
        seeds=base.FORMAL_SEEDS,
    )
    base.save_records(OUTPUT / "FORMAL_RECORDS.json", formal_records)
    final = aggregate_formal(formal_records)
    write_final(final, search_records)

    time.sleep(2.0)
    gpu_state = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    ).stdout.strip()
    base.atomic_json(
        OUTPUT / "LIFECYCLE.json",
        {
            "completed_at": base.now_iso(),
            "gpu_state_after_children": gpu_state,
            "warm_reclassified_count": len(warm),
            "new_search_result_count": len(search_records) - len(warm),
            "formal_result_count": len(formal_records),
        },
    )
    base.update_status(
        stage="complete",
        completed_jobs=1,
        total_jobs=1,
        failed_jobs=[],
        eta_seconds=0.0,
    )
    (OUTPUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
    (OUTPUT / "DONE").write_text(base.now_iso() + "\n", encoding="ascii")
    print(
        f"[speed-stable] COMPLETE {OUTPUT / 'FINAL_RESULTS.json'}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "EXIT_CODE").write_text("1\n", encoding="ascii")
        raise
