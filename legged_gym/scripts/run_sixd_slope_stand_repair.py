#!/usr/bin/env python3
"""Repair the six-dimensional slope search with a stand-only success metric.

The previous search saved every per-environment metric but ranked designs with
an overly strict stationary-hold criterion. This script reclassifies those 48
evaluations as no-fall posture-stability trials, adds two GP-EI batches, and
formally repeats the corrected top three plus the nominal design.
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
        "SLOPE_STAND_SOURCE",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_single_objective_v1_2026-07-26",
    )
)
OUTPUT = Path(
    os.environ.get(
        "SLOPE_STAND_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_slope_stand_repair_v1_2026-07-26",
    )
)
os.environ["SIXD_OUT"] = str(OUTPUT)
os.environ["PHYS_EVAL_SLOPE_SUCCESS_MODE"] = "stand"

import run_sixd_single_objective_search as base  # noqa: E402


TASK = "slope"
BO1_COUNT = int(os.environ.get("SLOPE_STAND_BO1", "10"))
BO2_COUNT = int(os.environ.get("SLOPE_STAND_BO2", "4"))
ROLL_LIMIT_DEG = float(
    os.environ.get("PHYS_EVAL_HOLD_MAX_ROLL_DEG", "15.0")
)
PITCH_LIMIT_DEG = float(
    os.environ.get("PHYS_EVAL_HOLD_MAX_PITCH_DEG", "20.0")
)


def safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def continuous_reliable_limit(levels, success_rates, threshold=0.90):
    reliable = []
    still_reliable = True
    for rate in success_rates:
        still_reliable = still_reliable and rate >= threshold
        reliable.append(bool(still_reliable))
    valid = [
        float(level)
        for level, passes in zip(levels, reliable)
        if passes
    ]
    return (max(valid) if valid else float("nan")), reliable


def interpolated_reliable_limit(levels, success_rates, threshold=0.90):
    levels = np.asarray(levels, dtype=np.float64)
    rates = np.minimum.accumulate(
        np.asarray(success_rates, dtype=np.float64)
    )
    if rates[0] < threshold:
        return float("nan")
    below = np.flatnonzero(rates < threshold)
    if not len(below):
        return float(levels[-1])
    right = int(below[0])
    left = right - 1
    rate_span = rates[left] - rates[right]
    if rate_span <= 1.0e-12:
        return float(levels[left])
    fraction = (rates[left] - threshold) / rate_span
    return float(
        levels[left] + fraction * (levels[right] - levels[left])
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


def stand_summary(result):
    raw = np.load(source_path(result["raw_path"]))
    level = raw["level"]
    levels = np.unique(level)
    stand = (
        (~raw["failed"].astype(bool))
        & (np.rad2deg(raw["roll_p95_rad"]) <= ROLL_LIMIT_DEG)
        & (np.rad2deg(raw["pitch_p95_rad"]) <= PITCH_LIMIT_DEG)
    )
    rates = np.asarray(
        [float(np.mean(stand[level == value])) for value in levels],
        dtype=np.float64,
    )
    level_values = [float(value) for value in levels]
    rate_values = [float(value) for value in rates]
    value, continuous = continuous_reliable_limit(level_values, rate_values)
    interpolated = interpolated_reliable_limit(
        level_values, rate_values
    )
    return {
        "metric_name": "theta_stand_90",
        "metric_value": safe_float(value),
        "metric_value_interpolated": safe_float(interpolated),
        "unit": "deg",
        "success_threshold": 0.90,
        "success_mode": "stand",
        "success_definition": {
            "zero_motion_command": True,
            "no_fall_or_episode_reset": True,
            "maximum_roll_deg": ROLL_LIMIT_DEG,
            "maximum_pitch_deg": PITCH_LIMIT_DEG,
        },
        "levels_deg": level_values,
        "success_rates": rate_values,
        "continuous_reliable_mask": continuous,
        "lowest_level_passes": bool(rates[0] >= 0.90),
        "highest_level_fails": bool(rates[-1] < 0.90),
        "continuous_domain_right_censored": bool(all(continuous)),
    }


def reclassify_source_records():
    records = []
    for original in base.read_json(SOURCE / "ALL_SEARCH_RECORDS.json"):
        if original["task"] != TASK:
            continue
        result = base.read_json(source_path(original["result_path"]))
        summary = stand_summary(result)
        record = dict(original)
        record["source_phase"] = original["phase"]
        record["phase"] = "warm_reclassified_" + original["phase"]
        record["legacy_hold_objective"] = original["objective"]
        record["objective"] = float(
            summary.get("metric_value_interpolated")
            or summary.get("metric_value")
            or 0.0
        )
        record["summary"] = summary
        records.append(record)
    if len(records) != 48:
        raise RuntimeError(
            f"expected 48 source slope records, found {len(records)}"
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
    return {"slope": {"winner": rows[0], "finalists": rows}}


def write_final(final, search_records):
    base.atomic_json(OUTPUT / "FINAL_RESULTS.json", final)
    winner = final["slope"]["winner"]
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
        "# Corrected six-dimensional slope-stand result",
        "",
        "Metric: 90% reliable no-fall posture stability with zero command. "
        "Position drift, wheel contact, and slip remain diagnostics and do not "
        "veto a standing trial.",
        "",
        "| rank | thigh | shank | abad | hip | knee | wheel | "
        "theta_stand_90 mean +/- std (deg) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final["slope"]["finalists"]:
        design = row["design"]
        lines.append(
            f"| {row['rank']} | {design['thigh']:.4f} | "
            f"{design['shank']:.4f} | {design['abad']:.4f} | "
            f"{design['hip']:.4f} | {design['knee']:.4f} | "
            f"{design['wheel']:.4f} | {row['objective_mean']:.6g} "
            f"+/- {row['objective_std']:.3g} |"
        )
    lines.extend(
        [
            "",
            f"Warm-start reclassified evaluations: 48.",
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
        "legged_gym/scripts/run_sixd_slope_stand_repair.py",
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
            "protocol": "sixd_slope_stand_repair_v1_2026-07-26",
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
                "name": "theta_stand_90",
                "success_threshold": 0.90,
                "zero_motion_command": True,
                "no_fall_or_episode_reset": True,
                "maximum_roll_deg": ROLL_LIMIT_DEG,
                "maximum_pitch_deg": PITCH_LIMIT_DEG,
                "drift_contact_slip_are_diagnostics_only": True,
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
        search_records, BO1_COUNT, base.SEED + 300
    )
    gp_fits.append({"phase": "stand_bo1", "fit": fit})
    search_records.extend(
        evaluate("stand_bo1_search", designs, selected)
    )
    base.save_records(OUTPUT / "SEARCH_AFTER_BO1.json", search_records)

    designs, fit = base.propose_expected_improvement(
        search_records, BO2_COUNT, base.SEED + 400
    )
    gp_fits.append({"phase": "stand_bo2", "fit": fit})
    search_records.extend(
        evaluate("stand_bo2_search", designs, selected)
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
        "stand_formal_repeats",
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
    print(f"[slope-stand] COMPLETE {OUTPUT / 'FINAL_RESULTS.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "EXIT_CODE").write_text("1\n", encoding="ascii")
        raise
