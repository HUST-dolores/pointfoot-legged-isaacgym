#!/usr/bin/env python3
"""Diagnose whether slope capability is friction- or design-limited.

The frozen slope generalist is evaluated on real inclined terrain for three
mechanical designs, seven Coulomb friction coefficients, and three fixed
seeds. The stand-only capability metric is unchanged from the corrected
six-dimensional slope search.
"""

import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np


OUTPUT = Path(
    os.environ.get(
        "SLOPE_FRICTION_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "slope_friction_diagnosis_v1_2026-07-26",
    )
)
os.environ["SIXD_OUT"] = str(OUTPUT)
os.environ["PHYS_EVAL_SLOPE_SUCCESS_MODE"] = "stand"

import run_sixd_single_objective_search as base  # noqa: E402


POLICY_RUN = os.environ.get(
    "SLOPE_FRICTION_POLICY_RUN",
    "Jul26_02-05-10_physadapt_sixdgen_slope_static_v2_"
    "s20260726_u600",
)
POLICY_CHECKPOINT = int(
    os.environ.get("SLOPE_FRICTION_POLICY_CKPT", "27600")
)
FRICTIONS = tuple(
    float(value)
    for value in os.environ.get(
        "SLOPE_FRICTION_LEVELS",
        "0.3,0.4,0.5,0.6,0.8,1.0,1.2",
    ).split(",")
    if value.strip()
)
SEEDS = tuple(
    int(value)
    for value in os.environ.get(
        "SLOPE_FRICTION_SEEDS",
        "20260726,20260727,20260728",
    ).split(",")
    if value.strip()
)
SLOPE_LEVELS = os.environ.get(
    "SLOPE_FRICTION_SLOPE_LEVELS",
    ",".join(str(value) for value in range(0, 41, 2)),
)
REPLICAS = int(os.environ.get("SLOPE_FRICTION_REPLICAS", "64"))

DESIGNS = {
    "nominal": np.ones(6, dtype=np.float64),
    "slope_winner": np.asarray(
        [
            0.934330690652132,
            0.9515915792435408,
            1.0156492045149208,
            0.6179544826969504,
            0.6174186555668711,
            0.6176245896145701,
        ],
        dtype=np.float64,
    ),
    "long_thigh_control": np.asarray(
        [1.2, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    ),
}


def friction_token(value):
    return f"{float(value):.2f}".replace(".", "p")


def make_job(label, design, friction, seed):
    design_id = base.design_key(design)
    directory = (
        OUTPUT
        / "evaluations"
        / f"mu_{friction_token(friction)}"
        / label
        / f"seed_{seed}"
    )
    result_path = directory / "result.json"
    environment = base.common_motor_environment()
    environment.update(
        {
            "PHYS_EVAL_TASK": "slope",
            "PHYS_EVAL_RUN": POLICY_RUN,
            "PHYS_EVAL_CKPT": str(POLICY_CHECKPOINT),
            "PHYS_EVAL_SEED": str(seed),
            "PHYS_EVAL_OUT": str(directory),
            "PHYS_EVAL_PREFIX": (
                f"slope_friction_mu{friction_token(friction)}_"
                f"{label}_{design_id}_s{seed}"
            ),
            "PHYS_EVAL_SETTLE_S": "2.0",
            "PHYS_EVAL_DURATION_S": "8.0",
            "PHYS_EVAL_SLOPE_STATIC": "1",
            "PHYS_EVAL_SLOPE_SUCCESS_MODE": "stand",
            "PHYS_EVAL_FRICTION": f"{float(friction):.8f}",
            "PHYS_EVAL_LEVELS": SLOPE_LEVELS,
            "PHYS_EVAL_REPLICAS": str(REPLICAS),
        }
    )
    base.apply_design_environment(environment, "PHYS_EVAL", design)
    job = base.Job(
        f"friction-{friction_token(friction)}-{label}-s{seed}",
        [
            base.PYTHON,
            "legged_gym/scripts/physical_capability_eval.py",
            "--headless",
        ],
        environment,
        directory,
        result_path,
    )
    return {
        "label": label,
        "design": design,
        "design_id": design_id,
        "friction": float(friction),
        "seed": int(seed),
        "job": job,
    }


def metric_from_result(result):
    summary = result["summary"]
    value = summary.get("metric_value_interpolated")
    if value is None:
        value = summary.get("metric_value")
    return float(value or 0.0)


def diagnostic_at_limit(result, capability):
    rows = result["per_level"]
    row = min(
        rows,
        key=lambda value: abs(float(value["level"]) - capability),
    )
    keys = (
        "level",
        "stand_success_rate",
        "fall_or_reset_rate",
        "slip_ratio_mean",
        "torque_saturation_fraction_mean",
        "progress_abs_max_mean_m",
    )
    return {key: row.get(key) for key in keys}


def save_records(records):
    base.atomic_json(OUTPUT / "RAW_RECORDS.json", records)
    fields = (
        "design_label",
        "design_id",
        "seed",
        "friction",
        "friction_angle_deg",
        "capability_deg",
        "right_censored",
        "robot_mass_kg",
        "motor_module_mass_kg",
        "result_path",
    )
    with (OUTPUT / "RAW_RECORDS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fields})


def aggregate(records):
    rows = []
    for friction in FRICTIONS:
        for label in DESIGNS:
            selected = [
                record
                for record in records
                if record["design_label"] == label
                and np.isclose(record["friction"], friction)
            ]
            values = np.asarray(
                [record["capability_deg"] for record in selected],
                dtype=np.float64,
            )
            rows.append(
                {
                    "friction": float(friction),
                    "friction_angle_deg": float(
                        np.rad2deg(np.arctan(friction))
                    ),
                    "design_label": label,
                    "design": base.design_dict(DESIGNS[label]),
                    "capability_mean_deg": float(np.mean(values)),
                    "capability_std_deg": float(np.std(values, ddof=1)),
                    "seed_count": len(values),
                    "right_censored_seed_count": int(
                        sum(record["right_censored"] for record in selected)
                    ),
                    "robot_mass_mean_kg": float(
                        np.mean(
                            [
                                record["robot_mass_kg"]
                                for record in selected
                            ]
                        )
                    ),
                    "motor_module_mass_kg": float(
                        selected[0]["motor_module_mass_kg"]
                    ),
                }
            )
    return rows


def variance_decomposition(records, design_labels=None):
    if design_labels is None:
        design_labels = tuple(DESIGNS)
    else:
        design_labels = tuple(design_labels)
    records = [
        record
        for record in records
        if record["design_label"] in design_labels
    ]
    values = np.asarray(
        [record["capability_deg"] for record in records],
        dtype=np.float64,
    )
    grand = float(np.mean(values))
    friction_means = {
        friction: float(
            np.mean(
                [
                    record["capability_deg"]
                    for record in records
                    if np.isclose(record["friction"], friction)
                ]
            )
        )
        for friction in FRICTIONS
    }
    design_means = {
        label: float(
            np.mean(
                [
                    record["capability_deg"]
                    for record in records
                    if record["design_label"] == label
                ]
            )
        )
        for label in design_labels
    }
    cell_means = {
        (friction, label): float(
            np.mean(
                [
                    record["capability_deg"]
                    for record in records
                    if np.isclose(record["friction"], friction)
                    and record["design_label"] == label
                ]
            )
        )
        for friction in FRICTIONS
        for label in design_labels
    }
    n_seed = len(SEEDS)
    ss_total = float(np.sum(np.square(values - grand)))
    ss_friction = float(
        len(design_labels)
        * n_seed
        * sum(
            (friction_means[friction] - grand) ** 2
            for friction in FRICTIONS
        )
    )
    ss_design = float(
        len(FRICTIONS)
        * n_seed
        * sum(
            (design_means[label] - grand) ** 2
            for label in design_labels
        )
    )
    ss_interaction = float(
        n_seed
        * sum(
            (
                cell_means[(friction, label)]
                - friction_means[friction]
                - design_means[label]
                + grand
            )
            ** 2
            for friction in FRICTIONS
            for label in design_labels
        )
    )
    ss_residual = max(
        0.0,
        ss_total - ss_friction - ss_design - ss_interaction,
    )
    denominator = max(ss_total, 1.0e-12)
    return {
        "design_labels": list(design_labels),
        "grand_mean_deg": grand,
        "sum_squares": {
            "friction": ss_friction,
            "design": ss_design,
            "friction_design_interaction": ss_interaction,
            "seed_residual": ss_residual,
            "total": ss_total,
        },
        "fraction_of_total_variation": {
            "friction": ss_friction / denominator,
            "design": ss_design / denominator,
            "friction_design_interaction": ss_interaction / denominator,
            "seed_residual": ss_residual / denominator,
        },
        "interpretation_note": (
            "Balanced descriptive two-way sum-of-squares decomposition; "
            "fractions are effect-size diagnostics, not causal proof."
        ),
    }


def comparison_rows(aggregates):
    lookup = {
        (row["friction"], row["design_label"]): row
        for row in aggregates
    }
    rows = []
    for friction in FRICTIONS:
        nominal = lookup[(friction, "nominal")][
            "capability_mean_deg"
        ]
        winner = lookup[(friction, "slope_winner")][
            "capability_mean_deg"
        ]
        control = lookup[(friction, "long_thigh_control")][
            "capability_mean_deg"
        ]
        rows.append(
            {
                "friction": float(friction),
                "friction_angle_deg": float(
                    np.rad2deg(np.arctan(friction))
                ),
                "nominal_deg": nominal,
                "slope_winner_deg": winner,
                "long_thigh_control_deg": control,
                "winner_minus_nominal_deg": winner - nominal,
                "nominal_minus_long_thigh_deg": nominal - control,
            }
        )
    return rows


def write_aggregate_csv(rows):
    fields = (
        "friction",
        "friction_angle_deg",
        "design_label",
        "capability_mean_deg",
        "capability_std_deg",
        "seed_count",
        "right_censored_seed_count",
        "robot_mass_mean_kg",
        "motor_module_mass_kg",
    )
    with (OUTPUT / "AGGREGATED_RESULTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def write_report(
    aggregates,
    comparisons,
    decomposition,
    focused_decomposition,
):
    lookup = {
        (row["friction"], row["design_label"]): row
        for row in aggregates
    }
    lines = [
        "# Slope friction diagnosis",
        "",
        (
            "Metric: three-seed mean of the 90% reliable stand-only "
            "incline limit on real inclined trimesh terrain."
        ),
        "",
        (
            "| friction | atan(mu) | nominal | slope winner | "
            "long-thigh control | winner - nominal |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['friction']:.2f} "
            f"| {row['friction_angle_deg']:.2f} "
            f"| {row['nominal_deg']:.3f} "
            f"| {row['slope_winner_deg']:.3f} "
            f"| {row['long_thigh_control_deg']:.3f} "
            f"| {row['winner_minus_nominal_deg']:+.3f} |"
        )
    fractions = decomposition["fraction_of_total_variation"]
    focused = focused_decomposition["fraction_of_total_variation"]
    lines.extend(
        [
            "",
            "## Descriptive variation split",
            "",
            f"- friction: {100.0 * fractions['friction']:.2f}%",
            f"- mechanical design: {100.0 * fractions['design']:.2f}%",
            (
                "- friction x design interaction: "
                f"{100.0 * fractions['friction_design_interaction']:.2f}%"
            ),
            f"- seed residual: {100.0 * fractions['seed_residual']:.2f}%",
            "",
            "## Nominal-vs-winner local variation split",
            "",
            f"- friction: {100.0 * focused['friction']:.2f}%",
            f"- mechanical design: {100.0 * focused['design']:.2f}%",
            (
                "- friction x design interaction: "
                f"{100.0 * focused['friction_design_interaction']:.2f}%"
            ),
            f"- seed residual: {100.0 * focused['seed_residual']:.2f}%",
            "",
            (
                "This focused split excludes the deliberately degraded "
                "long-thigh control and measures the local optimization "
                "space around two reasonable designs."
            ),
            "",
            (
                "`atan(mu)` is a static friction-cone reference for a "
                "passive block, not an equality expected of an actuated "
                "wheeled biped."
            ),
            "",
            "## Per-cell repeat uncertainty",
            "",
        ]
    )
    for friction in FRICTIONS:
        entries = []
        for label in DESIGNS:
            row = lookup[(friction, label)]
            entries.append(
                f"{label} {row['capability_mean_deg']:.3f} +/- "
                f"{row['capability_std_deg']:.3f} deg"
            )
        lines.append(f"- mu={friction:.2f}: " + "; ".join(entries))
    (OUTPUT / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def snapshot_sources():
    sources = (
        "legged_gym/utils/terrain.py",
        "legged_gym/utils/motor_cost.py",
        "legged_gym/envs/base/base_task.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py",
        "legged_gym/scripts/physical_capability_eval.py",
        "legged_gym/scripts/run_sixd_single_objective_search.py",
        "legged_gym/scripts/run_slope_friction_diagnosis.py",
    )
    destination = OUTPUT / "code_snapshot"
    for relative in sources:
        source = base.ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main():
    aggregate_only = (
        os.environ.get("SLOPE_FRICTION_AGGREGATE_ONLY", "0") == "1"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not aggregate_only:
        (OUTPUT / "STARTED").write_text(
            base.now_iso() + "\n", encoding="ascii"
        )
        (OUTPUT / "DONE").unlink(missing_ok=True)
        (OUTPUT / "EXIT_CODE").unlink(missing_ok=True)
    snapshot_sources()
    manifest = {
        "protocol": "slope_friction_diagnosis_v1_2026-07-26",
        "started_at": base.now_iso(),
        "policy": {
            "run": POLICY_RUN,
            "checkpoint": POLICY_CHECKPOINT,
        },
        "metric": "theta_stand_90_interpolated_deg",
        "success_mode": "stand",
        "terrain": "real inclined trimesh",
        "frictions": list(FRICTIONS),
        "friction_angle_reference_deg": [
            float(np.rad2deg(np.arctan(value)))
            for value in FRICTIONS
        ],
        "seeds": list(SEEDS),
        "replicas_per_level": REPLICAS,
        "slope_levels_deg": [
            float(value) for value in SLOPE_LEVELS.split(",")
        ],
        "designs": {
            label: base.design_dict(design)
            for label, design in DESIGNS.items()
        },
        "gpus": list(base.GPU_IDS),
    }
    base.atomic_json(OUTPUT / "RUN_MANIFEST.json", manifest)

    requests = [
        make_job(label, design, friction, seed)
        for friction in FRICTIONS
        for label, design in DESIGNS.items()
        for seed in SEEDS
    ]
    if not aggregate_only:
        base.run_jobs(
            "slope_friction_formal_sweep",
            [request["job"] for request in requests],
        )

    records = []
    for request in requests:
        result = base.read_json(request["job"].expected_result)
        capability = metric_from_result(result)
        records.append(
            {
                "design_label": request["label"],
                "design_id": request["design_id"],
                "design": base.design_dict(request["design"]),
                "seed": request["seed"],
                "friction": request["friction"],
                "friction_angle_deg": float(
                    np.rad2deg(np.arctan(request["friction"]))
                ),
                "capability_deg": capability,
                "right_censored": bool(
                    result["summary"][
                        "continuous_domain_right_censored"
                    ]
                ),
                "robot_mass_kg": float(result["robot_mass_kg_env0"]),
                "motor_module_mass_kg": base.motor_mass_kg(
                    request["design"]
                ),
                "diagnostic_at_capability": diagnostic_at_limit(
                    result, capability
                ),
                "result_path": str(request["job"].expected_result),
            }
        )
    save_records(records)
    aggregates = aggregate(records)
    comparisons = comparison_rows(aggregates)
    decomposition = variance_decomposition(records)
    focused_decomposition = variance_decomposition(
        records, ("nominal", "slope_winner")
    )
    base.atomic_json(OUTPUT / "AGGREGATED_RESULTS.json", aggregates)
    base.atomic_json(OUTPUT / "COMPARISONS.json", comparisons)
    base.atomic_json(
        OUTPUT / "VARIANCE_DECOMPOSITION.json", decomposition
    )
    base.atomic_json(
        OUTPUT / "FOCUSED_VARIANCE_DECOMPOSITION.json",
        focused_decomposition,
    )
    write_aggregate_csv(aggregates)
    write_report(
        aggregates,
        comparisons,
        decomposition,
        focused_decomposition,
    )
    if aggregate_only:
        print(
            f"[slope-friction] REAGGREGATED "
            f"{OUTPUT / 'FINAL_REPORT.md'}",
            flush=True,
        )
        return

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
            "evaluation_count": len(records),
            "gpu_state_after_children": gpu_state,
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
        f"[slope-friction] COMPLETE {OUTPUT / 'FINAL_REPORT.md'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        error_text = traceback.format_exc()
        (OUTPUT / "ERROR.log").write_text(
            error_text, encoding="utf-8"
        )
        (OUTPUT / "EXIT_CODE").write_text("1\n", encoding="ascii")
        base.update_status(
            stage="failed",
            failed_jobs=[error_text],
            eta_seconds=None,
        )
        print(error_text, file=sys.stderr, flush=True)
        raise
