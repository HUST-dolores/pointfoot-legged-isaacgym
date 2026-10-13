#!/usr/bin/env python3
"""Run formal fixed-slope controls for the uphill-speed co-design result."""

import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np


OUTPUT = Path(
    os.environ.get(
        "UPHILL_CONTROL_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_uphill_speed_10deg_explanatory_v1_2026-07-27",
    )
)
os.environ["SIXD_OUT"] = str(OUTPUT)
os.environ["PHYS_EVAL_SPEED_SUCCESS_MODE"] = "stable_actual"
os.environ["PHYS_EVAL_SPEED_SLOPE_DEG"] = os.environ.get(
    "UPHILL_CONTROL_SLOPE_DEG", "10"
)
os.environ["PHYS_EVAL_SLOPE_LENGTH"] = os.environ.get(
    "UPHILL_CONTROL_SLOPE_LENGTH", "110"
)
os.environ["PHYS_EVAL_SLOPE_HORIZONTAL_SCALE"] = os.environ.get(
    "UPHILL_CONTROL_HORIZONTAL_SCALE", "0.1"
)

import run_sixd_single_objective_search as base  # noqa: E402


RUN_NAME = os.environ.get(
    "UPHILL_CONTROL_RUN",
    "Jul27_09-40-33_physadapt_sixdgen_speed_ext_"
    "direct_stage1_3p4_s20260727_u300",
)
CHECKPOINT = int(os.environ.get("UPHILL_CONTROL_CKPT", "29400"))
COMMAND_MAX = float(os.environ.get("UPHILL_CONTROL_COMMAND_MAX", "5.2"))
SEEDS = tuple(
    int(value)
    for value in os.environ.get(
        "UPHILL_CONTROL_SEEDS", "20260727,20260728,20260729"
    ).split(",")
    if value.strip()
)

DESIGNS = {
    "flat_speed_champion": np.asarray(
        [
            0.9165454473346472,
            0.8298767622560264,
            0.807666621543467,
            1.1777029590681196,
            0.9549397187307477,
            1.5218089999631048,
        ],
        dtype=np.float64,
    ),
    "nominal_geometry_wheel_0p6": np.asarray(
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.6],
        dtype=np.float64,
    ),
    "nominal_geometry_wheel_1p6": np.asarray(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.6],
        dtype=np.float64,
    ),
}


def command_grid():
    values = list(
        np.round(np.arange(0.5, COMMAND_MAX + 1.0e-9, 0.1), 6)
    )
    if not values or not np.isclose(values[-1], COMMAND_MAX):
        values.append(COMMAND_MAX)
    return sorted(set(float(value) for value in values))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "STARTED").write_text(base.now_iso() + "\n", encoding="ascii")
    (OUTPUT / "DONE").unlink(missing_ok=True)
    (OUTPUT / "EXIT_CODE").unlink(missing_ok=True)
    levels = command_grid()
    base.FORMAL_LEVELS["speed"] = ",".join(
        f"{value:g}" for value in levels
    )
    selected = {
        "speed": {
            "run_name": RUN_NAME,
            "checkpoint": CHECKPOINT,
        }
    }
    records = base.evaluate_design_stage(
        "uphill_speed_explanatory_formal",
        {"speed": list(DESIGNS.values())},
        selected,
        fidelity="formal",
        seeds=SEEDS,
    )
    base.save_records(OUTPUT / "FORMAL_RECORDS.json", records)

    labels_by_id = {
        base.design_key(design): label
        for label, design in DESIGNS.items()
    }
    groups = {}
    for record in records:
        groups.setdefault(record["design_id"], []).append(record)
    rows = []
    for design_id, values in groups.items():
        objectives = [float(value["objective"]) for value in values]
        rows.append(
            {
                "label": labels_by_id[design_id],
                "design_id": design_id,
                "design": values[0]["design"],
                "design_vector": values[0]["design_vector"],
                "seeds": [value["seed"] for value in values],
                "objective_by_seed_mps": objectives,
                "objective_mean_mps": statistics.mean(objectives),
                "objective_std_mps": statistics.stdev(objectives),
                "command_at_metric_by_seed_mps": [
                    value["summary"]["command_at_metric_mps"]
                    for value in values
                ],
                "no_reset_rate_by_seed": [
                    value["summary"]["no_reset_rate_at_command"]
                    for value in values
                ],
                "positive_uphill_rate_by_seed": [
                    value["summary"][
                        "positive_uphill_progress_rate_at_command"
                    ]
                    for value in values
                ],
                "wheel_abs_torque_mean_nm_by_seed": [
                    value["summary"][
                        "wheel_abs_torque_mean_at_command_nm"
                    ]
                    for value in values
                ],
                "wheel_torque_ratio_mean_by_seed": [
                    value["summary"][
                        "wheel_torque_ratio_mean_at_command"
                    ]
                    for value in values
                ],
                "wheel_torque_ratio_p95_by_seed": [
                    value["summary"][
                        "wheel_torque_ratio_p95_mean_at_command"
                    ]
                    for value in values
                ],
                "wheel_saturation_fraction_by_seed": [
                    value["summary"][
                        "wheel_torque_saturation_fraction_at_command"
                    ]
                    for value in values
                ],
                "wheel_mechanical_power_mean_w_by_seed": [
                    value["summary"][
                        "wheel_mechanical_power_mean_at_command_w"
                    ]
                    for value in values
                ],
                "right_censored_by_seed": [
                    value["summary"][
                        "right_censored_by_training_command_domain"
                    ]
                    for value in values
                ],
                "result_paths": [
                    value["result_path"] for value in values
                ],
            }
        )
    rows.sort(key=lambda row: -row["objective_mean_mps"])
    base.atomic_json(
        OUTPUT / "EXPLANATORY_RESULTS.json",
        {
            "protocol": "uphill_speed_explanatory_v1_2026-07-27",
            "policy": selected["speed"],
            "slope_deg": float(
                os.environ["PHYS_EVAL_SPEED_SLOPE_DEG"]
            ),
            "positive_direction": (
                "+world-x and +world-z along the ascending tangent"
            ),
            "command_levels_mps": levels,
            "seeds": list(SEEDS),
            "results": rows,
        },
    )
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "label": row["label"],
                **row["design"],
                "objective_mean_mps": row["objective_mean_mps"],
                "objective_std_mps": row["objective_std_mps"],
            }
        )
    with (OUTPUT / "EXPLANATORY_RESULTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(flat_rows[0])
        )
        writer.writeheader()
        writer.writerows(flat_rows)

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
            "job_count": len(records),
            "failed_jobs": [],
            "gpu_state_after_children": gpu_state,
        },
    )
    (OUTPUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
    (OUTPUT / "DONE").write_text(base.now_iso() + "\n", encoding="ascii")
    print(json.dumps(rows, indent=2, ensure_ascii=True))
    time.sleep(2.0)


if __name__ == "__main__":
    main()
