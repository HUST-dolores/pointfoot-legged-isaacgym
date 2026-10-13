#!/usr/bin/env python3
"""Test whether a fixed nominal robot can reach the frozen 2.5 m/s gate."""

import json
import os
from pathlib import Path
import shutil

import numpy as np


OUTPUT = Path(
    os.environ.get(
        "NOMINAL_FEAS_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "uphill_power_nominal_feasibility_2026-07-27",
    )
)
BASE_RUN = os.environ.get(
    "NOMINAL_FEAS_BASE_RUN",
    "Jul27_09-40-33_physadapt_sixdgen_speed_ext_"
    "direct_stage1_3p4_s20260727_u300",
)
BASE_CHECKPOINT = int(
    os.environ.get("NOMINAL_FEAS_BASE_CKPT", "29400")
)
BASE_COMMAND_MAX = float(
    os.environ.get("NOMINAL_FEAS_BASE_COMMAND_MAX", "5.2")
)
UPDATES = int(os.environ.get("NOMINAL_FEAS_UPDATES", "400"))
SAVE_INTERVAL = int(
    os.environ.get("NOMINAL_FEAS_SAVE_INTERVAL", "50")
)
TRAIN_SEED = int(os.environ.get("NOMINAL_FEAS_TRAIN_SEED", "20260730"))
EVAL_SEED = int(os.environ.get("NOMINAL_FEAS_EVAL_SEED", "20260732"))
MIN_SPEED_MPS = float(
    os.environ.get("NOMINAL_FEAS_MIN_SPEED_MPS", "2.5")
)
STEP_COMPONENT = float(
    os.environ.get("NOMINAL_FEAS_STEP_COMPONENT", "0.0")
)
if STEP_COMPONENT < 0.0:
    raise ValueError("NOMINAL_FEAS_STEP_COMPONENT must be non-negative")

os.environ["SIXD_OUT"] = str(OUTPUT)
os.environ["PHYS_EVAL_SPEED_SUCCESS_MODE"] = "stable_actual"
os.environ["PHYS_EVAL_STEP_COMPONENT"] = str(STEP_COMPONENT)
os.environ["PHYS_EVAL_SPEED_SLOPE_DEG"] = "10"
os.environ["PHYS_EVAL_SLOPE_LENGTH"] = "110"
os.environ["PHYS_EVAL_SLOPE_HORIZONTAL_SCALE"] = "0.1"

import run_sixd_single_objective_search as base  # noqa: E402


NOMINAL = np.ones(6, dtype=np.float64)


def comma_values(values):
    return ",".join(f"{float(value):g}" for value in values)


def configured_levels(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return tuple(float(value) for value in default)
    values = tuple(
        float(value) for value in raw.split(",") if value.strip()
    )
    if len(values) != 8:
        raise ValueError(f"{name} must contain exactly eight levels")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return values


BRANCHES = {
    "conservative": {
        "levels": configured_levels(
            "NOMINAL_FEAS_CONSERVATIVE_LEVELS",
            (0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6),
        ),
        "learning_rate": float(
            os.environ.get("NOMINAL_FEAS_CONSERVATIVE_LR", "2e-5")
        ),
        "tracking_sigma": float(
            os.environ.get("NOMINAL_FEAS_CONSERVATIVE_SIGMA", "0.05")
        ),
        "tracking_reward": float(
            os.environ.get("NOMINAL_FEAS_CONSERVATIVE_REWARD", "4")
        ),
        "velocity_error_reward": float(
            os.environ.get("NOMINAL_FEAS_CONSERVATIVE_ERROR", "-1")
        ),
    },
    "broader_tracking": {
        "levels": configured_levels(
            "NOMINAL_FEAS_BROADER_LEVELS",
            (0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9, 3.3),
        ),
        "learning_rate": float(
            os.environ.get("NOMINAL_FEAS_BROADER_LR", "5e-5")
        ),
        "tracking_sigma": float(
            os.environ.get("NOMINAL_FEAS_BROADER_SIGMA", "0.20")
        ),
        "tracking_reward": float(
            os.environ.get("NOMINAL_FEAS_BROADER_REWARD", "4")
        ),
        "velocity_error_reward": float(
            os.environ.get("NOMINAL_FEAS_BROADER_ERROR", "-2")
        ),
    },
}


def make_training_job(label, spec):
    directory = OUTPUT / "training" / label
    result_path = directory / "result.json"
    environment = base.common_motor_environment()
    environment.update(
        {
            "PHYS_MAX_GPU_CONTACT_PAIRS": str(2**25),
            "PHYS_TRAIN_TASK": "speed",
            "PHYS_TRAIN_BASE_RUN": BASE_RUN,
            "PHYS_TRAIN_BASE_CKPT": str(BASE_CHECKPOINT),
            "PHYS_TRAIN_RANDOMIZE_DESIGN": "0",
            "PHYS_TRAIN_THIGH": "1",
            "PHYS_TRAIN_SHANK": "1",
            "PHYS_TRAIN_ABAD": "1",
            "PHYS_TRAIN_HIP": "1",
            "PHYS_TRAIN_KNEE": "1",
            "PHYS_TRAIN_WHEEL": "1",
            "PHYS_TRAIN_NUM_ENVS": "4096",
            "PHYS_TRAIN_UPDATES": str(UPDATES),
            "PHYS_TRAIN_SAVE_INTERVAL": str(SAVE_INTERVAL),
            "PHYS_TRAIN_SEED": str(TRAIN_SEED),
            "PHYS_TRAIN_TAG": (
                f"power_nominal_feas_{label}_s{TRAIN_SEED}_u{UPDATES}"
            ),
            "PHYS_TRAIN_RESULT_JSON": str(result_path),
            "PHYS_TRAIN_CLEARANCE_LEVELS": (
                "0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22"
            ),
            "PHYS_TRAIN_SLOPE_LEVELS": "0,3,6,9,12,15,18,21",
            "PHYS_TRAIN_PAYLOAD_LEVELS": "0,5,10,15,20,25,30,35",
            "PHYS_TRAIN_SPEED_LEVELS": comma_values(spec["levels"]),
            "PHYS_TRAIN_SPEED_STEP_COMPONENT": str(STEP_COMPONENT),
            "PHYS_TRAIN_SPEED_SLOPE_DEG": "10",
            "PHYS_TRAIN_TERRAIN_LENGTH": "240",
            "PHYS_TRAIN_SLOPE_HORIZONTAL_SCALE": "0.1",
            "PHYS_TRAIN_TERRAIN_WIDTH": "6",
            "PHYS_TRAIN_TERRAIN_BORDER": "10",
            "PHYS_TRAIN_LEARNING_RATE": str(spec["learning_rate"]),
            "PHYS_TRAIN_SCHEDULE": "fixed",
            "PHYS_TRAIN_ENTROPY_COEF": "0.0005",
            "PHYS_TRAIN_EST_LEARNING_RATE": "0",
            "PHYS_TRAIN_TOTAL_VEL_REWARD": str(
                spec["tracking_reward"]
            ),
            "PHYS_TRAIN_TOTAL_VEL_SIGMA": str(
                spec["tracking_sigma"]
            ),
            "PHYS_TRAIN_TOTAL_VEL_ERROR_REWARD": str(
                spec["velocity_error_reward"]
            ),
        }
    )
    return base.Job(
        f"train-nominal-feas-{label}",
        [
            base.PYTHON,
            "legged_gym/scripts/train_physical_capability.py",
            "--headless",
        ],
        environment,
        directory,
        result_path,
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "STARTED").write_text(base.now_iso() + "\n", encoding="ascii")
    (OUTPUT / "DONE").unlink(missing_ok=True)
    (OUTPUT / "EXIT_CODE").unlink(missing_ok=True)
    jobs = [
        make_training_job(label, spec)
        for label, spec in BRANCHES.items()
    ]
    base.run_jobs("nominal_feasibility_training", jobs)
    training = {
        job.directory.name: base.read_json(job.expected_result)
        for job in jobs
    }
    for label, result in training.items():
        design_spec = result.get("design_spec", {})
        if not bool(design_spec.get("motor_power_envelope", False)):
            raise RuntimeError(
                f"{label} trained without the motor power envelope"
            )
        actual_step = float(
            result.get("speed_step_component_mps", float("nan"))
        )
        if not np.isclose(actual_step, STEP_COMPONENT):
            raise RuntimeError(
                f"{label} speed-step mismatch: expected "
                f"{STEP_COMPONENT}, got {actual_step}"
            )
    base.atomic_json(OUTPUT / "TRAINING_RESULTS.json", training)

    policies = [
        {
            "label": "base_policy",
            "run_name": BASE_RUN,
            "checkpoint": BASE_CHECKPOINT,
            "training_command_max_mps": BASE_COMMAND_MAX,
        }
    ]
    for label, result in training.items():
        for checkpoint in range(
            BASE_CHECKPOINT + SAVE_INTERVAL,
            int(result["final_iteration"]) + 1,
            SAVE_INTERVAL,
        ):
            policies.append(
                {
                    "label": f"{label}_ckpt{checkpoint}",
                    "run_name": result["run_name"],
                    "checkpoint": checkpoint,
                    "training_command_max_mps": max(
                        result["speed_levels_mps"]
                    ),
                }
            )

    command_max = max(
        policy["training_command_max_mps"] for policy in policies
    )
    levels = list(
        np.round(np.arange(0.5, command_max + 1.0e-9, 0.1), 6)
    )
    base.FORMAL_LEVELS["speed"] = comma_values(levels)
    requests = []
    request_policy = {}
    for policy in policies:
        request = base.EvalRequest(
            "speed",
            policy["run_name"],
            policy["checkpoint"],
            NOMINAL,
            EVAL_SEED,
            f"nominal_feas_{policy['label']}",
            "formal",
        )
        requests.append(request)
        request_policy[id(request)] = policy
    base.run_jobs(
        "nominal_feasibility_checkpoint_validation",
        [request.job for request in requests],
    )

    rows = []
    for request in requests:
        policy = request_policy[id(request)]
        result = base.read_json(request.job.expected_result)
        power_model = result.get("motor_power_model", {})
        if not bool(power_model.get("enabled", False)):
            raise RuntimeError("nominal feasibility ran without power envelope")
        actual_step = float(
            result.get("command_decomposition", {}).get(
                "step_component_mps", float("nan")
            )
        )
        if not np.isclose(actual_step, STEP_COMPONENT):
            raise RuntimeError(
                f"evaluation speed-step mismatch: expected "
                f"{STEP_COMPONENT}, got {actual_step}"
            )
        rows.append(
            {
                **policy,
                "objective_mps": base.objective_from_result(
                    "speed", result
                ),
                "command_at_metric_mps": result["summary"][
                    "command_at_metric_mps"
                ],
                "no_reset_rate_at_metric": result["summary"][
                    "no_reset_rate_at_command"
                ],
                "result_path": str(request.job.expected_result),
            }
        )
    rows.sort(
        key=lambda row: (-row["objective_mps"], row["checkpoint"])
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    passed = rows[0]["objective_mps"] >= MIN_SPEED_MPS
    report = {
        "protocol": "uphill_power_nominal_feasibility_2026-07-27",
        "base_policy": {
            "run_name": BASE_RUN,
            "checkpoint": BASE_CHECKPOINT,
            "training_command_max_mps": BASE_COMMAND_MAX,
        },
        "branches": BRANCHES,
        "training_seed": TRAIN_SEED,
        "evaluation_seed": EVAL_SEED,
        "minimum_speed_mps": MIN_SPEED_MPS,
        "speed_step_component_mps": STEP_COMPONENT,
        "passed": passed,
        "selected": rows[0],
        "policies": rows,
    }
    base.atomic_json(OUTPUT / "FEASIBILITY.json", report)
    base.atomic_json(
        OUTPUT / "SELECTED_POLICY.json",
        {
            "run_name": rows[0]["run_name"],
            "checkpoint": rows[0]["checkpoint"],
            "training_command_max_mps": rows[0][
                "training_command_max_mps"
            ],
            "training_slope_deg": 10.0,
            "validation": rows,
        },
    )
    (OUTPUT / ("PASS" if passed else "FAIL")).write_text(
        base.now_iso() + "\n", encoding="ascii"
    )
    (OUTPUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
    (OUTPUT / "DONE").write_text(base.now_iso() + "\n", encoding="ascii")
    for relative in (
        "legged_gym/scripts/run_uphill_power_nominal_feasibility.py",
        "legged_gym/scripts/train_physical_capability.py",
        "legged_gym/scripts/physical_capability_eval.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py",
        "legged_gym/utils/motor_cost.py",
    ):
        source = base.ROOT / relative
        target = OUTPUT / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(
        f"[nominal-feasibility] {'PASS' if passed else 'FAIL'} "
        f"best={rows[0]['objective_mps']:.6f}",
        flush=True,
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
