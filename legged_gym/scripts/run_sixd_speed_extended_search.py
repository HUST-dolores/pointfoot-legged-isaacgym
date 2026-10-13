#!/usr/bin/env python3
"""Train an extended-domain speed generalist and rerun six-dimensional BO.

Two equal-budget curriculum branches continue an existing speed generalist.
Intermediate checkpoints are compared on held-out mechanical designs using
stable actual speed. The selected frozen policy can then be used for a fresh
34-point search, two GP-EI batches, and three-seed formal repeats.

Selection-only mode stops after checkpoint validation. Frozen-policy mode
skips training and validation, allowing an adaptive outer loop to perform
several inexpensive domain extensions before one final co-design search.
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
        "SPEED_EXT_SOURCE",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_single_objective_v1_2026-07-26",
    )
)
OUTPUT = Path(
    os.environ.get(
        "SPEED_EXT_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_speed_extended_v1_2026-07-26",
    )
)
os.environ["SIXD_OUT"] = str(OUTPUT)
os.environ["PHYS_EVAL_SPEED_SUCCESS_MODE"] = "stable_actual"
STEP_COMPONENT = float(
    os.environ.get("SPEED_EXT_STEP_COMPONENT", "0.40")
)
if STEP_COMPONENT < 0.0:
    raise ValueError("SPEED_EXT_STEP_COMPONENT must be non-negative")
os.environ["PHYS_EVAL_STEP_COMPONENT"] = str(STEP_COMPONENT)
os.environ["PHYS_TRAIN_SPEED_STEP_COMPONENT"] = str(STEP_COMPONENT)

import run_sixd_single_objective_search as base  # noqa: E402


TASK = "speed"
BASE_RUN = os.environ.get(
    "SPEED_EXT_BASE_RUN",
    "Jul26_01-23-14_physadapt_sixdgen_speed_s20260726_u800",
)
BASE_CHECKPOINT = int(os.environ.get("SPEED_EXT_BASE_CKPT", "27600"))
BASE_COMMAND_MAX = float(
    os.environ.get("SPEED_EXT_BASE_COMMAND_MAX", "2.6")
)
STAGE_UPDATES = int(os.environ.get("SPEED_EXT_STAGE_UPDATES", "400"))
BO1_COUNT = int(os.environ.get("SPEED_EXT_BO1", "10"))
BO2_COUNT = int(os.environ.get("SPEED_EXT_BO2", "4"))
FINALIST_COUNT = int(
    os.environ.get("SPEED_EXT_FINALIST_COUNT", "0")
)
FINALIST_SEEDS = tuple(
    int(value)
    for value in os.environ.get(
        "SPEED_EXT_FINALIST_SEEDS",
        ",".join(str(value) for value in base.FORMAL_SEEDS),
    ).split(",")
    if value.strip()
)
FORMAL_TOP_COUNT = int(
    os.environ.get("SPEED_EXT_FORMAL_TOP_COUNT", "3")
)
SELECTION_ONLY = os.environ.get(
    "SPEED_EXT_SELECTION_ONLY", "0"
) == "1"
FROZEN_RUN = os.environ.get("SPEED_EXT_FROZEN_RUN", "").strip()
FROZEN_CHECKPOINT = int(
    os.environ.get("SPEED_EXT_FROZEN_CKPT", "0")
)
FROZEN_COMMAND_MAX = float(
    os.environ.get("SPEED_EXT_FROZEN_COMMAND_MAX", "0")
)
FIXED_SLOPE_DEG = float(
    os.environ.get("SPEED_EXT_FIXED_SLOPE_DEG", "0.0")
)
BASE_TRAINING_SLOPE_DEG = float(
    os.environ.get("SPEED_EXT_BASE_TRAINING_SLOPE_DEG", "0.0")
)
EVAL_SLOPE_LENGTH = float(
    os.environ.get("SPEED_EXT_EVAL_SLOPE_LENGTH", "110.0")
)
TRAIN_SLOPE_LENGTH = float(
    os.environ.get("SPEED_EXT_TRAIN_SLOPE_LENGTH", "240.0")
)
SLOPE_HORIZONTAL_SCALE = float(
    os.environ.get("SPEED_EXT_SLOPE_HORIZONTAL_SCALE", "0.1")
)
POWER_ENVELOPE = os.environ.get(
    "PHYS_MOTOR_POWER_ENVELOPE", "0"
).strip().lower() in ("1", "true", "yes", "on")
SPECIFIC_POWER_W_PER_KG = float(
    os.environ.get("PHYS_MOTOR_SPECIFIC_POWER_W_PER_KG", "450.0")
)
POWER_EXPONENT = float(
    os.environ.get("PHYS_MOTOR_POWER_EXPONENT", "1.0")
)
VALIDATION_FIDELITY = os.environ.get(
    "SPEED_EXT_VALIDATION_FIDELITY",
    "formal" if POWER_ENVELOPE else "search",
).strip().lower()
if VALIDATION_FIDELITY not in ("search", "formal"):
    raise ValueError(
        "SPEED_EXT_VALIDATION_FIDELITY must be search or formal"
    )
VALIDATE_STAGE1_INTERMEDIATE = os.environ.get(
    "SPEED_EXT_VALIDATE_STAGE1_INTERMEDIATE",
    "1" if POWER_ENVELOPE else "0",
).strip().lower() in ("1", "true", "yes", "on")
if FIXED_SLOPE_DEG < 0.0 or FIXED_SLOPE_DEG >= 45.0:
    raise ValueError("SPEED_EXT_FIXED_SLOPE_DEG must be in [0, 45)")
if SLOPE_HORIZONTAL_SCALE <= 0.0:
    raise ValueError("SPEED_EXT_SLOPE_HORIZONTAL_SCALE must be positive")
if FIXED_SLOPE_DEG > 0.0:
    os.environ["PHYS_EVAL_SPEED_SLOPE_DEG"] = str(FIXED_SLOPE_DEG)
    os.environ["PHYS_EVAL_SLOPE_LENGTH"] = str(EVAL_SLOPE_LENGTH)
    os.environ["PHYS_EVAL_SLOPE_HORIZONTAL_SCALE"] = str(
        SLOPE_HORIZONTAL_SCALE
    )
    os.environ["PHYS_TRAIN_SPEED_SLOPE_DEG"] = str(FIXED_SLOPE_DEG)
    os.environ["PHYS_TRAIN_TERRAIN_LENGTH"] = str(TRAIN_SLOPE_LENGTH)
    os.environ["PHYS_TRAIN_SLOPE_HORIZONTAL_SCALE"] = str(
        SLOPE_HORIZONTAL_SCALE
    )


def curriculum_from_environment(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return tuple(float(value) for value in default)
    values = tuple(
        float(value) for value in raw.split(",") if value.strip()
    )
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two levels")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} levels must be strictly increasing")
    return values


CURRICULA = {
    "gradual_stage1_3p0": curriculum_from_environment(
        "SPEED_EXT_GRADUAL_STAGE1_LEVELS",
        (0.5, 0.8, 1.1, 1.4, 1.8, 2.2, 2.6, 3.0),
    ),
    "direct_stage1_3p4": curriculum_from_environment(
        "SPEED_EXT_DIRECT_STAGE1_LEVELS",
        (0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9, 3.4),
    ),
    "gradual_stage2_3p4": curriculum_from_environment(
        "SPEED_EXT_GRADUAL_STAGE2_LEVELS",
        (0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9, 3.4),
    ),
    "wide_stage2_3p8": curriculum_from_environment(
        "SPEED_EXT_WIDE_STAGE2_LEVELS",
        (0.5, 0.9, 1.3, 1.7, 2.1, 2.6, 3.1, 3.8),
    ),
}
def validation_designs_from_environment():
    raw = os.environ.get("SPEED_EXT_VALIDATION_DESIGNS_JSON")
    if raw is None:
        return (
            base.NOMINAL.copy(),
            np.asarray(
                [
                    0.804805,
                    0.812361,
                    1.023780,
                    0.756790,
                    0.623159,
                    1.478820,
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [0.8, 0.8, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float64,
            ),
        )
    designs = np.asarray(json.loads(raw), dtype=np.float64)
    if designs.ndim != 2 or designs.shape[1] != len(base.PARAMETERS):
        raise ValueError(
            "SPEED_EXT_VALIDATION_DESIGNS_JSON must be an Nx6 array"
        )
    if len(designs) < 2:
        raise ValueError("at least two validation designs are required")
    if np.any(designs < base.LOWER) or np.any(designs > base.UPPER):
        raise ValueError("validation design lies outside six-dimensional bounds")
    return tuple(row.copy() for row in designs)


VALIDATION_DESIGNS = validation_designs_from_environment()


def formal_control_designs_from_environment():
    raw = os.environ.get("SPEED_EXT_FORMAL_CONTROLS_JSON", "").strip()
    if not raw:
        return (base.NOMINAL.copy(),)
    designs = np.asarray(json.loads(raw), dtype=np.float64)
    if designs.ndim != 2 or designs.shape[1] != len(base.PARAMETERS):
        raise ValueError(
            "SPEED_EXT_FORMAL_CONTROLS_JSON must be an Nx6 array"
        )
    if np.any(designs < base.LOWER) or np.any(designs > base.UPPER):
        raise ValueError("formal control design lies outside bounds")
    return tuple(row.copy() for row in designs)


FORMAL_CONTROL_DESIGNS = formal_control_designs_from_environment()


def comma_values(values):
    return ",".join(f"{float(value):g}" for value in values)


def command_grid(maximum, step):
    values = list(
        np.round(np.arange(0.5, maximum + 1.0e-9, step), 6)
    )
    if not values or not np.isclose(values[-1], maximum):
        values.append(float(maximum))
    return sorted(set(float(value) for value in values))


def make_training_job(name, base_run, base_checkpoint, levels):
    directory = OUTPUT / "training" / name
    result_path = directory / "result.json"
    environment = base.common_motor_environment()
    environment.update(
        {
            "PHYS_MAX_GPU_CONTACT_PAIRS": str(2**25),
            "PHYS_TRAIN_TASK": TASK,
            "PHYS_TRAIN_BASE_RUN": str(base_run),
            "PHYS_TRAIN_BASE_CKPT": str(base_checkpoint),
            "PHYS_TRAIN_RANDOMIZE_DESIGN": "1",
            "PHYS_TRAIN_NUM_ENVS": "4096",
            "PHYS_TRAIN_MORPH_BUCKETS": "64",
            "PHYS_TRAIN_UPDATES": str(STAGE_UPDATES),
            "PHYS_TRAIN_SEED": str(base.SEED),
            "PHYS_TRAIN_TAG": (
                f"sixdgen_speed_ext_{name}_s{base.SEED}_"
                f"u{STAGE_UPDATES}"
            ),
            "PHYS_TRAIN_RESULT_JSON": str(result_path),
            "PHYS_TRAIN_CLEARANCE_LEVELS": base.TRAIN_LEVELS[
                "clearance"
            ],
            "PHYS_TRAIN_SLOPE_LEVELS": base.TRAIN_LEVELS["slope"],
            "PHYS_TRAIN_PAYLOAD_LEVELS": base.TRAIN_LEVELS["payload"],
            "PHYS_TRAIN_SPEED_LEVELS": comma_values(levels),
            "PHYS_TRAIN_SPEED_STEP_COMPONENT": str(STEP_COMPONENT),
            "PHYS_TRAIN_LEARNING_RATE": os.environ.get(
                "SPEED_EXT_LEARNING_RATE", "0.0001"
            ),
            "PHYS_TRAIN_SCHEDULE": os.environ.get(
                "SPEED_EXT_SCHEDULE", "fixed"
            ),
            "PHYS_TRAIN_ENTROPY_COEF": os.environ.get(
                "SPEED_EXT_ENTROPY_COEF", "0.001"
            ),
            "PHYS_TRAIN_EST_LEARNING_RATE": os.environ.get(
                "SPEED_EXT_EST_LEARNING_RATE", "0.0"
            ),
            "PHYS_TRAIN_SPEED_SLOPE_DEG": str(FIXED_SLOPE_DEG),
            "PHYS_TRAIN_TERRAIN_LENGTH": str(TRAIN_SLOPE_LENGTH),
            "PHYS_TRAIN_SLOPE_HORIZONTAL_SCALE": str(
                SLOPE_HORIZONTAL_SCALE
            ),
            "PHYS_TRAIN_TERRAIN_WIDTH": "6.0",
            "PHYS_TRAIN_TERRAIN_BORDER": "10.0",
        }
    )
    stage_name = "STAGE1" if "stage1" in name else "STAGE2"
    for parameter in base.PARAMETERS:
        value = os.environ.get(
            f"SPEED_EXT_{stage_name}_{parameter.upper()}_RANGE", ""
        ).strip()
        if value:
            environment[
                f"PHYS_TRAIN_{parameter.upper()}_RANGE"
            ] = value
    return base.Job(
        f"train-{name}",
        [
            base.PYTHON,
            "legged_gym/scripts/train_physical_capability.py",
            "--headless",
        ],
        environment,
        directory,
        result_path,
    )


def run_training_stages():
    stage1_jobs = [
        make_training_job(
            "gradual_stage1_3p0",
            BASE_RUN,
            BASE_CHECKPOINT,
            CURRICULA["gradual_stage1_3p0"],
        ),
        make_training_job(
            "direct_stage1_3p4",
            BASE_RUN,
            BASE_CHECKPOINT,
            CURRICULA["direct_stage1_3p4"],
        ),
    ]
    base.run_jobs("speed_extended_training_stage1", stage1_jobs)
    stage1 = {
        job.directory.name: base.read_json(job.expected_result)
        for job in stage1_jobs
    }

    stage2_jobs = [
        make_training_job(
            "gradual_stage2_3p4",
            stage1["gradual_stage1_3p0"]["run_name"],
            stage1["gradual_stage1_3p0"]["final_iteration"],
            CURRICULA["gradual_stage2_3p4"],
        ),
        make_training_job(
            "wide_stage2_3p8",
            stage1["direct_stage1_3p4"]["run_name"],
            stage1["direct_stage1_3p4"]["final_iteration"],
            CURRICULA["wide_stage2_3p8"],
        ),
    ]
    base.run_jobs("speed_extended_training_stage2", stage2_jobs)
    stage2 = {
        job.directory.name: base.read_json(job.expected_result)
        for job in stage2_jobs
    }
    training = {**stage1, **stage2}
    for label, result in training.items():
        reported_slope = float(result.get("speed_slope_deg", 0.0))
        if not np.isclose(reported_slope, FIXED_SLOPE_DEG):
            raise RuntimeError(
                f"{label} training slope mismatch: expected "
                f"{FIXED_SLOPE_DEG}, got {reported_slope}"
            )
        reported_step = float(
            result.get("speed_step_component_mps", 0.40)
        )
        if not np.isclose(reported_step, STEP_COMPONENT):
            raise RuntimeError(
                f"{label} speed-step mismatch: expected "
                f"{STEP_COMPONENT}, got {reported_step}"
            )
        design_spec = result.get("design_spec", {})
        if bool(design_spec.get("motor_power_envelope", False)) != (
            POWER_ENVELOPE
        ):
            raise RuntimeError(
                f"{label} motor-power-envelope provenance mismatch"
            )
        if POWER_ENVELOPE and (
            not np.isclose(
                float(
                    design_spec.get(
                        "motor_specific_power_w_per_kg",
                        float("nan"),
                    )
                ),
                SPECIFIC_POWER_W_PER_KG,
            )
            or not np.isclose(
                float(
                    design_spec.get(
                        "motor_power_exponent",
                        float("nan"),
                    )
                ),
                POWER_EXPONENT,
            )
        ):
            raise RuntimeError(
                f"{label} motor power-model settings mismatch"
            )
    base.atomic_json(OUTPUT / "TRAINING_RESULTS.json", training)
    return training


def validation_policies(training):
    policies = [
        {
            "label": "base_policy",
            "run_name": BASE_RUN,
            "checkpoint": BASE_CHECKPOINT,
            "training_command_max_mps": BASE_COMMAND_MAX,
            "training_slope_deg": BASE_TRAINING_SLOPE_DEG,
        }
    ]
    for label in ("gradual_stage1_3p0", "direct_stage1_3p4"):
        result = training[label]
        if VALIDATE_STAGE1_INTERMEDIATE:
            first = int(result["base_checkpoint"]) + 100
            final = int(result["final_iteration"])
            for checkpoint in range(first, final, 100):
                policies.append(
                    {
                        "label": f"{label}_ckpt{checkpoint}",
                        "run_name": result["run_name"],
                        "checkpoint": checkpoint,
                        "training_command_max_mps": max(
                            result["speed_levels_mps"]
                        ),
                        "training_slope_deg": float(
                            result.get("speed_slope_deg", 0.0)
                        ),
                    }
                )
        policies.append(
            {
                "label": label + "_final",
                "run_name": result["run_name"],
                "checkpoint": result["final_iteration"],
                "training_command_max_mps": max(
                    result["speed_levels_mps"]
                ),
                "training_slope_deg": float(
                    result.get("speed_slope_deg", 0.0)
                ),
            }
        )
    for label in ("gradual_stage2_3p4", "wide_stage2_3p8"):
        result = training[label]
        first = int(result["base_checkpoint"]) + 100
        final = int(result["final_iteration"])
        for checkpoint in range(first, final + 1, 100):
            policies.append(
                {
                    "label": f"{label}_ckpt{checkpoint}",
                    "run_name": result["run_name"],
                    "checkpoint": checkpoint,
                    "training_command_max_mps": max(
                        result["speed_levels_mps"]
                    ),
                    "training_slope_deg": float(
                        result.get("speed_slope_deg", 0.0)
                    ),
                }
            )
    return policies


def select_checkpoint(training):
    validation_maximum = max(
        [BASE_COMMAND_MAX]
        + [max(levels) for levels in CURRICULA.values()]
    )
    validation_levels = command_grid(validation_maximum, 0.1)
    base.SEARCH_LEVELS[TASK] = comma_values(validation_levels)
    base.FORMAL_LEVELS[TASK] = comma_values(validation_levels)
    policies = validation_policies(training)
    requests = []
    request_policy = {}
    for policy in policies:
        for design in VALIDATION_DESIGNS:
            phase = (
                "extended_checkpoint_validation_"
                + policy["label"]
            )
            request = base.EvalRequest(
                TASK,
                policy["run_name"],
                policy["checkpoint"],
                design,
                base.SEED,
                phase,
                VALIDATION_FIDELITY,
            )
            requests.append(request)
            request_policy[id(request)] = policy
    expected_paths = [
        str(request.job.expected_result) for request in requests
    ]
    if len(expected_paths) != len(set(expected_paths)):
        raise RuntimeError(
            "checkpoint validation result paths are not unique"
        )
    base.run_jobs(
        "speed_extended_checkpoint_validation",
        [request.job for request in requests],
    )
    records = []
    for request in requests:
        policy = request_policy[id(request)]
        result = base.read_json(request.job.expected_result)
        if (
            result.get("run") != policy["run_name"]
            or int(result.get("checkpoint", -1))
            != int(policy["checkpoint"])
        ):
            raise RuntimeError(
                "checkpoint validation provenance mismatch for "
                f"{policy['label']}: {request.job.expected_result}"
            )
        reported_limit = float(
            result.get("summary", {}).get(
                "training_command_max_mps", float("nan")
            )
        )
        reported_slope = float(
            result.get("summary", {}).get(
                "fixed_slope_deg", float("nan")
            )
        )
        if not np.isclose(reported_slope, FIXED_SLOPE_DEG):
            raise RuntimeError(
                "checkpoint validation evaluation-slope mismatch for "
                f"{policy['label']}: expected {FIXED_SLOPE_DEG}, "
                f"got {reported_slope}"
            )
        if not np.isclose(
            reported_limit,
            policy["training_command_max_mps"],
        ):
            raise RuntimeError(
                "checkpoint validation training-domain mismatch for "
                f"{policy['label']}: expected "
                f"{policy['training_command_max_mps']}, got "
                f"{reported_limit}"
            )
        reported_step = float(
            result.get("command_decomposition", {}).get(
                "step_component_mps", float("nan")
            )
        )
        if not np.isclose(reported_step, STEP_COMPONENT):
            raise RuntimeError(
                "checkpoint validation speed-step mismatch for "
                f"{policy['label']}: expected {STEP_COMPONENT}, "
                f"got {reported_step}"
            )
        power_model = result.get("motor_power_model", {})
        if bool(power_model.get("enabled", False)) != POWER_ENVELOPE:
            raise RuntimeError(
                "checkpoint validation power-envelope mismatch for "
                f"{policy['label']}"
            )
        if POWER_ENVELOPE and not np.isclose(
            float(
                power_model.get(
                    "specific_power_w_per_kg", float("nan")
                )
            ),
            SPECIFIC_POWER_W_PER_KG,
        ):
            raise RuntimeError(
                "checkpoint validation specific-power mismatch for "
                f"{policy['label']}"
            )
        record = base.record_from_request(request)
        record["policy_label"] = policy["label"]
        records.append(record)
    base.save_records(OUTPUT / "CHECKPOINT_VALIDATION_RECORDS.json", records)

    rows = []
    for policy in policies:
        values = [
            record
            for record in records
            if record["policy_label"] == policy["label"]
        ]
        objective = np.asarray(
            [record["objective"] for record in values],
            dtype=np.float64,
        )
        right_censored = [
            bool(
                record["summary"].get(
                    "right_censored_by_training_command_domain",
                    False,
                )
            )
            for record in values
        ]
        rows.append(
            {
                **policy,
                "design_count": len(values),
                "objective_by_design": objective.tolist(),
                "mean_objective": float(np.mean(objective)),
                "minimum_objective": float(np.min(objective)),
                "std_objective": float(np.std(objective, ddof=1)),
                "right_censored_count": int(sum(right_censored)),
                "summaries": [
                    record["summary"] for record in values
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            -row["mean_objective"],
            -row["minimum_objective"],
            row["std_objective"],
            row["checkpoint"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    selected = rows[0]
    base.atomic_json(
        OUTPUT / "CHECKPOINT_SELECTION.json",
        {
            "selected": selected,
            "policies": rows,
            "validation_designs": [
                base.design_dict(design)
                for design in VALIDATION_DESIGNS
            ],
            "validation_commands_mps": validation_levels,
        },
    )
    return {
        "run_name": selected["run_name"],
        "checkpoint": selected["checkpoint"],
        "training_command_max_mps": selected[
            "training_command_max_mps"
        ],
        "training_slope_deg": selected["training_slope_deg"],
        "evaluation_slope_deg": FIXED_SLOPE_DEG,
        "validation": rows,
    }


def evaluate(phase, designs, selected, fidelity="search", seeds=None):
    if seeds is None:
        seeds = (base.SEED,)
    records = base.evaluate_design_stage(
        phase,
        {TASK: base.unique_designs(designs)},
        {TASK: selected},
        fidelity=fidelity,
        seeds=seeds,
    )
    for record in records:
        result = base.read_json(Path(record["result_path"]))
        power_model = result.get("motor_power_model", {})
        if bool(power_model.get("enabled", False)) != POWER_ENVELOPE:
            raise RuntimeError(
                f"evaluation power-envelope mismatch: {record['result_path']}"
            )
        if POWER_ENVELOPE and not np.isclose(
            float(
                power_model.get(
                    "specific_power_w_per_kg", float("nan")
                )
            ),
            SPECIFIC_POWER_W_PER_KG,
        ):
            raise RuntimeError(
                f"evaluation specific-power mismatch: {record['result_path']}"
            )
        reported_step = float(
            result.get("command_decomposition", {}).get(
                "step_component_mps", float("nan")
            )
        )
        if not np.isclose(reported_step, STEP_COMPONENT):
            raise RuntimeError(
                f"evaluation speed-step mismatch: expected "
                f"{STEP_COMPONENT}, got {reported_step}: "
                f"{record['result_path']}"
            )
    return records


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
                "positive_uphill_progress_rate_by_seed": [
                    record["summary"][
                        "positive_uphill_progress_rate_at_command"
                    ]
                    for record in values
                ],
                "wheel_torque_ratio_mean_by_seed": [
                    record["summary"][
                        "wheel_torque_ratio_mean_at_command"
                    ]
                    for record in values
                ],
                "wheel_power_limited_fraction_by_seed": [
                    record["summary"][
                        "wheel_power_limited_fraction_at_command"
                    ]
                    for record in values
                ],
                "wheel_available_torque_mean_nm_by_seed": [
                    record["summary"][
                        "wheel_available_torque_mean_at_command_nm"
                    ]
                    for record in values
                ],
                "wheel_abs_torque_mean_nm_by_seed": [
                    record["summary"][
                        "wheel_abs_torque_mean_at_command_nm"
                    ]
                    for record in values
                ],
                "wheel_mechanical_power_mean_w_by_seed": [
                    record["summary"][
                        "wheel_mechanical_power_mean_at_command_w"
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
    return {TASK: {"winner": rows[0], "finalists": rows}}


def write_final(final, search_records, selected):
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
        (
            "# Fixed-slope six-dimensional uphill-speed result"
            if FIXED_SLOPE_DEG > 0.0
            else "# Extended-domain six-dimensional speed result"
        ),
        "",
        "Metric: maximum in-training-domain 10th percentile of actual mean "
        "forward speed, with every reset trial assigned zero speed.",
        "",
        f"Selected policy: `{selected['run_name']}/"
        f"model_{selected['checkpoint']}.pt`.",
        f"Training command maximum: "
        f"{selected['training_command_max_mps']:g} m/s.",
        f"Evaluation slope: {FIXED_SLOPE_DEG:g} deg; positive speed is "
        "the +world-x/+world-z ascending tangent.",
        "",
        "| rank | thigh | shank | abad | hip | knee | wheel | "
        "stable actual speed mean +/- std (m/s) | commands by seed | "
        "right-censored seeds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in final[TASK]["finalists"]:
        design = row["design"]
        commands = "/".join(
            f"{value:g}" for value in row["command_at_metric_by_seed"]
        )
        censored = sum(
            row["right_censored_by_training_domain_by_seed"]
        )
        lines.append(
            f"| {row['rank']} | {design['thigh']:.4f} | "
            f"{design['shank']:.4f} | {design['abad']:.4f} | "
            f"{design['hip']:.4f} | {design['knee']:.4f} | "
            f"{design['wheel']:.4f} | {row['objective_mean']:.6g} "
            f"+/- {row['objective_std']:.3g} | {commands} | "
            f"{censored}/{row['seed_count']} |"
        )
    lines.extend(
        [
            "",
            f"Fresh initial evaluations: {len(base.initial_designs())}.",
            f"New BO evaluations: {BO1_COUNT + BO2_COUNT}.",
            f"Total fresh search records: {len(search_records)}.",
            f"Formal seeds: {list(base.FORMAL_SEEDS)}.",
        ]
    )
    (OUTPUT / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def archive_selected_policy(selected):
    run_dir = (
        base.ROOT
        / "logs"
        / "wheelfoot_flat"
        / "WF_TRON1A"
        / selected["run_name"]
    )
    target = OUTPUT / "selected_policy"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        f"model_{selected['checkpoint']}.pt",
        "env_cfg.json",
        "train_cfg.json",
        "physical_training_manifest.json",
        "physical_training_result.json",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, target / name)
    base.atomic_json(target / "selection.json", selected)


def snapshot_sources():
    for relative in (
        "legged_gym/scripts/physical_capability_eval.py",
        "legged_gym/scripts/train_physical_capability.py",
        "legged_gym/scripts/run_sixd_single_objective_search.py",
        "legged_gym/scripts/run_sixd_speed_extended_search.py",
        "legged_gym/scripts/run_sixd_speed_limit_convergence.py",
        "legged_gym/envs/base/base_task.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py",
        "legged_gym/utils/motor_cost.py",
    ):
        source = base.ROOT / relative
        if not source.is_file():
            continue
        target = OUTPUT / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main():
    if SELECTION_ONLY and FROZEN_RUN:
        raise ValueError(
            "selection-only and frozen-policy modes are mutually exclusive"
        )
    if FROZEN_RUN and (
        FROZEN_CHECKPOINT <= 0 or FROZEN_COMMAND_MAX <= 0.0
    ):
        raise ValueError(
            "frozen-policy mode requires checkpoint and command maximum"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "STARTED").write_text(base.now_iso() + "\n", encoding="ascii")
    (OUTPUT / "DONE").unlink(missing_ok=True)
    (OUTPUT / "EXIT_CODE").unlink(missing_ok=True)
    snapshot_sources()
    base.atomic_json(
        OUTPUT / "RUN_MANIFEST.json",
        {
            "protocol": "sixd_speed_extended_v1_2026-07-26",
            "started_at": base.now_iso(),
            "source": str(SOURCE),
            "output": str(OUTPUT),
            "base_policy": {
                "run_name": BASE_RUN,
                "checkpoint": BASE_CHECKPOINT,
                "training_command_max_mps": BASE_COMMAND_MAX,
                "training_slope_deg": BASE_TRAINING_SLOPE_DEG,
            },
            "fixed_evaluation_slope_deg": FIXED_SLOPE_DEG,
            "positive_speed_direction": (
                "+world_x and +world_z along the ascending terrain tangent"
                if FIXED_SLOPE_DEG > 0.0
                else "+forward on flat terrain"
            ),
            "evaluation_slope_length_m": EVAL_SLOPE_LENGTH,
            "training_slope_length_m": TRAIN_SLOPE_LENGTH,
            "slope_horizontal_scale_m": SLOPE_HORIZONTAL_SCALE,
            "speed_step_component_mps": STEP_COMPONENT,
            "training_stage_updates": STAGE_UPDATES,
            "curricula": {
                name: list(levels)
                for name, levels in CURRICULA.items()
            },
            "parameters": list(base.PARAMETERS),
            "bounds": {
                name: [float(low), float(high)]
                for name, low, high in zip(
                    base.PARAMETERS, base.LOWER, base.UPPER
                )
            },
            "metric": {
                "name": (
                    "V_uphill_stable_actual_90"
                    if FIXED_SLOPE_DEG > 0.0
                    else "V_stable_actual_90"
                ),
                "reset_trial_score_mps": 0.0,
                "actual_speed_quantile": 0.10,
                "command_tracking_error_used": False,
            },
            "motor_power_model": {
                "enabled": POWER_ENVELOPE,
                "specific_power_w_per_kg": SPECIFIC_POWER_W_PER_KG,
                "power_exponent": POWER_EXPONENT,
            },
            "fresh_initial_design_count": len(base.initial_designs()),
            "initial_space_filling_sampler": base.INITIAL_SAMPLER,
            "initial_space_filling_count": base.SEARCH_INITIAL_LHS,
            "bo_batches": [BO1_COUNT, BO2_COUNT],
            "formal_seeds": list(base.FORMAL_SEEDS),
            "finalist_count": FINALIST_COUNT,
            "finalist_seeds": list(FINALIST_SEEDS),
            "formal_top_count": FORMAL_TOP_COUNT,
            "formal_control_designs": [
                base.design_dict(design)
                for design in FORMAL_CONTROL_DESIGNS
            ],
            "mode": (
                "frozen_search"
                if FROZEN_RUN
                else "selection_only"
                if SELECTION_ONLY
                else "train_select_search"
            ),
            "validation_designs": [
                base.design_dict(design)
                for design in VALIDATION_DESIGNS
            ],
            "checkpoint_validation_fidelity": VALIDATION_FIDELITY,
            "validate_stage1_intermediate": (
                VALIDATE_STAGE1_INTERMEDIATE
            ),
        },
    )

    training = {}
    if FROZEN_RUN:
        selected = {
            "run_name": FROZEN_RUN,
            "checkpoint": FROZEN_CHECKPOINT,
            "training_command_max_mps": FROZEN_COMMAND_MAX,
            "training_slope_deg": float(
                os.environ.get(
                    "SPEED_EXT_FROZEN_TRAINING_SLOPE_DEG",
                    str(FIXED_SLOPE_DEG),
                )
            ),
            "evaluation_slope_deg": FIXED_SLOPE_DEG,
            "validation": [],
        }
    else:
        training = run_training_stages()
        selected = select_checkpoint(training)
    base.atomic_json(OUTPUT / "SELECTED_POLICY.json", selected)
    archive_selected_policy(selected)

    if SELECTION_ONLY:
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
                "training_run_count": len(training),
                "checkpoint_validation_policy_count": len(
                    selected["validation"]
                ),
                "search_result_count": 0,
                "formal_result_count": 0,
                "mode": "selection_only",
            },
        )
        base.update_status(
            stage="selection_complete",
            completed_jobs=1,
            total_jobs=1,
            failed_jobs=[],
            eta_seconds=0.0,
        )
        (OUTPUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
        (OUTPUT / "DONE").write_text(
            base.now_iso() + "\n", encoding="ascii"
        )
        print(
            f"[speed-extended] SELECTION COMPLETE "
            f"{OUTPUT / 'CHECKPOINT_SELECTION.json'}"
        )
        return

    search_levels = command_grid(
        selected["training_command_max_mps"], 0.2
    )
    formal_levels = command_grid(
        selected["training_command_max_mps"], 0.1
    )
    base.SEARCH_LEVELS[TASK] = comma_values(search_levels)
    base.FORMAL_LEVELS[TASK] = comma_values(formal_levels)
    base.atomic_json(
        OUTPUT / "EVALUATION_COMMAND_GRIDS.json",
        {
            "search_mps": search_levels,
            "formal_mps": formal_levels,
        },
    )

    initial = evaluate(
        "extended_speed_initial_search",
        base.initial_designs(),
        selected,
    )
    base.save_records(OUTPUT / "INITIAL_SEARCH_RECORDS.json", initial)
    search_records = list(initial)
    gp_fits = []

    designs, fit = base.propose_expected_improvement(
        search_records, BO1_COUNT, base.SEED + 700
    )
    gp_fits.append({"phase": "extended_speed_bo1", "fit": fit})
    search_records.extend(
        evaluate("extended_speed_bo1_search", designs, selected)
    )
    base.save_records(OUTPUT / "SEARCH_AFTER_BO1.json", search_records)

    designs, fit = base.propose_expected_improvement(
        search_records, BO2_COUNT, base.SEED + 800
    )
    gp_fits.append({"phase": "extended_speed_bo2", "fit": fit})
    search_records.extend(
        evaluate("extended_speed_bo2_search", designs, selected)
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
    if FINALIST_COUNT > 0:
        finalist_designs = [
            np.asarray(record["design_vector"], dtype=np.float64)
            for record in ranked[:FINALIST_COUNT]
        ]
        finalist_designs = base.unique_designs(
            finalist_designs + [base.NOMINAL]
        )
        finalist_records = evaluate(
            "extended_speed_finalist_repeats",
            finalist_designs,
            selected,
            fidelity="formal",
            seeds=FINALIST_SEEDS,
        )
        base.save_records(
            OUTPUT / "FINALIST_RECORDS.json", finalist_records
        )
        finalist_aggregate = aggregate_formal(finalist_records)
        base.atomic_json(
            OUTPUT / "FINALIST_RESULTS.json",
            finalist_aggregate,
        )
        formal_designs = [
            np.asarray(row["design_vector"], dtype=np.float64)
            for row in finalist_aggregate[TASK]["finalists"][
                :FORMAL_TOP_COUNT
            ]
        ]
    else:
        formal_designs = [
            np.asarray(record["design_vector"], dtype=np.float64)
            for record in ranked[:FORMAL_TOP_COUNT]
        ]
    formal_designs = base.unique_designs(
        formal_designs + list(FORMAL_CONTROL_DESIGNS)
    )
    formal_records = evaluate(
        "extended_speed_formal_repeats",
        formal_designs,
        selected,
        fidelity="formal",
        seeds=base.FORMAL_SEEDS,
    )
    base.save_records(OUTPUT / "FORMAL_RECORDS.json", formal_records)
    final = aggregate_formal(formal_records)
    write_final(final, search_records, selected)

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
            "training_run_count": len(training),
            "checkpoint_validation_policy_count": len(
                selected["validation"]
            ),
            "search_result_count": len(search_records),
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
        f"[speed-extended] COMPLETE {OUTPUT / 'FINAL_RESULTS.json'}"
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
