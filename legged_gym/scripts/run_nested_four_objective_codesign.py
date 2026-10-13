#!/usr/bin/env python3
"""Run nested four-objective Bayesian hardware optimization.

The outer loop proposes six-dimensional hardware designs.  The inner loop
fine-tunes one policy per task and candidate, selects an early-stopping
checkpoint on a validation seed, and measures the selected checkpoint on an
independent test seed.  The four physical objectives are then normalized and
combined with equal weights.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
from scipy.stats import norm, qmc


REPO = Path(
    os.environ.get(
        "NESTED_BO_REPO",
        "/home/ps/mydrive/xu/pointfoot-legged-gym",
    )
)
PYTHON = Path(
    os.environ.get(
        "NESTED_BO_PYTHON",
        "/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python",
    )
)
ROOT = Path(
    os.environ.get(
        "NESTED_BO_ROOT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "nested_four_objective_sixd_2026-07-30",
    )
)
GENERALIST_ROOT = Path(
    os.environ.get(
        "NESTED_BO_GENERALIST_ROOT",
        str(ROOT / "generalists"),
    )
)
OUTPUT = Path(
    os.environ.get(
        "NESTED_BO_OUTPUT",
        str(ROOT / "nested_bo"),
    )
)
PRECOMPUTE_ROOT = ROOT / "precompute_ready_tasks"
OLD_BO_ROOT = Path(
    os.environ.get(
        "NESTED_BO_OLD_ROOT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "four_objective_sixd_v2_uphill_speed_2026-07-27",
    )
)

os.environ["SIXD_ROOT"] = str(REPO)
os.environ["SIXD_PYTHON"] = str(PYTHON)
os.environ["SIXD_OUT"] = str(OUTPUT)
os.environ.setdefault("SIXD_GPUS", "0,1")
os.environ.setdefault("SIXD_SEED", "20260805")

import run_sixd_single_objective_search as base  # noqa: E402


TASKS = ("payload", "speed", "clearance", "jump")
PARAMETERS = base.PARAMETERS
LOWER = base.LOWER
UPPER = base.UPPER
NOMINAL = base.NOMINAL
WEIGHTS = {task: 0.25 for task in TASKS}
INNER_OFFSETS = (0, 50, 100, 200, 400)
INNER_UPDATES = 400
GENERALIST_CHECKPOINT_OFFSETS = (2000, 4000, 6000, 7000)
SEARCH_TRAIN_SEED = int(
    os.environ.get("NESTED_BO_SEARCH_SEED", "20260805")
)
FORMAL_SEEDS = tuple(
    int(value)
    for value in os.environ.get(
        "NESTED_BO_FORMAL_SEEDS",
        "20260811,20260812,20260813",
    ).split(",")
    if value.strip()
)
INITIAL_COVERAGE_COUNT = int(
    os.environ.get("NESTED_BO_INITIAL_COVERAGE", "6")
)
BO_BATCH_SIZE = int(os.environ.get("NESTED_BO_BATCH_SIZE", "4"))
MIN_BO_BATCHES = int(os.environ.get("NESTED_BO_MIN_BATCHES", "2"))
MAX_BO_BATCHES = int(os.environ.get("NESTED_BO_MAX_BATCHES", "10"))
MIN_SCORE_IMPROVEMENT = float(
    os.environ.get("NESTED_BO_MIN_SCORE_IMPROVEMENT", "0.02")
)
MAX_EI_THRESHOLD = float(
    os.environ.get("NESTED_BO_MAX_EI_THRESHOLD", "0.01")
)
MAX_HV_RELATIVE_GAIN = float(
    os.environ.get("NESTED_BO_MAX_HV_RELATIVE_GAIN", "0.01")
)
MOTOR_MASSES = (1.35, 1.35, 0.89, 0.365)
SPEED_LEVELS = tuple(
    np.round(np.arange(0.5, 4.0001, 0.2), 6).tolist()
    + [4.0]
)
PAYLOAD_LEVELS = tuple(
    list(range(0, 65, 5)) + [70, 75, 80]
)
MIN_NORMALIZATION_SPAN = {
    "payload": 5.0,
    "speed": 0.25,
    "clearance": 0.02,
    "jump": 0.0005,
}
UNIT = {
    "payload": "kg",
    "speed": "m/s",
    "clearance": "m",
    "jump": "m^2",
}


SPECIALISTS = {
    "payload": np.asarray(
        [
            1.157807829976082,
            1.0520169399678707,
            0.9914530286565423,
            0.8961383629590273,
            1.5578197859227658,
            0.607684264332056,
        ],
        dtype=np.float64,
    ),
    "speed": np.asarray(
        [
            0.9648397047072649,
            0.9025079853832721,
            0.9153309095650911,
            0.6738600976765156,
            0.7679443968459964,
            1.5596084440127016,
        ],
        dtype=np.float64,
    ),
    "clearance": np.asarray(
        [
            1.1970247339457272,
            1.198773255571723,
            1.566337169520557,
            1.1520810587331654,
            1.1318710934370757,
            0.8972766384482384,
        ],
        dtype=np.float64,
    ),
    "jump": np.asarray(
        [
            1.0835728153586388,
            1.0553055875003339,
            0.8760620675981045,
            0.606938312202692,
            1.0510720726102591,
            0.769737096875906,
        ],
        dtype=np.float64,
    ),
}
OLD_JOINT_WINNER = np.asarray(
    [
        1.034134305,
        1.004982102,
        1.498098486,
        0.817097429,
        1.578590983,
        1.556014975,
    ],
    dtype=np.float64,
)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix
        + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
    temporary.replace(path)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def comma_values(values):
    return ",".join(f"{float(value):g}" for value in sorted(set(values)))


base.SEARCH_LEVELS["speed"] = comma_values(SPEED_LEVELS)
base.FORMAL_LEVELS["speed"] = comma_values(SPEED_LEVELS)
base.SEARCH_LEVELS["payload"] = comma_values(PAYLOAD_LEVELS)
base.FORMAL_LEVELS["payload"] = comma_values(PAYLOAD_LEVELS)


def wait_for_generalists(poll_seconds):
    started = time.time()
    while True:
        result_path = GENERALIST_ROOT / "RESULTS.json"
        if result_path.is_file():
            result = read_json(result_path)
            if set(result.get("selected", {})) == set(TASKS):
                return result
        exit_path = GENERALIST_ROOT / "DRIVER_EXIT_CODE"
        if exit_path.is_file() and exit_path.read_text().strip() not in (
            "",
            "0",
        ):
            raise RuntimeError("staged generalist driver failed")
        generalist_status = None
        status_path = GENERALIST_ROOT / "STATUS.json"
        if status_path.is_file():
            generalist_status = read_json(status_path)
        atomic_json(
            OUTPUT / "WAITING_FOR_GENERALISTS.json",
            {
                "updated_at": now(),
                "elapsed_s": time.time() - started,
                "generalist_root": str(GENERALIST_ROOT),
                "generalist_status": generalist_status,
            },
        )
        print(
            "[nested-bo] waiting for staged generalists; "
            f"elapsed_s={time.time() - started:.0f}",
            flush=True,
        )
        time.sleep(poll_seconds)


def wait_for_optional_precompute(poll_seconds):
    started_path = PRECOMPUTE_ROOT / "STARTED"
    if not started_path.is_file():
        return
    while not (PRECOMPUTE_ROOT / "DONE").is_file():
        exit_paths = (
            PRECOMPUTE_ROOT / "EXIT_CODE",
            PRECOMPUTE_ROOT / "DRIVER_EXIT_CODE",
        )
        failed = [
            {
                "path": str(path),
                "exit_code": path.read_text().strip(),
            }
            for path in exit_paths
            if path.is_file()
            and path.read_text().strip() not in ("", "0")
        ]
        if failed:
            atomic_json(
                OUTPUT / "PRECOMPUTE_FALLBACK.json",
                {
                    "recorded_at": now(),
                    "reason": (
                        "optional ready-task precompute failed; main "
                        "driver will rerun incomplete jobs"
                    ),
                    "failures": failed,
                },
            )
            return
        print(
            "[nested-bo] waiting for ready-task cache precompute",
            flush=True,
        )
        time.sleep(poll_seconds)


def selected_generalists(result):
    selected = {}
    for task in TASKS:
        record = result["selected"][task]
        if int(record["cumulative_randomized_updates"]) < 14000:
            raise RuntimeError(
                f"{task} has fewer than 14000 randomized updates"
            )
        policy = record["result"]
        checkpoint = Path(policy["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        design_spec = policy.get("design_spec", {})
        expected = {
            "thigh": [0.8, 1.2],
            "shank": [0.8, 1.2],
            "abad": [0.6, 1.6],
            "hip": [0.6, 1.6],
            "knee": [0.6, 1.6],
            "wheel": [0.6, 1.6],
        }
        for name, bounds in expected.items():
            if not np.allclose(
                design_spec.get(name, []),
                bounds,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    f"{task} selected policy has invalid {name} range"
                )
        if not bool(policy.get("hardware_context", False)):
            raise RuntimeError(f"{task} policy lacks hardware context")
        selected[task] = {
            "run_name": policy["run_name"],
            "checkpoint": int(policy["final_iteration"]),
            "checkpoint_path": str(checkpoint),
            "training_result": policy,
        }
    return selected


def generalist_validation_designs():
    return (
        NOMINAL.copy(),
        np.asarray(
            [0.85, 1.15, 0.75, 1.25, 1.35, 0.80],
            dtype=np.float64,
        ),
        np.asarray(
            [1.15, 0.85, 1.35, 0.80, 0.75, 1.30],
            dtype=np.float64,
        ),
    )


def select_generalist_checkpoints(policies):
    selection_path = OUTPUT / "GENERALIST_SELECTION.json"
    if selection_path.is_file():
        selection = read_json(selection_path)
        selected = selection.get("selected", {})
        if set(selected) == set(TASKS) and all(
            Path(selected[task]["checkpoint_path"]).is_file()
            for task in TASKS
        ):
            return selected

    requests = []
    for task in TASKS:
        policy = policies[task]
        stage2_base = int(
            policy["training_result"]["base_checkpoint"]
        )
        if int(policy["checkpoint"]) != (
            stage2_base + max(GENERALIST_CHECKPOINT_OFFSETS)
        ):
            raise RuntimeError(
                f"{task} Stage 2 final checkpoint is inconsistent"
            )
        for offset in GENERALIST_CHECKPOINT_OFFSETS:
            checkpoint = stage2_base + offset
            for design in generalist_validation_designs():
                requests.append(
                    make_eval_request(
                        task,
                        policy["run_name"],
                        checkpoint,
                        design,
                        20260806,
                        "generalist_checkpoint_selection",
                        "search",
                    )
                )
    base.run_jobs(
        "generalist_checkpoint_selection",
        [request.job for request in requests],
    )
    records = []
    for request in requests:
        validate_evaluation_protocol(request)
        records.append(base.record_from_request(request))
    base.save_records(
        OUTPUT / "GENERALIST_CHECKPOINT_RECORDS.json",
        records,
    )

    selected = {}
    validation = {}
    for task in TASKS:
        policy = policies[task]
        environment = (
            "wheelfoot_jump_forward"
            if task == "jump"
            else "wheelfoot_flat"
        )
        run_dir = (
            REPO
            / "logs"
            / environment
            / "WF_TRON1A"
            / policy["run_name"]
        )
        checkpoint_rows = []
        for checkpoint in sorted(
            {
                int(record["checkpoint"])
                for record in records
                if record["task"] == task
            }
        ):
            values = [
                float(record["objective"])
                for record in records
                if record["task"] == task
                and int(record["checkpoint"]) == checkpoint
            ]
            checkpoint_rows.append(
                {
                    "checkpoint": checkpoint,
                    "full_sixd_updates": (
                        checkpoint
                        - int(
                            policy["training_result"][
                                "base_checkpoint"
                            ]
                        )
                    ),
                    "mean_objective": float(np.mean(values)),
                    "minimum_objective": float(np.min(values)),
                    "values": values,
                }
            )
        if (
            not checkpoint_rows
            or max(row["mean_objective"] for row in checkpoint_rows)
            <= 0.0
        ):
            raise RuntimeError(
                f"{task} has no positive generalist checkpoint"
            )
        best = max(
            checkpoint_rows,
            key=lambda row: (
                row["mean_objective"],
                row["minimum_objective"],
                row["checkpoint"],
            ),
        )
        checkpoint_path = run_dir / f"model_{best['checkpoint']}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        selected[task] = {
            **policy,
            "checkpoint": int(best["checkpoint"]),
            "checkpoint_path": str(checkpoint_path),
            "selection": (
                "best mean physical objective over nominal and two "
                "fixed coverage designs"
            ),
            "full_sixd_updates": int(best["full_sixd_updates"]),
        }
        validation[task] = checkpoint_rows
    atomic_json(
        selection_path,
        {
            "selected_at": now(),
            "candidate_full_sixd_update_offsets": list(
                GENERALIST_CHECKPOINT_OFFSETS
            ),
            "validation_designs": [
                base.design_dict(design)
                for design in generalist_validation_designs()
            ],
            "validation_seed": 20260806,
            "selected": selected,
            "validation": validation,
        },
    )
    return selected


def snapshot_sources():
    sources = (
        "legged_gym/scripts/train_physical_capability.py",
        "legged_gym/scripts/train_physical_jump.py",
        "legged_gym/scripts/physical_capability_eval.py",
        "legged_gym/scripts/measure_jumpfwd.py",
        "legged_gym/scripts/run_four_task_staged_generalists.py",
        "legged_gym/scripts/run_nested_four_objective_codesign.py",
        "legged_gym/algorithm/on_policy_runner.py",
        "legged_gym/algorithm/ppo.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py",
        "legged_gym/utils/motor_cost.py",
    )
    for relative in sources:
        source = REPO / relative
        if source.is_file():
            target = OUTPUT / "code_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def archive_generalists(policies):
    for task, policy in policies.items():
        environment = (
            "wheelfoot_jump_forward"
            if task == "jump"
            else "wheelfoot_flat"
        )
        source_dir = (
            REPO
            / "logs"
            / environment
            / "WF_TRON1A"
            / policy["run_name"]
        )
        target = OUTPUT / "selected_generalists" / task
        target.mkdir(parents=True, exist_ok=True)
        model = f"model_{policy['checkpoint']}.pt"
        for name in (
            model,
            "env_cfg.json",
            "train_cfg.json",
            "physical_training_manifest.json",
            "physical_training_result.json",
            "physical_jump_training_manifest.json",
            "physical_jump_training_result.json",
        ):
            source = source_dir / name
            if source.is_file():
                shutil.copy2(source, target / name)
        atomic_json(target / "selection.json", policy)


def common_motor_environment():
    return {
        "PHYS_MOTOR_MASS_EXPONENT": "1.0",
        "PHYS_MOTOR_NOMINAL_MASSES_KG": ",".join(
            str(value) for value in MOTOR_MASSES
        ),
    }


def apply_design(environment, prefix, design):
    for name, value in zip(PARAMETERS, design):
        environment[f"{prefix}_{name.upper()}"] = (
            f"{float(value):.10f}"
        )


def make_adaptation_job(
    task,
    policy,
    design,
    train_seed,
    phase,
):
    design = np.asarray(design, dtype=np.float64)
    design_id = base.design_key(design)
    directory = (
        OUTPUT
        / "adaptations"
        / phase
        / design_id
        / f"seed_{train_seed}"
        / task
    )
    result_path = directory / "result.json"
    environment = common_motor_environment()
    if task == "jump":
        environment.update(
            {
                "PHYS_JUMP_INITIALIZATION": "pretrained",
                "PHYS_JUMP_BASE_RUN": policy["run_name"],
                "PHYS_JUMP_BASE_CKPT": str(policy["checkpoint"]),
                "PHYS_JUMP_UPDATES": str(INNER_UPDATES),
                "PHYS_JUMP_NUM_ENVS": "2048",
                "PHYS_JUMP_SEED": str(train_seed),
                "PHYS_JUMP_TAG": (
                    f"nested_{phase}_{design_id}_s{train_seed}"
                ),
                "PHYS_JUMP_RANDOMIZE_DESIGN": "0",
                "PHYS_JUMP_HARDWARE_CONTEXT": "1",
                "PHYS_JUMP_ANCHOR_FRACTION": "0.0",
                "PHYS_JUMP_REFERENCE_POLICY_COEF": "0.0",
                "PHYS_JUMP_HEIGHT_RANGE": "0.10,0.16",
                "PHYS_JUMP_DISTANCE_RANGE": "0.05,0.14",
                "PHYS_JUMP_FRICTION": "0.8",
                "PHYS_JUMP_LEARNING_RATE": "2e-5",
                "PHYS_JUMP_SCHEDULE": "fixed",
                "PHYS_JUMP_ENTROPY_COEF": "0.0",
                "PHYS_JUMP_EST_LEARNING_RATE": "0.0",
                "PHYS_JUMP_RESUME_NOISE_STD": "0.05",
                "PHYS_JUMP_SAVE_INTERVAL": "50",
                "PHYS_JUMP_SAVE_INITIAL": "1",
                "PHYS_JUMP_INIT_RANDOM_EP_LEN": "1",
                "PHYS_JUMP_RESULT_JSON": str(result_path),
            }
        )
        apply_design(environment, "PHYS_JUMP", design)
        script = "legged_gym/scripts/train_physical_jump.py"
    else:
        learning_rate = "2e-5" if task == "speed" else "5e-5"
        environment.update(
            {
                "PHYS_TRAIN_INITIALIZATION": "pretrained",
                "PHYS_TRAIN_TASK": task,
                "PHYS_TRAIN_BASE_RUN": policy["run_name"],
                "PHYS_TRAIN_BASE_CKPT": str(policy["checkpoint"]),
                "PHYS_TRAIN_UPDATES": str(INNER_UPDATES),
                "PHYS_TRAIN_NUM_ENVS": "2048",
                "PHYS_TRAIN_SEED": str(train_seed),
                "PHYS_TRAIN_TAG": (
                    f"nested_{phase}_{task}_{design_id}_s{train_seed}"
                ),
                "PHYS_TRAIN_RANDOMIZE_DESIGN": "0",
                "PHYS_TRAIN_HARDWARE_CONTEXT": "1",
                "PHYS_TRAIN_ANCHOR_FRACTION": "0.0",
                "PHYS_TRAIN_REFERENCE_POLICY_COEF": "0.0",
                "PHYS_TRAIN_CLEARANCE_LEVELS": (
                    "0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22"
                ),
                "PHYS_TRAIN_PAYLOAD_LEVELS": (
                    "0,5,10,15,20,30,40,50"
                ),
                "PHYS_TRAIN_SPEED_LEVELS": (
                    "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
                ),
                "PHYS_TRAIN_SPEED_SLOPE_DEG": (
                    "10.0" if task == "speed" else "0.0"
                ),
                "PHYS_TRAIN_SPEED_STEP_COMPONENT": (
                    "0.0" if task == "speed" else "0.40"
                ),
                "PHYS_TRAIN_LEARNING_RATE": learning_rate,
                "PHYS_TRAIN_SCHEDULE": "fixed",
                "PHYS_TRAIN_ENTROPY_COEF": (
                    "0.0005" if task == "speed" else "0.001"
                ),
                "PHYS_TRAIN_EST_LEARNING_RATE": "0.0",
                "PHYS_TRAIN_SAVE_INTERVAL": "50",
                "PHYS_TRAIN_SAVE_INITIAL": "1",
                "PHYS_TRAIN_RESULT_JSON": str(result_path),
                "PHYS_MOTOR_POWER_ENVELOPE": (
                    "1" if task == "speed" else "0"
                ),
                "PHYS_MOTOR_SPECIFIC_POWER_W_PER_KG": "450.0",
                "PHYS_MOTOR_POWER_EXPONENT": "1.0",
                "PHYS_MOTOR_POWER_MIN_SPEED_RAD_S": "0.001",
            }
        )
        apply_design(environment, "PHYS_TRAIN", design)
        script = "legged_gym/scripts/train_physical_capability.py"
    job = base.Job(
        f"adapt-{phase}-{task}-{design_id}-s{train_seed}",
        [
            str(PYTHON),
            script,
            "--headless",
            "--sim_device=cuda:0",
            "--rl_device=cuda:0",
        ],
        environment,
        directory,
        result_path,
    )
    return {
        "task": task,
        "design": design,
        "design_id": design_id,
        "train_seed": int(train_seed),
        "policy": policy,
        "job": job,
    }


def validate_adaptation(record):
    result = read_json(record["job"].expected_result)
    policy = record["policy"]
    if (
        result["base_run"] != policy["run_name"]
        or int(result["base_checkpoint"]) != int(policy["checkpoint"])
    ):
        raise RuntimeError("adaptation source policy mismatch")
    if int(result["final_iteration"]) != (
        int(policy["checkpoint"]) + INNER_UPDATES
    ):
        raise RuntimeError("adaptation update budget mismatch")
    if not bool(result.get("hardware_context", False)):
        raise RuntimeError("adapted policy lost hardware context")
    result_design = np.asarray(
        [
            result["design"][name]
            for name in PARAMETERS
        ],
        dtype=np.float64,
    )
    if not np.allclose(
        result_design,
        record["design"],
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise RuntimeError("adapted policy hardware design mismatch")
    record["result"] = result


def apply_evaluation_protocol(request):
    environment = request.job.environment
    if request.task == "speed":
        environment.update(
            {
                "PHYS_EVAL_SPEED_SUCCESS_MODE": "stable_actual",
                "PHYS_EVAL_SPEED_SLOPE_DEG": "10.0",
                "PHYS_EVAL_STEP_COMPONENT": "0.0",
                "PHYS_EVAL_SLOPE_LENGTH": "110.0",
                "PHYS_EVAL_SLOPE_HORIZONTAL_SCALE": "0.1",
                "PHYS_MOTOR_POWER_ENVELOPE": "1",
                "PHYS_MOTOR_SPECIFIC_POWER_W_PER_KG": "450.0",
                "PHYS_MOTOR_POWER_EXPONENT": "1.0",
                "PHYS_MOTOR_POWER_MIN_SPEED_RAD_S": "0.001",
                "PHYS_EVAL_LEVELS": comma_values(SPEED_LEVELS),
            }
        )
    elif request.task != "jump":
        environment.update(
            {
                "PHYS_EVAL_SPEED_SLOPE_DEG": "0.0",
                "PHYS_EVAL_STEP_COMPONENT": "0.40",
                "PHYS_MOTOR_POWER_ENVELOPE": "0",
            }
        )
        if request.task == "payload":
            environment["PHYS_EVAL_LEVELS"] = comma_values(
                PAYLOAD_LEVELS
            )


def validate_evaluation_protocol(request):
    if request.task != "speed":
        return
    result = read_json(request.job.expected_result)
    decomposition = result.get("command_decomposition", {})
    power = result.get("motor_power_model", {})
    checks = (
        np.isclose(float(result.get("speed_slope_deg", -1.0)), 10.0),
        np.isclose(
            float(decomposition.get("step_component_mps", -1.0)),
            0.0,
        ),
        bool(power.get("enabled", False)),
        result.get("summary", {}).get("metric_name")
        == "V_uphill_stable_actual_90",
    )
    if not all(checks):
        raise RuntimeError(
            f"speed evaluation violates protocol: "
            f"{request.job.expected_result}"
        )


def make_eval_request(
    task,
    run_name,
    checkpoint,
    design,
    seed,
    phase,
    fidelity,
):
    request = base.EvalRequest(
        task,
        run_name,
        checkpoint,
        design,
        seed,
        phase,
        fidelity,
    )
    apply_evaluation_protocol(request)
    return request


def evaluate_nested(
    phase,
    designs,
    policies,
    train_seeds,
    test_fidelity,
):
    designs = base.unique_designs(designs)
    adaptations = []
    for design in designs:
        for train_seed in train_seeds:
            for task in TASKS:
                adaptations.append(
                    make_adaptation_job(
                        task,
                        policies[task],
                        design,
                        train_seed,
                        phase,
                    )
                )
    base.run_jobs(
        f"{phase}_adaptation",
        [record["job"] for record in adaptations],
    )
    for record in adaptations:
        validate_adaptation(record)

    validation_requests = []
    validation_index = {}
    for adaptation in adaptations:
        result = adaptation["result"]
        validation_seed = adaptation["train_seed"] + 1000
        for offset in INNER_OFFSETS:
            checkpoint = int(adaptation["policy"]["checkpoint"]) + offset
            request = make_eval_request(
                adaptation["task"],
                result["run_name"],
                checkpoint,
                adaptation["design"],
                validation_seed,
                f"{phase}_checkpoint_validation",
                "search",
            )
            validation_requests.append(request)
            validation_index[id(request)] = (
                adaptation,
                offset,
            )
    base.run_jobs(
        f"{phase}_checkpoint_validation",
        [request.job for request in validation_requests],
    )
    validation_records = {}
    trajectory_rows = []
    for request in validation_requests:
        validate_evaluation_protocol(request)
        record = base.record_from_request(request)
        adaptation, offset = validation_index[id(request)]
        key = (
            adaptation["task"],
            adaptation["design_id"],
            adaptation["train_seed"],
        )
        validation_records.setdefault(key, []).append(
            (offset, record)
        )
        trajectory_rows.append(
            {
                **record,
                "train_seed": adaptation["train_seed"],
                "checkpoint_offset": offset,
            }
        )
    base.save_records(
        OUTPUT / f"{phase}_CHECKPOINT_TRAJECTORIES.json",
        trajectory_rows,
    )

    selections = {}
    selected_requests = []
    selected_index = {}
    for adaptation in adaptations:
        key = (
            adaptation["task"],
            adaptation["design_id"],
            adaptation["train_seed"],
        )
        options = validation_records[key]
        offset, record = max(
            options,
            key=lambda pair: (
                float(pair[1]["objective"]),
                -int(pair[0]),
            ),
        )
        checkpoint = (
            int(adaptation["policy"]["checkpoint"]) + int(offset)
        )
        selections[key] = {
            "checkpoint_offset": int(offset),
            "checkpoint": checkpoint,
            "validation_objective": float(record["objective"]),
            "validation_seed": int(record["seed"]),
            "validation_result_path": record["result_path"],
            "run_name": adaptation["result"]["run_name"],
        }
        test_seed = adaptation["train_seed"] + 2000
        request = make_eval_request(
            adaptation["task"],
            adaptation["result"]["run_name"],
            checkpoint,
            adaptation["design"],
            test_seed,
            f"{phase}_selected_test",
            test_fidelity,
        )
        selected_requests.append(request)
        selected_index[id(request)] = adaptation
    base.run_jobs(
        f"{phase}_selected_test",
        [request.job for request in selected_requests],
    )

    test_records = {}
    for request in selected_requests:
        validate_evaluation_protocol(request)
        record = base.record_from_request(request)
        adaptation = selected_index[id(request)]
        key = (
            adaptation["task"],
            adaptation["design_id"],
            adaptation["train_seed"],
        )
        test_records[key] = record

    rows = []
    for design in designs:
        design_id = base.design_key(design)
        for train_seed in train_seeds:
            raw = {}
            selected = {}
            result_paths = {}
            for task in TASKS:
                key = (task, design_id, int(train_seed))
                raw[task] = float(test_records[key]["objective"])
                selected[task] = selections[key]
                result_paths[task] = test_records[key]["result_path"]
            rows.append(
                {
                    "phase": phase,
                    "design_id": design_id,
                    "design": base.design_dict(design),
                    "design_vector": [
                        float(value) for value in design
                    ],
                    "train_seed": int(train_seed),
                    "validation_seed": int(train_seed) + 1000,
                    "test_seed": int(train_seed) + 2000,
                    "raw_by_task": raw,
                    "selected_checkpoint_by_task": selected,
                    "test_result_paths": result_paths,
                    "motor_module_mass_kg": base.motor_mass_kg(design),
                }
            )
    atomic_json(OUTPUT / f"{phase}_NESTED_RECORDS.json", rows)
    return rows


def old_search_designs():
    path = OLD_BO_ROOT / "ALL_SEARCH_AGGREGATED_RECORDS.json"
    if not path.is_file():
        return []
    rows = read_json(path)
    return [
        np.asarray(row["design_vector"], dtype=np.float64)
        for row in rows
    ]


def maximin_coverage(candidates, anchors, count):
    selected = [
        base.normalized(design)
        for design in base.unique_designs(anchors)
    ]
    pool = [
        base.normalized(design)
        for design in base.unique_designs(candidates)
        if base.design_key(design)
        not in {base.design_key(anchor) for anchor in anchors}
    ]
    result = []
    while pool and len(result) < count:
        distances = [
            min(np.linalg.norm(point - chosen) for chosen in selected)
            for point in pool
        ]
        index = int(np.argmax(distances))
        point = pool.pop(index)
        selected.append(point)
        result.append(base.denormalized(point))
    return result


def initial_designs():
    anchors = [
        NOMINAL.copy(),
        *[SPECIALISTS[task].copy() for task in TASKS],
        OLD_JOINT_WINNER.copy(),
    ]
    old = old_search_designs()
    if not old:
        sampler = qmc.Sobol(d=6, scramble=True, seed=20260805)
        old = [
            base.denormalized(point)
            for point in sampler.random_base2(m=5)
        ]
    coverage = maximin_coverage(
        old,
        anchors,
        INITIAL_COVERAGE_COUNT,
    )
    return base.unique_designs(anchors + coverage)


def calibration_anchors(initial_rows):
    by_id = {row["design_id"]: row for row in initial_rows}
    nominal_id = base.design_key(NOMINAL)
    nominal = by_id[nominal_id]
    anchors = {}
    for task in TASKS:
        specialist_id = base.design_key(SPECIALISTS[task])
        specialist_value = float(
            by_id[specialist_id]["raw_by_task"][task]
        )
        nominal_value = float(nominal["raw_by_task"][task])
        observed = max(
            float(row["raw_by_task"][task])
            for row in initial_rows
        )
        observed_id = max(
            initial_rows,
            key=lambda row: float(row["raw_by_task"][task]),
        )["design_id"]
        span = max(
            specialist_value - nominal_value,
            observed - nominal_value,
            MIN_NORMALIZATION_SPAN[task],
        )
        anchors[task] = {
            "nominal": nominal_value,
            "official_specialist": specialist_value,
            "observed_initial_best": observed,
            "span": float(span),
            "nominal_design_id": nominal_id,
            "official_specialist_design_id": specialist_id,
            "observed_initial_best_design_id": observed_id,
            "span_rule": (
                "max(official specialist - nominal, initial best - "
                "nominal, preregistered minimum physical span)"
            ),
            "unit": UNIT[task],
        }
    return anchors


def normalized_value(task, value, anchors):
    anchor = anchors[task]
    return (
        float(value) - float(anchor["nominal"])
    ) / float(anchor["span"])


def score_rows(rows, anchors):
    scored = []
    for row in rows:
        normalized = {
            task: normalized_value(
                task, row["raw_by_task"][task], anchors
            )
            for task in TASKS
        }
        score = float(
            sum(
                WEIGHTS[task] * normalized[task]
                for task in TASKS
            )
        )
        scored.append(
            {
                **row,
                "normalized_by_task": normalized,
                "equal_weight_score": score,
                "objective": score,
            }
        )
    scored.sort(
        key=lambda row: (
            -row["equal_weight_score"],
            row["motor_module_mass_kg"],
        )
    )
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    return scored


def pareto_front(rows):
    front = []
    for index, row in enumerate(rows):
        value = np.asarray(
            [row["raw_by_task"][task] for task in TASKS],
            dtype=np.float64,
        )
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            other_value = np.asarray(
                [other["raw_by_task"][task] for task in TASKS],
                dtype=np.float64,
            )
            if np.all(other_value >= value) and np.any(
                other_value > value
            ):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def hypervolume(rows, seed):
    front = pareto_front(rows)
    points = np.asarray(
        [
            [
                np.clip(row["normalized_by_task"][task], -1.0, 2.0)
                for task in TASKS
            ]
            for row in front
        ],
        dtype=np.float64,
    )
    if not len(points):
        return 0.0
    reference = np.full(4, -1.0, dtype=np.float64)
    upper = np.full(4, 2.0, dtype=np.float64)
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    samples = reference + sampler.random_base2(m=15) * (
        upper - reference
    )
    dominated = np.zeros(len(samples), dtype=bool)
    for point in points:
        dominated |= np.all(samples <= point, axis=1)
    return float(
        np.mean(dominated) * np.prod(upper - reference)
    )


def propose_expected_improvement(rows, count, seed):
    x = np.asarray(
        [base.normalized(row["design_vector"]) for row in rows],
        dtype=np.float64,
    )
    y = np.asarray(
        [row["equal_weight_score"] for row in rows],
        dtype=np.float64,
    )
    sampler = qmc.Sobol(d=6, scramble=True, seed=seed)
    pool = sampler.random_base2(m=16)
    if float(np.std(y)) < 1.0e-12:
        selected = pool[:count]
        return (
            [base.denormalized(point) for point in selected],
            {
                "fallback": "constant_objective",
                "max_expected_improvement": None,
            },
        )
    model = base.GaussianProcess(x, y)
    means = np.empty(len(pool), dtype=np.float64)
    deviations = np.empty(len(pool), dtype=np.float64)
    for start in range(0, len(pool), 8192):
        stop = min(start + 8192, len(pool))
        means[start:stop], deviations[start:stop] = model.predict(
            pool[start:stop]
        )
    xi = 0.01 * max(float(np.std(y)), 1.0e-9)
    improvement = means - float(np.max(y)) - xi
    z_value = improvement / np.maximum(deviations, 1.0e-12)
    expected_improvement = (
        improvement * norm.cdf(z_value)
        + deviations * norm.pdf(z_value)
    )
    order = np.argsort(expected_improvement)[::-1]
    selected = []
    selected_ei = []
    for index in order:
        point = pool[index]
        if np.min(np.linalg.norm(x - point, axis=1)) < 0.035:
            continue
        if selected and min(
            np.linalg.norm(point - previous)
            for previous in selected
        ) < 0.08:
            continue
        selected.append(point)
        selected_ei.append(float(expected_improvement[index]))
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in order:
            point = pool[index]
            if any(
                np.linalg.norm(point - previous) < 0.03
                for previous in selected
            ):
                continue
            selected.append(point)
            selected_ei.append(float(expected_improvement[index]))
            if len(selected) == count:
                break
    diagnostics = {
        **model.fit,
        "xi": xi,
        "max_expected_improvement": float(
            np.max(expected_improvement)
        ),
        "selected_expected_improvement": selected_ei,
        "current_best": float(np.max(y)),
    }
    return (
        [base.denormalized(point) for point in selected],
        diagnostics,
    )


def aggregate_formal(rows, anchors):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["design_id"], []).append(row)
    result = []
    for design_id, values in grouped.items():
        values = sorted(values, key=lambda row: row["train_seed"])
        raw_mean = {
            task: float(
                np.mean(
                    [row["raw_by_task"][task] for row in values]
                )
            )
            for task in TASKS
        }
        raw_std = {
            task: float(
                np.std(
                    [row["raw_by_task"][task] for row in values],
                    ddof=1,
                )
            )
            for task in TASKS
        }
        scores = []
        for row in values:
            scores.append(
                float(
                    sum(
                        WEIGHTS[task]
                        * normalized_value(
                            task,
                            row["raw_by_task"][task],
                            anchors,
                        )
                        for task in TASKS
                    )
                )
            )
        result.append(
            {
                "design_id": design_id,
                "design": values[0]["design"],
                "design_vector": values[0]["design_vector"],
                "seeds": [row["train_seed"] for row in values],
                "raw_mean_by_task": raw_mean,
                "raw_std_by_task": raw_std,
                "normalized_mean_by_task": {
                    task: normalized_value(
                        task, raw_mean[task], anchors
                    )
                    for task in TASKS
                },
                "score_by_seed": scores,
                "equal_weight_score_mean": float(np.mean(scores)),
                "equal_weight_score_std": float(
                    np.std(scores, ddof=1)
                ),
                "motor_module_mass_kg": values[0][
                    "motor_module_mass_kg"
                ],
                "nested_records": values,
            }
        )
    result.sort(
        key=lambda row: (
            -row["equal_weight_score_mean"],
            row["equal_weight_score_std"],
            row["motor_module_mass_kg"],
        )
    )
    front_rows = [
        {
            "raw_by_task": row["raw_mean_by_task"],
            "design_id": row["design_id"],
        }
        for row in result
    ]
    front_ids = {
        row["design_id"] for row in pareto_front(front_rows)
    }
    for rank, row in enumerate(result, start=1):
        row["rank"] = rank
        row["pareto_nondominated"] = (
            row["design_id"] in front_ids
        )
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "rank",
            *PARAMETERS,
            "equal_weight_score_mean",
            "equal_weight_score_std",
            *TASKS,
            "pareto_nondominated",
            "motor_module_mass_kg",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row["rank"],
                    **row["design"],
                    "equal_weight_score_mean": row[
                        "equal_weight_score_mean"
                    ],
                    "equal_weight_score_std": row[
                        "equal_weight_score_std"
                    ],
                    **row["raw_mean_by_task"],
                    "pareto_nondominated": row[
                        "pareto_nondominated"
                    ],
                    "motor_module_mass_kg": row[
                        "motor_module_mass_kg"
                    ],
                }
            )


def write_report(final_rows, anchors, diagnostics):
    lines = [
        "# Nested four-objective six-dimensional co-design",
        "",
        "Each outer-BO candidate receives four independent 400-update "
        "fixed-hardware adaptations. Checkpoints u0/u50/u100/u200/u400 "
        "are selected on a validation seed and measured on an independent "
        "test seed.",
        "",
        "| rank | design | score mean +/- std | payload kg | speed m/s | "
        "clearance m | jump m^2 | Pareto |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in final_rows:
        raw = row["raw_mean_by_task"]
        lines.append(
            f"| {row['rank']} | "
            f"`{[round(value, 6) for value in row['design_vector']]}` | "
            f"{row['equal_weight_score_mean']:.5f} +/- "
            f"{row['equal_weight_score_std']:.5f} | "
            f"{raw['payload']:.5g} | {raw['speed']:.5g} | "
            f"{raw['clearance']:.5g} | {raw['jump']:.6g} | "
            f"{'yes' if row['pareto_nondominated'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Normalization: `{anchors}`.",
            "",
            f"BO stopping diagnostics: `{diagnostics}`.",
        ]
    )
    (OUTPUT / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    snapshot_sources()
    initial = initial_designs()
    if args.plan_only:
        atomic_json(
            OUTPUT / "PLAN_ONLY.json",
            {
                "created_at": now(),
                "initial_designs": [
                    base.design_dict(design) for design in initial
                ],
                "initial_count": len(initial),
                "tasks": list(TASKS),
                "inner_offsets": list(INNER_OFFSETS),
                "formal_seeds": list(FORMAL_SEEDS),
            },
        )
        print(f"plan-only initial_count={len(initial)}")
        return

    (OUTPUT / "STARTED").write_text(now() + "\n", encoding="ascii")
    generalist_result = wait_for_generalists(args.poll_seconds)
    wait_for_optional_precompute(args.poll_seconds)
    policies = selected_generalists(generalist_result)
    policies = select_generalist_checkpoints(policies)
    archive_generalists(policies)
    manifest = {
        "protocol": "nested_four_objective_sixd_v1_2026-07-30",
        "started_at": now(),
        "tasks": list(TASKS),
        "weights": WEIGHTS,
        "parameters": list(PARAMETERS),
        "bounds": {
            name: [float(low), float(high)]
            for name, low, high in zip(PARAMETERS, LOWER, UPPER)
        },
        "selected_generalists": policies,
        "generalist_curriculum_updates": [7000, 7000],
        "generalist_checkpoint_candidates_full_sixd_updates": list(
            GENERALIST_CHECKPOINT_OFFSETS
        ),
        "inner_updates": INNER_UPDATES,
        "inner_checkpoint_offsets": list(INNER_OFFSETS),
        "checkpoint_selection": (
            "validation seed; independent selected-checkpoint test seed"
        ),
        "search_train_seed": SEARCH_TRAIN_SEED,
        "formal_train_seeds": list(FORMAL_SEEDS),
        "initial_design_count": len(initial),
        "bo_batch_size": BO_BATCH_SIZE,
        "minimum_bo_batches": MIN_BO_BATCHES,
        "maximum_bo_batches": MAX_BO_BATCHES,
        "stopping": {
            "two_consecutive_batches_score_improvement_below": (
                MIN_SCORE_IMPROVEMENT
            ),
            "next_max_expected_improvement_below": MAX_EI_THRESHOLD,
            "two_consecutive_batches_relative_hypervolume_gain_below": (
                MAX_HV_RELATIVE_GAIN
            ),
        },
        "old_results_read_only": True,
        "old_joint_winner": base.design_dict(OLD_JOINT_WINNER),
        "single_task_specialists_are_fixed_calibration_designs": {
            task: base.design_dict(design)
            for task, design in SPECIALISTS.items()
        },
        "speed_protocol": {
            "slope_deg": 10.0,
            "step_component_mps": 0.0,
            "training_command_max_mps": 4.0,
            "motor_power_envelope": True,
            "specific_power_w_per_kg": 450.0,
        },
    }
    atomic_json(OUTPUT / "RUN_MANIFEST.json", manifest)

    initial_rows = evaluate_nested(
        "initial",
        initial,
        policies,
        (SEARCH_TRAIN_SEED,),
        "search",
    )
    anchors = calibration_anchors(initial_rows)
    atomic_json(OUTPUT / "NORMALIZATION.json", anchors)
    all_rows = list(initial_rows)
    scored = score_rows(all_rows, anchors)
    atomic_json(OUTPUT / "SEARCH_INITIAL.json", scored)

    diagnostics = []
    stopped_reason = None
    for batch_index in range(1, MAX_BO_BATCHES + 1):
        proposals, proposal_diagnostics = (
            propose_expected_improvement(
                scored,
                BO_BATCH_SIZE,
                SEARCH_TRAIN_SEED + 100 * batch_index,
            )
        )
        existing_ids = {row["design_id"] for row in scored}
        proposals = [
            design
            for design in proposals
            if base.design_key(design) not in existing_ids
        ]
        if not proposals:
            stopped_reason = "no_unique_expected_improvement_proposals"
            break
        best_before = float(scored[0]["equal_weight_score"])
        hv_before = hypervolume(
            scored, SEARCH_TRAIN_SEED + 5000 + batch_index
        )
        batch_rows = evaluate_nested(
            f"bo_batch_{batch_index}",
            proposals,
            policies,
            (SEARCH_TRAIN_SEED,),
            "search",
        )
        all_rows.extend(batch_rows)
        scored = score_rows(all_rows, anchors)
        best_after = float(scored[0]["equal_weight_score"])
        hv_after = hypervolume(
            scored, SEARCH_TRAIN_SEED + 5000 + batch_index
        )
        improvement = max(best_after - best_before, 0.0)
        relative_hv_gain = max(hv_after - hv_before, 0.0) / max(
            abs(hv_before), 1.0e-9
        )
        _, next_diagnostics = propose_expected_improvement(
            scored,
            BO_BATCH_SIZE,
            SEARCH_TRAIN_SEED + 100 * (batch_index + 1),
        )
        summary = {
            "batch": batch_index,
            "proposal_diagnostics": proposal_diagnostics,
            "designs": [
                base.design_dict(design) for design in proposals
            ],
            "score_before": best_before,
            "score_after": best_after,
            "best_score_improvement": improvement,
            "hypervolume_before": hv_before,
            "hypervolume_after": hv_after,
            "relative_hypervolume_gain": relative_hv_gain,
            "next_max_expected_improvement": next_diagnostics.get(
                "max_expected_improvement"
            ),
            "search_design_count": len(scored),
        }
        diagnostics.append(summary)
        atomic_json(
            OUTPUT / f"BO_BATCH_{batch_index}_SUMMARY.json",
            summary,
        )
        atomic_json(OUTPUT / "SEARCH_ALL.json", scored)
        atomic_json(
            OUTPUT / "SEARCH_PARETO_FRONT.json",
            pareto_front(scored),
        )
        atomic_json(OUTPUT / "BO_DIAGNOSTICS.json", diagnostics)

        if batch_index >= MIN_BO_BATCHES and len(diagnostics) >= 2:
            recent = diagnostics[-2:]
            small_scores = all(
                record["best_score_improvement"]
                < MIN_SCORE_IMPROVEMENT
                for record in recent
            )
            small_hv = all(
                record["relative_hypervolume_gain"]
                < MAX_HV_RELATIVE_GAIN
                for record in recent
            )
            max_ei = summary["next_max_expected_improvement"]
            small_ei = (
                max_ei is not None and max_ei < MAX_EI_THRESHOLD
            )
            if small_scores and small_hv and small_ei:
                stopped_reason = (
                    "two_consecutive_small_score_and_hypervolume_gains_"
                    "with_low_expected_improvement"
                )
                break
    if stopped_reason is None:
        stopped_reason = "maximum_bo_batch_guard_reached"
    atomic_json(
        OUTPUT / "BO_STOPPING.json",
        {
            "stopped_at": now(),
            "reason": stopped_reason,
            "completed_batches": len(diagnostics),
            "diagnostics": diagnostics,
        },
    )

    formal_designs = [
        np.asarray(row["design_vector"], dtype=np.float64)
        for row in scored[:3]
    ]
    formal_designs.extend(
        [OLD_JOINT_WINNER.copy(), NOMINAL.copy()]
    )
    formal_designs = base.unique_designs(formal_designs)
    formal_nested_rows = evaluate_nested(
        "formal",
        formal_designs,
        policies,
        FORMAL_SEEDS,
        "formal",
    )
    final_rows = aggregate_formal(formal_nested_rows, anchors)
    old_id = base.design_key(OLD_JOINT_WINNER)
    nominal_id = base.design_key(NOMINAL)
    old_rows = [row for row in final_rows if row["design_id"] == old_id]
    nominal_rows = [
        row for row in final_rows if row["design_id"] == nominal_id
    ]
    old_row = old_rows[0] if old_rows else None
    winner = final_rows[0]
    replacement_margin = (
        winner["equal_weight_score_mean"]
        - old_row["equal_weight_score_mean"]
        if old_row is not None
        else None
    )
    replacement_standard_error = (
        math.sqrt(
            winner["equal_weight_score_std"] ** 2
            / len(winner["seeds"])
            + old_row["equal_weight_score_std"] ** 2
            / len(old_row["seeds"])
        )
        if old_row is not None
        else None
    )
    replacement_confident = bool(
        old_row is not None
        and winner["design_id"] != old_id
        and replacement_margin
        > max(
            MIN_SCORE_IMPROVEMENT,
            1.96 * replacement_standard_error,
        )
    )
    final = {
        "winner": winner,
        "finalists": final_rows,
        "old_frozen_solution_under_nested_protocol": (
            old_row
        ),
        "nominal_under_nested_protocol": (
            nominal_rows[0] if nominal_rows else None
        ),
        "replacement_score_margin": replacement_margin,
        "replacement_margin_standard_error": (
            replacement_standard_error
        ),
        "replace_old_solution_recommended": replacement_confident,
        "replacement_rule": (
            "new mean exceeds old mean by both 0.02 normalized score "
            "and 1.96 pooled standard errors"
        ),
        "normalization": anchors,
        "bo_stopping_reason": stopped_reason,
    }
    atomic_json(OUTPUT / "FINAL_RESULTS.json", final)
    write_csv(OUTPUT / "FINAL_RESULTS.csv", final_rows)
    write_report(final_rows, anchors, diagnostics)

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
    atomic_json(
        OUTPUT / "LIFECYCLE.json",
        {
            "completed_at": now(),
            "gpu_state_after_children": gpu_state,
            "search_design_count": len(scored),
            "formal_design_count": len(final_rows),
            "formal_nested_record_count": len(formal_nested_rows),
        },
    )
    (OUTPUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
    (OUTPUT / "DONE").write_text(now() + "\n", encoding="ascii")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    if MIN_BO_BATCHES < 2 or MAX_BO_BATCHES < MIN_BO_BATCHES:
        parser.error("invalid BO batch bounds")
    if BO_BATCH_SIZE < 1:
        parser.error("BO batch size must be positive")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        import traceback

        traceback.print_exc()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "EXIT_CODE").write_text("1\n", encoding="ascii")
        raise
