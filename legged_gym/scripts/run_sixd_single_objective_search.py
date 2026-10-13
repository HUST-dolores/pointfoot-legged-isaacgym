#!/usr/bin/env python3
"""Unattended five-task, six-dimensional single-objective co-design search.

The script trains one design-randomized generalist per physical task, selects
an intermediate checkpoint on held-out designs, evaluates a shared space-
filling design set, adds task-specific Gaussian-process expected-improvement
points, and repeats the best candidates with higher-fidelity evaluation seeds.

It is intended to run on the two-GPU workstation under tmux.  All subprocesses
are explicit and leave STARTED, EXIT_CODE, DONE, console, and result artifacts.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm, qmc, spearmanr


ROOT = Path(
    os.environ.get(
        "SIXD_ROOT",
        "/home/ps/mydrive/xu/pointfoot-legged-gym",
    )
)
PYTHON = os.environ.get(
    "SIXD_PYTHON",
    "/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python",
)
OUT = Path(
    os.environ.get(
        "SIXD_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_single_objective_v1_2026-07-26",
    )
)
SEED = int(os.environ.get("SIXD_SEED", "20260726"))
GPU_IDS = [
    value.strip()
    for value in os.environ.get("SIXD_GPUS", "0,1").split(",")
    if value.strip()
]
PARAMETERS = ("thigh", "shank", "abad", "hip", "knee", "wheel")
TASKS = ("clearance", "slope", "payload", "speed", "jump")
LOWER = np.asarray([0.8, 0.8, 0.6, 0.6, 0.6, 0.6], dtype=np.float64)
UPPER = np.asarray([1.2, 1.2, 1.6, 1.6, 1.6, 1.6], dtype=np.float64)
NOMINAL = np.ones(6, dtype=np.float64)
MOTOR_NOMINAL_MASS_KG = np.asarray(
    [1.35, 1.35, 0.89, 0.365], dtype=np.float64
)
MOTOR_MASS_EXPONENT = float(
    os.environ.get("SIXD_MOTOR_MASS_EXPONENT", "1.0")
)
SEARCH_INITIAL_LHS = int(
    os.environ.get("SIXD_INITIAL_LHS", "14")
)
INITIAL_SAMPLER = os.environ.get(
    "SIXD_INITIAL_SAMPLER", "latin_hypercube"
).strip().lower()
if INITIAL_SAMPLER not in ("latin_hypercube", "sobol"):
    raise ValueError(
        "SIXD_INITIAL_SAMPLER must be latin_hypercube or sobol"
    )
SEARCH_BO1 = int(os.environ.get("SIXD_BO1", "10"))
SEARCH_BO2 = int(os.environ.get("SIXD_BO2", "4"))
FORMAL_SEEDS = tuple(
    int(value)
    for value in os.environ.get(
        "SIXD_FORMAL_SEEDS",
        "20260726,20260727,20260728",
    ).split(",")
    if value.strip()
)

TRAIN_SPECS = {
    "clearance": {
        "updates": 400,
        "checkpoint_offsets": [100, 200, 300, 400],
        "base_checkpoint": 27000,
    },
    "slope": {
        "updates": 600,
        "checkpoint_offsets": [300, 400, 500, 600],
        "base_checkpoint": 27000,
    },
    "payload": {
        "updates": 400,
        "checkpoint_offsets": [100, 200, 300, 400],
        "base_checkpoint": 27000,
    },
    "speed": {
        "updates": 800,
        "checkpoint_offsets": [200, 400, 600, 800],
        "base_checkpoint": 27000,
    },
    "jump": {
        "updates": 0,
        "checkpoint_offsets": [0],
        "base_checkpoint": 4000,
    },
}

FROZEN_JUMP_POLICY = {
    "run_name": "Jul11_21-43-33_jumpfwd23_residual_s79",
    "checkpoint": 4000,
    "protocol": (
        "frozen task-specific policy; no candidate adaptation and no "
        "six-dimensional design randomization"
    ),
}

TRAIN_LEVELS = {
    "clearance": "0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22",
    "slope": "0,0,0,5,10,15,20,25",
    "payload": "0,5,10,15,20,30,40,50",
    "speed": "0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6",
}

SEARCH_LEVELS = {
    "slope": ",".join(str(value) for value in range(0, 41, 4)),
    "payload": ",".join(
        str(value)
        for value in list(range(0, 65, 5)) + [70, 80]
    ),
    "speed": "0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,2.9,3.2",
}

FORMAL_LEVELS = {
    "slope": ",".join(str(value) for value in range(0, 41, 2)),
    "payload": ",".join(str(value) for value in range(0, 81, 5)),
    "speed": ",".join(
        f"{value:.1f}" for value in np.arange(0.5, 3.31, 0.2)
    ),
}

JUMP_HEIGHTS = "0.10,0.11,0.12,0.13,0.14,0.15,0.16"
JUMP_DISTANCES = "0.05,0.065,0.08,0.095,0.11,0.125,0.14"


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
    os.replace(str(temporary), str(path))


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def normalized(design):
    return (np.asarray(design, dtype=np.float64) - LOWER) / (UPPER - LOWER)


def denormalized(point):
    return LOWER + np.asarray(point, dtype=np.float64) * (UPPER - LOWER)


def design_dict(design):
    return {
        name: float(value)
        for name, value in zip(PARAMETERS, design)
    }


def design_key(design):
    payload = ",".join(f"{float(value):.8f}" for value in design)
    return hashlib.sha1(payload.encode("ascii")).hexdigest()[:10]


def unique_designs(designs):
    result = []
    seen = set()
    for design in designs:
        clipped = np.clip(
            np.asarray(design, dtype=np.float64), LOWER, UPPER
        )
        key = design_key(clipped)
        if key not in seen:
            seen.add(key)
            result.append(clipped)
    return result


def motor_mass_kg(design):
    scales = np.asarray(design, dtype=np.float64)[2:]
    return float(
        2.0
        * np.sum(
            MOTOR_NOMINAL_MASS_KG
            * np.power(scales, MOTOR_MASS_EXPONENT)
        )
    )


class Job:
    def __init__(
        self,
        name,
        command,
        environment,
        directory,
        expected_result,
    ):
        self.name = str(name)
        self.command = list(command)
        self.environment = dict(environment)
        self.directory = Path(directory)
        self.expected_result = Path(expected_result)
        self.elapsed_s = None


class EvalRequest:
    def __init__(
        self,
        task,
        run_name,
        checkpoint,
        design,
        seed,
        phase,
        fidelity,
    ):
        self.task = task
        self.run_name = run_name
        self.checkpoint = int(checkpoint)
        self.design = np.asarray(design, dtype=np.float64)
        self.seed = int(seed)
        self.phase = phase
        self.fidelity = fidelity
        self.design_id = design_key(self.design)
        self.job = make_eval_job(self)


_STATUS_LOCK = threading.Lock()
_STATUS = {
    "started_at": now_iso(),
    "stage": "initializing",
    "completed_jobs": 0,
    "total_jobs": 0,
    "failed_jobs": [],
    "last_update": now_iso(),
}


def update_status(**values):
    with _STATUS_LOCK:
        _STATUS.update(values)
        _STATUS["last_update"] = now_iso()
        atomic_json(OUT / "STATUS.json", _STATUS)


def run_one_job(job, gpu_id):
    job.directory.mkdir(parents=True, exist_ok=True)
    done_path = job.directory / "DONE"
    exit_path = job.directory / "EXIT_CODE"
    if (
        done_path.is_file()
        and exit_path.is_file()
        and exit_path.read_text().strip() == "0"
        and job.expected_result.is_file()
    ):
        job.elapsed_s = 0.0
        return "cached"

    (job.directory / "STARTED").write_text(
        now_iso() + "\n", encoding="ascii"
    )
    done_path.unlink(missing_ok=True)
    exit_path.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update(job.environment)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    environment["ROBOT_TYPE"] = "WF_TRON1A"
    environment["PATH"] = (
        str(Path(PYTHON).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    started = time.time()
    with (job.directory / "console.log").open(
        "w", encoding="utf-8"
    ) as console:
        process = subprocess.run(
            job.command,
            cwd=str(ROOT),
            env=environment,
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        )
    job.elapsed_s = time.time() - started
    exit_path.write_text(f"{process.returncode}\n", encoding="ascii")
    if process.returncode != 0:
        raise RuntimeError(
            f"{job.name} failed with exit code {process.returncode}; "
            f"see {job.directory / 'console.log'}"
        )
    if not job.expected_result.is_file():
        raise RuntimeError(
            f"{job.name} exited 0 without {job.expected_result}"
        )
    done_path.write_text(now_iso() + "\n", encoding="ascii")
    return "ran"


def run_jobs(stage, jobs):
    jobs = list(jobs)
    if not jobs:
        return
    work_queue = queue.Queue()
    for job in jobs:
        work_queue.put(job)
    failures = []
    elapsed = []
    state = {"completed": 0}
    lock = threading.Lock()
    update_status(
        stage=stage,
        completed_jobs=0,
        total_jobs=len(jobs),
        failed_jobs=[],
        eta_seconds=None,
    )

    def worker(gpu_id):
        while True:
            try:
                job = work_queue.get_nowait()
            except queue.Empty:
                return
            print(
                f"[sixd] START stage={stage} gpu={gpu_id} job={job.name}",
                flush=True,
            )
            try:
                outcome = run_one_job(job, gpu_id)
                print(
                    f"[sixd] DONE stage={stage} gpu={gpu_id} "
                    f"job={job.name} mode={outcome} "
                    f"elapsed_s={job.elapsed_s:.1f}",
                    flush=True,
                )
                with lock:
                    if job.elapsed_s and job.elapsed_s > 0.0:
                        elapsed.append(job.elapsed_s)
            except Exception as error:
                message = f"{job.name}: {error}"
                failures.append(message)
                print(f"[sixd] FAILED {message}", flush=True)
            finally:
                with lock:
                    state["completed"] += 1
                    remaining = len(jobs) - state["completed"]
                    mean_elapsed = (
                        float(np.mean(elapsed)) if elapsed else None
                    )
                    eta = (
                        remaining
                        * mean_elapsed
                        / max(len(GPU_IDS), 1)
                        if mean_elapsed is not None
                        else None
                    )
                    update_status(
                        stage=stage,
                        completed_jobs=state["completed"],
                        total_jobs=len(jobs),
                        failed_jobs=list(failures),
                        eta_seconds=eta,
                    )
                work_queue.task_done()

    threads = [
        threading.Thread(
            target=worker,
            args=(gpu_id,),
            daemon=False,
            name=f"gpu-worker-{gpu_id}",
        )
        for gpu_id in GPU_IDS
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(
            f"{len(failures)} jobs failed in stage {stage}: "
            + "; ".join(failures)
        )


def common_motor_environment():
    return {
        "PHYS_MOTOR_MASS_EXPONENT": str(MOTOR_MASS_EXPONENT),
        "PHYS_MOTOR_NOMINAL_MASSES_KG": ",".join(
            str(value) for value in MOTOR_NOMINAL_MASS_KG
        ),
    }


def make_training_job(task):
    spec = TRAIN_SPECS[task]
    directory = OUT / "generalists" / task / "train"
    result_path = directory / "result.json"
    environment = common_motor_environment()
    environment.update(
        {
            "PHYS_MAX_GPU_CONTACT_PAIRS": str(2**25),
        }
    )
    tag = f"sixdgen_{task}_s{SEED}_u{spec['updates']}"
    if task == "jump":
        environment.update(
            {
                "PHYS_JUMP_RANDOMIZE_DESIGN": "1",
                "PHYS_JUMP_NUM_ENVS": "4096",
                "PHYS_JUMP_MORPH_BUCKETS": "64",
                "PHYS_JUMP_UPDATES": str(spec["updates"]),
                "PHYS_JUMP_SEED": str(SEED),
                "PHYS_JUMP_TAG": tag,
                "PHYS_JUMP_RESULT_JSON": str(result_path),
                "PHYS_JUMP_HEIGHT_RANGE": "0.10,0.16",
                "PHYS_JUMP_DISTANCE_RANGE": "0.05,0.14",
            }
        )
        script = "legged_gym/scripts/train_physical_jump.py"
    else:
        environment.update(
            {
                "PHYS_TRAIN_TASK": task,
                "PHYS_TRAIN_RANDOMIZE_DESIGN": "1",
                "PHYS_TRAIN_NUM_ENVS": "4096",
                "PHYS_TRAIN_MORPH_BUCKETS": "64",
                "PHYS_TRAIN_UPDATES": str(spec["updates"]),
                "PHYS_TRAIN_SEED": str(SEED),
                "PHYS_TRAIN_TAG": tag,
                "PHYS_TRAIN_RESULT_JSON": str(result_path),
                "PHYS_TRAIN_CLEARANCE_LEVELS": TRAIN_LEVELS["clearance"],
                "PHYS_TRAIN_SLOPE_LEVELS": TRAIN_LEVELS["slope"],
                "PHYS_TRAIN_PAYLOAD_LEVELS": TRAIN_LEVELS["payload"],
                "PHYS_TRAIN_SPEED_LEVELS": TRAIN_LEVELS["speed"],
                "PHYS_TRAIN_SLOPE_STATIC": "1" if task == "slope" else "0",
            }
        )
        if task == "slope":
            environment.update(
                {
                    "PHYS_TRAIN_TOTAL_VEL_REWARD": "16.0",
                    "PHYS_TRAIN_TOTAL_VEL_SIGMA": "0.02",
                    "PHYS_TRAIN_TOTAL_VEL_ERROR_REWARD": "-8.0",
                    "PHYS_TRAIN_STATIC_XY_VEL_REWARD": "8.0",
                    "PHYS_TRAIN_STATIC_ROLL_VEL_REWARD": "6.0",
                    "PHYS_TRAIN_STATIC_STEP_VEL_REWARD": "6.0",
                }
            )
        script = "legged_gym/scripts/train_physical_capability.py"
    return Job(
        f"train-{task}",
        [PYTHON, script, "--headless"],
        environment,
        directory,
        result_path,
    )


def apply_design_environment(environment, prefix, design):
    for name, value in zip(PARAMETERS, design):
        environment[f"{prefix}_{name.upper()}"] = f"{float(value):.8f}"


def make_eval_job(request):
    directory = (
        OUT
        / "evaluations"
        / request.phase
        / request.task
        / f"ckpt_{request.checkpoint}"
        / request.design_id
        / f"seed_{request.seed}"
    )
    environment = common_motor_environment()
    if request.task == "jump":
        result_path = directory / "result.json"
        environment.update(
            {
                "JUMPFWD_RUN": request.run_name,
                "JUMPFWD_CKPT": str(request.checkpoint),
                "JUMPFWD_SEED": str(request.seed),
                "JUMPFWD_NOMINAL_EVAL": "1",
                "JUMPFWD_FLAT_EVAL": "1",
                "JUMPFWD_EVAL_FRICTION": "0.8",
                "JUMPFWD_GRID_HEIGHTS": JUMP_HEIGHTS,
                "JUMPFWD_GRID_DISTANCES": JUMP_DISTANCES,
                "JUMPFWD_EVAL_PREFIX": (
                    f"sixd_{request.phase}_{request.design_id}_"
                    f"ckpt{request.checkpoint}_s{request.seed}"
                ),
                "JUMPFWD_CSV": str(directory / "events.csv"),
                "JUMPFWD_SUMMARY": str(result_path),
                "JUMPFWD_GRID_REPLICAS": (
                    "16" if request.fidelity == "formal" else "8"
                ),
                "JUMPFWD_STEPS": (
                    "2400" if request.fidelity == "formal" else "1400"
                ),
            }
        )
        apply_design_environment(
            environment, "JUMPFWD_EVAL", request.design
        )
        script = "legged_gym/scripts/measure_jumpfwd.py"
    else:
        result_path = directory / "result.json"
        formal = request.fidelity == "formal"
        environment.update(
            {
                "PHYS_EVAL_TASK": request.task,
                "PHYS_EVAL_RUN": request.run_name,
                "PHYS_EVAL_CKPT": str(request.checkpoint),
                "PHYS_EVAL_SEED": str(request.seed),
                "PHYS_EVAL_OUT": str(directory),
                "PHYS_EVAL_PREFIX": (
                    f"sixd_{request.phase}_{request.task}_"
                    f"{request.design_id}_ckpt{request.checkpoint}_"
                    f"s{request.seed}"
                ),
                "PHYS_EVAL_SETTLE_S": "2.0",
                "PHYS_EVAL_DURATION_S": "8.0" if formal else "6.0",
                "PHYS_EVAL_SLOPE_STATIC": (
                    "1" if request.task == "slope" else "0"
                ),
                "PHYS_EVAL_REPLICAS": "64" if formal else "20",
                "PHYS_EVAL_NUM_ENVS": "256" if formal else "160",
            }
        )
        if request.task in SEARCH_LEVELS:
            environment["PHYS_EVAL_LEVELS"] = (
                FORMAL_LEVELS[request.task]
                if formal
                else SEARCH_LEVELS[request.task]
            )
        apply_design_environment(
            environment, "PHYS_EVAL", request.design
        )
        script = "legged_gym/scripts/physical_capability_eval.py"
    return Job(
        (
            f"eval-{request.phase}-{request.task}-"
            f"{request.design_id}-s{request.seed}"
        ),
        [PYTHON, script, "--headless"],
        environment,
        directory,
        result_path,
    )


def objective_from_result(task, result):
    if task == "jump":
        value = result.get("reachable_area", {}).get(
            "metric_value_m2"
        )
    else:
        summary = result.get("summary", {})
        value = summary.get("metric_value_interpolated")
        if value is None:
            value = summary.get("metric_value")
    return float(value) if value is not None and np.isfinite(value) else 0.0


def record_from_request(request):
    result = read_json(request.job.expected_result)
    objective = objective_from_result(request.task, result)
    if request.task == "jump":
        raw_summary = result.get("reachable_area", {})
    else:
        raw_summary = result.get("summary", {})
    return {
        "task": request.task,
        "phase": request.phase,
        "fidelity": request.fidelity,
        "run_name": request.run_name,
        "checkpoint": request.checkpoint,
        "seed": request.seed,
        "design_id": request.design_id,
        "design": design_dict(request.design),
        "design_vector": [float(value) for value in request.design],
        "objective": objective,
        "motor_module_mass_kg": motor_mass_kg(request.design),
        "result_path": str(request.job.expected_result),
        "summary": raw_summary,
    }


def save_records(path, records):
    atomic_json(path, records)
    csv_path = Path(path).with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "task",
            "phase",
            "fidelity",
            "seed",
            "checkpoint",
            *PARAMETERS,
            "objective",
            "motor_module_mass_kg",
            "result_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                key: record[key]
                for key in (
                    "task",
                    "phase",
                    "fidelity",
                    "seed",
                    "checkpoint",
                    "objective",
                    "motor_module_mass_kg",
                    "result_path",
                )
            }
            row.update(record["design"])
            writer.writerow(row)


def validation_designs():
    return [
        NOMINAL.copy(),
        np.asarray([0.85, 1.15, 0.75, 1.25, 1.35, 0.80]),
        np.asarray([1.15, 0.85, 1.35, 0.80, 0.75, 1.30]),
    ]


def select_generalist_checkpoints(training_results):
    requests = []
    for task in TASKS:
        training = training_results[task]
        spec = TRAIN_SPECS[task]
        for offset in spec["checkpoint_offsets"]:
            checkpoint = spec["base_checkpoint"] + offset
            for design in validation_designs():
                requests.append(
                    EvalRequest(
                        task,
                        training["run_name"],
                        checkpoint,
                        design,
                        SEED,
                        "checkpoint_validation",
                        "search",
                    )
                )
    run_jobs(
        "checkpoint_validation",
        [request.job for request in requests],
    )
    records = [record_from_request(request) for request in requests]
    save_records(OUT / "checkpoint_validation.json", records)

    selection = {}
    for task in TASKS:
        checkpoint_groups = {}
        for record in records:
            if record["task"] != task:
                continue
            checkpoint_groups.setdefault(
                record["checkpoint"], []
            ).append(record["objective"])
        rows = []
        for checkpoint, values in sorted(checkpoint_groups.items()):
            rows.append(
                {
                    "checkpoint": int(checkpoint),
                    "mean_objective": float(np.mean(values)),
                    "minimum_objective": float(np.min(values)),
                    "values": [float(value) for value in values],
                }
            )
        if not rows or max(row["mean_objective"] for row in rows) <= 0.0:
            raise RuntimeError(
                f"{task} checkpoint validation has no positive physical "
                "objective; refusing to enter hardware search"
            )
        best = max(
            rows,
            key=lambda row: (
                row["mean_objective"],
                row["minimum_objective"],
                -row["checkpoint"],
            ),
        )
        selection[task] = {
            "run_name": training_results[task]["run_name"],
            "checkpoint": best["checkpoint"],
            "validation": rows,
        }
    atomic_json(OUT / "selected_generalists.json", selection)
    return selection


def initial_designs():
    designs = [NOMINAL.copy()]
    for dimension in range(6):
        low = NOMINAL.copy()
        high = NOMINAL.copy()
        low[dimension] = LOWER[dimension]
        high[dimension] = UPPER[dimension]
        designs.extend((low, high))
    designs.extend(
        (
            LOWER.copy(),
            UPPER.copy(),
            np.asarray([0.8, 0.8, 1.0, 1.0, 1.0, 1.0]),
            np.asarray([1.2, 1.2, 1.0, 1.0, 1.0, 1.0]),
            np.asarray([0.9, 0.9, 1.0, 1.0, 1.0, 1.0]),
            np.asarray([1.1, 1.1, 1.0, 1.0, 1.0, 1.0]),
            np.asarray([0.8, 0.8, 0.9571, 0.9571, 0.9571, 0.9571]),
        )
    )
    if INITIAL_SAMPLER == "sobol":
        sampler = qmc.Sobol(d=6, scramble=True, seed=SEED)
        exponent = int(math.ceil(math.log2(max(SEARCH_INITIAL_LHS, 1))))
        samples = sampler.random_base2(exponent)[:SEARCH_INITIAL_LHS]
    else:
        sampler = qmc.LatinHypercube(d=6, seed=SEED)
        samples = sampler.random(SEARCH_INITIAL_LHS)
    designs.extend(
        denormalized(point)
        for point in samples
    )
    return unique_designs(designs)


def matern52(x1, x2, length_scales):
    difference = (
        x1[:, None, :] - x2[None, :, :]
    ) / length_scales.reshape(1, 1, -1)
    radius = np.sqrt(np.sum(difference * difference, axis=2))
    sqrt5_radius = math.sqrt(5.0) * radius
    return (
        1.0
        + sqrt5_radius
        + (5.0 / 3.0) * radius * radius
    ) * np.exp(-sqrt5_radius)


class GaussianProcess:
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=np.float64)
        raw_y = np.asarray(y, dtype=np.float64)
        self.y_mean = float(np.mean(raw_y))
        self.y_std = float(np.std(raw_y))
        if self.y_std < 1.0e-9:
            self.y_std = 1.0
        self.y = (raw_y - self.y_mean) / self.y_std
        dimension = self.x.shape[1]

        def nll(parameters):
            length_scales = np.exp(parameters[:dimension])
            signal = math.exp(parameters[dimension])
            noise = math.exp(parameters[dimension + 1])
            covariance = (
                signal * signal
                * matern52(self.x, self.x, length_scales)
            )
            covariance.flat[:: len(self.x) + 1] += (
                noise * noise + 1.0e-7
            )
            try:
                factor = cho_factor(
                    covariance, lower=True, check_finite=False
                )
                alpha = cho_solve(
                    factor, self.y, check_finite=False
                )
            except Exception:
                return 1.0e20
            log_determinant = 2.0 * np.sum(
                np.log(np.diag(factor[0]))
            )
            return float(
                0.5 * np.dot(self.y, alpha)
                + 0.5 * log_determinant
                + 0.5 * len(self.x) * math.log(2.0 * math.pi)
            )

        initial = np.concatenate(
            (
                np.full(dimension, math.log(0.35)),
                [0.0, math.log(0.08)],
            )
        )
        bounds = [
            (math.log(0.05), math.log(2.0))
        ] * dimension + [
            (math.log(0.1), math.log(10.0)),
            (math.log(0.005), math.log(0.6)),
        ]
        result = minimize(
            nll,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 160},
        )
        parameters = (
            result.x if np.all(np.isfinite(result.x)) else initial
        )
        self.length_scales = np.exp(parameters[:dimension])
        self.signal = math.exp(parameters[dimension])
        self.noise = math.exp(parameters[dimension + 1])
        covariance = (
            self.signal
            * self.signal
            * matern52(self.x, self.x, self.length_scales)
        )
        covariance.flat[:: len(self.x) + 1] += (
            self.noise * self.noise + 1.0e-7
        )
        self.factor = cho_factor(
            covariance, lower=True, check_finite=False
        )
        self.alpha = cho_solve(
            self.factor, self.y, check_finite=False
        )
        self.fit = {
            "length_scales": self.length_scales.tolist(),
            "signal": self.signal,
            "noise": self.noise,
            "optimizer_success": bool(result.success),
            "negative_log_likelihood": float(nll(parameters)),
        }

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        cross = (
            self.signal
            * self.signal
            * matern52(x, self.x, self.length_scales)
        )
        mean_standard = np.dot(cross, self.alpha)
        solved = cho_solve(
            self.factor, cross.T, check_finite=False
        )
        variance_standard = np.maximum(
            self.signal * self.signal
            - np.sum(cross * solved.T, axis=1),
            1.0e-12,
        )
        return (
            self.y_mean + self.y_std * mean_standard,
            self.y_std * np.sqrt(variance_standard),
        )


def propose_expected_improvement(records, count, seed):
    x = np.asarray(
        [normalized(record["design_vector"]) for record in records],
        dtype=np.float64,
    )
    y = np.asarray(
        [record["objective"] for record in records],
        dtype=np.float64,
    )
    if float(np.std(y)) < 1.0e-12:
        sampler = qmc.LatinHypercube(d=6, seed=seed)
        return [
            denormalized(point)
            for point in sampler.random(count)
        ], {"fallback": "constant_objective"}
    model = GaussianProcess(x, y)
    sampler = qmc.Sobol(d=6, scramble=True, seed=seed)
    pool = sampler.random_base2(m=16)
    means = np.empty(len(pool), dtype=np.float64)
    standard_deviations = np.empty(len(pool), dtype=np.float64)
    chunk_size = 8192
    for start in range(0, len(pool), chunk_size):
        stop = min(start + chunk_size, len(pool))
        means[start:stop], standard_deviations[start:stop] = (
            model.predict(pool[start:stop])
        )
    improvement = means - float(np.max(y)) - 0.01 * max(
        float(np.std(y)), 1.0e-9
    )
    z_value = improvement / np.maximum(
        standard_deviations, 1.0e-12
    )
    expected_improvement = (
        improvement * norm.cdf(z_value)
        + standard_deviations * norm.pdf(z_value)
    )
    order = np.argsort(expected_improvement)[::-1]
    selected = []
    minimum_distance = 0.08
    for index in order:
        point = pool[index]
        existing_distance = np.min(
            np.linalg.norm(x - point, axis=1)
        )
        selected_distance = (
            min(
                np.linalg.norm(point - previous)
                for previous in selected
            )
            if selected
            else float("inf")
        )
        if (
            existing_distance >= 0.035
            and selected_distance >= minimum_distance
        ):
            selected.append(point)
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in order:
            point = pool[index]
            if all(
                np.linalg.norm(point - previous) >= 0.03
                for previous in selected
            ):
                selected.append(point)
            if len(selected) == count:
                break
    return [
        denormalized(point) for point in selected[:count]
    ], model.fit


def evaluate_design_stage(
    phase,
    designs_by_task,
    selected_generalists,
    fidelity="search",
    seeds=(SEED,),
):
    requests = []
    for task, designs in designs_by_task.items():
        selected = selected_generalists[task]
        for design in designs:
            for seed in seeds:
                requests.append(
                    EvalRequest(
                        task,
                        selected["run_name"],
                        selected["checkpoint"],
                        design,
                        seed,
                        phase,
                        fidelity,
                    )
                )
    run_jobs(phase, [request.job for request in requests])
    return [record_from_request(request) for request in requests]


def parameter_correlations(records):
    result = {}
    for task in TASKS:
        task_records = [
            record
            for record in records
            if record["task"] == task
            and record["phase"] == "initial_search"
        ]
        objective = np.asarray(
            [record["objective"] for record in task_records]
        )
        correlations = {}
        for parameter_id, parameter in enumerate(PARAMETERS):
            values = np.asarray(
                [
                    record["design_vector"][parameter_id]
                    for record in task_records
                ]
            )
            coefficient, p_value = spearmanr(values, objective)
            correlations[parameter] = {
                "spearman_rho": (
                    float(coefficient)
                    if np.isfinite(coefficient)
                    else None
                ),
                "p_value_descriptive_only": (
                    float(p_value)
                    if np.isfinite(p_value)
                    else None
                ),
            }
        result[task] = correlations
    return result


def formal_aggregate(records):
    groups = {}
    for record in records:
        key = (record["task"], record["design_id"])
        groups.setdefault(key, []).append(record)
    task_rows = {task: [] for task in TASKS}
    for (task, design_id), values in groups.items():
        objectives = np.asarray(
            [record["objective"] for record in values],
            dtype=np.float64,
        )
        row = {
            "task": task,
            "design_id": design_id,
            "design": values[0]["design"],
            "design_vector": values[0]["design_vector"],
            "seed_count": len(values),
            "seeds": [record["seed"] for record in values],
            "objective_by_seed": objectives.tolist(),
            "objective_mean": float(np.mean(objectives)),
            "objective_std": (
                float(np.std(objectives, ddof=1))
                if len(objectives) > 1
                else 0.0
            ),
            "motor_module_mass_kg": values[0][
                "motor_module_mass_kg"
            ],
            "result_paths": [
                record["result_path"] for record in values
            ],
            "summaries": [
                record["summary"] for record in values
            ],
        }
        task_rows[task].append(row)
    final = {}
    for task, rows in task_rows.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                -row["objective_mean"],
                row["objective_std"],
                row["motor_module_mass_kg"],
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        final[task] = {
            "winner": ranked[0],
            "finalists": ranked,
        }
    return final


def write_final_report(final, correlations, all_search_records):
    atomic_json(OUT / "FINAL_RESULTS.json", final)
    atomic_json(OUT / "PARAMETER_CORRELATIONS.json", correlations)
    rows = []
    for task in TASKS:
        winner = final[task]["winner"]
        row = {
            "task": task,
            **winner["design"],
            "objective_mean": winner["objective_mean"],
            "objective_std": winner["objective_std"],
            "seed_count": winner["seed_count"],
            "motor_module_mass_kg": winner[
                "motor_module_mass_kg"
            ],
        }
        rows.append(row)
    with (OUT / "FINAL_RESULTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    units = {
        "clearance": "m",
        "slope": "deg",
        "payload": "kg",
        "speed": "m/s",
        "jump": "m^2",
    }
    lines = [
        "# Six-dimensional single-objective results",
        "",
        "Each row uses one task-specific fixed policy. Motor mass is "
        "baseline-anchored with alpha=1.0. Clearance, slope, payload, and "
        "speed use design-randomized generalists; jump freezes the original "
        "s79 controller for every design. No 20% weighted aggregate is used.",
        "",
        "| Task | thigh | shank | abad | hip | knee | wheel | "
        "objective mean +/- std | unit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for task in TASKS:
        winner = final[task]["winner"]
        design = winner["design"]
        lines.append(
            f"| {task} | {design['thigh']:.4f} | "
            f"{design['shank']:.4f} | {design['abad']:.4f} | "
            f"{design['hip']:.4f} | {design['knee']:.4f} | "
            f"{design['wheel']:.4f} | "
            f"{winner['objective_mean']:.6g} +/- "
            f"{winner['objective_std']:.3g} | {units[task]} |"
        )
    lines.extend(
        [
            "",
            "Selection order is mean capability, then lower repeat standard "
            "deviation, then lower actuator-module mass only as an exact tie-break.",
            "",
            f"Search evaluations: {len(all_search_records)}.",
            f"Formal seeds: {list(FORMAL_SEEDS)}.",
        ]
    )
    (OUT / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def snapshot_sources():
    destination = OUT / "code_snapshot"
    sources = (
        "legged_gym/utils/motor_cost.py",
        "legged_gym/envs/base/base_task.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py",
        "legged_gym/scripts/train_physical_capability.py",
        "legged_gym/scripts/train_physical_jump.py",
        "legged_gym/scripts/physical_capability_eval.py",
        "legged_gym/scripts/measure_jumpfwd.py",
        "legged_gym/scripts/run_sixd_single_objective_search.py",
    )
    for relative in sources:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STARTED").write_text(now_iso() + "\n", encoding="ascii")
    (OUT / "DONE").unlink(missing_ok=True)
    (OUT / "EXIT_CODE").unlink(missing_ok=True)
    snapshot_sources()
    manifest = {
        "protocol": "sixd_single_objective_v1_2026-07-26",
        "started_at": now_iso(),
        "root": str(ROOT),
        "python": PYTHON,
        "output": str(OUT),
        "seed": SEED,
        "gpus": GPU_IDS,
        "tasks": list(TASKS),
        "parameters": list(PARAMETERS),
        "bounds": {
            name: [float(low), float(high)]
            for name, low, high in zip(PARAMETERS, LOWER, UPPER)
        },
        "motor_mass_model": {
            "type": "baseline_anchored_power_law",
            "exponent": MOTOR_MASS_EXPONENT,
            "nominal_masses_kg": MOTOR_NOMINAL_MASS_KG.tolist(),
        },
        "training": TRAIN_SPECS,
        "frozen_jump_policy": FROZEN_JUMP_POLICY,
        "formal_seeds": list(FORMAL_SEEDS),
    }
    atomic_json(OUT / "RUN_MANIFEST.json", manifest)

    training_jobs = [
        make_training_job(task) for task in TASKS if task != "jump"
    ]
    run_jobs("generalist_training", training_jobs)
    training_results = {}
    for task in TASKS:
        if task == "jump":
            training_results[task] = dict(FROZEN_JUMP_POLICY)
        else:
            training_results[task] = read_json(
                OUT / "generalists" / task / "train" / "result.json"
            )
    selected = select_generalist_checkpoints(training_results)

    shared_designs = initial_designs()
    initial_records = evaluate_design_stage(
        "initial_search",
        {task: shared_designs for task in TASKS},
        selected,
    )
    save_records(OUT / "initial_search_records.json", initial_records)
    search_records = list(initial_records)
    gp_history = {}

    bo1_designs = {}
    for task in TASKS:
        task_records = [
            record for record in search_records if record["task"] == task
        ]
        designs, fit = propose_expected_improvement(
            task_records,
            SEARCH_BO1,
            SEED + 100 + TASKS.index(task),
        )
        bo1_designs[task] = unique_designs(designs)
        gp_history.setdefault(task, []).append(
            {"phase": "bo1", "fit": fit}
        )
    bo1_records = evaluate_design_stage(
        "bo1_search", bo1_designs, selected
    )
    search_records.extend(bo1_records)
    save_records(OUT / "search_records_after_bo1.json", search_records)

    bo2_designs = {}
    for task in TASKS:
        task_records = [
            record for record in search_records if record["task"] == task
        ]
        designs, fit = propose_expected_improvement(
            task_records,
            SEARCH_BO2,
            SEED + 200 + TASKS.index(task),
        )
        bo2_designs[task] = unique_designs(designs)
        gp_history.setdefault(task, []).append(
            {"phase": "bo2", "fit": fit}
        )
    bo2_records = evaluate_design_stage(
        "bo2_search", bo2_designs, selected
    )
    search_records.extend(bo2_records)
    save_records(OUT / "ALL_SEARCH_RECORDS.json", search_records)
    atomic_json(OUT / "GP_FITS.json", gp_history)

    formal_designs = {}
    for task in TASKS:
        task_records = sorted(
            (
                record
                for record in search_records
                if record["task"] == task
            ),
            key=lambda record: (
                -record["objective"],
                record["motor_module_mass_kg"],
            ),
        )
        top = [
            np.asarray(record["design_vector"], dtype=np.float64)
            for record in task_records[:3]
        ]
        formal_designs[task] = unique_designs(top + [NOMINAL])

    formal_records = evaluate_design_stage(
        "formal_repeats",
        formal_designs,
        selected,
        fidelity="formal",
        seeds=FORMAL_SEEDS,
    )
    save_records(OUT / "FORMAL_RECORDS.json", formal_records)
    final = formal_aggregate(formal_records)
    correlations = parameter_correlations(search_records)
    write_final_report(final, correlations, search_records)

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
        OUT / "LIFECYCLE.json",
        {
            "completed_at": now_iso(),
            "gpu_state_after_children": gpu_state,
            "formal_result_count": len(formal_records),
            "search_result_count": len(search_records),
        },
    )
    update_status(
        stage="complete",
        completed_jobs=1,
        total_jobs=1,
        failed_jobs=[],
        eta_seconds=0.0,
    )
    (OUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
    (OUT / "DONE").write_text(now_iso() + "\n", encoding="ascii")
    print(
        f"[sixd] COMPLETE results={OUT / 'FINAL_RESULTS.json'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        error_text = traceback.format_exc()
        (OUT / "ERROR.log").write_text(
            error_text, encoding="utf-8"
        )
        (OUT / "EXIT_CODE").write_text("1\n", encoding="ascii")
        update_status(
            stage="failed",
            failed_jobs=[error_text],
            eta_seconds=None,
        )
        print(error_text, file=sys.stderr, flush=True)
        raise
