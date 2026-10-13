#!/usr/bin/env python3
"""Train four task policies with a 2D-to-6D hardware curriculum.

Each task is trained in two consecutive stages:

1. Randomize thigh and shank scales while keeping all motors nominal.
2. Keep the leg randomization and independently randomize all four motor groups.

The driver is restartable, never edits shared configuration files, and assigns
one task pipeline to each GPU worker at a time.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time


DEFAULT_REPO = Path("/home/ps/mydrive/xu/pointfoot-legged-gym")
DEFAULT_PYTHON = Path(
    "/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
)
DEFAULT_ROOT = Path(
    "/home/ps/mydrive/xu/codesign_runs/"
    "nested_four_objective_sixd_2026-07-30/generalists"
)
FLAT_BASE_RUN = "Jul03_15-59-41_"
FLAT_BASE_CHECKPOINT = 13000
JUMP_BASE_RUN = "Jul11_21-43-33_jumpfwd23_residual_s79"
JUMP_BASE_CHECKPOINT = 4000
MOTOR_GROUPS = ("abad", "hip", "knee", "wheel")
MOTOR_MASSES = (1.35, 1.35, 0.89, 0.365)
TASKS = ("clearance", "payload", "speed", "jump")
LOG_DIR_PATTERN = re.compile(r"\blog_dir=(\S+)")
CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")


TASK_CONFIGS = {
    "clearance": {
        "kind": "flat",
        "learning_rates": (1.0e-4, 5.0e-5),
        "entropy": (0.001, 0.001),
        "reference_coef": (2.0, 5.0),
        "anchor_fraction": 0.25,
        "extra": {
            "PHYS_TRAIN_CLEARANCE_LEVELS": (
                "0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22"
            ),
        },
    },
    "payload": {
        "kind": "flat",
        "learning_rates": (1.0e-4, 5.0e-5),
        "entropy": (0.001, 0.001),
        "reference_coef": (2.0, 5.0),
        "anchor_fraction": 0.25,
        "extra": {
            "PHYS_TRAIN_PAYLOAD_LEVELS": (
                "0,5,10,15,20,30,40,50"
            ),
        },
    },
    "speed": {
        "kind": "flat",
        "learning_rates": (1.0e-4, 3.0e-5),
        "entropy": (0.001, 0.0005),
        "reference_coef": (2.0, 5.0),
        "anchor_fraction": 0.25,
        "extra": {
            "PHYS_TRAIN_SPEED_LEVELS": (
                "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
            ),
            "PHYS_TRAIN_SPEED_SLOPE_DEG": "10.0",
            "PHYS_TRAIN_SPEED_STEP_COMPONENT": "0.0",
            "PHYS_MOTOR_POWER_ENVELOPE": "1",
            "PHYS_MOTOR_SPECIFIC_POWER_W_PER_KG": "450.0",
            "PHYS_MOTOR_POWER_EXPONENT": "1.0",
        },
    },
    "jump": {
        "kind": "jump",
        "learning_rates": (3.0e-5, 2.0e-5),
        "entropy": (0.0, 0.0),
        "reference_coef": (10.0, 15.0),
        "anchor_fraction": 0.50,
        "extra": {
            "PHYS_JUMP_HEIGHT_RANGE": "0.10,0.16",
            "PHYS_JUMP_DISTANCE_RANGE": "0.05,0.14",
            "PHYS_JUMP_FRICTION": "0.8",
            "PHYS_JUMP_RESUME_NOISE_STD": "0.05",
            "PHYS_JUMP_INIT_RANDOM_EP_LEN": "1",
        },
    },
}


def parse_csv(value, converter):
    return tuple(
        converter(item.strip())
        for item in value.split(",")
        if item.strip()
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


def valid_result(path):
    path = Path(path)
    if not path.exists():
        return False
    try:
        result = read_json(path)
    except (OSError, ValueError):
        return False
    checkpoint = Path(result.get("checkpoint", ""))
    return checkpoint.is_file()


def latest_checkpoint(log_dir, target_iteration):
    log_dir = Path(log_dir)
    candidates = []
    if not log_dir.is_dir():
        return None
    for path in log_dir.glob("model_*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            iteration = int(match.group(1))
            if iteration <= target_iteration:
                candidates.append((iteration, path))
    return max(candidates, default=None)


def prior_attempt_resume(stage_dir, target_iteration):
    candidates = []
    for console in sorted(Path(stage_dir).glob("console_attempt_*.log")):
        try:
            text = console.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        matches = LOG_DIR_PATTERN.findall(text)
        if not matches:
            continue
        log_dir = Path(matches[-1])
        latest = latest_checkpoint(log_dir, target_iteration)
        if latest is not None:
            candidates.append((latest[0], log_dir, latest[1]))
    return max(candidates, default=None)


def base_source(task):
    if TASK_CONFIGS[task]["kind"] == "jump":
        return {
            "run_name": JUMP_BASE_RUN,
            "checkpoint_iteration": JUMP_BASE_CHECKPOINT,
        }
    return {
        "run_name": FLAT_BASE_RUN,
        "checkpoint_iteration": FLAT_BASE_CHECKPOINT,
    }


def source_from_result(result):
    return {
        "run_name": result["run_name"],
        "checkpoint_iteration": int(result["final_iteration"]),
    }


def stage_ranges(stage):
    motors = (1.0, 1.0) if stage == 1 else (0.6, 1.6)
    return {
        "thigh": (0.8, 1.2),
        "shank": (0.8, 1.2),
        **{group: motors for group in MOTOR_GROUPS},
    }


def flat_environment(
    args, task, stage, source, result_path, updates
):
    config = TASK_CONFIGS[task]
    ranges = stage_ranges(stage)
    environment = {
        "PHYS_TRAIN_INITIALIZATION": "pretrained",
        "PHYS_TRAIN_TASK": task,
        "PHYS_TRAIN_BASE_RUN": source["run_name"],
        "PHYS_TRAIN_BASE_CKPT": str(source["checkpoint_iteration"]),
        "PHYS_TRAIN_UPDATES": str(updates),
        "PHYS_TRAIN_NUM_ENVS": str(args.flat_envs),
        "PHYS_TRAIN_SEED": str(args.seed),
        "PHYS_TRAIN_TAG": (
            f"nestedbo_{task}_stage{stage}_"
            f"s{args.seed}_u{updates}"
        ),
        "PHYS_TRAIN_RANDOMIZE_DESIGN": "1",
        "PHYS_TRAIN_HARDWARE_CONTEXT": "1",
        "PHYS_TRAIN_ANCHOR_FRACTION": str(
            config["anchor_fraction"]
        ),
        "PHYS_TRAIN_REFERENCE_POLICY_COEF": str(
            config["reference_coef"][stage - 1]
        ),
        "PHYS_TRAIN_REFERENCE_ANCHOR_TOL": "1e-6",
        "PHYS_TRAIN_MORPH_BUCKETS": str(args.morphology_buckets),
        "PHYS_TRAIN_LEARNING_RATE": str(
            config["learning_rates"][stage - 1]
        ),
        "PHYS_TRAIN_SCHEDULE": "fixed",
        "PHYS_TRAIN_ENTROPY_COEF": str(
            config["entropy"][stage - 1]
        ),
        "PHYS_TRAIN_EST_LEARNING_RATE": "0.0",
        "PHYS_TRAIN_SAVE_INTERVAL": str(args.save_interval),
        "PHYS_TRAIN_SAVE_INITIAL": "1",
        "PHYS_TRAIN_RESULT_JSON": str(result_path),
        "PHYS_MOTOR_MASS_EXPONENT": "1.0",
        "PHYS_MOTOR_NOMINAL_MASSES_KG": ",".join(
            str(value) for value in MOTOR_MASSES
        ),
    }
    for name, bounds in ranges.items():
        environment[
            f"PHYS_TRAIN_{name.upper()}_RANGE"
        ] = ",".join(str(value) for value in bounds)
    environment.update(config["extra"])
    return environment


def jump_environment(args, stage, source, result_path, updates):
    config = TASK_CONFIGS["jump"]
    ranges = stage_ranges(stage)
    environment = {
        "PHYS_JUMP_INITIALIZATION": "pretrained",
        "PHYS_JUMP_BASE_RUN": source["run_name"],
        "PHYS_JUMP_BASE_CKPT": str(source["checkpoint_iteration"]),
        "PHYS_JUMP_UPDATES": str(updates),
        "PHYS_JUMP_NUM_ENVS": str(args.jump_envs),
        "PHYS_JUMP_SEED": str(args.seed),
        "PHYS_JUMP_TAG": (
            f"nestedbo_jump_stage{stage}_"
            f"s{args.seed}_u{updates}"
        ),
        "PHYS_JUMP_RANDOMIZE_DESIGN": "1",
        "PHYS_JUMP_HARDWARE_CONTEXT": "1",
        "PHYS_JUMP_ANCHOR_FRACTION": str(
            config["anchor_fraction"]
        ),
        "PHYS_JUMP_REFERENCE_POLICY_COEF": str(
            config["reference_coef"][stage - 1]
        ),
        "PHYS_JUMP_REFERENCE_ANCHOR_TOL": "1e-6",
        "PHYS_JUMP_MORPH_BUCKETS": str(args.morphology_buckets),
        "PHYS_JUMP_LEARNING_RATE": str(
            config["learning_rates"][stage - 1]
        ),
        "PHYS_JUMP_SCHEDULE": "fixed",
        "PHYS_JUMP_ENTROPY_COEF": str(
            config["entropy"][stage - 1]
        ),
        "PHYS_JUMP_EST_LEARNING_RATE": "0.0",
        "PHYS_JUMP_SAVE_INTERVAL": str(args.save_interval),
        "PHYS_JUMP_SAVE_INITIAL": "1",
        "PHYS_JUMP_RESULT_JSON": str(result_path),
        "PHYS_MOTOR_MASS_EXPONENT": "1.0",
        "PHYS_MOTOR_NOMINAL_MASSES_KG": ",".join(
            str(value) for value in MOTOR_MASSES
        ),
    }
    for name, bounds in ranges.items():
        environment[
            f"PHYS_JUMP_{name.upper()}_RANGE"
        ] = ",".join(str(value) for value in bounds)
    environment.update(config["extra"])
    return environment


class Driver:
    def __init__(self, args):
        self.args = args
        self.root = args.root
        self.lock = threading.Lock()
        self.running = {}
        self.completed = []
        self.failures = []
        self.stop_monitor = threading.Event()
        self.started_at = time.time()

    def write_status(self):
        with self.lock:
            seconds_per_update = 1.75
            total_stage_units = 2.0 * len(self.args.tasks)
            completed_stage_units = float(len(self.completed))
            running_progress = 0.0
            for record in self.running.values():
                detail = record.get("detail") or {}
                started_epoch = detail.get("started_epoch")
                remaining_updates = detail.get(
                    "remaining_updates", self.args.stage_updates
                )
                if started_epoch is not None and remaining_updates > 0:
                    running_progress += min(
                        (time.time() - started_epoch)
                        / (remaining_updates * seconds_per_update),
                        0.95,
                    )
            remaining_stage_units = max(
                total_stage_units
                - completed_stage_units
                - running_progress,
                0.0,
            )
            eta_s = (
                remaining_stage_units
                * self.args.stage_updates
                * seconds_per_update
                / max(len(self.args.gpu_ids), 1)
            )
            status = {
                "protocol": "staged_2d_to_6d_generalists_v1",
                "started_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z",
                    time.localtime(self.started_at),
                ),
                "updated_at": now(),
                "stage_updates": self.args.stage_updates,
                "running": dict(self.running),
                "completed": list(self.completed),
                "failures": list(self.failures),
                "elapsed_s": time.time() - self.started_at,
                "eta_model_seconds_per_update": seconds_per_update,
                "estimated_remaining_s": eta_s,
                "estimated_completion_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z",
                    time.localtime(time.time() + eta_s),
                ),
            }
        atomic_json(self.root / "STATUS.json", status)

    def monitor(self):
        gpu_log = self.root / "gpu_monitor.csv"
        write_header = not gpu_log.exists()
        while not self.stop_monitor.is_set():
            rows = []
            try:
                process = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,memory.total,"
                        "utilization.gpu,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                for line in process.stdout.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if len(values) == 5:
                        rows.append(values)
            except OSError:
                rows = []
            if rows:
                with gpu_log.open("a", encoding="ascii", newline="") as stream:
                    writer = csv.writer(stream)
                    if write_header:
                        writer.writerow(
                            [
                                "timestamp",
                                "gpu",
                                "memory_used_mib",
                                "memory_total_mib",
                                "utilization_percent",
                                "temperature_c",
                            ]
                        )
                        write_header = False
                    for row in rows:
                        writer.writerow([now(), *row])
            self.write_status()
            self.stop_monitor.wait(self.args.monitor_interval)

    def set_running(self, gpu_id, task, stage, detail=None):
        with self.lock:
            key = str(gpu_id)
            if task is None:
                self.running.pop(key, None)
            else:
                self.running[key] = {
                    "task": task,
                    "stage": stage,
                    "detail": detail,
                    "updated_at": now(),
                }
        self.write_status()

    def recover_or_prepare_source(
        self,
        task,
        stage,
        source,
        stage_dir,
        target_iteration,
    ):
        recovered = prior_attempt_resume(stage_dir, target_iteration)
        if recovered is None:
            return source, self.args.stage_updates
        iteration, log_dir, checkpoint = recovered
        if iteration >= target_iteration:
            result = {
                "protocol": "staged_2d_to_6d_recovered_checkpoint_v1",
                "run_name": log_dir.name,
                "log_dir": str(log_dir),
                "checkpoint": str(checkpoint),
                "base_run": source["run_name"],
                "base_checkpoint": source["checkpoint_iteration"],
                "final_iteration": iteration,
                "updates": iteration - source["checkpoint_iteration"],
                "task": task,
                "stage": stage,
                "recovered_at": now(),
            }
            atomic_json(stage_dir / "result.json", result)
            (stage_dir / "DONE").write_text(now() + "\n", encoding="ascii")
            return None, 0
        return (
            {
                "run_name": log_dir.name,
                "checkpoint_iteration": iteration,
            },
            target_iteration - iteration,
        )

    def run_stage(self, gpu_id, task, stage, source):
        stage_dir = self.root / task / f"stage{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        result_path = stage_dir / "result.json"
        if valid_result(result_path):
            return read_json(result_path), "cached"

        target_iteration = (
            int(source["checkpoint_iteration"]) + self.args.stage_updates
        )
        resume_source, remaining = self.recover_or_prepare_source(
            task, stage, source, stage_dir, target_iteration
        )
        if resume_source is None:
            return read_json(result_path), "recovered"

        attempt = len(list(stage_dir.glob("console_attempt_*.log"))) + 1
        console = stage_dir / f"console_attempt_{attempt:02d}.log"
        config = TASK_CONFIGS[task]
        if config["kind"] == "jump":
            environment = jump_environment(
                self.args,
                stage,
                resume_source,
                result_path,
                remaining,
            )
            script = "legged_gym/scripts/train_physical_jump.py"
        else:
            environment = flat_environment(
                self.args,
                task,
                stage,
                resume_source,
                result_path,
                remaining,
            )
            script = "legged_gym/scripts/train_physical_capability.py"

        command = [
            str(self.args.python),
            script,
            "--headless",
            "--sim_device=cuda:0",
            "--rl_device=cuda:0",
        ]
        atomic_json(
            stage_dir / f"attempt_{attempt:02d}.json",
            {
                "task": task,
                "stage": stage,
                "gpu_id": gpu_id,
                "started_at": now(),
                "source": resume_source,
                "original_source": source,
                "target_iteration": target_iteration,
                "remaining_updates": remaining,
                "command": command,
                "environment": environment,
            },
        )
        if self.args.dry_run:
            result = {
                "run_name": f"dry_run_{task}_stage{stage}",
                "final_iteration": target_iteration,
                "checkpoint": str(stage_dir / "dry_run_model.pt"),
                "dry_run": True,
            }
            atomic_json(result_path, result)
            return result, "dry-run"

        process_environment = os.environ.copy()
        process_environment.update(environment)
        process_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "ROBOT_TYPE": "WF_TRON1A",
                "PLAY_EXPORT_POLICY": "0",
                "OMP_NUM_THREADS": "1",
                "PATH": (
                    f"{self.args.python.parent}:"
                    f"{process_environment.get('PATH', '')}"
                ),
            }
        )
        self.set_running(
            gpu_id,
            task,
            stage,
            {
                "attempt": attempt,
                "remaining_updates": remaining,
                "target_iteration": target_iteration,
                "console": str(console),
                "started_epoch": time.time(),
            },
        )
        started = time.time()
        with console.open("w", encoding="utf-8") as log_stream:
            process = subprocess.run(
                command,
                cwd=str(self.args.repo),
                env=process_environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed_s = time.time() - started
        (stage_dir / f"EXIT_CODE_attempt_{attempt:02d}").write_text(
            f"{process.returncode}\n", encoding="ascii"
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"{task} stage {stage} exited {process.returncode}; "
                f"see {console}"
            )
        if not valid_result(result_path):
            raise RuntimeError(
                f"{task} stage {stage} did not create a valid result"
            )
        result = read_json(result_path)
        if int(result["final_iteration"]) != target_iteration:
            raise RuntimeError(
                f"{task} stage {stage} ended at "
                f"{result['final_iteration']}, expected {target_iteration}"
            )
        (stage_dir / "DONE").write_text(now() + "\n", encoding="ascii")
        atomic_json(
            stage_dir / "completion.json",
            {
                "task": task,
                "stage": stage,
                "gpu_id": gpu_id,
                "elapsed_s": elapsed_s,
                "completed_at": now(),
                "result": result,
            },
        )
        return result, "ran"

    def run_task(self, gpu_id, task):
        source = base_source(task)
        outcomes = []
        for stage in (1, 2):
            result, outcome = self.run_stage(
                gpu_id, task, stage, source
            )
            outcomes.append(outcome)
            source = source_from_result(result)
            with self.lock:
                self.completed.append(
                    {
                        "task": task,
                        "stage": stage,
                        "outcome": outcome,
                        "final_iteration": result["final_iteration"],
                        "completed_at": now(),
                    }
                )
            self.write_status()
        atomic_json(
            self.root / task / "selected_policy.json",
            {
                "task": task,
                "selection": "stage2_final",
                "cumulative_randomized_updates": (
                    2 * self.args.stage_updates
                ),
                "stage1_randomized_dimensions": [
                    "thigh",
                    "shank",
                ],
                "stage2_randomized_dimensions": [
                    "thigh",
                    "shank",
                    "abad",
                    "hip",
                    "knee",
                    "wheel",
                ],
                "result": result,
                "outcomes": outcomes,
                "selected_at": now(),
            },
        )

    def worker(self, gpu_id, pending):
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                self.set_running(gpu_id, None, None)
                return
            try:
                self.run_task(gpu_id, task)
                print(
                    f"[generalists] gpu={gpu_id} task={task} complete",
                    flush=True,
                )
            except Exception as error:
                message = f"{task}: {error}"
                with self.lock:
                    self.failures.append(message)
                print(f"[generalists] FAILED {message}", flush=True)
            finally:
                self.set_running(gpu_id, None, None)
                pending.task_done()

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.root / "PROTOCOL.json",
            {
                "protocol": "staged_2d_to_6d_generalists_v1",
                "created_at": now(),
                "repo": str(self.args.repo),
                "python": str(self.args.python),
                "tasks": list(self.args.tasks),
                "gpu_ids": list(self.args.gpu_ids),
                "seed": self.args.seed,
                "stage_updates": self.args.stage_updates,
                "flat_base": {
                    "run": FLAT_BASE_RUN,
                    "checkpoint": FLAT_BASE_CHECKPOINT,
                },
                "jump_base": {
                    "run": JUMP_BASE_RUN,
                    "checkpoint": JUMP_BASE_CHECKPOINT,
                    "reason": (
                        "jump task has a distinct control contract and cannot "
                        "load the flat model_13000 checkpoint"
                    ),
                },
                "stage1_ranges": stage_ranges(1),
                "stage2_ranges": stage_ranges(2),
                "task_configs": TASK_CONFIGS,
                "old_results_are_read_only": True,
            },
        )
        pending = queue.Queue()
        for task in self.args.tasks:
            pending.put(task)
        monitor = threading.Thread(target=self.monitor, daemon=False)
        monitor.start()
        workers = [
            threading.Thread(
                target=self.worker,
                args=(gpu_id, pending),
                daemon=False,
            )
            for gpu_id in self.args.gpu_ids
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.stop_monitor.set()
        monitor.join()
        self.write_status()
        if self.failures:
            raise RuntimeError(
                f"{len(self.failures)} task pipeline(s) failed: "
                + "; ".join(self.failures)
            )
        selected = {}
        for task in self.args.tasks:
            selected[task] = read_json(
                self.root / task / "selected_policy.json"
            )
        atomic_json(
            self.root / "RESULTS.json",
            {
                "protocol": "staged_2d_to_6d_generalists_v1",
                "completed_at": now(),
                "selected": selected,
            },
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--tasks", default="clearance,payload,speed,jump")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--stage-updates", type=int, default=7000)
    parser.add_argument("--flat-envs", type=int, default=4096)
    parser.add_argument("--jump-envs", type=int, default=4096)
    parser.add_argument("--morphology-buckets", type=int, default=64)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--monitor-interval", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.gpu_ids = parse_csv(args.gpu_ids, int)
    args.tasks = parse_csv(args.tasks, str)
    if not args.gpu_ids:
        parser.error("--gpu-ids must not be empty")
    if not args.tasks or any(task not in TASKS for task in args.tasks):
        parser.error(f"--tasks must contain only {','.join(TASKS)}")
    if args.stage_updates < 2000:
        parser.error("--stage-updates must be at least 2000")
    if args.seed == 1:
        parser.error("--seed=1 is nondeterministic in this codebase")
    if args.flat_envs % 8 or args.jump_envs < 1:
        parser.error("flat env count must be divisible by 8")
    if args.save_interval < 1:
        parser.error("--save-interval must be positive")
    return args


if __name__ == "__main__":
    Driver(parse_args()).run()
