#!/usr/bin/env python3
"""Run the jump scratch/zero-shot/fine-tuning ablation on two GPUs."""

import argparse
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np


DEFAULT_REPO = Path("/home/ps/mydrive/xu/pointfoot-legged-gym")
DEFAULT_PYTHON = Path("/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python")
DEFAULT_ROOT = Path(
    "/home/ps/mydrive/xu/codesign_runs/"
    "jump_pretrain_finetune_ablation_2026-07-30"
)
SOURCE_RUN = "Jul30_09-02-13_physjump_jumpgenv2_stageb_full"
SOURCE_CHECKPOINT = 4100
DESIGN = {
    "thigh": 1.034134305,
    "shank": 1.004982102,
    "abad": 1.498098486,
    "hip": 0.817097429,
    "knee": 1.578590983,
    "wheel": 1.556014975,
}
MOTOR_MASSES = (1.35, 1.35, 0.89, 0.365)


def parse_csv(value, converter):
    return tuple(
        converter(item.strip())
        for item in value.split(",")
        if item.strip()
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
    temporary.replace(path)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


class Job:
    def __init__(self, name, command, environment, directory, result):
        self.name = name
        self.command = list(command)
        self.environment = dict(environment)
        self.directory = Path(directory)
        self.result = Path(result)
        self.elapsed_s = None


def execute_job(job, gpu_id, repo):
    done_path = job.directory / "DONE"
    exit_path = job.directory / "EXIT_CODE"
    if done_path.exists() and job.result.exists():
        return "cached"

    job.directory.mkdir(parents=True, exist_ok=True)
    atomic_json(
        job.directory / "STARTED.json",
        {
            "name": job.name,
            "gpu_id": gpu_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": job.command,
        },
    )
    environment = os.environ.copy()
    environment.update(job.environment)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "ROBOT_TYPE": "WF_TRON1A",
            "PLAY_EXPORT_POLICY": "0",
            "OMP_NUM_THREADS": "1",
            "PATH": (
                f"{Path(job.command[0]).parent}:"
                f"{environment.get('PATH', '')}"
            ),
        }
    )
    started = time.time()
    with (job.directory / "console.log").open(
        "w", encoding="utf-8"
    ) as log_stream:
        process = subprocess.run(
            job.command,
            cwd=str(repo),
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    job.elapsed_s = time.time() - started
    exit_path.write_text(f"{process.returncode}\n", encoding="ascii")
    if process.returncode != 0:
        raise RuntimeError(
            f"{job.name} exited {process.returncode}; "
            f"see {job.directory / 'console.log'}"
        )
    if not job.result.exists():
        raise RuntimeError(
            f"{job.name} did not create {job.result}"
        )
    done_path.write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n",
        encoding="ascii",
    )
    return "ran"


def run_jobs(stage, jobs, gpu_ids, repo, root):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    lock = threading.Lock()
    failures = []
    durations = []
    completed = 0

    def update_status():
        mean_s = float(np.mean(durations)) if durations else None
        remaining = len(jobs) - completed
        eta_s = (
            remaining * mean_s / max(len(gpu_ids), 1)
            if mean_s is not None
            else None
        )
        atomic_json(
            root / "STATUS.json",
            {
                "stage": stage,
                "completed": completed,
                "total": len(jobs),
                "failures": list(failures),
                "mean_job_s": mean_s,
                "eta_s": eta_s,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )

    def worker(gpu_id):
        nonlocal completed
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                outcome = execute_job(job, gpu_id, repo)
                print(
                    f"[{stage}] gpu={gpu_id} job={job.name} "
                    f"outcome={outcome} elapsed_s={job.elapsed_s}",
                    flush=True,
                )
                if job.elapsed_s is not None:
                    with lock:
                        durations.append(job.elapsed_s)
            except Exception as error:
                with lock:
                    failures.append(f"{job.name}: {error}")
                print(f"[{stage}] FAILED {job.name}: {error}", flush=True)
            finally:
                with lock:
                    completed += 1
                    update_status()
                pending.task_done()

    update_status()
    threads = [
        threading.Thread(target=worker, args=(gpu_id,), daemon=False)
        for gpu_id in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(
            f"{stage} had {len(failures)} failure(s): "
            + "; ".join(failures)
        )


def training_environment(args, mode, seed, result):
    environment = {
        "PHYS_JUMP_INITIALIZATION": (
            "pretrained" if mode == "fine_tune" else "scratch"
        ),
        "PHYS_JUMP_BASE_RUN": SOURCE_RUN,
        "PHYS_JUMP_BASE_CKPT": str(SOURCE_CHECKPOINT),
        "PHYS_JUMP_UPDATES": str(args.updates),
        "PHYS_JUMP_NUM_ENVS": str(args.train_envs),
        "PHYS_JUMP_SEED": str(seed),
        "PHYS_JUMP_TAG": (
            f"pfa_jump_{mode}_s{seed}_u{args.updates}"
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
        "PHYS_JUMP_EST_LEARNING_RATE": (
            "0.0" if mode == "fine_tune" else "0.001"
        ),
        "PHYS_JUMP_RESUME_NOISE_STD": "0.05",
        "PHYS_JUMP_SAVE_INTERVAL": "25",
        "PHYS_JUMP_SAVE_INITIAL": "1",
        "PHYS_JUMP_INIT_RANDOM_EP_LEN": "1",
        "PHYS_JUMP_RESULT_JSON": str(result),
        "PHYS_MOTOR_MASS_EXPONENT": "1.0",
        "PHYS_MOTOR_NOMINAL_MASSES_KG": ",".join(
            str(value) for value in MOTOR_MASSES
        ),
    }
    for name, value in DESIGN.items():
        environment[f"PHYS_JUMP_{name.upper()}"] = str(value)
    return environment


def make_training_jobs(args, root):
    jobs = []
    for seed in args.seeds:
        for mode in ("fine_tune", "scratch"):
            directory = root / "training" / mode / str(seed)
            result = directory / "result.json"
            jobs.append(
                Job(
                    f"train-jump-{mode}-s{seed}",
                    [
                        str(args.python),
                        "legged_gym/scripts/train_physical_jump.py",
                        "--headless",
                        "--sim_device=cuda:0",
                        "--rl_device=cuda:0",
                    ],
                    training_environment(args, mode, seed, result),
                    directory,
                    result,
                )
            )
    return jobs


def evaluation_environment(run_name, checkpoint, seed, directory, tag):
    environment = {
        "JUMPFWD_RUN": run_name,
        "JUMPFWD_CKPT": str(checkpoint),
        "JUMPFWD_STEPS": "2400",
        "JUMPFWD_GRID_HEIGHTS": "0.10,0.11,0.12,0.13,0.14,0.15,0.16",
        "JUMPFWD_GRID_DISTANCES": "0.05,0.065,0.08,0.095,0.11,0.125,0.14",
        "JUMPFWD_GRID_REPLICAS": "16",
        "JUMPFWD_SEED": str(seed),
        "JUMPFWD_NOMINAL_EVAL": "1",
        "JUMPFWD_FLAT_EVAL": "1",
        "JUMPFWD_EVAL_FRICTION": "0.8",
        "JUMPFWD_EVAL_PREFIX": tag,
        "JUMPFWD_CSV": str(directory / "jumps.csv"),
        "JUMPFWD_SUMMARY": str(directory / "result.json"),
        "PHYS_MOTOR_MASS_EXPONENT": "1.0",
        "PHYS_MOTOR_NOMINAL_MASSES_KG": ",".join(
            str(value) for value in MOTOR_MASSES
        ),
    }
    environment["JUMPFWD_EVAL_THIGH"] = str(DESIGN["thigh"])
    environment["JUMPFWD_EVAL_SHANK"] = str(DESIGN["shank"])
    for group in ("abad", "hip", "knee", "wheel"):
        environment[f"JUMPFWD_EVAL_{group.upper()}"] = str(
            DESIGN[group]
        )
    return environment


def make_evaluation_jobs(args, root):
    jobs = []
    for seed in args.seeds:
        eval_seed = seed + 1000
        for mode in ("fine_tune", "scratch"):
            training_result = read_json(
                root / "training" / mode / str(seed) / "result.json"
            )
            run_name = training_result["run_name"]
            initial = SOURCE_CHECKPOINT if mode == "fine_tune" else 0
            for offset in args.checkpoints:
                checkpoint = initial + offset
                directory = (
                    root
                    / "evaluations"
                    / mode
                    / str(seed)
                    / f"u{offset:04d}"
                )
                jobs.append(
                    Job(
                        f"eval-jump-{mode}-s{seed}-u{offset}",
                        [
                            str(args.python),
                            "legged_gym/scripts/measure_jumpfwd.py",
                            "--headless",
                            "--sim_device=cuda:0",
                            "--rl_device=cuda:0",
                        ],
                        evaluation_environment(
                            run_name,
                            checkpoint,
                            eval_seed,
                            directory,
                            f"pfa_jump_{mode}_{seed}_{offset}",
                        ),
                        directory,
                        directory / "result.json",
                    )
                )
    return jobs


def aggregate(args, root):
    raw_rows = []
    for seed in args.seeds:
        for mode in ("fine_tune", "scratch"):
            training_result = read_json(
                root / "training" / mode / str(seed) / "result.json"
            )
            initial = SOURCE_CHECKPOINT if mode == "fine_tune" else 0
            for offset in args.checkpoints:
                result_path = (
                    root
                    / "evaluations"
                    / mode
                    / str(seed)
                    / f"u{offset:04d}"
                    / "result.json"
                )
                result = read_json(result_path)
                area = result["reachable_area"]
                raw_rows.append(
                    {
                        "task": "jump",
                        "condition": mode,
                        "seed": seed,
                        "eval_seed": seed + 1000,
                        "ppo_updates": offset,
                        "metric_value": area["metric_value_m2"],
                        "unit": "m^2",
                        "reliable_grid_count": area[
                            "reliable_grid_count"
                        ],
                        "physical_success_rate": result[
                            "physical_success_rate_per_resolved"
                        ],
                        "single_jump_rate": result[
                            "single_jump_success_rate_per_trigger"
                        ],
                        "height_mean_m": result["height_mean_single_m"],
                        "distance_mean_m": result[
                            "distance_mean_single_m"
                        ],
                        "run_name": training_result["run_name"],
                        "checkpoint": initial + offset,
                        "result_path": str(result_path),
                    }
                )

    zero_by_seed = {
        row["seed"]: row
        for row in raw_rows
        if row["condition"] == "fine_tune"
        and row["ppo_updates"] == 0
    }
    for seed in args.seeds:
        source = zero_by_seed[seed]
        for offset in args.checkpoints:
            raw_rows.append(
                {
                    **source,
                    "condition": "zero_shot",
                    "ppo_updates": offset,
                }
            )

    raw_fields = list(raw_rows[0].keys())
    with (root / "raw_curves.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows = []
    for condition in ("scratch", "zero_shot", "fine_tune"):
        for offset in args.checkpoints:
            values = np.asarray(
                [
                    row["metric_value"]
                    for row in raw_rows
                    if row["condition"] == condition
                    and row["ppo_updates"] == offset
                ],
                dtype=np.float64,
            )
            if len(values) != len(args.seeds):
                raise RuntimeError(
                    f"missing rows for {condition}/u{offset}"
                )
            summary_rows.append(
                {
                    "task": "jump",
                    "condition": condition,
                    "ppo_updates": offset,
                    "mean": float(values.mean()),
                    "std": (
                        float(values.std(ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "n": len(values),
                    "unit": "m^2",
                }
            )
    with (root / "summary_curves.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summary_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    atomic_json(
        root / "FINAL_DATA.json",
        {
            "protocol": "jump_pretrain_finetune_ablation_2026-07-30",
            "source": {
                "run": SOURCE_RUN,
                "checkpoint": SOURCE_CHECKPOINT,
            },
            "design": DESIGN,
            "seeds": list(args.seeds),
            "training_envs": args.train_envs,
            "ppo_updates": args.updates,
            "checkpoint_offsets": list(args.checkpoints),
            "actor_learning_rate": 2.0e-5,
            "fine_tune_estimator_learning_rate": 0.0,
            "scratch_estimator_learning_rate": 1.0e-3,
            "metric": "reliable jump area A_jump",
            "error_band": "sample mean +/- one sample standard deviation",
            "raw_rows": len(raw_rows),
            "summary_rows": len(summary_rows),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--seeds", default="20260801,20260802,20260803")
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--train-envs", type=int, default=2048)
    parser.add_argument("--checkpoints", default="0,25,50,100,200,400")
    parser.add_argument(
        "--stage",
        choices=("all", "train", "evaluate", "aggregate"),
        default="all",
    )
    args = parser.parse_args()
    args.gpu_ids = parse_csv(args.gpu_ids, int)
    args.seeds = parse_csv(args.seeds, int)
    args.checkpoints = parse_csv(args.checkpoints, int)
    if not args.gpu_ids or not args.seeds:
        raise ValueError("at least one GPU and seed are required")
    if args.checkpoints[0] != 0 or args.checkpoints[-1] != args.updates:
        raise ValueError("checkpoint offsets must span zero to --updates")
    if any(
        right <= left
        for left, right in zip(args.checkpoints, args.checkpoints[1:])
    ):
        raise ValueError("checkpoint offsets must be strictly increasing")

    args.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": str(args.repo),
        "python": str(args.python),
        "root": str(args.root),
        "source_run": SOURCE_RUN,
        "source_checkpoint": SOURCE_CHECKPOINT,
        "design": DESIGN,
        "gpu_ids": list(args.gpu_ids),
        "seeds": list(args.seeds),
        "updates": args.updates,
        "train_envs": args.train_envs,
        "checkpoints": list(args.checkpoints),
    }
    atomic_json(args.root / "RUN_MANIFEST.json", manifest)

    if args.stage in ("all", "train"):
        run_jobs(
            "training",
            make_training_jobs(args, args.root),
            args.gpu_ids,
            args.repo,
            args.root,
        )
    if args.stage in ("all", "evaluate"):
        run_jobs(
            "evaluation",
            make_evaluation_jobs(args, args.root),
            args.gpu_ids,
            args.repo,
            args.root,
        )
    if args.stage in ("all", "aggregate"):
        aggregate(args, args.root)
    atomic_json(
        args.root / "STATUS.json",
        {
            "stage": "complete",
            "completed": True,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    (args.root / "DONE").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
