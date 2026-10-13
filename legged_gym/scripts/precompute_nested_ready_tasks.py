#!/usr/bin/env python3
"""Precompute nested-BO artifacts for task policies that finish early.

This helper uses exactly the same paths, seeds, policies, checkpoint-selection
rule, and physical protocols as the main nested BO.  It lets an otherwise idle
GPU evaluate clearance, payload, and speed while the jump generalist is still
training.  The main driver later treats these artifacts as ordinary cached
jobs and only fills in the missing jump task.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

import run_nested_four_objective_codesign as nested


DEFAULT_TASKS = ("clearance", "payload", "speed")
PRECOMPUTE_ROOT = nested.PRECOMPUTE_ROOT


def wait_for_ready_policies(tasks, poll_seconds):
    while True:
        paths = {
            task: (
                nested.GENERALIST_ROOT
                / task
                / "selected_policy.json"
            )
            for task in tasks
        }
        if all(path.is_file() for path in paths.values()):
            return paths
        nested.atomic_json(
            PRECOMPUTE_ROOT / "WAITING.json",
            {
                "updated_at": nested.now(),
                "tasks": list(tasks),
                "ready": {
                    task: path.is_file()
                    for task, path in paths.items()
                },
            },
        )
        print(
            "[precompute] waiting for "
            + ",".join(
                task
                for task, path in paths.items()
                if not path.is_file()
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def final_policies(paths):
    policies = {}
    for task, path in paths.items():
        selected = nested.read_json(path)
        result = selected["result"]
        checkpoint = Path(result["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        policies[task] = {
            "run_name": result["run_name"],
            "checkpoint": int(result["final_iteration"]),
            "checkpoint_path": str(checkpoint),
            "training_result": result,
        }
    return policies


def select_checkpoints(tasks, policies):
    requests = []
    for task in tasks:
        policy = policies[task]
        stage2_base = int(
            policy["training_result"]["base_checkpoint"]
        )
        for offset in nested.GENERALIST_CHECKPOINT_OFFSETS:
            checkpoint = stage2_base + offset
            for design in nested.generalist_validation_designs():
                requests.append(
                    nested.make_eval_request(
                        task,
                        policy["run_name"],
                        checkpoint,
                        design,
                        20260806,
                        "generalist_checkpoint_selection",
                        "search",
                    )
                )
    nested.base.run_jobs(
        "precompute_generalist_checkpoint_selection",
        [request.job for request in requests],
    )
    records = []
    for request in requests:
        nested.validate_evaluation_protocol(request)
        records.append(nested.base.record_from_request(request))

    selected = {}
    validation = {}
    for task in tasks:
        policy = policies[task]
        environment = (
            "wheelfoot_jump_forward"
            if task == "jump"
            else "wheelfoot_flat"
        )
        run_dir = (
            nested.REPO
            / "logs"
            / environment
            / "WF_TRON1A"
            / policy["run_name"]
        )
        rows = []
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
            rows.append(
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
        best = max(
            rows,
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
                "same frozen-design physical checkpoint rule as main"
            ),
            "full_sixd_updates": int(best["full_sixd_updates"]),
        }
        validation[task] = rows
    nested.atomic_json(
        PRECOMPUTE_ROOT / "GENERALIST_SELECTION.json",
        {
            "selected": selected,
            "validation": validation,
            "record_count": len(records),
        },
    )
    return selected


def precompute_initial(tasks, policies):
    designs = nested.initial_designs()
    adaptations = []
    for design in designs:
        for task in tasks:
            adaptations.append(
                nested.make_adaptation_job(
                    task,
                    policies[task],
                    design,
                    nested.SEARCH_TRAIN_SEED,
                    "initial",
                )
            )
    nested.base.run_jobs(
        "precompute_initial_adaptation",
        [record["job"] for record in adaptations],
    )
    for record in adaptations:
        nested.validate_adaptation(record)

    validation_requests = []
    validation_index = {}
    for adaptation in adaptations:
        validation_seed = adaptation["train_seed"] + 1000
        for offset in nested.INNER_OFFSETS:
            checkpoint = (
                int(adaptation["policy"]["checkpoint"]) + offset
            )
            request = nested.make_eval_request(
                adaptation["task"],
                adaptation["result"]["run_name"],
                checkpoint,
                adaptation["design"],
                validation_seed,
                "initial_checkpoint_validation",
                "search",
            )
            validation_requests.append(request)
            validation_index[id(request)] = (
                adaptation,
                offset,
            )
    nested.base.run_jobs(
        "precompute_initial_checkpoint_validation",
        [request.job for request in validation_requests],
    )

    validation = {}
    for request in validation_requests:
        nested.validate_evaluation_protocol(request)
        record = nested.base.record_from_request(request)
        adaptation, offset = validation_index[id(request)]
        key = (
            adaptation["task"],
            adaptation["design_id"],
            adaptation["train_seed"],
        )
        validation.setdefault(key, []).append((offset, record))

    test_requests = []
    test_index = {}
    selections = {}
    for adaptation in adaptations:
        key = (
            adaptation["task"],
            adaptation["design_id"],
            adaptation["train_seed"],
        )
        offset, record = max(
            validation[key],
            key=lambda pair: (
                float(pair[1]["objective"]),
                -int(pair[0]),
            ),
        )
        checkpoint = (
            int(adaptation["policy"]["checkpoint"]) + int(offset)
        )
        selections["|".join(str(value) for value in key)] = {
            "checkpoint_offset": int(offset),
            "checkpoint": checkpoint,
            "validation_objective": float(record["objective"]),
        }
        request = nested.make_eval_request(
            adaptation["task"],
            adaptation["result"]["run_name"],
            checkpoint,
            adaptation["design"],
            adaptation["train_seed"] + 2000,
            "initial_selected_test",
            "search",
        )
        test_requests.append(request)
        test_index[id(request)] = adaptation
    nested.base.run_jobs(
        "precompute_initial_selected_test",
        [request.job for request in test_requests],
    )

    tests = []
    for request in test_requests:
        nested.validate_evaluation_protocol(request)
        record = nested.base.record_from_request(request)
        adaptation = test_index[id(request)]
        tests.append(
            {
                "task": adaptation["task"],
                "design_id": adaptation["design_id"],
                "train_seed": adaptation["train_seed"],
                "objective": record["objective"],
                "result_path": record["result_path"],
            }
        )
    nested.atomic_json(
        PRECOMPUTE_ROOT / "INITIAL_PRECOMPUTE_RESULTS.json",
        {
            "tasks": list(tasks),
            "design_count": len(designs),
            "adaptation_count": len(adaptations),
            "checkpoint_validation_count": len(
                validation_requests
            ),
            "selected_test_count": len(test_requests),
            "selections": selections,
            "tests": tests,
        },
    )


def run(args):
    PRECOMPUTE_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("DONE", "EXIT_CODE"):
        (PRECOMPUTE_ROOT / name).unlink(missing_ok=True)
    (PRECOMPUTE_ROOT / "STARTED").write_text(
        nested.now() + "\n", encoding="ascii"
    )
    paths = wait_for_ready_policies(args.tasks, args.poll_seconds)
    policies = final_policies(paths)
    selected = select_checkpoints(args.tasks, policies)
    precompute_initial(args.tasks, selected)
    (PRECOMPUTE_ROOT / "EXIT_CODE").write_text(
        "0\n", encoding="ascii"
    )
    (PRECOMPUTE_ROOT / "DONE").write_text(
        nested.now() + "\n", encoding="ascii"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", default=",".join(DEFAULT_TASKS)
    )
    parser.add_argument("--gpu-workers", default="0,0")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    args = parser.parse_args()
    args.tasks = tuple(
        value.strip()
        for value in args.tasks.split(",")
        if value.strip()
    )
    if not args.tasks:
        parser.error("--tasks must not be empty")
    if any(task not in nested.TASKS for task in args.tasks):
        parser.error("unknown task")
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    nested.base.GPU_IDS = [
        value.strip()
        for value in args.gpu_workers.split(",")
        if value.strip()
    ]
    if not nested.base.GPU_IDS:
        parser.error("--gpu-workers must not be empty")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        import traceback

        traceback.print_exc()
        PRECOMPUTE_ROOT.mkdir(parents=True, exist_ok=True)
        (PRECOMPUTE_ROOT / "EXIT_CODE").write_text(
            "1\n", encoding="ascii"
        )
        raise
