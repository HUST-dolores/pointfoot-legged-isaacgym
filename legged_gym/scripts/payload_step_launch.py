#!/usr/bin/env python3
"""Run two sequential GPU queues; preserve every process log and exit status."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from legged_gym.scripts.payload_step_protocol import POLICIES, SEEDS, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "formal"), required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    logs = args.output / "process_logs"
    logs.mkdir(parents=True, exist_ok=True)
    steps = 300 if args.phase == "preflight" else 2000
    seeds = (12345,) if args.phase == "preflight" else SEEDS
    tasks = [(policy, seed, condition) for pindex, policy in enumerate(POLICIES)
             for sindex, seed in enumerate(seeds)
             for condition in (("empty", "free_box") if (pindex + sindex) % 2 == 0
                               else ("free_box", "empty"))]
    start = time.monotonic()
    lock = threading.Lock()
    results = []
    status = {"phase": args.phase, "started": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "total_batches": len(tasks), "status": "running", "processes": results}

    def worker(gpu, queue):
        for policy, seed, condition in queue:
            batch = f"{policy}_eval{seed}_{condition}"
            target = args.output / ("preflight" if steps < 2000 else "batches") / batch / "result.json"
            log = logs / f"{args.phase}_{batch}.log"
            before = time.monotonic()
            command = [sys.executable, "-m", "legged_gym.scripts.payload_step_experiment",
                       "--output", str(args.output), "--policy", policy, "--seed", str(seed),
                       "--condition", condition, "--steps", str(steps), "--gpu", "0"]
            if target.exists():
                result = {"batch_id": batch, "gpu": gpu, "exit_code": 0, "reused_complete": True,
                          "elapsed_s": 0.0, "log": str(log)}
            else:
                with log.open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"command": command, "parent_pid": os.getpid()}) + "\n")
                    handle.flush()
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                    completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                               env=environment, check=False, timeout=1800)
                result = {"batch_id": batch, "gpu": gpu, "exit_code": completed.returncode,
                          "reused_complete": False, "elapsed_s": time.monotonic() - before,
                          "log": str(log)}
            with lock:
                results.append(result)
                status["completed_batches"] = sum(r["exit_code"] == 0 for r in results)
                status["elapsed_s"] = time.monotonic() - start
                active_count = max(1, sum(not r["reused_complete"] for r in results))
                status["rough_remaining_s"] = ((len(tasks) - len(results)) *
                    sum(r["elapsed_s"] for r in results) / active_count / 2)
                write_json(args.output / f"{args.phase}_status.json", status)
                print(f"[{args.phase}] {len(results)}/{len(tasks)} {batch} "
                      f"exit={result['exit_code']} wall={result['elapsed_s']:.1f}s "
                      f"remaining~{status['rough_remaining_s']:.0f}s", flush=True)
            if result["exit_code"]:
                raise RuntimeError(f"batch failed: {batch}; inspect {log}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu, tasks[gpu::2]) for gpu in (0, 1)]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
    status["status"] = "failed" if errors else "completed"
    status["errors"] = errors
    status["elapsed_s"] = time.monotonic() - start
    write_json(args.output / f"{args.phase}_status.json", status)
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
