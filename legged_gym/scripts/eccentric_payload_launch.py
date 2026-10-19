#!/usr/bin/env python3
"""Two isolated GPU queues for the preregistered static-load matrix."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from legged_gym.scripts.eccentric_payload_protocol import POLICIES, MASSES, PLACEMENTS, SEEDS, batch_id, write_json
from legged_gym.scripts.payload_step_experiment import timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "formal"), required=True)
    a = parser.parse_args()
    a.output = a.output.resolve()
    logs = a.output / "process_logs"
    logs.mkdir(parents=True, exist_ok=True)
    seeds = (12345,) if a.phase == "preflight" else SEEDS
    masses = (8, 26) if a.phase == "preflight" else MASSES
    steps = 300 if a.phase == "preflight" else 2000
    tasks = [(p,m,c,s) for m in masses for c in PLACEMENTS for s in seeds for p in POLICIES]
    status = {"started": timestamp(), "phase": a.phase, "total_batches": len(tasks),
              "completed_batches": 0, "status": "running", "processes": []}
    start, lock = time.monotonic(), threading.Lock()

    def worker(gpu):
        for policy, mass, placement, seed in tasks[gpu::2]:
            batch = batch_id(policy, mass, placement, seed)
            log = logs / f"{a.phase}_{batch}.log"
            target = a.output / ("preflight" if steps < 2000 else "batches") / batch / "result.json"
            before = time.monotonic()
            command = [sys.executable, "-u", "-m", "legged_gym.scripts.eccentric_payload_experiment",
                "--output", str(a.output), "--policy", policy, "--mass", str(mass),
                "--placement", placement, "--seed", str(seed), "--steps", str(steps)]
            if target.exists():
                code, reused = 0, True
            else:
                with log.open("w") as handle:
                    result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), timeout=1800)
                code, reused = result.returncode, False
            with lock:
                status["processes"].append({"batch_id": batch, "gpu": gpu, "exit_code": code,
                    "elapsed_s": time.monotonic()-before, "reused_complete": reused, "log": str(log)})
                status["completed_batches"] = sum(r["exit_code"] == 0 for r in status["processes"])
                status["elapsed_s"] = time.monotonic()-start
                done = len(status["processes"])
                status["rough_remaining_s"] = (len(tasks)-done)*sum(r["elapsed_s"] for r in status["processes"])/done/2
                write_json(a.output / f"{a.phase}_status.json", status)
                print(f"{a.phase} {done}/{len(tasks)} {batch} exit={code} remaining~{status['rough_remaining_s']:.0f}s", flush=True)
            if code:
                raise RuntimeError(f"failed {batch}; {log}")

    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in (0,1)]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
    status.update(status="failed" if errors else "completed", errors=errors,
                  finished=timestamp(), elapsed_s=time.monotonic()-start)
    write_json(a.output / f"{a.phase}_status.json", status)
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    main()
