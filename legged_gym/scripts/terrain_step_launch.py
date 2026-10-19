#!/usr/bin/env python3
"""Two-GPU evaluation queues with graphics restricted to physical GPU0."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from legged_gym.scripts.payload_step_experiment import timestamp
from legged_gym.scripts.terrain_step_protocol import POLICIES, TERRAINS, SEEDS, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "formal"), required=True)
    parser.add_argument("--capture-main", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    logs = root / "process_logs"
    logs.mkdir(parents=True, exist_ok=True)
    steps = 150 if args.phase == "preflight" else 300
    folder = "preflight" if args.phase == "preflight" else "batches"
    seeds = (12345,) if args.phase == "preflight" else SEEDS
    tasks = [(p, t, s) for s in seeds for t in TERRAINS for p in POLICIES]
    queues = [[], []]
    if args.capture_main:
        graphical = [task for task in tasks if task[2] == 12345]
        others = [task for task in tasks if task[2] != 12345]
        queues[0] = graphical + others[:2]
        queues[1] = others[2:]
    else:
        queues = [tasks[0::2], tasks[1::2]]
    start = time.monotonic()
    lock = threading.Lock()
    status = {"phase": args.phase, "started": timestamp(), "status": "running",
              "total_batches": len(tasks), "processes": []}

    def worker(gpu, queue):
        for policy, terrain, seed in queue:
            batch = f"{policy}_{terrain}_eval{seed}"
            result_path = root / folder / batch / "result.json"
            capture = args.capture_main and seed == 12345
            command = [sys.executable, "-u", "-m", "legged_gym.scripts.terrain_step_experiment",
                       "--output", str(root), "--policy", policy, "--terrain", terrain,
                       "--seed", str(seed), "--steps", str(steps), "--phase", folder]
            if capture:
                command.extend(("--capture", "--camera-clean-exit"))
            log = logs / f"{args.phase}_{batch}.log"
            before = time.monotonic()
            reused = False
            if result_path.exists():
                old = json.loads(result_path.read_text())
                reused = old.get("cleanup_completed", False) and old.get("capture_enabled") == capture
                if not reused:
                    raise RuntimeError(f"incomplete or incompatible existing batch: {batch}")
                code = 0
            else:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json")
                with log.open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"command": command, "physical_gpu": gpu}) + "\n")
                    handle.flush()
                    code = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                          timeout=600, check=False).returncode
            entry = {"batch_id": batch, "gpu": gpu, "exit_code": code, "log": str(log),
                     "elapsed_s": time.monotonic()-before, "reused_complete": reused}
            with lock:
                status["processes"].append(entry)
                status["completed_batches"] = sum(r["exit_code"] == 0 for r in status["processes"])
                status["elapsed_s"] = time.monotonic()-start
                write_json(root / f"{args.phase}_status.json", status)
                print(f"[{args.phase}] {len(status['processes'])}/{len(tasks)} {batch} "
                      f"exit={code} wall={entry['elapsed_s']:.1f}s", flush=True)
            if code:
                raise RuntimeError(f"failed {batch}: {log}")

    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, index, queue) for index, queue in enumerate(queues)]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
    status.update(status="failed" if errors else "completed", errors=errors,
                  elapsed_s=time.monotonic()-start, finished=timestamp())
    write_json(root / f"{args.phase}_status.json", status)
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
