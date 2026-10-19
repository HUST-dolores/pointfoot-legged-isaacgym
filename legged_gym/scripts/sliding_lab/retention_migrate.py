"""Migrate only explicitly identified retention jobs at complete checkpoints."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .geometry import write_json, sha256


def main(a):
    root = Path(__file__).resolve().parents[3]
    data = root / "logs/payload_retention_20261018"
    records = []
    started = time.monotonic()
    for seed, pid, iteration in ((303, a.pid303, 1250), (404, a.pid404, 1000)):
        old = data / f"train_residual_s{seed}"
        checkpoint = old / f"model_{iteration}.pt"
        new = data / f"train_residual_gpu1_s{seed}"
        if new.exists():
            raise FileExistsError(new)
        while not checkpoint.is_file():
            if time.monotonic() - started > 900:
                raise TimeoutError("Expected migration checkpoint not produced")
            time.sleep(.5)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        assert "legged_gym.scripts.sliding_lab.retention_train" in cmdline
        assert "--output" in cmdline and cmdline[cmdline.index("--output") + 1] == str(old)
        assert Path(f"/proc/{pid}").stat().st_uid == os.getuid()
        status = json.loads((old / "status.json").read_text())
        os.kill(pid, signal.SIGINT)
        stop_start = time.monotonic()
        while Path(f"/proc/{pid}").exists():
            state = Path(f"/proc/{pid}/status").read_text()
            if "State:\tZ" in state:
                break
            if time.monotonic() - stop_start > 90:
                raise TimeoutError("Old job did not finish cleanup")
            time.sleep(.5)
        record = {"seed": seed, "old_pid": pid, "old_directory": str(old), "new_directory": str(new),
            "resume_checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "resume_iteration": iteration, "last_reported_iteration_before_interrupt": status["iteration"],
            "old_elapsed_s_last_report": status["elapsed_s"], "reason": "GPU0 shared with three new ship-motion jobs; no other job stopped",
            "time": datetime.now().isoformat(), "old_process_exited": True}
        write_json(old / "STOPPED.json", record)
        config = json.loads((old / "metadata.json").read_text())["argv"]
        config.update(output=str(new), resume=str(checkpoint), architecture="residual")
        args = [sys.executable, "-u", "-m", "legged_gym.scripts.sliding_lab.retention_train"]
        for key, value in config.items():
            if value is not None:
                args.extend(("--" + key.replace("_", "-"), str(value)))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
        log = data / (new.name + ".log")
        with log.open("wb") as out:
            child = subprocess.Popen(args, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(new_pid=child.pid, stdout_log=str(log))
        records.append(record)
        write_json(data / "migration.json", {"jobs": records, "jobs_started_not_finished": True})
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid303", type=int, required=True)
    p.add_argument("--pid404", type=int, required=True)
    main(p.parse_args())
