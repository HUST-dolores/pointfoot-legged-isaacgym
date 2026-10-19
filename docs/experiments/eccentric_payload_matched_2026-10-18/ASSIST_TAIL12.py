#!/usr/bin/env python3
"""Precompute the distant queue tail on spare GPU0 without changing trial definitions."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from legged_gym.scripts.eccentric_payload_protocol import PLACEMENTS, SEEDS, batch_id, write_json
from legged_gym.scripts.payload_step_experiment import timestamp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); a.output=a.output.resolve()
    start=time.monotonic()
    status={"started":timestamp(),"status":"running","processes":[],
            "scope":"only unstarted Model-guided mass 23/26 kg queue tail; no protocol changes"}
    tasks=[(m,c,s) for m in (26,23) for c in reversed(tuple(PLACEMENTS)) for s in reversed(SEEDS)]
    for mass,placement,seed in tasks:
        primary=json.loads((a.output / "formal_status.json").read_text())
        # Leave six full original batches between the helper and primary queue.
        # A helper batch times out at 120 s, well below this observed buffer (>400 s).
        if any(r["batch_id"].startswith("mg_m18_") and r["exit_code"]==0 for r in primary["processes"]):
            status["stopped_to_preserve_queue_separation"]=True
            break
        batch=batch_id("mg",mass,placement,seed)
        folder=a.output / "batches" / batch
        if (folder / "result.json").exists():
            continue
        before=time.monotonic()
        command=[sys.executable,"-u","-m","legged_gym.scripts.eccentric_payload_experiment",
            "--output",str(a.output),"--policy","mg","--mass",str(mass),"--placement",placement,"--seed",str(seed)]
        log=a.output / "process_logs" / f"assist_{batch}.log"
        try:
            with log.open("w") as handle:
                result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0"),timeout=120)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
        status["processes"].append({"batch_id":batch,"gpu":0,"exit_code":code,
            "elapsed_s":time.monotonic()-before,"log":str(log)})
        status["elapsed_s"]=time.monotonic()-start
        write_json(a.output / "assist_status.json",status)
        print(f"assist {len(status['processes'])}/12 {batch} exit={code}",flush=True)
        if code:
            status.update(status="failed",finished=timestamp())
            write_json(a.output / "assist_status.json",status)
            raise RuntimeError(f"assist failed: {batch}; primary retains its original queue")
    status.update(status="completed",finished=timestamp(),elapsed_s=time.monotonic()-start)
    write_json(a.output / "assist_status.json",status)


if __name__=="__main__":
    main()
