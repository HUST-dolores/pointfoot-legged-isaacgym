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
    p.add_argument("--scope",choices=("tail12","mid3"),default="tail12")
    a=p.parse_args(); a.output=a.output.resolve()
    start=time.monotonic()
    status={"started":timestamp(),"status":"running","processes":[],
            "scope":"only unstarted Model-guided mass 23/26 kg queue tail; no protocol changes"}
    tasks=([(18,"eccentric",s) for s in reversed(SEEDS)] if a.scope=="mid3" else
           [(m,c,s) for m in (26,23) for c in reversed(tuple(PLACEMENTS)) for s in reversed(SEEDS)])
    status["scope"]=a.scope
    status_file=a.output / ("assist_status.json" if a.scope=="tail12" else "assist_mid3_status.json")
    for mass,placement,seed in tasks:
        primary=json.loads((a.output / "formal_status.json").read_text())
        # Stop when the original queue reaches the 18 kg center cases.
        # mid3 targets only the later eccentric cases; each helper call is bounded.
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
        log=a.output / "process_logs" / f"assist_{a.scope}_{batch}.log"
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
        write_json(status_file,status)
        print(f"assist {a.scope} {len(status['processes'])}/{len(tasks)} {batch} exit={code}",flush=True)
        if code:
            status.update(status="failed",finished=timestamp())
            write_json(status_file,status)
            raise RuntimeError(f"assist failed: {batch}; primary retains its original queue")
    status.update(status="completed",finished=timestamp(),elapsed_s=time.monotonic()-start)
    write_json(status_file,status)


if __name__=="__main__":
    main()
