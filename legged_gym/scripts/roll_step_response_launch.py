#!/usr/bin/env python3
"""Run the three evaluation seeds on two isolated CUDA devices."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from legged_gym.scripts.roll_step_response_protocol import SEEDS,write_json
from legged_gym.scripts.payload_step_experiment import timestamp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--phase",choices=("preflight","formal"),required=True)
    a=p.parse_args(); a.output=a.output.resolve()
    logs=a.output/"process_logs"; logs.mkdir(parents=True,exist_ok=True)
    seeds=(12345,) if a.phase=="preflight" else SEEDS
    steps=1600 if a.phase=="preflight" else 2000
    start=time.monotonic(); lock=threading.Lock()
    status={"started":timestamp(),"status":"running","total_batches":len(seeds),"completed_batches":0,"processes":[]}
    def worker(gpu,queue):
        for seed in queue:
            command=[sys.executable,"-u","-m","legged_gym.scripts.roll_step_response_experiment",
                "--output",str(a.output),"--phase","preflight" if a.phase=="preflight" else "batches",
                "--seed",str(seed),"--steps",str(steps)]
            log=logs/f"{a.phase}_eval{seed}.log"; before=time.monotonic()
            with log.open("w") as handle:
                r=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),timeout=600)
            with lock:
                status["processes"].append({"evaluation_seed":seed,"gpu":gpu,"exit_code":r.returncode,
                    "elapsed_s":time.monotonic()-before,"log":str(log)})
                status["completed_batches"]=sum(v["exit_code"]==0 for v in status["processes"])
                status["elapsed_s"]=time.monotonic()-start
                write_json(a.output/f"{a.phase}_status.json",status)
                print(f"{a.phase} {len(status['processes'])}/{len(seeds)} eval{seed} gpu{gpu} exit={r.returncode}",flush=True)
            if r.returncode: raise RuntimeError(f"failed eval{seed}: {log}")
    errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,g,seeds[g::2]) for g in (0,1)]
        for f in futures:
            try: f.result()
            except Exception as e: errors.append(str(e))
    status.update(status="failed" if errors else "completed",errors=errors,finished=timestamp(),elapsed_s=time.monotonic()-start)
    write_json(a.output/f"{a.phase}_status.json",status)
    if errors: raise RuntimeError(errors)


if __name__=="__main__": main()
