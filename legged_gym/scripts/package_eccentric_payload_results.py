#!/usr/bin/env python3
"""Self-contained archive and small paper-figure handoffs for the static-load study."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

from legged_gym.scripts.eccentric_payload_protocol import POLICIES, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/experiments/eccentric_payload_matched_2026-10-18"
SCRIPTS = ("eccentric_payload_protocol.py", "eccentric_payload_experiment.py",
    "eccentric_payload_launch.py", "eccentric_payload_assist.py", "eccentric_payload_analysis.py", "test_eccentric_payload_protocol.py",
    "package_eccentric_payload_results.py", "payload_step_protocol.py", "payload_step_experiment.py",
    "payload_step_analysis.py", "terrain_step_protocol.py", "terrain_step_analysis.py")


def copy_tree(source, target):
    shutil.copytree(source, target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "SHA256SUMS"))


def checksum(target):
    rows = [f"{sha256_file(p)}  ./{p.relative_to(target).as_posix()}"
            for p in sorted(target.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (target / "SHA256SUMS").write_text("\n".join(rows)+"\n", encoding="utf-8")
    return len(rows)


def flatten(mapping, prefix=""):
    result={}
    for k,v in mapping.items():
        name=f"{prefix}.{k}" if prefix else k
        if isinstance(v,dict):
            result.update(flatten(v,name))
        else:
            result[name]=v
    return result


def audit_execution(experiment):
    primary=json.loads((experiment / "formal_status.json").read_text())
    if primary["status"]!="completed" or primary["completed_batches"]!=60:
        raise ValueError("formal queue is incomplete")
    executed=[r for r in primary["processes"] if not r["reused_complete"]]
    helpers=[]
    for path in sorted(experiment.glob("assist*_status.json")):
        status=json.loads(path.read_text())
        if status["status"]!="completed":
            raise ValueError("helper did not exit cleanly")
        helpers.append(path.name)
        executed.extend(status["processes"])
    ids=[r["batch_id"] for r in executed]
    if len(ids)!=60 or len(set(ids))!=60 or any(r["exit_code"] for r in executed):
        raise ValueError("duplicate, missing, or failed actual evaluation process")
    rows=[]
    for r in executed:
        folder=experiment / "batches" / r["batch_id"]
        meta=json.loads((folder / "result.json").read_text())
        log=experiment / "process_logs" / Path(r["log"]).name
        if not log.exists() or not meta["cleanup_completed"] or str(r["gpu"])!=meta["visible_gpu"]:
            raise ValueError("actual process log, cleanup, or GPU mismatch")
        run=ROOT / "logs/wheelfoot_flat/WF_TRON1A" / meta["run_name"]
        if sha256_file(run / f"model_{meta['checkpoint']}.pt")!=meta["checkpoint_sha256"]:
            raise ValueError("runtime policy does not match archived checkpoint")
        if sha256_file(run / "wheelfoot_flat.py")!=meta["archived_source_sha256"]:
            raise ValueError("runtime task does not match archived source")
        rows.append({"batch_id":r["batch_id"],"gpu":r["gpu"],"exit_code":r["exit_code"],
            "elapsed_s":r["elapsed_s"],"actual_log":str(log.relative_to(experiment)),
            "runtime_checkpoint_and_source_verified":True,"cleanup_completed":True})
    result={"status":"passed","actual_executed_batches":60,"unique_batches":60,
        "helper_status_files":helpers,"primary_reused_batches":sum(r["reused_complete"] for r in primary["processes"]),
        "formal_wall_s":primary["elapsed_s"],"processes":sorted(rows,key=lambda r:r["batch_id"])}
    write_json(experiment / "analysis/execution_audit.json",result)
    return result


def package(experiment,target,collection):
    execution=audit_execution(experiment)
    target.mkdir(parents=True,exist_ok=True)
    copy_tree(experiment,target / "experiment")
    copy_tree(DOCS,target / "protocol")
    shutil.copy2(DOCS / "RESULTS.md",target / "README.md")
    code=target / "code/repo"
    for name in ("envs","algorithm","utils"):
        copy_tree(ROOT / "legged_gym" / name,code / "legged_gym" / name)
    shutil.copy2(ROOT / "legged_gym/__init__.py",code / "legged_gym/__init__.py")
    scripts=code / "legged_gym/scripts"; scripts.mkdir(parents=True,exist_ok=True)
    for name in SCRIPTS:
        shutil.copy2(ROOT / "legged_gym/scripts" / name,scripts / name)
    (scripts / "figs_chx").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "legged_gym/scripts/figs_chx/fig0_sixd_design_variables.py",
                 scripts / "figs_chx/fig0_sixd_design_variables.py")
    robot=code / "resources/robots/WF_TRON1A"
    copy_tree(ROOT / "resources/robots/WF_TRON1A/meshes",robot / "meshes")
    (robot / "urdf").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "resources/robots/WF_TRON1A/urdf/robot.urdf",robot / "urdf/robot.urdf")
    inputs,envs,trains={},{},{}
    for p,(_,run,ckpt) in POLICIES.items():
        source=ROOT / "logs/wheelfoot_flat/WF_TRON1A" / run
        dest=code / "logs/wheelfoot_flat/WF_TRON1A" / run; dest.mkdir(parents=True,exist_ok=True)
        files=(f"model_{ckpt}.pt","env_cfg.json","train_cfg.json","wheelfoot_flat.py","wheelfoot_flat_config.py")
        for name in files:
            shutil.copy2(source / name,dest / name)
        inputs[p]={name:sha256_file(dest / name) for name in files}
        envs[p]=flatten(json.loads((source / "env_cfg.json").read_text()))
        trains[p]=flatten(json.loads((source / "train_cfg.json").read_text()))
    def diff(group):
        return {k:{p:v.get(k) for p,v in group.items()} for k in sorted(set().union(*(set(v) for v in group.values())))
                if group["rl"].get(k)!=group["mg"].get(k)}
    write_json(target / "source_audit.json",{"inputs":inputs,"saved_env_differences":diff(envs),
        "saved_train_differences":diff(trains),"framework_commit":subprocess.check_output(
            ["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        "framework_is_current_compatibility_snapshot":True})
    versions={k:importlib.metadata.version(k) for k in ("numpy","scipy","matplotlib","torch","Pillow","fonttools")}
    status=json.loads((experiment / "formal_status.json").read_text())
    write_json(target / "delivery_manifest.json",{"date":"2026-10-18","formal_batches":60,
        "trajectories":1800,"training_seeds_per_method":1,"evaluation_seeds":3,
        "new_training":False,"formal_status":status["status"],"formal_wall_s":status["elapsed_s"],
        "versions_analysis":versions,"source_hashes":{n:sha256_file(scripts/n) for n in SCRIPTS},
        "freeze":"protocol/FROZEN_BEFORE_FORMAL.md","old_experiments_unchanged":True,
        "actual_execution_audit":execution["status"],"primary_reused_batches":execution["primary_reused_batches"]})
    analysis=experiment / "analysis"
    figures=target / "figures"; figures.mkdir(exist_ok=True)
    for pattern in ("*.png","*.pdf","*.svg"):
        for path in analysis.glob(pattern):
            shutil.copy2(path,figures / path.name)
    copy_tree(analysis / "fonts",figures / "fonts")
    shutil.copy2(DOCS / "FIGURE_CAPTIONS.md",figures / "图注_Captions.md")
    print(f"Archive: {target}; {checksum(target)} files")
    if collection is None:
        return
    for folder,prefix,note in (
        ("22_居中与偏心负载姿态及失效率","center_eccentric_posture_","FIG22.md"),
        ("23_居中与偏心负载同步动态","center_eccentric_dynamics_","FIG23.md")):
        dest=collection / folder; dest.mkdir(parents=True,exist_ok=True)
        for path in figures.glob(prefix+"*"):
            shutil.copy2(path,dest / path.name)
        shutil.copy2(DOCS / note,dest / "README.md")
        shutil.copy2(DOCS / "FIGURE_CAPTIONS.md",dest / "图注_Captions.md")
        copy_tree(analysis / "fonts",dest / "fonts")
        shutil.copy2(ROOT / "legged_gym/scripts/eccentric_payload_analysis.py",dest / "eccentric_payload_analysis.py")
        for name in ("summary.csv","per_evaluation_seed_metrics.csv","main_m13_timeseries.csv"):
            shutil.copy2(analysis / name,dest / name)
        print(f"Figure folder: {dest}; {checksum(dest)} files")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--collection",type=Path)
    a=p.parse_args(); package(a.experiment.resolve(),a.target.resolve(),a.collection)
