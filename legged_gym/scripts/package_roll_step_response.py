#!/usr/bin/env python3
"""Package raw command-response evidence, frozen policy and paper handoff."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

from legged_gym.scripts.roll_step_response_protocol import RUN,CHECKPOINT,SEEDS,sha256_file,write_json


ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/"docs/experiments/roll_step_response_2026-10-18"
SCRIPTS=("roll_step_response_protocol.py","roll_step_response_experiment.py","roll_step_response_launch.py",
    "roll_step_response_analysis.py","test_roll_step_response_protocol.py","package_roll_step_response.py",
    "payload_step_protocol.py","payload_step_experiment.py","payload_step_analysis.py",
    "eccentric_payload_protocol.py","terrain_step_protocol.py","terrain_step_analysis.py")


def copy_tree(source,dest):
    shutil.copytree(source,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc","SHA256SUMS"))


def checksum(dest):
    files=sorted(p for p in dest.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
    (dest/"SHA256SUMS").write_text("\n".join(f"{sha256_file(p)}  ./{p.relative_to(dest).as_posix()}" for p in files)+"\n")
    return len(files)


def package(experiment,dest,collection):
    status=json.loads((experiment/"formal_status.json").read_text())
    assert status["status"]=="completed" and status["completed_batches"]==3
    assert {r["evaluation_seed"] for r in status["processes"]}==set(SEEDS)
    assert all(r["exit_code"]==0 for r in status["processes"])
    source=ROOT/"logs/wheelfoot_flat/WF_TRON1A"/RUN
    hashes={n:sha256_file(source/n) for n in
            (f"model_{CHECKPOINT}.pt","env_cfg.json","train_cfg.json","wheelfoot_flat.py","wheelfoot_flat_config.py")}
    logs=[]
    for row in status["processes"]:
        meta=json.loads((experiment/"batches"/f"unified_eval{row['evaluation_seed']}"/"result.json").read_text())
        assert meta["checkpoint_sha256"]==hashes[f"model_{CHECKPOINT}.pt"]
        assert meta["archived_task_sha256"]==hashes["wheelfoot_flat.py"]
        log=experiment/"process_logs"/Path(row["log"]).name
        assert log.exists() and meta["cleanup_completed"] and str(row["gpu"])==meta["visible_gpu"]
        logs.append({**row,"actual_relative_log":str(log.relative_to(experiment))})
    write_json(experiment/"analysis/execution_audit.json",{"status":"passed","unique_batches":3,
        "all_exit_codes_zero":True,"runtime_checkpoint_source_verified":True,"processes":logs})
    copy_tree(experiment,dest/"experiment"); copy_tree(DOCS,dest/"protocol")
    shutil.copy2(DOCS/"RESULTS.md",dest/"README.md")
    code=dest/"code/repo"
    for name in ("envs","algorithm","utils"): copy_tree(ROOT/"legged_gym"/name,code/"legged_gym"/name)
    shutil.copy2(ROOT/"legged_gym/__init__.py",code/"legged_gym/__init__.py")
    scripts=code/"legged_gym/scripts"; scripts.mkdir(parents=True,exist_ok=True)
    for name in SCRIPTS: shutil.copy2(ROOT/"legged_gym/scripts"/name,scripts/name)
    (scripts/"figs_chx").mkdir(exist_ok=True)
    shutil.copy2(ROOT/"legged_gym/scripts/figs_chx/fig0_sixd_design_variables.py",scripts/"figs_chx/fig0_sixd_design_variables.py")
    run=code/"logs/wheelfoot_flat/WF_TRON1A"/RUN; run.mkdir(parents=True,exist_ok=True)
    for name in hashes: shutil.copy2(source/name,run/name)
    robot=code/"resources/robots/WF_TRON1A"
    copy_tree(ROOT/"resources/robots/WF_TRON1A/meshes",robot/"meshes")
    (robot/"urdf").mkdir(exist_ok=True)
    shutil.copy2(ROOT/"resources/robots/WF_TRON1A/urdf/robot.urdf",robot/"urdf/robot.urdf")
    write_json(dest/"source_audit.json",{"run":RUN,"checkpoint":CHECKPOINT,"files":hashes,
        "framework_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        "task_source_from_saved_run":True,"shared_framework_current_compatibility_snapshot":True})
    write_json(dest/"delivery_manifest.json",{"date":"2026-10-18","formal_batches":3,"trajectories":90,
        "phases_per_trial":4,"training_seeds":1,"evaluation_seeds":3,"new_training":False,
        "formal_wall_s":status["elapsed_s"],"versions_analysis":{k:importlib.metadata.version(k)
            for k in ("numpy","scipy","matplotlib","torch","Pillow","fonttools")},
        "source_hashes":{n:sha256_file(scripts/n) for n in SCRIPTS},"actual_execution_audit":"passed"})
    figures=dest/"figures"; figures.mkdir(exist_ok=True)
    for pattern in ("*.png","*.pdf","*.svg"):
        for path in (experiment/"analysis").glob(pattern): shutil.copy2(path,figures/path.name)
    copy_tree(experiment/"analysis/fonts",figures/"fonts")
    shutil.copy2(DOCS/"FIGURE_CAPTIONS.md",figures/"图注_Captions.md")
    print(f"Archive: {dest}; {checksum(dest)} files")
    if collection:
        figure=collection/"24_统一滚动迈步策略命令响应"; figure.mkdir(parents=True,exist_ok=True)
        copy_tree(figures,figure)
        shutil.copy2(DOCS/"FIG24.md",figure/"README.md")
        for name in ("CONTACT_AND_MOTION_TABLE.md","phase_summary.csv","condition_summary.csv","preregistered_timeseries.csv"):
            shutil.copy2(experiment/"analysis"/name,figure/name)
        shutil.copy2(ROOT/"legged_gym/scripts/roll_step_response_analysis.py",figure/"roll_step_response_analysis.py")
        print(f"Figure folder: {figure}; {checksum(figure)} files")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",type=Path,required=True); p.add_argument("--target",type=Path,required=True)
    p.add_argument("--collection",type=Path)
    a=p.parse_args(); package(a.experiment.resolve(),a.target.resolve(),a.collection)
