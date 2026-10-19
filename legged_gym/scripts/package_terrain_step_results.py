#!/usr/bin/env python3
"""Archive matched terrain evaluations, original camera pixels and their code."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

from legged_gym.scripts.terrain_step_protocol import POLICIES, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]


def copy_tree(source, target):
    shutil.copytree(source, target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "SHA256SUMS"))


def flatten(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def package(experiment, target):
    target.mkdir(parents=True, exist_ok=True)
    copy_tree(experiment, target / "experiment")
    docs = ROOT / "docs/experiments/terrain_step_matched_2026-10-16"
    copy_tree(docs, target / "protocol")
    shutil.copy2(docs / "RESULTS.md", target / "README.md")
    code = target / "code/repo"
    for name in ("envs", "algorithm", "utils"):
        copy_tree(ROOT / "legged_gym" / name, code / "legged_gym" / name)
    shutil.copy2(ROOT / "legged_gym/__init__.py", code / "legged_gym/__init__.py")
    scripts = code / "legged_gym/scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    names = ("terrain_step_protocol.py", "terrain_step_experiment.py", "terrain_step_launch.py",
             "terrain_step_analysis.py", "test_terrain_step_protocol.py", "probe_wheel_collision.py",
             "package_terrain_step_results.py", "payload_step_protocol.py", "payload_step_experiment.py",
             "payload_step_analysis.py")
    for name in names:
        shutil.copy2(ROOT / "legged_gym/scripts" / name, scripts / name)
    (scripts / "figs_chx").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "legged_gym/scripts/figs_chx/fig0_sixd_design_variables.py",
                 scripts / "figs_chx/fig0_sixd_design_variables.py")
    robot = code / "resources/robots/WF_TRON1A"
    copy_tree(ROOT / "resources/robots/WF_TRON1A/meshes", robot / "meshes")
    (robot / "urdf").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "resources/robots/WF_TRON1A/urdf/robot.urdf", robot / "urdf/robot.urdf")
    configs, train_configs, inputs = [], [], {}
    for policy, (_, run, checkpoint) in POLICIES.items():
        source = ROOT / "logs/wheelfoot_flat/WF_TRON1A" / run
        destination = code / "logs/wheelfoot_flat/WF_TRON1A" / run
        destination.mkdir(parents=True, exist_ok=True)
        files = (f"model_{checkpoint}.pt", "env_cfg.json", "train_cfg.json", "wheelfoot_flat.py", "wheelfoot_flat_config.py")
        for name in files:
            shutil.copy2(source / name, destination / name)
        inputs[policy] = {name: sha256_file(destination / name) for name in files}
        configs.append(flatten(json.loads((source / "env_cfg.json").read_text())))
        train_configs.append(flatten(json.loads((source / "train_cfg.json").read_text())))
    def differences(values):
        return {k: [v.get(k) for v in values] for k in sorted(set().union(*(set(v) for v in values)))
                if values[0].get(k) != values[1].get(k)}
    write_json(target / "source_audit.json", {"checkpoint_files": inputs,
        "saved_env_differences": differences(configs), "saved_train_differences": differences(train_configs),
        "same_archived_task_source": inputs["base"]["wheelfoot_flat.py"] == inputs["adapted"]["wheelfoot_flat.py"],
        "current_shared_framework_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()})
    figures = target / "figures"
    figures.mkdir(exist_ok=True)
    for pattern in ("*.png", "*.pdf", "*.svg"):
        for source in (experiment / "analysis").glob(pattern):
            shutil.copy2(source, figures / source.name)
    copy_tree(experiment / "analysis/fonts", figures / "fonts")
    shutil.copy2(docs / "FIGURE_CAPTIONS.md", figures / "图注_Captions.md")
    versions = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "matplotlib", "torch", "Pillow", "fonttools")}
    write_json(target / "delivery_manifest.json", {"date": "2026-10-16", "status": "completed",
        "formal_batches": 18, "trajectories": 540, "policy_pairs": 270,
        "formal_camera_original_frames": 24, "new_training": False, "versions_analysis": versions,
        "final_sources": {name: sha256_file(scripts/name) for name in names},
        "frozen_protocol": "protocol/FROZEN_BEFORE_FORMAL.md",
        "post_freeze_analysis_changes": "plot layout/cropping, MATLAB export, forward-displacement diagnostics and audits only",
        "performance_metrics_unchanged": True})
    hashes = [f"{sha256_file(p)}  ./{p.relative_to(target).as_posix()}" for p in sorted(target.rglob("*"))
              if p.is_file() and p.name != "SHA256SUMS"]
    (target / "SHA256SUMS").write_text("\n".join(hashes)+"\n", encoding="utf-8")
    print(f"Packaged {len(hashes)} files: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    package(args.experiment.resolve(), args.target.resolve())
