#!/usr/bin/env python3
"""Build a self-contained paper handoff without touching prior experiment files."""

import argparse
import importlib.metadata
from pathlib import Path
import shutil

from legged_gym.scripts.payload_step_protocol import POLICIES, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]


def copy_tree(source, target):
    shutil.copytree(source, target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "SHA256SUMS"))


def package(experiment, target):
    target.mkdir(parents=True, exist_ok=True)
    copy_tree(experiment, target / "experiment")
    docs = ROOT / "docs/experiments/payload_step_matched_2026-10-16"
    copy_tree(docs, target / "protocol")
    shutil.copy2(docs / "RESULTS.md", target / "README.md")
    for name in ("PROBLEMS.md", "RUN_STATUS.md", "audit_inputs.py", "audit_manifest.json", "evaluation_matrix.csv"):
        shutil.copy2(docs / name, target / name)
    code = target / "code/repo"
    for name in ("envs", "algorithm", "utils"):
        copy_tree(ROOT / "legged_gym" / name, code / "legged_gym" / name)
    shutil.copy2(ROOT / "legged_gym/__init__.py", code / "legged_gym/__init__.py")
    scripts = code / "legged_gym/scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    script_names = ("payload_step_protocol.py", "payload_step_experiment.py", "payload_step_launch.py",
                    "payload_step_analysis.py", "test_payload_step_protocol.py", "package_payload_step_results.py")
    for name in script_names:
        shutil.copy2(ROOT / "legged_gym/scripts" / name, scripts / name)
    (scripts / "figs_chx").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "legged_gym/scripts/figs_chx/fig0_sixd_design_variables.py",
                 scripts / "figs_chx/fig0_sixd_design_variables.py")
    robot = code / "resources/robots/WF_TRON1A"
    copy_tree(ROOT / "resources/robots/WF_TRON1A/meshes", robot / "meshes")
    (robot / "urdf").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "resources/robots/WF_TRON1A/urdf/robot.urdf", robot / "urdf/robot.urdf")
    for _, _, run, checkpoint, _ in POLICIES.values():
        source = ROOT / "logs/wheelfoot_flat/WF_TRON1A" / run
        destination = code / "logs/wheelfoot_flat/WF_TRON1A" / run
        destination.mkdir(parents=True, exist_ok=True)
        for name in (f"model_{checkpoint}.pt", "env_cfg.json", "train_cfg.json",
                     "wheelfoot_flat.py", "wheelfoot_flat_config.py"):
            shutil.copy2(source / name, destination / name)
    if (ROOT / "LICENSE").exists():
        shutil.copy2(ROOT / "LICENSE", code / "LICENSE")
    figures = target / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("*.png", "*.pdf", "*.svg"):
        for source in (experiment / "analysis").glob(suffix):
            shutil.copy2(source, figures / source.name)
    copy_tree(experiment / "analysis/fonts", figures / "fonts")
    shutil.copy2(docs / "FIGURE_CAPTIONS.md", figures / "图注_Captions.md")
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "fonttools", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(target / "delivery_manifest.json", {
        "date": "2026-10-16", "timezone": "Asia/Shanghai", "status": "completed",
        "formal_batches": 24, "trajectories": 720, "paired_trajectories": 360,
        "new_training": False, "raw_data_format": "numpy NPZ, allow_pickle=False",
        "analysis_environment_versions": versions,
        "final_source_sha256": {name: sha256_file(scripts / name) for name in script_names},
        "frozen_before_formal": "experiment/protocol/FROZEN_BEFORE_FORMAL.md",
        "post_freeze_changes": "plotting, export, and additional audits only; no metric/event/window changes",
        "static_audit_note": "root audit_manifest.json is the earlier plan audit, not current run status",
    })
    checksums = []
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{sha256_file(path)}  ./{path.relative_to(target).as_posix()}")
    (target / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(f"Packaged {len(checksums)} files: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    package(args.experiment.resolve(), args.target.resolve())
