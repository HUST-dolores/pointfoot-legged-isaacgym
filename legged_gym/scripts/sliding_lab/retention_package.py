"""Create a self-contained retention handoff without copying Git or private settings."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil

from .geometry import sha256, write_json


def main(a):
    from .robot import ROOT, RUN
    source = ROOT
    data = source / "logs/payload_retention_20261018"
    completion = json.loads((data / "task_finished.json").read_text())
    assert completion["all_retention_processes_exited"]
    verification = json.loads((data / "reports/verification.json").read_text())
    assert verification["status"] == "passed"
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    repo = a.output / "repo"
    shutil.copytree(source, repo, ignore=shutil.ignore_patterns(
        ".git", "logs", "__pycache__", "*.pyc", ".pytest_cache", ".agents", ".codex", ".aws", ".ssh",
        ".vscode", ".env", ".env.*"))
    destination_data = repo / "logs/payload_retention_20261018"
    shutil.copytree(data, destination_data)
    original_source = source / "logs/wheelfoot_flat/WF_TRON1A" / RUN
    original_target = repo / "logs/wheelfoot_flat/WF_TRON1A" / RUN
    original_target.mkdir(parents=True)
    for name in ("model_11000.pt", "env_cfg.json", "train_cfg.json", "wheelfoot_flat.py", "wheelfoot_flat_config.py"):
        if (original_source / name).is_file():
            shutil.copy2(original_source / name, original_target / name)
    shutil.copytree(data / "media", a.output / "media")
    shutil.copytree(data / "reports", a.output / "reports")
    shutil.copy2(a.readme, a.output / "README.md")
    shutil.copy2(source / "RETENTION_EXPERIMENT.md", a.output / "EXPERIMENT.md")
    shutil.copy2(source / "SLIDING_PROBLEMS.md", a.output / "PROBLEMS.md")
    index = []
    formal_directories = {str(Path("logs/payload_retention_20261018") / name)
                          for name in completion["formal_directories"]}
    for path in sorted(destination_data.rglob("result.json")):
        meta = json.loads(path.read_text())
        if "trials" not in meta:
            continue
        relative = path.parent.relative_to(repo)
        name = path.parent.name
        category = "formal" if str(relative) in formal_directories else "validation" if "validation" in str(relative) else "diagnostic_or_demo"
        index.append({"category": category, "directory": str(relative), "seed": meta["seed"],
                      "trials": meta["trials"], "duration_s": meta["settings"]["duration"],
                      "checkpoint": meta["checkpoint"], "success_rate": meta["disturbed_success_rate"]})
    with (a.output / "experiment_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    write_json(a.output / "manifest.json", {"created": datetime.now().isoformat(), "source_worktree": str(source),
        "task_completion": completion, "independent_verification_passed": True,
        "copied_without_git_or_private_settings": True, "original_frozen_checkpoint_included": True,
        "pilot_validation_and_formal_data_separated_in_index": True})
    files = sorted(p for p in a.output.rglob("*") if p.is_file())
    with (a.output / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for path in files:
            f.write(sha256(path) + "  " + str(path.relative_to(a.output)) + "\n")
    print(json.dumps({"output": str(a.output), "files_hashed": len(files),
                      "formal_trials": sum(r["trials"] for r in index if r["category"] == "formal")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--readme", type=Path, required=True)
    main(p.parse_args())
