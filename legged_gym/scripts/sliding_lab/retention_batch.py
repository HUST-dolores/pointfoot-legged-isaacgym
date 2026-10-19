"""Bounded sequential evaluation jobs; sleeping checkpoint watcher never busy-spins."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .geometry import write_json, sha256


def evaluate(output, seed, checkpoint=None, repeats=8):
    expected = sha256(checkpoint) if checkpoint else None
    if (output / "result.json").is_file():
        result = json.loads((output / "result.json").read_text())
        assert result["seed"] == seed and result["checkpoint_sha256"] == expected
        return result
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "legged_gym.scripts.sliding_lab.retention_eval",
               "--output", str(output), "--seed", str(seed), "--repeats", str(repeats)]
    if checkpoint:
        command.extend(("--checkpoint", str(checkpoint)))
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((output / "result.json").read_text())
    assert result["cleanup_completed"]
    return result


def main(a):
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = []
    if not a.watch:
        for seed in a.seeds:
            out = a.output / (a.name + "_seed" + str(seed))
            result = evaluate(out, seed, a.checkpoint, a.repeats)
            rows.append({"directory": str(out), "seed": seed,
                         "success_rate": result["disturbed_success_rate"]})
            print(json.dumps(rows[-1]), flush=True)
    else:
        done = set()
        while True:
            for train in a.watch:
                for checkpoint in sorted(train.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1])):
                    iteration = int(checkpoint.stem.split("_")[-1])
                    key = (train.name, iteration)
                    if iteration == 0 or key in done:
                        continue
                    out = a.output / (train.name + "_i" + str(iteration))
                    result = evaluate(out, 12345, checkpoint, a.repeats)
                    cases = json.loads((out / "cases.json").read_text())
                    inside = [c for c in cases if 0 < c["nominal_dv"] <= .75]
                    control = [c for c in cases if c["nominal_dv"] == 0]
                    row = {"train": train.name, "iteration": iteration, "directory": str(out),
                           "training_seed": json.loads((train / "metadata.json").read_text())["seed"],
                           "checkpoint": str(checkpoint), "checkpoint_sha256": result["checkpoint_sha256"],
                           "in_domain_success": sum(c["success"] for c in inside) / len(inside),
                           "all_strength_success": result["disturbed_success_rate"],
                           "control_success": sum(c["success"] for c in control) / len(control),
                           "fall_rate": result["disturbed_fall_rate"]}
                    rows.append(row)
                    done.add(key)
                    write_json(a.output / "validation_progress.json", rows)
                    print(json.dumps(row), flush=True)
            if all((train / "result.json").is_file() or (train / "STOPPED.json").is_file() for train in a.watch):
                break
            if time.monotonic() - started > a.max_wait:
                raise TimeoutError("Checkpoint watcher exceeded its bounded runtime")
            time.sleep(10)
        selected = {}
        for seed in sorted({r["training_seed"] for r in rows}):
            group = [r for r in rows if r["training_seed"] == seed]
            if not group:
                raise ValueError("No checkpoint was evaluated")
            # Do not use formal held-out seeds, or the .85 stress case, for model selection.
            selected["seed_" + str(seed)] = max(group, key=lambda r: (r["control_success"] == 1., r["in_domain_success"],
                                                            -r["fall_rate"], -r["iteration"]))
        write_json(a.output / "selected_checkpoints.json", selected)
    write_json(a.output / "batch_result.json", {"elapsed_s": time.monotonic() - started, "rows": rows,
                                               "all_child_processes_exited": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="formal_baseline")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=[34567, 45678])
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--watch", nargs="+", type=Path)
    p.add_argument("--max-wait", type=float, default=7200.)
    main(p.parse_args())
