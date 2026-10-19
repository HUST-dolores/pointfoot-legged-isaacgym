"""Bounded two-GPU reward study, with periodic validation and held-out testing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

from .geometry import sha256
from .retention_reward_ablation import ARMS


ROOT = Path(__file__).resolve().parents[3]
SEEDS = (303, 404)
FORMAL_SEEDS = (61001, 61002, 61003)
START_HASHES = {
    303: "c39a01bf480ccb9348c8d288c2c8a18bbe0997d172ec412ff7666c02dfe3bd69",
    404: "b1eb6f789a8589089c0bf151ea465f06f869e72bfb1e05fb53a0058ae3afa038",
}


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def evaluation_metrics(directory):
    result = read_json(directory / "result.json")
    assert result["cleanup_completed"] and result["all_trials_single_spawn"]
    assert result["no_pose_writes_after_spawn"]
    cases = read_json(directory / "cases.json")
    groups = {"in_domain": [c for c in cases if 0 < c["nominal_dv"] <= .75],
              "stress": [c for c in cases if c["nominal_dv"] > .75],
              "control": [c for c in cases if c["nominal_dv"] == 0]}
    row = {"directory": str(directory), "checkpoint": result["checkpoint"],
           "checkpoint_sha256": result["checkpoint_sha256"], "evaluation_seed": result["seed"],
           "scenario_sha256": sha256(directory / "scenarios.json")}
    for name, group in groups.items():
        row[name + "_success"] = sum(c["success"] for c in group) / len(group)
        row[name + "_fall_rate"] = sum(c["fall"] for c in group) / len(group)
        row[name + "_mean_displacement_m"] = sum(c["max_robot_displacement_before_failure_m"] for c in group) / len(group)
    return row


def selection_key(row):
    # Preserve the preceding study's selection rule; stress and test seeds never select models.
    return (row["control_success"] == 1., row["in_domain_success"],
            -row["in_domain_fall_rate"], -row["iteration"])


class Study:
    def __init__(self, output, max_hours):
        self.output = output
        self.started = time.monotonic()
        self.deadline = self.started + max_hours * 3600
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.children = {}
        self.state = {"started": stamp(), "state": "starting", "jobs": {},
                      "validation": [], "selection": {}, "formal": {}, "errors": []}

    def persist(self):
        with self.lock:
            self.state["updated"] = stamp()
            self.state["elapsed_s"] = time.monotonic() - self.started
            self.state["live_child_pids"] = [p.pid for p, _ in self.children.values() if p.poll() is None]
            atomic_json(self.output / "status.json", self.state)

    def event(self, kind, **kwargs):
        with self.lock:
            with (self.output / "events.jsonl").open("a", buffering=1) as handle:
                handle.write(json.dumps({"time": stamp(), "event": kind, **kwargs}) + "\n")

    def guard(self):
        if self.stop.is_set():
            raise RuntimeError("Another study worker failed; stopping owned children")
        if time.monotonic() > self.deadline:
            raise TimeoutError("Bounded study runtime exceeded")

    def start(self, key, gpu, module, args, log):
        self.guard()
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu), ROBOT_TYPE="WF_TRON1A",
                           PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("x", buffering=1)
        command = [sys.executable, "-u", "-m", "legged_gym.scripts.sliding_lab." + module, *map(str, args)]
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            handle.close()
            raise
        with self.lock:
            self.children[key] = (process, handle)
        self.event("child_started", key=key, gpu=gpu, pid=process.pid, command=command, log=str(log))
        return process

    def reap(self, key):
        with self.lock:
            process, handle = self.children[key]
        code = process.poll()
        assert code is not None
        process.wait()
        handle.close()
        self.event("child_exited", key=key, pid=process.pid, exit_code=code)
        with self.lock:
            del self.children[key]
        if code != 0:
            raise RuntimeError(f"{key} exited {code}; see its stdout log")

    def start_eval(self, key, gpu, checkpoint, directory, seed, repeats):
        return self.start(key, gpu, "retention_eval",
                          ["--checkpoint", checkpoint, "--output", directory,
                           "--seed", seed, "--repeats", repeats, "--clean-exit"], directory.with_suffix(".log"))

    def evaluate(self, key, gpu, checkpoint, directory, seed, repeats):
        process = self.start_eval(key, gpu, checkpoint, directory, seed, repeats)
        while process.poll() is None:
            self.guard()
            time.sleep(5)
        self.reap(key)
        return evaluation_metrics(directory)

    def record_validation(self, seed, arm, iteration, row):
        row.update(training_seed=seed, arm=arm, iteration=iteration)
        with self.lock:
            self.state["validation"].append(row)
        self.event("validated", **row)
        self.persist()
        return row

    def train_arm(self, seed, gpu, arm, initial):
        name = f"{arm}_s{seed}"
        directory = self.output / "training" / name
        checkpoint = Path(initial["checkpoint"])
        weights = ARMS[arm]
        args = ["--output", directory, "--resume", checkpoint, "--seed", seed,
                "--num-envs", 512, "--iterations", 2000, "--save-every", 250,
                "--lr", "3e-5", "--action-std", .08, "--architecture", "residual",
                "--alive-weight", weights["alive_weight"], "--drop-cost", weights["drop_cost"]]
        process = self.start("train_" + name, gpu, "retention_train", args, directory.with_suffix(".log"))
        with self.lock:
            self.state["jobs"][name] = {"phase": "training", "gpu": gpu, "pid": process.pid,
                                       "started": stamp(), "weights": weights, "initial_iteration": 1000}
        self.persist()
        seen = set()
        active = None
        candidates = [initial]
        reaped = False
        while True:
            self.guard()
            if (directory / "status.json").is_file():
                try:
                    progress = read_json(directory / "status.json")
                except (ValueError, OSError):
                    progress = None
                if progress:
                    with self.lock:
                        self.state["jobs"][name]["progress"] = progress
            if active and active[1].poll() is not None:
                key, _, iteration, out = active
                self.reap(key)
                row = self.record_validation(seed, arm, iteration, evaluation_metrics(out))
                candidates.append(row)
                active = None
            ready = sorted(directory.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
            ready = [p for p in ready if p.name not in seen]
            if active is None and ready:
                path = ready[0]
                iteration = int(path.stem.split("_")[-1])
                out = self.output / "validation" / f"{name}_i{iteration}"
                key = "val_" + out.name
                active = (key, self.start_eval(key, gpu, path, out, 12345, 4), iteration, out)
                seen.add(path.name)
            if process.poll() is not None and not reaped:
                self.reap("train_" + name)
                result = read_json(directory / "result.json")
                assert result["cleanup_completed"] and result["encoder_unchanged"] and result["base_actor_unchanged"]
                assert result["iterations"] == 2000
                metadata = read_json(directory / "metadata.json")
                assert metadata["resume_checkpoint_sha256"] == START_HASHES[seed]
                assert metadata["additional_iterations"] == 1000
                expected = set() if arm == "control" else {"alive_weight" if arm == "alive4" else "drop_cost"}
                assert set(metadata["reward_settings_changes"]) == expected
                with self.lock:
                    self.state["jobs"][name].update(phase="finishing_validation", result=result)
                reaped = True
            if reaped and active is None and all(p.name in seen for p in ready):
                break
            self.persist()
            time.sleep(5)
        assert len(candidates) == 5
        selected = dict(max(candidates, key=selection_key))
        selected["selected_for_arm"] = arm
        with self.lock:
            self.state["selection"][name] = selected
            self.state["jobs"][name].update(phase="completed", completed=stamp())
        self.event("selected", name=name, **selected)
        self.persist()
        return selected

    def worker(self, seed, gpu):
        try:
            checkpoint = ROOT / f"logs/warm_start_s{seed}/model_1000.pt"
            assert sha256(checkpoint) == START_HASHES[seed]
            out = self.output / "validation" / f"initial_s{seed}_i1000"
            initial = self.record_validation(seed, "initial", 1000,
                self.evaluate(out.name, gpu, checkpoint, out, 12345, 4))
            order = ("alive4", "control", "drop24") if seed == 303 else ("drop24", "control", "alive4")
            selected = {"initial": initial}
            for arm in order:
                selected[arm] = self.train_arm(seed, gpu, arm, initial)
            cache = {}
            for arm in ("initial", "control", "alive4", "drop24"):
                chosen = selected[arm]
                for evaluation_seed in FORMAL_SEEDS:
                    identity = (chosen["checkpoint_sha256"], evaluation_seed)
                    if identity not in cache:
                        out = self.output / "formal" / f"s{seed}_{identity[0][:12]}_e{evaluation_seed}"
                        cache[identity] = self.evaluate(out.name, gpu, Path(chosen["checkpoint"]), out, evaluation_seed, 8)
                        self.event("formal_evaluated", training_seed=seed, **cache[identity])
                    with self.lock:
                        self.state["formal"][f"{arm}_s{seed}_e{evaluation_seed}"] = cache[identity]
                    self.persist()
        except BaseException:
            with self.lock:
                self.state["errors"].append({"training_seed": seed, "traceback": traceback.format_exc()})
            self.stop.set()
            self.persist()
            raise

    def cleanup(self):
        with self.lock:
            children = list(self.children.items())
        for key, (process, _) in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                self.event("owned_child_terminated", key=key, pid=process.pid)
        for key, (process, handle) in children:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            handle.close()
        with self.lock:
            self.children.clear()

    def report(self):
        import numpy as np
        rows = []
        scenario_digests = {}
        for label, row in sorted(self.state["formal"].items()):
            arm, suffix = label.rsplit("_s", 1)
            training_seed, evaluation_seed = map(int, suffix.split("_e"))
            directory = Path(row["directory"])
            digest = row["scenario_sha256"]
            assert evaluation_seed not in scenario_digests or scenario_digests[evaluation_seed] == digest
            scenario_digests[evaluation_seed] = digest
            with np.load(directory / "trace.npz") as trace:
                root = trace["root"]
                time_s = trace["time_s"]
                q = root[..., 3:7]
                x, y, z, w = (q[..., i] for i in range(4))
                roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
                pitch = np.arcsin(np.clip(2*(w*y-z*x), -1., 1.))
                for case in read_json(directory / "cases.json"):
                    i = case["env"]
                    mask = time_s[:, i] >= .7
                    rows.append({"arm": arm, "training_seed": training_seed,
                        "evaluation_seed": evaluation_seed, "checkpoint_sha256": row["checkpoint_sha256"],
                        "source_directory": str(directory), **case,
                        "post_load_roll_rms_deg": float(np.rad2deg(np.sqrt(np.mean(roll[mask, i]**2)))),
                        "post_load_pitch_rms_deg": float(np.rad2deg(np.sqrt(np.mean(pitch[mask, i]**2)))),
                        "whole_trial_max_robot_displacement_m": float(np.linalg.norm(root[:, i, :2]-root[0, i, :2], axis=-1).max())})
        report = self.output / "reports"
        report.mkdir()
        with (report / "all_trials.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = []
        for arm in ("initial", "control", "alive4", "drop24"):
            for seed in SEEDS:
                for domain, predicate in (("control", lambda d: d == 0),
                        ("in_domain", lambda d: 0 < d <= .75), ("stress", lambda d: d > .75)):
                    group = [r for r in rows if r["arm"] == arm and r["training_seed"] == seed and predicate(r["nominal_dv"])]
                    summary.append({"arm": arm, "training_seed": seed, "domain": domain,
                        "trials": len(group), "successes": sum(r["success"] for r in group),
                        "success_rate": float(np.mean([r["success"] for r in group])),
                        "drop_rate": float(np.mean([r["drop"] for r in group])),
                        "fall_rate": float(np.mean([r["fall"] for r in group])),
                        "mean_whole_trial_displacement_m": float(np.mean([r["whole_trial_max_robot_displacement_m"] for r in group])),
                        "mean_post_load_roll_rms_deg": float(np.mean([r["post_load_roll_rms_deg"] for r in group])),
                        "mean_post_load_pitch_rms_deg": float(np.mean([r["post_load_pitch_rms_deg"] for r in group]))})
        with (report / "physical_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        atomic_json(report / "summary.json", {"rows": summary, "all_failures_retained": True,
            "same_checkpoint_test_batches_reused_not_independent": True,
            "posture_window": "0.7--10 s, including after failure; not survivor-only",
            "test_seeds_not_used_for_checkpoint_selection": True,
            "no_disturbance_group_still_carries_load": True,
            "zero_command_is_not_position_holding": True})
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        colors = {"control": "#60666b", "alive4": "#087e8b", "drop24": "#c55136"}
        for ax, seed in zip(axes, SEEDS):
            initial = next(r for r in self.state["validation"] if r["training_seed"] == seed and r["arm"] == "initial")
            for arm in ARMS:
                data = sorted([initial] + [r for r in self.state["validation"] if r["training_seed"] == seed and r["arm"] == arm], key=lambda r: r["iteration"])
                ax.plot([r["iteration"]-1000 for r in data], [r["in_domain_success"]*100 for r in data], marker="o", ms=4, label=arm, color=colors[arm])
            ax.set(title=f"Training seed {seed}", xlabel="Additional PPO updates", ylim=(0, 103))
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Validation retention success (%)")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(report / ("validation_learning_curves." + extension), dpi=220)
        plt.close(fig)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "protocol.json", {
        "registered": stamp(), "arms": ARMS, "training_seeds": SEEDS,
        "gpu_mapping": {303: 0, 404: 1}, "initial_checkpoint_hashes": START_HASHES,
        "additional_updates_per_arm_seed": 1000, "num_envs": 512, "steps_per_update": 24,
        "additional_samples_per_arm_seed": 12288000, "total_additional_samples": 73728000,
        "learning_rate": 3e-5, "optimizer_state_resumed": True, "physics_state_reinitialized": True,
        "validation_seed": 12345, "validation_repeats": 4, "validation_every_updates": 250,
        "formal_seeds": FORMAL_SEEDS, "formal_repeats": 8,
        "incumbent_included_in_selection": True,
        "selection": "control success == 1; in-domain success; lower fall rate; earlier checkpoint",
        "nominal_training_impulse_dv_max_mps": .75, "stress_impulse_dv_mps": .85,
        "unchanged": ["physics", "friction", "mass", "disturbance curriculum", "observations",
                      "frozen encoder", "frozen base actor", "residual mean-action limits", "PPO core"],
        "max_runtime_hours": args.max_hours,
        "source_hashes": {p.name: sha256(p) for p in Path(__file__).parent.glob("retention*.py")}})
    study = Study(args.output, args.max_hours)
    study.state["state"] = "running"
    study.persist()
    failed = False
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(study.worker, seed, gpu) for gpu, seed in enumerate(SEEDS)]
            while not all(f.done() for f in futures):
                study.persist()
                if study.stop.is_set() or time.monotonic() > study.deadline:
                    study.stop.set()
                    study.cleanup()
                time.sleep(15)
            for future in futures:
                future.result()
        study.report()
        study.state.update(state="completed", completed=stamp())
    except BaseException:
        failed = True
        study.stop.set()
        with study.lock:
            study.state["errors"].append({"controller_traceback": traceback.format_exc()})
        study.state.update(state="failed", completed=stamp())
    finally:
        study.cleanup()
        study.persist()
        atomic_json(args.output / "finished.json", {
            "time": stamp(), "state": study.state["state"], "all_owned_children_exited": True,
            "elapsed_s": time.monotonic()-study.started})
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=6.)
    main(parser.parse_args())
