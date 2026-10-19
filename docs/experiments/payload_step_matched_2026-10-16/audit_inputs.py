#!/usr/bin/env python3
"""Audit a frozen evaluation plan without importing the simulator or policies."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


RUNS = (
    ("rl_s1", "RL-only", 1, "Jul05_15-13-14_rlonly_loadest", 16000, 20250705),
    ("rl_s2", "RL-only", 2, "Jul05_15-21-33_rlonly_loadest_s2", 16000, 20250706),
    ("mg_s1", "Model-guided", 1, "Jul05_23-30-43_mg_e1_s1", 11000, 20250705),
    ("mg_s2", "Model-guided", 2, "Jul05_23-33-51_mg_e1_s2", 16000, 20250706),
)
EVALUATION_SEEDS = (12345, 20261016, 20261017)
ALLOWED_DIFFERENCES = {
    "env_cfg": {
        "seed", "env.num_observations", "env.num_critic_observations",
        "env.use_qs_in_obs", "env.use_residual_learning", "env.use_torques_in_obs",
    },
    "train_cfg": {
        "seed", "MLP_Encoder.num_input_dim", "algorithm.use_load_residual_estimation",
        "runner.run_name",
    },
}


def flatten(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def audit(repo, output_dir):
    records, configs, matrix = [], {"env_cfg": [], "train_cfg": []}, []
    for policy_id, method, repeat, run, checkpoint, training_seed in RUNS:
        directory = repo / "logs/wheelfoot_flat/WF_TRON1A" / run
        env = json.loads((directory / "env_cfg.json").read_text())
        train = json.loads((directory / "train_cfg.json").read_text())
        guided = method == "Model-guided"
        if (env["seed"], train["seed"]) != (training_seed, training_seed):
            raise ValueError(f"training seed mismatch: {policy_id}")
        expected = (51, 58, 940) if guided else (31, 38, 540)
        actual = (env["env"]["num_observations"], env["env"]["num_critic_observations"],
                  train["MLP_Encoder"]["num_input_dim"])
        if actual != expected:
            raise ValueError(f"saved dimension mismatch: {policy_id}")
        if any(env["env"][flag] != guided for flag in (
            "use_qs_in_obs", "use_torques_in_obs", "use_residual_learning"
        )):
            raise ValueError(f"method flags mismatch: {policy_id}")
        if (train["MLP_Encoder"]["obs_history_length"],
                train["MLP_Encoder"]["num_output_dim"]) != (10, 7):
            raise ValueError(f"saved history/output mismatch: {policy_id}")
        if env["domain_rand"]["add_load_range"] != [2, 30]:
            raise ValueError(f"unexpected training load range: {policy_id}")
        if train["algorithm"]["use_load_residual_estimation"] != guided:
            raise ValueError(f"residual interpretation mismatch: {policy_id}")
        if not all(env["env"][flag] for flag in ("use_gait_stepping", "use_gait_phase")):
            raise ValueError(f"missing step/phase input: {policy_id}")
        for key, value in (("env_cfg", env), ("train_cfg", train)):
            configs[key].append(flatten(value))
        files = [directory / f"model_{checkpoint}.pt", directory / "env_cfg.json",
                 directory / "train_cfg.json", directory / "wheelfoot_flat.py",
                 directory / "wheelfoot_flat_config.py"]
        records.append({"policy_id": policy_id, "method": method, "training_repeat": repeat,
                        "training_seed": training_seed, "checkpoint": checkpoint,
                        "saved_dimensions": list(actual), "files": [file_record(p) for p in files]})
        for seed in EVALUATION_SEEDS:
            for condition, mass in (("empty", 0.0), ("free_box", 5.9268727)):
                matrix.append({
                    "batch_id": f"{policy_id}_eval{seed}_{condition}",
                    "status": "planned_not_run", "policy_id": policy_id, "method": method,
                    "training_repeat": repeat, "training_seed": training_seed,
                    "evaluation_seed": seed, "condition": condition, "payload_mass_kg": mass,
                    "num_envs": 30, "duration_s": 40, "control_dt_s": 0.02,
                    "roll_command_mps": 0.5, "step_command_mps": 0.4,
                    "total_forward_target_mps": 0.9,
                })
    differences = {}
    missing = object()
    for name, values in configs.items():
        keys = sorted(set().union(*(set(value) for value in values)))
        changed = [key for key in keys
                   if any(value.get(key, missing) != values[0].get(key, missing)
                          for value in values[1:])]
        if set(changed) - ALLOWED_DIFFERENCES[name]:
            raise ValueError(f"unexpected configuration differences: {name}: {changed}")
        differences[name] = {key: [value.get(key, "<missing>") for value in values]
                             for key in changed}
    source_paths = (
        "legged_gym/envs/base/base_task.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py",
        "legged_gym/algorithm/ppo.py",
        "legged_gym/algorithm/on_policy_runner.py",
        "legged_gym/algorithm/mlp_encoder.py",
        "legged_gym/scripts/play.py",
        "legged_gym/scripts/analyze_payload_motion_metrics.py",
        "legged_gym/scripts/measure_load_estimation_walk.py",
        "resources/robots/WF_TRON1A/urdf/robot.urdf",
    )
    manifest = {
        "date": "2026-10-16", "timezone": "Asia/Shanghai", "status": "plan_and_static_audit_only",
        "repo_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "audit_scope": "file existence, hashes, saved configurations; no checkpoint execution",
        "checkpoint_order": [row[0] for row in RUNS], "checkpoints": records,
        "configuration_differences": differences,
        "current_sources_audit_only_not_training_snapshot": [file_record(repo / p) for p in source_paths],
        "planned_batch_count": len(matrix), "planned_trajectories": sum(r["num_envs"] for r in matrix),
        "planned_control_transitions": 1440000, "aggregate_environment_hours_not_wallclock_eta": 8.0,
        "independent_training_repeats_per_method": 2, "evaluation_seeds": list(EVALUATION_SEEDS),
        "scenario_noise_tables_generated": False, "simulator_preflight_completed": False,
        "evaluations_started": False, "new_measurement_results": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "evaluation_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    print(json.dumps({"audit": "passed", "planned_batches": len(matrix),
                      "planned_trajectories": manifest["planned_trajectories"],
                      "evaluations_started": False, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.repo.resolve(), args.output_dir.resolve())
