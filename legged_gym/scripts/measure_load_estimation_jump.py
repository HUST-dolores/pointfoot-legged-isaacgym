#!/usr/bin/env python3
"""Record encoder payload estimates through complete jump phases."""

import argparse
import copy
import csv
import json
import os
import sys

import isaacgym  # noqa: F401
import numpy as np
import torch
from scipy.io import savemat

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.algorithm.mlp_encoder import MLP_Encoder
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.scripts.play import _apply_saved_env_cfg
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path


def _phase_metrics(true_mass, estimate, jump_state):
    rows = {}
    names = {0: "IDLE", 1: "CROUCH", 2: "FLIGHT", 3: "LANDED"}
    for state, name in names.items():
        mask = (jump_state == state) & (true_mass > 0.3)
        if not np.any(mask):
            rows[name] = {"samples": 0}
            continue
        error = estimate[mask] - true_mass[mask]
        rows[name] = {
            "samples": int(mask.sum()),
            "rmse_kg": float(np.sqrt(np.mean(error ** 2))),
            "bias_kg": float(np.mean(error)),
            "estimate_mean_kg": float(np.mean(estimate[mask])),
            "true_mean_kg": float(np.mean(true_mass[mask])),
        }
    return rows


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--payload_mode", choices=("attached", "free"), default="attached"
    )
    parser.add_argument("--load_mass", type=float, default=5.0)
    parser.add_argument("--jump_height", type=float, default=0.22)
    parser.add_argument("--jump_dist", type=float, default=0.18)
    parser.add_argument("--record_steps", type=int, default=1200)
    parser.add_argument("--output_dir", required=True)
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name
    )
    _apply_saved_env_cfg(env_cfg, train_cfg, args, log_root)

    env_cfg.env.num_envs = int(args.num_envs or 64)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.terrain.critic_measure_heights = False
    env_cfg.domain_rand.add_random_load = extra.payload_mode == "free"
    env_cfg.domain_rand.per_env_load_mass = extra.payload_mode == "free"
    env_cfg.domain_rand.add_load_range = [extra.load_mass, extra.load_mass]
    env_cfg.domain_rand.load_start_time_s = 0.5
    env_cfg.domain_rand.load_duration_range_s = [1.0e6, 1.0e6]
    env_cfg.domain_rand.load_interval_range_s = [1.0e6, 1.0e6]
    env_cfg.domain_rand.load_on_body_position_only = extra.payload_mode == "free"
    env_cfg.domain_rand.load_drop_terminal = extra.payload_mode == "free"
    env_cfg.domain_rand.load_offset_range_x = [0.0, 0.0]
    env_cfg.domain_rand.load_offset_range_y = [0.0, 0.0]
    env_cfg.domain_rand.load_offset_range_z = [0.10, 0.10]
    env_cfg.domain_rand.attached_payload = extra.payload_mode == "attached"
    env_cfg.domain_rand.attached_payload_mass_range = [
        extra.load_mass,
        extra.load_mass,
    ]
    env_cfg.domain_rand.attached_payload_offset_range_x = [0.0, 0.0]
    env_cfg.domain_rand.attached_payload_offset_range_y = [0.0, 0.0]
    env_cfg.domain_rand.attached_payload_offset_range_z = [0.10, 0.10]
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_inertia = False
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.noise.add_noise = False
    env_cfg.commands.jump_height_cmd_range = [extra.jump_height, extra.jump_height]
    if getattr(env_cfg.env, "use_jump_distance", False):
        env_cfg.commands.jump_dist_cmd_range = [extra.jump_dist, extra.jump_dist]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.load_offset_range["x"] = (0.0, 0.0)
    env.load_offset_range["y"] = (0.0, 0.0)
    env.load_offset_range["z"] = (0.10, 0.10)
    obs, obs_history, commands, critic_obs = env.get_observations()

    resume_path = get_load_path(
        log_root, load_run=args.load_run, checkpoint=args.checkpoint
    )
    print(f"[jump-load] loading model from: {resume_path}")
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    loaded = torch.load(resume_path, map_location=env.device)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
    runner.alg.encoder.load_state_dict(loaded["encoder_state_dict"])
    policy = runner.get_inference_policy(device=env.device)
    encoder = runner.get_inference_encoder(device=env.device)
    control_encoder = None
    control_encoder_input_dim = None
    control_encoder_history_length = None
    control_encoder_frame_size = None
    estimator_history_length = None
    estimator_frame_size = None
    if "control_encoder_state_dict" in loaded:
        encoder_metadata = loaded.get("encoder_only_metadata", {})
        control_cfg = encoder_metadata.get(
            "control_encoder_cfg"
        )
        control_encoder = (
            MLP_Encoder(**control_cfg).to(env.device)
            if control_cfg is not None
            else copy.deepcopy(runner.alg.encoder)
        )
        control_encoder.load_state_dict(loaded["control_encoder_state_dict"])
        control_encoder.eval()
        control_encoder_input_dim = int(
            control_cfg["num_input_dim"]
            if control_cfg is not None
            else obs_history.shape[1]
        )
        if control_cfg is not None:
            control_encoder_history_length = int(
                control_cfg["obs_history_length"]
            )
            control_encoder_frame_size = (
                control_encoder_input_dim
                // control_encoder_history_length
            )
        estimator_history_length = int(
            encoder_metadata.get(
                "estimator_history_length",
                getattr(env.cfg.env, "obs_history_length", 1),
            )
        )
        estimator_frame_size = int(
            encoder_metadata.get(
                "estimator_frame_size",
                obs_history.shape[1] // estimator_history_length,
            )
        )

    history = {
        "true_mass": [],
        "estimate": [],
        "jump_state": [],
        "base_height": [],
        "base_vz": [],
        "load_on_body": [],
        "done": [],
    }
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    mass_scale = float(env.cfg.normalization.obs_scales.mass_scale)

    for _ in range(extra.record_steps):
        with torch.no_grad():
            estimate_scaled = encoder(obs_history, obs)
            if control_encoder is not None:
                if control_encoder_frame_size is not None:
                    estimator_frames = obs_history.reshape(
                        env.num_envs,
                        estimator_history_length,
                        estimator_frame_size,
                    )
                    control_history = estimator_frames[
                        :,
                        -control_encoder_history_length:,
                        :control_encoder_frame_size,
                    ].reshape(env.num_envs, control_encoder_input_dim)
                else:
                    control_history = obs_history[
                        :, -control_encoder_input_dim:
                    ]
                control_latent = control_encoder.encode(control_history)
            else:
                control_latent = estimate_scaled
            actions = policy(
                torch.cat((control_latent, obs, commands), dim=-1).detach()
            )

        if extra.payload_mode == "attached":
            true_mass = env.base_mass - env.base_mass0
            on_body = true_mass > 0.3
        elif hasattr(env, "is_load_on_body"):
            on_body = env.is_load_on_body(env_ids)
            if hasattr(env, "has_load"):
                on_body = on_body & env.has_load
            load_mass = env.load_mass_per_env
            true_mass = torch.where(on_body, load_mass, torch.zeros_like(load_mass))
        else:
            on_body = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            true_mass = torch.zeros_like(env.base_mass)

        history["true_mass"].append(true_mass.detach().cpu().numpy())
        history["estimate"].append(
            (estimate_scaled[:, 3] / mass_scale).detach().cpu().numpy()
        )
        history["jump_state"].append(env.jump_state.detach().cpu().numpy())
        history["base_height"].append(
            env.root_states[env.actor_indices, 2].detach().cpu().numpy()
        )
        history["base_vz"].append(env.base_lin_vel[:, 2].detach().cpu().numpy())
        history["load_on_body"].append(on_body.float().detach().cpu().numpy())

        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(
            actions.detach()
        )
        history["done"].append(dones.detach().cpu().numpy())

    arrays = {key: np.asarray(value) for key, value in history.items()}
    loaded_flight = (
        (arrays["jump_state"] == 2) & (arrays["true_mass"] > 0.3)
    ).sum(axis=0)
    loaded_fraction = arrays["load_on_body"].mean(axis=0)
    done_count = arrays["done"].sum(axis=0)
    score = loaded_flight + 10.0 * loaded_fraction - 2.0 * done_count
    representative_env = int(np.argmax(score))

    dt = float(env.dt)
    time_s = np.arange(extra.record_steps, dtype=float) * dt
    rep = {key: value[:, representative_env] for key, value in arrays.items()}
    phase_metrics = _phase_metrics(
        rep["true_mass"], rep["estimate"], rep["jump_state"]
    )
    phase_metrics_all = _phase_metrics(
        arrays["true_mass"].reshape(-1),
        arrays["estimate"].reshape(-1),
        arrays["jump_state"].reshape(-1),
    )
    summary = {
        "task": args.task,
        "load_run": args.load_run,
        "checkpoint": int(args.checkpoint),
        "payload_mode": extra.payload_mode,
        "frozen_control_encoder": control_encoder is not None,
        "load_mass_kg": float(extra.load_mass),
        "jump_height_m": float(extra.jump_height),
        "jump_dist_m": float(extra.jump_dist),
        "dt_s": dt,
        "representative_env": representative_env,
        "loaded_fraction": float(loaded_fraction[representative_env]),
        "loaded_flight_samples": int(loaded_flight[representative_env]),
        "loaded_flight_env_fraction": float(np.mean(loaded_flight > 0)),
        "done_count": int(done_count[representative_env]),
        "done_env_fraction": float(np.mean(done_count > 0)),
        "phase_metrics": phase_metrics,
        "phase_metrics_all_envs": phase_metrics_all,
    }

    os.makedirs(extra.output_dir, exist_ok=True)
    mat_path = os.path.join(extra.output_dir, "jump_load_estimation.mat")
    csv_path = os.path.join(extra.output_dir, "jump_load_estimation.csv")
    summary_path = os.path.join(extra.output_dir, "summary.json")
    savemat(
        mat_path,
        {
            "t": time_s,
            "true_mass": rep["true_mass"],
            "estimate": rep["estimate"],
            "jump_state": rep["jump_state"],
            "base_height": rep["base_height"],
            "base_vz": rep["base_vz"],
            "load_on_body": rep["load_on_body"],
            "done": rep["done"],
            "true_mass_all": arrays["true_mass"],
            "estimate_all": arrays["estimate"],
            "jump_state_all": arrays["jump_state"],
            "summary": summary,
        },
    )
    with open(csv_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "t",
                "true_mass",
                "estimate",
                "jump_state",
                "base_height",
                "base_vz",
                "load_on_body",
                "done",
            ]
        )
        for i in range(extra.record_steps):
            writer.writerow(
                [
                    time_s[i],
                    rep["true_mass"][i],
                    rep["estimate"][i],
                    int(rep["jump_state"][i]),
                    rep["base_height"][i],
                    rep["base_vz"][i],
                    rep["load_on_body"][i],
                    rep["done"][i],
                ]
            )
    with open(summary_path, "w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"[jump-load] saved {mat_path}")


if __name__ == "__main__":
    main()
