# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import csv
import math
import os
import re
from datetime import datetime

import isaacgym
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import task_registry

import torch


# This script is intended for dual-head encoder checkpoints only.
USE_LOCAL_DEFAULTS = True
LOCAL_DEFAULTS = {
    "task": "wheelfoot_flat",
    "experiment_name": "WF_TRON1A",
    "load_run": "test22",
    "checkpoint_start": 500,
    "checkpoint_interval": 500,
    "checkpoint_end": -1,
    "eval_seconds": 20.0,
    "eval_steps": None,
    "num_envs": 50,
    "output_csv": "",
}


def get_eval_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "wheelfoot_flat"},
        {"name": "--resume", "action": "store_true", "default": False},
        {"name": "--experiment_name", "type": str},
        {"name": "--run_name", "type": str},
        {"name": "--load_run", "type": str},
        {"name": "--checkpoint", "type": int},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--horovod", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--num_envs", "type": int},
        {"name": "--seed", "type": int},
        {"name": "--max_iterations", "type": int},
        {"name": "--exptid", "type": str, "default": ""},
        {"name": "--eval_seconds", "type": float, "default": 20.0},
        {"name": "--eval_steps", "type": int, "default": None},
        {"name": "--checkpoint_start", "type": int, "default": 500},
        {"name": "--checkpoint_interval", "type": int, "default": 500},
        {"name": "--checkpoint_end", "type": int, "default": -1},
        {"name": "--output_csv", "type": str, "default": ""},
    ]

    args = gymutil.parse_arguments(
        description="Batch extra-loss evaluation for dual-head encoder",
        custom_parameters=custom_parameters,
    )

    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def apply_local_defaults(args):
    if not USE_LOCAL_DEFAULTS:
        return args
    for key, value in LOCAL_DEFAULTS.items():
        if value is not None:
            setattr(args, key, value)
    return args


def resolve_run_dir(log_root, load_run):
    if load_run == -1 or str(load_run) == "-1":
        runs = [name for name in os.listdir(log_root) if name != "exported"]
        if not runs:
            raise ValueError(f"No runs found in {log_root}")
        runs.sort()
        return os.path.join(log_root, runs[-1])
    return os.path.join(log_root, str(load_run))


def list_available_checkpoints(run_dir):
    checkpoints = []
    pattern = re.compile(r"model_(\d+)\.pt$")
    for file_name in os.listdir(run_dir):
        match = pattern.match(file_name)
        if match:
            checkpoints.append(int(match.group(1)))
    checkpoints.sort()
    return checkpoints


def select_checkpoints(available_checkpoints, start, interval, end):
    selected = []
    for checkpoint in available_checkpoints:
        if checkpoint < start:
            continue
        if end != -1 and checkpoint > end:
            continue
        if (checkpoint - start) % interval != 0:
            continue
        selected.append(checkpoint)
    return selected


def build_commands(env, robot_type):
    commands_val = torch.zeros(env.num_commands, device=env.device)
    if robot_type and robot_type.startswith("PF"):
        preset = torch.tensor([0.5, 0.0, 0.0, 0.0], device=env.device)
    elif robot_type == "WF_TRON1A":
        preset = torch.tensor([0.0, 0.0, 0.0], device=env.device)
    else:
        preset = torch.zeros(env.num_commands, device=env.device)
    n = min(commands_val.numel(), preset.numel())
    commands_val[:n] = preset[:n]
    return commands_val


def compute_step_metrics(est, critic_obs):
    vel_mse = (est[:, 0:3] - critic_obs[:, 0:3]).pow(2).mean().item()
    mass_mse = (est[:, 3:4] - critic_obs[:, 3:4]).pow(2).mean().item()
    com_x_mse = (est[:, 4] - critic_obs[:, 4]).pow(2).mean().item()
    com_y_mse = (est[:, 5] - critic_obs[:, 5]).pow(2).mean().item()
    com_z_mse = (est[:, 6] - critic_obs[:, 6]).pow(2).mean().item()
    return {
        "extra_loss_vel": vel_mse,
        "extra_loss_mass": mass_mse,
        "extra_loss_com_x": com_x_mse,
        "extra_loss_com_y": com_y_mse,
        "extra_loss_com_z": com_z_mse,
    }


def summarize_series(series):
    if not series:
        return 0.0, 0.0
    mean_value = sum(series) / len(series)
    rms_value = math.sqrt(sum(value * value for value in series) / len(series))
    return mean_value, rms_value


def summarize_mse_series(series):
    mean_mse, time_rms_mse = summarize_series(series)
    strict_rmse = math.sqrt(max(mean_mse, 0.0))
    return mean_mse, strict_rmse, time_rms_mse


def evaluate_loaded_runner(env, ppo_runner, args):
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    env.reset()
    obs, obs_history, commands, critic_obs = env.get_observations()
    robot_type = os.getenv("ROBOT_TYPE")
    commands_val = build_commands(env, robot_type)

    num_steps = args.eval_steps if args.eval_steps is not None else max(1, int(args.eval_seconds / env.dt))

    metric_history = {
        "extra_loss_vel": [],
        "extra_loss_mass": [],
        "extra_loss_com_x": [],
        "extra_loss_com_y": [],
        "extra_loss_com_z": [],
    }

    with torch.no_grad():
        for _ in range(num_steps):
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            env.commands[:, :] = commands_val
            obs, _, _, _, obs_history, commands, critic_obs = env.step(actions.detach())
            step_metrics = compute_step_metrics(est, critic_obs)
            for key, value in step_metrics.items():
                metric_history[key].append(value)

    results = {"num_steps": num_steps, "eval_seconds": num_steps * env.dt}
    for key, values in metric_history.items():
        mean_mse, strict_rmse, time_rms_mse = summarize_mse_series(values)
        results[f"{key}_ave"] = mean_mse
        results[f"{key}_strict_rmse"] = strict_rmse
        results[f"{key}_time_rms_mse"] = time_rms_mse
    return results


def checkpoint_is_dual_head(encoder_state_dict):
    return any(key.startswith("encoder.backbone.") for key in encoder_state_dict.keys())


def load_checkpoint_dual_safe(ppo_runner, ckpt_path):
    loaded = torch.load(ckpt_path)
    model_sd = loaded["model_state_dict"]
    encoder_sd = loaded["encoder_state_dict"]

    if not checkpoint_is_dual_head(encoder_sd):
        return False, "checkpoint encoder is single-head, not dual-head"

    ppo_runner.alg.actor_critic.load_state_dict(model_sd)
    ppo_runner.alg.encoder.load_state_dict(encoder_sd)
    ppo_runner.current_learning_iteration = loaded.get("iter", 0)
    return True, "ok"


def main():
    args = apply_local_defaults(get_eval_args())
    args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run

    env_cfg.env.episode_length_s = 60
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, args.num_envs if args.num_envs is not None else 50)

    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.terrain.max_init_terrain_level = 4
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 3
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.domain_rand.load_debug = True

    # Force dual-head encoder for this script.
    train_cfg.MLP_Encoder.use_dual_head = True

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name)
    run_dir = resolve_run_dir(log_root, load_run)
    selected_checkpoints = select_checkpoints(
        list_available_checkpoints(run_dir),
        args.checkpoint_start,
        args.checkpoint_interval,
        args.checkpoint_end,
    )
    if not selected_checkpoints:
        raise ValueError("No checkpoints matched the selected range")

    output_csv = args.output_csv
    if not output_csv:
        output_dir = os.path.join(run_dir, "extra_loss_eval")
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(
            output_dir,
            f"extra_loss_summary_dual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    train_cfg.runner.resume = False
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    rows = []
    print(f"[dual_eval] run_dir={run_dir}")
    print(f"[dual_eval] checkpoints={selected_checkpoints}")

    for checkpoint in selected_checkpoints:
        ckpt_path = os.path.join(run_dir, f"model_{checkpoint}.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[dual_eval] skip missing checkpoint: {ckpt_path}")
            continue

        ok, reason = load_checkpoint_dual_safe(ppo_runner, ckpt_path)
        if not ok:
            print(f"[dual_eval] skip ckpt={checkpoint}: {reason}")
            continue

        results = evaluate_loaded_runner(env, ppo_runner, args)
        results["checkpoint"] = checkpoint
        rows.append(results)
        print(
            "[dual_eval] "
            f"ckpt={checkpoint} "
            f"vel_rmse={results['extra_loss_vel_strict_rmse']:.6f} "
            f"mass_rmse={results['extra_loss_mass_strict_rmse']:.6f} "
            f"com_x_rmse={results['extra_loss_com_x_strict_rmse']:.6f} "
            f"com_y_rmse={results['extra_loss_com_y_strict_rmse']:.6f} "
            f"com_z_rmse={results['extra_loss_com_z_strict_rmse']:.6f}"
        )

    if hasattr(env, "close"):
        env.close()
    del ppo_runner
    del env
    torch.cuda.empty_cache()

    fieldnames = [
        "checkpoint",
        "num_steps",
        "eval_seconds",
        "extra_loss_vel_ave",
        "extra_loss_vel_strict_rmse",
        "extra_loss_vel_time_rms_mse",
        "extra_loss_mass_ave",
        "extra_loss_mass_strict_rmse",
        "extra_loss_mass_time_rms_mse",
        "extra_loss_com_x_ave",
        "extra_loss_com_x_strict_rmse",
        "extra_loss_com_x_time_rms_mse",
        "extra_loss_com_y_ave",
        "extra_loss_com_y_strict_rmse",
        "extra_loss_com_y_time_rms_mse",
        "extra_loss_com_z_ave",
        "extra_loss_com_z_strict_rmse",
        "extra_loss_com_z_time_rms_mse",
    ]

    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[dual_eval] saved summary to {output_csv}")


if __name__ == "__main__":
    main()
