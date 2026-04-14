# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import csv
import math
import os
import re
from datetime import datetime

import isaacgym
from isaacgym import gymutil
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import task_registry

import torch


# Play-like convenient overrides. Set USE_LOCAL_DEFAULTS=True to make these take effect.
USE_LOCAL_DEFAULTS = True
LOCAL_DEFAULTS = {
    "task": "wheelfoot_flat",
    "experiment_name": "WF_TRON1A",
    "load_run": "test21",
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
        {
            "name": "--task",
            "type": str,
            "default": "a1_flat",
            "help": "Task name to evaluate.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume from a checkpoint.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Experiment name to load.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Run name override.",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Run directory to load. If -1, the last run is used.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Checkpoint number to load.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off.",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training.",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm.",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum training iterations.",
        },
        {
            "name": "--exptid",
            "type": str,
            "default": "",
            "help": "Experiment id suffix.",
        },
        {
            "name": "--eval_seconds",
            "type": float,
            "default": 20.0,
            "help": "Evaluation window in seconds.",
        },
        {
            "name": "--eval_steps",
            "type": int,
            "default": None,
            "help": "Evaluation window in steps. If set, overrides eval_seconds.",
        },
        {
            "name": "--checkpoint_start",
            "type": int,
            "default": 500,
            "help": "First checkpoint to evaluate.",
        },
        {
            "name": "--checkpoint_interval",
            "type": int,
            "default": 500,
            "help": "Checkpoint interval.",
        },
        {
            "name": "--checkpoint_end",
            "type": int,
            "default": -1,
            "help": "Last checkpoint to evaluate. -1 means all available checkpoints.",
        },
        {
            "name": "--output_csv",
            "type": str,
            "default": "",
            "help": "Optional CSV output path.",
        },
    ]

    args = gymutil.parse_arguments(
        description="Batch extra-loss evaluation", custom_parameters=custom_parameters
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
    commands_val[: min(commands_val.numel(), preset.numel())] = preset[: min(commands_val.numel(), preset.numel())]
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
    # mean_mse: mean over time of per-step MSE
    # strict_rmse: sqrt(mean_mse), equivalent to flattening all samples in time window
    # time_rms_mse: RMS over time of per-step MSE, emphasizes temporal spikes
    mean_mse, time_rms_mse = summarize_series(series)
    strict_rmse = math.sqrt(max(mean_mse, 0.0))
    return mean_mse, strict_rmse, time_rms_mse


def evaluate_loaded_runner(env, ppo_runner, args):
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    # Start each checkpoint evaluation from a clean env state.
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


def main():
    args = get_eval_args()
    args = apply_local_defaults(args)
    args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run

    env_cfg.env.episode_length_s = 60
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)

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

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name)
    run_dir = resolve_run_dir(log_root, load_run)
    available_checkpoints = list_available_checkpoints(run_dir)
    selected_checkpoints = select_checkpoints(
        available_checkpoints,
        args.checkpoint_start,
        args.checkpoint_interval,
        args.checkpoint_end,
    )

    if not selected_checkpoints:
        raise ValueError(
            f"No checkpoints matched start={args.checkpoint_start}, interval={args.checkpoint_interval}, end={args.checkpoint_end} in {run_dir}"
        )

    output_csv = args.output_csv
    if not output_csv:
        output_dir = os.path.join(run_dir, "extra_loss_eval")
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(
            output_dir,
            f"extra_loss_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    rows = []
    print(f"[extra_loss_eval] run_dir={run_dir}")
    print(f"[extra_loss_eval] checkpoints={selected_checkpoints}")
    print(f"[extra_loss_eval] output_csv={output_csv}")

    # Build simulator once. Re-load checkpoint weights each iteration to avoid
    # reinitializing PhysX Foundation in the same process.
    train_cfg.runner.resume = False
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )

    for checkpoint in selected_checkpoints:
        ckpt_path = os.path.join(run_dir, f"model_{checkpoint}.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[extra_loss_eval] skip missing checkpoint: {ckpt_path}")
            continue

        ppo_runner.load(ckpt_path, load_optimizer=False)
        results = evaluate_loaded_runner(env, ppo_runner, args)
        results["checkpoint"] = checkpoint
        rows.append(results)
        print(
            "[extra_loss_eval] "
            f"ckpt={checkpoint} "
            f"vel_rmse={results['extra_loss_vel_strict_rmse']:.6f} vel_time_rms_mse={results['extra_loss_vel_time_rms_mse']:.6f} "
            f"mass_rmse={results['extra_loss_mass_strict_rmse']:.6f} mass_time_rms_mse={results['extra_loss_mass_time_rms_mse']:.6f} "
            f"com_x_rmse={results['extra_loss_com_x_strict_rmse']:.6f} com_x_time_rms_mse={results['extra_loss_com_x_time_rms_mse']:.6f} "
            f"com_y_rmse={results['extra_loss_com_y_strict_rmse']:.6f} com_y_time_rms_mse={results['extra_loss_com_y_time_rms_mse']:.6f} "
            f"com_z_rmse={results['extra_loss_com_z_strict_rmse']:.6f} com_z_time_rms_mse={results['extra_loss_com_z_time_rms_mse']:.6f} "
            f"eval_seconds={results['eval_seconds']:.2f}"
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

    print(f"[extra_loss_eval] saved summary to {output_csv}")


if __name__ == "__main__":
    main()