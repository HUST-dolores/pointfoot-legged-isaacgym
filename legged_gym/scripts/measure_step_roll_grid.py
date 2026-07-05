#!/usr/bin/env python3
"""Measure realized roll/step velocities over a command grid.

Loads a trained WF policy once, then sweeps command pairs:
  cmd_roll = env.commands[:, 0]
  cmd_step = env.cmd_step[:, 0]

For each pair it resets all envs, warms up, records the last window, and prints
Markdown/CSV tables with realized v_roll and v_step.
"""
import argparse
import os
import sys

import isaacgym  # noqa: F401
from isaacgym.torch_utils import to_torch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.scripts.play import _apply_saved_env_cfg
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path


def _parse_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _fmt_pair(v_roll, v_step):
    return f"{v_roll:+.2f}/{v_step:+.2f}"


def _print_pair_table(results, roll_cmds, step_cmds):
    print("\n[grid] realized velocities table: cell = v_roll / v_step [m/s]")
    header = "| cmd_step \\ cmd_roll | " + " | ".join(f"{r:+g}" for r in roll_cmds) + " |"
    sep = "|---" * (len(roll_cmds) + 1) + "|"
    print(header)
    print(sep)
    for s in step_cmds:
        row = [f"{s:+g}"]
        for r in roll_cmds:
            m = results[(r, s)]
            row.append(_fmt_pair(m["v_roll_mean"], m["v_step_mean"]))
        print("| " + " | ".join(row) + " |")


def _print_error_table(results, roll_cmds, step_cmds):
    print("\n[grid] absolute tracking error table: cell = |roll_err| / |step_err| [m/s]")
    header = "| cmd_step \\ cmd_roll | " + " | ".join(f"{r:+g}" for r in roll_cmds) + " |"
    sep = "|---" * (len(roll_cmds) + 1) + "|"
    print(header)
    print(sep)
    for s in step_cmds:
        row = [f"{s:+g}"]
        for r in roll_cmds:
            m = results[(r, s)]
            row.append(f"{abs(m['v_roll_mean'] - r):.2f}/{abs(m['v_step_mean'] - s):.2f}")
        print("| " + " | ".join(row) + " |")


def _print_csv(results, roll_cmds, step_cmds):
    print("\n[grid] CSV")
    print("cmd_roll,cmd_step,v_roll_mean,v_step_mean,base_vx_mean,v_roll_std,v_step_std,base_vx_std,done_frac")
    for s in step_cmds:
        for r in roll_cmds:
            m = results[(r, s)]
            print(
                f"{r:.6g},{s:.6g},"
                f"{m['v_roll_mean']:.6g},{m['v_step_mean']:.6g},{m['base_vx_mean']:.6g},"
                f"{m['v_roll_std']:.6g},{m['v_step_std']:.6g},{m['base_vx_std']:.6g},"
                f"{m['done_frac']:.6g}"
            )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--roll_cmds", default="-2,-1,0,1,2")
    parser.add_argument("--step_cmds", default="-0.2,0,0.2,0.4,0.6,0.8")
    parser.add_argument("--warmup_steps", type=int, default=250)
    parser.add_argument("--record_steps", type=int, default=250)
    parser.add_argument("--plane", action="store_true", default=True)
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()

    roll_cmds = _parse_list(extra.roll_cmds)
    step_cmds = _parse_list(extra.step_cmds)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name)
    _apply_saved_env_cfg(env_cfg, train_cfg, args, log_root)

    env_cfg.env.num_envs = int(args.num_envs or 64)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.terrain.critic_measure_heights = False
    env_cfg.domain_rand.add_random_load = False
    env_cfg.domain_rand.push_robots = False
    if hasattr(env_cfg, "noise") and hasattr(env_cfg.noise, "add_noise"):
        env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history, commands, critic_obs = env.get_observations()

    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    print(f"[grid] loading model from: {resume_path}")
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    loaded = torch.load(resume_path, map_location=env.device)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
    runner.alg.encoder.load_state_dict(loaded["encoder_state_dict"])
    policy = runner.get_inference_policy(device=env.device)
    encoder = runner.get_inference_encoder(device=env.device)

    if not hasattr(env, "cmd_step"):
        raise RuntimeError("env has no cmd_step; use a stepping-gait config/checkpoint.")

    all_ids = torch.arange(env.num_envs, device=env.device)
    results = {}
    total_steps = int(extra.warmup_steps + extra.record_steps)
    print(
        f"[grid] run={args.load_run} ckpt={args.checkpoint} envs={env.num_envs} "
        f"warmup={extra.warmup_steps} record={extra.record_steps} dt={env.dt}"
    )

    for step_cmd in step_cmds:
        for roll_cmd in roll_cmds:
            env.reset_idx(all_ids)
            obs, obs_history, commands, critic_obs = env.get_observations()
            env.commands[:, :] = 0.0
            env.commands[:, 0] = float(roll_cmd)
            env.cmd_step[:, 0] = float(step_cmd)
            vals = {"v_roll": [], "v_step": [], "base_vx": [], "done": []}

            for i in range(total_steps):
                with torch.no_grad():
                    est = encoder(obs_history, obs)
                    actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
                # Keep commands fixed across env-internal resampling/reset.
                env.commands[:, :] = 0.0
                env.commands[:, 0] = float(roll_cmd)
                env.cmd_step[:, 0] = float(step_cmd)
                obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(actions.detach())
                env.commands[:, :] = 0.0
                env.commands[:, 0] = float(roll_cmd)
                env.cmd_step[:, 0] = float(step_cmd)
                if i >= extra.warmup_steps:
                    vals["v_roll"].append(env.v_roll.detach().cpu().numpy().copy())
                    vals["v_step"].append(env.v_step.detach().cpu().numpy().copy())
                    vals["base_vx"].append(env.base_lin_vel[:, 0].detach().cpu().numpy().copy())
                    vals["done"].append(dones.detach().cpu().numpy().copy())

            out = {}
            for k in ("v_roll", "v_step", "base_vx"):
                arr = np.asarray(vals[k]).reshape(-1)
                out[f"{k}_mean"] = float(np.mean(arr))
                out[f"{k}_std"] = float(np.std(arr))
            done_arr = np.asarray(vals["done"]).reshape(-1)
            out["done_frac"] = float(np.mean(done_arr.astype(np.float32)))
            results[(roll_cmd, step_cmd)] = out
            print(
                f"[grid] roll={roll_cmd:+.2f} step={step_cmd:+.2f} -> "
                f"v_roll={out['v_roll_mean']:+.3f} v_step={out['v_step_mean']:+.3f} "
                f"base_vx={out['base_vx_mean']:+.3f} done={out['done_frac']:.3f}"
            )

    _print_pair_table(results, roll_cmds, step_cmds)
    _print_error_table(results, roll_cmds, step_cmds)
    _print_csv(results, roll_cmds, step_cmds)


if __name__ == "__main__":
    main()
