#!/usr/bin/env python3
"""Reconcile the f(k) eval vs the interactive play collapse.
Measures mean EPISODE LENGTH + fall fraction + knee saturation for a FIXED k,
under two load-application modes:
  LOAD_MODE=held    -> load smoothly held from 0.5s, CENTERED  (my f(k) eval condition, quasi-static)
  LOAD_MODE=dynamic -> load slams on/off (dur[3,4], int[5,6]), default offset (the play condition, transient)
Usage:
  ROBOT_TYPE=WF_TRON1A LOAD_MODE=dynamic FORCE_K=0.4 python legged_gym/scripts/measure_episode.py \
     --task=wheelfoot_flat --load_run Jun30_23-49-24_ --checkpoint 16000 --num_envs 512 --headless \
     --load_mass_min 50 --load_mass_max 50 --cmd_vx 0.5
"""
import os

import isaacgym  # noqa: F401
from isaacgym.torch_utils import *  # noqa: F401,F403
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR

import numpy as np
import torch

from legged_gym.scripts.play import _apply_saved_env_cfg
from legged_gym.scripts.eval_motor_fk import (
    _load_saved_env_dict, _align_motor_design_cfg, _make_eval_env_cfg, CMD_VY, CMD_YAW,
)

FORCE_K = float(os.environ.get("FORCE_K", "0.4"))
LOAD_MODE = os.environ.get("LOAD_MODE", "dynamic")   # held | dynamic
N_STEPS = 2000   # one full 40s episode


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name)
    _apply_saved_env_cfg(env_cfg, train_cfg, args, log_root)
    saved_env = _load_saved_env_dict(args, log_root)
    _align_motor_design_cfg(env_cfg, saved_env)
    cond = _make_eval_env_cfg(env_cfg, args)
    cmd_vx = float(getattr(args, "cmd_vx", 0.0) or 0.0) or 0.5

    env_cfg.domain_rand.motor_torque_scale_range = [FORCE_K, FORCE_K]
    lm = float(getattr(args, "load_mass_min", 50.0) or 50.0)
    # load application mode
    env_cfg.domain_rand.add_random_load = True
    env_cfg.domain_rand.add_load_range = [lm, lm]
    if LOAD_MODE == "held":
        env_cfg.domain_rand.load_start_time_s = 0.5
        env_cfg.domain_rand.load_duration_range_s = [1.0e6, 1.0e6]
        env_cfg.domain_rand.load_interval_range_s = [1.0e6, 1.0e6]
    else:  # dynamic (play-like): keep the training on/off timing
        env_cfg.domain_rand.load_start_time_s = 0.5
        env_cfg.domain_rand.load_duration_range_s = [3.0, 4.0]
        env_cfg.domain_rand.load_interval_range_s = [5.0, 6.0]
    print(f"[ep] FORCE_K={FORCE_K}  LOAD_MODE={LOAD_MODE}  load={lm}kg")

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if LOAD_MODE == "held" and hasattr(env, "load_offset_range"):
        env.load_offset_range["x"] = (0.0, 0.0)
        env.load_offset_range["y"] = (0.0, 0.0)
        env.load_offset_range["z"] = (0.10, 0.10)

    commands_val = to_torch([cmd_vx, CMD_VY, CMD_YAW], device=env.device)
    obs, obs_history, commands, critic_obs = env.get_observations()
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)
    env.commands[:, :] = commands_val

    N = env.num_envs
    dev = env.device
    dof_names = list(env.dof_names)
    knee_idx = [j for j, n in enumerate(dof_names) if "knee" in n.lower()]
    tl = env.torque_limits.detach().float().view(1, -1)
    eff_knee = float((tl[0, knee_idx] * FORCE_K).max())

    cur_len = torch.zeros(N, device=dev)
    ep_len_sum = torch.zeros((), device=dev)
    ep_count = torch.zeros((), device=dev)
    fall_count = torch.zeros((), device=dev)
    timeout_count = torch.zeros((), device=dev)
    knee_sat_sum = torch.zeros((), device=dev)
    steps_counted = 0
    dt = float(getattr(env, "dt", 0.02))

    for i in range(N_STEPS):
        with torch.no_grad():
            est = encoder(obs_history, obs)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        env.commands[:, :] = commands_val
        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(actions.detach())
        env.commands[:, :] = commands_val
        d = dones.view(-1).bool()
        cur_len += 1.0
        # knee saturation after load is on (i>60 ~ 1.2s)
        if i > 60:
            tau = env.torques.detach().float()
            knee_sat_sum += (tau[:, knee_idx].abs() >= 0.98 * eff_knee).float().mean()
            steps_counted += 1
        if d.any():
            to = getattr(env, "time_out_buf", None)
            to = to.view(-1).bool() if to is not None else torch.zeros_like(d)
            fell = d & (~to)
            tout = d & to
            ep_len_sum += cur_len[d].sum()
            ep_count += d.sum()
            fall_count += fell.sum()
            timeout_count += tout.sum()
            cur_len[d] = 0.0

    mean_ep_len = float(ep_len_sum / ep_count.clamp(min=1))
    ep_c = int(ep_count.item())
    print(f"[ep] ===== k={FORCE_K} load={lm}kg mode={LOAD_MODE} : {N} envs x {N_STEPS} steps =====")
    print(f"[ep]   completed episodes (resets): {ep_c}   (falls={int(fall_count)} timeouts={int(timeout_count)})")
    print(f"[ep]   MEAN EPISODE LENGTH = {mean_ep_len:.0f} steps = {mean_ep_len*dt:.1f} s   (cap = {N_STEPS} = {N_STEPS*dt:.0f}s)")
    print(f"[ep]   fall fraction of resets = {100*float(fall_count/ep_count.clamp(min=1)):.0f}%")
    print(f"[ep]   knee saturation (load on) = {100*float(knee_sat_sum/max(steps_counted,1)):.1f}%   eff_knee_lim={eff_knee:.0f}Nm")


if __name__ == "__main__":
    main()
