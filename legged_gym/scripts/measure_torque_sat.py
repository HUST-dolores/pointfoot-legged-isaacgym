#!/usr/bin/env python3
"""Measure per-joint torque + saturation for a FIXED k under a fixed payload.
Answers: does the knee/hip actually hit the k*80 N·m limit when holding the load?
Usage (workstation):
  ROBOT_TYPE=WF_TRON1A python legged_gym/scripts/measure_torque_sat.py \
     --task=wheelfoot_flat --load_run Jun30_23-49-24_ --checkpoint 16000 \
     --num_envs 256 --headless --load_mass_min 50 --load_mass_max 50 --cmd_vx 0.5
Set FORCE_K env var to fix all envs to that k (default 0.4).
"""
import os

import isaacgym  # noqa: F401  (must precede torch)
from isaacgym.torch_utils import *  # noqa: F401,F403
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR

import numpy as np
import torch

from legged_gym.scripts.play import _apply_saved_env_cfg
from legged_gym.scripts.eval_motor_fk import (
    _load_saved_env_dict, _align_motor_design_cfg, _make_eval_env_cfg,
    CMD_VY, CMD_YAW,
)

FORCE_K = float(os.environ.get("FORCE_K", "0.4"))
N_STEPS = 1200
SETTLE = 250  # skip transient + load-application window


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name)
    _apply_saved_env_cfg(env_cfg, train_cfg, args, log_root)
    saved_env = _load_saved_env_dict(args, log_root)
    _align_motor_design_cfg(env_cfg, saved_env)
    cond = _make_eval_env_cfg(env_cfg, args)
    cmd_vx = float(getattr(args, "cmd_vx", 0.0) or 0.0) or 0.5

    # FORCE all envs to one k (both mass and torque-limit) so the test is clean
    env_cfg.domain_rand.motor_torque_scale_range = [FORCE_K, FORCE_K]
    print(f"[sat] FORCING all envs to k={FORCE_K} (limit = torque_limits*{FORCE_K})")

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if cond["payload_on"] and hasattr(env, "load_offset_range"):
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

    # per-dof effective torque limit
    dof_names = list(env.dof_names) if hasattr(env, "dof_names") else [f"dof{i}" for i in range(env.num_dof)]
    tl = env.torque_limits.detach().float()
    if tl.dim() == 1:
        tl = tl.view(1, -1)
    kscale = env.motor_torque_limit_scale.detach().float().view(-1, 1)   # [N,1]
    eff_limit = (tl * kscale)                                            # [N, num_dof]
    print(f"[sat] dof order: {dof_names}")
    print(f"[sat] torque_limits (nominal): {tl.view(-1).cpu().numpy()}")
    print(f"[sat] k (per-env) sample: {kscale.view(-1)[:3].cpu().numpy()}  -> eff knee/hip limit ~{float(eff_limit[0].max()):.1f} Nm")

    Nd = env.num_dof
    abs_sum = torch.zeros(Nd, device=env.device)
    sat_sum = torch.zeros(Nd, device=env.device)
    max_abs = torch.zeros(Nd, device=env.device)
    pos_sum = torch.zeros(Nd, device=env.device)
    cnt = 0
    all_abs = []  # for p95 (subsample)

    for i in range(N_STEPS):
        with torch.no_grad():
            est = encoder(obs_history, obs)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        env.commands[:, :] = commands_val
        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(actions.detach())
        env.commands[:, :] = commands_val
        if i >= SETTLE:
            tau = env.torques.detach().float()               # [N, num_dof]
            a = tau.abs()
            abs_sum += a.mean(0)
            max_abs = torch.maximum(max_abs, a.max(0).values)
            sat = (a >= 0.98 * eff_limit).float()
            sat_sum += sat.mean(0)
            pos_sum += env.dof_pos.detach().float().mean(0)
            cnt += 1
            if i % 200 == 0:
                all_abs.append(a.cpu().numpy())

    mean_abs = (abs_sum / cnt).cpu().numpy()
    sat_frac = (sat_sum / cnt).cpu().numpy()
    maxa = max_abs.cpu().numpy()
    mean_pos = (pos_sum / cnt).cpu().numpy()
    eff = eff_limit[0].cpu().numpy()
    allA = np.concatenate(all_abs, 0) if all_abs else np.zeros((1, Nd))
    p95 = np.percentile(allA, 95, axis=0)

    print("\n[sat] ===================== PER-JOINT TORQUE @ k=%.2f, load=%.0fkg, vx=%.1f =====================" % (
        FORCE_K, cond["load_mass"], cmd_vx))
    print(f"[sat] {'joint':<14}{'eff_lim':>8}{'mean|t|':>9}{'p95|t|':>9}{'max|t|':>9}{'SAT%':>8}{'meanpos':>9}")
    for j in range(Nd):
        print(f"[sat] {dof_names[j]:<14}{eff[j]:>8.1f}{mean_abs[j]:>9.2f}{p95[j]:>9.2f}{maxa[j]:>9.2f}{100*sat_frac[j]:>8.1f}{mean_pos[j]:>9.3f}")
    # focus summary on leg holding joints (hip/knee)
    leg_idx = [j for j, n in enumerate(dof_names) if ("hip" in n.lower() or "knee" in n.lower())]
    if leg_idx:
        hk_sat = float(np.mean([sat_frac[j] for j in leg_idx]))
        print(f"\n[sat] HIP+KNEE mean saturation = {100*hk_sat:.1f}%  (fraction of time the leg motor is pinned at the {FORCE_K}x limit)")
        print(f"[sat] VERDICT: {'SATURATION-BOUND — the k=%.2f limit BINDS while holding the load (user was right)'%FORCE_K if hk_sat>0.03 else 'limit rarely binds — load held with torque headroom to spare (posture keeps lever arm small)'}")


if __name__ == "__main__":
    main()
