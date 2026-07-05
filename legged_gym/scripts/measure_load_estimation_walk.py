#!/usr/bin/env python3
"""Measure load-mass estimation during walking for one trained WF policy.

For old wide-load policies that do not have the step/roll command split, the
script uses the ordinary forward velocity command (`fallback_vx`). For stepping
policies it can instead use `cmd_roll` + `cmd_step`.
"""
import argparse
import json
import os
import sys

import isaacgym  # noqa: F401
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.scripts.play import _apply_saved_env_cfg
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path


def _parse_edges(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _saved_env_dict(task, experiment_name, load_run):
    run_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", task, experiment_name, load_run)
    path = os.path.join(run_dir, "env_cfg.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _apply_gait_flags_from_saved(env_cfg, saved_env):
    saved = saved_env.get("env", {}) if isinstance(saved_env.get("env"), dict) else {}
    # Older wide-load runs predate these fields. Missing means no step/roll split.
    for key in ("use_gait_stepping", "use_gait_phase"):
        setattr(env_cfg.env, key, bool(saved.get(key, False)))


def _variant_name(env_cfg, train_cfg):
    qs = bool(getattr(env_cfg.env, "use_qs_in_obs", False))
    resid = bool(getattr(train_cfg.algorithm, "use_load_residual_estimation", False))
    torq = bool(getattr(env_cfg.env, "use_torques_in_obs", True))
    if qs and resid:
        return "Model-guided"
    if (not qs) and (not resid) and (not torq):
        return "RL-only"
    if qs and (not resid):
        return "Estimate-guided"
    return "Source-guided"


def _bin_stats(load, est, edges):
    rows = []
    load = np.asarray(load)
    est = np.asarray(est)
    err = est - load
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (load >= lo) & (load < hi)
        if not np.any(m):
            rows.append((lo, hi, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        e = err[m]
        rows.append((
            lo, hi, int(m.sum()),
            float(np.nanmean(load[m])),
            float(np.sqrt(np.nanmean(e ** 2))),
            float(np.nanmean(np.abs(e))),
            float(np.nanmean(e)),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--load_edges", default="5,9,13,17,21,25.0001")
    parser.add_argument("--load_min", type=float, default=5.0)
    parser.add_argument("--load_max", type=float, default=25.0)
    parser.add_argument("--cmd_roll", type=float, default=0.0)
    parser.add_argument("--cmd_step", type=float, default=0.2)
    parser.add_argument("--fallback_vx", type=float, default=0.2)
    parser.add_argument("--warmup_steps", type=int, default=350)
    parser.add_argument("--record_steps", type=int, default=500)
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name)
    _apply_saved_env_cfg(env_cfg, train_cfg, args, log_root)
    saved_env = _saved_env_dict(args.task, train_cfg.runner.experiment_name, args.load_run)
    _apply_gait_flags_from_saved(env_cfg, saved_env)

    env_cfg.env.num_envs = int(args.num_envs or 128)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.terrain.critic_measure_heights = False
    env_cfg.domain_rand.add_random_load = True
    env_cfg.domain_rand.per_env_load_mass = True
    env_cfg.domain_rand.add_load_range = [float(extra.load_min), float(extra.load_max)]
    env_cfg.domain_rand.load_start_time_s = 0.5
    env_cfg.domain_rand.load_duration_range_s = [1.0e6, 1.0e6]
    env_cfg.domain_rand.load_interval_range_s = [1.0e6, 1.0e6]
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False
    if hasattr(env_cfg, "noise") and hasattr(env_cfg.noise, "add_noise"):
        env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history, commands, critic_obs = env.get_observations()

    resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    print(f"[loadwalk] loading model from: {resume_path}")
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    loaded = torch.load(resume_path, map_location=env.device)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
    runner.alg.encoder.load_state_dict(loaded["encoder_state_dict"])
    policy = runner.get_inference_policy(device=env.device)
    encoder = runner.get_inference_encoder(device=env.device)

    has_step_cmd = bool(getattr(env.cfg.env, "use_gait_stepping", False)) and hasattr(env, "cmd_step")
    if has_step_cmd:
        cmd_desc = f"cmd_roll={extra.cmd_roll:g}, cmd_step={extra.cmd_step:g}"
    else:
        cmd_desc = f"ordinary cmd_vx={extra.fallback_vx:g} (checkpoint has no cmd_step/cmd_roll split)"
    print(
        f"[loadwalk] variant={_variant_name(env.cfg, train_cfg)} run={args.load_run} "
        f"ckpt={args.checkpoint} envs={env.num_envs} loads=[{extra.load_min:g},{extra.load_max:g}] "
        f"{cmd_desc}"
    )

    load_est_samples = []
    true_mass_samples = []
    base_vx_samples = []
    done_samples = []
    on_body_samples = []
    total_steps = int(extra.warmup_steps + extra.record_steps)
    for i in range(total_steps):
        if has_step_cmd:
            env.commands[:, :] = 0.0
            env.commands[:, 0] = float(extra.cmd_roll)
            env.cmd_step[:, 0] = float(extra.cmd_step)
        else:
            env.commands[:, :] = 0.0
            env.commands[:, 0] = float(extra.fallback_vx)

        with torch.no_grad():
            est = encoder(obs_history, obs)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(actions.detach())

        if i >= extra.warmup_steps:
            mass_scale = float(env.cfg.normalization.obs_scales.mass_scale)
            enc_mass = (est[:, 3] / mass_scale).detach().cpu().numpy()
            if hasattr(env, "load_mass_per_env"):
                true_mass = env.load_mass_per_env.detach().cpu().numpy()
            else:
                true_mass = np.full((env.num_envs,), float(env.load_mass), dtype=np.float32)
            load_est_samples.append(enc_mass.copy())
            true_mass_samples.append(true_mass.copy())
            base_vx_samples.append(env.base_lin_vel[:, 0].detach().cpu().numpy().copy())
            done_samples.append(dones.detach().cpu().numpy().copy())
            if hasattr(env, "load_on_body"):
                on_body_samples.append(env.load_on_body.detach().cpu().numpy().copy())

    # Per-env summary: true mass is fixed per env; estimate is averaged over the record window.
    est_mean = np.asarray(load_est_samples).mean(axis=0)
    true_mean = np.asarray(true_mass_samples).mean(axis=0)
    base_vx = np.asarray(base_vx_samples).reshape(-1)
    dones = np.asarray(done_samples).reshape(-1).astype(bool)
    on_body_frac = float(np.asarray(on_body_samples).mean()) if on_body_samples else np.nan
    edges = _parse_edges(extra.load_edges)
    rows = _bin_stats(true_mean, est_mean, edges)
    err = est_mean - true_mean

    print("\n[loadwalk] overall")
    print(f"  RMSE={np.sqrt(np.mean(err ** 2)):.3f} kg  MAE={np.mean(np.abs(err)):.3f} kg  bias={np.mean(err):+.3f} kg")
    print(f"  base_vx_mean={np.mean(base_vx):+.3f} m/s  done_frac={np.mean(dones):.4f}  load_on_body_frac={on_body_frac:.4f}")
    print("\n[loadwalk] by load bin")
    print("| load bin kg | n env | true mean kg | RMSE kg | MAE kg | bias kg |")
    print("|---|---:|---:|---:|---:|---:|")
    for lo, hi, n, tru, rmse, mae, bias in rows:
        if n == 0:
            print(f"| {lo:g}-{hi:g} | 0 | — | — | — | — |")
        else:
            print(f"| {lo:g}-{hi:g} | {n} | {tru:.2f} | {rmse:.2f} | {mae:.2f} | {bias:+.2f} |")


if __name__ == "__main__":
    main()
