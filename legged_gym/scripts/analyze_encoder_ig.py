# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Integrated Gradients analysis for the wheelfoot encoder.

This script collects a rollout dataset, uses the dataset mean
`encoder_obs_history` as the Integrated Gradients baseline, then reports which
observation groups contribute most to encoder outputs.
"""

import csv
import os
from datetime import datetime

import isaacgym  # noqa: F401
from isaacgym import gymutil

import numpy as np
import torch
try:
    from scipy.io import savemat
except ImportError:
    savemat = None

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_load_path, task_registry


def get_analysis_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "wheelfoot_flat"},
        {"name": "--resume", "action": "store_true", "default": False},
        {"name": "--experiment_name", "type": str},
        {"name": "--run_name", "type": str},
        {"name": "--load_run", "type": str},
        {"name": "--checkpoint", "type": int},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--render", "action": "store_true", "default": False},
        {"name": "--horovod", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--num_envs", "type": int, "default": 50},
        {"name": "--seed", "type": int},
        {"name": "--max_iterations", "type": int},
        {"name": "--exptid", "type": str, "default": ""},
        {
            "name": "--rollout_steps",
            "type": int,
            "default": 400,
            "help": "Steps used to collect candidate encoder histories.",
        },
        {
            "name": "--warmup_steps",
            "type": int,
            "default": 20,
            "help": "Initial rollout steps skipped before sampling histories.",
        },
        {
            "name": "--sample_stride",
            "type": int,
            "default": 2,
            "help": "Collect histories every N rollout steps.",
        },
        {
            "name": "--ig_samples",
            "type": int,
            "default": 1024,
            "help": "Maximum number of histories used for Integrated Gradients.",
        },
        {
            "name": "--ig_steps",
            "type": int,
            "default": 32,
            "help": "Number of interpolation points from baseline to sample.",
        },
        {
            "name": "--ig_batch_size",
            "type": int,
            "default": 64,
            "help": "Batch size for Integrated Gradients.",
        },
        {
            "name": "--window_steps",
            "type": int,
            "default": 0,
            "help": "If >0, also compute IG in rollout-time windows of this many steps.",
        },
        {
            "name": "--window_ig_samples",
            "type": int,
            "default": 256,
            "help": "Maximum histories used per time window.",
        },
        {
            "name": "--targets",
            "type": str,
            "default": "vel,mass,com,all",
            "help": "Comma separated encoder output heads: vel,mass,com,all.",
        },
        {"name": "--command_x", "type": float, "default": 0.5},
        {"name": "--command_y", "type": float, "default": 0.0},
        {"name": "--command_yaw", "type": float, "default": 0.0},
        {
            "name": "--output_dir",
            "type": str,
            "default": "",
            "help": "Optional output directory. Defaults under the loaded run.",
        },
        {
            "name": "--no_mat",
            "action": "store_true",
            "default": False,
            "help": "Disable MATLAB .mat table output.",
        },
    ]

    args = gymutil.parse_arguments(
        description="Encoder Integrated Gradients analysis",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    args.headless = not args.render
    return args


def make_command_tensor(env, args):
    command = torch.zeros(env.num_commands, device=env.device)
    values = torch.tensor(
        [args.command_x, args.command_y, args.command_yaw],
        dtype=command.dtype,
        device=env.device,
    )
    command[: min(command.numel(), values.numel())] = values[: min(command.numel(), values.numel())]
    return command


def set_eval_overrides(env_cfg, args):
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = max(float(env_cfg.env.episode_length_s), 60.0)

    # Keep the rollout focused on encoder/load behavior rather than push events.
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_curriculum = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.domain_rand.load_debug = True
    return env_cfg


def build_observation_groups(num_obs, num_actions, use_qs=True, use_torques=True):
    """Return per-step groups for raw obs plus filtered obs.

    Each group is a tuple: (label, branch, name, start, end).
    Starts/ends are indices within one per-step encoder input frame.

    Obs 顺序需与 wheelfoot_flat.compute_group_observations() 严格一致：
        ang_vel(3) | gravity(3) | dof_pos(6) | dof_vel(8) |
        [torques(8) if use_torques] | [qs_load(8) + qs_residual(4) if use_qs] | prev_actions(num_actions)
    """

    filtered_size = num_obs - num_actions
    filtered_offset = num_obs
    groups = []

    def add(branch, name, start, end):
        if end > start:
            groups.append((f"{branch}/{name}", branch, name, start, end))

    def add_pre_action_groups(branch, offset):
        cur = 0  # offset within filtered slice

        def chunk(name, span):
            nonlocal cur
            end = min(cur + span, filtered_size)
            add(branch, name, offset + cur, offset + end)
            cur = end

        chunk("base_ang_vel", 3)
        chunk("projected_gravity", 3)
        chunk("dof_pos", 6)
        chunk("dof_vel", 8)
        if use_torques:
            chunk("torques", 8)
        if use_qs:
            qs_load_start = cur
            chunk("qs_load_features", 8)
            _qs_load_names = ["payload_mass", "load_x", "load_y", "payload_present",
                              "sin_lieangle_L_thigh", "cos_lieangle_L_thigh",
                              "sin_lieangle_R_thigh", "cos_lieangle_R_thigh"]
            for _i, _n in enumerate(_qs_load_names):
                add(branch, f"qs_dim_{_n}",
                    offset + qs_load_start + _i, offset + qs_load_start + _i + 1)
            qs_resid_start = cur
            chunk("qs_residual_baseline", 4)
            _qs_resid_names = ["mass_delta", "com_delta_x", "com_delta_y", "com_delta_z"]
            for _i, _n in enumerate(_qs_resid_names):
                add(branch, f"qs_dim_{_n}",
                    offset + qs_resid_start + _i, offset + qs_resid_start + _i + 1)
        # 任何剩余的 pre-action 字段（理论上不应该有）
        if cur < filtered_size:
            add(branch, "extra_pre_action", offset + cur, offset + filtered_size)

    add_pre_action_groups("raw", 0)
    add("raw", "previous_actions", filtered_size, num_obs)
    add_pre_action_groups("filtered", filtered_offset)
    return groups


def target_indices_from_name(name):
    target_map = {
        "vel": [0, 1, 2],
        "mass": [3],
        "com": [4, 5, 6],
        "all": [0, 1, 2, 3, 4, 5, 6],
    }
    if name not in target_map:
        raise ValueError(f"Unknown target '{name}'. Expected one of {sorted(target_map)}")
    return target_map[name]


def collect_encoder_histories(env, ppo_runner, args):
    policy = ppo_runner.get_inference_policy(device=env.device)
    inference_encoder = ppo_runner.get_inference_encoder(device=env.device)
    command_value = make_command_tensor(env, args)

    env.reset()
    obs, obs_history, commands, _ = env.get_observations()
    samples = []
    sample_steps = []
    max_samples = int(args.ig_samples)
    num_steps = int(args.warmup_steps + args.rollout_steps)
    collect_all_windows = int(args.window_steps) > 0

    with torch.no_grad():
        for step in range(num_steps):
            env.commands[:, :] = command_value
            commands = env.commands[:, : env.num_commands] * env.commands_scale
            est = inference_encoder(obs_history, obs)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            obs, _, _, _, obs_history, commands, _ = env.step(actions.detach())

            after_warmup = step >= args.warmup_steps
            on_stride = ((step - args.warmup_steps) % args.sample_stride) == 0
            if after_warmup and on_stride:
                if collect_all_windows:
                    take_n = obs_history.shape[0]
                else:
                    remaining = max_samples - sum(batch.shape[0] for batch in samples)
                    if remaining <= 0:
                        break
                    take_n = min(remaining, obs_history.shape[0])
                samples.append(obs_history[:take_n].detach().clone())
                sample_steps.append(
                    torch.full((take_n,), step, dtype=torch.long, device=obs_history.device)
                )

    if not samples:
        raise RuntimeError("No encoder histories were collected.")
    return torch.cat(samples, dim=0), torch.cat(sample_steps, dim=0)


def select_evenly_spaced(samples, sample_steps, max_samples):
    if max_samples <= 0 or samples.shape[0] <= max_samples:
        return samples, sample_steps
    indices = torch.linspace(
        0,
        samples.shape[0] - 1,
        max_samples,
        dtype=torch.long,
        device=samples.device,
    )
    return samples[indices], sample_steps[indices]


def integrated_gradients_for_target(
    encoder,
    samples,
    baseline,
    output_indices,
    groups,
    hist_len,
    per_step_dim,
    ig_steps,
    batch_size,
):
    was_training = encoder.training
    # cuDNN-backed GRU backward needs training mode even when only input
    # gradients are requested. This does not update weights.
    encoder.train()

    device = samples.device
    group_time = torch.zeros(len(groups), hist_len, device=device)
    group_total = torch.zeros(len(groups), device=device)

    alphas = torch.linspace(
        0.5 / ig_steps,
        1.0 - 0.5 / ig_steps,
        ig_steps,
        dtype=samples.dtype,
        device=device,
    )

    try:
        for start in range(0, samples.shape[0], batch_size):
            end = min(start + batch_size, samples.shape[0])
            x = samples[start:end]
            x0 = baseline.expand_as(x)
            delta = x - x0
            batch_attr_abs = torch.zeros_like(x)

            for out_idx in output_indices:
                avg_grad = torch.zeros_like(x)
                for alpha in alphas:
                    x_alpha = (x0 + alpha * delta).detach().requires_grad_(True)
                    out = encoder(x_alpha)
                    target = out[:, out_idx].sum()
                    grad = torch.autograd.grad(target, x_alpha, retain_graph=False)[0]
                    avg_grad += grad
                attr = delta * avg_grad / float(ig_steps)
                batch_attr_abs += attr.abs()

            attr_by_step = batch_attr_abs.reshape(x.shape[0], hist_len, per_step_dim)
            for group_idx, (_, _, _, g_start, g_end) in enumerate(groups):
                values = attr_by_step[:, :, g_start:g_end]
                group_time[group_idx] += values.sum(dim=(0, 2))
                group_total[group_idx] += values.sum()
    finally:
        encoder.train(was_training)

    return group_total.detach().cpu(), group_time.detach().cpu()


def get_summary_header():
    return [
        "target",
        "label",
        "branch",
        "group",
        "feature_dim",
        "attribution",
        "percent",
        "mean_per_feature_attribution",
    ]


def build_summary_rows(target_summaries, groups, hist_len):
    rows = []
    for target, (group_total, _) in target_summaries.items():
        total = group_total.sum().item()
        for idx, (label, branch, group, start, end) in enumerate(groups):
            value = group_total[idx].item()
            feature_dim = end - start
            denom = max(total, 1e-12)
            rows.append(
                [
                    target,
                    label,
                    branch,
                    group,
                    feature_dim,
                    value,
                    100.0 * value / denom,
                    value / max(feature_dim * hist_len, 1),
                ]
            )
    return rows


def write_summary_csv(path, target_summaries, groups, hist_len):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(get_summary_header())
        writer.writerows(build_summary_rows(target_summaries, groups, hist_len))


def get_time_header():
    return [
        "target",
        "history_index",
        "history_from_now",
        "label",
        "branch",
        "group",
        "feature_dim",
        "attribution",
        "percent_of_target",
    ]


def build_time_rows(target_summaries, groups, hist_len):
    rows = []
    for target, (group_total, group_time) in target_summaries.items():
        total = group_total.sum().item()
        denom = max(total, 1e-12)
        for group_idx, (label, branch, group, start, end) in enumerate(groups):
            for hist_idx in range(hist_len):
                value = group_time[group_idx, hist_idx].item()
                rows.append(
                    [
                        target,
                        hist_idx,
                        hist_idx - (hist_len - 1),
                        label,
                        branch,
                        group,
                        end - start,
                        value,
                        100.0 * value / denom,
                    ]
                )
    return rows


def write_time_csv(path, target_summaries, groups, hist_len):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(get_time_header())
        writer.writerows(build_time_rows(target_summaries, groups, hist_len))


def get_window_header():
    return [
        "target",
        "window_index",
        "analysis_step_start",
        "analysis_step_end",
        "rollout_step_start",
        "rollout_step_end",
        "time_start_s",
        "time_end_s",
        "num_candidate_samples",
        "num_ig_samples",
        "target_total_attribution",
        "label",
        "branch",
        "group",
        "feature_dim",
        "attribution",
        "percent",
    ]


def write_window_csv(path, window_rows):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(get_window_header())
        writer.writerows(window_rows)


def rows_to_matlab_cell(header, rows):
    table_rows = [header] + rows
    if not table_rows:
        return np.empty((0, 0), dtype=object)

    cell = np.empty((len(table_rows), len(header)), dtype=object)
    for row_idx, row in enumerate(table_rows):
        for col_idx, value in enumerate(row):
            cell[row_idx, col_idx] = value
    return cell


def write_mat_tables(
    path,
    target_summaries,
    groups,
    hist_len,
    window_rows,
    baseline,
    checkpoint_path,
    args,
    env,
):
    if savemat is None:
        print("[encoder IG] scipy is not installed; skipped MATLAB .mat output.")
        return False

    groups_header = ["label", "branch", "group", "start", "end"]
    groups_rows = [list(group) for group in groups]
    summary_header = get_summary_header()
    time_header = get_time_header()
    window_header = get_window_header()

    savemat(
        path,
        {
            "summary_cell": rows_to_matlab_cell(
                summary_header,
                build_summary_rows(target_summaries, groups, hist_len),
            ),
            "time_cell": rows_to_matlab_cell(
                time_header,
                build_time_rows(target_summaries, groups, hist_len),
            ),
            "window_cell": rows_to_matlab_cell(window_header, window_rows),
            "groups_cell": rows_to_matlab_cell(groups_header, groups_rows),
            "baseline": baseline.detach().cpu().numpy(),
            "checkpoint_path": checkpoint_path,
            "task": args.task,
            "load_run": str(args.load_run),
            "checkpoint": int(args.checkpoint) if args.checkpoint is not None else -1,
            "num_envs": int(args.num_envs),
            "rollout_steps": int(args.rollout_steps),
            "warmup_steps": int(args.warmup_steps),
            "sample_stride": int(args.sample_stride),
            "ig_samples": int(args.ig_samples),
            "ig_steps": int(args.ig_steps),
            "window_steps": int(args.window_steps),
            "window_ig_samples": int(args.window_ig_samples),
            "dt": float(env.dt),
        },
        do_compression=True,
    )
    return True


def compute_window_summaries(
    encoder,
    samples,
    sample_steps,
    baseline,
    target_names,
    groups,
    hist_len,
    per_step_dim,
    args,
    env_dt,
):
    window_rows = []
    window_steps = int(args.window_steps)
    if window_steps <= 0:
        return window_rows

    relative_steps = sample_steps - int(args.warmup_steps)
    num_windows = (int(args.rollout_steps) + window_steps - 1) // window_steps

    for window_idx in range(num_windows):
        analysis_start = window_idx * window_steps
        analysis_end = min((window_idx + 1) * window_steps, int(args.rollout_steps))
        mask = (relative_steps >= analysis_start) & (relative_steps < analysis_end)
        candidate_count = int(mask.sum().item())
        if candidate_count == 0:
            continue

        window_samples = samples[mask]
        window_steps_tensor = sample_steps[mask]
        window_samples, window_steps_tensor = select_evenly_spaced(
            window_samples,
            window_steps_tensor,
            int(args.window_ig_samples),
        )
        ig_count = int(window_samples.shape[0])

        rollout_start = int(args.warmup_steps) + analysis_start
        rollout_end = int(args.warmup_steps) + analysis_end
        time_start_s = rollout_start * env_dt
        time_end_s = rollout_end * env_dt

        print(
            "[encoder IG] window "
            f"{window_idx}: analysis_steps=[{analysis_start}, {analysis_end}), "
            f"candidates={candidate_count}, ig_samples={ig_count}"
        )

        for target_name in target_names:
            output_indices = target_indices_from_name(target_name)
            group_total, _ = integrated_gradients_for_target(
                encoder=encoder,
                samples=window_samples,
                baseline=baseline,
                output_indices=output_indices,
                groups=groups,
                hist_len=hist_len,
                per_step_dim=per_step_dim,
                ig_steps=args.ig_steps,
                batch_size=args.ig_batch_size,
            )
            total = group_total.sum().item()
            denom = max(total, 1e-12)
            for group_idx, (label, branch, group, start, end) in enumerate(groups):
                value = group_total[group_idx].item()
                window_rows.append(
                    [
                        target_name,
                        window_idx,
                        analysis_start,
                        analysis_end,
                        rollout_start,
                        rollout_end,
                        time_start_s,
                        time_end_s,
                        candidate_count,
                        ig_count,
                        total,
                        label,
                        branch,
                        group,
                        end - start,
                        value,
                        100.0 * value / denom,
                    ]
                )
    return window_rows


def print_top_groups(target_summaries, groups, top_k=12):
    for target, (group_total, _) in target_summaries.items():
        total = group_total.sum().item()
        denom = max(total, 1e-12)
        order = torch.argsort(group_total, descending=True)
        print(f"\n[encoder IG] target={target} top {min(top_k, len(groups))}")
        for rank, group_idx in enumerate(order[:top_k], start=1):
            idx = int(group_idx.item())
            label = groups[idx][0]
            value = group_total[idx].item()
            print(f"  {rank:2d}. {label:<34s} {100.0 * value / denom:7.3f}%")


def normalize_load_run(load_run):
    if str(load_run) == "-1":
        return -1
    return load_run


def load_checkpoint_into_runner(ppo_runner, checkpoint_path, device):
    load_kwargs = {"map_location": torch.device(device)}
    try:
        loaded = torch.load(checkpoint_path, weights_only=True, **load_kwargs)
    except TypeError:
        loaded = torch.load(checkpoint_path, **load_kwargs)
    ppo_runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
    ppo_runner.alg.encoder.load_state_dict(loaded["encoder_state_dict"])
    ppo_runner.current_learning_iteration = loaded.get("iter", 0)


def main():
    args = get_analysis_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg = set_eval_overrides(env_cfg, args)

    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    train_cfg.runner.load_run = normalize_load_run(train_cfg.runner.load_run)
    train_cfg.runner.resume = False
    args.resume = False

    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        args.task,
        train_cfg.runner.experiment_name,
    )
    checkpoint_path = get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )
    run_dir = os.path.dirname(checkpoint_path)

    # Auto-align env_cfg from saved train-time cfg (avoids encoder shape mismatch when
    # in-tree cfg differs from training config, e.g., flipped use_qs_in_obs / use_torques_in_obs).
    _env_p = os.path.join(run_dir, "env_cfg.json")
    if os.path.isfile(_env_p):
        import json as _json
        try:
            _se = _json.load(open(_env_p))
            for k in ("num_observations", "num_critic_observations"):
                sv = _se.get("env", {}).get(k)
                if sv is not None and getattr(env_cfg.env, k, None) != sv:
                    setattr(env_cfg.env, k, int(sv))
            # 字段缺失（older runs）：use_torques_in_obs 的默认是 True（flag 2026-05-24 才加）。
            _flag_defaults_when_missing = {"use_torques_in_obs": True}
            for k in ("use_qs_in_obs", "use_residual_learning", "use_torques_in_obs"):
                sv = _se.get("env", {}).get(k)
                if sv is not None:
                    if getattr(env_cfg.env, k, None) != bool(sv):
                        setattr(env_cfg.env, k, bool(sv))
                elif k in _flag_defaults_when_missing:
                    setattr(env_cfg.env, k, _flag_defaults_when_missing[k])
            print(f"[encoder IG] auto-aligned env_cfg from saved: "
                  f"n_obs={env_cfg.env.num_observations}, "
                  f"use_qs_in_obs={env_cfg.env.use_qs_in_obs}, "
                  f"use_torques_in_obs={getattr(env_cfg.env, 'use_torques_in_obs', True)}")
        except Exception as e:
            print(f"[encoder IG] warn: could not align cfg from saved: {e}")

    # Also sync encoder input dim from saved train_cfg.json (so MLP_Encoder rebuild
    # matches the saved ckpt's per-step obs width — otherwise shape mismatch on load).
    _tr_p2 = os.path.join(run_dir, "train_cfg.json")
    if os.path.isfile(_tr_p2):
        import json as _json2
        try:
            _str2 = _json2.load(open(_tr_p2))
            sm = _str2.get("MLP_Encoder", {})
            if sm.get("num_input_dim") is not None:
                train_cfg.MLP_Encoder.num_input_dim = int(sm["num_input_dim"])
            if sm.get("obs_history_length") is not None:
                train_cfg.MLP_Encoder.obs_history_length = int(sm["obs_history_length"])
            # Also sync algorithm.use_load_residual_estimation (E3 has this False;
            # leaving in-tree True triggers a 4D-baseline shape check that fails on
            # 36-dim obs). Mirror play.py's behavior.
            sa = _str2.get("algorithm", {})
            if sa.get("use_load_residual_estimation") is not None:
                train_cfg.algorithm.use_load_residual_estimation = bool(sa["use_load_residual_estimation"])
            print(f"[encoder IG] auto-aligned train_cfg: "
                  f"num_input_dim={train_cfg.MLP_Encoder.num_input_dim}, "
                  f"use_load_residual_estimation={getattr(train_cfg.algorithm, 'use_load_residual_estimation', '?')}")
        except Exception as e:
            print(f"[encoder IG] warn: could not align train_cfg from saved: {e}")

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=log_root,
    )
    load_checkpoint_into_runner(ppo_runner, checkpoint_path, args.rl_device)

    collected_samples, collected_steps = collect_encoder_histories(env, ppo_runner, args)
    baseline = collected_samples.mean(dim=0, keepdim=True)
    samples, sample_steps = select_evenly_spaced(
        collected_samples,
        collected_steps,
        int(args.ig_samples),
    )

    hist_len = int(train_cfg.MLP_Encoder.obs_history_length)
    per_step_dim = samples.shape[1] // hist_len
    use_qs_flag = bool(getattr(env.cfg.env, "use_qs_in_obs", True))
    use_torq_flag = bool(getattr(env.cfg.env, "use_torques_in_obs", True))
    groups = build_observation_groups(env.num_obs, env.num_actions,
                                      use_qs=use_qs_flag, use_torques=use_torq_flag)

    print(
        "[encoder IG] collected "
        f"{collected_samples.shape[0]} candidate samples, "
        f"using {samples.shape[0]} total-IG samples, "
        f"history_len={hist_len}, per_step_dim={per_step_dim}, "
        "baseline=dataset_mean"
    )
    print(f"[encoder IG] headless={args.headless} (pass --render to show viewer)")
    print(f"[encoder IG] checkpoint={checkpoint_path}")

    encoder = ppo_runner.alg.encoder
    encoder.eval()
    target_names = [name.strip() for name in args.targets.split(",") if name.strip()]
    target_summaries = {}
    for target_name in target_names:
        output_indices = target_indices_from_name(target_name)
        group_total, group_time = integrated_gradients_for_target(
            encoder=encoder,
            samples=samples,
            baseline=baseline,
            output_indices=output_indices,
            groups=groups,
            hist_len=hist_len,
            per_step_dim=per_step_dim,
            ig_steps=args.ig_steps,
            batch_size=args.ig_batch_size,
        )
        target_summaries[target_name] = (group_total, group_time)

    window_rows = compute_window_summaries(
        encoder=encoder,
        samples=collected_samples,
        sample_steps=collected_steps,
        baseline=baseline,
        target_names=target_names,
        groups=groups,
        hist_len=hist_len,
        per_step_dim=per_step_dim,
        args=args,
        env_dt=env.dt,
    )

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(run_dir, "encoder_ig")
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = os.path.join(output_dir, f"encoder_ig_summary_{stamp}.csv")
    time_csv = os.path.join(output_dir, f"encoder_ig_time_{stamp}.csv")
    window_csv = os.path.join(output_dir, f"encoder_ig_windows_{stamp}.csv")
    tables_mat = os.path.join(output_dir, f"encoder_ig_tables_{stamp}.mat")
    baseline_path = os.path.join(output_dir, f"encoder_ig_baseline_mean_{stamp}.pt")

    write_summary_csv(summary_csv, target_summaries, groups, hist_len)
    write_time_csv(time_csv, target_summaries, groups, hist_len)
    if window_rows:
        write_window_csv(window_csv, window_rows)
    wrote_mat = False
    if not args.no_mat:
        wrote_mat = write_mat_tables(
            tables_mat,
            target_summaries,
            groups,
            hist_len,
            window_rows,
            baseline,
            checkpoint_path,
            args,
            env,
        )
    torch.save(
        {
            "baseline": baseline.detach().cpu(),
            "num_collected_samples": collected_samples.shape[0],
            "num_total_ig_samples": samples.shape[0],
            "history_len": hist_len,
            "per_step_dim": per_step_dim,
            "num_obs": env.num_obs,
            "num_actions": env.num_actions,
            "checkpoint_path": checkpoint_path,
            "groups": groups,
        },
        baseline_path,
    )

    print_top_groups(target_summaries, groups)
    print(f"\n[encoder IG] summary_csv={summary_csv}")
    print(f"[encoder IG] time_csv={time_csv}")
    if window_rows:
        print(f"[encoder IG] window_csv={window_csv}")
    if wrote_mat:
        print(f"[encoder IG] tables_mat={tables_mat}")
    print(f"[encoder IG] baseline={baseline_path}")


if __name__ == "__main__":
    main()
