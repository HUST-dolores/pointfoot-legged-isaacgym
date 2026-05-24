"""Actor Integrated Gradients analysis.

Independent from analyze_encoder_ig.py. Targets the ACTOR network (not
encoder). Actor input is cat([encoder_out, obs, commands]); we attribute
each input dim's contribution to the action output magnitude.

Output groups (combine raw+filtered NOT applicable here — actor input is
flat, no history).

Encoder output (7 dims):
  - est_vel: 3 dims (lin_vel_x/y/z)
  - est_mass: 1 dim
  - est_com:  3 dims (com_x/y/z)

Observation (48 dims when use_qs_in_obs=True, 36 when False):
  - base_ang_vel:        3
  - projected_gravity:   3
  - dof_pos:             6
  - dof_vel:             8
  - torques:             8
  - [qs_load_features:    8]
  - [qs_residual_baseline:4]
  - previous_actions:    8

Commands: 3 (cmd_vx, cmd_vy, cmd_yaw)

Usage:
  python legged_gym/scripts/analyze_actor_ig.py --headless \
      --load_run <run> --checkpoint <iter>

Output:
  - <run>/actor_ig/actor_ig_summary_<ts>.csv (one row per group per target)
  - <run>/actor_ig/actor_ig_summary_<ts>.txt (console-friendly)
"""
import os
import sys

# Isaac gym must import before torch
import isaacgym  # noqa: F401
from isaacgym.torch_utils import *  # noqa: F401, F403

import csv
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401, F403  registers tasks
from legged_gym.utils import task_registry


# ---- CLI ----

def get_args():
    from isaacgym import gymutil
    custom_parameters = [
        {"name": "--task", "type": str, "default": "wheelfoot_flat"},
        {"name": "--resume", "action": "store_true", "default": False},
        {"name": "--experiment_name", "type": str},
        {"name": "--run_name", "type": str},
        {"name": "--load_run", "type": str},
        {"name": "--checkpoint", "type": int},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--num_envs", "type": int, "default": 50},
        {"name": "--seed", "type": int},
        {"name": "--max_iterations", "type": int},
        {"name": "--exptid", "type": str, "default": ""},
        {"name": "--horovod", "action": "store_true", "default": False},
        {"name": "--rollout_steps", "type": int, "default": 400,
         "help": "Steps used to collect candidate actor inputs."},
        {"name": "--warmup_steps", "type": int, "default": 20,
         "help": "Initial rollout steps skipped before sampling."},
        {"name": "--sample_stride", "type": int, "default": 2,
         "help": "Collect samples every N steps."},
        {"name": "--ig_samples", "type": int, "default": 1024,
         "help": "Max samples for IG."},
        {"name": "--ig_steps", "type": int, "default": 32,
         "help": "Interpolation points baseline→input."},
        {"name": "--ig_batch_size", "type": int, "default": 64},
        {"name": "--targets", "type": str, "default": "all,per_action",
         "help": "Comma-sep: all (L2 of action), per_action (per dim breakdown)."},
        {"name": "--command_x", "type": float, "default": 0.5},
        {"name": "--command_y", "type": float, "default": 0.0},
        {"name": "--command_yaw", "type": float, "default": 0.0},
        {"name": "--load_mass_min", "type": float, "default": -1.0,
         "help": "Override add_load_range[0] for OOD sampling. Negative = no override."},
        {"name": "--load_mass_max", "type": float, "default": -1.0,
         "help": "Override add_load_range[1] for OOD sampling. Negative = no override."},
        {"name": "--output_dir", "type": str, "default": "",
         "help": "Output dir; defaults to <run_dir>/actor_ig."},
    ]
    args = gymutil.parse_arguments(description="Actor IG", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


# ---- Group structure ----

def build_actor_input_groups(num_obs, num_actions, num_commands, encoder_dim=7):
    """Per-dim groups for actor input = cat([encoder_out, obs, commands]).

    Returns: list of (label, group_name, start, end) where indices are within
    one actor input vector (NOT including any history dim).
    """
    groups = []
    offset = 0

    def add(name, span):
        nonlocal offset
        end = offset + span
        groups.append((f"encoder/{name}" if offset < encoder_dim else
                       f"obs/{name}" if offset < encoder_dim + num_obs else
                       f"commands/{name}",
                       name, offset, end))
        offset = end

    # Encoder output decomposition (7 dims = 3 vel + 1 mass + 3 com)
    add("est_lin_vel", 3)          # 0-2
    add("est_mass", 1)             # 3
    add("est_com_delta", 3)        # 4-6

    # Observations: 48 (with qs) or 36 (without qs)
    filtered_size = num_obs - num_actions  # 40 or 28
    # Pre-action obs slice (always 28 dims for the non-QS part, plus optionally 12 QS dims)
    add("base_ang_vel", 3)         # obs[0:3]
    add("projected_gravity", 3)    # obs[3:6]
    add("dof_pos", 6)              # obs[6:12]
    add("dof_vel", 8)              # obs[12:20]
    add("torques", 8)              # obs[20:28]
    if num_obs >= 48:
        add("qs_load_features", 8)         # obs[28:36] — split per-dim below
        add("qs_residual_baseline", 4)     # obs[36:40]
    add("previous_actions", num_actions)   # obs[28:36] or obs[40:48]
    # Commands
    add("cmd", num_commands)
    return groups


def build_per_dim_groups(num_obs, num_actions, num_commands, encoder_dim=7):
    """Same as build_actor_input_groups but qs_load_features + qs_residual_baseline
    + encoder_out split per single dim. Used for fine-grained breakdown.
    """
    groups = []
    offset = 0

    def add(prefix, name, span):
        nonlocal offset
        groups.append((f"{prefix}/{name}", name, offset, offset + span))
        offset += span

    # Encoder per-dim
    enc_names = ["est_vel_x", "est_vel_y", "est_vel_z",
                 "est_mass",
                 "est_com_x", "est_com_y", "est_com_z"]
    for n in enc_names:
        add("encoder", n, 1)

    # Obs groups (same chunking as encoder analysis)
    add("obs", "base_ang_vel", 3)
    add("obs", "projected_gravity", 3)
    add("obs", "dof_pos", 6)
    add("obs", "dof_vel", 8)
    add("obs", "torques", 8)
    if num_obs >= 48:
        # qs_load_features per-dim
        qs_names = ["payload_mass", "load_x", "load_y", "payload_present",
                    "sin_lieangle_L_thigh", "cos_lieangle_L_thigh",
                    "sin_lieangle_R_thigh", "cos_lieangle_R_thigh"]
        for n in qs_names:
            add("obs", f"qs_{n}", 1)
        for n in ["mass_delta", "com_delta_x", "com_delta_y", "com_delta_z"]:
            add("obs", f"qs_{n}", 1)
    add("obs", "previous_actions", num_actions)
    # commands per-dim
    cmd_names = ["cmd_vx", "cmd_vy", "cmd_yaw"][:num_commands]
    for n in cmd_names:
        add("commands", n, 1)
    return groups


# ---- Rollout collection ----

def collect_actor_inputs(env, ppo_runner, args):
    """Run a rollout, save cat([encoder_out, obs, commands]) tensors per step."""
    policy = ppo_runner.get_inference_policy(device=env.device)
    inference_encoder = ppo_runner.get_inference_encoder(device=env.device)
    cmd_val = torch.tensor(
        [float(args.command_x), float(args.command_y), float(args.command_yaw)],
        device=env.device, dtype=torch.float32,
    )

    env.reset()
    obs, obs_history, commands, _ = env.get_observations()

    samples = []
    num_steps = int(args.warmup_steps) + int(args.rollout_steps)
    max_samples = int(args.ig_samples)

    with torch.no_grad():
        for step in range(num_steps):
            env.commands[:, : env.num_commands] = cmd_val[: env.num_commands]
            commands_scaled = env.commands[:, : env.num_commands] * env.commands_scale
            est = inference_encoder(obs_history, obs)
            actor_in = torch.cat((est, obs, commands_scaled), dim=-1)
            actions = policy(actor_in.detach())
            obs, _, _, _, obs_history, commands, _ = env.step(actions.detach())

            if step >= args.warmup_steps and ((step - args.warmup_steps) % args.sample_stride == 0):
                remaining = max_samples - sum(s.shape[0] for s in samples)
                if remaining <= 0:
                    break
                take = min(remaining, actor_in.shape[0])
                samples.append(actor_in[:take].detach().clone())

    if not samples:
        raise RuntimeError("No actor inputs collected.")
    return torch.cat(samples, dim=0)


# ---- IG core ----

def integrated_gradients_actor(
    actor_callable, samples, baseline, output_indices, ig_steps, batch_size,
):
    """Run IG: for each output_idx, compute |grad×delta| then accumulate.

    Args:
        actor_callable: function input → output (mean actions, deterministic).
        samples: [N, D] input tensors.
        baseline: [1, D] baseline tensor.
        output_indices: list of output dims (e.g., [0..7] for action dims) or
                        special string 'norm' for L2 norm of all outputs.
        ig_steps: number of interpolation points.
        batch_size: process N samples per batch.

    Returns:
        attribution_abs: [D] cumulative |attribution|.
    """
    device = samples.device
    D = samples.shape[1]
    total_attr_abs = torch.zeros(D, device=device)

    alphas = torch.linspace(
        0.5 / ig_steps, 1.0 - 0.5 / ig_steps, ig_steps,
        dtype=samples.dtype, device=device,
    )

    for start in range(0, samples.shape[0], batch_size):
        end = min(start + batch_size, samples.shape[0])
        x = samples[start:end]
        x0 = baseline.expand_as(x)
        delta = x - x0

        for out_spec in output_indices:
            avg_grad = torch.zeros_like(x)
            for alpha in alphas:
                x_alpha = (x0 + alpha * delta).detach().requires_grad_(True)
                out = actor_callable(x_alpha)
                if out_spec == 'norm':
                    target = (out ** 2).sum()
                else:
                    target = out[:, out_spec].sum()
                grad = torch.autograd.grad(target, x_alpha, retain_graph=False)[0]
                avg_grad = avg_grad + grad
            attr = delta * avg_grad / float(ig_steps)
            total_attr_abs += attr.abs().sum(dim=0)

    return total_attr_abs.detach().cpu()


# ---- Aggregation ----

def aggregate_to_groups(attribution, groups):
    """attribution: [D]. groups: list of (label, name, start, end).
    Returns: list of (label, name, dim, total_attr).
    """
    rows = []
    for label, name, start, end in groups:
        total = float(attribution[start:end].sum().item())
        rows.append((label, name, end - start, total))
    return rows


def print_table(title, rows, total_attr_sum):
    print(f"\n=== {title} ===")
    print(f"{'#':<3} {'label':<28} {'dim':>5} {'attr':>12} {'percent':>8} {'per-dim%':>10}")
    print("-" * 80)
    sorted_rows = sorted(rows, key=lambda r: -r[3])
    for i, (label, name, dim, attr) in enumerate(sorted_rows, 1):
        pct = 100.0 * attr / max(total_attr_sum, 1e-12)
        per_dim = pct / max(dim, 1)
        print(f"{i:<3} {label:<28} {dim:>5} {attr:>12.4f} {pct:>7.2f}% {per_dim:>9.3f}%")


# ---- Main ----

def main():
    args = get_args()

    # Build env (single test setup, fewer envs OK)
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = max(1, int(args.num_envs))
    env_cfg.env.episode_length_s = 60
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 5
    if env_cfg.terrain.mesh_type == "trimesh":
        env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False  # avoid sample contamination
    env_cfg.domain_rand.add_random_load = True
    # OOD override
    _lm_min = float(getattr(args, "load_mass_min", -1.0) or -1.0)
    _lm_max = float(getattr(args, "load_mass_max", -1.0) or -1.0)
    if _lm_min >= 0.0 and _lm_max >= 0.0:
        env_cfg.domain_rand.add_load_range = [_lm_min, _lm_max]
        print(f"[actor IG] override add_load_range = {env_cfg.domain_rand.add_load_range}")

    # Auto-align cfg with saved ckpt cfg to handle architectures with different
    # obs dims (e.g., history_only n_obs=36 vs main n_obs=48).
    import json as _json
    _log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, env_cfg.__class__.__module__)
    # Read directly from the ckpt's run dir
    _, _train_cfg_pre = task_registry.get_cfgs(name=args.task)
    _run_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", args.task,
        _train_cfg_pre.runner.experiment_name, args.load_run or "",
    )
    _env_p = os.path.join(_run_dir, "env_cfg.json")
    _tr_p  = os.path.join(_run_dir, "train_cfg.json")
    if os.path.isfile(_env_p):
        try:
            _se = _json.load(open(_env_p))
            _str = _json.load(open(_tr_p)) if os.path.isfile(_tr_p) else None
            for k in ("num_observations", "num_critic_observations"):
                sv = _se.get("env", {}).get(k)
                if sv is not None and getattr(env_cfg.env, k, None) != sv:
                    setattr(env_cfg.env, k, int(sv))
            for k in ("use_qs_in_obs", "use_residual_learning"):
                sv = _se.get("env", {}).get(k)
                if sv is not None and getattr(env_cfg.env, k, None) != bool(sv):
                    setattr(env_cfg.env, k, bool(sv))
            print(f"[actor IG] auto-aligned env_cfg from saved: "
                  f"n_obs={env_cfg.env.num_observations}, "
                  f"use_qs_in_obs={env_cfg.env.use_qs_in_obs}")
        except Exception as e:
            print(f"[actor IG] warn: could not align cfg from saved: {e}")

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    # Also align train_cfg side (encoder dim, use_load_residual_estimation)
    if os.path.isfile(_tr_p):
        try:
            _str = _json.load(open(_tr_p))
            sa = _str.get("algorithm", {})
            sm = _str.get("MLP_Encoder", {})
            if sa.get("use_load_residual_estimation") is not None:
                train_cfg.algorithm.use_load_residual_estimation = bool(sa["use_load_residual_estimation"])
            if sm.get("num_input_dim") is not None:
                train_cfg.MLP_Encoder.num_input_dim = int(sm["num_input_dim"])
            if sm.get("obs_history_length") is not None:
                train_cfg.MLP_Encoder.obs_history_length = int(sm["obs_history_length"])
        except Exception as e:
            print(f"[actor IG] warn: could not align train_cfg: {e}")
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg,
    )
    actor_critic = ppo_runner.alg.actor_critic
    actor_critic.eval()
    actor = actor_critic.act_inference

    num_actions = int(env.num_actions)
    num_obs = int(env.num_obs)
    num_commands = int(env.num_commands)
    encoder_dim = 7

    actor_input_dim = encoder_dim + num_obs + num_commands
    print(f"\n[actor IG] task={args.task}  load_run={args.load_run}  ckpt={args.checkpoint}")
    print(f"[actor IG] actor input dims: {encoder_dim} (encoder) + {num_obs} (obs) + {num_commands} (commands) = {actor_input_dim}")
    print(f"[actor IG] num_actions = {num_actions}")
    print(f"[actor IG] uniform per-dim baseline = {100.0 / actor_input_dim:.3f}%")

    # Collect samples
    samples = collect_actor_inputs(env, ppo_runner, args)
    print(f"[actor IG] collected {samples.shape[0]} samples")

    # Baseline = mean of samples
    baseline = samples.mean(dim=0, keepdim=True)

    # Targets
    target_specs = []
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if "all" in targets:
        target_specs.append(("all", ['norm']))   # L2 norm of action vector
    if "per_action" in targets:
        for i in range(num_actions):
            target_specs.append((f"act_{i}", [i]))

    # Build group structures
    groups_coarse = build_actor_input_groups(num_obs, num_actions, num_commands, encoder_dim)
    groups_fine = build_per_dim_groups(num_obs, num_actions, num_commands, encoder_dim)

    # Run IG for each target
    output_dir = args.output_dir
    if not output_dir:
        run_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", args.task,
            train_cfg.runner.experiment_name, args.load_run,
        )
        output_dir = os.path.join(run_dir, "actor_ig")
    os.makedirs(output_dir, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(output_dir, f"actor_ig_summary_{stamp}.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["target", "groupset", "label", "group", "dim", "attribution", "percent", "per_dim_percent"])

        for tgt_name, tgt_indices in target_specs:
            print(f"\n========== TARGET: {tgt_name} ==========")
            attr = integrated_gradients_actor(
                actor, samples, baseline, tgt_indices,
                ig_steps=int(args.ig_steps),
                batch_size=int(args.ig_batch_size),
            )
            total = float(attr.sum().item())

            # Coarse groups
            rows_coarse = aggregate_to_groups(attr, groups_coarse)
            print_table(f"{tgt_name} — group view", rows_coarse, total)
            for label, name, dim, a in rows_coarse:
                pct = 100.0 * a / max(total, 1e-12)
                writer.writerow([tgt_name, "coarse", label, name, dim, a, pct, pct / max(dim, 1)])

            # Fine per-dim
            rows_fine = aggregate_to_groups(attr, groups_fine)
            print_table(f"{tgt_name} — per-dim view", rows_fine, total)
            for label, name, dim, a in rows_fine:
                pct = 100.0 * a / max(total, 1e-12)
                writer.writerow([tgt_name, "fine", label, name, dim, a, pct, pct / max(dim, 1)])

    print(f"\n[actor IG] CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
