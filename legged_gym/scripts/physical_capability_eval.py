#!/usr/bin/env python3
"""Batch physical-capability evaluation for the wheelfoot co-design study.

Supported tasks:
  clearance  reliable swing-wheel ground clearance on a plane
  slope      reliable static hold on real tiled slope terrain
  payload    reliable speed with rigidly attached body mass
  speed      maximum stable actual speed on a plane or fixed real slope

Configuration is passed through PHYS_EVAL_* environment variables so the
Isaac Gym argument parser remains unchanged. See PHYSICAL_METRICS_PROTOCOL.md
in the work-log archive for the frozen metric definitions.
"""

import json
import os
import sys
import time

import isaacgym  # noqa: F401
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.saved_config import load_saved_run_configs
import numpy as np
import torch


DEFAULT_LEVELS = {
    "slope": [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0],
    "payload": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
    "speed": [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6],
}
MOTOR_GROUPS = ("abad", "hip", "knee", "wheel")


def _parse_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value")


def _parse_levels(task):
    raw = os.environ.get("PHYS_EVAL_LEVELS", "").strip()
    if raw:
        values = [float(value) for value in raw.split(",") if value.strip()]
    else:
        values = DEFAULT_LEVELS.get(task, [0.0])
    if not values:
        raise ValueError("PHYS_EVAL_LEVELS must not be empty")
    if any(b <= a for a, b in zip(values[:-1], values[1:])):
        raise ValueError("PHYS_EVAL_LEVELS must be strictly increasing")
    return values


def _disable_domain_randomization(env_cfg):
    names = (
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_mass",
        "add_random_load",
        "per_env_load_mass",
        "randomize_base_com",
        "randomize_inertia",
        "push_robots",
        "rand_force",
        "randomize_Kp",
        "randomize_Kd",
        "randomize_motor_torque",
        "randomize_default_dof_pos",
        "randomize_action_delay",
        "randomize_imu_offset",
        "attached_payload",
    )
    for name in names:
        setattr(env_cfg.domain_rand, name, False)
    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False


def _configure_fixed_design(env_cfg, design, seed, prefix):
    thigh, shank, abad, hip, knee, wheel = design
    env_cfg.domain_rand.design_anchor_fraction = 0.0
    env_cfg.domain_rand.randomize_morphology = True
    env_cfg.domain_rand.morphology_regenerate = True
    env_cfg.domain_rand.morphology_num_buckets = 1
    env_cfg.domain_rand.morphology_thigh_scale_range = [thigh, thigh]
    env_cfg.domain_rand.morphology_shank_scale_range = [shank, shank]
    env_cfg.domain_rand.morphology_prefix = prefix
    env_cfg.domain_rand.morphology_seed = seed
    env_cfg.domain_rand.randomize_motor_design = True
    motor_values = (abad, hip, knee, wheel)
    env_cfg.domain_rand.motor_group_torque_scale_ranges = [
        [value, value] for value in motor_values
    ]
    env_cfg.domain_rand.motor_torque_scale_range = [
        min(motor_values),
        max(motor_values),
    ]
    env_cfg.domain_rand.motor_mass_exponent = float(
        os.environ.get("PHYS_MOTOR_MASS_EXPONENT", "1.0")
    )
    env_cfg.domain_rand.motor_power_envelope = _parse_bool(
        "PHYS_MOTOR_POWER_ENVELOPE",
        getattr(env_cfg.domain_rand, "motor_power_envelope", False),
    )
    env_cfg.domain_rand.motor_specific_power_w_per_kg = float(
        os.environ.get(
            "PHYS_MOTOR_SPECIFIC_POWER_W_PER_KG",
            str(
                getattr(
                    env_cfg.domain_rand,
                    "motor_specific_power_w_per_kg",
                    450.0,
                )
            ),
        )
    )
    env_cfg.domain_rand.motor_power_exponent = float(
        os.environ.get(
            "PHYS_MOTOR_POWER_EXPONENT",
            str(
                getattr(
                    env_cfg.domain_rand,
                    "motor_power_exponent",
                    1.0,
                )
            ),
        )
    )
    env_cfg.domain_rand.motor_power_min_speed_rad_s = float(
        os.environ.get(
            "PHYS_MOTOR_POWER_MIN_SPEED_RAD_S",
            str(
                getattr(
                    env_cfg.domain_rand,
                    "motor_power_min_speed_rad_s",
                    1.0e-3,
                )
            ),
        )
    )
    nominal_mass_raw = os.environ.get(
        "PHYS_MOTOR_NOMINAL_MASSES_KG", ""
    ).strip()
    if nominal_mass_raw:
        nominal_masses = [
            float(value)
            for value in nominal_mass_raw.split(",")
            if value.strip()
        ]
        if len(nominal_masses) != 4 or any(value <= 0.0 for value in nominal_masses):
            raise ValueError(
                "PHYS_MOTOR_NOMINAL_MASSES_KG must contain four positive values"
            )
        env_cfg.domain_rand.motor_nominal_masses_kg = nominal_masses
    env_cfg.domain_rand.motor_design_seed = seed


def _quaternion_roll(quat):
    x, y, z, w = quat.unbind(dim=1)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    return torch.atan2(sin_roll, cos_roll)


def _quaternion_pitch(quat):
    x, y, z, w = quat.unbind(dim=1)
    sin_pitch = 2.0 * (w * y - z * x)
    return torch.asin(torch.clamp(sin_pitch, -1.0, 1.0))


def _set_commands(env, roll_command, step_command):
    env.commands[:, :3] = 0.0
    env.commands[:, 0] = roll_command
    if hasattr(env, "cmd_step"):
        env.cmd_step[:, 0] = step_command


def _level_ids(num_envs, num_levels, device):
    return torch.div(
        torch.arange(num_envs, device=device) * num_levels,
        num_envs,
        rounding_mode="floor",
    ).clamp(max=num_levels - 1)


def _training_speed_limit(log_root, run_name, fallback):
    cfg_path = os.path.join(log_root, run_name, "env_cfg.json")
    try:
        saved = json.load(open(cfg_path, encoding="utf-8"))
        commands = saved.get("commands", {})
        scenario = saved.get("scenario", {})
        physical_levels = scenario.get("physical_speed_levels", [])
        limits = [
            float(commands.get("smooth_max_lin_vel_x", 0.0)),
            float(commands.get("ranges", {}).get("lin_vel_x", [0.0, 0.0])[1]),
            float(scenario.get("accel_vx_range", [0.0, 0.0])[1]),
            max([float(value) for value in physical_levels], default=0.0),
        ]
        return max(limits)
    except Exception:
        return float(fallback)


def _continuous_reliable_limit(levels, success_rates, threshold=0.90):
    reliable = []
    still_continuous = True
    for level, rate in zip(levels, success_rates):
        still_continuous = still_continuous and rate >= threshold
        reliable.append(still_continuous)
    valid = [level for level, passed in zip(levels, reliable) if passed]
    return (max(valid) if valid else float("nan")), reliable


def _interpolated_reliable_limit(levels, success_rates, threshold=0.90):
    """Interpolate the first threshold crossing after enforcing continuity."""
    levels = np.asarray(levels, dtype=np.float64)
    rates = np.minimum.accumulate(
        np.asarray(success_rates, dtype=np.float64)
    )
    if rates[0] < threshold:
        return float("nan")
    below = np.flatnonzero(rates < threshold)
    if not len(below):
        return float(levels[-1])
    right = int(below[0])
    left = right - 1
    rate_span = rates[left] - rates[right]
    if rate_span <= 1.0e-12:
        return float(levels[left])
    fraction = (rates[left] - threshold) / rate_span
    return float(
        levels[left] + fraction * (levels[right] - levels[left])
    )


def _safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _configure_environment(
    env_cfg,
    task,
    levels,
    num_envs,
    total_s,
    speed_slope_deg,
):
    _disable_domain_randomization(env_cfg)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = max(float(total_s) + 5.0, 20.0)
    env_cfg.scenario.partition = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = max(float(total_s) + 100.0, 1000.0)
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.cmd_step_range = [0.0, 0.0]
    env_cfg.init_state.dof_pos_randomization_range = [-0.05, 0.05]
    env_cfg.terrain.static_friction = float(
        os.environ.get("PHYS_EVAL_FRICTION", "0.8")
    )
    env_cfg.terrain.dynamic_friction = env_cfg.terrain.static_friction
    env_cfg.terrain.restitution = 0.0

    real_slope = task == "slope" or (
        task == "speed" and speed_slope_deg > 0.0
    )
    if real_slope:
        env_cfg.terrain.mesh_type = "trimesh"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        env_cfg.terrain.horizontal_scale = float(
            os.environ.get(
                "PHYS_EVAL_SLOPE_HORIZONTAL_SCALE",
                env_cfg.terrain.horizontal_scale,
            )
        )
        env_cfg.terrain.vertical_scale = float(
            os.environ.get(
                "PHYS_EVAL_SLOPE_VERTICAL_SCALE",
                env_cfg.terrain.vertical_scale,
            )
        )
        env_cfg.terrain.evaluation_slope_degrees = (
            list(levels) if task == "slope" else [speed_slope_deg]
        )
        env_cfg.terrain.evaluation_spawn_xy_jitter = 0.0
        env_cfg.terrain.evaluation_reset_xy_jitter = 0.15
        # Every speed command uses the same physical incline. A single shared
        # terrain row avoids duplicating a 110 m mesh for every command level.
        env_cfg.terrain.num_rows = len(levels) if task == "slope" else 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.max_init_terrain_level = (
            len(levels) - 1 if task == "slope" else 0
        )
        default_length = "120.0" if task == "speed" else "12.0"
        env_cfg.terrain.terrain_length = float(
            os.environ.get("PHYS_EVAL_SLOPE_LENGTH", default_length)
        )
        env_cfg.terrain.terrain_width = float(
            os.environ.get("PHYS_EVAL_SLOPE_WIDTH", "6.0")
        )
        env_cfg.terrain.border_size = float(
            os.environ.get("PHYS_EVAL_TERRAIN_BORDER", "5.0")
        )
    else:
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.evaluation_slope_degrees = None

    if task == "payload":
        env_cfg.domain_rand.attached_payload = True
        env_cfg.domain_rand.attached_payload_mass_range = [
            min(levels),
            max(levels),
        ]
        env_cfg.domain_rand.attached_payload_eval_masses = list(levels)
        env_cfg.domain_rand.attached_payload_offset_range_x = [0.0, 0.0]
        env_cfg.domain_rand.attached_payload_offset_range_y = [0.0, 0.0]
        env_cfg.domain_rand.attached_payload_offset_range_z = [0.10, 0.10]


def _clearance_summary(cycles, failed, num_envs):
    valid_rows = [
        row for row in cycles if not bool(failed[int(row[0])].item())
    ]
    grouped = {}
    for env_id, foot_id, peak, air_steps in valid_rows:
        grouped.setdefault((int(env_id), int(foot_id)), []).append(float(peak))

    medians = []
    left = {}
    right = {}
    for (env_id, foot_id), peaks in grouped.items():
        median_peak = float(np.median(peaks))
        medians.append(median_peak)
        (left if foot_id == 0 else right)[env_id] = median_peak
    paired = sorted(set(left) & set(right))
    asymmetry = [abs(left[env_id] - right[env_id]) for env_id in paired]
    valid_env = [
        env_id
        for env_id in range(num_envs)
        if not bool(failed[env_id].item())
        and len(grouped.get((env_id, 0), [])) >= 2
        and len(grouped.get((env_id, 1), [])) >= 2
    ]
    medians_np = np.asarray(medians, dtype=np.float64)
    return {
        "metric_name": "H_clear_90",
        "metric_value": _safe_float(np.percentile(medians_np, 10))
        if len(medians_np)
        else None,
        "unit": "m",
        "median_m": _safe_float(np.median(medians_np))
        if len(medians_np)
        else None,
        "maximum_m": _safe_float(np.max(medians_np))
        if len(medians_np)
        else None,
        "cycle_count": len(valid_rows),
        "valid_env_count": len(valid_env),
        "valid_env_rate": len(valid_env) / max(num_envs, 1),
        "left_right_abs_difference_median_m": _safe_float(np.median(asymmetry))
        if asymmetry
        else None,
    }


def main():
    task = os.environ.get("PHYS_EVAL_TASK", "clearance").strip().lower()
    if task not in ("clearance", "slope", "payload", "speed"):
        raise ValueError(
            "PHYS_EVAL_TASK must be clearance, slope, payload, or speed"
        )
    run_name = os.environ.get("PHYS_EVAL_RUN", "Jul05_00-02-15_")
    checkpoint = int(os.environ.get("PHYS_EVAL_CKPT", "27000"))
    seed = int(os.environ.get("PHYS_EVAL_SEED", "20260724"))
    if seed == 1:
        raise ValueError("seed=1 is randomized by helpers.set_seed; use a fixed seed")
    levels = _parse_levels(task)
    replicas = int(os.environ.get("PHYS_EVAL_REPLICAS", "32"))
    num_envs = (
        int(os.environ.get("PHYS_EVAL_NUM_ENVS", "256"))
        if task == "clearance"
        else len(levels) * replicas
    )
    settle_s = float(os.environ.get("PHYS_EVAL_SETTLE_S", "2.0"))
    eval_s = float(os.environ.get("PHYS_EVAL_DURATION_S", "8.0"))
    total_s = settle_s + eval_s
    slope_static_hold = (
        os.environ.get("PHYS_EVAL_SLOPE_STATIC", "0") == "1"
    )
    slope_success_mode = os.environ.get(
        "PHYS_EVAL_SLOPE_SUCCESS_MODE", "hold"
    ).strip().lower()
    if slope_success_mode not in ("hold", "stand"):
        raise ValueError(
            "PHYS_EVAL_SLOPE_SUCCESS_MODE must be hold or stand"
        )
    speed_success_mode = os.environ.get(
        "PHYS_EVAL_SPEED_SUCCESS_MODE", "tracking"
    ).strip().lower()
    if speed_success_mode not in ("tracking", "stable_actual"):
        raise ValueError(
            "PHYS_EVAL_SPEED_SUCCESS_MODE must be tracking or stable_actual"
        )
    speed_slope_deg = float(
        os.environ.get("PHYS_EVAL_SPEED_SLOPE_DEG", "0.0")
    )
    if speed_slope_deg < 0.0 or speed_slope_deg >= 45.0:
        raise ValueError(
            "PHYS_EVAL_SPEED_SLOPE_DEG must be in [0, 45)"
        )
    if task != "speed" and speed_slope_deg != 0.0:
        raise ValueError(
            "PHYS_EVAL_SPEED_SLOPE_DEG is only valid for task=speed"
        )
    real_slope = task == "slope" or speed_slope_deg > 0.0
    out_dir = os.environ.get(
        "PHYS_EVAL_OUT",
        os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            "physical_eval",
            f"{task}_{run_name}_ckpt{checkpoint}_seed{seed}",
        ),
    )
    legacy_motor = float(os.environ.get("PHYS_EVAL_MOTOR", "1.0"))
    design = (
        float(os.environ.get("PHYS_EVAL_THIGH", "1.0")),
        float(os.environ.get("PHYS_EVAL_SHANK", "1.0")),
        float(os.environ.get("PHYS_EVAL_ABAD", legacy_motor)),
        float(os.environ.get("PHYS_EVAL_HIP", legacy_motor)),
        float(os.environ.get("PHYS_EVAL_KNEE", legacy_motor)),
        float(os.environ.get("PHYS_EVAL_WHEEL", legacy_motor)),
    )
    prefix = os.environ.get(
        "PHYS_EVAL_PREFIX",
        f"physeval_{task}_{os.getpid()}",
    )

    args = get_args()
    args.task = "wheelfoot_flat"
    args.headless = True
    args.num_envs = num_envs
    args.load_run = run_name
    args.checkpoint = checkpoint
    args.resume = True
    args.seed = seed

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        args.task,
        train_cfg.runner.experiment_name,
    )
    load_saved_run_configs(env_cfg, train_cfg, log_root, run_name)
    _configure_environment(
        env_cfg,
        task,
        levels,
        num_envs,
        total_s,
        speed_slope_deg,
    )
    _configure_fixed_design(env_cfg, design, seed, prefix)
    env_cfg.seed = seed

    started = time.time()
    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )
    level_ids = _level_ids(num_envs, len(levels), env.device)
    if task == "slope":
        if not torch.equal(level_ids, env.terrain_levels):
            raise RuntimeError(
                "real-slope terrain assignment does not match level grouping"
            )
        actual_slopes = np.asarray(
            env.cfg.terrain.evaluation_slope_degrees,
            dtype=np.float64,
        )
        expected_slopes = np.asarray(levels, dtype=np.float64)
        if not np.allclose(actual_slopes, expected_slopes):
            raise RuntimeError(
                "real-slope terrain degrees do not match requested values"
            )
    elif speed_slope_deg > 0.0:
        if torch.any(env.terrain_levels != 0):
            raise RuntimeError(
                "fixed speed-slope evaluation must use shared terrain row 0"
            )
        actual_slopes = np.asarray(
            env.cfg.terrain.evaluation_slope_degrees,
            dtype=np.float64,
        )
        if actual_slopes.shape != (1,) or not np.isclose(
            actual_slopes[0], speed_slope_deg
        ):
            raise RuntimeError(
                "shared speed-slope terrain does not match requested angle"
            )
    if task == "payload":
        if not torch.equal(level_ids, env.attached_payload_level_ids):
            raise RuntimeError("payload assignment does not match level grouping")
        actual_payload = env.attached_payload_mass.detach().cpu().numpy()
        expected_payload = np.asarray(levels)[level_ids.cpu().numpy()]
        if not np.allclose(actual_payload, expected_payload):
            raise RuntimeError("physical attached payload masses do not match levels")

    actual_morph = env.env_morphology[0].detach().cpu().numpy()
    actual_motor = env.env_motor_group_scale[0].detach().cpu().numpy()
    if bool(env.design_anchor_mask.any()):
        raise RuntimeError(
            "fixed-design evaluation must not contain nominal anchor envs"
        )
    if not np.allclose(actual_morph, design[:2]) or not np.allclose(
        actual_motor, design[2:]
    ):
        raise RuntimeError("fixed mechanical design was not applied")
    expected_dof_scales = torch.empty_like(env.motor_torque_limit_scale)
    for dof_id, dof_name in enumerate(env.dof_names):
        matches = [
            group_id
            for group_id, group in enumerate(MOTOR_GROUPS)
            if group in dof_name.lower()
        ]
        if len(matches) != 1:
            raise RuntimeError(f"cannot verify motor group for {dof_name}")
        expected_dof_scales[:, dof_id] = env.env_motor_group_scale[
            :, matches[0]
        ]
    if not torch.allclose(
        env.motor_torque_limit_scale, expected_dof_scales
    ):
        raise RuntimeError("per-DOF torque-limit scales do not match motor groups")
    body_props_0 = env.gym.get_actor_rigid_body_properties(
        env.envs[0], env.actor_handles[0]
    )
    robot_mass_kg = float(sum(prop.mass for prop in body_props_0))
    motor_host_body_masses_kg = {
        group: float(
            sum(
                prop.mass
                for prop, body_name in zip(body_props_0, env.body_names)
                if group in body_name.lower()
            )
        )
        for group in MOTOR_GROUPS
    }

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = run_name
    train_cfg.runner.checkpoint = checkpoint
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = runner.get_inference_policy(device=env.device)
    encoder = runner.get_inference_encoder(device=env.device)
    obs, obs_history, commands, _ = env.get_observations()
    expected_command_dim = 3 + int(env.policy_hardware_context_dim)
    if commands.shape != (num_envs, expected_command_dim):
        raise RuntimeError(
            "policy command/context shape mismatch: "
            f"expected {(num_envs, expected_command_dim)}, "
            f"got {tuple(commands.shape)}"
        )

    dt = float(env.dt)
    settle_steps = int(round(settle_s / dt))
    eval_steps = int(round(eval_s / dt))
    total_steps = settle_steps + eval_steps
    failed = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    level_tensor = torch.as_tensor(
        levels, dtype=torch.float, device=env.device
    )[level_ids]
    step_component = float(
        os.environ.get("PHYS_EVAL_STEP_COMPONENT", "0.40")
    )
    if task == "speed":
        target_step = torch.minimum(
            level_tensor,
            torch.full_like(level_tensor, step_component),
        )
        target_roll = level_tensor - target_step
    elif task == "clearance":
        target_roll = torch.full(
            (num_envs,),
            float(os.environ.get("PHYS_EVAL_CLEARANCE_ROLL", "0.30")),
            device=env.device,
        )
        target_step = torch.full(
            (num_envs,),
            float(os.environ.get("PHYS_EVAL_CLEARANCE_STEP", "0.40")),
            device=env.device,
        )
    elif task == "slope" and slope_static_hold:
        target_roll = torch.zeros(num_envs, device=env.device)
        target_step = torch.zeros_like(target_roll)
    else:
        target_total_fixed = torch.full(
            (num_envs,),
            float(os.environ.get("PHYS_EVAL_TEST_SPEED", "0.50")),
            device=env.device,
        )
        target_step = torch.minimum(
            target_total_fixed,
            torch.full_like(target_total_fixed, step_component),
        )
        target_roll = target_total_fixed - target_step
    target_total = target_roll + target_step
    if task == "slope":
        slope_rad = torch.deg2rad(level_tensor)
    elif task == "speed" and speed_slope_deg > 0.0:
        slope_rad = torch.full_like(
            target_total,
            np.deg2rad(speed_slope_deg),
        )
    else:
        slope_rad = torch.zeros_like(target_total)
    surface_tangent = torch.stack(
        (
            torch.cos(slope_rad),
            torch.zeros_like(slope_rad),
            torch.sin(slope_rad),
        ),
        dim=1,
    )
    surface_normal = torch.stack(
        (
            -torch.sin(slope_rad),
            torch.zeros_like(slope_rad),
            torch.cos(slope_rad),
        ),
        dim=1,
    )

    vx_samples = torch.zeros(eval_steps, num_envs, device=env.device)
    roll_samples = torch.zeros_like(vx_samples)
    pitch_samples = torch.zeros_like(vx_samples)
    slip_samples = torch.zeros_like(vx_samples)
    torque_sat_samples = torch.zeros_like(vx_samples)
    wheel_torque_ratio_samples = torch.zeros_like(vx_samples)
    wheel_peak_torque_ratio_samples = torch.zeros_like(vx_samples)
    wheel_torque_sat_samples = torch.zeros_like(vx_samples)
    wheel_power_limited_samples = torch.zeros_like(vx_samples)
    wheel_available_torque_samples = torch.zeros_like(vx_samples)
    wheel_abs_torque_samples = torch.zeros_like(vx_samples)
    wheel_power_samples = torch.zeros_like(vx_samples)
    progress_samples = torch.zeros_like(vx_samples)
    lateral_samples = torch.zeros_like(vx_samples)
    both_contact_samples = torch.zeros_like(vx_samples)
    energy = torch.zeros(num_envs, device=env.device)
    start_position = None
    cycles = []
    swing_active = torch.zeros(
        num_envs, 2, dtype=torch.bool, device=env.device
    )
    swing_air_steps = torch.zeros(
        num_envs, 2, dtype=torch.long, device=env.device
    )
    swing_peak = torch.zeros(num_envs, 2, device=env.device)
    previous_contact = (
        env.contact_forces[:, env.feet_indices, 2] > 1.0
    )
    wheel_dof_ids = torch.as_tensor(
        [
            dof_id
            for dof_id, dof_name in enumerate(env.dof_names)
            if "wheel" in dof_name.lower()
        ],
        dtype=torch.long,
        device=env.device,
    )
    if wheel_dof_ids.numel() != 2:
        raise RuntimeError(
            f"expected two wheel DOFs, got {wheel_dof_ids.numel()}"
        )

    print(
        f"[physical-eval] task={task} run={run_name} ckpt={checkpoint} "
        f"design={design} envs={num_envs} levels={levels} "
        f"speed_slope_deg={speed_slope_deg:g} "
        f"settle={settle_s:g}s eval={eval_s:g}s dt={dt:g}s",
        flush=True,
    )
    for step in range(total_steps):
        if step < settle_steps:
            _set_commands(
                env,
                torch.zeros_like(target_roll),
                torch.zeros_like(target_step),
            )
        else:
            _set_commands(env, target_roll, target_step)
            if step == settle_steps:
                start_position = env.root_states[
                    env.actor_indices, 0:3
                ].clone()
                previous_contact = (
                    env.contact_forces[:, env.feet_indices, 2] > 1.0
                )

        commands = env._get_policy_commands()
        with torch.no_grad():
            estimate = encoder(obs_history, obs)
            actions = policy(
                torch.cat((estimate, obs, commands), dim=-1).detach()
            )
        (
            obs,
            rewards,
            dones,
            _,
            obs_history,
            commands,
            _,
        ) = env.step(actions.detach())
        failed |= dones.bool()

        if step >= settle_steps:
            sample_id = step - settle_steps
            world_velocity = env.root_states[env.actor_indices, 7:10]
            vx = torch.sum(world_velocity * surface_tangent, dim=1)
            roll = torch.abs(_quaternion_roll(env.base_quat))
            pitch = torch.abs(_quaternion_pitch(env.base_quat))
            position_delta = (
                env.root_states[env.actor_indices, 0:3]
                - start_position
            )
            progress_now = torch.sum(
                position_delta * surface_tangent, dim=1
            )
            lateral_now = position_delta[:, 1]
            wheel_state = env.rigid_body_state.view(
                env.num_envs, env.num_bodies, 13
            )[:, env.feet_indices]
            wheel_angular_velocity = wheel_state[:, :, 10:13]
            contact_radius = (
                -float(env.cfg.asset.foot_radius)
                * surface_normal.unsqueeze(1)
            )
            contact_velocity = env.foot_velocities + torch.cross(
                wheel_angular_velocity,
                contact_radius.expand_as(wheel_angular_velocity),
                dim=2,
            )
            contact_slip_speed = torch.abs(
                torch.sum(
                    contact_velocity * surface_tangent.unsqueeze(1),
                    dim=2,
                )
            )
            contact = (
                torch.linalg.vector_norm(
                    env.contact_forces[:, env.feet_indices], dim=2
                )
                > 1.0
            ).float()
            contact_count = contact.sum(dim=1)
            mean_contact_slip = torch.where(
                contact_count > 0,
                (contact_slip_speed * contact).sum(dim=1)
                / contact_count.clamp_min(1.0),
                torch.zeros_like(vx),
            )
            slip = mean_contact_slip / torch.maximum(
                torch.abs(vx),
                torch.full_like(vx, 0.2),
            )
            peak_limits = getattr(
                env,
                "motor_peak_torque_limits",
                env.torque_limits.view(1, -1)
                * env.motor_torque_limit_scale,
            ).clamp_min(1.0e-6)
            effective_limits = getattr(
                env,
                "motor_effective_torque_limits",
                peak_limits,
            ).clamp_min(1.0e-6)
            power_limited_mask = getattr(
                env,
                "motor_power_limited_mask",
                torch.zeros_like(effective_limits, dtype=torch.bool),
            )
            torque_ratio = torch.abs(env.torques) / effective_limits
            peak_torque_ratio = torch.abs(env.torques) / peak_limits
            torque_sat = (torque_ratio >= 0.98).float().mean(dim=1)
            wheel_torque = env.torques[:, wheel_dof_ids]
            wheel_torque_ratio = torque_ratio[:, wheel_dof_ids]
            wheel_peak_torque_ratio = peak_torque_ratio[:, wheel_dof_ids]
            wheel_torque_ratio_mean = wheel_torque_ratio.mean(dim=1)
            wheel_peak_torque_ratio_mean = wheel_peak_torque_ratio.mean(dim=1)
            wheel_torque_sat = (
                wheel_torque_ratio >= 0.98
            ).float().mean(dim=1)
            wheel_power_limited = (
                power_limited_mask[:, wheel_dof_ids].float().mean(dim=1)
            )
            wheel_available_torque = effective_limits[
                :, wheel_dof_ids
            ].mean(dim=1)
            wheel_abs_torque = torch.abs(wheel_torque).mean(dim=1)
            wheel_power = torch.sum(
                torch.abs(
                    wheel_torque * env.dof_vel[:, wheel_dof_ids]
                ),
                dim=1,
            )
            vx_samples[sample_id] = vx
            roll_samples[sample_id] = roll
            pitch_samples[sample_id] = pitch
            slip_samples[sample_id] = slip
            torque_sat_samples[sample_id] = torque_sat
            wheel_torque_ratio_samples[sample_id] = (
                wheel_torque_ratio_mean
            )
            wheel_peak_torque_ratio_samples[sample_id] = (
                wheel_peak_torque_ratio_mean
            )
            wheel_torque_sat_samples[sample_id] = wheel_torque_sat
            wheel_power_limited_samples[sample_id] = wheel_power_limited
            wheel_available_torque_samples[sample_id] = (
                wheel_available_torque
            )
            wheel_abs_torque_samples[sample_id] = wheel_abs_torque
            wheel_power_samples[sample_id] = wheel_power
            progress_samples[sample_id] = progress_now
            lateral_samples[sample_id] = lateral_now
            both_contact_samples[sample_id] = (
                contact_count == 2
            ).float()
            energy += (
                torch.sum(torch.abs(env.torques * env.dof_vel), dim=1) * dt
            )

            if task == "clearance":
                contact = env.contact_forces[:, env.feet_indices, 2] > 1.0
                liftoff = previous_contact & (~contact)
                swing_active |= liftoff
                swing_air_steps = torch.where(
                    swing_active & (~contact),
                    swing_air_steps + 1,
                    swing_air_steps,
                )
                swing_peak = torch.where(
                    swing_active & (~contact),
                    torch.maximum(swing_peak, env.foot_heights),
                    swing_peak,
                )
                touchdown = swing_active & contact
                valid_touchdown = touchdown & (swing_air_steps >= 3)
                for env_id, foot_id in (
                    valid_touchdown.nonzero(as_tuple=False).detach().cpu().tolist()
                ):
                    cycles.append(
                        (
                            env_id,
                            foot_id,
                            float(swing_peak[env_id, foot_id].item()),
                            int(swing_air_steps[env_id, foot_id].item()),
                        )
                    )
                swing_active &= ~touchdown
                swing_air_steps = torch.where(
                    touchdown,
                    torch.zeros_like(swing_air_steps),
                    swing_air_steps,
                )
                swing_peak = torch.where(
                    touchdown,
                    torch.zeros_like(swing_peak),
                    swing_peak,
                )
                swing_active &= ~failed.unsqueeze(1)
                previous_contact = contact

        if step % max(int(round(2.0 / dt)), 1) == 0:
            print(
                f"[physical-eval] step={step}/{total_steps} "
                f"failed={int(failed.sum())}/{num_envs}",
                flush=True,
            )

    elapsed_s = time.time() - started
    vx_mean = vx_samples.mean(dim=0)
    vx_rmse = torch.sqrt(
        torch.square(vx_samples - target_total.unsqueeze(0)).mean(dim=0)
    )
    roll_p95 = torch.quantile(roll_samples, 0.95, dim=0)
    pitch_p95 = torch.quantile(pitch_samples, 0.95, dim=0)
    slip_mean = slip_samples.mean(dim=0)
    torque_sat_mean = torque_sat_samples.mean(dim=0)
    wheel_torque_ratio_mean = wheel_torque_ratio_samples.mean(dim=0)
    wheel_torque_ratio_p95 = torch.quantile(
        wheel_torque_ratio_samples, 0.95, dim=0
    )
    wheel_peak_torque_ratio_mean = wheel_peak_torque_ratio_samples.mean(dim=0)
    wheel_peak_torque_ratio_p95 = torch.quantile(
        wheel_peak_torque_ratio_samples, 0.95, dim=0
    )
    wheel_torque_sat_mean = wheel_torque_sat_samples.mean(dim=0)
    wheel_power_limited_mean = wheel_power_limited_samples.mean(dim=0)
    wheel_available_torque_mean = wheel_available_torque_samples.mean(dim=0)
    wheel_available_torque_p05 = torch.quantile(
        wheel_available_torque_samples, 0.05, dim=0
    )
    wheel_abs_torque_mean = wheel_abs_torque_samples.mean(dim=0)
    wheel_power_mean = wheel_power_samples.mean(dim=0)
    progress_abs_max = torch.abs(progress_samples).max(dim=0).values
    lateral_abs_max = torch.abs(lateral_samples).max(dim=0).values
    both_contact_fraction = both_contact_samples.mean(dim=0)
    final_position_delta = (
        env.root_states[env.actor_indices, 0:3] - start_position
    )
    progress = torch.sum(
        final_position_delta * surface_tangent, dim=1
    )
    world_x_progress = final_position_delta[:, 0]
    height_gain = final_position_delta[:, 2]
    if task == "speed" and speed_slope_deg > 0.0:
        positive_uphill_progress = (
            (progress > 0.0)
            & (world_x_progress > 0.0)
        )
    else:
        positive_uphill_progress = progress > 0.0

    slope_stand_success = None
    slope_hold_success = None
    speed_stable_actual_score = None
    if task == "speed":
        invalid_direction = (
            ~positive_uphill_progress
            if speed_slope_deg > 0.0
            else torch.zeros_like(failed)
        )
        speed_stable_actual_score = torch.where(
            failed | invalid_direction,
            torch.zeros_like(vx_mean),
            torch.clamp(vx_mean, min=0.0),
        )
        if speed_success_mode == "stable_actual":
            motion_success = torch.ones_like(failed)
        else:
            tracking_limit = torch.maximum(
                torch.full_like(target_total, 0.25),
                0.20 * target_total,
            )
            motion_success = (
                (vx_mean >= 0.80 * target_total)
                & (vx_rmse <= tracking_limit)
                & (roll_p95 <= np.deg2rad(15.0))
            )
    elif task == "slope" and slope_static_hold:
        max_roll_deg = float(
            os.environ.get("PHYS_EVAL_HOLD_MAX_ROLL_DEG", "15.0")
        )
        max_pitch_deg = float(
            os.environ.get("PHYS_EVAL_HOLD_MAX_PITCH_DEG", "20.0")
        )
        max_progress_m = float(
            os.environ.get("PHYS_EVAL_HOLD_MAX_PROGRESS_M", "0.25")
        )
        max_lateral_m = float(
            os.environ.get("PHYS_EVAL_HOLD_MAX_LATERAL_M", "0.10")
        )
        min_both_contact = float(
            os.environ.get("PHYS_EVAL_HOLD_MIN_BOTH_CONTACT", "0.95")
        )
        max_slip_ratio = float(
            os.environ.get("PHYS_EVAL_HOLD_MAX_SLIP_RATIO", "0.50")
        )
        stand_motion_success = (
            (
                roll_p95
                <= np.deg2rad(max_roll_deg)
            )
            & (
                pitch_p95
                <= np.deg2rad(max_pitch_deg)
            )
        )
        hold_motion_success = (
            stand_motion_success
            & (
                progress_abs_max <= max_progress_m
            )
            & (
                lateral_abs_max <= max_lateral_m
            )
            & (
                both_contact_fraction >= min_both_contact
            )
            & (
                slip_mean <= max_slip_ratio
            )
        )
        slope_stand_success = (~failed) & stand_motion_success
        slope_hold_success = (~failed) & hold_motion_success
        motion_success = (
            stand_motion_success
            if slope_success_mode == "stand"
            else hold_motion_success
        )
    elif task in ("slope", "payload"):
        motion_success = (
            (vx_mean >= 0.35)
            & (vx_rmse <= 0.30)
            & (roll_p95 <= np.deg2rad(15.0))
        )
    else:
        motion_success = (
            (vx_mean >= 0.45)
            & (roll_p95 <= np.deg2rad(15.0))
        )
    success = (~failed) & motion_success

    per_level = []
    success_rates = []
    for level_id, level in enumerate(levels):
        mask = level_ids == level_id
        success_rate = float(success[mask].float().mean().item())
        success_rates.append(success_rate)
        level_result = {
                "level": float(level),
                "trials": int(mask.sum().item()),
                "successes": int(success[mask].sum().item()),
                "success_rate": success_rate,
                "fall_or_reset_rate": float(
                    failed[mask].float().mean().item()
                ),
                "vx_mean_mps": float(vx_mean[mask].mean().item()),
                "vx_rmse_mean_mps": float(vx_rmse[mask].mean().item()),
                "progress_mean_m": float(progress[mask].mean().item()),
                "roll_p95_mean_deg": float(
                    torch.rad2deg(roll_p95[mask]).mean().item()
                ),
                "pitch_p95_mean_deg": float(
                    torch.rad2deg(pitch_p95[mask]).mean().item()
                ),
                "progress_abs_max_mean_m": float(
                    progress_abs_max[mask].mean().item()
                ),
                "lateral_abs_max_mean_m": float(
                    lateral_abs_max[mask].mean().item()
                ),
                "both_wheels_contact_fraction_mean": float(
                    both_contact_fraction[mask].mean().item()
                ),
                "slip_ratio_mean": float(slip_mean[mask].mean().item()),
                "torque_saturation_fraction_mean": float(
                    torque_sat_mean[mask].mean().item()
                ),
                "wheel_torque_ratio_mean": float(
                    wheel_torque_ratio_mean[mask].mean().item()
                ),
                "wheel_torque_ratio_p95_mean": float(
                    wheel_torque_ratio_p95[mask].mean().item()
                ),
                "wheel_peak_torque_ratio_mean": float(
                    wheel_peak_torque_ratio_mean[mask].mean().item()
                ),
                "wheel_peak_torque_ratio_p95_mean": float(
                    wheel_peak_torque_ratio_p95[mask].mean().item()
                ),
                "wheel_torque_saturation_fraction_mean": float(
                    wheel_torque_sat_mean[mask].mean().item()
                ),
                "wheel_power_limited_fraction_mean": float(
                    wheel_power_limited_mean[mask].mean().item()
                ),
                "wheel_available_torque_mean_nm": float(
                    wheel_available_torque_mean[mask].mean().item()
                ),
                "wheel_available_torque_p05_mean_nm": float(
                    wheel_available_torque_p05[mask].mean().item()
                ),
                "wheel_abs_torque_mean_nm": float(
                    wheel_abs_torque_mean[mask].mean().item()
                ),
                "wheel_mechanical_power_mean_w": float(
                    wheel_power_mean[mask].mean().item()
                ),
                "world_x_progress_mean_m": float(
                    world_x_progress[mask].mean().item()
                ),
                "height_gain_mean_m": float(
                    height_gain[mask].mean().item()
                ),
                "positive_uphill_progress_rate": float(
                    positive_uphill_progress[mask].float().mean().item()
                ),
                "mechanical_energy_mean_j": float(
                    energy[mask].mean().item()
                ),
            }
        if slope_stand_success is not None:
            level_result["stand_success_rate"] = float(
                slope_stand_success[mask].float().mean().item()
            )
            level_result["hold_success_rate"] = float(
                slope_hold_success[mask].float().mean().item()
            )
        if speed_stable_actual_score is not None:
            level_result["no_reset_rate"] = float(
                (~failed[mask]).float().mean().item()
            )
            level_result["stable_actual_speed_q10_mps"] = float(
                torch.quantile(
                    speed_stable_actual_score[mask], 0.10
                ).item()
            )
        per_level.append(level_result)

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw_env_metrics.npz")
    np.savez_compressed(
        raw_path,
        level=np.asarray(levels, dtype=np.float64)[
            level_ids.detach().cpu().numpy()
        ],
        level_id=level_ids.detach().cpu().numpy(),
        failed=failed.detach().cpu().numpy(),
        success=success.detach().cpu().numpy(),
        vx_mean=vx_mean.detach().cpu().numpy(),
        vx_rmse=vx_rmse.detach().cpu().numpy(),
        progress=progress.detach().cpu().numpy(),
        roll_p95_rad=roll_p95.detach().cpu().numpy(),
        pitch_p95_rad=pitch_p95.detach().cpu().numpy(),
        progress_abs_max=progress_abs_max.detach().cpu().numpy(),
        lateral_abs_max=lateral_abs_max.detach().cpu().numpy(),
        both_contact_fraction=both_contact_fraction.detach().cpu().numpy(),
        slip_mean=slip_mean.detach().cpu().numpy(),
        torque_sat_mean=torque_sat_mean.detach().cpu().numpy(),
        wheel_torque_ratio_mean=(
            wheel_torque_ratio_mean.detach().cpu().numpy()
        ),
        wheel_torque_ratio_p95=(
            wheel_torque_ratio_p95.detach().cpu().numpy()
        ),
        wheel_peak_torque_ratio_mean=(
            wheel_peak_torque_ratio_mean.detach().cpu().numpy()
        ),
        wheel_peak_torque_ratio_p95=(
            wheel_peak_torque_ratio_p95.detach().cpu().numpy()
        ),
        wheel_torque_sat_mean=(
            wheel_torque_sat_mean.detach().cpu().numpy()
        ),
        wheel_power_limited_mean=(
            wheel_power_limited_mean.detach().cpu().numpy()
        ),
        wheel_available_torque_mean_nm=(
            wheel_available_torque_mean.detach().cpu().numpy()
        ),
        wheel_available_torque_p05_nm=(
            wheel_available_torque_p05.detach().cpu().numpy()
        ),
        wheel_abs_torque_mean_nm=(
            wheel_abs_torque_mean.detach().cpu().numpy()
        ),
        wheel_mechanical_power_mean_w=(
            wheel_power_mean.detach().cpu().numpy()
        ),
        world_x_progress=world_x_progress.detach().cpu().numpy(),
        height_gain=height_gain.detach().cpu().numpy(),
        positive_uphill_progress=(
            positive_uphill_progress.detach().cpu().numpy()
        ),
        energy_j=energy.detach().cpu().numpy(),
        clearance_cycles=np.asarray(cycles, dtype=np.float64).reshape(-1, 4),
        slope_stand_success=(
            slope_stand_success.detach().cpu().numpy()
            if slope_stand_success is not None
            else np.asarray([], dtype=np.bool_)
        ),
        slope_hold_success=(
            slope_hold_success.detach().cpu().numpy()
            if slope_hold_success is not None
            else np.asarray([], dtype=np.bool_)
        ),
        speed_stable_actual_score=(
            speed_stable_actual_score.detach().cpu().numpy()
            if speed_stable_actual_score is not None
            else np.asarray([], dtype=np.float32)
        ),
    )

    result = {
        "protocol_version": (
            "sixd_uphill_speed_power_v2_2026-07-27"
            if env_cfg.domain_rand.motor_power_envelope
            else "sixd_single_objective_v1_2026-07-26"
        ),
        "task": task,
        "run": run_name,
        "checkpoint": checkpoint,
        "seed": seed,
        "design": {
            "thigh": design[0],
            "shank": design[1],
            "abad": design[2],
            "hip": design[3],
            "knee": design[4],
            "wheel": design[5],
        },
        "motor_mass_model": {
            "type": "baseline_anchored_power_law",
            "exponent": float(env_cfg.domain_rand.motor_mass_exponent),
            "nominal_masses_kg": [
                float(value)
                for value in env_cfg.domain_rand.motor_nominal_masses_kg
            ],
        },
        "motor_power_model": {
            "enabled": bool(
                env_cfg.domain_rand.motor_power_envelope
            ),
            "type": "constant_torque_constant_power",
            "specific_power_w_per_kg": float(
                env_cfg.domain_rand.motor_specific_power_w_per_kg
            ),
            "power_exponent": float(
                env_cfg.domain_rand.motor_power_exponent
            ),
            "min_speed_rad_s": float(
                env_cfg.domain_rand.motor_power_min_speed_rad_s
            ),
            "formula": (
                "min(k*tau_peak_nominal, "
                "m0*specific_power*k**power_exponent/max(abs(omega),eps))"
            ),
        },
        "robot_mass_kg_env0": robot_mass_kg,
        "motor_host_body_masses_kg_env0": motor_host_body_masses_kg,
        "base_height_target_m": float(
            env._base_height_target_tensor().mean().item()
        ),
        "num_envs": num_envs,
        "replicas_per_level": replicas if task != "clearance" else None,
        "settle_s": settle_s,
        "evaluation_s": eval_s,
        "elapsed_s": elapsed_s,
        "friction": float(env_cfg.terrain.static_friction),
        "speed_slope_deg": speed_slope_deg,
        "positive_speed_direction": (
            "+world_x and +world_z along the ascending terrain tangent"
            if task == "speed" and speed_slope_deg > 0.0
            else "+forward"
        ),
        "terrain_discretization": (
            {
                "horizontal_scale_m": float(
                    env_cfg.terrain.horizontal_scale
                ),
                "vertical_scale_m": float(env_cfg.terrain.vertical_scale),
            }
            if real_slope
            else None
        ),
        "command_decomposition": {
            "step_component_mps": step_component,
            "total_speed_is_roll_plus_step": True,
            "slope_static_hold": slope_static_hold,
            "speed_fixed_slope_deg": speed_slope_deg,
        },
        "slip_definition": (
            "mean absolute tangent velocity of contacting wheel-ground points "
            "divided by max(abs(surface speed), 0.2 m/s)"
        ),
        "levels": levels,
        "per_level": per_level,
        "raw_path": raw_path,
    }
    if task == "clearance":
        result["summary"] = _clearance_summary(cycles, failed, num_envs)
    elif task == "speed" and speed_success_mode == "stable_actual":
        training_limit = _training_speed_limit(
            log_root,
            run_name,
            getattr(env_cfg.commands, "smooth_max_lin_vel_x", 0.0),
        )
        eligible = [
            (index, level_result)
            for index, level_result in enumerate(per_level)
            if float(level_result["level"]) <= training_limit + 1.0e-9
        ]
        if not eligible:
            raise RuntimeError(
                "stable-actual speed evaluation has no in-domain command level"
            )
        best_index, best_level_result = max(
            eligible,
            key=lambda item: (
                item[1]["stable_actual_speed_q10_mps"],
                -item[1]["level"],
            ),
        )
        metric_value = best_level_result[
            "stable_actual_speed_q10_mps"
        ]
        considered_levels = [
            float(level_result["level"])
            for _, level_result in eligible
        ]
        result["summary"] = {
            "metric_name": (
                "V_uphill_stable_actual_90"
                if speed_slope_deg > 0.0
                else "V_stable_actual_90"
            ),
            "metric_value": _safe_float(metric_value),
            "unit": "m/s",
            "success_mode": speed_success_mode,
            "reliability_quantile": 0.10,
            "command_at_metric_mps": float(
                best_level_result["level"]
            ),
            "no_reset_rate_at_command": float(
                best_level_result["no_reset_rate"]
            ),
            "actual_speed_mean_at_command_mps": float(
                best_level_result["vx_mean_mps"]
            ),
            "fixed_slope_deg": speed_slope_deg,
            "positive_speed_direction": (
                "+world_x and +world_z along the ascending terrain tangent"
                if speed_slope_deg > 0.0
                else "+forward"
            ),
            "positive_uphill_progress_rate_at_command": float(
                best_level_result["positive_uphill_progress_rate"]
            ),
            "world_x_progress_mean_at_command_m": float(
                best_level_result["world_x_progress_mean_m"]
            ),
            "height_gain_mean_at_command_m": float(
                best_level_result["height_gain_mean_m"]
            ),
            "wheel_torque_ratio_mean_at_command": float(
                best_level_result["wheel_torque_ratio_mean"]
            ),
            "wheel_torque_ratio_p95_mean_at_command": float(
                best_level_result["wheel_torque_ratio_p95_mean"]
            ),
            "wheel_peak_torque_ratio_mean_at_command": float(
                best_level_result["wheel_peak_torque_ratio_mean"]
            ),
            "wheel_peak_torque_ratio_p95_mean_at_command": float(
                best_level_result["wheel_peak_torque_ratio_p95_mean"]
            ),
            "wheel_torque_saturation_fraction_at_command": float(
                best_level_result[
                    "wheel_torque_saturation_fraction_mean"
                ]
            ),
            "wheel_power_limited_fraction_at_command": float(
                best_level_result["wheel_power_limited_fraction_mean"]
            ),
            "wheel_available_torque_mean_at_command_nm": float(
                best_level_result["wheel_available_torque_mean_nm"]
            ),
            "wheel_available_torque_p05_at_command_nm": float(
                best_level_result["wheel_available_torque_p05_mean_nm"]
            ),
            "wheel_abs_torque_mean_at_command_nm": float(
                best_level_result["wheel_abs_torque_mean_nm"]
            ),
            "wheel_mechanical_power_mean_at_command_w": float(
                best_level_result["wheel_mechanical_power_mean_w"]
            ),
            "training_command_max_mps": training_limit,
            "in_domain_command_levels_mps": considered_levels,
            "contains_ood_levels": max(levels) > training_limit,
            "right_censored_by_training_command_domain": bool(
                best_index == eligible[-1][0]
                and np.isclose(
                    best_level_result["level"],
                    max(considered_levels),
                )
            ),
            "success_definition": {
                "failed_or_reset_trial_score_mps": 0.0,
                "nonfailed_trial_score": (
                    "mean actual forward speed, clipped below at zero"
                ),
                "command_tracking_error_used": False,
                "roll_threshold_used": False,
                "metric": (
                    "maximum 10th-percentile trial score over "
                    "in-training-domain command levels"
                ),
                "uphill_direction_enforcement": (
                    "score uses velocity projected onto the +x/+z "
                    "ascending terrain tangent and requires positive "
                    "net +world-x displacement; downhill trials score zero"
                    if speed_slope_deg > 0.0
                    else "not applicable on flat terrain"
                ),
            },
        }
    else:
        limit, continuous_mask = _continuous_reliable_limit(
            levels, success_rates
        )
        metric_names = {
            "slope": (
                (
                    "theta_stand_90"
                    if slope_static_hold and slope_success_mode == "stand"
                    else "theta_hold_90"
                    if slope_static_hold
                    else "theta90"
                ),
                "deg",
            ),
            "payload": ("M90", "kg"),
            "speed": ("V90", "m/s"),
        }
        metric_name, unit = metric_names[task]
        result["summary"] = {
            "metric_name": metric_name,
            "metric_value": _safe_float(limit),
            "metric_value_interpolated": _safe_float(
                _interpolated_reliable_limit(
                    levels, success_rates
                )
            ),
            "unit": unit,
            "success_threshold": 0.90,
            "continuous_reliable_mask": continuous_mask,
            "lowest_level_passes": success_rates[0] >= 0.90,
            "highest_level_fails": success_rates[-1] < 0.90,
            "continuous_domain_right_censored": all(continuous_mask),
        }
        if task == "slope" and slope_static_hold:
            result["summary"]["success_mode"] = slope_success_mode
            result["summary"]["success_definition"] = {
                "mode": slope_success_mode,
                "zero_motion_command": True,
                "no_fall_or_episode_reset": True,
                "maximum_roll_deg": float(
                    os.environ.get(
                        "PHYS_EVAL_HOLD_MAX_ROLL_DEG", "15.0"
                    )
                ),
                "maximum_pitch_deg": float(
                    os.environ.get(
                        "PHYS_EVAL_HOLD_MAX_PITCH_DEG", "20.0"
                    )
                ),
            }
            hold_thresholds = {
                "maximum_along_slope_excursion_m": float(
                    os.environ.get(
                        "PHYS_EVAL_HOLD_MAX_PROGRESS_M", "0.25"
                    )
                ),
                "maximum_lateral_excursion_m": float(
                    os.environ.get(
                        "PHYS_EVAL_HOLD_MAX_LATERAL_M", "0.10"
                    )
                ),
                "minimum_both_wheels_contact_fraction": float(
                    os.environ.get(
                        "PHYS_EVAL_HOLD_MIN_BOTH_CONTACT", "0.95"
                    )
                ),
                "maximum_slip_ratio": float(
                    os.environ.get(
                        "PHYS_EVAL_HOLD_MAX_SLIP_RATIO", "0.50"
                    )
                ),
            }
            if slope_success_mode == "stand":
                result["summary"]["diagnostic_thresholds"] = hold_thresholds
                result["summary"]["diagnostic_only"] = {
                    "along_slope_excursion": True,
                    "lateral_excursion": True,
                    "both_wheels_contact_fraction": True,
                    "slip_ratio": True,
                }
            else:
                result["summary"]["success_definition"].update(
                    hold_thresholds
                )
        if task == "speed":
            training_limit = _training_speed_limit(
                log_root,
                run_name,
                getattr(env_cfg.commands, "smooth_max_lin_vel_x", 0.0),
            )
            result["summary"]["training_command_max_mps"] = training_limit
            result["summary"]["contains_ood_levels"] = max(levels) > training_limit

    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, ensure_ascii=True)
    print(
        f"[physical-eval] summary={result['summary']} elapsed={elapsed_s:.1f}s",
        flush=True,
    )
    print(f"[physical-eval] saved {result_path}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
