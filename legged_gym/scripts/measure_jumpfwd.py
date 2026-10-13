"""Headless measurement for wheelfoot_jump_forward policies.

Configure with environment variables:
  JUMPFWD_RUN, JUMPFWD_CKPT, JUMPFWD_NUM_ENVS, JUMPFWD_STEPS, JUMPFWD_CSV
  JUMPFWD_GRID_HEIGHTS, JUMPFWD_GRID_DISTANCES, JUMPFWD_GRID_REPLICAS
  JUMPFWD_SEED, JUMPFWD_NOMINAL_EVAL
  JUMPFWD_EVAL_THIGH, JUMPFWD_EVAL_SHANK,
  JUMPFWD_EVAL_ABAD/HIP/KNEE/WHEEL, JUMPFWD_EVAL_PREFIX
"""
import csv
import json
import os
import sys

import isaacgym  # noqa: F401
from isaacgym import gymapi  # noqa: F401
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.saved_config import load_saved_run_configs
import numpy as np
import torch


def parse_axis(name):
    raw = os.environ.get(name, "").strip()
    return [float(value) for value in raw.split(",") if value.strip()] if raw else []


RUN = os.environ.get("JUMPFWD_RUN", "-1")
CKPT = int(os.environ.get("JUMPFWD_CKPT", "-1"))
STEPS = int(os.environ.get("JUMPFWD_STEPS", "2500"))
CSV_PATH = os.environ.get("JUMPFWD_CSV", "")
SUMMARY_PATH = os.environ.get(
    "JUMPFWD_SUMMARY",
    os.path.splitext(CSV_PATH)[0] + "_summary.json" if CSV_PATH else "",
)
SEED = int(os.environ.get("JUMPFWD_SEED", "1"))
NOMINAL_EVAL = os.environ.get("JUMPFWD_NOMINAL_EVAL", "0") == "1"
FLAT_EVAL = os.environ.get("JUMPFWD_FLAT_EVAL", "0") == "1"
EVAL_FRICTION = float(os.environ.get("JUMPFWD_EVAL_FRICTION", "0.8"))
MOTOR_GROUPS = ("abad", "hip", "knee", "wheel")
_LEGACY_MOTOR_RAW = os.environ.get(
    "JUMPFWD_EVAL_MOTOR", ""
).strip()
_GROUP_MOTOR_RAW = tuple(
    os.environ.get(
        f"JUMPFWD_EVAL_{group.upper()}",
        _LEGACY_MOTOR_RAW,
    ).strip()
    for group in MOTOR_GROUPS
)
DESIGN_RAW = (
    os.environ.get("JUMPFWD_EVAL_THIGH", "").strip(),
    os.environ.get("JUMPFWD_EVAL_SHANK", "").strip(),
    *_GROUP_MOTOR_RAW,
)
if any(DESIGN_RAW) and not all(DESIGN_RAW):
    raise ValueError(
        "fixed jump design requires thigh, shank, and all four motor groups; "
        "legacy JUMPFWD_EVAL_MOTOR may supply all four groups"
    )
FIXED_DESIGN = tuple(float(value) for value in DESIGN_RAW) if all(DESIGN_RAW) else None
DESIGN_PREFIX = os.environ.get("JUMPFWD_EVAL_PREFIX", "jumpdesign")
GRID_HEIGHTS = parse_axis("JUMPFWD_GRID_HEIGHTS")
GRID_DISTANCES = parse_axis("JUMPFWD_GRID_DISTANCES")
GRID_REPLICAS = int(os.environ.get("JUMPFWD_GRID_REPLICAS", "16"))
if bool(GRID_HEIGHTS) != bool(GRID_DISTANCES):
    raise ValueError("JUMPFWD_GRID_HEIGHTS and JUMPFWD_GRID_DISTANCES must be set together")
GRID_POINTS = [(height, distance) for height in GRID_HEIGHTS for distance in GRID_DISTANCES]
NUM_ENVS = len(GRID_POINTS) * GRID_REPLICAS if GRID_POINTS else int(os.environ.get("JUMPFWD_NUM_ENVS", "64"))
if GRID_POINTS and GRID_REPLICAS < 1:
    raise ValueError("JUMPFWD_GRID_REPLICAS must be positive")
PHYSICAL_SUCCESS_RATE = float(os.environ.get("JUMPFWD_SUCCESS_RATE", "0.90"))
PHYSICAL_MIN_QUALITY = float(os.environ.get("JUMPFWD_SUCCESS_MIN_QUALITY", "0.35"))
PHYSICAL_MAX_ROLL_RAD = np.deg2rad(
    float(os.environ.get("JUMPFWD_SUCCESS_MAX_ROLL_DEG", "15.0"))
)
PHYSICAL_MAX_LATERAL_M = float(
    os.environ.get("JUMPFWD_SUCCESS_MAX_LATERAL_M", "0.08")
)
PHYSICAL_MAX_LAT_RATIO = float(
    os.environ.get("JUMPFWD_SUCCESS_MAX_LAT_RATIO", "0.55")
)
PHYSICAL_RECOVERY_S = float(
    os.environ.get("JUMPFWD_SUCCESS_RECOVERY_S", "0.50")
)
PHYSICAL_RECOVERY_MAX_SPEED = float(
    os.environ.get("JUMPFWD_SUCCESS_RECOVERY_MAX_SPEED", "1.0")
)
PHYSICAL_RECOVERY_MAX_HEIGHT_ERROR = float(
    os.environ.get("JUMPFWD_SUCCESS_RECOVERY_MAX_HEIGHT_ERROR", "0.40")
)


def configure_nominal_eval(env_cfg):
    disabled = (
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
        "randomize_morphology",
        "randomize_motor_design",
        "attached_payload",
    )
    for name in disabled:
        setattr(env_cfg.domain_rand, name, False)
    env_cfg.domain_rand.design_anchor_fraction = 0.0


def configure_flat_eval(env_cfg):
    if not FLAT_EVAL:
        return
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.static_friction = EVAL_FRICTION
    env_cfg.terrain.dynamic_friction = EVAL_FRICTION
    env_cfg.terrain.restitution = 0.0


def configure_fixed_design(env_cfg):
    if FIXED_DESIGN is None:
        return
    thigh, shank, abad, hip, knee, wheel = FIXED_DESIGN
    env_cfg.domain_rand.randomize_morphology = True
    env_cfg.domain_rand.design_anchor_fraction = 0.0
    env_cfg.domain_rand.morphology_regenerate = True
    env_cfg.domain_rand.morphology_num_buckets = 1
    env_cfg.domain_rand.morphology_thigh_scale_range = [thigh, thigh]
    env_cfg.domain_rand.morphology_shank_scale_range = [shank, shank]
    env_cfg.domain_rand.morphology_prefix = DESIGN_PREFIX
    env_cfg.domain_rand.morphology_seed = SEED
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
    env_cfg.domain_rand.motor_design_seed = SEED


def print_tracking(name, command, actual):
    error = actual - command
    mae = np.abs(error).mean()
    rmse = np.sqrt(np.square(error).mean())
    bias = error.mean()
    corr = np.corrcoef(command, actual)[0, 1] if np.std(command) > 1e-8 and np.std(actual) > 1e-8 else float("nan")
    slope, intercept = np.polyfit(command, actual, 1) if len(command) >= 2 and np.std(command) > 1e-8 else (float("nan"), float("nan"))
    print(
        f"[FWD-MEAS] ★{name} command→actual: MAE={mae:.3f}m RMSE={rmse:.3f}m "
        f"bias={bias:+.3f}m corr={corr:.3f} fit y={slope:.3f}x{intercept:+.3f} "
        f"|err|≤3cm/5cm={100*np.mean(np.abs(error)<=0.03):.0f}%/{100*np.mean(np.abs(error)<=0.05):.0f}%",
        flush=True,
    )
    edges = np.linspace(command.min(), command.max(), 5)
    for bin_idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (command >= lo) & ((command <= hi) if bin_idx == len(edges) - 2 else (command < hi))
        if mask.any():
            print(
                f"[FWD-MEAS]   {name} cmd∈[{lo:.3f},{hi:.3f}] n={mask.sum()}: "
                f"cmd={command[mask].mean():.3f} actual={actual[mask].mean():.3f} "
                f"MAE={np.abs(error[mask]).mean():.3f}",
                flush=True,
            )


def is_physical_success(row):
    return (
        int(row["takeoffs"]) == 1
        and float(row["takeoff_quality"]) >= PHYSICAL_MIN_QUALITY
        and float(row["max_roll"]) <= PHYSICAL_MAX_ROLL_RAD
        and float(row["max_lat_ratio"]) <= PHYSICAL_MAX_LAT_RATIO
        and abs(float(row["land_lateral_disp"])) <= PHYSICAL_MAX_LATERAL_M
        and float(row["height"]) > 0.01
        and float(row["disp"]) >= 0.0
        and int(row["recovery_success"]) == 1
    )


def physical_grid_summary(
    jumps,
    trigger_counts,
    timeout_counts,
    reset_terminal_counts,
    grid_ids,
):
    if not GRID_POINTS or grid_ids is None:
        return None
    trigger_counts_np = trigger_counts.detach().cpu().numpy()
    timeout_counts_np = timeout_counts.detach().cpu().numpy()
    reset_terminal_counts_np = (
        reset_terminal_counts.detach().cpu().numpy()
    )
    grid_ids_np = grid_ids.detach().cpu().numpy()
    per_grid = []
    reliable_points = []
    for grid_id, (height_cmd, distance_cmd) in enumerate(GRID_POINTS):
        trigger_count = int(trigger_counts_np[grid_ids_np == grid_id].sum())
        timeout_count = int(
            timeout_counts_np[grid_ids_np == grid_id].sum()
        )
        reset_terminal_count = int(
            reset_terminal_counts_np[grid_ids_np == grid_id].sum()
        )
        rows = [row for row in jumps if int(row["grid_id"]) == grid_id]
        successful = [row for row in rows if is_physical_success(row)]
        recovery_unresolved_count = sum(
            int(row["recovery_unresolved"]) for row in rows
        )
        terminal_count = len(rows) + timeout_count + reset_terminal_count
        physical_resolved_count = (
            terminal_count - recovery_unresolved_count
        )
        success_rate = len(successful) / max(physical_resolved_count, 1)
        record = {
            "grid_id": grid_id,
            "height_command_m": float(height_cmd),
            "distance_command_m": float(distance_cmd),
            "trigger_count": trigger_count,
            "touchdown_count": len(rows),
            "timeout_count": timeout_count,
            "reset_terminal_count": reset_terminal_count,
            "terminal_count": terminal_count,
            "physical_resolved_count": physical_resolved_count,
            "physical_resolution_rate": (
                physical_resolved_count / max(trigger_count, 1)
            ),
            "recovery_unresolved_count": recovery_unresolved_count,
            "physical_success_count": len(successful),
            "physical_success_rate": success_rate,
            "reliable": bool(
                trigger_count > 0 and success_rate >= PHYSICAL_SUCCESS_RATE
            ),
        }
        if successful:
            actual_height = np.asarray(
                [float(row["height"]) for row in successful], dtype=float
            )
            actual_distance = np.asarray(
                [float(row["disp"]) for row in successful], dtype=float
            )
            record.update(
                {
                    "actual_height_mean_m": float(actual_height.mean()),
                    "actual_height_q10_m": float(
                        np.percentile(actual_height, 10)
                    ),
                    "actual_height_std_m": float(actual_height.std()),
                    "actual_distance_mean_m": float(actual_distance.mean()),
                    "actual_distance_q10_m": float(
                        np.percentile(actual_distance, 10)
                    ),
                    "actual_distance_std_m": float(actual_distance.std()),
                }
            )
            if record["reliable"]:
                reliable_points.append(
                    (
                        max(record["actual_height_q10_m"], 0.0),
                        max(record["actual_distance_q10_m"], 0.0),
                        grid_id,
                    )
                )
        per_grid.append(record)

    pareto = []
    for height, distance, grid_id in reliable_points:
        dominated = any(
            other_height >= height
            and other_distance >= distance
            and (
                other_height > height
                or other_distance > distance
            )
            for other_height, other_distance, _ in reliable_points
        )
        if not dominated:
            pareto.append(
                {
                    "grid_id": grid_id,
                    "height_q10_m": height,
                    "distance_q10_m": distance,
                }
            )
    pareto.sort(key=lambda row: row["height_q10_m"])

    area = 0.0
    previous_height = 0.0
    for height in sorted(set(point[0] for point in reliable_points)):
        distance_envelope = max(
            point[1] for point in reliable_points if point[0] >= height
        )
        area += max(height - previous_height, 0.0) * distance_envelope
        previous_height = height
    return {
        "metric_name": "A_jump",
        "metric_value_m2": area,
        "physical_success_threshold": PHYSICAL_SUCCESS_RATE,
        "success_criteria": {
            "single_takeoff": True,
            "minimum_takeoff_quality": PHYSICAL_MIN_QUALITY,
            "maximum_roll_deg": float(np.degrees(PHYSICAL_MAX_ROLL_RAD)),
            "maximum_lateral_displacement_m": PHYSICAL_MAX_LATERAL_M,
            "maximum_lateral_ratio": PHYSICAL_MAX_LAT_RATIO,
            "recovery_window_s": PHYSICAL_RECOVERY_S,
            "recovery_max_speed_mps": PHYSICAL_RECOVERY_MAX_SPEED,
            "recovery_max_height_error_m": PHYSICAL_RECOVERY_MAX_HEIGHT_ERROR,
            "stable_recovery_required": True,
        },
        "reliable_grid_count": len(reliable_points),
        "total_grid_count": len(GRID_POINTS),
        "reliable_height_max_m": (
            max(point[0] for point in reliable_points)
            if reliable_points
            else None
        ),
        "reliable_distance_max_m": (
            max(point[1] for point in reliable_points)
            if reliable_points
            else None
        ),
        "pareto_front": pareto,
        "per_grid": per_grid,
    }


def main():
    args = get_args()
    args.task = "wheelfoot_jump_forward"
    args.headless = True
    args.num_envs = NUM_ENVS
    args.load_run = RUN
    args.checkpoint = CKPT
    args.resume = True
    args.seed = SEED

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        args.task,
        train_cfg.runner.experiment_name,
    )
    load_saved_run_configs(env_cfg, train_cfg, log_root, RUN)
    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.seed = SEED
    if NOMINAL_EVAL:
        configure_nominal_eval(env_cfg)
    configure_flat_eval(env_cfg)
    configure_fixed_design(env_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    control_contract = {
        name: getattr(env.cfg.control, name, None)
        for name in (
            "jump_knee_height_residual",
            "jump_knee_height_bias",
            "jump_hip_distance_residual",
            "jump_hip_distance_bias",
            "jump_hip_landing_residual",
            "jump_hip_landing_bias",
            "jump_residual_profile",
            "action_scale_pos",
            "action_scale_vel",
        )
    }
    print(
        f"[FWD-MEAS] restored control contract={control_contract}",
        flush=True,
    )
    if FIXED_DESIGN is not None:
        actual_morph = env.env_morphology[0].detach().cpu().numpy()
        actual_motor = (
            env.env_motor_group_scale[0].detach().cpu().numpy()
        )
        if not np.allclose(actual_morph, FIXED_DESIGN[:2]) or not np.allclose(
            actual_motor, FIXED_DESIGN[2:]
        ):
            raise RuntimeError("fixed six-dimensional jump design was not applied")
        print(
            f"[FWD-MEAS] 固定设计 thigh={actual_morph[0]:.4f} shank={actual_morph[1]:.4f} "
            f"abad/hip/knee/wheel={actual_motor.tolist()} prefix={DESIGN_PREFIX}",
            flush=True,
        )
    grid_ids = None
    if GRID_POINTS:
        grid_commands = np.repeat(np.asarray(GRID_POINTS, dtype=np.float32), GRID_REPLICAS, axis=0)
        env.jump_command_override = torch.as_tensor(grid_commands, device=env.device)
        grid_ids = torch.arange(len(GRID_POINTS), device=env.device).repeat_interleave(GRID_REPLICAS)
        print(
            f"[FWD-MEAS] 固定二维网格 {len(GRID_HEIGHTS)}x{len(GRID_DISTANCES)}="
            f"{len(GRID_POINTS)}点, 每点{GRID_REPLICAS}个环境, seed={SEED}, "
            f"nominal={int(NOMINAL_EVAL)}",
            flush=True,
        )
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    obs, obs_history, commands, _ = env.get_observations()
    target_by_env = env._base_height_target_tensor()
    target = float(target_by_env.mean().item())
    nenv = env.num_envs
    takeoffs = torch.zeros(nenv, device=env.device)
    jump_counts = torch.zeros(nenv, dtype=torch.long, device=env.device)
    prev_state = torch.zeros(nenv, dtype=torch.long, device=env.device)
    prev_cmd = torch.zeros(nenv, device=env.device)
    trigger_counts = torch.zeros(nenv, dtype=torch.long, device=env.device)
    timeout_counts = torch.zeros(
        nenv, dtype=torch.long, device=env.device
    )
    reset_terminal_counts = torch.zeros(
        nenv, dtype=torch.long, device=env.device
    )
    cycle_open = torch.zeros(nenv, dtype=torch.bool, device=env.device)
    jump_returns = torch.zeros(nenv, device=env.device)
    jump_active_steps = torch.zeros(nenv, dtype=torch.long, device=env.device)
    jump_sat_sums = torch.zeros(nenv, device=env.device)
    jump_peak_torque_ratios = torch.zeros(nenv, device=env.device)
    jump_energy_j = torch.zeros(nenv, device=env.device)
    pending_row_index = torch.full(
        (nenv,), -1, dtype=torch.long, device=env.device
    )
    pending_recovery_steps = torch.zeros(
        nenv, dtype=torch.long, device=env.device
    )
    recovery_window_steps = max(
        int(np.ceil(PHYSICAL_RECOVERY_S / float(env.dt))), 1
    )
    effective_torque_limits = (
        env.torque_limits.view(1, -1) * env.motor_torque_limit_scale
    ).clamp_min(1.0e-6)
    jumps = []
    terminal_returns = []
    timeout_returns = []
    reset_returns = []
    idle_heights = []
    idle_rolls = []
    idle_lat_ratios = []
    idle_leg_diffs = []
    idle_leg_left = []
    idle_leg_right = []
    idle_foot_x_diffs = []
    idle_foot_z_diffs = []
    idle_foot_mid_ys = []
    idle_abad_mirror_errs = []
    idle_abad_left = []
    idle_abad_right = []
    idle_abad_abs_mean = []
    idle_hip_left = []
    idle_hip_right = []
    idle_knee_left = []
    idle_knee_right = []
    timeouts = 0

    for i in range(STEPS):
        with torch.no_grad():
            estimate = encoder(obs_history, obs)
            actions = policy(torch.cat((estimate, obs, commands), dim=-1).detach())
        obs, rewards, dones, _, obs_history, commands, _ = env.step(actions.detach())
        cmd = env.cmd_jump[:, 0]
        state = env.jump_state
        roll_abs, lat_ratio, _ = env._jump_lateral_metrics()
        symmetry = env._jump_symmetry_errors()
        start = (prev_cmd < 0.5) & (cmd > 0.5)
        trigger_counts += start.long()
        cycle_open |= start
        takeoffs = torch.where(start, torch.zeros_like(takeoffs), takeoffs)
        jump_returns = torch.where(start, torch.zeros_like(jump_returns), jump_returns)
        jump_active_steps = torch.where(start, torch.zeros_like(jump_active_steps), jump_active_steps)
        jump_sat_sums = torch.where(start, torch.zeros_like(jump_sat_sums), jump_sat_sums)
        jump_peak_torque_ratios = torch.where(
            start, torch.zeros_like(jump_peak_torque_ratios), jump_peak_torque_ratios
        )
        jump_energy_j = torch.where(start, torch.zeros_like(jump_energy_j), jump_energy_j)
        entered_flight = (prev_state != 2) & (state == 2)
        takeoffs = takeoffs + entered_flight.float()
        timeout_fire = env.jump_timeout_fire & cycle_open
        timeouts += int(timeout_fire.sum().item())
        timeout_counts += timeout_fire.long()
        land_fire = env.jump_land_fire & cycle_open
        reset_terminal = (
            dones.bool()
            & cycle_open
            & (~timeout_fire)
            & (~land_fire)
        )
        reset_terminal_counts += reset_terminal.long()
        active = (cmd > 0.5) | (state != 0) | timeout_fire | reset_terminal
        torque_ratio = torch.abs(env.torques) / effective_torque_limits
        torque_sat = (torque_ratio >= 0.98).float().mean(dim=1)
        mechanical_power = torch.sum(torch.abs(env.torques * env.dof_vel), dim=1)
        jump_returns += torch.where(active, rewards, torch.zeros_like(rewards))
        jump_active_steps += active.long()
        jump_sat_sums += torch.where(active, torque_sat, torch.zeros_like(torque_sat))
        jump_peak_torque_ratios = torch.where(
            active,
            torch.maximum(jump_peak_torque_ratios, torque_ratio.max(dim=1).values),
            jump_peak_torque_ratios,
        )
        jump_energy_j += torch.where(
            active, mechanical_power * float(env.dt), torch.zeros_like(mechanical_power)
        )
        if timeout_fire.any():
            timeout_values = jump_returns[timeout_fire].detach().cpu().tolist()
            timeout_returns.extend(timeout_values)
            terminal_returns.extend(timeout_values)
        if reset_terminal.any():
            reset_values = jump_returns[reset_terminal].detach().cpu().tolist()
            reset_returns.extend(reset_values)
            terminal_returns.extend(reset_values)

        if land_fire.any():
            terminal_returns.extend(jump_returns[land_fire].detach().cpu().tolist())
            for idx in land_fire.nonzero(as_tuple=False).flatten().tolist():
                active_steps = max(int(jump_active_steps[idx].item()), 1)
                row = {
                    "env_id": idx,
                    "grid_id": int(grid_ids[idx].item()) if grid_ids is not None else -1,
                    "repeat": int(jump_counts[idx].item()),
                    "design_thigh": float(env.env_morphology[idx, 0].item()) if FIXED_DESIGN is not None else 1.0,
                    "design_shank": float(env.env_morphology[idx, 1].item()) if FIXED_DESIGN is not None else 1.0,
                    "design_motor": float(env.env_motor_scale[idx].item()) if FIXED_DESIGN is not None else 1.0,
                    "design_abad": float(env.env_motor_group_scale[idx, 0].item()) if FIXED_DESIGN is not None else 1.0,
                    "design_hip": float(env.env_motor_group_scale[idx, 1].item()) if FIXED_DESIGN is not None else 1.0,
                    "design_knee": float(env.env_motor_group_scale[idx, 2].item()) if FIXED_DESIGN is not None else 1.0,
                    "design_wheel": float(env.env_motor_group_scale[idx, 3].item()) if FIXED_DESIGN is not None else 1.0,
                    "height": env.jump_peak_height[idx].item(),
                    "hstar": env.jump_height_cmd[idx, 0].item(),
                    "disp": env.jump_land_disp[idx].item(),
                    "dstar": env.jump_dist_cmd[idx, 0].item(),
                    "takeoffs": int(takeoffs[idx].item()),
                    "jump_return": jump_returns[idx].item(),
                    "jump_steps": active_steps,
                    "jump_reward_per_step": jump_returns[idx].item() / active_steps,
                    "torque_sat_fraction": jump_sat_sums[idx].item() / active_steps,
                    "peak_torque_ratio": jump_peak_torque_ratios[idx].item(),
                    "mechanical_energy_j": jump_energy_j[idx].item(),
                    "takeoff_leg_gain": env.jump_takeoff_leg_gain[idx].item(),
                    "takeoff_height_gain": env.jump_takeoff_height_gain[idx].item(),
                    "takeoff_leg_vel": env.jump_takeoff_leg_vel[idx].item(),
                    "takeoff_crouch_time": env.jump_takeoff_crouch_time[idx].item(),
                    "takeoff_quality": env.jump_takeoff_quality[idx].item(),
                    "bad_takeoff": float(env.jump_takeoff_quality[idx].item() < float(getattr(env.cfg.rewards, "jump_takeoff_quality_min", 0.35))),
                    "takeoff_roll": env.jump_takeoff_roll[idx].item(),
                    "takeoff_lat_ratio": env.jump_takeoff_lat_ratio[idx].item(),
                    "takeoff_angle": env.jump_takeoff_angle[idx].item(),
                    "takeoff_fwd_vel": env.jump_takeoff_fwd_vel[idx].item(),
                    "takeoff_lat_vel": env.jump_takeoff_lat_vel[idx].item(),
                    "takeoff_vz": env.jump_takeoff_vz[idx].item(),
                    "takeoff_base_vz": env.jump_takeoff_base_vz[idx].item(),
                    "takeoff_leg_diff": env.jump_takeoff_leg_diff[idx].item(),
                    "takeoff_leg_left": env.jump_takeoff_leg_left[idx].item(),
                    "takeoff_leg_right": env.jump_takeoff_leg_right[idx].item(),
                    "takeoff_foot_x_diff": env.jump_takeoff_foot_x_diff[idx].item(),
                    "takeoff_foot_z_diff": env.jump_takeoff_foot_z_diff[idx].item(),
                    "takeoff_abad_mirror_err": env.jump_takeoff_abad_mirror_err[idx].item(),
                    "takeoff_abad_left": env.jump_takeoff_abad_left[idx].item(),
                    "takeoff_abad_right": env.jump_takeoff_abad_right[idx].item(),
                    "takeoff_hip_left": env.jump_takeoff_hip_left[idx].item(),
                    "takeoff_hip_right": env.jump_takeoff_hip_right[idx].item(),
                    "takeoff_knee_left": env.jump_takeoff_knee_left[idx].item(),
                    "takeoff_knee_right": env.jump_takeoff_knee_right[idx].item(),
                    "takeoff_yaw_rate": env.jump_takeoff_yaw_rate[idx].item(),
                    "takeoff_leg_len": env.jump_takeoff_leg_len[idx].item(),
                    "min_flight_leg_len": env.jump_min_flight_leg_len[idx].item(),
                    "preland_leg_len": env.jump_preland_leg_len[idx].item(),
                    "land_leg_len": env.jump_land_leg_len[idx].item(),
                    "max_roll": env.jump_max_roll[idx].item(),
                    "max_lat_ratio": env.jump_max_lat_ratio[idx].item(),
                    "max_yaw_rate": env.jump_max_yaw_rate[idx].item(),
                    "max_lat_vel": env.jump_max_lat_vel[idx].item(),
                    "land_lateral_disp": env.jump_land_lateral_disp[idx].item(),
                    "recovery_success": 0,
                    "recovery_reset": 0,
                    "recovery_timeout": 0,
                    "recovery_unresolved": 0,
                    "recovery_steps": 0,
                    "recovery_speed": float("nan"),
                    "recovery_roll": float("nan"),
                    "recovery_lat_ratio": float("nan"),
                    "recovery_height_error": float("nan"),
                }
                jumps.append(row)
                pending_row_index[idx] = len(jumps) - 1
                pending_recovery_steps[idx] = 0
                jump_counts[idx] += 1
        cycle_open &= ~(timeout_fire | reset_terminal | land_fire)

        pending = pending_row_index >= 0
        pending_recovery_steps += pending.long()
        recovery_reset = pending & dones.bool()
        recovery_done = (
            pending
            & (pending_recovery_steps >= recovery_window_steps)
            & (~dones.bool())
        )
        if recovery_reset.any():
            for idx in recovery_reset.nonzero(as_tuple=False).flatten().tolist():
                row_idx = int(pending_row_index[idx].item())
                jumps[row_idx]["recovery_reset"] = 1
                jumps[row_idx]["recovery_steps"] = int(
                    pending_recovery_steps[idx].item()
                )
            pending_row_index[recovery_reset] = -1
            pending_recovery_steps[recovery_reset] = 0
        if recovery_done.any():
            on_ground = (
                env.contact_forces[:, env.feet_indices, 2] > 1.0
            ).any(dim=1)
            base_speed = torch.norm(env.base_lin_vel, dim=1)
            recovery_height_target = env._base_height_target_tensor()
            height_error = torch.abs(
                env.base_height - recovery_height_target
            )
            stable = (
                on_ground
                & (base_speed < PHYSICAL_RECOVERY_MAX_SPEED)
                & (height_error < PHYSICAL_RECOVERY_MAX_HEIGHT_ERROR)
                & (roll_abs < PHYSICAL_MAX_ROLL_RAD)
                & (lat_ratio < PHYSICAL_MAX_LAT_RATIO)
            )
            for idx in recovery_done.nonzero(as_tuple=False).flatten().tolist():
                row_idx = int(pending_row_index[idx].item())
                is_stable = bool(stable[idx].item())
                jumps[row_idx]["recovery_success"] = int(is_stable)
                jumps[row_idx]["recovery_timeout"] = int(not is_stable)
                jumps[row_idx]["recovery_steps"] = int(
                    pending_recovery_steps[idx].item()
                )
                jumps[row_idx]["recovery_speed"] = float(
                    base_speed[idx].item()
                )
                jumps[row_idx]["recovery_roll"] = float(
                    roll_abs[idx].item()
                )
                jumps[row_idx]["recovery_lat_ratio"] = float(
                    lat_ratio[idx].item()
                )
                jumps[row_idx]["recovery_height_error"] = float(
                    height_error[idx].item()
                )
            pending_row_index[recovery_done] = -1
            pending_recovery_steps[recovery_done] = 0

        idle = (cmd < 0.5) & (state == 0)
        if idle.any() and i > 100:
            idle_heights.append(env.base_height[idle].mean().item())
            idle_rolls.append(roll_abs[idle].mean().item())
            idle_lat_ratios.append(lat_ratio[idle].mean().item())
            idle_leg_diffs.append(symmetry[0][idle].mean().item())
            idle_leg_lengths = env._jump_leg_lengths()[idle]
            idle_leg_left.append(idle_leg_lengths[:, 0].mean().item())
            idle_leg_right.append(idle_leg_lengths[:, 1].mean().item())
            idle_foot_x_diffs.append(symmetry[1][idle].mean().item())
            idle_foot_z_diffs.append(symmetry[2][idle].mean().item())
            idle_foot_mid_ys.append(symmetry[3][idle].mean().item())
            idle_abad_mirror_errs.append(symmetry[4][idle].mean().item())
            idle_abad_left.append(env.dof_pos[idle, 0].mean().item())
            idle_abad_right.append(env.dof_pos[idle, 4].mean().item())
            idle_abad_abs_mean.append(torch.abs(env.dof_pos[idle][:, [0, 4]]).mean().item())
            idle_hip_left.append(env.dof_pos[idle, 1].mean().item())
            idle_hip_right.append(env.dof_pos[idle, 5].mean().item())
            idle_knee_left.append(env.dof_pos[idle, 2].mean().item())
            idle_knee_right.append(env.dof_pos[idle, 6].mean().item())
        prev_cmd = cmd.clone()
        prev_state = state.clone()

    unresolved_recovery = pending_row_index >= 0
    if unresolved_recovery.any():
        for idx in unresolved_recovery.nonzero(as_tuple=False).flatten().tolist():
            row_idx = int(pending_row_index[idx].item())
            jumps[row_idx]["recovery_unresolved"] = 1
            jumps[row_idx]["recovery_steps"] = int(
                pending_recovery_steps[idx].item()
            )

    total_triggers = int(trigger_counts.sum().item())
    completion_rate = 100.0 * len(jumps) / max(total_triggers, 1)
    terminal_rate = 100.0 * len(terminal_returns) / max(total_triggers, 1)
    unresolved_count = int(cycle_open.sum().item())
    physical_success_rows = [
        row for row in jumps if is_physical_success(row)
    ]
    recovery_unresolved_count = sum(
        int(row["recovery_unresolved"]) for row in jumps
    )
    physical_resolved_count = max(
        len(terminal_returns) - recovery_unresolved_count,
        0,
    )
    reachable_area = physical_grid_summary(
        jumps,
        trigger_counts,
        timeout_counts,
        reset_terminal_counts,
        grid_ids,
    )
    print(
        f"[FWD-MEAS] 前跳 {RUN} ckpt{CKPT} envs={nenv} steps={STEPS} "
        f"触发数={total_triggers} 完成跳跃数={len(jumps)} 完成率={completion_rate:.1f}% "
        f"timeout未起跳数={timeouts} 周期中reset数={len(reset_returns)} "
        f"未结算数={unresolved_count} 终止覆盖率={terminal_rate:.1f}%",
        flush=True,
    )
    if reachable_area is not None:
        print(
            f"[FWD-MEAS] ★物理可靠可达面积 A_jump="
            f"{reachable_area['metric_value_m2']:.6f}m^2 "
            f"可靠格点={reachable_area['reliable_grid_count']}/"
            f"{reachable_area['total_grid_count']} "
            f"物理成功率={100*len(physical_success_rows)/max(physical_resolved_count,1):.1f}% "
            f"结算覆盖率={100*physical_resolved_count/max(total_triggers,1):.1f}%",
            flush=True,
        )
    if terminal_returns:
        print(
            f"[FWD-MEAS] ★每触发终止回报(着陆+未起跳超时+周期中reset) mean/std="
            f"{np.mean(terminal_returns):.3f}/{np.std(terminal_returns):.3f}",
            flush=True,
        )
    if len(jumps):
        def arr(name):
            return np.array([j[name] for j in jumps], dtype=float)

        disp = arr("disp")
        dstar = arr("dstar")
        tk = arr("takeoffs").astype(int)
        jump_return = arr("jump_return")
        jump_reward_per_step = arr("jump_reward_per_step")
        torque_sat_fraction = arr("torque_sat_fraction")
        peak_torque_ratio = arr("peak_torque_ratio")
        mechanical_energy_j = arr("mechanical_energy_j")
        print(
            f"[FWD-MEAS] ★跳跃能力回报 trigger→touchdown mean/std={jump_return.mean():.3f}/"
            f"{jump_return.std():.3f}, per-step={jump_reward_per_step.mean():.4f}, "
            f"torque_sat={100*torque_sat_fraction.mean():.2f}%, "
            f"peak_torque_ratio p95={np.percentile(peak_torque_ratio,95):.2f}, "
            f"|tau*qdot|积分 mean={mechanical_energy_j.mean():.1f}J",
            flush=True,
        )
        print(
            f"[FWD-MEAS] ★单跳率(恰1次起跳)={100*np.mean(tk==1):.0f}%  "
            f"(0次={100*np.mean(tk==0):.0f}% 2+次={100*np.mean(tk>=2):.0f}%)",
            flush=True,
        )
        real = tk >= 1
        if real.sum() > 0:
            height, hstar = arr("height")[real], arr("hstar")[real]
            d, ds = disp[real], dstar[real]
            print(
                f"[FWD-MEAS] 真跳 n={real.sum()}: 双轮最低净空峰值mean={height.mean():.3f} "
                f"目标h*mean={hstar.mean():.3f} 前向轮心跳距mean={d.mean():.3f} "
                f"目标d*mean={ds.mean():.3f} ★跟踪误差|disp-d*|mean={np.abs(d-ds).mean():.3f}",
                flush=True,
            )
            print_tracking("轮胎净空高度", hstar, height)
            print_tracking("轮心前向跳距", ds, d)
            for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45)]:
                mask = real & (dstar >= lo) & (dstar < hi)
                if mask.sum() > 0:
                    print(
                        f"[FWD-MEAS]   d*∈[{lo},{hi}) n={mask.sum()}: "
                        f"实际跳距mean={disp[mask].mean():.3f} std={disp[mask].std():.3f} "
                        f"(目标中值{(lo+hi)/2:.2f})",
                        flush=True,
                    )
            print(f"[FWD-MEAS] 负跳距(往后/异常)占比={100*np.mean(d<-0.02):.0f}%", flush=True)
            gain = arr("takeoff_leg_gain")[real]
            hgain = arr("takeoff_height_gain")[real]
            lvel = arr("takeoff_leg_vel")[real]
            ctime = arr("takeoff_crouch_time")[real]
            quality = arr("takeoff_quality")[real]
            bad_takeoff = arr("bad_takeoff")[real]
            roll = arr("takeoff_roll")[real]
            mroll = arr("max_roll")[real]
            lat = arr("takeoff_lat_ratio")[real]
            mlat = arr("max_lat_ratio")[real]
            takeoff_len = arr("takeoff_leg_len")[real]
            min_flight_len = arr("min_flight_leg_len")[real]
            preland_len = arr("preland_leg_len")[real]
            land_len = arr("land_leg_len")[real]
            tuck = takeoff_len - min_flight_len
            pre_ext = preland_len - min_flight_len
            land_comp = preland_len - land_len
            angle = arr("takeoff_angle")[real]
            vfwd = arr("takeoff_fwd_vel")[real]
            vlat = arr("takeoff_lat_vel")[real]
            vvz = arr("takeoff_vz")[real]
            base_vz = arr("takeoff_base_vz")[real]
            leg_diff = arr("takeoff_leg_diff")[real]
            leg_left = arr("takeoff_leg_left")[real]
            leg_right = arr("takeoff_leg_right")[real]
            foot_x_diff = arr("takeoff_foot_x_diff")[real]
            foot_z_diff = arr("takeoff_foot_z_diff")[real]
            abad_err = arr("takeoff_abad_mirror_err")[real]
            abad_left = arr("takeoff_abad_left")[real]
            abad_right = arr("takeoff_abad_right")[real]
            hip_left = arr("takeoff_hip_left")[real]
            hip_right = arr("takeoff_hip_right")[real]
            knee_left = arr("takeoff_knee_left")[real]
            knee_right = arr("takeoff_knee_right")[real]
            yaw_rate = arr("takeoff_yaw_rate")[real]
            max_yaw_rate = arr("max_yaw_rate")[real]
            max_lat_vel = arr("max_lat_vel")[real]
            lateral_disp = arr("land_lateral_disp")[real]
            gain_thr = float(getattr(env.cfg.rewards, "jump_takeoff_leg_gain_target", 0.08))
            hgain_thr = float(getattr(env.cfg.rewards, "jump_takeoff_height_gain_target", 0.08))
            lat_lim = float(getattr(env.cfg.rewards, "jump_lateral_ratio_limit", 0.45))
            print(
                f"[FWD-MEAS] ★真蹬腿: leg_gain mean={gain.mean():.3f}m "
                f"(>={gain_thr:.2f}m占比={100*np.mean(gain>=gain_thr):.0f}%)  "
                f"height_gain mean={hgain.mean():.3f}m "
                f"(>={hgain_thr:.2f}m占比={100*np.mean(hgain>=hgain_thr):.0f}%)  "
                f"leg_vel mean={lvel.mean():.2f}m/s  crouch_time={ctime.mean():.2f}s",
                flush=True,
            )
            print(
                f"[FWD-MEAS] ★起飞质量: quality mean={quality.mean():.3f} "
                f"bad_takeoff占比={100*np.mean(bad_takeoff>0.5):.0f}%",
                flush=True,
            )
            print(
                f"[FWD-MEAS] ★出射: angle mean={np.degrees(angle).mean():.1f}deg "
                f"vfwd={vfwd.mean():.2f}m/s wheel_vz={vvz.mean():.2f}m/s base_vz={base_vz.mean():.2f}m/s  "
                f"vlat={vlat.mean():+.2f}m/s(+左) |vlat|={np.abs(vlat).mean():.2f}m/s",
                flush=True,
            )
            print(
                f"[FWD-MEAS] ★左右镜像@起飞: leg_diff={leg_diff.mean():.3f}m  "
                f"leg_L/R={leg_left.mean():.3f}/{leg_right.mean():.3f}m "
                f"wheel_x_diff={foot_x_diff.mean():.3f}m wheel_z_diff={foot_z_diff.mean():.3f}m  "
                f"abad_L/R={np.degrees(abad_left).mean():+.1f}/{np.degrees(abad_right).mean():+.1f}deg "
                f"mean|abad|={np.degrees(0.5*(np.abs(abad_left)+np.abs(abad_right))).mean():.1f}deg "
                f"|abad_L+R|={np.degrees(abad_err).mean():.1f}deg",
                flush=True,
            )
            print(
                f"[FWD-MEAS] ★髋膝@起飞: hip_L/R={np.degrees(hip_left).mean():+.1f}/{np.degrees(hip_right).mean():+.1f}deg "
                f"knee_L/R={np.degrees(knee_left).mean():+.1f}/{np.degrees(knee_right).mean():+.1f}deg "
                f"|hip_L+R|={np.degrees(np.abs(hip_left + hip_right)).mean():.1f}deg "
                f"|knee_L+R|={np.degrees(np.abs(knee_left + knee_right)).mean():.1f}deg",
                flush=True,
            )
            print(
                f"[FWD-MEAS] ★横向轨迹: 落点lateral mean={lateral_disp.mean():+.3f}m(+左) "
                f"|lateral|mean/p95={np.abs(lateral_disp).mean():.3f}/{np.percentile(np.abs(lateral_disp),95):.3f}m  "
                f"向左>5cm占比={100*np.mean(lateral_disp>0.05):.0f}%  "
                f"takeoff_yaw_rate={yaw_rate.mean():+.2f}rad/s max|yaw_rate|p95={np.percentile(max_yaw_rate,95):.2f}rad/s "
                f"max|vlat|p95={np.percentile(max_lat_vel,95):.2f}m/s",
                flush=True,
            )
            print(
                f"[FWD-MEAS] ★横向稳定: takeoff_roll mean/p95={np.degrees(roll).mean():.1f}/"
                f"{np.degrees(np.percentile(roll,95)):.1f}deg  max_roll p95={np.degrees(np.percentile(mroll,95)):.1f}deg  "
                f"takeoff_lat mean/p95={lat.mean():.2f}/{np.percentile(lat,95):.2f}  "
                f"max_lat p95={np.percentile(mlat,95):.2f}  "
                f"lat<{lat_lim:.2f}占比={100*np.mean(mlat<lat_lim):.0f}%",
                flush=True,
            )
            print(
                f"[FWD-MEAS] ★空中相位: tuck(takeoff-min_flight)={tuck.mean():.3f}m  "
                f"preland_extend={pre_ext.mean():.3f}m  landing_compress={land_comp.mean():.3f}m",
                flush=True,
            )
        if CSV_PATH:
            csv_dir = os.path.dirname(os.path.abspath(CSV_PATH))
            os.makedirs(csv_dir, exist_ok=True)
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(jumps[0].keys()))
                writer.writeheader()
                writer.writerows(jumps)
            print(f"[FWD-MEAS] 逐跳数据已写入 {CSV_PATH}", flush=True)
    if len(idle_heights):
        heights = np.array(idle_heights)
        print(f"[FWD-MEAS] 命令外站立高度 mean={heights.mean():.3f} (站高{target:.3f})", flush=True)
    if len(idle_rolls):
        rolls = np.array(idle_rolls)
        lats = np.array(idle_lat_ratios)
        print(
            f"[FWD-MEAS] 命令外横向稳定 roll mean={np.degrees(rolls).mean():.1f}deg "
            f"lat_ratio mean={lats.mean():.2f}",
            flush=True,
        )
    if len(idle_leg_diffs):
        print(
            f"[FWD-MEAS] 命令外左右镜像 leg_diff={np.mean(idle_leg_diffs):.3f}m "
            f"leg_L/R={np.mean(idle_leg_left):.3f}/{np.mean(idle_leg_right):.3f}m "
            f"wheel_x_diff={np.mean(idle_foot_x_diffs):.3f}m "
            f"wheel_z_diff={np.mean(idle_foot_z_diffs):.3f}m "
            f"wheel_mid_y_offset={np.mean(idle_foot_mid_ys):.3f}m "
            f"abad_L/R={np.degrees(np.mean(idle_abad_left)):+.1f}/{np.degrees(np.mean(idle_abad_right)):+.1f}deg "
            f"mean|abad|={np.degrees(np.mean(idle_abad_abs_mean)):.1f}deg "
            f"|abad_L+R|={np.degrees(np.mean(idle_abad_mirror_errs)):.1f}deg",
            flush=True,
        )
        print(
            f"[FWD-MEAS] 命令外髋膝 hip_L/R={np.degrees(np.mean(idle_hip_left)):+.1f}/{np.degrees(np.mean(idle_hip_right)):+.1f}deg "
            f"knee_L/R={np.degrees(np.mean(idle_knee_left)):+.1f}/{np.degrees(np.mean(idle_knee_right)):+.1f}deg",
            flush=True,
        )
    if SUMMARY_PATH:
        summary_dir = os.path.dirname(os.path.abspath(SUMMARY_PATH))
        os.makedirs(summary_dir, exist_ok=True)
        completed = len(jumps)
        single = [row for row in jumps if int(row["takeoffs"]) == 1]

        def mean_or_nan(rows, name):
            return float(np.mean([row[name] for row in rows])) if rows else float("nan")

        def p95_abs_or_nan(rows, name):
            return float(np.percentile(np.abs([row[name] for row in rows]), 95)) if rows else float("nan")

        summary = {
            "protocol": (
                "physical jump evaluation; command-grid reliability uses true single takeoff, "
                "touchdown, lateral stability, and stable landing recovery"
            ),
            "run": RUN,
            "checkpoint": CKPT,
            "seed": SEED,
            "num_envs": nenv,
            "steps": STEPS,
            "nominal_eval": NOMINAL_EVAL,
            "flat_eval": FLAT_EVAL,
            "friction": EVAL_FRICTION if FLAT_EVAL else None,
            "design": {
                "thigh": FIXED_DESIGN[0] if FIXED_DESIGN is not None else 1.0,
                "shank": FIXED_DESIGN[1] if FIXED_DESIGN is not None else 1.0,
                "abad": FIXED_DESIGN[2] if FIXED_DESIGN is not None else 1.0,
                "hip": FIXED_DESIGN[3] if FIXED_DESIGN is not None else 1.0,
                "knee": FIXED_DESIGN[4] if FIXED_DESIGN is not None else 1.0,
                "wheel": FIXED_DESIGN[5] if FIXED_DESIGN is not None else 1.0,
            },
            "motor_mass_model": {
                "type": "baseline_anchored_power_law",
                "exponent": float(env.cfg.domain_rand.motor_mass_exponent),
                "nominal_masses_kg": [
                    float(value)
                    for value in env.cfg.domain_rand.motor_nominal_masses_kg
                ],
            },
            "base_height_target_m": target,
            "control_contract": control_contract,
            "command_grid": {
                "heights": GRID_HEIGHTS,
                "distances": GRID_DISTANCES,
                "replicas": GRID_REPLICAS if GRID_POINTS else None,
            },
            "trigger_count": total_triggers,
            "completed_count": completed,
            "timeout_count": timeouts,
            "reset_terminal_count": len(reset_returns),
            "terminal_count": len(terminal_returns),
            "terminal_rate": len(terminal_returns) / max(total_triggers, 1),
            "unresolved_count": unresolved_count,
            "completion_rate": completed / max(total_triggers, 1),
            "single_jump_success_count": len(single),
            "single_jump_success_rate_per_trigger": len(single) / max(total_triggers, 1),
            "physical_success_count": len(physical_success_rows),
            "physical_success_rate_per_trigger": (
                len(physical_success_rows) / max(total_triggers, 1)
            ),
            "physical_resolved_count": physical_resolved_count,
            "physical_resolution_rate": (
                physical_resolved_count / max(total_triggers, 1)
            ),
            "physical_success_rate_per_resolved": (
                len(physical_success_rows)
                / max(physical_resolved_count, 1)
            ),
            "stable_recovery_count": sum(
                int(row["recovery_success"]) for row in jumps
            ),
            "recovery_reset_count": sum(
                int(row["recovery_reset"]) for row in jumps
            ),
            "recovery_timeout_count": sum(
                int(row["recovery_timeout"]) for row in jumps
            ),
            "recovery_unresolved_count": recovery_unresolved_count,
            "jump_return_mean_all_terminal": (
                float(np.mean(terminal_returns)) if terminal_returns else float("nan")
            ),
            "jump_return_std_all_terminal": (
                float(np.std(terminal_returns)) if terminal_returns else float("nan")
            ),
            "jump_return_mean_timeout": (
                float(np.mean(timeout_returns)) if timeout_returns else float("nan")
            ),
            "jump_return_mean_reset_terminal": (
                float(np.mean(reset_returns)) if reset_returns else float("nan")
            ),
            "jump_return_mean": mean_or_nan(jumps, "jump_return"),
            "jump_return_std": float(np.std([row["jump_return"] for row in jumps])) if jumps else float("nan"),
            "jump_reward_per_step_mean": mean_or_nan(jumps, "jump_reward_per_step"),
            "height_mean_single_m": mean_or_nan(single, "height"),
            "distance_mean_single_m": mean_or_nan(single, "disp"),
            "height_mae_single_m": (
                float(np.mean([abs(row["height"] - row["hstar"]) for row in single]))
                if single else float("nan")
            ),
            "distance_mae_single_m": (
                float(np.mean([abs(row["disp"] - row["dstar"]) for row in single]))
                if single else float("nan")
            ),
            "takeoff_quality_mean_single": mean_or_nan(single, "takeoff_quality"),
            "lateral_abs_p95_single_m": p95_abs_or_nan(single, "land_lateral_disp"),
            "max_roll_p95_single_deg": (
                float(np.degrees(np.percentile([row["max_roll"] for row in single], 95)))
                if single else float("nan")
            ),
            "torque_sat_fraction_mean": mean_or_nan(jumps, "torque_sat_fraction"),
            "peak_torque_ratio_p95": (
                float(np.percentile([row["peak_torque_ratio"] for row in jumps], 95))
                if jumps else float("nan")
            ),
            "mechanical_energy_mean_j": mean_or_nan(jumps, "mechanical_energy_j"),
            "reachable_area": reachable_area,
        }
        with open(SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        print(f"[FWD-MEAS] 汇总已写入 {SUMMARY_PATH}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
