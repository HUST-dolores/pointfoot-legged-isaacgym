#!/usr/bin/env python3
"""Fine-tune one locomotion policy on the four physical capability scenarios.

The environment is split deterministically into four equal scenario groups:
clearance, real slope, rigidly attached payload, and total speed. Each group
contains the same number of fixed difficulty levels. Configuration is supplied
through PHYS_TRAIN_* environment variables so simultaneous GPU jobs never edit
the shared source config.
"""

import json
import os
import shutil
import time

import isaacgym  # noqa: F401
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.saved_config import load_saved_run_configs
import torch


DEFAULT_CLEARANCE = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
DEFAULT_SLOPES = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
DEFAULT_PAYLOADS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
DEFAULT_SPEEDS = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6]
PHYSICAL_TASKS = ("all", "clearance", "slope", "payload", "speed")
MOTOR_GROUPS = ("abad", "hip", "knee", "wheel")
TASK_TO_SCENARIO = {
    "clearance": 0,
    "slope": 1,
    "payload": 2,
    "speed": 3,
}


def _parse_values(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _parse_range(name, default):
    values = _parse_values(name, default)
    if len(values) != 2 or values[0] <= 0.0 or values[1] < values[0]:
        raise ValueError(f"{name} must be a positive nondecreasing min,max pair")
    return values


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


def _disable_nuisance_randomization(env_cfg):
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
    )
    for name in names:
        setattr(env_cfg.domain_rand, name, False)
    env_cfg.noise.add_noise = False


def _configure_design(env_cfg, design, seed, prefix, randomized):
    thigh, shank, abad, hip, knee, wheel = design
    if randomized:
        thigh_range = _parse_range("PHYS_TRAIN_THIGH_RANGE", [0.8, 1.2])
        shank_range = _parse_range("PHYS_TRAIN_SHANK_RANGE", [0.8, 1.2])
        motor_ranges = [
            _parse_range(
                f"PHYS_TRAIN_{group.upper()}_RANGE",
                [0.6, 1.6],
            )
            for group in MOTOR_GROUPS
        ]
        num_buckets = int(
            os.environ.get("PHYS_TRAIN_MORPH_BUCKETS", "64")
        )
    else:
        thigh_range = [thigh, thigh]
        shank_range = [shank, shank]
        motor_ranges = [[value, value] for value in (abad, hip, knee, wheel)]
        num_buckets = 1
    if num_buckets < 1:
        raise ValueError("PHYS_TRAIN_MORPH_BUCKETS must be positive")

    env_cfg.domain_rand.randomize_morphology = True
    env_cfg.domain_rand.morphology_regenerate = True
    env_cfg.domain_rand.morphology_num_buckets = num_buckets
    env_cfg.domain_rand.morphology_thigh_scale_range = thigh_range
    env_cfg.domain_rand.morphology_shank_scale_range = shank_range
    env_cfg.domain_rand.morphology_prefix = prefix
    env_cfg.domain_rand.morphology_seed = seed
    env_cfg.domain_rand.randomize_motor_design = True
    env_cfg.domain_rand.motor_group_torque_scale_ranges = motor_ranges
    env_cfg.domain_rand.motor_torque_scale_range = [
        min(bounds[0] for bounds in motor_ranges),
        max(bounds[1] for bounds in motor_ranges),
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
    return {
        "mode": "randomized" if randomized else "fixed",
        "thigh": thigh_range,
        "shank": shank_range,
        **{
            group: list(bounds)
            for group, bounds in zip(MOTOR_GROUPS, motor_ranges)
        },
        "morphology_buckets": num_buckets,
        "motor_mass_exponent": float(
            env_cfg.domain_rand.motor_mass_exponent
        ),
        "motor_power_envelope": bool(
            env_cfg.domain_rand.motor_power_envelope
        ),
        "motor_specific_power_w_per_kg": float(
            env_cfg.domain_rand.motor_specific_power_w_per_kg
        ),
        "motor_power_exponent": float(
            env_cfg.domain_rand.motor_power_exponent
        ),
        "motor_power_min_speed_rad_s": float(
            env_cfg.domain_rand.motor_power_min_speed_rad_s
        ),
        "motor_nominal_masses_kg": [
            float(value)
            for value in env_cfg.domain_rand.motor_nominal_masses_kg
        ],
    }


def _configure_physical_training(
    env_cfg,
    physical_task,
    clearance_levels,
    slope_levels,
    payload_levels,
    speed_levels,
    num_envs,
    slope_static,
    speed_slope_deg,
):
    lengths = {
        len(clearance_levels),
        len(slope_levels),
        len(payload_levels),
        len(speed_levels),
    }
    if len(lengths) != 1:
        raise ValueError(
            "all four PHYS_TRAIN level lists must have the same length"
        )
    levels_per_scenario = lengths.pop()
    if levels_per_scenario < 1:
        raise ValueError("physical training requires at least one level")
    total_rows = (
        4 * levels_per_scenario
        if physical_task == "all"
        else levels_per_scenario
    )
    if num_envs % total_rows != 0:
        raise ValueError(
            f"num_envs={num_envs} must be divisible by {total_rows} physical rows"
        )

    _disable_nuisance_randomization(env_cfg)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = float(
        os.environ.get("PHYS_TRAIN_EPISODE_S", "20.0")
    )
    env_cfg.init_state.dof_pos_randomization_range = [-0.05, 0.05]

    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    if physical_task == "all":
        terrain_slopes = (
            [0.0] * levels_per_scenario
            + list(slope_levels)
            + [0.0] * levels_per_scenario
            + [speed_slope_deg] * levels_per_scenario
        )
        terrain_payloads = (
            [0.0] * levels_per_scenario
            + [0.0] * levels_per_scenario
            + list(payload_levels)
            + [0.0] * levels_per_scenario
        )
    else:
        if physical_task == "slope":
            terrain_slopes = list(slope_levels)
        elif physical_task == "speed":
            terrain_slopes = [speed_slope_deg] * levels_per_scenario
        else:
            terrain_slopes = [0.0] * levels_per_scenario
        terrain_payloads = (
            list(payload_levels)
            if physical_task == "payload"
            else [0.0] * levels_per_scenario
        )
    env_cfg.terrain.evaluation_slope_degrees = terrain_slopes
    env_cfg.terrain.evaluation_spawn_xy_jitter = 0.0
    env_cfg.terrain.evaluation_reset_xy_jitter = 0.15
    env_cfg.terrain.num_rows = total_rows
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = total_rows - 1
    env_cfg.terrain.horizontal_scale = float(
        os.environ.get(
            "PHYS_TRAIN_SLOPE_HORIZONTAL_SCALE",
            env_cfg.terrain.horizontal_scale,
        )
    )
    env_cfg.terrain.vertical_scale = float(
        os.environ.get(
            "PHYS_TRAIN_SLOPE_VERTICAL_SCALE",
            env_cfg.terrain.vertical_scale,
        )
    )
    env_cfg.terrain.terrain_length = float(
        os.environ.get("PHYS_TRAIN_TERRAIN_LENGTH", "12.0")
    )
    env_cfg.terrain.terrain_width = float(
        os.environ.get("PHYS_TRAIN_TERRAIN_WIDTH", "6.0")
    )
    env_cfg.terrain.border_size = float(
        os.environ.get("PHYS_TRAIN_TERRAIN_BORDER", "5.0")
    )
    env_cfg.terrain.static_friction = float(
        os.environ.get("PHYS_TRAIN_FRICTION", "0.8")
    )
    env_cfg.terrain.dynamic_friction = env_cfg.terrain.static_friction
    env_cfg.terrain.restitution = 0.0

    env_cfg.domain_rand.attached_payload = True
    env_cfg.domain_rand.attached_payload_mass_range = [
        min(payload_levels),
        max(payload_levels),
    ]
    env_cfg.domain_rand.attached_payload_eval_masses = terrain_payloads
    env_cfg.domain_rand.attached_payload_offset_range_x = [0.0, 0.0]
    env_cfg.domain_rand.attached_payload_offset_range_y = [0.0, 0.0]
    env_cfg.domain_rand.attached_payload_offset_range_z = [0.10, 0.10]

    env_cfg.scenario.partition = True
    env_cfg.scenario.curriculum = False
    env_cfg.scenario.physical_partition = True
    env_cfg.scenario.physical_task = physical_task
    env_cfg.scenario.physical_levels_per_scenario = levels_per_scenario
    env_cfg.scenario.physical_clearance_targets = list(clearance_levels)
    env_cfg.scenario.physical_slope_degrees = list(slope_levels)
    env_cfg.scenario.physical_payload_masses = list(payload_levels)
    env_cfg.scenario.physical_speed_levels = list(speed_levels)
    env_cfg.scenario.physical_clearance_roll = 0.30
    env_cfg.scenario.physical_clearance_step = 0.40
    env_cfg.scenario.physical_motion_roll = (
        0.0 if physical_task == "slope" and slope_static else 0.10
    )
    env_cfg.scenario.physical_motion_step = (
        0.0 if physical_task == "slope" and slope_static else 0.40
    )
    speed_step_component = float(
        os.environ.get("PHYS_TRAIN_SPEED_STEP_COMPONENT", "0.40")
    )
    if speed_step_component < 0.0:
        raise ValueError(
            "PHYS_TRAIN_SPEED_STEP_COMPONENT must be non-negative"
        )
    env_cfg.scenario.physical_speed_step_cap = speed_step_component

    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 5.0
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.cmd_step_range = [0.0, 0.0]
    env_cfg.commands.cmd_step_zero_prob = 0.0
    env_cfg.rewards.scales.tracking_total_vel = float(
        os.environ.get("PHYS_TRAIN_TOTAL_VEL_REWARD", "4.0")
    )
    env_cfg.rewards.total_vel_tracking_sigma = float(
        os.environ.get("PHYS_TRAIN_TOTAL_VEL_SIGMA", "0.05")
    )
    env_cfg.rewards.scales.total_vel_error = float(
        os.environ.get("PHYS_TRAIN_TOTAL_VEL_ERROR_REWARD", "-1.0")
    )
    if physical_task == "slope" and slope_static:
        env_cfg.rewards.scales.tracking_lin_vel = float(
            os.environ.get(
                "PHYS_TRAIN_STATIC_XY_VEL_REWARD", "8.0"
            )
        )
        env_cfg.rewards.scales.tracking_roll_vel = float(
            os.environ.get(
                "PHYS_TRAIN_STATIC_ROLL_VEL_REWARD", "6.0"
            )
        )
        env_cfg.rewards.scales.tracking_step_vel = float(
            os.environ.get(
                "PHYS_TRAIN_STATIC_STEP_VEL_REWARD", "6.0"
            )
        )


def _verify_environment(
    env,
    expected_rows,
    design,
    design_spec,
    physical_task,
):
    expected_level = torch.div(
        torch.arange(env.num_envs, device=env.device) * expected_rows,
        env.num_envs,
        rounding_mode="floor",
    )
    if not (env.terrain_levels == expected_level).all():
        raise RuntimeError("unexpected physical terrain level assignment")
    if physical_task == "all":
        levels_per_scenario = expected_rows // 4
        expected_scenario = env.terrain_levels // levels_per_scenario
    else:
        levels_per_scenario = expected_rows
        expected_scenario = torch.full_like(
            env.terrain_levels,
            TASK_TO_SCENARIO[physical_task],
        )
    if not (env.scenario_id == expected_scenario).all():
        raise RuntimeError("physical scenario IDs do not match terrain rows")
    expected_payload = torch.as_tensor(
        env.cfg.domain_rand.attached_payload_eval_masses,
        device=env.device,
        dtype=env.attached_payload_mass.dtype,
    )[env.terrain_levels]
    if not torch.allclose(env.attached_payload_mass, expected_payload):
        raise RuntimeError("attached payload masses do not match terrain rows")
    if env.env_motor_group_scale.shape != (env.num_envs, 4):
        raise RuntimeError("motor group scale tensor must have shape [num_envs,4]")
    if design_spec["mode"] == "fixed":
        if not torch.allclose(
            env.env_morphology,
            torch.as_tensor(design[:2], device=env.device).view(1, 2),
        ):
            raise RuntimeError("fixed morphology was not applied")
        expected_motor = torch.as_tensor(
            design[2:], device=env.device
        ).view(1, 4)
        if not torch.allclose(env.env_motor_group_scale, expected_motor):
            raise RuntimeError("fixed motor group scales were not applied")
    else:
        design_values = torch.cat(
            (env.env_morphology, env.env_motor_group_scale),
            dim=1,
        )
        range_values = [
            design_spec["thigh"],
            design_spec["shank"],
            *[design_spec[group] for group in MOTOR_GROUPS],
        ]
        for column, bounds in enumerate(range_values):
            values = design_values[:, column]
            if float(values.min()) < float(bounds[0]) - 1.0e-6 or float(
                values.max()
            ) > float(bounds[1]) + 1.0e-6:
                raise RuntimeError(
                    f"randomized design column {column} escaped {bounds}"
                )
            if bounds[1] > bounds[0] and float(values.std()) <= 1.0e-4:
                raise RuntimeError(
                    f"randomized design column {column} has no variation"
                )
        anchor_fraction = float(design_spec.get("anchor_fraction", 0.0))
        if anchor_fraction > 0.0:
            anchor_mask = env.design_anchor_mask
            expected_count = int(round(anchor_fraction * env.num_envs))
            if int(anchor_mask.sum().item()) != expected_count:
                raise RuntimeError("nominal design-anchor count is incorrect")
            anchored_designs = design_values[anchor_mask]
            if not torch.allclose(
                anchored_designs,
                torch.ones_like(anchored_designs),
            ):
                raise RuntimeError(
                    "nominal design anchors are not exactly 1.0"
                )
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
    hardware_context = bool(
        getattr(env.cfg.env, "use_hardware_context_in_policy", False)
    )
    expected_commands = int(env.cfg.commands.num_commands) + (
        6 if hardware_context else 0
    )
    if env.num_commands != expected_commands:
        raise RuntimeError("policy hardware-context command width is incorrect")
    if env._get_policy_commands().shape != (env.num_envs, expected_commands):
        raise RuntimeError("policy command tensor has an incorrect shape")

    expected_roll = torch.zeros(env.num_envs, device=env.device)
    expected_step = torch.zeros_like(expected_roll)
    clearance = expected_scenario == 0
    slope_or_load = (expected_scenario == 1) | (expected_scenario == 2)
    speed = expected_scenario == 3
    expected_roll[clearance] = float(
        env.cfg.scenario.physical_clearance_roll
    )
    expected_step[clearance] = float(
        env.cfg.scenario.physical_clearance_step
    )
    expected_roll[slope_or_load] = float(
        env.cfg.scenario.physical_motion_roll
    )
    expected_step[slope_or_load] = float(
        env.cfg.scenario.physical_motion_step
    )
    speed_values = torch.as_tensor(
        env.cfg.scenario.physical_speed_levels,
        device=env.device,
        dtype=expected_roll.dtype,
    )
    speed_target = speed_values[env.physical_level_id[speed]]
    speed_step = torch.minimum(
        speed_target,
        torch.full_like(
            speed_target,
            float(env.cfg.scenario.physical_speed_step_cap),
        ),
    )
    expected_step[speed] = speed_step
    expected_roll[speed] = speed_target - speed_step
    if not torch.allclose(env.commands[:, 0], expected_roll):
        raise RuntimeError("roll commands do not match physical scenario rows")
    if not torch.allclose(env.cmd_step[:, 0], expected_step):
        raise RuntimeError("step commands do not match physical scenario rows")

    row_records = []
    for row in range(expected_rows):
        env_id = row * env.num_envs // expected_rows
        body_props = env.gym.get_actor_rigid_body_properties(
            env.envs[env_id], env.actor_handles[env_id]
        )
        actual_mass = float(body_props[0].mass)
        expected_mass = float(env.base_mass[env_id])
        actual_com = torch.tensor(
            [
                body_props[0].com.x,
                body_props[0].com.y,
                body_props[0].com.z,
            ],
            device=env.device,
        )
        expected_com = env.base_com[env_id]
        if abs(actual_mass - expected_mass) > 1.0e-4:
            raise RuntimeError(
                f"row {row} simulator base mass {actual_mass} "
                f"!= expected {expected_mass}"
            )
        if not torch.allclose(
            actual_com, expected_com, atol=1.0e-4, rtol=0.0
        ):
            raise RuntimeError(
                f"row {row} simulator base COM {actual_com.tolist()} "
                f"!= expected {expected_com.tolist()}"
            )
        row_records.append(
            (
                row,
                int(expected_scenario[env_id]),
                float(
                    env.cfg.terrain.evaluation_slope_degrees[row]
                ),
                float(
                    env.cfg.domain_rand.attached_payload_eval_masses[row]
                ),
                float(expected_roll[env_id]),
                float(expected_step[env_id]),
                actual_mass,
            )
        )
    print(
        "[physical-train] verified rows "
        "(row,scenario,slope_deg,payload_kg,roll,step,base_mass_kg):",
        flush=True,
    )
    for record in row_records:
        print(f"[physical-train] row={record}", flush=True)


def main():
    task = "wheelfoot_flat"
    initialization = os.environ.get(
        "PHYS_TRAIN_INITIALIZATION", "pretrained"
    ).strip().lower()
    if initialization not in ("pretrained", "scratch"):
        raise ValueError(
            "PHYS_TRAIN_INITIALIZATION must be pretrained or scratch"
        )
    resume_training = initialization == "pretrained"
    physical_task = os.environ.get(
        "PHYS_TRAIN_TASK", "all"
    ).strip().lower()
    if physical_task not in PHYSICAL_TASKS:
        raise ValueError(
            "PHYS_TRAIN_TASK must be one of "
            f"{', '.join(PHYSICAL_TASKS)}, got {physical_task!r}"
        )
    base_run = os.environ.get("PHYS_TRAIN_BASE_RUN", "Jul05_00-02-15_")
    base_ckpt = int(os.environ.get("PHYS_TRAIN_BASE_CKPT", "27000"))
    updates = int(os.environ.get("PHYS_TRAIN_UPDATES", "100"))
    num_envs = int(os.environ.get("PHYS_TRAIN_NUM_ENVS", "4096"))
    seed = int(os.environ.get("PHYS_TRAIN_SEED", "20260724"))
    tag = os.environ.get("PHYS_TRAIN_TAG", "nominal_u100")
    randomized_design = (
        os.environ.get("PHYS_TRAIN_RANDOMIZE_DESIGN", "0") == "1"
    )
    hardware_context = _parse_bool("PHYS_TRAIN_HARDWARE_CONTEXT", False)
    anchor_fraction = float(
        os.environ.get("PHYS_TRAIN_ANCHOR_FRACTION", "0.0")
    )
    reference_policy_coef = float(
        os.environ.get("PHYS_TRAIN_REFERENCE_POLICY_COEF", "0.0")
    )
    slope_static = (
        os.environ.get("PHYS_TRAIN_SLOPE_STATIC", "0") == "1"
    )
    speed_slope_deg = float(
        os.environ.get("PHYS_TRAIN_SPEED_SLOPE_DEG", "0.0")
    )
    if speed_slope_deg < 0.0 or speed_slope_deg >= 45.0:
        raise ValueError(
            "PHYS_TRAIN_SPEED_SLOPE_DEG must be in [0, 45)"
        )
    if updates < 1:
        raise ValueError("PHYS_TRAIN_UPDATES must be positive")
    if seed == 1:
        raise ValueError("seed=1 is randomized by helpers.set_seed; use a fixed seed")
    if not 0.0 <= anchor_fraction <= 1.0:
        raise ValueError("PHYS_TRAIN_ANCHOR_FRACTION must be in [0,1]")
    if anchor_fraction > 0.0 and not randomized_design:
        raise ValueError("design anchors require randomized-design training")
    if reference_policy_coef > 0.0 and not hardware_context:
        raise ValueError(
            "reference-policy preservation requires hardware context"
        )
    legacy_motor = float(os.environ.get("PHYS_TRAIN_MOTOR", "1.0"))
    design = (
        float(os.environ.get("PHYS_TRAIN_THIGH", "1.0")),
        float(os.environ.get("PHYS_TRAIN_SHANK", "1.0")),
        float(os.environ.get("PHYS_TRAIN_ABAD", legacy_motor)),
        float(os.environ.get("PHYS_TRAIN_HIP", legacy_motor)),
        float(os.environ.get("PHYS_TRAIN_KNEE", legacy_motor)),
        float(os.environ.get("PHYS_TRAIN_WHEEL", legacy_motor)),
    )
    clearance_levels = _parse_values(
        "PHYS_TRAIN_CLEARANCE_LEVELS", DEFAULT_CLEARANCE
    )
    slope_levels = _parse_values("PHYS_TRAIN_SLOPE_LEVELS", DEFAULT_SLOPES)
    payload_levels = _parse_values(
        "PHYS_TRAIN_PAYLOAD_LEVELS", DEFAULT_PAYLOADS
    )
    speed_levels = _parse_values("PHYS_TRAIN_SPEED_LEVELS", DEFAULT_SPEEDS)

    args = get_args()
    args.task = task
    args.headless = True
    args.num_envs = num_envs
    args.max_iterations = updates
    args.seed = seed
    args.resume = resume_training
    args.load_run = base_run
    args.checkpoint = base_ckpt

    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        task,
        train_cfg.runner.experiment_name,
    )
    base_dir = load_saved_run_configs(
        env_cfg, train_cfg, log_root, base_run
    )
    env_cfg.sim.physx.max_gpu_contact_pairs = int(
        os.environ.get(
            "PHYS_MAX_GPU_CONTACT_PAIRS",
            str(env_cfg.sim.physx.max_gpu_contact_pairs),
        )
    )
    _configure_physical_training(
        env_cfg,
        physical_task,
        clearance_levels,
        slope_levels,
        payload_levels,
        speed_levels,
        num_envs,
        slope_static,
        speed_slope_deg,
    )
    design_spec = _configure_design(
        env_cfg,
        design,
        seed,
        f"phystrain_{tag}_{os.getpid()}",
        randomized_design,
    )
    design_spec["anchor_fraction"] = anchor_fraction
    env_cfg.domain_rand.design_anchor_fraction = anchor_fraction
    env_cfg.env.use_hardware_context_in_policy = hardware_context
    env_cfg.seed = seed

    train_cfg.seed = seed
    train_cfg.runner.resume = resume_training
    train_cfg.runner.load_run = base_run
    train_cfg.runner.checkpoint = base_ckpt
    train_cfg.runner.max_iterations = updates
    train_cfg.runner.run_name = f"physadapt_{tag}"
    train_cfg.runner.save_interval = int(
        os.environ.get(
            "PHYS_TRAIN_SAVE_INTERVAL",
            str(min(100, updates)),
        )
    )
    if train_cfg.runner.save_interval < 1:
        raise ValueError("PHYS_TRAIN_SAVE_INTERVAL must be positive")
    train_cfg.algorithm.learning_rate = float(
        os.environ.get("PHYS_TRAIN_LEARNING_RATE", "0.0001")
    )
    train_cfg.algorithm.schedule = os.environ.get(
        "PHYS_TRAIN_SCHEDULE", "fixed"
    )
    train_cfg.algorithm.entropy_coef = float(
        os.environ.get("PHYS_TRAIN_ENTROPY_COEF", "0.001")
    )
    train_cfg.algorithm.est_learning_rate = float(
        os.environ.get("PHYS_TRAIN_EST_LEARNING_RATE", "0.0")
    )
    train_cfg.algorithm.reference_policy_coef = reference_policy_coef
    train_cfg.algorithm.reference_policy_context_dim = (
        6 if hardware_context else 0
    )
    train_cfg.algorithm.reference_policy_anchor_tol = float(
        os.environ.get("PHYS_TRAIN_REFERENCE_ANCHOR_TOL", "1e-6")
    )

    started = time.time()
    env, _ = task_registry.make_env(
        name=task,
        args=args,
        env_cfg=env_cfg,
    )
    # Scenario IDs are initialized by the runner's first reset.
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=task,
        args=args,
        train_cfg=train_cfg,
    )
    expected_rows = (
        4 * len(slope_levels)
        if physical_task == "all"
        else len(slope_levels)
    )
    _verify_environment(
        env,
        expected_rows,
        design,
        design_spec,
        physical_task,
    )
    task_registry.save_cfgs(name=task)

    manifest = {
        "protocol": (
            "sixd_physical_generalist_hardware_context_v2_2026-07-30"
            if hardware_context
            else "sixd_single_objective_generalist_v1_2026-07-26"
        ),
        "physical_task": physical_task,
        "initialization": initialization,
        "base_run": base_run,
        "base_checkpoint": base_ckpt,
        "base_directory": base_dir,
        "updates": updates,
        "num_envs": num_envs,
        "seed": seed,
        "tag": tag,
        "design": {
            "thigh": design[0],
            "shank": design[1],
            "abad": design[2],
            "hip": design[3],
            "knee": design[4],
            "wheel": design[5],
        },
        "design_spec": design_spec,
        "hardware_context": hardware_context,
        "hardware_context_dim": (
            train_cfg.algorithm.reference_policy_context_dim
        ),
        "reference_policy_coef": reference_policy_coef,
        "reference_policy_anchor_tol": (
            train_cfg.algorithm.reference_policy_anchor_tol
        ),
        "slope_static_hold": slope_static,
        "speed_slope_deg": speed_slope_deg,
        "speed_step_component_mps": float(
            env_cfg.scenario.physical_speed_step_cap
        ),
        "terrain_discretization": {
            "horizontal_scale_m": float(
                env_cfg.terrain.horizontal_scale
            ),
            "vertical_scale_m": float(env_cfg.terrain.vertical_scale),
            "length_m": float(env_cfg.terrain.terrain_length),
            "width_m": float(env_cfg.terrain.terrain_width),
        },
        "base_height_target_m": float(
            env._base_height_target_tensor().mean().item()
        ),
        "clearance_levels_m": clearance_levels,
        "slope_levels_deg": slope_levels,
        "payload_levels_kg": payload_levels,
        "speed_levels_mps": speed_levels,
        "learning_rate": train_cfg.algorithm.learning_rate,
        "schedule": train_cfg.algorithm.schedule,
        "entropy_coef": train_cfg.algorithm.entropy_coef,
        "est_learning_rate": train_cfg.algorithm.est_learning_rate,
        "save_interval": train_cfg.runner.save_interval,
        "tracking_total_vel_reward": (
            env_cfg.rewards.scales.tracking_total_vel
        ),
        "total_vel_tracking_sigma": (
            env_cfg.rewards.total_vel_tracking_sigma
        ),
        "total_vel_error_reward": (
            env_cfg.rewards.scales.total_vel_error
        ),
        "static_xy_vel_reward": float(
            getattr(env_cfg.rewards.scales, "tracking_lin_vel", 0.0)
        ),
        "static_roll_vel_reward": float(
            env_cfg.rewards.scales.tracking_roll_vel
        ),
        "static_step_vel_reward": float(
            env_cfg.rewards.scales.tracking_step_vel
        ),
    }
    with open(
        os.path.join(runner.log_dir, "physical_training_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    for source in (
        __file__,
        os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "legged_gym",
            "utils",
            "terrain.py",
        ),
        os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "legged_gym",
            "envs",
            "base",
            "base_task.py",
        ),
    ):
        shutil.copy2(source, runner.log_dir)

    print(
        "[physical-train] "
        f"tag={tag} initialization={initialization} "
        f"design_spec={design_spec} seed={seed} envs={num_envs} "
        f"base={base_run}/model_{base_ckpt}.pt updates={updates} "
        f"log_dir={runner.log_dir}",
        flush=True,
    )
    if _parse_bool("PHYS_TRAIN_SAVE_INITIAL", False):
        initial_iteration = int(runner.current_learning_iteration)
        runner.save(
            os.path.join(
                runner.log_dir,
                f"model_{initial_iteration}.pt",
            ),
            iteration=initial_iteration,
        )
        print(
            "[physical-train] saved initialization checkpoint "
            f"model_{initial_iteration}.pt",
            flush=True,
        )
    runner.learn(
        num_learning_iterations=updates,
        init_at_random_ep_len=True,
    )
    final_iteration = runner.current_learning_iteration
    elapsed_s = time.time() - started
    result = {
        **manifest,
        "run_name": os.path.basename(runner.log_dir),
        "log_dir": runner.log_dir,
        "final_iteration": final_iteration,
        "checkpoint": os.path.join(
            runner.log_dir, f"model_{final_iteration}.pt"
        ),
        "elapsed_s": elapsed_s,
    }
    with open(
        os.path.join(runner.log_dir, "physical_training_result.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    external_result = os.environ.get(
        "PHYS_TRAIN_RESULT_JSON", ""
    ).strip()
    if external_result:
        os.makedirs(
            os.path.dirname(os.path.abspath(external_result)),
            exist_ok=True,
        )
        with open(
            external_result, "w", encoding="utf-8"
        ) as handle:
            json.dump(result, handle, indent=2, ensure_ascii=True)
    print(
        "PHYSICAL_TRAIN_RESULT "
        f"run={result['run_name']} checkpoint={final_iteration} "
        f"elapsed_s={elapsed_s:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
