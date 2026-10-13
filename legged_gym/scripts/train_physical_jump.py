#!/usr/bin/env python3
"""Train or fine-tune a forward-jump controller under the physical protocol."""

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


MOTOR_GROUPS = ("abad", "hip", "knee", "wheel")


def _parse_range(name, default):
    raw = os.environ.get(name, "").strip()
    values = (
        [float(value) for value in raw.split(",") if value.strip()]
        if raw
        else list(default)
    )
    if len(values) != 2 or values[1] < values[0]:
        raise ValueError(f"{name} must be a nondecreasing min,max pair")
    return values


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
        "attached_payload",
    )
    for name in names:
        setattr(env_cfg.domain_rand, name, False)
    env_cfg.noise.add_noise = False


def _configure_design(env_cfg, design, seed, prefix, randomized):
    thigh, shank, abad, hip, knee, wheel = design
    if randomized:
        thigh_range = _parse_range("PHYS_JUMP_THIGH_RANGE", [0.8, 1.2])
        shank_range = _parse_range("PHYS_JUMP_SHANK_RANGE", [0.8, 1.2])
        motor_ranges = [
            _parse_range(
                f"PHYS_JUMP_{group.upper()}_RANGE",
                [0.6, 1.6],
            )
            for group in MOTOR_GROUPS
        ]
        num_buckets = int(
            os.environ.get("PHYS_JUMP_MORPH_BUCKETS", "64")
        )
    else:
        thigh_range = [thigh, thigh]
        shank_range = [shank, shank]
        motor_ranges = [[value, value] for value in (abad, hip, knee, wheel)]
        num_buckets = 1
    if num_buckets < 1:
        raise ValueError("PHYS_JUMP_MORPH_BUCKETS must be positive")

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
        "motor_nominal_masses_kg": [
            float(value)
            for value in env_cfg.domain_rand.motor_nominal_masses_kg
        ],
    }


def _control_contract(env_cfg):
    return {
        name: getattr(env_cfg.control, name, None)
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


def _verify_environment(env, design, design_spec, expected_contract):
    if env.env_motor_group_scale.shape != (env.num_envs, 4):
        raise RuntimeError("motor group scale tensor must have shape [num_envs,4]")
    if design_spec["mode"] == "fixed":
        if not torch.allclose(
            env.env_morphology,
            torch.as_tensor(design[:2], device=env.device).view(1, 2),
        ):
            raise RuntimeError("fixed jump morphology was not applied")
        expected_motor = torch.as_tensor(
            design[2:], device=env.device
        ).view(1, 4)
        if not torch.allclose(env.env_motor_group_scale, expected_motor):
            raise RuntimeError("fixed jump motor group scales were not applied")
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
                raise RuntimeError("nominal design anchors are not exactly 1.0")
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
    actual_contract = _control_contract(env.cfg)
    if actual_contract != expected_contract:
        raise RuntimeError(
            f"jump control contract changed: {actual_contract} "
            f"!= {expected_contract}"
        )
    body_props = env.gym.get_actor_rigid_body_properties(
        env.envs[0], env.actor_handles[0]
    )
    if abs(float(body_props[0].mass) - float(env.base_mass[0])) > 1.0e-4:
        raise RuntimeError("simulator jump base mass does not match config")
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
    print(
        f"[physical-jump-train] verified control={actual_contract} "
        f"base_mass_kg={float(body_props[0].mass):.6f} "
        f"hardware_context={hardware_context} "
        f"anchor_fraction={design_spec.get('anchor_fraction', 0.0):.3f}",
        flush=True,
    )


def main():
    task = "wheelfoot_jump_forward"
    initialization = os.environ.get(
        "PHYS_JUMP_INITIALIZATION", "pretrained"
    ).strip().lower()
    if initialization not in ("pretrained", "scratch"):
        raise ValueError(
            "PHYS_JUMP_INITIALIZATION must be pretrained or scratch"
        )
    resume_training = initialization == "pretrained"
    base_run = os.environ.get(
        "PHYS_JUMP_BASE_RUN",
        "Jul11_21-43-33_jumpfwd23_residual_s79",
    )
    base_ckpt = int(os.environ.get("PHYS_JUMP_BASE_CKPT", "4000"))
    updates = int(os.environ.get("PHYS_JUMP_UPDATES", "800"))
    num_envs = int(os.environ.get("PHYS_JUMP_NUM_ENVS", "4096"))
    seed = int(os.environ.get("PHYS_JUMP_SEED", "20260724"))
    tag = os.environ.get("PHYS_JUMP_TAG", "nominal_u800")
    randomized_design = (
        os.environ.get("PHYS_JUMP_RANDOMIZE_DESIGN", "0") == "1"
    )
    hardware_context = (
        os.environ.get("PHYS_JUMP_HARDWARE_CONTEXT", "0") == "1"
    )
    anchor_fraction = float(
        os.environ.get("PHYS_JUMP_ANCHOR_FRACTION", "0.0")
    )
    reference_policy_coef = float(
        os.environ.get("PHYS_JUMP_REFERENCE_POLICY_COEF", "0.0")
    )
    if updates < 1:
        raise ValueError("PHYS_JUMP_UPDATES must be positive")
    if seed == 1:
        raise ValueError("seed=1 is randomized by helpers.set_seed; use a fixed seed")
    if not 0.0 <= anchor_fraction <= 1.0:
        raise ValueError("PHYS_JUMP_ANCHOR_FRACTION must be in [0,1]")
    if anchor_fraction > 0.0 and not randomized_design:
        raise ValueError("design anchors require randomized-design training")
    if reference_policy_coef > 0.0 and not hardware_context:
        raise ValueError("reference-policy preservation requires hardware context")
    legacy_motor = float(os.environ.get("PHYS_JUMP_MOTOR", "1.0"))
    design = (
        float(os.environ.get("PHYS_JUMP_THIGH", "1.0")),
        float(os.environ.get("PHYS_JUMP_SHANK", "1.0")),
        float(os.environ.get("PHYS_JUMP_ABAD", legacy_motor)),
        float(os.environ.get("PHYS_JUMP_HIP", legacy_motor)),
        float(os.environ.get("PHYS_JUMP_KNEE", legacy_motor)),
        float(os.environ.get("PHYS_JUMP_WHEEL", legacy_motor)),
    )
    height_range = _parse_range(
        "PHYS_JUMP_HEIGHT_RANGE", [0.10, 0.16]
    )
    distance_range = _parse_range(
        "PHYS_JUMP_DISTANCE_RANGE", [0.05, 0.14]
    )

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
    saved_control_contract = _control_contract(env_cfg)

    _disable_nuisance_randomization(env_cfg)
    design_spec = _configure_design(
        env_cfg,
        design,
        seed,
        f"physjump_{tag}_{os.getpid()}",
        randomized_design,
    )
    design_spec["anchor_fraction"] = anchor_fraction
    env_cfg.domain_rand.design_anchor_fraction = anchor_fraction
    env_cfg.env.use_hardware_context_in_policy = hardware_context
    env_cfg.seed = seed
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = float(
        os.environ.get("PHYS_JUMP_EPISODE_S", "20.0")
    )
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.static_friction = float(
        os.environ.get("PHYS_JUMP_FRICTION", "0.8")
    )
    env_cfg.terrain.dynamic_friction = env_cfg.terrain.static_friction
    env_cfg.terrain.restitution = 0.0
    env_cfg.commands.jump_height_cmd_range = height_range
    env_cfg.commands.jump_dist_cmd_range = distance_range
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.scenario.partition = False

    train_cfg.seed = seed
    train_cfg.runner.resume = resume_training
    train_cfg.runner.load_run = base_run
    train_cfg.runner.checkpoint = base_ckpt
    train_cfg.runner.max_iterations = updates
    train_cfg.runner.run_name = f"physjump_{tag}"
    train_cfg.runner.save_interval = int(
        os.environ.get(
            "PHYS_JUMP_SAVE_INTERVAL",
            str(min(100, updates)),
        )
    )
    if train_cfg.runner.save_interval < 1:
        raise ValueError("PHYS_JUMP_SAVE_INTERVAL must be positive")
    train_cfg.runner.resume_noise_std = float(
        os.environ.get("PHYS_JUMP_RESUME_NOISE_STD", "0.5")
    )
    train_cfg.algorithm.learning_rate = float(
        os.environ.get("PHYS_JUMP_LEARNING_RATE", "0.001")
    )
    train_cfg.algorithm.schedule = os.environ.get(
        "PHYS_JUMP_SCHEDULE", "adaptive"
    )
    train_cfg.algorithm.entropy_coef = float(
        os.environ.get("PHYS_JUMP_ENTROPY_COEF", "0.005")
    )
    train_cfg.algorithm.est_learning_rate = float(
        os.environ.get("PHYS_JUMP_EST_LEARNING_RATE", "0.001")
    )
    train_cfg.algorithm.reference_policy_coef = reference_policy_coef
    train_cfg.algorithm.reference_policy_context_dim = (
        6 if hardware_context else 0
    )
    train_cfg.algorithm.reference_policy_anchor_tol = float(
        os.environ.get("PHYS_JUMP_REFERENCE_ANCHOR_TOL", "1e-6")
    )

    started = time.time()
    env, _ = task_registry.make_env(
        name=task,
        args=args,
        env_cfg=env_cfg,
    )
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=task,
        args=args,
        train_cfg=train_cfg,
    )
    _verify_environment(
        env,
        design,
        design_spec,
        saved_control_contract,
    )
    task_registry.save_cfgs(name=task)

    manifest = {
        "protocol": (
            "jump_generalist_hardware_context_v2_2026-07-30"
            if hardware_context
            else "sixd_single_objective_jump_generalist_v1_2026-07-26"
        ),
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
        "base_height_target_m": float(
            env._base_height_target_tensor().mean().item()
        ),
        "height_command_range_m": height_range,
        "distance_command_range_m": distance_range,
        "friction": float(env_cfg.terrain.static_friction),
        "resume_noise_std": train_cfg.runner.resume_noise_std,
        "learning_rate": train_cfg.algorithm.learning_rate,
        "schedule": train_cfg.algorithm.schedule,
        "entropy_coef": train_cfg.algorithm.entropy_coef,
        "est_learning_rate": train_cfg.algorithm.est_learning_rate,
        "hardware_context": hardware_context,
        "hardware_context_dim": (
            train_cfg.algorithm.reference_policy_context_dim
        ),
        "reference_policy_coef": reference_policy_coef,
        "reference_policy_anchor_tol": (
            train_cfg.algorithm.reference_policy_anchor_tol
        ),
        "control_contract": saved_control_contract,
    }
    with open(
        os.path.join(runner.log_dir, "physical_jump_training_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    for source in (
        __file__,
        os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "legged_gym",
            "envs",
            "wheelfoot_flat",
            "wheelfoot_flat.py",
        ),
        os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "legged_gym",
            "envs",
            "wheelfoot_flat",
            "wheelfoot_flat_config.py",
        ),
    ):
        shutil.copy2(source, runner.log_dir)

    print(
        "[physical-jump-train] "
        f"tag={tag} initialization={initialization} "
        f"design_spec={design_spec} seed={seed} envs={num_envs} "
        f"base={base_run}/model_{base_ckpt}.pt updates={updates} "
        f"log_dir={runner.log_dir}",
        flush=True,
    )
    if os.environ.get("PHYS_JUMP_SAVE_INITIAL", "0") == "1":
        initial_iteration = int(runner.current_learning_iteration)
        runner.save(
            os.path.join(
                runner.log_dir,
                f"model_{initial_iteration}.pt",
            ),
            iteration=initial_iteration,
        )
        print(
            "[physical-jump-train] saved initialization checkpoint "
            f"model_{initial_iteration}.pt",
            flush=True,
        )
    init_random_ep_len = (
        os.environ.get("PHYS_JUMP_INIT_RANDOM_EP_LEN", "1") == "1"
    )
    runner.learn(
        num_learning_iterations=updates,
        init_at_random_ep_len=init_random_ep_len,
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
        os.path.join(runner.log_dir, "physical_jump_training_result.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    external_result = os.environ.get(
        "PHYS_JUMP_RESULT_JSON", ""
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
        "PHYSICAL_JUMP_TRAIN_RESULT "
        f"run={result['run_name']} checkpoint={final_iteration} "
        f"elapsed_s={elapsed_s:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
