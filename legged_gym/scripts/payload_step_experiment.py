#!/usr/bin/env python3
"""Evaluate frozen July policies with paired scenes and read-only synchronous logging."""

import argparse
import copy
import csv
from datetime import datetime, timezone, timedelta
import importlib.util
import json
import os
from pathlib import Path
import time

from legged_gym.scripts.payload_step_protocol import (
    DT, MASS_KG, POLICIES, SEEDS, array_digest, make_scene, sha256_file, write_json,
)


ROOT = Path(__file__).resolve().parents[2]


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def overlay(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            if not hasattr(target, key):
                from types import SimpleNamespace
                setattr(target, key, SimpleNamespace())
            if isinstance(getattr(target, key), dict):
                setattr(target, key, copy.deepcopy(value))
            else:
                overlay(getattr(target, key), value)
        else:
            setattr(target, key, copy.deepcopy(value))


def prepare(output):
    import numpy as np
    directory = output / "scenarios"
    directory.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for seed in SEEDS:
        scene = make_scene(seed)
        target = directory / f"seed_{seed}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as old:
                if array_digest(dict(old)) != array_digest(scene):
                    raise ValueError(f"existing scene mismatch: {target}")
        else:
            np.savez_compressed(target, **scene)
        hashes[str(seed)] = {"array_sha256": array_digest(scene), "file_sha256": sha256_file(target)}
        for env in range(30):
            rows.append({"evaluation_seed": seed, "env_id": env,
                         "pair_id": f"{seed}:{env}",
                         **{f"box_offset_{a}_m": float(scene["box_offset_body"][env, i])
                            for i, a in enumerate("xyz")},
                         "gait_frequency_hz": float(scene["gait"][env, 0]),
                         "imu_pitch_rad": float(scene["imu_pitch_roll_rad"][env, 0]),
                         "imu_roll_rad": float(scene["imu_pitch_roll_rad"][env, 1])})
    with (directory / "paired_scenarios.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "manifest.json", {"prepared": timestamp(), "num_envs": 30,
                                             "steps": 2000, "dt_s": DT, "seeds": hashes})
    print(f"Prepared 3 scene/noise tables: {directory}", flush=True)


def configure(cfg, saved, args):
    overlay(cfg, saved)
    cfg.env.num_envs = args.num_envs
    cfg.env.episode_length_s = 60.0
    cfg.env.use_jump = cfg.env.use_jump_distance = False
    cfg.env.use_hardware_context_in_policy = False
    cfg.env.encoder_include_torques = False
    cfg.env.use_morphology_in_critic = cfg.env.use_motor_design_in_critic = False
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.curriculum = cfg.terrain.measure_heights = cfg.terrain.critic_measure_heights = False
    cfg.terrain.evaluation_spawn_xy_jitter = cfg.terrain.evaluation_reset_xy_jitter = 0.0
    cfg.commands.curriculum = cfg.commands.heading_command = False
    cfg.noise.add_noise = True
    cfg.noise.noise_level = 0.5
    disabled = (
        "randomize_friction", "randomize_restitution", "randomize_base_com", "randomize_base_mass",
        "push_robots", "push_curriculum", "randomize_Kp", "randomize_Kd", "randomize_motor_torque",
        "randomize_default_dof_pos", "randomize_action_delay", "randomize_morphology",
        "randomize_motor_design", "attached_payload", "load_drop_terminal", "load_debug",
        "ext_force_enable", "motor_power_envelope", "load_on_body_position_only",
    )
    for key in disabled:
        setattr(cfg.domain_rand, key, False)
    cfg.domain_rand.randomize_inertia = cfg.domain_rand.randomize_imu_offset = True
    cfg.domain_rand.add_random_load = args.condition == "free_box"
    cfg.domain_rand.per_env_load_mass = False
    cfg.domain_rand.add_load_range = [MASS_KG, MASS_KG]
    cfg.domain_rand.load_start_time_s = 0.5
    cfg.domain_rand.load_duration_s = cfg.domain_rand.load_interval_s = 1e6
    cfg.domain_rand.load_duration_range_s = cfg.domain_rand.load_interval_range_s = [1e6, 1e6]
    if hasattr(cfg, "scenario"):
        cfg.scenario.partition = False
    cfg.sim.use_gpu_pipeline = True
    cfg.sim.gravity = [0.0, 0.0, -9.81]
    cfg.sim.physx.use_gpu = True
    cfg.sim.physx.max_gpu_contact_pairs = max(2 ** 23, cfg.sim.physx.max_gpu_contact_pairs)
    return cfg


def environment_type(archived):
    import numpy as np
    import torch
    from isaacgym import gymtorch
    from isaacgym.torch_utils import quat_mul, quat_rotate, quat_rotate_inverse

    class MatchedPayloadEnvironment(archived):
        def __init__(self, scene, *args, **kwargs):
            self.scene_np = scene
            self.recording = False
            self.noise_index = 0
            super().__init__(*args, **kwargs)
            self.scene = {key: torch.as_tensor(value, device=self.device)
                          for key, value in scene.items()}
            pitch, roll = self.scene["imu_pitch_roll_rad"].unbind(-1)
            zeros = torch.zeros_like(pitch)
            qp = torch.stack((zeros, torch.sin(pitch / 2), zeros, torch.cos(pitch / 2)), -1)
            qr = torch.stack((torch.sin(roll / 2), zeros, zeros, torch.cos(roll / 2)), -1)
            self.random_imu_offset = quat_mul(qp, qr)
            self.first_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.failure_time = torch.full((self.num_envs,), float("nan"), device=self.device)
            self.failure_reason = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.terminal_root = torch.full((self.num_envs, 13), float("nan"), device=self.device)

        def _process_rigid_body_props(self, props, env_id):
            enabled = self.cfg.domain_rand.randomize_inertia
            self.cfg.domain_rand.randomize_inertia = False
            try:
                props = super()._process_rigid_body_props(props, env_id)
            finally:
                self.cfg.domain_rand.randomize_inertia = enabled
            for index, prop in enumerate(props):
                scale = float(self.scene_np["body_mass_inertia_scale"][env_id, index])
                prop.mass *= scale
                prop.inertia.x.x *= scale
                prop.inertia.y.y *= scale
                prop.inertia.z.z *= scale
            self.base_mass0[env_id] = props[0].mass
            return props

        def _reset_root_states(self, env_ids):
            self.root_states[self.actor_indices[env_ids]] = self.base_init_state
            self.root_states[self.actor_indices[env_ids], :3] += self.env_origins[env_ids]
            self.root_states[self.actor_indices[env_ids], 7:13] = self.scene["initial_root_velocity"][env_ids]
            ids = self.actor_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(ids), len(ids))

        def _reset_dofs(self, env_ids):
            self.dof_pos[env_ids] = self.default_dof_pos[env_ids] + self.scene["initial_joint_offset"][env_ids]
            self.dof_vel[env_ids] = 0.0
            ids = self.actor_indices[env_ids].to(torch.int32)
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(ids), len(ids))

        def _resample_commands(self, env_ids):
            self.commands[env_ids] = 0.0
            self.commands[env_ids, 0] = 0.5
            self.cmd_step[env_ids, 0] = 0.4

        def _resample_gaits(self, env_ids):
            self.gaits[env_ids] = self.scene["gait"][env_ids]

        def _maybe_spawn_loads(self):
            if not self.cfg.domain_rand.add_random_load:
                return
            times = self.episode_length_buf * self.dt
            ids = ((~self.has_load) & (times >= 0.5)).nonzero(as_tuple=False).flatten()
            if not ids.numel():
                return
            local = self.scene["box_offset_body"][ids]
            quat = self.root_states[self.actor_indices[ids], 3:7]
            pos = self.root_states[self.actor_indices[ids], :3] + quat_rotate(quat, local)
            orientation = torch.zeros((len(ids), 4), device=self.device)
            orientation[:, 3] = 1.0
            self.add_load(ids, pos, orientation)
            self.has_load[ids] = True
            self.loads_spawned_count[ids] += 1
            self.load_remove_time[ids] = 1e6
            self.next_load_spawn_time[ids] = 1e6
            self.load_contact_grace_counter[ids] = self.load_contact_grace_steps

        def reset_idx(self, env_ids):
            if not self.recording:
                return super().reset_idx(env_ids)
            new = env_ids[~self.first_failure[env_ids]]
            if new.numel():
                self.first_failure[new] = True
                self.failure_time[new] = self.episode_length_buf[new] * self.dt
                self.terminal_root[new] = self.root_states[self.actor_indices[new]]
                contact = torch.any(torch.norm(
                    self.contact_forces[new][:, self.termination_contact_indices], dim=-1) > 10.0, dim=1)
                attitude = self.projected_gravity[new, 2] > -0.1
                self.failure_reason[new] = contact.int() + 2 * attitude.int() + 4 * self.time_out_buf[new].int()
            # Failed slots continue physically, but are never reset or counted as new trials.

        def compute_observations(self):
            self.obs_buf, self.critic_obs_buf = self.compute_group_observations()
            self.last_noise_injected = torch.zeros_like(self.obs_buf)
            self.last_noise_injected[:, :20] = (
                self.scene["noise_uniform"][self.noise_index] * self.noise_scale_vec[:20])
            self.obs_buf += self.last_noise_injected
            self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], self.obs_buf), -1)
            # Preserve the saved inference ordering: IMU offset overwrites the first six noisy slots.
            randomized = quat_mul(self.random_imu_offset, self.base_quat)
            self.obs_buf[:, :3] = quat_rotate_inverse(
                randomized, self.root_states[self.actor_indices, 10:13]) * self.obs_scales.ang_vel
            self.obs_buf[:, 3:6] = quat_rotate_inverse(randomized, self.gravity_vec)
            self.last_noise_effective = self.last_noise_injected.clone()
            self.last_noise_effective[:, :6] = 0.0
            raw = self.obs_buf[:, :self.filtered_size]
            filtered = raw.clone()
            x = raw[:, self.filter_feature_indices]
            y = (self.butter_b0 * x + self.butter_b1 * self.butter_x1
                 + self.butter_b2 * self.butter_x2 - self.butter_a1 * self.butter_y1
                 - self.butter_a2 * self.butter_y2)
            self.butter_x2, self.butter_x1 = self.butter_x1, x
            self.butter_y2, self.butter_y1 = self.butter_y1, y
            filtered[:, self.filter_feature_indices] = y
            self.filtered_obs_buf = filtered
            cat = torch.cat((self.obs_buf, filtered), -1)
            self.encoder_obs_history = torch.cat((self.encoder_obs_history[:, cat.shape[1]:], cat), -1)
            self.noise_index += 1

    return MatchedPayloadEnvironment


def run(args):
    import isaacgym  # noqa: F401  # Isaac Gym must precede torch imports.
    from isaacgym import gymapi, gymutil
    from isaacgym.torch_utils import quat_rotate_inverse
    import numpy as np
    import torch
    from legged_gym.algorithm.actor_critic import ActorCritic
    from legged_gym.algorithm.mlp_encoder import MLP_Encoder
    from legged_gym.envs.wheelfoot_flat.wheelfoot_flat_config import BipedCfgWF
    from legged_gym.utils.helpers import class_to_dict, set_seed

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.gpu)
    method, repeat, run_name, checkpoint, training_seed = POLICIES[args.policy]
    run_dir = ROOT / "logs/wheelfoot_flat/WF_TRON1A" / run_name
    saved = json.loads((run_dir / "env_cfg.json").read_text())
    train = json.loads((run_dir / "train_cfg.json").read_text())
    set_seed(args.seed)
    batch = f"{args.policy}_eval{args.seed}_{args.condition}"
    directory = args.output / ("preflight" if args.steps < 2000 else "batches") / batch
    if args.minimal:
        directory = directory.with_name(directory.name + "_minimal")
    if (directory / "result.json").exists():
        raise FileExistsError(f"completed result exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    scene_path = args.output / "scenarios" / f"seed_{args.seed}.npz"
    with np.load(scene_path, allow_pickle=False) as loaded:
        scene = {key: value[:, :args.num_envs] if key == "noise_uniform" else value[:args.num_envs]
                 for key, value in loaded.items()}
    cfg = configure(BipedCfgWF(), saved, args)
    name = "legged_gym.envs.wheelfoot_flat._payload_step_archived"
    spec = importlib.util.spec_from_file_location(name, run_dir / "wheelfoot_flat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Env = environment_type(module.BipedWF)
    sim = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), sim)
    sim.use_gpu_pipeline = sim.physx.use_gpu = True
    sim.physx.num_threads = 4
    env = Env(scene, cfg=cfg, sim_params=sim, physics_engine=gymapi.SIM_PHYSX,
              sim_device=f"cuda:{args.gpu}", headless=True)
    try:
        if abs(env.dt - DT) > 1e-8:
            raise ValueError(f"unexpected control dt: {env.dt}")
        encoder = MLP_Encoder(**train["MLP_Encoder"]).to(env.device).eval()
        critic_dim = env.num_critic_obs + 3 + (7 if train["algorithm"]["critic_take_latent"] else 0)
        actor = ActorCritic(env.num_obs + 10, critic_dim, env.num_actions,
                            **train["policy"]).to(env.device).eval()
        model_path = run_dir / f"model_{checkpoint}.pt"
        checkpoint_data = torch.load(model_path, map_location=env.device, weights_only=False)
        actor.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
        encoder.load_state_dict(checkpoint_data["encoder_state_dict"], strict=True)
        actor.requires_grad_(False)
        encoder.requires_grad_(False)
        residual = train["algorithm"]["use_load_residual_estimation"]
        baseline_start = train["algorithm"].get("load_residual_baseline_obs_start", 36)
        physical_scales = torch.tensor([env.obs_scales.lin_vel] * 3 + [env.obs_scales.mass_scale]
                                        + [env.obs_scales.com_scale] * 3, device=env.device)
        initial_root_before_step = None
        with torch.inference_mode():
            env.reset()
            env.recording = True
            env.first_failure[:] = False
            initial_root_before_step = env.root_states[env.actor_indices].cpu().numpy().copy()
            records = {}
            def add(key, value):
                records.setdefault(key, []).append(value.detach().cpu().numpy().copy())
            for index in range(args.steps):
                obs, history, commands, critic = env.get_observations()
                raw = encoder.encode(history)
                fused = raw.clone()
                if residual:
                    fused[:, 3:7] += obs[:, baseline_start:baseline_start + 4]
                actions = actor.act_inference(torch.cat((fused, obs, commands), -1))
                if not (torch.allclose(commands[:, 0], torch.full_like(commands[:, 0],
                           0.5 * env.commands_scale[0])) and torch.allclose(obs[:, -3],
                           torch.full_like(obs[:, -3], 0.4))):
                    raise ValueError("actor command mismatch")
                if not bool(torch.isfinite(actions).all()):
                    raise ValueError("nonfinite policy action")
                add("time_s", env.episode_length_buf[0].float() * env.dt)
                add("root_state", env.root_states[env.actor_indices])
                add("actions", actions)
                add("estimate", fused / physical_scales)
                add("first_failure", env.first_failure)
                if not args.minimal:
                    root = env.root_states[env.actor_indices]
                    box = env.root_states[env.load_indices]
                    relative = quat_rotate_inverse(root[:, 3:7], box[:, :3] - root[:, :3])
                    prior = env.last_load_estimates
                    add("box_state", box)
                    add("box_relative_position", relative)
                    add("mass_label", env.base_mass - env.base_mass0)
                    add("com_label", env.base_com - env.base_com0)
                    add("network_raw_mass_com", raw[:, 3:7] / physical_scales[3:7])
                    add("analytic_prior", torch.cat((prior["qs_mass_delta"][:, None],
                                                     prior["qs_com_delta"]), -1))
                    add("body_velocity", env.base_lin_vel)
                    add("body_angular_velocity", env.base_ang_vel)
                    add("wheel_forces", env.contact_forces[:, env.feet_indices])
                    add("base_contact_force", env.contact_forces[:, 0])
                    add("box_contact_force", env.contact_forces[:, env.num_bodies])
                    add("has_load", env.has_load)
                    add("on_body", env.load_on_body_last)
                    add("on_body_raw", env.load_on_body_raw_last)
                    add("actor_commands", commands)
                    add("task_commands", torch.cat((env.commands[:, :3], env.cmd_step), -1))
                    add("observations", obs)
                    add("noise_injected", env.last_noise_injected[:, :20])
                    add("noise_effective", env.last_noise_effective[:, :20])
                    add("gait_phase", env.gait_indices)
                    add("desired_contact", env.desired_contact_states)
                    add("dof_position", env.dof_pos)
                    add("dof_velocity", env.dof_vel)
                    add("torque", env.torques)
                    add("failure_reason", env.failure_reason)
                    add("timeout", env.time_out_buf)
                    add("fail_counter", env.fail_buf)
                if index + 1 < args.steps:
                    env.step(actions)
                if (index + 1) % 500 == 0:
                    print(f"[EVAL] {batch} {index+1}/{args.steps} "
                          f"failed={int(env.first_failure.sum())} elapsed={time.monotonic()-start:.1f}s", flush=True)
        arrays = {key: np.stack(value) for key, value in records.items()}
        arrays["terminal_root"] = env.terminal_root.cpu().numpy().copy()
        arrays["first_failure_time_s"] = env.failure_time.cpu().numpy().copy()
        arrays["initial_root_after_zero_step"] = initial_root_before_step
        arrays["base_mass0"] = env.base_mass0.cpu().numpy().copy()
        arrays["base_com0"] = env.base_com0.cpu().numpy().copy()
        arrays["imu_offset_quaternion"] = env.random_imu_offset.cpu().numpy().copy()
        arrays["noise_scale_vec"] = env.noise_scale_vec.cpu().numpy().copy()
        np.savez_compressed(directory / "trace.npz", **arrays)
        body_props = []
        for env_id, (handle, actor_handle) in enumerate(zip(env.envs, env.actor_handles)):
            props = env.gym.get_actor_rigid_body_properties(handle, actor_handle)
            body_props.append({"env_id": env_id, "mass_kg": [p.mass for p in props],
                               "inertia_diagonal": [[p.inertia.x.x, p.inertia.y.y, p.inertia.z.z] for p in props]})
        measured_load_mass = [env.gym.get_actor_rigid_body_properties(handle, load)[0].mass
                              for handle, load in zip(env.envs, env.load_handles)]
        write_json(directory / "effective_env_cfg.json", class_to_dict(cfg))
        write_json(directory / "actual_body_properties.json", body_props)
        elapsed = time.monotonic() - start
        write_json(directory / "result.json", {
            "batch_id": batch, "policy_id": args.policy, "method": method, "training_repeat": repeat,
            "training_seed": training_seed, "evaluation_seed": args.seed, "condition": args.condition,
            "checkpoint": checkpoint, "checkpoint_sha256": sha256_file(model_path),
            "task_source_sha256": sha256_file(run_dir / "wheelfoot_flat.py"),
            "scene_array_sha256": array_digest(scene), "scene_file_sha256": sha256_file(scene_path),
            "started": args.started, "finished": timestamp(), "elapsed_s": elapsed,
            "gpu": args.gpu, "gpu_name": torch.cuda.get_device_name(args.gpu),
            "physical_gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "num_envs": args.num_envs, "steps": args.steps, "dt_s": env.dt,
            "duration_s": args.steps * env.dt, "payload_asset_mass_kg": MASS_KG,
            "measured_payload_mass_kg": measured_load_mass,
            "wheel_indices": env.feet_indices.cpu().tolist(), "body_names": env.body_names,
            "dof_names": env.dof_names, "robot_failed_count": int(env.first_failure.sum()),
            "mass_supervision": "base_mass-base_mass0", "com_supervision": "base_com-base_com0",
            "no_reset_after_first_failure": True, "first_state_after_zero_action_initialization_s": DT,
            "encoder_output": "analytic_prior_plus_residual" if residual else "direct_estimate",
            "observations_shape": list(arrays.get("observations", np.empty((0,))).shape),
            "effective_cfg_sha256": sha256_file(directory / "effective_env_cfg.json"),
            "trace_sha256": sha256_file(directory / "trace.npz"), "minimal_recorder": args.minimal,
            "exit_status": "completed",
        })
        print(f"[DONE] {batch} elapsed={elapsed:.2f}s output={directory}", flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--condition", choices=("empty", "free_box"))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=12345)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=30)
    parser.add_argument("--minimal", action="store_true")
    args = parser.parse_args()
    args.started = timestamp()
    args.output = args.output.resolve()
    if args.prepare:
        prepare(args.output)
    else:
        if args.policy is None or args.condition is None:
            parser.error("--policy and --condition are required for evaluation")
        if not 1 <= args.num_envs <= 30 or not 1 <= args.steps <= 2000:
            parser.error("num-envs must be 1..30 and steps must be 1..2000")
        run(args)


if __name__ == "__main__":
    main()
