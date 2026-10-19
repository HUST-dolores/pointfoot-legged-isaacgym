#!/usr/bin/env python3
"""Frozen June 2--30 kg policies, zero commands, paired free-box placement trials."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from legged_gym.scripts.eccentric_payload_protocol import (
    DT, POLICIES, MASSES, PLACEMENTS, SEEDS, make_scene, batch_id,
    array_digest, sha256_file, write_json,
)
from legged_gym.scripts.payload_step_experiment import (
    configure as walking_configure, environment_type, timestamp,
)


ROOT = Path(__file__).resolve().parents[2]


def prepare(output):
    import numpy as np
    folder = output / "scenarios"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for seed in SEEDS:
        scene = make_scene(seed)
        target = folder / f"seed_{seed}.npz"
        if target.exists():
            if array_digest(dict(np.load(target, allow_pickle=False))) != array_digest(scene):
                raise ValueError("existing paired scene differs")
        else:
            np.savez_compressed(target, **scene)
        manifest[str(seed)] = {"array_sha256": array_digest(scene), "sha256": sha256_file(target)}
    write_json(folder / "manifest.json", {"prepared": timestamp(), "seeds": manifest,
        "num_envs": 30, "steps": 2000, "box_x_overridden_only_at_spawn": PLACEMENTS})


def configure(cfg, saved, args):
    cfg = walking_configure(cfg, saved, SimpleNamespace(num_envs=args.num_envs, condition="free_box"))
    cfg.env.use_gait_stepping = cfg.env.use_gait_phase = False
    cfg.rewards.scales = SimpleNamespace(**saved["rewards"]["scales"])
    cfg.domain_rand.add_load_range = [args.mass, args.mass]
    return cfg


def run(args):
    import isaacgym  # noqa: F401
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
    torch.cuda.set_device(0)
    set_seed(args.seed)
    method, run_name, checkpoint = POLICIES[args.policy]
    source = ROOT / "logs/wheelfoot_flat/WF_TRON1A" / run_name
    saved = json.loads((source / "env_cfg.json").read_text())
    train = json.loads((source / "train_cfg.json").read_text())
    assert saved["domain_rand"]["add_load_range"] == [2, 30]
    assert train["seed"] == 1 and checkpoint == 11000
    batch = batch_id(args.policy, args.mass, args.placement, args.seed)
    folder = args.output / ("preflight" if args.steps < 2000 else "batches") / batch
    if (folder / "result.json").exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    scene_file = args.output / "scenarios" / f"seed_{args.seed}.npz"
    full_scene = dict(np.load(scene_file, allow_pickle=False))
    scene = {k: v[:, :args.num_envs] if k == "noise_uniform" else v[:args.num_envs]
             for k, v in full_scene.items()}
    scene = {k: v.copy() for k, v in scene.items()}
    scene["box_offset_body"][:, 0] = PLACEMENTS[args.placement]
    cfg = configure(BipedCfgWF(), saved, args)
    spec = importlib.util.spec_from_file_location(
        "legged_gym.envs.wheelfoot_flat._eccentric_archived", source / "wheelfoot_flat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Matched = environment_type(module.BipedWF)

    class StaticEnvironment(Matched):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.spawn_root = torch.full((self.num_envs, 13), float("nan"), device=self.device)
            self.spawn_box = torch.full_like(self.spawn_root, float("nan"))

        def _resample_commands(self, env_ids):
            self.commands[env_ids] = 0.0

        def _maybe_spawn_loads(self):
            ids = ((~self.has_load) & (self.episode_length_buf*self.dt >= .5)).nonzero(as_tuple=False).flatten()
            super()._maybe_spawn_loads()
            if ids.numel():
                self.spawn_root[ids] = self.root_states[self.actor_indices[ids]]
                self.spawn_box[ids] = self.root_states[self.load_indices[ids]]

    sim = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), sim)
    sim.use_gpu_pipeline = sim.physx.use_gpu = True
    sim.physx.num_threads = 4
    before = time.monotonic()
    started = timestamp()
    env = StaticEnvironment(scene, cfg=cfg, sim_params=sim, physics_engine=gymapi.SIM_PHYSX,
                            sim_device="cuda:0", headless=True)
    try:
        np.testing.assert_allclose(env.dt, DT, atol=1e-8)
        encoder = MLP_Encoder(**train["MLP_Encoder"]).to(env.device).eval()
        actor = ActorCritic(env.num_obs+10, env.num_critic_obs+10, env.num_actions,
                            **train["policy"]).to(env.device).eval()
        model_path = source / f"model_{checkpoint}.pt"
        checkpoint_data = torch.load(model_path, map_location=env.device, weights_only=False)
        actor.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
        encoder.load_state_dict(checkpoint_data["encoder_state_dict"], strict=True)
        actor.requires_grad_(False)
        encoder.requires_grad_(False)
        scales = torch.tensor([env.obs_scales.lin_vel]*3+[env.obs_scales.mass_scale]
                              + [env.obs_scales.com_scale]*3, device=env.device)
        records = {}

        def add(key, value):
            records.setdefault(key, []).append(value.detach().cpu().numpy().copy())

        with torch.inference_mode():
            env.reset()
            env.recording = True
            env.first_failure[:] = False
            initial = env.root_states[env.actor_indices].cpu().numpy().copy()
            for index in range(args.steps):
                obs, history, commands, critic = env.get_observations()
                raw = encoder.encode(history)
                fused = raw.clone()
                if train["algorithm"]["use_load_residual_estimation"]:
                    baseline = train["algorithm"]["load_residual_baseline_obs_start"]
                    fused[:, 3:7] += obs[:, baseline:baseline+4]
                action = actor.act_inference(torch.cat((fused, obs, commands), -1))
                if bool(commands.any()) or not bool(torch.isfinite(action).all()):
                    raise ValueError("nonzero command or nonfinite action")
                root, box = env.root_states[env.actor_indices], env.root_states[env.load_indices]
                add("time_s", env.episode_length_buf[0].float()*env.dt)
                add("root_state", root)
                add("box_state", box)
                add("box_relative_position", quat_rotate_inverse(root[:, 3:7], box[:, :3]-root[:, :3]))
                add("actions", action)
                add("estimate", fused/scales)
                add("mass_label", env.base_mass-env.base_mass0)
                add("com_label", env.base_com-env.base_com0)
                add("network_raw_mass_com", raw[:, 3:7]/scales[3:7])
                prior = env.last_load_estimates
                add("analytic_prior", torch.cat((prior["qs_mass_delta"][:, None], prior["qs_com_delta"]), -1))
                add("body_velocity", env.base_lin_vel)
                add("wheel_forces", env.contact_forces[:, env.feet_indices])
                add("base_contact_force", env.contact_forces[:, 0])
                add("box_contact_force", env.contact_forces[:, env.num_bodies])
                for key, value in (("has_load", env.has_load), ("on_body", env.load_on_body_last),
                    ("on_body_raw", env.load_on_body_raw_last), ("first_failure", env.first_failure),
                    ("failure_reason", env.failure_reason), ("fail_counter", env.fail_buf),
                    ("observations", obs), ("actor_commands", commands),
                    ("task_commands", env.commands[:, :3]), ("dof_position", env.dof_pos),
                    ("dof_velocity", env.dof_vel), ("torque", env.torques),
                    ("noise_injected", env.last_noise_injected[:, :20]),
                    ("noise_effective", env.last_noise_effective[:, :20])):
                    add(key, value)
                if index+1 < args.steps:
                    env.step(action)
                if (index+1) % 500 == 0:
                    print(f"[EVAL] {batch} {index+1}/{args.steps} wall={time.monotonic()-before:.1f}s", flush=True)
        arrays = {k: np.stack(v) for k, v in records.items()}
        for key, value in (("base_mass0", env.base_mass0), ("base_com0", env.base_com0),
            ("imu_offset_quaternion", env.random_imu_offset), ("noise_scale_vec", env.noise_scale_vec),
            ("first_failure_time_s", env.failure_time), ("terminal_root", env.terminal_root),
            ("spawn_root_state", env.spawn_root), ("spawn_box_state", env.spawn_box)):
            arrays[key] = value.cpu().numpy().copy()
        arrays["initial_root_after_zero_step"] = initial
        np.savez_compressed(folder / "trace.npz", **arrays)
        bodies, shapes, masses = [], [], []
        for i, (handle, actor_handle, box_handle) in enumerate(zip(env.envs, env.actor_handles, env.load_handles)):
            props = env.gym.get_actor_rigid_body_properties(handle, actor_handle)
            bodies.append({"env_id": i, "mass_kg": [p.mass for p in props],
                "inertia_diagonal": [[p.inertia.x.x,p.inertia.y.y,p.inertia.z.z] for p in props]})
            row = {"env_id": i}
            for name, actor_h in (("robot", actor_handle), ("box", box_handle)):
                row[name] = [{k: float(getattr(p,k)) for k in
                              ("friction", "rolling_friction", "torsion_friction", "restitution", "contact_offset", "rest_offset")}
                             for p in env.gym.get_actor_rigid_shape_properties(handle, actor_h)]
            shapes.append(row)
            masses.append(env.gym.get_actor_rigid_body_properties(handle, box_handle)[0].mass)
        np.testing.assert_allclose(masses, args.mass, atol=1e-5)
        write_json(folder / "actual_body_properties.json", bodies)
        write_json(folder / "actual_shape_properties.json", shapes)
        write_json(folder / "effective_env_cfg.json", class_to_dict(cfg))
        metadata = {"batch_id": batch, "policy_id": args.policy, "method": method,
            "run_name": run_name, "checkpoint": checkpoint, "training_seed": train["seed"],
            "evaluation_seed": args.seed, "mass_kg": args.mass, "placement": args.placement,
            "initial_box_offset_body_m": [PLACEMENTS[args.placement], 0., .10],
            "started": started, "finished": timestamp(), "elapsed_s": time.monotonic()-before,
            "num_envs": args.num_envs, "steps": args.steps, "dt_s": env.dt,
            "no_reset_after_first_failure": True, "no_scheduled_unload": True,
            "measured_payload_mass_kg": masses, "checkpoint_sha256": sha256_file(model_path),
            "archived_source_sha256": sha256_file(source / "wheelfoot_flat.py"),
            "paired_scene_file_sha256": sha256_file(scene_file), "used_scene_array_sha256": array_digest(scene),
            "trace_sha256": sha256_file(folder / "trace.npz"),
            "gpu_name": torch.cuda.get_device_name(0), "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "body_names": env.body_names, "dof_names": env.dof_names,
            "encoder_input_dim": train["MLP_Encoder"]["num_input_dim"], "obs_dim": env.num_obs,
            "residual_fusion": train["algorithm"]["use_load_residual_estimation"],
            "mass_supervision": "base_mass-base_mass0", "position_supervision": "base_com-base_com0"}
    finally:
        env.gym.destroy_sim(env.sim)
    metadata.update(cleanup_completed=True, exit_status="completed")
    write_json(folder / "result.json", metadata)
    print(f"[DONE] {batch} wall={metadata['elapsed_s']:.2f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--policy", choices=POLICIES)
    p.add_argument("--mass", type=int, choices=MASSES)
    p.add_argument("--placement", choices=PLACEMENTS)
    p.add_argument("--seed", type=int, choices=SEEDS, default=12345)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--num-envs", type=int, default=30)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.prepare:
        prepare(args.output)
    else:
        if args.policy is None or args.mass is None or args.placement is None:
            p.error("policy, mass and placement required")
        if not 1 <= args.steps <= 2000 or not 1 <= args.num_envs <= 30:
            p.error("invalid steps/envs")
        run(args)
