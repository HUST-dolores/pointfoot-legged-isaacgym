#!/usr/bin/env python3
"""Matched fixed-policy stepping on explicit, disjoint triangle-mesh patches."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from legged_gym.scripts.payload_step_experiment import configure, environment_type, timestamp
from legged_gym.scripts.terrain_step_protocol import (
    DT, DX, X0, X1, Y0, Y1, POLICIES, TERRAINS, SEEDS, FRAME_TIMES,
    FRICTION, RESTITUTION, triangle_mesh, wheel_geometry,
    array_digest, sha256_file, write_json,
)


ROOT = Path(__file__).resolve().parents[2]


def terrain_environment(archived):
    import numpy as np
    import torch
    from isaacgym import gymapi
    from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

    matched = environment_type(archived)

    class TerrainEnvironment(matched):
        def __init__(self, scene, terrain, capture, *args, **kwargs):
            self.terrain_name, self.capture_enabled = terrain, capture
            self.capture_metadata = []
            super().__init__(scene, *args, **kwargs)
            self.camera = None
            self.patch_exited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if capture:
                properties = gymapi.CameraProperties()
                properties.width, properties.height = 960, 720
                properties.horizontal_fov = 50
                properties.use_collision_geometry = True
                properties.near_plane, properties.far_plane = .02, 40.
                self.camera = self.gym.create_camera_sensor(self.envs[0], properties)
                if self.camera < 0:
                    raise RuntimeError("offscreen camera creation failed")
                palette = {"base": (.35, .58, .78), "abad": (.78, .48, .56),
                           "hip": (.88, .69, .30), "knee": (.29, .68, .56),
                           "wheel": (.41, .37, .61)}
                for body_id, name in enumerate(self.body_names):
                    color = next((c for group, c in palette.items() if group in name.lower()), (.5, .6, .7))
                    self.gym.set_rigid_body_color(self.envs[0], self.actor_handles[0], body_id,
                                                 gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*color))
                self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(.85, .85, .85),
                                              gymapi.Vec3(.5, .5, .5), gymapi.Vec3(-.4, -.6, -1))

        def create_sim(self):
            self.up_axis_idx = 2
            self.graphics_device_id = self.sim_device_id if self.capture_enabled else -1
            self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                           self.physics_engine, self.sim_params)
            if self.sim is None:
                raise RuntimeError("simulation creation failed")
            self.terrain_heights = torch.as_tensor(self.scene_np["height_"+self.terrain_name], device=self.device)
            self.mesh_anchors = torch.as_tensor(self.scene_np["mesh_anchors_xy"], device=self.device)
            vertices, triangles = triangle_mesh(self.scene_np["height_"+self.terrain_name],
                                                 self.scene_np["mesh_anchors_xy"])
            self.mesh_digest = array_digest({"vertices": vertices, "triangles": triangles})
            parameters = gymapi.TriangleMeshParams()
            parameters.nb_vertices, parameters.nb_triangles = len(vertices), len(triangles)
            parameters.static_friction = parameters.dynamic_friction = FRICTION
            parameters.restitution = RESTITUTION
            self.gym.add_triangle_mesh(self.sim, vertices.ravel(), triangles.ravel(), parameters)
            self._create_envs()

        def _get_env_origins(self):
            self.custom_origins = False
            self.env_origins = torch.zeros((self.num_envs, 3), device=self.device)
            self.env_origins[:, :2] = self.mesh_anchors
            self.env_origins[:, 2] = self.terrain_heights[:, 50, 40]
            self.terrain_x_min, self.terrain_x_max = -1000., 1000.
            self.terrain_y_min, self.terrain_y_max = -1000., 1000.

        def _process_rigid_shape_props(self, props, env_id):
            props = super()._process_rigid_shape_props(props, env_id)
            for prop in props:
                prop.friction, prop.restitution = FRICTION, 0.
            return props

        def sample_ground(self, points):
            original_shape = points.shape[:-1]
            points = points.reshape(self.num_envs, -1, points.shape[-1])
            local = points[..., :2] - self.mesh_anchors[:, None]
            u = ((local[..., 0] - X0) / DX).clamp(0, self.terrain_heights.shape[1]-1)
            v = ((local[..., 1] - Y0) / DX).clamp(0, self.terrain_heights.shape[2]-1)
            i = u.floor().long().clamp(max=self.terrain_heights.shape[1]-2)
            j = v.floor().long().clamp(max=self.terrain_heights.shape[2]-2)
            u, v = u-i, v-j
            ids = torch.arange(self.num_envs, device=self.device)[:, None]
            h = self.terrain_heights
            z00, z10, z01, z11 = h[ids, i, j], h[ids, i+1, j], h[ids, i, j+1], h[ids, i+1, j+1]
            result = torch.where(u+v <= 1, z00+(z10-z00)*u+(z01-z00)*v,
                                 z11+(z01-z11)*(1-u)+(z10-z11)*(1-v))
            return result.reshape(original_shape)

        def _get_foot_heights(self):
            return self.sample_ground(self.foot_positions)

        def _get_heights(self, env_ids=None):
            height = self.sample_ground(self.root_states[self.actor_indices, :3])[:, None]
            return height if env_ids is None else height[env_ids]

        def _post_physics_step_callback(self):
            super()._post_physics_step_callback()
            self.measured_heights = self._get_heights()
            self.base_height = self.root_states[self.actor_indices, 2] - self.measured_heights[:, 0]

        def _resample_commands(self, env_ids):
            self.commands[env_ids] = 0.
            self.cmd_step[env_ids, 0] = .4

        def check_termination(self):
            super().check_termination()
            local = self.root_states[self.actor_indices, :2] - self.mesh_anchors
            outside = ((local[:, 0] < X0+.5) | (local[:, 0] > X1-.5)
                       | (local[:, 1] < Y0+.5) | (local[:, 1] > Y1-.5))
            if hasattr(self, "patch_exited"):
                self.patch_exited |= outside
            self.edge_reset_buf |= outside
            self.reset_buf |= outside

        def wheel_lowest(self, offsets, radii, widths):
            state = self.rigid_body_state[:, self.feet_indices]
            q = state[..., 3:7].reshape(-1, 4)
            center = state[..., :3] + quat_rotate(q, offsets[None].expand(self.num_envs, -1, -1).reshape(-1, 3)).reshape(self.num_envs, 2, 3)
            local_axis = torch.zeros_like(center)
            local_axis[..., 1] = 1
            axis = quat_rotate(q, local_axis.reshape(-1, 3)).reshape_as(center)
            downward = torch.zeros_like(center)
            downward[..., 2] = -1
            dot = -axis[..., 2]
            if self.cfg.asset.replace_cylinder_with_capsule:
                radial = downward
            else:
                radial = downward - axis * dot[..., None]
                radial /= torch.norm(radial, dim=-1, keepdim=True).clamp(min=1e-7)
            lowest = center + radii[None, :, None] * radial
            lowest += .5 * widths[None, :, None] * torch.sign(dot)[..., None] * axis
            ground = self.sample_ground(lowest)
            return lowest, ground, lowest[..., 2] - ground

        def capture(self, directory, time_s):
            from PIL import Image
            root = self.root_states[self.actor_indices[0], :3].detach().cpu().numpy()
            target = root + np.array([0., 0., -.35])
            camera_position = target + np.array([1.4, -1.8, 2.0])
            self.gym.set_camera_location(self.camera, self.envs[0], gymapi.Vec3(*camera_position), gymapi.Vec3(*target))
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            rgba = self.gym.get_camera_image(self.sim, self.envs[0], self.camera, gymapi.IMAGE_COLOR)
            rgba = np.asarray(rgba).reshape(720, 960, 4)
            if rgba[..., :3].std() < 4:
                raise RuntimeError("blank offscreen screenshot")
            name = f"t_{time_s:.2f}.png"
            Image.fromarray(rgba[..., :3]).save(directory / name)
            self.capture_metadata.append({"time_s": time_s, "file": name, "env_id": 0,
                "camera_position": camera_position.tolist(), "target": target.tolist(),
                "collision_geometry": True, "post_failure": bool(self.first_failure[0]),
                "rgb_std": float(rgba[..., :3].std())})

    return TerrainEnvironment


def run(args):
    import isaacgym  # noqa: F401
    from isaacgym import gymapi, gymutil
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
    name, run_name, checkpoint = POLICIES[args.policy]
    run_dir = ROOT / "logs/wheelfoot_flat/WF_TRON1A" / run_name
    saved = json.loads((run_dir / "env_cfg.json").read_text())
    train = json.loads((run_dir / "train_cfg.json").read_text())
    if saved["env"]["num_observations"] != 31 or train["MLP_Encoder"]["num_input_dim"] != 540:
        raise ValueError("unexpected frozen architecture")
    cfg = configure(BipedCfgWF(), saved, SimpleNamespace(num_envs=30, condition="empty"))
    cfg.env.episode_length_s = 10.
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.static_friction = cfg.terrain.dynamic_friction = FRICTION
    cfg.terrain.restitution = RESTITUTION
    cfg.rewards.scales = SimpleNamespace(**saved["rewards"]["scales"])
    scene_file = args.output / "scenarios" / f"seed_{args.seed}.npz"
    with np.load(scene_file, allow_pickle=False) as loaded:
        scene = dict(loaded)
    spec = importlib.util.spec_from_file_location(
        "legged_gym.envs.wheelfoot_flat._terrain_step_archived", run_dir / "wheelfoot_flat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = terrain_environment(module.BipedWF)
    sim = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), sim)
    sim.use_gpu_pipeline = sim.physx.use_gpu = True
    sim.physx.num_threads = 4
    batch = f"{args.policy}_{args.terrain}_eval{args.seed}"
    directory = args.output / args.phase / batch
    if args.render_probe:
        directory = directory.with_name(directory.name + "_render_probe")
    if (directory / "result.json").exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "frames").mkdir(exist_ok=True)
    start = time.monotonic()
    env = cls(scene, args.terrain, args.capture, cfg=cfg, sim_params=sim,
               physics_engine=gymapi.SIM_PHYSX, sim_device="cuda:0", headless=True)
    try:
        encoder = MLP_Encoder(**train["MLP_Encoder"]).to(env.device).eval()
        actor = ActorCritic(41, 48, 8, **train["policy"]).to(env.device).eval()
        model = run_dir / f"model_{checkpoint}.pt"
        state = torch.load(model, map_location=env.device, weights_only=False)
        actor.load_state_dict(state["model_state_dict"], strict=True)
        encoder.load_state_dict(state["encoder_state_dict"], strict=True)
        actor.requires_grad_(False)
        encoder.requires_grad_(False)
        geometry = wheel_geometry(ROOT / "resources/robots/WF_TRON1A/urdf/robot.urdf")
        offsets, radii, widths = [torch.as_tensor(v, device=env.device) for v in geometry]
        records = {}
        def add(key, value):
            records.setdefault(key, []).append(value.detach().cpu().numpy().copy())
        with torch.inference_mode():
            env.reset()
            env.recording = True
            for index in range(args.steps):
                obs, history, command, critic = env.get_observations()
                estimate = encoder.encode(history)
                actions = actor.act_inference(torch.cat((estimate, obs, command), -1))
                if not bool((command == 0).all()) or not torch.allclose(obs[:, -3], torch.full_like(obs[:, -3], .4)):
                    raise ValueError("fixed pure-step command mismatch")
                if not bool(torch.isfinite(actions).all()):
                    raise ValueError("nonfinite action")
                lowest, ground, clearance = env.wheel_lowest(offsets, radii, widths)
                for key, value in {
                    "time_s": env.episode_length_buf[0].float() * DT,
                    "root_state": env.root_states[env.actor_indices],
                    "rigid_body_state": env.rigid_body_state,
                    "body_velocity": env.base_lin_vel, "body_angular_velocity": env.base_ang_vel,
                    "actions": actions, "estimate_normalized": estimate,
                    "observations": obs, "actor_commands": command,
                    "task_commands": torch.cat((env.commands[:, :3], env.cmd_step), -1),
                    "wheel_forces": env.contact_forces[:, env.feet_indices],
                    "wheel_lowest_world": lowest, "ground_at_lowest": ground,
                    "wheel_clearance_signed": clearance,
                    "ground_at_base": env.sample_ground(env.root_states[env.actor_indices, :3]),
                    "gait_phase": env.gait_indices, "desired_contact": env.desired_contact_states,
                    "first_failure": env.first_failure, "failure_reason": env.failure_reason,
                    "patch_exited": env.patch_exited, "fail_counter": env.fail_buf,
                    "dof_position": env.dof_pos, "dof_velocity": env.dof_vel, "torque": env.torques,
                    "noise_injected": env.last_noise_injected[:, :20],
                    "noise_effective": env.last_noise_effective[:, :20],
                    "has_load": env.has_load,
                }.items():
                    add(key, value)
                t = (index + 1) * DT
                if args.capture and any(abs(t-target) < 1e-6 for target in FRAME_TIMES):
                    env.capture(directory / "frames", t)
                if index + 1 < args.steps:
                    env.step(actions)
        arrays = {key: np.stack(value) for key, value in records.items()}
        arrays.update(base_mass0=env.base_mass0.cpu().numpy().copy(),
                      base_com0=env.base_com0.cpu().numpy().copy(),
                      imu_offset_quaternion=env.random_imu_offset.cpu().numpy().copy(),
                      mesh_anchors_xy=scene["mesh_anchors_xy"],
                      terminal_root=env.terminal_root.cpu().numpy().copy(),
                      first_failure_time_s=env.failure_time.cpu().numpy().copy())
        np.savez_compressed(directory / "trace.npz", **arrays)
        body_properties, shape_properties = [], []
        for eid, (handle, actor_handle) in enumerate(zip(env.envs, env.actor_handles)):
            props = env.gym.get_actor_rigid_body_properties(handle, actor_handle)
            body_properties.append({"env_id": eid, "mass_kg": [p.mass for p in props],
                "inertia_diagonal": [[p.inertia.x.x, p.inertia.y.y, p.inertia.z.z] for p in props]})
            shape_properties.append({"env_id": eid, "shapes": [
                {"friction": p.friction, "restitution": p.restitution,
                 "contact_offset": getattr(p, "contact_offset", None),
                 "rest_offset": getattr(p, "rest_offset", None)}
                for p in env.gym.get_actor_rigid_shape_properties(handle, actor_handle)]})
        write_json(directory / "actual_body_properties.json", body_properties)
        write_json(directory / "actual_shape_properties.json", shape_properties)
        write_json(directory / "effective_env_cfg.json", class_to_dict(cfg))
        write_json(directory / "frames/metadata.json", env.capture_metadata)
        result = {
            "batch_id": batch, "policy_id": args.policy, "policy_name": name,
            "run": run_name, "checkpoint": checkpoint, "checkpoint_sha256": sha256_file(model),
            "terrain": args.terrain, "evaluation_seed": args.seed, "training_seed": train["seed"],
            "started": args.started, "finished": timestamp(), "elapsed_s": time.monotonic()-start,
            "steps": args.steps, "num_envs": 30, "dt_s": DT,
            "physical_gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "gpu_name": torch.cuda.get_device_name(0), "capture_enabled": args.capture,
            "scene_array_sha256": array_digest(scene), "scene_file_sha256": sha256_file(scene_file),
            "mesh_sha256": env.mesh_digest, "task_source_sha256": sha256_file(run_dir / "wheelfoot_flat.py"),
            "trace_sha256": sha256_file(directory / "trace.npz"), "body_names": env.body_names,
            "dof_names": env.dof_names, "feet_indices": env.feet_indices.cpu().tolist(),
            "wheel_collision_offset_m": geometry[0].tolist(), "wheel_radius_m": geometry[1].tolist(),
            "wheel_width_m": geometry[2].tolist(), "robot_failed_count": int(env.first_failure.sum()),
            "wheel_collision_geometry": "capsule" if cfg.asset.replace_cylinder_with_capsule else "cylinder",
            "camera_clean_exit_requested": args.camera_clean_exit,
            "patch_exit_count": int(env.patch_exited.sum()), "camera_frame_count": len(env.capture_metadata),
            "no_reset_after_failure": True, "exit_status": "data_written_cleanup_pending"}
        write_json(directory / "result.json", result)
    finally:
        if env.camera is not None:
            env.gym.destroy_camera_sensor(env.sim, env.envs[0], env.camera)
        env.gym.destroy_sim(env.sim)
    result["exit_status"] = "completed"
    result["cleanup_completed"] = True
    write_json(directory / "result.json", result)
    print(f"[DONE] {batch} steps={args.steps} wall={time.monotonic()-start:.1f}s "
          f"failed={result['robot_failed_count']} exits={result['patch_exit_count']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--terrain", choices=TERRAINS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=12345)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--phase", choices=("preflight", "batches"), default="batches")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--render-probe", action="store_true")
    parser.add_argument("--camera-clean-exit", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.started = timestamp()
    run(args)
    if args.capture and args.camera_clean_exit:
        # All sensors and the simulator were explicitly destroyed; only native module unloading is skipped.
        print("[EXIT] Explicit Gym cleanup completed; bypassing unstable graphics-library unload.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
