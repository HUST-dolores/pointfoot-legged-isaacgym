"""Isolated free-payload retention task using the archived balancing environment."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymtorch, gymutil
import torch

from legged_gym.scripts.payload_step_experiment import configure
from legged_gym.envs.wheelfoot_flat.wheelfoot_flat_config import BipedCfgWF
from legged_gym.utils.helpers import class_to_dict
from .geometry import sha256
from .robot import ROOT, RUN
from .retention_protocol import payload_geometry, quat_rotate, quat_inverse_rotate, retained_reward


def source_directory():
    return ROOT / "logs/wheelfoot_flat/WF_TRON1A" / RUN


def build_environment(settings, num_envs, seed, train=True, viewer=False, capture=False, evaluation_cases=None):
    source = source_directory()
    saved = json.loads((source / "env_cfg.json").read_text())
    cfg = configure(BipedCfgWF(), saved, SimpleNamespace(num_envs=num_envs, condition="free_box"))
    cfg.env.use_gait_stepping = cfg.env.use_gait_phase = False
    cfg.env.episode_length_s = settings.duration
    cfg.env.env_spacing = 5.0
    cfg.asset.file = str(ROOT / "resources/robots/WF_TRON1A/urdf/sliding_lab_raised_collision.urdf")
    if not Path(cfg.asset.file).is_file():
        raise FileNotFoundError("Prepare the existing raised sliding asset before launching concurrent jobs")
    cfg.domain_rand.add_load_range = [settings.mass, settings.mass]
    cfg.domain_rand.randomize_inertia = cfg.domain_rand.randomize_imu_offset = False
    cfg.domain_rand.load_contact_grace_s = 0.2
    cfg.noise.add_noise = False
    cfg.sim.physx.num_position_iterations = 8
    cfg.sim.physx.num_velocity_iterations = 1
    cfg.rewards.scales = SimpleNamespace()
    spec = importlib.util.spec_from_file_location("legged_gym.envs.wheelfoot_flat._retention_archived", source / "wheelfoot_flat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class RetentionEnvironment(module.BipedWF):
        def __init__(self, *args, **kwargs):
            self.retention_ready = False
            self.training_task = train
            self.evaluation_mode = False
            self.reset_permitted = True
            self.material_audit = []
            self.settings = settings
            self.rng = torch.Generator(device="cuda:0")
            self.rng.manual_seed(seed + 179)
            super().__init__(*args, **kwargs)
            n = self.num_envs
            self.spawned_once = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.episode_spawns = torch.zeros(n, dtype=torch.int, device=self.device)
            self.pose_writes = torch.zeros(n, dtype=torch.int, device=self.device)
            self.pending_roots = torch.zeros(len(self.root_states), dtype=torch.bool, device=self.device)
            self.root_write_indices = torch.empty(0, dtype=torch.int32, device=self.device)
            self.awaiting_spawn_check = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.verified_spawns = 0
            self.completed = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.drop_counter = torch.zeros(n, dtype=torch.int, device=self.device)
            self.fall_counter = torch.zeros_like(self.drop_counter)
            self.drop_now = torch.zeros_like(self.completed)
            self.fall_now = torch.zeros_like(self.completed)
            self.drop_latched = torch.zeros_like(self.completed)
            self.fall_latched = torch.zeros_like(self.completed)
            self.time_latched = torch.zeros_like(self.completed)
            self.pulse_dv = torch.zeros(n, device=self.device)
            self.pulse_angle = torch.zeros(n, device=self.device)
            self.pulse_step = torch.full((n,), round(settings.pulse_time / self.dt), dtype=torch.long, device=self.device)
            self.spawn_xy = torch.zeros(n, 2, device=self.device)
            self.applied_force = torch.zeros(n, 3, device=self.device)
            self.pulse_world = torch.zeros_like(self.applied_force)
            self.external_forces = torch.zeros_like(self.contact_forces)
            self.min_margin = torch.full((n,), 1., device=self.device)
            self.max_slip = torch.zeros(n, device=self.device)
            self.spawn_local_xy = torch.zeros(n, 2, device=self.device)
            self.retention_episode_returns = torch.zeros(n, device=self.device)
            self.episode_stats = []
            self.term_means = {}
            self.last_transition = {}
            self.pulse_max_dv = settings.start_dv
            self.cases = evaluation_cases
            self.retention_ready = True
            self._clear_retention(torch.arange(n, device=self.device))

        def create_sim(self):
            if capture or viewer:
                self.graphics_device_id = 0
            return super().create_sim()

        def _create_envs(self):
            super()._create_envs()
            for handle, actor, box in zip(self.envs, self.actor_handles, self.load_handles):
                ranges = self.gym.get_actor_rigid_body_shape_indices(handle, actor)
                r = ranges[self.body_names.index("base_Link")]
                indices = list(range(r.start, r.start + r.count))
                props = self.gym.get_actor_rigid_shape_properties(handle, actor)
                before = [p.friction for p in props]
                for index in indices:
                    props[index].friction = settings.friction
                    props[index].contact_offset = .002
                    props[index].rest_offset = props[index].restitution = 0.
                    props[index].rolling_friction = props[index].torsion_friction = 0.
                self.gym.set_actor_rigid_shape_properties(handle, actor, props)
                bp = self.gym.get_actor_rigid_shape_properties(handle, box)
                for p in bp:
                    p.friction = settings.friction
                    p.contact_offset = .002
                    p.rest_offset = p.restitution = p.rolling_friction = p.torsion_friction = 0.
                self.gym.set_actor_rigid_shape_properties(handle, box, bp)
                after = self.gym.get_actor_rigid_shape_properties(handle, actor)
                assert all(after[i].friction == before[i] for i in range(len(after)) if i not in indices)
                self.material_audit.append({"base_shapes": indices, "robot_friction": [p.friction for p in after],
                    "box_friction": [p.friction for p in self.gym.get_actor_rigid_shape_properties(handle, box)],
                    "wheel_friction_unchanged": True,
                    "mass": self.gym.get_actor_rigid_body_properties(handle, box)[0].mass})

        def _reset_root_states(self, ids):
            self.root_states[self.actor_indices[ids]] = self.base_init_state
            self.root_states[self.actor_indices[ids], :3] += self.env_origins[ids]
            self.pending_roots[self.actor_indices[ids]] = True

        def remove_load(self, ids):
            ids = self._to_env_ids_tensor(ids)
            self.root_states[self.load_indices[ids], :3] = self.root_states.new_tensor([0., 0., -100.])
            self.root_states[self.load_indices[ids], 7:13] = 0.
            self.pending_roots[self.load_indices[ids]] = True

        def _flush_root_writes(self):
            # GPU Tensor API permits one root setter between simulate calls, not one per actor group.
            indices = self.pending_roots.nonzero().flatten().to(torch.int32)
            if indices.numel():
                self.root_write_indices = indices
                self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                    gymtorch.unwrap_tensor(self.root_write_indices), self.root_write_indices.numel())
                self.pending_roots[:] = False

        def _reset_dofs(self, ids):
            self.dof_pos[ids] = self.default_dof_pos[ids]
            self.dof_vel[ids] = 0.
            actor_ids = self.actor_indices[ids].to(torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(actor_ids), len(ids))

        def _resample_commands(self, ids):
            self.commands[ids] = 0.
            if hasattr(self, "cmd_step"):
                self.cmd_step[ids] = 0.

        def set_learning_iteration(self, iteration):
            self.learning_iteration = iteration
            f = min(1., iteration / max(1, settings.curriculum_iterations))
            self.pulse_max_dv = settings.start_dv + f * (settings.max_dv - settings.start_dv)

        def _clear_retention(self, ids):
            self.spawned_once[ids] = False
            self.awaiting_spawn_check[ids] = False
            self.episode_spawns[ids] = 0
            self.completed[ids] = False
            self.drop_counter[ids] = self.fall_counter[ids] = 0
            self.drop_now[ids] = self.fall_now[ids] = False
            self.drop_latched[ids] = self.fall_latched[ids] = self.time_latched[ids] = False
            self.min_margin[ids] = 1.
            self.max_slip[ids] = 0.
            self.retention_episode_returns[ids] = 0.
            n = len(ids)
            if self.cases is not None:
                self.pulse_dv[ids] = torch.tensor([self.cases[i]["nominal_dv"] for i in ids.tolist()], device=self.device)
                self.pulse_angle[ids] = torch.tensor([self.cases[i]["angle"] for i in ids.tolist()], device=self.device)
                self.pulse_step[ids] = round(settings.pulse_time / self.dt)
            else:
                self.pulse_dv[ids] = (0.4 + 0.6 * torch.rand(n, device=self.device, generator=self.rng)) * self.pulse_max_dv
                self.pulse_angle[ids] = torch.rand(n, device=self.device, generator=self.rng) * (2 * math.pi)
                self.pulse_step[ids] = round(settings.pulse_time / self.dt) + torch.randint(-5, 6, (n,), device=self.device, generator=self.rng)
            self.spawn_xy[ids] = (2 * torch.rand(n, 2, device=self.device, generator=self.rng) - 1) * settings.initial_offset
            self.applied_force[ids] = self.pulse_world[ids] = 0.

        def reset_idx(self, ids):
            if not len(ids):
                return
            if self.retention_ready:
                if self.evaluation_mode:
                    self.completed[ids] = True
                    return
                if not self.reset_permitted:
                    assert bool((self.drop_now[ids] | self.fall_now[ids] | self.time_out_buf[ids]).all()), "unrequested payload reset"
                for i in ids.tolist():
                    if self.episode_length_buf[i] > 0:
                        self.episode_stats.append({"length_s": float(self.episode_length_buf[i]) * self.dt,
                            "drop": bool(self.drop_now[i]), "fall": bool(self.fall_now[i]),
                            "timeout": bool(self.time_out_buf[i]), "nominal_dv": float(self.pulse_dv[i]),
                            "angle": float(self.pulse_angle[i]), "min_margin": float(self.min_margin[i]),
                            "max_slip": float(self.max_slip[i]), "spawns": int(self.episode_spawns[i]),
                            "return": float(self.retention_episode_returns[i])})
            super().reset_idx(ids)
            if self.retention_ready:
                self._clear_retention(ids)
                self.pose_writes[ids] += 1

        def _maybe_spawn_loads(self):
            if not self.retention_ready:
                return
            verify_ids = self.awaiting_spawn_check.nonzero().flatten()
            if verify_ids.numel():
                separation = (self.root_states[self.load_indices[verify_ids], :3] -
                              self.root_states[self.actor_indices[verify_ids], :3]).norm(dim=-1)
                assert bool((separation < .5).all()), "Payload spawn root state was not applied by PhysX"
                self.verified_spawns += verify_ids.numel()
                self.awaiting_spawn_check[verify_ids] = False
            ids = ((~self.spawned_once) & (self.episode_length_buf >= round(settings.spawn_time / self.dt))).nonzero().flatten()
            if not ids.numel():
                return
            root = self.root_states[self.actor_indices[ids]]
            offset = torch.cat((self.spawn_xy[ids], torch.full((len(ids), 1), settings.plate_z + .0555, device=self.device)), -1)
            displacement = quat_rotate(root[:, 3:7], offset)
            state = root.clone()
            state[:, :3] += displacement
            state[:, 7:10] += torch.cross(root[:, 10:13], displacement, dim=-1)
            self.root_states[self.load_indices[ids]] = state
            self.pending_roots[self.load_indices[ids]] = True
            self.awaiting_spawn_check[ids] = True
            self.spawned_once[ids] = self.has_load[ids] = True
            self.episode_spawns[ids] += 1
            self.pose_writes[ids] += 1
            assert bool((self.episode_spawns[ids] == 1).all())
            self.loads_spawned_count[ids] += 1
            self.load_remove_time[ids] = self.next_load_spawn_time[ids] = 1e6
            self.load_contact_grace_counter[ids] = self.load_contact_grace_steps
            self.spawn_local_xy[ids] = self.spawn_xy[ids]

        def geometry(self):
            return payload_geometry(self.root_states[self.actor_indices], self.root_states[self.load_indices],
                                    self.contact_forces[:, self.num_bodies], settings)

        def get_observations(self):
            geom = self.geometry()
            context = geom["context"] * self.has_load[:, None]
            return self.obs_buf, self.encoder_obs_history, torch.cat((self.commands[:, :3] * self.commands_scale, context), -1), self.critic_obs_buf

        def _post_physics_step_callback(self):
            self._resample_commands(torch.arange(self.num_envs, device=self.device))
            return super()._post_physics_step_callback()

        def check_termination(self):
            if not self.retention_ready:
                return super().check_termination()
            geom = self.geometry()
            t = self.episode_length_buf * self.dt
            active = self.has_load & (t > settings.spawn_time + .2)
            self.drop_counter = torch.where(active & geom["departure"], self.drop_counter + 1, 0)
            root = self.root_states[self.actor_indices]
            falling = (self.projected_gravity[:, 2] > -.5) | (root[:, 2] < .35)
            self.fall_counter = torch.where(falling, self.fall_counter + 1, 0)
            self.drop_now = self.drop_counter >= round(settings.drop_confirm / self.dt)
            self.fall_now = self.fall_counter >= round(settings.fall_confirm / self.dt)
            self.time_out_buf = (self.episode_length_buf >= round(settings.duration / self.dt)) & (~self.drop_now) & (~self.fall_now)
            self.drop_latched |= self.drop_now
            self.fall_latched |= self.fall_now
            self.time_latched |= self.time_out_buf
            self.edge_reset_buf[:] = False
            self.reset_buf = self.drop_now | self.fall_now | self.time_out_buf
            self.min_margin = torch.where(active, torch.minimum(self.min_margin, geom["margins"].amin(-1)), self.min_margin)
            slip = (geom["local"][:, :2] - self.spawn_local_xy).norm(dim=-1)
            self.max_slip = torch.where(active & geom["supported"], torch.maximum(self.max_slip, slip), self.max_slip)
            self.last_transition = {"drop": self.drop_now.clone(), "fall": self.fall_now.clone(),
                "timeout": self.time_out_buf.clone(), "spawns": self.episode_spawns.clone(),
                "pose_writes": self.pose_writes.clone(), "root": root.clone(),
                "box": self.root_states[self.load_indices].clone(), "margins": geom["margins"].clone(),
                "supported": geom["supported"].clone(), "gap": geom["gap"].clone(),
                "normal_force": geom["normal_force"].clone(), "local": geom["local"].clone(),
                "relative_velocity": geom["relative_velocity"].clone(), "force": self.applied_force.clone()}
            self.extras = {"time_outs": self.time_out_buf.clone()}

        def compute_reward(self):
            if not self.retention_ready:
                self.rew_buf[:] = 0.
                return
            self.rew_buf[:], terms = retained_reward(self.geometry(), self.has_load, self.projected_gravity,
                self.base_lin_vel, self.base_ang_vel, self.root_states[self.actor_indices, 2], self.actions,
                self.last_actions[:, :, 0], self.torques, self.drop_now, self.fall_now, settings, self.dt)
            self.retention_episode_returns += self.rew_buf
            self.term_means = {key: float(value.mean()) for key, value in terms.items()}

        def step(self, actions):
            self._flush_root_writes()
            self.reset_permitted = False
            steps = self.episode_length_buf
            first = self.has_load & (steps == self.pulse_step)
            if bool(first.any()):
                local = torch.stack((self.pulse_angle.cos(), self.pulse_angle.sin(), torch.zeros_like(self.pulse_angle)), -1)
                self.pulse_world[first] = quat_rotate(self.base_quat[first], local[first]) * (settings.mass * self.pulse_dv[first, None] / settings.pulse_duration)
            pulse = self.has_load & (steps >= self.pulse_step) & (steps < self.pulse_step + round(settings.pulse_duration / self.dt))
            self.applied_force = self.pulse_world * pulse[:, None]
            self.external_forces[:] = 0.
            self.external_forces[:, self.num_bodies] = self.applied_force
            result = super().step(actions)
            self._flush_root_writes()
            obs, history, commands, critic = self.get_observations()
            return obs, result[1], result[2], result[3], history, commands, critic

        def _compute_torques(self, actions):
            if self.retention_ready:
                self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.external_forces.view(-1, 3)), None, gymapi.ENV_SPACE)
            return super()._compute_torques(actions)

    sim = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), sim)
    sim.use_gpu_pipeline = sim.physx.use_gpu = True
    sim.physx.num_threads = 2
    env = RetentionEnvironment(cfg=cfg, sim_params=sim, physics_engine=gymapi.SIM_PHYSX,
                               sim_device="cuda:0", headless=not viewer)
    return env, cfg
