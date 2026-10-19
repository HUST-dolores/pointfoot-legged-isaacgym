"""Moving support + deployable ordinary deck-IMU policy observations.

Reuse the preallocated payload actor as a one-body deck: robot DOF layout and
all actor/body tensor sizes stay unchanged. This task cannot add a free load.
The deck is a massive gravity-disabled dynamic body whose consistent pose AND
velocities are prescribed each physics tick. Small contact reactions are
allowed within a tick; this approximation must be checked by the probe.
"""
import math
import torch
from isaacgym import gymapi, gymtorch
from legged_gym.envs.wheelfoot_flat.wheelfoot_flat import BipedWF
from legged_gym.utils.ship_motion import (
    motion_state, rotate, inverse_rotate, surface_pose, plane_height,
    point_velocity, imu_sample, gravity_projected_support_state, heading_yaw_rate,
)


class BipedWFShip(BipedWF):
    imu_slice = slice(20, 29)

    def _create_ground_plane(self):
        # No hidden world floor: ALL support contacts must come from the deck.
        pass

    def _auxiliary_start_pose(self, env_id):
        # Broadphase pairs are allocated in prepare_sim, before _init_buffers.
        # Never initially overlap thousands of 6 x 4 m decks at the payload's
        # shared hiding location: collision groups do not avoid that allocation.
        origin = self.env_origins[env_id].detach().cpu().tolist()
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(origin[0], origin[1], origin[2]
                            + self.cfg.ship.surface_z - self.cfg.ship.thickness/2)
        return pose

    def _create_auxiliary_asset(self, options):
        cfg = self.cfg.ship
        if min(cfg.length, cfg.width, cfg.thickness, cfg.mass_kg, cfg.ramp_s) <= 0:
            raise ValueError("deck dimensions, mass and ramp must be positive")
        if self.cfg.domain_rand.add_random_load or self.cfg.domain_rand.per_env_load_mass:
            raise ValueError("ship task reserves the auxiliary actor for the deck")
        options.fix_base_link = False
        options.disable_gravity = True
        options.density = cfg.mass_kg / (cfg.length * cfg.width * cfg.thickness)
        options.linear_damping = 0.0
        options.angular_damping = 0.0
        return self.gym.create_box(self.sim, cfg.length, cfg.width, cfg.thickness, options)

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.ship
        self.deck_indices = self.load_indices  # real simulation-domain actor indices
        self.ship_time = torch.zeros(self.num_envs, device=self.device)
        self.ship_difficulty = torch.full_like(self.ship_time, cfg.initial_difficulty)
        self.ship_amplitude = torch.zeros(self.num_envs, 3, device=self.device)
        self.ship_frequency = torch.zeros_like(self.ship_amplitude)
        self.ship_phase = torch.zeros_like(self.ship_amplitude)
        self.ship_pivot = self.env_origins.clone()
        self.ship_pivot[:, 2] += cfg.surface_z
        self.ship_pending_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ship_robot_reset_state = torch.zeros(self.num_envs, 13, device=self.device)
        self.deck_imu = torch.zeros(self.num_envs, 9, device=self.device)
        self.deck_imu[:, 5] = 9.81
        self.deck_imu[:, 8] = -1.0
        delay = int(cfg.imu_delay_steps)
        if delay < 0:
            raise ValueError("imu_delay_steps must be nonnegative")
        self.ship_imu_history = self.deck_imu[:, None, :].repeat(1, delay+1, 1)
        self.ship_previous_twist = torch.zeros(self.num_envs, 6, device=self.device)
        self.ship_anchor = torch.zeros(self.num_envs, 2, device=self.device)
        self.ship_pose_error_max = torch.zeros_like(self.ship_time)
        self._reset_ship(torch.arange(self.num_envs, device=self.device))
        self.root_states[self.deck_indices] = self._deck_command()[0]

    def _deck_command(self):
        return motion_state(self.ship_time, self.ship_amplitude, self.ship_frequency,
                            self.ship_phase, self.ship_pivot, self.cfg.ship.thickness,
                            self.cfg.ship.ramp_s)

    def _reset_ship(self, env_ids):
        cfg = self.cfg.ship
        if self.init_done and cfg.curriculum:
            survived = self.episode_length_buf[env_ids] >= 0.9*self.max_episode_length
            delta = torch.where(survived, cfg.curriculum_step, -cfg.curriculum_step)
            self.ship_difficulty[env_ids] = (self.ship_difficulty[env_ids]+delta).clamp(0, 1)
        self.ship_time[env_ids] = 0
        n = len(env_ids)
        low = torch.tensor(cfg.amplitude_min, device=self.device)
        high = torch.tensor(cfg.amplitude_max, device=self.device)
        self.ship_amplitude[env_ids] = low + torch.rand(n, 3, device=self.device)*(high-low)
        corner_fraction = float(getattr(cfg,'corner_fraction',0.0))
        if not 0.0 <= corner_fraction <= 1.0:
            raise ValueError('corner_fraction must lie in [0,1]')
        corner = env_ids < int(self.num_envs*corner_fraction)
        if corner_fraction > 0:
            self.ship_amplitude[env_ids[corner]] = high
        self.ship_amplitude[env_ids] *= self.ship_difficulty[env_ids, None]
        if getattr(cfg, "training_schedule", False):
            # Freeze the scheduled amplitude for each episode. This avoids pose
            # jumps when the PPO iteration changes while a ship is in motion.
            self.ship_amplitude[env_ids] *= self.training_motion_scale()
        low = torch.tensor(cfg.frequency_min, device=self.device)
        high = torch.tensor(cfg.frequency_max, device=self.device)
        self.ship_frequency[env_ids] = low + torch.rand(n, 3, device=self.device)*(high-low)
        if corner_fraction > 0:
            self.ship_frequency[env_ids[corner]] = high
        self.ship_phase[env_ids] = 2*math.pi*torch.rand(n, 3, device=self.device)
        self.ship_previous_twist[env_ids] = 0
        self.deck_imu[env_ids] = 0
        self.deck_imu[env_ids, 5] = 9.81
        self.deck_imu[env_ids, 8] = -1
        self.ship_imu_history[env_ids] = self.deck_imu[env_ids, None, :]
        command = self._deck_command()[0]
        self.root_states[self.deck_indices[env_ids]] = command[env_ids]

    def training_motion_scale(self):
        cfg = self.cfg.ship
        warmup = int(getattr(cfg, "warmup_iterations", 1000))
        if getattr(cfg,"motion_gate_enabled",False):
            start=getattr(cfg,"motion_start_iteration",None)
            if start is None:
                return torch.zeros(3,device=self.device)
            warmup=int(start)
        ramp = max(1, int(getattr(cfg, "motion_ramp_iterations", 4000)))
        u = min(1.0, max(0.0, (self.learning_iteration-warmup)/ramp))
        return torch.tensor([u, max(0., (u-.2)/.8), max(0., (u-.4)/.6)],
                            device=self.device)

    def remove_load(self, env_id):
        # Parent reset must not hide the support actor at z=-100.
        pass

    def is_load_on_body(self, env_ids=None, **kwargs):
        n = self.num_envs if env_ids is None else len(self._to_env_ids_tensor(env_ids))
        return torch.zeros(n, dtype=torch.bool, device=self.device)

    def _reset_root_states(self, env_ids):
        state = self.base_init_state.repeat(len(env_ids), 1)
        top, _ = surface_pose(self.root_states[self.deck_indices[env_ids]], self.cfg.ship.thickness)
        state[:, :3] += top
        state[:, 7:13] = 0
        self.root_states[self.actor_indices[env_ids]] = state
        self.ship_robot_reset_state[env_ids] = state
        self.ship_pending_reset[env_ids] = True
        # Parent reset seeds raw/filtered GRU history immediately after this
        # callback. Synchronize proprioception BEFORE that history is built.
        self.base_position[env_ids] = state[:, :3]
        self.base_quat[env_ids] = state[:, 3:7]
        self.base_lin_vel[env_ids] = 0
        self.base_ang_vel[env_ids] = 0
        self.projected_gravity[env_ids] = inverse_rotate(state[:,3:7],self.gravity_vec[env_ids])
        self.base_height[env_ids] = state[:,2]-top[:,2]
        # Flush together with deck states before the next simulate. GPU tensor
        # root-state setters must be called only once between simulate calls.

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_ship(env_ids)
        super().reset_idx(env_ids)
        state = self.root_states[self.actor_indices[env_ids]]
        self.base_position[env_ids] = state[:, :3]
        self.base_quat[env_ids] = state[:, 3:7]
        self.base_lin_vel[env_ids] = 0
        self.base_ang_vel[env_ids] = 0
        self.last_base_position[env_ids] = state[:, :3]
        self.ship_anchor[env_ids] = 0
        if self.cfg.env.send_timeouts:
            # Falling off this finite deck is terminal, not a time-limit bootstrap.
            self.extras["time_outs"] = self.time_out_buf

    def _before_physics_substep(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        target, _, _ = self._deck_command()
        actual = self.root_states[self.deck_indices]
        valid = (self.ship_time > 0) & (~self.ship_pending_reset)
        error = torch.where(valid, (actual[:, :3]-target[:, :3]).norm(dim=-1),
                            torch.zeros_like(self.ship_time))
        self.ship_pose_error_max = torch.maximum(
            self.ship_pose_error_max, error)
        self.root_states[self.deck_indices] = target
        reset_ids = self.ship_pending_reset.nonzero(as_tuple=False).flatten()
        if reset_ids.numel():
            self.root_states[self.actor_indices[reset_ids]] = self.ship_robot_reset_state[reset_ids]
        # Retain this buffer through simulate: Gym borrows the tensor memory.
        self.ship_root_update_indices = torch.cat(
            (self.deck_indices, self.actor_indices[reset_ids])).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self.ship_root_update_indices),
            self.ship_root_update_indices.numel())
        self.ship_pending_reset[:] = False
        self.ship_time += self.sim_params.dt

    def _sample_deck_imu(self):
        root = self.root_states[self.deck_indices]
        twist = root[:, 7:13]
        derivative = (twist-self.ship_previous_twist) / self.dt
        self.ship_previous_twist[:] = twist
        offset = torch.tensor(self.cfg.ship.sensor_offset, device=self.device)
        sample = imu_sample(root, derivative[:, :3], derivative[:, 3:], offset)
        cfg = self.cfg.ship
        noise = torch.tensor([cfg.gyro_noise_std]*3 + [cfg.accel_noise_std]*3
                             + [cfg.attitude_noise_std]*3, device=self.device)
        sample = sample + torch.randn_like(sample)*noise
        self.ship_imu_history = torch.cat((self.ship_imu_history[:, 1:], sample[:, None]), dim=1)
        self.deck_imu[:] = self.ship_imu_history[:, 0]

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._sample_deck_imu()
        self._update_ship_relative_state()

    def _update_ship_relative_state(self):
        deck = self.root_states[self.deck_indices]
        robot = self.root_states[self.actor_indices]
        top, normal = surface_pose(deck, self.cfg.ship.thickness)
        self.ship_body_local_position = inverse_rotate(deck[:,3:7],robot[:,:3]-top)
        mode = getattr(self.cfg.ship,'reference_mode','body')
        if mode == 'body':
            rel_world = robot[:, 7:10] - point_velocity(deck, robot[:, :3])
            self.ship_relative_velocity = inverse_rotate(deck[:,3:7],rel_world)
            self.ship_local_position = self.ship_body_local_position
        elif mode == 'gravity_projection':
            (self.ship_local_position,self.ship_relative_velocity,rel_world) = gravity_projected_support_state(
                deck,robot,self.cfg.ship.thickness)
        else:
            raise ValueError('unknown ship reference_mode: '+str(mode))
        self.base_lin_vel[:] = inverse_rotate(robot[:, 3:7], rel_world)
        self.base_height[:] = ((robot[:, :3]-top)*normal).sum(-1)

    def _get_foot_heights(self):
        top, normal = surface_pose(self.root_states[self.deck_indices], self.cfg.ship.thickness)
        return plane_height(self.foot_positions[:, :, :2], top, normal)

    def compute_group_observations(self):
        obs, _ = super().compute_group_observations()
        imu = self.deck_imu.clone()
        imu[:, :3] *= self.cfg.ship.gyro_scale
        imu[:, 3:6] *= self.cfg.ship.accel_scale
        if not self.cfg.ship.imu_enabled:
            imu.zero_()  # removes sensor from actor, critic obs, AND encoder history
        obs = torch.cat((obs[:, :-8], imu, obs[:, -8:]), dim=-1)
        # Rebuild with CURRENT ordinary obs, retaining the fixed 7-dim critic prefix.
        critic = torch.cat((self.base_lin_vel*self.obs_scales.lin_vel,
                            (self.base_mass-self.base_mass0)[:, None]*self.obs_scales.mass_scale,
                            (self.base_com-self.base_com0)*self.obs_scales.com_scale, obs), dim=-1)
        if obs.shape[1] != self.num_obs or critic.shape[1] != self.num_critic_obs:
            raise RuntimeError("ship observation contract does not match configuration")
        return obs, critic

    def check_termination(self):
        # Base plane termination otherwise reuses the previous tick's edge
        # flag, causing a second terminal transition after a successful reset.
        self.edge_reset_buf.zero_()
        super().check_termination()
        cfg = self.cfg.ship
        # Keep physical termination identical across reward-reference studies.
        self.edge_reset_buf = ((self.ship_body_local_position[:, 0].abs() > cfg.length/2-cfg.edge_margin)
                               | (self.ship_body_local_position[:, 1].abs() > cfg.width/2-cfg.edge_margin))
        self.reset_buf |= self.edge_reset_buf | (self.base_height < 0.35)
        self.ship_termination_causes = {
            "contact": (self.contact_forces[:, self.termination_contact_indices].norm(dim=-1) > 10).any(dim=1),
            "inversion": self.projected_gravity[:, 2] > -0.1,
            "edge": self.edge_reset_buf.clone(),
            "low_height": self.base_height < 0.35,
        }
        if self.cfg.env.send_timeouts:
            # Update EVERY tick, not only in reset_idx. BaseTask replaces
            # time_out_buf each tick; extras otherwise retains an old True
            # tensor and PPO repeatedly bootstraps on nonterminal transitions.
            self.extras["time_outs"] = self.time_out_buf.clone()

    def _reward_ship_tracking(self):
        return torch.exp(-((self.ship_relative_velocity[:, :2]-self.commands[:, :2])**2).sum(-1)/0.15)

    def _reward_ship_yaw_rate(self):
        robot = self.root_states[self.actor_indices]
        rate = heading_yaw_rate(robot[:, 3:7], robot[:, 10:13])
        sigma = self.cfg.rewards.heading_yaw_tracking_sigma
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('heading yaw tracking sigma must be finite and positive')
        return torch.exp(-(rate-self.commands[:, 2]).square()/sigma)

    def _reward_ship_position(self):
        stand = self.commands[:, :2].norm(dim=-1) < 0.05
        error = self.ship_local_position[:, :2] - self.ship_anchor
        return torch.exp(-(error**2).sum(-1)/0.25) * stand.float()

    def _reward_base_height(self):
        return (self.base_height-self._base_height_target_tensor())**2
