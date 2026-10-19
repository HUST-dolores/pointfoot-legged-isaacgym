"""Deck-relative locomotion without changing the established standing task."""
import torch
from .wheelfoot_ship import BipedWFShip
from legged_gym.envs.wheelfoot_flat.wheelfoot_flat import BipedWF
from legged_gym.utils.ship_motion import (
    shuttle_velocity, mobility_clock, mobility_observations, foot_deck_state,
    surface_pose,
)


class BipedWFShipMobility(BipedWFShip):
    def _init_buffers(self):
        super()._init_buffers()
        if self.cfg.ship_mobility.mode not in ('roll', 'step'):
            raise ValueError('ship mobility mode must be roll or step')
        if self.cfg.env.use_gait_phase or self.cfg.env.use_gait_stepping:
            raise ValueError('ship mobility uses its own clock, not flat-task gait suffixes')
        self.mobility_clearance = torch.zeros(self.num_envs, 2, device=self.device)
        self.mobility_slip = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.mobility_contacts = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.mobility_desired_contacts = torch.ones(self.num_envs, 2, device=self.device)

    def _resample_commands(self, env_ids):
        # Parent reset calls this BEFORE zeroing episode_length_buf. Never seed
        # a new episode/history with the old episode's shuttle command.
        self.commands[env_ids, :3] = 0.
        self.cmd_step[env_ids] = 0.

    def _mobility_clock(self):
        cfg = self.cfg.ship_mobility
        time = self.episode_length_buf*self.dt
        return mobility_clock(time, cfg.mode == 'step', cfg.gait_frequency_hz, cfg.stance_duty)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        cfg = self.cfg.ship_mobility
        scale = 1.
        if cfg.speed_curriculum:
            progress = min(1., max(0., self.learning_iteration/max(1, cfg.speed_ramp_iterations)))
            scale = cfg.initial_speed_fraction+(1-cfg.initial_speed_fraction)*progress
        self.commands[:, 0] = shuttle_velocity(self.episode_length_buf*self.dt,
            cfg.max_speed_m_s*scale, cfg.period_s, cfg.command_ramp_s)
        self.commands[:, 1:3] = 0.
        _, self.mobility_desired_contacts = self._mobility_clock()
        feet = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        deck = self.root_states[self.deck_indices]
        self.mobility_clearance, self.mobility_slip = foot_deck_state(
            deck, feet, self.cfg.ship.thickness, self.cfg.asset.foot_radius)
        _, normal = surface_pose(deck, self.cfg.ship.thickness)
        force = self.contact_forces[:, self.feet_indices]
        self.mobility_contacts = (force*normal[:, None, :]).sum(-1) > cfg.contact_threshold_n

    def compute_group_observations(self):
        # Calling the flat builder directly avoids its action-after-gait layout
        # and the standing task's fixed last-eight-actions assumption.
        flat, _ = BipedWF.compute_group_observations(self)
        imu = self.deck_imu.clone()
        imu[:, :3] *= self.cfg.ship.gyro_scale
        imu[:, 3:6] *= self.cfg.ship.accel_scale
        if not self.cfg.ship.imu_enabled:
            imu.zero_()
        features, _ = self._mobility_clock()
        obs = mobility_observations(flat, imu, features)
        critic = torch.cat((self.base_lin_vel*self.obs_scales.lin_vel,
            (self.base_mass-self.base_mass0)[:, None]*self.obs_scales.mass_scale,
            (self.base_com-self.base_com0)*self.obs_scales.com_scale, obs), dim=-1)
        if obs.shape[-1] != self.num_obs or critic.shape[-1] != self.num_critic_obs:
            raise RuntimeError('ship mobility observation dimensions disagree')
        return obs, critic

    def _reward_ship_mobility_tracking(self):
        # Commands are robot-frame forward/lateral velocities relative to the
        # physical deck, not inaccessible deck-heading coordinates.
        return torch.exp(-(self.base_lin_vel[:, :2]-self.commands[:, :2]).square().sum(-1)/.15)

    def _reward_ship_mobility_contacts(self):
        return (self.mobility_contacts.float() == self.mobility_desired_contacts).float().mean(-1)

    def _reward_ship_mobility_clearance(self):
        if self.cfg.ship_mobility.mode != 'step':
            return torch.zeros(self.num_envs, device=self.device)
        cfg = self.cfg.ship_mobility
        swing = 1-self.mobility_desired_contacts
        quality = torch.exp(-((self.mobility_clearance-cfg.clearance_target_m)/cfg.clearance_sigma_m).square())
        return (swing*quality).sum(-1)/swing.sum(-1).clamp(min=1)

    def _reward_ship_mobility_slip(self):
        return (self.mobility_slip.square().sum(-1)*self.mobility_contacts).mean(-1).clamp(max=4.)

    def _reward_ship_mobility_center(self):
        return self.ship_local_position[:, :2].square().sum(-1).clamp(max=9.)
