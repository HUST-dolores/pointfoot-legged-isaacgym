"""Opt-in standing ship task with deployable geometric observation references."""
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.wheelfoot_flat.wheelfoot_flat import BipedWF
from .wheelfoot_ship import BipedWFShip
from legged_gym.utils.ship_model_prior import WheelLegKinematics, ShipModelPrior, FEATURE_DIM


class BipedWFShipPrior(BipedWFShip):
    prior_slice = slice(29, 29+FEATURE_DIM)

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.ship_model_prior
        if not cfg.enabled or self.num_obs != 37+FEATURE_DIM or self.num_critic_obs != 44+FEATURE_DIM:
            raise ValueError('ship prior task requires enabled prior and obs61/critic68')
        if getattr(self.cfg.domain_rand, 'randomize_morphology', False):
            raise ValueError('ship prior currently requires one nominal calibrated URDF, not per-env morphology')
        if (self.cfg.env.use_gait_phase or self.cfg.env.use_gait_stepping
            or self.cfg.env.use_qs_in_obs or self.cfg.env.use_torques_in_obs
            or getattr(self.cfg.env, 'encoder_include_torques', False)):
            raise ValueError('ship prior observation layout is standing state20 + IMU9 + prior24 + actions8')
        if len(self.feet_indices) != 2:
            raise ValueError('ship prior requires exactly two wheel contact sensors')
        path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        kinematics = WheelLegKinematics(path, cfg.joint_names, cfg.wheel_joint_names, self.device)
        names = [name for leg in cfg.joint_names for name in leg]
        self.prior_joint_indices = torch.tensor([self.dof_names.index(name) for name in names], device=self.device)
        # Explicitly map L/R wheel names; never assume PhysX body order matches
        # the configured URDF chain order or infer contacts from world normals.
        foot_names = [self.body_names[index] for index in self.feet_indices.tolist()]
        self.prior_contact_indices = torch.tensor(
            [self.feet_indices[foot_names.index(name)] for name in cfg.wheel_body_names], device=self.device)
        self.ship_model_prior = ShipModelPrior(kinematics, self.num_envs, self.dt, self.cfg.asset.foot_radius, cfg)

    def reset_idx(self, env_ids):
        # BipedWF.reset_idx constructs both raw and filtered histories during
        # its own call. Clear first, never after stale references enter history.
        if hasattr(self, 'ship_model_prior'):
            self.ship_model_prior.reset(env_ids)
        super().reset_idx(env_ids)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if hasattr(self, 'ship_model_prior'):
            q = self.dof_pos[:, self.prior_joint_indices].reshape(self.num_envs, 2, 3)
            force = self.contact_forces[:, self.prior_contact_indices].norm(dim=-1)
            # These are the existing IMU/joint measurement caches. No deck or
            # robot root quaternion, foot world pose, or true future command.
            self.ship_model_prior.update(self.deck_imu, self.base_ang_vel,
                self.projected_gravity, q, force, imu_enabled=self.cfg.ship.imu_enabled)

    def compute_group_observations(self):
        flat, _ = BipedWF.compute_group_observations(self)
        imu = self.deck_imu.clone()
        imu[:, :3] *= self.cfg.ship.gyro_scale
        imu[:, 3:6] *= self.cfg.ship.accel_scale
        if not self.cfg.ship.imu_enabled:
            imu.zero_()
        if hasattr(self, 'ship_model_prior'):
            features = self.ship_model_prior.features
            if not self.cfg.ship.imu_enabled:
                features = torch.zeros_like(features)
        else:
            features = flat.new_zeros(self.num_envs, FEATURE_DIM)
        obs = torch.cat((flat[:, :20], imu, features, flat[:, -8:]), dim=-1)
        critic = torch.cat((self.base_lin_vel*self.obs_scales.lin_vel,
            (self.base_mass-self.base_mass0)[:, None]*self.obs_scales.mass_scale,
            (self.base_com-self.base_com0)*self.obs_scales.com_scale, obs), dim=-1)
        if obs.shape[-1] != self.num_obs or critic.shape[-1] != self.num_critic_obs:
            raise RuntimeError('ship prior ordinary-observation contract disagrees')
        return obs, critic
