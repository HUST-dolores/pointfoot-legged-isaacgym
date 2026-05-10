import math
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import random

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .wheelfoot_flat_config import BipedCfgWF
from legged_gym.utils.helpers import class_to_dict

class BipedWF(BaseTask):
    def __init__(
        self, cfg: BipedCfgWF, sim_params, physics_engine, sim_device, headless
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2
        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        print(f"=== 实际机器人类型: 轮足机器人+板 ===")
        print(f"===============================")

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)
        self._reset_load_timers(env_ids)
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        # self._resample_gaits(env_ids)
        self.remove_load(env_ids)
        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.root_states[self.actor_indices[env_ids], :3]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        if hasattr(self, "load_estimation_filter_initialized"):
            self.load_estimation_filter_initialized[env_ids] = False
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        
        raw_state_reset = obs_buf[env_ids, :self.filtered_size]
        self.filtered_obs_buf[env_ids] = raw_state_reset
        reset_filter_state = raw_state_reset[:, self.filter_feature_indices]
        self.butter_x1[env_ids] = reset_filter_state
        self.butter_x2[env_ids] = reset_filter_state
        self.butter_y1[env_ids] = reset_filter_state
        self.butter_y2[env_ids] = reset_filter_state
        cat_obs = torch.cat([obs_buf[env_ids], self.filtered_obs_buf[env_ids]], dim=-1)
        self.encoder_obs_history[env_ids] = cat_obs.repeat(1, self.obs_history_length)

        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # 负载标签覆盖率/一致性统计（用于诊断extra_loss监督有效样本）
        if hasattr(self, "has_load"):
            has_load_sub = self.has_load[env_ids]
            self.extras["episode"]["load_has_ratio"] = has_load_sub.float().mean()

            if hasattr(self, "load_on_body_last"):
                on_body_sub = self.load_on_body_last[env_ids]
                self.extras["episode"]["load_on_body_ratio"] = on_body_sub.float().mean()
                denom = has_load_sub.float().sum().clamp_min(1.0)
                self.extras["episode"]["load_on_body_given_has"] = (
                    (on_body_sub & has_load_sub).float().sum() / denom
                )
            if hasattr(self, "load_hysteresis_agree_last"):
                self.extras["episode"]["load_hysteresis_agree"] = self.load_hysteresis_agree_last
        if self.cfg.domain_rand.push_robots:
            self.extras["episode"]["max_push_vel_xy"] = torch.tensor(
                self._get_curriculum_push_vel_xy(), device=self.device
            )
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf
    # def _reset_load(self, env_ids):

    #     goal_displacement = gymapi.Vec3(0, 0, 0.1)
    #     goal_displacement_tensor = to_torch(
    #         [goal_displacement.x, goal_displacement.y, goal_displacement.z], device=self.device)

    #     self.load_indices = to_torch(self.load_indices, dtype=torch.long, device=self.device)
    #     self.root_states[self.load_indices[env_ids], 0:3] = self.load_states[env_ids, 0:3]+ goal_displacement_tensor
    #     self.root_states[self.load_indices[env_ids], 3:7] = self.load_states[env_ids, 3:7]
    #     self.root_states[self.load_indices[env_ids], 7:13] = torch.zeros_like(self.root_states[self.load_indices[env_ids], 7:13])
    #     # 旋转重置：单位四元数
    #     load_indices = self.load_indices[env_ids].to(torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                      gymtorch.unwrap_tensor(self.root_states),
    #                                                      gymtorch.unwrap_tensor(load_indices), len(load_indices))
    #     print(f"self.root_states dtype: {self.root_states.dtype}, shape: {self.root_states.shape}")
    #     print(f"self.load_indices dtype: {self.load_indices.dtype}, shape: {self.load_indices.shape}")
    #     print(f"env_ids dtype: {env_ids.dtype}, shape: {env_ids.shape}")
    #     print(f"self.load_states dtype: {self.load_states.dtype}, shape: {self.load_states.shape}")
    #     print(f"goal_displacement_tensor dtype: {goal_displacement_tensor.dtype}, shape: {goal_displacement_tensor.shape}")
    def step(self, actions):
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.encoder_obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )
        
    def _action_clip(self, actions):
        self.actions = actions
        
    def _compute_torques(self, actions):
        pos_action = (
            torch.cat(
                (
                    actions[:, 0:3], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                    actions[:, 4:7], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_pos
        )
        vel_action = (
            torch.cat(
                (
                    torch.zeros_like(actions[:, 0:3]), actions[:, 3].view(self.num_envs, 1),
                    torch.zeros_like(actions[:, 0:3]), actions[:, 7].view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_vel
        )
        # pd controller
        torques = self.p_gains * (pos_action + self.default_dof_pos - self.dof_pos) + self.d_gains * (vel_action - self.dof_vel)
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits )  # torque limit is lower than the torque-requiring lower bound
        return torques * self.torques_scale #notice that even send torque at torque limit , real motor may generate bigger torque that limit!!!!!!!!!!

    def post_physics_step(self):
        super().post_physics_step()
        self.wheel_lin_vel = self.foot_velocities[:, 0, :] + self.foot_velocities[:, 1, :]

    def get_observations(self):
        return (
            self.obs_buf,
            self.encoder_obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf
        )

    def compute_observations(self):
        # 1. Calls base_task's compute_observations() to populate obs_buf, add logic, update native obs_history
        super().compute_observations()
        
        # 2. Build filtered branch with a true 2nd-order Butterworth low-pass filter.
        raw_state = self.obs_buf[:, :self.filtered_size]
        filtered_state = raw_state.clone()

        x_now = raw_state[:, self.filter_feature_indices]
        y_now = (
            self.butter_b0 * x_now
            + self.butter_b1 * self.butter_x1
            + self.butter_b2 * self.butter_x2
            - self.butter_a1 * self.butter_y1
            - self.butter_a2 * self.butter_y2
        )

        self.butter_x2 = self.butter_x1
        self.butter_x1 = x_now
        self.butter_y2 = self.butter_y1
        self.butter_y1 = y_now

        filtered_state[:, self.filter_feature_indices] = y_now
        self.filtered_obs_buf = filtered_state
        
        # 3. Concatenate BOTH raw `obs_buf` and `filtered_obs_buf` at EACH timestep
        cat_obs = torch.cat([self.obs_buf, self.filtered_obs_buf], dim=-1)
        
        # 4. Update special `encoder_obs_history`
        self.encoder_obs_history = torch.cat(
            (self.encoder_obs_history[:, cat_obs.shape[1] :], cat_obs), dim=-1
        )

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        dof_list = [0,1,2,4,5,6]
        dof_pos = (self.dof_pos - self.default_dof_pos)[:,dof_list]
        # dof_pos = torch.remainder(dof_pos + self.pi, 2 * self.pi) - self.pi
        load_estimates = self._compute_load_estimates()
        load_estimation_obs = torch.stack(
            (
                load_estimates["payload_mass"] * self.obs_scales.load_mass,
                load_estimates["load_x"] * self.obs_scales.load_pos,
                load_estimates["load_y"] * self.obs_scales.load_pos,
                load_estimates["payload_present"],
                load_estimates["sin_lieangle_L_thigh"],
                load_estimates["cos_lieangle_L_thigh"],
                load_estimates["sin_lieangle_R_thigh"],
                load_estimates["cos_lieangle_R_thigh"],
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                dof_pos * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.torques * self.obs_scales.torque,
                load_estimation_obs,
                self.actions,
                # self.clock_inputs_sin.view(self.num_envs, 1),
                # self.clock_inputs_cos.view(self.num_envs, 1),
                # self.gaits,
            ),
            dim=-1,
        )
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            (self.base_mass - self.base_mass0).unsqueeze(-1) * self.obs_scales.mass_scale,     # Δmass
            (self.base_com - self.base_com0)[:, :3] * self.obs_scales.com_scale,            # Δcom(xy)
            # self.base_inertia * self.obs_scales.inertia_scale,     ##比例
            # priv_load_feat,
            self.obs_buf), dim=-1)
        return obs_buf, critic_obs_buf
    
    def _compute_load_estimates(self):
        """
        基于动力学估算环境内机器人的负载状态(质量和负载x/y坐标)。
        你可以将该函数的输出加入到 observation 中，以帮助策略网络更好地感知当前的负载情况。
        """
        def _dof_index(name: str, fallback: int):
            return self.dof_names.index(name) if name in self.dof_names else fallback

        hip_L_idx = _dof_index("hip_L_Joint", 1)
        hip_R_idx = _dof_index("hip_R_Joint", 5)
        knee_L_idx = _dof_index("knee_L_Joint", 2)
        knee_R_idx = _dof_index("knee_R_Joint", 6)

        power_lhip = self.torques[:, hip_L_idx]
        power_lknee = self.torques[:, knee_L_idx]
        power_rhip = self.torques[:, hip_R_idx]
        power_rknee = self.torques[:, knee_R_idx]

        theta_lhip = self.dof_pos[:, hip_L_idx]
        theta_rhip = self.dof_pos[:, hip_R_idx]

        # 这里用四元数直接恢复 pitch，避免依赖额外的 euler 状态缓存
        qx = self.base_quat[:, 0]
        qy = self.base_quat[:, 1]
        qz = self.base_quat[:, 2]
        qw = self.base_quat[:, 3]
        pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0 + 1e-6, 1.0 - 1e-6))

        zero_thigh_angle = float(getattr(self.cfg.asset, "load_estimation_zero_thigh_angle", 2.0 * math.pi / 3.0))
        thigh_len = float(getattr(self.cfg.asset, "load_estimation_thigh_length", 0.3))
        gravity_cfg = getattr(self.sim_params, "gravity", None)
        if gravity_cfg is None:
            gravity = 9.81
        else:
            gravity_z = getattr(gravity_cfg, "z", None)
            if gravity_z is None:
                gravity = 9.81
            else:
                gravity = float(abs(gravity_z))
        mass_offset = float(getattr(self.cfg.asset, "load_estimation_mass_offset", 9.585))
        robot_width = float(getattr(self.cfg.asset, "load_estimation_robot_width", 0.251))
        com_x_bias = float(getattr(self.cfg.asset, "load_estimation_com_x_bias", 0.2632))
        position_limit = float(getattr(self.cfg.asset, "load_estimation_position_limit", 0.5))
        position_zero_mass_threshold = float(
            getattr(self.cfg.asset, "load_estimation_position_zero_mass_threshold", 1.0)
        )

        # WF_TRON1A zero pose: thigh is 120 deg from +X in the sagittal plane.
        # Left hip axis is +Y, right hip axis is -Y, so the hip angle signs differ.
        lieangle_L_thigh = 3.14159 - (zero_thigh_angle - theta_lhip) - pitch   #zero_thigh_angle - theta_lhip应该是机体和髋夹角
        lieangle_R_thigh = 3.14159 - (zero_thigh_angle + theta_rhip) - pitch   #zero_thigh_angle + theta_lhip应该是机体和髋夹角

        sin_lieangle_L_thigh = torch.sin(lieangle_L_thigh)
        cos_lieangle_L_thigh = torch.cos(lieangle_L_thigh)
        sin_lieangle_R_thigh = torch.sin(lieangle_R_thigh)
        cos_lieangle_R_thigh = torch.cos(lieangle_R_thigh)

        cos_l = cos_lieangle_L_thigh
        cos_l = torch.where(
            torch.abs(cos_l) < 1e-3,
            torch.where(cos_l >= 0.0, torch.full_like(cos_l, 1e-3), torch.full_like(cos_l, -1e-3)),
            cos_l,
        )
        cos_r = cos_lieangle_R_thigh
        cos_r = torch.where(
            torch.abs(cos_r) < 1e-3,
            torch.where(cos_r >= 0.0, torch.full_like(cos_r, 1e-3), torch.full_like(cos_r, -1e-3)),
            cos_r,
        )
        sin_pitch = torch.sin(pitch)
        sin_pitch = torch.where(
            torch.abs(sin_pitch) < 1e-3,
            torch.where(sin_pitch >= 0.0, torch.full_like(sin_pitch, 1e-3), torch.full_like(sin_pitch, -1e-3)),
            sin_pitch,
        )
        tan_pitch = torch.tan(pitch)
        cos_pitch = torch.cos(pitch)
        load_torque_left = -power_lknee - power_lhip
        load_torque_right = power_rknee + power_rhip
        payload_mass_left = 0.5*(load_torque_left / ((thigh_len * cos_r+0.05144) * gravity )- mass_offset)
        payload_mass_right = 0.5*(load_torque_right / ((thigh_len * cos_r+0.05144) * gravity )- mass_offset)

        payload_mass = payload_mass_left + payload_mass_right  # mass_offset 是为了修正机体质量引入的一个经验值，实际使用时可以根据具体情况调整或通过校准获得。 000
        payload_mass_safe = torch.clamp(payload_mass, min=1e-6)


        load_y = 0.5 * robot_width * (payload_mass_left - payload_mass_right) / payload_mass_safe
        load_x = ((-power_rhip + power_lhip) / payload_mass_safe / gravity / cos_pitch) - com_x_bias * tan_pitch - 0.05144
        low_payload_mass = payload_mass_safe < position_zero_mass_threshold
        load_x = torch.where(low_payload_mass, torch.zeros_like(load_x), load_x)
        load_y = torch.where(low_payload_mass, torch.zeros_like(load_y), load_y)
        load_x = torch.clamp(load_x, -position_limit, position_limit)
        load_y = torch.clamp(load_y, -position_limit, position_limit)

        self.estimated_payload_mass_raw = payload_mass_safe
        self.estimated_load_y_raw = load_y
        self.estimated_load_x_raw = load_x
        payload_mass_safe, load_x, load_y = self._filter_load_estimation_outputs(
            payload_mass_safe, load_x, load_y
        )

        self.estimated_payload_mass = payload_mass_safe
        self.estimated_load_y = load_y
        self.estimated_load_x = load_x
        payload_present = (payload_mass_safe >= position_zero_mass_threshold).float()
        self.estimated_payload_present = payload_present
        self.load_estimation_sin_lieangle_L_thigh = sin_lieangle_L_thigh
        self.load_estimation_cos_lieangle_L_thigh = cos_lieangle_L_thigh
        self.load_estimation_sin_lieangle_R_thigh = sin_lieangle_R_thigh
        self.load_estimation_cos_lieangle_R_thigh = cos_lieangle_R_thigh

        load_estimates = {
            "payload_mass": payload_mass_safe,
            "load_x": load_x,
            "load_y": load_y,
            "payload_present": payload_present,
            "sin_lieangle_L_thigh": sin_lieangle_L_thigh,
            "cos_lieangle_L_thigh": cos_lieangle_L_thigh,
            "sin_lieangle_R_thigh": sin_lieangle_R_thigh,
            "cos_lieangle_R_thigh": cos_lieangle_R_thigh,
        }
        self.last_load_estimates = load_estimates
        return load_estimates

    def _filter_load_estimation_outputs(self, payload_mass, load_x, load_y):
        if not hasattr(self, "load_estimation_filter_x1"):
            return payload_mass, load_x, load_y

        x_now = torch.stack((payload_mass, load_x, load_y), dim=-1)
        y_now = (
            self.load_estimation_b0 * x_now
            + self.load_estimation_b1 * self.load_estimation_filter_x1
            - self.load_estimation_a1 * self.load_estimation_filter_y1
        )

        init_mask = ~self.load_estimation_filter_initialized
        if init_mask.any():
            y_now = torch.where(init_mask.unsqueeze(-1), x_now, y_now)

        self.load_estimation_filter_x1 = x_now
        self.load_estimation_filter_y1 = y_now
        self.load_estimation_filter_initialized[:] = True

        return y_now[:, 0], y_now[:, 1], y_now[:, 2]
    
    
    
    
    
    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        # self._resample_gaits(env_ids)
        # self._step_contact_targets()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 0.1 * wrap_to_pi(self.commands[:, 3] - heading)

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[self.actor_indices, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        #set 50% of resample to go straight
        resample_nums = len(env_ids)
        env_list = list(range(resample_nums))
        half_env_list = random.sample(env_list, resample_nums // 2)
        # forward = quat_apply(self.base_quat[env_ids[half_env_list]], \
        #                      self.forward_vec[env_ids[half_env_list]])
        # heading = torch.atan2(forward[:,1], forward[:,0])
        # self.commands[env_ids[half_env_list], 3] = heading
        
        # set 20% of the rest 50% to be stand still
        rest_env_list = list(set(env_list) - set(half_env_list))
        zero_cmd_env_idx_ = random.sample(rest_env_list, resample_nums // 2 // 5)

        self.commands[env_ids[zero_cmd_env_idx_], 0] = 0.0
        self.commands[env_ids[zero_cmd_env_idx_], 1] = 0.0
        self.commands[env_ids[zero_cmd_env_idx_], 2] = 0.0
        #use heading
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat[env_ids[zero_cmd_env_idx_]], \
                                 self.forward_vec[env_ids[zero_cmd_env_idx_]])
            heading = torch.atan2(forward[:,1], forward[:,0])
            self.commands[env_ids[zero_cmd_env_idx_], 3] = heading
            
    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12:20] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[20:28] = 0.0  # raw torques
        noise_vec[28:36] = 0.0  # load estimation features
        noise_vec[36:] = 0.0  # previous actions
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.wheel_lin_vel = torch.zeros_like(self.foot_velocities)
        self.wheel_ang_vel = torch.zeros_like(self.base_ang_vel)

        self.filtered_size = self.num_obs - self.cfg.env.num_actions
        self.filter_feature_indices = torch.arange(
            0, self.filtered_size, dtype=torch.long, device=self.device
        )
        # Keep dof_pos (6:12) unfiltered; torque uses the same filter path as other dynamic terms.
        dof_pos_start = 6
        dof_pos_end = 12
        self.filter_feature_indices = torch.cat(
            (
                self.filter_feature_indices[:dof_pos_start],
                self.filter_feature_indices[dof_pos_end:],
            ),
            dim=0,
        )

        cutoff_hz = float(getattr(self.cfg.env, "obs_butter_cutoff_hz", 10.0))
        nyquist_hz = 0.5 / self.dt
        cutoff_hz = max(1e-3, min(cutoff_hz, 0.99 * nyquist_hz))
        w = math.tan(math.pi * cutoff_hz * self.dt)
        w2 = w * w
        sqrt2 = math.sqrt(2.0)
        norm = 1.0 / (1.0 + sqrt2 * w + w2)
        self.butter_b0 = w2 * norm
        self.butter_b1 = 2.0 * self.butter_b0
        self.butter_b2 = self.butter_b0
        self.butter_a1 = 2.0 * (w2 - 1.0) * norm
        self.butter_a2 = (1.0 - sqrt2 * w + w2) * norm
        
        self.filtered_obs_buf = torch.zeros(
            self.num_envs, self.filtered_size, dtype=torch.float, device=self.device, requires_grad=False
        )
        n_filtered_features = self.filter_feature_indices.numel()
        self.butter_x1 = torch.zeros(
            self.num_envs, n_filtered_features, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.butter_x2 = torch.zeros(
            self.num_envs, n_filtered_features, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.butter_y1 = torch.zeros(
            self.num_envs, n_filtered_features, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.butter_y2 = torch.zeros(
            self.num_envs, n_filtered_features, dtype=torch.float, device=self.device, requires_grad=False
        )

        load_cutoff_normalized = float(
            getattr(self.cfg.asset, "load_estimation_filter_cutoff_normalized", 1.0 / 10.0)
        )
        load_cutoff_normalized = max(1e-6, min(load_cutoff_normalized, 0.999999))
        load_filter_w = math.tan(0.5 * math.pi * load_cutoff_normalized)
        load_filter_norm = 1.0 / (1.0 + load_filter_w)
        self.load_estimation_b0 = load_filter_w * load_filter_norm
        self.load_estimation_b1 = self.load_estimation_b0
        self.load_estimation_a1 = (load_filter_w - 1.0) * load_filter_norm
        self.load_estimation_filter_x1 = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.load_estimation_filter_y1 = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.load_estimation_filter_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.encoder_obs_history = torch.zeros(
            self.num_envs,
            (self.num_obs + self.filtered_size) * self.obs_history_length,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    # ------------ reward functions----------------

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_nominal_foot_position(self):
        #1. calculate foot postion wrt base in base frame  
        nominal_base_height = -(self.cfg.rewards.base_height_target- self.cfg.asset.foot_radius)
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        reward = 0
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
            height_error = nominal_base_height - foot_positions_base[:, i, 2]
            reward += torch.exp(-(height_error ** 2)/ self.cfg.rewards.nominal_foot_position_tracking_sigma)
        vel_cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        return reward / len(self.feet_indices)*torch.exp(-(vel_cmd_norm ** 2)/self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)
    
    def _reward_same_foot_z_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_z_position_err = foot_positions_base[:,0,2] - foot_positions_base[:,1,2]
        return foot_z_position_err ** 2

    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        return torch.exp(-(leg_symmetry_err ** 2)/ self.cfg.rewards.leg_symmetry_tracking_sigma)

    def _reward_same_foot_x_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        reward = torch.abs(foot_x_position_err)
        return reward

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_tracking_ang_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[self.actor_indices, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.abs(base_height - self.cfg.rewards.base_height_target)
