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
        self.task_num_commands = self.num_commands
        self.policy_hardware_context_dim = (
            6
            if bool(
                getattr(
                    self.cfg.env,
                    "use_hardware_context_in_policy",
                    False,
                )
            )
            else 0
        )
        self.num_commands += self.policy_hardware_context_dim
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
        self._update_scenario_curriculum(env_ids)   # co-design: per-env scenario difficulty (no-op if off)
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        if getattr(self.cfg.env, "use_gait_phase", False):
            self._resample_gaits(env_ids)
            self.gait_indices[env_ids] = 0.0
        self.remove_load(env_ids)
        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.root_states[self.actor_indices[env_ids], :3]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.foot_stance_steps[env_ids] = 0.0
        if getattr(self.cfg.env, "use_jump", False):
            self.cmd_jump[env_ids] = 0.0
            self.jump_height_cmd[env_ids] = 0.0
            self.jump_state[env_ids] = 0
            self.jump_trigger_time[env_ids] = 0.0
            self.jump_phase_time[env_ids] = 0.0
            self.jump_peak_height[env_ids] = 0.0
            self.jump_prev_vz[env_ids] = 0.0
            self.jumped_flag[env_ids] = False
            self.jump_apex_fire[env_ids] = False
            self.jump_landed_once[env_ids] = False
            self.jump_air_steps[env_ids] = 0.0
            self.jump_takeoff_fire[env_ids] = False
            self.jump_init_base_height[env_ids] = 0.0
            self.jump_min_base_height[env_ids] = 0.0
            self.jump_init_leg_len[env_ids] = 0.0
            self.jump_prev_leg_len[env_ids] = 0.0
            self.jump_min_leg_len[env_ids] = 0.0
            self.jump_takeoff_leg_len[env_ids] = 0.0
            self.jump_takeoff_leg_gain[env_ids] = 0.0
            self.jump_takeoff_height_gain[env_ids] = 0.0
            self.jump_takeoff_leg_vel[env_ids] = 0.0
            self.jump_takeoff_crouch_time[env_ids] = 0.0
            self.jump_takeoff_quality[env_ids] = 0.0
            self.jump_bad_takeoff_fire[env_ids] = False
            self.jump_timeout_fire[env_ids] = False
            self.jump_takeoff_roll[env_ids] = 0.0
            self.jump_takeoff_lat_ratio[env_ids] = 0.0
            self.jump_takeoff_fwd_vel[env_ids] = 0.0
            self.jump_takeoff_lat_vel[env_ids] = 0.0
            self.jump_takeoff_vz[env_ids] = 0.0
            self.jump_takeoff_base_vz[env_ids] = 0.0
            self.jump_takeoff_angle[env_ids] = 0.0
            self.jump_takeoff_leg_diff[env_ids] = 0.0
            self.jump_takeoff_leg_left[env_ids] = 0.0
            self.jump_takeoff_leg_right[env_ids] = 0.0
            self.jump_takeoff_foot_x_diff[env_ids] = 0.0
            self.jump_takeoff_foot_z_diff[env_ids] = 0.0
            self.jump_takeoff_abad_mirror_err[env_ids] = 0.0
            self.jump_takeoff_abad_left[env_ids] = 0.0
            self.jump_takeoff_abad_right[env_ids] = 0.0
            self.jump_takeoff_hip_left[env_ids] = 0.0
            self.jump_takeoff_hip_right[env_ids] = 0.0
            self.jump_takeoff_knee_left[env_ids] = 0.0
            self.jump_takeoff_knee_right[env_ids] = 0.0
            self.jump_takeoff_yaw_rate[env_ids] = 0.0
            self.jump_max_roll[env_ids] = 0.0
            self.jump_max_lat_ratio[env_ids] = 0.0
            self.jump_max_yaw_rate[env_ids] = 0.0
            self.jump_max_lat_vel[env_ids] = 0.0
            self.jump_min_flight_leg_len[env_ids] = 0.0
            self.jump_preland_leg_len[env_ids] = 0.0
            self.jump_land_leg_len[env_ids] = 0.0
            if getattr(self.cfg.env, "use_jump_distance", False):
                self.jump_dist_cmd[env_ids] = 0.0
                self.jump_start_pos[env_ids] = 0.0
                self.jump_takeoff_pos[env_ids] = 0.0
                self.jump_takeoff_dir[env_ids] = 0.0
                self.jump_land_disp[env_ids] = 0.0
                self.jump_land_lateral_disp[env_ids] = 0.0
                self.jump_land_fire[env_ids] = False
                self.jump_land_rewarded[env_ids] = False
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
        encoder_parts = [obs_buf[env_ids], self.filtered_obs_buf[env_ids]]
        if bool(getattr(self.cfg.env, "encoder_include_torques", False)):
            encoder_parts.append(
                self.torques[env_ids] * self.obs_scales.torque
            )
        cat_obs = torch.cat(encoder_parts, dim=-1)
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
        if int(getattr(self.cfg.control, "jump_residual_warmup_steps", 0)) > 0:
            self.jump_residual_step_counter = getattr(self, "jump_residual_step_counter", 0) + 1
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
            self._apply_sustained_ext_force()  # 实验A：持续外力（默认关闭，不影响训练）
            self._apply_scenario_forces()      # co-design 场景外力（坡=水平力/负载=向下力，默认关闭）
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
            self._get_policy_commands(),
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
        project_all_legs = getattr(self.cfg.control, "jump_project_leg_symmetry", False)
        if project_all_legs or getattr(self.cfg.control, "jump_project_abad_symmetry", False):
            strength = float(getattr(self.cfg.control, "jump_abad_projection_strength", 1.0))
            projected_dofs = 3 if project_all_legs else 1
            mirror_common = 0.5 * (
                pos_action[:, :projected_dofs] + pos_action[:, 4 : 4 + projected_dofs]
            )
            pos_action[:, :projected_dofs] -= strength * mirror_common
            pos_action[:, 4 : 4 + projected_dofs] -= strength * mirror_common
        if getattr(self.cfg.control, "jump_abad_recenter", False) and getattr(self.cfg.env, "use_jump", False):
            # Remove only the common abad mode while grounded. The antisymmetric mode
            # remains available for roll control, and flight is left untouched.
            on_ground = (self.contact_forces[:, self.feet_indices, 2] > 1.0).any(dim=1)
            grounded_phase = (self.jump_state != 2) & on_ground
            gain = float(getattr(self.cfg.control, "jump_abad_recenter_gain", 0.25))
            gain_t = torch.full_like(self.dof_pos[:, 0], gain)
            gain_t = torch.where(self.jump_state == 1, gain_t * 0.15, gain_t)
            common_pos = 0.5 * (self.dof_pos[:, 0] + self.dof_pos[:, 4])
            correction = -gain_t * common_pos * grounded_phase.float()
            pos_action[:, 0] += correction
            pos_action[:, 4] += correction
        hip_vel_delta = None
        if getattr(self.cfg.env, "use_jump_distance", False):
            knee_gain = float(getattr(self.cfg.control, "jump_knee_height_residual", 0.0))
            knee_bias = float(getattr(self.cfg.control, "jump_knee_height_bias", 0.0))
            hip_gain = float(getattr(self.cfg.control, "jump_hip_distance_residual", 0.0))
            hip_bias = float(getattr(self.cfg.control, "jump_hip_distance_bias", 0.0))
            hip_vel_gain = float(getattr(self.cfg.control, "jump_hip_distance_vel_residual", 0.0))
            hip_vel_bias = float(getattr(self.cfg.control, "jump_hip_distance_vel_bias", 0.0))
            hip_land_gain = float(getattr(self.cfg.control, "jump_hip_landing_residual", 0.0))
            hip_land_bias = float(getattr(self.cfg.control, "jump_hip_landing_bias", 0.0))
            warmup_steps = int(getattr(self.cfg.control, "jump_residual_warmup_steps", 0))
            if warmup_steps > 0:
                blend = min(float(getattr(self, "jump_residual_step_counter", 0)) / warmup_steps, 1.0)
                knee_gain_base = float(getattr(self.cfg.control, "jump_knee_height_residual_base", knee_gain))
                knee_bias_base = float(getattr(self.cfg.control, "jump_knee_height_bias_base", knee_bias))
                hip_gain_base = float(getattr(self.cfg.control, "jump_hip_distance_residual_base", hip_gain))
                hip_bias_base = float(getattr(self.cfg.control, "jump_hip_distance_bias_base", hip_bias))
                hip_land_gain_base = float(getattr(self.cfg.control, "jump_hip_landing_residual_base", hip_land_gain))
                hip_land_bias_base = float(getattr(self.cfg.control, "jump_hip_landing_bias_base", hip_land_bias))
                knee_gain = knee_gain_base + blend * (knee_gain - knee_gain_base)
                knee_bias = knee_bias_base + blend * (knee_bias - knee_bias_base)
                hip_gain = hip_gain_base + blend * (hip_gain - hip_gain_base)
                hip_bias = hip_bias_base + blend * (hip_bias - hip_bias_base)
                hip_land_gain = hip_land_gain_base + blend * (hip_land_gain - hip_land_gain_base)
                hip_land_bias = hip_land_bias_base + blend * (hip_land_bias - hip_land_bias_base)
            if any((knee_gain, knee_bias, hip_gain, hip_bias, hip_vel_gain, hip_vel_bias, hip_land_gain, hip_land_bias)):
                push_start = float(getattr(self.cfg.rewards, "jump_push_start_s", 0.12))
                phase_t = self._jump_phase_elapsed()
                push_gate = ((self.jump_state == 1) & (phase_t >= push_start)).float()
                profile = getattr(self.cfg.control, "jump_residual_profile", "step")
                hip_shape = push_gate
                knee_shape = push_gate
                if profile in ("ramp", "pulse"):
                    def smooth_ramp(start, end):
                        u = ((phase_t - start) / max(end - start, 1e-6)).clamp(0.0, 1.0)
                        return u * u * (3.0 - 2.0 * u)

                    hip_ramp_end = float(getattr(self.cfg.control, "jump_hip_ramp_end_s", 0.30))
                    knee_ramp_start = float(getattr(self.cfg.control, "jump_knee_ramp_start_s", 0.18))
                    knee_ramp_end = float(getattr(self.cfg.control, "jump_knee_ramp_end_s", 0.42))
                    hip_shape = push_gate * smooth_ramp(push_start, hip_ramp_end)
                    knee_shape = push_gate * smooth_ramp(knee_ramp_start, knee_ramp_end)
                    if profile == "pulse":
                        release_start = float(getattr(self.cfg.control, "jump_hip_release_start_s", 0.45))
                        release_end = float(getattr(self.cfg.control, "jump_hip_release_end_s", 0.65))
                        hip_shape *= 1.0 - smooth_ramp(release_start, release_end)
                h_ref = getattr(self.cfg.control, "jump_residual_height_ref_range", None)
                d_ref = getattr(self.cfg.control, "jump_residual_distance_ref_range", None)
                lo_h, hi_h = h_ref or self.cfg.commands.jump_height_cmd_range
                lo_d, hi_d = d_ref or self.cfg.commands.jump_dist_cmd_range
                h_half = max(0.5 * (hi_h - lo_h), 1e-6)
                d_half = max(0.5 * (hi_d - lo_d), 1e-6)
                signal_clip = float(getattr(self.cfg.control, "jump_residual_signal_clip", 1.0))
                h_signal = ((self.jump_height_cmd[:, 0] - 0.5 * (lo_h + hi_h)) / h_half).clamp(-signal_clip, signal_clip)
                d_signal = ((self.jump_dist_cmd[:, 0] - 0.5 * (lo_d + hi_d)) / d_half).clamp(-signal_clip, signal_clip)
                knee_delta = (knee_gain * h_signal + knee_bias) * knee_shape
                hip_delta = (hip_gain * d_signal + hip_bias) * hip_shape
                hip_vel_delta = (hip_vel_gain * d_signal + hip_vel_bias) * hip_shape
                descending = ((self.jump_state == 2) & self.jumped_flag).float()
                hip_land_delta = (hip_land_gain * d_signal + hip_land_bias) * descending
                # Positive knee angle bends the left leg; high h* therefore subtracts
                # on the left and adds on the mirrored right. Hip signs are mirrored too.
                pos_action[:, 2] -= knee_delta
                pos_action[:, 6] += knee_delta
                pos_action[:, 1] += hip_delta
                pos_action[:, 5] -= hip_delta
                pos_action[:, 1] += hip_land_delta
                pos_action[:, 5] -= hip_land_delta
        if getattr(self.cfg.control, "jump_use_explicit_trajectory", False):
            pos_action = self._apply_jump_explicit_trajectory(pos_action)
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
        if hip_vel_delta is not None:
            vel_action[:, 1] += hip_vel_delta
            vel_action[:, 5] -= hip_vel_delta
        # pd controller
        torques = self.p_gains * (pos_action + self.default_dof_pos - self.dof_pos) + self.d_gains * (vel_action - self.dof_vel)
        _lim = self._motor_available_torque_limits()
        torques = torch.clip(torques, -_lim, _lim)  # torque limit is lower than the torque-requiring lower bound
        return torques * self.torques_scale #notice that even send torque at torque limit , real motor may generate bigger torque that limit!!!!!!!!!!

    def _motor_available_torque_limits(self):
        """Return per-env torque limits under the optional power envelope."""
        peak_limits = self.motor_peak_torque_limits
        if not self.motor_power_envelope_enabled:
            self.motor_effective_torque_limits = peak_limits
            self.motor_power_limited_mask.zero_()
            return peak_limits

        power_limited = self.motor_power_limits_w / torch.clamp(
            torch.abs(self.dof_vel),
            min=self.motor_power_min_speed_rad_s,
        )
        effective_limits = torch.minimum(peak_limits, power_limited)
        self.motor_effective_torque_limits = effective_limits
        self.motor_power_limited_mask = power_limited < (
            peak_limits * (1.0 - 1.0e-6)
        )
        return effective_limits

    def post_physics_step(self):
        super().post_physics_step()
        self.wheel_lin_vel = self.foot_velocities[:, 0, :] + self.foot_velocities[:, 1, :]

    def _get_policy_commands(self):
        policy_commands = self.commands[:, :3] * self.commands_scale
        if self.policy_hardware_context_dim == 0:
            return policy_commands
        if (
            getattr(self, "env_morphology", None) is None
            or getattr(self, "env_motor_group_scale", None) is None
        ):
            raise RuntimeError(
                "hardware-conditioned policy requires morphology and motor-design tensors"
            )
        morph_scale = float(
            getattr(self.cfg.env, "hardware_context_morph_scale", 5.0)
        )
        motor_scale = float(
            getattr(self.cfg.env, "hardware_context_motor_scale", 2.0)
        )
        hardware_context = torch.cat(
            (
                (self.env_morphology - 1.0) * morph_scale,
                (self.env_motor_group_scale - 1.0) * motor_scale,
            ),
            dim=-1,
        )
        return torch.cat((policy_commands, hardware_context), dim=-1)

    def get_observations(self):
        return (
            self.obs_buf,
            self.encoder_obs_history,
            self._get_policy_commands(),
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
        encoder_parts = [self.obs_buf, self.filtered_obs_buf]
        if bool(getattr(self.cfg.env, "encoder_include_torques", False)):
            encoder_parts.append(self.torques * self.obs_scales.torque)
        cat_obs = torch.cat(encoder_parts, dim=-1)
        
        # 4. Update special `encoder_obs_history`
        self.encoder_obs_history = torch.cat(
            (self.encoder_obs_history[:, cat_obs.shape[1] :], cat_obs), dim=-1
        )

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        dof_list = [0,1,2,4,5,6]
        dof_pos = (self.dof_pos - self.default_dof_pos)[:,dof_list]
        # dof_pos = torch.remainder(dof_pos + self.pi, 2 * self.pi) - self.pi
        # 总是计算 load_estimates（play 时要 log 给 Model C RMSE 用；env 内部 base_com / base_mass
        # 的真值也来自 sim，跟这个无关）。但是否把它喂进 obs 由 cfg.env.use_qs_in_obs 控制。
        load_estimates = self._compute_load_estimates()
        # 第5章 测试时估计消融(仅 play 用;默认 'none' 对训练完全无影响)。
        # 只篡改"喂进 obs 的负载估计",策略权重一字不动;encoder 读 obs 历史也会看到被改的值。
        _cm = getattr(self, "qs_corrupt_mode", "none")
        if _cm != "none":
            _cv = float(getattr(self, "qs_corrupt_val", 0.0))
            if _cm in ("zero", "zero_all"):  # 估计=无负载
                for _k in ("payload_mass", "load_x", "load_y", "payload_present", "qs_mass_delta"):
                    if _k in load_estimates: load_estimates[_k] = torch.zeros_like(load_estimates[_k])
                if "qs_com_delta" in load_estimates: load_estimates["qs_com_delta"] = torch.zeros_like(load_estimates["qs_com_delta"])
                if _cm == "zero_all":
                    # Strict QS-channel ablation: also remove the analytic angle features
                    # that are appended together with the load estimate.
                    for _k in ("sin_lieangle_L_thigh", "cos_lieangle_L_thigh",
                               "sin_lieangle_R_thigh", "cos_lieangle_R_thigh"):
                        if _k in load_estimates:
                            load_estimates[_k] = torch.zeros_like(load_estimates[_k])
            elif _cm == "scale":        # 质量估计 ×_cv(0.5 低估 / 2.0 高估)
                load_estimates["payload_mass"] = load_estimates["payload_mass"] * _cv
                if "qs_mass_delta" in load_estimates: load_estimates["qs_mass_delta"] = load_estimates["qs_mass_delta"] * _cv
            elif _cm == "fixed":        # 质量估计恒为 _cv kg(无视真实负载)
                load_estimates["payload_mass"] = torch.full_like(load_estimates["payload_mass"], _cv)
                if "qs_mass_delta" in load_estimates: load_estimates["qs_mass_delta"] = torch.zeros_like(load_estimates["qs_mass_delta"])
            elif _cm == "noise":        # 质量估计 + N(0,_cv)
                load_estimates["payload_mass"] = load_estimates["payload_mass"] + _cv * torch.randn_like(load_estimates["payload_mass"])
        use_qs_in_obs = bool(getattr(self.cfg.env, "use_qs_in_obs", True))
        use_torques_in_obs = bool(getattr(self.cfg.env, "use_torques_in_obs", True))

        # 基础 obs：跟 use_qs_in_obs / use_torques_in_obs 无关的部分
        obs_components = [
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
        ]
        # Ablation: raw joint torques (default on; off = "true history-only" baseline,
        # encoder 失去对 QS 公式底层信号的访问，用于验证 QS-derivable signal 是否必要)
        if use_torques_in_obs:
            obs_components.append(self.torques * self.obs_scales.torque)

        # QS 部分（only when ablation flag on）
        if use_qs_in_obs:
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
            load_residual_baseline_obs = torch.cat(
                (
                    load_estimates["qs_mass_delta"].unsqueeze(-1) * self.obs_scales.mass_scale,
                    load_estimates["qs_com_delta"] * self.obs_scales.com_scale,
                ),
                dim=-1,
            )
            obs_components.append(load_estimation_obs)
            obs_components.append(load_residual_baseline_obs)

        obs_components.append(self.actions)
        # Stepping mode: append sin/cos gait clock so the policy can phase-lock its gait.
        if getattr(self.cfg.env, "use_gait_stepping", False):
            obs_components.append(self.cmd_step)   # commanded stepping-velocity target
        if getattr(self.cfg.env, "use_gait_phase", False):
            obs_components.append(self.clock_inputs_sin.reshape(self.num_envs, 1))
            obs_components.append(self.clock_inputs_cos.reshape(self.num_envs, 1))
        if getattr(self.cfg.env, "use_jump", False):
            jump_obs_scale = float(getattr(self.cfg.commands, "jump_obs_scale", 2.0))
            obs_components.append(self.cmd_jump)   # command-triggered jump signal (jump task)
            obs_components.append(self.jump_height_cmd * jump_obs_scale)
            if getattr(self.cfg.env, "use_jump_distance", False):
                obs_components.append(self.jump_dist_cmd * jump_obs_scale)
        obs_buf = torch.cat(obs_components, dim=-1)
        # Critic privileged prefix: lin_vel(0:3), Δmass(3), Δcom(4:7) — these slots are read by the
        # encoder's vel/mass/com supervision heads at fixed offsets, so nothing may be inserted before them.
        critic_components = [
            self.base_lin_vel * self.obs_scales.lin_vel,
            (self.base_mass - self.base_mass0).unsqueeze(-1) * self.obs_scales.mass_scale,     # Δmass
            (self.base_com - self.base_com0)[:, :3] * self.obs_scales.com_scale,            # Δcom(xyz)
        ]
        # Co-design: append per-env leg morphology (thigh,shank scale) AFTER index 7, as privileged
        # info for the critic. Centered at 0 (xi-1) so nominal morphology contributes nothing.
        if bool(getattr(self.cfg.env, "use_morphology_in_critic", False)) and getattr(self, "env_morphology", None) is not None:
            morph_scale = float(getattr(self.obs_scales, "morph_scale", 5.0))
            critic_components.append((self.env_morphology - 1.0) * morph_scale)
        # Co-design Path A: append per-env motor design scale (1 dim) as privileged critic info, after index 7.
        if bool(getattr(self.cfg.env, "use_motor_design_in_critic", False)) and getattr(self, "env_motor_scale", None) is not None:
            motor_scale = float(getattr(self.obs_scales, "motor_scale", 2.0))
            critic_components.append(((self.env_motor_scale - 1.0) * motor_scale).view(-1, 1))
        critic_components.append(self.obs_buf)
        critic_obs_buf = torch.cat(critic_components, dim=-1)
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
        abad_L_idx = _dof_index("abad_L_Joint", 0)
        abad_R_idx = _dof_index("abad_R_Joint", 4)

        power_lhip = self.torques[:, hip_L_idx]
        power_lknee = self.torques[:, knee_L_idx]
        power_rhip = self.torques[:, hip_R_idx]
        power_rknee = self.torques[:, knee_R_idx]

        theta_lhip = self.dof_pos[:, hip_L_idx]
        theta_rhip = self.dof_pos[:, hip_R_idx]
        theta_labad = self.dof_pos[:, abad_L_idx]
        theta_rabad = self.dof_pos[:, abad_R_idx]

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
        mass_offset = float(getattr(self.cfg.asset, "load_estimation_mass_offset", 9.585))  # 旧公式用
        robot_width = float(getattr(self.cfg.asset, "load_estimation_robot_width", 0.251))
        com_x_bias = float(getattr(self.cfg.asset, "load_estimation_com_x_bias", 0.0784))
        # Model G 标定参数（universal G_all, avg over main_lb3_s42 + s43 + direct）
        # m_leg = alpha * R + gamma + beta_hip*(theta_lhip-theta_rhip)
        #                          + beta_pitch*sin(pitch)
        #                          + beta_abad*(theta_labad-theta_rabad)
        # 比 Model C (5 params) avg RMSE 改善 14%; universal 系数对 main_lb3 + direct cluster 通用
        # main_lb6 是 outlier，那种 ablation 需要单独 cal
        alpha_L  = float(getattr(self.cfg.asset, "load_estimation_alpha_L",  0.4154))
        alpha_R  = float(getattr(self.cfg.asset, "load_estimation_alpha_R",  0.4235))
        gamma_L  = float(getattr(self.cfg.asset, "load_estimation_gamma_L", -0.7638))
        gamma_R  = float(getattr(self.cfg.asset, "load_estimation_gamma_R", -0.8474))
        beta_pitch = float(getattr(self.cfg.asset, "load_estimation_beta_pitch", -2.1976))
        beta_hip   = float(getattr(self.cfg.asset, "load_estimation_beta_hip",   -9.1708))
        beta_abad  = float(getattr(self.cfg.asset, "load_estimation_beta_abad",  +3.5004))
        x_offset = float(getattr(self.cfg.asset, "load_estimation_x_offset", -0.0487))
        position_limit = float(getattr(self.cfg.asset, "load_estimation_position_limit", 0.5))
        position_zero_mass_threshold = float(
            getattr(self.cfg.asset, "load_estimation_position_zero_mass_threshold", 1.0)
        )
        # abad-related geometry: effective vertical drop from abad axis to wheel-ground contact
        # at zero pose; used to convert abad joint angle into lateral foot shift.
        leg_eff_length = float(getattr(self.cfg.asset, "load_estimation_leg_eff_length", 0.55))
        # Sign of right-leg abad axis relative to left-leg's. If URDF gives both legs same +X
        # axis, set to +1; if opposite axes (typical, mirrors hip convention), set to -1.
        abad_R_sign = float(getattr(self.cfg.asset, "load_estimation_abad_R_sign", -1.0))

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

        # abad correction factor 1: hip/knee torque axes tilt by abad angle, so a vertical
        # foot force F_w produces tau_hip = x_foot * F_w * cos(theta_abad). To back out F_w
        # from measured torque we therefore divide by cos(theta_abad).
        cos_abad_L = torch.cos(theta_labad)
        cos_abad_R = torch.cos(theta_rabad)
        # avoid div-by-zero near +-pi/2
        cos_abad_L_safe = torch.where(
            torch.abs(cos_abad_L) < 1e-3,
            torch.where(cos_abad_L >= 0.0, torch.full_like(cos_abad_L, 1e-3), torch.full_like(cos_abad_L, -1e-3)),
            cos_abad_L,
        )
        cos_abad_R_safe = torch.where(
            torch.abs(cos_abad_R) < 1e-3,
            torch.where(cos_abad_R >= 0.0, torch.full_like(cos_abad_R, 1e-3), torch.full_like(cos_abad_R, -1e-3)),
            cos_abad_R,
        )

        load_torque_left = -power_lknee - power_lhip
        load_torque_right = power_rknee + power_rhip

        # ===== 旧公式 (Model A, 在 (0,0,5kg) 处手调，远端偏差大) =====
        # payload_mass_left  = 0.75*(load_torque_left  / ((thigh_len * cos_l + 0.05144) * gravity * cos_abad_L_safe) - mass_offset) - 2.5
        # payload_mass_right = 0.75*(load_torque_right / ((thigh_len * cos_r + 0.05144) * gravity * cos_abad_R_safe) - mass_offset) + 0.65
        # ============================================================

        # Model G: 在 Model C 基础上加 sin(pitch) 和 abad_diff 两个对称物理修正项。
        # universal G_all RMSE avg 0.71 kg（main_lb3 + direct cluster, lb=6 outlier 单独处理）。
        hip_diff = theta_lhip - theta_rhip
        abad_diff = theta_labad - theta_rabad
        mass_moment_arm_L = thigh_len * cos_l + 0.05144
        mass_moment_arm_R = thigh_len * cos_r + 0.05144
        mass_denominator_L = mass_moment_arm_L * gravity * cos_abad_L_safe
        mass_denominator_R = mass_moment_arm_R * gravity * cos_abad_R_safe
        R_L = load_torque_left / mass_denominator_L
        R_R = load_torque_right / mass_denominator_R
        payload_mass_left  = (alpha_L * R_L + gamma_L
                              + beta_hip * hip_diff
                              + beta_pitch * sin_pitch
                              + beta_abad * abad_diff)
        payload_mass_right = (alpha_R * R_R + gamma_R
                              + beta_hip * hip_diff
                              + beta_pitch * sin_pitch
                              + beta_abad * abad_diff)

        payload_mass = payload_mass_left + payload_mass_right
        # payload_mass_safe = torch.clamp(payload_mass, min=1e-6)
        payload_mass_safe = payload_mass

        # abad correction factor 2: each foot's actual lateral position in body frame is
        # shifted by leg_eff_length * sin(theta_abad). Asymmetric abad angles thus produce
        # a y bias if we keep using +-W/2 as the foot positions.
        y_foot_L = 0.5 * robot_width + leg_eff_length * torch.sin(theta_labad)
        y_foot_R = -0.5 * robot_width + abad_R_sign * leg_eff_length * torch.sin(theta_rabad)

        # Moment balance about body X-axis (roll), using actual foot y-positions:
        #   y_L * m_L = y_foot_L * m_left + y_foot_R * m_right
        payload_mass_safe_for_div = torch.where(
            torch.abs(payload_mass_safe) < 1e-3,
            torch.where(payload_mass_safe >= 0.0, torch.full_like(payload_mass_safe, 1e-3), torch.full_like(payload_mass_safe, -1e-3)),
            payload_mass_safe,
        )
        load_y = (y_foot_L * payload_mass_left + y_foot_R * payload_mass_right) / payload_mass_safe_for_div

        # ===== 旧 load_x 公式（手调 T_body_x=14, +0.12 偏置）=====
        # T_body_x_old = 14.0
        # load_x = ((-power_rhip + power_lhip - T_body_x_old) / payload_mass_safe / gravity / cos_pitch) - 0.2632 * tan_pitch + 0.12
        # 更早版本：
        # load_x = ((-power_rhip + power_lhip) / payload_mass_safe / gravity / cos_pitch) - com_x_bias * tan_pitch - 0.23
        # =========================================================

        # Model C：T_body_x=6.17, com_x_bias=0.649, x_offset=-0.037
        T_body_x = float(getattr(self.cfg.asset, "load_estimation_t_body_x", 6.3447))
        load_x = ((-power_rhip + power_lhip - T_body_x) / payload_mass_safe / gravity / cos_pitch) - com_x_bias * tan_pitch + x_offset
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
        body_mass = float(getattr(self.cfg.asset, "load_estimation_body_mass", 9.58))
        body_com0_cfg = getattr(
            self.cfg.asset,
            "load_estimation_body_com0",
            [0.0, 0.0, 0.0],
        )
        load_z = float(getattr(self.cfg.asset, "load_estimation_load_z", 0.10))
        qs_mass_delta = torch.where(
            payload_present > 0.0,
            payload_mass_safe,
            torch.zeros_like(payload_mass_safe),
        )
        load_pos = torch.stack(
            (load_x, load_y, torch.full_like(load_x, load_z)),
            dim=-1,
        )
        body_com0 = torch.as_tensor(
            body_com0_cfg,
            device=self.device,
            dtype=load_pos.dtype,
        ).view(1, 3)
        qs_total_mass = body_mass + qs_mass_delta
        qs_com = (
            body_mass * body_com0 + qs_mass_delta.unsqueeze(-1) * load_pos
        ) / qs_total_mass.clamp_min(1e-6).unsqueeze(-1)
        qs_com_delta = qs_com - body_com0
        self.estimated_payload_present = payload_present
        self.load_estimation_sin_lieangle_L_thigh = sin_lieangle_L_thigh
        self.load_estimation_cos_lieangle_L_thigh = cos_lieangle_L_thigh
        self.load_estimation_sin_lieangle_R_thigh = sin_lieangle_R_thigh
        self.load_estimation_cos_lieangle_R_thigh = cos_lieangle_R_thigh
        self.load_estimation_qs_mass_delta = qs_mass_delta
        self.load_estimation_qs_com_delta = qs_com_delta

        load_estimates = {
            "payload_mass": payload_mass_safe,
            "payload_mass_left": payload_mass_left,
            "payload_mass_right": payload_mass_right,
            # Diagnostics only: these expose the analytical inversion without
            # changing any observation or estimator behavior.
            "payload_mass_raw": self.estimated_payload_mass_raw,
            "mass_moment_arm_left": mass_moment_arm_L,
            "mass_moment_arm_right": mass_moment_arm_R,
            "mass_denominator_left": mass_denominator_L,
            "mass_denominator_right": mass_denominator_R,
            "load_torque_left": load_torque_left,
            "load_torque_right": load_torque_right,
            "load_x": load_x,
            "load_y": load_y,
            "payload_present": payload_present,
            "sin_lieangle_L_thigh": sin_lieangle_L_thigh,
            "cos_lieangle_L_thigh": cos_lieangle_L_thigh,
            "sin_lieangle_R_thigh": sin_lieangle_R_thigh,
            "cos_lieangle_R_thigh": cos_lieangle_R_thigh,
            "qs_mass_delta": qs_mass_delta,
            "qs_com_delta": qs_com_delta,
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
    
    
    
    
    
    # ==================== SCENARIO PARTITION (co-design multi-scenario training) ====================
    # Each env gets ONE primary scenario (0=obstacle 1=slope 2=load 3=accel) + a per-env curriculum
    # level. Realized via: obstacle=high foot-clearance target + commanded stepping; slope=per-env
    # horizontal down-slope force + tilted gravity obs; load=per-env downward force (force~weight, Exp A);
    # accel=high forward-speed command. Difficulty ramps per-env on survival. cfg.scenario.partition=False
    # -> every method here is a no-op (existing single-condition behavior preserved).
    def _init_scenarios_if_needed(self):
        if hasattr(self, "scenario_id"):
            return
        sc = getattr(self.cfg, "scenario", None)
        if sc is None or not getattr(sc, "partition", False):
            self.scenario_id = None
            return
        if bool(getattr(sc, "physical_partition", False)):
            levels_per_scenario = int(sc.physical_levels_per_scenario)
            physical_task = str(
                getattr(sc, "physical_task", "all")
            ).strip().lower()
            task_to_scenario = {
                "clearance": 0,
                "slope": 1,
                "payload": 2,
                "speed": 3,
            }
            if physical_task == "all":
                expected_rows = 4 * levels_per_scenario
            elif physical_task in task_to_scenario:
                expected_rows = levels_per_scenario
            else:
                raise ValueError(
                    "physical_task must be all, clearance, slope, payload, "
                    f"or speed; got {physical_task!r}"
                )
            if self.cfg.terrain.num_rows != expected_rows:
                raise ValueError(
                    "physical scenario partition requires terrain.num_rows="
                    f"{expected_rows}, got {self.cfg.terrain.num_rows}"
                )
            if not hasattr(self, "terrain_levels"):
                raise ValueError(
                    "physical scenario partition requires deterministic terrain levels"
                )
            if physical_task == "all":
                self.scenario_id = torch.div(
                    self.terrain_levels,
                    levels_per_scenario,
                    rounding_mode="floor",
                ).clamp(max=3)
                self.physical_level_id = torch.remainder(
                    self.terrain_levels, levels_per_scenario
                )
            else:
                self.scenario_id = torch.full_like(
                    self.terrain_levels,
                    task_to_scenario[physical_task],
                )
                self.physical_level_id = self.terrain_levels.clone()
            if levels_per_scenario > 1:
                self.scenario_level = (
                    self.physical_level_id.float()
                    / float(levels_per_scenario - 1)
                )
            else:
                self.scenario_level = torch.zeros(
                    self.num_envs, device=self.device
                )
        else:
            w = torch.tensor([float(x) for x in sc.weights], device=self.device)
            self.scenario_id = torch.multinomial(
                w, self.num_envs, replacement=True
            )
            self.scenario_level = torch.full(
                (self.num_envs,),
                float(sc.curriculum_start_frac),
                device=self.device,
            )
        self.swing_height_target = torch.full(
            (self.num_envs,), float(self.cfg.rewards.feet_swing_height_target), device=self.device
        )
        self._update_scenario_conditions(torch.arange(self.num_envs, device=self.device))
        c = [int((self.scenario_id == k).sum()) for k in range(4)]
        mode = (
            "physical"
            if bool(getattr(sc, "physical_partition", False))
            else "proxy"
        )
        print(
            f"[scenario] {mode} partition ON: obstacle={c[0]} "
            f"slope={c[1]} load={c[2]} accel={c[3]} / "
            f"{self.num_envs} envs"
        )

    def _scenario_diff(self, lo, hi, env_ids=None):
        lvl = self.scenario_level if env_ids is None else self.scenario_level[env_ids]
        return lo + (hi - lo) * lvl

    def _update_scenario_conditions(self, env_ids):
        # Refresh per-env quantities that depend on curriculum level: obstacle swing target + slope gravity obs.
        if getattr(self, "scenario_id", None) is None:
            return
        sc = self.cfg.scenario
        sid = self.scenario_id[env_ids]
        obst = sid == 0
        if obst.any():
            e = env_ids[obst]
            if bool(getattr(sc, "physical_partition", False)):
                targets = torch.as_tensor(
                    sc.physical_clearance_targets,
                    device=self.device,
                    dtype=self.swing_height_target.dtype,
                )
                self.swing_height_target[e] = targets[
                    self.physical_level_id[e]
                ]
            else:
                self.swing_height_target[e] = self._scenario_diff(
                    *sc.obstacle_swing_target_range, e
                )
        slope = sid == 1
        if slope.any() and not bool(
            getattr(sc, "physical_partition", False)
        ):
            e = env_ids[slope]
            theta = torch.deg2rad(self._scenario_diff(*sc.slope_deg_range, e))
            self.gravity_vec[e, 0] = torch.sin(theta)
            self.gravity_vec[e, 1] = 0.0
            self.gravity_vec[e, 2] = -torch.cos(theta)

    def _apply_scenario_commands(self, env_ids):
        # Override commands per scenario (called at the end of _resample_commands). Lazily inits scenarios.
        self._init_scenarios_if_needed()
        if getattr(self, "scenario_id", None) is None:
            return
        sc = self.cfg.scenario
        sid = self.scenario_id[env_ids]
        if bool(getattr(sc, "physical_partition", False)):
            local_level = self.physical_level_id[env_ids]
            clearance = sid == 0
            if clearance.any():
                e = env_ids[clearance]
                self.commands[e, 0] = float(sc.physical_clearance_roll)
                self.cmd_step[e, 0] = float(sc.physical_clearance_step)
            slope_or_load = (sid == 1) | (sid == 2)
            if slope_or_load.any():
                e = env_ids[slope_or_load]
                self.commands[e, 0] = float(sc.physical_motion_roll)
                self.cmd_step[e, 0] = float(sc.physical_motion_step)
            speed = sid == 3
            if speed.any():
                e = env_ids[speed]
                speed_levels = torch.as_tensor(
                    sc.physical_speed_levels,
                    device=self.device,
                    dtype=self.commands.dtype,
                )
                target_total = speed_levels[local_level[speed]]
                step_cap = float(sc.physical_speed_step_cap)
                target_step = torch.minimum(
                    target_total,
                    torch.full_like(target_total, step_cap),
                )
                self.cmd_step[e, 0] = target_step
                self.commands[e, 0] = target_total - target_step
            self.commands[env_ids, 1:3] = 0.0
            return
        obst = sid == 0
        if obst.any():
            self.cmd_step[env_ids[obst], 0] = float(sc.obstacle_cmd_step)   # obstacle -> command stepping
        acc = sid == 3
        if acc.any():
            e = env_ids[acc]
            self.commands[e, 0] = self._scenario_diff(*sc.accel_vx_range, e)  # accel -> high forward speed
            self.cmd_step[e, 0] = 0.0                                         # ...rolling
        rolln = (sid == 1) | (sid == 2)
        if rolln.any():
            self.cmd_step[env_ids[rolln], 0] = 0.0                            # slope/load -> rolling

    def _apply_scenario_forces(self):
        # Per-env base force inside the decimation substep: slope=horizontal down-slope, load=downward weight.
        if getattr(self, "scenario_id", None) is None:
            return
        sc = self.cfg.scenario
        if bool(getattr(sc, "physical_partition", False)):
            return
        g = 9.81
        self.rigid_body_external_forces[:] = 0
        slope = (self.scenario_id == 1).float()
        theta = torch.deg2rad(self._scenario_diff(*sc.slope_deg_range))
        self.rigid_body_external_forces[:, self.ext_force_body_idx, 0] += (
            -float(sc.robot_weight_n) * torch.sin(theta) * slope
        )
        load = (self.scenario_id == 2).float()
        m_load = self._scenario_diff(*sc.load_kg_range)
        self.rigid_body_external_forces[:, self.ext_force_body_idx, 2] += -m_load * g * load
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces.view(-1, 3)),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques.view(-1, 3)),
            gymapi.ENV_SPACE,
        )

    def _update_scenario_curriculum(self, env_ids):
        # Per-env adaptive difficulty: advance on survival, back off on early fall; then refresh conditions.
        if (
            getattr(self, "scenario_id", None) is None
            or bool(
                getattr(self.cfg.scenario, "physical_partition", False)
            )
            or not getattr(self.cfg.scenario, "curriculum", True)
        ):
            return
        success = self.episode_length_buf[env_ids] > (0.8 * self.max_episode_length)
        step = float(self.cfg.scenario.curriculum_step)
        delta = torch.where(
            success,
            torch.full_like(self.scenario_level[env_ids], step),
            torch.full_like(self.scenario_level[env_ids], -step),
        )
        self.scenario_level[env_ids] = torch.clamp(self.scenario_level[env_ids] + delta, 0.0, 1.0)
        self._update_scenario_conditions(env_ids)

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
        if getattr(self.cfg.env, "use_gait_phase", False):
            self._step_contact_targets()
        if getattr(self.cfg.env, "use_gait_stepping", False):
            self._compute_step_roll_vel()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 0.1 * wrap_to_pi(self.commands[:, 3] - heading)

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[self.actor_indices, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        if getattr(self.cfg.env, "use_jump", False):
            self._update_jump_command()

    def _base_height_target_tensor(self):
        target = float(self.cfg.rewards.base_height_target)
        morphology_height = getattr(self, "env_base_height", None)
        if morphology_height is None:
            return self.base_height.new_full((self.num_envs,), target)
        # Every morphology spawns at its zero-pose geometry height plus the
        # same nominal offset. Preserve the checkpoint's nominal knee-bend
        # offset instead of forcing all leg lengths to one absolute height.
        standing_offset = target - float(self.cfg.init_state.pos[2])
        return (
            morphology_height
            + float(self._morph_spawn_offset)
            + standing_offset
        )

    def _update_jump_command(self):
        """Advance the latched IDLE/CROUCH/FLIGHT/LANDED jump state machine.

        Command height is the peak clearance of the lower wheel; command distance is
        mean wheel-center displacement from liftoff to touchdown, projected along the
        robot heading at liftoff.  Using the lower wheel prevents a one-leg swing from
        being counted as a successful jump.
        """
        t = self.episode_length_buf.float() * self.dt
        lo, hi = self.cfg.commands.jump_interval_s
        on_ground = (self.contact_forces[:, self.feet_indices, 2] > 1.0).any(dim=1)
        airborne = ~on_ground
        self.jump_air_steps = torch.where(airborne, self.jump_air_steps + 1, torch.zeros_like(self.jump_air_steps))  # 连续腾空步数
        vz = self.root_states[self.actor_indices, 9]                          # 世界系竖直速度
        h_above = self.base_height - self._base_height_target_tensor()       # 相对站高(可负)
        wheel_mid = self.foot_positions.mean(dim=1)
        wheel_vel = self.foot_velocities.mean(dim=1)
        wheel_clearance = self.foot_heights.min(dim=1).values
        st = self.jump_state.clone()                                          # 本步快照, 每 env 至多进一个相
        leg_len = self._jump_leg_length()
        prev_leg_len = torch.where(self.jump_prev_leg_len > 1e-6, self.jump_prev_leg_len, leg_len)
        leg_len_dot = (leg_len - prev_leg_len) / self.dt
        roll_abs, lat_ratio, _ = self._jump_lateral_metrics()
        self.jump_apex_fire[:] = False
        self.jump_takeoff_fire[:] = False
        self.jump_bad_takeoff_fire[:] = False
        self.jump_timeout_fire[:] = False
        if getattr(self.cfg.env, "use_jump_distance", False):
            self.jump_land_fire[:] = False

        active = st != 0
        if active.any():
            self.jump_max_roll = torch.where(active, torch.maximum(self.jump_max_roll, roll_abs), self.jump_max_roll)
            self.jump_max_lat_ratio = torch.where(active, torch.maximum(self.jump_max_lat_ratio, lat_ratio), self.jump_max_lat_ratio)
            self.jump_max_yaw_rate = torch.where(
                active, torch.maximum(self.jump_max_yaw_rate, torch.abs(self.base_ang_vel[:, 2])), self.jump_max_yaw_rate
            )
            self.jump_max_lat_vel = torch.where(
                active, torch.maximum(self.jump_max_lat_vel, torch.abs(self.base_lin_vel[:, 1])), self.jump_max_lat_vel
            )
        crouching = st == 1
        if crouching.any():
            self.jump_min_leg_len = torch.where(crouching, torch.minimum(self.jump_min_leg_len, leg_len), self.jump_min_leg_len)
            self.jump_min_base_height = torch.where(crouching, torch.minimum(self.jump_min_base_height, self.base_height), self.jump_min_base_height)

        init = self.jump_trigger_time == 0                                    # 首步排触发时间
        if init.any():
            self.jump_trigger_time[init] = t[init] + lo + (hi - lo) * torch.rand(int(init.sum()), device=self.device)

        trig = (st == 0) & (t >= self.jump_trigger_time)                      # IDLE 到点 -> 触发一次跳
        if trig.any():
            ids = trig.nonzero(as_tuple=False).flatten()
            self.jump_state[ids] = 1
            self.cmd_jump[ids, 0] = 1.0
            self.jumped_flag[ids] = False
            self.jump_landed_once[ids] = False
            self.jump_peak_height[ids] = 0.0
            self.jump_phase_time[ids] = t[ids]
            self.jump_init_base_height[ids] = self.base_height[ids]
            self.jump_min_base_height[ids] = self.base_height[ids]
            self.jump_init_leg_len[ids] = leg_len[ids]
            self.jump_prev_leg_len[ids] = leg_len[ids]
            self.jump_min_leg_len[ids] = leg_len[ids]
            self.jump_takeoff_leg_len[ids] = 0.0
            self.jump_takeoff_leg_gain[ids] = 0.0
            self.jump_takeoff_height_gain[ids] = 0.0
            self.jump_takeoff_leg_vel[ids] = 0.0
            self.jump_takeoff_crouch_time[ids] = 0.0
            self.jump_takeoff_quality[ids] = 0.0
            self.jump_bad_takeoff_fire[ids] = False
            self.jump_timeout_fire[ids] = False
            self.jump_takeoff_roll[ids] = roll_abs[ids]
            self.jump_takeoff_lat_ratio[ids] = lat_ratio[ids]
            self.jump_takeoff_fwd_vel[ids] = 0.0
            self.jump_takeoff_lat_vel[ids] = 0.0
            self.jump_takeoff_vz[ids] = 0.0
            self.jump_takeoff_base_vz[ids] = 0.0
            self.jump_takeoff_angle[ids] = 0.0
            self.jump_takeoff_leg_diff[ids] = 0.0
            self.jump_takeoff_leg_left[ids] = 0.0
            self.jump_takeoff_leg_right[ids] = 0.0
            self.jump_takeoff_foot_x_diff[ids] = 0.0
            self.jump_takeoff_foot_z_diff[ids] = 0.0
            self.jump_takeoff_abad_mirror_err[ids] = 0.0
            self.jump_takeoff_abad_left[ids] = 0.0
            self.jump_takeoff_abad_right[ids] = 0.0
            self.jump_takeoff_hip_left[ids] = 0.0
            self.jump_takeoff_hip_right[ids] = 0.0
            self.jump_takeoff_knee_left[ids] = 0.0
            self.jump_takeoff_knee_right[ids] = 0.0
            self.jump_takeoff_yaw_rate[ids] = 0.0
            self.jump_max_roll[ids] = roll_abs[ids]
            self.jump_max_lat_ratio[ids] = lat_ratio[ids]
            self.jump_max_yaw_rate[ids] = torch.abs(self.base_ang_vel[ids, 2])
            self.jump_max_lat_vel[ids] = torch.abs(self.base_lin_vel[ids, 1])
            self.jump_min_flight_leg_len[ids] = leg_len[ids]
            self.jump_preland_leg_len[ids] = leg_len[ids]
            self.jump_land_leg_len[ids] = 0.0
            command_override = getattr(self, "jump_command_override", None)
            if command_override is not None:
                self.jump_height_cmd[ids, 0] = command_override[ids, 0]
                if getattr(self.cfg.env, "use_jump_distance", False):
                    self.jump_dist_cmd[ids, 0] = command_override[ids, 1]
            else:
                lo_h, hi_h = self.cfg.commands.jump_height_cmd_range
                if getattr(self.cfg.commands, "jump_sample_edges", False):
                    h_sample = torch.randint(0, 2, (len(ids),), device=self.device, dtype=torch.long).float()
                else:
                    h_sample = torch.rand(len(ids), device=self.device)
                self.jump_height_cmd[ids, 0] = lo_h + (hi_h - lo_h) * h_sample
                if getattr(self.cfg.env, "use_jump_distance", False):
                    lo_d, hi_d = self.cfg.commands.jump_dist_cmd_range        # ★采样目标前跳距离 d*
                    if getattr(self.cfg.commands, "jump_sample_edges", False):
                        d_sample = torch.randint(0, 2, (len(ids),), device=self.device, dtype=torch.long).float()
                    else:
                        d_sample = torch.rand(len(ids), device=self.device)
                    self.jump_dist_cmd[ids, 0] = lo_d + (hi_d - lo_d) * d_sample
            if getattr(self.cfg.env, "use_jump_distance", False):
                self.jump_start_pos[ids] = wheel_mid[ids, :2].clone()             # 触发轮心中点(滚而不跳终止基准)
                self.jump_land_lateral_disp[ids] = 0.0
                self.jump_land_rewarded[ids] = False

        takeoff = (st == 1) & airborne                                       # CROUCH 离地 -> FLIGHT
        if takeoff.any():
            ids = takeoff.nonzero(as_tuple=False).flatten()
            crouch_time = t[ids] - self.jump_phase_time[ids]
            self.jump_state[ids] = 2
            self.jump_phase_time[ids] = t[ids]
            fwd = quat_apply(self.base_quat[ids], self.forward_vec[ids])[:, :2]   # 世界系前向
            fwd_dir = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-6)
            side_dir = torch.stack((-fwd_dir[:, 1], fwd_dir[:, 0]), dim=1)        # 世界系左向(+body y)
            world_vxy = wheel_vel[ids, :2]
            vfwd = (world_vxy * fwd_dir).sum(dim=1)
            vlat = (world_vxy * side_dir).sum(dim=1)
            wheel_vz = wheel_vel[ids, 2]
            leg_lengths = self._jump_leg_lengths()
            leg_diff, foot_x_diff, foot_z_diff, _, abad_err, _, _ = self._jump_symmetry_errors()
            self.jump_takeoff_fire[ids] = True
            self.jump_takeoff_leg_len[ids] = leg_len[ids]
            self.jump_takeoff_leg_gain[ids] = leg_len[ids] - self.jump_min_leg_len[ids]
            self.jump_takeoff_height_gain[ids] = self.base_height[ids] - self.jump_min_base_height[ids]
            self.jump_takeoff_leg_vel[ids] = leg_len_dot[ids]
            self.jump_takeoff_crouch_time[ids] = crouch_time
            self.jump_takeoff_roll[ids] = roll_abs[ids]
            self.jump_takeoff_lat_ratio[ids] = lat_ratio[ids]
            self.jump_takeoff_fwd_vel[ids] = vfwd
            self.jump_takeoff_lat_vel[ids] = vlat
            self.jump_takeoff_vz[ids] = wheel_vz
            self.jump_takeoff_base_vz[ids] = vz[ids]
            self.jump_takeoff_angle[ids] = torch.atan2(vfwd, wheel_vz.clamp(min=0.05))
            self.jump_takeoff_leg_diff[ids] = leg_diff[ids]
            self.jump_takeoff_leg_left[ids] = leg_lengths[ids, 0]
            self.jump_takeoff_leg_right[ids] = leg_lengths[ids, 1]
            self.jump_takeoff_foot_x_diff[ids] = foot_x_diff[ids]
            self.jump_takeoff_foot_z_diff[ids] = foot_z_diff[ids]
            self.jump_takeoff_abad_mirror_err[ids] = abad_err[ids]
            self.jump_takeoff_abad_left[ids] = self.dof_pos[ids, 0]
            self.jump_takeoff_abad_right[ids] = self.dof_pos[ids, 4]
            self.jump_takeoff_hip_left[ids] = self.dof_pos[ids, 1]
            self.jump_takeoff_hip_right[ids] = self.dof_pos[ids, 5]
            self.jump_takeoff_knee_left[ids] = self.dof_pos[ids, 2]
            self.jump_takeoff_knee_right[ids] = self.dof_pos[ids, 6]
            self.jump_takeoff_yaw_rate[ids] = self.base_ang_vel[ids, 2]
            q = self._jump_takeoff_quality(ids)
            self.jump_takeoff_quality[ids] = q
            q_min = float(getattr(self.cfg.rewards, "jump_takeoff_quality_min", 0.35))
            self.jump_bad_takeoff_fire[ids] = q < q_min
            self.jump_min_flight_leg_len[ids] = leg_len[ids]
            self.jump_preland_leg_len[ids] = leg_len[ids]
            if getattr(self.cfg.env, "use_jump_distance", False):            # ★锁起跳基准(跳距只从此算)
                self.jump_takeoff_pos[ids] = wheel_mid[ids, :2].clone()
                self.jump_takeoff_dir[ids] = fwd_dir
        timeout_s = float(getattr(self.cfg.rewards, "jump_crouch_timeout_s", 1.2))
        ctimeout = (st == 1) & (~airborne) & (t - self.jump_phase_time > timeout_s)  # 超时没起跳 -> 清 latch
        if ctimeout.any():
            ids = ctimeout.nonzero(as_tuple=False).flatten()
            self.jump_timeout_fire[ids] = True
            self.jump_state[ids] = 0
            self.cmd_jump[ids, 0] = 0.0
            self.jump_trigger_time[ids] = t[ids] + lo + (hi - lo) * torch.rand(len(ids), device=self.device)

        infl = (st == 2)                                                     # FLIGHT 记录双轮保守净空峰值
        if infl.any():
            self.jump_peak_height = torch.where(
                infl, torch.maximum(self.jump_peak_height, wheel_clearance), self.jump_peak_height
            )
            self.jump_min_flight_leg_len = torch.where(infl, torch.minimum(self.jump_min_flight_leg_len, leg_len), self.jump_min_flight_leg_len)
            descending = infl & (vz < 0.0)
            self.jump_preland_leg_len = torch.where(descending, torch.maximum(self.jump_preland_leg_len, leg_len), self.jump_preland_leg_len)
            apex = infl & (self.jump_prev_vz > 0) & (vz <= 0) & (~self.jumped_flag)   # vz 过零
            if apex.any():
                self.jumped_flag[apex] = True
            land = infl & on_ground                                         # 落地 -> LANDED
            if land.any():
                # Wheel clearance can peak after the base apex while the legs tuck, so
                # settle the final height exactly once when the complete flight is known.
                first_land_event = land & (~self.jump_landed_once)
                self.jump_apex_fire[first_land_event] = True
                self.jumped_flag[first_land_event] = True
                ids = land.nonzero(as_tuple=False).flatten()
                self.jump_state[ids] = 3
                self.jump_landed_once[ids] = True
                self.jump_phase_time[ids] = t[ids]
                self.jump_land_leg_len[ids] = leg_len[ids]
                if getattr(self.cfg.env, "use_jump_distance", False):       # ★结算前向有效跳距(只算腾空段)
                    first_land = ~self.jump_land_rewarded[ids]
                    if first_land.any():
                        fids = ids[first_land]
                        d_world = wheel_mid[fids, :2] - self.jump_takeoff_pos[fids]
                        self.jump_land_disp[fids] = (d_world * self.jump_takeoff_dir[fids]).sum(dim=1)
                        side_dir = torch.stack(
                            (-self.jump_takeoff_dir[fids, 1], self.jump_takeoff_dir[fids, 0]), dim=1
                        )
                        self.jump_land_lateral_disp[fids] = (d_world * side_dir).sum(dim=1)
                        self.jump_land_fire[fids] = True
                        self.jump_land_rewarded[fids] = True

        landed = (st == 3)                                                  # LANDED 站稳/超时 -> IDLE(清 latch)
        if landed.any():
            base_speed = torch.norm(self.base_lin_vel, dim=1)
            if getattr(self.cfg.rewards, "jump_require_centered_done", False):
                done_lat = float(getattr(self.cfg.rewards, "jump_done_lat_ratio", 0.55))
                done_roll = float(getattr(self.cfg.rewards, "jump_done_roll", 0.25))
                done_center = (lat_ratio < done_lat) & (roll_abs < done_roll)
            else:
                done_center = torch.ones_like(landed, dtype=torch.bool)
            landed_timeout_s = float(getattr(self.cfg.rewards, "jump_landed_timeout_s", 0.8))
            done = (
                landed
                & on_ground
                & (base_speed < 0.3)
                & (h_above.abs() < 0.12)
                & done_center
            ) | (landed & (t - self.jump_phase_time > landed_timeout_s))
            if done.any():
                ids = done.nonzero(as_tuple=False).flatten()
                self.jump_state[ids] = 0
                self.cmd_jump[ids, 0] = 0.0
                self.jump_landed_once[ids] = False
                self.jump_trigger_time[ids] = t[ids] + lo + (hi - lo) * torch.rand(len(ids), device=self.device)
            reflight = landed & airborne & (self.jump_air_steps >= 3) & (~done)   # 二次起飞(持续腾空≥3步, 排除落地弹跳); jumped_flag已锁不再给apex
            if reflight.any():
                self.jump_state[reflight.nonzero(as_tuple=False).flatten()] = 2

        self.jump_prev_vz = vz.clone()
        self.jump_prev_leg_len = leg_len.clone()

    def _reward_jump_apex(self):
        # Once-per-jump reward for the lower wheel's peak ground clearance.
        sigma = float(getattr(self.cfg.rewards, "jump_height_track_sigma", 0.01))
        r = torch.exp(-(self.jump_peak_height - self.jump_height_cmd[:, 0]) ** 2 / sigma)
        return r * self.jump_apex_fire.float() * self.jump_takeoff_quality

    def _reward_jump_height_error(self):
        scale = float(getattr(self.cfg.rewards, "jump_height_error_scale", 0.05))
        error = torch.square((self.jump_peak_height - self.jump_height_cmd[:, 0]) / max(scale, 1e-6))
        return error.clamp(0.0, 4.0) * self.jump_apex_fire.float() * self.jump_takeoff_quality

    def _reward_jump_est_height(self):
        # Predict the lower wheel's ballistic peak during the first ascent.
        limiting_wheel = self.foot_heights.argmin(dim=1, keepdim=True)
        wheel_vz = self.foot_velocities[:, :, 2].gather(1, limiting_wheel).squeeze(1)
        clearance = self.foot_heights.min(dim=1).values
        h_pred = clearance + (wheel_vz.clamp(min=0.0) ** 2) / (2 * 9.81)
        # ★仅第一跳上升段(CROUCH蓄力+FLIGHT上升)且apex前给; apex后(jumped_flag)停 -> 第二跳无est可farming
        ascending = (((self.jump_state == 1) | (self.jump_state == 2)) & (wheel_vz > 0) & (~self.jumped_flag)).float()
        sigma = float(getattr(self.cfg.rewards, "jump_height_track_sigma", 0.01))
        return torch.exp(-(h_pred - self.jump_height_cmd[:, 0]) ** 2 / sigma) * ascending

    def _reward_jump_relaunch(self):
        # ★惩罚二次起飞(双跳): 已落地过 且 持续腾空≥3步(真re-takeoff, 非落地弹跳) -> 罚。第一跳/短暂弹跳不误伤。
        airborne = ~(self.contact_forces[:, self.feet_indices, 2] > 1.0).any(dim=1)
        return (self.jump_landed_once & airborne & (self.jump_air_steps >= 3)).float()

    def _reward_jump_takeoff_sym(self):
        # CROUCH 中同时约束双轮受力与左右镜像几何，避免一条腿先/多蹬造成侧跳。
        f = self.contact_forces[:, self.feet_indices, 2]
        both = ((f > 1.0).sum(dim=1) == 2).float()
        force_sym = torch.exp(-torch.abs(f[:, 0] - f[:, 1]) / 50.0)
        leg_diff, foot_x_diff, foot_z_diff, _, abad_err, _, _ = self._jump_symmetry_errors()
        pose_sym = torch.exp(
            -torch.square(leg_diff / 0.10)
            -torch.square(foot_x_diff / 0.15)
            -torch.square(foot_z_diff / 0.10)
            -0.25 * torch.square(abad_err / 0.80)
        )
        sym = force_sym * pose_sym
        return both * sym * (self.jump_state == 1).float()

    def _reward_jump_upright_air(self):
        # Forward jump needs sagittal pitch; keep roll upright without forcing a vertical launch.
        if getattr(self.cfg.env, "use_jump_distance", False):
            upright = torch.sqrt((1.0 - torch.square(self.projected_gravity[:, 1])).clamp(0.0, 1.0))
        else:
            upright = (-self.projected_gravity[:, 2]).clamp(0.0, 1.0)
        return upright * (self.jump_state == 2).float()

    def _reward_jump_stand(self):
        # ★落地站住相(IDLE 待命): 机身零速 + 两轮不转 + base 回站高 + 直立。第二跳发生在该站住的相 -> 必摧毁此奖励 -> 结构性压制双跳。
        # 轮足务必显式约束轮不转(dof 3/7), 否则用滚代替站/蹭分。
        idle = (self.jump_state == 0).float()
        on_ground = (self.contact_forces[:, self.feet_indices, 2] > 1.0).any(dim=1).float()
        base_still = torch.exp(-(torch.norm(self.base_lin_vel, dim=1) ** 2) / 0.1)
        wheel_still = torch.exp(-((self.dof_vel[:, [3, 7]] ** 2).sum(dim=1)) / 4.0)
        sigma = float(getattr(self.cfg.rewards, "jump_stand_height_sigma", 0.02))
        height_r = torch.exp(
            -torch.square(
                self.base_height - self._base_height_target_tensor()
            )
            / sigma
        )
        upright = (-self.projected_gravity[:, 2]).clamp(0.0, 1.0)
        return idle * on_ground * base_still * wheel_still * height_r * upright

    def _reward_jump_posture(self):
        # IDLE 待命站立地基(站到目标高度+直立, 至少一脚触地)。防塌陷早死。
        idle = (self.jump_state == 0).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).sum(dim=1) >= 1).float()
        sigma = float(getattr(self.cfg.rewards, "jump_posture_height_sigma", 0.08))
        height_r = torch.exp(
            -torch.square(
                self.base_height - self._base_height_target_tensor()
            )
            / sigma
        )
        upright = (-self.projected_gravity[:, 2]).clamp(0.0, 1.0)
        return idle * on_ground * height_r * upright

    def _reward_jump_idle_low_height(self):
        idle = (self.jump_state == 0).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).any(dim=1)).float()
        margin = float(getattr(self.cfg.rewards, "jump_idle_height_margin", 0.03))
        err_scale = float(getattr(self.cfg.rewards, "jump_idle_height_error_scale", 0.12))
        low_err = torch.relu(
            self._base_height_target_tensor()
            - self.base_height
            - margin
        )
        return idle * on_ground * (low_err / max(err_scale, 1e-6)).clamp(0.0, 1.0)

    def _reward_jump_idle_lateral(self):
        idle = (self.jump_state == 0).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).any(dim=1)).float()
        roll_abs, lat_ratio, _ = self._jump_lateral_metrics()
        lat_limit = float(getattr(self.cfg.rewards, "jump_idle_lateral_limit", 0.45))
        lat_scale = float(getattr(self.cfg.rewards, "jump_idle_lateral_scale", 0.45))
        roll_limit = float(getattr(self.cfg.rewards, "jump_roll_soft_limit", 0.20))
        lat_pen = torch.square(torch.relu(lat_ratio - lat_limit) / max(lat_scale, 1e-6))
        roll_pen = torch.square(torch.relu(roll_abs - roll_limit) / max(roll_limit, 1e-6))
        return idle * on_ground * (lat_pen + 0.5 * roll_pen).clamp(0.0, 4.0)

    def _reward_jump_landed_recover(self):
        landed = (self.jump_state == 3).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).any(dim=1)).float()
        base_still = torch.exp(-(torch.norm(self.base_lin_vel, dim=1) ** 2) / 0.1)
        wheel_still = torch.exp(-((self.dof_vel[:, [3, 7]] ** 2).sum(dim=1)) / 4.0)
        sigma = float(getattr(self.cfg.rewards, "jump_landed_recover_height_sigma", 0.03))
        height_r = torch.exp(
            -torch.square(
                self.base_height - self._base_height_target_tensor()
            )
            / sigma
        )
        return landed * on_ground * base_still * wheel_still * height_r * self._jump_centering_quality()

    def _reward_jump_stance(self):
        # IDLE 站姿整形: 双轮相对机身同前后、同高、横向镜像且轮距合理。
        idle = (self.jump_state == 0).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).sum(dim=1) >= 1).float()
        fpb = self._jump_foot_positions_body()
        dx = fpb[:, 0, 0] - fpb[:, 1, 0]
        dz = fpb[:, 0, 2] - fpb[:, 1, 2]
        mid_y = 0.5 * (fpb[:, 0, 1] + fpb[:, 1, 1])
        dy = torch.abs(fpb[:, 0, 1] - fpb[:, 1, 1])
        no_fore_aft = torch.exp(-(dx ** 2) / 0.01)
        same_height = torch.exp(-(dz ** 2) / 0.0025)
        lateral_mirror = torch.exp(-(mid_y ** 2) / 0.0025)
        w = float(getattr(self.cfg.rewards, "jump_stance_width", 0.33))
        width_ok = torch.exp(-((dy - w) ** 2) / 0.01)
        stance_quality = 0.25 * (no_fore_aft + same_height + lateral_mirror + width_ok)
        return idle * on_ground * stance_quality

    def _jump_foot_positions_body(self):
        foot_delta = self.foot_positions - self.base_position.unsqueeze(1)
        return torch.stack(
            [quat_rotate_inverse(self.base_quat, foot_delta[:, i, :]) for i in range(len(self.feet_indices))],
            dim=1,
        )

    def _jump_leg_lengths(self):
        fpb = self._jump_foot_positions_body()
        sagittal_len = torch.sqrt(torch.clamp(fpb[:, :, 0] ** 2 + fpb[:, :, 2] ** 2, min=1e-8))
        return sagittal_len

    def _jump_leg_length(self):
        return self._jump_leg_lengths().mean(dim=1)

    def _jump_symmetry_errors(self):
        """Physical mirror errors; mirrored URDF axes mean symmetric q satisfies q_L + q_R = 0."""
        fpb = self._jump_foot_positions_body()
        leg_len = torch.sqrt(torch.clamp(fpb[:, :, 0] ** 2 + fpb[:, :, 2] ** 2, min=1e-8))
        leg_diff = torch.abs(leg_len[:, 0] - leg_len[:, 1])
        foot_x_diff = torch.abs(fpb[:, 0, 0] - fpb[:, 1, 0])
        foot_z_diff = torch.abs(fpb[:, 0, 2] - fpb[:, 1, 2])
        foot_mid_y = torch.abs(0.5 * (fpb[:, 0, 1] + fpb[:, 1, 1]))
        abad_mirror = torch.abs(self.dof_pos[:, 0] + self.dof_pos[:, 4])
        hip_mirror = torch.abs(self.dof_pos[:, 1] + self.dof_pos[:, 5])
        knee_mirror = torch.abs(self.dof_pos[:, 2] + self.dof_pos[:, 6])
        return leg_diff, foot_x_diff, foot_z_diff, foot_mid_y, abad_mirror, hip_mirror, knee_mirror

    def _jump_lateral_metrics(self):
        side_vec = torch.zeros_like(self.forward_vec)
        side_vec[:, 1] = 1.0
        side_world = quat_apply(self.base_quat, side_vec)
        wheel_mid = self.foot_positions.mean(dim=1)
        wheel_delta = self.foot_positions[:, 0, :] - self.foot_positions[:, 1, :]
        half_width = 0.5 * torch.abs((wheel_delta * side_world).sum(dim=1)).clamp(min=0.05)
        lat_signed = ((self.base_position - wheel_mid) * side_world).sum(dim=1)
        lat_ratio = torch.abs(lat_signed) / half_width
        roll_abs = torch.abs(torch.atan2(self.projected_gravity[:, 1], -self.projected_gravity[:, 2].clamp(max=-1e-6)))
        return roll_abs, lat_ratio, lat_signed

    def _jump_phase_elapsed(self):
        return (self.episode_length_buf.float() * self.dt - self.jump_phase_time).clamp(min=0.0)

    def _apply_jump_explicit_trajectory(self, pos_action):
        """Blend deterministic mirrored sagittal IK targets with a small RL residual."""
        cfg = self.cfg.control
        phase_t = self._jump_phase_elapsed()

        def smooth_step(value):
            value = value.clamp(0.0, 1.0)
            return value * value * (3.0 - 2.0 * value)

        lo_h, hi_h = self.cfg.commands.jump_height_cmd_range
        lo_d, hi_d = self.cfg.commands.jump_dist_cmd_range
        height_unit = ((self.jump_height_cmd[:, 0] - lo_h) / max(hi_h - lo_h, 1e-6)).clamp(0.0, 1.0)
        distance_unit = ((self.jump_dist_cmd[:, 0] - lo_d) / max(hi_d - lo_d, 1e-6)).clamp(0.0, 1.0)

        takeoff_length_range = getattr(cfg, "jump_trajectory_takeoff_length_range", [0.48, 0.56])
        takeoff_length = takeoff_length_range[0] + (
            takeoff_length_range[1] - takeoff_length_range[0]
        ) * height_unit
        takeoff_angle_range = getattr(cfg, "jump_trajectory_takeoff_angle_range", [0.55, 0.95])
        takeoff_angle = -(takeoff_angle_range[0] + (
            takeoff_angle_range[1] - takeoff_angle_range[0]
        ) * distance_unit)
        preland_angle_range = getattr(cfg, "jump_trajectory_preland_angle_range", [0.20, 0.40])
        preland_angle = preland_angle_range[0] + (
            preland_angle_range[1] - preland_angle_range[0]
        ) * distance_unit

        stance_length = float(getattr(cfg, "jump_trajectory_stance_length", 0.37))
        stance_angle = float(getattr(cfg, "jump_trajectory_stance_angle", 0.20))
        crouch_length = float(getattr(cfg, "jump_trajectory_crouch_length", 0.30))
        crouch_angle = float(getattr(cfg, "jump_trajectory_crouch_angle", 0.10))
        crouch_end = float(getattr(cfg, "jump_trajectory_crouch_end_s", 0.18))
        push_end = float(getattr(cfg, "jump_trajectory_push_end_s", 0.48))
        crouch_blend = smooth_step(phase_t / max(crouch_end, 1e-6))
        push_blend = smooth_step((phase_t - crouch_end) / max(push_end - crouch_end, 1e-6))
        ground_length = stance_length + (crouch_length - stance_length) * crouch_blend
        ground_length = ground_length + (takeoff_length - crouch_length) * push_blend
        ground_angle = stance_angle + (crouch_angle - stance_angle) * crouch_blend
        ground_angle = ground_angle + (takeoff_angle - crouch_angle) * push_blend

        tuck_length = float(getattr(cfg, "jump_trajectory_tuck_length", 0.30))
        tuck_angle = float(getattr(cfg, "jump_trajectory_tuck_angle", 0.0))
        tuck_time = float(getattr(cfg, "jump_trajectory_tuck_s", 0.16))
        tuck_blend = smooth_step(phase_t / max(tuck_time, 1e-6))
        flight_length = takeoff_length + (tuck_length - takeoff_length) * tuck_blend
        flight_angle = takeoff_angle + (tuck_angle - takeoff_angle) * tuck_blend
        down_vz = -self.root_states[self.actor_indices, 9]
        preland_vz_scale = float(getattr(cfg, "jump_trajectory_preland_vz_scale", 1.0))
        preland_blend = smooth_step(down_vz / max(preland_vz_scale, 1e-6))
        descending_length = tuck_length + (
            float(getattr(cfg, "jump_trajectory_preland_length", 0.46)) - tuck_length
        ) * preland_blend
        descending_angle = tuck_angle + (preland_angle - tuck_angle) * preland_blend
        flight_length = torch.where(self.jumped_flag, descending_length, flight_length)
        flight_angle = torch.where(self.jumped_flag, descending_angle, flight_angle)

        landing_hold = float(getattr(cfg, "jump_trajectory_landing_hold_s", 0.20))
        landing_blend = smooth_step(phase_t / max(landing_hold, 1e-6))
        landing_length = float(getattr(cfg, "jump_trajectory_preland_length", 0.46)) + (
            stance_length - float(getattr(cfg, "jump_trajectory_preland_length", 0.46))
        ) * landing_blend
        landing_angle = preland_angle + (stance_angle - preland_angle) * landing_blend

        state = self.jump_state
        target_length = torch.where(state == 1, ground_length, flight_length)
        target_angle = torch.where(state == 1, ground_angle, flight_angle)
        target_length = torch.where(state == 3, landing_length, target_length)
        target_angle = torch.where(state == 3, landing_angle, target_angle)

        segment_length = float(getattr(cfg, "jump_trajectory_segment_length", 0.30))
        zero_half_angle = float(getattr(cfg, "jump_trajectory_zero_half_angle", 0.5235987755982988))
        length_ratio = (target_length / (2.0 * segment_length)).clamp(0.05, 0.999)
        knee_left = 2.0 * (torch.acos(length_ratio) - zero_half_angle)
        hip_left = 0.5 * knee_left - target_angle
        hip_left = torch.maximum(
            torch.minimum(hip_left, self.dof_pos_limits[1, 1] - 0.03),
            self.dof_pos_limits[1, 0] + 0.03,
        )
        knee_left = torch.maximum(
            torch.minimum(knee_left, self.dof_pos_limits[2, 1] - 0.03),
            self.dof_pos_limits[2, 0] + 0.03,
        )

        leg_indices = [1, 2, 5, 6]
        target = torch.stack((hip_left, knee_left, -hip_left, -knee_left), dim=1)
        target_delta = target - self.default_dof_pos[:, leg_indices]
        strength = float(getattr(cfg, "jump_trajectory_strength", 1.0))
        blend_in = float(getattr(cfg, "jump_trajectory_blend_in_s", 0.08))
        active_blend = strength * smooth_step(phase_t / max(blend_in, 1e-6))
        active_blend = torch.where(state == 2, torch.full_like(active_blend, strength), active_blend)
        active_blend = torch.where(state == 3, strength * (1.0 - landing_blend), active_blend)
        active_blend *= ((state >= 1) & (state <= 3)).float()
        rl_scale = float(getattr(cfg, "jump_trajectory_rl_scale", 0.15))
        current = pos_action[:, leg_indices]
        pos_action[:, leg_indices] = (
            (1.0 - active_blend.unsqueeze(1)) * current
            + active_blend.unsqueeze(1) * (target_delta + rl_scale * current)
        )
        return pos_action

    def _jump_regularizer_weight(self):
        if not getattr(self.cfg.env, "use_jump", False):
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        active = ((self.cmd_jump[:, 0] > 0.5) | (self.jump_state != 0)).float()
        relax = float(getattr(self.cfg.rewards, "jump_regularizer_relax", 1.0))
        return 1.0 - active * (1.0 - relax)

    def _jump_centering_quality(self):
        roll_abs, lat_ratio, _ = self._jump_lateral_metrics()
        lat_scale = float(getattr(self.cfg.rewards, "jump_center_lat_scale", 0.55))
        roll_scale = float(getattr(self.cfg.rewards, "jump_center_roll_scale", 0.25))
        lat_q = torch.exp(-torch.square(lat_ratio / max(lat_scale, 1e-6)))
        roll_q = torch.exp(-torch.square(roll_abs / max(roll_scale, 1e-6)))
        return lat_q * roll_q

    def _reward_jump_lateral_stability(self):
        roll_abs, lat_ratio, _ = self._jump_lateral_metrics()
        lat_limit = float(getattr(self.cfg.rewards, "jump_lateral_ratio_limit", 0.45))
        roll_limit = float(getattr(self.cfg.rewards, "jump_roll_soft_limit", 0.20))
        active = ((self.cmd_jump[:, 0] > 0.5) | (self.jump_state != 0)).float()
        idle = (self.jump_state == 0).float()
        lat_pen = torch.relu(lat_ratio - lat_limit) ** 2
        roll_pen = torch.relu(roll_abs - roll_limit) ** 2
        roll_rate_pen = 0.05 * torch.square(self.base_ang_vel[:, 0])
        yaw_rate_pen = 0.05 * torch.square(self.base_ang_vel[:, 2])
        return (idle + active) * (lat_pen + roll_pen + roll_rate_pen + yaw_rate_pen)

    def _reward_jump_centered(self):
        prep = ((self.jump_state == 0) | (self.jump_state == 1)).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).any(dim=1)).float()
        return prep * on_ground * self._jump_centering_quality()

    def _reward_jump_abad_still(self):
        abad_pos = self.dof_pos[:, [0, 4]]
        abad_vel = self.dof_vel[:, [0, 4]]
        return torch.sum(torch.square(abad_pos), dim=1) + 0.02 * torch.sum(torch.square(abad_vel), dim=1)

    def _reward_jump_abad_upright(self):
        prep = ((self.jump_state == 0) | (self.jump_state == 1)).float()
        scale = float(getattr(self.cfg.rewards, "jump_abad_upright_scale", 0.20))
        per_leg = 1.0 / (1.0 + torch.square(self.dof_pos[:, [0, 4]] / max(scale, 1e-6)))
        return prep * per_leg.mean(dim=1)

    def _reward_jump_abad_common(self):
        prep = ((self.jump_state == 0) | (self.jump_state == 1)).float()
        scale = float(getattr(self.cfg.rewards, "jump_abad_common_scale", 0.25))
        common_mode = 0.5 * (self.dof_pos[:, 0] + self.dof_pos[:, 4])
        return prep * torch.square(common_mode / max(scale, 1e-6)).clamp(0.0, 4.0)

    def _reward_jump_leg_sync(self):
        leg_diff, foot_x_diff, foot_z_diff, foot_mid_y, abad_err, hip_err, knee_err = self._jump_symmetry_errors()
        mirror_vel = self.dof_vel[:, [0, 1, 2]] + self.dof_vel[:, [4, 5, 6]]
        geometry_err = (
            torch.square(leg_diff / 0.06)
            + 0.5 * torch.square(foot_x_diff / 0.10)
            + 0.5 * torch.square(foot_z_diff / 0.06)
            + 0.25 * torch.square(foot_mid_y / 0.12)
        )
        joint_err = 0.10 * torch.square(abad_err / 0.60) + 0.03 * (
            torch.square(hip_err / 0.50) + torch.square(knee_err / 0.50)
        )
        velocity_err = 0.002 * torch.sum(torch.square(mirror_vel), dim=1)
        return (geometry_err + joint_err + velocity_err).clamp(0.0, 4.0)

    def _reward_jump_mirror_pose(self):
        leg_diff, foot_x_diff, foot_z_diff, foot_mid_y, abad_err, hip_err, knee_err = self._jump_symmetry_errors()
        qualities = torch.stack(
            (
                1.0 / (1.0 + torch.square(leg_diff / 0.06)),
                1.0 / (1.0 + torch.square(foot_x_diff / 0.10)),
                1.0 / (1.0 + torch.square(foot_z_diff / 0.06)),
                1.0 / (1.0 + torch.square(foot_mid_y / 0.12)),
                1.0 / (1.0 + torch.square(abad_err / 0.60)),
                1.0 / (1.0 + torch.square(hip_err / 0.50)),
                1.0 / (1.0 + torch.square(knee_err / 0.50)),
            ),
            dim=1,
        )
        return qualities.mean(dim=1)

    def _reward_jump_lateral_velocity(self):
        active = ((self.cmd_jump[:, 0] > 0.5) | (self.jump_state != 0)).float()
        idle = (self.jump_state == 0).float()
        scale = float(getattr(self.cfg.rewards, "jump_lateral_vel_scale", 0.35))
        return (active + 0.5 * idle) * torch.square(self.base_lin_vel[:, 1] / max(scale, 1e-6)).clamp(0.0, 4.0)

    def _reward_jump_yaw_stability(self):
        yaw_scale = float(getattr(self.cfg.rewards, "jump_yaw_rate_scale", 0.6))
        wheel_diff_scale = float(getattr(self.cfg.rewards, "jump_wheel_diff_scale", 4.0))
        yaw_rate = torch.square(self.base_ang_vel[:, 2] / max(yaw_scale, 1e-6))
        wheel_diff = torch.square((self.dof_vel[:, 3] - self.dof_vel[:, 7]) / max(wheel_diff_scale, 1e-6))
        return (yaw_rate + 0.1 * wheel_diff).clamp(0.0, 4.0)

    def _reward_jump_crouch_low(self):
        crouch = (self.jump_state == 1).float()
        target = float(getattr(self.cfg.rewards, "jump_crouch_height_drop_target", 0.08))
        low_window_s = float(getattr(self.cfg.rewards, "jump_crouch_low_reward_s", 0.35))
        time_gate = (self._jump_phase_elapsed() <= low_window_s).float()
        drop = (self.jump_init_base_height - self.base_height).clamp(min=0.0)
        return crouch * time_gate * (drop / max(target, 1e-6)).clamp(0.0, 1.0)

    def _reward_jump_crouch_extend(self):
        crouch = (self.jump_state == 1).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).sum(dim=1) == 2).float()
        phase_t = self._jump_phase_elapsed()
        push_start = float(getattr(self.cfg.rewards, "jump_push_start_s", 0.12))
        push_end = float(getattr(self.cfg.rewards, "jump_push_end_s", 0.55))
        time_gate = ((phase_t >= push_start) & (phase_t <= push_end)).float()
        leg_len = self._jump_leg_length()
        prev_leg_len = torch.where(self.jump_prev_leg_len > 1e-6, self.jump_prev_leg_len, leg_len)
        leg_vel = (leg_len - prev_leg_len) / self.dt
        drop_target = float(getattr(self.cfg.rewards, "jump_crouch_height_drop_target", 0.08))
        gain_target = float(getattr(self.cfg.rewards, "jump_takeoff_leg_gain_target", 0.08))
        vel_target = float(getattr(self.cfg.rewards, "jump_takeoff_leg_vel_target", 0.45))
        drop = (self.jump_init_base_height - self.base_height).clamp(min=0.0)
        extend = (leg_len - self.jump_min_leg_len).clamp(min=0.0)
        drop_ok = (drop / max(drop_target, 1e-6)).clamp(0.0, 1.0)
        extend_r = (extend / max(gain_target, 1e-6)).clamp(0.0, 1.0)
        vel_r = (leg_vel.clamp(min=0.0) / max(vel_target, 1e-6)).clamp(0.0, 1.0)
        center_gate = (0.2 + 0.8 * self._jump_centering_quality()).clamp(0.0, 1.0)
        return crouch * on_ground * time_gate * center_gate * drop_ok * (0.5 * extend_r + 0.5 * vel_r)

    def _jump_takeoff_quality(self, env_ids=None):
        if env_ids is None:
            gain = self.jump_takeoff_leg_gain
            height_gain = self.jump_takeoff_height_gain
            leg_vel = self.jump_takeoff_leg_vel
            crouch_time = self.jump_takeoff_crouch_time
            lat_ratio = self.jump_takeoff_lat_ratio
            roll = self.jump_takeoff_roll
        else:
            gain = self.jump_takeoff_leg_gain[env_ids]
            height_gain = self.jump_takeoff_height_gain[env_ids]
            leg_vel = self.jump_takeoff_leg_vel[env_ids]
            crouch_time = self.jump_takeoff_crouch_time[env_ids]
            lat_ratio = self.jump_takeoff_lat_ratio[env_ids]
            roll = self.jump_takeoff_roll[env_ids]
        gain_target = float(getattr(self.cfg.rewards, "jump_takeoff_leg_gain_target", 0.08))
        height_target = float(getattr(self.cfg.rewards, "jump_takeoff_height_gain_target", 0.08))
        vel_target = float(getattr(self.cfg.rewards, "jump_takeoff_leg_vel_target", 0.45))
        time_target = float(getattr(self.cfg.rewards, "jump_takeoff_crouch_time_target", 0.20))
        gain_q = (gain / max(gain_target, 1e-6)).clamp(0.0, 1.0)
        height_q = (height_gain / max(height_target, 1e-6)).clamp(0.0, 1.0)
        vel_q = (leg_vel / max(vel_target, 1e-6)).clamp(0.0, 1.0)
        time_q = (crouch_time / max(time_target, 1e-6)).clamp(0.0, 1.0)
        core_q = torch.minimum(torch.minimum(gain_q, height_q), torch.minimum(vel_q, time_q))
        lat_limit = float(getattr(self.cfg.rewards, "jump_lateral_ratio_limit", 0.45))
        roll_limit = float(getattr(self.cfg.rewards, "jump_roll_soft_limit", 0.20))
        lat_q = (1.0 - torch.relu(lat_ratio - lat_limit) / max(lat_limit, 1e-6)).clamp(0.0, 1.0)
        roll_q = (1.0 - torch.relu(roll - roll_limit) / max(roll_limit, 1e-6)).clamp(0.0, 1.0)
        return core_q * torch.minimum(lat_q, roll_q)

    def _reward_jump_takeoff_leg_extend(self):
        return self.jump_takeoff_fire.float() * self.jump_takeoff_quality

    def _reward_jump_bad_takeoff(self):
        return self.jump_bad_takeoff_fire.float() * (1.0 - self.jump_takeoff_quality).clamp(0.0, 1.0)

    def _reward_jump_no_takeoff(self):
        return self.jump_timeout_fire.float()

    def _reward_jump_crouch_overtime(self):
        crouch = (self.jump_state == 1).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).any(dim=1)).float()
        deadline = float(getattr(self.cfg.rewards, "jump_takeoff_deadline_s", 0.55))
        timeout_s = float(getattr(self.cfg.rewards, "jump_crouch_timeout_s", 0.85))
        overtime = torch.relu(self._jump_phase_elapsed() - deadline) / max(timeout_s - deadline, 1e-6)
        return crouch * on_ground * overtime.clamp(0.0, 1.0)

    def _reward_jump_push_up(self):
        crouch = (self.jump_state == 1).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).sum(dim=1) == 2).float()
        phase_t = self._jump_phase_elapsed()
        push_start = float(getattr(self.cfg.rewards, "jump_push_start_s", 0.12))
        push_end = float(getattr(self.cfg.rewards, "jump_push_end_s", 0.55))
        time_gate = ((phase_t >= push_start) & (phase_t <= push_end)).float()
        leg_len = self._jump_leg_length()
        prev_leg_len = torch.where(self.jump_prev_leg_len > 1e-6, self.jump_prev_leg_len, leg_len)
        leg_vel = (leg_len - prev_leg_len) / self.dt
        drop_target = float(getattr(self.cfg.rewards, "jump_crouch_height_drop_target", 0.08))
        gain_target = float(getattr(self.cfg.rewards, "jump_takeoff_leg_gain_target", 0.08))
        leg_vel_target = float(getattr(self.cfg.rewards, "jump_takeoff_leg_vel_target", 0.45))
        if getattr(self.cfg.rewards, "jump_push_vz_from_command", False):
            command_gain = float(getattr(self.cfg.rewards, "jump_takeoff_vz_command_gain", 0.8))
            vz_target = command_gain * torch.sqrt(2.0 * 9.81 * self.jump_height_cmd[:, 0].clamp(min=0.02))
        else:
            vz_target = torch.full_like(leg_vel, float(getattr(self.cfg.rewards, "jump_push_vz_target", 0.8)))
        drop = (self.jump_init_base_height - self.base_height).clamp(min=0.0)
        extend = (leg_len - self.jump_min_leg_len).clamp(min=0.0)
        drop_gate = (drop / max(drop_target, 1e-6)).clamp(0.0, 1.0)
        extend_gate = (extend / max(gain_target, 1e-6)).clamp(0.0, 1.0)
        leg_r = (leg_vel.clamp(min=0.0) / max(leg_vel_target, 1e-6)).clamp(0.0, 1.0)
        vz = self.root_states[self.actor_indices, 9]
        vz_r = (vz.clamp(min=0.0) / vz_target.clamp(min=1e-6)).clamp(0.0, 1.0)
        center_gate = (0.2 + 0.8 * self._jump_centering_quality()).clamp(0.0, 1.0)
        return crouch * on_ground * time_gate * center_gate * drop_gate * (0.5 + 0.5 * extend_gate) * (0.4 * leg_r + 0.6 * vz_r)

    def _reward_jump_push_vz_cmd(self):
        crouch = (self.jump_state == 1).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).sum(dim=1) == 2).float()
        phase_t = self._jump_phase_elapsed()
        push_start = float(getattr(self.cfg.rewards, "jump_push_start_s", 0.12))
        push_end = float(getattr(self.cfg.rewards, "jump_push_end_s", 0.55))
        time_gate = ((phase_t >= push_start) & (phase_t <= push_end)).float()
        command_gain = float(getattr(self.cfg.rewards, "jump_takeoff_vz_command_gain", 0.8))
        target = command_gain * torch.sqrt(2.0 * 9.81 * self.jump_height_cmd[:, 0].clamp(min=0.02))
        vz = self.root_states[self.actor_indices, 9]
        progress = (vz.clamp(min=0.0) / target.clamp(min=1e-6)).clamp(0.0, 1.0)
        sigma = float(getattr(self.cfg.rewards, "jump_push_vz_sigma", 0.04))
        tracking = torch.exp(-torch.square(vz - target) / max(sigma, 1e-6))
        leg_len = self._jump_leg_length()
        extend = (leg_len - self.jump_min_leg_len).clamp(min=0.0)
        gain_target = float(getattr(self.cfg.rewards, "jump_takeoff_leg_gain_target", 0.08))
        extend_gate = (0.25 + 0.75 * (extend / max(gain_target, 1e-6)).clamp(0.0, 1.0))
        return crouch * on_ground * time_gate * extend_gate * (0.5 * progress + 0.5 * tracking)

    def _reward_jump_push_fwd_cmd(self):
        crouch = (self.jump_state == 1).float()
        on_ground = ((self.contact_forces[:, self.feet_indices, 2] > 1.0).sum(dim=1) == 2).float()
        phase_t = self._jump_phase_elapsed()
        push_start = float(getattr(self.cfg.rewards, "jump_push_start_s", 0.12))
        push_end = float(getattr(self.cfg.rewards, "jump_push_end_s", 0.55))
        time_gate = ((phase_t >= push_start) & (phase_t <= push_end)).float()
        h_star = self.jump_height_cmd[:, 0].clamp(min=0.05)
        flight_time = 2.0 * torch.sqrt(2.0 * h_star / 9.81)
        target = (self.jump_dist_cmd[:, 0] / flight_time.clamp(min=0.15)).clamp(min=0.05, max=3.0)
        vfwd = self.base_lin_vel[:, 0]
        progress = (vfwd.clamp(min=0.0) / target).clamp(0.0, 1.0)
        sigma = float(getattr(self.cfg.rewards, "jump_push_fwd_sigma", 0.04))
        tracking = torch.exp(-torch.square(vfwd - target) / max(sigma, 1e-6))
        leg_len = self._jump_leg_length()
        extend = (leg_len - self.jump_min_leg_len).clamp(min=0.0)
        gain_target = float(getattr(self.cfg.rewards, "jump_takeoff_leg_gain_target", 0.08))
        extend_gate = (0.25 + 0.75 * (extend / max(gain_target, 1e-6)).clamp(0.0, 1.0))
        return crouch * on_ground * time_gate * extend_gate * (0.5 * progress + 0.5 * tracking)

    def _reward_jump_takeoff_vz(self):
        lat_limit = float(getattr(self.cfg.rewards, "jump_lateral_ratio_limit", 0.45))
        roll_limit = float(getattr(self.cfg.rewards, "jump_roll_soft_limit", 0.20))
        lat_q = (1.0 - torch.relu(self.jump_takeoff_lat_ratio - lat_limit) / max(lat_limit, 1e-6)).clamp(0.0, 1.0)
        roll_q = (1.0 - torch.relu(self.jump_takeoff_roll - roll_limit) / max(roll_limit, 1e-6)).clamp(0.0, 1.0)
        gain = float(getattr(self.cfg.rewards, "jump_takeoff_vz_command_gain", 0.8))
        vz_target = gain * torch.sqrt(2.0 * 9.81 * self.jump_height_cmd[:, 0].clamp(min=0.02))
        sigma = float(getattr(self.cfg.rewards, "jump_takeoff_vz_track_sigma", 0.16))
        vz_q = torch.exp(-torch.square(self.jump_takeoff_base_vz - vz_target) / max(sigma, 1e-6))
        return self.jump_takeoff_fire.float() * vz_q * torch.minimum(lat_q, roll_q)

    def _reward_jump_takeoff_angle(self):
        h_star = self.jump_height_cmd[:, 0].clamp(min=0.05)
        angle_gain = float(getattr(self.cfg.rewards, "jump_takeoff_angle_gain", 1.0))
        target = angle_gain * torch.atan((self.jump_dist_cmd[:, 0] / (4.0 * h_star)).clamp(0.0, 2.0))
        sigma = float(getattr(self.cfg.rewards, "jump_takeoff_angle_sigma", 0.04))
        return self.jump_takeoff_fire.float() * self.jump_takeoff_quality * torch.exp(-torch.square(self.jump_takeoff_angle - target) / sigma)

    def _reward_jump_air_tuck(self):
        vz = self.root_states[self.actor_indices, 9]
        leg_len = self._jump_leg_length()
        delta = float(getattr(self.cfg.rewards, "jump_air_tuck_len_delta", 0.05))
        tuck = (self.jump_takeoff_leg_len - leg_len) / max(delta, 1e-6)
        ascending_flight = ((self.jump_state == 2) & (vz >= 0.0)).float()
        return ascending_flight * self.jump_takeoff_quality * tuck.clamp(0.0, 1.0)

    def _reward_jump_preland_extend(self):
        vz = self.root_states[self.actor_indices, 9]
        leg_len = self._jump_leg_length()
        delta = float(getattr(self.cfg.rewards, "jump_preland_extend_delta", 0.08))
        extend = (leg_len - self.jump_min_flight_leg_len) / max(delta, 1e-6)
        descending_flight = ((self.jump_state == 2) & (vz < 0.0)).float()
        return descending_flight * self.jump_takeoff_quality * extend.clamp(0.0, 1.0)

    def _reward_jump_landing_compress(self):
        delta = float(getattr(self.cfg.rewards, "jump_landing_compress_delta", 0.05))
        compress = (self.jump_preland_leg_len - self.jump_land_leg_len) / max(delta, 1e-6)
        return self.jump_land_fire.float() * self.jump_takeoff_quality * compress.clamp(0.0, 1.0)

    def _reward_jump_lateral_drift(self):
        scale = float(getattr(self.cfg.rewards, "jump_land_lateral_scale", 0.08))
        drift = torch.square(self.jump_land_lateral_disp / max(scale, 1e-6)).clamp(0.0, 4.0)
        return self.jump_land_fire.float() * drift

    def _reward_jump_land_leg_sync(self):
        vz = self.root_states[self.actor_indices, 9]
        descending_or_landed = (((self.jump_state == 2) & (vz < 0.0)) | (self.jump_state == 3)).float()
        foot_h_err = torch.abs(self.foot_heights[:, 0] - self.foot_heights[:, 1])
        fz = self.contact_forces[:, self.feet_indices, 2].clamp(min=0.0)
        force_imb = torch.abs(fz[:, 0] - fz[:, 1]) / (fz[:, 0] + fz[:, 1]).clamp(min=1.0)
        return descending_or_landed * (torch.square(foot_h_err / 0.05) + 0.5 * torch.square(force_imb)).clamp(0.0, 4.0)

    # ================= 前跳(立定跳远, use_jump_distance) =================
    def _jump_dist_long_gain(self):
        hi_d = max(float(self.cfg.commands.jump_dist_cmd_range[1]), 1e-6)
        gain = float(getattr(self.cfg.rewards, "jump_dist_long_gain", 0.0))
        d_norm = (self.jump_dist_cmd[:, 0] / hi_d).clamp(0.0, 1.0)
        return 1.0 + gain * d_norm * d_norm

    def _reward_jump_land_dist(self):
        # Once-per-jump wheel-midpoint displacement from liftoff to touchdown.
        sigma = float(getattr(self.cfg.rewards, "jump_dist_track_sigma", 0.01))
        r = torch.exp(-(self.jump_land_disp - self.jump_dist_cmd[:, 0]) ** 2 / sigma)
        return r * self._jump_dist_long_gain() * self.jump_land_fire.float() * self.jump_takeoff_quality

    def _reward_jump_dist_error(self):
        scale = float(getattr(self.cfg.rewards, "jump_dist_error_scale", 0.05))
        error = torch.square((self.jump_land_disp - self.jump_dist_cmd[:, 0]) / max(scale, 1e-6))
        return error.clamp(0.0, 4.0) * self.jump_land_fire.float() * self.jump_takeoff_quality

    def _reward_jump_est_dist(self):
        # Predict touchdown of the same wheel-midpoint quantity used by the event reward.
        wheel_mid = self.foot_positions.mean(dim=1)
        wheel_vel = self.foot_velocities.mean(dim=1)
        clearance = self.foot_heights.min(dim=1).values
        vz = wheel_vel[:, 2]
        t_land = ((vz + torch.sqrt((vz ** 2 + 2 * 9.81 * clearance).clamp(min=0.0))) / 9.81).clamp(0.0, 1.0)
        vfwd = (wheel_vel[:, :2] * self.jump_takeoff_dir).sum(dim=1)
        cur_disp = ((wheel_mid[:, :2] - self.jump_takeoff_pos) * self.jump_takeoff_dir).sum(dim=1)
        x_hat = (cur_disp + vfwd * t_land).clamp(-1.0, 2.0)                                 # clamp防数值spike->崩溃
        infl = ((self.jump_state == 2) & (~self.jumped_flag)).float()
        sigma = float(getattr(self.cfg.rewards, "jump_dist_track_sigma", 0.01))
        return torch.exp(-(x_hat - self.jump_dist_cmd[:, 0]) ** 2 / sigma) * self._jump_dist_long_gain() * infl * self.jump_takeoff_quality

    def _reward_jump_fwd_vel(self):
        # Match wheel-midpoint flight velocity to the command's nominal ballistic velocity.
        # 破"原地竖直跳局部最优": 直接奖励起跳带上正确前向速度(est_dist太间接)。
        h_star = self.jump_height_cmd[:, 0].clamp(min=0.05)
        T_flight = 2.0 * torch.sqrt(2.0 * h_star / 9.81)                     # 抛体滞空时间
        v_des = (self.jump_dist_cmd[:, 0] / T_flight.clamp(min=0.15)).clamp(max=3.0)
        world_vxy = self.foot_velocities.mean(dim=1)[:, :2]
        vfwd = (world_vxy * self.jump_takeoff_dir).sum(dim=1)               # 沿起跳朝向前向速度
        infl = ((self.jump_state == 2) & (~self.jumped_flag)).float()
        return torch.exp(-(vfwd - v_des) ** 2 / 0.2) * self._jump_dist_long_gain() * infl * self.jump_takeoff_quality

    def _reward_jump_wheel_spin(self):
        # 防滚: 仅 FLIGHT 惩罚轮自转(空中两轮应静止)。★不罚 CROUCH: 蹬地起跳时轮会随base前移被动滚, 罚它=压死起跳。
        active = (self.jump_state == 2).float()
        return (self.dof_vel[:, [3, 7]] ** 2).sum(dim=1) * active

    def _reward_jump_ground_roll(self):
        # 防漂移: 仅 IDLE(待命)惩罚 base 水平速度, 保持起跳前静止。
        # ★不再罚 CROUCH: 前跳的蹬地起跳必须在CROUCH段建立前向速度, 罚它=压死起跳。CROUCH防滚交给 roll-term + 跳距只算腾空段。
        pre = (self.jump_state == 0).float()
        return (self.base_lin_vel[:, :2] ** 2).sum(dim=1) * pre

    def _reward_jump_land_brake(self):
        # ★落地刹停(LANDED): base零速 + 两轮锁死。防落地滚走, 保落点精度。
        landed = (self.jump_state == 3).float()
        base_still = torch.exp(-(torch.norm(self.base_lin_vel, dim=1) ** 2) / 0.1)
        wheel_still = torch.exp(-((self.dof_vel[:, [3, 7]] ** 2).sum(dim=1)) / 4.0)
        return landed * base_still * wheel_still

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
        # step/roll decomposition: sample the commanded STEPPING-velocity component
        _sr = self.cfg.commands.cmd_step_range
        cs = torch.rand(len(env_ids), device=self.device) * (_sr[1] - _sr[0]) + _sr[0]
        # Unify roll+step in ONE policy: force a fraction of envs to pure-roll (cmd_step=0) each resample,
        # so the policy learns to ROLL on command (cmd_step~0) and STEP on command (cmd_step>0), instead of
        # always stepping. The stepping-shaping rewards are gated by (cmd_step>0.05) so pure-roll envs get
        # no stepping incentive. cmd_step_zero_prob=0.0 -> old behavior (always some stepping).
        p_roll = float(getattr(self.cfg.commands, "cmd_step_zero_prob", 0.0))
        if p_roll > 0.0:
            roll_mask = torch.rand(len(env_ids), device=self.device) < p_roll
            cs = torch.where(roll_mask, torch.zeros_like(cs), cs)
        self.cmd_step[env_ids, 0] = cs
        self._apply_scenario_commands(env_ids)   # co-design: override cmd per scenario (no-op if partition off)
        if bool(
            getattr(self.cfg.scenario, "physical_partition", False)
        ):
            # Physical capability rows are fixed trials. The legacy sampler
            # below injects a random standstill subset after scenario override,
            # which would silently corrupt the row's commanded difficulty.
            return
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
        # 20:end 全部 0 noise（torques / QS features / prev_actions）。
        # 即使 use_torques_in_obs / use_qs_in_obs 关掉，对应区段在 obs_buf 里消失，
        # 这里的越界 slice 会被 PyTorch silently clamp，行为仍然正确（zeros_like 起底）。
        noise_vec[20:28] = 0.0  # raw torques (if use_torques_in_obs)
        noise_vec[28:36] = 0.0  # QS load estimation features (if use_qs_in_obs)
        noise_vec[36:40] = 0.0  # QS residual baseline (if use_qs_in_obs)
        noise_vec[40:] = 0.0    # previous actions
        return noise_vec

    def _init_motor_power_envelope(self):
        domain_rand = self.cfg.domain_rand
        self.motor_power_envelope_enabled = bool(
            getattr(domain_rand, "motor_power_envelope", False)
        )
        self.motor_specific_power_w_per_kg = float(
            getattr(domain_rand, "motor_specific_power_w_per_kg", 450.0)
        )
        self.motor_power_exponent = float(
            getattr(domain_rand, "motor_power_exponent", 1.0)
        )
        self.motor_power_min_speed_rad_s = float(
            getattr(domain_rand, "motor_power_min_speed_rad_s", 1.0e-3)
        )
        if self.motor_specific_power_w_per_kg <= 0.0:
            raise ValueError("motor_specific_power_w_per_kg must be positive")
        if self.motor_power_exponent <= 0.0:
            raise ValueError("motor_power_exponent must be positive")
        if self.motor_power_min_speed_rad_s <= 0.0:
            raise ValueError("motor_power_min_speed_rad_s must be positive")

        nominal_masses = tuple(
            float(value)
            for value in getattr(
                domain_rand,
                "motor_nominal_masses_kg",
                (1.35, 1.35, 0.89, 0.365),
            )
        )
        if len(nominal_masses) != len(self.motor_group_names) or any(
            value <= 0.0 for value in nominal_masses
        ):
            raise ValueError(
                "motor_nominal_masses_kg must contain four positive values"
            )

        group_by_dof = []
        for dof_name in self.dof_names:
            matches = [
                group_id
                for group_id, group_name in enumerate(self.motor_group_names)
                if group_name in dof_name.lower()
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"DOF {dof_name!r} must map to exactly one motor group"
                )
            group_by_dof.append(matches[0])
        nominal_mass_by_dof = torch.as_tensor(
            [nominal_masses[group_id] for group_id in group_by_dof],
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        self.motor_nominal_mass_by_dof_kg = nominal_mass_by_dof
        self.motor_peak_torque_limits = (
            self.torque_limits.view(1, -1) * self.motor_torque_limit_scale
        )
        self.motor_power_limits_w = (
            nominal_mass_by_dof
            * self.motor_specific_power_w_per_kg
            * torch.pow(
                self.motor_torque_limit_scale,
                self.motor_power_exponent,
            )
        )
        self.motor_effective_torque_limits = self.motor_peak_torque_limits
        self.motor_power_limited_mask = torch.zeros_like(
            self.motor_peak_torque_limits,
            dtype=torch.bool,
        )
        if self.motor_power_envelope_enabled:
            print(
                "[motor-power] enabled constant-torque/constant-power envelope "
                f"specific_power={self.motor_specific_power_w_per_kg:g} W/kg "
                f"power_exponent={self.motor_power_exponent:g}"
            )

    def _init_buffers(self):
        super()._init_buffers()
        self._init_motor_power_envelope()
        self.wheel_lin_vel = torch.zeros_like(self.foot_velocities)
        self.cmd_step = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        if getattr(self.cfg.env, "use_jump", False):
            self.cmd_jump = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)          # 锁存位(obs)
            self.jump_height_cmd = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)    # 指定跳高目标 h*(obs, m)
            self.jump_state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)             # 0 IDLE/1 CROUCH/2 FLIGHT/3 LANDED
            self.jump_trigger_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)     # 下次触发时刻
            self.jump_phase_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)       # 当前相开始时刻(超时用)
            self.jump_peak_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)      # 本跳两轮最低净空的峰值
            self.jump_prev_vz = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)          # 上步竖直速度(apex过零判)
            self.jumped_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)            # 本latch是否已结算apex
            self.jump_apex_fire = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)         # 本步是否结算apex(reward读)
            self.jump_landed_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)       # 本latch是否已落地过(判二次起飞)
            self.jump_air_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)        # 连续腾空步数(区分落地弹跳与真二次起飞)
            self.jump_takeoff_fire = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)       # 本步是否起飞(真实蹬腿质量奖励读)
            self.jump_init_base_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_min_base_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_init_leg_len = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_prev_leg_len = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_min_leg_len = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_leg_len = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_leg_gain = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_height_gain = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_leg_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_crouch_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_quality = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_bad_takeoff_fire = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.jump_timeout_fire = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.jump_takeoff_roll = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_lat_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_fwd_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_lat_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_vz = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_base_vz = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_angle = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_leg_diff = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_leg_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_leg_right = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_foot_x_diff = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_foot_z_diff = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_abad_mirror_err = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_abad_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_abad_right = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_hip_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_hip_right = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_knee_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_knee_right = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_takeoff_yaw_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_max_roll = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_max_lat_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_max_yaw_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_max_lat_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_min_flight_leg_len = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_preland_leg_len = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.jump_land_leg_len = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            if getattr(self.cfg.env, "use_jump_distance", False):   # ★前跳(立定跳远)增量缓冲
                self.jump_dist_cmd = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)    # 目标前跳距离 d*(obs)
                self.jump_start_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)   # 触发时轮心中点 xy(滚而不跳终止用)
                self.jump_takeoff_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device) # 起飞瞬间轮心中点 xy
                self.jump_takeoff_dir = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device) # 起飞前向单位向量(投影用)
                self.jump_land_disp = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)      # 落地前向有效跳距
                self.jump_land_lateral_disp = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # 落地横向位移(+为左)
                self.jump_land_fire = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)       # 本步是否结算落点(reward读)
                self.jump_land_rewarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)   # 每个 latch 只允许一次落点奖励
        self.foot_stance_steps = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.v_roll = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.v_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
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
            getattr(self.cfg.asset, "load_estimation_filter_cutoff_normalized", 1.0 / 1000.0)
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
        encoder_extra_dims = (
            self.num_actions
            if bool(getattr(self.cfg.env, "encoder_include_torques", False))
            else 0
        )
        self.encoder_obs_history = torch.zeros(
            self.num_envs,
            (
                self.num_obs
                + self.filtered_size
                + encoder_extra_dims
            )
            * self.obs_history_length,
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

    # ---- Stepping-gait rewards (PF-style contact schedule; active only when
    #      use_gait_stepping=True populates self.desired_contact_states) ----
    def _reward_tracking_contacts_shaped_force(self):
        # Penalize ground contact force during the SWING phase (desired_contact=0).
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
        else:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * (
                    1 - torch.exp(-foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))
        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        # Reward foot stillness during the STANCE phase (desired_contact=1).
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)
        else:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * (
                    1 - torch.exp(-foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma))
        return reward / len(self.feet_indices)

    def _reward_feet_regulation(self):
        # Penalize planar foot velocity while the foot is on the ground.
        feet_height = (
            self._base_height_target_tensor().unsqueeze(-1) * 0.001
        )
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_foot_landing_vel(self):
        # Penalize downward velocity of a foot about to land (soft touchdown).
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_foot_clearance(self):
        # Reward the SWING foot (desired_contact~0) for lifting toward feet_height_target.
        # This is the direct "pick your foot up" drive that forces real stepping (vs rolling).
        target = self.cfg.rewards.feet_height_target
        reward = 0
        for i in range(len(self.feet_indices)):
            swing = 1.0 - self.desired_contact_states[:, i]
            clearance = torch.clip(self.foot_heights[:, i], 0.0, target) / target
            reward += swing * clearance
        return reward / len(self.feet_indices)

    def _reward_wheel_vel(self):
        # Penalize wheel spin -> removes the free-rolling shortcut so the robot must STEP to move.
        # Wheels stay UNLOCKED (velocity-mode, can still spin passively); this only discourages
        # actively driving them for propulsion. Wheel DOFs are indices 3 (wheel_L) and 7 (wheel_R).
        return torch.sum(torch.square(self.dof_vel[:, [3, 7]]), dim=1)

    def _reward_swing_phase(self):
        # Phase-gated ALTERNATION: reward the LEFT foot lifting during gait phase [0,0.5) and the
        # RIGHT foot during [0.5,1). Imposes which foot swings when -> forces genuine left/right
        # alternation (the emergent gait had no timing, so one foot dominated = limp). Soft: rewards
        # the correct foot's swing clearance; no harsh contact penalty. Needs use_gait_phase=True so
        # self.gait_indices advances.
        _sht = getattr(self, "swing_height_target", None)   # per-env target (obstacle scenario curriculum)
        if _sht is None:
            _sht = self.foot_heights.new_full((self.num_envs,), float(self.cfg.rewards.feet_swing_height_target))
        target = _sht   # always a [num_envs] tensor now (clip needs matched min/max types)
        left_should = (self.gait_indices < 0.5).float()
        clr_L = torch.minimum(self.foot_heights[:, 0].clamp(min=0.0), target) / target
        clr_R = torch.minimum(self.foot_heights[:, 1].clamp(min=0.0), target) / target
        step_gate = (self.cmd_step[:, 0] > 0.05).float()   # unify roll+step: only demand stepping when commanded
        return (left_should * clr_L + (1.0 - left_should) * clr_R) * step_gate

    def _reward_weight_shift(self):
        # Teach the lateral WEIGHT TRANSFER needed to lift a foot: when the LEFT foot should swing
        # (gait phase < 0.5), reward the vertical CONTACT FORCE being on the RIGHT foot; when the RIGHT
        # foot should swing, reward force on the LEFT. Defined via contact force so there is no left/right
        # sign ambiguity. This is the missing skill for 2-wheel alternating stepping (phase timing alone
        # made it fall; this directly rewards shifting the body weight onto the intended stance foot).
        fL = self.contact_forces[:, self.feet_indices[0], 2].clamp(min=0.0)
        fR = self.contact_forces[:, self.feet_indices[1], 2].clamp(min=0.0)
        total = (fL + fR).clamp(min=1.0)
        right_frac = fR / total
        left_should_swing = (self.gait_indices < 0.5).float()
        step_gate = (self.cmd_step[:, 0] > 0.05).float()   # gate off when rolling (cmd_step~0)
        return (left_should_swing * right_frac + (1.0 - left_should_swing) * (1.0 - right_frac)) * step_gate

    def _reward_feet_air_time(self):
        # Reward each foot for a proper SWING (airborne >= target) between contacts, scored at
        # touchdown. A foot that never lifts earns 0; a foot that only "paws" (air_time < target)
        # is PENALIZED (air_time-target < 0). Maximizing forces BOTH feet to take real alternating
        # swings -> genuine stepping (kills the "one foot planted + one foot pawing" cheat).
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew = torch.sum((self.feet_air_time - self.cfg.rewards.feet_air_time_target) * first_contact, dim=1)
        stepping = self.cmd_step[:, 0] > 0.05   # only reward swings when stepping is commanded (unify roll+step)
        rew *= stepping.float()
        self.feet_air_time *= ~contact_filt
        return rew

    def _reward_stance_symmetry(self):
        # Penalize one foot dominating stance over the episode (the "one foot always planted" cheat).
        # Tracks cumulative stance steps per foot; penalty = normalized |left_stance - right_stance|.
        contact = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
        self.foot_stance_steps += contact
        total = self.foot_stance_steps.sum(dim=1).clamp(min=1.0)
        imbalance = torch.abs(self.foot_stance_steps[:, 0] - self.foot_stance_steps[:, 1]) / total
        step_gate = (self.cmd_step[:, 0] > 0.05).float()   # L/R stance balance only matters when stepping
        return imbalance * step_gate

    def _compute_step_roll_vel(self):
        # Decompose forward base velocity into WHEEL-ROLLING vs LEG-STEPPING parts.
        # v_roll = contact-weighted mean wheel rolling speed (omega * radius);
        # v_step = base_vx - v_roll (the part the body gains by the legs vaulting over planted feet).
        r = self.cfg.asset.foot_radius
        sign = getattr(self.cfg.commands, "wheel_roll_sign", 1.0)
        wheel_w = self.dof_vel[:, [3, 7]]                                        # [N,2]
        v_roll_per = wheel_w * r * sign                                          # [N,2]
        contact = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()   # [N,2]
        csum = contact.sum(dim=1)
        v_roll = torch.where(
            csum > 0,
            (v_roll_per * contact).sum(dim=1) / csum.clamp(min=1.0),
            v_roll_per.mean(dim=1),
        )
        self.v_roll = v_roll
        self.v_step = self.base_lin_vel[:, 0] - v_roll

    def _reward_tracking_step_vel(self):
        # Track the commanded STEPPING velocity component.
        err = torch.square(self.cmd_step[:, 0] - self.v_step)
        return torch.exp(-err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_roll_vel(self):
        # Track the commanded ROLLING velocity component (commands[:,0] = cmd_roll).
        err = torch.square(self.commands[:, 0] - self.v_roll)
        return torch.exp(-err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_total_vel(self):
        # Physical capability protocol: total forward target is roll + step.
        target = self.commands[:, 0] + self.cmd_step[:, 0]
        err = torch.square(target - self.base_lin_vel[:, 0])
        sigma = float(
            getattr(
                self.cfg.rewards,
                "total_vel_tracking_sigma",
                self.cfg.rewards.tracking_sigma,
            )
        )
        return torch.exp(-err / max(sigma, 1.0e-6))

    def _reward_total_vel_error(self):
        # Retain a useful gradient when high-speed commands are far outside the
        # resumed policy's initial response and the Gaussian reward saturates.
        target = self.commands[:, 0] + self.cmd_step[:, 0]
        return torch.square(target - self.base_lin_vel[:, 0]).clamp(max=4.0)

    def _reward_no_fly(self):
        # Reward exactly ONE foot in contact (single support = mid-step).
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        single = (torch.sum(contact.float(), dim=1) == 1)
        step_gate = (self.cmd_step[:, 0] > 0.05).float()   # single-support only wanted while stepping; at roll want both down
        return single.float() * step_gate

    def _reward_no_jump(self):
        # Reward BOTH feet in contact (prevents both-feet-off / hopping).
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        both = (torch.sum(contact.float(), dim=1) == 2)
        return both.float()

    def _reward_feet_swing_height(self):
        # Penalize a SWING foot (not in contact) for deviating from target clearance -> forces lift.
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        _sht = getattr(self, "swing_height_target", None)   # per-env target (obstacle scenario curriculum)
        target = _sht.unsqueeze(-1) if _sht is not None else self.cfg.rewards.feet_swing_height_target
        err = torch.square(self.foot_heights - target) * (~contact).float()
        step_gate = (self.cmd_step[:, 0] > 0.05).float()   # don't demand swing clearance when rolling
        return torch.sum(err, dim=1) * step_gate

    def _reward_nominal_foot_position(self):
        #1. calculate foot postion wrt base in base frame  
        nominal_base_height = -(
            self._base_height_target_tensor()
            - self.cfg.asset.foot_radius
        )
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
        full = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        if not getattr(self.cfg.env, "use_jump_distance", False):
            return full
        active = ((self.cmd_jump[:, 0] > 0.5) | (self.jump_state != 0)).float()
        roll_only = torch.square(self.base_ang_vel[:, 0])
        return (1.0 - active) * full + active * roll_only

    def _reward_orientation(self):
        # Penalize non flat base orientation
        full = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        if not getattr(self.cfg.env, "use_jump_distance", False):
            return full
        active = ((self.cmd_jump[:, 0] > 0.5) | (self.jump_state != 0)).float()
        roll_only = torch.square(self.projected_gravity[:, 1])
        return (1.0 - active) * full + active * roll_only

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1) * self._jump_regularizer_weight()

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1) * self._jump_regularizer_weight()

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1) * self._jump_regularizer_weight()

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * self._jump_regularizer_weight()

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
        return torch.abs(
            base_height - self._base_height_target_tensor()
        )
