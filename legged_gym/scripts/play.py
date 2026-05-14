# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    export_mlp_as_onnx,
    task_registry,
    Logger,
)

import numpy as np
import torch
import matplotlib.pyplot as plt


def play(args):
    # 单关节测试模式：通过位置控制让指定关节缓慢旋转，观察动力学
    test_joint_mode = getattr(args, 'test_joint_mode', False)
    test_joint_name = getattr(args, 'test_joint_name', None)
    test_joint_amplitude = getattr(args, 'test_joint_amplitude', 0.3)  # rad
    test_joint_period = getattr(args, 'test_joint_period', 4.0)  # seconds
    test_joint_offset = getattr(args, 'test_joint_offset', 0.0)  # rad
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 60
    env_cfg.env.num_envs = min(env_cfg.env.num_envs,1)

    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    # env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.terrain.terrain_proportions = [0, 1, 0, 0, 0]
    env_cfg.terrain.max_init_terrain_level = 2
    env_cfg.terrain.curriculum = True

    # Trimesh can crash silently when the mesh is too large for GPU PhysX.
    # Use a lighter terrain map in play mode for debugging.
    if env_cfg.terrain.mesh_type == "trimesh":
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 10
        env_cfg.terrain.terrain_length = 6.0
        env_cfg.terrain.terrain_width = 6.0
        env_cfg.terrain.border_size = 8
        env_cfg.terrain.max_init_terrain_level = 1
        print(
            "[PLAY] Using reduced trimesh size for stability: "
            f"rows={env_cfg.terrain.num_rows}, cols={env_cfg.terrain.num_cols}, "
            f"length={env_cfg.terrain.terrain_length}, width={env_cfg.terrain.terrain_width}, "
            f"border={env_cfg.terrain.border_size}"
        )

    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 3
    env_cfg.domain_rand.push_curriculum = False
    env_cfg.domain_rand.max_push_vel_xy = 2.0
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.domain_rand.load_debug = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if hasattr(env, "terrain_types"):
        terrain_types_cpu = env.terrain_types.detach().cpu()
        unique_types, counts = torch.unique(terrain_types_cpu, return_counts=True)
        per_type = {int(t.item()): int(c.item()) for t, c in zip(unique_types, counts)}
        print(f"[PLAY] terrain_types(count by type id): {per_type}")

        smooth_cnt = int(((terrain_types_cpu >= 0) & (terrain_types_cpu < 2)).sum().item())
        rough_cnt = int(((terrain_types_cpu >= 2) & (terrain_types_cpu < 4)).sum().item())
        stairs_up_cnt = int(((terrain_types_cpu >= 4) & (terrain_types_cpu < 11)).sum().item())
        stairs_down_cnt = int(((terrain_types_cpu >= 11) & (terrain_types_cpu < 16)).sum().item())
        discrete_cnt = int(((terrain_types_cpu >= 16) & (terrain_types_cpu < 20)).sum().item())
        print(
            "[PLAY] terrain group counts: "
            f"smooth={smooth_cnt}, rough={rough_cnt}, stairs_up={stairs_up_cnt}, "
            f"stairs_down={stairs_down_cnt}, discrete={discrete_cnt}"
        )

    # get robot_type
    robot_type = os.getenv("ROBOT_TYPE")
    commands_val = to_torch([0.5, 0.0, 0, 0], device=env.device) if robot_type.startswith("PF")\
        else to_torch([0, 0.0, 0.0], device=env.device) if robot_type == "WF_TRON1A" else to_torch([0, 0.0, 0.0, 0.0, 0.0])
    action_scale = env.cfg.control.action_scale_pos if robot_type == "WF_TRON1A"\
        else env.cfg.control.action_scale
    obs, obs_history, commands, critic_obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    # train_cfg.runner.checkpoint = -1

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            path,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            path,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    
    # 单关节测试模式初始化
    test_joint_idx = None
    if test_joint_mode and test_joint_name:
        if test_joint_name in env.dof_names:
            test_joint_idx = env.dof_names.index(test_joint_name)
            print(f"[TEST-MODE] Enabled single-joint test mode: {test_joint_name} (DOF #{test_joint_idx})")
            print(
                f"[TEST-MODE] Amplitude={test_joint_amplitude:.3f} rad, "
                f"Offset={test_joint_offset:.3f} rad, Period={test_joint_period:.3f} s"
            )
            print(f"[TEST-MODE] All joints at neutral (0.0), {test_joint_name} will sweep.")
        else:
            print(f"[TEST-MODE] ERROR: Joint '{test_joint_name}' not found in dof_names: {env.dof_names}")
            test_joint_mode = False
    
    # 关节方向调试：用于确认 dof_pos 的零位与正方向
    debug_hip_sign = True
    debug_hip_every = 20
    hip_l_idx = env.dof_names.index("hip_L_Joint") if "hip_L_Joint" in env.dof_names else None
    hip_r_idx = env.dof_names.index("hip_R_Joint") if "hip_R_Joint" in env.dof_names else None
    prev_theta_l = None
    prev_theta_r = None
    if debug_hip_sign and (hip_l_idx is not None) and (hip_r_idx is not None):
        print(
            "[HIP-DEBUG] dof_pos is raw joint position from simulator. "
            "URDF axis: hip_L=[0,1,0], hip_R=[0,-1,0]."
        )
    stop_state_log = 2000  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1.0, 1.0, 0.0])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    est = None
    for i in range(10 * int(env.max_episode_length)):
        if test_joint_mode and test_joint_idx is not None:
            # 单关节测试模式：使用位置控制，只让被测试的关节旋转
            # 计算时间和目标角度（正弦波）
            time_s = i * env.dt
            phase = 2.0 * np.pi * time_s / test_joint_period
            target_angle = test_joint_offset + test_joint_amplitude * np.sin(phase)
            
            # 创建动作张量：所有关节目标位置都是 0.0（中间位置），除了被测试的关节
            actions = torch.zeros((env.num_envs, env.num_dofs), device=env.device, dtype=torch.float32)
            actions[:, test_joint_idx] = target_angle / action_scale  # 需要除以 action_scale
            
            if (i % 100) == 0:
                print(f"[TEST-MODE] step={i} time={time_s:.2f}s target_angle={target_angle:.4f} rad phase={phase:.2f}")
        else:
            # 正常策略推理模式
            est = encoder(obs_history, obs)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())

        env.commands[:, :] = commands_val

        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(
            actions.detach()
        )
        if debug_hip_sign and (hip_l_idx is not None) and (hip_r_idx is not None):
            theta_l = env.dof_pos[robot_index, hip_l_idx].item()
            theta_r = env.dof_pos[robot_index, hip_r_idx].item()
            if prev_theta_l is None:
                dtheta_l = 0.0
                dtheta_r = 0.0
            else:
                dtheta_l = theta_l - prev_theta_l
                dtheta_r = theta_r - prev_theta_r
            prev_theta_l = theta_l
            prev_theta_r = theta_r

            # if (i % debug_hip_every) == 0:
            #     print(
            #         f"[HIP-DEBUG] step={i:5d} "
            #         f"theta_l={theta_l:+.5f} rad dtheta_l={dtheta_l:+.5f} | "
            #         f"theta_r={theta_r:+.5f} rad dtheta_r={dtheta_r:+.5f}"
            #     )
        load_est = None
        if hasattr(env, "last_load_estimates"):
            load_est = env.last_load_estimates
        elif hasattr(env, "_compute_load_estimates"):
            load_est = env._compute_load_estimates()
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            target_position[2] = 0
            camera_position = target_position + camera_offset
            # env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            # 实时计算真实负载质量：负载在机体上时才计入
            load_on_body = False
            if hasattr(env, "is_load_on_body"):
                load_on_body_mask = env.is_load_on_body(torch.tensor([robot_index], device=env.device, dtype=torch.long))
                load_on_body = bool(load_on_body_mask[0].item())
                if hasattr(env, "has_load"):
                    load_on_body = load_on_body and bool(env.has_load[robot_index].item())

            if hasattr(env, "base_mass0") and hasattr(env, "load_mass"):
                current_true_load_mass = float(env.load_mass) if load_on_body else 0.0
                current_true_mass = float(env.base_mass0[robot_index].item()) + current_true_load_mass
            elif hasattr(env, "base_mass") and hasattr(env, "base_mass0"):
                current_true_mass = float(env.base_mass[robot_index].item())
                current_true_load_mass = float(
                    current_true_mass - env.base_mass0[robot_index].item()
                )
            else:
                current_true_mass = 0.0
                current_true_load_mass = 0.0

            if load_est is not None:
                payload_mass_est = load_est["payload_mass"][robot_index].item()
                load_y_est = load_est["load_y"][robot_index].item()
                load_x_est = load_est["load_x"][robot_index].item()
                body_mass_for_load_estimation = float(
                    getattr(env.cfg.asset, "load_estimation_body_mass", 9.58)
                )
                body_com0 = env.base_com0[robot_index]
                estimated_total_mass = body_mass_for_load_estimation + payload_mass_est
                estimated_total_mass_safe = max(estimated_total_mass, 1e-6)
                estimated_com_x = (
                    body_mass_for_load_estimation * body_com0[0].item()
                    + payload_mass_est * load_x_est
                ) / estimated_total_mass_safe
                estimated_com_y = (
                    body_mass_for_load_estimation * body_com0[1].item()
                    + payload_mass_est * load_y_est
                ) / estimated_total_mass_safe

                actual_load_x = 0.0
                actual_load_y = 0.0
                if load_on_body and hasattr(env, "actor_indices") and hasattr(env, "load_indices"):
                    base_actor_idx = env.actor_indices[robot_index]
                    load_actor_idx = env.load_indices[robot_index]
                    base_pos = env.root_states[base_actor_idx, 0:3]
                    base_quat = env.root_states[base_actor_idx, 3:7]
                    load_pos = env.root_states[load_actor_idx, 0:3]
                    rel_body = quat_rotate_inverse(
                        base_quat.unsqueeze(0), (load_pos - base_pos).unsqueeze(0)
                    )[0]
                    actual_load_x = rel_body[0].item()
                    actual_load_y = rel_body[1].item()
                actual_total_mass = body_mass_for_load_estimation + current_true_load_mass
                actual_total_mass_safe = max(actual_total_mass, 1e-6)
                actual_com_x = (
                    body_mass_for_load_estimation * body_com0[0].item()
                    + current_true_load_mass * actual_load_x
                ) / actual_total_mass_safe
                actual_com_y = (
                    body_mass_for_load_estimation * body_com0[1].item()
                    + current_true_load_mass * actual_load_y
                ) / actual_total_mass_safe

                est_payload_mass = min(payload_mass_est, 20.0)
                ref_payload_mass = min(current_true_load_mass, 10.0)

                est_load_y = min(load_y_est, 0.5)
                ref_load_y = min(actual_load_y, 0.5)

                est_load_x = min(load_x_est, 0.5)
                ref_load_x = min(actual_load_x, 0.5)

                logger.log_states(
                    {
                        "payload_mass": est_payload_mass,
                        "payload_mass_ref": ref_payload_mass,
                        "load_y": est_load_y,
                        "load_y_ref": ref_load_y,
                        "load_x": est_load_x,
                        "load_x_ref": ref_load_x,
                        "payload_mass_est": payload_mass_est,
                        "payload_mass_actual": current_true_load_mass,
                        "load_x_est": load_x_est,
                        "load_x_actual": actual_load_x,
                        "load_y_est": load_y_est,
                        "load_y_actual": actual_load_y,
                        "robot_mass_est": estimated_total_mass,
                        "robot_mass_actual": actual_total_mass,
                        "robot_com_x_est": estimated_com_x,
                        "robot_com_x_actual": actual_com_x,
                        "robot_com_y_est": estimated_com_y,
                        "robot_com_y_actual": actual_com_y,
                    }
                )
                if (i % 50) == 0:
                    print(
                        f"[PLAY] t={i * env.dt:.2f}s "
                        f"payload_mass={est_payload_mass:.4f}/{ref_payload_mass:.4f} "
                        f"load_x={est_load_x:.4f}/{ref_load_x:.4f} "
                        f"load_y={est_load_y:.4f}/{ref_load_y:.4f}"
                    )

            logger.log_states(
                    {
                        "dof_pos_target": actions[robot_index, joint_index].item() * action_scale,
                        "dof_pos": (
                            env.dof_pos[robot_index, joint_index]
                            - env.raw_default_dof_pos[joint_index]
                        ).item(),
                        "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                        "filtered_dof_vel": env.filtered_obs_buf[robot_index, 12 + joint_index].item() / env.cfg.normalization.obs_scales.dof_vel if hasattr(env, "filtered_obs_buf") else 0.0,
                        "dof_torque": env.torques[robot_index, joint_index].item(),
                        "filtered_dof_torque": env.filtered_obs_buf[robot_index, 20 + joint_index].item() / env.cfg.normalization.obs_scales.torque if hasattr(env, "filtered_obs_buf") else 0.0,
                        "command_x": env.commands[robot_index, 0].item(),
                        "command_y": env.commands[robot_index, 1].item(),
                        "command_yaw": env.commands[robot_index, 2].item(),
                        "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                        "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                        "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                        "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                        "filtered_base_vel_yaw": env.filtered_obs_buf[robot_index, 2].item() / env.cfg.normalization.obs_scales.ang_vel if hasattr(env, "filtered_obs_buf") else 0.0,
                        "power": torch.sum(env.power[robot_index, :]).item(),
                        "joint_pos_abad_L": (env.dof_pos[robot_index, 0] - env.raw_default_dof_pos[0]).item(),
                        "joint_pos_hip_L": (env.dof_pos[robot_index, 1] - env.raw_default_dof_pos[1]).item(),
                        "joint_pos_knee_L": (env.dof_pos[robot_index, 2] - env.raw_default_dof_pos[2]).item(),
                        "joint_pos_wheel_L": (env.dof_pos[robot_index, 3] - env.raw_default_dof_pos[3]).item(),
                        "joint_pos_abad_R": (env.dof_pos[robot_index, 4] - env.raw_default_dof_pos[4]).item(),
                        "joint_pos_hip_R": (env.dof_pos[robot_index, 5] - env.raw_default_dof_pos[5]).item(),
                        "joint_pos_knee_R": (env.dof_pos[robot_index, 6] - env.raw_default_dof_pos[6]).item(),
                        "joint_pos_wheel_R": (env.dof_pos[robot_index, 7] - env.raw_default_dof_pos[7]).item(),
                        "torque_abad_L": env.torques[robot_index, 0].item(),
                        "torque_hip_L": env.torques[robot_index, 1].item(),
                        "torque_knee_L": env.torques[robot_index, 2].item(),
                        "torque_wheel_L": env.torques[robot_index, 3].item(),
                        "torque_abad_R": env.torques[robot_index, 4].item(),
                        "torque_hip_R": env.torques[robot_index, 5].item(),
                        "torque_knee_R": env.torques[robot_index, 6].item(),
                        "torque_wheel_R": env.torques[robot_index, 7].item(),
                        "mass": current_true_mass,
                        "CoM_x": env.base_com[robot_index, 0].item(),
                        "CoM_y": env.base_com[robot_index, 1].item(),
                        "CoM_z": env.base_com[robot_index, 2].item(),
                        "load_on_body": float(load_on_body),
                        # "inertia_xx": env.base_inertia[robot_index, 0].item(),
                        # "inertia_yy": env.base_inertia[robot_index, 1].item(),
                        # "inertia_zz": env.base_inertia[robot_index, 2].item(),   
                        
                        "contact_forces_z": env.contact_forces[
                            robot_index, env.feet_indices, 2
                        ]
                        .cpu()
                        .numpy(),
                    }
                )
            # print(torch.sum(env.power[robot_index, :]).item())
            if est != None:                # 计算并记录 extra_loss 分量（速度 / 质量 / 质心）
                extra_loss_vel = (
                    (est[:, 0:3] - critic_obs[:, 0:3]).pow(2).mean().item()
                )
                extra_loss_mass = (
                    (est[:, 3] - critic_obs[:, 3]).pow(2).mean().item()
                )
                extra_loss_com = (
                    (est[:, 4:7] - critic_obs[:, 4:7]).pow(2).mean().item()
                )
                logger.log_states(
                    {
                        "extra_loss_vel": extra_loss_vel,
                        "extra_loss_mass": extra_loss_mass,
                        "extra_loss_com": extra_loss_com,
                    }
                )
                if (i % 50) == 0:
                    # obs_vec = obs[robot_index].detach().cpu().tolist()
                    # cmd_vec = env.commands[robot_index].detach().cpu().tolist()
                    print(
                        f"[PLAY] t={i * env.dt:.2f}s "
                        f"extra_loss_vel={extra_loss_vel:.6f} "
                        f"extra_loss_mass={extra_loss_mass:.6f} "
                        f"extra_loss_com={extra_loss_com:.6f}"
                    )
                    # print(f"[PLAY] obs[{robot_index}]={obs_vec}")
                    # print(f"[PLAY] commands[{robot_index}]={cmd_vec}")
                logger.log_states(
                    {
                        "est_lin_vel_x": est[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": est[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": est[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,                        
                        "mass_est": est[robot_index, 3].item()
                        / env.cfg.normalization.obs_scales.mass_scale + float(env.base_mass0[robot_index].item()),
                        "CoM_est_x": est[robot_index, 4].item()
                        / env.cfg.normalization.obs_scales.com_scale + float(env.base_com0[robot_index, 0].item()),
                        "CoM_est_y": est[robot_index, 5].item()
                        / env.cfg.normalization.obs_scales.com_scale + float(env.base_com0[robot_index, 1].item()),
                        "CoM_est_z": est[robot_index, 6].item()
                        / env.cfg.normalization.obs_scales.com_scale + float(env.base_com0[robot_index, 2].item()),
                    }
                )
            elif test_joint_mode and test_joint_idx is not None:
                # 单关节测试模式：记录被测试关节的详细数据
                logger.log_states(
                    {
                        "dof_pos_target": actions[robot_index, test_joint_idx].item() * action_scale,
                        "dof_pos": (
                            env.dof_pos[robot_index, test_joint_idx]
                            - env.raw_default_dof_pos[test_joint_idx]
                        ).item(),
                        "dof_vel": env.dof_vel[robot_index, test_joint_idx].item(),
                        "dof_torque": env.torques[robot_index, test_joint_idx].item(),
                    }
                )
        elif i == stop_state_log:
            mat_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name, "exported", "play_data.mat")
            logger.save_to_mat(mat_path)
            logger.plot_states()

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
