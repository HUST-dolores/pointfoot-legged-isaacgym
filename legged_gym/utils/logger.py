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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import torch

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):        
        if isinstance(value, torch.Tensor): 
            value = value.detach().cpu() 
            if value.numel() == 1: 
                value = value.item() 
            else: 
                value = value.numpy() 

        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 4
        nb_cols = 4
        fig, axs = plt.subplots(nb_rows, nb_cols)
                # 计算要跳过的数据点数（前0.5秒）
        skip_points = int(0.5 / self.dt)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]:
            a.plot(time, log["dof_pos"], label="measured")
        if log["dof_pos_target"]:
            a.plot(time, log["dof_pos_target"], label="target")
        a.set(xlabel="time [s]", ylabel="Position [rad]", title="DOF Position")
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]:
            a.plot(time, log["dof_vel"], label="measured")
        if log["dof_vel_target"]:
            a.plot(time, log["dof_vel_target"], label="target")
        a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity")
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]:
            bx = np.array(log["base_vel_x"])
            if len(bx) > skip_points:
                bx = bx[skip_points:]
                t_bx = np.linspace(0.5, len(log["base_vel_x"]) * self.dt, len(bx))
            else:
                t_bx = time
            a.plot(t_bx, bx, label="measured")
        if log["command_x"]:
            cx = np.array(log["command_x"])
            if len(cx) > skip_points:
                cx = cx[skip_points:]
                t_cx = np.linspace(0.5, len(log["command_x"]) * self.dt, len(cx))
            else:
                t_cx = time
            a.plot(t_cx, cx, label="commanded")
        if log["est_lin_vel_x"]:
            ex = np.array(log["est_lin_vel_x"])
            if len(ex) > skip_points:
                ex = ex[skip_points:]
                t_ex = np.linspace(0.5, len(log["est_lin_vel_x"]) * self.dt, len(ex))
            else:
                t_ex = time
            a.plot(t_ex, ex, label="est")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity x")
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]:
            by = np.array(log["base_vel_y"])
            if len(by) > skip_points:
                by = by[skip_points:]
                t_by = np.linspace(0.5, len(log["base_vel_y"]) * self.dt, len(by))
            else:
                t_by = time
            a.plot(t_by, by, label="measured")
        if log["command_y"]:
            cy = np.array(log["command_y"])
            if len(cy) > skip_points:
                cy = cy[skip_points:]
                t_cy = np.linspace(0.5, len(log["command_y"]) * self.dt, len(cy))
            else:
                t_cy = time
            a.plot(t_cy, cy, label="commanded")
        if log["est_lin_vel_y"]:
            ey = np.array(log["est_lin_vel_y"])
            if len(ey) > skip_points:
                ey = ey[skip_points:]
                t_ey = np.linspace(0.5, len(log["est_lin_vel_y"]) * self.dt, len(ey))
            else:
                t_ey = time
            a.plot(t_ey, ey, label="est")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity y")
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]:
            a.plot(time, log["base_vel_yaw"], label="measured")
        if log["command_yaw"]:
            a.plot(time, log["command_yaw"], label="commanded")
        a.set(
            xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base velocity yaw"
        )
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]:
            a.plot(time, log["base_vel_z"], label="measured")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity z")
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            if len(forces) > skip_points:
                # 跳过前0.5秒的数据
                forces_skipped = forces[skip_points:]
                # 生成对应的时间轴
                time_skipped = np.linspace(0.5, len(forces) * self.dt, len(forces_skipped))
                for i in range(forces_skipped.shape[1]):
                    a.plot(time_skipped, forces_skipped[:, i], label=f"force {i}")
            else:
                # 如果数据不够长，使用全部数据
                for i in range(forces.shape[1]):
                    a.plot(time, forces[:, i], label=f"force {i}")
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f"force {i}")
        a.set(xlabel="time [s]", ylabel="Forces z [N]", title="Vertical Contact forces")
        a.legend()
        # plot torque/vel curves
        a = axs[2, 3]
        # if log["dof_vel"] != [] and log["dof_torque"] != []:
        #     a.plot(log["dof_vel"], log["dof_torque"], "x", label="measured")
        # a.set(
        #     xlabel="Joint vel [rad/s]",
        #     ylabel="Joint Torque [Nm]",
        #     title="Torque/velocity curves",
        # )
        if log["torque_knee_L"]:
            a.plot(time, log["torque_knee_L"], label="torque_knee_L")
        if log["torque_knee_R"]:
            a.plot(time, log["torque_knee_R"], label="torque_knee_R")
        a.set(xlabel="time [s]", ylabel="power [w]", title="torque_i_care")
        a.legend()
        # plot torques
        a = axs[2, 2]
        if log["dof_torque"] != []:
            a.plot(time, log["dof_torque"], label="measured")
        a.set(xlabel="time [s]", ylabel="Joint Torque [Nm]", title="Torque")
        a.legend()
        # plot mass     
        a = axs[3, 3]
        if log["mass"]:
            a.plot(time, log["mass"], label="Actual")
        if log["mass_est"]:
            a.plot(time, log["mass_est"], label="Estimated")
        a.set(xlabel="time [s]", ylabel="mass [kg]", title="mass")
        a.legend()
        
        a = axs[3, 0]
        if log["CoM_x"]:
            a.plot(time, log["CoM_x"], label="Actual")
        if log["CoM_est_x"]:
            a.plot(time, log["CoM_est_x"], label="Estimated")
        a.set(xlabel="time [s]", ylabel="CoM_x [m]", title="CoM_x")
        a.legend()

        a = axs[3, 1]
        if log["CoM_y"]:
            a.plot(time, log["CoM_y"], label="Actual")
        if log["CoM_est_y"]:
            a.plot(time, log["CoM_est_y"], label="Estimated")
        a.set(xlabel="time [s]", ylabel="CoM_y [m]", title="CoM_y")
        a.legend()
        
        a = axs[3, 2]
        if log["CoM_z"]:
            a.plot(time, log["CoM_z"], label="Actual")
        if log["CoM_est_z"]:
            a.plot(time, log["CoM_est_z"], label="Estimated")
        a.set(xlabel="time [s]", ylabel="CoM_z [m]", title="CoM_z")
        a.legend()
        
        a = axs[0, 3]
        if log["extra_loss_vel"]:
            a.plot(time, log["extra_loss_vel"], label="extra_loss_vel")
        a.set(xlabel="time [s]", ylabel="loss", title="extra_loss_vel")
        a.legend()
        
        a = axs[1, 3]
        if log["extra_loss_mass"]:
            a.plot(time, log["extra_loss_mass"], label="extra_loss_mass")
        a.set(xlabel="time [s]", ylabel="loss", title="extra_loss_mass")
        a.legend()
        
        a = axs[2, 1]
        if log["extra_loss_com"]:
            a.plot(time, log["extra_loss_com"], label="extra_loss_com")
        a.set(xlabel="time [s]", ylabel="loss", title="extra_loss_com")
        a.legend()
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
