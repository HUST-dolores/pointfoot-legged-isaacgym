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
from legged_gym.envs.base.base_config import BaseConfig


class BipedCfgWF(BaseConfig):
    class env:
        # 8GB-class GPU friendly default; increase gradually after stability is confirmed.
        num_envs = 1024
        # Ablation flag #1: True = QS estimates fed into obs as features for actor.
        #                   False = history_only baseline (obs 减 12 维，encoder 从 obs history 学起)。
        use_qs_in_obs = False
        # Ablation flag #2: True = encoder 学残差 (encoder output 加到 QS baseline 上)。
        #                   False = encoder 直接输出绝对 mass/com（即使 QS 在 obs 也不做残差结构）。
        #                   仅当 use_qs_in_obs=True 时才有意义；False+False 等价 history_only。
        use_residual_learning = False
        # Ablation flag #3: True = 8-dim raw joint torques 进入 policy obs（encoder 也通过 history 看到）。
        #                   False = encoder 失去对 raw torques 的直接访问；QS features 仍可在 obs 里（如果 use_qs_in_obs=True）。
        #                   常用组合：(T, T, T) = E1 main；(T, F, T) = E2 direct；(F, N/A, T) = E3 histonly；
        #                            (F, N/A, F) = E4 true history-only；(T, T, F) = E5 QS-only-path（无 raw torques）。
        use_torques_in_obs = False
        _qs_obs_dims = (8 + 4) if use_qs_in_obs else 0
        _torque_obs_dims = 8 if use_torques_in_obs else 0
        # Stepping/gait mode: expose sin/cos gait clock to policy obs (+2) so the policy can
        # phase-lock a stepping gait (PF-style contact-schedule rewards). Set True together with
        # the stepping reward scales. Default off -> baseline unchanged. Wheels stay free-rolling.
        use_gait_stepping = True   # step/roll decomposition: adds cmd_step to obs + stepping rewards
        use_gait_phase = True      # variant A: also add sin/cos gait clock + phase contact-schedule rewards
        _gait_obs_dims = (1 if use_gait_stepping else 0) + (2 if use_gait_phase else 0)
        num_observations = 30 + 6 - 2 - 4 - 2 + _torque_obs_dims + _qs_obs_dims + _gait_obs_dims  # [+2 gait clock if stepping]
        # Co-design: expose per-env leg morphology (thigh,shank scale = 2 dims) to the critic as
        # privileged info. Set True together with domain_rand.randomize_morphology. Default off → no change.
        use_morphology_in_critic = False
        _morph_critic_dims = 2 if use_morphology_in_critic else 0
        # Co-design Path A: expose per-env motor-torque design scale (1 dim) to the critic.
        # Set True together with domain_rand.randomize_motor_design. Default off -> no change.
        use_motor_design_in_critic = False
        _motor_critic_dims = 1 if use_motor_design_in_critic else 0
        num_critic_observations = 7 + _morph_critic_dims + _motor_critic_dims + num_observations
        num_height_samples = 117
        num_actions = 8
        obs_butter_cutoff_hz = 2.0  # 2nd-order Butterworth low-pass cutoff for filtered branch
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 40  # episode length in seconds
        obs_history_length = 10  # number of observations stacked together
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

    class terrain:
        mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        # 用 curriculum=True 走 curiculum() 函数，配合 num_rows=1 → difficulty=0/1=0 (最易)。
        # curriculum=False 会走 randomized_terrain，那里硬编码用 [0.5, 0.75, 0.9] 中高难度。
        curriculum = True
        static_friction = 0.4
        dynamic_friction = 0.4
        restitution = 0.8
        # rough terrain only:
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # 仅第一级
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 1  # 只保留 1 行难度（最易），不再多级
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.5, 0.5, 0, 0, 0]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class commands:
        curriculum = False
        cmd_step_range = [0.0, 0.4]   # [m/s] commanded stepping-velocity component (roll-dominant: keep small)
        cmd_step_zero_prob = 0.0      # unify roll+step: fraction of envs forced to pure-roll (cmd_step=0) each resample. 0=off
        wheel_roll_sign = 1.0         # sign so that +wheel_dof_vel*r == +forward roll (flip to -1.0 if smoke shows anti-correlation)
        smooth_max_lin_vel_x = 2.0
        smooth_max_lin_vel_y = 1.0
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 1.0
        max_ang_vel_yaw = 3.0
        curriculum_threshold = 0.75
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error, only work on adaptive group
        min_norm = 0.1

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [0, 0]  # min max [m/s]
            # lin_vel_x = [-1.7, 1.7]  # min max [m/s]
            # lin_vel_y = [-1.7, 1.7]  # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
            heading = [-3.14159, 3.14159]

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]

        class ranges:
            frequencies = [1.5, 2.5]
            offsets = [0, 1]  # offset is hard to learn
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.0, 0.1]

    class init_state:
        pos = [0.0, 0.0, 0.8 + 0.1664]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }

    class control:
        action_scale_pos = 0.25
        action_scale_vel = 0.5
        control_type = "P"
        stiffness = {
            "abad_L_Joint": 42,
            "hip_L_Joint": 42,
            "knee_L_Joint": 42,
            "abad_R_Joint": 42,
            "hip_R_Joint": 42,
            "knee_R_Joint": 42,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 2.5,
            "hip_L_Joint": 2.5,
            "knee_L_Joint": 2.5,
            "abad_R_Joint": 2.5,
            "hip_R_Joint": 2.5,
            "knee_R_Joint": 2.5,
            "wheel_L_Joint": 0.8,
            "wheel_R_Joint": 0.8,
        }  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        user_torque_limit = 80.0
        max_power = 1000.0  # [W]
        # Co-design: correct leg-joint PD gains for scaled legs via eta(xi) polynomial
        # (a*x^3 + b*x^2 + c*x + d). Applied ONLY when domain_rand.randomize_morphology is on;
        # wheel joints are never touched. Defaults: eta_Kp = xi^2 (joint inertia ~ length^2),
        # eta_Kd = xi. hip joints use thigh scale, knee joints use shank scale, abad uses the mean.
        morphology_pd_correction = True
        morphology_kp_poly = [0.0, 1.0, 0.0, 0.0]
        morphology_kd_poly = [0.0, 0.0, 1.0, 0.0]

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/WF_TRON1A/urdf/robot.urdf"
        name = "wheelfoot_flat"
        foot_name = "wheel"
        foot_radius = 0.127
        load_estimation_thigh_length = 0.30000206  # URDF hip->knee sagittal length
        load_estimation_zero_thigh_angle = 2.0943951023931953  # 120 deg zero pose
        load_estimation_mass_offset = 12.08  # 仅旧版本公式 (Model A) 使用，Model C 不再使用
        load_estimation_body_mass = 9.58
        load_estimation_body_com0 = [0.04576, 0.00014, -0.16398]
        load_estimation_load_z = 0.10
        load_estimation_filter_cutoff_normalized = 1.0 /40  # butter(1, 1/10, "low")  截止频率  1/100几乎直线
        load_estimation_robot_width = 0.251
        load_estimation_position_limit = 0.5
        load_estimation_position_zero_mass_threshold = 0.3
        # Model C 标定参数（2026-05-18 用 fit_load_estimation.py 在 4 个 mass × 16 (x,y) 工况上拟合得到）
        # mass:  m_L = alpha_L * R_L + gamma_L + beta_hip * (theta_lhip - theta_rhip)
        # mass:  m_R = alpha_R * R_R + gamma_R + beta_hip * (theta_lhip - theta_rhip)
        # x:     load_x = (tau_lhip - tau_rhip - T_body_x) / (m * g * cos_pitch) - com_x_bias * tan_pitch + x_offset
        load_estimation_alpha_L = 0.367
        load_estimation_alpha_R = 0.429
        load_estimation_gamma_L = 0.158
        load_estimation_gamma_R = -0.499
        load_estimation_beta_hip = -8.507
        load_estimation_t_body_x = 6.17
        load_estimation_com_x_bias = 0.649  # Model C 拟合值
        # load_estimation_com_x_bias = 0.2632  # 旧值 (0.0932+0.19 前者为髋关节到机体下表面，后者为机体厚度)
        load_estimation_x_offset = -0.037
        # abad-related geometry for load estimation (compensates non-zero abad angle)
        load_estimation_leg_eff_length = 0.55  # vertical distance from abad axis to wheel-ground contact at zero pose
        load_estimation_abad_R_sign = 1.0     # sign of right abad axis relative to left (-1 if URDF axes mirror like hip)
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "knee", "hip"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            False  # Some .obj meshes must be flipped from y-up to z-up
        )

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.6]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = False
        added_mass_range = [-0.1, 0.1]         #
        add_random_load = False  # STEPPING run: isolate walking first (re-enable to add load-carrying later)
        add_load_range = [2, 30]  # [kg] load mass range. With per_env_load_mass=True
                                 # each env independently samples one mass from this
                                 # range at actor creation (encoder sees 1024 distinct
                                 # masses → better OOD generalization than legacy
                                 # single-scalar mode).
        # When True, each env's load actor is created with its own density
        # (mass sampled from add_load_range). When False, all envs share one
        # globally sampled mass (legacy pre-2026-05-23 behavior; only useful
        # for reproducing old baselines).
        per_env_load_mass = True
        randomize_base_com = True
        # load_enable_iter = 1000    #000 
        rand_com_vec = [0.03, 0.02, 0.03]
        # load_offset_range_xy = [0.13, 0.12]   #000
        randomize_inertia = True
        randomize_inertia_range = [0.95, 1.05]
        push_robots = True
        push_interval_s = 5.0
        push_curriculum = True
        push_curriculum_start_iter = 0
        push_curriculum_end_iter = 4000
        push_curriculum_min_vel_xy = 0.5
        load_start_time_s = 0.5     # 机器人开始添加负载的时间（秒）
        # 训练用：随机时序（多样的 load on/off）。Exp A play 时若要固定阶跃,临时改成 [6,6]/[10,10]。
        load_duration_range_s = [3.0, 4.0]  # 负载持续时间随机范围
        load_interval_range_s = [5.0, 6.0]  # 两次负载之间的时间随机范围
        load_contact_grace_s = 0.2  # 负载生成后忽略接触终止的宽限时间（秒）
        # 负载判定滞回（降低 on/off 抖动）
        load_on_body_on_steps = 2
        load_on_body_off_steps = 3
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.0
        rand_force_curriculum_level = 0
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        randomize_imu_offset = True
        randomize_imu_offset_range = [-1.2, 1.2]
        delay_ms_range = [0, 20]
        max_push_vel_xy = 2.0
        # --- Leg-morphology spatial domain randomization (co-design Phase 1) ---
        # When True, each parallel env loads a robot with a different leg geometry, bucketed into
        # morphology_num_buckets distinct URDFs generated from asset.file. Default False → single
        # URDF (existing behavior unchanged). Pair with env.use_morphology_in_critic to feed the
        # per-env (thigh,shank) scale to the critic as privileged info, and control.morphology_pd_correction
        # to adjust PD gains. nominal spawn height per morphology is handled in _reset_root_states.
        randomize_morphology = False
        morphology_num_buckets = 64
        morphology_thigh_scale_range = [0.8, 1.2]
        morphology_shank_scale_range = [0.8, 1.2]
        morphology_regenerate = True   # (re)generate morph_*.urdf at startup from asset.file
        morphology_prefix = "morph"
        morphology_seed = 0            # seed for sampling the bucket scales (reproducible URDF set)
        # --- Motor-torque co-design (Path A): per-env motor "size" = torque-limit scale + mass cost ---
        # When True, each env samples a motor_torque_scale k in motor_torque_scale_range; the joint
        # torque LIMITS scale by k, and per the ENCOS cost curve (legged_gym/utils/motor_cost.py) the
        # extra motor mass is added to each actuated joint's link (a real "no free torque" trade-off).
        # Default False -> nominal motors, behavior unchanged. 1-D design variable for the simplest pipeline.
        randomize_motor_design = False
        motor_torque_scale_range = [0.6, 1.6]
        motor_design_seed = 0

    class scenario:
        # Co-design multi-scenario training (KNOWLEDGE_NOTES §11-12). Each env is assigned ONE primary
        # scenario + a per-env curriculum level; conditions are realized per-env. partition=False -> no-op.
        partition = False
        weights = [1.0, 1.0, 1.0, 1.0]         # env-fraction weights: [obstacle, slope, load, accel]
        curriculum = True
        curriculum_start_frac = 0.35           # each env starts at this fraction of its scenario difficulty range
        curriculum_step = 0.04                 # +/- level per reset on survive/early-fall
        # obstacle: reward high foot-lift clearance + command stepping (proxy for obstacle height; no real obstacle)
        obstacle_swing_target_range = [0.06, 0.18]   # [m] required swing clearance (curriculum)
        obstacle_cmd_step = 0.35                      # [m/s] commanded stepping velocity for obstacle envs
        # slope: per-env horizontal down-slope force = robot_weight_n*sin(theta) (climb +x); gravity obs tilted
        slope_deg_range = [1.0, 10.0]                 # [deg] recalib 2026-07-04: [4,22]饱和(13°=100%摔); 过渡4-10°
        robot_weight_n = 220.0                        # ~22.4 kg * 9.81 (nominal robot weight for slope force)
        # load: per-env downward force = m_load*g  (force ~ weight per Exp A; avoids load-actor machinery)
        load_kg_range = [3.0, 28.0]                   # [kg] (curriculum)
        # accel: per-env high forward-speed command, rolling (tests wheel-motor torque)
        accel_vx_range = [1.0, 2.5]                   # [m/s] (curriculum)

    class rewards:
        class scales:
            # termination related rewards
            keep_balance = 1.0

            # tracking related rewards
            tracking_ang_vel = 2.0
            tracking_ang_vel_pb = 0.2
            # step/roll decomposition tracking (replaces tracking_lin_vel)
            tracking_step_vel = 2.0   # track commanded stepping-velocity component
            tracking_roll_vel = 2.0   # track commanded rolling-velocity component
            no_fly = 2.5              # reward single-support (mid-step)
            swing_phase = 3.0         # phase-gated alternation: reward correct foot swinging per gait phase
            weight_shift = 2.0        # reward shifting weight onto the intended stance foot per gait phase
            no_jump = 3.0             # reward double-support (no both-feet-off)
            feet_air_time = 2.0        # reward proper alternating swings (force both feet to step)
            stance_symmetry = -1.0     # penalize one foot dominating stance (force L/R alternation)
            feet_swing_height = -20.0 # force swing foot up to target clearance

            # regulation related rewards
            # --- STEPPING MODE (use_gait_stepping=True): the posture-lock rewards below are
            #     zeroed/loosened because they penalize lifting/striding feet (see WORKLOG 2026-07-01). ---
            nominal_foot_position = 0.0     # was 4.0 — pinned foot height, blocked stepping
            leg_symmetry = 0.0              # was 0.5
            same_foot_x_position = 0.0      # was -50 — penalized fore/aft stride
            same_foot_z_position = 0.0      # was -10 — penalized lifting one foot
            lin_vel_z = -0.3
            ang_vel_xy = -0.3
            torques = -0.00016
            dof_acc = -1.5e-7
            action_rate = -0.03
            dof_pos_limits = -2.0
            collision = -50
            action_smooth = -0.03
            orientation = -12.0
            feet_distance = -20             # was -100 — loosened for stepping
            base_height = -5                # was -20 — allow vertical motion for stepping
            # --- stepping-gait contact-schedule rewards (PF-style; need use_gait_stepping=True) ---
            tracking_contacts_shaped_force = -4.0   # STEPPING v2: was -2.0 — enforce swing-off-ground harder
            tracking_contacts_shaped_vel = -2.0
            foot_landing_vel = -0.15
            feet_regulation = -0.05

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5
        leg_symmetry_tracking_sigma = 0.001
        foot_x_position_sigma = 0.001
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.6 + 0.1664
        feet_height_target = 0.10
        feet_swing_height_target = 0.08   # [m] target swing-foot clearance for feet_swing_height reward
        feet_air_time_target = 0.3   # [s] target swing duration; airborne longer than this is rewarded
        min_feet_distance = 0.32
        max_feet_distance = 0.35
        max_contact_force = 100.0  # forces above this value are penalized
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005
        feet_height_tracking_sigma = 0.005
        about_landing_threshold = 0.08  # foot below this height + descending -> penalize landing vel (stepping)

    class normalization:
        class obs_scales:
            lin_vel = 2.0  #修改 本来被我修改为了0.1 原版2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            contact_forces = 0.01
            torque = 0.05
            load_mass = 0.25
            load_pos = 2.0
            mass_scale = 0.05
            com_scale = 5.0
            inertia_scale = 5.0
            morph_scale = 5.0  # co-design: scales centered morphology features (xi-1)*morph_scale for the critic
            motor_scale = 2.0  # co-design Path A: scales centered motor design feature (k-1)*motor_scale for the critic
        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [5, -5, 3]  # [m]
        # lookat = [11.0, 5, 3.0]  # [m]
        lookat = [0, 0, 0]  # [m]
        realtime_plot = True

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 0
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24  # 2**24 -> needed for ~4000+ envs (WF has robot+load actor per env)
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )

class BipedCfgPPOWF(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        is_gru = True
        obs_history_length = BipedCfgWF.env.obs_history_length
        num_input_dim = (BipedCfgWF.env.num_observations + BipedCfgWF.env.num_observations - BipedCfgWF.env.num_actions) * BipedCfgWF.env.obs_history_length
        num_output_dim = 7
        hidden_dims = [256, 128]  #曾经被我修改为[256,256, 128]
        # sym:use_dual_head
        use_dual_head = True
        # sym:use_hierarchical_com_estimation
        use_hierarchical_com_estimation = False
        # sym:com_use_mass_detach
        com_use_mass_detach = True
        #True：com 分支不把梯度回传到 mass 分支，更稳
        #False：com/mass 强耦合联合优化，更激进
        
        # vel branch (3): trunk -> vel
        # sym:vel_head_hidden_dims
        vel_head_hidden_dims = [64]
        # mass branch (1): trunk -> mass
        # sym:mass_head_hidden_dims
        mass_head_hidden_dims = [64]
        # com branch (3): trunk + mass -> com
        # sym:com_head_hidden_dims
        com_head_hidden_dims = [64]
        activation = "elu"
        orthogonal_init = False

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        orthogonal_init = False

    class algorithm:
        # PPO training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        # Co-design: discount regularization (paper eq 2-3). When use_discount_regularization=True,
        # returns/advantages (and the time-out bootstrap) use gamma_reg (< gamma) to shorten the
        # effective horizon and curb value-network memorization across morphologies. Default off.
        use_discount_regularization = False
        gamma_reg = 0.97
        desired_kl = 0.01
        max_grad_norm = 1.0

        # Extra training params
        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4
        critic_take_latent = True
        # Encoder extra loss weighting
        extra_loss_vel_w = 1.0
        extra_loss_mass_w = 2.0
        extra_loss_com_w = 6.0
        # 残差估计需要两个条件：① QS 在 obs 里 (baseline 可读)，② 启用残差学习。
        # 三种典型组合：
        #   (True, True)   = main method（QS in obs + 残差 encoder）
        #   (True, False)  = "QS as feature, direct encoder"（QS in obs 但 encoder 直接输出）
        #   (False, False) = history_only baseline
        use_load_residual_estimation = BipedCfgWF.env.use_qs_in_obs and BipedCfgWF.env.use_residual_learning
        load_residual_baseline_obs_start = 36
        # 当样本检测到负载在体时，对mass/com监督加权
        extra_loss_load_boost = 3.0
        extra_loss_mass_eps = 1.0e-3
        extra_loss_com_eps = 1.0e-3
        # 额外监督回归形式："mse"(原方案) | "smooth_l1"/"huber"
        extra_loss_regression = "huber"
        # SmoothL1/Huber 的 beta(delta)
        extra_loss_huber_delta = 1.0e-2

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 16000  # number of policy updates

        # logging
        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_WF"
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = "WF_TRON1A"
        run_name = ""
        # load and resume
        resume = False
        load_run = "-1"  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = "None"  # updated from load_run and chkpt
