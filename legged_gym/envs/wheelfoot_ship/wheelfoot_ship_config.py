"""First ship experiment: rolling balance, then deck-relative motion commands."""
from legged_gym.envs.wheelfoot_flat.wheelfoot_flat_config import BipedCfgWF, BipedCfgPPOWF


class BipedCfgWFShip(BipedCfgWF):
    class env(BipedCfgWF.env):
        num_envs = 512
        env_spacing = 8.0
        episode_length_s = 20.0
        use_gait_stepping = False
        use_gait_phase = False
        use_jump = False
        use_qs_in_obs = False
        use_torques_in_obs = False
        use_residual_learning = False
        use_morphology_in_critic = False
        use_motor_design_in_critic = False
        use_hardware_context_in_policy = False
        num_observations = 37  # base IMU/q/qd 20 + deck IMU 9 + previous actions 8
        num_critic_observations = 44  # existing velocity/mass/COM prefix 7 + obs 37
        fail_to_terminal_time_s = 0.1

    class terrain(BipedCfgWF.terrain):
        mesh_type = "plane"  # selects grid origins; subclass does NOT add a ground plane
        curriculum = False
        measure_heights = False
        critic_measure_heights = False
        evaluation_spawn_xy_jitter = 0.0

    class ship:
        length = 6.0
        width = 4.0
        thickness = 0.20
        surface_z = 1.5
        mass_kg = 10000.0  # prescribed massive dynamic support, NOT a kinematic actor
        friction = 0.9
        reference_mode = 'body'  # v4-compatible default; alternative: gravity_projection
        corner_fraction = 0.0  # opt-in fraction at ALL maximum amplitudes/frequencies
        # Initial curriculum: mild roll/pitch/heave; increase only after baseline balances.
        amplitude_min = [0.02, 0.02, 0.005]  # rad, rad, m
        amplitude_max = [0.08, 0.06, 0.025]
        frequency_min = [0.15, 0.12, 0.18]  # Hz
        frequency_max = [0.35, 0.30, 0.40]
        ramp_s = 2.0
        curriculum = True
        initial_difficulty = 0.25
        curriculum_step = 0.05
        # Gravity estimate is from an idealized attitude filter. All 9 channels
        # reach the actor and its history, with configurable noise and delay.
        imu_enabled = True
        imu_delay_steps = 1  # one policy tick, default 20 ms
        gyro_noise_std = 0.005
        accel_noise_std = 0.03
        attitude_noise_std = 0.002
        sensor_offset = [0.0, 0.0, 0.10]  # deck COM -> top-center sensor, meters
        gyro_scale = 1.0
        accel_scale = 1.0 / 9.81
        edge_margin = 0.4

    class domain_rand(BipedCfgWF.domain_rand):
        add_random_load = False  # existing auxiliary actor is used as the deck
        per_env_load_mass = False
        attached_payload = False
        load_drop_terminal = False
        load_friction = 0.9
        randomize_morphology = False
        randomize_motor_design = False
        push_robots = False
        randomize_default_dof_pos = False
        randomize_base_mass = False
        randomize_base_com = False
        randomize_inertia = False
        randomize_Kp = False
        randomize_Kd = False
        randomize_motor_torque = False
        randomize_action_delay = False
        randomize_imu_offset = False
        randomize_friction = False
        randomize_restitution = False

    class init_state(BipedCfgWF.init_state):
        dof_pos_randomization_range = [-0.02, 0.02]

    class scenario(BipedCfgWF.scenario):
        partition = False
        physical_partition = False

    class commands(BipedCfgWF.commands):
        curriculum = False
        class ranges(BipedCfgWF.commands.ranges):
            lin_vel_x = [0.0, 0.0]  # balance first; set e.g. [-0.3,0.3] for movement
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(BipedCfgWF.rewards):
        heading_yaw_tracking_sigma = 0.25  # (rad/s)^2; optional ship_yaw_rate reward
        class scales:
            keep_balance = 1.0
            ship_tracking = 3.0
            ship_position = 2.0
            orientation = -4.0  # world upright; do not blindly follow deck roll
            base_height = -8.0  # normal distance to moving deck
            torques = -0.0001
            dof_acc = -1.0e-7
            action_rate = -0.02
            action_smooth = -0.01
            dof_pos_limits = -2.0
            collision = -5.0


class BipedCfgPPOWFShip(BipedCfgPPOWF):
    class MLP_Encoder(BipedCfgPPOWF.MLP_Encoder):
        num_input_dim = (37 + 37 - 8) * 10
    class algorithm(BipedCfgPPOWF.algorithm):
        use_load_residual_estimation = False
        extra_loss_mass_w = 0.0
        extra_loss_com_w = 0.0
    class runner(BipedCfgPPOWF.runner):
        experiment_name = "WF_TRON1A_ship"
        run_name = "deck_imu"
        save_interval = 100
        max_iterations = 3000
