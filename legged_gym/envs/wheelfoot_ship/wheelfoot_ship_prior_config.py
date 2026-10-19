"""Separate opt-in prior task: old ship37 checkpoints/configs stay unchanged."""
from .wheelfoot_ship_config import BipedCfgWFShip, BipedCfgPPOWFShip


class BipedCfgWFShipPrior(BipedCfgWFShip):
    class env(BipedCfgWFShip.env):
        num_observations = 61
        num_critic_observations = 68

    class ship_model_prior:
        enabled = True
        # R_BD initial yaw calibration. Gravity cannot observe this heading;
        # with two IMUs it subsequently drifts according to gyro biases.
        initial_relative_yaw_rad = 0.
        prediction_horizon_s = .10  # relative to the available delayed IMU estimate
        gravity_correction_gain = .04
        upright_gain = .5
        max_upright_correction_rad = .10
        contact_on_threshold_n = 5.
        contact_off_threshold_n = 2.
        max_contact_displacement_m = .06
        max_joint_delta_rad = .25
        ik_damping = .02
        ik_iterations = 5
        ik_tolerance_m = .005
        joint_limit_margin_rad = .01
        minimum_normal_cosine = .35
        contact_displacement_scale = 10.
        leg_length_scale = 10.
        joint_angle_scale = 1.
        joint_names = [['abad_L_Joint', 'hip_L_Joint', 'knee_L_Joint'],
                       ['abad_R_Joint', 'hip_R_Joint', 'knee_R_Joint']]
        wheel_joint_names = ['wheel_L_Joint', 'wheel_R_Joint']
        wheel_body_names = ['wheel_L_Link', 'wheel_R_Link']


class BipedCfgPPOWFShipPrior(BipedCfgPPOWFShip):
    class MLP_Encoder(BipedCfgPPOWFShip.MLP_Encoder):
        num_input_dim = (61+53)*10

    class runner(BipedCfgPPOWFShip.runner):
        experiment_name = 'WF_TRON1A_ship_prior'
        run_name = 'deck_imu_model_prior'
