"""Independent ship rolling/stepping tasks; the standing task is unchanged."""
from .wheelfoot_ship_config import BipedCfgWFShip, BipedCfgPPOWFShip


class BipedCfgWFShipRoll(BipedCfgWFShip):
    class env(BipedCfgWFShip.env):
        # Own gait clock: do not enable the flat-task observation suffix/rewards.
        use_gait_stepping = False
        use_gait_phase = False
        num_observations = 40
        num_critic_observations = 47
        episode_length_s = 60.

    class ship(BipedCfgWFShip.ship):
        reference_mode = 'gravity_projection'

    class commands(BipedCfgWFShip.commands):
        cmd_step_range = [0., 0.]
        heading_command = False
        # Continuous shuttle commands replace random command resampling.

    class ship_mobility:
        mode = 'roll'
        max_speed_m_s = .2
        period_s = 8.
        command_ramp_s = 2.
        speed_curriculum = True
        speed_ramp_iterations = 2000
        initial_speed_fraction = .25
        gait_frequency_hz = 1.5
        stance_duty = .6
        clearance_target_m = .06
        clearance_sigma_m = .025
        contact_threshold_n = 5.

    class rewards(BipedCfgWFShip.rewards):
        class scales(BipedCfgWFShip.rewards.scales):
            ship_tracking = 0.  # standing task used deck-frame zero commands
            ship_position = 0.  # shuttle endpoints are not the standing anchor
            ship_mobility_tracking = 4.
            ship_mobility_contacts = 2.
            ship_mobility_clearance = 1.5
            ship_mobility_slip = -.05
            ship_mobility_center = -.1


class BipedCfgWFShipStep(BipedCfgWFShipRoll):
    class ship_mobility(BipedCfgWFShipRoll.ship_mobility):
        mode = 'step'


class BipedCfgPPOWFShipMobility(BipedCfgPPOWFShip):
    class MLP_Encoder(BipedCfgPPOWFShip.MLP_Encoder):
        num_input_dim = (40+32)*10

    class runner(BipedCfgPPOWFShip.runner):
        experiment_name = 'WF_TRON1A_ship_mobility'
        max_iterations = 12000
