"""Boundary and time-alignment tests, not fabricated experimental observations."""

import unittest
import numpy as np
from legged_gym.scripts.roll_step_response_protocol import phase_at_step,command_at_step,COMMANDS,first_failure


class CommandTests(unittest.TestCase):
    def test_boundaries_for_next_action(self):
        np.testing.assert_array_equal(phase_at_step([499,500,999,1000,1499,1500,2000]),[0,1,1,2,2,3,3])

    def test_preceding_interval_alignment(self):
        np.testing.assert_array_equal(phase_at_step([500,501,1000,1001,1500,1501],True),[0,1,1,2,2,3])

    def test_exact_500_samples_per_phase(self):
        phases=phase_at_step(np.arange(1,2001),True)
        np.testing.assert_array_equal(np.bincount(phases),[500]*4)

    def test_total_target_constant(self):
        np.testing.assert_allclose(command_at_step(np.arange(2001)).sum(-1),.5,atol=1e-7)

    def test_return_is_same_command_not_extra_condition(self):
        np.testing.assert_array_equal(COMMANDS[0],COMMANDS[3])

    def fixture(self):
        root=np.zeros((4,1,13)); root[:,:,2]=.7; root[:,:,6]=1.
        forces=np.zeros((4,1,2,3)); forces[:,0,:,2]=[[10,10],[10,0],[0,10],[0,0]]
        return {"time_s":np.arange(1,5)*.02,"root_state":root,"body_velocity":np.full((4,1,3),.5),
            "wheel_forces":forces,"wheel_clearance_signed":np.full((4,1,2),.02),
            "v_roll_proxy":np.full((4,1),.3),"v_step_proxy":np.full((4,1),.2),
            "first_failure":np.zeros((4,1),bool)}

    def test_four_contact_states_partition(self):
        from legged_gym.scripts.roll_step_response_analysis import motion_metrics
        r=motion_metrics(self.fixture(),0,np.ones(4,bool))
        self.assertEqual(sum(r[k] for k in ("double_support_fraction","left_only_fraction",
            "right_only_fraction","neither_contact_fraction")),1.)
        self.assertEqual(r["left_swing_fraction"],.5)
        self.assertEqual(r["right_swing_fraction"],.5)

    def test_total_speed_error_not_roll_command_error(self):
        from legged_gym.scripts.roll_step_response_analysis import motion_metrics
        self.assertEqual(motion_metrics(self.fixture(),0,np.ones(4,bool))["speed_rmse_mps"],0.)

    def test_empty_window_missing_not_zero(self):
        from legged_gym.scripts.roll_step_response_analysis import motion_metrics
        self.assertTrue(all(v is None for v in motion_metrics(self.fixture(),0,np.zeros(4,bool)).values()))

    def test_no_contact_not_necessarily_air_clearance(self):
        from legged_gym.scripts.roll_step_response_analysis import motion_metrics
        z=self.fixture(); z["wheel_clearance_signed"][:]=0
        r=motion_metrics(z,0,np.ones(4,bool))
        self.assertEqual(r["left_swing_fraction"],.5)
        self.assertEqual(r["left_air_clearance_fraction"],0.)

    def test_original_termination_is_separate_from_geometric_fall(self):
        z=self.fixture(); z["first_failure"][2:]=True
        r=first_failure(z,0)
        self.assertEqual(r["original_termination_s"],.06)
        self.assertIsNone(r["fall_start_s"])


if __name__=="__main__": unittest.main()
