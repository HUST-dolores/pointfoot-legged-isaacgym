"""CPU-only contract tests; fabricated arrays here are tests, never experiment data."""

import unittest
import numpy as np

from legged_gym.scripts.eccentric_payload_protocol import make_scene, first_sustained, angles, events


class ProtocolTests(unittest.TestCase):
    def test_reproducible_paired_tables(self):
        a, b = make_scene(12345), make_scene(12345)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_allclose(a["box_offset_body"], np.tile([0, 0, .1], (30, 1)))

    def test_sustained_duration_not_sample_count(self):
        t = np.arange(20)*.02
        self.assertEqual(first_sustained(t, np.arange(20) >= 3, .1), (.06, .16))
        self.assertEqual(first_sustained(t[:8], np.arange(8) >= 3, .1), (None, None))

    def test_angles(self):
        root = np.zeros((2,13))
        root[:,6] = 1
        np.testing.assert_array_equal(angles(root)[0], 0)
        root[1,3:7] = [np.sin(np.pi/3), 0, 0, np.cos(np.pi/3)]
        np.testing.assert_allclose(angles(root)[0][1], 120)
        self.assertLess(angles(root)[2][1], .5)

    def trace(self):
        n = 150
        root = np.zeros((n,1,13)); root[:,:,2] = .7; root[:,:,6] = 1
        return {"time_s": np.arange(1,n+1)*.02, "root_state": root,
            "on_body": (np.arange(n) >= 30)[:,None], "has_load": (np.arange(n) >= 24)[:,None],
            "box_relative_position": np.tile([0.,0.,.1], (n,1,1)), "first_failure": np.zeros((n,1), bool)}

    def test_fall_and_later_drop_both_count(self):
        z = self.trace()
        z["root_state"][60:,0,2] = .1
        z["box_relative_position"][100:,0,0] = .5
        z["on_body"][100:] = False
        row = events(z,0)
        self.assertEqual((row["fall"],row["drop"],row["drop_after_or_with_fall"]), (1,1,1))

    def test_termination_not_automatically_fall(self):
        z = self.trace(); z["first_failure"][80:] = True
        row = events(z,0)
        self.assertEqual((row["fall"],row["original_termination"]), (0,1))

    def test_no_support_is_catch_failure_not_drop(self):
        z = self.trace(); z["on_body"][:] = False; z["box_relative_position"][:,:,0] = 10
        row = events(z,0)
        self.assertEqual((row["catch_failed"],row["drop"],row["intact_40s"]), (1,0,0))

    def test_short_departure_not_drop(self):
        z = self.trace(); z["on_body"][70:80] = False; z["box_relative_position"][70:80,0,0] = .5
        self.assertEqual(events(z,0)["drop"], 0)

    def test_all_assigned_metric_retains_late_bad_samples(self):
        from legged_gym.scripts.eccentric_payload_analysis import metrics
        root = np.zeros((3,1,13)); root[:,0,0] = [0,1,10]
        pitch = np.array([[0.],[0.],[60.]])
        row = metrics(root,np.zeros_like(pitch),pitch,0,np.ones(3,bool),np.zeros(2))
        self.assertAlmostEqual(row["pitch_rms_deg"], np.sqrt(1200))
        self.assertEqual(row["max_planar_drift_m"], 10.)

    def test_missing_window_not_zero(self):
        from legged_gym.scripts.eccentric_payload_analysis import metrics
        row = metrics(np.zeros((1,1,13)),np.zeros((1,1)),np.zeros((1,1)),0,np.array([False]),np.zeros(2))
        self.assertTrue(all(value is None for value in row.values()))


if __name__ == "__main__":
    unittest.main()
