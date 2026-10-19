import unittest
from types import SimpleNamespace

import numpy as np

from legged_gym.scripts.payload_step_experiment import overlay
from legged_gym.scripts.payload_step_analysis import aggregate, motion_metrics
from legged_gym.scripts.payload_step_protocol import array_digest, make_scene, trajectory_events


class ProtocolTests(unittest.TestCase):
    def trace(self, steps=250):
        return {"time_s": np.arange(1, steps + 1) * 0.02,
                "first_failure": np.zeros((steps, 1), dtype=bool),
                "on_body": np.ones((steps, 1), dtype=bool),
                "has_load": np.ones((steps, 1), dtype=bool),
                "box_relative_position": np.tile([0.0, 0.0, 0.075], (steps, 1, 1))}

    def test_scene_reproducibility_and_streams(self):
        a, b = make_scene(12345), make_scene(12345)
        self.assertEqual(array_digest(a), array_digest(b))
        self.assertNotEqual(array_digest(a), array_digest(make_scene(20261016)))
        self.assertTrue(np.all((a["body_mass_inertia_scale"] >= .95) &
                               (a["body_mass_inertia_scale"] <= 1.05)))
        np.testing.assert_array_equal(a["initial_joint_offset"], make_scene(12345, steps=300)["initial_joint_offset"])

    def test_overlay_preserves_dictionary_fields(self):
        cfg = SimpleNamespace(control=SimpleNamespace(stiffness={"hip": 1}, existing=True))
        overlay(cfg, {"control": {"stiffness": {"hip": 30}}})
        self.assertEqual(cfg.control.stiffness["hip"], 30)
        self.assertTrue(cfg.control.existing)

    def test_short_contact_loss_is_not_drop(self):
        trace = self.trace()
        trace["on_body"][30:33] = False
        trace["box_relative_position"][30:33, 0, 0] = 0.5
        self.assertFalse(trajectory_events(trace, 0, "free_box", 5)["drop_confirmed"])

    def test_persistent_geometry_loss_is_drop(self):
        trace = self.trace()
        trace["on_body"][50:] = False
        trace["box_relative_position"][50:, 0, 0] = 0.5
        event = trajectory_events(trace, 0, "free_box", 5)
        self.assertTrue(event["drop_confirmed"])
        self.assertAlmostEqual(event["first_drop_start_s"], 1.02)
        self.assertAlmostEqual(event["first_drop_confirm_s"], 1.22)

    def test_partial_overhang_does_not_prove_departure(self):
        trace = self.trace()
        trace["on_body"][30:] = False
        trace["box_relative_position"][30:, 0, 0] = .27
        self.assertFalse(trajectory_events(trace, 0, "free_box", 5)["drop_confirmed"])

    def test_contact_loss_without_geometry_is_not_drop(self):
        trace = self.trace()
        trace["on_body"][20:] = False
        self.assertFalse(trajectory_events(trace, 0, "free_box", 5)["drop_confirmed"])

    def test_failure_censors_later_box_departure(self):
        trace = self.trace()
        trace["first_failure"][40:] = True
        trace["on_body"][50:] = False
        trace["box_relative_position"][50:, 0, 0] = 0.5
        event = trajectory_events(trace, 0, "free_box", 5)
        self.assertFalse(event["drop_confirmed"])
        self.assertFalse(event["robot_completed"])
        self.assertEqual(event["eligible_samples"], 40)

    def test_failed_catch_is_kept(self):
        trace = self.trace()
        trace["on_body"][:] = False
        event = trajectory_events(trace, 0, "free_box", 5)
        self.assertTrue(event["catch_failed"])
        self.assertEqual(event["motion_end_exclusive_s"], 2.0)

    def test_no_box_metric_for_empty_condition(self):
        event = trajectory_events(self.trace(), 0, "empty", 5)
        self.assertIsNone(event["loaded_completed"])
        self.assertTrue(event["robot_completed"])

    def test_pitch_bias_and_fluctuation_are_distinct(self):
        trace = self.trace()
        count = len(trace["time_s"])
        trace["root_state"] = np.zeros((count, 1, 13))
        trace["root_state"][:, 0, 4] = np.sin(.1 / 2)
        trace["root_state"][:, 0, 6] = np.cos(.1 / 2)
        trace["body_velocity"] = np.full((count, 1, 3), .9)
        result = motion_metrics(trace, 0, np.ones(count, dtype=bool))
        self.assertAlmostEqual(result["speed_rmse_mps"], 0)
        self.assertAlmostEqual(result["pitch_rms_deg"], np.rad2deg(.1))
        self.assertAlmostEqual(result["pitch_sd_deg"], 0)
        missing = motion_metrics(trace, 0, np.zeros(count, dtype=bool))
        self.assertTrue(all(value is None for value in missing.values()))

    def test_equal_seed_means_not_sample_pooling(self):
        rows = [{"seed": 1, "metric": 0}, {"seed": 1, "metric": 0}, {"seed": 2, "metric": 100}]
        by_seed = aggregate(rows, ("seed",), ("metric",))
        self.assertEqual(np.mean([row["metric"] for row in by_seed]), 50)


if __name__ == "__main__":
    unittest.main()
