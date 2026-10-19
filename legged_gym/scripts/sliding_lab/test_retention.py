import math
import unittest

import numpy as np
import torch

from .geometry import measure
from .retention_protocol import RetentionSettings, payload_geometry, retained_reward, eval_cases


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.s = RetentionSettings()
        self.root = torch.zeros(1, 13, dtype=torch.float64)
        self.root[:, 6] = 1.
        self.box = self.root.clone()
        self.box[:, 2] = self.s.plate_z + .0525
        self.force = torch.tensor([[0., 0., self.s.mass * 9.81]], dtype=torch.float64)

    def geom(self):
        return payload_geometry(self.root, self.box, self.force, self.s)

    def test_flat_support(self):
        g = self.geom()
        self.assertTrue(g["supported"].item())
        self.assertFalse(g["departure"].item())
        self.assertAlmostEqual(g["margins"].amin().item(), .2)
        self.assertAlmostEqual(g["gap"].item(), 0.)

    def test_partial_overhang_not_drop(self):
        self.box[:, 0] = .22
        g = self.geom()
        self.assertLess(g["margins"].amin().item(), 0.)
        self.assertTrue(g["overlap"].item())
        self.assertFalse(g["departure"].item())

    def test_airborne_not_drop(self):
        self.box[:, 2] += .1
        self.force[:] = 0.
        self.assertFalse(self.geom()["departure"].item())
        self.assertFalse(self.geom()["supported"].item())

    def test_departure(self):
        self.box[:, 0] = .4
        self.box[:, 2] -= .08
        self.assertTrue(self.geom()["departure"].item())

    def test_ground_contact_not_retained(self):
        self.box[:, 2] = -.5
        self.assertFalse(self.geom()["supported"].item())
        self.assertTrue(self.geom()["departure"].item())

    def test_angular_transport_velocity(self):
        self.root[:, 12] = 2.
        self.box[:, 0] = .1
        self.box[:, 8] = .2
        np.testing.assert_allclose(self.geom()["relative_velocity"].numpy(), 0., atol=1e-12)

    def test_tensor_matches_existing_numpy_geometry(self):
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_euler("xyz", [.2, -.1, .4])
        self.root[:, :3] = torch.tensor([4., 5., .8])
        self.root[:, 3:7] = torch.from_numpy(rot.as_quat())
        self.box[:, 3:7] = self.root[:, 3:7]
        self.box[:, :3] = self.root[:, :3] + torch.from_numpy(rot.apply([.03, -.02, self.s.plate_z + .0525]))
        self.force[:] = torch.from_numpy(rot.apply([0., 0., self.s.mass * 9.81]))
        old = measure(self.box[0, :3].numpy(), rot.as_matrix(),
                      self.root[0, :3].numpy() + rot.apply([0, 0, self.s.plate_z]), rot.as_matrix(), self.force[0].numpy())
        self.assertAlmostEqual(self.geom()["margins"].amin().item(), old["footprint_margin_m"])
        self.assertAlmostEqual(self.geom()["gap"].item(), old["normal_gap_m"])

    def test_relative_rotation_changes_footprint(self):
        self.box[:, 5] = math.sin(math.pi / 8)
        self.box[:, 6] = math.cos(math.pi / 8)
        self.assertAlmostEqual(self.geom()["margins"].amin().item(), .25 - .05 * math.sqrt(2))

    def test_drop_penalty_not_dt_scaled(self):
        g = self.geom()
        kwargs = dict(geometry=g, active=torch.ones(1, dtype=torch.bool), gravity=torch.tensor([[0., 0., -1.]]),
                      velocity=torch.zeros(1, 3), ang_velocity=torch.zeros(1, 3), height=torch.tensor([.7664]),
                      actions=torch.zeros(1, 8), previous_actions=torch.zeros(1, 8), torques=torch.zeros(1, 8),
                      fall=torch.zeros(1, dtype=torch.bool), settings=self.s, dt=.02)
        r0, _ = retained_reward(drop=torch.zeros(1, dtype=torch.bool), **kwargs)
        r1, _ = retained_reward(drop=torch.ones(1, dtype=torch.bool), **kwargs)
        self.assertAlmostEqual((r0 - r1).item(), self.s.drop_cost)

    def test_eval_has_four_directions_and_no_disturbance_controls(self):
        cases = eval_cases()
        self.assertEqual(len(cases), 136)
        self.assertEqual(sum(c["nominal_dv"] == 0 for c in cases), 8)
        self.assertEqual(len({(c["nominal_dv"], c["direction"], c["repeat"]) for c in cases}), 136)

    def test_residual_starts_at_original_and_is_disabled_without_load(self):
        from .retention_residual import RetentionResidualActor
        base = torch.nn.Sequential(torch.nn.Linear(58, 16), torch.nn.ELU(), torch.nn.Linear(16, 8))
        policy = RetentionResidualActor(base)
        x = torch.randn(5, 73)
        torch.testing.assert_close(policy(x), base(x[:, :58]))
        with torch.no_grad():
            policy.correction[-1].bias[:] = 1.
        x[:, -1] = 0.
        torch.testing.assert_close(policy(x), base(x[:, :58]))

    def test_residual_limits_and_frozen_base(self):
        from .retention_residual import RetentionResidualActor
        base = torch.nn.Sequential(torch.nn.Linear(58, 8))
        policy = RetentionResidualActor(base)
        x = torch.randn(5, 73)
        x[:, -1] = 1.
        with torch.no_grad():
            policy.correction[-1].bias[:] = 100.
        residual = policy(x) - base(x[:, :58])
        self.assertTrue(bool((residual.abs() <= policy.limits + 1e-6).all()))
        self.assertTrue(all(not p.requires_grad for p in policy.base.parameters()))


if __name__ == "__main__":
    unittest.main()
