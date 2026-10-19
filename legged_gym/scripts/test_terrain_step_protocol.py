import unittest

import numpy as np

from legged_gym.scripts.terrain_step_protocol import (
    DX, NX, NY, X0, Y0, anchors, cylinder_vertical_extent, capsule_vertical_extent, make_terrain_scene,
    sample_height_numpy, triangle_mesh,
)
from legged_gym.scripts.payload_step_protocol import array_digest
from legged_gym.scripts.terrain_step_analysis import metrics


class TerrainProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = make_terrain_scene(12345)

    def test_exact_plane_and_slope_sampling(self):
        xy = np.array([[.03, .26], [1.35, -.48], [4.75, 1.11]])
        env = np.array([0, 3, 29])
        np.testing.assert_allclose(sample_height_numpy(self.scene["height_flat"], env, xy), 0)
        np.testing.assert_allclose(sample_height_numpy(self.scene["height_slope5"], env, xy),
                                   xy[:, 0] * np.tan(np.deg2rad(5)), atol=1e-7)

    def test_triangle_interior_is_not_bilinear(self):
        heights = np.zeros((1, NX, NY))
        heights[0, 1, 1] = 1
        points = np.array([[X0+.2*DX, Y0+.3*DX], [X0+.8*DX, Y0+.7*DX]])
        result = sample_height_numpy(heights, np.zeros(2, dtype=int), points)
        np.testing.assert_allclose(result, [0, .5], atol=1e-12)

    def test_mesh_normals_up_and_disjoint_patches(self):
        vertices, triangles = triangle_mesh(self.scene["height_slope5"][:2], anchors(2))
        first = vertices[triangles[:1000]]
        cross = np.cross(first[:, 1]-first[:, 0], first[:, 2]-first[:, 0])
        self.assertTrue(np.all(cross[:, 2] > 0))
        self.assertEqual(len(vertices), 2 * NX * NY)
        self.assertEqual(len(triangles), 2 * 2 * (NX-1) * (NY-1))

    def test_rough_bounds_and_reproducibility(self):
        self.assertGreaterEqual(self.scene["height_rough5cm"].min(), -.05)
        self.assertLessEqual(self.scene["height_rough5cm"].max(), .05)
        self.assertEqual(array_digest(self.scene), array_digest(make_terrain_scene(12345)))
        self.assertFalse(np.array_equal(self.scene["height_rough5cm"][0], self.scene["height_rough5cm"][1]))

    def test_cylinder_lowest_extent(self):
        values = cylinder_vertical_extent(np.array([0, 1, 1/np.sqrt(2)]), .127, .05)
        np.testing.assert_allclose(values, [.127, .025, (.127+.025)/np.sqrt(2)])

    def test_native_capsule_extent(self):
        values = capsule_vertical_extent(np.array([0, 1, 1/np.sqrt(2)]), .127, .05)
        np.testing.assert_allclose(values, [.127, .152, .127+.025/np.sqrt(2)])

    def test_contact_fractions_and_forward_sign(self):
        state = np.zeros((4, 1, 13))
        state[..., 6] = 1
        state[..., 7] = -.1
        force = np.zeros((4, 1, 2, 3))
        force[0, 0, 0, 2] = 100
        force[1, 0, 1, 2] = 100
        force[2, 0, :, 2] = 100
        z = {"root_state": state, "body_velocity": np.zeros((4, 1, 3)), "wheel_forces": force}
        result = metrics(z, 0, np.ones(4, dtype=bool), "slope5")
        self.assertEqual(result["left_contact_fraction"], .5)
        self.assertEqual(result["right_contact_fraction"], .5)
        self.assertEqual(sum(result[k] for k in ("left_only_fraction", "right_only_fraction",
                                                 "double_support_fraction", "no_contact_fraction")), 1)
        self.assertLess(result["world_forward_mean_mps"], 0)
        self.assertTrue(all(v is None for v in metrics(z, 0, np.zeros(4, dtype=bool), "flat").values()))


if __name__ == "__main__":
    unittest.main()
