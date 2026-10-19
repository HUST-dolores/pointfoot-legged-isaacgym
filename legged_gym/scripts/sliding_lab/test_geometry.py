import unittest
from pathlib import Path
import numpy as np
from .geometry import measure,urdf_audit


class GeometryTests(unittest.TestCase):
    def test_old_plate_overlap(self):
        p=Path(__file__).resolve().parents[3]/"resources/robots/WF_TRON1A/urdf/robot.urdf"
        a=urdf_audit(p)
        self.assertAlmostEqual(a["base_collision_top_z_m"],.023)
        self.assertAlmostEqual(a["plate_top_z_m"],.0225)

    def test_flat_supported(self):
        r=measure(np.array([0.,0.,.0525]),np.eye(3),np.zeros(3),np.eye(3),np.array([0,0,58.]))
        self.assertAlmostEqual(r["normal_gap_m"],0.)
        self.assertAlmostEqual(r["footprint_margin_m"],.2)
        self.assertAlmostEqual(r["normal_force_N"],58.)

    def test_airborne_not_contact(self):
        r=measure(np.array([0.,0.,.2]),np.eye(3),np.zeros(3),np.eye(3),np.zeros(3))
        self.assertGreater(r["normal_gap_m"],.1)

    def test_edge_uses_extent_not_center(self):
        r=measure(np.array([.22,0.,.0525]),np.eye(3),np.zeros(3),np.eye(3),np.zeros(3))
        self.assertLess(r["footprint_margin_m"],0.)

    def test_measures_invariant_to_translated_rotated_plate(self):
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_euler("xyz",[.2,-.1,.4]).as_matrix()
        translation=np.array([7.,-4.,.8])
        r=measure(translation+rotation@np.array([.03,-.02,.0525]),rotation,
                  translation,rotation,rotation@np.array([0.,0.,58.]))
        self.assertAlmostEqual(float(r["normal_gap_m"]),0.)
        np.testing.assert_allclose(r["bottom_center_plate"][:2],[.03,-.02],atol=1e-10)
        self.assertAlmostEqual(float(r["normal_force_N"]),58.)


if __name__=="__main__": unittest.main()
