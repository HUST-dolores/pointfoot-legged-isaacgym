import unittest

import torch

from .retention_protocol import RetentionSettings, payload_geometry, retained_reward
from .retention_reward_ablation import ARMS, override_rewards, settings_changes


class RewardAblationTests(unittest.TestCase):
    def test_no_override_preserves_nondefault_saved_settings(self):
        saved = RetentionSettings(mass=7., friction=.08, alive_weight=3., drop_cost=17.)
        self.assertEqual(override_rewards(saved).dictionary(), saved.dictionary())

    def test_each_arm_changes_only_declared_reward_term(self):
        original = RetentionSettings()
        expected = {"control": set(), "alive4": {"alive_weight"}, "drop24": {"drop_cost"}}
        for arm, kwargs in ARMS.items():
            actual = override_rewards(original, **kwargs)
            self.assertEqual(set(settings_changes(original, actual)), expected[arm])
            self.assertEqual(original.alive_weight, 2.)
            self.assertEqual(original.drop_cost, 12.)

    def test_invalid_values_rejected(self):
        for value in (0., -1., float("nan"), float("inf")):
            for name in ("alive_weight", "drop_cost"):
                with self.assertRaises(ValueError):
                    override_rewards(RetentionSettings(), **{name: value})

    def test_actual_reward_differences_have_correct_units(self):
        settings = RetentionSettings()
        root = torch.zeros(1, 13, dtype=torch.float64)
        root[:, 6] = 1.
        box = root.clone()
        box[:, 2] = settings.plate_z + .0525
        geometry = payload_geometry(root, box, torch.tensor([[0., 0., settings.mass * 9.81]], dtype=root.dtype), settings)
        kwargs = dict(geometry=geometry, active=torch.ones(1, dtype=torch.bool),
                      gravity=torch.tensor([[0., 0., -1.]]), velocity=torch.zeros(1, 3),
                      ang_velocity=torch.zeros(1, 3), height=torch.tensor([.7664]),
                      actions=torch.zeros(1, 8), previous_actions=torch.zeros(1, 8),
                      torques=torch.zeros(1, 8), fall=torch.zeros(1, dtype=torch.bool), dt=.02)
        zero = torch.zeros(1, dtype=torch.bool)
        one = torch.ones(1, dtype=torch.bool)
        original, _ = retained_reward(settings=settings, drop=zero, **kwargs)
        doubled, _ = retained_reward(settings=override_rewards(settings, alive_weight=4.), drop=zero, **kwargs)
        self.assertAlmostEqual((doubled - original).item(), .04)
        original_drop, _ = retained_reward(settings=settings, drop=one, **kwargs)
        doubled_drop, _ = retained_reward(settings=override_rewards(settings, drop_cost=24.), drop=one, **kwargs)
        self.assertAlmostEqual((original_drop - doubled_drop).item(), 12.)

    def test_selection_prefers_incumbent_on_equal_validation(self):
        from .retention_reward_study import selection_key
        initial = dict(control_success=1., in_domain_success=.8, in_domain_fall_rate=0., iteration=1000)
        later = dict(initial, iteration=1250)
        self.assertGreater(selection_key(initial), selection_key(later))

    def test_stress_results_cannot_select_checkpoint(self):
        from .retention_reward_study import selection_key
        initial = dict(control_success=1., in_domain_success=.8, in_domain_fall_rate=0., iteration=1000,
                       stress_success=0.)
        later = dict(initial, iteration=1250, stress_success=1.)
        self.assertGreater(selection_key(initial), selection_key(later))


if __name__ == "__main__":
    unittest.main()
