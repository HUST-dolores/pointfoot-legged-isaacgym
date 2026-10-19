"""Controller tests use temporary synthetic fixtures, never experimental records."""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .retention_reward_study import Study, START_HASHES


class DoneProcess:
    pid = 999999

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


class ControllerTests(unittest.TestCase):
    def test_all_checkpoints_validated_and_incumbent_retained(self):
        with tempfile.TemporaryDirectory(prefix="retention_controller_fixture_") as directory:
            out = Path(directory)
            study = Study(out, 1.)
            initial = dict(checkpoint=str(out / "initial.pt"), checkpoint_sha256=START_HASHES[303],
                           training_seed=303, arm="initial", iteration=1000, control_success=1.,
                           in_domain_success=.9, in_domain_fall_rate=0.)

            def start(key, gpu, module, args, log):
                options = dict(zip(map(str, args[::2]), map(str, args[1::2])))
                target = Path(options["--output"])
                target.mkdir(parents=True)
                if module == "retention_train":
                    for iteration in (1250, 1500, 1750, 2000):
                        (target / f"model_{iteration}.pt").touch()
                    (target / "result.json").write_text(json.dumps(dict(cleanup_completed=True,
                        encoder_unchanged=True, base_actor_unchanged=True, iterations=2000)))
                    (target / "metadata.json").write_text(json.dumps(dict(
                        resume_checkpoint_sha256=START_HASHES[303], additional_iterations=1000,
                        reward_settings_changes={"alive_weight": {"before": 2., "after": 4.}})))
                study.children[key] = (DoneProcess(), io.StringIO())
                return study.children[key][0]

            def fake_metrics(directory):
                return dict(control_success=1., in_domain_success=.8, in_domain_fall_rate=0.,
                            checkpoint=str(directory), checkpoint_sha256="fixture")

            with patch.object(study, "start", side_effect=start), \
                    patch("legged_gym.scripts.sliding_lab.retention_reward_study.evaluation_metrics", side_effect=fake_metrics), \
                    patch("legged_gym.scripts.sliding_lab.retention_reward_study.time.sleep"):
                selected = study.train_arm(303, 0, "alive4", initial)
            self.assertEqual(selected["iteration"], 1000)
            self.assertEqual(selected["arm"], "initial")
            self.assertEqual([r["iteration"] for r in study.state["validation"]], [1250, 1500, 1750, 2000])
            self.assertFalse(study.children)

    def test_report_handles_shared_test_batches_without_dropping_failures(self):
        with tempfile.TemporaryDirectory(prefix="retention_report_fixture_") as directory:
            out = Path(directory)
            study = Study(out, 1.)
            for seed in (303, 404):
                raw = out / f"fixture_s{seed}"
                raw.mkdir()
                root = np.zeros((2, 3, 13))
                root[..., 6] = 1.
                np.savez_compressed(raw / "trace.npz", root=root,
                                    time_s=np.array([[.8]*3, [9.98]*3]))
                cases = [dict(env=i, nominal_dv=dv, success=(i != 2), drop=(i == 2), fall=False)
                         for i, dv in enumerate((0., .25, .85))]
                (raw / "cases.json").write_text(json.dumps(cases))
                initial = dict(training_seed=seed, arm="initial", iteration=1000, in_domain_success=1.)
                study.state["validation"].append(initial)
                for arm in ("initial", "control", "alive4", "drop24"):
                    study.state["formal"][f"{arm}_s{seed}_e61001"] = dict(
                        directory=str(raw), scenario_sha256="same_fixture_scenarios",
                        checkpoint_sha256=f"fixture_s{seed}")
            study.report()
            with (out / "reports/all_trials.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 24)
            self.assertEqual(sum(row["drop"] == "True" for row in rows), 8)
            self.assertEqual(len({row["source_directory"] for row in rows}), 2)
            summary = json.loads((out / "reports/summary.json").read_text())
            self.assertTrue(summary["same_checkpoint_test_batches_reused_not_independent"])
            self.assertTrue((out / "reports/validation_learning_curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
