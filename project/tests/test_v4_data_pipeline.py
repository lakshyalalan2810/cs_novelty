"""Tiny smoke tests for the V4 data/model/calibration pipeline.

Uses --smoke configurations only (6 trajectories, 1 seed, 2 epochs,
20 bootstrap replicates). All outputs go to pytest tmp_path, never to
data/processed/v4/ or results/v4/.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v4_calibrate
import v4_generate_dataset
import v4_train_models


class V4DataPipelineSmokeTests(unittest.TestCase):
    def test_smoke_end_to_end(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "smoke.npz"
            # NOTE: closed_loop_mpc is excluded here only because this
            # container cannot execute native MPC solves (torch+MKL load
            # both OpenMP runtimes -> OMP Error #15 abort in scipy SLSQP).
            # The rollout logic itself is covered by StubMpcRolloutTests,
            # and --smoke without exclusions is the real pre-run check.
            out = v4_generate_dataset.main(
                ["--smoke", "--out", str(dataset), "--seed", "99001",
                 "--exclude-kind", "closed_loop_mpc"])
            self.assertTrue(out.is_file())
            manifest = json.loads(
                (tmp_path / "smoke_manifest.json").read_text())
            self.assertEqual(manifest["n_trajectories"], 6)

            with np.load(dataset, allow_pickle=True) as loaded:
                voltage_n = loaded["voltage"].shape[0]
                train_ids = set(int(v) for v in loaded["split_run_ids_train"])
                val_ids = set(int(v)
                              for v in loaded["split_run_ids_validation"])
                test_ids = set(int(v) for v in loaded["split_run_ids_test"])
                window_train = set(int(v)
                                   for v in loaded["window_run_ids_train"])
                x_train_mean = np.asarray(
                    loaded["X_train"], dtype=np.float64).mean(axis=(0, 1))
                norm_mean = np.asarray(loaded["normalization_input_mean"])
                current = np.asarray(loaded["current"])
                current_measured = np.asarray(loaded["current_measured"])
                excitation = set(str(v) for v in loaded["excitation_type"])
            self.assertEqual(voltage_n, 6)
            # Whole-trajectory split: disjoint run ids covering all runs.
            self.assertEqual(train_ids & val_ids, set())
            self.assertEqual(train_ids & test_ids, set())
            self.assertEqual(val_ids & test_ids, set())
            self.assertEqual(train_ids | val_ids | test_ids, set(range(6)))
            # Window run ids stay within their split.
            self.assertTrue(window_train <= train_ids)
            # Normalization was fit on training windows only.
            np.testing.assert_allclose(norm_mean, x_train_mean, rtol=1e-6)
            # Default current semantics: measured == true (noiseless).
            np.testing.assert_array_equal(current, current_measured)
            self.assertTrue(len(excitation) >= 2)

            models_dir = tmp_path / "models"
            models_dir.mkdir()
            v4_train_models.main(["--smoke", "--dataset", str(dataset),
                                  "--out-dir", str(models_dir)])
            for name in ("main_seed_2029.pt", "main_seed_2029_config.json",
                         "main_seed_2029_metrics.json",
                         "main_seed_2029_history.npz", "aux_seed_2029.pt",
                         "aux_seed_2029_config.json", "aux_seed_2029_metrics.json",
                         "aux_seed_2029_history.npz", "training_summary.json"):
                self.assertTrue((models_dir / name).is_file(), name)
            main_config = json.loads(
                (models_dir / "main_seed_2029_config.json").read_text())
            training = main_config["training"]
            self.assertIn("best_epoch", training)
            self.assertIn("converged", training)
            self.assertIn(training["stop_reason"], ("patience", "max_epochs"))

            calibration_dir = tmp_path / "calibration"
            calibration_dir.mkdir()
            v4_calibrate.main(["--smoke", "--dataset", str(dataset),
                               "--models-dir", str(models_dir),
                               "--out-dir", str(calibration_dir)])
            doc = json.loads((calibration_dir
                              / "sensor_seed2029_v4_calibration.json")
                             .read_text())
            values = doc["computed_values"]
            for key in ("instant_threshold", "threshold", "allowance",
                        "aux_recovery_gate", "witness_aux_gate",
                        "witness_ekf_gate"):
                self.assertTrue(np.isfinite(values[key]), key)
                self.assertGreater(values[key], 0, key)
            self.assertIn(values["cusum_percentile"],
                          (90.0, 95.0, 97.5, 99.0, 99.5, 99.9))
            for key, stats in doc["bootstrap_uncertainty"].items():
                with self.subTest(key=key):
                    # 1-ulp tolerance: the single-trajectory smoke pool
                    # makes every replicate identical up to float noise.
                    self.assertLessEqual(stats["ci_lo"], stats["mean"] + 1e-9)
                    self.assertLessEqual(stats["mean"], stats["ci_hi"] + 1e-9)
                    self.assertEqual(stats["replicates"], 20)


class StubMpcRolloutTests(unittest.TestCase):
    """Rollout-logic coverage without native MPC solves (see OMP note)."""

    def test_closed_loop_mpc_rollout_with_stub_solver(self):
        sys.path.insert(0, str(ROOT / "src"))
        from types import SimpleNamespace

        from motor_model import DCMotorParams
        from mpc import PIConfig, PIController

        class StubMpc:
            def __init__(self, succeed_after):
                self.calls = 0
                self.succeed_after = succeed_after

            def reset(self):
                self.calls = 0

            def compute_control(self, history, reference, previous_voltage,
                                **kwargs):
                self.calls += 1
                if self.calls <= self.succeed_after:
                    return {"success": False, "voltage": float("nan")}
                return {"success": True, "voltage": 6.0}

        dt = 0.01
        time = np.arange(0.0, 1.0, dt)
        reference = np.full_like(time, 35.0)
        load = np.full_like(time, 0.03)
        stack = (StubMpc(succeed_after=2),
                 SimpleNamespace(horizon=20, move_blocks=(5, 15)),
                 PIController(PIConfig(0.35, 0.8, (0.0, 12.0), 2.0)))
        rollout = v4_generate_dataset.closed_loop_rollout(
            "closed_loop_mpc", np.random.default_rng(7), time, dt,
            DCMotorParams(), (0.0, 0.0), reference, load, stack)
        self.assertEqual(rollout["fallbacks"], 2)
        for key in ("voltage", "current", "y_true", "noise"):
            self.assertEqual(rollout[key].shape, time.shape)
            self.assertTrue(np.isfinite(rollout[key]).all(), key)
        # Stub voltage (6.0) was applied after the failing solves.
        self.assertIn(6.0, set(rollout["voltage"].tolist()))

    def test_mixture_exclusion_is_recorded(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "excluded.npz"
            v4_generate_dataset.main(
                ["--smoke", "--out", str(dataset), "--seed", "99001",
                 "--exclude-kind", "closed_loop_mpc",
                 "--exclude-kind", "open_multisine"])
            manifest = json.loads(
                (Path(tmp) / "excluded_manifest.json").read_text())
            self.assertEqual(manifest["excluded_kinds"],
                             ["closed_loop_mpc", "open_multisine"])
            with np.load(dataset, allow_pickle=True) as loaded:
                kinds = set(str(v) for v in loaded["excitation_type"])
            self.assertNotIn("closed_loop_mpc", kinds)
            self.assertNotIn("open_multisine", kinds)


if __name__ == "__main__":
    unittest.main()
