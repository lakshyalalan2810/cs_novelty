"""Tests for the V4 classical baselines (E2/S1/S2/S3)."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines_v4 import (
    CusumCore,
    EKFResidualDetector,
    FixedThresholdDetector,
    LuenbergerBaseline,
    MedianRatePlausibilityFilter,
    calibrate_cusum_grid,
    calibrate_fixed_gate,
    calibrate_plausibility,
    design_luenberger_gain,
    detector_from_cusum_doc,
)
from ekf_observer import EKFConfig
from motor_model import DCMotorParams, dc_motor_dynamics

REQUIRED_KEYS = ("trusted", "sensor_suspect", "substitute", "estimate")


def small_ekf_config() -> EKFConfig:
    return EKFConfig(Q=np.diag([1e-6, 1e-4, 1e-6]), R=1e-6,
                     P0=np.diag([1.0, 100.0, 0.01]), dt=0.01,
                     params=DCMotorParams())


class InterfaceTests(unittest.TestCase):
    def test_all_baselines_share_the_output_interface(self):
        e2 = EKFResidualDetector(
            small_ekf_config(), CusumCore(1.0, 0.0, 0.1, 1.0))
        s1 = FixedThresholdDetector(1.0)
        s2 = MedianRatePlausibilityFilter()
        s3 = LuenbergerBaseline(CusumCore(2.0, 0.0, 0.2, 2.0))
        for out in (e2.update(6.0, 2.0, 35.0), s1.update(35.0, 35.0),
                    s2.update(35.0), s3.update(6.0, 2.0, 35.0)):
            for key in REQUIRED_KEYS:
                self.assertIn(key, out)
            self.assertIsInstance(out["trusted"], bool)
            self.assertTrue(np.isfinite(out["estimate"]))


class CusumCoreTests(unittest.TestCase):
    def test_entry_and_recovery(self):
        core = CusumCore(1.0, 0.0, 0.1, 1.0)
        for _ in range(20):
            out = core.update(0.0)
        self.assertFalse(out["sensor_suspect"])
        # Minimal latching fault (enter_count samples); the unclamped core
        # mirrors frozen discharge dynamics (slow after long faults).
        for _ in range(3):
            out = core.update(5.0)
        self.assertTrue(out["sensor_suspect"])
        for _ in range(500):
            out = core.update(0.0)
        self.assertFalse(out["sensor_suspect"])


class E2Tests(unittest.TestCase):
    def test_detects_bias_and_recovers(self):
        detector = EKFResidualDetector(
            small_ekf_config(), CusumCore(1.0, 0.0, 0.1, 1.0))
        # Settle the observer, then match the healthy measurement to its
        # steady state so residuals start near zero by construction.
        for _ in range(300):
            out = detector.update(6.0, 2.0, 35.0)
        healthy = out["estimate"]
        detector.detector.reset()
        for _ in range(20):
            out = detector.update(6.0, 2.0, healthy)
        self.assertFalse(out["sensor_suspect"])
        entered = False
        for _ in range(3):
            out = detector.update(6.0, 2.0, healthy + 10.0)
            entered |= out["sensor_suspect"]
        self.assertTrue(entered)
        for _ in range(2000):
            out = detector.update(6.0, 2.0, healthy)
        self.assertFalse(out["sensor_suspect"])
        self.assertIn("innovation", out)


class S1Tests(unittest.TestCase):
    def test_fixed_gate_with_persistence(self):
        detector = FixedThresholdDetector(1.0)
        for _ in range(10):
            out = detector.update(35.0, 35.0)
        self.assertFalse(out["substitute"])
        # Single-sample spike: instantaneous substitution, no latch.
        out = detector.update(37.0, 35.0)
        self.assertTrue(out["substitute"])
        self.assertFalse(out["sensor_suspect"])
        for _ in range(3):
            out = detector.update(37.0, 35.0)
        self.assertTrue(out["sensor_suspect"])
        for _ in range(5):
            out = detector.update(35.0, 35.0)
        self.assertFalse(out["sensor_suspect"])


class S2Tests(unittest.TestCase):
    def test_detects_bias_and_dropout(self):
        filtr = MedianRatePlausibilityFilter(window=20, median_gate=1.0,
                                             rate_limit=500.0)
        for _ in range(30):
            filtr.update(35.0)
        entered = False
        for _ in range(5):
            out = filtr.update(45.0)
            entered |= out["sensor_suspect"]
        self.assertTrue(entered)
        filtr.reset()
        for _ in range(30):
            filtr.update(35.0)
        out = filtr.update(0.0)  # dropout: rate violation
        self.assertTrue(out["rate_violation"])

    def test_slow_drift_is_trusted_known_blind_spot(self):
        filtr = MedianRatePlausibilityFilter(window=20, median_gate=1.0,
                                             rate_limit=500.0)
        trusted = True
        value = 35.0
        for _ in range(200):
            value += 0.02  # 2 rad/s per second: within rate and median
            out = filtr.update(value)
            trusted &= out["trusted"]
        self.assertTrue(trusted)

    def test_suspect_samples_do_not_drag_the_median(self):
        filtr = MedianRatePlausibilityFilter(window=20, median_gate=1.0,
                                             rate_limit=500.0)
        for _ in range(30):
            filtr.update(35.0)
        for _ in range(50):
            filtr.update(45.0)
        out = filtr.update(35.05)
        # Median stayed near 35 (suspect samples were never ingested).
        self.assertAlmostEqual(out["estimate"], 35.0, delta=0.5)
        # The debounced state still needs exit_count healthy samples.
        self.assertTrue(out["sensor_suspect"])
        for _ in range(4):
            out = filtr.update(35.05)
        self.assertFalse(out["sensor_suspect"])
        self.assertTrue(out["trusted"])


class S3Tests(unittest.TestCase):
    def test_gain_is_stable(self):
        gain = design_luenberger_gain()
        self.assertEqual(gain.shape, (2, 1))
        self.assertTrue(np.isfinite(gain).all())

    def test_gain_places_poles(self):
        # Independent check of the Ackermann implementation: closed-loop
        # observer poles must equal the desired discrete poles.
        from baselines_v4 import _acker_2x2, _expm_small
        from ekf_observer import smooth_friction_derivative
        params = DCMotorParams()
        slope = smooth_friction_derivative(35.0, params)
        a_cont = np.array(
            [[-params.resistance / params.inductance,
              -params.back_emf_constant / params.inductance],
             [params.torque_constant / params.inertia,
              -(params.viscous_friction + slope) / params.inertia]])
        a_disc = np.asarray(_expm_small(a_cont * 0.01).real)
        desired = np.exp(4.0 * np.linalg.eigvals(a_cont) * 0.01)
        gain = _acker_2x2(a_disc, np.array([[1.0, 0.0]]), desired)
        placed = np.linalg.eigvals(a_disc - gain @ np.array([[1.0, 0.0]]))
        np.testing.assert_allclose(sorted(placed, key=abs),
                                   sorted(desired, key=abs), rtol=1e-6)

    def test_baselines_module_has_no_scipy_dependency(self):
        # Regression guard: scipy native calls after torch init abort
        # this container (OMP Error #15), so baselines_v4 stays numpy-only.
        source = (ROOT / "src" / "baselines_v4.py").read_text()
        self.assertNotIn("import scipy", source)
        self.assertNotIn("from scipy", source)
        self.assertNotIn("import torch", source)

    def test_observer_converges_on_nominal_plant(self):
        params = DCMotorParams()
        observer = LuenbergerBaseline(CusumCore(50.0, 0.0, 1.0, 100.0),
                                      params=params)
        state = np.zeros(2)
        dt = 0.01
        for _ in range(600):
            derivative = lambda val: dc_motor_dynamics(0.0, val, 6.0, 0.03,
                                                       params)
            k1 = derivative(state)
            k2 = derivative(state + dt * k1 / 2)
            k3 = derivative(state + dt * k2 / 2)
            k4 = derivative(state + dt * k3)
            state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            out = observer.update(6.0, float(state[0]), float(state[1]))
        self.assertAlmostEqual(out["estimate"], float(state[1]), delta=2.0)

    def test_detects_bias(self):
        observer = LuenbergerBaseline(CusumCore(1.0, 0.0, 0.1, 1.0))
        for _ in range(300):
            out = observer.update(6.0, 2.0, 35.0)
        healthy = out["estimate"]
        observer.detector.reset()
        for _ in range(20):
            out = observer.update(6.0, 2.0, healthy)
        self.assertFalse(out["sensor_suspect"])
        entered = False
        for _ in range(10):
            out = observer.update(6.0, 2.0, healthy + 10.0)
            entered |= out["sensor_suspect"]
        self.assertTrue(entered)


class CalibrationHelperTests(unittest.TestCase):
    def test_fixed_gate_matches_percentile(self):
        rng = np.random.default_rng(3)
        sequences = [rng.normal(0, 0.25, 500), rng.normal(0, 0.25, 500)]
        gate = calibrate_fixed_gate(sequences)
        self.assertAlmostEqual(
            gate, float(np.percentile(np.abs(np.concatenate(sequences)),
                                      99.9)))
        with self.assertRaises(ValueError):
            calibrate_fixed_gate([np.full(10, np.nan)])

    def test_cusum_grid_selection_is_consistent(self):
        rng = np.random.default_rng(4)
        sequences = [rng.normal(0, 0.25, 400) for _ in range(3)]
        result = calibrate_cusum_grid(sequences)
        grid = result["calibration_grid"]
        self.assertEqual([entry["percentile"] for entry in grid],
                         [90.0, 95.0, 97.5, 99.0, 99.5, 99.9])
        selected = next(entry for entry in grid
                        if entry["cusum_threshold"] == result["threshold"])
        before = grid[:grid.index(selected)]
        self.assertTrue(all(entry["validation_false_alarm_rate"] > 0.001
                            for entry in before))
        self.assertTrue(
            selected["validation_false_alarm_rate"] <= 0.001
            or selected is grid[-1])

    def test_plausibility_bounds(self):
        rng = np.random.default_rng(5)
        sequences = [35.0 + np.cumsum(rng.normal(0, 0.02, 300))
                     for _ in range(2)]
        result = calibrate_plausibility(sequences)
        self.assertGreater(result["median_gate"], 0)
        self.assertGreater(result["rate_limit"], 0)

    def test_detector_factory_schema(self):
        doc = {"instant_threshold": 0.9, "center": 0.01, "allowance": 0.14,
               "threshold": 1.5}
        core = detector_from_cusum_doc(doc)
        self.assertEqual((core.residual_gate, core.enter_count,
                          core.exit_count), (0.9, 3, 5))
        json.dumps(doc)  # schema must stay JSON-serializable


class BaselineCalibrationSmokeTests(unittest.TestCase):
    def test_smoke_calibration_schema(self):
        import tempfile
        sys.path.insert(0, str(ROOT / "scripts"))
        import v4_calibrate_baselines
        import v4_generate_dataset
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "smoke.npz"
            v4_generate_dataset.main(
                ["--smoke", "--out", str(dataset), "--seed", "99001",
                 "--exclude-kind", "closed_loop_mpc"])
            out = tmp_path / "baselines.json"
            v4_calibrate_baselines.main(
                ["--smoke", "--dataset", str(dataset), "--out", str(out)])
            doc = json.loads(out.read_text())
            for key in ("e2_ekf_cusum", "s1_fixed_gate", "s2_plausibility",
                        "s3_luenberger_cusum", "bootstrap_uncertainty"):
                self.assertIn(key, doc)
            self.assertGreater(doc["s1_fixed_gate"]["gate"], 0)
            self.assertGreater(doc["s2_plausibility"]["median_gate"], 0)
            self.assertGreater(doc["s2_plausibility"]["rate_limit"], 0)
            for key in ("threshold", "instant_threshold"):
                self.assertGreater(doc["e2_ekf_cusum"][key], 0)
                self.assertGreater(doc["s3_luenberger_cusum"][key], 0)
            for key, stats in doc["bootstrap_uncertainty"].items():
                with self.subTest(key=key):
                    self.assertLessEqual(stats["ci_lo"],
                                         stats["mean"] + 1e-9)
                    self.assertLessEqual(stats["mean"],
                                         stats["ci_hi"] + 1e-9)
                    self.assertEqual(stats["replicates"], 20)
            # Factory round-trip on the calibrated documents.
            e2_core = detector_from_cusum_doc(doc["e2_ekf_cusum"])
            s3_core = detector_from_cusum_doc(doc["s3_luenberger_cusum"])
            self.assertGreater(e2_core.threshold, 0)
            self.assertGreater(s3_core.threshold, 0)


if __name__ == "__main__":
    unittest.main()
