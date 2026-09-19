"""V4 detector tests: frozen bit-identity plus per-flag behavior."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reliability import SensorReliabilityMonitor
from reliability_v4 import (
    FROZEN_RESULT_KEYS,
    ReliabilityMonitorV4,
    v4_from_sensor_calibration_path,
)

CAL_2026 = (ROOT / "results/training_seed_robustness/configs"
            / "sensor_seed2026_calibration.json")
TRACE = ROOT / "results/metrics/recovery_regression_trace.csv"


def frozen_from_v4(monitor: ReliabilityMonitorV4) -> SensorReliabilityMonitor:
    return SensorReliabilityMonitor(
        residual_gate=monitor.residual_gate, center=monitor.center,
        allowance=monitor.allowance, threshold=monitor.threshold,
        enter_count=monitor.enter_count, exit_count=monitor.exit_count,
        aux_recovery_gate=monitor.aux_recovery_gate)


class FrozenBitIdentityTests(unittest.TestCase):
    def assert_identical(self, frozen_out: dict, v4_out: dict) -> None:
        for key in FROZEN_RESULT_KEYS:
            with self.subTest(key=key):
                left, right = frozen_out[key], v4_out[key]
                if isinstance(left, float) and isinstance(right, float):
                    # Exact equality, compared NaN-aware since NaN != NaN.
                    self.assertTrue(
                        (left == right)
                        or (np.isnan(left) and np.isnan(right)))
                else:
                    self.assertEqual(left, right)

    def test_defaults_bit_identical_on_recorded_residuals(self):
        residuals = pd.read_csv(TRACE)["residual"].to_numpy(float)
        self.assertEqual(len(residuals), 1200)
        v4 = v4_from_sensor_calibration_path(CAL_2026)
        frozen = frozen_from_v4(v4)
        for value in residuals:
            self.assert_identical(frozen.update(float(value)),
                                  v4.update(float(value)))

    def test_defaults_bit_identical_with_aux_gate_and_edges(self):
        v4 = v4_from_sensor_calibration_path(
            CAL_2026, aux_recovery_gate=9.047693253)
        frozen = frozen_from_v4(v4)
        sequence = ([0.0] * 30 + [3.0] * 60 + [0.0] * 200
                    + [float("nan")] + [0.0] * 10 + [float("inf")]
                    + [-5.0] * 10 + [0.0] * 200)
        aux_sequence = [0.0] * len(sequence)
        aux_sequence[100] = 50.0  # aux disagreement mid-recovery
        for value, aux in zip(sequence, aux_sequence):
            self.assert_identical(frozen.update(value, aux_residual=aux),
                                  v4.update(value, aux_residual=aux))

    def test_builder_reads_frozen_calibration(self):
        v4 = v4_from_sensor_calibration_path(CAL_2026)
        self.assertAlmostEqual(v4.residual_gate, 0.937210063934329)
        self.assertAlmostEqual(v4.allowance, 0.14038589929835443)
        self.assertAlmostEqual(v4.threshold, 1.5696827323213676)
        self.assertEqual((v4.enter_count, v4.exit_count), (3, 5))
        self.assertIsNone(v4.cusum_clamp_multiple)
        self.assertEqual(v4.warmup_blank_samples, 0)
        self.assertIsNone(v4.witness_residual_gate)
        self.assertEqual(v4.recovery_boost, 0.0)


class ClampTests(unittest.TestCase):
    def test_clamp_bounds_accumulator_and_discharge(self):
        base = dict(residual_gate=0.937, center=0.0, allowance=0.14,
                    threshold=1.57)
        frozen = SensorReliabilityMonitor(**base)
        clamped = ReliabilityMonitorV4(**base, cusum_clamp_multiple=2.0)
        # Long fault: frozen CUSUM grows unbounded, clamped caps at 2*thr.
        for _ in range(400):
            frozen.update(3.0)
            out = clamped.update(3.0)
        self.assertGreater(frozen.positive, 2.0 * 1.57)
        self.assertEqual(clamped.positive, 2.0 * 1.57)
        self.assertTrue(out["cusum_clamped"])
        # Frozen accumulation scales with fault duration (no clamp);
        # clamped accumulation does not.
        short_frozen = SensorReliabilityMonitor(**base)
        short_clamped = ReliabilityMonitorV4(**base, cusum_clamp_multiple=2.0)
        for _ in range(100):
            short_frozen.update(3.0)
            short_clamped.update(3.0)
        self.assertAlmostEqual(frozen.positive / short_frozen.positive, 4.0,
                               places=6)
        self.assertEqual(short_clamped.positive, clamped.positive)
        # Discharge on healthy residuals: clamped recovers quickly, frozen
        # needs ~accumulation/allowance samples (here ~8000+).
        frozen_steps = clamped_steps = None
        for step in range(12000):
            frozen.update(0.0)
            out_c = clamped.update(0.0)
            if frozen_steps is None and not frozen.active:
                frozen_steps = step
            if clamped_steps is None and not out_c["sensor_suspect"]:
                clamped_steps = step
            if frozen_steps is not None and clamped_steps is not None:
                break
        self.assertIsNotNone(clamped_steps)
        self.assertIsNotNone(frozen_steps)
        self.assertLess(clamped_steps, frozen_steps)
        # Analytic bound: k*thr/(allowance) + exit_count, with margin.
        self.assertLessEqual(clamped_steps, int(2.0 * 1.57 / 0.14) + 5 + 2)


class WarmupTests(unittest.TestCase):
    def test_warmup_suppresses_entry_and_trusts_sensor(self):
        monitor = ReliabilityMonitorV4(
            residual_gate=0.937, center=0.0, allowance=0.14, threshold=1.57,
            warmup_blank_samples=10, warmup_trust_sensor=True)
        for _ in range(10):
            out = monitor.update(3.0)
            self.assertTrue(out["warmup_active"])
            self.assertFalse(out["sensor_suspect"])
            self.assertFalse(out["substitute"])
            self.assertTrue(out["trusted"])
        # Fault persists past warmup: entry follows after enter_count.
        entered = False
        for _ in range(10):
            out = monitor.update(3.0)
            entered |= out["sensor_suspect"]
        self.assertTrue(entered)

    def test_warmup_without_trust_keeps_instantaneous_path(self):
        monitor = ReliabilityMonitorV4(
            residual_gate=0.937, center=0.0, allowance=0.14, threshold=1.57,
            warmup_blank_samples=10, warmup_trust_sensor=False)
        out = monitor.update(3.0)
        self.assertTrue(out["warmup_active"])
        self.assertFalse(out["sensor_suspect"])  # no entry ...
        self.assertTrue(out["substitute"])  # ... but instantaneous substitution


class WitnessTests(unittest.TestCase):
    def test_witness_agreement_suppresses_entry(self):
        monitor = ReliabilityMonitorV4(
            residual_gate=0.937, center=0.0, allowance=0.14, threshold=1.57,
            witness_residual_gate=0.937)
        # Main residual faulty-looking, witness agrees with sensor.
        for _ in range(50):
            out = monitor.update(3.0, witness_residual=0.05)
        self.assertFalse(out["sensor_suspect"])
        self.assertFalse(out["witness_abnormal"])
        self.assertFalse(out["witness_unavailable"])

    def test_witness_disagreement_allows_entry(self):
        monitor = ReliabilityMonitorV4(
            residual_gate=0.937, center=0.0, allowance=0.14, threshold=1.57,
            witness_residual_gate=0.937)
        entered = False
        for _ in range(50):
            out = monitor.update(3.0, witness_residual=2.5)
            entered |= out["sensor_suspect"]
        self.assertTrue(entered)
        self.assertTrue(out["witness_abnormal"])

    def test_missing_witness_fails_safe_to_main_only(self):
        monitor = ReliabilityMonitorV4(
            residual_gate=0.937, center=0.0, allowance=0.14, threshold=1.57,
            witness_residual_gate=0.937)
        entered = False
        for _ in range(50):
            out = monitor.update(3.0, witness_residual=None)
            entered |= out["sensor_suspect"]
        self.assertTrue(entered)  # frozen-like entry, not silent suppression
        self.assertTrue(out["witness_unavailable"])
        out = monitor.update(3.0, witness_residual=float("nan"))
        self.assertTrue(out["witness_unavailable"])


class RecoveryBoostTests(unittest.TestCase):
    def test_boost_discharges_faster_than_zero(self):
        base = dict(residual_gate=0.937, center=0.0, allowance=0.14,
                    threshold=1.57, cusum_clamp_multiple=2.0)
        plain = ReliabilityMonitorV4(**base)
        boosted = ReliabilityMonitorV4(**base, recovery_boost=0.5)
        for _ in range(100):
            plain.update(3.0)
            boosted.update(3.0)
        self.assertTrue(plain.active and boosted.active)
        plain_steps = boosted_steps = None
        for step in range(1000):
            plain.update(0.0)
            out_b = boosted.update(0.0)
            if plain_steps is None and not plain.active:
                plain_steps = step
            if boosted_steps is None and not out_b["sensor_suspect"]:
                boosted_steps = step
        self.assertLess(boosted_steps, plain_steps)


class AblationConfigTests(unittest.TestCase):
    def test_factorial_cells_present_and_buildable(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import v4_detector_configs as configs
        for name in ("clamp_only", "gating_only", "clamp_gating",
                     "witness_only", "frozen_baseline", "full"):
            self.assertIn(name, configs.VARIANTS)
        base = v4_from_sensor_calibration_path(CAL_2026)
        frozen = configs.resolve_variant(base, "frozen_baseline")
        self.assertIsNone(frozen.cusum_clamp_multiple)
        self.assertEqual(frozen.warmup_blank_samples, 0)
        self.assertIsNone(frozen.witness_residual_gate)
        full = configs.resolve_variant(base, "full")
        self.assertEqual(full.cusum_clamp_multiple, configs.CLAMP_K)
        self.assertEqual(full.warmup_blank_samples,
                         configs.WARMUP_BLANK_SAMPLES)
        self.assertEqual(full.witness_residual_gate, base.residual_gate)
        with self.assertRaises(ValueError):
            configs.resolve_variant(base, "nonexistent")


if __name__ == "__main__":
    unittest.main()
