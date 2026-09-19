import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reliability import DualVirtualSensorArbitrator, SensorReliabilityMonitor


def arbitrator(**overrides):
    config = {
        "agreement_threshold": 2.0,
        "param_mismatch_threshold": 5.0,
        "param_mismatch_recovery_threshold": 1.0,
        "param_mismatch_recovery_count": 3,
        "ewma_alpha": 1.0,
        "startup_blanking_time": 0.0,
        "aux_speed_bounds": (0.0, 100.0),
    }
    config.update(overrides)
    return DualVirtualSensorArbitrator(**config)


class CorrectiveV3Tests(unittest.TestCase):
    def test_auxiliary_recovery_evidence_is_mandatory(self):
        monitor = SensorReliabilityMonitor(1.0, 0.0, 0.1, 0.5, 2, 2, aux_recovery_gate=3.0)
        monitor.update(2.0, aux_residual=0.0)
        monitor.update(2.0, aux_residual=0.0)
        for missing in [None, np.nan] * 25:
            decision = monitor.update(0.0, aux_residual=missing)
        self.assertTrue(decision["sensor_suspect"])
        self.assertEqual(decision["healthy_run"], 0)
        monitor.update(0.0, aux_residual=1.0)
        self.assertFalse(monitor.update(0.0, aux_residual=1.0)["sensor_suspect"])

    def test_source_codes_match_actual_feedback(self):
        arb = arbitrator()
        physical = arb.update(1.0, 30.0, 29.0, 30.0, True, 25.0)
        main = arb.update(2.0, 50.0, 29.0, 30.0, False, 25.0)
        auxiliary = arb.update(3.0, 50.0, 20.0, 30.0, False, 25.0)
        arb.aux_param_mismatch = True
        fallback = arb.update(4.0, 50.0, 20.0, 30.0, False, 25.0)
        self.assertEqual(
            [(item["source_code"], item["feedback"]) for item in [physical, main, auxiliary, fallback]],
            [(0, 30.0), (1, 29.0), (2, 30.0), (3, 25.0)],
        )
        self.assertNotEqual(fallback["feedback"], 20.0)

    def test_fallback_requires_finite_speed(self):
        arb = arbitrator()
        arb.aux_param_mismatch = True
        with self.assertRaises(ValueError):
            arb.update(2.0, 50.0, 20.0, 30.0, False, np.nan)

    def test_transient_parameter_mismatch_recovers_for_later_fault(self):
        arb = arbitrator()
        arb.update(1.0, 30.0, 30.0, 30.0, True, 30.0)
        arb.update(2.0, 30.0, 30.0, 20.0, True, 30.0)
        self.assertTrue(arb.aux_param_mismatch)
        for index in range(3):
            result = arb.update(3.0 + index, 30.0, 30.0, 30.0, True, 30.0)
        self.assertFalse(result["aux_param_mismatch"])
        later_fault = arb.update(6.0, 50.0, 15.0, 30.0, False, 30.0)
        self.assertEqual(later_fault["source"], "AUX_VIRTUAL")

    def test_permanent_parameter_mismatch_stays_protected(self):
        arb = arbitrator()
        for index in range(20):
            result = arb.update(float(index), 30.0, 30.0, 20.0, True, 30.0)
        self.assertTrue(result["aux_param_mismatch"])
        fault = arb.update(21.0, 50.0, 15.0, 20.0, False, 30.0)
        self.assertEqual(fault["source"], "FALLBACK")

    def test_bad_sample_resets_mismatch_recovery_run(self):
        arb = arbitrator()
        arb.update(1.0, 30.0, 30.0, 20.0, True, 30.0)
        arb.update(2.0, 30.0, 30.0, 30.0, True, 30.0)
        self.assertEqual(arb.aux_param_recovery_run, 1)
        arb.update(3.0, 30.0, 30.0, np.nan, True, 30.0)
        self.assertEqual(arb.aux_param_recovery_run, 0)

    def test_arbitration_does_not_chatter(self):
        arb = arbitrator()
        arb.update(1.0, 30.0, 30.0, 30.0, True, 30.0)
        for index in range(10):
            arb.update(2.0 + index, 50.0, 30.0, 30.5, False, 30.0)
        for index in range(10):
            arb.update(12.0 + index, 50.0, 20.0, 30.0, False, 30.0)
        self.assertLessEqual(arb.total_switches, 2)

    def test_online_arbitrator_has_no_truth_input(self):
        source = inspect.getsource(DualVirtualSensorArbitrator.update)
        self.assertNotIn("y_true", source)
        self.assertNotIn("true_speed", source)


if __name__ == "__main__":
    unittest.main()
