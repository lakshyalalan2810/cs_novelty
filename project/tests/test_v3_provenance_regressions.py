import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import calibrate_v3_arbitration as calibration_script
import evaluate_v3_closed_loop as evaluator
import verify_v3_results as verifier
from reliability import DualVirtualSensorArbitrator


class V3ProvenanceRegressionTests(unittest.TestCase):
    def test_calibration_and_verifier_ewma_match_deployed_startup_blanking(self):
        times = np.array([0.2, 0.5, 1.0, 1.1])
        residual = np.array([100.0, 100.0, 2.0, 4.0])
        expected = np.array([1.0, 2.5])

        calibration_values = calibration_script.post_blanking_ewma(times, residual, 0.5, 1.0)
        verifier_values = verifier.independently_post_blanking_ewma(times, residual, 0.5, 1.0)
        np.testing.assert_allclose(calibration_values, expected, rtol=0, atol=0)
        np.testing.assert_allclose(verifier_values, expected, rtol=0, atol=0)

        arbitrator = DualVirtualSensorArbitrator(
            agreement_threshold=2.0,
            param_mismatch_threshold=10.0,
            param_mismatch_recovery_threshold=1.0,
            param_mismatch_recovery_count=3,
            ewma_alpha=0.5,
            startup_blanking_time=1.0,
            aux_speed_bounds=(0.0, 200.0),
        )
        deployed_values = []
        for time_value, residual_value in zip(times, residual):
            arbitrator.update(time_value, residual_value, 0.0, 0.0, True, 0.0)
            if time_value >= 1.0:
                deployed_values.append(arbitrator.ewma_aux_trusted)
        np.testing.assert_allclose(deployed_values, expected, rtol=0, atol=0)

    def test_freeze_record_binding_rejects_tampered_canonical_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            matrix = Path(temp_dir) / "matrix.csv"
            matrix.write_bytes(b"controller,seed\nC3,29026\n")
            digest = hashlib.sha256(matrix.read_bytes()).hexdigest()
            record = {
                "artifacts": {
                    "v3_final_275_run_matrix": {
                        "path": "results/metrics/v3_final_11scenario_runs.csv",
                        "sha256": digest,
                    }
                }
            }
            self.assertEqual(
                verifier.verify_recorded_artifact(
                    record,
                    "v3_final_275_run_matrix",
                    matrix,
                    "results/metrics/v3_final_11scenario_runs.csv",
                    digest,
                ),
                digest,
            )
            matrix.write_bytes(matrix.read_bytes() + b"C3,29027\n")
            with self.assertRaises(AssertionError):
                verifier.verify_recorded_artifact(
                    record,
                    "v3_final_275_run_matrix",
                    matrix,
                    "results/metrics/v3_final_11scenario_runs.csv",
                    digest,
                )

    def test_verification_report_is_stable_across_tolerated_reconstruction_noise(self):
        frozen = {
            "agreement_threshold": 4.341867446899414,
            "param_mismatch_threshold": 8.700958862330452,
            "param_mismatch_recovery_threshold": 5.0486354952328725,
            "aux_recovery_gate": 9.047693252563477,
        }
        verification_path = PROJECT / "results/metrics/v3_verification_report.json"
        before_bytes = verification_path.read_bytes()
        before_report = verifier.build_report()
        noisy = {name: value + 1e-6 for name, value in frozen.items()}
        noisy_differences = verifier.verify_reconstruction_equivalence(
            noisy,
            frozen,
            frozen,
        )

        self.assertGreater(max(noisy_differences.values()), 0.0)
        self.assertEqual(verification_path.read_bytes(), before_bytes)
        self.assertEqual(verifier.build_report(), before_report)

        outside_tolerance = dict(frozen)
        outside_tolerance["agreement_threshold"] += 1e-2
        with self.assertRaises(AssertionError):
            verifier.verify_reconstruction_equivalence(
                outside_tolerance,
                frozen,
                frozen,
            )

    def test_evaluator_argparse_preserves_flag_and_rejects_unknown_args(self):
        args = evaluator.parse_args(["--reference-events-only"])
        self.assertTrue(args.reference_events_only)
        with self.assertRaises(SystemExit):
            evaluator.parse_args(["--reference-event-only"])

if __name__ == "__main__":
    unittest.main()
