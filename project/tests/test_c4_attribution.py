import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import scripts.calibrate_c4_attribution as calibrator
from reliability import (
    C4_ATTRIBUTION_STATE_NAMES,
    DualVirtualSensorArbitrator,
    SuppressionEvidenceAction,
    SuppressionEvidencePolicy,
    ThreeWayAttributionState,
    ThreeWayConsistencyAttributor,
)


def attributor(**overrides):
    config = {
        "agreement_thresholds": (1.0, 1.0, 1.0),
        "disagreement_thresholds": (3.0, 3.0, 3.0),
        "ewma_alpha": 1.0,
        "enter_count": 1,
        "exit_count": 1,
    }
    config.update(overrides)
    return ThreeWayConsistencyAttributor(**config)


class CalibrationScriptWriteSafetyTests(unittest.TestCase):
    @staticmethod
    def _sample_calibration() -> dict:
        return {
            "calibration_version": "1.0.0",
            "diagnostic_filter_definition": {"alpha": 0.05},
            "raw_pairwise_quantiles": {"d_sm": {"p50": 1.0}},
            "trajectory_quantiles": [
                {"run_id": 7, "sample_count": 100, "raw": {"d_sm": {"p50": 1.0}}}
            ],
            "computed_thresholds": {
                "agreement_percentile": 75.0,
                "agreement": {"d_sm": 1.0},
            },
            "provenance": {
                "dataset": {"sha256": "dataset-hash"},
                "calibration_code": {
                    "path": "scripts/calibrate_c4_attribution.py",
                    "version": "1.0.0",
                    "sha256": "historical-generator-hash",
                },
                "runtime": {"python": "3.11.9", "numpy": "2.2.6", "torch": "2.13.0"},
            },
        }

    def test_default_check_mode_leaves_canonical_hash_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "c4_attribution_calibration.json"
            canonical.write_text('{"frozen": true}\n', encoding="utf-8")
            before = calibrator.sha256(canonical)

            args = calibrator.parse_args([])
            with mock.patch.object(calibrator, "OUTPUT", canonical):
                written = calibrator.write_requested_output(
                    {"recomputed": True},
                    args.output,
                    force=args.force,
                )

            self.assertIsNone(written)
            self.assertEqual(calibrator.sha256(canonical), before)

    def test_explicit_separate_output_writes_artifact_without_touching_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "c4_attribution_calibration.json"
            separate = root / "candidate_calibration.json"
            canonical.write_text('{"frozen": true}\n', encoding="utf-8")
            before = calibrator.sha256(canonical)
            payload = {"calibration_version": "1.0.0", "sentinel": 7}

            with mock.patch.object(calibrator, "OUTPUT", canonical):
                written = calibrator.write_requested_output(payload, separate)

            self.assertEqual(written, separate.resolve())
            self.assertEqual(json.loads(separate.read_text(encoding="utf-8")), payload)
            self.assertEqual(calibrator.sha256(canonical), before)

    def test_canonical_overwrite_is_refused_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "c4_attribution_calibration.json"
            canonical.write_text('{"frozen": true}\n', encoding="utf-8")
            before = calibrator.sha256(canonical)

            with mock.patch.object(calibrator, "OUTPUT", canonical):
                with self.assertRaisesRegex(RuntimeError, "without --force"):
                    calibrator.write_requested_output({"replacement": True}, canonical)

            self.assertEqual(calibrator.sha256(canonical), before)

    def test_force_allows_explicit_canonical_overwrite_only_in_temp_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "c4_attribution_calibration.json"
            canonical.write_text('{"frozen": true}\n', encoding="utf-8")
            payload = {"replacement": True}

            with mock.patch.object(calibrator, "OUTPUT", canonical):
                written = calibrator.write_requested_output(
                    payload,
                    canonical,
                    force=True,
                )

            self.assertEqual(written, canonical.resolve())
            self.assertEqual(json.loads(canonical.read_text(encoding="utf-8")), payload)

    def test_canonical_check_ignores_generation_metadata_and_tolerates_inference_noise(self):
        frozen = self._sample_calibration()
        recomputed = json.loads(json.dumps(frozen))
        recomputed["raw_pairwise_quantiles"]["d_sm"]["p50"] += 5e-7
        recomputed["provenance"]["calibration_code"]["sha256"] = "current-checker-hash"
        recomputed["provenance"]["runtime"] = {
            "python": "9.9.9",
            "numpy": "99.0",
            "torch": "99.0",
            "extra": "generation-only",
        }

        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "c4_attribution_calibration.json"
            canonical.write_text(json.dumps(frozen) + "\n", encoding="utf-8")
            before = calibrator.sha256(canonical)
            result = calibrator.check_canonical_equivalence(recomputed, canonical)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["differences"], [])
            self.assertEqual(calibrator.sha256(canonical), before)

    def test_canonical_check_rejects_exact_policy_and_structure_mismatch(self):
        frozen = self._sample_calibration()
        recomputed = json.loads(json.dumps(frozen))
        recomputed["diagnostic_filter_definition"]["alpha"] = 0.0500001
        recomputed["trajectory_quantiles"][0]["sample_count"] = 101
        del recomputed["provenance"]["dataset"]

        differences = calibrator.compare_calibrations(frozen, recomputed)

        self.assertTrue(any("diagnostic_filter_definition.alpha" in item for item in differences))
        self.assertTrue(any("trajectory_quantiles[0].sample_count" in item for item in differences))
        self.assertTrue(any("provenance.dataset" in item for item in differences))

    def test_canonical_check_rejects_large_inference_change_and_exits_nonzero(self):
        frozen = self._sample_calibration()
        recomputed = json.loads(json.dumps(frozen))
        recomputed["computed_thresholds"]["agreement"]["d_sm"] = 1.001

        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "c4_attribution_calibration.json"
            canonical.write_text(json.dumps(frozen) + "\n", encoding="utf-8")
            result = calibrator.check_canonical_equivalence(recomputed, canonical)

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("computed_thresholds.agreement.d_sm" in item for item in result["differences"]))
        with self.assertRaises(SystemExit) as raised:
            calibrator.require_check_passed(result)
        self.assertEqual(raised.exception.code, 1)


class ThreeWayConsistencyAttributorTests(unittest.TestCase):
    def test_locked_default_persistence_is_three_enter_five_exit(self):
        gate = ThreeWayConsistencyAttributor(
            agreement_thresholds=(1.0, 1.0, 1.0),
            disagreement_thresholds=(3.0, 3.0, 3.0),
            ewma_alpha=0.1,
        )
        self.assertEqual(gate.enter_count, 3)
        self.assertEqual(gate.exit_count, 5)

    def test_state_enum_names_are_stable(self):
        self.assertEqual(
            C4_ATTRIBUTION_STATE_NAMES.tolist(),
            [
                "NORMAL",
                "LIKELY_SENSOR_FAULT",
                "LIKELY_PLANT_OR_MAIN_MISMATCH",
                "LIKELY_AUX_MISMATCH",
                "AMBIGUOUS",
            ],
        )
        self.assertEqual(int(ThreeWayAttributionState.AMBIGUOUS), 4)

    def test_three_way_logical_patterns(self):
        cases = [
            ((10.0, 10.2, 10.1), "NORMAL", False),
            ((10.0, 0.0, 0.2), "LIKELY_SENSOR_FAULT", False),
            ((10.0, 0.0, 10.2), "LIKELY_PLANT_OR_MAIN_MISMATCH", True),
            ((10.0, 10.2, 0.0), "LIKELY_AUX_MISMATCH", False),
            ((0.0, 5.0, 10.0), "AMBIGUOUS", False),
        ]
        for estimates, expected_state, expected_suppressed in cases:
            with self.subTest(expected_state=expected_state):
                result = attributor().update(
                    *estimates,
                    auxiliary_trusted=True,
                    detector_requests_entry=True,
                )
                self.assertEqual(result["state"], expected_state)
                self.assertEqual(result["entry_suppressed"], expected_suppressed)
                self.assertEqual(
                    result["entry_authorized"], not expected_suppressed
                )

    def test_pairwise_disagreements_and_ewma_are_exposed(self):
        gate = attributor(ewma_alpha=0.5)
        first = gate.update(
            10.0,
            8.0,
            9.0,
            auxiliary_trusted=True,
            detector_requests_entry=False,
        )
        second = gate.update(
            10.0,
            4.0,
            8.0,
            auxiliary_trusted=True,
            detector_requests_entry=False,
        )
        self.assertEqual((first["d_sm"], first["d_sa"], first["d_ma"]), (2.0, 1.0, 1.0))
        self.assertEqual((second["d_sm"], second["d_sa"], second["d_ma"]), (6.0, 2.0, 4.0))
        self.assertEqual(
            (
                second["filtered_d_sm"],
                second["filtered_d_sa"],
                second["filtered_d_ma"],
            ),
            (4.0, 1.5, 2.5),
        )

    def test_ewma_is_diagnostic_only_and_cannot_change_candidate_or_veto(self):
        gate = attributor(ewma_alpha=0.01, enter_count=1, exit_count=1)
        gate.update(
            10.0,
            10.1,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=False,
        )
        plant = gate.update(
            10.0,
            0.0,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=True,
        )

        # The EWMA remains close to the prior nominal sample and is not itself
        # a plant-mismatch pattern.  Raw distances alone must drive C4 logic.
        self.assertLess(plant["filtered_d_sm"], 1.0)
        self.assertLess(plant["filtered_d_ma"], 1.0)
        self.assertEqual(plant["raw_state"], "LIKELY_PLANT_OR_MAIN_MISMATCH")
        self.assertEqual(plant["candidate_state"], "LIKELY_PLANT_OR_MAIN_MISMATCH")
        self.assertEqual(plant["state"], "LIKELY_PLANT_OR_MAIN_MISMATCH")
        self.assertTrue(plant["entry_suppressed"])

    def test_missing_or_nonfinite_auxiliary_is_ambiguous_and_preserves_entry(self):
        for auxiliary in (None, np.nan, np.inf):
            with self.subTest(auxiliary=auxiliary):
                result = attributor().update(
                    10.0,
                    0.0,
                    auxiliary,
                    auxiliary_trusted=True,
                    detector_requests_entry=True,
                )
                self.assertEqual(result["state"], "AMBIGUOUS")
                self.assertFalse(result["auxiliary_trusted"])
                self.assertFalse(result["entry_suppressed"])
                self.assertTrue(result["entry_authorized"])
                self.assertEqual(result["suppression_evidence_action"], "NONE")

    def test_auxiliary_parameter_mismatch_cannot_veto_entry(self):
        result = attributor().update(
            10.0,
            0.0,
            10.2,
            auxiliary_trusted=False,
            detector_requests_entry=True,
        )
        self.assertEqual(result["state"], "AMBIGUOUS")
        self.assertFalse(result["entry_suppressed"])
        self.assertTrue(result["entry_authorized"])

    def test_persistence_prevents_one_sample_chatter(self):
        gate = attributor(enter_count=2, exit_count=2)
        nominal = gate.update(
            10.0,
            10.2,
            10.1,
            auxiliary_trusted=True,
            detector_requests_entry=False,
        )
        transient = gate.update(
            10.0,
            0.0,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=True,
        )
        recovered = gate.update(
            10.0,
            10.2,
            10.1,
            auxiliary_trusted=True,
            detector_requests_entry=False,
        )
        self.assertEqual(nominal["state"], "NORMAL")
        self.assertEqual(transient["candidate_state"], "LIKELY_PLANT_OR_MAIN_MISMATCH")
        self.assertEqual(transient["state"], "NORMAL")
        self.assertFalse(transient["entry_suppressed"])
        self.assertEqual(recovered["state"], "NORMAL")
        self.assertEqual(recovered["state_switches"], 0)

    def test_suppression_evidence_policy_resets_once_then_freezes(self):
        gate = attributor()
        policy = SuppressionEvidencePolicy()
        first = gate.update(
            10.0,
            0.0,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=True,
        )
        self.assertEqual(first["suppression_evidence_action"], "RESET")
        self.assertEqual(
            policy.apply(SuppressionEvidenceAction.RESET, 8.0, 2.0, 12.0, 3.0),
            (0.0, 0.0),
        )

        continuing = gate.update(
            10.0,
            0.0,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=False,
        )
        self.assertEqual(continuing["suppression_evidence_action"], "FREEZE")
        self.assertEqual(
            policy.apply("FREEZE", 0.0, 0.0, 6.0, 1.0),
            (0.0, 0.0),
        )

    def test_raw_sensor_fault_pattern_stops_veto_before_persisted_state_changes(self):
        gate = attributor(ewma_alpha=0.1, enter_count=3, exit_count=3)
        for _ in range(3):
            plant = gate.update(
                10.0,
                0.0,
                10.2,
                auxiliary_trusted=True,
                detector_requests_entry=True,
            )
        self.assertEqual(plant["state"], "LIKELY_PLANT_OR_MAIN_MISMATCH")
        self.assertTrue(plant["entry_suppressed"])

        later_fault = gate.update(
            20.0,
            10.0,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=True,
        )
        self.assertEqual(later_fault["raw_state"], "LIKELY_SENSOR_FAULT")
        self.assertEqual(later_fault["state"], "LIKELY_PLANT_OR_MAIN_MISMATCH")
        self.assertFalse(later_fault["entry_suppressed"])
        self.assertTrue(later_fault["entry_authorized"])
        self.assertEqual(later_fault["suppression_evidence_action"], "NONE")

    def test_plant_mismatch_then_later_sensor_fault_reenables_entry(self):
        gate = attributor()

        gate.update(
            10.0,
            10.1,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=False,
        )
        suppressed = gate.update(
            10.0,
            0.0,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=True,
        )
        frozen = gate.update(
            10.0,
            0.0,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=False,
        )
        nominal_again = gate.update(
            10.0,
            10.1,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=False,
        )
        later_fault = gate.update(
            20.0,
            10.0,
            10.2,
            auxiliary_trusted=True,
            detector_requests_entry=True,
        )

        self.assertTrue(suppressed["entry_suppressed"])
        self.assertEqual(frozen["suppression_evidence_action"], "FREEZE")
        self.assertFalse(nominal_again["suppression_active"])
        self.assertEqual(nominal_again["suppression_evidence_action"], "NONE")
        self.assertEqual(later_fault["state"], "LIKELY_SENSOR_FAULT")
        self.assertTrue(later_fault["entry_authorized"])
        self.assertFalse(later_fault["entry_suppressed"])
        self.assertEqual(later_fault["suppression_evidence_action"], "NONE")

    def test_online_attributor_has_no_oracle_inputs(self):
        parameters = inspect.signature(ThreeWayConsistencyAttributor.update).parameters
        forbidden_parameters = {
            "y_true",
            "true_speed",
            "scenario",
            "scenario_name",
            "fault_flag",
            "fault_label",
            "load_disturbance",
        }
        self.assertTrue(forbidden_parameters.isdisjoint(parameters))
        source = inspect.getsource(ThreeWayConsistencyAttributor.update)
        for token in ("y_true", "true_speed", "scenario_name", "fault_label"):
            self.assertNotIn(token, source)

    def test_c3_arbitrator_remains_independent_of_c4(self):
        signature = inspect.signature(DualVirtualSensorArbitrator.update)
        self.assertNotIn("attribution", signature.parameters)
        self.assertNotIn("ThreeWayConsistencyAttributor", inspect.getsource(DualVirtualSensorArbitrator))

        arb = DualVirtualSensorArbitrator(
            agreement_threshold=2.0,
            param_mismatch_threshold=5.0,
            param_mismatch_recovery_threshold=1.0,
            param_mismatch_recovery_count=3,
            ewma_alpha=1.0,
            startup_blanking_time=0.0,
            aux_speed_bounds=(0.0, 100.0),
        )
        physical = arb.update(1.0, 30.0, 29.0, 30.0, True, 25.0)
        main = arb.update(2.0, 50.0, 29.0, 30.0, False, 25.0)
        auxiliary = arb.update(3.0, 50.0, 20.0, 30.0, False, 25.0)
        arb.aux_param_mismatch = True
        fallback = arb.update(4.0, 50.0, 20.0, 30.0, False, 25.0)
        self.assertEqual(
            [(item["source_code"], item["feedback"]) for item in [physical, main, auxiliary, fallback]],
            [(0, 30.0), (1, 29.0), (2, 30.0), (3, 25.0)],
        )


if __name__ == "__main__":
    unittest.main()
