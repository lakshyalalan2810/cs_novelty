import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import scripts.evaluate_c4_closed_loop as evaluator
import scripts.verify_c4_results as verifier


FINAL_SEEDS = [39026, 39027, 39028, 39029, 39030]


def _exact_final_attributor_block() -> dict:
    calibration_path = evaluator.PROJECT / "results/configs/c4_attribution_calibration.json"
    params = evaluator.load_c4_attribution_parameters(calibration_path)
    return {
        "agreement_thresholds": list(params["agreement_thresholds"]),
        "disagreement_thresholds": list(params["disagreement_thresholds"]),
        "ewma_alpha": params["ewma_alpha"],
        "enter_count": params["enter_count"],
        "exit_count": params["exit_count"],
    }


def _complete_final_config(stage_gate_hash: str = "gate-hash", development_config_hash: str = "dev-hash") -> dict:
    calibration_path = evaluator.PROJECT / "results/configs/c4_attribution_calibration.json"
    provenance_path = evaluator.PROJECT / "results/configs/c4_candidate_seed_audit.json"
    return {
        "final_holdout_seeds": FINAL_SEEDS.copy(),
        "scenarios": evaluator.FINAL_SCENARIOS.copy(),
        "candidate_seed_audit": "results/configs/c4_candidate_seed_audit.json",
        "attributor": _exact_final_attributor_block(),
        "hashes": {
            "c4_development_stage_gate": stage_gate_hash,
            "c4_development_config": development_config_hash,
            "c4_attribution_calibration": evaluator.sha256(calibration_path),
            "c4_candidate_seed_audit": evaluator.sha256(provenance_path),
        },
    }


class SeedCoercionRegressionTests(unittest.TestCase):
    def test_accepts_canonical_integer_strings_after_coercion(self):
        self.assertEqual(
            evaluator._coerce_seed_list(["39026", "39027", "39028", "39029", "39030"], label="test seeds"),
            FINAL_SEEDS,
        )

    def test_rejects_mixed_type_duplicate_after_coercion(self):
        with self.assertRaisesRegex(RuntimeError, "unique seeds after integer coercion"):
            evaluator._coerce_seed_list([39026, "39026", 39028, 39029, 39030], label="test seeds")

    def test_rejects_boolean_seed(self):
        with self.assertRaisesRegex(RuntimeError, "boolean seed"):
            evaluator._coerce_seed_list([True, 39027, 39028, 39029, 39030], label="test seeds")

    def test_rejects_nonintegral_float(self):
        with self.assertRaisesRegex(RuntimeError, "non-integral seed"):
            evaluator._coerce_seed_list([39026.5, 39027, 39028, 39029, 39030], label="test seeds")

    def test_rejects_noncanonical_integer_string(self):
        with self.assertRaisesRegex(RuntimeError, "non-canonical integer seed"):
            evaluator._coerce_seed_list(["039026", 39027, 39028, 39029, 39030], label="test seeds")


class MissingLatencyPolicyRegressionTests(unittest.TestCase):
    @staticmethod
    def _frame(values):
        return pd.DataFrame(
            {
                "scenario": ["sensor_bias_15"] * 5,
                "controller": ["C3_arbitration_MPC"] * 5,
                "detection_latency_s": values,
            }
        )

    def test_policy_text_is_identical(self):
        self.assertEqual(evaluator.LATENCY_POLICY, verifier.LATENCY_POLICY)

    def test_finite_latency_mean_matches_between_evaluator_and_verifier(self):
        runs = self._frame([0.10, 0.15, 0.20, 0.25, 0.30])
        eval_mean, eval_complete = evaluator._complete_mean_latency(
            runs, "sensor_bias_15", "C3_arbitration_MPC"
        )
        verify_mean, verify_complete = verifier._latency_mean_and_complete(
            runs, "C3_arbitration_MPC", "sensor_bias_15", "detection_latency_s"
        )
        self.assertTrue(eval_complete)
        self.assertTrue(verify_complete)
        self.assertEqual(eval_mean, verify_mean)

    def test_missing_or_nonfinite_latency_is_incomplete_in_both(self):
        for bad_value in (np.nan, np.inf):
            with self.subTest(bad_value=bad_value):
                runs = self._frame([0.10, 0.15, bad_value, 0.25, 0.30])
                eval_mean, eval_complete = evaluator._complete_mean_latency(
                    runs, "sensor_bias_15", "C3_arbitration_MPC"
                )
                verify_mean, verify_complete = verifier._latency_mean_and_complete(
                    runs, "C3_arbitration_MPC", "sensor_bias_15", "detection_latency_s"
                )
                self.assertFalse(eval_complete)
                self.assertFalse(verify_complete)
                self.assertTrue(np.isnan(eval_mean))
                self.assertTrue(np.isnan(verify_mean))


class StateFractionSchemaRegressionTests(unittest.TestCase):
    def test_saved_development_state_fraction_columns_resolve(self):
        runs = pd.read_csv(
            evaluator.PROJECT / "results/metrics/c4_development_runs.csv",
            float_precision="round_trip",
        )
        c4 = runs[runs["controller"] == "C4_attribution_MPC"]

        self.assertEqual(
            verifier._state_fraction_columns(c4),
            {
                "NORMAL": "attr_frac_normal",
                "LIKELY_SENSOR_FAULT": "attr_frac_likely_sensor_fault",
                "LIKELY_PLANT_OR_MAIN_MISMATCH": "attr_frac_likely_plant_or_main_mismatch",
                "LIKELY_AUX_MISMATCH": "attr_frac_likely_aux_mismatch",
                "AMBIGUOUS": "attr_frac_ambiguous",
            },
        )
        verifier.verify_attribution_metrics(runs, expected_c4_rows=35)

    def test_saved_development_gate_remains_no_go(self):
        metrics = evaluator.PROJECT / "results/metrics"
        configs = evaluator.PROJECT / "results/configs"
        runs = pd.read_csv(metrics / "c4_development_runs.csv", float_precision="round_trip")
        criteria = json.loads((configs / "c4_go_no_go_criteria.json").read_text(encoding="utf-8"))
        saved_gate = json.loads((metrics / "c4_development_stage_gate.json").read_text(encoding="utf-8"))
        saved_summary = json.loads((metrics / "c4_development_summary.json").read_text(encoding="utf-8"))

        decision, _ = verifier.independently_assess_development_gate(runs, criteria)

        self.assertEqual(decision, "NO_GO")
        self.assertEqual(saved_gate["decision"], "NO_GO")
        self.assertEqual(saved_summary["development_stage_gate"]["decision"], "NO_GO")


class AttributionEventDiagnosticRegressionTests(unittest.TestCase):
    @staticmethod
    def _saved_development_events() -> pd.DataFrame:
        return pd.read_csv(
            evaluator.PROJECT / "results/metrics/c4_development_attribution_events.csv",
            float_precision="round_trip",
        )

    @staticmethod
    def _verify(events: pd.DataFrame) -> None:
        verifier.verify_attribution_events(
            events,
            set(evaluator.DEVELOPMENT_SEEDS),
            set(evaluator.DEVELOPMENT_SCENARIOS),
        )

    def test_saved_events_accept_only_historical_unavailable_witness_nan_triplets(self):
        events = self._saved_development_events()
        filtered = ["filtered_d_sm", "filtered_d_sa", "filtered_d_ma"]
        any_missing = events[filtered].isna().any(axis=1)
        jointly_missing = events[filtered].isna().all(axis=1)

        self.assertEqual(int(jointly_missing.sum()), 13)
        self.assertTrue(any_missing.equals(jointly_missing))
        self.assertFalse(events.loc[jointly_missing, filtered].notna().any(axis=1).any())
        self.assertTrue((events.loc[jointly_missing, "attribution_state"] == "AMBIGUOUS").all())
        self.assertTrue((events.loc[jointly_missing, "candidate_state"] == "AMBIGUOUS").all())
        self.assertTrue((events.loc[jointly_missing, "auxiliary_trusted_fraction"] == 0.0).all())
        self.assertTrue(
            events.loc[jointly_missing, "auxiliary_trust_reason"]
            .isin(verifier.AUXILIARY_UNAVAILABLE_REASONS)
            .all()
        )

        self._verify(events)

    def test_saved_ready_event_rejects_filtered_nan(self):
        events = self._saved_development_events()
        ready_index = events.index[
            (events["auxiliary_trust_reason"] == "READY")
            & (events["auxiliary_trusted_fraction"] > 0.0)
        ][0]
        events.loc[ready_index, "filtered_d_sm"] = np.nan

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "filtered event diagnostics are either fully present or jointly missing",
        ):
            self._verify(events)

    def test_saved_nan_triplet_rejects_ready_reason(self):
        events = self._saved_development_events()
        missing_index = events.index[
            events[["filtered_d_sm", "filtered_d_sa", "filtered_d_ma"]].isna().all(axis=1)
        ][0]
        events.loc[missing_index, "auxiliary_trust_reason"] = "READY"

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "explicit unavailable-auxiliary reason",
        ):
            self._verify(events)


class DevelopmentFreezeRegressionTests(unittest.TestCase):
    def test_repaired_verifier_preserves_historical_development_self_hash(self):
        project = evaluator.PROJECT
        calibration = json.loads(
            (project / "results/configs/c4_attribution_calibration.json").read_text(
                encoding="utf-8"
            )
        )
        _config, bindings = verifier.verify_development_config_contract(
            project / "results/configs/c4_development_config.json",
            calibration,
        )

        self.assertEqual(
            bindings["c4_verifier_script"],
            verifier.FROZEN_C4_DEVELOPMENT_VERIFIER_SHA256,
        )
        self.assertNotEqual(
            verifier.sha256(project / "scripts/verify_c4_results.py"),
            verifier.FROZEN_C4_DEVELOPMENT_VERIFIER_SHA256,
        )

    def test_frozen_calibration_keeps_historical_generator_hash_after_checker_edit(self):
        calibration = json.loads(
            (evaluator.PROJECT / "results/configs/c4_attribution_calibration.json").read_text(
                encoding="utf-8"
            )
        )
        v3_freeze = json.loads(
            (evaluator.PROJECT / "results/configs/c4_v3_frozen_hashes.json").read_text(
                encoding="utf-8"
            )
        )
        historical_hash = calibration["provenance"]["calibration_code"]["sha256"]

        self.assertEqual(
            historical_hash,
            verifier.FROZEN_C4_CALIBRATION_GENERATOR_SHA256,
        )
        self.assertNotEqual(
            historical_hash,
            verifier.sha256(evaluator.PROJECT / "scripts/calibrate_c4_attribution.py"),
        )
        verifier.verify_calibration_provenance(calibration, v3_freeze)

    def test_frozen_script_hashes_fail_when_stale_and_pass_after_refreeze(self):
        params = {
            "agreement_thresholds": (1.0, 2.0, 3.0),
            "disagreement_thresholds": (4.0, 5.0, 6.0),
            "ewma_alpha": 0.05,
            "enter_count": 3,
            "exit_count": 5,
        }

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            config_path = project / "results/configs/c4_development_config.json"
            config_path.parent.mkdir(parents=True)

            def fake_sha256(path: Path) -> str:
                path = Path(path)
                if path.resolve() == Path(evaluator.__file__).resolve():
                    return "live-evaluator"
                if path.name == "verify_c4_results.py":
                    return "live-verifier"
                return f"hash-{path.name}"

            required_hashes = {
                "c4_attribution_calibration_json": "hash-c4_attribution_calibration.json",
                "c4_calibration_script": "hash-calibrate_c4_attribution.py",
                "c4_go_no_go_criteria_json": "hash-c4_go_no_go_criteria.json",
                "c4_evaluator_script": "stale-evaluator",
                "c4_verifier_script": "live-verifier",
                "reliability_py": "hash-reliability.py",
                "v3_main_model": "hash-lstm_model_weights.pt",
                "v3_auxiliary_model": "hash-auxiliary_model_weights.pt",
                "v3_arbitration_calibration": "hash-v3_arbitration_calibration.json",
                "v3_arbitration_config": "hash-v3_arbitration_config.json",
                "v3_final_275_run_matrix": "hash-v3_final_11scenario_runs.csv",
            }
            config = {
                "status": "frozen_before_first_completed_c4_development_evidence",
                "development_design": {
                    "seeds": evaluator.DEVELOPMENT_SEEDS.copy(),
                    "scenarios": evaluator.DEVELOPMENT_SCENARIOS.copy(),
                    "controllers": evaluator.CONTROLLERS.copy(),
                    "expected_run_count": 105,
                },
                "predeclared_criteria": "results/configs/c4_go_no_go_criteria.json",
                "attributor": {
                    "agreement_thresholds": list(params["agreement_thresholds"]),
                    "disagreement_thresholds": list(params["disagreement_thresholds"]),
                    "ewma_alpha": params["ewma_alpha"],
                    "enter_count": params["enter_count"],
                    "exit_count": params["exit_count"],
                },
                "hashes": required_hashes,
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with mock.patch.object(evaluator, "PROJECT", project), mock.patch.object(
                evaluator, "sha256", side_effect=fake_sha256
            ), mock.patch.object(evaluator, "load_c4_attribution_parameters", return_value=params):
                with self.assertRaisesRegex(RuntimeError, "frozen hash mismatch for c4_evaluator_script"):
                    evaluator.load_and_validate_development_config()

                config["hashes"]["c4_evaluator_script"] = "live-evaluator"
                config["hashes"]["c4_verifier_script"] = "stale-verifier"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "frozen hash mismatch for c4_verifier_script"):
                    evaluator.load_and_validate_development_config()

                config["hashes"]["c4_verifier_script"] = "live-verifier"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                _, resolved_path, bindings = evaluator.load_and_validate_development_config()

            self.assertEqual(resolved_path, config_path)
            self.assertEqual(bindings["c4_evaluator_script"], "live-evaluator")
            self.assertEqual(bindings["c4_verifier_script"], "live-verifier")


class FinalPreflightRegressionTests(unittest.TestCase):
    def test_minimal_final_config_is_rejected_at_each_missing_contract_layer(self):
        calibration_path = evaluator.PROJECT / "results/configs/c4_attribution_calibration.json"
        fake_development = {
            "stage_gate_sha256": "gate-hash",
            "development_config_sha256": "dev-hash",
        }

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            evaluator, "validate_saved_development_gate", return_value=fake_development
        ):
            path = Path(tmp) / "c4_frozen_config.json"

            config = {
                "final_holdout_seeds": FINAL_SEEDS.copy(),
                "scenarios": evaluator.FINAL_SCENARIOS.copy(),
            }
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "bind the exact saved GO stage gate and development config"):
                evaluator.parse_final_plan(path, calibration_path)

            config["hashes"] = {
                "c4_development_stage_gate": "gate-hash",
                "c4_development_config": "dev-hash",
            }
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "bind the exact supplied C4 attribution calibration"):
                evaluator.parse_final_plan(path, calibration_path)

            config["hashes"]["c4_attribution_calibration"] = evaluator.sha256(calibration_path)
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly one complete frozen attributor block"):
                evaluator.parse_final_plan(path, calibration_path)

            config["attributor"] = _exact_final_attributor_block()
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reference a pre-use seed provenance artifact"):
                evaluator.parse_final_plan(path, calibration_path)

    def test_complete_bounded_final_config_passes_preflight_without_running_matrix(self):
        calibration_path = evaluator.PROJECT / "results/configs/c4_attribution_calibration.json"
        fake_development = {
            "stage_gate_sha256": "gate-hash",
            "development_config_sha256": "dev-hash",
        }
        config = _complete_final_config()

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            evaluator, "validate_saved_development_gate", return_value=fake_development
        ):
            path = Path(tmp) / "c4_frozen_config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            seeds, loaded = evaluator.parse_final_plan(path, calibration_path)

        self.assertEqual(seeds, FINAL_SEEDS)
        self.assertEqual(loaded["scenarios"], evaluator.FINAL_SCENARIOS)

    def test_strict_final_attributor_rejects_missing_and_mismatched_block(self):
        calibration_path = evaluator.PROJECT / "results/configs/c4_attribution_calibration.json"
        calibration_hash = evaluator.sha256(calibration_path)
        base = {"hashes": {"c4_attribution_calibration": calibration_hash}}

        with self.assertRaisesRegex(RuntimeError, "exactly one complete frozen attributor block"):
            evaluator._validate_final_attributor_contract(base, calibration_path)

        mismatched = {
            **base,
            "attributor": _exact_final_attributor_block(),
        }
        mismatched["attributor"]["agreement_thresholds"][0] += 1.0
        with self.assertRaisesRegex(RuntimeError, "differ from the supplied calibration"):
            evaluator._validate_final_attributor_contract(mismatched, calibration_path)

    def test_current_candidate_audit_requires_exact_reference_hash_and_seed_set(self):
        provenance_path = evaluator.PROJECT / "results/configs/c4_candidate_seed_audit.json"
        provenance_hash = evaluator.sha256(provenance_path)

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "c4_frozen_config.json"
            config = {
                "candidate_seed_audit": "results/configs/c4_candidate_seed_audit.json",
                "hashes": {"c4_candidate_seed_audit": provenance_hash},
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            self.assertEqual(
                evaluator._validate_preuse_seed_provenance(config_path, config, FINAL_SEEDS),
                provenance_path.resolve(),
            )

            wrong_hash = json.loads(json.dumps(config))
            wrong_hash["hashes"]["c4_candidate_seed_audit"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "bind the exact pre-use seed provenance artifact"):
                evaluator._validate_preuse_seed_provenance(config_path, wrong_hash, FINAL_SEEDS)

            wrong_path = json.loads(json.dumps(config))
            wrong_path["candidate_seed_audit"] = "results/configs/does_not_exist.json"
            with self.assertRaisesRegex(RuntimeError, "seed provenance artifact does not exist"):
                evaluator._validate_preuse_seed_provenance(config_path, wrong_path, FINAL_SEEDS)

            wrong_seeds = [39026, 39027, 39028, 39029, 39031]
            with self.assertRaisesRegex(RuntimeError, "does not exactly cover the selected final seed set"):
                evaluator._validate_preuse_seed_provenance(config_path, config, wrong_seeds)


if __name__ == "__main__":
    unittest.main()
