"""Non-evaluation regression tests for the preregistered EKF closed-loop runner."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import evaluate_ekf_closed_loop as ekf_eval  # noqa: E402
from motor_model import DCMotorParams  # noqa: E402
from reliability import SensorReliabilityMonitor  # noqa: E402


class EKFClosedLoopContractTests(unittest.TestCase):
    def test_development_design_is_declared_before_any_run(self) -> None:
        self.assertEqual(ekf_eval.CONTROLLER, "E_EKF_virtual_MPC")
        self.assertEqual(
            ekf_eval.DEVELOPMENT_SEEDS,
            (19026, 19027, 19028, 19029, 19030),
        )
        self.assertEqual(
            ekf_eval.PRIMARY_SCENARIOS,
            (
                "sensor_bias_5",
                "sensor_dropout",
                "load_disturbance",
                "combined_fault_load",
                "parameter_variation",
            ),
        )
        self.assertEqual(
            ekf_eval.BASELINE_CONTROLLERS,
            ("B_plain_MPC", "C1_sensor_MPC", "C3_arbitration_MPC"),
        )
        self.assertTrue(set(ekf_eval.DEVELOPMENT_SEEDS).isdisjoint(ekf_eval.FORBIDDEN_SEEDS))
        self.assertEqual(len(ekf_eval.expected_matrix_keys()), 100)

    def test_actual_pre_run_plan_binds_current_evaluator_and_is_hashable(self) -> None:
        lock = ekf_eval.verify_closed_loop_plan()
        self.assertEqual(
            lock["closed_loop_plan_sha256"],
            ekf_eval.sha256(ekf_eval.CLOSED_LOOP_PLAN_PATH),
        )
        self.assertEqual(
            lock["closed_loop_evaluator_sha256"],
            ekf_eval.sha256(Path(ekf_eval.__file__).resolve()),
        )

    def test_runtime_accounting_separates_total_observer_reliability_from_slsqp(self) -> None:
        columns = ekf_eval._runtime_columns([1.0, 2.0, 3.0, 4.0], [10.0, 20.0])
        self.assertEqual(columns["total_reliability_observer_overhead_sample_count"], 4)
        self.assertEqual(columns["slsqp_solve_count"], 2)
        self.assertAlmostEqual(columns["mean_total_reliability_observer_overhead_ms"], 2.5)
        self.assertAlmostEqual(columns["median_total_reliability_observer_overhead_ms"], 2.5)
        self.assertAlmostEqual(columns["max_total_reliability_observer_overhead_ms"], 4.0)
        self.assertAlmostEqual(columns["mean_slsqp_solve_ms"], 15.0)
        self.assertAlmostEqual(columns["median_slsqp_solve_ms"], 15.0)
        self.assertAlmostEqual(columns["max_slsqp_solve_ms"], 20.0)
        self.assertGreater(columns["p99_slsqp_solve_ms"], columns["p95_slsqp_solve_ms"])

    def test_missing_frozen_config_is_a_hard_pre_evaluation_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "ekf_frozen_config.json"
            with self.assertRaisesRegex(RuntimeError, "closed-loop EKF evaluation is forbidden"):
                ekf_eval.load_frozen_ekf_config(missing)

    def test_frozen_config_parser_requires_27_candidates_and_uses_nominal_parameters(self) -> None:
        record = {
            "status": "frozen_before_closed_loop",
            "candidate_table": [{"candidate_id": index} for index in range(27)],
            "selected_candidate": {
                "Q_diag": [1e-7, 2e-5, 3e-8],
                "R": 4e-6,
            },
            "initialization": {"P0_diag": [1.0, 100.0, 0.01]},
            "closed_loop_development_seeds": list(ekf_eval.DEVELOPMENT_SEEDS),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ekf_frozen_config.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            loaded, config = ekf_eval.load_frozen_ekf_config(path)

        self.assertEqual(len(loaded["candidate_table"]), 27)
        np.testing.assert_array_equal(config.Q, np.diag([1e-7, 2e-5, 3e-8]))
        np.testing.assert_array_equal(config.P0, np.diag([1.0, 100.0, 0.01]))
        self.assertEqual(config.R, 4e-6)
        self.assertEqual(config.dt, 0.01)
        self.assertEqual(config.params, DCMotorParams())

    def test_frozen_config_rejects_incomplete_candidate_audit(self) -> None:
        record = {
            "status": "frozen",
            "candidate_table": [{"candidate_id": index} for index in range(26)],
            "selected_candidate": {"Q_diag": [1e-7, 1e-5, 1e-8], "R": 1e-5},
            "initialization": {"P0_diag": [1.0, 100.0, 0.01]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ekf_frozen_config.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "27"):
                ekf_eval.load_frozen_ekf_config(path)

    def test_detection_seam_has_no_ekf_argument_and_matches_direct_v3_monitor_update(self) -> None:
        parameters = dict(
            residual_gate=0.5,
            center=0.0,
            allowance=0.1,
            threshold=1.0,
            enter_count=3,
            exit_count=5,
            aux_recovery_gate=0.75,
        )
        wrapped = SensorReliabilityMonitor(**parameters)
        direct = SensorReliabilityMonitor(**parameters)
        signature = inspect.signature(ekf_eval.update_v3_monitor)
        self.assertEqual(
            list(signature.parameters),
            ["monitor", "measured_speed", "main_speed", "auxiliary_speed"],
        )

        samples = [
            (10.0, 10.1, 10.2),
            (10.0, 8.0, 10.1),
            (10.0, 8.1, 10.2),
            (10.0, 8.0, 10.1),
            (10.0, 10.0, 10.0),
        ]
        for measured, main, auxiliary in samples:
            with self.subTest(measured=measured, main=main, auxiliary=auxiliary):
                expected = direct.update(
                    measured - main,
                    aux_residual=measured - auxiliary,
                )
                actual = ekf_eval.update_v3_monitor(wrapped, measured, main, auxiliary)
                self.assertEqual(actual, expected)

    def test_ekf_advances_continuously_before_any_substitution_decision(self) -> None:
        class FakeObserver:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def initialize(self, current: float) -> np.ndarray:
                self.calls.append(("initialize", current))
                return np.array([current, 0.0, 0.0])

            def step(self, voltage: float, current: float) -> tuple[np.ndarray, object]:
                self.calls.append(("step", voltage, current))
                return np.array([current, 1.0, 0.0]), object()

        observer = FakeObserver()
        states = []
        for index, (voltage, current) in enumerate(
            [(0.0, 0.0), (1.0, 0.1), (2.0, 0.2), (3.0, 0.3)]
        ):
            state, _ = ekf_eval.advance_ekf(observer, index, voltage, current)  # type: ignore[arg-type]
            states.append(state)

        self.assertEqual(
            observer.calls,
            [
                ("initialize", 0.0),
                ("step", 1.0, 0.1),
                ("step", 2.0, 0.2),
                ("step", 3.0, 0.3),
            ],
        )
        self.assertEqual(len(states), 4)
        self.assertNotIn("substitute", inspect.signature(ekf_eval.advance_ekf).parameters)

    def test_monitor_factory_keeps_c3_auxiliary_recovery_gate(self) -> None:
        sensor_cfg = {
            "instant_threshold": 0.937,
            "center": -0.009,
            "allowance": 0.140,
            "threshold": 1.57,
            "enter_count": 3,
            "exit_count": 5,
        }
        v3_cfg = {"arbitrator": {"aux_recovery_gate": 9.047}}
        monitor = ekf_eval.make_v3_monitor(sensor_cfg, v3_cfg)
        self.assertEqual(monitor.aux_recovery_gate, 9.047)
        self.assertEqual(monitor.enter_count, 3)
        self.assertEqual(monitor.exit_count, 5)

    def test_primary_speed_corruptions_match_frozen_v3_semantics(self) -> None:
        rng = np.random.default_rng(123)
        true_speed = 30.0
        base_noise = 0.25
        full_scale = ekf_eval.FULL_SCALE

        self.assertEqual(
            ekf_eval.speed_measurement("sensor_bias_5", 2.5, true_speed, base_noise, rng),
            true_speed + base_noise + 0.05 * full_scale,
        )
        self.assertEqual(
            ekf_eval.speed_measurement("sensor_dropout", 2.5, true_speed, base_noise, rng),
            0.0,
        )
        self.assertEqual(
            ekf_eval.speed_measurement("combined_fault_load", 3.5, true_speed, base_noise, rng),
            true_speed + base_noise + 0.05 * full_scale,
        )
        self.assertEqual(
            ekf_eval.speed_measurement("load_disturbance", 3.5, true_speed, base_noise, rng),
            true_speed + base_noise,
        )
        self.assertEqual(
            ekf_eval.speed_measurement("parameter_variation", 3.5, true_speed, base_noise, rng),
            true_speed + base_noise,
        )

    def test_current_frozen_c4_rows_pass_exact_provenance_and_cover_four_primary_scenarios(self) -> None:
        rows, report = ekf_eval.inspect_reusable_c4_baselines()
        self.assertTrue(report["verified"], report)
        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(len(rows), 40)
        self.assertEqual(
            set(rows["scenario"]),
            {
                "sensor_dropout",
                "load_disturbance",
                "combined_fault_load",
                "parameter_variation",
            },
        )
        self.assertNotIn("sensor_bias_5", set(rows["scenario"]))
        self.assertEqual(set(rows["seed"].astype(int)), set(ekf_eval.DEVELOPMENT_SEEDS))
        self.assertEqual(set(rows["controller"]), set(ekf_eval.REUSABLE_BASELINE_CONTROLLERS))

    def test_baseline_collector_reuses_only_proven_b_c3_and_reruns_c1_plus_missing_bias(self) -> None:
        calls: list[tuple[str, str, int, bool]] = []

        def stub(args: tuple[str, str, int, bool]) -> dict[str, object]:
            calls.append(args)
            controller, scenario, seed, _ = args
            return {
                "controller": controller,
                "scenario": scenario,
                "seed": seed,
                "overall_rmse": 1.0,
                "overall_mae": 1.0,
                "fault_window_rmse": 1.0,
                "sub_fraction": 0.0,
                "sub_samples": 0,
            }

        rows, report = ekf_eval.collect_baseline_rows(rerun_fn=stub)
        self.assertTrue(report["verified"], report)
        self.assertEqual(len(rows), 75)
        self.assertEqual(len(calls), 35)
        c1_calls = [call for call in calls if call[0] == "C1_sensor_MPC"]
        other_calls = [call for call in calls if call[0] != "C1_sensor_MPC"]
        self.assertEqual(len(c1_calls), 25)
        self.assertEqual(
            {scenario for _, scenario, _, _ in c1_calls},
            set(ekf_eval.PRIMARY_SCENARIOS),
        )
        self.assertEqual(len(other_calls), 10)
        self.assertTrue(all(scenario == "sensor_bias_5" for _, scenario, _, _ in other_calls))
        self.assertEqual(
            {controller for controller, _, _, _ in other_calls},
            set(ekf_eval.REUSABLE_BASELINE_CONTROLLERS),
        )
        self.assertEqual({seed for _, _, seed, _ in calls}, set(ekf_eval.DEVELOPMENT_SEEDS))

    def test_role_gate_encodes_preregistered_pairwise_thresholds(self) -> None:
        rows: list[dict[str, object]] = []
        for scenario in ekf_eval.PRIMARY_SCENARIOS:
            for seed in ekf_eval.DEVELOPMENT_SEEDS:
                plain_rmse = 1.0
                c3_rmse = 1.0
                ekf_rmse = 1.0
                if scenario == "load_disturbance":
                    plain_rmse, c3_rmse, ekf_rmse = 1.0, 2.0, 1.7
                elif scenario == "combined_fault_load":
                    c3_rmse, ekf_rmse = 2.0, 1.7
                rows.extend(
                    [
                        {
                            "controller": "B_plain_MPC",
                            "scenario": scenario,
                            "seed": seed,
                            "fault_window_rmse": plain_rmse,
                            "sub_fraction": 0.0,
                            "sub_samples": 0,
                        },
                        {
                            "controller": "C3_arbitration_MPC",
                            "scenario": scenario,
                            "seed": seed,
                            "fault_window_rmse": c3_rmse,
                            "sub_fraction": 0.10 if scenario == "load_disturbance" else 0.0,
                            "sub_samples": 60 if scenario == "load_disturbance" else 0,
                        },
                        {
                            "controller": ekf_eval.CONTROLLER,
                            "scenario": scenario,
                            "seed": seed,
                            "fault_window_rmse": ekf_rmse,
                            "run_complete": True,
                            "ekf_finite_sample_rate": 1.0,
                            "ekf_numerical_failures": 0,
                            "voltage_violations": 0,
                            "rate_violations": 0,
                            "optimizer_failures": 0,
                            "nonfinite_events": 0,
                            "main_prediction_failures": 0,
                            "aux_prediction_failures": 0,
                            "false_substitution_count": 30 if scenario == "load_disturbance" else 0,
                            "false_substitution_fraction": 0.05 if scenario == "load_disturbance" else 0.0,
                            "sub_fraction": 0.05 if scenario == "load_disturbance" else 0.0,
                            "sub_samples": 30 if scenario == "load_disturbance" else 0,
                        },
                    ]
                )
        gate = ekf_eval.evaluate_role_gate(pd.DataFrame(rows), calibration_locked=True)
        self.assertEqual(gate["decision"], "POTENTIAL_SUCCESSOR_FOLLOW_UP")
        self.assertEqual(gate["combined_fault_load"]["paired_seed_passes"], 5)
        self.assertEqual(gate["load_disturbance"]["paired_tracking_seed_passes"], 5)
        self.assertTrue(gate["all_successor_criteria_pass"])


if __name__ == "__main__":
    unittest.main()
