import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import evaluate_v2
from motor_model import DCMotorParams, simulate_motor


class V2ScenarioRegressionTests(unittest.TestCase):
    def test_sensor_bias_scenarios_are_percent_full_scale(self):
        measured = np.zeros(500, dtype=float)

        bias_5 = evaluate_v2.inject_fault(
            measured, evaluate_v2.SCENARIOS["sensor_bias_5"]
        )
        bias_15 = evaluate_v2.inject_fault(
            measured, evaluate_v2.SCENARIOS["sensor_bias_15"]
        )

        self.assertAlmostEqual(bias_5[200], 0.05 * evaluate_v2.FULL_SCALE)
        self.assertAlmostEqual(bias_15[200], 0.15 * evaluate_v2.FULL_SCALE)
        self.assertEqual(bias_5[199], 0.0)
        self.assertEqual(bias_5[400], 0.0)

    def test_plant_events_preserve_pre_event_state_and_change_post_event_state(self):
        dt = 0.01
        params = DCMotorParams()
        time, states = simulate_motor(
            lambda _time: 6.0,
            lambda _time: 0.03,
            simulation_time=4.0,
            timestep=dt,
            params=params,
        )
        current, speed = states.T
        voltage = np.full(len(time), 6.0)
        load = np.full(len(time), 0.03)
        measured = speed.copy()

        for scenario_name in ("load_disturbance", "parameter_variation"):
            with self.subTest(scenario=scenario_name):
                scenario = evaluate_v2.SCENARIOS[scenario_name]
                result = evaluate_v2.apply_scenario_to_trajectory(
                    voltage,
                    current,
                    speed,
                    measured,
                    load,
                    params,
                    scenario,
                    dt,
                )
                start = scenario["start"]
                np.testing.assert_array_equal(result["current"][: start + 1], current[: start + 1])
                np.testing.assert_array_equal(result["y_true"][: start + 1], speed[: start + 1])
                self.assertGreater(
                    np.max(np.abs(result["y_true"][start + 1 :] - speed[start + 1 :])),
                    1e-4,
                )

    def test_combined_scenario_contains_both_load_response_and_five_percent_bias(self):
        dt = 0.01
        params = DCMotorParams()
        time, states = simulate_motor(
            lambda _time: 6.0,
            lambda _time: 0.03,
            simulation_time=4.0,
            timestep=dt,
            params=params,
        )
        current, speed = states.T
        voltage = np.full(len(time), 6.0)
        load = np.full(len(time), 0.03)
        scenario = evaluate_v2.SCENARIOS["combined_fault_load"]

        result = evaluate_v2.apply_scenario_to_trajectory(
            voltage,
            current,
            speed,
            speed,
            load,
            params,
            scenario,
            dt,
        )

        start = scenario["start"]
        self.assertAlmostEqual(
            result["y_measured"][start] - result["y_true"][start],
            0.05 * evaluate_v2.FULL_SCALE,
            places=5,
        )
        self.assertGreater(
            np.max(np.abs(result["y_true"][start + 1 :] - speed[start + 1 :])),
            1e-4,
        )


class V2MetricRegressionTests(unittest.TestCase):
    def test_detection_recovery_metrics_follow_latched_sensor_alarm(self):
        window_length = 2
        prediction = np.zeros(8, dtype=float)
        result = {
            "prediction": prediction,
            "substituted": np.array(
                [False, False, True, False, False, False, False, False],
                dtype=bool,
            ),
            "sensor_alarm": np.array(
                [False, False, False, False, True, True, True, False],
                dtype=bool,
            ),
        }
        scenario = {"type": "bias", "start": 4, "end": 8}
        y_true = np.zeros(10, dtype=float)

        metrics = evaluate_v2.compute_metrics(
            y_true, y_true, result, scenario, window_length
        )

        self.assertEqual(metrics["detection_latency"], 2)
        self.assertEqual(metrics["false_recoveries"], 0)
        self.assertEqual(metrics["recovery_latency"], 1)
        self.assertEqual(metrics["sub_duration"], 1)

    def test_recovery_latency_requires_a_fault_window_latch(self):
        window_length = 2
        prediction = np.zeros(8, dtype=float)
        scenario = {"type": "bias", "start": 4, "end": 8}
        y_true = np.zeros(10, dtype=float)

        never_latched = {
            "prediction": prediction,
            "substituted": np.zeros(8, dtype=bool),
            "sensor_alarm": np.zeros(8, dtype=bool),
        }
        metrics = evaluate_v2.compute_metrics(
            y_true, y_true, never_latched, scenario, window_length
        )
        self.assertTrue(np.isnan(metrics["recovery_latency"]))

        recovered_during_fault = {
            "prediction": prediction,
            "substituted": np.zeros(8, dtype=bool),
            "sensor_alarm": np.array(
                [False, False, True, False, False, False, False, False],
                dtype=bool,
            ),
        }
        metrics = evaluate_v2.compute_metrics(
            y_true, y_true, recovered_during_fault, scenario, window_length
        )
        self.assertEqual(metrics["recovery_latency"], 0)


class V2ArtifactRegressionTests(unittest.TestCase):
    def test_offline_and_closed_loop_quality_artifacts_are_distinct(self):
        offline_source = (PROJECT / "scripts/evaluate_v2.py").read_text(encoding="utf-8")
        closed_loop_source = (PROJECT / "scripts/evaluate_v2_closed_loop.py").read_text(
            encoding="utf-8"
        )
        notebook_source = (PROJECT / "scripts/build_04d_notebook.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("v2_offline_aux_virtual_sensor_quality.csv", offline_source)
        self.assertIn("v2_closed_loop_aux_virtual_sensor_quality.csv", closed_loop_source)
        self.assertIn(
            "results/metrics/v2_closed_loop_aux_virtual_sensor_quality.csv",
            notebook_source,
        )
        self.assertNotIn(
            'results/metrics/aux_virtual_sensor_quality.csv', notebook_source
        )


if __name__ == "__main__":
    unittest.main()
