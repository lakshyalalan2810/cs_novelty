import sys
import typing
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auxiliary_sensor_model import build_auxiliary_sequences, fit_auxiliary_normalization
from data_utils import fit_normalization, make_random_step_signal, split_runs
from motor_model import DCMotorParams, simulate_motor
from mpc import PIConfig, PIController
from reliability import guarded_predict_v2


class MotorSimulationRegressionTests(unittest.TestCase):
    def test_invalid_motor_parameters_fail_immediately(self):
        with self.assertRaisesRegex(ValueError, "friction_smoothing_speed"):
            DCMotorParams(friction_smoothing_speed=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            DCMotorParams(inertia=np.nan)

    def test_endpoint_is_present_once_for_exact_multiple(self):
        duration = 0.033
        timestep = 0.011
        time, states = simulate_motor(
            lambda _time: 0.0,
            simulation_time=duration,
            timestep=timestep,
        )

        self.assertEqual(len(time), len(states))
        self.assertEqual(time[-1], duration)
        self.assertTrue(np.all(np.diff(time) > 0.0))
        self.assertEqual(np.count_nonzero(time == duration), 1)

    def test_endpoint_is_present_once_for_partial_final_step(self):
        duration = 0.035
        timestep = 0.011
        time, _ = simulate_motor(
            lambda _time: 0.0,
            simulation_time=duration,
            timestep=timestep,
        )

        np.testing.assert_allclose(time, [0.0, 0.011, 0.022, 0.033, duration])
        self.assertTrue(np.all(np.diff(time) > 0.0))


class AuxiliarySequenceRegressionTests(unittest.TestCase):
    def test_valid_sequences_remain_aligned(self):
        voltage = np.arange(6, dtype=float)
        current = voltage + 10.0
        target = voltage + 100.0

        X, y = build_auxiliary_sequences(voltage, current, target, window_length=3)

        self.assertEqual(X.shape, (3, 3, 2))
        self.assertEqual(y.shape, (3, 1))
        np.testing.assert_allclose(X[0, :, 0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(X[-1, :, 0], [2.0, 3.0, 4.0])
        np.testing.assert_allclose(y[:, 0], [103.0, 104.0, 105.0])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            build_auxiliary_sequences(
                np.arange(6), np.arange(5), np.arange(6), window_length=3
            )

    def test_rejects_non_1d_inputs(self):
        with self.assertRaisesRegex(ValueError, "must be 1-D"):
            build_auxiliary_sequences(
                np.zeros((2, 3)), np.arange(6), np.arange(6), window_length=3
            )

    def test_rejects_invalid_window(self):
        values = np.arange(6)
        for window_length in (0, -1, 2.5, 6, 7):
            with self.subTest(window_length=window_length):
                with self.assertRaises(ValueError):
                    build_auxiliary_sequences(
                        values, values, values, window_length=window_length
                    )

    def test_auxiliary_normalization_rejects_invalid_data(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            fit_auxiliary_normalization(
                np.empty((0, 20, 2), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "same sample count"):
            fit_auxiliary_normalization(
                np.zeros((2, 20, 2), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "3-D"):
            fit_auxiliary_normalization(
                np.zeros((20, 2), dtype=np.float32),
                np.zeros((20, 1), dtype=np.float32),
            )
        bad_inputs = np.zeros((2, 20, 2), dtype=np.float32)
        bad_inputs[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_auxiliary_normalization(
                bad_inputs, np.zeros((2, 1), dtype=np.float32)
            )


class PIControllerRegressionTests(unittest.TestCase):
    def test_nonfinite_inputs_hold_output_and_preserve_state(self):
        controller = PIController(PIConfig())
        valid_voltage = controller.compute_control(10.0, 0.0, 0.1)
        integral_before = controller.integral
        voltage_before = controller.previous_voltage

        invalid_cases = [
            (10.0, 0.0, np.nan),
            (10.0, 0.0, np.inf),
            (np.nan, 0.0, 0.1),
            (10.0, np.inf, 0.1),
            (10.0, 0.0, 0.0),
            (10.0, 0.0, -0.1),
        ]
        for reference, speed, timestep in invalid_cases:
            with self.subTest(reference=reference, speed=speed, timestep=timestep):
                held = controller.compute_control(reference, speed, timestep)
                self.assertEqual(held, voltage_before)
                self.assertEqual(controller.previous_voltage, voltage_before)
                self.assertEqual(controller.integral, integral_before)

        self.assertEqual(valid_voltage, voltage_before)
        resumed = controller.compute_control(10.0, 0.0, 0.1)
        self.assertTrue(np.isfinite(resumed))
        self.assertTrue(np.isfinite(controller.integral))


class DataUtilsRegressionTests(unittest.TestCase):
    def test_split_runs_rejects_out_of_range_fractions(self):
        trajectories = [{"run_id": np.array([index])} for index in range(5)]
        rng = np.random.default_rng(123)
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            split_runs(trajectories, rng, fractions=(1.2, -0.1, -0.1))

    def test_random_step_signal_rejects_invalid_step_duration(self):
        rng = np.random.default_rng(123)
        with self.assertRaisesRegex(ValueError, "step_duration"):
            make_random_step_signal(rng, duration=1.0, step_duration=0.0, levels=[0.0, 1.0])

    def test_fit_normalization_rejects_empty_or_malformed_data(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            fit_normalization(
                np.empty((0, 20, 2), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "same sample count"):
            fit_normalization(
                np.zeros((2, 20, 2), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "3-D"):
            fit_normalization(
                np.zeros((20, 2), dtype=np.float32),
                np.zeros((20, 1), dtype=np.float32),
            )
        bad_inputs = np.zeros((2, 20, 2), dtype=np.float32)
        bad_inputs[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_normalization(bad_inputs, np.zeros((2, 1), dtype=np.float32))

    def test_valid_data_utils_behavior_is_preserved(self):
        rng = np.random.default_rng(123)
        signal = make_random_step_signal(rng, duration=1.0, step_duration=0.25, levels=[0.0, 1.0])
        self.assertTrue(np.isfinite(signal(0.5)))

        trajectories = [{"run_id": np.array([index])} for index in range(10)]
        splits = split_runs(trajectories, np.random.default_rng(123))
        self.assertEqual(
            (len(splits["train"]), len(splits["validation"]), len(splits["test"])),
            (6, 2, 2),
        )

        inputs = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
        targets = np.arange(3, dtype=np.float32)[:, None]
        stats = fit_normalization(inputs, targets)
        self.assertEqual(stats["input_mean"].shape, (2,))
        self.assertEqual(stats["target_mean"].shape, (1,))


class PublicAnnotationRegressionTests(unittest.TestCase):
    def test_guarded_predict_v2_type_hints_are_resolvable(self):
        hints = typing.get_type_hints(guarded_predict_v2)
        self.assertEqual(hints["aux_model"].__name__, "AuxiliarySpeedEstimator")


if __name__ == "__main__":
    unittest.main()
