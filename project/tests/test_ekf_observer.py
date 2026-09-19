"""Focused numerical and causality tests for the preregistered EKF observer."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ekf_observer import (  # noqa: E402
    AugmentedStateEKF,
    EKFConfig,
    EKFNumericalError,
    augmented_dynamics,
    rk4_state_jacobian,
    rk4_step,
)
from motor_model import DCMotorParams, dc_motor_dynamics  # noqa: E402


DT = 0.01


def _finite_difference_jacobian(
    state: np.ndarray,
    voltage: float,
    params: DCMotorParams,
) -> np.ndarray:
    jacobian = np.empty((3, 3), dtype=float)
    for index in range(3):
        step = 1e-6 * max(1.0, abs(float(state[index])))
        offset = np.zeros(3, dtype=float)
        offset[index] = step
        jacobian[:, index] = (
            rk4_step(state + offset, voltage, params, DT)
            - rk4_step(state - offset, voltage, params, DT)
        ) / (2.0 * step)
    return jacobian


def _plant_rk4_step(
    state: np.ndarray,
    voltage: float,
    load_torque: float,
    params: DCMotorParams,
) -> np.ndarray:
    def derivative(candidate: np.ndarray) -> np.ndarray:
        return dc_motor_dynamics(0.0, candidate, voltage, load_torque, params)

    k1 = derivative(state)
    k2 = derivative(state + DT * k1 / 2.0)
    k3 = derivative(state + DT * k2 / 2.0)
    k4 = derivative(state + DT * k3)
    return state + DT * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _config() -> EKFConfig:
    return EKFConfig(
        Q=np.diag([1e-7, 1e-5, 1e-8]),
        R=1e-5,
        P0=np.diag([1.0, 100.0, 0.01]),
        dt=DT,
    )


class EKFObserverTests(unittest.TestCase):
    def test_augmented_rk4_matches_repository_motor_and_is_deterministic(self) -> None:
        params = DCMotorParams()
        state = np.array([1.8, 24.0, 0.07], dtype=float)
        voltage = 7.5

        first = rk4_step(state, voltage, params, DT)
        second = rk4_step(state, voltage, params, DT)
        expected_motor = _plant_rk4_step(state[:2], voltage, state[2], params)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first[:2], expected_motor, rtol=0.0, atol=1e-14)
        self.assertEqual(first[2], state[2])

    def test_augmented_dynamics_load_state_is_deterministic_random_walk_mean(self) -> None:
        state = np.array([2.0, 30.0, 0.123], dtype=float)
        derivative = augmented_dynamics(state, 8.0, DCMotorParams())
        propagated = rk4_step(state, 8.0, DCMotorParams(), DT)

        self.assertEqual(derivative[2], 0.0)
        self.assertEqual(propagated[2], state[2])

    def test_analytic_rk4_jacobian_matches_centered_finite_difference(self) -> None:
        params = DCMotorParams()
        for state in (
            np.array([0.0, 0.0, 0.03]),
            np.array([2.0, 35.0, 0.06]),
            np.array([-0.8, -3.0, 0.0]),
        ):
            with self.subTest(state=state.tolist()):
                analytic = rk4_state_jacobian(state, 6.0, params, DT)
                numerical = _finite_difference_jacobian(state, 6.0, params)
                np.testing.assert_allclose(analytic, numerical, atol=1e-7, rtol=1e-7)

    def test_step_uses_joseph_covariance_update(self) -> None:
        config = _config()
        observer = AugmentedStateEKF(config)
        initial_state = observer.initialize(0.4)
        initial_covariance = observer.covariance
        voltage = 5.0
        measurement = 0.43

        transition = rk4_state_jacobian(initial_state, voltage, config.params, config.dt)
        predicted_covariance = (
            transition @ initial_covariance @ transition.T + config.Q
        )
        predicted_state = rk4_step(initial_state, voltage, config.params, config.dt)
        c_matrix = np.array([[1.0, 0.0, 0.0]], dtype=float)
        innovation_variance = float(
            (c_matrix @ predicted_covariance @ c_matrix.T).item() + config.R
        )
        gain = (predicted_covariance @ c_matrix.T)[:, 0] / innovation_variance
        correction = np.eye(3) - np.outer(gain, c_matrix[0])
        expected_covariance = (
            correction @ predicted_covariance @ correction.T
            + np.outer(gain, gain) * config.R
        )
        expected_covariance = 0.5 * (expected_covariance + expected_covariance.T)
        expected_state = predicted_state + gain * (measurement - predicted_state[0])

        state, diagnostics = observer.step(voltage, measurement)

        np.testing.assert_allclose(state, expected_state, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(
            observer.covariance, expected_covariance, atol=1e-14, rtol=1e-14
        )
        np.testing.assert_allclose(
            diagnostics.updated_covariance, expected_covariance, atol=1e-14, rtol=1e-14
        )

    def test_covariance_stays_symmetric_and_psd(self) -> None:
        observer = AugmentedStateEKF(_config())
        observer.initialize(0.0)
        plant = np.array([0.0, 0.0], dtype=float)
        params = DCMotorParams()

        for index in range(250):
            voltage = 8.0 if index < 150 else 4.0
            load = 0.03 if index < 100 else 0.08
            plant = _plant_rk4_step(plant, voltage, load, params)
            state, diagnostics = observer.step(voltage, plant[0])
            covariance = observer.covariance

            self.assertTrue(np.isfinite(state).all())
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-14, rtol=0.0)
            self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(covariance))), -1e-10)
            self.assertGreaterEqual(diagnostics.min_covariance_eigenvalue, -1e-10)

    def test_online_api_exposes_no_speed_or_truth_input(self) -> None:
        initialize_parameters = list(inspect.signature(AugmentedStateEKF.initialize).parameters)
        step_parameters = list(inspect.signature(AugmentedStateEKF.step).parameters)

        self.assertEqual(initialize_parameters, ["self", "first_current"])
        self.assertEqual(step_parameters, ["self", "voltage", "current_measurement"])
        forbidden = {"speed", "omega", "measured_speed", "y_true", "true_speed"}
        self.assertTrue(forbidden.isdisjoint(initialize_parameters))
        self.assertTrue(forbidden.isdisjoint(step_parameters))

    def test_nominal_operation_remains_finite(self) -> None:
        observer = AugmentedStateEKF(_config())
        observer.initialize(0.0)
        plant = np.array([0.0, 0.0], dtype=float)
        params = DCMotorParams()

        for index in range(500):
            voltage = 10.0 if index < 200 else (6.0 if index < 350 else 2.0)
            load = 0.02 if index < 300 else 0.10
            plant = _plant_rk4_step(plant, voltage, load, params)
            state, diagnostics = observer.step(voltage, plant[0])

            self.assertTrue(np.isfinite(state).all())
            self.assertTrue(np.isfinite(observer.covariance).all())
            self.assertTrue(np.isfinite(diagnostics.innovation))
            self.assertTrue(np.isfinite(diagnostics.innovation_variance))
            self.assertTrue(np.isfinite(diagnostics.kalman_gain).all())
        self.assertFalse(observer.failed)
        self.assertIsNone(observer.failure_reason)

    def test_initialization_uses_first_current_zero_speed_and_zero_load(self) -> None:
        observer = AugmentedStateEKF(_config())
        state = observer.initialize(1.234)

        np.testing.assert_array_equal(state, np.array([1.234, 0.0, 0.0]))
        np.testing.assert_array_equal(observer.covariance, observer.config.P0)

    def test_nonfinite_online_input_reports_failure_without_reset(self) -> None:
        observer = AugmentedStateEKF(_config())
        initial = observer.initialize(0.25)

        with self.assertRaisesRegex(EKFNumericalError, "input: current_measurement must be finite"):
            observer.step(5.0, np.nan)

        self.assertTrue(observer.failed)
        self.assertIn("current_measurement", observer.failure_reason or "")
        np.testing.assert_array_equal(observer.state, initial)


if __name__ == "__main__":
    unittest.main()
