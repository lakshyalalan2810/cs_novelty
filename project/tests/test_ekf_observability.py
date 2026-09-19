"""Regression tests for the analysis-only EKF observability tooling."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT / "scripts" / "analyze_ekf_observability.py"
SPEC = importlib.util.spec_from_file_location("analyze_ekf_observability", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def test_continuous_jacobian_matches_finite_difference() -> None:
    params = analysis.DCMotorParams()
    for state in (
        np.array([0.0, 0.0, 0.03]),
        np.array([1.5, 20.0, 0.06]),
        np.array([4.5, 55.0, 0.09]),
    ):
        analytic = analysis.continuous_jacobian(state, 6.0, params)
        numerical = analysis.finite_difference_jacobian(
            lambda candidate: analysis.augmented_dynamics(candidate, 6.0, params),
            state,
        )
        np.testing.assert_allclose(analytic, numerical, atol=1e-7, rtol=1e-7)


def test_rk4_jacobian_matches_finite_difference() -> None:
    params = analysis.DCMotorParams()
    state = np.array([2.0, 35.0, 0.03])
    analytic = analysis.rk4_state_jacobian(state, 8.0, params, 0.01)
    numerical = analysis.finite_difference_jacobian(
        lambda candidate: analysis.rk4_augmented_step(candidate, 8.0, params, 0.01),
        state,
    )
    np.testing.assert_allclose(analytic, numerical, atol=1e-7, rtol=1e-7)


def test_known_observability_rank_cases() -> None:
    params = analysis.DCMotorParams()
    state = np.array([1.0, 25.0, 0.03])
    nominal = analysis.continuous_observability_matrix(state, 6.0, params)
    no_back_emf = analysis.continuous_observability_matrix(
        state,
        6.0,
        replace(params, back_emf_constant=0.0),
    )

    assert np.linalg.matrix_rank(nominal) == 3
    assert np.linalg.matrix_rank(no_back_emf) < 3


def test_proposed_measurement_contract_has_no_speed_truth_or_scenario_leakage() -> None:
    contract = analysis.PROPOSED_EKF_CONTRACT
    assert contract["state"] == ["current", "omega", "load_torque"]
    assert contract["known_input"] == ["voltage"]
    assert contract["measurement"] == ["current"]

    online_channels = set(contract["known_input"] + contract["measurement"])
    forbidden = set(contract["forbidden_online_inputs"])
    assert online_channels == {"voltage", "current"}
    assert online_channels.isdisjoint(forbidden)
    assert not any("speed" in channel for channel in online_channels)
    assert "y_true" not in online_channels
    assert "scenario" not in online_channels


def test_smooth_friction_derivative_matches_finite_difference() -> None:
    params = analysis.DCMotorParams()
    epsilon = 1e-7
    for speed in (-4.0, -0.1, 0.0, 0.1, 40.0):
        numerical = (
            analysis.smooth_friction(speed + epsilon, params)
            - analysis.smooth_friction(speed - epsilon, params)
        ) / (2.0 * epsilon)
        analytic = analysis.smooth_friction_derivative(speed, params)
        np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=1e-6)
