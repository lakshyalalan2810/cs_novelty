"""Analysis-only observability study for a current-only augmented DC-motor observer.

This module does not implement an EKF controller.  It derives the exact
continuous Jacobian of the repository motor model, constructs the exact
Jacobian of the explicit RK4 map used by the frozen closed-loop evaluators,
and measures finite-horizon local observability over the saved healthy
training/validation operating population.

Online observer contract under study:

    x = [armature current i, angular speed omega, load torque T_L]
    u = applied voltage V
    y = armature current i

The saved auxiliary LSTM receives the same causal voltage/current information.
No speed measurement, truth signal, scenario label, or fault label is part of
this observer contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from motor_model import DCMotorParams  # noqa: E402


DATASET_PATH = PROJECT / "data" / "processed" / "dc_motor_lstm_dataset.npz"
SUMMARY_PATH = PROJECT / "results" / "analysis" / "ekf_observability_summary.json"
PLOT_PATH = PROJECT / "results" / "plots" / "ekf_observability_conditioning.png"

PRIMARY_HORIZON_SAMPLES = 20
SECONDARY_HORIZON_SAMPLES = 50
STARTUP_DURATION_S = 0.5

PROPOSED_EKF_CONTRACT = {
    "state": ["current", "omega", "load_torque"],
    "known_input": ["voltage"],
    "measurement": ["current"],
    "forbidden_online_inputs": [
        "measured_speed",
        "y_measured_speed",
        "y_true",
        "true_speed",
        "fault_label",
        "fault_flag",
        "scenario",
        "scenario_identity",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth_friction(speed: float, params: DCMotorParams) -> float:
    """Repository smooth Coulomb friction torque."""
    return float(
        params.coulomb_friction
        * np.tanh(speed / params.friction_smoothing_speed)
    )


def smooth_friction_derivative(speed: float, params: DCMotorParams) -> float:
    """Exact d/domega of Fc*tanh(omega/ws), evaluated stably."""
    tanh_value = np.tanh(speed / params.friction_smoothing_speed)
    sech_squared = 1.0 - tanh_value * tanh_value
    return float(
        params.coulomb_friction
        / params.friction_smoothing_speed
        * sech_squared
    )


def augmented_dynamics(
    state: np.ndarray,
    voltage: float,
    params: DCMotorParams,
) -> np.ndarray:
    """Continuous augmented dynamics for [i, omega, T_L]."""
    current, speed, load_torque = np.asarray(state, dtype=float)
    current_dot = (
        voltage
        - params.resistance * current
        - params.back_emf_constant * speed
    ) / params.inductance
    speed_dot = (
        params.torque_constant * current
        - params.viscous_friction * speed
        - load_torque
        - smooth_friction(speed, params)
    ) / params.inertia
    return np.array([current_dot, speed_dot, 0.0], dtype=float)


def continuous_jacobian(
    state: np.ndarray,
    voltage: float,
    params: DCMotorParams,
) -> np.ndarray:
    """Exact A_c = df/dx for the repository motor and load augmentation."""
    del voltage  # affine input does not affect df/dx
    _current, speed, _load_torque = np.asarray(state, dtype=float)
    friction_slope = smooth_friction_derivative(speed, params)
    return np.array(
        [
            [
                -params.resistance / params.inductance,
                -params.back_emf_constant / params.inductance,
                0.0,
            ],
            [
                params.torque_constant / params.inertia,
                -(params.viscous_friction + friction_slope) / params.inertia,
                -1.0 / params.inertia,
            ],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def measurement_jacobian() -> np.ndarray:
    """C = dh/dx for h(x)=i."""
    return np.array([[1.0, 0.0, 0.0]], dtype=float)


def finite_difference_jacobian(
    function,
    state: np.ndarray,
    *,
    relative_step: float = 1e-6,
) -> np.ndarray:
    """Centered finite-difference Jacobian for analysis/test validation."""
    state = np.asarray(state, dtype=float)
    base = np.asarray(function(state), dtype=float)
    jacobian = np.empty((base.size, state.size), dtype=float)
    for index in range(state.size):
        step = relative_step * max(1.0, abs(float(state[index])))
        offset = np.zeros_like(state)
        offset[index] = step
        jacobian[:, index] = (
            np.asarray(function(state + offset), dtype=float)
            - np.asarray(function(state - offset), dtype=float)
        ) / (2.0 * step)
    return jacobian


def rk4_augmented_step(
    state: np.ndarray,
    voltage: float,
    params: DCMotorParams,
    timestep: float,
) -> np.ndarray:
    """Explicit RK4 map matching the frozen closed-loop plant integration."""
    state = np.asarray(state, dtype=float)
    k1 = augmented_dynamics(state, voltage, params)
    k2 = augmented_dynamics(state + timestep * k1 / 2.0, voltage, params)
    k3 = augmented_dynamics(state + timestep * k2 / 2.0, voltage, params)
    k4 = augmented_dynamics(state + timestep * k3, voltage, params)
    return state + timestep * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def rk4_state_jacobian(
    state: np.ndarray,
    voltage: float,
    params: DCMotorParams,
    timestep: float,
) -> np.ndarray:
    """Exact Jacobian of the explicit RK4 state map via tangent propagation."""
    state = np.asarray(state, dtype=float)
    identity = np.eye(3, dtype=float)

    k1 = augmented_dynamics(state, voltage, params)
    dk1 = continuous_jacobian(state, voltage, params)

    x2 = state + timestep * k1 / 2.0
    dx2 = identity + timestep * dk1 / 2.0
    k2 = augmented_dynamics(x2, voltage, params)
    dk2 = continuous_jacobian(x2, voltage, params) @ dx2

    x3 = state + timestep * k2 / 2.0
    dx3 = identity + timestep * dk2 / 2.0
    k3 = augmented_dynamics(x3, voltage, params)
    dk3 = continuous_jacobian(x3, voltage, params) @ dx3

    x4 = state + timestep * k3
    dx4 = identity + timestep * dk3
    dk4 = continuous_jacobian(x4, voltage, params) @ dx4

    return identity + timestep * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4) / 6.0


def continuous_observability_matrix(
    state: np.ndarray,
    voltage: float,
    params: DCMotorParams,
) -> np.ndarray:
    """Local continuous observability matrix [C; CA; CA^2]."""
    a_matrix = continuous_jacobian(state, voltage, params)
    c_matrix = measurement_jacobian()
    return np.vstack((c_matrix, c_matrix @ a_matrix, c_matrix @ a_matrix @ a_matrix))


def finite_horizon_observability_matrix(
    transition_jacobians: Iterable[np.ndarray],
    horizon_samples: int,
) -> np.ndarray:
    """Build time-varying local O=[C; C F0; C F1 F0; ...]."""
    transitions = list(transition_jacobians)
    if horizon_samples < 3:
        raise ValueError("horizon_samples must be at least 3")
    if len(transitions) < horizon_samples - 1:
        raise ValueError("insufficient transition Jacobians for requested horizon")
    c_matrix = measurement_jacobian()
    phi = np.eye(3, dtype=float)
    rows = [c_matrix.copy()]
    for step in range(horizon_samples - 1):
        phi = np.asarray(transitions[step], dtype=float) @ phi
        rows.append(c_matrix @ phi)
    return np.vstack(rows)


def observability_metrics(
    matrix: np.ndarray,
    state_scales: np.ndarray | None = None,
    output_scale: float | None = None,
) -> dict[str, float | int | list[float]]:
    """Rank, singular values, condition number and state-column sensitivities."""
    matrix = np.asarray(matrix, dtype=float)
    if state_scales is not None:
        scales = np.asarray(state_scales, dtype=float)
        if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("state_scales must be three finite positive values")
        matrix = matrix @ np.diag(scales)
    if output_scale is not None:
        if not np.isfinite(output_scale) or output_scale <= 0:
            raise ValueError("output_scale must be finite and positive")
        matrix = matrix / float(output_scale)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    smallest = float(singular_values[-1])
    condition = float(singular_values[0] / smallest) if smallest > 0 else math.inf
    column_norms = np.linalg.norm(matrix, axis=0)
    gramian_eigenvalues = np.linalg.eigvalsh(matrix.T @ matrix)
    return {
        "rank": rank,
        "singular_values": [float(value) for value in singular_values],
        "smallest_singular_value": smallest,
        "condition_number": condition,
        "column_norms": [float(value) for value in column_norms],
        "gramian_eigenvalues": [float(value) for value in gramian_eigenvalues],
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {key: math.nan for key in ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max")}
    probabilities = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    names = ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"]
    result = np.quantile(finite, probabilities)
    return {name: float(value) for name, value in zip(names, result)}


def _distribution_summary(records: list[dict[str, float]]) -> dict:
    if not records:
        return {"count": 0}
    rank = np.asarray([record["rank"] for record in records], dtype=float)
    smin = np.asarray([record["smallest_singular_value"] for record in records], dtype=float)
    sigma_1 = np.asarray([record["singular_value_1"] for record in records], dtype=float)
    sigma_2 = np.asarray([record["singular_value_2"] for record in records], dtype=float)
    sigma_3 = np.asarray([record["singular_value_3"] for record in records], dtype=float)
    cond = np.asarray([record["condition_number"] for record in records], dtype=float)
    speed_sensitivity = np.asarray([record["speed_sensitivity"] for record in records], dtype=float)
    load_sensitivity = np.asarray([record["load_sensitivity"] for record in records], dtype=float)
    load_to_speed = np.divide(
        load_sensitivity,
        speed_sensitivity,
        out=np.full_like(load_sensitivity, np.nan),
        where=speed_sensitivity > 0,
    )
    return {
        "count": int(len(records)),
        "rank_counts": {str(int(value)): int(np.sum(rank == value)) for value in np.unique(rank)},
        "full_rank_fraction": float(np.mean(rank == 3)),
        "singular_values": {
            "sigma_1_largest": _quantiles(sigma_1),
            "sigma_2": _quantiles(sigma_2),
            "sigma_3_smallest": _quantiles(sigma_3),
        },
        "smallest_singular_value": _quantiles(smin),
        "condition_number": _quantiles(cond),
        "speed_column_sensitivity": _quantiles(speed_sensitivity),
        "load_column_sensitivity": _quantiles(load_sensitivity),
        "load_to_speed_sensitivity_ratio": _quantiles(load_to_speed),
    }


def shifted_closed_loop_params(nominal: DCMotorParams) -> DCMotorParams:
    """Exact parameter-variation definition used by the frozen V3/C4 evaluators."""
    return replace(
        nominal,
        resistance=1.20 * nominal.resistance,
        inductance=0.85 * nominal.inductance,
        back_emf_constant=1.15 * nominal.back_emf_constant,
        torque_constant=0.85 * nominal.torque_constant,
        inertia=1.20 * nominal.inertia,
        viscous_friction=1.20 * nominal.viscous_friction,
        coulomb_friction=1.20 * nominal.coulomb_friction,
    )


def self_check() -> dict:
    """Validate analytic continuous/RK4 Jacobians and structural rank cases."""
    params = DCMotorParams()
    representative = [
        np.array([0.0, 0.0, 0.03]),
        np.array([1.5, 20.0, 0.06]),
        np.array([4.5, 55.0, 0.09]),
        np.array([-1.0, -4.0, 0.0]),
    ]
    continuous_errors = []
    rk4_errors = []
    for state in representative:
        analytic = continuous_jacobian(state, 6.0, params)
        numerical = finite_difference_jacobian(
            lambda candidate: augmented_dynamics(candidate, 6.0, params), state
        )
        continuous_errors.append(float(np.max(np.abs(analytic - numerical))))

        analytic_rk4 = rk4_state_jacobian(state, 6.0, params, 0.01)
        numerical_rk4 = finite_difference_jacobian(
            lambda candidate: rk4_augmented_step(candidate, 6.0, params, 0.01),
            state,
        )
        rk4_errors.append(float(np.max(np.abs(analytic_rk4 - numerical_rk4))))

    nominal_rank = int(np.linalg.matrix_rank(continuous_observability_matrix(representative[1], 6.0, params)))
    no_back_emf = replace(params, back_emf_constant=0.0)
    no_back_emf_rank = int(
        np.linalg.matrix_rank(
            continuous_observability_matrix(representative[1], 6.0, no_back_emf)
        )
    )
    if max(continuous_errors) > 1e-7:
        raise AssertionError(f"continuous Jacobian finite-difference mismatch: {continuous_errors}")
    if max(rk4_errors) > 1e-7:
        raise AssertionError(f"RK4 Jacobian finite-difference mismatch: {rk4_errors}")
    if nominal_rank != 3 or no_back_emf_rank >= 3:
        raise AssertionError(
            f"unexpected structural ranks nominal={nominal_rank}, Kb=0={no_back_emf_rank}"
        )
    return {
        "representative_states": [state.tolist() for state in representative],
        "continuous_jacobian_max_abs_error": max(continuous_errors),
        "rk4_jacobian_max_abs_error": max(rk4_errors),
        "nominal_structural_rank": nominal_rank,
        "kb_zero_structural_rank": no_back_emf_rank,
    }


def _parameter_ranges(data: np.lib.npyio.NpzFile) -> dict:
    names = [str(name) for name in data["parameter_names"]]
    values = np.asarray(data["parameter_values"], dtype=float)
    nominal = asdict(DCMotorParams())
    return {
        name: {
            "nominal": float(nominal[name]),
            "saved_min": float(values[:, index].min()),
            "saved_max": float(values[:, index].max()),
            "saved_min_ratio_to_nominal": float(values[:, index].min() / nominal[name]),
            "saved_max_ratio_to_nominal": float(values[:, index].max() / nominal[name]),
        }
        for index, name in enumerate(names)
    }


def _range(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p99": float(np.quantile(values, 0.99)),
    }


def analyze() -> dict:
    checks = self_check()
    data = np.load(DATASET_PATH, allow_pickle=False)
    timestep = float(data["timestep"])
    if not np.isclose(timestep, 0.01, rtol=0, atol=1e-15):
        raise RuntimeError(f"unexpected saved timestep {timestep}; expected 0.01 s")

    train_ids = np.asarray(data["split_run_ids_train"], dtype=int)
    validation_ids = np.asarray(data["split_run_ids_validation"], dtype=int)
    development_ids = np.concatenate((train_ids, validation_ids))
    if len(set(development_ids.tolist())) != len(development_ids):
        raise RuntimeError("train/validation run IDs overlap")

    nominal_params = DCMotorParams()
    shifted_params = shifted_closed_loop_params(nominal_params)

    current_all = np.asarray(data["current"], dtype=float)
    speed_all = np.asarray(data["y_true"], dtype=float)
    voltage_all = np.asarray(data["voltage"], dtype=float)
    load_all = np.asarray(data["load_torque"], dtype=float)
    time = np.asarray(data["time"], dtype=float)

    current = current_all[development_ids]
    speed = speed_all[development_ids]
    voltage = voltage_all[development_ids]
    load = load_all[development_ids]

    current_dot = np.gradient(current, timestep, axis=1)
    acceleration = np.gradient(speed, timestep, axis=1)
    electrical_drive = (
        voltage - nominal_params.resistance * current - nominal_params.back_emf_constant * speed
    )

    current_abs = np.abs(current)
    speed_abs = np.abs(speed)
    acceleration_abs = np.abs(acceleration)
    current_dot_abs = np.abs(current_dot)
    electrical_drive_abs = np.abs(electrical_drive)

    thresholds = {
        "low_current_abs_q25_A": float(np.quantile(current_abs, 0.25)),
        "high_current_abs_q75_A": float(np.quantile(current_abs, 0.75)),
        "low_speed_abs_q25_rad_s": float(np.quantile(speed_abs, 0.25)),
        "high_speed_abs_q75_rad_s": float(np.quantile(speed_abs, 0.75)),
        "high_acceleration_abs_q75_rad_s2": float(np.quantile(acceleration_abs, 0.75)),
        "low_voltage_q25_V": float(np.quantile(voltage, 0.25)),
        "low_electrical_drive_abs_q25_V": float(np.quantile(electrical_drive_abs, 0.25)),
        "steady_current_derivative_abs_q25_A_s": float(np.quantile(current_dot_abs, 0.25)),
        "steady_acceleration_abs_q25_rad_s2": float(np.quantile(acceleration_abs, 0.25)),
        "voltage_saturation_low_V": 0.01,
        "voltage_saturation_high_V": 11.99,
        "load_change_neighborhood_s": 0.10,
    }

    state_scales = np.array(
        [
            float(np.std(current)),
            float(np.std(speed)),
            float(np.std(load)),
        ],
        dtype=float,
    )
    if np.any(state_scales <= 0):
        raise RuntimeError(f"invalid development state scales {state_scales}")
    output_scale = float(np.std(current))

    primary_nominal: list[dict[str, float]] = []
    primary_shifted: list[dict[str, float]] = []
    secondary_nominal: list[dict[str, float]] = []
    secondary_shifted: list[dict[str, float]] = []
    metadata: list[dict[str, float | int | bool]] = []

    max_horizon = max(PRIMARY_HORIZON_SAMPLES, SECONDARY_HORIZON_SAMPLES)
    max_start = current.shape[1] - max_horizon

    for local_run_index, run_id in enumerate(development_ids):
        nominal_transitions = []
        shifted_transitions = []
        for sample in range(current.shape[1] - 1):
            state = np.array(
                [current[local_run_index, sample], speed[local_run_index, sample], load[local_run_index, sample]],
                dtype=float,
            )
            sample_voltage = float(voltage[local_run_index, sample])
            nominal_transitions.append(
                rk4_state_jacobian(state, sample_voltage, nominal_params, timestep)
            )
            shifted_transitions.append(
                rk4_state_jacobian(state, sample_voltage, shifted_params, timestep)
            )

        load_change_indices = np.flatnonzero(
            np.abs(np.diff(load[local_run_index], prepend=load[local_run_index, 0])) > 1e-12
        )
        load_change_radius = int(round(thresholds["load_change_neighborhood_s"] / timestep))

        for sample in range(max_start + 1):
            nominal_primary_matrix = finite_horizon_observability_matrix(
                nominal_transitions[sample : sample + PRIMARY_HORIZON_SAMPLES - 1],
                PRIMARY_HORIZON_SAMPLES,
            )
            shifted_primary_matrix = finite_horizon_observability_matrix(
                shifted_transitions[sample : sample + PRIMARY_HORIZON_SAMPLES - 1],
                PRIMARY_HORIZON_SAMPLES,
            )
            nominal_secondary_matrix = finite_horizon_observability_matrix(
                nominal_transitions[sample : sample + SECONDARY_HORIZON_SAMPLES - 1],
                SECONDARY_HORIZON_SAMPLES,
            )
            shifted_secondary_matrix = finite_horizon_observability_matrix(
                shifted_transitions[sample : sample + SECONDARY_HORIZON_SAMPLES - 1],
                SECONDARY_HORIZON_SAMPLES,
            )

            def record(matrix: np.ndarray) -> dict[str, float]:
                values = observability_metrics(matrix, state_scales, output_scale)
                singular_values = values["singular_values"]
                return {
                    "rank": float(values["rank"]),
                    "singular_value_1": float(singular_values[0]),
                    "singular_value_2": float(singular_values[1]),
                    "singular_value_3": float(singular_values[2]),
                    "smallest_singular_value": float(values["smallest_singular_value"]),
                    "condition_number": float(values["condition_number"]),
                    "speed_sensitivity": float(values["column_norms"][1]),
                    "load_sensitivity": float(values["column_norms"][2]),
                }

            primary_nominal.append(record(nominal_primary_matrix))
            primary_shifted.append(record(shifted_primary_matrix))
            secondary_nominal.append(record(nominal_secondary_matrix))
            secondary_shifted.append(record(shifted_secondary_matrix))

            i_abs = float(current_abs[local_run_index, sample])
            w_abs = float(speed_abs[local_run_index, sample])
            a_abs = float(acceleration_abs[local_run_index, sample])
            v = float(voltage[local_run_index, sample])
            edrive = float(electrical_drive_abs[local_run_index, sample])
            near_load_change = bool(
                load_change_indices.size
                and np.min(np.abs(load_change_indices - sample)) <= load_change_radius
            )
            metadata.append(
                {
                    "run_id": int(run_id),
                    "sample": int(sample),
                    "time_s": float(time[sample]),
                    "startup": bool(time[sample] < STARTUP_DURATION_S),
                    "low_current": i_abs <= thresholds["low_current_abs_q25_A"],
                    "high_current": i_abs >= thresholds["high_current_abs_q75_A"],
                    "low_speed": w_abs <= thresholds["low_speed_abs_q25_rad_s"],
                    "high_speed": w_abs >= thresholds["high_speed_abs_q75_rad_s"],
                    "high_acceleration": a_abs >= thresholds["high_acceleration_abs_q75_rad_s2"],
                    "low_voltage": v <= thresholds["low_voltage_q25_V"],
                    "voltage_saturation": v <= thresholds["voltage_saturation_low_V"] or v >= thresholds["voltage_saturation_high_V"],
                    "low_excitation": edrive <= thresholds["low_electrical_drive_abs_q25_V"],
                    "steady_operation": (
                        float(current_dot_abs[local_run_index, sample]) <= thresholds["steady_current_derivative_abs_q25_A_s"]
                        and a_abs <= thresholds["steady_acceleration_abs_q25_rad_s2"]
                    ),
                    "load_change": near_load_change,
                }
            )

    if not (
        len(primary_nominal)
        == len(primary_shifted)
        == len(secondary_nominal)
        == len(secondary_shifted)
        == len(metadata)
    ):
        raise RuntimeError("observability result arrays are misaligned")

    region_names = [
        "startup",
        "low_current",
        "high_current",
        "low_speed",
        "high_speed",
        "high_acceleration",
        "low_voltage",
        "voltage_saturation",
        "low_excitation",
        "steady_operation",
        "load_change",
    ]
    regions = {}
    for name in region_names:
        indices = [index for index, row in enumerate(metadata) if bool(row[name])]
        regions[name] = {
            "nominal_primary": _distribution_summary([primary_nominal[index] for index in indices]),
            "shifted_primary": _distribution_summary([primary_shifted[index] for index in indices]),
        }

    nominal_conditions = np.asarray([row["condition_number"] for row in primary_nominal], dtype=float)
    shifted_conditions = np.asarray([row["condition_number"] for row in primary_shifted], dtype=float)
    condition_ratio = shifted_conditions / nominal_conditions
    nominal_smin = np.asarray([row["smallest_singular_value"] for row in primary_nominal], dtype=float)
    shifted_smin = np.asarray([row["smallest_singular_value"] for row in primary_shifted], dtype=float)

    sample_state = np.array([1.0, 35.0, 0.03], dtype=float)
    analytic_o = continuous_observability_matrix(sample_state, 6.0, nominal_params)
    analytic_det = float(np.linalg.det(analytic_o))
    expected_det = float(
        -(nominal_params.back_emf_constant**2)
        / (nominal_params.inductance**2 * nominal_params.inertia)
    )

    nominal_ids = []
    nominal_vector = np.array(list(asdict(nominal_params).values()), dtype=float)
    for run_id, values in enumerate(np.asarray(data["parameter_values"], dtype=float)):
        if np.allclose(values, nominal_vector, rtol=0, atol=1e-7):
            nominal_ids.append(run_id)

    summary = {
        "analysis_role": "development observability/conditioning analysis only; no EKF controller implemented",
        "dataset": {
            "path": str(DATASET_PATH.relative_to(PROJECT)).replace("\\", "/"),
            "sha256": sha256(DATASET_PATH),
            "saved_seed": int(data["seed"]),
            "timestep_s": timestep,
            "duration_s": float(data["duration"]),
            "speed_measurement_noise_std_rad_s": float(data["speed_noise_std"]),
            "current_measurement_noise": "none; saved current is the exact simulated state and is fed directly to the auxiliary LSTM",
            "train_run_ids": train_ids.tolist(),
            "validation_run_ids": validation_ids.tolist(),
            "development_run_ids": development_ids.tolist(),
            "exact_nominal_parameter_run_ids": nominal_ids,
            "exact_nominal_parameter_train_validation_run_ids": [int(run_id) for run_id in development_ids if int(run_id) in nominal_ids],
            "excitation_types": {str(int(run_id)): str(data["excitation_type"][run_id]) for run_id in development_ids},
        },
        "motor": {
            "nominal_parameters": asdict(nominal_params),
            "closed_loop_parameter_variation": asdict(shifted_params),
            "dataset_generation_parameter_ranges": _parameter_ranges(data),
            "smooth_friction": "tau_f = Fc * tanh(omega / omega_s)",
            "smooth_friction_derivative": "d tau_f/d omega = (Fc/omega_s) * (1 - tanh(omega/omega_s)^2)",
            "continuous_dynamics": {
                "di_dt": "(V - R*i - Kb*omega)/L",
                "domega_dt": "(Kt*i - B*omega - T_L - Fc*tanh(omega/omega_s))/J",
                "dT_L_dt": "0 deterministic; future EKF process noise models slow random walk",
                "measurement": "y=i",
            },
        },
        "operating_ranges": {
            "all_saved_30_runs": {
                "current_A": _range(current_all),
                "speed_rad_s": _range(speed_all),
                "voltage_V": _range(voltage_all),
                "load_torque_Nm": _range(load_all),
            },
            "healthy_train_validation_population": {
                "current_A": _range(current),
                "speed_rad_s": _range(speed),
                "voltage_V": _range(voltage),
                "load_torque_Nm": _range(load),
                "acceleration_rad_s2": _range(acceleration),
            },
            "actuator_contract": {
                "dataset_voltage_limits_V": [float(value) for value in data["voltage_limits"]],
                "frozen_closed_loop_voltage_limits_V": [0.0, 12.0],
                "frozen_closed_loop_max_voltage_step_V_per_control_update": 2.0,
                "frozen_closed_loop_control_update_s": 0.05,
                "plant_step_s": 0.01,
            },
        },
        "observer_contract": PROPOSED_EKF_CONTRACT,
        "discretization": {
            "method": "exact tangent Jacobian of the explicit fixed-input RK4 state map used by frozen V3/C4 closed-loop evaluators",
            "plant_step_s": timestep,
            "primary_horizon_samples": PRIMARY_HORIZON_SAMPLES,
            "primary_nominal_duration_s": PRIMARY_HORIZON_SAMPLES * timestep,
            "secondary_horizon_samples": SECONDARY_HORIZON_SAMPLES,
            "secondary_nominal_duration_s": SECONDARY_HORIZON_SAMPLES * timestep,
            "note": "The saved dataset itself was generated with solve_ivp(max_step=0.01); RK4 is used here because it is the exact frozen closed-loop plant step relevant to the future comparator.",
        },
        "structural_observability": {
            "continuous_observability_matrix_at_representative_state": analytic_o.tolist(),
            "determinant_numeric": analytic_det,
            "determinant_closed_form": "-Kb^2/(L^2*J)",
            "determinant_closed_form_numeric": expected_det,
            "rank_condition": "rank=3 for finite positive L,J when Kb != 0; friction slope, input voltage, current, speed and load do not alter this algebraic rank condition",
            "physical_path": "omega affects current through back-EMF; T_L affects omega_dot and reaches current through the omega-to-back-EMF path",
        },
        "normalization_for_conditioning": {
            "reason": "raw observability conditioning mixes A, rad/s and N*m state units; dimensionless scaling is primary for numerical comparisons",
            "state_std_scales_train_validation": {
                "current_A": float(state_scales[0]),
                "speed_rad_s": float(state_scales[1]),
                "load_torque_Nm": float(state_scales[2]),
            },
            "output_current_std_A": output_scale,
            "region_thresholds": thresholds,
        },
        "numerical_observability": {
            "window_count": len(primary_nominal),
            "primary_0p2s_nominal": _distribution_summary(primary_nominal),
            "primary_0p2s_shifted_model": _distribution_summary(primary_shifted),
            "secondary_0p5s_nominal": _distribution_summary(secondary_nominal),
            "secondary_0p5s_shifted_model": _distribution_summary(secondary_shifted),
            "paired_shifted_to_nominal_condition_ratio": _quantiles(condition_ratio),
            "paired_nominal_smallest_singular_value": _quantiles(nominal_smin),
            "paired_shifted_smallest_singular_value": _quantiles(shifted_smin),
            "regions": regions,
        },
        "self_checks": checks,
    }
    return summary


def write_outputs(summary: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    nominal = summary["numerical_observability"]["primary_0p2s_nominal"]["condition_number"]
    shifted = summary["numerical_observability"]["primary_0p2s_shifted_model"]["condition_number"]
    labels = ["p05", "p25", "median", "p75", "p95", "p99"]
    positions = np.arange(len(labels), dtype=float)
    width = 0.36
    nominal_values = [nominal[label] for label in labels]
    shifted_values = [shifted[label] for label in labels]

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.bar(positions - width / 2, nominal_values, width=width, label="Nominal model")
    axis.bar(positions + width / 2, shifted_values, width=width, label="Shifted model")
    axis.set_xticks(positions, labels)
    axis.set_yscale("log")
    axis.set_ylabel("Dimensionless finite-horizon condition number")
    axis.set_xlabel("Distribution quantile across train/validation windows")
    axis.set_title("Augmented current-only observability conditioning (20 samples / 0.2 s)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check-only",
        action="store_true",
        help="Run analytic/numerical Jacobian and structural-rank checks without writing outputs.",
    )
    args = parser.parse_args()
    if args.self_check_only:
        print(json.dumps(self_check(), indent=2))
        return
    summary = analyze()
    write_outputs(summary)
    primary = summary["numerical_observability"]["primary_0p2s_nominal"]
    shifted = summary["numerical_observability"]["primary_0p2s_shifted_model"]
    print(f"Wrote {SUMMARY_PATH.relative_to(PROJECT)}")
    print(f"Wrote {PLOT_PATH.relative_to(PROJECT)}")
    print(f"Nominal full-rank fraction: {primary['full_rank_fraction']:.6f}")
    print(f"Nominal median condition number: {primary['condition_number']['median']:.6g}")
    print(f"Shifted median condition number: {shifted['condition_number']['median']:.6g}")


if __name__ == "__main__":
    main()
