"""Estimator-only comparison of the frozen EKF and auxiliary LSTM.

This script never retunes Q/R.  It evaluates the frozen observer on the saved
held-out test trajectories and on the preregistered frozen parameter-variation
transfer, where the plant shifts at 3 s while the EKF remains fixed nominal.
Truth is used only after both estimators have produced their estimates, for
offline scoring and descriptive strata.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from auxiliary_sensor_model import load_auxiliary_model, auxiliary_predict_online  # noqa: E402
from ekf_observer import (  # noqa: E402
    AugmentedStateEKF,
    CURRENT_MEASUREMENT_JACOBIAN,
    EKFConfig,
    EKFNumericalError,
    rk4_state_jacobian,
)
from motor_model import DCMotorParams, dc_motor_dynamics  # noqa: E402


DATASET_PATH = PROJECT / "data/processed/dc_motor_lstm_dataset.npz"
FROZEN_CONFIG_PATH = PROJECT / "results/configs/ekf_frozen_config.json"
CALIBRATION_CANDIDATES_PATH = PROJECT / "results/metrics/ekf_calibration_candidates.csv"
OBSERVABILITY_PATH = PROJECT / "results/analysis/ekf_observability_summary.json"
AUX_WEIGHTS_PATH = PROJECT / "results/auxiliary_model_weights.pt"
AUX_CONFIG_PATH = PROJECT / "results/configs/v2_auxiliary_config.json"
OBSERVER_PATH = PROJECT / "src/ekf_observer.py"
MOTOR_MODEL_PATH = PROJECT / "src/motor_model.py"

COMPARISON_PATH = PROJECT / "results/metrics/ekf_estimator_comparison.csv"
TRANSFER_PATH = PROJECT / "results/metrics/ekf_estimator_transfer.csv"
RUNTIME_PATH = PROJECT / "results/metrics/ekf_estimator_runtime_samples.csv"
TRACE_PATH = PROJECT / "results/metrics/ekf_estimator_trace.csv"

WINDOW_LENGTH = 20
STARTUP_REPORT_END_S = 0.5
PARAMETER_SHIFT_START_S = 3.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_config() -> tuple[dict, EKFConfig]:
    record = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    if "frozen" not in str(record.get("status", "")).lower():
        raise RuntimeError("EKF config is not frozen")
    candidates = record.get("candidate_table")
    if isinstance(candidates, list):
        candidate_count = len(candidates)
    else:
        if not CALIBRATION_CANDIDATES_PATH.exists():
            raise RuntimeError("frozen config is missing the supporting calibration candidate table")
        candidate_frame = pd.read_csv(CALIBRATION_CANDIDATES_PATH)
        candidate_count = len(candidate_frame)
        if candidate_frame[["q_dyn", "m_T", "r"]].drop_duplicates().shape[0] != 27:
            raise RuntimeError("supporting calibration table does not contain 27 unique grid candidates")
        expected_hash = record.get("provenance", {}).get("candidate_table_sha256")
        if expected_hash != sha256(CALIBRATION_CANDIDATES_PATH):
            raise RuntimeError("supporting calibration table hash differs from frozen provenance")
    if candidate_count != 27:
        raise RuntimeError("frozen calibration evidence must contain exactly 27 candidates")
    selected = record.get("selected_candidate")
    if not isinstance(selected, dict):
        raise RuntimeError("frozen config is missing selected_candidate")
    q = np.asarray(selected["Q"], dtype=float)
    p0 = np.asarray(selected["P0"], dtype=float)
    config = EKFConfig(
        Q=q,
        R=float(selected["R"]),
        P0=p0,
        dt=float(record["dt_s"]),
        params=DCMotorParams(),
    )
    return record, config


def shifted_params(nominal: DCMotorParams) -> DCMotorParams:
    """Exact frozen V3/C4 parameter-variation definition."""
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


def rk4_motor_step(
    state: np.ndarray,
    voltage: float,
    load_torque: float,
    params: DCMotorParams,
    dt: float,
) -> np.ndarray:
    """Fixed-input 2-state RK4 step matching the frozen closed-loop plant form."""
    state = np.asarray(state, dtype=float)
    derivative = lambda candidate: dc_motor_dynamics(  # noqa: E731
        0.0, candidate, float(voltage), float(load_torque), params
    )
    k1 = derivative(state)
    k2 = derivative(state + dt * k1 / 2.0)
    k3 = derivative(state + dt * k2 / 2.0)
    k4 = derivative(state + dt * k3)
    result = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if not np.isfinite(result).all():
        raise RuntimeError("transfer plant produced a nonfinite state")
    return result


def frozen_transfer_trajectory(
    voltage: np.ndarray,
    time_s: np.ndarray,
    dt: float,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay held-out voltage under an exact frozen V3 transfer scenario."""
    voltage = np.asarray(voltage, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if voltage.shape != time_s.shape:
        raise ValueError("transfer voltage/time arrays must have equal shape")
    if scenario not in {"load_disturbance", "parameter_variation"}:
        raise ValueError(f"unsupported transfer scenario {scenario!r}")
    nominal = DCMotorParams()
    shifted = shifted_params(nominal)
    load_torque = np.full(len(time_s), 0.03, dtype=float)
    if scenario == "load_disturbance":
        load_torque[time_s >= PARAMETER_SHIFT_START_S] = 0.15
    states = np.zeros((len(time_s), 2), dtype=float)
    for sample in range(len(time_s) - 1):
        params = (
            shifted
            if scenario == "parameter_variation"
            and float(time_s[sample]) >= PARAMETER_SHIFT_START_S
            else nominal
        )
        states[sample + 1] = rk4_motor_step(
            states[sample],
            float(voltage[sample]),
            float(load_torque[sample]),
            params,
            dt,
        )
    return states[:, 0], states[:, 1], load_torque


def run_estimators(
    voltage: np.ndarray,
    current: np.ndarray,
    ekf_config: EKFConfig,
    aux_model,
    aux_normalization: dict[str, np.ndarray],
) -> dict:
    """Produce estimates using only causal voltage/current inputs."""
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if voltage.shape != current.shape or voltage.ndim != 1:
        raise ValueError("voltage/current must be equal-length 1-D arrays")

    size = len(current)
    ekf_speed = np.full(size, np.nan, dtype=float)
    ekf_load = np.full(size, np.nan, dtype=float)
    ekf_innovation = np.full(size, np.nan, dtype=float)
    ekf_innovation_variance = np.full(size, np.nan, dtype=float)
    ekf_runtime_ns = np.full(size, np.nan, dtype=float)
    aux_speed = np.full(size, np.nan, dtype=float)
    aux_runtime_ns = np.full(size, np.nan, dtype=float)

    observer = AugmentedStateEKF(ekf_config)
    initial = observer.initialize(float(current[0]))
    ekf_speed[0] = initial[1]
    ekf_load[0] = initial[2]
    ekf_failure_count = 0
    ekf_failure_sample = -1
    ekf_failure_reason = ""

    for sample in range(1, size):
        started = perf_counter_ns()
        try:
            state, diagnostics = observer.step(
                float(voltage[sample - 1]), float(current[sample])
            )
        except (EKFNumericalError, ValueError, np.linalg.LinAlgError) as exc:
            ekf_runtime_ns[sample] = float(perf_counter_ns() - started)
            ekf_failure_count += 1
            ekf_failure_sample = sample
            ekf_failure_reason = str(exc)
            break
        ekf_runtime_ns[sample] = float(perf_counter_ns() - started)
        ekf_speed[sample] = state[1]
        ekf_load[sample] = state[2]
        ekf_innovation[sample] = diagnostics.innovation
        ekf_innovation_variance[sample] = diagnostics.innovation_variance

    aux_failure_count = 0
    aux_failure_sample = -1
    aux_failure_reason = ""
    for sample in range(WINDOW_LENGTH, size):
        started = perf_counter_ns()
        try:
            estimate = auxiliary_predict_online(
                aux_model,
                voltage[sample - WINDOW_LENGTH : sample],
                current[sample - WINDOW_LENGTH : sample],
                aux_normalization,
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            aux_runtime_ns[sample] = float(perf_counter_ns() - started)
            aux_failure_count += 1
            aux_failure_sample = sample
            aux_failure_reason = str(exc)
            break
        aux_runtime_ns[sample] = float(perf_counter_ns() - started)
        if not np.isfinite(estimate):
            aux_failure_count += 1
            aux_failure_sample = sample
            aux_failure_reason = "nonfinite auxiliary estimate"
            break
        aux_speed[sample] = float(estimate)

    return {
        "ekf_speed": ekf_speed,
        "ekf_load": ekf_load,
        "ekf_innovation": ekf_innovation,
        "ekf_innovation_variance": ekf_innovation_variance,
        "ekf_runtime_ns": ekf_runtime_ns,
        "ekf_failure_count": ekf_failure_count,
        "ekf_failure_sample": ekf_failure_sample,
        "ekf_failure_reason": ekf_failure_reason,
        "aux_speed": aux_speed,
        "aux_runtime_ns": aux_runtime_ns,
        "aux_failure_count": aux_failure_count,
        "aux_failure_sample": aux_failure_sample,
        "aux_failure_reason": aux_failure_reason,
    }


def weak_observability_mask(
    voltage: np.ndarray,
    current: np.ndarray,
    speed: np.ndarray,
    load: np.ndarray,
    observability: dict,
    *,
    dt: float,
) -> np.ndarray:
    """Offline 0.2 s diagnostic using the frozen development thresholds."""
    horizon = int(observability["discretization"]["primary_horizon_samples"])
    normalization = observability["normalization_for_conditioning"]
    scale_record = normalization["state_std_scales_train_validation"]
    state_scales = np.array(
        [
            scale_record["current_A"],
            scale_record["speed_rad_s"],
            scale_record["load_torque_Nm"],
        ],
        dtype=float,
    )
    output_scale = float(normalization["output_current_std_A"])
    numerical = observability["numerical_observability"]["primary_0p2s_nominal"]
    condition_threshold = float(numerical["condition_number"]["p99"])
    smin_threshold = float(numerical["smallest_singular_value"]["p01"])

    states = np.column_stack((current, speed, load)).astype(float)
    transitions = [
        rk4_state_jacobian(states[sample], float(voltage[sample]), DCMotorParams(), dt)
        for sample in range(len(states) - 1)
    ]
    mask = np.zeros(len(states), dtype=bool)
    c_matrix = CURRENT_MEASUREMENT_JACOBIAN
    for sample in range(0, len(states) - horizon + 1):
        phi = np.eye(3, dtype=float)
        rows = [c_matrix.copy()]
        for offset in range(horizon - 1):
            phi = transitions[sample + offset] @ phi
            rows.append(c_matrix @ phi)
        matrix = np.vstack(rows) @ np.diag(state_scales) / output_scale
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        smin = float(singular_values[-1])
        condition = float(singular_values[0] / smin) if smin > 0 else np.inf
        mask[sample] = condition > condition_threshold or smin < smin_threshold
    return mask


def build_strata(
    time_s: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    true_speed: np.ndarray,
    load: np.ndarray,
    weak_mask: np.ndarray,
    observability: dict,
    *,
    dt: float,
) -> dict[str, np.ndarray]:
    """Create offline masks from thresholds frozen in development analysis."""
    thresholds = observability["normalization_for_conditioning"]["region_thresholds"]
    abs_current = np.abs(current)
    abs_speed = np.abs(true_speed)
    abs_acceleration = np.abs(np.gradient(true_speed, dt))
    changes = np.flatnonzero(np.abs(np.diff(load, prepend=load[0])) > 1e-12)
    radius = int(round(float(thresholds["load_change_neighborhood_s"]) / dt))
    load_change = np.zeros(len(load), dtype=bool)
    for index in changes:
        load_change[max(0, index - radius) : min(len(load), index + radius + 1)] = True
    paired = np.arange(len(time_s)) >= WINDOW_LENGTH
    return {
        "all_paired": paired,
        "startup_pre_lstm_window_0_0p2s": np.arange(len(time_s)) < WINDOW_LENGTH,
        "startup_post_window_to_0p5s": paired & (time_s < STARTUP_REPORT_END_S),
        "load_change": paired & load_change,
        "low_current": paired & (abs_current <= float(thresholds["low_current_abs_q25_A"])),
        "high_current": paired & (abs_current >= float(thresholds["high_current_abs_q75_A"])),
        "low_speed": paired & (abs_speed <= float(thresholds["low_speed_abs_q25_rad_s"])),
        "high_speed": paired & (abs_speed >= float(thresholds["high_speed_abs_q75_rad_s"])),
        "low_acceleration": paired
        & (abs_acceleration <= float(thresholds["steady_acceleration_abs_q25_rad_s2"])),
        "high_acceleration": paired
        & (abs_acceleration >= float(thresholds["high_acceleration_abs_q75_rad_s2"])),
        "voltage_saturation": paired
        & (
            (voltage <= float(thresholds["voltage_saturation_low_V"]))
            | (voltage >= float(thresholds["voltage_saturation_high_V"]))
        ),
        "weak_observability": paired & weak_mask,
    }


def metric_row(
    *,
    population: str,
    estimator: str,
    stratum: str,
    mask: np.ndarray,
    estimate: np.ndarray,
    truth: np.ndarray,
    runtime_ns: np.ndarray,
    run_ids: np.ndarray,
    numerical_failure_count: int,
    failure_run_count: int,
    provenance: dict[str, str],
) -> dict:
    selected = np.asarray(mask, dtype=bool)
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    runtime_ns = np.asarray(runtime_ns, dtype=float)
    finite = selected & np.isfinite(estimate) & np.isfinite(truth)
    selected_count = int(np.sum(selected))
    finite_count = int(np.sum(finite))
    error = estimate[finite] - truth[finite]
    runtime = runtime_ns[finite & np.isfinite(runtime_ns)] / 1000.0
    sample_run_ids = np.asarray(run_ids, dtype=int)
    return {
        "population": population,
        "estimator": estimator,
        "stratum": stratum,
        "sample_count": selected_count,
        "trajectory_count": int(np.unique(sample_run_ids[selected]).size) if selected_count else 0,
        "finite_sample_count": finite_count,
        "nonfinite_sample_count": selected_count - finite_count,
        "finite_fraction": float(finite_count / selected_count) if selected_count else np.nan,
        "rmse_rad_s": float(np.sqrt(np.mean(np.square(error)))) if error.size else np.nan,
        "mae_rad_s": float(np.mean(np.abs(error))) if error.size else np.nan,
        "bias_rad_s": float(np.mean(error)) if error.size else np.nan,
        "numerical_failure_count": int(numerical_failure_count),
        "failure_run_count": int(failure_run_count),
        "runtime_sample_count": int(runtime.size),
        "runtime_mean_us": float(np.mean(runtime)) if runtime.size else np.nan,
        "runtime_p50_us": float(np.quantile(runtime, 0.50)) if runtime.size else np.nan,
        "runtime_p95_us": float(np.quantile(runtime, 0.95)) if runtime.size else np.nan,
        "runtime_p99_us": float(np.quantile(runtime, 0.99)) if runtime.size else np.nan,
        "runtime_max_us": float(np.max(runtime)) if runtime.size else np.nan,
        **provenance,
    }


def evaluate_population(
    *,
    population: str,
    data: np.lib.npyio.NpzFile,
    test_ids: np.ndarray,
    ekf_config: EKFConfig,
    aux_model,
    aux_normalization: dict[str, np.ndarray],
    observability: dict,
    transfer_scenario: str | None,
    provenance: dict[str, str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Evaluate one population and return metrics, runtimes, and traces."""
    dt = float(data["timestep"])
    time_s = np.asarray(data["time"], dtype=float)
    aggregate: list[dict] = []
    runtime_rows: list[dict] = []
    trace_rows: list[dict] = []

    speed_by_run: list[np.ndarray] = []
    ekf_by_run: list[np.ndarray] = []
    aux_by_run: list[np.ndarray] = []
    ekf_runtime_by_run: list[np.ndarray] = []
    aux_runtime_by_run: list[np.ndarray] = []
    stratum_by_run: dict[str, list[np.ndarray]] = {}
    run_id_by_run: list[np.ndarray] = []
    ekf_failures = 0
    aux_failures = 0
    ekf_failure_runs = 0
    aux_failure_runs = 0

    for run_id in test_ids:
        voltage = np.asarray(data["voltage"][run_id], dtype=float)
        if transfer_scenario is not None:
            current, true_speed, load = frozen_transfer_trajectory(
                voltage, time_s, dt, transfer_scenario
            )
        else:
            load = np.asarray(data["load_torque"][run_id], dtype=float)
            current = np.asarray(data["current"][run_id], dtype=float)
            true_speed = np.asarray(data["y_true"][run_id], dtype=float)

        result = run_estimators(
            voltage, current, ekf_config, aux_model, aux_normalization
        )
        ekf_failures += int(result["ekf_failure_count"])
        aux_failures += int(result["aux_failure_count"])
        ekf_failure_runs += int(result["ekf_failure_count"] > 0)
        aux_failure_runs += int(result["aux_failure_count"] > 0)

        weak = weak_observability_mask(
            voltage, current, true_speed, load, observability, dt=dt
        )
        strata = build_strata(
            time_s,
            voltage,
            current,
            true_speed,
            load,
            weak,
            observability,
            dt=dt,
        )
        if transfer_scenario == "parameter_variation":
            strata["parameter_shift_post_3s"] = (
                (np.arange(len(time_s)) >= WINDOW_LENGTH)
                & (time_s >= PARAMETER_SHIFT_START_S)
            )
        if transfer_scenario == "load_disturbance":
            strata["load_step_post_3s"] = (
                (np.arange(len(time_s)) >= WINDOW_LENGTH)
                & (time_s >= PARAMETER_SHIFT_START_S)
            )
        for name, mask in strata.items():
            stratum_by_run.setdefault(name, []).append(mask)

        speed_by_run.append(true_speed)
        ekf_by_run.append(np.asarray(result["ekf_speed"], dtype=float))
        aux_by_run.append(np.asarray(result["aux_speed"], dtype=float))
        ekf_runtime_by_run.append(np.asarray(result["ekf_runtime_ns"], dtype=float))
        aux_runtime_by_run.append(np.asarray(result["aux_runtime_ns"], dtype=float))
        run_id_by_run.append(np.full(len(time_s), int(run_id), dtype=int))

        for sample in range(len(time_s)):
            for estimator, runtime_ns in (
                ("EKF", result["ekf_runtime_ns"]),
                ("auxiliary_LSTM", result["aux_runtime_ns"]),
            ):
                value = float(runtime_ns[sample])
                if np.isfinite(value):
                    runtime_rows.append(
                        {
                            "population": population,
                            "run_id": int(run_id),
                            "sample": sample,
                            "time_s": float(time_s[sample]),
                            "estimator": estimator,
                            "update_runtime_us": value / 1000.0,
                        }
                    )

            innovation = float(result["ekf_innovation"][sample])
            innovation_variance = float(result["ekf_innovation_variance"][sample])
            trace_rows.append(
                {
                    "population": population,
                    "run_id": int(run_id),
                    "sample": sample,
                    "time_s": float(time_s[sample]),
                    "voltage_V": float(voltage[sample]),
                    "current_A": float(current[sample]),
                    "true_speed_rad_s": float(true_speed[sample]),
                    "true_load_Nm": float(load[sample]),
                    "ekf_speed_rad_s": float(result["ekf_speed"][sample]),
                    "ekf_load_Nm": float(result["ekf_load"][sample]),
                    "aux_speed_rad_s": float(result["aux_speed"][sample]),
                    "ekf_innovation_A": innovation,
                    "ekf_innovation_variance_A2": innovation_variance,
                    "ekf_nis": (
                        innovation * innovation / innovation_variance
                        if np.isfinite(innovation)
                        and np.isfinite(innovation_variance)
                        and innovation_variance > 0
                        else np.nan
                    ),
                    "weak_observability": bool(weak[sample]),
                    "after_parameter_shift": bool(
                        transfer_scenario == "parameter_variation"
                        and float(time_s[sample]) >= PARAMETER_SHIFT_START_S
                    ),
                    "after_load_step": bool(
                        transfer_scenario == "load_disturbance"
                        and float(time_s[sample]) >= PARAMETER_SHIFT_START_S
                    ),
                    "ekf_failure_sample": int(result["ekf_failure_sample"]),
                    "aux_failure_sample": int(result["aux_failure_sample"]),
                }
            )

    truth = np.concatenate(speed_by_run)
    ekf_estimate = np.concatenate(ekf_by_run)
    aux_estimate = np.concatenate(aux_by_run)
    ekf_runtime = np.concatenate(ekf_runtime_by_run)
    aux_runtime = np.concatenate(aux_runtime_by_run)
    sample_run_ids = np.concatenate(run_id_by_run)
    masks = {name: np.concatenate(parts) for name, parts in stratum_by_run.items()}

    for stratum, mask in masks.items():
        if stratum == "startup_pre_lstm_window_0_0p2s":
            aggregate.append(
                metric_row(
                    population=population,
                    estimator="EKF",
                    stratum=stratum,
                    mask=mask,
                    estimate=ekf_estimate,
                    truth=truth,
                    runtime_ns=ekf_runtime,
                    run_ids=sample_run_ids,
                    numerical_failure_count=ekf_failures,
                    failure_run_count=ekf_failure_runs,
                    provenance=provenance,
                )
            )
            continue
        aggregate.append(
            metric_row(
                population=population,
                estimator="EKF",
                stratum=stratum,
                mask=mask,
                estimate=ekf_estimate,
                truth=truth,
                runtime_ns=ekf_runtime,
                run_ids=sample_run_ids,
                numerical_failure_count=ekf_failures,
                failure_run_count=ekf_failure_runs,
                provenance=provenance,
            )
        )
        aggregate.append(
            metric_row(
                population=population,
                estimator="auxiliary_LSTM",
                stratum=stratum,
                mask=mask,
                estimate=aux_estimate,
                truth=truth,
                runtime_ns=aux_runtime,
                run_ids=sample_run_ids,
                numerical_failure_count=aux_failures,
                failure_run_count=aux_failure_runs,
                provenance=provenance,
            )
        )
    return aggregate, runtime_rows, trace_rows


def main() -> None:
    frozen_record, ekf_config = load_frozen_config()
    data = np.load(DATASET_PATH, allow_pickle=False)
    test_ids = np.asarray(data["split_run_ids_test"], dtype=int)
    if set(test_ids) & set(frozen_record["provenance"]["train_run_ids"]):
        raise RuntimeError("held-out test IDs overlap calibration training IDs")
    if set(test_ids) & set(frozen_record["provenance"]["validation_run_ids"]):
        raise RuntimeError("held-out test IDs overlap calibration validation IDs")
    if sha256(DATASET_PATH) != frozen_record["provenance"]["dataset_sha256"]:
        raise RuntimeError("dataset hash no longer matches frozen calibration provenance")

    observability = json.loads(OBSERVABILITY_PATH.read_text(encoding="utf-8"))
    aux_model, aux_config = load_auxiliary_model(AUX_WEIGHTS_PATH, AUX_CONFIG_PATH)
    if int(aux_config["window_length"]) != WINDOW_LENGTH:
        raise RuntimeError("auxiliary model window length changed from preregistered 20 samples")
    aux_normalization = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in aux_config["normalization"].items()
    }

    provenance = {
        "ekf_config_sha256": sha256(FROZEN_CONFIG_PATH),
        "dataset_sha256": sha256(DATASET_PATH),
        "auxiliary_lstm_weights_sha256": sha256(AUX_WEIGHTS_PATH),
        "auxiliary_lstm_config_sha256": sha256(AUX_CONFIG_PATH),
        "observer_source_sha256": sha256(OBSERVER_PATH),
        "motor_model_source_sha256": sha256(MOTOR_MODEL_PATH),
        "observability_summary_sha256": sha256(OBSERVABILITY_PATH),
        "evaluation_script_sha256": sha256(Path(__file__).resolve()),
        "selected_candidate_id": str(frozen_record["selected_candidate"]["candidate_id"]),
    }

    clean_metrics, clean_runtime, clean_trace = evaluate_population(
        population="clean_heldout_test",
        data=data,
        test_ids=test_ids,
        ekf_config=ekf_config,
        aux_model=aux_model,
        aux_normalization=aux_normalization,
        observability=observability,
        transfer_scenario=None,
        provenance=provenance,
    )
    load_metrics, load_runtime, load_trace = evaluate_population(
        population="load_disturbance_fixed_nominal_observer",
        data=data,
        test_ids=test_ids,
        ekf_config=ekf_config,
        aux_model=aux_model,
        aux_normalization=aux_normalization,
        observability=observability,
        transfer_scenario="load_disturbance",
        provenance=provenance,
    )
    parameter_metrics, parameter_runtime, parameter_trace = evaluate_population(
        population="parameter_variation_fixed_nominal_observer",
        data=data,
        test_ids=test_ids,
        ekf_config=ekf_config,
        aux_model=aux_model,
        aux_normalization=aux_normalization,
        observability=observability,
        transfer_scenario="parameter_variation",
        provenance=provenance,
    )

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clean_metrics).to_csv(COMPARISON_PATH, index=False)
    transfer_metrics = load_metrics + parameter_metrics
    pd.DataFrame(transfer_metrics).to_csv(TRANSFER_PATH, index=False)
    pd.DataFrame(clean_runtime + load_runtime + parameter_runtime).to_csv(RUNTIME_PATH, index=False)
    pd.DataFrame(clean_trace + load_trace + parameter_trace).to_csv(TRACE_PATH, index=False)

    clean = pd.DataFrame(clean_metrics)
    transfer = pd.DataFrame(transfer_metrics)
    clean_all = clean.loc[clean["stratum"].eq("all_paired"), ["estimator", "rmse_rad_s", "mae_rad_s", "bias_rad_s", "numerical_failure_count"]]
    transfer_all = transfer.loc[transfer["stratum"].eq("all_paired"), ["estimator", "rmse_rad_s", "mae_rad_s", "bias_rad_s", "numerical_failure_count"]]
    print(f"wrote {COMPARISON_PATH.relative_to(PROJECT)}")
    print(f"wrote {TRANSFER_PATH.relative_to(PROJECT)}")
    print(f"wrote {RUNTIME_PATH.relative_to(PROJECT)}")
    print(f"wrote {TRACE_PATH.relative_to(PROJECT)}")
    print("clean all-paired:\n" + clean_all.to_string(index=False))
    print("transfer all-paired:\n" + transfer_all.to_string(index=False))


if __name__ == "__main__":
    main()
