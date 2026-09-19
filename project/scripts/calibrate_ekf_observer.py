"""Preregistered 27-candidate Q/R calibration for the current-only motor EKF."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from ekf_observer import AugmentedStateEKF, EKFConfig, EKFNumericalError  # noqa: E402
from motor_model import DCMotorParams  # noqa: E402


DATASET_PATH = PROJECT / "data/processed/dc_motor_lstm_dataset.npz"
PREREG_PATH = PROJECT / "EKF_OBSERVER_PREREGISTRATION.md"
OBSERVER_PATH = PROJECT / "src/ekf_observer.py"
MOTOR_MODEL_PATH = PROJECT / "src/motor_model.py"
OBSERVABILITY_PATH = PROJECT / "results/analysis/ekf_observability_summary.json"
MAIN_WEIGHTS_PATH = PROJECT / "results/lstm_model_weights.pt"
MAIN_CONFIG_PATH = PROJECT / "results/configs/lstm_model_config.json"
AUX_WEIGHTS_PATH = PROJECT / "results/auxiliary_model_weights.pt"
AUX_CONFIG_PATH = PROJECT / "results/configs/v2_auxiliary_config.json"
CONFIG_PATH = PROJECT / "results/configs/ekf_frozen_config.json"
CANDIDATE_PATH = PROJECT / "results/metrics/ekf_calibration_candidates.csv"
SELECTED_TRACE_PATH = PROJECT / "results/metrics/ekf_calibration_selected_trace.csv"

Q_DYN_VALUES = (1e-8, 1e-6, 1e-4)
M_T_VALUES = (1e-2, 1e-1, 1.0)
R_VALUES = (1e-8, 1e-6, 1e-4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_scales(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Return training-only std([current, speed, load torque])."""
    train_ids = np.asarray(data["split_run_ids_train"], dtype=int)
    values = np.stack(
        (
            np.asarray(data["current"], dtype=float)[train_ids].reshape(-1),
            np.asarray(data["y_true"], dtype=float)[train_ids].reshape(-1),
            np.asarray(data["load_torque"], dtype=float)[train_ids].reshape(-1),
        ),
        axis=1,
    )
    scales = values.std(axis=0, dtype=np.float64)
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RuntimeError(f"invalid training-only state scales: {scales}")
    return scales


def candidate_covariances(
    scales: np.ndarray, q_dyn: float, m_t: float, r: float
) -> tuple[np.ndarray, float]:
    variances = np.square(np.asarray(scales, dtype=float))
    q = np.diag(
        [variances[0] * q_dyn, variances[1] * q_dyn, variances[2] * q_dyn * m_t]
    )
    measurement_r = float(variances[0] * r)
    return q, measurement_r


def run_ekf_trajectory(
    voltage: np.ndarray,
    current: np.ndarray,
    q: np.ndarray,
    measurement_r: float,
    p0: np.ndarray,
    *,
    params: DCMotorParams | None = None,
    dt: float = 0.01,
) -> dict[str, np.ndarray | int | str | None]:
    """Run one causal EKF trajectory and retain audit diagnostics."""
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if voltage.shape != current.shape or voltage.ndim != 1 or len(voltage) < 2:
        raise ValueError("voltage/current must be equal-length 1-D trajectories")

    config = EKFConfig(Q=q, R=measurement_r, P0=p0, dt=dt, params=params or DCMotorParams())
    observer = AugmentedStateEKF(config)
    states = np.full((len(current), 3), np.nan, dtype=float)
    innovations = np.full(len(current), np.nan, dtype=float)
    innovation_variances = np.full(len(current), np.nan, dtype=float)
    min_covariance_eigenvalues = np.full(len(current), np.nan, dtype=float)
    states[0] = observer.initialize(float(current[0]))
    failure_count = 0
    failure_reason: str | None = None

    for sample in range(1, len(current)):
        try:
            state, diagnostics = observer.step(
                float(voltage[sample - 1]), float(current[sample])
            )
        except (EKFNumericalError, ValueError, np.linalg.LinAlgError) as exc:
            failure_count += 1
            failure_reason = f"sample {sample}: {exc}"
            break
        states[sample] = state
        innovations[sample] = diagnostics.innovation
        innovation_variances[sample] = diagnostics.innovation_variance
        min_covariance_eigenvalues[sample] = diagnostics.min_covariance_eigenvalue

    return {
        "states": states,
        "innovations": innovations,
        "innovation_variances": innovation_variances,
        "min_covariance_eigenvalues": min_covariance_eigenvalues,
        "failure_count": failure_count,
        "failure_reason": failure_reason,
    }


def score_candidate(
    data: np.lib.npyio.NpzFile,
    validation_ids: np.ndarray,
    q: np.ndarray,
    measurement_r: float,
    p0: np.ndarray,
    *,
    dt: float,
) -> dict[str, float | int | str | bool]:
    errors: list[np.ndarray] = []
    innovations: list[np.ndarray] = []
    nis_values: list[np.ndarray] = []
    loads: list[np.ndarray] = []
    total_failures = 0
    nonfinite_estimate_count = 0
    failure_runs = 0
    first_failure = ""

    for run_id in validation_ids:
        result = run_ekf_trajectory(
            data["voltage"][run_id],
            data["current"][run_id],
            q,
            measurement_r,
            p0,
            dt=dt,
        )
        failures = int(result["failure_count"])
        total_failures += failures
        failure_runs += int(failures > 0)
        if failures and not first_failure:
            first_failure = f"run {int(run_id)}: {result['failure_reason']}"

        states = np.asarray(result["states"], dtype=float)
        valid = np.isfinite(states[:, 1])
        nonfinite_estimate_count += int(np.sum(~valid))
        if np.any(valid):
            errors.append(states[valid, 1] - np.asarray(data["y_true"][run_id], dtype=float)[valid])
            loads.append(states[valid, 2])
        innovation = np.asarray(result["innovations"], dtype=float)
        innovation_variance = np.asarray(result["innovation_variances"], dtype=float)
        audit = np.isfinite(innovation) & np.isfinite(innovation_variance) & (innovation_variance > 0)
        if np.any(audit):
            innovations.append(innovation[audit])
            nis_values.append(np.square(innovation[audit]) / innovation_variance[audit])

    if not errors:
        return {
            "stable": False,
            "validation_rmse": np.inf,
            "validation_mae": np.inf,
            "validation_bias": np.inf,
            "validation_abs_bias": np.inf,
            "validation_sample_count": 0,
            "numerical_failure_count": total_failures,
            "nonfinite_estimate_count": nonfinite_estimate_count,
            "failure_run_count": failure_runs,
            "first_failure": first_failure or "no finite speed estimates",
            "innovation_mean": np.nan,
            "innovation_std": np.nan,
            "nis_mean": np.nan,
            "nis_median": np.nan,
            "load_estimate_min": np.nan,
            "load_estimate_max": np.nan,
            "load_estimate_abs_p99": np.nan,
        }

    error = np.concatenate(errors)
    innovation = np.concatenate(innovations) if innovations else np.array([], dtype=float)
    nis = np.concatenate(nis_values) if nis_values else np.array([], dtype=float)
    load = np.concatenate(loads) if loads else np.array([], dtype=float)
    stable = bool(total_failures == 0 and nonfinite_estimate_count == 0 and np.isfinite(error).all())
    return {
        "stable": stable,
        "validation_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "validation_mae": float(np.mean(np.abs(error))),
        "validation_bias": float(np.mean(error)),
        "validation_abs_bias": float(abs(np.mean(error))),
        "validation_sample_count": int(error.size),
        "numerical_failure_count": int(total_failures),
        "nonfinite_estimate_count": int(nonfinite_estimate_count),
        "failure_run_count": int(failure_runs),
        "first_failure": first_failure,
        "innovation_mean": float(np.mean(innovation)) if innovation.size else np.nan,
        "innovation_std": float(np.std(innovation)) if innovation.size else np.nan,
        "nis_mean": float(np.mean(nis)) if nis.size else np.nan,
        "nis_median": float(np.median(nis)) if nis.size else np.nan,
        "load_estimate_min": float(np.min(load)) if load.size else np.nan,
        "load_estimate_max": float(np.max(load)) if load.size else np.nan,
        "load_estimate_abs_p99": float(np.quantile(np.abs(load), 0.99)) if load.size else np.nan,
    }


def choose_candidate(table: pd.DataFrame) -> pd.Series:
    stable = table.loc[table["stable"].astype(bool)].copy()
    if stable.empty:
        raise RuntimeError("all 27 preregistered Q/R candidates were numerically invalid")
    best_rmse = float(stable["validation_rmse"].min())
    within_one_percent = stable.loc[stable["validation_rmse"] <= best_rmse * 1.01 + 1e-15]
    return within_one_percent.sort_values(
        ["validation_mae", "validation_abs_bias", "validation_rmse", "candidate_id"],
        kind="mergesort",
    ).iloc[0]


def write_selected_trace(
    data: np.lib.npyio.NpzFile,
    validation_ids: np.ndarray,
    q: np.ndarray,
    measurement_r: float,
    p0: np.ndarray,
    *,
    dt: float,
) -> None:
    rows: list[dict] = []
    time = np.asarray(data["time"], dtype=float)
    for run_id in validation_ids:
        result = run_ekf_trajectory(
            data["voltage"][run_id], data["current"][run_id], q, measurement_r, p0, dt=dt
        )
        states = np.asarray(result["states"], dtype=float)
        innovation = np.asarray(result["innovations"], dtype=float)
        innovation_variance = np.asarray(result["innovation_variances"], dtype=float)
        min_eig = np.asarray(result["min_covariance_eigenvalues"], dtype=float)
        true_speed = np.asarray(data["y_true"][run_id], dtype=float)
        true_load = np.asarray(data["load_torque"][run_id], dtype=float)
        for sample in range(len(time)):
            nis = (
                float(innovation[sample] ** 2 / innovation_variance[sample])
                if np.isfinite(innovation[sample])
                and np.isfinite(innovation_variance[sample])
                and innovation_variance[sample] > 0
                else np.nan
            )
            rows.append(
                {
                    "run_id": int(run_id),
                    "sample": sample,
                    "time_s": float(time[sample]),
                    "speed_estimate_rad_s": float(states[sample, 1]),
                    "true_speed_rad_s": float(true_speed[sample]),
                    "load_estimate_Nm": float(states[sample, 2]),
                    "true_load_Nm": float(true_load[sample]),
                    "innovation_A": float(innovation[sample]),
                    "innovation_variance_A2": float(innovation_variance[sample]),
                    "nis": nis,
                    "min_covariance_eigenvalue": float(min_eig[sample]),
                }
            )
    pd.DataFrame(rows).to_csv(SELECTED_TRACE_PATH, index=False)


def main() -> None:
    data = np.load(DATASET_PATH, allow_pickle=False)
    dt = float(data["timestep"])
    if not np.isclose(dt, 0.01, rtol=0.0, atol=1e-15):
        raise RuntimeError(f"unexpected dataset timestep {dt}")
    train_ids = np.asarray(data["split_run_ids_train"], dtype=int)
    validation_ids = np.asarray(data["split_run_ids_validation"], dtype=int)
    test_ids = np.asarray(data["split_run_ids_test"], dtype=int)
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
        raise RuntimeError("saved trajectory splits overlap")

    scales = training_scales(data)
    p0 = np.diag(np.square(scales))
    rows: list[dict] = []
    for candidate_index, (q_dyn, m_t, r) in enumerate(
        product(Q_DYN_VALUES, M_T_VALUES, R_VALUES), start=1
    ):
        q, measurement_r = candidate_covariances(scales, q_dyn, m_t, r)
        metrics = score_candidate(
            data, validation_ids, q, measurement_r, p0, dt=dt
        )
        rows.append(
            {
                "candidate_id": f"EKF_{candidate_index:02d}",
                "q_dyn": q_dyn,
                "m_T": m_t,
                "r": r,
                "q_current": q[0, 0],
                "q_speed": q[1, 1],
                "q_load": q[2, 2],
                "R": measurement_r,
                **metrics,
            }
        )

    table = pd.DataFrame(rows)
    if len(table) != 27 or table[["q_dyn", "m_T", "r"]].drop_duplicates().shape[0] != 27:
        raise AssertionError("calibration table must contain exactly the 27 preregistered candidates")
    selected = choose_candidate(table)
    table["selected"] = table["candidate_id"].eq(selected["candidate_id"])
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CANDIDATE_PATH, index=False)

    selected_q = np.diag(
        [float(selected["q_current"]), float(selected["q_speed"]), float(selected["q_load"])]
    )
    write_selected_trace(
        data,
        validation_ids,
        selected_q,
        float(selected["R"]),
        p0,
        dt=dt,
    )
    provenance = {
        "dataset_path": str(DATASET_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "dataset_sha256": sha256(DATASET_PATH),
        "preregistration_path": PREREG_PATH.name,
        "preregistration_sha256": sha256(PREREG_PATH),
        "observer_source_path": str(OBSERVER_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "observer_source_sha256": sha256(OBSERVER_PATH),
        "motor_model_source_path": str(MOTOR_MODEL_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "motor_model_source_sha256": sha256(MOTOR_MODEL_PATH),
        "calibration_script_path": str(Path(__file__).resolve().relative_to(PROJECT)).replace("\\", "/"),
        "calibration_script_sha256": sha256(Path(__file__).resolve()),
        "observability_summary_sha256": sha256(OBSERVABILITY_PATH),
        "main_lstm_weights_path": str(MAIN_WEIGHTS_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "main_lstm_weights_sha256": sha256(MAIN_WEIGHTS_PATH),
        "main_lstm_config_path": str(MAIN_CONFIG_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "main_lstm_config_sha256": sha256(MAIN_CONFIG_PATH),
        "auxiliary_lstm_weights_path": str(AUX_WEIGHTS_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "auxiliary_lstm_weights_sha256": sha256(AUX_WEIGHTS_PATH),
        "auxiliary_lstm_config_path": str(AUX_CONFIG_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "auxiliary_lstm_config_sha256": sha256(AUX_CONFIG_PATH),
        "candidate_table_path": str(CANDIDATE_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "candidate_table_sha256": sha256(CANDIDATE_PATH),
        "selected_validation_trace_path": str(SELECTED_TRACE_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "selected_validation_trace_sha256": sha256(SELECTED_TRACE_PATH),
        "train_run_ids": train_ids.tolist(),
        "validation_run_ids": validation_ids.tolist(),
        "excluded_test_run_ids": test_ids.tolist(),
        "selection_population": "saved validation trajectories only",
        "P0_population": "saved training trajectories only",
        "forbidden_selection_inputs": [
            "test trajectories",
            "fault scenarios",
            "V3 final holdout",
            "C4 development/final outcomes",
            "future EKF closed-loop outcomes",
        ],
    }
    config = {
        "status": "frozen after preregistered train/validation-only calibration",
        "state_order": ["current", "omega", "load_torque"],
        "online_inputs": ["voltage", "current"],
        "current_measurement_noise_injected": False,
        "dt_s": dt,
        "nominal_motor_params": asdict(DCMotorParams()),
        "training_only_state_std": {
            "current_A": float(scales[0]),
            "speed_rad_s": float(scales[1]),
            "load_torque_Nm": float(scales[2]),
        },
        "P0": p0.tolist(),
        "grid": {
            "q_dyn": list(Q_DYN_VALUES),
            "m_T": list(M_T_VALUES),
            "r": list(R_VALUES),
            "candidate_count": 27,
        },
        "selection_rule": "reject unstable; find minimum validation RMSE; among stable candidates within 1% of that RMSE choose lower MAE, then lower absolute bias, then deterministic candidate id",
        "selected_candidate": {
            "candidate_id": str(selected["candidate_id"]),
            "q_dyn": float(selected["q_dyn"]),
            "m_T": float(selected["m_T"]),
            "r": float(selected["r"]),
            "Q": selected_q.tolist(),
            "R": float(selected["R"]),
            "P0": p0.tolist(),
            "validation_rmse": float(selected["validation_rmse"]),
            "validation_mae": float(selected["validation_mae"]),
            "validation_bias": float(selected["validation_bias"]),
            "validation_abs_bias": float(selected["validation_abs_bias"]),
            "validation_sample_count": int(selected["validation_sample_count"]),
            "numerical_failure_count": int(selected["numerical_failure_count"]),
            "nonfinite_estimate_count": int(selected["nonfinite_estimate_count"]),
            "failure_run_count": int(selected["failure_run_count"]),
            "innovation_mean": float(selected["innovation_mean"]),
            "innovation_std": float(selected["innovation_std"]),
            "nis_mean": float(selected["nis_mean"]),
            "nis_median": float(selected["nis_median"]),
            "load_estimate_min": float(selected["load_estimate_min"]),
            "load_estimate_max": float(selected["load_estimate_max"]),
            "load_estimate_abs_p99": float(selected["load_estimate_abs_p99"]),
            "provenance": provenance,
        },
        "provenance": provenance,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {CANDIDATE_PATH.relative_to(PROJECT)} ({len(table)} candidates)")
    print(f"wrote {SELECTED_TRACE_PATH.relative_to(PROJECT)}")
    print(f"wrote {CONFIG_PATH.relative_to(PROJECT)}")
    print(
        f"selected {selected['candidate_id']}: RMSE={selected['validation_rmse']:.6g}, "
        f"MAE={selected['validation_mae']:.6g}, bias={selected['validation_bias']:.6g}, "
        f"numerical_failures={int(selected['numerical_failure_count'])}"
    )


if __name__ == "__main__":
    main()
