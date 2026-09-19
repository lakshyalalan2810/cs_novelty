"""Fail-closed independent verifier for the preregistered EKF comparator.

This verifier only reads saved evidence.  It never generates, repairs, or
substitutes scientific results.  Missing artifacts, incomplete schemas, stale
hash bindings, or arithmetic mismatches are verification failures.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
CONFIGS = RESULTS / "configs"
METRICS = RESULTS / "metrics"

FROZEN_EVIDENCE_PATH = CONFIGS / "ekf_frozen_evidence_hashes.json"
EKF_CONFIG_PATH = CONFIGS / "ekf_frozen_config.json"
CLOSED_LOOP_PLAN_PATH = CONFIGS / "ekf_closed_loop_plan.json"
CALIBRATION_CANDIDATES_PATH = METRICS / "ekf_calibration_candidates.csv"
CALIBRATION_TRACE_PATH = METRICS / "ekf_calibration_selected_trace.csv"
CLOSED_LOOP_RUNS_PATH = METRICS / "ekf_closed_loop_runs.csv"
CLOSED_LOOP_SUMMARY_PATH = METRICS / "ekf_closed_loop_summary.json"
CLOSED_LOOP_TRACES_PATH = METRICS / "ekf_closed_loop_traces.csv"
CLOSED_LOOP_EVALUATOR_PATH = PROJECT / "scripts" / "evaluate_ekf_closed_loop.py"

ESTIMATOR_COMPARISON_PATH = METRICS / "ekf_estimator_comparison.csv"
ESTIMATOR_TRANSFER_PATH = METRICS / "ekf_estimator_transfer.csv"
ESTIMATOR_RUNTIME_PATH = METRICS / "ekf_estimator_runtime_samples.csv"
ESTIMATOR_TRACE_PATH = METRICS / "ekf_estimator_trace.csv"
ESTIMATOR_SCRIPT_PATH = PROJECT / "scripts" / "evaluate_ekf_estimator.py"
OBSERVABILITY_PATH = PROJECT / "results" / "analysis" / "ekf_observability_summary.json"
AUX_CONFIG_PATH = CONFIGS / "v2_auxiliary_config.json"

Q_DYN_VALUES = (1e-8, 1e-6, 1e-4)
M_T_VALUES = (1e-2, 1e-1, 1.0)
R_VALUES = (1e-8, 1e-6, 1e-4)
EXPECTED_GRID = set(product(Q_DYN_VALUES, M_T_VALUES, R_VALUES))

EXPECTED_TRAIN_IDS = (0, 1, 2, 4, 5, 6, 7, 10, 12, 13, 14, 18, 20, 21, 24, 26, 27, 29)
EXPECTED_VALIDATION_IDS = (3, 9, 11, 16, 19, 23)
EXPECTED_TEST_IDS = (8, 15, 17, 22, 25, 28)
DEVELOPMENT_SEEDS = (19026, 19027, 19028, 19029, 19030)
FORBIDDEN_SEEDS = frozenset({39026, 39027, 39028, 39029, 39030})
PRIMARY_SCENARIOS = (
    "sensor_bias_5",
    "sensor_dropout",
    "load_disturbance",
    "combined_fault_load",
    "parameter_variation",
)
CONTROLLERS = (
    "B_plain_MPC",
    "C1_sensor_MPC",
    "C3_arbitration_MPC",
    "E_EKF_virtual_MPC",
)
EKF_CONTROLLER = "E_EKF_virtual_MPC"
CONTROL_STRIDE = 5

HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(RuntimeError):
    """Raised when saved EKF evidence violates the frozen contract."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)
    print(f"PASS: {message}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    check(path.is_file(), f"required file exists: {display_path(path)}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    check(path.is_file(), f"required JSON exists: {display_path(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    check(isinstance(value, dict), f"{path.name} contains one JSON object")
    return value


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    check(not missing, f"{label} contains required columns (missing={missing})")


def numeric(frame: pd.DataFrame, column: str, label: str, *, allow_missing: bool = False) -> pd.Series:
    raw = frame[column]
    values = pd.to_numeric(raw, errors="coerce")
    parse_failure = raw.notna() & values.isna()
    check(not parse_failure.any(), f"{label} contains only numeric values")
    if not allow_missing:
        check(values.notna().all(), f"{label} is non-missing")
    present = values.notna()
    check(np.isfinite(values[present].to_numpy(float)).all(), f"present {label} values are finite")
    return values.astype(float)


def integer_counts(frame: pd.DataFrame, column: str, label: str, *, allow_missing: bool = False) -> pd.Series:
    values = numeric(frame, column, label, allow_missing=allow_missing)
    present = values.notna()
    check((values[present] >= 0).all(), f"present {label} values are nonnegative")
    check(
        np.allclose(values[present].to_numpy(float), np.round(values[present].to_numpy(float)), rtol=0.0, atol=0.0),
        f"present {label} values are exact integers",
    )
    return values


def bool_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    check(normalized.isin({"true", "false", "1", "0"}).all(), f"{label} contains booleans")
    return normalized.isin({"true", "1"})


def seed_series(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    check(values.notna().all(), f"{label} seeds are numeric")
    check(np.isfinite(values.to_numpy(float)).all(), f"{label} seeds are finite")
    check(
        np.allclose(values.to_numpy(float), np.round(values.to_numpy(float)), rtol=0.0, atol=0.0),
        f"{label} seeds are integers",
    )
    return values.astype(int)


def _matrix(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    check(array.shape == (3, 3), f"{label} is exactly 3x3")
    check(np.isfinite(array).all(), f"{label} is finite")
    return array


def _json_equal(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        check(isinstance(left, Mapping) and isinstance(right, Mapping), f"{path} has matching mapping type")
        check(set(left) == set(right), f"{path} has matching keys")
        for key in left:
            _json_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) or isinstance(right, list):
        check(isinstance(left, list) and isinstance(right, list), f"{path} has matching list type")
        check(len(left) == len(right), f"{path} has matching list length")
        for index, (a, b) in enumerate(zip(left, right)):
            _json_equal(a, b, f"{path}[{index}]")
        return
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        a = float(left)
        b = float(right)
        if math.isnan(a) or math.isnan(b):
            check(math.isnan(a) and math.isnan(b), f"{path} has matching NaN state")
        elif math.isinf(a) or math.isinf(b):
            check(a == b, f"{path} has matching infinity")
        else:
            check(np.isclose(a, b, rtol=1e-12, atol=1e-12), f"{path} numeric value reproduces")
        return
    check(left == right, f"{path} value reproduces")


def verify_frozen_evidence() -> dict[str, Any]:
    record = load_json(FROZEN_EVIDENCE_PATH)
    check(
        tuple(record.get("forbidden_unused_seed_family", [])) == tuple(sorted(FORBIDDEN_SEEDS)),
        "pre-EKF lock preserves reserved unused seed family 39026-39030",
    )
    scientific = record.get("frozen_scientific_artifacts")
    supporting = record.get("supporting_provenance")
    check(isinstance(scientific, Mapping), "pre-EKF lock contains frozen scientific artifacts")
    check(isinstance(supporting, Mapping), "pre-EKF lock contains supporting provenance")

    expected_scientific = {
        "main_lstm_weights",
        "auxiliary_lstm_weights",
        "v3_arbitration_calibration",
        "v3_arbitration_config",
        "v3_275_run_matrix",
        "c4_development_matrix",
        "c4_attribution_calibration",
        "c4_go_no_go_criteria",
        "ekf_preregistration",
    }
    check(expected_scientific <= set(scientific), "pre-EKF lock contains all required frozen V3/C4/preregistration entries")
    for group_name, group in (("scientific", scientific), ("supporting", supporting)):
        for name, item in group.items():
            check(isinstance(item, Mapping), f"{group_name} hash record {name} is structured")
            relative = item.get("path")
            digest = item.get("sha256")
            check(isinstance(relative, str) and relative, f"{name} records a path")
            check(isinstance(digest, str) and HASH_RE.fullmatch(digest) is not None, f"{name} records a SHA256")
            check(sha256(PROJECT / relative) == digest, f"frozen evidence hash reproduces for {name}")

    prereg = scientific["ekf_preregistration"]
    check(prereg["path"] == "EKF_OBSERVER_PREREGISTRATION.md", "lock points to the canonical EKF preregistration")
    check(sha256(PROJECT / prereg["path"]) == prereg["sha256"], "EKF preregistration hash reproduces exactly")
    return record


def _training_scales(dataset_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(dataset_path, allow_pickle=False) as data:
        train_ids = np.asarray(data["split_run_ids_train"], dtype=int)
        validation_ids = np.asarray(data["split_run_ids_validation"], dtype=int)
        test_ids = np.asarray(data["split_run_ids_test"], dtype=int)
        check(tuple(train_ids.tolist()) == EXPECTED_TRAIN_IDS, "dataset train split IDs are unchanged")
        check(tuple(validation_ids.tolist()) == EXPECTED_VALIDATION_IDS, "dataset validation split IDs are unchanged")
        check(tuple(test_ids.tolist()) == EXPECTED_TEST_IDS, "dataset test split IDs are unchanged")
        check(np.isclose(float(data["timestep"]), 0.01, rtol=0.0, atol=1e-15), "dataset timestep remains 10 ms")
        values = np.stack(
            (
                np.asarray(data["current"], dtype=float)[train_ids].reshape(-1),
                np.asarray(data["y_true"], dtype=float)[train_ids].reshape(-1),
                np.asarray(data["load_torque"], dtype=float)[train_ids].reshape(-1),
            ),
            axis=1,
        )
        scales = values.std(axis=0, dtype=np.float64)
        trace_source = {
            "time": np.asarray(data["time"], dtype=float).copy(),
            "y_true": np.asarray(data["y_true"], dtype=float).copy(),
            "load_torque": np.asarray(data["load_torque"], dtype=float).copy(),
        }
    check(scales.shape == (3,) and np.isfinite(scales).all() and (scales > 0).all(), "training-only state scales are valid")
    return scales, trace_source


def _provenance_path_fields(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    def walk(node: Any, prefix: str = "") -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if "path" in str(key).lower() and isinstance(child, str):
                    found.append((path, child))
                walk(child, path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{prefix}[{index}]")
    walk(value)
    return found


def verify_no_fault_result_provenance(provenance: Mapping[str, Any]) -> None:
    allowed_path_keys = {
        "dataset_path",
        "preregistration_path",
        "observer_source_path",
        "motor_model_source_path",
        "calibration_script_path",
        "main_lstm_weights_path",
        "main_lstm_config_path",
        "auxiliary_lstm_weights_path",
        "auxiliary_lstm_config_path",
        "candidate_table_path",
        "selected_validation_trace_path",
    }
    path_fields = _provenance_path_fields(provenance)
    for dotted_key, raw_path in path_fields:
        leaf_key = dotted_key.split(".")[-1]
        check(leaf_key in allowed_path_keys, f"calibration provenance path field is an approved pre-fault source: {dotted_key}")
        lowered = raw_path.replace("\\", "/").lower()
        forbidden_tokens = ("closed_loop", "fault", "final_holdout", "c4_development", "c4_final")
        check(not any(token in lowered for token in forbidden_tokens), f"calibration provenance path contains no fault/result source: {raw_path}")

    # Descriptive exclusions are allowed only under this declared list.  Any
    # other provenance field carrying scenario/fault outcome data fails closed.
    for key, child in provenance.items():
        lowered = str(key).lower()
        if key == "forbidden_selection_inputs":
            continue
        check(
            not any(token in lowered for token in ("fault_result", "fault_metric", "scenario_result", "closed_loop_result")),
            f"calibration provenance contains no fault/scenario result field: {key}",
        )


def _selected_candidate(table: pd.DataFrame) -> pd.Series:
    stable_mask = bool_series(table["stable"], "calibration stable flag")
    stable = table.loc[stable_mask].copy()
    check(not stable.empty, "at least one preregistered calibration candidate is stable")
    for column in ("validation_rmse", "validation_mae", "validation_abs_bias"):
        stable[column] = numeric(stable, column, f"stable candidate {column}")
    best_rmse = float(stable["validation_rmse"].min())
    within = stable.loc[stable["validation_rmse"] <= best_rmse * 1.01 + 1e-15]
    return within.sort_values(
        ["validation_mae", "validation_abs_bias", "validation_rmse", "candidate_id"],
        kind="mergesort",
    ).iloc[0]


def verify_calibration(frozen: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    config = load_json(EKF_CONFIG_PATH)
    check("frozen" in str(config.get("status", "")).lower(), "EKF config is explicitly frozen")
    check(config.get("state_order") == ["current", "omega", "load_torque"], "frozen EKF state order matches preregistration")
    check(config.get("online_inputs") == ["voltage", "current"], "frozen EKF online inputs are voltage/current only")
    check(config.get("current_measurement_noise_injected") is False, "primary calibration injects no current measurement noise")
    check(np.isclose(float(config.get("dt_s")), 0.01, rtol=0.0, atol=1e-15), "frozen EKF timestep is 10 ms")

    grid = config.get("grid")
    check(isinstance(grid, Mapping), "frozen EKF config records the calibration grid")
    check(tuple(float(v) for v in grid.get("q_dyn", [])) == Q_DYN_VALUES, "frozen q_dyn grid is exact")
    check(tuple(float(v) for v in grid.get("m_T", [])) == M_T_VALUES, "frozen m_T grid is exact")
    check(tuple(float(v) for v in grid.get("r", [])) == R_VALUES, "frozen r grid is exact")
    check(int(grid.get("candidate_count", -1)) == 27, "frozen grid declares exactly 27 candidates")

    scientific = frozen["frozen_scientific_artifacts"]
    supporting = frozen["supporting_provenance"]
    dataset_path = PROJECT / supporting["dataset"]["path"]
    scales, trace_source = _training_scales(dataset_path)
    expected_p0 = np.diag(np.square(scales))
    saved_scales = config.get("training_only_state_std")
    check(isinstance(saved_scales, Mapping), "frozen config records training-only state scales")
    observed_scales = np.array(
        [saved_scales.get("current_A"), saved_scales.get("speed_rad_s"), saved_scales.get("load_torque_Nm")],
        dtype=float,
    )
    check(np.array_equal(observed_scales, scales), "frozen training-only state scales reproduce exactly")
    check(np.array_equal(_matrix(config.get("P0"), "frozen P0"), expected_p0), "frozen P0 is exactly training-only diagonal variance")

    candidates = pd.read_csv(CALIBRATION_CANDIDATES_PATH, float_precision="round_trip")
    required = [
        "candidate_id", "q_dyn", "m_T", "r", "q_current", "q_speed", "q_load", "R",
        "stable", "validation_rmse", "validation_mae", "validation_bias", "validation_abs_bias",
        "validation_sample_count", "numerical_failure_count", "nonfinite_estimate_count",
        "failure_run_count", "selected",
    ]
    require_columns(candidates, required, "EKF calibration candidate table")
    check(len(candidates) == 27, "calibration candidate table has exactly 27 rows")
    tuples = set(
        zip(
            numeric(candidates, "q_dyn", "q_dyn"),
            numeric(candidates, "m_T", "m_T"),
            numeric(candidates, "r", "r"),
        )
    )
    check(tuples == EXPECTED_GRID, "candidate table contains the exact 3x3x3 preregistered grid")
    check(candidates[["q_dyn", "m_T", "r"]].duplicated().sum() == 0, "candidate table contains no duplicate covariance tuple")
    check(list(candidates["candidate_id"].astype(str)) == [f"EKF_{index:02d}" for index in range(1, 28)], "candidate IDs preserve deterministic grid order")

    variances = np.square(scales)
    for _, row in candidates.iterrows():
        q_dyn, m_t, r_value = float(row.q_dyn), float(row.m_T), float(row.r)
        expected = (
            variances[0] * q_dyn,
            variances[1] * q_dyn,
            variances[2] * q_dyn * m_t,
            variances[0] * r_value,
        )
        observed = tuple(float(row[name]) for name in ("q_current", "q_speed", "q_load", "R"))
        check(np.allclose(observed, expected, rtol=1e-14, atol=0.0), f"candidate {row.candidate_id} Q/R values reproduce from training scales")

    for column in ("validation_sample_count", "numerical_failure_count", "nonfinite_estimate_count", "failure_run_count"):
        integer_counts(candidates, column, f"candidate {column}")
    stable = bool_series(candidates["stable"], "candidate stable flag")
    selected_flags = bool_series(candidates["selected"], "candidate selected flag")
    check(int(selected_flags.sum()) == 1, "candidate table marks exactly one selected row")
    zero_failures = (
        (candidates["numerical_failure_count"].astype(float) == 0)
        & (candidates["nonfinite_estimate_count"].astype(float) == 0)
        & (candidates["failure_run_count"].astype(float) == 0)
    )
    check((stable == zero_failures).all(), "candidate stable flag exactly reflects numerical/nonfinite failure evidence")

    chosen = _selected_candidate(candidates)
    selected_row = candidates.loc[selected_flags].iloc[0]
    check(str(selected_row.candidate_id) == str(chosen.candidate_id), "selected row exactly follows preregistered lexicographic rule")

    selected = config.get("selected_candidate")
    check(isinstance(selected, Mapping), "frozen EKF config contains one selected_candidate block")
    for name in ("candidate_id", "q_dyn", "m_T", "r"):
        expected = selected_row[name]
        actual = selected.get(name)
        check(str(actual) == str(expected) if name == "candidate_id" else float(actual) == float(expected), f"selected config {name} matches candidate table")
    expected_q = np.diag([float(selected_row.q_current), float(selected_row.q_speed), float(selected_row.q_load)])
    check(np.array_equal(_matrix(selected.get("Q"), "selected Q"), expected_q), "selected Q exactly matches chosen candidate")
    check(float(selected.get("R")) == float(selected_row.R), "selected R exactly matches chosen candidate")
    check(np.array_equal(_matrix(selected.get("P0"), "selected P0"), expected_p0), "selected P0 exactly matches training-only P0")
    for name in (
        "validation_rmse", "validation_mae", "validation_bias", "validation_abs_bias",
        "validation_sample_count", "numerical_failure_count", "nonfinite_estimate_count", "failure_run_count",
    ):
        check(np.isclose(float(selected.get(name)), float(selected_row[name]), rtol=0.0, atol=0.0), f"selected config {name} exactly matches chosen row")

    provenance = config.get("provenance")
    check(isinstance(provenance, Mapping), "frozen EKF config records calibration provenance")
    check(selected.get("provenance") == provenance, "selected candidate and top-level config share identical provenance")
    verify_no_fault_result_provenance(provenance)
    check(tuple(provenance.get("train_run_ids", [])) == EXPECTED_TRAIN_IDS, "calibration provenance records exact training population")
    check(tuple(provenance.get("validation_run_ids", [])) == EXPECTED_VALIDATION_IDS, "calibration provenance records exact validation selection population")
    check(tuple(provenance.get("excluded_test_run_ids", [])) == EXPECTED_TEST_IDS, "calibration provenance records excluded test trajectories")
    check(provenance.get("selection_population") == "saved validation trajectories only", "Q/R selection population is validation-only")
    check(provenance.get("P0_population") == "saved training trajectories only", "P0 population is training-only")
    check(provenance.get("preregistration_sha256") == scientific["ekf_preregistration"]["sha256"], "calibration provenance binds exact preregistration")
    check(provenance.get("dataset_sha256") == supporting["dataset"]["sha256"], "calibration provenance binds exact dataset")
    check(provenance.get("main_lstm_weights_sha256") == scientific["main_lstm_weights"]["sha256"], "calibration provenance binds frozen main LSTM")
    check(provenance.get("auxiliary_lstm_weights_sha256") == scientific["auxiliary_lstm_weights"]["sha256"], "calibration provenance binds frozen auxiliary LSTM")
    check(provenance.get("motor_model_source_sha256") == supporting["motor_model"]["sha256"], "calibration provenance binds frozen motor model")
    check(provenance.get("candidate_table_sha256") == sha256(CALIBRATION_CANDIDATES_PATH), "frozen config binds exact candidate CSV")
    check(provenance.get("selected_validation_trace_sha256") == sha256(CALIBRATION_TRACE_PATH), "frozen config binds exact selected validation trace")

    trace = pd.read_csv(CALIBRATION_TRACE_PATH, float_precision="round_trip")
    required_trace = [
        "run_id", "sample", "time_s", "speed_estimate_rad_s", "true_speed_rad_s",
        "load_estimate_Nm", "true_load_Nm", "innovation_A", "innovation_variance_A2",
        "nis", "min_covariance_eigenvalue",
    ]
    require_columns(trace, required_trace, "selected calibration trace")
    run_ids = seed_series(trace["run_id"], "selected calibration trace run_id")
    check(set(run_ids) == set(EXPECTED_VALIDATION_IDS), "selected trace contains exactly validation trajectories")
    time = trace_source["time"]
    for run_id in EXPECTED_VALIDATION_IDS:
        block = trace.loc[run_ids == run_id].copy()
        samples = seed_series(block["sample"], f"selected trace run {run_id} sample")
        check(list(samples) == list(range(len(time))), f"selected trace run {run_id} has every sample exactly once in order")
        check(np.array_equal(numeric(block, "time_s", f"selected trace run {run_id} time").to_numpy(), time), f"selected trace run {run_id} times reproduce dataset")
        check(np.array_equal(numeric(block, "true_speed_rad_s", f"selected trace run {run_id} truth").to_numpy(), trace_source["y_true"][run_id]), f"selected trace run {run_id} true speed reproduces dataset")
        check(np.array_equal(numeric(block, "true_load_Nm", f"selected trace run {run_id} load truth").to_numpy(), trace_source["load_torque"][run_id]), f"selected trace run {run_id} true load reproduces dataset")
        numeric(block, "speed_estimate_rad_s", f"selected trace run {run_id} speed estimate")
        numeric(block, "load_estimate_Nm", f"selected trace run {run_id} load estimate")
        diagnostics = block.iloc[1:]
        innovation = numeric(diagnostics, "innovation_A", f"selected trace run {run_id} innovation")
        innovation_variance = numeric(diagnostics, "innovation_variance_A2", f"selected trace run {run_id} innovation variance")
        nis = numeric(diagnostics, "nis", f"selected trace run {run_id} NIS")
        min_eigenvalue = numeric(diagnostics, "min_covariance_eigenvalue", f"selected trace run {run_id} covariance eigenvalue")
        check((innovation_variance > 0).all(), f"selected trace run {run_id} innovation variance stays positive")
        check((nis >= 0).all(), f"selected trace run {run_id} NIS stays nonnegative")
        check((min_eigenvalue >= -1e-10).all(), f"selected trace run {run_id} covariance remains PSD within tolerance")
        check(np.allclose(nis, np.square(innovation) / innovation_variance, rtol=1e-12, atol=1e-12), f"selected trace run {run_id} NIS recomputes")
    return config, candidates


def _estimator_metric_keys(extra_stratum: str | None = None) -> set[tuple[str, str]]:
    strata = {
        "all_paired",
        "startup_post_window_to_0p5s",
        "load_change",
        "low_current",
        "high_current",
        "low_speed",
        "high_speed",
        "low_acceleration",
        "high_acceleration",
        "voltage_saturation",
        "weak_observability",
    }
    keys = {(estimator, stratum) for estimator in ("EKF", "auxiliary_LSTM") for stratum in strata}
    keys.add(("EKF", "startup_pre_lstm_window_0_0p2s"))
    if extra_stratum is not None:
        keys |= {(estimator, extra_stratum) for estimator in ("EKF", "auxiliary_LSTM")}
    return keys


def _estimator_masks(block: pd.DataFrame, observability: Mapping[str, Any]) -> dict[str, np.ndarray]:
    thresholds = observability["normalization_for_conditioning"]["region_thresholds"]
    time_s = pd.to_numeric(block["time_s"], errors="raise").to_numpy(float)
    current = pd.to_numeric(block["current_A"], errors="raise").to_numpy(float)
    speed = pd.to_numeric(block["true_speed_rad_s"], errors="raise").to_numpy(float)
    load = pd.to_numeric(block["true_load_Nm"], errors="raise").to_numpy(float)
    voltage = pd.to_numeric(block["voltage_V"], errors="raise").to_numpy(float)
    weak = bool_series(block["weak_observability"], "estimator weak-observability trace").to_numpy(bool)
    sample = seed_series(block["sample"], "estimator trace sample").to_numpy(int)
    check(np.array_equal(sample, np.arange(len(block))), "each estimator trace run has contiguous sample indices")
    check(np.allclose(np.diff(time_s), 0.01, rtol=0.0, atol=1e-6), "each estimator trace run uses the saved approximately 10 ms grid")

    abs_current = np.abs(current)
    abs_speed = np.abs(speed)
    abs_acceleration = np.abs(np.gradient(speed, 0.01))
    changes = np.flatnonzero(np.abs(np.diff(load, prepend=load[0])) > 1e-12)
    radius = int(round(float(thresholds["load_change_neighborhood_s"]) / 0.01))
    load_change = np.zeros(len(load), dtype=bool)
    for index in changes:
        load_change[max(0, index - radius) : min(len(load), index + radius + 1)] = True
    paired = sample >= 20
    return {
        "all_paired": paired,
        "startup_pre_lstm_window_0_0p2s": sample < 20,
        "startup_post_window_to_0p5s": paired & (time_s < 0.5),
        "load_change": paired & load_change,
        "low_current": paired & (abs_current <= float(thresholds["low_current_abs_q25_A"])),
        "high_current": paired & (abs_current >= float(thresholds["high_current_abs_q75_A"])),
        "low_speed": paired & (abs_speed <= float(thresholds["low_speed_abs_q25_rad_s"])),
        "high_speed": paired & (abs_speed >= float(thresholds["high_speed_abs_q75_rad_s"])),
        "low_acceleration": paired & (abs_acceleration <= float(thresholds["steady_acceleration_abs_q25_rad_s2"])),
        "high_acceleration": paired & (abs_acceleration >= float(thresholds["high_acceleration_abs_q75_rad_s2"])),
        "voltage_saturation": paired & (
            (voltage <= float(thresholds["voltage_saturation_low_V"]))
            | (voltage >= float(thresholds["voltage_saturation_high_V"]))
        ),
        "weak_observability": paired & weak,
    }


def _recompute_estimator_metric(
    traces: pd.DataFrame,
    runtimes: pd.DataFrame,
    population: str,
    estimator: str,
    stratum: str,
    observability: Mapping[str, Any],
) -> dict[str, float | int]:
    population_trace = traces[traces.population == population].copy()
    mask_parts: list[np.ndarray] = []
    estimate_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    run_id_parts: list[np.ndarray] = []
    runtime_parts: list[np.ndarray] = []
    failure_runs = 0
    for run_id in EXPECTED_TEST_IDS:
        block = population_trace[population_trace.run_id.astype(int) == run_id].sort_values("sample")
        check(not block.empty, f"estimator trace covers {population} test run {run_id}")
        masks = _estimator_masks(block, observability)
        if population == "parameter_variation_fixed_nominal_observer":
            masks["parameter_shift_post_3s"] = (
                seed_series(block["sample"], "parameter-transfer sample").to_numpy(int) >= 20
            ) & (pd.to_numeric(block["time_s"], errors="raise").to_numpy(float) >= 3.0)
        elif population == "load_disturbance_fixed_nominal_observer":
            masks["load_step_post_3s"] = (
                seed_series(block["sample"], "load-transfer sample").to_numpy(int) >= 20
            ) & (pd.to_numeric(block["time_s"], errors="raise").to_numpy(float) >= 3.0)
        mask = masks[stratum]
        estimate_column = "ekf_speed_rad_s" if estimator == "EKF" else "aux_speed_rad_s"
        estimate = pd.to_numeric(block[estimate_column], errors="coerce").to_numpy(float)
        truth = pd.to_numeric(block["true_speed_rad_s"], errors="coerce").to_numpy(float)
        mask_parts.append(mask)
        estimate_parts.append(estimate)
        truth_parts.append(truth)
        run_id_parts.append(np.full(len(block), run_id, dtype=int))
        failure_column = "ekf_failure_sample" if estimator == "EKF" else "aux_failure_sample"
        failure_values = pd.to_numeric(block[failure_column], errors="raise").astype(int)
        check(failure_values.nunique() == 1, f"{population}/{run_id}/{estimator} records one consistent failure marker")
        failure_runs += int(int(failure_values.iloc[0]) >= 0)

        runtime_block = runtimes[
            (runtimes.population == population)
            & (runtimes.run_id.astype(int) == run_id)
            & (runtimes.estimator == estimator)
        ]
        runtime_by_sample = dict(
            zip(
                runtime_block["sample"].astype(int),
                pd.to_numeric(runtime_block["update_runtime_us"], errors="raise").astype(float),
            )
        )
        runtime = np.full(len(block), np.nan, dtype=float)
        runtime_indices = np.fromiter(runtime_by_sample.keys(), dtype=int)
        check(
            runtime_indices.size == 0
            or bool(((runtime_indices >= 0) & (runtime_indices < len(block))).all()),
            f"runtime sample indices are in range for {population}/{run_id}/{estimator}",
        )
        for sample_index, value in runtime_by_sample.items():
            runtime[sample_index] = value
        runtime_parts.append(runtime)

    selected = np.concatenate(mask_parts)
    estimate = np.concatenate(estimate_parts)
    truth = np.concatenate(truth_parts)
    sample_run_ids = np.concatenate(run_id_parts)
    runtime = np.concatenate(runtime_parts)
    finite = selected & np.isfinite(estimate) & np.isfinite(truth)
    error = estimate[finite] - truth[finite]
    runtime_values = runtime[finite & np.isfinite(runtime)]
    selected_count = int(selected.sum())
    finite_count = int(finite.sum())
    return {
        "sample_count": selected_count,
        "trajectory_count": int(np.unique(sample_run_ids[selected]).size) if selected_count else 0,
        "finite_sample_count": finite_count,
        "nonfinite_sample_count": selected_count - finite_count,
        "finite_fraction": float(finite_count / selected_count) if selected_count else np.nan,
        "rmse_rad_s": float(np.sqrt(np.mean(np.square(error)))) if error.size else np.nan,
        "mae_rad_s": float(np.mean(np.abs(error))) if error.size else np.nan,
        "bias_rad_s": float(np.mean(error)) if error.size else np.nan,
        "numerical_failure_count": failure_runs,
        "failure_run_count": failure_runs,
        "runtime_sample_count": int(runtime_values.size),
        "runtime_mean_us": float(np.mean(runtime_values)) if runtime_values.size else np.nan,
        "runtime_p50_us": float(np.quantile(runtime_values, 0.50)) if runtime_values.size else np.nan,
        "runtime_p95_us": float(np.quantile(runtime_values, 0.95)) if runtime_values.size else np.nan,
        "runtime_p99_us": float(np.quantile(runtime_values, 0.99)) if runtime_values.size else np.nan,
        "runtime_max_us": float(np.max(runtime_values)) if runtime_values.size else np.nan,
    }


def verify_estimator_csvs(config_hash: str, frozen: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (ESTIMATOR_COMPARISON_PATH, ESTIMATOR_TRANSFER_PATH, ESTIMATOR_RUNTIME_PATH, ESTIMATOR_TRACE_PATH):
        check(path.is_file(), f"required estimator artifact exists: {display_path(path)}")
    comparison = pd.read_csv(ESTIMATOR_COMPARISON_PATH, float_precision="round_trip")
    transfer = pd.read_csv(ESTIMATOR_TRANSFER_PATH, float_precision="round_trip")
    runtimes = pd.read_csv(ESTIMATOR_RUNTIME_PATH, float_precision="round_trip")
    traces = pd.read_csv(ESTIMATOR_TRACE_PATH, float_precision="round_trip")

    metric_columns = [
        "population", "estimator", "stratum", "sample_count", "trajectory_count",
        "finite_sample_count", "nonfinite_sample_count", "finite_fraction", "rmse_rad_s",
        "mae_rad_s", "bias_rad_s", "numerical_failure_count", "failure_run_count",
        "runtime_sample_count", "runtime_mean_us", "runtime_p50_us", "runtime_p95_us",
        "runtime_p99_us", "runtime_max_us", "ekf_config_sha256", "dataset_sha256",
        "auxiliary_lstm_weights_sha256", "auxiliary_lstm_config_sha256",
        "observer_source_sha256", "motor_model_source_sha256", "observability_summary_sha256",
        "evaluation_script_sha256", "selected_candidate_id",
    ]
    require_columns(comparison, metric_columns, "clean estimator comparison")
    require_columns(transfer, metric_columns, "transfer estimator comparison")
    check(len(comparison) == 23, "clean estimator comparison contains exactly 23 estimator/stratum rows")
    check(set(comparison.population.astype(str)) == {"clean_heldout_test"}, "clean estimator comparison has the exact population label")
    check(
        set(zip(comparison.estimator.astype(str), comparison.stratum.astype(str))) == _estimator_metric_keys(),
        "clean estimator comparison contains the exact estimator/stratum coverage",
    )
    check(not comparison.duplicated(["population", "estimator", "stratum"]).any(), "clean estimator comparison contains no duplicate keys")

    transfer_design = {
        "load_disturbance_fixed_nominal_observer": "load_step_post_3s",
        "parameter_variation_fixed_nominal_observer": "parameter_shift_post_3s",
    }
    check(set(transfer.population.astype(str)) == set(transfer_design), "transfer comparison contains exactly load-disturbance and parameter-variation populations")
    check(len(transfer) == 50, "transfer comparison contains exactly 25 rows per preregistered transfer population")
    check(not transfer.duplicated(["population", "estimator", "stratum"]).any(), "transfer comparison contains no duplicate population/estimator/stratum keys")
    for population, extra_stratum in transfer_design.items():
        block = transfer[transfer.population == population]
        check(len(block) == 25, f"{population} contains exactly 25 estimator/stratum rows")
        check(
            set(zip(block.estimator.astype(str), block.stratum.astype(str))) == _estimator_metric_keys(extra_stratum),
            f"{population} contains exact base strata plus {extra_stratum}",
        )

    require_columns(runtimes, ["population", "run_id", "sample", "time_s", "estimator", "update_runtime_us"], "estimator runtime samples")
    require_columns(
        traces,
        [
            "population", "run_id", "sample", "time_s", "voltage_V", "current_A", "true_speed_rad_s",
            "true_load_Nm", "ekf_speed_rad_s", "ekf_load_Nm", "aux_speed_rad_s", "ekf_innovation_A",
            "ekf_innovation_variance_A2", "ekf_nis", "weak_observability", "after_parameter_shift",
            "after_load_step", "ekf_failure_sample", "aux_failure_sample",
        ],
        "estimator trace",
    )
    expected_populations = {"clean_heldout_test", *transfer_design.keys()}
    check(set(traces.population.astype(str)) == expected_populations, "estimator trace contains exactly clean, load-transfer, and parameter-transfer populations")
    check(set(runtimes.population.astype(str)) == expected_populations, "estimator runtime contains exactly clean, load-transfer, and parameter-transfer populations")
    trace_run_ids = seed_series(traces["run_id"], "estimator trace run_id")
    check(set(trace_run_ids) == set(EXPECTED_TEST_IDS), "estimator trace uses exactly held-out test trajectories")
    runtime_run_ids = seed_series(runtimes["run_id"], "estimator runtime run_id")
    check(set(runtime_run_ids) == set(EXPECTED_TEST_IDS), "estimator runtime uses exactly held-out test trajectories")
    check(set(runtimes.estimator.astype(str)) == {"EKF", "auxiliary_LSTM"}, "estimator runtime covers exactly EKF and auxiliary LSTM")
    runtime_values = numeric(runtimes, "update_runtime_us", "estimator update runtime")
    check((runtime_values >= 0).all(), "estimator update runtimes are nonnegative")
    check(not runtimes.duplicated(["population", "run_id", "sample", "estimator"]).any(), "estimator runtime samples have unique population/run/sample/estimator keys")
    check(not traces.duplicated(["population", "run_id", "sample"]).any(), "estimator trace samples have unique population/run/sample keys")

    dataset_path = PROJECT / frozen["supporting_provenance"]["dataset"]["path"]
    with np.load(dataset_path, allow_pickle=False) as data:
        saved_time = np.asarray(data["time"], dtype=float)
    for population in expected_populations:
        for run_id in EXPECTED_TEST_IDS:
            block = traces[(traces.population == population) & (traces.run_id.astype(int) == run_id)].sort_values("sample")
            check(len(block) == len(saved_time), f"{population} trace run {run_id} has the full saved sample grid")
            observed_time = pd.to_numeric(block["time_s"], errors="raise").to_numpy(float)
            check(np.allclose(observed_time, saved_time, rtol=0.0, atol=2e-15), f"{population} trace run {run_id} timestamps reproduce the locked dataset grid")

    scientific = frozen["frozen_scientific_artifacts"]
    supporting = frozen["supporting_provenance"]
    expected_provenance = {
        "ekf_config_sha256": config_hash,
        "dataset_sha256": supporting["dataset"]["sha256"],
        "auxiliary_lstm_weights_sha256": scientific["auxiliary_lstm_weights"]["sha256"],
        "auxiliary_lstm_config_sha256": sha256(AUX_CONFIG_PATH),
        "observer_source_sha256": sha256(PROJECT / "src" / "ekf_observer.py"),
        "motor_model_source_sha256": supporting["motor_model"]["sha256"],
        "observability_summary_sha256": sha256(OBSERVABILITY_PATH),
        "evaluation_script_sha256": sha256(ESTIMATOR_SCRIPT_PATH),
    }
    selected_id = load_json(EKF_CONFIG_PATH)["selected_candidate"]["candidate_id"]
    for frame, label in ((comparison, "clean estimator comparison"), (transfer, "transfer estimator comparison")):
        for column, expected in expected_provenance.items():
            check(set(frame[column].astype(str)) == {str(expected)}, f"{label} {column} binds current frozen source")
        check(set(frame["selected_candidate_id"].astype(str)) == {str(selected_id)}, f"{label} binds selected frozen candidate")

    observability = load_json(OBSERVABILITY_PATH)
    for population in ("clean_heldout_test", *transfer_design.keys()):
        frame = comparison if population == "clean_heldout_test" else transfer[transfer.population == population]
        for _, row in frame.iterrows():
            recomputed = _recompute_estimator_metric(
                traces,
                runtimes,
                population,
                str(row.estimator),
                str(row.stratum),
                observability,
            )
            for name, expected in recomputed.items():
                actual = row[name]
                if isinstance(expected, float) and math.isnan(expected):
                    check(pd.isna(actual), f"{population}/{row.estimator}/{row.stratum} {name} preserves empty-stratum NaN")
                elif isinstance(expected, float):
                    check(np.isclose(float(actual), expected, rtol=1e-12, atol=1e-12), f"{population}/{row.estimator}/{row.stratum} {name} recomputes")
                else:
                    check(int(actual) == int(expected), f"{population}/{row.estimator}/{row.stratum} {name} recomputes")

    # The paired comparison starts only at sample 20; startup before that is
    # reported for EKF separately and never used as auxiliary-vs-EKF evidence.
    for population in ("clean_heldout_test", *transfer_design.keys()):
        frame = comparison if population == "clean_heldout_test" else transfer[transfer.population == population]
        all_rows = frame[frame.stratum == "all_paired"]
        pre_rows = frame[frame.stratum == "startup_pre_lstm_window_0_0p2s"]
        label = population
        check(set(all_rows.estimator) == {"EKF", "auxiliary_LSTM"}, f"{label} primary paired estimator comparison contains both estimators")
        check(set(pre_rows.estimator) == {"EKF"}, f"{label} pre-window startup is reported for EKF separately")
        check((all_rows.sample_count > 0).all(), f"{label} all-paired primary estimator endpoint has scored samples")
    parameter_endpoint = transfer[
        (transfer.population == "parameter_variation_fixed_nominal_observer")
        & (transfer.stratum == "parameter_shift_post_3s")
    ]
    load_endpoint = transfer[
        (transfer.population == "load_disturbance_fixed_nominal_observer")
        & (transfer.stratum == "load_step_post_3s")
    ]
    check(set(parameter_endpoint.estimator) == {"EKF", "auxiliary_LSTM"}, "parameter-transfer primary endpoint covers both estimators after the frozen shift")
    check(set(load_endpoint.estimator) == {"EKF", "auxiliary_LSTM"}, "load-disturbance transfer endpoint covers both estimators after the frozen load step")
    return comparison, transfer


def expected_closed_loop_keys() -> set[tuple[str, str, int]]:
    return {
        (controller, scenario, seed)
        for controller in CONTROLLERS
        for scenario in PRIMARY_SCENARIOS
        for seed in DEVELOPMENT_SEEDS
    }


def verify_closed_loop_matrix(runs: pd.DataFrame) -> pd.DataFrame:
    require_columns(runs, ["controller", "scenario", "seed"], "EKF closed-loop matrix")
    runs = runs.copy()
    runs["seed"] = seed_series(runs["seed"], "EKF closed-loop matrix")
    check(len(runs) == 100, "closed-loop matrix contains exactly 4 x 5 x 5 = 100 rows")
    check(set(runs.controller.astype(str)) == set(CONTROLLERS), "closed-loop matrix contains exactly B/C1/C3/E")
    check(set(runs.scenario.astype(str)) == set(PRIMARY_SCENARIOS), "closed-loop matrix contains exactly the five preregistered primary scenarios")
    check(set(runs.seed) == set(DEVELOPMENT_SEEDS), "closed-loop matrix uses exactly development seeds 19026-19030")
    check(set(runs.seed).isdisjoint(FORBIDDEN_SEEDS), "closed-loop matrix does not use reserved seeds 39026-39030")
    keys = list(zip(runs.controller.astype(str), runs.scenario.astype(str), runs.seed.astype(int)))
    check(len(keys) == len(set(keys)), "closed-loop matrix has no duplicate controller/scenario/seed keys")
    check(set(keys) == expected_closed_loop_keys(), "closed-loop matrix has no missing or unexpected paired keys")
    return runs


def verify_ekf_run_evidence(runs: pd.DataFrame) -> None:
    ekf = runs[runs.controller == EKF_CONTROLLER].copy()
    check(len(ekf) == 25, "EKF contributes exactly 25 primary closed-loop rows")
    required = [
        "run_complete", "completed_samples", "ekf_finite_sample_rate", "ekf_numerical_failures",
        "ekf_nonfinite_failures", "ekf_psd_failures", "ekf_symmetry_failures", "ekf_failure_message",
        "voltage_violations", "rate_violations", "optimizer_failures", "nonfinite_events",
        "main_prediction_failures", "aux_prediction_failures", "fault_window_rmse", "sub_fraction", "sub_samples",
        "false_substitution_count", "false_substitution_fraction", "mean_ekf_update_ms",
        "median_ekf_update_ms", "p95_ekf_update_ms", "p99_ekf_update_ms", "max_ekf_update_ms",
        "total_reliability_observer_overhead_sample_count",
        "mean_total_reliability_observer_overhead_ms",
        "median_total_reliability_observer_overhead_ms",
        "p95_total_reliability_observer_overhead_ms",
        "p99_total_reliability_observer_overhead_ms",
        "max_total_reliability_observer_overhead_ms",
        "slsqp_solve_count", "mean_slsqp_solve_ms", "median_slsqp_solve_ms",
        "p95_slsqp_solve_ms", "p99_slsqp_solve_ms", "max_slsqp_solve_ms", "sim_time_s",
    ]
    require_columns(ekf, required, "EKF closed-loop rows")
    complete = bool_series(ekf["run_complete"], "EKF run_complete")
    completed_samples = integer_counts(ekf, "completed_samples", "EKF completed_samples")
    finite_rate = numeric(ekf, "ekf_finite_sample_rate", "EKF finite sample rate")
    check(((finite_rate >= 0) & (finite_rate <= 1)).all(), "EKF finite sample rates lie in [0,1]")
    check((completed_samples >= 0).all(), "EKF completed sample counts are nonnegative")
    for column in (
        "ekf_numerical_failures", "ekf_nonfinite_failures", "ekf_psd_failures", "ekf_symmetry_failures",
        "optimizer_failures", "nonfinite_events", "main_prediction_failures", "aux_prediction_failures",
    ):
        integer_counts(ekf, column, f"EKF {column}")
    for column in ("voltage_violations", "rate_violations"):
        integer_counts(ekf, column, f"EKF {column}", allow_missing=True)
    for column in (
        "total_reliability_observer_overhead_sample_count",
        "slsqp_solve_count",
    ):
        counts = integer_counts(ekf, column, f"EKF {column}")
        check((counts >= 0).all(), f"EKF {column} is nonnegative")
    for column in (
        "mean_ekf_update_ms", "median_ekf_update_ms", "p95_ekf_update_ms", "p99_ekf_update_ms",
        "max_ekf_update_ms", "mean_total_reliability_observer_overhead_ms",
        "median_total_reliability_observer_overhead_ms", "p95_total_reliability_observer_overhead_ms",
        "p99_total_reliability_observer_overhead_ms", "max_total_reliability_observer_overhead_ms",
        "mean_slsqp_solve_ms", "median_slsqp_solve_ms", "p95_slsqp_solve_ms",
        "p99_slsqp_solve_ms", "max_slsqp_solve_ms", "sim_time_s",
    ):
        values = numeric(ekf, column, f"EKF {column}")
        check((values >= 0).all(), f"EKF {column} is nonnegative")

    check(
        (
            (ekf["median_ekf_update_ms"] <= ekf["p95_ekf_update_ms"])
            & (ekf["p95_ekf_update_ms"] <= ekf["p99_ekf_update_ms"])
            & (ekf["p99_ekf_update_ms"] <= ekf["max_ekf_update_ms"])
        ).all(),
        "EKF update median/p95/p99/max ordering is valid",
    )
    check(
        (
            (ekf["median_total_reliability_observer_overhead_ms"] <= ekf["p95_total_reliability_observer_overhead_ms"])
            & (ekf["p95_total_reliability_observer_overhead_ms"] <= ekf["p99_total_reliability_observer_overhead_ms"])
            & (ekf["p99_total_reliability_observer_overhead_ms"] <= ekf["max_total_reliability_observer_overhead_ms"])
        ).all(),
        "total reliability+observer overhead median/p95/p99/max ordering is valid",
    )
    check(
        (
            (ekf["median_slsqp_solve_ms"] <= ekf["p95_slsqp_solve_ms"])
            & (ekf["p95_slsqp_solve_ms"] <= ekf["p99_slsqp_solve_ms"])
            & (ekf["p99_slsqp_solve_ms"] <= ekf["max_slsqp_solve_ms"])
        ).all(),
        "SLSQP median/p95/p99/max ordering is valid",
    )

    if complete.any():
        successful = ekf.loc[complete]
        for column in ("fault_window_rmse", "sub_fraction", "sub_samples", "false_substitution_count", "false_substitution_fraction"):
            values = numeric(successful, column, f"complete EKF {column}")
            check((values >= 0).all(), f"complete EKF {column} is nonnegative")
        check((numeric(successful, "ekf_finite_sample_rate", "complete EKF finite rate") == 1.0).all(), "completed EKF runs are finite at every sample")
    incomplete = ekf.loc[~complete]
    if not incomplete.empty:
        check(incomplete["ekf_failure_message"].fillna("").astype(str).str.len().gt(0).all(), "every incomplete EKF run records an explicit failure message")


def _paired(runs: pd.DataFrame, scenario: str, metric: str) -> pd.DataFrame:
    return runs[runs.scenario == scenario].pivot(index="seed", columns="controller", values=metric).sort_index()


def _timing_distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    check(len(array) > 0, "timing distribution contains at least one finite sample")
    check((array >= 0).all(), "timing distribution samples are nonnegative")
    return {
        "sample_count": int(len(array)),
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "max_ms": float(np.max(array)),
    }


def recompute_runtime_overhead_summary(runs: pd.DataFrame) -> dict[str, Any]:
    """Independently reproduce the evaluator's scenario-level runtime summary."""
    summary: dict[str, Any] = {}
    for scenario in PRIMARY_SCENARIOS:
        subset = runs[runs["scenario"] == scenario]
        ekf_rows = subset[subset["controller"] == EKF_CONTROLLER]
        c3_rows = subset[subset["controller"] == "C3_arbitration_MPC"]
        check(len(ekf_rows) == 5 and len(c3_rows) == 5, f"runtime summary has five paired E/C3 rows for {scenario}")
        ekf_sim = float(pd.to_numeric(ekf_rows["sim_time_s"]).mean())
        c3_sim = float(pd.to_numeric(c3_rows["sim_time_s"]).mean())
        delta = ekf_sim - c3_sim
        summary[scenario] = {
            "mean_ekf_sim_time_s": ekf_sim,
            "mean_c3_sim_time_s": c3_sim,
            "mean_overhead_s": delta,
            "mean_overhead_fraction_vs_c3": delta / c3_sim if c3_sim > 0 else np.nan,
            "mean_ekf_update_ms": float(pd.to_numeric(ekf_rows["mean_ekf_update_ms"]).mean()),
            "p95_ekf_update_ms_across_seed_summaries": float(pd.to_numeric(ekf_rows["p95_ekf_update_ms"]).mean()),
            "total_reliability_observer_overhead_ms": {
                "mean_of_run_means": float(pd.to_numeric(ekf_rows["mean_total_reliability_observer_overhead_ms"]).mean()),
                "mean_of_run_medians": float(pd.to_numeric(ekf_rows["median_total_reliability_observer_overhead_ms"]).mean()),
                "mean_of_run_p95s": float(pd.to_numeric(ekf_rows["p95_total_reliability_observer_overhead_ms"]).mean()),
                "mean_of_run_p99s": float(pd.to_numeric(ekf_rows["p99_total_reliability_observer_overhead_ms"]).mean()),
                "max_across_runs": float(pd.to_numeric(ekf_rows["max_total_reliability_observer_overhead_ms"]).max()),
            },
            "slsqp_solve_ms": {
                "mean_of_run_means": float(pd.to_numeric(ekf_rows["mean_slsqp_solve_ms"]).mean()),
                "mean_of_run_medians": float(pd.to_numeric(ekf_rows["median_slsqp_solve_ms"]).mean()),
                "mean_of_run_p95s": float(pd.to_numeric(ekf_rows["p95_slsqp_solve_ms"]).mean()),
                "mean_of_run_p99s": float(pd.to_numeric(ekf_rows["p99_slsqp_solve_ms"]).mean()),
                "max_across_runs": float(pd.to_numeric(ekf_rows["max_slsqp_solve_ms"]).max()),
            },
        }
    return summary


def recompute_role_gate(runs: pd.DataFrame, calibration_locked: bool = True) -> dict[str, Any]:
    """Independently reproduce the evaluator's preregistered role-gate arithmetic."""
    ekf = runs[runs.controller == EKF_CONTROLLER]
    complete = bool(
        len(ekf) == 25
        and bool_series(ekf["run_complete"], "role-gate EKF run_complete").all()
        and np.allclose(pd.to_numeric(ekf["ekf_finite_sample_rate"]), 1.0, rtol=0.0, atol=0.0)
        and int(pd.to_numeric(ekf["ekf_numerical_failures"]).fillna(0).sum()) == 0
    )
    safety_columns = [
        "voltage_violations", "rate_violations", "optimizer_failures", "nonfinite_events",
        "main_prediction_failures", "aux_prediction_failures",
    ]
    safety_failures = int(sum(float(pd.to_numeric(ekf[column], errors="coerce").fillna(0).sum()) for column in safety_columns))
    safety_pass = safety_failures == 0

    isolated: dict[str, Any] = {}
    isolated_pass = True
    for scenario in ("sensor_bias_5", "sensor_dropout"):
        paired = _paired(runs, scenario, "fault_window_rmse")
        ratio = paired[EKF_CONTROLLER] / paired["C3_arbitration_MPC"] - 1.0
        e_mean = float(runs[(runs.scenario == scenario) & (runs.controller == EKF_CONTROLLER)].fault_window_rmse.mean())
        c3_mean = float(runs[(runs.scenario == scenario) & (runs.controller == "C3_arbitration_MPC")].fault_window_rmse.mean())
        mean_ratio = e_mean / c3_mean - 1.0
        seed_passes = int(np.sum(ratio <= 0.05 + 1e-12))
        passed = bool(mean_ratio <= 0.05 + 1e-12 and seed_passes >= 4)
        isolated[scenario] = {"mean_relative_degradation": mean_ratio, "paired_seed_passes": seed_passes, "pass": passed}
        isolated_pass &= passed

    combined = _paired(runs, "combined_fault_load", "fault_window_rmse")
    combined_improvement = (combined["C3_arbitration_MPC"] - combined[EKF_CONTROLLER]) / combined["C3_arbitration_MPC"]
    combined_c3_mean = float(runs[(runs.scenario == "combined_fault_load") & (runs.controller == "C3_arbitration_MPC")].fault_window_rmse.mean())
    combined_e_mean = float(runs[(runs.scenario == "combined_fault_load") & (runs.controller == EKF_CONTROLLER)].fault_window_rmse.mean())
    combined_mean_improvement = (combined_c3_mean - combined_e_mean) / combined_c3_mean
    combined_seed_passes = int(np.sum(combined_improvement >= 0.10 - 1e-12))
    combined_pass = bool(combined_mean_improvement >= 0.10 - 1e-12 and combined_seed_passes >= 4)

    load = _paired(runs, "load_disturbance", "fault_window_rmse")
    plain_penalty = load["B_plain_MPC"]
    c3_penalty = load["C3_arbitration_MPC"] - plain_penalty
    e_penalty = load[EKF_CONTROLLER] - plain_penalty
    paired_reduction = np.where(c3_penalty > 0, (c3_penalty - e_penalty) / c3_penalty, np.where(e_penalty <= c3_penalty, 1.0, -np.inf))
    load_seed_passes = int(np.sum(paired_reduction >= 0.10 - 1e-12))
    load_rows = runs[runs.scenario == "load_disturbance"]
    b_mean = float(load_rows[load_rows.controller == "B_plain_MPC"].fault_window_rmse.mean())
    c3_mean = float(load_rows[load_rows.controller == "C3_arbitration_MPC"].fault_window_rmse.mean())
    e_mean = float(load_rows[load_rows.controller == EKF_CONTROLLER].fault_window_rmse.mean())
    c3_mean_penalty = c3_mean - b_mean
    e_mean_penalty = e_mean - b_mean
    load_mean_reduction = (
        (c3_mean_penalty - e_mean_penalty) / c3_mean_penalty
        if c3_mean_penalty > 0
        else (1.0 if e_mean_penalty <= c3_mean_penalty else -np.inf)
    )
    c3_load = load_rows[load_rows.controller == "C3_arbitration_MPC"].set_index("seed")
    e_load = load_rows[load_rows.controller == EKF_CONTROLLER].set_index("seed")
    c3_count = pd.to_numeric(c3_load["sub_samples"], errors="coerce")
    e_count = pd.to_numeric(e_load["false_substitution_count"], errors="coerce")
    c3_fraction = pd.to_numeric(c3_load["sub_fraction"], errors="coerce")
    e_fraction = pd.to_numeric(e_load["false_substitution_fraction"], errors="coerce")
    false_seed_passes = int(np.sum((e_count <= c3_count + 1e-12) & (e_fraction <= c3_fraction + 1e-12)))
    false_nonworse = bool(
        e_count.mean() <= c3_count.mean() + 1e-12
        and e_fraction.mean() <= c3_fraction.mean() + 1e-12
        and false_seed_passes >= 4
    )
    load_pass = bool(load_mean_reduction >= 0.10 - 1e-12 and load_seed_passes >= 4 and false_nonworse)
    all_pass = bool(complete and safety_pass and isolated_pass and combined_pass and load_pass and calibration_locked)
    return {
        "decision": "POTENTIAL_SUCCESSOR_FOLLOW_UP" if all_pass else "BASELINE_ONLY",
        "finite_every_sample": complete,
        "zero_new_safety_failures": safety_pass,
        "safety_failure_count": safety_failures,
        "isolated_sensor_faults": isolated,
        "combined_fault_load": {
            "mean_relative_improvement": float(combined_mean_improvement),
            "paired_seed_passes": combined_seed_passes,
            "pass": combined_pass,
        },
        "load_disturbance": {
            "mean_tracking_penalty_reduction": float(load_mean_reduction),
            "paired_tracking_seed_passes": load_seed_passes,
            "false_substitution_nonworse_seed_passes": false_seed_passes,
            "false_substitution_nonworse": false_nonworse,
            "pass": load_pass,
        },
        "qr_selected_from_preregistered_27_candidate_calibration": bool(calibration_locked),
        "all_successor_criteria_pass": all_pass,
    }


def verify_closed_loop_summary(
    runs: pd.DataFrame,
    summary: Mapping[str, Any],
    config_hash: str,
    plan_hash: str,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    check(summary.get("evidence_role") == "development_closed_loop_comparator", "closed-loop summary is labeled development comparator evidence")
    check(summary.get("controller") == EKF_CONTROLLER, "closed-loop summary names the EKF comparator")
    check(tuple(summary.get("development_seeds", [])) == DEVELOPMENT_SEEDS, "closed-loop summary records exact development seeds")
    check(tuple(summary.get("primary_scenarios", [])) == PRIMARY_SCENARIOS, "closed-loop summary records exact primary scenarios")
    check(int(summary.get("run_count", -1)) == 100, "closed-loop summary run_count matches 100-row paired matrix")
    check(summary.get("ekf_config_sha256") == config_hash, "closed-loop summary binds exact frozen EKF config")
    check(summary.get("closed_loop_plan_sha256") == plan_hash, "closed-loop summary binds immutable pre-run plan")
    check(
        summary.get("closed_loop_evaluator_sha256") == sha256(CLOSED_LOOP_EVALUATOR_PATH),
        "closed-loop summary binds evaluator source frozen by the pre-run plan",
    )
    check(
        summary.get("runtime_accounting")
        == {
            "per_sample_total_overhead": "ekf_update_ms + aux_inference_ms + reliability_update_ms",
            "optimizer_timing": "slsqp_solve_ms recorded separately",
            "distribution_statistics": ["mean", "median", "p95", "p99", "max"],
        },
        "closed-loop summary records the frozen runtime-accounting contract",
    )

    source_hashes = summary.get("runtime_source_hashes")
    check(isinstance(source_hashes, Mapping), "closed-loop summary records runtime source hashes")
    sci = frozen["frozen_scientific_artifacts"]
    sup = frozen["supporting_provenance"]
    expected = {
        "main_lstm_weights": sci["main_lstm_weights"]["sha256"],
        "auxiliary_lstm_weights": sci["auxiliary_lstm_weights"]["sha256"],
        "v3_arbitration_calibration": sci["v3_arbitration_calibration"]["sha256"],
        "v3_arbitration_config": sci["v3_arbitration_config"]["sha256"],
        "motor_model": sup["motor_model"]["sha256"],
        "mpc": sup["mpc"]["sha256"],
        "reliability": sup["reliability"]["sha256"],
        "v3_evaluator": sup["v3_evaluator"]["sha256"],
    }
    check(dict(source_hashes) == expected, "closed-loop runtime sources exactly match pre-EKF frozen evidence")

    saved_runtime = summary.get("runtime_overhead_vs_c3")
    check(isinstance(saved_runtime, Mapping), "closed-loop summary records separate runtime overhead evidence")
    recomputed_runtime = recompute_runtime_overhead_summary(runs)
    _json_equal(saved_runtime, recomputed_runtime, "runtime_overhead_vs_c3")

    saved_gate = summary.get("role_gate")
    check(isinstance(saved_gate, Mapping), "closed-loop summary contains saved role gate")
    recomputed = recompute_role_gate(runs, calibration_locked=True)
    _json_equal(saved_gate, recomputed, "role_gate")
    return recomputed


def verify_closed_loop_trace(runs: pd.DataFrame) -> None:
    trace = pd.read_csv(CLOSED_LOOP_TRACES_PATH, float_precision="round_trip")
    required = [
        "controller", "scenario", "seed", "step", "time", "current", "true", "ekf",
        "ekf_load", "innovation", "nis", "min_cov_eigenvalue", "cov_asymmetry",
        "ekf_update_ms", "aux_inference_ms", "reliability_update_ms",
        "total_reliability_observer_overhead_ms", "slsqp_solve_ms",
    ]
    require_columns(trace, required, "EKF closed-loop trace")
    check(set(trace.controller.astype(str)) <= {EKF_CONTROLLER}, "closed-loop trace contains only EKF comparator samples")
    trace["seed"] = seed_series(trace["seed"], "closed-loop trace")
    check(set(trace.seed).isdisjoint(FORBIDDEN_SEEDS), "closed-loop trace uses no reserved 39026-39030 seed")
    e_runs = runs[runs.controller == EKF_CONTROLLER]
    for _, run in e_runs.iterrows():
        block = trace[(trace.scenario == run.scenario) & (trace.seed == int(run.seed))]
        expected_samples = int(run.completed_samples)
        check(len(block) == expected_samples, f"trace sample count matches {run.scenario}/{int(run.seed)} completed_samples")
        if expected_samples:
            steps = seed_series(block["step"], f"trace {run.scenario}/{int(run.seed)} step")
            check(list(steps) == list(range(expected_samples)), f"trace steps are contiguous for {run.scenario}/{int(run.seed)}")
            numeric(block, "ekf", f"trace {run.scenario}/{int(run.seed)} EKF speed")
            numeric(block, "ekf_load", f"trace {run.scenario}/{int(run.seed)} EKF load")
            asymmetry = numeric(block, "cov_asymmetry", f"trace {run.scenario}/{int(run.seed)} covariance asymmetry", allow_missing=True)
            present_asymmetry = asymmetry.dropna()
            check((present_asymmetry <= 1e-10 + 1e-15).all(), f"trace covariance symmetry stays within tolerance for {run.scenario}/{int(run.seed)}")
            min_eig = numeric(block, "min_cov_eigenvalue", f"trace {run.scenario}/{int(run.seed)} covariance eigenvalue", allow_missing=True)
            check((min_eig.dropna() >= -1e-10).all(), f"trace covariance remains PSD for {run.scenario}/{int(run.seed)}")

            ekf_timing = numeric(block, "ekf_update_ms", f"trace {run.scenario}/{int(run.seed)} EKF timing")
            aux_timing = numeric(block, "aux_inference_ms", f"trace {run.scenario}/{int(run.seed)} auxiliary timing")
            reliability_timing = numeric(
                block,
                "reliability_update_ms",
                f"trace {run.scenario}/{int(run.seed)} reliability timing",
            )
            total_timing = numeric(
                block,
                "total_reliability_observer_overhead_ms",
                f"trace {run.scenario}/{int(run.seed)} total reliability+observer overhead",
            )
            slsqp_timing = numeric(
                block,
                "slsqp_solve_ms",
                f"trace {run.scenario}/{int(run.seed)} SLSQP timing",
                allow_missing=True,
            )
            for values, label in (
                (ekf_timing, "EKF update"),
                (aux_timing, "auxiliary inference"),
                (reliability_timing, "reliability update"),
                (total_timing, "total reliability+observer overhead"),
            ):
                check((values >= 0).all(), f"trace {run.scenario}/{int(run.seed)} {label} timing is nonnegative")
            check(
                (slsqp_timing.dropna() >= 0).all(),
                f"trace {run.scenario}/{int(run.seed)} SLSQP timing is nonnegative when present",
            )
            check(
                np.allclose(
                    total_timing.to_numpy(float),
                    ekf_timing.to_numpy(float) + aux_timing.to_numpy(float) + reliability_timing.to_numpy(float),
                    rtol=1e-12,
                    atol=1e-12,
                ),
                f"trace {run.scenario}/{int(run.seed)} total overhead equals EKF + auxiliary + reliability timing",
            )

            startup = steps < 20
            check(
                (aux_timing[startup] == 0.0).all() and (reliability_timing[startup] == 0.0).all(),
                f"trace {run.scenario}/{int(run.seed)} pre-window auxiliary/reliability overhead is zero",
            )
            control_step = (steps % CONTROL_STRIDE) == 0
            check(
                slsqp_timing[control_step].notna().all() and slsqp_timing[~control_step].isna().all(),
                f"trace {run.scenario}/{int(run.seed)} SLSQP timing exists only at control updates",
            )

            ekf_stats = _timing_distribution(ekf_timing.to_numpy(float))
            overhead_stats = _timing_distribution(total_timing.to_numpy(float))
            slsqp_stats = _timing_distribution(slsqp_timing.dropna().to_numpy(float))
            expected_run_values = {
                "mean_ekf_update_ms": ekf_stats["mean_ms"],
                "median_ekf_update_ms": ekf_stats["median_ms"],
                "p95_ekf_update_ms": ekf_stats["p95_ms"],
                "p99_ekf_update_ms": ekf_stats["p99_ms"],
                "max_ekf_update_ms": ekf_stats["max_ms"],
                "total_reliability_observer_overhead_sample_count": overhead_stats["sample_count"],
                "mean_total_reliability_observer_overhead_ms": overhead_stats["mean_ms"],
                "median_total_reliability_observer_overhead_ms": overhead_stats["median_ms"],
                "p95_total_reliability_observer_overhead_ms": overhead_stats["p95_ms"],
                "p99_total_reliability_observer_overhead_ms": overhead_stats["p99_ms"],
                "max_total_reliability_observer_overhead_ms": overhead_stats["max_ms"],
                "slsqp_solve_count": slsqp_stats["sample_count"],
                "mean_slsqp_solve_ms": slsqp_stats["mean_ms"],
                "median_slsqp_solve_ms": slsqp_stats["median_ms"],
                "p95_slsqp_solve_ms": slsqp_stats["p95_ms"],
                "p99_slsqp_solve_ms": slsqp_stats["p99_ms"],
                "max_slsqp_solve_ms": slsqp_stats["max_ms"],
            }
            for column, expected in expected_run_values.items():
                actual = float(run[column])
                if column.endswith("_count"):
                    check(int(actual) == int(expected), f"run {run.scenario}/{int(run.seed)} {column} recomputes from raw timing")
                else:
                    check(
                        np.isclose(actual, float(expected), rtol=1e-12, atol=1e-12),
                        f"run {run.scenario}/{int(run.seed)} {column} recomputes from raw timing",
                    )


def _json_forbidden_seed_use(value: Any, key_path: tuple[str, ...] = ()) -> set[int]:
    hits: set[int] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            hits |= _json_forbidden_seed_use(child, key_path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits |= _json_forbidden_seed_use(child, key_path + (str(index),))
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) in FORBIDDEN_SEEDS:
        joined = ".".join(key_path).lower()
        if "forbidden" not in joined and "reserved" not in joined and "unused" not in joined:
            hits.add(int(value))
    return hits


def verify_no_forbidden_seed_result_use() -> None:
    conflicts: dict[str, list[int]] = {}
    for path in sorted(RESULTS.rglob("*.csv")):
        if "configs" in {part.lower() for part in path.parts}:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        seed_like = [column for column in frame.columns if column.lower() == "seed" or column.lower().endswith("_seed")]
        for column in seed_like:
            values = pd.to_numeric(frame[column], errors="coerce").dropna().astype(int)
            hits = sorted(set(values) & FORBIDDEN_SEEDS)
            if hits:
                conflicts.setdefault(display_path(path), []).extend(hits)
    for path in sorted(RESULTS.rglob("*.json")):
        if "configs" in {part.lower() for part in path.parts}:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        hits = sorted(_json_forbidden_seed_use(value))
        if hits:
            conflicts[display_path(path)] = hits
    for path in sorted(RESULTS.rglob("*.npz")):
        if "configs" in {part.lower() for part in path.parts}:
            continue
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                if "seed" not in key.lower():
                    continue
                values = np.asarray(archive[key]).reshape(-1)
                try:
                    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(int)
                except (TypeError, ValueError):
                    continue
                hits = sorted(set(numeric_values) & FORBIDDEN_SEEDS)
                if hits:
                    conflicts.setdefault(display_path(path), []).extend(hits)
    check(not conflicts, f"reserved seeds 39026-39030 are absent from non-config result evidence (conflicts={conflicts})")


def verify_closed_loop_plan(frozen: Mapping[str, Any], config_hash: str) -> str:
    plan = load_json(CLOSED_LOOP_PLAN_PATH)
    scientific = frozen["frozen_scientific_artifacts"]
    check(plan.get("status") == "frozen_before_first_closed_loop_run", "closed-loop plan was frozen before the first run")
    check(plan.get("ekf_config_sha256") == config_hash, "closed-loop plan binds the exact frozen EKF config")
    check(
        plan.get("ekf_preregistration_sha256") == scientific["ekf_preregistration"]["sha256"],
        "closed-loop plan binds the exact EKF preregistration",
    )
    check(
        plan.get("frozen_evidence_manifest_sha256") == sha256(FROZEN_EVIDENCE_PATH),
        "closed-loop plan binds the complete pre-EKF frozen evidence manifest",
    )
    check(tuple(plan.get("controllers", [])) == CONTROLLERS, "closed-loop plan freezes exact B/C1/C3/E controller order")
    check(tuple(plan.get("primary_scenarios", [])) == PRIMARY_SCENARIOS, "closed-loop plan freezes exact five primary scenarios")
    check(tuple(plan.get("development_seeds", [])) == DEVELOPMENT_SEEDS, "closed-loop plan freezes development seeds 19026-19030")
    check(tuple(plan.get("reserved_unused_seeds", [])) == tuple(sorted(FORBIDDEN_SEEDS)), "closed-loop plan preserves reserved seeds 39026-39030")
    check(
        plan.get("frozen_v3_matrix_sha256") == scientific["v3_275_run_matrix"]["sha256"],
        "closed-loop plan binds the frozen V3 matrix",
    )
    check(
        plan.get("frozen_c4_development_matrix_sha256") == scientific["c4_development_matrix"]["sha256"],
        "closed-loop plan binds the frozen C4 development matrix",
    )
    check(
        plan.get("runtime_accounting")
        == {
            "per_sample_total_overhead": "ekf_update_ms + aux_inference_ms + reliability_update_ms",
            "optimizer_timing": "slsqp_solve_ms recorded separately",
            "distribution_statistics": ["mean", "median", "p95", "p99", "max"],
        },
        "closed-loop plan freezes separate observer/reliability and SLSQP runtime accounting",
    )
    evaluator_hash = sha256(CLOSED_LOOP_EVALUATOR_PATH)
    check(
        plan.get("closed_loop_evaluator_sha256_before_first_run") == evaluator_hash,
        "closed-loop plan binds the exact evaluator source used before the first run",
    )
    return sha256(CLOSED_LOOP_PLAN_PATH)


def main() -> None:
    frozen = verify_frozen_evidence()
    config, _ = verify_calibration(frozen)
    config_hash = sha256(EKF_CONFIG_PATH)
    verify_estimator_csvs(config_hash, frozen)
    verify_no_forbidden_seed_result_use()
    plan_hash = verify_closed_loop_plan(frozen, config_hash)

    check(CLOSED_LOOP_RUNS_PATH.is_file(), "closed-loop run matrix exists")
    check(CLOSED_LOOP_SUMMARY_PATH.is_file(), "closed-loop summary exists")
    check(CLOSED_LOOP_TRACES_PATH.is_file(), "closed-loop trace evidence exists")
    runs = verify_closed_loop_matrix(pd.read_csv(CLOSED_LOOP_RUNS_PATH, float_precision="round_trip"))
    verify_ekf_run_evidence(runs)
    summary = load_json(CLOSED_LOOP_SUMMARY_PATH)
    role_gate = verify_closed_loop_summary(runs, summary, config_hash, plan_hash, frozen)
    verify_closed_loop_trace(runs)

    print(f"\nEKF role decision: {role_gate['decision']}")
    print("EKF VERIFIER RESULT: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nEKF VERIFIER RESULT: FAIL\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
