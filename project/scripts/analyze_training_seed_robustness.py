"""Analyze the preregistered three-training-seed V3 robustness study.

This is a post-evaluation analysis only.  It refuses to run unless the exact
150-row primary matrix and the matched P2026/P2027/P2028 artifact manifest
already exist.  It never trains models, calibrates thresholds, simulates the
plant, or edits frozen V3/C4/EKF evidence.

Outputs are machine-readable and live under::

    results/training_seed_robustness/analysis/

The final Markdown report is deliberately outside this script's scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
RESULT_ROOT = PROJECT / "results" / "training_seed_robustness"
ANALYSIS_ROOT = RESULT_ROOT / "analysis"
RUNS_PATH = RESULT_ROOT / "training_seed_runs.csv"
PAIR_MANIFEST_PATH = RESULT_ROOT / "model_pair_manifest.json"
EVALUATION_MANIFEST_PATH = RESULT_ROOT / "training_seed_evaluation_manifest.json"
SEED_FREEZE_PATH = RESULT_ROOT / "simulation_seed_freeze.json"

TRAINING_SEEDS = (2026, 2027, 2028)
SIMULATION_SEEDS = (49026, 49027, 49028, 49029, 49030)
VALIDATION_RUN_IDS = (3, 9, 11, 16, 19, 23)
CONTROLLERS = ("B_plain_MPC", "C3_arbitration_MPC")
SCENARIOS = (
    "sensor_bias_5",
    "sensor_dropout",
    "load_disturbance",
    "combined_fault_load",
    "parameter_variation",
)
HEADLINE_SCENARIOS = ("sensor_bias_5", "sensor_dropout", "combined_fault_load")
EXPECTED_RUN_COUNT = 150

DEFAULT_ANALYSIS_SEED = 20260916
DEFAULT_BOOTSTRAP_REPLICATES = 20_000

SAFETY_COLUMNS = (
    "optimizer_failures",
    "main_prediction_failures",
    "aux_prediction_failures",
    "voltage_violations",
    "slew_violations",
    "nonfinite_events",
)

OUTPUT_FILES = {
    "headline_paired_differences": "headline_paired_differences.csv",
    "scenario_training_seed_summary": "scenario_training_seed_summary.csv",
    "load_disturbance": "load_disturbance.csv",
    "parameter_variation": "parameter_variation.csv",
    "main_model_metrics_by_seed": "main_model_metrics_by_seed.csv",
    "main_model_metric_summary": "main_model_metric_summary.csv",
    "auxiliary_model_metrics_by_seed": "auxiliary_model_metrics_by_seed.csv",
    "auxiliary_model_metric_summary": "auxiliary_model_metric_summary.csv",
    "auxiliary_per_trajectory": "auxiliary_per_trajectory.csv",
    "auxiliary_closed_loop_transfer": "auxiliary_closed_loop_transfer.csv",
    "calibration_by_seed": "calibration_by_seed.csv",
    "calibration_variability": "calibration_variability.csv",
    "safety_by_training_seed": "safety_by_training_seed.csv",
    "hierarchical_bootstrap_ci": "hierarchical_bootstrap_ci.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT.resolve())).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(rel(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{rel(path)} must contain a JSON object")
    return payload


def resolve_project_file(raw_path: str, *, label: str) -> Path:
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else PROJECT / candidate
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes project root: {raw_path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} missing: {raw_path}")
    return resolved


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def numeric_series(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        raise RuntimeError(f"{label}.{column} contains missing/nonfinite values")
    return values.astype(float)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sample_sd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if len(array) < 2:
        return float("nan")
    return float(np.std(array, ddof=1))


def finite_or_nan(values: Iterable[Any]) -> np.ndarray:
    return pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)


def variability(values: Iterable[Any]) -> dict[str, float]:
    array = finite_or_nan(values)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"mean": np.nan, "sample_sd": np.nan, "min": np.nan, "max": np.nan, "range": np.nan}
    return {
        "mean": float(np.mean(array)),
        "sample_sd": sample_sd(array),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "range": float(np.ptp(array)),
    }


def validate_matrix(runs: pd.DataFrame) -> pd.DataFrame:
    required = [
        "training_seed",
        "simulation_seed",
        "seed",
        "controller",
        "scenario",
        "run_complete",
        "overall_rmse",
        "overall_mae",
        "fault_window_rmse",
        "reliability_entries",
        "sub_fraction",
        "event_frac_aux",
        "event_frac_fb",
        "frac_aux",
        "frac_fb",
        "main_model_sha256",
        "aux_model_sha256",
        "main_config_sha256",
        "aux_config_sha256",
        "sensor_calibration_sha256",
        "v3_calibration_sha256",
        "pair_config_sha256",
        *SAFETY_COLUMNS,
    ]
    require_columns(runs, required, "training_seed_runs.csv")
    if len(runs) != EXPECTED_RUN_COUNT:
        raise RuntimeError(f"expected exactly {EXPECTED_RUN_COUNT} rows, found {len(runs)}")

    runs = runs.copy()
    for column in ("training_seed", "simulation_seed", "seed"):
        values = numeric_series(runs, column, "training_seed_runs.csv")
        if not np.array_equal(values.to_numpy(), values.astype(int).to_numpy()):
            raise RuntimeError(f"{column} contains non-integer values")
        runs[column] = values.astype(int)

    if set(runs["training_seed"]) != set(TRAINING_SEEDS):
        raise RuntimeError("training seed set is not exactly {2026,2027,2028}")
    if set(runs["simulation_seed"]) != set(SIMULATION_SEEDS):
        raise RuntimeError("simulation seed set is not exactly {49026..49030}")
    if not np.array_equal(runs["seed"].to_numpy(), runs["simulation_seed"].to_numpy()):
        raise RuntimeError("legacy seed alias differs from simulation_seed")
    if set(runs["controller"].astype(str)) != set(CONTROLLERS):
        raise RuntimeError("controller set is not exactly B_plain_MPC/C3_arbitration_MPC")
    if set(runs["scenario"].astype(str)) != set(SCENARIOS):
        raise RuntimeError("scenario set differs from preregistered five-scenario matrix")

    key_columns = ["training_seed", "scenario", "simulation_seed", "controller"]
    if runs.duplicated(key_columns).any():
        raise RuntimeError("duplicate primary matrix keys")
    expected = {
        (training_seed, scenario, simulation_seed, controller)
        for training_seed in TRAINING_SEEDS
        for scenario in SCENARIOS
        for simulation_seed in SIMULATION_SEEDS
        for controller in CONTROLLERS
    }
    actual = set(map(tuple, runs[key_columns].itertuples(index=False, name=None)))
    if actual != expected:
        raise RuntimeError(
            f"primary matrix keys mismatch: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )

    for training_seed in TRAINING_SEEDS:
        seed_rows = runs[runs.training_seed == training_seed]
        if len(seed_rows) != 50:
            raise RuntimeError(f"training seed {training_seed} does not contain exactly 50 rows")
        for column in (
            "main_model_sha256",
            "aux_model_sha256",
            "main_config_sha256",
            "aux_config_sha256",
            "sensor_calibration_sha256",
            "v3_calibration_sha256",
            "pair_config_sha256",
        ):
            if seed_rows[column].nunique(dropna=False) != 1:
                raise RuntimeError(f"training seed {training_seed} mixes {column}")

    return runs


def validate_seed_freeze() -> dict[str, Any]:
    freeze = read_json(SEED_FREEZE_PATH)
    selected = tuple(int(value) for value in freeze.get("selected_simulation_seeds", []))
    if selected != SIMULATION_SEEDS or freeze.get("frozen_before_closed_loop") is not True:
        raise RuntimeError("simulation seed freeze does not bind 49026-49030 before closed loop")
    scan = freeze.get("provenance_scan")
    if not isinstance(scan, dict):
        raise RuntimeError("simulation seed freeze lacks provenance_scan")
    for key in ("strict_text_matches", "filename_matches", "csv_seed_matches", "npy_npz_seed_matches"):
        if scan.get(key) != []:
            raise RuntimeError(f"simulation seed provenance is contaminated: {key}={scan.get(key)}")
    return freeze


def validate_evaluation_manifest() -> dict[str, Any]:
    manifest = read_json(EVALUATION_MANIFEST_PATH)
    if manifest.get("study") != "training_seed_robustness":
        raise RuntimeError("training_seed_evaluation_manifest.json belongs to the wrong study")
    if int(manifest.get("run_count", -1)) != EXPECTED_RUN_COUNT:
        raise RuntimeError("evaluation manifest does not record exactly 150 runs")
    if tuple(int(value) for value in manifest.get("training_seeds", [])) != TRAINING_SEEDS:
        raise RuntimeError("evaluation manifest training seeds changed")
    if tuple(int(value) for value in manifest.get("simulation_seeds", [])) != SIMULATION_SEEDS:
        raise RuntimeError("evaluation manifest simulation seeds changed")
    output = manifest.get("output")
    if not isinstance(output, dict) or output.get("runs_sha256") != sha256(RUNS_PATH):
        raise RuntimeError("evaluation manifest does not bind the exact training_seed_runs.csv")
    return manifest


def load_pair_manifest() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest = read_json(PAIR_MANIFEST_PATH)
    if manifest.get("study") != "training_seed_robustness":
        raise RuntimeError("model_pair_manifest.json belongs to the wrong study")
    if tuple(int(value) for value in manifest.get("validation_run_ids", [])) != VALIDATION_RUN_IDS:
        raise RuntimeError("model_pair_manifest.json validation population changed")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, dict) or set(pairs) != {f"P{seed}" for seed in TRAINING_SEEDS}:
        raise RuntimeError("model_pair_manifest.json must contain exactly P2026/P2027/P2028")

    result: dict[int, dict[str, Any]] = {}
    for seed in TRAINING_SEEDS:
        entry = pairs[f"P{seed}"]
        if not isinstance(entry, dict) or int(entry.get("training_seed", -1)) != seed:
            raise RuntimeError(f"invalid manifest record P{seed}")
        for logical_name, path_key, hash_key in (
            ("main", "weights_path", "weights_sha256"),
            ("main", "config_path", "config_sha256"),
            ("auxiliary", "weights_path", "weights_sha256"),
            ("auxiliary", "config_path", "config_sha256"),
            ("sensor_calibration", "path", "sha256"),
            ("v3_calibration", "path", "sha256"),
            ("pair_config", "path", "sha256"),
        ):
            ref = entry.get(logical_name)
            if not isinstance(ref, dict):
                raise RuntimeError(f"P{seed}.{logical_name} missing")
            path = resolve_project_file(str(ref.get(path_key, "")), label=f"P{seed}.{logical_name}.{path_key}")
            declared = str(ref.get(hash_key, "")).lower()
            if len(declared) != 64 or sha256(path) != declared:
                raise RuntimeError(f"P{seed}.{logical_name}.{path_key} hash mismatch")
        result[seed] = entry
    return manifest, result


def validate_run_bindings(runs: pd.DataFrame, pairs: dict[int, dict[str, Any]]) -> None:
    mapping = {
        "main_model_sha256": ("main", "weights_sha256"),
        "main_config_sha256": ("main", "config_sha256"),
        "aux_model_sha256": ("auxiliary", "weights_sha256"),
        "aux_config_sha256": ("auxiliary", "config_sha256"),
        "sensor_calibration_sha256": ("sensor_calibration", "sha256"),
        "v3_calibration_sha256": ("v3_calibration", "sha256"),
        "pair_config_sha256": ("pair_config", "sha256"),
    }
    for seed, pair in pairs.items():
        subset = runs[runs.training_seed == seed]
        for run_column, (section, key) in mapping.items():
            expected = str(pair[section][key]).lower()
            actual = {str(value).lower() for value in subset[run_column]}
            if actual != {expected}:
                raise RuntimeError(f"training seed {seed} run binding mismatch for {run_column}")


def paired_runs(runs: pd.DataFrame, scenarios: Iterable[str]) -> pd.DataFrame:
    subset = runs[runs.scenario.isin(list(scenarios))].copy()
    rows: list[dict[str, Any]] = []
    for (training_seed, scenario, simulation_seed), group in subset.groupby(
        ["training_seed", "scenario", "simulation_seed"], sort=True
    ):
        if set(group.controller) != set(CONTROLLERS) or len(group) != 2:
            raise RuntimeError("paired B/C3 cell is incomplete")
        b = group[group.controller == "B_plain_MPC"].iloc[0]
        c3 = group[group.controller == "C3_arbitration_MPC"].iloc[0]
        rows.append(
            {
                "training_seed": int(training_seed),
                "scenario": str(scenario),
                "simulation_seed": int(simulation_seed),
                "b_fault_window_rmse": float(b.fault_window_rmse),
                "c3_fault_window_rmse": float(c3.fault_window_rmse),
                "c3_minus_b_fault_window_rmse": float(c3.fault_window_rmse - b.fault_window_rmse),
                "c3_improves": bool(c3.fault_window_rmse < b.fault_window_rmse),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "training_seed", "simulation_seed"]).reset_index(drop=True)


def scenario_seed_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, training_seed), group in pairs.groupby(["scenario", "training_seed"], sort=True):
        values = group["c3_minus_b_fault_window_rmse"].to_numpy(float)
        rows.append(
            {
                "scenario": scenario,
                "training_seed": int(training_seed),
                "paired_count": int(len(values)),
                "mean_c3_minus_b_fault_window_rmse": float(np.mean(values)),
                "sample_sd_c3_minus_b_fault_window_rmse": sample_sd(values),
                "min_c3_minus_b_fault_window_rmse": float(np.min(values)),
                "max_c3_minus_b_fault_window_rmse": float(np.max(values)),
                "favorable_pair_count": int(np.sum(values < 0.0)),
                "mean_effect_favorable": bool(np.mean(values) < 0.0),
            }
        )
    return pd.DataFrame(rows)


def load_disturbance_table(runs: pd.DataFrame) -> pd.DataFrame:
    subset = runs[runs.scenario == "load_disturbance"]
    rows: list[dict[str, Any]] = []
    for (training_seed, simulation_seed), group in subset.groupby(
        ["training_seed", "simulation_seed"], sort=True
    ):
        b = group[group.controller == "B_plain_MPC"].iloc[0]
        c3 = group[group.controller == "C3_arbitration_MPC"].iloc[0]
        rows.append(
            {
                "training_seed": int(training_seed),
                "simulation_seed": int(simulation_seed),
                "b_fault_window_rmse": float(b.fault_window_rmse),
                "c3_fault_window_rmse": float(c3.fault_window_rmse),
                "tracking_penalty_c3_minus_b": float(c3.fault_window_rmse - b.fault_window_rmse),
                "c3_false_reliability_entries": int(c3.reliability_entries),
                "c3_substitution_fraction": float(c3.sub_fraction),
                "c3_event_aux_fraction": float(c3.event_frac_aux),
                "c3_event_fallback_fraction": float(c3.event_frac_fb),
            }
        )
    return pd.DataFrame(rows)


def parameter_variation_table(runs: pd.DataFrame) -> pd.DataFrame:
    subset = runs[runs.scenario == "parameter_variation"]
    rows: list[dict[str, Any]] = []
    for (training_seed, simulation_seed), group in subset.groupby(
        ["training_seed", "simulation_seed"], sort=True
    ):
        b = group[group.controller == "B_plain_MPC"].iloc[0]
        c3 = group[group.controller == "C3_arbitration_MPC"].iloc[0]
        rows.append(
            {
                "training_seed": int(training_seed),
                "simulation_seed": int(simulation_seed),
                "b_fault_window_rmse": float(b.fault_window_rmse),
                "c3_fault_window_rmse": float(c3.fault_window_rmse),
                "c3_minus_b_fault_window_rmse": float(c3.fault_window_rmse - b.fault_window_rmse),
                "c3_substitution_fraction": float(c3.sub_fraction),
                "c3_aux_fraction": float(c3.frac_aux),
                "c3_fallback_fraction": float(c3.frac_fb),
                "c3_event_aux_fraction": float(c3.event_frac_aux),
                "c3_event_fallback_fraction": float(c3.event_frac_fb),
            }
        )
    return pd.DataFrame(rows)


def main_metrics_path(seed: int) -> Path:
    if seed == 2026:
        return PROJECT / "results" / "metrics" / "lstm_test_metrics.json"
    return RESULT_ROOT / "training" / f"main_seed{seed}_metrics.json"


def aux_metrics_path(seed: int) -> Path:
    if seed == 2026:
        return PROJECT / "results" / "metrics" / "v2_auxiliary_model_metrics.json"
    return RESULT_ROOT / "training" / f"aux_seed{seed}_metrics.json"


def main_config_path(seed: int, pair: dict[str, Any]) -> Path:
    return resolve_project_file(pair["main"]["config_path"], label=f"P{seed}.main.config_path")


def aux_config_path(seed: int, pair: dict[str, Any]) -> Path:
    return resolve_project_file(pair["auxiliary"]["config_path"], label=f"P{seed}.auxiliary.config_path")


def extract_main_metrics(pairs: dict[int, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    required_metric_names = ("rmse", "mae", "r2", "h5_rmse", "h10_rmse", "h15_rmse", "best_epoch")
    for seed in TRAINING_SEEDS:
        metrics_path = main_metrics_path(seed)
        metrics = read_json(metrics_path)
        config = read_json(main_config_path(seed, pairs[seed]))
        test = metrics.get("test")
        if not isinstance(test, dict):
            raise RuntimeError(f"{rel(metrics_path)} missing test metric block")
        recursive = metrics.get("recursive_key_horizons")
        if isinstance(recursive, dict):
            h5, h10, h15 = recursive.get("H5"), recursive.get("H10"), recursive.get("H15")
        else:
            horizons = metrics.get("recursive_rmse_by_horizon")
            if not isinstance(horizons, list) or len(horizons) < 15:
                raise RuntimeError(f"{rel(metrics_path)} missing H5/H10/H15 recursive metrics")
            h5, h10, h15 = horizons[4], horizons[9], horizons[14]
        training = config.get("training")
        if not isinstance(training, dict) or int(training.get("seed", -1)) != seed:
            raise RuntimeError(f"main config does not bind training seed {seed}")
        row = {
            "training_seed": seed,
            "rmse": float(test["rmse"]),
            "mae": float(test["mae"]),
            "r2": float(test["r2"]),
            "h5_rmse": float(h5),
            "h10_rmse": float(h10),
            "h15_rmse": float(h15),
            "best_epoch": int(metrics.get("best_epoch", training.get("best_epoch"))),
            "metrics_path": rel(metrics_path),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    finite = bool(np.isfinite(frame[list(required_metric_names)].to_numpy(float)).all())
    summary_rows: list[dict[str, Any]] = []
    for metric in required_metric_names:
        values = frame[metric].to_numpy(float)
        summary_rows.append({"metric": metric, **variability(values)})
    return frame, pd.DataFrame(summary_rows), finite


def extract_aux_metrics(
    pairs: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    required = ("test_rmse", "test_mae", "test_r2")
    all_finite = True
    for seed in TRAINING_SEEDS:
        metrics_path = aux_metrics_path(seed)
        metrics = read_json(metrics_path)
        config = read_json(aux_config_path(seed, pairs[seed]))
        training = config.get("training")
        if not isinstance(training, dict) or int(training.get("seed", -1)) != seed:
            raise RuntimeError(f"aux config does not bind training seed {seed}")
        per_test = metrics.get("per_trajectory_test")
        if not isinstance(per_test, list) or not per_test:
            raise RuntimeError(f"{rel(metrics_path)} missing per_trajectory_test")
        rmse_values = [float(row["rmse"]) for row in per_test]
        traj_stats = variability(rmse_values)
        row = {
            "training_seed": seed,
            "test_rmse": float(metrics["test_rmse"]),
            "test_mae": float(metrics["test_mae"]),
            "test_r2": float(metrics["test_r2"]),
            "best_epoch": int(metrics.get("best_epoch", training.get("best_epoch"))),
            "per_trajectory_rmse_mean": traj_stats["mean"],
            "per_trajectory_rmse_sample_sd": traj_stats["sample_sd"],
            "per_trajectory_rmse_min": traj_stats["min"],
            "per_trajectory_rmse_max": traj_stats["max"],
            "per_trajectory_rmse_range": traj_stats["range"],
            "training_runtime_seconds": float(metrics.get("training_runtime_seconds", np.nan)),
            "metrics_path": rel(metrics_path),
        }
        metric_rows.append(row)
        required_values = [row[name] for name in required]
        all_finite = all_finite and bool(np.isfinite(required_values).all())
        for trajectory in per_test:
            trajectory_rows.append(
                {
                    "training_seed": seed,
                    "split": "test",
                    "run_id": int(trajectory["run_id"]),
                    "rmse": float(trajectory["rmse"]),
                    "mae": float(trajectory["mae"]),
                    "has_load_change": bool(trajectory.get("has_load_change", False)),
                    "load_torque_range": float(trajectory.get("load_torque_range", np.nan)),
                }
            )
    metric_frame = pd.DataFrame(metric_rows)
    summary_rows: list[dict[str, Any]] = []
    for metric in (
        "test_rmse",
        "test_mae",
        "test_r2",
        "best_epoch",
        "per_trajectory_rmse_mean",
        "per_trajectory_rmse_sample_sd",
        "per_trajectory_rmse_range",
    ):
        summary_rows.append({"metric": metric, **variability(metric_frame[metric])})
    return metric_frame, pd.DataFrame(summary_rows), pd.DataFrame(trajectory_rows), all_finite


def auxiliary_transfer_table(runs: pd.DataFrame) -> pd.DataFrame:
    c3 = runs[
        (runs.controller == "C3_arbitration_MPC")
        & (runs.scenario.isin(["load_disturbance", "parameter_variation"]))
    ].copy()
    optional_fields = [
        "aux_virtual_rmse",
        "aux_virtual_mae",
        "main_virtual_rmse",
        "selected_virtual_rmse",
        "sub_samples",
    ]
    for field in optional_fields:
        if field not in c3:
            c3[field] = np.nan

    rows: list[dict[str, Any]] = []
    for (training_seed, scenario), group in c3.groupby(["training_seed", "scenario"], sort=True):
        row: dict[str, Any] = {
            "training_seed": int(training_seed),
            "scenario": scenario,
            "run_count": int(len(group)),
        }
        for field in (
            "aux_virtual_rmse",
            "aux_virtual_mae",
            "main_virtual_rmse",
            "selected_virtual_rmse",
            "event_frac_aux",
            "event_frac_fb",
            "sub_fraction",
            "sub_samples",
        ):
            stats = variability(group[field])
            for key, value in stats.items():
                row[f"{field}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_tables(
    pairs: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        pair = pairs[seed]
        sensor_path = resolve_project_file(pair["sensor_calibration"]["path"], label=f"P{seed}.sensor_calibration")
        v3_path = resolve_project_file(pair["v3_calibration"]["path"], label=f"P{seed}.v3_calibration")
        sensor = read_json(sensor_path)
        v3 = read_json(v3_path)
        if int(sensor.get("training_seed", -1)) != seed or int(v3.get("training_seed", -1)) != seed:
            raise RuntimeError(f"training seed {seed} calibration ownership mismatch")
        if tuple(int(value) for value in sensor.get("validation_run_ids", [])) != VALIDATION_RUN_IDS:
            raise RuntimeError(f"training seed {seed} sensor calibration population changed")
        if tuple(int(value) for value in v3.get("validation_run_ids", [])) != VALIDATION_RUN_IDS:
            raise RuntimeError(f"training seed {seed} V3 calibration population changed")
        if "clean validation" not in str(sensor.get("calibration_population", "")).lower():
            raise RuntimeError(f"training seed {seed} sensor calibration is not validation-only")
        if "clean validation" not in str(v3.get("calibration_population", "")).lower():
            raise RuntimeError(f"training seed {seed} V3 calibration is not validation-only")
        sensor_values = sensor.get("computed_values")
        v3_values = v3.get("computed_values")
        v3_stats = v3.get("statistics")
        sensor_false_alarm = sensor.get("clean_validation_false_alarm_statistics")
        if not all(isinstance(value, dict) for value in (sensor_values, v3_values, v3_stats, sensor_false_alarm)):
            raise RuntimeError(f"training seed {seed} calibration schema incomplete")
        rows.append(
            {
                "training_seed": seed,
                "residual_gate": float(sensor_values["instant_threshold"]),
                "residual_center": float(sensor_values["residual_center"]),
                "cusum_center": float(sensor_values["center"]),
                "cusum_allowance": float(sensor_values["allowance"]),
                "cusum_threshold": float(sensor_values["threshold"]),
                "sensor_signed_ewma_threshold": float(sensor_values["signed_ewma_threshold"]),
                "sensor_absolute_ewma_threshold": float(sensor_values["absolute_ewma_threshold"]),
                "sensor_validation_alarm_rate": float(sensor_false_alarm["selected_debounced_sensor_alarm_rate"]),
                "agreement_threshold": float(v3_values["agreement_threshold"]),
                "parameter_mismatch_threshold": float(v3_values["param_mismatch_threshold"]),
                "recovery_threshold": float(v3_values["param_mismatch_recovery_threshold"]),
                "auxiliary_recovery_gate": float(v3_values["aux_recovery_gate"]),
                "v3_agreement_p95": float(v3_stats["agreement_p95"]),
                "v3_agreement_p99": float(v3_stats["agreement_p99"]),
                "v3_aux_prediction_min": float(v3_stats["aux_prediction_min"]),
                "v3_aux_prediction_max": float(v3_stats["aux_prediction_max"]),
            }
        )
    frame = pd.DataFrame(rows)
    variability_rows: list[dict[str, Any]] = []
    for column in frame.columns:
        if column == "training_seed":
            continue
        stats = variability(frame[column])
        mean = stats["mean"]
        cv = stats["sample_sd"] / abs(mean) if np.isfinite(mean) and abs(mean) > 1e-12 else np.nan
        variability_rows.append({"calibration_statistic": column, **stats, "coefficient_of_variation": cv})
    return frame, pd.DataFrame(variability_rows)


def safety_table(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], bool]:
    frame = runs.copy()
    if "slew_violations" not in frame and "rate_violations" in frame:
        frame["slew_violations"] = frame["rate_violations"]
    for column in SAFETY_COLUMNS:
        frame[column] = numeric_series(frame, column, "training_seed_runs.csv")

    run_complete = frame["run_complete"]
    if run_complete.dtype == bool:
        complete_mask = run_complete
    else:
        normalized = run_complete.astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false", "1", "0"}).all():
            raise RuntimeError("run_complete contains non-boolean values")
        complete_mask = normalized.isin({"true", "1"})

    rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        subset = frame[frame.training_seed == seed]
        row: dict[str, Any] = {"training_seed": seed, "run_count": int(len(subset))}
        for column in SAFETY_COLUMNS:
            row[column] = int(pd.to_numeric(subset[column]).sum())
        row["incomplete_runs"] = int((~complete_mask.loc[subset.index]).sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    totals = {column: int(table[column].sum()) for column in SAFETY_COLUMNS}
    totals["incomplete_runs"] = int(table["incomplete_runs"].sum())
    safe = all(value == 0 for value in totals.values())
    return table, totals, safe


def hierarchical_bootstrap(
    all_pairs: pd.DataFrame,
    *,
    analysis_seed: int,
    replicates: int,
) -> pd.DataFrame:
    if replicates < 1000:
        raise ValueError("bootstrap replicates must be at least 1000 for stable descriptive intervals")
    rng = np.random.default_rng(analysis_seed)
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        scenario_pairs = all_pairs[all_pairs.scenario == scenario]
        by_seed = {
            seed: scenario_pairs[scenario_pairs.training_seed == seed]
            .sort_values("simulation_seed")["c3_minus_b_fault_window_rmse"]
            .to_numpy(float)
            for seed in TRAINING_SEEDS
        }
        if any(len(values) != len(SIMULATION_SEEDS) for values in by_seed.values()):
            raise RuntimeError(f"bootstrap input incomplete for {scenario}")
        seed_means = np.asarray([np.mean(by_seed[seed]) for seed in TRAINING_SEEDS])
        bootstrap_values = np.empty(replicates, dtype=float)
        for index in range(replicates):
            sampled_training_seeds = rng.choice(TRAINING_SEEDS, size=len(TRAINING_SEEDS), replace=True)
            sampled_cluster_means = []
            for training_seed in sampled_training_seeds:
                values = by_seed[int(training_seed)]
                sampled = rng.choice(values, size=len(values), replace=True)
                sampled_cluster_means.append(float(np.mean(sampled)))
            bootstrap_values[index] = float(np.mean(sampled_cluster_means))
        low, high = np.quantile(bootstrap_values, [0.025, 0.975])
        rows.append(
            {
                "scenario": scenario,
                "point_estimate_mean_of_training_seed_means": round(float(np.mean(seed_means)), 6),
                "ci95_low": round(float(low), 6),
                "ci95_high": round(float(high), 6),
                "analysis_seed": int(analysis_seed),
                "bootstrap_replicates": int(replicates),
                "training_seed_clusters": len(TRAINING_SEEDS),
                "simulation_seeds_per_cluster": len(SIMULATION_SEEDS),
            }
        )
    return pd.DataFrame(rows)


def aggregate_load(load_table: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        group = load_table[load_table.training_seed == seed]
        total_false_entries = int(group.c3_false_reliability_entries.sum())
        rows.append(
            {
                "training_seed": seed,
                "mean_tracking_penalty_c3_minus_b": float(group.tracking_penalty_c3_minus_b.mean()),
                "total_false_reliability_entries": total_false_entries,
                "mean_substitution_fraction": float(group.c3_substitution_fraction.mean()),
                "tracking_degradation_consistent_with_frozen_limitation": bool(
                    group.tracking_penalty_c3_minus_b.mean() > 0.0
                ),
                "false_entry_weakness_consistent_with_frozen_limitation": bool(
                    total_false_entries > 0
                ),
            }
        )
    return rows


def aggregate_parameter(parameter_table: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        group = parameter_table[parameter_table.training_seed == seed]
        rows.append(
            {
                "training_seed": seed,
                "b_fault_window_rmse_mean": float(group.b_fault_window_rmse.mean()),
                "b_fault_window_rmse_sample_sd": sample_sd(group.b_fault_window_rmse),
                "c3_fault_window_rmse_mean": float(group.c3_fault_window_rmse.mean()),
                "c3_fault_window_rmse_sample_sd": sample_sd(group.c3_fault_window_rmse),
                "c3_minus_b_mean": float(group.c3_minus_b_fault_window_rmse.mean()),
                "c3_minus_b_sample_sd": sample_sd(group.c3_minus_b_fault_window_rmse),
                "c3_substitution_fraction_mean": float(group.c3_substitution_fraction.mean()),
                "c3_substitution_fraction_sample_sd": sample_sd(group.c3_substitution_fraction),
                "c3_event_aux_fraction_mean": float(group.c3_event_aux_fraction.mean()),
                "c3_event_aux_fraction_sample_sd": sample_sd(group.c3_event_aux_fraction),
                "c3_event_fallback_fraction_mean": float(group.c3_event_fallback_fraction.mean()),
                "c3_event_fallback_fraction_sample_sd": sample_sd(group.c3_event_fallback_fraction),
            }
        )
    return rows


def verdict(
    scenario_summary: pd.DataFrame,
    load_summary: list[dict[str, Any]],
    *,
    run_safety_ok: bool,
    main_metrics_finite: bool,
    aux_metrics_finite: bool,
) -> dict[str, Any]:
    headline_checks: dict[str, bool] = {}
    headline_seed_means: dict[str, dict[str, float]] = {}
    for scenario in HEADLINE_SCENARIOS:
        group = scenario_summary[scenario_summary.scenario == scenario]
        if set(group.training_seed) != set(TRAINING_SEEDS):
            raise RuntimeError(f"headline summary incomplete for {scenario}")
        means = {
            str(int(row.training_seed)): float(row.mean_c3_minus_b_fault_window_rmse)
            for row in group.itertuples(index=False)
        }
        headline_seed_means[scenario] = means
        headline_checks[scenario] = all(value < 0.0 for value in means.values())

    model_pair_numerical_safety_ok = bool(run_safety_ok and main_metrics_finite and aux_metrics_finite)
    load_weakness_consistent = all(
        row["tracking_degradation_consistent_with_frozen_limitation"]
        and row["false_entry_weakness_consistent_with_frozen_limitation"]
        for row in load_summary
    )
    qualitative_paper_conclusions_hold = bool(load_weakness_consistent)

    criteria = {
        "sensor_bias_5_improvement_all_training_seeds": headline_checks["sensor_bias_5"],
        "sensor_dropout_improvement_all_training_seeds": headline_checks["sensor_dropout"],
        "combined_fault_load_favorable_all_training_seeds": headline_checks["combined_fault_load"],
        "no_model_pair_numerical_or_safety_failure": model_pair_numerical_safety_ok,
        "qualitative_paper_conclusions_do_not_reverse": qualitative_paper_conclusions_hold,
    }
    final = (
        "ROBUST ACROSS TRAINING SEEDS"
        if all(criteria.values())
        else "TRAINING-SEED-SENSITIVE"
    )
    return {
        "verdict": final,
        "criteria": criteria,
        "headline_training_seed_mean_c3_minus_b_fault_window_rmse": headline_seed_means,
        "load_weakness_consistent_across_training_seeds": load_weakness_consistent,
        "parameter_variation_verdict_role": (
            "descriptive unless it produces numerical/safety failure; no post-hoc parameter-variation "
            "effect-size threshold is introduced"
        ),
        "verdict_rule": (
            "ROBUST ACROSS TRAINING SEEDS only if sensor_bias_5, sensor_dropout, and "
            "combined_fault_load have negative per-training-seed mean paired C3-B fault-window RMSE "
            "for all 2026/2027/2028; no model-pair numerical/safety failure occurs; and the frozen "
            "load-disturbance tracking degradation and false reliability-entry weakness remain "
            "qualitatively consistent across training seeds. "
            "Otherwise TRAINING-SEED-SENSITIVE."
        ),
    }


def write_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = ANALYSIS_ROOT / filename
    frame.to_csv(path, index=False)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-seed", type=int, default=DEFAULT_ANALYSIS_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing analysis outputs; primary scientific inputs are never modified",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not RUNS_PATH.is_file():
        raise FileNotFoundError(
            f"{rel(RUNS_PATH)} does not exist; do not run analysis before the 150-run matrix exists"
        )
    if not PAIR_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing {rel(PAIR_MANIFEST_PATH)}")

    if ANALYSIS_ROOT.exists() and any(ANALYSIS_ROOT.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"analysis outputs already exist under {rel(ANALYSIS_ROOT)}; use --overwrite only for an intentional re-analysis"
        )

    runs = validate_matrix(pd.read_csv(RUNS_PATH, float_precision="round_trip"))
    seed_freeze = validate_seed_freeze()
    evaluation_manifest = validate_evaluation_manifest()
    pair_manifest, pairs = load_pair_manifest()
    validate_run_bindings(runs, pairs)

    all_pairs = paired_runs(runs, SCENARIOS)
    headline_pairs = all_pairs[all_pairs.scenario.isin(HEADLINE_SCENARIOS)].copy()
    if len(headline_pairs) != len(TRAINING_SEEDS) * len(HEADLINE_SCENARIOS) * len(SIMULATION_SEEDS):
        raise RuntimeError("headline paired-difference table is incomplete")
    seed_summary = scenario_seed_summary(all_pairs)
    load_table = load_disturbance_table(runs)
    parameter_table = parameter_variation_table(runs)
    main_by_seed, main_summary, main_finite = extract_main_metrics(pairs)
    aux_by_seed, aux_summary, aux_per_trajectory, aux_finite = extract_aux_metrics(pairs)
    aux_transfer = auxiliary_transfer_table(runs)
    calibration_by_seed, calibration_variability = calibration_tables(pairs)
    safety_by_seed, safety_totals, run_safety_ok = safety_table(runs)
    bootstrap = hierarchical_bootstrap(
        all_pairs,
        analysis_seed=args.analysis_seed,
        replicates=args.bootstrap_replicates,
    )

    load_summary = aggregate_load(load_table)
    parameter_summary = aggregate_parameter(parameter_table)
    verdict_payload = verdict(
        seed_summary,
        load_summary,
        run_safety_ok=run_safety_ok,
        main_metrics_finite=main_finite,
        aux_metrics_finite=aux_finite,
    )

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    tables = {
        "headline_paired_differences": headline_pairs,
        "scenario_training_seed_summary": seed_summary,
        "load_disturbance": load_table,
        "parameter_variation": parameter_table,
        "main_model_metrics_by_seed": main_by_seed,
        "main_model_metric_summary": main_summary,
        "auxiliary_model_metrics_by_seed": aux_by_seed,
        "auxiliary_model_metric_summary": aux_summary,
        "auxiliary_per_trajectory": aux_per_trajectory,
        "auxiliary_closed_loop_transfer": aux_transfer,
        "calibration_by_seed": calibration_by_seed,
        "calibration_variability": calibration_variability,
        "safety_by_training_seed": safety_by_seed,
        "hierarchical_bootstrap_ci": bootstrap,
    }
    written_tables = {
        name: write_csv(frame, OUTPUT_FILES[name]) for name, frame in tables.items()
    }

    summary = {
        "study": "training_seed_robustness",
        "analysis_role": "post_evaluation_machine_readable_analysis",
        "training_seeds": list(TRAINING_SEEDS),
        "simulation_seeds": list(SIMULATION_SEEDS),
        "controllers": list(CONTROLLERS),
        "scenarios": list(SCENARIOS),
        "run_count": int(len(runs)),
        "source_hashes": {
            "training_seed_runs_csv": sha256(RUNS_PATH),
            "model_pair_manifest_json": sha256(PAIR_MANIFEST_PATH),
            "simulation_seed_freeze_json": sha256(SEED_FREEZE_PATH),
            "training_seed_evaluation_manifest_json": (
                sha256(EVALUATION_MANIFEST_PATH) if EVALUATION_MANIFEST_PATH.is_file() else None
            ),
        },
        "seed_provenance_conclusion": seed_freeze["provenance_scan"].get("conclusion"),
        "headline_scenario_training_seed_summary": json_ready(
            seed_summary[seed_summary.scenario.isin(HEADLINE_SCENARIOS)].to_dict(orient="records")
        ),
        "load_disturbance": load_summary,
        "parameter_variation": parameter_summary,
        "main_model_metrics": {
            "by_training_seed": json_ready(main_by_seed.to_dict(orient="records")),
            "across_training_seed_variability": json_ready(main_summary.to_dict(orient="records")),
        },
        "auxiliary_model_metrics": {
            "by_training_seed": json_ready(aux_by_seed.to_dict(orient="records")),
            "across_training_seed_variability": json_ready(aux_summary.to_dict(orient="records")),
            "closed_loop_transfer": json_ready(aux_transfer.to_dict(orient="records")),
        },
        "calibration": {
            "by_training_seed": json_ready(calibration_by_seed.to_dict(orient="records")),
            "variability": json_ready(calibration_variability.to_dict(orient="records")),
        },
        "safety": {
            "totals_across_150_runs": safety_totals,
            "all_required_safety_and_numerical_counts_zero": run_safety_ok,
            "main_required_metrics_finite": main_finite,
            "auxiliary_required_metrics_finite": aux_finite,
            "model_calibration_run_binding_verified": True,
        },
        "hierarchical_bootstrap": {
            "analysis_seed": int(args.analysis_seed),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "method": (
                "For each replicate and scenario, sample the three training-seed clusters with replacement; "
                "within each selected training-seed cluster sample its five paired simulation-seed C3-B "
                "differences with replacement; average within clusters then across sampled clusters."
            ),
            "interval": "descriptive percentile 95% CI",
            "training_seed_level_limitation": (
                "Only three training seeds are available; these intervals are descriptive and do not justify "
                "high-precision population inference about training-initialization variability."
            ),
            "results": json_ready(bootstrap.to_dict(orient="records")),
        },
        "verdict": verdict_payload,
        "outputs": {
            name: {"path": rel(path), "sha256": sha256(path)}
            for name, path in written_tables.items()
        },
        "evaluation_manifest_run_count": int(evaluation_manifest["run_count"]),
        "model_pair_manifest_protocol_sha256": (
            pair_manifest.get("protocol_manifest", {}).get("sha256")
            if isinstance(pair_manifest.get("protocol_manifest"), dict)
            else None
        ),
    }
    summary_path = ANALYSIS_ROOT / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(json_ready(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved machine-readable training-seed analysis to {rel(ANALYSIS_ROOT)}")
    print(verdict_payload["verdict"])


if __name__ == "__main__":
    main()
