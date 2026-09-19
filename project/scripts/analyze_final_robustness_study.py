"""Analyze the preregistered Part B final robustness/severity study.

The script is intentionally analysis-only.  It consumes the frozen study
protocol plus the completed Part B run CSV, validates the exact preregistered
matrix and model/calibration bindings, then writes machine-readable summaries
under ``results/final_robustness``.  It never trains, simulates, recalibrates,
or changes a classification rule from the frozen protocol.

The expected completed matrix is:

* 600 primary B_plain_MPC / C3_arbitration_MPC rows
* 135 current-boundary B / C3 / E_EKF_virtual_MPC rows

The default run CSV is ``results/final_robustness/final_robustness_runs.csv``.  A
different completed CSV may be supplied with ``--runs-csv``; its contents are
still required to match the frozen protocol exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
RESULT_ROOT = PROJECT / "results" / "final_robustness"
PROTOCOL_PATH = RESULT_ROOT / "study_protocol.json"
PAIR_MANIFEST_PATH = PROJECT / "results" / "training_seed_robustness" / "model_pair_manifest.json"
DEFAULT_RUNS_PATH = RESULT_ROOT / "final_robustness_runs.csv"

TRAINING_SEEDS = (2026, 2027, 2028)
PRIMARY_CONTROLLERS = ("B_plain_MPC", "C3_arbitration_MPC")
CURRENT_CONTROLLERS = ("B_plain_MPC", "C3_arbitration_MPC", "E_EKF_virtual_MPC")
PRIMARY_FAMILIES = ("bias", "dropout", "drift", "load", "combined_bias_load")
CLASSIFIED_FAMILIES = ("bias", "dropout", "drift", "combined_bias_load")

SAFETY_COLUMNS = (
    "optimizer_failures",
    "main_prediction_failures",
    "aux_prediction_failures",
    "ekf_numerical_failures",
    "nonfinite_events",
    "voltage_violations",
    "slew_violations",
)

HASH_BINDINGS = {
    "main_model_sha256": ("main", "weights_sha256"),
    "main_config_sha256": ("main", "config_sha256"),
    "aux_model_sha256": ("auxiliary", "weights_sha256"),
    "aux_config_sha256": ("auxiliary", "config_sha256"),
    "sensor_calibration_sha256": ("sensor_calibration", "sha256"),
    "v3_calibration_sha256": ("v3_calibration", "sha256"),
    "pair_config_sha256": ("pair_config", "sha256"),
}

OUTPUT_FILES = {
    "paired_points": "severity_paired_points.csv",
    "training_seed_means": "severity_training_seed_means.csv",
    "detection_wilson": "detection_wilson_intervals.csv",
    "severity_bootstrap": "severity_hierarchical_bootstrap.csv",
    "source_recovery": "source_recovery_summary.csv",
    "load_summary": "load_false_entry_summary.csv",
    "current_paired_points": "current_sensor_paired_points.csv",
    "current_summary": "current_sensor_summary.csv",
    "current_bootstrap": "current_sensor_hierarchical_bootstrap.csv",
    "combined_effects": "combined_grid_effects.csv",
    "safety": "part_b_safety_summary.csv",
    "operating_envelope": "operating_envelope_classification.csv",
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


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def numeric(frame: pd.DataFrame, column: str, *, allow_nan: bool = False) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    array = values.to_numpy(float)
    if allow_nan:
        if np.isinf(array).any():
            raise RuntimeError(f"{column} contains infinite values")
    elif values.isna().any() or not np.isfinite(array).all():
        raise RuntimeError(f"{column} contains missing/nonfinite values")
    return values.astype(float)


def bool_series(values: pd.Series, label: str, *, allow_nan: bool = False) -> pd.Series:
    if values.dtype == bool:
        return values.astype("boolean") if allow_nan else values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
        "nan": pd.NA,
        "none": pd.NA,
        "": pd.NA,
        "<na>": pd.NA,
    }
    parsed = normalized.map(mapping)
    invalid = parsed.isna() & ~normalized.isin({"nan", "none", "", "<na>"})
    if invalid.any():
        bad = sorted(set(values[invalid].astype(str)))[:5]
        raise RuntimeError(f"{label} contains non-boolean values: {bad}")
    if not allow_nan and parsed.isna().any():
        raise RuntimeError(f"{label} contains missing boolean values")
    return parsed.astype("boolean") if allow_nan else parsed.astype(bool)


def _copy_alias(frame: pd.DataFrame, target: str, aliases: Sequence[str], *, required: bool = False) -> None:
    if target in frame.columns:
        return
    present = [name for name in aliases if name in frame.columns]
    if len(present) > 1:
        first = frame[present[0]]
        for name in present[1:]:
            if not first.equals(frame[name]):
                raise RuntimeError(f"ambiguous aliases for {target}: {present}")
    if present:
        frame[target] = frame[present[0]]
    elif required:
        raise KeyError(f"run CSV missing {target}; accepted aliases={list(aliases)}")


def _normalize_family(value: Any) -> str:
    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bias": "bias",
        "speed_bias": "bias",
        "bias_severity": "bias",
        "dropout": "dropout",
        "sensor_dropout": "dropout",
        "dropout_duration": "dropout",
        "drift": "drift",
        "sensor_drift": "drift",
        "drift_severity": "drift",
        "load": "load",
        "load_disturbance": "load",
        "load_magnitude": "load",
        "combined": "combined_bias_load",
        "combined_fault_load": "combined_bias_load",
        "combined_bias_load": "combined_bias_load",
        "current": "current_sensor",
        "current_sensor": "current_sensor",
        "current_boundary": "current_sensor",
        "current_sensor_boundary": "current_sensor",
    }
    if raw in aliases:
        return aliases[raw]
    for prefix, canonical in (
        ("bias_", "bias"),
        ("dropout_", "dropout"),
        ("drift_", "drift"),
        ("load_", "load"),
        ("combined_", "combined_bias_load"),
        ("current_", "current_sensor"),
    ):
        if raw.startswith(prefix):
            return canonical
    raise RuntimeError(f"unrecognized sweep family {value!r}")


def canonicalize_runs(raw: pd.DataFrame) -> pd.DataFrame:
    runs = raw.copy()
    aliases: dict[str, tuple[str, ...]] = {
        "family": ("sweep_family", "study_family", "scenario_family", "sweep"),
        "simulation_seed": ("seed", "sim_seed"),
        "bias_percent": ("speed_bias_percent", "bias_pct", "combined_bias_percent"),
        "dropout_duration_s": ("dropout_s", "duration_s", "dropout_duration"),
        "drift_final_percent": ("drift_percent", "final_drift_percent", "drift_final_pct"),
        "post_step_load_Nm": ("load_Nm", "load_torque_Nm", "post_step_load"),
        "current_case": ("current_condition", "boundary_case", "condition_id", "scenario_name"),
        "severity_value": ("severity", "level", "condition_value"),
        "frac_phys": ("physical_fraction",),
        "frac_main": ("main_fraction",),
        "frac_aux": ("auxiliary_fraction",),
        "frac_fb": ("fallback_fraction",),
        "event_frac_phys": ("event_physical_fraction",),
        "event_frac_main": ("event_main_fraction",),
        "event_frac_aux": ("event_auxiliary_fraction",),
        "event_frac_fb": ("event_fallback_fraction",),
        "slew_violations": ("rate_violations",),
        "first_post_event_reliability_entry_time_s": (
            "first_reliability_entry_time_s",
            "first_entry_time_s",
        ),
        "post_event_reliability_active_duration_s": (
            "post_event_active_episode_duration_s",
            "post_event_active_duration_s",
            "post_event_sub_duration_s",
            "event_sub_duration_s",
        ),
        "aux_speed_rmse_event_window": (
            "aux_estimator_rmse_event_window",
            "auxiliary_speed_rmse_event_window",
        ),
        "aux_virtual_rmse": ("aux_estimator_rmse", "auxiliary_estimator_rmse"),
        "ekf_speed_rmse_event_window": ("ekf_estimator_rmse",),
        "ekf_event_sub_fraction": ("event_sub_fraction",),
        "recovered_after_event": ("sensor_recovered_after_event",),
        "ekf_numerical_failures": ("ekf_failures",),
    }
    for target, names in aliases.items():
        _copy_alias(runs, target, names, required=target in {"family", "simulation_seed"})

    require_columns(runs, ["training_seed", "simulation_seed", "controller", "family"], "Part B run CSV")
    runs["family"] = runs["family"].map(_normalize_family)
    runs["training_seed"] = numeric(runs, "training_seed").astype(int)
    runs["simulation_seed"] = numeric(runs, "simulation_seed").astype(int)
    if "seed" in runs.columns:
        seed_alias = pd.to_numeric(runs["seed"], errors="coerce")
        if seed_alias.isna().any() or not np.array_equal(seed_alias.astype(int), runs["simulation_seed"]):
            raise RuntimeError("legacy seed alias differs from simulation_seed")

    if "severity_value" in runs.columns:
        severity = pd.to_numeric(runs["severity_value"], errors="coerce")
        for family, target in (
            ("bias", "bias_percent"),
            ("dropout", "dropout_duration_s"),
            ("drift", "drift_final_percent"),
            ("load", "post_step_load_Nm"),
        ):
            if target not in runs.columns:
                runs[target] = np.nan
            mask = runs["family"].eq(family) & runs[target].isna()
            runs.loc[mask, target] = severity.loc[mask]

    return runs


def validate_protocol(protocol: dict[str, Any]) -> tuple[tuple[int, ...], dict[str, Any]]:
    if protocol.get("study") != "final_robustness_hardening":
        raise RuntimeError("study_protocol.json belongs to the wrong study")
    if protocol.get("status") != "frozen_before_any_part_b_scientific_run":
        raise RuntimeError("study_protocol.json was not frozen before Part B runs")
    training = tuple(int(value) for value in protocol.get("training_seeds", []))
    if training != TRAINING_SEEDS:
        raise RuntimeError(f"training seeds changed: {training}")

    freeze_entry = protocol.get("simulation_seed_freeze")
    if not isinstance(freeze_entry, dict):
        raise RuntimeError("protocol missing simulation_seed_freeze binding")
    freeze_path = PROJECT / str(freeze_entry.get("path", ""))
    if not freeze_path.is_file() or sha256(freeze_path) != str(freeze_entry.get("sha256", "")).lower():
        raise RuntimeError("simulation seed freeze hash does not match protocol")
    freeze = read_json(freeze_path)
    seeds = tuple(int(value) for value in freeze.get("selected_simulation_seeds", []))
    declared = tuple(int(value) for value in freeze_entry.get("seeds", []))
    if seeds != declared or len(seeds) != 5 or len(set(seeds)) != 5:
        raise RuntimeError("simulation seed block differs between protocol and seed freeze")
    if freeze.get("status") != "frozen_before_any_part_b_scientific_run":
        raise RuntimeError("simulation seed block was not prospectively frozen")
    scan = freeze.get("provenance_scan")
    if not isinstance(scan, dict):
        raise RuntimeError("simulation seed freeze missing provenance scan")
    for key in ("strict_text_matches", "filename_matches", "csv_numeric_matches", "npy_npz_numeric_matches"):
        if scan.get(key) != []:
            raise RuntimeError(f"selected simulation seed provenance is contaminated: {key}")

    pair_entry = protocol.get("model_pair_manifest")
    if not isinstance(pair_entry, dict):
        raise RuntimeError("protocol missing model-pair manifest binding")
    pair_path = PROJECT / str(pair_entry.get("path", ""))
    if pair_path.resolve() != PAIR_MANIFEST_PATH.resolve():
        raise RuntimeError("protocol model-pair manifest path changed")
    if not pair_path.is_file() or sha256(pair_path) != str(pair_entry.get("sha256", "")).lower():
        raise RuntimeError("model-pair manifest hash does not match protocol")

    stats = protocol.get("statistics")
    if not isinstance(stats, dict):
        raise RuntimeError("protocol missing statistics block")
    if tuple(stats.get("hierarchical_bootstrap_order", [])) != ("training_seed", "simulation_seed"):
        raise RuntimeError("bootstrap hierarchy differs from preregistration")
    if int(stats.get("bootstrap_replicates", -1)) != 20_000:
        raise RuntimeError("bootstrap replicate count differs from preregistered 20,000")
    if int(stats.get("part_b_bootstrap_rng_seed", -1)) != 20260918:
        raise RuntimeError("Part B bootstrap RNG seed differs from preregistered 20260918")

    matrix = protocol.get("expected_matrix")
    if not isinstance(matrix, dict) or int(matrix.get("primary_runs", -1)) != 600 or int(matrix.get("current_runs", -1)) != 135:
        raise RuntimeError("protocol expected matrix is not 600 primary + 135 current rows")
    return seeds, freeze


def load_pair_bindings() -> dict[int, dict[str, str]]:
    manifest = read_json(PAIR_MANIFEST_PATH)
    if manifest.get("study") != "training_seed_robustness":
        raise RuntimeError("model_pair_manifest.json belongs to the wrong study")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, dict) or set(pairs) != {f"P{seed}" for seed in TRAINING_SEEDS}:
        raise RuntimeError("model_pair_manifest.json does not contain exactly P2026/P2027/P2028")
    result: dict[int, dict[str, str]] = {}
    for seed in TRAINING_SEEDS:
        pair = pairs[f"P{seed}"]
        if int(pair.get("training_seed", -1)) != seed:
            raise RuntimeError(f"P{seed} training-seed binding changed")
        row: dict[str, str] = {}
        for run_column, (section, key) in HASH_BINDINGS.items():
            section_value = pair.get(section)
            if not isinstance(section_value, dict):
                raise RuntimeError(f"P{seed}.{section} missing")
            digest = str(section_value.get(key, "")).lower()
            if len(digest) != 64:
                raise RuntimeError(f"P{seed}.{section}.{key} is not a SHA-256 digest")
            path_key = "weights_path" if key == "weights_sha256" else "config_path" if key == "config_sha256" else "path"
            artifact = PROJECT / str(section_value.get(path_key, ""))
            if not artifact.is_file() or sha256(artifact) != digest:
                raise RuntimeError(f"P{seed}.{section} artifact hash mismatch")
            row[run_column] = digest
        result[seed] = row
    return result


def _float_key(value: Any) -> float:
    return round(float(value), 10)


def primary_conditions(protocol: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    sweeps = protocol.get("primary_sweeps")
    if not isinstance(sweeps, dict):
        raise RuntimeError("protocol missing primary_sweeps")
    bias = tuple(_float_key(value) for value in sweeps["bias"]["levels_percent"])
    dropout = tuple(_float_key(value) for value in sweeps["dropout"]["durations_s"])
    drift = tuple(_float_key(value) for value in sweeps["drift"]["final_percent"])
    load = tuple(_float_key(value) for value in sweeps["load"]["post_step_load_Nm"])
    combined = tuple(
        (_float_key(bias_value), _float_key(load_value))
        for bias_value in sweeps["combined_bias_load"]["bias_percent"]
        for load_value in sweeps["combined_bias_load"]["post_step_load_Nm"]
    )
    conditions = {
        "bias": bias,
        "dropout": dropout,
        "drift": drift,
        "load": load,
        "combined_bias_load": combined,
    }
    if any(len(values) != 4 for values in conditions.values()):
        raise RuntimeError("each preregistered primary sweep must contain exactly four conditions")
    return conditions


def current_cases(protocol: dict[str, Any]) -> tuple[str, ...]:
    boundary = protocol.get("current_sensor_boundary")
    if not isinstance(boundary, dict):
        raise RuntimeError("protocol missing current_sensor_boundary")
    cases = boundary.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("current_sensor_boundary.cases must be a list")
    names = tuple(str(entry.get("name", "")) for entry in cases if isinstance(entry, dict))
    if len(names) != 3 or len(set(names)) != 3:
        raise RuntimeError("protocol must contain exactly three unique current-boundary cases")
    return names


def row_condition(row: pd.Series) -> Any:
    family = str(row["family"])
    if family == "bias":
        return _float_key(row["bias_percent"])
    if family == "dropout":
        return _float_key(row["dropout_duration_s"])
    if family == "drift":
        return _float_key(row["drift_final_percent"])
    if family == "load":
        return _float_key(row["post_step_load_Nm"])
    if family == "combined_bias_load":
        return (_float_key(row["bias_percent"]), _float_key(row["post_step_load_Nm"]))
    if family == "current_sensor":
        return str(row["current_case"])
    raise RuntimeError(f"unsupported family {family}")


def _condition_label(family: str, condition: Any) -> str:
    if family == "bias":
        return f"bias_{float(condition):g}pct"
    if family == "dropout":
        return f"dropout_{float(condition):g}s"
    if family == "drift":
        return f"drift_{float(condition):g}pct"
    if family == "load":
        return f"load_{float(condition):g}Nm"
    if family == "combined_bias_load":
        bias, load = condition
        return f"combined_bias_{float(bias):g}pct_load_{float(load):g}Nm"
    return str(condition)


def validate_matrix_and_bindings(
    runs: pd.DataFrame,
    protocol: dict[str, Any],
    simulation_seeds: tuple[int, ...],
    pair_bindings: dict[int, dict[str, str]],
) -> pd.DataFrame:
    conditions = primary_conditions(protocol)
    cases = current_cases(protocol)
    require_columns(
        runs,
        [
            "run_complete",
            "overall_rmse",
            "fault_window_rmse",
            "sensor_fault_detected",
            "detection_latency_s",
            "reliability_entries",
            "post_event_reliability_entries",
            "sub_duration_s",
            "sub_fraction",
            "recovery_events",
            *SAFETY_COLUMNS,
            *HASH_BINDINGS.keys(),
        ],
        "Part B run CSV",
    )

    for column in (
        "overall_rmse",
        "fault_window_rmse",
        "reliability_entries",
        "sub_duration_s",
        "sub_fraction",
        "recovery_events",
    ):
        runs[column] = numeric(runs, column)
    runs["post_event_reliability_entries"] = numeric(
        runs, "post_event_reliability_entries", allow_nan=True
    )
    for column in (
        "frac_phys",
        "frac_main",
        "frac_aux",
        "frac_fb",
        "event_frac_phys",
        "event_frac_main",
        "event_frac_aux",
        "event_frac_fb",
        "recovery_latency_s",
        "avg_episode_duration_s",
        "first_post_event_reliability_entry_time_s",
        "post_event_reliability_active_duration_s",
        "aux_speed_rmse_event_window",
        "aux_virtual_rmse",
        "ekf_speed_rmse_event_window",
        "ekf_event_sub_fraction",
    ):
        if column not in runs.columns:
            runs[column] = np.nan
        runs[column] = numeric(runs, column, allow_nan=True)
    runs["detection_latency_s"] = numeric(runs, "detection_latency_s", allow_nan=True)
    if "recovered_after_event" not in runs.columns:
        runs["recovered_after_event"] = pd.NA
    runs["recovered_after_event"] = bool_series(
        runs["recovered_after_event"], "recovered_after_event", allow_nan=True
    )

    # EKF-specific failure counters may be undefined on B/C3 rows in a mixed
    # DataFrame.  Those cells are structurally not applicable and count as
    # zero, while every EKF row must carry an explicit finite value.
    if "ekf_numerical_failures" not in runs.columns:
        raise KeyError("Part B run CSV missing ekf_numerical_failures/ekf_failures")
    ekf_mask = runs.controller.astype(str).eq("E_EKF_virtual_MPC")
    ekf_failure_values = pd.to_numeric(runs["ekf_numerical_failures"], errors="coerce")
    if ekf_failure_values.loc[ekf_mask].isna().any():
        raise RuntimeError("E_EKF_virtual_MPC rows are missing ekf_numerical_failures")
    runs["ekf_numerical_failures"] = ekf_failure_values.fillna(0.0)
    for column in SAFETY_COLUMNS:
        if column == "ekf_numerical_failures":
            continue
        runs[column] = numeric(runs, column)
    runs["run_complete"] = bool_series(runs["run_complete"], "run_complete")
    runs["sensor_fault_detected"] = bool_series(runs["sensor_fault_detected"], "sensor_fault_detected", allow_nan=True)

    if set(runs["training_seed"]) != set(TRAINING_SEEDS):
        raise RuntimeError("Part B run CSV training seeds are not exactly {2026,2027,2028}")
    if set(runs["simulation_seed"]) != set(simulation_seeds):
        raise RuntimeError("Part B run CSV simulation seeds differ from frozen block")
    if not set(runs["family"]).issubset(set(PRIMARY_FAMILIES) | {"current_sensor"}):
        raise RuntimeError("Part B run CSV contains an unregistered family")

    for severity_column in ("bias_percent", "dropout_duration_s", "drift_final_percent", "post_step_load_Nm"):
        if severity_column not in runs.columns:
            runs[severity_column] = np.nan
    if "current_case" not in runs.columns:
        runs["current_case"] = ""

    condition_values: list[Any] = []
    condition_labels: list[str] = []
    for _, row in runs.iterrows():
        condition = row_condition(row)
        family = str(row["family"])
        condition_values.append(condition)
        condition_labels.append(_condition_label(family, condition))
    runs["condition_key"] = condition_values
    runs["condition_label"] = condition_labels

    primary = runs[runs.family.isin(PRIMARY_FAMILIES)].copy()
    current = runs[runs.family.eq("current_sensor")].copy()
    if len(primary) != 600 or len(current) != 135 or len(runs) != 735:
        raise RuntimeError(f"expected 735 Part B rows (600 primary + 135 current), found {len(runs)} ({len(primary)} + {len(current)})")

    expected_primary = {
        (family, condition, training_seed, simulation_seed, controller)
        for family, family_conditions in conditions.items()
        for condition in family_conditions
        for training_seed in TRAINING_SEEDS
        for simulation_seed in simulation_seeds
        for controller in PRIMARY_CONTROLLERS
    }
    actual_primary = {
        (str(row.family), row.condition_key, int(row.training_seed), int(row.simulation_seed), str(row.controller))
        for row in primary.itertuples(index=False)
    }
    if actual_primary != expected_primary:
        missing = sorted(expected_primary - actual_primary, key=str)[:10]
        extra = sorted(actual_primary - expected_primary, key=str)[:10]
        raise RuntimeError(f"primary matrix mismatch; missing={missing}, extra={extra}")
    if primary.duplicated(["family", "condition_label", "training_seed", "simulation_seed", "controller"]).any():
        raise RuntimeError("duplicate primary matrix keys")

    expected_current = {
        (case, training_seed, simulation_seed, controller)
        for case in cases
        for training_seed in TRAINING_SEEDS
        for simulation_seed in simulation_seeds
        for controller in CURRENT_CONTROLLERS
    }
    actual_current = {
        (str(row.current_case), int(row.training_seed), int(row.simulation_seed), str(row.controller))
        for row in current.itertuples(index=False)
    }
    if actual_current != expected_current:
        missing = sorted(expected_current - actual_current, key=str)[:10]
        extra = sorted(actual_current - expected_current, key=str)[:10]
        raise RuntimeError(f"current-boundary matrix mismatch; missing={missing}, extra={extra}")
    if current.duplicated(["current_case", "training_seed", "simulation_seed", "controller"]).any():
        raise RuntimeError("duplicate current-boundary matrix keys")

    for seed in TRAINING_SEEDS:
        subset = runs[runs.training_seed.eq(seed)]
        expected = pair_bindings[seed]
        for column, digest in expected.items():
            actual = {str(value).lower() for value in subset[column]}
            if actual != {digest}:
                raise RuntimeError(f"training seed {seed} run binding mismatch for {column}")

    if not runs["run_complete"].all():
        # Incomplete rows remain analyzable for safety accounting only, but a
        # completed scientific matrix is required before this script may write
        # endpoint/classification outputs.
        raise RuntimeError("Part B matrix contains incomplete runs")

    # Source/recovery summaries are scientifically required for C3.  Other
    # controllers may legitimately leave C3-only source fractions undefined.
    c3 = runs[runs.controller.eq("C3_arbitration_MPC")]
    for column in (
        "post_event_reliability_entries",
        "frac_phys",
        "frac_main",
        "frac_aux",
        "frac_fb",
        "event_frac_phys",
        "event_frac_main",
        "event_frac_aux",
        "event_frac_fb",
        "avg_episode_duration_s",
    ):
        if c3[column].isna().any() or not np.isfinite(c3[column].to_numpy(float)).all():
            raise RuntimeError(f"C3 rows are missing finite {column} values")

    finite_window = c3[c3.family.isin(["bias", "dropout"])]
    if finite_window["recovered_after_event"].isna().any():
        raise RuntimeError("bias/dropout C3 rows are missing recovered_after_event recovery behavior")

    load_c3 = c3[c3.family.eq("load")]
    if (
        load_c3["post_event_reliability_active_duration_s"].isna().any()
        or not np.isfinite(load_c3["post_event_reliability_active_duration_s"].to_numpy(float)).all()
    ):
        raise RuntimeError("load C3 rows are missing post-event reliability-active duration")

    current_c3 = c3[c3.family.eq("current_sensor")]
    if (
        current_c3["aux_speed_rmse_event_window"].isna().any()
        or not np.isfinite(current_c3["aux_speed_rmse_event_window"].to_numpy(float)).all()
    ):
        raise RuntimeError("current-boundary C3 rows are missing aux_speed_rmse_event_window")
    current_ekf = runs[
        runs.family.eq("current_sensor") & runs.controller.eq("E_EKF_virtual_MPC")
    ]
    if (
        current_ekf["ekf_speed_rmse_event_window"].isna().any()
        or not np.isfinite(current_ekf["ekf_speed_rmse_event_window"].to_numpy(float)).all()
    ):
        raise RuntimeError("current-boundary EKF rows are missing ekf_speed_rmse_event_window")
    return runs


def paired_primary_points(runs: pd.DataFrame) -> pd.DataFrame:
    primary = runs[runs.family.isin(PRIMARY_FAMILIES)]
    rows: list[dict[str, Any]] = []
    group_columns = ["family", "condition_label", "training_seed", "simulation_seed"]
    for keys, group in primary.groupby(group_columns, sort=True):
        if set(group.controller) != set(PRIMARY_CONTROLLERS) or len(group) != 2:
            raise RuntimeError(f"incomplete B/C3 pairing for {keys}")
        b = group[group.controller.eq("B_plain_MPC")].iloc[0]
        c3 = group[group.controller.eq("C3_arbitration_MPC")].iloc[0]
        family, condition_label, training_seed, simulation_seed = keys
        row = {
            "family": family,
            "condition_label": condition_label,
            "training_seed": int(training_seed),
            "simulation_seed": int(simulation_seed),
            "bias_percent": c3.get("bias_percent", np.nan),
            "dropout_duration_s": c3.get("dropout_duration_s", np.nan),
            "drift_final_percent": c3.get("drift_final_percent", np.nan),
            "post_step_load_Nm": c3.get("post_step_load_Nm", np.nan),
            "b_fault_window_rmse": float(b.fault_window_rmse),
            "c3_fault_window_rmse": float(c3.fault_window_rmse),
            "c3_minus_b_fault_window_rmse": float(c3.fault_window_rmse - b.fault_window_rmse),
            "favorable_c3_minus_b": bool(c3.fault_window_rmse < b.fault_window_rmse),
            "b_overall_rmse": float(b.overall_rmse),
            "c3_overall_rmse": float(c3.overall_rmse),
            "c3_sensor_fault_detected": c3.sensor_fault_detected,
            "c3_detection_latency_s": float(c3.detection_latency_s) if pd.notna(c3.detection_latency_s) else np.nan,
            "c3_reliability_entries": int(c3.reliability_entries),
            "c3_post_event_reliability_entries": int(c3.post_event_reliability_entries),
            "c3_sub_fraction": float(c3.sub_fraction),
            "c3_frac_phys": float(c3.frac_phys),
            "c3_frac_main": float(c3.frac_main),
            "c3_frac_aux": float(c3.frac_aux),
            "c3_frac_fb": float(c3.frac_fb),
            "c3_event_frac_phys": float(c3.event_frac_phys) if pd.notna(c3.event_frac_phys) else np.nan,
            "c3_event_frac_main": float(c3.event_frac_main) if pd.notna(c3.event_frac_main) else np.nan,
            "c3_event_frac_aux": float(c3.event_frac_aux) if pd.notna(c3.event_frac_aux) else np.nan,
            "c3_event_frac_fb": float(c3.event_frac_fb) if pd.notna(c3.event_frac_fb) else np.nan,
            "c3_recovery_events": int(c3.recovery_events),
            "c3_recovery_latency_s": float(c3.recovery_latency_s) if pd.notna(c3.recovery_latency_s) else np.nan,
            "c3_recovered_after_event": c3.recovered_after_event,
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 300:
        raise RuntimeError(f"expected 300 paired primary points, found {len(result)}")
    return result.sort_values(["family", "condition_label", "training_seed", "simulation_seed"]).reset_index(drop=True)


def training_seed_means(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, condition_label, training_seed), group in points.groupby(
        ["family", "condition_label", "training_seed"], sort=True
    ):
        if len(group) != 5:
            raise RuntimeError("each family/condition/training-seed summary must contain five paired simulations")
        delta = group.c3_minus_b_fault_window_rmse.to_numpy(float)
        detected = group.c3_sensor_fault_detected.dropna().astype(bool)
        rows.append(
            {
                "family": family,
                "condition_label": condition_label,
                "training_seed": int(training_seed),
                "paired_count": int(len(group)),
                "mean_b_fault_window_rmse": float(group.b_fault_window_rmse.mean()),
                "mean_c3_fault_window_rmse": float(group.c3_fault_window_rmse.mean()),
                "mean_c3_minus_b_fault_window_rmse": float(np.mean(delta)),
                "sample_sd_c3_minus_b_fault_window_rmse": float(np.std(delta, ddof=1)),
                "favorable_pair_count": int(np.sum(delta < 0.0)),
                "mean_effect_favorable": bool(np.mean(delta) < 0.0),
                "c3_detection_probability": float(detected.mean()) if len(detected) else np.nan,
                "c3_mean_sub_fraction": float(group.c3_sub_fraction.mean()),
                "c3_mean_frac_phys": float(group.c3_frac_phys.mean()),
                "c3_mean_frac_main": float(group.c3_frac_main.mean()),
                "c3_mean_frac_aux": float(group.c3_frac_aux.mean()),
                "c3_mean_frac_fb": float(group.c3_frac_fb.mean()),
                "c3_mean_event_frac_phys": float(group.c3_event_frac_phys.mean()),
                "c3_mean_event_frac_main": float(group.c3_event_frac_main.mean()),
                "c3_mean_event_frac_aux": float(group.c3_event_frac_aux.mean()),
                "c3_mean_event_frac_fb": float(group.c3_event_frac_fb.mean()),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 60:
        raise RuntimeError(f"expected 60 per-training-seed severity summaries, found {len(result)}")
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if total <= 0:
        return (np.nan, np.nan, np.nan)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return p, max(0.0, center - half), min(1.0, center + half)


def detection_wilson_table(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    detected_families = ("bias", "dropout", "drift", "combined_bias_load")
    subset = points[points.family.isin(detected_families)]
    for (family, condition_label), group in subset.groupby(["family", "condition_label"], sort=True):
        for training_seed in (*TRAINING_SEEDS, "pooled"):
            block = group if training_seed == "pooled" else group[group.training_seed.eq(training_seed)]
            values = block.c3_sensor_fault_detected.dropna().astype(bool)
            expected_n = 15 if training_seed == "pooled" else 5
            if len(values) != expected_n:
                raise RuntimeError(f"detection endpoint missing observations for {family}/{condition_label}/{training_seed}")
            successes = int(values.sum())
            p, low, high = wilson_interval(successes, len(values))
            rows.append(
                {
                    "endpoint": "sensor_fault_detection_probability",
                    "family": family,
                    "condition_label": condition_label,
                    "training_seed": training_seed,
                    "successes": successes,
                    "trials": int(len(values)),
                    "probability": p,
                    "wilson95_low": low,
                    "wilson95_high": high,
                }
            )

    load = points[points.family.eq("load")]
    for condition_label, group in load.groupby("condition_label", sort=True):
        for training_seed in (*TRAINING_SEEDS, "pooled"):
            block = group if training_seed == "pooled" else group[group.training_seed.eq(training_seed)]
            flags = block.c3_post_event_reliability_entries.to_numpy(int) > 0
            expected_n = 15 if training_seed == "pooled" else 5
            if len(flags) != expected_n:
                raise RuntimeError(f"load false-entry endpoint missing observations for {condition_label}/{training_seed}")
            successes = int(np.sum(flags))
            p, low, high = wilson_interval(successes, len(flags))
            rows.append(
                {
                    "endpoint": "load_false_reliability_entry_probability",
                    "family": "load",
                    "condition_label": condition_label,
                    "training_seed": training_seed,
                    "successes": successes,
                    "trials": int(len(flags)),
                    "probability": p,
                    "wilson95_low": low,
                    "wilson95_high": high,
                }
            )
    return pd.DataFrame(rows)


def hierarchical_bootstrap_pairs(
    points: pd.DataFrame,
    *,
    analysis_seed: int,
    replicates: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(analysis_seed)
    rows: list[dict[str, Any]] = []
    for (family, condition_label), group in points.groupby(["family", "condition_label"], sort=True):
        by_seed = {
            seed: group[group.training_seed.eq(seed)]
            .sort_values("simulation_seed")
            .c3_minus_b_fault_window_rmse.to_numpy(float)
            for seed in TRAINING_SEEDS
        }
        if any(len(values) != 5 for values in by_seed.values()):
            raise RuntimeError(f"bootstrap input incomplete for {family}/{condition_label}")
        seed_means = np.asarray([np.mean(by_seed[seed]) for seed in TRAINING_SEEDS], dtype=float)
        boot = np.empty(replicates, dtype=float)
        for index in range(replicates):
            sampled_training = rng.choice(TRAINING_SEEDS, size=len(TRAINING_SEEDS), replace=True)
            cluster_means = []
            for training_seed in sampled_training:
                values = by_seed[int(training_seed)]
                sampled = rng.choice(values, size=len(values), replace=True)
                cluster_means.append(float(np.mean(sampled)))
            boot[index] = float(np.mean(cluster_means))
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "endpoint": "mean_paired_c3_minus_b_fault_window_rmse",
                "family": family,
                "condition_label": condition_label,
                "point_estimate_mean_of_training_seed_means": float(np.mean(seed_means)),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_probability_delta_lt_zero": float(np.mean(boot < 0.0)),
                "analysis_seed": int(analysis_seed),
                "bootstrap_replicates": int(replicates),
                "training_seed_clusters": 3,
                "simulation_seeds_per_cluster": 5,
            }
        )
    return pd.DataFrame(rows)


def source_recovery_summary(runs: pd.DataFrame) -> pd.DataFrame:
    c3 = runs[runs.controller.eq("C3_arbitration_MPC") & runs.family.isin(PRIMARY_FAMILIES)].copy()
    rows: list[dict[str, Any]] = []
    for (family, condition_label, training_seed), group in c3.groupby(
        ["family", "condition_label", "training_seed"], sort=True
    ):
        detected = group.sensor_fault_detected.dropna().astype(bool)
        finite_detection = group.detection_latency_s[np.isfinite(group.detection_latency_s)]
        finite_recovery = group.recovery_latency_s[np.isfinite(group.recovery_latency_s)]
        finite_post_event_duration = group.post_event_reliability_active_duration_s[
            np.isfinite(group.post_event_reliability_active_duration_s)
        ]
        recovered = group.recovered_after_event.dropna().astype(bool)
        rows.append(
            {
                "family": family,
                "condition_label": condition_label,
                "training_seed": int(training_seed),
                "run_count": int(len(group)),
                "detection_probability": float(detected.mean()) if len(detected) else np.nan,
                "mean_detection_latency_s_when_detected": float(finite_detection.mean()) if len(finite_detection) else np.nan,
                "mean_substitution_fraction": float(group.sub_fraction.mean()),
                "mean_frac_phys": float(group.frac_phys.mean()),
                "mean_frac_main": float(group.frac_main.mean()),
                "mean_frac_aux": float(group.frac_aux.mean()),
                "mean_frac_fb": float(group.frac_fb.mean()),
                "mean_event_frac_phys": float(group.event_frac_phys.mean()),
                "mean_event_frac_main": float(group.event_frac_main.mean()),
                "mean_event_frac_aux": float(group.event_frac_aux.mean()),
                "mean_event_frac_fb": float(group.event_frac_fb.mean()),
                "total_recovery_events": int(group.recovery_events.sum()),
                "recovery_probability_when_defined": float(recovered.mean()) if len(recovered) else np.nan,
                "mean_recovery_latency_s_when_defined": float(finite_recovery.mean()) if len(finite_recovery) else np.nan,
                "mean_post_event_reliability_active_duration_s_when_defined": (
                    float(finite_post_event_duration.mean()) if len(finite_post_event_duration) else np.nan
                ),
                "mean_total_sub_duration_s": float(group.sub_duration_s.mean()),
                "mean_source_switch_episode_duration_s": float(group.avg_episode_duration_s.mean()),
            }
        )
    return pd.DataFrame(rows)


def load_summary(runs: pd.DataFrame, points: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    load_step = float(protocol["primary_sweeps"]["load"]["step_time_s"])
    c3 = runs[runs.family.eq("load") & runs.controller.eq("C3_arbitration_MPC")].copy()
    rows: list[dict[str, Any]] = []
    for condition_label in sorted(c3.condition_label.unique()):
        condition_runs = c3[c3.condition_label.eq(condition_label)]
        condition_points = points[points.condition_label.eq(condition_label) & points.family.eq("load")]
        for training_seed in (*TRAINING_SEEDS, "pooled"):
            block = condition_runs if training_seed == "pooled" else condition_runs[condition_runs.training_seed.eq(training_seed)]
            pair_block = condition_points if training_seed == "pooled" else condition_points[condition_points.training_seed.eq(training_seed)]
            flags = block.post_event_reliability_entries.to_numpy(int) > 0
            probability, low, high = wilson_interval(int(flags.sum()), len(flags))
            entry_times = block.first_post_event_reliability_entry_time_s.to_numpy(float)
            latencies = entry_times[np.isfinite(entry_times)] - load_step
            post_event_durations = block.post_event_reliability_active_duration_s.to_numpy(float)
            post_event_durations = post_event_durations[np.isfinite(post_event_durations)]
            rows.append(
                {
                    "family": "load",
                    "condition_label": condition_label,
                    "training_seed": training_seed,
                    "run_count": int(len(block)),
                    "false_entry_runs": int(flags.sum()),
                    "false_entry_probability": probability,
                    "false_entry_wilson95_low": low,
                    "false_entry_wilson95_high": high,
                    "total_post_event_reliability_entries": int(block.post_event_reliability_entries.sum()),
                    "total_reliability_entries_all_times": int(block.reliability_entries.sum()),
                    "mean_substitution_fraction": float(block.sub_fraction.mean()),
                    "mean_entry_latency_from_load_step_s_when_entry": float(np.mean(latencies)) if len(latencies) else np.nan,
                    "mean_post_event_reliability_active_duration_s_when_defined": (
                        float(np.mean(post_event_durations)) if len(post_event_durations) else np.nan
                    ),
                    "mean_total_sub_duration_s": float(block.sub_duration_s.mean()),
                    "mean_source_switch_episode_duration_s": float(block.avg_episode_duration_s.mean()),
                    "mean_c3_fault_window_rmse": float(pair_block.c3_fault_window_rmse.mean()),
                    "mean_b_fault_window_rmse": float(pair_block.b_fault_window_rmse.mean()),
                    "mean_tracking_penalty_c3_minus_b": float(pair_block.c3_minus_b_fault_window_rmse.mean()),
                    "favorable_pair_count": int(pair_block.favorable_c3_minus_b.sum()),
                }
            )
    return pd.DataFrame(rows)


def current_paired_points(runs: pd.DataFrame) -> pd.DataFrame:
    current = runs[runs.family.eq("current_sensor")].copy()
    rows: list[dict[str, Any]] = []
    for (current_case, training_seed, simulation_seed), group in current.groupby(
        ["current_case", "training_seed", "simulation_seed"], sort=True
    ):
        if set(group.controller) != set(CURRENT_CONTROLLERS) or len(group) != 3:
            raise RuntimeError("current-boundary controller triplet is incomplete")
        b = group[group.controller.eq("B_plain_MPC")].iloc[0]
        c3 = group[group.controller.eq("C3_arbitration_MPC")].iloc[0]
        ekf = group[group.controller.eq("E_EKF_virtual_MPC")].iloc[0]
        b_rmse = float(b.fault_window_rmse)
        row = {
            "current_case": current_case,
            "training_seed": int(training_seed),
            "simulation_seed": int(simulation_seed),
            "b_fault_window_rmse": b_rmse,
            "c3_fault_window_rmse": float(c3.fault_window_rmse),
            "ekf_fault_window_rmse": float(ekf.fault_window_rmse),
            "c3_minus_b_fault_window_rmse": float(c3.fault_window_rmse - b_rmse),
            "ekf_minus_b_fault_window_rmse": float(ekf.fault_window_rmse - b_rmse),
            "c3_to_b_fault_window_rmse_ratio": float(c3.fault_window_rmse / b_rmse) if abs(b_rmse) > 1e-12 else np.nan,
            "ekf_to_b_fault_window_rmse_ratio": float(ekf.fault_window_rmse / b_rmse) if abs(b_rmse) > 1e-12 else np.nan,
            "c3_sub_fraction": float(c3.sub_fraction),
            "c3_frac_phys": float(c3.frac_phys),
            "c3_frac_main": float(c3.frac_main),
            "c3_frac_aux": float(c3.frac_aux),
            "c3_frac_fb": float(c3.frac_fb),
            "c3_event_frac_phys": float(c3.event_frac_phys) if pd.notna(c3.event_frac_phys) else np.nan,
            "c3_event_frac_main": float(c3.event_frac_main) if pd.notna(c3.event_frac_main) else np.nan,
            "c3_event_frac_aux": float(c3.event_frac_aux) if pd.notna(c3.event_frac_aux) else np.nan,
            "c3_event_frac_fb": float(c3.event_frac_fb) if pd.notna(c3.event_frac_fb) else np.nan,
            "c3_aux_estimator_rmse": (
                float(c3.aux_speed_rmse_event_window)
                if "aux_speed_rmse_event_window" in c3.index and pd.notna(c3.aux_speed_rmse_event_window)
                else float(c3.aux_virtual_rmse)
                if "aux_virtual_rmse" in c3.index and pd.notna(c3.aux_virtual_rmse)
                else np.nan
            ),
            "ekf_estimator_rmse": float(ekf.ekf_speed_rmse_event_window) if "ekf_speed_rmse_event_window" in ekf.index and pd.notna(ekf.ekf_speed_rmse_event_window) else np.nan,
            "ekf_sub_fraction": float(ekf.sub_fraction),
            "ekf_event_sub_fraction": (
                float(ekf.ekf_event_sub_fraction)
                if "ekf_event_sub_fraction" in ekf.index and pd.notna(ekf.ekf_event_sub_fraction)
                else np.nan
            ),
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 45:
        raise RuntimeError(f"expected 45 paired current-boundary triplets, found {len(result)}")
    return result


def current_summary(runs: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    current = runs[runs.family.eq("current_sensor")]
    rows: list[dict[str, Any]] = []
    for (current_case, controller, training_seed), group in current.groupby(
        ["current_case", "controller", "training_seed"], sort=True
    ):
        estimator = np.full(len(group), np.nan)
        if controller == "C3_arbitration_MPC" and "aux_speed_rmse_event_window" in group.columns:
            preferred = pd.to_numeric(group.aux_speed_rmse_event_window, errors="coerce").to_numpy(float)
            fallback = pd.to_numeric(group.aux_virtual_rmse, errors="coerce").to_numpy(float)
            estimator = np.where(np.isfinite(preferred), preferred, fallback)
        elif controller == "E_EKF_virtual_MPC" and "ekf_speed_rmse_event_window" in group.columns:
            estimator = pd.to_numeric(group.ekf_speed_rmse_event_window, errors="coerce").to_numpy(float)
        finite_estimator = estimator[np.isfinite(estimator)]
        rows.append(
            {
                "current_case": current_case,
                "controller": controller,
                "training_seed": int(training_seed),
                "run_count": int(len(group)),
                "mean_overall_rmse": float(group.overall_rmse.mean()),
                "mean_fault_window_rmse": float(group.fault_window_rmse.mean()),
                "mean_estimator_rmse_when_defined": float(np.mean(finite_estimator)) if len(finite_estimator) else np.nan,
                "mean_sub_fraction": float(group.sub_fraction.mean()),
                "mean_frac_phys": float(group.frac_phys.mean()),
                "mean_frac_main": float(group.frac_main.mean()),
                "mean_frac_aux": float(group.frac_aux.mean()),
                "mean_frac_fb": float(group.frac_fb.mean()),
                "mean_event_frac_phys": float(group.event_frac_phys.mean()),
                "mean_event_frac_main": float(group.event_frac_main.mean()),
                "mean_event_frac_aux": float(group.event_frac_aux.mean()),
                "mean_event_frac_fb": float(group.event_frac_fb.mean()),
                "mean_ekf_event_sub_fraction_when_defined": (
                    float(pd.to_numeric(group.ekf_event_sub_fraction, errors="coerce").mean())
                    if controller == "E_EKF_virtual_MPC" and group.ekf_event_sub_fraction.notna().any()
                    else np.nan
                ),
                "safety_failure_total": int(sum(pd.to_numeric(group[column]).sum() for column in SAFETY_COLUMNS)),
            }
        )

    for (current_case, training_seed), group in points.groupby(["current_case", "training_seed"], sort=True):
        rows.append(
            {
                "current_case": current_case,
                "controller": "paired_effects",
                "training_seed": int(training_seed),
                "run_count": int(len(group)),
                "mean_overall_rmse": np.nan,
                "mean_fault_window_rmse": np.nan,
                "mean_estimator_rmse_when_defined": np.nan,
                "mean_sub_fraction": float(group.c3_sub_fraction.mean()),
                "mean_frac_phys": float(group.c3_frac_phys.mean()),
                "mean_frac_main": float(group.c3_frac_main.mean()),
                "mean_frac_aux": float(group.c3_frac_aux.mean()),
                "mean_frac_fb": float(group.c3_frac_fb.mean()),
                "mean_event_frac_phys": float(group.c3_event_frac_phys.mean()),
                "mean_event_frac_main": float(group.c3_event_frac_main.mean()),
                "mean_event_frac_aux": float(group.c3_event_frac_aux.mean()),
                "mean_event_frac_fb": float(group.c3_event_frac_fb.mean()),
                "mean_ekf_event_sub_fraction_when_defined": float(group.ekf_event_sub_fraction.mean()),
                "safety_failure_total": 0,
                "mean_c3_minus_b_fault_window_rmse": float(group.c3_minus_b_fault_window_rmse.mean()),
                "mean_ekf_minus_b_fault_window_rmse": float(group.ekf_minus_b_fault_window_rmse.mean()),
                "mean_c3_to_b_rmse_ratio": float(group.c3_to_b_fault_window_rmse_ratio.mean()),
                "mean_ekf_to_b_rmse_ratio": float(group.ekf_to_b_fault_window_rmse_ratio.mean()),
            }
        )
    return pd.DataFrame(rows)


def current_bootstrap(points: pd.DataFrame, *, analysis_seed: int, replicates: int) -> pd.DataFrame:
    rng = np.random.default_rng(analysis_seed)
    rows: list[dict[str, Any]] = []
    for current_case, case_points in points.groupby("current_case", sort=True):
        for effect_column in ("c3_minus_b_fault_window_rmse", "ekf_minus_b_fault_window_rmse"):
            by_seed = {
                seed: case_points[case_points.training_seed.eq(seed)]
                .sort_values("simulation_seed")[effect_column]
                .to_numpy(float)
                for seed in TRAINING_SEEDS
            }
            if any(len(values) != 5 for values in by_seed.values()):
                raise RuntimeError(f"current bootstrap input incomplete for {current_case}/{effect_column}")
            boot = np.empty(replicates, dtype=float)
            for index in range(replicates):
                sampled_training = rng.choice(TRAINING_SEEDS, size=3, replace=True)
                means = []
                for training_seed in sampled_training:
                    values = by_seed[int(training_seed)]
                    means.append(float(np.mean(rng.choice(values, size=5, replace=True))))
                boot[index] = float(np.mean(means))
            seed_means = [float(np.mean(by_seed[seed])) for seed in TRAINING_SEEDS]
            low, high = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "current_case": current_case,
                    "endpoint": effect_column,
                    "point_estimate_mean_of_training_seed_means": float(np.mean(seed_means)),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "bootstrap_probability_effect_lt_zero": float(np.mean(boot < 0.0)),
                    "analysis_seed": int(analysis_seed),
                    "bootstrap_replicates": int(replicates),
                    "training_seed_clusters": 3,
                    "simulation_seeds_per_cluster": 5,
                }
            )
    return pd.DataFrame(rows)


def combined_effects(points: pd.DataFrame) -> pd.DataFrame:
    combined = points[points.family.eq("combined_bias_load")]
    rows: list[dict[str, Any]] = []
    for condition_label, group in combined.groupby("condition_label", sort=True):
        delta = group.c3_minus_b_fault_window_rmse.to_numpy(float)
        rows.append(
            {
                "condition_label": condition_label,
                "aggregation_axis": "pooled",
                "aggregation_value": "all",
                "mean_c3_minus_b_fault_window_rmse": float(np.mean(delta)),
                "favorable_count": int(np.sum(delta < 0.0)),
                "total_count": int(len(delta)),
            }
        )
        for training_seed, seed_group in group.groupby("training_seed", sort=True):
            values = seed_group.c3_minus_b_fault_window_rmse.to_numpy(float)
            rows.append(
                {
                    "condition_label": condition_label,
                    "aggregation_axis": "training_seed",
                    "aggregation_value": str(int(training_seed)),
                    "mean_c3_minus_b_fault_window_rmse": float(np.mean(values)),
                    "favorable_count": int(np.sum(values < 0.0)),
                    "total_count": int(len(values)),
                }
            )
        for simulation_seed, sim_group in group.groupby("simulation_seed", sort=True):
            values = sim_group.c3_minus_b_fault_window_rmse.to_numpy(float)
            rows.append(
                {
                    "condition_label": condition_label,
                    "aggregation_axis": "simulation_seed",
                    "aggregation_value": str(int(simulation_seed)),
                    "mean_c3_minus_b_fault_window_rmse": float(np.mean(values)),
                    "favorable_count": int(np.sum(values < 0.0)),
                    "total_count": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def safety_summary(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    for family, group in runs.groupby("family", sort=True):
        row: dict[str, Any] = {"family": family, "run_count": int(len(group))}
        for column in SAFETY_COLUMNS:
            row[column] = int(pd.to_numeric(group[column]).sum())
        row["incomplete_runs"] = int((~group.run_complete.astype(bool)).sum())
        row["total_failure_safety_count"] = int(sum(row[column] for column in SAFETY_COLUMNS) + row["incomplete_runs"])
        rows.append(row)
    table = pd.DataFrame(rows)
    totals = {column: int(pd.to_numeric(runs[column]).sum()) for column in SAFETY_COLUMNS}
    totals["incomplete_runs"] = int((~runs.run_complete.astype(bool)).sum())
    totals["total_failure_safety_count"] = int(sum(totals.values()))
    return table, totals


def condition_safety_total(runs: pd.DataFrame, family: str, condition_label: str) -> int:
    block = runs[runs.family.eq(family) & runs.condition_label.eq(condition_label)]
    total = sum(int(pd.to_numeric(block[column]).sum()) for column in SAFETY_COLUMNS)
    total += int((~block.run_complete.astype(bool)).sum())
    return int(total)


def operating_envelope(
    runs: pd.DataFrame,
    points: pd.DataFrame,
    seed_means: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    rules = protocol.get("operating_envelope_rules")
    if not isinstance(rules, dict):
        raise RuntimeError("protocol missing operating_envelope_rules")
    supported = rules.get("SUPPORTED")
    limited = rules.get("LIMITED")
    if not isinstance(supported, dict) or not isinstance(limited, dict):
        raise RuntimeError("protocol operating-envelope rule blocks are malformed")
    supported_detection_min = float(supported["per_training_seed_detection_probability_min"])
    supported_favorable_count = int(supported["favorable_paired_sign_count_min_of_15"])
    limited_detection_min = float(limited["pooled_detection_probability_min"])
    limited_favorable_seed_count = int(limited["training_seeds_with_favorable_mean_min_of_3"])
    if tuple(rules.get("applicable_families", [])) != CLASSIFIED_FAMILIES:
        raise RuntimeError("operating-envelope family set changed from frozen preregistration")
    if supported_detection_min != 0.8 or supported_favorable_count != 12:
        raise RuntimeError("SUPPORTED numeric thresholds changed from frozen preregistration")
    if limited_detection_min != 0.5 or limited_favorable_seed_count != 2:
        raise RuntimeError("LIMITED numeric thresholds changed from frozen preregistration")
    if supported.get("per_training_seed_mean_paired_delta_C3_minus_B") != "< 0 for all 3 training seeds":
        raise RuntimeError("SUPPORTED paired-delta rule changed from frozen preregistration")
    if int(supported.get("safety_numerical_incomplete_failures", -1)) != 0:
        raise RuntimeError("SUPPORTED safety rule changed from frozen preregistration")
    if limited.get("pooled_mean_paired_delta_C3_minus_B") != "< 0":
        raise RuntimeError("LIMITED pooled paired-delta rule changed from frozen preregistration")
    if int(limited.get("safety_numerical_incomplete_failures", -1)) != 0:
        raise RuntimeError("LIMITED safety rule changed from frozen preregistration")
    if limited.get("only_if_not_SUPPORTED") is not True:
        raise RuntimeError("LIMITED precedence rule changed from frozen preregistration")

    rows: list[dict[str, Any]] = []
    for (family, condition_label), group in points[points.family.isin(CLASSIFIED_FAMILIES)].groupby(
        ["family", "condition_label"], sort=True
    ):
        seed_group = seed_means[
            seed_means.family.eq(family) & seed_means.condition_label.eq(condition_label)
        ].sort_values("training_seed")
        if len(seed_group) != 3:
            raise RuntimeError(f"classification seed summaries incomplete for {family}/{condition_label}")
        per_seed_detection = {
            int(row.training_seed): float(row.c3_detection_probability)
            for row in seed_group.itertuples(index=False)
        }
        if any(not np.isfinite(value) for value in per_seed_detection.values()):
            raise RuntimeError(f"classification detection data missing for {family}/{condition_label}")
        per_seed_means = {
            int(row.training_seed): float(row.mean_c3_minus_b_fault_window_rmse)
            for row in seed_group.itertuples(index=False)
        }
        pooled_detection = float(group.c3_sensor_fault_detected.astype(bool).mean())
        pooled_delta = float(group.c3_minus_b_fault_window_rmse.mean())
        favorable_count = int(group.favorable_c3_minus_b.sum())
        favorable_seed_count = int(sum(value < 0.0 for value in per_seed_means.values()))
        safety_total = condition_safety_total(runs, family, condition_label)

        supported_checks = {
            "all_training_seed_detection_probability_ge_threshold": all(
                value >= supported_detection_min for value in per_seed_detection.values()
            ),
            "all_training_seed_mean_delta_negative": all(value < 0.0 for value in per_seed_means.values()),
            "favorable_pair_count_ge_threshold": favorable_count >= supported_favorable_count,
            "safety_numerical_incomplete_failures_zero": safety_total == 0,
        }
        limited_checks = {
            "pooled_detection_probability_ge_threshold": pooled_detection >= limited_detection_min,
            "pooled_mean_delta_negative": pooled_delta < 0.0,
            "favorable_training_seed_mean_count_ge_threshold": favorable_seed_count >= limited_favorable_seed_count,
            "safety_numerical_incomplete_failures_zero": safety_total == 0,
        }
        if all(supported_checks.values()):
            classification = "SUPPORTED"
        elif all(limited_checks.values()):
            classification = "LIMITED"
        else:
            classification = "UNSUPPORTED"

        rows.append(
            {
                "family": family,
                "condition_label": condition_label,
                "classification": classification,
                "pooled_detection_probability": pooled_detection,
                "training_seed_2026_detection_probability": per_seed_detection[2026],
                "training_seed_2027_detection_probability": per_seed_detection[2027],
                "training_seed_2028_detection_probability": per_seed_detection[2028],
                "pooled_mean_c3_minus_b_fault_window_rmse": pooled_delta,
                "training_seed_2026_mean_delta": per_seed_means[2026],
                "training_seed_2027_mean_delta": per_seed_means[2027],
                "training_seed_2028_mean_delta": per_seed_means[2028],
                "favorable_pair_count_of_15": favorable_count,
                "training_seeds_with_favorable_mean_of_3": favorable_seed_count,
                "safety_numerical_incomplete_failure_total": safety_total,
                "supported_checks_json": json.dumps(supported_checks, sort_keys=True),
                "limited_checks_json": json.dumps(limited_checks, sort_keys=True),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 16:
        raise RuntimeError(f"expected 16 classified severity conditions, found {len(result)}")
    return result


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = RESULT_ROOT / OUTPUT_FILES[name]
    frame.to_csv(path, index=False)
    return path


def locate_runs_path(protocol: dict[str, Any], requested: str | None) -> Path:
    if requested:
        candidate = Path(requested)
        path = candidate if candidate.is_absolute() else PROJECT / candidate
        return path.resolve()
    outputs = protocol.get("outputs")
    if isinstance(outputs, dict) and isinstance(outputs.get("part_b_runs"), str):
        return (PROJECT / outputs["part_b_runs"]).resolve()
    candidates = [
        DEFAULT_RUNS_PATH,
        RESULT_ROOT / "part_b_runs.csv",
        RESULT_ROOT / "severity_current_runs.csv",
        RESULT_ROOT / "final_robustness_study_runs.csv",
    ]
    existing = [path.resolve() for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(f"multiple candidate Part B run CSVs exist; pass --runs-csv explicitly: {[rel(path) for path in existing]}")
    return DEFAULT_RUNS_PATH.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-csv",
        default=None,
        help="completed Part B run CSV; defaults to protocol output or final_robustness_runs.csv",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing Part B analysis summaries")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = read_json(PROTOCOL_PATH)
    simulation_seeds, seed_freeze = validate_protocol(protocol)
    pair_bindings = load_pair_bindings()
    runs_path = locate_runs_path(protocol, args.runs_csv)
    if not runs_path.is_file():
        raise FileNotFoundError(
            f"completed Part B run CSV does not exist at {rel(runs_path)}; do not run analysis before scientific outputs exist"
        )

    output_paths = [RESULT_ROOT / filename for filename in OUTPUT_FILES.values()]
    summary_path = RESULT_ROOT / "part_b_analysis_summary.json"
    existing = [path for path in [*output_paths, summary_path] if path.exists()]
    if existing and not args.overwrite:
        raise RuntimeError(
            "Part B analysis outputs already exist; use --overwrite only for intentional deterministic re-analysis: "
            + ", ".join(rel(path) for path in existing)
        )

    runs = canonicalize_runs(pd.read_csv(runs_path, float_precision="round_trip"))
    runs = validate_matrix_and_bindings(runs, protocol, simulation_seeds, pair_bindings)
    points = paired_primary_points(runs)
    seed_means = training_seed_means(points)
    wilson = detection_wilson_table(points)

    stats = protocol["statistics"]
    analysis_seed = int(stats["part_b_bootstrap_rng_seed"])
    replicates = int(stats["bootstrap_replicates"])
    bootstrap = hierarchical_bootstrap_pairs(points, analysis_seed=analysis_seed, replicates=replicates)
    sources = source_recovery_summary(runs)
    load = load_summary(runs, points, protocol)
    current_points = current_paired_points(runs)
    current = current_summary(runs, current_points)
    current_boot = current_bootstrap(current_points, analysis_seed=analysis_seed, replicates=replicates)
    combined = combined_effects(points)
    safety, safety_totals = safety_summary(runs)
    envelope = operating_envelope(runs, points, seed_means, protocol)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    tables = {
        "paired_points": points,
        "training_seed_means": seed_means,
        "detection_wilson": wilson,
        "severity_bootstrap": bootstrap,
        "source_recovery": sources,
        "load_summary": load,
        "current_paired_points": current_points,
        "current_summary": current,
        "current_bootstrap": current_boot,
        "combined_effects": combined,
        "safety": safety,
        "operating_envelope": envelope,
    }
    written = {name: write_csv(frame, name) for name, frame in tables.items()}

    summary = {
        "study": "final_robustness_hardening",
        "analysis_role": "part_b_post_evaluation_machine_readable_analysis",
        "source": {
            "part_b_runs_csv": {"path": rel(runs_path), "sha256": sha256(runs_path)},
            "study_protocol_json": {"path": rel(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
            "simulation_seed_freeze_json": {
                "path": str(protocol["simulation_seed_freeze"]["path"]),
                "sha256": sha256(PROJECT / str(protocol["simulation_seed_freeze"]["path"])),
            },
            "model_pair_manifest_json": {"path": rel(PAIR_MANIFEST_PATH), "sha256": sha256(PAIR_MANIFEST_PATH)},
        },
        "matrix": {
            "training_seeds": list(TRAINING_SEEDS),
            "simulation_seeds": list(simulation_seeds),
            "primary_controllers": list(PRIMARY_CONTROLLERS),
            "current_controllers": list(CURRENT_CONTROLLERS),
            "primary_rows": int(runs.family.isin(PRIMARY_FAMILIES).sum()),
            "current_rows": int(runs.family.eq("current_sensor").sum()),
            "total_rows": int(len(runs)),
            "paired_primary_points": int(len(points)),
            "current_controller_triplets": int(len(current_points)),
            "exact_cartesian_matrix_verified": True,
            "model_calibration_hash_bindings_verified": True,
        },
        "statistics": {
            "hierarchical_bootstrap_order": ["training_seed", "simulation_seed"],
            "part_b_bootstrap_rng_seed": analysis_seed,
            "bootstrap_replicates": replicates,
            "bootstrap_interval": "descriptive percentile 95% CI",
            "training_seed_level_limitation": (
                "Only three training seeds are available; hierarchical intervals are descriptive and do not support "
                "high-precision population inference about training-initialization variability."
            ),
            "detection_interval": "Wilson 95%",
        },
        "primary_endpoints": list(protocol["primary_endpoints"]),
        "severity_training_seed_means": json_ready(seed_means.to_dict(orient="records")),
        "detection_and_false_entry_wilson": json_ready(wilson.to_dict(orient="records")),
        "severity_bootstrap": json_ready(bootstrap.to_dict(orient="records")),
        "load_false_entry": json_ready(load.to_dict(orient="records")),
        "combined_grid_effects": json_ready(combined.to_dict(orient="records")),
        "current_sensor_boundary": {
            "claim_boundary": protocol["current_sensor_boundary"]["claim_boundary"],
            "summary": json_ready(current.to_dict(orient="records")),
            "bootstrap": json_ready(current_boot.to_dict(orient="records")),
        },
        "operating_envelope": {
            "rules_source": "results/final_robustness/study_protocol.json:operating_envelope_rules",
            "classifications": json_ready(envelope.to_dict(orient="records")),
            "load_rule": protocol["operating_envelope_rules"]["load_disturbance"],
            "current_rule": protocol["operating_envelope_rules"]["current_sensor_boundary"],
        },
        "safety": {
            "totals_across_part_b_runs": safety_totals,
            "all_failure_safety_counts_zero": safety_totals["total_failure_safety_count"] == 0,
        },
        "seed_provenance_conclusion": seed_freeze["provenance_scan"].get("conclusion"),
        "outputs": {
            name: {"path": rel(path), "sha256": sha256(path)}
            for name, path in written.items()
        },
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved Part B machine-readable analysis to {rel(RESULT_ROOT)}")


if __name__ == "__main__":
    main()
