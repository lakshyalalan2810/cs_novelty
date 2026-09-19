"""Independent verifier for the final frozen-V3 robustness hardening study.

The verifier is intentionally downstream of the scientific runner/analyzers.
It does not train, calibrate, simulate, retune, or rewrite study artifacts.  It
fails closed when required outputs are absent or when a frozen binding, matrix
cell, statistic, classification, or safety total differs from an independent
recalculation.

The Part B scientific matrix may not exist while this file is first written.
That is expected: ``py_compile``/Ruff can validate this script before the study
runs, while executing this verifier must fail until the required raw outputs
are present.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
RESULT_ROOT = PROJECT / "results" / "final_robustness"

PROTOCOL_PATH = RESULT_ROOT / "study_protocol.json"
PRESTUDY_MANIFEST_PATH = RESULT_ROOT / "prestudy_frozen_hash_manifest.json"
SEED_FREEZE_PATH = RESULT_ROOT / "simulation_seed_freeze.json"
PAIR_MANIFEST_PATH = PROJECT / "results" / "training_seed_robustness" / "model_pair_manifest.json"
TRAINING_VERIFIER_PATH = PROJECT / "scripts" / "verify_training_seed_robustness.py"

PART_A_RUNS_PATH = PROJECT / "results" / "training_seed_robustness" / "training_seed_runs.csv"
PART_A_DECOMPOSITION_PATH = RESULT_ROOT / "variance_decomposition.csv"
PART_A_BOOTSTRAP_PATH = RESULT_ROOT / "variance_bootstrap.csv"
PART_A_SUMMARY_PATH = RESULT_ROOT / "variance_decomposition_summary.json"

PART_B_RUNS_PATH = RESULT_ROOT / "final_robustness_runs.csv"
PART_B_SUMMARY_PATH = RESULT_ROOT / "part_b_analysis_summary.json"
FINAL_REPORT_PATH = PROJECT / "FINAL_ROBUSTNESS_AND_SEVERITY_STUDY.md"

EXPECTED_PROTOCOL_SHA256 = "0536abd9bf6aa2d69fcfeae13dcad5f5c238a4d7898552edff1e5d21fa205bad"
EXPECTED_PRESTUDY_MANIFEST_SHA256 = "523b54342fb1be8427498f35ab5ef5c1be9d32faceaf282c6ab96af5842c7508"
EXPECTED_SEED_FREEZE_SHA256 = "a77953c351d92e2c0ec91bc13fce814b27d63e6b44c1bf2419f1f239ae6a96e5"
EXPECTED_PAIR_MANIFEST_SHA256 = "b1c4ac1db684249d4eccd46691fd1b8ad30c36bb3be56cc19a50a7581c189d5e"
EXPECTED_TRAINING_VERIFIER_SHA256 = "7a7910a112132bae3a7f57bcb1162fa86f98aeb2038bce566fa6ffa1920277bf"
EXPECTED_SENSOR_CALIBRATION_ALGORITHM_SHA256 = (
    "82253ac4cb2a068fab42fc381015af1ac2bb4919babec36af790c0896ca2469e"
)
EXPECTED_V3_CALIBRATION_ALGORITHM_SHA256 = (
    "310f886193ca6bc1584f7e0985794efc32dd08535b1dfc4e1992a9dc88ba26c2"
)

TRAINING_SEEDS = (2026, 2027, 2028)
OLD_SIMULATION_SEEDS = (49026, 49027, 49028, 49029, 49030)
NEW_SIMULATION_SEEDS = (59026, 59027, 59028, 59029, 59030)
FORBIDDEN_PRIOR_SEED_BLOCKS = (
    tuple(range(19026, 19031)),
    tuple(range(29026, 29031)),
    tuple(range(39026, 39031)),
    OLD_SIMULATION_SEEDS,
)

PRIMARY_CONTROLLERS = ("B_plain_MPC", "C3_arbitration_MPC")
CURRENT_CONTROLLERS = ("B_plain_MPC", "C3_arbitration_MPC", "E_EKF_virtual_MPC")
PRIMARY_FAMILIES = ("bias", "dropout", "drift", "load", "combined_bias_load")
CLASSIFIED_FAMILIES = ("bias", "dropout", "drift", "combined_bias_load")
PART_A_SCENARIOS = (
    "combined_fault_load",
    "sensor_bias_5",
    "sensor_dropout",
    "load_disturbance",
    "parameter_variation",
)

VALIDATION_IDS = (3, 9, 11, 16, 19, 23)
PART_A_BOOTSTRAP_SEED = 20260917
PART_B_BOOTSTRAP_SEED = 20260918
BOOTSTRAP_REPLICATES = 20_000

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

PART_B_OUTPUT_FILES = {
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


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing required JSON artifact: {rel(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{rel(path)} must contain a JSON object")
    return payload


def assert_close(actual: Any, expected: Any, label: str, *, atol: float = 1e-11) -> None:
    a = float(actual)
    e = float(expected)
    if math.isnan(e):
        require(math.isnan(a), f"{label}: expected NaN, got {a}")
        return
    require(np.isclose(a, e, rtol=1e-11, atol=atol), f"{label}: {a} != {e}")


def numeric(frame: pd.DataFrame, column: str, label: str, *, allow_nan: bool = False) -> pd.Series:
    require(column in frame.columns, f"{label} missing {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    array = values.to_numpy(float)
    if allow_nan:
        require(not np.isinf(array).any(), f"{label}.{column} contains infinity")
    else:
        require(not values.isna().any(), f"{label}.{column} contains missing/non-numeric values")
        require(np.isfinite(array).all(), f"{label}.{column} contains nonfinite values")
    return values.astype(float)


def integer_counts(frame: pd.DataFrame, column: str, label: str, *, allow_nan: bool = False) -> pd.Series:
    values = numeric(frame, column, label, allow_nan=allow_nan)
    finite = values.notna()
    require((values[finite] >= 0).all(), f"{label}.{column} contains a negative count")
    rounded = np.round(values[finite].to_numpy(float))
    require(
        np.array_equal(rounded, values[finite].to_numpy(float)),
        f"{label}.{column} contains a non-integer count",
    )
    return values


def bool_series(values: pd.Series, label: str, *, allow_nan: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype("boolean") if allow_nan else values.astype(bool)
    # Pandas-3-safe normalization: astype(str) no longer renders NA as "nan",
    # so normalize through the nullable string dtype and fill NA explicitly.
    normalized = values.astype("string").str.strip().str.lower().fillna("nan")
    mapping: dict[str, Any] = {
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
    missing_tokens = {"nan", "none", "", "<na>"}
    invalid = parsed.isna() & ~normalized.isin(missing_tokens)
    require(not invalid.any(), f"{label} contains invalid boolean values: {sorted(set(values[invalid]))[:5]}")
    if not allow_nan:
        require(not parsed.isna().any(), f"{label} contains missing boolean values")
        return parsed.astype(bool)
    return parsed.astype("boolean")


def compare_frame(expected: pd.DataFrame, path: Path, label: str) -> None:
    require(path.is_file(), f"missing required analysis table: {rel(path)}")
    actual = pd.read_csv(path, float_precision="round_trip")
    require(list(actual.columns) == list(expected.columns), f"{label} column schema mismatch")
    require(len(actual) == len(expected), f"{label} row count {len(actual)} != {len(expected)}")
    for column in expected.columns:
        left = actual[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(right.dtype) and not pd.api.types.is_bool_dtype(right.dtype):
            left_num = pd.to_numeric(left, errors="coerce").to_numpy(float)
            right_num = pd.to_numeric(right, errors="coerce").to_numpy(float)
            require(
                np.allclose(left_num, right_num, rtol=1e-11, atol=1e-11, equal_nan=True),
                f"{label}.{column} differs from independent recalculation",
            )
            continue
        left_norm = [None if pd.isna(value) else str(value) for value in left]
        right_norm = [None if pd.isna(value) else str(value) for value in right]
        require(left_norm == right_norm, f"{label}.{column} differs from independent recalculation")


def verify_pinned_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    require(PROTOCOL_PATH.is_file(), f"missing frozen protocol: {rel(PROTOCOL_PATH)}")
    require(sha256(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA256, "study_protocol.json SHA-256 changed")
    protocol = read_json(PROTOCOL_PATH)
    require(protocol.get("study") == "final_robustness_hardening", "protocol study id changed")
    require(
        protocol.get("status") == "frozen_before_any_part_b_scientific_run",
        "protocol was not frozen before Part B execution",
    )
    require(tuple(protocol.get("training_seeds", [])) == TRAINING_SEEDS, "protocol training seeds changed")

    freeze_binding = protocol.get("simulation_seed_freeze")
    require(isinstance(freeze_binding, Mapping), "protocol missing simulation_seed_freeze binding")
    require(str(freeze_binding.get("sha256", "")).lower() == EXPECTED_SEED_FREEZE_SHA256, "seed-freeze binding changed")
    require(tuple(freeze_binding.get("seeds", [])) == NEW_SIMULATION_SEEDS, "protocol simulation seeds changed")
    require(SEED_FREEZE_PATH.is_file(), f"missing seed freeze: {rel(SEED_FREEZE_PATH)}")
    require(sha256(SEED_FREEZE_PATH) == EXPECTED_SEED_FREEZE_SHA256, "simulation_seed_freeze.json changed")
    freeze = read_json(SEED_FREEZE_PATH)
    require(
        freeze.get("status") == "frozen_before_any_part_b_scientific_run",
        "simulation seed freeze status changed",
    )
    require(
        tuple(freeze.get("selected_simulation_seeds", [])) == NEW_SIMULATION_SEEDS,
        "selected simulation seed block changed",
    )
    require(freeze.get("preferred_candidate_accepted") is True, "preferred clean 59026-59030 block was not accepted")
    prior_union = {seed for block in FORBIDDEN_PRIOR_SEED_BLOCKS for seed in block}
    require(not (set(NEW_SIMULATION_SEEDS) & prior_union), "new simulation seed block overlaps historical/reserved blocks")
    scan = freeze.get("provenance_scan")
    require(isinstance(scan, Mapping), "seed freeze lacks provenance scan")
    for key in ("strict_text_matches", "filename_matches", "csv_numeric_matches", "npy_npz_numeric_matches"):
        require(scan.get(key) == [], f"pre-study simulation seed provenance scan is non-empty: {key}")

    manifest_binding = protocol.get("prestudy_frozen_hash_manifest")
    require(isinstance(manifest_binding, Mapping), "protocol missing prestudy manifest binding")
    require(
        str(manifest_binding.get("sha256", "")).lower() == EXPECTED_PRESTUDY_MANIFEST_SHA256,
        "prestudy manifest binding changed",
    )
    pair_binding = protocol.get("model_pair_manifest")
    require(isinstance(pair_binding, Mapping), "protocol missing model-pair binding")
    require(str(pair_binding.get("sha256", "")).lower() == EXPECTED_PAIR_MANIFEST_SHA256, "pair-manifest binding changed")

    runtime = protocol.get("frozen_runtime")
    require(isinstance(runtime, Mapping), "protocol missing frozen_runtime")
    expected_runtime = {
        "dt_s": 0.01,
        "duration_s": 6.0,
        "control_stride": 5,
        "speed_full_scale_rad_s": 65.234375,
        "speed_noise_std_rad_s": 0.25,
        "reference_rad_s": 35.0,
        "load_baseline_Nm": 0.03,
        "bias_window_s": [2.0, 4.0],
        "dropout_onset_s": 2.0,
        "drift_onset_s": 2.0,
        "drift_ramp_duration_s": 4.0,
        "load_step_time_s": 3.0,
        "combined_event_time_s": 3.0,
    }
    for key, expected in expected_runtime.items():
        require(runtime.get(key) == expected, f"protocol frozen_runtime.{key} changed")

    stats = protocol.get("statistics")
    require(isinstance(stats, Mapping), "protocol missing statistics")
    require(tuple(stats.get("hierarchical_bootstrap_order", [])) == ("training_seed", "simulation_seed"), "bootstrap hierarchy changed")
    require(int(stats.get("part_a_bootstrap_rng_seed", -1)) == PART_A_BOOTSTRAP_SEED, "Part A bootstrap seed changed")
    require(int(stats.get("part_b_bootstrap_rng_seed", -1)) == PART_B_BOOTSTRAP_SEED, "Part B bootstrap seed changed")
    require(int(stats.get("bootstrap_replicates", -1)) == BOOTSTRAP_REPLICATES, "bootstrap replicate count changed")
    require(stats.get("detection_interval") == "Wilson 95%", "detection interval changed")
    require(stats.get("time_samples_independent") is False, "protocol now treats time samples as independent")

    matrix = protocol.get("expected_matrix")
    require(isinstance(matrix, Mapping), "protocol missing expected_matrix")
    require(int(matrix.get("primary_runs", -1)) == 600, "primary run count changed")
    require(int(matrix.get("current_runs", -1)) == 135, "current-boundary run count changed")
    require(int(matrix.get("maximum_total_runs", -1)) == 735, "total Part B run count changed")
    require(matrix.get("duplicate_cell_reruns_forbidden") is True, "duplicate-cell prohibition changed")
    return protocol, freeze


def verify_prestudy_frozen_hashes() -> dict[str, Any]:
    require(PRESTUDY_MANIFEST_PATH.is_file(), f"missing prestudy manifest: {rel(PRESTUDY_MANIFEST_PATH)}")
    require(
        sha256(PRESTUDY_MANIFEST_PATH) == EXPECTED_PRESTUDY_MANIFEST_SHA256,
        "prestudy_frozen_hash_manifest.json SHA-256 changed",
    )
    manifest = read_json(PRESTUDY_MANIFEST_PATH)
    require(manifest.get("study") == "final_robustness_hardening", "prestudy manifest belongs to another study")
    require(manifest.get("status") == "frozen_before_part_a_or_part_b_outputs", "prestudy manifest status changed")
    frozen = manifest.get("frozen_artifacts")
    require(isinstance(frozen, Mapping), "prestudy manifest missing frozen_artifacts")
    require(int(manifest.get("artifact_count", -1)) == 43 and len(frozen) == 43, "prestudy lock is not exactly 43 artifacts")
    for relative_path, entry in frozen.items():
        require(isinstance(entry, Mapping), f"malformed prestudy entry for {relative_path}")
        path = PROJECT / str(relative_path)
        require(path.is_file(), f"frozen historical artifact missing: {relative_path}")
        require(sha256(path) == str(entry.get("sha256", "")).lower(), f"frozen artifact hash changed: {relative_path}")
        require(path.stat().st_size == int(entry.get("bytes", -1)), f"frozen artifact byte count changed: {relative_path}")
    require(
        str(frozen.get("scripts/verify_training_seed_robustness.py", {}).get("sha256", "")).lower()
        == EXPECTED_TRAINING_VERIFIER_SHA256,
        "historical training-seed verifier binding changed",
    )
    return manifest


def load_pair_bindings() -> tuple[dict[int, dict[str, str]], dict[str, Any]]:
    require(PAIR_MANIFEST_PATH.is_file(), f"missing pair manifest: {rel(PAIR_MANIFEST_PATH)}")
    require(sha256(PAIR_MANIFEST_PATH) == EXPECTED_PAIR_MANIFEST_SHA256, "model_pair_manifest.json changed")
    manifest = read_json(PAIR_MANIFEST_PATH)
    require(manifest.get("study") == "training_seed_robustness", "pair manifest belongs to another study")
    require(tuple(manifest.get("validation_run_ids", [])) == VALIDATION_IDS, "pair manifest validation population changed")
    algorithms = manifest.get("algorithm_hashes")
    require(isinstance(algorithms, Mapping), "pair manifest missing algorithm_hashes")
    sensor_alg = algorithms.get("sensor_calibration_source")
    v3_alg = algorithms.get("v3_calibration_source")
    require(isinstance(sensor_alg, Mapping) and isinstance(v3_alg, Mapping), "pair manifest calibration algorithms malformed")
    require(str(sensor_alg.get("sha256", "")).lower() == EXPECTED_SENSOR_CALIBRATION_ALGORITHM_SHA256, "sensor calibration algorithm hash changed")
    require(str(v3_alg.get("sha256", "")).lower() == EXPECTED_V3_CALIBRATION_ALGORITHM_SHA256, "V3 calibration algorithm hash changed")
    for block, expected_hash in (
        (sensor_alg, EXPECTED_SENSOR_CALIBRATION_ALGORITHM_SHA256),
        (v3_alg, EXPECTED_V3_CALIBRATION_ALGORITHM_SHA256),
    ):
        path = PROJECT / str(block.get("path", ""))
        require(path.is_file() and sha256(path) == expected_hash, f"frozen calibration algorithm source changed: {block.get('path')}")

    pairs = manifest.get("pairs")
    require(isinstance(pairs, Mapping), "pair manifest missing pairs")
    require(set(pairs) == {f"P{seed}" for seed in TRAINING_SEEDS}, "pair manifest pair set changed")
    result: dict[int, dict[str, str]] = {}
    for seed in TRAINING_SEEDS:
        pair = pairs[f"P{seed}"]
        require(int(pair.get("training_seed", -1)) == seed, f"P{seed} training_seed binding changed")
        row: dict[str, str] = {}
        for run_column, (section, key) in HASH_BINDINGS.items():
            section_value = pair.get(section)
            require(isinstance(section_value, Mapping), f"P{seed}.{section} missing")
            digest = str(section_value.get(key, "")).lower()
            require(len(digest) == 64, f"P{seed}.{section}.{key} is not a SHA-256")
            path_key = "weights_path" if key == "weights_sha256" else "config_path" if key == "config_sha256" else "path"
            artifact = PROJECT / str(section_value.get(path_key, ""))
            require(artifact.is_file() and sha256(artifact) == digest, f"P{seed}.{section} artifact hash mismatch")
            row[run_column] = digest
        result[seed] = row
    return result, manifest


def load_frozen_training_verifier() -> ModuleType:
    require(TRAINING_VERIFIER_PATH.is_file(), f"missing historical verifier: {rel(TRAINING_VERIFIER_PATH)}")
    require(sha256(TRAINING_VERIFIER_PATH) == EXPECTED_TRAINING_VERIFIER_SHA256, "historical training verifier changed")
    spec = importlib.util.spec_from_file_location("frozen_training_seed_verifier", TRAINING_VERIFIER_PATH)
    require(spec is not None and spec.loader is not None, "could not load frozen training-seed verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_independent_seed_calibrations() -> None:
    """Reuse only the already-frozen independent calibration reconstruction.

    The helper's source is itself re-hashed by the 43-artifact historical lock
    before import, so this remains independent of the new Part B evaluator and
    analyzer while avoiding a second copy of several hundred calibration lines.
    """

    verifier = load_frozen_training_verifier()
    pairs, _ = verifier.bind_pair_artifacts()
    verifier.verify_independent_calibrations(pairs)


def verify_part_a_source_and_pairs() -> pd.DataFrame:
    require(PART_A_RUNS_PATH.is_file(), f"missing frozen Part A source: {rel(PART_A_RUNS_PATH)}")
    prestudy = read_json(PRESTUDY_MANIFEST_PATH)["frozen_artifacts"]
    source_entry = prestudy.get(rel(PART_A_RUNS_PATH))
    require(isinstance(source_entry, Mapping), "prestudy lock does not bind the Part A 150-run source")
    require(sha256(PART_A_RUNS_PATH) == str(source_entry.get("sha256", "")).lower(), "Part A source matrix changed")
    runs = pd.read_csv(PART_A_RUNS_PATH, float_precision="round_trip")
    required = {"training_seed", "simulation_seed", "controller", "scenario", "fault_window_rmse", *HASH_BINDINGS}
    require(required.issubset(runs.columns), f"Part A source missing columns: {sorted(required - set(runs.columns))}")
    require(len(runs) == 150, f"Part A source must contain 150 rows, found {len(runs)}")
    runs["training_seed"] = numeric(runs, "training_seed", "Part A source").astype(int)
    runs["simulation_seed"] = numeric(runs, "simulation_seed", "Part A source").astype(int)
    runs["fault_window_rmse"] = numeric(runs, "fault_window_rmse", "Part A source")
    expected_keys = {
        (training_seed, scenario, simulation_seed, controller)
        for training_seed in TRAINING_SEEDS
        for scenario in PART_A_SCENARIOS
        for simulation_seed in OLD_SIMULATION_SEEDS
        for controller in PRIMARY_CONTROLLERS
    }
    actual_keys = set(
        map(
            tuple,
            runs[["training_seed", "scenario", "simulation_seed", "controller"]].itertuples(index=False, name=None),
        )
    )
    require(actual_keys == expected_keys, "Part A source is not the exact 3x5x5x2 matrix")
    require(
        not runs.duplicated(["training_seed", "scenario", "simulation_seed", "controller"]).any(),
        "Part A source contains duplicate keys",
    )

    rows: list[dict[str, Any]] = []
    for keys, group in runs.groupby(["training_seed", "scenario", "simulation_seed"], sort=False):
        require(len(group) == 2 and set(group.controller) == set(PRIMARY_CONTROLLERS), f"Part A incomplete B/C3 pair: {keys}")
        for column in HASH_BINDINGS:
            require(group[column].nunique(dropna=False) == 1, f"Part A B/C3 pair mixes {column}: {keys}")
        b = group[group.controller.eq("B_plain_MPC")].iloc[0]
        c3 = group[group.controller.eq("C3_arbitration_MPC")].iloc[0]
        rows.append(
            {
                "training_seed": int(keys[0]),
                "scenario": str(keys[1]),
                "simulation_seed": int(keys[2]),
                "delta": float(c3.fault_window_rmse - b.fault_window_rmse),
            }
        )
    paired = pd.DataFrame(rows)
    require(len(paired) == 75, f"Part A must contain 75 paired Delta cells, found {len(paired)}")
    return paired


def part_a_matrix(paired: pd.DataFrame, scenario: str) -> np.ndarray:
    frame = paired[paired.scenario.eq(scenario)].pivot(
        index="training_seed", columns="simulation_seed", values="delta"
    )
    frame = frame.reindex(index=TRAINING_SEEDS, columns=OLD_SIMULATION_SEEDS)
    require(frame.shape == (3, 5) and not frame.isna().any().any(), f"Part A {scenario} is not a complete 3x5 matrix")
    return frame.to_numpy(float)


def recompute_part_a_decomposition(paired: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for scenario in PART_A_SCENARIOS:
        values = part_a_matrix(paired, scenario)
        grand = float(values.mean())
        train_means = values.mean(axis=1)
        sim_means = values.mean(axis=0)
        train_effect = train_means - grand
        sim_effect = sim_means - grand
        residual = values - grand - train_effect[:, None] - sim_effect[None, :]
        centered = values - grand
        ss_total = float(np.sum(centered**2))
        ss_training = float(5 * np.sum(train_effect**2))
        ss_simulation = float(3 * np.sum(sim_effect**2))
        ss_residual = float(np.sum(residual**2))
        require(
            np.isclose(ss_total, ss_training + ss_simulation + ss_residual, rtol=1e-11, atol=1e-11),
            f"Part A {scenario} sum-of-squares decomposition does not close",
        )
        if ss_total > 0:
            proportions = (ss_training / ss_total, ss_simulation / ss_total, ss_residual / ss_total)
        else:
            proportions = (0.0, 0.0, 0.0)
        favorable = values < 0.0
        unfavorable = values > 0.0
        zero = values == 0.0
        row: dict[str, Any] = {
            "scenario": scenario,
            "paired_delta_count": 15,
            "grand_mean_delta": grand,
            "delta_min": float(values.min()),
            "delta_max": float(values.max()),
            "delta_range": float(np.ptp(values)),
            "ss_total_centered": ss_total,
            "ss_training_seed": ss_training,
            "ss_simulation_seed": ss_simulation,
            "ss_residual_interaction_unexplained": ss_residual,
            "prop_training_seed": float(proportions[0]),
            "prop_simulation_seed": float(proportions[1]),
            "prop_residual_interaction_unexplained": float(proportions[2]),
            "training_seed_effect_range": float(np.ptp(train_effect)),
            "simulation_seed_effect_range": float(np.ptp(sim_effect)),
            "residual_range": float(np.ptp(residual)),
            "residual_max_abs": float(np.max(np.abs(residual))),
            "favorable_delta_count": int(favorable.sum()),
            "unfavorable_delta_count": int(unfavorable.sum()),
            "zero_delta_count": int(zero.sum()),
            "favorable_delta_fraction": float(favorable.mean()),
            "favorable_training_seed_mean_count": int(np.sum(train_means < 0.0)),
            "favorable_simulation_seed_mean_count": int(np.sum(sim_means < 0.0)),
        }
        for index, seed in enumerate(TRAINING_SEEDS):
            row[f"training_seed_{seed}_mean_delta"] = float(train_means[index])
            row[f"training_seed_{seed}_effect"] = float(train_effect[index])
            row[f"training_seed_{seed}_favorable_simulation_count"] = int(favorable[index].sum())
        for index, seed in enumerate(OLD_SIMULATION_SEEDS):
            row[f"simulation_seed_{seed}_mean_delta"] = float(sim_means[index])
            row[f"simulation_seed_{seed}_effect"] = float(sim_effect[index])
            row[f"simulation_seed_{seed}_favorable_training_count"] = int(favorable[:, index].sum())
        rows.append(row)
        details[scenario] = {
            "grand_mean_delta": grand,
            "training_seed_means": {str(seed): float(train_means[i]) for i, seed in enumerate(TRAINING_SEEDS)},
            "simulation_seed_means": {str(seed): float(sim_means[i]) for i, seed in enumerate(OLD_SIMULATION_SEEDS)},
            "training_seed_effects": {str(seed): float(train_effect[i]) for i, seed in enumerate(TRAINING_SEEDS)},
            "simulation_seed_effects": {str(seed): float(sim_effect[i]) for i, seed in enumerate(OLD_SIMULATION_SEEDS)},
            "sum_of_squares": {
                "total_centered": ss_total,
                "training_seed": ss_training,
                "simulation_seed": ss_simulation,
                "residual_interaction_unexplained": ss_residual,
            },
            "proportion_of_total_centered_variation": {
                "training_seed": float(proportions[0]),
                "simulation_seed": float(proportions[1]),
                "residual_interaction_unexplained": float(proportions[2]),
            },
            "effect_ranges": {
                "training_seed": float(np.ptp(train_effect)),
                "simulation_seed": float(np.ptp(sim_effect)),
                "residual": float(np.ptp(residual)),
            },
            "delta_sign_counts": {
                "negative_favorable": int(favorable.sum()),
                "positive_unfavorable": int(unfavorable.sum()),
                "zero_tie": int(zero.sum()),
            },
            "per_training_seed_favorable_simulation_count": {
                str(seed): int(favorable[i].sum()) for i, seed in enumerate(TRAINING_SEEDS)
            },
            "per_simulation_seed_favorable_training_count": {
                str(seed): int(favorable[:, i].sum()) for i, seed in enumerate(OLD_SIMULATION_SEEDS)
            },
            "delta_matrix": {
                str(training_seed): {
                    str(sim_seed): float(values[row_index, column_index])
                    for column_index, sim_seed in enumerate(OLD_SIMULATION_SEEDS)
                }
                for row_index, training_seed in enumerate(TRAINING_SEEDS)
            },
        }
    return pd.DataFrame(rows), details


def recompute_part_a_bootstrap(paired: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(PART_A_BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for scenario in PART_A_SCENARIOS:
        matrix = part_a_matrix(paired, scenario)
        boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for index in range(BOOTSTRAP_REPLICATES):
            sampled_training = rng.integers(0, 3, size=3)
            cluster_means = np.empty(3, dtype=float)
            for cluster_index, training_index in enumerate(sampled_training):
                sampled_simulation = rng.integers(0, 5, size=5)
                cluster_means[cluster_index] = float(matrix[int(training_index), sampled_simulation].mean())
            boot[index] = float(cluster_means.mean())
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "scenario": scenario,
                "point_estimate_mean_delta": float(matrix.mean()),
                "bootstrap_mean_delta": float(boot.mean()),
                "bootstrap_sd_delta": float(boot.std(ddof=1)),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "prob_mean_delta_lt_zero": float(np.mean(boot < 0.0)),
                "analysis_rng_seed": PART_A_BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "training_seed_clusters": 3,
                "simulation_seeds_per_cluster": 5,
            }
        )
    return pd.DataFrame(rows)


def compare_nested_numbers(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        require(isinstance(actual, Mapping), f"{label} is not an object")
        for key, value in expected.items():
            require(key in actual, f"{label} missing key {key}")
            compare_nested_numbers(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool):
        assert_close(actual, expected, label)
        return
    require(actual == expected, f"{label}: {actual!r} != {expected!r}")


def verify_part_a_outputs() -> None:
    paired = verify_part_a_source_and_pairs()
    expected_decomposition, details = recompute_part_a_decomposition(paired)
    expected_bootstrap = recompute_part_a_bootstrap(paired)
    compare_frame(expected_decomposition, PART_A_DECOMPOSITION_PATH, "variance_decomposition.csv")
    compare_frame(expected_bootstrap, PART_A_BOOTSTRAP_PATH, "variance_bootstrap.csv")

    summary = read_json(PART_A_SUMMARY_PATH)
    require(summary.get("study") == "final_robustness_hardening_part_a", "Part A summary study id changed")
    source = summary.get("source")
    require(isinstance(source, Mapping), "Part A summary missing source binding")
    require(source.get("path") == rel(PART_A_RUNS_PATH), "Part A summary source path changed")
    require(source.get("sha256") == sha256(PART_A_RUNS_PATH), "Part A summary source hash mismatch")
    require(source.get("prestudy_manifest_sha256") == EXPECTED_PRESTUDY_MANIFEST_SHA256, "Part A summary prestudy hash mismatch")
    design = summary.get("design")
    require(isinstance(design, Mapping), "Part A summary missing design")
    require(design.get("source_run_count") == 150, "Part A source run count mismatch")
    require(design.get("paired_delta_count_total") == 75, "Part A paired count mismatch")
    require(design.get("paired_delta_count_per_scenario") == 15, "Part A per-scenario paired count mismatch")
    require(design.get("no_missing_combinations") is True, "Part A summary does not attest complete matrix")
    require(design.get("no_duplicate_combinations") is True, "Part A summary does not attest unique matrix")
    require(design.get("exact_b_c3_pairing_verified") is True, "Part A summary does not attest exact B/C3 pairing")

    scenario_results = summary.get("decomposition", {}).get("scenario_results", [])
    require(isinstance(scenario_results, list) and len(scenario_results) == 5, "Part A summary scenario result count mismatch")
    actual_details = {str(item.get("scenario")): item for item in scenario_results if isinstance(item, Mapping)}
    require(set(actual_details) == set(PART_A_SCENARIOS), "Part A summary scenario set mismatch")
    for scenario, expected in details.items():
        compare_nested_numbers(actual_details[scenario], {"scenario": scenario, **expected}, f"Part A {scenario}")

    boot_summary = summary.get("hierarchical_bootstrap")
    require(isinstance(boot_summary, Mapping), "Part A summary missing hierarchical_bootstrap")
    require(boot_summary.get("analysis_rng_seed") == PART_A_BOOTSTRAP_SEED, "Part A bootstrap seed mismatch")
    require(boot_summary.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES, "Part A bootstrap replicate mismatch")
    require(tuple(boot_summary.get("scenario_order", [])) == PART_A_SCENARIOS, "Part A bootstrap scenario order changed")
    for key, path in (
        ("variance_decomposition_csv", PART_A_DECOMPOSITION_PATH),
        ("variance_bootstrap_csv", PART_A_BOOTSTRAP_PATH),
    ):
        entry = summary.get("outputs", {}).get(key)
        require(isinstance(entry, Mapping), f"Part A summary missing output binding {key}")
        require(entry.get("sha256") == sha256(path), f"Part A summary hash mismatch for {key}")


def copy_alias(frame: pd.DataFrame, target: str, aliases: Sequence[str], *, required: bool = False) -> None:
    if target in frame.columns:
        return
    present = [alias for alias in aliases if alias in frame.columns]
    if len(present) > 1:
        first = frame[present[0]]
        for alias in present[1:]:
            require(first.equals(frame[alias]), f"ambiguous aliases for {target}: {present}")
    if present:
        frame[target] = frame[present[0]]
    elif required:
        fail(f"Part B run CSV missing {target}; accepted aliases={list(aliases)}")


def normalize_family(value: Any) -> str:
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
    fail(f"unrecognized Part B family: {value!r}")
    return ""  # pragma: no cover


def canonicalize_part_b_runs(raw: pd.DataFrame) -> pd.DataFrame:
    runs = raw.copy()
    aliases: dict[str, tuple[str, ...]] = {
        "family": ("study_family", "sweep_family", "study_family", "scenario_family", "sweep"),
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
        "slew_violations": ("rate_violations",),
        "ekf_numerical_failures": ("ekf_failures",),
        "first_post_event_reliability_entry_time_s": ("first_reliability_entry_time_s", "first_entry_time_s"),
        "post_event_reliability_active_duration_s": (
            "post_event_reliability_episode_duration_s",
            "reliability_active_duration_post_event_s",
        ),
        "aux_speed_rmse_event_window": ("aux_virtual_rmse_event_window", "aux_estimator_rmse_event_window"),
        "ekf_speed_rmse_event_window": ("ekf_estimator_rmse",),
    }
    for target, names in aliases.items():
        copy_alias(runs, target, names, required=target in {"family", "simulation_seed"})
    required = {"training_seed", "simulation_seed", "controller", "family"}
    require(required.issubset(runs.columns), f"Part B runs missing columns: {sorted(required - set(runs.columns))}")
    runs["family"] = runs["family"].map(normalize_family)
    runs["training_seed"] = numeric(runs, "training_seed", "Part B runs").astype(int)
    runs["simulation_seed"] = numeric(runs, "simulation_seed", "Part B runs").astype(int)
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
            mask = runs.family.eq(family) & runs[target].isna()
            runs.loc[mask, target] = severity.loc[mask]
    return runs


def float_key(value: Any) -> float:
    return round(float(value), 10)


def primary_conditions(protocol: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    sweeps = protocol["primary_sweeps"]
    conditions = {
        "bias": tuple(float_key(value) for value in sweeps["bias"]["levels_percent"]),
        "dropout": tuple(float_key(value) for value in sweeps["dropout"]["durations_s"]),
        "drift": tuple(float_key(value) for value in sweeps["drift"]["final_percent"]),
        "load": tuple(float_key(value) for value in sweeps["load"]["post_step_load_Nm"]),
        "combined_bias_load": tuple(
            (float_key(bias), float_key(load))
            for bias in sweeps["combined_bias_load"]["bias_percent"]
            for load in sweeps["combined_bias_load"]["post_step_load_Nm"]
        ),
    }
    expected = {
        "bias": (2.5, 5.0, 10.0, 15.0),
        "dropout": (0.1, 0.5, 1.0, 2.0),
        "drift": (2.5, 5.0, 10.0, 15.0),
        "load": (0.06, 0.1, 0.15, 0.2),
        "combined_bias_load": ((5.0, 0.1), (5.0, 0.15), (15.0, 0.1), (15.0, 0.15)),
    }
    require(conditions == expected, f"Part B severity grid changed: {conditions}")
    return conditions


def current_cases(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    boundary = protocol["current_sensor_boundary"]
    cases = tuple(str(entry["name"]) for entry in boundary["cases"])
    expected = ("current_bias_5", "current_dropout_2s", "speed_bias_5_plus_current_bias_5")
    require(cases == expected, f"current-boundary cases changed: {cases}")
    require(float(boundary["current_reference_scale_A"]) == 5.122337818145752, "current reference scale changed")
    require(float(boundary["bias_A"]) == 0.2561168909072876, "current bias magnitude changed")
    require(tuple(boundary["window_s"]) == (2.0, 4.0), "current corruption window changed")
    return cases


def row_condition(row: pd.Series) -> Any:
    family = str(row["family"])
    if family == "bias":
        return float_key(row["bias_percent"])
    if family == "dropout":
        return float_key(row["dropout_duration_s"])
    if family == "drift":
        return float_key(row["drift_final_percent"])
    if family == "load":
        return float_key(row["post_step_load_Nm"])
    if family == "combined_bias_load":
        return (float_key(row["bias_percent"]), float_key(row["post_step_load_Nm"]))
    if family == "current_sensor":
        return str(row["current_case"])
    fail(f"unsupported Part B family {family}")
    return None


def condition_label(family: str, condition: Any) -> str:
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


def verify_part_b_matrix(
    runs: pd.DataFrame,
    protocol: Mapping[str, Any],
    pair_bindings: Mapping[int, Mapping[str, str]],
) -> pd.DataFrame:
    required = {
        "study",
        "matrix",
        "run_complete",
        "overall_rmse",
        "fault_window_rmse",
        "sensor_fault_detected",
        "detection_latency_s",
        "reliability_entries",
        "post_event_reliability_entries",
        "sub_fraction",
        "sub_duration_s",
        "recovery_events",
        "recovered_after_event",
        *SAFETY_COLUMNS,
        *HASH_BINDINGS,
    }
    require(required.issubset(runs.columns), f"Part B raw runs missing required columns: {sorted(required - set(runs.columns))}")
    for column in ("overall_rmse", "fault_window_rmse", "reliability_entries", "post_event_reliability_entries", "sub_fraction", "recovery_events"):
        runs[column] = numeric(runs, column, "Part B runs")
    for column in (
        "frac_phys",
        "frac_main",
        "frac_aux",
        "frac_fb",
        "event_frac_phys",
        "event_frac_main",
        "event_frac_aux",
        "event_frac_fb",
        "ekf_event_sub_fraction",
        "recovery_latency_s",
        "avg_episode_duration_s",
        "sub_duration_s",
        "first_post_event_reliability_entry_time_s",
        "post_event_reliability_active_duration_s",
        "aux_speed_rmse_event_window",
        "aux_virtual_rmse",
        "ekf_speed_rmse_event_window",
    ):
        if column not in runs.columns:
            runs[column] = np.nan
        runs[column] = numeric(runs, column, "Part B runs", allow_nan=True)
    runs["detection_latency_s"] = numeric(runs, "detection_latency_s", "Part B runs", allow_nan=True)
    runs["run_complete"] = bool_series(runs["run_complete"], "Part B run_complete")
    runs["sensor_fault_detected"] = bool_series(runs["sensor_fault_detected"], "Part B sensor_fault_detected", allow_nan=True)
    runs["recovered_after_event"] = bool_series(
        runs["recovered_after_event"], "Part B recovered_after_event", allow_nan=True
    )

    ekf_mask = runs.controller.astype(str).eq("E_EKF_virtual_MPC")
    ekf_values = pd.to_numeric(runs["ekf_numerical_failures"], errors="coerce")
    require(not ekf_values.loc[ekf_mask].isna().any(), "EKF rows omit ekf_numerical_failures")
    runs["ekf_numerical_failures"] = ekf_values.fillna(0.0)
    for column in SAFETY_COLUMNS:
        integer_counts(runs, column, "Part B runs")
    for column in (
        "incomplete_run_failures",
        "total_failure_count",
        "total_safety_violation_count",
    ):
        integer_counts(runs, column, "Part B runs")
    require(
        np.array_equal(
            pd.to_numeric(runs["incomplete_run_failures"]).to_numpy(int),
            (~runs["run_complete"]).astype(int).to_numpy(),
        ),
        "incomplete_run_failures is inconsistent with run_complete",
    )
    expected_failure_count = (
        pd.to_numeric(runs["optimizer_failures"])
        + pd.to_numeric(runs["main_prediction_failures"])
        + pd.to_numeric(runs["aux_prediction_failures"])
        + pd.to_numeric(runs["ekf_numerical_failures"])
    ).to_numpy(int)
    require(
        np.array_equal(pd.to_numeric(runs["total_failure_count"]).to_numpy(int), expected_failure_count),
        "total_failure_count arithmetic is inconsistent",
    )
    expected_safety_count = (
        pd.to_numeric(runs["voltage_violations"])
        + pd.to_numeric(runs["slew_violations"])
        + pd.to_numeric(runs["nonfinite_events"])
        + pd.to_numeric(runs["incomplete_run_failures"])
    ).to_numpy(int)
    require(
        np.array_equal(
            pd.to_numeric(runs["total_safety_violation_count"]).to_numpy(int), expected_safety_count
        ),
        "total_safety_violation_count arithmetic is inconsistent",
    )
    if "rate_violations" in runs.columns:
        require(
            np.array_equal(
                pd.to_numeric(runs["rate_violations"]).to_numpy(int),
                pd.to_numeric(runs["slew_violations"]).to_numpy(int),
            ),
            "rate_violations alias differs from slew_violations",
        )

    require(set(runs.training_seed) == set(TRAINING_SEEDS), "Part B training seed set changed")
    require(set(runs.simulation_seed) == set(NEW_SIMULATION_SEEDS), "Part B simulation seed set changed")
    require(set(runs.family).issubset(set(PRIMARY_FAMILIES) | {"current_sensor"}), "Part B contains an unregistered family")
    require(set(runs.study.astype(str)) == {"final_robustness_hardening"}, "Part B study id changed")
    for column in ("bias_percent", "dropout_duration_s", "drift_final_percent", "post_step_load_Nm"):
        if column not in runs.columns:
            runs[column] = np.nan
    if "current_case" not in runs.columns:
        runs["current_case"] = ""

    runs["condition_key"] = [row_condition(row) for _, row in runs.iterrows()]
    runs["condition_label"] = [
        condition_label(str(row.family), row.condition_key) for row in runs.itertuples(index=False)
    ]
    primary = runs[runs.family.isin(PRIMARY_FAMILIES)].copy()
    current = runs[runs.family.eq("current_sensor")].copy()
    require(set(primary.matrix.astype(str)) == {"primary"}, "primary rows have wrong matrix label")
    require(set(current.matrix.astype(str)) == {"current_boundary"}, "current rows have wrong matrix label")
    require(len(primary) == 600 and len(current) == 135 and len(runs) == 735, "Part B matrix must be exactly 600 primary + 135 current = 735 rows")

    conditions = primary_conditions(protocol)
    expected_primary = {
        (family, condition, training_seed, simulation_seed, controller)
        for family, values in conditions.items()
        for condition in values
        for training_seed in TRAINING_SEEDS
        for simulation_seed in NEW_SIMULATION_SEEDS
        for controller in PRIMARY_CONTROLLERS
    }
    actual_primary = {
        (str(row.family), row.condition_key, int(row.training_seed), int(row.simulation_seed), str(row.controller))
        for row in primary.itertuples(index=False)
    }
    require(actual_primary == expected_primary, f"primary Part B Cartesian matrix mismatch; missing={sorted(expected_primary - actual_primary, key=str)[:5]}, extra={sorted(actual_primary - expected_primary, key=str)[:5]}")
    require(not primary.duplicated(["family", "condition_label", "training_seed", "simulation_seed", "controller"]).any(), "duplicate primary Part B cells")

    cases = current_cases(protocol)
    expected_current = {
        (case, training_seed, simulation_seed, controller)
        for case in cases
        for training_seed in TRAINING_SEEDS
        for simulation_seed in NEW_SIMULATION_SEEDS
        for controller in CURRENT_CONTROLLERS
    }
    actual_current = {
        (str(row.current_case), int(row.training_seed), int(row.simulation_seed), str(row.controller))
        for row in current.itertuples(index=False)
    }
    require(actual_current == expected_current, f"current-boundary Cartesian matrix mismatch; missing={sorted(expected_current - actual_current, key=str)[:5]}, extra={sorted(actual_current - expected_current, key=str)[:5]}")
    require(not current.duplicated(["current_case", "training_seed", "simulation_seed", "controller"]).any(), "duplicate current-boundary cells")

    for seed in TRAINING_SEEDS:
        block = runs[runs.training_seed.eq(seed)]
        for column, expected_digest in pair_bindings[seed].items():
            values = {str(value).lower() for value in block[column]}
            require(values == {expected_digest}, f"training seed {seed} cross-contaminated {column}: {values}")
    require(runs.run_complete.all(), "Part B matrix contains incomplete runs")

    c3 = runs[runs.controller.eq("C3_arbitration_MPC")]
    for column in ("frac_phys", "frac_main", "frac_aux", "frac_fb"):
        require(not c3[column].isna().any(), f"C3 rows omit {column}")
        require(np.isfinite(c3[column].to_numpy(float)).all(), f"C3 rows contain nonfinite {column}")
    fractions = c3[["frac_phys", "frac_main", "frac_aux", "frac_fb"]].sum(axis=1).to_numpy(float)
    require(np.allclose(fractions, 1.0, rtol=1e-9, atol=1e-9), "C3 source fractions do not sum to one")
    return runs


def verify_run_protocol_bindings(runs: pd.DataFrame) -> None:
    protocol_columns = ("study_protocol_sha256", "protocol_sha256")
    freeze_columns = ("simulation_seed_freeze_sha256", "seed_freeze_sha256")
    protocol_present = [column for column in protocol_columns if column in runs.columns]
    freeze_present = [column for column in freeze_columns if column in runs.columns]
    require(protocol_present, "Part B raw rows do not expose a study-protocol hash binding")
    require(freeze_present, "Part B raw rows do not expose a simulation-seed-freeze hash binding")
    for column in protocol_present:
        require({str(value).lower() for value in runs[column]} == {EXPECTED_PROTOCOL_SHA256}, f"Part B {column} binding mismatch")
    for column in freeze_present:
        require({str(value).lower() for value in runs[column]} == {EXPECTED_SEED_FREEZE_SHA256}, f"Part B {column} binding mismatch")

    fixed_hashes = {
        "prestudy_manifest_sha256": EXPECTED_PRESTUDY_MANIFEST_SHA256,
        "model_pair_manifest_sha256": EXPECTED_PAIR_MANIFEST_SHA256,
        "sensor_calibration_algorithm_sha256": EXPECTED_SENSOR_CALIBRATION_ALGORITHM_SHA256,
        "v3_calibration_algorithm_sha256": EXPECTED_V3_CALIBRATION_ALGORITHM_SHA256,
    }
    for column, digest in fixed_hashes.items():
        require(column in runs.columns, f"Part B raw rows do not expose required binding {column}")
        require({str(value).lower() for value in runs[column]} == {digest}, f"Part B {column} binding mismatch")

    prestudy = read_json(PRESTUDY_MANIFEST_PATH)["frozen_artifacts"]
    historical_bindings = {
        "ekf_config_sha256": "results/configs/ekf_frozen_config.json",
        "frozen_v3_evaluator_sha256": "scripts/evaluate_v3_closed_loop.py",
        "training_seed_evaluator_sha256": "scripts/evaluate_training_seed_robustness.py",
        "ekf_evaluator_sha256": "scripts/evaluate_ekf_closed_loop.py",
    }
    for column, path_key in historical_bindings.items():
        require(column in runs.columns, f"Part B raw rows do not expose required binding {column}")
        entry = prestudy.get(path_key)
        require(isinstance(entry, Mapping), f"prestudy lock does not bind {path_key}")
        digest = str(entry.get("sha256", "")).lower()
        require({str(value).lower() for value in runs[column]} == {digest}, f"Part B {column} binding mismatch")


def verify_current_corruption_evidence(runs: pd.DataFrame, protocol: Mapping[str, Any]) -> None:
    current = runs[runs.family.eq("current_sensor")].copy()
    boundary = protocol["current_sensor_boundary"]
    bias_a = float(boundary["bias_A"])
    start, end = map(float, boundary["window_s"])

    if "event_start_s" in current.columns:
        values = pd.to_numeric(current.event_start_s, errors="coerce")
        require(np.allclose(values, start), "current-boundary event_start_s differs from frozen [2,4) window")
    if "event_end_s" in current.columns:
        values = pd.to_numeric(current.event_end_s, errors="coerce")
        require(np.allclose(values, end), "current-boundary event_end_s differs from frozen [2,4) window")
    if "current_bias_A" in current.columns:
        values = pd.to_numeric(current.current_bias_A, errors="coerce")
        biased = current.current_case.isin(["current_bias_5", "speed_bias_5_plus_current_bias_5"])
        require(np.allclose(values[biased], bias_a), "current-boundary +5% current bias magnitude changed")

    require("current_reference_scale_A" in current.columns, "current rows omit current_reference_scale_A")
    scale_values = pd.to_numeric(current.current_reference_scale_A, errors="coerce")
    require(
        np.allclose(scale_values, float(boundary["current_reference_scale_A"])),
        "current reference scale differs from the frozen protocol",
    )
    require("current_corruption_mode" in current.columns, "current rows omit current_corruption_mode")
    expected_modes = {
        "current_bias_5": "additive_bias",
        "current_dropout_2s": "dropout_zero",
        "speed_bias_5_plus_current_bias_5": "additive_bias",
    }
    for case, mode in expected_modes.items():
        modes = set(current.loc[current.current_case.eq(case), "current_corruption_mode"].astype(str))
        require(modes == {mode}, f"{case} current corruption mode changed: {modes}")

    # The scientific runner records a per-run count for its shared-current
    # contract.  On E rows, the same measured-current scalar is supplied to the
    # EKF immediately and then appended to the auxiliary-LSTM current history.
    # A nonzero count invalidates the boundary study.
    if "current_equivalence_mismatch_count" in current.columns:
        counts = integer_counts(current, "current_equivalence_mismatch_count", "current-boundary runs")
        e_rows = current.controller.eq("E_EKF_virtual_MPC")
        require(
            (counts.loc[e_rows] == 0).all(),
            "E_EKF current-boundary rows report shared-current-sample mismatches",
        )
        return

    # Preferred proof is a sample-sequence digest for the common corrupted
    # current and each consumer.  A zero max-absolute-difference audit is also
    # accepted when the runner emits that instead.  The comparison is always
    # within one controller run; different controllers can have different plant
    # current trajectories and therefore must not be forced to share a digest.
    common_candidates = ("corrupted_current_sha256", "current_measurement_sha256", "current_input_sha256")
    aux_candidates = ("aux_current_input_sha256", "aux_current_sha256")
    ekf_candidates = ("ekf_current_input_sha256", "ekf_current_sha256")
    common = next((column for column in common_candidates if column in current.columns), None)
    aux = next((column for column in aux_candidates if column in current.columns), None)
    ekf = next((column for column in ekf_candidates if column in current.columns), None)
    if common is not None and aux is not None:
        c3_or_ekf = current.controller.isin(["C3_arbitration_MPC", "E_EKF_virtual_MPC"])
        require(
            (current.loc[c3_or_ekf, common].astype(str) == current.loc[c3_or_ekf, aux].astype(str)).all(),
            "auxiliary LSTM did not receive the exact corrupted current sample sequence",
        )
        if ekf is not None:
            e_rows = current.controller.eq("E_EKF_virtual_MPC")
            require(
                (current.loc[e_rows, common].astype(str) == current.loc[e_rows, ekf].astype(str)).all(),
                "EKF did not receive the exact corrupted current sample sequence",
            )
            return

    aux_diff = next(
        (column for column in ("aux_current_input_max_abs_diff", "aux_corrupted_current_max_abs_diff") if column in current.columns),
        None,
    )
    ekf_diff = next(
        (column for column in ("ekf_current_input_max_abs_diff", "ekf_corrupted_current_max_abs_diff") if column in current.columns),
        None,
    )
    if aux_diff is not None:
        applicable = current.controller.isin(["C3_arbitration_MPC", "E_EKF_virtual_MPC"])
        require(np.allclose(pd.to_numeric(current.loc[applicable, aux_diff]), 0.0), "aux corrupted-current audit is nonzero")
        if ekf_diff is not None:
            e_rows = current.controller.eq("E_EKF_virtual_MPC")
            require(np.allclose(pd.to_numeric(current.loc[e_rows, ekf_diff]), 0.0), "EKF corrupted-current audit is nonzero")
            return
    fail(
        "Part B current-boundary rows do not contain sufficient sample-level current-consumer equivalence evidence "
        "(expected current_equivalence_mismatch_count, matching input SHA-256 columns, or zero max-abs-difference audit columns)"
    )


def paired_primary_points(runs: pd.DataFrame) -> pd.DataFrame:
    primary = runs[runs.family.isin(PRIMARY_FAMILIES)]
    rows: list[dict[str, Any]] = []
    for keys, group in primary.groupby(["family", "condition_label", "training_seed", "simulation_seed"], sort=True):
        require(len(group) == 2 and set(group.controller) == set(PRIMARY_CONTROLLERS), f"incomplete Part B B/C3 pair: {keys}")
        for column in HASH_BINDINGS:
            require(group[column].nunique(dropna=False) == 1, f"Part B B/C3 pair mixes {column}: {keys}")
        b = group[group.controller.eq("B_plain_MPC")].iloc[0]
        c3 = group[group.controller.eq("C3_arbitration_MPC")].iloc[0]
        family, label, training_seed, simulation_seed = keys
        rows.append(
            {
                "family": family,
                "condition_label": label,
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
                "c3_event_frac_phys": float(c3.event_frac_phys),
                "c3_event_frac_main": float(c3.event_frac_main),
                "c3_event_frac_aux": float(c3.event_frac_aux),
                "c3_event_frac_fb": float(c3.event_frac_fb),
                "c3_recovery_events": int(c3.recovery_events),
                "c3_recovery_latency_s": float(c3.recovery_latency_s) if pd.notna(c3.recovery_latency_s) else np.nan,
                "c3_recovered_after_event": c3.recovered_after_event,
            }
        )
    result = pd.DataFrame(rows).sort_values(["family", "condition_label", "training_seed", "simulation_seed"]).reset_index(drop=True)
    require(len(result) == 300, f"expected 300 primary paired points, found {len(result)}")
    return result


def training_seed_means(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in points.groupby(["family", "condition_label", "training_seed"], sort=True):
        require(len(group) == 5, f"severity per-training summary is not five simulations: {keys}")
        delta = group.c3_minus_b_fault_window_rmse.to_numpy(float)
        detected = group.c3_sensor_fault_detected.dropna().astype(bool)
        rows.append(
            {
                "family": keys[0],
                "condition_label": keys[1],
                "training_seed": int(keys[2]),
                "paired_count": 5,
                "mean_b_fault_window_rmse": float(group.b_fault_window_rmse.mean()),
                "mean_c3_fault_window_rmse": float(group.c3_fault_window_rmse.mean()),
                "mean_c3_minus_b_fault_window_rmse": float(delta.mean()),
                "sample_sd_c3_minus_b_fault_window_rmse": float(delta.std(ddof=1)),
                "favorable_pair_count": int(np.sum(delta < 0.0)),
                "mean_effect_favorable": bool(delta.mean() < 0.0),
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
    require(len(result) == 60, f"expected 60 per-training severity summaries, found {len(result)}")
    return result


def wilson_interval(successes: int, total: int) -> tuple[float, float, float]:
    require(total > 0, "Wilson interval requires at least one trial")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return p, max(0.0, center - half), min(1.0, center + half)


def detection_wilson_table(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    detected = points[points.family.isin(CLASSIFIED_FAMILIES)]
    for keys, group in detected.groupby(["family", "condition_label"], sort=True):
        for seed in (*TRAINING_SEEDS, "pooled"):
            block = group if seed == "pooled" else group[group.training_seed.eq(seed)]
            values = block.c3_sensor_fault_detected.dropna().astype(bool)
            expected_n = 15 if seed == "pooled" else 5
            require(len(values) == expected_n, f"detection endpoint missing observations: {keys}/{seed}")
            successes = int(values.sum())
            p, low, high = wilson_interval(successes, len(values))
            rows.append(
                {
                    "endpoint": "sensor_fault_detection_probability",
                    "family": keys[0],
                    "condition_label": keys[1],
                    "training_seed": seed,
                    "successes": successes,
                    "trials": int(len(values)),
                    "probability": p,
                    "wilson95_low": low,
                    "wilson95_high": high,
                }
            )
    load = points[points.family.eq("load")]
    for label, group in load.groupby("condition_label", sort=True):
        for seed in (*TRAINING_SEEDS, "pooled"):
            block = group if seed == "pooled" else group[group.training_seed.eq(seed)]
            flags = block.c3_post_event_reliability_entries.to_numpy(int) > 0
            expected_n = 15 if seed == "pooled" else 5
            require(len(flags) == expected_n, f"load false-entry endpoint missing observations: {label}/{seed}")
            successes = int(flags.sum())
            p, low, high = wilson_interval(successes, len(flags))
            rows.append(
                {
                    "endpoint": "load_false_reliability_entry_probability",
                    "family": "load",
                    "condition_label": label,
                    "training_seed": seed,
                    "successes": successes,
                    "trials": int(len(flags)),
                    "probability": p,
                    "wilson95_low": low,
                    "wilson95_high": high,
                }
            )
    return pd.DataFrame(rows)


def severity_bootstrap(points: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(PART_B_BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for keys, group in points.groupby(["family", "condition_label"], sort=True):
        by_seed = {
            seed: group[group.training_seed.eq(seed)]
            .sort_values("simulation_seed")
            .c3_minus_b_fault_window_rmse.to_numpy(float)
            for seed in TRAINING_SEEDS
        }
        require(all(len(values) == 5 for values in by_seed.values()), f"severity bootstrap input incomplete: {keys}")
        boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for index in range(BOOTSTRAP_REPLICATES):
            sampled_training = rng.choice(TRAINING_SEEDS, size=3, replace=True)
            cluster_means = []
            for seed in sampled_training:
                values = by_seed[int(seed)]
                cluster_means.append(float(np.mean(rng.choice(values, size=5, replace=True))))
            boot[index] = float(np.mean(cluster_means))
        low, high = np.quantile(boot, [0.025, 0.975])
        seed_means = np.asarray([np.mean(by_seed[seed]) for seed in TRAINING_SEEDS], dtype=float)
        rows.append(
            {
                "endpoint": "mean_paired_c3_minus_b_fault_window_rmse",
                "family": keys[0],
                "condition_label": keys[1],
                "point_estimate_mean_of_training_seed_means": float(seed_means.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_probability_delta_lt_zero": float(np.mean(boot < 0.0)),
                "analysis_seed": PART_B_BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "training_seed_clusters": 3,
                "simulation_seeds_per_cluster": 5,
            }
        )
    return pd.DataFrame(rows)


def current_paired_points(runs: pd.DataFrame) -> pd.DataFrame:
    current = runs[runs.family.eq("current_sensor")]
    rows: list[dict[str, Any]] = []
    for keys, group in current.groupby(["current_case", "training_seed", "simulation_seed"], sort=True):
        require(len(group) == 3 and set(group.controller) == set(CURRENT_CONTROLLERS), f"current controller triplet incomplete: {keys}")
        b = group[group.controller.eq("B_plain_MPC")].iloc[0]
        c3 = group[group.controller.eq("C3_arbitration_MPC")].iloc[0]
        ekf = group[group.controller.eq("E_EKF_virtual_MPC")].iloc[0]
        b_rmse = float(b.fault_window_rmse)
        rows.append(
            {
                "current_case": keys[0],
                "training_seed": int(keys[1]),
                "simulation_seed": int(keys[2]),
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
                "c3_event_frac_phys": float(c3.event_frac_phys),
                "c3_event_frac_main": float(c3.event_frac_main),
                "c3_event_frac_aux": float(c3.event_frac_aux),
                "c3_event_frac_fb": float(c3.event_frac_fb),
                "c3_aux_estimator_rmse": float(c3.aux_speed_rmse_event_window) if pd.notna(c3.aux_speed_rmse_event_window) else np.nan,
                "ekf_estimator_rmse": float(ekf.ekf_speed_rmse_event_window) if pd.notna(ekf.ekf_speed_rmse_event_window) else np.nan,
                "ekf_sub_fraction": float(ekf.sub_fraction),
                "ekf_event_sub_fraction": float(ekf.ekf_event_sub_fraction) if pd.notna(ekf.ekf_event_sub_fraction) else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    require(len(result) == 45, f"expected 45 current controller triplets, found {len(result)}")
    return result


def current_summary(runs: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    current = runs[runs.family.eq("current_sensor")]
    rows: list[dict[str, Any]] = []
    for (current_case, controller, training_seed), group in current.groupby(
        ["current_case", "controller", "training_seed"], sort=True
    ):
        estimator = np.full(len(group), np.nan)
        if controller == "C3_arbitration_MPC":
            preferred = pd.to_numeric(group.aux_speed_rmse_event_window, errors="coerce").to_numpy(float)
            fallback = pd.to_numeric(group.aux_virtual_rmse, errors="coerce").to_numpy(float)
            estimator = np.where(np.isfinite(preferred), preferred, fallback)
        elif controller == "E_EKF_virtual_MPC":
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
                "safety_failure_total": int(
                    sum(pd.to_numeric(group[column]).sum() for column in SAFETY_COLUMNS)
                ),
            }
        )
    for (current_case, training_seed), group in points.groupby(
        ["current_case", "training_seed"], sort=True
    ):
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


def current_bootstrap(points: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(PART_B_BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for current_case, group in points.groupby("current_case", sort=True):
        for effect_column in ("c3_minus_b_fault_window_rmse", "ekf_minus_b_fault_window_rmse"):
            by_seed = {
                seed: group[group.training_seed.eq(seed)].sort_values("simulation_seed")[effect_column].to_numpy(float)
                for seed in TRAINING_SEEDS
            }
            require(all(len(values) == 5 for values in by_seed.values()), f"current bootstrap input incomplete: {current_case}/{effect_column}")
            boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            for index in range(BOOTSTRAP_REPLICATES):
                sampled_training = rng.choice(TRAINING_SEEDS, size=3, replace=True)
                cluster_means = [
                    float(np.mean(rng.choice(by_seed[int(seed)], size=5, replace=True))) for seed in sampled_training
                ]
                boot[index] = float(np.mean(cluster_means))
            low, high = np.quantile(boot, [0.025, 0.975])
            seed_means = [float(np.mean(by_seed[seed])) for seed in TRAINING_SEEDS]
            rows.append(
                {
                    "current_case": current_case,
                    "endpoint": effect_column,
                    "point_estimate_mean_of_training_seed_means": float(np.mean(seed_means)),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "bootstrap_probability_effect_lt_zero": float(np.mean(boot < 0.0)),
                    "analysis_seed": PART_B_BOOTSTRAP_SEED,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "training_seed_clusters": 3,
                    "simulation_seeds_per_cluster": 5,
                }
            )
    return pd.DataFrame(rows)


def combined_effects(points: pd.DataFrame) -> pd.DataFrame:
    combined = points[points.family.eq("combined_bias_load")]
    rows: list[dict[str, Any]] = []
    for label, group in combined.groupby("condition_label", sort=True):
        values = group.c3_minus_b_fault_window_rmse.to_numpy(float)
        rows.append(
            {
                "condition_label": label,
                "aggregation_axis": "pooled",
                "aggregation_value": "all",
                "mean_c3_minus_b_fault_window_rmse": float(values.mean()),
                "favorable_count": int(np.sum(values < 0.0)),
                "total_count": int(len(values)),
            }
        )
        for seed, seed_group in group.groupby("training_seed", sort=True):
            delta = seed_group.c3_minus_b_fault_window_rmse.to_numpy(float)
            rows.append(
                {
                    "condition_label": label,
                    "aggregation_axis": "training_seed",
                    "aggregation_value": str(int(seed)),
                    "mean_c3_minus_b_fault_window_rmse": float(delta.mean()),
                    "favorable_count": int(np.sum(delta < 0.0)),
                    "total_count": int(len(delta)),
                }
            )
        for seed, seed_group in group.groupby("simulation_seed", sort=True):
            delta = seed_group.c3_minus_b_fault_window_rmse.to_numpy(float)
            rows.append(
                {
                    "condition_label": label,
                    "aggregation_axis": "simulation_seed",
                    "aggregation_value": str(int(seed)),
                    "mean_c3_minus_b_fault_window_rmse": float(delta.mean()),
                    "favorable_count": int(np.sum(delta < 0.0)),
                    "total_count": int(len(delta)),
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
    totals = {column: int(pd.to_numeric(runs[column]).sum()) for column in SAFETY_COLUMNS}
    totals["incomplete_runs"] = int((~runs.run_complete.astype(bool)).sum())
    totals["total_failure_safety_count"] = int(sum(totals.values()))
    return pd.DataFrame(rows), totals


def condition_safety_total(runs: pd.DataFrame, family: str, label: str) -> int:
    block = runs[runs.family.eq(family) & runs.condition_label.eq(label)]
    return int(
        sum(int(pd.to_numeric(block[column]).sum()) for column in SAFETY_COLUMNS)
        + int((~block.run_complete.astype(bool)).sum())
    )


def operating_envelope(
    runs: pd.DataFrame,
    points: pd.DataFrame,
    seed_means: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    rules = protocol["operating_envelope_rules"]
    supported = rules["SUPPORTED"]
    limited = rules["LIMITED"]
    require(float(supported["per_training_seed_detection_probability_min"]) == 0.8, "SUPPORTED detection threshold changed")
    require(supported["per_training_seed_mean_paired_delta_C3_minus_B"] == "< 0 for all 3 training seeds", "SUPPORTED delta rule changed")
    require(int(supported["favorable_paired_sign_count_min_of_15"]) == 12, "SUPPORTED favorable-count rule changed")
    require(int(supported["safety_numerical_incomplete_failures"]) == 0, "SUPPORTED safety rule changed")
    require(float(limited["pooled_detection_probability_min"]) == 0.5, "LIMITED detection threshold changed")
    require(limited["pooled_mean_paired_delta_C3_minus_B"] == "< 0", "LIMITED delta rule changed")
    require(int(limited["training_seeds_with_favorable_mean_min_of_3"]) == 2, "LIMITED favorable-seed rule changed")
    require(int(limited["safety_numerical_incomplete_failures"]) == 0, "LIMITED safety rule changed")
    require(limited["only_if_not_SUPPORTED"] is True, "LIMITED precedence changed")

    rows: list[dict[str, Any]] = []
    classified = points[points.family.isin(CLASSIFIED_FAMILIES)]
    for keys, group in classified.groupby(["family", "condition_label"], sort=True):
        family, label = keys
        seed_group = seed_means[seed_means.family.eq(family) & seed_means.condition_label.eq(label)].sort_values("training_seed")
        require(len(seed_group) == 3, f"classification missing seed means: {family}/{label}")
        per_seed_detection = {
            int(row.training_seed): float(row.c3_detection_probability) for row in seed_group.itertuples(index=False)
        }
        require(all(np.isfinite(value) for value in per_seed_detection.values()), f"classification detection data missing: {family}/{label}")
        per_seed_means = {
            int(row.training_seed): float(row.mean_c3_minus_b_fault_window_rmse)
            for row in seed_group.itertuples(index=False)
        }
        pooled_detection = float(group.c3_sensor_fault_detected.astype(bool).mean())
        pooled_delta = float(group.c3_minus_b_fault_window_rmse.mean())
        favorable_count = int(group.favorable_c3_minus_b.sum())
        favorable_seed_count = int(sum(value < 0.0 for value in per_seed_means.values()))
        safety_total = condition_safety_total(runs, family, label)
        supported_checks = {
            "all_training_seed_detection_probability_ge_threshold": all(value >= 0.8 for value in per_seed_detection.values()),
            "all_training_seed_mean_delta_negative": all(value < 0.0 for value in per_seed_means.values()),
            "favorable_pair_count_ge_threshold": favorable_count >= 12,
            "safety_numerical_incomplete_failures_zero": safety_total == 0,
        }
        limited_checks = {
            "pooled_detection_probability_ge_threshold": pooled_detection >= 0.5,
            "pooled_mean_delta_negative": pooled_delta < 0.0,
            "favorable_training_seed_mean_count_ge_threshold": favorable_seed_count >= 2,
            "safety_numerical_incomplete_failures_zero": safety_total == 0,
        }
        classification = "SUPPORTED" if all(supported_checks.values()) else "LIMITED" if all(limited_checks.values()) else "UNSUPPORTED"
        rows.append(
            {
                "family": family,
                "condition_label": label,
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
    require(len(result) == 16, f"operating envelope must classify 16 severity conditions, found {len(result)}")
    return result


def source_recovery_summary(runs: pd.DataFrame) -> pd.DataFrame:
    c3 = runs[
        runs.controller.eq("C3_arbitration_MPC")
        & runs.family.isin(PRIMARY_FAMILIES)
    ]
    rows: list[dict[str, Any]] = []
    for (family, label, training_seed), group in c3.groupby(
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
                "condition_label": label,
                "training_seed": int(training_seed),
                "run_count": int(len(group)),
                "detection_probability": float(detected.mean()) if len(detected) else np.nan,
                "mean_detection_latency_s_when_detected": (
                    float(finite_detection.mean()) if len(finite_detection) else np.nan
                ),
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
                "mean_recovery_latency_s_when_defined": (
                    float(finite_recovery.mean()) if len(finite_recovery) else np.nan
                ),
                "mean_post_event_reliability_active_duration_s_when_defined": (
                    float(finite_post_event_duration.mean())
                    if len(finite_post_event_duration)
                    else np.nan
                ),
                "mean_total_sub_duration_s": float(group.sub_duration_s.mean()),
                "mean_source_switch_episode_duration_s": float(group.avg_episode_duration_s.mean()),
            }
        )
    return pd.DataFrame(rows)


def load_summary(runs: pd.DataFrame, points: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    load_step = float(protocol["primary_sweeps"]["load"]["step_time_s"])
    c3 = runs[runs.family.eq("load") & runs.controller.eq("C3_arbitration_MPC")]
    rows: list[dict[str, Any]] = []
    for label in sorted(c3.condition_label.unique()):
        condition_runs = c3[c3.condition_label.eq(label)]
        condition_points = points[points.family.eq("load") & points.condition_label.eq(label)]
        for seed in (*TRAINING_SEEDS, "pooled"):
            block = condition_runs if seed == "pooled" else condition_runs[condition_runs.training_seed.eq(seed)]
            pair_block = condition_points if seed == "pooled" else condition_points[condition_points.training_seed.eq(seed)]
            flags = block.post_event_reliability_entries.to_numpy(int) > 0
            probability, low, high = wilson_interval(int(flags.sum()), len(flags))
            entry_times = block.first_post_event_reliability_entry_time_s.to_numpy(float)
            latencies = entry_times[np.isfinite(entry_times)] - load_step
            durations = block.post_event_reliability_active_duration_s.to_numpy(float)
            durations = durations[np.isfinite(durations)]
            rows.append(
                {
                    "family": "load",
                    "condition_label": label,
                    "training_seed": seed,
                    "run_count": int(len(block)),
                    "false_entry_runs": int(flags.sum()),
                    "false_entry_probability": probability,
                    "false_entry_wilson95_low": low,
                    "false_entry_wilson95_high": high,
                    "total_post_event_reliability_entries": int(block.post_event_reliability_entries.sum()),
                    "total_reliability_entries_all_times": int(block.reliability_entries.sum()),
                    "mean_substitution_fraction": float(block.sub_fraction.mean()),
                    "mean_entry_latency_from_load_step_s_when_entry": float(np.mean(latencies)) if len(latencies) else np.nan,
                    "mean_post_event_reliability_active_duration_s_when_defined": float(np.mean(durations)) if len(durations) else np.nan,
                    "mean_total_sub_duration_s": float(block.sub_duration_s.mean()),
                    "mean_source_switch_episode_duration_s": float(block.avg_episode_duration_s.mean()),
                    "mean_c3_fault_window_rmse": float(pair_block.c3_fault_window_rmse.mean()),
                    "mean_b_fault_window_rmse": float(pair_block.b_fault_window_rmse.mean()),
                    "mean_tracking_penalty_c3_minus_b": float(pair_block.c3_minus_b_fault_window_rmse.mean()),
                    "favorable_pair_count": int(pair_block.favorable_c3_minus_b.sum()),
                }
            )
    return pd.DataFrame(rows)


def verify_part_b_outputs(protocol: Mapping[str, Any], pair_bindings: Mapping[int, Mapping[str, str]]) -> tuple[pd.DataFrame, dict[str, int]]:
    require(PART_B_RUNS_PATH.is_file(), f"missing required Part B raw matrix: {rel(PART_B_RUNS_PATH)}")
    runs = canonicalize_part_b_runs(pd.read_csv(PART_B_RUNS_PATH, float_precision="round_trip"))
    runs = verify_part_b_matrix(runs, protocol, pair_bindings)
    verify_run_protocol_bindings(runs)
    verify_current_corruption_evidence(runs, protocol)

    points = paired_primary_points(runs)
    seed_means = training_seed_means(points)
    wilson = detection_wilson_table(points)
    bootstrap = severity_bootstrap(points)
    sources = source_recovery_summary(runs)
    current_points = current_paired_points(runs)
    current = current_summary(runs, current_points)
    current_boot = current_bootstrap(current_points)
    combined = combined_effects(points)
    safety, safety_totals = safety_summary(runs)
    envelope = operating_envelope(runs, points, seed_means, protocol)
    load = load_summary(runs, points, protocol)

    expected_tables = {
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
    for name, expected in expected_tables.items():
        compare_frame(expected, RESULT_ROOT / PART_B_OUTPUT_FILES[name], PART_B_OUTPUT_FILES[name])

    summary = read_json(PART_B_SUMMARY_PATH)
    require(summary.get("study") == "final_robustness_hardening", "Part B analysis summary study id changed")
    source = summary.get("source")
    require(isinstance(source, Mapping), "Part B analysis summary missing source")
    source_runs = source.get("part_b_runs_csv")
    require(isinstance(source_runs, Mapping), "Part B summary missing run-matrix binding")
    require(source_runs.get("path") == rel(PART_B_RUNS_PATH), "Part B summary run path changed")
    require(source_runs.get("sha256") == sha256(PART_B_RUNS_PATH), "Part B summary run hash mismatch")
    require(source.get("study_protocol_json", {}).get("sha256") == EXPECTED_PROTOCOL_SHA256, "Part B summary protocol hash mismatch")
    require(source.get("simulation_seed_freeze_json", {}).get("sha256") == EXPECTED_SEED_FREEZE_SHA256, "Part B summary seed-freeze hash mismatch")
    require(source.get("model_pair_manifest_json", {}).get("sha256") == EXPECTED_PAIR_MANIFEST_SHA256, "Part B summary pair-manifest hash mismatch")

    matrix = summary.get("matrix")
    require(isinstance(matrix, Mapping), "Part B summary missing matrix block")
    require(matrix.get("primary_rows") == 600 and matrix.get("current_rows") == 135 and matrix.get("total_rows") == 735, "Part B summary matrix counts mismatch")
    require(matrix.get("paired_primary_points") == 300 and matrix.get("current_controller_triplets") == 45, "Part B summary paired counts mismatch")
    require(matrix.get("exact_cartesian_matrix_verified") is True, "Part B summary lacks exact matrix attestation")
    require(matrix.get("model_calibration_hash_bindings_verified") is True, "Part B summary lacks binding attestation")

    stats = summary.get("statistics")
    require(isinstance(stats, Mapping), "Part B summary missing statistics")
    require(tuple(stats.get("hierarchical_bootstrap_order", [])) == ("training_seed", "simulation_seed"), "Part B summary bootstrap hierarchy mismatch")
    require(stats.get("part_b_bootstrap_rng_seed") == PART_B_BOOTSTRAP_SEED, "Part B summary bootstrap seed mismatch")
    require(stats.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES, "Part B summary bootstrap replicate mismatch")
    require(tuple(summary.get("primary_endpoints", [])) == tuple(protocol["primary_endpoints"]), "Part B primary endpoint list changed")

    outputs = summary.get("outputs")
    require(isinstance(outputs, Mapping), "Part B summary missing output hashes")
    for name, filename in PART_B_OUTPUT_FILES.items():
        path = RESULT_ROOT / filename
        require(path.is_file(), f"missing Part B analysis output: {filename}")
        entry = outputs.get(name)
        require(isinstance(entry, Mapping), f"Part B summary missing output binding for {name}")
        require(entry.get("path") == rel(path), f"Part B summary path mismatch for {name}")
        require(entry.get("sha256") == sha256(path), f"Part B summary hash mismatch for {name}")

    summary_safety = summary.get("safety", {}).get("totals_across_part_b_runs")
    require(isinstance(summary_safety, Mapping), "Part B summary missing safety totals")
    require(dict(summary_safety) == safety_totals, f"Part B summary safety totals mismatch: {summary_safety} != {safety_totals}")
    require(
        summary.get("safety", {}).get("all_failure_safety_counts_zero")
        == (safety_totals["total_failure_safety_count"] == 0),
        "Part B summary zero-failure boolean is inconsistent",
    )
    return envelope, safety_totals


def verify_final_report_consistency(envelope: pd.DataFrame, safety_totals: Mapping[str, int]) -> None:
    require(FINAL_REPORT_PATH.is_file(), f"missing required final study report: {rel(FINAL_REPORT_PATH)}")
    text = FINAL_REPORT_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    required_phrases = (
        "variance decomposition",
        "bias",
        "dropout",
        "drift",
        "load",
        "combined",
        "current",
        "hierarchical",
        "safety",
        "limitations",
    )
    for phrase in required_phrases:
        require(phrase in lowered, f"final report omits required topic: {phrase}")
    for seed in (*TRAINING_SEEDS, *NEW_SIMULATION_SEEDS):
        require(str(seed) in text, f"final report omits required seed {seed}")
    require(str(PART_A_BOOTSTRAP_SEED) in text and str(PART_B_BOOTSTRAP_SEED) in text, "final report omits fixed analysis RNG seeds")
    require("735" in text, "final report does not state the complete 735-run Part B scale")
    classifications = set(envelope.classification.astype(str))
    for classification in classifications:
        require(classification in text, f"final report does not mention observed envelope class {classification}")
    total = int(safety_totals["total_failure_safety_count"])
    if total == 0:
        require("zero" in lowered or "0" in text, "final report does not communicate zero aggregate failure/safety count")


def verify_required_scientific_outputs_present() -> None:
    required = [
        PART_A_DECOMPOSITION_PATH,
        PART_A_BOOTSTRAP_PATH,
        PART_A_SUMMARY_PATH,
        PART_B_RUNS_PATH,
        PART_B_SUMMARY_PATH,
        FINAL_REPORT_PATH,
        *(RESULT_ROOT / filename for filename in PART_B_OUTPUT_FILES.values()),
    ]
    missing = [rel(path) for path in required if not path.is_file()]
    require(not missing, "final robustness study outputs are incomplete; missing: " + ", ".join(missing))


def main() -> None:
    protocol, _ = verify_pinned_protocol()
    verify_prestudy_frozen_hashes()
    verify_required_scientific_outputs_present()
    pair_bindings, _ = load_pair_bindings()
    verify_independent_seed_calibrations()
    verify_part_a_outputs()
    envelope, safety_totals = verify_part_b_outputs(protocol, pair_bindings)
    verify_final_report_consistency(envelope, safety_totals)
    print("PASS: final robustness study independently verified")
    print("HISTORICAL_FROZEN_ARTIFACTS: 43/43 unchanged")
    print("PART_A_PAIRED_CELLS: 75 (15 per scenario)")
    print("PART_B_PRIMARY_RUNS: 600")
    print("PART_B_CURRENT_RUNS: 135")
    print("PART_B_TOTAL_RUNS: 735")
    print(f"PART_B_TOTAL_FAILURE_SAFETY_COUNT: {safety_totals['total_failure_safety_count']}")


if __name__ == "__main__":
    main()
