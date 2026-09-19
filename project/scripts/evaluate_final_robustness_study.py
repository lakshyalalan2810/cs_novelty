"""Run the preregistered final robustness/severity and current-boundary study.

This runner is intentionally fail closed.  Before any model is loaded or any
closed-loop cell is executed it validates the frozen Part-B protocol, the
prospectively selected simulation seeds, every artifact in the prestudy hash
manifest, the three matched training-seed model/calibration bundles, and the
single frozen EKF covariance selection.

The exact scientific matrix is fixed at:

* 600 primary rows: five four-condition sweeps x three training seeds x five
  simulation seeds x B_plain_MPC/C3_arbitration_MPC.
* 135 current-boundary rows: three cases x three training seeds x five
  simulation seeds x B_plain_MPC/C3_arbitration_MPC/E_EKF_virtual_MPC.

Primary output:
    results/final_robustness/final_robustness_runs.csv

No scientific work is performed when ``--preflight-only`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_ekf_closed_loop as frozen_ekf  # noqa: E402
import evaluate_training_seed_robustness as training_eval  # noqa: E402
import evaluate_v3_closed_loop as frozen_v3  # noqa: E402
from auxiliary_sensor_model import AuxiliarySpeedEstimator, auxiliary_predict_online  # noqa: E402
from ekf_observer import AugmentedStateEKF, EKFNumericalError  # noqa: E402
from mpc import LSTMMPC, MPCConfig, PIConfig, PIController  # noqa: E402
from reliability import DualVirtualSensorArbitrator, SensorReliabilityMonitor, load_lstm_model  # noqa: E402


STUDY = "final_robustness_hardening"
EXPECTED_PROTOCOL_SHA256 = "0536abd9bf6aa2d69fcfeae13dcad5f5c238a4d7898552edff1e5d21fa205bad"
TRAINING_SEEDS = (2026, 2027, 2028)
SIMULATION_SEEDS = (59026, 59027, 59028, 59029, 59030)
PRIMARY_CONTROLLERS = ("B_plain_MPC", "C3_arbitration_MPC")
CURRENT_CONTROLLERS = (
    "B_plain_MPC",
    "C3_arbitration_MPC",
    "E_EKF_virtual_MPC",
)
E_CONTROLLER = "E_EKF_virtual_MPC"

RESULT_DIR = PROJECT / "results" / "final_robustness"
PROTOCOL_PATH = RESULT_DIR / "study_protocol.json"
PRESTUDY_MANIFEST_PATH = RESULT_DIR / "prestudy_frozen_hash_manifest.json"
SIMULATION_SEED_FREEZE_PATH = RESULT_DIR / "simulation_seed_freeze.json"
PAIR_MANIFEST_PATH = PROJECT / "results" / "training_seed_robustness" / "model_pair_manifest.json"
EKF_CONFIG_PATH = PROJECT / "results" / "configs" / "ekf_frozen_config.json"
DATASET_PATH = PROJECT / "data" / "processed" / "dc_motor_lstm_dataset.npz"

RUNS_PATH = RESULT_DIR / "final_robustness_runs.csv"
EVENTS_PATH = RESULT_DIR / "final_robustness_reliability_events.csv"
EVALUATION_MANIFEST_PATH = RESULT_DIR / "final_robustness_evaluation_manifest.json"

DT = frozen_v3.DT
CONTROL_STRIDE = frozen_v3.CONTROL_STRIDE
CONTROL_DT = frozen_v3.CONTROL_DT
DURATION = frozen_v3.DURATION
TIME = frozen_v3.TIME
FULL_SCALE = frozen_v3.FULL_SCALE
SPEED_NOISE_STD = frozen_v3.SPEED_NOISE_STD

EXPECTED_PRIMARY_RUNS = 600
EXPECTED_CURRENT_RUNS = 135
EXPECTED_TOTAL_RUNS = EXPECTED_PRIMARY_RUNS + EXPECTED_CURRENT_RUNS

EXPECTED_BIAS = (2.5, 5.0, 10.0, 15.0)
EXPECTED_DROPOUT = (0.1, 0.5, 1.0, 2.0)
EXPECTED_DRIFT = (2.5, 5.0, 10.0, 15.0)
EXPECTED_LOAD = (0.06, 0.10, 0.15, 0.20)
EXPECTED_COMBINED_BIAS = (5.0, 15.0)
EXPECTED_COMBINED_LOAD = (0.10, 0.15)
EXPECTED_CURRENT_CASES = (
    "current_bias_5",
    "current_dropout_2s",
    "speed_bias_5_plus_current_bias_5",
)

# Stable, explicit CSV schema.  Downstream analysis consumes the canonical
# names used here; aliases are included only where historical tooling expects
# them (for example, rate_violations alongside slew_violations).
RUN_COLUMNS = (
    "study",
    "matrix",
    "family",
    "condition_id",
    "current_case",
    "severity_value",
    "bias_percent",
    "dropout_duration_s",
    "drift_final_percent",
    "post_step_load_Nm",
    "load_baseline_Nm",
    "current_bias_percent",
    "current_bias_A",
    "current_reference_scale_A",
    "current_corruption_mode",
    "current_corruption_value_A",
    "current_corruption_mean_A",
    "current_corruption_rmse_A",
    "corrupted_current_sha256",
    "aux_current_input_sha256",
    "ekf_current_input_sha256",
    "event_start_s",
    "event_end_s",
    "event_end_inclusive",
    "speed_fault_present",
    "current_fault_present",
    "training_seed",
    "simulation_seed",
    "seed",
    "controller",
    "science_executed",
    "reuse_source_key",
    "reuse_provenance",
    "run_complete",
    "completed_samples",
    "finite_sample_rate",
    "overall_rmse",
    "overall_mae",
    "fault_window_rmse",
    "sensor_fault_detected",
    "detection_latency_s",
    "first_post_event_reliability_entry_time_s",
    "post_event_reliability_entries",
    "reliability_entries",
    "recovery_events",
    "false_recovery_events",
    "first_post_event_recovery_time_s",
    "recovery_latency_s",
    "recovered_after_event",
    "post_event_reliability_active_duration_s",
    "load_false_entry_latency_s",
    "load_false_entry_count",
    "sub_duration_s",
    "sub_fraction",
    "event_sub_fraction",
    "ekf_event_sub_fraction",
    "switches",
    "source_switch_count",
    "avg_episode_duration_s",
    "frac_phys",
    "frac_main",
    "frac_aux",
    "frac_fb",
    "frac_ekf",
    "event_frac_phys",
    "event_frac_main",
    "event_frac_aux",
    "event_frac_fb",
    "event_frac_ekf",
    "physical_fraction",
    "main_fraction",
    "auxiliary_fraction",
    "fallback_fraction",
    "main_speed_rmse_all_samples",
    "main_speed_rmse_event_window",
    "aux_speed_rmse_all_samples",
    "aux_speed_rmse_event_window",
    "aux_virtual_rmse",
    "ekf_speed_rmse_all_samples",
    "ekf_speed_rmse_event_window",
    "selected_feedback_rmse_event_window",
    "control_effort_u2",
    "control_variation_du2",
    "voltage_violations",
    "slew_violations",
    "rate_violations",
    "optimizer_failures",
    "main_prediction_failures",
    "aux_prediction_failures",
    "ekf_numerical_failures",
    "ekf_nonfinite_failures",
    "ekf_psd_failures",
    "ekf_symmetry_failures",
    "nonfinite_events",
    "incomplete_run_failures",
    "total_failure_count",
    "total_safety_violation_count",
    "prediction_failure_messages",
    "run_failure_message",
    "current_equivalence_mismatch_count",
    "mean_solve_ms",
    "mean_aux_inference_ms",
    "mean_reliability_update_ms",
    "mean_arbitration_ms",
    "mean_ekf_update_ms",
    "mean_control_compute_ms",
    "sim_time_s",
    "controller_delta_vs_B_fault_window_rmse",
    "controller_amplification_vs_B",
    "study_protocol_sha256",
    "prestudy_manifest_sha256",
    "simulation_seed_freeze_sha256",
    "model_pair_manifest_sha256",
    "main_model_sha256",
    "main_config_sha256",
    "aux_model_sha256",
    "aux_config_sha256",
    "sensor_calibration_sha256",
    "v3_calibration_sha256",
    "pair_config_sha256",
    "sensor_calibration_algorithm_sha256",
    "v3_calibration_algorithm_sha256",
    "calibration_algorithm_sha256",
    "ekf_config_sha256",
    "frozen_v3_evaluator_sha256",
    "training_seed_evaluator_sha256",
    "ekf_evaluator_sha256",
)


@dataclass(frozen=True)
class Condition:
    matrix: str
    family: str
    condition_id: str
    current_case: str = ""
    severity_value: float = np.nan
    bias_percent: float = np.nan
    dropout_duration_s: float = np.nan
    drift_final_percent: float = np.nan
    post_step_load_Nm: float = np.nan
    current_bias_percent: float = np.nan
    current_bias_A: float = np.nan
    current_reference_scale_A: float = np.nan
    current_corruption_mode: str = "none"
    current_corruption_value_A: float = np.nan
    event_start_s: float = np.nan
    event_end_s: float = np.nan
    event_end_inclusive: bool = False
    speed_fault_present: bool = False
    current_fault_present: bool = False


_RESOURCE_CACHE: dict[int, dict[str, Any]] = {}
_EKF_RECORD_CACHE: dict[str, Any] | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT.resolve())).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{rel(path)} must contain a JSON object")
    return payload


def _exact_float_tuple(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _assert_float(value: Any, expected: float, label: str, *, atol: float = 1e-15) -> None:
    actual = float(value)
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise RuntimeError(f"{label} changed: expected {expected}, found {actual}")


def _assert_float_tuple(values: Any, expected: tuple[float, ...], label: str) -> None:
    actual = _exact_float_tuple(values)
    if len(actual) != len(expected) or not np.allclose(actual, expected, rtol=0.0, atol=1e-15):
        raise RuntimeError(f"{label} changed: expected {expected}, found {actual}")


def _verify_bound_file(entry: dict[str, Any], expected_path: Path, label: str) -> str:
    raw_path = entry.get("path")
    expected_hash = str(entry.get("sha256", "")).lower()
    if not isinstance(raw_path, str) or len(expected_hash) != 64:
        raise RuntimeError(f"{label} binding is malformed")
    resolved = (PROJECT / raw_path).resolve()
    if resolved != expected_path.resolve():
        raise RuntimeError(f"{label} path changed: {raw_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {raw_path}")
    actual_hash = sha256(resolved)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected_hash}, found {actual_hash}"
        )
    return actual_hash


def validate_protocol() -> tuple[dict[str, Any], dict[str, str]]:
    actual_protocol_hash = sha256(PROTOCOL_PATH)
    if actual_protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "study_protocol.json hash mismatch: "
            f"expected {EXPECTED_PROTOCOL_SHA256}, found {actual_protocol_hash}"
        )
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("study") != STUDY:
        raise RuntimeError("study_protocol.json belongs to the wrong study")
    if protocol.get("status") != "frozen_before_any_part_b_scientific_run":
        raise RuntimeError("study_protocol.json is not prospectively frozen")
    if tuple(int(seed) for seed in protocol.get("training_seeds", [])) != TRAINING_SEEDS:
        raise RuntimeError("training-seed block differs from the frozen design")

    seed_hash = _verify_bound_file(
        protocol.get("simulation_seed_freeze", {}),
        SIMULATION_SEED_FREEZE_PATH,
        "simulation_seed_freeze",
    )
    prestudy_hash = _verify_bound_file(
        protocol.get("prestudy_frozen_hash_manifest", {}),
        PRESTUDY_MANIFEST_PATH,
        "prestudy_frozen_hash_manifest",
    )
    pair_hash = _verify_bound_file(
        protocol.get("model_pair_manifest", {}),
        PAIR_MANIFEST_PATH,
        "model_pair_manifest",
    )
    if tuple(int(seed) for seed in protocol["simulation_seed_freeze"].get("seeds", [])) != SIMULATION_SEEDS:
        raise RuntimeError("protocol simulation-seed list differs from the frozen Part-B block")

    runtime = protocol.get("frozen_runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("protocol is missing frozen_runtime")
    _assert_float(runtime.get("dt_s"), DT, "frozen_runtime.dt_s")
    _assert_float(runtime.get("duration_s"), DURATION, "frozen_runtime.duration_s")
    if int(runtime.get("control_stride", -1)) != CONTROL_STRIDE:
        raise RuntimeError("frozen runtime control_stride changed")
    _assert_float(runtime.get("speed_full_scale_rad_s"), FULL_SCALE, "speed full scale")
    _assert_float(runtime.get("speed_noise_std_rad_s"), SPEED_NOISE_STD, "speed noise std")
    _assert_float(runtime.get("reference_rad_s"), 35.0, "reference")
    _assert_float(runtime.get("load_baseline_Nm"), 0.03, "load baseline")
    _assert_float_tuple(runtime.get("bias_window_s", []), (2.0, 4.0), "bias window")
    _assert_float(runtime.get("dropout_onset_s"), 2.0, "dropout onset")
    _assert_float(runtime.get("drift_onset_s"), 2.0, "drift onset")
    _assert_float(runtime.get("drift_ramp_duration_s"), 4.0, "drift ramp duration")
    _assert_float(runtime.get("load_step_time_s"), 3.0, "load step time")
    _assert_float(runtime.get("combined_event_time_s"), 3.0, "combined event time")
    if runtime.get("mpc") != "unchanged frozen V3 LSTM-MPC configuration":
        raise RuntimeError("protocol MPC declaration changed")

    if tuple(protocol.get("primary_controllers", [])) != PRIMARY_CONTROLLERS:
        raise RuntimeError("primary controller set changed")
    sweeps = protocol.get("primary_sweeps")
    if not isinstance(sweeps, dict):
        raise RuntimeError("protocol is missing primary_sweeps")
    _assert_float_tuple(sweeps["bias"].get("levels_percent", []), EXPECTED_BIAS, "bias levels")
    _assert_float(sweeps["bias"].get("onset_s"), 2.0, "bias onset")
    _assert_float(sweeps["bias"].get("end_s"), 4.0, "bias end")
    _assert_float_tuple(
        sweeps["dropout"].get("durations_s", []), EXPECTED_DROPOUT, "dropout durations"
    )
    _assert_float(sweeps["dropout"].get("onset_s"), 2.0, "dropout onset")
    _assert_float_tuple(
        sweeps["drift"].get("final_percent", []), EXPECTED_DRIFT, "drift levels"
    )
    _assert_float(sweeps["drift"].get("onset_s"), 2.0, "drift onset")
    _assert_float(sweeps["drift"].get("ramp_duration_s"), 4.0, "drift ramp")
    _assert_float_tuple(
        sweeps["load"].get("post_step_load_Nm", []), EXPECTED_LOAD, "load levels"
    )
    _assert_float(sweeps["load"].get("baseline_Nm"), 0.03, "load baseline")
    _assert_float(sweeps["load"].get("step_time_s"), 3.0, "load step")
    if sweeps["load"].get("speed_sensor_fault") is not False:
        raise RuntimeError("load sweep unexpectedly declares a speed-sensor fault")
    combined = sweeps["combined_bias_load"]
    _assert_float_tuple(
        combined.get("bias_percent", []), EXPECTED_COMBINED_BIAS, "combined bias levels"
    )
    _assert_float_tuple(
        combined.get("post_step_load_Nm", []),
        EXPECTED_COMBINED_LOAD,
        "combined load levels",
    )
    _assert_float(combined.get("event_time_s"), 3.0, "combined event onset")
    if combined.get("grid_expansion_forbidden") is not True:
        raise RuntimeError("combined grid expansion lock changed")

    current = protocol.get("current_sensor_boundary")
    if not isinstance(current, dict):
        raise RuntimeError("protocol is missing current_sensor_boundary")
    if tuple(current.get("controllers", [])) != CURRENT_CONTROLLERS:
        raise RuntimeError("current-boundary controller set changed")
    _assert_float(current.get("current_reference_scale_A"), 5.122337818145752, "current scale")
    _assert_float(current.get("bias_percent_of_reference_scale"), 5.0, "current bias percent")
    _assert_float(current.get("bias_A"), 0.2561168909072876, "current bias A")
    _assert_float_tuple(current.get("window_s", []), (2.0, 4.0), "current fault window")
    cases = current.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("current_sensor_boundary.cases must be a list")
    case_names = tuple(str(case.get("name")) for case in cases if isinstance(case, dict))
    if case_names != EXPECTED_CURRENT_CASES:
        raise RuntimeError(f"current-boundary cases changed: {case_names}")
    expected_equivalence = (
        "the same corrupted current sample at each time step is fed to the C3 auxiliary-LSTM "
        "current channel and the EKF current measurement; all other measurements remain unchanged"
    )
    if current.get("equivalence_requirement") != expected_equivalence:
        raise RuntimeError("current-channel equivalence requirement changed")

    matrix = protocol.get("expected_matrix")
    if not isinstance(matrix, dict):
        raise RuntimeError("protocol is missing expected_matrix")
    if int(matrix.get("primary_runs", -1)) != EXPECTED_PRIMARY_RUNS:
        raise RuntimeError("protocol primary matrix is no longer 600 cells")
    if int(matrix.get("current_runs", -1)) != EXPECTED_CURRENT_RUNS:
        raise RuntimeError("protocol current matrix is no longer 135 cells")
    if int(matrix.get("maximum_total_runs", -1)) != EXPECTED_TOTAL_RUNS:
        raise RuntimeError("protocol total matrix is no longer 735 cells")
    if matrix.get("duplicate_cell_reruns_forbidden") is not True:
        raise RuntimeError("duplicate-cell rerun lock changed")

    statistics = protocol.get("statistics")
    if not isinstance(statistics, dict):
        raise RuntimeError("protocol is missing statistics")
    if statistics.get("paired_controller_comparison") is not True:
        raise RuntimeError("paired-controller comparison flag changed")
    if tuple(statistics.get("hierarchical_bootstrap_order", [])) != (
        "training_seed",
        "simulation_seed",
    ):
        raise RuntimeError("hierarchical bootstrap order changed")
    if int(statistics.get("part_a_bootstrap_rng_seed", -1)) != 20260917:
        raise RuntimeError("Part-A bootstrap seed changed")
    if int(statistics.get("part_b_bootstrap_rng_seed", -1)) != 20260918:
        raise RuntimeError("Part-B bootstrap seed changed")
    if int(statistics.get("bootstrap_replicates", -1)) != 20_000:
        raise RuntimeError("bootstrap replicate count changed")
    _assert_float_tuple(
        statistics.get("bootstrap_ci_percent", []), (2.5, 97.5), "bootstrap CI percent"
    )
    if statistics.get("detection_interval") != "Wilson 95%":
        raise RuntimeError("detection interval changed")
    if statistics.get("time_samples_independent") is not False:
        raise RuntimeError("protocol now treats time samples as independent")

    return protocol, {
        "study_protocol_sha256": actual_protocol_hash,
        "prestudy_manifest_sha256": prestudy_hash,
        "simulation_seed_freeze_sha256": seed_hash,
        "model_pair_manifest_sha256": pair_hash,
    }


def validate_prestudy_hash_manifest() -> dict[str, Any]:
    manifest = read_json(PRESTUDY_MANIFEST_PATH)
    if manifest.get("study") != STUDY:
        raise RuntimeError("prestudy hash manifest belongs to the wrong study")
    if manifest.get("status") != "frozen_before_part_a_or_part_b_outputs":
        raise RuntimeError("prestudy hash manifest was not frozen before Part B")
    artifacts = manifest.get("frozen_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("prestudy hash manifest has no frozen_artifacts")
    if int(manifest.get("artifact_count", -1)) != len(artifacts):
        raise RuntimeError("prestudy hash manifest artifact_count is inconsistent")
    for raw_path, record in artifacts.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"malformed prestudy record: {raw_path}")
        path = (PROJECT / raw_path).resolve()
        try:
            path.relative_to(PROJECT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"prestudy path escapes project: {raw_path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"frozen prestudy artifact is missing: {raw_path}")
        expected_hash = str(record.get("sha256", "")).lower()
        if len(expected_hash) != 64 or sha256(path) != expected_hash:
            raise RuntimeError(f"prestudy artifact hash mismatch: {raw_path}")
        expected_bytes = int(record.get("bytes", -1))
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"prestudy artifact byte-size mismatch: {raw_path}")
    return {
        "artifact_count": len(artifacts),
        "all_hashes_match": True,
        "manifest_sha256": sha256(PRESTUDY_MANIFEST_PATH),
    }


def validate_simulation_seed_freeze(protocol: dict[str, Any]) -> tuple[int, ...]:
    freeze = read_json(SIMULATION_SEED_FREEZE_PATH)
    if freeze.get("study") != STUDY:
        raise RuntimeError("simulation seed freeze belongs to the wrong study")
    if freeze.get("status") != "frozen_before_any_part_b_scientific_run":
        raise RuntimeError("simulation seed freeze status changed")
    selected = tuple(int(seed) for seed in freeze.get("selected_simulation_seeds", []))
    if selected != SIMULATION_SEEDS or len(set(selected)) != len(selected):
        raise RuntimeError(f"selected Part-B simulation seeds changed: {selected}")
    declared = tuple(int(seed) for seed in protocol["simulation_seed_freeze"]["seeds"])
    if selected != declared:
        raise RuntimeError("protocol and seed-freeze simulation seeds disagree")
    prior = freeze.get("reserved_and_prior_seed_blocks")
    if not isinstance(prior, dict):
        raise RuntimeError("seed freeze is missing prior/reserved blocks")
    prior_values = {int(seed) for block in prior.values() for seed in block}
    if prior_values.intersection(selected):
        raise RuntimeError("selected Part-B seeds overlap a prior/reserved seed block")
    scan = freeze.get("provenance_scan")
    if not isinstance(scan, dict):
        raise RuntimeError("seed freeze is missing provenance scan")
    for key in (
        "strict_text_matches",
        "filename_matches",
        "csv_numeric_matches",
        "npy_npz_numeric_matches",
    ):
        if scan.get(key) != []:
            raise RuntimeError(f"selected Part-B seed provenance is contaminated: {key}")
    return selected


def validate_dataset_current_scale(protocol: dict[str, Any]) -> dict[str, Any]:
    ekf_record = read_json(EKF_CONFIG_PATH)
    provenance = ekf_record.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("frozen EKF config is missing provenance")
    expected_dataset_hash = str(provenance.get("dataset_sha256", ""))
    actual_dataset_hash = sha256(DATASET_PATH)
    if actual_dataset_hash != expected_dataset_hash:
        raise RuntimeError("dataset hash differs from frozen EKF calibration provenance")
    with np.load(DATASET_PATH, allow_pickle=False) as data:
        current = np.asarray(data["current"], dtype=float)
    boundary = protocol["current_sensor_boundary"]
    scale = float(np.max(np.abs(current)))
    expected_scale = float(boundary["current_reference_scale_A"])
    if not np.isclose(scale, expected_scale, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"current reference scale changed: expected {expected_scale}, found {scale}")
    actual_range = (float(np.min(current)), float(np.max(current)))
    declared_range = tuple(float(value) for value in boundary["frozen_dataset_current_range_A"])
    if not np.allclose(actual_range, declared_range, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"dataset current range changed: expected {declared_range}, found {actual_range}")
    abs_p99 = float(np.quantile(np.abs(current), 0.99))
    _assert_float(
        abs_p99,
        float(boundary["frozen_dataset_current_abs_p99_A"]),
        "dataset current abs p99",
        atol=1e-12,
    )
    return {
        "dataset_sha256": actual_dataset_hash,
        "current_reference_scale_A": scale,
        "current_range_A": list(actual_range),
        "current_abs_p99_A": abs_p99,
    }


def validate_training_seed_bundles() -> dict[int, training_eval.PairBundle]:
    manifest_hash = sha256(PAIR_MANIFEST_PATH)
    bundles: dict[int, training_eval.PairBundle] = {}
    dataset_hash = sha256(DATASET_PATH)
    for training_seed in TRAINING_SEEDS:
        bundle = training_eval.load_pair_bundle(training_seed)
        if bundle.manifest_sha256 != manifest_hash:
            raise RuntimeError(f"training seed {training_seed} pair-manifest hash differs")
        sensor_payload = read_json(bundle.sensor_calibration.path)
        v3_payload = read_json(bundle.v3_calibration.path)
        if sensor_payload.get("dataset_sha256") != dataset_hash:
            raise RuntimeError(f"training seed {training_seed} sensor calibration dataset binding changed")
        if v3_payload.get("dataset_sha256") != dataset_hash:
            raise RuntimeError(f"training seed {training_seed} V3 calibration dataset binding changed")
        bundles[training_seed] = bundle
    return bundles


def validate_ekf_lock() -> dict[str, Any]:
    record, config = frozen_ekf.load_frozen_ekf_config(EKF_CONFIG_PATH)
    runtime_hashes = frozen_ekf.verify_frozen_runtime_sources()
    plan_lock = frozen_ekf.verify_closed_loop_plan()
    if not np.isclose(config.dt, DT, rtol=0.0, atol=1e-15):
        raise RuntimeError("frozen EKF timestep differs from the final-study timestep")
    return {
        "ekf_config_sha256": sha256(EKF_CONFIG_PATH),
        "selected_candidate_id": str(record["selected_candidate"]["candidate_id"]),
        "runtime_hashes": runtime_hashes,
        "closed_loop_plan": plan_lock,
    }


def _condition_id(prefix: str, *values: tuple[str, float]) -> str:
    parts = [prefix]
    for label, value in values:
        parts.append(f"{label}_{float(value):g}")
    return "_".join(parts)


def build_conditions(protocol: dict[str, Any]) -> tuple[tuple[Condition, ...], tuple[Condition, ...]]:
    sweeps = protocol["primary_sweeps"]
    primary: list[Condition] = []
    for value in sweeps["bias"]["levels_percent"]:
        pct = float(value)
        primary.append(
            Condition(
                matrix="primary",
                family="bias",
                condition_id=_condition_id("bias", ("pct", pct)),
                severity_value=pct,
                bias_percent=pct,
                event_start_s=2.0,
                event_end_s=4.0,
                speed_fault_present=True,
            )
        )
    for value in sweeps["dropout"]["durations_s"]:
        duration = float(value)
        primary.append(
            Condition(
                matrix="primary",
                family="dropout",
                condition_id=_condition_id("dropout", ("s", duration)),
                severity_value=duration,
                dropout_duration_s=duration,
                event_start_s=2.0,
                event_end_s=2.0 + duration,
                speed_fault_present=True,
            )
        )
    for value in sweeps["drift"]["final_percent"]:
        pct = float(value)
        primary.append(
            Condition(
                matrix="primary",
                family="drift",
                condition_id=_condition_id("drift", ("pct", pct)),
                severity_value=pct,
                drift_final_percent=pct,
                event_start_s=2.0,
                event_end_s=DURATION,
                event_end_inclusive=True,
                speed_fault_present=True,
            )
        )
    for value in sweeps["load"]["post_step_load_Nm"]:
        load = float(value)
        primary.append(
            Condition(
                matrix="primary",
                family="load",
                condition_id=_condition_id("load", ("Nm", load)),
                severity_value=load,
                post_step_load_Nm=load,
                event_start_s=3.0,
                event_end_s=DURATION,
                event_end_inclusive=True,
            )
        )
    for bias_value in sweeps["combined_bias_load"]["bias_percent"]:
        for load_value in sweeps["combined_bias_load"]["post_step_load_Nm"]:
            bias = float(bias_value)
            load = float(load_value)
            primary.append(
                Condition(
                    matrix="primary",
                    family="combined_bias_load",
                    condition_id=_condition_id("combined", ("bias_pct", bias), ("load_Nm", load)),
                    bias_percent=bias,
                    post_step_load_Nm=load,
                    event_start_s=3.0,
                    event_end_s=DURATION,
                    event_end_inclusive=True,
                    speed_fault_present=True,
                )
            )

    boundary = protocol["current_sensor_boundary"]
    current_bias = float(boundary["bias_A"])
    current_scale = float(boundary["current_reference_scale_A"])
    current_pct = float(boundary["bias_percent_of_reference_scale"])
    current = (
        Condition(
            matrix="current_boundary",
            family="current_sensor",
            condition_id="current_bias_5",
            current_case="current_bias_5",
            severity_value=current_pct,
            current_bias_percent=current_pct,
            current_bias_A=current_bias,
            current_reference_scale_A=current_scale,
            current_corruption_mode="additive_bias",
            current_corruption_value_A=current_bias,
            event_start_s=2.0,
            event_end_s=4.0,
            current_fault_present=True,
        ),
        Condition(
            matrix="current_boundary",
            family="current_sensor",
            condition_id="current_dropout_2s",
            current_case="current_dropout_2s",
            severity_value=2.0,
            dropout_duration_s=2.0,
            current_reference_scale_A=current_scale,
            current_corruption_mode="dropout_zero",
            current_corruption_value_A=0.0,
            event_start_s=2.0,
            event_end_s=4.0,
            current_fault_present=True,
        ),
        Condition(
            matrix="current_boundary",
            family="current_sensor",
            condition_id="speed_bias_5_plus_current_bias_5",
            current_case="speed_bias_5_plus_current_bias_5",
            bias_percent=5.0,
            current_bias_percent=current_pct,
            current_bias_A=current_bias,
            current_reference_scale_A=current_scale,
            current_corruption_mode="additive_bias",
            current_corruption_value_A=current_bias,
            event_start_s=2.0,
            event_end_s=4.0,
            speed_fault_present=True,
            current_fault_present=True,
        ),
    )
    if len(primary) != 20 or len(current) != 3:
        raise RuntimeError(f"condition grid changed: primary={len(primary)}, current={len(current)}")
    return tuple(primary), current


def build_tasks(
    primary_conditions: tuple[Condition, ...],
    current_conditions: tuple[Condition, ...],
    bundles: dict[int, training_eval.PairBundle],
) -> tuple[list[tuple[Condition, int, int, str, dict[str, Any]]], list[tuple[Condition, int, int, str, dict[str, Any]]]]:
    payloads = {seed: bundle.to_worker_payload() for seed, bundle in bundles.items()}
    primary_tasks = [
        (condition, training_seed, simulation_seed, controller, payloads[training_seed])
        for condition in primary_conditions
        for training_seed in TRAINING_SEEDS
        for simulation_seed in SIMULATION_SEEDS
        for controller in PRIMARY_CONTROLLERS
    ]
    current_tasks = [
        (condition, training_seed, simulation_seed, controller, payloads[training_seed])
        for condition in current_conditions
        for training_seed in TRAINING_SEEDS
        for simulation_seed in SIMULATION_SEEDS
        for controller in CURRENT_CONTROLLERS
    ]
    if len(primary_tasks) != EXPECTED_PRIMARY_RUNS:
        raise RuntimeError(f"primary task matrix changed: {len(primary_tasks)}")
    if len(current_tasks) != EXPECTED_CURRENT_RUNS:
        raise RuntimeError(f"current-boundary task matrix changed: {len(current_tasks)}")
    primary_keys = {
        (task[0].family, task[0].condition_id, task[1], task[2], task[3])
        for task in primary_tasks
    }
    current_keys = {
        (task[0].current_case, task[1], task[2], task[3]) for task in current_tasks
    }
    if len(primary_keys) != EXPECTED_PRIMARY_RUNS or len(current_keys) != EXPECTED_CURRENT_RUNS:
        raise RuntimeError("task construction produced duplicate scientific cells")
    return primary_tasks, current_tasks


def build_execution_tasks(
    primary_tasks: list[tuple[Condition, int, int, str, dict[str, Any]]],
    current_tasks: list[tuple[Condition, int, int, str, dict[str, Any]]],
) -> tuple[
    list[tuple[Condition, int, int, str, dict[str, Any]]],
    list[tuple[Condition, int, int, str, dict[str, Any]]],
]:
    """Return unique scientific executions and labeled rows materialized by reuse.

    B does not consume armature current.  Therefore, for each matched
    training/simulation seed pair, B/current_bias_5 and B/current_dropout_2s
    have the same closed-loop trajectory.  B/speed_bias_5_plus_current_bias_5
    is likewise identical to the already executed primary 5% speed-bias cell.
    The protocol still requires all 735 labeled rows, so those 30 labels are
    materialized after execution with explicit provenance and exact bundle
    identity checks.
    """

    reused: list[tuple[Condition, int, int, str, dict[str, Any]]] = []
    unique_current: list[tuple[Condition, int, int, str, dict[str, Any]]] = []
    for task in current_tasks:
        condition, _training_seed, _simulation_seed, controller, _bundle = task
        if controller == "B_plain_MPC" and condition.current_case in {
            "current_dropout_2s",
            "speed_bias_5_plus_current_bias_5",
        }:
            reused.append(task)
        else:
            unique_current.append(task)
    executions = [*primary_tasks, *unique_current]
    if len(executions) != 705 or len(reused) != 30:
        raise RuntimeError(
            f"reuse plan changed: unique executions={len(executions)}, reused labels={len(reused)}"
        )
    return executions, reused


def run_preflight() -> tuple[dict[str, Any], dict[int, training_eval.PairBundle], dict[str, Any]]:
    protocol, protocol_hashes = validate_protocol()
    prestudy = validate_prestudy_hash_manifest()
    simulation_seeds = validate_simulation_seed_freeze(protocol)
    dataset = validate_dataset_current_scale(protocol)
    bundles = validate_training_seed_bundles()
    ekf = validate_ekf_lock()
    primary_conditions, current_conditions = build_conditions(protocol)
    primary_tasks, current_tasks = build_tasks(primary_conditions, current_conditions, bundles)
    execution_tasks, reused_tasks = build_execution_tasks(primary_tasks, current_tasks)
    report = {
        "study": STUDY,
        "status": "PREFLIGHT_OK",
        "protocol_hashes": protocol_hashes,
        "prestudy": prestudy,
        "simulation_seeds": list(simulation_seeds),
        "training_seeds": list(TRAINING_SEEDS),
        "dataset_current_scale": dataset,
        "ekf_lock": ekf,
        "pair_bindings": {
            str(seed): {
                "main_model_sha256": bundle.main_model.sha256,
                "main_config_sha256": bundle.main_config.sha256,
                "aux_model_sha256": bundle.aux_model.sha256,
                "aux_config_sha256": bundle.aux_config.sha256,
                "sensor_calibration_sha256": bundle.sensor_calibration.sha256,
                "v3_calibration_sha256": bundle.v3_calibration.sha256,
                "pair_config_sha256": bundle.pair_config.sha256,
            }
            for seed, bundle in bundles.items()
        },
        "matrix": {
            "primary_conditions": len(primary_conditions),
            "current_conditions": len(current_conditions),
            "primary_tasks": len(primary_tasks),
            "current_tasks": len(current_tasks),
            "total_tasks": len(primary_tasks) + len(current_tasks),
            "unique_simulation_executions": len(execution_tasks),
            "reuse_materialized_labels": len(reused_tasks),
            "duplicate_cells": 0,
        },
        "statistics_lock": {
            "paired_controller_comparison": True,
            "hierarchical_bootstrap_order": ["training_seed", "simulation_seed"],
            "part_a_bootstrap_rng_seed": 20260917,
            "part_b_bootstrap_rng_seed": 20260918,
            "bootstrap_replicates": 20_000,
            "bootstrap_ci_percent": [2.5, 97.5],
            "detection_interval": "Wilson 95%",
            "time_samples_independent": False,
        },
        "output_path": rel(RUNS_PATH),
        "csv_schema": list(RUN_COLUMNS),
    }
    return protocol, bundles, report


def _worker_resources(training_seed: int, bundle: dict[str, Any]) -> dict[str, Any]:
    if training_seed in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[training_seed]
    if int(bundle["training_seed"]) != training_seed:
        raise RuntimeError("worker task training seed does not match artifact bundle")
    torch.set_num_threads(1)
    device = "cpu"
    main_model, main_cfg = load_lstm_model(
        Path(bundle["main_model"]["path"]),
        Path(bundle["main_config"]["path"]),
        device=device,
    )
    aux_cfg = read_json(Path(bundle["aux_config"]["path"]))
    sensor_payload = read_json(Path(bundle["sensor_calibration"]["path"]))
    v3_payload = read_json(Path(bundle["v3_calibration"]["path"]))
    training_eval._assert_payload_seed(main_cfg, training_seed, "main config", require=True)
    training_eval._assert_payload_seed(aux_cfg, training_seed, "aux config", require=True)
    training_eval._assert_payload_seed(
        sensor_payload, training_seed, "sensor calibration", require=True
    )
    training_eval._assert_payload_seed(v3_payload, training_seed, "V3 calibration", require=True)
    aux_model = AuxiliarySpeedEstimator(**aux_cfg["model"]).to(device)
    aux_model.load_state_dict(
        torch.load(Path(bundle["aux_model"]["path"]), map_location=device, weights_only=True)
    )
    aux_model.eval()
    resources = {
        "main_model": main_model,
        "main_cfg": main_cfg,
        "aux_model": aux_model,
        "aux_cfg": aux_cfg,
        "sensor_cfg": training_eval._sensor_parameters(sensor_payload),
        "v3_cfg": training_eval._v3_parameters(v3_payload),
    }
    _RESOURCE_CACHE[training_seed] = resources
    return resources


def _frozen_ekf_record() -> dict[str, Any]:
    global _EKF_RECORD_CACHE
    if _EKF_RECORD_CACHE is None:
        _EKF_RECORD_CACHE = read_json(EKF_CONFIG_PATH)
    return _EKF_RECORD_CACHE


def _make_sensor(sensor_cfg: dict[str, Any], v3_cfg: dict[str, Any], active: bool) -> SensorReliabilityMonitor:
    return SensorReliabilityMonitor(
        sensor_cfg["instant_threshold"],
        sensor_cfg["center"],
        sensor_cfg["allowance"],
        sensor_cfg["threshold"],
        sensor_cfg["enter_count"],
        sensor_cfg["exit_count"],
        aux_recovery_gate=v3_cfg["aux_recovery_gate"] if active else None,
    )


def _make_arbitrator(v3_cfg: dict[str, Any]) -> DualVirtualSensorArbitrator:
    return DualVirtualSensorArbitrator(
        **{key: value for key, value in v3_cfg.items() if key != "aux_recovery_gate"}
    )


def _make_mpc(main_model: Any, main_norm: dict[str, Any]) -> tuple[LSTMMPC, MPCConfig, PIController]:
    mpc_config = MPCConfig(
        horizon=20,
        tracking_weight=1.0,
        move_weight=0.5,
        voltage_limits=(0.0, 12.0),
        max_voltage_step=2.0,
        max_iterations=8,
        tolerance=0.05,
        move_blocks=(5, 15),
        warm_start=True,
        control_interval_steps=5,
    )
    pi = PIController(PIConfig(0.35, 0.8, (0.0, 12.0), 2.0))
    return LSTMMPC(main_model, main_norm, 20, mpc_config), mpc_config, pi


def _event_mask(condition: Condition) -> np.ndarray:
    mask = TIME >= condition.event_start_s
    if np.isfinite(condition.event_end_s):
        if condition.event_end_inclusive:
            mask &= TIME <= condition.event_end_s + 1e-12
        else:
            mask &= TIME < condition.event_end_s - 1e-12 / 2.0
    return mask


def _speed_measurement(
    condition: Condition,
    time_s: float,
    true_speed: float,
    base_noise: float,
) -> float:
    measured = float(true_speed + base_noise)
    if condition.family == "bias" and 2.0 <= time_s < 4.0:
        measured += condition.bias_percent / 100.0 * FULL_SCALE
    elif condition.family == "dropout" and 2.0 <= time_s < 2.0 + condition.dropout_duration_s:
        measured = 0.0
    elif condition.family == "drift" and time_s >= 2.0:
        ramp = min(1.0, (time_s - 2.0) / 4.0)
        measured += condition.drift_final_percent / 100.0 * FULL_SCALE * ramp
    elif condition.family == "combined_bias_load" and time_s >= 3.0:
        # Frozen combined-fault semantics: the bias begins at t=3 and persists
        # with the load step through the end of the 6 s run.
        measured += condition.bias_percent / 100.0 * FULL_SCALE
    elif (
        condition.current_case == "speed_bias_5_plus_current_bias_5"
        and 2.0 <= time_s < 4.0
    ):
        measured += 0.05 * FULL_SCALE
    return measured


def _current_measurement(condition: Condition, time_s: float, true_current: float) -> float:
    if not condition.current_fault_present or not (2.0 <= time_s < 4.0):
        return float(true_current)
    if condition.current_corruption_mode == "additive_bias":
        return float(true_current + condition.current_bias_A)
    if condition.current_corruption_mode == "dropout_zero":
        return 0.0
    raise RuntimeError(f"unsupported current corruption mode: {condition.current_corruption_mode}")


def _load_value(condition: Condition, time_s: float) -> float:
    if condition.family in {"load", "combined_bias_load"} and time_s >= 3.0:
        return float(condition.post_step_load_Nm)
    return 0.03


def _rmse(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(truth), dtype=bool)
    selected = np.asarray(mask, dtype=bool) & np.isfinite(estimate) & np.isfinite(truth)
    if not np.any(selected):
        return np.nan
    error = np.asarray(estimate, dtype=float)[selected] - np.asarray(truth, dtype=float)[selected]
    return float(np.sqrt(np.mean(error**2)))


def _float64_sequence_sha256(values: list[float]) -> str:
    array = np.asarray(values, dtype=np.dtype("<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _sequence_mismatch_count(reference: list[float], consumer: list[float]) -> int:
    common = min(len(reference), len(consumer))
    mismatches = abs(len(reference) - len(consumer))
    if common:
        reference_array = np.asarray(reference[:common], dtype=np.float64)
        consumer_array = np.asarray(consumer[:common], dtype=np.float64)
        mismatches += int(np.sum(reference_array.view(np.uint64) != consumer_array.view(np.uint64)))
    return mismatches


def _classify_ekf_failure(message: str) -> dict[str, int]:
    lower = message.lower()
    return {
        "ekf_numerical_failures": 1,
        "ekf_nonfinite_failures": int(
            "nonfinite" in lower or ("finite" in lower and "must be finite" in lower)
        ),
        "ekf_psd_failures": int(
            "positive semidefinite" in lower or "eigenvalue" in lower
        ),
        "ekf_symmetry_failures": int("asymmetric" in lower or "symmetr" in lower),
    }


def _pair_metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "main_model_sha256": bundle["main_model"]["sha256"],
        "main_config_sha256": bundle["main_config"]["sha256"],
        "aux_model_sha256": bundle["aux_model"]["sha256"],
        "aux_config_sha256": bundle["aux_config"]["sha256"],
        "sensor_calibration_sha256": bundle["sensor_calibration"]["sha256"],
        "v3_calibration_sha256": bundle["v3_calibration"]["sha256"],
        "pair_config_sha256": bundle["pair_config"]["sha256"],
        "sensor_calibration_algorithm_sha256": bundle[
            "sensor_calibration_algorithm_sha256"
        ],
        "v3_calibration_algorithm_sha256": bundle["v3_calibration_algorithm_sha256"],
        "calibration_algorithm_sha256": bundle["calibration_algorithm_sha256"],
    }


def _condition_metadata(condition: Condition) -> dict[str, Any]:
    return {
        "study": STUDY,
        "matrix": condition.matrix,
        "family": condition.family,
        "condition_id": condition.condition_id,
        "current_case": condition.current_case,
        "severity_value": condition.severity_value,
        "bias_percent": condition.bias_percent,
        "dropout_duration_s": condition.dropout_duration_s,
        "drift_final_percent": condition.drift_final_percent,
        "post_step_load_Nm": condition.post_step_load_Nm,
        "load_baseline_Nm": 0.03,
        "current_bias_percent": condition.current_bias_percent,
        "current_bias_A": condition.current_bias_A,
        "current_reference_scale_A": condition.current_reference_scale_A,
        "current_corruption_mode": condition.current_corruption_mode,
        "current_corruption_value_A": condition.current_corruption_value_A,
        "event_start_s": condition.event_start_s,
        "event_end_s": condition.event_end_s,
        "event_end_inclusive": condition.event_end_inclusive,
        "speed_fault_present": condition.speed_fault_present,
        "current_fault_present": condition.current_fault_present,
    }


def _post_event_active_duration(
    active: np.ndarray, entry_indices: np.ndarray, event_start_s: float
) -> float:
    post_entries = entry_indices[TIME[entry_indices] >= event_start_s - 1e-12]
    if not len(post_entries):
        return np.nan
    entry = int(post_entries[0])
    later_recovery = np.flatnonzero(active[entry:-1] & ~active[entry + 1 :]) + entry + 1
    if len(later_recovery):
        return float(TIME[int(later_recovery[0])] - TIME[entry])
    return float(max(0.0, DURATION - TIME[entry]))


def simulate_final_run(
    args: tuple[Condition, int, int, str, dict[str, Any]],
) -> dict[str, Any]:
    condition, training_seed, simulation_seed, controller, bundle = args
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported training seed {training_seed}")
    if simulation_seed not in SIMULATION_SEEDS:
        raise ValueError(f"simulation seed {simulation_seed} is outside the frozen block")
    allowed = CURRENT_CONTROLLERS if condition.matrix == "current_boundary" else PRIMARY_CONTROLLERS
    if controller not in allowed:
        raise ValueError(f"controller {controller} is invalid for {condition.matrix}")

    resources = _worker_resources(training_seed, bundle)
    main_model = resources["main_model"]
    main_cfg = resources["main_cfg"]
    aux_model = resources["aux_model"]
    aux_cfg = resources["aux_cfg"]
    sensor_cfg = resources["sensor_cfg"]
    v3_cfg = resources["v3_cfg"]
    use_aux = controller in {"C3_arbitration_MPC", E_CONTROLLER}
    sensor = _make_sensor(sensor_cfg, v3_cfg, active=use_aux)
    arbitrator = _make_arbitrator(v3_cfg)
    mpc, mpc_config, pi = _make_mpc(main_model, main_cfg["normalization"])

    observer: AugmentedStateEKF | None = None
    if controller == E_CONTROLLER:
        observer = AugmentedStateEKF(frozen_ekf.ekf_config_from_frozen(_frozen_ekf_record()))

    rng = np.random.default_rng(simulation_seed)
    base_noise = rng.normal(0.0, SPEED_NOISE_STD, len(TIME))
    state = np.zeros(2, dtype=float)
    prev_voltage = 0.0
    history = np.zeros((20, 2), dtype=np.float32)
    aux_history = np.zeros((20, 2), dtype=np.float32)
    ref = np.full(len(TIME), 35.0, dtype=float)
    event_mask = _event_mask(condition)

    size = len(TIME)
    true_speed_arr = np.zeros(size)
    true_current_arr = np.zeros(size)
    measured_speed_arr = np.zeros(size)
    measured_current_arr = np.zeros(size)
    main_speed_arr = np.full(size, np.nan)
    aux_speed_arr = np.full(size, np.nan)
    ekf_speed_arr = np.full(size, np.nan)
    feedback_arr = np.zeros(size)
    voltage_arr = np.zeros(size)
    sub_arr = np.zeros(size, dtype=bool)
    sensor_state_arr = np.zeros(size, dtype=bool)
    source_arr = np.zeros(size, dtype=int)

    solve_times: list[float] = []
    aux_times: list[float] = []
    reliability_times: list[float] = []
    arbitration_times: list[float] = []
    ekf_times: list[float] = []
    control_times: list[float] = []
    reliability_events: list[dict[str, Any]] = []
    optimizer_failures = 0
    main_prediction_failures = 0
    aux_prediction_failures = 0
    prediction_failure_messages: set[str] = set()
    nonfinite_events = 0
    ekf_failure_counts = {
        "ekf_numerical_failures": 0,
        "ekf_nonfinite_failures": 0,
        "ekf_psd_failures": 0,
        "ekf_symmetry_failures": 0,
    }
    run_failure_message = ""
    completed_samples = 0
    last_trusted_speed = 0.0
    last_main_prediction = 0.0
    common_current_inputs: list[float] = []
    aux_current_inputs: list[float] = []
    ekf_current_inputs: list[float] = []

    run_started = perf_counter()
    for index, time_s in enumerate(TIME):
        true_current, true_speed = float(state[0]), float(state[1])
        true_current_arr[index] = true_current
        true_speed_arr[index] = true_speed
        measured_current = _current_measurement(condition, float(time_s), true_current)
        measured_current_arr[index] = measured_current
        if condition.matrix == "current_boundary" and use_aux:
            common_current_inputs.append(float(measured_current))

        ekf_estimate = np.array([np.nan, np.nan, np.nan], dtype=float)
        if controller == E_CONTROLLER:
            assert observer is not None
            ekf_current_inputs.append(float(measured_current))
            ekf_started = perf_counter()
            try:
                if index == 0:
                    ekf_estimate = observer.initialize(measured_current)
                else:
                    ekf_estimate, _ = observer.step(prev_voltage, measured_current)
            except (EKFNumericalError, ValueError, np.linalg.LinAlgError) as exc:
                run_failure_message = f"{type(exc).__name__}: {exc}"
                ekf_failure_counts = _classify_ekf_failure(run_failure_message)
                break
            ekf_times.append(1000.0 * (perf_counter() - ekf_started))
            ekf_speed_arr[index] = float(ekf_estimate[1])

        measured_speed = _speed_measurement(
            condition,
            float(time_s),
            true_speed,
            float(base_noise[index]),
        )
        measured_speed_arr[index] = measured_speed

        if index >= 20:
            try:
                main_speed = float(
                    mpc.predict(history, np.array([prev_voltage], dtype=np.float32))[0]
                )
                last_main_prediction = main_speed
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                main_prediction_failures += 1
                prediction_failure_messages.add(f"main:{type(exc).__name__}:{exc}")
                main_speed = last_main_prediction
        else:
            main_speed = measured_speed
            if np.isfinite(main_speed):
                last_main_prediction = main_speed
        main_speed_arr[index] = main_speed

        auxiliary_speed = np.nan
        aux_ms = 0.0
        if index >= 20 and use_aux:
            aux_started = perf_counter()
            try:
                auxiliary_speed = auxiliary_predict_online(
                    aux_model,
                    aux_history[:, 0],
                    aux_history[:, 1],
                    aux_cfg["normalization"],
                )
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                aux_prediction_failures += 1
                prediction_failure_messages.add(f"aux:{type(exc).__name__}:{exc}")
            aux_ms = 1000.0 * (perf_counter() - aux_started)
            aux_times.append(aux_ms)
        aux_speed_arr[index] = auxiliary_speed

        reliability_ms = 0.0
        arbitration_ms = 0.0
        if controller == "B_plain_MPC" or index < 20:
            is_substitute = False
            is_suspect = False
            feedback = measured_speed
            source_code = 0
        else:
            main_residual = measured_speed - main_speed
            aux_residual = (
                measured_speed - auxiliary_speed if np.isfinite(auxiliary_speed) else None
            )
            reliability_started = perf_counter()
            was_active = sensor.active
            decision = sensor.update(main_residual, aux_residual=aux_residual)
            reliability_ms = 1000.0 * (perf_counter() - reliability_started)
            reliability_times.append(reliability_ms)
            is_substitute = bool(decision["substitute"])
            is_suspect = bool(decision["sensor_suspect"])
            if sensor.active != was_active:
                reliability_events.append(
                    {
                        "matrix": condition.matrix,
                        "family": condition.family,
                        "condition_id": condition.condition_id,
                        "current_case": condition.current_case,
                        "training_seed": training_seed,
                        "simulation_seed": simulation_seed,
                        "controller": controller,
                        "step": index,
                        "time_s": float(time_s),
                        "event": "entry" if sensor.active else "recovery",
                        "raw_main_residual": float(main_residual),
                        "aux_residual": (
                            float(aux_residual) if aux_residual is not None else np.nan
                        ),
                        "cusum_score": float(decision["score"]),
                    }
                )
            if controller == "C3_arbitration_MPC":
                arbitration_started = perf_counter()
                arbitration = arbitrator.update(
                    t=float(time_s),
                    y_measured=measured_speed,
                    y_main=main_speed,
                    y_aux=auxiliary_speed,
                    sensor_trusted=(not is_substitute),
                    fallback_speed=last_trusted_speed,
                )
                arbitration_ms = 1000.0 * (perf_counter() - arbitration_started)
                arbitration_times.append(arbitration_ms)
                feedback = float(arbitration["feedback"])
                source_code = int(arbitration["source_code"])
            else:
                if not np.isfinite(ekf_estimate[1]):
                    nonfinite_events += 1
                feedback = float(ekf_estimate[1]) if is_substitute else measured_speed
                source_code = 4 if is_substitute else 0

        if not is_substitute and np.isfinite(measured_speed):
            last_trusted_speed = float(np.clip(measured_speed, *arbitrator.aux_speed_bounds))

        required = [measured_speed, feedback, main_speed]
        if controller == E_CONTROLLER:
            required.append(float(ekf_estimate[1]))
        nonfinite_events += int(not np.isfinite(required).all())
        feedback_arr[index] = feedback
        sub_arr[index] = is_substitute
        sensor_state_arr[index] = is_suspect
        source_arr[index] = source_code

        history = np.vstack((history[1:], [prev_voltage, feedback])).astype(np.float32)
        # Causal current-channel contract: prediction_i consumed history through
        # i-1 above.  The one measured_current value produced for sample i is
        # now appended for future auxiliary predictions.  For E the exact same
        # scalar was already supplied to the EKF update at sample i.
        if condition.matrix == "current_boundary" and use_aux:
            aux_current_inputs.append(float(measured_current))
        aux_history = np.vstack(
            (aux_history[1:], [prev_voltage, measured_current])
        ).astype(np.float32)

        voltage = prev_voltage
        if index % CONTROL_STRIDE == 0:
            control_started = perf_counter()
            fallback_voltage = pi.compute_control(ref[index], feedback, CONTROL_DT)
            solve_started = perf_counter()
            output = mpc.compute_control(
                history,
                np.full(mpc_config.horizon, 35.0, dtype=float),
                prev_voltage,
                horizon=mpc_config.horizon,
                move_blocks=mpc_config.move_blocks,
                fallback_voltage=fallback_voltage,
            )
            solve_times.append(1000.0 * (perf_counter() - solve_started))
            if not output["success"]:
                optimizer_failures += 1
            voltage = float(output["voltage"])
            control_times.append(1000.0 * (perf_counter() - control_started))

        voltage_arr[index] = voltage
        history[-1, 0] = voltage
        aux_history[-1, 0] = voltage
        prev_voltage = voltage
        completed_samples = index + 1

        if index < size - 1:
            state = frozen_v3.rk4_step(
                state,
                voltage,
                _load_value(condition, float(time_s)),
                frozen_v3.nominal_params,
            )

    sim_time_s = perf_counter() - run_started
    run_complete = completed_samples == size and not run_failure_message
    finite_sample_rate = float(completed_samples / size)
    valid = slice(0, completed_samples)
    true_speed_valid = true_speed_arr[valid]
    ref_valid = ref[valid]
    error = ref_valid - true_speed_valid
    overall_rmse = float(np.sqrt(np.mean(error**2))) if completed_samples else np.nan
    overall_mae = float(np.mean(np.abs(error))) if completed_samples else np.nan
    valid_event_mask = event_mask[:completed_samples]
    fault_window_rmse = (
        float(np.sqrt(np.mean(error[valid_event_mask] ** 2)))
        if completed_samples and np.any(valid_event_mask)
        else np.nan
    )

    active = sensor_state_arr[:completed_samples]
    sub = sub_arr[:completed_samples]
    source = source_arr[:completed_samples]
    time_valid = TIME[:completed_samples]
    event_valid = event_mask[:completed_samples]
    if completed_samples >= 2:
        entry_indices = np.flatnonzero(~active[:-1] & active[1:]) + 1
        recovery_indices = np.flatnonzero(active[:-1] & ~active[1:]) + 1
        switches = int(np.sum(source[:-1] != source[1:]))
    else:
        entry_indices = np.array([], dtype=int)
        recovery_indices = np.array([], dtype=int)
        switches = 0
    post_entry_mask = time_valid[entry_indices] >= condition.event_start_s - 1e-12
    if np.isfinite(condition.event_end_s) and not condition.event_end_inclusive:
        post_entry_mask &= time_valid[entry_indices] < condition.event_end_s - 1e-12 / 2.0
    event_entry_indices = entry_indices[post_entry_mask]
    first_entry_time = float(time_valid[event_entry_indices[0]]) if len(event_entry_indices) else np.nan
    detection_latency = (
        float(first_entry_time - condition.event_start_s) if len(event_entry_indices) else np.nan
    )
    sensor_fault_detected: bool | float
    if condition.speed_fault_present:
        sensor_fault_detected = bool(len(event_entry_indices))
    else:
        sensor_fault_detected = np.nan

    post_recovery = recovery_indices[
        time_valid[recovery_indices] >= condition.event_start_s - 1e-12
    ]
    first_recovery_time = float(time_valid[post_recovery[0]]) if len(post_recovery) else np.nan
    finite_fault = np.isfinite(condition.event_end_s) and not condition.event_end_inclusive
    recovery_latency = np.nan
    recovered_after_event: bool | float = False if finite_fault else np.nan
    if finite_fault and np.any(active & event_valid):
        recovered = np.flatnonzero(
            (time_valid >= condition.event_end_s - 1e-12) & ~active
        )
        recovery_latency = (
            float(time_valid[recovered[0]] - condition.event_end_s) if len(recovered) else np.nan
        )
        recovered_after_event = bool(len(recovered))

    if completed_samples:
        frac_phys = float(np.mean(source == 0))
        frac_main = float(np.mean(source == 1))
        frac_aux = float(np.mean(source == 2))
        frac_fb = float(np.mean(source == 3))
        frac_ekf = float(np.mean(source == 4))
    else:
        frac_phys = frac_main = frac_aux = frac_fb = frac_ekf = np.nan
    if np.any(event_valid):
        event_source = source[event_valid]
        event_frac_phys = float(np.mean(event_source == 0))
        event_frac_main = float(np.mean(event_source == 1))
        event_frac_aux = float(np.mean(event_source == 2))
        event_frac_fb = float(np.mean(event_source == 3))
        event_frac_ekf = float(np.mean(event_source == 4))
        event_sub_fraction = float(np.mean(sub[event_valid]))
    else:
        event_frac_phys = event_frac_main = event_frac_aux = event_frac_fb = np.nan
        event_frac_ekf = event_sub_fraction = np.nan

    sub_duration_s = float(np.sum(sub) * DT)
    sub_fraction = float(np.mean(sub)) if completed_samples else np.nan
    avg_episode_duration_s = float(DURATION / (switches + 1)) if completed_samples else np.nan
    post_active_duration = (
        _post_event_active_duration(active, entry_indices, condition.event_start_s)
        if completed_samples
        else np.nan
    )
    if condition.family == "load" and not np.isfinite(post_active_duration):
        post_active_duration = 0.0
    load_false_entry_latency = (
        float(first_entry_time - condition.event_start_s)
        if condition.family == "load" and len(event_entry_indices)
        else np.nan
    )
    load_false_entry_count = int(len(event_entry_indices)) if condition.family == "load" else 0

    control_voltages = voltage_arr[:completed_samples:CONTROL_STRIDE]
    control_moves = np.diff(np.r_[0.0, control_voltages])
    voltage_violations = int(
        np.sum((control_voltages < -1e-8) | (control_voltages > 12.0 + 1e-8))
    )
    slew_violations = int(np.sum(np.abs(control_moves) > 2.0 + 1e-8))

    main_all_rmse = _rmse(main_speed_arr[:completed_samples], true_speed_valid)
    main_event_rmse = _rmse(
        main_speed_arr[:completed_samples], true_speed_valid, valid_event_mask
    )
    aux_all_rmse = _rmse(aux_speed_arr[:completed_samples], true_speed_valid)
    aux_event_rmse = _rmse(
        aux_speed_arr[:completed_samples], true_speed_valid, valid_event_mask
    )
    ekf_all_rmse = _rmse(ekf_speed_arr[:completed_samples], true_speed_valid)
    ekf_event_rmse = _rmse(
        ekf_speed_arr[:completed_samples], true_speed_valid, valid_event_mask
    )
    selected_event_rmse = _rmse(
        feedback_arr[:completed_samples], true_speed_valid, valid_event_mask
    )
    sub_indices = np.flatnonzero(sub)
    aux_virtual_rmse = (
        _rmse(aux_speed_arr[:completed_samples][sub_indices], true_speed_valid[sub_indices])
        if len(sub_indices)
        else np.nan
    )

    current_error = measured_current_arr[:completed_samples] - true_current_arr[:completed_samples]
    current_event_error = current_error[valid_event_mask]
    current_corruption_mean = (
        float(np.mean(current_event_error))
        if condition.current_fault_present and len(current_event_error)
        else 0.0
    )
    current_corruption_rmse = (
        float(np.sqrt(np.mean(current_event_error**2)))
        if condition.current_fault_present and len(current_event_error)
        else 0.0
    )
    corrupted_current_sha256 = ""
    aux_current_input_sha256 = ""
    ekf_current_input_sha256 = ""
    current_equivalence_mismatch_count = 0
    if condition.matrix == "current_boundary" and use_aux:
        common_completed = common_current_inputs[:completed_samples]
        aux_completed = aux_current_inputs[:completed_samples]
        corrupted_current_sha256 = _float64_sequence_sha256(common_completed)
        aux_current_input_sha256 = _float64_sequence_sha256(aux_completed)
        current_equivalence_mismatch_count += _sequence_mismatch_count(
            common_completed, aux_completed
        )
        if controller == E_CONTROLLER:
            ekf_completed = ekf_current_inputs[:completed_samples]
            ekf_current_input_sha256 = _float64_sequence_sha256(ekf_completed)
            current_equivalence_mismatch_count += _sequence_mismatch_count(
                common_completed, ekf_completed
            )

    total_failure_count = (
        optimizer_failures
        + main_prediction_failures
        + aux_prediction_failures
        + ekf_failure_counts["ekf_numerical_failures"]
    )
    incomplete_run_failures = int(not run_complete)
    total_safety_violation_count = (
        voltage_violations + slew_violations + nonfinite_events + incomplete_run_failures
    )
    result = {
        **_condition_metadata(condition),
        "training_seed": training_seed,
        "simulation_seed": simulation_seed,
        "seed": simulation_seed,
        "controller": controller,
        "science_executed": True,
        "reuse_source_key": "",
        "reuse_provenance": "",
        "run_complete": run_complete,
        "completed_samples": completed_samples,
        "finite_sample_rate": finite_sample_rate,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "fault_window_rmse": fault_window_rmse,
        "sensor_fault_detected": sensor_fault_detected,
        "detection_latency_s": detection_latency,
        "first_post_event_reliability_entry_time_s": first_entry_time,
        "post_event_reliability_entries": int(len(event_entry_indices)),
        "reliability_entries": int(len(entry_indices)),
        "recovery_events": int(len(recovery_indices)),
        "false_recovery_events": int(np.sum(event_valid[recovery_indices])) if len(recovery_indices) else 0,
        "first_post_event_recovery_time_s": first_recovery_time,
        "recovery_latency_s": recovery_latency,
        "recovered_after_event": recovered_after_event,
        "post_event_reliability_active_duration_s": post_active_duration,
        "load_false_entry_latency_s": load_false_entry_latency,
        "load_false_entry_count": load_false_entry_count,
        "sub_duration_s": sub_duration_s,
        "sub_fraction": sub_fraction,
        "event_sub_fraction": event_sub_fraction,
        "ekf_event_sub_fraction": event_sub_fraction if controller == E_CONTROLLER else np.nan,
        "switches": switches,
        "source_switch_count": switches,
        "avg_episode_duration_s": avg_episode_duration_s,
        "frac_phys": frac_phys,
        "frac_main": frac_main,
        "frac_aux": frac_aux,
        "frac_fb": frac_fb,
        "frac_ekf": frac_ekf,
        "event_frac_phys": event_frac_phys,
        "event_frac_main": event_frac_main,
        "event_frac_aux": event_frac_aux,
        "event_frac_fb": event_frac_fb,
        "event_frac_ekf": event_frac_ekf,
        "physical_fraction": frac_phys,
        "main_fraction": frac_main,
        "auxiliary_fraction": frac_aux,
        "fallback_fraction": frac_fb,
        "main_speed_rmse_all_samples": main_all_rmse,
        "main_speed_rmse_event_window": main_event_rmse,
        "aux_speed_rmse_all_samples": aux_all_rmse,
        "aux_speed_rmse_event_window": aux_event_rmse,
        "aux_virtual_rmse": aux_virtual_rmse,
        "ekf_speed_rmse_all_samples": ekf_all_rmse,
        "ekf_speed_rmse_event_window": ekf_event_rmse,
        "selected_feedback_rmse_event_window": selected_event_rmse,
        "current_corruption_mean_A": current_corruption_mean,
        "current_corruption_rmse_A": current_corruption_rmse,
        "corrupted_current_sha256": corrupted_current_sha256,
        "aux_current_input_sha256": aux_current_input_sha256,
        "ekf_current_input_sha256": ekf_current_input_sha256,
        "control_effort_u2": float(np.sum(control_voltages**2)),
        "control_variation_du2": float(np.sum(control_moves**2)),
        "voltage_violations": voltage_violations,
        "slew_violations": slew_violations,
        "rate_violations": slew_violations,
        "optimizer_failures": optimizer_failures,
        "main_prediction_failures": main_prediction_failures,
        "aux_prediction_failures": aux_prediction_failures,
        **ekf_failure_counts,
        "nonfinite_events": nonfinite_events,
        "incomplete_run_failures": incomplete_run_failures,
        "total_failure_count": total_failure_count,
        "total_safety_violation_count": total_safety_violation_count,
        "prediction_failure_messages": " | ".join(sorted(prediction_failure_messages)),
        "run_failure_message": run_failure_message,
        "current_equivalence_mismatch_count": current_equivalence_mismatch_count,
        "mean_solve_ms": float(np.mean(solve_times)) if solve_times else 0.0,
        "mean_aux_inference_ms": float(np.mean(aux_times)) if aux_times else 0.0,
        "mean_reliability_update_ms": (
            float(np.mean(reliability_times)) if reliability_times else 0.0
        ),
        "mean_arbitration_ms": (
            float(np.mean(arbitration_times)) if arbitration_times else 0.0
        ),
        "mean_ekf_update_ms": float(np.mean(ekf_times)) if ekf_times else 0.0,
        "mean_control_compute_ms": float(np.mean(control_times)) if control_times else 0.0,
        "sim_time_s": sim_time_s,
        **_pair_metadata(bundle),
        "reliability_events": reliability_events,
    }
    return result


def _expected_primary_keys(primary_conditions: tuple[Condition, ...]) -> set[tuple[Any, ...]]:
    return {
        (condition.family, condition.condition_id, training_seed, simulation_seed, controller)
        for condition in primary_conditions
        for training_seed in TRAINING_SEEDS
        for simulation_seed in SIMULATION_SEEDS
        for controller in PRIMARY_CONTROLLERS
    }


def _expected_current_keys(current_conditions: tuple[Condition, ...]) -> set[tuple[Any, ...]]:
    return {
        (condition.current_case, training_seed, simulation_seed, controller)
        for condition in current_conditions
        for training_seed in TRAINING_SEEDS
        for simulation_seed in SIMULATION_SEEDS
        for controller in CURRENT_CONTROLLERS
    }


def validate_result_matrix(
    runs: pd.DataFrame,
    primary_conditions: tuple[Condition, ...],
    current_conditions: tuple[Condition, ...],
) -> None:
    if len(runs) != EXPECTED_TOTAL_RUNS:
        raise RuntimeError(f"expected exactly {EXPECTED_TOTAL_RUNS} rows, found {len(runs)}")
    primary = runs[runs["matrix"].eq("primary")]
    current = runs[runs["matrix"].eq("current_boundary")]
    if len(primary) != EXPECTED_PRIMARY_RUNS or len(current) != EXPECTED_CURRENT_RUNS:
        raise RuntimeError(
            f"result matrix count mismatch: primary={len(primary)}, current={len(current)}"
        )
    actual_primary = {
        (
            str(row.family),
            str(row.condition_id),
            int(row.training_seed),
            int(row.simulation_seed),
            str(row.controller),
        )
        for row in primary.itertuples(index=False)
    }
    expected_primary = _expected_primary_keys(primary_conditions)
    if actual_primary != expected_primary:
        missing = sorted(expected_primary - actual_primary, key=str)[:10]
        extra = sorted(actual_primary - expected_primary, key=str)[:10]
        raise RuntimeError(f"primary result matrix mismatch; missing={missing}, extra={extra}")
    actual_current = {
        (
            str(row.current_case),
            int(row.training_seed),
            int(row.simulation_seed),
            str(row.controller),
        )
        for row in current.itertuples(index=False)
    }
    expected_current = _expected_current_keys(current_conditions)
    if actual_current != expected_current:
        missing = sorted(expected_current - actual_current, key=str)[:10]
        extra = sorted(actual_current - expected_current, key=str)[:10]
        raise RuntimeError(
            f"current-boundary result matrix mismatch; missing={missing}, extra={extra}"
        )
    if primary.duplicated(
        ["family", "condition_id", "training_seed", "simulation_seed", "controller"]
    ).any():
        raise RuntimeError("duplicate primary result cells detected")
    if current.duplicated(
        ["current_case", "training_seed", "simulation_seed", "controller"]
    ).any():
        raise RuntimeError("duplicate current-boundary result cells detected")
    if not np.array_equal(
        runs["seed"].to_numpy(dtype=int), runs["simulation_seed"].to_numpy(dtype=int)
    ):
        raise RuntimeError("legacy seed alias differs from simulation_seed")
    for training_seed in TRAINING_SEEDS:
        subset = runs[runs["training_seed"].eq(training_seed)]
        for column in (
            "main_model_sha256",
            "main_config_sha256",
            "aux_model_sha256",
            "aux_config_sha256",
            "sensor_calibration_sha256",
            "v3_calibration_sha256",
            "pair_config_sha256",
        ):
            if subset[column].nunique(dropna=False) != 1:
                raise RuntimeError(f"training seed {training_seed} mixes {column}")
    current_e = current[current["controller"].eq(E_CONTROLLER)]
    if not (current_e["current_equivalence_mismatch_count"].astype(int) == 0).all():
        raise RuntimeError("E current-boundary rows violate the shared-current-sample contract")
    current_c3 = current[current["controller"].eq("C3_arbitration_MPC")]
    if not (current_c3["current_equivalence_mismatch_count"].astype(int) == 0).all():
        raise RuntimeError("C3 current-boundary rows violate the shared-current-sample contract")
    if not (
        current_c3["corrupted_current_sha256"].astype(str)
        == current_c3["aux_current_input_sha256"].astype(str)
    ).all():
        raise RuntimeError("C3 common/aux current-input hashes differ")
    if not (
        (current_e["corrupted_current_sha256"].astype(str) == current_e["aux_current_input_sha256"].astype(str))
        & (current_e["corrupted_current_sha256"].astype(str) == current_e["ekf_current_input_sha256"].astype(str))
    ).all():
        raise RuntimeError("E common/aux/EKF current-input hashes differ")


def _row_source_key(row: pd.Series) -> str:
    if row["matrix"] == "primary":
        condition = f"{row['family']}:{row['condition_id']}"
    else:
        condition = f"current_sensor:{row['current_case']}"
    return (
        f"{condition}|training_seed={int(row['training_seed'])}|"
        f"simulation_seed={int(row['simulation_seed'])}|controller={row['controller']}"
    )


def _assert_reuse_identity(source: pd.Series, bundle: dict[str, Any]) -> None:
    expected = _pair_metadata(bundle)
    for column in (
        "main_model_sha256",
        "main_config_sha256",
        "aux_model_sha256",
        "aux_config_sha256",
        "sensor_calibration_sha256",
        "v3_calibration_sha256",
        "pair_config_sha256",
    ):
        if str(source[column]).lower() != str(expected[column]).lower():
            raise RuntimeError(f"reuse source has wrong training-seed binding for {column}")


def materialize_reused_rows(
    executed: pd.DataFrame,
    reused_tasks: list[tuple[Condition, int, int, str, dict[str, Any]]],
) -> pd.DataFrame:
    rows = [executed]
    for condition, training_seed, simulation_seed, controller, bundle in reused_tasks:
        if controller != "B_plain_MPC":
            raise RuntimeError("only B_plain_MPC cells are eligible for final-study reuse")
        if condition.current_case == "current_dropout_2s":
            source_mask = (
                executed["matrix"].eq("current_boundary")
                & executed["current_case"].eq("current_bias_5")
                & executed["training_seed"].eq(training_seed)
                & executed["simulation_seed"].eq(simulation_seed)
                & executed["controller"].eq(controller)
            )
            provenance = "B does not consume current; reused matched current_bias_5 B trajectory"
        elif condition.current_case == "speed_bias_5_plus_current_bias_5":
            source_mask = (
                executed["matrix"].eq("primary")
                & executed["family"].eq("bias")
                & np.isclose(executed["bias_percent"].astype(float), 5.0, rtol=0.0, atol=1e-12)
                & executed["training_seed"].eq(training_seed)
                & executed["simulation_seed"].eq(simulation_seed)
                & executed["controller"].eq(controller)
            )
            provenance = (
                "B does not consume current; reused matched primary 5% speed-bias B trajectory"
            )
        else:
            raise RuntimeError(f"unregistered reuse target {condition.current_case}")
        source_rows = executed.loc[source_mask]
        if len(source_rows) != 1:
            raise RuntimeError(
                f"expected exactly one reuse source for {condition.current_case}/"
                f"{training_seed}/{simulation_seed}, found {len(source_rows)}"
            )
        source = source_rows.iloc[0]
        _assert_reuse_identity(source, bundle)
        clone = source.copy()
        for key, value in _condition_metadata(condition).items():
            clone[key] = value
        clone["science_executed"] = False
        clone["reuse_source_key"] = _row_source_key(source)
        clone["reuse_provenance"] = provenance
        # These diagnostic summaries describe the current signal itself.  B
        # never consumes that channel, so the scientific row carries the
        # preregistered corruption declaration and leaves trajectory-derived
        # current-error summaries not applicable on reused B labels.
        clone["current_corruption_mean_A"] = np.nan
        clone["current_corruption_rmse_A"] = np.nan
        clone["corrupted_current_sha256"] = ""
        clone["aux_current_input_sha256"] = ""
        clone["ekf_current_input_sha256"] = ""
        clone["current_equivalence_mismatch_count"] = 0
        rows.append(pd.DataFrame([clone]))
    result = pd.concat(rows, ignore_index=True)
    if len(result) != EXPECTED_TOTAL_RUNS:
        raise RuntimeError(f"reuse materialization yielded {len(result)} rows, expected 735")
    if int(result["science_executed"].astype(bool).sum()) != 705:
        raise RuntimeError("reuse materialization no longer records exactly 705 executions")
    return result


def _attach_global_hashes(
    runs: pd.DataFrame,
    protocol_hashes: dict[str, str],
) -> pd.DataFrame:
    result = runs.copy()
    for column, digest in protocol_hashes.items():
        result[column] = digest
    result["ekf_config_sha256"] = sha256(EKF_CONFIG_PATH)
    result["frozen_v3_evaluator_sha256"] = sha256(PROJECT / "scripts" / "evaluate_v3_closed_loop.py")
    result["training_seed_evaluator_sha256"] = sha256(
        PROJECT / "scripts" / "evaluate_training_seed_robustness.py"
    )
    result["ekf_evaluator_sha256"] = sha256(PROJECT / "scripts" / "evaluate_ekf_closed_loop.py")
    return result


def _attach_b_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    result = runs.copy()
    result["controller_delta_vs_B_fault_window_rmse"] = np.nan
    result["controller_amplification_vs_B"] = np.nan
    primary_key = ["family", "condition_id", "training_seed", "simulation_seed"]
    current_key = ["current_case", "training_seed", "simulation_seed"]
    for matrix, key in (("primary", primary_key), ("current_boundary", current_key)):
        mask = result["matrix"].eq(matrix)
        block = result.loc[mask]
        baseline = block[block["controller"].eq("B_plain_MPC")].set_index(key)["fault_window_rmse"]
        for index, row in block.iterrows():
            lookup = tuple(row[column] for column in key)
            if len(key) == 1:
                lookup = lookup[0]
            b_value = float(baseline.loc[lookup])
            value = float(row["fault_window_rmse"])
            result.at[index, "controller_delta_vs_B_fault_window_rmse"] = value - b_value
            result.at[index, "controller_amplification_vs_B"] = (
                value / b_value - 1.0 if np.isfinite(b_value) and abs(b_value) > 1e-12 else np.nan
            )
    return result


def _output_exists() -> list[Path]:
    return [
        path
        for path in (RUNS_PATH, EVENTS_PATH, EVALUATION_MANIFEST_PATH)
        if path.exists()
    ]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate every frozen study/model/calibration/EKF lock without running science",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(int(os.environ.get("FINAL_ROBUSTNESS_WORKERS", "8")), os.cpu_count() or 1),
        help="number of process workers for the fixed 735-cell scientific matrix",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    protocol, bundles, preflight = run_preflight()
    if args.preflight_only:
        print(json.dumps(_json_ready(preflight), indent=2, allow_nan=False))
        return

    existing = _output_exists()
    if existing:
        raise RuntimeError(
            "final robustness scientific outputs already exist; refusing to overwrite: "
            + ", ".join(rel(path) for path in existing)
        )

    primary_conditions, current_conditions = build_conditions(protocol)
    primary_tasks, current_tasks = build_tasks(primary_conditions, current_conditions, bundles)
    tasks, reused_tasks = build_execution_tasks(primary_tasks, current_tasks)
    if len(tasks) != 705 or len(reused_tasks) != 30:
        raise RuntimeError("internal execution/reuse matrix changed")

    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(simulate_final_run, tasks, chunksize=1))
    wall_time_s = perf_counter() - started

    reliability_event_rows: list[dict[str, Any]] = []
    for result in results:
        reliability_event_rows.extend(result.pop("reliability_events", []))
    runs = materialize_reused_rows(pd.DataFrame(results), reused_tasks)
    validate_result_matrix(runs, primary_conditions, current_conditions)
    runs = _attach_b_comparisons(runs)
    runs = _attach_global_hashes(runs, preflight["protocol_hashes"])
    missing_columns = [column for column in RUN_COLUMNS if column not in runs.columns]
    extra_columns = [column for column in runs.columns if column not in RUN_COLUMNS]
    if missing_columns or extra_columns:
        raise RuntimeError(
            f"run schema mismatch before write; missing={missing_columns}, extra={extra_columns}"
        )
    runs = runs.loc[:, RUN_COLUMNS].sort_values(
        ["matrix", "family", "condition_id", "training_seed", "simulation_seed", "controller"]
    ).reset_index(drop=True)
    validate_result_matrix(runs, primary_conditions, current_conditions)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_PATH, index=False)
    written = pd.read_csv(RUNS_PATH, float_precision="round_trip")
    validate_result_matrix(written, primary_conditions, current_conditions)
    if tuple(written.columns) != RUN_COLUMNS:
        raise RuntimeError("written final_robustness_runs.csv schema changed")

    if reliability_event_rows:
        pd.DataFrame(reliability_event_rows).sort_values(
            ["matrix", "family", "condition_id", "training_seed", "simulation_seed", "controller", "step"]
        ).to_csv(EVENTS_PATH, index=False)

    evaluation_manifest = {
        "study": STUDY,
        "role": "part_b_scientific_evaluation",
        "run_count": len(runs),
        "unique_simulation_executions": int(runs["science_executed"].astype(bool).sum()),
        "reuse_materialized_rows": int((~runs["science_executed"].astype(bool)).sum()),
        "primary_run_count": int(runs["matrix"].eq("primary").sum()),
        "current_boundary_run_count": int(runs["matrix"].eq("current_boundary").sum()),
        "training_seeds": list(TRAINING_SEEDS),
        "simulation_seeds": list(SIMULATION_SEEDS),
        "primary_controllers": list(PRIMARY_CONTROLLERS),
        "current_controllers": list(CURRENT_CONTROLLERS),
        "matrix_verified_exact_600_plus_135": True,
        "duplicate_cells": 0,
        "preflight": preflight,
        "output": {
            "runs": rel(RUNS_PATH),
            "runs_sha256": sha256(RUNS_PATH),
            "schema": list(RUN_COLUMNS),
            "reliability_events": rel(EVENTS_PATH) if EVENTS_PATH.exists() else None,
            "reliability_events_sha256": sha256(EVENTS_PATH) if EVENTS_PATH.exists() else None,
        },
        "evaluation_wall_time_s": wall_time_s,
    }
    EVALUATION_MANIFEST_PATH.write_text(
        json.dumps(_json_ready(evaluation_manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {len(runs)} final robustness rows to {rel(RUNS_PATH)}; "
        f"wall time {wall_time_s:.1f}s."
    )


if __name__ == "__main__":
    main()
