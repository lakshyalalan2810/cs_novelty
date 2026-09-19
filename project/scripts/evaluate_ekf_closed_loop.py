"""Closed-loop comparator for the preregistered current-only EKF.

This module adds one separate controller, ``E_EKF_virtual_MPC``.  It reuses the
frozen V3 LSTM-MPC stack, scenario definitions, plant integration, and
``SensorReliabilityMonitor``.  The EKF runs continuously from sample zero using
only applied voltage and armature current.  It does not participate in fault
detection: the unchanged V3 monitor still consumes the raw physical speed
residual against the main LSTM plus the frozen auxiliary-recovery residual.
Only after that monitor requests substitution does EKF omega become feedback.

The executable refuses to start any closed-loop simulation unless
``results/configs/ekf_frozen_config.json`` exists, contains a complete frozen
EKF covariance selection, and records the preregistered 27-candidate
calibration evidence.  The development seed family is declared here before
the first run by design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import evaluate_v3_closed_loop as v3  # noqa: E402
from auxiliary_sensor_model import (  # noqa: E402
    AuxiliarySpeedEstimator,
    auxiliary_predict_online,
)
from ekf_observer import (  # noqa: E402
    AugmentedStateEKF,
    EKFConfig,
    EKFNumericalError,
)
from motor_model import DCMotorParams  # noqa: E402
from mpc import LSTMMPC, MPCConfig, PIConfig, PIController  # noqa: E402
from reliability import SensorReliabilityMonitor, load_lstm_model  # noqa: E402


CONTROLLER = "E_EKF_virtual_MPC"
BASELINE_CONTROLLERS = (
    "B_plain_MPC",
    "C1_sensor_MPC",
    "C3_arbitration_MPC",
)
REUSABLE_BASELINE_CONTROLLERS = ("B_plain_MPC", "C3_arbitration_MPC")

# This declaration is intentionally module-level and precedes every executable
# simulation path.  The reserved C4 candidate family is explicitly excluded.
DEVELOPMENT_SEEDS = (19026, 19027, 19028, 19029, 19030)
FORBIDDEN_SEEDS = frozenset({39026, 39027, 39028, 39029, 39030})
PRIMARY_SCENARIOS = (
    "sensor_bias_5",
    "sensor_dropout",
    "load_disturbance",
    "combined_fault_load",
    "parameter_variation",
)

EKF_CONFIG_PATH = PROJECT / "results" / "configs" / "ekf_frozen_config.json"
PREREG_PATH = PROJECT / "EKF_OBSERVER_PREREGISTRATION.md"
FROZEN_EVIDENCE_PATH = (
    PROJECT / "results" / "configs" / "ekf_frozen_evidence_hashes.json"
)
CLOSED_LOOP_PLAN_PATH = PROJECT / "results" / "configs" / "ekf_closed_loop_plan.json"
C4_CONFIG_PATH = PROJECT / "results" / "configs" / "c4_development_config.json"
C4_SUMMARY_PATH = PROJECT / "results" / "metrics" / "c4_development_summary.json"
C4_RUNS_PATH = PROJECT / "results" / "metrics" / "c4_development_runs.csv"
C4_EVALUATOR_PATH = PROJECT / "scripts" / "evaluate_c4_closed_loop.py"

RUNS_PATH = PROJECT / "results" / "metrics" / "ekf_closed_loop_runs.csv"
SUMMARY_PATH = PROJECT / "results" / "metrics" / "ekf_closed_loop_summary.json"
TRACE_PATH = PROJECT / "results" / "metrics" / "ekf_closed_loop_traces.csv"

DT = v3.DT
TIME = v3.TIME
CONTROL_STRIDE = v3.CONTROL_STRIDE
CONTROL_DT = v3.CONTROL_DT
FULL_SCALE = v3.FULL_SCALE
SPEED_NOISE_STD = v3.SPEED_NOISE_STD


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _matrix(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (3,):
        array = np.diag(array)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise RuntimeError(f"{label} must be a finite length-3 diagonal or 3x3 matrix")
    return array


def _first_value(blocks: list[dict[str, Any]], names: tuple[str, ...]) -> Any:
    for block in blocks:
        for name in names:
            if name in block:
                return block[name]
    return None


def _candidate_table(config: dict[str, Any]) -> list[dict[str, Any]] | None:
    containers = [config]
    for name in ("calibration", "validation", "selection"):
        value = config.get(name)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for name in (
            "candidate_table",
            "candidates",
            "validation_candidates",
            "grid_results",
            "calibration_candidates",
        ):
            value = container.get(name)
            if isinstance(value, list) and all(isinstance(row, dict) for row in value):
                return value
    return None


def validate_calibration_lock(
    config: dict[str, Any],
    config_path: Path = EKF_CONFIG_PATH,
) -> dict[str, Any]:
    """Verify the preregistered 27-candidate calibration before fault evaluation."""

    expected_grid = {
        (q_dyn, m_t, r)
        for q_dyn in (1e-8, 1e-6, 1e-4)
        for m_t in (1e-2, 1e-1, 1.0)
        for r in (1e-8, 1e-6, 1e-4)
    }
    embedded = _candidate_table(config)
    if embedded is not None:
        if len(embedded) != 27:
            raise RuntimeError("frozen EKF config must preserve exactly 27 calibration candidates")
        return {
            "verified": True,
            "storage": "embedded",
            "candidate_count": 27,
        }

    provenance = config.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("frozen EKF config is missing calibration provenance")
    candidate_relpath = provenance.get("candidate_table_path")
    expected_hash = provenance.get("candidate_table_sha256")
    if not isinstance(candidate_relpath, str) or not isinstance(expected_hash, str):
        raise RuntimeError("frozen EKF config must bind the external 27-candidate table by path and SHA256")
    candidate_path = (PROJECT / candidate_relpath).resolve()
    try:
        candidate_path.relative_to(PROJECT.resolve())
    except ValueError as exc:
        raise RuntimeError("candidate table path escapes the project root") from exc
    if not candidate_path.exists():
        raise RuntimeError(f"frozen candidate table is missing: {candidate_relpath}")
    actual_hash = sha256(candidate_path)
    if actual_hash != expected_hash:
        raise RuntimeError("frozen candidate table SHA256 does not match calibration provenance")

    table = pd.read_csv(candidate_path, float_precision="round_trip")
    required = {
        "candidate_id",
        "q_dyn",
        "m_T",
        "r",
        "q_current",
        "q_speed",
        "q_load",
        "R",
        "stable",
        "selected",
    }
    if len(table) != 27 or not required <= set(table.columns):
        raise RuntimeError("external EKF calibration table is not the complete 27-candidate audit")
    actual_grid = {
        (float(row.q_dyn), float(row.m_T), float(row.r))
        for row in table.itertuples(index=False)
    }
    if actual_grid != expected_grid:
        raise RuntimeError("external EKF calibration table differs from the preregistered 3x3x3 grid")
    if table["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("external EKF calibration table contains duplicate candidate ids")

    selected_mask = table["selected"].astype(str).str.lower().isin({"true", "1"})
    if int(selected_mask.sum()) != 1:
        raise RuntimeError("external EKF calibration table must mark exactly one selected candidate")
    selected_row = table.loc[selected_mask].iloc[0]
    selected_block = config.get("selected_candidate")
    if not isinstance(selected_block, dict):
        raise RuntimeError("frozen EKF config is missing selected_candidate")
    if str(selected_block.get("candidate_id")) != str(selected_row["candidate_id"]):
        raise RuntimeError("selected candidate id does not match the frozen candidate table")
    if str(selected_row["stable"]).lower() not in {"true", "1"}:
        raise RuntimeError("selected EKF candidate is not marked numerically stable")

    q = _matrix(selected_block.get("Q"), label="selected Q")
    expected_q = np.diag(
        [
            float(selected_row["q_current"]),
            float(selected_row["q_speed"]),
            float(selected_row["q_load"]),
        ]
    )
    if not np.allclose(q, expected_q, rtol=1e-12, atol=0.0):
        raise RuntimeError("selected Q does not match the selected calibration-table row")
    if not np.isclose(float(selected_block.get("R")), float(selected_row["R"]), rtol=1e-12, atol=0.0):
        raise RuntimeError("selected R does not match the selected calibration-table row")

    return {
        "verified": True,
        "storage": str(candidate_path.relative_to(PROJECT)).replace("\\", "/"),
        "candidate_count": 27,
        "candidate_table_sha256": actual_hash,
        "selected_candidate_id": str(selected_row["candidate_id"]),
    }


def _selected_blocks(config: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for name in ("selected_candidate", "selected", "ekf", "frozen_ekf"):
        value = config.get(name)
        if isinstance(value, dict):
            blocks.append(value)
            covariance = value.get("covariances")
            if isinstance(covariance, dict):
                blocks.append(covariance)
    initialization = config.get("initialization")
    if isinstance(initialization, dict):
        blocks.append(initialization)
    blocks.append(config)
    return blocks


def ekf_config_from_frozen(config: dict[str, Any]) -> EKFConfig:
    """Build the fixed nominal EKF from a frozen calibration record.

    The parser accepts matrix or diagonal-vector forms (``Q``/``Q_diag`` and
    ``P0``/``P0_diag``) so the calibration artifact can remain human-readable.
    It deliberately ignores scenario identity and always installs nominal motor
    parameters, including during the plant's parameter-variation scenario.
    """

    blocks = _selected_blocks(config)
    q_value = _first_value(blocks, ("Q", "Q_diag", "q_matrix", "q_diag"))
    r_value = _first_value(blocks, ("R", "r_i", "measurement_variance"))
    p0_value = _first_value(blocks, ("P0", "P0_diag", "p0", "p0_diag"))
    if q_value is None or r_value is None or p0_value is None:
        raise RuntimeError("frozen EKF config must contain selected Q, R, and P0")

    dt_value = _first_value(blocks, ("dt", "dt_s", "timestep", "timestep_s"))
    dt = DT if dt_value is None else float(dt_value)
    if not np.isclose(dt, DT, rtol=0.0, atol=1e-15):
        raise RuntimeError(f"frozen EKF timestep must be {DT} s, got {dt}")

    return EKFConfig(
        Q=_matrix(q_value, label="Q"),
        R=float(r_value),
        P0=_matrix(p0_value, label="P0"),
        dt=DT,
        params=DCMotorParams(),
    )


def load_frozen_ekf_config(path: Path = EKF_CONFIG_PATH) -> tuple[dict[str, Any], EKFConfig]:
    """Load and validate the pre-fault frozen EKF configuration.

    This is the first executable guard in :func:`main`; model loading and every
    simulation path occur only after it succeeds.
    """

    if not path.exists():
        raise RuntimeError(
            f"{path.name} is absent; closed-loop EKF evaluation is forbidden until calibration is frozen"
        )
    config = _json(path)
    status = config.get("status")
    if status is not None and "frozen" not in str(status).lower():
        raise RuntimeError(f"EKF config status is not frozen: {status!r}")
    validate_calibration_lock(config, path)

    if config.get("state_order") not in (None, ["current", "omega", "load_torque"]):
        raise RuntimeError("frozen EKF state order differs from [current, omega, load_torque]")
    if config.get("online_inputs") not in (None, ["voltage", "current"]):
        raise RuntimeError("frozen EKF online inputs must be voltage/current only")
    if config.get("current_measurement_noise_injected") not in (None, False):
        raise RuntimeError("primary closed-loop comparison forbids injected current measurement noise")

    nominal = config.get("nominal_motor_params")
    if nominal is not None and nominal != asdict(DCMotorParams()):
        raise RuntimeError("frozen EKF motor parameters differ from repository nominal parameters")

    selected = config.get("selected_candidate")
    if isinstance(selected, dict):
        for metric in ("numerical_failure_count", "nonfinite_estimate_count", "failure_run_count"):
            if metric in selected and int(selected[metric]) != 0:
                raise RuntimeError(f"selected EKF candidate is not stable: {metric}={selected[metric]}")

    training_std = config.get("training_only_state_std")
    if isinstance(training_std, dict):
        scales = np.array(
            [
                training_std.get("current_A"),
                training_std.get("speed_rad_s"),
                training_std.get("load_torque_Nm"),
            ],
            dtype=float,
        )
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise RuntimeError("frozen training-only state scales are invalid")
        expected_p0 = np.diag(np.square(scales))
        actual_p0 = _matrix(_first_value(_selected_blocks(config), ("P0", "P0_diag", "p0", "p0_diag")), label="P0")
        if not np.allclose(actual_p0, expected_p0, rtol=1e-12, atol=1e-15):
            raise RuntimeError("frozen P0 is not the training-only state-variance initialization")

    declared = config.get("closed_loop_development_seeds")
    if declared is not None and tuple(int(seed) for seed in declared) != DEVELOPMENT_SEEDS:
        raise RuntimeError("frozen EKF config disagrees with the predeclared development seed family")
    if FORBIDDEN_SEEDS.intersection(DEVELOPMENT_SEEDS):
        raise RuntimeError("development seed declaration overlaps the reserved C4 candidate family")
    return config, ekf_config_from_frozen(config)


def verify_frozen_runtime_sources() -> dict[str, str]:
    """Bind the future run to the exact V3/model artifacts locked pre-EKF."""

    if not FROZEN_EVIDENCE_PATH.exists():
        raise RuntimeError("pre-EKF frozen evidence hash record is missing")
    evidence = _json(FROZEN_EVIDENCE_PATH)
    scientific = evidence.get("frozen_scientific_artifacts", {})
    supporting = evidence.get("supporting_provenance", {})
    keys = {
        "main_lstm_weights": scientific,
        "auxiliary_lstm_weights": scientific,
        "v3_arbitration_calibration": scientific,
        "v3_arbitration_config": scientific,
        "motor_model": supporting,
        "mpc": supporting,
        "reliability": supporting,
        "v3_evaluator": supporting,
    }
    verified: dict[str, str] = {}
    for name, mapping in keys.items():
        record = mapping.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError(f"frozen evidence record is missing {name}")
        path = PROJECT / record["path"]
        expected = record.get("sha256")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen source changed: {record['path']}")
        verified[name] = actual
    return verified


def verify_closed_loop_plan() -> dict[str, str]:
    """Verify the immutable pre-run plan and bind it to this evaluator source."""

    if not CLOSED_LOOP_PLAN_PATH.exists():
        raise RuntimeError("immutable EKF closed-loop pre-run plan is missing")
    plan = _json(CLOSED_LOOP_PLAN_PATH)
    if plan.get("status") != "frozen_before_first_closed_loop_run":
        raise RuntimeError("EKF closed-loop plan is not frozen before the first run")
    if tuple(plan.get("controllers", [])) != (*BASELINE_CONTROLLERS, CONTROLLER):
        raise RuntimeError("EKF closed-loop plan controller design mismatch")
    if tuple(plan.get("primary_scenarios", [])) != PRIMARY_SCENARIOS:
        raise RuntimeError("EKF closed-loop plan primary scenarios mismatch")
    if tuple(int(seed) for seed in plan.get("development_seeds", [])) != DEVELOPMENT_SEEDS:
        raise RuntimeError("EKF closed-loop plan development seeds mismatch")
    if set(int(seed) for seed in plan.get("reserved_unused_seeds", [])) != FORBIDDEN_SEEDS:
        raise RuntimeError("EKF closed-loop plan reserved seed family mismatch")

    bindings = {
        "ekf_config_sha256": sha256(EKF_CONFIG_PATH),
        "ekf_preregistration_sha256": sha256(PREREG_PATH),
        "frozen_evidence_manifest_sha256": sha256(FROZEN_EVIDENCE_PATH),
        "closed_loop_evaluator_sha256_before_first_run": sha256(Path(__file__).resolve()),
    }
    for key, expected in bindings.items():
        if plan.get(key) != expected:
            raise RuntimeError(f"EKF closed-loop plan binding mismatch: {key}")

    evidence = _json(FROZEN_EVIDENCE_PATH).get("frozen_scientific_artifacts", {})
    matrix_bindings = {
        "frozen_v3_matrix_sha256": evidence.get("v3_275_run_matrix", {}).get("sha256"),
        "frozen_c4_development_matrix_sha256": evidence.get("c4_development_matrix", {}).get(
            "sha256"
        ),
    }
    for key, expected in matrix_bindings.items():
        if not isinstance(expected, str) or plan.get(key) != expected:
            raise RuntimeError(f"EKF closed-loop plan frozen matrix binding mismatch: {key}")

    runtime = plan.get("runtime_accounting")
    if not isinstance(runtime, dict):
        raise RuntimeError("EKF closed-loop plan is missing runtime accounting rules")
    if runtime.get("per_sample_total_overhead") != "ekf_update_ms + aux_inference_ms + reliability_update_ms":
        raise RuntimeError("EKF closed-loop plan total overhead definition mismatch")
    if runtime.get("optimizer_timing") != "slsqp_solve_ms recorded separately":
        raise RuntimeError("EKF closed-loop plan SLSQP timing definition mismatch")
    if runtime.get("distribution_statistics") != ["mean", "median", "p95", "p99", "max"]:
        raise RuntimeError("EKF closed-loop plan timing distribution mismatch")

    return {
        "closed_loop_plan_sha256": sha256(CLOSED_LOOP_PLAN_PATH),
        "closed_loop_evaluator_sha256": bindings[
            "closed_loop_evaluator_sha256_before_first_run"
        ],
    }


def inspect_reusable_c4_baselines() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Return frozen B/C3 development rows only after exact provenance checks.

    A failed check returns ``None`` instead of partially trusting the matrix;
    the future runner then regenerates the affected baselines through the
    existing frozen :func:`evaluate_v3_closed_loop.simulate_v3_run` path.
    """

    report: dict[str, Any] = {"verified": False, "reasons": []}
    required = [FROZEN_EVIDENCE_PATH, C4_CONFIG_PATH, C4_SUMMARY_PATH, C4_RUNS_PATH, C4_EVALUATOR_PATH]
    missing = [str(path.relative_to(PROJECT)) for path in required if not path.exists()]
    if missing:
        report["reasons"].append(f"missing artifacts: {missing}")
        return None, report

    try:
        evidence = _json(FROZEN_EVIDENCE_PATH)
        c4_config = _json(C4_CONFIG_PATH)
        c4_summary = _json(C4_SUMMARY_PATH)
        runs = pd.read_csv(C4_RUNS_PATH, float_precision="round_trip")

        scientific = evidence["frozen_scientific_artifacts"]
        expected_matrix_hash = scientific["c4_development_matrix"]["sha256"]
        actual_matrix_hash = sha256(C4_RUNS_PATH)
        if actual_matrix_hash != expected_matrix_hash:
            raise RuntimeError("C4 development matrix hash differs from pre-EKF lock")

        current_c4_evaluator_hash = sha256(C4_EVALUATOR_PATH)
        if c4_summary.get("hashes", {}).get("c4_evaluator_script") != current_c4_evaluator_hash:
            raise RuntimeError("C4 summary is not bound to the current frozen C4 evaluator")
        if c4_config.get("hashes", {}).get("c4_evaluator_script") != current_c4_evaluator_hash:
            raise RuntimeError("C4 config is not bound to the current frozen C4 evaluator")

        runtime_hashes = verify_frozen_runtime_sources()
        summary_hashes = c4_summary.get("hashes", {})
        config_hashes = c4_config.get("hashes", {})
        bindings = {
            "main_model": runtime_hashes["main_lstm_weights"],
            "auxiliary_model": runtime_hashes["auxiliary_lstm_weights"],
            "v3_arbitration_config": runtime_hashes["v3_arbitration_config"],
            "v3_arbitration_calibration": runtime_hashes["v3_arbitration_calibration"],
        }
        for name, expected in bindings.items():
            if summary_hashes.get(name) != expected:
                raise RuntimeError(f"C4 summary {name} binding mismatch")
        config_aliases = {
            "v3_main_model": runtime_hashes["main_lstm_weights"],
            "v3_auxiliary_model": runtime_hashes["auxiliary_lstm_weights"],
            "v3_arbitration_config": runtime_hashes["v3_arbitration_config"],
            "v3_arbitration_calibration": runtime_hashes["v3_arbitration_calibration"],
        }
        for name, expected in config_aliases.items():
            if config_hashes.get(name) != expected:
                raise RuntimeError(f"C4 config {name} binding mismatch")

        design = c4_config.get("development_design", {})
        if tuple(design.get("seeds", [])) != DEVELOPMENT_SEEDS:
            raise RuntimeError("C4 development seeds differ from the declared EKF pairing seeds")
        if tuple(c4_summary.get("seeds", [])) != DEVELOPMENT_SEEDS:
            raise RuntimeError("C4 summary seeds differ from the declared EKF pairing seeds")
        if c4_summary.get("mode") != "development":
            raise RuntimeError("C4 matrix is not labeled development evidence")

        key_columns = ["controller", "scenario", "seed"]
        if runs.duplicated(key_columns).any():
            raise RuntimeError("C4 development matrix contains duplicate controller/scenario/seed keys")
        available_scenarios = set(PRIMARY_SCENARIOS).intersection(set(design.get("scenarios", [])))
        subset = runs[
            runs["controller"].isin(REUSABLE_BASELINE_CONTROLLERS)
            & runs["scenario"].isin(available_scenarios)
            & runs["seed"].isin(DEVELOPMENT_SEEDS)
        ].copy()
        expected_keys = {
            (controller, scenario, seed)
            for controller in REUSABLE_BASELINE_CONTROLLERS
            for scenario in available_scenarios
            for seed in DEVELOPMENT_SEEDS
        }
        actual_keys = set(
            zip(subset["controller"], subset["scenario"], subset["seed"].astype(int))
        )
        if actual_keys != expected_keys:
            raise RuntimeError("C4 baseline subset is missing or contains unexpected paired keys")

        subset["baseline_provenance"] = "verified_frozen_c4_development_runs"
        subset["baseline_matrix_sha256"] = actual_matrix_hash
        report.update(
            {
                "verified": True,
                "reasons": [],
                "matrix_sha256": actual_matrix_hash,
                "available_scenarios": sorted(available_scenarios),
                "row_count": int(len(subset)),
                "c4_evaluator_sha256": current_c4_evaluator_hash,
                "v3_evaluator_sha256": runtime_hashes["v3_evaluator"],
            }
        )
        return subset, report
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        report["reasons"].append(str(exc))
        return None, report


def collect_baseline_rows(
    rerun_fn: Callable[[tuple[str, str, int, bool]], dict[str, Any]] = v3.simulate_v3_run,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse exact frozen rows where proven; rerun all remaining paired keys."""

    frozen, provenance = inspect_reusable_c4_baselines()
    rows: list[dict[str, Any]] = []
    frozen_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    if frozen is not None:
        frozen_lookup = {
            (str(row["controller"]), str(row["scenario"]), int(row["seed"])): row.to_dict()
            for _, row in frozen.iterrows()
        }

    for scenario in PRIMARY_SCENARIOS:
        for seed in DEVELOPMENT_SEEDS:
            for controller in BASELINE_CONTROLLERS:
                key = (controller, scenario, seed)
                if key in frozen_lookup:
                    rows.append(frozen_lookup[key])
                    continue
                result = dict(rerun_fn((controller, scenario, seed, False)))
                result["baseline_provenance"] = "rerun_v3_simulate_v3_run"
                result["baseline_matrix_sha256"] = np.nan
                rows.append(result)
    return pd.DataFrame(rows), provenance


def make_v3_monitor(sensor_cfg: dict[str, Any], v3_cfg: dict[str, Any]) -> SensorReliabilityMonitor:
    """Construct the exact C3 speed monitor, including its auxiliary recovery gate."""

    return SensorReliabilityMonitor(
        sensor_cfg["instant_threshold"],
        sensor_cfg["center"],
        sensor_cfg["allowance"],
        sensor_cfg["threshold"],
        sensor_cfg["enter_count"],
        sensor_cfg["exit_count"],
        aux_recovery_gate=v3_cfg["arbitrator"]["aux_recovery_gate"],
    )


def update_v3_monitor(
    monitor: SensorReliabilityMonitor,
    measured_speed: float,
    main_speed: float,
    auxiliary_speed: float,
) -> dict[str, float | int | bool]:
    """Update detection from V3 signals only; EKF has no input to this seam."""

    r_main = measured_speed - main_speed
    r_aux = measured_speed - auxiliary_speed if np.isfinite(auxiliary_speed) else None
    return monitor.update(r_main, aux_residual=r_aux)


def speed_measurement(
    scenario_name: str,
    time_s: float,
    true_speed: float,
    base_noise: float,
    rng: np.random.Generator,
) -> float:
    """Apply the frozen V3 physical-speed corruption semantics exactly."""

    measured = float(true_speed + base_noise)
    if scenario_name == "sensor_bias_5" and 2.0 <= time_s < 4.0:
        measured += 0.05 * FULL_SCALE
    elif scenario_name == "sensor_bias_15" and 2.0 <= time_s < 4.0:
        measured += 0.15 * FULL_SCALE
    elif scenario_name == "sensor_dropout" and 2.0 <= time_s < 4.0:
        measured = 0.0
    elif scenario_name == "sensor_drift" and time_s >= 2.0:
        measured += 0.15 * FULL_SCALE * min(1.0, (time_s - 2.0) / 4.0)
    elif scenario_name == "combined_fault_load" and time_s >= 3.0:
        measured += 0.05 * FULL_SCALE
    elif scenario_name == "sensor_noise" and 2.0 <= time_s < 4.0:
        measured += float(rng.normal(0.0, 2.0))
    return measured


def advance_ekf(
    observer: AugmentedStateEKF,
    sample_index: int,
    applied_voltage: float,
    current_measurement: float,
) -> tuple[np.ndarray, Any | None]:
    """Advance the EKF at every sample, independently of monitor state.

    Sample zero performs the frozen causal initialization.  Every later sample
    executes one predict/update using the voltage applied during the preceding
    10 ms plant interval and the current available at the present sample.
    """

    if sample_index == 0:
        return observer.initialize(current_measurement), None
    return observer.step(applied_voltage, current_measurement)


def _event_mask(scenario_name: str, time: np.ndarray = TIME) -> np.ndarray:
    definition = v3.SCENARIOS[scenario_name]
    start = definition["event_start"]
    end = definition["event_end"]
    if start is None:
        return np.ones(len(time), dtype=bool)
    mask = time >= float(start)
    if end is not None:
        mask &= time < float(end)
    return mask


def _percentile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability)) if values else 0.0


def _timing_distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "sample_count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "sample_count": int(len(array)),
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "max_ms": float(np.max(array)),
    }


def _runtime_columns(
    total_overhead_ms: list[float], slsqp_solve_ms: list[float]
) -> dict[str, float | int]:
    overhead = _timing_distribution(total_overhead_ms)
    slsqp = _timing_distribution(slsqp_solve_ms)
    return {
        "total_reliability_observer_overhead_sample_count": overhead["sample_count"],
        "mean_total_reliability_observer_overhead_ms": overhead["mean_ms"],
        "median_total_reliability_observer_overhead_ms": overhead["median_ms"],
        "p95_total_reliability_observer_overhead_ms": overhead["p95_ms"],
        "p99_total_reliability_observer_overhead_ms": overhead["p99_ms"],
        "max_total_reliability_observer_overhead_ms": overhead["max_ms"],
        "slsqp_solve_count": slsqp["sample_count"],
        "mean_slsqp_solve_ms": slsqp["mean_ms"],
        "median_slsqp_solve_ms": slsqp["median_ms"],
        "p95_slsqp_solve_ms": slsqp["p95_ms"],
        "p99_slsqp_solve_ms": slsqp["p99_ms"],
        "max_slsqp_solve_ms": slsqp["max_ms"],
    }


def _finite_quality(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    if len(estimate) == 0:
        return np.nan, np.nan, np.nan
    error = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return (
        float(np.sqrt(np.mean(error**2))),
        float(np.mean(np.abs(error))),
        float(np.mean(error)),
    )


def _classify_ekf_failure(message: str) -> dict[str, int]:
    lower = message.lower()
    return {
        "ekf_numerical_failures": 1,
        "ekf_nonfinite_failures": int("nonfinite" in lower or "finite" in lower and "must be finite" in lower),
        "ekf_psd_failures": int("positive semidefinite" in lower or "eigenvalue" in lower),
        "ekf_symmetry_failures": int("asymmetric" in lower or "symmetr" in lower),
    }


def simulate_ekf_run(
    args: tuple[str, int, dict[str, Any]],
) -> dict[str, Any]:
    """Run one preregistered E_EKF_virtual_MPC closed-loop trajectory."""

    scenario_name, seed, frozen_config = args
    if scenario_name not in PRIMARY_SCENARIOS:
        raise ValueError(f"scenario {scenario_name!r} is not a preregistered primary EKF scenario")
    if seed not in DEVELOPMENT_SEEDS or seed in FORBIDDEN_SEEDS:
        raise ValueError(f"seed {seed} is not in the declared EKF development family")

    torch.set_num_threads(1)
    device = "cpu"
    main_model, main_cfg = load_lstm_model(
        PROJECT / "results/lstm_model_weights.pt",
        PROJECT / "results/configs/lstm_model_config.json",
        device=device,
    )
    aux_cfg = _json(PROJECT / "results/configs/v2_auxiliary_config.json")
    v3_cfg = _json(PROJECT / "results/configs/v3_arbitration_config.json")
    rel_cfg = _json(PROJECT / "results/configs/reliability_final_config.json")
    aux_model = AuxiliarySpeedEstimator(**aux_cfg["model"]).to(device)
    aux_model.load_state_dict(
        torch.load(
            PROJECT / "results/auxiliary_model_weights.pt",
            map_location=device,
            weights_only=True,
        )
    )
    aux_model.eval()

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
    pi_config = PIConfig(0.35, 0.8, (0.0, 12.0), 2.0)
    sensor = make_v3_monitor(rel_cfg["sensor"], v3_cfg)
    observer = AugmentedStateEKF(ekf_config_from_frozen(frozen_config))
    pi = PIController(pi_config)
    mpc = LSTMMPC(main_model, main_cfg["normalization"], 20, mpc_config)

    rng = np.random.default_rng(seed)
    base_noise = rng.normal(0.0, SPEED_NOISE_STD, len(TIME))
    state = np.zeros(2, dtype=float)
    prev_voltage = 0.0
    history = np.zeros((20, 2), dtype=np.float32)
    aux_history = np.zeros((20, 2), dtype=np.float32)
    ref = v3.reference_values(scenario_name)

    size = len(TIME)
    true_speed_arr = np.zeros(size)
    current_arr = np.zeros(size)
    measured_arr = np.zeros(size)
    main_arr = np.full(size, np.nan)
    aux_arr = np.full(size, np.nan)
    ekf_speed_arr = np.full(size, np.nan)
    ekf_load_arr = np.full(size, np.nan)
    feedback_arr = np.zeros(size)
    voltage_arr = np.zeros(size)
    sub_arr = np.zeros(size, dtype=bool)
    suspect_arr = np.zeros(size, dtype=bool)
    innovation_arr = np.full(size, np.nan)
    nis_arr = np.full(size, np.nan)
    min_eig_arr = np.full(size, np.nan)
    symmetry_arr = np.full(size, np.nan)
    ekf_timing_arr = np.full(size, np.nan)
    aux_timing_arr = np.zeros(size)
    reliability_timing_arr = np.zeros(size)
    total_overhead_timing_arr = np.full(size, np.nan)
    slsqp_timing_arr = np.full(size, np.nan)

    ekf_times: list[float] = []
    main_times: list[float] = []
    aux_times: list[float] = []
    reliability_times: list[float] = []
    total_overhead_times: list[float] = []
    control_times: list[float] = []
    slsqp_times: list[float] = []
    optimizer_failures = 0
    main_prediction_failures = 0
    aux_prediction_failures = 0
    prediction_failure_messages: set[str] = set()
    nonfinite_events = 0
    last_main_prediction = 0.0
    completed_samples = 0
    failure_message = ""
    failure_counts = {
        "ekf_numerical_failures": 0,
        "ekf_nonfinite_failures": 0,
        "ekf_psd_failures": 0,
        "ekf_symmetry_failures": 0,
    }

    started_run = perf_counter()
    for index, time_s in enumerate(TIME):
        current_val, true_speed = float(state[0]), float(state[1])
        current_arr[index] = current_val
        true_speed_arr[index] = true_speed

        ekf_started = perf_counter()
        try:
            estimate, diagnostics = advance_ekf(
                observer,
                index,
                prev_voltage,
                current_val,
            )
        except (EKFNumericalError, ValueError, np.linalg.LinAlgError) as exc:
            failure_message = f"{type(exc).__name__}: {exc}"
            failure_counts = _classify_ekf_failure(failure_message)
            break
        ekf_ms = 1000.0 * (perf_counter() - ekf_started)
        ekf_times.append(ekf_ms)
        ekf_timing_arr[index] = ekf_ms
        ekf_speed_arr[index] = float(estimate[1])
        ekf_load_arr[index] = float(estimate[2])
        covariance = observer.covariance
        symmetry_arr[index] = float(np.max(np.abs(covariance - covariance.T)))
        min_eig_arr[index] = float(np.min(np.linalg.eigvalsh(covariance)))
        if diagnostics is not None:
            innovation_arr[index] = diagnostics.innovation
            nis_arr[index] = diagnostics.innovation**2 / diagnostics.innovation_variance

        measured = speed_measurement(
            scenario_name, float(time_s), true_speed, float(base_noise[index]), rng
        )
        measured_arr[index] = measured

        if index >= 20:
            main_started = perf_counter()
            try:
                main_speed = float(
                    mpc.predict(history, np.array([prev_voltage], dtype=np.float32))[0]
                )
                last_main_prediction = main_speed
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                main_prediction_failures += 1
                prediction_failure_messages.add(f"main:{type(exc).__name__}:{exc}")
                main_speed = last_main_prediction
            main_times.append(1000.0 * (perf_counter() - main_started))
        else:
            main_speed = measured
            if np.isfinite(main_speed):
                last_main_prediction = main_speed
        main_arr[index] = main_speed

        auxiliary_speed = np.nan
        aux_ms = 0.0
        if index >= 20:
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
        aux_timing_arr[index] = aux_ms
        aux_arr[index] = auxiliary_speed

        reliability_ms = 0.0
        if index < 20:
            is_substitute = False
            is_suspect = False
            feedback = measured
        else:
            reliability_started = perf_counter()
            decision = update_v3_monitor(sensor, measured, main_speed, auxiliary_speed)
            reliability_ms = 1000.0 * (perf_counter() - reliability_started)
            reliability_times.append(reliability_ms)
            is_substitute = bool(decision["substitute"])
            is_suspect = bool(decision["sensor_suspect"])
            feedback = float(estimate[1]) if is_substitute else measured
        reliability_timing_arr[index] = reliability_ms
        total_overhead_ms = ekf_ms + aux_ms + reliability_ms
        total_overhead_timing_arr[index] = total_overhead_ms
        total_overhead_times.append(total_overhead_ms)

        required = np.asarray([measured, feedback, main_speed, estimate[1]], dtype=float)
        nonfinite_events += int(not np.isfinite(required).all())
        feedback_arr[index] = feedback
        sub_arr[index] = is_substitute
        suspect_arr[index] = is_suspect

        history = np.vstack((history[1:], [prev_voltage, feedback])).astype(np.float32)
        aux_history = np.vstack((aux_history[1:], [prev_voltage, current_val])).astype(np.float32)

        voltage = prev_voltage
        if index % CONTROL_STRIDE == 0:
            control_started = perf_counter()
            fallback_voltage = pi.compute_control(ref[index], feedback, CONTROL_DT)
            solve_started = perf_counter()
            output = mpc.compute_control(
                history,
                v3.reference_values(
                    scenario_name,
                    time_s + DT * np.arange(1, mpc_config.horizon + 1),
                ),
                prev_voltage,
                horizon=mpc_config.horizon,
                move_blocks=mpc_config.move_blocks,
                fallback_voltage=fallback_voltage,
            )
            slsqp_ms = 1000.0 * (perf_counter() - solve_started)
            slsqp_times.append(slsqp_ms)
            slsqp_timing_arr[index] = slsqp_ms
            if not output["success"]:
                optimizer_failures += 1
            voltage = float(output["voltage"])
            control_times.append(1000.0 * (perf_counter() - control_started))

        voltage_arr[index] = voltage
        history[-1, 0] = voltage
        aux_history[-1, 0] = voltage
        prev_voltage = voltage
        completed_samples = index + 1

        load_value = (
            0.15
            if scenario_name in ("load_disturbance", "combined_fault_load") and time_s >= 3.0
            else 0.03
        )
        plant_params = (
            v3.shifted_params
            if scenario_name == "parameter_variation" and time_s >= 3.0
            else v3.nominal_params
        )
        if index < size - 1:
            state = v3.rk4_step(state, voltage, load_value, plant_params)

    total_sim_time = perf_counter() - started_run
    run_complete = completed_samples == size and not failure_message
    finite_sample_rate = float(completed_samples / size)

    if not run_complete:
        trace = {
            "time": TIME[:completed_samples].copy(),
            "current": current_arr[:completed_samples].copy(),
            "true": true_speed_arr[:completed_samples].copy(),
            "meas": measured_arr[:completed_samples].copy(),
            "main": main_arr[:completed_samples].copy(),
            "aux": aux_arr[:completed_samples].copy(),
            "ekf": ekf_speed_arr[:completed_samples].copy(),
            "ekf_load": ekf_load_arr[:completed_samples].copy(),
            "fb": feedback_arr[:completed_samples].copy(),
            "substitute": sub_arr[:completed_samples].copy(),
            "sensor_suspect": suspect_arr[:completed_samples].copy(),
            "ref": ref[:completed_samples].copy(),
            "voltage": voltage_arr[:completed_samples].copy(),
            "innovation": innovation_arr[:completed_samples].copy(),
            "nis": nis_arr[:completed_samples].copy(),
            "min_cov_eigenvalue": min_eig_arr[:completed_samples].copy(),
            "cov_asymmetry": symmetry_arr[:completed_samples].copy(),
            "ekf_update_ms": ekf_timing_arr[:completed_samples].copy(),
            "aux_inference_ms": aux_timing_arr[:completed_samples].copy(),
            "reliability_update_ms": reliability_timing_arr[:completed_samples].copy(),
            "total_reliability_observer_overhead_ms": total_overhead_timing_arr[
                :completed_samples
            ].copy(),
            "slsqp_solve_ms": slsqp_timing_arr[:completed_samples].copy(),
        }
        return {
            "controller": CONTROLLER,
            "scenario": scenario_name,
            "seed": seed,
            "run_complete": False,
            "completed_samples": completed_samples,
            "ekf_finite_sample_rate": finite_sample_rate,
            "ekf_failure_step": completed_samples,
            "ekf_failure_message": failure_message,
            **failure_counts,
            "overall_rmse": np.nan,
            "overall_mae": np.nan,
            "fault_window_rmse": np.nan,
            "optimizer_failures": optimizer_failures,
            "main_prediction_failures": main_prediction_failures,
            "aux_prediction_failures": aux_prediction_failures,
            "nonfinite_events": nonfinite_events,
            "voltage_violations": np.nan,
            "rate_violations": np.nan,
            "mean_ekf_update_ms": float(np.mean(ekf_times)) if ekf_times else 0.0,
            "median_ekf_update_ms": float(np.median(ekf_times)) if ekf_times else 0.0,
            "p95_ekf_update_ms": _percentile(ekf_times, 0.95),
            "p99_ekf_update_ms": _percentile(ekf_times, 0.99),
            "max_ekf_update_ms": float(np.max(ekf_times)) if ekf_times else 0.0,
            **_runtime_columns(total_overhead_times, slsqp_times),
            "sim_time_s": total_sim_time,
            "trace": trace,
        }

    error = ref - true_speed_arr
    event_mask = _event_mask(scenario_name)
    fault_window_rmse = float(np.sqrt(np.mean(error[event_mask] ** 2)))
    ctrl_voltages = voltage_arr[::CONTROL_STRIDE]
    ctrl_moves = np.diff(np.r_[0.0, ctrl_voltages])
    voltage_violations = int(np.sum((ctrl_voltages < -1e-8) | (ctrl_voltages > 12.0 + 1e-8)))
    rate_violations = int(np.sum(np.abs(ctrl_moves) > 2.0 + 1e-8))
    reliability_entries = int(np.sum(~suspect_arr[:-1] & suspect_arr[1:]))
    recovery_events = int(np.sum(suspect_arr[:-1] & ~suspect_arr[1:]))
    substitution_indices = np.flatnonzero(sub_arr)
    ekf_sub_rmse, ekf_sub_mae, ekf_sub_bias = _finite_quality(
        ekf_speed_arr[substitution_indices], true_speed_arr[substitution_indices]
    )
    corrupted_rmse, corrupted_mae, corrupted_bias = _finite_quality(
        measured_arr[substitution_indices], true_speed_arr[substitution_indices]
    )
    ekf_all_rmse, ekf_all_mae, ekf_all_bias = _finite_quality(ekf_speed_arr, true_speed_arr)
    ekf_event_rmse, ekf_event_mae, ekf_event_bias = _finite_quality(
        ekf_speed_arr[event_mask], true_speed_arr[event_mask]
    )

    scenario_type = v3.SCENARIOS[scenario_name]["type"]
    actual_fault_mask = event_mask if scenario_type in {"sensor", "combined"} else np.zeros(size, dtype=bool)
    false_sub_mask = sub_arr & ~actual_fault_mask
    event_substitution_fraction = float(np.mean(sub_arr[event_mask]))
    detected_indices = np.flatnonzero(sub_arr & actual_fault_mask)
    detection_latency = (
        float(TIME[detected_indices[0]] - v3.SCENARIOS[scenario_name]["event_start"])
        if len(detected_indices)
        else np.nan
    )

    finite_innovation = innovation_arr[np.isfinite(innovation_arr)]
    finite_nis = nis_arr[np.isfinite(nis_arr)]
    result = {
        "controller": CONTROLLER,
        "scenario": scenario_name,
        "seed": seed,
        "run_complete": True,
        "completed_samples": size,
        "overall_rmse": float(np.sqrt(np.mean(error**2))),
        "overall_mae": float(np.mean(np.abs(error))),
        "fault_window_rmse": fault_window_rmse,
        "reliability_entries": reliability_entries,
        "recovery_events": recovery_events,
        "sub_duration_s": float(np.sum(sub_arr) * DT),
        "sub_fraction": float(np.mean(sub_arr)),
        "sub_samples": int(np.sum(sub_arr)),
        "event_sub_fraction": event_substitution_fraction,
        "false_substitution_count": int(np.sum(false_sub_mask)),
        "false_substitution_fraction": float(np.mean(false_sub_mask)),
        "detection_latency_s": detection_latency,
        "sensor_fault_detected": bool(len(detected_indices)) if scenario_type in {"sensor", "combined"} else np.nan,
        "ekf_substitution_rmse": ekf_sub_rmse,
        "ekf_substitution_mae": ekf_sub_mae,
        "ekf_substitution_bias": ekf_sub_bias,
        "corrupted_sensor_substitution_rmse": corrupted_rmse,
        "corrupted_sensor_substitution_mae": corrupted_mae,
        "corrupted_sensor_substitution_bias": corrupted_bias,
        "ekf_speed_rmse_all_samples": ekf_all_rmse,
        "ekf_speed_mae_all_samples": ekf_all_mae,
        "ekf_speed_bias_all_samples": ekf_all_bias,
        "ekf_speed_rmse_event_window": ekf_event_rmse,
        "ekf_speed_mae_event_window": ekf_event_mae,
        "ekf_speed_bias_event_window": ekf_event_bias,
        "parameter_transfer_estimator_speed_rmse": (
            ekf_event_rmse if scenario_name == "parameter_variation" else np.nan
        ),
        "control_effort_u2": float(np.sum(ctrl_voltages**2)),
        "control_variation_du2": float(np.sum(ctrl_moves**2)),
        "voltage_violations": voltage_violations,
        "rate_violations": rate_violations,
        "optimizer_failures": optimizer_failures,
        "main_prediction_failures": main_prediction_failures,
        "aux_prediction_failures": aux_prediction_failures,
        "prediction_failure_messages": " | ".join(sorted(prediction_failure_messages)),
        "nonfinite_events": nonfinite_events,
        **failure_counts,
        "ekf_finite_sample_rate": finite_sample_rate,
        "ekf_failure_step": np.nan,
        "ekf_failure_message": "",
        "ekf_min_cov_eigenvalue": float(np.nanmin(min_eig_arr)),
        "ekf_max_cov_asymmetry": float(np.nanmax(symmetry_arr)),
        "innovation_mean": float(np.mean(finite_innovation)) if len(finite_innovation) else np.nan,
        "innovation_mae": float(np.mean(np.abs(finite_innovation))) if len(finite_innovation) else np.nan,
        "nis_mean": float(np.mean(finite_nis)) if len(finite_nis) else np.nan,
        "nis_p95": float(np.quantile(finite_nis, 0.95)) if len(finite_nis) else np.nan,
        "mean_solve_ms": float(np.mean(slsqp_times)) if slsqp_times else 0.0,
        "mean_main_inference_ms": float(np.mean(main_times)) if main_times else 0.0,
        "mean_aux_inference_ms": float(np.mean(aux_times)) if aux_times else 0.0,
        "mean_reliability_update_ms": float(np.mean(reliability_times)) if reliability_times else 0.0,
        "mean_control_compute_ms": float(np.mean(control_times)) if control_times else 0.0,
        "mean_ekf_update_ms": float(np.mean(ekf_times)),
        "median_ekf_update_ms": float(np.median(ekf_times)),
        "p95_ekf_update_ms": _percentile(ekf_times, 0.95),
        "p99_ekf_update_ms": _percentile(ekf_times, 0.99),
        "max_ekf_update_ms": float(np.max(ekf_times)),
        **_runtime_columns(total_overhead_times, slsqp_times),
        "sim_time_s": total_sim_time,
        "observer_parameter_mode": "fixed_nominal",
        "trace": {
            "time": TIME.copy(),
            "current": current_arr,
            "true": true_speed_arr,
            "meas": measured_arr,
            "main": main_arr,
            "aux": aux_arr,
            "ekf": ekf_speed_arr,
            "ekf_load": ekf_load_arr,
            "fb": feedback_arr,
            "substitute": sub_arr,
            "sensor_suspect": suspect_arr,
            "ref": ref,
            "voltage": voltage_arr,
            "innovation": innovation_arr,
            "nis": nis_arr,
            "min_cov_eigenvalue": min_eig_arr,
            "cov_asymmetry": symmetry_arr,
            "ekf_update_ms": ekf_timing_arr,
            "aux_inference_ms": aux_timing_arr,
            "reliability_update_ms": reliability_timing_arr,
            "total_reliability_observer_overhead_ms": total_overhead_timing_arr,
            "slsqp_solve_ms": slsqp_timing_arr,
        },
    }
    return result


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


def _paired_frame(runs: pd.DataFrame, scenario: str, metric: str) -> pd.DataFrame:
    subset = runs[runs["scenario"] == scenario]
    return subset.pivot(index="seed", columns="controller", values=metric).sort_index()


def expected_matrix_keys() -> set[tuple[str, str, int]]:
    """Return the exact preregistered 4-controller paired development design."""

    return {
        (controller, scenario, seed)
        for controller in (*BASELINE_CONTROLLERS, CONTROLLER)
        for scenario in PRIMARY_SCENARIOS
        for seed in DEVELOPMENT_SEEDS
    }


def runtime_overhead_summary(runs: pd.DataFrame) -> dict[str, Any]:
    """Report total observer/reliability overhead separately from SLSQP timing."""

    summary: dict[str, Any] = {}
    for scenario in PRIMARY_SCENARIOS:
        subset = runs[runs["scenario"] == scenario]
        ekf_rows = subset[subset["controller"] == CONTROLLER]
        c3_rows = subset[subset["controller"] == "C3_arbitration_MPC"]
        ekf_sim = float(ekf_rows["sim_time_s"].mean())
        c3_sim = float(c3_rows["sim_time_s"].mean())
        delta = ekf_sim - c3_sim
        summary[scenario] = {
            "mean_ekf_sim_time_s": ekf_sim,
            "mean_c3_sim_time_s": c3_sim,
            "mean_overhead_s": delta,
            "mean_overhead_fraction_vs_c3": delta / c3_sim if c3_sim > 0 else np.nan,
            "mean_ekf_update_ms": float(ekf_rows["mean_ekf_update_ms"].mean()),
            "p95_ekf_update_ms_across_seed_summaries": float(
                ekf_rows["p95_ekf_update_ms"].mean()
            ),
            "total_reliability_observer_overhead_ms": {
                "mean_of_run_means": float(
                    ekf_rows["mean_total_reliability_observer_overhead_ms"].mean()
                ),
                "mean_of_run_medians": float(
                    ekf_rows["median_total_reliability_observer_overhead_ms"].mean()
                ),
                "mean_of_run_p95s": float(
                    ekf_rows["p95_total_reliability_observer_overhead_ms"].mean()
                ),
                "mean_of_run_p99s": float(
                    ekf_rows["p99_total_reliability_observer_overhead_ms"].mean()
                ),
                "max_across_runs": float(
                    ekf_rows["max_total_reliability_observer_overhead_ms"].max()
                ),
            },
            "slsqp_solve_ms": {
                "mean_of_run_means": float(ekf_rows["mean_slsqp_solve_ms"].mean()),
                "mean_of_run_medians": float(ekf_rows["median_slsqp_solve_ms"].mean()),
                "mean_of_run_p95s": float(ekf_rows["p95_slsqp_solve_ms"].mean()),
                "mean_of_run_p99s": float(ekf_rows["p99_slsqp_solve_ms"].mean()),
                "max_across_runs": float(ekf_rows["max_slsqp_solve_ms"].max()),
            },
        }
    return summary


def evaluate_role_gate(runs: pd.DataFrame, calibration_locked: bool) -> dict[str, Any]:
    """Evaluate the preregistered successor gate without changing its criteria."""

    ekf = runs[runs["controller"] == CONTROLLER]
    complete = bool(
        len(ekf) == len(PRIMARY_SCENARIOS) * len(DEVELOPMENT_SEEDS)
        and ekf["run_complete"].astype("boolean").fillna(False).all()
        and np.allclose(ekf["ekf_finite_sample_rate"].astype(float), 1.0, rtol=0.0, atol=0.0)
        and int(ekf["ekf_numerical_failures"].fillna(0).sum()) == 0
    )
    safety_columns = [
        "voltage_violations",
        "rate_violations",
        "optimizer_failures",
        "nonfinite_events",
        "main_prediction_failures",
        "aux_prediction_failures",
    ]
    safety_failures = int(sum(float(ekf[column].fillna(0).sum()) for column in safety_columns))
    safety_pass = safety_failures == 0

    isolated: dict[str, Any] = {}
    isolated_pass = True
    for scenario in ("sensor_bias_5", "sensor_dropout"):
        paired = _paired_frame(runs, scenario, "fault_window_rmse")
        ratio = paired[CONTROLLER] / paired["C3_arbitration_MPC"] - 1.0
        mean_ratio = float(
            runs[(runs.scenario == scenario) & (runs.controller == CONTROLLER)]["fault_window_rmse"].mean()
            / runs[(runs.scenario == scenario) & (runs.controller == "C3_arbitration_MPC")]["fault_window_rmse"].mean()
            - 1.0
        )
        seed_passes = int(np.sum(ratio <= 0.05 + 1e-12))
        passed = mean_ratio <= 0.05 + 1e-12 and seed_passes >= 4
        isolated[scenario] = {
            "mean_relative_degradation": mean_ratio,
            "paired_seed_passes": seed_passes,
            "pass": bool(passed),
        }
        isolated_pass &= bool(passed)

    combined = _paired_frame(runs, "combined_fault_load", "fault_window_rmse")
    combined_improvement = (
        combined["C3_arbitration_MPC"] - combined[CONTROLLER]
    ) / combined["C3_arbitration_MPC"]
    combined_mean_c3 = float(
        runs[(runs.scenario == "combined_fault_load") & (runs.controller == "C3_arbitration_MPC")]["fault_window_rmse"].mean()
    )
    combined_mean_ekf = float(
        runs[(runs.scenario == "combined_fault_load") & (runs.controller == CONTROLLER)]["fault_window_rmse"].mean()
    )
    combined_mean_improvement = (combined_mean_c3 - combined_mean_ekf) / combined_mean_c3
    combined_seed_passes = int(np.sum(combined_improvement >= 0.10 - 1e-12))
    combined_pass = combined_mean_improvement >= 0.10 - 1e-12 and combined_seed_passes >= 4

    load_rmse = _paired_frame(runs, "load_disturbance", "fault_window_rmse")
    plain_penalty = load_rmse["B_plain_MPC"]
    c3_penalty = load_rmse["C3_arbitration_MPC"] - plain_penalty
    ekf_penalty = load_rmse[CONTROLLER] - plain_penalty
    paired_penalty_reduction = np.where(
        c3_penalty > 0,
        (c3_penalty - ekf_penalty) / c3_penalty,
        np.where(ekf_penalty <= c3_penalty, 1.0, -np.inf),
    )
    load_seed_passes = int(np.sum(paired_penalty_reduction >= 0.10 - 1e-12))
    load_rows = runs[runs.scenario == "load_disturbance"]
    b_mean = float(load_rows[load_rows.controller == "B_plain_MPC"]["fault_window_rmse"].mean())
    c3_mean = float(load_rows[load_rows.controller == "C3_arbitration_MPC"]["fault_window_rmse"].mean())
    ekf_mean = float(load_rows[load_rows.controller == CONTROLLER]["fault_window_rmse"].mean())
    c3_mean_penalty = c3_mean - b_mean
    ekf_mean_penalty = ekf_mean - b_mean
    load_mean_reduction = (
        (c3_mean_penalty - ekf_mean_penalty) / c3_mean_penalty
        if c3_mean_penalty > 0
        else (1.0 if ekf_mean_penalty <= c3_mean_penalty else -np.inf)
    )

    c3_load = load_rows[load_rows.controller == "C3_arbitration_MPC"].set_index("seed")
    ekf_load = load_rows[load_rows.controller == CONTROLLER].set_index("seed")
    # In a plant-only load scenario every substitution is a false substitution.
    c3_count = c3_load["sub_samples"].astype(float)
    ekf_count = ekf_load["false_substitution_count"].astype(float)
    c3_fraction = c3_load["sub_fraction"].astype(float)
    ekf_fraction = ekf_load["false_substitution_fraction"].astype(float)
    false_nonworse_seed_passes = int(
        np.sum((ekf_count <= c3_count + 1e-12) & (ekf_fraction <= c3_fraction + 1e-12))
    )
    false_nonworse = bool(
        ekf_count.mean() <= c3_count.mean() + 1e-12
        and ekf_fraction.mean() <= c3_fraction.mean() + 1e-12
        and false_nonworse_seed_passes >= 4
    )
    load_pass = bool(
        load_mean_reduction >= 0.10 - 1e-12
        and load_seed_passes >= 4
        and false_nonworse
    )

    all_pass = bool(
        complete
        and safety_pass
        and isolated_pass
        and combined_pass
        and load_pass
        and calibration_locked
    )
    return {
        "decision": "POTENTIAL_SUCCESSOR_FOLLOW_UP" if all_pass else "BASELINE_ONLY",
        "finite_every_sample": complete,
        "zero_new_safety_failures": safety_pass,
        "safety_failure_count": safety_failures,
        "isolated_sensor_faults": isolated,
        "combined_fault_load": {
            "mean_relative_improvement": float(combined_mean_improvement),
            "paired_seed_passes": combined_seed_passes,
            "pass": bool(combined_pass),
        },
        "load_disturbance": {
            "mean_tracking_penalty_reduction": float(load_mean_reduction),
            "paired_tracking_seed_passes": load_seed_passes,
            "false_substitution_nonworse_seed_passes": false_nonworse_seed_passes,
            "false_substitution_nonworse": false_nonworse,
            "pass": load_pass,
        },
        "qr_selected_from_preregistered_27_candidate_calibration": bool(calibration_locked),
        "all_successor_criteria_pass": all_pass,
    }


def _trace_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    trace = result.pop("trace", None)
    if trace is None:
        return []
    rows: list[dict[str, Any]] = []
    length = len(trace["time"])
    for index in range(length):
        rows.append(
            {
                "controller": result["controller"],
                "scenario": result["scenario"],
                "seed": result["seed"],
                "step": index,
                **{name: values[index] for name, values in trace.items()},
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate frozen EKF/source/baseline provenance without executing a simulation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Hard stop must remain before model loading, baseline regeneration, or any
    # new closed-loop seed use.
    frozen_record, _ = load_frozen_ekf_config(EKF_CONFIG_PATH)
    plan_lock = verify_closed_loop_plan()
    runtime_hashes = verify_frozen_runtime_sources()
    baseline_rows, baseline_provenance = inspect_reusable_c4_baselines()
    if args.preflight_only:
        print(
            json.dumps(
                _json_ready(
                    {
                        "ekf_config_sha256": sha256(EKF_CONFIG_PATH),
                        **plan_lock,
                        "development_seeds": DEVELOPMENT_SEEDS,
                        "primary_scenarios": PRIMARY_SCENARIOS,
                        "runtime_hashes": runtime_hashes,
                        "reusable_baselines": baseline_provenance,
                        "reusable_baseline_rows": 0 if baseline_rows is None else len(baseline_rows),
                    }
                ),
                indent=2,
            )
        )
        return

    started = perf_counter()
    ekf_results: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for scenario in PRIMARY_SCENARIOS:
        for seed in DEVELOPMENT_SEEDS:
            result = simulate_ekf_run((scenario, seed, frozen_record))
            trace_rows.extend(_trace_rows(result))
            ekf_results.append(result)

    baselines, baseline_provenance = collect_baseline_rows()
    ekf_frame = pd.DataFrame(ekf_results)
    runs = pd.concat([baselines, ekf_frame], ignore_index=True, sort=False)
    runs = runs.sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)

    expected_keys = expected_matrix_keys()
    actual_keys = set(zip(runs.controller, runs.scenario, runs.seed.astype(int)))
    if len(runs) != 100 or runs.duplicated(["controller", "scenario", "seed"]).any() or actual_keys != expected_keys:
        raise RuntimeError("EKF closed-loop matrix is not the exact 4 x 5 x 5 paired design")

    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_PATH, index=False)
    pd.DataFrame(trace_rows).to_csv(TRACE_PATH, index=False)

    aggregate_metrics = [
        "overall_rmse",
        "fault_window_rmse",
        "sub_fraction",
        "sub_samples",
        "control_effort_u2",
        "control_variation_du2",
        "voltage_violations",
        "rate_violations",
        "optimizer_failures",
        "nonfinite_events",
        "sim_time_s",
        "mean_ekf_update_ms",
        "parameter_transfer_estimator_speed_rmse",
    ]
    available_metrics = [name for name in aggregate_metrics if name in runs.columns]
    grouped = runs.groupby(["scenario", "controller"])[available_metrics].agg(["mean", "std", "median"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    summary_rows = grouped.reset_index().to_dict(orient="records")
    calibration_lock = validate_calibration_lock(frozen_record, EKF_CONFIG_PATH)
    calibration_locked = bool(calibration_lock["verified"])
    summary = {
        "evidence_role": "development_closed_loop_comparator",
        "controller": CONTROLLER,
        "detector": "unchanged frozen V3 SensorReliabilityMonitor; EKF excluded from detector inputs",
        "substitution_source": "EKF omega only when frozen V3 monitor requests substitution",
        "observer_online_inputs": ["applied_voltage", "armature_current"],
        "observer_parameters": asdict(DCMotorParams()),
        "observer_parameter_mode": "fixed_nominal_even_under_parameter_variation",
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "primary_scenarios": list(PRIMARY_SCENARIOS),
        "run_count": int(len(runs)),
        "paired_design": "same scenario and seed; frozen B/C3 rows reused only after exact provenance verification, otherwise simulate_v3_run rerun",
        "ekf_config_sha256": sha256(EKF_CONFIG_PATH),
        "closed_loop_plan_sha256": plan_lock["closed_loop_plan_sha256"],
        "closed_loop_evaluator_sha256": plan_lock["closed_loop_evaluator_sha256"],
        "calibration_lock": calibration_lock,
        "runtime_source_hashes": runtime_hashes,
        "baseline_provenance": baseline_provenance,
        "runtime_accounting": {
            "per_sample_total_overhead": "ekf_update_ms + aux_inference_ms + reliability_update_ms",
            "optimizer_timing": "slsqp_solve_ms recorded separately",
            "distribution_statistics": ["mean", "median", "p95", "p99", "max"],
        },
        "runtime_overhead_vs_c3": runtime_overhead_summary(runs),
        "role_gate": evaluate_role_gate(runs, calibration_locked),
        "summary": summary_rows,
        "wall_time_s": perf_counter() - started,
        "artifacts": {
            "runs": str(RUNS_PATH.relative_to(PROJECT)).replace("\\", "/"),
            "traces": str(TRACE_PATH.relative_to(PROJECT)).replace("\\", "/"),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(runs)} paired development rows to {RUNS_PATH.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
