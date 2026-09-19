"""Closed-loop evaluator for the C4 three-way consistency attribution extension.

The frozen V3 evaluator remains the source of truth for B_plain_MPC and
C3_arbitration_MPC.  This script adds only C4_attribution_MPC and evaluates the
three controllers on paired trajectories.

Development mode is fixed to seeds 19026..19030 and the seven predeclared C4
development scenarios.  Final mode requires an explicit frozen C4 config with
a GO development-stage gate and five fresh seeds; it is never selected by
default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from auxiliary_sensor_model import AuxiliarySpeedEstimator, auxiliary_predict_online
from evaluate_v3_closed_loop import simulate_v3_run
from motor_model import DCMotorParams, dc_motor_dynamics
from mpc import LSTMMPC, MPCConfig, PIConfig, PIController
from reliability import (
    C4_ATTRIBUTION_STATE_NAMES,
    DualVirtualSensorArbitrator,
    SensorReliabilityMonitor,
    SuppressionEvidenceAction,
    SuppressionEvidencePolicy,
    ThreeWayConsistencyAttributor,
    load_lstm_model,
)


DEVELOPMENT_SEEDS = [19026, 19027, 19028, 19029, 19030]
V3_FINAL_SEEDS = [29026, 29027, 29028, 29029, 29030]
CONTROLLERS = ["B_plain_MPC", "C3_arbitration_MPC", "C4_attribution_MPC"]
DEVELOPMENT_SCENARIOS = [
    "load_disturbance",
    "combined_fault_load",
    "sensor_bias_15",
    "sensor_dropout",
    "parameter_variation",
    "step_reference",
    "changing_reference",
]

DT = 0.01
CONTROL_STRIDE = 5
CONTROL_DT = CONTROL_STRIDE * DT
DURATION = 6.0
TIME = np.arange(0.0, DURATION + DT / 2, DT)
FULL_SCALE = 65.234375
SPEED_NOISE_STD = 0.25

SCENARIOS = {
    "nominal_tracking": {
        "title": "Nominal tracking",
        "reference": "constant",
        "event_start": None,
        "event_end": None,
        "type": "reference",
    },
    "step_reference": {
        "title": "Step reference",
        "reference": "step",
        "event_start": None,
        "event_end": None,
        "type": "reference",
    },
    "changing_reference": {
        "title": "Changing reference",
        "reference": "changing",
        "event_start": None,
        "event_end": None,
        "type": "reference",
    },
    "sensor_noise": {
        "title": "Sensor noise",
        "event_start": 2.0,
        "event_end": 4.0,
        "type": "sensor",
    },
    "sensor_bias_5": {
        "title": "+5% sensor bias",
        "event_start": 2.0,
        "event_end": 4.0,
        "type": "sensor",
    },
    "sensor_bias_15": {
        "title": "+15% sensor bias",
        "event_start": 2.0,
        "event_end": 4.0,
        "type": "sensor",
    },
    "sensor_dropout": {
        "title": "Sensor dropout",
        "event_start": 2.0,
        "event_end": 4.0,
        "type": "sensor",
    },
    "sensor_drift": {
        "title": "Sensor drift",
        "event_start": 2.0,
        "event_end": None,
        "type": "sensor",
    },
    "load_disturbance": {
        "title": "Load disturbance",
        "event_start": 3.0,
        "event_end": None,
        "type": "plant",
    },
    "parameter_variation": {
        "title": "Parameter variation",
        "event_start": 3.0,
        "event_end": None,
        "type": "plant",
    },
    "combined_fault_load": {
        "title": "Sensor fault + load disturbance",
        "event_start": 3.0,
        "event_end": None,
        "type": "combined",
    },
}

FINAL_SCENARIOS = list(SCENARIOS)
SOURCE_NAMES = np.array(["PHYSICAL", "MAIN_VIRTUAL", "AUX_VIRTUAL", "FALLBACK"])

nominal_params = DCMotorParams()
shifted_params = replace(
    nominal_params,
    resistance=1.20 * nominal_params.resistance,
    inductance=0.85 * nominal_params.inductance,
    back_emf_constant=1.15 * nominal_params.back_emf_constant,
    torque_constant=0.85 * nominal_params.torque_constant,
    inertia=1.20 * nominal_params.inertia,
    viscous_friction=1.20 * nominal_params.viscous_friction,
    coulomb_friction=1.20 * nominal_params.coulomb_friction,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_c4_script_hashes() -> dict[str, str]:
    return {
        "c4_evaluator_script": sha256(Path(__file__).resolve()),
        "c4_verifier_script": sha256(PROJECT / "scripts/verify_c4_results.py"),
    }


def _coerce_seed_list(values: Any, *, label: str) -> list[int]:
    if not isinstance(values, list) or len(values) != 5:
        raise RuntimeError(f"{label} must contain exactly five seeds")
    seeds: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise RuntimeError(f"{label} contains a boolean seed")
        try:
            seed = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{label} contains a non-integer seed: {value!r}") from exc
        if isinstance(value, float) and not value.is_integer():
            raise RuntimeError(f"{label} contains a non-integral seed: {value!r}")
        if isinstance(value, str) and str(seed) != value.strip():
            raise RuntimeError(f"{label} contains a non-canonical integer seed: {value!r}")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"{label} must contain five unique seeds after integer coercion")
    return seeds


def _hash_binding(mapping: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, str):
            return value
    return None


def _final_config_hash_mapping(config: dict[str, Any]) -> dict[str, Any]:
    hashes = config.get("hashes")
    return hashes if isinstance(hashes, dict) else config


def _validate_final_attributor_contract(config: dict[str, Any], calibration_path: Path) -> None:
    calibration_hash = sha256(calibration_path)
    hash_mapping = _final_config_hash_mapping(config)
    configured_calibration_hash = _hash_binding(
        hash_mapping,
        "c4_attribution_calibration",
        "c4_attribution_calibration_json",
        "c4_attribution_calibration_sha256",
        "calibration_sha256",
    ) or _hash_binding(
        config,
        "c4_attribution_calibration_sha256",
        "calibration_sha256",
    )
    if configured_calibration_hash != calibration_hash:
        raise RuntimeError("final C4 config must bind the exact supplied C4 attribution calibration")

    attributor_blocks = [
        value
        for key in ("attributor", "attribution")
        if isinstance((value := config.get(key)), dict)
    ]
    if len(attributor_blocks) != 1:
        raise RuntimeError("final C4 config must contain exactly one complete frozen attributor block")
    frozen = attributor_blocks[0]
    required = {"agreement_thresholds", "disagreement_thresholds", "ewma_alpha", "enter_count", "exit_count"}
    if not required <= set(frozen):
        raise RuntimeError("final C4 config frozen attributor block is incomplete")

    calibrated = load_c4_attribution_parameters(calibration_path)
    candidate = {
        "agreement_thresholds": tuple(float(value) for value in frozen["agreement_thresholds"]),
        "disagreement_thresholds": tuple(float(value) for value in frozen["disagreement_thresholds"]),
        "ewma_alpha": float(frozen["ewma_alpha"]),
        "enter_count": int(frozen["enter_count"]),
        "exit_count": int(frozen["exit_count"]),
    }
    if candidate != calibrated:
        raise RuntimeError("final C4 config frozen attributor parameters differ from the supplied calibration")


def _validate_preuse_seed_provenance(config_path: Path, config: dict[str, Any], seeds: list[int]) -> Path:
    reference = (
        config.get("seed_provenance")
        or config.get("seed_provenance_path")
        or config.get("final_seed_provenance")
        or config.get("candidate_seed_audit")
    )
    if not isinstance(reference, str) or not reference.strip():
        final_status = config.get("final_holdout_status")
        if isinstance(final_status, dict):
            reference = final_status.get("seed_provenance") or final_status.get("candidate_seed_audit")
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("final C4 config must reference a pre-use seed provenance artifact")
    provenance_path = (PROJECT / reference).resolve()
    try:
        provenance_path.relative_to(PROJECT.resolve())
    except ValueError as exc:
        raise RuntimeError("final C4 seed provenance path must stay inside the project") from exc
    if not provenance_path.is_file():
        raise RuntimeError("final C4 seed provenance artifact does not exist")
    if provenance_path.stat().st_mtime > config_path.stat().st_mtime:
        raise RuntimeError("final C4 seed provenance artifact must predate the frozen final config")

    hash_mapping = _final_config_hash_mapping(config)
    expected_hash = _hash_binding(
        hash_mapping,
        "c4_final_seed_provenance",
        "c4_final_seed_provenance_sha256",
        "c4_seed_provenance",
        "seed_provenance_sha256",
        "c4_candidate_seed_audit",
        "c4_candidate_seed_audit_json",
        "c4_candidate_seed_audit_sha256",
    )
    if expected_hash != sha256(provenance_path):
        raise RuntimeError("final C4 config must bind the exact pre-use seed provenance artifact")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    seed_lists = [
        value
        for key, value in provenance.items()
        if "seed" in str(key).lower() and isinstance(value, list)
    ]
    exact_seed_coverage = False
    for values in seed_lists:
        try:
            provenance_seeds = _coerce_seed_list(values, label="seed provenance")
        except RuntimeError:
            continue
        if set(provenance_seeds) == set(seeds):
            exact_seed_coverage = True
            break
    if not exact_seed_coverage:
        raise RuntimeError("pre-use seed provenance does not exactly cover the selected final seed set")

    zero_prior_use = False
    for key, value in provenance.items():
        lowered = str(key).lower()
        if "match" in lowered and "before" in lowered and "use" in lowered:
            if isinstance(value, dict) and value and all(float(count) == 0.0 for count in value.values()):
                zero_prior_use = True
        if "prior" in lowered and "use" in lowered and isinstance(value, list) and not value:
            zero_prior_use = True
    status = str(provenance.get("status", "")).lower()
    if not zero_prior_use or "before" not in status or ("final" not in status and "use" not in status):
        raise RuntimeError("pre-use seed provenance does not establish zero prior use before final evaluation")
    return provenance_path


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def rk4_step(state: np.ndarray, voltage: float, load: float, params: DCMotorParams) -> np.ndarray:
    derivative = lambda val: dc_motor_dynamics(0.0, val, voltage, load, params)
    k1 = derivative(state)
    k2 = derivative(state + DT * k1 / 2)
    k3 = derivative(state + DT * k2 / 2)
    k4 = derivative(state + DT * k3)
    return state + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def reference_values(name: str, times: np.ndarray | float = TIME) -> np.ndarray:
    times = np.asarray(times)
    kind = SCENARIOS[name].get("reference", "constant")
    if kind == "step":
        return np.where(times < 1.5, 20.0, 40.0)
    if kind == "changing":
        return np.where(times < 2.0, 20.0, np.where(times < 4.0, 42.0, 30.0))
    return np.full(times.shape, 35.0)


def inject_measurement(scenario_name: str, t: float, true_speed: float, base_noise: float, rng: np.random.Generator) -> float:
    """Apply the offline scenario definition to the physical speed measurement."""
    measured = true_speed + base_noise
    if scenario_name == "sensor_bias_5" and 2.0 <= t < 4.0:
        measured += 0.05 * FULL_SCALE
    elif scenario_name == "sensor_bias_15" and 2.0 <= t < 4.0:
        measured += 0.15 * FULL_SCALE
    elif scenario_name == "sensor_dropout" and 2.0 <= t < 4.0:
        measured = 0.0
    elif scenario_name == "sensor_drift" and t >= 2.0:
        measured += 0.15 * FULL_SCALE * min(1.0, (t - 2.0) / 4.0)
    elif scenario_name == "combined_fault_load" and t >= 3.0:
        measured += 0.05 * FULL_SCALE
    elif scenario_name == "sensor_noise" and 2.0 <= t < 4.0:
        measured += rng.normal(0.0, 2.0)
    return float(measured)


def load_c4_attribution_parameters(calibration_path: Path, final_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load frozen/calibrated attributor parameters without outcome-dependent tuning."""
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibrated = calibration["computed_thresholds"]
    params = {
        "agreement_thresholds": tuple(float(v) for v in calibrated["agreement_thresholds_ordered"]),
        "disagreement_thresholds": tuple(float(v) for v in calibrated["disagreement_thresholds_ordered"]),
        "ewma_alpha": float(calibrated["diagnostic_ewma_alpha"]),
        "enter_count": int(calibrated["enter_count"]),
        "exit_count": int(calibrated["exit_count"]),
    }
    if final_config is None:
        return params

    expected_hash = final_config.get("calibration_sha256") or final_config.get("c4_attribution_calibration_sha256")
    if expected_hash is not None and expected_hash != sha256(calibration_path):
        raise RuntimeError("C4 final config calibration hash does not match the supplied calibration file")

    frozen = final_config.get("attributor") or final_config.get("attribution")
    if isinstance(frozen, dict):
        candidate = {
            "agreement_thresholds": tuple(frozen.get("agreement_thresholds", params["agreement_thresholds"])),
            "disagreement_thresholds": tuple(frozen.get("disagreement_thresholds", params["disagreement_thresholds"])),
            "ewma_alpha": float(frozen.get("ewma_alpha", params["ewma_alpha"])),
            "enter_count": int(frozen.get("enter_count", params["enter_count"])),
            "exit_count": int(frozen.get("exit_count", params["exit_count"])),
        }
        for key in ("agreement_thresholds", "disagreement_thresholds"):
            candidate[key] = tuple(float(v) for v in candidate[key])
        if candidate != params:
            raise RuntimeError("frozen C4 final attributor parameters differ from clean calibration")
    return params


def _snapshot_monitor(sensor: SensorReliabilityMonitor) -> dict[str, float | int | bool]:
    return {
        "positive": float(sensor.positive),
        "negative": float(sensor.negative),
        "abnormal_run": int(sensor.abnormal_run),
        "healthy_run": int(sensor.healthy_run),
        "active": bool(sensor.active),
    }


def apply_c4_entry_attribution(
    *,
    sensor: SensorReliabilityMonitor,
    attributor: ThreeWayConsistencyAttributor,
    evidence_policy: SuppressionEvidencePolicy,
    y_sensor: float,
    y_main: float,
    y_aux: float,
    auxiliary_trusted: bool,
    c3_decision: dict[str, Any],
    prior_monitor: dict[str, float | int | bool],
) -> dict[str, Any]:
    """Authorize one C3 entry/substitution candidate using only online signals.

    This function is intentionally isolated so regression/static checks can
    verify that C4 entry authorization has no scenario, event, or truth input.
    Once a C3 latch was already active, C4 preserves that latch until the
    normal C3 recovery logic releases it.
    """
    was_active = bool(prior_monitor["active"])
    raw_c3_candidate = bool(c3_decision["substitute"])
    new_entry_candidate = bool(raw_c3_candidate and not was_active)

    attr = attributor.update(
        y_sensor,
        y_main,
        y_aux,
        auxiliary_trusted=auxiliary_trusted,
        detector_requests_entry=new_entry_candidate,
    )

    action = SuppressionEvidenceAction[attr["suppression_evidence_action"]]
    if not was_active and action != SuppressionEvidenceAction.NONE:
        positive, negative = evidence_policy.apply(
            action,
            float(prior_monitor["positive"]),
            float(prior_monitor["negative"]),
            float(sensor.positive),
            float(sensor.negative),
        )
        sensor.positive = positive
        sensor.negative = negative
        if action == SuppressionEvidenceAction.RESET:
            sensor.abnormal_run = 0
            sensor.healthy_run = 0
        else:
            sensor.abnormal_run = int(prior_monitor["abnormal_run"])
            sensor.healthy_run = int(prior_monitor["healthy_run"])

    suppressed = bool(not was_active and attr["entry_suppressed"])
    if suppressed:
        sensor.active = False
        sensor.abnormal_run = 0
        sensor.healthy_run = 0
        actual_substitute = False
        actual_sensor_suspect = False
    else:
        actual_substitute = raw_c3_candidate
        actual_sensor_suspect = bool(c3_decision["sensor_suspect"])

    return {
        **attr,
        "raw_c3_candidate": raw_c3_candidate,
        "new_c3_entry_candidate": new_entry_candidate,
        "actual_substitute": actual_substitute,
        "actual_sensor_suspect": actual_sensor_suspect,
    }


def c4_auxiliary_witness_status(
    *,
    t: float,
    y_aux: float,
    arbitrator: DualVirtualSensorArbitrator,
    trusted_consistency_count: int,
) -> tuple[bool, str]:
    """Qualify the auxiliary estimator as a conservative C4 attribution witness.

    Readiness is stricter than frozen C3 arbitration.  It requires the frozen
    V3 startup blanking, finite/bounded auxiliary output, no latched auxiliary
    parameter mismatch, and a full V3 recovery-count history of prior trusted
    physical/auxiliary consistency observations.
    """
    low, high = arbitrator.aux_speed_bounds
    if t < arbitrator.startup_blanking_time:
        return False, "STARTUP_BLANKING"
    if not np.isfinite(y_aux):
        return False, "AUX_NONFINITE"
    if not low <= y_aux <= high:
        return False, "AUX_OUT_OF_BOUNDS"
    if arbitrator.aux_param_mismatch:
        return False, "AUX_PARAM_MISMATCH"
    if trusted_consistency_count < arbitrator.param_mismatch_recovery_count:
        return False, "INSUFFICIENT_TRUSTED_HISTORY"
    if arbitrator.ewma_aux_trusted > arbitrator.param_mismatch_recovery_threshold:
        return False, "AUX_EWMA_INCONSISTENT"
    return True, "READY"


def update_c4_auxiliary_readiness(
    *,
    prior_count: int,
    auxiliary_consistent_now: bool,
    raw_c3_candidate: bool,
    actual_sensor_latched: bool,
) -> int:
    """Advance C4 auxiliary-witness history without circular certification.

    Only ordinary C3-trusted, non-candidate samples accrue readiness.  A
    transient instantaneous C3 candidate while the sensor latch is inactive
    freezes whatever historical count already exists, including a partial
    count below the 50-sample readiness requirement.  A confirmed C4 veto is
    therefore also a freeze, because it is still a C3 candidate sample.

    Hard auxiliary invalidity/inconsistency is represented by
    ``auxiliary_consistent_now=False`` and resets the history.  An actually
    accepted/active sensor latch also resets it.  This keeps historical witness
    evidence available long enough for the locked three-sample raw PLANT
    attribution persistence to become reachable without accruing trust through
    the candidate samples themselves.
    """
    if prior_count < 0:
        raise ValueError("prior_count must be nonnegative")
    if not auxiliary_consistent_now or actual_sensor_latched:
        return 0
    if raw_c3_candidate:
        return int(prior_count)
    return int(prior_count) + 1


def run_c4_synthetic_integration_regression(attribution_params: dict[str, Any]) -> dict[str, Any]:
    """Exercise the C4 readiness/veto path without running development evidence.

    The sequence starts from a historically READY auxiliary witness, applies
    three raw main/plant-mismatch samples, and verifies that the candidate
    samples freeze rather than erase readiness so the third sample reaches the
    locked attribution persistence and veto.  It then changes immediately to a
    genuine physical-sensor-outlier pattern and verifies that veto stops and the
    unchanged three-sample C3 detector can latch the later fault.
    """
    sensor = SensorReliabilityMonitor(
        residual_gate=0.5,
        center=0.0,
        allowance=0.05,
        threshold=1.0,
        enter_count=3,
        exit_count=5,
    )
    attributor = ThreeWayConsistencyAttributor(**attribution_params)
    evidence_policy = SuppressionEvidencePolicy()
    arbitrator = DualVirtualSensorArbitrator()
    arbitrator.ewma_aux_trusted = 0.0
    trusted_count = arbitrator.param_mismatch_recovery_count

    plant_decisions: list[dict[str, Any]] = []
    for _ in range(3):
        prior = _snapshot_monitor(sensor)
        c3 = sensor.update(10.0 - 20.0, aux_residual=0.0)
        auxiliary_trusted, reason = c4_auxiliary_witness_status(
            t=2.0,
            y_aux=10.0,
            arbitrator=arbitrator,
            trusted_consistency_count=trusted_count,
        )
        if not auxiliary_trusted or reason != "READY":
            raise AssertionError("synthetic READY witness unexpectedly unavailable")
        decision = apply_c4_entry_attribution(
            sensor=sensor,
            attributor=attributor,
            evidence_policy=evidence_policy,
            y_sensor=10.0,
            y_main=20.0,
            y_aux=10.0,
            auxiliary_trusted=auxiliary_trusted,
            c3_decision=c3,
            prior_monitor=prior,
        )
        trusted_count = update_c4_auxiliary_readiness(
            prior_count=trusted_count,
            auxiliary_consistent_now=True,
            raw_c3_candidate=bool(decision["raw_c3_candidate"]),
            actual_sensor_latched=bool(decision["actual_sensor_suspect"]),
        )
        plant_decisions.append(decision)

    if [bool(row["entry_suppressed"]) for row in plant_decisions] != [False, False, True]:
        raise AssertionError("three raw PLANT samples did not reach the expected third-sample veto")
    if trusted_count != arbitrator.param_mismatch_recovery_count:
        raise AssertionError("PLANT candidates changed historical auxiliary readiness")
    if sensor.active:
        raise AssertionError("confirmed PLANT veto left the C3 sensor latch active")
    if not all(float(row["d_sa"]) <= attribution_params["agreement_thresholds"][1] for row in plant_decisions):
        raise AssertionError("synthetic PLANT veto did not satisfy the current raw d_sa agreement gate")

    sensor_fault_decisions: list[dict[str, Any]] = []
    for _ in range(3):
        prior = _snapshot_monitor(sensor)
        c3 = sensor.update(20.0 - 10.0, aux_residual=10.0)
        auxiliary_trusted, _ = c4_auxiliary_witness_status(
            t=2.1,
            y_aux=10.0,
            arbitrator=arbitrator,
            trusted_consistency_count=trusted_count,
        )
        decision = apply_c4_entry_attribution(
            sensor=sensor,
            attributor=attributor,
            evidence_policy=evidence_policy,
            y_sensor=20.0,
            y_main=10.0,
            y_aux=10.0,
            auxiliary_trusted=auxiliary_trusted,
            c3_decision=c3,
            prior_monitor=prior,
        )
        trusted_count = update_c4_auxiliary_readiness(
            prior_count=trusted_count,
            auxiliary_consistent_now=True,
            raw_c3_candidate=bool(decision["raw_c3_candidate"]),
            actual_sensor_latched=bool(decision["actual_sensor_suspect"]),
        )
        sensor_fault_decisions.append(decision)

    if any(bool(row["entry_suppressed"]) for row in sensor_fault_decisions):
        raise AssertionError("later genuine sensor-fault pattern remained vetoed")
    if not bool(sensor_fault_decisions[-1]["actual_sensor_suspect"]) or not sensor.active:
        raise AssertionError("later genuine sensor fault failed to latch after mismatch veto")
    if trusted_count != 0:
        raise AssertionError("accepted sensor latch did not reset auxiliary readiness")

    # Current raw sensor/aux agreement remains mandatory for the PLANT state.
    raw_gate_attributor = ThreeWayConsistencyAttributor(**attribution_params)
    above_agreement_aux = 10.0 + attribution_params["agreement_thresholds"][1] + 0.01
    raw_gate_rows = [
        raw_gate_attributor.update(
            10.0,
            25.0,
            above_agreement_aux,
            auxiliary_trusted=True,
            detector_requests_entry=True,
        )
        for _ in range(3)
    ]
    if any(bool(row["entry_suppressed"]) for row in raw_gate_rows):
        raise AssertionError("PLANT veto occurred with raw d_sa above the p75 agreement threshold")

    return {
        "ready_count": arbitrator.param_mismatch_recovery_count,
        "plant_suppression_pattern": [bool(row["entry_suppressed"]) for row in plant_decisions],
        "plant_final_state": plant_decisions[-1]["state"],
        "later_fault_detected": bool(sensor_fault_decisions[-1]["actual_sensor_suspect"]),
        "later_fault_suppressed": any(bool(row["entry_suppressed"]) for row in sensor_fault_decisions),
        "readiness_after_accepted_fault": trusted_count,
        "raw_d_sa_gate_preserved": True,
    }


def event_mask_for_scenario(scenario_name: str) -> np.ndarray:
    config = SCENARIOS[scenario_name]
    start = config["event_start"]
    end = config["event_end"]
    if start is None:
        return np.ones(len(TIME), dtype=bool)
    mask = TIME >= start
    if end is not None:
        mask &= TIME < end
    return mask


def is_genuine_sensor_fault_time(scenario_name: str, time_s: float) -> bool:
    """Offline-only scoring of an entry/suppression after simulation is complete."""
    config = SCENARIOS[scenario_name]
    if config["type"] not in {"sensor", "combined"}:
        return False
    start = config["event_start"]
    end = config["event_end"]
    if start is None or time_s < start:
        return False
    return end is None or time_s < end


def simulate_c4_run(args: tuple[str, int, dict[str, Any], bool]) -> dict[str, Any]:
    """Run one C4 closed-loop trajectory with frozen V3 plant/MPC semantics."""
    scenario_name, seed, attribution_params, record_trace = args
    torch.set_num_threads(1)

    device = "cpu"
    main_model, main_cfg = load_lstm_model(
        PROJECT / "results/lstm_model_weights.pt",
        PROJECT / "results/configs/lstm_model_config.json",
        device=device,
    )
    aux_cfg = json.loads((PROJECT / "results/configs/v2_auxiliary_config.json").read_text(encoding="utf-8"))
    v3_cfg = json.loads((PROJECT / "results/configs/v3_arbitration_config.json").read_text(encoding="utf-8"))
    rel_cfg = json.loads((PROJECT / "results/configs/reliability_final_config.json").read_text(encoding="utf-8"))

    aux_model = AuxiliarySpeedEstimator(**aux_cfg["model"]).to(device)
    aux_model.load_state_dict(
        torch.load(PROJECT / "results/auxiliary_model_weights.pt", map_location=device, weights_only=True)
    )
    aux_model.eval()

    sensor_cfg = rel_cfg["sensor"]
    residual_gate = float(sensor_cfg["instant_threshold"])
    cusum_threshold = float(sensor_cfg["threshold"])
    aux_recovery_gate = float(v3_cfg["arbitrator"]["aux_recovery_gate"])
    sensor = SensorReliabilityMonitor(
        residual_gate,
        float(sensor_cfg["center"]),
        float(sensor_cfg["allowance"]),
        cusum_threshold,
        int(sensor_cfg["enter_count"]),
        int(sensor_cfg["exit_count"]),
        aux_recovery_gate=aux_recovery_gate,
    )
    arbitrator = DualVirtualSensorArbitrator(
        **{key: value for key, value in v3_cfg["arbitrator"].items() if key != "aux_recovery_gate"}
    )
    attributor = ThreeWayConsistencyAttributor(**attribution_params)
    evidence_policy = SuppressionEvidencePolicy()

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
    mpc = LSTMMPC(main_model, main_cfg["normalization"], 20, mpc_config)
    aux_norm = aux_cfg["normalization"]

    rng = np.random.default_rng(seed)
    base_noise = rng.normal(0.0, SPEED_NOISE_STD, len(TIME))
    ref = reference_values(scenario_name)

    state = np.zeros(2)
    prev_voltage = 0.0
    history = np.zeros((20, 2), dtype=np.float32)
    aux_history = np.zeros((20, 2), dtype=np.float32)
    last_trusted_speed = 0.0
    last_main_prediction = 0.0

    true_speed_arr = np.zeros(len(TIME))
    meas_speed_arr = np.zeros(len(TIME))
    main_speed_arr = np.full(len(TIME), np.nan)
    aux_speed_arr = np.full(len(TIME), np.nan)
    feedback_arr = np.zeros(len(TIME))
    voltage_arr = np.zeros(len(TIME))
    sub_arr = np.zeros(len(TIME), dtype=bool)
    sensor_state_arr = np.zeros(len(TIME), dtype=bool)
    proposed_sensor_state_arr = np.zeros(len(TIME), dtype=bool)
    source_arr = np.zeros(len(TIME), dtype=int)
    c3_candidate_arr = np.zeros(len(TIME), dtype=bool)
    new_candidate_arr = np.zeros(len(TIME), dtype=bool)
    suppressed_arr = np.zeros(len(TIME), dtype=bool)
    aux_trusted_arr = np.zeros(len(TIME), dtype=bool)
    aux_hard_valid_arr = np.zeros(len(TIME), dtype=bool)
    aux_witness_reason_arr = np.full(len(TIME), "WARMUP", dtype=object)
    aux_param_mismatch_arr = np.zeros(len(TIME), dtype=bool)
    attr_state_arr = np.full(len(TIME), -1, dtype=int)
    raw_attr_state_arr = np.full(len(TIME), -1, dtype=int)
    d_sm_arr = np.full(len(TIME), np.nan)
    d_sa_arr = np.full(len(TIME), np.nan)
    d_ma_arr = np.full(len(TIME), np.nan)
    fd_sm_arr = np.full(len(TIME), np.nan)
    fd_sa_arr = np.full(len(TIME), np.nan)
    fd_ma_arr = np.full(len(TIME), np.nan)
    suppression_action_arr = np.zeros(len(TIME), dtype=int)

    solve_times: list[float] = []
    aux_times: list[float] = []
    reliability_times: list[float] = []
    attribution_times: list[float] = []
    arbitration_times: list[float] = []
    extension_times: list[float] = []
    control_times: list[float] = []
    timing_samples: list[dict[str, Any]] = []
    attribution_steps: list[dict[str, Any]] = []
    opt_failures = 0
    main_prediction_failures = 0
    aux_prediction_failures = 0
    prediction_failure_messages: set[str] = set()
    nonfinite_events = 0
    trusted_consistency_count = 0

    run_started = perf_counter()
    for i, t in enumerate(TIME):
        current_val, true_speed = float(state[0]), float(state[1])
        true_speed_arr[i] = true_speed
        measured = inject_measurement(scenario_name, float(t), true_speed, float(base_noise[i]), rng)
        meas_speed_arr[i] = measured

        if i >= 20:
            try:
                y_main = float(mpc.predict(history, np.array([prev_voltage], dtype=np.float32))[0])
                last_main_prediction = y_main
            except (ValueError, FloatingPointError, RuntimeError) as error:
                main_prediction_failures += 1
                prediction_failure_messages.add(f"main:{type(error).__name__}:{error}")
                y_main = last_main_prediction
        else:
            y_main = measured
            if np.isfinite(y_main):
                last_main_prediction = y_main
        main_speed_arr[i] = y_main

        y_aux = np.nan
        aux_ms = reliability_ms = attribution_ms = arbitration_ms = 0.0
        if i >= 20:
            aux_started = perf_counter()
            try:
                y_aux = auxiliary_predict_online(aux_model, aux_history[:, 0], aux_history[:, 1], aux_norm)
            except (ValueError, FloatingPointError, RuntimeError) as error:
                aux_prediction_failures += 1
                prediction_failure_messages.add(f"aux:{type(error).__name__}:{error}")
            aux_ms = 1000.0 * (perf_counter() - aux_started)
            aux_times.append(aux_ms)
        aux_speed_arr[i] = y_aux

        if i < 20:
            is_sub = False
            is_suspect = False
            y_feedback = measured
            source_code = 0
        else:
            main_residual = measured - y_main
            aux_residual = measured - y_aux if np.isfinite(y_aux) else None
            prior_monitor = _snapshot_monitor(sensor)
            reliability_started = perf_counter()
            c3_decision = sensor.update(main_residual, aux_residual=aux_residual)
            reliability_ms = 1000.0 * (perf_counter() - reliability_started)
            reliability_times.append(reliability_ms)
            proposed_sensor_state_arr[i] = bool(c3_decision["sensor_suspect"])

            attribution_started = perf_counter()
            trusted_consistency_count_before = trusted_consistency_count
            low, high = arbitrator.aux_speed_bounds
            aux_hard_valid = bool(
                np.isfinite(y_aux)
                and low <= y_aux <= high
                and not arbitrator.aux_param_mismatch
            )
            auxiliary_trusted, auxiliary_trust_reason = c4_auxiliary_witness_status(
                t=float(t),
                y_aux=y_aux,
                arbitrator=arbitrator,
                trusted_consistency_count=trusted_consistency_count,
            )
            c4_decision = apply_c4_entry_attribution(
                sensor=sensor,
                attributor=attributor,
                evidence_policy=evidence_policy,
                y_sensor=measured,
                y_main=y_main,
                y_aux=y_aux,
                auxiliary_trusted=auxiliary_trusted,
                c3_decision=c3_decision,
                prior_monitor=prior_monitor,
            )
            attribution_ms = 1000.0 * (perf_counter() - attribution_started)
            attribution_times.append(attribution_ms)

            is_sub = bool(c4_decision["actual_substitute"])
            is_suspect = bool(c4_decision["actual_sensor_suspect"])
            c3_candidate_arr[i] = bool(c4_decision["raw_c3_candidate"])
            new_candidate_arr[i] = bool(c4_decision["new_c3_entry_candidate"])
            suppressed_arr[i] = bool(c4_decision["entry_suppressed"] and not prior_monitor["active"])
            aux_trusted_arr[i] = bool(c4_decision["auxiliary_trusted"])
            aux_hard_valid_arr[i] = aux_hard_valid
            aux_witness_reason_arr[i] = auxiliary_trust_reason
            aux_param_mismatch_arr[i] = bool(arbitrator.aux_param_mismatch)
            attr_state_arr[i] = int(c4_decision["state_code"])
            raw_attr_state_arr[i] = int(c4_decision["raw_state_code"])
            d_sm_arr[i] = float(c4_decision["d_sm"])
            d_sa_arr[i] = float(c4_decision["d_sa"])
            d_ma_arr[i] = float(c4_decision["d_ma"])
            fd_sm_arr[i] = float(c4_decision["filtered_d_sm"])
            fd_sa_arr[i] = float(c4_decision["filtered_d_sa"])
            fd_ma_arr[i] = float(c4_decision["filtered_d_ma"])
            suppression_action_arr[i] = int(c4_decision["suppression_evidence_action_code"])

            arbitration_started = perf_counter()
            arb_decision = arbitrator.update(
                t=float(t),
                y_measured=measured,
                y_main=y_main,
                y_aux=y_aux,
                sensor_trusted=not is_sub,
                fallback_speed=last_trusted_speed,
            )
            arbitration_ms = 1000.0 * (perf_counter() - arbitration_started)
            arbitration_times.append(arbitration_ms)
            y_feedback = float(arb_decision["feedback"])
            source_code = int(arb_decision["source_code"])

            # C4-only witness-readiness history.  Ordinary non-candidate,
            # C3-trusted physical samples extend the history.  Instantaneous
            # C3 candidates freeze the historical count while the sensor latch
            # remains inactive; a confirmed C4 veto therefore freezes too.
            # Invalid/inconsistent auxiliary evidence or an actually accepted
            # sensor latch resets readiness.
            auxiliary_consistent_now = bool(
                float(t) >= arbitrator.startup_blanking_time
                and np.isfinite(y_aux)
                and low <= y_aux <= high
                and not arbitrator.aux_param_mismatch
                and arbitrator.ewma_aux_trusted <= arbitrator.param_mismatch_recovery_threshold
            )
            trusted_consistency_count = update_c4_auxiliary_readiness(
                prior_count=trusted_consistency_count_before,
                auxiliary_consistent_now=auxiliary_consistent_now,
                raw_c3_candidate=bool(c4_decision["raw_c3_candidate"]),
                actual_sensor_latched=is_suspect,
            )

            attribution_steps.append(
                {
                    "step": i,
                    "time_s": float(t),
                    "state": c4_decision["state"],
                    "state_code": int(c4_decision["state_code"]),
                    "raw_state": c4_decision["raw_state"],
                    "candidate_state": c4_decision["candidate_state"],
                    "state_persistence": int(c4_decision["state_persistence"]),
                    "d_sm": float(c4_decision["d_sm"]),
                    "d_sa": float(c4_decision["d_sa"]),
                    "d_ma": float(c4_decision["d_ma"]),
                    "filtered_d_sm": float(c4_decision["filtered_d_sm"]),
                    "filtered_d_sa": float(c4_decision["filtered_d_sa"]),
                    "filtered_d_ma": float(c4_decision["filtered_d_ma"]),
                    "auxiliary_trusted": bool(c4_decision["auxiliary_trusted"]),
                    "auxiliary_trust_reason": auxiliary_trust_reason,
                    "auxiliary_hard_valid": aux_hard_valid,
                    "aux_param_mismatch": bool(arbitrator.aux_param_mismatch),
                    "trusted_consistency_count_before": int(trusted_consistency_count_before),
                    "c3_requested_entry": bool(c4_decision["new_c3_entry_candidate"]),
                    "c3_candidate": bool(c4_decision["raw_c3_candidate"]),
                    "c4_suppressed_entry": bool(suppressed_arr[i]),
                    "actual_substitute": is_sub,
                    "selected_source": str(arb_decision["source"]),
                    "source_code": source_code,
                    "suppression_evidence_action": c4_decision["suppression_evidence_action"],
                    "attribution_ms": attribution_ms,
                }
            )

        if not is_sub and np.isfinite(measured):
            last_trusted_speed = float(np.clip(measured, *arbitrator.aux_speed_bounds))

        required = [measured, y_feedback, y_main]
        nonfinite_events += int(not np.isfinite(required).all())
        feedback_arr[i] = y_feedback
        sub_arr[i] = is_sub
        sensor_state_arr[i] = is_suspect
        source_arr[i] = source_code

        history = np.vstack((history[1:], [prev_voltage, y_feedback])).astype(np.float32)
        aux_history = np.vstack((aux_history[1:], [prev_voltage, current_val])).astype(np.float32)

        voltage = prev_voltage
        if i % CONTROL_STRIDE == 0:
            control_started = perf_counter()
            fallback_voltage = pi.compute_control(ref[i], y_feedback, CONTROL_DT)
            solve_started = perf_counter()
            output = mpc.compute_control(
                history,
                reference_values(scenario_name, t + DT * np.arange(1, mpc_config.horizon + 1)),
                prev_voltage,
                horizon=mpc_config.horizon,
                move_blocks=mpc_config.move_blocks,
                fallback_voltage=fallback_voltage,
            )
            solve_times.append(perf_counter() - solve_started)
            if not output["success"]:
                opt_failures += 1
            voltage = float(output["voltage"])
            control_ms = 1000.0 * (perf_counter() - control_started)
            control_times.append(control_ms)
        else:
            control_ms = np.nan

        if i >= 20:
            extension_ms = aux_ms + reliability_ms + attribution_ms + arbitration_ms
            extension_times.append(extension_ms)
            timing_samples.append(
                {
                    "controller": "C4_attribution_MPC",
                    "scenario": scenario_name,
                    "seed": seed,
                    "step": i,
                    "time_s": float(t),
                    "aux_inference_ms": aux_ms,
                    "reliability_update_ms": reliability_ms,
                    "attribution_ms": attribution_ms,
                    "arbitration_ms": arbitration_ms,
                    "extension_ms": extension_ms,
                    "control_compute_ms": control_ms,
                }
            )

        voltage_arr[i] = voltage
        history[-1, 0] = voltage
        aux_history[-1, 0] = voltage
        prev_voltage = voltage

        load_val = 0.15 if scenario_name in ("load_disturbance", "combined_fault_load") and t >= 3.0 else 0.03
        params = shifted_params if scenario_name == "parameter_variation" and t >= 3.0 else nominal_params
        if i < len(TIME) - 1:
            state = rk4_step(state, voltage, load_val, params)

    total_sim_time = perf_counter() - run_started
    error = ref - true_speed_arr
    event_mask = event_mask_for_scenario(scenario_name)
    overall_rmse = float(np.sqrt(np.mean(error**2)))
    overall_mae = float(np.mean(np.abs(error)))
    fault_window_rmse = float(np.sqrt(np.mean(error[event_mask] ** 2)))

    source_switches = int(np.sum(source_arr[:-1] != source_arr[1:]))
    active_entries = np.flatnonzero(~sensor_state_arr[:-1] & sensor_state_arr[1:]) + 1
    actual_entry_times = [float(TIME[index]) for index in active_entries]
    genuine_entries = sum(is_genuine_sensor_fault_time(scenario_name, value) for value in actual_entry_times)
    false_entries = len(actual_entry_times) - genuine_entries
    suppressed_indices = np.flatnonzero(suppressed_arr)
    suppression_episode_bounds: list[tuple[int, int]] = []
    if len(suppressed_indices):
        episode_start = int(suppressed_indices[0])
        previous = int(suppressed_indices[0])
        for current_value in suppressed_indices[1:]:
            current = int(current_value)
            if current != previous + 1:
                suppression_episode_bounds.append((episode_start, previous))
                episode_start = current
            previous = current
        suppression_episode_bounds.append((episode_start, previous))

    genuine_suppressed = 0
    for start_index, end_index in suppression_episode_bounds:
        if any(
            is_genuine_sensor_fault_time(scenario_name, float(TIME[index]))
            for index in range(start_index, end_index + 1)
        ):
            genuine_suppressed += 1
    suppressed_false = int(len(suppression_episode_bounds) - genuine_suppressed)

    scenario_config = SCENARIOS[scenario_name]
    event_start = scenario_config["event_start"]
    detection_latency_s = np.nan
    detected_event = np.nan
    if scenario_config["type"] in {"sensor", "combined"} and event_start is not None:
        event_sub = np.flatnonzero(sub_arr & event_mask)
        detected_event = bool(len(event_sub))
        if len(event_sub):
            detection_latency_s = float(TIME[event_sub[0]] - event_start)

    valid_attr = attr_state_arr >= 0
    attribution_switches = int(np.sum(attr_state_arr[np.flatnonzero(valid_attr)][:-1] != attr_state_arr[np.flatnonzero(valid_attr)][1:])) if np.any(valid_attr) else 0
    attribution_samples = int(np.sum(valid_attr))
    state_fractions = {}
    for code, name in enumerate(C4_ATTRIBUTION_STATE_NAMES):
        state_fractions[f"attr_frac_{name.lower()}"] = (
            float(np.mean(attr_state_arr[valid_attr] == code)) if attribution_samples else np.nan
        )

    ctrl_voltages = voltage_arr[::CONTROL_STRIDE]
    ctrl_moves = np.diff(np.r_[0.0, ctrl_voltages])
    sub_indices = np.flatnonzero(sub_arr)
    if len(sub_indices):
        main_error = main_speed_arr[sub_indices] - true_speed_arr[sub_indices]
        selected_error = feedback_arr[sub_indices] - true_speed_arr[sub_indices]
        corrupted_error = meas_speed_arr[sub_indices] - true_speed_arr[sub_indices]
        valid_aux = np.isfinite(aux_speed_arr[sub_indices])
        aux_rmse = (
            float(np.sqrt(np.mean((aux_speed_arr[sub_indices][valid_aux] - true_speed_arr[sub_indices][valid_aux]) ** 2)))
            if np.any(valid_aux)
            else np.nan
        )
        quality = {
            "main_virtual_rmse": float(np.sqrt(np.mean(main_error**2))),
            "aux_virtual_rmse": aux_rmse,
            "selected_virtual_rmse": float(np.sqrt(np.mean(selected_error**2))),
            "corrupted_rmse": float(np.sqrt(np.mean(corrupted_error**2))),
            "sub_samples": int(len(sub_indices)),
        }
    else:
        quality = {
            "main_virtual_rmse": np.nan,
            "aux_virtual_rmse": np.nan,
            "selected_virtual_rmse": np.nan,
            "corrupted_rmse": np.nan,
            "sub_samples": 0,
        }

    result: dict[str, Any] = {
        "controller": "C4_attribution_MPC",
        "scenario": scenario_name,
        "seed": seed,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "fault_window_rmse": fault_window_rmse,
        "detection_latency_s": detection_latency_s,
        "detected_event": detected_event,
        "reliability_entries": int(len(active_entries)),
        "genuine_sensor_fault_entries": int(genuine_entries),
        "genuine_sensor_fault_entries_permitted": int(genuine_entries),
        "false_sensor_fault_entries": int(false_entries),
        "c3_candidate_samples": int(np.sum(c3_candidate_arr)),
        "c3_new_entry_candidate_samples": int(np.sum(new_candidate_arr)),
        "suppressed_entry_count": int(len(suppression_episode_bounds)),
        "suppressed_candidate_samples": int(np.sum(suppressed_arr)),
        "suppressed_false_entry_count": suppressed_false,
        "suppressed_false_entries": suppressed_false,
        "genuine_sensor_fault_suppressed": genuine_suppressed,
        "genuine_sensor_fault_entries_suppressed": genuine_suppressed,
        "sensor_fault_detected": detected_event,
        "suppression_episodes": int(len(suppression_episode_bounds)),
        "attributor_suppression_episodes": int(attributor.suppression_episodes),
        "sub_duration_s": float(np.sum(sub_arr) * DT),
        "sub_fraction": float(np.mean(sub_arr)),
        "switches": source_switches,
        "avg_episode_duration_s": float(DURATION / (source_switches + 1)),
        "frac_phys": float(np.mean(source_arr == 0)),
        "frac_main": float(np.mean(source_arr == 1)),
        "frac_aux": float(np.mean(source_arr == 2)),
        "frac_fb": float(np.mean(source_arr == 3)),
        "event_frac_aux": float(np.mean(source_arr[event_mask] == 2)),
        "event_frac_fb": float(np.mean(source_arr[event_mask] == 3)),
        "auxiliary_misuse_events": int(np.sum((source_arr == 2) & ~aux_hard_valid_arr)),
        "aux_selected_while_witness_unready_samples": int(np.sum((source_arr == 2) & ~aux_trusted_arr)),
        "aux_witness_ready_fraction": float(np.mean(aux_trusted_arr[20:])),
        "untrusted_aux_fraction": float(np.mean(~aux_trusted_arr[20:])),
        "untrusted_aux_candidate_samples": int(np.sum(c3_candidate_arr & ~aux_trusted_arr)),
        "attribution_switches": attribution_switches,
        "mean_attribution_dwell_s": float(attribution_samples * DT / (attribution_switches + 1)) if attribution_samples else np.nan,
        "control_effort_u2": float(np.sum(ctrl_voltages**2)),
        "control_variation_du2": float(np.sum(ctrl_moves**2)),
        "voltage_violations": int(np.sum((ctrl_voltages < -1e-8) | (ctrl_voltages > 12.0 + 1e-8))),
        "rate_violations": int(np.sum(np.abs(ctrl_moves) > 2.0 + 1e-8)),
        "optimizer_failures": opt_failures,
        "main_prediction_failures": main_prediction_failures,
        "aux_prediction_failures": aux_prediction_failures,
        "prediction_failure_messages": " | ".join(sorted(prediction_failure_messages)),
        "nonfinite_events": nonfinite_events,
        "mean_solve_ms": float(np.mean(solve_times) * 1000.0) if solve_times else 0.0,
        "mean_aux_inference_ms": float(np.mean(aux_times)) if aux_times else 0.0,
        "mean_reliability_update_ms": float(np.mean(reliability_times)) if reliability_times else 0.0,
        "mean_attribution_ms": float(np.mean(attribution_times)) if attribution_times else 0.0,
        "median_attribution_ms": float(np.median(attribution_times)) if attribution_times else 0.0,
        "p95_attribution_ms": float(np.percentile(attribution_times, 95.0)) if attribution_times else 0.0,
        "p99_attribution_ms": float(np.percentile(attribution_times, 99.0)) if attribution_times else 0.0,
        "mean_arbitration_ms": float(np.mean(arbitration_times)) if arbitration_times else 0.0,
        "mean_extension_ms": float(np.mean(extension_times)) if extension_times else 0.0,
        "mean_control_compute_ms": float(np.mean(control_times)) if control_times else 0.0,
        "sim_time_s": total_sim_time,
        **state_fractions,
        **quality,
        "actual_entry_times": actual_entry_times,
        "timing_samples": timing_samples,
        "attribution_steps": attribution_steps,
    }

    if record_trace:
        result["trace"] = {
            "time": TIME.copy(),
            "true": true_speed_arr,
            "meas": meas_speed_arr,
            "main": main_speed_arr,
            "aux": aux_speed_arr,
            "fb": feedback_arr,
            "source": source_arr,
            "ref": ref,
            "voltage": voltage_arr,
            "sub": sub_arr,
            "sensor_state": sensor_state_arr,
            "proposed_sensor_state": proposed_sensor_state_arr,
            "c3_candidate": c3_candidate_arr,
            "new_candidate": new_candidate_arr,
            "suppressed": suppressed_arr,
            "aux_trusted": aux_trusted_arr,
            "aux_hard_valid": aux_hard_valid_arr,
            "aux_witness_reason": aux_witness_reason_arr,
            "aux_param_mismatch": aux_param_mismatch_arr,
            "attr_state": attr_state_arr,
            "raw_attr_state": raw_attr_state_arr,
            "d_sm": d_sm_arr,
            "d_sa": d_sa_arr,
            "d_ma": d_ma_arr,
            "filtered_d_sm": fd_sm_arr,
            "filtered_d_sa": fd_sa_arr,
            "filtered_d_ma": fd_ma_arr,
            "suppression_action": suppression_action_arr,
        }
    return result


def _baseline_detection_fields(result: dict[str, Any]) -> None:
    """Add offline event scoring to frozen V3 baseline reruns without changing V3."""
    scenario_name = result["scenario"]
    scenario_config = SCENARIOS[scenario_name]
    result.setdefault("genuine_sensor_fault_entries", np.nan)
    result.setdefault("genuine_sensor_fault_entries_permitted", np.nan)
    result.setdefault("false_sensor_fault_entries", np.nan)
    result.setdefault("suppressed_entry_count", 0)
    result.setdefault("suppressed_candidate_samples", 0)
    result.setdefault("suppressed_false_entry_count", 0)
    result.setdefault("suppressed_false_entries", 0)
    result.setdefault("genuine_sensor_fault_suppressed", 0)
    result.setdefault("genuine_sensor_fault_entries_suppressed", 0)
    result.setdefault("suppression_episodes", 0)
    result.setdefault("attributor_suppression_episodes", 0)
    result.setdefault("c3_candidate_samples", np.nan)
    result.setdefault("c3_new_entry_candidate_samples", np.nan)
    result.setdefault("attribution_switches", np.nan)
    result.setdefault("mean_attribution_dwell_s", np.nan)
    result.setdefault("auxiliary_misuse_events", 0)
    result.setdefault("aux_selected_while_witness_unready_samples", np.nan)
    result.setdefault("aux_witness_ready_fraction", np.nan)
    result.setdefault("untrusted_aux_fraction", np.nan)
    result.setdefault("untrusted_aux_candidate_samples", np.nan)
    result.setdefault("mean_attribution_ms", np.nan)
    result.setdefault("median_attribution_ms", np.nan)
    result.setdefault("p95_attribution_ms", np.nan)
    result.setdefault("p99_attribution_ms", np.nan)
    for name in C4_ATTRIBUTION_STATE_NAMES:
        result.setdefault(f"attr_frac_{name.lower()}", np.nan)

    if scenario_config["type"] in {"plant", "reference"}:
        result["false_sensor_fault_entries"] = int(result.get("reliability_entries", 0) or 0)
        result["genuine_sensor_fault_entries"] = 0
        result["genuine_sensor_fault_entries_permitted"] = 0

    trace = result.get("trace")
    result["detection_latency_s"] = np.nan
    result["detected_event"] = np.nan
    result["sensor_fault_detected"] = np.nan
    if trace is not None and scenario_config["type"] in {"sensor", "combined"}:
        event_mask = event_mask_for_scenario(scenario_name)
        nonphysical = np.asarray(trace["source"]) != 0
        detected_indices = np.flatnonzero(nonphysical & event_mask)
        result["detected_event"] = bool(len(detected_indices))
        result["sensor_fault_detected"] = bool(len(detected_indices))
        if len(detected_indices):
            result["detection_latency_s"] = float(TIME[detected_indices[0]] - scenario_config["event_start"])

        # V3 entries are scored only after the trajectory.  The frozen V3
        # evaluator exposes aggregate entry count rather than entry timestamps;
        # the primary abrupt-fault gate uses actual source takeover and latency.
        result["genuine_sensor_fault_entries_permitted"] = np.nan


def _simulate_task(task: tuple[str, str, int, dict[str, Any]]) -> dict[str, Any]:
    controller, scenario, seed, attribution_params = task
    if controller == "C4_attribution_MPC":
        return simulate_c4_run((scenario, seed, attribution_params, True))
    record_trace = controller == "C3_arbitration_MPC"
    result = simulate_v3_run((controller, scenario, seed, record_trace))
    _baseline_detection_fields(result)
    return result


def contiguous_attribution_episodes(steps: list[dict[str, Any]], scenario: str, seed: int) -> list[dict[str, Any]]:
    if not steps:
        return []
    episodes: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(steps) + 1):
        if index < len(steps) and steps[index]["state"] == steps[start]["state"]:
            continue
        block = steps[start:index]
        source_codes = np.array([row["source_code"] for row in block], dtype=int)
        modal_source = int(np.bincount(source_codes, minlength=4).argmax())
        candidate_states = [str(row["candidate_state"]) for row in block]
        modal_candidate = max(dict.fromkeys(candidate_states), key=candidate_states.count)
        trust_reasons = [str(row["auxiliary_trust_reason"]) for row in block]
        modal_trust_reason = max(dict.fromkeys(trust_reasons), key=trust_reasons.count)
        row = {
            "scenario": scenario,
            "seed": seed,
            "start_step": int(block[0]["step"]),
            "end_step": int(block[-1]["step"]),
            "start_time_s": float(block[0]["time_s"]),
            "end_time_s": float(block[-1]["time_s"]),
            "duration_s": float(len(block) * DT),
            "attribution_state": block[0]["state"],
            "candidate_state": modal_candidate,
            "d_sm": float(np.mean([r["d_sm"] for r in block])),
            "d_sa": float(np.mean([r["d_sa"] for r in block])),
            "d_ma": float(np.mean([r["d_ma"] for r in block])),
            "filtered_d_sm": float(np.nanmean([r["filtered_d_sm"] for r in block])),
            "filtered_d_sa": float(np.nanmean([r["filtered_d_sa"] for r in block])),
            "filtered_d_ma": float(np.nanmean([r["filtered_d_ma"] for r in block])),
            "c3_requested_entry": bool(any(r["c3_requested_entry"] for r in block)),
            "c3_candidate_samples": int(sum(r["c3_candidate"] for r in block)),
            "c4_suppressed_entry": bool(any(r["c4_suppressed_entry"] for r in block)),
            "suppressed_candidate_samples": int(sum(r["c4_suppressed_entry"] for r in block)),
            "auxiliary_trusted_fraction": float(np.mean([r["auxiliary_trusted"] for r in block])),
            "auxiliary_trust_reason": modal_trust_reason,
            "selected_source": str(SOURCE_NAMES[modal_source]),
            "selected_source_code": modal_source,
            "attribution_ms_mean": float(np.mean([r["attribution_ms"] for r in block])),
        }
        episodes.append(row)
        start = index
    return episodes


def flatten_trace(trace: dict[str, Any], scenario: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, time_s in enumerate(trace["time"]):
        attr_code = int(trace["attr_state"][index])
        rows.append(
            {
                "controller": "C4_attribution_MPC",
                "scenario": scenario,
                "seed": seed,
                "step": index,
                "time_s": float(time_s),
                "reference": float(trace["ref"][index]),
                "true_speed": float(trace["true"][index]),
                "measured_speed": float(trace["meas"][index]),
                "main_speed": float(trace["main"][index]),
                "aux_speed": float(trace["aux"][index]),
                "feedback_speed": float(trace["fb"][index]),
                "source_code": int(trace["source"][index]),
                "source": str(SOURCE_NAMES[int(trace["source"][index])]),
                "voltage": float(trace["voltage"][index]),
                "substitute": bool(trace["sub"][index]),
                "sensor_latched": bool(trace["sensor_state"][index]),
                "c3_proposed_sensor_latched": bool(trace["proposed_sensor_state"][index]),
                "c3_candidate": bool(trace["c3_candidate"][index]),
                "new_c3_entry_candidate": bool(trace["new_candidate"][index]),
                "c4_suppressed_entry": bool(trace["suppressed"][index]),
                "auxiliary_trusted": bool(trace["aux_trusted"][index]),
                "auxiliary_hard_valid": bool(trace["aux_hard_valid"][index]),
                "auxiliary_trust_reason": str(trace["aux_witness_reason"][index]),
                "aux_param_mismatch": bool(trace["aux_param_mismatch"][index]),
                "attribution_state_code": attr_code,
                "attribution_state": str(C4_ATTRIBUTION_STATE_NAMES[attr_code]) if attr_code >= 0 else "WARMUP",
                "raw_attribution_state_code": int(trace["raw_attr_state"][index]),
                "d_sm": float(trace["d_sm"][index]),
                "d_sa": float(trace["d_sa"][index]),
                "d_ma": float(trace["d_ma"][index]),
                "filtered_d_sm": float(trace["filtered_d_sm"][index]),
                "filtered_d_sa": float(trace["filtered_d_sa"][index]),
                "filtered_d_ma": float(trace["filtered_d_ma"][index]),
                "suppression_evidence_action_code": int(trace["suppression_action"][index]),
            }
        )
    return rows


def mechanism_plot(trace: dict[str, Any], scenario: str, seed: int, attribution_params: dict[str, Any], save_path: Path) -> None:
    time_s = np.asarray(trace["time"])
    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    axes[0].plot(time_s, trace["ref"], "k--", label="Reference", linewidth=1.4)
    axes[0].plot(time_s, trace["true"], label="True speed", linewidth=1.8)
    axes[0].plot(time_s, trace["meas"], label="Measured speed", alpha=0.75)
    axes[0].plot(time_s, trace["main"], label="Main virtual", alpha=0.8)
    axes[0].plot(time_s, trace["aux"], label="Aux virtual", alpha=0.8)
    axes[0].plot(time_s, trace["fb"], label="Selected feedback", linewidth=1.8)
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title(f"C4 three-way consistency mechanism: {scenario}, seed {seed}")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_s, trace["d_sm"], label="raw d_sm", linewidth=1.25)
    axes[1].plot(time_s, trace["d_sa"], label="raw d_sa", linewidth=1.25)
    axes[1].plot(time_s, trace["d_ma"], label="raw d_ma", linewidth=1.25)
    axes[1].plot(time_s, trace["filtered_d_sm"], label="diagnostic EWMA d_sm", linewidth=0.8, alpha=0.35)
    axes[1].plot(time_s, trace["filtered_d_sa"], label="diagnostic EWMA d_sa", linewidth=0.8, alpha=0.35)
    axes[1].plot(time_s, trace["filtered_d_ma"], label="diagnostic EWMA d_ma", linewidth=0.8, alpha=0.35)
    agreement = attribution_params["agreement_thresholds"]
    disagreement = attribution_params["disagreement_thresholds"]
    for pair_name, agree, disagree in zip(("sm", "sa", "ma"), agreement, disagreement):
        axes[1].axhline(agree, linestyle=":", linewidth=0.8, alpha=0.65, label=f"{pair_name} p75 agree")
        axes[1].axhline(disagree, linestyle="--", linewidth=0.8, alpha=0.65, label=f"{pair_name} p99.9 disagree")
    axes[1].set_ylabel("Pairwise disagreement")
    axes[1].legend(ncol=4, fontsize=7)
    axes[1].grid(True, alpha=0.25)

    axes[2].step(time_s, trace["attr_state"], where="post", label="Attribution state")
    axes[2].step(time_s, np.asarray(trace["c3_candidate"], dtype=int) * 5, where="post", label="C3 candidate ×5", alpha=0.65)
    axes[2].step(time_s, np.asarray(trace["suppressed"], dtype=int) * 6, where="post", label="C4 suppression ×6", alpha=0.65)
    axes[2].step(time_s, np.asarray(trace["sub"], dtype=int) * 7, where="post", label="Actual substitution ×7", alpha=0.65)
    axes[2].set_yticks(range(len(C4_ATTRIBUTION_STATE_NAMES)))
    axes[2].set_yticklabels(C4_ATTRIBUTION_STATE_NAMES, fontsize=8)
    axes[2].set_ylabel("Attribution / decisions")
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].grid(True, alpha=0.25)

    axes[3].step(time_s, trace["source"], where="post")
    axes[3].set_yticks([0, 1, 2, 3])
    axes[3].set_yticklabels(SOURCE_NAMES, fontsize=8)
    axes[3].set_ylabel("Feedback source")
    axes[3].grid(True, alpha=0.25)

    axes[4].step(time_s, trace["voltage"], where="post")
    axes[4].set_ylabel("Voltage (V)")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(True, alpha=0.25)

    config = SCENARIOS[scenario]
    if config["event_start"] is not None:
        for axis in axes:
            axis.axvline(config["event_start"], linestyle=":", linewidth=1)
            if config["event_end"] is not None:
                axis.axvline(config["event_end"], linestyle=":", linewidth=1)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_rmse",
        "overall_mae",
        "fault_window_rmse",
        "sub_fraction",
        "reliability_entries",
        "switches",
        "frac_phys",
        "frac_main",
        "frac_aux",
        "frac_fb",
        "control_effort_u2",
        "control_variation_du2",
        "optimizer_failures",
        "voltage_violations",
        "rate_violations",
        "nonfinite_events",
        "false_sensor_fault_entries",
        "suppressed_entry_count",
        "suppressed_false_entries",
        "genuine_sensor_fault_entries_permitted",
        "genuine_sensor_fault_entries_suppressed",
        "auxiliary_misuse_events",
        "aux_witness_ready_fraction",
        "untrusted_aux_fraction",
        "attribution_switches",
        "mean_attribution_dwell_s",
        "mean_attribution_ms",
        "median_attribution_ms",
        "p95_attribution_ms",
        "p99_attribution_ms",
    ] + [f"attr_frac_{name.lower()}" for name in C4_ATTRIBUTION_STATE_NAMES]
    grouped = runs.groupby(["scenario", "controller"], sort=True)[metrics].agg(["mean", "std", "median"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def relative_reduction(old: float, new: float) -> float:
    if abs(old) <= 1e-15:
        return 1.0 if abs(new) <= 1e-15 else -np.inf
    return float((old - new) / old)


LATENCY_POLICY = (
    "For each abrupt sensor-fault scenario, the latency criterion is evaluable only when all five paired C3 "
    "and all five paired C4 runs have finite detection_latency_s values. Any missing/nonfinite latency makes "
    "that scenario's latency criterion fail; detection-rate preservation is evaluated separately."
)


def _complete_mean_latency(runs: pd.DataFrame, scenario: str, controller: str) -> tuple[float, bool]:
    values = pd.to_numeric(
        runs[(runs.scenario == scenario) & (runs.controller == controller)]["detection_latency_s"],
        errors="coerce",
    )
    complete = bool(len(values) == len(DEVELOPMENT_SEEDS) and values.notna().all() and np.isfinite(values.to_numpy(float)).all())
    return (float(values.mean()) if complete else np.nan), complete


def assess_development_gate(runs: pd.DataFrame, criteria: dict[str, Any]) -> dict[str, Any]:
    rules = criteria["criteria"]

    def mean(scenario: str, controller: str, field: str) -> float:
        values = runs[(runs.scenario == scenario) & (runs.controller == controller)][field]
        return float(values.mean())

    load_rule = rules["load_disturbance"]
    c3_load_entries = mean("load_disturbance", "C3_arbitration_MPC", "false_sensor_fault_entries")
    c4_load_entries = mean("load_disturbance", "C4_attribution_MPC", "false_sensor_fault_entries")
    c3_load_sub = mean("load_disturbance", "C3_arbitration_MPC", "sub_fraction")
    c4_load_sub = mean("load_disturbance", "C4_attribution_MPC", "sub_fraction")
    plain_load_rmse = mean("load_disturbance", "B_plain_MPC", "fault_window_rmse")
    c3_load_rmse = mean("load_disturbance", "C3_arbitration_MPC", "fault_window_rmse")
    c4_load_rmse = mean("load_disturbance", "C4_attribution_MPC", "fault_window_rmse")
    penalty = max(0.0, c3_load_rmse - plain_load_rmse)
    penalty_removed = 1.0 if penalty == 0 and c4_load_rmse <= c3_load_rmse else ((c3_load_rmse - c4_load_rmse) / penalty if penalty > 0 else -np.inf)
    load_pivot = runs[runs.scenario == "load_disturbance"].pivot(index="seed", columns="controller", values="false_sensor_fault_entries")
    no_new_seed_entry = bool(((load_pivot["C3_arbitration_MPC"] > 0) | (load_pivot["C4_attribution_MPC"] == 0)).all())

    checks: dict[str, Any] = {
        "load_false_entry_relative_reduction": relative_reduction(c3_load_entries, c4_load_entries),
        "load_false_entry_pass": relative_reduction(c3_load_entries, c4_load_entries) >= load_rule["false_sensor_fault_entries_relative_reduction_min"],
        "load_substitution_relative_reduction": relative_reduction(c3_load_sub, c4_load_sub),
        "load_substitution_pass": relative_reduction(c3_load_sub, c4_load_sub) >= load_rule["substitution_fraction_relative_reduction_min"],
        "load_tracking_penalty_fraction_removed": float(penalty_removed),
        "load_tracking_pass": penalty_removed >= load_rule["c3_tracking_penalty_toward_plain_fraction_removed_min"],
        "load_no_new_false_entry_on_clean_c3_seed": no_new_seed_entry,
        "load_no_new_false_entry_on_clean_c3_seed_pass": no_new_seed_entry,
    }

    def preservation_check(scenario: str, max_degradation: float, latency_limit: float | None = None) -> dict[str, Any]:
        c3_rmse = mean(scenario, "C3_arbitration_MPC", "fault_window_rmse")
        c4_rmse = mean(scenario, "C4_attribution_MPC", "fault_window_rmse")
        degradation = (c4_rmse - c3_rmse) / c3_rmse if c3_rmse > 0 else np.inf
        c3_detection = mean(scenario, "C3_arbitration_MPC", "detected_event")
        c4_detection = mean(scenario, "C4_attribution_MPC", "detected_event")
        payload = {
            "rmse_relative_degradation": float(degradation),
            "rmse_pass": bool(degradation <= max_degradation),
            "c3_detection_rate": c3_detection,
            "c4_detection_rate": c4_detection,
            "detection_rate_pass": bool(c4_detection + 1e-12 >= c3_detection),
        }
        if latency_limit is not None:
            c3_latency, c3_latency_complete = _complete_mean_latency(runs, scenario, "C3_arbitration_MPC")
            c4_latency, c4_latency_complete = _complete_mean_latency(runs, scenario, "C4_attribution_MPC")
            latency_complete = bool(c3_latency_complete and c4_latency_complete)
            increase = c4_latency - c3_latency if latency_complete else np.nan
            payload.update(
                {
                    "latency_policy": LATENCY_POLICY,
                    "latency_complete": latency_complete,
                    "mean_detection_latency_increase_s": float(increase),
                    "latency_pass": bool(latency_complete and np.isfinite(increase) and increase <= latency_limit),
                }
            )
        return payload

    combined_rule = rules["combined_fault_load"]
    checks["combined_fault_load"] = preservation_check(
        "combined_fault_load", combined_rule["mean_fault_window_rmse_relative_degradation_max"]
    )
    checks["combined_fault_load"]["genuine_fault_suppression_count"] = int(
        runs[(runs.scenario == "combined_fault_load") & (runs.controller == "C4_attribution_MPC")]["genuine_sensor_fault_suppressed"].sum()
    )
    checks["combined_fault_load"]["genuine_fault_suppression_pass"] = checks["combined_fault_load"]["genuine_fault_suppression_count"] == 0

    abrupt_rule = rules["abrupt_sensor_faults"]
    checks["abrupt_sensor_faults"] = {
        scenario: preservation_check(
            scenario,
            abrupt_rule["mean_fault_window_rmse_relative_degradation_max"],
            abrupt_rule["mean_detection_latency_increase_s_max"],
        )
        for scenario in abrupt_rule["scenarios"]
    }
    for scenario, payload in checks["abrupt_sensor_faults"].items():
        payload["genuine_fault_suppression_count"] = int(
            runs[(runs.scenario == scenario) & (runs.controller == "C4_attribution_MPC")]["genuine_sensor_fault_suppressed"].sum()
        )
        payload["genuine_fault_suppression_pass"] = payload["genuine_fault_suppression_count"] == 0

    parameter_rule = rules["parameter_variation"]
    plain_parameter = mean("parameter_variation", "B_plain_MPC", "fault_window_rmse")
    c3_parameter = mean("parameter_variation", "C3_arbitration_MPC", "fault_window_rmse")
    c4_parameter = mean("parameter_variation", "C4_attribution_MPC", "fault_window_rmse")
    comparator = min(plain_parameter, c3_parameter)
    parameter_degradation = (c4_parameter - comparator) / comparator if comparator > 0 else np.inf
    c3_parameter_aux = mean("parameter_variation", "C3_arbitration_MPC", "event_frac_aux")
    c4_parameter_aux = mean("parameter_variation", "C4_attribution_MPC", "event_frac_aux")
    c4_parameter_misuse = int(
        runs[(runs.scenario == "parameter_variation") & (runs.controller == "C4_attribution_MPC")]["auxiliary_misuse_events"].sum()
    )
    checks["parameter_variation"] = {
        "rmse_relative_degradation_vs_better_baseline": float(parameter_degradation),
        "rmse_pass": bool(parameter_degradation <= parameter_rule["mean_fault_window_rmse_relative_degradation_vs_better_of_plain_or_c3_max"]),
        "c3_event_aux_fraction": c3_parameter_aux,
        "c4_event_aux_fraction": c4_parameter_aux,
        "auxiliary_misuse_events": c4_parameter_misuse,
        "auxiliary_misuse_pass": c4_parameter_misuse == 0,
    }

    reference_rule = rules["reference_transients"]
    reference_checks: dict[str, Any] = {}
    for scenario in reference_rule["scenarios"]:
        c4_rows = runs[(runs.scenario == scenario) & (runs.controller == "C4_attribution_MPC")]
        c3_rmse = mean(scenario, "C3_arbitration_MPC", "overall_rmse")
        c4_rmse = mean(scenario, "C4_attribution_MPC", "overall_rmse")
        relative = (c4_rmse - c3_rmse) / c3_rmse if c3_rmse > 0 else np.inf
        reference_checks[scenario] = {
            "reliability_latches": int(c4_rows.reliability_entries.sum()),
            "latch_pass": int(c4_rows.reliability_entries.sum()) <= reference_rule["reliability_latches_max_total"],
            "mean_substitution_fraction": float(c4_rows.sub_fraction.mean()),
            "substitution_pass": float(c4_rows.sub_fraction.mean()) <= reference_rule["mean_substitution_fraction_max"],
            "overall_rmse_relative_degradation_vs_c3": float(relative),
            "tracking_pass": bool(relative <= reference_rule["mean_overall_rmse_relative_degradation_vs_c3_max"]),
        }
    checks["reference_transients"] = reference_checks

    c4 = runs[runs.controller == "C4_attribution_MPC"]
    safety_rule = rules["safety"]
    checks["safety"] = {
        "optimizer_failures": int(c4.optimizer_failures.sum()),
        "voltage_violations": int(c4.voltage_violations.sum()),
        "slew_violations": int(c4.rate_violations.sum()),
        "nonfinite_events": int(c4.nonfinite_events.sum()),
    }
    checks["safety"]["pass"] = bool(
        checks["safety"]["optimizer_failures"] == safety_rule["optimizer_failures"]
        and checks["safety"]["voltage_violations"] == safety_rule["voltage_violations"]
        and checks["safety"]["slew_violations"] == safety_rule["slew_violations"]
        and checks["safety"]["nonfinite_events"] == safety_rule["nonfinite_events"]
    )

    scalar_passes = [value for key, value in checks.items() if key.endswith("_pass")]
    combined_passes = [value for key, value in checks["combined_fault_load"].items() if key.endswith("_pass")]
    abrupt_passes = [value for payload in checks["abrupt_sensor_faults"].values() for key, value in payload.items() if key.endswith("_pass")]
    parameter_passes = [value for key, value in checks["parameter_variation"].items() if key.endswith("_pass")]
    reference_passes = [value for payload in checks["reference_transients"].values() for key, value in payload.items() if key.endswith("_pass")]
    all_pass = all(bool(value) for value in scalar_passes + combined_passes + abrupt_passes + parameter_passes + reference_passes) and checks["safety"]["pass"]
    return {"decision": "GO" if all_pass else "NO_GO", "latency_policy": LATENCY_POLICY, "checks": checks}


def load_and_validate_development_config() -> tuple[dict[str, Any], Path, dict[str, str]]:
    path = PROJECT / "results/configs/c4_development_config.json"
    if not path.is_file():
        raise RuntimeError("development mode requires results/configs/c4_development_config.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_first_completed_c4_development_evidence":
        raise RuntimeError("C4 development config has an unexpected freeze status")

    design = config.get("development_design")
    if not isinstance(design, dict):
        raise RuntimeError("C4 development config is missing development_design")
    config_seeds = _coerce_seed_list(design.get("seeds"), label="C4 development config")
    if config_seeds != DEVELOPMENT_SEEDS:
        raise RuntimeError("C4 development config seed list does not match the locked development seeds")
    if list(design.get("scenarios", [])) != DEVELOPMENT_SCENARIOS:
        raise RuntimeError("C4 development config scenario list does not match the locked development scenarios")
    if list(design.get("controllers", [])) != CONTROLLERS:
        raise RuntimeError("C4 development config controller list does not match the locked controller set")
    if int(design.get("expected_run_count", -1)) != len(CONTROLLERS) * len(DEVELOPMENT_SCENARIOS) * len(DEVELOPMENT_SEEDS):
        raise RuntimeError("C4 development config expected_run_count is not 105")

    criteria_path = PROJECT / "results/configs/c4_go_no_go_criteria.json"
    if config.get("predeclared_criteria") != str(criteria_path.relative_to(PROJECT)).replace("\\", "/"):
        raise RuntimeError("C4 development config does not reference the canonical predeclared criteria")
    calibration_path = PROJECT / "results/configs/c4_attribution_calibration.json"
    hashes = config.get("hashes")
    if not isinstance(hashes, dict):
        raise RuntimeError("C4 development config is missing its frozen hash block")
    required_hashes = {
        "c4_attribution_calibration_json": calibration_path,
        "c4_calibration_script": PROJECT / "scripts/calibrate_c4_attribution.py",
        "c4_go_no_go_criteria_json": criteria_path,
        "c4_evaluator_script": Path(__file__).resolve(),
        "c4_verifier_script": PROJECT / "scripts/verify_c4_results.py",
        "reliability_py": PROJECT / "src/reliability.py",
        "v3_main_model": PROJECT / "results/lstm_model_weights.pt",
        "v3_auxiliary_model": PROJECT / "results/auxiliary_model_weights.pt",
        "v3_arbitration_calibration": PROJECT / "results/configs/v3_arbitration_calibration.json",
        "v3_arbitration_config": PROJECT / "results/configs/v3_arbitration_config.json",
        "v3_final_275_run_matrix": PROJECT / "results/metrics/v3_final_11scenario_runs.csv",
    }
    for key, artifact in required_hashes.items():
        if hashes.get(key) != sha256(artifact):
            raise RuntimeError(f"C4 development config frozen hash mismatch for {key}")

    attribution_params = load_c4_attribution_parameters(calibration_path)
    frozen_attributor = config.get("attributor")
    if not isinstance(frozen_attributor, dict):
        raise RuntimeError("C4 development config is missing the frozen attributor block")
    expected_attributor = {
        "agreement_thresholds": tuple(float(value) for value in frozen_attributor.get("agreement_thresholds", [])),
        "disagreement_thresholds": tuple(float(value) for value in frozen_attributor.get("disagreement_thresholds", [])),
        "ewma_alpha": float(frozen_attributor.get("ewma_alpha", np.nan)),
        "enter_count": int(frozen_attributor.get("enter_count", -1)),
        "exit_count": int(frozen_attributor.get("exit_count", -1)),
    }
    if expected_attributor != attribution_params:
        raise RuntimeError("C4 development config attributor parameters do not match the frozen calibration")

    bindings = {
        "c4_development_config": sha256(path),
        "c4_attribution_calibration": sha256(calibration_path),
        "c4_go_no_go_criteria": sha256(criteria_path),
        **_current_c4_script_hashes(),
    }
    return config, path, bindings


def validate_saved_development_gate() -> dict[str, Any]:
    _, development_config_path, expected_bindings = load_and_validate_development_config()
    metrics_dir = PROJECT / "results/metrics"
    gate_path = metrics_dir / "c4_development_stage_gate.json"
    runs_path = metrics_dir / "c4_development_runs.csv"
    summary_path = metrics_dir / "c4_development_summary.json"
    criteria_path = PROJECT / "results/configs/c4_go_no_go_criteria.json"
    for required in (gate_path, runs_path, summary_path, criteria_path):
        if not required.is_file():
            raise RuntimeError(f"final mode requires saved development evidence: {required.relative_to(PROJECT)}")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("decision") != "GO":
        raise RuntimeError("final mode requires actual saved c4_development_stage_gate.json decision GO")
    gate_bindings = gate.get("bindings")
    if not isinstance(gate_bindings, dict) or any(gate_bindings.get(key) != value for key, value in expected_bindings.items()):
        raise RuntimeError("saved C4 development stage gate is not bound to the current development config/evaluator/verifier")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mode") != "development" or summary.get("evidence_role") != "development_evidence":
        raise RuntimeError("saved C4 development summary has an invalid evidence role")
    summary_hashes = summary.get("hashes")
    if not isinstance(summary_hashes, dict) or any(summary_hashes.get(key) != value for key, value in expected_bindings.items()):
        raise RuntimeError("saved C4 development summary is not bound to the current development config/evaluator/verifier")

    runs = pd.read_csv(runs_path, float_precision="round_trip")
    required_columns = {"controller", "scenario", "seed"}
    if not required_columns <= set(runs.columns):
        raise RuntimeError("saved C4 development run matrix is missing paired-design key columns")
    runs = runs.copy()
    seed_values = pd.to_numeric(runs["seed"], errors="coerce")
    if (
        seed_values.isna().any()
        or not np.isfinite(seed_values.to_numpy(float)).all()
        or not np.allclose(seed_values.to_numpy(float), np.round(seed_values.to_numpy(float)), rtol=0, atol=0)
    ):
        raise RuntimeError("saved C4 development run matrix contains non-integer seeds")
    runs["seed"] = seed_values.astype(int)
    expected_keys = {
        (controller, scenario, seed)
        for controller in CONTROLLERS
        for scenario in DEVELOPMENT_SCENARIOS
        for seed in DEVELOPMENT_SEEDS
    }
    actual_keys = set(map(tuple, runs[["controller", "scenario", "seed"]].itertuples(index=False, name=None)))
    if len(runs) != len(expected_keys) or runs[["controller", "scenario", "seed"]].duplicated().any() or actual_keys != expected_keys:
        raise RuntimeError("saved C4 development run matrix does not match the locked 105-run paired design")
    criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
    recomputed = assess_development_gate(runs, criteria)
    if recomputed["decision"] != "GO" or recomputed["decision"] != gate.get("decision"):
        raise RuntimeError("saved C4 development GO does not reproduce from the saved 105-run evidence")
    summary_gate = summary.get("development_stage_gate")
    if not isinstance(summary_gate, dict) or summary_gate.get("decision") != "GO":
        raise RuntimeError("saved C4 development summary does not record the GO stage gate")

    return {
        "stage_gate_sha256": sha256(gate_path),
        "development_config_sha256": sha256(development_config_path),
        "development_runs_sha256": sha256(runs_path),
        "development_summary_sha256": sha256(summary_path),
        **expected_bindings,
    }


def parse_final_plan(path: Path, calibration_path: Path) -> tuple[list[int], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    development_evidence = validate_saved_development_gate()
    frozen_bindings = _final_config_hash_mapping(config)
    stage_gate_binding = _hash_binding(
        frozen_bindings,
        "c4_development_stage_gate",
        "c4_development_stage_gate_sha256",
        "development_stage_gate_sha256",
    )
    development_config_binding = _hash_binding(
        frozen_bindings,
        "c4_development_config",
        "c4_development_config_sha256",
        "development_config_sha256",
    )
    if (
        stage_gate_binding != development_evidence["stage_gate_sha256"]
        or development_config_binding != development_evidence["development_config_sha256"]
    ):
        raise RuntimeError("final C4 config must bind the exact saved GO stage gate and development config")

    _validate_final_attributor_contract(config, calibration_path)

    raw_seeds = config.get("final_holdout_seeds") or config.get("fresh_final_seeds") or config.get("seeds")
    seeds = _coerce_seed_list(raw_seeds, label="final C4 config")
    forbidden = set(DEVELOPMENT_SEEDS) | set(V3_FINAL_SEEDS)
    if forbidden.intersection(seeds):
        raise RuntimeError("final C4 seeds overlap development or frozen V3 final seeds")
    configured_scenarios = config.get("scenarios")
    if not isinstance(configured_scenarios, list) or configured_scenarios != FINAL_SCENARIOS:
        raise RuntimeError("final C4 config must explicitly declare the exact frozen 11-scenario list")
    _validate_preuse_seed_provenance(path.resolve(), config, seeds)
    return seeds, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "final"), default="development")
    parser.add_argument("--final-config", type=Path, help="Required only for --mode final; must contain a GO stage gate and five fresh seeds.")
    parser.add_argument("--workers", type=int, default=min(int(os.environ.get("C4_WORKERS", "8")), os.cpu_count() or 1))
    parser.add_argument("--calibration", type=Path, default=PROJECT / "results/configs/c4_attribution_calibration.json")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT / "results/metrics")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT / "results/figures")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.mode == "final":
        if args.final_config is None:
            parser.error("--mode final requires --final-config")
        seeds, final_config = parse_final_plan(args.final_config, args.calibration)
        scenarios = FINAL_SCENARIOS
        development_config_path = None
        development_bindings = None
    else:
        if args.final_config is not None:
            parser.error("--final-config is only valid with --mode final")
        _, development_config_path, development_bindings = load_and_validate_development_config()
        seeds = DEVELOPMENT_SEEDS
        scenarios = DEVELOPMENT_SCENARIOS
        final_config = None

    attribution_params = load_c4_attribution_parameters(args.calibration, final_config)
    tasks = [
        (controller, scenario, seed, attribution_params)
        for scenario in scenarios
        for seed in seeds
        for controller in CONTROLLERS
    ]
    expected_count = len(CONTROLLERS) * len(scenarios) * len(seeds)
    print(f"C4 {args.mode}: {expected_count} paired runs using {args.workers} workers")
    started = perf_counter()
    if args.workers == 1:
        results = [_simulate_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_simulate_task, tasks))
    wall_time = perf_counter() - started

    attribution_events: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    representative_traces: dict[str, dict[str, Any]] = {}
    for result in results:
        timing_samples = result.pop("timing_samples", [])
        if result["controller"] == "C4_attribution_MPC":
            timing_rows.extend(timing_samples)
        steps = result.pop("attribution_steps", [])
        if result["controller"] == "C4_attribution_MPC":
            attribution_events.extend(contiguous_attribution_episodes(steps, result["scenario"], int(result["seed"])))
        result.pop("reference_substitution_events", None)
        result.pop("recovery_diagnostics", None)
        trace = result.pop("trace", None)
        if result["controller"] == "C4_attribution_MPC" and trace is not None:
            trace_rows.extend(flatten_trace(trace, result["scenario"], int(result["seed"])))
            if int(result["seed"]) == seeds[0] and result["scenario"] in {"load_disturbance", "combined_fault_load"}:
                representative_traces[result["scenario"]] = trace
        result.pop("actual_entry_times", None)

    runs = pd.DataFrame(results).sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)
    if len(runs) != expected_count or runs[["controller", "scenario", "seed"]].duplicated().any():
        raise RuntimeError("C4 run matrix is incomplete or contains duplicate controller/scenario/seed keys")
    expected_keys = {(controller, scenario, seed) for controller in CONTROLLERS for scenario in scenarios for seed in seeds}
    actual_keys = set(map(tuple, runs[["controller", "scenario", "seed"]].itertuples(index=False, name=None)))
    if actual_keys != expected_keys:
        raise RuntimeError("C4 run matrix keys do not match the requested paired design")

    summary_frame = aggregate_runs(runs)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    prefix = "c4_development" if args.mode == "development" else "c4_final_holdout"
    runs_path = args.metrics_dir / f"{prefix}_runs.csv"
    summary_path = args.metrics_dir / f"{prefix}_summary.csv"
    events_path = args.metrics_dir / f"{prefix}_attribution_events.csv"
    trace_path = args.metrics_dir / f"{prefix}_trace.csv"
    timing_path = args.metrics_dir / f"{prefix}_timing.csv"
    runs.to_csv(runs_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    events_frame = pd.DataFrame(attribution_events)
    timing_frame = pd.DataFrame(timing_rows)
    events_frame.to_csv(events_path, index=False)
    pd.DataFrame(trace_rows).to_csv(trace_path, index=False)
    timing_frame.to_csv(timing_path, index=False)

    gate_result = None
    if args.mode == "development":
        criteria_path = PROJECT / "results/configs/c4_go_no_go_criteria.json"
        criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
        gate_result = assess_development_gate(runs, criteria)
        gate_result["bindings"] = dict(development_bindings or {})
        (args.metrics_dir / "c4_development_stage_gate.json").write_text(
            json.dumps(json_ready(gate_result), indent=2, allow_nan=False), encoding="utf-8"
        )

    plot_paths: list[str] = []
    if not args.no_plots:
        for scenario in ("load_disturbance", "combined_fault_load"):
            trace = representative_traces.get(scenario)
            if trace is None:
                continue
            plot_path = args.figures_dir / f"{prefix}_{scenario}_timeline.png"
            mechanism_plot(trace, scenario, seeds[0], attribution_params, plot_path)
            plot_paths.append(str(plot_path.relative_to(PROJECT)))

    raw_attribution_ms = timing_frame["attribution_ms"].to_numpy(float) if len(timing_frame) else np.array([], dtype=float)
    attribution_runtime = {
        "sample_count": int(len(raw_attribution_ms)),
        "mean_ms": float(np.mean(raw_attribution_ms)) if len(raw_attribution_ms) else 0.0,
        "median_ms": float(np.median(raw_attribution_ms)) if len(raw_attribution_ms) else 0.0,
        "p95_ms": float(np.percentile(raw_attribution_ms, 95.0)) if len(raw_attribution_ms) else 0.0,
        "p99_ms": float(np.percentile(raw_attribution_ms, 99.0)) if len(raw_attribution_ms) else 0.0,
    }
    hash_payload = {
        "c4_attribution_calibration": sha256(args.calibration),
        "v3_arbitration_config": sha256(PROJECT / "results/configs/v3_arbitration_config.json"),
        "v3_arbitration_calibration": sha256(PROJECT / "results/configs/v3_arbitration_calibration.json"),
        "main_model": sha256(PROJECT / "results/lstm_model_weights.pt"),
        "auxiliary_model": sha256(PROJECT / "results/auxiliary_model_weights.pt"),
        **_current_c4_script_hashes(),
    }
    if args.mode == "development" and development_config_path is not None:
        hash_payload.update(development_bindings or {})
    if args.mode == "final" and args.final_config is not None:
        hash_payload["c4_frozen_config"] = sha256(args.final_config)
    summary_json = {
        "mode": args.mode,
        "evidence_role": "development_evidence" if args.mode == "development" else "fresh_c4_final_holdout",
        "controllers": CONTROLLERS,
        "scenarios": scenarios,
        "seeds": seeds,
        "run_count": int(len(runs)),
        "paired_design": "identical scenario, seed, base-noise and injected-fault realization across B, C3 and C4",
        "latency_policy": LATENCY_POLICY,
        "attribution_parameters": attribution_params,
        "attribution_runtime": attribution_runtime,
        "development_stage_gate": gate_result,
        "wall_time_s": wall_time,
        "artifacts": {
            "runs": str(runs_path.relative_to(PROJECT)),
            "summary_csv": str(summary_path.relative_to(PROJECT)),
            "attribution_events": str(events_path.relative_to(PROJECT)),
            "trace": str(trace_path.relative_to(PROJECT)),
            "timing": str(timing_path.relative_to(PROJECT)),
            "plots": plot_paths,
        },
        "hashes": hash_payload,
        "summary": summary_frame.to_dict(orient="records"),
    }
    json_path = args.metrics_dir / f"{prefix}_summary.json"
    json_path.write_text(json.dumps(json_ready(summary_json), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(runs)} runs to {runs_path}; wall time {wall_time:.2f}s")
    if gate_result is not None:
        print(f"Development stage gate: {gate_result['decision']}")


if __name__ == "__main__":
    main()
