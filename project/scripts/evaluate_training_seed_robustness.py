"""Evaluate B/C3 closed-loop robustness across matched training-seed pairs.

This study runner preserves the simulation semantics of
``scripts/evaluate_v3_closed_loop.py`` while varying only the matched main/aux
training pair and that pair's prospectively recalibrated sensor/V3 thresholds.

The script consumes the study-wide artifact manifest written by
``scripts/run_training_seed_models.py`` at
``results/training_seed_robustness/model_pair_manifest.json``.  Its binding
schema is::

    {
      "study": "training_seed_robustness",
      "protocol_manifest": {"path": "...", "sha256": "..."},
      "validation_run_ids": [3, 9, 11, 16, 19, 23],
      "algorithm_hashes": {...},
      "pairs": {
        "P2026": {
          "training_seed": 2026,
          "main": {
            "weights_path": "...", "config_path": "...",
            "weights_sha256": "...", "config_sha256": "..."
          },
          "auxiliary": {
            "weights_path": "...", "config_path": "...",
            "weights_sha256": "...", "config_sha256": "..."
          },
          "sensor_calibration": {"path": "...", "sha256": "..."},
          "v3_calibration": {"path": "...", "sha256": "..."},
          "pair_config": {"path": "...", "sha256": "..."}
        }
      }
    }

All artifact paths are project-relative.  Every declared hash is rechecked and
every seed-bearing artifact is asserted to belong to the requested training
pair before the first closed-loop task is launched.

Primary output:
    results/training_seed_robustness/training_seed_runs.csv  (exactly 150 rows)

Raw diagnostics, when present, are written only inside the same study
namespace.  This module does not run anything on import.
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

import evaluate_v3_closed_loop as frozen_v3  # noqa: E402
from auxiliary_sensor_model import AuxiliarySpeedEstimator, auxiliary_predict_online  # noqa: E402
from mpc import LSTMMPC, MPCConfig, PIConfig, PIController  # noqa: E402
from reliability import DualVirtualSensorArbitrator, SensorReliabilityMonitor, load_lstm_model  # noqa: E402


TRAINING_SEEDS = (2026, 2027, 2028)
EXPECTED_SIMULATION_SEEDS = (49026, 49027, 49028, 49029, 49030)
RESERVED_SIMULATION_SEEDS = (39026, 39027, 39028, 39029, 39030)
CONTROLLERS = ("B_plain_MPC", "C3_arbitration_MPC")
SCENARIOS = (
    "sensor_bias_5",
    "sensor_dropout",
    "load_disturbance",
    "combined_fault_load",
    "parameter_variation",
)
EXPECTED_RUN_COUNT = len(TRAINING_SEEDS) * len(EXPECTED_SIMULATION_SEEDS) * len(CONTROLLERS) * len(SCENARIOS)

RESULT_DIR = PROJECT / "results" / "training_seed_robustness"
MODEL_PAIR_MANIFEST_PATH = RESULT_DIR / "model_pair_manifest.json"
RUNS_PATH = RESULT_DIR / "training_seed_runs.csv"
TIMING_PATH = RESULT_DIR / "training_seed_timing.csv"
RELIABILITY_EVENTS_PATH = RESULT_DIR / "training_seed_reliability_events.csv"
RUN_MANIFEST_PATH = RESULT_DIR / "training_seed_evaluation_manifest.json"
SIMULATION_SEED_FREEZE_PATH = RESULT_DIR / "simulation_seed_freeze.json"

DT = frozen_v3.DT
CONTROL_STRIDE = frozen_v3.CONTROL_STRIDE
CONTROL_DT = frozen_v3.CONTROL_DT
DURATION = frozen_v3.DURATION
TIME = frozen_v3.TIME
FULL_SCALE = frozen_v3.FULL_SCALE
SPEED_NOISE_STD = frozen_v3.SPEED_NOISE_STD


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: str
    training_seed: int | None = None


@dataclass(frozen=True)
class PairBundle:
    training_seed: int
    manifest_path: Path
    manifest_sha256: str
    main_model: ArtifactRef
    main_config: ArtifactRef
    aux_model: ArtifactRef
    aux_config: ArtifactRef
    sensor_calibration: ArtifactRef
    v3_calibration: ArtifactRef
    pair_config: ArtifactRef
    protocol_manifest_path: Path
    protocol_manifest_sha256: str
    algorithm_hashes: dict[str, Any]
    sensor_calibration_algorithm_sha256: str
    v3_calibration_algorithm_sha256: str
    calibration_algorithm_sha256: str

    def to_worker_payload(self) -> dict[str, Any]:
        def artifact(ref: ArtifactRef) -> dict[str, Any]:
            return {
                "path": str(ref.path),
                "sha256": ref.sha256,
                "training_seed": ref.training_seed,
            }

        return {
            "training_seed": self.training_seed,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "main_model": artifact(self.main_model),
            "main_config": artifact(self.main_config),
            "aux_model": artifact(self.aux_model),
            "aux_config": artifact(self.aux_config),
            "sensor_calibration": artifact(self.sensor_calibration),
            "v3_calibration": artifact(self.v3_calibration),
            "pair_config": artifact(self.pair_config),
            "protocol_manifest_path": str(self.protocol_manifest_path),
            "protocol_manifest_sha256": self.protocol_manifest_sha256,
            "algorithm_hashes": self.algorithm_hashes,
            "sensor_calibration_algorithm_sha256": self.sensor_calibration_algorithm_sha256,
            "v3_calibration_algorithm_sha256": self.v3_calibration_algorithm_sha256,
            "calibration_algorithm_sha256": self.calibration_algorithm_sha256,
        }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT.resolve())).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{project_relative(path)} must contain a JSON object")
    return value


def _resolve_project_path(raw_path: str, *, label: str) -> Path:
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else PROJECT / candidate
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the project root: {raw_path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {raw_path}")
    return resolved


def _artifact_ref(entry: dict[str, Any], path_key: str, hash_key: str, label: str) -> ArtifactRef:
    raw_path = entry.get("path")
    if path_key != "path":
        raw_path = entry.get(path_key)
    declared_hash = entry.get(hash_key)
    if not isinstance(raw_path, str) or not raw_path:
        raise TypeError(f"{label}.{path_key} must be a non-empty string")
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        raise TypeError(f"{label}.{hash_key} must be a SHA-256 hex digest")
    path = _resolve_project_path(raw_path, label=label)
    actual_hash = sha256(path)
    if actual_hash != declared_hash.lower():
        raise RuntimeError(
            f"hash mismatch for {label}: declared {declared_hash}, actual {actual_hash}"
        )
    return ArtifactRef(path=path, sha256=actual_hash)


def _payload_seed(payload: dict[str, Any]) -> int | None:
    for key in ("training_seed", "seed"):
        if key in payload:
            return int(payload[key])
    training = payload.get("training")
    if isinstance(training, dict) and "seed" in training:
        return int(training["seed"])
    return None


def _assert_payload_seed(
    payload: dict[str, Any],
    training_seed: int,
    label: str,
    *,
    require: bool,
) -> None:
    seed_value = _payload_seed(payload)
    if seed_value is None:
        if require:
            raise RuntimeError(f"{label} does not declare its training seed")
        return
    if seed_value != training_seed:
        raise RuntimeError(f"{label} belongs to training seed {seed_value}, expected {training_seed}")


def _assert_model_hash_bindings(
    calibration: dict[str, Any],
    main_hash: str,
    main_config_hash: str,
    aux_hash: str,
    aux_config_hash: str,
    *,
    label: str,
) -> None:
    bindings = calibration.get("model_sha256")
    if not isinstance(bindings, dict):
        return
    main_declared = (
        bindings.get("main_weights")
        or bindings.get("main")
        or bindings.get("main_model")
    )
    aux_declared = (
        bindings.get("auxiliary_weights")
        or bindings.get("auxiliary")
        or bindings.get("aux")
        or bindings.get("aux_model")
    )
    if main_declared is not None and main_declared != main_hash:
        raise RuntimeError(f"{label} main-model hash does not match its training pair")
    if aux_declared is not None and aux_declared != aux_hash:
        raise RuntimeError(f"{label} auxiliary-model hash does not match its training pair")
    main_config_declared = bindings.get("main_config")
    aux_config_declared = bindings.get("auxiliary_config") or bindings.get("aux_config")
    if main_config_declared is not None and main_config_declared != main_config_hash:
        raise RuntimeError(f"{label} main-config hash does not match its training pair")
    if aux_config_declared is not None and aux_config_declared != aux_config_hash:
        raise RuntimeError(f"{label} auxiliary-config hash does not match its training pair")


def _assert_pair_config_bindings(
    pair_config: dict[str, Any],
    main_model: ArtifactRef,
    main_config: ArtifactRef,
    aux_model: ArtifactRef,
    aux_config: ArtifactRef,
    sensor_calibration: ArtifactRef,
    v3_calibration: ArtifactRef,
) -> None:
    expected = {
        "main_model": {
            "weights_path": project_relative(main_model.path),
            "config_path": project_relative(main_config.path),
            "weights_sha256": main_model.sha256,
            "config_sha256": main_config.sha256,
        },
        "auxiliary_model": {
            "weights_path": project_relative(aux_model.path),
            "config_path": project_relative(aux_config.path),
            "weights_sha256": aux_model.sha256,
            "config_sha256": aux_config.sha256,
        },
    }
    for section, expected_values in expected.items():
        actual = pair_config.get(section)
        if not isinstance(actual, dict):
            raise RuntimeError(f"pair config missing {section} binding")
        for key, expected_value in expected_values.items():
            if actual.get(key) != expected_value:
                raise RuntimeError(f"pair config {section}.{key} does not match model_pair_manifest.json")

    if pair_config.get("sensor_calibration_file") != project_relative(sensor_calibration.path):
        raise RuntimeError("pair config sensor calibration path does not match model_pair_manifest.json")
    if pair_config.get("sensor_calibration_sha256") != sensor_calibration.sha256:
        raise RuntimeError("pair config sensor calibration hash does not match model_pair_manifest.json")
    if pair_config.get("v3_calibration_file") != project_relative(v3_calibration.path):
        raise RuntimeError("pair config V3 calibration path does not match model_pair_manifest.json")
    if pair_config.get("v3_calibration_sha256") != v3_calibration.sha256:
        raise RuntimeError("pair config V3 calibration hash does not match model_pair_manifest.json")


def _validate_algorithm_hashes(algorithm_hashes: dict[str, Any]) -> tuple[dict[str, Any], str, str, str]:
    required = ("sensor_calibration_source", "v3_calibration_source")
    for key in required:
        if not isinstance(algorithm_hashes.get(key), dict):
            raise TypeError(f"algorithm_hashes.{key} must be an object")

    verified: dict[str, Any] = {}
    for key, entry in algorithm_hashes.items():
        if not isinstance(entry, dict):
            raise TypeError(f"algorithm_hashes.{key} must be an object")
        ref = _artifact_ref(entry, "path", "sha256", f"algorithm_hashes.{key}")
        verified[key] = {
            "path": project_relative(ref.path),
            "sha256": ref.sha256,
        }

    sensor_hash = verified["sensor_calibration_source"]["sha256"]
    v3_hash = verified["v3_calibration_source"]["sha256"]
    combined = hashlib.sha256(
        json.dumps(
            {
                "sensor_calibration_source": sensor_hash,
                "v3_calibration_source": v3_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return verified, sensor_hash, v3_hash, combined


def load_pair_bundle(training_seed: int) -> PairBundle:
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported training seed {training_seed}")
    manifest_path = MODEL_PAIR_MANIFEST_PATH
    manifest = read_json(manifest_path)
    if manifest.get("study") != "training_seed_robustness":
        raise RuntimeError("model_pair_manifest.json belongs to the wrong study")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, dict):
        raise TypeError("model_pair_manifest.json missing pairs object")
    pair_key = f"P{training_seed}"
    pair = pairs.get(pair_key)
    if not isinstance(pair, dict):
        raise KeyError(f"model_pair_manifest.json missing pairs.{pair_key}")
    if int(pair.get("training_seed", -1)) != training_seed:
        raise RuntimeError(f"pairs.{pair_key}.training_seed does not match {training_seed}")

    main = pair.get("main")
    auxiliary = pair.get("auxiliary")
    sensor_entry = pair.get("sensor_calibration")
    v3_entry = pair.get("v3_calibration")
    pair_config_entry = pair.get("pair_config")
    if not all(isinstance(value, dict) for value in (main, auxiliary, sensor_entry, v3_entry, pair_config_entry)):
        raise TypeError(f"pairs.{pair_key} has incomplete artifact bindings")

    main_model = _artifact_ref(main, "weights_path", "weights_sha256", f"pairs.{pair_key}.main weights")
    main_config = _artifact_ref(main, "config_path", "config_sha256", f"pairs.{pair_key}.main config")
    aux_model = _artifact_ref(
        auxiliary, "weights_path", "weights_sha256", f"pairs.{pair_key}.auxiliary weights"
    )
    aux_config = _artifact_ref(
        auxiliary, "config_path", "config_sha256", f"pairs.{pair_key}.auxiliary config"
    )
    sensor_calibration = _artifact_ref(
        sensor_entry, "path", "sha256", f"pairs.{pair_key}.sensor_calibration"
    )
    v3_calibration = _artifact_ref(
        v3_entry, "path", "sha256", f"pairs.{pair_key}.v3_calibration"
    )
    pair_config = _artifact_ref(
        pair_config_entry, "path", "sha256", f"pairs.{pair_key}.pair_config"
    )

    protocol_entry = manifest.get("protocol_manifest")
    if not isinstance(protocol_entry, dict):
        raise TypeError("model_pair_manifest.json missing protocol_manifest binding")
    protocol_manifest = _artifact_ref(
        protocol_entry, "path", "sha256", "protocol_manifest"
    )
    validation_ids = tuple(int(value) for value in manifest.get("validation_run_ids", []))
    if validation_ids != (3, 9, 11, 16, 19, 23):
        raise RuntimeError(f"unexpected calibration validation IDs: {validation_ids}")
    algorithm_hashes = manifest.get("algorithm_hashes")
    if not isinstance(algorithm_hashes, dict) or not algorithm_hashes:
        raise TypeError("model_pair_manifest.json missing algorithm_hashes")
    (
        verified_algorithm_hashes,
        sensor_algorithm_hash,
        v3_algorithm_hash,
        combined_algorithm_hash,
    ) = _validate_algorithm_hashes(algorithm_hashes)

    main_config_payload = read_json(main_config.path)
    aux_config_payload = read_json(aux_config.path)
    sensor_payload = read_json(sensor_calibration.path)
    v3_payload = read_json(v3_calibration.path)
    pair_config_payload = read_json(pair_config.path)
    _assert_payload_seed(main_config_payload, training_seed, "main config", require=True)
    _assert_payload_seed(aux_config_payload, training_seed, "aux config", require=True)
    _assert_payload_seed(sensor_payload, training_seed, "sensor calibration", require=True)
    _assert_payload_seed(v3_payload, training_seed, "V3 calibration", require=True)
    _assert_payload_seed(pair_config_payload, training_seed, "pair config", require=True)
    _assert_model_hash_bindings(
        sensor_payload,
        main_model.sha256,
        main_config.sha256,
        aux_model.sha256,
        aux_config.sha256,
        label="sensor calibration",
    )
    _assert_model_hash_bindings(
        v3_payload,
        main_model.sha256,
        main_config.sha256,
        aux_model.sha256,
        aux_config.sha256,
        label="V3 calibration",
    )
    _assert_pair_config_bindings(
        pair_config_payload,
        main_model,
        main_config,
        aux_model,
        aux_config,
        sensor_calibration,
        v3_calibration,
    )

    return PairBundle(
        training_seed=training_seed,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=sha256(manifest_path),
        main_model=main_model,
        main_config=main_config,
        aux_model=aux_model,
        aux_config=aux_config,
        sensor_calibration=sensor_calibration,
        v3_calibration=v3_calibration,
        pair_config=pair_config,
        protocol_manifest_path=protocol_manifest.path,
        protocol_manifest_sha256=protocol_manifest.sha256,
        algorithm_hashes=verified_algorithm_hashes,
        sensor_calibration_algorithm_sha256=sensor_algorithm_hash,
        v3_calibration_algorithm_sha256=v3_algorithm_hash,
        calibration_algorithm_sha256=combined_algorithm_hash,
    )


def validate_simulation_seed_freeze() -> tuple[int, ...]:
    freeze = read_json(SIMULATION_SEED_FREEZE_PATH)
    if freeze.get("study") != "training_seed_robustness":
        raise RuntimeError("simulation seed freeze belongs to the wrong study")
    if freeze.get("frozen_before_closed_loop") is not True:
        raise RuntimeError("simulation seeds were not frozen before closed-loop evaluation")
    selected = tuple(int(value) for value in freeze.get("selected_simulation_seeds", []))
    if selected != EXPECTED_SIMULATION_SEEDS:
        raise RuntimeError(
            f"expected selected simulation seeds {EXPECTED_SIMULATION_SEEDS}, found {selected}"
        )
    reserved = tuple(int(value) for value in freeze.get("reserved_seeds_untouched", []))
    if reserved != RESERVED_SIMULATION_SEEDS:
        raise RuntimeError(
            f"reserved seed declaration changed: expected {RESERVED_SIMULATION_SEEDS}, found {reserved}"
        )
    if set(selected) & set(reserved):
        raise RuntimeError("selected and reserved simulation seed families overlap")
    return selected


def _sensor_parameters(sensor_payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(sensor_payload.get("sensor"), dict):
        sensor = sensor_payload["sensor"]
    elif isinstance(sensor_payload.get("computed_values"), dict):
        sensor = sensor_payload["computed_values"]
    else:
        sensor = sensor_payload
    required = (
        "instant_threshold",
        "center",
        "allowance",
        "threshold",
        "enter_count",
        "exit_count",
    )
    missing = [key for key in required if key not in sensor]
    if missing:
        raise KeyError(f"sensor calibration missing runtime fields: {missing}")
    return sensor


def _v3_parameters(v3_payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(v3_payload.get("arbitrator"), dict):
        values = v3_payload["arbitrator"]
    elif isinstance(v3_payload.get("computed_values"), dict):
        values = v3_payload["computed_values"]
    else:
        values = v3_payload
    required = (
        "agreement_threshold",
        "param_mismatch_threshold",
        "param_mismatch_recovery_threshold",
        "aux_recovery_gate",
        "ewma_alpha",
        "startup_blanking_time",
        "param_mismatch_recovery_count",
        "aux_speed_bounds",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError(f"V3 calibration missing runtime fields: {missing}")
    return values


def _event_mask(scenario_name: str) -> np.ndarray:
    scenario = frozen_v3.SCENARIOS[scenario_name]
    event_start = scenario["event_start"]
    event_end = scenario["event_end"]
    if event_start is None:
        return np.ones(len(TIME), dtype=bool)
    mask = TIME >= event_start
    if event_end is not None:
        mask &= TIME < event_end
    return mask


def _pair_metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "training_seed": int(bundle["training_seed"]),
        "pair_manifest_path": project_relative(Path(bundle["manifest_path"])),
        "pair_manifest_sha256": bundle["manifest_sha256"],
        "main_model_sha256": bundle["main_model"]["sha256"],
        "main_config_sha256": bundle["main_config"]["sha256"],
        "aux_model_sha256": bundle["aux_model"]["sha256"],
        "aux_config_sha256": bundle["aux_config"]["sha256"],
        "sensor_calibration_path": project_relative(Path(bundle["sensor_calibration"]["path"])),
        "sensor_calibration_sha256": bundle["sensor_calibration"]["sha256"],
        "v3_calibration_path": project_relative(Path(bundle["v3_calibration"]["path"])),
        "v3_calibration_sha256": bundle["v3_calibration"]["sha256"],
        "pair_config_path": project_relative(Path(bundle["pair_config"]["path"])),
        "pair_config_sha256": bundle["pair_config"]["sha256"],
        "calibration_path": project_relative(Path(bundle["pair_config"]["path"])),
        "calibration_sha256": bundle["pair_config"]["sha256"],
        "sensor_calibration_algorithm_sha256": bundle["sensor_calibration_algorithm_sha256"],
        "v3_calibration_algorithm_sha256": bundle["v3_calibration_algorithm_sha256"],
        "calibration_algorithm_sha256": bundle["calibration_algorithm_sha256"],
    }


def simulate_training_seed_run(
    args: tuple[int, str, str, int, dict[str, Any]],
) -> dict[str, Any]:
    """Run one B/C3 cell with frozen V3 simulation semantics."""

    training_seed, controller_type, scenario_name, simulation_seed, bundle = args
    if training_seed != int(bundle["training_seed"]):
        raise RuntimeError("task training seed does not match its artifact bundle")
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported training seed {training_seed}")
    if controller_type not in CONTROLLERS:
        raise ValueError(f"unsupported controller {controller_type}")
    if scenario_name not in SCENARIOS:
        raise ValueError(f"unsupported scenario {scenario_name}")
    if simulation_seed not in EXPECTED_SIMULATION_SEEDS:
        raise ValueError(f"simulation seed {simulation_seed} is outside the frozen study block")

    torch.set_num_threads(1)
    device = "cpu"
    main_model_path = Path(bundle["main_model"]["path"])
    main_config_path = Path(bundle["main_config"]["path"])
    aux_model_path = Path(bundle["aux_model"]["path"])
    aux_config_path = Path(bundle["aux_config"]["path"])
    sensor_calibration_path = Path(bundle["sensor_calibration"]["path"])
    v3_calibration_path = Path(bundle["v3_calibration"]["path"])

    main_model, main_cfg = load_lstm_model(main_model_path, main_config_path, device=device)
    aux_cfg = read_json(aux_config_path)
    sensor_payload = read_json(sensor_calibration_path)
    v3_payload = read_json(v3_calibration_path)
    _assert_payload_seed(main_cfg, training_seed, "main config", require=True)
    _assert_payload_seed(aux_cfg, training_seed, "aux config", require=True)
    _assert_payload_seed(sensor_payload, training_seed, "sensor calibration", require=True)
    _assert_payload_seed(v3_payload, training_seed, "V3 calibration", require=True)

    sensor_cfg = _sensor_parameters(sensor_payload)
    v3_cfg = _v3_parameters(v3_payload)
    main_norm = main_cfg["normalization"]
    aux_norm = aux_cfg["normalization"]

    aux_model: AuxiliarySpeedEstimator | None = None
    if controller_type == "C3_arbitration_MPC":
        aux_model = AuxiliarySpeedEstimator(**aux_cfg["model"]).to(device)
        aux_model.load_state_dict(
            torch.load(aux_model_path, map_location=device, weights_only=True)
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

    sensor = SensorReliabilityMonitor(
        sensor_cfg["instant_threshold"],
        sensor_cfg["center"],
        sensor_cfg["allowance"],
        sensor_cfg["threshold"],
        sensor_cfg["enter_count"],
        sensor_cfg["exit_count"],
        aux_recovery_gate=v3_cfg["aux_recovery_gate"] if controller_type == "C3_arbitration_MPC" else None,
    )
    arbitrator = DualVirtualSensorArbitrator(
        **{key: value for key, value in v3_cfg.items() if key != "aux_recovery_gate"}
    )

    rng = np.random.default_rng(simulation_seed)
    base_noise = rng.normal(0.0, SPEED_NOISE_STD, len(TIME))

    state = np.zeros(2)
    prev_voltage = 0.0
    history = np.zeros((20, 2), dtype=np.float32)
    aux_history = np.zeros((20, 2), dtype=np.float32)
    pi = PIController(pi_config)
    mpc = LSTMMPC(main_model, main_norm, 20, mpc_config)
    ref = frozen_v3.reference_values(scenario_name)

    true_speed_arr = np.zeros(len(TIME))
    meas_speed_arr = np.zeros(len(TIME))
    main_speed_arr = np.full(len(TIME), np.nan)
    aux_speed_arr = np.full(len(TIME), np.nan)
    feedback_arr = np.zeros(len(TIME))
    voltage_arr = np.zeros(len(TIME))
    sub_arr = np.zeros(len(TIME), dtype=bool)
    sensor_state_arr = np.zeros(len(TIME), dtype=bool)
    source_arr = np.zeros(len(TIME), dtype=int)

    solve_times: list[float] = []
    aux_times: list[float] = []
    reliability_times: list[float] = []
    arbitration_times: list[float] = []
    extension_times: list[float] = []
    control_times: list[float] = []
    timing_samples: list[dict[str, Any]] = []
    reliability_events: list[dict[str, Any]] = []
    optimizer_failures = 0
    main_prediction_failures = 0
    aux_prediction_failures = 0
    prediction_failure_messages: set[str] = set()
    nonfinite_events = 0
    last_trusted_speed = 0.0
    last_main_prediction = 0.0

    run_started = perf_counter()
    for i, t in enumerate(TIME):
        current_val, true_speed = float(state[0]), float(state[1])
        true_speed_arr[i] = true_speed

        ym = true_speed + base_noise[i]
        if scenario_name == "sensor_bias_5" and 2.0 <= t < 4.0:
            ym += 0.05 * FULL_SCALE
        elif scenario_name == "sensor_dropout" and 2.0 <= t < 4.0:
            ym = 0.0
        elif scenario_name == "combined_fault_load" and t >= 3.0:
            ym += 0.05 * FULL_SCALE
        meas_speed_arr[i] = ym

        if i >= 20:
            try:
                y_hat = float(mpc.predict(history, np.array([prev_voltage], dtype=np.float32))[0])
                last_main_prediction = y_hat
            except (ValueError, FloatingPointError, RuntimeError) as error:
                main_prediction_failures += 1
                prediction_failure_messages.add(f"main:{type(error).__name__}:{error}")
                y_hat = last_main_prediction
        else:
            y_hat = ym
            if np.isfinite(y_hat):
                last_main_prediction = y_hat
        main_speed_arr[i] = y_hat

        y_aux = np.nan
        aux_ms = reliability_ms = arbitration_ms = 0.0
        if i >= 20 and controller_type == "C3_arbitration_MPC":
            started = perf_counter()
            try:
                assert aux_model is not None
                y_aux = auxiliary_predict_online(
                    aux_model,
                    aux_history[:, 0],
                    aux_history[:, 1],
                    aux_norm,
                )
            except (ValueError, FloatingPointError, RuntimeError) as error:
                aux_prediction_failures += 1
                prediction_failure_messages.add(f"aux:{type(error).__name__}:{error}")
            aux_ms = 1000.0 * (perf_counter() - started)
            aux_times.append(aux_ms)
        aux_speed_arr[i] = y_aux

        if controller_type == "B_plain_MPC" or i < 20:
            is_sub = False
            is_suspect = False
            y_fb = ym
            source_code = 0
        else:
            r_main = ym - y_hat
            r_aux = (ym - y_aux) if np.isfinite(y_aux) else None
            started = perf_counter()
            was_active = sensor.active
            decision = sensor.update(r_main, aux_residual=r_aux)
            reliability_ms = 1000.0 * (perf_counter() - started)
            is_sub = bool(decision["substitute"])
            is_suspect = bool(decision["sensor_suspect"])
            if sensor.active != was_active:
                reliability_events.append(
                    {
                        "training_seed": training_seed,
                        "controller": controller_type,
                        "scenario": scenario_name,
                        "simulation_seed": simulation_seed,
                        "seed": simulation_seed,
                        "step": i,
                        "time_s": float(t),
                        "event": "entry" if sensor.active else "recovery",
                        "raw_main_residual": float(r_main),
                        "aux_residual": float(r_aux) if r_aux is not None else np.nan,
                        "cusum_score": float(decision["score"]),
                    }
                )

            arbitration_started = perf_counter()
            arbitration = arbitrator.update(
                t=t,
                y_measured=ym,
                y_main=y_hat,
                y_aux=y_aux,
                sensor_trusted=(not is_sub),
                fallback_speed=last_trusted_speed,
            )
            arbitration_ms = 1000.0 * (perf_counter() - arbitration_started)
            arbitration_times.append(arbitration_ms)
            y_fb = arbitration["feedback"]
            source_code = arbitration["source_code"]

        if not is_sub and np.isfinite(ym):
            last_trusted_speed = float(np.clip(ym, *arbitrator.aux_speed_bounds))
        if controller_type == "C3_arbitration_MPC" and i >= 20:
            reliability_times.append(reliability_ms)
            extension_ms = aux_ms + reliability_ms + arbitration_ms
            extension_times.append(extension_ms)
            timing_samples.append(
                {
                    "training_seed": training_seed,
                    "controller": controller_type,
                    "scenario": scenario_name,
                    "simulation_seed": simulation_seed,
                    "seed": simulation_seed,
                    "step": i,
                    "time_s": float(t),
                    "aux_inference_ms": aux_ms,
                    "reliability_update_ms": reliability_ms,
                    "arbitration_ms": arbitration_ms,
                    "extension_ms": extension_ms,
                    "control_compute_ms": np.nan,
                }
            )

        required = [ym, y_fb, y_hat]
        nonfinite_events += int(not np.isfinite(required).all())

        feedback_arr[i] = y_fb
        sub_arr[i] = is_sub
        sensor_state_arr[i] = is_suspect
        source_arr[i] = source_code

        history = np.vstack((history[1:], [prev_voltage, y_fb])).astype(np.float32)
        aux_history = np.vstack((aux_history[1:], [prev_voltage, current_val])).astype(np.float32)

        voltage = prev_voltage
        if i % CONTROL_STRIDE == 0:
            control_started = perf_counter()
            fallback_voltage = pi.compute_control(ref[i], y_fb, CONTROL_DT)
            solve_started = perf_counter()
            output = mpc.compute_control(
                history,
                frozen_v3.reference_values(
                    scenario_name,
                    t + DT * np.arange(1, mpc_config.horizon + 1),
                ),
                prev_voltage,
                horizon=mpc_config.horizon,
                move_blocks=mpc_config.move_blocks,
                fallback_voltage=fallback_voltage,
            )
            solve_times.append(perf_counter() - solve_started)
            if not output["success"]:
                optimizer_failures += 1
            voltage = output["voltage"]
            control_ms = 1000.0 * (perf_counter() - control_started)
            control_times.append(control_ms)
            if controller_type == "C3_arbitration_MPC" and timing_samples:
                timing_samples[-1]["control_compute_ms"] = control_ms

        voltage_arr[i] = voltage
        history[-1, 0] = voltage
        aux_history[-1, 0] = voltage
        prev_voltage = voltage

        load_value = (
            0.15
            if scenario_name in ("load_disturbance", "combined_fault_load") and t >= 3.0
            else 0.03
        )
        params = (
            frozen_v3.shifted_params
            if scenario_name == "parameter_variation" and t >= 3.0
            else frozen_v3.nominal_params
        )
        if i < len(TIME) - 1:
            state = frozen_v3.rk4_step(state, voltage, load_value, params)

    total_sim_time = perf_counter() - run_started

    error = ref - true_speed_arr
    overall_rmse = float(np.sqrt(np.mean(error**2)))
    overall_mae = float(np.mean(np.abs(error)))
    scenario_config = frozen_v3.SCENARIOS[scenario_name]
    event_start = scenario_config["event_start"]
    event_end = scenario_config["event_end"]
    event_mask = _event_mask(scenario_name)
    fault_window_rmse = float(np.sqrt(np.mean(error[event_mask] ** 2)))

    switches = int(np.sum(source_arr[:-1] != source_arr[1:]))
    frac_phys = float(np.mean(source_arr == 0))
    frac_main = float(np.mean(source_arr == 1))
    frac_aux = float(np.mean(source_arr == 2))
    frac_fb = float(np.mean(source_arr == 3))
    event_frac_aux = float(np.mean(source_arr[event_mask] == 2))
    event_frac_fb = float(np.mean(source_arr[event_mask] == 3))
    avg_episode_duration_s = float(DURATION / (switches + 1))

    active_states = sensor_state_arr.copy()
    entry_indices = np.flatnonzero(~active_states[:-1] & active_states[1:]) + 1
    recovery_indices = np.flatnonzero(active_states[:-1] & ~active_states[1:]) + 1
    reliability_entries = int(len(entry_indices))
    false_recovery_events = int(np.sum(event_mask[recovery_indices])) if event_start is not None else 0
    total_recovery_events = int(len(recovery_indices))

    if event_start is None:
        post_event_entry_indices = entry_indices
    else:
        detection_mask = TIME[entry_indices] >= event_start
        if event_end is not None:
            detection_mask &= TIME[entry_indices] < event_end
        post_event_entry_indices = entry_indices[detection_mask]
    first_post_event_entry_time = (
        float(TIME[post_event_entry_indices[0]]) if len(post_event_entry_indices) else np.nan
    )
    detection_latency_s = np.nan
    sensor_fault_detected: bool | float = np.nan
    if event_start is not None and len(post_event_entry_indices):
        detection_latency_s = float(TIME[post_event_entry_indices[0]] - event_start)
    if scenario_config["type"] in {"sensor", "combined"} and event_start is not None:
        sensor_fault_detected = bool(len(post_event_entry_indices))

    post_event_recovery_indices = (
        recovery_indices[TIME[recovery_indices] >= event_start]
        if event_start is not None
        else recovery_indices
    )
    first_post_event_recovery_time = (
        float(TIME[post_event_recovery_indices[0]]) if len(post_event_recovery_indices) else np.nan
    )
    if event_end is not None and np.any(active_states & event_mask):
        recovered = np.flatnonzero((TIME >= event_end) & ~active_states)
        recovery_latency_s = (
            float(TIME[recovered[0]] - event_end) if len(recovered) else np.nan
        )
        recovered_after_event: bool | float = bool(len(recovered))
    else:
        recovery_latency_s = np.nan
        recovered_after_event = np.nan

    sub_count = int(np.sum(sub_arr))
    sub_duration_s = float(sub_count * DT)
    sub_fraction = float(sub_count / len(TIME))

    ctrl_voltages = voltage_arr[::CONTROL_STRIDE]
    ctrl_moves = np.diff(np.r_[0.0, ctrl_voltages])
    control_effort_u2 = float(np.sum(ctrl_voltages**2))
    control_variation_du2 = float(np.sum(ctrl_moves**2))
    voltage_violations = int(np.sum((ctrl_voltages < -1e-8) | (ctrl_voltages > 12.0 + 1e-8)))
    rate_violations = int(np.sum(np.abs(ctrl_moves) > 2.0 + 1e-8))

    sub_indices = np.where(sub_arr)[0]
    if len(sub_indices) > 0:
        main_error = np.abs(main_speed_arr[sub_indices] - true_speed_arr[sub_indices])
        corrupted_error = np.abs(meas_speed_arr[sub_indices] - true_speed_arr[sub_indices])
        selected_error = np.abs(feedback_arr[sub_indices] - true_speed_arr[sub_indices])
        valid_aux = np.isfinite(aux_speed_arr[sub_indices])
        if np.any(valid_aux):
            aux_error = np.abs(
                aux_speed_arr[sub_indices][valid_aux] - true_speed_arr[sub_indices][valid_aux]
            )
            aux_rmse = float(np.sqrt(np.mean(aux_error**2)))
            aux_mae = float(np.mean(aux_error))
        else:
            aux_rmse = np.nan
            aux_mae = np.nan
        quality = {
            "main_virtual_rmse": float(np.sqrt(np.mean(main_error**2))),
            "main_virtual_mae": float(np.mean(main_error)),
            "aux_virtual_rmse": aux_rmse,
            "aux_virtual_mae": aux_mae,
            "selected_virtual_rmse": float(np.sqrt(np.mean(selected_error**2))),
            "selected_virtual_mae": float(np.mean(selected_error)),
            "corrupted_rmse": float(np.sqrt(np.mean(corrupted_error**2))),
            "corrupted_mae": float(np.mean(corrupted_error)),
            "sub_samples": int(len(sub_indices)),
        }
    else:
        quality = {
            "main_virtual_rmse": np.nan,
            "main_virtual_mae": np.nan,
            "aux_virtual_rmse": np.nan,
            "aux_virtual_mae": np.nan,
            "selected_virtual_rmse": np.nan,
            "selected_virtual_mae": np.nan,
            "corrupted_rmse": np.nan,
            "corrupted_mae": np.nan,
            "sub_samples": 0,
        }

    total_failure_count = optimizer_failures + main_prediction_failures + aux_prediction_failures
    total_safety_violation_count = voltage_violations + rate_violations + nonfinite_events
    result: dict[str, Any] = {
        **_pair_metadata(bundle),
        "controller": controller_type,
        "scenario": scenario_name,
        "seed": simulation_seed,
        "simulation_seed": simulation_seed,
        "run_complete": True,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "fault_window_rmse": fault_window_rmse,
        "detection_latency_s": detection_latency_s,
        "sensor_fault_detected": sensor_fault_detected,
        "first_post_event_reliability_entry_time_s": first_post_event_entry_time,
        "post_event_reliability_entries": int(len(post_event_entry_indices)),
        "recovery_latency_s": recovery_latency_s,
        "reliability_entries": reliability_entries,
        "recovery_events": total_recovery_events,
        "false_recovery_events": false_recovery_events,
        "first_post_event_recovery_time_s": first_post_event_recovery_time,
        "post_event_recovery_events": int(len(post_event_recovery_indices)),
        "recovered_after_event": recovered_after_event,
        "sub_duration_s": sub_duration_s,
        "sub_fraction": sub_fraction,
        "switches": switches,
        "source_switch_count": switches,
        "avg_episode_duration_s": avg_episode_duration_s,
        "frac_phys": frac_phys,
        "frac_main": frac_main,
        "frac_aux": frac_aux,
        "frac_fb": frac_fb,
        "physical_fraction": frac_phys,
        "main_fraction": frac_main,
        "auxiliary_fraction": frac_aux,
        "fallback_fraction": frac_fb,
        "event_frac_aux": event_frac_aux,
        "event_frac_fb": event_frac_fb,
        "control_effort_u2": control_effort_u2,
        "control_variation_du2": control_variation_du2,
        "voltage_violations": voltage_violations,
        "rate_violations": rate_violations,
        "slew_violations": rate_violations,
        "optimizer_failures": optimizer_failures,
        "main_prediction_failures": main_prediction_failures,
        "aux_prediction_failures": aux_prediction_failures,
        "total_failure_count": total_failure_count,
        "failure_count": total_failure_count,
        "prediction_failure_messages": " | ".join(sorted(prediction_failure_messages)),
        "nonfinite_events": nonfinite_events,
        "total_safety_violation_count": total_safety_violation_count,
        "safety_violation_count": total_safety_violation_count,
        "mean_solve_ms": float(np.mean(solve_times) * 1000.0) if solve_times else 0.0,
        "mean_aux_inference_ms": float(np.mean(aux_times)) if aux_times else 0.0,
        "mean_reliability_update_ms": float(np.mean(reliability_times)) if reliability_times else 0.0,
        "mean_arbitration_ms": float(np.mean(arbitration_times)) if arbitration_times else 0.0,
        "mean_extension_ms": float(np.mean(extension_times)) if extension_times else 0.0,
        "mean_control_compute_ms": float(np.mean(control_times)) if control_times else 0.0,
        "sim_time_s": total_sim_time,
        "runtime_s": total_sim_time,
        **quality,
    }
    if timing_samples:
        result["timing_samples"] = timing_samples
    if reliability_events:
        result["reliability_events"] = reliability_events
    return result


def expected_matrix_keys(
    simulation_seeds: tuple[int, ...] = EXPECTED_SIMULATION_SEEDS,
) -> set[tuple[int, str, str, int]]:
    return {
        (training_seed, controller, scenario, simulation_seed)
        for training_seed in TRAINING_SEEDS
        for controller in CONTROLLERS
        for scenario in SCENARIOS
        for simulation_seed in simulation_seeds
    }


def _validate_result_matrix(runs: pd.DataFrame) -> None:
    if len(runs) != EXPECTED_RUN_COUNT:
        raise RuntimeError(f"expected exactly {EXPECTED_RUN_COUNT} result rows, found {len(runs)}")
    keys = {
        (int(row.training_seed), row.controller, row.scenario, int(row.simulation_seed))
        for row in runs.itertuples(index=False)
    }
    expected = expected_matrix_keys()
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise RuntimeError(f"result matrix mismatch; missing={missing[:10]}, extra={extra[:10]}")
    if runs.duplicated(["training_seed", "controller", "scenario", "simulation_seed"]).any():
        raise RuntimeError("duplicate training-seed robustness cells detected")
    if not np.array_equal(runs["seed"].to_numpy(int), runs["simulation_seed"].to_numpy(int)):
        raise RuntimeError("legacy seed alias diverged from simulation_seed")

    for training_seed in TRAINING_SEEDS:
        subset = runs[runs.training_seed == training_seed]
        if len(subset) != len(CONTROLLERS) * len(SCENARIOS) * len(EXPECTED_SIMULATION_SEEDS):
            raise RuntimeError(f"training seed {training_seed} does not have exactly 50 rows")
        metadata_columns = (
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
        )
        for column in metadata_columns:
            if subset[column].nunique(dropna=False) != 1:
                raise RuntimeError(f"training seed {training_seed} mixes {column} values")

        # B and C3 must be paired on the exact same main artifact for every
        # scenario/simulation seed.  Both rows also carry the same matched aux
        # and calibration provenance even though B does not use them online.
        paired = subset.pivot(
            index=["scenario", "simulation_seed"],
            columns="controller",
            values=[
                "main_model_sha256",
                "main_config_sha256",
                "aux_model_sha256",
                "aux_config_sha256",
                "sensor_calibration_sha256",
                "v3_calibration_sha256",
                "pair_config_sha256",
            ],
        )
        for field in paired.columns.levels[0]:
            left = paired[(field, "B_plain_MPC")]
            right = paired[(field, "C3_arbitration_MPC")]
            if not left.equals(right):
                raise RuntimeError(
                    f"training seed {training_seed} has cross-controller metadata mixing in {field}"
                )


def _output_exists() -> list[Path]:
    return [path for path in (RUNS_PATH, TIMING_PATH, RELIABILITY_EVENTS_PATH, RUN_MANIFEST_PATH) if path.exists()]


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
        "--workers",
        type=int,
        default=min(int(os.environ.get("TRAINING_SEED_WORKERS", "8")), os.cpu_count() or 1),
        help="number of process workers used for the 150-cell matrix",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    existing = _output_exists()
    if existing:
        raise RuntimeError(
            "study evaluation outputs already exist; refusing to overwrite: "
            + ", ".join(project_relative(path) for path in existing)
        )

    simulation_seeds = validate_simulation_seed_freeze()
    bundles = {training_seed: load_pair_bundle(training_seed) for training_seed in TRAINING_SEEDS}
    payloads = {seed: bundle.to_worker_payload() for seed, bundle in bundles.items()}

    tasks = [
        (training_seed, controller, scenario, simulation_seed, payloads[training_seed])
        for training_seed in TRAINING_SEEDS
        for scenario in SCENARIOS
        for simulation_seed in simulation_seeds
        for controller in CONTROLLERS
    ]
    if len(tasks) != EXPECTED_RUN_COUNT:
        raise AssertionError(f"internal task count changed: {len(tasks)}")

    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(simulate_training_seed_run, tasks))
    wall_time_s = perf_counter() - started

    timing_rows: list[dict[str, Any]] = []
    reliability_event_rows: list[dict[str, Any]] = []
    for result in results:
        timing_rows.extend(result.pop("timing_samples", []))
        reliability_event_rows.extend(result.pop("reliability_events", []))

    runs = pd.DataFrame(results).sort_values(
        ["training_seed", "scenario", "simulation_seed", "controller"]
    ).reset_index(drop=True)
    _validate_result_matrix(runs)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_PATH, index=False)
    if len(pd.read_csv(RUNS_PATH)) != EXPECTED_RUN_COUNT:
        raise RuntimeError("written training_seed_runs.csv does not contain exactly 150 rows")

    if timing_rows:
        pd.DataFrame(timing_rows).sort_values(
            ["training_seed", "scenario", "simulation_seed", "step"]
        ).to_csv(TIMING_PATH, index=False)
    if reliability_event_rows:
        pd.DataFrame(reliability_event_rows).sort_values(
            ["training_seed", "scenario", "simulation_seed", "step"]
        ).to_csv(RELIABILITY_EVENTS_PATH, index=False)

    evaluation_manifest = {
        "study": "training_seed_robustness",
        "run_count": int(len(runs)),
        "training_seeds": list(TRAINING_SEEDS),
        "simulation_seeds": list(simulation_seeds),
        "controllers": list(CONTROLLERS),
        "scenarios": list(SCENARIOS),
        "paired_design": "same scenario and simulation seed across B/C3 and all three matched training pairs",
        "simulation_seed_freeze_path": project_relative(SIMULATION_SEED_FREEZE_PATH),
        "simulation_seed_freeze_sha256": sha256(SIMULATION_SEED_FREEZE_PATH),
        "frozen_v3_evaluator_path": project_relative(PROJECT / "scripts/evaluate_v3_closed_loop.py"),
        "frozen_v3_evaluator_sha256": sha256(PROJECT / "scripts/evaluate_v3_closed_loop.py"),
        "pair_manifests": {
            str(seed): {
                "path": project_relative(bundle.manifest_path),
                "sha256": bundle.manifest_sha256,
                "main_model_sha256": bundle.main_model.sha256,
                "main_config_sha256": bundle.main_config.sha256,
                "aux_model_sha256": bundle.aux_model.sha256,
                "aux_config_sha256": bundle.aux_config.sha256,
                "sensor_calibration_sha256": bundle.sensor_calibration.sha256,
                "v3_calibration_sha256": bundle.v3_calibration.sha256,
                "pair_config_sha256": bundle.pair_config.sha256,
                "protocol_manifest_sha256": bundle.protocol_manifest_sha256,
                "algorithm_hashes": bundle.algorithm_hashes,
                "sensor_calibration_algorithm_sha256": bundle.sensor_calibration_algorithm_sha256,
                "v3_calibration_algorithm_sha256": bundle.v3_calibration_algorithm_sha256,
                "calibration_algorithm_sha256": bundle.calibration_algorithm_sha256,
            }
            for seed, bundle in bundles.items()
        },
        "output": {
            "runs": project_relative(RUNS_PATH),
            "runs_sha256": sha256(RUNS_PATH),
            "timing": project_relative(TIMING_PATH) if TIMING_PATH.exists() else None,
            "timing_sha256": sha256(TIMING_PATH) if TIMING_PATH.exists() else None,
            "reliability_events": (
                project_relative(RELIABILITY_EVENTS_PATH) if RELIABILITY_EVENTS_PATH.exists() else None
            ),
            "reliability_events_sha256": (
                sha256(RELIABILITY_EVENTS_PATH) if RELIABILITY_EVENTS_PATH.exists() else None
            ),
        },
        "evaluation_wall_time_s": wall_time_s,
    }
    RUN_MANIFEST_PATH.write_text(
        json.dumps(_json_ready(evaluation_manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved {len(runs)} rows to {RUNS_PATH}; wall time {wall_time_s:.1f}s.")


if __name__ == "__main__":
    main()
