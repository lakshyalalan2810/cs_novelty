"""Independent verifier for the preregistered training-seed robustness study.

The verifier is intentionally separate from the training/evaluation runners. It
rebinds every study artifact by SHA-256, reconstructs the clean-validation
sensor/CUSUM and V3 arbitration calibrations from the saved models, validates
the complete 150-cell paired matrix, and recomputes the robustness verdict from
the raw rows.

This module performs no work on import. Run it only after the training,
closed-loop matrix, and analysis outputs exist.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auxiliary_sensor_model import (  # noqa: E402
    AuxiliarySpeedEstimator,
    build_auxiliary_sequences,
    normalize_auxiliary,
)
from data_utils import build_training_sequences  # noqa: E402
from lstm_model import LSTMForecaster  # noqa: E402


STUDY = "training_seed_robustness"
TRAINING_SEEDS = (2026, 2027, 2028)
SIMULATION_SEEDS = (49026, 49027, 49028, 49029, 49030)
RESERVED_SEEDS = (39026, 39027, 39028, 39029, 39030)
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
VALIDATION_IDS = (3, 9, 11, 16, 19, 23)
TEST_IDS = (8, 15, 17, 22, 25, 28)
WINDOW_LENGTH = 20

MAIN_MODEL_SPEC = {
    "input_size": 2,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "output_size": 1,
    "residual": True,
}
MAIN_TRAINING_SPEC = {
    "batch_size": 512,
    "learning_rate": 0.001,
    "max_epochs": 60,
    "patience": 8,
    "min_delta": 1e-6,
}
AUX_MODEL_SPEC = {
    "input_size": 2,
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.0,
    "output_size": 1,
}
AUX_TRAINING_SPEC = {
    "batch_size": 512,
    "learning_rate": 0.001,
    "max_epochs": 100,
    "patience": 10,
    "min_delta": 1e-6,
}

SENSOR_PERCENTILES = (90.0, 95.0, 97.5, 99.0, 99.5, 99.9)
SENSOR_TARGET_FAR = 0.001
SENSOR_ENTER = 3
SENSOR_EXIT = 5
SENSOR_EWMA_ALPHA = 0.10
SENSOR_ROLLING_WINDOW = 20
ARBITRATION_EWMA_ALPHA = 0.05
ARBITRATION_STARTUP_BLANKING_S = 1.0
ARBITRATION_RECOVERY_COUNT = 50
AUX_SPEED_BOUNDS = [0.0, 100.0]

# Same portability tolerance used for the already-independent V3 verifier. It
# allows only final float32 CPU inference/reduction noise, not threshold drift.
MODEL_RTOL = 2e-6
MODEL_ATOL = 2e-6

RESULT_ROOT = PROJECT / "results" / STUDY
MODEL_PAIR_MANIFEST = RESULT_ROOT / "model_pair_manifest.json"
PROTOCOL_MANIFEST = RESULT_ROOT / "configs" / "training_protocol_manifest.json"
SIMULATION_SEED_FREEZE = RESULT_ROOT / "simulation_seed_freeze.json"
PREWORK_HASH_MANIFEST = RESULT_ROOT / "prework_frozen_hash_manifest.json"
RUNTIME_HASH_MANIFEST = RESULT_ROOT / "frozen_runtime_dependency_hashes.json"
RUNS_PATH = RESULT_ROOT / "training_seed_runs.csv"
EVALUATION_MANIFEST = RESULT_ROOT / "training_seed_evaluation_manifest.json"
ANALYSIS_SUMMARY = RESULT_ROOT / "analysis" / "analysis_summary.json"
ANALYSIS_ROOT = RESULT_ROOT / "analysis"
DATASET = PROJECT / "data" / "processed" / "dc_motor_lstm_dataset.npz"

ANALYSIS_TABLE_PATHS = {
    "load_disturbance": ANALYSIS_ROOT / "load_disturbance.csv",
    "parameter_variation": ANALYSIS_ROOT / "parameter_variation.csv",
    "main_model_metrics_by_seed": ANALYSIS_ROOT / "main_model_metrics_by_seed.csv",
    "main_model_metric_summary": ANALYSIS_ROOT / "main_model_metric_summary.csv",
    "auxiliary_model_metrics_by_seed": ANALYSIS_ROOT / "auxiliary_model_metrics_by_seed.csv",
    "auxiliary_model_metric_summary": ANALYSIS_ROOT / "auxiliary_model_metric_summary.csv",
    "auxiliary_per_trajectory": ANALYSIS_ROOT / "auxiliary_per_trajectory.csv",
    "auxiliary_closed_loop_transfer": ANALYSIS_ROOT / "auxiliary_closed_loop_transfer.csv",
    "calibration_by_seed": ANALYSIS_ROOT / "calibration_by_seed.csv",
    "calibration_variability": ANALYSIS_ROOT / "calibration_variability.csv",
    "hierarchical_bootstrap_ci": ANALYSIS_ROOT / "hierarchical_bootstrap_ci.csv",
}


class VerificationError(AssertionError):
    """Raised when one preregistered study invariant is violated."""


def check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"required JSON artifact is missing: {path.relative_to(PROJECT)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    check(isinstance(value, dict), f"{path.relative_to(PROJECT)} must contain a JSON object")
    return value


def resolve_project_path(raw: str, label: str) -> Path:
    check(isinstance(raw, str) and bool(raw), f"{label} path must be a non-empty string")
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else PROJECT / candidate
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT.resolve())
    except ValueError as exc:
        raise VerificationError(f"{label} escapes the project root: {raw}") from exc
    check(resolved.is_file(), f"{label} is missing: {raw}")
    return resolved


def exact_int_tuple(values: Iterable[Any], label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} is not an integer sequence") from exc
    return result


def assert_close(actual: float, expected: float, label: str) -> None:
    check(
        np.isclose(float(actual), float(expected), rtol=MODEL_RTOL, atol=MODEL_ATOL),
        f"{label} mismatch: reconstructed={actual!r}, saved={expected!r}",
    )


def verify_declared_file(
    entry: dict[str, Any],
    *,
    label: str,
    path_key: str = "path",
    hash_key: str = "sha256",
) -> Path:
    check(isinstance(entry, dict), f"{label} binding must be an object")
    raw_path = entry.get(path_key)
    declared = entry.get(hash_key)
    check(isinstance(declared, str) and len(declared) == 64, f"{label} has invalid SHA-256")
    path = resolve_project_path(raw_path, label)
    actual = sha256(path)
    check(actual == declared.lower(), f"{label} SHA-256 mismatch: {actual} != {declared}")
    return path


def verify_frozen_entry(entry: dict[str, Any], label: str) -> None:
    path = verify_declared_file(entry, label=label)
    if "bytes" in entry:
        check(path.stat().st_size == int(entry["bytes"]), f"{label} byte count changed")
    if "rows" in entry:
        check(path.suffix.lower() == ".csv", f"{label} row count declared for non-CSV")
        check(len(pd.read_csv(path)) == int(entry["rows"]), f"{label} row count changed")


def verify_prework_frozen_hashes() -> None:
    manifest = load_json(PREWORK_HASH_MANIFEST)
    check(manifest.get("study") == STUDY, "prework hash manifest belongs to the wrong study")

    frozen = manifest.get("frozen_artifacts")
    check(isinstance(frozen, dict) and frozen, "prework frozen-artifact map is missing")
    for name, entry in frozen.items():
        verify_frozen_entry(entry, f"prework frozen_artifacts.{name}")

    verifiers = manifest.get("existing_verifiers")
    check(isinstance(verifiers, list) and len(verifiers) >= 4, "prework verifier bindings are incomplete")
    for index, entry in enumerate(verifiers):
        verify_frozen_entry(entry, f"prework existing_verifiers[{index}]")

    provenance = manifest.get("provenance_sources")
    check(isinstance(provenance, dict) and provenance, "prework provenance-source map is missing")
    for name, entry in provenance.items():
        verify_frozen_entry(entry, f"prework provenance_sources.{name}")

    runtime = load_json(RUNTIME_HASH_MANIFEST)
    dependencies = runtime.get("dependencies")
    check(isinstance(dependencies, list) and dependencies, "runtime dependency freeze is missing")
    for index, entry in enumerate(dependencies):
        verify_frozen_entry(entry, f"runtime dependencies[{index}]")


def verify_simulation_seed_provenance() -> dict[str, Any]:
    freeze = load_json(SIMULATION_SEED_FREEZE)
    check(freeze.get("study") == STUDY, "simulation-seed freeze belongs to the wrong study")
    check(freeze.get("frozen_before_closed_loop") is True, "simulation seeds were not frozen pre-evaluation")
    check(
        exact_int_tuple(freeze.get("selected_simulation_seeds", []), "selected simulation seeds")
        == SIMULATION_SEEDS,
        f"selected simulation seed block must be exactly {SIMULATION_SEEDS}",
    )
    check(
        exact_int_tuple(freeze.get("reserved_seeds_untouched", []), "reserved simulation seeds")
        == RESERVED_SEEDS,
        f"reserved simulation seed family must remain exactly {RESERVED_SEEDS}",
    )
    check(set(SIMULATION_SEEDS).isdisjoint(RESERVED_SEEDS), "selected and reserved seeds overlap")
    check(freeze.get("preferred_candidate_accepted") is True, "preferred unused seed block was not accepted prospectively")

    provenance = freeze.get("provenance_scan")
    check(isinstance(provenance, dict), "simulation-seed provenance scan is missing")
    for key in ("strict_text_matches", "filename_matches", "csv_seed_matches", "npy_npz_seed_matches"):
        check(provenance.get(key) == [], f"pre-study provenance scan reports contamination in {key}")
    check(int(provenance.get("csv_seed_column_files_scanned", 0)) > 0, "CSV seed provenance scan was not recorded")
    check(int(provenance.get("npy_npz_payloads_scanned", 0)) > 0, "NPY/NPZ seed provenance scan was not recorded")
    conclusion = str(provenance.get("conclusion", "")).lower()
    check("no pre-study use" in conclusion, "seed-freeze conclusion does not establish unused-before-study provenance")
    if RUNS_PATH.exists():
        check(
            SIMULATION_SEED_FREEZE.stat().st_mtime_ns <= RUNS_PATH.stat().st_mtime_ns,
            "simulation-seed freeze file was recorded after the raw closed-loop matrix",
        )
    return freeze


@dataclass(frozen=True)
class PairArtifacts:
    seed: int
    main_weights: Path
    main_config: Path
    main_weights_sha256: str
    main_config_sha256: str
    aux_weights: Path
    aux_config: Path
    aux_weights_sha256: str
    aux_config_sha256: str
    sensor_calibration: Path
    sensor_calibration_sha256: str
    v3_calibration: Path
    v3_calibration_sha256: str
    pair_config: Path
    pair_config_sha256: str


def payload_seed(payload: dict[str, Any]) -> int | None:
    if "training_seed" in payload:
        return int(payload["training_seed"])
    if "seed" in payload:
        return int(payload["seed"])
    training = payload.get("training")
    if isinstance(training, dict) and "seed" in training:
        return int(training["seed"])
    return None


def verify_model_config(config: dict[str, Any], seed: int, *, auxiliary: bool) -> None:
    kind = "auxiliary" if auxiliary else "main"
    check(payload_seed(config) == seed, f"{kind} config seed does not match P{seed}")
    expected_model = AUX_MODEL_SPEC if auxiliary else MAIN_MODEL_SPEC
    expected_training = AUX_TRAINING_SPEC if auxiliary else MAIN_TRAINING_SPEC
    check(config.get("model") == expected_model, f"P{seed} {kind} architecture drift")
    training = config.get("training")
    check(isinstance(training, dict), f"P{seed} {kind} training config missing")
    for key, expected in expected_training.items():
        check(training.get(key) == expected, f"P{seed} {kind} hyperparameter drift: {key}")
    check(int(config.get("window_length", -1)) == WINDOW_LENGTH, f"P{seed} {kind} window length drift")
    if auxiliary:
        check(config.get("features") == ["voltage", "current"], f"P{seed} auxiliary feature drift")
    else:
        check(config.get("features") == ["voltage", "y_measured"], f"P{seed} main feature drift")
    check(config.get("target") == "y_true", f"P{seed} {kind} target drift")


def bind_pair_artifacts() -> tuple[dict[int, PairArtifacts], dict[str, Any]]:
    manifest = load_json(MODEL_PAIR_MANIFEST)
    check(manifest.get("study") == STUDY, "model-pair manifest belongs to the wrong study")
    check(
        exact_int_tuple(manifest.get("validation_run_ids", []), "manifest validation IDs") == VALIDATION_IDS,
        "model-pair manifest calibration population changed",
    )

    protocol_entry = manifest.get("protocol_manifest")
    protocol_path = verify_declared_file(protocol_entry, label="protocol manifest")
    check(protocol_path.resolve() == PROTOCOL_MANIFEST.resolve(), "model-pair manifest points to a noncanonical protocol manifest")
    protocol = load_json(protocol_path)
    check(protocol.get("study") == STUDY, "training protocol belongs to the wrong study")
    check(protocol.get("training_pairs") == ["P2026", "P2027", "P2028"], "training-pair protocol changed")
    check(protocol.get("new_training_seeds") == [2027, 2028], "new-training-seed protocol changed")
    calibration_policy = str(protocol.get("calibration_data_policy", "")).lower()
    for phrase in ("validation", "no test", "fault", "closed-loop", "simulation-seed"):
        check(phrase in calibration_policy, f"calibration data policy is missing leakage exclusion: {phrase}")

    dataset_binding = protocol.get("dataset")
    check(isinstance(dataset_binding, dict), "protocol dataset binding missing")
    check(dataset_binding.get("path") == "data/processed/dc_motor_lstm_dataset.npz", "protocol dataset path changed")
    check(dataset_binding.get("sha256") == sha256(DATASET), "protocol dataset hash does not match frozen dataset")
    split_ids = dataset_binding.get("split_run_ids")
    check(isinstance(split_ids, dict), "protocol trajectory split binding missing")
    check(tuple(split_ids.get("validation", [])) == VALIDATION_IDS, "protocol validation split drift")
    check(tuple(split_ids.get("test", [])) == TEST_IDS, "protocol test split drift")
    check(set(VALIDATION_IDS).isdisjoint(TEST_IDS), "validation/test split leakage")

    frozen_sources = protocol.get("frozen_source_hashes")
    check(isinstance(frozen_sources, dict) and frozen_sources, "protocol frozen source hashes missing")
    for name, entry in frozen_sources.items():
        verify_declared_file(entry, label=f"protocol frozen_source_hashes.{name}")

    algorithm_hashes = manifest.get("algorithm_hashes")
    check(isinstance(algorithm_hashes, dict), "model-pair manifest algorithm_hashes missing")
    for key in ("sensor_calibration_source", "v3_calibration_source", "runner"):
        verify_declared_file(algorithm_hashes.get(key), label=f"algorithm_hashes.{key}")

    pairs = manifest.get("pairs")
    check(isinstance(pairs, dict), "model-pair manifest pairs map missing")
    check(set(pairs) == {f"P{seed}" for seed in TRAINING_SEEDS}, "model-pair manifest must contain exactly P2026/P2027/P2028")

    result: dict[int, PairArtifacts] = {}
    for seed in TRAINING_SEEDS:
        key = f"P{seed}"
        pair = pairs[key]
        check(isinstance(pair, dict), f"{key} manifest entry must be an object")
        check(int(pair.get("training_seed", -1)) == seed, f"{key} training_seed mismatch")
        main = pair.get("main")
        aux = pair.get("auxiliary")
        sensor = pair.get("sensor_calibration")
        v3 = pair.get("v3_calibration")
        pair_cfg = pair.get("pair_config")
        main_w = verify_declared_file(main, label=f"{key}.main weights", path_key="weights_path", hash_key="weights_sha256")
        main_c = verify_declared_file(main, label=f"{key}.main config", path_key="config_path", hash_key="config_sha256")
        aux_w = verify_declared_file(aux, label=f"{key}.auxiliary weights", path_key="weights_path", hash_key="weights_sha256")
        aux_c = verify_declared_file(aux, label=f"{key}.auxiliary config", path_key="config_path", hash_key="config_sha256")
        sensor_p = verify_declared_file(sensor, label=f"{key}.sensor_calibration")
        v3_p = verify_declared_file(v3, label=f"{key}.v3_calibration")
        pair_p = verify_declared_file(pair_cfg, label=f"{key}.pair_config")

        main_config = load_json(main_c)
        aux_config = load_json(aux_c)
        verify_model_config(main_config, seed, auxiliary=False)
        verify_model_config(aux_config, seed, auxiliary=True)

        sensor_payload = load_json(sensor_p)
        v3_payload = load_json(v3_p)
        pair_payload = load_json(pair_p)
        for label, payload in (
            ("sensor calibration", sensor_payload),
            ("V3 calibration", v3_payload),
            ("pair config", pair_payload),
        ):
            check(payload_seed(payload) == seed, f"P{seed} {label} seed mismatch")
            check(payload.get("pair_id") == key, f"P{seed} {label} pair ID mismatch")

        check(
            exact_int_tuple(sensor_payload.get("validation_run_ids", []), "sensor calibration validation IDs") == VALIDATION_IDS,
            f"P{seed} sensor calibration population drift",
        )
        check(
            exact_int_tuple(v3_payload.get("validation_run_ids", []), "V3 calibration validation IDs") == VALIDATION_IDS,
            f"P{seed} V3 calibration population drift",
        )
        check(
            exact_int_tuple(pair_payload.get("validation_run_ids", []), "pair config validation IDs") == VALIDATION_IDS,
            f"P{seed} pair config validation population drift",
        )

        for label, payload in (("sensor", sensor_payload), ("V3", v3_payload), ("pair config", pair_payload)):
            no_fault = str(payload.get("no_fault_data_statement", "")).lower()
            for phrase in ("validation", "no test", "fault", "closed-loop", "simulation-seed"):
                check(phrase in no_fault, f"P{seed} {label} leakage statement missing {phrase!r}")

        sensor_hashes = sensor_payload.get("model_sha256")
        check(isinstance(sensor_hashes, dict), f"P{seed} sensor calibration model binding missing")
        check(sensor_hashes.get("main") == sha256(main_w), f"P{seed} sensor calibration main-weight mixing")
        check(sensor_hashes.get("main_config") == sha256(main_c), f"P{seed} sensor calibration main-config mixing")
        check(sensor_payload.get("dataset_sha256") == sha256(DATASET), f"P{seed} sensor calibration dataset hash mismatch")
        sensor_algorithm = sensor_payload.get("algorithm")
        check(isinstance(sensor_algorithm, dict), f"P{seed} sensor calibration algorithm provenance missing")
        check(
            sensor_algorithm.get("source_sha256")
            == algorithm_hashes["sensor_calibration_source"]["sha256"],
            f"P{seed} sensor calibration source hash mismatch",
        )
        check(
            sensor_algorithm.get("reliability_source_sha256")
            == sha256(PROJECT / "src" / "reliability.py"),
            f"P{seed} sensor calibration reliability-source hash mismatch",
        )

        v3_hashes = v3_payload.get("model_sha256")
        expected_v3_hashes = {
            "main": sha256(main_w),
            "main_config": sha256(main_c),
            "auxiliary": sha256(aux_w),
            "auxiliary_config": sha256(aux_c),
        }
        check(v3_hashes == expected_v3_hashes, f"P{seed} V3 calibration cross-seed model mixing")
        check(v3_payload.get("dataset_sha256") == sha256(DATASET), f"P{seed} V3 calibration dataset hash mismatch")
        v3_algorithm = v3_payload.get("algorithm")
        check(isinstance(v3_algorithm, dict), f"P{seed} V3 calibration algorithm provenance missing")
        check(
            v3_algorithm.get("source_sha256")
            == algorithm_hashes["v3_calibration_source"]["sha256"],
            f"P{seed} V3 calibration source hash mismatch",
        )

        main_pair = pair_payload.get("main_model")
        aux_pair = pair_payload.get("auxiliary_model")
        check(isinstance(main_pair, dict) and isinstance(aux_pair, dict), f"P{seed} pair config model binding missing")
        check(main_pair.get("weights_sha256") == sha256(main_w), f"P{seed} pair config main-weight mismatch")
        check(main_pair.get("config_sha256") == sha256(main_c), f"P{seed} pair config main-config mismatch")
        check(aux_pair.get("weights_sha256") == sha256(aux_w), f"P{seed} pair config auxiliary-weight mismatch")
        check(aux_pair.get("config_sha256") == sha256(aux_c), f"P{seed} pair config auxiliary-config mismatch")
        check(pair_payload.get("sensor_calibration_sha256") == sha256(sensor_p), f"P{seed} pair config sensor calibration mismatch")
        check(pair_payload.get("v3_calibration_sha256") == sha256(v3_p), f"P{seed} pair config V3 calibration mismatch")

        result[seed] = PairArtifacts(
            seed=seed,
            main_weights=main_w,
            main_config=main_c,
            main_weights_sha256=sha256(main_w),
            main_config_sha256=sha256(main_c),
            aux_weights=aux_w,
            aux_config=aux_c,
            aux_weights_sha256=sha256(aux_w),
            aux_config_sha256=sha256(aux_c),
            sensor_calibration=sensor_p,
            sensor_calibration_sha256=sha256(sensor_p),
            v3_calibration=v3_p,
            v3_calibration_sha256=sha256(v3_p),
            pair_config=pair_p,
            pair_config_sha256=sha256(pair_p),
        )

    canonical = protocol.get("canonical_model_hashes")
    check(isinstance(canonical, dict), "protocol canonical model hashes missing")
    check(result[2026].main_weights_sha256 == canonical.get("main_weights"), "P2026 main weight is not canonical")
    check(result[2026].main_config_sha256 == canonical.get("main_config"), "P2026 main config is not canonical")
    check(result[2026].aux_weights_sha256 == canonical.get("aux_weights"), "P2026 auxiliary weight is not canonical")
    check(result[2026].aux_config_sha256 == canonical.get("aux_config"), "P2026 auxiliary config is not canonical")
    return result, manifest


def load_main_model(pair: PairArtifacts) -> tuple[LSTMForecaster, dict[str, Any]]:
    config = load_json(pair.main_config)
    model = LSTMForecaster(**config["model"])
    model.load_state_dict(torch.load(pair.main_weights, map_location="cpu", weights_only=True))
    model.eval()
    return model, config


def load_aux_model(pair: PairArtifacts) -> tuple[AuxiliarySpeedEstimator, dict[str, Any]]:
    config = load_json(pair.aux_config)
    model = AuxiliarySpeedEstimator(**config["model"])
    model.load_state_dict(torch.load(pair.aux_weights, map_location="cpu", weights_only=True))
    model.eval()
    return model, config


@torch.no_grad()
def predict_physical(
    model: torch.nn.Module,
    inputs: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(inputs), 1024):
        chunks.append(model(torch.from_numpy(inputs[start : start + 1024]))[:, 0].numpy())
    prediction = np.concatenate(chunks) * float(std) + float(mean)
    check(np.isfinite(prediction).all(), "independent calibration prediction contains NaN/Inf")
    return prediction


def independent_cusum_parameters(residual_rows: np.ndarray, percentile: float) -> dict[str, float]:
    rows = np.asarray(residual_rows, dtype=float)
    check(rows.ndim == 2, "independent CUSUM requires one clean residual row per validation trajectory")
    flat = rows.reshape(-1)
    center = float(np.median(flat))
    allowance = 0.5 * float(np.std(flat, ddof=1))
    scores = []
    for row in rows:
        positive = 0.0
        negative = 0.0
        row_scores = []
        for value in row:
            centered = float(value) - center
            positive = max(0.0, positive + centered - allowance)
            negative = max(0.0, negative - centered - allowance)
            row_scores.append(max(positive, negative))
        scores.extend(row_scores)
    return {
        "center": center,
        "allowance": allowance,
        "threshold": float(np.percentile(np.asarray(scores), percentile)),
    }


def independent_temporal_ewma(row: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(row, dtype=float)
    signed = np.empty_like(values)
    absolute = np.empty_like(values)
    if not len(values):
        return signed, absolute
    signed[0] = values[0]
    absolute[0] = abs(values[0])
    for index in range(1, len(values)):
        signed[index] = alpha * values[index] + (1.0 - alpha) * signed[index - 1]
        absolute[index] = alpha * abs(values[index]) + (1.0 - alpha) * absolute[index - 1]
    return signed, absolute


@torch.no_grad()
def independent_guarded_validation(
    model: LSTMForecaster,
    voltage: np.ndarray,
    measured: np.ndarray,
    normalization: dict[str, Any],
    residual_gate: float,
    cusum: dict[str, float],
) -> tuple[float, float]:
    """Reimplement the frozen guarded feedback state machine for clean FAR."""
    voltage = np.asarray(voltage, dtype=np.float32)
    measured = np.asarray(measured, dtype=np.float32)
    check(voltage.shape == measured.shape and voltage.ndim == 2, "validation arrays have unexpected shape")

    input_mean = np.asarray(normalization["input_mean"], dtype=np.float32)
    input_std = np.asarray(normalization["input_std"], dtype=np.float32)
    target_mean = float(normalization["target_mean"][0])
    target_std = float(normalization["target_std"][0])
    batch = len(voltage)
    history = np.stack((voltage[:, :WINDOW_LENGTH], measured[:, :WINDOW_LENGTH]), axis=2)
    positive = np.zeros(batch, dtype=float)
    negative = np.zeros(batch, dtype=float)
    active = np.zeros(batch, dtype=bool)
    abnormal_run = np.zeros(batch, dtype=int)
    healthy_run = np.zeros(batch, dtype=int)
    alarm_samples = 0
    substitute_samples = 0
    output_samples = 0

    for sample_index in range(WINDOW_LENGTH, voltage.shape[1]):
        normalized = ((history - input_mean) / input_std).astype(np.float32)
        output = model(torch.from_numpy(normalized)).numpy()[:, 0]
        physical = output * target_std + target_mean
        error = measured[:, sample_index] - physical
        finite = np.isfinite(error)
        centered = error - float(cusum["center"])
        positive = np.where(
            finite,
            np.maximum(0.0, positive + centered - float(cusum["allowance"])),
            positive,
        )
        negative = np.where(
            finite,
            np.maximum(0.0, negative - centered - float(cusum["allowance"])),
            negative,
        )
        score = np.where(finite, np.maximum(positive, negative), np.inf)
        instantaneous = (~finite) | (np.abs(error) > residual_gate)
        abnormal = instantaneous | (score > float(cusum["threshold"]))
        previously_active = active.copy()
        abnormal_run = np.where((~active) & abnormal, abnormal_run + 1, 0)
        active |= abnormal_run >= SENSOR_ENTER
        healthy = finite & (np.abs(error) <= residual_gate) & (score <= float(cusum["threshold"]))
        healthy_run = np.where(previously_active & healthy, healthy_run + 1, 0)
        recovered = previously_active & (healthy_run >= SENSOR_EXIT)
        active[recovered] = False
        positive[recovered] = 0.0
        negative[recovered] = 0.0
        abnormal_run[active] = 0

        substitute = active | instantaneous
        alarm_samples += int(np.sum(active))
        substitute_samples += int(np.sum(substitute))
        output_samples += batch
        feedback = np.where(substitute, physical, measured[:, sample_index])
        next_sample = np.stack((voltage[:, sample_index], feedback), axis=1)
        history = np.concatenate((history[:, 1:], next_sample[:, None, :]), axis=1)

    return alarm_samples / output_samples, substitute_samples / output_samples


@torch.no_grad()
def independently_recompute_sensor_calibration(
    model: LSTMForecaster,
    config: dict[str, Any],
    data: np.lib.npyio.NpzFile,
) -> dict[str, Any]:
    validation_voltage = np.asarray(data["voltage"])[list(VALIDATION_IDS)].astype(float)
    validation_measured = np.asarray(data["y_measured"])[list(VALIDATION_IDS)].astype(float)
    norm = config["normalization"]
    input_mean = np.asarray(norm["input_mean"], dtype=float)
    input_std = np.asarray(norm["input_std"], dtype=float)
    target_mean = float(norm["target_mean"][0])
    target_std = float(norm["target_std"][0])

    windows = []
    for voltage_row, measured_row in zip(validation_voltage, validation_measured):
        features = np.column_stack((voltage_row, measured_row))
        row = np.stack(
            [features[start : start + WINDOW_LENGTH] for start in range(len(features) - WINDOW_LENGTH)]
        )
        windows.append(((row - input_mean) / input_std).astype(np.float32))
    flat_prediction = predict_physical(model, np.concatenate(windows), target_mean, target_std)
    prediction = flat_prediction.reshape(len(VALIDATION_IDS), -1)
    residual = validation_measured[:, WINDOW_LENGTH:] - prediction
    flat_residual = residual.reshape(-1)
    instant_threshold = float(np.percentile(np.abs(flat_residual), 99.9))
    residual_center = float(np.median(flat_residual))

    signed_ewmas = []
    absolute_ewmas = []
    for row in residual:
        signed, absolute = independent_temporal_ewma(row - residual_center, SENSOR_EWMA_ALPHA)
        signed_ewmas.extend(signed)
        absolute_ewmas.extend(absolute)

    grid = []
    selected_params: dict[str, float] | None = None
    selected_percentile = SENSOR_PERCENTILES[-1]
    selected_far = float("nan")
    selected_substitution = float("nan")
    for percentile in SENSOR_PERCENTILES:
        candidate = independent_cusum_parameters(residual, percentile)
        far, substitution = independent_guarded_validation(
            model,
            validation_voltage,
            validation_measured,
            norm,
            instant_threshold,
            candidate,
        )
        grid.append((percentile, candidate, far, substitution))
        if selected_params is None and far <= SENSOR_TARGET_FAR:
            selected_params = candidate
            selected_percentile = percentile
            selected_far = far
            selected_substitution = substitution
    if selected_params is None:
        selected_percentile, selected_params, selected_far, selected_substitution = grid[-1]

    return {
        "sample_count": int(residual.size),
        "instant_threshold": instant_threshold,
        "cusum_percentile": float(selected_percentile),
        "residual_center": residual_center,
        "center": float(selected_params["center"]),
        "allowance": float(selected_params["allowance"]),
        "threshold": float(selected_params["threshold"]),
        "signed_ewma_threshold": float(np.percentile(np.abs(signed_ewmas), 99.9)),
        "absolute_ewma_threshold": float(np.percentile(absolute_ewmas, 99.9)),
        "selected_far": float(selected_far),
        "substitution_rate": float(selected_substitution),
        "grid": grid,
    }


def independent_post_blanking_ewma(times: np.ndarray, residual: np.ndarray) -> np.ndarray:
    values = []
    ewma = 0.0
    for time_value, residual_value in zip(times, residual):
        if float(time_value) >= ARBITRATION_STARTUP_BLANKING_S:
            ewma = (1.0 - ARBITRATION_EWMA_ALPHA) * ewma + ARBITRATION_EWMA_ALPHA * float(residual_value)
            values.append(ewma)
    return np.asarray(values, dtype=float)


@torch.no_grad()
def independently_recompute_v3_calibration(
    main_model: LSTMForecaster,
    main_config: dict[str, Any],
    aux_model: AuxiliarySpeedEstimator,
    aux_config: dict[str, Any],
    data: np.lib.npyio.NpzFile,
) -> dict[str, Any]:
    main_norm = main_config["normalization"]
    aux_norm = aux_config["normalization"]
    agreement: list[float] = []
    aux_residual: list[float] = []
    ewma_values: list[float] = []

    for run_id in VALIDATION_IDS:
        trajectory = {
            "voltage": data["voltage"][run_id],
            "y_measured": data["y_measured"][run_id],
            "y_true": data["y_true"][run_id],
            "run_id": np.full(len(data["time"]), run_id),
        }
        main_inputs, _, _ = build_training_sequences(trajectory, WINDOW_LENGTH, 1)
        main_inputs = (
            (main_inputs - np.asarray(main_norm["input_mean"])) / np.asarray(main_norm["input_std"])
        ).astype(np.float32)
        aux_inputs, aux_targets = build_auxiliary_sequences(
            data["voltage"][run_id], data["current"][run_id], data["y_true"][run_id], WINDOW_LENGTH
        )
        aux_inputs, _ = normalize_auxiliary(aux_inputs, aux_targets, aux_norm)
        y_main = predict_physical(
            main_model,
            main_inputs,
            float(main_norm["target_mean"][0]),
            float(main_norm["target_std"][0]),
        )
        y_aux = predict_physical(
            aux_model,
            aux_inputs,
            float(aux_norm["target_mean"][0]),
            float(aux_norm["target_std"][0]),
        )
        measured = np.asarray(data["y_measured"][run_id, WINDOW_LENGTH:], dtype=float)
        residual = np.abs(measured - y_aux)
        agreement.extend(np.abs(y_main - y_aux))
        aux_residual.extend(residual)
        ewma_values.extend(
            independent_post_blanking_ewma(np.asarray(data["time"])[WINDOW_LENGTH:], residual)
        )

    agreement_arr = np.asarray(agreement, dtype=float)
    aux_residual_arr = np.asarray(aux_residual, dtype=float)
    ewma_arr = np.asarray(ewma_values, dtype=float)
    return {
        "agreement_threshold": float(np.percentile(agreement_arr, 90.0)),
        "param_mismatch_threshold": float(np.percentile(ewma_arr, 99.9)),
        "param_mismatch_recovery_threshold": float(np.percentile(ewma_arr, 95.0)),
        "aux_recovery_gate": float(np.percentile(aux_residual_arr, 99.9)),
        "agreement_count": int(len(agreement_arr)),
        "aux_recovery_count": int(len(aux_residual_arr)),
        "ewma_count": int(len(ewma_arr)),
    }


def verify_independent_calibrations(pairs: dict[int, PairArtifacts]) -> None:
    check(DATASET.is_file(), "frozen dataset is missing")
    with np.load(DATASET, allow_pickle=False) as data:
        check(exact_int_tuple(data["split_run_ids_validation"], "dataset validation IDs") == VALIDATION_IDS, "dataset validation split drift")
        check(exact_int_tuple(data["split_run_ids_test"], "dataset test IDs") == TEST_IDS, "dataset test split drift")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for seed, pair in pairs.items():
                main_model, main_config = load_main_model(pair)
                aux_model, aux_config = load_aux_model(pair)
                sensor_saved = load_json(pair.sensor_calibration)
                v3_saved = load_json(pair.v3_calibration)

                reconstructed_sensor = independently_recompute_sensor_calibration(main_model, main_config, data)
                sensor_values = sensor_saved.get("computed_values")
                check(isinstance(sensor_values, dict), f"P{seed} sensor computed_values missing")
                for key in (
                    "instant_threshold",
                    "cusum_percentile",
                    "residual_center",
                    "center",
                    "allowance",
                    "threshold",
                    "signed_ewma_threshold",
                    "absolute_ewma_threshold",
                ):
                    assert_close(reconstructed_sensor[key], sensor_values[key], f"P{seed} sensor {key}")
                check(int(sensor_saved.get("sample_count", -1)) == reconstructed_sensor["sample_count"], f"P{seed} sensor sample count mismatch")
                saved_false_alarm = sensor_saved.get("clean_validation_false_alarm_statistics", {})
                assert_close(
                    reconstructed_sensor["selected_far"],
                    saved_false_alarm.get("selected_debounced_sensor_alarm_rate"),
                    f"P{seed} sensor validation FAR",
                )
                assert_close(
                    reconstructed_sensor["substitution_rate"],
                    saved_false_alarm.get("immediate_substitution_rate"),
                    f"P{seed} sensor validation substitution rate",
                )
                saved_grid = sensor_saved.get("calibration_grid")
                check(isinstance(saved_grid, list) and len(saved_grid) == len(SENSOR_PERCENTILES), f"P{seed} sensor calibration grid changed")
                for saved_row, rebuilt in zip(saved_grid, reconstructed_sensor["grid"]):
                    percentile, candidate, far, _ = rebuilt
                    assert_close(saved_row["percentile"], percentile, f"P{seed} CUSUM grid percentile")
                    assert_close(saved_row["cusum_threshold"], candidate["threshold"], f"P{seed} CUSUM grid threshold p{percentile}")
                    assert_close(saved_row["validation_false_alarm_rate"], far, f"P{seed} CUSUM grid FAR p{percentile}")

                reconstructed_v3 = independently_recompute_v3_calibration(
                    main_model, main_config, aux_model, aux_config, data
                )
                v3_values = v3_saved.get("computed_values")
                check(isinstance(v3_values, dict), f"P{seed} V3 computed_values missing")
                for key in (
                    "agreement_threshold",
                    "param_mismatch_threshold",
                    "param_mismatch_recovery_threshold",
                    "aux_recovery_gate",
                ):
                    assert_close(reconstructed_v3[key], v3_values[key], f"P{seed} V3 {key}")
                assert_close(v3_values.get("ewma_alpha"), ARBITRATION_EWMA_ALPHA, f"P{seed} V3 EWMA alpha")
                assert_close(v3_values.get("startup_blanking_time"), ARBITRATION_STARTUP_BLANKING_S, f"P{seed} V3 startup blanking")
                check(int(v3_values.get("param_mismatch_recovery_count", -1)) == ARBITRATION_RECOVERY_COUNT, f"P{seed} V3 recovery count drift")
                check(v3_values.get("aux_speed_bounds") == AUX_SPEED_BOUNDS, f"P{seed} V3 auxiliary speed bounds drift")
                counts = v3_saved.get("sample_counts")
                check(isinstance(counts, dict), f"P{seed} V3 sample counts missing")
                check(int(counts.get("agreement", -1)) == reconstructed_v3["agreement_count"], f"P{seed} agreement sample count mismatch")
                check(int(counts.get("aux_recovery", -1)) == reconstructed_v3["aux_recovery_count"], f"P{seed} auxiliary residual sample count mismatch")
                check(int(counts.get("ewma_after_startup_blanking", -1)) == reconstructed_v3["ewma_count"], f"P{seed} EWMA sample count mismatch")
        finally:
            torch.set_num_threads(previous_threads)


def expected_matrix_keys() -> set[tuple[int, str, str, int]]:
    return set(product(TRAINING_SEEDS, CONTROLLERS, SCENARIOS, SIMULATION_SEEDS))


def verify_evaluation_manifest(runs: pd.DataFrame, pair_manifest: dict[str, Any]) -> None:
    manifest = load_json(EVALUATION_MANIFEST)
    check(manifest.get("study") == STUDY, "evaluation manifest belongs to the wrong study")
    check(int(manifest.get("run_count", -1)) == EXPECTED_RUN_COUNT, "evaluation manifest run count is not 150")
    check(tuple(manifest.get("training_seeds", [])) == TRAINING_SEEDS, "evaluation manifest training seeds changed")
    check(tuple(manifest.get("simulation_seeds", [])) == SIMULATION_SEEDS, "evaluation manifest simulation seeds changed")
    check(tuple(manifest.get("controllers", [])) == CONTROLLERS, "evaluation manifest controllers changed")
    check(tuple(manifest.get("scenarios", [])) == SCENARIOS, "evaluation manifest scenarios changed")
    check(manifest.get("simulation_seed_freeze_sha256") == sha256(SIMULATION_SEED_FREEZE), "evaluation manifest seed-freeze hash mismatch")
    check(
        manifest.get("frozen_v3_evaluator_sha256")
        == sha256(PROJECT / "scripts" / "evaluate_v3_closed_loop.py"),
        "evaluation manifest frozen V3 evaluator hash mismatch",
    )
    output = manifest.get("output")
    check(isinstance(output, dict), "evaluation output binding missing")
    check(output.get("runs") == "results/training_seed_robustness/training_seed_runs.csv", "evaluation manifest raw-matrix path changed")
    check(output.get("runs_sha256") == sha256(RUNS_PATH), "evaluation manifest raw-matrix hash mismatch")

    declared_pairs = manifest.get("pair_manifests")
    check(isinstance(declared_pairs, dict) and set(declared_pairs) == {str(seed) for seed in TRAINING_SEEDS}, "evaluation pair manifest bindings incomplete")
    actual_pair_manifest_hash = sha256(MODEL_PAIR_MANIFEST)
    for seed in TRAINING_SEEDS:
        entry = declared_pairs[str(seed)]
        check(entry.get("sha256") == actual_pair_manifest_hash, f"evaluation manifest P{seed} pair-manifest hash mismatch")
        pair = pair_manifest["pairs"][f"P{seed}"]
        check(entry.get("main_model_sha256") == pair["main"]["weights_sha256"], f"evaluation manifest P{seed} main hash mismatch")
        check(entry.get("aux_model_sha256") == pair["auxiliary"]["weights_sha256"], f"evaluation manifest P{seed} auxiliary hash mismatch")
        check(entry.get("sensor_calibration_sha256") == pair["sensor_calibration"]["sha256"], f"evaluation manifest P{seed} sensor calibration hash mismatch")
        check(entry.get("v3_calibration_sha256") == pair["v3_calibration"]["sha256"], f"evaluation manifest P{seed} V3 calibration hash mismatch")
        check(entry.get("pair_config_sha256") == pair["pair_config"]["sha256"], f"evaluation manifest P{seed} pair config hash mismatch")
        check(
            entry.get("protocol_manifest_sha256") == pair_manifest["protocol_manifest"]["sha256"],
            f"evaluation manifest P{seed} protocol-manifest hash mismatch",
        )

    check(len(runs) == int(manifest["run_count"]), "evaluation manifest/raw matrix row-count mismatch")


def verify_required_model_metrics(
    pairs: dict[int, PairArtifacts],
) -> tuple[bool, bool]:
    """Independently establish that every saved learned model has finite required metrics."""
    main_finite = True
    aux_finite = True
    for seed in TRAINING_SEEDS:
        main_metrics_path = (
            PROJECT / "results" / "metrics" / "lstm_test_metrics.json"
            if seed == 2026
            else RESULT_ROOT / "training" / f"main_seed{seed}_metrics.json"
        )
        aux_metrics_path = (
            PROJECT / "results" / "metrics" / "v2_auxiliary_model_metrics.json"
            if seed == 2026
            else RESULT_ROOT / "training" / f"aux_seed{seed}_metrics.json"
        )
        main_metrics = load_json(main_metrics_path)
        aux_metrics = load_json(aux_metrics_path)
        main_config = load_json(pairs[seed].main_config)

        main_test = main_metrics.get("test")
        check(isinstance(main_test, dict), f"P{seed} main test metrics missing")
        horizons = main_metrics.get("recursive_key_horizons")
        if isinstance(horizons, dict):
            recursive = [horizons.get("H5"), horizons.get("H10"), horizons.get("H15")]
        else:
            series = main_metrics.get("recursive_rmse_by_horizon")
            check(isinstance(series, list) and len(series) >= 15, f"P{seed} main recursive metrics incomplete")
            recursive = [series[4], series[9], series[14]]
        main_values = [
            main_test.get("rmse"),
            main_test.get("mae"),
            main_test.get("r2"),
            *recursive,
            main_metrics.get("best_epoch", main_config["training"].get("best_epoch")),
        ]
        this_main_finite = bool(np.isfinite(np.asarray(main_values, dtype=float)).all())
        main_finite = main_finite and this_main_finite

        aux_values = [
            aux_metrics.get("test_rmse"),
            aux_metrics.get("test_mae"),
            aux_metrics.get("test_r2"),
        ]
        this_aux_finite = bool(np.isfinite(np.asarray(aux_values, dtype=float)).all())
        aux_finite = aux_finite and this_aux_finite
    return main_finite, aux_finite


def sample_sd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if len(array) < 2:
        return float("nan")
    return float(np.std(array, ddof=1))


def variability(values: Iterable[Any]) -> dict[str, float]:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "mean": np.nan,
            "sample_sd": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
        }
    return {
        "mean": float(np.mean(array)),
        "sample_sd": sample_sd(array),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "range": float(np.ptp(array)),
    }


def check_frame_matches(
    saved: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
    sort_by: list[str],
) -> None:
    missing = [column for column in expected.columns if column not in saved.columns]
    check(not missing, f"{label} missing columns: {missing}")
    saved_view = saved.loc[:, expected.columns].sort_values(sort_by).reset_index(drop=True)
    expected_view = expected.sort_values(sort_by).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            saved_view,
            expected_view,
            check_dtype=False,
            check_exact=False,
            rtol=MODEL_RTOL,
            atol=MODEL_ATOL,
        )
    except AssertionError as exc:
        raise VerificationError(f"{label} disagrees with independent reconstruction: {exc}") from exc


def read_analysis_csv(name: str) -> pd.DataFrame:
    path = ANALYSIS_TABLE_PATHS[name]
    check(path.is_file(), f"analysis table is missing: {path.relative_to(PROJECT)}")
    return pd.read_csv(path, float_precision="round_trip")


def expected_main_metric_tables(
    pairs: dict[int, PairArtifacts],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    metric_names = ("rmse", "mae", "r2", "h5_rmse", "h10_rmse", "h15_rmse", "best_epoch")
    for seed in TRAINING_SEEDS:
        metrics_path = (
            PROJECT / "results" / "metrics" / "lstm_test_metrics.json"
            if seed == 2026
            else RESULT_ROOT / "training" / f"main_seed{seed}_metrics.json"
        )
        metrics = load_json(metrics_path)
        config = load_json(pairs[seed].main_config)
        if "training_seed" in metrics:
            check(int(metrics["training_seed"]) == seed, f"P{seed} main metrics seed mismatch")
        test = metrics.get("test")
        check(isinstance(test, dict), f"P{seed} main test metrics missing")
        recursive = metrics.get("recursive_key_horizons")
        if isinstance(recursive, dict):
            h5 = recursive.get("H5")
            h10 = recursive.get("H10")
            h15 = recursive.get("H15")
        else:
            series = metrics.get("recursive_rmse_by_horizon")
            check(isinstance(series, list) and len(series) >= 15, f"P{seed} main recursive metrics incomplete")
            h5, h10, h15 = series[4], series[9], series[14]
        training = config.get("training")
        check(isinstance(training, dict) and int(training.get("seed", -1)) == seed, f"P{seed} main config seed mismatch")
        rows.append(
            {
                "training_seed": seed,
                "rmse": float(test["rmse"]),
                "mae": float(test["mae"]),
                "r2": float(test["r2"]),
                "h5_rmse": float(h5),
                "h10_rmse": float(h10),
                "h15_rmse": float(h15),
                "best_epoch": int(metrics.get("best_epoch", training.get("best_epoch"))),
                "metrics_path": metrics_path.relative_to(PROJECT).as_posix(),
            }
        )
    by_seed = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [{"metric": metric, **variability(by_seed[metric])} for metric in metric_names]
    )
    return by_seed, summary


def expected_aux_metric_tables(
    pairs: dict[int, PairArtifacts],
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        metrics_path = (
            PROJECT / "results" / "metrics" / "v2_auxiliary_model_metrics.json"
            if seed == 2026
            else RESULT_ROOT / "training" / f"aux_seed{seed}_metrics.json"
        )
        metrics = load_json(metrics_path)
        config = load_json(pairs[seed].aux_config)
        if "training_seed" in metrics:
            check(int(metrics["training_seed"]) == seed, f"P{seed} auxiliary metrics seed mismatch")
        training = config.get("training")
        check(isinstance(training, dict) and int(training.get("seed", -1)) == seed, f"P{seed} auxiliary config seed mismatch")
        per_test = metrics.get("per_trajectory_test")
        check(isinstance(per_test, list) and per_test, f"P{seed} auxiliary per-trajectory test metrics missing")
        traj_stats = variability(float(row["rmse"]) for row in per_test)
        metric_rows.append(
            {
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
                "metrics_path": metrics_path.relative_to(PROJECT).as_posix(),
            }
        )
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

    by_seed = pd.DataFrame(metric_rows)
    summary_metrics = (
        "test_rmse",
        "test_mae",
        "test_r2",
        "best_epoch",
        "per_trajectory_rmse_mean",
        "per_trajectory_rmse_sample_sd",
        "per_trajectory_rmse_range",
    )
    summary = pd.DataFrame(
        [{"metric": metric, **variability(by_seed[metric])} for metric in summary_metrics]
    )
    per_trajectory = pd.DataFrame(trajectory_rows)

    c3 = runs[
        (runs["controller"] == "C3_arbitration_MPC")
        & runs["scenario"].isin(["load_disturbance", "parameter_variation"])
    ].copy()
    transfer_fields = (
        "aux_virtual_rmse",
        "aux_virtual_mae",
        "main_virtual_rmse",
        "selected_virtual_rmse",
        "event_frac_aux",
        "event_frac_fb",
        "sub_fraction",
        "sub_samples",
    )
    for field in transfer_fields:
        if field not in c3.columns:
            c3[field] = np.nan
    transfer_rows: list[dict[str, Any]] = []
    for (seed, scenario), group in c3.groupby(["training_seed", "scenario"], sort=True):
        row: dict[str, Any] = {
            "training_seed": int(seed),
            "scenario": str(scenario),
            "run_count": int(len(group)),
        }
        for field in transfer_fields:
            stats = variability(group[field])
            for statistic, value in stats.items():
                row[f"{field}_{statistic}"] = value
        transfer_rows.append(row)
    return by_seed, summary, per_trajectory, pd.DataFrame(transfer_rows)


def expected_calibration_tables(
    pairs: dict[int, PairArtifacts],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        sensor = load_json(pairs[seed].sensor_calibration)
        v3 = load_json(pairs[seed].v3_calibration)
        sensor_values = sensor.get("computed_values")
        v3_values = v3.get("computed_values")
        v3_stats = v3.get("statistics")
        sensor_far = sensor.get("clean_validation_false_alarm_statistics")
        check(
            all(isinstance(value, dict) for value in (sensor_values, v3_values, v3_stats, sensor_far)),
            f"P{seed} calibration analysis schema incomplete",
        )
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
                "sensor_validation_alarm_rate": float(sensor_far["selected_debounced_sensor_alarm_rate"]),
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
    by_seed = pd.DataFrame(rows)
    variability_rows = []
    for column in by_seed.columns:
        if column == "training_seed":
            continue
        stats = variability(by_seed[column])
        mean = stats["mean"]
        cv = stats["sample_sd"] / abs(mean) if np.isfinite(mean) and abs(mean) > 1e-12 else np.nan
        variability_rows.append(
            {"calibration_statistic": column, **stats, "coefficient_of_variation": cv}
        )
    return by_seed, pd.DataFrame(variability_rows)


def expected_load_tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = runs[runs["scenario"] == "load_disturbance"]
    rows: list[dict[str, Any]] = []
    for (training_seed, simulation_seed), group in subset.groupby(
        ["training_seed", "simulation_seed"], sort=True
    ):
        b = group[group["controller"] == "B_plain_MPC"].iloc[0]
        c3 = group[group["controller"] == "C3_arbitration_MPC"].iloc[0]
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
    table = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        group = table[table["training_seed"] == seed]
        total_false_entries = int(group["c3_false_reliability_entries"].sum())
        mean_penalty = float(group["tracking_penalty_c3_minus_b"].mean())
        summary_rows.append(
            {
                "training_seed": seed,
                "mean_tracking_penalty_c3_minus_b": mean_penalty,
                "total_false_reliability_entries": total_false_entries,
                "mean_substitution_fraction": float(group["c3_substitution_fraction"].mean()),
                "tracking_degradation_consistent_with_frozen_limitation": bool(mean_penalty > 0.0),
                "false_entry_weakness_consistent_with_frozen_limitation": bool(total_false_entries > 0),
            }
        )
    return table, pd.DataFrame(summary_rows)


def expected_parameter_tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = runs[runs["scenario"] == "parameter_variation"]
    rows: list[dict[str, Any]] = []
    for (training_seed, simulation_seed), group in subset.groupby(
        ["training_seed", "simulation_seed"], sort=True
    ):
        b = group[group["controller"] == "B_plain_MPC"].iloc[0]
        c3 = group[group["controller"] == "C3_arbitration_MPC"].iloc[0]
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
    table = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        group = table[table["training_seed"] == seed]
        summary_rows.append(
            {
                "training_seed": seed,
                "b_fault_window_rmse_mean": float(group["b_fault_window_rmse"].mean()),
                "b_fault_window_rmse_sample_sd": sample_sd(group["b_fault_window_rmse"]),
                "c3_fault_window_rmse_mean": float(group["c3_fault_window_rmse"].mean()),
                "c3_fault_window_rmse_sample_sd": sample_sd(group["c3_fault_window_rmse"]),
                "c3_minus_b_mean": float(group["c3_minus_b_fault_window_rmse"].mean()),
                "c3_minus_b_sample_sd": sample_sd(group["c3_minus_b_fault_window_rmse"]),
                "c3_substitution_fraction_mean": float(group["c3_substitution_fraction"].mean()),
                "c3_substitution_fraction_sample_sd": sample_sd(group["c3_substitution_fraction"]),
                "c3_event_aux_fraction_mean": float(group["c3_event_aux_fraction"].mean()),
                "c3_event_aux_fraction_sample_sd": sample_sd(group["c3_event_aux_fraction"]),
                "c3_event_fallback_fraction_mean": float(group["c3_event_fallback_fraction"].mean()),
                "c3_event_fallback_fraction_sample_sd": sample_sd(group["c3_event_fallback_fraction"]),
            }
        )
    return table, pd.DataFrame(summary_rows)


def expected_hierarchical_bootstrap(
    runs: pd.DataFrame,
    *,
    analysis_seed: int,
    replicates: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(analysis_seed)
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        scenario_runs = runs[runs["scenario"] == scenario]
        by_seed: dict[int, np.ndarray] = {}
        for seed in TRAINING_SEEDS:
            subset = scenario_runs[scenario_runs["training_seed"].astype(int) == seed]
            pivot = subset.pivot(
                index="simulation_seed", columns="controller", values="fault_window_rmse"
            ).sort_index()
            check(
                tuple(int(value) for value in pivot.index) == SIMULATION_SEEDS,
                f"bootstrap P{seed}/{scenario} simulation-seed set changed",
            )
            by_seed[seed] = (
                pivot["C3_arbitration_MPC"].to_numpy(float)
                - pivot["B_plain_MPC"].to_numpy(float)
            )

        seed_means = np.asarray([np.mean(by_seed[seed]) for seed in TRAINING_SEEDS])
        bootstrap_values = np.empty(replicates, dtype=float)
        for index in range(replicates):
            sampled_training_seeds = rng.choice(
                TRAINING_SEEDS, size=len(TRAINING_SEEDS), replace=True
            )
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


def verify_run_matrix(
    pairs: dict[int, PairArtifacts], pair_manifest: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    check(RUNS_PATH.is_file(), "training_seed_runs.csv does not exist")
    runs = pd.read_csv(RUNS_PATH)
    check(len(runs) == EXPECTED_RUN_COUNT, f"expected exactly 150 rows, found {len(runs)}")

    required_columns = {
        "training_seed",
        "pair_manifest_sha256",
        "controller",
        "scenario",
        "simulation_seed",
        "seed",
        "run_complete",
        "main_model_sha256",
        "main_config_sha256",
        "aux_model_sha256",
        "aux_config_sha256",
        "sensor_calibration_sha256",
        "v3_calibration_sha256",
        "pair_config_sha256",
        "calibration_sha256",
        "sensor_calibration_algorithm_sha256",
        "v3_calibration_algorithm_sha256",
        "calibration_algorithm_sha256",
        "overall_rmse",
        "overall_mae",
        "fault_window_rmse",
        "sub_fraction",
        "reliability_entries",
        "source_switch_count",
        "physical_fraction",
        "main_fraction",
        "auxiliary_fraction",
        "fallback_fraction",
        "optimizer_failures",
        "main_prediction_failures",
        "aux_prediction_failures",
        "total_failure_count",
        "voltage_violations",
        "slew_violations",
        "rate_violations",
        "nonfinite_events",
        "total_safety_violation_count",
        "runtime_s",
    }
    missing = sorted(required_columns - set(runs.columns))
    check(not missing, f"raw matrix is missing required columns: {missing}")

    keys = {
        (int(row.training_seed), str(row.controller), str(row.scenario), int(row.simulation_seed))
        for row in runs.itertuples(index=False)
    }
    check(keys == expected_matrix_keys(), "raw matrix keys differ from the exact 3x2x5x5 preregistered product")
    check(
        not runs.duplicated(["training_seed", "controller", "scenario", "simulation_seed"]).any(),
        "raw matrix contains duplicate primary keys",
    )
    check(set(runs["training_seed"].astype(int)) == set(TRAINING_SEEDS), "raw matrix training seeds are not exact")
    check(set(runs["simulation_seed"].astype(int)) == set(SIMULATION_SEEDS), "raw matrix simulation seeds are not exact")
    check(set(runs["controller"].astype(str)) == set(CONTROLLERS), "raw matrix controller set is not exact")
    check(set(runs["scenario"].astype(str)) == set(SCENARIOS), "raw matrix scenario set is not exact")
    check(np.array_equal(runs["seed"].astype(int), runs["simulation_seed"].astype(int)), "legacy seed alias differs from simulation_seed")
    check(set(runs["simulation_seed"].astype(int)).isdisjoint(RESERVED_SEEDS), "reserved 39026-39030 seed family appears in study matrix")

    run_complete = runs["run_complete"]
    if run_complete.dtype == bool:
        complete_mask = run_complete.to_numpy()
    else:
        complete_mask = run_complete.astype(str).str.lower().isin({"true", "1"}).to_numpy()
    check(bool(np.all(complete_mask)), "one or more closed-loop rows are incomplete")

    for seed, pair in pairs.items():
        subset = runs[runs["training_seed"].astype(int) == seed]
        check(len(subset) == 50, f"P{seed} must have exactly 50 rows")
        algorithm_hashes = pair_manifest["algorithm_hashes"]
        sensor_algorithm_hash = str(algorithm_hashes["sensor_calibration_source"]["sha256"])
        v3_algorithm_hash = str(algorithm_hashes["v3_calibration_source"]["sha256"])
        combined_algorithm_hash = hashlib.sha256(
            json.dumps(
                {
                    "sensor_calibration_source": sensor_algorithm_hash,
                    "v3_calibration_source": v3_algorithm_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        expected_hashes = {
            "pair_manifest_sha256": sha256(MODEL_PAIR_MANIFEST),
            "main_model_sha256": pair.main_weights_sha256,
            "main_config_sha256": pair.main_config_sha256,
            "aux_model_sha256": pair.aux_weights_sha256,
            "aux_config_sha256": pair.aux_config_sha256,
            "sensor_calibration_sha256": pair.sensor_calibration_sha256,
            "v3_calibration_sha256": pair.v3_calibration_sha256,
            "pair_config_sha256": pair.pair_config_sha256,
            "calibration_sha256": pair.pair_config_sha256,
            "sensor_calibration_algorithm_sha256": sensor_algorithm_hash,
            "v3_calibration_algorithm_sha256": v3_algorithm_hash,
            "calibration_algorithm_sha256": combined_algorithm_hash,
        }
        for column, expected in expected_hashes.items():
            check(subset[column].nunique(dropna=False) == 1, f"P{seed} mixes {column} across rows")
            check(str(subset[column].iloc[0]) == expected, f"P{seed} row {column} does not match model-pair manifest")

        for scenario in SCENARIOS:
            for simulation_seed in SIMULATION_SEEDS:
                paired = subset[
                    (subset["scenario"] == scenario)
                    & (subset["simulation_seed"].astype(int) == simulation_seed)
                ]
                check(set(paired["controller"]) == set(CONTROLLERS) and len(paired) == 2, f"P{seed}/{scenario}/{simulation_seed} is not a complete B/C3 pair")
                for column in expected_hashes:
                    check(paired[column].nunique(dropna=False) == 1, f"P{seed}/{scenario}/{simulation_seed} mixes {column} between B/C3")

    required_finite_numeric = (
        "training_seed",
        "simulation_seed",
        "overall_rmse",
        "overall_mae",
        "fault_window_rmse",
        "sub_fraction",
        "reliability_entries",
        "source_switch_count",
        "physical_fraction",
        "main_fraction",
        "auxiliary_fraction",
        "fallback_fraction",
        "optimizer_failures",
        "main_prediction_failures",
        "aux_prediction_failures",
        "total_failure_count",
        "voltage_violations",
        "rate_violations",
        "slew_violations",
        "nonfinite_events",
        "total_safety_violation_count",
        "runtime_s",
    )
    check(
        np.isfinite(runs.loc[:, required_finite_numeric].to_numpy(dtype=float)).all(),
        "required numeric raw-matrix fields contain NaN/Inf",
    )

    for column in (
        "optimizer_failures",
        "main_prediction_failures",
        "aux_prediction_failures",
        "total_failure_count",
        "voltage_violations",
        "rate_violations",
        "slew_violations",
        "nonfinite_events",
        "total_safety_violation_count",
    ):
        values = runs[column].to_numpy(dtype=float)
        check(np.all(values >= 0) and np.all(values == np.floor(values)), f"{column} contains invalid counts")

    recomputed_failures = (
        runs["optimizer_failures"].astype(int)
        + runs["main_prediction_failures"].astype(int)
        + runs["aux_prediction_failures"].astype(int)
    )
    check(np.array_equal(recomputed_failures, runs["total_failure_count"].astype(int)), "total_failure_count arithmetic mismatch")
    check(np.array_equal(runs["rate_violations"].astype(int), runs["slew_violations"].astype(int)), "rate/slew violation aliases diverge")
    recomputed_safety = (
        runs["voltage_violations"].astype(int)
        + runs["rate_violations"].astype(int)
        + runs["nonfinite_events"].astype(int)
    )
    check(np.array_equal(recomputed_safety, runs["total_safety_violation_count"].astype(int)), "total_safety_violation_count arithmetic mismatch")

    verify_evaluation_manifest(runs, pair_manifest)

    headline: dict[str, Any] = {}
    for scenario in HEADLINE_SCENARIOS:
        headline[scenario] = {}
        for seed in TRAINING_SEEDS:
            subset = runs[
                (runs["training_seed"].astype(int) == seed)
                & (runs["scenario"] == scenario)
            ]
            pivot = subset.pivot(index="simulation_seed", columns="controller", values="fault_window_rmse")
            check(tuple(sorted(int(value) for value in pivot.index)) == SIMULATION_SEEDS, f"P{seed}/{scenario} paired seed set changed")
            difference = (
                pivot["C3_arbitration_MPC"].to_numpy(dtype=float)
                - pivot["B_plain_MPC"].to_numpy(dtype=float)
            )
            check(len(difference) == 5 and np.isfinite(difference).all(), f"P{seed}/{scenario} paired difference arithmetic invalid")
            headline[scenario][str(seed)] = {
                "paired_differences": difference.tolist(),
                "mean_difference": float(np.mean(difference)),
                "favorable": bool(float(np.mean(difference)) < 0.0),
            }

    failure_columns = (
        "optimizer_failures",
        "main_prediction_failures",
        "aux_prediction_failures",
        "voltage_violations",
        "slew_violations",
        "nonfinite_events",
    )
    safety_totals = {column: int(runs[column].sum()) for column in failure_columns}
    no_numerical_or_safety_failure = all(value == 0 for value in safety_totals.values())
    main_metrics_finite, aux_metrics_finite = verify_required_model_metrics(pairs)
    no_model_pair_numerical_or_safety_failure = bool(
        no_numerical_or_safety_failure and main_metrics_finite and aux_metrics_finite
    )
    headline_all_favorable = all(
        headline[scenario][str(seed)]["favorable"]
        for scenario in HEADLINE_SCENARIOS
        for seed in TRAINING_SEEDS
    )
    load_direction: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        subset = runs[
            (runs["training_seed"].astype(int) == seed)
            & (runs["scenario"] == "load_disturbance")
        ]
        pivot = subset.pivot(index="simulation_seed", columns="controller", values="fault_window_rmse")
        difference = (
            pivot["C3_arbitration_MPC"].to_numpy(dtype=float)
            - pivot["B_plain_MPC"].to_numpy(dtype=float)
        )
        mean_penalty = float(np.mean(difference))
        c3_rows = subset[subset["controller"] == "C3_arbitration_MPC"]
        total_false_entries = int(c3_rows["reliability_entries"].astype(int).sum())
        load_direction[str(seed)] = {
            "paired_differences": difference.tolist(),
            "mean_tracking_penalty_c3_minus_b": mean_penalty,
            "total_false_reliability_entries": total_false_entries,
            "tracking_degradation_consistent_with_frozen_limitation": bool(mean_penalty > 0.0),
            "false_entry_weakness_consistent_with_frozen_limitation": bool(total_false_entries > 0),
        }
    load_weakness_consistent = all(
        load_direction[str(seed)]["tracking_degradation_consistent_with_frozen_limitation"]
        and load_direction[str(seed)]["false_entry_weakness_consistent_with_frozen_limitation"]
        for seed in TRAINING_SEEDS
    )
    verdict = (
        "ROBUST ACROSS TRAINING SEEDS"
        if headline_all_favorable
        and no_model_pair_numerical_or_safety_failure
        and load_weakness_consistent
        else "TRAINING-SEED-SENSITIVE"
    )
    recomputed = {
        "headline": headline,
        "load_direction": load_direction,
        "safety_totals": safety_totals,
        "headline_all_favorable": headline_all_favorable,
        "run_safety_ok": no_numerical_or_safety_failure,
        "main_metrics_finite": main_metrics_finite,
        "aux_metrics_finite": aux_metrics_finite,
        "no_model_pair_numerical_or_safety_failure": no_model_pair_numerical_or_safety_failure,
        "load_weakness_consistent": load_weakness_consistent,
        "verdict": verdict,
    }
    return runs, recomputed


def verify_analysis_summary(recomputed: dict[str, Any]) -> None:
    """Verify the analyzer's machine-readable output against independent arithmetic."""
    check(ANALYSIS_SUMMARY.is_file(), "analysis/analysis_summary.json is missing")
    summary = load_json(ANALYSIS_SUMMARY)
    check(summary.get("study") == STUDY, "analysis summary belongs to another study")
    check(tuple(summary.get("training_seeds", [])) == TRAINING_SEEDS, "analysis training seed set changed")
    check(tuple(summary.get("simulation_seeds", [])) == SIMULATION_SEEDS, "analysis simulation seed set changed")
    check(tuple(summary.get("controllers", [])) == CONTROLLERS, "analysis controller set changed")
    check(tuple(summary.get("scenarios", [])) == SCENARIOS, "analysis scenario set changed")
    check(int(summary.get("run_count", -1)) == EXPECTED_RUN_COUNT, "analysis run count is not 150")

    source_hashes = summary.get("source_hashes")
    check(isinstance(source_hashes, dict), "analysis source hash block missing")
    check(source_hashes.get("training_seed_runs_csv") == sha256(RUNS_PATH), "analysis raw-matrix source hash mismatch")
    check(source_hashes.get("model_pair_manifest_json") == sha256(MODEL_PAIR_MANIFEST), "analysis model-pair source hash mismatch")
    check(source_hashes.get("simulation_seed_freeze_json") == sha256(SIMULATION_SEED_FREEZE), "analysis seed-freeze source hash mismatch")
    check(source_hashes.get("training_seed_evaluation_manifest_json") == sha256(EVALUATION_MANIFEST), "analysis evaluation-manifest source hash mismatch")

    seed_provenance = str(summary.get("seed_provenance_conclusion", "")).lower()
    check("no pre-study use" in seed_provenance, "analysis does not preserve unused-before-study seed provenance")

    safety = summary.get("safety")
    check(isinstance(safety, dict), "analysis safety block missing")
    saved_totals = safety.get("totals_across_150_runs")
    check(isinstance(saved_totals, dict), "analysis safety totals missing")
    for key, expected in recomputed["safety_totals"].items():
        check(int(saved_totals.get(key, -1)) == expected, f"analysis safety total mismatch for {key}")
    check(int(saved_totals.get("incomplete_runs", -1)) == 0, "analysis reports incomplete runs")
    check(bool(safety.get("all_required_safety_and_numerical_counts_zero")) == recomputed["run_safety_ok"], "analysis run safety boolean mismatch")
    check(bool(safety.get("main_required_metrics_finite")) == recomputed["main_metrics_finite"], "analysis main-metric finiteness mismatch")
    check(bool(safety.get("auxiliary_required_metrics_finite")) == recomputed["aux_metrics_finite"], "analysis auxiliary-metric finiteness mismatch")

    saved_load = summary.get("load_disturbance")
    check(isinstance(saved_load, list) and len(saved_load) == len(TRAINING_SEEDS), "analysis load summary incomplete")
    by_seed = {int(row["training_seed"]): row for row in saved_load}
    check(set(by_seed) == set(TRAINING_SEEDS), "analysis load summary seed set changed")
    for seed in TRAINING_SEEDS:
        expected = recomputed["load_direction"][str(seed)]
        row = by_seed[seed]
        assert_close(
            row["mean_tracking_penalty_c3_minus_b"],
            expected["mean_tracking_penalty_c3_minus_b"],
            f"analysis P{seed} load tracking penalty",
        )
        check(
            int(row["total_false_reliability_entries"])
            == int(expected["total_false_reliability_entries"]),
            f"analysis P{seed} false-entry total mismatch",
        )
        check(
            bool(row["tracking_degradation_consistent_with_frozen_limitation"])
            == expected["tracking_degradation_consistent_with_frozen_limitation"],
            f"analysis P{seed} load-direction classification mismatch",
        )
        check(
            bool(row["false_entry_weakness_consistent_with_frozen_limitation"])
            == expected["false_entry_weakness_consistent_with_frozen_limitation"],
            f"analysis P{seed} false-entry weakness classification mismatch",
        )

    saved_verdict = summary.get("verdict")
    check(isinstance(saved_verdict, dict), "analysis verdict block missing")
    check(saved_verdict.get("verdict") == recomputed["verdict"], f"analysis verdict disagrees with independent rule: {saved_verdict.get('verdict')} vs {recomputed['verdict']}")
    criteria = saved_verdict.get("criteria")
    check(isinstance(criteria, dict), "analysis verdict criteria missing")
    expected_criteria = {
        "sensor_bias_5_improvement_all_training_seeds": all(
            recomputed["headline"]["sensor_bias_5"][str(seed)]["favorable"] for seed in TRAINING_SEEDS
        ),
        "sensor_dropout_improvement_all_training_seeds": all(
            recomputed["headline"]["sensor_dropout"][str(seed)]["favorable"] for seed in TRAINING_SEEDS
        ),
        "combined_fault_load_favorable_all_training_seeds": all(
            recomputed["headline"]["combined_fault_load"][str(seed)]["favorable"] for seed in TRAINING_SEEDS
        ),
        "no_model_pair_numerical_or_safety_failure": recomputed["no_model_pair_numerical_or_safety_failure"],
        "qualitative_paper_conclusions_do_not_reverse": recomputed["load_weakness_consistent"],
    }
    for key, expected in expected_criteria.items():
        check(bool(criteria.get(key)) == bool(expected), f"analysis verdict criterion mismatch: {key}")

    headline_rows = summary.get("headline_scenario_training_seed_summary")
    check(isinstance(headline_rows, list) and len(headline_rows) == 9, "analysis headline summary must contain 3 scenarios x 3 training seeds")
    for row in headline_rows:
        scenario = str(row["scenario"])
        seed = int(row["training_seed"])
        check(scenario in HEADLINE_SCENARIOS and seed in TRAINING_SEEDS, "analysis headline row has unexpected scenario/seed")
        expected = recomputed["headline"][scenario][str(seed)]
        assert_close(
            row["mean_c3_minus_b_fault_window_rmse"],
            expected["mean_difference"],
            f"analysis P{seed}/{scenario} headline mean",
        )
        check(bool(row["mean_effect_favorable"]) == expected["favorable"], f"analysis P{seed}/{scenario} favorable flag mismatch")

    headline_csv = RESULT_ROOT / "analysis" / "headline_paired_differences.csv"
    check(headline_csv.is_file(), "analysis headline paired-differences CSV is missing")
    headline_frame = pd.read_csv(headline_csv)
    required = {
        "training_seed",
        "scenario",
        "simulation_seed",
        "c3_minus_b_fault_window_rmse",
    }
    check(required.issubset(headline_frame.columns), "headline paired-differences CSV schema is incomplete")
    check(len(headline_frame) == 45, "headline paired-differences CSV must contain exactly 45 rows")
    for scenario in HEADLINE_SCENARIOS:
        for seed in TRAINING_SEEDS:
            group = headline_frame[
                (headline_frame["scenario"] == scenario)
                & (headline_frame["training_seed"].astype(int) == seed)
            ].sort_values("simulation_seed")
            check(tuple(group["simulation_seed"].astype(int)) == SIMULATION_SEEDS, f"analysis P{seed}/{scenario} paired-difference sim seeds changed")
            saved = group["c3_minus_b_fault_window_rmse"].to_numpy(dtype=float)
            expected = np.asarray(
                recomputed["headline"][scenario][str(seed)]["paired_differences"], dtype=float
            )
            check(
                np.allclose(saved, expected, rtol=MODEL_RTOL, atol=MODEL_ATOL),
                f"analysis P{seed}/{scenario} five paired differences disagree with raw matrix",
            )


def verify_extended_analysis_outputs(
    runs: pd.DataFrame,
    pairs: dict[int, PairArtifacts],
) -> None:
    """Independently reconstruct the remaining preregistered analysis outputs."""
    summary = load_json(ANALYSIS_SUMMARY)
    outputs = summary.get("outputs")
    check(isinstance(outputs, dict), "analysis output-hash bindings are missing")
    for name, path in ANALYSIS_TABLE_PATHS.items():
        entry = outputs.get(name)
        check(isinstance(entry, dict), f"analysis output binding missing for {name}")
        check(
            entry.get("path") == path.relative_to(PROJECT).as_posix(),
            f"analysis output path changed for {name}",
        )
        check(entry.get("sha256") == sha256(path), f"analysis output hash mismatch for {name}")

    load_table, load_summary = expected_load_tables(runs)
    check_frame_matches(
        read_analysis_csv("load_disturbance"),
        load_table,
        label="analysis load_disturbance.csv",
        sort_by=["training_seed", "simulation_seed"],
    )
    check_frame_matches(
        pd.DataFrame(summary.get("load_disturbance", [])),
        load_summary,
        label="analysis load-disturbance summary",
        sort_by=["training_seed"],
    )

    parameter_table, parameter_summary = expected_parameter_tables(runs)
    check_frame_matches(
        read_analysis_csv("parameter_variation"),
        parameter_table,
        label="analysis parameter_variation.csv",
        sort_by=["training_seed", "simulation_seed"],
    )
    check_frame_matches(
        pd.DataFrame(summary.get("parameter_variation", [])),
        parameter_summary,
        label="analysis parameter-variation summary",
        sort_by=["training_seed"],
    )

    main_by_seed, main_metric_summary = expected_main_metric_tables(pairs)
    check_frame_matches(
        read_analysis_csv("main_model_metrics_by_seed"),
        main_by_seed,
        label="analysis main_model_metrics_by_seed.csv",
        sort_by=["training_seed"],
    )
    check_frame_matches(
        read_analysis_csv("main_model_metric_summary"),
        main_metric_summary,
        label="analysis main_model_metric_summary.csv",
        sort_by=["metric"],
    )
    saved_main = summary.get("main_model_metrics")
    check(isinstance(saved_main, dict), "analysis main-model summary block missing")
    check_frame_matches(
        pd.DataFrame(saved_main.get("by_training_seed", [])),
        main_by_seed,
        label="analysis main-model by-training-seed summary",
        sort_by=["training_seed"],
    )
    check_frame_matches(
        pd.DataFrame(saved_main.get("across_training_seed_variability", [])),
        main_metric_summary,
        label="analysis main-model variability summary",
        sort_by=["metric"],
    )

    aux_by_seed, aux_metric_summary, aux_per_trajectory, aux_transfer = expected_aux_metric_tables(
        pairs, runs
    )
    check_frame_matches(
        read_analysis_csv("auxiliary_model_metrics_by_seed"),
        aux_by_seed,
        label="analysis auxiliary_model_metrics_by_seed.csv",
        sort_by=["training_seed"],
    )
    check_frame_matches(
        read_analysis_csv("auxiliary_model_metric_summary"),
        aux_metric_summary,
        label="analysis auxiliary_model_metric_summary.csv",
        sort_by=["metric"],
    )
    check_frame_matches(
        read_analysis_csv("auxiliary_per_trajectory"),
        aux_per_trajectory,
        label="analysis auxiliary_per_trajectory.csv",
        sort_by=["training_seed", "run_id"],
    )
    check_frame_matches(
        read_analysis_csv("auxiliary_closed_loop_transfer"),
        aux_transfer,
        label="analysis auxiliary_closed_loop_transfer.csv",
        sort_by=["training_seed", "scenario"],
    )
    saved_aux = summary.get("auxiliary_model_metrics")
    check(isinstance(saved_aux, dict), "analysis auxiliary-model summary block missing")
    check_frame_matches(
        pd.DataFrame(saved_aux.get("by_training_seed", [])),
        aux_by_seed,
        label="analysis auxiliary-model by-training-seed summary",
        sort_by=["training_seed"],
    )
    check_frame_matches(
        pd.DataFrame(saved_aux.get("across_training_seed_variability", [])),
        aux_metric_summary,
        label="analysis auxiliary-model variability summary",
        sort_by=["metric"],
    )
    check_frame_matches(
        pd.DataFrame(saved_aux.get("closed_loop_transfer", [])),
        aux_transfer,
        label="analysis auxiliary closed-loop-transfer summary",
        sort_by=["training_seed", "scenario"],
    )

    calibration_by_seed, calibration_variability = expected_calibration_tables(pairs)
    check_frame_matches(
        read_analysis_csv("calibration_by_seed"),
        calibration_by_seed,
        label="analysis calibration_by_seed.csv",
        sort_by=["training_seed"],
    )
    check_frame_matches(
        read_analysis_csv("calibration_variability"),
        calibration_variability,
        label="analysis calibration_variability.csv",
        sort_by=["calibration_statistic"],
    )
    saved_calibration = summary.get("calibration")
    check(isinstance(saved_calibration, dict), "analysis calibration summary block missing")
    check_frame_matches(
        pd.DataFrame(saved_calibration.get("by_training_seed", [])),
        calibration_by_seed,
        label="analysis calibration by-training-seed summary",
        sort_by=["training_seed"],
    )
    check_frame_matches(
        pd.DataFrame(saved_calibration.get("variability", [])),
        calibration_variability,
        label="analysis calibration variability summary",
        sort_by=["calibration_statistic"],
    )

    bootstrap_meta = summary.get("hierarchical_bootstrap")
    check(isinstance(bootstrap_meta, dict), "analysis hierarchical-bootstrap block missing")
    saved_analysis_seed = bootstrap_meta.get("analysis_seed")
    saved_replicates = bootstrap_meta.get("bootstrap_replicates")
    check(isinstance(saved_analysis_seed, int), "analysis bootstrap seed is not explicitly recorded")
    check(isinstance(saved_replicates, int), "analysis bootstrap replicate count is not explicitly recorded")
    analysis_seed = int(saved_analysis_seed)
    replicates = int(saved_replicates)
    check(replicates >= 1000, "analysis bootstrap replicate count violates the preregistered minimum")
    check(
        bootstrap_meta.get("interval") == "descriptive percentile 95% CI",
        "analysis bootstrap interval definition changed",
    )
    method = str(bootstrap_meta.get("method", "")).lower()
    check(
        "training-seed clusters" in method and "simulation-seed" in method and "with replacement" in method,
        "analysis bootstrap method description no longer records the two-level resampling design",
    )
    bootstrap = expected_hierarchical_bootstrap(
        runs,
        analysis_seed=analysis_seed,
        replicates=replicates,
    )
    check_frame_matches(
        read_analysis_csv("hierarchical_bootstrap_ci"),
        bootstrap,
        label="analysis hierarchical_bootstrap_ci.csv",
        sort_by=["scenario"],
    )
    check_frame_matches(
        pd.DataFrame(bootstrap_meta.get("results", [])),
        bootstrap,
        label="analysis hierarchical-bootstrap summary",
        sort_by=["scenario"],
    )


def main() -> None:
    # Deliberately no output generation: this verifier reads only completed
    # study artifacts and emits its verdict to stdout.
    verify_prework_frozen_hashes()
    verify_simulation_seed_provenance()
    pairs, pair_manifest = bind_pair_artifacts()
    verify_independent_calibrations(pairs)
    runs, recomputed = verify_run_matrix(pairs, pair_manifest)
    verify_analysis_summary(recomputed)
    verify_extended_analysis_outputs(runs, pairs)

    print("PASS: training-seed robustness study independently verified")
    print(f"VERDICT: {recomputed['verdict']}")
    print(f"TRAINING_SEEDS: {TRAINING_SEEDS}")
    print(f"SIMULATION_SEEDS: {SIMULATION_SEEDS}")
    print(f"RUNS: {EXPECTED_RUN_COUNT}")


if __name__ == "__main__":
    main()
