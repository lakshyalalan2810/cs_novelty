"""Calibrate C4 three-way consistency attribution from clean validation only.

This script is intentionally limited to the frozen clean validation split used
by the existing model artifacts.  It computes the three pairwise speed
disagreements between the physical sensor, main LSTM, and auxiliary LSTM,
then calibrates conservative agreement/disagreement bands from per-trajectory
EWMA traces.

The script does not read scenario labels, injected-fault flags, development
seeds, or holdout seeds, and it does not use ``y_true`` in any calibration
statistic or model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from auxiliary_sensor_model import load_auxiliary_model
from reliability import load_lstm_model


CALIBRATION_VERSION = "1.0.0"

DATASET = PROJECT / "data/processed/dc_motor_lstm_dataset.npz"
MAIN_WEIGHTS = PROJECT / "results/lstm_model_weights.pt"
MAIN_CONFIG = PROJECT / "results/configs/lstm_model_config.json"
AUX_WEIGHTS = PROJECT / "results/auxiliary_model_weights.pt"
AUX_CONFIG = PROJECT / "results/configs/v2_auxiliary_config.json"
V3_CONFIG = PROJECT / "results/configs/v3_arbitration_config.json"
V3_CALIBRATION = PROJECT / "results/configs/v3_arbitration_calibration.json"
V3_FINAL_RUNS = PROJECT / "results/metrics/v3_final_11scenario_runs.csv"
V3_FROZEN_HASHES = PROJECT / "results/configs/c4_v3_frozen_hashes.json"
OUTPUT = PROJECT / "results/configs/c4_attribution_calibration.json"

WINDOW_LENGTH = 20
DIAGNOSTIC_EWMA_ALPHA = 0.05
ATTRIBUTION_PERSISTENCE_COUNT = 3
ATTRIBUTION_EXIT_COUNT = 5
AGREEMENT_PERCENTILE = 75.0
DISAGREEMENT_PERCENTILE = 99.9
QUANTILES = (50.0, 75.0, 90.0, 95.0, 97.5, 99.0, 99.5, 99.9, 100.0)

CHECK_NUMERIC_RTOL = 1e-6
CHECK_NUMERIC_ATOL = 1e-5
CHECK_IGNORED_METADATA_PATHS = {
    "provenance.calibration_code.sha256",
    "provenance.runtime",
}
CHECK_TOLERANT_NUMERIC_PREFIXES = (
    "raw_pairwise_quantiles.",
    "ewma_pairwise_quantiles.",
    "trajectory_quantiles[",
    "clean_policy_diagnostics.raw_state_fractions[",
    "clean_policy_diagnostics.persisted_state_fractions[",
    "computed_thresholds.agreement.",
    "computed_thresholds.disagreement.",
    "computed_thresholds.agreement_thresholds_ordered[",
    "computed_thresholds.disagreement_thresholds_ordered[",
)

# These seed families are excluded by construction: the calibration source is
# the fixed clean validation dataset, not any closed-loop evaluation output.
DEVELOPMENT_SEEDS = list(range(19026, 19031))
V3_FINAL_HOLDOUT_SEEDS = list(range(29026, 29031))

PAIR_DEFINITIONS = {
    "d_sm": "absolute difference |ys - ym| between physical measured speed and main-LSTM speed estimate",
    "d_sa": "absolute difference |ys - ya| between physical measured speed and auxiliary speed estimate",
    "d_ma": "absolute difference |ym - ya| between main-LSTM and auxiliary speed estimates",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile_key(value: float) -> str:
    text = f"{value:g}".replace(".", "_")
    return f"p{text}"


def quantile_table(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("quantile input must be a non-empty finite 1-D array")
    return {
        percentile_key(percentile): float(np.percentile(values, percentile))
        for percentile in QUANTILES
    }


def build_windows(first: np.ndarray, second: np.ndarray, window_length: int) -> np.ndarray:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("window channels must be equal-length 1-D arrays")
    count = len(first) - window_length
    if count <= 0:
        raise ValueError("trajectory is too short for requested window length")
    features = np.column_stack((first, second)).astype(np.float32, copy=False)
    return np.asarray(
        [features[start : start + window_length] for start in range(count)],
        dtype=np.float32,
    )


@torch.no_grad()
def predict_physical(
    model: torch.nn.Module,
    inputs: np.ndarray,
    normalization: dict,
    batch_size: int = 1024,
) -> np.ndarray:
    input_mean = np.asarray(normalization["input_mean"], dtype=np.float32)
    input_std = np.asarray(normalization["input_std"], dtype=np.float32)
    target_mean = float(normalization["target_mean"][0])
    target_std = float(normalization["target_std"][0])
    normalized = ((inputs - input_mean) / input_std).astype(np.float32)

    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    for start in range(0, len(normalized), batch_size):
        batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
        chunks.append(model(batch)[:, 0].detach().cpu().numpy())
    predicted = np.concatenate(chunks).astype(np.float64)
    physical = predicted * target_std + target_mean
    if not np.isfinite(physical).all():
        raise FloatingPointError("model prediction contains NaN or Inf")
    return physical


def ewma_trace(values: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("EWMA input must be a non-empty finite 1-D array")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EWMA alpha must be in (0, 1]")

    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered


def classify_pattern(
    values: np.ndarray,
    agreement: np.ndarray,
    disagreement: np.ndarray,
) -> int:
    """Return the C4 state code for one raw disagreement triplet."""
    sm, sa, ma = values
    a_sm, a_sa, a_ma = agreement
    d_sm, d_sa, d_ma = disagreement
    if sm <= a_sm and sa <= a_sa and ma <= a_ma:
        return 0  # NORMAL
    if sm >= d_sm and sa >= d_sa and ma <= a_ma:
        return 1  # LIKELY_SENSOR_FAULT
    if sm >= d_sm and sa <= a_sa and ma >= d_ma:
        return 2  # LIKELY_PLANT_OR_MAIN_MISMATCH
    if sm <= a_sm and sa >= d_sa and ma >= d_ma:
        return 3  # LIKELY_AUX_MISMATCH
    return 4  # AMBIGUOUS


def apply_persistence(states: np.ndarray, enter_count: int, exit_count: int) -> np.ndarray:
    """Persist raw attribution states, resetting independently per trajectory."""
    states = np.asarray(states, dtype=np.int8)
    persisted = np.empty_like(states)
    current = 0
    candidate = 0
    candidate_run = 0
    for index, state in enumerate(states):
        state = int(state)
        if state == current:
            candidate = state
            candidate_run = 0
        else:
            if state == candidate:
                candidate_run += 1
            else:
                candidate = state
                candidate_run = 1
            required = exit_count if state == 0 else enter_count
            if candidate_run >= required:
                current = state
                candidate_run = 0
        persisted[index] = current
    return persisted


def validate_frozen_v3_hashes() -> dict[str, str]:
    """Confirm the protected V3 artifacts still match the pre-C4 hash record."""
    frozen = json.loads(V3_FROZEN_HASHES.read_text(encoding="utf-8"))
    expected = frozen["artifacts"]
    artifact_paths = {
        "main_model": MAIN_WEIGHTS,
        "auxiliary_model": AUX_WEIGHTS,
        "v3_calibration": V3_CALIBRATION,
        "v3_arbitration_config": V3_CONFIG,
        "v3_final_275_run_matrix": V3_FINAL_RUNS,
    }

    unvalidated = sorted(set(expected).difference(artifact_paths))
    if unvalidated:
        raise RuntimeError(
            "frozen V3 hash record contains artifacts with no calibration-side "
            f"validation: {unvalidated}"
        )
    missing = sorted(set(artifact_paths).difference(expected))
    if missing:
        raise RuntimeError(f"frozen V3 hash record is missing required artifacts: {missing}")

    current: dict[str, str] = {}
    for name, artifact_path in artifact_paths.items():
        record = expected[name]
        recorded_path = PROJECT / record["path"]
        if recorded_path.resolve() != artifact_path.resolve():
            raise RuntimeError(
                f"protected V3 artifact path changed for {name}: "
                f"expected {artifact_path}, got {recorded_path}"
            )
        actual_hash = sha256(artifact_path)
        expected_hash = record["sha256"]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"protected V3 artifact hash changed for {name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        current[name] = actual_hash
    return current


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the C4 attribution calibration in memory. By default the "
            "script runs in read/check mode and does not write the canonical artifact."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Write the recomputed calibration to this path. Relative paths are "
            "resolved from the project root."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --output to overwrite the canonical frozen calibration artifact.",
    )
    args = parser.parse_args(argv)
    if args.force and args.output is None:
        parser.error("--force requires --output")
    return args


def resolve_output_path(output: Path) -> Path:
    output = Path(output)
    if not output.is_absolute():
        output = PROJECT / output
    return output.resolve()


def display_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _child_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _is_ignored_check_path(path: str) -> bool:
    return path in CHECK_IGNORED_METADATA_PATHS or path.startswith("provenance.runtime.")


def _uses_tolerant_numeric_check(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in CHECK_TOLERANT_NUMERIC_PREFIXES)


def compare_calibrations(
    frozen: object,
    recomputed: object,
    *,
    path: str = "",
) -> list[str]:
    """Return scientific differences between frozen and recomputed calibrations."""
    if _is_ignored_check_path(path):
        return []

    differences: list[str] = []
    if isinstance(frozen, dict):
        if not isinstance(recomputed, dict):
            return [
                f"type mismatch at {path or '<root>'}: "
                f"frozen=dict recomputed={type(recomputed).__name__}"
            ]

        frozen_keys = {
            key for key in frozen if not _is_ignored_check_path(_child_path(path, key))
        }
        recomputed_keys = {
            key for key in recomputed if not _is_ignored_check_path(_child_path(path, key))
        }
        for key in sorted(frozen_keys - recomputed_keys):
            differences.append(f"missing key in recomputed calibration: {_child_path(path, key)}")
        for key in sorted(recomputed_keys - frozen_keys):
            differences.append(f"unexpected key in recomputed calibration: {_child_path(path, key)}")
        for key in sorted(frozen_keys & recomputed_keys):
            differences.extend(
                compare_calibrations(
                    frozen[key],
                    recomputed[key],
                    path=_child_path(path, key),
                )
            )
        return differences

    if isinstance(frozen, list):
        if not isinstance(recomputed, list):
            return [
                f"type mismatch at {path or '<root>'}: "
                f"frozen=list recomputed={type(recomputed).__name__}"
            ]
        if len(frozen) != len(recomputed):
            differences.append(
                f"length mismatch at {path}: frozen={len(frozen)} recomputed={len(recomputed)}"
            )
        for index, (frozen_item, recomputed_item) in enumerate(zip(frozen, recomputed)):
            differences.extend(
                compare_calibrations(
                    frozen_item,
                    recomputed_item,
                    path=f"{path}[{index}]",
                )
            )
        return differences

    if _uses_tolerant_numeric_check(path):
        if type(frozen) is float and type(recomputed) is float:
            if not np.isclose(
                float(frozen),
                float(recomputed),
                rtol=CHECK_NUMERIC_RTOL,
                atol=CHECK_NUMERIC_ATOL,
            ):
                differences.append(
                    f"numeric mismatch at {path}: frozen={frozen!r} recomputed={recomputed!r} "
                    f"(rtol={CHECK_NUMERIC_RTOL:g}, atol={CHECK_NUMERIC_ATOL:g})"
                )
            return differences

    if type(frozen) is not type(recomputed):
        differences.append(
            f"type mismatch at {path or '<root>'}: "
            f"frozen={type(frozen).__name__} recomputed={type(recomputed).__name__}"
        )
    elif frozen != recomputed:
        differences.append(
            f"value mismatch at {path or '<root>'}: frozen={frozen!r} recomputed={recomputed!r}"
        )
    return differences


def check_canonical_equivalence(
    calibration: dict,
    canonical_path: Path | None = None,
) -> dict[str, object]:
    canonical = Path(canonical_path) if canonical_path is not None else OUTPUT
    if not canonical.exists():
        raise RuntimeError(f"canonical C4 attribution calibration does not exist: {canonical}")
    try:
        frozen = json.loads(canonical.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"canonical C4 attribution calibration is invalid JSON: {canonical}") from exc
    if not isinstance(frozen, dict):
        raise RuntimeError("canonical C4 attribution calibration must contain a JSON object")

    differences = compare_calibrations(frozen, calibration)
    return {
        "status": "PASS" if not differences else "FAIL",
        "differences": differences,
        "numeric_rtol": CHECK_NUMERIC_RTOL,
        "numeric_atol": CHECK_NUMERIC_ATOL,
        "ignored_generation_metadata": sorted(CHECK_IGNORED_METADATA_PATHS),
    }


def require_check_passed(check_result: dict[str, object] | None) -> None:
    if check_result is not None and check_result["status"] != "PASS":
        raise SystemExit(1)


def write_requested_output(
    calibration: dict,
    output: Path | None,
    *,
    force: bool = False,
) -> Path | None:
    """Write only when an explicit output path is supplied.

    The canonical calibration is frozen scientific evidence. Replacing it
    requires both an explicit canonical ``--output`` path and ``--force``.
    """
    if output is None:
        return None

    target = resolve_output_path(output)
    canonical = OUTPUT.resolve()
    if target == canonical and not force:
        raise RuntimeError(
            "refusing to overwrite canonical C4 attribution calibration without --force"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(1)
    protected_hashes = validate_frozen_v3_hashes()

    data = np.load(DATASET)
    required = {
        "voltage",
        "current",
        "y_measured",
        "split_run_ids_validation",
        "seed",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"dataset missing required arrays: {missing}")

    validation_ids = [int(value) for value in data["split_run_ids_validation"]]
    if not validation_ids:
        raise RuntimeError("clean validation split is empty")

    main_model, main_config = load_lstm_model(MAIN_WEIGHTS, MAIN_CONFIG)
    aux_model, aux_config = load_auxiliary_model(AUX_WEIGHTS, AUX_CONFIG)
    if int(main_config.get("window_length", WINDOW_LENGTH)) != WINDOW_LENGTH:
        raise RuntimeError("main model window length does not match C4 calibration")
    if int(aux_config.get("window_length", WINDOW_LENGTH)) != WINDOW_LENGTH:
        raise RuntimeError("auxiliary model window length does not match C4 calibration")

    raw_by_pair: dict[str, list[np.ndarray]] = {name: [] for name in PAIR_DEFINITIONS}
    filtered_by_pair: dict[str, list[np.ndarray]] = {name: [] for name in PAIR_DEFINITIONS}
    trajectory_summaries: list[dict] = []

    for run_id in validation_ids:
        voltage = np.asarray(data["voltage"][run_id], dtype=np.float32)
        current = np.asarray(data["current"][run_id], dtype=np.float32)
        y_sensor_full = np.asarray(data["y_measured"][run_id], dtype=np.float64)
        if not (
            voltage.shape == current.shape == y_sensor_full.shape
            and voltage.ndim == 1
            and np.isfinite(voltage).all()
            and np.isfinite(current).all()
            and np.isfinite(y_sensor_full).all()
        ):
            raise ValueError(f"invalid clean validation trajectory {run_id}")

        main_inputs = build_windows(voltage, y_sensor_full, WINDOW_LENGTH)
        aux_inputs = build_windows(voltage, current, WINDOW_LENGTH)
        y_main = predict_physical(main_model, main_inputs, main_config["normalization"])
        y_aux = predict_physical(aux_model, aux_inputs, aux_config["normalization"])
        y_sensor = y_sensor_full[WINDOW_LENGTH:].astype(np.float64, copy=False)

        if not (len(y_sensor) == len(y_main) == len(y_aux)):
            raise RuntimeError(f"prediction alignment failed for validation trajectory {run_id}")

        raw = {
            "d_sm": np.abs(y_sensor - y_main),
            "d_sa": np.abs(y_sensor - y_aux),
            "d_ma": np.abs(y_main - y_aux),
        }
        filtered = {
            name: ewma_trace(values, DIAGNOSTIC_EWMA_ALPHA)
            for name, values in raw.items()
        }

        for name in PAIR_DEFINITIONS:
            raw_by_pair[name].append(raw[name])
            filtered_by_pair[name].append(filtered[name])

        trajectory_summaries.append(
            {
                "run_id": run_id,
                "sample_count": int(len(y_sensor)),
                "raw": {name: quantile_table(raw[name]) for name in PAIR_DEFINITIONS},
                "ewma": {name: quantile_table(filtered[name]) for name in PAIR_DEFINITIONS},
            }
        )

    raw_flat = {name: np.concatenate(parts) for name, parts in raw_by_pair.items()}
    filtered_flat = {name: np.concatenate(parts) for name, parts in filtered_by_pair.items()}

    raw_quantiles = {name: quantile_table(values) for name, values in raw_flat.items()}
    filtered_quantiles = {
        name: quantile_table(values) for name, values in filtered_flat.items()
    }
    agreement_thresholds = {
        name: float(np.percentile(raw_flat[name], AGREEMENT_PERCENTILE))
        for name in PAIR_DEFINITIONS
    }
    disagreement_thresholds = {
        name: float(np.percentile(raw_flat[name], DISAGREEMENT_PERCENTILE))
        for name in PAIR_DEFINITIONS
    }
    for name in PAIR_DEFINITIONS:
        if agreement_thresholds[name] >= disagreement_thresholds[name]:
            raise RuntimeError(f"invalid clean calibration band for {name}")

    sample_counts = {name: int(len(values)) for name, values in raw_flat.items()}
    if len(set(sample_counts.values())) != 1:
        raise RuntimeError("pairwise calibration sample counts do not align")

    agreement_array = np.asarray(
        [agreement_thresholds["d_sm"], agreement_thresholds["d_sa"], agreement_thresholds["d_ma"]],
        dtype=np.float64,
    )
    disagreement_array = np.asarray(
        [
            disagreement_thresholds["d_sm"],
            disagreement_thresholds["d_sa"],
            disagreement_thresholds["d_ma"],
        ],
        dtype=np.float64,
    )
    raw_state_counts = np.zeros(5, dtype=np.int64)
    persisted_state_counts = np.zeros(5, dtype=np.int64)
    for run_index in range(len(validation_ids)):
        triplets = np.column_stack(
            [raw_by_pair[name][run_index] for name in ("d_sm", "d_sa", "d_ma")]
        )
        raw_states = np.asarray(
            [classify_pattern(row, agreement_array, disagreement_array) for row in triplets],
            dtype=np.int8,
        )
        persisted_states = apply_persistence(
            raw_states,
            ATTRIBUTION_PERSISTENCE_COUNT,
            ATTRIBUTION_EXIT_COUNT,
        )
        raw_state_counts += np.bincount(raw_states, minlength=5)
        persisted_state_counts += np.bincount(persisted_states, minlength=5)

    calibration = {
        "calibration_version": CALIBRATION_VERSION,
        "calibration_population": (
            "clean validation trajectories from data/processed/dc_motor_lstm_dataset.npz only"
        ),
        "online_statistic_definitions": PAIR_DEFINITIONS,
        "units": "rad/s",
        "quantile_method": "numpy.percentile with method='linear' (NumPy default)",
        "classification_statistic": {
            "type": "raw pairwise absolute disagreement",
            "temporal_filter": "none",
            "temporal_stabilization": "state persistence only",
        },
        "diagnostic_filter_definition": {
            "type": "per-trajectory exponentially weighted moving average",
            "formula": "z_t = alpha*d_t + (1-alpha)*z_(t-1)",
            "initialization": "z_0 = d_0 independently for every validation trajectory and pair",
            "alpha": DIAGNOSTIC_EWMA_ALPHA,
            "trajectory_reset": True,
            "used_for_classification_or_veto": False,
        },
        "classification_threshold_policy": {
            "agreement": (
                f"raw pairwise disagreement <= clean-validation p{AGREEMENT_PERCENTILE:g}"
            ),
            "disagreement": (
                f"raw pairwise disagreement >= clean-validation p{DISAGREEMENT_PERCENTILE:g}"
            ),
            "neutral_band": (
                f"values between p{AGREEMENT_PERCENTILE:g} and p{DISAGREEMENT_PERCENTILE:g} "
                "are not forced into agreement or disagreement"
            ),
            "rationale": (
                "p75 defines strong witness concordance for a safety inhibit, while p99.9 "
                "requires rare clean-data disagreement before declaring an outlier; pair-specific "
                "thresholds are frozen before C4 development evaluation"
            ),
        },
        "state_pattern_definition": {
            "NORMAL": "all three raw pairwise disagreements are in their agreement bands",
            "LIKELY_SENSOR_FAULT": "d_sm and d_sa are in disagreement bands while d_ma is in its agreement band",
            "LIKELY_PLANT_OR_MAIN_MISMATCH": "d_sm and d_ma are in disagreement bands while d_sa is in its agreement band",
            "LIKELY_AUX_MISMATCH": "d_sa and d_ma are in disagreement bands while d_sm is in its agreement band",
            "AMBIGUOUS": "all remaining patterns, including neutral-band and all-large cases",
        },
        "persistence": {
            "enter_count": ATTRIBUTION_PERSISTENCE_COUNT,
            "exit_count_to_normal": ATTRIBUTION_EXIT_COUNT,
            "source": (
                "reuses the existing V3/C1 SensorReliabilityMonitor defaults: "
                "3 samples to enter a non-normal attribution state and 5 samples to return NORMAL"
            ),
            "fitted_from_data": False,
        },
        "validation_run_ids": validation_ids,
        "validation_trajectory_count": len(validation_ids),
        "dataset_seed": int(data["seed"]),
        "sample_counts": {
            "per_pair_raw": sample_counts,
            "per_pair_ewma": {name: int(len(values)) for name, values in filtered_flat.items()},
            "total_aligned_samples": next(iter(sample_counts.values())),
            "samples_per_trajectory": {
                str(item["run_id"]): item["sample_count"] for item in trajectory_summaries
            },
        },
        "raw_pairwise_quantiles": raw_quantiles,
        "ewma_pairwise_quantiles": filtered_quantiles,
        "ewma_pairwise_quantiles_role": "diagnostic only; not used for C4 classification or entry veto",
        "trajectory_quantiles": trajectory_summaries,
        "clean_policy_diagnostics": {
            "state_order": [
                "NORMAL",
                "LIKELY_SENSOR_FAULT",
                "LIKELY_PLANT_OR_MAIN_MISMATCH",
                "LIKELY_AUX_MISMATCH",
                "AMBIGUOUS",
            ],
            "raw_state_counts": raw_state_counts.tolist(),
            "raw_state_fractions": (raw_state_counts / raw_state_counts.sum()).tolist(),
            "persisted_state_counts": persisted_state_counts.tolist(),
            "persisted_state_fractions": (
                persisted_state_counts / persisted_state_counts.sum()
            ).tolist(),
        },
        "computed_thresholds": {
            "pair_order": ["d_sm", "d_sa", "d_ma"],
            "classification_statistic": "raw_pairwise_absolute_disagreement",
            "agreement_percentile": AGREEMENT_PERCENTILE,
            "disagreement_percentile": DISAGREEMENT_PERCENTILE,
            "agreement": agreement_thresholds,
            "disagreement": disagreement_thresholds,
            "agreement_thresholds_ordered": [
                agreement_thresholds["d_sm"],
                agreement_thresholds["d_sa"],
                agreement_thresholds["d_ma"],
            ],
            "disagreement_thresholds_ordered": [
                disagreement_thresholds["d_sm"],
                disagreement_thresholds["d_sa"],
                disagreement_thresholds["d_ma"],
            ],
            "enter_count": ATTRIBUTION_PERSISTENCE_COUNT,
            "exit_count": ATTRIBUTION_EXIT_COUNT,
            "diagnostic_ewma_alpha": DIAGNOSTIC_EWMA_ALPHA,
        },
        "provenance": {
            "dataset": {
                "path": str(DATASET.relative_to(PROJECT)).replace("\\", "/"),
                "sha256": sha256(DATASET),
            },
            "models": {
                "main_weights_sha256": protected_hashes["main_model"],
                "main_config_sha256": sha256(MAIN_CONFIG),
                "auxiliary_weights_sha256": protected_hashes["auxiliary_model"],
                "auxiliary_config_sha256": sha256(AUX_CONFIG),
            },
            "v3": {
                "arbitration_config_sha256": protected_hashes["v3_arbitration_config"],
                "arbitration_calibration_sha256": protected_hashes["v3_calibration"],
                "final_275_run_matrix_sha256": protected_hashes["v3_final_275_run_matrix"],
                "pre_c4_hash_record_sha256": sha256(V3_FROZEN_HASHES),
            },
            "calibration_code": {
                "path": str(Path(__file__).resolve().relative_to(PROJECT)).replace("\\", "/"),
                "version": CALIBRATION_VERSION,
                "sha256": sha256(Path(__file__).resolve()),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
        },
        "exclusions": {
            "development_seeds": DEVELOPMENT_SEEDS,
            "v3_final_holdout_seeds": V3_FINAL_HOLDOUT_SEEDS,
            "closed_loop_scenarios": "not read or used",
            "scenario_labels": "not read or used",
            "fault_labels_or_flags": "not read or used",
            "y_true": "not read or used by this calibration script",
            "future_c4_holdout": "not selected or used during calibration",
        },
    }

    written_output = write_requested_output(calibration, args.output, force=args.force)
    check_result = None
    if written_output is None:
        check_result = check_canonical_equivalence(calibration)

    print(
        json.dumps(
            {
                "mode": "write" if written_output is not None else "check",
                "output": display_path(written_output) if written_output is not None else None,
                "canonical_output": display_path(OUTPUT),
                "canonical_sha256": sha256(OUTPUT) if OUTPUT.exists() else None,
                "canonical_equivalence": check_result,
                "validation_run_ids": validation_ids,
                "total_aligned_samples": next(iter(sample_counts.values())),
                "agreement_thresholds": agreement_thresholds,
                "disagreement_thresholds": disagreement_thresholds,
                "enter_count": ATTRIBUTION_PERSISTENCE_COUNT,
                "exit_count": ATTRIBUTION_EXIT_COUNT,
                "diagnostic_ewma_alpha": DIAGNOSTIC_EWMA_ALPHA,
            },
            indent=2,
        )
    )
    require_check_passed(check_result)


if __name__ == "__main__":
    main()
