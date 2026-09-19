"""Reproduce and freeze V3 arbitration thresholds from clean validation runs."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from auxiliary_sensor_model import build_auxiliary_sequences, load_auxiliary_model, normalize_auxiliary
from data_utils import build_training_sequences
from reliability import load_lstm_model

DATASET = PROJECT / "data/processed/dc_motor_lstm_dataset.npz"
CALIBRATION = PROJECT / "results/configs/v3_arbitration_calibration.json"
CONFIG = PROJECT / "results/configs/v3_arbitration_config.json"
MAIN_MODEL_CONFIG = PROJECT / "results/configs/lstm_model_config.json"
AUXILIARY_MODEL_CONFIG = PROJECT / "results/configs/v2_auxiliary_config.json"
FINAL_HOLDOUT_SEEDS = list(range(29026, 29031))
DEVELOPMENT_SEEDS = list(range(19026, 19031))
WINDOW_LENGTH = 20
EWMA_ALPHA = 0.05
STARTUP_BLANKING_TIME = 1.0
AGREEMENT_PERCENTILE = 90.0
MISMATCH_PERCENTILE = 99.9
RECOVERY_PERCENTILE = 95.0
RECOVERY_COUNT = 50
AUX_SPEED_BOUNDS = [0.0, 100.0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def post_blanking_ewma(
    times: np.ndarray,
    residual: np.ndarray,
    alpha: float,
    startup_blanking_time: float,
) -> np.ndarray:
    """Match the deployed arbitrator: do not accumulate EWMA during startup blanking."""
    values = []
    ewma = 0.0
    for time_value, residual_value in zip(times, residual):
        if time_value >= startup_blanking_time:
            ewma = (1 - alpha) * ewma + alpha * float(residual_value)
            values.append(ewma)
    return np.asarray(values, dtype=float)


@torch.no_grad()
def predict(model, inputs: np.ndarray, mean: float, std: float) -> np.ndarray:
    values = []
    for start in range(0, len(inputs), 1024):
        values.append(model(torch.from_numpy(inputs[start : start + 1024]))[:, 0].numpy())
    result = np.concatenate(values) * std + mean
    if not np.isfinite(result).all():
        raise FloatingPointError("calibration prediction contains NaN or Inf")
    return result


def main() -> None:
    torch.set_num_threads(1)
    data = np.load(DATASET)
    validation_ids = [int(value) for value in data["split_run_ids_validation"]]
    main_model, main_config = load_lstm_model(
        PROJECT / "results/lstm_model_weights.pt",
        MAIN_MODEL_CONFIG,
    )
    aux_model, aux_config = load_auxiliary_model(
        PROJECT / "results/auxiliary_model_weights.pt",
        AUXILIARY_MODEL_CONFIG,
    )
    main_norm = main_config["normalization"]
    aux_norm = aux_config["normalization"]

    agreement, aux_residual, ewma_values, aux_predictions = [], [], [], []
    for run_id in validation_ids:
        trajectory = {
            "voltage": data["voltage"][run_id],
            "y_measured": data["y_measured"][run_id],
            "y_true": data["y_true"][run_id],
            "run_id": np.full(len(data["time"]), run_id),
        }
        main_inputs, _, _ = build_training_sequences(trajectory, WINDOW_LENGTH, 1)
        main_inputs = (
            (main_inputs - np.asarray(main_norm["input_mean"]))
            / np.asarray(main_norm["input_std"])
        ).astype(np.float32)
        aux_inputs, aux_target = build_auxiliary_sequences(
            data["voltage"][run_id],
            data["current"][run_id],
            data["y_true"][run_id],
            WINDOW_LENGTH,
        )
        aux_inputs, _ = normalize_auxiliary(aux_inputs, aux_target, aux_norm)
        y_main = predict(
            main_model,
            main_inputs,
            float(main_norm["target_mean"][0]),
            float(main_norm["target_std"][0]),
        )
        y_aux = predict(
            aux_model,
            aux_inputs,
            float(aux_norm["target_mean"][0]),
            float(aux_norm["target_std"][0]),
        )
        y_measured = data["y_measured"][run_id, WINDOW_LENGTH:]
        times = data["time"][WINDOW_LENGTH:]
        agreement.extend(np.abs(y_main - y_aux))
        residual = np.abs(y_measured - y_aux)
        aux_residual.extend(residual)
        aux_predictions.extend(y_aux)
        ewma_values.extend(
            post_blanking_ewma(times, residual, EWMA_ALPHA, STARTUP_BLANKING_TIME)
        )

    agreement = np.asarray(agreement)
    aux_residual = np.asarray(aux_residual)
    ewma_values = np.asarray(ewma_values)
    aux_predictions = np.asarray(aux_predictions)
    thresholds = {
        "agreement_threshold": float(np.percentile(agreement, AGREEMENT_PERCENTILE)),
        "param_mismatch_threshold": float(np.percentile(ewma_values, MISMATCH_PERCENTILE)),
        "param_mismatch_recovery_threshold": float(np.percentile(ewma_values, RECOVERY_PERCENTILE)),
        "aux_recovery_gate": float(np.percentile(aux_residual, 99.9)),
    }
    if thresholds["param_mismatch_recovery_threshold"] >= thresholds["param_mismatch_threshold"]:
        raise AssertionError("calibrated recovery threshold must be below the latch threshold")

    calibration = {
        "metric_definition": {
            "agreement": "absolute difference between main and auxiliary one-step speed predictions",
            "parameter_mismatch": f"EWMA(alpha={EWMA_ALPHA}) of absolute measured-minus-auxiliary residual while the physical sensor is trusted",
            "aux_recovery": "absolute measured-minus-auxiliary residual",
        },
        "calibration_population": "clean validation trajectories only; no closed-loop holdout scenarios",
        "dataset_sha256": sha256(DATASET),
        "model_sha256": {
            "main": sha256(PROJECT / "results/lstm_model_weights.pt"),
            "main_config": sha256(MAIN_MODEL_CONFIG),
            "auxiliary": sha256(PROJECT / "results/auxiliary_model_weights.pt"),
            "auxiliary_config": sha256(AUXILIARY_MODEL_CONFIG),
        },
        "dataset_seed": int(data["seed"]),
        "validation_run_ids": validation_ids,
        "sample_counts": {
            "agreement": int(len(agreement)),
            "aux_recovery": int(len(aux_residual)),
            "ewma_after_startup_blanking": int(len(ewma_values)),
        },
        "statistics": {
            "agreement_p90": float(np.percentile(agreement, 90.0)),
            "agreement_p95": float(np.percentile(agreement, 95.0)),
            "agreement_p99": float(np.percentile(agreement, 99.0)),
            "ewma_p95": float(np.percentile(ewma_values, 95.0)),
            "ewma_p99_9": float(np.percentile(ewma_values, 99.9)),
            "aux_residual_p99_9": float(np.percentile(aux_residual, 99.9)),
            "aux_prediction_min": float(aux_predictions.min()),
            "aux_prediction_max": float(aux_predictions.max()),
        },
        "threshold_derivation": {
            "agreement_threshold": f"p{AGREEMENT_PERCENTILE:g} of clean-validation agreement",
            "param_mismatch_threshold": f"p{MISMATCH_PERCENTILE:g} of clean-validation EWMA",
            "param_mismatch_recovery_threshold": f"p{RECOVERY_PERCENTILE:g} of clean-validation EWMA (lower hysteresis threshold)",
            "aux_recovery_gate": "p99.9 of clean-validation auxiliary residual; independently reproduces V2 gate",
            "aux_speed_bounds": "fixed application/simulation operating envelope in rad/s, not a fitted percentile",
            "param_mismatch_recovery_count": "50 samples = 0.5 s at the fixed 0.01 s simulation step",
        },
        "computed_values": {
            **thresholds,
            "ewma_alpha": EWMA_ALPHA,
            "startup_blanking_time": STARTUP_BLANKING_TIME,
            "param_mismatch_recovery_count": RECOVERY_COUNT,
            "aux_speed_bounds": AUX_SPEED_BOUNDS,
        },
        "excluded_final_holdout_seeds": FINAL_HOLDOUT_SEEDS,
    }
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    config = {
        "status": "frozen_before_final_holdout",
        "arbitrator": calibration["computed_values"],
        "feedback_fallback": "bounded hold of last finite trusted physical speed",
        "control_fallback": "rate-limited PI voltage only when MPC optimization fails",
        "aux_speed_bounds_meaning": calibration["threshold_derivation"]["aux_speed_bounds"],
        "calibration_file": str(CALIBRATION.relative_to(PROJECT)).replace("\\", "/"),
        "calibration_sha256": sha256(CALIBRATION),
        "development_extension_seeds": DEVELOPMENT_SEEDS,
        "final_holdout_seeds": FINAL_HOLDOUT_SEEDS,
        "scenarios": [
            "sensor_noise", "sensor_bias_5", "sensor_bias_15", "sensor_dropout",
            "sensor_drift", "load_disturbance", "parameter_variation", "combined_fault_load",
        ],
        "controllers": ["A_PI", "B_plain_MPC", "C1_sensor_MPC", "C2_aux_recovery_MPC", "C3_arbitration_MPC"],
    }
    CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"calibration": str(CALIBRATION), "config": str(CONFIG), **thresholds}, indent=2))


if __name__ == "__main__":
    main()
