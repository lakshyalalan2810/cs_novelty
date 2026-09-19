"""Train and evaluate the auxiliary speed estimator.

This script:
1. Loads the existing dataset (voltage, current, y_true)
2. Builds [voltage, current] -> speed sequences
3. Trains a compact auxiliary LSTM
4. Evaluates on clean test trajectories
5. Calibrates auxiliary thresholds from clean validation data
6. Saves model weights, config, and calibration results

Usage:
    python scripts/train_auxiliary_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auxiliary_sensor_model import (
    AuxiliarySpeedEstimator,
    build_auxiliary_sequences,
    fit_auxiliary_normalization,
    normalize_auxiliary,
    save_auxiliary_model,
    auxiliary_predict_online,
)

PROJECT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT / "data" / "processed" / "dc_motor_lstm_dataset.npz"
RESULTS = PROJECT / "results"
WEIGHTS_PATH = RESULTS / "auxiliary_model_weights.pt"
CONFIG_PATH = RESULTS / "configs" / "v2_auxiliary_config.json"
METRICS_PATH = RESULTS / "metrics" / "v2_auxiliary_model_metrics.json"
CALIBRATION_PATH = RESULTS / "metrics" / "v2_auxiliary_calibration.json"

# Hyperparameters
WINDOW_LENGTH = 20
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.0
BATCH_SIZE = 512
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-6
SEED = 2026


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset() -> dict:
    """Load the existing dataset and return raw trajectory data."""
    data = np.load(DATASET_PATH, allow_pickle=True)
    return {
        "voltage": data["voltage"],           # (30, 1201)
        "current": data["current"],           # (30, 1201)
        "y_true": data["y_true"],             # (30, 1201)
        "y_measured": data["y_measured"],       # (30, 1201)
        "load_torque": data["load_torque"],     # (30, 1201)
        "split_run_ids_train": data["split_run_ids_train"],
        "split_run_ids_validation": data["split_run_ids_validation"],
        "split_run_ids_test": data["split_run_ids_test"],
        "run_ids": data["run_ids"],
    }


def build_split_sequences(
    data: dict, run_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build auxiliary sequences for a split."""
    all_X, all_y = [], []
    for rid in run_ids:
        X, y = build_auxiliary_sequences(
            data["voltage"][rid],
            data["current"][rid],
            data["y_true"][rid],
            window_length=WINDOW_LENGTH,
        )
        all_X.append(X)
        all_y.append(y)
    return np.concatenate(all_X), np.concatenate(all_y)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    normalization: dict,
) -> tuple[AuxiliarySpeedEstimator, dict]:
    """Train the auxiliary LSTM with early stopping."""
    # Normalize
    X_train_n, y_train_n = normalize_auxiliary(X_train, y_train, normalization)
    X_val_n, y_val_n = normalize_auxiliary(X_val, y_val, normalization)

    # Dataloaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train_n), torch.from_numpy(y_train_n),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_n), torch.from_numpy(y_val_n),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    model = AuxiliarySpeedEstimator(
        input_size=2,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    print(f"Training auxiliary model: {sum(p.numel() for p in model.parameters())} params")
    print(f"Train: {len(X_train_n)} samples, Val: {len(X_val_n)} samples")

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}, best epoch {best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    print(f"  Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")
    return model, {"best_epoch": best_epoch, "history": history}


def evaluate_clean_test(
    model: AuxiliarySpeedEstimator,
    data: dict,
    test_ids: np.ndarray,
    normalization: dict,
) -> dict:
    """Evaluate on clean test trajectories."""
    X_test, y_test = build_split_sequences(data, test_ids)
    X_test_n, y_test_n = normalize_auxiliary(X_test, y_test, normalization)

    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(X_test_n)).numpy()

    # Denormalize
    target_mean = normalization["target_mean"][0]
    target_std = normalization["target_std"][0]
    pred_phys = pred_n * target_std + target_mean
    true_phys = y_test  # already in physical units

    errors = pred_phys - true_phys
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((true_phys - true_phys.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    return {
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "test_samples": len(y_test),
    }


def evaluate_per_trajectory(
    model: AuxiliarySpeedEstimator,
    data: dict,
    run_ids: np.ndarray,
    normalization: dict,
) -> list[dict]:
    """Per-trajectory evaluation for load/parameter analysis."""
    results = []
    target_mean = normalization["target_mean"][0]
    target_std = normalization["target_std"][0]

    for rid in run_ids:
        X, y = build_auxiliary_sequences(
            data["voltage"][rid],
            data["current"][rid],
            data["y_true"][rid],
            window_length=WINDOW_LENGTH,
        )
        X_n, y_n = normalize_auxiliary(X, y, normalization)
        model.eval()
        with torch.no_grad():
            pred_n = model(torch.from_numpy(X_n)).numpy()
        pred_phys = pred_n * target_std + target_mean
        errors = pred_phys - y

        # Check if this trajectory has load changes
        lt = data["load_torque"][rid]
        has_load_change = bool(lt.max() - lt.min() > 0.01)

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        results.append({
            "run_id": int(rid),
            "rmse": rmse,
            "mae": mae,
            "has_load_change": has_load_change,
            "load_torque_range": float(lt.max() - lt.min()),
        })

    return results


def calibrate_auxiliary_thresholds(
    model: AuxiliarySpeedEstimator,
    data: dict,
    val_ids: np.ndarray,
    normalization: dict,
) -> dict:
    """Calibrate auxiliary consistency thresholds from clean validation data.

    Computes |y_measured - y_aux| on clean validation trajectories and
    extracts percentile-based thresholds.
    """
    target_mean = normalization["target_mean"][0]
    target_std = normalization["target_std"][0]

    all_residuals = []
    for rid in val_ids:
        voltage = data["voltage"][rid]
        current = data["current"][rid]
        y_meas = data["y_measured"][rid]
        y_true = data["y_true"][rid]

        X, y = build_auxiliary_sequences(voltage, current, y_true, WINDOW_LENGTH)
        X_n, _ = normalize_auxiliary(X, y, normalization)
        model.eval()
        with torch.no_grad():
            pred_n = model(torch.from_numpy(X_n)).numpy()[:, 0]
        pred_phys = pred_n * target_std + target_mean

        # Residual: measured - auxiliary prediction
        y_meas_aligned = y_meas[WINDOW_LENGTH : WINDOW_LENGTH + len(pred_phys)]
        residual = y_meas_aligned - pred_phys
        all_residuals.append(residual)

    all_residuals = np.concatenate(all_residuals)
    abs_residuals = np.abs(all_residuals)

    thresholds = {}
    for pct in [90.0, 95.0, 97.5, 99.0, 99.5, 99.9]:
        thresholds[f"p{pct}"] = float(np.percentile(abs_residuals, pct))

    calibration = {
        "residual_mean": float(np.mean(all_residuals)),
        "residual_std": float(np.std(all_residuals)),
        "abs_residual_mean": float(np.mean(abs_residuals)),
        "abs_residual_median": float(np.median(abs_residuals)),
        "thresholds": thresholds,
        "recommended_recovery_gate": thresholds["p99.9"],
        "calibration_source": "clean validation data only",
        "num_samples": len(all_residuals),
    }
    return calibration


def main() -> None:
    set_seed(SEED)
    print("=" * 60)
    print("AUXILIARY VIRTUAL SPEED ESTIMATOR — Training & Calibration")
    print("=" * 60)

    # Load data
    data = load_dataset()
    train_ids = data["split_run_ids_train"]
    val_ids = data["split_run_ids_validation"]
    test_ids = data["split_run_ids_test"]

    print(f"\nTrajectories: train={list(train_ids)}, val={list(val_ids)}, test={list(test_ids)}")

    # Build sequences
    print("\nBuilding auxiliary sequences [voltage, current] -> speed...")
    X_train, y_train = build_split_sequences(data, train_ids)
    X_val, y_val = build_split_sequences(data, val_ids)
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")

    # Fit normalization (train only)
    normalization = fit_auxiliary_normalization(X_train, y_train)
    print(f"  Input mean:  {normalization['input_mean']}")
    print(f"  Input std:   {normalization['input_std']}")
    print(f"  Target mean: {normalization['target_mean']}")
    print(f"  Target std:  {normalization['target_std']}")

    # Train
    print("\n--- Training ---")
    model, train_info = train_model(X_train, y_train, X_val, y_val, normalization)

    # Evaluate on clean test
    print("\n--- Clean Test Evaluation ---")
    test_metrics = evaluate_clean_test(model, data, test_ids, normalization)
    print(f"  RMSE: {test_metrics['test_rmse']:.4f} rad/s")
    print(f"  MAE:  {test_metrics['test_mae']:.4f} rad/s")
    print(f"  R²:   {test_metrics['test_r2']:.6f}")

    # Per-trajectory analysis
    print("\n--- Per-Trajectory Analysis (Test Set) ---")
    per_traj = evaluate_per_trajectory(model, data, test_ids, normalization)
    for t in per_traj:
        load_marker = " [LOAD_CHANGE]" if t["has_load_change"] else ""
        print(f"  Run {t['run_id']:2d}: RMSE={t['rmse']:.4f}, MAE={t['mae']:.4f}, "
              f"load_range={t['load_torque_range']:.4f}{load_marker}")

    # Also evaluate on validation for completeness
    print("\n--- Per-Trajectory Analysis (Validation Set) ---")
    per_traj_val = evaluate_per_trajectory(model, data, val_ids, normalization)
    for t in per_traj_val:
        load_marker = " [LOAD_CHANGE]" if t["has_load_change"] else ""
        print(f"  Run {t['run_id']:2d}: RMSE={t['rmse']:.4f}, MAE={t['mae']:.4f}, "
              f"load_range={t['load_torque_range']:.4f}{load_marker}")

    # Calibrate thresholds
    print("\n--- Threshold Calibration (Clean Validation Data) ---")
    calibration = calibrate_auxiliary_thresholds(model, data, val_ids, normalization)
    print(f"  Residual mean: {calibration['residual_mean']:.4f}")
    print(f"  Residual std:  {calibration['residual_std']:.4f}")
    print(f"  |Residual| median: {calibration['abs_residual_median']:.4f}")
    print(f"  Thresholds:")
    for pct_key, val in calibration["thresholds"].items():
        print(f"    {pct_key}: {val:.4f}")
    print(f"  Recommended recovery gate (p99.9): {calibration['recommended_recovery_gate']:.4f}")

    # Save model
    model_config = {
        "model": {
            "input_size": 2,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "output_size": 1,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "seed": SEED,
            "best_epoch": train_info["best_epoch"],
        },
        "window_length": WINDOW_LENGTH,
        "features": ["voltage", "current"],
        "target": "y_true",
        "calibration": calibration,
    }
    save_auxiliary_model(model, normalization, model_config, WEIGHTS_PATH, CONFIG_PATH)
    print(f"\n  Model saved to: {WEIGHTS_PATH}")
    print(f"  Config saved to: {CONFIG_PATH}")

    # Save metrics
    all_metrics = {
        **test_metrics,
        "per_trajectory_test": per_traj,
        "per_trajectory_val": per_traj_val,
        "model_params": sum(p.numel() for p in model.parameters()),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8",
    )
    print(f"  Metrics saved to: {METRICS_PATH}")

    # Save calibration
    CALIBRATION_PATH.write_text(
        json.dumps(calibration, indent=2), encoding="utf-8",
    )
    print(f"  Calibration saved to: {CALIBRATION_PATH}")

    print("\n" + "=" * 60)
    print("DONE — Auxiliary model trained and calibrated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
