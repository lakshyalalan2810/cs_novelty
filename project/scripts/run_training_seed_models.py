"""Run the preregistered V3 training-seed model and calibration stages.

This runner is intentionally limited to the model-training/calibration half of
the training-seed robustness study.  It trains the paired 2027 and 2028 main
and auxiliary LSTMs, reproduces the canonical 2026 calibration without
retraining 2026, and writes only to the dedicated study namespaces.

It does not execute any closed-loop experiment.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auxiliary_sensor_model import (  # noqa: E402
    AuxiliarySpeedEstimator,
    build_auxiliary_sequences,
    fit_auxiliary_normalization,
    load_auxiliary_model,
    normalize_auxiliary,
)
from data_utils import build_training_sequences  # noqa: E402
from lstm_model import LSTMForecaster, recursive_forecast  # noqa: E402
from reliability import calibrate_cusum, guarded_predict, load_lstm_model, temporal_residual_features  # noqa: E402


STUDY = "training_seed_robustness"
TRAINING_SEEDS = (2027, 2028)
PAIR_SEEDS = (2026, 2027, 2028)
EXPECTED_SPLITS = {
    "train": [0, 1, 2, 4, 5, 6, 7, 10, 12, 13, 14, 18, 20, 21, 24, 26, 27, 29],
    "validation": [3, 9, 11, 16, 19, 23],
    "test": [8, 15, 17, 22, 25, 28],
}

DATASET = PROJECT / "data/processed/dc_motor_lstm_dataset.npz"
CANONICAL_MAIN_WEIGHTS = PROJECT / "results/lstm_model_weights.pt"
CANONICAL_MAIN_CONFIG = PROJECT / "results/configs/lstm_model_config.json"
CANONICAL_AUX_WEIGHTS = PROJECT / "results/auxiliary_model_weights.pt"
CANONICAL_AUX_CONFIG = PROJECT / "results/configs/v2_auxiliary_config.json"
CANONICAL_SENSOR_CONFIG = PROJECT / "results/configs/reliability_final_config.json"
CANONICAL_V3_CALIBRATION = PROJECT / "results/configs/v3_arbitration_calibration.json"

RESULTS_ROOT = PROJECT / "results/training_seed_robustness"
MODELS_ROOT = PROJECT / "models/training_seed_robustness"
CONFIG_ROOT = RESULTS_ROOT / "configs"
TRAINING_ROOT = RESULTS_ROOT / "training"
PROTOCOL_MANIFEST = CONFIG_ROOT / "training_protocol_manifest.json"
MODEL_PAIR_MANIFEST = RESULTS_ROOT / "model_pair_manifest.json"

MAIN_MODEL_CONFIG = {
    "input_size": 2,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "output_size": 1,
    "residual": True,
}
MAIN_TRAINING_CONFIG = {
    "batch_size": 512,
    "learning_rate": 1e-3,
    "max_epochs": 60,
    "patience": 8,
    "min_delta": 1e-6,
}
AUX_MODEL_CONFIG = {
    "input_size": 2,
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.0,
    "output_size": 1,
}
AUX_TRAINING_CONFIG = {
    "batch_size": 512,
    "learning_rate": 1e-3,
    "max_epochs": 100,
    "patience": 10,
    "min_delta": 1e-6,
}
WINDOW_LENGTH = 20
MAX_RECURSIVE_HORIZON = 15
ROLLOUT_STRIDE = 20

SENSOR_ENTER = 3
SENSOR_EXIT = 5
SENSOR_EWMA_ALPHA = 0.10
SENSOR_ROLLING_WINDOW = 20
SENSOR_TARGET_VALIDATION_FAR = 0.001
SENSOR_PERCENTILES = (90.0, 95.0, 97.5, 99.0, 99.5, 99.9)

ARBITRATION_EWMA_ALPHA = 0.05
STARTUP_BLANKING_TIME = 1.0
AGREEMENT_PERCENTILE = 90.0
MISMATCH_PERCENTILE = 99.9
RECOVERY_PERCENTILE = 95.0
RECOVERY_COUNT = 50
AUX_SPEED_BOUNDS = [0.0, 100.0]

# CPU LSTM inference can differ by a few final float32 ULPs across compatible
# PyTorch/BLAS execution details.  2e-6 is far below any scientific threshold
# scale here while still rejecting an algorithmically different calibration.
REPRO_ATOL = 2e-6
REPRO_RTOL = 2e-6

NO_FAULT_CALIBRATION_STATEMENT = (
    "Calibration uses only the frozen clean, fault-free validation trajectories; "
    "no test trajectories, fault scenarios, closed-loop results, or simulation-seed results are used."
)

EXPECTED_HASHES = {
    "dataset": "5f77f76468908656b35dabbc48e8aa20cc0583eae3724cc989278d94fe2bab26",
    "canonical_main_weights": "d1f4178199f682560f164ccac1d56816aae341eb05fbed6ac7ad73a4e2056297",
    "canonical_main_config": "9254bbd088b0c9f7bd7ba71ee75c09fe4501ce28eb8e26efab7fb7d5130ce4cc",
    "canonical_aux_weights": "619f677ccc63c6a40c863858d8eee4175f72b1770dc6c8ba202e7b92c2768480",
    "canonical_aux_config": "04fd322ba365b394751e4f60163ebea4b2164ff6fe699d7d2c2cc052eaae8ad2",
    "canonical_sensor_config": "a093f5a436f0a4737a7d7f4092383e64ca0a9f64e54adc4f0f41473da4a8e71b",
    "canonical_v3_calibration": "1dfcedf45110971e776c9c9ec6639b556b2c058d3a0535d9a69f5e23be99b1c8",
    "main_training_notebook": "cb732e2944819bb3f68fe2e698871869720a21590ef2d26c9cca9f18649aa121",
    "aux_training_script": "699f7668b2dc76aae0065226fa998f1031f9a01392a9620f9fb5d0a4a548921e",
    "sensor_calibration_notebook": "82253ac4cb2a068fab42fc381015af1ac2bb4919babec36af790c0896ca2469e",
    "v3_calibration_script": "310f886193ca6bc1584f7e0985794efc32dd08535b1dfc4e1992a9dc88ba26c2",
    "data_utils_source": "a09203dacf6c30ca67460cc81d8f8311c83ace79a3f0bf810ce3dac4eaaf7d15",
    "lstm_model_source": "88f1a38c885572414b7f3077e6137285941f635df835e059b0c61f4dfdcd4ec2",
    "auxiliary_model_source": "e2a1cdba68fe97630b882c2a36dd8b21077d64705b5f5d3db56bb1efa5387bae",
    "reliability_source": "909fd117453e66a6a034c230a8de47ee63f6d8623836e642c3788fe675931f26",
}

SOURCE_PATHS = {
    "main_training_notebook": PROJECT / "notebooks/03_lstm_training.ipynb",
    "aux_training_script": PROJECT / "scripts/train_auxiliary_model.py",
    "sensor_calibration_notebook": PROJECT / "notebooks/04c_reliability_finalization.ipynb",
    "v3_calibration_script": PROJECT / "scripts/calibrate_v3_arbitration.py",
    "data_utils_source": PROJECT / "src/data_utils.py",
    "lstm_model_source": PROJECT / "src/lstm_model.py",
    "auxiliary_model_source": PROJECT / "src/auxiliary_sensor_model.py",
    "reliability_source": PROJECT / "src/reliability.py",
}


def sha256(path: Path) -> str:
    """Return a file SHA256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    """Return a project-relative POSIX path."""
    return path.relative_to(PROJECT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def assert_close(name: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=REPRO_RTOL, atol=REPRO_ATOL):
        raise AssertionError(
            f"2026 calibration reproduction failed for {name}: "
            f"actual={actual!r}, expected={expected!r}, "
            f"rtol={REPRO_RTOL}, atol={REPRO_ATOL}"
        )


def ensure_finite(name: str, values: list[float] | np.ndarray) -> None:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def artifact_paths(seed: int) -> dict[str, Path]:
    if seed == 2026:
        return {
            "main_weights": CANONICAL_MAIN_WEIGHTS,
            "main_config": CANONICAL_MAIN_CONFIG,
            "aux_weights": CANONICAL_AUX_WEIGHTS,
            "aux_config": CANONICAL_AUX_CONFIG,
            "sensor_calibration": CONFIG_ROOT / "sensor_seed2026_calibration.json",
            "v3_calibration": CONFIG_ROOT / "v3_seed2026_calibration.json",
            "pair_config": CONFIG_ROOT / "v3_seed2026_config.json",
        }
    return {
        "main_weights": MODELS_ROOT / f"main_seed_{seed}.pt",
        "main_config": CONFIG_ROOT / f"main_seed{seed}_config.json",
        "main_history": TRAINING_ROOT / f"main_seed{seed}_history.npz",
        "main_metrics": TRAINING_ROOT / f"main_seed{seed}_metrics.json",
        "aux_weights": MODELS_ROOT / f"aux_seed_{seed}.pt",
        "aux_config": CONFIG_ROOT / f"aux_seed{seed}_config.json",
        "aux_history": TRAINING_ROOT / f"aux_seed{seed}_history.npz",
        "aux_metrics": TRAINING_ROOT / f"aux_seed{seed}_metrics.json",
        "sensor_calibration": CONFIG_ROOT / f"sensor_seed{seed}_calibration.json",
        "v3_calibration": CONFIG_ROOT / f"v3_seed{seed}_calibration.json",
        "pair_config": CONFIG_ROOT / f"v3_seed{seed}_config.json",
    }


def intended_new_outputs() -> list[Path]:
    paths = [PROTOCOL_MANIFEST, MODEL_PAIR_MANIFEST]
    for seed in PAIR_SEEDS:
        seed_paths = artifact_paths(seed)
        for key, path in seed_paths.items():
            if seed == 2026 and key in {"main_weights", "main_config", "aux_weights", "aux_config"}:
                continue
            paths.append(path)
    return paths


def preflight_frozen_inputs() -> tuple[np.lib.npyio.NpzFile, dict[str, str]]:
    """Fail closed if frozen inputs or output namespace do not match preregistration."""
    fixed_paths = {
        "dataset": DATASET,
        "canonical_main_weights": CANONICAL_MAIN_WEIGHTS,
        "canonical_main_config": CANONICAL_MAIN_CONFIG,
        "canonical_aux_weights": CANONICAL_AUX_WEIGHTS,
        "canonical_aux_config": CANONICAL_AUX_CONFIG,
        "canonical_sensor_config": CANONICAL_SENSOR_CONFIG,
        "canonical_v3_calibration": CANONICAL_V3_CALIBRATION,
        **SOURCE_PATHS,
    }
    current_hashes: dict[str, str] = {}
    for name, path in fixed_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required frozen input is missing: {path}")
        current_hashes[name] = sha256(path)
        expected = EXPECTED_HASHES[name]
        if current_hashes[name] != expected:
            raise RuntimeError(
                f"frozen input hash drift for {name}: "
                f"expected {expected}, found {current_hashes[name]}"
            )

    collisions = [path for path in intended_new_outputs() if path.exists()]
    if collisions:
        formatted = "\n".join(f"  - {rel(path)}" for path in collisions)
        raise FileExistsError(
            "study output already exists; refusing to overwrite reproducibility evidence:\n"
            + formatted
        )

    data = np.load(DATASET)
    actual_splits = {
        split: [int(value) for value in data[f"split_run_ids_{split}"]]
        for split in ("train", "validation", "test")
    }
    if actual_splits != EXPECTED_SPLITS:
        raise AssertionError(f"frozen trajectory split drift: {actual_splits}")
    if int(data["window_length"]) != WINDOW_LENGTH:
        raise AssertionError("frozen dataset window length changed")
    if data["feature_names"].tolist() != ["voltage", "y_measured"]:
        raise AssertionError("frozen main feature definition changed")
    if str(data["target_name"]) != "y_true":
        raise AssertionError("frozen target definition changed")

    canonical_main_config = load_json(CANONICAL_MAIN_CONFIG)
    if canonical_main_config["model"] != MAIN_MODEL_CONFIG:
        raise AssertionError("canonical main architecture differs from preregistered architecture")
    canonical_main_training = canonical_main_config["training"]
    for key, expected in MAIN_TRAINING_CONFIG.items():
        if canonical_main_training[key] != expected:
            raise AssertionError(f"canonical main training hyperparameter changed: {key}")

    canonical_aux_config = load_json(CANONICAL_AUX_CONFIG)
    if canonical_aux_config["model"] != AUX_MODEL_CONFIG:
        raise AssertionError("canonical auxiliary architecture differs from preregistered architecture")
    canonical_aux_training = canonical_aux_config["training"]
    for key, expected in AUX_TRAINING_CONFIG.items():
        if canonical_aux_training[key] != expected:
            raise AssertionError(f"canonical auxiliary training hyperparameter changed: {key}")
    if int(canonical_aux_config["window_length"]) != WINDOW_LENGTH:
        raise AssertionError("canonical auxiliary window length changed")
    if canonical_aux_config["features"] != ["voltage", "current"]:
        raise AssertionError("canonical auxiliary feature definition changed")
    if canonical_aux_config["target"] != "y_true":
        raise AssertionError("canonical auxiliary target definition changed")

    return data, current_hashes


def protocol_payload(data: np.lib.npyio.NpzFile, source_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "study": STUDY,
        "training_pairs": [f"P{seed}" for seed in PAIR_SEEDS],
        "new_training_seeds": list(TRAINING_SEEDS),
        "canonical_seed_2026_action": "reuse frozen canonical weights; do not retrain",
        "dataset": {
            "path": rel(DATASET),
            "sha256": source_hashes["dataset"],
            "dataset_seed": int(data["seed"]),
            "window_length": int(data["window_length"]),
            "main_arrays_are_pre_normalized": True,
            "split_run_ids": EXPECTED_SPLITS,
            "sample_counts": {
                "train": int(len(data["X_train"])),
                "validation": int(len(data["X_validation"])),
                "test": int(len(data["X_test"])),
            },
        },
        "main_model": {
            "architecture": MAIN_MODEL_CONFIG,
            "training": MAIN_TRAINING_CONFIG,
            "optimizer": "torch.optim.Adam with frozen default options except learning_rate=1e-3",
            "loss": "torch.nn.MSELoss",
            "features": ["voltage", "y_measured"],
            "target": "y_true",
            "normalization": "consume frozen train-only-normalized X/y arrays and saved normalization statistics from dataset archive",
            "rng_sources": {
                "numpy": "np.random.seed(training_seed), matching notebook setup",
                "torch_global": "torch.manual_seed(training_seed): weight initialization and inter-layer LSTM dropout",
                "train_dataloader": "dedicated torch.Generator().manual_seed(training_seed) for shuffle order",
                "determinism": "torch.use_deterministic_algorithms(True); CPU; notebook-compatible thread cap <=4",
            },
        },
        "auxiliary_model": {
            "architecture": AUX_MODEL_CONFIG,
            "training": AUX_TRAINING_CONFIG,
            "optimizer": "torch.optim.Adam with frozen default options except learning_rate=1e-3",
            "loss": "torch.nn.MSELoss",
            "features": ["voltage", "current"],
            "target": "y_true",
            "normalization": "fit feature/target Z-score statistics on frozen auxiliary training trajectories only",
            "rng_sources": {
                "numpy": "np.random.seed(training_seed)",
                "torch_global": "torch.manual_seed(training_seed) before DataLoader/model; governs model initialization and shuffled sampler exactly as frozen script",
                "train_dataloader": "shuffle=True with no explicit generator, matching frozen auxiliary trainer",
                "dropout": "0.0 with one LSTM layer",
            },
        },
        "sensor_calibration": {
            "population": "frozen validation trajectories only",
            "validation_run_ids": EXPECTED_SPLITS["validation"],
            "torch_threads": "min(4, current thread count), matching notebook 04c setup",
            "instant_gate": "p99.9 absolute raw measured-minus-main residual",
            "cusum_percentile_grid": list(SENSOR_PERCENTILES),
            "cusum_allowance_sigma": 0.5,
            "selection": f"first percentile with debounced validation FAR <= {SENSOR_TARGET_VALIDATION_FAR}; otherwise final grid percentile",
            "enter_count": SENSOR_ENTER,
            "exit_count": SENSOR_EXIT,
            "signed_ewma_alpha": SENSOR_EWMA_ALPHA,
            "signed_ewma_percentile": 99.9,
        },
        "v3_arbitration_calibration": {
            "population": "frozen validation trajectories only",
            "validation_run_ids": EXPECTED_SPLITS["validation"],
            "torch_threads": 1,
            "agreement_percentile": AGREEMENT_PERCENTILE,
            "mismatch_ewma_percentile": MISMATCH_PERCENTILE,
            "recovery_ewma_percentile": RECOVERY_PERCENTILE,
            "aux_recovery_percentile": 99.9,
            "ewma_alpha": ARBITRATION_EWMA_ALPHA,
            "startup_blanking_time": STARTUP_BLANKING_TIME,
            "recovery_count": RECOVERY_COUNT,
            "aux_speed_bounds": AUX_SPEED_BOUNDS,
        },
        "calibration_data_policy": NO_FAULT_CALIBRATION_STATEMENT,
        "reproduction_tolerance": {
            "rtol": REPRO_RTOL,
            "atol": REPRO_ATOL,
            "reason": "permits only final float32 CPU inference/reduction ULP-scale differences; rejects scientifically material threshold drift",
        },
        "frozen_source_hashes": {
            name: {"path": rel(SOURCE_PATHS[name]), "sha256": source_hashes[name]}
            for name in SOURCE_PATHS
        },
        "canonical_model_hashes": {
            "main_weights": source_hashes["canonical_main_weights"],
            "main_config": source_hashes["canonical_main_config"],
            "aux_weights": source_hashes["canonical_aux_weights"],
            "aux_config": source_hashes["canonical_aux_config"],
            "sensor_config": source_hashes["canonical_sensor_config"],
            "v3_calibration": source_hashes["canonical_v3_calibration"],
        },
    }


def regression_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - true
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error**2) / np.sum((true - true.mean()) ** 2)),
        "nrmse_range": float(rmse / (true.max() - true.min())),
    }


@torch.no_grad()
def evaluate_mse(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    for inputs, targets in loader:
        predictions = model(inputs)
        squared_error += torch.sum((predictions - targets) ** 2).item()
        count += targets.numel()
    return squared_error / count


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    return np.concatenate([model(inputs).cpu().numpy() for inputs, _ in loader])


def train_main_model(seed: int, data: np.lib.npyio.NpzFile, paths: dict[str, Path]) -> None:
    """Train one main model using the exact frozen notebook protocol except seed/output paths."""
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"main training is preregistered only for seeds {TRAINING_SEEDS}, got {seed}")
    if not paths["main_weights"].is_relative_to(MODELS_ROOT):
        raise ValueError("new main weights must stay inside models/training_seed_robustness")
    np.random.seed(seed)
    torch.manual_seed(seed)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, previous_threads))
    started = time.perf_counter()
    try:
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_validation = data["X_validation"]
        y_validation = data["y_validation"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        def make_loader(inputs: np.ndarray, targets: np.ndarray, shuffle: bool = False) -> DataLoader:
            dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
            generator = torch.Generator().manual_seed(seed) if shuffle else None
            return DataLoader(
                dataset,
                batch_size=MAIN_TRAINING_CONFIG["batch_size"],
                shuffle=shuffle,
                generator=generator,
            )

        # Construction order is frozen: loaders first, then model initialization.
        train_loader = make_loader(X_train, y_train, shuffle=True)
        validation_loader = make_loader(X_validation, y_validation)
        test_loader = make_loader(X_test, y_test)
        model = LSTMForecaster(**MAIN_MODEL_CONFIG)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=MAIN_TRAINING_CONFIG["learning_rate"])

        history = {"train_loss": [], "validation_loss": []}
        best_validation_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, MAIN_TRAINING_CONFIG["max_epochs"] + 1):
            model.train()
            total_loss = 0.0
            count = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"main seed {seed} produced non-finite training loss")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(inputs)
                count += len(inputs)

            train_loss = total_loss / count
            validation_loss = evaluate_mse(model, validation_loader)
            if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
                raise FloatingPointError(f"main seed {seed} produced non-finite epoch metrics")
            history["train_loss"].append(float(train_loss))
            history["validation_loss"].append(float(validation_loss))

            if validation_loss < best_validation_loss - MAIN_TRAINING_CONFIG["min_delta"]:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= MAIN_TRAINING_CONFIG["patience"]:
                    break

        if best_state is None:
            raise RuntimeError(f"main seed {seed} never produced a valid checkpoint")
        model.load_state_dict(best_state)
        model.eval()

        input_mean = data["normalization_input_mean"]
        input_std = data["normalization_input_std"]
        target_mean = data["normalization_target_mean"]
        target_std = data["normalization_target_std"]

        test_prediction_normalized = predict_loader(model, test_loader)
        test_prediction = test_prediction_normalized[:, 0] * target_std[0] + target_mean[0]
        test_true = y_test[:, 0] * target_std[0] + target_mean[0]
        persistence_prediction = X_test[:, -1, 1] * input_std[1] + input_mean[1]
        ensure_finite(f"main seed {seed} test prediction", test_prediction)
        test_metrics = regression_metrics(test_true, test_prediction)
        persistence_metrics = regression_metrics(test_true, persistence_prediction)

        histories: list[np.ndarray] = []
        future_voltages: list[np.ndarray] = []
        recursive_true: list[np.ndarray] = []
        for run_id_value in data["split_run_ids_test"]:
            run_id = int(run_id_value)
            features = np.column_stack((data["voltage"][run_id], data["y_measured"][run_id]))
            normalized_features = (features - input_mean) / input_std
            normalized_voltage = (data["voltage"][run_id] - input_mean[0]) / input_std[0]
            for target_index in range(
                WINDOW_LENGTH,
                len(data["time"]) - MAX_RECURSIVE_HORIZON + 1,
                ROLLOUT_STRIDE,
            ):
                histories.append(normalized_features[target_index - WINDOW_LENGTH : target_index])
                future_voltages.append(
                    normalized_voltage[target_index : target_index + MAX_RECURSIVE_HORIZON]
                )
                recursive_true.append(
                    data["y_true"][
                        run_id,
                        target_index : target_index + MAX_RECURSIVE_HORIZON,
                    ]
                )

        history_tensor = torch.tensor(np.asarray(histories), dtype=torch.float32)
        future_voltage_tensor = torch.tensor(np.asarray(future_voltages), dtype=torch.float32)
        normalization_tensors = [
            torch.tensor(values, dtype=torch.float32)
            for values in (input_mean, input_std, target_mean, target_std)
        ]
        recursive_prediction_normalized = recursive_forecast(
            model,
            history_tensor,
            future_voltage_tensor,
            *normalization_tensors,
        ).cpu().numpy()
        recursive_prediction = recursive_prediction_normalized * target_std[0] + target_mean[0]
        recursive_true_array = np.asarray(recursive_true)
        ensure_finite(f"main seed {seed} recursive prediction", recursive_prediction.ravel())
        horizon_rmse = np.sqrt(
            np.mean((recursive_prediction - recursive_true_array) ** 2, axis=0)
        )

        runtime_seconds = float(time.perf_counter() - started)
        config_payload = {
            "model": MAIN_MODEL_CONFIG,
            "training": {**MAIN_TRAINING_CONFIG, "seed": seed, "best_epoch": best_epoch},
            "dataset": rel(DATASET),
            "window_length": WINDOW_LENGTH,
            "features": data["feature_names"].tolist(),
            "target": str(data["target_name"]),
            "normalization": {
                "input_mean": input_mean.tolist(),
                "input_std": input_std.tolist(),
                "target_mean": target_mean.tolist(),
                "target_std": target_std.tolist(),
            },
        }
        metrics_payload = {
            "training_seed": seed,
            "best_validation_mse_normalized": float(best_validation_loss),
            "best_epoch": best_epoch,
            "test": test_metrics,
            "persistence": persistence_metrics,
            "recursive_rmse_by_horizon": [float(value) for value in horizon_rmse],
            "recursive_key_horizons": {
                "H5": float(horizon_rmse[4]),
                "H10": float(horizon_rmse[9]),
                "H15": float(horizon_rmse[14]),
            },
            "training_runtime_seconds": runtime_seconds,
            "model_params": int(sum(parameter.numel() for parameter in model.parameters())),
        }

        paths["main_weights"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), paths["main_weights"])
        write_json(paths["main_config"], config_payload)
        paths["main_history"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            paths["main_history"],
            train_loss=np.asarray(history["train_loss"]),
            validation_loss=np.asarray(history["validation_loss"]),
            best_epoch=np.asarray(best_epoch),
        )
        write_json(paths["main_metrics"], metrics_payload)

        reloaded_model, _ = load_lstm_model(paths["main_weights"], paths["main_config"])
        with torch.no_grad():
            if not torch.allclose(
                reloaded_model(torch.from_numpy(X_test[:32])),
                model(torch.from_numpy(X_test[:32])),
            ):
                raise RuntimeError(f"main seed {seed} serialization reproduction failed")
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


def load_raw_dataset(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    return {
        "voltage": data["voltage"],
        "current": data["current"],
        "y_true": data["y_true"],
        "y_measured": data["y_measured"],
        "load_torque": data["load_torque"],
    }


def build_aux_split(
    raw: dict[str, np.ndarray], run_ids: list[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    all_inputs, all_targets = [], []
    for run_id_value in run_ids:
        run_id = int(run_id_value)
        inputs, targets = build_auxiliary_sequences(
            raw["voltage"][run_id],
            raw["current"][run_id],
            raw["y_true"][run_id],
            window_length=WINDOW_LENGTH,
        )
        all_inputs.append(inputs)
        all_targets.append(targets)
    return np.concatenate(all_inputs), np.concatenate(all_targets)


@torch.no_grad()
def aux_predict_physical(
    model: AuxiliarySpeedEstimator,
    inputs: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> np.ndarray:
    model.eval()
    # Match train_auxiliary_model.py exactly: its clean-test, per-trajectory,
    # and auxiliary-threshold evaluations each use one full forward pass.
    normalized = model(torch.from_numpy(inputs)).numpy()[:, 0]
    result = normalized * float(normalization["target_std"][0]) + float(
        normalization["target_mean"][0]
    )
    ensure_finite("auxiliary prediction", result)
    return result


def auxiliary_clean_validation_calibration(
    model: AuxiliarySpeedEstimator,
    raw: dict[str, np.ndarray],
    validation_ids: list[int],
    normalization: dict[str, np.ndarray],
) -> dict[str, Any]:
    residuals = []
    for run_id in validation_ids:
        inputs, targets = build_auxiliary_sequences(
            raw["voltage"][run_id],
            raw["current"][run_id],
            raw["y_true"][run_id],
            WINDOW_LENGTH,
        )
        inputs_normalized, _ = normalize_auxiliary(inputs, targets, normalization)
        prediction = aux_predict_physical(model, inputs_normalized, normalization)
        measured = raw["y_measured"][run_id, WINDOW_LENGTH : WINDOW_LENGTH + len(prediction)]
        residuals.append(measured - prediction)

    residual = np.concatenate(residuals)
    absolute = np.abs(residual)
    thresholds = {
        f"p{percentile}": float(np.percentile(absolute, percentile))
        for percentile in SENSOR_PERCENTILES
    }
    return {
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "abs_residual_mean": float(np.mean(absolute)),
        "abs_residual_median": float(np.median(absolute)),
        "thresholds": thresholds,
        "recommended_recovery_gate": thresholds["p99.9"],
        "calibration_source": "clean validation data only",
        "num_samples": int(len(residual)),
    }


def evaluate_aux_split(
    model: AuxiliarySpeedEstimator,
    inputs: np.ndarray,
    targets: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> dict[str, float | int]:
    normalized_inputs, _ = normalize_auxiliary(inputs, targets, normalization)
    prediction = aux_predict_physical(model, normalized_inputs, normalization)
    true = targets[:, 0].astype(float)
    error = prediction - true
    ss_res = np.sum(error**2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return {
        "test_rmse": float(np.sqrt(np.mean(error**2))),
        "test_mae": float(np.mean(np.abs(error))),
        "test_r2": float(1 - ss_res / ss_tot),
        "test_samples": int(len(true)),
    }


def evaluate_aux_per_trajectory(
    model: AuxiliarySpeedEstimator,
    raw: dict[str, np.ndarray],
    run_ids: list[int],
    normalization: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for run_id in run_ids:
        inputs, targets = build_auxiliary_sequences(
            raw["voltage"][run_id],
            raw["current"][run_id],
            raw["y_true"][run_id],
            WINDOW_LENGTH,
        )
        normalized_inputs, _ = normalize_auxiliary(inputs, targets, normalization)
        prediction = aux_predict_physical(model, normalized_inputs, normalization)
        true = targets[:, 0].astype(float)
        error = prediction - true
        load_torque = raw["load_torque"][run_id]
        load_range = float(load_torque.max() - load_torque.min())
        rows.append(
            {
                "run_id": run_id,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "has_load_change": bool(load_range > 0.01),
                "load_torque_range": load_range,
            }
        )
    return rows


def train_aux_model(seed: int, data: np.lib.npyio.NpzFile, paths: dict[str, Path]) -> None:
    """Train one auxiliary model using the exact frozen script semantics except seed/output paths."""
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"auxiliary training is preregistered only for seeds {TRAINING_SEEDS}, got {seed}")
    if not paths["aux_weights"].is_relative_to(MODELS_ROOT):
        raise ValueError("new auxiliary weights must stay inside models/training_seed_robustness")
    # Preserve the frozen trainer's global RNG semantics: seed before sequence
    # preparation, DataLoader construction, and model initialization.
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.perf_counter()

    raw = load_raw_dataset(data)
    train_ids = EXPECTED_SPLITS["train"]
    validation_ids = EXPECTED_SPLITS["validation"]
    test_ids = EXPECTED_SPLITS["test"]
    X_train, y_train = build_aux_split(raw, train_ids)
    X_validation, y_validation = build_aux_split(raw, validation_ids)
    normalization = fit_auxiliary_normalization(X_train, y_train)
    X_train_normalized, y_train_normalized = normalize_auxiliary(X_train, y_train, normalization)
    X_validation_normalized, y_validation_normalized = normalize_auxiliary(
        X_validation, y_validation, normalization
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_normalized), torch.from_numpy(y_train_normalized)
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(X_validation_normalized), torch.from_numpy(y_validation_normalized)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=AUX_TRAINING_CONFIG["batch_size"],
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=AUX_TRAINING_CONFIG["batch_size"],
    )

    model = AuxiliarySpeedEstimator(**AUX_MODEL_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=AUX_TRAINING_CONFIG["learning_rate"])
    criterion = nn.MSELoss()
    best_validation_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, AUX_TRAINING_CONFIG["max_epochs"] + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"aux seed {seed} produced non-finite training loss")
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                loss = criterion(model(X_batch), y_batch)
                validation_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise FloatingPointError(f"aux seed {seed} produced non-finite epoch metrics")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(validation_loss)

        if validation_loss < best_validation_loss - AUX_TRAINING_CONFIG["min_delta"]:
            best_validation_loss = validation_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= AUX_TRAINING_CONFIG["patience"]:
                break

    if best_state is None:
        raise RuntimeError(f"aux seed {seed} never produced a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    X_test, y_test = build_aux_split(raw, test_ids)
    test_metrics = evaluate_aux_split(model, X_test, y_test, normalization)
    per_trajectory_test = evaluate_aux_per_trajectory(model, raw, test_ids, normalization)
    per_trajectory_validation = evaluate_aux_per_trajectory(
        model, raw, validation_ids, normalization
    )
    clean_validation_calibration = auxiliary_clean_validation_calibration(
        model, raw, validation_ids, normalization
    )
    runtime_seconds = float(time.perf_counter() - started)

    serializable_normalization = {
        key: value.tolist() for key, value in normalization.items()
    }
    config_payload = {
        "model": AUX_MODEL_CONFIG,
        "training": {
            **AUX_TRAINING_CONFIG,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        "window_length": WINDOW_LENGTH,
        "features": ["voltage", "current"],
        "target": "y_true",
        "calibration": clean_validation_calibration,
        "normalization": serializable_normalization,
    }
    metrics_payload = {
        **test_metrics,
        "training_seed": seed,
        "best_validation_loss_normalized": float(best_validation_loss),
        "best_epoch": best_epoch,
        "per_trajectory_test": per_trajectory_test,
        "per_trajectory_val": per_trajectory_validation,
        "model_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_runtime_seconds": runtime_seconds,
    }

    paths["aux_weights"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), paths["aux_weights"])
    write_json(paths["aux_config"], config_payload)
    paths["aux_history"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["aux_history"],
        train_loss=np.asarray(history["train_loss"]),
        val_loss=np.asarray(history["val_loss"]),
        best_epoch=np.asarray(best_epoch),
    )
    write_json(paths["aux_metrics"], metrics_payload)

    reloaded_model, reloaded_config = load_auxiliary_model(paths["aux_weights"], paths["aux_config"])
    sample_inputs, sample_targets = build_aux_split(raw, [test_ids[0]])
    sample_inputs_normalized, _ = normalize_auxiliary(
        sample_inputs[:32], sample_targets[:32], normalization
    )
    with torch.no_grad():
        original = model(torch.from_numpy(sample_inputs_normalized))
        reloaded = reloaded_model(torch.from_numpy(sample_inputs_normalized))
        if not torch.allclose(original, reloaded):
            raise RuntimeError(f"aux seed {seed} serialization reproduction failed")
    if reloaded_config["training"]["seed"] != seed:
        raise RuntimeError(f"aux seed {seed} serialized config binding failed")


@torch.no_grad()
def deterministic_main_rows(
    model: LSTMForecaster,
    voltage: np.ndarray,
    measured: np.ndarray,
    normalization: dict[str, Any],
    batch_size: int = 512,
) -> np.ndarray:
    input_mean = np.asarray(normalization["input_mean"])
    input_std = np.asarray(normalization["input_std"])
    target_mean = float(normalization["target_mean"][0])
    target_std = float(normalization["target_std"][0])

    def make_windows(voltage_row: np.ndarray, measured_row: np.ndarray) -> np.ndarray:
        features = np.column_stack((voltage_row, measured_row))
        values = np.stack(
            [
                features[start : start + WINDOW_LENGTH]
                for start in range(len(features) - WINDOW_LENGTH)
            ]
        )
        return ((values - input_mean) / input_std).astype(np.float32)

    values = np.concatenate(
        [make_windows(v, y) for v, y in zip(voltage, measured)]
    )
    predicted = []
    model.eval()
    for start in range(0, len(values), batch_size):
        predicted.append(model(torch.from_numpy(values[start : start + batch_size])).cpu().numpy()[:, 0])
    physical = (np.concatenate(predicted) * target_std + target_mean).reshape(len(voltage), -1)
    ensure_finite("sensor calibration main prediction", physical.ravel())
    return physical


def calibrate_sensor_monitor(
    seed: int,
    model: LSTMForecaster,
    main_config: dict[str, Any],
    data: np.lib.npyio.NpzFile,
    main_model_hash: str,
    main_config_hash: str,
) -> dict[str, Any]:
    validation_ids = EXPECTED_SPLITS["validation"]
    validation_voltage = data["voltage"][validation_ids].astype(float)
    validation_measured = data["y_measured"][validation_ids].astype(float)
    normalization = main_config["normalization"]
    validation_prediction = deterministic_main_rows(
        model, validation_voltage, validation_measured, normalization
    )
    validation_residual = validation_measured[:, WINDOW_LENGTH:] - validation_prediction
    residual_center = float(np.median(validation_residual))
    instant_threshold = float(np.percentile(np.abs(validation_residual), 99.9))

    validation_temporal = [
        temporal_residual_features(
            row - residual_center,
            SENSOR_ROLLING_WINDOW,
            SENSOR_EWMA_ALPHA,
        )
        for row in validation_residual
    ]
    signed_ewma_threshold = float(
        np.percentile(
            np.abs(
                np.concatenate([features["signed_ewma"] for features in validation_temporal])
            ),
            99.9,
        )
    )
    absolute_ewma_threshold = float(
        np.percentile(
            np.concatenate([features["ewma"] for features in validation_temporal]),
            99.9,
        )
    )

    calibration_grid = []
    guarded_by_percentile: dict[float, dict[str, np.ndarray]] = {}
    for percentile in SENSOR_PERCENTILES:
        candidate = calibrate_cusum(validation_residual, percentile, allowance_sigma=0.5)
        guarded = guarded_predict(
            model,
            validation_voltage,
            validation_measured,
            normalization,
            WINDOW_LENGTH,
            instant_threshold,
            candidate,
            enter_count=SENSOR_ENTER,
            exit_count=SENSOR_EXIT,
        )
        guarded_by_percentile[percentile] = guarded
        calibration_grid.append(
            {
                "percentile": percentile,
                "instant_gate": instant_threshold,
                "cusum_threshold": float(candidate["threshold"]),
                "validation_false_alarm_rate": float(np.mean(guarded["sensor_alarm"])),
            }
        )

    eligible = [
        row
        for row in calibration_grid
        if row["validation_false_alarm_rate"] <= SENSOR_TARGET_VALIDATION_FAR
    ]
    selected = eligible[0] if eligible else calibration_grid[-1]
    selected_percentile = float(selected["percentile"])
    sensor_cusum = calibrate_cusum(
        validation_residual,
        selected_percentile,
        allowance_sigma=0.5,
    )
    selected_guarded = guarded_by_percentile[selected_percentile]
    sample_count = int(validation_residual.size)

    values_to_check = [
        instant_threshold,
        residual_center,
        signed_ewma_threshold,
        absolute_ewma_threshold,
        sensor_cusum["center"],
        sensor_cusum["allowance"],
        sensor_cusum["threshold"],
    ]
    ensure_finite(f"sensor calibration seed {seed}", values_to_check)
    if instant_threshold <= 0 or sensor_cusum["threshold"] <= 0 or sensor_cusum["allowance"] < 0:
        raise AssertionError(f"sensor seed {seed} calibration produced invalid ranges")

    return {
        "study": STUDY,
        "pair_id": f"P{seed}",
        "training_seed": seed,
        "calibration_population": "clean validation trajectories only",
        "no_fault_data_statement": NO_FAULT_CALIBRATION_STATEMENT,
        "validation_run_ids": validation_ids,
        "sample_count": sample_count,
        "dataset_sha256": sha256(DATASET),
        "model_sha256": {
            "main": main_model_hash,
            "main_config": main_config_hash,
        },
        "algorithm": {
            "source_path": rel(SOURCE_PATHS["sensor_calibration_notebook"]),
            "source_sha256": sha256(SOURCE_PATHS["sensor_calibration_notebook"]),
            "reliability_source_sha256": sha256(SOURCE_PATHS["reliability_source"]),
            "instant_percentile": 99.9,
            "cusum_percentile_grid": list(SENSOR_PERCENTILES),
            "allowance_sigma": 0.5,
            "target_validation_false_alarm_rate": SENSOR_TARGET_VALIDATION_FAR,
            "enter_count": SENSOR_ENTER,
            "exit_count": SENSOR_EXIT,
            "signed_ewma_alpha": SENSOR_EWMA_ALPHA,
            "rolling_window": SENSOR_ROLLING_WINDOW,
        },
        "calibration_grid": calibration_grid,
        "computed_values": {
            "instant_threshold": instant_threshold,
            "cusum_percentile": selected_percentile,
            "residual_center": residual_center,
            "center": float(sensor_cusum["center"]),
            "allowance": float(sensor_cusum["allowance"]),
            "threshold": float(sensor_cusum["threshold"]),
            "signed_ewma_threshold": signed_ewma_threshold,
            "absolute_ewma_threshold": absolute_ewma_threshold,
            "enter_count": SENSOR_ENTER,
            "exit_count": SENSOR_EXIT,
        },
        "clean_validation_false_alarm_statistics": {
            "selected_debounced_sensor_alarm_rate": float(np.mean(selected_guarded["sensor_alarm"])),
            "immediate_substitution_rate": float(np.mean(selected_guarded["substituted"])),
            "instantaneous_residual_exceedance_rate": float(
                np.mean(np.abs(validation_residual) > instant_threshold)
            ),
        },
    }


def post_blanking_ewma(
    times: np.ndarray,
    residual: np.ndarray,
    alpha: float,
    startup_blanking_time: float,
) -> np.ndarray:
    values = []
    ewma = 0.0
    for time_value, residual_value in zip(times, residual):
        if time_value >= startup_blanking_time:
            ewma = (1 - alpha) * ewma + alpha * float(residual_value)
            values.append(ewma)
    return np.asarray(values, dtype=float)


@torch.no_grad()
def predict_physical_batched(
    model: nn.Module,
    inputs: np.ndarray,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    values = []
    model.eval()
    for start in range(0, len(inputs), 1024):
        values.append(model(torch.from_numpy(inputs[start : start + 1024]))[:, 0].numpy())
    result = np.concatenate(values) * target_std + target_mean
    ensure_finite("V3 calibration prediction", result)
    return result


def calibrate_v3_pair(
    seed: int,
    main_model: LSTMForecaster,
    main_config: dict[str, Any],
    aux_model: AuxiliarySpeedEstimator,
    aux_config: dict[str, Any],
    data: np.lib.npyio.NpzFile,
    model_hashes: dict[str, str],
) -> dict[str, Any]:
    validation_ids = EXPECTED_SPLITS["validation"]
    main_normalization = main_config["normalization"]
    aux_normalization = aux_config["normalization"]
    agreement: list[float] = []
    aux_residual: list[float] = []
    ewma_values: list[float] = []
    aux_predictions: list[float] = []

    for run_id in validation_ids:
        trajectory = {
            "voltage": data["voltage"][run_id],
            "y_measured": data["y_measured"][run_id],
            "y_true": data["y_true"][run_id],
            "run_id": np.full(len(data["time"]), run_id),
        }
        main_inputs, _, _ = build_training_sequences(trajectory, WINDOW_LENGTH, 1)
        main_inputs = (
            (main_inputs - np.asarray(main_normalization["input_mean"]))
            / np.asarray(main_normalization["input_std"])
        ).astype(np.float32)
        aux_inputs, aux_targets = build_auxiliary_sequences(
            data["voltage"][run_id],
            data["current"][run_id],
            data["y_true"][run_id],
            WINDOW_LENGTH,
        )
        aux_inputs, _ = normalize_auxiliary(aux_inputs, aux_targets, aux_normalization)
        y_main = predict_physical_batched(
            main_model,
            main_inputs,
            float(main_normalization["target_mean"][0]),
            float(main_normalization["target_std"][0]),
        )
        y_aux = predict_physical_batched(
            aux_model,
            aux_inputs,
            float(aux_normalization["target_mean"][0]),
            float(aux_normalization["target_std"][0]),
        )
        measured = data["y_measured"][run_id, WINDOW_LENGTH:]
        times = data["time"][WINDOW_LENGTH:]
        agreement.extend(np.abs(y_main - y_aux))
        residual = np.abs(measured - y_aux)
        aux_residual.extend(residual)
        aux_predictions.extend(y_aux)
        ewma_values.extend(
            post_blanking_ewma(
                times,
                residual,
                ARBITRATION_EWMA_ALPHA,
                STARTUP_BLANKING_TIME,
            )
        )

    agreement_array = np.asarray(agreement, dtype=float)
    aux_residual_array = np.asarray(aux_residual, dtype=float)
    ewma_array = np.asarray(ewma_values, dtype=float)
    aux_prediction_array = np.asarray(aux_predictions, dtype=float)
    thresholds = {
        "agreement_threshold": float(np.percentile(agreement_array, AGREEMENT_PERCENTILE)),
        "param_mismatch_threshold": float(np.percentile(ewma_array, MISMATCH_PERCENTILE)),
        "param_mismatch_recovery_threshold": float(
            np.percentile(ewma_array, RECOVERY_PERCENTILE)
        ),
        "aux_recovery_gate": float(np.percentile(aux_residual_array, 99.9)),
    }
    ensure_finite(f"V3 calibration seed {seed}", list(thresholds.values()))
    if thresholds["param_mismatch_recovery_threshold"] >= thresholds["param_mismatch_threshold"]:
        raise AssertionError(f"V3 seed {seed} recovery threshold must be below latch threshold")
    if thresholds["agreement_threshold"] <= 0 or thresholds["aux_recovery_gate"] <= 0:
        raise AssertionError(f"V3 seed {seed} calibration produced non-positive threshold")

    return {
        "study": STUDY,
        "pair_id": f"P{seed}",
        "training_seed": seed,
        "metric_definition": {
            "agreement": "absolute difference between main and auxiliary one-step speed predictions",
            "parameter_mismatch": f"EWMA(alpha={ARBITRATION_EWMA_ALPHA}) of absolute measured-minus-auxiliary residual while the physical sensor is trusted",
            "aux_recovery": "absolute measured-minus-auxiliary residual",
        },
        "calibration_population": "clean validation trajectories only; no closed-loop holdout scenarios",
        "no_fault_data_statement": NO_FAULT_CALIBRATION_STATEMENT,
        "dataset_sha256": sha256(DATASET),
        "dataset_seed": int(data["seed"]),
        "model_sha256": model_hashes,
        "validation_run_ids": validation_ids,
        "sample_counts": {
            "agreement": int(len(agreement_array)),
            "aux_recovery": int(len(aux_residual_array)),
            "ewma_after_startup_blanking": int(len(ewma_array)),
        },
        "statistics": {
            "agreement_p90": float(np.percentile(agreement_array, 90.0)),
            "agreement_p95": float(np.percentile(agreement_array, 95.0)),
            "agreement_p99": float(np.percentile(agreement_array, 99.0)),
            "ewma_p95": float(np.percentile(ewma_array, 95.0)),
            "ewma_p99_9": float(np.percentile(ewma_array, 99.9)),
            "aux_residual_p99_9": float(np.percentile(aux_residual_array, 99.9)),
            "aux_prediction_min": float(aux_prediction_array.min()),
            "aux_prediction_max": float(aux_prediction_array.max()),
        },
        "clean_validation_false_alarm_statistics": {
            "agreement_exceedance_rate_at_p90": float(
                np.mean(agreement_array > thresholds["agreement_threshold"])
            ),
            "mismatch_ewma_exceedance_rate_at_p99_9": float(
                np.mean(ewma_array > thresholds["param_mismatch_threshold"])
            ),
            "aux_recovery_exceedance_rate_at_p99_9": float(
                np.mean(aux_residual_array > thresholds["aux_recovery_gate"])
            ),
        },
        "threshold_derivation": {
            "agreement_threshold": f"p{AGREEMENT_PERCENTILE:g} of clean-validation agreement",
            "param_mismatch_threshold": f"p{MISMATCH_PERCENTILE:g} of clean-validation EWMA",
            "param_mismatch_recovery_threshold": f"p{RECOVERY_PERCENTILE:g} of clean-validation EWMA (lower hysteresis threshold)",
            "aux_recovery_gate": "p99.9 of clean-validation auxiliary residual",
            "aux_speed_bounds": "fixed application/simulation operating envelope in rad/s, not a fitted percentile",
            "param_mismatch_recovery_count": "50 samples = 0.5 s at the fixed 0.01 s simulation step",
        },
        "algorithm": {
            "source_path": rel(SOURCE_PATHS["v3_calibration_script"]),
            "source_sha256": sha256(SOURCE_PATHS["v3_calibration_script"]),
        },
        "computed_values": {
            **thresholds,
            "ewma_alpha": ARBITRATION_EWMA_ALPHA,
            "startup_blanking_time": STARTUP_BLANKING_TIME,
            "param_mismatch_recovery_count": RECOVERY_COUNT,
            "aux_speed_bounds": AUX_SPEED_BOUNDS,
        },
    }


def assert_2026_reproduction(
    sensor_calibration: dict[str, Any],
    v3_calibration: dict[str, Any],
) -> None:
    canonical_sensor = load_json(CANONICAL_SENSOR_CONFIG)
    canonical_sensor_values = canonical_sensor["sensor"]
    reproduced_sensor = sensor_calibration["computed_values"]
    for key in ("instant_threshold", "center", "allowance", "threshold", "signed_ewma_threshold"):
        assert_close(
            f"sensor.{key}",
            float(reproduced_sensor[key]),
            float(canonical_sensor_values[key]),
        )
    assert_close(
        "sensor.cusum_percentile",
        float(reproduced_sensor["cusum_percentile"]),
        float(canonical_sensor_values["cusum_percentile"]),
    )
    assert_close(
        "sensor.validation_false_alarm_rate",
        float(sensor_calibration["clean_validation_false_alarm_statistics"]["selected_debounced_sensor_alarm_rate"]),
        float(canonical_sensor["validation_sensor_false_alarm_rate"]),
    )

    canonical_v3 = load_json(CANONICAL_V3_CALIBRATION)
    if v3_calibration["validation_run_ids"] != canonical_v3["validation_run_ids"]:
        raise AssertionError("2026 V3 validation trajectory IDs do not reproduce canonical IDs")
    if v3_calibration["sample_counts"] != canonical_v3["sample_counts"]:
        raise AssertionError("2026 V3 validation sample counts do not reproduce canonical counts")
    for key in (
        "agreement_threshold",
        "param_mismatch_threshold",
        "param_mismatch_recovery_threshold",
        "aux_recovery_gate",
        "ewma_alpha",
        "startup_blanking_time",
    ):
        assert_close(
            f"v3.{key}",
            float(v3_calibration["computed_values"][key]),
            float(canonical_v3["computed_values"][key]),
        )
    if (
        int(v3_calibration["computed_values"]["param_mismatch_recovery_count"])
        != int(canonical_v3["computed_values"]["param_mismatch_recovery_count"])
    ):
        raise AssertionError("2026 V3 recovery count does not reproduce canonical value")
    if v3_calibration["computed_values"]["aux_speed_bounds"] != canonical_v3["computed_values"]["aux_speed_bounds"]:
        raise AssertionError("2026 V3 auxiliary speed bounds do not reproduce canonical value")


def pair_config_payload(
    seed: int,
    paths: dict[str, Path],
    sensor_calibration: dict[str, Any],
    v3_calibration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "study": STUDY,
        "status": "calibrated_before_training_seed_closed_loop",
        "pair_id": f"P{seed}",
        "training_seed": seed,
        "main_model": {
            "weights_path": rel(paths["main_weights"]),
            "config_path": rel(paths["main_config"]),
            "weights_sha256": sha256(paths["main_weights"]),
            "config_sha256": sha256(paths["main_config"]),
        },
        "auxiliary_model": {
            "weights_path": rel(paths["aux_weights"]),
            "config_path": rel(paths["aux_config"]),
            "weights_sha256": sha256(paths["aux_weights"]),
            "config_sha256": sha256(paths["aux_config"]),
        },
        "sensor_monitor": sensor_calibration["computed_values"],
        "arbitrator": v3_calibration["computed_values"],
        "sensor_calibration_file": rel(paths["sensor_calibration"]),
        "sensor_calibration_sha256": sha256(paths["sensor_calibration"]),
        "v3_calibration_file": rel(paths["v3_calibration"]),
        "v3_calibration_sha256": sha256(paths["v3_calibration"]),
        "validation_run_ids": EXPECTED_SPLITS["validation"],
        "no_fault_data_statement": NO_FAULT_CALIBRATION_STATEMENT,
    }


def calibrate_and_save_pair(
    seed: int,
    data: np.lib.npyio.NpzFile,
    paths: dict[str, Path],
    *,
    assert_canonical_2026: bool,
) -> dict[str, Any]:
    main_model, main_config = load_lstm_model(paths["main_weights"], paths["main_config"])
    aux_model, aux_config = load_auxiliary_model(paths["aux_weights"], paths["aux_config"])
    model_hashes = {
        "main": sha256(paths["main_weights"]),
        "main_config": sha256(paths["main_config"]),
        "auxiliary": sha256(paths["aux_weights"]),
        "auxiliary_config": sha256(paths["aux_config"]),
    }
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(4, previous_threads))
        sensor_calibration = calibrate_sensor_monitor(
            seed,
            main_model,
            main_config,
            data,
            model_hashes["main"],
            model_hashes["main_config"],
        )
        torch.set_num_threads(1)
        v3_calibration = calibrate_v3_pair(
            seed,
            main_model,
            main_config,
            aux_model,
            aux_config,
            data,
            model_hashes,
        )
    finally:
        torch.set_num_threads(previous_threads)
    if assert_canonical_2026:
        assert_2026_reproduction(sensor_calibration, v3_calibration)

    write_json(paths["sensor_calibration"], sensor_calibration)
    write_json(paths["v3_calibration"], v3_calibration)
    pair_config = pair_config_payload(seed, paths, sensor_calibration, v3_calibration)
    write_json(paths["pair_config"], pair_config)
    return {
        "pair_id": f"P{seed}",
        "training_seed": seed,
        "validation_run_ids": EXPECTED_SPLITS["validation"],
        "calibration_sample_counts": {
            "sensor_residual": int(sensor_calibration["sample_count"]),
            "v3": v3_calibration["sample_counts"],
        },
        "main": {
            "weights_path": rel(paths["main_weights"]),
            "config_path": rel(paths["main_config"]),
            "weights_sha256": model_hashes["main"],
            "config_sha256": model_hashes["main_config"],
        },
        "auxiliary": {
            "weights_path": rel(paths["aux_weights"]),
            "config_path": rel(paths["aux_config"]),
            "weights_sha256": model_hashes["auxiliary"],
            "config_sha256": model_hashes["auxiliary_config"],
        },
        "sensor_calibration": {
            "path": rel(paths["sensor_calibration"]),
            "sha256": sha256(paths["sensor_calibration"]),
        },
        "v3_calibration": {
            "path": rel(paths["v3_calibration"]),
            "sha256": sha256(paths["v3_calibration"]),
        },
        "pair_config": {
            "path": rel(paths["pair_config"]),
            "sha256": sha256(paths["pair_config"]),
        },
    }


def main() -> None:
    data, source_hashes = preflight_frozen_inputs()
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    protocol = protocol_payload(data, source_hashes)
    write_json(PROTOCOL_MANIFEST, protocol)

    # First prove that the frozen calibration implementation reproduces P2026.
    pair_records: dict[str, dict[str, Any]] = {}
    paths_2026 = artifact_paths(2026)
    pair_records["P2026"] = calibrate_and_save_pair(
        2026,
        data,
        paths_2026,
        assert_canonical_2026=True,
    )

    # Train exactly the two preregistered additional pairs, then calibrate each
    # independently on the same frozen clean validation trajectories.
    for seed in TRAINING_SEEDS:
        paths = artifact_paths(seed)
        train_main_model(seed, data, paths)
        train_aux_model(seed, data, paths)
        pair_records[f"P{seed}"] = calibrate_and_save_pair(
            seed,
            data,
            paths,
            assert_canonical_2026=False,
        )

    runner_hash = sha256(Path(__file__).resolve())
    manifest = {
        "study": STUDY,
        "protocol_manifest": {
            "path": rel(PROTOCOL_MANIFEST),
            "sha256": sha256(PROTOCOL_MANIFEST),
        },
        "validation_run_ids": EXPECTED_SPLITS["validation"],
        "algorithm_hashes": {
            "sensor_calibration_source": {
                "path": rel(SOURCE_PATHS["sensor_calibration_notebook"]),
                "sha256": source_hashes["sensor_calibration_notebook"],
            },
            "v3_calibration_source": {
                "path": rel(SOURCE_PATHS["v3_calibration_script"]),
                "sha256": source_hashes["v3_calibration_script"],
            },
            "runner": {
                "path": rel(Path(__file__).resolve()),
                "sha256": runner_hash,
            },
        },
        "pairs": pair_records,
    }
    write_json(MODEL_PAIR_MANIFEST, manifest)

    print(
        json.dumps(
            {
                "status": "MODEL_TRAINING_AND_CALIBRATION_COMPLETE",
                "model_pair_manifest": rel(MODEL_PAIR_MANIFEST),
                "model_pair_manifest_sha256": sha256(MODEL_PAIR_MANIFEST),
                "closed_loop_runs_executed": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
