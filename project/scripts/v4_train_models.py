"""V4 model training: main + auxiliary LSTMs across seeds 2026-2036.

Mirrors the frozen training procedures (architectures, features, losses,
optimizer, RNG contracts) with raised maximum epochs and patience-based
early stopping. Every run logs best_epoch, the stopping reason, and a
convergence flag (best_epoch < max_epochs addresses the frozen finding
that seed 2026 hit best_epoch == max_epochs == 60).

Default RNG semantics (identical to the frozen protocols):
- main: np.random.seed(s), torch.manual_seed(s), deterministic CPU
  algorithms, dedicated torch.Generator().manual_seed(s) for shuffling.
- aux: np/torch global seeds, shuffle=True with no explicit generator.

Usage:
    python scripts/v4_train_models.py --seeds 2029 2030
    python scripts/v4_train_models.py --smoke   # 1 seed, 2 epochs, CPU
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from auxiliary_sensor_model import (
    AuxiliarySpeedEstimator,
    build_auxiliary_sequences,
    fit_auxiliary_normalization,
    normalize_auxiliary,
)
from data_utils import fit_normalization, normalize_sequences
from lstm_model import LSTMForecaster, recursive_forecast

DEFAULT_DATASET = PROJECT / "data" / "processed" / "v4" / "dc_motor_v4_dataset.npz"
DEFAULT_OUT = PROJECT / "results" / "v4" / "models"

ALL_SEEDS = list(range(2026, 2037))  # 2026-2028 frozen + 2029-2036 new

MAIN_ARCH = {"input_size": 2, "hidden_size": 64, "num_layers": 2,
             "dropout": 0.2, "output_size": 1, "residual": True}
AUX_ARCH = {"input_size": 2, "hidden_size": 32, "num_layers": 1,
            "dropout": 0.0, "output_size": 1}

WINDOW_LENGTH = 20
BATCH_SIZE = 512
LEARNING_RATE = 0.001
MIN_DELTA = 1e-6
# Raised ceilings (frozen: main 60 / aux 100); patience keeps the
# frozen values so stopping remains patience-driven, not epoch-capped.
MAX_EPOCHS_MAIN = 200
PATIENCE_MAIN = 8
MAX_EPOCHS_AUX = 200
PATIENCE_AUX = 10


def fail(message: str) -> None:
    raise SystemExit(f"V4_TRAIN FAILED: {message}")


def train_loop(model: nn.Module, train_loader: DataLoader,
               val_loader: DataLoader, max_epochs: int,
               patience: int) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    stopped_epoch = max_epochs
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                val_losses.append(criterion(model(x_batch), y_batch).item())
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.clone()
                          for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                stopped_epoch = epoch
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_val_loss": best_val,
            "stopped_epoch": stopped_epoch, "history": history,
            "stop_reason": ("patience" if stopped_epoch < max_epochs
                            else "max_epochs"),
            "converged": best_epoch < max_epochs}


def regression_metrics(true: np.ndarray, predicted: np.ndarray) -> dict:
    errors = predicted - true
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {"rmse": rmse, "mae": mae,
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


@torch.no_grad()
def recursive_endpoint_rmse(model: LSTMForecaster, raw: dict,
                            test_ids: np.ndarray, normalization: dict,
                            horizons=(5, 10, 15)) -> dict:
    """Recursive multi-step RMSE on raw test trajectories (frozen metric)."""
    input_mean = torch.tensor(np.asarray(normalization["input_mean"]),
                              dtype=torch.float32)
    input_std = torch.tensor(np.asarray(normalization["input_std"]),
                             dtype=torch.float32)
    target_mean = torch.tensor(np.asarray(normalization["target_mean"]),
                               dtype=torch.float32)
    target_std = torch.tensor(np.asarray(normalization["target_std"]),
                              dtype=torch.float32)
    model.eval()
    out = {}
    for horizon in horizons:
        squared, count = 0.0, 0
        for rid in test_ids:
            voltage = raw["voltage"][int(rid)]
            measured = raw["y_measured"][int(rid)]
            truth = raw["y_true"][int(rid)]
            for start in range(0, len(voltage) - WINDOW_LENGTH - horizon,
                               horizon):
                window = np.stack(
                    (voltage[start:start + WINDOW_LENGTH],
                     measured[start:start + WINDOW_LENGTH]), axis=1)
                window_n = ((window - normalization["input_mean"])
                            / normalization["input_std"]).astype(np.float32)
                future_v = voltage[start + WINDOW_LENGTH:
                                   start + WINDOW_LENGTH + horizon]
                future_vn = ((future_v - normalization["input_mean"][0])
                             / normalization["input_std"][0]).astype(np.float32)
                forecast = recursive_forecast(
                    model, torch.from_numpy(window_n[None]),
                    torch.from_numpy(future_vn[None, :]),
                    input_mean, input_std, target_mean, target_std)
                physical = (forecast[0].numpy() * normalization["target_std"][0]
                            + normalization["target_mean"][0])
                endpoint = truth[start + WINDOW_LENGTH + horizon - 1]
                squared += float((physical[-1] - endpoint) ** 2)
                count += 1
        out[f"h{horizon}_rmse"] = float(np.sqrt(squared / count)) if count else float("nan")
    return out


def train_main(seed: int, data: np.lib.npyio.NpzFile, out_dir: Path,
               max_epochs: int, patience: int) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        normalization = {
            "input_mean": np.asarray(data["normalization_input_mean"]),
            "input_std": np.asarray(data["normalization_input_std"]),
            "target_mean": np.asarray(data["normalization_target_mean"]),
            "target_std": np.asarray(data["normalization_target_std"]),
        }
        x_train, y_train = normalize_sequences(
            np.asarray(data["X_train"], dtype=np.float32),
            np.asarray(data["y_train"], dtype=np.float32), normalization)
        x_val, y_val = normalize_sequences(
            np.asarray(data["X_validation"], dtype=np.float32),
            np.asarray(data["y_validation"], dtype=np.float32), normalization)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train),
                          torch.from_numpy(y_train)),
            batch_size=BATCH_SIZE, shuffle=True, generator=generator)
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
            batch_size=BATCH_SIZE)
        model = LSTMForecaster(**MAIN_ARCH)
        run = train_loop(model, train_loader, val_loader, max_epochs,
                         patience)
        # Test metrics (physical units, one-step).
        x_test = np.asarray(data["X_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=np.float32)
        x_test_n = ((x_test - normalization["input_mean"])
                    / normalization["input_std"]).astype(np.float32)
        with torch.no_grad():
            pred_n = model(torch.from_numpy(x_test_n)).numpy()
        pred = (pred_n * normalization["target_std"][0]
                + normalization["target_mean"][0])
        metrics = regression_metrics(y_test[:, 0], pred[:, 0])
        raw = {"voltage": np.asarray(data["voltage"]),
               "y_measured": np.asarray(data["y_measured"]),
               "y_true": np.asarray(data["y_true"])}
        metrics.update(recursive_endpoint_rmse(
            model, raw, np.asarray(data["split_run_ids_test"]),
            normalization))
        weights_path = out_dir / f"main_seed_{seed}.pt"
        torch.save(model.state_dict(), weights_path)
        serializable_norm = {key: np.asarray(value).tolist()
                             for key, value in normalization.items()}
        config = {"model": MAIN_ARCH, "normalization": serializable_norm,
                  "features": ["voltage", "y_measured"], "target": "y_true",
                  "window_length": WINDOW_LENGTH, "batch_size": BATCH_SIZE,
                  "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                  "training": {"seed": seed, "max_epochs": max_epochs,
                               "patience": patience, "min_delta": MIN_DELTA,
                               "best_epoch": run["best_epoch"],
                               "stopped_epoch": run["stopped_epoch"],
                               "stop_reason": run["stop_reason"],
                               "converged": run["converged"]}}
        (out_dir / f"main_seed_{seed}_config.json").write_text(
            json.dumps(config, indent=2))
        (out_dir / f"main_seed_{seed}_metrics.json").write_text(json.dumps(
            {"seed": seed, **metrics, "best_epoch": run["best_epoch"],
             "converged": run["converged"]}, indent=2))
        np.savez_compressed(
            out_dir / f"main_seed_{seed}_history.npz",
            train_loss=np.asarray(run["history"]["train_loss"]),
            val_loss=np.asarray(run["history"]["val_loss"]),
            best_epoch=np.asarray(run["best_epoch"]))
        print(f"  main seed {seed}: best_epoch={run['best_epoch']} "
              f"({run['stop_reason']}), test_rmse={metrics['rmse']:.6f}")
        return {"best_epoch": run["best_epoch"], **metrics}
    finally:
        torch.use_deterministic_algorithms(previous)


def train_aux(seed: int, data: np.lib.npyio.NpzFile, out_dir: Path,
              max_epochs: int, patience: int) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    def build(ids: np.ndarray):
        parts_x, parts_y = [], []
        for rid in ids:
            x, y = build_auxiliary_sequences(
                np.asarray(data["voltage"][int(rid)]),
                np.asarray(data["current_measured"][int(rid)]),
                np.asarray(data["y_true"][int(rid)]),
                window_length=WINDOW_LENGTH)
            parts_x.append(x)
            parts_y.append(y)
        return np.concatenate(parts_x), np.concatenate(parts_y)

    train_ids = np.asarray(data["split_run_ids_train"])
    val_ids = np.asarray(data["split_run_ids_validation"])
    test_ids = np.asarray(data["split_run_ids_test"])
    x_train, y_train = build(train_ids)
    x_val, y_val = build(val_ids)
    normalization = fit_auxiliary_normalization(x_train, y_train)
    x_train_n, y_train_n = normalize_auxiliary(x_train, y_train,
                                               normalization)
    x_val_n, y_val_n = normalize_auxiliary(x_val, y_val, normalization)
    # Frozen aux semantics: shuffle=True with no explicit generator.
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_n),
                      torch.from_numpy(y_train_n)),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val_n), torch.from_numpy(y_val_n)),
        batch_size=BATCH_SIZE)
    model = AuxiliarySpeedEstimator(**AUX_ARCH)
    run = train_loop(model, train_loader, val_loader, max_epochs, patience)
    x_test, y_test = build(test_ids)
    x_test_n, _ = normalize_auxiliary(x_test, y_test, normalization)
    with torch.no_grad():
        pred_n = model(torch.from_numpy(x_test_n)).numpy()
    pred = (pred_n * normalization["target_std"][0]
            + normalization["target_mean"][0])
    metrics = regression_metrics(y_test[:, 0], pred[:, 0])
    weights_path = out_dir / f"aux_seed_{seed}.pt"
    torch.save(model.state_dict(), weights_path)
    serializable_norm = {key: np.asarray(value).tolist()
                         for key, value in normalization.items()}
    config = {"model": AUX_ARCH, "normalization": serializable_norm,
              "features": ["voltage", "current_measured"], "target": "y_true",
              "window_length": WINDOW_LENGTH, "batch_size": BATCH_SIZE,
              "optimizer": "Adam", "learning_rate": LEARNING_RATE,
              "training": {"seed": seed, "max_epochs": max_epochs,
                           "patience": patience, "min_delta": MIN_DELTA,
                           "best_epoch": run["best_epoch"],
                           "stopped_epoch": run["stopped_epoch"],
                           "stop_reason": run["stop_reason"],
                           "converged": run["converged"]}}
    (out_dir / f"aux_seed_{seed}_config.json").write_text(
        json.dumps(config, indent=2))
    (out_dir / f"aux_seed_{seed}_metrics.json").write_text(json.dumps(
        {"seed": seed, **metrics, "best_epoch": run["best_epoch"],
         "converged": run["converged"]}, indent=2))
    np.savez_compressed(
        out_dir / f"aux_seed_{seed}_history.npz",
        train_loss=np.asarray(run["history"]["train_loss"]),
        val_loss=np.asarray(run["history"]["val_loss"]),
        best_epoch=np.asarray(run["best_epoch"]))
    print(f"  aux seed {seed}: best_epoch={run['best_epoch']} "
          f"({run['stop_reason']}), test_rmse={metrics['rmse']:.6f}")
    return {"best_epoch": run["best_epoch"], **metrics}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-epochs-main", type=int, default=MAX_EPOCHS_MAIN)
    parser.add_argument("--patience-main", type=int, default=PATIENCE_MAIN)
    parser.add_argument("--max-epochs-aux", type=int, default=MAX_EPOCHS_AUX)
    parser.add_argument("--patience-aux", type=int, default=PATIENCE_AUX)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run: seed 2029, 2 epochs each")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--aux-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    seeds = ALL_SEEDS if args.seeds is None else args.seeds
    if args.smoke:
        seeds = [2029]
        args.max_epochs_main = args.max_epochs_aux = 2
    if not seeds:
        fail("no training seeds selected")
    if not args.dataset.is_file():
        fail(f"dataset not found: {args.dataset}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    with np.load(args.dataset, allow_pickle=True) as data:
        for seed in seeds:
            print(f"training seed {seed} ...")
            entry: dict = {}
            if not args.aux_only:
                entry["main"] = train_main(seed, data, args.out_dir,
                                           args.max_epochs_main,
                                           args.patience_main)
            if not args.main_only:
                entry["aux"] = train_aux(seed, data, args.out_dir,
                                         args.max_epochs_aux,
                                         args.patience_aux)
            summary[str(seed)] = entry
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"trained {len(seeds)} seed(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
