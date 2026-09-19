"""V4 calibration: sensor/CUSUM/witness thresholds with bootstrap uncertainty.

Mirrors the frozen sensor-calibration rules on the larger V4 validation
pool (24 trajectories by default instead of 6):
- instant residual gate: p99.9 of |y_meas - main_pred|.
- CUSUM center/allowance from clean residuals; threshold from the frozen
  percentile grid [90..99.9], first candidate with debounced validation
  false-alarm rate <= 0.001, else the last grid candidate.
- aux recovery gate: p99.9 of |y_meas - aux_pred|.
- witness gates (new): p99.9 of |y_meas - witness| for the aux-LSTM and
  frozen-EKF witnesses (entry AND-rule).

Threshold uncertainty comes from a trajectory bootstrap (resample
validation trajectories with replacement): mean/std/95% interval per
threshold. The debounce-selected CUSUM percentile is fixed from the full
validation pool; replicates recompute the threshold at that percentile.

Usage:
    python scripts/v4_calibrate.py --seeds 2029
    python scripts/v4_calibrate.py --smoke   # 1 seed, 20 replicates
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from auxiliary_sensor_model import load_auxiliary_model
from ekf_observer import AugmentedStateEKF, EKFNumericalError
from reliability import (
    SensorReliabilityMonitor,
    cusum_monitor,
    load_lstm_model,
)

DEFAULT_DATASET = PROJECT / "data" / "processed" / "v4" / "dc_motor_v4_dataset.npz"
DEFAULT_MODELS = PROJECT / "results" / "v4" / "models"
DEFAULT_OUT = PROJECT / "results" / "v4" / "calibration"

WINDOW_LENGTH = 20
CUSUM_GRID = (90.0, 95.0, 97.5, 99.0, 99.5, 99.9)
TARGET_FALSE_ALARM = 0.001
ENTER_COUNT = 3
EXIT_COUNT = 5


def fail(message: str) -> None:
    raise SystemExit(f"V4_CALIBRATE FAILED: {message}")


def frozen_ekf():
    """Build the frozen EKF from its locked calibration (read-only reuse)."""
    import evaluate_ekf_closed_loop as frozen_ekf_eval
    _, config = frozen_ekf_eval.load_frozen_ekf_config()
    return AugmentedStateEKF(config)


def validation_residuals(data, main_model, main_norm, aux_model, aux_norm,
                         val_ids: np.ndarray) -> dict:
    """One-step main/aux residuals and EKF witness residuals per trajectory."""
    main_residuals, aux_residuals, ekf_residuals = [], [], []
    ekf_failures = 0
    for rid in (int(v) for v in val_ids):
        voltage = np.asarray(data["voltage"][rid], dtype=np.float32)
        measured = np.asarray(data["y_measured"][rid], dtype=np.float32)
        truth = np.asarray(data["y_true"][rid], dtype=np.float32)
        current = np.asarray(data["current_measured"][rid], dtype=np.float32)
        count = len(voltage) - WINDOW_LENGTH
        # Main: [V, y_meas] windows -> residual vs measured-next.
        features = np.stack((voltage, measured), axis=1)
        windows = np.stack(
            [features[s:s + WINDOW_LENGTH] for s in range(count)]).astype(
                np.float32)
        windows_n = ((windows - main_norm["input_mean"])
                     / main_norm["input_std"]).astype(np.float32)
        with torch.no_grad():
            pred_n = main_model(torch.from_numpy(windows_n)).numpy()[:, 0]
        pred = (pred_n * main_norm["target_std"][0]
                + main_norm["target_mean"][0])
        main_residuals.append(measured[WINDOW_LENGTH:] - pred)
        # Aux: [V, current] windows -> residual vs measured-next.
        aux_features = np.stack((voltage, current), axis=1)
        aux_windows = np.stack(
            [aux_features[s:s + WINDOW_LENGTH]
             for s in range(count)]).astype(np.float32)
        aux_windows_n = ((aux_windows - aux_norm["input_mean"])
                         / aux_norm["input_std"]).astype(np.float32)
        with torch.no_grad():
            aux_pred_n = aux_model(torch.from_numpy(aux_windows_n)).numpy()[:, 0]
        aux_pred = (aux_pred_n * aux_norm["target_std"][0]
                    + aux_norm["target_mean"][0])
        aux_residuals.append(measured[WINDOW_LENGTH:] - aux_pred)
        # EKF witness rollout on voltage + measured current, mirroring the
        # frozen advance_ekf alignment: sample 0 initializes, sample s > 0
        # steps with the voltage applied during the preceding interval and
        # the current measured at the present sample.
        try:
            ekf = frozen_ekf()
            ekf.initialize(float(current[0]))
            estimates = np.zeros(len(voltage))
            for s in range(1, len(voltage)):
                state, _ = ekf.step(float(voltage[s - 1]), float(current[s]))
                estimates[s] = state[1]
            ekf_residuals.append(
                measured[WINDOW_LENGTH:] - estimates[WINDOW_LENGTH:])
        except EKFNumericalError:
            ekf_failures += 1
    return {"main": main_residuals, "aux": aux_residuals, "ekf": ekf_residuals,
            "ekf_failures": ekf_failures, "n_trajectories": len(val_ids)}


def debounced_alarm_rate(sequences: list[np.ndarray], instant_gate: float,
                         center: float, allowance: float,
                         threshold: float) -> float:
    """Fraction of validation samples with the debounced sensor alarm active."""
    flagged = total = 0
    for sequence in sequences:
        monitor = SensorReliabilityMonitor(
            residual_gate=instant_gate, center=center, allowance=allowance,
            threshold=threshold, enter_count=ENTER_COUNT,
            exit_count=EXIT_COUNT)
        for value in sequence:
            flagged += monitor.update(float(value))["sensor_suspect"]
            total += 1
    return flagged / total if total else float("nan")


def point_thresholds(residuals: dict) -> dict:
    main = residuals["main"]
    flat_main = np.concatenate(main)
    center = float(np.median(flat_main))
    sigma = float(np.std(flat_main, ddof=1))
    allowance = 0.5 * sigma
    instant_gate = float(np.percentile(np.abs(flat_main), 99.9))
    grid = []
    for percentile in CUSUM_GRID:
        scores = np.concatenate(
            [cusum_monitor(seq, center, allowance, np.inf)["score"]
             for seq in main])
        threshold = float(np.percentile(scores, percentile))
        rate = debounced_alarm_rate(main, instant_gate, center, allowance,
                                    threshold)
        grid.append({"percentile": percentile, "cusum_threshold": threshold,
                     "validation_false_alarm_rate": rate})
    selected = next((entry for entry in grid
                     if entry["validation_false_alarm_rate"]
                     <= TARGET_FALSE_ALARM), grid[-1])
    flat_aux = np.concatenate(residuals["aux"])
    aux_gate = float(np.percentile(np.abs(flat_aux), 99.9))
    out = {"center": center, "allowance": allowance,
           "instant_threshold": instant_gate,
           "threshold": selected["cusum_threshold"],
           "cusum_percentile": selected["percentile"],
           "calibration_grid": grid, "aux_recovery_gate": aux_gate,
           "witness_aux_gate": aux_gate}
    if residuals["ekf"]:
        flat_ekf = np.concatenate(residuals["ekf"])
        out["witness_ekf_gate"] = float(np.percentile(np.abs(flat_ekf), 99.9))
    else:
        out["witness_ekf_gate"] = float("nan")
    return out


def bootstrap_thresholds(residuals: dict, percentile: float,
                         n_replicates: int, seed: int) -> dict:
    """Trajectory bootstrap of every threshold statistic."""
    rng = np.random.default_rng(seed)
    n = len(residuals["main"])
    keys = ("center", "allowance", "instant_threshold", "threshold",
            "aux_recovery_gate", "witness_aux_gate", "witness_ekf_gate")
    samples: dict[str, list[float]] = {key: [] for key in keys}
    has_ekf = len(residuals["ekf"]) == n
    for _ in range(n_replicates):
        picks = rng.integers(0, n, n)
        main = [residuals["main"][i] for i in picks]
        aux = [residuals["aux"][i] for i in picks]
        flat_main = np.concatenate(main)
        center = float(np.median(flat_main))
        allowance = 0.5 * float(np.std(flat_main, ddof=1))
        scores = np.concatenate(
            [cusum_monitor(seq, center, allowance, np.inf)["score"]
             for seq in main])
        flat_aux = np.concatenate(aux)
        aux_gate = float(np.percentile(np.abs(flat_aux), 99.9))
        samples["center"].append(center)
        samples["allowance"].append(allowance)
        samples["instant_threshold"].append(
            float(np.percentile(np.abs(flat_main), 99.9)))
        samples["threshold"].append(float(np.percentile(scores, percentile)))
        samples["aux_recovery_gate"].append(aux_gate)
        samples["witness_aux_gate"].append(aux_gate)
        if has_ekf:
            ekf = [residuals["ekf"][i] for i in picks]
            samples["witness_ekf_gate"].append(
                float(np.percentile(np.abs(np.concatenate(ekf)), 99.9)))
        else:
            samples["witness_ekf_gate"].append(float("nan"))
    summary = {}
    for key, values in samples.items():
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if len(finite) == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan"),
                            "ci_lo": float("nan"), "ci_hi": float("nan"),
                            "replicates": n_replicates}
        else:
            summary[key] = {
                "mean": float(np.mean(finite)), "std": float(np.std(finite)),
                "ci_lo": float(np.percentile(finite, 2.5)),
                "ci_hi": float(np.percentile(finite, 97.5)),
                "replicates": n_replicates}
    return summary


def calibrate_pair(seed: int, data, models_dir: Path, out_dir: Path,
                   n_bootstrap: int, bootstrap_seed: int) -> dict:
    main_model, main_config = load_lstm_model(
        models_dir / f"main_seed_{seed}.pt",
        models_dir / f"main_seed_{seed}_config.json", device="cpu")
    aux_model, aux_config = load_auxiliary_model(
        models_dir / f"aux_seed_{seed}.pt",
        models_dir / f"aux_seed_{seed}_config.json", device="cpu")
    main_norm = {key: np.asarray(value)
                 for key, value in main_config["normalization"].items()}
    aux_norm = {key: np.asarray(value)
                for key, value in aux_config["normalization"].items()}
    val_ids = np.asarray(data["split_run_ids_validation"])
    if len(val_ids) == 0:
        fail(f"seed {seed}: empty validation split")
    residuals = validation_residuals(data, main_model, main_norm, aux_model,
                                     aux_norm, val_ids)
    if residuals["ekf_failures"] > len(val_ids) // 2:
        fail(f"seed {seed}: EKF failed on "
             f"{residuals['ekf_failures']}/{len(val_ids)} validation runs")
    point = point_thresholds(residuals)
    boot = bootstrap_thresholds(residuals, point["cusum_percentile"],
                                n_bootstrap, bootstrap_seed + seed)
    doc = {
        "study": "v4_calibration",
        "training_seed": seed,
        "pair_id": f"P{seed}",
        "calibration_population": "V4 clean validation trajectories only",
        "validation_run_ids": [int(v) for v in val_ids],
        "sample_count": int(sum(len(seq) for seq in residuals["main"])),
        "ekf_validation_failures": residuals["ekf_failures"],
        "algorithm": {
            "instant_percentile": 99.9,
            "cusum_percentile_grid": list(CUSUM_GRID),
            "allowance_sigma": 0.5,
            "target_validation_false_alarm_rate": TARGET_FALSE_ALARM,
            "enter_count": ENTER_COUNT,
            "exit_count": EXIT_COUNT,
            "witness_percentile": 99.9,
            "bootstrap_replicates": n_bootstrap,
            "bootstrap_seed": bootstrap_seed + seed,
            "bootstrap_unit": "whole validation trajectory",
        },
        "computed_values": {
            "center": point["center"], "allowance": point["allowance"],
            "threshold": point["threshold"],
            "cusum_percentile": point["cusum_percentile"],
            "instant_threshold": point["instant_threshold"],
            "aux_recovery_gate": point["aux_recovery_gate"],
            "witness_aux_gate": point["witness_aux_gate"],
            "witness_ekf_gate": point["witness_ekf_gate"],
            "enter_count": ENTER_COUNT, "exit_count": EXIT_COUNT,
        },
        "calibration_grid": point["calibration_grid"],
        "bootstrap_uncertainty": boot,
        "no_fault_data_statement": (
            "Calibration uses only clean, fault-free V4 validation "
            "trajectories; no test trajectories, fault scenarios, "
            "closed-loop results, or simulation-seed results are used."),
    }
    (out_dir / f"sensor_seed{seed}_v4_calibration.json").write_text(
        json.dumps(doc, indent=2))
    print(f"  seed {seed}: gate={point['instant_threshold']:.4f} "
          f"thr={point['threshold']:.4f} (p{point['cusum_percentile']}) "
          f"witness_aux={point['witness_aux_gate']:.4f} "
          f"witness_ekf={point['witness_ekf_gate']:.4f}")
    return doc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260920)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run: seed 2029, 20 replicates")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.smoke:
        args.seeds = [2029]
        args.n_bootstrap = 20
    from v4_train_models import ALL_SEEDS
    seeds = ALL_SEEDS if args.seeds is None else args.seeds
    if not seeds:
        fail("no seeds selected")
    if not args.dataset.is_file():
        fail(f"dataset not found: {args.dataset}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    with np.load(args.dataset, allow_pickle=True) as data:
        for seed in seeds:
            print(f"calibrating seed {seed} ...")
            doc = calibrate_pair(seed, data, args.models_dir, args.out_dir,
                                 args.n_bootstrap, args.bootstrap_seed)
            summary[str(seed)] = doc["computed_values"]
    (args.out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"calibrated {len(seeds)} pair(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
