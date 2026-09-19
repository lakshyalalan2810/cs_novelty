"""V4 baseline calibration: E2/S1/S2/S3 detectors on validation data only.

- E2 (EKF CUSUM) and S1 (fixed gate): frozen-EKF speed residuals
  |y_meas - omega_hat| on V4 validation trajectories.
- S2 (plausibility): trailing-median gate + rate limit from clean
  measured-speed validation sequences.
- S3 (Luenberger CUSUM): Luenberger speed residuals on the same pool.

Baselines are training-seed-free (no learned weights), so this produces a
single calibration document with trajectory-bootstrap gate uncertainty:

    python scripts/v4_calibrate_baselines.py --smoke
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from baselines_v4 import (
    CusumCore,
    LuenbergerBaseline,
    calibrate_cusum_grid,
    calibrate_fixed_gate,
    calibrate_plausibility,
)
from ekf_observer import AugmentedStateEKF

DEFAULT_DATASET = PROJECT / "data" / "processed" / "v4" / "dc_motor_v4_dataset.npz"
DEFAULT_OUT = PROJECT / "results" / "v4" / "calibration" / "baselines_v4_calibration.json"


def fail(message: str) -> None:
    raise SystemExit(f"V4_BASELINE_CALIBRATE FAILED: {message}")


def frozen_ekf() -> AugmentedStateEKF:
    import evaluate_ekf_closed_loop as frozen_ekf_eval
    _, config = frozen_ekf_eval.load_frozen_ekf_config()
    return AugmentedStateEKF(config)


def ekf_residual_sequences(data, val_ids: np.ndarray) -> list[np.ndarray]:
    sequences = []
    for rid in (int(v) for v in val_ids):
        voltage = np.asarray(data["voltage"][rid], dtype=float)
        measured = np.asarray(data["y_measured"][rid], dtype=float)
        current = np.asarray(data["current_measured"][rid], dtype=float)
        ekf = frozen_ekf()
        ekf.initialize(float(current[0]))
        estimates = np.zeros(len(voltage))
        for s in range(1, len(voltage)):
            state, _ = ekf.step(float(voltage[s - 1]), float(current[s]))
            estimates[s] = state[1]
        sequences.append(measured[20:] - estimates[20:])
    return sequences


def luenberger_residual_sequences(data, val_ids: np.ndarray) -> list[np.ndarray]:
    # Permissive detector: residuals are unaffected since the detector never
    # feeds back into the observer.
    sequences = []
    for rid in (int(v) for v in val_ids):
        voltage = np.asarray(data["voltage"][rid], dtype=float)
        measured = np.asarray(data["y_measured"][rid], dtype=float)
        current = np.asarray(data["current_measured"][rid], dtype=float)
        observer = LuenbergerBaseline(
            CusumCore(1e9, 0.0, 1e9, 1e18))
        estimates = np.zeros(len(voltage))
        for s in range(len(voltage)):
            estimates[s] = observer.update(
                float(voltage[s]), float(current[s]),
                float(measured[s]))["estimate"]
        sequences.append(measured - estimates)
    return sequences


def bootstrap_gate(samples_fn, n_replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_replicates):
        values.append(samples_fn(rng))
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "replicates": n_replicates}
    return {"mean": float(np.mean(finite)), "std": float(np.std(finite)),
            "ci_lo": float(np.percentile(finite, 2.5)),
            "ci_hi": float(np.percentile(finite, 97.5)),
            "replicates": n_replicates}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260921)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.smoke:
        args.n_bootstrap = 20
    if not args.dataset.is_file():
        fail(f"dataset not found: {args.dataset}")
    with np.load(args.dataset, allow_pickle=True) as data:
        val_ids = np.asarray(data["split_run_ids_validation"])
        if len(val_ids) == 0:
            fail("empty validation split")
        print(f"collecting EKF residuals on {len(val_ids)} validation runs ...")
        ekf_sequences = ekf_residual_sequences(data, val_ids)
        print("collecting Luenberger residuals ...")
        luenberger_sequences = luenberger_residual_sequences(data, val_ids)
        measured_sequences = [np.asarray(data["y_measured"][int(v)])
                              for v in val_ids]

    e2 = calibrate_cusum_grid(ekf_sequences)
    s1_gate = calibrate_fixed_gate(ekf_sequences)
    s2 = calibrate_plausibility(measured_sequences)
    s3 = calibrate_cusum_grid(luenberger_sequences)

    def resample(pool: list[np.ndarray], rng: np.random.Generator):
        picks = rng.integers(0, len(pool), len(pool))
        return [pool[i] for i in picks]

    from baselines_v4 import _cusum_scores
    uncertainty = {
        "e2_threshold": bootstrap_gate(
            lambda rng: float(np.percentile(np.concatenate(
                [_cusum_scores(seq, e2["center"], e2["allowance"])
                 for seq in resample(ekf_sequences, rng)]),
                e2["cusum_percentile"])),
            args.n_bootstrap, args.bootstrap_seed),
        "s1_gate": bootstrap_gate(
            lambda rng: calibrate_fixed_gate(
                resample(ekf_sequences, rng)),
            args.n_bootstrap, args.bootstrap_seed + 1),
        "s2_median_gate": bootstrap_gate(
            lambda rng: calibrate_plausibility(
                resample(measured_sequences, rng))["median_gate"],
            args.n_bootstrap, args.bootstrap_seed + 2),
        "s2_rate_limit": bootstrap_gate(
            lambda rng: calibrate_plausibility(
                resample(measured_sequences, rng))["rate_limit"],
            args.n_bootstrap, args.bootstrap_seed + 3),
        "s3_threshold": bootstrap_gate(
            lambda rng: float(np.percentile(np.concatenate(
                [_cusum_scores(seq, s3["center"], s3["allowance"])
                 for seq in resample(luenberger_sequences, rng)]),
                s3["cusum_percentile"])),
            args.n_bootstrap, args.bootstrap_seed + 4),
    }
    doc = {
        "study": "v4_baseline_calibration",
        "calibration_population": "V4 clean validation trajectories only",
        "validation_run_ids": [int(v) for v in val_ids],
        "e2_ekf_cusum": e2,
        "s1_fixed_gate": {"gate": s1_gate, "enter_count": 3, "exit_count": 5},
        "s2_plausibility": s2,
        "s3_luenberger_cusum": s3,
        "bootstrap_uncertainty": uncertainty,
        "no_fault_data_statement": (
            "Baseline detectors use only clean, fault-free V4 validation "
            "trajectories (plus the frozen EKF calibration); no test "
            "trajectories, fault scenarios, or closed-loop results."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"e2 gate={e2['instant_threshold']:.4f} thr={e2['threshold']:.4f} "
          f"s1 gate={s1_gate:.4f} s2 gate={s2['median_gate']:.4f} "
          f"rate={s2['rate_limit']:.1f} s3 thr={s3['threshold']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
