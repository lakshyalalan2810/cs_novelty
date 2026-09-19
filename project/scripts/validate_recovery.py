"""Deterministic regression check for premature sensor-monitor recovery."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reliability import SensorReliabilityMonitor


_NO_ASSERTION_MESSAGE = object()


def require(condition: object, message: object = _NO_ASSERTION_MESSAGE) -> None:
    """Raise AssertionError on failed verification even under optimized Python."""
    if condition:
        return
    if message is _NO_ASSERTION_MESSAGE:
        raise AssertionError
    raise AssertionError(message)


def legacy_states(residuals: np.ndarray, config: dict) -> np.ndarray:
    """Reproduce the pre-fix exit rule: instantaneous residual only."""
    monitor = SensorReliabilityMonitor(
        config["instant_threshold"], config["center"], config["allowance"],
        config["threshold"], config["enter_count"], config["exit_count"],
    )
    states = []
    for residual in residuals:
        finite = bool(np.isfinite(residual))
        instantaneous = not finite or abs(residual) > monitor.residual_gate
        if finite:
            centered = residual - monitor.center
            monitor.positive = max(0.0, monitor.positive + centered - monitor.allowance)
            monitor.negative = max(0.0, monitor.negative - centered - monitor.allowance)
        score = max(monitor.positive, monitor.negative) if finite else np.inf
        abnormal = instantaneous or score > monitor.threshold
        if not monitor.active:
            monitor.abnormal_run = monitor.abnormal_run + 1 if abnormal else 0
            if monitor.abnormal_run >= monitor.enter_count:
                monitor.active = True
                monitor.abnormal_run = 0
        else:
            monitor.healthy_run = monitor.healthy_run + 1 if finite and not instantaneous else 0
            if monitor.healthy_run >= monitor.exit_count:
                monitor.active = False
                monitor.healthy_run = 0
                monitor.positive = monitor.negative = 0.0
        states.append(monitor.active)
    return np.asarray(states)


def synthetic_cases() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    count = 300
    step = np.arange(count)
    active = (step >= 5) & (step < 8)

    true = np.zeros(count)
    persistent_raw = np.where(step >= 5, 3.0, 0.0)
    temporary_raw = np.where(active, 3.0, 0.0)

    dropout_true = np.full(count, 10.0)
    dropout_raw = np.where(active, 0.0, dropout_true)

    combined_true = np.where(step >= 5, 5.0, 0.0)
    combined_raw = combined_true + np.where(step >= 5, 3.0, 0.0)
    return [
        ("persistent_constant_bias", true, persistent_raw, true, step >= 5),
        ("temporary_bias", true, temporary_raw, true, active),
        ("temporary_dropout", dropout_true, dropout_raw, dropout_true, active),
        ("combined_bias_load", combined_true, combined_raw, combined_true, step >= 5),
    ]


def run_case(name: str, true: np.ndarray, raw: np.ndarray, prediction: np.ndarray,
             faulty: np.ndarray, config: dict) -> tuple[list[dict], np.ndarray]:
    monitor = SensorReliabilityMonitor(
        config["instant_threshold"], config["center"], config["allowance"],
        config["threshold"], config["enter_count"], config["exit_count"],
    )
    rows = []
    states = []
    for step, (y_true, raw_y_measured, y_hat) in enumerate(zip(true, raw, prediction)):
        result = monitor.update(float(raw_y_measured - y_hat))
        states.append(bool(result["sensor_suspect"]))
        rows.append({
            "scenario": name,
            "step": step,
            "raw_y_measured": raw_y_measured,
            "y_true": y_true,
            "y_hat": y_hat,
            "residual": result["residual"],
            "positive_cusum": result["positive_cusum"],
            "negative_cusum": result["negative_cusum"],
            "active_fault_state": result["sensor_suspect"],
            "healthy_run": result["healthy_run"],
            "substitution_flag": result["substitute"],
            "physical_sensor_fault": bool(faulty[step]),
        })
    return rows, np.asarray(states)


def exits(states: np.ndarray) -> np.ndarray:
    return np.flatnonzero(states[:-1] & ~states[1:]) + 1


def main() -> None:
    config = json.loads(
        (ROOT / "results/configs/reliability_final_config.json").read_text(encoding="utf-8")
    )["sensor"]
    rows = []
    for name, true, raw, prediction, faulty in synthetic_cases():
        case_rows, states = run_case(name, true, raw, prediction, faulty, config)
        false_exits = [index for index in exits(states) if faulty[index]]
        require(not false_exits, (name, false_exits))
        if name in {"temporary_bias", "temporary_dropout"}:
            require(any(index >= 8 for index in exits(states)), (name, exits(states)))
        rows.extend(case_rows)

    # The preserved combined-case trace proves the old instantaneous-only
    # recovery exited at 3.24 s while the sensor bias was still injected.
    with np.load(ROOT / "results/baseline_prefix/final_representative_traces.npz") as traces:
        prefix = "D_adaptive__combined_fault_load__"
        time = traces[prefix + "time"]
        residual = traces[prefix + "measured"] - traces[prefix + "virtual"]
        legacy = legacy_states(residual, config)
        fixed = SensorReliabilityMonitor(
            config["instant_threshold"], config["center"], config["allowance"],
            config["threshold"], config["enter_count"], config["exit_count"],
        )
        fixed_states = np.asarray([fixed.update(float(value))["sensor_suspect"] for value in residual])
    legacy_false = [index for index in exits(legacy) if time[index] >= 3.0]
    require(legacy_false and np.isclose(time[legacy_false[0]], 3.24))
    require(fixed_states[np.searchsorted(time, 3.24)])

    output = ROOT / "results/metrics/recovery_regression_trace.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Recovery regression passed; pre-fix false exit reproduced at {time[legacy_false[0]]:.2f} s.")


if __name__ == "__main__":
    main()
