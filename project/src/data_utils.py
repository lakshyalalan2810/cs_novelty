"""Dataset generation helpers for DC motor system identification."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from motor_model import DCMotorParams, simulate_motor


def make_random_step_signal(
    rng: np.random.Generator,
    duration: float,
    step_duration: float,
    levels: Sequence[float],
) -> Callable[[float], float]:
    """Return a reproducible piecewise-constant random signal."""
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    if not np.isfinite(step_duration) or step_duration <= 0:
        raise ValueError("step_duration must be finite and positive")
    if len(levels) == 0:
        raise ValueError("levels must not be empty")
    values = rng.choice(levels, size=int(np.ceil(duration / step_duration)) + 1)
    return lambda time: float(values[min(int(time / step_duration), len(values) - 1)])


def make_multisine_voltage(
    rng: np.random.Generator,
    voltage_limits: tuple[float, float] = (0.0, 12.0),
) -> Callable[[float], float]:
    """Return a bounded multi-sine voltage signal with random phases."""
    frequencies = np.array([0.13, 0.41, 0.89])
    amplitudes = np.array([0.5, 0.3, 0.2])
    phases = rng.uniform(0.0, 2 * np.pi, len(frequencies))
    low, high = voltage_limits

    def voltage(time: float) -> float:
        wave = np.sum(amplitudes * np.sin(2 * np.pi * frequencies * time + phases))
        return float(np.clip((low + high) / 2 + (high - low) * wave / 2, low, high))

    return voltage


def vary_motor_params(
    params: DCMotorParams,
    rng: np.random.Generator,
    variation: float = 0.1,
) -> DCMotorParams:
    """Independently vary nominal parameters by up to ``variation``."""
    varied = {
        name: value * rng.uniform(1 - variation, 1 + variation)
        for name, value in asdict(params).items()
    }
    return replace(params, **varied)


def generate_trajectory(
    run_id: int,
    voltage: Callable[[float], float],
    load_torque: Callable[[float], float],
    params: DCMotorParams,
    rng: np.random.Generator,
    *,
    duration: float,
    timestep: float,
    speed_noise_std: float,
) -> dict[str, np.ndarray]:
    """Simulate and record one healthy motor trajectory."""
    time, states = simulate_motor(
        voltage,
        load_torque,
        simulation_time=duration,
        timestep=timestep,
        params=params,
    )
    current, speed = states.T
    return {
        "run_id": np.full(len(time), run_id, dtype=np.int32),
        "time": time,
        "voltage": np.array([voltage(t) for t in time]),
        "current": current,
        "y_true": speed,
        "y_measured": speed + rng.normal(0.0, speed_noise_std, len(time)),
        "rpm": speed * 60 / (2 * np.pi),
        "load_torque": np.array([load_torque(t) for t in time]),
    }


def split_runs(
    trajectories: Sequence[dict[str, np.ndarray]],
    rng: np.random.Generator,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, list[dict[str, np.ndarray]]]:
    """Split whole trajectories before any overlapping windows are created."""
    if len(fractions) != 3:
        raise ValueError("split fractions must contain train, validation, and test values")
    if not np.isfinite(fractions).all() or any(fraction < 0 or fraction > 1 for fraction in fractions):
        raise ValueError("split fractions must be finite values between zero and one")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to one")
    order = rng.permutation(len(trajectories))
    train_end = int(len(order) * fractions[0])
    validation_end = train_end + int(len(order) * fractions[1])
    return {
        "train": [trajectories[index] for index in order[:train_end]],
        "validation": [trajectories[index] for index in order[train_end:validation_end]],
        "test": [trajectories[index] for index in order[validation_end:]],
    }


def build_training_sequences(
    trajectory: dict[str, np.ndarray],
    window_length: int = 20,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``[voltage, measured speed]`` windows and future true-speed targets."""
    features = np.column_stack((trajectory["voltage"], trajectory["y_measured"]))
    count = len(features) - window_length - horizon + 1
    if count <= 0:
        raise ValueError("trajectory is too short for the requested window and horizon")
    inputs = np.array(
        [features[start : start + window_length] for start in range(count)],
        dtype=np.float32,
    )
    targets = np.array(
        [
            trajectory["y_true"][
                start + window_length : start + window_length + horizon
            ]
            for start in range(count)
        ],
        dtype=np.float32,
    )
    run_ids = np.full(count, trajectory["run_id"][0], dtype=np.int32)
    return inputs, targets, run_ids


def fit_normalization(inputs: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    """Fit feature-wise statistics on training windows only."""
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    if inputs.ndim != 3 or targets.ndim != 2:
        raise ValueError("inputs must be 3-D and targets must be 2-D")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same sample count")
    if 0 in inputs.shape or 0 in targets.shape:
        raise ValueError("inputs and targets must be non-empty")
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("inputs and targets must contain only finite values")
    input_std = inputs.std(axis=(0, 1), dtype=np.float64)
    target_std = targets.std(axis=0, dtype=np.float64)
    return {
        "input_mean": inputs.mean(axis=(0, 1), dtype=np.float64),
        "input_std": np.where(input_std > 0, input_std, 1.0),
        "target_mean": targets.mean(axis=0, dtype=np.float64),
        "target_std": np.where(target_std > 0, target_std, 1.0),
    }


def normalize_sequences(
    inputs: np.ndarray,
    targets: np.ndarray,
    statistics: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply saved training statistics to input windows and targets."""
    normalized_inputs = (inputs - statistics["input_mean"]) / statistics["input_std"]
    normalized_targets = (targets - statistics["target_mean"]) / statistics["target_std"]
    return normalized_inputs.astype(np.float32), normalized_targets.astype(np.float32)


def save_processed_dataset(path: str | Path, **arrays: np.ndarray) -> None:
    """Save dataset arrays and metadata in one compressed NumPy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
