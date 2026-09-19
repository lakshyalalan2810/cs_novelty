"""V4 shared analysis helpers: Wilson, hierarchical bootstrap, Holm, DET.

All statistics treat the training seed as the top-level unit (resampled
first) and simulation seeds nested within, mirroring the frozen Part A/B
analysis. Time samples are never treated as independent observations.
"""

import math

import numpy as np
import pandas as pd

Z95 = 1.96


def wilson_ci(count: int, total: int, z: float = Z95) -> tuple[float, float]:
    """Wilson interval for a binomial proportion."""
    if total < 0 or count < 0 or count > total:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return (float("nan"), float("nan"))
    p = count / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total
                         + z * z / (4.0 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def hierarchical_bootstrap_mean(frame: pd.DataFrame, value_col: str,
                                n_replicates: int, rng_seed: int,
                                train_col: str = "training_seed",
                                sim_col: str = "simulation_seed",
                                ci: float = 0.95) -> dict:
    """Hierarchical bootstrap of the mean: training seeds, then sim seeds."""
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    trains = sorted(frame[train_col].unique())
    groups = {train: frame[frame[train_col] == train][value_col].to_numpy(float)
              for train in trains}
    rng = np.random.default_rng(rng_seed)
    means = np.zeros(n_replicates)
    for replicate in range(n_replicates):
        picks = rng.integers(0, len(trains), len(trains))
        pooled = np.concatenate(
            [rng.choice(groups[trains[pick]], len(groups[trains[pick]]),
                        replace=True) for pick in picks])
        means[replicate] = float(np.mean(pooled))
    alpha = 1.0 - ci
    observed = float(frame[value_col].mean())
    return {"observed_mean": observed,
            "bootstrap_mean": float(np.mean(means)),
            "ci_lo": float(np.percentile(means, 100 * alpha / 2)),
            "ci_hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
            "replicates": n_replicates, "rng_seed": rng_seed,
            "n_training_seeds": len(trains),
            "bootstrap_replicates": means}


def bootstrap_two_sided_p_value(replicates: np.ndarray,
                                null: float = 0.0) -> float:
    """Two-sided empirical p-value of the replicate distribution vs null."""
    replicates = np.asarray(replicates, dtype=float)
    shift = np.mean(replicates) - null
    tail = np.mean(np.abs(replicates - shift) >= abs(shift))
    return float(min(1.0, 2.0 * tail))


def cohens_dz(paired_deltas: np.ndarray) -> float:
    """Standardized paired effect (mean / sample SD); NaN if degenerate."""
    deltas = np.asarray(paired_deltas, dtype=float)
    if len(deltas) < 2:
        return float("nan")
    sd = float(np.std(deltas, ddof=1))
    if sd == 0:
        return float("nan")
    return float(np.mean(deltas) / sd)


def holm(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm step-down procedure; returns per-hypothesis adjusted p + reject."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    total = len(p_values)
    adjusted = [0.0] * total
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (total - rank) * p_values[index]
        running_max = max(running_max, min(1.0, candidate))
        adjusted[index] = running_max
    rejected = [False] * total
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (total - rank):
            rejected[index] = True
        else:
            break
    return [{"hypothesis": index, "p_value": p_values[index],
             "holm_adjusted_p": adjusted[index], "reject": rejected[index]}
            for index in range(total)]


def det_table(false_alarm_rates: list[float],
              miss_rates: list[float],
              labels: list[str]) -> pd.DataFrame:
    """Assemble a detection-error-tradeoff table from a threshold sweep."""
    if not (len(false_alarm_rates) == len(miss_rates) == len(labels)):
        raise ValueError("DET inputs must have equal length")
    frame = pd.DataFrame({"threshold_label": labels,
                          "false_alarm_rate": false_alarm_rates,
                          "miss_rate": miss_rates})
    frame["detection_rate"] = 1.0 - frame["miss_rate"]
    return frame.sort_values("false_alarm_rate", ignore_index=True)


def minimum_detectable_magnitude(magnitudes: list[float],
                                 detection_rates: list[float],
                                 target: float = 0.9) -> dict:
    """Interpolate the magnitude where detection first reaches target.

    Linear interpolation in log2(magnitude); returns NaN with an explicit
    flag when the target is never reached or already exceeded everywhere.
    """
    if len(magnitudes) != len(detection_rates) or len(magnitudes) < 2:
        raise ValueError("need at least two matched grid points")
    order = sorted(range(len(magnitudes)), key=lambda i: magnitudes[i])
    mags = [magnitudes[i] for i in order]
    rates = [detection_rates[i] for i in order]
    if any(mag <= 0 for mag in mags):
        raise ValueError("magnitudes must be positive")
    if all(rate >= target for rate in rates):
        return {"mdm": float("nan"), "flag": "below_grid"}
    if all(rate < target for rate in rates):
        return {"mdm": float("nan"), "flag": "above_grid"}
    for lower in range(len(mags) - 1):
        if rates[lower] < target <= rates[lower + 1]:
            x0, x1 = math.log2(mags[lower]), math.log2(mags[lower + 1])
            frac = (target - rates[lower]) / (rates[lower + 1]
                                              - rates[lower])
            return {"mdm": float(2.0 ** (x0 + frac * (x1 - x0))),
                    "flag": "interpolated"}
    raise AssertionError("unreachable")
