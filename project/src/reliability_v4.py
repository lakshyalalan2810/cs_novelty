"""V4 reliability monitor: frozen-compatible detector with optional fixes.

This module NEVER changes frozen behavior: with default flags,
:class:`ReliabilityMonitorV4` reproduces
:class:`reliability.SensorReliabilityMonitor` bit-identically (same
operations in the same order; proven by
tests/test_v4_detector_regression.py on recorded residual sequences).

Optional, independently flaggable fixes (all default Off / frozen):

(i)  ``cusum_clamp_multiple``: anti-windup clamp of both CUSUM accumulators
     at ``k * threshold``. Bounds post-fault discharge time, which in the
     frozen monitor grows without bound with fault duration.
(ii) ``warmup_blank_samples``: suppress reliability *entries* during the
     first N samples (startup transient). With ``warmup_trust_sensor`` the
     sensor is additionally trusted (no instantaneous substitution) during
     warmup. CUSUM still accumulates, so a genuine fault spanning warmup
     can enter immediately afterwards.
(iii) ``witness_residual_gate``: independent-witness AND rule. The caller
      passes ``witness_residual = y_meas - y_witness`` where the witness
      estimator does not consume measured speed (aux LSTM or EKF). Entry
      requires the main residual/CUSUM abnormality AND witness abnormality,
      so startup LSTM lag and load disturbances (witness agrees with the
      sensor) do not latch, while true sensor faults (witness disagrees)
      still enter. A missing/nonfinite witness fails safe toward the
      frozen main-only decision and raises a diagnostic flag.
(iv) ``recovery_boost``: extra per-sample CUSUM discharge applied while in
      the active state with an instantaneously healthy residual. ``0.0``
      reproduces frozen discharge dynamics exactly; combined with (i) the
      worst-case discharge time is bounded by
      ``k * threshold / (allowance + boost)`` samples plus ``exit_count``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


FROZEN_RESULT_KEYS = ("trusted", "sensor_suspect", "substitute", "residual",
                      "positive_cusum", "negative_cusum", "score",
                      "healthy_run")


@dataclass
class ReliabilityMonitorV4:
    """Sensor reliability monitor with optional startup/windup/witness fixes."""

    residual_gate: float
    center: float
    allowance: float
    threshold: float
    enter_count: int = 3
    exit_count: int = 5
    aux_recovery_gate: float | None = None
    # (i) anti-windup: clamp accumulators at k * threshold (None = frozen).
    cusum_clamp_multiple: float | None = None
    # (ii) startup gating: suppress entries for the first N samples.
    warmup_blank_samples: int = 0
    warmup_trust_sensor: bool = True
    # (iii) witness AND-rule gate on |y_meas - y_witness| (None = disabled).
    witness_residual_gate: float | None = None
    # (iv) extra CUSUM discharge per healthy sample while active.
    recovery_boost: float = 0.0

    def __post_init__(self) -> None:
        if (self.residual_gate <= 0 or self.allowance < 0
                or self.threshold <= 0 or self.enter_count < 1
                or self.exit_count < 1):
            raise ValueError("invalid reliability monitor configuration")
        if self.aux_recovery_gate is not None and self.aux_recovery_gate <= 0:
            raise ValueError("aux_recovery_gate must be positive when specified")
        if (self.cusum_clamp_multiple is not None
                and self.cusum_clamp_multiple <= 0):
            raise ValueError("cusum_clamp_multiple must be positive when set")
        if self.warmup_blank_samples < 0:
            raise ValueError("warmup_blank_samples must be nonnegative")
        if (self.witness_residual_gate is not None
                and self.witness_residual_gate <= 0):
            raise ValueError("witness_residual_gate must be positive when set")
        if self.recovery_boost < 0:
            raise ValueError("recovery_boost must be nonnegative")
        self.reset()

    def reset(self) -> None:
        self.positive = self.negative = 0.0
        self.abnormal_run = self.healthy_run = 0
        self.active = False
        self.sample_index = 0

    @property
    def _clamp(self) -> float | None:
        if self.cusum_clamp_multiple is None:
            return None
        return self.cusum_clamp_multiple * self.threshold

    def update(
        self,
        residual: float,
        *,
        aux_residual: float | None = None,
        witness_residual: float | None = None,
    ) -> dict[str, float | int | bool | None]:
        """Update with a raw-sensor residual; see module docstring."""
        finite = bool(np.isfinite(residual))
        instantaneous = not finite or abs(residual) > self.residual_gate
        clamped = False
        if finite:
            centered = residual - self.center
            self.positive = max(0.0, self.positive + centered - self.allowance)
            self.negative = max(0.0, self.negative - centered - self.allowance)
            cap = self._clamp
            if cap is not None:
                if self.positive > cap:
                    self.positive = cap
                    clamped = True
                if self.negative > cap:
                    self.negative = cap
                    clamped = True
        # (iv) accelerated discharge while active and instantaneously healthy.
        if (self.active and self.recovery_boost > 0.0 and finite
                and abs(residual) <= self.residual_gate):
            self.positive = max(0.0, self.positive - self.recovery_boost)
            self.negative = max(0.0, self.negative - self.recovery_boost)
        score = max(self.positive, self.negative) if finite else np.inf
        main_abnormal = instantaneous or score > self.threshold
        # (iii) independent-witness AND rule.
        if self.witness_residual_gate is None:
            witness_abnormal: bool | None = None
            witness_unavailable = False
            abnormal = main_abnormal
        else:
            if witness_residual is None or not bool(
                    np.isfinite(witness_residual)):
                # Fail safe: fall back to the main-only decision.
                witness_abnormal = True
                witness_unavailable = True
            else:
                witness_abnormal = (
                    abs(witness_residual) > self.witness_residual_gate)
                witness_unavailable = False
            abnormal = main_abnormal and witness_abnormal
        # (ii) warmup gating suppresses entries, never forces them.
        warmup_active = self.sample_index < self.warmup_blank_samples
        if not self.active:
            if warmup_active:
                self.abnormal_run = 0
            else:
                self.abnormal_run = self.abnormal_run + 1 if abnormal else 0
            if self.abnormal_run >= self.enter_count:
                self.active = True
                self.abnormal_run = 0
        else:
            healthy_candidate = (
                finite
                and abs(residual) <= self.residual_gate
                and score <= self.threshold
            )
            if healthy_candidate and self.aux_recovery_gate is not None:
                healthy_candidate = (
                    aux_residual is not None
                    and bool(np.isfinite(aux_residual))
                    and abs(aux_residual) <= self.aux_recovery_gate
                )
            self.healthy_run = self.healthy_run + 1 if healthy_candidate else 0
            if self.healthy_run >= self.exit_count:
                self.active = False
                self.healthy_run = 0
                self.positive = self.negative = 0.0
        if warmup_active and self.warmup_trust_sensor:
            substitute = False
        else:
            substitute = self.active or instantaneous
        self.sample_index += 1
        return {
            "trusted": not substitute,
            "sensor_suspect": self.active,
            "substitute": substitute,
            "residual": float(residual),
            "positive_cusum": float(self.positive),
            "negative_cusum": float(self.negative),
            "score": float(score),
            "healthy_run": self.healthy_run,
            "cusum_clamped": clamped,
            "warmup_active": warmup_active,
            "witness_abnormal": witness_abnormal,
            "witness_unavailable": witness_unavailable,
            "witness_residual": (None if witness_residual is None
                                 else float(witness_residual)),
        }


def v4_from_sensor_calibration(calibration: dict,
                               **overrides) -> ReliabilityMonitorV4:
    """Build a V4 monitor from a frozen sensor-calibration document.

    ``calibration`` is the parsed JSON of a ``sensor_seed*_calibration.json``
    file (``computed_values`` keys). Overrides set any V4 flag.
    """
    values = calibration["computed_values"]
    return ReliabilityMonitorV4(
        residual_gate=float(values["instant_threshold"]),
        center=float(values["center"]),
        allowance=float(values["allowance"]),
        threshold=float(values["threshold"]),
        enter_count=int(values.get("enter_count", 3)),
        exit_count=int(values.get("exit_count", 5)),
        **overrides,
    )


def v4_from_sensor_calibration_path(path: str | Path,
                                    **overrides) -> ReliabilityMonitorV4:
    """Load a sensor-calibration JSON file and build a V4 monitor."""
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    return v4_from_sensor_calibration(calibration, **overrides)
