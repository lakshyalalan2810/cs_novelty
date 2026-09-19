"""V4 classical baselines, each with its own detector and substitution source.

All four baselines share one controller-ready interface::

    baseline.reset()
    out = baseline.update(...)  # baseline-specific inputs
    out["trusted"]        # bool: feedback may be used
    out["sensor_suspect"] # bool: debounced detector state
    out["substitute"]     # bool: harness must use out["estimate"] as feedback
    out["estimate"]       # float: substitution value (observer/filter output)

- E2 (``EKFResidualDetector``): owns an ``AugmentedStateEKF`` (voltage +
  current only) and runs a frozen-semantics CUSUM detector on the speed
  residual ``y_meas - omega_hat``. Substitution source: EKF omega. The
  current innovation is recorded as a diagnostic (a speed-sensor fault is
  invisible to it, which is why the speed residual drives detection).
- S1 (``FixedThresholdDetector``): fixed gate on ``|y_meas - estimate|``
  with enter/exit persistence. Estimate-agnostic: the harness feeds
  frozen-EKF omega (or any witness).
- S2 (``MedianRatePlausibilityFilter``): model-free plausibility on the
  speed channel alone: trailing-median deviation or rate-limit violation,
  with enter/exit persistence. Substitution value: trailing trusted
  median. Known blind spot (pinned by tests): slow drift within the rate
  limit is trusted.
- S3 (``LuenbergerBaseline``): current-driven extended Luenberger
  observer for ``[i, omega]`` exploiting back-EMF coupling, with its own
  CUSUM speed-residual detector. Gain from pole placement on the nominal
  linearization (no seed tuning). Unknown load torque is an unmodeled
  disturbance (documented limitation).

Calibration helpers implement the frozen percentile/grid rules on
train/validation residual sequences only; see scripts/v4_calibrate_baselines.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ekf_observer import AugmentedStateEKF, EKFConfig, EKFNumericalError
from motor_model import DCMotorParams

CUSUM_GRID = (90.0, 95.0, 97.5, 99.0, 99.5, 99.9)


@dataclass
class CusumCore:
    """Frozen-semantics residual/CUSUM gate shared by E2 and S3."""

    residual_gate: float
    center: float
    allowance: float
    threshold: float
    enter_count: int = 3
    exit_count: int = 5
    positive: float = field(default=0.0, init=False, repr=False)
    negative: float = field(default=0.0, init=False, repr=False)
    abnormal_run: int = field(default=0, init=False, repr=False)
    healthy_run: int = field(default=0, init=False, repr=False)
    active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if (self.residual_gate <= 0 or self.allowance < 0
                or self.threshold <= 0 or self.enter_count < 1
                or self.exit_count < 1):
            raise ValueError("invalid CUSUM core configuration")
        self.reset()

    def reset(self) -> None:
        self.positive = self.negative = 0.0
        self.abnormal_run = self.healthy_run = 0
        self.active = False

    def update(self, residual: float) -> dict[str, float | int | bool]:
        finite = bool(np.isfinite(residual))
        instantaneous = not finite or abs(residual) > self.residual_gate
        if finite:
            centered = residual - self.center
            self.positive = max(
                0.0, self.positive + centered - self.allowance)
            self.negative = max(
                0.0, self.negative - centered - self.allowance)
        score = max(self.positive, self.negative) if finite else np.inf
        abnormal = instantaneous or score > self.threshold
        if not self.active:
            self.abnormal_run = self.abnormal_run + 1 if abnormal else 0
            if self.abnormal_run >= self.enter_count:
                self.active = True
                self.abnormal_run = 0
        else:
            healthy = (finite and abs(residual) <= self.residual_gate
                       and score <= self.threshold)
            self.healthy_run = self.healthy_run + 1 if healthy else 0
            if self.healthy_run >= self.exit_count:
                self.active = False
                self.healthy_run = 0
                self.positive = self.negative = 0.0
        substitute = self.active or instantaneous
        return {"trusted": not substitute, "sensor_suspect": self.active,
                "substitute": substitute, "residual": float(residual),
                "positive_cusum": float(self.positive),
                "negative_cusum": float(self.negative),
                "score": float(score), "healthy_run": self.healthy_run}


class EKFResidualDetector:
    """E2: frozen EKF + own CUSUM speed-residual detector + EKF substitution."""

    def __init__(self, ekf_config: EKFConfig, detector: CusumCore) -> None:
        self.ekf = AugmentedStateEKF(ekf_config)
        self.detector = detector
        self._samples = 0
        self._failed = False

    def reset(self) -> None:
        self.ekf = AugmentedStateEKF(self.ekf.config)
        self.detector.reset()
        self._samples = 0
        self._failed = False

    def update(self, voltage: float, current_measurement: float,
               y_meas: float) -> dict:
        if self._failed:
            raise EKFNumericalError("E2 observer already failed")
        try:
            if self._samples == 0:
                estimate = self.ekf.initialize(current_measurement)
                innovation = 0.0
            else:
                estimate, diag = self.ekf.step(voltage, current_measurement)
                innovation = float(diag.innovation)
        except EKFNumericalError:
            self._failed = True
            raise
        self._samples += 1
        omega_hat = float(estimate[1])
        out = self.detector.update(y_meas - omega_hat)
        out["estimate"] = omega_hat
        out["innovation"] = innovation
        out["ekf_current"] = float(estimate[0])
        out["ekf_load"] = float(estimate[2])
        return out


class FixedThresholdDetector:
    """S1: fixed gate on |y_meas - estimate| with persistence."""

    def __init__(self, gate: float, enter_count: int = 3,
                 exit_count: int = 5) -> None:
        if gate <= 0 or enter_count < 1 or exit_count < 1:
            raise ValueError("invalid fixed-threshold configuration")
        self.gate = gate
        self.enter_count = enter_count
        self.exit_count = exit_count
        self.reset()

    def reset(self) -> None:
        self.abnormal_run = self.healthy_run = 0
        self.active = False

    def update(self, y_meas: float, estimate: float) -> dict:
        finite = bool(np.isfinite(y_meas)) and bool(np.isfinite(estimate))
        residual = y_meas - estimate if finite else float("nan")
        instantaneous = not finite or abs(residual) > self.gate
        if not self.active:
            self.abnormal_run = self.abnormal_run + 1 if instantaneous else 0
            if self.abnormal_run >= self.enter_count:
                self.active = True
                self.abnormal_run = 0
        else:
            healthy = finite and abs(residual) <= self.gate
            self.healthy_run = self.healthy_run + 1 if healthy else 0
            if self.healthy_run >= self.exit_count:
                self.active = False
                self.healthy_run = 0
        substitute = self.active or instantaneous
        return {"trusted": not substitute, "sensor_suspect": self.active,
                "substitute": substitute, "estimate": float(estimate),
                "residual": float(residual), "healthy_run": self.healthy_run}


class MedianRatePlausibilityFilter:
    """S2: model-free median/rate plausibility on the speed channel."""

    def __init__(self, window: int = 20, median_gate: float = 1.0,
                 rate_limit: float = 500.0, enter_count: int = 3,
                 exit_count: int = 5, dt: float = 0.01) -> None:
        if window < 1 or median_gate <= 0 or rate_limit <= 0:
            raise ValueError("invalid plausibility configuration")
        if enter_count < 1 or exit_count < 1 or dt <= 0:
            raise ValueError("invalid plausibility configuration")
        self.window = window
        self.median_gate = median_gate
        self.rate_limit = rate_limit
        self.enter_count = enter_count
        self.exit_count = exit_count
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.trusted_window: deque[float] = deque(maxlen=self.window)
        self.last_trusted: float | None = None
        self.abnormal_run = self.healthy_run = 0
        self.active = False

    def update(self, y_meas: float) -> dict:
        finite = bool(np.isfinite(y_meas))
        if not finite or self.last_trusted is None:
            median_dev = rate_dev = False
            median = y_meas if finite else float("nan")
        else:
            median = float(np.median(list(self.trusted_window)))
            median_dev = abs(y_meas - median) > self.median_gate
            rate_dev = (abs(y_meas - self.last_trusted)
                        > self.rate_limit * self.dt)
        instantaneous = (not finite) or median_dev or rate_dev
        if not self.active:
            self.abnormal_run = self.abnormal_run + 1 if instantaneous else 0
            if self.abnormal_run >= self.enter_count:
                self.active = True
                self.abnormal_run = 0
        else:
            healthy = finite and not median_dev and not rate_dev
            self.healthy_run = self.healthy_run + 1 if healthy else 0
            if self.healthy_run >= self.exit_count:
                self.active = False
                self.healthy_run = 0
        substitute = self.active or instantaneous
        if finite and not substitute:
            # Ingest trusted samples only; suspect samples must not drag
            # the median.
            self.trusted_window.append(float(y_meas))
            self.last_trusted = float(y_meas)
            median = float(np.median(list(self.trusted_window)))
        return {"trusted": not substitute, "sensor_suspect": self.active,
                "substitute": substitute, "estimate": float(median),
                "median_deviation": bool(median_dev),
                "rate_violation": bool(rate_dev),
                "healthy_run": self.healthy_run}


def _expm_small(matrix: np.ndarray, order: int = 16) -> np.ndarray:
    """Taylor matrix exponential (numpy-only; accurate for small norms).

    Deliberately avoids scipy.linalg.expm: in this container any scipy
    native call after torch initialization aborts the process (OMP
    Error #15, duplicate libiomp). With dt = 0.01 the argument norm is
    ~0.1, where order 16 is accurate to machine precision.
    """
    matrix = np.asarray(matrix, dtype=complex)
    term = np.eye(matrix.shape[0], dtype=complex)
    total = term.copy()
    for k in range(1, order + 1):
        term = term @ matrix / k
        total = total + term
    return total


def _acker_2x2(a_disc: np.ndarray, c_row: np.ndarray,
               poles: np.ndarray) -> np.ndarray:
    """Ackermann pole placement for a 2-state single-output observer."""
    p1, p2 = complex(poles[0]), complex(poles[1])
    # Desired characteristic polynomial: z^2 + a1 z + a0.
    a1 = -(p1 + p2)
    a0 = p1 * p2
    controllability = np.column_stack(
        (c_row.T, a_disc.T @ c_row.T)).astype(complex)
    if abs(np.linalg.det(controllability)) < 1e-12:
        raise RuntimeError("observer pair is not observable")
    selector = np.array([[0.0, 1.0]], dtype=complex)
    poly = a_disc @ a_disc + a1 * a_disc + a0 * np.eye(2)
    gain_t = selector @ np.linalg.inv(controllability) @ poly.T
    gain = np.asarray(gain_t.T, dtype=complex)
    if np.max(np.abs(gain.imag)) > 1e-9:
        raise RuntimeError("non-real Ackermann gain")
    return gain.real


def design_luenberger_gain(params: DCMotorParams | None = None,
                           dt: float = 0.01,
                           pole_factor: float = 4.0,
                           operating_speed: float = 35.0) -> np.ndarray:
    """Pole-placement gain for the current-measurement Luenberger observer.

    Linearizes the nominal motor at steady state for ``operating_speed``,
    discretizes exactly, and places observer poles ``pole_factor`` times
    faster (in continuous time) than the plant poles. Returns shape (2, 1).

    Numpy-only by design (see _expm_small): no scipy native calls, so the
    gain can be computed in any process regardless of torch import order.
    """
    from ekf_observer import smooth_friction_derivative
    params = params or DCMotorParams()
    friction_slope = smooth_friction_derivative(operating_speed, params)
    a_cont = np.array([
        [-params.resistance / params.inductance,
         -params.back_emf_constant / params.inductance],
        [params.torque_constant / params.inertia,
         -(params.viscous_friction + friction_slope) / params.inertia]])
    c_row = np.array([[1.0, 0.0]])
    a_disc = np.asarray(_expm_small(a_cont * dt).real, dtype=float)
    plant_poles = np.linalg.eigvals(a_cont)
    desired_disc = np.exp(pole_factor * plant_poles * dt)
    gain = _acker_2x2(a_disc, c_row, desired_disc)
    if gain.shape != (2, 1):
        raise RuntimeError("unexpected Luenberger gain shape")
    if np.max(np.abs(np.linalg.eigvals(a_disc - gain @ c_row))) >= 1.0:
        raise RuntimeError("Luenberger observer poles are not stable")
    return gain


class LuenbergerBaseline:
    """S3: back-EMF Luenberger observer + own CUSUM speed-residual detector."""

    def __init__(self, detector: CusumCore,
                 params: DCMotorParams | None = None, dt: float = 0.01,
                 pole_factor: float = 4.0,
                 gain: np.ndarray | None = None) -> None:
        self.detector = detector
        self.params = params or DCMotorParams()
        self.dt = dt
        self.gain = (np.asarray(gain, dtype=float).reshape(2, 1)
                     if gain is not None
                     else design_luenberger_gain(self.params, dt,
                                                 pole_factor))
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(2)
        self.detector.reset()

    def _predict(self, state: np.ndarray, voltage: float) -> np.ndarray:
        from motor_model import dc_motor_dynamics
        # Open-loop nominal prediction over one sample (load unknown to
        # the observer, assumed at its 0.03 N m baseline).
        derivative = lambda val: dc_motor_dynamics(
            0.0, val, voltage, 0.03, self.params)
        k1 = derivative(state)
        k2 = derivative(state + self.dt * k1 / 2)
        k3 = derivative(state + self.dt * k2 / 2)
        k4 = derivative(state + self.dt * k3)
        return state + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def update(self, voltage: float, current_measurement: float,
               y_meas: float) -> dict:
        if not (np.isfinite(voltage) and np.isfinite(current_measurement)):
            raise ValueError("observer inputs must be finite")
        predicted = self._predict(self.state, float(voltage))
        innovation = float(current_measurement) - float(predicted[0])
        self.state = predicted + (self.gain[:, 0] * innovation)
        if not np.isfinite(self.state).all():
            raise RuntimeError("Luenberger observer went nonfinite")
        omega_hat = float(self.state[1])
        out = self.detector.update(y_meas - omega_hat)
        out["estimate"] = omega_hat
        out["innovation"] = float(innovation)
        out["observer_current"] = float(self.state[0])
        return out


# ---------------------------------------------------------------------------
# Calibration-document factories (schema shared with v4_calibrate_baselines)
# ---------------------------------------------------------------------------

def detector_from_cusum_doc(doc: dict) -> CusumCore:
    """Build a CusumCore from a calibrated threshold document."""
    return CusumCore(
        residual_gate=float(doc["instant_threshold"]),
        center=float(doc["center"]),
        allowance=float(doc["allowance"]),
        threshold=float(doc["threshold"]),
        enter_count=int(doc.get("enter_count", 3)),
        exit_count=int(doc.get("exit_count", 5)))


# ---------------------------------------------------------------------------
# Calibration helpers (train/validation residual sequences only)
# ---------------------------------------------------------------------------

def calibrate_fixed_gate(residual_sequences: list[np.ndarray],
                         percentile: float = 99.9) -> float:
    """Fixed gate at a percentile of absolute clean residuals."""
    flat = np.concatenate([np.asarray(seq, dtype=float)
                           for seq in residual_sequences])
    if len(flat) == 0 or not np.isfinite(flat).all():
        raise ValueError("calibration residuals must be finite and non-empty")
    gate = float(np.percentile(np.abs(flat), percentile))
    if gate <= 0:
        raise ValueError("calibrated gate is not positive")
    return gate


def _cusum_scores(sequence: np.ndarray, center: float,
                  allowance: float) -> np.ndarray:
    positive = negative = 0.0
    scores = np.zeros(len(sequence))
    for index, value in enumerate(sequence):
        centered = value - center
        positive = max(0.0, positive + centered - allowance)
        negative = max(0.0, negative - centered - allowance)
        scores[index] = max(positive, negative)
    return scores


def calibrate_cusum_grid(residual_sequences: list[np.ndarray],
                         grid: tuple[float, ...] = CUSUM_GRID,
                         target_false_alarm: float = 0.001,
                         enter_count: int = 3,
                         exit_count: int = 5) -> dict:
    """Frozen grid rule: center/allowance + first grid threshold with
    debounced false-alarm rate at most target (else the last candidate)."""
    sequences = [np.asarray(seq, dtype=float) for seq in residual_sequences]
    flat = np.concatenate(sequences)
    if len(flat) == 0 or not np.isfinite(flat).all():
        raise ValueError("calibration residuals must be finite and non-empty")
    center = float(np.median(flat))
    allowance = 0.5 * float(np.std(flat, ddof=1))
    instant_gate = float(np.percentile(np.abs(flat), 99.9))
    entries = []
    for percentile in grid:
        scores = np.concatenate(
            [_cusum_scores(seq, center, allowance) for seq in sequences])
        threshold = float(np.percentile(scores, percentile))
        flagged = total = 0
        for seq in sequences:
            core = CusumCore(instant_gate, center, allowance, threshold,
                             enter_count, exit_count)
            for value in seq:
                flagged += core.update(float(value))["sensor_suspect"]
                total += 1
        rate = flagged / total if total else float("nan")
        entries.append({"percentile": percentile,
                        "cusum_threshold": threshold,
                        "validation_false_alarm_rate": rate})
    selected = next((entry for entry in entries
                     if entry["validation_false_alarm_rate"]
                     <= target_false_alarm), entries[-1])
    return {"center": center, "allowance": allowance,
            "instant_threshold": instant_gate,
            "threshold": selected["cusum_threshold"],
            "cusum_percentile": selected["percentile"],
            "calibration_grid": entries}


def calibrate_plausibility(measured_sequences: list[np.ndarray],
                           window: int = 20, dt: float = 0.01,
                           percentile: float = 99.9) -> dict:
    """Trailing-median gate and rate limit from clean speed sequences."""
    deviations, rates = [], []
    for seq in measured_sequences:
        seq = np.asarray(seq, dtype=float)
        if len(seq) <= window or not np.isfinite(seq).all():
            raise ValueError("calibration sequences must be finite and "
                             "longer than the median window")
        for index in range(window, len(seq)):
            baseline = float(np.median(seq[index - window:index]))
            deviations.append(abs(seq[index] - baseline))
            rates.append(abs(seq[index] - seq[index - 1]) / dt)
    gate = float(np.percentile(deviations, percentile))
    limit = float(np.percentile(rates, percentile))
    if gate <= 0 or limit <= 0:
        raise ValueError("calibrated plausibility bounds are not positive")
    return {"window": window, "median_gate": gate, "rate_limit": limit,
            "dt": dt}
