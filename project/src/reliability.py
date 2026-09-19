"""Reliability signals for the learned motor model."""

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch

from auxiliary_sensor_model import AuxiliarySpeedEstimator
from lstm_model import LSTMForecaster

DIAGNOSTIC_STATE_NAMES = np.array(
    ["NORMAL", "SENSOR_FAULT", "MODEL_UNCERTAINTY", "BOTH"]
)
REDESIGN_STATE_NAMES = np.array(["NORMAL", "SENSOR_SUSPECT", "MODEL_MISMATCH"])


@dataclass
class SensorReliabilityMonitor:
    """Incremental residual/CUSUM gate for closed-loop measurement selection."""

    residual_gate: float
    center: float
    allowance: float
    threshold: float
    enter_count: int = 3
    exit_count: int = 5
    aux_recovery_gate: float | None = None

    def __post_init__(self) -> None:
        if self.residual_gate <= 0 or self.allowance < 0 or self.threshold <= 0 or self.enter_count < 1 or self.exit_count < 1:
            raise ValueError("invalid reliability monitor configuration")
        if self.aux_recovery_gate is not None and self.aux_recovery_gate <= 0:
            raise ValueError("aux_recovery_gate must be positive when specified")
        self.reset()

    def reset(self) -> None:
        self.positive = self.negative = 0.0
        self.abnormal_run = self.healthy_run = 0
        self.active = False

    def update(
        self,
        residual: float,
        *,
        aux_residual: float | None = None,
    ) -> dict[str, float | int | bool]:
        """Update the monitor with a new residual and return feedback decision.

        The ``residual`` must be computed from the **raw physical sensor
        measurement** minus the LSTM prediction.  It must never be computed
        from substituted/virtual feedback, otherwise recovery would be
        self-consistent and meaningless.

        Recovery from the ``active`` (sensor-suspect) state requires **all**
        of the following for ``exit_count`` consecutive samples:

        1. The residual is finite.
        2. ``|residual| <= residual_gate`` (instantaneous check).
        3. ``max(positive, negative) <= threshold`` (CUSUM has subsided).
        4. If ``aux_recovery_gate`` is set, ``aux_residual`` must exist, be
           finite, and satisfy ``|aux_residual| <= aux_recovery_gate``.

        CUSUM accumulators are reset to zero only **after** the full recovery
        condition has been satisfied, never merely because an instantaneous
        residual becomes small.
        """
        finite = bool(np.isfinite(residual))
        instantaneous = not finite or abs(residual) > self.residual_gate
        if finite:
            centered = residual - self.center
            self.positive = max(0.0, self.positive + centered - self.allowance)
            self.negative = max(0.0, self.negative - centered - self.allowance)
        score = max(self.positive, self.negative) if finite else np.inf
        abnormal = instantaneous or score > self.threshold
        if not self.active:
            self.abnormal_run = self.abnormal_run + 1 if abnormal else 0
            if self.abnormal_run >= self.enter_count:
                self.active = True
                self.abnormal_run = 0
        else:
            # Recovery requires BOTH instantaneous residual within gate
            # AND CUSUM score below threshold (not just instantaneous).
            healthy_candidate = (
                finite
                and abs(residual) <= self.residual_gate
                and score <= self.threshold
            )
            # V2: additionally require auxiliary agreement if configured
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
        substitute = self.active or instantaneous
        return {
            "trusted": not substitute,
            "sensor_suspect": self.active,
            "substitute": substitute,
            "residual": float(residual),
            "positive_cusum": float(self.positive),
            "negative_cusum": float(self.negative),
            "score": float(score),
            "healthy_run": self.healthy_run,
        }


def load_lstm_model(
    weights_path: str | Path,
    config_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[LSTMForecaster, dict]:
    """Load the trained Phase 3 model and its saved configuration."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = LSTMForecaster(**config["model"]).to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model, config


@torch.no_grad()
def mc_dropout_predict(
    model: LSTMForecaster,
    inputs: np.ndarray | torch.Tensor,
    passes: int = 30,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MC prediction mean and population variance with dropout active."""
    if passes < 2 or batch_size < 1:
        raise ValueError("passes must be at least 2 and batch_size must be positive")

    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    device = next(model.parameters()).device
    was_training = model.training
    means, variances = [], []
    model.train()
    try:
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size].to(device)
            repeated = batch.repeat_interleave(passes, dim=0)
            samples = model(repeated).reshape(len(batch), passes, -1)
            means.append(samples.mean(dim=1).cpu())
            variances.append(samples.var(dim=1, unbiased=False).cpu())
    finally:
        model.train(was_training)
    return torch.cat(means), torch.cat(variances)


def calibrate_thresholds(
    clean_residual: np.ndarray,
    clean_variance: np.ndarray,
    percentile: float = 95.0,
) -> dict[str, float]:
    """Calibrate both thresholds from clean validation samples only."""
    return {
        "tau_r": float(np.percentile(clean_residual, percentile)),
        "tau_sigma": float(np.percentile(clean_variance, percentile)),
    }


def classify_diagnostics(
    residual: np.ndarray,
    variance: np.ndarray,
    tau_r: float,
    tau_sigma: float,
) -> np.ndarray:
    """Map threshold exceedances to NORMAL, SENSOR_FAULT, MODEL_UNCERTAINTY, BOTH."""
    high_residual = np.asarray(residual) > tau_r
    high_variance = np.asarray(variance) > tau_sigma
    return high_residual.astype(np.int8) + 2 * high_variance.astype(np.int8)


def temporal_residual_features(
    residual: np.ndarray,
    window: int = 20,
    ewma_alpha: float = 0.1,
    scale: float | None = None,
) -> dict[str, np.ndarray]:
    """Return interpretable instantaneous and smoothed residual features."""
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 1 or window < 1 or not 0 < ewma_alpha <= 1:
        raise ValueError("residual must be 1-D, window positive, and alpha in (0, 1]")

    absolute = np.abs(residual)
    count = np.minimum(np.arange(1, len(residual) + 1), window)

    def rolling_mean(values: np.ndarray) -> np.ndarray:
        cumulative = np.r_[0.0, np.cumsum(values)]
        starts = np.maximum(0, np.arange(len(values)) - window + 1)
        return (cumulative[1:] - cumulative[starts]) / count

    signed_ewma = np.empty_like(residual)
    ewma = np.empty_like(absolute)
    if len(ewma):
        signed_ewma[0] = residual[0]
        ewma[0] = absolute[0]
        for index in range(1, len(ewma)):
            signed_ewma[index] = (
                ewma_alpha * residual[index] + (1 - ewma_alpha) * signed_ewma[index - 1]
            )
            ewma[index] = ewma_alpha * absolute[index] + (1 - ewma_alpha) * ewma[index - 1]
    denominator = 1.0 if scale is None else float(scale)
    if denominator <= 0:
        raise ValueError("scale must be positive")
    return {
        "signed": residual,
        "absolute": absolute,
        "rolling_mean": rolling_mean(absolute),
        "rolling_rms": np.sqrt(rolling_mean(residual**2)),
        "signed_ewma": signed_ewma,
        "ewma": ewma,
        "normalized": absolute / denominator,
    }


def persistent_alarm(
    abnormal: np.ndarray,
    enter_count: int = 3,
    exit_count: int = 20,
) -> np.ndarray:
    """Debounce one or more Boolean traces along their final axis."""
    abnormal = np.asarray(abnormal, dtype=bool)
    if abnormal.ndim not in (1, 2) or enter_count < 1 or exit_count < 1:
        raise ValueError("abnormal must be 1-D or 2-D and counts must be positive")
    squeeze = abnormal.ndim == 1
    rows = abnormal[None, :] if squeeze else abnormal
    alarm = np.zeros_like(rows)
    for row_index, row in enumerate(rows):
        active = False
        run = 0
        for index, value in enumerate(row):
            if value == active:
                run = 0
            else:
                run += 1
                if run >= (exit_count if active else enter_count):
                    active = not active
                    run = 0
            alarm[row_index, index] = active
    return alarm[0] if squeeze else alarm


def cusum_monitor(
    residual: np.ndarray,
    center: float,
    allowance: float,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Run a two-sided CUSUM and return both accumulators and alarm state."""
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 1 or allowance < 0 or threshold <= 0:
        raise ValueError("residual must be 1-D, allowance nonnegative, threshold positive")
    positive = np.zeros_like(residual)
    negative = np.zeros_like(residual)
    for index, value in enumerate(residual):
        previous_positive = positive[index - 1] if index else 0.0
        previous_negative = negative[index - 1] if index else 0.0
        centered = value - center
        positive[index] = max(0.0, previous_positive + centered - allowance)
        negative[index] = max(0.0, previous_negative - centered - allowance)
    score = np.maximum(positive, negative)
    return {
        "positive": positive,
        "negative": negative,
        "score": score,
        "alarm": score > threshold,
    }


def calibrate_cusum(
    clean_residuals: list[np.ndarray] | np.ndarray,
    percentile: float = 99.0,
    allowance_sigma: float = 0.5,
) -> dict[str, float]:
    """Fit CUSUM center, allowance, and threshold from clean sequences only."""
    array = np.asarray(clean_residuals, dtype=float)
    sequences = [array] if array.ndim == 1 else list(array)
    flat = np.concatenate(sequences)
    center = float(np.median(flat))
    sigma = float(np.std(flat, ddof=1))
    allowance = allowance_sigma * sigma
    scores = np.concatenate(
        [cusum_monitor(sequence, center, allowance, np.inf)["score"] for sequence in sequences]
    )
    return {
        "center": center,
        "allowance": allowance,
        "threshold": float(np.percentile(scores, percentile)),
    }


@torch.no_grad()
def guarded_predict(
    model: LSTMForecaster,
    voltage: np.ndarray,
    measured: np.ndarray,
    normalization: dict,
    window_length: int,
    residual_gate: float,
    cusum: dict[str, float] | None = None,
    *,
    enter_count: int = 1,
    exit_count: int = 1,
) -> dict[str, np.ndarray]:
    """Predict online, judging raw measurements and substituting only feedback history."""
    voltage = np.asarray(voltage, dtype=np.float32)
    measured = np.asarray(measured, dtype=np.float32)
    squeeze = voltage.ndim == 1
    if squeeze:
        voltage, measured = voltage[None, :], measured[None, :]
    if voltage.shape != measured.shape or voltage.ndim != 2:
        raise ValueError("voltage and measured must have equal 1-D or 2-D shapes")
    if (
        window_length < 1
        or voltage.shape[1] <= window_length
        or residual_gate <= 0
        or enter_count < 1
        or exit_count < 1
    ):
        raise ValueError("invalid window length, residual gate, or persistence count")

    input_mean = np.asarray(normalization["input_mean"], dtype=np.float32)
    input_std = np.asarray(normalization["input_std"], dtype=np.float32)
    target_mean = float(normalization["target_mean"][0])
    target_std = float(normalization["target_std"][0])
    device = next(model.parameters()).device
    batch_size, sample_count = voltage.shape
    history = np.stack((voltage[:, :window_length], measured[:, :window_length]), axis=2)
    length = sample_count - window_length
    prediction = np.empty((batch_size, length), dtype=np.float32)
    residual = np.empty_like(prediction)
    substituted = np.zeros((batch_size, length), dtype=bool)
    sensor_alarm = np.zeros((batch_size, length), dtype=bool)
    positive = np.zeros(batch_size)
    negative = np.zeros(batch_size)
    cusum_score = np.zeros_like(prediction)
    active = np.zeros(batch_size, dtype=bool)
    abnormal_run = np.zeros(batch_size, dtype=int)
    healthy_run = np.zeros(batch_size, dtype=int)

    was_training = model.training
    model.eval()
    try:
        for output_index, sample_index in enumerate(range(window_length, sample_count)):
            normalized = (history - input_mean) / input_std
            output = model(torch.from_numpy(normalized).to(device)).cpu().numpy()[:, 0]
            physical = output * target_std + target_mean
            if not np.isfinite(physical).all():
                raise FloatingPointError("LSTM produced NaN or Inf")
            error = measured[:, sample_index] - physical
            finite = np.isfinite(error)
            if cusum is not None:
                centered = error - cusum["center"]
                positive = np.where(
                    finite,
                    np.maximum(0.0, positive + centered - cusum["allowance"]),
                    positive,
                )
                negative = np.where(
                    finite,
                    np.maximum(0.0, negative - centered - cusum["allowance"]),
                    negative,
                )
            score = np.where(finite, np.maximum(positive, negative), np.inf)
            instantaneous = ~finite | (np.abs(error) > residual_gate)
            abnormal = instantaneous.copy()
            if cusum is not None:
                abnormal |= score > cusum["threshold"]

            previously_active = active.copy()
            abnormal_run = np.where(~active & abnormal, abnormal_run + 1, 0)
            active |= abnormal_run >= enter_count
            healthy = np.isfinite(error) & (np.abs(error) <= residual_gate)
            if cusum is not None:
                healthy &= score <= cusum["threshold"]
            healthy_run = np.where(previously_active & healthy, healthy_run + 1, 0)
            recovered = previously_active & (healthy_run >= exit_count)
            active[recovered] = False
            positive[recovered] = 0.0
            negative[recovered] = 0.0
            abnormal_run[active] = 0

            prediction[:, output_index] = physical
            residual[:, output_index] = error
            # Protect the history immediately; expose the debounced state separately.
            feedback_suspect = active | instantaneous
            substituted[:, output_index] = feedback_suspect
            sensor_alarm[:, output_index] = active
            cusum_score[:, output_index] = score
            feedback = np.where(feedback_suspect, physical, measured[:, sample_index])
            next_sample = np.stack((voltage[:, sample_index], feedback), axis=1)
            history = np.concatenate((history[:, 1:], next_sample[:, None, :]), axis=1)
    finally:
        model.train(was_training)

    result = {
        "prediction": prediction,
        "residual": residual,
        "substituted": substituted,
        "sensor_alarm": sensor_alarm,
        "cusum_score": cusum_score,
    }
    return {key: value[0] for key, value in result.items()} if squeeze else result


@torch.no_grad()
def guarded_predict_v2(
    model: LSTMForecaster,
    aux_model: "AuxiliarySpeedEstimator",
    voltage: np.ndarray,
    measured: np.ndarray,
    current: np.ndarray,
    normalization: dict,
    aux_normalization: dict,
    window_length: int,
    residual_gate: float,
    aux_recovery_gate: float,
    cusum: dict[str, float] | None = None,
    *,
    enter_count: int = 1,
    exit_count: int = 1,
) -> dict[str, np.ndarray]:
    """V2 guarded prediction with auxiliary recovery agreement.

    Extends :func:`guarded_predict` by maintaining a parallel
    ``[voltage, current]`` history for the auxiliary speed estimator.
    Recovery requires the auxiliary estimate to agree with the measured
    speed within ``aux_recovery_gate``.

    Operates on 1-D trajectories only.
    """
    voltage = np.asarray(voltage, dtype=np.float32)
    measured = np.asarray(measured, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if voltage.ndim != 1 or measured.ndim != 1 or current.ndim != 1:
        raise ValueError("voltage, measured, and current must be 1-D")
    if not (voltage.shape == measured.shape == current.shape):
        raise ValueError("voltage, measured, and current must have equal shapes")
    if (
        window_length < 1
        or voltage.shape[0] <= window_length
        or residual_gate <= 0
        or aux_recovery_gate <= 0
        or enter_count < 1
        or exit_count < 1
    ):
        raise ValueError("invalid parameters")

    # Main model normalization
    input_mean = np.asarray(normalization["input_mean"], dtype=np.float32)
    input_std = np.asarray(normalization["input_std"], dtype=np.float32)
    target_mean = float(normalization["target_mean"][0])
    target_std = float(normalization["target_std"][0])
    device = next(model.parameters()).device

    # Auxiliary model normalization
    aux_input_mean = np.asarray(aux_normalization["input_mean"], dtype=np.float32)
    aux_input_std = np.asarray(aux_normalization["input_std"], dtype=np.float32)
    aux_target_mean = float(aux_normalization["target_mean"][0])
    aux_target_std = float(aux_normalization["target_std"][0])

    sample_count = len(voltage)
    # Main history: [voltage, feedback_speed]
    history = np.stack((voltage[:window_length], measured[:window_length]), axis=1)
    # Aux history: [voltage, current] — always uses real current
    aux_history = np.stack(
        (voltage[:window_length], current[:window_length]), axis=1,
    )

    length = sample_count - window_length
    prediction = np.empty(length, dtype=np.float32)
    residual_arr = np.empty(length, dtype=np.float32)
    aux_prediction_arr = np.empty(length, dtype=np.float32)
    aux_residual_arr = np.empty(length, dtype=np.float32)
    substituted = np.zeros(length, dtype=bool)
    sensor_alarm = np.zeros(length, dtype=bool)
    positive_val = 0.0
    negative_val = 0.0
    cusum_score_arr = np.zeros(length, dtype=np.float32)
    active = False
    abnormal_run = 0
    healthy_run = 0

    was_main_training = model.training
    was_aux_training = aux_model.training
    model.eval()
    aux_model.eval()
    try:
        for output_index, sample_index in enumerate(
            range(window_length, sample_count)
        ):
            # --- Main LSTM prediction ---
            normalized = (history - input_mean) / input_std
            output = model(
                torch.from_numpy(normalized[None]).to(device)
            ).cpu().numpy()[0, 0]
            physical = output * target_std + target_mean

            # --- Auxiliary prediction ---
            aux_normalized = (aux_history - aux_input_mean) / aux_input_std
            aux_output = aux_model(
                torch.from_numpy(aux_normalized[None]).to(device)
            ).cpu().numpy()[0, 0]
            aux_physical = aux_output * aux_target_std + aux_target_mean

            # --- Residuals ---
            error = measured[sample_index] - physical
            aux_error = measured[sample_index] - aux_physical

            # --- CUSUM ---
            if cusum is not None:
                centered = error - cusum["center"]
                positive_val = max(0.0, positive_val + centered - cusum["allowance"])
                negative_val = max(
                    0.0, negative_val - centered - cusum["allowance"]
                )
            score = max(positive_val, negative_val)

            instantaneous = (
                not np.isfinite(error) or abs(error) > residual_gate
            )
            abnormal = instantaneous or (
                cusum is not None and score > cusum["threshold"]
            )

            # --- State machine ---
            previously_active = active
            if not active:
                abnormal_run = abnormal_run + 1 if abnormal else 0
                if abnormal_run >= enter_count:
                    active = True
                    abnormal_run = 0
            else:
                # Recovery: main residual + CUSUM + auxiliary agreement
                healthy_candidate = (
                    np.isfinite(error)
                    and abs(error) <= residual_gate
                    and (
                        cusum is None or score <= cusum["threshold"]
                    )
                    and np.isfinite(aux_error)
                    and abs(aux_error) <= aux_recovery_gate
                )
                healthy_run = healthy_run + 1 if healthy_candidate else 0
                if healthy_run >= exit_count:
                    active = False
                    healthy_run = 0
                    positive_val = 0.0
                    negative_val = 0.0
                abnormal_run = 0 if active else abnormal_run

            # --- Record ---
            prediction[output_index] = physical
            residual_arr[output_index] = error
            aux_prediction_arr[output_index] = aux_physical
            aux_residual_arr[output_index] = aux_error
            feedback_suspect = active or instantaneous
            substituted[output_index] = feedback_suspect
            sensor_alarm[output_index] = active
            cusum_score_arr[output_index] = score

            # --- Update main history ---
            feedback = physical if feedback_suspect else measured[sample_index]
            next_main = np.array(
                [voltage[sample_index], feedback], dtype=np.float32,
            )
            history = np.concatenate(
                (history[1:], next_main[None, :]), axis=0,
            )
            # --- Update aux history (always real current) ---
            next_aux = np.array(
                [voltage[sample_index], current[sample_index]],
                dtype=np.float32,
            )
            aux_history = np.concatenate(
                (aux_history[1:], next_aux[None, :]), axis=0,
            )
    finally:
        model.train(was_main_training)
        aux_model.train(was_aux_training)

    return {
        "prediction": prediction,
        "residual": residual_arr,
        "substituted": substituted,
        "sensor_alarm": sensor_alarm,
        "cusum_score": cusum_score_arr,
        "aux_prediction": aux_prediction_arr,
        "aux_residual": aux_residual_arr,
    }



@torch.no_grad()
def multistep_error(
    model: LSTMForecaster,
    voltage: np.ndarray,
    measured: np.ndarray,
    normalization: dict,
    window_length: int,
    horizon: int = 15,
    batch_size: int = 256,
    *,
    observed: np.ndarray | None = None,
    metric: str = "rms",
    scale: float | None = None,
) -> np.ndarray:
    """Return causal H-step error, reported when all H observations exist."""
    voltage = np.asarray(voltage, dtype=np.float32)
    measured = np.asarray(measured, dtype=np.float32)
    observed = measured if observed is None else np.asarray(observed, dtype=np.float32)
    count = len(voltage) - window_length - horizon + 1
    if (
        voltage.ndim != 1
        or voltage.shape != measured.shape
        or observed.shape != measured.shape
        or count < 1
        or batch_size < 1
        or metric not in {"mae", "rms", "normalized"}
        or (metric == "normalized" and (scale is None or scale <= 0))
    ):
        raise ValueError("invalid trajectory, horizon, batch size, metric, or scale")

    input_mean = np.asarray(normalization["input_mean"], dtype=np.float32)
    input_std = np.asarray(normalization["input_std"], dtype=np.float32)
    target_mean = float(normalization["target_mean"][0])
    target_std = float(normalization["target_std"][0])
    features = np.column_stack((voltage, measured))
    score = np.full(len(voltage) - window_length, np.nan, dtype=np.float32)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for start in range(0, count, batch_size):
            origins = np.arange(start, min(start + batch_size, count))
            history = np.stack(
                [features[index : index + window_length] for index in origins]
            )
            window = torch.from_numpy((history - input_mean) / input_std).to(device)
            predictions = []
            for step in range(horizon):
                normalized_prediction = model(window)[:, 0]
                physical_prediction = normalized_prediction * target_std + target_mean
                predictions.append(physical_prediction.cpu().numpy())
                future_index = origins + window_length + step
                normalized_voltage = torch.from_numpy(
                    (voltage[future_index] - input_mean[0]) / input_std[0]
                ).to(device)
                normalized_speed = (physical_prediction - input_mean[1]) / input_std[1]
                next_sample = torch.stack((normalized_voltage, normalized_speed), dim=1)
                window = torch.cat((window[:, 1:], next_sample[:, None, :]), dim=1)
            predictions = np.stack(predictions, axis=1)
            targets = np.stack(
                [observed[index + window_length : index + window_length + horizon] for index in origins]
            )
            error = targets - predictions
            if metric == "mae":
                values = np.mean(np.abs(error), axis=1)
            else:
                if metric == "normalized":
                    error = error / float(scale)
                values = np.sqrt(np.mean(error**2, axis=1))
            score[origins + horizon - 1] = values
    finally:
        model.train(was_training)
    return score


def classify_redesign(sensor_alarm: np.ndarray, mismatch_score: np.ndarray, threshold: float) -> np.ndarray:
    """Use sensor precedence because mismatch error is not sensor-independent."""
    sensor_alarm = np.asarray(sensor_alarm, dtype=bool)
    mismatch_alarm = np.nan_to_num(np.asarray(mismatch_score) > threshold, nan=False)
    if sensor_alarm.shape != mismatch_alarm.shape:
        raise ValueError("sensor and mismatch signals must have equal shapes")
    return np.where(sensor_alarm, 1, np.where(mismatch_alarm, 2, 0)).astype(np.int8)


@dataclass
class DualVirtualSensorArbitrator:
    """Interpretable hard-gating arbitrator between physical, main, and auxiliary sensors (V3).

    Hierarchy:
    1. If physical sensor trusted:
           feedback = y_measured
    2. Elif main virtual model considered reliable:
           feedback = y_main
    3. Elif auxiliary virtual model considered reliable:
           feedback = y_aux
    4. Else:
           feedback = bounded hold of the last trusted physical speed
    """

    agreement_threshold: float = 3.5
    param_mismatch_threshold: float = 7.0
    param_mismatch_recovery_threshold: float = 3.5
    param_mismatch_recovery_count: int = 50
    ewma_alpha: float = 0.05
    startup_blanking_time: float = 1.0
    aux_speed_bounds: tuple[float, float] = (0.0, 100.0)

    def __post_init__(self) -> None:
        low, high = self.aux_speed_bounds
        values = (
            self.agreement_threshold,
            self.param_mismatch_threshold,
            self.param_mismatch_recovery_threshold,
            self.ewma_alpha,
            self.startup_blanking_time,
            low,
            high,
        )
        if (
            not np.isfinite(values).all()
            or self.agreement_threshold <= 0
            or self.param_mismatch_recovery_threshold <= 0
            or self.param_mismatch_threshold <= self.param_mismatch_recovery_threshold
            or not 0 < self.ewma_alpha <= 1
            or self.startup_blanking_time < 0
            or self.param_mismatch_recovery_count < 1
            or low >= high
        ):
            raise ValueError("invalid arbitration configuration")
        self.reset()

    def reset(self) -> None:
        self.ewma_aux_trusted = 0.0
        self.aux_param_mismatch = False
        self.aux_param_recovery_run = 0
        self.main_degraded_latch = False
        self.current_source = "PHYSICAL"
        self.current_source_code = 0
        self.total_switches = 0

    def update(
        self,
        t: float,
        y_measured: float,
        y_main: float,
        y_aux: float,
        sensor_trusted: bool,
        fallback_speed: float,
    ) -> dict[str, float | int | bool | str]:
        if sensor_trusted:
            source = "PHYSICAL"
            source_code = 0
            feedback = y_measured
            # Update auxiliary consistency while sensor is trustworthy
            if t >= self.startup_blanking_time and np.isfinite(y_aux):
                r_aux = abs(y_measured - y_aux)
                self.ewma_aux_trusted = (
                    (1 - self.ewma_alpha) * self.ewma_aux_trusted
                    + self.ewma_alpha * r_aux
                )
                if self.ewma_aux_trusted > self.param_mismatch_threshold:
                    self.aux_param_mismatch = True
                    self.aux_param_recovery_run = 0
                elif self.aux_param_mismatch:
                    clean = (
                        r_aux <= self.param_mismatch_recovery_threshold
                        and self.ewma_aux_trusted <= self.param_mismatch_recovery_threshold
                    )
                    self.aux_param_recovery_run = self.aux_param_recovery_run + 1 if clean else 0
                    if self.aux_param_recovery_run >= self.param_mismatch_recovery_count:
                        self.aux_param_mismatch = False
                        self.aux_param_recovery_run = 0
            elif self.aux_param_mismatch:
                self.aux_param_recovery_run = 0
            # When physical sensor is healthy, main virtual degradation is unlatched
            self.main_degraded_latch = False
        else:
            # Physical sensor is untrusted -> Arbitrate between Main, Aux, Fallback
            diff_main_aux = (
                abs(y_main - y_aux)
                if (np.isfinite(y_main) and np.isfinite(y_aux))
                else np.inf
            )
            if diff_main_aux > self.agreement_threshold:
                self.main_degraded_latch = True

            main_reliable = (
                np.isfinite(y_main)
                and not self.main_degraded_latch
            )
            aux_reliable = (
                np.isfinite(y_aux)
                and not self.aux_param_mismatch
                and (self.aux_speed_bounds[0] <= y_aux <= self.aux_speed_bounds[1])
            )

            if main_reliable:
                source = "MAIN_VIRTUAL"
                source_code = 1
                feedback = y_main
            elif aux_reliable:
                source = "AUX_VIRTUAL"
                source_code = 2
                feedback = y_aux
            else:
                if not np.isfinite(fallback_speed):
                    raise ValueError("fallback_speed must be finite when fallback is selected")
                source = "FALLBACK"
                source_code = 3
                feedback = fallback_speed

        switched = (source_code != self.current_source_code)
        if switched:
            self.total_switches += 1
            self.current_source = source
            self.current_source_code = source_code

        return {
            "feedback": float(feedback),
            "source": source,
            "source_code": source_code,
            "switched": switched,
            "total_switches": self.total_switches,
            "aux_param_mismatch": self.aux_param_mismatch,
            "aux_param_recovery_run": self.aux_param_recovery_run,
            "ewma_aux_trusted": float(self.ewma_aux_trusted),
            "main_degraded": self.main_degraded_latch,
        }


class ThreeWayAttributionState(IntEnum):
    """Interpretable C4 consistency states."""

    NORMAL = 0
    LIKELY_SENSOR_FAULT = 1
    LIKELY_PLANT_OR_MAIN_MISMATCH = 2
    LIKELY_AUX_MISMATCH = 3
    AMBIGUOUS = 4


C4_ATTRIBUTION_STATE_NAMES = np.array([state.name for state in ThreeWayAttributionState])


class SuppressionEvidenceAction(IntEnum):
    """How a detector should treat accumulated entry evidence during suppression."""

    NONE = 0
    RESET = 1
    FREEZE = 2


@dataclass(frozen=True)
class SuppressionEvidencePolicy:
    """Apply the deterministic C4 stale-entry-evidence policy.

    A positively identified main/plant mismatch invalidates residual evidence
    accumulated for a physical-sensor entry.  The first suppressed entry asks
    the caller to reset that evidence once.  While the same positive mismatch
    remains active, subsequent updates ask the caller to freeze the reset
    evidence.  As soon as the mismatch evidence disappears, normal detector
    accumulation resumes from that clean baseline.
    """

    def apply(
        self,
        action: SuppressionEvidenceAction | str | int,
        previous_positive: float,
        previous_negative: float,
        proposed_positive: float,
        proposed_negative: float,
    ) -> tuple[float, float]:
        values = np.asarray(
            [previous_positive, previous_negative, proposed_positive, proposed_negative],
            dtype=float,
        )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("CUSUM evidence values must be finite and nonnegative")
        action = (
            SuppressionEvidenceAction[action]
            if isinstance(action, str)
            else SuppressionEvidenceAction(action)
        )
        if action == SuppressionEvidenceAction.RESET:
            return 0.0, 0.0
        if action == SuppressionEvidenceAction.FREEZE:
            return float(previous_positive), float(previous_negative)
        return float(proposed_positive), float(proposed_negative)


@dataclass
class ThreeWayConsistencyAttributor:
    """C4 attribution from physical, main-model, and auxiliary speed consistency.

    Threshold tuples are ordered ``(sensor-main, sensor-aux, main-aux)``.  The
    agreement thresholds are the clean-validation p75 pairwise distances and
    the disagreement thresholds are the clean-validation p99.9 pairwise
    distances.  The agreement threshold for each pair must therefore be lower
    than its strong-disagreement threshold.  This component consumes those
    locked calibration values; it does not fit or tune them online.

    Classification and enter/exit persistence use the current RAW pairwise
    disagreements only.  EWMA distances are exposed solely as diagnostics and
    never influence candidate state, persisted state, or entry veto.  Entry
    suppression is conservative: a detector request is inhibited only while
    both the persisted state and the current raw pattern positively identify
    the main model as the outlier and the auxiliary witness is trusted.
    """

    agreement_thresholds: tuple[float, float, float]
    disagreement_thresholds: tuple[float, float, float]
    ewma_alpha: float
    enter_count: int = 3
    exit_count: int = 5

    def __post_init__(self) -> None:
        agreement = np.asarray(self.agreement_thresholds, dtype=float)
        disagreement = np.asarray(self.disagreement_thresholds, dtype=float)
        if (
            agreement.shape != (3,)
            or disagreement.shape != (3,)
            or not np.isfinite(agreement).all()
            or not np.isfinite(disagreement).all()
            or np.any(agreement < 0)
            or np.any(disagreement <= agreement)
            or not np.isfinite(self.ewma_alpha)
            or not 0 < self.ewma_alpha <= 1
            or self.enter_count < 1
            or self.exit_count < 1
        ):
            raise ValueError("invalid C4 attribution configuration")
        self._agreement = agreement
        self._disagreement = disagreement
        self.reset()

    def reset(self) -> None:
        self.state = ThreeWayAttributionState.NORMAL
        self.candidate_state = ThreeWayAttributionState.NORMAL
        self.candidate_run = 0
        self.state_persistence = 0
        self.state_switches = 0
        self.suppression_active = False
        self.suppression_episodes = 0
        self._filtered = np.full(3, np.nan, dtype=float)

    def _classify(self, distances: np.ndarray) -> ThreeWayAttributionState:
        sm, sa, ma = distances
        a_sm, a_sa, a_ma = self._agreement
        d_sm, d_sa, d_ma = self._disagreement
        if sm <= a_sm and sa <= a_sa and ma <= a_ma:
            return ThreeWayAttributionState.NORMAL
        if sm >= d_sm and sa >= d_sa and ma <= a_ma:
            return ThreeWayAttributionState.LIKELY_SENSOR_FAULT
        if sm >= d_sm and sa <= a_sa and ma >= d_ma:
            return ThreeWayAttributionState.LIKELY_PLANT_OR_MAIN_MISMATCH
        if sm <= a_sm and sa >= d_sa and ma >= d_ma:
            return ThreeWayAttributionState.LIKELY_AUX_MISMATCH
        return ThreeWayAttributionState.AMBIGUOUS

    def _update_persisted_state(self, candidate: ThreeWayAttributionState) -> None:
        if candidate == self.state:
            self.candidate_state = candidate
            self.candidate_run = 0
            self.state_persistence += 1
            return

        if candidate == self.candidate_state:
            self.candidate_run += 1
        else:
            self.candidate_state = candidate
            self.candidate_run = 1

        required = (
            self.exit_count
            if candidate == ThreeWayAttributionState.NORMAL
            else self.enter_count
        )
        if self.candidate_run >= required:
            self.state = candidate
            self.state_switches += 1
            self.state_persistence = 1
            self.candidate_run = 0
        else:
            self.state_persistence += 1

    def _force_ambiguous(self) -> None:
        """Fail closed when the auxiliary witness cannot support attribution."""
        if self.state != ThreeWayAttributionState.AMBIGUOUS:
            self.state = ThreeWayAttributionState.AMBIGUOUS
            self.state_switches += 1
            self.state_persistence = 1
        else:
            self.state_persistence += 1
        self.candidate_state = ThreeWayAttributionState.AMBIGUOUS
        self.candidate_run = 0
        self.suppression_active = False
        self._filtered[:] = np.nan

    def update(
        self,
        y_sensor: float,
        y_main: float,
        y_aux: float | None,
        *,
        auxiliary_trusted: bool,
        detector_requests_entry: bool,
    ) -> dict[str, float | int | bool | str]:
        """Update C4 attribution and authorize an existing detector entry request."""
        sensor_finite = bool(np.isfinite(y_sensor))
        main_finite = bool(np.isfinite(y_main))
        aux_finite = y_aux is not None and bool(np.isfinite(y_aux))
        witness_trusted = bool(auxiliary_trusted and aux_finite)

        d_sm = abs(float(y_sensor) - float(y_main)) if sensor_finite and main_finite else np.inf
        d_sa = (
            abs(float(y_sensor) - float(y_aux))
            if sensor_finite and aux_finite
            else np.inf
        )
        d_ma = (
            abs(float(y_main) - float(y_aux))
            if main_finite and aux_finite
            else np.inf
        )

        valid_triplet = sensor_finite and main_finite and witness_trusted
        if not valid_triplet:
            self._force_ambiguous()
            raw_state = ThreeWayAttributionState.AMBIGUOUS
            candidate = ThreeWayAttributionState.AMBIGUOUS
        else:
            raw = np.asarray([d_sm, d_sa, d_ma], dtype=float)
            raw_state = self._classify(raw)
            candidate = raw_state
            self._update_persisted_state(candidate)

            # Diagnostic-only EWMA.  Locked C4 attribution logic is based on
            # raw pairwise distances; these values must not feed classification,
            # persistence, or veto decisions.
            missing_filter = ~np.isfinite(self._filtered)
            self._filtered[missing_filter] = raw[missing_filter]
            self._filtered[~missing_filter] = (
                (1.0 - self.ewma_alpha) * self._filtered[~missing_filter]
                + self.ewma_alpha * raw[~missing_filter]
            )

        affirmative_main_mismatch = bool(
            valid_triplet
            and self.state == ThreeWayAttributionState.LIKELY_PLANT_OR_MAIN_MISMATCH
            and raw_state == ThreeWayAttributionState.LIKELY_PLANT_OR_MAIN_MISMATCH
        )
        entry_suppressed = bool(detector_requests_entry and affirmative_main_mismatch)
        entry_authorized = bool(detector_requests_entry and not entry_suppressed)

        if affirmative_main_mismatch and (self.suppression_active or entry_suppressed):
            if not self.suppression_active:
                self.suppression_active = True
                self.suppression_episodes += 1
                evidence_action = SuppressionEvidenceAction.RESET
            else:
                evidence_action = SuppressionEvidenceAction.FREEZE
        else:
            self.suppression_active = False
            evidence_action = SuppressionEvidenceAction.NONE

        filtered_sm, filtered_sa, filtered_ma = self._filtered
        return {
            "d_sm": float(d_sm),
            "d_sa": float(d_sa),
            "d_ma": float(d_ma),
            "filtered_d_sm": float(filtered_sm),
            "filtered_d_sa": float(filtered_sa),
            "filtered_d_ma": float(filtered_ma),
            "state": self.state.name,
            "state_code": int(self.state),
            "raw_state": raw_state.name,
            "raw_state_code": int(raw_state),
            "candidate_state": candidate.name,
            "candidate_state_code": int(candidate),
            "candidate_run": self.candidate_run,
            "state_persistence": self.state_persistence,
            "state_switches": self.state_switches,
            "auxiliary_trusted": witness_trusted,
            "detector_requests_entry": bool(detector_requests_entry),
            "entry_authorized": entry_authorized,
            "entry_suppressed": entry_suppressed,
            "suppression_active": self.suppression_active,
            "suppression_episodes": self.suppression_episodes,
            "suppression_evidence_action": evidence_action.name,
            "suppression_evidence_action_code": int(evidence_action),
        }


if __name__ == "__main__":
    sample = np.r_[np.zeros(20), np.ones(20)]
    config = calibrate_cusum([np.zeros(40), np.r_[np.zeros(39), 0.01]])
    features = temporal_residual_features(sample)
    monitored = cusum_monitor(sample, **config)
    assert features["ewma"][-1] > features["ewma"][20]
    assert features["signed_ewma"][-1] > features["signed_ewma"][20]
    assert monitored["alarm"][-1]
    assert np.array_equal(
        persistent_alarm([False, True, True, False, False], 2, 2),
        [False, False, True, True, False],
    )
    assert np.array_equal(
        classify_redesign([False, True, False], [0.0, 2.0, 2.0], 1.0),
        [0, 1, 2],
    )
    gate = SensorReliabilityMonitor(1.0, 0.0, 0.1, 0.5, 2, 2)
    assert not gate.update(2.0)["trusted"]
    assert gate.update(2.0)["sensor_suspect"]
    assert gate.update(0.0)["sensor_suspect"]
    while gate.update(0.0)["sensor_suspect"]:
        pass
    # V2: test auxiliary recovery gate
    gate_v2 = SensorReliabilityMonitor(1.0, 0.0, 0.1, 0.5, 2, 2, aux_recovery_gate=3.0)
    gate_v2.update(2.0)
    gate_v2.update(2.0)  # now active
    assert gate_v2.active
    # Residual is OK but aux_residual is too large -> should NOT recover
    # (CUSUM needs ~34 steps to drain from 3.8 to below 0.5, so 40 is enough
    #  to prove aux blocks recovery even after CUSUM drains)
    for _ in range(40):
        gate_v2.update(0.0, aux_residual=5.0)
    assert gate_v2.active, "V2: should stay active when aux_residual exceeds gate"
    # Now aux_residual is also small -> should recover (CUSUM already drained)
    for _ in range(10):
        gate_v2.update(0.0, aux_residual=1.0)
    assert not gate_v2.active, "V2: should recover when aux_residual is within gate"
    # V3: test DualVirtualSensorArbitrator
    arb = DualVirtualSensorArbitrator(agreement_threshold=3.5, param_mismatch_threshold=7.0)
    # 1. Normal condition: physical trusted -> source=PHYSICAL
    res = arb.update(t=1.5, y_measured=35.0, y_main=35.1, y_aux=35.2, sensor_trusted=True, fallback_speed=35.0)
    assert res["source"] == "PHYSICAL" and res["feedback"] == 35.0
    # 2. Sensor untrusted, models agree -> source=MAIN_VIRTUAL
    res = arb.update(t=2.5, y_measured=50.0, y_main=35.0, y_aux=35.2, sensor_trusted=False, fallback_speed=35.0)
    assert res["source"] == "MAIN_VIRTUAL" and res["feedback"] == 35.0
    # 3. Sensor untrusted, models disagree (load disturbance) -> source=AUX_VIRTUAL
    res = arb.update(t=3.5, y_measured=50.0, y_main=35.0, y_aux=23.0, sensor_trusted=False, fallback_speed=35.0)
    assert res["source"] == "AUX_VIRTUAL" and res["feedback"] == 23.0
    # 4. Parameter mismatch detected when sensor trusted -> aux_param_mismatch=True
    arb2 = DualVirtualSensorArbitrator(agreement_threshold=3.5, param_mismatch_threshold=5.0)
    for _ in range(50):
        arb2.update(t=2.0, y_measured=35.0, y_main=35.0, y_aux=20.0, sensor_trusted=True, fallback_speed=35.0)
    assert arb2.aux_param_mismatch
    # When sensor untrusted and models disagree, but aux has param mismatch -> FALLBACK
    res = arb2.update(t=3.0, y_measured=50.0, y_main=35.0, y_aux=20.0, sensor_trusted=False, fallback_speed=30.0)
    assert res["source"] == "FALLBACK" and res["feedback"] == 30.0
    print("Reliability redesign checks passed (including V2 and V3).")
