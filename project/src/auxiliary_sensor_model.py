"""Auxiliary virtual speed estimator using voltage and current only.

This module provides a compact LSTM that estimates motor speed from
[voltage, armature_current] history without any access to the speed
sensor channel.  It is used as an independent cross-check during
sensor-fault recovery — not as a replacement for the main LSTM-MPC
plant model.

Key constraint: the auxiliary model NEVER uses measured speed as input.
Known limitation: it assumes the current sensor itself is healthy.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AuxiliarySpeedEstimator(nn.Module):
    """Predict speed from [voltage, current] history.

    Architecture: single-layer LSTM + linear head.
    No residual connection (input channel 1 is current, not speed).
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1 or output_size < 1:
            raise ValueError("all size/layer arguments must be positive")
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return speed estimate for each batch element.

        Parameters
        ----------
        sequence : Tensor of shape ``(batch, window_length, input_size)``
            Normalized ``[voltage, current]`` history windows.

        Returns
        -------
        Tensor of shape ``(batch, output_size)``
            Predicted normalized speed (one-step ahead).
        """
        features, _ = self.lstm(sequence)
        return self.output(features[:, -1])


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass
class AuxiliaryDataConfig:
    """Track auxiliary model data pipeline settings."""

    window_length: int = 20
    features: tuple[str, ...] = ("voltage", "current")
    target: str = "y_true"
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None
    target_mean: np.ndarray | None = None
    target_std: np.ndarray | None = None


def build_auxiliary_sequences(
    voltage: np.ndarray,
    current: np.ndarray,
    target_speed: np.ndarray,
    window_length: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [voltage, current] input windows and speed targets.

    Parameters
    ----------
    voltage, current : arrays of shape ``(num_steps,)``
    target_speed : array of shape ``(num_steps,)``
        Ground-truth clean speed (``y_true``).
    window_length : int

    Returns
    -------
    X : array of shape ``(count, window_length, 2)``
    y : array of shape ``(count, 1)``
    """
    voltage = np.asarray(voltage)
    current = np.asarray(current)
    target_speed = np.asarray(target_speed)
    if voltage.ndim != 1 or current.ndim != 1 or target_speed.ndim != 1:
        raise ValueError("voltage, current, and target_speed must be 1-D")
    if not (len(voltage) == len(current) == len(target_speed)):
        raise ValueError("voltage, current, and target_speed must have equal lengths")
    if not isinstance(window_length, (int, np.integer)) or window_length < 1:
        raise ValueError("window_length must be a positive integer")
    if window_length >= len(voltage):
        raise ValueError("trajectory is too short for the requested window length")

    features = np.column_stack((voltage, current)).astype(np.float32)
    count = len(features) - window_length
    X = np.array(
        [features[start : start + window_length] for start in range(count)],
        dtype=np.float32,
    )
    y = target_speed[window_length : window_length + count].astype(np.float32)[:, None]
    return X, y


def fit_auxiliary_normalization(
    inputs: np.ndarray, targets: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit feature-wise Z-score statistics on training data only."""
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


def normalize_auxiliary(
    inputs: np.ndarray,
    targets: np.ndarray,
    stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply saved training statistics to inputs and targets."""
    norm_in = ((inputs - stats["input_mean"]) / stats["input_std"]).astype(np.float32)
    norm_tgt = ((targets - stats["target_mean"]) / stats["target_std"]).astype(np.float32)
    return norm_in, norm_tgt


# ---------------------------------------------------------------------------
# Online inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def auxiliary_predict_online(
    model: AuxiliarySpeedEstimator,
    voltage_history: np.ndarray,
    current_history: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> float:
    """Return a single physical-unit speed estimate for one time step.

    Parameters
    ----------
    model : trained AuxiliarySpeedEstimator
    voltage_history : array of shape ``(window_length,)``
    current_history : array of shape ``(window_length,)``
    normalization : dict with ``input_mean``, ``input_std``,
        ``target_mean``, ``target_std``

    Returns
    -------
    float
        Estimated speed in physical units (rad/s).
    """
    input_mean = np.asarray(normalization["input_mean"], dtype=np.float32)
    input_std = np.asarray(normalization["input_std"], dtype=np.float32)
    target_mean = float(normalization["target_mean"][0])
    target_std = float(normalization["target_std"][0])

    features = np.column_stack((voltage_history, current_history)).astype(np.float32)
    normalized = (features - input_mean) / input_std
    device = next(model.parameters()).device
    tensor = torch.from_numpy(normalized[None]).to(device)

    was_training = model.training
    model.eval()
    try:
        prediction = model(tensor)[0, 0].item()
    finally:
        model.train(was_training)

    return prediction * target_std + target_mean


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_auxiliary_model(
    model: AuxiliarySpeedEstimator,
    normalization: dict[str, np.ndarray],
    config: dict,
    weights_path: str | Path,
    config_path: str | Path,
) -> None:
    """Save model weights and full configuration to disk."""
    weights_path = Path(weights_path)
    config_path = Path(config_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), weights_path)

    serializable_norm = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in normalization.items()
    }
    full_config = {**config, "normalization": serializable_norm}
    config_path.write_text(
        json.dumps(full_config, indent=2), encoding="utf-8",
    )


def load_auxiliary_model(
    weights_path: str | Path,
    config_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[AuxiliarySpeedEstimator, dict]:
    """Load trained auxiliary model and its configuration."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_kwargs = config["model"]
    model = AuxiliarySpeedEstimator(**model_kwargs).to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify model forward pass
    model = AuxiliarySpeedEstimator(input_size=2, hidden_size=32)
    dummy = torch.randn(4, 20, 2)
    output = model(dummy)
    assert output.shape == (4, 1), f"unexpected output shape: {output.shape}"

    # Verify sequence building
    v = np.random.randn(100).astype(np.float32)
    c = np.random.randn(100).astype(np.float32)
    s = np.random.randn(100).astype(np.float32)
    X, y = build_auxiliary_sequences(v, c, s, window_length=20)
    assert X.shape == (80, 20, 2), f"unexpected X shape: {X.shape}"
    assert y.shape == (80, 1), f"unexpected y shape: {y.shape}"

    # Verify normalization
    stats = fit_auxiliary_normalization(X, y)
    Xn, yn = normalize_auxiliary(X, y, stats)
    assert np.allclose(Xn.mean(axis=(0, 1)), 0, atol=1e-5)

    # Verify online prediction
    speed = auxiliary_predict_online(model, v[:20], c[:20], stats)
    assert np.isfinite(speed), "non-finite prediction"

    print("Auxiliary sensor model checks passed.")
