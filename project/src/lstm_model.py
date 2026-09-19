"""Compact LSTM plant model and recursive forecasting helper."""

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Predict next normalized speed from ``[voltage, measured speed]`` history."""

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)
        self.residual = residual

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return a one-step prediction for each batch element."""
        features, _ = self.lstm(sequence)
        prediction = self.output(features[:, -1])
        if self.residual:
            prediction = prediction + sequence[:, -1, 1:2]
        return prediction


@torch.no_grad()
def recursive_forecast(
    model: LSTMForecaster,
    history: torch.Tensor,
    future_voltage: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """Roll the model forward, feeding predicted speed back into each window."""
    window = history.clone()
    predictions = []
    for step in range(future_voltage.shape[1]):
        prediction = model(window)
        predictions.append(prediction[:, 0])
        physical_speed = prediction[:, 0] * target_std[0] + target_mean[0]
        feedback_speed = (physical_speed - input_mean[1]) / input_std[1]
        next_sample = torch.stack((future_voltage[:, step], feedback_speed), dim=1)
        window = torch.cat((window[:, 1:], next_sample[:, None, :]), dim=1)
    return torch.stack(predictions, dim=1)
