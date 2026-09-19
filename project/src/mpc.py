"""Constrained LSTM-MPC, PI fallback, and continuous model-quality monitoring."""

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, minimize


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 10
    tracking_weight: float = 1.0
    move_weight: float = 0.05
    voltage_limits: tuple[float, float] = (0.0, 12.0)
    max_voltage_step: float = 2.0
    max_iterations: int = 25
    tolerance: float = 5e-2
    control_horizon: int | None = None
    move_blocks: tuple[int, ...] | None = None
    warm_start: bool = True
    control_interval_steps: int = 1

    def __post_init__(self) -> None:
        low, high = self.voltage_limits
        max_control_horizon = (self.horizon - 1) // self.control_interval_steps + 1 if self.control_interval_steps > 0 else 0
        control_horizon = max_control_horizon if self.control_horizon is None else self.control_horizon
        blocks = self.move_blocks
        if (
            self.horizon < 1
            or not 1 <= self.control_interval_steps <= self.horizon
            or not 1 <= control_horizon <= max_control_horizon
            or (blocks is not None and self.control_horizon is not None)
            or (blocks is not None and (not blocks or any(block < 1 for block in blocks) or sum(blocks) != self.horizon))
            or (
                blocks is not None
                and self.control_interval_steps > 1
                and len(blocks) > 1
                and (blocks[0] != self.control_interval_steps or any(sum(blocks[:index]) % self.control_interval_steps for index in range(1, len(blocks))))
            )
            or self.tracking_weight <= 0
            or self.move_weight < 0
            or low >= high
            or self.max_voltage_step <= 0
            or self.max_iterations < 1
            or self.tolerance <= 0
        ):
            raise ValueError("invalid MPC configuration")


@dataclass(frozen=True)
class PIConfig:
    proportional_gain: float = 0.35
    integral_gain: float = 0.8
    voltage_limits: tuple[float, float] = (0.0, 12.0)
    max_voltage_step: float = 2.0


class PIController:
    """Rate-limited PI controller with simple back-calculation anti-windup."""

    def __init__(self, config: PIConfig = PIConfig()) -> None:
        self.config = config
        self.integral = 0.0
        self.previous_voltage = 0.0

    def reset(self, voltage: float = 0.0) -> None:
        self.integral = 0.0
        self.previous_voltage = float(np.clip(voltage, *self.config.voltage_limits))

    def compute_control(self, reference: float, speed: float, timestep: float) -> float:
        if (
            not np.isfinite(timestep)
            or timestep <= 0
            or not np.isfinite([reference, speed]).all()
        ):
            return self.previous_voltage
        error = reference - speed
        self.integral += self.config.integral_gain * error * timestep
        raw = self.config.proportional_gain * error + self.integral
        low = max(self.config.voltage_limits[0], self.previous_voltage - self.config.max_voltage_step)
        high = min(self.config.voltage_limits[1], self.previous_voltage + self.config.max_voltage_step)
        voltage = float(np.clip(raw, low, high))
        self.integral += 0.5 * (voltage - raw)
        self.previous_voltage = voltage
        return voltage


class LSTMMPC:
    """Receding-horizon MPC using recursive differentiable LSTM predictions."""

    def __init__(self, model: torch.nn.Module, normalization: dict, window_length: int, config: MPCConfig = MPCConfig()) -> None:
        if window_length < 1:
            raise ValueError("window_length must be positive")
        self.model = model.eval()
        self.window_length = window_length
        self.config = config
        self.device = next(model.parameters()).device
        self.model.requires_grad_(False)
        self.input_mean = torch.as_tensor(normalization["input_mean"], dtype=torch.float32, device=self.device)
        self.input_std = torch.as_tensor(normalization["input_std"], dtype=torch.float32, device=self.device)
        self.target_mean = torch.as_tensor(normalization["target_mean"][0], dtype=torch.float32, device=self.device)
        self.target_std = torch.as_tensor(normalization["target_std"][0], dtype=torch.float32, device=self.device)
        self._warm_start: np.ndarray | None = None
        self._blocking_cache: dict[tuple[int, ...], tuple[np.ndarray, torch.Tensor]] = {}

    def reset(self) -> None:
        self._warm_start = None

    def _normalized_history(self, history: np.ndarray) -> torch.Tensor:
        return ((torch.as_tensor(history, dtype=torch.float32, device=self.device) - self.input_mean) / self.input_std)[None]

    def _rollout(self, window: torch.Tensor, future_voltage: torch.Tensor) -> torch.Tensor:
        predictions = []
        for voltage in future_voltage:
            normalized_voltage = (voltage - self.input_mean[0]) / self.input_std[0]
            controlled_last = torch.stack((normalized_voltage, window[0, -1, 1]))
            controlled_window = torch.cat((window[:, :-1], controlled_last[None, None]), dim=1)
            normalized_speed = self.model(controlled_window)[0, 0]
            speed = normalized_speed * self.target_std + self.target_mean
            predictions.append(speed)
            next_sample = torch.stack((normalized_voltage, (speed - self.input_mean[1]) / self.input_std[1]))
            window = torch.cat((controlled_window[:, 1:], next_sample[None, None]), dim=1)
        return torch.stack(predictions)

    def _blocking(self, blocks: tuple[int, ...]) -> tuple[np.ndarray, torch.Tensor]:
        if blocks not in self._blocking_cache:
            difference = np.eye(len(blocks))
            difference[1:, :-1] -= np.eye(len(blocks) - 1)
            indices = torch.repeat_interleave(
                torch.arange(len(blocks), device=self.device),
                torch.as_tensor(blocks, device=self.device),
            )
            self._blocking_cache[blocks] = difference, indices
        return self._blocking_cache[blocks]

    @torch.no_grad()
    def predict(self, history: np.ndarray, future_voltage: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32)
        future_voltage = np.asarray(future_voltage, dtype=np.float32)
        if history.shape != (self.window_length, 2) or future_voltage.ndim != 1:
            raise ValueError("history or voltage sequence has the wrong shape")
        prediction = self._rollout(self._normalized_history(history), torch.from_numpy(future_voltage).to(self.device))
        values = prediction.cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("LSTM produced NaN or Inf")
        return values

    def compute_control(
        self,
        history: np.ndarray,
        reference: float | np.ndarray,
        previous_voltage: float,
        *,
        horizon: int | None = None,
        control_horizon: int | None = None,
        move_blocks: tuple[int, ...] | None = None,
        warm_start: bool | None = None,
        move_weight: float | None = None,
        max_voltage_step: float | None = None,
        fallback_voltage: float | None = None,
        control_interval_steps: int | None = None,
    ) -> dict:
        """Optimize on the model grid and return one control-interval command."""
        started = perf_counter()
        horizon = self.config.horizon if horizon is None else int(horizon)
        control_interval_steps = self.config.control_interval_steps if control_interval_steps is None else int(control_interval_steps)
        if move_blocks is None and control_horizon is None:
            move_blocks = self.config.move_blocks
            control_horizon = self.config.control_horizon
        if move_blocks is None:
            control_horizon = (horizon - 1) // max(control_interval_steps, 1) + 1 if control_horizon is None else int(control_horizon)
            blocks = (control_interval_steps,) * (control_horizon - 1) + (horizon - control_interval_steps * (control_horizon - 1),)
        else:
            blocks = tuple(int(block) for block in move_blocks)
            control_horizon = len(blocks)
        warm_start = self.config.warm_start if warm_start is None else bool(warm_start)
        move_weight = self.config.move_weight if move_weight is None else float(move_weight)
        max_voltage_step = self.config.max_voltage_step if max_voltage_step is None else float(max_voltage_step)
        low, high = self.config.voltage_limits
        safe_fallback = previous_voltage if fallback_voltage is None else fallback_voltage
        safe_fallback = float(np.clip(safe_fallback, max(low, previous_voltage - max_voltage_step), min(high, previous_voltage + max_voltage_step)))

        try:
            history = np.asarray(history, dtype=np.float32)
            if (
                history.shape != (self.window_length, 2)
                or not np.isfinite(history).all()
                or horizon < 1
                or not blocks
                or any(block < 1 for block in blocks)
                or sum(blocks) != horizon
                or not 1 <= control_interval_steps <= horizon
                or (
                    control_interval_steps > 1
                    and len(blocks) > 1
                    and (blocks[0] != control_interval_steps or any(sum(blocks[:index]) % control_interval_steps for index in range(1, len(blocks))))
                )
                or move_weight < 0
                or max_voltage_step <= 0
            ):
                raise ValueError("invalid MPC input")
            target = np.asarray(reference, dtype=np.float32).reshape(-1)
            if not len(target) or not np.isfinite(target).all():
                raise ValueError("invalid reference")
            target = np.pad(target, (0, max(0, horizon - len(target))), mode="edge")[:horizon]
            target_tensor = torch.from_numpy(target).to(self.device)
            history_tensor = self._normalized_history(history)
            difference, block_indices = self._blocking(blocks)

            warm_start_used = warm_start and self._warm_start is not None
            if warm_start_used:
                shift = min(control_interval_steps, len(self._warm_start))
                shifted = np.r_[self._warm_start[shift:], np.repeat(self._warm_start[-1], shift)]
                if len(shifted) < horizon:
                    shifted = np.pad(shifted, (0, horizon - len(shifted)), mode="edge")
                shifted = shifted[:horizon]
                initial = shifted[np.r_[0, np.cumsum(blocks)[:-1]]]
            else:
                initial = np.full(len(blocks), previous_voltage, dtype=float)
            initial[0] = np.clip(initial[0], previous_voltage - max_voltage_step, previous_voltage + max_voltage_step)
            for index in range(1, len(initial)):
                initial[index] = np.clip(initial[index], initial[index - 1] - max_voltage_step, initial[index - 1] + max_voltage_step)
            initial = np.clip(initial, low, high)

            cache: dict[str, object] = {"evaluations": 0}

            def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
                if "x" not in cache or not np.array_equal(values, cache["x"]):
                    voltage = torch.tensor(values, dtype=torch.float32, device=self.device, requires_grad=True)
                    future_voltage = voltage[block_indices]
                    prediction = self._rollout(history_tensor, future_voltage)
                    moves = torch.diff(torch.cat((voltage.new_tensor([previous_voltage]), future_voltage)))
                    cost = self.config.tracking_weight * torch.sum((prediction - target_tensor) ** 2) + move_weight * torch.sum(moves**2)
                    gradient = torch.autograd.grad(cost, voltage)[0]
                    cache.update(
                        x=values.copy(),
                        value=float(cost.detach()),
                        gradient=gradient.detach().cpu().numpy().astype(float),
                        prediction=prediction.detach().cpu().numpy(),
                        evaluations=int(cache["evaluations"]) + 1,
                    )
                return cache["value"], cache["gradient"]

            lower = np.full(len(blocks), -max_voltage_step)
            upper = np.full(len(blocks), max_voltage_step)
            lower[0], upper[0] = previous_voltage - max_voltage_step, previous_voltage + max_voltage_step
            result = minimize(
                lambda values: objective(values)[0],
                initial,
                jac=lambda values: objective(values)[1],
                method="SLSQP",
                bounds=Bounds(low, high),
                constraints=LinearConstraint(difference, lower, upper),
                options={"maxiter": self.config.max_iterations, "ftol": self.config.tolerance, "disp": False},
            )
            if not result.success or not np.isfinite(result.x).all():
                raise RuntimeError(result.message)
            decisions = np.asarray(result.x, dtype=float)
            sequence = decisions[np.repeat(np.arange(len(blocks)), blocks)]
            if "x" in cache and np.array_equal(result.x, cache["x"]):
                prediction = np.asarray(cache["prediction"])
            else:
                prediction = self.predict(history, sequence)
            self._warm_start = sequence if warm_start else None
            return {
                "voltage": float(sequence[0]),
                "sequence": sequence,
                "decision_sequence": decisions,
                "move_blocks": blocks,
                "control_horizon": len(blocks),
                "control_interval_steps": control_interval_steps,
                "prediction": prediction,
                "success": True,
                "used_fallback": False,
                "warm_start_used": warm_start_used,
                "message": str(result.message),
                "iterations": int(result.nit),
                "objective_evaluations": int(cache["evaluations"]),
                "compute_ms": 1000 * (perf_counter() - started),
            }
        except (ValueError, FloatingPointError, RuntimeError) as error:
            self._warm_start = None
            return {
                "voltage": safe_fallback,
                "sequence": np.full(horizon, safe_fallback),
                "decision_sequence": np.full(len(blocks), safe_fallback),
                "move_blocks": blocks,
                "control_horizon": len(blocks),
                "control_interval_steps": control_interval_steps,
                "prediction": np.full(horizon, np.nan),
                "success": False,
                "used_fallback": True,
                "warm_start_used": False,
                "message": str(error),
                "iterations": 0,
                "objective_evaluations": 0,
                "compute_ms": 1000 * (perf_counter() - started),
            }


class RollingPredictionQuality:
    """Causal rolling RMS over every due point of prior multi-step forecasts."""

    def __init__(self, scale: float, rolling_window: int = 20, horizon: int = 10) -> None:
        if scale <= 0 or rolling_window < 1 or horizon < 1:
            raise ValueError("scale, window, and horizon must be positive")
        self.scale = float(scale)
        self.errors = deque(maxlen=rolling_window * horizon)
        self.forecasts: list[list] = []
        self.horizon = horizon

    def reset(self) -> None:
        self.errors.clear()
        self.forecasts.clear()

    def add_forecast(self, prediction: np.ndarray) -> None:
        values = np.asarray(prediction, dtype=float)[: self.horizon]
        if len(values) and np.isfinite(values).all():
            self.forecasts.append([values, 0])

    def update(self, observation: float, trusted: bool = True) -> float:
        retained = []
        for prediction, age in self.forecasts:
            if trusted and np.isfinite(observation):
                self.errors.append((float(observation) - prediction[age]) ** 2)
            age += 1
            if age < len(prediction):
                retained.append([prediction, age])
        self.forecasts = retained
        return self.score

    @property
    def score(self) -> float:
        return float(np.sqrt(np.mean(self.errors)) / self.scale) if self.errors else 1.0


def adaptation_region(score: float, medium_threshold: float, high_threshold: float) -> str:
    """Map continuous model quality to the three requested aggressiveness regions."""
    if not np.isfinite(score) or score >= high_threshold:
        return "HIGH"
    return "MEDIUM" if score >= medium_threshold else "LOW"


if __name__ == "__main__":
    class DemoModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, sequence: torch.Tensor) -> torch.Tensor:
            return 0.9 * sequence[:, -1, 1:2] + 0.1 * sequence[:, -1, 0:1] + 0 * self.anchor

    class FlatModel(DemoModel):
        def forward(self, sequence: torch.Tensor) -> torch.Tensor:
            return 0 * sequence[:, -1, :1] + 0 * self.anchor

    controller = LSTMMPC(
        DemoModel(),
        {"input_mean": [0.0, 0.0], "input_std": [1.0, 1.0], "target_mean": [0.0], "target_std": [1.0]},
        3,
        MPCConfig(horizon=4, control_horizon=2, max_iterations=20),
    )
    checked = controller.compute_control(np.zeros((3, 2)), 1.0, 0.0)
    assert checked["success"] and 0.0 <= checked["voltage"] <= 2.0
    assert checked["sequence"].shape == (4,) and np.allclose(checked["sequence"][1:], checked["sequence"][1])
    warmed = controller.compute_control(np.zeros((3, 2)), 1.0, checked["voltage"])
    assert warmed["warm_start_used"]
    shortened = controller.compute_control(np.zeros((3, 2)), 1.0, warmed["voltage"], horizon=2, control_horizon=2)
    regrown = controller.compute_control(np.zeros((3, 2)), 1.0, shortened["voltage"], horizon=4, control_horizon=4)
    assert shortened["success"] and regrown["success"] and regrown["warm_start_used"]
    multirate = LSTMMPC(
        FlatModel(),
        {"input_mean": [0.0, 0.0], "input_std": [1.0, 1.0], "target_mean": [0.0], "target_std": [1.0]},
        3,
        MPCConfig(horizon=20, control_horizon=2, move_weight=0.0, max_iterations=20, control_interval_steps=5),
    )
    multirate._warm_start = np.r_[np.ones(5), np.full(5, 3.0), np.full(10, 5.0)]
    advanced = multirate.compute_control(np.zeros((3, 2)), 0.0, 3.0)
    assert advanced["success"] and advanced["warm_start_used"]
    assert advanced["move_blocks"] == (5, 15) and advanced["control_interval_steps"] == 5
    assert np.allclose(advanced["decision_sequence"], [3.0, 5.0])
    try:
        MPCConfig(horizon=20, move_blocks=(1, 19), control_interval_steps=5)
    except ValueError:
        pass
    else:
        raise AssertionError("sub-control-interval move block accepted")
    failed = controller.compute_control(np.full((3, 2), np.nan), 1.0, checked["voltage"], fallback_voltage=0.5)
    assert failed["used_fallback"] and abs(failed["voltage"] - 0.5) < 1e-9
    monitor = RollingPredictionQuality(1.0, 2, 2)
    monitor.add_forecast([0.0, 0.0])
    assert monitor.update(2.0) > 1.0 and adaptation_region(3.0, 2.0, 4.0) == "MEDIUM"
    print("MPC checks passed.")
