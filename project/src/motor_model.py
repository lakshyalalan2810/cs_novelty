"""Nonlinear permanent-magnet DC motor model."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

InputFunction = Callable[[float], float]


@dataclass
class DCMotorParams:
    """Editable nominal motor parameters in SI units."""

    resistance: float = 2.0  # R [ohm]
    inductance: float = 0.5  # L [H]
    back_emf_constant: float = 0.1  # Kb [V s/rad]
    torque_constant: float = 0.1  # Kt [N m/A]
    inertia: float = 0.01  # J [kg m^2]
    viscous_friction: float = 0.002  # B [N m s/rad]
    coulomb_friction: float = 0.02  # Coulomb friction torque [N m]
    friction_smoothing_speed: float = 0.1  # tanh transition speed [rad/s]

    def __post_init__(self) -> None:
        values = {
            "resistance": self.resistance,
            "inductance": self.inductance,
            "back_emf_constant": self.back_emf_constant,
            "torque_constant": self.torque_constant,
            "inertia": self.inertia,
            "viscous_friction": self.viscous_friction,
            "coulomb_friction": self.coulomb_friction,
            "friction_smoothing_speed": self.friction_smoothing_speed,
        }
        if not np.isfinite(list(values.values())).all():
            raise ValueError("motor parameters must be finite")
        if self.resistance < 0:
            raise ValueError("resistance must be nonnegative")
        if self.inductance <= 0 or self.inertia <= 0 or self.friction_smoothing_speed <= 0:
            raise ValueError(
                "inductance, inertia, and friction_smoothing_speed must be positive"
            )


def dc_motor_dynamics(
    _time: float,
    state: np.ndarray,
    voltage: float,
    load_torque: float,
    params: DCMotorParams,
) -> np.ndarray:
    """Return derivatives for state ``[armature current, angular speed]``."""
    current, speed = state
    friction_torque = params.coulomb_friction * np.tanh(
        speed / params.friction_smoothing_speed
    )
    current_dot = (
        voltage
        - params.resistance * current
        - params.back_emf_constant * speed
    ) / params.inductance
    speed_dot = (
        params.torque_constant * current
        - params.viscous_friction * speed
        - load_torque
        - friction_torque
    ) / params.inertia
    return np.array([current_dot, speed_dot])


def simulate_motor(
    voltage: InputFunction,
    load_torque: InputFunction | None = None,
    *,
    simulation_time: float = 5.0,
    timestep: float = 0.01,
    params: DCMotorParams | None = None,
    initial_state: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the motor and return ``(time, [current, speed])``."""
    if (
        not np.isfinite(simulation_time)
        or not np.isfinite(timestep)
        or simulation_time <= 0
        or timestep <= 0
    ):
        raise ValueError("simulation_time and timestep must be positive")

    params = params or DCMotorParams()
    load_torque = load_torque or (lambda _time: 0.0)
    regular_time = np.arange(0.0, simulation_time, timestep, dtype=float)
    regular_time = regular_time[regular_time < simulation_time]
    time = np.concatenate((regular_time, np.array([simulation_time], dtype=float)))

    solution = solve_ivp(
        lambda t, y: dc_motor_dynamics(
            t, y, voltage(t), load_torque(t), params
        ),
        (0.0, simulation_time),
        initial_state,
        t_eval=time,
        max_step=timestep,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return time, solution.y.T
