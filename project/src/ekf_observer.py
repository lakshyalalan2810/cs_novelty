"""Current-only augmented-state EKF for the repository DC motor.

The online contract is deliberately small: the observer consumes only applied
armature voltage and measured armature current.  Its state is
``[current, omega, load_torque]``.  Load torque is deterministic in the state
transition (``dT_L/dt = 0``); slow load variation is represented only through
the configured process covariance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from motor_model import DCMotorParams


STATE_DIM = 3
DEFAULT_DT = 0.01
CURRENT_MEASUREMENT_JACOBIAN = np.array([[1.0, 0.0, 0.0]], dtype=float)


class EKFNumericalError(RuntimeError):
    """Raised when an EKF run encounters an explicit numerical failure."""


def _state_vector(state: np.ndarray) -> np.ndarray:
    vector = np.asarray(state, dtype=float)
    if vector.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape ({STATE_DIM},)")
    if not np.isfinite(vector).all():
        raise ValueError("state must be finite")
    return vector


def _finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _positive_scalar(name: str, value: float) -> float:
    scalar = _finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def smooth_friction(speed: float, params: DCMotorParams) -> float:
    """Return the repository smooth Coulomb-friction torque."""
    return float(
        params.coulomb_friction
        * np.tanh(float(speed) / params.friction_smoothing_speed)
    )


def smooth_friction_derivative(speed: float, params: DCMotorParams) -> float:
    """Return the exact derivative of the repository friction law."""
    tanh_value = np.tanh(float(speed) / params.friction_smoothing_speed)
    return float(
        params.coulomb_friction
        / params.friction_smoothing_speed
        * (1.0 - tanh_value * tanh_value)
    )


def augmented_dynamics(
    state: np.ndarray,
    voltage: float,
    params: DCMotorParams | None = None,
) -> np.ndarray:
    """Continuous dynamics for ``[current, omega, load_torque]``."""
    state = _state_vector(state)
    voltage = _finite_scalar("voltage", voltage)
    params = params or DCMotorParams()
    current, speed, load_torque = state

    current_dot = (
        voltage
        - params.resistance * current
        - params.back_emf_constant * speed
    ) / params.inductance
    speed_dot = (
        params.torque_constant * current
        - params.viscous_friction * speed
        - load_torque
        - smooth_friction(speed, params)
    ) / params.inertia
    return np.array([current_dot, speed_dot, 0.0], dtype=float)


def continuous_state_jacobian(
    state: np.ndarray,
    voltage: float,
    params: DCMotorParams | None = None,
) -> np.ndarray:
    """Exact continuous state Jacobian ``df/dx`` for the augmented model."""
    state = _state_vector(state)
    _finite_scalar("voltage", voltage)
    params = params or DCMotorParams()
    speed = float(state[1])
    friction_slope = smooth_friction_derivative(speed, params)
    return np.array(
        [
            [
                -params.resistance / params.inductance,
                -params.back_emf_constant / params.inductance,
                0.0,
            ],
            [
                params.torque_constant / params.inertia,
                -(params.viscous_friction + friction_slope) / params.inertia,
                -1.0 / params.inertia,
            ],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def rk4_step(
    state: np.ndarray,
    voltage: float,
    params: DCMotorParams | None = None,
    dt: float = DEFAULT_DT,
) -> np.ndarray:
    """Propagate the augmented state with the frozen explicit 10 ms RK4 map."""
    state = _state_vector(state)
    voltage = _finite_scalar("voltage", voltage)
    dt = _positive_scalar("dt", dt)
    params = params or DCMotorParams()

    k1 = augmented_dynamics(state, voltage, params)
    k2 = augmented_dynamics(state + dt * k1 / 2.0, voltage, params)
    k3 = augmented_dynamics(state + dt * k2 / 2.0, voltage, params)
    k4 = augmented_dynamics(state + dt * k3, voltage, params)
    result = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if not np.isfinite(result).all():
        raise EKFNumericalError("RK4 state propagation produced a nonfinite state")
    return result


def rk4_state_jacobian(
    state: np.ndarray,
    voltage: float,
    params: DCMotorParams | None = None,
    dt: float = DEFAULT_DT,
) -> np.ndarray:
    """Exact tangent Jacobian of :func:`rk4_step` via RK4 tangent propagation."""
    state = _state_vector(state)
    voltage = _finite_scalar("voltage", voltage)
    dt = _positive_scalar("dt", dt)
    params = params or DCMotorParams()
    identity = np.eye(STATE_DIM, dtype=float)

    k1 = augmented_dynamics(state, voltage, params)
    dk1 = continuous_state_jacobian(state, voltage, params)

    x2 = state + dt * k1 / 2.0
    dx2 = identity + dt * dk1 / 2.0
    k2 = augmented_dynamics(x2, voltage, params)
    dk2 = continuous_state_jacobian(x2, voltage, params) @ dx2

    x3 = state + dt * k2 / 2.0
    dx3 = identity + dt * dk2 / 2.0
    k3 = augmented_dynamics(x3, voltage, params)
    dk3 = continuous_state_jacobian(x3, voltage, params) @ dx3

    x4 = state + dt * k3
    dx4 = identity + dt * dk3
    dk4 = continuous_state_jacobian(x4, voltage, params) @ dx4

    jacobian = identity + dt * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4) / 6.0
    if not np.isfinite(jacobian).all():
        raise EKFNumericalError("RK4 tangent propagation produced a nonfinite Jacobian")
    return jacobian


def _validated_covariance(
    name: str,
    covariance: np.ndarray,
    *,
    symmetry_tolerance: float,
    psd_tolerance: float,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (STATE_DIM, STATE_DIM):
        raise ValueError(f"{name} must have shape ({STATE_DIM}, {STATE_DIM})")
    if not np.isfinite(matrix).all():
        raise EKFNumericalError(f"{name} is nonfinite")

    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if asymmetry > symmetry_tolerance:
        raise EKFNumericalError(
            f"{name} is materially asymmetric: max |P-P.T|={asymmetry:.3e} "
            f"> tolerance {symmetry_tolerance:.3e}"
        )

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue < -psd_tolerance:
        raise EKFNumericalError(
            f"{name} is not positive semidefinite: min eigenvalue "
            f"{min_eigenvalue:.3e} < {-psd_tolerance:.3e}"
        )
    return symmetric


@dataclass(frozen=True)
class EKFConfig:
    """Fixed EKF numerical configuration; Q/R/P0 come from external calibration."""

    Q: np.ndarray
    R: float
    P0: np.ndarray
    dt: float = DEFAULT_DT
    params: DCMotorParams = field(default_factory=DCMotorParams)
    symmetry_tolerance: float = 1e-10
    psd_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        dt = _positive_scalar("dt", self.dt)
        measurement_variance = _positive_scalar("R", self.R)
        symmetry_tolerance = _positive_scalar(
            "symmetry_tolerance", self.symmetry_tolerance
        )
        psd_tolerance = _positive_scalar("psd_tolerance", self.psd_tolerance)

        q_matrix = _validated_covariance(
            "Q",
            self.Q,
            symmetry_tolerance=symmetry_tolerance,
            psd_tolerance=psd_tolerance,
        )
        p0_matrix = _validated_covariance(
            "P0",
            self.P0,
            symmetry_tolerance=symmetry_tolerance,
            psd_tolerance=psd_tolerance,
        )
        object.__setattr__(self, "Q", q_matrix.copy())
        object.__setattr__(self, "R", measurement_variance)
        object.__setattr__(self, "P0", p0_matrix.copy())
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "symmetry_tolerance", symmetry_tolerance)
        object.__setattr__(self, "psd_tolerance", psd_tolerance)


@dataclass(frozen=True)
class EKFDiagnostics:
    """Per-step numerical diagnostics returned without any truth information."""

    innovation: float
    innovation_variance: float
    kalman_gain: np.ndarray
    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    updated_covariance: np.ndarray
    min_covariance_eigenvalue: float


class AugmentedStateEKF:
    """Augmented ``[current, omega, load_torque]`` EKF with current measurement."""

    def __init__(self, config: EKFConfig) -> None:
        self.config = config
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._failed = False
        self._failure_reason: str | None = None

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def state(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("EKF must be initialized before reading state")
        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            raise RuntimeError("EKF must be initialized before reading covariance")
        return self._covariance.copy()

    def initialize(self, first_current: float) -> np.ndarray:
        """Initialize causally as ``[first_current, 0 rad/s, 0 N m]``."""
        current = _finite_scalar("first_current", first_current)
        self._state = np.array([current, 0.0, 0.0], dtype=float)
        self._covariance = self.config.P0.copy()
        self._failed = False
        self._failure_reason = None
        return self.state

    def _fail(self, stage: str, message: str) -> None:
        self._failed = True
        self._failure_reason = f"{stage}: {message}"
        raise EKFNumericalError(self._failure_reason)

    def step(
        self,
        voltage: float,
        current_measurement: float,
    ) -> tuple[np.ndarray, EKFDiagnostics]:
        """Run one predict/update cycle using only voltage and measured current."""
        if self._state is None or self._covariance is None:
            raise RuntimeError("EKF must be initialized before step")
        if self._failed:
            raise EKFNumericalError(
                f"EKF is already in a failed state: {self._failure_reason}"
            )

        try:
            voltage = _finite_scalar("voltage", voltage)
            measurement = _finite_scalar("current_measurement", current_measurement)
        except ValueError as exc:
            self._fail("input", str(exc))

        try:
            predicted_state = rk4_step(
                self._state, voltage, self.config.params, self.config.dt
            )
            transition = rk4_state_jacobian(
                self._state, voltage, self.config.params, self.config.dt
            )
            predicted_covariance_raw = (
                transition @ self._covariance @ transition.T + self.config.Q
            )
            predicted_covariance = _validated_covariance(
                "predicted covariance",
                predicted_covariance_raw,
                symmetry_tolerance=self.config.symmetry_tolerance,
                psd_tolerance=self.config.psd_tolerance,
            )
        except (ValueError, EKFNumericalError, np.linalg.LinAlgError) as exc:
            self._fail("prediction", str(exc))

        c_matrix = CURRENT_MEASUREMENT_JACOBIAN
        innovation = float(measurement - predicted_state[0])
        innovation_variance = float(
            (c_matrix @ predicted_covariance @ c_matrix.T).item() + self.config.R
        )
        if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
            self._fail(
                "update",
                f"innovation covariance must be finite and positive, got {innovation_variance}",
            )

        kalman_gain = (predicted_covariance @ c_matrix.T)[:, 0] / innovation_variance
        if not np.isfinite(kalman_gain).all():
            self._fail("update", "Kalman gain is nonfinite")

        updated_state = predicted_state + kalman_gain * innovation
        if not np.isfinite(updated_state).all():
            self._fail("update", "updated state is nonfinite")

        identity = np.eye(STATE_DIM, dtype=float)
        correction = identity - np.outer(kalman_gain, c_matrix[0])
        # Joseph form is required by the preregistration.
        updated_covariance_raw = (
            correction @ predicted_covariance @ correction.T
            + np.outer(kalman_gain, kalman_gain) * self.config.R
        )
        try:
            updated_covariance = _validated_covariance(
                "updated covariance",
                updated_covariance_raw,
                symmetry_tolerance=self.config.symmetry_tolerance,
                psd_tolerance=self.config.psd_tolerance,
            )
        except (EKFNumericalError, np.linalg.LinAlgError) as exc:
            self._fail("update", str(exc))

        if not np.isfinite(innovation):
            self._fail("update", "innovation is nonfinite")

        min_eigenvalue = float(np.min(np.linalg.eigvalsh(updated_covariance)))
        self._state = updated_state
        self._covariance = updated_covariance

        diagnostics = EKFDiagnostics(
            innovation=innovation,
            innovation_variance=innovation_variance,
            kalman_gain=kalman_gain.copy(),
            predicted_state=predicted_state.copy(),
            predicted_covariance=predicted_covariance.copy(),
            updated_covariance=updated_covariance.copy(),
            min_covariance_eigenvalue=min_eigenvalue,
        )
        return self.state, diagnostics
