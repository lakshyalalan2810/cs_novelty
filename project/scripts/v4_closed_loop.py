"""V4 closed-loop evaluation engine (new module; frozen evaluators untouched).

Runs one closed-loop simulation for any V4 controller with parameterized
faults, references, and robustness corruptions. All protocol scripts
(v4_protocol_*.py) are thin drivers over :func:`run_v4_cell`.

Controllers:
- PI: PI control on measured speed (no detector; anchor reference).
- B: plain LSTM-MPC on measured speed (no detector).
- C3: frozen V3 stack (SensorReliabilityMonitor + DualVirtualSensorArbitrator
  with frozen per-pair calibrations; training seeds 2026-2028 only).
- V4_<variant>_<witness>: ReliabilityMonitorV4 (scripts/v4_detector_configs
  flags) with direct main-LSTM substitution (C1-style, works for all 11
  seeds); witness in {aux, ekf} feeds the AND-rule residual.
- E2/S1/S2/S3: baselines_v4 detectors + frozen-config MPC.

Fault magnitudes are specified in physical units; protocols convert from
sigma units via SIGMA_TO_RAD_S. Timing is recorded per control step.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from auxiliary_sensor_model import (
    AuxiliarySpeedEstimator,
    auxiliary_predict_online,
    load_auxiliary_model,
)
from baselines_v4 import (
    CusumCore,
    EKFResidualDetector,
    FixedThresholdDetector,
    LuenbergerBaseline,
    MedianRatePlausibilityFilter,
    detector_from_cusum_doc,
)
from ekf_observer import AugmentedStateEKF
from motor_model import DCMotorParams, dc_motor_dynamics
from mpc import LSTMMPC, MPCConfig, PIConfig, PIController
from reliability import (
    DualVirtualSensorArbitrator,
    SensorReliabilityMonitor,
    load_lstm_model,
)
from reliability_v4 import ReliabilityMonitorV4, v4_from_sensor_calibration

DT = 0.01
CONTROL_STRIDE = 5
DURATION = 6.0
SPEED_NOISE_STD = 0.25
VOLTAGE_LIMITS = (0.0, 12.0)
SIGMA_TO_RAD_S = SPEED_NOISE_STD

V4_MODELS = PROJECT / "results" / "v4" / "models"
V4_CALIBRATION = PROJECT / "results" / "v4" / "calibration"
BASELINE_CALIBRATION = V4_CALIBRATION / "baselines_v4_calibration.json"
PAIR_MANIFEST = (PROJECT / "results" / "training_seed_robustness"
                 / "model_pair_manifest.json")

# Uniform parameter-mismatch presets (physical parameters only; the
# friction-smoothing speed is numerical and stays nominal).
_MISMATCH_PARAMS = ("resistance", "inductance", "back_emf_constant",
                    "torque_constant", "inertia", "viscous_friction",
                    "coulomb_friction")
PARAM_PRESETS = {
    "nominal": {},
    "frozen_shifted": {"resistance": 1.20, "inductance": 0.85,
                       "back_emf_constant": 1.15, "torque_constant": 0.85,
                       "inertia": 1.20, "viscous_friction": 1.20,
                       "coulomb_friction": 1.20},
    "uniform_p10": {name: 1.10 for name in _MISMATCH_PARAMS},
    "uniform_m10": {name: 0.90 for name in _MISMATCH_PARAMS},
    "uniform_p20": {name: 1.20 for name in _MISMATCH_PARAMS},
    "uniform_m20": {name: 0.80 for name in _MISMATCH_PARAMS},
    "uniform_p30": {name: 1.30 for name in _MISMATCH_PARAMS},
    "uniform_m30": {name: 0.70 for name in _MISMATCH_PARAMS},
}


def fail(message: str) -> None:
    raise SystemExit(f"V4_CLOSED_LOOP FAILED: {message}")


def sigma_to_rad_s(sigma: float) -> float:
    return float(sigma) * SIGMA_TO_RAD_S


def apply_param_preset(params: DCMotorParams, preset: str) -> DCMotorParams:
    from dataclasses import replace
    if preset not in PARAM_PRESETS:
        fail(f"unknown parameter preset: {preset}")
    scales = PARAM_PRESETS[preset]
    return replace(params, **{name: getattr(params, name) * scale
                              for name, scale in scales.items()})


def rk4_step(state: np.ndarray, voltage: float, load: float,
             params: DCMotorParams) -> np.ndarray:
    derivative = lambda val: dc_motor_dynamics(0.0, val, voltage, load, params)
    k1 = derivative(state)
    k2 = derivative(state + DT * k1 / 2)
    k3 = derivative(state + DT * k2 / 2)
    k4 = derivative(state + DT * k3)
    return state + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def reference_profile(kind: str, time: np.ndarray) -> np.ndarray:
    if kind == "nominal":
        return np.full_like(time, 35.0)
    if kind == "step":
        return np.where(time < 1.5, 20.0, 40.0)
    if kind == "changing":
        return np.where(time < 2.0, 20.0, np.where(time < 4.0, 42.0, 30.0))
    if kind == "wide":
        return np.where(time < 1.5, 10.0,
                        np.where(time < 3.5, 90.0, 50.0))
    fail(f"unknown reference profile: {kind}")
    raise AssertionError("unreachable")


@dataclass
class Fault:
    """Fault injection specification (physical units, NaN-safe defaults)."""

    kind: str = "none"  # none|bias|dropout|drift|load|combined
    onset_s: float = 2.0
    end_s: float | None = 4.0  # None = persists to horizon
    magnitude_rad_s: float = 0.0  # bias / drift-total; NaN if n/a
    magnitude_sigma: float = float("nan")
    dropout_duration_s: float = 0.0
    drift_rate_rad_s2: float = 0.0
    load_step_Nm: float = 0.0  # post-step absolute load (combined/load)
    load_baseline_Nm: float = 0.03

    def event_window(self, time: np.ndarray) -> np.ndarray:
        mask = time >= self.onset_s
        if self.end_s is not None:
            mask &= time < self.end_s
        return mask

    def apply_speed(self, measured: float, t: float) -> float:
        if self.kind in ("bias", "combined") and t >= self.onset_s and (
                self.end_s is None or t < self.end_s):
            return measured + self.magnitude_rad_s
        if self.kind == "dropout" and self.onset_s <= t < (
                self.onset_s + self.dropout_duration_s):
            return 0.0
        if self.kind == "drift" and t >= self.onset_s:
            span = (self.end_s - self.onset_s) if self.end_s else 4.0
            return measured + self.drift_rate_rad_s2 * min(t - self.onset_s,
                                                           span)
        return measured

    def load_at(self, t: float) -> float:
        if self.kind in ("load", "combined") and t >= self.onset_s:
            return self.load_step_Nm
        return self.load_baseline_Nm


@dataclass
class RunConfig:
    controller: str
    training_seed: int | None = None  # None for PI (model-free)
    simulation_seed: int = 0
    reference: str = "nominal"
    duration: float = DURATION
    fault: Fault = field(default_factory=Fault)
    current_noise_std: float = 0.0
    current_quantization_a: float | None = None
    sample_delay: int = 0  # measurement delay in samples
    param_preset: str = "nominal"
    control_stride: int = CONTROL_STRIDE
    threshold_scale: float = 1.0  # joint DET-sweep multiplier on gates
    record_traces: bool = False


def frozen_pair_bundle(training_seed: int) -> dict:
    """Load frozen weights/configs/calibrations for seeds 2026-2028."""
    if training_seed not in (2026, 2027, 2028):
        fail(f"C3 frozen bundle exists only for 2026-2028, got {training_seed}")
    manifest = json.loads(PAIR_MANIFEST.read_text())
    pair = manifest["pairs"][f"P{training_seed}"]
    main_model, main_config = load_lstm_model(
        PROJECT / pair["main"]["weights_path"],
        PROJECT / pair["main"]["config_path"], device="cpu")
    aux_model, aux_config = load_auxiliary_model(
        PROJECT / pair["auxiliary"]["weights_path"],
        PROJECT / pair["auxiliary"]["config_path"], device="cpu")
    sensor_cal = json.loads(
        (PROJECT / pair["sensor_calibration"]["path"]).read_text())
    pair_config = json.loads((PROJECT / pair["pair_config"]["path"]).read_text())
    return {"main_model": main_model, "main_config": main_config,
            "aux_model": aux_model, "aux_config": aux_config,
            "sensor_cal": sensor_cal, "pair_config": pair_config}


def v4_pair_bundle(training_seed: int) -> dict:
    """Load V4-trained weights/configs/calibration for any 2026-2036 seed."""
    main_model, main_config = load_lstm_model(
        V4_MODELS / f"main_seed_{training_seed}.pt",
        V4_MODELS / f"main_seed_{training_seed}_config.json", device="cpu")
    aux_model, aux_config = load_auxiliary_model(
        V4_MODELS / f"aux_seed_{training_seed}.pt",
        V4_MODELS / f"aux_seed_{training_seed}_config.json", device="cpu")
    sensor_cal = json.loads(
        (V4_CALIBRATION / f"sensor_seed{training_seed}_v4_calibration.json")
        .read_text())
    return {"main_model": main_model, "main_config": main_config,
            "aux_model": aux_model, "aux_config": aux_config,
            "sensor_cal": sensor_cal}


def frozen_ekf_config():
    sys.path.insert(0, str(PROJECT / "scripts"))
    import evaluate_ekf_closed_loop as frozen_ekf_eval
    _, config = frozen_ekf_eval.load_frozen_ekf_config()
    return config


def mpc_stack(main_model, main_config) -> tuple[LSTMMPC, MPCConfig,
                                                PIController]:
    config = MPCConfig(horizon=20, tracking_weight=1.0, move_weight=0.5,
                       voltage_limits=VOLTAGE_LIMITS, max_voltage_step=2.0,
                       max_iterations=8, tolerance=0.05,
                       move_blocks=(5, 15), warm_start=True,
                       control_interval_steps=CONTROL_STRIDE)
    return (LSTMMPC(main_model, main_config["normalization"], 20, config),
            config, PIController(PIConfig(0.35, 0.8, VOLTAGE_LIMITS, 2.0)))


def parse_controller(name: str) -> tuple[str, str | None, str | None]:
    """Split V4_<variant>_<witness>; others return (name, None, None)."""
    if name.startswith("V4_"):
        parts = name.split("_")
        if len(parts) < 3 or parts[-1] not in ("aux", "ekf"):
            fail(f"malformed V4 controller name: {name} "
                 "(expected V4_<variant>_<aux|ekf>)")
        return ("V4", "_".join(parts[1:-1]), parts[-1])
    return (name, None, None)


def load_baseline_calibration() -> dict:
    if not BASELINE_CALIBRATION.is_file():
        fail(f"baseline calibration not found: {BASELINE_CALIBRATION} "
             "(run scripts/v4_calibrate_baselines.py)")
    return json.loads(BASELINE_CALIBRATION.read_text())


class ControllerStack:
    """Per-run controller bundle: models, observers, detectors, MPC/PI."""

    def __init__(self, config: RunConfig) -> None:
        torch.set_num_threads(1)
        self.config = config
        self.kind, self.variant, self.witness = parse_controller(
            config.controller)
        self.main_model = self.main_norm = None
        self.aux_model = self.aux_norm = None
        self.mpc = self.mpc_config = self.pi = None
        self.sensor = self.arbitrator = None
        self.v4_monitor = None
        self.witness_ekf = None
        self.baseline = None
        self.harness_ekf = None
        if self.kind in ("B", "V4", "E2", "S1", "S2", "S3"):
            if config.training_seed is None:
                fail(f"{config.controller} requires a training seed")
            bundle = v4_pair_bundle(config.training_seed)
            self.main_model = bundle["main_model"]
            self.main_norm = bundle["main_config"]["normalization"]
            self.aux_model = bundle["aux_model"]
            self.aux_norm = bundle["aux_config"]["normalization"]
            self.v4_sensor_cal = bundle["sensor_cal"]
            self.mpc, self.mpc_config, self.pi = mpc_stack(
                self.main_model, bundle["main_config"])
        elif self.kind == "PI":
            self.pi = PIController(PIConfig(0.35, 0.8, VOLTAGE_LIMITS, 2.0))
        elif self.kind == "C3":
            if config.training_seed is None:
                fail("C3 requires a training seed (2026-2028)")
            bundle = frozen_pair_bundle(config.training_seed)
            self.main_model = bundle["main_model"]
            self.main_norm = bundle["main_config"]["normalization"]
            self.aux_model = bundle["aux_model"]
            self.aux_norm = bundle["aux_config"]["normalization"]
            self.mpc, self.mpc_config, self.pi = mpc_stack(
                self.main_model, bundle["main_config"])
            values = bundle["sensor_cal"]["computed_values"]
            arb = bundle["pair_config"]["arbitrator"]
            self.sensor = SensorReliabilityMonitor(
                residual_gate=float(values["instant_threshold"]),
                center=float(values["center"]),
                allowance=float(values["allowance"]),
                threshold=float(values["threshold"]),
                enter_count=int(values.get("enter_count", 3)),
                exit_count=int(values.get("exit_count", 5)),
                aux_recovery_gate=float(arb["aux_recovery_gate"]))
            self.arbitrator = DualVirtualSensorArbitrator(**{
                key: (tuple(value) if key == "aux_speed_bounds" else value)
                for key, value in arb.items() if key != "aux_recovery_gate"})
        else:
            fail(f"unknown controller: {config.controller}")

        if self.kind == "V4":
            sys.path.insert(0, str(PROJECT / "scripts"))
            import v4_detector_configs as variants
            base = v4_from_sensor_calibration(
                self.v4_sensor_cal,
                aux_recovery_gate=float(
                    self.v4_sensor_cal["computed_values"]
                    ["aux_recovery_gate"]))
            self.v4_monitor = variants.resolve_variant(base, self.variant)
            witness_gate_key = ("witness_aux_gate" if self.witness == "aux"
                                else "witness_ekf_gate")
            if self.v4_monitor.witness_residual_gate is not None:
                # Replace the placeholder gate with the calibrated witness
                # gate from the pair's V4 calibration.
                self.v4_monitor.witness_residual_gate = float(
                    self.v4_sensor_cal["computed_values"][witness_gate_key])
            if self.witness == "ekf":
                self.witness_ekf = AugmentedStateEKF(frozen_ekf_config())
        elif self.kind in ("E2", "S1", "S2", "S3"):
            calibration = load_baseline_calibration()
            if self.kind == "E2":
                self.baseline = EKFResidualDetector(
                    frozen_ekf_config(),
                    detector_from_cusum_doc(calibration["e2_ekf_cusum"]))
            elif self.kind == "S1":
                self.harness_ekf = AugmentedStateEKF(frozen_ekf_config())
                gate = calibration["s1_fixed_gate"]
                self.baseline = FixedThresholdDetector(
                    gate=float(gate["gate"]),
                    enter_count=int(gate.get("enter_count", 3)),
                    exit_count=int(gate.get("exit_count", 5)))
            elif self.kind == "S2":
                s2 = calibration["s2_plausibility"]
                self.baseline = MedianRatePlausibilityFilter(
                    window=int(s2["window"]),
                    median_gate=float(s2["median_gate"]),
                    rate_limit=float(s2["rate_limit"]))
            else:
                self.baseline = LuenbergerBaseline(
                    detector_from_cusum_doc(
                        calibration["s3_luenberger_cusum"]))
        self._apply_threshold_scale(config.threshold_scale)

    def _apply_threshold_scale(self, scale: float) -> None:
        """Jointly scale every detector gate/threshold (DET sweep)."""
        if scale <= 0:
            fail(f"threshold_scale must be positive, got {scale}")
        if scale == 1.0:
            return
        if self.sensor is not None:
            self.sensor.residual_gate *= scale
            self.sensor.threshold *= scale
        if self.v4_monitor is not None:
            self.v4_monitor.residual_gate *= scale
            self.v4_monitor.threshold *= scale
            if self.v4_monitor.witness_residual_gate is not None:
                self.v4_monitor.witness_residual_gate *= scale
        if self.baseline is not None:
            if isinstance(self.baseline, FixedThresholdDetector):
                self.baseline.gate *= scale
            elif isinstance(self.baseline, MedianRatePlausibilityFilter):
                self.baseline.median_gate *= scale
                self.baseline.rate_limit *= scale
            else:
                self.baseline.detector.residual_gate *= scale
                self.baseline.detector.threshold *= scale


def run_v4_cell(config: RunConfig) -> tuple[dict, list[dict], dict | None]:
    """Run one closed-loop cell; return (row, events, traces-or-None)."""
    stack = ControllerStack(config)
    fault = config.fault
    grid = np.arange(0.0, config.duration, DT, dtype=float)
    grid = grid[grid < config.duration]
    time = np.concatenate((grid, np.array([config.duration], dtype=float)))
    steps = len(time)
    rng = np.random.default_rng(config.simulation_seed)
    speed_noise = rng.normal(0.0, SPEED_NOISE_STD, steps)
    current_noise = (rng.normal(0.0, config.current_noise_std, steps)
                     if config.current_noise_std > 0 else np.zeros(steps))
    params = apply_param_preset(DCMotorParams(), config.param_preset)
    reference = reference_profile(config.reference, time)
    delay = config.sample_delay

    state = np.zeros(2)
    prev_voltage = 0.0
    history = np.zeros((20, 2), dtype=np.float32)
    aux_history = np.zeros((20, 2), dtype=np.float32)
    last_main = 0.0
    last_trusted_speed = 0.0

    raw_ym = np.zeros(steps)
    raw_current = np.zeros(steps)
    # Per-sample arrays (filled in the loop).
    true_speed = np.zeros(steps)
    meas_speed = np.zeros(steps)
    feedback = np.zeros(steps)
    voltage_arr = np.zeros(steps)
    estimate_arr = np.full(steps, np.nan)
    sub_arr = np.zeros(steps, dtype=bool)
    suspect_arr = np.zeros(steps, dtype=bool)
    solve_ms: list[float] = []
    detector_ms: list[float] = []
    observer_ms: list[float] = []
    events: list[dict] = []
    opt_failures = main_failures = aux_failures = observer_failures = 0
    nonfinite_events = voltage_violations = slew_violations = 0
    prev_suspect = False
    prev_voltage_applied = 0.0

    def delayed(series: np.ndarray, i: int) -> float:
        return float(series[max(0, i - delay)])

    for i in range(steps):
        t = float(time[i])
        true_speed[i] = state[1]
        ym_raw = state[1] + speed_noise[i]
        ym_raw = fault.apply_speed(ym_raw, t)
        meas_speed[i] = ym_raw
        current_raw = state[0] + current_noise[i]
        if config.current_quantization_a is not None:
            step_q = config.current_quantization_a
            current_raw = round(current_raw / step_q) * step_q
        raw_current[i] = current_raw
        raw_ym[i] = ym_raw
        ym = delayed(raw_ym, i)
        current_val = delayed(raw_current, i)

        # --- Estimates ---
        y_hat = ym
        if i >= 20 and stack.main_model is not None:
            started = perf_counter()
            try:
                y_hat = float(stack.mpc.predict(
                    history, np.array([prev_voltage],
                                      dtype=np.float32))[0])
                last_main = y_hat
            except (ValueError, FloatingPointError, RuntimeError):
                main_failures += 1
                y_hat = last_main
        elif np.isfinite(ym):
            last_main = ym
        y_aux = np.nan
        needs_aux = stack.kind in ("C3",) or (
            stack.kind == "V4" and stack.witness == "aux")
        if i >= 20 and needs_aux:
            started = perf_counter()
            try:
                y_aux = auxiliary_predict_online(
                    stack.aux_model, aux_history[:, 0], aux_history[:, 1],
                    stack.aux_norm)
            except (ValueError, FloatingPointError, RuntimeError):
                aux_failures += 1
            observer_ms.append(1000 * (perf_counter() - started))

        # --- Detection / feedback selection ---
        is_sub = is_suspect = False
        estimate = np.nan
        detector_started = perf_counter()
        if stack.kind in ("PI", "B") or i < 20:
            y_fb = ym
        elif stack.kind == "C3":
            r_main = ym - y_hat
            r_aux = (ym - y_aux) if np.isfinite(y_aux) else None
            dec = stack.sensor.update(r_main, aux_residual=r_aux)
            is_sub = dec["substitute"]
            is_suspect = dec["sensor_suspect"]
            arb = stack.arbitrator.update(
                t=t, y_measured=ym, y_main=y_hat, y_aux=y_aux,
                sensor_trusted=(not is_sub),
                fallback_speed=last_trusted_speed)
            y_fb = arb["feedback"]
        elif stack.kind == "V4":
            r_main = ym - y_hat
            r_aux = (ym - y_aux) if np.isfinite(y_aux) else None
            if stack.witness == "aux":
                witness_est = y_aux
            else:
                observer_started = perf_counter()
                try:
                    if i == 0:
                        witness_state = stack.witness_ekf.initialize(
                            current_val)
                    else:
                        witness_state, _ = stack.witness_ekf.step(
                            prev_voltage_applied, current_val)
                    witness_est = float(witness_state[1])
                except RuntimeError:
                    observer_failures += 1
                    witness_est = np.nan
                observer_ms.append(1000 * (perf_counter()
                                           - observer_started))
            witness_res = ((ym - witness_est)
                           if np.isfinite(witness_est) else None)
            dec = stack.v4_monitor.update(
                r_main, aux_residual=r_aux, witness_residual=witness_res)
            is_sub = dec["substitute"]
            is_suspect = dec["sensor_suspect"]
            estimate = y_hat
            y_fb = y_hat if is_sub else ym
        elif stack.kind == "E2":
            try:
                dec = stack.baseline.update(prev_voltage_applied,
                                            current_val, ym)
            except RuntimeError:
                observer_failures += 1
                raise
            is_sub = dec["substitute"]
            is_suspect = dec["sensor_suspect"]
            estimate = dec["estimate"]
            y_fb = dec["estimate"] if is_sub else ym
        elif stack.kind == "S1":
            observer_started = perf_counter()
            try:
                if i == 0:
                    ekf_state = stack.harness_ekf.initialize(current_val)
                else:
                    ekf_state, _ = stack.harness_ekf.step(
                        prev_voltage_applied, current_val)
                omega = float(ekf_state[1])
            except RuntimeError:
                observer_failures += 1
                raise
            observer_ms.append(1000 * (perf_counter() - observer_started))
            dec = stack.baseline.update(ym, omega)
            is_sub = dec["substitute"]
            is_suspect = dec["sensor_suspect"]
            estimate = omega
            y_fb = omega if is_sub else ym
        elif stack.kind == "S2":
            dec = stack.baseline.update(ym)
            is_sub = dec["substitute"]
            is_suspect = dec["sensor_suspect"]
            estimate = dec["estimate"]
            y_fb = dec["estimate"] if is_sub else ym
        elif stack.kind == "S3":
            try:
                dec = stack.baseline.update(prev_voltage_applied,
                                            current_val, ym)
            except (RuntimeError, ValueError):
                observer_failures += 1
                raise
            is_sub = dec["substitute"]
            is_suspect = dec["sensor_suspect"]
            estimate = dec["estimate"]
            y_fb = dec["estimate"] if is_sub else ym
        detector_ms.append(1000 * (perf_counter() - detector_started))
        if not is_sub and np.isfinite(ym):
            last_trusted_speed = float(np.clip(ym, 0.0, 100.0))
        if is_suspect and not prev_suspect:
            events.append({"time_s": t, "event": "entry",
                           "residual": float(ym - y_hat)
                           if np.isfinite(y_hat) else float("nan")})
        elif prev_suspect and not is_suspect:
            events.append({"time_s": t, "event": "recovery"})
        prev_suspect = is_suspect
        feedback[i] = y_fb
        estimate_arr[i] = estimate
        sub_arr[i] = is_sub
        suspect_arr[i] = is_suspect
        nonfinite_events += int(
            not np.isfinite([ym, y_fb]).all()
            or (stack.main_model is not None
                and not np.isfinite(y_hat)))

        # --- History + control ---
        history = np.vstack((history[1:], [prev_voltage, y_fb])).astype(
            np.float32)
        aux_history = np.vstack(
            (aux_history[1:], [prev_voltage, current_val])).astype(np.float32)
        applied = prev_voltage
        if i % config.control_stride == 0:
            solve_started = perf_counter()
            if stack.kind == "PI":
                applied = stack.pi.compute_control(
                    reference[i], y_fb, config.control_stride * DT)
            else:
                fallback_v = stack.pi.compute_control(
                    reference[i], y_fb, config.control_stride * DT)
                future_ref = np.interp(
                    t + DT * np.arange(1, stack.mpc_config.horizon + 1),
                    time, reference)
                out = stack.mpc.compute_control(
                    history, future_ref, prev_voltage,
                    horizon=stack.mpc_config.horizon,
                    move_blocks=stack.mpc_config.move_blocks,
                    fallback_voltage=fallback_v)
                solve_ms.append(1000 * (perf_counter() - solve_started))
                if out["success"] and np.isfinite(out["voltage"]):
                    applied = float(out["voltage"])
                else:
                    opt_failures += 1
                    applied = float(np.clip(fallback_v, *VOLTAGE_LIMITS))
        voltage_arr[i] = applied
        history[-1, 0] = applied
        aux_history[-1, 0] = applied
        voltage_violations += int(
            not (VOLTAGE_LIMITS[0] - 1e-9 <= applied <= VOLTAGE_LIMITS[1]
                 + 1e-9))
        if i > 0:
            slew_violations += int(abs(applied - prev_voltage) > 2.0 + 1e-9)
        prev_voltage = applied
        prev_voltage_applied = applied

        # --- Plant step ---
        if i < steps - 1:
            state = rk4_step(state, applied, fault.load_at(t), params)

    return _summarize(config, time, reference, true_speed, feedback,
                      estimate_arr, sub_arr, suspect_arr, events, solve_ms,
                      detector_ms, observer_ms, opt_failures, main_failures,
                      aux_failures, observer_failures, nonfinite_events,
                      voltage_violations, slew_violations,
                      voltage_arr if config.record_traces else None,
                      meas_speed if config.record_traces else None)


def _summarize(config: RunConfig, time: np.ndarray, reference: np.ndarray,
               true_speed: np.ndarray, feedback: np.ndarray,
               estimate: np.ndarray, sub_arr: np.ndarray,
               suspect_arr: np.ndarray, events: list[dict],
               solve_ms: list[float], detector_ms: list[float],
               observer_ms: list[float], opt_failures: int,
               main_failures: int, aux_failures: int, observer_failures: int,
               nonfinite_events: int, voltage_violations: int,
               slew_violations: int, voltage_trace, meas_trace):
    fault = config.fault
    error = reference - true_speed
    overall_rmse = float(np.sqrt(np.mean(error ** 2)))
    window = fault.event_window(time)
    fault_rmse = (float(np.sqrt(np.mean(error[window] ** 2)))
                  if window.any() else float("nan"))
    entries = [event for event in events if event["event"] == "entry"]
    recoveries = [event for event in events if event["event"] == "recovery"]
    post_entries = [event for event in entries
                    if event["time_s"] >= fault.onset_s]
    pre_entries = [event for event in entries
                   if event["time_s"] < fault.onset_s]
    detected = len(post_entries) > 0
    latency = (float(post_entries[0]["time_s"] - fault.onset_s)
               if detected else float("nan"))
    release_at = fault.end_s if fault.end_s is not None else fault.onset_s
    post_release = [event for event in recoveries
                    if event["time_s"] >= release_at]
    recovery_latency = (float(post_release[0]["time_s"] - release_at)
                        if post_release else float("nan"))
    event_sub = float(np.mean(sub_arr[window])) if window.any() else 0.0
    row = {
        "controller": config.controller,
        "training_seed": (config.training_seed
                           if config.training_seed is not None else -1),
        "simulation_seed": config.simulation_seed,
        "reference": config.reference,
        "duration_s": config.duration,
        "fault_kind": fault.kind,
        "fault_onset_s": fault.onset_s,
        "fault_end_s": (fault.end_s if fault.end_s is not None
                        else float("nan")),
        "fault_magnitude_rad_s": fault.magnitude_rad_s,
        "fault_magnitude_sigma": fault.magnitude_sigma,
        "dropout_duration_s": fault.dropout_duration_s,
        "drift_rate_rad_s2": fault.drift_rate_rad_s2,
        "load_step_Nm": fault.load_step_Nm,
        "current_noise_std": config.current_noise_std,
        "current_quantization_a": (config.current_quantization_a
                                   if config.current_quantization_a is not None
                                   else float("nan")),
        "sample_delay": config.sample_delay,
        "param_preset": config.param_preset,
        "control_stride": config.control_stride,
        "threshold_scale": config.threshold_scale,
        "run_complete": True,
        "overall_rmse": overall_rmse,
        "fault_window_rmse": fault_rmse,
        "sensor_fault_detected": detected,
        "detection_latency_s": latency,
        "reliability_entries": len(entries),
        "post_event_reliability_entries": len(post_entries),
        "pre_event_reliability_entries": len(pre_entries),
        "first_entry_time_s": (float(entries[0]["time_s"]) if entries
                               else float("nan")),
        "first_post_event_entry_time_s": (
            float(post_entries[0]["time_s"]) if detected else float("nan")),
        "recovery_events": len(recoveries),
        "recovery_latency_s": recovery_latency,
        "sub_fraction": float(np.mean(sub_arr)),
        "event_sub_fraction": event_sub,
        "mean_solve_ms": float(np.mean(solve_ms)) if solve_ms else 0.0,
        "p95_solve_ms": (float(np.percentile(solve_ms, 95)) if solve_ms
                         else 0.0),
        "max_solve_ms": float(np.max(solve_ms)) if solve_ms else 0.0,
        "mean_detector_ms": (float(np.mean(detector_ms)) if detector_ms
                             else 0.0),
        "mean_observer_ms": (float(np.mean(observer_ms)) if observer_ms
                             else 0.0),
        "optimizer_failures": opt_failures,
        "main_prediction_failures": main_failures,
        "aux_prediction_failures": aux_failures,
        "observer_failures": observer_failures,
        "nonfinite_events": nonfinite_events,
        "voltage_violations": voltage_violations,
        "slew_violations": slew_violations,
    }
    traces = None
    if config.record_traces:
        traces = {"time_s": time, "reference": reference,
                  "true_speed": true_speed, "feedback": feedback,
                  "estimate": estimate, "voltage": voltage_trace,
                  "measured": meas_trace, "substitute": sub_arr,
                  "suspect": suspect_arr}
    return row, events, traces


def run_batch(cells: list[RunConfig], workers: int = 1):
    """Run cells sequentially (workers=1) or in a process pool."""
    if workers <= 1:
        return [run_v4_cell(cell) for cell in cells]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_v4_cell, cells))
