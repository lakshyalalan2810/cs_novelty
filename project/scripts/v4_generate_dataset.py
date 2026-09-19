"""V4 dataset generator: closed-loop and startup-rich healthy trajectories.

Writes data/processed/v4/dc_motor_v4_dataset.npz plus a manifest JSON. The
frozen dataset (30 open-loop trajectories, no startup-from-rest coverage) is
never modified or read here except for shape conventions.

Trajectory mixture (default 120 trajectories, all healthy, no sensor faults):
- startup_from_rest: PI-driven step from (0, 0) to 30-90 rad/s (saturated
  12 V starts, the frozen blind spot).
- closed_loop_mpc: MPC-driven tracking (frozen main model as bootstrap)
  with random initial state, 10-90 rad/s references, load steps.
- closed_loop_pi: PI-driven tracking, same randomization.
- open_multisine / open_random_step: frozen-style open-loop excitation.

Split is by whole trajectory before windowing (frozen split_runs helper);
normalization is fit on training windows only. Current measurement effects
are opt-in (default off = frozen noiseless semantics) and stored alongside
the true current as current_measured.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from motor_model import DCMotorParams, simulate_motor
from data_utils import (
    build_training_sequences,
    fit_normalization,
    make_multisine_voltage,
    make_random_step_signal,
    save_processed_dataset,
    split_runs,
    vary_motor_params,
)
from mpc import LSTMMPC, MPCConfig, PIConfig, PIController
from reliability import load_lstm_model

DEFAULT_SEED = 62026  # preregistered in V4_PREREGISTRATION.md
DEFAULT_OUT = PROJECT / "data" / "processed" / "v4" / "dc_motor_v4_dataset.npz"
FROZEN_WEIGHTS = PROJECT / "results" / "lstm_model_weights.pt"
FROZEN_MODEL_CONFIG = PROJECT / "results" / "configs" / "lstm_model_config.json"

WINDOW_LENGTH = 20
CONTROL_STRIDE = 5
SPEED_NOISE_STD = 0.25
VOLTAGE_LIMITS = (0.0, 12.0)

PARAM_NAMES = ("resistance", "inductance", "back_emf_constant",
               "torque_constant", "inertia", "viscous_friction",
               "coulomb_friction", "friction_smoothing_speed")

MIXTURE = (("startup_from_rest", 0.25), ("closed_loop_mpc", 0.25),
           ("closed_loop_pi", 0.20), ("open_multisine", 0.15),
           ("open_random_step", 0.15))


def fail(message: str) -> None:
    raise SystemExit(f"V4_DATASET FAILED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rk4_step(state: np.ndarray, voltage: float, load: float,
             params: DCMotorParams, dt: float) -> np.ndarray:
    from motor_model import dc_motor_dynamics
    derivative = lambda val: dc_motor_dynamics(0.0, val, voltage, load, params)
    k1 = derivative(state)
    k2 = derivative(state + dt * k1 / 2)
    k3 = derivative(state + dt * k2 / 2)
    k4 = derivative(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def frozen_mpc_stack():
    """Load the frozen main model + MPC/PI controllers (read-only bootstrap)."""
    import torch
    torch.set_num_threads(1)
    model, config = load_lstm_model(FROZEN_WEIGHTS, FROZEN_MODEL_CONFIG,
                                    device="cpu")
    mpc_config = MPCConfig(horizon=20, tracking_weight=1.0, move_weight=0.5,
                           voltage_limits=VOLTAGE_LIMITS, max_voltage_step=2.0,
                           max_iterations=8, tolerance=0.05,
                           move_blocks=(5, 15), warm_start=True,
                           control_interval_steps=CONTROL_STRIDE)
    mpc = LSTMMPC(model, config["normalization"], WINDOW_LENGTH, mpc_config)
    pi = PIController(PIConfig(0.35, 0.8, VOLTAGE_LIMITS, 2.0))
    return mpc, mpc_config, pi


def reference_profile(rng: np.random.Generator, time: np.ndarray,
                      ref_min: float, ref_max: float,
                      startup: bool) -> np.ndarray:
    if startup:
        return np.full_like(time, rng.uniform(30.0, ref_max))
    style = rng.choice(["constant", "step", "changing"])
    if style == "constant":
        return np.full_like(time, rng.uniform(ref_min, ref_max))
    if style == "step":
        first = rng.uniform(ref_min, ref_min + 0.5 * (ref_max - ref_min))
        second = rng.uniform(ref_min + 0.5 * (ref_max - ref_min), ref_max)
        step_time = rng.uniform(0.25 * time[-1], 0.6 * time[-1])
        return np.where(time < step_time, first, second)
    first = rng.uniform(ref_min, ref_min + 0.4 * (ref_max - ref_min))
    second = rng.uniform(ref_min + 0.4 * (ref_max - ref_min), ref_max)
    third = rng.uniform(ref_min, ref_max)
    t1 = rng.uniform(0.2 * time[-1], 0.4 * time[-1])
    t2 = rng.uniform(0.55 * time[-1], 0.8 * time[-1])
    return np.where(time < t1, first, np.where(time < t2, second, third))


def load_profile(rng: np.random.Generator, time: np.ndarray):
    base = float(rng.uniform(0.0, 0.10))
    if rng.random() < 0.5:
        return np.full_like(time, base)
    step_time = float(rng.uniform(0.2 * time[-1], 0.7 * time[-1]))
    stepped = float(rng.uniform(0.05, 0.25))
    return np.where(time < step_time, base, stepped)


def closed_loop_rollout(kind: str, rng: np.random.Generator, time: np.ndarray,
                        dt: float, params: DCMotorParams,
                        initial_state: tuple[float, float],
                        reference: np.ndarray, load: np.ndarray,
                        stack) -> dict:
    """Healthy closed-loop rollout with PI or frozen-bootstrap MPC."""
    mpc, mpc_config, pi = stack
    duration_steps = len(time)
    noise = rng.normal(0.0, SPEED_NOISE_STD, duration_steps)
    state = np.array(initial_state, dtype=float)
    history = np.zeros((WINDOW_LENGTH, 2), dtype=np.float32)
    prev_voltage = 0.0
    if mpc is not None:
        mpc.reset()
    pi.reset()
    voltage = np.zeros(duration_steps)
    current = np.zeros(duration_steps)
    y_true = np.zeros(duration_steps)
    fallbacks = 0
    for i in range(duration_steps):
        t = time[i]
        y_true[i] = state[1]
        current[i] = state[0]
        y_meas = state[1] + noise[i]
        history = np.vstack((history[1:], [prev_voltage, y_meas])).astype(
            np.float32)
        applied = prev_voltage
        if i % CONTROL_STRIDE == 0:
            if kind == "closed_loop_pi" or kind == "startup_from_rest":
                applied = pi.compute_control(reference[i], y_meas,
                                             CONTROL_STRIDE * dt)
            else:
                future_ref = np.interp(
                    t + dt * np.arange(1, mpc_config.horizon + 1),
                    time, reference)
                out = mpc.compute_control(
                    history, future_ref, prev_voltage,
                    horizon=mpc_config.horizon,
                    move_blocks=mpc_config.move_blocks,
                    fallback_voltage=pi.compute_control(
                        reference[i], y_meas, CONTROL_STRIDE * dt))
                if out["success"] and np.isfinite(out["voltage"]):
                    applied = float(out["voltage"])
                else:
                    fallbacks += 1
                    applied = float(np.clip(
                        pi.compute_control(reference[i], y_meas,
                                           CONTROL_STRIDE * dt),
                        *VOLTAGE_LIMITS))
        voltage[i] = applied
        history[-1, 0] = applied
        prev_voltage = applied
        if i < duration_steps - 1:
            state = rk4_step(state, applied, load[i], params, dt)
    if not (np.isfinite(voltage).all() and np.isfinite(y_true).all()
            and np.isfinite(current).all()):
        fail("nonfinite closed-loop rollout")
    return {"voltage": voltage, "current": current, "y_true": y_true,
            "noise": noise, "fallbacks": fallbacks}


def open_loop_rollout(kind: str, rng: np.random.Generator,
                      params: DCMotorParams,
                      initial_state: tuple[float, float], load: np.ndarray,
                      duration: float, dt: float) -> dict:
    """Frozen-style open-loop rollout with random initial state."""
    if kind == "open_multisine":
        voltage_fn = make_multisine_voltage(rng,
                                            voltage_limits=VOLTAGE_LIMITS)
    else:
        levels = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        voltage_fn = make_random_step_signal(rng, duration, 1.0, levels)
    time, states = simulate_motor(
        voltage_fn, lambda t: float(np.interp(t, np.arange(len(load)) * dt,
                                              load)),
        simulation_time=duration, timestep=dt, params=params,
        initial_state=initial_state)
    current, speed = states.T
    voltage = np.array([voltage_fn(t) for t in time])
    return {"time": time, "voltage": voltage, "current": current,
            "y_true": speed,
            "noise": rng.normal(0.0, SPEED_NOISE_STD, len(time))}


def mixture_assignment(n: int, seed_seq: np.random.SeedSequence,
                       exclude: tuple[str, ...] = ()) -> list[str]:
    mixture = [(kind, fraction) for kind, fraction in MIXTURE
               if kind not in exclude]
    if not mixture:
        fail("mixture exclusion removed every trajectory kind")
    total = sum(fraction for _, fraction in mixture)
    kinds: list[str] = []
    for kind, fraction in mixture:
        kinds.extend([kind] * int(round(n * fraction / total)))
    while len(kinds) < n:
        kinds.append(mixture[0][0])
    kinds = kinds[:n]
    rng = np.random.default_rng(seed_seq)
    order = rng.permutation(n)
    return [kinds[i] for i in order]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trajectories", type=int, default=120)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--timestep", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny configuration for tests (6 trajectories)")
    parser.add_argument("--current-noise-std", type=float, default=0.0)
    parser.add_argument("--current-quantization-a", type=float, default=None)
    parser.add_argument("--param-variation", type=float, default=0.3)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--ref-min", type=float, default=10.0)
    parser.add_argument("--ref-max", type=float, default=90.0)
    parser.add_argument("--exclude-kind", action="append", default=[],
                        help="drop a mixture kind (repeatable); recorded")
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    if args.smoke:
        args.n_trajectories = 6
        args.duration = 2.0
    n = args.n_trajectories
    dt = args.timestep
    if n < 3 or args.duration <= 0 or dt <= 0:
        fail("invalid size/duration/timestep")
    if args.param_variation < 0 or args.param_variation >= 1:
        fail("param variation must be in [0, 1)")
    if args.current_noise_std < 0:
        fail("current noise std must be nonnegative")
    if (args.current_quantization_a is not None
            and args.current_quantization_a <= 0):
        fail("current quantization step must be positive")

    grid = np.arange(0.0, args.duration, dt, dtype=float)
    grid = grid[grid < args.duration]
    time = np.concatenate((grid, np.array([args.duration], dtype=float)))
    steps = len(time)

    # One SeedSequence spawn stream: n trajectory streams plus mixture,
    # split, and current-noise streams. Deterministic in (seed, n).
    children = np.random.SeedSequence(args.seed).spawn(n + 3)
    traj_children, mixture_child, split_child, noise_child = (
        children[:n], children[n], children[n + 1], children[n + 2])
    split_rng = np.random.default_rng(split_child)
    kinds = mixture_assignment(n, mixture_child,
                               tuple(args.exclude_kind))
    needs_mpc = any(kind == "closed_loop_mpc" for kind in kinds)
    stack = frozen_mpc_stack() if needs_mpc else (None, None, PIController(
        PIConfig(0.35, 0.8, VOLTAGE_LIMITS, 2.0)))

    nominal = DCMotorParams()
    voltage = np.zeros((n, steps), dtype=np.float32)
    current = np.zeros((n, steps), dtype=np.float32)
    y_true = np.zeros((n, steps), dtype=np.float32)
    y_measured = np.zeros((n, steps), dtype=np.float32)
    load = np.zeros((n, steps), dtype=np.float32)
    reference = np.zeros((n, steps), dtype=np.float32)
    initial_state = np.zeros((n, 2), dtype=np.float32)
    parameter_values = np.zeros((n, len(PARAM_NAMES)), dtype=np.float32)
    total_fallbacks = 0
    for run_id in range(n):
        rng = np.random.default_rng(traj_children[run_id])
        kind = kinds[run_id]
        params = vary_motor_params(nominal, rng, args.param_variation)
        parameter_values[run_id] = [getattr(params, name)
                                    for name in PARAM_NAMES]
        if kind == "startup_from_rest":
            initial = (0.0, 0.0)
        else:
            initial = (float(rng.uniform(-1.0, 1.0)),
                       float(rng.uniform(0.0, 15.0)))
        initial_state[run_id] = initial
        load_i = load_profile(rng, time)
        load[run_id] = load_i
        if kind in ("startup_from_rest", "closed_loop_mpc", "closed_loop_pi"):
            ref = reference_profile(rng, time, args.ref_min, args.ref_max,
                                    startup=(kind == "startup_from_rest"))
            reference[run_id] = ref
            rollout = closed_loop_rollout(kind, rng, time, dt, params,
                                          initial, ref, load_i, stack)
            total_fallbacks += rollout["fallbacks"]
        else:
            rollout = open_loop_rollout(kind, rng, params, initial, load_i,
                                        args.duration, dt)
            if len(rollout["time"]) != steps or not np.allclose(
                    rollout["time"], time):
                fail("open-loop time grid mismatch")
        voltage[run_id] = rollout["voltage"]
        current[run_id] = rollout["current"]
        y_true[run_id] = rollout["y_true"]
        y_measured[run_id] = rollout["y_true"] + rollout["noise"]
    for name, array in (("voltage", voltage), ("current", current),
                        ("y_true", y_true), ("y_measured", y_measured)):
        if not np.isfinite(array).all():
            fail(f"nonfinite {name} array")

    current_measured = current.astype(np.float64)
    if args.current_noise_std > 0:
        noise_rng = np.random.default_rng(noise_child)
        current_measured = current_measured + noise_rng.normal(
            0.0, args.current_noise_std, current_measured.shape)
    if args.current_quantization_a is not None:
        step = args.current_quantization_a
        current_measured = np.round(current_measured / step) * step
    current_measured = current_measured.astype(np.float32)

    trajectories = [{"run_id": np.full(steps, run_id, dtype=np.int32),
                     "time": time,
                     "voltage": voltage[run_id],
                     "current": current[run_id],
                     "current_measured": current_measured[run_id],
                     "y_true": y_true[run_id],
                     "y_measured": y_measured[run_id],
                     "load_torque": load[run_id],
                     "reference": reference[run_id]}
                    for run_id in range(n)]
    fractions = (args.train_frac, args.val_frac,
                 1.0 - args.train_frac - args.val_frac)
    split = split_runs(trajectories, split_rng, fractions)

    def windows(subset):
        inputs, targets, run_ids = [], [], []
        for trajectory in subset:
            x, y, r = build_training_sequences(
                trajectory, window_length=WINDOW_LENGTH, horizon=1)
            inputs.append(x)
            targets.append(y)
            run_ids.append(r)
        return (np.concatenate(inputs), np.concatenate(targets),
                np.concatenate(run_ids))

    split_ids = {name: np.array(sorted(trajectory["run_id"][0]
                                       for trajectory in subset), dtype=np.int32)
                 for name, subset in split.items()}
    x_train, y_train, train_runs = windows(split["train"])
    x_val, y_val, val_runs = windows(split["validation"])
    x_test, y_test, test_runs = windows(split["test"])
    normalization = fit_normalization(x_train, y_train)

    out = Path(args.out)
    save_processed_dataset(
        out,
        time=time.astype(np.float32),
        run_ids=np.arange(n, dtype=np.int32),
        excitation_type=np.array(kinds),
        parameter_names=np.array(PARAM_NAMES),
        parameter_values=parameter_values,
        initial_state=initial_state,
        feature_names=np.array(["voltage", "y_measured"]),
        target_name=np.array("y_true"),
        window_length=np.int64(WINDOW_LENGTH),
        horizon=np.int64(1),
        timestep=np.float64(dt),
        duration=np.float64(args.duration),
        seed=np.int64(args.seed),
        speed_noise_std=np.float64(SPEED_NOISE_STD),
        current_noise_std=np.float64(args.current_noise_std),
        current_quantization_a=np.float64(
            args.current_quantization_a
            if args.current_quantization_a is not None else np.nan),
        param_variation=np.float64(args.param_variation),
        voltage_limits=np.array(VOLTAGE_LIMITS),
        voltage=voltage, current=current, current_measured=current_measured,
        y_true=y_true, y_measured=y_measured,
        load_torque=load, reference=reference,
        normalization_input_mean=normalization["input_mean"],
        normalization_input_std=normalization["input_std"],
        normalization_target_mean=normalization["target_mean"],
        normalization_target_std=normalization["target_std"],
        X_train=x_train, y_train=y_train,
        window_run_ids_train=train_runs,
        split_run_ids_train=split_ids["train"],
        X_validation=x_val, y_validation=y_val,
        window_run_ids_validation=val_runs,
        split_run_ids_validation=split_ids["validation"],
        X_test=x_test, y_test=y_test,
        window_run_ids_test=test_runs,
        split_run_ids_test=split_ids["test"])
    manifest = {
        "generator": "scripts/v4_generate_dataset.py",
        "seed": args.seed,
        "n_trajectories": n,
        "mixture": {kind: kinds.count(kind) for kind, _ in MIXTURE},
        "excluded_kinds": list(args.exclude_kind),
        "split_run_ids": {name: ids.tolist()
                          for name, ids in split_ids.items()},
        "mpc_fallback_controls": total_fallbacks,
        "current_noise_std": args.current_noise_std,
        "current_quantization_a": args.current_quantization_a,
        "param_variation": args.param_variation,
        "normalization_fit": "training windows only",
        "sha256": sha256(out),
    }
    manifest_path = out.with_name(out.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} ({n} trajectories, "
          f"train/val/test {len(split_ids['train'])}/"
          f"{len(split_ids['validation'])}/{len(split_ids['test'])}, "
          f"MPC fallbacks {total_fallbacks})")
    return out


if __name__ == "__main__":
    main()
