"""Comprehensive closed-loop evaluation of V2 Auxiliary-Recovery LSTM-MPC.

Compares:
- Part 1: C1 (V1 bug-fixed reliability-aware MPC) vs C2 (V2 auxiliary-recovery MPC)
          on canonical seeds [12026, 12027, 12028, 12029, 12030] across 7 scenarios.
- Part 2: B (Plain LSTM-MPC) vs C1 vs C2 on development/extension seeds [19026..19030].
- Part 3: Virtual sensor quality during substitution (main vs aux vs corrupted).

All MPC and plant parameters match canonical baseline exactly.
Auxiliary model uses [voltage, current] only and never accesses measured speed.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

CLOSED_LOOP_AUX_QUALITY_FILENAME = "v2_closed_loop_aux_virtual_sensor_quality.csv"

from motor_model import DCMotorParams, dc_motor_dynamics
from mpc import LSTMMPC, MPCConfig, PIConfig, PIController
from reliability import SensorReliabilityMonitor, load_lstm_model
from auxiliary_sensor_model import AuxiliarySpeedEstimator, auxiliary_predict_online

# ── Global configuration constants ──────────────────────────────────────────
CANONICAL_SEEDS = [12026, 12027, 12028, 12029, 12030]
EXTENSION_SEEDS = [19026, 19027, 19028, 19029, 19030]

DT = 0.01
CONTROL_STRIDE = 5
CONTROL_DT = CONTROL_STRIDE * DT
DURATION = 6.0
TIME = np.arange(0.0, DURATION + DT / 2, DT)
FULL_SCALE = 65.234375
SPEED_NOISE_STD = 0.25

SCENARIOS = {
    "sensor_bias_5": {
        "title": "+5% sensor bias",
        "event_start": 2.0,
        "event_end": 4.0,
        "type": "sensor",
    },
    "sensor_bias_15": {
        "title": "+15% sensor bias",
        "event_start": 2.0,
        "event_end": 4.0,
        "type": "sensor",
    },
    "sensor_dropout": {
        "title": "Sensor dropout",
        "event_start": 2.0,
        "event_end": 4.0,
        "type": "sensor",
    },
    "sensor_drift": {
        "title": "Sensor drift",
        "event_start": 2.0,
        "event_end": None,
        "type": "sensor",
    },
    "load_disturbance": {
        "title": "Load disturbance",
        "event_start": 3.0,
        "event_end": None,
        "type": "plant",
    },
    "parameter_variation": {
        "title": "Parameter variation",
        "event_start": 3.0,
        "event_end": None,
        "type": "plant",
    },
    "combined_fault_load": {
        "title": "Sensor fault + load disturbance",
        "event_start": 3.0,
        "event_end": None,
        "type": "combined",
    },
}

nominal_params = DCMotorParams()
shifted_params = replace(
    nominal_params,
    resistance=1.20 * nominal_params.resistance,
    inductance=0.85 * nominal_params.inductance,
    back_emf_constant=1.15 * nominal_params.back_emf_constant,
    torque_constant=0.85 * nominal_params.torque_constant,
    inertia=1.20 * nominal_params.inertia,
    viscous_friction=1.20 * nominal_params.viscous_friction,
    coulomb_friction=1.20 * nominal_params.coulomb_friction,
)

def rk4_step(state, voltage, load, params):
    derivative = lambda val: dc_motor_dynamics(0.0, val, voltage, load, params)
    k1 = derivative(state)
    k2 = derivative(state + DT * k1 / 2)
    k3 = derivative(state + DT * k2 / 2)
    k4 = derivative(state + DT * k3)
    return state + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simulate_run(args: tuple) -> dict:
    """Run a single closed-loop simulation."""
    controller_type, scenario_name, seed = args
    torch.set_num_threads(1)

    device = "cpu"
    main_model, main_cfg = load_lstm_model(
        PROJECT / "results/lstm_model_weights.pt",
        PROJECT / "results/configs/lstm_model_config.json",
        device=device,
    )
    aux_cfg = json.loads((PROJECT / "results/configs/v2_auxiliary_config.json").read_text())
    aux_model = AuxiliarySpeedEstimator(**aux_cfg["model"]).to(device)
    aux_model.load_state_dict(torch.load(
        PROJECT / "results/auxiliary_model_weights.pt", map_location=device, weights_only=True
    ))
    aux_model.eval()

    rel_cfg = json.loads((PROJECT / "results/configs/reliability_final_config.json").read_text())
    sensor_cfg = rel_cfg["sensor"]

    main_norm = main_cfg["normalization"]
    aux_norm = aux_cfg["normalization"]

    mpc_config = MPCConfig(
        horizon=20,
        tracking_weight=1.0,
        move_weight=0.5,
        voltage_limits=(0.0, 12.0),
        max_voltage_step=2.0,
        max_iterations=8,
        tolerance=0.05,
        move_blocks=(5, 15),
        warm_start=True,
        control_interval_steps=5,
    )
    pi_config = PIConfig(0.35, 0.8, (0.0, 12.0), 2.0)

    residual_gate = sensor_cfg["instant_threshold"]
    cusum_center = sensor_cfg["center"]
    cusum_allowance = sensor_cfg["allowance"]
    cusum_threshold = sensor_cfg["threshold"]
    enter_count = sensor_cfg["enter_count"]
    exit_count = sensor_cfg["exit_count"]
    aux_recovery_gate = aux_cfg["calibration"]["recommended_recovery_gate"]

    gate = aux_recovery_gate if controller_type == "C2_aux_recovery_MPC" else None
    sensor = SensorReliabilityMonitor(
        residual_gate, cusum_center, cusum_allowance, cusum_threshold,
        enter_count, exit_count, aux_recovery_gate=gate,
    )

    rng = np.random.default_rng(seed)
    base_noise = rng.normal(0.0, SPEED_NOISE_STD, len(TIME))

    state = np.zeros(2)  # [current, speed]
    prev_voltage = 0.0
    history = np.zeros((20, 2), dtype=np.float32)
    aux_history = np.zeros((20, 2), dtype=np.float32)

    pi = PIController(pi_config)
    mpc = LSTMMPC(main_model, main_norm, 20, mpc_config)

    ref = np.full(len(TIME), 35.0)

    # Arrays for metrics
    true_speed_arr = np.zeros(len(TIME))
    meas_speed_arr = np.zeros(len(TIME))
    virtual_speed_arr = np.zeros(len(TIME))
    aux_speed_arr = np.zeros(len(TIME))
    feedback_arr = np.zeros(len(TIME))
    voltage_arr = np.zeros(len(TIME))
    sub_arr = np.zeros(len(TIME), dtype=bool)
    sensor_state_arr = np.zeros(len(TIME), dtype=bool)
    solve_times = []
    aux_times = []
    prediction_failures = 0
    prediction_failure_messages = set()
    last_prediction = 0.0

    t_run_start = perf_counter()

    for i, t in enumerate(TIME):
        current_val, true_speed = float(state[0]), float(state[1])
        true_speed_arr[i] = true_speed

        # Measured speed with scenario injection
        ym = true_speed + base_noise[i]
        if scenario_name == "sensor_bias_5" and 2.0 <= t < 4.0:
            ym += 0.05 * FULL_SCALE
        elif scenario_name == "sensor_bias_15" and 2.0 <= t < 4.0:
            ym += 0.15 * FULL_SCALE
        elif scenario_name == "sensor_dropout" and 2.0 <= t < 4.0:
            ym = 0.0
        elif scenario_name == "sensor_drift" and t >= 2.0:
            ym += 0.15 * FULL_SCALE * min(1.0, (t - 2.0) / 4.0)
        elif scenario_name == "combined_fault_load" and t >= 3.0:
            ym += 0.05 * FULL_SCALE
        meas_speed_arr[i] = ym

        # Main LSTM 1-step ahead prediction
        if i >= 20:
            prediction_failed = False
            try:
                y_hat = float(mpc.predict(history, np.array([prev_voltage], dtype=np.float32))[0])
                last_prediction = y_hat
            except (ValueError, FloatingPointError, RuntimeError) as error:
                prediction_failed = True
                prediction_failures += 1
                prediction_failure_messages.add(f"{type(error).__name__}:{error}")
                y_hat = last_prediction
        else:
            prediction_failed = False
            y_hat = ym
            if np.isfinite(y_hat):
                last_prediction = y_hat
        virtual_speed_arr[i] = y_hat

        # Auxiliary 1-step ahead prediction
        y_aux = np.nan
        if i >= 20:
            t_aux0 = perf_counter()
            y_aux = auxiliary_predict_online(
                aux_model, aux_history[:, 0], aux_history[:, 1], aux_norm
            )
            aux_times.append(perf_counter() - t_aux0)
        aux_speed_arr[i] = y_aux

        # Sensor Reliability Monitor
        if controller_type == "B_plain_MPC" or i < 20:
            is_sub = False
            is_suspect = False
            y_fb = ym
        else:
            r_main = np.nan if prediction_failed else ym - y_hat
            r_aux = (ym - y_aux) if (controller_type == "C2_aux_recovery_MPC" and np.isfinite(y_aux)) else None
            dec = sensor.update(r_main, aux_residual=r_aux)
            is_sub = dec["substitute"]
            is_suspect = dec["sensor_suspect"]
            y_fb = y_hat if is_sub else ym

        feedback_arr[i] = y_fb
        sub_arr[i] = is_sub
        sensor_state_arr[i] = is_suspect

        # Update histories
        history = np.vstack((history[1:], [prev_voltage, y_fb])).astype(np.float32)
        aux_history = np.vstack((aux_history[1:], [prev_voltage, current_val])).astype(np.float32)

        # Control decision
        voltage = prev_voltage
        if i % CONTROL_STRIDE == 0:
            fb_v = pi.compute_control(ref[i], y_fb, CONTROL_DT)
            if controller_type == "A_PI":
                voltage = fb_v
            else:
                t_solve0 = perf_counter()
                out = mpc.compute_control(
                    history, np.full(mpc_config.horizon, ref[i]), prev_voltage,
                    horizon=mpc_config.horizon, move_blocks=mpc_config.move_blocks,
                    fallback_voltage=fb_v,
                )
                solve_times.append(perf_counter() - t_solve0)
                voltage = out["voltage"]

        voltage_arr[i] = voltage
        history[-1, 0] = voltage
        aux_history[-1, 0] = voltage
        prev_voltage = voltage

        # Plant step (RK4)
        load_val = 0.15 if (scenario_name in ("load_disturbance", "combined_fault_load") and t >= 3.0) else 0.03
        params = shifted_params if (scenario_name == "parameter_variation" and t >= 3.0) else nominal_params
        if i < len(TIME) - 1:
            state = rk4_step(state, voltage, load_val, params)

    total_sim_time = perf_counter() - t_run_start

    # ── Metric computations ───────────────────────────────────────────────────
    error = ref - true_speed_arr
    overall_rmse = float(np.sqrt(np.mean(error ** 2)))
    overall_mae = float(np.mean(np.abs(error)))

    sc_conf = SCENARIOS[scenario_name]
    ev_start = sc_conf["event_start"]
    ev_end = sc_conf["event_end"]

    if ev_start is None:
        ev_mask = np.ones(len(TIME), dtype=bool)
    else:
        ev_mask = TIME >= ev_start
        if ev_end is not None:
            ev_mask &= TIME < ev_end

    fault_window_rmse = float(np.sqrt(np.mean(error[ev_mask] ** 2)))

    # Transitions & Latencies
    active_states = sensor_state_arr.copy()
    recovery_indices = np.flatnonzero(active_states[:-1] & ~active_states[1:]) + 1
    false_recovery_events = int(np.sum(ev_mask[recovery_indices])) if ev_start is not None else 0
    total_recovery_events = int(len(recovery_indices))

    # Detection latency
    detections = np.flatnonzero((TIME >= ev_start) & active_states) if ev_start is not None else []
    detection_latency_s = float(TIME[detections[0]] - ev_start) if len(detections) else np.nan

    # Recovery latency
    if ev_end is not None and len(detections):
        sensor_recovered = np.flatnonzero((TIME >= ev_end) & ~active_states)
        recovery_latency_s = float(TIME[sensor_recovered[0]] - ev_end) if len(sensor_recovered) else np.nan
    else:
        recovery_latency_s = np.nan

    # Substitution
    sub_count = int(np.sum(sub_arr))
    sub_duration_s = float(sub_count * DT)
    sub_fraction = float(sub_count / len(TIME))

    # Control effort
    ctrl_voltages = voltage_arr[::CONTROL_STRIDE]
    ctrl_moves = np.diff(np.r_[0.0, ctrl_voltages])
    control_effort_u2 = float(np.sum(ctrl_voltages ** 2))
    control_variation_du2 = float(np.sum(ctrl_moves ** 2))

    # Virtual sensor quality during substitution
    sub_idx = np.where(sub_arr)[0]
    if len(sub_idx) > 0:
        main_err = np.abs(virtual_speed_arr[sub_idx] - true_speed_arr[sub_idx])
        corr_err = np.abs(meas_speed_arr[sub_idx] - true_speed_arr[sub_idx])
        valid_aux = np.isfinite(aux_speed_arr[sub_idx])
        if np.any(valid_aux):
            aux_err = np.abs(aux_speed_arr[sub_idx][valid_aux] - true_speed_arr[sub_idx][valid_aux])
            aux_rmse = float(np.sqrt(np.mean(aux_err ** 2)))
            aux_mae = float(np.mean(aux_err))
        else:
            aux_rmse, aux_mae = np.nan, np.nan

        quality = {
            "main_virtual_rmse": float(np.sqrt(np.mean(main_err ** 2))),
            "main_virtual_mae": float(np.mean(main_err)),
            "aux_virtual_rmse": aux_rmse,
            "aux_virtual_mae": aux_mae,
            "corrupted_rmse": float(np.sqrt(np.mean(corr_err ** 2))),
            "corrupted_mae": float(np.mean(corr_err)),
            "sub_samples": len(sub_idx),
        }
    else:
        quality = {
            "main_virtual_rmse": np.nan,
            "main_virtual_mae": np.nan,
            "aux_virtual_rmse": np.nan,
            "aux_virtual_mae": np.nan,
            "corrupted_rmse": np.nan,
            "corrupted_mae": np.nan,
            "sub_samples": 0,
        }

    return {
        "controller": controller_type,
        "scenario": scenario_name,
        "seed": seed,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "fault_window_rmse": fault_window_rmse,
        "detection_latency_s": detection_latency_s,
        "recovery_latency_s": recovery_latency_s,
        "recovery_events": total_recovery_events,
        "false_recovery_events": false_recovery_events,
        "sub_duration_s": sub_duration_s,
        "sub_fraction": sub_fraction,
        "control_effort_u2": control_effort_u2,
        "control_variation_du2": control_variation_du2,
        "prediction_failures": prediction_failures,
        "prediction_failure_messages": " | ".join(sorted(prediction_failure_messages)),
        "mean_solve_ms": float(np.mean(solve_times) * 1000) if solve_times else 0.0,
        "p95_solve_ms": float(np.percentile(solve_times, 95) * 1000) if solve_times else 0.0,
        "mean_aux_ms": float(np.mean(aux_times) * 1000) if aux_times else 0.0,
        "sim_time_s": total_sim_time,
        **quality,
    }


def main():
    print("=" * 80)
    print("  V2 AUXILIARY RECOVERY CLOSED-LOOP EVALUATION")
    print("=" * 80)

    workers = min(10, os.cpu_count() or 4)
    print(f"Parallel execution with {workers} worker processes.\n")

    # ── PART 1: V1 vs V2 Comparison on Canonical Seeds ────────────────────────
    print("Part 1: Running V1 (C1_sensor_MPC) vs V2 (C2_aux_recovery_MPC) on 5 canonical seeds...")
    p1_tasks = []
    for sc in SCENARIOS:
        for seed in CANONICAL_SEEDS:
            p1_tasks.append(("C1_sensor_MPC", sc, seed))
            p1_tasks.append(("C2_aux_recovery_MPC", sc, seed))

    print(f"Total Part 1 runs: {len(p1_tasks)}")
    t0 = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        p1_results = list(executor.map(simulate_run, p1_tasks))
    t1 = perf_counter()
    print(f"Part 1 finished in {t1 - t0:.1f}s.\n")

    df_p1 = pd.DataFrame(p1_results)
    out_dir = PROJECT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_p1.to_csv(out_dir / "v2_closed_loop_comparison.csv", index=False)

    # ── PART 2: Development Extension (B vs C1 vs C2) ─────────────────────────────────
    print("Part 2: Running development extension (B vs C1 vs C2) on reused seeds...")
    p2_tasks = []
    for sc in SCENARIOS:
        for seed in EXTENSION_SEEDS:
            p2_tasks.append(("B_plain_MPC", sc, seed))
            p2_tasks.append(("C1_sensor_MPC", sc, seed))
            p2_tasks.append(("C2_aux_recovery_MPC", sc, seed))

    print(f"Total Part 2 runs: {len(p2_tasks)}")
    t0 = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        p2_results = list(executor.map(simulate_run, p2_tasks))
    t1 = perf_counter()
    print(f"Part 2 finished in {t1 - t0:.1f}s.\n")

    df_p2 = pd.DataFrame(p2_results)
    df_p2.to_csv(out_dir / "v2_development_extension_closed_loop.csv", index=False)

    # ── PART 3: Virtual Sensor Quality Dataset ─────────────────────────────────
    # Collect all virtual quality records where substitution was active
    quality_cols = [
        "controller", "scenario", "seed", "sub_samples",
        "main_virtual_rmse", "main_virtual_mae",
        "aux_virtual_rmse", "aux_virtual_mae",
        "corrupted_rmse", "corrupted_mae",
    ]
    df_all = pd.concat([df_p1, df_p2], ignore_index=True)
    df_quality = df_all[df_all["sub_samples"] > 0][quality_cols].copy()
    df_quality.to_csv(out_dir / CLOSED_LOOP_AUX_QUALITY_FILENAME, index=False)
    print(
        f"Saved {len(df_quality)} virtual sensor quality records to "
        f"{CLOSED_LOOP_AUX_QUALITY_FILENAME}\n"
    )

    # ── Summary Tables ────────────────────────────────────────────────────────
    print("=" * 80)
    print("PART 1 SUMMARY: V1 (C1) vs V2 (C2) on Canonical Seeds (Mean across 5 seeds)")
    print("=" * 80)
    summary_cols = ["overall_rmse", "fault_window_rmse", "false_recovery_events",
                    "recovery_latency_s", "sub_fraction", "control_effort_u2"]

    agg_p1 = df_p1.groupby(["scenario", "controller"])[summary_cols].mean().reset_index()
    print(agg_p1.to_string(index=False))

    print("\n" + "=" * 80)
    print("PART 2 SUMMARY: Development/extension evidence (B vs C1 vs C2; mean across 5 seeds)")
    print("=" * 80)
    agg_p2 = df_p2.groupby(["scenario", "controller"])[summary_cols].mean().reset_index()
    print(agg_p2.to_string(index=False))

    # Save summary json
    summary_data = {
        "canonical_comparison": agg_p1.to_dict(orient="records"),
        "development_extension": agg_p2.to_dict(orient="records"),
    }
    (out_dir / "v2_closed_loop_summary.json").write_text(
        json.dumps(summary_data, indent=2), encoding="utf-8"
    )
    print(f"\nAll artifacts successfully saved to {out_dir}")

if __name__ == "__main__":
    main()

