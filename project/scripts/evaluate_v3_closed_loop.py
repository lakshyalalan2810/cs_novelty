"""Comprehensive closed-loop evaluation of V3 Dual-Virtual-Sensor Arbitration LSTM-MPC.

Compares:
- B  = Plain LSTM-MPC
- C1 = Bug-fixed V1 reliability-aware MPC
- C2 = V2 Independent-recovery MPC
- C3 = V3 Dual-virtual-sensor arbitration MPC

Evaluates the three preserved baseline reference scenarios and merges them with
the frozen eight-scenario evidence. Existing final runs are never regenerated.

Runs once on the frozen, unused final holdout seeds [29026..29030].
The older [19026..19030] results are development/extension evidence because V2 used them.
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from motor_model import DCMotorParams, dc_motor_dynamics
from mpc import LSTMMPC, MPCConfig, PIConfig, PIController
from reliability import (
    SensorReliabilityMonitor,
    DualVirtualSensorArbitrator,
    load_lstm_model,
)
from auxiliary_sensor_model import AuxiliarySpeedEstimator, auxiliary_predict_online

FINAL_HOLDOUT_SEEDS = [29026, 29027, 29028, 29029, 29030]
CONTROLLERS = [
    "A_PI",
    "B_plain_MPC",
    "C1_sensor_MPC",
    "C2_aux_recovery_MPC",
    "C3_arbitration_MPC",
]

DT = 0.01
CONTROL_STRIDE = 5
CONTROL_DT = CONTROL_STRIDE * DT
DURATION = 6.0
TIME = np.arange(0.0, DURATION + DT / 2, DT)
FULL_SCALE = 65.234375
SPEED_NOISE_STD = 0.25

SCENARIOS = {
    "nominal_tracking": {
        "title": "Nominal tracking",
        "reference": "constant",
        "event_start": None,
        "event_end": None,
        "type": "reference",
    },
    "step_reference": {
        "title": "Step reference",
        "reference": "step",
        "event_start": None,
        "event_end": None,
        "type": "reference",
    },
    "changing_reference": {
        "title": "Changing reference",
        "reference": "changing",
        "event_start": None,
        "event_end": None,
        "type": "reference",
    },
    "sensor_noise": {
        "title": "Sensor noise",
        "event_start": 2.0,
        "event_end": 4.0,
        "type": "sensor",
    },
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

REFERENCE_SCENARIOS = {"nominal_tracking", "step_reference", "changing_reference"}
FROZEN_SCENARIOS = set(SCENARIOS) - REFERENCE_SCENARIOS

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


def reference_values(name, times=TIME):
    """Exact reference profiles from the validated baseline evaluation."""
    times = np.asarray(times)
    kind = SCENARIOS[name].get("reference", "constant")
    if kind == "step":
        return np.where(times < 1.5, 20.0, 40.0)
    if kind == "changing":
        return np.where(times < 2.0, 20.0, np.where(times < 4.0, 42.0, 30.0))
    return np.full(times.shape, 35.0)


def simulate_v3_run(args: tuple) -> dict:
    """Run a single closed-loop simulation."""
    controller_type, scenario_name, seed, record_traces = args
    torch.set_num_threads(1)

    device = "cpu"
    main_model, main_cfg = load_lstm_model(
        PROJECT / "results/lstm_model_weights.pt",
        PROJECT / "results/configs/lstm_model_config.json",
        device=device,
    )
    aux_cfg = json.loads((PROJECT / "results/configs/v2_auxiliary_config.json").read_text())
    v3_cfg = json.loads((PROJECT / "results/configs/v3_arbitration_config.json").read_text())
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
    aux_recovery_gate = v3_cfg["arbitrator"]["aux_recovery_gate"]

    # In C1, aux_recovery_gate is None (V1 baseline)
    # In C2 and C3, aux_recovery_gate is enforced for recovery
    gate = aux_recovery_gate if controller_type in ("C2_aux_recovery_MPC", "C3_arbitration_MPC") else None
    sensor = SensorReliabilityMonitor(
        residual_gate, cusum_center, cusum_allowance, cusum_threshold,
        enter_count, exit_count, aux_recovery_gate=gate,
    )

    arbitrator = DualVirtualSensorArbitrator(**{
        key: value for key, value in v3_cfg["arbitrator"].items()
        if key != "aux_recovery_gate"
    })

    rng = np.random.default_rng(seed)
    base_noise = rng.normal(0.0, SPEED_NOISE_STD, len(TIME))

    state = np.zeros(2)
    prev_voltage = 0.0
    history = np.zeros((20, 2), dtype=np.float32)
    aux_history = np.zeros((20, 2), dtype=np.float32)

    pi = PIController(pi_config)
    mpc = LSTMMPC(main_model, main_norm, 20, mpc_config)

    ref = reference_values(scenario_name)

    # Logging arrays
    true_speed_arr = np.zeros(len(TIME))
    meas_speed_arr = np.zeros(len(TIME))
    main_speed_arr = np.full(len(TIME), np.nan)
    aux_speed_arr = np.full(len(TIME), np.nan)
    feedback_arr = np.zeros(len(TIME))
    voltage_arr = np.zeros(len(TIME))
    sub_arr = np.zeros(len(TIME), dtype=bool)
    sensor_state_arr = np.zeros(len(TIME), dtype=bool)
    source_arr = np.zeros(len(TIME), dtype=int) # 0: PHYS, 1: MAIN, 2: AUX, 3: FALLBACK

    solve_times = []
    aux_times = []
    reliability_times = []
    arb_times = []
    extension_times = []
    control_times = []
    timing_samples = []
    opt_failures = 0
    main_prediction_failures = 0
    aux_prediction_failures = 0
    prediction_failure_messages = set()
    nonfinite_events = 0
    last_trusted_speed = 0.0
    last_main_prediction = 0.0
    recovery_diagnostics = []

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
        elif scenario_name == "sensor_noise" and 2.0 <= t < 4.0:
            ym += rng.normal(0.0, 2.0)
        meas_speed_arr[i] = ym

        # Main LSTM 1-step prediction
        if i >= 20 and controller_type != "A_PI":
            try:
                y_hat = float(mpc.predict(history, np.array([prev_voltage], dtype=np.float32))[0])
                last_main_prediction = y_hat
            except (ValueError, FloatingPointError, RuntimeError) as error:
                main_prediction_failures += 1
                prediction_failure_messages.add(f"main:{type(error).__name__}:{error}")
                y_hat = last_main_prediction
        else:
            y_hat = ym
            if np.isfinite(y_hat):
                last_main_prediction = y_hat
        main_speed_arr[i] = y_hat

        # Auxiliary 1-step prediction
        y_aux = np.nan
        aux_ms = reliability_ms = arbitration_ms = 0.0
        if i >= 20 and controller_type in ("C2_aux_recovery_MPC", "C3_arbitration_MPC"):
            started = perf_counter()
            try:
                y_aux = auxiliary_predict_online(aux_model, aux_history[:, 0], aux_history[:, 1], aux_norm)
            except (ValueError, FloatingPointError, RuntimeError) as error:
                aux_prediction_failures += 1
                prediction_failure_messages.add(f"aux:{type(error).__name__}:{error}")
            aux_ms = 1000 * (perf_counter() - started)
            if controller_type == "C3_arbitration_MPC":
                aux_times.append(aux_ms)
        aux_speed_arr[i] = y_aux

        # Reliability decision
        if controller_type in ("A_PI", "B_plain_MPC") or i < 20:
            is_sub = False
            is_suspect = False
            y_fb = ym
            source_code = 0
        else:
            r_main = ym - y_hat
            r_aux = (ym - y_aux) if (controller_type in ("C2_aux_recovery_MPC", "C3_arbitration_MPC") and np.isfinite(y_aux)) else None
            started = perf_counter()
            was_active = sensor.active
            prior_healthy_run = sensor.healthy_run
            dec = sensor.update(r_main, aux_residual=r_aux)
            reliability_ms = 1000 * (perf_counter() - started)
            is_sub = dec["substitute"]
            is_suspect = dec["sensor_suspect"]

            if controller_type == "C2_aux_recovery_MPC" and was_active:
                residual_ok = bool(np.isfinite(r_main) and abs(r_main) <= residual_gate)
                cusum_ok = bool(dec["score"] <= cusum_threshold)
                auxiliary_ok = bool(r_aux is not None and np.isfinite(r_aux) and abs(r_aux) <= aux_recovery_gate)
                signal_conditions_ok = residual_ok and cusum_ok and auxiliary_ok
                recovery_diagnostics.append({
                    "scenario": scenario_name,
                    "seed": seed,
                    "step": i,
                    "time_s": float(t),
                    "raw_residual": float(r_main),
                    "residual_gate": residual_gate,
                    "cusum_score": dec["score"],
                    "cusum_threshold": cusum_threshold,
                    "prior_healthy_run": prior_healthy_run,
                    "exit_count": exit_count,
                    "aux_residual": float(r_aux) if r_aux is not None else np.nan,
                    "aux_recovery_gate": aux_recovery_gate,
                    "residual_condition": residual_ok,
                    "cusum_condition": cusum_ok,
                    "persistence_condition": signal_conditions_ok and prior_healthy_run + 1 >= exit_count,
                    "auxiliary_condition": auxiliary_ok,
                    "blocked_by_residual": not residual_ok,
                    "blocked_by_cusum": not cusum_ok,
                    "blocked_by_persistence": signal_conditions_ok and prior_healthy_run + 1 < exit_count,
                    "blocked_only_by_auxiliary": residual_ok and cusum_ok and not auxiliary_ok,
                    "recovered_this_step": not dec["sensor_suspect"],
                })

            if controller_type in ("C1_sensor_MPC", "C2_aux_recovery_MPC"):
                y_fb = y_hat if is_sub else ym
                source_code = 1 if is_sub else 0
            elif controller_type == "C3_arbitration_MPC":
                t_arb0 = perf_counter()
                arb_dec = arbitrator.update(
                    t=t, y_measured=ym, y_main=y_hat, y_aux=y_aux,
                    sensor_trusted=(not is_sub),
                    fallback_speed=last_trusted_speed,
                )
                arbitration_ms = 1000 * (perf_counter() - t_arb0)
                arb_times.append(arbitration_ms)
                y_fb = arb_dec["feedback"]
                source_code = arb_dec["source_code"]

        if not is_sub and np.isfinite(ym):
            last_trusted_speed = float(np.clip(ym, *arbitrator.aux_speed_bounds))
        if controller_type == "C3_arbitration_MPC" and i >= 20:
            reliability_times.append(reliability_ms)
            extension_ms = aux_ms + reliability_ms + arbitration_ms
            extension_times.append(extension_ms)
            timing_samples.append({
                "controller": controller_type,
                "scenario": scenario_name,
                "seed": seed,
                "step": i,
                "time_s": float(t),
                "aux_inference_ms": aux_ms,
                "reliability_update_ms": reliability_ms,
                "arbitration_ms": arbitration_ms,
                "extension_ms": extension_ms,
                "control_compute_ms": np.nan,
            })

        required = [ym, y_fb] + ([] if controller_type == "A_PI" else [y_hat])
        nonfinite_events += int(not np.isfinite(required).all())

        feedback_arr[i] = y_fb
        sub_arr[i] = is_sub
        sensor_state_arr[i] = is_suspect
        source_arr[i] = source_code

        # History updates
        history = np.vstack((history[1:], [prev_voltage, y_fb])).astype(np.float32)
        aux_history = np.vstack((aux_history[1:], [prev_voltage, current_val])).astype(np.float32)

        # Control step
        voltage = prev_voltage
        if i % CONTROL_STRIDE == 0:
            control_started = perf_counter()
            if controller_type == "A_PI":
                voltage = pi.compute_control(ref[i], y_fb, CONTROL_DT)
            else:
                fb_v = pi.compute_control(ref[i], y_fb, CONTROL_DT)
                t_solve0 = perf_counter()
                out = mpc.compute_control(
                    history, reference_values(scenario_name, t + DT * np.arange(1, mpc_config.horizon + 1)), prev_voltage,
                    horizon=mpc_config.horizon, move_blocks=mpc_config.move_blocks,
                    fallback_voltage=fb_v,
                )
                solve_times.append(perf_counter() - t_solve0)
                if not out["success"]:
                    opt_failures += 1
                voltage = out["voltage"]
            control_ms = 1000 * (perf_counter() - control_started)
            control_times.append(control_ms)
            if controller_type == "C3_arbitration_MPC" and timing_samples:
                timing_samples[-1]["control_compute_ms"] = control_ms

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

    # Source switches & fractions
    switches = int(np.sum(source_arr[:-1] != source_arr[1:]))
    frac_phys = float(np.mean(source_arr == 0))
    frac_main = float(np.mean(source_arr == 1))
    frac_aux = float(np.mean(source_arr == 2))
    frac_fb = float(np.mean(source_arr == 3))
    event_frac_aux = float(np.mean(source_arr[ev_mask] == 2))
    event_frac_fb = float(np.mean(source_arr[ev_mask] == 3))

    # Average episode duration per source switch
    episodes = switches + 1
    avg_episode_duration_s = float(DURATION / episodes)

    # Transitions & Latencies
    active_states = sensor_state_arr.copy()
    reliability_entries = int(np.sum(~active_states[:-1] & active_states[1:]))
    recovery_indices = np.flatnonzero(active_states[:-1] & ~active_states[1:]) + 1
    false_recovery_events = int(np.sum(ev_mask[recovery_indices])) if ev_start is not None else 0
    total_recovery_events = int(len(recovery_indices))

    # Recovery latency
    if ev_end is not None and np.any(active_states & ev_mask):
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

    # Constraint violations
    voltage_violations = int(np.sum((ctrl_voltages < -1e-8) | (ctrl_voltages > 12.0 + 1e-8)))
    rate_violations = int(np.sum(np.abs(ctrl_moves) > 2.0 + 1e-8))

    # Virtual sensor quality during substitution (offline)
    sub_idx = np.where(sub_arr)[0]
    if len(sub_idx) > 0:
        main_err = np.abs(main_speed_arr[sub_idx] - true_speed_arr[sub_idx])
        corr_err = np.abs(meas_speed_arr[sub_idx] - true_speed_arr[sub_idx])
        sel_err = np.abs(feedback_arr[sub_idx] - true_speed_arr[sub_idx])

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
            "selected_virtual_rmse": float(np.sqrt(np.mean(sel_err ** 2))),
            "selected_virtual_mae": float(np.mean(sel_err)),
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
            "selected_virtual_rmse": np.nan,
            "selected_virtual_mae": np.nan,
            "corrupted_rmse": np.nan,
            "corrupted_mae": np.nan,
            "sub_samples": 0,
        }

    res_dict = {
        "controller": controller_type,
        "scenario": scenario_name,
        "seed": seed,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "fault_window_rmse": fault_window_rmse,
        "recovery_latency_s": recovery_latency_s,
        "reliability_entries": reliability_entries,
        "recovery_events": total_recovery_events,
        "false_recovery_events": false_recovery_events,
        "sub_duration_s": sub_duration_s,
        "sub_fraction": sub_fraction,
        "switches": switches,
        "avg_episode_duration_s": avg_episode_duration_s,
        "frac_phys": frac_phys,
        "frac_main": frac_main,
        "frac_aux": frac_aux,
        "frac_fb": frac_fb,
        "event_frac_aux": event_frac_aux,
        "event_frac_fb": event_frac_fb,
        "control_effort_u2": control_effort_u2,
        "control_variation_du2": control_variation_du2,
        "voltage_violations": voltage_violations,
        "rate_violations": rate_violations,
        "optimizer_failures": opt_failures,
        "main_prediction_failures": main_prediction_failures,
        "aux_prediction_failures": aux_prediction_failures,
        "prediction_failure_messages": " | ".join(sorted(prediction_failure_messages)),
        "nonfinite_events": nonfinite_events,
        "mean_solve_ms": float(np.mean(solve_times) * 1000) if solve_times else 0.0,
        "mean_aux_inference_ms": float(np.mean(aux_times)) if aux_times else 0.0,
        "mean_reliability_update_ms": float(np.mean(reliability_times)) if reliability_times else 0.0,
        "mean_arbitration_ms": float(np.mean(arb_times)) if arb_times else 0.0,
        "mean_extension_ms": float(np.mean(extension_times)) if extension_times else 0.0,
        "mean_control_compute_ms": float(np.mean(control_times)) if control_times else 0.0,
        "sim_time_s": total_sim_time,
        **quality,
    }

    if timing_samples:
        res_dict["timing_samples"] = timing_samples

    if recovery_diagnostics:
        res_dict["recovery_diagnostics"] = recovery_diagnostics

    if controller_type == "C3_arbitration_MPC" and scenario_name in REFERENCE_SCENARIOS:
        res_dict["reference_substitution_events"] = [
            {
                "scenario": scenario_name,
                "seed": seed,
                "step": int(index),
                "time_s": float(TIME[index]),
                "reference": float(ref[index]),
                "reliability_active": bool(sensor_state_arr[index]),
                "source_code": int(source_arr[index]),
            }
            for index in np.flatnonzero(sub_arr)
        ]

    if record_traces:
        res_dict["trace"] = {
            "time": TIME,
            "true": true_speed_arr,
            "meas": meas_speed_arr,
            "main": main_speed_arr,
            "aux": aux_speed_arr,
            "fb": feedback_arr,
            "source": source_arr,
            "ref": ref,
            "voltage": voltage_arr,
        }

    return res_dict


def plot_combined_fault_timeline(trace: dict, save_path: Path):
    """Create comprehensive timeline plot for combined_fault_load."""
    t = trace["time"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Speeds
    ax1 = axes[0]
    ax1.plot(t, trace["ref"], "k--", label="Reference (35 rad/s)", linewidth=1.5)
    ax1.plot(t, trace["true"], "b-", label="True Speed ω(t)", linewidth=2.0)
    ax1.plot(t, trace["meas"], "r:", label="Measured Speed ym(t) (+5% bias at t≥3s)", alpha=0.7)
    ax1.plot(t, trace["main"], color="orange", linestyle="-.", label="Main Virtual y_main", alpha=0.8)
    ax1.plot(t, trace["aux"], color="green", linestyle="--", label="Auxiliary Virtual y_aux", linewidth=1.8)
    ax1.plot(t, trace["fb"], color="magenta", linestyle="-", label="Selected Feedback y_fb", linewidth=2.0)
    ax1.axvline(3.0, color="gray", linestyle=":", label="Fault & Load Event (t=3.0s)")
    ax1.set_ylabel("Speed (rad/s)", fontsize=11)
    ax1.set_title("V3 Dual-Virtual-Sensor Arbitration under Combined Fault + Load Disturbance", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9, ncol=2)

    # 2. Source Selection State
    ax2 = axes[1]
    source = trace["source"]
    ax2.step(t, source, where="post", color="purple", linewidth=2.0)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(["0: PHYSICAL", "1: MAIN VIRTUAL", "2: AUX VIRTUAL", "3: FALLBACK"], fontsize=10)
    ax2.axvline(3.0, color="gray", linestyle=":")
    ax2.set_ylabel("Active Source", fontsize=11)
    ax2.set_title("Arbitrator Selection State (Clean transitions: Physical → Main → Auxiliary)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 3.5)

    # 3. Control Voltage
    ax3 = axes[2]
    ax3.step(t, trace["voltage"], where="post", color="darkcyan", linewidth=1.8, label="Control Voltage V(t)")
    ax3.axvline(3.0, color="gray", linestyle=":")
    ax3.set_ylabel("Voltage (V)", fontsize=11)
    ax3.set_xlabel("Time (s)", fontsize=11)
    ax3.set_title("MPC Control Action (Voltage smoothly increases to reject 0.15 N*m load step)", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.5, 12.5)
    ax3.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Timeline plot saved to: {save_path}")


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-events-only",
        action="store_true",
        help="rerun only C3 reference scenarios and save substitution-event diagnostics",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = PROJECT / "results/metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((PROJECT / "results/configs/v3_arbitration_config.json").read_text())
    if config["status"] != "frozen_before_final_holdout" or config["final_holdout_seeds"] != FINAL_HOLDOUT_SEEDS:
        raise RuntimeError("V3 configuration is not frozen for the expected final holdout")

    frozen_path = out_dir / "v3_final_holdout_runs.csv"
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if frozen_hash != "160a114a77709d73b3503921dd672000a05bfdc918838b0d13619ac8a0fbcddd":
        raise RuntimeError("frozen 200-run artifact changed; refusing to regenerate evidence")
    frozen_runs = pd.read_csv(frozen_path, float_precision="round_trip")

    workers = min(int(os.environ.get("V3_WORKERS", "8")), os.cpu_count() or 1)
    events_only = args.reference_events_only
    task_keys = {
        (controller, scenario, seed)
        for scenario in REFERENCE_SCENARIOS
        for seed in FINAL_HOLDOUT_SEEDS
        for controller in CONTROLLERS
    }
    if events_only:
        task_keys = {key for key in task_keys if key[0] == "C3_arbitration_MPC"}
    else:
        task_keys.update(
            ("C2_aux_recovery_MPC", scenario, seed)
            for scenario in FROZEN_SCENARIOS
            for seed in FINAL_HOLDOUT_SEEDS
        )
        task_keys.update(
            ("C3_arbitration_MPC", "load_disturbance", seed)
            for seed in FINAL_HOLDOUT_SEEDS
        )
    tasks = [(controller, scenario, seed, False) for controller, scenario, seed in sorted(task_keys)]
    if events_only:
        print(f"Running {len(tasks)} reference-event diagnostic reruns with {workers} workers.")
    else:
        print(f"Running 75 new reference runs plus {len(tasks) - 75} diagnostic-only reruns with {workers} workers.")
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(simulate_v3_run, tasks))
    wall_time = perf_counter() - started

    diagnostics = []
    reference_events = []
    timing_rows = []
    for result in results:
        timing_rows.extend(result.pop("timing_samples", []))
        diagnostics.extend(result.pop("recovery_diagnostics", []))
        reference_events.extend(result.pop("reference_substitution_events", []))

    if events_only:
        pd.DataFrame(reference_events, columns=[
            "scenario", "seed", "step", "time_s", "reference", "reliability_active", "source_code",
        ]).sort_values(["scenario", "seed", "step"]).to_csv(
            out_dir / "v3_reference_substitution_events.csv", index=False
        )
        print(f"Saved {len(reference_events)} reference substitution events.")
        return

    reruns = pd.DataFrame(results)
    continuous_behavior = [
        "overall_rmse", "overall_mae", "fault_window_rmse", "recovery_latency_s",
        "sub_duration_s", "sub_fraction", "avg_episode_duration_s", "frac_phys", "frac_main", "frac_aux",
        "frac_fb", "event_frac_aux", "event_frac_fb", "control_effort_u2",
        "control_variation_du2", "main_virtual_rmse", "main_virtual_mae",
        "selected_virtual_rmse", "selected_virtual_mae", "corrupted_rmse",
        "corrupted_mae",
    ]
    exact_behavior = [
        "recovery_events", "false_recovery_events", "switches", "voltage_violations",
        "rate_violations", "optimizer_failures", "main_prediction_failures",
        "aux_prediction_failures", "nonfinite_events", "sub_samples",
    ]
    diagnostic_reruns = reruns[reruns.scenario.isin(FROZEN_SCENARIOS)]
    for row in diagnostic_reruns.itertuples(index=False):
        saved = frozen_runs[
            (frozen_runs.controller == row.controller)
            & (frozen_runs.scenario == row.scenario)
            & (frozen_runs.seed == row.seed)
        ]
        counts_match = len(saved) == 1 and np.array_equal(
            saved[exact_behavior].to_numpy(),
            np.asarray([[getattr(row, column) for column in exact_behavior]]),
            equal_nan=True,
        )
        values_match = len(saved) == 1 and np.allclose(
            saved[continuous_behavior].to_numpy(float),
            np.asarray([[getattr(row, column) for column in continuous_behavior]], dtype=float),
            rtol=1e-8,
            atol=1e-7,
            equal_nan=True,
        )
        if not counts_match or not values_match:
            raise RuntimeError(f"controller behavior changed for {row.controller}/{row.scenario}/{row.seed}")

    reference_runs = reruns[reruns.scenario.isin(REFERENCE_SCENARIOS)].copy()
    reference_runs.to_csv(out_dir / "v3_final_reference_runs.csv", index=False)
    runs = pd.concat([frozen_runs, reference_runs], ignore_index=True).sort_values(["scenario", "seed", "controller"])
    load_entries = reruns[
        (reruns.controller == "C3_arbitration_MPC") & (reruns.scenario == "load_disturbance")
    ].set_index("seed")["reliability_entries"]
    load_mask = (runs.controller == "C3_arbitration_MPC") & (runs.scenario == "load_disturbance")
    runs.loc[load_mask, "reliability_entries"] = runs.loc[load_mask, "seed"].map(load_entries)
    runs_path = out_dir / "v3_final_11scenario_runs.csv"
    runs.to_csv(runs_path, index=False)

    diagnostic_frame = pd.DataFrame(diagnostics).sort_values(["scenario", "seed", "step"])
    diagnostic_frame.to_csv(out_dir / "c2_recovery_gate_diagnostics.csv", index=False)
    pd.DataFrame(reference_events, columns=[
        "scenario", "seed", "step", "time_s", "reference", "reliability_active", "source_code",
    ]).sort_values(["scenario", "seed", "step"]).to_csv(
        out_dir / "v3_reference_substitution_events.csv", index=False
    )

    aggregate_metrics = [
        "overall_rmse", "overall_mae", "fault_window_rmse", "sub_fraction", "reliability_entries",
        "switches", "frac_phys", "frac_main", "frac_aux", "frac_fb", "control_effort_u2",
        "event_frac_aux", "event_frac_fb", "optimizer_failures",
        "voltage_violations", "rate_violations", "main_prediction_failures",
        "aux_prediction_failures", "nonfinite_events",
    ]
    comparison = runs.groupby(["scenario", "controller"])[aggregate_metrics].agg(["mean", "std", "median"])
    comparison.columns = [f"{metric}_{stat}" for metric, stat in comparison.columns]
    comparison = comparison.reset_index()
    comparison.to_csv(out_dir / "v3_final_11scenario_summary.csv", index=False)

    pivot = runs.pivot(index=["scenario", "seed"], columns="controller", values="fault_window_rmse")
    paired = []
    for (scenario, seed), row in pivot.iterrows():
        paired.append({
            "scenario": scenario,
            "seed": int(seed),
            **{
                f"C3_minus_{controller}": float(row["C3_arbitration_MPC"] - row[controller])
                for controller in CONTROLLERS if controller != "C3_arbitration_MPC"
            },
        })

    equivalence_metrics = [
        "overall_rmse", "overall_mae", "fault_window_rmse", "recovery_events",
        "sub_duration_s", "sub_fraction", "frac_phys", "frac_main", "frac_fb",
        "control_effort_u2", "control_variation_du2", "optimizer_failures",
        "voltage_violations", "rate_violations", "nonfinite_events",
    ]
    c1 = runs[runs.controller == "C1_sensor_MPC"].sort_values(["scenario", "seed"])
    c2 = runs[runs.controller == "C2_aux_recovery_MPC"].sort_values(["scenario", "seed"])
    c2_identical = c1[["scenario", "seed"]].reset_index(drop=True).equals(c2[["scenario", "seed"]].reset_index(drop=True)) and np.allclose(
        c1[equivalence_metrics].to_numpy(float), c2[equivalence_metrics].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True
    )
    diagnostic_summary = {
        "recovery_opportunities": int(len(diagnostic_frame)),
        "blocked_by_residual": int(diagnostic_frame["blocked_by_residual"].sum()),
        "blocked_by_cusum": int(diagnostic_frame["blocked_by_cusum"].sum()),
        "blocked_by_persistence": int(diagnostic_frame["blocked_by_persistence"].sum()),
        "blocked_only_by_auxiliary": int(diagnostic_frame["blocked_only_by_auxiliary"].sum()),
        "recoveries": int(diagnostic_frame["recovered_this_step"].sum()),
        "auxiliary_gate_delayed_recovery_relative_to_c1": bool(
            (c2["sub_duration_s"].to_numpy() > c1["sub_duration_s"].to_numpy() + 1e-12).any()
        ),
        "c2_behaviorally_identical_to_c1": bool(c2_identical),
    }

    def file_hash(relative):
        return hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest()

    summary = {
        "evidence_role": "untouched_final_holdout",
        "final_holdout_seeds": FINAL_HOLDOUT_SEEDS,
        "controllers": CONTROLLERS,
        "scenarios": list(SCENARIOS),
        "run_count": int(len(runs)),
        "paired_design": "same scenario, seed, base noise, and injected fault realization across controllers",
        "thresholds_frozen_before_holdout": True,
        "calibration_sha256": config["calibration_sha256"],
        "frozen_artifact_sha256": {
            "main_model": file_hash("results/lstm_model_weights.pt"),
            "auxiliary_model": file_hash("results/auxiliary_model_weights.pt"),
            "v3_arbitration_config": file_hash("results/configs/v3_arbitration_config.json"),
            "v3_calibration": file_hash("results/configs/v3_arbitration_calibration.json"),
            "original_200_runs": frozen_hash,
        },
        "optimizer_failures": int(runs["optimizer_failures"].sum()),
        "voltage_violations": int(runs["voltage_violations"].sum()),
        "rate_violations": int(runs["rate_violations"].sum()),
        "main_prediction_failures": int(runs["main_prediction_failures"].sum()),
        "aux_prediction_failures": int(runs["aux_prediction_failures"].sum()),
        "nonfinite_events": int(runs["nonfinite_events"].sum()),
        "comparison": comparison.to_dict(orient="records"),
        "paired_fault_window_rmse_differences": paired,
        "c2_recovery_gate_diagnostics": diagnostic_summary,
        "reference_evaluation_wall_time_s": wall_time,
    }
    (out_dir / "v3_final_11scenario_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Saved {len(runs)} runs to {runs_path}; wall time {wall_time:.1f}s.")


if __name__ == "__main__":
    main()
