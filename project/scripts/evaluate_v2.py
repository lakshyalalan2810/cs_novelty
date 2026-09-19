"""V2 Auxiliary-Recovery Reliability Evaluation.

Compares V1 (existing bug-fixed baseline) vs V2 (auxiliary recovery agreement)
across all 7 fault scenarios, computes virtual sensor quality, and runs fresh
holdout evaluation.

Usage:
    python scripts/evaluate_v2.py
"""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from reliability import guarded_predict, SensorReliabilityMonitor
from lstm_model import LSTMForecaster
from auxiliary_sensor_model import AuxiliarySpeedEstimator
from motor_model import DCMotorParams, dc_motor_dynamics, simulate_motor
from data_utils import make_random_step_signal, make_multisine_voltage

METRICS_DIR = PROJECT / "results" / "metrics"
FULL_SCALE = 65.234375
LOAD_DISTURBANCE_TORQUE = 0.15
OFFLINE_AUX_QUALITY_FILENAME = "v2_offline_aux_virtual_sensor_quality.csv"

# ─── Fault scenario definitions ──────────────────────────────────────────────
# Fault window uses dataset indices (0-based into 1201-point trajectories).
# Matching existing evaluation: fault active during [start, end).
SCENARIOS = {
    "sensor_bias_5":       {"type": "bias",     "magnitude": 0.05 * FULL_SCALE, "start": 200, "end": 400},
    "sensor_bias_15":      {"type": "bias",     "magnitude": 0.15 * FULL_SCALE, "start": 200, "end": 400},
    "sensor_dropout":      {"type": "dropout",  "start": 200, "end": 400},
    "sensor_drift":        {"type": "drift",    "rate": 0.15, "start": 200, "end": None},
    "load_disturbance":    {"type": "load",     "start": 300, "end": None},
    "parameter_variation": {"type": "param",    "start": 300, "end": None},
    "combined_fault_load": {"type": "combined", "bias_mag": 0.05 * FULL_SCALE, "start": 300, "end": None},
}


# ─── Fault injection ─────────────────────────────────────────────────────────

def inject_fault(y_measured: np.ndarray, scenario: dict) -> np.ndarray:
    """Corrupt measured speed according to fault scenario."""
    corrupted = y_measured.copy()
    s = scenario["start"]
    e = scenario["end"] if scenario["end"] is not None else len(corrupted)

    if scenario["type"] == "bias":
        corrupted[s:e] += scenario["magnitude"]
    elif scenario["type"] == "dropout":
        corrupted[s:e] = 0.0
    elif scenario["type"] == "drift":
        # Linearly growing bias from start onwards
        for i in range(s, e):
            corrupted[i] += scenario["rate"] * (i - s) * 0.01  # rate * elapsed_time
    elif scenario["type"] == "combined":
        corrupted[s:e] += scenario["bias_mag"]
    # load / param types don't corrupt the sensor
    return corrupted


def shifted_motor_params(params: DCMotorParams) -> DCMotorParams:
    """Apply the frozen V2 parameter-variation factors to one plant instance."""
    return replace(
        params,
        resistance=1.20 * params.resistance,
        inductance=0.85 * params.inductance,
        back_emf_constant=1.15 * params.back_emf_constant,
        torque_constant=0.85 * params.torque_constant,
        inertia=1.20 * params.inertia,
        viscous_friction=1.20 * params.viscous_friction,
        coulomb_friction=1.20 * params.coulomb_friction,
    )


def rk4_step(state, voltage, load, params, timestep):
    """Advance the recorded open-loop plant with the evaluator's fixed timestep."""
    derivative = lambda value: dc_motor_dynamics(0.0, value, voltage, load, params)
    k1 = derivative(state)
    k2 = derivative(state + timestep * k1 / 2)
    k3 = derivative(state + timestep * k2 / 2)
    k4 = derivative(state + timestep * k3)
    return state + timestep * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def apply_scenario_to_trajectory(
    voltage: np.ndarray,
    current: np.ndarray,
    y_true: np.ndarray,
    y_measured: np.ndarray,
    load_torque: np.ndarray,
    params: DCMotorParams,
    scenario: dict,
    timestep: float,
) -> dict[str, np.ndarray]:
    """Create the physical trajectory and sensor trace for one V2 scenario.

    Dataset trajectories are retained exactly before the plant-event boundary.
    For load/parameter scenarios, the state is re-integrated from the recorded
    event-start state so the claimed disturbance is physically present.  The
    original measurement-noise realization is then applied to the regenerated
    true speed before any sensor fault is injected.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_measured = np.asarray(y_measured, dtype=float)
    load_torque = np.asarray(load_torque, dtype=float)
    if not (
        voltage.ndim == current.ndim == y_true.ndim == y_measured.ndim == load_torque.ndim == 1
        and voltage.shape == current.shape == y_true.shape == y_measured.shape == load_torque.shape
    ):
        raise ValueError("trajectory arrays must be equal-length 1-D arrays")
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("timestep must be finite and positive")

    event_start = int(scenario["start"])
    if event_start < 0 or event_start >= len(voltage):
        raise ValueError("scenario start must lie inside the trajectory")

    scenario_current = current.copy()
    scenario_true = y_true.copy()
    if scenario["type"] in {"load", "param", "combined"}:
        shifted = shifted_motor_params(params)
        state = np.array([current[event_start], y_true[event_start]], dtype=float)
        for index in range(event_start, len(voltage) - 1):
            load = (
                LOAD_DISTURBANCE_TORQUE
                if scenario["type"] in {"load", "combined"}
                else float(load_torque[index])
            )
            active_params = shifted if scenario["type"] == "param" else params
            state = rk4_step(
                state,
                float(voltage[index]),
                load,
                active_params,
                timestep,
            )
            scenario_current[index + 1] = state[0]
            scenario_true[index + 1] = state[1]

    measurement_noise = y_measured - y_true
    scenario_measured = scenario_true + measurement_noise
    corrupted = inject_fault(scenario_measured, scenario)
    return {
        "current": scenario_current,
        "y_true": scenario_true,
        "y_measured": corrupted,
    }


# ─── V2 guarded predict ──────────────────────────────────────────────────────

@torch.no_grad()
def run_guarded_v2(
    model, aux_model, voltage, measured, current,
    normalization, aux_normalization, window_length,
    residual_gate, aux_recovery_gate, cusum,
    enter_count, exit_count,
):
    """Run V2 online prediction with auxiliary recovery agreement."""
    voltage = np.asarray(voltage, dtype=np.float32)
    measured = np.asarray(measured, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)

    inp_mean = np.asarray(normalization["input_mean"], dtype=np.float32)
    inp_std = np.asarray(normalization["input_std"], dtype=np.float32)
    tgt_mean = float(normalization["target_mean"][0])
    tgt_std = float(normalization["target_std"][0])

    aux_inp_mean = np.asarray(aux_normalization["input_mean"], dtype=np.float32)
    aux_inp_std = np.asarray(aux_normalization["input_std"], dtype=np.float32)
    aux_tgt_mean = float(aux_normalization["target_mean"][0])
    aux_tgt_std = float(aux_normalization["target_std"][0])

    device = next(model.parameters()).device
    n = len(voltage)
    wl = window_length

    # Histories
    hist = np.stack((voltage[:wl], measured[:wl]), axis=1)     # [voltage, speed]
    aux_hist = np.stack((voltage[:wl], current[:wl]), axis=1)  # [voltage, current]

    length = n - wl
    pred_arr = np.empty(length, dtype=np.float32)
    resid_arr = np.empty(length, dtype=np.float32)
    aux_pred_arr = np.empty(length, dtype=np.float32)
    aux_resid_arr = np.empty(length, dtype=np.float32)
    sub_arr = np.zeros(length, dtype=bool)
    alarm_arr = np.zeros(length, dtype=bool)

    monitor = SensorReliabilityMonitor(
        residual_gate=residual_gate,
        center=cusum["center"],
        allowance=cusum["allowance"],
        threshold=cusum["threshold"],
        enter_count=enter_count,
        exit_count=exit_count,
        aux_recovery_gate=aux_recovery_gate,
    )

    model.eval()
    aux_model.eval()

    for oi, si in enumerate(range(wl, n)):
        # Main prediction
        norm = (hist - inp_mean) / inp_std
        out = model(torch.from_numpy(norm[None]).to(device)).cpu().numpy()[0, 0]
        phys = out * tgt_std + tgt_mean

        # Aux prediction
        aux_norm = (aux_hist - aux_inp_mean) / aux_inp_std
        aux_out = aux_model(torch.from_numpy(aux_norm[None]).to(device)).cpu().numpy()[0, 0]
        aux_phys = aux_out * aux_tgt_std + aux_tgt_mean

        error = measured[si] - phys
        aux_error = measured[si] - aux_phys

        decision = monitor.update(error, aux_residual=aux_error)

        pred_arr[oi] = phys
        resid_arr[oi] = error
        aux_pred_arr[oi] = aux_phys
        aux_resid_arr[oi] = aux_error
        sub_arr[oi] = decision["substitute"]
        alarm_arr[oi] = decision["sensor_suspect"]

        feedback = phys if decision["substitute"] else measured[si]
        hist = np.concatenate((hist[1:], [[voltage[si], feedback]]), axis=0)
        aux_hist = np.concatenate((aux_hist[1:], [[voltage[si], current[si]]]), axis=0)

    return {
        "prediction": pred_arr, "residual": resid_arr,
        "substituted": sub_arr, "sensor_alarm": alarm_arr,
        "aux_prediction": aux_pred_arr, "aux_residual": aux_resid_arr,
    }


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true, corrupted, result, scenario, wl):
    """Compute per-run evaluation metrics."""
    pred = result["prediction"]
    sub = result["substituted"]
    alarm = result["sensor_alarm"]
    length = len(pred)
    yt = y_true[wl: wl + length]

    overall_rmse = float(np.sqrt(np.mean((yt - pred) ** 2)))

    s = max(0, scenario["start"] - wl)
    e = (scenario["end"] - wl) if scenario["end"] is not None else length
    e = min(e, length)

    if s < e:
        fault_rmse = float(np.sqrt(np.mean((yt[s:e] - pred[s:e]) ** 2)))
    else:
        fault_rmse = np.nan

    # Detection latency
    det_lat = np.nan
    if scenario["type"] in ("bias", "dropout", "drift", "combined") and s < length:
        hits = np.where(alarm[s:min(e, length)])[0]
        if len(hits):
            det_lat = int(hits[0])

    # False recoveries during active fault window
    false_rec = 0
    if scenario["type"] in ("bias", "dropout", "drift", "combined") and s < e:
        sw = alarm[s:e].astype(int)
        false_rec = int(np.sum(np.diff(sw) == -1))

    # Recovery latency (after fault ends)
    rec_lat = np.nan
    if scenario["end"] is not None and e < length:
        fault_alarm = alarm[s:e]
        if np.any(fault_alarm):
            if not fault_alarm[-1]:
                # The latched alarm recovered while the finite fault was still active.
                rec_lat = 0
            else:
                post = alarm[e:]
                recs = np.where(~post)[0]
                if len(recs):
                    rec_lat = int(recs[0])
                else:
                    rec_lat = len(post)  # never recovered

    sub_dur = int(np.sum(sub))
    sub_frac = float(sub_dur / length)

    return {
        "overall_rmse": overall_rmse,
        "fault_rmse": fault_rmse,
        "detection_latency": det_lat,
        "recovery_latency": rec_lat,
        "false_recoveries": false_rec,
        "sub_duration": sub_dur,
        "sub_fraction": sub_frac,
    }


def compute_virtual_sensor_quality(y_true, corrupted, result, scenario, wl):
    """Compute quality metrics during substitution timesteps."""
    pred = result["prediction"]
    sub = result["substituted"]
    length = len(pred)
    yt = y_true[wl: wl + length]
    ym = corrupted[wl: wl + length]

    sub_idx = np.where(sub)[0]
    if len(sub_idx) == 0:
        return None

    main_err = np.abs(pred[sub_idx] - yt[sub_idx])
    corr_err = np.abs(ym[sub_idx] - yt[sub_idx])

    quality = {
        "main_virtual_rmse": float(np.sqrt(np.mean(main_err ** 2))),
        "main_virtual_mae": float(np.mean(main_err)),
        "corrupted_rmse": float(np.sqrt(np.mean(corr_err ** 2))),
        "corrupted_mae": float(np.mean(corr_err)),
        "sub_count": len(sub_idx),
    }

    # Aux quality if available
    if "aux_prediction" in result:
        aux_err = np.abs(result["aux_prediction"][sub_idx] - yt[sub_idx])
        quality["aux_rmse"] = float(np.sqrt(np.mean(aux_err ** 2)))
        quality["aux_mae"] = float(np.mean(aux_err))

    return quality


# ─── Fresh holdout trajectory generation ─────────────────────────────────────

def generate_fresh_trajectory(seed, duration=12.0, dt=0.01):
    """Generate a fresh motor trajectory with a new seed."""
    rng = np.random.default_rng(seed)

    # Random voltage signal
    voltage_fn = make_multisine_voltage(rng, voltage_limits=(0.0, 12.0))

    # Random load steps
    load_levels = [0.0, 0.02, 0.05, 0.08, 0.1]
    load_fn = make_random_step_signal(rng, duration, 2.0, np.array(load_levels))

    params = DCMotorParams()
    time_arr, states = simulate_motor(
        voltage_fn, load_fn, simulation_time=duration, timestep=dt, params=params,
    )
    current_arr, speed_arr = states.T
    voltage_arr = np.array([voltage_fn(t) for t in time_arr], dtype=np.float32)
    noise = rng.normal(0.0, 0.25, len(speed_arr))

    return {
        "voltage": voltage_arr,
        "current": current_arr.astype(np.float32),
        "y_true": speed_arr.astype(np.float32),
        "y_measured": (speed_arr + noise).astype(np.float32),
        "load_torque": np.array([load_fn(t) for t in time_arr], dtype=np.float32),
        "params": params,
        "timestep": float(dt),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("  V2 AUXILIARY-RECOVERY RELIABILITY EVALUATION")
    print("=" * 70)

    device = "cpu"

    # ── Load models ───────────────────────────────────────────────────────
    print("\n1. Loading models...")
    main_cfg = json.loads((PROJECT / "results/configs/lstm_model_config.json").read_text())
    model = LSTMForecaster(**main_cfg["model"]).to(device)
    model.load_state_dict(torch.load(
        PROJECT / "results/lstm_model_weights.pt", map_location=device, weights_only=True,
    ))
    model.eval()
    print(f"   Main LSTM loaded ({sum(p.numel() for p in model.parameters())} params)")

    aux_cfg = json.loads((PROJECT / "results/configs/v2_auxiliary_config.json").read_text())
    aux_model = AuxiliarySpeedEstimator(**aux_cfg["model"]).to(device)
    aux_model.load_state_dict(torch.load(
        PROJECT / "results/auxiliary_model_weights.pt", map_location=device, weights_only=True,
    ))
    aux_model.eval()
    print(f"   Auxiliary LSTM loaded ({sum(p.numel() for p in aux_model.parameters())} params)")

    # ── Load configs ──────────────────────────────────────────────────────
    rel_cfg = json.loads((PROJECT / "results/configs/reliability_final_config.json").read_text())
    sensor_cfg = rel_cfg["sensor"]

    wl = 20
    residual_gate = sensor_cfg["instant_threshold"]
    cusum = {
        "center": sensor_cfg["center"],
        "allowance": sensor_cfg["allowance"],
        "threshold": sensor_cfg["threshold"],
    }
    enter_count = sensor_cfg["enter_count"]
    exit_count = sensor_cfg["exit_count"]
    aux_recovery_gate = aux_cfg["calibration"]["recommended_recovery_gate"]

    print(f"   Sensor gate: {residual_gate:.4f}")
    print(f"   CUSUM threshold: {cusum['threshold']:.4f}")
    print(f"   Aux recovery gate: {aux_recovery_gate:.4f}")
    print(f"   Enter/exit counts: {enter_count}/{exit_count}")

    # ── Load dataset ──────────────────────────────────────────────────────
    print("\n2. Loading dataset...")
    data = np.load(PROJECT / "data/processed/dc_motor_lstm_dataset.npz")
    test_ids = data["split_run_ids_test"]
    print(f"   Test trajectory IDs: {list(test_ids)}")

    normalization = main_cfg["normalization"]
    aux_normalization = aux_cfg["normalization"]
    dataset_timestep = float(data["timestep"])
    parameter_names = [str(name) for name in data["parameter_names"]]

    # ── Run scenarios on dataset test trajectories ────────────────────────
    print("\n3. Running V1 vs V2 on dataset test trajectories...")
    all_rows = []
    quality_rows = []

    for sc_name, sc_conf in SCENARIOS.items():
        print(f"\n   Scenario: {sc_name}")
        for rid in test_ids:
            v = data["voltage"][rid]
            base_params = DCMotorParams(**dict(zip(
                parameter_names,
                (float(value) for value in data["parameter_values"][rid]),
            )))
            trajectory = apply_scenario_to_trajectory(
                v,
                data["current"][rid],
                data["y_true"][rid],
                data["y_measured"][rid],
                data["load_torque"][rid],
                base_params,
                sc_conf,
                dataset_timestep,
            )
            c = trajectory["current"]
            yt = trajectory["y_true"]
            corrupted = trajectory["y_measured"]

            # V1: standard guarded_predict (no aux)
            v1_res = guarded_predict(
                model, v, corrupted, normalization, wl, residual_gate, cusum,
                enter_count=enter_count, exit_count=exit_count,
            )
            v1_m = compute_metrics(yt, corrupted, v1_res, sc_conf, wl)
            v1_m.update(scenario=sc_name, run_id=int(rid), version="V1")
            all_rows.append(v1_m)

            # V2: guarded with auxiliary recovery
            v2_res = run_guarded_v2(
                model, aux_model, v, corrupted, c,
                normalization, aux_normalization, wl,
                residual_gate, aux_recovery_gate, cusum,
                enter_count, exit_count,
            )
            v2_m = compute_metrics(yt, corrupted, v2_res, sc_conf, wl)
            v2_m.update(scenario=sc_name, run_id=int(rid), version="V2")
            all_rows.append(v2_m)

            # Virtual sensor quality (V1)
            q1 = compute_virtual_sensor_quality(yt, corrupted, v1_res, sc_conf, wl)
            if q1 is not None:
                q1.update(scenario=sc_name, run_id=int(rid), version="V1")
                quality_rows.append(q1)

            # Virtual sensor quality (V2)
            q2 = compute_virtual_sensor_quality(yt, corrupted, v2_res, sc_conf, wl)
            if q2 is not None:
                q2.update(scenario=sc_name, run_id=int(rid), version="V2")
                quality_rows.append(q2)

        # Aggregate per scenario
        for ver in ("V1", "V2"):
            rows = [r for r in all_rows if r["scenario"] == sc_name and r["version"] == ver]
            avg_rmse = np.mean([r["overall_rmse"] for r in rows])
            avg_fault = np.nanmean([r["fault_rmse"] for r in rows])
            avg_false = np.mean([r["false_recoveries"] for r in rows])
            print(f"     {ver}: RMSE={avg_rmse:.4f}, FaultRMSE={avg_fault:.4f}, "
                  f"FalseRec={avg_false:.1f}")

    # ── Save metrics ──────────────────────────────────────────────────────
    print("\n4. Saving metrics...")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(METRICS_DIR / "v2_vs_v1_comparison.csv", index=False)

    df_v2_only = df_all[df_all["version"] == "V2"]
    df_v2_only.to_csv(METRICS_DIR / "v2_reliability_analysis_metrics.csv", index=False)

    df_quality = pd.DataFrame(quality_rows) if quality_rows else pd.DataFrame()
    df_quality.to_csv(METRICS_DIR / OFFLINE_AUX_QUALITY_FILENAME, index=False)

    # ── Print comparison summary ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  V1 vs V2 COMPARISON SUMMARY (averaged over test trajectories)")
    print("=" * 70)
    print(f"{'Scenario':<25} {'V1 RMSE':>9} {'V2 RMSE':>9} {'V1 FltRMSE':>11} "
          f"{'V2 FltRMSE':>11} {'V1 FalseR':>9} {'V2 FalseR':>9}")
    print("-" * 90)

    for sc_name in SCENARIOS:
        v1_sc = df_all[(df_all["scenario"] == sc_name) & (df_all["version"] == "V1")]
        v2_sc = df_all[(df_all["scenario"] == sc_name) & (df_all["version"] == "V2")]
        print(f"{sc_name:<25} "
              f"{v1_sc['overall_rmse'].mean():9.4f} {v2_sc['overall_rmse'].mean():9.4f} "
              f"{v1_sc['fault_rmse'].mean():11.4f} {v2_sc['fault_rmse'].mean():11.4f} "
              f"{v1_sc['false_recoveries'].mean():9.1f} {v2_sc['false_recoveries'].mean():9.1f}")

    # ── Fresh holdout ─────────────────────────────────────────────────────
    print("\n5. Running fresh holdout evaluation...")
    fresh_rows = []
    fresh_seeds = [9001, 9002, 9003, 9004, 9005]

    for seed in fresh_seeds:
        traj = generate_fresh_trajectory(seed)
        v = traj["voltage"]

        for sc_name, sc_conf in SCENARIOS.items():
            trajectory = apply_scenario_to_trajectory(
                v,
                traj["current"],
                traj["y_true"],
                traj["y_measured"],
                traj["load_torque"],
                traj["params"],
                sc_conf,
                traj["timestep"],
            )
            c = trajectory["current"]
            yt = trajectory["y_true"]
            corrupted = trajectory["y_measured"]

            v1_res = guarded_predict(
                model, v, corrupted, normalization, wl, residual_gate, cusum,
                enter_count=enter_count, exit_count=exit_count,
            )
            v1_m = compute_metrics(yt, corrupted, v1_res, sc_conf, wl)
            v1_m.update(scenario=sc_name, seed=seed, version="V1")
            fresh_rows.append(v1_m)

            v2_res = run_guarded_v2(
                model, aux_model, v, corrupted, c,
                normalization, aux_normalization, wl,
                residual_gate, aux_recovery_gate, cusum,
                enter_count, exit_count,
            )
            v2_m = compute_metrics(yt, corrupted, v2_res, sc_conf, wl)
            v2_m.update(scenario=sc_name, seed=seed, version="V2")
            fresh_rows.append(v2_m)

    df_fresh = pd.DataFrame(fresh_rows)
    df_fresh.to_csv(METRICS_DIR / "v2_fresh_holdout.csv", index=False)

    print("\n  Fresh Holdout Summary:")
    print(f"  {'Scenario':<25} {'V1 RMSE':>9} {'V2 RMSE':>9} {'V1 FalseR':>9} {'V2 FalseR':>9}")
    print("  " + "-" * 60)
    for sc_name in SCENARIOS:
        v1_f = df_fresh[(df_fresh["scenario"] == sc_name) & (df_fresh["version"] == "V1")]
        v2_f = df_fresh[(df_fresh["scenario"] == sc_name) & (df_fresh["version"] == "V2")]
        print(f"  {sc_name:<25} "
              f"{v1_f['overall_rmse'].mean():9.4f} {v2_f['overall_rmse'].mean():9.4f} "
              f"{v1_f['false_recoveries'].mean():9.1f} {v2_f['false_recoveries'].mean():9.1f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n  Files saved:")
    print(f"    {METRICS_DIR / 'v2_vs_v1_comparison.csv'}")
    print(f"    {METRICS_DIR / 'v2_reliability_analysis_metrics.csv'}")
    print(f"    {METRICS_DIR / OFFLINE_AUX_QUALITY_FILENAME}")
    print(f"    {METRICS_DIR / 'v2_fresh_holdout.csv'}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
