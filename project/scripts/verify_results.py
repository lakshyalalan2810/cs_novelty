"""Independently verify the canonical saved project evidence."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
CONFIGS = ROOT / "results" / "configs"
PLOTS = ROOT / "results" / "plots"
RAW = ROOT / "results" / "raw"
sys.path.insert(0, str(ROOT / "src"))

from lstm_model import LSTMForecaster, recursive_forecast
from reliability import SensorReliabilityMonitor


_NO_ASSERTION_MESSAGE = object()


def require(condition: object, message: object = _NO_ASSERTION_MESSAGE) -> None:
    """Raise AssertionError on failed verification even under optimized Python."""
    if condition:
        return
    if message is _NO_ASSERTION_MESSAGE:
        raise AssertionError
    raise AssertionError(message)


def read_csv(name: str) -> list[dict[str, str]]:
    with (METRICS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], controller: str, scenarios: set[str], field: str) -> float:
    return fmean(float(row[field]) for row in rows if row["controller"] == controller and row["scenario"] in scenarios)


def assert_close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    require(abs(actual - expected) <= tolerance * max(1.0, abs(expected)), (actual, expected))


def verify_dataset_and_model() -> dict[str, float]:
    data = np.load(ROOT / "data" / "processed" / "dc_motor_lstm_dataset.npz")
    splits = {
        name: data[f"split_run_ids_{name}"].astype(int)
        for name in ("train", "validation", "test")
    }
    require([len(splits[name]) for name in splits] == [18, 6, 6])
    require(all(
        set(splits[first]).isdisjoint(splits[second])
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ))
    require(set(np.concatenate(list(splits.values()))) == set(data["run_ids"].astype(int)))
    for name, run_ids in splits.items():
        require(set(np.unique(data[f"window_run_ids_{name}"])) == set(run_ids))

    window = int(data["window_length"])
    raw_inputs = []
    raw_targets = []
    for run_id in splits["train"]:
        features = np.column_stack((data["voltage"][run_id], data["y_measured"][run_id]))
        raw_inputs.extend(features[index : index + window] for index in range(len(features) - window))
        raw_targets.append(data["y_true"][run_id, window:])
    raw_inputs = np.asarray(raw_inputs)
    raw_targets = np.concatenate(raw_targets)[:, None]
    expected_stats = {
        "input_mean": raw_inputs.mean((0, 1), dtype=np.float64),
        "input_std": raw_inputs.std((0, 1), dtype=np.float64),
        "target_mean": raw_targets.mean(0, dtype=np.float64),
        "target_std": raw_targets.std(0, dtype=np.float64),
    }
    for name, expected in expected_stats.items():
        require(np.allclose(expected, data[f"normalization_{name}"], atol=1e-10))
    require(np.allclose(data["X_train"].mean((0, 1), dtype=np.float64), 0.0, atol=1e-4))
    require(np.allclose(data["X_train"].std((0, 1), dtype=np.float64), 1.0, atol=1e-4))

    config = json.loads((CONFIGS / "lstm_model_config.json").read_text(encoding="utf-8"))
    saved = json.loads((METRICS / "lstm_test_metrics.json").read_text(encoding="utf-8"))
    require(config["dataset"] == "data/processed/dc_motor_lstm_dataset.npz")
    model = LSTMForecaster(**config["model"])
    model.load_state_dict(
        torch.load(ROOT / "results" / "lstm_model_weights.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        predicted = np.concatenate(
            [model(torch.from_numpy(data["X_test"][start : start + 512])).numpy() for start in range(0, len(data["X_test"]), 512)]
        )[:, 0]
    target_mean = float(data["normalization_target_mean"][0])
    target_std = float(data["normalization_target_std"][0])
    true = data["y_test"][:, 0] * target_std + target_mean
    predicted = predicted * target_std + target_mean
    error = predicted - true
    test = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error**2) / np.sum((true - true.mean()) ** 2)),
        "nrmse_range": float(np.sqrt(np.mean(error**2)) / np.ptp(true)),
    }
    for name, value in test.items():
        assert_close(value, saved["test"][name])

    input_mean = data["normalization_input_mean"]
    input_std = data["normalization_input_std"]
    histories, voltages, truths = [], [], []
    for run_id in splits["test"]:
        features = np.column_stack((data["voltage"][run_id], data["y_measured"][run_id]))
        normalized = (features - input_mean) / input_std
        normalized_voltage = (data["voltage"][run_id] - input_mean[0]) / input_std[0]
        for target in range(window, len(data["time"]) - 15 + 1, 20):
            histories.append(normalized[target - window : target])
            voltages.append(normalized_voltage[target : target + 15])
            truths.append(data["y_true"][run_id, target : target + 15])
    tensors = [
        torch.tensor(np.asarray(values), dtype=torch.float32)
        for values in (histories, voltages, input_mean, input_std, data["normalization_target_mean"], data["normalization_target_std"])
    ]
    recursive = recursive_forecast(model, *tensors).numpy() * target_std + target_mean
    recursive_rmse = np.sqrt(np.mean((recursive - np.asarray(truths)) ** 2, axis=0))
    require(np.allclose(recursive_rmse, saved["recursive_rmse_by_horizon"], atol=1e-6))
    return {
        **test,
        "persistence_rmse": float(saved["persistence"]["rmse"]),
        "recursive_rmse_h15": float(recursive_rmse[-1]),
    }


def verify_controller_results() -> tuple[dict[str, float | int | bool], dict, dict]:
    runs = read_csv("final_controller_runs.csv")
    comparison = read_csv("final_controller_comparison.csv")
    runtime_samples = read_csv("final_runtime_samples.csv")
    virtual_quality = read_csv("virtual_feedback_quality.csv")
    summary = json.loads((METRICS / "final_summary.json").read_text(encoding="utf-8"))
    config = json.loads((CONFIGS / "final_mpc_config.json").read_text(encoding="utf-8"))

    expected_controllers = {"A_PI", "B_plain_MPC", "C_sensor_MPC", "D_adaptive"}
    expected_scenarios = {
        "nominal_tracking", "step_reference", "changing_reference",
        "load_disturbance", "parameter_variation", "sensor_noise",
        "bias_5", "bias_15", "sensor_dropout", "sensor_drift",
        "combined_fault_load",
    }
    expected_seeds = {12026, 12027, 12028, 12029, 12030}

    require(config["evaluation_role"] == "untouched_final_holdout")
    require(config["seed_repetitions"] == sorted(expected_seeds))
    assert_close(config["plant_timestep_s"], 0.01)
    assert_close(config["control_interval_s"], 0.05)
    require(config["control_interval_steps"] == 5)
    require(config["mpc"]["horizon"] == 20)
    require(config["mpc"]["move_blocks"] == [5, 15])
    require(len(config["mpc"]["move_blocks"]) == 2)
    require(config["mpc"]["control_interval_steps"] == 5)
    require(config["mpc"]["max_iterations"] == 8)
    require(config["selected_strategy"] == "fixed_multirate_blocks")
    require(summary["selected_configuration"] == config)
    require(summary["statistical_repetitions"] == 5)

    counts = Counter((row["controller"], row["scenario"]) for row in runs)
    require(len(runs) == 220 and len(comparison) == 44)
    require(len(counts) == 44 and set(counts.values()) == {5})
    require({row["controller"] for row in runs} == expected_controllers)
    require({row["scenario"] for row in runs} == expected_scenarios)
    require({int(row["seed"]) for row in runs} == expected_seeds)
    for controller in expected_controllers:
        for scenario in expected_scenarios:
            seeds = {
                int(row["seed"])
                for row in runs
                if row["controller"] == controller and row["scenario"] == scenario
            }
            require(seeds == expected_seeds)
    required_columns = {
        "RMSE", "IAE", "ISE", "max_abs_error", "control_effort_sum_u2",
        "recovery_time_s", "constraint_violations", "optimizer_failures",
        "voltage_violations", "rate_violations", "nonfinite_events",
        "average_solve_ms", "p95_solve_ms", "max_solve_ms",
        "substitution_fraction", "recovery_events", "false_recovery_events",
    }
    require(required_columns <= runs[0].keys())

    comparison_lookup = {(row["controller"], row["scenario"]): row for row in comparison}
    for key, repetitions in counts.items():
        aggregate = comparison_lookup[key]
        require(int(float(aggregate["repetitions"])) == repetitions)
        group = [row for row in runs if (row["controller"], row["scenario"]) == key]
        for field in ("RMSE", "fault_interval_RMSE", "optimizer_failures",
                      "voltage_violations", "rate_violations", "nonfinite_events"):
            assert_close(fmean(float(row[field]) for row in group), float(aggregate[field]))

    sensor = {"sensor_noise", "bias_5", "bias_15", "sensor_dropout", "sensor_drift"}
    disturbance = {"load_disturbance", "parameter_variation", "combined_fault_load"}
    plain_sensor = mean(runs, "B_plain_MPC", sensor, "fault_interval_RMSE")
    reliable_sensor = mean(runs, "C_sensor_MPC", sensor, "fault_interval_RMSE")
    sensor_improvement = 100 * (plain_sensor - reliable_sensor) / plain_sensor
    plain_disturbance = mean(runs, "B_plain_MPC", disturbance, "fault_interval_RMSE")
    adaptive_disturbance = mean(runs, "D_adaptive", disturbance, "fault_interval_RMSE")
    adaptive_degradation = 100 * (adaptive_disturbance - plain_disturbance) / plain_disturbance
    headline = {
        "plain_sensor_rmse": plain_sensor,
        "reliable_sensor_rmse": reliable_sensor,
        "sensor_improvement_percent": sensor_improvement,
        "plain_disturbance_rmse": plain_disturbance,
        "adaptive_disturbance_rmse": adaptive_disturbance,
        "adaptive_degradation_percent": adaptive_degradation,
    }
    for name, value in headline.items():
        assert_close(value, float(summary["headline_metrics"][name]))

    safety = {
        field: sum(int(float(row[field])) for row in runs)
        for field in ("optimizer_failures", "voltage_violations", "rate_violations", "nonfinite_events")
    }
    require(safety == summary["safety"])
    require(set(safety.values()) == {0})

    require(len(runtime_samples) == 3 * 11 * 5 * 121)
    require({row["controller"] for row in runtime_samples} == expected_controllers - {"A_PI"})
    require({row["scenario"] for row in runtime_samples} == expected_scenarios)
    require({int(row["seed"]) for row in runtime_samples} == expected_seeds)
    runtime_keys = [
        (row["controller"], row["scenario"], int(row["seed"]), int(row["sample_index"]))
        for row in runtime_samples
    ]
    require(len(runtime_keys) == len(set(runtime_keys)))
    samples_per_run = Counter(key[:3] for key in runtime_keys)
    require(len(samples_per_run) == 3 * 11 * 5 and set(samples_per_run.values()) == {121})
    compute_ms = np.asarray([float(row["compute_ms"]) for row in runtime_samples])
    require(np.isfinite(compute_ms).all() and (compute_ms >= 0).all())
    require(all(row["optimizer_success"].lower() == "true" for row in runtime_samples))
    for row in runtime_samples:
        sample_index = int(row["sample_index"])
        require(sample_index % 5 == 0)
        require(int(row["control_step"]) == sample_index // 5)
        assert_close(float(row["time_s"]), sample_index * config["plant_timestep_s"])
    runtime = {
        "sample_count": len(compute_ms),
        "mean_ms": float(np.mean(compute_ms)),
        "median_ms": float(np.median(compute_ms)),
        "p95_ms": float(np.percentile(compute_ms, 95)),
        "p99_ms": float(np.percentile(compute_ms, 99)),
        "max_ms": float(np.max(compute_ms)),
        "over_50_ms_percent": float(100 * np.mean(compute_ms > 50)),
        "over_100_ms_percent": float(100 * np.mean(compute_ms > 100)),
    }
    for name, value in runtime.items():
        assert_close(value, float(summary["runtime"][name]))
    require(summary["runtime"]["measurement"] == "MPC solver/control-computation time")
    assert_close(summary["runtime"]["control_interval_ms"], 50.0)
    expected_real_time = all(runtime[name] <= 50 for name in ("mean_ms", "p95_ms", "max_ms"))
    require(bool(summary["runtime"]["real_time_20_hz"]) == expected_real_time)

    require(len(virtual_quality) == 2 * 6 * 5)
    require({row["controller"] for row in virtual_quality} == {"C_sensor_MPC", "D_adaptive"})
    require({row["scenario"] for row in virtual_quality} == {
        "sensor_noise", "bias_5", "bias_15", "sensor_dropout", "sensor_drift", "combined_fault_load"
    })
    for row in virtual_quality:
        duration = float(row["substitution_duration_s"])
        require(duration >= 0)
        for field in ("virtual_feedback_RMSE", "corrupted_sensor_RMSE", "virtual_feedback_MAE", "corrupted_sensor_MAE"):
            value = float(row[field] or "nan")
            require((np.isfinite(value) and value >= 0) if duration else np.isnan(value))

    fallback = json.loads((METRICS / "pi_fallback_validation.json").read_text(encoding="utf-8"))
    require(fallback["forced_consecutive_updates"] == 10)
    require(fallback["max_abs_speed_rad_s"] <= 120)
    require(fallback["voltage_violations"] == fallback["slew_violations"] == 0)
    require(fallback["mpc_resumed"])
    require(fallback["resume_voltage_jump_v"] <= fallback["resume_slew_limit_v"] + 1e-8)

    with np.load(RAW / "final_representative_traces.npz") as traces:
        require(len(traces.files) > 0)
        require(all(np.isfinite(traces[name]).all() for name in traces.files if name.endswith(("__true", "__voltage"))))

    report = {
        **headline,
        **safety,
        **runtime,
        "real_time_20_hz": expected_real_time,
    }
    return report, summary, config


def verify_recovery_logic() -> None:
    config = json.loads((CONFIGS / "reliability_final_config.json").read_text(encoding="utf-8"))["sensor"]
    require(config["health_residual"] == "raw_y_measured - y_hat")
    require("CUSUM score" in config["recovery_conditions"])
    require(config["cusum_reset"] == "after complete persistent recovery only")
    monitor = SensorReliabilityMonitor(
        config["instant_threshold"], config["center"], config["allowance"],
        config["threshold"], config["enter_count"], config["exit_count"],
    )
    for _ in range(config["enter_count"]):
        monitor.update(3.0)
    require(monitor.active and monitor.positive > config["threshold"])
    evidence = monitor.positive
    for _ in range(config["exit_count"]):
        monitor.update(0.0)
    require(monitor.active and 0 < monitor.positive < evidence)
    while monitor.active:
        monitor.update(0.0)
    require(monitor.positive == monitor.negative == 0)

    notebook = (ROOT / "notebooks/06_final_evaluation.ipynb").read_text(encoding="utf-8")
    require("sensor.update(raw_y_measured - y_hat)" in notebook)
    require("y_feedback = y_hat if reliability" in notebook)
    require("event_mask &= TIME < event_end" in notebook)
    require("2.0 <= instant < 4.0" in notebook)

    regression = read_csv("recovery_regression_trace.csv")
    require(len(regression) == 4 * 300)
    require({row["scenario"] for row in regression} == {
        "persistent_constant_bias", "temporary_bias", "temporary_dropout", "combined_bias_load"
    })
    require({
        "raw_y_measured", "y_true", "y_hat", "residual", "positive_cusum",
        "negative_cusum", "active_fault_state", "healthy_run", "substitution_flag",
    } <= regression[0].keys())

    comparison = read_csv("bugfix_scenario_comparison.csv")
    require(len(comparison) == 4)
    lookup = {(row["phase"], row["scenario"]): row for row in comparison}
    require(float(lookup[("pre_fix", "combined_fault_load")]["false_recovery_events"]) == 1)
    require(float(lookup[("post_fix", "combined_fault_load")]["false_recovery_events"]) == 0)
    assert_close(
        float(lookup[("pre_fix", "sensor_drift")]["tracking_RMSE"]),
        float(lookup[("post_fix", "sensor_drift")]["tracking_RMSE"]),
    )

    baseline = ROOT / "results/baseline_prefix"
    require(all((baseline / name).stat().st_size > 0 for name in (
        "final_mpc_config.json", "final_summary.json", "final_controller_runs.csv",
        "final_controller_comparison.csv", "final_runtime_samples.csv",
        "final_representative_traces.npz", "FINAL_REPO_AUDIT_POSTFIX.md",
    )))


def verify_documentation(summary: dict, config: dict) -> None:
    documents = [
        ROOT / "README.md",
        ROOT / "PROJECT_PIPELINE_AND_STATUS.md",
        ROOT / "FINAL_REPO_AUDIT_POSTFIX.md",
    ]
    tokens = [
        "H=20", "Nc=2", "blocks=(5, 15)", "maxiter=8",
        *[f"{float(value):.6f}" for value in summary["headline_metrics"].values()],
        *[f"{float(summary['runtime'][name]):.6f}" for name in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms")],
    ]
    require(config["mpc"]["move_blocks"] == [5, 15])
    for document in documents:
        text = document.read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in text]
        require(not missing, (document.name, missing))
    for document in documents[:2]:
        text = document.read_text(encoding="utf-8")
        tokens = [f"{float(summary['runtime'][name]):.6f}" for name in ("over_50_ms_percent", "over_100_ms_percent")]
        require(not [token for token in tokens if token not in text])


def main() -> None:
    verify_recovery_logic()
    controller, summary, config = verify_controller_results()
    report = {
        "lstm": verify_dataset_and_model(),
        "controller": controller,
    }
    required_plots = [
        "final_nominal_step_tracking.png",
        "final_load_disturbance.png",
        "final_sensor_bias.png",
        "final_sensor_dropout.png",
        "final_combined_fault_load.png",
        "final_runtime_comparison.png",
        "final_runtime_distribution.png",
    ]
    require(all((PLOTS / name).stat().st_size > 0 for name in required_plots))
    verify_documentation(summary, config)
    print(json.dumps(report, indent=2))
    print("Saved-evidence verification passed.")


if __name__ == "__main__":
    main()
