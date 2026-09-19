"""Fail-loud independent verifier for the frozen 11-scenario V3 evidence."""

import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from auxiliary_sensor_model import build_auxiliary_sequences, load_auxiliary_model, normalize_auxiliary
from data_utils import build_training_sequences
from generate_v3_report import build_c2_ablation, build_report
from reliability import DualVirtualSensorArbitrator, load_lstm_model

EXPECTED_SEEDS = set(range(29026, 29031))
EXPECTED_CONTROLLERS = {"A_PI", "B_plain_MPC", "C1_sensor_MPC", "C2_aux_recovery_MPC", "C3_arbitration_MPC"}
EXPECTED_SCENARIOS = {
    "nominal_tracking", "step_reference", "changing_reference", "sensor_noise",
    "sensor_bias_5", "sensor_bias_15", "sensor_dropout", "sensor_drift",
    "load_disturbance", "parameter_variation", "combined_fault_load",
}
REFERENCE_SCENARIOS = {"nominal_tracking", "step_reference", "changing_reference"}
FROZEN_SCENARIOS = EXPECTED_SCENARIOS - REFERENCE_SCENARIOS
MODEL_RTOL = 1e-6
MODEL_ATOL = 1e-5
EXPECTED_HASHES = {
    "main_model": "d1f4178199f682560f164ccac1d56816aae341eb05fbed6ac7ad73a4e2056297",
    "auxiliary_model": "619f677ccc63c6a40c863858d8eee4175f72b1770dc6c8ba202e7b92c2768480",
    "v3_arbitration_config": "7b4f86934ece8d426148ff968b0152a5fac6cfa321615b229bfd35c63a179b35",
    "v3_calibration": "1dfcedf45110971e776c9c9ec6639b556b2c058d3a0535d9a69f5e23be99b1c8",
    "original_200_runs": "160a114a77709d73b3503921dd672000a05bfdc918838b0d13619ac8a0fbcddd",
}
EXPECTED_MAIN_CONFIG_SHA256 = "9254bbd088b0c9f7bd7ba71ee75c09fe4501ce28eb8e26efab7fb7d5130ce4cc"
EXPECTED_CANONICAL_275_SHA256 = "a919e0b16321765fb1be5f97c15dc61062ccea737cd4059aed887ca6aa647086"
EXPECTED_VERIFICATION_REPORT_SHA256 = "c1d5987788fd8750ac0f17f80559687970a7a59fbdbb7c0d1a71a4e048480106"
FROZEN_HASH_RECORD = PROJECT / "results/configs/c4_v3_frozen_hashes.json"
CANONICAL_275_PATH = PROJECT / "results/metrics/v3_final_11scenario_runs.csv"
EXPECTED_THRESHOLDS = {
    "agreement_threshold": 4.341867446899414,
    "param_mismatch_threshold": 8.700958862330452,
    "param_mismatch_recovery_threshold": 5.0486354952328725,
    "aux_recovery_gate": 9.047693252563477,
    "aux_speed_bounds": [0.0, 100.0],
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independently_post_blanking_ewma(times, residual, alpha, startup_blanking_time):
    """Reproduce deployed startup semantics without sharing calibration implementation."""
    ewma = 0.0
    values = []
    for time_value, residual_value in zip(times, residual):
        if time_value >= startup_blanking_time:
            ewma = (1 - alpha) * ewma + alpha * float(residual_value)
            values.append(ewma)
    return np.asarray(values, dtype=float)


def verify_recorded_artifact(
    record: dict,
    artifact_name: str,
    actual_path: Path,
    expected_relative_path: str,
    expected_sha256: str | None = None,
) -> str:
    """Bind a local artifact to its pre-recorded path/hash and optional fixed hash."""
    artifact = record.get("artifacts", {}).get(artifact_name)
    if not isinstance(artifact, dict):
        raise AssertionError(f"freeze record is missing {artifact_name!r}")
    if artifact.get("path") != expected_relative_path:
        raise AssertionError(f"freeze record path for {artifact_name!r} is not canonical")
    actual_sha256 = sha256(actual_path)
    if artifact.get("sha256") != actual_sha256:
        raise AssertionError(f"{artifact_name!r} does not match its freeze-record SHA256")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise AssertionError(f"{artifact_name!r} does not match the expected frozen SHA256")
    return actual_sha256


@torch.no_grad()
def model_prediction(model, inputs, mean, std):
    chunks = [model(torch.from_numpy(inputs[start : start + 1024]))[:, 0].numpy() for start in range(0, len(inputs), 1024)]
    return np.concatenate(chunks) * std + mean


def independently_recompute_calibration(
    data,
    validation_ids,
    ewma_alpha,
    startup_blanking_time,
):
    main, main_config = load_lstm_model(PROJECT / "results/lstm_model_weights.pt", PROJECT / "results/configs/lstm_model_config.json")
    auxiliary, aux_config = load_auxiliary_model(PROJECT / "results/auxiliary_model_weights.pt", PROJECT / "results/configs/v2_auxiliary_config.json")
    main_norm, aux_norm = main_config["normalization"], aux_config["normalization"]
    agreement, aux_residual, ewma_values = [], [], []
    for run_id in validation_ids:
        trajectory = {
            "voltage": data["voltage"][run_id],
            "y_measured": data["y_measured"][run_id],
            "y_true": data["y_true"][run_id],
            "run_id": np.full(len(data["time"]), run_id),
        }
        main_inputs, _, _ = build_training_sequences(trajectory, 20, 1)
        main_inputs = ((main_inputs - np.asarray(main_norm["input_mean"])) / np.asarray(main_norm["input_std"])).astype(np.float32)
        aux_inputs, targets = build_auxiliary_sequences(data["voltage"][run_id], data["current"][run_id], data["y_true"][run_id], 20)
        aux_inputs, _ = normalize_auxiliary(aux_inputs, targets, aux_norm)
        y_main = model_prediction(main, main_inputs, main_norm["target_mean"][0], main_norm["target_std"][0])
        y_aux = model_prediction(auxiliary, aux_inputs, aux_norm["target_mean"][0], aux_norm["target_std"][0])
        residual = np.abs(data["y_measured"][run_id, 20:] - y_aux)
        agreement.extend(np.abs(y_main - y_aux))
        aux_residual.extend(residual)
        ewma_values.extend(
            independently_post_blanking_ewma(
                data["time"][20:], residual, ewma_alpha, startup_blanking_time
            )
        )
    return {
        "agreement_threshold": float(np.percentile(agreement, 90)),
        "param_mismatch_threshold": float(np.percentile(ewma_values, 99.9)),
        "param_mismatch_recovery_threshold": float(np.percentile(ewma_values, 95)),
        "aux_recovery_gate": float(np.percentile(aux_residual, 99.9)),
    }


def verify_reconstruction_equivalence(recalibrated, calibration_values, config_values):
    """Check reconstruction numerically and return local diagnostic differences."""
    observed_differences = {}
    for name in sorted(recalibrated):
        value = recalibrated[name]
        calibration_value = calibration_values[name]
        config_value = config_values[name]
        observed_differences[name] = abs(value - calibration_value)
        calibration_matches = bool(
            np.isclose(value, calibration_value, rtol=MODEL_RTOL, atol=MODEL_ATOL)
        )
        config_matches = bool(
            np.isclose(value, config_value, rtol=MODEL_RTOL, atol=MODEL_ATOL)
        )
        check(calibration_matches, f"portably reproduced model-derived {name}")
        check(config_matches, f"frozen config numerically matches model-derived {name}")
    return observed_differences


def main() -> None:
    torch.set_num_threads(1)
    metrics = PROJECT / "results/metrics"
    config_path = PROJECT / "results/configs/v3_arbitration_config.json"
    calibration_path = PROJECT / "results/configs/v3_arbitration_calibration.json"
    main_config_path = PROJECT / "results/configs/lstm_model_config.json"
    config = json.loads(config_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    frozen_hash_record = json.loads(FROZEN_HASH_RECORD.read_text())
    summary = json.loads((metrics / "v3_final_11scenario_summary.json").read_text())
    check(frozen_hash_record.get("recorded_before_c4_implementation") is True, "V3 freeze record predates C4 implementation")
    canonical_matrix_hash = verify_recorded_artifact(
        frozen_hash_record,
        "v3_final_275_run_matrix",
        CANONICAL_275_PATH,
        "results/metrics/v3_final_11scenario_runs.csv",
        EXPECTED_CANONICAL_275_SHA256,
    )
    check(canonical_matrix_hash == EXPECTED_CANONICAL_275_SHA256, "canonical 275-run matrix matches the pre-C4 frozen SHA256")
    runs = pd.read_csv(CANONICAL_275_PATH, float_precision="round_trip")
    comparison = pd.read_csv(metrics / "v3_final_11scenario_summary.csv", float_precision="round_trip")
    reference_runs = pd.read_csv(metrics / "v3_final_reference_runs.csv", float_precision="round_trip")
    original_runs = pd.read_csv(metrics / "v3_final_holdout_runs.csv", float_precision="round_trip")
    diagnostics = pd.read_csv(metrics / "c2_recovery_gate_diagnostics.csv")
    reference_events = pd.read_csv(metrics / "v3_reference_substitution_events.csv", float_precision="round_trip")
    verification_report_path = metrics / "v3_verification_report.json"
    check(
        sha256(verification_report_path) == EXPECTED_VERIFICATION_REPORT_SHA256,
        "frozen V3 verification report bytes are unchanged",
    )
    json.loads(verification_report_path.read_text())

    actual_hashes = {
        "main_model": sha256(PROJECT / "results/lstm_model_weights.pt"),
        "auxiliary_model": sha256(PROJECT / "results/auxiliary_model_weights.pt"),
        "v3_arbitration_config": sha256(config_path),
        "v3_calibration": sha256(calibration_path),
        "original_200_runs": sha256(metrics / "v3_final_holdout_runs.csv"),
    }
    check(actual_hashes == EXPECTED_HASHES == summary["frozen_artifact_sha256"], "all frozen model/config/calibration/original-run SHA256 hashes match exactly")
    check(calibration["model_sha256"]["main"] == EXPECTED_HASHES["main_model"], "main model hash matches calibration provenance")
    check(calibration["model_sha256"]["auxiliary"] == EXPECTED_HASHES["auxiliary_model"], "auxiliary model hash matches calibration provenance")
    main_config_hash = sha256(main_config_path)
    frozen_main_config = frozen_hash_record.get("artifacts", {}).get("main_config")
    if frozen_main_config is not None:
        verify_recorded_artifact(
            frozen_hash_record,
            "main_config",
            main_config_path,
            "results/configs/lstm_model_config.json",
            EXPECTED_MAIN_CONFIG_SHA256,
        )
        check(True, "main LSTM config matches its freeze-record SHA256")
    else:
        check(main_config_hash == EXPECTED_MAIN_CONFIG_SHA256, "main LSTM config matches the independently frozen V3 SHA256")
    historical_main_config_hash = calibration["model_sha256"].get("main_config")
    if historical_main_config_hash is not None:
        check(historical_main_config_hash == main_config_hash, "calibration provenance binds the main LSTM config")
    else:
        print("PASS: historical frozen calibration predates the main-config provenance field; current config verified independently")
    check(config["calibration_sha256"] == EXPECTED_HASHES["v3_calibration"] == summary["calibration_sha256"], "arbitration config binds the exact calibration artifact")
    check(all(config["arbitrator"][key] == value for key, value in EXPECTED_THRESHOLDS.items()), "frozen thresholds and physical speed envelope are unchanged")
    check(set(config["scenarios"]) == FROZEN_SCENARIOS, "frozen eight-scenario arbitration config is unchanged")

    data_path = PROJECT / "data/processed/dc_motor_lstm_dataset.npz"
    data = np.load(data_path)
    validation_ids = [int(value) for value in data["split_run_ids_validation"]]
    check(sha256(data_path) == calibration["dataset_sha256"] and validation_ids == calibration["validation_run_ids"], "calibration dataset and validation-run provenance match exactly")
    check(EXPECTED_SEEDS.isdisjoint(validation_ids) and set(calibration["excluded_final_holdout_seeds"]) == EXPECTED_SEEDS, "final seeds are excluded from calibration")
    snapshot_text = "\n".join(path.read_text(errors="ignore") for path in (PROJECT / "results/baseline_v3_prefix").rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md", ".py", ".txt"})
    seed_pattern = rf"(?<![\d.])(?:{'|'.join(map(str, sorted(EXPECTED_SEEDS)))})(?![\d.])"
    check(re.search(seed_pattern, snapshot_text) is None, "final seeds were absent from preserved pre-fix evidence")

    recalibrated = independently_recompute_calibration(
        data,
        validation_ids,
        config["arbitrator"]["ewma_alpha"],
        config["arbitrator"]["startup_blanking_time"],
    )
    differences = verify_reconstruction_equivalence(
        recalibrated,
        calibration["computed_values"],
        config["arbitrator"],
    )

    expected_keys = pd.MultiIndex.from_product([EXPECTED_CONTROLLERS, EXPECTED_SCENARIOS, EXPECTED_SEEDS], names=["controller", "scenario", "seed"])
    actual_keys = pd.MultiIndex.from_frame(runs[["controller", "scenario", "seed"]])
    check(len(runs) == summary["run_count"] == 275, "final row count is exactly 275")
    check(set(runs.controller) == EXPECTED_CONTROLLERS and set(runs.scenario) == EXPECTED_SCENARIOS and set(runs.seed) == EXPECTED_SEEDS, "controller, scenario, and seed sets match exactly")
    check(not actual_keys.duplicated().any(), "controller/scenario/seed keys contain no duplicates")
    check(set(actual_keys) == set(expected_keys), "all 275 controller/scenario/seed combinations exist")

    canonical_old = runs[runs.scenario.isin(FROZEN_SCENARIOS)][original_runs.columns].sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)
    saved_old = original_runs.sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(canonical_old, saved_old, check_exact=True)
    check(True, "the original 200 frozen rows are numerically unchanged")
    canonical_reference = runs[runs.scenario.isin(REFERENCE_SCENARIOS)][reference_runs.columns].sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        canonical_reference,
        reference_runs.sort_values(["scenario", "seed", "controller"]).reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
    )
    check(len(reference_runs) == 75, "the saved reference artifact contains exactly the 75 rows extracted from the hash-verified canonical matrix")

    aggregate_metrics = [column[:-5] for column in comparison if column.endswith("_mean")]
    rebuilt = runs.groupby(["scenario", "controller"])[aggregate_metrics].agg(["mean", "std", "median"])
    rebuilt.columns = [f"{metric}_{stat}" for metric, stat in rebuilt.columns]
    rebuilt = rebuilt.reset_index().sort_values(["scenario", "controller"]).reset_index(drop=True)
    saved = comparison.sort_values(["scenario", "controller"]).reset_index(drop=True)
    check(saved[["scenario", "controller"]].equals(rebuilt[["scenario", "controller"]]), "summary keys match independently rebuilt groups")
    check(np.allclose(saved.drop(columns=["scenario", "controller"]), rebuilt.drop(columns=["scenario", "controller"]), equal_nan=True), "summary means, sample standard deviations, and medians reproduce")

    count_columns = ["optimizer_failures", "voltage_violations", "rate_violations", "main_prediction_failures", "aux_prediction_failures", "nonfinite_events"]
    totals = runs[count_columns].sum().astype(int)
    check((totals == 0).all(), "optimizer, prediction, voltage, slew, and nonfinite event counts are exactly zero")
    check(all(summary[column] == int(totals[column]) for column in count_columns), "summary failure and violation counts match exactly")
    check(np.allclose(runs[["frac_phys", "frac_main", "frac_aux", "frac_fb"]].sum(axis=1), 1.0), "source fractions sum to one per run")

    combined = runs[runs.scenario == "combined_fault_load"].pivot(index="seed", columns="controller", values="fault_window_rmse")
    check((combined["C3_arbitration_MPC"] < combined["B_plain_MPC"]).all(), "combined-fault C3 improvement over plain MPC is paired-consistent")
    drift = runs[runs.scenario == "sensor_drift"].pivot(index="seed", columns="controller", values="fault_window_rmse")
    check((drift["C3_arbitration_MPC"] - drift["B_plain_MPC"]).abs().max() < 0.001, "drift evidence still supports no material improvement")
    load = runs[(runs.scenario == "load_disturbance") & (runs.controller == "C3_arbitration_MPC")]
    check(load.reliability_entries.notna().all() and np.array_equal(load.reliability_entries, load.reliability_entries.astype(int)), "load-disturbance false sensor-fault entry counts are present and integral")
    reference_c3 = runs[(runs.scenario.isin(REFERENCE_SCENARIOS)) & (runs.controller == "C3_arbitration_MPC")]
    check(reference_c3[["sub_fraction", "sub_samples", "reliability_entries", "switches", "frac_phys", "frac_main", "frac_aux", "frac_fb"]].notna().all().all(), "nominal/reference false-substitution and source-selection metrics are complete")
    expected_event_counts = reference_c3.groupby("scenario").sub_samples.sum().astype(int).sort_index()
    actual_event_counts = reference_events.groupby("scenario").size().reindex(expected_event_counts.index, fill_value=0).sort_index()
    check(expected_event_counts.equals(actual_event_counts), "reference substitution event rows reproduce exact per-scenario substitution counts")
    check(not reference_events.reliability_active.any(), "all reference substitutions were instantaneous guards, not latched reliability entries")

    equivalence_metrics = [
        "overall_rmse", "overall_mae", "fault_window_rmse", "recovery_events",
        "sub_duration_s", "sub_fraction", "frac_phys", "frac_main", "frac_fb",
        "control_effort_u2", "control_variation_du2", "optimizer_failures",
        "voltage_violations", "rate_violations", "nonfinite_events",
    ]
    c1 = runs[runs.controller == "C1_sensor_MPC"].sort_values(["scenario", "seed"])
    c2 = runs[runs.controller == "C2_aux_recovery_MPC"].sort_values(["scenario", "seed"])
    check(np.array_equal(c1[equivalence_metrics].to_numpy(), c2[equivalence_metrics].to_numpy(), equal_nan=True), "C2 and C1 behavioral metrics are exactly identical in all 55 paired cases")

    required_diagnostic_columns = {
        "scenario", "seed", "step", "time_s", "residual_condition", "cusum_condition",
        "persistence_condition", "auxiliary_condition", "blocked_by_residual",
        "blocked_by_cusum", "blocked_by_persistence", "blocked_only_by_auxiliary",
        "recovered_this_step",
    }
    check(required_diagnostic_columns.issubset(diagnostics.columns), "C2 diagnostics record every requested recovery condition")
    diagnostic_summary = summary["c2_recovery_gate_diagnostics"]
    expected_counts = {
        "recovery_opportunities": len(diagnostics),
        "blocked_by_residual": int(diagnostics.blocked_by_residual.sum()),
        "blocked_by_cusum": int(diagnostics.blocked_by_cusum.sum()),
        "blocked_by_persistence": int(diagnostics.blocked_by_persistence.sum()),
        "blocked_only_by_auxiliary": int(diagnostics.blocked_only_by_auxiliary.sum()),
        "recoveries": int(diagnostics.recovered_this_step.sum()),
    }
    check(all(diagnostic_summary[key] == value for key, value in expected_counts.items()), "C2 diagnostic totals reproduce exactly")
    check(diagnostic_summary["blocked_only_by_auxiliary"] == 0 and not diagnostic_summary["auxiliary_gate_delayed_recovery_relative_to_c1"], "auxiliary recovery gate was never uniquely binding and never delayed recovery")
    check(diagnostic_summary["c2_behaviorally_identical_to_c1"], "saved C2 ablation interpretation matches the evidence")

    update_source = inspect.getsource(DualVirtualSensorArbitrator.update)
    check("y_true" not in update_source and "true_speed" not in update_source, "online arbitrator has no ground-truth input")
    check((PROJECT / "V3_DUAL_VIRTUAL_SENSOR_EXTENSION.md").read_text(encoding="utf-8") == build_report(), "V3 report reproduces from canonical artifacts")
    check((PROJECT / "C2_ABLATION_ANALYSIS.md").read_text(encoding="utf-8") == build_c2_ablation(), "C2 ablation report reproduces from canonical artifacts")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    pipeline = (PROJECT / "PROJECT_PIPELINE_AND_STATUS.md").read_text(encoding="utf-8")
    check("275 paired runs" in readme and "275 paired runs" in pipeline, "README and pipeline status use the canonical run count")
    snapshot = PROJECT / "results/baseline_v3_prefix"
    check(all((snapshot / relative).exists() for relative in ["metrics/v3_summary.json", "metrics/v2_closed_loop_summary.json", "metrics/final_summary.json", "configs/v3_arbitration_config.json", "reports/README.md", "verifiers/verify_results.py"]), "preserved V1/V2/V3 snapshot remains complete")
    print(f"\nMaximum calibration reconstruction difference: {max(differences.values()):.12g}")
    print("V3 VERIFIER RESULT: PASS")


if __name__ == "__main__":
    main()
