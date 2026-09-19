from __future__ import annotations

import ast
import importlib.util
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = PROJECT / "scripts" / "verify_ekf_results.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_ekf_results_test_module", VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_closed_loop_matrix(verifier) -> pd.DataFrame:
    rows: list[dict] = []
    for controller, scenario, seed in product(
        verifier.CONTROLLERS,
        verifier.PRIMARY_SCENARIOS,
        verifier.DEVELOPMENT_SEEDS,
    ):
        fault_rmse = 10.0
        if scenario == "load_disturbance":
            if controller == "B_plain_MPC":
                fault_rmse = 10.0
            elif controller == "C3_arbitration_MPC":
                fault_rmse = 20.0
            elif controller == verifier.EKF_CONTROLLER:
                fault_rmse = 18.0
        elif scenario in {"sensor_bias_5", "sensor_dropout"}:
            if controller == "C3_arbitration_MPC":
                fault_rmse = 10.0
            elif controller == verifier.EKF_CONTROLLER:
                fault_rmse = 10.4
        elif scenario == "combined_fault_load":
            if controller == "C3_arbitration_MPC":
                fault_rmse = 10.0
            elif controller == verifier.EKF_CONTROLLER:
                fault_rmse = 8.5

        rows.append(
            {
                "controller": controller,
                "scenario": scenario,
                "seed": seed,
                "fault_window_rmse": fault_rmse,
                "sub_samples": 10.0 if controller == "C3_arbitration_MPC" else 0.0,
                "sub_fraction": 0.10 if controller == "C3_arbitration_MPC" else 0.0,
                "false_substitution_count": 9.0 if controller == verifier.EKF_CONTROLLER else 0.0,
                "false_substitution_fraction": 0.09 if controller == verifier.EKF_CONTROLLER else 0.0,
                "run_complete": True,
                "completed_samples": 601,
                "ekf_finite_sample_rate": 1.0,
                "ekf_numerical_failures": 0,
                "ekf_nonfinite_failures": 0,
                "ekf_psd_failures": 0,
                "ekf_symmetry_failures": 0,
                "ekf_failure_message": "",
                "voltage_violations": 0,
                "rate_violations": 0,
                "optimizer_failures": 0,
                "nonfinite_events": 0,
                "main_prediction_failures": 0,
                "aux_prediction_failures": 0,
                "mean_ekf_update_ms": 0.05,
                "median_ekf_update_ms": 0.04,
                "p95_ekf_update_ms": 0.08,
                "p99_ekf_update_ms": 0.10,
                "max_ekf_update_ms": 0.12,
                "total_reliability_observer_overhead_sample_count": 601,
                "mean_total_reliability_observer_overhead_ms": 0.10,
                "median_total_reliability_observer_overhead_ms": 0.09,
                "p95_total_reliability_observer_overhead_ms": 0.15,
                "p99_total_reliability_observer_overhead_ms": 0.18,
                "max_total_reliability_observer_overhead_ms": 0.20,
                "slsqp_solve_count": 121,
                "mean_slsqp_solve_ms": 2.0,
                "median_slsqp_solve_ms": 1.9,
                "p95_slsqp_solve_ms": 3.0,
                "p99_slsqp_solve_ms": 3.5,
                "max_slsqp_solve_ms": 4.0,
                "sim_time_s": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_verifier_has_no_bare_assert_statements() -> None:
    tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    bare_asserts = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert bare_asserts == []


def test_frozen_evidence_missing_file_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = load_verifier()
    monkeypatch.setattr(verifier, "FROZEN_EVIDENCE_PATH", tmp_path / "missing.json")
    with pytest.raises(verifier.VerificationError, match="required JSON exists"):
        verifier.verify_frozen_evidence()


def test_preregistered_grid_and_selection_are_exact() -> None:
    verifier = load_verifier()
    expected = set(product((1e-8, 1e-6, 1e-4), (1e-2, 1e-1, 1.0), (1e-8, 1e-6, 1e-4)))
    assert verifier.EXPECTED_GRID == expected
    assert len(verifier.EXPECTED_GRID) == 27

    rows = []
    for index, (q_dyn, m_t, r_value) in enumerate(product(verifier.Q_DYN_VALUES, verifier.M_T_VALUES, verifier.R_VALUES), start=1):
        rows.append(
            {
                "candidate_id": f"EKF_{index:02d}",
                "q_dyn": q_dyn,
                "m_T": m_t,
                "r": r_value,
                "stable": True,
                "validation_rmse": 2.0,
                "validation_mae": 2.0,
                "validation_abs_bias": 2.0,
            }
        )
    table = pd.DataFrame(rows)
    table.loc[0, ["validation_rmse", "validation_mae", "validation_abs_bias"]] = [1.0, 1.0, 1.0]
    # Candidate 02 is within 1% of the best RMSE and wins the preregistered
    # tie-break on MAE, proving selection is not simply argmin(RMSE).
    table.loc[1, ["validation_rmse", "validation_mae", "validation_abs_bias"]] = [1.005, 0.5, 0.5]
    selected = verifier._selected_candidate(table)
    assert selected.candidate_id == "EKF_02"


def test_calibration_provenance_rejects_fault_result_path() -> None:
    verifier = load_verifier()
    safe = {
        "dataset_path": "data/processed/dc_motor_lstm_dataset.npz",
        "candidate_table_path": "results/metrics/ekf_calibration_candidates.csv",
        "forbidden_selection_inputs": ["fault scenarios", "future EKF closed-loop outcomes"],
    }
    verifier.verify_no_fault_result_provenance(safe)

    bad = dict(safe)
    bad["fault_result_path"] = "results/metrics/ekf_closed_loop_runs.csv"
    with pytest.raises(verifier.VerificationError):
        verifier.verify_no_fault_result_provenance(bad)


def test_estimator_schema_encodes_clean_and_both_transfer_endpoints() -> None:
    verifier = load_verifier()
    base = verifier._estimator_metric_keys()
    assert ("EKF", "startup_pre_lstm_window_0_0p2s") in base
    assert ("auxiliary_LSTM", "startup_pre_lstm_window_0_0p2s") not in base
    assert len(base) == 23
    load = verifier._estimator_metric_keys("load_step_post_3s")
    parameter = verifier._estimator_metric_keys("parameter_shift_post_3s")
    assert len(load) == 25
    assert len(parameter) == 25
    assert ("EKF", "load_step_post_3s") in load
    assert ("auxiliary_LSTM", "parameter_shift_post_3s") in parameter


def test_closed_loop_matrix_requires_exact_100_paired_rows() -> None:
    verifier = load_verifier()
    runs = make_closed_loop_matrix(verifier)
    checked = verifier.verify_closed_loop_matrix(runs)
    assert len(checked) == 100
    assert set(zip(checked.controller, checked.scenario, checked.seed)) == verifier.expected_closed_loop_keys()

    missing = runs.iloc[:-1].copy()
    with pytest.raises(verifier.VerificationError, match="exactly 4 x 5 x 5"):
        verifier.verify_closed_loop_matrix(missing)

    forbidden = runs.copy()
    forbidden.loc[0, "seed"] = 39026
    with pytest.raises(verifier.VerificationError):
        verifier.verify_closed_loop_matrix(forbidden)


def test_ekf_run_evidence_requires_complete_runtime_distributions() -> None:
    verifier = load_verifier()
    runs = verifier.verify_closed_loop_matrix(make_closed_loop_matrix(verifier))
    verifier.verify_ekf_run_evidence(runs)

    missing_max = runs.drop(columns=["max_ekf_update_ms"])
    with pytest.raises(verifier.VerificationError, match="required columns"):
        verifier.verify_ekf_run_evidence(missing_max)


def test_closed_loop_plan_hash_and_runtime_contract_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = load_verifier()
    frozen = verifier.load_json(verifier.FROZEN_EVIDENCE_PATH)
    config_hash = verifier.sha256(verifier.EKF_CONFIG_PATH)
    expected_hash = verifier.verify_closed_loop_plan(frozen, config_hash)
    assert expected_hash == verifier.sha256(verifier.CLOSED_LOOP_PLAN_PATH)

    bad_plan = json.loads(verifier.CLOSED_LOOP_PLAN_PATH.read_text(encoding="utf-8"))
    bad_plan["runtime_accounting"]["per_sample_total_overhead"] = "ekf_update_ms only"
    bad_path = tmp_path / "ekf_closed_loop_plan.json"
    bad_path.write_text(json.dumps(bad_plan), encoding="utf-8")
    monkeypatch.setattr(verifier, "CLOSED_LOOP_PLAN_PATH", bad_path)
    with pytest.raises(verifier.VerificationError, match="runtime accounting"):
        verifier.verify_closed_loop_plan(frozen, config_hash)


def test_closed_loop_summary_rejects_wrong_plan_hash_before_accepting_results() -> None:
    verifier = load_verifier()
    runs = verifier.verify_closed_loop_matrix(make_closed_loop_matrix(verifier))
    frozen = verifier.load_json(verifier.FROZEN_EVIDENCE_PATH)
    config_hash = verifier.sha256(verifier.EKF_CONFIG_PATH)
    summary = {
        "evidence_role": "development_closed_loop_comparator",
        "controller": verifier.EKF_CONTROLLER,
        "development_seeds": list(verifier.DEVELOPMENT_SEEDS),
        "primary_scenarios": list(verifier.PRIMARY_SCENARIOS),
        "run_count": 100,
        "ekf_config_sha256": config_hash,
        "closed_loop_plan_sha256": "0" * 64,
    }
    with pytest.raises(verifier.VerificationError, match="immutable pre-run plan"):
        verifier.verify_closed_loop_summary(
            runs,
            summary,
            config_hash,
            verifier.sha256(verifier.CLOSED_LOOP_PLAN_PATH),
            frozen,
        )


def test_closed_loop_trace_recomputes_ekf_and_total_overhead_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = load_verifier()
    sample_count = 25
    steps = np.arange(sample_count)
    ekf_ms = 0.10 + 0.001 * steps
    aux_ms = np.where(steps >= 20, 0.20 + 0.001 * steps, 0.0)
    reliability_ms = np.where(steps >= 20, 0.03 + 0.0005 * steps, 0.0)
    total_ms = ekf_ms + aux_ms + reliability_ms
    slsqp_ms = np.where(steps % verifier.CONTROL_STRIDE == 0, 1.0 + 0.01 * steps, np.nan)

    ekf_stats = verifier._timing_distribution(ekf_ms)
    total_stats = verifier._timing_distribution(total_ms)
    slsqp_stats = verifier._timing_distribution(slsqp_ms[np.isfinite(slsqp_ms)])
    run = pd.Series(
        {
            "controller": verifier.EKF_CONTROLLER,
            "scenario": "sensor_bias_5",
            "seed": 19026,
            "completed_samples": sample_count,
            "mean_ekf_update_ms": ekf_stats["mean_ms"],
            "median_ekf_update_ms": ekf_stats["median_ms"],
            "p95_ekf_update_ms": ekf_stats["p95_ms"],
            "p99_ekf_update_ms": ekf_stats["p99_ms"],
            "max_ekf_update_ms": ekf_stats["max_ms"],
            "total_reliability_observer_overhead_sample_count": total_stats["sample_count"],
            "mean_total_reliability_observer_overhead_ms": total_stats["mean_ms"],
            "median_total_reliability_observer_overhead_ms": total_stats["median_ms"],
            "p95_total_reliability_observer_overhead_ms": total_stats["p95_ms"],
            "p99_total_reliability_observer_overhead_ms": total_stats["p99_ms"],
            "max_total_reliability_observer_overhead_ms": total_stats["max_ms"],
            "slsqp_solve_count": slsqp_stats["sample_count"],
            "mean_slsqp_solve_ms": slsqp_stats["mean_ms"],
            "median_slsqp_solve_ms": slsqp_stats["median_ms"],
            "p95_slsqp_solve_ms": slsqp_stats["p95_ms"],
            "p99_slsqp_solve_ms": slsqp_stats["p99_ms"],
            "max_slsqp_solve_ms": slsqp_stats["max_ms"],
        }
    )
    trace = pd.DataFrame(
        {
            "controller": verifier.EKF_CONTROLLER,
            "scenario": "sensor_bias_5",
            "seed": 19026,
            "step": steps,
            "time": steps * 0.01,
            "current": 0.0,
            "true": 0.0,
            "ekf": 0.0,
            "ekf_load": 0.0,
            "innovation": np.nan,
            "nis": np.nan,
            "min_cov_eigenvalue": 0.0,
            "cov_asymmetry": 0.0,
            "ekf_update_ms": ekf_ms,
            "aux_inference_ms": aux_ms,
            "reliability_update_ms": reliability_ms,
            "total_reliability_observer_overhead_ms": total_ms,
            "slsqp_solve_ms": slsqp_ms,
        }
    )
    trace_path = tmp_path / "ekf_closed_loop_traces.csv"
    trace.to_csv(trace_path, index=False)
    monkeypatch.setattr(verifier, "CLOSED_LOOP_TRACES_PATH", trace_path)
    verifier.verify_closed_loop_trace(pd.DataFrame([run]))

    tampered = trace.copy()
    tampered.loc[24, "total_reliability_observer_overhead_ms"] += 1.0
    tampered.to_csv(trace_path, index=False)
    with pytest.raises(verifier.VerificationError, match="total overhead equals"):
        verifier.verify_closed_loop_trace(pd.DataFrame([run]))


def test_role_gate_arithmetic_is_independent_and_exact() -> None:
    verifier = load_verifier()
    runs = verifier.verify_closed_loop_matrix(make_closed_loop_matrix(verifier))
    gate = verifier.recompute_role_gate(runs, calibration_locked=True)
    assert gate["decision"] == "POTENTIAL_SUCCESSOR_FOLLOW_UP"
    assert gate["all_successor_criteria_pass"] is True
    assert gate["isolated_sensor_faults"]["sensor_bias_5"]["paired_seed_passes"] == 5
    assert gate["isolated_sensor_faults"]["sensor_dropout"]["paired_seed_passes"] == 5
    assert gate["combined_fault_load"]["paired_seed_passes"] == 5
    assert gate["load_disturbance"]["paired_tracking_seed_passes"] == 5
    assert gate["load_disturbance"]["false_substitution_nonworse_seed_passes"] == 5

    tampered = dict(gate)
    tampered["decision"] = "BASELINE_ONLY"
    with pytest.raises(verifier.VerificationError):
        verifier._json_equal(tampered, gate, "role_gate")

    degraded = runs.copy()
    mask = (
        (degraded.controller == verifier.EKF_CONTROLLER)
        & (degraded.scenario == "combined_fault_load")
    )
    degraded.loc[mask, "fault_window_rmse"] = 9.5
    degraded_gate = verifier.recompute_role_gate(degraded, calibration_locked=True)
    assert degraded_gate["combined_fault_load"]["pass"] is False
    assert degraded_gate["decision"] == "BASELINE_ONLY"


def test_forbidden_seed_scan_catches_result_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = load_verifier()
    monkeypatch.setattr(verifier, "RESULTS", tmp_path)
    pd.DataFrame(
        [{"controller": verifier.EKF_CONTROLLER, "scenario": "sensor_bias_5", "seed": 39026}]
    ).to_csv(tmp_path / "ekf_bad_result.csv", index=False)
    with pytest.raises(verifier.VerificationError, match="39026-39030"):
        verifier.verify_no_forbidden_seed_result_use()

    (tmp_path / "ekf_bad_result.csv").unlink()
    pd.DataFrame(
        [{"controller": verifier.EKF_CONTROLLER, "scenario": "sensor_bias_5", "seed": 19026}]
    ).to_csv(tmp_path / "ekf_good_result.csv", index=False)
    verifier.verify_no_forbidden_seed_result_use()
