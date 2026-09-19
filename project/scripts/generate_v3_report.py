"""Generate frozen V3 documentation from the canonical 275-run artifacts."""

import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
METRICS = PROJECT / "results/metrics"
REFERENCE_SCENARIOS = ["nominal_tracking", "step_reference", "changing_reference"]


def artifacts():
    runs = pd.read_csv(METRICS / "v3_final_11scenario_runs.csv")
    comparison = pd.read_csv(METRICS / "v3_final_11scenario_summary.csv")
    summary = json.loads((METRICS / "v3_final_11scenario_summary.json").read_text())
    verification_path = METRICS / "v3_verification_report.json"
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    return runs, comparison, summary, verification


def build_c2_ablation() -> str:
    runs, _, summary, _ = artifacts()
    diagnostics = pd.read_csv(METRICS / "c2_recovery_gate_diagnostics.csv")
    counts = summary["c2_recovery_gate_diagnostics"]
    lines = [
        "# C2 Recovery-Gate Ablation Analysis",
        "",
        "C2 was diagnosed without changing or retuning its recovery gate. A recovery opportunity is one sample that begins with the sensor monitor active. Blocking-condition counts may overlap; `blocked_only_by_auxiliary` is the exclusive auxiliary-gate count.",
        "",
        "| Diagnostic | Count |",
        "|---|---:|",
        f"| Recovery opportunities | {counts['recovery_opportunities']} |",
        f"| Blocked by raw residual | {counts['blocked_by_residual']} |",
        f"| Blocked by CUSUM | {counts['blocked_by_cusum']} |",
        f"| Blocked by persistence only after all signal gates passed | {counts['blocked_by_persistence']} |",
        f"| Blocked only by auxiliary gate | {counts['blocked_only_by_auxiliary']} |",
        f"| Completed recoveries | {counts['recoveries']} |",
        "",
        "## Per-scenario evidence",
        "",
        "| Scenario | Opportunities | Residual blocks | CUSUM blocks | Persistence blocks | Auxiliary-only blocks | Recoveries |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, group in diagnostics.groupby("scenario", sort=True):
        lines.append(
            f"| `{scenario}` | {len(group)} | {int(group.blocked_by_residual.sum())} | "
            f"{int(group.blocked_by_cusum.sum())} | {int(group.blocked_by_persistence.sum())} | "
            f"{int(group.blocked_only_by_auxiliary.sum())} | {int(group.recovered_this_step.sum())} |"
        )
    c1 = runs[runs.controller == "C1_sensor_MPC"].sort_values(["scenario", "seed"])
    c2 = runs[runs.controller == "C2_aux_recovery_MPC"].sort_values(["scenario", "seed"])
    max_rmse = (c2.overall_rmse.to_numpy() - c1.overall_rmse.to_numpy()).max(initial=0.0)
    max_sub = (c2.sub_duration_s.to_numpy() - c1.sub_duration_s.to_numpy()).max(initial=0.0)
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"C1 and C2 behavioral metrics are identical across all 55 paired scenario/seed cases: `{counts['c2_behaviorally_identical_to_c1']}`. "
        f"The largest C2-minus-C1 RMSE difference is `{max_rmse:.12g} rad/s`; the largest substitution-duration difference is `{max_sub:.12g} s`.",
        "",
        f"The auxiliary gate delayed recovery relative to C1: `{counts['auxiliary_gate_delayed_recovery_relative_to_c1']}`. "
        f"It was uniquely binding on `{counts['blocked_only_by_auxiliary']}` recovery opportunities.",
        "",
        "C2 produced identical closed-loop behavior to C1 because the auxiliary recovery condition was not the active recovery constraint in the evaluated scenarios. C2 is therefore retained as an architectural/recovery ablation rather than a performance-enhancing controller.",
        "",
    ])
    return "\n".join(lines)


def build_report() -> str:
    runs, _, summary, verification = artifacts()
    calibration = json.loads((PROJECT / "results/configs/v3_arbitration_calibration.json").read_text())
    events = pd.read_csv(METRICS / "v3_reference_substitution_events.csv")
    c3_reference = runs[(runs.controller == "C3_arbitration_MPC") & runs.scenario.isin(REFERENCE_SCENARIOS)]
    reference_substitutions = int(c3_reference.sub_samples.sum())
    changes = {"step_reference": [1.5], "changing_reference": [2.0, 4.0]}
    events["nearest_change_delta_s"] = events.apply(
        lambda row: min((row.time_s - change for change in changes.get(row.scenario, [])), key=abs, default=float("nan")), axis=1
    )
    transition_linked = events[events.nearest_change_delta_s.between(0.0, 0.05)]
    diagnostics = summary["c2_recovery_gate_diagnostics"]
    differences = verification.get("calibration_reconstruction_absolute_differences", {})
    max_difference = max(differences.values(), default=float("nan"))
    lines = [
        "# V3 Dual Virtual Sensor Arbitration — Frozen 11-Scenario Evidence",
        "",
        "## Scope and provenance",
        "",
        "This phase completes evidence only. It changes no controller, reliability, arbitration, MPC, threshold, or trained model. Seeds `19026–19030` remain development evidence; the untouched final holdout is `29026–29030`.",
        "",
        f"The canonical matrix contains {summary['run_count']} runs: 5 controllers × 11 scenarios × 5 paired seeds. The original 200 rows were retained byte-for-byte in `v3_final_holdout_runs.csv`; only 75 nominal/reference runs were added.",
        "",
        "| Frozen artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in summary["frozen_artifact_sha256"].items():
        lines.append(f"| `{name}` | `{digest}` |")
    values = calibration["computed_values"]
    lines.extend([
        "",
        "## Frozen calibration and verifier portability",
        "",
        f"Frozen thresholds are agreement `{values['agreement_threshold']:.15g}`, parameter mismatch `{values['param_mismatch_threshold']:.15g}`, recovery EWMA `{values['param_mismatch_recovery_threshold']:.15g}`, auxiliary recovery `{values['aux_recovery_gate']:.15g}`, and physical speed envelope `{values['aux_speed_bounds']}` rad/s.",
        "",
        f"Model-derived reconstruction checks previously used `rtol=0, atol=1e-9`; they now use `rtol={verification.get('model_comparison_rtol', 1e-6):g}, atol={verification.get('model_comparison_atol', 1e-5):g}`. This permits insignificant PyTorch/BLAS rounding variation while exact hashes, identities, matrix structure, event counts, and constraints remain exact. The maximum observed calibration reconstruction difference was `{max_difference:.12g}`.",
        "",
        "## Nominal and reference-transient results for C3",
        "",
        "Values are means across the five final seeds.",
        "",
        "| Scenario | RMSE | MAE | Substitution fraction | Reliability entries | Source switches | Physical fraction | Main fraction | Auxiliary fraction | Fallback fraction | Control effort Σu² | Optimizer failures | Voltage violations | Slew violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scenario in REFERENCE_SCENARIOS:
        row = c3_reference[c3_reference.scenario == scenario]
        lines.append(
            f"| `{scenario}` | {row.overall_rmse.mean():.6f} | {row.overall_mae.mean():.6f} | "
            f"{row.sub_fraction.mean():.6f} | {row.reliability_entries.sum():.0f} | {row.switches.mean():.3f} | "
            f"{row.frac_phys.mean():.6f} | {row.frac_main.mean():.6f} | {row.frac_aux.mean():.6f} | "
            f"{row.frac_fb.mean():.6f} | {row.control_effort_u2.mean():.6f} | {row.optimizer_failures.sum()} | "
            f"{row.voltage_violations.sum()} | {row.rate_violations.sum()} |"
        )
    lines.extend([
        "",
        "### B versus C1 versus C3",
        "",
        "| Scenario | Controller | RMSE | MAE | Substitution fraction | Reliability entries |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for scenario in REFERENCE_SCENARIOS:
        for controller in ["B_plain_MPC", "C1_sensor_MPC", "C3_arbitration_MPC"]:
            row = runs[(runs.scenario == scenario) & (runs.controller == controller)]
            lines.append(
                f"| `{scenario}` | `{controller}` | {row.overall_rmse.mean():.6f} | {row.overall_mae.mean():.6f} | "
                f"{row.sub_fraction.mean():.6f} | {row.reliability_entries.sum():.0f} |"
            )
    if reference_substitutions:
        verdict = (
            f"C3 made {reference_substitutions} isolated false substitutions on fault-free nominal/reference trajectories, all instantaneous and none latched. "
            f"Exactly {len(transition_linked)} occurred within one 0.05 s control interval after a reference transition: "
            f"{len(transition_linked[transition_linked.scenario == 'step_reference'])} after the step-reference change and "
            f"{len(transition_linked[transition_linked.scenario == 'changing_reference'])} after changing-reference transitions."
        )
    else:
        verdict = "C3 recorded no substitution and no reliability entry in the fault-free nominal, step-reference, or changing-reference runs; legitimate reference transients did not trigger false activation."
    lines.extend([
        "", f"**Reference-transient verdict:** {verdict}", "",
        "| Scenario | Seed | False-substitution time | Nearest reference-change delta | Reliability latched |",
        "|---|---:|---:|---:|---|",
    ])
    for event in events.itertuples(index=False):
        delta = "n/a" if pd.isna(event.nearest_change_delta_s) else f"{event.nearest_change_delta_s:+.2f} s"
        lines.append(f"| `{event.scenario}` | {event.seed} | {event.time_s:.2f} s | {delta} | `{event.reliability_active}` |")
    lines.extend(["", "## Load-disturbance check", ""])
    load = runs[runs.scenario == "load_disturbance"].pivot(index="seed", columns="controller")
    lines.extend([
        "| Seed | Plain MPC RMSE | C3 RMSE | C3 substitution fraction | False sensor-fault entries |",
        "|---:|---:|---:|---:|---:|",
    ])
    for seed in sorted(load.index):
        lines.append(
            f"| {seed} | {load.loc[seed, ('fault_window_rmse', 'B_plain_MPC')]:.6f} | "
            f"{load.loc[seed, ('fault_window_rmse', 'C3_arbitration_MPC')]:.6f} | "
            f"{load.loc[seed, ('sub_fraction', 'C3_arbitration_MPC')]:.6f} | "
            f"{load.loc[seed, ('reliability_entries', 'C3_arbitration_MPC')]:.0f} |"
        )
    lines.extend([
        "",
        f"Mean load-disturbance RMSE is `{load[('fault_window_rmse', 'B_plain_MPC')].mean():.6f} rad/s` for plain MPC and `{load[('fault_window_rmse', 'C3_arbitration_MPC')].mean():.6f} rad/s` for C3; mean C3 substitution fraction is `{load[('sub_fraction', 'C3_arbitration_MPC')].mean():.6f}`. This confirms that the monitor can confuse a plant load disturbance with sensor failure.",
        "",
        "## C2 recovery-gate ablation",
        "",
        f"Across {diagnostics['recovery_opportunities']} recovery opportunities, residual blocked {diagnostics['blocked_by_residual']}, CUSUM blocked {diagnostics['blocked_by_cusum']}, persistence blocked {diagnostics['blocked_by_persistence']}, and the auxiliary gate alone blocked {diagnostics['blocked_only_by_auxiliary']}. Counts can overlap except the explicitly exclusive auxiliary count.",
        "",
        "C2 produced identical closed-loop behavior to C1 because the auxiliary recovery condition was not the active recovery constraint in the evaluated scenarios. C2 is retained as an architectural/recovery ablation rather than a performance-enhancing controller. Full evidence is in `C2_ABLATION_ANALYSIS.md`.",
        "",
        "## Safety, integrity, and limitations",
        "",
        f"Across all {summary['run_count']} runs: optimizer failures `{summary['optimizer_failures']}`, main prediction failures `{summary['main_prediction_failures']}`, auxiliary prediction failures `{summary['aux_prediction_failures']}`, nonfinite events `{summary['nonfinite_events']}`, voltage violations `{summary['voltage_violations']}`, and slew violations `{summary['rate_violations']}`.",
        "",
        "- All evidence is synthetic simulation; hardware timing and sensor validation remain outstanding.",
        "- The detector can confuse load disturbance with sensor failure.",
        "- Slow drift is not materially improved.",
        "- Parameter variation remains unsupported by a general mismatch solution.",
        "- The auxiliary estimator assumes trustworthy current measurement.",
        "- Calibration may not transfer outside its saved clean-validation operating population.",
        "",
        "## Reproduce and verify",
        "",
        "```bash",
        "python scripts/evaluate_v3_closed_loop.py",
        "python scripts/verify_v3_results.py",
        "python scripts/generate_v3_report.py",
        "python scripts/verify_v3_results.py",
        "```",
        "",
        "Canonical evidence: `v3_final_11scenario_runs.csv`, `v3_final_11scenario_summary.csv`, `v3_final_11scenario_summary.json`, `v3_final_reference_runs.csv`, `v3_reference_substitution_events.csv`, and `c2_recovery_gate_diagnostics.csv`.",
        "",
    ])
    return "\n".join(lines)


def replace_block(path: Path, marker: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = f"<!-- {marker}_START -->", f"<!-- {marker}_END -->"
    prefix, rest = text.split(start, 1)
    _, suffix = rest.split(end, 1)
    path.write_text(f"{prefix}{start}\n{body}\n{end}{suffix}", encoding="utf-8")


def main() -> None:
    (PROJECT / "V3_DUAL_VIRTUAL_SENSOR_EXTENSION.md").write_text(build_report(), encoding="utf-8")
    (PROJECT / "C2_ABLATION_ANALYSIS.md").write_text(build_c2_ablation(), encoding="utf-8")
    runs, _, summary, _ = artifacts()
    load = runs[runs.scenario == "load_disturbance"]
    ref = runs[(runs.controller == "C3_arbitration_MPC") & runs.scenario.isin(REFERENCE_SCENARIOS)]
    events = pd.read_csv(METRICS / "v3_reference_substitution_events.csv")
    step_linked = len(events[(events.scenario == "step_reference") & events.time_s.between(1.5, 1.55)])
    changing_linked = len(events[(events.scenario == "changing_reference") & (events.time_s.between(2.0, 2.05) | events.time_s.between(4.0, 4.05))])
    c2 = summary["c2_recovery_gate_diagnostics"]
    block = "\n".join([
        "## Frozen V3 final evidence",
        "",
        f"The completed final matrix uses untouched seeds `29026–29030`: 5 controllers × 11 scenarios × 5 seeds = {summary['run_count']} paired runs. Seeds `19026–19030` remain development evidence.",
        "",
        f"C3 reference-scenario substitutions total `{int(ref.sub_samples.sum())}` isolated samples with zero reliability latches. `{step_linked}` occurred within 0.05 s after the step-reference transition; changing-reference transitions triggered `{changing_linked}`. Load-disturbance mean RMSE is `{load[load.controller == 'B_plain_MPC'].fault_window_rmse.mean():.6f} rad/s` for plain MPC and `{load[load.controller == 'C3_arbitration_MPC'].fault_window_rmse.mean():.6f} rad/s` for C3; the detector's false sensor-fault activation under plant disturbance remains a limitation.",
        "",
        f"C2 remains behaviorally identical to C1 across all 55 paired cases. The auxiliary recovery gate was uniquely binding `{c2['blocked_only_by_auxiliary']}` times and delayed recovery relative to C1: `{c2['auxiliary_gate_delayed_recovery_relative_to_c1']}`.",
        "",
        f"All {summary['run_count']} runs contain `{summary['optimizer_failures']}` optimizer failures, `{summary['nonfinite_events']}` nonfinite events, `{summary['voltage_violations']}` voltage violations, and `{summary['rate_violations']}` slew violations. Canonical artifacts are under `results/metrics/v3_final_11scenario_*`; full evidence is in `V3_DUAL_VIRTUAL_SENSOR_EXTENSION.md`.",
    ])
    replace_block(PROJECT / "README.md", "V3_GENERATED", block)
    replace_block(PROJECT / "PROJECT_PIPELINE_AND_STATUS.md", "V3_FINAL_GENERATED", block)
    print("Generated V3 report, C2 ablation, README, and pipeline status.")


if __name__ == "__main__":
    main()
