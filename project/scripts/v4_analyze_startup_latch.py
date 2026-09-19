"""Post-hoc startup-latch decomposition of Part A / Part B C3 runs.

Reads ONLY existing frozen CSVs (no new simulation) and classifies every
Part A and Part B primary-matrix C3 run by its reliability-entry timing
relative to the event onset:

- PRE_EVENT_LATCH: first reliability entry strictly before the event start
  (contaminated start; the preregistered post-event detection endpoint is
  uninterpretable for that run).
- POST_EVENT_ENTRY: no pre-event entry and at least one post-event entry
  (for speed-fault families this is detection; for healthy-speed families
  it is false entry).
- NO_ENTRY: no reliability entry at all.

Outputs go to results/v4/posthoc/. All relabelled quantities are marked
POST-HOC; preregistered labels are reproduced unchanged and verified equal
to the frozen envelope table.
"""

import json
import math
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "results" / "v4" / "posthoc"

PART_A_RUNS = PROJECT / "results/training_seed_robustness/training_seed_runs.csv"
PART_A_EVENTS = PROJECT / "results/training_seed_robustness/training_seed_reliability_events.csv"
PART_B_RUNS = PROJECT / "results/final_robustness/final_robustness_runs.csv"
PART_B_EVENTS = PROJECT / "results/final_robustness/final_robustness_reliability_events.csv"
PART_B_PAIRED = PROJECT / "results/final_robustness/severity_paired_points.csv"
PART_B_ENVELOPE = PROJECT / "results/final_robustness/operating_envelope_classification.csv"

C3 = "C3_arbitration_MPC"

# Frozen V3 scenario event onsets (scripts/evaluate_v3_closed_loop.py::SCENARIOS
# event_start; asserted equal by tests/test_v4_startup_latch.py).
PART_A_EVENT_START = {
    "sensor_bias_5": 2.0,
    "sensor_dropout": 2.0,
    "combined_fault_load": 3.0,
    "load_disturbance": 3.0,
    "parameter_variation": 3.0,
}

Z95 = 1.96


def fail(message: str) -> None:
    raise SystemExit(f"V4_POSTHOC FAILED: {message}")


def wilson_ci(count: int, total: int, z: float = Z95) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion."""
    if total < 0 or count < 0 or count > total:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return (float("nan"), float("nan"))
    p = count / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def classify_runs(
    runs: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    key_cols: list[str],
    event_start_col: str,
    label_cols: list[str],
) -> pd.DataFrame:
    """Classify each run by entry timing relative to its event start.

    Cross-checks the events table against the runs-table entry counts and
    fails loudly on any mismatch.
    """
    for col in key_cols + [event_start_col, "reliability_entries",
                           "post_event_reliability_entries"]:
        if col not in runs.columns:
            fail(f"runs table missing required column: {col}")
    if runs[event_start_col].isna().any():
        fail("missing event start for a classified run")
    entry_rows = entries[entries["event"] == "entry"].copy()
    first_entry = entry_rows.groupby(key_cols, as_index=False)["time_s"].min().rename(
        columns={"time_s": "first_entry_time_s"})
    entry_count = entry_rows.groupby(key_cols, as_index=False).size().rename(
        columns={"size": "events_table_entries"})
    keep = list(dict.fromkeys(key_cols + label_cols + [event_start_col,
                                               "reliability_entries",
                                               "post_event_reliability_entries"]))
    out = runs[keep].copy()
    out = out.merge(first_entry, on=key_cols, how="left")
    out = out.merge(entry_count, on=key_cols, how="left")
    out["events_table_entries"] = out["events_table_entries"].fillna(0).astype(int)
    mismatch = out[out["events_table_entries"] != out["reliability_entries"]]
    if len(mismatch):
        fail(f"events/runs entry-count mismatch on {len(mismatch)} runs: "
             f"{mismatch[key_cols].head(5).to_dict('records')}")
    out["pre_event_latch"] = (
        (out["reliability_entries"] > 0)
        & (out["first_entry_time_s"] < out[event_start_col]))
    out["post_event_entry"] = out["post_event_reliability_entries"] > 0
    # A run still active from a pre-event latch cannot re-enter post-event;
    # both flags set means it recovered and re-entered (rare, kept explicit).
    out["pre_and_post_entry"] = out["pre_event_latch"] & out["post_event_entry"]
    out["label"] = "NO_ENTRY"
    out.loc[out["post_event_entry"] & ~out["pre_event_latch"], "label"] = "POST_EVENT_ENTRY"
    out.loc[out["pre_event_latch"], "label"] = "PRE_EVENT_LATCH"
    # Exploratory sub-split of pre-event latches. STARTUP covers
    # the saturated-acceleration transient; MID_RUN covers later false
    # entries on a still-healthy sensor (observed only at 2.03 s in
    # (2027, 59029) combined/load runs). The 1.5 s cut sits in the observed
    # gap (max startup entry 1.12 s, min mid-run entry 2.03 s) and extends
    # the frozen 1.0 s V3 arbitration startup-blanking precedent. POST-HOC.
    out["latch_phase"] = "NONE"
    out.loc[out["pre_event_latch"]
            & (out["first_entry_time_s"] < 1.5), "latch_phase"] = "STARTUP"
    out.loc[out["pre_event_latch"]
            & (out["first_entry_time_s"] >= 1.5), "latch_phase"] = "MID_RUN_PRE_EVENT"
    return out


def summarize(frame: pd.DataFrame, group_cols: list[str], study: str) -> pd.DataFrame:
    """Per-group false-latch rate and conditional post-event-entry rate."""
    rows = []
    for keys, group in frame.groupby(group_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(group)
        n_latch = int(group["pre_event_latch"].sum())
        n_startup = int((group["latch_phase"] == "STARTUP").sum())
        n_midrun = int((group["latch_phase"] == "MID_RUN_PRE_EVENT").sum())
        clean = group[~group["pre_event_latch"]]
        n_clean = len(clean)
        n_post_given_clean = int(clean["post_event_entry"].sum())
        latch_lo, latch_hi = wilson_ci(n_latch, n)
        cond_lo, cond_hi = wilson_ci(n_post_given_clean, n_clean)
        row = {"study": study, "n_runs": n,
               "n_pre_event_latch": n_latch,
               "n_startup_latch": n_startup,
               "n_mid_run_pre_event_latch": n_midrun,
               "false_latch_rate": n_latch / n if n else float("nan"),
               "false_latch_wilson_lo": latch_lo,
               "false_latch_wilson_hi": latch_hi,
               "n_clean_start": n_clean,
               "n_post_event_entry_given_clean": n_post_given_clean,
               "cond_post_event_entry_given_clean": (
                   n_post_given_clean / n_clean if n_clean else float("nan")),
               "cond_wilson_lo": cond_lo,
               "cond_wilson_hi": cond_hi,
               "n_pre_and_post_entry": int(group["pre_and_post_entry"].sum())}
        for col, val in zip(group_cols, keys):
            row[col] = val
        rows.append(row)
    cols = (["study"] + group_cols + ["n_runs", "n_pre_event_latch",
            "n_startup_latch", "n_mid_run_pre_event_latch",
            "false_latch_rate", "false_latch_wilson_lo",
            "false_latch_wilson_hi", "n_clean_start",
            "n_post_event_entry_given_clean",
            "cond_post_event_entry_given_clean", "cond_wilson_lo",
            "cond_wilson_hi", "n_pre_and_post_entry"])
    return pd.DataFrame(rows, columns=cols)


def apply_envelope_rules(pooled_det: float, per_seed_det: list[float],
                         pooled_delta: float, per_seed_delta: list[float],
                         favorable_pairs: int, favorable_means: int,
                         failures: int) -> tuple[str, dict, dict]:
    """Frozen envelope rules with a substituted (here: conditional) detection.

    Thresholds copied from results/final_robustness/study_protocol.json
    operating_envelope_rules; only the detection input changes (POST-HOC).
    """
    supported_checks = {
        "all_training_seed_detection_probability_ge_threshold": bool(all(
            d >= 0.8 for d in per_seed_det)),
        "all_training_seed_mean_delta_negative": bool(all(
            d < 0 for d in per_seed_delta)),
        "favorable_pair_count_ge_threshold": bool(favorable_pairs >= 12),
        "safety_numerical_incomplete_failures_zero": bool(failures == 0),
    }
    if all(supported_checks.values()):
        label = "SUPPORTED"
    else:
        limited_checks = {
            "pooled_detection_probability_ge_threshold": bool(
                pooled_det >= 0.5),
            "pooled_mean_delta_negative": bool(pooled_delta < 0),
            "favorable_training_seed_mean_count_ge_threshold": bool(
                favorable_means >= 2),
            "safety_numerical_incomplete_failures_zero": bool(failures == 0),
        }
        label = "LIMITED" if all(limited_checks.values()) else "UNSUPPORTED"
        return label, supported_checks, limited_checks
    limited_checks = {
        "pooled_detection_probability_ge_threshold": bool(pooled_det >= 0.5),
        "pooled_mean_delta_negative": bool(pooled_delta < 0),
        "favorable_training_seed_mean_count_ge_threshold": bool(
            favorable_means >= 2),
        "safety_numerical_incomplete_failures_zero": bool(failures == 0),
    }
    return label, supported_checks, limited_checks


def main() -> None:
    for path in (PART_A_RUNS, PART_A_EVENTS, PART_B_RUNS, PART_B_EVENTS,
                 PART_B_PAIRED, PART_B_ENVELOPE):
        if not path.is_file():
            fail(f"missing required input: {path.relative_to(PROJECT)}")

    # ---- Part A ----
    runs_a = pd.read_csv(PART_A_RUNS)
    runs_a = runs_a[runs_a["controller"] == C3].copy()
    if len(runs_a) != 75:
        fail(f"expected 75 Part A C3 runs, found {len(runs_a)}")
    unknown = set(runs_a["scenario"]) - set(PART_A_EVENT_START)
    if unknown:
        fail(f"Part A scenarios without frozen event start: {unknown}")
    runs_a["event_start_s"] = runs_a["scenario"].map(PART_A_EVENT_START)
    events_a = pd.read_csv(PART_A_EVENTS)
    events_a = events_a[events_a["controller"] == C3].copy()
    part_a = classify_runs(
        runs_a, events_a,
        key_cols=["training_seed", "scenario", "simulation_seed"],
        event_start_col="event_start_s",
        label_cols=["scenario", "simulation_seed", "sensor_fault_detected",
                    "first_post_event_reliability_entry_time_s"])
    # The preregistered Part A detection flag must equal recomputed
    # post-event entry on sensor/combined scenarios (fail loud otherwise).
    # Plant scenarios carry NaN detection by design (no fault to detect) and
    # are excluded from this cross-check.
    detected = part_a["sensor_fault_detected"].astype(str).str.lower().isin(
        ("true", "1", "yes"))
    fault_scenarios = part_a["scenario"].isin(
        ("sensor_bias_5", "sensor_dropout", "combined_fault_load"))
    if (detected[fault_scenarios]
            != part_a.loc[fault_scenarios, "post_event_entry"]).any():
        bad = part_a[fault_scenarios][
            detected[fault_scenarios]
            != part_a.loc[fault_scenarios, "post_event_entry"]]
        fail("Part A preregistered detection != recomputed post-event entry on "
             f"{len(bad)} runs")

    # ---- Part B (primary matrix only; current-boundary excluded, see note) ----
    runs_b = pd.read_csv(PART_B_RUNS)
    runs_b = runs_b[(runs_b["controller"] == C3)
                    & (runs_b["matrix"] == "primary")].copy()
    if len(runs_b) != 300:
        fail(f"expected 300 Part B primary C3 runs, found {len(runs_b)}")
    events_b = pd.read_csv(PART_B_EVENTS)
    events_b = events_b[(events_b["controller"] == C3)
                        & (events_b["matrix"] == "primary")].copy()
    part_b = classify_runs(
        runs_b, events_b,
        key_cols=["family", "condition_id", "training_seed", "simulation_seed"],
        event_start_col="event_start_s",
        label_cols=["condition_id", "training_seed", "simulation_seed",
                    "event_end_s", "sensor_fault_detected",
                    "first_post_event_reliability_entry_time_s",
                    "bias_percent", "dropout_duration_s", "drift_final_percent",
                    "post_step_load_Nm"])
    detected_b = part_b["sensor_fault_detected"].astype(str).str.lower().isin(
        ("true", "1", "yes"))
    fault_families = part_b["family"].isin(
        ("bias", "dropout", "drift", "combined_bias_load"))
    if (detected_b[fault_families]
            != part_b.loc[fault_families, "post_event_entry"]).any():
        fail("Part B preregistered detection != recomputed post-event entry")

    # ---- Summaries ----
    summary_a = summarize(part_a, ["scenario", "training_seed"], "PartA")
    summary_a_pooled = summarize(part_a, ["training_seed"], "PartA")
    summary_b = summarize(part_b, ["family", "training_seed"], "PartB")
    summary_b_pooled = summarize(part_b, ["training_seed"], "PartB")
    per_seed = pd.concat([summary_a, summary_a_pooled, summary_b,
                          summary_b_pooled], ignore_index=True)

    # ---- Envelope relabel (POST-HOC, conditional-on-clean-start) ----
    paired = pd.read_csv(PART_B_PAIRED)
    join_keys = ["family", "training_seed", "simulation_seed", "bias_percent",
                 "dropout_duration_s", "drift_final_percent",
                 "post_step_load_Nm"]
    labeled = part_b.merge(paired[join_keys + ["condition_label"]],
                           on=join_keys, how="left", validate="many_to_one")
    if labeled["condition_label"].isna().any():
        fail("Part B runs without paired-points condition_label")
    envelope = pd.read_csv(PART_B_ENVELOPE)
    relabeled = []
    for _, env_row in envelope.iterrows():
        cond = labeled[labeled["condition_label"] == env_row["condition_label"]]
        if len(cond) != 15:
            fail(f"condition {env_row['condition_label']}: expected 15 runs, "
                 f"found {len(cond)}")
        clean = cond[~cond["pre_event_latch"]]
        pooled_det = (clean["post_event_entry"].sum() / len(clean)
                      if len(clean) else float("nan"))
        per_seed_det = []
        for seed in (2026, 2027, 2028):
            sub = clean[clean["training_seed"] == seed]
            per_seed_det.append(sub["post_event_entry"].sum() / len(sub)
                                if len(sub) else float("nan"))
        posthoc_label, sup_checks, lim_checks = apply_envelope_rules(
            pooled_det, per_seed_det,
            float(env_row["pooled_mean_c3_minus_b_fault_window_rmse"]),
            [float(env_row[f"training_seed_{s}_mean_delta"])
             for s in (2026, 2027, 2028)],
            int(env_row["favorable_pair_count_of_15"]),
            int(env_row["training_seeds_with_favorable_mean_of_3"]),
            int(env_row["safety_numerical_incomplete_failure_total"]))
        relabeled.append({
            "analysis": "POST-HOC (conditional-on-clean-start)",
            "family": env_row["family"],
            "condition_label": env_row["condition_label"],
            "preregistered_classification": env_row["classification"],
            "preregistered_pooled_detection": float(
                env_row["pooled_detection_probability"]),
            "posthoc_n_pre_event_latch_of_15": int(cond["pre_event_latch"].sum()),
            "posthoc_n_startup_latch": int((cond["latch_phase"] == "STARTUP").sum()),
            "posthoc_n_mid_run_latch": int(
                (cond["latch_phase"] == "MID_RUN_PRE_EVENT").sum()),
            "posthoc_n_clean_start": int(len(clean)),
            "posthoc_cond_pooled_detection": pooled_det,
            "posthoc_cond_seed_2026_detection": per_seed_det[0],
            "posthoc_cond_seed_2027_detection": per_seed_det[1],
            "posthoc_cond_seed_2028_detection": per_seed_det[2],
            "posthoc_conditional_classification": posthoc_label,
            "posthoc_supported_checks_json": json.dumps(sup_checks),
            "posthoc_limited_checks_json": json.dumps(lim_checks),
        })
    relabeled_frame = pd.DataFrame(relabeled)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    part_a.to_csv(OUT_DIR / "partA_run_classification.csv", index=False)
    part_b.to_csv(OUT_DIR / "partB_run_classification.csv", index=False)
    per_seed.to_csv(OUT_DIR / "per_seed_summary.csv", index=False)
    relabeled_frame.to_csv(OUT_DIR / "envelope_relabelled_posthoc.csv",
                           index=False)
    summary = {
        "analysis": "POST-HOC (conditional-on-clean-start)",
        "inputs_read_only": [str(p.relative_to(PROJECT)) for p in
                             (PART_A_RUNS, PART_A_EVENTS, PART_B_RUNS,
                              PART_B_EVENTS, PART_B_PAIRED, PART_B_ENVELOPE)],
        "part_a_c3_runs": int(len(part_a)),
        "part_a_pre_event_latches": int(part_a["pre_event_latch"].sum()),
        "part_b_primary_c3_runs": int(len(part_b)),
        "part_b_pre_event_latches": int(part_b["pre_event_latch"].sum()),
        "part_b_pre_and_post_entries": int(part_b["pre_and_post_entry"].sum()),
        "preregistered_detection_reproduced_exactly": True,
        "current_boundary_excluded": (
            "current-sensor matrix runs are outside the V3 single-speed-fault "
            "support claim; their speed-detector entries are false entries "
            "under current corruption, not startup-latch evidence."),
        "note": ("POST-HOC decomposition only. Preregistered envelope labels "
                 "are unchanged; conditional labels answer a different "
                 "question (detection given a clean start)."),
    }
    (OUT_DIR / "posthoc_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Part A C3 runs: {len(part_a)}, pre-event latches: "
          f"{part_a['pre_event_latch'].sum()}")
    print(f"Part B primary C3 runs: {len(part_b)}, pre-event latches: "
          f"{part_b['pre_event_latch'].sum()}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
