"""Fail-loud verifier for the V4 post-hoc startup-latch decomposition.

Independently re-derives the classification from the frozen source CSVs and
checks every posthoc artifact for schema, count, label, and marker
consistency. Preregistered labels must be byte-identical to the frozen
envelope table.
"""

import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "results" / "v4" / "posthoc"

EXPECTED_FILES = ("partA_run_classification.csv", "partB_run_classification.csv",
                  "per_seed_summary.csv", "envelope_relabelled_posthoc.csv",
                  "posthoc_summary.json")


def fail(message: str) -> None:
    raise SystemExit(f"V4_VERIFY_POSTHOC FAILED: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def check_classification(name: str, key_cols: list[str], event_col: str,
                         runs_path: Path, events_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(OUT_DIR / name)
    runs = pd.read_csv(runs_path)
    events = pd.read_csv(events_path)
    require(len(frame) == len(frame.drop_duplicates(key_cols)),
            f"{name}: duplicate run keys")
    for _, row in frame.iterrows():
        mask = True
        for col in key_cols:
            mask &= events[col] == row[col]
        run_times = events[mask & (events["event"] == "entry")]["time_s"]
        expect_entries = len(run_times)
        require(int(row["reliability_entries"]) == expect_entries,
                f"{name}: entry count mismatch at {row[key_cols].to_dict()}")
        if expect_entries:
            require(abs(float(row["first_entry_time_s"])
                        - float(run_times.min())) < 1e-9,
                    f"{name}: first-entry time mismatch")
            expect_latch = float(run_times.min()) < float(row[event_col])
        else:
            require(pd.isna(row["first_entry_time_s"]),
                    f"{name}: entry time present with zero entries")
            expect_latch = False
        require(bool(row["pre_event_latch"]) == expect_latch,
                f"{name}: pre_event_latch mismatch")
        require(bool(row["post_event_entry"])
                == (int(row["post_event_reliability_entries"]) > 0),
                f"{name}: post_event_entry mismatch")
        expect_label = ("PRE_EVENT_LATCH" if expect_latch
                        else ("POST_EVENT_ENTRY"
                              if int(row["post_event_reliability_entries"]) > 0
                              else "NO_ENTRY"))
        require(row["label"] == expect_label, f"{name}: label mismatch")
        if expect_latch:
            expect_phase = ("STARTUP" if float(row["first_entry_time_s"]) < 1.5
                            else "MID_RUN_PRE_EVENT")
        else:
            expect_phase = "NONE"
        require(row["latch_phase"] == expect_phase,
                f"{name}: latch_phase mismatch")
    print(f"PASS: {name} ({len(frame)} runs independently re-derived)")
    return frame


def main() -> None:
    for name in EXPECTED_FILES:
        require((OUT_DIR / name).is_file(), f"missing output: {name}")
    part_a = check_classification(
        "partA_run_classification.csv",
        ["training_seed", "scenario", "simulation_seed"], "event_start_s",
        PROJECT / "results/training_seed_robustness/training_seed_runs.csv",
        PROJECT / "results/training_seed_robustness"
        / "training_seed_reliability_events.csv")
    require(len(part_a) == 75, "Part A must classify 75 C3 runs")
    part_b = check_classification(
        "partB_run_classification.csv",
        ["family", "condition_id", "training_seed", "simulation_seed"],
        "event_start_s",
        PROJECT / "results/final_robustness/final_robustness_runs.csv",
        PROJECT / "results/final_robustness"
        / "final_robustness_reliability_events.csv")
    require(len(part_b) == 300, "Part B must classify 300 primary C3 runs")

    summary = pd.read_csv(OUT_DIR / "per_seed_summary.csv")
    require((summary["n_pre_event_latch"] == summary["n_startup_latch"]
             + summary["n_mid_run_pre_event_latch"]).all(),
            "per-seed latch split must sum to the total")
    require(((summary["false_latch_rate"] < 0)
             | (summary["false_latch_rate"] > 1)).sum() == 0,
            "false-latch rate out of [0, 1]")
    require(((summary["cond_post_event_entry_given_clean"] < 0)
             | (summary["cond_post_event_entry_given_clean"] > 1)
             | summary["cond_post_event_entry_given_clean"].isna()
             & (summary["n_clean_start"] == 0)).sum() == 0,
            "conditional rate out of range")
    print(f"PASS: per_seed_summary.csv ({len(summary)} groups consistent)")

    relabeled = pd.read_csv(OUT_DIR / "envelope_relabelled_posthoc.csv")
    envelope = pd.read_csv(PROJECT / "results/final_robustness"
                           / "operating_envelope_classification.csv")
    require(len(relabeled) == len(envelope) == 16,
            "envelope relabel must cover all 16 conditions")
    merged = relabeled.merge(envelope, on=["family", "condition_label"],
                             suffixes=("", "_frozen"))
    require(len(merged) == 16, "envelope join must match all conditions")
    require((merged["preregistered_classification"]
             == merged["classification"]).all(),
            "preregistered labels must be unchanged")
    require((relabeled["analysis"]
             == "POST-HOC (conditional-on-clean-start)").all(),
            "every relabeled row must carry the POST-HOC marker")
    for _, row in relabeled.iterrows():
        require(int(row["posthoc_n_pre_event_latch_of_15"])
                + int(row["posthoc_n_clean_start"]) == 15,
                "latch + clean must equal 15")
    print("PASS: envelope_relabelled_posthoc.csv (16 conditions, "
          "preregistered labels unchanged, POST-HOC marked)")

    doc = json.loads((OUT_DIR / "posthoc_summary.json").read_text())
    require(doc.get("analysis", "").startswith("POST-HOC"),
            "summary must be marked POST-HOC")
    require(int(doc["part_a_pre_event_latches"])
            == int(part_a["pre_event_latch"].sum()),
            "summary Part A latch count must match the classification")
    require(int(doc["part_b_pre_event_latches"])
            == int(part_b["pre_event_latch"].sum()),
            "summary Part B latch count must match the classification")
    print("PASS: posthoc_summary.json counts match classifications")
    print("V4_VERIFY_POSTHOC RESULT: PASS")


if __name__ == "__main__":
    main()
