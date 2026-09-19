"""Confirmatory analysis: paired hypotheses + hierarchical bootstrap + Holm.

Reads the protocol runs.csv files and evaluates the preregistered
hypotheses (see V4_PREREGISTRATION.md). Every hypothesis is a paired
comparison with hierarchy training seed -> simulation seed, a two-sided
hierarchical-bootstrap 95% CI, Cohen's dz, and a Holm decision across
the family at alpha = 0.05.

Fails loudly on missing inputs, unpaired cells, or out-of-block seeds.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from v4_analysis import (  # noqa: E402
    bootstrap_two_sided_p_value,
    cohens_dz,
    hierarchical_bootstrap_mean,
    holm,
    wilson_ci,
)
from v4_protocol_common import FAULTFREE_SEEDS, SEVERITY_SEEDS  # noqa: E402

RESULTS = PROJECT / "results" / "v4"
OUT_DIR = RESULTS / "confirmatory"

ALPHA = 0.05
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_BASE_SEED = 20260930

# Hypothesis registry: (id, description, direction). Positive paired delta
# supports the hypothesis in every case (signs arranged at pairing time).
HYPOTHESES = [
    ("H1", "V4_full_aux false-latch prob < C3 (fault-free, all refs)",
     "positive"),
    ("H2", "V4_full_ekf false-latch prob < C3 (fault-free, all refs)",
     "positive"),
    ("H3", "V4_full_aux detection prob > C3 at bias 2 sigma (clean pairs)",
     "positive"),
    ("H4", "V4_full_ekf detection prob > C3 at bias 2 sigma (clean pairs)",
     "positive"),
    ("H5", "V4_full_aux recovery prob > C3, finite bias 8 sigma (clean)",
     "positive"),
    ("H6", "V4_full_aux fault-window RMSE < B at bias 8 sigma", "positive"),
    ("H7", "V4_full_aux fault-free tracking penalty < C3 penalty", "positive"),
    ("H8", "V4_full_aux load false-entry prob < C3 at 0.15 Nm (clean)",
     "positive"),
    ("H9", "V4_full_ekf load false-entry prob < C3 at 0.15 Nm (clean)",
     "positive"),
]


def fail(message: str) -> None:
    raise SystemExit(f"V4_CONFIRM FAILED: {message}")


def load_runs(name: str, required_block: list[int]) -> pd.DataFrame:
    path = RESULTS / name / "runs.csv"
    if not path.is_file():
        fail(f"missing protocol output: {path.relative_to(PROJECT)} "
             f"(run scripts/v4_protocol_{name}.py)")
    frame = pd.read_csv(path)
    block = set(required_block)
    outside = set(frame["simulation_seed"]) - block
    if outside:
        fail(f"{name}: {len(outside)} sim seeds outside the preregistered "
             f"block (e.g. {sorted(outside)[:3]})")
    return frame


def paired_deltas(frame: pd.DataFrame, left: str, right: str, value: str,
                  keys: list[str]) -> pd.DataFrame:
    """Paired left-minus-right deltas; fails loudly on unpaired cells."""
    pivot = frame.pivot_table(index=keys, columns="controller",
                              values=value, aggfunc="first")
    for controller in (left, right):
        if controller not in pivot.columns:
            fail(f"controller {controller} absent; cannot pair {left}/{right}")
    missing = pivot[[left, right]].isna().any(axis=1)
    if missing.any():
        fail(f"unpaired {left}/{right} cells: {int(missing.sum())} "
             f"(keys {keys})")
    deltas = (pivot[left] - pivot[right]).reset_index()
    deltas.columns = [*keys, "delta"]
    return deltas


def paired_clean_deltas(frame: pd.DataFrame, left: str, right: str,
                        value: str, keys: list[str]) -> pd.DataFrame:
    """Paired deltas on clean-start pairs only (pair-exclusion rule).

    Drops pairs where EITHER controller latched before the event onset
    (those cells cannot exhibit post-onset detection/entry by
    construction). The exclusion count is attached for reporting.
    """
    pivot = frame.pivot_table(index=keys, columns="controller",
                              values=[value, "pre_event_reliability_entries"],
                              aggfunc="first")
    for controller in (left, right):
        if controller not in pivot[value].columns:
            fail(f"controller {controller} absent; cannot pair {left}/{right}")
    missing = pivot[value][[left, right]].isna().any(axis=1)
    if missing.any():
        fail(f"unpaired {left}/{right} cells: {int(missing.sum())}")
    pre = pivot["pre_event_reliability_entries"][[left, right]].fillna(0)
    clean = ((pre[left] == 0) & (pre[right] == 0))
    deltas = (pivot[value][left][clean] - pivot[value][right][clean])
    result = deltas.reset_index()
    result.columns = [*keys, "delta"]
    result.attrs["n_excluded_pairs"] = int((~clean).sum())
    result.attrs["n_total_pairs"] = int(len(pivot))
    return result


def hypothesis_frames(faultfree: pd.DataFrame, sweep: pd.DataFrame,
                      recovery: pd.DataFrame) -> dict[str, pd.DataFrame]:
    faultfree = faultfree.copy()
    faultfree["false_latch"] = (
        faultfree["reliability_entries"] > 0).astype(float)
    out: dict[str, pd.DataFrame] = {}
    keys_ref = ["training_seed", "simulation_seed", "reference"]
    out["H1"] = paired_deltas(faultfree, "C3", "V4_full_aux", "false_latch",
                              keys_ref)
    out["H2"] = paired_deltas(faultfree, "C3", "V4_full_ekf", "false_latch",
                              keys_ref)
    sweep_bias2 = sweep[(sweep["fault_kind"] == "bias")
                        & (sweep["fault_magnitude_sigma"] == 2.0)].copy()
    sweep_bias2["detected"] = sweep_bias2["sensor_fault_detected"].astype(
        float)
    keys = ["training_seed", "simulation_seed"]
    out["H3"] = paired_clean_deltas(sweep_bias2, "V4_full_aux", "C3",
                                    "detected", keys)
    out["H4"] = paired_clean_deltas(sweep_bias2, "V4_full_ekf", "C3",
                                    "detected", keys)
    recovery_bias = recovery[
        (recovery["fault_kind"] == "bias")].copy()
    recovery_bias["recovered"] = (
        recovery_bias["recovery_latency_s"].notna()).astype(float)
    out["H5"] = paired_clean_deltas(recovery_bias, "V4_full_aux", "C3",
                                    "recovered", keys)
    sweep_bias8 = sweep[(sweep["fault_kind"] == "bias")
                        & (sweep["fault_magnitude_sigma"] == 8.0)]
    out["H6"] = paired_deltas(sweep_bias8, "B", "V4_full_aux",
                              "fault_window_rmse", keys)
    penalty = faultfree.pivot_table(
        index=keys_ref, columns="controller", values="overall_rmse",
        aggfunc="first")
    for controller in ("B", "C3", "V4_full_aux"):
        if controller not in penalty.columns or penalty[controller].isna().any():
            fail(f"H7: unpaired or missing {controller} fault-free cells")
    h7 = pd.DataFrame(
        {"delta": ((penalty["C3"] - penalty["B"])
                   - (penalty["V4_full_aux"] - penalty["B"]))}).reset_index()
    out["H7"] = h7
    sweep_load = sweep[(sweep["fault_kind"] == "load")
                       & (sweep["load_step_Nm"] == 0.15)].copy()
    sweep_load["false_entry"] = (
        sweep_load["post_event_reliability_entries"] > 0).astype(float)
    out["H8"] = paired_clean_deltas(sweep_load, "C3", "V4_full_aux",
                                    "false_entry", keys)
    out["H9"] = paired_clean_deltas(sweep_load, "C3", "V4_full_ekf",
                                    "false_entry", keys)
    return out


def evaluate(frames: dict[str, pd.DataFrame], replicates: int) -> pd.DataFrame:
    rows = []
    p_values = []
    for position, (hypothesis, description, _) in enumerate(HYPOTHESES):
        frame = frames[hypothesis]
        result = hierarchical_bootstrap_mean(
            frame, "delta", replicates, BOOTSTRAP_BASE_SEED + position)
        p_value = bootstrap_two_sided_p_value(
            result.pop("bootstrap_replicates"))
        p_values.append(p_value)
        rows.append({
            "hypothesis": hypothesis,
            "description": description,
            "n_pairs": len(frame),
            "n_excluded_pairs": frame.attrs.get("n_excluded_pairs", 0),
            "n_training_seeds": result["n_training_seeds"],
            "observed_mean_delta": result["observed_mean"],
            "bootstrap_mean_delta": result["bootstrap_mean"],
            "ci_lo": result["ci_lo"],
            "ci_hi": result["ci_hi"],
            "cohens_dz": cohens_dz(frame["delta"].to_numpy(float)),
            "p_value_two_sided": p_value,
        })
    decisions = holm(p_values, alpha=ALPHA)
    for row, decision in zip(rows, decisions):
        row["holm_adjusted_p"] = decision["holm_adjusted_p"]
        row["holm_reject"] = decision["reject"]
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int,
                        default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    faultfree = load_runs("faultfree", FAULTFREE_SEEDS)
    sweep = load_runs("sweep", SEVERITY_SEEDS)
    recovery = load_runs("recovery", SEVERITY_SEEDS)
    frames = hypothesis_frames(faultfree, sweep, recovery)
    table = evaluate(frames, args.replicates)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "hypothesis_table.csv", index=False)
    summary = {"alpha": ALPHA, "replicates": args.replicates,
               "hypotheses": table.to_dict("records")}
    (args.out_dir / "confirmatory_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(table.to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
