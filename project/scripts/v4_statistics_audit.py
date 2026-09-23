"""Correct the V4 confirmatory bootstrap hierarchy from frozen run CSVs only."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

import v4_confirm_analysis as historical  # noqa: E402

OUT = PROJECT / "results" / "v4" / "confirmatory"
REPLICATES = 20_000
BASE_SEED = 20_260_930


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hierarchical_bootstrap(frame: pd.DataFrame, seed: int,
                           replicates: int = REPLICATES) -> dict:
    """Resample training seeds, then nested simulation-seed clusters."""
    trains = sorted(frame.training_seed.unique())
    groups = {
        train: {
            sim: rows.delta.to_numpy(float)
            for sim, rows in frame[frame.training_seed == train].groupby(
                "simulation_seed", sort=True)
        }
        for train in trains
    }
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for replicate in range(replicates):
        sampled_trains = rng.integers(0, len(trains), len(trains))
        pooled = []
        for train_index in sampled_trains:
            simulations = groups[trains[train_index]]
            sim_ids = sorted(simulations)
            sampled_sims = rng.integers(0, len(sim_ids), len(sim_ids))
            pooled.extend(simulations[sim_ids[index]] for index in sampled_sims)
        means[replicate] = np.concatenate(pooled).mean()
    center = means.mean()
    tail_hits = int(np.count_nonzero(
        np.abs(means - center) >= abs(center)))
    return {
        "observed_mean_delta": float(frame.delta.mean()),
        "bootstrap_mean_delta": float(center),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "p_value_two_sided": float(min(1.0, 2.0 * tail_hits / replicates)),
        "tail_hits": tail_hits,
    }


def holm(p_values: list[float]) -> list[dict]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(order)
    rejected = [False] * len(order)
    running = 0.0
    stopped = False
    audit = []
    for rank, index in enumerate(order):
        multiplier = len(order) - rank
        candidate = min(1.0, multiplier * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
        reject = not stopped and p_values[index] <= 0.05 / multiplier
        rejected[index] = reject
        stopped = stopped or not reject
        audit.append({"rank": rank + 1, "hypothesis": f"H{index + 1}",
                      "raw_p": p_values[index], "multiplier": multiplier,
                      "candidate_adjusted_p": candidate,
                      "monotonic_adjusted_p": running, "reject": reject})
    return [{"adjusted": adjusted[i], "reject": rejected[i]}
            for i in range(len(order))], audit


def excluded_pairs(frame: pd.DataFrame, left: str, right: str,
                   filters: pd.Series) -> list[dict]:
    selected = frame[filters & frame.controller.isin([left, right])]
    pre = selected.pivot(index=["training_seed", "simulation_seed"],
                         columns="controller",
                         values="pre_event_reliability_entries")
    excluded = pre[(pre[left] != 0) | (pre[right] != 0)]
    return [
        {"training_seed": int(train), "simulation_seed": int(sim),
         f"{left}_pre_entries": int(row[left]),
         f"{right}_pre_entries": int(row[right])}
        for (train, sim), row in excluded.iterrows()
    ]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    faultfree = historical.load_runs("faultfree", historical.FAULTFREE_SEEDS)
    sweep = historical.load_runs("sweep", historical.SEVERITY_SEEDS)
    recovery = historical.load_runs("recovery", historical.SEVERITY_SEEDS)
    frames = historical.hypothesis_frames(faultfree, sweep, recovery)

    corrected = []
    for position, (hypothesis, _, _) in enumerate(historical.HYPOTHESES):
        stats = hierarchical_bootstrap(frames[hypothesis], BASE_SEED + position)
        row = {
            "hypothesis": hypothesis,
            "n_pairs": len(frames[hypothesis]),
            "n_excluded_pairs": frames[hypothesis].attrs.get(
                "n_excluded_pairs", 0),
            "n_training_seeds": frames[hypothesis].training_seed.nunique(),
            **{key: stats[key] for key in (
                "observed_mean_delta", "bootstrap_mean_delta", "ci_lo",
                "ci_hi", "p_value_two_sided")},
            "cohens_dz": historical.cohens_dz(
                frames[hypothesis].delta.to_numpy(float)),
        }
        corrected.append((row, stats["tail_hits"]))

    decisions, holm_audit = holm(
        [row["p_value_two_sided"] for row, _ in corrected])
    historical_table = pd.read_csv(OUT / "hypothesis_table.csv")
    table = historical_table.copy()
    for position, ((stats, _), decision) in enumerate(zip(corrected, decisions)):
        for key, value in stats.items():
            table.loc[position, key] = value
        table.loc[position, "holm_adjusted_p"] = decision["adjusted"]
        table.loc[position, "holm_reject"] = decision["reject"]
    table.to_csv(OUT / "hypothesis_table_corrected.csv", index=False)

    write_json(OUT / "confirmatory_summary_corrected.json", {
        "alpha": 0.05, "replicates": REPLICATES,
        "hierarchy": "training_seed then nested simulation_seed cluster",
        "hypotheses": json.loads(table.to_json(orient="records")),
    })

    final = json.loads((OUT / "final_summary.json").read_text())
    for target, ((stats, _), decision) in zip(final["hypotheses"],
                                               zip(corrected, decisions)):
        target.update({
            "observed_delta": stats["observed_mean_delta"],
            "bootstrap_estimate": stats["bootstrap_mean_delta"],
            "ci_95": [stats["ci_lo"], stats["ci_hi"]],
            "raw_p": stats["p_value_two_sided"],
            "holm_adjusted_p": decision["adjusted"],
            "decision": "reject" if decision["reject"] else "do-not-reject",
        })
    final["status"] = "CORRECTED STATISTICAL SUMMARY FROM FROZEN CORE"
    final["correction"] = {
        "classification": "IMPLEMENTATION INCONSISTENCY / BUG",
        "scope": ["H1", "H2", "H7"],
        "scientific_simulations_rerun": 0,
        "holm_survivors_changed": False,
    }
    write_json(OUT / "final_summary_corrected.json", final)

    common = sweep.training_seed.isin([2026, 2027, 2028])
    exclusion_specs = {
        "H3": (sweep, "V4_full_aux", "C3",
               common & (sweep.fault_kind == "bias")
               & (sweep.fault_magnitude_sigma == 2.0)),
        "H4": (sweep, "V4_full_ekf", "C3",
               common & (sweep.fault_kind == "bias")
               & (sweep.fault_magnitude_sigma == 2.0)),
        "H5": (recovery, "V4_full_aux", "C3",
               recovery.training_seed.isin([2026, 2027, 2028])
               & (recovery.fault_kind == "bias")),
        "H8": (sweep, "C3", "V4_full_aux",
               common & (sweep.fault_kind == "load")
               & (sweep.load_step_Nm == 0.15)),
        "H9": (sweep, "C3", "V4_full_ekf",
               common & (sweep.fault_kind == "load")
               & (sweep.load_step_Nm == 0.15)),
    }
    exclusions = {
        hypothesis: excluded_pairs(source, left, right, filters)
        for hypothesis, (source, left, right, filters) in exclusion_specs.items()
    }
    write_json(OUT / "exclusion_audit.json", {
        "rule": "exclude iff either controller has pre_event_reliability_entries != 0",
        "outcome_fields_used": [],
        "hypotheses": {key: {"count": len(value), "pair_ids": value}
                       for key, value in exclusions.items()},
        "matched_variant_sets": {
            "H3_equals_H4": [(x["training_seed"], x["simulation_seed"])
                              for x in exclusions["H3"]]
            == [(x["training_seed"], x["simulation_seed"])
                for x in exclusions["H4"]],
            "H8_equals_H9": [(x["training_seed"], x["simulation_seed"])
                              for x in exclusions["H8"]]
            == [(x["training_seed"], x["simulation_seed"])
                for x in exclusions["H9"]],
        },
    })

    artifact_names = [
        "hypothesis_table_corrected.csv",
        "confirmatory_summary_corrected.json",
        "final_summary_corrected.json",
        "exclusion_audit.json",
    ]
    historical_names = [
        "hypothesis_table.csv", "confirmatory_summary.json",
        "h1_h9_detailed_report.csv", "independent_verification.json",
        "final_summary.json",
    ]
    write_json(OUT / "statistics_correction_record.json", {
        "classification": "IMPLEMENTATION INCONSISTENCY / BUG",
        "defect": ("historical bootstrap resampled individual rows within "
                   "training seed and did not use simulation_seed; this "
                   "split four-reference simulation clusters in H1/H2/H7"),
        "corrected_method": ("resample training seeds, then simulation seeds "
                             "within each selected training seed, retaining "
                             "all reference rows in each sampled cluster"),
        "impacted_hypotheses": ["H1", "H2", "H7"],
        "unaffected_hypotheses": ["H3", "H4", "H5", "H6", "H8", "H9"],
        "scientific_simulations_rerun": 0,
        "correction_source": {
            "path": "scripts/v4_statistics_audit.py",
            "sha256": sha256(Path(__file__)),
        },
        "holm_survivors_before": ["H8", "H9"],
        "holm_survivors_after": ["H8", "H9"],
        "historical_artifacts": {
            name: sha256(OUT / name) for name in historical_names},
        "corrected_artifacts": {
            name: sha256(OUT / name) for name in artifact_names},
        "tail_hits": {f"H{i + 1}": hits
                      for i, (_, hits) in enumerate(corrected)},
        "p_value_resolution": 2 / REPLICATES,
        "holm_recomputation": holm_audit,
    })


if __name__ == "__main__":
    main()
