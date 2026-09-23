"""Independent integrity and H1-H9 verifier for the frozen V4 core."""

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results" / "v4"
CONFIRMATORY = RESULTS / "confirmatory"
PLAN = RESULTS / "prereg" / "h1_h9_execution_plan.json"
PREEXECUTION = RESULTS / "prereg" / "preexecution_hashes.json"

PLAN_SHA256 = "fb3e2e614b2b35dd3ae66be3c821d389b59573291140ab2a2f906a2912166ae5"
RUNNER_SHA256 = "f7ee31ae6f266aacf6a7ecc28de9e6ef2cfaac8417bbdf4e1b909b9d1632a07f"
NOTE_SHA256 = "170038598acdcce2a29e2d7b98da71e1e734090a9263386daac304ef83582006"
PREEXECUTION_SHA256 = "d19370e61d36d70b9b94d6a196a423a7b1da9477fd21142b27ae08afe2b680a2"
OLD_PROVENANCE = "7979c98f08431d51c9828bc282f706893abee75de16dc9483d948c16e6a8eed1"
REPAIRED_PROVENANCE = "39f1f08802baea34024ad3a323ae0b4a0c4095f13eb8a8e4752906073d2e6a50"
EXPECTED_HYPOTHESIS_CELLS = {
    "H1": 4800, "H2": 4800, "H3": 300, "H4": 300, "H5": 600,
    "H6": 1100, "H7": 7200, "H8": 300, "H9": 300,
}
HYPOTHESES = {
    "H1": ("C3 vs V4_full_aux", "false-latch probability",
           "C3 - V4_full_aux", 2400, "none"),
    "H2": ("C3 vs V4_full_ekf", "false-latch probability",
           "C3 - V4_full_ekf", 2400, "none"),
    "H3": ("V4_full_aux vs C3", "bias-2sigma detection probability",
           "V4_full_aux - C3", 150, "either controller pre-latched"),
    "H4": ("V4_full_ekf vs C3", "bias-2sigma detection probability",
           "V4_full_ekf - C3", 150, "either controller pre-latched"),
    "H5": ("V4_full_aux vs C3", "bias-8sigma recovery probability",
           "V4_full_aux - C3", 300, "either controller pre-latched"),
    "H6": ("B vs V4_full_aux", "bias-8sigma fault-window RMSE",
           "B - V4_full_aux", 550, "none (intention-to-treat)"),
    "H7": ("C3/B vs V4_full_aux/B", "fault-free tracking penalty",
           "(C3 - B) - (V4_full_aux - B)", 2400,
           "none (intention-to-treat)"),
    "H8": ("C3 vs V4_full_aux", "load-0.15 false-entry probability",
           "C3 - V4_full_aux", 150, "either controller pre-latched"),
    "H9": ("C3 vs V4_full_ekf", "load-0.15 false-entry probability",
           "C3 - V4_full_ekf", 150, "either controller pre-latched"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def same(left, right, tolerance=1e-12) -> bool:
    if left is None or (isinstance(left, float) and math.isnan(left)):
        return right is None or (isinstance(right, float) and math.isnan(right))
    if right is None or (isinstance(right, float) and math.isnan(right)):
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance,
                            abs_tol=tolerance)
    return left == right


def expected_memberships(cell: dict) -> set[str]:
    controller = cell["controller"]
    training = cell["training_seed"]
    simulation = cell["simulation_seed"]
    kind = cell["fault_kind"]
    common = training in (2026, 2027, 2028)
    faultfree = (kind == "none" and simulation in range(71000, 71200)
                 and cell["duration_s"] == 6.0
                 and cell["reference"] in ("nominal", "step", "changing", "wide"))
    severity = (simulation in range(72000, 72050)
                and cell["reference"] == "nominal"
                and cell["duration_s"] == 6.0)
    memberships = set()
    if faultfree and common and controller in ("C3", "V4_full_aux"):
        memberships.add("H1")
    if faultfree and common and controller in ("C3", "V4_full_ekf"):
        memberships.add("H2")
    if (severity and common and kind == "bias"
            and cell["fault_magnitude_sigma"] == 2.0
            and controller in ("C3", "V4_full_aux")):
        memberships.add("H3")
    if (severity and common and kind == "bias"
            and cell["fault_magnitude_sigma"] == 2.0
            and controller in ("C3", "V4_full_ekf")):
        memberships.add("H4")
    if (cell["protocol"] == "recovery" and common and kind == "bias"
            and cell["fault_magnitude_sigma"] == 8.0
            and simulation in range(72000, 72100)
            and cell["duration_s"] == 10.0
            and controller in ("C3", "V4_full_aux")):
        memberships.add("H5")
    if (severity and kind == "bias" and cell["fault_magnitude_sigma"] == 8.0
            and training in range(2026, 2037)
            and controller in ("B", "V4_full_aux")):
        memberships.add("H6")
    if (faultfree and common
            and controller in ("B", "C3", "V4_full_aux")):
        memberships.add("H7")
    if (severity and common and kind == "load"
            and cell["load_step_Nm"] == 0.15
            and controller in ("C3", "V4_full_aux")):
        memberships.add("H8")
    if (severity and common and kind == "load"
            and cell["load_step_Nm"] == 0.15
            and controller in ("C3", "V4_full_ekf")):
        memberships.add("H9")
    return memberships


def verify_bindings() -> dict:
    manifest = json.loads((PROJECT / "results" / "training_seed_robustness"
                           / "model_pair_manifest.json").read_text())
    for seed in (2026, 2027, 2028):
        pair = manifest["pairs"][f"P{seed}"]
        require(pair["training_seed"] == seed, f"C3 pair seed mismatch: {seed}")
        for key in ("main", "auxiliary"):
            for field in ("weights", "config"):
                path = PROJECT / pair[key][f"{field}_path"]
                require(path.is_file(), f"missing C3 binding: {path}")
                require(sha256(path) == pair[key][f"{field}_sha256"],
                        f"changed C3 binding: {path}")
        for key in ("sensor_calibration", "pair_config"):
            path = PROJECT / pair[key]["path"]
            require(path.is_file() and sha256(path) == pair[key]["sha256"],
                    f"changed C3 calibration binding: {path}")
    for seed in range(2026, 2037):
        for kind in ("main", "aux"):
            weights = RESULTS / "models" / f"{kind}_seed_{seed}.pt"
            config = RESULTS / "models" / f"{kind}_seed_{seed}_config.json"
            require(weights.is_file() and config.is_file(),
                    f"missing V4 model binding for {kind}/{seed}")
            require(json.loads(config.read_text())["training"]["seed"] == seed,
                    f"V4 model seed mismatch: {config}")
        calibration = RESULTS / "calibration" / f"sensor_seed{seed}_v4_calibration.json"
        document = json.loads(calibration.read_text())
        require(document["training_seed"] == seed
                and document["pair_id"] == f"P{seed}",
                f"V4 calibration seed mismatch: {calibration}")
    return {"c3_seed_bundles": 3, "v4_seed_bundles": 11,
            "status": "PASS"}


def verify_core() -> tuple[dict, dict[str, pd.DataFrame]]:
    require(sha256(PLAN) == PLAN_SHA256, "re-frozen plan hash mismatch")
    require(sha256(PROJECT / "scripts" / "v4_protocol_core.py") == RUNNER_SHA256,
            "core runner hash mismatch")
    require(sha256(PROJECT / "V4_EXECUTION_PLAN_REFREEZE_NOTE.md") == NOTE_SHA256,
            "re-freeze note hash mismatch")
    require(sha256(PREEXECUTION) == PREEXECUTION_SHA256,
            "preexecution manifest hash mismatch")
    preexecution = json.loads(PREEXECUTION.read_text())
    for relative, expected in preexecution["files"].items():
        require(sha256(PROJECT / relative) == expected,
                f"preexecution-bound artifact changed: {relative}")

    plan = json.loads(PLAN.read_text())
    require(plan["union_count"] == 12200 and len(plan["cells"]) == 12200,
            "plan union is not 12,200")
    require(plan["hypothesis_cell_counts"] == EXPECTED_HYPOTHESIS_CELLS,
            "plan hypothesis counts changed")
    require(plan["original_umbrella_count"] == 268910
            and plan["original_umbrella_deferred"] == 257260,
            "umbrella/deferred counts changed")
    plan_keys = [cell["scientific_key"] for cell in plan["cells"]]
    require(len(set(plan_keys)) == 12200, "duplicate plan scientific keys")
    for cell in plan["cells"]:
        require(set(cell["required_by"]) == expected_memberships(cell),
                f"dependency mismatch for {cell['scientific_key']}")
        require(cell["simulation_seed"] not in set(range(39026, 39031)),
                "forbidden simulation seed")
        require(cell["control_stride"] == 5
                and cell["threshold_scale"] == 1.0,
                "non-preregistered scenario setting")
        if cell["fault_kind"] == "bias":
            require(cell["fault_onset_s"] == 2.0 and cell["fault_end_s"] == 4.0,
                    "bias timing mismatch")
        if cell["fault_kind"] == "load":
            require(cell["fault_onset_s"] == 3.0 and cell["fault_end_s"] is None,
                    "load timing mismatch")
    anchors = [cell for cell in plan["cells"]
               if cell["controller"] == "B"
               and cell["fault_kind"] == "bias"
               and cell["fault_magnitude_sigma"] == 8.0]
    require(len(anchors) == 550, "H6 B/bias-8sigma anchor count mismatch")

    expected_columns = None
    frames = {}
    validity_totals = {name: 0 for name in (
        "optimizer_failures", "main_prediction_failures",
        "aux_prediction_failures", "observer_failures", "nonfinite_events",
        "voltage_violations", "slew_violations")}
    key_counts = {"missing": 0, "unexpected": 0, "duplicates": 0}
    protocol_counts = {}
    event_counts = {}
    for protocol in ("faultfree", "sweep", "recovery"):
        cells = [cell for cell in plan["cells"] if cell["protocol"] == protocol]
        expected_keys = {cell["scientific_key"] for cell in cells}
        connection = sqlite3.connect(RESULTS / protocol / "checkpoint.sqlite3")
        records = connection.execute(
            "SELECT key, provenance, row_json, events_json FROM results").fetchall()
        connection.close()
        require(len(records) == len(expected_keys),
                f"{protocol}: checkpoint row count mismatch")
        keys = [record[0] for record in records]
        key_counts["duplicates"] += len(keys) - len(set(keys))
        key_counts["missing"] += len(expected_keys - set(keys))
        key_counts["unexpected"] += len(set(keys) - expected_keys)
        by_key = {key: (provenance, json.loads(row), json.loads(events))
                  for key, provenance, row, events in records}
        ordered_rows, ordered_events = [], []
        for cell in cells:
            key = cell["scientific_key"]
            provenance, row, events = by_key[key]
            expected_provenance = (REPAIRED_PROVENANCE
                                   if cell["controller"] == "V4_full_ekf"
                                   else OLD_PROVENANCE)
            require(provenance == expected_provenance,
                    f"{protocol}: provenance mismatch for {key}")
            config_fields = {
                "controller": cell["controller"],
                "training_seed": cell["training_seed"],
                "simulation_seed": cell["simulation_seed"],
                "reference": cell["reference"],
                "duration_s": cell["duration_s"],
                "fault_kind": cell["fault_kind"],
                "fault_onset_s": (float("inf") if cell["fault_kind"] == "none"
                                    else cell["fault_onset_s"]),
                "fault_end_s": cell["fault_end_s"],
                "fault_magnitude_rad_s": cell["fault_magnitude_rad_s"],
                "fault_magnitude_sigma": cell["fault_magnitude_sigma"],
                "dropout_duration_s": cell["dropout_duration_s"],
                "drift_rate_rad_s2": cell["drift_rate_rad_s2"],
                "load_step_Nm": cell["load_step_Nm"],
                "control_stride": cell["control_stride"],
                "threshold_scale": cell["threshold_scale"],
            }
            for field, expected in config_fields.items():
                require(same(row[field], expected),
                        f"{protocol}: {field} mismatch for {key}")
            require(row["current_noise_std"] == 0.0
                    and row["sample_delay"] == 0
                    and row["param_preset"] == "nominal"
                    and math.isnan(row["current_quantization_a"]),
                    f"{protocol}: noncanonical scenario for {key}")
            require(row["run_complete"] is True, f"incomplete run: {key}")
            for field in ("overall_rmse", "sub_fraction", "event_sub_fraction",
                          "mean_solve_ms", "p95_solve_ms", "max_solve_ms",
                          "mean_detector_ms", "mean_observer_ms"):
                require(math.isfinite(row[field]), f"nonfinite {field}: {key}")
            if cell["fault_kind"] != "none":
                require(math.isfinite(row["fault_window_rmse"]),
                        f"nonfinite fault metric: {key}")
            for field in validity_totals:
                validity_totals[field] += int(row[field])
            require(row["optimizer_failures"] == 0
                    and row["main_prediction_failures"] == 0
                    and row["aux_prediction_failures"] == 0
                    and row["observer_failures"] == 0
                    and row["nonfinite_events"] == 0
                    and row["voltage_violations"] == 0,
                    f"execution validity failure: {key}")
            entries = [event for event in events if event["event"] == "entry"]
            recoveries = [event for event in events
                          if event["event"] == "recovery"]
            require(len(entries) == row["reliability_entries"]
                    and len(recoveries) == row["recovery_events"],
                    f"event count mismatch: {key}")
            require(all(0.0 <= event["time_s"] <= row["duration_s"]
                        for event in events), f"event outside run: {key}")
            require([event["time_s"] for event in events]
                    == sorted(event["time_s"] for event in events),
                    f"event order mismatch: {key}")
            require(all(math.isfinite(event["residual"]) for event in entries),
                    f"nonfinite event residual: {key}")
            ordered_rows.append(row)
            for event in events:
                ordered_events.append({"controller": cell["controller"],
                                       "training_seed": cell["training_seed"],
                                       "simulation_seed": cell["simulation_seed"],
                                       **event})
        canonical = pd.read_csv(RESULTS / protocol / "runs.csv")
        expected_frame = pd.DataFrame(ordered_rows)
        require(list(canonical.columns) == list(expected_frame.columns),
                f"{protocol}: malformed canonical columns")
        pd.testing.assert_frame_equal(canonical, expected_frame,
                                      check_dtype=False, check_exact=False,
                                      rtol=1e-12, atol=1e-12)
        canonical_events = pd.read_csv(RESULTS / protocol / "events.csv")
        expected_events = pd.DataFrame(ordered_events)
        pd.testing.assert_frame_equal(canonical_events, expected_events,
                                      check_dtype=False, check_exact=False,
                                      rtol=1e-12, atol=1e-12)
        if expected_columns is None:
            expected_columns = list(canonical.columns)
        require(list(canonical.columns) == expected_columns,
                f"{protocol}: inconsistent canonical schema")
        frames[protocol] = canonical
        protocol_counts[protocol] = len(canonical)
        event_counts[protocol] = len(canonical_events)
    require(key_counts == {"missing": 0, "unexpected": 0, "duplicates": 0},
            f"scientific key integrity failed: {key_counts}")
    require(protocol_counts == {"faultfree": 9600, "sweep": 2000,
                                "recovery": 600},
            f"protocol counts mismatch: {protocol_counts}")
    status = json.loads((RESULTS / "core_execution_status.json").read_text())
    require(status["state"] == "complete"
            and status["frozen_plan_sha256"] == PLAN_SHA256
            and status["integrity_repair"]["status"] == "complete",
            "core execution status is not complete/repaired")
    incident = json.loads(
        (RESULTS / "ekf_witness_integrity_repair.json").read_text())
    require(incident["status"] == "repair_complete"
            and incident["target_counts"] == {"faultfree": 2400, "sweep": 300},
            "EKF witness integrity repair is incomplete")
    report = {
        "expected_cells": 12200, "actual_unique_cells": 12200,
        **key_counts, "protocol_counts": protocol_counts,
        "event_counts": event_counts, "validity_failures": 0,
        "validity_counters": validity_totals,
        "slew_note": ("Recorded safety outcomes retained without exclusion; "
                      "not execution corruption."),
        "dependency_counts": EXPECTED_HYPOTHESIS_CELLS,
        "h6_b_bias8_anchor_cells": len(anchors),
        "model_and_calibration_bindings": verify_bindings(),
        "umbrella_cells": 268910, "executed_core_cells": 12200,
        "deferred_umbrella_cells": 257260, "status": "PASS",
    }
    return report, frames


def pair(frame, left, right, value, keys, clean=False) -> pd.DataFrame:
    values = frame.pivot(index=keys, columns="controller", values=value)
    require(left in values and right in values, f"missing pair {left}/{right}")
    require(not values[[left, right]].isna().any(axis=None),
            f"unpaired values for {left}/{right}")
    keep = pd.Series(True, index=values.index)
    if clean:
        pre = frame.pivot(index=keys, columns="controller",
                          values="pre_event_reliability_entries")
        require(not pre[[left, right]].isna().any(axis=None),
                f"unpaired clean flags for {left}/{right}")
        keep = (pre[left] == 0) & (pre[right] == 0)
    result = (values.loc[keep, left] - values.loc[keep, right]).rename(
        "delta").reset_index()
    result.attrs["total_pairs"] = len(values)
    result.attrs["excluded_pairs"] = int((~keep).sum())
    return result


def hypothesis_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    faultfree, sweep, recovery = (frames[name].copy()
                                  for name in ("faultfree", "sweep", "recovery"))
    common = [2026, 2027, 2028]
    faultfree = faultfree[faultfree.training_seed.isin(common)]
    faultfree["false_latch"] = (faultfree.reliability_entries > 0).astype(float)
    keys_ref = ["training_seed", "simulation_seed", "reference"]
    keys = ["training_seed", "simulation_seed"]
    out = {
        "H1": pair(faultfree, "C3", "V4_full_aux", "false_latch", keys_ref),
        "H2": pair(faultfree, "C3", "V4_full_ekf", "false_latch", keys_ref),
    }
    bias2 = sweep[(sweep.training_seed.isin(common))
                  & (sweep.fault_kind == "bias")
                  & (sweep.fault_magnitude_sigma == 2.0)].copy()
    bias2["detected"] = bias2.sensor_fault_detected.astype(float)
    out["H3"] = pair(bias2, "V4_full_aux", "C3", "detected", keys, True)
    out["H4"] = pair(bias2, "V4_full_ekf", "C3", "detected", keys, True)
    recovered = recovery[recovery.training_seed.isin(common)].copy()
    recovered["recovered"] = recovered.recovery_latency_s.notna().astype(float)
    out["H5"] = pair(recovered, "V4_full_aux", "C3", "recovered", keys, True)
    bias8 = sweep[(sweep.fault_kind == "bias")
                  & (sweep.fault_magnitude_sigma == 8.0)]
    out["H6"] = pair(bias8, "B", "V4_full_aux", "fault_window_rmse", keys)
    penalties = faultfree.pivot(
        index=keys_ref, columns="controller", values="overall_rmse")
    require(not penalties[["B", "C3", "V4_full_aux"]].isna().any(axis=None),
            "H7 unpaired tracking cells")
    h7 = ((penalties.C3 - penalties.B)
          - (penalties.V4_full_aux - penalties.B)).rename("delta").reset_index()
    h7.attrs["total_pairs"] = len(h7)
    h7.attrs["excluded_pairs"] = 0
    out["H7"] = h7
    load = sweep[(sweep.training_seed.isin(common))
                 & (sweep.fault_kind == "load")
                 & (sweep.load_step_Nm == 0.15)].copy()
    load["false_entry"] = (load.post_event_reliability_entries > 0).astype(float)
    out["H8"] = pair(load, "C3", "V4_full_aux", "false_entry", keys, True)
    out["H9"] = pair(load, "C3", "V4_full_ekf", "false_entry", keys, True)
    return out


def bootstrap(frame: pd.DataFrame, seed: int, replicates=20000) -> dict:
    trains = sorted(frame.training_seed.unique())
    groups = {train: frame.loc[frame.training_seed == train, "delta"].to_numpy(float)
              for train in trains}
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for index in range(replicates):
        picks = rng.integers(0, len(trains), len(trains))
        pooled = np.concatenate([
            rng.choice(groups[trains[pick]], len(groups[trains[pick]]),
                       replace=True) for pick in picks])
        means[index] = pooled.mean()
    shift = means.mean()
    p_value = min(1.0, 2.0 * np.mean(np.abs(means - shift) >= abs(shift)))
    delta = frame.delta.to_numpy(float)
    sd = delta.std(ddof=1)
    return {
        "observed_mean_delta": float(delta.mean()),
        "bootstrap_mean_delta": float(means.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "p_value_two_sided": float(p_value),
        "cohens_dz": float(delta.mean() / sd) if sd else float("nan"),
        "n_training_seeds": len(trains),
    }


def holm(p_values: list[float]) -> tuple[list[float], list[bool]]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted, rejected, running = [0.0] * len(order), [False] * len(order), 0.0
    stopped = False
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[index]))
        adjusted[index] = running
        if not stopped and p_values[index] <= 0.05 / (len(order) - rank):
            rejected[index] = True
        else:
            stopped = True
    return adjusted, rejected


def verify_analysis(frames: dict[str, pd.DataFrame]) -> tuple[list[dict], dict]:
    table_path = CONFIRMATORY / "hypothesis_table.csv"
    summary_path = CONFIRMATORY / "confirmatory_summary.json"
    require(table_path.is_file() and summary_path.is_file(),
            "confirmatory analysis outputs are missing")
    source = sha256(PROJECT / "scripts" / "v4_confirm_analysis.py")
    require(source == json.loads(PREEXECUTION.read_text())["files"]
            ["scripts/v4_confirm_analysis.py"], "analysis source changed")
    hypothesis_data = hypothesis_frames(frames)
    rows = []
    for position, hypothesis in enumerate(HYPOTHESES):
        data = hypothesis_data[hypothesis]
        controllers, endpoint, direction, expected, reason = HYPOTHESES[hypothesis]
        require(data.attrs["total_pairs"] == expected,
                f"{hypothesis}: expected pair count mismatch")
        result = bootstrap(data, 20260930 + position)
        rows.append({
            "hypothesis": hypothesis, "controllers": controllers,
            "endpoint": endpoint, "direction_convention": direction,
            "expected_pairs": expected,
            "actual_pairs": len(data),
            "excluded_pairs": data.attrs["excluded_pairs"],
            "exclusion_reason": reason, **result,
        })
    adjusted, rejected = holm([row["p_value_two_sided"] for row in rows])
    for row, p_value, decision in zip(rows, adjusted, rejected):
        row["holm_adjusted_p"] = p_value
        row["holm_reject"] = decision
        row["decision"] = "reject" if decision else "do-not-reject"
        row["classification"] = (
            "statistically significant preregistered result" if decision
            else ("directionally favorable descriptive result"
                  if row["observed_mean_delta"] > 0
                  else ("null/uncertain result"
                        if row["observed_mean_delta"] == 0
                        else "unfavorable result")))
    generated = pd.read_csv(table_path)
    require(list(generated.hypothesis) == list(HYPOTHESES),
            "generated hypothesis order mismatch")
    for row in rows:
        actual = generated.loc[generated.hypothesis == row["hypothesis"]].iloc[0]
        mappings = {
            "n_pairs": "actual_pairs", "n_excluded_pairs": "excluded_pairs",
            "n_training_seeds": "n_training_seeds",
            "observed_mean_delta": "observed_mean_delta",
            "bootstrap_mean_delta": "bootstrap_mean_delta",
            "ci_lo": "ci_lo", "ci_hi": "ci_hi", "cohens_dz": "cohens_dz",
            "p_value_two_sided": "p_value_two_sided",
            "holm_adjusted_p": "holm_adjusted_p", "holm_reject": "holm_reject",
        }
        for generated_field, independent_field in mappings.items():
            require(same(actual[generated_field], row[independent_field]),
                    f"{row['hypothesis']}: mismatch in {generated_field}")
    summary = json.loads(summary_path.read_text())
    require(summary["alpha"] == 0.05 and summary["replicates"] == 20000,
            "confirmatory settings mismatch")
    generated_records = generated.to_dict("records")
    require(len(summary["hypotheses"]) == len(generated_records),
            "summary/table row-count mismatch")
    for summary_row, table_row in zip(summary["hypotheses"], generated_records):
        require(summary_row.keys() == table_row.keys()
                and all(same(summary_row[key], table_row[key])
                        for key in summary_row),
                f"summary/table mismatch for {summary_row['hypothesis']}")
    checks = {
        "pair_construction": "PASS", "pair_counts": "PASS",
        "exclusions": "PASS", "observed_deltas": "PASS",
        "bootstrap_input_structure": "PASS", "bootstrap_replicates": 20000,
        "bootstrap_base_seed": 20260930, "confidence_intervals": "PASS",
        "effect_sizes": "PASS", "holm_adjustment": "PASS",
        "decisions": "PASS", "generated_table_regeneration": "PASS",
        "forbidden_data_dependencies": {
            "deferred_umbrella_cells": False,
            "aborted_400_cell_checkpoint": False,
            "development_seeds": False, "C4_development_runs": False,
            "historical_V3_holdout": False, "exploratory_robustness": False,
        },
    }
    return rows, checks


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-analysis", action="store_true")
    args = parser.parse_args(argv)
    core, frames = verify_core()
    if args.pre_analysis:
        output = CONFIRMATORY / "preanalysis_integrity.json"
        write_json(output, {"phase": "pre-analysis", "core_integrity": core,
                            "status": "PASS"})
        print(json.dumps(core, indent=2))
        print(f"wrote {output}")
        return
    hypotheses, checks = verify_analysis(frames)
    detailed = pd.DataFrame(hypotheses)
    detailed.to_csv(CONFIRMATORY / "h1_h9_detailed_report.csv", index=False)
    output = CONFIRMATORY / "independent_verification.json"
    write_json(output, {"core_integrity": core, "analysis_checks": checks,
                        "hypotheses": hypotheses, "status": "PASS"})
    print(detailed.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
