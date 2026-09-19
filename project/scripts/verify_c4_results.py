"""Fail-loud independent verifier for the C4 three-way-attribution extension.

The verifier deliberately treats C4 as evidence layered on top of frozen V3.
It does not generate or repair scientific results.  Missing, ambiguous, or
internally inconsistent artifacts are verification failures.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import re
import sys
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
METRICS = PROJECT / "results" / "metrics"
CONFIGS = PROJECT / "results" / "configs"

EXPECTED_CONTROLLERS = {
    "B_plain_MPC",
    "C3_arbitration_MPC",
    "C4_attribution_MPC",
}
EXPECTED_SCENARIOS = {
    "nominal_tracking",
    "step_reference",
    "changing_reference",
    "sensor_noise",
    "sensor_bias_5",
    "sensor_bias_15",
    "sensor_dropout",
    "sensor_drift",
    "load_disturbance",
    "parameter_variation",
    "combined_fault_load",
}
EXPECTED_SCENARIO_ORDER = [
    "nominal_tracking",
    "step_reference",
    "changing_reference",
    "sensor_noise",
    "sensor_bias_5",
    "sensor_bias_15",
    "sensor_dropout",
    "sensor_drift",
    "load_disturbance",
    "parameter_variation",
    "combined_fault_load",
]
DEVELOPMENT_SEEDS = set(range(19026, 19031))
V3_FINAL_SEEDS = set(range(29026, 29031))
EXPECTED_RUN_COUNT = 3 * 11 * 5
LATENCY_POLICY = (
    "For each abrupt sensor-fault scenario, the latency criterion is evaluable only when all five paired C3 "
    "and all five paired C4 runs have finite detection_latency_s values. Any missing/nonfinite latency makes "
    "that scenario's latency criterion fail; detection-rate preservation is evaluated separately."
)

ATTRIBUTION_STATES = {
    "NORMAL",
    "LIKELY_SENSOR_FAULT",
    "LIKELY_PLANT_OR_MAIN_MISMATCH",
    "LIKELY_AUX_MISMATCH",
    "AMBIGUOUS",
}

# The attributor intentionally clears its diagnostic EWMA when the auxiliary
# witness is unavailable.  Classification, persistence, and veto decisions use
# the raw pairwise disagreements instead, so a fully missing filtered triplet is
# valid only while the witness is explicitly unavailable and attribution is
# forced AMBIGUOUS.
AUXILIARY_UNAVAILABLE_REASONS = {
    "STARTUP_BLANKING",
    "AUX_NONFINITE",
    "AUX_OUT_OF_BOUNDS",
    "AUX_PARAM_MISMATCH",
    "INSUFFICIENT_TRUSTED_HISTORY",
    "AUX_EWMA_INCONSISTENT",
}

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
INT_TOKEN_RE_TEMPLATE = r"(?<!\d){seed}(?!\d)"
FROZEN_C4_CALIBRATION_GENERATOR_SHA256 = (
    "b7ce55487188967a368fb42e5a4c2b86ae6eb2cf20f47e52169b806a3615002a"
)
FROZEN_C4_DEVELOPMENT_VERIFIER_SHA256 = (
    "294bfec07f45e5570e5a16401b5a372d4cf6793feca864f37d0ee8dc96f8424d"
)


class VerificationError(RuntimeError):
    """Raised when saved C4 evidence fails an invariant."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)
    print(f"PASS: {message}")


def sha256(path: Path) -> str:
    check(path.is_file(), f"required file exists: {path.relative_to(PROJECT)}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_c4_script_hashes() -> dict[str, str]:
    return {
        "c4_evaluator_script": sha256(PROJECT / "scripts/evaluate_c4_closed_loop.py"),
        "c4_verifier_script": sha256(Path(__file__).resolve()),
    }


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    check(isinstance(value, dict), f"{path.name} contains a JSON object")
    return value


def _artifact_candidates(exact: Sequence[Path], patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for path in exact:
        if path.is_file() and path not in seen:
            paths.append(path)
            seen.add(path)
    for pattern in patterns:
        for path in sorted(PROJECT.glob(pattern)):
            if path.is_file() and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def require_one_artifact(label: str, exact: Sequence[Path], patterns: Sequence[str] = ()) -> Path:
    paths = _artifact_candidates(exact, patterns)
    if not paths:
        expected = [str(path.relative_to(PROJECT)) for path in exact]
        expected.extend(patterns)
        raise VerificationError(f"missing {label}; expected one of: {expected}")
    if len(paths) > 1:
        names = [str(path.relative_to(PROJECT)) for path in paths]
        raise VerificationError(f"ambiguous {label}; found multiple candidate artifacts: {names}")
    path = paths[0]
    print(f"PASS: resolved {label}: {path.relative_to(PROJECT)}")
    return path


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    check(not missing, f"{label} contains required columns (missing={missing})")


def choose_column(frame: pd.DataFrame, aliases: Sequence[str], label: str) -> str:
    found = [name for name in aliases if name in frame.columns]
    if not found:
        raise VerificationError(f"{label} missing; expected one of columns {list(aliases)}")
    return found[0]


def numeric(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    check(values.notna().all(), f"{label} is numeric and non-missing")
    check(np.isfinite(values.to_numpy(float)).all(), f"{label} is finite")
    return values.astype(float)


def coerce_seed_series(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    check(values.notna().all() and np.isfinite(values.to_numpy(float)).all(), f"{label} seeds are numeric and finite")
    check(np.allclose(values.to_numpy(float), np.round(values.to_numpy(float)), rtol=0, atol=0), f"{label} seeds are exact integers")
    return values.astype(int)


def hash_binding(mapping: Mapping, *names: str) -> str | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, str):
            return value
    return None


def _walk_json(value, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_json(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, path + (str(index),))
    else:
        yield path, value


def json_contains_hash(value: Mapping, digest: str) -> bool:
    digest = digest.lower()
    return any(isinstance(leaf, str) and leaf.lower() == digest for _, leaf in _walk_json(value))


def json_seed_lists(value: Mapping) -> list[tuple[tuple[str, ...], set[int]]]:
    found: list[tuple[tuple[str, ...], set[int]]] = []

    def integer_like_list(node: list) -> set[int] | None:
        if not node:
            return None
        seeds: list[int] = []
        for item in node:
            if isinstance(item, bool):
                return None
            if isinstance(item, int):
                seeds.append(int(item))
                continue
            if isinstance(item, float):
                if not np.isfinite(item) or not item.is_integer():
                    return None
                seeds.append(int(item))
                continue
            if isinstance(item, str):
                try:
                    seed = int(item)
                except ValueError:
                    return None
                if str(seed) != item.strip():
                    return None
                seeds.append(seed)
                continue
            return None
        return set(seeds)

    def visit(node, path: tuple[str, ...] = ()) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                visit(child, path + (str(key),))
        elif isinstance(node, list):
            converted = integer_like_list(node)
            if converted is not None:
                found.append((path, converted))
                return
            for index, child in enumerate(node):
                visit(child, path + (str(index),))
    visit(value)
    return found


def find_mappings_with_keys(value, required: set[str]) -> list[Mapping]:
    found: list[Mapping] = []
    if isinstance(value, Mapping):
        if required <= set(value):
            found.append(value)
        for child in value.values():
            found.extend(find_mappings_with_keys(child, required))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_mappings_with_keys(child, required))
    return found


def has_named_seed_set(value: Mapping, expected: set[int], required_terms: Sequence[str]) -> bool:
    for path, seeds in json_seed_lists(value):
        joined = ".".join(path).lower()
        if seeds == expected and all(term.lower() in joined for term in required_terms):
            return True
    return False


def find_named_values(value, terms: Sequence[str]) -> list[object]:
    matches: list[object] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_lower = str(key).lower()
            if all(term.lower() in key_lower for term in terms):
                matches.append(child)
            matches.extend(find_named_values(child, terms))
    elif isinstance(value, list):
        for child in value:
            matches.extend(find_named_values(child, terms))
    return matches


def verify_v3_freeze() -> dict:
    baseline_path = CONFIGS / "c4_v3_frozen_hashes.json"
    baseline = load_json(baseline_path)
    check(baseline.get("recorded_before_c4_implementation") is True, "V3 freeze record predates C4 implementation")
    artifacts = baseline.get("artifacts")
    check(isinstance(artifacts, dict) and artifacts, "V3 freeze record lists frozen artifacts")
    for name, record in artifacts.items():
        check(isinstance(record, dict), f"V3 freeze entry {name} is structured")
        relative = record.get("path")
        expected_hash = record.get("sha256")
        check(isinstance(relative, str) and relative, f"V3 freeze entry {name} records a path")
        check(isinstance(expected_hash, str) and HASH_RE.fullmatch(expected_hash) is not None, f"V3 freeze entry {name} records a SHA256")
        actual = sha256(PROJECT / relative)
        check(actual == expected_hash, f"frozen V3 artifact unchanged: {name}")

    run_record = artifacts.get("v3_final_275_run_matrix")
    check(isinstance(run_record, dict), "V3 freeze record includes the 275-run matrix")
    v3_runs = pd.read_csv(PROJECT / run_record["path"], float_precision="round_trip")
    check(len(v3_runs) == 275, "frozen V3 primary matrix still contains exactly 275 rows")

    source_hashes = baseline.get("frozen_v3_class_source_sha256", {})
    check(
        set(source_hashes) >= {"SensorReliabilityMonitor", "DualVirtualSensorArbitrator"},
        "V3 freeze record includes both frozen reliability class source hashes",
    )
    sys.path.insert(0, str(PROJECT / "src"))
    from reliability import DualVirtualSensorArbitrator, SensorReliabilityMonitor

    actual_source_hashes = {
        "SensorReliabilityMonitor": hashlib.sha256(
            inspect.getsource(SensorReliabilityMonitor).encode("utf-8")
        ).hexdigest(),
        "DualVirtualSensorArbitrator": hashlib.sha256(
            inspect.getsource(DualVirtualSensorArbitrator).encode("utf-8")
        ).hexdigest(),
    }
    check(actual_source_hashes == source_hashes, "frozen V3 reliability/arbitration class sources are unchanged")
    return baseline


def _ast_identifiers(node: ast.AST) -> set[str]:
    identifiers: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            identifiers.add(child.id)
        elif isinstance(child, ast.Attribute):
            identifiers.add(child.attr)
    return identifiers


def _contains_decision_key(node: ast.AST, keys: set[str]) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        slice_node = child.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str) and slice_node.value in keys:
            return True
    return False


def _oracle_identifiers(identifiers: Iterable[str]) -> set[str]:
    forbidden: set[str] = set()
    exact = {
        "y_true",
        "true_speed",
        "scenario",
        "scenario_name",
        "scenario_label",
        "fault_flag",
        "fault_label",
        "event_start",
        "event_end",
        "event_mask",
        "load_disturbance_flag",
    }
    for identifier in identifiers:
        lowered = identifier.lower()
        if lowered in exact or lowered.startswith("scenario_") or lowered.startswith("known_fault"):
            forbidden.add(identifier)
    return forbidden


def verify_no_oracle_online_logic() -> None:
    sys.path.insert(0, str(PROJECT / "src"))
    from reliability import ThreeWayConsistencyAttributor

    signature = inspect.signature(ThreeWayConsistencyAttributor.update)
    forbidden_parameters = _oracle_identifiers(signature.parameters)
    check(not forbidden_parameters, f"C4 attributor update signature has no oracle inputs ({sorted(forbidden_parameters)})")

    update_source = textwrap.dedent(inspect.getsource(ThreeWayConsistencyAttributor.update))
    update_tree = ast.parse(update_source)
    source_forbidden = _oracle_identifiers(_ast_identifiers(update_tree))
    check(not source_forbidden, f"C4 attributor update implementation has no oracle identifiers ({sorted(source_forbidden)})")

    evaluator = PROJECT / "scripts" / "evaluate_c4_closed_loop.py"
    check(evaluator.is_file(), "C4 evaluator source exists for static no-oracle verification")
    tree = ast.parse(evaluator.read_text(encoding="utf-8"), filename=str(evaluator))
    attributor_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "update":
            continue
        keyword_names = {keyword.arg for keyword in node.keywords if keyword.arg}
        if {"auxiliary_trusted", "detector_requests_entry"} <= keyword_names:
            attributor_calls.append(node)
    check(len(attributor_calls) >= 1, "C4 evaluator contains an identifiable attributor update callsite")
    for call in attributor_calls:
        expression_nodes: list[ast.AST] = list(call.args) + [keyword.value for keyword in call.keywords]
        identifiers = set().union(*(_ast_identifiers(node) for node in expression_nodes)) if expression_nodes else set()
        forbidden = _oracle_identifiers(identifiers)
        check(not forbidden, f"C4 evaluator attributor call arguments contain no oracle identifiers ({sorted(forbidden)})")

    decision_keys = {"entry_authorized", "entry_suppressed"}
    checked_decision_nodes = 0
    for node in ast.walk(tree):
        relevant = False
        expression: ast.AST | None = None
        if isinstance(node, ast.Assign):
            target_names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and target.id in decision_keys
            }
            relevant = bool(target_names) or _contains_decision_key(node.value, decision_keys)
            expression = node.value
        elif isinstance(node, ast.If):
            names = _ast_identifiers(node.test)
            relevant = bool(names & decision_keys) or _contains_decision_key(node.test, decision_keys)
            expression = node.test
        if not relevant or expression is None:
            continue
        checked_decision_nodes += 1
        forbidden = _oracle_identifiers(_ast_identifiers(expression))
        check(not forbidden, f"C4 evaluator entry authorization/suppression expression has no oracle identifiers ({sorted(forbidden)})")
    check(checked_decision_nodes >= 1, "C4 evaluator exposes entry authorization/suppression logic to static inspection")


def resolve_c4_artifacts() -> dict[str, Path]:
    dedicated_provenance = [
        CONFIGS / "c4_final_seed_provenance.json",
        CONFIGS / "c4_final_holdout_seed_provenance.json",
        CONFIGS / "c4_final_seed_selection.json",
    ]
    present_provenance = [path for path in dedicated_provenance if path.is_file()]
    if len(present_provenance) > 1:
        raise VerificationError(
            "ambiguous pre-use C4 final-seed provenance; found multiple dedicated artifacts: "
            f"{[str(path.relative_to(PROJECT)) for path in present_provenance]}"
        )
    seed_provenance = present_provenance[0] if present_provenance else require_one_artifact(
        "pre-use C4 final-seed provenance fallback",
        [CONFIGS / "c4_candidate_seed_audit.json"],
    )

    return {
        "calibration": require_one_artifact(
            "C4 calibration",
            [CONFIGS / "c4_attribution_calibration.json"],
        ),
        "frozen_config": require_one_artifact(
            "C4 frozen config",
            [CONFIGS / "c4_frozen_config.json"],
        ),
        "criteria": require_one_artifact(
            "predeclared C4 go/no-go criteria",
            [CONFIGS / "c4_go_no_go_criteria.json"],
        ),
        "seed_provenance": seed_provenance,
        "runs": require_one_artifact(
            "C4 final primary run matrix",
            [METRICS / "c4_final_holdout_runs.csv"],
        ),
        "summary": require_one_artifact(
            "C4 final summary",
            [METRICS / "c4_final_holdout_summary.json"],
        ),
        "events": require_one_artifact(
            "C4 attribution event evidence",
            [METRICS / "c4_final_holdout_attribution_events.csv"],
        ),
        "runtime": require_one_artifact(
            "C4 attribution runtime evidence",
            [METRICS / "c4_final_holdout_timing.csv"],
        ),
        "documentation": require_one_artifact(
            "C4 extension documentation",
            [PROJECT / "C4_THREE_WAY_ATTRIBUTION_EXTENSION.md"],
        ),
    }


def resolve_development_artifacts() -> dict[str, Path]:
    return {
        "development_config": require_one_artifact(
            "frozen C4 development config",
            [CONFIGS / "c4_development_config.json"],
        ),
        "calibration": require_one_artifact(
            "C4 calibration",
            [CONFIGS / "c4_attribution_calibration.json"],
        ),
        "criteria": require_one_artifact(
            "predeclared C4 go/no-go criteria",
            [CONFIGS / "c4_go_no_go_criteria.json"],
        ),
        "runs": require_one_artifact(
            "C4 development primary run matrix",
            [METRICS / "c4_development_runs.csv"],
        ),
        "summary": require_one_artifact(
            "C4 development summary",
            [METRICS / "c4_development_summary.json"],
        ),
        "events": require_one_artifact(
            "C4 development attribution events",
            [METRICS / "c4_development_attribution_events.csv"],
        ),
        "runtime": require_one_artifact(
            "C4 development timing evidence",
            [METRICS / "c4_development_timing.csv"],
        ),
        "gate": require_one_artifact(
            "C4 development stage-gate record",
            [METRICS / "c4_development_stage_gate.json"],
        ),
    }


def verify_c4_hash_chain(paths: Mapping[str, Path], summary: Mapping) -> tuple[dict, dict, str, str]:
    calibration = load_json(paths["calibration"])
    frozen_config = load_json(paths["frozen_config"])
    calibration_hash = sha256(paths["calibration"])
    frozen_config_hash = sha256(paths["frozen_config"])

    check(
        json_contains_hash(frozen_config, calibration_hash),
        "frozen C4 config cryptographically binds the exact C4 calibration",
    )
    frozen_hashes = frozen_config.get("hashes") if isinstance(frozen_config.get("hashes"), Mapping) else frozen_config
    check(
        hash_binding(
            frozen_hashes,
            "c4_attribution_calibration",
            "c4_attribution_calibration_json",
            "c4_attribution_calibration_sha256",
            "calibration_sha256",
        ) == calibration_hash
        or hash_binding(
            frozen_config,
            "c4_attribution_calibration_sha256",
            "calibration_sha256",
        ) == calibration_hash,
        "frozen C4 config has a dedicated exact calibration hash binding",
    )
    check(
        isinstance(frozen_config.get("scenarios"), list) and frozen_config.get("scenarios") == EXPECTED_SCENARIO_ORDER,
        "frozen C4 config explicitly declares the exact final scenario list",
    )

    summary_hashes = summary.get("hashes")
    check(isinstance(summary_hashes, Mapping), "final summary contains a dedicated hash block")
    check(
        summary_hashes.get("c4_frozen_config") == frozen_config_hash,
        "final summary dedicated frozen-config hash matches the exact current C4 frozen config",
    )
    check(
        summary_hashes.get("c4_attribution_calibration") == calibration_hash,
        "final summary dedicated calibration hash matches the exact C4 calibration",
    )
    script_hashes = current_c4_script_hashes()
    for name, digest in script_hashes.items():
        check(summary_hashes.get(name) == digest, f"final summary binds exact current {name}")
    return calibration, frozen_config, calibration_hash, frozen_config_hash


def verify_calibration_provenance(calibration: Mapping, v3_freeze: Mapping) -> None:
    validation_ids = calibration.get("validation_run_ids")
    check(
        validation_ids == [3, 9, 11, 16, 19, 23],
        "C4 calibration uses only the frozen clean-validation trajectory IDs",
    )
    check(
        calibration.get("calibration_population")
        == "clean validation trajectories from data/processed/dc_motor_lstm_dataset.npz only",
        "C4 calibration explicitly identifies the clean-validation population",
    )
    exclusions = calibration.get("exclusions", {})
    check(
        set(exclusions.get("development_seeds", [])) == DEVELOPMENT_SEEDS
        and set(exclusions.get("v3_final_holdout_seeds", [])) == V3_FINAL_SEEDS,
        "C4 calibration records exclusion of development and V3-final seed families",
    )
    check(
        exclusions.get("closed_loop_scenarios") == "not read or used"
        and exclusions.get("scenario_labels") == "not read or used"
        and exclusions.get("fault_labels_or_flags") == "not read or used",
        "C4 calibration provenance excludes closed-loop scenario/fault-label evidence",
    )

    provenance = calibration.get("provenance")
    check(isinstance(provenance, Mapping), "C4 calibration records provenance")
    dataset = provenance.get("dataset", {})
    dataset_path = PROJECT / str(dataset.get("path", ""))
    check(
        dataset_path.is_file() and sha256(dataset_path) == dataset.get("sha256"),
        "C4 calibration dataset hash reproduces exactly",
    )
    models = provenance.get("models", {})
    frozen_artifacts = v3_freeze["artifacts"]
    check(
        models.get("main_weights_sha256") == frozen_artifacts["main_model"]["sha256"]
        and models.get("auxiliary_weights_sha256") == frozen_artifacts["auxiliary_model"]["sha256"],
        "C4 calibration model hashes match the pre-C4 V3 freeze record",
    )
    v3 = provenance.get("v3", {})
    check(
        v3.get("arbitration_config_sha256") == frozen_artifacts["v3_arbitration_config"]["sha256"]
        and v3.get("arbitration_calibration_sha256") == frozen_artifacts["v3_calibration"]["sha256"]
        and v3.get("final_275_run_matrix_sha256") == frozen_artifacts["v3_final_275_run_matrix"]["sha256"]
        and v3.get("pre_c4_hash_record_sha256") == sha256(CONFIGS / "c4_v3_frozen_hashes.json"),
        "C4 calibration binds the exact frozen V3 config/calibration/275-run matrix and pre-C4 hash record",
    )
    calibration_code = provenance.get("calibration_code", {})
    code_path = PROJECT / str(calibration_code.get("path", ""))
    check(
        code_path.resolve() == (PROJECT / "scripts/calibrate_c4_attribution.py").resolve()
        and calibration_code.get("version") == "1.0.0"
        and calibration_code.get("sha256") == FROZEN_C4_CALIBRATION_GENERATOR_SHA256,
        "C4 calibration records the exact historical generator path/version/SHA256",
    )

    thresholds = calibration.get("computed_thresholds")
    check(isinstance(thresholds, Mapping), "C4 calibration contains computed threshold values")
    check(
        thresholds.get("classification_statistic") == "raw_pairwise_absolute_disagreement",
        "C4 calibrated classification statistic is raw pairwise absolute disagreement",
    )
    classification = calibration.get("classification_statistic", {})
    check(
        isinstance(classification, Mapping) and classification.get("temporal_filter") == "none",
        "C4 online classification has no temporal filter",
    )
    diagnostic_filter = calibration.get("diagnostic_filter_definition", {})
    check(
        isinstance(diagnostic_filter, Mapping)
        and diagnostic_filter.get("used_for_classification_or_veto") is False,
        "C4 diagnostic EWMA is explicitly excluded from classification and veto",
    )
    agreement = np.asarray(thresholds.get("agreement_thresholds_ordered"), dtype=float)
    disagreement = np.asarray(thresholds.get("disagreement_thresholds_ordered"), dtype=float)
    check(
        agreement.shape == (3,)
        and disagreement.shape == (3,)
        and np.isfinite(agreement).all()
        and np.isfinite(disagreement).all()
        and (agreement < disagreement).all(),
        "C4 calibrated agreement/disagreement bands are finite and ordered",
    )


def verify_development_config_contract(path: Path, calibration: Mapping) -> tuple[dict, dict[str, str]]:
    config = load_json(path)
    check(
        config.get("status") == "frozen_before_first_completed_c4_development_evidence",
        "C4 development config has the expected pre-evidence freeze status",
    )
    design = config.get("development_design")
    check(isinstance(design, Mapping), "C4 development config contains development_design")
    check(
        [int(seed) for seed in design.get("seeds", [])] == sorted(DEVELOPMENT_SEEDS),
        "C4 development config records the locked development seeds",
    )
    expected_scenarios = [
        "load_disturbance",
        "combined_fault_load",
        "sensor_bias_15",
        "sensor_dropout",
        "parameter_variation",
        "step_reference",
        "changing_reference",
    ]
    check(list(design.get("scenarios", [])) == expected_scenarios, "C4 development config records the locked scenario order")
    check(list(design.get("controllers", [])) == ["B_plain_MPC", "C3_arbitration_MPC", "C4_attribution_MPC"], "C4 development config records the locked controllers")
    check(int(design.get("expected_run_count", -1)) == 105, "C4 development config records the 105-run design")
    check(
        config.get("predeclared_criteria") == "results/configs/c4_go_no_go_criteria.json",
        "C4 development config references the canonical predeclared criteria",
    )

    frozen_hashes = config.get("hashes")
    check(isinstance(frozen_hashes, Mapping), "C4 development config contains frozen artifact hashes")
    expected_artifacts = {
        "c4_attribution_calibration_json": CONFIGS / "c4_attribution_calibration.json",
        "c4_calibration_script": PROJECT / "scripts/calibrate_c4_attribution.py",
        "c4_go_no_go_criteria_json": CONFIGS / "c4_go_no_go_criteria.json",
        "c4_evaluator_script": PROJECT / "scripts/evaluate_c4_closed_loop.py",
        "reliability_py": PROJECT / "src/reliability.py",
        "v3_main_model": PROJECT / "results/lstm_model_weights.pt",
        "v3_auxiliary_model": PROJECT / "results/auxiliary_model_weights.pt",
        "v3_arbitration_calibration": CONFIGS / "v3_arbitration_calibration.json",
        "v3_arbitration_config": CONFIGS / "v3_arbitration_config.json",
        "v3_final_275_run_matrix": METRICS / "v3_final_11scenario_runs.csv",
    }
    for name, artifact in expected_artifacts.items():
        check(frozen_hashes.get(name) == sha256(artifact), f"C4 development config frozen hash reproduces for {name}")
    check(
        frozen_hashes.get("c4_verifier_script") == FROZEN_C4_DEVELOPMENT_VERIFIER_SHA256,
        "C4 development config preserves the exact historical verifier hash recorded before development evidence",
    )

    thresholds = calibration["computed_thresholds"]
    attributor = config.get("attributor")
    check(isinstance(attributor, Mapping), "C4 development config contains a frozen attributor block")
    check(
        np.array_equal(np.asarray(attributor.get("agreement_thresholds"), dtype=float), np.asarray(thresholds["agreement_thresholds_ordered"], dtype=float))
        and np.array_equal(np.asarray(attributor.get("disagreement_thresholds"), dtype=float), np.asarray(thresholds["disagreement_thresholds_ordered"], dtype=float))
        and float(attributor.get("ewma_alpha")) == float(thresholds["diagnostic_ewma_alpha"])
        and int(attributor.get("enter_count")) == int(thresholds["enter_count"])
        and int(attributor.get("exit_count")) == int(thresholds["exit_count"]),
        "C4 development config attributor parameters exactly match calibration",
    )
    bindings = {
        "c4_development_config": sha256(path),
        "c4_attribution_calibration": sha256(CONFIGS / "c4_attribution_calibration.json"),
        "c4_go_no_go_criteria": sha256(CONFIGS / "c4_go_no_go_criteria.json"),
        "c4_evaluator_script": sha256(PROJECT / "scripts/evaluate_c4_closed_loop.py"),
        "c4_verifier_script": FROZEN_C4_DEVELOPMENT_VERIFIER_SHA256,
    }
    return config, bindings


def verify_frozen_config_against_calibration(
    calibration: Mapping,
    frozen_config: Mapping,
    final_seeds: set[int] | None = None,
) -> None:
    thresholds = calibration["computed_thresholds"]
    agreement = np.asarray(thresholds["agreement_thresholds_ordered"], dtype=float)
    disagreement = np.asarray(thresholds["disagreement_thresholds_ordered"], dtype=float)

    config_blocks = find_mappings_with_keys(
        frozen_config,
        {"agreement_thresholds", "disagreement_thresholds", "ewma_alpha", "enter_count", "exit_count"},
    )
    check(len(config_blocks) == 1, "frozen C4 config contains one unambiguous attributor parameter block")
    block = config_blocks[0]
    check(
        np.array_equal(np.asarray(block["agreement_thresholds"], dtype=float), agreement),
        "frozen C4 agreement thresholds exactly match calibration",
    )
    check(
        np.array_equal(np.asarray(block["disagreement_thresholds"], dtype=float), disagreement),
        "frozen C4 disagreement thresholds exactly match calibration",
    )
    check(
        float(block["ewma_alpha"]) == float(thresholds["diagnostic_ewma_alpha"]),
        "frozen C4 diagnostic EWMA alpha exactly matches calibration",
    )
    check(int(block["enter_count"]) == int(thresholds["enter_count"]), "frozen C4 attribution enter persistence exactly matches calibration")
    check(int(block["exit_count"]) == int(thresholds["exit_count"]), "frozen C4 attribution exit persistence exactly matches calibration")
    if final_seeds is not None:
        check(
            final_seeds.isdisjoint(set(int(value) for value in calibration["validation_run_ids"])),
            "C4 final seeds are disjoint from calibration validation trajectory IDs",
        )


def _seed_usage_in_json(path: Path, selected: set[int]) -> set[int]:
    """Return selected seeds used as evidence, ignoring explicit exclusion metadata."""
    value = load_json(path)
    hits: set[int] = set()
    ignored_terms = {
        "exclude",
        "excluded",
        "forbidden",
        "unused",
        "candidate",
        "reserved",
        "future",
        "final_holdout",
    }
    for key_path, leaf in _walk_json(value):
        if not isinstance(leaf, int) or isinstance(leaf, bool) or leaf not in selected:
            continue
        joined = ".".join(key_path).lower()
        if any(term in joined for term in ignored_terms):
            continue
        hits.add(int(leaf))
    return hits


def verify_final_seed_provenance(
    provenance_path: Path,
    selected: set[int],
    frozen_config: Mapping,
    summary: Mapping,
    runs_path: Path,
) -> None:
    provenance = load_json(provenance_path)
    provenance_reference = (
        frozen_config.get("seed_provenance")
        or frozen_config.get("seed_provenance_path")
        or frozen_config.get("final_seed_provenance")
        or frozen_config.get("candidate_seed_audit")
    )
    if not isinstance(provenance_reference, str) or not provenance_reference.strip():
        final_status = frozen_config.get("final_holdout_status")
        if isinstance(final_status, Mapping):
            provenance_reference = final_status.get("candidate_seed_audit") or final_status.get("seed_provenance")
    check(isinstance(provenance_reference, str) and bool(provenance_reference.strip()), "frozen final C4 config references a pre-use seed provenance artifact")
    referenced_path = (PROJECT / str(provenance_reference)).resolve()
    check(referenced_path == provenance_path.resolve(), "frozen final C4 config references the exact verifier-resolved seed provenance artifact")
    check(provenance_path.stat().st_mtime <= (CONFIGS / "c4_frozen_config.json").stat().st_mtime, "pre-use seed provenance predates the frozen final C4 config")
    frozen_hashes = frozen_config.get("hashes") if isinstance(frozen_config.get("hashes"), Mapping) else frozen_config
    provenance_hash = sha256(provenance_path)
    check(
        hash_binding(
            frozen_hashes,
            "c4_final_seed_provenance",
            "c4_final_seed_provenance_sha256",
            "c4_seed_provenance",
            "seed_provenance_sha256",
            "c4_candidate_seed_audit",
            "c4_candidate_seed_audit_json",
            "c4_candidate_seed_audit_sha256",
        ) == provenance_hash,
        "frozen final C4 config binds the exact pre-use seed provenance artifact",
    )
    check(
        has_named_seed_set(provenance, selected, ["seed"]),
        "pre-use provenance records the exact selected final seed set",
    )
    check(
        provenance_path.stat().st_mtime <= runs_path.stat().st_mtime,
        "final-seed provenance artifact predates the saved final-run matrix",
    )

    truthy_freshness = []
    for terms in (("unused",), ("fresh",), ("no_prior_use",), ("verified", "unused")):
        truthy_freshness.extend(value for value in find_named_values(provenance, terms) if value is True)
    empty_prior_matches = any(
        isinstance(value, list) and not value
        for terms in (("prior", "match"), ("prior", "use"), ("collision",))
        for value in find_named_values(provenance, terms)
    )
    zero_prior_matches = False
    match_maps = find_named_values(provenance, ("matches", "before", "use"))
    for value in match_maps:
        if isinstance(value, Mapping) and value and all(
            isinstance(count, (int, float)) and not isinstance(count, bool) and float(count) == 0.0
            for count in value.values()
        ):
            zero_prior_matches = True
    status = str(provenance.get("status", "")).lower()
    check(
        (bool(truthy_freshness) or empty_prior_matches or zero_prior_matches)
        and ("before" in status and ("final" in status or "use" in status)),
        "final-seed provenance explicitly records no prior-use evidence",
    )


def verify_final_seeds(
    selected: set[int],
    frozen_config: Mapping,
    summary: Mapping,
    runs_path: Path,
    provenance_path: Path,
) -> None:
    check(len(selected) == 5, "C4 final evidence uses exactly five seeds")
    check(selected.isdisjoint(DEVELOPMENT_SEEDS), "C4 final seeds are disjoint from C4 development seeds 19026-19030")
    check(selected.isdisjoint(V3_FINAL_SEEDS), "C4 final seeds are disjoint from inspected V3 final seeds 29026-29030")
    check(
        has_named_seed_set(frozen_config, selected, ["seed"]),
        "frozen C4 config records the exact selected final seed set",
    )
    check(
        has_named_seed_set(summary, selected, ["seed"]),
        "final summary records the exact selected final seed set",
    )
    verify_final_seed_provenance(provenance_path, selected, frozen_config, summary, runs_path)

    # Establish repository provenance from evidence that necessarily predates
    # the final run.  Merely declaring a future holdout seed in exclusion or
    # reservation metadata is allowed; using it as calibration/development
    # evidence is not.
    conflicts: dict[str, list[int]] = {}
    baseline = PROJECT / "results" / "baseline_v3_prefix"
    if baseline.exists():
        for path in baseline.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".md", ".py", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits = [seed for seed in selected if re.search(INT_TOKEN_RE_TEMPLATE.format(seed=seed), text)]
            if hits:
                conflicts[str(path.relative_to(PROJECT))] = hits

    prior_c4_json = [CONFIGS / "c4_attribution_calibration.json", CONFIGS / "c4_go_no_go_criteria.json"]
    prior_c4_json.extend(METRICS.glob("c4*development*.json"))
    prior_c4_json.extend(METRICS.glob("c4*dev*.json"))
    for path in prior_c4_json:
        if not path.is_file():
            continue
        hits = sorted(_seed_usage_in_json(path, selected))
        if hits:
            conflicts[str(path.relative_to(PROJECT))] = hits

    prior_c4_csv = list(METRICS.glob("c4*development*.csv")) + list(METRICS.glob("c4*dev*.csv"))
    for path in prior_c4_csv:
        if path.resolve() == runs_path.resolve():
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "seed" not in frame.columns:
            continue
        seeds = set(pd.to_numeric(frame["seed"], errors="coerce").dropna().astype(int))
        hits = sorted(seeds & selected)
        if hits:
            conflicts[str(path.relative_to(PROJECT))] = hits
    check(not conflicts, f"selected C4 final seeds are absent from prior calibration/development evidence (conflicts={conflicts})")


def verify_matrix(
    runs: pd.DataFrame,
    summary: Mapping,
    frozen_config: Mapping,
    runs_path: Path,
    provenance_path: Path,
) -> set[int]:
    require_columns(runs, ["controller", "scenario", "seed"], "C4 final run matrix")
    runs = runs.copy()
    runs["seed"] = coerce_seed_series(runs["seed"], "C4 final matrix")
    selected = set(int(value) for value in runs["seed"].unique())

    check(len(runs) == EXPECTED_RUN_COUNT, "C4 final primary matrix contains exactly 165 rows")
    check(set(runs["controller"]) == EXPECTED_CONTROLLERS, "C4 final matrix has exactly the three primary controllers")
    check(set(runs["scenario"]) == EXPECTED_SCENARIOS, "C4 final matrix has exactly the 11 frozen scenarios")
    check(len(selected) == 5, "C4 final matrix has exactly five seeds")

    actual_keys = pd.MultiIndex.from_frame(runs[["controller", "scenario", "seed"]])
    expected_keys = pd.MultiIndex.from_product(
        [sorted(EXPECTED_CONTROLLERS), sorted(EXPECTED_SCENARIOS), sorted(selected)],
        names=["controller", "scenario", "seed"],
    )
    check(not actual_keys.duplicated().any(), "C4 final matrix has no duplicate controller/scenario/seed keys")
    check(set(actual_keys) == set(expected_keys), "C4 final matrix has no missing controller/scenario/seed keys")

    for scenario in EXPECTED_SCENARIOS:
        subset = runs[runs["scenario"] == scenario]
        c3 = set(subset.loc[subset["controller"] == "C3_arbitration_MPC", "seed"].astype(int))
        c4 = set(subset.loc[subset["controller"] == "C4_attribution_MPC", "seed"].astype(int))
        check(c3 == c4 == selected, f"C3/C4 final seeds are paired exactly for {scenario}")

    if "run_count" in summary:
        check(int(summary["run_count"]) == EXPECTED_RUN_COUNT, "final summary run_count matches the 165-row matrix")
    verify_final_seeds(selected, frozen_config, summary, runs_path, provenance_path)
    return selected


def verify_development_matrix(runs: pd.DataFrame, summary: Mapping, criteria_json: Mapping) -> set[int]:
    scenarios = set(criteria_json.get("development_scenarios", []))
    controllers = set(criteria_json.get("controllers", []))
    seeds = set(int(value) for value in criteria_json.get("development_seeds", []))
    check(scenarios == {
        "load_disturbance",
        "combined_fault_load",
        "sensor_bias_15",
        "sensor_dropout",
        "parameter_variation",
        "step_reference",
        "changing_reference",
    }, "development evidence uses exactly the seven predeclared scenarios")
    check(controllers == EXPECTED_CONTROLLERS, "development evidence uses exactly B, C3, and C4")
    check(seeds == DEVELOPMENT_SEEDS, "development evidence uses exactly seeds 19026-19030")
    require_columns(runs, ["controller", "scenario", "seed"], "C4 development run matrix")
    actual = runs.copy()
    actual["seed"] = coerce_seed_series(actual["seed"], "C4 development matrix")
    expected_count = 3 * 7 * 5
    check(len(actual) == expected_count, "C4 development matrix contains exactly 105 paired rows")
    check(set(actual.controller) == controllers and set(actual.scenario) == scenarios and set(actual.seed) == seeds,
          "C4 development controller/scenario/seed sets match the predeclared design")
    keys = pd.MultiIndex.from_frame(actual[["controller", "scenario", "seed"]])
    expected = pd.MultiIndex.from_product(
        [sorted(controllers), sorted(scenarios), sorted(seeds)], names=["controller", "scenario", "seed"]
    )
    check(not keys.duplicated().any() and set(keys) == set(expected), "C4 development matrix has no duplicate or missing paired keys")
    if "run_count" in summary:
        check(int(summary["run_count"]) == expected_count, "development summary run_count matches the 105-row matrix")
    check(set(summary.get("seeds", [])) == seeds, "development summary records the exact development seeds")
    return seeds


def verify_safety(runs: pd.DataFrame, summary: Mapping) -> None:
    aliases = {
        "optimizer_failures": ("optimizer_failures",),
        "voltage_violations": ("voltage_violations",),
        "slew_violations": ("rate_violations", "slew_violations"),
        "nonfinite_events": ("nonfinite_events",),
    }
    totals: dict[str, int] = {}
    for semantic, choices in aliases.items():
        column = choose_column(runs, choices, semantic)
        values = numeric(runs, column, semantic)
        check((values >= 0).all() and np.allclose(values, np.round(values)), f"{semantic} counts are nonnegative integers")
        totals[semantic] = int(values.sum())
        check(totals[semantic] == 0, f"C4 final safety total is zero for {semantic}")

    for semantic, total in totals.items():
        possible = [semantic]
        if semantic == "slew_violations":
            possible.append("rate_violations")
        matched = [name for name in possible if name in summary]
        if matched:
            check(int(summary[matched[0]]) == total, f"summary {semantic} total reproduces from primary runs")


def _state_fraction_columns(runs: pd.DataFrame) -> dict[str, str]:
    aliases = {
        "NORMAL": ("attr_frac_normal", "attribution_frac_normal", "attribution_normal_fraction"),
        "LIKELY_SENSOR_FAULT": (
            "attr_frac_likely_sensor_fault",
            "attr_frac_sensor_fault",
            "attribution_frac_sensor_fault",
            "attribution_sensor_fault_fraction",
        ),
        "LIKELY_PLANT_OR_MAIN_MISMATCH": (
            "attr_frac_likely_plant_or_main_mismatch",
            "attr_frac_plant_or_main_mismatch",
            "attr_frac_plant_main_mismatch",
            "attribution_frac_plant_or_main_mismatch",
            "attribution_plant_or_main_mismatch_fraction",
        ),
        "LIKELY_AUX_MISMATCH": (
            "attr_frac_likely_aux_mismatch",
            "attr_frac_aux_mismatch",
            "attribution_frac_aux_mismatch",
            "attribution_aux_mismatch_fraction",
        ),
        "AMBIGUOUS": ("attr_frac_ambiguous", "attribution_frac_ambiguous", "ambiguous_fraction"),
    }
    return {state: choose_column(runs, names, f"C4 attribution-state fraction for {state}") for state, names in aliases.items()}


def verify_attribution_metrics(runs: pd.DataFrame, expected_c4_rows: int = 55) -> dict[str, str]:
    c4 = runs[runs["controller"] == "C4_attribution_MPC"].copy()
    check(len(c4) == expected_c4_rows, f"C4 controller contributes exactly {expected_c4_rows} evidence rows")

    aliases = {
        "false_entries": ("false_sensor_fault_entries", "false_fault_entries"),
        "suppressed_false": (
            "suppressed_false_entries",
            "suppressed_false_entry_count",
            "suppressed_false_sensor_fault_entries",
        ),
        "genuine_permitted": (
            "genuine_sensor_fault_entries_permitted",
            "genuine_sensor_fault_entries",
            "true_sensor_fault_entries_permitted",
            "genuine_fault_entries_permitted",
        ),
        "genuine_suppressed": (
            "genuine_sensor_fault_entries_suppressed",
            "genuine_sensor_fault_suppressed",
            "suppressed_genuine_sensor_fault_entries",
            "genuine_fault_entries_suppressed",
        ),
        "attribution_switches": ("attribution_switches", "attribution_state_switches"),
        "mean_dwell": ("mean_attribution_dwell_s", "attribution_mean_dwell_s"),
    }
    resolved = {semantic: choose_column(c4, names, semantic) for semantic, names in aliases.items()}

    for semantic in ("false_entries", "suppressed_false", "genuine_permitted", "genuine_suppressed", "attribution_switches"):
        values = numeric(c4, resolved[semantic], semantic)
        check((values >= 0).all() and np.allclose(values, np.round(values)), f"C4 {semantic} values are nonnegative integer counts")
    dwell = numeric(c4, resolved["mean_dwell"], "mean attribution dwell time")
    check((dwell >= 0).all(), "C4 mean attribution dwell times are nonnegative")

    fraction_columns = _state_fraction_columns(c4)
    fractions = np.column_stack([numeric(c4, column, column).to_numpy(float) for column in fraction_columns.values()])
    check(((fractions >= -1e-12) & (fractions <= 1 + 1e-12)).all(), "C4 attribution-state fractions lie in [0, 1]")
    check(np.allclose(fractions.sum(axis=1), 1.0, atol=1e-8, rtol=0), "C4 attribution-state fractions sum to one per run")
    return {**resolved, **{f"state:{state}": column for state, column in fraction_columns.items()}}


def _bool_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    check(normalized.isin(allowed).all(), f"{label} contains boolean values")
    return normalized.isin({"true", "1"})


def verify_attribution_events(
    events: pd.DataFrame,
    evidence_seeds: set[int],
    evidence_scenarios: set[str],
) -> None:
    require_columns(events, ["scenario", "seed"], "C4 attribution events")
    d_columns = {
        pair: choose_column(events, (pair, f"mean_{pair}"), f"event {pair}")
        for pair in ("d_sm", "d_sa", "d_ma")
    }
    state_col = choose_column(events, ("attribution_state", "state"), "attribution state")
    candidate_state_col = choose_column(events, ("candidate_state",), "attribution candidate state")
    candidate_col = choose_column(events, ("c3_requested_entry", "c3_entry_candidate"), "C3 entry-candidate diagnostic")
    suppressed_col = choose_column(events, ("c4_suppressed_entry", "entry_suppressed"), "C4 entry-suppression diagnostic")
    choose_column(events, ("selected_feedback_source", "selected_source", "source"), "selected feedback source")
    duration_col = choose_column(events, ("duration_s", "dwell_duration_s", "episode_duration_s"), "attribution episode duration")
    trusted_fraction_col = choose_column(
        events,
        ("auxiliary_trusted_fraction", "aux_trusted_fraction"),
        "auxiliary trusted fraction",
    )
    trust_reason_col = choose_column(
        events,
        ("auxiliary_trust_reason", "aux_trust_reason"),
        "auxiliary trust reason",
    )

    filtered_columns = {
        pair: choose_column(
            events,
            (f"filtered_{pair}", f"mean_filtered_{pair}"),
            f"filtered event {pair}",
        )
        for pair in ("d_sm", "d_sa", "d_ma")
    }

    events = events.copy()
    events["seed"] = coerce_seed_series(events["seed"], "C4 attribution event")
    final_events = events[events["seed"].isin(evidence_seeds)]
    check(not final_events.empty, "attribution event artifact contains the requested evidence")
    check(set(final_events["scenario"]) == evidence_scenarios, "attribution event evidence covers every requested scenario")
    check(set(final_events["seed"]) == evidence_seeds, "attribution event evidence covers every requested seed")
    states = set(final_events[state_col].dropna().astype(str))
    check(states <= ATTRIBUTION_STATES and states, f"attribution events use only declared states: {sorted(states)}")
    _bool_series(final_events[candidate_col], "C3 requested-entry flag")
    _bool_series(final_events[suppressed_col], "C4 suppressed-entry flag")
    duration = numeric(final_events, duration_col, "attribution event duration")
    check((duration >= 0).all(), "attribution event durations are nonnegative")
    for column in d_columns.values():
        values = numeric(final_events, column, f"event {column}")
        check((values >= 0).all(), f"event {column} values are nonnegative")

    # Filtered disagreements are diagnostic-only EWMAs.  The deployed
    # attributor resets all three to NaN whenever the auxiliary witness is
    # unavailable, while forcing attribution/candidate state to AMBIGUOUS.
    # Preserve fail-loud behavior for malformed, infinite, negative, partial,
    # or unexplained missing diagnostics.
    missing_masks: list[pd.Series] = []
    for column in filtered_columns.values():
        raw = final_events[column]
        values = pd.to_numeric(raw, errors="coerce")
        parse_failure = raw.notna() & values.isna()
        check(not parse_failure.any(), f"event {column} contains only numeric values or missing diagnostics")
        present = values.notna()
        check(np.isfinite(values[present].to_numpy(float)).all(), f"present event {column} values are finite")
        check((values[present] >= 0).all(), f"present event {column} values are nonnegative")
        missing_masks.append(values.isna())

    missing_count = sum(mask.astype(int) for mask in missing_masks)
    check(((missing_count == 0) | (missing_count == 3)).all(), "filtered event diagnostics are either fully present or jointly missing")

    jointly_missing = missing_count == 3
    if jointly_missing.any():
        trusted_fraction = pd.to_numeric(final_events[trusted_fraction_col], errors="coerce")
        missing_trust = trusted_fraction[jointly_missing]
        check(
            missing_trust.notna().all() and np.isfinite(missing_trust.to_numpy(float)).all(),
            "jointly missing filtered diagnostics retain finite auxiliary-trust evidence",
        )
        check(
            np.allclose(missing_trust.to_numpy(float), 0.0, rtol=0, atol=1e-12),
            "jointly missing filtered diagnostics occur only with zero trusted-auxiliary fraction",
        )
        missing_states = final_events.loc[jointly_missing, state_col].astype(str)
        missing_candidates = final_events.loc[jointly_missing, candidate_state_col].astype(str)
        check(
            (missing_states == "AMBIGUOUS").all() and (missing_candidates == "AMBIGUOUS").all(),
            "jointly missing filtered diagnostics occur only in forced AMBIGUOUS attribution",
        )
        missing_reasons = final_events.loc[jointly_missing, trust_reason_col].astype(str)
        check(
            missing_reasons.isin(AUXILIARY_UNAVAILABLE_REASONS).all(),
            "jointly missing filtered diagnostics have an explicit unavailable-auxiliary reason",
        )


def mean_metric(runs: pd.DataFrame, controller: str, scenario: str, column: str) -> float:
    values = pd.to_numeric(
        runs[(runs["controller"] == controller) & (runs["scenario"] == scenario)][column],
        errors="coerce",
    )
    check(len(values) == 5 and values.notna().all() and np.isfinite(values.to_numpy(float)).all(), f"five finite {column} values exist for {controller}/{scenario}")
    return float(values.mean())


def _relative_degradation(new: float, baseline: float) -> float:
    if abs(baseline) <= 1e-15:
        return 0.0 if abs(new) <= 1e-15 else math.inf
    return (new - baseline) / abs(baseline)


def _relative_reduction(new: float, baseline: float) -> float:
    if baseline <= 1e-15:
        return 1.0 if new <= 1e-15 else -math.inf
    return (baseline - new) / baseline


def _latency_mean_and_complete(runs: pd.DataFrame, controller: str, scenario: str, column: str) -> tuple[float, bool]:
    values = pd.to_numeric(
        runs[(runs["controller"] == controller) & (runs["scenario"] == scenario)][column],
        errors="coerce",
    )
    complete = bool(len(values) == 5 and values.notna().all() and np.isfinite(values.to_numpy(float)).all())
    return (float(values.mean()) if complete else math.nan), complete


def _detected_mask(frame: pd.DataFrame) -> pd.Series:
    for name in ("sensor_fault_detected", "genuine_sensor_fault_detected", "fault_detected"):
        if name in frame.columns:
            return _bool_series(frame[name], name)
    for name in ("genuine_sensor_fault_entries_permitted", "true_sensor_fault_entries", "reliability_entries"):
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce")
            check(values.notna().all(), f"{name} is complete for fault-detection evidence")
            return values > 0
    raise VerificationError("fault-detection evidence missing; expected an explicit detection flag/count or reliability_entries")


def _fault_detection_rate(runs: pd.DataFrame, controller: str, scenario: str) -> float:
    frame = runs[(runs["controller"] == controller) & (runs["scenario"] == scenario)]
    check(len(frame) == 5, f"five paired rows exist for detection-rate check {controller}/{scenario}")
    return float(_detected_mask(frame).mean())


def _false_entry_column_or_reliability(runs: pd.DataFrame) -> str:
    for name in ("false_sensor_fault_entries", "false_fault_entries"):
        if name in runs.columns:
            return name
    # load_disturbance has a healthy physical speed sensor by construction, so
    # each reliability latch is an offline-classified false sensor-fault entry.
    return choose_column(runs, ("reliability_entries",), "load-disturbance false-entry evidence")


def verify_final_criteria(runs: pd.DataFrame, criteria_json: Mapping, metric_columns: Mapping[str, str]) -> dict[str, float]:
    criteria = criteria_json.get("criteria")
    check(isinstance(criteria, dict), "go/no-go artifact contains quantitative criteria")
    check(criteria_json.get("status") == "predeclared_before_c4_development_results", "go/no-go criteria were predeclared before C4 development results")
    check(set(criteria_json.get("development_seeds", [])) == DEVELOPMENT_SEEDS, "predeclared development seed set remains 19026-19030")

    fault_rmse = choose_column(runs, ("fault_window_rmse", "fault_interval_RMSE"), "fault-window RMSE")
    overall_rmse = choose_column(runs, ("overall_rmse", "RMSE"), "overall RMSE")
    sub_fraction = choose_column(runs, ("sub_fraction", "substitution_fraction"), "substitution fraction")
    false_entries = _false_entry_column_or_reliability(runs)

    load_rule = criteria["load_disturbance"]
    c3_false = mean_metric(runs, "C3_arbitration_MPC", "load_disturbance", false_entries)
    c4_false = mean_metric(runs, "C4_attribution_MPC", "load_disturbance", false_entries)
    false_reduction = _relative_reduction(c4_false, c3_false)
    check(
        false_reduction >= float(load_rule["false_sensor_fault_entries_relative_reduction_min"]) - 1e-12,
        "load-disturbance false sensor-fault entries meet the predeclared reduction target",
    )

    c3_sub = mean_metric(runs, "C3_arbitration_MPC", "load_disturbance", sub_fraction)
    c4_sub = mean_metric(runs, "C4_attribution_MPC", "load_disturbance", sub_fraction)
    sub_reduction = _relative_reduction(c4_sub, c3_sub)
    check(
        sub_reduction >= float(load_rule["substitution_fraction_relative_reduction_min"]) - 1e-12,
        "load-disturbance substitution fraction meets the predeclared reduction target",
    )

    plain_load = mean_metric(runs, "B_plain_MPC", "load_disturbance", fault_rmse)
    c3_load = mean_metric(runs, "C3_arbitration_MPC", "load_disturbance", fault_rmse)
    c4_load = mean_metric(runs, "C4_attribution_MPC", "load_disturbance", fault_rmse)
    load_penalty = max(0.0, c3_load - plain_load)
    if load_penalty > 1e-15:
        penalty_removed = (c3_load - c4_load) / load_penalty
    else:
        penalty_removed = 1.0 if c4_load <= c3_load + 1e-12 else -math.inf
    check(
        penalty_removed >= float(load_rule["c3_tracking_penalty_toward_plain_fraction_removed_min"]) - 1e-12,
        "load-disturbance tracking penalty meets the predeclared recovery target",
    )

    load_rows = runs[runs["scenario"] == "load_disturbance"].pivot(index="seed", columns="controller", values=false_entries)
    no_c3_entry = pd.to_numeric(load_rows["C3_arbitration_MPC"], errors="coerce") == 0
    check(
        (pd.to_numeric(load_rows.loc[no_c3_entry, "C4_attribution_MPC"], errors="coerce") == 0).all(),
        "C4 creates no new load-disturbance false latch on seeds where C3 had none",
    )

    combined_rule = criteria["combined_fault_load"]
    c3_combined = mean_metric(runs, "C3_arbitration_MPC", "combined_fault_load", fault_rmse)
    c4_combined = mean_metric(runs, "C4_attribution_MPC", "combined_fault_load", fault_rmse)
    combined_degradation = _relative_degradation(c4_combined, c3_combined)
    check(
        combined_degradation <= float(combined_rule["mean_fault_window_rmse_relative_degradation_max"]) + 1e-12,
        "combined-fault C4 fault-window RMSE stays within the predeclared degradation limit",
    )
    check(
        _fault_detection_rate(runs, "C4_attribution_MPC", "combined_fault_load")
        >= _fault_detection_rate(runs, "C3_arbitration_MPC", "combined_fault_load") - 1e-12,
        "combined-fault sensor-fault detection rate does not drop versus C3",
    )
    genuine_suppressed = metric_columns["genuine_suppressed"]
    combined_c4 = runs[(runs["controller"] == "C4_attribution_MPC") & (runs["scenario"] == "combined_fault_load")]
    check(
        pd.to_numeric(combined_c4[genuine_suppressed], errors="coerce").sum() == 0,
        "C4 suppresses no genuine combined-fault sensor entries",
    )

    abrupt_rule = criteria["abrupt_sensor_faults"]
    detection_latency = choose_column(
        runs,
        ("detection_latency_s", "fault_detection_latency_s", "sensor_fault_detection_latency_s"),
        "abrupt-fault detection latency",
    )
    for scenario in abrupt_rule["scenarios"]:
        c3_rmse = mean_metric(runs, "C3_arbitration_MPC", scenario, fault_rmse)
        c4_rmse = mean_metric(runs, "C4_attribution_MPC", scenario, fault_rmse)
        check(
            _relative_degradation(c4_rmse, c3_rmse)
            <= float(abrupt_rule["mean_fault_window_rmse_relative_degradation_max"]) + 1e-12,
            f"{scenario} C4 fault-window RMSE stays within the abrupt-fault degradation limit",
        )
        check(
            _fault_detection_rate(runs, "C4_attribution_MPC", scenario)
            >= _fault_detection_rate(runs, "C3_arbitration_MPC", scenario) - 1e-12,
            f"{scenario} C4 detection rate does not drop versus C3",
        )
        c3_latency, c3_latency_complete = _latency_mean_and_complete(runs, "C3_arbitration_MPC", scenario, detection_latency)
        c4_latency, c4_latency_complete = _latency_mean_and_complete(runs, "C4_attribution_MPC", scenario, detection_latency)
        check(
            c3_latency_complete and c4_latency_complete,
            f"{scenario} has finite detection latency for all five paired C3/C4 runs under the declared missing-latency policy",
        )
        check(
            c4_latency - c3_latency <= float(abrupt_rule["mean_detection_latency_increase_s_max"]) + 1e-12,
            f"{scenario} C4 mean detection latency stays within the predeclared limit",
        )
        abrupt_c4 = runs[(runs["controller"] == "C4_attribution_MPC") & (runs["scenario"] == scenario)]
        check(
            pd.to_numeric(abrupt_c4[genuine_suppressed], errors="coerce").sum() == 0,
            f"C4 suppresses no genuine sensor-fault entries in {scenario}",
        )

    parameter_rule = criteria["parameter_variation"]
    plain_parameter = mean_metric(runs, "B_plain_MPC", "parameter_variation", fault_rmse)
    c3_parameter = mean_metric(runs, "C3_arbitration_MPC", "parameter_variation", fault_rmse)
    c4_parameter = mean_metric(runs, "C4_attribution_MPC", "parameter_variation", fault_rmse)
    better_parameter = min(plain_parameter, c3_parameter)
    check(
        _relative_degradation(c4_parameter, better_parameter)
        <= float(parameter_rule["mean_fault_window_rmse_relative_degradation_vs_better_of_plain_or_c3_max"]) + 1e-12,
        "parameter-variation C4 fault-window RMSE stays within the predeclared limit",
    )
    if parameter_rule.get("no_new_auxiliary_misuse"):
        misuse_col = choose_column(runs, ("auxiliary_misuse_events",), "parameter-variation auxiliary-misuse evidence")
        parameter_c4 = runs[(runs["controller"] == "C4_attribution_MPC") & (runs["scenario"] == "parameter_variation")]
        check(
            pd.to_numeric(parameter_c4[misuse_col], errors="coerce").sum() == 0,
            "C4 never selects auxiliary feedback while the auxiliary channel is hard-invalid",
        )

    reference_rule = criteria["reference_transients"]
    reliability_entries = choose_column(runs, ("reliability_entries",), "reference reliability-entry count")
    for scenario in reference_rule["scenarios"]:
        c4_ref = runs[(runs["controller"] == "C4_attribution_MPC") & (runs["scenario"] == scenario)]
        check(
            pd.to_numeric(c4_ref[reliability_entries], errors="coerce").sum()
            <= int(reference_rule["reliability_latches_max_total"]),
            f"{scenario} C4 reliability latches satisfy the predeclared reference limit",
        )
        c4_ref_sub = mean_metric(runs, "C4_attribution_MPC", scenario, sub_fraction)
        check(
            c4_ref_sub <= float(reference_rule["mean_substitution_fraction_max"]) + 1e-12,
            f"{scenario} C4 substitution fraction satisfies the predeclared reference limit",
        )
        c3_ref_rmse = mean_metric(runs, "C3_arbitration_MPC", scenario, overall_rmse)
        c4_ref_rmse = mean_metric(runs, "C4_attribution_MPC", scenario, overall_rmse)
        check(
            _relative_degradation(c4_ref_rmse, c3_ref_rmse)
            <= float(reference_rule["mean_overall_rmse_relative_degradation_vs_c3_max"]) + 1e-12,
            f"{scenario} C4 tracking stays within the predeclared reference-transient limit",
        )

    return {
        "c3_load_false_entries_mean": c3_false,
        "c4_load_false_entries_mean": c4_false,
        "load_false_entry_relative_reduction": false_reduction,
        "c3_load_sub_fraction_mean": c3_sub,
        "c4_load_sub_fraction_mean": c4_sub,
        "load_sub_fraction_relative_reduction": sub_reduction,
        "plain_load_fault_window_rmse_mean": plain_load,
        "c3_load_fault_window_rmse_mean": c3_load,
        "c4_load_fault_window_rmse_mean": c4_load,
        "load_tracking_penalty_fraction_removed": penalty_removed,
        "c3_combined_fault_window_rmse_mean": c3_combined,
        "c4_combined_fault_window_rmse_mean": c4_combined,
        "combined_relative_degradation": combined_degradation,
    }


def independently_assess_development_gate(runs: pd.DataFrame, criteria_json: Mapping) -> tuple[str, dict[str, bool]]:
    """Recompute the predeclared development gate without requiring it to pass."""
    rules = criteria_json["criteria"]
    decisions: dict[str, bool] = {}

    false_entries = choose_column(runs, ("false_sensor_fault_entries",), "development false-entry evidence")
    sub_fraction = choose_column(runs, ("sub_fraction",), "development substitution fraction")
    fault_rmse = choose_column(runs, ("fault_window_rmse",), "development fault-window RMSE")
    overall_rmse = choose_column(runs, ("overall_rmse",), "development overall RMSE")
    detection_latency = choose_column(runs, ("detection_latency_s",), "development detection latency")
    genuine_suppressed = choose_column(
        runs,
        ("genuine_sensor_fault_suppressed", "genuine_sensor_fault_entries_suppressed"),
        "development genuine-fault suppression count",
    )

    load = rules["load_disturbance"]
    c3_false = mean_metric(runs, "C3_arbitration_MPC", "load_disturbance", false_entries)
    c4_false = mean_metric(runs, "C4_attribution_MPC", "load_disturbance", false_entries)
    decisions["load_false_entry"] = _relative_reduction(c4_false, c3_false) >= float(
        load["false_sensor_fault_entries_relative_reduction_min"]
    )
    c3_sub = mean_metric(runs, "C3_arbitration_MPC", "load_disturbance", sub_fraction)
    c4_sub = mean_metric(runs, "C4_attribution_MPC", "load_disturbance", sub_fraction)
    decisions["load_substitution"] = _relative_reduction(c4_sub, c3_sub) >= float(
        load["substitution_fraction_relative_reduction_min"]
    )
    plain_load = mean_metric(runs, "B_plain_MPC", "load_disturbance", fault_rmse)
    c3_load = mean_metric(runs, "C3_arbitration_MPC", "load_disturbance", fault_rmse)
    c4_load = mean_metric(runs, "C4_attribution_MPC", "load_disturbance", fault_rmse)
    load_penalty = max(0.0, c3_load - plain_load)
    removed = (
        (c3_load - c4_load) / load_penalty
        if load_penalty > 1e-15
        else (1.0 if c4_load <= c3_load + 1e-12 else -math.inf)
    )
    decisions["load_tracking"] = removed >= float(load["c3_tracking_penalty_toward_plain_fraction_removed_min"])
    load_pivot = runs[runs.scenario == "load_disturbance"].pivot(index="seed", columns="controller", values=false_entries)
    decisions["load_no_new_false_entry"] = bool(
        ((pd.to_numeric(load_pivot["C3_arbitration_MPC"]) > 0) | (pd.to_numeric(load_pivot["C4_attribution_MPC"]) == 0)).all()
    )

    combined = rules["combined_fault_load"]
    c3_comb = mean_metric(runs, "C3_arbitration_MPC", "combined_fault_load", fault_rmse)
    c4_comb = mean_metric(runs, "C4_attribution_MPC", "combined_fault_load", fault_rmse)
    decisions["combined_rmse"] = _relative_degradation(c4_comb, c3_comb) <= float(
        combined["mean_fault_window_rmse_relative_degradation_max"]
    )
    decisions["combined_detection"] = _fault_detection_rate(
        runs, "C4_attribution_MPC", "combined_fault_load"
    ) + 1e-12 >= _fault_detection_rate(runs, "C3_arbitration_MPC", "combined_fault_load")
    combined_c4 = runs[(runs.controller == "C4_attribution_MPC") & (runs.scenario == "combined_fault_load")]
    decisions["combined_no_genuine_suppression"] = int(
        pd.to_numeric(combined_c4[genuine_suppressed], errors="coerce").sum()
    ) == 0

    abrupt = rules["abrupt_sensor_faults"]
    for scenario in abrupt["scenarios"]:
        c3_value = mean_metric(runs, "C3_arbitration_MPC", scenario, fault_rmse)
        c4_value = mean_metric(runs, "C4_attribution_MPC", scenario, fault_rmse)
        decisions[f"{scenario}_rmse"] = _relative_degradation(c4_value, c3_value) <= float(
            abrupt["mean_fault_window_rmse_relative_degradation_max"]
        )
        decisions[f"{scenario}_detection"] = _fault_detection_rate(
            runs, "C4_attribution_MPC", scenario
        ) + 1e-12 >= _fault_detection_rate(runs, "C3_arbitration_MPC", scenario)
        c3_latency, c3_latency_complete = _latency_mean_and_complete(runs, "C3_arbitration_MPC", scenario, detection_latency)
        c4_latency, c4_latency_complete = _latency_mean_and_complete(runs, "C4_attribution_MPC", scenario, detection_latency)
        latency_complete = bool(c3_latency_complete and c4_latency_complete)
        decisions[f"{scenario}_latency"] = bool(
            latency_complete
            and np.isfinite(c4_latency - c3_latency)
            and c4_latency - c3_latency <= float(abrupt["mean_detection_latency_increase_s_max"])
        )
        c4_rows = runs[(runs.controller == "C4_attribution_MPC") & (runs.scenario == scenario)]
        decisions[f"{scenario}_no_genuine_suppression"] = int(
            pd.to_numeric(c4_rows[genuine_suppressed], errors="coerce").sum()
        ) == 0

    parameter = rules["parameter_variation"]
    plain_parameter = mean_metric(runs, "B_plain_MPC", "parameter_variation", fault_rmse)
    c3_parameter = mean_metric(runs, "C3_arbitration_MPC", "parameter_variation", fault_rmse)
    c4_parameter = mean_metric(runs, "C4_attribution_MPC", "parameter_variation", fault_rmse)
    decisions["parameter_rmse"] = _relative_degradation(
        c4_parameter, min(plain_parameter, c3_parameter)
    ) <= float(parameter["mean_fault_window_rmse_relative_degradation_vs_better_of_plain_or_c3_max"])
    misuse = choose_column(runs, ("auxiliary_misuse_events",), "development auxiliary-misuse count")
    parameter_c4 = runs[(runs.controller == "C4_attribution_MPC") & (runs.scenario == "parameter_variation")]
    decisions["parameter_no_aux_misuse"] = int(pd.to_numeric(parameter_c4[misuse], errors="coerce").sum()) == 0

    reference = rules["reference_transients"]
    for scenario in reference["scenarios"]:
        c4_rows = runs[(runs.controller == "C4_attribution_MPC") & (runs.scenario == scenario)]
        decisions[f"{scenario}_latches"] = int(pd.to_numeric(c4_rows["reliability_entries"], errors="coerce").sum()) <= int(
            reference["reliability_latches_max_total"]
        )
        decisions[f"{scenario}_substitution"] = float(pd.to_numeric(c4_rows[sub_fraction], errors="coerce").mean()) <= float(
            reference["mean_substitution_fraction_max"]
        )
        c3_ref = mean_metric(runs, "C3_arbitration_MPC", scenario, overall_rmse)
        c4_ref = mean_metric(runs, "C4_attribution_MPC", scenario, overall_rmse)
        decisions[f"{scenario}_tracking"] = _relative_degradation(c4_ref, c3_ref) <= float(
            reference["mean_overall_rmse_relative_degradation_vs_c3_max"]
        )

    safety = rules["safety"]
    c4 = runs[runs.controller == "C4_attribution_MPC"]
    safety_columns = {
        "optimizer_failures": "optimizer_failures",
        "voltage_violations": "voltage_violations",
        "slew_violations": choose_column(runs, ("rate_violations", "slew_violations"), "development slew violations"),
        "nonfinite_events": "nonfinite_events",
    }
    for semantic, column in safety_columns.items():
        values = numeric(c4, column, f"development {semantic}")
        check((values >= 0).all() and np.allclose(values, np.round(values)), f"development {semantic} are valid counts")
        decisions[f"safety_{semantic}"] = int(values.sum()) == int(safety[semantic])

    decision = "GO" if all(decisions.values()) else "NO_GO"
    print(f"PASS: independently recomputed development stage-gate decision = {decision}")
    failed = sorted(name for name, passed in decisions.items() if not passed)
    if failed:
        print(f"INFO: development gate failed criteria: {failed}")
    return decision, decisions


def verify_runtime(runtime: pd.DataFrame, summary: Mapping, evidence_seeds: set[int]) -> dict[str, float]:
    timing_col = choose_column(
        runtime,
        ("attribution_ms", "attribution_overhead_ms", "c4_attribution_ms", "attribution_update_ms"),
        "separate C4 attribution runtime",
    )
    values = numeric(runtime, timing_col, "C4 attribution runtime samples")
    check(len(values) > 0 and (values >= 0).all(), "C4 attribution runtime evidence contains nonnegative samples")
    if "seed" in runtime.columns:
        runtime_seeds = set(coerce_seed_series(runtime["seed"], "C4 runtime"))
        check(evidence_seeds <= runtime_seeds, "runtime evidence covers every requested C4 seed")
    if "controller" in runtime.columns:
        check(
            set(runtime["controller"].dropna().astype(str)) <= {"C4_attribution_MPC"},
            "attribution runtime samples are scoped to C4 attribution",
        )

    stats = {
        "mean_ms": float(values.mean()),
        "median_ms": float(values.median()),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }
    runtime_summary = summary.get("attribution_runtime")
    if runtime_summary is None and isinstance(summary.get("runtime"), dict):
        runtime_summary = summary["runtime"].get("attribution")
    check(isinstance(runtime_summary, Mapping), "summary records separate C4 attribution runtime evidence")
    required_saved = ("mean_ms", "median_ms", "p95_ms", "p99_ms")
    check(
        all(name in runtime_summary for name in required_saved),
        "summary records canonical raw attribution runtime mean/median/p95/p99",
    )
    check(
        all(np.isfinite(float(runtime_summary[name])) and float(runtime_summary[name]) >= 0 for name in required_saved),
        "summary attribution runtime headline values are finite and nonnegative",
    )
    for name in required_saved:
        check(
            np.isclose(float(runtime_summary[name]), stats[name], rtol=1e-12, atol=1e-12),
            f"summary {name} attribution runtime reproduces from raw timing samples",
        )
    if "sample_count" in runtime_summary:
        check(
            int(runtime_summary["sample_count"]) == len(values),
            "summary attribution runtime sample count matches raw timing samples",
        )
    return stats


def _numeric_token_present(text: str, value: float) -> bool:
    candidates = {
        f"{value:.3f}",
        f"{value:.4f}",
        f"{value:.5f}",
        f"{value:.6f}",
        f"{value:.3g}",
        f"{value:.4g}",
    }
    return any(token in text for token in candidates)


def verify_documentation(path: Path, final_seeds: set[int], headline: Mapping[str, float], runtime: Mapping[str, float]) -> None:
    text = path.read_text(encoding="utf-8")
    check("three-way consistency attribution" in text.lower(), "C4 documentation names the three-way consistency attribution method")
    check("165" in text, "C4 documentation records the 165-run primary final matrix")
    check(all(str(seed) in text for seed in sorted(final_seeds)), "C4 documentation records all fresh final holdout seeds")
    check("AMBIGUOUS" in text and "C3" in text, "C4 documentation describes ambiguous-state fallback to C3 semantics")

    headline_names = (
        "c3_load_fault_window_rmse_mean",
        "c4_load_fault_window_rmse_mean",
        "c3_combined_fault_window_rmse_mean",
        "c4_combined_fault_window_rmse_mean",
    )
    missing = [name for name in headline_names if not _numeric_token_present(text, float(headline[name]))]
    check(not missing, f"C4 documentation contains critical load/combined headline values (missing={missing})")
    runtime_missing = [name for name in ("mean_ms", "median_ms", "p95_ms", "p99_ms") if not _numeric_token_present(text, float(runtime[name]))]
    check(not runtime_missing, f"C4 documentation contains attribution runtime headline values (missing={runtime_missing})")


def verify_development_only(v3_freeze: Mapping) -> None:
    paths = resolve_development_artifacts()
    calibration = load_json(paths["calibration"])
    verify_calibration_provenance(calibration, v3_freeze)
    _, expected_bindings = verify_development_config_contract(paths["development_config"], calibration)
    verify_no_oracle_online_logic()

    criteria_json = load_json(paths["criteria"])
    summary = load_json(paths["summary"])
    check(summary.get("mode") == "development", "C4 development summary is explicitly development evidence")
    check(
        summary.get("evidence_role") == "development_evidence",
        "C4 development summary is not mislabeled as final evidence",
    )
    check(summary.get("latency_policy") == LATENCY_POLICY, "development summary records the declared missing-latency policy")
    summary_hashes = summary.get("hashes")
    check(isinstance(summary_hashes, Mapping), "development summary contains a dedicated hash block")
    for name, digest in expected_bindings.items():
        check(summary_hashes.get(name) == digest, f"development summary binds exact current {name}")
    calibration_hash = sha256(paths["calibration"])
    check(
        json_contains_hash(summary, calibration_hash),
        "development summary cryptographically binds the exact C4 calibration",
    )

    runs = pd.read_csv(paths["runs"], float_precision="round_trip")
    dev_seeds = verify_development_matrix(runs, summary, criteria_json)
    verify_attribution_metrics(runs, expected_c4_rows=35)
    events = pd.read_csv(paths["events"], float_precision="round_trip")
    verify_attribution_events(events, dev_seeds, set(criteria_json["development_scenarios"]))
    runtime = pd.read_csv(paths["runtime"], float_precision="round_trip")
    verify_runtime(runtime, summary, dev_seeds)

    independent_decision, _ = independently_assess_development_gate(runs, criteria_json)
    saved_gate = load_json(paths["gate"])
    check(saved_gate.get("decision") in {"GO", "NO_GO"}, "saved development stage gate has a valid decision")
    check(saved_gate.get("latency_policy") == LATENCY_POLICY, "saved development stage gate records the declared missing-latency policy")
    saved_bindings = saved_gate.get("bindings")
    check(isinstance(saved_bindings, Mapping), "saved development stage gate contains binding hashes")
    for name, digest in expected_bindings.items():
        check(saved_bindings.get(name) == digest, f"saved development stage gate binds exact current {name}")
    check(
        saved_gate["decision"] == independent_decision,
        "saved development stage-gate decision matches independent verifier recomputation",
    )
    summary_gate = summary.get("development_stage_gate")
    check(isinstance(summary_gate, Mapping), "development summary contains its saved stage-gate record")
    check(
        summary_gate.get("decision") == independent_decision,
        "development summary stage-gate decision matches independent recomputation",
    )
    print(f"\nC4 calibration SHA256: {calibration_hash}")
    print(f"C4 development stage gate: {independent_decision}")
    print("C4 DEVELOPMENT VERIFIER RESULT: PASS")


def verify_saved_development_go_for_final(v3_freeze: Mapping, frozen_config: Mapping) -> None:
    paths = resolve_development_artifacts()
    calibration = load_json(paths["calibration"])
    verify_calibration_provenance(calibration, v3_freeze)
    _, expected_bindings = verify_development_config_contract(paths["development_config"], calibration)
    summary = load_json(paths["summary"])
    check(summary.get("mode") == "development" and summary.get("evidence_role") == "development_evidence", "final release uses a valid saved development summary")
    check(summary.get("latency_policy") == LATENCY_POLICY, "saved development summary uses the declared missing-latency policy")
    summary_hashes = summary.get("hashes")
    check(isinstance(summary_hashes, Mapping), "saved development summary contains binding hashes")
    for name, digest in expected_bindings.items():
        check(summary_hashes.get(name) == digest, f"saved development summary binds exact current {name}")

    runs = pd.read_csv(paths["runs"], float_precision="round_trip")
    verify_development_matrix(runs, summary, load_json(paths["criteria"]))
    independent_decision, _ = independently_assess_development_gate(runs, load_json(paths["criteria"]))
    saved_gate = load_json(paths["gate"])
    check(saved_gate.get("decision") == "GO", "actual saved c4_development_stage_gate.json is GO")
    check(independent_decision == "GO", "saved 105-run development evidence independently recomputes to GO")
    check(saved_gate.get("latency_policy") == LATENCY_POLICY, "saved GO gate uses the declared missing-latency policy")
    gate_bindings = saved_gate.get("bindings")
    check(isinstance(gate_bindings, Mapping), "saved GO gate contains binding hashes")
    for name, digest in expected_bindings.items():
        check(gate_bindings.get(name) == digest, f"saved GO gate binds exact current {name}")

    frozen_hashes = frozen_config.get("hashes") if isinstance(frozen_config.get("hashes"), Mapping) else frozen_config
    stage_gate_binding = hash_binding(
        frozen_hashes,
        "c4_development_stage_gate",
        "c4_development_stage_gate_sha256",
        "development_stage_gate_sha256",
    )
    development_config_binding = hash_binding(
        frozen_hashes,
        "c4_development_config",
        "c4_development_config_sha256",
        "development_config_sha256",
    )
    check(
        stage_gate_binding == sha256(paths["gate"]),
        "frozen final C4 config binds the exact saved GO development stage gate",
    )
    check(
        development_config_binding == sha256(paths["development_config"]),
        "frozen final C4 config binds the exact development config",
    )


def verify_final(v3_freeze: Mapping) -> None:
    paths = resolve_c4_artifacts()
    summary = load_json(paths["summary"])
    check(summary.get("mode") == "final", "C4 final summary is explicitly final evidence")
    check(summary.get("evidence_role") == "fresh_c4_final_holdout", "C4 final summary is labeled fresh final holdout evidence")
    check(summary.get("latency_policy") == LATENCY_POLICY, "C4 final summary records the declared missing-latency policy")
    calibration, frozen_config, calibration_hash, frozen_config_hash = verify_c4_hash_chain(paths, summary)
    verify_calibration_provenance(calibration, v3_freeze)
    verify_frozen_config_against_calibration(calibration, frozen_config)
    verify_saved_development_go_for_final(v3_freeze, frozen_config)
    verify_no_oracle_online_logic()

    runs = pd.read_csv(paths["runs"], float_precision="round_trip")
    final_seeds = verify_matrix(
        runs,
        summary,
        frozen_config,
        paths["runs"],
        paths["seed_provenance"],
    )
    verify_frozen_config_against_calibration(calibration, frozen_config, final_seeds)
    verify_safety(runs, summary)
    metric_columns = verify_attribution_metrics(runs, expected_c4_rows=55)

    events = pd.read_csv(paths["events"], float_precision="round_trip")
    verify_attribution_events(events, final_seeds, EXPECTED_SCENARIOS)
    criteria_json = load_json(paths["criteria"])
    headline = verify_final_criteria(runs, criteria_json, metric_columns)

    runtime = pd.read_csv(paths["runtime"], float_precision="round_trip")
    runtime_stats = verify_runtime(runtime, summary, final_seeds)
    verify_documentation(paths["documentation"], final_seeds, headline, runtime_stats)

    print(f"\nC4 calibration SHA256: {calibration_hash}")
    print(f"C4 frozen config SHA256: {frozen_config_hash}")
    print(f"C4 final seeds: {sorted(final_seeds)}")
    print("C4 VERIFIER RESULT: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Verify development evidence and the GO/NO_GO stage gate without requiring final-holdout artifacts.",
    )
    args = parser.parse_args()
    v3_freeze = verify_v3_freeze()
    if args.development_only:
        verify_development_only(v3_freeze)
    else:
        verify_final(v3_freeze)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nC4 VERIFIER RESULT: FAIL\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
