"""Plan or execute only the preregistered H1-H9 V4 core."""

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_protocol_faultfree as faultfree
import v4_protocol_recovery as recovery
import v4_protocol_robustness as robustness
import v4_protocol_sweep as sweep
import v4_protocol_timing as timing
from v4_protocol_common import (
    C3_LEGACY_SEEDS,
    PROJECT,
    TRAINING_SEEDS,
    execute,
    scientific_key,
)

PLAN_PATH = PROJECT / "results" / "v4" / "prereg" / "h1_h9_execution_plan.json"
HASHES_PATH = PROJECT / "results" / "v4" / "prereg" / "preexecution_hashes.json"
STATUS_PATH = PROJECT / "results" / "v4" / "core_execution_status.json"
PLAN_RELATIVE_PATH = "results/v4/prereg/h1_h9_execution_plan.json"


def _select(cells, controllers, seeds, predicate=lambda cell: True):
    return [cell for cell in cells
            if cell.controller in controllers
            and cell.training_seed in seeds and predicate(cell)]


def hypothesis_cells() -> dict[str, list]:
    common = set(C3_LEGACY_SEEDS)
    all_v4 = set(TRAINING_SEEDS)
    ff = faultfree.plan(controllers=["C3", "V4_full_aux",
                                     "V4_full_ekf", "B"])
    sw = sweep.plan(controllers=["C3", "V4_full_aux",
                                 "V4_full_ekf", "B"], with_det=False)
    rec = recovery.plan(controllers=["C3", "V4_full_aux"])
    bias2 = lambda cell: (cell.fault.kind == "bias"
                          and cell.fault.magnitude_sigma == 2.0)
    bias8 = lambda cell: (cell.fault.kind == "bias"
                          and cell.fault.magnitude_sigma == 8.0)
    load15 = lambda cell: (cell.fault.kind == "load"
                           and cell.fault.load_step_Nm == 0.15)
    finite_bias8 = lambda cell: bias8(cell) and cell.duration == 10.0
    return {
        "H1": _select(ff, {"C3", "V4_full_aux"}, common),
        "H2": _select(ff, {"C3", "V4_full_ekf"}, common),
        "H3": _select(sw, {"C3", "V4_full_aux"}, common, bias2),
        "H4": _select(sw, {"C3", "V4_full_ekf"}, common, bias2),
        "H5": _select(rec, {"C3", "V4_full_aux"}, common, finite_bias8),
        "H6": _select(sw, {"B", "V4_full_aux"}, all_v4, bias8),
        "H7": _select(ff, {"B", "C3", "V4_full_aux"}, common),
        "H8": _select(sw, {"C3", "V4_full_aux"}, common, load15),
        "H9": _select(sw, {"C3", "V4_full_ekf"}, common, load15),
    }


def core_cells(frames: dict[str, list] | None = None) -> list:
    frames = hypothesis_cells() if frames is None else frames
    cells, seen = [], set()
    for hypothesis in [f"H{i}" for i in range(1, 10)]:
        for cell in frames[hypothesis]:
            key = scientific_key(cell)
            if key not in seen:
                seen.add(key)
                cells.append(cell)
    return cells


def _protocol(cell) -> str:
    if cell.duration == recovery.DURATION:
        return "recovery"
    if cell.fault.kind == "none":
        return "faultfree"
    return "sweep"


def _number(value):
    return value if isinstance(value, int) or math.isfinite(value) else None


def _cell_record(cell, required_by) -> dict:
    fault = cell.fault
    return {
        "scientific_key": scientific_key(cell),
        "required_by": required_by,
        "protocol": _protocol(cell),
        "controller": cell.controller,
        "training_seed": cell.training_seed,
        "simulation_seed": cell.simulation_seed,
        "reference": cell.reference,
        "fault_kind": fault.kind,
        "fault_magnitude_sigma": _number(fault.magnitude_sigma),
        "fault_magnitude_rad_s": _number(fault.magnitude_rad_s),
        "dropout_duration_s": fault.dropout_duration_s,
        "drift_rate_rad_s2": fault.drift_rate_rad_s2,
        "load_step_Nm": fault.load_step_Nm,
        "fault_onset_s": _number(fault.onset_s),
        "fault_end_s": fault.end_s,
        "duration_s": cell.duration,
        "control_stride": cell.control_stride,
        "threshold_scale": cell.threshold_scale,
    }


def build_plan() -> tuple[dict, list]:
    frames = hypothesis_cells()
    cells = core_cells(frames)
    required_by: dict[str, list[str]] = {}
    for hypothesis, hypothesis_plan in frames.items():
        for cell in hypothesis_plan:
            required_by.setdefault(scientific_key(cell), []).append(hypothesis)
    original_plans = {
        "faultfree": faultfree.plan(),
        "sweep": sweep.plan(),
        "recovery": recovery.plan(),
        "robustness": robustness.plan(),
        "timing": timing.plan(),
    }
    core_by_protocol = {
        name: [cell for cell in cells if _protocol(cell) == name]
        for name in original_plans
    }
    protocol_counts = {}
    missing = []
    for name, original in original_plans.items():
        original_keys = {scientific_key(cell) for cell in original}
        core_keys = {scientific_key(cell) for cell in core_by_protocol[name]}
        absent = core_keys - original_keys
        missing.extend(absent)
        overlap = len(core_keys & original_keys)
        protocol_counts[name] = {
            "original_cells": len(original),
            "h1_h9_required_cells": len(core_keys),
            "required_present_in_original": overlap,
            "required_missing_from_original": len(absent),
            "original_cells_deferred": len(original) - overlap,
        }
    plan = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Exact deduplicated pre-execution plan for H1-H9",
        "c3_comparison_training_seeds": C3_LEGACY_SEEDS,
        "h6_training_seeds": TRAINING_SEEDS,
        "hypothesis_cell_counts": {key: len(value)
                                    for key, value in frames.items()},
        "union_count": len(cells),
        "original_umbrella_count": sum(len(value)
                                        for value in original_plans.values()),
        "original_umbrella_required_intersection": sum(
            value["required_present_in_original"]
            for value in protocol_counts.values()),
        "original_umbrella_deferred": sum(
            value["original_cells_deferred"]
            for value in protocol_counts.values()),
        "required_missing_from_original_umbrella": len(missing),
        "protocol_counts": protocol_counts,
        "cells": [_cell_record(cell, required_by[scientific_key(cell)])
                  for cell in cells],
    }
    return plan, cells


def validated_frozen_plan() -> tuple[dict, list, str]:
    """Load the frozen plan without ever modifying it."""
    expected = json.loads(HASHES_PATH.read_text())["files"][PLAN_RELATIVE_PATH]
    actual = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"frozen execution-plan SHA256 mismatch: expected {expected}, "
            f"got {actual}")
    frozen = json.loads(PLAN_PATH.read_text())
    generated, cells = build_plan()
    generated["generated_utc"] = frozen.get("generated_utc")
    if generated != frozen:
        raise RuntimeError("generated H1-H9 cells differ from the frozen plan")
    return frozen, cells, actual


def write_status(status: dict) -> None:
    """Atomically write mutable runtime metadata outside frozen artifacts."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(temporary, STATUS_PATH)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    plan, cells, plan_hash = validated_frozen_plan()
    print(json.dumps({key: plan[key] for key in (
        "hypothesis_cell_counts", "union_count", "original_umbrella_count",
        "original_umbrella_required_intersection",
        "original_umbrella_deferred",
        "required_missing_from_original_umbrella")}, indent=2))
    print(f"plan={PLAN_PATH}")
    if args.dry_run:
        return
    status = {
        "state": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "workers": args.workers,
        "frozen_plan_sha256": plan_hash,
        "completed_protocols": [],
    }
    write_status(status)
    try:
        for name in ("faultfree", "sweep", "recovery"):
            protocol_cells = [cell for cell in cells if _protocol(cell) == name]
            execute(name, PROJECT / "results" / "v4" / name,
                    protocol_cells, workers=args.workers)
            status["completed_protocols"].append(name)
            write_status(status)
    except BaseException:
        status["state"] = "interrupted"
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_status(status)
        raise
    status["state"] = "complete"
    status["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_status(status)


if __name__ == "__main__":
    main()
