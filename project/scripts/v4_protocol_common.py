"""Shared grids, seed blocks, and I/O for the V4 evaluation protocols.

Seed blocks (frozen in V4_PREREGISTRATION.md before any run):
- fault-free confirmatory sim seeds: 71000-71999.
- severity/robustness sim seeds: 72000-72099.
- training seeds: 2026-2036 (2026-2028 frozen, 2029-2036 new).
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from v4_closed_loop import (  # noqa: E402
    Fault,
    RunConfig,
    run_batch,
    sigma_to_rad_s,
)

TRAINING_SEEDS = list(range(2026, 2037))
C3_LEGACY_SEEDS = [2026, 2027, 2028]
FAULTFREE_SEEDS = list(range(71000, 71200))
SEVERITY_SEEDS = list(range(72000, 72100))
# Preregistered prefixes of the severity block (compute budget; the full
# block stays frozen for exploratory follow-ups).
SWEEP_SEEDS = SEVERITY_SEEDS[:50]
RECOVERY_SEEDS = SEVERITY_SEEDS  # full 100 (small grid)
ROBUSTNESS_SEEDS = SEVERITY_SEEDS[:25]
DET_SEEDS = SEVERITY_SEEDS[:20]
# Dedicated smoke seeds (outside every confirmatory block): protocol
# --smoke modes and engine smoke tests use ONLY these, so confirmatory
# blocks stay literally unexecuted until the confirmatory runs.
SMOKE_SEEDS = [99101, 99102]
SMOKE_DATASET_SEED = 99001
REFERENCES = ["nominal", "step", "changing", "wide"]

PRIMARY_V4 = ["V4_full_aux", "V4_full_ekf"]
STRUCTURAL_CONTROL = ["V4_frozen_baseline_aux"]
BASELINES = ["E2", "S1", "S2", "S3"]
LEGACY = ["C3"]
ANCHOR = ["B"]
FULL_DETECTOR_SET = PRIMARY_V4 + STRUCTURAL_CONTROL + BASELINES + LEGACY


def training_seeds_for(controller: str) -> list[int]:
    if controller == "C3":
        return list(C3_LEGACY_SEEDS)
    if controller in ("PI",):
        return [None]
    return list(TRAINING_SEEDS)


def expand(cells: list[RunConfig]) -> list[RunConfig]:
    return cells


def describe(cells: list[RunConfig]) -> dict:
    controllers: dict[str, int] = {}
    for cell in cells:
        controllers[cell.controller] = controllers.get(cell.controller, 0) + 1
    sim_seeds = sorted({cell.simulation_seed for cell in cells})
    return {"cells": len(cells),
            "controllers": controllers,
            "sim_seed_min": min(sim_seeds) if sim_seeds else None,
            "sim_seed_max": max(sim_seeds) if sim_seeds else None,
            "n_sim_seeds": len(sim_seeds)}


def write_outputs(name: str, out_dir: Path, rows: list[dict],
                  events: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "runs.csv", index=False)
    pd.DataFrame(events).to_csv(out_dir / "events.csv", index=False)
    print(f"[{name}] wrote {len(rows)} runs, {len(events)} events -> {out_dir}")


def execute(name: str, out_dir: Path, cells: list[RunConfig],
            workers: int = 1) -> None:
    print(f"[{name}] running {len(cells)} cells ({workers} workers) ...")
    results = run_batch(cells, workers=workers)
    rows, all_events = [], []
    for (row, cell_events, _), cell in zip(results, cells):
        rows.append(row)
        for event in cell_events:
            all_events.append({"controller": cell.controller,
                               "training_seed": cell.training_seed,
                               "simulation_seed": cell.simulation_seed,
                               **event})
    write_outputs(name, out_dir, rows, all_events)


def no_fault() -> Fault:
    # Onset at +inf: every entry is a pre-event (false) latch by construction.
    return Fault(kind="none", onset_s=float("inf"), end_s=None)


def bias_fault(sigma: float, onset: float = 2.0,
               end: float | None = 4.0) -> Fault:
    return Fault(kind="bias", onset_s=onset, end_s=end,
                 magnitude_rad_s=sigma_to_rad_s(sigma),
                 magnitude_sigma=float(sigma))
