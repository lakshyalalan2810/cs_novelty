"""Shared grids, seed blocks, and I/O for the V4 evaluation protocols.

Seed blocks (frozen in V4_PREREGISTRATION.md before any run):
- fault-free confirmatory sim seeds: 71000-71999.
- severity/robustness sim seeds: 72000-72099.
- training seeds: 2026-2036 (2026-2028 frozen, 2029-2036 new).
"""

import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

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


def scientific_key(cell: RunConfig) -> str:
    """Stable full-config key used for deduplication and crash-safe resume."""
    payload = json.dumps(asdict(cell), sort_keys=True, separators=(",", ":"),
                         allow_nan=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _provenance_fingerprint() -> str:
    """Conservative hash of code and frozen inputs that can affect a cell."""
    paths = [PROJECT / "requirements.txt",
             PROJECT / "scripts" / "v4_closed_loop.py",
             PROJECT / "scripts" / "v4_detector_configs.py"]
    for directory in (PROJECT / "src", PROJECT / "results" / "v4" / "models",
                      PROJECT / "results" / "v4" / "calibration",
                      PROJECT / "results" / "training_seed_robustness" / "configs",
                      PROJECT / "models" / "training_seed_robustness"):
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*")
                         if path.is_file() and path.suffix != ".pyc")
    paths.extend(path for path in (PROJECT / "results" / "configs").glob("*.json"))
    paths.extend([PROJECT / "results" / "lstm_model_weights.pt",
                  PROJECT / "results" / "auxiliary_model_weights.pt",
                  PROJECT / "results" / "training_seed_robustness"
                  / "model_pair_manifest.json"])
    digest = hashlib.sha256()
    digest.update(sys.version.encode())
    for module in ("numpy", "pandas", "torch", "scipy"):
        package = sys.modules.get(module)
        digest.update(f"{module}={getattr(package, '__version__', '')}".encode())
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_outputs(name: str, out_dir: Path, rows: list[dict],
                  events: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(rows), out_dir / "runs.csv")
    _atomic_csv(pd.DataFrame(events), out_dir / "events.csv")
    print(f"[{name}] wrote {len(rows)} runs, {len(events)} events -> {out_dir}")


def execute(name: str, out_dir: Path, cells: list[RunConfig],
            workers: int = 1, checkpoint_every: int = 50,
            progress_interval_s: float = 30.0) -> None:
    """Execute with an atomic SQLite checkpoint and deterministic CSV order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keyed = [(scientific_key(cell), cell) for cell in cells]
    if len({key for key, _ in keyed}) != len(keyed):
        raise ValueError(f"{name}: duplicate RunConfig scientific keys")
    provenance = _provenance_fingerprint()
    checkpoint = out_dir / "checkpoint.sqlite3"
    connection = sqlite3.connect(checkpoint)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS results ("
        "key TEXT PRIMARY KEY, provenance TEXT NOT NULL, "
        "row_json TEXT NOT NULL, events_json TEXT NOT NULL)")
    valid = {row[0] for row in connection.execute(
        "SELECT key FROM results WHERE provenance = ?", (provenance,))}
    pending = [(key, cell) for key, cell in keyed if key not in valid]
    total = len(cells)
    completed = total - len(pending)
    print(f"[{name}] {completed}/{total} already complete; "
          f"running {len(pending)} cells ({workers} workers); "
          f"checkpoint={checkpoint}")
    started = last_report = perf_counter()
    recent_started, recent_completed = started, completed
    batch_size = max(1, checkpoint_every)
    try:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            results = run_batch([cell for _, cell in batch], workers=workers)
            if len(results) != len(batch):
                raise RuntimeError(f"{name}: batch returned {len(results)} "
                                   f"results for {len(batch)} cells")
            with connection:
                for (key, _), (row, events, _) in zip(batch, results):
                    connection.execute(
                        "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
                        (key, provenance,
                         json.dumps(row, allow_nan=True,
                                    default=_json_scalar),
                         json.dumps(events, allow_nan=True,
                                    default=_json_scalar)))
            completed += len(batch)
            now = perf_counter()
            if now - last_report >= progress_interval_s or completed == total:
                minutes = max((now - recent_started) / 60.0, 1e-12)
                rate = (completed - recent_completed) / minutes
                print(f"[{name}] {completed}/{total} "
                      f"({100 * completed / total:.1f}%) "
                      f"elapsed={(now - started) / 60:.1f} min "
                      f"recent={rate:.1f} runs/min checkpoint={checkpoint}")
                last_report = recent_started = now
                recent_completed = completed
    except BaseException:
        connection.close()
        raise
    records = {key: (row, events) for key, row, events in connection.execute(
        "SELECT key, row_json, events_json FROM results WHERE provenance = ?",
        (provenance,))}
    connection.close()
    rows, all_events = [], []
    for key, cell in keyed:
        row_json, events_json = records[key]
        rows.append(json.loads(row_json))
        for event in json.loads(events_json):
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
