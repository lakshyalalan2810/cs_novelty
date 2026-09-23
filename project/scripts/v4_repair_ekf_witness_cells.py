"""Repair only V4_full_ekf core cells corrupted by missing EKF initialization."""

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_closed_loop import run_batch  # noqa: E402
from v4_protocol_common import (  # noqa: E402
    PROJECT,
    _provenance_fingerprint,
    scientific_key,
    write_outputs,
)
from v4_protocol_core import _protocol, validated_frozen_plan  # noqa: E402

RESULTS = PROJECT / "results" / "v4"
INCIDENT = RESULTS / "ekf_witness_integrity_repair.json"
STATUS = RESULTS / "core_execution_status.json"
OLD_PROVENANCE = "7979c98f08431d51c9828bc282f706893abee75de16dc9483d948c16e6a8eed1"
EXPECTED = {"faultfree": 2400, "sweep": 300}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_records(protocol: str) -> dict[str, tuple[str, str, str]]:
    connection = sqlite3.connect(RESULTS / protocol / "checkpoint.sqlite3")
    try:
        return {key: (provenance, row, events) for key, provenance, row, events
                in connection.execute(
                    "SELECT key, provenance, row_json, events_json FROM results")}
    finally:
        connection.close()


def materialize(protocol: str, cells: list) -> None:
    records = checkpoint_records(protocol)
    rows, events = [], []
    for cell in cells:
        key = scientific_key(cell)
        if key not in records:
            raise RuntimeError(f"{protocol}: missing checkpoint key {key}")
        _, row_json, events_json = records[key]
        rows.append(json.loads(row_json))
        for event in json.loads(events_json):
            events.append({"controller": cell.controller,
                           "training_seed": cell.training_seed,
                           "simulation_seed": cell.simulation_seed,
                           **event})
    write_outputs(protocol, RESULTS / protocol, rows, events)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    frozen, cells, plan_hash = validated_frozen_plan()
    targets = [cell for cell in cells if cell.controller == "V4_full_ekf"]
    counts = {name: sum(_protocol(cell) == name for cell in targets)
              for name in EXPECTED}
    if counts != EXPECTED or len(targets) != 2700:
        raise RuntimeError(f"unexpected repair target counts: {counts}")

    new_provenance = _provenance_fingerprint()
    before = {}
    for protocol in EXPECTED:
        directory = RESULTS / protocol
        before.update({
            f"results/v4/{protocol}/{name}": sha256(directory / name)
            for name in ("runs.csv", "events.csv", "checkpoint.sqlite3")
        })
    if not INCIDENT.exists():
        write_json(INCIDENT, {
            "status": "repair_running",
            "reason": ("V4_full_ekf called step() before initialize(); every "
                       "affected 6 s cell recorded 581 observer failures"),
            "scientific_scope": "all and only 2700 planned V4_full_ekf cells",
            "plan_sha256": plan_hash,
            "old_checkpoint_provenance": OLD_PROVENANCE,
            "new_checkpoint_provenance": new_provenance,
            "pre_repair_source_sha256":
                "4b5300863afeecceef378307ab30ab43cf223fb97ad48192a538c13b94713de2",
            "repaired_source_sha256": sha256(
                PROJECT / "scripts" / "v4_closed_loop.py"),
            "target_counts": counts,
            "pre_repair_artifact_hashes": before,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        })

    batch_size = 50
    for protocol, expected in EXPECTED.items():
        protocol_targets = [cell for cell in targets
                            if _protocol(cell) == protocol]
        records = checkpoint_records(protocol)
        pending = []
        for cell in protocol_targets:
            key = scientific_key(cell)
            provenance, row_json, _ = records[key]
            failures = int(json.loads(row_json)["observer_failures"])
            if provenance == new_provenance and failures == 0:
                continue
            if provenance != OLD_PROVENANCE or failures != 581:
                raise RuntimeError(
                    f"{protocol}: unexpected pre-repair state for {key}: "
                    f"provenance={provenance}, observer_failures={failures}")
            pending.append((key, cell))
        print(f"[{protocol}] {expected - len(pending)}/{expected} repaired; "
              f"running {len(pending)} cells")
        checkpoint = RESULTS / protocol / "checkpoint.sqlite3"
        connection = sqlite3.connect(checkpoint)
        try:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset:offset + batch_size]
                output = run_batch([cell for _, cell in batch],
                                   workers=args.workers)
                with connection:
                    for (key, _), (row, events, _) in zip(batch, output):
                        if row["observer_failures"] != 0:
                            raise RuntimeError(
                                f"{protocol}: repaired cell still failed: {key}")
                        connection.execute(
                            "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
                            (key, new_provenance,
                             json.dumps(row, allow_nan=True),
                             json.dumps(events, allow_nan=True)))
                complete = min(offset + len(batch), len(pending))
                print(f"[{protocol}] repaired {complete}/{len(pending)} pending")
        finally:
            connection.close()
        materialize(protocol, [cell for cell in cells
                               if _protocol(cell) == protocol])

    incident = json.loads(INCIDENT.read_text())
    incident["status"] = "repair_complete"
    incident["completed_utc"] = datetime.now(timezone.utc).isoformat()
    incident["post_repair_artifact_hashes"] = {
        f"results/v4/{protocol}/{name}": sha256(
            RESULTS / protocol / name)
        for protocol in EXPECTED
        for name in ("runs.csv", "events.csv", "checkpoint.sqlite3")
    }
    write_json(INCIDENT, incident)
    status = json.loads(STATUS.read_text())
    status["integrity_repair"] = {
        "status": "complete", "controller": "V4_full_ekf",
        "cells": 2700, "incident": "results/v4/ekf_witness_integrity_repair.json"}
    write_json(STATUS, status)
    print(f"wrote {INCIDENT}")


if __name__ == "__main__":
    main()
