"""Freeze every V4 confirmatory seed in a hashed manifest (run once, first).

Writes results/v4/prereg/seed_manifest.json plus a detached .sha256 file.
Fails loudly if any new block overlaps a previously used, forbidden, or
avoided seed. No confirmatory run may start before this manifest exists;
protocols validate their seeds against these blocks.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from v4_protocol_common import (  # noqa: E402
    DET_SEEDS,
    FAULTFREE_SEEDS,
    RECOVERY_SEEDS,
    ROBUSTNESS_SEEDS,
    SEVERITY_SEEDS,
    SMOKE_DATASET_SEED,
    SMOKE_SEEDS,
    SWEEP_SEEDS,
    TRAINING_SEEDS,
)

OUT_DIR = PROJECT / "results" / "v4" / "prereg"
MANIFEST = OUT_DIR / "seed_manifest.json"

DATASET_SEED = 62026
BOOTSTRAP_SEEDS = {
    "confirm_base": 20260930,  # + hypothesis index (v4_confirm_analysis)
    "v4_calibration": 20260920,  # + training seed (v4_calibrate)
    "baseline_calibration": 20260921,  # + statistic index
}

USED_SIM_BLOCKS = {
    "final_holdout_12026": list(range(12026, 12031)),
    "development_19026": list(range(19026, 19031)),
    "v3_final_29026": list(range(29026, 29031)),
    "part_a_49026": list(range(49026, 49031)),
    "part_b_59026": list(range(59026, 59031)),
}
FORBIDDEN = list(range(39026, 39031))
AVOIDED_EXTERNAL_SCAN = list(range(91000, 91060))
USED_TRAINING = [2026, 2027, 2028]  # reused as V4 retraining seeds (new data)


def fail(message: str) -> None:
    raise SystemExit(f"V4_FREEZE_SEEDS FAILED: {message}")


def main() -> None:
    new_sim = set(FAULTFREE_SEEDS) | set(SEVERITY_SEEDS)
    used_sim = {seed for block in USED_SIM_BLOCKS.values() for seed in block}
    overlap = new_sim & (used_sim | set(FORBIDDEN) | set(AVOIDED_EXTERNAL_SCAN))
    if overlap:
        fail(f"new sim seeds overlap used/forbidden blocks: {sorted(overlap)[:5]}")
    if set(SMOKE_SEEDS) & (new_sim | used_sim | set(AVOIDED_EXTERNAL_SCAN)):
        fail("smoke sim seeds must sit outside every other block")
    if DATASET_SEED in new_sim | used_sim | set(AVOIDED_EXTERNAL_SCAN):
        fail("dataset seed collides with a sim block")
    if SMOKE_DATASET_SEED in new_sim | used_sim | set(AVOIDED_EXTERNAL_SCAN) | {DATASET_SEED}:
        fail("smoke dataset seed collides")

    manifest = {
        "study": "v4_confirmatory",
        "status": "frozen_before_any_v4_confirmatory_run",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seeds": TRAINING_SEEDS,
        "new_training_seeds": [2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036],
        "dataset_seed": DATASET_SEED,
        "faultfree_sim_seeds": {"first": FAULTFREE_SEEDS[0],
                                "last": FAULTFREE_SEEDS[-1],
                                "count": len(FAULTFREE_SEEDS),
                                "seeds": FAULTFREE_SEEDS},
        "severity_sim_seeds": {"first": SEVERITY_SEEDS[0],
                               "last": SEVERITY_SEEDS[-1],
                               "count": len(SEVERITY_SEEDS),
                               "seeds": SEVERITY_SEEDS},
        "severity_prefixes": {
            "sweep": SWEEP_SEEDS,
            "recovery": RECOVERY_SEEDS,
            "robustness": ROBUSTNESS_SEEDS,
            "det_subset": DET_SEEDS,
            "timing_subset": SEVERITY_SEEDS[:5],
        },
        "bootstrap_seeds": BOOTSTRAP_SEEDS,
        "smoke_seeds": {"dataset": SMOKE_DATASET_SEED, "sim": SMOKE_SEEDS},
        "excluded_sim_blocks": {name: {"first": block[0], "last": block[-1]}
                                for name, block in USED_SIM_BLOCKS.items()},
        "forbidden_sim_block": {"first": FORBIDDEN[0], "last": FORBIDDEN[-1]},
        "avoided_external_scan": {"first": AVOIDED_EXTERNAL_SCAN[0],
                                  "last": AVOIDED_EXTERNAL_SCAN[-1]},
        "reused_training_seeds_note": (
            "2026-2028 are reused as V4 retraining seeds on the NEW V4 "
            "dataset (new weights); no previously used SIM seed is reused."),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True)
    MANIFEST.write_text(text)
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    (OUT_DIR / "seed_manifest.sha256").write_text(f"{digest}  seed_manifest.json\n")
    print(f"froze {len(TRAINING_SEEDS)} training + {len(new_sim)} sim seeds")
    print(f"manifest sha256: {digest}")


if __name__ == "__main__":
    main()
