"""Factorial ablation configs for the V4 detector variants.

Defines flag-only variants over frozen per-pair calibrations (base
residual gate / CUSUM parameters always come from the pair's own frozen
calibration at build time). Run to write
results/v4/configs/detector_ablation.json:

    python scripts/v4_detector_configs.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from reliability_v4 import ReliabilityMonitorV4  # noqa: E402

OUT_PATH = PROJECT / "results" / "v4" / "configs" / "detector_ablation.json"

# Anti-windup clamp multiple: round-number choice (not tuned). Bounds
# post-fault discharge to ~k*threshold/allowance samples plus exit_count.
CLAMP_K = 2.0

# Warmup blanking in monitor samples at the 100 Hz plant sample rate:
# 150 samples = 1.5 s, covering the maximum observed startup latch
# (1.12 s) with margin; matches the post-hoc STARTUP/MID_RUN cut.
# Round-number choice (not tuned); the Phase 6 protocol sweeps this.
WARMUP_BLANK_SAMPLES = 150

# Sentinel: resolve the witness gate from the pair's own residual gate at
# build time (same physical units, rad/s). A calibrated witness gate from
# the Phase 4 calibration protocol replaces this placeholder.
WITNESS_GATE_SENTINEL = "RESIDUAL_GATE"

VARIANTS: dict[str, dict] = {
    # Required factorial cells.
    "clamp_only": {"cusum_clamp_multiple": CLAMP_K},
    "gating_only": {"warmup_blank_samples": WARMUP_BLANK_SAMPLES,
                    "warmup_trust_sensor": True},
    "clamp_gating": {"cusum_clamp_multiple": CLAMP_K,
                     "warmup_blank_samples": WARMUP_BLANK_SAMPLES,
                     "warmup_trust_sensor": True},
    "witness_only": {"witness_residual_gate": WITNESS_GATE_SENTINEL},
    # Reference cells.
    "frozen_baseline": {},
    "full": {"cusum_clamp_multiple": CLAMP_K,
             "warmup_blank_samples": WARMUP_BLANK_SAMPLES,
             "warmup_trust_sensor": True,
             "witness_residual_gate": WITNESS_GATE_SENTINEL},
}


def resolve_variant(base: ReliabilityMonitorV4, name: str) -> ReliabilityMonitorV4:
    """Build a variant monitor from a frozen-parameter base monitor."""
    if name not in VARIANTS:
        raise ValueError(f"unknown detector variant: {name}")
    overrides = dict(VARIANTS[name])
    if overrides.get("witness_residual_gate") == WITNESS_GATE_SENTINEL:
        overrides["witness_residual_gate"] = base.residual_gate
    return replace(base, **overrides)


def main() -> None:
    doc = {
        "study": "v4_detector_ablation",
        "status": "factorial flags only; base parameters come from each "
                  "pair's frozen calibration at build time",
        "constants": {
            "CLAMP_K": CLAMP_K,
            "WARMUP_BLANK_SAMPLES": WARMUP_BLANK_SAMPLES,
            "WARMUP_SECONDS_AT_100HZ": WARMUP_BLANK_SAMPLES / 100.0,
            "WITNESS_GATE_SENTINEL": WITNESS_GATE_SENTINEL,
        },
        "variants": VARIANTS,
        "notes": [
            "frozen_baseline reproduces SensorReliabilityMonitor bit-identically.",
            "Witness residual is y_meas - y_witness (aux LSTM or EKF); the "
            "witness gate placeholder equals the pair residual gate until "
            "Phase 4 calibration provides a calibrated value.",
            "recovery_boost stays 0.0 in every cell; clamp alone bounds "
            "discharge. Boost is available as a flag for the Phase 6 sweep.",
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2))
    print(f"wrote {OUT_PATH} ({len(VARIANTS)} variants)")


if __name__ == "__main__":
    main()
