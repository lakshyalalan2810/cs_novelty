"""Protocol (e): timing benchmark (DESCRIPTIVE, machine-dependent).

Single-threaded closed loops (nominal tracking, 5 severity seeds, all
training seeds where applicable) at control strides 5 (20 Hz) and 10
(10 Hz, lower-rate option). Warm-up (first 20 control steps) is excluded
from the reported distributions; the engine records per-solve samples.

Endpoints: solve-time distribution (mean/p50/p95/p99/max), control-period
exceedance rate, detector/observer overhead. Timing is hardware- and
load-dependent; the confirmatory report must record the machine.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_closed_loop import RunConfig  # noqa: E402
from v4_protocol_common import (  # noqa: E402
    ANCHOR,
    FULL_DETECTOR_SET,
    PROJECT,
    SEVERITY_SEEDS,
    SMOKE_SEEDS,
    TRAINING_SEEDS,
    describe,
    execute,
    no_fault,
    training_seeds_for,
)

OUT_DIR = PROJECT / "results" / "v4" / "timing"
TIMING_SEEDS = SEVERITY_SEEDS[:5]
STRIDES = (5, 10)
CONTROLLERS = FULL_DETECTOR_SET + ANCHOR


def plan(sim_seeds=None, controllers=None) -> list:
    sim_seeds = TIMING_SEEDS if sim_seeds is None else sim_seeds
    controllers = CONTROLLERS if controllers is None else controllers
    cells = []
    for controller in controllers:
        for training_seed in training_seeds_for(controller):
            for sim_seed in sim_seeds:
                for stride in STRIDES:
                    cells.append(RunConfig(
                        controller=controller, training_seed=training_seed,
                        simulation_seed=sim_seed, reference="nominal",
                        fault=no_fault(), control_stride=stride))
    return cells


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.smoke:
        cells = plan(sim_seeds=SMOKE_SEEDS, controllers=["PI"])
    else:
        cells = plan()
    print(f"[timing] {describe(cells)}")
    if args.dry_run:
        return
    if args.workers != 1:
        print("[timing] WARNING: timing must be single-worker; forcing 1")
    execute("timing", OUT_DIR, cells, workers=1)


if __name__ == "__main__":
    main()
