"""Protocol (c): recovery tests (CONFIRMATORY).

Finite faults on a 10 s horizon (nominal reference; severity sim seeds):
- finite bias 8 sigma over 2-4 s.
- finite dropout 1.0 s from 2.0 s.
- forced false latch: 0.2 s dropout from 2.0 s (latches the detector,
  then releases; measures time to release on a healthy sensor).

Endpoints: recovery probability, time to release (first recovery after
fault end; right-censored at the horizon), post-release tracking RMSE.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_closed_loop import Fault, RunConfig  # noqa: E402
from v4_protocol_common import (  # noqa: E402
    FULL_DETECTOR_SET,
    PROJECT,
    SMOKE_SEEDS,
    SEVERITY_SEEDS,
    bias_fault,
    describe,
    execute,
    training_seeds_for,
)

OUT_DIR = PROJECT / "results" / "v4" / "recovery"
DURATION = 10.0
CONTROLLERS = FULL_DETECTOR_SET


def recovery_faults() -> list[Fault]:
    return [
        bias_fault(8.0, onset=2.0, end=4.0),
        Fault(kind="dropout", onset_s=2.0, end_s=None,
              dropout_duration_s=1.0),
        Fault(kind="dropout", onset_s=2.0, end_s=None,
              dropout_duration_s=0.2),
    ]


def plan(sim_seeds=None, controllers=None) -> list:
    sim_seeds = SEVERITY_SEEDS if sim_seeds is None else sim_seeds
    controllers = CONTROLLERS if controllers is None else controllers
    cells = []
    for controller in controllers:
        for training_seed in training_seeds_for(controller):
            for sim_seed in sim_seeds:
                for fault in recovery_faults():
                    cells.append(RunConfig(
                        controller=controller, training_seed=training_seed,
                        simulation_seed=sim_seed, reference="nominal",
                        duration=DURATION, fault=fault))
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
    print(f"[recovery] {describe(cells)}")
    if args.dry_run:
        return
    execute("recovery", OUT_DIR, cells, workers=args.workers)


if __name__ == "__main__":
    main()
