"""Protocol (a): fault-free Monte Carlo (CONFIRMATORY).

Grid: 200 sim seeds x 11 training seeds x 4 references x detector
controllers (+B anchor). No faults; every reliability entry is a false
latch by construction (fault onset at +inf).

Endpoints: false-latch probability (Wilson), time to false latch,
substitution fraction, tracking penalty vs B.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_closed_loop import RunConfig  # noqa: E402
from v4_protocol_common import (  # noqa: E402
    ANCHOR,
    FAULTFREE_SEEDS,
    FULL_DETECTOR_SET,
    PROJECT,
    SMOKE_SEEDS,
    REFERENCES,
    TRAINING_SEEDS,
    describe,
    execute,
    no_fault,
    training_seeds_for,
)

OUT_DIR = PROJECT / "results" / "v4" / "faultfree"
CONTROLLERS = FULL_DETECTOR_SET + ANCHOR


def plan(sim_seeds=None, references=None, controllers=None) -> list:
    sim_seeds = FAULTFREE_SEEDS if sim_seeds is None else sim_seeds
    references = REFERENCES if references is None else references
    controllers = CONTROLLERS if controllers is None else controllers
    cells = []
    for controller in controllers:
        for training_seed in training_seeds_for(controller):
            for sim_seed in sim_seeds:
                for reference in references:
                    cells.append(RunConfig(
                        controller=controller,
                        training_seed=training_seed,
                        simulation_seed=sim_seed, reference=reference,
                        fault=no_fault()))
    return cells


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.smoke:
        cells = plan(sim_seeds=SMOKE_SEEDS, references=["nominal"],
                     controllers=["PI"])
    else:
        cells = plan()
    summary = describe(cells)
    print(f"[faultfree] {summary}")
    if args.dry_run:
        return
    execute("faultfree", OUT_DIR, cells, workers=args.workers)


if __name__ == "__main__":
    main()
