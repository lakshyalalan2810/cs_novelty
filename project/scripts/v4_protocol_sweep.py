"""Protocol (b): small-magnitude fault sweep in sigma units (CONFIRMATORY).

Families (nominal reference; severity sim seeds 72000-72049):
- bias: 0.5/1/2/4/8/16 sigma, onset 2.0 s, end 4.0 s.
- dropout: 0.1/0.25/0.5/1.0/2.0 s from 2.0 s.
- drift: 0.125/0.25/0.5/1.0/2.0 rad/s^2 ramps from 2.0 s over 4 s.
- load (secondary): post-step 0.06/0.10/0.15/0.20 N m at 3.0 s.
- combined (secondary): 8 sigma + 0.10/0.15 N m at 3.0 s, persistent.

DET/ROC support: threshold multipliers 0.5/0.75/1.0/1.5/2.0 on a 20-seed
subset for bias 2/8 sigma plus matched fault-free cells.

Endpoints: detection probability, detection delay, minimum detectable
magnitude (interpolated), DET points by threshold sweep.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_closed_loop import Fault, RunConfig, sigma_to_rad_s  # noqa: E402
from v4_protocol_common import (  # noqa: E402
    DET_SEEDS,
    FULL_DETECTOR_SET,
    PROJECT,
    SMOKE_SEEDS,
    SEVERITY_SEEDS,
    SWEEP_SEEDS,
    TRAINING_SEEDS,
    bias_fault,
    describe,
    execute,
    no_fault,
    training_seeds_for,
)

OUT_DIR = PROJECT / "results" / "v4" / "sweep"

BIAS_SIGMAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DROPOUT_LENGTHS = (0.1, 0.25, 0.5, 1.0, 2.0)
DRIFT_RATES = (0.125, 0.25, 0.5, 1.0, 2.0)
LOAD_LEVELS = (0.06, 0.10, 0.15, 0.20)
COMBINED_CELLS = ((8.0, 0.10), (8.0, 0.15))  # (bias sigma, load Nm)
THRESHOLD_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
DET_BIASES = (2.0, 8.0)
CONTROLLERS = FULL_DETECTOR_SET


def sweep_faults() -> list[Fault]:
    faults = [bias_fault(sigma) for sigma in BIAS_SIGMAS]
    faults += [Fault(kind="dropout", onset_s=2.0, end_s=None,
                     dropout_duration_s=length) for length in DROPOUT_LENGTHS]
    faults += [Fault(kind="drift", onset_s=2.0, end_s=6.0,
                     drift_rate_rad_s2=rate,
                     magnitude_rad_s=rate * 4.0)
               for rate in DRIFT_RATES]
    # Secondary families: load false-entry retest + combined cells.
    faults += [Fault(kind="load", onset_s=3.0, end_s=None,
                     load_step_Nm=level) for level in LOAD_LEVELS]
    faults += [Fault(kind="combined", onset_s=3.0, end_s=None,
                     magnitude_rad_s=sigma_to_rad_s(sigma),
                     magnitude_sigma=float(sigma), load_step_Nm=load)
               for sigma, load in COMBINED_CELLS]
    return faults


def plan(sim_seeds=None, controllers=None, with_det: bool = True) -> list:
    sim_seeds = SWEEP_SEEDS if sim_seeds is None else sim_seeds
    controllers = CONTROLLERS if controllers is None else controllers
    cells = []
    for controller in controllers:
        for training_seed in training_seeds_for(controller):
            for sim_seed in sim_seeds:
                for fault in sweep_faults():
                    cells.append(RunConfig(
                        controller=controller, training_seed=training_seed,
                        simulation_seed=sim_seed, reference="nominal",
                        fault=fault))
    if with_det:
        for controller in controllers:
            for training_seed in training_seeds_for(controller):
                for sim_seed in DET_SEEDS:
                    if sim_seed not in sim_seeds:
                        continue
                    for scale in THRESHOLD_SCALES:
                        if scale == 1.0:
                            continue  # covered by the main grid
                        for sigma in DET_BIASES:
                            cells.append(RunConfig(
                                controller=controller,
                                training_seed=training_seed,
                                simulation_seed=sim_seed, reference="nominal",
                                fault=bias_fault(sigma),
                                threshold_scale=scale))
                        cells.append(RunConfig(
                            controller=controller, training_seed=training_seed,
                            simulation_seed=sim_seed, reference="nominal",
                            fault=no_fault(), threshold_scale=scale))
    return cells


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.smoke:
        cells = plan(sim_seeds=SMOKE_SEEDS, controllers=["PI"],
                     with_det=False)
    else:
        cells = plan()
    print(f"[sweep] {describe(cells)}")
    if args.dry_run:
        return
    execute("sweep", OUT_DIR, cells, workers=args.workers)


if __name__ == "__main__":
    main()
