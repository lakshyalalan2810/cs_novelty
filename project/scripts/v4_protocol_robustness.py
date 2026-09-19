"""Protocol (d): robustness to sensing/compute/plant variation (CONFIRMATORY).

One-factor-at-a-time design on bias-8-sigma cells plus matched fault-free
cells (nominal reference; severity sim seeds):
- current noise std: 0.0/0.01/0.05/0.10 A.
- current quantization: none/0.01/0.05 A.
- sample delay: 0/1/2/5 samples.
- parameter presets: nominal/frozen_shifted/uniform +-10/20/30%.

Endpoints: detection probability, false-latch probability, fault-window
RMSE degradation vs the nominal corner.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_closed_loop import PARAM_PRESETS, RunConfig  # noqa: E402
from v4_protocol_common import (  # noqa: E402
    FULL_DETECTOR_SET,
    PROJECT,
    SMOKE_SEEDS,
    ROBUSTNESS_SEEDS,
    SEVERITY_SEEDS,
    bias_fault,
    describe,
    execute,
    no_fault,
    training_seeds_for,
)

OUT_DIR = PROJECT / "results" / "v4" / "robustness"

CURRENT_NOISE_LEVELS = (0.0, 0.01, 0.05, 0.10)
QUANTIZATION_LEVELS = (None, 0.01, 0.05)
SAMPLE_DELAYS = (0, 1, 2, 5)
PARAM_LEVELS = ("nominal", "frozen_shifted", "uniform_p10", "uniform_m10",
                "uniform_p20", "uniform_m20", "uniform_p30", "uniform_m30")
CONTROLLERS = FULL_DETECTOR_SET


def corners() -> list[dict]:
    """OFAT corners: nominal plus one varied factor at a time."""
    corners = [{"current_noise_std": 0.0, "current_quantization_a": None,
                "sample_delay": 0, "param_preset": "nominal"}]
    for level in CURRENT_NOISE_LEVELS[1:]:
        corners.append({"current_noise_std": level,
                        "current_quantization_a": None, "sample_delay": 0,
                        "param_preset": "nominal"})
    for level in QUANTIZATION_LEVELS[1:]:
        corners.append({"current_noise_std": 0.0,
                        "current_quantization_a": level, "sample_delay": 0,
                        "param_preset": "nominal"})
    for level in SAMPLE_DELAYS[1:]:
        corners.append({"current_noise_std": 0.0,
                        "current_quantization_a": None,
                        "sample_delay": level, "param_preset": "nominal"})
    for level in PARAM_LEVELS[1:]:
        assert level in PARAM_PRESETS, level
        corners.append({"current_noise_std": 0.0,
                        "current_quantization_a": None, "sample_delay": 0,
                        "param_preset": level})
    return corners


def plan(sim_seeds=None, controllers=None) -> list:
    sim_seeds = ROBUSTNESS_SEEDS if sim_seeds is None else sim_seeds
    controllers = CONTROLLERS if controllers is None else controllers
    faults = [bias_fault(8.0), no_fault()]
    cells = []
    for controller in controllers:
        for training_seed in training_seeds_for(controller):
            for sim_seed in sim_seeds:
                for corner in corners():
                    for fault in faults:
                        cells.append(RunConfig(
                            controller=controller,
                            training_seed=training_seed,
                            simulation_seed=sim_seed, reference="nominal",
                            fault=fault, **corner))
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
    print(f"[robustness] {describe(cells)}")
    if args.dry_run:
        return
    execute("robustness", OUT_DIR, cells, workers=args.workers)


if __name__ == "__main__":
    main()
