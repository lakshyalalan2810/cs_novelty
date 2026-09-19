# Reliability-Aware LSTM-MPC for Sensor-Fault-Tolerant DC Motor Control

This repository evaluates constrained LSTM model-predictive control (MPC) with virtual feedback when a DC-motor speed sensor is unreliable. The defensible contribution is improved robustness to the tested sensor faults in simulation. The evidence does not establish hardware readiness or universal superiority of adaptive MPC.

## Final method

```text
reference -> constrained LSTM-MPC -> voltage -> nonlinear DC motor -> sensor
    ^                                                        |
    +-- measured speed or LSTM virtual feedback <- reliability monitor
```

- Plant/model sample time: `0.01 s`.
- Control update interval: `0.05 s` (20 Hz), so each applied command is held for five plant/model samples.
- Frozen final MPC: `H=20`, `Nc=2`, `blocks=(5, 15)`, `maxiter=8`, warm start enabled.
- The warm-start plan shifts by five model samples at each control update.
- Each objective uses future references `k+1 ... k+H`.
- Rolling prediction quality compares only the five forecast ages generated under the command actually held for that 50 ms interval.
- Voltage is constrained to `0–12 V`; the low-mismatch maximum move is `2 V` per control update.

The configuration is stored in `results/configs/final_mpc_config.json`. Notebook 06 loads it as a fixed final choice; wall-clock timing is reported but is not used to select the final controller. A development comparison found H=20/Nc=1 and H=20/Nc=2 effectively tied in RMSE (`9.564905` versus `9.566216` on the screening case), so the documented H=20/Nc=2 design was retained before final holdout execution.

## Canonical final evidence

Notebook 06 was freshly executed on 2026-09-07 using previously untouched final seeds `12026–12030`: four controllers × eleven scenarios × five paired seeds = 220 runs. Reliability thresholds and the final controller structure were frozen using earlier development/validation evidence; the final seeds were not used to redesign them.

The authoritative artifacts are:

- `results/metrics/final_controller_runs.csv` — 220 per-run records.
- `results/metrics/final_controller_comparison.csv` — 44 controller/scenario aggregates.
- `results/metrics/final_runtime_samples.csv` — all 19,965 final MPC timing samples.
- `results/metrics/virtual_feedback_quality.csv` — offline virtual-versus-true and sensor-versus-true errors only during substitution.
- `results/metrics/bugfix_scenario_comparison.csv` — preserved pre-fix versus corrected drift/combined evidence.
- `results/metrics/pi_fallback_validation.json` — targeted ten-update fallback safety check.
- `results/raw/final_representative_traces.npz` — representative trajectories.
- `results/metrics/final_summary.json` — derived headline, safety, and runtime values.
- `results/configs/final_mpc_config.json` — the frozen configuration and seed role.

Canonical machine-readable values, shown to six decimals for cross-checking:

| Quantity | Value |
|---|---:|
| Plain MPC sensor-fault interval RMSE | `7.584044` rad/s |
| Reliability-aware MPC sensor-fault interval RMSE | `1.374104` rad/s |
| Sensor-fault RMSE improvement | `81.881647%` |
| Plain MPC disturbance/combined interval RMSE | `1.932806` rad/s |
| Adaptive MPC disturbance/combined interval RMSE | `4.709976` rad/s |
| Signed adaptive degradation metric `(adaptive-plain)/plain` | `143.685962%` (worsening) |
| Runtime mean | `67.963908 ms` |
| Runtime median | `60.554900 ms` |
| Runtime p95 | `135.782580 ms` |
| Runtime p99 | `166.619316 ms` |
| Runtime maximum | `666.868900 ms` |

The sensor comparison is the macro mean across noise, 5% bias, 15% bias, dropout, and drift fault intervals. The adaptive comparison is the macro mean across load disturbance, parameter variation, and combined sensor-fault/load intervals. Fault masks with a finite end are end-exclusive.

All 220 runs recorded zero optimizer failures, zero voltage violations, zero slew violations, and zero non-finite events. Runtime is measured around MPC solver/control computation only; it is machine/load dependent and is not an end-to-end deployment latency. `58.442274%` of solves exceeded 50 ms and `23.060356%` exceeded 100 ms, so this implementation is **not reliably real-time at 20 Hz**. The histogram and ECDF are in `results/plots/final_runtime_distribution.png`.

The combined sensor-fault/load case is the main limitation. The corrected latch prevents the pre-fix false recovery, but continued virtual feedback cannot follow the concurrent plant disturbance: during substitution its mean RMSE is about `11.71 rad/s` versus `3.13 rad/s` for the corrupted sensor, and adaptive fault-window RMSE changes from `3.717753` to `11.715164` rad/s. The three-scenario adaptive aggregate therefore worsens by `143.685962%`. Correct recovery semantics are retained despite the performance loss.

For the finite bias/dropout cases, accumulated CUSUM evidence did not fully discharge within the two-second post-fault observation window; recovery latency is therefore right-censored (`NaN`) in the quality CSV, not silently reported as zero. The forced ten-update PI fallback check remained bounded (`38.976 rad/s` maximum), respected voltage/slew limits, and resumed MPC with a `0.094 V` jump below the `2 V` limit.

Reliability thresholds were calibrated on the clean validation operating distribution and may not transfer to substantially different operating points. The five final closed-loop seeds test robustness to simulation noise/randomness; training-seed robustness was not tested because the single frozen LSTM was not retrained.

## LSTM evidence

- Held-out one-step RMSE: `0.140873 rad/s`; MAE: `0.113106 rad/s`; R²: `0.999906`.
- Persistence RMSE: `0.289778 rad/s`.
- Recursive endpoint RMSE at H=5/10/15: `0.210952/0.336461/0.487796 rad/s`.

Complete trajectories are split before overlapping windows are made, and normalization is fitted from training trajectories only. The LSTM is open-loop trained, recursive error grows with horizon, and all controller evidence is simulation-only.

## Reproduce

Python 3.11 or newer is recommended. Dependencies are exactly pinned in `requirements.txt`.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
jupyter lab
```

Run notebooks in order:

1. `01_dc_motor_simulation.ipynb`
2. `02_dataset_generation.ipynb`
3. `03_lstm_training.ipynb`
4. `04_reliability_analysis.ipynb`
5. `04b_reliability_redesign.ipynb`
6. `04c_reliability_finalization.ipynb`
7. `05_mpc_experiments.ipynb`
8. `06_final_evaluation.ipynb`

Notebook 06 is deterministic in configuration and scenarios, but timing values vary by machine and load. Its final holdout seeds must not be used for future tuning.

For a read-only check of the saved evidence:

```bash
python scripts/verify_results.py
```

The verifier reloads the dataset/model, recomputes LSTM metrics, reconstructs headline comparisons and runtime percentiles from raw CSVs, checks safety totals and representative traces, and fails if the configuration, summary, tables, or canonical documentation disagree.

## Repository map

```text
data/processed/                 processed trajectory dataset
notebooks/                      ordered research pipeline
results/configs/                frozen model/reliability/MPC settings
results/metrics/                per-run, aggregate, and summary evidence
results/raw/                    saved diagnostic arrays and final traces
results/plots/                  publication and diagnostic figures
scripts/verify_results.py       independent saved-evidence verifier
scripts/validate_recovery.py    deterministic false-recovery regression
src/                            plant, data, LSTM, reliability, and MPC code
```

See `PROJECT_PIPELINE_AND_STATUS.md` for the phase history, `FINAL_REPO_AUDIT.md` for the pre-fix independent audit, and `FINAL_REPO_AUDIT_POSTFIX.md` for the post-fix verdict.
