# Disturbance-Aware Witness Gating for Reliability-Aware LSTM-MPC

This repository evaluates constrained LSTM model-predictive control (MPC) with virtual feedback when a DC-motor speed sensor is unreliable. The preserved V1 result shows improvement for tested sensor faults in simulation, while the combined-fault-plus-load result is training-initialization-dependent: its hierarchical-bootstrap interval crosses zero and the seed-2026 realization is unfavorable. Pure load disturbance can trigger false CUSUM entry. The C4 three-way scalar attributor failed its development gates; the later V4 witness gate instead received narrow confirmatory support for reduced false entries at the tested 0.15 N·m load step. The evidence does not establish hardware readiness or universal superiority.

## V4 H1–H9 confirmatory closeout

The exact 12,200-cell H1–H9 dependency core has been completed and
independently verified; it is not the full 268,910-cell umbrella, whose
remaining 257,260 cells are deferred. Only H8 and H9 survive the
pre-registered Holm correction: under clean-start pairs at the tested 0.15
N·m load step, both `V4_full_aux` and `V4_full_ekf` reduce false-entry
probability relative to C3 by 0.217742 (95% CIs [0.138686, 0.315315]). H1/H2
are directionally favorable but nonsignificant; H3/H4/H6/H7 are unfavorable,
and H5 is exactly null.

The earlier 400-cell attempt occurred before confirmatory analysis and was
fully discarded; the plan was independently reconstructed 12,200/12,200 and
the core restarted from zero. A pre-analysis integrity audit then found an
uninitialized EKF witness in 2,700 required cells. All and only those corrupted
cells were repaired and rerun before the one-time analysis; 9,500 valid cells
were not rerun. See `V4_CONFIRMATORY_RESULTS.md` for the complete table,
negative findings, deferred scope, and hashes. The historical result manifest
is preserved; the superseding corrected result-manifest SHA256 is
`0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e`.

<!-- V3_GENERATED_START -->
## Frozen V3 final evidence

The completed final matrix uses untouched seeds `29026–29030`: 5 controllers × 11 scenarios × 5 seeds = 275 paired runs. Seeds `19026–19030` remain development evidence.

C3 reference-scenario substitutions total `8` isolated samples with zero reliability latches. `1` occurred within 0.05 s after the step-reference transition; changing-reference transitions triggered `0`. Load-disturbance mean RMSE is `0.937127 rad/s` for plain MPC and `2.538714 rad/s` for C3; the detector's false sensor-fault activation under plant disturbance remains a limitation.

C2 remains behaviorally identical to C1 across all 55 paired cases. The auxiliary recovery gate was uniquely binding `0` times and delayed recovery relative to C1: `False`.

All 275 runs contain `0` optimizer failures, `0` nonfinite events, `0` voltage violations, and `0` slew violations. Canonical artifacts are under `results/metrics/v3_final_11scenario_*`; full evidence is in `V3_DUAL_VIRTUAL_SENSOR_EXTENSION.md`.
<!-- V3_GENERATED_END -->

## Preserved V1 method

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

## Preserved V1 canonical evidence

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

Reliability thresholds were calibrated on the clean validation operating distribution and may not transfer to substantially different operating points. The five final closed-loop seeds test robustness to simulation noise/randomness. (Historical V1 note, now superseded: at V1 time training-seed robustness was not tested because the single frozen LSTM was not retrained. A later study trained matched model pairs for seeds 2027/2028 under frozen procedures; see `TRAINING_SEED_ROBUSTNESS.md` and `FINAL_ROBUSTNESS_AND_SEVERITY_STUDY.md`. The combined-fault-plus-load result is training-initialization-dependent as summarized at the top of this README.)

## LSTM evidence

- Held-out one-step RMSE: `0.140873 rad/s`; MAE: `0.113106 rad/s`; R²: `0.999906`.
- Persistence RMSE: `0.289778 rad/s`.
- Recursive endpoint RMSE at H=5/10/15: `0.210952/0.336461/0.487796 rad/s`.

Complete trajectories are split before overlapping windows are made, and normalization is fitted from training trajectories only. The LSTM is open-loop trained, recursive error grows with horizon, and all controller evidence is simulation-only.

## Reproduce

Python 3.11 or newer is recommended. Dependencies are exactly pinned in `requirements.txt` (pinned `torch==2.13.0`).

Environment record: the pinned stack declares `torch==2.13.0`; the independent review ran `torch 2.14.0` with Python 3.12 and `pandas 3.0.5` (see `V4_BOOL_SERIES_REPAIR_NOTE.md` for the pandas-3 verifier fix). V4 preparation used the previously recorded environments. The completed V4 core checkpoint provenance and confirmatory analysis match Python 3.11.15 (Anaconda build), `torch 2.6.0+cu124`, `numpy 2.2.6`, `pandas 2.3.3`, and `scipy 1.13.1`; exact source and artifact hashes are frozen in `results/v4/confirmatory/result_manifest.json`.

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
