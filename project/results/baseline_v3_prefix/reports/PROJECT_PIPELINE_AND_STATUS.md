# Project Pipeline and Status

## Scope and current status

The project evaluates a nonlinear DC-motor plant controlled by constrained LSTM-MPC, with a residual/CUSUM sensor monitor that substitutes LSTM virtual feedback when the measured speed is suspect. The final claim is deliberately narrow: the tested reliability-aware controller improves simulated sensor-fault robustness. Hardware readiness and universal adaptive-MPC superiority are not claimed.

Status as of 2026-09-07:

- the plant, dataset, model, reliability, and MPC implementations are complete;
- notebook 03 has a clean sequential execution record and its saved model metrics were reproduced;
- notebook 06 has a clean sequential execution record and freshly regenerated all final evidence;
- the final configuration is fixed, multirate-consistent, and independent of measured wall-clock selection;
- the 220-run final holdout uses fresh seeds not used to redesign reliability logic;
- raw per-solve timings and representative final traces are persisted;
- the independent saved-evidence verifier passes before packaging cleanup;
- the remaining limitations are research/deployment limitations, not hidden artifact inconsistencies.

## Development pipeline

| Phase | Notebook/module | Purpose and retained decision |
|---:|---|---|
| 1 | `01_dc_motor_simulation.ipynb`, `src/motor_model.py` | Validate nonlinear current/speed dynamics, voltage response, load response, and passive coast-down. |
| 2 | `02_dataset_generation.ipynb`, `src/data_utils.py` | Generate 30 seeded open-loop trajectories; split whole runs 18/6/6 before windowing; fit normalization on training data only. |
| 3 | `03_lstm_training.ipynb`, `src/lstm_model.py` | Train a two-layer residual LSTM and save held-out one-step plus recursive prediction evidence. |
| 4A | `04_reliability_analysis.ipynb` | Test residual plus MC Dropout; retain MC Dropout only as a negative ablation. |
| 4B | `04b_reliability_redesign.ipynb` | Study guarded history, EWMA/CUSUM, and multi-step mismatch evidence. |
| 4C | `04c_reliability_finalization.ipynb`, `src/reliability.py` | Freeze the residual/CUSUM sensor monitor on development/validation evidence. |
| 5 | `05_mpc_experiments.ipynb`, `src/mpc.py` | Establish the constrained PI/MPC/reliability/adaptive closed loops and development studies. |
| 6 | `06_final_evaluation.ipynb` | Load a fixed controller configuration and run the fresh paired A/B/C/D final matrix. |

## Model evidence

The processed archive contains 18 training, 6 validation, and 6 test trajectories with no run-ID overlap. Each input window contains normalized voltage and measured speed; the target is true speed one sample ahead. Saved normalization matches independent reconstruction from training trajectories.

The residual LSTM has two recurrent layers, 64 hidden units per layer, dropout 0.2 between layers, and 50,753 trainable parameters. Training is deterministic on CPU with seed 2026, Adam, learning rate 0.001, batch size 512, at most 60 epochs, and validation best-state restoration.

- held-out RMSE: `0.140873 rad/s`;
- held-out MAE: `0.113106 rad/s`;
- held-out R²: `0.999906`;
- persistence RMSE: `0.289778 rad/s`;
- recursive endpoint RMSE H=5/10/15: `0.210952/0.336461/0.487796 rad/s`.

These recursive values are endpoint/lead-time RMSE, not an aggregate across all earlier forecast steps. The model is accurate enough for the simulation study but remains open-loop trained and accumulates recursive error.

## Reliability mechanism

At every plant sample, the LSTM produces a one-step virtual speed. The final online monitor evaluates measured-minus-virtual residual using an instantaneous gate and two-sided CUSUM with persistence/hysteresis. Suspected samples are replaced before entering the next autoregressive history window.

Health is evaluated from `raw_y_measured - y_hat`; substituted `y_feedback` is never used as recovery evidence. Exit requires a finite in-gate raw residual and CUSUM score below threshold for five consecutive samples, and CUSUM is reset only after that complete condition.

The active saved parameters are an instantaneous absolute threshold of `0.9372101 rad/s`, center `-0.0094168`, allowance `0.1403859`, CUSUM threshold `1.5696827`, three abnormal samples to enter, and five trusted samples to exit. They were calibrated on the clean validation operating distribution before the final holdout and may not transfer perfectly to substantially different operating points. MC Dropout is not used by the final controller.

Small bias, slow drift, dropout recovery, plant mismatch, and simultaneous sensor/plant disturbance remain imperfect. A single speed sensor cannot uniquely distinguish every sensor and plant cause.

## Corrected multirate MPC

The plant and LSTM advance every `dt_model=0.01 s`; the controller updates every `dt_control=0.05 s`. Therefore one control action is held for five model samples.

The frozen controller is `H=20`, `Nc=2`, `blocks=(5, 15)`, `maxiter=8`. Its first optimized move spans the next five model samples and its second spans the remaining fifteen. Every block boundary is aligned with the control-update cadence. The previous plan is expanded and shifted by five model samples for warm start. The objective is aligned to references `k+1 ... k+H`.

The same timing rule applies to model-quality evidence: after each adaptive solve, only the first five predicted speeds are queued because they correspond to the command that is actually held. Later ages from a candidate plan are not scored after receding-horizon replanning.

SLSQP receives analytic PyTorch gradients, voltage bounds `0–12 V`, and active slew limits. A bounded/rate-limited PI action is available as fallback. Telemetry records success, timing, iterations, objective evaluations, and all explicit safety-event classes.

Notebook 06 does not select H/Nc from wall-clock measurements. The configuration was frozen before the final seeds. On the development screen, H=20/Nc=1 versus H=20/Nc=2 produced RMSE `9.564905` versus `9.566216`; this negligible difference did not justify changing the documented two-move design. Runtime remains evidence, not a selection objective.

## Fresh final evaluation

Evaluation role: `untouched_final_holdout`. Seeds: `12026, 12027, 12028, 12029, 12030`. The paired matrix covers four controllers and eleven scenarios for 220 total runs. Finite-duration fault masks use `[event_start, event_end)`.

Controllers:

- A: PI;
- B: plain LSTM-MPC;
- C: LSTM-MPC with sensor reliability/virtual feedback;
- D: adaptive reliability-aware LSTM-MPC.

Scenarios include nominal/reference tracking, load disturbance, parameter variation, five isolated sensor-fault cases, and combined sensor-fault/load disturbance.

### Canonical headline values

| Quantity | Value |
|---|---:|
| Plain sensor-fault interval RMSE | `7.584044` rad/s |
| Reliability-aware sensor-fault interval RMSE | `1.374104` rad/s |
| Sensor-fault RMSE improvement | `81.881647%` |
| Plain disturbance/combined interval RMSE | `1.932806` rad/s |
| Adaptive disturbance/combined interval RMSE | `4.709976` rad/s |
| Signed adaptive degradation metric `(adaptive-plain)/plain` | `143.685962%` (worsening) |

The sensor result is a macro average across noise, 5% bias, 15% bias, dropout, and drift. The adaptive result is a macro average across load disturbance, parameter variation, and the combined case. Correcting recovery removes the combined-case false exit but increases continued substitution under plant mismatch, so the positive signed change is a substantial worsening rather than an improvement. Combined-case virtual-feedback RMSE during substitution is about `11.71 rad/s`, worse than the corrupted-sensor RMSE of about `3.13 rad/s`.

The five closed-loop seeds measure robustness across simulation noise/random seeds. They do not measure robustness across LSTM training seeds; the single frozen LSTM was not retrained in this bug-fix phase.

### Safety and runtime

Across all 220 runs:

- optimizer failures: 0;
- voltage-limit violations: 0;
- slew-limit violations: 0;
- non-finite events: 0.

A targeted parameter-variation run forced PI fallback for ten consecutive controller updates. Speed remained bounded at `38.976 rad/s`, voltage/slew violations stayed at zero, and MPC resumed with a `0.094 V` voltage jump under the `2 V` active limit.

Notebook 06 persisted 19,965 individual MPC solver/control-computation samples. Recomputed statistics are:

| Statistic | Value |
|---|---:|
| Mean | `67.963908 ms` |
| Median | `60.554900 ms` |
| p95 | `135.782580 ms` |
| p99 | `166.619316 ms` |
| Maximum | `666.868900 ms` |

These are solver/control-computation measurements, not end-to-end deployment latency. They are CPU/load dependent. `58.442274%` exceeded 50 ms and `23.060356%` exceeded 100 ms. The histogram/ECDF therefore confirms that the implementation is simulation-oriented and not established as real-time at 20 Hz.

## Canonical artifact set

- `results/configs/final_mpc_config.json`
- `results/metrics/final_controller_runs.csv`
- `results/metrics/final_controller_comparison.csv`
- `results/metrics/final_runtime_samples.csv`
- `results/metrics/final_summary.json`
- `results/metrics/virtual_feedback_quality.csv`
- `results/metrics/bugfix_scenario_comparison.csv`
- `results/metrics/pi_fallback_validation.json`
- `results/raw/final_representative_traces.npz`
- `results/plots/final_nominal_step_tracking.png`
- `results/plots/final_load_disturbance.png`
- `results/plots/final_sensor_bias.png`
- `results/plots/final_sensor_dropout.png`
- `results/plots/final_combined_fault_load.png`
- `results/plots/final_runtime_comparison.png`
- `results/plots/final_runtime_distribution.png`

`final_controller_runs.csv` is the source for paired metrics, `final_runtime_samples.csv` is the source for pooled runtime statistics, and `final_summary.json` contains only quantities derived from those sources plus the exact frozen configuration.

## Reproduction and verification

All declared Python requirements are exactly version-pinned. Execute notebooks 01, 02, 03, 04, 04b, 04c, 05, and 06 in that order from the repository root. Notebook 06 is computationally expensive and timing values will vary, but its controller configuration and scenario seeds do not vary by machine.

For saved-evidence verification:

```bash
python scripts/verify_results.py
```

The verifier independently checks split leakage, normalization, model metrics, exact final configuration, matrix completeness/pairing, aggregate CSV consistency, headline formulas, raw runtime distribution, safety totals, trace finiteness, plots, and documentation agreement.

## Final limitations

- All evidence is simulated; no hardware timing, calibration, or safety validation has been performed.
- The LSTM is trained open loop and recursive error grows with horizon.
- The reliability residual can react to operating-point/model mismatch.
- Small bias and drift remain difficult. Finite bias/dropout recovery is conservative enough that latency is right-censored beyond the two-second post-fault observation window.
- The combined sensor-fault/load case remains substantially harder: correct latching exposes poor virtual-feedback quality under concurrent plant mismatch.
- Adaptive behavior is scenario-dependent and the corrected three-scenario aggregate worsens.
- The SLSQP/PyTorch implementation is not reliably real-time at 20 Hz.
- Future tuning must use new development data; seeds `12026–12030` are now consumed final holdout evidence.

The pre-fix evidence conflict is documented in `FINAL_REPO_AUDIT.md`; the remediated independent verdict is in `FINAL_REPO_AUDIT_POSTFIX.md`.
