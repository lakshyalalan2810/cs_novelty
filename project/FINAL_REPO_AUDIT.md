# Final Independent Repository Audit

**Project:** Reliability-Aware LSTM-Based Model Predictive Control for Fault-Tolerant Nonlinear DC Motor Control  
**Audit date:** 2026-09-07  
**Scope:** Independent verification of the current repository, saved artifacts, methodology, implementation, numerical claims, reproducibility, and documentation. No model retraining, controller retuning, or experimental artifact modification was performed.

# Executive verdict

## NOT READY

The LSTM dataset, train-only normalization, one-step metrics, persistence baseline, and genuinely recursive evaluation are independently reproducible and methodologically sound. The local MPC objective, voltage bounds, decision-to-decision slew constraints, and clipped failure fallback are also implemented correctly on the optimizer's internal grid.

Submission is nevertheless blocked for two independent reasons:

1. The current canonical controller artifacts are an exploratory `H=5`, effective `Nc=1` run, not the documented final `H=20`, `Nc=2`, `maxiter=8` experiment. They reproduce `0.0%`, not `81.4289%`, sensor-fault improvement and do not support the documented adaptive or runtime claims.
2. The documented `H=20`, `Nc=2` plan is temporally inconsistent with the simulated 20 Hz controller. The optimizer assumes its first move lasts one 10 ms model step, while the plant actually holds that move for the 50 ms control interval. This also contaminates the adaptive model-quality score.

The first problem is an evidence-integrity failure. The second is a control-methodology error that requires correction and a new final evaluation. Restoring old `H=20` artifacts alone would not resolve the second problem.

# Critical findings

## 1. Current canonical artifacts contradict the headline controller results

`results/configs/final_mpc_config.json` and `results/metrics/final_summary.json` record `H=5`, `move_blocks=[5]` (one independent move, effective `Nc=1`), and `maxiter=8`. The 220-row `results/metrics/final_controller_runs.csv` is internally complete but belongs to that configuration.

Independent recomputation from the current CSV gives:

- Sensor scenarios: plain MPC `33.9014302611 rad/s`, sensor-reliability MPC `33.9014302611 rad/s`, improvement `0.0%`.
- Disturbance/combined scenarios: plain MPC `29.1408232472 rad/s`, adaptive MPC `29.1408232472 rad/s`, degradation `0.0%`.
- MPC solve time: mean `9.2811014075 ms`, saved pooled p95 `14.8764400 ms`, maximum `31.6085000 ms`.
- Optimizer failures, voltage violations, and slew violations: `0` for the current H=5 run.

`scripts/verify_results.py:179` correctly fails with `(0.0, 81.42894781890575)`. The documented `81.4289%`, `18.5827%`, and `69.4594/131.6933/290.2360 ms` values have no current canonical H=20 CSV/NPZ evidence. `README.md:111` and `PROJECT_PIPELINE_AND_STATUS.md:279,415,425-431,440` disclose the overwrite, but disclosure does not make the submission evidence reproducible.

**Impact:** The central closed-loop result, final configuration, final safety counts, and exact final runtime values cannot be independently verified from this repository.

## 2. The H=20/Nc=2 prediction plan conflicts with the 50 ms actuation grid

Notebook 06 uses a 10 ms plant/model step (`DT=0.01`) and calls the controller only every five samples (`CONTROL_STRIDE=5`), holding the returned voltage for 50 ms. In `src/mpc.py:157-160`, `H=20`, `Nc=2` expands to blocks `(1,19)`. The optimizer therefore predicts:

- decision 0 for 10 ms;
- decision 1 for the remaining 190 ms.

The simulated plant actually receives decision 0 for 50 ms before the next optimization. For the implemented 20 Hz zero-order hold, two planned moves on the 10 ms model grid would need temporally compatible blocking such as `(5,15)`, or the controller would need to execute every 10 ms. The existing `(1,19)` result is algebraically a two-variable plan but not the plan executed by the plant.

`RollingPredictionQuality` compounds the problem: it compares observations at every 10 ms sample with forecasts that assume the second decision was applied after 10 ms. From forecast age 1 until the next 50 ms controller update, it can interpret plan/actuation mismatch as model mismatch and alter the adaptive horizon, move penalty, and slew limit.

**Impact:** The documented final MPC and adaptive model-quality experiments do not validate a temporally consistent 20 Hz formulation. Correcting this changes experimental meaning and requires rerunning the final matrix; it was therefore not modified during this audit.

# Major findings

## Final configuration cannot be regenerated deterministically

Notebook 06 does not encode `H=20/Nc=2` as a fixed final choice. It dynamically selects `SELECTED_H` and `SELECTED_NC` from a score containing measured wall-clock time, then dynamically selects a blocking strategy. The saved run selected H=5/Nc=2 followed by `(5,)`, producing effective Nc=1. No later H=20 override exists.

The documented instruction to rerun notebook 06 is therefore insufficient: an unchanged rerun is hardware/load dependent and currently regenerates H=5, not the stated final configuration.

## Reliability evaluation reuses its nominal “test” trajectories during redesign

The LSTM train/validation/test split is clean. However, Phases 4, 4B, and 4C repeatedly use the same `split_run_ids_test` trajectories and related injected fault families while diagnosing, redesigning, and selecting the reliability architecture. Phase-4C fault results are held out from LSTM fitting but are not an untouched reliability test.

Threshold values themselves are fitted from clean validation trajectories only. The limitation is architectural/test reuse, not direct numerical threshold fitting on fault-test samples. Reliability results should be described as development/benchmark evidence unless evaluated on a new reserved set.

## Virtual-feedback recovery is circular and was integrated despite a no-go result

Immediate substitution and history protection are implemented as documented. While a sensor is suspect, however, predictions are inserted into the autoregressive history; recovery is then judged from measured-minus-predicted residuals generated from that virtual history. Exit checks five samples below the instantaneous gate but does not require the CUSUM score to have recovered.

`results/metrics/reliability_final_decision.json` records `ready_for_mpc=false`, `40.6554%` dropout post-recovery alarms, and recommends an independent recovery reference before closed-loop use. Later notebooks correctly removed MC Dropout and hard model-mismatch switching, but retained this sensor latch. It is defensible as an experimental gross-fault fallback, not a deployment-safe recovered-sensor selector.

## Reliability calibration does not transfer cleanly to closed loop

The saved validation sensor-state false-alarm rate is `0.07056%`, while held-out clean reliability data reports `2.25797%`. Current H=5 closed-loop artifacts show roughly `17-19%` substitution even in nominal/reference scenarios and about `47-48%` under load/parameter mismatch. The residual reacts to operating-point/model mismatch as well as sensor corruption.

The saved validation FAR also counts the debounced alarm, whereas actual substitution is `active OR instantaneous`; it therefore understates false virtual substitutions.

## Runtime baseline and pooled p95 are not independently supported

Notebook 06 hard-codes the Phase-5 baseline as `199.6 ms`. Saved Phase-5 traces and `mpc_all_metrics.csv` instead give an all-MPC mean of `151.4969673 ms`, matching `mpc_conclusions.json` and notebook 05. A narrow subset is near 199.8 ms, but the subset is not documented as the baseline. The claimed `65.2007%` reduction therefore compares unlike or undocumented populations; using the saved all-MPC mean would give about `54.15%`, subject to the missing final H=20 evidence.

The pooled final p95 is calculated from per-call samples that are not persisted. Per-run mean/p95/max columns cannot reconstruct a pooled percentile, and `scripts/verify_results.py:184-185` only trusts the JSON value and checks that p95 exceeds the mean.

## Notebook 03 is not a clean sequential execution record

The first setup cells have null execution counts, the configuration cell preserves an `ipykernel` error, and later cells resume with execution counts 3-13 and populated outputs. The source is self-contained and the saved LSTM artifacts independently verify, but the notebook itself shows stale/hidden execution provenance. It should be restarted and run linearly in the submission environment.

## Clean-clone/version restoration is unavailable

The delivered project has no project-scoped Git metadata, tracked history, or remote. The documented instruction to restore H=20 artifacts from version history cannot be performed from this repository. Exact environment reproduction is also weakened by unpinned requirements; validated versions appear only in prose.

# Minor findings

- `src/lstm_model.py:35` adds input-normalized measured speed directly to a target-normalized output although those normalization coordinates differ slightly. The trained head compensates and measured accuracy is valid, but this is not an exact physical persistence residual. Do not change it without retraining.
- Recursive H=5/10/15 values are endpoint/lead-time RMSE, not aggregate RMSE across steps 1 through H. The aggregate values are `0.172393`, `0.236360`, and `0.313426 rad/s`; documentation should say “endpoint RMSE.”
- Finite sensor faults are injected over `start <= t < end`, while the metric mask uses `start <= t <= end`, adding one clean sample at `t=end` and slightly diluting fault-window RMSE.
- The first MPC prediction is for `y[k+1]`, but notebook 06 builds its reference vector on `t[k]...t[k+H-1]`. Reference changes are shifted by one 10 ms sample; constant-reference cases are unaffected.
- Warm start shifts the old expanded sequence by one model step, not the five-step control interval (`src/mpc.py:191-197`). For `(1,19)` the numerical initial guess is coincidentally the same after a one- or five-step shift, but other horizon/blocking strategies receive inconsistent warm starts, biasing runtime/strategy comparisons.
- The PI backup evolves separately from the applied MPC command, so fallback is not bumpless. The MPC and notebook re-clip it against the actual previous command, preserving hard voltage/slew safety.
- Notebook 03 calls PyTorch's stacked-layer dropout “recurrent dropout”; `src/lstm_model.py:24` implements inter-layer dropout.
- `requirements.txt` lists the used top-level packages but does not pin versions or provide a lock file.
- `.venv`, `.uv-cache`, and Python caches have returned. `.venv` plus `.uv-cache` contain about 69,500 files and 1.95 GB. They are ignored by `.gitignore`, but would bloat a folder/ZIP submission. This contradicts cleanup claims in `REPOSITORY_AUDIT_REPORT.md` and parts of `PROJECT_PIPELINE_AND_STATUS.md`.

# Confirmed correct components

- Whole motor trajectories are split before overlapping windows are constructed. Train, validation, and test run IDs are pairwise disjoint, and window run IDs match their parent split.
- Normalization statistics are fitted only on training windows. Independent recomputation matched saved statistics to a maximum absolute error of `5.41e-13`.
- Input/target alignment is correct: 20 samples `[s...s+19]` predict true speed at `s+20`.
- No injected fault data enters LSTM training.
- LSTM architecture, parameter count, loss, optimizer, batch size, validation selection, save/load, and inverse normalization match the saved configuration.
- One-step LSTM, persistence, and recursive metrics reproduce from saved data and weights.
- Recursive evaluation is genuinely recursive: predicted speed, not future measured/true speed, is fed back. Future voltage is treated as a known exogenous input, and future truth is used only for scoring.
- R² is computed correctly. Its very high value is partly expected from the large between-operating-point variance; per-run R² remains high, so no artificial inflation mechanism was found.
- Reliability residual, rolling/EWMA features, two-sided CUSUM equations, per-trajectory reset, immediate guard, and 3-enter/5-exit persistence match the implementation.
- Direct reliability thresholds use clean validation IDs, not test faults.
- MC Dropout remains available only as an ablation utility and is absent from the final closed-loop core.
- The MPC objective uses physical-speed tracking error plus squared voltage moves, including the first move from the previously applied voltage.
- Optimizer bounds and decision-to-decision slew constraints are correctly constructed on the optimizer grid.
- Only the first optimized command is applied; unsuccessful/non-finite SLSQP results reset warm start and return a voltage/rate-clipped fallback.
- Applied commands are clipped again in the notebook, and applied voltage/slew violations are measured at control instants.
- The experimental matrix is balanced and paired: every controller receives the same eleven scenarios and seeds 2026-2030; noise is generated only from scenario and seed.
- Controller percentage formulas use the correct baseline denominator and consistently use `fault_interval_RMSE`. They are macro averages of per-run RMSE, not pooled-sample RMSE.
- Documentation prominently retains negative results: MC Dropout rejection, weak parameter-shift detection, small-bias/drift and recovery limitations, adaptive degradation, combined sensor/load weakness, and simulation-only evidence.
- The Phase-3 H=15 open-loop recommendation versus H=20 closed-loop choice is explained as different criteria, not a contradiction. The current tuning table supports lower tail oscillation at H20/Nc2 than H15/Nc2, though it does not validate the final multirate implementation.

# Verified numerical results

| Claim | Documented value | Verified value | Status | Source artifact/code |
|---|---:|---:|---|---|
| LSTM test RMSE | `0.1408727 rad/s` | `0.1408727020` | PASS | Dataset, weights, `lstm_test_metrics.json`, independent inference |
| LSTM test MAE | `0.1131055 rad/s` | `0.1131054834` | PASS | Same as above |
| LSTM test R² | `0.9999062` | `0.9999061823` | PASS | Same as above |
| LSTM range-NRMSE | `0.00211557` | `0.00211557397` | PASS | Same as above |
| Persistence RMSE | `0.2897776 rad/s` | `0.2897775835` | PASS | Recomputed from last measured-speed input |
| Recursive endpoint RMSE H=5 | `0.2109516 rad/s` | `0.2109516` | PASS | Independent recursive rollout |
| Recursive endpoint RMSE H=10 | `0.3364611 rad/s` | `0.3364611` | PASS | Independent recursive rollout |
| Recursive endpoint RMSE H=15 | `0.4877964 rad/s` | `0.4877962768` | PASS | Independent recursive rollout |
| Final configuration | `H=20, Nc=2, maxiter=8` | Current `H=5, Nc=1, maxiter=8` | FAIL | `final_mpc_config.json`, `final_summary.json` |
| Sensor-fault RMSE improvement | `81.4289%` (`7.587208 -> 1.409024`) | Current artifacts: `0.0%` (`33.901430 -> 33.901430`) | FAIL | Current 220-row controller CSV |
| Adaptive disturbance degradation | `18.5827%` (`1.909355 -> 2.264166`) | Current artifacts: `0.0%` (`29.140823 -> 29.140823`) | FAIL | Current 220-row controller CSV |
| Optimizer failures | `0` in final H20 matrix | `0` in current H5 matrix; H20 final unavailable | WARNING | Current controller CSV |
| Voltage violations | `0` in final H20 matrix | `0` in current H5 matrix; H20 final unavailable | WARNING | Current controller CSV |
| Slew violations | `0` in final H20 matrix | `0` in current H5 matrix; H20 final unavailable | WARNING | Current controller CSV |
| Final mean solve time | `69.4594 ms` | Current H5: `9.2811 ms`; H20 final unavailable | FAIL | Current controller CSV/summary |
| Final pooled p95 | `131.6933 ms` | Current saved H5: `14.87644 ms`; neither pooled sample set persisted | FAIL | `final_summary.json`, notebook calculation |
| Final maximum solve time | `290.2360 ms` | Current H5: `31.6085 ms`; H20 final unavailable | FAIL | Current controller CSV/summary |
| Phase-5 runtime baseline | `199.6 ms` | Saved all-MPC mean `151.49697 ms` | FAIL | `mpc_closed_loop_traces.npz`, `mpc_all_metrics.csv`, `mpc_conclusions.json` |
| Reliable 20 Hz execution | `No` for documented H20 | Directionally supported, exact final evidence unavailable; current H5 JSON says `true` | WARNING | H20/Nc2 tuning row, Phase-5 traces, current H5 summary |

The historical percentage formulas are arithmetically correct: `(7.5872084-1.4090244)/7.5872084 = 81.428948%` and `(2.2641661-1.9093554)/1.9093554 = 18.582748%`. The failure is missing current source evidence, not arithmetic.

# Methodology verification

## Dataset

**PASS for LSTM development.** Thirty complete trajectories are split 18/6/6 before windowing. Overlapping windows stay inside a single trajectory and split. Saved windows and targets reconstruct exactly from raw trajectories. Normalization is training-only.

**WARNING for reliability evaluation.** The nominal test trajectories are reused during detector diagnosis and redesign, so reliability fault metrics are developmental rather than a pristine final holdout.

## LSTM

**PASS.** Architecture and training match configuration. Test metrics and persistence independently reproduce. Recursive prediction is causal and teacher-forcing-free. The very high R² is real for this synthetic operating range, though RMSE and the persistence comparison are more informative.

## Reliability

**WARNING.** Threshold calibration itself is clean-validation-only and the signal equations are correct. Gross-fault history protection is immediate, but small bias, drift, parameter mismatch, online false substitution, and dropout recovery remain weak. Recovery is not sensor-independent. MC Dropout is correctly rejected from the core.

## MPC

**FAIL overall.** The mathematical cost and constraints are locally correct, but the planned move timing is inconsistent with the 50 ms zero-order hold. H20/Nc2 is implemented as two decisions `(1,19)`, not as two temporally executable decisions for a five-model-step control interval. Warm-start timing and the reference grid also need alignment before final rerun.

## Metrics

**PASS for formulas and current-table aggregation; FAIL for headline evidence.** Scenario/seed pairing, true-speed error, denominators, and fault-window use are sound. Aggregates are equal-weight macro means. The current table is H5 and cannot verify historical H20 claims; the finite-fault endpoint includes one clean sample.

## Runtime

**FAIL for exact final values.** Timing measures only `LSTMMPC.compute_control`, excluding reliability, quality-monitor, PI-backup, and plant work, so it is core solve time rather than end-to-end loop latency. The documented H20 values are absent, pooled p95 samples were not retained, and the Phase-5 baseline is inconsistent. Available H20 tuning/Phase-5 evidence still supports the conservative conclusion that reliable 20 Hz execution has not been demonstrated.

# Reproducibility assessment

Another person cannot currently reproduce the submitted result from a clean clone:

- There is no project Git history or remote from which to clone or restore H20 artifacts.
- Notebook 06 dynamically selects a hardware-dependent H/Nc and currently produces H5/Nc1.
- Notebook 03 is not saved as a clean sequential execution.
- Exact package versions are not pinned.
- The H20 final CSV, config, trace evidence, and pooled timing samples are absent.
- Local `.venv` and `.uv-cache` directories total about 1.95 GB and must not enter a folder/ZIP submission.

Relative source/config paths, notebook ordering, explicit seeds, saved dataset/weights, and dependency coverage are otherwise adequate.

# Documentation consistency

The documentation is commendably direct about negative results and the artifact overwrite. It does not claim universal adaptive superiority, reliable parameter-shift detection, MC Dropout effectiveness, combined-fault success, or proven 20 Hz real-time operation. The combined sensor/load weakness is prominent, and the H15 open-loop versus H20 closed-loop explanation is clear.

It is still not submission-consistent because sections titled “Verified main results” present historical numbers unsupported by current canonical artifacts. `PROJECT_PIPELINE_AND_STATUS.md` correctly marks submission blocked, but its advice to rerun notebook 06 is incomplete because the current source reselects H5. Its cleanup/environment-complete statements and `REPOSITORY_AUDIT_REPORT.md` cleanup record are also stale because local environments/caches have returned.

# Known limitations

- Combined sensor corruption plus load disturbance remains the most important application limitation and is visible enough in the docs.
- Small bias and slow drift remain difficult; final reliability evidence is development-used rather than independently held out.
- Parameter-shift/model-mismatch detection is weak and should remain monitoring-only.
- Dropout recovery can remain latched; virtual-history recovery lacks an independent reference.
- Adaptive disturbance handling did not consistently improve tracking in the historical record; its claimed `18.6%` degradation is currently not independently reproducible.
- The 81.4% historical improvement covers isolated fault-window macro averages only. It excludes recovery and the combined sensor/load scenario.
- Combined testing covers one +5%-full-scale bias plus load case, not combined noise, dropout, or drift with load.
- Recursive LSTM error grows with horizon; all plant/controller evidence is simulation-only.
- End-to-end reliable 20 Hz execution has not been demonstrated.

# Final defensible contribution

The current repository supports the following statement:

> A compact residual LSTM was trained on disjoint synthetic DC-motor trajectories using train-only normalization and achieves independently reproducible one-step and recursive prediction accuracy. The repository implements a constrained SLSQP LSTM-MPC and a clean-validation-calibrated residual/CUSUM virtual-feedback monitor. Development evidence shows that gross isolated sensor faults can be detected more reliably than small bias, drift, plant mismatch, or recovery, while MC Dropout and universal adaptive-MPC superiority are not supported.

The current repository does **not** support a submission claim that the final H20/Nc2 controller achieved an independently verified 81.4% sensor-fault improvement, an 18.6% adaptive degradation, zero final H20 failures/violations, or the exact documented runtime distribution. Those claims require a temporally corrected, fixed-configuration, freshly executed final matrix with preserved artifacts.

# Submission checklist

| # | Cross-check | Result | Answer |
|---:|---|---|---|
| 1 | Any train/test/data leakage? | WARNING | No LSTM split or normalization leakage. Reliability “test” trajectories were reused during method redesign, so those results are not an untouched test. |
| 2 | Are all LSTM metrics computed correctly? | PASS | Yes. One-step, persistence, R², inverse normalization, and recursive values reproduce. Clarify that H5/H10/H15 are endpoint RMSE. |
| 3 | Is recursive prediction genuinely recursive? | PASS | Yes. Predictions are fed back; future truth is scoring-only. |
| 4 | Is reliability calibration methodologically valid? | WARNING | Numerical thresholds use clean validation only, but architecture selection reuses test scenarios and calibration FAR transfers poorly online. |
| 5 | Is virtual sensor substitution implemented correctly? | WARNING | Immediate causal substitution/history protection is correct; recovery is circular, can latch, and was not independently validated. |
| 6 | Is the MPC formulation correct? | FAIL | Cost/constraints are correct locally, but planned control timing conflicts with the 50 ms execution interval. |
| 7 | Are H=20 and Nc=2 implemented as documented? | FAIL | The code maps them to `(1,19)`, which is temporally wrong at 20 Hz, and current final artifacts are H5/Nc1. |
| 8 | Are constraints genuinely enforced? | WARNING | Bounds/slew and applied-command clipping are real. Zero violations are verified only for current H5, not the missing final H20 run. |
| 9 | Are warm start and move blocking correct? | FAIL | Move timing and the one-step warm shift do not match the five-step receding interval; the `(1,19)` warm guess is only coincidentally unaffected. |
| 10 | Are optimizer failures handled safely? | WARNING | Failure returns a voltage/rate-clipped command and resets warm start. Backup PI is not bumpless, and no current H20 matrix verifies the claimed zero failures. |
| 11 | Is the ~81.4% sensor-fault improvement reproducible and fair? | FAIL | Formula/pairing are fair for a macro isolated-fault-window comparison, but current canonical artifacts reproduce 0.0%, not 81.4%. |
| 12 | Is the ~18.6% adaptive disturbance degradation correct? | FAIL | Formula is arithmetically correct, but current canonical artifacts reproduce 0.0%; historical source evidence is missing. |
| 13 | Are runtime measurements trustworthy? | FAIL | Exact H20 artifacts and pooled samples are missing; timing is solve-only and the 199.6 ms baseline conflicts with saved Phase-5 evidence. |
| 14 | Is the “not real-time at 20 Hz” conclusion correct? | WARNING | Directionally conservative and supported by available H20 tuning/Phase-5 evidence, but the exact final H20 runtime claim is not independently reproducible. |
| 15 | Are any claims stronger than the evidence? | FAIL | “Verified” H20 headline values are stronger than current artifacts; reliability results are development-used and recovery/combined coverage is narrow. |
| 16 | Is the combined sensor + load weakness visible enough? | PASS | Yes. README and status documents identify it repeatedly as the main limitation. |
| 17 | Does PROJECT_PIPELINE_AND_STATUS.md accurately reflect the repository? | WARNING | It correctly records the overwrite and submission block, but the rerun path, cleanup status, and historical evidence wording are incomplete/stale. |
| 18 | Can the project be reproduced from a clean environment? | FAIL | Not the final submitted result: no clone/history, no fixed H20 execution path, unpinned environment, stale notebook 03 state, and missing H20 artifacts. |

## Required pre-submission actions

1. Correct the multirate control blocking and warm-start semantics for the 50 ms execution interval; align the reference grid.
2. Encode one explicit final configuration instead of dynamically reselecting it from wall-clock timing.
3. Rerun the complete paired final matrix after the correction; preserve the config, 220-run CSV, aggregate table, representative traces, and per-call timing samples.
4. Make the verifier derive expected claims from the canonical final configuration/artifacts and require the configured H/Nc/maxiter.
5. Rerun notebook 03 with Restart/Run All and preserve a clean sequential record.
6. Describe reliability fault results as development benchmarks or add a genuinely untouched reliability holdout; keep recovery and combined-fault limitations explicit.
7. Reconcile the Phase-5 runtime baseline, pin or lock the validated environment, place the project under version control, and exclude local environments/caches from the submitted package.

No code, model, controller, threshold, metric, or experimental result was changed during this audit.
