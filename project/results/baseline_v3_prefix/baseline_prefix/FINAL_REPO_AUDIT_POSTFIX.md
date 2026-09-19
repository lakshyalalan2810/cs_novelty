# Final Independent Post-Fix Repository Audit

Audit date: 2026-09-07  
Scope: corrective validation of the repository after the blockers in `FINAL_REPO_AUDIT.md`  
Auditor role: independent post-fix reviewer; no implementation changes were made during this audit

## Executive verdict

## READY FOR SUBMISSION

The canonical source, fixed configuration, executed notebooks, raw final results, summary, verifier, and current documentation now agree. The multirate timing defect has been corrected and the final A/B/C/D evidence is independently reproducible from the saved per-run and per-solve records.

This verdict is for the stated simulation research contribution. It is not a hardware-deployment verdict: the measured MPC solver distribution does not reliably meet the 50 ms control deadline.

`FINAL_REPO_AUDIT.md` remains the historical pre-fix audit. This report supersedes its repository-readiness verdict.

## Audit method and evidence

I inspected the current implementation and notebook sources, parsed both executed notebooks directly, recomputed the controller aggregates and runtime distribution from the raw CSV rows using an independent PowerShell calculation, decompressed and inventoried the representative NPZ, checked the final plots, reviewed the verifier line by line, and checked the packaged tree for excluded caches/environments.

The full Python verifier had passed immediately before packaging cleanup. I did not rely on that result alone: the independent raw-row calculation below reproduced every canonical controller, runtime, and safety value.

## Corrective fixes verified

- Multirate MPC now carries an explicit `control_interval_steps=5` invariant and rejects sub-control-interval multi-block schedules such as `(1, 19)`.
- The fixed `H=20`, `Nc=2` plan expands as `blocks=(5, 15)`.
- Warm start shifts the prior expanded plan by five model samples, and `src/mpc.py` includes a deterministic assertion demonstrating the shift.
- The prediction target is aligned to future references `r[k+1] ... r[k+H]`.
- Adaptive model-quality monitoring retains only the first five forecast ages, for which the predicted command and the actually held command are identical.
- The final evaluation loads a fixed JSON configuration; development timing results cannot select or mutate it.
- Fresh final seeds are paired across every controller/scenario cell and are absent from the earlier reliability-development notebooks.
- Finite fault injection and scoring both use the end-exclusive interval `[event_start, event_end)`.
- Every final MPC timing observation, all four safety-event classes, and representative trajectories are persisted.
- Notebook 03 and notebook 06 are clean, sequential execution records.
- The validated dependency versions are exact-pinned, and local environments/caches are absent from the packaged tree.

## Corrected timing interpretation

The model/plant grid is `dt_model=0.01 s`; the receding controller grid is `dt_control=0.05 s`. One optimization therefore advances five model steps.

For the fixed plan, decision 1 is repeated over prediction steps 1-5 and is applied by the simulated plant for the five transitions ending at `y[k+1] ... y[k+5]`. Decision 2 covers prediction steps 6-20. At the next optimization, the old expanded sequence advances by five samples. Reference construction uses `instant + dt_model * [1, ..., H]`, so the first LSTM prediction is scored against `r[k+1]`, not `r[k]`.

`RollingPredictionQuality` is also actuation-consistent. A solve queues only prediction ages 1-5 from its `(5, 15)` schedule; those observations mature before the next plan replaces the held command. Predictions from the unexecuted tail are not scored.

## Frozen final configuration

| Field | Canonical value |
|---|---:|
| Evaluation role | `untouched_final_holdout` |
| Final seeds | `12026, 12027, 12028, 12029, 12030` |
| Plant/model step | `0.01 s` |
| Control interval | `0.05 s` |
| Control stride | `5` model samples |
| Prediction horizon | `H=20` |
| Effective control horizon | `Nc=2` |
| Move blocks | `blocks=(5, 15)` |
| Iteration cap | `maxiter=8` |
| Warm start | enabled; five-sample shift |
| Voltage bounds | `0-12 V` |
| Nominal maximum slew | `2 V` per control update |

The configuration file predates the regenerated run artifacts, is loaded before the final matrix is created, and is copied exactly into `final_summary.json`. The final path contains no runtime-driven H/Nc selection.

The development screen gives RMSE `9.564905` for H=20/Nc=1 and `9.566216` for H=20/Nc=2. The latter is worse by only about 0.0137% in RMSE and reduces control variation; it is a genuine control-performance near-tie, not a material regression. Nc=1 is faster, but runtime was correctly excluded from final control-performance selection. Retaining the intended two-move design is therefore defensible and disclosed.

## Final holdout and artifact completeness

The final matrix contains four controllers, eleven scenarios, and five identical paired seeds per controller/scenario cell:

- `final_controller_runs.csv`: 220 rows;
- `final_controller_comparison.csv`: 44 rows, each aggregating five repetitions;
- `final_runtime_samples.csv`: 19,965 unique rows;
- runtime coverage: 165 MPC controller/scenario/seed runs with exactly 121 solve samples each;
- `final_representative_traces.npz`: 968 arrays covering all 44 controller/scenario pairs and 22 fields;
- six required final plots are present and non-empty.

The seeds `12026-12030` occur in the final configuration/evaluation evidence and verification/documentation, but not in notebooks 04b, 04c, or 05. Earlier reliability experiments use development seed 2026. No threshold or final structural parameter is fitted from the final matrix.

## Independently reproduced numerical results

The sensor aggregate is the equal-weight mean of per-run fault-window RMSE over sensor noise, 5% bias, 15% bias, dropout, and drift. The disturbance aggregate covers load disturbance, parameter variation, and the combined sensor-fault/load case.

| Quantity | Independent result | Saved/docs | Result |
|---|---:|---:|---|
| Plain MPC sensor-fault RMSE | `7.584044` | `7.584044` | PASS |
| Reliability-aware sensor-fault RMSE | `1.375062` | `1.375062` | PASS |
| Sensor-fault improvement | `81.869014%` | `81.869014%` | PASS |
| Plain MPC disturbance/combined RMSE | `1.932806` | `1.932806` | PASS |
| Adaptive disturbance/combined RMSE | `1.757076` | `1.757076` | PASS |
| Signed adaptive degradation `(adaptive-plain)/plain` | `-9.091949%` | `-9.091949%` | PASS |

The negative adaptive value means a 9.091949% aggregate improvement, not degradation. The result is mixed by scenario: adaptive control is slightly worse on isolated load and parameter-variation cases and better on the combined case. The repository correctly avoids a universal-superiority claim.

## Runtime and safety

The following values were recomputed directly from all 19,965 `compute_ms` values using the ordinary linear 95th percentile, not from per-run p95 values:

| Statistic | Independent result | Saved/docs | Result |
|---|---:|---:|---|
| Mean | `72.760214 ms` | `72.760214 ms` | PASS |
| Median | `67.904100 ms` | `67.904100 ms` | PASS |
| p95 | `147.414480 ms` | `147.414480 ms` | PASS |
| Maximum | `373.799900 ms` | `373.799900 ms` | PASS |

All raw timing rows are finite and non-negative, lie on five-sample control instants, have unique controller/scenario/seed/sample keys, and record optimizer success. The obsolete undocumented 199.6 ms comparison is not used.

Independent sums over all 220 final runs are:

| Safety event | Count | Result |
|---|---:|---|
| Optimizer failures | `0` | PASS |
| Voltage violations | `0` | PASS |
| Slew/rate violations | `0` | PASS |
| NaN/Inf events | `0` | PASS |

These timings measure `compute_control` solver/control computation only. They exclude end-to-end scheduling, I/O, plant integration, and hardware latency and are machine/load dependent. Mean, p95, and maximum exceed the 50 ms interval, so reliable real-time 20 Hz execution has not been established.

## Notebook, verifier, and packaging checks

- `03_lstm_training.ipynb`: 13 code cells with execution counts 1-13, no null counts, no error outputs.
- `06_final_evaluation.ipynb`: 6 code cells with execution counts 1-6, no null counts, no error outputs.
- `scripts/verify_results.py` checks dataset split isolation, train-only normalization, saved model metrics, exact final H/Nc/blocks/stride/maxiter, all matrix cells and seeds, comparison-table means, headline formulas, all four safety totals, raw timing uniqueness/grid/mean/median/p95/max, trace finiteness, required plots, and exact six-decimal documentation tokens.
- `README.md` and `PROJECT_PIPELINE_AND_STATUS.md` contain the same canonical configuration and ten six-decimal headline/runtime values as the JSON/CSV evidence.
- `requirements.txt` contains only exact `==` pins.
- No `.venv`, `.uv-cache`, `.uv-python`, `__pycache__`, `.ipynb_checkpoints`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `build`, or `dist` directory was found.

## Remaining limitations

- All controller and safety evidence is simulated; there is no hardware validation.
- The open-loop-trained LSTM accumulates recursive forecast error.
- The residual monitor cannot always distinguish sensor corruption from plant/model mismatch with one speed signal.
- Drift receives no material benefit in the current isolated-fault comparison, and recovery/combined behavior remains scenario-dependent.
- The three-scenario adaptive improvement is driven by the combined case; isolated disturbance results do not establish universal improvement.
- The present SLSQP/PyTorch implementation is not reliably real-time at 20 Hz.
- Timing must be rerun for any deployment processor and should eventually include end-to-end loop latency.
- Final seeds `12026-12030` are now consumed holdout evidence and must not be used for future tuning.

These are disclosed scope limitations rather than hidden evidence inconsistencies. They do not invalidate the bounded simulation-study contribution.

## Submission checklist against the original audit

In this table, PASS means the current repository supports its deliberately bounded submission claim; a noted deployment limitation is not treated as an artifact failure.

| # | Original concern | Result | Post-fix evidence |
|---:|---|---|---|
| 1 | Train/test/data leakage | PASS | LSTM splits remain run-disjoint; reliability-used data is treated as development, and final seeds are fresh. |
| 2 | LSTM metric correctness | PASS | The full verifier reproduced saved one-step, persistence, normalization, and recursive metrics. |
| 3 | Genuine recursive prediction | PASS | Recursive forecasts feed predictions back; truth remains scoring-only. |
| 4 | Reliability calibration validity | PASS | Thresholds are frozen before final evaluation; earlier scenario work is development evidence, not final test evidence. |
| 5 | Virtual sensor substitution | PASS | Substitution is causal and protects history; recovery and single-sensor identifiability remain disclosed limitations. |
| 6 | MPC formulation/timing | PASS | First decision duration now equals the 50 ms plant hold. |
| 7 | Documented H and Nc | PASS | Canonical source/config/artifacts use `H=20`, `Nc=2`, `blocks=(5, 15)`. |
| 8 | Constraint enforcement | PASS | Bounds/slew clipping are active; the new H20 matrix has zero measured voltage and rate violations. |
| 9 | Warm start/move blocking | PASS | Plan shift is five samples and blocks are control-grid aligned; deterministic assertions cover both. |
| 10 | Safe optimizer fallback | PASS | Fallback is bounded/rate-limited and warm state resets; the final matrix recorded zero failures. |
| 11 | Sensor-fault improvement | PASS | Fresh raw runs reproduce `7.584044 -> 1.375062`, or `81.869014%`. |
| 12 | Adaptive disturbance result | PASS | Fresh raw runs reproduce `1.932806 -> 1.757076`, signed change `-9.091949%`. |
| 13 | Runtime evidence | PASS | Every final sample is saved; mean/median/p95/max reproduce with no unsupported baseline. |
| 14 | 20 Hz real-time conclusion | PASS | New H20 samples directly support the conservative conclusion that 20 Hz is not reliable. |
| 15 | Claims no stronger than evidence | PASS | Docs limit the contribution to simulated tested-fault robustness and explicitly reject hardware/universal-adaptive claims. |
| 16 | Combined-case weakness visible | PASS | Scenario dependence and the combined case are explicit in the summary and current docs. |
| 17 | Pipeline/status accuracy | PASS | Current status, configuration, counts, results, runtime scope, and limitations match canonical artifacts. |
| 18 | Clean-environment reproducibility | PASS | Exact dependency pins, clean notebooks, fixed config, verifier, and cache-free package provide the reconstruction path. |

Final result: 18 PASS, 0 FAIL. The repository is ready for submission as a reproducible simulation study, subject to the stated non-real-time and no-hardware limitations.
