# Independent Bug-Fix Audit

Date: 2026-09-07  
Scope: recovery-semantics fix and its saved evidence  
Auditor constraint: no implementation or result artifact was edited during this audit; only this report was created.

## Executive result

The requested behavior is correct in the current source and the saved headline metrics are internally consistent. The monitor judges the raw physical measurement, never substituted feedback; recovery is gated by finite instantaneous and CUSUM evidence for the configured persistent run; CUSUM resets only after that run completes. The finite injection and scoring windows are both `[start, end)`, and `y_true` is used for virtual-feedback quality only after a simulation completes.

The verdict is qualified for two evidence limitations, neither of which changes controller decisions or reported headline metrics:

1. All 30 finite bias/dropout virtual-quality rows have right-censored/blank recovery latency at the 6 s horizon. The conservative latch therefore prevents false recovery but the final closed-loop study does not demonstrate recovery within its observation window.
2. The packaged project has no usable version-control history, so “MPC source remained untouched” cannot be proven historically. Its frozen MPC configuration is byte-identical to the preserved pre-fix baseline, the current timing/blocking implementation satisfies its invariants, and its executable self-check passes.

## Required checks

| # | Requirement | Result | Evidence |
|---|---|---|---|
| 1 | Sensor health uses the raw physical measurement | PASS | `notebooks/06_final_evaluation.ipynb` computes `sensor.update(raw_y_measured - y_hat)` before choosing feedback (JSON source lines 237-244). `SensorReliabilityMonitor.update` documents the same contract at `src/reliability.py:39-56`. Replay of all 12 saved C/D sensor-scenario representative traces reproduced every monitor state. |
| 2 | Substituted feedback is not mistaken for recovery | PASS | The notebook computes health from `raw_y_measured - y_hat`; only afterward does it assign `y_feedback = y_hat if reliability["substitute"] else raw_y_measured`, then stores that feedback in history (JSON source lines 237-244 and 231-244 in the extracted code). No recovery decision reads `y_feedback`. |
| 3 | Recovery requires persistent instantaneous + CUSUM evidence | PASS | `src/reliability.py:73-83` requires a finite residual, `abs(residual) <= residual_gate`, and `score <= threshold`, increments only consecutive candidates, and exits at `exit_count`. Frozen `exit_count` is 5. In-memory regression runs produced no false exit: persistent bias and combined bias/load remained active; temporary bias and dropout exited only after the fault ended. |
| 4 | CUSUM reset timing | PASS | `src/reliability.py:80-83` resets both accumulators only inside the completed persistent-recovery branch. Returned accumulator/healthy-run telemetry reflects that reset on the exit sample, consistently with the saved traces. The verifier also proves that five nominal samples alone do not clear a large CUSUM and that reset occurs only after eventual full recovery (`scripts/verify_results.py:287-300`). |
| 5 | Finite injection and scoring masks are end-exclusive | PASS | All four finite injections use `2.0 <= instant < 4.0` (`notebooks/06_final_evaluation.ipynb` JSON lines 149-159); scoring uses `TIME >= event_start` followed by `event_mask &= TIME < event_end` (JSON lines 381-415). The 0.01 s grid therefore contains exactly 200 scored samples, not 201. |
| 6 | Virtual-feedback metrics use `y_true` offline only | PASS | Online monitor/control logic ends before the final-matrix post-processing block. Only that offline block masks saved `virtual`, `measured`, and `true` arrays to compute RMSE/MAE (`notebooks/06_final_evaluation.ipynb` JSON lines 2671-2686). Recalculation from the retained seed-12026 representative traces matched every corresponding CSV value exactly (maximum absolute difference 0). |
| 7 | Combined pre/post comparison is reproducible | PASS | The preserved and current five-seed run CSVs independently give D-adaptive combined fault-window RMSE `3.717753344555` pre-fix and `11.715164332701` post-fix. The representative comparison CSV independently shows the seed-12026 false-recovery count changing from 1 to 0 and RMSE from `3.691747994721` to `11.688699950612`. The legacy trace replay reproduced the false exit at 3.24 s; current logic remained latched at that instant. |
| 8 | Final metrics match documentation | PASS | Direct aggregation of 220 run rows reproduced plain/reliable sensor RMSE `7.584044274508`/`1.374103919223` and `81.881646922324%` improvement; plain/adaptive disturbance RMSE `1.932805769037`/`4.709976323643` and `143.685961574409%` degradation. `README.md`, `PROJECT_PIPELINE_AND_STATUS.md`, `FINAL_REPO_AUDIT_POSTFIX.md`, and `final_summary.json` agree to six decimals. The verifier passed. |
| 9 | MPC timing/blocking remains correct | PASS WITH LIMITATION | `src/mpc.py:28-50` rejects blocks that split a 5-sample held interval; `src/mpc.py:169-173` creates multirate blocks; `src/mpc.py:211-217` warm-shifts by five samples; `src/mpc.py:229-252` expands decisions and enforces slew constraints. The final schedule is H=20, blocks `(5,15)`, stride 5; the adaptive high schedule is `(5,10)`. The config SHA-256 is identical to the preserved baseline. `src/mpc.py` self-check passed, including rejection of `(1,19)`. Historical source immutability cannot be proven without VCS metadata. |

## Regression, fallback, drift, and runtime evidence

### Recovery regression

`scripts/validate_recovery.py` explicitly models the legacy instantaneous-only exit and four fixed-monitor cases. I exercised its functions in memory to avoid rewriting its CSV:

- persistent constant bias: no exit, active at end;
- temporary bias: exit at sample 65, no exit during physical fault;
- temporary dropout: exit at sample 198, no exit during physical fault;
- combined persistent bias/load: no exit, active at end;
- preserved combined trace: legacy exit at 3.24 s; fixed monitor active at 3.24 s.

The refreshed `recovery_regression_trace.csv` contains 1,200 rows (300 per case) and no false exits. For the two temporary cases, recorded `healthy_run` reaches 4 and returns to 0 on the fifth qualifying/exit sample, matching the monitor’s reset-state telemetry contract.

The regression is appropriately conservative but does not claim fast recovery. In the actual final evaluation, all 30 C/D rows for finite 5%, 15%, and dropout faults have blank recovery latency; substitution durations range from 4.01 to 5.38 s. This is consistent with the documentation’s “conservative” recovery limitation and is why the audit is not an unqualified PASS.

### PI fallback

The final notebook forces ten consecutive MPC failures, uses the bounded/rate-limited PI result as `fallback_voltage`, and checks resumption on the next update. Saved validation reports:

- 10/10 forced fallback updates;
- maximum absolute speed `38.975634 rad/s` (below the 120 rad/s safety bound);
- zero voltage and slew violations;
- successful MPC resumption;
- resume jump `0.093681 V` against a `2.0 V` limit.

`scripts/verify_results.py` independently checks these fields.

### Drift and combined behavior

Drift remains a disclosed weak case. Five-seed fault-window RMSE is `5.065854` for plain MPC, `5.065946` for sensor-aware MPC, and `5.074223` for adaptive MPC; substitution is only about `0.004659`. The fix therefore neither fabricates a drift gain nor materially changes the representative drift RMSE.

The combined case correctly stays latched (zero recovery and false-recovery events across the five post-fix runs), but long virtual-feedback use under plant mismatch raises D-adaptive fault-window RMSE to `11.715164`. That adverse result is prominently documented rather than hidden.

### Runtime distribution

Direct calculation from 19,965 unique per-solve rows reproduced:

| Statistic | Value |
|---|---:|
| Mean | 67.963908 ms |
| Median | 60.554900 ms |
| P95 | 135.782580 ms |
| P99 | 166.619316 ms |
| Maximum | 666.868900 ms |
| Over 50 ms | 58.442274% |
| Over 100 ms | 23.060356% |

All rows are finite/nonnegative, lie on five-sample control instants, and report optimizer success. These are solver/control-computation timings, not end-to-end latency; the documented conclusion that 20 Hz real time is not established is correct.

## Executed checks

- `C:\Users\Lakshya\anaconda3\envs\ml_env\python.exe scripts\verify_results.py` — exit 0, saved-evidence verification passed.
- `C:\Users\Lakshya\anaconda3\envs\ml_env\python.exe src\reliability.py` — exit 0.
- `C:\Users\Lakshya\anaconda3\envs\ml_env\python.exe src\mpc.py` — exit 0.
- Independent in-memory CSV/NPZ aggregation, monitor replay, virtual-quality recomputation, mask count, pre/post comparison, and regression-function exercise — passed.

## Final verdict

PASS WITH LIMITATIONS
