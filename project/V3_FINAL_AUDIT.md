# V3 Final Independent Audit

Audit date: 2026-09-12  
Verdict: **READY WITH LIMITATIONS**

## Audit basis

This audit was performed independently after the corrective work. It inspected the implementation, frozen configuration and calibration provenance, preserved pre-fix snapshot, canonical V3 CSV/JSON/timing/quality evidence, generated V3 report and notebook, README, regression tests, and both verifiers.

Independent saved-evidence checks found:

- exactly 200 run rows: 5 controllers × 8 scenarios × 5 seeds;
- no duplicate or missing `(controller, scenario, seed)` keys;
- all 40 comparison rows reproduce exactly from `results/metrics/v3_final_holdout_runs.csv`, including sample standard deviations and medians;
- all source fractions sum to one within floating-point tolerance;
- zero optimizer, prediction, voltage, slew, and nonfinite failures;
- 23,240 unique V3 timing rows, with `extension_ms = aux_inference_ms + reliability_update_ms + arbitration_ms` for every row;
- `V3_DUAL_VIRTUAL_SENSOR_EXTENSION.md` exactly equals `scripts/generate_v3_report.py::build_report()`, and notebook 04e contains the same generated report;
- all eight focused tests in `tests/test_v3_corrective.py` pass against the implemented reliability classes;
- the stock `scripts/verify_v3_results.py` completed with `V3 VERIFIER RESULT: PASS`, and `scripts/verify_results.py` completed with `Saved-evidence verification passed.`

## Required questions

1. **Is V3 holdout truly unused? — Yes, to the limit that repository evidence can establish.** The exact seed tokens `29026–29030` are absent from all textual files in the 101-file pre-fix snapshot `results/baseline_v3_prefix/`, while reused seeds `19026–19030` are present there. The calibration/config files were frozen at 13:46 before the final run artifacts were written at 13:57. The final seed set appears only in the corrected final-evaluation path. There is no external immutable run registry, so prior unrecorded use cannot be proved impossible.

2. **Are thresholds reproducibly calibrated? — Yes.** `scripts/calibrate_v3_arbitration.py` uses only clean validation run IDs `[3, 9, 11, 16, 19, 23]`, records dataset/model/config hashes and sample counts, and excludes final seeds. `results/configs/v3_arbitration_calibration.json` records p90 agreement `4.3418674469`, p99.9 EWMA mismatch `8.7009588623`, p95 EWMA recovery `5.0486354952`, and p99.9 auxiliary recovery `9.0476932526` rad/s. Its SHA-256 is bound by both the frozen config and final summary, and the stock verifier independently recomputes the values.

3. **Is fallback semantics correct? — Yes.** `DualVirtualSensorArbitrator.update()` requires a finite `fallback_speed` and returns that value with source code 3; it cannot silently reuse `y_main`. `scripts/evaluate_v3_closed_loop.py` supplies the clipped last finite trusted physical speed. The separate PI path supplies voltage only when MPC fails and is never labeled as a speed source.

4. **Is auxiliary recovery mandatory when configured? — Yes.** `SensorReliabilityMonitor.update()` permits recovery only when `aux_residual` exists, is finite, and lies within `aux_recovery_gate`. Missing or NaN auxiliary evidence resets the healthy run. The regression test covers both cases.

5. **Can the parameter-mismatch latch recover safely? — Yes.** Recovery requires trusted physical sensing, instantaneous auxiliary consistency, EWMA below the lower threshold, and 50 consecutive clean samples. An independent actual-config sequence latched after a transient mismatch, cleared after 76 clean samples, and selected `AUX_VIRTUAL` during a later sensor fault. A permanent mismatch stayed latched and selected held fallback. Regression tests also cover transient, permanent, and interrupted recovery.

6. **Are V3 metrics reproducible? — Yes.** The saved matrix is complete and paired, finite fault windows use `[event_start, event_end)`, every aggregate in `v3_final_holdout_comparison.csv` reproduces from the canonical run CSV, and summary safety/failure totals agree. The evaluator is deterministic by `(scenario, seed)` and gives each controller the same base noise and injected-fault realization. Full numerical reruns require the pinned environment in `requirements.txt`.

7. **Are documentation tables generated from canonical artifacts? — Yes.** `scripts/generate_v3_report.py` reads the canonical run, comparison, quality, summary, calibration, and config artifacts. The checked-in V3 report exactly matches its generated output, notebook 04e mirrors that report, and the README V3 block uses the same saved artifacts. The active V2 development artifact/key now says `development_extension`; the historical filename remains only in the preserved snapshot.

8. **Is runtime overhead measured correctly? — Yes for the defined V3 extension.** `results/metrics/v3_runtime_overhead_samples.csv` contains 23,240 per-step samples and includes auxiliary inference, reliability update, and arbitration while excluding MPC solve time. The reported mean is `0.655378 ms`, and the eight-process CPU-contention context is explicitly labeled as non-real-time benchmarking. `control_compute_ms` measures PI/MPC computation after source selection and excludes the extension, so exact sensor-to-actuator end-to-end latency was not measured and must not be inferred.

9. **Are drift claims seed-robust? — Yes, because the corrected claim is that no material improvement is supported.** The maximum absolute paired C3-minus-plain difference is only `0.000572 rad/s`; the mean is slightly worse for C3 (`5.305418` versus `5.305297 rad/s`). The report makes no robust drift-improvement claim.

10. **Does V3 improve `combined_fault_load` consistently? — Yes within this holdout.** C3 fault-window RMSE is lower than plain MPC for all five paired seeds by `0.220598–0.670162 rad/s`. Mean C3 RMSE is `3.963268 rad/s`, versus `4.343202` for plain MPC and `11.775776` for C2. This is correctly presented as scenario-specific evidence, not universal dominance.

11. **Are parameter-variation limitations honestly reported? — Yes.** C3 mean fault-window RMSE is `0.669664 rad/s`, essentially equal to C1/C2 (`0.669758`) and slightly worse than plain MPC (`0.668845`). C3 selects no auxiliary feedback during any parameter-variation event; one seed uses held fallback for `0.003322` of the event. The report explicitly says parameter mismatch remains unresolved.

12. **Are all baseline V1/V2 artifacts preserved? — Yes.** `results/baseline_v3_prefix/` contains 101 files: the seven-file V1 baseline bundle, eight configs, 48 metric files, 21 plots, five raw artifacts, eight reports, both model weights, the V2 log, and the original verifier. Active stale V3 copies were removed only after preservation; the original V2 filename and pre-fix evidence remain recoverable in this snapshot.

## Remaining limitations

- Evidence is simulation-only; there is no hardware validation or deployment timing.
- The holdout non-use claim is supported by repository provenance, not an external immutable experiment registry.
- Slow drift is not materially improved, and plant-parameter mismatch remains unresolved.
- The auxiliary model assumes a trustworthy current measurement.
- Arbitration is deliberately not an oracle: in combined fault/load, selected-feedback RMSE (`2.105085 rad/s`) improves on main virtual RMSE (`2.209393`) but does not reach auxiliary-only RMSE (`1.700307`); in drift and noise it retains main priority despite a lower offline auxiliary error.
- Full end-to-end sensor-to-actuator latency was not captured; multi-process CPU timing must not be treated as a real-time guarantee.
