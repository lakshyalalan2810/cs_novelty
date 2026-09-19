# V4 confirmatory preregistration: reliability-aware LSTM-MPC with dual witnesses

Status: FROZEN BEFORE ANY V4 CONFIRMATORY RUN. This document, the seed
manifest (`results/v4/prereg/seed_manifest.json`, sha256
`13f607fa71936de028105432c82bf673c9b43de1db2f5beb3ad6e1e803c7e28f`),
and the scripts named below were completed before any confirmatory seed
was executed. See "Preregistration timing" for the smoke-test disclosure.

## 1. Objectives

V4 tests whether detector fixes motivated by the frozen post-hoc analysis
(startup false-latching, unbounded CUSUM discharge, load-induced false
entry) improve sensor-fault tolerance of LSTM-MPC closed-loop DC-motor
control on fresh seeds, fresh models, and fresh data, against frozen
(C3) and classical (E2/S1/S2/S3) references. The study is framed as a
rigorous evaluation of learned residual-based fault tolerance, not as a
claim of general superiority.

## 2. Controllers under test

Implemented in `scripts/v4_closed_loop.py`; MPC law and PI fallback are
the frozen configuration for every MPC-based controller.

- `B`: plain LSTM-MPC on measured speed (no detector; tracking anchor).
- `C3`: frozen V3 stack (SensorReliabilityMonitor +
  DualVirtualSensorArbitrator with frozen per-pair calibrations; training
  seeds 2026-2028 only; legacy reference).
- `V4_full_aux` / `V4_full_ekf`: ReliabilityMonitorV4 with the `full`
  factorial flags (CUSUM clamp k=2.0, 150-sample warmup with sensor trust,
  witness AND-rule) and direct main-LSTM substitution; witness residual
  from the aux LSTM (`_aux`) or frozen EKF (`_ekf`). Witness gate is the
  pair's calibrated V4 witness gate (placeholder replaced at build time).
- `V4_frozen_baseline_aux`: frozen-structure monitor on V4-trained models
  (structural control isolating new data/models from the new detector).
- `E2`: frozen EKF + own CUSUM speed-residual detector + EKF substitution.
- `S1`: fixed gate on |y_meas - EKF omega| + EKF substitution.
- `S2`: model-free median/rate plausibility + median substitution.
- `S3`: back-EMF Luenberger observer + own CUSUM detector + observer
  substitution.
- `PI`: PI on measured speed (mechanics anchor; no detector).

Primary V4 controllers: `V4_full_aux`, `V4_full_ekf`. Remaining factorial
cells (`clamp_only`, `gating_only`, `clamp_gating`, `witness_only`) are
EXPLORATORY (same engine, same seeds, labeled as such).

## 3. Data, models, calibration (frozen procedures)

- Dataset: `scripts/v4_generate_dataset.py`, 120 healthy trajectories
  (startup-from-rest / closed-loop-MPC / closed-loop-PI / open-multisine /
  open-random-step), seed 62026, whole-trajectory split 72/24/24,
  normalization from training windows only. Optional current
  noise/quantization default OFF for the confirmatory dataset.
- Training: `scripts/v4_train_models.py`, seeds 2026-2036 (11 pairs),
  frozen architectures/features/losses/RNG contracts, max epochs raised to
  200/200 (main/aux) with frozen patience 8/10. `best_epoch`, stop reason,
  and the `converged` flag (best_epoch < max_epochs) are logged per run.
- Sensor calibration: `scripts/v4_calibrate.py` on the 24 V4 validation
  trajectories (frozen p99.9/CUSUM-grid rules + aux/witness gates, 1000
  trajectory-bootstrap replicates, seed 20260920 + training seed).
- Baseline calibration: `scripts/v4_calibrate_baselines.py` (single
  seed-free document, 1000 replicates, seed 20260921 + statistic index).
- No test trajectory, fault scenario, or closed-loop result is used in
  training or calibration. E2/S1/EKF-witness reuse the frozen EKF
  calibration (frozen train/val evidence, not V4 test data).

## 4. Confirmatory hypotheses (family alpha = 0.05, Holm)

Evaluated by `scripts/v4_confirm_analysis.py`. All comparisons are paired;
positive delta supports the hypothesis. Hierarchy: training seed, then
simulation seed (20000 bootstrap replicates, seed 20260930 + hypothesis
index, two-sided 95% CI, Cohen's dz, Holm step-down).

- H1: `V4_full_aux` false-latch probability < `C3` (fault-free, all 4
  references pooled; paired by train/sim/reference).
- H2: `V4_full_ekf` false-latch probability < `C3` (same design).
- H3: `V4_full_aux` detection probability > `C3` at bias 2 sigma,
  clean-start pairs only (paired by train/sim).
- H4: `V4_full_ekf` detection probability > `C3` at bias 2 sigma,
  clean-start pairs only.
- H5: `V4_full_aux` recovery probability > `C3` on finite bias 8 sigma
  (10 s horizon), clean-start pairs only.
- H6: `V4_full_aux` fault-window RMSE < `B` at bias 8 sigma
  (intention-to-treat, all pairs).
- H7: `V4_full_aux` fault-free tracking penalty vs `B` < `C3` penalty vs
  `B` (intention-to-treat, all pairs/references).
- H8: `V4_full_aux` load false-entry probability < `C3` at 0.15 N m
  post-step, clean-start pairs only.
- H9: `V4_full_ekf` load false-entry probability < `C3` at 0.15 N m
  post-step, clean-start pairs only.

## 5. Endpoints and metric definitions

Separation of false latch from missed detection (the frozen conflation):

- False latch (fault-free runs): `reliability_entries > 0`. Time to false
  latch: `first_entry_time_s`.
- Pre-event latch (fault runs): `pre_event_reliability_entries > 0`, i.e.
  any entry strictly before `fault_onset_s`.
- Conditional detection: `sensor_fault_detected` (post-onset entry)
  evaluated on clean-start pairs; missed detection is its complement.
- Pair-exclusion rule (H3/H4/H5/H8/H9): drop (training, sim) pairs where
  EITHER controller pre-latched; report excluded counts alongside every
  conditional endpoint.
- Intention-to-treat (H6/H7, all RMSE/tracking endpoints): all pairs,
  including pre-latched runs; no exclusions, no redefinition.
- Detection latency: first post-onset entry minus onset (reported
  conditionally on detection; no hypothesis test).
- Recovery: first inactive transition at/after fault end;
  right-censored at the horizon. Recovery probability and time to release.
- Minimum detectable magnitude: log2-interpolated bias where detection
  reaches 0.9 (secondary/descriptive; flags below/above grid).
- DET points: detection rate (bias 2/8 sigma, 20-seed subset) vs
  false-entry rate (matched fault-free cells) at threshold scales
  0.5/0.75/1.0/1.5/2.0 (secondary/descriptive).
- Tracking penalty: controller overall RMSE minus `B` overall RMSE on the
  same (train, sim, reference) cell.
- Timing: per-solve ms distribution (warm-up excluded), control-period
  exceedance rate (descriptive, machine-recorded).

Run classification rule (confirmatory): a fault run is `PRE_EVENT_LATCH`
if any entry precedes onset, else `DETECTED` if any post-onset entry
exists, else `MISSED`. Fault-free runs are `FALSE_LATCH` iff any entry
exists. V4 envelope (descriptive): frozen SUPPORTED/LIMITED/UNSUPPORTED
rule structure with conditional detection substituted for pooled
detection and intention-to-treat deltas; thresholds unchanged.

## 6. Protocols and grids (scripts only until the confirmatory runs)

- (a) Fault-free Monte Carlo: `v4_protocol_faultfree.py`, 200 sim seeds
  x 11 training seeds x 4 references x 9 controllers = 72,800 cells.
- (b) Small-magnitude sweep: `v4_protocol_sweep.py`, 50 severity seeds x
  22 fault cells (6 bias sigma / 5 dropout lengths / 5 drift rates /
  4 load levels / 2 combined) + DET subset = 107,200 cells.
- (c) Recovery: `v4_protocol_recovery.py`, 100 severity seeds x 3 finite
  faults x 10 s horizon = 24,000 cells.
- (d) Robustness OFAT: `v4_protocol_robustness.py`, 25 severity seeds x
  16 corners x 2 (bias-8-sigma + fault-free) = 64,000 cells.
- (e) Timing: `v4_protocol_timing.py`, single-worker, strides 5/10,
  910 cells (descriptive).
- Confirmatory analysis: `v4_confirm_analysis.py` (H1-H9, Holm).

Total: ~268,910 closed-loop runs. Every protocol supports `--dry-run`
(plan counts only) and `--smoke` (PI-only mechanics on smoke seeds).

## 7. Seed manifest

`results/v4/prereg/seed_manifest.json` (sha256 recorded at the top of
this document; detached signature in `seed_manifest.sha256`):

- Training seeds 2026-2036 (2026-2028 retrained on NEW data; 2029-2036
  new). Dataset seed 62026.
- Fault-free sim seeds 71000-71999 (200). Severity sim seeds 72000-72099
  (100) with prefixes: sweep 72000-72049, recovery full block,
  robustness 72000-72024, DET 72000-72019, timing 72000-72004.
- Analysis RNG: confirmatory bootstrap base 20260930, sensor-calibration
  bootstrap 20260920, baseline-calibration bootstrap 20260921.
- Excluded: previously used sim blocks (12026-12030, 19026-19030,
  29026-29030, 49026-49030, 59026-59030), forbidden 39026-39030,
  external exploratory scan 91000-91059, smoke seeds 99001/99101/99102.
- `v4_confirm_analysis.py` refuses out-of-block sim seeds.

## 8. Deviation policy and labeling

- Any departure from this document (grid, seed, metric, exclusion,
  stopping rule) is reported as a DEVIATION with rationale; affected
  analyses are labeled EXPLORATORY/POST-HOC and excluded from the Holm
  family (no silent redefinition of endpoints to improve labels).
- CONFIRMATORY: H1-H9 as specified above, run once on the frozen grids.
- EXPLORATORY: remaining factorial cells, DET/MDM details, drift/combined
  magnitudes, timing, and any follow-up not listed in Section 4.
- Pre-registered metric definitions are never redefined after seeing
  results; a metric that proves uninformative is reported as such.

## 9. Retained negative results

The following frozen negative results are retained verbatim and are NOT
overturned by design; V4 re-measures them on fresh seeds:

- Gradual drift is hard to detect (frozen pooled 4/15); V4 sweep includes
  drift rates as a secondary endpoint with no superiority hypothesis.
- Load disturbances cause false reliability entries under the frozen
  detector; H8/H9 test reduction, but the frozen finding stands for C3.
- Current-channel corruption degrades the aux witness and V3 arbitration
  (Part B current matrix); V4 robustness re-measures current noise and
  quantization effects on every controller.
- The combined-bias-plus-load frozen outcome is training-seed-sensitive;
  V4 combined cells are secondary/descriptive.
- V3 arbitration was validated only for single-speed-sensor faults;
  simultaneous multi-channel corruption remains out of scope.

## 10. Preregistration timing and smoke disclosure

This document and all scripts above were written before any
confirmatory-block seed was executed. During preparation, only the
following ran (all outputs discarded to pytest tmp paths, no thresholds
or designs tuned, no confirmatory seed touched):

- Dataset/training/calibration `--smoke` runs on smoke dataset seed 99001
  (6 trajectories) and a discarded 2-epoch training smoke on seed 2029.
- PI-only engine mechanics checks on smoke sim seeds 99101/99102.
- Unit, regression, and dry-run plan tests (no simulation of record).
- The post-hoc Phase 2 decomposition, which reads frozen CSVs only.

Frozen evidence is untouched: the 43 prestudy artifacts and all
`*_frozen_hashes*.json` contents verify byte-identical (see
`scripts/run_all_verifiers.py`). The one-line pandas-3 verifier repair is
recorded in `V4_BOOL_SERIES_REPAIR_NOTE.md` with behavior-only scope.

