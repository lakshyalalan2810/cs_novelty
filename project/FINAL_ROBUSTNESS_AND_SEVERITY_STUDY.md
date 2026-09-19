# Final Robustness and Severity Study

This document is the frozen closeout report of the completed V3 robustness and
severity study. It is generated strictly from canonical saved evidence. No
controller, model, observer, calibration rule, MPC setting, training result,
severity definition, threshold, seed, or historical scientific artifact was
changed to produce it. Throughout, the paired controller effect is

`Delta = fault-window RMSE(C3_arbitration_MPC) - fault-window RMSE(B_plain_MPC)`,

so negative Delta favors C3 and positive Delta favors the plain LSTM-MPC
baseline B. The EKF comparator is denoted E.

## 1. Study purpose

The study defines the empirical operating boundary of the frozen V3
dual-virtual-sensor arbitration architecture instead of developing another
controller. Part A is a retrospective training-vs-simulation variance analysis
of the already completed 3-training-seed x 5-simulation-seed robustness matrix
(150 rows, 75 exact B/C3 pairs). Part B is a prospectively frozen severity
study using a new simulation-seed block, fixed model/calibration pairings,
fixed severity levels, fixed current-boundary cases, and fixed analysis rules.
Part B also includes a current-sensor boundary experiment because both the C3
auxiliary virtual sensor and the EKF comparator depend on armature current.

## 2. Frozen inputs/provenance

- Training seeds: `2026, 2027, 2028`. Each training seed is bound to its own
  main LSTM, auxiliary LSTM, sensor-reliability calibration, and V3
  arbitration calibration; the verifier confirms no cross-training-seed mixing
  in any of the 735 rows.
- Part A simulation seeds: `49026, 49027, 49028, 49029, 49030`.
- Part B simulation seeds: `59026, 59027, 59028, 59029, 59030`, frozen in
  `results/final_robustness/simulation_seed_freeze.json` before any Part B
  scientific run, with an empty pre-study provenance scan. Seeds
  39026-39030 were never used.
- Frozen protocol: `results/final_robustness/study_protocol.json`
  (SHA256 `0536abd9bf6aa2d69fcfeae13dcad5f5c238a4d7898552edff1e5d21fa205bad`).
- Pre-study lock: `results/final_robustness/prestudy_frozen_hash_manifest.json`
  covering 43 historical artifacts; the post-study rehash confirms 43/43
  byte-identical (see section 13).
- Analysis RNG seeds: Part A hierarchical bootstrap `20260917`, Part B
  hierarchical bootstrap `20260918`, each with 20,000 replicates, resampling
  training seed first and simulation seed second. Detection and false-entry
  probabilities use Wilson 95% intervals. Time samples are never treated as
  independent observations.

## 3. Experimental matrix

Part B contains exactly 735 labeled rows: 600 primary B/C3 rows
(5 sweep families x 4 conditions x 3 training seeds x 5 simulation seeds x 2
controllers) and 135 current-boundary rows (3 current cases x 3 training seeds
x 5 simulation seeds x 3 controllers). Recorded execution provenance shows 705
unique scientific executions plus 30 exact B-row reuses. The 30 reuses are all
`B_plain_MPC` rows: 15 current-dropout B rows reuse the matched
same-training-seed/same-simulation-seed `current_bias_5` B trajectory, and 15
simultaneous speed-plus-current B rows reuse the matched primary 5% speed-bias
B trajectory. B does not consume current online, so these trajectories are
scientifically identical; independent comparison confirms all 30 reused rows
match their sources exactly on every recorded science column. Every primary
family/condition/training-seed/simulation-seed/controller cell and every
current-case/training-seed/simulation-seed/controller cell is present exactly
once; there are no duplicate scientific keys and no missing combinations.

Primary severity grids are exactly: bias 2.5/5/10/15%; dropout 0.1/0.5/1.0/2.0
s from 2.0 s; drift final 2.5/5/10/15% ramped over 2.0-6.0 s; post-step load
0.06/0.10/0.15/0.20 N m (baseline 0.03 N m, step at 3.0 s); combined bias/load
cells (5%, 0.10), (5%, 0.15), (15%, 0.10), (15%, 0.15). Current-boundary cases
are `current_bias_5` (+0.25612 A, i.e. 5% of the frozen 5.12234 A reference
scale, over 2-4 s), `current_dropout_2s` (measured current forced to zero over
2-4 s), and `speed_bias_5_plus_current_bias_5` (both corruptions at once).

## 4. Training-vs-simulation variance decomposition

Part A decomposes the paired Delta surface descriptively as training-seed
effect + simulation-seed effect + residual interaction/nonadditivity. With one
observation per training x simulation cell, the residual includes interaction
and other nonadditive variation; these are descriptive variance proportions,
not inferential ANOVA claims.

| Scenario | Grand mean Delta | Training SS proportion | Simulation SS proportion | Residual | Training effect range | Simulation effect range | Favorable cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| combined fault + load | -0.568571 | 0.739 | 0.122 | 0.138 | 1.866112 | 0.825270 | 11/15 |
| 5% speed bias | -1.640732 | 0.137 | 0.555 | 0.309 | 0.776146 | 1.593809 | 13/15 |
| speed dropout | -19.221775 | 0.147 | 0.573 | 0.280 | 1.036455 | 2.029381 | 15/15 |
| load disturbance | +2.062685 | 0.334 | 0.543 | 0.123 | 2.005433 | 2.896723 | 0/15 |
| parameter variation | +2.690691 | 0.001 | 0.641 | 0.358 | 0.268144 | 6.963600 | 1/15 |

Full-precision canonical values for the combined-fault cell are grand mean
Delta `-0.568571303995` with training fraction 0.739239, simulation fraction
0.122369, and residual 0.138392. The combined-fault result is the only Part A
scenario in which training-seed variation is the largest component. Its
training-seed means differ in sign: `+0.310712` for seed 2026, `-1.555400`
for seed 2027, and `-0.461026` for seed 2028. In the other four scenarios,
simulation-seed variation is the largest component (54.3% to 64.1%).
Interpretation remains bounded: combined-fault behavior varies materially
with learned-model realization, and the hierarchical interval crosses zero
(see section 12).

## 5. Bias severity results

All four abrupt speed-bias severities are classified LIMITED. Detection
probability is identical across the four severities: pooled `11/15 = 0.7333`
(Wilson 95% `[0.4805, 0.8910]`), with training-seed probabilities `0.8`,
`0.4`, and `1.0` for seeds 2026, 2027, and 2028. The seed-2027 detection
probability prevents every bias level from meeting the SUPPORTED rule, which
requires per-training-seed detection of at least 0.8.

| Bias | B fault-window RMSE | C3 fault-window RMSE | Mean Delta | Hierarchical 95% CI | Favorable pairs | Classification |
|---|---:|---:|---:|---:|---:|---|
| 2.5% | 1.501679 | 0.977251 | -0.524428 | [-0.872836, -0.067268] | 13/15 | LIMITED |
| 5% | 2.861512 | 0.938292 | -1.923220 | [-2.262268, -1.476446] | 14/15 | LIMITED |
| 10% | 5.485865 | 0.938292 | -4.547573 | [-4.877131, -4.084986] | 15/15 | LIMITED |
| 15% | 7.939140 | 0.938292 | -7.000848 | [-7.333915, -6.508802] | 15/15 | LIMITED |

Mean detection latency when detected is `0.02 s`. No recovery event was
recorded in any bias run; the active state persisted to the 6 s horizon
(active duration 3.98 s from a typical 2.02 s entry). The defensible result is
therefore narrower than blanket bias tolerance: C3 shows favorable
fault-window tracking at every tested bias magnitude, but fault detection is
training-seed-sensitive and recovery is not demonstrated within the run
horizon.

## 6. Dropout-duration results

Dropout detection is pooled `11/15 = 0.7333` with seed-wise probabilities
`0.8`, `0.4`, and `1.0`, identical to the bias study. The `0.1 s` dropout is
UNSUPPORTED because its pooled mean effect is unfavorable (+0.458399) and only
one training seed has a favorable mean. The `0.5 s`, `1.0 s`, and `2.0 s`
dropouts are LIMITED.

| Dropout duration | B fault-window RMSE | C3 fault-window RMSE | Mean Delta | Hierarchical 95% CI | Favorable pairs | Classification |
|---|---:|---:|---:|---:|---:|---|
| 0.1 s | 0.629142 | 1.087541 | +0.458399 | [-0.138745, +1.425865] | 9/15 | UNSUPPORTED |
| 0.5 s | 4.074779 | 1.043695 | -3.031083 | [-4.225666, -1.411672] | 15/15 | LIMITED |
| 1.0 s | 10.379175 | 0.948127 | -9.431049 | [-10.551504, -7.893538] | 15/15 | LIMITED |
| 2.0 s | 20.446593 | 0.938292 | -19.508301 | [-20.282188, -18.551349] | 15/15 | LIMITED |

Mean detection latency when detected is `0.02 s`, and as in the bias runs no
recovery event occurs before the 6 s horizon ends. The data support a
duration-dependent tracking benefit for dropouts of at least 0.5 s in the
tested grid, with LIMITED rather than SUPPORTED status because detection
remains inconsistent across training seeds.

## 7. Drift results

No tested drift severity is supported. Detection is rare: pooled
probabilities are `0.0667` at 2.5%, 5%, and 10%, and `0.1333` at 15%. The
highest observed seed-specific probability is `0.2`.

| Final drift | B fault-window RMSE | C3 fault-window RMSE | Mean Delta | Hierarchical 95% CI | Pooled detection | Classification |
|---|---:|---:|---:|---:|---:|---|
| 2.5% | 0.869543 | 1.027034 | +0.157491 | [-0.064343, +0.577891] | 0.0667 | UNSUPPORTED |
| 5% | 1.708437 | 1.620794 | -0.087643 | [-0.496692, +0.348001] | 0.0667 | UNSUPPORTED |
| 10% | 3.411950 | 2.829120 | -0.582830 | [-1.571479, -0.000231] | 0.0667 | UNSUPPORTED |
| 15% | 5.120873 | 3.900363 | -1.220510 | [-2.771024, -0.000748] | 0.1333 | UNSUPPORTED |

The negative pooled effects at 10% and 15% do not establish drift tolerance
because the fault-detection endpoint is essentially absent. The current
reliability architecture does not provide demonstrated drift tolerance at any
tested ramp magnitude. This negative conclusion is stated without weakening.

## 8. Load-disturbance boundary

Load disturbance is not assigned SUPPORTED/LIMITED/UNSUPPORTED labels; its
endpoint is false sensor-fault entry under a healthy speed sensor.

| Post-step load | False-entry probability | Wilson 95% CI | Mean false-entry latency when present | Mean active duration when present | B fault-window RMSE | C3 fault-window RMSE | C3-B tracking penalty |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.06 N m | 0/15 = 0.0000 | [0.0000, 0.2039] | n/a | 0.000 s | 0.300673 | 0.782988 | +0.482315 |
| 0.10 N m | 0/15 = 0.0000 | [0.0000, 0.2039] | n/a | 0.000 s | 0.658601 | 1.185525 | +0.526924 |
| 0.15 N m | 4/15 = 0.2667 | [0.1090, 0.5195] | 0.300 s | 0.531 s | 1.134822 | 2.202871 | +1.068050 |
| 0.20 N m | 10/15 = 0.6667 | [0.4171, 0.8482] | 0.144 s | 1.717 s | 1.677077 | 4.760918 | +3.083841 |

The load-only tracking effect is unfavorable at every tested magnitude, with
hierarchical mean Delta positive at all four loads (descriptive 95% intervals:
0.06 `[-0.000385, +1.099539]`; 0.10 `[+0.000330, +1.243543]`; 0.15
`[+0.362960, +1.758602]`; 0.20 `[+1.647139, +4.184516]`). Increasing load
magnitude increases false sensor-fault entry probability and tracking
degradation. This is a demonstrated structural weakness of the frozen V3
reliability architecture.

## 9. Combined bias/load results

All four preregistered combined cells are LIMITED. Pooled detection
probability is `10/15 = 0.6667` (Wilson 95% `[0.4171, 0.8482]`) for every
cell, but training-seed probabilities are `0.8`, `0.2`, and `1.0`, so none
meets the SUPPORTED requirement for consistent detection across all three
training seeds.

| Bias + post-step load | Mean Delta | Hierarchical 95% CI | Favorable pairs | Classification |
|---|---:|---:|---:|---|
| 5% + 0.10 N m | -0.911594 | [-1.693786, -0.268149] | 14/15 | LIMITED |
| 5% + 0.15 N m | -0.721122 | [-1.617214, +0.026890] | 13/15 | LIMITED |
| 15% + 0.10 N m | -6.568228 | [-7.256240, -5.985766] | 15/15 | LIMITED |
| 15% + 0.15 N m | -6.300033 | [-7.248110, -5.559148] | 15/15 | LIMITED |

The 15% combined-fault cells are favorable in every paired
training/simulation realization and show the stronger, more consistent paired
advantage. The 5% cells are less robust (14/15 and 13/15 favorable), and the
5% + 0.15 N m bootstrap interval narrowly crosses zero. Benefits exist in
portions of the tested severity space, particularly at larger speed-bias
magnitude, but the effect is training-realization-sensitive and not
universally favorable; no combined condition is SUPPORTED.

## 10. Current-sensor boundary

The current-sensor experiment is outside the V3 single-speed-sensor-fault
support claim. Identical current corruption is applied to the C3 auxiliary
input and the EKF measurement, and the recorded equivalence mismatch count is
zero on all E rows. B does not use current online and functions as the
unaffected controller reference for current-only faults. The current reference
scale is the frozen dataset maximum absolute current, `5.12234 A`; the 5%
current-bias case uses `+0.25612 A` from 2 s to 4 s; the dropout case sets
measured current to zero over the same interval; the simultaneous case
combines the same 5% current bias with 5% speed bias.

| Current-boundary condition | B fault-window RMSE | C3 fault-window RMSE | EKF fault-window RMSE | C3-B Delta | C3-B 95% CI | EKF-B Delta | EKF-B 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5% current bias | 0.306050 | 1.787525 | 0.799369 | +1.481476 | [-0.008704, +3.580641] | +0.493320 | [-0.014803, +1.272700] |
| 2 s current dropout | 0.306050 | 2.778437 | 2.504214 | +2.472387 | [+0.364538, +5.505498] | +2.198165 | [-0.004074, +5.638666] |
| 5% speed bias + 5% current bias | 2.861512 | 4.560627 | 4.177304 | +1.699114 | [+1.425082, +2.016817] | +1.315792 | [+1.148934, +1.484524] |

Estimator degradation is substantial. Under current dropout, mean
event-window auxiliary-estimator RMSE is `20.0984 rad/s` and EKF estimator
RMSE is `23.0972 rad/s`. With current bias, the corresponding means are
`5.7159` and `4.4846 rad/s`; in the simultaneous case, `5.7097` and
`4.4846 rad/s`. Source-selection behavior confirms the shared dependence on
current: under simultaneous corruption C3 uses the auxiliary source for
`98.77%` of the event window while the EKF controller substitutes EKF
feedback for `100%`; under current dropout the values are `45.7%` and
`19.9%`; under current bias, `33.2%` and `13.3%`. Neither C3 nor EKF supports
current-sensor-fault-tolerance claims. The correct statement is that the
auxiliary learned sensor and EKF are independent of the speed-sensor channel
under an assumed healthy current measurement channel.

## 11. EKF current-boundary comparison

The physics-based EKF remains a stable classical comparator: zero EKF
numerical failures occurred across the whole study. Under current-sensor
corruption it shows the same qualitative dependence as C3. Mean EKF-B
fault-window Delta is `+0.493320` for current bias (95% CI `[-0.014803,
+1.272700]`), `+2.198165` for current dropout (95% CI `[-0.004074,
+5.638666]`), and `+1.315792` for simultaneous speed-plus-current bias (95%
CI `[+1.148934, +1.484524]`, clearly positive). EKF event-window estimator
RMSE reaches `23.10 rad/s` under current dropout. The issue is
estimation/control degradation rather than numerical breakdown. Nominal model
mismatch limits the EKF's disturbance and parameter robustness, consistent
with its BASELINE_ONLY role: it is a valid classical comparator, not a
current-fault-tolerant alternative.

## 12. Hierarchical uncertainty

Part A used hierarchical bootstrap RNG seed `20260917` with 20,000
replicates; descriptive 95% intervals are combined fault/load `[-1.517776,
+0.286116]`, 5% bias `[-2.191698, -1.011005]`, dropout `[-19.957548,
-18.532747]`, load disturbance `[+0.951946, +3.201546]`, and parameter
variation `[+0.942957, +4.485360]`. Part B used hierarchical bootstrap RNG
seed `20260918` with 20,000 replicates; per-condition intervals are reported
in sections 5-11. Because only three training seeds exist, these intervals
describe uncertainty conditional on the observed model realizations and do not
estimate training-population uncertainty precisely.

The bootstrap confirms clearly negative paired RMSE effects for large bias,
dropout durations of at least 0.5 s, and the 15% combined-fault cells. It also
shows why pooled RMSE alone cannot replace the classification rules: drift
10% and 15% have negative descriptive intervals but almost no detection, while
the 5% + 0.15 N m combined cell has a favorable pooled mean but an interval
extending slightly above zero. On the current boundary, C3 degradation is
clearly positive for current dropout and simultaneous corruption, and EKF
degradation is clearly positive for the simultaneous case; the current-bias
C3/EKF intervals and the current-dropout EKF interval include values just
below zero, reflecting the small three-training-seed sample.

## 13. Safety/failure evidence

All 735 labeled Part B rows completed. Recorded totals across all families:

- optimizer failures: `0`
- main-model prediction failures: `0`
- auxiliary-model prediction failures: `0`
- EKF numerical failures: `0`
- nonfinite events: `0`
- voltage violations: `0`
- slew/rate violations: `0`
- incomplete runs: `0`
- current-corruption equivalence mismatches: `0`

Thus the study observed zero safety, numerical, and incomplete-run failures.
This establishes numerical and execution integrity for the tested study
matrix; it does not by itself establish hardware safety certification or
safety outside the tested operating domain. The frozen-artifact integrity
result is 43/43 matching prestudy hashes, and the Part B analysis
independently verifies exact model/calibration bindings in the completed
matrix.

## 14. Empirical operating envelope

No tested severity level is classified SUPPORTED. The envelope below is
reconstructed from the preregistered rules (SUPPORTED: per-training-seed
detection >= 0.8, all three training-seed mean Deltas negative, at least 12/15
favorable pairs, zero safety/numerical/incomplete failures; LIMITED: pooled
detection >= 0.5, pooled mean Delta negative, at least 2/3 favorable
training-seed means, zero failures; otherwise UNSUPPORTED).

| Family | Condition | Classification / reporting rule |
|---|---|---|
| abrupt speed bias | 2.5% | LIMITED |
| abrupt speed bias | 5% | LIMITED |
| abrupt speed bias | 10% | LIMITED |
| abrupt speed bias | 15% | LIMITED |
| speed dropout | 0.1 s | UNSUPPORTED |
| speed dropout | 0.5 s | LIMITED |
| speed dropout | 1.0 s | LIMITED |
| speed dropout | 2.0 s | LIMITED |
| speed drift | 2.5% | UNSUPPORTED |
| speed drift | 5% | UNSUPPORTED |
| speed drift | 10% | UNSUPPORTED |
| speed drift | 15% | UNSUPPORTED |
| combined bias + load | 5% + 0.10 N m | LIMITED |
| combined bias + load | 5% + 0.15 N m | LIMITED |
| combined bias + load | 15% + 0.10 N m | LIMITED |
| combined bias + load | 15% + 0.15 N m | LIMITED |
| load disturbance | 0.06/0.10/0.15/0.20 N m | report false-entry risk only |
| current-sensor corruption | all three cases | outside the V3 single-fault support claim |

The principal reason no bias, dropout, or combined cell reaches SUPPORTED is
detector inconsistency across training seeds: for abrupt bias and dropout,
seed 2027 detects only 2/5 cases per condition; for the combined grid, seed
2027 detects only 1/5 cases. Zero safety failures satisfy the safety part of
every rule but cannot compensate for inconsistent detection.

## 15. Claims supported

The following factual claims are supported by the completed evidence (this
list concerns demonstrated facts; no tested severity carries a SUPPORTED
envelope classification):

1. Across three independently trained model pairs and five simulation seeds
   per pair (seeds 59026-59030), C3 fault-window tracking is favorable
   relative to B at every tested abrupt bias (13/15 to 15/15 favorable
   paired cells) and every tested dropout of at least 0.5 s (15/15 each).
2. Healthy-sensor load disturbance produces a measured false-entry curve of
   0% at 0.06 and 0.10 N m, 26.7% at 0.15 N m, and 66.7% at 0.20 N m, with
   tracking penalty rising from +0.48 to +3.08 RMSE.
3. Part A variance attribution is scenario-dependent: combined-fault variation
   is descriptively training-seed dominated (73.9%), while the other four
   scenarios are simulation-seed dominated (54.3%-64.1%).
4. C3 and the EKF comparator both depend materially on the current channel;
   current dropout and simultaneous speed-plus-current corruption degrade
   closed-loop tracking and estimator quality.
5. The complete Part B matrix contains 735 labeled rows (705 unique executions
   plus 30 exact B-row reuses) with zero numerical, safety,
   current-equivalence, or incomplete-run failures and 43/43 prestudy hash
   matches.
6. The EKF comparator is numerically stable throughout (zero numerical
   failures) while showing disturbance/parameter sensitivity consistent with
   nominal model mismatch.

## 16. Claims limited

The following hold only in the LIMITED sense of the preregistered envelope
(favorable pooled tracking plus adequate pooled detection, but inconsistent
detection across training seeds):

1. Abrupt speed-bias tolerance at 2.5%, 5%, 10%, and 15%: favorable paired
   tracking at every level, but seed-2027 detection is only 40%.
2. Dropout tolerance at 0.5 s, 1.0 s, and 2.0 s: 15/15 favorable paired cells
   per duration, but seed-2027 detection is only 40%.
3. Combined bias-plus-load behavior in all four tested cells: 13/15 to 15/15
   favorable paired cells with the strongest effect at 15% bias, but
   seed-2027 detection is only 20%.

V3 often provides favorable paired tracking performance across tested
training seeds for abrupt bias/dropout, but detector reliability varies with
model realization, so the tested conditions are LIMITED rather than SUPPORTED
under the preregistered envelope definition.

## 17. Claims unsupported

The evidence does not support the following; the paper must not state them:

1. Any SUPPORTED operating point at any tested severity (global result: none).
2. Tolerance of 0.1 s dropout (UNSUPPORTED: positive pooled Delta, one
   favorable training-seed mean).
3. Tolerance of drift at any tested ramp magnitude (all four levels
   UNSUPPORTED; detection 6.7%-13.3%).
4. Immunity to load disturbances (false-entry risk and tracking penalty rise
   with load magnitude).
5. Tolerance of current-sensor faults by C3 or EKF, or any
   more-than-one-sensor-channel fault-tolerance statement.
6. Reliable post-fault recovery for abrupt bias/dropout (zero recovery events
   before the 6 s horizon ends).
7. Training-seed invariance of detector behavior.
8. Guaranteed performance outside the tested seed/severity grid, hardware
   robustness, timing/feasibility statements, or certified stability
   statements.

The favorable drift RMSE at 10%/15% must not be described as successful drift
handling (detection is nearly absent); load-only C3 behavior must not be
described as beneficial (the penalty is positive at every tested load); the
combined grid must not be described as SUPPORTED (seed-2027 detection is
20%).

## 18. Paper implications

The manuscript must present a bounded envelope, not a general tolerance
result. Abrupt speed-bias (2.5%-15%) and dropout (0.5-2.0 s) may be reported
as favorable-but-LIMITED tracking results with explicit training-seed
detection caveats. The 0.1 s dropout, all drift levels, and load immunity
must be reported as negative results. The combined-fault section must
distinguish the stronger 15%-bias evidence from the weaker 5%-bias evidence.
The current-sensor section must be framed as a dependency boundary: the
auxiliary learned sensor and EKF are independent of the speed-sensor channel
under an assumed healthy current measurement channel, and the paper must not
extend this to current-channel faults. The EKF must be presented as a stable
classical comparator with known model-mismatch limits. All envelope
statements must cite the preregistered rules and the three-training-seed
sample size.

## 19. Explicit limitations

Only three training seeds are available. The hierarchical bootstrap improves
Monte Carlo stability conditional on these three trained models but does not
create additional independent evidence about training-initialization
variability. Part A cannot separately estimate training x simulation
interaction because there is one observation per crossed cell. The study is
simulation-only with a 6 s horizon, fixed plant/reference/noise conventions,
fixed severity grids, and frozen V3/EKF implementations; the absence of
failures applies only to this tested matrix. Recovery is not demonstrated for
abrupt bias or dropout because no recovery event occurs before runs end. Drift
was evaluated with the preregistered ramp convention and results must not be
generalized to every drift waveform. The load study reports false-entry risk
rather than sensor-fault support. Current corruption is a boundary test of a
shared dependency, not an independently designed current-fault-tolerant
architecture. B is unaffected by current-only faults (it does not consume
current online), so it is a valid controller reference for degradation but
not a current-estimator comparator. No inference should be made about untested
magnitudes, untested fault combinations, hardware timing, unmodeled
nonlinearities, other motors, other noise distributions, or additional
training seeds.

## 20. Final study verdict

The completed evidence is sufficient to state the present architecture's
supported and unsupported boundaries and proceed to paper writing. There is
no scientific requirement for an additional architecture, retraining,
recalibration, threshold change, or new scenario before publication. The
measured boundary is: LIMITED abrupt-bias and dropout (at or above 0.5 s)
tracking with training-sensitive detection; UNSUPPORTED short dropout and
drift; a load-magnitude-dependent false-entry weakness; LIMITED combined
bias/load cells favoring larger bias; a current-channel dependency boundary
for both C3 and EKF; and zero safety/numerical failures across 735 labeled
rows with 43/43 historical hashes intact. The study closes with a bounded,
negative-result-preserving envelope that the manuscript must respect.
