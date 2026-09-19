# Next Research Architecture Plan

**Decision date:** 2026-09-15  
**Decision:** Stop three-distance attribution work. Preserve V3 as the strongest frozen architecture. Build one minimum classical observer comparator next, then harden the paper with training-seed and fault-severity evidence.  
**Scope of this document:** retrospective read/analysis/plan only. No controller was tuned, no model was trained, no architecture was implemented, no frozen artifact was changed, C4-v1 was not rerun, and seeds 39026–39030 were not inspected or used.

The numerical diagnostics below were computed in memory from existing CSV/JSON/NPZ/model artifacts. They are **NON-CANONICAL retrospective diagnostics**, not new final evidence or classifier results.

## 1. Current project state

The repository contains three relevant research stages:

| Stage | Frozen status | What the evidence establishes | Present decision |
|---|---|---|---|
| V3 | Frozen strongest architecture | Complete 275-run matrix: 5 controllers × 11 scenarios × 5 evaluation seeds; frozen calibration; exact artifact hashes; independent verifier passes | Preserve unchanged |
| C2 | Frozen negative ablation | C2 is behaviorally identical to C1; the auxiliary recovery gate was uniquely binding zero times | Closed result; do not revisit |
| C4-v1 | Frozen negative development result | Complete 105-run paired development matrix; C4 is behaviorally identical to C3 and fails its predeclared gate | NO-GO; do not retune or run final seeds |

The implemented plant is a nonlinear permanent-magnet DC motor with state `[armature current, angular speed]`, smooth Coulomb friction, a 10 ms plant step, and a 12 V actuator limit. The main learned model is a 50,753-parameter, two-layer residual LSTM with a 20-sample `[voltage, measured speed]` history. The auxiliary model is a 4,641-parameter, one-layer LSTM using only a 20-sample `[voltage, armature current]` history. Both frozen models were trained once with seed 2026 on nominal simulated data.

V3 uses the physical speed while trusted. When the residual/CUSUM monitor substitutes, it selects the main LSTM while main and auxiliary estimates agree; it can switch to the auxiliary estimate when the main/auxiliary disagreement is large and the auxiliary witness is trusted. The monitor has a raw residual gate of 0.9372 rad/s, a two-sided CUSUM threshold of 1.5697, three-sample entry persistence, and five-sample recovery persistence. CUSUM is reset only after complete recovery.

The controller is a constrained nonlinear LSTM-MPC with horizon 20, two move blocks `(5, 15)`, a 50 ms control update, warm starting, and SLSQP limited to eight iterations. The objective already uses exact PyTorch autograd gradients through the recursive LSTM rollout; its runtime problem is nonlinear optimization and recurrent rollout cost, not absence of a Jacobian.

### Evidence and integrity cross-check

The review inspected the implementations, callers, configs, tests, verifiers, reports, and primary artifacts, including:

- `src/motor_model.py`, `src/lstm_model.py`, `src/auxiliary_sensor_model.py`, `src/reliability.py`, and `src/mpc.py`;
- V3 and C4 calibration/evaluation/report/verifier scripts;
- `results/metrics/v3_final_11scenario_runs.csv`, the V3 source-selection and virtual-sensor quality tables, C2 recovery diagnostics, and runtime samples;
- `results/metrics/c4_development_runs.csv`, `c4_development_trace.csv`, and both attribution-event CSVs;
- reliability contamination, MC-dropout, multistep, and architecture-decision studies;
- V3/C4 regression tests and provenance records; and
- all project limitation/audit reports.

The V3 verifier was executed read-only during this review and returned `V3 VERIFIER RESULT: PASS`; it independently reconstructed calibration values, checked hashes, reproduced all 275 rows and summaries, confirmed zero recorded safety violations/failures, and confirmed C2/C1 equivalence.

The C4 development verifier currently returns `C4 VERIFIER RESULT: FAIL`, before reaching its metric reconstruction, because its accepted aliases omit the evaluator's actual columns such as `attr_frac_likely_sensor_fault`. The saved matrix uses `attr_frac_likely_sensor_fault`, `attr_frac_likely_plant_or_main_mismatch`, and `attr_frac_likely_aux_mismatch`, whereas `_state_fraction_columns()` looks for names without `likely`. Existing workflow tests do not exercise the full verifier against the saved development matrix, so they did not catch the contract mismatch. This is an evidence-infrastructure defect, not evidence that the reported C4 NO-GO is wrong: the row counts, distributions, C3/C4 equality, state counts, and stage-gate arithmetic were independently cross-checked directly from the saved artifacts. Nevertheless, the paper must not call the current C4 development evidence independently verifier-PASS until that small schema contract and regression coverage are repaired. No repair is made in this planning task.

`results/metrics/c4_attribution_events.csv` and `c4_development_attribution_events.csv` have the same SHA-256 and the same 347 rows. Their persisted event segments comprise 181 `AMBIGUOUS`, 155 `NORMAL`, 10 `LIKELY_SENSOR_FAULT`, 1 `LIKELY_AUX_MISMATCH`, and zero `LIKELY_PLANT_OR_MAIN_MISMATCH` segments.

## 2. V3 strengths

V3 is a credible, bounded result rather than a universally robust controller.

1. **Strong paired abrupt-fault performance.** In the frozen five-seed matrix, C3 reduces mean fault-window RMSE relative to plain MPC by 87.2% for 5% bias (2.9076 to 0.3712 rad/s), 95.4% for 15% bias (7.9958 to 0.3712), and 98.2% for dropout (20.6028 to 0.3712). Every paired seed has the same favorable sign in those three cases.
2. **Useful combined-fault behavior.** For combined 5% bias plus load, C3 reduces mean fault-window RMSE from 4.3432 to 3.9633 rad/s (8.75%), with improvement in all five paired seeds. During substitution, the selected virtual estimate has RMSE 2.1051 rad/s versus 2.2094 for the main model and 1.7003 for the auxiliary model. The arbitrator therefore extracts some value from its input-diverse auxiliary sensor.
3. **Safety and provenance discipline.** The canonical V3 matrix records zero optimizer failures, prediction failures, nonfinite events, voltage-limit violations, and slew-limit violations. Frozen model, calibration, source, and result hashes are independently checked.
4. **A clear negative boundary.** V3 does not hide that load disturbance is harmful: C3 load fault-window RMSE is 2.5387 versus 0.9371 rad/s for plain MPC, driven by three of five false latches. Drift is essentially unchanged from plain MPC. This honesty strengthens a bounded paper.
5. **Low extension overhead.** Reliability, auxiliary inference, and arbitration add far less computation than the MPC solver. C4 itself averaged only 0.0437 ms per measured attribution update; attribution runtime was never its failure mechanism.
6. **Good one-step main-model representation.** The held-out main LSTM one-step RMSE is about 0.141 rad/s, with recursive RMSE about 0.488 rad/s at horizon 15. The data support using it as a nominal predictor while also showing why long autoregressive substitution is a different problem.

The limitations are material: one trained instance of each LSTM, fixed fault magnitudes, a nominal-simulator training distribution, simulation-only evidence, an assumed-healthy current sensor, no classical observer comparator, weak drift detection, false load substitutions, and no reliable 20 Hz timing.

## 3. C4-v1 negative result

C4-v1 adds a three-way state machine using the scalar distances

```text
d_sm = |sensor - main|
d_sa = |sensor - auxiliary|
d_ma = |main - auxiliary|
```

Its clean-validation thresholds are:

| Pair | Agreement, p75 (rad/s) | Strong disagreement, p99.9 (rad/s) |
|---|---:|---:|
| sensor–main | 0.3254 | 0.9372 |
| sensor–auxiliary | 2.8549 | 9.0477 |
| main–auxiliary | 2.8734 | 8.9381 |

The attributor classifies current **raw** distances; its EWMA is diagnostic only. A new C3 substitution request is suppressed only when both the persisted and current raw states equal `LIKELY_PLANT_OR_MAIN_MISMATCH` and the auxiliary witness is ready. It cannot clear an already active C3 latch.

The predeclared development result is unequivocal:

- 105/105 paired runs are present: B, C3, and C4 across seven scenarios and seeds 19026–19030;
- C4 and C3 are numerically identical in all seven scenarios;
- load false-entry, substitution, and tracking-penalty reductions are all 0%;
- raw and persisted `LIKELY_PLANT_OR_MAIN_MISMATCH` samples are both zero;
- combined, 15% bias, and dropout behavior is preserved only because C4 suppresses nothing;
- parameter-variation RMSE degradation versus the better baseline is 59.7%, failing the gate;
- step and changing-reference cases each contain one false latch and mean substitution fraction 0.0549, failing their gates;
- genuine sensor-fault suppressions are zero, but genuine suppressions attempted are also zero; and
- the stage decision is `NO_GO`.

The unused seeds 39026–39030 remain untouched. There is no scientific basis for spending them on C4-v1.

## 4. Root-cause analysis of C4-v1

### 4.1 The intended load geometry does not occur

The central premise was that a pure load disturbance would produce `sensor ≈ auxiliary` while the main model differed. The saved trace shows the opposite. The following values are medians / p90 / p99 over the defined event or transition windows:

| Scenario/window | `d_sm` | `d_sa` | `d_ma` | Closest pair | Interpretation |
|---|---:|---:|---:|---|---|
| Load, `t ≥ 3 s` | 0.227 / 0.742 / 3.976 | 1.044 / 1.789 / 2.603 | 0.911 / 1.592 / 2.599 | sensor–main 83.1% | Main usually follows the healthy sensor; auxiliary is the less accurate witness |
| Combined fault+load, `t ≥ 3 s` | 1.510 / 3.242 / 4.057 | 1.766 / 3.156 / 3.964 | 0.358 / 1.235 / 4.103 | main–auxiliary 79.4% | Correct qualitative sensor-outlier geometry, but too small for C4's `d_sa` p99.9 threshold |
| 15% bias window | 9.636 / 10.123 / 10.516 | 9.731 / 10.648 / 11.839 | 0.281 / 0.997 / 1.791 | main–auxiliary 100% | Severe sensor outlier |
| Dropout window | 34.992 / 35.354 / 36.151 | 34.897 / 35.474 / 36.144 | 0.281 / 0.997 / 1.791 | main–auxiliary 100% | Severe sensor outlier |
| Parameter variation, `t ≥ 3 s` | 0.282 / 1.797 / 4.888 | 11.655 / 12.831 / 13.407 | 11.555 / 13.609 / 14.363 | sensor–main 97.7% | Auxiliary is strongly out of distribution |
| Step-reference transition | 0.193 / 0.438 / 0.638 | 0.371 / 0.887 / 1.238 | 0.364 / 0.787 / 0.906 | sensor–main 58.0% | Small nominal transient disagreements |
| Changing-reference transitions | 0.189 / 0.482 / 0.688 | 1.163 / 2.540 / 3.121 | 1.098 / 2.623 / 3.019 | sensor–main 83.0% | Geometry is load-like, not a distinct causal signature |

For load disturbance, the intended pattern is therefore empirically false, not merely under-observed. At the exact three load latches, `d_sm` is only 0.168–0.377 rad/s, while `d_sa` and `d_ma` are about 0.41–1.21 rad/s. The main estimate is not the outlier.

### 4.2 Absolute thresholds are restrictive, but retuning them would not fix the premise

The unequal clean scales create unintuitive joint requirements. `LIKELY_PLANT_OR_MAIN_MISMATCH` requires

```text
d_sm ≥ 0.9372, d_sa ≤ 2.8549, d_ma ≥ 8.9381.
```

By the triangle inequality, `d_ma ≥ 8.9381` and `d_sa ≤ 2.8549` imply `d_sm ≥ 6.0832` rad/s. Thus the rule's effective sensor–main separation is over 6 rad/s, even though the nominal `d_sm` threshold is 0.9372. Load entry candidates never approach this: their maximum `d_sm` is 1.0718 rad/s.

The sensor-fault rule has a corresponding severity effect. It requires `d_sa ≥ 9.0477`; combined 5% bias candidates have only 2.55–3.40 rad/s sensor–auxiliary disagreement and are all `AMBIGUOUS`, despite main and auxiliary usually agreeing. The rule recognizes the 15% bias and dropout entry samples, not the milder combined fault.

Clean calibration already signaled this fragility: among 7,086 clean-validation samples, raw attribution was 56.03% `NORMAL`, 43.96% `AMBIGUOUS`, one sample `LIKELY_AUX_MISMATCH`, and zero sensor or plant/main states. After persistence, 36.69% remained `AMBIGUOUS`. A classifier that is ambiguous on roughly two fifths of its calibration population has little margin for causal veto decisions.

Threshold choice is therefore part of the failure, but lowering thresholds would turn C4 into severity retuning around the existing residual detector. It would not manufacture the missing load geometry.

### 4.3 C4 observes the wrong time statistic for a temporal CUSUM decision

All seven actual non-sensor false latches—three load, two parameter, one step-reference, and one changing-reference—occur at a sample where the instantaneous residual is **inside** the 0.9372 rad/s gate. Entry is caused by accumulated CUSUM evidence from prior residuals.

For example, load seed 19027 has residuals `-0.881`, `-0.134`, and `+0.237` rad/s over the final three samples; the corresponding CUSUM scores are 2.062, 2.047, and 1.660, all above the 1.570 threshold. At the latch sample the current geometry is `NORMAL`, and the physical-sensor error is only 0.030 rad/s. In load seed 19029, two preceding approximately `-1.0` rad/s residuals push CUSUM over threshold, but the latch sample residual is only `-0.377`. Similar sequences occur for parameter and reference cases.

C4's veto is based on the current raw three-way geometry plus a categorical persistence state; it does not consume the signed residual path or CUSUM decomposition that actually caused the request. Requiring the current raw state to remain `LIKELY_PLANT_OR_MAIN_MISMATCH` at the latch further guarantees that a subsided transient cannot be suppressed.

### 4.4 The false entries are stochastic monitor-tail events, then architecture-amplified

False latches recur at the same seed/time across different non-fault scenarios: seed 19029 at 4.39 s appears in load, parameter, step-reference, and changing-reference; seed 19030 at 3.85 s appears in load and parameter. The scenario definitions and implementation path are the same between the development and frozen V3 evaluation. This points to base-noise/model-residual tail sensitivity rather than a scenario-specific causal signature.

Before entry, the history is physical, so virtual-history contamination is not the initial cause. After a false latch, the healthy physical sensor is replaced and predicted feedback enters the LSTM history. That converts a small stochastic detector error into a potentially long autoregressive episode. Under load seed 19029, main-model absolute error averages 0.565 rad/s in the first 0.1 s after latching, rises to 3.272 over 0.1–0.5 s, and reaches 5.509. Thus the root chain is:

```text
tail residuals / fixed CUSUM calibration
    → false persistent latch
    → healthy measurement removed
    → autoregressive virtual history under plant mismatch
    → tracking penalty and delayed recovery
```

High operating demand is a plausible risk modifier, not a complete explanation. In clean validation, all eight CUSUM-threshold exceedances occur in the top voltage quartile, and the top current and acceleration quartiles contain all eight as well. However, two load latches occur well after the load step at low instantaneous acceleration. An operating-point-aware monitor is worth a later diagnostic study, but the present evidence does not justify retuning it before the paper-hardening baselines.

### 4.5 Offline truth identifies the physical sensor, not the main model, at every false latch

`y_true` was used only retrospectively. At all seven actual non-sensor false latches, the physical sensor is closest to truth:

| Class | Actual latches | Closest-to-truth result at latch | Auxiliary status |
|---|---:|---|---|
| Load | 3 | physical 3/3 | trusted |
| Parameter variation | 2 | physical 2/2 | untrusted due parameter-mismatch logic |
| Step reference | 1 | physical 1/1 | trusted |
| Changing reference | 1 | physical 1/1 | trusted |
| Combined fault+load | 5 | main 4/5, auxiliary 1/5 | trusted at entry |
| 15% bias | 5 | main 5/5 | trusted at entry |
| Dropout | 5 | main 5/5 | trusted at entry |

This is strong retrospective evidence that the load/reference entries are detector false positives, but it is not an online feature. It also confirms that C4's hypothesized “main outlier under load” is absent at precisely the decision time.

### 4.6 Auxiliary witness readiness makes long-lived attribution mostly ambiguous

The witness is available at the abrupt entry samples, which is enough for C4's one-time veto design. Once C3 latches, V3 no longer accumulates trusted physical history, so auxiliary readiness becomes `INSUFFICIENT_TRUSTED_HISTORY`; C4 then fails closed to `AMBIGUOUS`. This explains why only the first three raw samples per severe bias/dropout run are `LIKELY_SENSOR_FAULT`, with only one persisted sample per run, even though the numerical main/auxiliary agreement remains sensor-fault-like. It is not the reason load suppression is zero—load geometry is already wrong at entry—but it means C4 is not a useful sustained fault-attribution state machine.

## 5. Separability findings

### 5.1 Structural information limit

The three estimates are scalars on one real line. Their three pairwise absolute distances satisfy, to floating-point precision, that the largest distance equals the sum of the other two. The maximum observed deviation from this identity in the saved trace is approximately `1.4e-14`. Only two distance degrees of freedom exist.

The proposed relative scores do not add information. For whichever estimate lies between the other two, its outlier score is exactly zero; the other scores are algebraic transformations of the same two gaps. Normalization by positive clean scales likewise changes units, not rank ordering or univariate ROC-AUC.

This is visible empirically:

- among true sensor-fault entry candidates, correlation(`d_sm`, `d_sa`) is 0.999;
- among false candidates, correlation(`d_sa`, `d_ma`) is 0.996; and
- over all candidates, correlation(`d_sm`, `d_sa`) is 0.971.

The three estimates are input-diverse, but their scalar pairwise distances are not three independent causal witnesses.

### 5.2 Development-candidate diagnostics

There are 45 true sensor-fault new-entry candidates (combined, 15% bias, dropout) and 21 non-sensor candidates (load, parameter, step, changing-reference). Direction-free univariate ROC-AUC and range summaries are:

| Diagnostic | ROC-AUC | False candidate min / median / max | Sensor-fault candidate min / median / max |
|---|---:|---:|---:|
| `d_sm` | 1.000 | 0.141 / 0.966 / 1.087 | 2.429 / 9.870 / 35.292 |
| `d_sa` | 0.814 | 0.226 / 1.561 / 12.121 | 2.547 / 11.025 / 34.208 |
| `d_ma` | 0.512 | 0.011 / 0.614 / 11.425 | 0.007 / 0.790 / 1.686 |
| normalized `z_sm` | 1.000 | same ordering as `d_sm` | same ordering as `d_sm` |
| `main_outlier_score` | 0.729 | 0.000 / 0.000 / 0.474 | 0.000 / 0.014 / 3.372 |
| `sensor_outlier_score` | 1.000 | 0.000 / 1.932 / 2.173 | 4.858 / 19.741 / 68.415 |
| `aux_outlier_score` | 0.692 | 0.000 / 1.227 / 22.851 | 0.000 / 0.000 / 3.372 |
| `d_sm / (d_sa + d_ma)` | 0.910 | median 0.340 | median 1.000 |

A post-hoc threshold placed in the observed `d_sm` gap (for example 1.75 rad/s) produces a 45 TP / 21 TN / 0 FP / 0 FN confusion matrix on these candidates, with retrospective precision and recall both 1.0. This is **not** adequate evidence for a classifier or C4-v2:

1. It simply separates three large abrupt fault magnitudes (minimum 2.429 rad/s at entry) from residual-tail candidates (maximum 1.087).
2. The same scalar is already the reliability monitor's residual magnitude; the result is a severity gate, not causal three-way attribution.
3. It excludes the unresolved drift behavior, sensor-noise overlap, sub-5% bias, shorter dropouts, broader loads, training-weight variation, and calibration transfer.
4. The 66 candidates come from only five correlated simulation seeds and fixed event definitions.
5. Combined-fault candidates are all `AMBIGUOUS` under C4; the 30 classified sensor candidates are only the easier 15% bias and dropout cases.

Across the full event windows rather than cherry-picked entry candidates, overlap reappears. `d_sm` ROC-AUC is 0.933, but the false-event p90 is 1.353 while the sensor-fault p10 is 0.740. `d_sa` falls to 0.642. Thus the apparent perfect gap is local to the trigger construction and fixed severities.

### 5.3 Load and reference transitions are not separable in this space

For seven load entry candidates versus eight step/changing-reference candidates, direction-free AUC is 0.554 for `d_sm`, 0.500 for `d_sa`, and 0.643 for `d_ma`. Two-sample KS p-values are 0.42, 0.91, and 0.19 respectively; sample sizes are small, but there is no evidence of a useful distinction. Shared seed/time latches reinforce the same conclusion.

### 5.4 Separability conclusion

Normalized/relative geometry is numerically cleaner than C4-v1's absolute small/large state definitions, but it is not materially more causal. Existing information can distinguish the fixed severe faults from small CUSUM-tail candidates by magnitude. It cannot justify suppressing a fault detector because “the plant or main model, not the sensor, is at fault.” That stronger claim needs an additional independent signal—such as a physics observer innovation, load estimate, or independently measured state—not another transform of the same three scalar estimates.

## 6. Whether C4-v2 is scientifically justified

**Decision: C4-v2 is not justified.**

Do not implement normalized thresholds, relative outlier scores, likelihood ratios, or another neural classifier over `(d_sm, d_sa, d_ma)` as the next architecture. Such a C4-v2 would capitalize on the post-hoc 1.087-to-2.429 rad/s severity gap while leaving the failed load premise, temporal CUSUM mismatch, mild-fault coverage, and two-degree-of-freedom geometry unresolved.

This decision could be reopened only if new development evidence adds a genuinely independent online signal and passes a pre-registered causal gate. A reasonable future gate would require all of the following before any untouched final seeds are used:

- development includes mild bias, drift, sensor noise, dropout duration, load magnitude, parameter variation, and reference transitions;
- at least three independently trained model pairs are represented;
- sensor-fault-versus-plant/reference candidate AUROC has a bootstrap lower 95% bound of at least 0.85;
- sensitivity is at least 90% at no more than 5% false suppression of plant/reference candidates, separately within every critical scenario rather than only pooled;
- the proposed feature adds information beyond `|sensor-main|` in an ablation;
- online logic uses no truth, event labels, or scenario identity; and
- the design is frozen before a truly untouched seed block is selected.

If the added signal is an EKF innovation or estimated load torque, the work is scientifically a model-based diagnostic or hybrid observer—not merely C4-v2—and should be named accordingly.

## 7. Remaining architectural ambiguities

### A. Load disturbance

The immediate cause of substitution is a fixed CUSUM reacting to a rare signed residual sequence, sometimes seeded by one or two instantaneous residual excursions. It is not an observed main-model outlier at the latch. High-demand operating regions raise clean residual/CUSUM tails, but the load latches are often delayed and occur at low instantaneous acceleration, so a simple transition blanking window would not solve the problem.

The large tracking penalty is a second-stage effect: after the false latch, virtual feedback replaces an accurate physical measurement, enters the recurrent history, and can diverge under unmodeled load. The load seed with the long episode reaches main error 5.509 rad/s. Root-cause attribution is therefore a combination of monitor calibration/noise sensitivity and post-latch autoregressive amplification; control transient alone and virtual-history contamination alone are incomplete explanations.

### B. Parameter variation

The C4 development block latches for seeds 19029 and 19030, while the final V3 block has no persistent parameter latch in seeds 29026–29030 and is essentially identical to plain MPC. The parameter definition and simulation path are the same. The repeated false-latch seed/times across parameter, load, and reference cases show that this difference is primarily evaluation-noise-seed sensitivity interacting with the monitor's CUSUM, not a changed scenario definition.

When a latch does occur, the auxiliary model is grossly out of distribution: event-window auxiliary error is about 11–12.5 rad/s and V3 correctly marks it untrusted. The main model is then used autoregressively; in parameter seed 19029 its mean error during the episode is 3.115 rad/s and its maximum is 5.243. Parameter variation therefore exposes both monitor seed sensitivity and the auxiliary model's nominal-parameter dependence. The V3 trust guard prevents auxiliary misuse but cannot make the remaining main estimate robust.

### C. Drift

Drift is primarily an **entry-detector sensitivity/input-contamination** problem. Because the main LSTM consumes measured speed, slow corruption enters both the residual's measurement and predictor input, allowing the residual to absorb the drift. The saved guarded-input ablation materially improves small/large bias and dropout detection but leaves drift detection around 1.3%, and V3 substitutes for only about 0.1% of drift samples. Existing auxiliary evidence under nominal-plant drift is substantially better than the corrupted sensor, so virtual-feedback quality is not the first bottleneck. A drift-capable detector would still need validation before assuming long-horizon virtual feedback remains stable.

### D. Auxiliary `[voltage, current]` model

The frozen clean test RMSE is 3.4376 rad/s (MAE 2.5764, R² 0.9441), with per-trajectory RMSE 1.17–5.67. Retrospective operating-region bins show:

| Region | RMSE (rad/s) | Comparison |
|---|---:|---|
| Lowest current quartile | 2.612 | Not a low-current failure |
| Highest current quartile | 4.402 | Clear degradation |
| Lowest speed quartile | 2.951 | Better region |
| Highest speed quartile | 4.194 | Degraded |
| Highest acceleration quartile | 4.182 | Degraded transient region |
| Exactly 12 V saturation | 4.922 | Worse than 3.186 off saturation |
| First 0.5 s after available windows | 3.597 | Only modestly worse than 3.433 later |

These bins are correlated and should not be read causally, but they rule out “mainly low current” and “mainly startup” explanations. The dominant demonstrated failure is nominal-model transfer: under the simultaneous parameter shift, auxiliary error is around 11–12.5 rad/s. Under load and combined-fault cases it is often useful (about 1.6–1.7 rad/s during selected/substitution windows), because current carries load information. Its other major unresolved risk is not an operating region but the untested assumption that armature current remains healthy.

### E. Main model autoregressive feedback

The main model remains accurate during isolated nominal-plant abrupt faults: virtual RMSE is about 0.319 rad/s for 5%/15% bias and dropout in the frozen matrix. Divergence appears when all three conditions align:

1. the physical sensor is removed for more than an instantaneous guard;
2. main predictions are recursively inserted into the measured-speed input channel; and
3. the plant/reference behavior is outside or near the edge of nominal open-loop training behavior.

Examples from the development trace are exact: load error grows from 0.565 in the first 0.1 s to 3.272 rad/s over 0.1–0.5 s in the long false episode; parameter error grows from 0.396 to 2.617 over 1–3 s; combined fault+load reaches approximately 3.12 over 0.1–0.5 s and maxima of 5.09–5.84. Step/changing-reference false episodes settle around 0.9–1.1 rad/s rather than catastrophic divergence. The issue is therefore distribution shift plus recursive closure, not poor clean one-step fit.

### F. C2

There is no hidden C2 issue. Across 10,251 recorded recovery opportunities, CUSUM blocks all 10,251, the raw residual condition blocks 5,849, persistence uniquely blocks zero, the auxiliary gate uniquely blocks zero, and recovery occurs zero times. C2 is a valid negative ablation showing that an extra auxiliary recovery gate cannot help while the existing CUSUM is always binding. Do not spend more work on it.

### G. Evidence tooling

The C4 verifier/evaluator schema mismatch is the only newly found repository-level integrity defect relevant to this decision. It should receive a minimal alias fix plus one regression that runs `_state_fraction_columns()` against the actual saved column set. This is evidence maintenance, not a scientific rerun and not a reason to alter the frozen C4 result.

## 8. Paper-readiness gaps

### Must have before journal submission

| Addition | Why it is mandatory | Main value added |
|---|---|---|
| One fair classical observer baseline | The plant equations are available and the auxiliary LSTM uses the same voltage/current signals. Without it, “why two LSTMs?” is unanswered. Direct ANN-versus-EKF comparison is established practice in DC-motor speed estimation. | Reviewer confidence; methodological rigor |
| Three total independent training seeds per main/auxiliary pair | Current evaluation seeds vary simulation noise, not learned weights. A single favorable initialization could determine agreement thresholds and arbitration behavior. | Reviewer confidence; robustness |
| Compact fault-severity envelope with frozen thresholds | Fixed 5%/15% bias, one dropout duration, one drift slope, and one load step do not establish an operating envelope. | Real-world credibility; rigor |
| Paired effect sizes and uncertainty | Means over five seeds are not a substitute for uncertainty. Use paired controller differences, confidence intervals, and a small set of predeclared primary endpoints. | Methodological rigor |
| Current-sensor assumption treatment | At minimum, explicitly state a single-fault assumption and add a small current-bias/dropout sensitivity study. Both the auxiliary LSTM and proposed observer depend on current. | Real-world credibility |
| Repair and regression-test the C4 verifier schema before citing it as independently verified | Current development verification stops on column aliases. | Evidence integrity |
| Bound all claims | State simulation-only scope, drift weakness, load false-entry behavior, parameter-transfer limits, lack of formal stability proof, and failure to meet 20 Hz timing. If “real time” is claimed, timing must first be fixed. | Scientific accuracy |

### Should have

| Addition | Rationale |
|---|---|
| Additional one-at-a-time motor/plant parameter variations | The current simultaneous ±15–20% shift identifies a failure but not which parameters matter or how calibration transfers. |
| Calibration-transfer study | Train/calibrate on nominal trajectories, test unchanged thresholds across operating ranges and modest plant families; do not recalibrate per test case. |
| Operating-point residual analysis | Clean CUSUM exceedances concentrate at high voltage/current/acceleration. A stratified diagnostic can motivate future calibration, but should not become another tuned controller in this paper. |
| End-to-end timing on a controlled CPU target | Even if the paper remains simulation-focused, report controller deadline misses and observer overhead on specified hardware. |
| HIL or a small benchtop experiment if targeting an applied-control venue | This would materially improve external validity, but it is not the shortest path for a clearly labeled simulation/method paper. |

### Optional / future work

| Addition | Why it can wait |
|---|---|
| Trajectory-linearized/QP LSTM-MPC | High scope and substantial existing literature overlap; it is better framed as a separate real-time-control contribution. |
| C4-v2 or a learned attribution classifier | Present signals do not justify it. |
| UKF, sliding-mode observer, and multiple classical observers | One defensible EKF is enough unless it exposes a specific failure that demands another comparator. |
| Formal robust stability/recursive-feasibility proof | Essential only if such guarantees are claimed. The present paper should explicitly state that empirical constraint satisfaction is not a proof. |
| Full hardware campaign | High value but hardware-dependent; pursue after the simulation claims survive observer and training-seed baselines. |
| Redundant current-sensor architecture | Add only if the compact current-fault study shows that the explicit single-fault assumption is unacceptable for the intended claim. |

## 9. Ranked extension options

Scores use 1–5. For scientific value, novelty, probability of success, and reviewer value, 5 is best/highest. For implementation effort, compute cost, and scope-creep risk, 5 means most expensive/risky. The ordering is a research judgment, not an arithmetic weighted sum.

| Rank | Option | Scientific value | Novelty | Success probability | Effort | Compute | Reviewer value | Scope risk | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Classical augmented-state EKF baseline | 5 | 2 | 5 | 2 | 1 | 5 | 1 | Do next |
| 2 | Three-total-training-seed robustness | 5 | 1 | 5 | 2 | 3 | 5 | 1 | Required after baseline |
| 3 | Compact fault-severity envelope | 4 | 2 | 5 | 2 | 3 | 5 | 1 | Required after seed gate |
| 4 | No more architecture; write the bounded V3 paper | 4 | 1 | 4 | 1 | 1 | 3 | 1 | Correct direction, but only after ranks 1–3 close the main gaps |
| 5 | Operating-point-aware reliability calibration | 3 | 3 | 3 | 3 | 2 | 3 | 3 | Diagnostic only for now |
| 6 | Hardware/HIL | 5 | 2 | 2 | 5 | 2 | 5 | 5 | Valuable later/venue-dependent |
| 7 | Trajectory-linearized or QP LSTM-MPC | 4 | 2 | 3 | 5 | 4 | 4 | 5 | Separate future contribution |
| 8 | C4-v2 normalized/relative attribution | 2 | 2 | 1 | 3 | 2 | 2 | 5 | Stop |

Operating-point calibration ranks above C4-v2 because the clean-validation CUSUM evidence has a real operating-region dependence. It still should not precede the observer and robustness baselines: any retuning now would risk another development-specific threshold search without resolving causal attribution.

## 10. Recommended next architecture

### Primary recommendation

**Implement one standalone augmented-state EKF as a fair classical virtual-speed comparator. Do not integrate a new attribution layer into frozen V3.**

Use the existing nonlinear motor equations and the minimum state

```text
x = [armature current i, angular speed ω, load torque T_L]
u = applied armature voltage V
measurement = armature current i_meas
```

Model `T_L` as a slow random walk. Use the repository's smooth-friction equation, nominal motor parameters, and 10 ms discretization. The current-only measurement makes the comparison input-fair with the auxiliary LSTM: both receive applied voltage and measured current, and neither may read the speed sensor or `y_true` while producing the independent speed estimate. Initialize from measured current, zero/declared speed, and nominal load; use a Joseph-form covariance update and explicit finite/positive-semidefinite checks. Calibrate only process and measurement covariance on the existing training/validation population, then freeze them.

Do not add a UKF, Luenberger observer, learned correction, adaptive parameter estimator, or observer ensemble at this stage. The EKF is the first rung that answers the reviewer question. If local observability or numerical conditioning fails in a documented region, that failure is itself informative and can motivate exactly one follow-up.

### Fair comparisons

1. **Estimator-only:** auxiliary LSTM versus EKF on the identical held-out clean trajectories, with RMSE/MAE by speed, current, acceleration, saturation, and trajectory.
2. **Transfer:** both estimators under load and frozen parameter variations, without online truth or test-set covariance fitting.
3. **Closed-loop comparator:** retain the same frozen V3 sensor monitor. When it requests substitution, use the EKF speed as the sole virtual feedback in a separate comparator controller. Compare it with frozen B, C1, and C3 rows; do not rewrite V3 evidence.
4. **Runtime:** report EKF update time separately and end-to-end controller timing.
5. **Current-fault sensitivity:** expose that EKF and auxiliary LSTM share the current-sensor assumption.

This baseline matters scientifically because a direct current/voltage-to-speed observer is not hypothetical. Published DC-motor work explicitly compares neural speed estimation with EKF under the same conditions ([Aydogmus & Aydogmus, 2015](https://doi.org/10.1016/j.measurement.2014.12.010)), and brushed-DC current-based speed monitoring has been demonstrated with Kalman filtering and experimental validation ([Zhang, Wen, & He, 2021](https://doi.org/10.1016/j.measurement.2021.109890)). If the EKF matches or beats the auxiliary LSTM, the manuscript must narrow or change its learned-sensor claim. If V3 wins under the same inputs, the learned contribution becomes much more defensible.

## 11. Exact staged experiment plan

Nothing in this section is executed by this planning task.

### Phase 1 — Evidence hygiene and observer pre-registration

1. Repair only the C4 verifier's state-fraction aliases and add one regression using the actual saved development column names. Confirm the repaired verifier reproduces the existing NO-GO without changing any scientific artifact.
2. Write the EKF equations, discrete Jacobians, initialization, covariance-calibration population, and no-oracle contract before coding the closed-loop comparator.
3. Check local observability/conditioning across the dataset's voltage/current/speed range, explicitly identifying weak excitation regions.
4. Predeclare primary endpoints: fault-window RMSE for 5% bias, dropout, combined fault+load, load false-entry/substitution rate, and parameter-transfer estimator RMSE.
5. Predeclare two additional training seeds so that the study has three total paired model seeds including 2026. Keep the data split and every hyperparameter fixed; do not select the best training seed.
6. Choose a future evaluation seed block that is distinct from 39026–39030, or leave seed choice sealed until the observer and protocol are frozen. The C4 final seed block stays unused.

**Phase-1 pass:** the current/voltage observer is locally observable over the claimed nonzero-excitation envelope; its implementation contract is causal; all calibration and primary endpoints are frozen; the C4 verifier contract is repaired without altering results.  
**Phase-1 stop:** current-only observation is structurally unobservable over substantial claimed operation, or a fair EKF would require speed/truth leakage. In that case, report a simpler physics reconstruction baseline and do not build a larger observer family.

### Phase 2 — Implement exactly one EKF comparator

1. Add a small, standalone EKF with `[i, ω, T_L]`, one configuration, and focused numerical/causality tests.
2. Calibrate `Q` and `R` on train/validation only. No evaluation-scenario tuning.
3. Run estimator-only checks first. Include clean trajectories, load transfer, simultaneous parameter variation, saturation, startup, and zero/low excitation.
4. Add one separate closed-loop comparator that uses the frozen sensor monitor and substitutes EKF speed. It must not modify C3/V3 behavior or artifacts.

The EKF result belongs in the paper regardless of whether it wins. Architectural adoption has a separate gate:

- **Adopt/investigate as a V3 successor only if** it is finite in every sample, causes zero constraint/safety failures, does not worsen isolated 5% bias or dropout fault-window RMSE by more than 5% versus C3, and improves median combined/load selected-feedback or fault-window RMSE by at least 10% in at least four of five paired seeds.
- **Baseline only** if it is valid but misses that performance gate. V3 remains the architecture.
- **Stop observer expansion** if it is numerically unstable or needs evaluation-specific covariance tuning. Do not respond by adding UKF/ensembles without a new hypothesis.

### Phase 3 — Collect the minimum journal evidence

#### 3A. Observer comparison

Use five critical scenarios—5% bias, dropout, load, combined fault+load, and parameter variation—across five paired simulation seeds. One new EKF comparator means 25 new closed-loop runs; frozen V3 B/C1/C3 rows can be reused when the scenario definitions and code hashes match.

#### 3B. Training-seed robustness

Use three total independently trained main/auxiliary pairs: existing seed 2026 plus two predeclared new seeds. For each pair, evaluate only B and C3 on the same five critical scenarios and five paired simulation seeds:

```text
3 training seeds × 5 scenarios × 5 simulation seeds × 2 controllers = 150 cells
```

If the existing seed-2026 50-cell subset is reused exactly, only 100 new closed-loop runs are needed. Never select or average models into an ensemble; the purpose is to expose training variance.

The recorded V3 and C4 evaluations took 311.6 s for 275 runs and 144.8 s for 105 runs respectively (about 1.13–1.38 effective wall-clock seconds per run with their parallel setup). A 150-cell matrix is therefore roughly 3–4 minutes of comparable evaluation wall time, or about 2–3 minutes for the 100 new cells, excluding model training, model loading differences, verification, and machine contention. Training wall time was not preserved, so it must be measured rather than invented.

#### 3C. Compact one-factor-at-a-time severity envelope

Keep all thresholds frozen. Use four levels for each factor:

- bias: 2.5%, 5%, 10%, 15% of full scale;
- dropout duration: 0.1, 0.5, 1.0, 2.0 s;
- drift: final 2.5%, 5%, 10%, 15% of full scale over the same four-second ramp;
- post-step load torque: 0.06, 0.10, 0.15, 0.20 N·m from the same 0.03 N·m baseline.

With B and C3 and five simulation seeds, this is `16 × 2 × 5 = 160` cells; exact existing cells may be reused only with matching provenance. The most publishable curves are:

- detection probability and Wilson interval versus severity;
- detection/recovery latency versus severity;
- fault-window and recovery-window RMSE, shown as paired C3–B differences;
- substitution fraction and episode duration;
- main/auxiliary/fallback source fraction; and
- false-entry probability under load magnitude.

Do not fit thresholds to make these curves look monotone. Non-monotonicity is a finding.

#### 3D. Current-sensor sensitivity

Add a deliberately small test: current bias, current dropout, and simultaneous speed/current corruption, each across the same five simulation seeds for C3 and the EKF comparator (`3 × 5 × 2 = 30` runs). This is not a multi-fault architecture project; it quantifies the boundary of the stated single-fault assumption.

#### 3E. Statistics

- Use paired per-seed controller differences, not unpaired means.
- Report median/mean effect, 95% confidence interval, and all individual paired points.
- With the three-by-five hierarchy, bootstrap training seed and simulation seed as clusters; do not treat all time samples as independent replicates.
- Predeclare a small primary endpoint family and use an exact paired permutation test or a transparent multiplicity correction for secondary endpoints.
- Report negative load and drift results without converting them into tuning targets.

At roughly 315 new closed-loop cells for 3A–3D after maximal valid reuse, comparable simulation wall time is approximately 6–8 minutes on the recorded setup, conservatively under 15 minutes excluding retraining and verification. Compute is not the limiting factor; experimental discipline is.

### Phase 4 — Journal assembly, then decide whether hardware is needed

1. Keep V3 frozen as the main learned architecture unless the predeclared EKF adoption gate passes.
2. Present C2 and C4-v1 as negative ablations that delimit what extra consistency logic does not solve.
3. Center the paper on paired fault-tolerant performance, observer comparison, training-weight robustness, and a measured severity envelope.
4. Include source selection, deadline misses, all safety/failure counts, and calibration-transfer limitations.
5. Make no formal stability, recursive-feasibility, fault-isolation, or real-time claim beyond the evidence.
6. Submit as a simulation/methodology paper if the target venue accepts that scope. Add HIL before submission only for a venue whose contribution standard requires experimental validation or if the paper claims deployability.

## 12. Stop/go criteria

### C4

**STOP now.** No C4-v2, no retuning, no final-seed use. Reopen only under the independent-signal and separability gate in Section 6.

### Observer baseline

**GO as a required comparator** after causal/observability checks; report it whether it wins or loses.  
**GO as a successor architecture** only under the no-regression and ≥10% paired improvement gate in Phase 2.  
**STOP expanding observer complexity** if one valid EKF is inferior or unstable; a negative baseline is enough for the paper.

### Training-seed robustness

The core abrupt-fault claim passes only if C3 improves over B for 5% bias and dropout for every one of the three training seeds, and the combined-fault paired effect remains favorable in each training seed. Load need not become favorable, but its false-entry/degradation direction must be stable and explicitly reported. If the headline conclusion reverses with training initialization, stop adding features and treat training instability as the main research result/gap.

### Severity envelope

Define the claimed operating envelope only where the lower confidence bound on detection and the paired RMSE evidence support it. Do not claim drift tolerance if drift detection remains negligible. Do not claim plant/sensor isolation while load false-entry probability is material. Zero safety violations in a finite simulation study is required but is not a proof of safety.

### Real-time MPC

The current SLSQP measurements—mean 67.96 ms, median 60.55, p95 135.78, p99 166.62, maximum 666.87, and 58.44% over a 50 ms deadline—fail a 20 Hz real-time claim.

If computational MPC becomes a future project, success must be demonstrated against the unchanged nonlinear SLSQP controller with:

- tracking RMSE and control variation within a predeclared small tolerance (for example 2%);
- identical voltage/slew constraints and zero optimizer/nonfinite failures;
- p99 solve time at or below 50 ms on a named target CPU;
- deadline-miss fraction below 0.1%, with maximum and jitter still reported; and
- paired nominal, reference, sensor-fault, load, combined, and parameter scenarios.

Trajectory-linearized LSTM-MPC is stronger than C4-v2, but not the next task. The literature already contains local linearized LSTM predictive control ([Schwedersky et al., 2019](https://doi.org/10.1016/j.ifacol.2019.06.106)), implementation/derivative/solver comparisons with real-time multiple shooting ([Jung et al., 2023](https://doi.org/10.1016/j.engappai.2023.106226)), and advanced trajectory linearization that converts LSTM/GRU MPC to quadratic programs and reports speedups ([Zarzycki & Lawryńczuk, 2022](https://doi.org/10.1016/j.ins.2022.10.078)). In this project, novelty would have to be deadline-aware integration with reliability-selected virtual feedback plus a rigorous same-model benchmark or HIL result—not trajectory linearization itself.

## 13. What NOT to implement

- Do not build C4-v2 from normalized distances, ratios, outlier scores, or another classifier over the same three scalar estimates.
- Do not retune C4-v1, C3, V3, reliability thresholds, or auxiliary trust thresholds in response to these diagnostics.
- Do not rerun C4-v1 or inspect/use seeds 39026–39030.
- Do not modify or overwrite any frozen V3/C4 metric, config, plot, model, hash, or report.
- Do not add multiple observer families, an observer ensemble, an adaptive observer, or a learned EKF correction before one EKF baseline is evaluated.
- Do not start trajectory-linearized MPC, explicit neural control, HIL, or hardware integration in the current paper-hardening phase.
- Do not add a new neural attribution network; the labels are simulation-defined, few, correlated, and severity-confounded.
- Do not claim the auxiliary LSTM is independent of all sensors; it assumes healthy current.
- Do not claim generalized drift tolerance, plant/sensor fault isolation, real-time feasibility, or formal stability/recursive feasibility.
- Do not convert the frozen negative C2 or C4 results into more tuning loops. Their value is that they close unproductive branches.

## 14. Recommended journal-level contribution story

The strongest defensible story is:

> A reliability-aware constrained LSTM-MPC architecture uses input-diverse learned virtual speed estimates to maintain control under abrupt speed-sensor faults. A frozen, paired, multi-scenario evaluation shows large gains for bias and dropout and consistent improvement for combined sensor-fault/load cases, while explicitly identifying failure boundaries under load false alarms, drift, plant-parameter shift, and real-time computation. A classical current/voltage EKF comparator, multiple training seeds, and a fixed-threshold severity envelope establish whether the learned dual-sensor design adds value beyond a physics observer. Negative C2 and C4 ablations show that additional recovery and scalar three-way consistency gates do not solve causal fault attribution.

The novelty should not be “three models vote” or “LSTM-MPC is real time.” It should be the rigorously bounded reliability/feedback-selection result, including where it fails and why. If the EKF wins, the honest contribution becomes a comparative result that motivates a simpler model-based or hybrid successor; the paper should not protect the two-LSTM story at the expense of the evidence.

The shortest credible path is therefore: repair the verifier contract, implement one EKF comparator, add two training initializations to make three total, run the reduced critical matrix and compact severity/current-sensor studies with paired uncertainty, then write the bounded V3 paper. Stop adding architecture unless those results expose a specific, independently measurable gap.

## Direct answers to the final questions

1. **Why exactly did C4-v1 fail?** The hypothesized load geometry is absent: at load entries the physical sensor and main model agree while the auxiliary is farther away. Pair-specific p75/p99.9 thresholds also impose an effective >6 rad/s main separation and miss the 5% combined fault. Most importantly, C3 false latches are caused by accumulated signed CUSUM history, while C4 vetoes on current raw geometry after the residual excursion has often subsided. Scalar-distance redundancy and witness-readiness collapse then make `AMBIGUOUS` dominant.
2. **Does the existing three-sensor disagreement information contain enough signal for C4-v2?** No for causal, generalizable attribution. It contains a trivial severity gap on the fixed development entry candidates, but only two independent distance degrees of freedom and substantial broader-window/mild-fault overlap.
3. **If yes, what should C4-v2 look like?** Not applicable under the evidence. A future method would need an independent causal signal such as observer innovation or estimated load torque and would no longer be merely normalized C4.
4. **If no, what should replace it?** A minimum augmented-state EKF comparator using applied voltage and measured current, followed by training-seed and severity robustness—not another online attribution layer.
5. **What unresolved ambiguity is most dangerous to the paper?** Whether the two learned virtual sensors add value over the obvious physics-based current/voltage observer. Because the simulator equations are known and both LSTMs are trained on one nominal model seed, the missing classical baseline is the clearest reviewer vulnerability.
6. **What experiment gives the highest reviewer value per unit effort?** The fair three-state EKF baseline on the five critical paired scenarios. It is small, cheap, directly tests the learned-sensor rationale, and can strengthen the paper whether it wins or loses.
7. **Is an observer baseline now mandatory?** Yes for a journal submission built around two LSTM virtual sensors.
8. **Is training-seed robustness necessary?** Yes. Evaluation seeds do not measure learned-weight uncertainty. Three total fixed-hyperparameter training seeds are the minimum useful study.
9. **Is trajectory-linearized MPC worth doing now?** No. It is more promising than C4-v2 but has substantial prior-art overlap, high scope, and does not close the paper's most vulnerable evidence gaps. Treat it as a later or separate real-time-control project.
10. **What should be the next implementation task?** After the small verifier-contract repair, implement the standalone `[i, ω, T_L]` EKF comparator and its causal/numerical checks. Do not alter V3.
11. **What should NOT be touched?** Frozen V3, C2, C4-v1 logic/calibration/evidence, models, source-selection rules, and seeds 39026–39030. Do not retune thresholds or start a new classifier/MPC/HIL architecture.
12. **What is the shortest credible path from current V3 to a journal-grade paper?** One EKF baseline, two additional training initializations, a reduced five-scenario paired robustness matrix, a compact frozen-threshold severity/current-sensor study, paired confidence intervals, and a tightly bounded simulation claim. Then write; add HIL only if the selected venue requires it.
