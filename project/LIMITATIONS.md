# Known Limitations and Mechanistic Explanations

This note records the measured boundaries of the frozen V3 architecture. All
numbers below reproduce from canonical saved evidence (the 275-run V3 matrix
`results/metrics/v3_final_11scenario_runs.csv` and the Part A/Part B final
robustness study). No threshold, seed, or artifact was changed to write it.

## Para 1 — Load disturbance false activation

CUSUM on the single residual r1 = |y_m − ŷ₁| cannot distinguish a load step
from a sensor fault for a mechanistic reason: a load step moves the true plant
speed, and the main LSTM — an autoregressive model trained on nominal dynamics
and conditioned on recent measured-speed history — mispredicts through the
transient, so the residual crosses the gate even though the sensor is healthy.
In the frozen V3 matrix, mean load-disturbance RMSE is 0.937127 rad/s for
plain MPC versus 2.538714 rad/s for C3 (a 2.7× degradation), with mean C3
substitution fraction 0.27; the Part B severity study confirms the mechanism
at larger scale, with pooled substitution 0.29–0.60 and false-entry risk
rising 0/15 → 10/15 across 0.06–0.20 N m. The architectural fix is three-way
attribution using r2 = |y_m − ŷ₂| and r12 = |ŷ₁ − ŷ₂|: under plant
disturbance the current-only auxiliary agrees with the physical sensor while
the main model drifts, which positively identifies the main model (not the
sensor) as the outlier. This fix is implemented and calibrated as the C4
`ThreeWayConsistencyAttributor` (clean-validation agreement/disagreement
bands, entry suppression only on positively identified main-model mismatch);
the C4 development stage gate remains NO_GO, so V3 ships with this limitation.

## Para 2 — Sensor drift zero benefit

An autoregressive LSTM absorbs a slow ramp bias into its input history after
the onset transient: each new corrupted measurement enters the conditioning
window, so subsequent predictions track the drifted signal and the residual
stays small indefinitely. CUSUM therefore never accumulates entry evidence
except at the brief onset. In the frozen V3 matrix, mean C3 substitution
during drift is 0.10% (per-run maximum 0.17%) with zero measured benefit
(improvement −0.00%), and the Part B severity study classifies every tested
drift level UNSUPPORTED with detection only 6.7–13.3%. This is a fundamental
property of autoregressive predictors, not a tuning issue: no gate setting
can detect a fault the model's own history has already normalized away.

## Para 3 — Training seed sensitivity of combined_fault_load

The combined-fault result is sensitive to LSTM initialization: Part A
training-seed mean paired deltas are +0.310712 (seed 2026, unfavorable) versus
−1.555400 and −0.461026 (seeds 2027/2028, favorable), training seed accounts
for 73.9% of centered variation, and the hierarchical bootstrap 95% interval
is [−1.52, +0.29] rad/s, crossing zero. The causal chain runs through onset
timing: seed-dependent main-LSTM prediction quality during the fault onset
transient determines how fast CUSUM accumulates entry evidence, which
determines whether substitution engages early enough before the load step
arrives at 3.0 s. A realization whose transient predictions are slightly worse
triggers late or not at all, and the concurrent plant disturbance then hits
while feedback is still corrupted or unsubstituted — so the same detector and
thresholds produce opposite outcomes from initialization alone.

## Para 4 — C2 recovery gate degeneracy

The aux_recovery_gate (9.047693253 rad/s) is calibrated from the 99.9th
percentile of auxiliary residuals on clean validation data, which makes it
structurally inert during faults: the auxiliary estimator uses current-only
input with no speed channel, so a speed-sensor fault cannot corrupt it, its
residual never approaches a gate sized to clean-data extremes, and the
auxiliary-agreement clause of the recovery condition never binds (verified:
uniquely binding 0 times across all paired cases). C2 is therefore
behaviorally identical to C1 — all 55 paired C1/C2 cases match exactly — and
any comparison of C2 against C1 measures nothing. Recovery-gate analysis
should target the CUSUM discharge dynamics, not the auxiliary clause.
