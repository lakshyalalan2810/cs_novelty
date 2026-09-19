# EKF observer preregistration

Status: **pre-implementation / pre-evaluation**. This document freezes the
classical-observer protocol before any production EKF, EKF closed-loop
controller, Q/R calibration result, or EKF fault-control result exists.

The planned comparator is a single augmented-state extended Kalman filter
(EKF). It is intended to answer whether a physics-based observer using the same
causal voltage/current information as the frozen auxiliary LSTM is a useful
classical baseline. This preregistration does not authorize changes to frozen
V3, C1/C2/C3, C4-v1, their thresholds, model weights, scientific artifacts, or
final-holdout evidence.

## 1. Repository contract frozen for the observer

The repository motor model is a nonlinear permanent-magnet DC motor with

\[
x = [i,\ \omega,\ T_L]^T, \qquad u=V, \qquad y=i_{meas}.
\]

The nominal parameter values are:

| Parameter | Value |
| --- | ---: |
| Armature resistance, \(R\) | 2.0 ohm |
| Armature inductance, \(L\) | 0.5 H |
| Back-EMF constant, \(K_b\) | 0.1 V s/rad |
| Torque constant, \(K_t\) | 0.1 N m/A |
| Rotor inertia, \(J\) | 0.01 kg m² |
| Viscous friction, \(B\) | 0.002 N m s/rad |
| Smooth Coulomb friction, \(F_c\) | 0.02 N m |
| Friction smoothing speed, \(\omega_s\) | 0.1 rad/s |

The exact deterministic augmented dynamics to be implemented in the next
phase are

\[
\dot i = \frac{V-Ri-K_b\omega}{L},
\]

\[
\dot\omega =
\frac{K_t i-B\omega-T_L-F_c\tanh(\omega/\omega_s)}{J},
\]

\[
\dot T_L=0.
\]

The future process model will represent load variation through process noise
on the \(T_L\) state, making it a slow random walk. The deterministic equation
remains \(\dot T_L=0\).

The measurement function is exactly

\[
h(x)=i, \qquad C=\frac{\partial h}{\partial x}=[1,0,0].
\]

The repository friction law must be retained exactly:

\[
\tau_f(\omega)=F_c\tanh(\omega/\omega_s),
\]

with exact derivative

\[
\frac{d\tau_f}{d\omega}
=\frac{F_c}{\omega_s}\left[1-\tanh^2(\omega/\omega_s)\right].
\]

No generic sign-function or linearized Coulomb-friction replacement is
permitted.

The continuous state Jacobian is therefore frozen as

\[
A_c =
\begin{bmatrix}
-R/L & -K_b/L & 0 \\
K_t/J & -(B+d\tau_f/d\omega)/J & -1/J \\
0 & 0 & 0
\end{bmatrix}.
\]

The plant step is 10 ms. The saved identification dataset was produced with
`solve_ivp` using 10 ms output spacing and `max_step=0.01`. The frozen V3/C4
closed-loop evaluators use an explicit RK4 step at exactly 10 ms. The future
closed-loop EKF comparator will therefore use the same 10 ms time base and a
nonlinear RK4 prediction consistent with the frozen closed-loop plant. The
prediction Jacobian must be either the analytic tangent Jacobian of that RK4
map or a numerically differentiated RK4 map validated against it. The analysis
script already implements and validates the analytic tangent form.

## 2. Causal information contract and fairness

The future EKF may consume only:

- applied armature voltage \(V\), causally available at the same sample timing
  used by the auxiliary LSTM; and
- armature current \(i_{meas}\).

The EKF must not consume measured speed, `y_true`, true speed, fault labels,
fault flags, scenario identity, or any future sample. Offline `y_true` is
allowed only after an estimate has been produced, for scoring.

The current repository does **not** contain current-measurement noise in the
saved training/validation/test trajectories or in the V3/C4 closed-loop
simulation. `current` is the exact simulated armature-current state and is fed
directly to the auxiliary LSTM. The existing 0.25 rad/s Gaussian measurement
noise applies to the speed sensor, not to current.

Therefore the primary auxiliary-LSTM-versus-EKF comparison must preserve this
input contract: both estimators receive the same causal applied voltage and
the same noiseless simulated current. No current noise may be injected only
for the EKF. A separate, later current-sensor sensitivity experiment is
predeclared below and must apply the same corruption to both estimators.

## 3. Observability result that permits implementation to proceed

For the local continuous model,

\[
O=[C;\ CA_c;\ CA_c^2]
\]

has determinant

\[
\det O=-\frac{K_b^2}{L^2J}.
\]

Thus the three-state augmented system is locally full rank for finite positive
\(L,J\) whenever \(K_b\neq0\). Physically, speed affects current through
back-EMF; load torque affects speed acceleration and therefore reaches the
current measurement one dynamic step later. Setting \(K_b=0\) removes that
path and the synthetic structural test loses full rank.

This algebraic result is not treated as proof of good practical
observability. The pre-implementation numerical analysis used the exact
tangent Jacobian of the 10 ms RK4 map and finite-horizon local observability
matrices over the saved healthy train/validation operating population. State
columns were dimensionlessly scaled by their saved train/validation standard
deviations because raw current, speed, and torque units make a raw condition
number physically misleading.

The primary horizon is the existing 20-sample history window, 0.2 s. Across
27,648 saved train/validation windows, all were rank 3. The nominal scaled
condition number had median about 919, range about 711 to 1,182; the smallest
scaled singular value had median about 0.00354 and minimum about 0.00274.
This is noticeably ill-conditioned, but it is not singular or catastrophic.
Over the predeclared 0.5 s sensitivity horizon, median condition number falls
to about 110 and the minimum singular value rises substantially, showing that
information accumulates with time.

The weak state is load torque. On the 0.2 s horizon, the median normalized
load-state column sensitivity is only about 2.0% of the speed-state column
sensitivity. On 0.5 s it rises to about 5.6%. This means short-horizon load
estimation is expected to be covariance/process-model dominated much more
often than speed estimation. This weakness must be reported if it appears in
the future EKF results; it may not be hidden by replacing the observer or
adding another measurement after results are seen.

Startup and very-low-speed windows contain the worst observed conditioning,
with condition number up to about 1,182. The cause is the strong slope of the
repository smooth-friction term around zero speed. Low current, low voltage,
low electrical drive, voltage saturation, steady operation, and load-change
neighborhoods all remained rank 3 in the saved population. High speed does not
change the algebraic back-EMF coefficient, so it does not create new
observability merely because back-EMF magnitude is larger; high-speed windows
are mainly more uniform because they lie away from the near-zero smooth-
friction transition. With the current project assumption of noiseless current,
there is no measured-current SNR benefit to claim at high speed.

The frozen V3/C4 parameter-variation definition is also structurally full
rank. Under the shifted local model (R ×1.20, L ×0.85, Kb ×1.15, Kt ×0.85,
J ×1.20, B ×1.20, Coulomb friction ×1.20), the 0.2 s median scaled condition
number is about 748 and every analyzed window remains rank 3. This means the
parameter-shift case does not fail because the shifted physical system becomes
unobservable. It does **not** establish robustness of a nominal-model EKF to
model mismatch. The future parameter-transfer experiment is therefore still
required, and no Q/R value may be selected using that evaluation outcome.

## 4. Future EKF implementation requirements

The next implementation phase must create one observer only, with state
`[current, omega, load_torque]`, input `voltage`, and measurement `current`.
No UKF, observer ensemble, learned correction, speed-sensor fusion, or new
attribution architecture may be introduced in that phase.

The EKF implementation must use nonlinear state prediction and an analytic or
validated numerical state-transition Jacobian. Covariance correction must use
the Joseph form

\[
P^+=(I-KC)P^-(I-KC)^T + KRK^T.
\]

After prediction and update, covariance must be explicitly symmetrized as
`0.5*(P + P.T)`. Every state, innovation covariance, Kalman gain, and
covariance must be finite. A numerical PSD check must be performed with a
small declared floating-point tolerance. A covariance or state that becomes
nonfinite, materially asymmetric, or non-PSD is a divergence/failure event;
the implementation may not silently reset from truth or the speed sensor.

Low-observability behavior is frozen as follows. The EKF continues ordinary
prediction and current measurement updates; it does not switch to a speed
measurement or retune Q/R online. A diagnostic weak-observability flag may be
computed from the already-established development thresholds (0.2 s condition
number above the development p99 or smallest singular value below the
development p01), but that flag is diagnostic only in the first comparator.
If covariance becomes invalid, the run is counted as a numerical failure. A
future current-dropout sensitivity run may use predict-only propagation while
current is unavailable, with no hidden speed correction.

## 5. Q/R calibration protocol

No Q or R value is selected in this task.

The future EKF uses diagonal discrete process covariance

\[
Q=\operatorname{diag}(q_i,q_\omega,q_T),
\]

and scalar current measurement covariance

\[
R=r_i.
\]

The calibration population is restricted to saved training and validation
trajectories. Test trajectories, V3 final-holdout outcomes, C4 development
outcomes, C4 candidate final seeds, fault labels, and future EKF fault-control
results may not influence Q/R selection.

Because the primary project current is noiseless, `R` is an assumed EKF
modeling covariance representing numerical/model mismatch; it does not imply
that random current noise is injected into the comparison. This distinction
must be stated in the future methods/results.

The search is bounded to **27 candidate covariance configurations**. Let
`s_i`, `s_omega`, and `s_T` be the training-only standard deviations of current,
speed, and load torque. Define a dimensionless dynamic process scale
`q_dyn`, a load random-walk multiplier `m_T`, and a dimensionless current
measurement scale `r`:

- `q_dyn ∈ {1e-8, 1e-6, 1e-4}`;
- `m_T ∈ {1e-2, 1e-1, 1}`;
- `r ∈ {1e-8, 1e-6, 1e-4}`.

For each candidate,

\[
Q=\operatorname{diag}(s_i^2q_{dyn},\ s_\omega^2q_{dyn},\ s_T^2q_{dyn}m_T),
\]

\[
R=s_i^2r.
\]

This scale-aware 3×3×3 protocol prevents a large unrestricted four-parameter
search. The grid itself may not be expanded after viewing fault-control or
final-holdout outcomes. If every candidate is numerically invalid, the result
is a failed calibration and must be reported rather than widening the search
post hoc.

Selection is lexicographic and uses validation trajectories only:

1. reject any candidate with nonfinite estimates/covariance, PSD failures, or
   divergence;
2. among stable candidates, minimize pooled validation speed-estimation RMSE;
3. if candidates are within 1% in RMSE, prefer lower MAE, then smaller absolute
   speed bias;
4. use current innovation/NIS behavior and load-estimate plausibility as
   diagnostics/tie breakers, not as permission to sacrifice speed accuracy or
   stability. Because current is noiseless in the primary data, NIS is a model
   consistency diagnostic rather than a claim of statistically correct sensor
   noise calibration.

Load estimates and innovations must be saved for audit. No scenario-specific
or fault-specific Q/R is permitted.

## 6. Initialization protocol

Initialization is frozen before fault evaluation:

- `i0 = first causal measured current`;
- `omega0 = 0 rad/s`;
- `T_L0 = 0 N m`;
- `P0 = diag(s_i_train^2, s_omega_train^2, s_T_train^2)`, where the scales are
  computed once from the saved training trajectories only.

The saved motor trajectories all begin with true current 0 and true speed 0,
and the frozen V3/C4 closed-loop evaluator also initializes motor current and
speed at zero. Therefore `omega0=0` is appropriate for every currently saved
trajectory and existing closed-loop scenario without truth leakage.

Saved dataset trajectories may begin with nonzero load torque even though
current and speed begin at zero; the true initial load is not available to the
observer, so `T_L0=0` remains the causal initialization. The load random-walk
state must learn it from subsequent voltage/current dynamics.

If a later study introduces trajectories with unknown nonzero initial speed,
that study requires a separately preregistered causal initializer based only
on voltage/current history. It may not read the speed sensor or true initial
speed. Such an initializer is outside this experiment and is not implemented
here.

## 7. Future comparison layer A: estimator only

The primary estimator comparison is the frozen auxiliary LSTM versus the EKF.
Both receive identical causal `[V, current]` information. The auxiliary LSTM's
existing 20-sample history means the primary paired error comparison begins at
sample 20 (0.2 s); startup behavior from 0 to 0.2 s is reported separately so
the EKF does not gain an unfair scoring advantage by producing estimates
earlier.

The estimator-only evaluation must cover:

- clean held-out test trajectories;
- startup;
- load disturbance/load changes;
- frozen parameter variation;
- low/high current regions;
- low/high speed regions;
- low/high acceleration regions;
- voltage saturation; and
- the preregistered weak-observability regions.

Metrics are speed RMSE, MAE, signed bias, finite/divergence rate, and runtime.
Offline true speed is used for scoring only. Parameter-transfer RMSE is a
primary endpoint and may not be used to retune Q/R.

## 8. Future comparison layer B: closed-loop comparator

The future closed-loop controller is a **new separate controller** named
`EKF_virtual_MPC`. It must not modify C3 or the frozen V3 sensor-reliability
monitor.

The controller uses the same frozen V3 reliability monitor. While the physical
speed sensor is trusted, the existing trusted physical feedback remains in
use. When the unchanged monitor requests substitution, feedback becomes the
EKF speed estimate. No C4 attribution layer is added to this comparator.

The primary closed-loop scenarios are frozen as:

- `sensor_bias_5`;
- `sensor_dropout`;
- `load_disturbance`;
- `combined_fault_load`; and
- `parameter_variation`.

The future paired simulation seed family must be declared before the first EKF
closed-loop run and may not use C4 candidate final seeds 39026–39030 unless a
separate approved protocol explicitly makes them available. No new final
holdout is run in the present task.

## 9. Future comparison layer C: current-sensor sensitivity

After the primary fair comparison is frozen, run one small boundary study that
applies identical current-channel corruption to the auxiliary LSTM and EKF:

- current bias;
- current dropout; and
- simultaneous speed/current corruption.

The corruption magnitudes/timing must be frozen before this boundary study.
No estimator may receive an uncorrupted private copy of current. Current
dropout handling must be explicit: the EKF may use predict-only propagation
when current is unavailable; the auxiliary LSTM must receive the same missing
channel semantics defined in that later sensitivity preregistration. This
study is a robustness boundary test and cannot be used to retune primary Q/R.

## 10. Primary endpoints frozen before EKF results

The future experiment has exactly these primary endpoints:

1. `sensor_bias_5` fault-window tracking RMSE;
2. `sensor_dropout` fault-window tracking RMSE;
3. `combined_fault_load` fault-window tracking RMSE;
4. `load_disturbance` false-substitution count/fraction and tracking penalty;
5. parameter-transfer estimator speed RMSE;
6. numerical stability / finite-sample rate; and
7. runtime overhead.

These endpoints may not be added, removed, or replaced after EKF results are
seen. Secondary descriptive metrics such as MAE, bias, innovation statistics,
load-estimate traces, and weak-region stratification may be reported, but they
do not replace the primary endpoints.

## 11. EKF role decision gate

The EKF is reported whether it wins or loses.

It is **BASELINE ONLY** if it is causal, scientifically fair, and numerically
stable but does not satisfy every successor criterion below.

It is a **POTENTIAL SUCCESSOR / FOLLOW-UP** only if all of the following are
true:

- the EKF is finite at every evaluated sample;
- it creates zero new safety failures;
- isolated `sensor_bias_5` and `sensor_dropout` fault-window RMSE each worsen
  by no more than 5% versus frozen C3;
- `combined_fault_load` fault-window RMSE improves by at least 10% versus C3;
- the `load_disturbance` tracking penalty relative to plain MPC is reduced by
  at least 10% versus C3, with false-substitution count/fraction non-worse;
- each required improvement occurs in at least 4 of 5 paired simulation
  seeds; and
- Q/R was selected only by the preregistered training/validation calibration
  protocol, with no evaluation-specific retuning.

Failure of any one condition leaves the EKF in the baseline-only role. The
gate may not be changed after results exist.

## 12. Learned-weight robustness study preregistration

The existing frozen main/auxiliary model pair was trained with seed **2026**.
Two additional training seeds are predeclared now:

- **2027**
- **2028**

Repository search before this preregistration found neither seed in existing
evaluation/final protocols. These are training seeds only. They must not be
selected or discarded based on performance.

For both additional seeds, keep fixed:

- the saved dataset and whole-trajectory train/validation/test split;
- main and auxiliary architectures;
- feature definitions (`[V, measured speed]` for the frozen main-model
  training protocol and `[V, current]` for the auxiliary estimator);
- normalization fit procedure and split provenance;
- sequence length 20;
- batch size 512;
- Adam optimizer, matching both existing training protocols;
- learning rate 0.001;
- main-model maximum epochs 60 and patience 8;
- auxiliary-model maximum epochs 100 and patience 10;
- existing minimum-delta settings, dropout, hidden sizes, and every other
  hyperparameter.

The purpose is learned-weight robustness, not seed selection. Both new model
pairs must be reported. No model is trained in the present task.

## 13. Reporting requirements for the next phase

The next phase must save the chosen Q/R candidate and the complete 27-candidate
validation table before any closed-loop fault evaluation. It must record the
initialization values, source hashes, dataset hash, model hashes, estimator
runtime, finite/PSD failures, innovation diagnostics, and load-state traces.

Estimator-only results must precede closed-loop interpretation. If the load
state remains weak or covariance dominated, report that explicitly. If the
EKF loses to the auxiliary LSTM or frozen V3/C3, retain it as the classical
baseline rather than retuning it into a favorable result.

This preregistration authorizes no production EKF code and no EKF fault-control
experiment. Those begin only after the current observability analysis and
repository integrity verification are accepted.
