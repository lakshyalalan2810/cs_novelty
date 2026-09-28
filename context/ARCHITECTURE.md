# Architecture

## System overview

The research system is a simulated nonlinear PMDC motor controlled by a multirate LSTM-MPC loop. A main speed-history LSTM predicts one-step speed. V3 uses an auxiliary current-informed LSTM as a second virtual sensor and arbitrates feedback. V4 keeps the main model as the substitute feedback and uses an auxiliary LSTM or EKF only as a witness for the reliability decision.

~~~text
                 reference r[k]
                       |
                       v
  +-----------+   +-----------+   +----------------------+
  | PMDC      |-->| measured  |-->| main LSTM predictor  |
  | plant     |   | speed y_m |   | [u, speed history]  |
  +-----+-----+   +-----+-----+   +----------+-----------+
        |               |                    |
        |               |                    v
        |               |              y_main, residual_main
        |               |
        |               +----------------------------+
        |                                            |
        v                                            v
  current I_m ---> auxiliary LSTM [u,I] ---> y_aux  |
        |                                            |
        +--------> augmented EKF [u,I] ---> omega_EKF|
                                                     v
                                      +-------------------------+
                                      | V3 arbitrator or V4     |
                                      | reliability/witness     |
                                      +------------+------------+
                                                   |
                                                   v
                                             feedback y_fb
                                                   |
                                                   v
                                  +----------------------------+
                                  | H20/Nc2 LSTM-MPC or PI     |
                                  | blocks (5,15), bounds     |
                                  +------------+---------------+
                                               |
                                               v
                                          applied voltage u
                                               |
                                               +----> plant
~~~

## Plant

The state is x = [current, angular speed]. The nominal equations are:

    di/dt = (u - R i - Kb omega) / L
    domega/dt = (Kt i - B omega - load - Tc tanh(omega / vs)) / J

Nominal parameters in project/src/motor_model.py:

- R = 2 ohm
- L = 0.5 H
- Kb = Kt = 0.1
- J = 0.01
- B = 0.002
- Coulomb friction Tc = 0.02
- friction smoothing vs = 0.1

V4 uses a fixed 0.01 s model step and a 0.05 s control stride. A normal V4 cell covers 6 s (601 grid samples); recovery cells use 10 s (1001 samples). The controller therefore targets a 20 Hz control cadence, but actual solver timing does not establish reliable end-to-end 20 Hz execution.

## Predictors

### Main LSTM

project/src/lstm_model.py defines a two-layer LSTM with hidden size 64 and dropout 0.2. Inputs are [voltage, selected/measured speed] over a 20-sample history. Its output is residual-style: a linear prediction plus the last normalized measured speed. Training targets are one-step-ahead true speed.

### Auxiliary LSTM

project/src/auxiliary_sensor_model.py defines a single-layer LSTM with hidden size 32 and inputs [voltage, measured current]. It never reads measured speed. It predicts physical-unit speed and is intended to provide input-diverse evidence. V4 uses it as a witness, not as direct feedback.

### EKF

project/src/ekf_observer.py estimates [current, angular speed, load torque] from applied voltage and measured current. It uses RK4 prediction, an analytic tangent Jacobian, current-only measurement, Joseph covariance update, and PSD/finite checks. In V4 it supplies witness residual only. The separate EKF closed-loop comparator follows a different continuous-estimator path and is not identical to the V4 EKF witness.

## Reliability and selection

### V3

project/src/reliability.py contains the frozen residual monitor and DualVirtualSensorArbitrator. Selection is:

~~~text
trusted physical speed
    -> PHYSICAL
otherwise main and auxiliary agree
    -> MAIN_VIRTUAL
otherwise auxiliary is finite, trusted, bounded
    -> AUX_VIRTUAL
otherwise
    -> bounded fallback / last trusted physical value
~~~

The monitor uses an instantaneous residual gate, two-sided CUSUM, entry persistence 3, recovery persistence 5, and an auxiliary recovery gate. V3’s central limitation is that plant/model mismatch, especially a load disturbance, can look like a sensor fault.

### V4

project/src/reliability_v4.py keeps the frozen monitor behavior while adding configurable CUSUM clamping, startup warmup, and a witness gate. The primary V4 configuration uses:

- CUSUM clamp at 2 times threshold;
- 150 monitor-sample warmup with sensor trusted;
- main residual AND witness residual for debounced latching;
- recovery boost zero in the primary run;
- main-model substitution whenever substitution is active.

The witness is either the auxiliary LSTM or EKF. Missing/nonfinite witness values fail safe to the main-only decision. V4 does not claim causal fault isolation.

## MPC

project/src/mpc.py provides an SLSQP controller with differentiable rollout, voltage bounds 0–12 V, a maximum command move of 2 V per control update, warm start, and PI fallback. The V4 caller fixes:

- horizon H = 20;
- two decisions;
- move blocks (5, 15);
- tracking weight 1.0;
- move weight 0.5;
- maximum 8 iterations;
- tolerance 0.05.

The generic class defaults are not sufficient to identify the V4 experiment; the caller configuration in project/scripts/v4_closed_loop.py is authoritative for V4.

## V4 variants

The definitions in project/scripts/v4_detector_configs.py include frozen_baseline, clamp_only, gating_only, clamp_gating, witness_only, and full. Confirmatory H1–H9 use the full auxiliary/EKF variants, with C3 and B as comparators/anchors. The variants do not turn the witness into a feedback source.

## Architecture boundaries

- C4’s ThreeWayConsistencyAttributor is a separate development path, not part of V4.
- V4 auxiliary/EKF outputs are witnesses, not selected fallback feedback.
- Current-channel health is an assumption; current corruption degraded both C3 and EKF in the boundary study.
- RollingPredictionQuality and adaptation_region exist in the MPC module but are not called by the V4 closed-loop engine.

Related: [DATA_FLOW.md](DATA_FLOW.md), [CODE_CONNECTIONS.md](CODE_CONNECTIONS.md), [GLOSSARY.md](GLOSSARY.md).

