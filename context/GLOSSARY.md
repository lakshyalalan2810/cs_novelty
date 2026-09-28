# Glossary

| Term | Meaning |
|---|---|
| B | Plain/frozen baseline controller used as a V4 anchor |
| C1 | Original V1 reliability-aware controller |
| C2 | C1 plus auxiliary agreement in recovery logic |
| C3 | V3 dual virtual-sensor arbitration controller |
| C4 | Development three-way attribution branch; NO_GO |
| CUSUM | Two-sided cumulative residual detector |
| DET | Detection/severity protocol family in the preregistered umbrella |
| EKF | Extended Kalman filter estimating current, speed, and load torque |
| Entry | Debounced reliability transition into the suspect state |
| Fault-free | V4 protocol with no injected sensor fault; load may still be part of other cells |
| Holm | Step-down multiple-comparison correction applied to H1–H9 |
| H1–H9 | Preregistered V4 confirmatory hypotheses |
| HIL | Hardware-in-the-loop; not performed here |
| ITT | Intent-to-treat analysis retaining all eligible pairs for the endpoint |
| LSTM | Long short-term memory recurrent neural network |
| MPC | Model predictive control |
| Nm | Newton-metre; load-torque unit |
| NO_GO | A development stage gate failed; branch is not promoted |
| RMSE | Root mean squared error |
| SLSQP | Sequential least-squares programming optimizer used by the MPC |
| Witness | Auxiliary estimate used to confirm a main residual abnormality |
| V1/V2/V3/V4 | Successive research protocol/architecture stages |
| main LSTM | Speed-history residual predictor using voltage and selected/measured speed |
| physical sensor | Simulated measured speed/current channel from the plant, before or after injected corruption |
| virtual sensor | Model-based speed estimate used when the physical speed signal is distrusted |
| reliability latch | Debounced suspect/active state entered after persistent abnormal residuals |
| false latch | Reliability entry before or without the intended injected sensor fault |
| fallback | Bounded safe feedback/PI behavior used when the normal virtual estimate is not trusted |
| holdout | Preserved evaluation evidence not used for tuning or threshold selection |
| development | Non-final exploratory or gate-setting evidence used to choose whether a branch proceeds |
| confirmatory | Frozen, preregistered comparison with declared endpoints and multiplicity |
| V4_full_aux | V4 full witness-gated variant using auxiliary LSTM witness |
| V4_full_ekf | V4 full witness-gated variant using EKF witness |
| clean-start | Pair retained only when no pre-event reliability entry occurred |
| false entry | Reliability entry caused by a condition treated as non-fault in the endpoint |
| fault-window RMSE | RMSE over the preregistered interval for a finite injected fault |
| pre-latch | Entry occurring before the injected event onset |
| recovery | First inactive state at or after fault end, right-censored when absent |
| training seed | Seed used to train model weights |
| simulation seed | Seed used for a runtime trajectory/fault cell |
| deferred | Planned in the umbrella but not executed in the exact confirmatory core |
| corrected artifact | Current summary produced after the hierarchical-bootstrap implementation correction |
| historical artifact | Preserved evidence that is superseded for current claims |

Related: [STATISTICS.md](STATISTICS.md), [ARCHITECTURE.md](ARCHITECTURE.md), [EXPERIMENTS.md](EXPERIMENTS.md).
