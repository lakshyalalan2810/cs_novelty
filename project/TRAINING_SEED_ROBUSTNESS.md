# Training-Seed Robustness Study

This document reports the completed training-seed robustness study using the frozen machine-readable evidence under `results/training_seed_robustness/`. The authoritative numerical sources are `analysis/analysis_summary.json` and its companion CSV tables; protocol and provenance are bound by `model_pair_manifest.json`, `configs/training_protocol_manifest.json`, and `simulation_seed_freeze.json`.

## 1. Motivation and study question

The original V3/C3 result was obtained from one trained main/auxiliary LSTM pair. This study asks whether the main closed-loop conclusions persist when only the model-training random seed changes while the dataset, split, architectures, training hyperparameters, calibration procedures, controller algorithms, scenarios, and closed-loop simulation seeds remain fixed.

The primary comparison is paired `C3_arbitration_MPC - B_plain_MPC` fault-window RMSE. A negative difference favors C3. The predeclared robustness rule requires the mean paired difference to remain negative for **each** training seed in `sensor_bias_5`, `sensor_dropout`, and `combined_fault_load`, with no numerical/safety failure and with the known load-disturbance weakness remaining qualitatively consistent. Failure of any required condition yields the exact verdict `TRAINING-SEED-SENSITIVE`.

No controller retuning, post-hoc threshold tuning from closed-loop outcomes, or training-seed selection was performed.

## 2. Exact frozen training and RNG protocol

The three matched model pairs are P2026, P2027, and P2028. P2026 reuses the frozen canonical weights and was **not retrained**. P2027 and P2028 were trained with the same frozen procedures, changing only `training_seed`.

For the main LSTM, the architecture is input size 2, hidden size 64, two LSTM layers, dropout 0.2, output size 1, and residual prediction enabled. Inputs are `[voltage, y_measured]`, target is `y_true`, sequence length is 20, batch size is 512, optimizer is Adam at `1e-3`, loss is MSE, maximum epochs are 60, patience is 8, and `min_delta=1e-6`. The RNG contract is `np.random.seed(training_seed)`, `torch.manual_seed(training_seed)`, deterministic PyTorch algorithms on CPU, and a dedicated `torch.Generator().manual_seed(training_seed)` for shuffled training batches. The PyTorch global seed governs weight initialization and inter-layer dropout.

For the auxiliary LSTM, the architecture is input size 2, hidden size 32, one LSTM layer, dropout 0, output size 1, using `[voltage, current] -> y_true`. Sequence length is 20, batch size 512, Adam `1e-3`, MSE, maximum 100 epochs, patience 10, and `min_delta=1e-6`. Its frozen RNG semantics are `np.random.seed(training_seed)` and `torch.manual_seed(training_seed)` before DataLoader/model construction, with `shuffle=True` and no separate DataLoader generator, matching the canonical auxiliary trainer.

Calibration uses only clean validation trajectories. The sensor monitor uses the frozen p99.9 instantaneous residual gate and the fixed CUSUM percentile grid `[90, 95, 97.5, 99, 99.5, 99.9]`, choosing the first candidate whose debounced validation false-alarm rate is at most 0.001, otherwise the final grid candidate. The frozen sensor persistence is enter count 3 and exit count 5. V3 arbitration uses p90 main/aux agreement, p99.9 post-blanking mismatch EWMA, p95 recovery EWMA, p99.9 auxiliary residual recovery gate, EWMA alpha 0.05, 1.0 s startup blanking, 50-sample mismatch recovery persistence, and auxiliary speed bounds `[0, 100]` rad/s.

## 3. Proof that dataset, splits, and hyperparameters were unchanged

The dataset is the same frozen `data/processed/dc_motor_lstm_dataset.npz`, SHA-256 `5f77f76468908656b35dabbc48e8aa20cc0583eae3724cc989278d94fe2bab26`, with dataset seed 2026 and window length 20. The sample counts remain 21,258 training sequences, 7,086 validation sequences, and 7,086 test sequences.

The frozen whole-trajectory split is unchanged:

- train: `[0, 1, 2, 4, 5, 6, 7, 10, 12, 13, 14, 18, 20, 21, 24, 26, 27, 29]`
- validation: `[3, 9, 11, 16, 19, 23]`
- test: `[8, 15, 17, 22, 25, 28]`

The protocol manifest binds the canonical source hashes for the original main-training notebook, auxiliary trainer, LSTM source, auxiliary-model source, reliability source, sensor-calibration notebook, data utilities, and V3 calibration script. It also binds the canonical P2026 main and auxiliary model/config hashes. `model_pair_manifest.json` separately binds every P2026/P2027/P2028 main weight, auxiliary weight, config, sensor calibration, V3 calibration, and pair configuration. Each pair used the same six validation trajectory IDs, 7,086 sensor-residual samples, 7,086 agreement/auxiliary-recovery samples, and 6,606 post-blanking EWMA samples.

Thus the experimental factor is the training seed and its resulting trained weights/calibrations; the dataset, split, architecture, loss, optimizer, learning rate, batch size, stopping rules, and calibration algorithms are frozen.

## 4. Per-seed main-model metrics

| Training seed | Test RMSE | Test MAE | Test R² | H5 RMSE | H10 RMSE | H15 RMSE | Best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 0.140873 | 0.113106 | 0.999906 | 0.210952 | 0.336461 | 0.487796 | 60 |
| 2027 | 0.254556 | 0.202115 | 0.999694 | 0.388099 | 0.642142 | 0.910245 | 6 |
| 2028 | 0.130747 | 0.105153 | 0.999919 | 0.198849 | 0.320091 | 0.463223 | 55 |

Across the three training seeds, test RMSE is `0.175392 ± 0.068745` sample SD with range `0.123809`; H15 RMSE is `0.620421 ± 0.251295` with range `0.447022`. Best epoch varies strongly (6, 55, 60), showing that identical training rules can reach materially different optimization trajectories even when held-out one-step accuracy remains high.

## 5. Per-seed auxiliary-model metrics

| Training seed | Test RMSE | Test MAE | Test R² | Best epoch | Mean per-trajectory RMSE | Per-trajectory RMSE SD | Per-trajectory RMSE range |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 3.437615 | 2.576372 | 0.944148 | 17 | 3.065411 | 1.704273 | 4.497397 |
| 2027 | 3.279101 | 2.330737 | 0.949180 | 23 | 2.770723 | 1.921125 | 5.204668 |
| 2028 | 3.425863 | 2.541861 | 0.944530 | 16 | 3.002812 | 1.806541 | 4.782387 |

Across seeds, auxiliary test RMSE is `3.380860 ± 0.088321` with range `0.158514`; auxiliary test R² is `0.945953 ± 0.002802`. The aggregate auxiliary test metrics are relatively stable, while trajectory-level error remains heterogeneous. Under closed-loop load disturbance, mean auxiliary virtual-sensor RMSE is 2.096792, 1.871151, and 2.124446 for P2026/P2027/P2028. Under parameter variation it rises to 7.008183, 9.416434, and 4.811122, showing that closed-loop transfer behavior is substantially more variable than the clean test-set aggregate metric alone suggests.

## 6. Per-seed calibration values, false-alarm rates, and variability

| Seed | Residual gate | CUSUM allowance | CUSUM threshold | Validation sensor alarm rate | Agreement threshold | Param-mismatch threshold | Recovery threshold | Aux recovery gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 0.937210 | 0.140386 | 1.569683 | 0.000706 | 4.341867 | 8.700959 | 5.048635 | 9.047693 |
| 2027 | 1.129457 | 0.173778 | 1.387323 | 0.242591 | 3.824803 | 10.007257 | 5.094724 | 10.477919 |
| 2028 | 0.923875 | 0.137605 | 2.261387 | 0.001129 | 4.138244 | 8.742745 | 4.794184 | 9.106845 |

The calibration variability is scientifically meaningful. Residual-gate CV is 0.1154; CUSUM-threshold CV is 0.2651; agreement-threshold CV is 0.0635; parameter-mismatch-threshold CV is 0.0811; recovery-threshold CV is 0.0325; and auxiliary-recovery-gate CV is 0.0848. The validation sensor-alarm-rate CV is 1.7126, dominated by P2027's `0.242591` alarm rate. P2027 therefore did not meet the 0.001 FAR target within the frozen candidate grid; the fixed rule falls through to its final percentile candidate. No new threshold was invented and no closed-loop result was used to retune it. P2028's alarm rate (`0.001129`) is also slightly above 0.001 under the same unchanged rule.

These differences are part of the training-seed sensitivity being measured: each model pair receives the same prospective clean-validation calibration algorithm, rather than sharing thresholds that were fitted to a different trained model.

## 7. Exact 150-run paired closed-loop design

The final matrix contains exactly `3 training seeds × 2 controllers × 5 scenarios × 5 simulation seeds = 150` complete rows. Each training seed contributes 50 runs.

The training seeds are 2026, 2027, and 2028. Controllers are `B_plain_MPC` and `C3_arbitration_MPC`. Scenarios are `sensor_bias_5`, `sensor_dropout`, `load_disturbance`, `combined_fault_load`, and `parameter_variation`. Every controller/training-pair comparison uses the same frozen simulation seeds `[49026, 49027, 49028, 49029, 49030]`.

The simulation-seed block was frozen before any closed-loop robustness evaluation. The provenance scan found no pre-study use of these selected seeds. The separate reserved block `[39026, 39027, 39028, 39029, 39030]` remained untouched. Pairing therefore controls simulation noise realization within each scenario while training-seed variation is evaluated independently.

## 8. Sensor-bias paired C3-B differences

For `sensor_bias_5`, all values below are paired fault-window RMSE differences `C3 - B`; negative values favor C3.

| Simulation seed | P2026 | P2027 | P2028 |
|---:|---:|---:|---:|
| 49026 | -2.362804034 | -2.572064896 | -1.992484866 |
| 49027 | +0.047035328 | -1.687251599 | -0.505711846 |
| 49028 | -2.547672348 | -1.674393441 | -1.937460606 |
| 49029 | +0.141949470 | -2.380086084 | -0.391175865 |
| 49030 | -2.501671868 | -2.320475185 | -1.926709621 |
| **Mean** | **-1.444632690** | **-2.126854241** | **-1.350708561** |

The mean effect is favorable for all three training seeds. P2026 has 3/5 favorable simulation pairs, while P2027 and P2028 have 5/5 each. Thus the sensor-bias component of the headline robustness criterion passes.

## 9. Sensor-dropout paired C3-B differences

| Simulation seed | P2026 | P2027 | P2028 |
|---:|---:|---:|---:|
| 49026 | -20.807977226 | -19.152741990 | -20.255940978 |
| 49027 | -17.358249919 | -18.195643756 | -18.574624733 |
| 49028 | -20.745892369 | -18.227272912 | -20.361755418 |
| 49029 | -17.531900418 | -18.676005719 | -18.769583121 |
| 49030 | -20.341762558 | -18.927622406 | -20.399656350 |
| **Mean** | **-19.357156498** | **-18.635857356** | **-19.672312120** |

Every one of the 15 paired dropout comparisons favors C3. The dropout conclusion is therefore stable across the tested training seeds and is the strongest repeated improvement in this study.

## 10. Combined sensor-fault + load paired C3-B differences

| Simulation seed | P2026 | P2027 | P2028 |
|---:|---:|---:|---:|
| 49026 | +0.059540105 | -1.388319936 | -0.390012182 |
| 49027 | +1.321263986 | -1.813594759 | +0.019459251 |
| 49028 | -0.410283368 | -1.917476326 | -0.620922009 |
| 49029 | +0.901698244 | -1.338936278 | -0.450299104 |
| 49030 | -0.318658320 | -1.318670664 | -0.863358201 |
| **Mean** | **+0.310712129** | **-1.555399593** | **-0.461026449** |

This is the decisive headline result. **P2026 combined mean is +0.310712, so it fails the required negative-mean condition.** Only 2/5 P2026 pairs favor C3. P2027 is favorable in 5/5 pairs with mean `-1.555400`; P2028 is favorable in 4/5 pairs with mean `-0.461026`. Therefore P2027 and P2028 are favorable, but the required all-training-seeds criterion is not satisfied.

## 11. Load-disturbance false entries, substitution, and tracking penalty

The load-disturbance scenario reproduces the known V3 weakness under every training pair: plant/load mismatch can trigger the residual reliability mechanism even though the physical speed sensor itself is healthy.

| Training seed | Mean C3-B tracking penalty | Total false reliability entries | Mean C3 substitution fraction |
|---:|---:|---:|---:|
| 2026 | +2.939923 | 4 | 0.531115 |
| 2027 | +0.934490 | 5 | 0.463894 |
| 2028 | +2.313643 | 4 | 0.529451 |

For P2026 and P2028, four of the five runs contain one false entry and the remaining run has none. P2027 contains five false entries across five runs, with simulation seed 49030 contributing two entries. Mean C3 substitution remains large at approximately 46-53%, and the tracking penalty is positive for all three training seeds. This satisfies the verdict rule's requirement that the frozen load-disturbance weakness remain qualitatively consistent; it does not make the weakness desirable.

## 12. Parameter-variation behavior

| Training seed | B fault-window RMSE mean | C3 fault-window RMSE mean | Mean C3-B | C3 substitution fraction | Event aux fraction | Event fallback fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 0.670705 | 3.350949 | +2.680244 | 0.335774 | 0.400000 | 0.000664 |
| 2027 | 1.273260 | 3.835102 | +2.561842 | 0.408319 | 0.400000 | 0.149502 |
| 2028 | 0.579862 | 3.409848 | +2.829986 | 0.335441 | 0.400000 | 0.000000 |

Parameter variation is unfavorable on average for C3 under all three training pairs, with substantial within-seed variation. P2026 paired differences are `[0, 6.704779, 0.001445, 6.694998, 0]`; P2027 values are `[0, 6.670504, 6.575757, 0, -0.437053]`; P2028 values are `[0, 7.078464, 0.000577, 7.070889, 0]`. The study's frozen verdict rule treats this scenario descriptively unless it creates a numerical or safety failure; no post-hoc parameter-variation threshold was introduced.

The auxiliary transfer metrics also show why this scenario is challenging: mean auxiliary virtual RMSE under parameter variation is 7.008183 for P2026, 9.416434 for P2027, and 4.811122 for P2028, with large within-seed dispersion. The physical/main/auxiliary arbitration outcome is therefore sensitive to a mismatch regime that was not represented by the nominal clean-validation calibration population.

## 13. Hierarchical bootstrap analysis

The analysis used the fixed analysis seed **20260916** and **20,000 bootstrap replicates**. Each replicate first samples the three training-seed clusters with replacement; within each selected training-seed cluster it then samples the five paired simulation-seed C3-B differences with replacement. Cluster means are averaged across the sampled training-seed clusters. The reported intervals are descriptive percentile 95% CIs.

| Scenario | Point estimate: mean of training-seed means | Descriptive 95% CI |
|---|---:|---:|
| sensor_bias_5 | -1.640732 | [-2.181868, -1.023271] |
| sensor_dropout | -19.221775 | [-19.969597, -18.540608] |
| load_disturbance | +2.062685 | [+0.951591, +3.204731] |
| combined_fault_load | -0.568571 | [-1.528067, +0.303891] |
| parameter_variation | +2.690691 | [+0.919036, +4.490970] |

The combined-fault interval crosses zero, consistent with the sign reversal at the individual training-seed level. The bias/dropout intervals remain negative, while load disturbance and parameter variation remain positive on this descriptive bootstrap scale.

Only **three training seeds** are available. The bootstrap therefore describes the observed hierarchical variation; it does not support high-precision population inference about all possible weight initializations or training trajectories. The 20,000 replicates improve Monte Carlo stability conditional on these three observed clusters but cannot replace additional independent training seeds.

## 14. Safety, numerical integrity, provenance, and study limitations

All 150 runs completed. For P2026, P2027, and P2028 separately, each 50-run block contains zero optimizer failures, zero main-prediction failures, zero auxiliary-prediction failures, zero voltage violations, zero slew violations, zero nonfinite events, and zero incomplete runs. Across all 150 runs every required numerical/safety total is therefore zero. Main and auxiliary required metrics are finite, and model/calibration/run binding was verified.

The selected simulation seeds 49026-49030 were prospectively frozen, with the provenance audit finding no pre-study use. The same five simulation seeds were then used for every training pair and both controllers. P2026 was retained as the canonical seed-2026 pair; P2027 and P2028 were the predeclared additional training seeds. **No training seed was accepted, rejected, replaced, or chosen based on the resulting closed-loop performance.** Likewise, no controller parameter or calibration rule was retuned after seeing the 150-run matrix.

The study still has important scope limits. Three training seeds provide direct evidence of failure of the preregistered across-training-seed robustness criterion but are too few to characterize the full distribution of possible training outcomes. The models share one frozen dataset and one frozen train/validation/test split, so dataset/split uncertainty is outside this experiment. The five simulation seeds quantify paired closed-loop variability within the selected block, not all possible disturbance/noise realizations. Because canonical P2026 itself is unfavorable for `combined_fault_load` on this newly frozen common simulation block, the exact `TRAINING-SEED-SENSITIVE` classification should not be interpreted as proving that training initialization alone caused the reversal; it establishes that the headline conclusion does not survive the full preregistered common-block test across all three trained pairs. The calibration procedure is deliberately model-specific, so observed differences combine changes in learned weights with the prospectively recalculated thresholds those weights induce. Finally, the high P2027 clean-validation sensor alarm rate is an observed result of the frozen calibration rule and should be treated as evidence of instability in this model/calibration chain rather than repaired post hoc inside this study.

## 15. Exact verdict

**`TRAINING-SEED-SENSITIVE`**

The frozen rule for `ROBUST ACROSS TRAINING SEEDS` is not met because `combined_fault_load` does not have a negative mean paired C3-B fault-window RMSE for every training seed. P2026 has **+0.310712129** (headline failure), while P2027 and P2028 are favorable at **-1.555399593** and **-0.461026449** respectively. Sensor bias and sensor dropout remain favorable on average for all three training seeds, all numerical/safety checks pass, and the known load-disturbance weakness remains qualitatively consistent. Those facts preserve important parts of the original C3 interpretation, but they do not override the predeclared all-seed combined-fault requirement.

Accordingly, the completed study shows that the C3 benefit is robust for sensor dropout and favorable on average for the 5% bias across the tested training seeds, but the combined fault+load advantage is dependent on the trained model pair. This conclusion follows the frozen rule exactly; there was **no retuning and no seed selection** after observing the results.
