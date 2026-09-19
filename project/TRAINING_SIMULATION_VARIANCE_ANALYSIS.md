# Training × Simulation Variance Analysis

This retrospective Part A analysis uses the completed frozen 3-training-seed × 5-simulation-seed robustness matrix. For every cell, `Delta = fault_window_RMSE(C3_arbitration_MPC) - fault_window_RMSE(B_plain_MPC)`, so negative Delta favors C3.

## Design verification

The source matrix contains exactly `150` rows and yields `75` exact B/C3 pairs: 3 training seeds × 5 simulation seeds × 5 scenarios. Every scenario therefore contains exactly 15 paired Delta values, with no missing or duplicate training×simulation combinations. The frozen source SHA-256 is `8d9cb9dc8a01c2aab4df5f140e69b32e44cad4fbc49e5d7d1651a269fbaa0842`.

Training seeds are `2026, 2027, 2028`; simulation seeds are `49026, 49027, 49028, 49029, 49030`.

## Descriptive decomposition

For each scenario, the balanced additive decomposition is `Delta[t,s] = mu + alpha_training[t] + beta_simulation[s] + residual[t,s]`, where `alpha_training` is the training-seed mean minus the grand mean and `beta_simulation` is the simulation-seed mean minus the grand mean. The centered total sum of squares is decomposed exactly into training-seed SS, simulation-seed SS, and residual SS.

Because there is one observation per training×simulation cell, the residual contains training-by-simulation interaction/nonadditivity together with any remaining unexplained variation. These proportions are descriptive variance attribution and are not inferential ANOVA significance claims.

| Scenario | Grand mean Δ | Training SS proportion | Simulation SS proportion | Residual proportion | Training effect range | Simulation effect range | Favorable / 15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| combined_fault_load | -0.568571 | 0.739 | 0.122 | 0.138 | 1.866112 | 0.825270 | 11 / 15 |
| sensor_bias_5 | -1.640732 | 0.137 | 0.555 | 0.309 | 0.776146 | 1.593809 | 13 / 15 |
| sensor_dropout | -19.221775 | 0.147 | 0.573 | 0.280 | 1.036455 | 2.029381 | 15 / 15 |
| load_disturbance | +2.062685 | 0.334 | 0.543 | 0.123 | 2.005433 | 2.896723 | 0 / 15 |
| parameter_variation | +2.690691 | 0.001 | 0.641 | 0.358 | 0.268144 | 6.963600 | 1 / 15 |

## Combined fault + load detail

The combined-fault grand mean is `-0.568571`. Its centered variation is attributed descriptively as `73.9%` training seed, `12.2%` simulation seed, and `13.8%` residual interaction/unexplained variation.

| Training seed | Mean Δ | Favorable simulation seeds |
|---:|---:|---:|
| 2026 | +0.310712 | 2 / 5 |
| 2027 | -1.555400 | 5 / 5 |
| 2028 | -0.461026 | 4 / 5 |

| Simulation seed | Mean Δ across training seeds | Favorable training seeds |
|---:|---:|---:|
| 49026 | -0.572931 | 2 / 3 |
| 49027 | -0.157624 | 1 / 3 |
| 49028 | -0.982894 | 3 / 3 |
| 49029 | -0.295846 | 2 / 3 |
| 49030 | -0.833562 | 3 / 3 |

Across the 15 combined-fault cells, `11` are favorable, `4` are unfavorable, and `0` are exact ties. The training-seed effect range is `1.866112` RMSE and the simulation-seed effect range is `0.825270` RMSE.

## Scenario-level sign structure

| Scenario | Training-seed means favorable | Favorable cells | Unfavorable cells | Exact ties |
|---|---:|---:|---:|---:|
| combined_fault_load | 2 / 3 | 11 | 4 | 0 |
| sensor_bias_5 | 3 / 3 | 13 | 2 | 0 |
| sensor_dropout | 3 / 3 | 15 | 0 | 0 |
| load_disturbance | 0 / 3 | 0 | 12 | 3 |
| parameter_variation | 0 / 3 | 1 | 8 | 6 |

The sensor-dropout result is favorable in every paired cell. Sensor bias remains favorable in most cells and has a negative mean for all three training seeds. Load disturbance has a positive mean Delta for all three training seeds, preserving the previously documented C3 load-mismatch weakness. Parameter variation remains heterogeneous, with several exact ties and large simulation-seed effects.

## Hierarchical bootstrap

The bootstrap uses the fixed new Part A analysis RNG seed **20260917** and **20,000 replicates**. Each replicate samples the three training-seed clusters with replacement, then samples five simulation-seed Delta values with replacement inside each selected training cluster, and averages the sampled cluster means. This preserves the training→simulation hierarchy rather than treating all 15 cells as independent.

| Scenario | Point mean Δ | Descriptive 95% CI | Pr(bootstrap mean Δ < 0) |
|---|---:|---:|---:|
| combined_fault_load | -0.568571 | [-1.517776, +0.286116] | 0.8831 |
| sensor_bias_5 | -1.640732 | [-2.191698, -1.011005] | 1.0000 |
| sensor_dropout | -19.221775 | [-19.957548, -18.532747] | 1.0000 |
| load_disturbance | +2.062685 | [+0.951946, +3.201546] | 0.0000 |
| parameter_variation | +2.690691 | [+0.942957, +4.485360] | 0.0004 |

Only three training seeds are observed. The hierarchical bootstrap is therefore descriptive; the training-level uncertainty is poorly estimated and the 20,000 replicates improve Monte Carlo stability conditional on these three observed model realizations rather than creating additional independent training evidence.

## Outputs and scope

Machine-readable results are in `results/final_robustness/variance_decomposition.csv`, `variance_decomposition_summary.json`, and `variance_bootstrap.csv`. This analysis is retrospective and introduces no controller, model, calibration, MPC, training, or simulation changes.
