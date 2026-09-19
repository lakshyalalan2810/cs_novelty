# V3 Dual Virtual Sensor Arbitration — Frozen 11-Scenario Evidence

## Scope and provenance

This phase completes evidence only. It changes no controller, reliability, arbitration, MPC, threshold, or trained model. Seeds `19026–19030` remain development evidence; the untouched final holdout is `29026–29030`.

The canonical matrix contains 275 runs: 5 controllers × 11 scenarios × 5 paired seeds. The original 200 rows were retained byte-for-byte in `v3_final_holdout_runs.csv`; only 75 nominal/reference runs were added.

| Frozen artifact | SHA256 |
|---|---|
| `main_model` | `d1f4178199f682560f164ccac1d56816aae341eb05fbed6ac7ad73a4e2056297` |
| `auxiliary_model` | `619f677ccc63c6a40c863858d8eee4175f72b1770dc6c8ba202e7b92c2768480` |
| `v3_arbitration_config` | `7b4f86934ece8d426148ff968b0152a5fac6cfa321615b229bfd35c63a179b35` |
| `v3_calibration` | `1dfcedf45110971e776c9c9ec6639b556b2c058d3a0535d9a69f5e23be99b1c8` |
| `original_200_runs` | `160a114a77709d73b3503921dd672000a05bfdc918838b0d13619ac8a0fbcddd` |

## Frozen calibration and verifier portability

Frozen thresholds are agreement `4.34186744689941`, parameter mismatch `8.70095886233045`, recovery EWMA `5.04863549523287`, auxiliary recovery `9.04769325256348`, and physical speed envelope `[0.0, 100.0]` rad/s.

Model-derived reconstruction checks previously used `rtol=0, atol=1e-9`; they now use `rtol=1e-06, atol=1e-05`. This permits insignificant PyTorch/BLAS rounding variation while exact hashes, identities, matrix structure, event counts, and constraints remain exact. The maximum observed calibration reconstruction difference was `0`.

## Nominal and reference-transient results for C3

Values are means across the five final seeds.

| Scenario | RMSE | MAE | Substitution fraction | Reliability entries | Source switches | Physical fraction | Main fraction | Auxiliary fraction | Fallback fraction | Control effort Σu² | Optimizer failures | Voltage violations | Slew violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nominal_tracking` | 10.807836 | 4.487264 | 0.000998 | 0 | 1.200 | 0.999002 | 0.000998 | 0.000000 | 0.000000 | 6210.675196 | 0 | 0 | 0 |
| `step_reference` | 7.075731 | 3.782032 | 0.000998 | 0 | 1.200 | 0.999002 | 0.000998 | 0.000000 | 0.000000 | 7180.631308 | 0 | 0 | 0 |
| `changing_reference` | 7.218066 | 4.235695 | 0.000666 | 0 | 0.800 | 0.999334 | 0.000666 | 0.000000 | 0.000000 | 6597.305053 | 0 | 0 | 0 |

### B versus C1 versus C3

| Scenario | Controller | RMSE | MAE | Substitution fraction | Reliability entries |
|---|---|---:|---:|---:|---:|
| `nominal_tracking` | `B_plain_MPC` | 10.807823 | 4.486692 | 0.000000 | 0 |
| `nominal_tracking` | `C1_sensor_MPC` | 10.807836 | 4.487264 | 0.000998 | 0 |
| `nominal_tracking` | `C3_arbitration_MPC` | 10.807836 | 4.487264 | 0.000998 | 0 |
| `step_reference` | `B_plain_MPC` | 7.075726 | 3.781923 | 0.000000 | 0 |
| `step_reference` | `C1_sensor_MPC` | 7.075731 | 3.782032 | 0.000998 | 0 |
| `step_reference` | `C3_arbitration_MPC` | 7.075731 | 3.782032 | 0.000998 | 0 |
| `changing_reference` | `B_plain_MPC` | 7.220168 | 4.236726 | 0.000000 | 0 |
| `changing_reference` | `C1_sensor_MPC` | 7.218066 | 4.235695 | 0.000666 | 0 |
| `changing_reference` | `C3_arbitration_MPC` | 7.218066 | 4.235695 | 0.000666 | 0 |

**Reference-transient verdict:** C3 made 8 isolated false substitutions on fault-free nominal/reference trajectories, all instantaneous and none latched. Exactly 1 occurred within one 0.05 s control interval after a reference transition: 1 after the step-reference change and 0 after changing-reference transitions.

| Scenario | Seed | False-substitution time | Nearest reference-change delta | Reliability latched |
|---|---:|---:|---:|---|
| `changing_reference` | 29028 | 1.54 s | -0.46 s | `False` |
| `changing_reference` | 29029 | 0.72 s | -1.28 s | `False` |
| `nominal_tracking` | 29028 | 1.54 s | n/a | `False` |
| `nominal_tracking` | 29029 | 0.72 s | n/a | `False` |
| `nominal_tracking` | 29030 | 3.36 s | n/a | `False` |
| `step_reference` | 29028 | 1.54 s | +0.04 s | `False` |
| `step_reference` | 29029 | 0.72 s | -0.78 s | `False` |
| `step_reference` | 29030 | 3.36 s | +1.86 s | `False` |

## Load-disturbance check

| Seed | Plain MPC RMSE | C3 RMSE | C3 substitution fraction | False sensor-fault entries |
|---:|---:|---:|---:|---:|
| 29026 | 0.970180 | 0.974326 | 0.001664 | 0 |
| 29027 | 0.983252 | 3.791717 | 0.485857 | 1 |
| 29028 | 0.925094 | 3.144745 | 0.387687 | 1 |
| 29029 | 0.870545 | 0.870545 | 0.001664 | 0 |
| 29030 | 0.936562 | 3.912237 | 0.482529 | 1 |

Mean load-disturbance RMSE is `0.937127 rad/s` for plain MPC and `2.538714 rad/s` for C3; mean C3 substitution fraction is `0.271880`. This confirms that the monitor can confuse a plant load disturbance with sensor failure.

## C2 recovery-gate ablation

Across 10251 recovery opportunities, residual blocked 5849, CUSUM blocked 10251, persistence blocked 0, and the auxiliary gate alone blocked 0. Counts can overlap except the explicitly exclusive auxiliary count.

C2 produced identical closed-loop behavior to C1 because the auxiliary recovery condition was not the active recovery constraint in the evaluated scenarios. C2 is retained as an architectural/recovery ablation rather than a performance-enhancing controller. Full evidence is in `C2_ABLATION_ANALYSIS.md`.

## Safety, integrity, and limitations

Across all 275 runs: optimizer failures `0`, main prediction failures `0`, auxiliary prediction failures `0`, nonfinite events `0`, voltage violations `0`, and slew violations `0`.

- All evidence is synthetic simulation; hardware timing and sensor validation remain outstanding.
- The detector can confuse load disturbance with sensor failure.
- Slow drift is not materially improved.
- Parameter variation remains unsupported by a general mismatch solution.
- The auxiliary estimator assumes trustworthy current measurement.
- Calibration may not transfer outside its saved clean-validation operating population.

## Reproduce and verify

```bash
python scripts/evaluate_v3_closed_loop.py
python scripts/verify_v3_results.py
python scripts/generate_v3_report.py
python scripts/verify_v3_results.py
```

Canonical evidence: `v3_final_11scenario_runs.csv`, `v3_final_11scenario_summary.csv`, `v3_final_11scenario_summary.json`, `v3_final_reference_runs.csv`, `v3_reference_substitution_events.csv`, and `c2_recovery_gate_diagnostics.csv`.
