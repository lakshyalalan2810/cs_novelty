# C2 Recovery-Gate Ablation Analysis

C2 was diagnosed without changing or retuning its recovery gate. A recovery opportunity is one sample that begins with the sensor monitor active. Blocking-condition counts may overlap; `blocked_only_by_auxiliary` is the exclusive auxiliary-gate count.

| Diagnostic | Count |
|---|---:|
| Recovery opportunities | 10251 |
| Blocked by raw residual | 5849 |
| Blocked by CUSUM | 10251 |
| Blocked by persistence only after all signal gates passed | 0 |
| Blocked only by auxiliary gate | 0 |
| Completed recoveries | 0 |

## Per-scenario evidence

| Scenario | Opportunities | Residual blocks | CUSUM blocks | Persistence blocks | Auxiliary-only blocks | Recoveries |
|---|---:|---:|---:|---:|---:|---:|
| `combined_fault_load` | 1490 | 1408 | 1490 | 0 | 0 | 0 |
| `load_disturbance` | 810 | 759 | 810 | 0 | 0 | 0 |
| `sensor_bias_15` | 1990 | 993 | 1990 | 0 | 0 | 0 |
| `sensor_bias_5` | 1990 | 993 | 1990 | 0 | 0 | 0 |
| `sensor_dropout` | 1990 | 993 | 1990 | 0 | 0 | 0 |
| `sensor_noise` | 1981 | 703 | 1981 | 0 | 0 | 0 |

## Interpretation

C1 and C2 behavioral metrics are identical across all 55 paired scenario/seed cases: `True`. The largest C2-minus-C1 RMSE difference is `0 rad/s`; the largest substitution-duration difference is `0 s`.

The auxiliary gate delayed recovery relative to C1: `False`. It was uniquely binding on `0` recovery opportunities.

C2 produced identical closed-loop behavior to C1 because the auxiliary recovery condition was not the active recovery constraint in the evaluated scenarios. C2 is therefore retained as an architectural/recovery ablation rather than a performance-enhancing controller.
