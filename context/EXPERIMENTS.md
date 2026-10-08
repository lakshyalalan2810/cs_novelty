# Experiment inventory

## Matrix catalog

| Matrix | Controllers | Scenarios/conditions | Training seeds | Simulation seeds | Count | Endpoints | Evidence/verifier | Status |
|---|---|---|---|---|---:|---|---|---|
| V3 final 11-scenario | B, C1/C2, C3 and comparison controllers | 11 frozen scenarios | model bundle used by V3 | 29026–29030 | 275 | fault-window RMSE, entries, recovery, safety | V3_DUAL_VIRTUAL_SENSOR_EXTENSION.md; verify_v3_results.py | frozen historical |
| C4 development | B, C3, C4 | 7 gate scenarios | development bundle | 19026–19030 | 105 | load false-entry/substitution/tracking, combined, parameter, reference, safety | c4_development_stage_gate.json; verify_c4_results.py | development NO_GO |
| EKF comparator | EKF virtual MPC and V3-related comparators | 5 primary fault/disturbance scenarios | fixed EKF grid selection | 19026–19030 | 100 | estimator/closed-loop RMSE, detection, safety | ekf_closed_loop_summary.json; verify_ekf_results.py | BASELINE_ONLY |
| Training-seed robustness | B and C3 | bias, dropout, combined, load, parameter variation | 2026–2028 | 49026–49030 | 150 | paired fault-window RMSE, seed clusters, safety | TRAINING_SEED_ROBUSTNESS.md; verify_training_seed_robustness.py | frozen boundary study |
| Robustness/severity Part B | B and C3 plus boundary cells | bias/dropout/drift/load/combined/current boundary | 2026–2028 | 59026–59030 | 735 rows; 705 unique | severity labels, detection, false entry, safety | FINAL_ROBUSTNESS_AND_SEVERITY_STUDY.md; verify_final_robustness_study.py | complete; no supported point |
| V4 H1–H9 core | B, C3, V4_full_aux, V4_full_ekf plus structural/baseline cells | fault-free, sweep, finite recovery | 2026–2036 as bound by hypothesis | 71000–71199 and 72000–72099 blocks as planned | 12,200 | H1–H9 preregistered endpoints | confirmatory manifest, integrity files, v4_confirm_verify.py plus corrected statistics audit | current confirmatory core |

## V1 final holdout

Status: FROZEN SUCCESS WITH LIMITATIONS.

- Final matrix: 4 controllers x 11 scenarios x 5 seeds 12026–12030 = 220 runs.
- Plant/model timing: 0.01 s model step, 0.05 s control interval, H20, Nc2, blocks (5,15).
- Isolated sensor-fault interval RMSE: plain 7.584044; reliability-aware 1.374104 rad/s.
- Improvement: 81.881647%.
- All 220 runs: zero optimizer, voltage, slew, and nonfinite failures.
- Drift remained weak; load/model mismatch caused false latches; combined fault/load worsened by 143.685962% in the corrected timing study.
- Runtime distribution does not support reliable 20 Hz end-to-end execution.

The old 81.4%/18.6% narrative belongs to a pre-fix configuration and must not be used as the current headline.

## V2 and C2

V2 added a current/voltage-only auxiliary LSTM and used it as an additional recovery condition. C2 did not feed the auxiliary estimate into MPC. Across 55 paired cases and 10,251 recovery opportunities, C1 and C2 behavior was identical and the auxiliary gate was uniquely binding zero times. The earlier report’s positive parameter-variation interpretation is superseded by the canonical equality analysis.

## V3/C3

Status: FROZEN STRONGEST ARCHITECTURE WITH LIMITATIONS.

- 5 controllers x 11 scenarios x 5 seeds 29026–29030 = 275 runs.
- Original 200 rows plus 75 nominal/reference rows.
- Compared with plain MPC, C3 reduced fault-window RMSE approximately 87.2% for 5% bias, 95.4% for 15% bias, and 98.2% for dropout.
- Load disturbance produced false entries and worsened RMSE: B 0.937127 versus C3 2.538714 rad/s.
- Reference false substitutions: 8 isolated substitutions, zero latches.
- Drift and current-channel robustness remain weak.

## C4

Status: DEVELOPMENT NO_GO; FINAL HOLDOUT NOT RUN.

C4 attempted three-way attribution using sensor–main, sensor–auxiliary, and main–auxiliary distances. It evaluated B, C3, and C4 over seven development scenarios and seeds 19026–19030, 105 development cells (35 per controller, including 35 C3/C4 scenario–seed pairs). C4 was numerically identical to C3, suppressed zero intended load false entries, and failed its load, parameter, and reference transition gates. No reserved final holdout 39026–39030 exists for C4.

## EKF comparator

Status: BASELINE_ONLY.

The augmented [current, speed, load torque] EKF consumed applied voltage and current only. It was stable and finite and had clean-test RMSE 3.1085 rad/s versus auxiliary LSTM 3.4376 rad/s, but it did not improve the combined fault/load case enough and worsened load behavior relative to C3. The separate 100-row closed-loop comparator is historical/development evidence, not a V4 result.

## Training-seed study

Status: TRAINING-SEED-SENSITIVE.

Matched models were trained with seeds 2026, 2027, and 2028 and evaluated against common simulation seeds 49026–49030: 150 rows. Combined fault/load means were +0.310712, -1.555400, and -0.461026 respectively. Bias and dropout remained directionally favorable, but the preregistered all-seed robustness rule failed. Load penalties were unfavorable for every training seed.

## Final robustness and severity study

Status: COMPLETE; NO SUPPORTED OPERATING POINT.

- Part A: 150 rows, reusing the training-seed matrix.
- Part B: 735 labeled rows, 705 unique executions plus 30 exact row reuses.
- All rows completed with zero numerical, safety, and incomplete-run failures; 43/43 prestudy hashes matched.
- Bias severities: LIMITED.
- Dropout 0.1 s: UNSUPPORTED; 0.5–2 s: LIMITED.
- Drift: UNSUPPORTED at every tested level.
- Load false entries: 0/15 at 0.06 and 0.10 N·m, 4/15 at 0.15 N·m, 10/15 at 0.20 N·m.
- Current-channel corruption degraded C3 and EKF; no current-fault tolerance claim.

## V4 confirmatory core

The preregistration covered 268,910 possible cells, but only the exact H1–H9 dependency core was executed. See [V4_CONFIRMATORY.md](V4_CONFIRMATORY.md) for counts, repairs, corrected results, and deferred scope.

Related: [RESEARCH_TIMELINE.md](RESEARCH_TIMELINE.md), [LIMITATIONS.md](LIMITATIONS.md), [PAPER_CONTEXT.md](PAPER_CONTEXT.md).
