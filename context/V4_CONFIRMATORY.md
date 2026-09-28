# V4 confirmatory core

## Scope and status

Status: CORRECTED_AND_VERIFIED for the executed H1–H9 dependency core.

The preregistered umbrella was 268,910 cells. The canonical confirmatory core is exactly 12,200 unique cells:

~~~text
268,910 planned umbrella
        |
        +--> 11,650 umbrella cells used by the core
        |      ├── 9,600 fault-free
        |      ├── 1,450 sweep cells in the umbrella
        |      └──   600 recovery
        |
        +--> 257,260 deferred umbrella cells
        |
        +--> 550 supplemental H6 B-anchor cells
               |
               v
             12,200 executed core
~~~

The deferred 257,260 cells do not support any current confirmatory claim. The 550 supplemental H6 B-anchor cells are executed core cells but were added outside the original umbrella accounting; this is why 268,910 - 12,200 is not the recorded deferred count.

## Planning and discarded attempt

The original plan hash was 7e409aef24c1472382b19964927c878f1daf44f1564d23b6ef1e10f07c35b75f. An administrative re-freeze changed only non-scientific generated metadata and produced plan hash fb3e2e614b2b35dd3ae66be3c821d389b59573291140ab2a2f906a2912166ae5. The core was independently reconstructed 12,200/12,200 with the same hypothesis counts and H6 anchor structure.

An aborted attempt ran 400 cells before the provenance issue was detected. It produced no confirmatory analysis. The 400 rows, partial checkpoints, and partial outputs were discarded. The clean 12,200-cell restart is the only confirmatory execution.

## Pre-analysis integrity

The saved preanalysis_integrity.json reports:

- expected cells 12,200;
- actual unique cells 12,200;
- missing, unexpected, and duplicate cells all zero;
- protocol counts 9,600 / 2,000 / 600;
- event counts 593 / 107 / 308;
- optimizer, main prediction, auxiliary prediction, observer, nonfinite, and voltage failures all zero;
- 3,053 slew counters retained as outcomes;
- H1/H2/H3/H4/H5/H6/H7/H8/H9 dependency counts 4,800 / 4,800 / 300 / 300 / 600 / 1,100 / 7,200 / 300 / 300;
- H6 includes 550 B bias-8 anchor cells;
- 3 C3 model bundles and 11 V4 model bundles bound;
- status PASS.

## EKF witness integrity repair

The V4_full_ekf path could call EKF step before initialization. The failure signature was deterministic: 581 failures per affected 6-second cell. The repair changed the branch to initialize whenever the observer initialized flag was false.

- affected: all and only 2,700 planned V4_full_ekf cells;
- distribution: 2,400 fault-free and 300 sweep;
- unaffected valid cells not rerun: 9,500;
- deferred cells rerun: zero;
- repair was selected by controller identity, provenance, and mechanical failure signature, not endpoint performance;
- no H1–H9 analysis had run before repair.

The repair incident is recorded in project/results/v4/ekf_witness_integrity_repair.json with SHA-256 8766c62f2e8a8cd75db494de160086874f6e6b05274d5ea7bba5eb01019a9214.

## Corrected result table

| H | Endpoint summary | n | Excluded | Observed delta | 95% CI | Raw p | Holm p | Outcome |
|---|---|---:|---:|---:|---|---:|---:|---|
| H1 | false-latch probability, V4 aux vs C3 | 2400 | 0 | +0.209167 | [0.049167, 0.428333] | 0.0627 | 0.2508 | unsupported; favorable direction |
| H2 | false-latch probability, V4 EKF vs C3 | 2400 | 0 | +0.209167 | [0.049583, 0.428333] | 0.0659 | 0.2508 | unsupported; favorable direction |
| H3 | detection at bias 2 sigma, V4 aux vs C3 | 129 | 21 | -0.100775 | [-0.178862, -0.043478] | 0.0125 | 0.0763 | unfavorable; unsupported |
| H4 | detection at bias 2 sigma, V4 EKF vs C3 | 129 | 21 | -0.100775 | [-0.175439, -0.043478] | 0.0109 | 0.0763 | unfavorable; unsupported |
| H5 | recovery at finite bias 8 sigma | 264 | 36 | 0.000000 | [0, 0] | 1.0000 | 1.0000 | exact null |
| H6 | fault-window RMSE at bias 8 sigma, V4 aux vs B | 550 | 0 | -2.557027 | [-4.467987, -0.809160] | 0.0110 | 0.0763 | unfavorable; unsupported |
| H7 | fault-free tracking penalty, V4 aux vs C3 | 2400 | 0 | -0.063573 | [-0.148889, -0.013249] | 0.0744 | 0.2508 | unfavorable; unsupported |
| H8 | load false-entry probability at 0.15 N·m, aux vs C3 | 124 | 26 | +0.217742 | [0.138686, 0.315315] | <0.0001 | <0.0009 | Holm-supported |
| H9 | load false-entry probability at 0.15 N·m, EKF vs C3 | 124 | 26 | +0.217742 | [0.138686, 0.315315] | 0.0001 | 0.0008 | Holm-supported |

Positive false-entry deltas mean a reduction in false entries under the preregistered comparison. All endpoint definitions and exclusions are in [STATISTICS.md](STATISTICS.md).

## Hypothesis-by-hypothesis bounded interpretation

- H1: V4_full_aux versus C3, fault-free all-reference false-latch probability; 2,400 pairs, no exclusions; +0.209167, favorable direction, but Holm-unsupported.
- H2: V4_full_ekf versus C3 under the same fault-free endpoint; 2,400 pairs, no exclusions; +0.209167, favorable direction, but Holm-unsupported.
- H3: V4_full_aux versus C3 at bias 2 sigma, clean-start detection; 129 retained of 150; -0.100775, unfavorable and Holm-unsupported.
- H4: V4_full_ekf versus C3 at bias 2 sigma, clean-start detection; 129 retained of 150; -0.100775, unfavorable and Holm-unsupported.
- H5: V4_full_aux versus C3 at finite bias 8 sigma, clean-start recovery; 264 retained of 300; exact delta 0, null.
- H6: V4_full_aux versus B at bias 8 sigma, fault-window RMSE, ITT over all 11 V4 training seeds; 550 pairs, no exclusions; -2.557027, unfavorable and Holm-unsupported.
- H7: V4_full_aux versus C3 fault-free tracking penalty, ITT; 2,400 pairs, no exclusions; -0.063573, unfavorable and Holm-unsupported.
- H8: V4_full_aux versus C3 at 0.15 N·m load, clean-start false-entry probability; 124 retained of 150; +0.217742, Holm-supported.
- H9: V4_full_ekf versus C3 at 0.15 N·m load, clean-start false-entry probability; 124 retained of 150; +0.217742, Holm-supported.

ONLY H8 AND H9 SURVIVE HOLM CORRECTION. Their bounded interpretation is reduced load-disturbance-induced false sensor-fault entries at 0.15 N·m; they do not establish general sensor-fault superiority, detection/recovery/tracking improvement, or causal fault isolation.

## Deferred work

Deferred cells include DET-wide sweeps, other severity conditions beyond the confirmatory subset, robustness OFAT/timing, broad dropout/drift V4 protocols, combined-fault expansions, and non-bias recovery expansions. Their absence is a scope boundary, not a missing result to silently fill in.

## Primary evidence files

- project/results/v4/confirmatory/result_manifest.json
- project/results/v4/confirmatory/final_summary_corrected.json
- project/results/v4/confirmatory/hypothesis_table_corrected.csv
- project/results/v4/confirmatory/statistics_correction_record.json
- project/results/v4/confirmatory/preanalysis_integrity.json
- project/results/v4/confirmatory/core_subsets.json
- project/V4_CONFIRMATORY_RESULTS.md

Related: [STATISTICS.md](STATISTICS.md), [PROVENANCE.md](PROVENANCE.md), [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md).
