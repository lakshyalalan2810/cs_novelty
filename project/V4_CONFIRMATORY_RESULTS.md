# V4 H1–H9 confirmatory results

Status: **AUDITED — CORRECTED STATISTICAL SUMMARY FROM FROZEN CORE**.

## Scope and provenance

The original V4 umbrella contained 268,910 cells. The executed confirmatory
dataset is the exact deduplicated 12,200-cell dependency core required by
H1–H9; the remaining 257,260 umbrella cells are deferred and were not used in
any confirmatory result.

An earlier 400-cell attempt exposed a plan-file provenance defect before any
H1–H9 analysis. Those rows and their checkpoint were fully discarded. The
scientific plan was independently reconstructed 12,200/12,200, then a clean
0/12,200 run started under the administratively re-frozen plan
`fb3e2e614b2b35dd3ae66be3c821d389b59573291140ab2a2f906a2912166ae5`.

Pre-analysis validation of the completed offline outputs found an execution
integrity defect in all 2,700 required `V4_full_ekf` cells: the witness called
`step()` before observer initialization, recording 581 failures per 6 s run.
No confirmatory inference had run. The root cause was fixed and all and only
those 2,700 corrupted cells were rerun; the 9,500 valid cells and every
deferred cell were not rerun. Selection used controller identity and the
mechanical failure signature, not an H1--H9 outcome. The incident and
pre-repair hashes are frozen in
`results/v4/ekf_witness_integrity_repair.json`.

The repaired canonical core has 12,200 unique planned keys, zero missing,
duplicate, or unexpected keys, and zero optimizer, prediction, observer,
nonfinite, or voltage failures. The 3,053 recorded slew-limit counters are
retained as safety outcomes rather than excluded. Canonical subsets are
explicitly labeled **H1–H9 CORE SUBSETS** in
`results/v4/confirmatory/core_subsets.json`.

## Confirmatory results

Positive delta supports each hypothesis. The corrected analysis resamples
training seeds and then nested simulation-seed clusters, retaining all four
reference rows when a fault-free simulation seed is selected. Confidence
intervals are uncentered 2.5th/97.5th percentiles of 20,000 bootstrap means.
Raw p-values are the preregistered two-sided empirical tail areas after
centering that bootstrap distribution at its bootstrap mean under the
zero-mean paired-delta null. These are different bootstrap constructions, so
a percentile CI can exclude zero while the centered empirical p-value exceeds
0.05. Family alpha is 0.05 with Holm step-down correction.

The final audit found that the historical analysis and verifier resampled
individual reference rows within training seed instead of simulation-seed
clusters for H1, H2, and H7. Their historical files are preserved; the table
below uses the corrected summaries recomputed from the unchanged 12,200-run
core. No scientific simulation was rerun and no reject/do-not-reject decision
changed.

| H | Endpoint and direction | Pairs (excluded) | Delta | 95% CI | Raw p | Holm p | Cohen dz | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| H1 | false latch, C3 − V4-aux | 2,400 (0) | 0.209167 | [0.049167, 0.428333] | 0.0627 | 0.2508 | 0.514178 | do not reject |
| H2 | false latch, C3 − V4-EKF | 2,400 (0) | 0.209167 | [0.049583, 0.428333] | 0.0659 | 0.2508 | 0.514178 | do not reject |
| H3 | bias-2σ detection, V4-aux − C3 | 129 (21) | −0.100775 | [−0.178862, −0.043478] | 0.0125 | 0.0763 | −0.333467 | do not reject |
| H4 | bias-2σ detection, V4-EKF − C3 | 129 (21) | −0.100775 | [−0.175439, −0.043478] | 0.0109 | 0.0763 | −0.333467 | do not reject |
| H5 | bias-8σ recovery, V4-aux − C3 | 264 (36) | 0.000000 | [0.000000, 0.000000] | 1.0000 | 1.0000 | undefined | do not reject |
| H6 | bias-8σ RMSE, B − V4-aux | 550 (0) | −2.557027 | [−4.467987, −0.809160] | 0.0110 | 0.0763 | −0.769107 | do not reject |
| H7 | tracking penalty, (C3−B) − (V4-aux−B) | 2,400 (0) | −0.063573 | [−0.148889, −0.013249] | 0.0744 | 0.2508 | −0.184820 | do not reject |
| H8 | load false entry, C3 − V4-aux | 124 (26) | 0.217742 | [0.138686, 0.315315] | <0.0001 | <0.0009 | 0.525458 | **reject** |
| H9 | load false entry, C3 − V4-EKF | 124 (26) | 0.217742 | [0.138686, 0.315315] | 0.0001 | 0.0008 | 0.525458 | **reject** |

H3–H5 and H8–H9 exclude pairs where either controller pre-latched. H6 and
H7 are intention-to-treat. H5's effect size is undefined because every paired
delta is zero.

The H8 implementation recorded zero tail hits; because its two-sided
empirical resolution is `2/20000 = 0.0001`, it is displayed as
`p < 0.0001` rather than exact zero. H9 had one tail hit and therefore retains
the exact implemented value `p = 0.0001` (Holm-adjusted `0.0008`).

Only H8 and H9 survive Holm correction. Under the preregistered conditions
tested here, both V4 witnesses reduce load-induced false entries at the 0.15
N·m load step. H1 and H2 are directionally favorable descriptions but are not
statistically supported. H3, H4, H6, and H7 have unfavorable observed
contrasts; H5 is exactly null. These findings do not establish general sensor
fault tolerance, universal superiority, formal fault isolation, real-time
feasibility, or hardware validity.

## Deferred / not executed in the H1–H9 core

- DET-wide protocol and other sweep severities
- dropout and drift V4 confirmatory protocols
- combined-fault V4 umbrella expansion
- robustness OFAT and timing umbrellas
- non-bias recovery experiments

Historical V3, C4 development, EKF, training-seed, and final 735-row
robustness evidence was read-only verified. The applicable C4 development
verifier independently retains the `NO_GO` decision; no C4 final-holdout
artifact is present or claimed.

## Historical frozen result hashes

```text
d8fbc3d7a3098ad695a25d92b33dffc7ff10e06938c5de3c9746255a0b4570cd  results/v4/faultfree/runs.csv
50290c64bb73963f59bbec9bff40ac27ae759e21dcffe5a05edc92cb32203c94  results/v4/faultfree/events.csv
8d63e71ec05263d649eefa70309ccaa808f98eea6b7dc7cd3b1bd9c51ce481d8  results/v4/sweep/runs.csv
d020f99fe4fdc7424cba3327402ef504e08e2da97e8774339e6c34d2d253a1b4  results/v4/sweep/events.csv
b09f0c37e509e02df56e08f76287010d426384a8441b1072191e89e68df6050d  results/v4/recovery/runs.csv
c6b89475c967661dd02974097623f3bae75344919da56f3ee576c8807c0a1dd4  results/v4/recovery/events.csv
5e2461380aae5f2b31440f013c128a29080475bfdcbf2dbd8e11fcb2f667f765  results/v4/confirmatory/hypothesis_table.csv
f108f327d2640390e6dcfeaebc12697e70e806f20579e2522f1edd69232f22a5  results/v4/confirmatory/confirmatory_summary.json
a9758642356937f9a31146e9bc81ae4b2cbb65b5a60aafbbb35ce6aaa64d68a7  results/v4/confirmatory/h1_h9_detailed_report.csv
2b5433e77d5e396c65f9974dce4686861a3d3fb8de6bd7f9d23e88ff6fc8b8d3  results/v4/confirmatory/independent_verification.json
68a1089054077381ee59dde7379297ba12da5effe89a6aeabb27096e32372ed6  results/v4/confirmatory/final_summary.json
```

These historical analysis artifacts were not overwritten. Corrected artifact
hashes and the superseding manifest hash are recorded in
`results/v4/confirmatory/result_manifest.json`; the complete rationale and
old/new values are in `results/v4/confirmatory/statistics_correction_record.json`
and `V4_STATISTICAL_AND_REPAIR_AUDIT.md`.

```text
d992473ce809f8736fa366e1e2b160b9f316f39d180391babcb1ad07f63d44f8  results/v4/confirmatory/hypothesis_table_corrected.csv
ed8d747fa90171ed9dfdb17ca3046811a7d80a0a44e7fc6c6e48ff0646c59381  results/v4/confirmatory/confirmatory_summary_corrected.json
00dc0814722f6ecdc9b97ca9bbab7031c86dc831629c1be6fe6f5c5f383a01ba  results/v4/confirmatory/final_summary_corrected.json
52cb4b03e03ff43b7e20ce13b766a3c29b2c24610263acc95629afdcfb89ffb2  results/v4/confirmatory/statistics_correction_record.json
0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e  results/v4/confirmatory/result_manifest.json
```
