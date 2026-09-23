# V4 statistical-consistency and EKF repair-provenance audit

## 1. Purpose and verdict

This audit used only the frozen 12,200-run canonical core. It ran no scientific
simulation, changed no hypothesis or endpoint, and did not alter the Holm
family. The statistics verdict is **C — IMPLEMENTATION INCONSISTENCY / BUG**:
the preregistration requires resampling training seed and then simulation seed,
but the historical analysis and its verifier ignored `simulation_seed` and
resampled individual rows within training seed. That split the four reference
rows belonging to each simulation seed in H1, H2, and H7. H3--H6 and H8--H9
have one analysis row per simulation seed, so their historical bootstrap was
already equivalent to the specified hierarchy.

Only the H1, H2, and H7 statistical summaries were recomputed, from the
unchanged canonical CSVs. The historical artifacts remain byte-for-byte
preserved. No observed delta, effect size, exclusion, or reject/do-not-reject
decision changed. H8 and H9 remain the only Holm rejections.

## 2. Exact CI and p-value constructions

For every hypothesis, positive paired delta supports the alternative. Let
`T_b` be a bootstrap replicate mean and let `B = 20,000`.

- Correct hierarchy: sample training seeds with replacement; within every
  selected training seed, sample its retained simulation seeds with
  replacement. A selected simulation seed contributes its complete cluster.
- H1/H2/H7 cluster content: four paired reference rows (`nominal`, `step`,
  `changing`, `wide`). All other hypotheses have one paired row per retained
  simulation seed.
- Point estimate: the arithmetic mean of the retained paired-row deltas.
- CI: the uncentered percentile interval
  `[quantile(T_b, 0.025), quantile(T_b, 0.975)]`. It is a hierarchical
  percentile CI, not basic or BCa.
- Exact null: zero mean paired delta.
- Raw p-value: two-sided empirical centered-bootstrap tail area,
  `min(1, 2 * mean(abs(T_b - mean(T_b)) >= abs(mean(T_b))))`. The threshold
  uses the bootstrap mean, not the observed delta. The code applies no
  plus-one correction.
- The CI and p-value therefore use different inferential objects: uncentered
  percentile quantiles versus a bootstrap-mean-centered empirical tail count.
  They are not algebraic inverses.

| H | Pairing / exclusions | Top level | Nested resampling unit | Corrected bootstrap mean | 95% percentile CI | Raw two-sided p |
|---|---|---:|---|---:|---:|---:|
| H1 | train/sim/reference; none | 3 training seeds | 200 sim clusters/train, 4 references/cluster | 0.209770 | [0.049167, 0.428333] | 0.0627 |
| H2 | train/sim/reference; none | 3 | 200 sim clusters/train, 4 references/cluster | 0.208123 | [0.049583, 0.428333] | 0.0659 |
| H3 | train/sim; 21 clean-start exclusions | 3 | one retained pair/sim | -0.101889 | [-0.178862, -0.043478] | 0.0125 |
| H4 | train/sim; 21 clean-start exclusions | 3 | one retained pair/sim | -0.101639 | [-0.175439, -0.043478] | 0.0109 |
| H5 | train/sim; 36 clean-start exclusions | 3 | one retained pair/sim | 0 | [0, 0] | 1.0000 |
| H6 | train/sim; intention-to-treat | 11 | one pair/sim | -2.556389 | [-4.467987, -0.809160] | 0.0110 |
| H7 | train/sim/reference; intention-to-treat | 3 | 200 sim clusters/train, 4 references/cluster | -0.063526 | [-0.148889, -0.013249] | 0.0744 |
| H8 | train/sim; 26 clean-start exclusions | 3 | one retained pair/sim | 0.219654 | [0.138686, 0.315315] | <0.0001 (stored 0; 0 hits) |
| H9 | train/sim; 26 clean-start exclusions | 3 | one retained pair/sim | 0.219611 | [0.138686, 0.315315] | 0.0001 (1 hit) |

## 3. Why H1, H2, and H7 have CI/p non-correspondence

After correcting the hierarchy, H1, H2, and H7 still have nominal percentile
CIs that exclude zero while their centered empirical p-values exceed 0.05.
This is valid under the implemented formulas: the percentile CI reads the
uncentered bootstrap quantiles, whereas the p-value recenters the same
replicates at their bootstrap mean and counts absolute tails. Discreteness,
only three top-level training-seed clusters, and skew/asymmetry make the
difference visible. Thus the numerical non-correspondence itself requires an
explicit methods explanation; the separate defect was the historical failure
to keep simulation-seed clusters intact.

## 4. Holm recomputation

The exact stored raw values were sorted and Holm step-down was recomputed at
family alpha 0.05. “Candidate” is `multiplier * raw p`, capped at one;
“monotonic” is the cumulative maximum in sorted order.

| Rank | H | Raw p | Multiplier | Candidate | Monotonic/final adjusted p | Decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | H8 | 0.0000 (display <0.0001) | 9 | 0.0000 | 0.0000 (display <0.0009) | reject |
| 2 | H9 | 0.0001 | 8 | 0.0008 | 0.0008 | reject |
| 3 | H4 | 0.0109 | 7 | 0.0763 | 0.0763 | do not reject |
| 4 | H6 | 0.0110 | 6 | 0.0660 | 0.0763 | do not reject |
| 5 | H3 | 0.0125 | 5 | 0.0625 | 0.0763 | do not reject |
| 6 | H1 | 0.0627 | 4 | 0.2508 | 0.2508 | do not reject |
| 7 | H2 | 0.0659 | 3 | 0.1977 | 0.2508 | do not reject |
| 8 | H7 | 0.0744 | 2 | 0.1488 | 0.2508 | do not reject |
| 9 | H5 | 1.0000 | 1 | 1.0000 | 1.0000 | do not reject |

The empirical p-value grid is `2/B = 0.0001`. H8's zero stored tail count is
not a mathematical probability of zero and is displayed as `p < 0.0001`;
its rank-one Holm bound is displayed as `<0.0009`. H9 is exactly the smallest
nonzero value produced by this implementation and remains `p = 0.0001`.

## 5. Effect-size audit

`Cohen d_z = mean(delta) / sample_sd(delta)` with denominator standard
deviation `ddof=1`. Its unit is the retained paired analysis row; signs use the
same positive-supports-H convention as the endpoints. Values are H1 0.514178,
H2 0.514178, H3 -0.333467, H4 -0.333467, H5 undefined, H6 -0.769107,
H7 -0.184820, H8 0.525458, and H9 0.525458. Exclusions are applied before the
calculation. H5 has 264 deltas all equal to zero, so both mean and sample SD
are zero. The implementation explicitly returns NaN for zero SD; “undefined,”
not zero, is mathematically and implementation-wise correct. **PASS.**

## 6. Exclusion audit

For H3--H5 and H8--H9, the code pivots only the endpoint and
`pre_event_reliability_entries`, then keeps a pair iff both controllers have
zero pre-event entries. No post-event detection, recovery, false-entry, or
other outcome participates in the mask. H6 and H7 are intention-to-treat and
have no exclusion path. Exact excluded `(training_seed, simulation_seed)` IDs:

- H3 = H4 (21): 2026 `{72026,72030,72031,72033,72042,72049}`; 2027
  `{72002,72006,72008,72013,72019,72026,72029,72030,72031,72046,72047,72049}`;
  2028 `{72030,72031,72049}`.
- H5 (36): 2026
  `{72026,72030,72031,72033,72042,72049,72058,72088,72099}`; 2027
  `{72002,72006,72008,72013,72019,72026,72029,72030,72031,72046,72047,72049,72056,72057,72063,72064,72072,72073,72076,72079,72088,72094,72096,72098}`;
  2028 `{72030,72031,72049}`.
- H8 = H9 (26): 2026
  `{72004,72026,72030,72031,72033,72042,72049}`; 2027
  `{72002,72004,72006,72008,72010,72013,72014,72019,72026,72029,72030,72031,72040,72046,72047,72049}`;
  2028 `{72030,72031,72049}`.

The complete controller-specific pre-entry values are in the machine-readable
`results/v4/confirmatory/exclusion_audit.json`. **PASS.**

## 7. EKF corruption and deterministic repair predicate

The V4 EKF witness is first used only after the 20-sample warm-up. The defective
branch tested `i == 0` at that point, which can never be true, so it called
`step()` before `initialize()`. Every 6 s V4_full_ekf run consequently recorded
581 observer failures and had no valid witness estimate for those samples.
The repaired branch initializes whenever `witness_ekf.initialized` is false.

Before repair, the deterministic target was the intersection of:

1. membership in the validated frozen 12,200-cell plan;
2. `controller == "V4_full_ekf"`;
3. old checkpoint provenance
   `7979c98f08431d51c9828bc282f706893abee75de16dc9483d948c16e6a8eed1`;
4. `observer_failures == 581`.

The controller predicate mechanically selected exactly 2,700 planned cells,
and every one had the provenance/failure signature. This predicate contains no
performance endpoint and was established before confirmatory inference.

## 8. Repair counts and canonicalization

| Dimension | Exact repaired count |
|---|---:|
| Protocol: faultfree | 2,400 |
| Protocol: sweep | 300 |
| Hypothesis membership: H2 | 2,400 |
| Hypothesis membership: H4 | 150 |
| Hypothesis membership: H9 | 150 |
| Training seed 2026 | 900 |
| Training seed 2027 | 900 |
| Training seed 2028 | 900 |

Fault-free cells span simulation seeds 71000--71199 and four references, 600
cells per reference. Sweep cells span 72000--72049: 150 nominal-reference
bias-2-sigma cells and 150 nominal-reference load-0.15 cells. These disjoint
counts sum to 2,700 and map to no other H1--H9 membership.

The other 9,500 plan keys were non-EKF targets and retained old provenance:
7,200 fault-free, 1,700 sweep, and 600 recovery. No non-EKF cell and no
already-valid EKF cell was rerun. Checkpoints now contain 12,200 rows and
12,200 unique primary keys: repaired provenance appears exactly 2,400 times in
fault-free and 300 times in sweep, with none in recovery. `INSERT OR REPLACE`
replaced the corrupted primary-key rows; it did not append duplicates.

Each target reused its frozen `RunConfig`; the SHA-256 scientific key covers
the full serialized configuration, including controller, training seed,
simulation seed, scenario/reference, fault settings, control stride, and
threshold scale. The new provenance fingerprint binds the repaired code,
Python/package versions, model files, calibration files, and configs. The
independent core verifier also checks every materialized row against its plan
configuration and verifies model/calibration bindings. Canonical CSVs were
materialized deterministically in frozen-plan order from checkpoint rows.
Pair construction read those repaired canonical CSVs, and independent core
verification reconstructed the same canonical frames from the checkpoints
before analyzing them. **Repair provenance PASS.**

## 9. Outcome-independence assessment

The repair is an **OUTCOME-INDEPENDENT INFRASTRUCTURE REPAIR**. Corruption was
detectable from observer lifecycle failures without reading H1--H9 outcomes;
the complete affected set was controller/configuration-defined; every affected
cell was rerun; no cell was cherry-picked; no endpoint, hypothesis, threshold,
or multiplicity rule changed; and no unaffected comparator was rerun for a
favorable result.

## 10. Hash and freeze validation

Before correction, manifest SHA-256
`9ff12f4184816a5ab646447c260a7ef842224e5a6094facca641824f2b65dd96`
and every bound core/statistical artifact validated. The canonical core hashes
remain unchanged:

- faultfree runs/events: `d8fbc3d7...570cd` / `50290c64...3c94`
- sweep runs/events: `8d63e71e...10d8` / `d020f99f...a1b4`
- recovery runs/events: `b09f0c37...6050d` / `c6b89475...e6a8`

Historical statistical artifacts also remain unchanged and are still bound in
manifest version 2. Corrected hashes are bound separately:

- `hypothesis_table_corrected.csv`: `d992473c...44f8`
- `confirmatory_summary_corrected.json`: `ed8d747f...9381`
- `final_summary_corrected.json`: `00dc0814...1ba`
- `exclusion_audit.json`: `093e6b38...209f`
- `statistics_correction_record.json`: `52cb4b03...ffb2`

The superseding manifest SHA-256 is
`0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e`.

## 11. Scientific result changes

Numerical bootstrap summaries changed only for H1, H2, and H7. Their observed
deltas and effect sizes did not change. All three remain do-not-reject. The
bounded conclusion is unchanged: only H8 and H9 support reduced
load-disturbance-induced false sensor-fault entries at the tested 0.15 N.m
condition for V4_full_aux and V4_full_ekf relative to C3. H1/H2 remain
directionally favorable but nonsignificant; H3/H4 are unfavorable; H5 has no
observed paired difference; H6/H7 are unfavorable; and deferred V4 scope
remains untested.
