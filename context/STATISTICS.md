# Statistical interpretation

## Unit of inference

The confirmatory analysis uses paired cell-level outcomes grouped by training seed and simulation seed. Reference rows are not independent observations for H1, H2, and H7: the corrected method preserves all four reference rows belonging to a selected simulation-seed cluster.

## Endpoint definitions

| Hypotheses | Endpoint and pairing |
|---|---|
| H1/H2 | Fault-free false-latch probability; all reference conditions; V4 witness versus C3 |
| H3/H4 | Conditional detection probability at bias 2 sigma; pairs with no pre-event latch are retained |
| H5 | Conditional recovery probability for finite bias 8 sigma; pairs with no pre-latch are retained |
| H6 | Fault-window RMSE at bias 8 sigma; V4 auxiliary versus B; intent-to-treat across all 11 V4 seeds |
| H7 | Fault-free tracking penalty; intent-to-treat; V4 auxiliary versus C3 |
| H8/H9 | Load false-entry probability at 0.15 N·m; clean-start pairs only; auxiliary or EKF V4 versus C3 |

For H3/H4/H5/H8/H9, a pre-latch excludes a pair from the clean-start conditional endpoint. H6/H7 are intent-to-treat. Recovery is the first inactive state at or after fault end, right-censored when absent. Tracking penalty is overall RMSE-B overall, with the sign interpreted according to the hypothesis table.

## Hierarchical bootstrap

The corrected hierarchy is:

1. sample training seeds with replacement;
2. within each selected training seed, sample simulation seeds with replacement;
3. retain all reference rows belonging to each selected simulation-seed cluster;
4. compute the paired mean for the resampled structure.

There are 20,000 replicates per hypothesis, base seed 20260930 plus the zero-based hypothesis index, 95% percentile intervals, two-sided centered empirical p-values, and Holm step-down correction at family alpha 0.05. Stored p=0 is reported as <0.0001; the p-value resolution is 0.0001.

## Why the correction matters

The historical analysis resampled individual rows within training seed and ignored simulation_seed for H1/H2/H7, splitting the four-reference clusters. The corrected audit affects only H1, H2, and H7. It reruns no scientific simulations. Observed deltas, effect sizes, exclusions, and Holm survivors do not change.

Historical versus corrected:

| H | Historical p / Holm | Corrected p / Holm |
|---|---|---|
| H1 | 0.0719 / 0.2844 | 0.0627 / 0.2508 |
| H2 | 0.0711 / 0.2844 | 0.0659 / 0.2508 |
| H7 | 0.0768 / 0.2844 | 0.0744 / 0.2508 |

The historical values remain in the tree only as immutable provenance. The corrected CSV and summary are current.

## Corrected outcomes

| H | n | Excluded | Delta | CI low | CI high | Cohen dz | raw p | Holm p | reject |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H1 | 2400 | 0 | +0.209167 | 0.049167 | 0.428333 | 0.514178 | 0.0627 | 0.2508 | no |
| H2 | 2400 | 0 | +0.209167 | 0.049583 | 0.428333 | 0.514178 | 0.0659 | 0.2508 | no |
| H3 | 129 | 21 | -0.100775 | -0.178862 | -0.043478 | -0.333467 | 0.0125 | 0.0763 | no |
| H4 | 129 | 21 | -0.100775 | -0.175439 | -0.043478 | -0.333467 | 0.0109 | 0.0763 | no |
| H5 | 264 | 36 | 0.000000 | 0.000000 | 0.000000 | undefined | 1.0000 | 1.0000 | no |
| H6 | 550 | 0 | -2.557027 | -4.467987 | -0.809160 | -0.769107 | 0.0110 | 0.0763 | no |
| H7 | 2400 | 0 | -0.063573 | -0.148889 | -0.013249 | -0.184820 | 0.0744 | 0.2508 | no |
| H8 | 124 | 26 | +0.217742 | 0.138686 | 0.315315 | 0.525458 | <0.0001 | <0.0009 | yes |
| H9 | 124 | 26 | +0.217742 | 0.138686 | 0.315315 | 0.525458 | 0.0001 | 0.0008 | yes |

## Interpretation guardrails

- A confidence interval and a centered empirical p-value are different constructions; a percentile interval can exclude zero while the centered empirical p-value exceeds 0.05.
- H8/H9 support the tested load step and clean-start endpoint only.
- H1/H2 are not significant after Holm even though their observed direction is favorable.
- A nonzero slew counter is a recorded controller outcome, not an automatic execution exclusion.

Related: [V4_CONFIRMATORY.md](V4_CONFIRMATORY.md), [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md), [GLOSSARY.md](GLOSSARY.md).

