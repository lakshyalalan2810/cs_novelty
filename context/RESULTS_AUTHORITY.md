# Results authority

When files disagree, use this order.

## Authority order

1. Current raw per-run/per-event artifacts bound by the current result manifest.
2. Corrected confirmatory artifacts: hypothesis_table_corrected.csv, final_summary_corrected.json, confirmatory_summary_corrected.json, exclusion_audit.json.
3. Current integrity and repair records: preanalysis_integrity.json, core_subsets.json, ekf_witness_integrity_repair.json, statistics_correction_record.json.
4. Current paper table and figure provenance: paper/tables/tab_h1_h9.tex, paper/FIGURE_MANIFEST.md, PAPER_FINAL_AUDIT.md.
5. Current project README and root README.
6. Historical summaries, old audits, pre-fix reports, and old hypothesis tables — provenance only.

## AUTHORITATIVE CURRENT ARTIFACTS

| Path | Status | Reason | Supersession |
|---|---|---|---|
| project/results/v4/confirmatory/result_manifest.json | current artifact-binding file; cited superseding SHA 0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e, recomputed current SHA documented below | binds raw V4 rows, environment, repairs, and corrected report | supersedes the historical V4 result manifest |
| project/results/v4/confirmatory/hypothesis_table_corrected.csv | authoritative numeric H1–H9 table | corrected hierarchy and current p/Holm values | supersedes hypothesis_table.csv |
| project/results/v4/confirmatory/final_summary_corrected.json | authoritative narrative | corrected manuscript-facing interpretation | supersedes final_summary.json |
| project/results/v4/confirmatory/statistics_correction_record.json | authoritative correction record | documents H1/H2/H7 bug and no rerun | supersedes no scientific rows; supersedes old bootstrap summaries |
| project/results/v4/confirmatory/preanalysis_integrity.json | authoritative core accounting | exact key completeness and failure counters | supersedes informal run-count prose |

## SUPERSEDED ARTIFACTS

| Path | Status | Reason | Superseded by |
|---|---|---|---|
| project/results/v4/confirmatory/hypothesis_table.csv | historical | old bootstrap split simulation clusters for H1/H2/H7 | hypothesis_table_corrected.csv |
| project/results/v4/confirmatory/final_summary.json | historical | old H1/H2/H7 summary | final_summary_corrected.json |
| project/results/v4/confirmatory/independent_verification.json | historical | verifier passed while reproducing the same clustering defect | corrected statistical audit |
| project/FINAL_REPO_AUDIT.md | historical pre-fix audit | predates corrected timing/configuration | FINAL_REPO_AUDIT_POSTFIX.md plus current V4 authority |
| project/PROJECT_OVERVIEW_AND_PUBLISHABILITY.md | untracked/stale overview | predates final V4 closeout | current README and corrected artifacts |

## HISTORICAL BUT VALID ARTIFACTS

V1 final-matrix artifacts, V3 275-run artifacts, the C2 equality analysis, the training-seed study, and the final robustness/severity study remain valid for their own stated scopes. They are not V4 H1–H9 evidence and should retain their historical experiment labels.

## DEVELOPMENT-ONLY ARTIFACTS

| Path | Status | Reason |
|---|---|---|
| project/results/metrics/c4_development_stage_gate.json | development-only NO_GO | no final C4 holdout was run |
| project/results/metrics/ekf_closed_loop_summary.json | development comparator | EKF is BASELINE_ONLY, not a promoted successor |
| project/results/v4/posthoc/ | post-hoc/exploratory | not a replacement for the frozen H1–H9 endpoints |

## DO NOT USE FOR CURRENT PAPER NUMBERS

Do not use old H1/H2/H7 p-values 0.0719, 0.0711, and 0.0768, their old Holm values, the pre-fix V1 timing narrative, stale V2/C2 positive interpretation, or the stale untracked overview for current paper claims. Use the corrected CSV/JSON and name the endpoint and scope.

## Current V4 authority bundle

| Artifact | Role | Current hash |
|---|---|---|
| project/results/v4/confirmatory/hypothesis_table_corrected.csv | corrected H1–H9 numeric table | d992473ce809f8736fa366e1e2b160b9f316f39d180391babcb1ad07f63d44f8 |
| project/results/v4/confirmatory/final_summary_corrected.json | corrected status and interpretation | 00dc0814722f6ecdc9b97ca9bbab7031c86dc831629c1be6fe6f5c5f383a01ba |
| project/results/v4/confirmatory/statistics_correction_record.json | correction classification and Holm audit | 52cb4b03e03ff43b7e20ce13b766a3c29b2c24610263acc95629afdcfb89ffb2 |
| project/results/v4/confirmatory/preanalysis_integrity.json | core completeness and validity | bddd97c6c6e63d72f186734409da890efc392f82e5f1bbc4e89c220d4f64561c |
| project/results/v4/confirmatory/core_subsets.json | protocol subset counts and run hashes | 786aeaece77bb423362d3c422b6183ba374a03387097a20a6f625dccdb6e0482 |
| project/results/v4/ekf_witness_integrity_repair.json | exact repair incident | 8766c62f2e8a8cd75db494de160086874f6e6b05274d5ea7bba5eb01019a9214 |

The scope fields have a deliberate accounting distinction: 11,650 executed cells belong to the original 268,910-cell umbrella, 550 supplemental H6 B-anchor cells were added outside that umbrella, and 257,260 original-umbrella cells remain deferred. Do not calculate deferred scope as umbrella minus total executed core.

## Manifest caveat

The current file project/results/v4/confirmatory/result_manifest.json is the provenance binding for the V4 artifact set. Existing paper text and project/tests/test_v4_confirmatory_freeze.py cite a superseding-manifest SHA of 0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e. A fresh SHA-256 over the current filesystem bytes on 2026-09-28 is 58c2ece8410574de388b0933e17a0b8295e65834f74858edf7d21956e9ce3ff8. This mismatch is a current repository consistency issue. Do not silently declare the test constant or the recomputed hash authoritative without reconciling the artifact bytes; use the current file contents and its bound artifact hashes for scientific interpretation.

## Historical evidence rule

The original V4 table, old final summary, and independent verification record are not deleted because they prove the correction chain. They are superseded for publication values. Likewise, FINAL_REPO_AUDIT.md is historical pre-fix material; FINAL_REPO_AUDIT_POSTFIX.md is the post-fix readiness audit but does not override the V4 corrected manifest.

Related: [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md), [PROVENANCE.md](PROVENANCE.md), [STATISTICS.md](STATISTICS.md).
