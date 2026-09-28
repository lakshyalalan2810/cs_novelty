# Source of truth

## One-sentence rule

For scientific claims, trust current raw artifacts and corrected summaries bound by the V4 manifest; use narrative documents only when they agree with those artifacts, and label every older or contradictory statement as historical.

## Decision tree

~~~text
Need a number or claim?
        |
        v
Is it H1-H9?
  | yes                         | no
  v                             v
corrected CSV/JSON +        identify the experiment
raw rows + manifest         and read its canonical report
        |                             |
        v                             v
check integrity/repair      check supersession ledger
and statistical scope       and limitations
        |
        v
state endpoint, sample unit, and scope
~~~

## Current scientific authority

1. H1–H9 numeric values: project/results/v4/confirmatory/hypothesis_table_corrected.csv.
2. H1–H9 interpretation: project/results/v4/confirmatory/final_summary_corrected.json and project/V4_CONFIRMATORY_RESULTS.md.
3. Scope/completeness: preanalysis_integrity.json and core_subsets.json.
4. Repair provenance: ekf_witness_integrity_repair.json.
5. Statistical correction: statistics_correction_record.json and scripts/v4_statistics_audit.py.
6. Artifact binding: result_manifest.json, subject to the current SHA mismatch documented in [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md).
7. Paper-facing consistency: paper/tables/tab_h1_h9.tex, PAPER_FINAL_AUDIT.md, and FIGURE_MANIFEST.md.

## Experiment-specific authority

| Topic | First file to read |
|---|---|
| V1 final matrix | project/results/metrics/final_summary.json and project/FINAL_REPO_AUDIT_POSTFIX.md |
| V3 final | project/V3_DUAL_VIRTUAL_SENSOR_EXTENSION.md and results/metrics/v3_final_11scenario_summary.json |
| C2 | project/C2_ABLATION_ANALYSIS.md |
| C4 | project/results/metrics/c4_development_stage_gate.json |
| EKF comparator | project/results/metrics/ekf_closed_loop_summary.json |
| Training-seed | project/TRAINING_SEED_ROBUSTNESS.md |
| Severity/robustness | project/FINAL_ROBUSTNESS_AND_SEVERITY_STUDY.md |
| V4 protocol | project/V4_PREREGISTRATION.md |
| V4 core | project/V4_CONFIRMATORY_RESULTS.md |
| Manuscript | project/paper/main.tex and project/PAPER_FINAL_AUDIT.md |

## Claim labels

Use these words precisely:

- SUPPORTED — Holm-rejected in the stated endpoint and scope.
- DIRECTIONALLY FAVORABLE — observed sign favors the proposed effect but the multiplicity-adjusted decision is not supported.
- UNFAVORABLE — observed sign goes against the proposed effect; do not turn a non-rejection into evidence of no effect.
- NULL — exact observed delta zero under the retained pairs.
- NO_GO — development gate failed; no final branch claim.
- BASELINE_ONLY — comparator is retained for context, not promoted as the architecture.
- NOT ESTABLISHED — no current evidence supports the stronger claim.

## What to do when an old README disagrees

Read the current artifacts first, record the conflict in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), and do not edit the old README merely to make this context package look cleaner. This package is an orientation aid, not a retroactive rewrite of the research history.

Related: [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md), [PROJECT_SNAPSHOT.md](PROJECT_SNAPSHOT.md), [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).

