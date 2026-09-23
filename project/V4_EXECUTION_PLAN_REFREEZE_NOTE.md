# V4 H1–H9 execution-plan administrative re-freeze

Administrative re-freeze: 2026-09-22T15:44:25+05:30

This is an administrative provenance re-freeze before confirmatory analysis,
not a scientific protocol amendment.

## Reason for re-freeze

The original execution plan was frozen with SHA256
`7e409aef24c1472382b19964927c878f1daf44f1564d23b6ef1e10f07c35b75f`.
The first core-runner implementation rebuilt and rewrote that file when it
started, changing the non-scientific `generated_utc` value. The resulting
current plan has SHA256
`fb3e2e614b2b35dd3ae66be3c821d389b59573291140ab2a2f906a2912166ae5`.

The old byte representation cannot be reconstructed locally; all scientific
plan content is independently reproduced and verified. Available provenance
identifies the observed mutation as regeneration of `generated_utc`, but this
note does not claim a direct byte-level comparison with the unavailable old
file.

## Aborted execution disclosure

The provenance defect was detected after 400 of 12,200 cells had executed.
No H1–H9 analysis was performed. All 400 rows, their checkpoint, and every
partial runtime/output artifact were deliberately discarded. The retained
confirmatory row count is zero, and the next execution will restart the full
12,200-cell core from zero.

## Independent scientific reconstruction

The plan was reconstructed independently from the fault-free, sweep, and
recovery protocol planners using the frozen amended H1–H9 dependency rules.
Comparison by scientific key and complete cell record established:

- 12,200 reconstructed union cells and 12,200 JSON union cells;
- 12,200/12,200 identical scientific records;
- zero missing, extra, duplicate, or changed scientific records;
- unchanged H1–H9 counts: 4,800, 4,800, 300, 300, 600, 1,100, 7,200, 300,
  and 300;
- all required controller, training-seed, simulation-seed, reference, fault,
  magnitude, onset/end, horizon, and dependency-membership fields match;
- all 550 B/bias-8σ H6 anchor cells are present; and
- no forbidden simulation seed appears.

No hypothesis, metric, training seed, simulation seed, severity, controller,
reference, horizon, pairing rule, exclusion rule, bootstrap setting, or Holm
procedure changed.

## Re-frozen artifacts

- Re-frozen plan SHA256:
  `fb3e2e614b2b35dd3ae66be3c821d389b59573291140ab2a2f906a2912166ae5`
- Immutable core runner SHA256:
  `f7ee31ae6f266aacf6a7ecc28de9e6ef2cfaac8417bbdf4e1b909b9d1632a07f`

The patched runner verifies the re-frozen plan hash and complete generated-plan
equivalence before execution. It never writes the plan. Runtime timestamps are
written separately to `results/v4/core_execution_status.json`.
