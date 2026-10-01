# Do not touch

This is the preservation and safety boundary for the repository.

## FROZEN HISTORICAL EVIDENCE

Preserve V1/V2/V3 reports, notebooks, prefix snapshots, old audits, and superseded V4 summaries. They document what was believed, corrected, or rejected at each stage; deleting them would erase the supersession chain.

## FINAL CONFIRMATORY EVIDENCE

Preserve project/results/v4/faultfree/, sweep/, recovery/, confirmatory/, and the EKF repair record. These bind the exact core, raw outcomes, exclusions, and repair provenance.

## PREREGISTRATION

Preserve V4_PREREGISTRATION.md, V4_PREEXECUTION_AMENDMENT.md, V4_EXECUTION_PLAN_REFREEZE_NOTE.md, seed manifests, and plan/config files. They define intended design and administrative changes before interpretation.

## CORRECTED STATISTICAL ARTIFACTS

Preserve hypothesis_table_corrected.csv, final_summary_corrected.json, statistics_correction_record.json, exclusion_audit.json, and the correction source. They supersede historical H1/H2/H7 summaries without changing raw simulations.

## PAPER FINAL NUMBERS

Preserve paper/tables/tab_h1_h9.tex, FIGURE_MANIFEST.md, figure sources, and PAPER_FINAL_AUDIT.md. They connect manuscript claims to corrected results and reveal the unverified LaTeX build status.

## HASH MANIFESTS

Preserve result_manifest.json and all recorded SHA-256 values. The archival LF-byte hash is `0591d54f…`; the former `58c2ece8…` Windows working-tree hash was caused only by CRLF checkout conversion and is documented in `project/MANIFEST_HASH_RECONCILIATION.md`.

## Do not modify

Unless the user explicitly requests a scoped code or paper change, leave these untouched:

- all existing files outside context/;
- project/results/v4/faultfree/, project/results/v4/sweep/, and project/results/v4/recovery/ raw CSVs and checkpoints;
- project/results/v4/confirmatory/ result artifacts;
- project/results/v4/models/ and project/models/ model weights;
- project/data/processed/ datasets and manifests;
- project/paper/ figures, tables, bibliography, and main.tex;
- historical reports, prefix snapshots, old audits, and superseded summaries;
- project/PROJECT_OVERVIEW_AND_PUBLISHABILITY.md, a tracked historical review document.

## Do not run for orientation

- full V4 protocols;
- training or dataset generation;
- calibration or EKF repair;
- commands that overwrite checkpoints, rows, events, or model files;
- broad robustness, DET, timing, or severity jobs;
- a full environment installation merely to read saved evidence;
- LaTeX installation or compilation merely to orient.

## Do not claim

- that the V4 umbrella was executed in full;
- that H1/H2 were significant;
- that H3/H4/H6/H7 improved;
- that H5 improved;
- that C4 has final-holdout evidence;
- that EKF is the chosen successor;
- that a witness provides causal fault isolation;
- that the method is hardware validated;
- that reliable 20 Hz execution is established;
- that current-fault robustness is shown;
- that the repository is byte-for-byte reproducible from the current requirements file.

## Safe changes

The following are normally safe and additive:

- context/ documentation;
- read-only hash, inventory, and JSON/CSV inspection;
- a separately requested paper wording patch after the evidence is cited;
- a separately requested code fix with caller tracing and a narrow test.

If a proposed change would overwrite or regenerate an existing artifact, stop and get explicit direction.

Related: [COMMANDS.md](COMMANDS.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [AI_HANDOFF.md](AI_HANDOFF.md).
