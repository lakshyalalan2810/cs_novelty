# Reproducibility

## What can be reproduced safely

Read-only checks can inspect source, JSON, CSV, hashes, plan structure, and context consistency without running scientific work. Examples are in [COMMANDS.md](COMMANDS.md).

The current saved evidence supports reconstruction of:

- source/config/model/calibration bindings;
- exact 12,200-cell key counts;
- corrected H1–H9 summaries from saved rows;
- the provenance chain for the discarded 400-cell attempt;
- the 2,700-cell EKF repair;
- the bootstrap correction from saved data;
- paper figure/table provenance.

## Source and archive availability (2026-10-08)

Source-only checks can build the saved manuscript assets and regenerate tracked-evidence tables. Full figure generation (Figure 6) and independent reconstruction from canonical rows require ignored scientific files. Original files were recovered from an existing local backup and restored/hash-checked without rerunning science; a local deterministic archive with exact baseline Git blobs is prepared. It is not a public deposit. See project/REPRODUCIBILITY.md for the operation/dependency map and project/paper/PUBLICATION_RELEASE_REPORT.md for remaining actions. Do not equate local presence with reader access.

## Recorded execution environment

The V4 result manifest records:

- Python 3.11.15 (Anaconda build);
- NumPy 2.2.6;
- pandas 2.3.3;
- SciPy 1.13.1;
- PyTorch 2.6.0+cu124.

`project/requirements-frozen-v4.txt` records those manifest versions. `project/requirements.txt` instead defines the supported reconstruction/development environment. The archival file is necessarily partial because the manifest did not capture the full OS, CUDA driver, solver, BLAS, or transitive-package state; it therefore does not guarantee identical floats, model bytes, or hashes.

## Reproduction order

1. Read the V4 preregistration and preexecution amendment.
2. Verify plan and source hashes from result_manifest.json.
3. Verify that the raw run/event files match core_subsets.json and the manifest.
4. Verify preanalysis_integrity.json.
5. Verify repair provenance and confirm that only 2,700 affected cells were rerun.
6. Verify corrected statistics from saved rows; do not rerun simulations.
7. Verify paper table and figure manifest.
8. Only then consider a new preregistered experiment.

## Normal evidence path

| Stage | Entry point or artifact | Output |
|---|---|---|
| Dataset | project/scripts/v4_generate_dataset.py | data/processed/v4 dataset and manifest |
| Training | project/scripts/v4_train_models.py | per-training-seed main/auxiliary bundles |
| Calibration | project/scripts/v4_calibrate.py and v4_calibrate_baselines.py | calibration JSON/config artifacts |
| Simulation | v4_protocol_faultfree.py, v4_protocol_sweep.py, v4_protocol_recovery.py through v4_closed_loop.py | checkpoints, runs.csv, events.csv |
| Core accounting | project/scripts/v4_protocol_core.py and preanalysis_integrity.json | exact H1–H9 key/dependency counts |
| Historical analysis | project/scripts/v4_confirm_analysis.py | superseded summaries |
| Corrected analysis | project/scripts/v4_statistics_audit.py | corrected CSV/JSON and correction record |
| Verification | project/scripts/v4_confirm_verify.py and related verify_*.py scripts | provenance/structure checks |
| Paper table | project/scripts/v4_paper_tables.py | LaTeX tables |
| Paper figures | project/scripts/v4_paper_figures.py | deterministic vector figures |
| Tests | project/tests/ and scripts/run_all_verifiers.py | recorded audit/test status |

The arrows are a provenance path, not a recommendation to rerun every stage. Existing raw outputs and hashes should be reused for orientation.

~~~text
preregistration -> frozen plan -> protocol core -> checkpoint
       -> canonical rows/events -> corrected statistics -> verifier
       -> paper tables/figures -> manuscript audit
~~~

## Explicit answer for a future AI

Saved-evidence verification is supported: inspect the bound raw artifacts, hashes, integrity report, repair record, corrected statistics, verifiers, and paper table/figure inputs in the order above. A fresh bit-for-bit scientific re-execution is not established because the environment record is partial and large runtime artifacts are locally ignored. Treat those as two different goals; do not report the second merely because the first succeeds.

## Expensive operations

The following are not orientation steps and should not be launched without explicit authorization and a defined new purpose:

- v4_protocol_core.py without dry-run;
- any full protocol wrapper with workers;
- v4_train_models.py;
- v4_generate_dataset.py;
- v4_calibrate.py;
- v4_repair_ekf_witness_cells.py;
- broad robustness, DET, timing, or retraining jobs;
- commands that overwrite checkpoints or result CSVs.

## Reproducibility caveat

The old confirmatory verifier passed while reproducing the H1/H2/H7 cluster defect. Passing a verifier is therefore not enough; the statistical method and hierarchy must also be read. The corrected source is v4_statistics_audit.py.

Related: [COMMANDS.md](COMMANDS.md), [PROVENANCE.md](PROVENANCE.md), [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md).
