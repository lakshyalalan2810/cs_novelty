# Reproducibility

Use `requirements.txt` for routine development and saved-evidence reconstruction on Python 3.11. It is a supported convenience environment, not the environment that produced the frozen V4 results.

`requirements-frozen-v4.txt` records every package version captured by `results/v4/confirmatory/result_manifest.json`: Python 3.11.15 (Anaconda), NumPy 2.2.6, pandas 2.3.3, SciPy 1.13.1, and PyTorch 2.6.0+cu124. The CUDA 12.4 PyTorch build requires a compatible NVIDIA platform. The manifest did not record the full OS, driver, solver, BLAS, or transitive-package state, so byte-identical re-execution is not established.

## Source-only checks versus archived-artifact checks

For the saved-evidence path, verify hashes, corrected CSV/JSON summaries, paper structure, and provenance records. Do not rerun dataset generation, training, calibration, or the V4 protocols merely to verify the repository.

| Operation | Clean source clone | Required ignored artifact |
|---|---|---|
| Build `paper/main.tex` from saved figures/tables | Supported with a real LaTeX/BibTeX toolchain | None |
| Regenerate all four tables with `scripts/v4_paper_tables.py` | Supported from tracked CSV inputs | None |
| Check the published corrected H1–H9 CSV/JSON and authority hashes | Supported; checks saved summaries rather than independently reconstructing raw endpoints | None |
| Regenerate figures 1–5 individually | Supported from tracked sources/post-hoc/corrected summaries | None |
| Run the complete `scripts/v4_paper_figures.py` generator | Archive required; it also regenerates figure 6 | `results/v4/sweep/runs.csv` |
| Independently reconstruct clean-start exclusions, H8/H9 rates, and corrected bootstrap inputs | Archive required | `faultfree/runs.csv`, `sweep/runs.csv`, `recovery/runs.csv` under `results/v4/` |
| Audit checkpoint rows/events against canonical CSVs | Archive required | The three `runs.csv`, three `events.csv`, and three `checkpoint.sqlite3` files |
| Execute the historical `scripts/v4_confirm_verify.py` completion/binding checks | Archive required; historical inference is superseded | Raw/checkpoint files, all V4 `.pt` files, and preserved `core_execution_status.json`; historical C3 bindings are already tracked |
| Inspect frozen training data, trained models, and histories | Archive required | Dataset NPZ, 22 V4 `.pt` binaries, 22 history NPZs |
| Independently retrain/re-execute the study with identical bytes | Not established by either package | Partial environment capture prevents a bitwise guarantee; no rerun is part of this release audit |

Figure 6 uses the original load-0.15 sweep rows and the preregistered clean-start mask. A clean source clone contains its saved PDF but cannot regenerate that PDF from the tracked summaries alone. The figure test in `tests/test_v4_paper.py` invokes the complete generator and therefore has the same archive dependency; a missing sweep CSV is a release-dependency failure, not authorization to simulate replacement data.

`scripts/v4_statistics_audit.py` implements the corrected cluster hierarchy, but its command-line entry point writes corrected artifacts. Read/reconstruct from saved rows without overwriting the frozen reports. The historical confirmatory verifier reproduced the H1/H2/H7 clustering defect and must not be cited as proof of corrected inference.

## Local archive status

The 2026-10-08 audit located the original ignored artifacts in a local backup, restored the exact original bytes into this workspace, and prepared `paper/release/v4-frozen-evidence.zip` with a detached SHA-256 inventory. Dataset and canonical row/event/checkpoint hashes match the frozen bindings. Model/history hashes are newly inventoried; associated model metadata match the baseline. See `ARCHIVAL_ARTIFACTS.md` for the original source, integrity classifications, and deposit procedure.

This makes artifact-assisted saved-evidence checking possible **on this machine**. The source repository still omits these ignored files, and no public archive deposit or DOI exists. Availability to a reader requires the separate bundle; local presence is not public release completeness.

The bundle preserves exact baseline Git-blob bytes. Seven hash-bound text files in this Windows working checkout differed only through LF-to-CRLF conversion during the audit; all their baseline Git blobs matched the frozen hashes. Use the archive bytes or inspect this explicitly when a strict filesystem-byte verifier fails. Do not change frozen hash values to accommodate checkout transformations.

The final packaging verification also detected that `git archive` applied Windows text conversions in the initial local ZIP. The corrected builder reads every source payload with `git show`; all source members of the rebuilt ZIP were independently compared with exact baseline blobs. This is a packaging correction, with no change to scientific data or inference.
