# Finalization change log — 2026-10-08

Baseline `4b7c51dabff16a702d230f2b9f87c14f73ebf9fd` (`main` = `origin/main`, `Cleanup`). No commit or push.
Scientific artifacts, hypotheses, seeds, exclusions, statistics, controllers,
and conclusions are unchanged. Overall publication state: **NOT_READY**.

## Changes

- Polished methods, denominators, post-hoc labeling, acronym definitions, and
  scope/causality language; retained every negative/null finding.
- Expanded Related Work around 16 verified, cited primary/foundational records;
  corrected bibliography metadata and explicitly bounded novelty.
- Repaired full SHA provenance and frozen-table collisions, compact table
  header, figure labels/legend, architecture routing, float order, fonts, and
  final reference-column balance. Only three figure PDFs changed.
- Produced a real 9-page review PDF with zero overfull boxes/font warnings,
  all-page visual/grayscale QA, retained compiler/BibTeX logs and build hashes.
- Restored 55 original ignored files and prepared a deterministic 666-entry
  source/evidence ZIP with inventory/checksums. Fixed Windows archive newline
  conversion; exact baseline Git bytes verified. No scientific data regenerated.
- Added independent science, visual, literature, release, venue, final-verification
  reports and human metadata checklist. Synchronized current docs while retaining
  expressly historical audit/snapshots. Added `/tmp/` to root ignore rules.

## Validation / review boundary

13 focused tests and 22 subtests pass, including figure/table generation;
independent read-only verifier adds nine passing tests. Edited Python files
compile. Real LaTeX/BibTeX references converge; all fonts embedded. Frozen-source
checks and full archive payload/CRC/determinism assertions pass. Exact commands,
failed intermediate attempts, and test boundaries are in
[VALIDATION_COMMANDS.md](VALIDATION_COMMANDS.md). Current evidence/verdicts are in
[PAPER_FINAL_AUDIT.md](../PAPER_FINAL_AUDIT.md).

## Exact Git-visible changed-file list (46 files)

This list combines `git diff --name-only HEAD` and
`git ls-files --others --exclude-standard`, plus the delivered review patch
itself. The patch includes tracked and newly added deliverables, including
binary PDF diffs; it excludes itself to avoid self-reference. No file is staged.

```text
.gitignore
README.md
context/AI_HANDOFF.md
context/DO_NOT_TOUCH.md
context/EXPERIMENTS.md
context/FILE_INDEX.md
context/LIMITATIONS.md
context/OPEN_QUESTIONS.md
context/PAPER_CONTEXT.md
context/PROJECT_SNAPSHOT.md
context/PROVENANCE.md
context/REPOSITORY_MAP.md
context/REPRODUCIBILITY.md
context/RESEARCH_TIMELINE.md
context/START_HERE.md
project/ARCHIVAL_ARTIFACTS.md
project/PAPER_FINAL_AUDIT.md
project/README.md
project/REPRODUCIBILITY.md
project/paper/BIBTEX_LOG.txt
project/paper/BUILD_REPORT.json
project/paper/COMPILE_LOG.txt
project/paper/FIGURE_MANIFEST.md
project/paper/FINALIZATION_CHANGELOG.md
project/paper/FINAL_REVIEW_DIFF.patch
project/paper/FINAL_VERIFICATION_REPORT.md
project/paper/LITERATURE_POSITIONING_AUDIT.md
project/paper/PUBLICATION_RELEASE_REPORT.md
project/paper/SCIENCE_CONTENT_AUDIT.md
project/paper/VALIDATION_COMMANDS.md
project/paper/VENUE_READINESS.md
project/paper/VISUAL_QA_REPORT.md
project/paper/figures/fig_architecture.pdf
project/paper/figures/fig_conditional_detection.pdf
project/paper/figures/fig_hypothesis_effects.pdf
project/paper/main.pdf
project/paper/main.tex
project/paper/references.bib
project/paper/release/.gitignore
project/paper/release/SHA256SUMS.txt
project/paper/release/prepare_v4_archive.py
project/paper/release/v4-artifact-inventory.json
project/paper/release/v4-frozen-evidence.zip.sha256
project/paper/tables/tab_posthoc_per_seed.tex
project/scripts/v4_paper_figures.py
project/scripts/v4_paper_tables.py
```

The ignored local ZIP and restored 55 originals are intentionally outside this
Git diff. Their exact names/bytes/SHA-256 are in
[v4-artifact-inventory.json](release/v4-artifact-inventory.json) and
[SHA256SUMS.txt](release/SHA256SUMS.txt); the ZIP checksum is in
[v4-frozen-evidence.zip.sha256](release/v4-frozen-evidence.zip.sha256).
Toolchain, render PNGs, and temporary check scripts are task intermediates in
ignored `tmp/`, not publication source changes.

## Actions still required

Public artifact deposit and licensing; target/funding decision and venue
formatting; real author metadata/declarations/AI disclosure; coauthor and final
diff approval. The present IEEE conference review draft is not an IEEE Access
submission package. No DOI, email, ORCID, funding fact, or role was invented.
No experiments, retraining, large simulations, commit, push, or upload were run.
