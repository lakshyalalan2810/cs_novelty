# Finalization validation commands — 2026-10-08

Repository root: `C:\Users\Lakshya\OneDrive\Desktop\cs_novelty\cs_novelty`.
Commands below are the actual build/check entry points used. Inline independent
CSV/hash assertions are described in the science and final-verification reports;
they read saved evidence and do not invoke simulation/training writers.

## Baseline and scope

```powershell
git rev-parse HEAD
git rev-parse origin/main
git status
git branch -vv
git diff --name-only -- project/src project/results project/data project/models project/V4_PREREGISTRATION.md project/V4_PREEXECUTION_AMENDMENT.md project/V4_EXECUTION_PLAN_REFREEZE_NOTE.md
git diff --check
```

HEAD and origin/main both returned
`4b7c51dabff16a702d230f2b9f87c14f73ebf9fd`; initial working tree was clean.
Recursive directory inspection found no `replication_runs`,
`manuscript_reconciliation`, or `training_seed_replication` in this checkout.
Historical `training_seed_robustness` is part of the baseline and was preserved.
The scientific-path diff check returned no changed files.

## Toolchain and PDF

Downloaded the official portable Tectonic 0.17.0 Windows release to
`tmp/paper_tools`; no system TeX installation or research-environment replacement
was performed. Downloaded ZIP SHA-256:
`f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f`.

```powershell
Invoke-WebRequest -Uri 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip' -OutFile 'tmp/paper_tools/tectonic.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath 'tmp/paper_tools/tectonic.zip'
Expand-Archive -LiteralPath 'tmp/paper_tools/tectonic.zip' -DestinationPath 'tmp/paper_tools' -Force
& 'tmp/paper_tools/tectonic.exe' --version
```

Before first PDF creation, the PDF skill's operation marker ran successfully:

```powershell
& 'C:\Users\Lakshya\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' 'C:\Users\Lakshya\.codex\plugins\cache\openai-primary-runtime\pdf\26.1007.11041\skills\pdf\container_tools\mark_artifact_operation_started.mjs' --operation-kind create --expected-output-count 1 --output-format pdf
```

The baseline was built from preserved source/assets in `tmp/paper_source_before`
with this command, then every one of its eight pages was rendered and inspected:

```powershell
& '../paper_tools/tectonic.exe' --keep-logs --keep-intermediates --outdir '../paper_build_before' main.tex
```

Successful final build from `project/paper`:

```powershell
$env:FONTCONFIG_FILE = (Resolve-Path -LiteralPath '../../tmp/fonts.conf').Path
& '../../tmp/paper_tools/tectonic.exe' --keep-logs --keep-intermediates --outdir '../../tmp/paper_build_final' main.tex
Copy-Item -LiteralPath '../../tmp/paper_build_final/main.pdf' -Destination 'main.pdf' -Force
pdfinfo main.pdf
```

Tectonic ran real TeX, BibTeX, repeated TeX passes until auxiliary files
converged, and xdvipdfmx. The retained final log is `COMPILE_LOG.txt`; parsed
diagnostics and source/output hashes are in `BUILD_REPORT.json`.

```powershell
pdftoppm -r 135 -png project/paper/main.pdf tmp/pdf_final_pages/page
```

The independent layout reviewer also rendered every final page, inspected
grayscale versions of all figure-bearing pages, and enumerated embedded PDF
font dictionaries using bundled `pypdf`. Read-only checks inspected all labels,
cross-references, citation keys, include paths, full provenance identifiers,
final BibTeX diagnostics, and generated table values. These are not inferred
from the compiler exit status alone.

## Saved-evidence and generator checks

From `project/`:

```powershell
python scripts/v4_paper_tables.py
python -m pytest tests/test_v4_paper.py tests/test_context_consistency.py tests/test_v4_statistics_audit.py tests/test_v4_confirmatory_freeze.py -q
python -c "import runpy; x=runpy.run_path('scripts/v4_paper_figures.py'); x['fig_architecture'](); x['fig_hypothesis_effects']()"
python -c "import runpy; x=runpy.run_path('scripts/v4_paper_figures.py'); x['fig_conditional_detection']()"
python -m py_compile scripts/v4_paper_figures.py scripts/v4_paper_tables.py paper/release/prepare_v4_archive.py
```

Focused suite: **13 tests and 22 subtests passed**, including complete table and
figure generation from the tracked inputs plus restored original sweep matrix.
The suite was repeated after final table/document changes and passed again.
Final repeat wrapper was `python tmp/run_final_checks.py`, which invoked the
exact pytest command above and restored the saved figure bytes after generation
to keep the built PDF's input assets fixed. Its backup/restore hashes were
asserted. Independent verifier additionally ran the context/freeze/statistics
unittest modules: nine tests passed.

Before the final presentation corrections, four unchanged numerical/schematic
figure assets were compared to regenerated assets by rendering both with Poppler
at 1800-pixel extent and checking PIL pixel differences: all pixels matched.
Only the three explicitly corrected graphics
are changed in the final diff (architecture, conditional legend, effect axis).

## Archive

From repository root:

```powershell
python project/paper/release/prepare_v4_archive.py 'C:\Users\Lakshya\OneDrive\Desktop\antenna\cs_novelty\project'
python -m py_compile project/paper/release/prepare_v4_archive.py
Get-FileHash -Algorithm SHA256 -LiteralPath 'project/paper/release/v4-frozen-evidence.zip'
```

The final builder uses `git ls-tree -r --name-only BASELINE` and
`git show BASELINE:path` for source bytes. Two independent reviewers compared
every one of its 608 source payloads with fresh Git-blob bytes, all 55 ignored
files with originals/local copies, all 663 checksum entries, and ZIP CRCs.
The builder's self-check recreates the ZIP and asserts deterministic bytes,
complete membership, CRC integrity and every extracted SHA-256.

## Diagnosed environment/build attempts

- A sandboxed toolchain download failed DNS resolution; the authorized official
  download and package fetch succeeded with network-enabled execution.
- Initial compilation was interrupted only to move it to a preserved baseline
  copy before manuscript edits. Final baseline compilation completed.
- A sandboxed final build could not locate the external user's bundle cache;
  the final successful build used the downloaded official bundle/cache.
- Initial XeTeX font warnings were fixed by selecting T1 encoding before the
  IEEE class, and a task-local fontconfig file removed missing-config errors.
- Two authoring commands used the wrong working directory; their build outputs
  did not exist and were not counted as successful validation.
- A trial compact-table header was repaired because `\\[... ]` was parsed as
  optional line spacing; the final generator groups the literal CI brackets.
- Trial float/page-break builds were inspected and replaced by the final
  typography build. Their warnings are not substituted for the final log.
- `pdffonts` was not on PATH; actual font embedding was inspected using `pypdf`.
- Initial ZIP creation via `git archive` transformed 503 source text members to
  CRLF. Independent verification caught it; exact Git-blob packaging replaced
  it and every source member was reverified. Scientific bytes were unchanged.

No dataset generation, model retraining, calibration, protocol execution,
confirmatory simulation, historical inference writer, commit, push, upload,
submission, or DOI creation was performed.

## Final saved-deliverable checks and review diff

After the last historical-status wording change, from `project/`:

```powershell
python -m pytest tests/test_context_consistency.py -q
```

All three context tests passed again (0.09 s). This is a focused documentation
recheck, not an additional full scientific or historical test-suite run.

From the repository root:

```powershell
& 'C:/Users/Lakshya/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' tmp/record_final_build.py
python tmp/write_final_audit.py
python tmp/write_final_review_diff.py
git diff --check
```

The assert-based build recorder reads the locked PDF, final logs, manuscript,
bibliography and assets; checks all citations/labels/includes/full hashes,
font embedding and unchanged baseline/scientific paths; then retains the logs
and machine-readable source/output fingerprints. There are 19 unique font names
in 24 embedded font dictionary objects. This is a font-counting distinction,
not an unembedded-font discrepancy. The visual report's original Windows-1252
en dashes were normalized to UTF-8 without changing its text or the PDF.

The diff writer captures raw bytes from
`git diff --no-textconv --no-ext-diff --binary HEAD` and appends
`git diff --no-index --no-textconv --no-ext-diff --binary -- /dev/null PATH`
for each new Git-visible file.
Return code 1 means a new-file diff exists. It excludes only the delivered
patch itself, writes no index/commit, and asserts exact path coverage against
the change log. The final review patch is checked against a temporary exact
baseline source copy; scientific artifact restoration and the ignored local ZIP
are separately inventoried, rather than included in the Git patch.
An initial patch-generation attempt triggered the configured PDF text-conversion
helper, whose Windows shell was unavailable in the sandbox. Disabling external
diff/text-conversion helpers retains actual binary PDF changes and avoids that
unnecessary process; the final patch is validated on exact baseline bytes.
