# Packaging validation commands — 2026-10-08

Run from the repository root unless another working directory is stated.
The commands below operated on saved evidence and manuscript files only.
Temporary tools/scripts and rendered images are under ignored `tmp/`.

## Independent evidence and literature checks

```powershell
& 'C:\Users\Lakshya\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tmp/access_packaging/check_archive.py
& 'C:\Users\Lakshya\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tmp/access_packaging/check_literature.py
```

Archive: CRC, safe paths, 666 unique entries, all 663 payload hashes, 608 fresh
baseline Git blobs, 55 original ignored files, extracted hashes and three SQLite
integrity checks PASS. No credential-pattern hits; historical machine paths and
verifier logs are explicitly documented in release/ARCHIVE_VERIFICATION.json.

Literature: 16 original scholarly entries all cited; 15 current Crossref DOI
records plus primary Holm issue record verified. Retraction Watch: 73,005 records,
zero DOI/title matches; no Crossref retraction relation. See LITERATURE_FINAL_CHECK.md.
Initial concurrent metadata requests hit HTTP 429/500; sequential retries obtained
the missing records. A console encoding failure after saving the results was
resolved by rerunning with UTF-8/cached records. No bibliography correction was
necessary. The additional current-pass Codex system reference uses its official URL.

## Existing focused tests

```powershell
& 'C:\Users\Lakshya\anaconda3\python.exe' tmp/run_final_checks.py
```

The existing wrapper executes, from `project/`:

```powershell
python -m pytest tests/test_v4_paper.py tests/test_context_consistency.py tests/test_v4_statistics_audit.py tests/test_v4_confirmatory_freeze.py -q
```

Result: **13 passed, 22 subtests passed in 15.53 s**; exit 0. Exact output is
IEEE_ACCESS_TEST_OUTPUT.txt. The wrapper restores original figure bytes after
saved-input figure checks; the before/after SHA check confirms no frozen or
audited file changed. The harmless Anaconda executable-location diagnostic is
retained in the output. No training/simulation/large experimental suite ran.

## Official-template build

The official 2026-05-13 template uses pdfTeX-only spotcolor. An initial Tectonic
attempt correctly failed that engine requirement; it was not used for the PDF.
A portable official TinyTeX 2026.10 tree was extracted under
`tmp/access_packaging/TinyTeX` (no global installation/PATH change). Release
SHA-256: `36c329c644683d981a668419d13c8804696e2816ae4d287ad755788fee231a31`.
Local package manager self-update and `cite`/`courier` installs supplied missing
standard dependencies; no repository/runtime dependency file changed.

```powershell
& 'C:\Users\Lakshya\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tmp/access_packaging/build_pdf.py
Copy-Item -LiteralPath tmp/access_packaging/build/main.pdf -Destination project/paper/submission_ieee_access/main.pdf
pdftoppm -r 135 -png project/paper/submission_ieee_access/main.pdf tmp/access_packaging/rendered/page
```

The exact resolved executable, arguments, working directories and exit codes are
in IEEE_ACCESS_BUILD_COMMANDS.json. Sequence: pdfLaTeX, BibTeX, pdfLaTeX,
pdfLaTeX. All four exit 0. Build intermediates stayed outside staging. Final
log/BibTeX log are IEEE_ACCESS_COMPILE_LOG.txt and IEEE_ACCESS_BIBTEX_LOG.txt.
The final ten images were visually inspected; see IEEE_ACCESS_VISUAL_QA.md.

## Preservation and clean bundle self-check

```powershell
& 'C:\Users\Lakshya\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tmp/access_packaging/finalize_package.py
```

Assertions compare scientific body/abstract, equations/captions, original
bibliography, figure/table bytes, official template dependencies, labels/citations,
embedded fonts and complete printed hashes. They compare every pre-existing
snapshotted file; only the two audit/report status-pointer additions are allowed.
The source ZIP is checked for unique safe entries, CRCs and exact extracted bytes.
It excludes the separately supplied final PDF and all LaTeX intermediates.
Outputs and exact hashes are in IEEE_ACCESS_BUILD_REPORT.json and the ZIP sidecar.

## Git review (no commit/push)

```powershell
git status --short
git diff --stat
git diff
```

Outputs are retained in `tmp/access_packaging/git-status-short.txt`,
`git-diff-stat.txt` and `git-diff.txt`. The requested literal `git diff` failed
with `error: waitpid for astextplain failed: No child processes` / `fatal: unable
to read files to diff`. Its partial output is git-diff-textconv-partial.txt.
The full diff was obtained with this safe fallback (no Git setting change):

```powershell
git diff --no-textconv --no-ext-diff
```

The cumulative stat is 28 tracked files, 629 insertions, 176 deletions; most
changes predate this packaging pass. Normal LF/CRLF checkout warnings are retained.
Untracked new packaging files are listed separately in
SUBMISSION_PACKAGING_CHANGED_FILES.txt because ordinary Git diff omits them.
