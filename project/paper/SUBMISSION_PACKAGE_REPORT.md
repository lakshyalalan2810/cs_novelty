# Final IEEE Access publication package — 2026-10-08

**Verdict: READY_AFTER_HUMAN_METADATA_AND_ARCHIVE_DEPOSIT.**
The official-template draft and source bundle are complete for author review.
The present PDF contains explicit human-action placeholders and must be completed
and rebuilt before submission. No commit, push, external upload or deposit occurred.

Review artifacts:

- `submission_ieee_access/main.tex` and `submission_ieee_access/main.pdf`
- `submission_ieee_access_source.zip` and its `.sha256` sidecar
- `HUMAN_METADATA_REQUIRED.md`, `AI_DISCLOSURE_AUDIT.md`, `SUBMISSION_DECLARATIONS.md`
- `release/DEPOSIT_INSTRUCTIONS.md` and `release/ARCHIVE_VERIFICATION.json`
- `IEEE_ACCESS_REQUIREMENTS.md`, `LITERATURE_FINAL_CHECK.md`, `IEEE_ACCESS_VISUAL_QA.md`
- `IEEE_ACCESS_BUILD_REPORT.json`, exact build commands/logs and test output

## A. Scientific status

**SCIENCE_INTEGRITY = PASS; MANUSCRIPT_CONTENT = PASS; unchanged.**
The scientific body and abstract match the independently audited review source
after normalizing only the enumerated layout commands: removed conference page
break, added a float flush before Discussion, post-hoc float placement and a
breakable filename. Equations, captions, numerical results and statistical
terminology are preserved. Six figure PDFs and four table sources are byte-identical;
the original 16 scholarly bibliography entries are unchanged. The conference
review source/PDF and all frozen scientific files remain untouched in this pass.

H8/H9 alone remain Holm-supported: reduced load-induced false reliability entry
at 0.15 N·m on the frozen clean-start endpoint, 124/150 retained pairs and 26
pre-latch exclusions (C3 27/124 versus 0/124 for each V4 package). The complete
package comparison does not isolate witness causality. H1/H2 remain favorable
but unsupported; H3/H4/H6/H7 unfavorable; H5 null with zero recoveries in 264
pairs. C4 remains NO_GO; EKF BASELINE_ONLY. Training-seed sensitivity and healthy,
noiseless, unquantized current assumptions remain explicit. No general detection
superiority, fault isolation, universal reliability, hardware readiness,
20 Hz feasibility or formal closed-loop stability claim was added.
No experiment, training, new statistical analysis or scientific simulation ran.

## B. IEEE Access requirements

Sources and access date (2026-10-08) are recorded in IEEE_ACCESS_REQUIREMENTS.md.
The current official template is the 2026-05-13 ZIP linked from the
[initial submission checklist](https://ieeeaccess.ieee.org/authors/submission-guidelines/).
The AI policy is also checked against
[Preparing Your Article](https://ieeeaccess.ieee.org/authors/preparing-your-article/).

| Initial submission item | Status | Evidence / action |
|---|---|---|
| Official Access template; two-column, single-spaced source | PASS | Separate draft; official dependencies unchanged |
| Matching source and PDF; each <=40 MB | PASS | Source ZIP 713,913 bytes; PDF 417,576 bytes; clean extraction/hashes |
| Page guidance | PASS | 10 pages; below recommended 20-page threshold |
| Title, both known authors and existing affiliation | PASS | Names/order and department copied from review manuscript |
| Author consent/order and portal extraction | HUMAN_ACTION_REQUIRED | Both authors approve; verify extraction at upload |
| Corresponding author and actual email | HUMAN_ACTION_REQUIRED | Explicit draft field; no invented identity/address |
| ORCID | HUMAN_ACTION_REQUIRED | Public populated submitting-account ORCID; collect actual IDs for both authors under general IEEE journal guidance |
| Biography for every author after references | HUMAN_ACTION_REQUIRED | Two structured fill-in forms and visible draft placeholders |
| Abstract and keywords | PASS | Preserved one-paragraph abstract, 192 whitespace words; seven alphabetized terms |
| References | PASS | 16 scholarly references exist/cited; zero current retraction matches; one additional cited official Codex system entry |
| Supplementary evidence available for review | HUMAN_ACTION_REQUIRED | Deposit unchanged evidence ZIP; insert actual published DOI and confirm portal designation |
| Data/code availability | HUMAN_ACTION_REQUIRED | Draft prepared; sharing encouraged by guidance; public deposit required by this project's release gate |
| Generated-content acknowledgment and section citations | HUMAN_ACTION_REQUIRED | Current pass disclosed/cited; earlier Related Work and other rewriting history needs author confirmation |
| Funding/support, applicable portal declarations | HUMAN_ACTION_REQUIRED | No funding, competing-interest or contribution facts assumed |
| Article type | PASS | Research Article is the proposed type |
| Originality, rights and no concurrent submission | HUMAN_ACTION_REQUIRED | Both authors confirm actual circumstances |
| Acceptance-stage deliverables distinguished | PASS | Photos, graphical abstract, FINAL Article.pdf, copyright and APC processing kept out of initial requirements |

No technical BLOCKED item remains. The missing mandatory human fields and public
identifier prevent READY_TO_SUBMIT. The official template's fonts/logos retain
their notices and are supplied for submission compilation; they are not relabeled
MIT and are not added to the frozen public evidence archive. Public redistribution
requires confirmation of rights or a download-from-IEEE instruction.

Keywords: analytical redundancy; fault detection; long short-term memory
networks; model predictive control; permanent-magnet DC motors; reliability
monitoring; virtual sensing.

## C. Archive status

**Preparation: READY_FOR_DEPOSIT. Public state: LOCAL_ONLY.**
The local ZIP is 50,309,924 bytes and has 666 unique members: 608 exact baseline
source blobs, 55 original ignored artifacts and three internal inventory/checksum/
README files. All 663 payload hashes match; CRC, safe paths, clean extraction and
three read-only SQLite integrity checks PASS. Detached and internal inventories
agree. Fresh Git-blob reads independently verify every baseline source file.

Exact archive SHA-256:
`96f390e72a72b11b1c6ca106f18b6ff463ca894c49c3a05cc9202ab70b626743`.

No credential patterns, private config, Git internals, caches or irrelevant OS
junk were found in the recorded scan. Thirteen historical notebook/report files
contain machine-path text; three historical verifier logs are intentional evidence.
These are explicitly inventoried, not silently deleted or certified absent.
The scan is pattern-based; authors still approve release rights/privacy. Models
were not unpickled and notebooks were not executed for clearance.

Zenodo is recommended after assessment of OSF and VIT repository guidance.
`release/DEPOSIT_INSTRUCTIONS.md` supplies copy-ready title, description, creators,
keywords, GitHub relation, proposed version, publication relation, license rationale,
upload procedure, download/hash verification and all DOI insertion locations.
No authenticated deposit capability/account was available. No DOI was invented.
Suggested CC BY 4.0 for author-owned evidence requires author approval; existing
MIT code licensing remains. This mixed-record choice is not applied automatically.
The archive is unchanged and is kept separate from the <=40 MB manuscript files.

## D. Metadata status

The exact fill-in checklist is HUMAN_METADATA_REQUIRED.md. Remaining inputs:

1. Author order/affiliation approval; corresponding author and actual contact email;
   submitting author/account; both actual ORCIDs and contact details.
2. Each author's actual academic status/degree, institution/role, research interests
   and optional real memberships for approved biographies.
3. Actual funding/support, competing interests, applicable contributions,
   acknowledgments/permissions and author approval/exclusivity declarations.
4. Actual prior AI systems, materially generated/rewritten passages/sections and
   extent of assistance, including documented Related Work replacement and any
   relevant figure/code assistance; approved disclosure and required section citations.
5. Evidence license/rights and public-release clearance; published archive version
   DOI/URL with signed-out access and downloaded SHA verification; portal supplement
   handling. Insert the DOI in the mapped current-facing locations.

After filling these fields, rebuild both PDF and source ZIP and inspect affected
pages. Do not submit the current placeholder files. No author facts were fabricated.

## E. Build status

**LATEX_BUILD = PASS; VISUAL_LAYOUT = PASS for the current draft.**
Proper pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX build: all four exit 0.
PDF: **10 pages, 417,576 bytes**. All ten final Poppler renders inspected;
no collision, clipping, blank page, table/caption problem or hash overflow.
Biographies are readable explicit placeholders. All 19 distinct fonts, including
nested figure fonts, are embedded; none is Type 3. There are no undefined
references/citations, missing assets, BibTeX warnings or font warnings. Seventeen
entries are cited; none is unused. The abstract and scientific body are unchanged.

The log retains 23 official-layout overfull hbox notices (two title boxes and
21 zero-width footer overlays) and 33 underfull spacing notices. All pages were
visually reviewed; none causes content clipping/collision. No meaningful overfull
body/table/hash box remains. The raw warnings are retained for review.

PDF SHA-256:
`cb279271e179238c3f9b82e5fc7ce6f70474d5153074f286ed4569040e459bf8`.

Source ZIP SHA-256:
`e3493af542489d996cbac17b4b273bf5a80d6040eab54237f837682f46943868`.

The staging directory has 45 files, including the PDF; the 44-entry ZIP contains
source/dependencies only. No intermediate LaTeX files, experiments, caches,
private config or unused figures are included. CRC and extracted byte hashes pass.

## F. Files changed

Exact repository-relative paths added/changed **by this packaging pass** are in
SUBMISSION_PACKAGING_CHANGED_FILES.txt. IEEE_ACCESS_BUILD_REPORT.json lists every
staging file and hash. The repository already contained the preceding audit's
uncommitted work; Git's cumulative diff includes those earlier changes.

Of 682 pre-existing snapshotted files, only `project/PAPER_FINAL_AUDIT.md` and
`project/paper/PUBLICATION_RELEASE_REPORT.md` changed, through small status pointers
to this package. The other 680, including the original source/PDF/bibliography,
science outputs, ignored artifacts and frozen ZIP, remain byte-identical.
HEAD remains `4b7c51dabff16a702d230f2b9f87c14f73ebf9fd`; no staged files, commit or push.

## G. Tests

Exact commands, engine provenance and retained outputs are in
IEEE_ACCESS_VALIDATION_COMMANDS.md and IEEE_ACCESS_BUILD_COMMANDS.json.
Existing focused command, working directory `project/`:

```powershell
python -m pytest tests/test_v4_paper.py tests/test_context_consistency.py tests/test_v4_statistics_audit.py tests/test_v4_confirmatory_freeze.py -q
```

**13 passed, 22 subtests passed in 15.53 s; exit 0.** Saved-input figure bytes were
restored by the existing wrapper and independently matched the pre-pass snapshot.
Packaging self-check also passed exact science/asset preservation, all labels and
citations, PDF/font/hash checks, archive immutability, and source-ZIP extraction.
No large scientific suite or simulations were invoked. Ignored raw evidence is
still an archival dependency; it was verified locally, not recreated.

## H. Final verdict

**READY_AFTER_HUMAN_METADATA_AND_ARCHIVE_DEPOSIT**

Complete the human checklist and earlier AI disclosure, approve licenses/rights,
publish and download-verify the evidence archive, insert its actual identifier,
then rebuild/inspect the final matching source/PDF. Acceptance-stage extras are
not initial-upload blockers. Everything remains local for author review.
