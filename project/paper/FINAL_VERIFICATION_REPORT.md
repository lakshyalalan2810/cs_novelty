# Independent final verification

Date: 2026-10-08. Reviewer role: independent final verification, separate from manuscript editing, science review, literature review, archive preparation, and page rendering. The reviewer read the complete supplied task and all four independent audit reports, examined the manuscript/bibliography and final diff, and performed the read-only checks below. This report is the only project file written by this reviewer.

**Final fingerprint checkpoint: PASS.** The locked PDF, all build inputs, retained compiler/BibTeX logs, UTF-8 visual report, current paper audit, and change log were independently checked together. The final PDF matches the exact artifact inspected by the separate page reviewer.

## Baseline and preservation

- `HEAD` and `origin/main` are both `4b7c51dabff16a702d230f2b9f87c14f73ebf9fd`; branch `main`, subject `Cleanup`.
- No tracked controller/source, results, data, or scientific configuration changes were found. Changes to the two paper generators concern schematic routing, label wording, and layout only.
- No `replication_runs`, `manuscript_reconciliation`, or `training_seed_replication` directory was found outside Git internals. No later research was reconstructed.
- No scientific simulation, retraining, calibration, repair run, large confirmatory resampling, commit, push, pruning, or reflog destruction was performed by this reviewer.
- Original and corrected historical artifacts remain distinct. The publication archive uses the exact frozen scientific baseline; the edited manuscript and final PDF accompany it separately.

## Independent scientific checks

The science reviewer reconstructed all nine paired endpoints, effects, exclusion IDs, table values, historical contrasts, and safety counters. This verifier checked that each required correction reached the manuscript and independently reconstructed the most consequential conclusions from the original restored rows:

| Check | Independent result |
|---|---|
| H8/H9 clean-start endpoint | 124 retained pairs, 26 excluded candidates; C3 27/124 versus 0/124 for each V4 variant; delta 0.21774193548387097 |
| H8/H9 by training seed | Seed 2026: 8/43 versus 0/43; 2027: 10/34 versus 0/34; 2028: 9/47 versus 0/47 |
| H5 null interpretation | 264 retained pairs; zero recorded recoveries in either controller, rather than equally successful recovery |
| Holm family | Independent monotone step-down recomputation matches all nine saved adjusted p-values; only H8/H9 survive |
| Frozen bindings | All manifest/correction/plan/runner checks in the read-only freeze suite pass; SQLite provenance counts preserve the 2,700 repaired versus 9,500 unrepaired cells |

All required science-review corrections are present: 11,650 original-umbrella plus 550 supplemental cells; S2's trusted speed median; historical EKF versus V4 E2 identities; correctly paired development RMSEs; 105 C4 development cells/35 C3-C4 pairs; three enabled V4 changes and disabled recovery boost; deferred component ablations; comparison of complete model/calibration/controller packages; debounced entry versus instantaneous substitution; selected-feedback main history; no direct witness speed input without causal-independence claims; historical auxiliary metrics; mixed Part B entry endpoint; four-condition Figure 4 aggregation and denominators; durations/onsets/noise/censoring; three versus eleven training seeds; reference clusters; repeated 2.03-s latches; and corrected separate schematic residual paths.

Abstract, results, discussion, and conclusion preserve the tested 0.15 N·m condition, retained-pair population, and H8/H9-only support. H1/H2 remain favorable but unsupported; H3/H4/H6/H7 remain unfavorable; H5 remains null; C4 remains NO_GO; EKF remains BASELINE_ONLY; training-seed sensitivity and healthy noiseless current are disclosed. Historical timing is identified separately. No hardware, formal isolation/stability, general detection/recovery/tracking improvement, or real-time 20 Hz claim is made.

## Bibliography, tables, figures, and inclusions

Independent parsing found 16 unique BibTeX entries, all 16 cited, no missing or unused entries, no duplicate DOI, no duplicate labels, no undefined referenced label, and every input table/graphic present. The literature audit supplies primary-source metadata verification and explains the corrected Bonassi DOI/year and replacement Gustafsson book record. The final verifier checked integration and citation roles; it did not claim a new exhaustive literature or retraction search.

The three unchanged numeric table files remain baseline artifacts. The per-seed table's changed header correctly identifies conditional post-event entry; its values are preserved. Figure-generator changes contain no numeric/selection/aggregation changes. The manifest's canonical LF generator SHA-256 and every listed CSV source hash independently match. No favorable representative trajectory was added.

## Archive integrity and defect resolved during final verification

The initial local ZIP was internally self-consistent but contradicted its exact-source-byte claim: **503 of 608 source members contained Windows CRLF transformations** introduced by `git archive`. This verifier found the discrepancy by comparing each member with a fresh `git show BASELINE:path` result. No non-newline difference and no scientific-artifact corruption was found.

The archive preparer corrected the builder to enumerate `git ls-tree` paths and read committed bytes directly through `git show`. This verifier then repeated the complete independent comparison:

- **608/608** source members match baseline Git blobs byte for byte.
- **55/55** original ignored files match ZIP, working checkout, and the original backup; this is 54 scientific artifacts plus the 507-byte final runtime provenance snapshot.
- All available frozen dataset/run/event/checkpoint/status hashes match; all member hashes, external/internal inventories, detached ZIP checksum, membership uniqueness, and ZIP CRC checks pass.
- The 45 original model configuration/metrics/training-summary JSONs match backup, baseline, and working checkout semantically.
- Model/history hashes are newly inventoried detached hashes, explicitly distinguished from preregistered/frozen standalone hashes that do not exist.

Final archive: **666 entries; 50,309,924 ZIP bytes**. SHA-256:

`96f390e72a72b11b1c6ca106f18b6ff463ca894c49c3a05cc9202ab70b626743`

The archive is local and not deposited; its inventory has `doi: null`. Source-only clones can build saved paper assets and regenerate tables, but require the separate sweep CSV for full figure generation and the raw matrices for independent endpoint reconstruction. The partial recorded environment does not establish bitwise scientific re-execution. These boundaries are stated consistently in the manuscript, release report, archival policy, and reproducibility documents.

## Build and visual verification boundary

The lead's real Tectonic/XeTeX, BibTeX, and xdvipdfmx transcript demonstrates repeated passes until references converge. This verifier independently parsed the locked nine-page PDF, all 16 bibliography items, all three complete 64-character provenance hashes, and the retained final-pass LaTeX/BibTeX logs. Final counts are zero undefined references/citations, missing assets, PDF-generation errors, replacement glyphs, overfull boxes, font warnings, and underfull vertical boxes; 13 underfull horizontal boxes remain and were visually inspected. The informational float-only-page warning refers to the intentional results page 7.

The independent layout reviewer inspected every locked final color page and all six figures in grayscale. Its report identifies exactly the independently measured PDF hash below, reports zero collisions/clipped labels, and confirms all figures precede the conclusion. Its font audit and the build report distinguish 19 unique embedded base fonts from 24 embedded font dictionary objects; there are no Type 3 or unembedded fonts. The UTF-8 normalization of the report changed documentation encoding only, with no PDF/source change.

## Exact independent commands and assertion checks

Commands were run from the repository root unless the working directory is specified. `python` resolved to `C:\Users\Lakshya\anaconda3\python.exe`; PDF parsing used the bundled runtime because Anaconda did not contain `pypdf`/PyMuPDF.

```powershell
git rev-parse HEAD
git rev-parse origin/main
git branch -vv
git status --short
git diff --stat
git diff --check
git diff -- project/paper/main.tex
git diff -- project/scripts/v4_paper_tables.py project/scripts/v4_paper_figures.py
git diff -- README.md project/README.md context project/ARCHIVAL_ARTIFACTS.md project/REPRODUCIBILITY.md .gitignore
```

From `project/`:

```powershell
python -m unittest tests.test_context_consistency tests.test_v4_confirmatory_freeze tests.test_v4_statistics_audit -v
```

**Result: 9 tests PASS, repeated successfully after the final documentation/build lock.** This included the tiny synthetic 200-replicate clustering fixture; it was not a confirmatory resampling or a scientific experiment. The lead's paper-generator/compile checks are recorded separately in `../PAPER_FINAL_AUDIT.md` and `FINALIZATION_CHANGELOG.md`; this verifier did not rerun writers against the frozen evidence while the final build was underway.

The exact comprehensive archive assertion command, repeated after the packaging fix, was:

```powershell
@'
from pathlib import Path
import hashlib,json,zipfile,subprocess
root=Path.cwd();release=root/'project/paper/release'
base='4b7c51dabff16a702d230f2b9f87c14f73ebf9fd'
sha=lambda b:hashlib.sha256(b).hexdigest()
zip_path=release/'v4-frozen-evidence.zip'
zip_hash=(release/'v4-frozen-evidence.zip.sha256').read_text().split()[0]
assert sha(zip_path.read_bytes())==zip_hash
inventory=json.loads((release/'v4-artifact-inventory.json').read_text())
assert inventory['baseline_commit']==base and inventory['doi'] is None
paths=subprocess.check_output(['git','ls-tree','-r','--name-only','-z',base]).decode().rstrip('\0').split('\0')
expected=json.loads(subprocess.check_output(['git','show',base+':project/results/v4/confirmatory/result_manifest.json']))['artifacts']
expected.update(json.loads(subprocess.check_output(['git','show',base+':project/results/v4/prereg/preexecution_hashes.json']))['files'])
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==666
 assert z.read('SHA256SUMS.txt')==(release/'SHA256SUMS.txt').read_bytes()
 assert json.loads(z.read('V4_ARCHIVE_INVENTORY.json'))==inventory
 for entry in z.read('SHA256SUMS.txt').decode().splitlines():
  digest,name=entry.split('  ',1);assert sha(z.read(name))==digest,name
 for name in paths:assert z.read(name)==subprocess.check_output(['git','show',base+':'+name]),name
 for row in inventory['scientific_artifacts']:
  name=row['path'];data=z.read(name)
  assert sha(data)==row['sha256'] and len(data)==row['bytes'],name
  assert (root/name).read_bytes()==data,name
  assert (Path('C:/Users/Lakshya/OneDrive/Desktop/antenna/cs_novelty')/name).read_bytes()==data,name
  relative=name.removeprefix('project/')
  if relative in expected:assert sha(data)==expected[relative],relative
 assert len(paths)==608 and len(inventory['scientific_artifacts'])==55
 assert sum(r['bytes'] for r in inventory['scientific_artifacts'])==35192703
print('PASS exact baseline/source/artifact/checksum/CRC integrity:608 exact Git blobs,55 original ignored files,666 members; ZIP bytes='+str(zip_path.stat().st_size)+' SHA256='+zip_hash)
'@ | python -
```

Other successful read-only inline assertions compared all 45 model metadata JSONs; parsed citation keys/DOIs/labels/includes; checked the figure-manifest source hashes; checked prohibited directories and frozen science paths; reconstructed H8/H9 and H5 from CSV dictionaries; and recomputed all nine Holm values. The first load-row assertion used `fault_kind='none'`, which selected no rows; inspecting the actual CSV corrected the checker to the saved `fault_kind='load'`. No scientific file was modified to satisfy a checker.

## Final fingerprint and verdict matrix

| Locked artifact | SHA-256 |
|---|---|
| `main.pdf` — 9 pages, 172,857 bytes | `15b22cb06d2cffc77f04b1cc2eff2015b6ca3e65d7219780920dd55511512e24` |
| `main.tex` | `d1622bcc0417e09bafeaf2a10a52e41c93a1b76565699dba0fbb662c13f647c1` |
| `references.bib` | `4aea1e43adf1f15393ef0633050b8dc798d1110530bdddc316c6f981f344b4bf` |
| `COMPILE_LOG.txt` | `af1c3f3298b38e16e5c01868594a1d8c4f5517bdefaeef858fae3cea503efbac` |
| `BIBTEX_LOG.txt` | `0d78a60abc96272e29fc0afda82e8681d243d9ddf183989ae112163835c94c7b` |

Every input hash in `BUILD_REPORT.json` matches its current file; the delivered PDF equals the build-output PDF byte for byte. The current audit and final visual report cite the same locked PDF hash. Final Git checks preserve HEAD/origin and all frozen scientific paths; `git diff --check` passes. The exact final fingerprint assertion command was:

```powershell
@'
from pathlib import Path
import hashlib,json,re,subprocess
from pypdf import PdfReader
root=Path.cwd();paper=root/'project/paper';sha=lambda b:hashlib.sha256(b).hexdigest()
report=json.loads((paper/'BUILD_REPORT.json').read_text(encoding='utf-8'))
pdf=paper/'main.pdf';digest=sha(pdf.read_bytes())
assert digest==report['pdf']['sha256']=='15b22cb06d2cffc77f04b1cc2eff2015b6ca3e65d7219780920dd55511512e24'
assert pdf.stat().st_size==report['pdf']['bytes']==172857
assert pdf.read_bytes()==(root/'tmp/paper_build_final/main.pdf').read_bytes()
for name,expected in report['input_sha256'].items():assert sha((paper/name).read_bytes())==expected,name
log=(paper/'COMPILE_LOG.txt').read_text(encoding='utf-8');blg=(paper/'BIBTEX_LOG.txt').read_text(encoding='utf-8')
assert sha((paper/'COMPILE_LOG.txt').read_bytes())==report['final_tex_log_sha256']
assert sha((paper/'BIBTEX_LOG.txt').read_bytes())==report['bibtex_log_sha256']
assert 'Overfull' not in log and 'LaTeX Font Warning' not in log
assert log.count('Underfull \\hbox')==13 and log.count('Underfull \\vbox')==0
assert not re.search(r'Citation .* undefined|Reference .* undefined|There were undefined references|LaTeX Error|Missing character',log)
assert not re.search(r'Warning--|error message',blg)
p=PdfReader(pdf);text='\n'.join(page.extract_text() for page in p.pages)
assert len(p.pages)==9 and text.count('\ufffd')==0 and len(re.findall(r'^\[\d+\]',text,re.M))==16
for token in report['structure']['full_provenance_hashes']:assert token in text,token
tex=(paper/'main.tex').read_text();bib=(paper/'references.bib').read_text()
keys=re.findall(r'@\w+\s*\{\s*([^,]+)',bib)
citations=set(k.strip() for body in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for k in body.split(','))
assert len(keys)==len(set(keys))==16 and citations==set(keys)
labels=re.findall(r'\\label\{([^}]+)\}',tex);refs=re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',tex)
assert len(labels)==len(set(labels))==31 and not set(refs)-set(labels)
visual=(paper/'VISUAL_QA_REPORT.md').read_text(encoding='utf-8')
assert digest in visual and '**VISUAL_LAYOUT: PASS.**' in visual and '**LATEX_BUILD: PASS.**' in visual
for name in ['SCIENCE_CONTENT_AUDIT.md','LITERATURE_POSITIONING_AUDIT.md','PUBLICATION_RELEASE_REPORT.md','VENUE_READINESS.md','FINALIZATION_CHANGELOG.md','VALIDATION_COMMANDS.md']:
 assert (paper/name).is_file();(paper/name).read_text(encoding='utf-8')
audit=(root/'project/PAPER_FINAL_AUDIT.md').read_text(encoding='utf-8')
assert digest in audit and 'NOT_READY' in audit and 'Historical' in audit
changes=subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()
for name in changes:assert not name.startswith(('project/src/','project/results/','project/data/','project/config/','project/configs/')),name
base='4b7c51dabff16a702d230f2b9f87c14f73ebf9fd'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==base
assert subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()==base
assert sha((paper/'release/v4-frozen-evidence.zip').read_bytes())=='96f390e72a72b11b1c6ca106f18b6ff463ca894c49c3a05cc9202ab70b626743'
print('PASS final lock:9 pages,172857 bytes,PDF/build-input/log hashes match;16 cites,31 labels;0 overfull/font/undefined/errors,13 underfullh;UTF8 visual report exact locked hash and PASS;current audit/changelog present;HEAD/origin/frozen paths/ZIP unchanged')
'@ | & 'C:\Users\Lakshya\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -
```

Final verdicts:

| Category | Verdict | Meaning |
|---|---|---|
| SCIENCE_INTEGRITY | PASS | Frozen evidence and bounded conclusions preserved |
| MANUSCRIPT_CONTENT | PASS | Required evidence-preserving reviewer corrections integrated |
| LATEX_BUILD | PASS | Real converged build, assets and citations resolved |
| VISUAL_LAYOUT | PASS | Every locked PDF page and all figures inspected; exact hash match |
| LITERATURE_COVERAGE | PASS_WITH_ACTIONS | Verified focused coverage; author/retraction and venue reference checks remain |
| REPRODUCIBILITY_RELEASE | PASS_WITH_ACTIONS | Original bundle verified locally; public deposit/licensing/identifier remain |
| VENUE_READINESS | PASS_WITH_ACTIONS | Justified IEEE Access recommendation; target decision, template and human metadata remain |
| REPOSITORY_CONSISTENCY | PASS | Current audit/change log, historical labels, source/archive boundaries and final consistency checks agree |

**Overall: NOT_READY.** Public archive release, chosen-venue formatting, and human-supplied metadata/declarations remain unfinished. A reviewed conference-format PDF and local ZIP do not make an IEEE Access submission package.

Skipped: scientific reruns, retraining, broad historical test suites, exhaustive literature/retraction searches, hardware, printed hardcopy, and venue upload validation.
Risk: inference remains bounded to complete frozen packages, three common training clusters, the tested healthy-current simulation, and the preregistered clean-start load endpoint; reader access still depends on publishing the archive.

## Lead agent's closing checks

The independent reviewer completed the locked fingerprint assertions and wrote
the final verdict matrix above before its final response hit an account usage
limit. The lead subsequently labeled the remaining manuscript-status row in
`context/PROJECT_SNAPSHOT.md` explicitly historical, checked that edit, and
prepared/validated the binary review patch. Those closing documentation/patch
checks are performed by the lead, rather than attributed to a fresh independent
reviewer turn. All three context tests passed again after the final historical
label change. The locked manuscript, PDF, science, and archive were not changed.
