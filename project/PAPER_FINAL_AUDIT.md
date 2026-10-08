> Packaging update, 2026-10-08: a separate ten-page official IEEE Access draft
> and clean source bundle now exist. Current readiness is
> READY_AFTER_HUMAN_METADATA_AND_ARCHIVE_DEPOSIT; see
> [SUBMISSION_PACKAGE_REPORT.md](paper/SUBMISSION_PACKAGE_REPORT.md).
> The audit below records the preceding conference-format review pass. Its
> scientific findings and evidence remain unchanged.

# Final paper audit — preceding 2026-10-08 finalization

**Overall state: NOT_READY for submission.** The manuscript is compiled and
visually reviewed, its saved evidence is independently checked, and the original
ignored artifacts are packaged locally. A public archive deposit, selected
venue's submission format, and human publication metadata remain required.

Baseline: `main` and `origin/main` both remain `4b7c51dabff16a702d230f2b9f87c14f73ebf9fd`
(`Cleanup`, intentionally restored Oct-1 research state). No commit or push was
made. No controller, endpoint, seed, threshold, exclusion, statistic, result,
scientific conclusion, or frozen scientific artifact was modified. No later
replication/reconciliation work, training, simulation, or new inference was run.

## Final verdicts

| Category | Verdict | Evidence / remaining action |
|---|---|---|
| SCIENCE_INTEGRITY | PASS | Original saved rows, frozen bindings, nine endpoints/exclusion sets, and H8/H9-only Holm decisions independently checked |
| MANUSCRIPT_CONTENT | PASS | Reviewer corrections integrated; full-package comparison, post-hoc populations, healthy-current assumption, and adverse/null findings explicit |
| LATEX_BUILD | PASS | Real repeated XeTeX/BibTeX build; no undefined citation/reference, missing asset, font warning, or PDF-generation error |
| VISUAL_LAYOUT | PASS | Every final page inspected; full hashes, single-column post-hoc table, wide frozen table, axis labels, legends, and float order repaired |
| LITERATURE_COVERAGE | PASS_WITH_ACTIONS | 16 verified and cited entries cover the ten requested areas; authors should perform the venue's final reference/retraction review |
| REPRODUCIBILITY_RELEASE | PASS_WITH_ACTIONS | Deterministic original-artifact bundle verified locally; public deposit, artifact licensing, and actual persistent identifier outstanding |
| VENUE_READINESS | PASS_WITH_ACTIONS | Three official venue guides compared; provisional IEEE Access recommendation requires target/funding decision, template conversion, and author metadata |
| REPOSITORY_CONSISTENCY | PASS | Current-facing status synchronized; Oct-1 audit and snapshots expressly historical; scientific-path diff empty |

PASS for the scientific audit is a preservation/traceability verdict, not a claim
of broad fault tolerance or submission acceptance. PASS_WITH_ACTIONS identifies
remaining publication work; it does not mean those actions have been completed.

## Science and manuscript content

H8/H9 alone support reduced debounced false reliability entry under the tested
0.15 N·m load condition. Among 124 retained clean-start pairs from 150 candidates,
historical C3 recorded 27 entries and each V4 variant recorded zero; the 26
pre-latched exclusions are the frozen rule. Per-training-seed C3 counts are
8/43, 10/34, and 9/47. H1/H2 remain favorable but unsupported;
H3/H4/H6/H7 remain unfavorable; H5 is null with zero recorded recoveries in both
controllers among 264 retained pairs. C4 remains NO_GO, the historical EKF
comparator BASELINE_ONLY, and historical robustness TRAINING-SEED-SENSITIVE.

The comparison changes complete controller/model/calibration packages. It does
not isolate witness-component causality. Debounced witness-gated entry is
distinguished from instantaneous feedback substitution. Current-informed
witnesses have no direct speed-sensor input, but share a closed loop and assume
the frozen dataset's healthy, noiseless, unquantized current channel. No general
detection, recovery, tracking, isolation, hardware, stability, or 20 Hz claim is
made. All safety counters, including 3,053 slew-limit violations, remain visible.

Methods now specify the selected-feedback main history, S2 speed-median exception,
historical versus V4 EKF identities, correctly paired historical metrics, 105 C4
development cells/35 C3-C4 pairs, three enabled V4 changes with recovery boost
disabled, deferred component ablations, event durations/onsets/noise scales,
censoring, and training/reference populations. The executed core comprises
11,650 original-umbrella plus 550 supplemental H6 cells; 257,260 original-umbrella
cells remain deferred. Prior re-freeze, EKF repair, and statistical correction
are disclosed as history; none was repeated during this task.

All four included numeric tables were checked against saved CSV/JSON. Only the
post-hoc table header changed. All six figures retain their source and selection
rules. Three assets changed for presentation/accuracy: architecture routing,
conditional-detection legend, and tracking-penalty axis label. No numeric values,
aggregation, or favorable trajectory selection changed.

Related Work now distinguishes established MPC, recurrent prediction, residual
diagnosis, analytical redundancy, CUSUM, motor-drive observer isolation, neural
sensor reconstruction, and reliability-aware MPC from this narrow entry study.
All 16 BibTeX entries are cited, with no missing/unused key or duplicate entry.
Primary-source metadata corrections and access limits are documented separately.

## Real build and every-page visual review

Portable official Tectonic 0.17.0 ran real XeTeX, BibTeX 0.99d, repeated passes
until auxiliary files converged, and xdvipdfmx. Final PDF: **9 letter-size
pages**, **172,857 bytes**, SHA-256:

`15b22cb06d2cffc77f04b1cc2eff2015b6ca3e65d7219780920dd55511512e24`

Final-pass diagnostics: **0 overfull horizontal,
0 overfull vertical,
13 underfull horizontal,
0 underfull vertical boxes; 0 font warnings;
0 unresolved citations/references; 0 missing included assets; 0 BibTeX warnings
or errors.** Every underfull paragraph was visually checked. Page 7 contains
result floats by design, with no collision. All 19 PDF fonts
are embedded and none is Type 3. Three complete 64-character provenance hashes
remain present and unchanged in the rendered PDF.

The independent layout reviewer inspected all eight baseline pages, then all
nine final pages individually and figure-bearing pages in grayscale. The known
hash and frozen-table collisions are repaired with double-column tables. The
post-hoc table fits one column at readable size. Figure 5's clipped axis and
Figure 4's undersized legend are repaired. Result floats precede the conclusion,
and the final reference columns are balanced. The report binds this exact PDF
hash; rendering success alone was not treated as a pass.

Retained build evidence: [BUILD_REPORT.json](paper/BUILD_REPORT.json),
[COMPILE_LOG.txt](paper/COMPILE_LOG.txt), [BIBTEX_LOG.txt](paper/BIBTEX_LOG.txt).
Commands and diagnosed intermediate attempts are in
[VALIDATION_COMMANDS.md](paper/VALIDATION_COMMANDS.md).

## Original-artifact release

55 original ignored files were located in an existing local backup and restored:
54 scientific artifacts plus the hash-bound final runtime provenance snapshot.
The dataset and all nine raw run/event/checkpoint files match frozen hashes;
22 model weights and 22 histories have newly inventoried detached hashes and
matching original/baseline model metadata. Those new hashes are not presented as
previously frozen standalone model hashes.

The deposit candidate contains 608 exact baseline Git blobs and the 55 originals,
plus its internal README/inventory/checksums: **666 ZIP entries**, **50,309,924
bytes**. SHA-256:

`96f390e72a72b11b1c6ca106f18b6ff463ca894c49c3a05cc9202ab70b626743`

Independent verification found that an initial `git archive` package applied
Windows newline conversion to 503 source members. The builder now reads exact
`git show` bytes. Two reviewers verified all 608 source members against fresh
Git-blob reads, all 55 ignored files against originals/local copies, all 663
listed payload hashes, ZIP membership/CRCs, and repeat-build determinism. This
was a packaging correction; scientific rows and inference were untouched.

The ZIP's source is the original scientific baseline. The approved final
manuscript/PDF must accompany the deposit separately. No public upload, DOI,
submission, or paid action occurred. A source-only clone can build saved paper
assets and regenerate tables; full figure generation and independent raw-row
inference require the separate bundle. Partial environment records do not prove
byte-identical scientific re-execution. See the [release report and human
metadata checklist](paper/PUBLICATION_RELEASE_REPORT.md),
[archive policy](ARCHIVAL_ARTIFACTS.md), and [reproducibility map](REPRODUCIBILITY.md).

## Validation and remaining submission actions

The focused paper/context/statistics/freeze suite passed **13 tests and 22
subtests**, including table and figure generation with the original sweep rows.
The final repeat restored the saved figure bytes after generation and asserted
their hashes. Independent read-only context/freeze/statistics verification passed
another nine tests. Python compile checks for both edited paper generators and
the archive builder passed. `git diff --check` passed; scientific-path diff is
empty. The historical 87-test/13,746-subtest result below was not rerun here.

The [venue comparison](paper/VENUE_READINESS.md) recommends IEEE Access
provisionally as a Research Article, based on verified official requirements,
and also evaluates JCAES and Transactions of the Institute of Measurement and
Control. Acceptance and funding are not assumed. The current corrected
IEEEtran conference PDF remains a review draft; it is not the Access template.

Required next actions are: choose target and funding route; convert to its
required format and repeat build/all-page QA; supply corresponding author,
emails, ORCID, biographies, accurate declarations/contributions/acknowledgments,
AI-assistance disclosure, and coauthor approval; confirm artifact licensing and
deposit the original bundle; verify uploaded bytes and add only the actual
persistent identifier; approve the final diff before any commit/push. No new
experiment is required for the existing bounded conclusion. The overall state
is NOT_READY because the remaining actions extend beyond human metadata alone.

## Independent reports and complete review package

- [Science/content audit](paper/SCIENCE_CONTENT_AUDIT.md)
- [All-page visual QA](paper/VISUAL_QA_REPORT.md)
- [Literature positioning and verified primary sources](paper/LITERATURE_POSITIONING_AUDIT.md)
- [Release gap and metadata checklist](paper/PUBLICATION_RELEASE_REPORT.md)
- [Venue recommendation](paper/VENUE_READINESS.md)
- [Independent final verification](paper/FINAL_VERIFICATION_REPORT.md)
- [Concise change log and exact changed-file list](paper/FINALIZATION_CHANGELOG.md)
- [Exact commands/tests](paper/VALIDATION_COMMANDS.md)
- [Binary-capable final Git review diff](paper/FINAL_REVIEW_DIFF.patch)

The five requested independent agent roles were used. Their earlier baseline
findings are retained with explicit scope; the final verifier checks resolution
against this current audit and the final PDF fingerprint.

---

# HISTORICAL/SUPERSEDED — Oct-1 precompiler paper audit

The following is the exact logical text of the baseline's previous audit.
Its READY wording, absent-toolchain statement, citation count, anonymity
statement, and historical test totals are superseded by the current audit above.
They describe the earlier audit and are retained to preserve provenance.

# Final Paper Audit

## Status

READY for final human read-through. A local LaTeX executable is unavailable,
so this status is based on complete structural checks, regenerated vector
figures/tables, visual figure inspection, frozen-evidence verification, and
the full V4 test suite. Submission metadata remains anonymous by design.

## Title audit

| Item | Text / assessment |
|---|---|
| Previous title | Reliability-Aware LSTM-MPC With Dual Witnesses for Sensor-Fault-Tolerant DC-Motor Control: A Preregistered Evaluation |
| Recommended and adopted title | Disturbance-Aware Witness Gating for Reliability-Aware LSTM-MPC of a Nonlinear DC Motor |
| Reason | The adopted title centers the supported V4 mechanism and avoids implying that both witnesses operate simultaneously or that general fault tolerance was established. |
| Claim risk | Low. “Disturbance-aware” is bounded in the abstract and paper to the preregistered 0.15 N·m load condition; no universal diagnosis, isolation, hardware, or real-time claim appears. |

## Section map

| Scientific phase | Final location | Status |
|---|---|---|
| Nonlinear PMDC plant | Method / Plant and control law | Present |
| Main LSTM predictive model | Method / Learned predictors | Present |
| Baseline reliability layer | Method / Frozen V3 detector | Present |
| Auxiliary current-informed virtual sensor | Method / Learned predictors | Present |
| V3 dual virtual-sensor arbitration | Method / Frozen V3 detector | Present |
| C4 negative ablation | Method and Results / Historical context | Present as NO_GO |
| EKF baseline | Method / Classical baselines | Present as BASELINE_ONLY |
| Training-seed robustness | Results / Historical context | Present as TRAINING-SEED-SENSITIVE |
| V3 robustness and severity limits | Introduction, Results, Limitations | Present |
| V4 motivation | Abstract, Introduction, evolution figure | Present |
| V4 witness mechanism | Method / V4 detector variants | Present |
| Preregistered confirmatory design | Experimental Protocol | Present |
| H1-H9 results | Results / Confirmatory results | Present |
| Statistical correction and provenance | Experimental Protocol / Reproducibility and provenance | Present |
| Limitations | Limitations | Present and synchronized with LIMITATIONS.md |
| Conclusions | Conclusion | Present and bounded |

## Evidence audit

| Section | Claim checked | Evidence artifact | Status | Action taken |
|---|---|---|---|---|
| Abstract | Only H8/H9 survive Holm; benefit is limited to load-induced false entry at 0.15 N·m | results/v4/confirmatory/final_summary_corrected.json | PASS | Rewritten around the narrow positive result and H1-H7 negatives |
| Contributions | Contribution list matches demonstrated architecture and confirmatory result | Corrected summary plus frozen V1-V3 reports | PASS | Replaced development-task list with six evidence-bounded contributions |
| Methods | Plant, LSTM, auxiliary model, MPC, EKF, detector, and witness descriptions match frozen code/configuration | src/motor_model.py; src/lstm_model.py; src/auxiliary_sensor_model.py; src/ekf_observer.py; scripts/v4_closed_loop.py; calibration artifacts | PASS | Added friction smoothing, MPC weights/tolerance, frozen residual shortcut, current-channel wording, EKF role, and witness-threshold source |
| H1-H9 table | Endpoints, n, delta, CI, raw p, Holm p, effect size, and decision use corrected values | results/v4/confirmatory/hypothesis_table_corrected.csv | PASS | Added generated complete table; H8/H9 visually labeled Supported |
| Discussion | Mechanism interpretation does not claim causal source isolation | Corrected H8/H9 results and historical V3 failure evidence | PASS | Added Discussion separating entry discrimination from general diagnosis |
| Limitations | Simulation, current-channel, seed, deferred-scope, runtime, proof, bootstrap, and repair limits agree | LIMITATIONS.md; final corrected summary; repair audit | PASS | Expanded and synchronized |
| Conclusion | No broader detection, recovery, tracking, hardware, or stability claim | Corrected H1-H9 table | PASS | Rewritten to mechanism-specific conclusion |
| Figures | Six main figures are deterministic, source-grounded, legible, and non-cherry-picked | paper/FIGURE_MANIFEST.md; scripts/v4_paper_figures.py | PASS | Added evolution, H1-H9 effect, and load false-entry figures; repaired two schematic layouts after visual review |
| Statistics | Correct hierarchy, 20,000 replicates, percentile CI, centered empirical p, Holm, Cohen dz, finite-p convention | V4_STATISTICAL_AND_REPAIR_AUDIT.md; statistics correction record | PASS | Added complete method and CI/p non-inversion explanation |
| EKF repair | Exactly 2,700 affected cells rerun by outcome-independent predicate; 9,500 untouched | results/v4/ekf_witness_integrity_repair.json; repair audit | PASS | Disclosed in protocol, limitations, and provenance |
| Re-freeze provenance | Historical plan and administrative re-freeze both retained | V4_EXECUTION_PLAN_REFREEZE_NOTE.md; result manifest | PASS | Added both hashes and restart history |
| Deferred scope | 257,260 umbrella cells are explicitly unexecuted | final_summary_corrected.json | PASS | Listed deferred protocols and prevented extrapolation |

## Claims audit

The repository-facing audit covered paper/main.tex, README.md, and
LIMITATIONS.md. Counts are claim instances or risky-phrase instances reviewed,
not unique words:

| Classification | Count | Result |
|---|---:|---|
| SUPPORTED | 12 | Retained with exact condition, comparator, or frozen historical scope |
| NEEDS QUALIFICATION | 24 | All final risky-term occurrences are now negative, limitation, or explicitly bounded contexts |
| REMOVED/REWRITTEN | 8 | Overbroad title/abstract language, dual-witness implication, causal C4 wording, “required fix” wording, placeholder literature claims, and general-superiority phrasing were removed or rewritten |

Important corrections include replacing “sensor-fault-tolerant” in the title,
changing “dual witnesses” to separately instantiated witnesses, removing the
implication that C4 positively identifies a physical source, and stating that
H3/H4/H6/H7 are unfavorable while H5 is null.

## Number consistency

PASS.

- Core: 12,200 unique cells; fault-free 9,600, sweep 2,000, recovery 600.
- Scope: original umbrella 268,910; deferred 257,260.
- Repair: 2,700 EKF cells repaired; 9,500 untouched.
- Inference: 20,000 bootstrap replicates; family alpha 0.05; Holm H1-H9.
- H1-H9 point estimates, intervals, p-values, adjusted p-values, effect sizes,
  analyzed n, exclusions, and decisions reproduce the superseding corrected
  CSV.
- The historical 0.0719, 0.0711, 0.0768, and 0.2844 values remain only in
  immutable, explicitly historical pre-correction artifacts. They do not
  appear in the manuscript, README, LIMITATIONS.md, corrected artifacts, or
  generated final table.

## Validation and build

| Check | Result |
|---|---|
| Complete V4 tests | PASS: 87 tests, 13,746 subtests |
| Focused statistics/freeze/paper tests | PASS: 10 tests, 22 subtests |
| Historical verifiers | PASS: C4 NO_GO, EKF BASELINE_ONLY, training-seed robustness TRAINING-SEED-SENSITIVE, 43/43 frozen final-robustness artifacts, V3, and saved baseline evidence |
| Python compileall | PASS for src, scripts, and tests |
| Figure generation | PASS: six vector PDFs; all exceed 5 kB and regenerate deterministically |
| Figure visual QA | PASS after correcting architecture/evolution label overlap |
| Include paths | PASS: 0 missing figures/tables |
| Labels/references | PASS: 0 duplicate labels; 0 missing referenced labels |
| Citations | PASS: 0 unresolved citation keys; four verified bibliography entries |
| Brace balance | PASS: 184 opening and 184 closing nonescaped braces |
| LaTeX compilation | NOT RUN: latexmk, pdflatex, and xelatex are not installed; no toolchain was installed |

The absent local LaTeX toolchain is a build-environment limitation, not a
known manuscript-structure defect.
