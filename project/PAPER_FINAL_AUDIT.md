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
