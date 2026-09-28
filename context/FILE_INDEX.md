# File index

## Root and project orientation

Read priority:

- P0 — read before interpreting or editing.
- P1 — important current implementation/evidence.
- P2 — supporting artifact or historical matrix.
- P3 — deep provenance, superseded, or optional background.

| Priority | Path | Type | Status | Role |
|---|---|---|---|---|
| P0 | context/START_HERE.md | Markdown | current | first orientation |
| P0 | context/SOURCE_OF_TRUTH.md | Markdown | current | precedence and authority |
| P0 | context/V4_CONFIRMATORY.md | Markdown | current | exact core and H1–H9 |
| P0 | context/STATISTICS.md | Markdown | current | endpoint and corrected inference |
| P0 | context/DO_NOT_TOUCH.md | Markdown | current | preservation boundary |
| P1 | project/scripts/v4_closed_loop.py | Python | current | runtime execution |
| P1 | project/src/reliability_v4.py | Python | current | V4 detector |
| P1 | project/src/mpc.py | Python | current | MPC/PI |
| P1 | project/results/v4/confirmatory/result_manifest.json | JSON | current binding | hashes and environment |
| P1 | project/results/v4/confirmatory/hypothesis_table_corrected.csv | CSV | current result | numeric H1–H9 |
| P1 | project/results/v4/confirmatory/final_summary_corrected.json | JSON | current result | interpretation |
| P1 | project/paper/main.tex | LaTeX | current manuscript | paper claims |
| P2 | project/src/motor_model.py | Python | current | plant |
| P2 | project/src/lstm_model.py | Python | current | main predictor |
| P2 | project/src/auxiliary_sensor_model.py | Python | current | auxiliary witness |
| P2 | project/src/ekf_observer.py | Python | current | EKF witness |
| P2 | project/scripts/v4_protocol_common.py | Python | current | plan/config/checkpoint plumbing |
| P2 | project/V3_DUAL_VIRTUAL_SENSOR_EXTENSION.md | Markdown | historical valid | V3 result |
| P2 | project/TRAINING_SEED_ROBUSTNESS.md | Markdown | historical valid | seed sensitivity |
| P2 | project/FINAL_ROBUSTNESS_AND_SEVERITY_STUDY.md | Markdown | historical valid | boundary study |
| P3 | project/scripts/v4_confirm_analysis.py | Python | historical | old bootstrap |
| P3 | project/results/v4/confirmatory/hypothesis_table.csv | CSV | superseded | old H1–H9 table |
| P3 | project/FINAL_REPO_AUDIT.md | Markdown | superseded | pre-fix audit |

| File | Why it matters |
|---|---|
| README.md | outer repository orientation; current title and bounded story |
| project/README.md | project-level pipeline/status and older-stage details |
| project/PROJECT_PIPELINE_AND_STATUS.md | V1–V3 historical pipeline |
| project/PROJECT_OVERVIEW_AND_PUBLISHABILITY.md | pre-existing untracked overview; stale and not authoritative |

## V4 protocol and provenance

| File | Why it matters |
|---|---|
| project/V4_PREREGISTRATION.md | frozen V4 objective, seeds, hypotheses, umbrella |
| project/V4_PREEXECUTION_AMENDMENT.md | valid pairing and H6 anchor correction before execution |
| project/V4_EXECUTION_PLAN_REFREEZE_NOTE.md | discarded 400-cell attempt and clean re-freeze |
| project/V4_CONFIRMATORY_RESULTS.md | current narrative of exact core and corrected results |
| project/V4_STATISTICAL_AND_REPAIR_AUDIT.md | integrated repair/correction audit |
| project/V4_BOOL_SERIES_REPAIR_NOTE.md | pandas compatibility repair note |
| project/results/v4/confirmatory/result_manifest.json | artifact binding and environment |
| project/results/v4/confirmatory/preanalysis_integrity.json | exact core completeness |
| project/results/v4/confirmatory/core_subsets.json | subset counts and run/event hashes |
| project/results/v4/confirmatory/hypothesis_table_corrected.csv | authoritative numeric table |
| project/results/v4/confirmatory/final_summary_corrected.json | authoritative corrected narrative |
| project/results/v4/confirmatory/statistics_correction_record.json | correction scope and Holm recomputation |
| project/results/v4/ekf_witness_integrity_repair.json | repair provenance |

## Runtime source

| File | Why it matters |
|---|---|
| project/src/motor_model.py | plant equations and integration |
| project/src/data_utils.py | data windows, splits, normalization |
| project/src/lstm_model.py | main model |
| project/src/auxiliary_sensor_model.py | auxiliary model |
| project/src/ekf_observer.py | EKF |
| project/src/mpc.py | MPC and PI fallback |
| project/src/reliability.py | V1/V3/C4 logic |
| project/src/reliability_v4.py | V4 monitor |
| project/scripts/v4_closed_loop.py | V4 execution engine |
| project/scripts/v4_protocol_common.py | protocol configurations and writer |
| project/scripts/v4_protocol_core.py | H1–H9 core planner |
| project/scripts/v4_detector_configs.py | frozen variants |

## Historical evidence

| File | Why it matters |
|---|---|
| project/V3_DUAL_VIRTUAL_SENSOR_EXTENSION.md | frozen V3 result and load limitation |
| project/C2_ABLATION_ANALYSIS.md | C1/C2 equality |
| project/results/metrics/c4_development_stage_gate.json | C4 NO_GO |
| project/EKF_OBSERVER_PREREGISTRATION.md | EKF preregistration |
| project/results/metrics/ekf_closed_loop_summary.json | EKF comparator |
| project/TRAINING_SEED_ROBUSTNESS.md | training-seed sensitivity |
| project/FINAL_ROBUSTNESS_AND_SEVERITY_STUDY.md | boundary/severity study |
| project/FINAL_REPO_AUDIT_POSTFIX.md | V1 post-fix audit; not V4 authority |
| project/PAPER_FINAL_AUDIT.md | manuscript audit and LaTeX limitation |

## Paper

| File | Why it matters |
|---|---|
| project/paper/main.tex | current manuscript |
| project/paper/FIGURE_MANIFEST.md | six figure provenance |
| project/paper/tables/tab_h1_h9.tex | paper-facing corrected H1–H9 table |
| project/paper/references.bib | citations |

## Context package

| File | Why it matters |
|---|---|
| context/README.md | package index |
| context/START_HERE.md | fast orientation |
| context/SOURCE_OF_TRUTH.md | precedence |
| context/context_manifest.json | machine-readable package manifest |
| context/file_relationships.json | machine-readable dependency map |

Large local-only artifacts such as NPZ datasets, PT weights, SQLite checkpoints, raw CSVs, and training histories are deliberately not copied into context/.

Related: [REPOSITORY_MAP.md](REPOSITORY_MAP.md), [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md), [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md).
