# Code connections

This is the shortest call/ownership map for the current implementation. Line numbers are intentionally omitted because this package is a snapshot of relationships, not a promise that future edits preserve line locations.

## Primary runtime graph

~~~text
v4_protocol_faultfree.py
v4_protocol_sweep.py
v4_protocol_recovery.py
              |
              v
      v4_protocol_core.py
              |
              v
      v4_protocol_common.py
              |
              v
       v4_closed_loop.py
       /      |       \
      v       v        v
 motor    LSTM/MPC  reliability
 model    + PI      + witness
                   /       \
                  v         v
             auxiliary      EKF
              LSTM        observer
              |
              v
        runs.csv/events.csv
              |
       confirm_analysis
              |
       statistics_audit
              |
       corrected result artifacts
~~~

## Source responsibilities

| Source | Defines | Used by |
|---|---|---|
| src/motor_model.py | DCMotorParams, dynamics, simulate_motor | dataset generation, V4 engine, legacy evaluators |
| src/data_utils.py | trajectory generators, splits, normalizers, windows | training and dataset scripts |
| src/lstm_model.py | main residual LSTM | training, V1/V3/V4 engine |
| src/auxiliary_sensor_model.py | current/voltage-only LSTM | training, V2/V3/V4 engine |
| src/ekf_observer.py | augmented EKF | EKF study and V4 witness |
| src/mpc.py | LSTMMPC, MPCConfig, PI fallback | legacy and V4 controller loops |
| src/reliability.py | V1 monitor, V3 arbitration, C4 helpers | legacy V1/V2/V3 and C4 |
| src/reliability_v4.py | ReliabilityMonitorV4 | V4 engine and V4 tests |
| scripts/v4_detector_configs.py | named V4 variants | V4 protocol planner/engine |
| scripts/v4_protocol_common.py | seeds, cell configs, keying, checkpoint writer | all V4 protocols |
| scripts/v4_closed_loop.py | ControllerStack, run_v4_cell, summaries | all V4 protocol wrappers |

## Read/write/scientific-status matrix

| File | Reads | Writes | Scientific role/status |
|---|---|---|---|
| project/src/motor_model.py | parameters, state, input voltage, load | in-memory state/trajectory | current plant implementation |
| project/src/data_utils.py | trajectory arrays and split settings | dataset windows, normalization metadata | current preprocessing |
| project/src/lstm_model.py | normalized [voltage, speed] windows, weights | predictions and in-memory state | current main predictor |
| project/src/auxiliary_sensor_model.py | physical [voltage, current] windows, weights | speed witness predictions | current auxiliary witness |
| project/src/ekf_observer.py | voltage, current, covariance/config | speed/load estimate, diagnostics | current EKF witness and comparator |
| project/src/mpc.py | feedback speed, reference, model, MPC config | voltage command, timing, fallback state | current controller; generic defaults are not V4 authority |
| project/src/reliability.py | residuals, monitor thresholds, virtual estimates | latch/arbitration state and C4 diagnostics | V1–V3/C4 historical/current branches |
| project/src/reliability_v4.py | main/witness residuals and V4 flags | latch, recovery, substitution decision | current V4 detector |
| project/scripts/v4_closed_loop.py | models, calibration, configs, fault/scenario state | in-memory run row/events/traces | current V4 execution engine |
| project/scripts/v4_protocol_common.py | RunConfig, seeds, source/config hashes, engine output | SQLite checkpoints, runs.csv, events.csv | current protocol I/O |
| project/scripts/v4_protocol_core.py | prereg plan, core cell set, protocol common | execution status and planned core output | current exact-core planner; actual run is expensive |
| project/scripts/v4_confirm_analysis.py | raw V4 rows/events and plan | historical H1–H9 tables/summary | historical analysis with clustering defect |
| project/scripts/v4_statistics_audit.py | frozen rows/events and historical summaries | corrected H1/H2/H7 values and correction record | current statistical correction |
| project/scripts/v4_confirm_verify.py | manifest, plan, raw rows, calibration/model bindings | verification report/status | historical verifier; pass does not override correction |
| project/scripts/v4_paper_tables.py | corrected JSON/CSV and paper inputs | LaTeX result tables | current paper generation |
| project/scripts/v4_paper_figures.py | corrected summaries and post-hoc CSVs | deterministic vector PDFs | current figure generation; no scientific simulation |
| project/scripts/verify_results.py | V1/V2 final CSV/JSON/configs | pass/fail diagnostics | historical V1/final-matrix verifier |
| project/scripts/verify_c4_results.py | C4 development outputs/configs | pass/fail diagnostics | current C4 gate verifier; decision remains NO_GO |

## Analysis and authority graph

| Script/artifact | Relationship |
|---|---|
| scripts/v4_confirm_analysis.py | historical analysis source; its H1/H2/H7 bootstrap grouping defect is preserved |
| scripts/v4_confirm_verify.py | historical verifier; passed but reproduced the same clustering defect |
| scripts/v4_statistics_audit.py | corrected hierarchical bootstrap, no simulation rerun |
| results/v4/confirmatory/statistics_correction_record.json | correction classification and impacted hypotheses |
| results/v4/confirmatory/hypothesis_table_corrected.csv | corrected H1–H9 table |
| results/v4/confirmatory/final_summary_corrected.json | corrected narrative/status summary |
| results/v4/confirmatory/result_manifest.json | artifact bindings and provenance |
| paper/tables/tab_h1_h9.tex | paper-facing table; must agree with corrected CSV |
| paper/FIGURE_MANIFEST.md | figure-to-artifact provenance |
| PAPER_FINAL_AUDIT.md | current manuscript audit |

## Historical branches

- project/scripts/evaluate_v3_closed_loop.py is the legacy V3 evaluator.
- project/src/reliability.py contains C4 classes, but C4 is not called by v4_closed_loop.py.
- project/scripts/evaluate_ekf_closed_loop.py is a separate comparator path, not the V4 witness path.
- project/scripts/verify_results.py covers the older V1/final repository matrix.

## Important implementation details

- V4 engine initializes the EKF when its initialized flag is false; the pre-repair path tested a loop index instead and caused the 2,700-cell repair.
- V4 uses fixed H20 and (5,15) move blocks at the caller; generic MPC defaults are not the experiment configuration.
- V4 monitor calls begin after the 20-sample history; warmup is therefore measured in monitor calls rather than absolute time from zero.
- V4 witness variants use witnesses to gate main-model reliability, not to choose a direct feedback estimate.

Related: [ARCHITECTURE.md](ARCHITECTURE.md), [REPOSITORY_MAP.md](REPOSITORY_MAP.md), [file_relationships.json](file_relationships.json).
