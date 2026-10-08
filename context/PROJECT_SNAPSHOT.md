# Project snapshot

Scientific snapshot date: 2026-10-01; current publication-status update: 2026-10-08. The finalization baseline HEAD/origin was 4b7c51d; the earlier observed HEAD below remains historical. Observation date: 2026-10-01. Source of evidence: current checkout, saved artifacts, paper/audit documents, and read-only checks. No scientific protocol, training, calibration, or result regeneration was performed; one existing unit test generated a six-trajectory fixture in a temporary directory and retained nothing.

## Identity

| Field | Value |
|---|---|
| Repository | cs_novelty |
| Research root | project/ |
| Branch | main |
| Baseline HEAD observed | 2a6933d8ddaa7d45d9175f7ce0558fbd52a454e4 |
| Language/tooling | Python scripts/notebooks, PyTorch, NumPy/SciPy/pandas, Matplotlib, LaTeX manuscript |
| Recorded libraries | Python 3.11.15, NumPy 2.2.6, pandas 2.3.3, SciPy 1.13.1, PyTorch 2.6.0+cu124 |
| Paper title | Disturbance-Aware Witness Gating for Reliability-Aware LSTM-MPC of a Nonlinear DC Motor |
| Primary manuscript | project/paper/main.tex |
| Current paper status | 2026-10-08 compiled/visually reviewed; venue format, public archive and human metadata pending |
| Recorded final test status | Paper audit reports 87 tests / 13,746 subtests passed; not rerun in this context build |
| Hardware status | No HIL or hardware evidence |
| Real-time status | Reliable 20 Hz end-to-end execution NOT ESTABLISHED |
| Major assumptions | Simulated plant, uncorrupted/noiseless/unquantized armature-current channel in V4, frozen calibration/model bindings |

## Current status board

| Branch or artifact | Status | Meaning |
|---|---|---|
| V1 final holdout | FROZEN SUCCESS WITH LIMITATIONS | Isolated abrupt sensor faults improved; broader claims narrowed |
| V2/C2 | CLOSED NEGATIVE ABLATION | Auxiliary recovery gate was uniquely binding zero times |
| V3/C3 | FROZEN STRONGEST ARCHITECTURE | Strong isolated-fault behavior; load confounding remains |
| C4 | NO_GO | Development branch failed its gate; no final holdout |
| EKF | BASELINE_ONLY | Valid comparator, not a successor architecture |
| Training-seed study | TRAINING-SEED-SENSITIVE | Combined-fault behavior changes by learned-weight seed |
| Severity study | COMPLETE; NO SUPPORTED OPERATING POINT | Tested conditions are LIMITED or UNSUPPORTED |
| V4 | CORRECTED_AND_VERIFIED | Exact 12,200-cell core audited; only H8/H9 supported |
| Manuscript (historical Oct 1) | READY FOR HUMAN READ-THROUGH | Historical structural/evidence audit passed; compiler was absent then |

## V4 scope

The original preregistered umbrella was 268,910 cells. The exact dependency core needed for H1–H9 was recorded as 12,200 cells:

- faultfree: 9,600 cells;
- sweep: 2,000 cells;
- recovery: 600 cells.

The manifest records 257,260 deferred umbrella cells. This is not the simple arithmetic complement of 12,200: the core contains 11,650 cells drawn from the original umbrella plus 550 supplemental B bias-8 anchor cells added for H6. Thus 11,650 + 257,260 = 268,910 and 11,650 + 550 = 12,200. Treat the manifest fields and the H6 anchor provenance as authoritative; do not recompute deferred scope as 268,910 - 12,200.

Integrity facts:

- 12,200 unique planned keys;
- zero missing, duplicate, unexpected, optimizer, main-prediction, auxiliary-prediction, observer, nonfinite, or voltage failures in the pre-analysis integrity report;
- 3,053 slew counters were retained as safety outcomes and were not treated as execution corruption;
- 2,700 V4_full_ekf cells were rerun after an outcome-independent initialization repair;
- 9,500 valid cells were not rerun;
- no deferred cell was rerun.

## V4 corrected decision table

| Hypothesis | Corrected observed delta | Raw p | Holm p | Decision |
|---|---:|---:|---:|---|
| H1 | +0.209167 | 0.0627 | 0.2508 | Directionally favorable; unsupported |
| H2 | +0.209167 | 0.0659 | 0.2508 | Directionally favorable; unsupported |
| H3 | -0.100775 | 0.0125 | 0.0763 | Unfavorable; unsupported |
| H4 | -0.100775 | 0.0109 | 0.0763 | Unfavorable; unsupported |
| H5 | 0.000000 | 1.0000 | 1.0000 | Exact null |
| H6 | -2.557027 | 0.0110 | 0.0763 | Unfavorable; unsupported |
| H7 | -0.063573 | 0.0744 | 0.2508 | Unfavorable; unsupported |
| H8 | +0.217742 | <0.0001 | <0.0009 | Supported |
| H9 | +0.217742 | 0.0001 | 0.0008 | Supported |

The deltas are endpoint-specific; see [STATISTICS.md](STATISTICS.md) for definitions and sample units.

## Environment evidence

The completed V4 manifest records Python 3.11.15, NumPy 2.2.6, pandas 2.3.3, SciPy 1.13.1, and PyTorch 2.6.0+cu124. `project/requirements-frozen-v4.txt` preserves those recorded versions, while `project/requirements.txt` remains the supported reconstruction/development stack. The manifest is not a complete OS, driver, BLAS, solver, or transitive-package lock, so bitwise reproduction remains NOT ESTABLISHED.

## Working tree note

The tracked `project/PROJECT_OVERVIEW_AND_PUBLISHABILITY.md` is preserved with a historical-document banner. Frozen scientific artifacts were not modified.

## CURRENT AUTHORITATIVE CONCLUSION

Only H8 and H9 are Holm-supported: the two V4 witness-gated variants reduce load-induced false sensor-fault entries at the tested 0.15 N·m load step under the retained clean-start endpoint. This is a bounded simulation result from the exact recorded core, not a universal controller or fault-tolerance result.

## CURRENT UNSUPPORTED CLAIMS

The repository does not establish general fault tolerance, improved detection or recovery, improved tracking, causal fault isolation, current-sensor-fault tolerance, hardware/HIL validity, formal stability, or reliable end-to-end 20 Hz execution.

## CURRENT NEXT LOGICAL RESEARCH STEP

If new work is authorized, independent external validation is more valuable than inventing another detector branch: consider an independently implemented plant (for example Simulink or co-simulation), HIL, or physical-motor validation with an explicit new protocol. This is a recommendation, not a current result.

Related: [V4_CONFIRMATORY.md](V4_CONFIRMATORY.md), [PROVENANCE.md](PROVENANCE.md), [LIMITATIONS.md](LIMITATIONS.md).
