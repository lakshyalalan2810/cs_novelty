# Start here

## What this repository is

This is a simulation research repository for disturbance-aware witness gating around an LSTM-MPC controller for a nonlinear permanent-magnet DC motor. It exists to test whether input-diverse witness evidence can reduce false reliability latches caused by load/model mismatch without overstating the capability of a learned controller.

## Final paper title

Disturbance-Aware Witness Gating for Reliability-Aware LSTM-MPC of a Nonlinear DC Motor

## Current final research question

Can an input-diverse witness gate prevent load/model mismatch from being misclassified as a speed-sensor fault in a reliability-aware LSTM-MPC loop, while preserving the bounded behavior already observed for isolated abrupt sensor faults?

The outer repository is cs_novelty. The research implementation and artifacts are under project/.

## Main architecture

Reference -> H20/Nc2 LSTM-MPC -> bounded voltage -> nonlinear PMDC plant -> noisy/corrupted speed and current -> main speed-history LSTM plus auxiliary current-informed LSTM or EKF witness -> V3 arbitration or V4 reliability gate -> selected feedback -> MPC. In V4, the auxiliary LSTM and EKF are witnesses; the main LSTM is the substitution feedback.

## Current final scientific conclusion

The current confirmatory result is narrow and should be stated exactly:

At a preregistered 0.15 N·m load disturbance, under clean-start pairing, V4 auxiliary and EKF witness gating both reduced false sensor-fault entries relative to frozen V3 C3. H8 and H9 are the only Holm-supported hypotheses.

Everything else is more qualified:

- H1/H2: directionally favorable false-latch effects, but unsupported after Holm.
- H3/H4: unfavorable detection deltas, not supported.
- H5: exact null recovery delta.
- H6: unfavorable fault-window RMSE delta, not supported.
- H7: unfavorable tracking-penalty delta, not supported.
- C4: NO_GO.
- EKF: BASELINE_ONLY.
- V3 broader combined-fault interpretation: TRAINING-SEED-SENSITIVE.
- V4 scope: simulation-only, one narrow confirmatory core.

## Main positive result

H8 and H9 support reduced load-disturbance-induced false sensor-fault entries at 0.15 N·m on the retained clean-start endpoint. They are the only multiplicity-supported results.

## Major negative/null results

H1/H2 are favorable in direction but unsupported; H3/H4/H6/H7 are unfavorable and unsupported; H5 is exactly null. C4 is NO_GO, EKF is BASELINE_ONLY, and broader learned-weight behavior is TRAINING-SEED-SENSITIVE.

## Non-negotiable numbers

| Quantity | Current evidence |
|---|---:|
| Original V4 umbrella | 268,910 cells |
| Executed exact H1–H9 core | 12,200 cells |
| Fault-free / sweep / recovery | 9,600 / 2,000 / 600 |
| Deferred umbrella cells | 257,260 |
| Supplemental H6 B-anchor cells | 550 |
| EKF witness cells repaired | 2,700 |
| Valid cells intentionally untouched | 9,500 |
| H1–H9 Holm survivors | H8 and H9 only |
| Confirmatory bootstrap replicates | 20,000 per hypothesis |

## Fast orientation path

~~~text
current repository
      |
      v
SOURCE_OF_TRUTH --> RESULTS_AUTHORITY --> V4_CONFIRMATORY --> STATISTICS
      |                    |                    |
      v                    v                    v
architecture         provenance            limitations/paper
~~~

1. Read [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md) before quoting numbers.
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) and [DATA_FLOW.md](DATA_FLOW.md) before editing code.
3. Read [V4_CONFIRMATORY.md](V4_CONFIRMATORY.md) and [STATISTICS.md](STATISTICS.md) before interpreting H1–H9.
4. Read [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md) before running anything.
5. Use [AI_HANDOFF.md](AI_HANDOFF.md) to continue work safely.

Accounting note: the manifest’s deferred count is not the simple arithmetic complement of the total core. The executed core contains 11,650 cells from the original umbrella plus 550 supplemental H6 B-anchor cells; 11,650 + 257,260 = 268,910, and 11,650 + 550 = 12,200.

## Thirty-minute deep orientation path

Read [PROJECT_SNAPSHOT.md](PROJECT_SNAPSHOT.md), [REPOSITORY_MAP.md](REPOSITORY_MAP.md), [ARCHITECTURE.md](ARCHITECTURE.md), [DATA_FLOW.md](DATA_FLOW.md), [CODE_CONNECTIONS.md](CODE_CONNECTIONS.md), [RESEARCH_TIMELINE.md](RESEARCH_TIMELINE.md), [EXPERIMENTS.md](EXPERIMENTS.md), [V4_CONFIRMATORY.md](V4_CONFIRMATORY.md), [STATISTICS.md](STATISTICS.md), [PROVENANCE.md](PROVENANCE.md), [PAPER_CONTEXT.md](PAPER_CONTEXT.md), and [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). Then inspect only the specific original files named by [FILE_INDEX.md](FILE_INDEX.md).

## What is not safe to infer

Do not infer:

- that 268,910 cells were executed;
- that the V4 witness is a general fault-isolation system;
- that H1/H2 are significant;
- that detection, recovery, tracking, or current-fault robustness improved generally;
- that the controller is hardware validated or reliably real-time at 20 Hz;
- that EKF replaces the auxiliary witness;
- that a new run is needed merely to make the paper story stronger.

## Current repository status

The only pre-existing change observed before this context package was the untracked file project/PROJECT_OVERVIEW_AND_PUBLISHABILITY.md. It was not modified. The context package itself is the only intended new tree.

## Authoritative artifacts

Use project/results/v4/confirmatory/hypothesis_table_corrected.csv, final_summary_corrected.json, statistics_correction_record.json, preanalysis_integrity.json, core_subsets.json, ekf_witness_integrity_repair.json, and result_manifest.json. The precedence and manifest hash caveat are in [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md).

## What not to modify

Do not modify the original project tree, raw V4 rows/events, checkpoints, model/data binaries, frozen configs, preregistration, corrected artifacts, or paper files during orientation. Read [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md).

Related: [PROJECT_SNAPSHOT.md](PROJECT_SNAPSHOT.md), [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md), [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md).

If you are another AI taking over this repository, read these files in this exact order: START_HERE.md, SOURCE_OF_TRUTH.md, PROJECT_SNAPSHOT.md, ARCHITECTURE.md, DATA_FLOW.md, V4_CONFIRMATORY.md, STATISTICS.md, RESULTS_AUTHORITY.md, PROVENANCE.md, LIMITATIONS.md, AI_HANDOFF.md, DO_NOT_TOUCH.md.
