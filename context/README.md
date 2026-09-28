# Research context package

This folder is the orientation layer for the repository at the parent level. It is intentionally additive: the original research tree was not edited, moved, deleted, or regenerated while this package was built.

Read this package before reading the implementation. It records the current repository evidence as observed on 2026-09-28, including the bounded scientific conclusion, the V4 provenance chain, known historical/superseded artifacts, and the one manifest-hash inconsistency that remains in the current checkout.

## Five-minute reading order

1. [START_HERE.md](START_HERE.md) — the shortest safe orientation.
2. [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md) — which artifacts win when documents disagree.
3. [PROJECT_SNAPSHOT.md](PROJECT_SNAPSHOT.md) — current status, counts, and immutable facts.
4. [ARCHITECTURE.md](ARCHITECTURE.md) and [DATA_FLOW.md](DATA_FLOW.md) — how the code actually runs.
5. [V4_CONFIRMATORY.md](V4_CONFIRMATORY.md) and [STATISTICS.md](STATISTICS.md) — the confirmatory scope and corrected H1–H9 results.
6. [LIMITATIONS.md](LIMITATIONS.md) and [AI_HANDOFF.md](AI_HANDOFF.md) — what may and may not be claimed or changed.

## Current one-paragraph conclusion

The repository studies reliability-aware LSTM-MPC for a simulated nonlinear PMDC motor. V1 established a bounded isolated abrupt-sensor-fault result; V2/C2 was a negative recovery ablation; V3 became the strongest frozen architecture but exposed load-induced false latches; C4 was a development-only NO_GO branch; the EKF is BASELINE_ONLY; training-seed and severity studies narrow the claims. V4 executed the exact confirmatory dependency core of 12,200 cells from a 268,910-cell umbrella: 9,600 fault-free, 2,000 sweep, and 600 recovery cells, with 257,260 cells deferred. After outcome-independent EKF witness repair and a post-run hierarchical-bootstrap correction, only H8 and H9 survive Holm correction. The supported claim is narrow: at the tested 0.15 N·m load step and clean-start pairing, both V4 witness variants reduced false sensor-fault entries relative to frozen V3 C3. General fault tolerance, improved detection/recovery/tracking, hardware validity, current-channel robustness, causal fault isolation, and reliable 20 Hz execution are not established.

## Navigation

- [PROJECT_SNAPSHOT.md](PROJECT_SNAPSHOT.md)
- [REPOSITORY_MAP.md](REPOSITORY_MAP.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DATA_FLOW.md](DATA_FLOW.md)
- [CODE_CONNECTIONS.md](CODE_CONNECTIONS.md)
- [RESEARCH_TIMELINE.md](RESEARCH_TIMELINE.md)
- [EXPERIMENTS.md](EXPERIMENTS.md)
- [V4_CONFIRMATORY.md](V4_CONFIRMATORY.md)
- [STATISTICS.md](STATISTICS.md)
- [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md)
- [PROVENANCE.md](PROVENANCE.md)
- [LIMITATIONS.md](LIMITATIONS.md)
- [PAPER_CONTEXT.md](PAPER_CONTEXT.md)
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- [COMMANDS.md](COMMANDS.md)
- [AI_HANDOFF.md](AI_HANDOFF.md)
- [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md)
- [GLOSSARY.md](GLOSSARY.md)
- [FILE_INDEX.md](FILE_INDEX.md)
- [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md)
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)
- [context_manifest.json](context_manifest.json)
- [file_relationships.json](file_relationships.json)

Small convenience snapshots are indexed in [snapshots/README.md](snapshots/README.md) and include [snapshots/README_CURRENT.md](snapshots/README_CURRENT.md), [snapshots/REQUIREMENTS.txt](snapshots/REQUIREMENTS.txt), [snapshots/IMPORTANT_CONFIGS/](snapshots/IMPORTANT_CONFIGS/), and [snapshots/IMPORTANT_SMALL_ARTIFACTS/](snapshots/IMPORTANT_SMALL_ARTIFACTS/). They are not replacements for the original evidence.

## Evidence labels used here

- FACT — directly observed in the current source, artifact, manifest, or saved audit.
- INFERENCE — a bounded interpretation derived from the facts.
- HISTORICAL — retained for provenance but superseded for current claims.
- NOT ESTABLISHED — the repository does not support the stronger statement.
- DO NOT RUN — the command can trigger expensive simulations, retraining, or artifact mutation.

Related: [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md), [AI_HANDOFF.md](AI_HANDOFF.md), [PROVENANCE.md](PROVENANCE.md).
