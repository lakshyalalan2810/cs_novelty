# Paper context

## Manuscript identity

Current source: project/paper/main.tex.

Title:

Disturbance-Aware Witness Gating for Reliability-Aware LSTM-MPC of a Nonlinear DC Motor

The manuscript frames the contribution as a disturbance-aware reliability gate that reduces load-induced false sensor-fault entries in a simulated nonlinear motor-control setting. It does not support a universal fault-tolerant, causal fault-isolation, hardware, or real-time claim.

## Section map

| Manuscript section | Context |
|---|---|
| Introduction | Failure-driven path V1 -> V3 load confounding -> C4 NO_GO -> V4 narrow result |
| Related Work | Learned MPC, virtual sensing, observer/witness reasoning |
| Method | PMDC plant, main LSTM, auxiliary LSTM, EKF, V3 monitor, V4 witness gate, MPC |
| Experimental Protocol | dataset/training, calibration, frozen hypotheses, seeds, provenance |
| Results | post-hoc decomposition, corrected H1–H9, historical V1/V3 context |
| Discussion | narrow H8/H9 support and negative/ambiguous branches |
| Limitations | simulation-only, current-health assumption, timing, seed sensitivity |
| Conclusion | bounded load false-entry result and future external validation |

## Claims that are supported

- The architecture and execution path are implemented and audited as described.
- H8 and H9 survive Holm correction under the exact stated endpoint and scope.
- The V4 core has complete key/integrity accounting for 12,200 cells.
- The EKF repair and statistical correction have explicit provenance.
- The paper’s negative branches are real evidence: C4 NO_GO, EKF BASELINE_ONLY, training-seed sensitivity.
- Earlier V1 isolated-fault improvement is supported by its own frozen final matrix and limitations.

## Claims that must stay qualified

- “Improves detection” — not supported by H3/H4.
- “Improves recovery” — H5 is null.
- “Improves tracking” — H7 is unfavorable and unsupported.
- “Works across faults” — not established.
- “Fault isolation” — not established.
- “Real-time at 20 Hz” — not established.
- “Hardware-ready” — not established.
- “EKF is better” — not established; it is BASELINE_ONLY.

## Figure and table provenance

The six vector figures are documented in project/paper/FIGURE_MANIFEST.md. Figure 5 uses the corrected H1–H9 CSV. Figure 6 uses the load-0.15 sweep rows and the same clean-start pairing rule. No representative trajectory was selected because the frozen core does not contain a manuscript-ready common-condition trajectory artifact.

The paper-facing H1–H9 table is project/paper/tables/tab_h1_h9.tex. Check it against [STATISTICS.md](STATISTICS.md) and the corrected CSV before changing prose.

## Build status

The current audit records 87 tests and 13,746 subtests, figures and structural checks passing. No LaTeX engine was available, so compilation is unverified. Keep the source open to a human/compiler review rather than reporting “paper complete” as “PDF built.”

## Publication status and final human-read tasks

- Publication status: ready for final human read-through, not yet a published or compiler-verified PDF.
- Author metadata: intentionally anonymous authors; no affiliation, venue, DOI, or submission record is present.
- Final human checks: resolve the E2 wording, label or replace legacy model metrics, add the clean-start sample qualifier to H8/H9 prose, state the noiseless simulated-current assumption, reconcile the manifest hash, and run an external LaTeX build.

Related: [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md), [LIMITATIONS.md](LIMITATIONS.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
