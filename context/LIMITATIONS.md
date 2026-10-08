# Limitations and non-claims

## HISTORICAL V3 LIMITATIONS

V3 load disturbances could trigger false sensor-fault entries and worsen closed-loop RMSE. Drift showed weak benefit, and the single-seed combined-fault result was narrowed by the training-seed study. The current V4 result does not erase those limitations.

## V4 CONFIRMATORY LIMITATIONS

- All controller, reliability, observer, and safety evidence is simulated.
- There is no HIL, hardware, sensor electronics, communication, actuator, or deployment evidence.
- Reliable end-to-end 20 Hz operation is NOT ESTABLISHED. Solver timings are machine/load dependent and exclude full scheduling, I/O, plant integration, and hardware latency.
- The V4 confirmatory evidence is 12,200 cells, not the full 268,910-cell umbrella.
- The deferred 257,260 cells cannot be treated as negative, positive, or completed evidence.

## Sensor and model assumptions

- The frozen V4 dataset gives the auxiliary LSTM and EKF an uncorrupted, noiseless simulated current measurement with no quantization; realistic current-sensor behavior is not established.
- Current corruption degraded both C3 and EKF in the boundary study.
- A one-speed-sensor residual cannot always distinguish sensor corruption from plant/model mismatch.
- Witness disagreement is not causal fault isolation.
- The LSTM is trained open loop and accumulates recursive prediction error.
- Load disturbance is a known V3 confounder; V4 only supports the tested narrow false-entry endpoint.

## Statistical and experimental limits

- H1/H2 are favorable in sign but not Holm-supported.
- H3/H4/H6/H7 are unfavorable in observed direction and not supported.
- H5 is an exact null under retained pairs.
- H8/H9 are supported only for 0.15 N·m load, clean-start pairing, and the frozen controller/witness setup.
- Training-seed sensitivity means single-seed broad claims are unsafe.
- Severity study found no tested severity classified SUPPORTED.
- The current project requirements differ from the recorded V4 execution environment; exact clean-environment reproduction is not established.
- A corrected bootstrap can report percentile CIs that differ from centered empirical p-value decisions; this is intentional and documented.

## GENERAL SYSTEM LIMITATIONS

- No formal stability proof is provided.
- No formal fault-isolation guarantee is provided.
- The top-level learned-weight study has only three training seeds for the common C3 comparison.
- The V4 core does not cover every preregistered severity, dropout, drift, robustness, DET, or timing cell.

## Paper and tooling limits

- The Oct-1 audit had no LaTeX engine. The 2026-10-08 finalization built a real Tectonic/XeTeX + BibTeX PDF and visually inspected every page; see project/PAPER_FINAL_AUDIT.md for current diagnostics.
- The historical 87-test / 13,746-subtest record remains historical. Current focused checks do not imply rerunning the scientific protocols.
- The verified archive is prepared locally, not publicly deposited. Source-only and artifact-assisted reproducibility differ; venue formatting and human metadata remain unresolved.
- The result-manifest hash mismatch was a Windows CRLF checkout transformation; `.gitattributes` now preserves the archival LF bytes. This does not establish bitwise reproduction of the full execution environment.

## Operational limits

- Do not launch the umbrella protocols, retraining, repair scripts, or expensive verifiers while merely orienting.
- Do not tune thresholds, select seeds, or add a new architecture to improve the current story without a new preregistration decision.
- Do not use the historical `project/PROJECT_OVERVIEW_AND_PUBLISHABILITY.md` as current authority.

Related: [PROJECT_SNAPSHOT.md](PROJECT_SNAPSHOT.md), [STATISTICS.md](STATISTICS.md), [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md), [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).
