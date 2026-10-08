# Open questions and remaining limitations

The 2026-10-01 cleanup resolved the manifest line-ending hash, E2 wording, H8/H9 sample qualification, current-channel wording, historical model-metric labeling, and frozen-versus-supported environment packaging. These are no longer open issues.

## Scope-count accounting

The recorded scope fields are intentionally not a simple subtraction: 11,650 cells come from the original umbrella, 550 supplemental H6 B-anchor cells bring the executed core to 12,200, and 257,260 original-umbrella cells remain deferred. A future archival audit may verify the planner classification, but no scientific result should be recomputed from subtraction.

## Ignored evidence distribution

The original V4 raw CSVs, checkpoints, model weights, dataset NPZ, histories, and final status snapshot were located in an existing local backup and restored on 2026-10-08. A deterministic, hash-checked local deposit bundle is prepared; `project/ARCHIVAL_ARTIFACTS.md` identifies it. It is ignored by Git, and no public upload or DOI exists yet. Source-only clones still require the separate bundle for raw evidence and full figure regeneration.

## Plan byte comparison

The original V4 plan bytes are not available, only its SHA-256. The re-freeze note says scientific content was independently reconstructed. A byte-level comparison cannot be performed without the original file.

## C4 and deferred protocols

No final C4 holdout exists. Robustness, timing, DET-wide, and other non-core V4 cells are deferred. No result should be invented to fill those gaps.

## Scientifically important

- Does the narrow H8/H9 effect survive an independently implemented plant or co-simulation?
- Does it persist under current-channel corruption, realistic noise, broader dropout/drift, parameter mismatch, and additional training seeds?
- Can causal sensor-versus-plant attribution be established without assuming a healthy current channel?
- Can formal stability or safety guarantees be proved for the switched MPC/reliability loop?

## Engineering important

- Can the MPC be accelerated or replaced so end-to-end timing meets a declared control deadline?
- Can HIL and physical-motor validation be built with logged latency, sensor electronics, and actuator constraints?
- Which archive should host the ignored local artifacts, and what DOI/hash inventory will identify it?

## Publication important

- Real compilation and page-by-page visual inspection were completed on 2026-10-08; repeat them after any venue-template change.
- Select a venue and apply its formatting/submission requirements.
- Create an immutable large-artifact deposit before claiming source-to-result reproducibility from a public release.

Related: [LIMITATIONS.md](LIMITATIONS.md), [AI_HANDOFF.md](AI_HANDOFF.md), [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md).
