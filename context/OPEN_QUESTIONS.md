# Open questions and unresolved consistency items

These are not reasons to reinterpret the current scientific result. They are the remaining questions a careful maintainer should resolve before calling the repository fully distributable or submission-ready.

## Manifest hash mismatch

Current project/tests/test_v4_confirmatory_freeze.py and project/paper/main.tex cite SHA-256 0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e for the superseding V4 manifest. The current filesystem bytes at project/results/v4/confirmatory/result_manifest.json hash to 58c2ece8410574de388b0933e17a0b8295e65834f74858edf7d21956e9ce3ff8. Determine whether the mismatch is a stale constant, a line-ending/artifact revision, or an unrecorded manifest replacement. Do not “fix” it by changing the test or manifest without inspecting the intended byte-level artifact.

## Scope-count accounting

The recorded scope fields are intentionally not a simple subtraction: 11,650 cells come from the original umbrella, 550 supplemental H6 B-anchor cells bring the executed core to 12,200, and 257,260 original-umbrella cells remain deferred. This is now stated explicitly in the context package. A future archival audit should verify the planner’s classification, but no scientific result should be recomputed from subtraction.

## Paper metric provenance

The paper audit found that several model metrics in project/paper/main.tex match legacy V1/V2 artifacts, while current V4 seed-2026 model metrics differ. Decide whether to label those numbers historical or replace them with V4 metrics. This is a paper edit, not a reason to rerun training.

## E2 wording

The manuscript describes E2 as CUSUM on EKF innovation, while project/src/baselines_v4.py applies CUSUM to measured speed minus EKF speed and only records current innovation. Resolve the wording against the implementation before submission.

## H8/H9 wording

The headline should state the clean-start condition, 124 retained pairs, and 26 exclusions. Confirm the final abstract, contribution bullets, and conclusion all retain those qualifiers.

## Current-channel assumption

The V4 dataset records zero current noise and no quantization. Decide how explicitly the manuscript should say “noiseless, uncorrupted simulated current” rather than only “healthy current.”

## Reproduction environment

The result manifest records Python 3.11.15 / NumPy 2.2.6 / pandas 2.3.3 / SciPy 1.13.1 / PyTorch 2.6.0+cu124, but requirements.txt pins newer/different versions. Establish a clean archival environment or label the current requirements as reconstruction-only.

## Ignored evidence distribution

The V4 raw CSVs, checkpoints, model weights, dataset NPZ, and histories are locally present but ignored by project/.gitignore. Decide which evidence belongs in a distributable archive and how its hashes will be transported. Do not copy large artifacts into context/.

## Plan byte comparison

The original V4 plan bytes are not available, only its SHA-256. The re-freeze note says scientific content was independently reconstructed. A byte-level comparison cannot be performed without the original file.

## Fresh verification

The repository records verifier passes, but this orientation pass did not execute them. A future audit may run the narrowest read-only checks in the recorded environment after resolving the environment and hash questions.

## C4 and deferred protocols

No final C4 holdout exists. Robustness, timing, DET-wide, and other non-core V4 cells are deferred. No result should be invented to fill those gaps.

## SCIENTIFICALLY IMPORTANT

- Does the narrow H8/H9 effect survive an independently implemented plant or co-simulation?
- Does it persist under current-channel corruption, realistic noise, broader dropout/drift, parameter mismatch, and additional training seeds?
- Can causal sensor-versus-plant attribution be established without assuming a healthy current channel?
- Can formal stability or safety guarantees be proved for the switched MPC/reliability loop?

## ENGINEERING IMPORTANT

- Can the MPC be accelerated or replaced so end-to-end timing, not only solver timing, meets a declared control deadline?
- Can HIL and physical-motor validation be built with logged latency, sensor electronics, and actuator constraints?
- Which ignored local artifacts should be archived with their hashes for an independently reproducible release?

## PUBLICATION IMPORTANT

- Resolve the 0591... versus 58c2... manifest hash mismatch.
- Label or replace legacy model metrics in the manuscript.
- Correct the E2 innovation wording.
- Add the 124 retained clean-start-pair and 26-exclusion qualifier to the H8/H9 headline.
- State the noiseless, uncorrupted simulated-current assumption explicitly.
- Run an external LaTeX build or otherwise document compiler verification.

## OPTIONAL / LOW PRIORITY

- Improve navigation or add a visual FAQ.
- Add a distributable archive manifest after deciding which large ignored artifacts belong in a release.

Related: [LIMITATIONS.md](LIMITATIONS.md), [AI_HANDOFF.md](AI_HANDOFF.md), [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md).
