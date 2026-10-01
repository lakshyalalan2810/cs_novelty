# AI handoff

You are inheriting a mature research repository. Do not begin by redesigning the controller.

This file is the operating brief for the next AI or human who opens the repository.

## Start with these five files

1. context/START_HERE.md
2. context/SOURCE_OF_TRUTH.md
3. context/ARCHITECTURE.md
4. context/V4_CONFIRMATORY.md
5. context/DO_NOT_TOUCH.md

Then use context/FILE_INDEX.md to jump to evidence and context/OPEN_QUESTIONS.md to see unresolved issues.

## Current task state

The requested orientation package is complete as an additive context/ tree. No original repository file was intentionally modified. No experiment, retraining, result regeneration, LaTeX compilation, or full verifier run was performed while building it.

## Safe answer pattern

When asked about the science:

1. name the endpoint;
2. name the comparison and scope;
3. give the current corrected value;
4. state the Holm decision;
5. state the nearest limitation.

Example:

H8 is the clean-start load false-entry comparison at 0.15 N·m. The corrected observed delta is +0.217742 on 124 retained pairs with 26 exclusions, raw p <0.0001 and Holm p <0.0009, so it is supported in that stated scope. This does not establish general fault tolerance or causal fault isolation.

## Safe editing pattern

- Read context/SOURCE_OF_TRUTH.md first.
- Inspect callers before changing a runtime function.
- Preserve current raw artifacts and historical supersession records.
- Prefer a small documentation correction over code or result regeneration.
- If a paper wording issue is found, report it in context/OPEN_QUESTIONS.md or a review; do not silently edit project files unless explicitly requested.
- If a new experiment is actually needed, write its scope, seed policy, endpoint, and stopping rule before running it.

## What has already been tried

V1 established bounded isolated abrupt-fault benefit. V2/C2 tested an auxiliary recovery gate and found it non-binding. V3/C3 used dual arbitration and exposed load confounding. C4 attempted attribution and failed its development gate. The EKF was tested as a classical comparator and retained BASELINE_ONLY. Training-seed and severity studies narrowed broad claims. V4 then preregistered witness gating, executed its exact core, repaired a deterministic EKF initialization defect, and corrected a bootstrap hierarchy implementation issue.

Do not invent V5 casually. Another detector branch would repeat already explored design space unless it addresses a specifically preregistered gap. Independent plant validation, HIL, physical-motor testing, or computational acceleration is likely more valuable than another unbounded algorithm variant.

## Before changing code

- [ ] Read ARCHITECTURE.md, DATA_FLOW.md, and CODE_CONNECTIONS.md.
- [ ] Trace every caller of the function to be changed.
- [ ] Check whether the path is V3 historical, V4 current, or C4 development-only.
- [ ] Identify whether the change would alter a frozen result or only fix infrastructure.
- [ ] Add the smallest relevant read-only regression check.

## Before running experiments

- [ ] State whether the run is development, exploratory, or confirmatory.
- [ ] Define seeds, endpoint, pairing, exclusions, and stopping rule.
- [ ] Confirm no frozen holdout seed or artifact will be reused for tuning.
- [ ] Confirm output paths and whether any checkpoint/result file will be overwritten.
- [ ] Get explicit authorization for expensive or mutating commands.

## Before changing paper claims

- [ ] Compare the sentence with the corrected CSV/JSON and endpoint definition.
- [ ] Add clean-start, condition, and sample-scope qualifiers where needed.
- [ ] Preserve C4 NO_GO, EKF BASELINE_ONLY, and training-seed sensitivity.
- [ ] Keep hardware, causal isolation, formal stability, and strong 20 Hz claims out.
- [ ] Check figure/table provenance and the resolved manifest-hash record.

## Before recomputing statistics

- [ ] Read STATISTICS.md and the correction record.
- [ ] Preserve training-seed and simulation-seed hierarchy.
- [ ] Keep all four reference rows together for H1/H2/H7.
- [ ] Reuse saved rows; do not rerun simulations to change p-values.
- [ ] Record any new method as a new analysis, not a silent replacement.

## Before committing

- [ ] Confirm only intended files changed.
- [ ] Confirm no model, dataset, checkpoint, raw CSV, or paper artifact was overwritten.
- [ ] Recheck hashes and scope counts.
- [ ] Treat the tracked overview as historical, not current authority.
- [ ] Do not commit or push unless the user asks.

## Claims to preserve

- V4 exact core = 12,200, not 268,910.
- The core includes 11,650 cells from the umbrella plus 550 supplemental H6 B-anchor cells; deferred umbrella scope is the manifest field 257,260.
- Only H8/H9 survive Holm.
- H1/H2 are directional, not significant.
- H3/H4/H6/H7 are unfavorable; H5 is null.
- C4 = NO_GO.
- EKF = BASELINE_ONLY.
- V3 broader robustness = TRAINING-SEED-SENSITIVE.
- Simulation-only; V4 assumes uncorrupted, noiseless, unquantized simulated armature current.
- No reliable 20 Hz or hardware claim.

## Paper-review status

The final cleanup corrected the E2 residual description, labeled the historical V1/V3 model metrics, added the H8/H9 124-retained/26-excluded qualifier, stated the noiseless/unquantized current assumption, and reconciled the manifest hash. LaTeX compilation remains external because no engine is installed.

## If asked to run something

Do not run the expensive protocol, training, calibration, repair, or broad verifier commands automatically. Ask whether the user wants a new scientific operation and identify the exact command and output paths first. Read [COMMANDS.md](COMMANDS.md) and [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md).

## Final handoff checklist

- [ ] Read current context package.
- [ ] Confirm working tree before editing.
- [ ] Keep original files unchanged unless explicitly authorized.
- [ ] Preserve `project/MANIFEST_HASH_RECONCILIATION.md` and `.gitattributes`.
- [ ] Label historical and current artifacts correctly.
- [ ] Report what was not run.

Related: [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md), [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md).
