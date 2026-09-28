# Commands

These commands are for orientation or read-only verification. Run them from the outer repository root unless a command explicitly changes directory.

## Safe inventory

~~~powershell
rg --files context
rg --files project/src project/scripts project/results/v4/confirmatory project/paper
git status --short
git log -1 --oneline
~~~

## Safe artifact checks

~~~powershell
Get-FileHash -Algorithm SHA256 project/results/v4/confirmatory/result_manifest.json
Get-FileHash -Algorithm SHA256 project/results/v4/confirmatory/hypothesis_table_corrected.csv
Get-FileHash -Algorithm SHA256 project/results/v4/confirmatory/final_summary_corrected.json
Get-Content project/results/v4/confirmatory/preanalysis_integrity.json
Get-Content project/results/v4/confirmatory/statistics_correction_record.json
~~~

## Safe structural checks

~~~powershell
python -m json.tool project/results/v4/confirmatory/result_manifest.json
python -m json.tool project/results/v4/confirmatory/final_summary_corrected.json
python -m json.tool project/results/v4/confirmatory/preanalysis_integrity.json
~~~

If the environment lacks Python or the dependencies, reading the files is still useful; do not install packages merely to orient.

## SAFE VERIFIERS

Read the verifier source and recorded artifact paths before running. Narrow read-only checks include:

~~~text
python scripts/v4_confirm_verify.py --help
python scripts/verify_c4_results.py --help
python scripts/verify_v3_results.py --help
~~~

These commands are documentation examples only in this task. A recorded verifier PASS is not a substitute for reading the statistical correction record.

## SAFE TESTS

The repository’s test modules are under project/tests/. Prefer a narrow test that only parses saved artifacts. Do not run a suite that regenerates data or uses missing historical environments without explicit authorization.

## PAPER/FIGURE COMMANDS

The generation scripts are:

~~~text
python scripts/v4_paper_tables.py
python scripts/v4_paper_figures.py
~~~

They are useful provenance entry points, but do not rerun them merely to orient. The figure generator is intended to be deterministic and does not run new scientific simulations; it can still overwrite paper outputs.

## TRAINING COMMANDS — EXPENSIVE

~~~text
python scripts/v4_generate_dataset.py
python scripts/v4_train_models.py
python scripts/v4_calibrate.py
python scripts/v4_calibrate_baselines.py
~~~

These create or alter scientific inputs and require an explicit new reconstruction/experiment decision.

## CONFIRMATORY PROTOCOL COMMANDS — DO NOT RUN CASUALLY

~~~text
python scripts/v4_protocol_core.py --workers 8
python scripts/v4_protocol_faultfree.py ...
python scripts/v4_protocol_sweep.py ...
python scripts/v4_protocol_recovery.py ...
~~~

These can launch expensive simulations, checkpoints, or output writes. The exact executed core is already recorded; do not rerun the 12,200 cells for orientation.

## DEFERRED PROTOCOL COMMANDS — NOT AUTHORIZED

Robustness OFAT, timing, DET-wide, expanded dropout/drift, combined-fault expansions, and other umbrella jobs remain deferred. No command should be launched to fill those gaps without a new protocol and user authorization.

## GIT COMMANDS

Safe inspection:

~~~text
git status --short
git log -1 --oneline
git diff --name-only
git ls-files
~~~

Do not commit, push, reset, checkout, delete, or move existing project files for this task.

## Safe context self-check

After this package is created, verify that all required paths exist:

~~~powershell
Get-ChildItem context -Recurse -File | Sort-Object FullName
Select-String -Path context/*.md -Pattern 'H8|H9|12,200|257,260|NO_GO|BASELINE_ONLY'
~~~

## Expensive or mutating commands

Do not run during orientation:

~~~text
python scripts/v4_protocol_core.py --workers ...
python scripts/v4_train_models.py ...
python scripts/v4_generate_dataset.py ...
python scripts/v4_calibrate.py ...
python scripts/v4_repair_ekf_witness_cells.py ...
python scripts/v4_protocol_faultfree.py ...
python scripts/v4_protocol_sweep.py ...
python scripts/v4_protocol_recovery.py ...
~~~

The dry-run form of the core planner is intended for plan inspection only:

~~~text
python scripts/v4_protocol_core.py --dry-run
~~~

Even a dry-run should be reviewed for side effects in the local version before use. Never use an output-producing command just to answer a question already answered by saved artifacts.

## Tests

The paper audit reports historical test results. Running a test suite is not required to orient and may require a different environment. If tests are explicitly requested, first read [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md) and [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md), then prefer the narrowest read-only test that does not regenerate artifacts.

Related: [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md), [AI_HANDOFF.md](AI_HANDOFF.md).
