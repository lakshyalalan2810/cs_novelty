# C4 final-holdout verifier status (pre-existing, out of scope)

## Observation

`python scripts/verify_c4_results.py` exits 1 after printing
`PASS: resolved C4 calibration: results/configs/c4_attribution_calibration.json`,
with no further message. `scripts/run_all_verifiers.py` therefore reports
7/8 PASS with `verify_c4_results.py` as the single failure.

## Cause (verified, not inferred)

The next artifact the verifier requires,
`results/configs/c4_frozen_config.json`, does not exist in the delivered
workspace. `main()` converts the resulting `VerificationError` into a
bare `SystemExit(1)`, which is why the failure is silent.

This is the expected end state of the recorded C4 history, not data loss:

- `results/configs/c4_predevelopment_abort.json` documents an aborted
  first development attempt (preflight defects, evidence rejected).
- `c4_post_ekf_verify.log` (269 lines, shipped with the repo) records a
  completed C4 *development* verification ending in
  `C4 development stage gate: NO_GO` / `DEVELOPMENT VERIFIER RESULT: PASS`.
- After a NO_GO development gate, no final holdout was run, so no final
  artifacts (`c4_frozen_config.json`, `c4_final_holdout_runs.csv`,
  `C4_THREE_WAY_ATTRIBUTION_EXTENSION.md`) were ever created.

## Proof of pre-existence

The read-only Phase 0 snapshot (`/tmp/cs_novelty_phase0_snapshot`,
recursive copy before any edit) fails identically: same final line,
same silent exit 1. The `results/configs/` listings of snapshot and
working tree are identical (21 files each, `c4_frozen_config.json`
absent from both).

## Why no fix is attempted

- Creating the missing artifacts would fabricate scientific evidence.
- Editing `verify_c4_results.py` (a frozen artifact with a locked hash)
  to soften the check would weaken a fail-loud gate.
- Removing it from `run_all_verifiers.py` would hide a real status
  signal.

The runner keeps failing loudly; this note records that the failure is
the correct verdict on an incomplete C4 final stage. A side observation
(not acted on): the bare `SystemExit(1)` makes the verdict needlessly
hard to diagnose; that UX wart lives in frozen code.
