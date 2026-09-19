# Why `results/baseline_v3_prefix/` is retained

## What it is

`results/baseline_v3_prefix/` is a 101-file pre-fix snapshot of the V1/V2/V3
evidence tree, preserved when stale V3 copies were removed from the active
tree (see `V3_FINAL_AUDIT.md` items 1 and 12). Contents: the seven-file V1
baseline bundle, eight configs, 48 metric files, 21 plots, five raw
artifacts, eight reports, both model weights, the V2 log, and the original
verifier.

## Why it must not be deleted

It is load-bearing for the frozen V3 verifier (`scripts/verify_v3_results.py`,
itself one of the 43 prestudy frozen artifacts):

1. **Holdout-unused proof** (`verify_v3_results.py:230-232`): the verifier
   scans every textual file under `results/baseline_v3_prefix/` and requires
   that the final holdout seeds `29026-29030` appear nowhere in the
   pre-fix evidence. Deleting the directory would silently remove this
   negative control.
2. **Snapshot-completeness gate** (`verify_v3_results.py:330-331`): the
   verifier requires six preserved files to exist
   (`metrics/v3_summary.json`, `metrics/v2_closed_loop_summary.json`,
   `metrics/final_summary.json`, `configs/v3_arbitration_config.json`,
   `reports/README.md`, `verifiers/verify_results.py`). Deleting the
   directory fails verification outright.

`scripts/verify_c4_results.py:875-883` additionally scans the directory for
C4 final-seed provenance when present, and
`results/configs/c4_candidate_seed_audit.json` records it in its audit scope.

## De-duplication assessment

No `*_hashes*.json` / `*frozen_hash_manifest*.json` manifest references any
path under `results/baseline_v3_prefix/`, so a de-duplication would not
violate a hash lock directly. It is nevertheless **not recommended**:

- Any de-duplication (hash-manifest replacement, pruning, or relocation)
  would require editing the frozen `verify_v3_results.py` to keep it green,
  which is itself a frozen artifact whose hash is locked in
  `results/final_robustness/prestudy_frozen_hash_manifest.json`.
- The directory's value is precisely its byte-level preservation of
  pre-fix evidence; replacing it with hashes would weaken the
  holdout-unused argument it supports.

Decision: retain `results/baseline_v3_prefix/` unchanged.
