# V4 pre-execution contract amendment

Amended: 2026-09-22T14:14:19+05:30

Status: FROZEN BEFORE ANY V4 CONFIRMATORY PROTOCOL EXECUTION. No H1-H9
outcome has been observed. Dataset generation, model training, sensor
calibration, and baseline calibration were already complete when this
amendment was made.

## Contract correction

The original implementation made direct C3/V4 pairing impossible for
training seeds 2029-2036: C3 has valid frozen model/config/calibration bundles
only for 2026-2028, while V4 has bundles for 2026-2036 and `training_seed` is
part of every paired-analysis key. Direct C3 comparisons (H1-H5 and H7-H9)
therefore use the valid common training-seed population `{2026, 2027, 2028}`.
No C3 artifact is fabricated and no missing pair may be silently discarded.

H6 compares B with V4_full_aux and retains all V4 training seeds 2026-2036.
The original umbrella sweep planner omitted B, so the exact H1-H9 core planner
explicitly includes the 550 required B bias-8-sigma cells. This repairs an
execution-plan omission without changing H6.

## Unchanged frozen elements

- The H1-H9 scientific questions are unchanged.
- Metrics and effect directions are unchanged.
- Family alpha and Holm correction are unchanged.
- The hierarchical bootstrap procedure, replicate count, and RNG seeds are
  unchanged.
- Clean-start pair-exclusion logic is unchanged.
- Fault magnitudes, references, horizons, and confirmatory simulation seeds
  are unchanged.
- H6 remains intention-to-treat on all 11 V4 training seeds.

This is a pre-execution contract correction, not a post-result change.

## Future execution operations

The audited machine has an Intel Core i7-13620H (10 physical cores, 16
logical processors), 15.70 GiB physical memory, and 5.29 GiB available during
this audit. Each controller process sets PyTorch intra-op threads to one.
Eight workers is the conservative starting point because it stays within the
physical-core count and leaves memory headroom:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python scripts/v4_protocol_core.py --workers 8
```

The timing protocol remains single-worker. The shared executor checkpoints
atomically after bounded batches and reports completed/total, percentage,
elapsed time, recent runs/minute, and checkpoint path. Future stall decisions
must use process health, checkpoint advancement, and CPU/memory state; absence
of stdout for 30 minutes is not a stop condition.
