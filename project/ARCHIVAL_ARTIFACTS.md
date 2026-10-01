# Archival artifacts

The Git repository intentionally carries small final summaries, manifests, figures, and historical evidence. The following V4 artifacts are local-only and ignored because they are generated or large:

| Local path | Purpose | Recorded integrity | Needed for |
|---|---|---|---|
| `data/processed/v4/dc_motor_v4_dataset.npz` | frozen V4 training dataset | SHA-256 `96f9187ca907d0bb524ad198dc618ea1dd9f079ee155821776f1bbdac0ea573f` | retraining or dataset-level audit |
| `results/v4/models/*.pt`, `*_history.npz` | eleven main/auxiliary model pairs and histories | configurations/metrics are tracked; compute a detached hash inventory before upload | model-level replay |
| `results/v4/faultfree/`, `sweep/`, `recovery/` | raw rows, events, and SQLite checkpoints for the 12,200-cell core | hashes are bound by `confirmatory/result_manifest.json` | full saved-row and checkpoint audit |
| `results/v4/core_execution_status.json` | mutable runtime progress/status | generated; not an archival result | resuming local execution only |

Store a future immutable bundle as a Zenodo/OSF or institutional archive deposit; a GitHub Release asset is acceptable for a convenience mirror. Publish a detached SHA-256 inventory beside the bundle and cite its DOI only after one exists. Do not copy these artifacts into `context/` or claim that a source-only clone can rerun the frozen study byte-for-byte.
