# Archival artifacts

The Git repository carries final summaries, manifests, figures, and historical evidence. A clean source clone omits the following V4 artifacts because they are ignored. They were absent from this workspace at the start of the 2026-10-08 release audit; the original files were subsequently located in a separate local backup and restored without rerunning science.

| Local path | Purpose | Recorded integrity | Needed for |
|---|---|---|---|
| `data/processed/v4/dc_motor_v4_dataset.npz` | frozen V4 training dataset | SHA-256 `96f9187ca907d0bb524ad198dc618ea1dd9f079ee155821776f1bbdac0ea573f` | retraining or dataset-level audit |
| `results/v4/models/*.pt`, `*_history.npz` | eleven main/auxiliary model pairs and histories | new detached hashes in `paper/release/v4-artifact-inventory.json`; associated config/metrics/summary metadata match the Oct-1 baseline | model-level inspection/replay |
| `results/v4/faultfree/`, `sweep/`, `recovery/` | raw rows, events, and SQLite checkpoints for the 12,200-cell core | hashes are bound by `confirmatory/result_manifest.json` | full saved-row and checkpoint audit |
| `results/v4/core_execution_status.json` | preserved final runtime snapshot | exact bytes bound by the final result manifest; included only as provenance | historical verifier's completion check; not an instruction to resume |

## Local release prepared on 2026-10-08

The original backup was `C:\Users\Lakshya\OneDrive\Desktop\antenna\cs_novelty\project`. Only the explicitly listed ignored V4 artifacts were copied into the current project. No backup source, later research, new controller, new scientific result, or regenerated data was imported.

- The dataset and all nine run/event/checkpoint files match their frozen SHA-256 values exactly.
- All 22 V4 model binaries and 22 histories were inventoried. Their 45 associated configuration, metrics, and training-summary JSON files match the baseline semantically. The baseline has no standalone model/history hashes; the inventory records newly measured detached hashes and does not pretend they were preregistered hashes.
- The final 507-byte runtime snapshot matches its frozen manifest hash. It is provenance only.
- `paper/release/v4-artifact-inventory.json` lists each ignored file, byte count, SHA-256, source-clone absence, role, and integrity classification.
- `paper/release/SHA256SUMS.txt` covers the source and scientific files inside the local bundle.
- `paper/release/v4-frozen-evidence.zip` contains the exact committed source bytes at `4b7c51dabff16a702d230f2b9f87c14f73ebf9fd`, the original ignored artifacts at their repository-relative paths, an inventory, checksums, and an archive README. The binary ZIP is ignored by Git; its detached checksum is provided beside it for inclusion in the reviewed source release.

The source snapshot inside this bundle is the frozen scientific baseline. Final manuscript edits and the final PDF are separate publication deliverables; the bundle does not claim to contain a subsequently approved source commit. Its checksum inventory distinguishes the historical statistical summaries from the corrected summaries, preserving both. Original unavailable pre-refreeze plan bytes and the deliberately discarded 400-cell attempt are not reconstructed.

Rebuild the bundle from the original backup with:

```powershell
python project/paper/release/prepare_v4_archive.py ORIGINAL_PROJECT_DIRECTORY
```

The standard-library builder sorts entries, fixes ZIP timestamps/permissions, preserves original file bytes, and checks deterministic repeat bytes, ZIP CRCs, complete membership, and every entry's SHA-256. Determinism was verified within the available runtime; the detached ZIP hash identifies the resulting bytes across compression-library versions.

Final verification caught Windows text conversion in the initial local source packaging through `git archive`. The builder now lists the baseline files with `git ls-tree` and reads every source file through `git show`, preserving committed bytes directly. The rebuilt ZIP's complete source membership was independently compared against fresh Git-blob reads. Only local packaging changed; all original scientific artifacts and frozen scientific hashes remain unchanged.

## Remaining publication action

The bundle is **local, prepared, and not deposited**. No archive URL or DOI exists. Deposit it in Zenodo/OSF or an institutional archive after the authors choose artifact licensing and supply deposit metadata. Publish the ZIP checksum and inventory beside the deposit and cite a DOI only after one exists. A GitHub Release can serve as a convenience mirror.

Do not claim that a clean source clone can regenerate every paper figure or reconstruct raw inference without the archive. See `REPRODUCIBILITY.md` for the precise operation/dependency map and `paper/PUBLICATION_RELEASE_REPORT.md` for remaining human actions. Do not copy scientific artifacts into `context/`.
