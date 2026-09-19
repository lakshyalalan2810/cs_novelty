# Final Repository Audit Report

**Project:** Reliability-Aware LSTM-Based Model Predictive Control for Fault-Tolerant Nonlinear DC Motor Control  
**Audit date:** 2026-09-01; final consistency update 2026-09-02  
**Scope:** cleanup, structure, reproducibility, saved-evidence verification, documentation, and publication packaging only. No controller/model algorithm or result tuning was performed.

## Readiness conclusion

The documentation and local Python environment are ready, but the repository is **not submission-ready** because a later exploratory `H=5` execution overwrote the audited final `H=20` configuration, controller tables, summary, plots, and notebook outputs. The verified values below are preserved as the prior audit record; the corresponding `H=20` artifacts must be restored from version history or regenerated before submission.

One operational prerequisite is outside this directory: `project/` has no project-scoped `.git` metadata and no configured remote, and it is not tracked by the parent Git index visible in this workspace. The content can be placed in or committed to a Git repository, but external clone hosting cannot be verified until the owner publishes it to a remote.

The current overwritten artifacts report `H=5`, `Nc=1`, a 0% sensor-fault improvement, a 0% adaptive disturbance difference, mean solve time `9.2811 ms`, p95 `14.8764 ms`, maximum `31.6085 ms`, zero optimizer failures, and zero constraint violations. Those values describe the later exploratory rerun and must not replace the verified final `H=20` submission record.

## Audit evidence

- Inspected all 79 packaged files present before this report was added.
- Parsed every notebook as JSON and compiled every code cell.
- Confirmed all code cells in notebooks 04B–06 have strictly increasing execution counts and no saved error outputs; notebooks 01–04 have intentionally cleared outputs but complete self-contained source.
- Read all five source modules and traced their notebook callers.
- Ran `src/reliability.py` and `src/mpc.py` self-checks successfully.
- Compiled `src/` and `scripts/` successfully.
- Reloaded the processed dataset and model weights and recomputed LSTM one-step and recursive metrics.
- Recomputed final controller claims from the 220-row run CSV.
- Visually inspected the five controller/scenario publication figures and the new runtime comparison.
- Scanned all packaged files for duplicate SHA-256 hashes; none were found.
- Scanned for remaining local environments, UV caches, Python bytecode caches, and notebook checkpoints; none remain.

## Files changed

### Added

- `.gitignore` — excludes local environments, UV caches, Python bytecode, and notebook checkpoints.
- `PROJECT_PIPELINE_AND_STATUS.md` — required evidence-backed project history, architecture, pipeline, results, limitations, and reproduction guide.
- `REPOSITORY_AUDIT_REPORT.md` — this audit record.
- `scripts/verify_results.py` — quick dataset/model/controller evidence verifier and runtime-plot generator.
- `results/plots/final_runtime_comparison.png` — missing publication-style runtime comparison against the 50 ms interval.
- `results/configs/` — dedicated configuration location.
- `results/raw/` — dedicated preserved raw diagnostics/trace location.

### Updated

- `README.md` — replaced the outdated stub/title with the final title, objective, architecture, actual tree, install steps, notebook order, full/quick reproduction, verified results, limitations, and contribution.
- `requirements.txt` — removed unused `control` and `cvxpy`; replaced the generic `jupyter` entry with the actual notebook interface `jupyterlab`.
- `results/configs/lstm_model_config.json` — normalized the saved dataset path from a Windows separator to `data/processed/dc_motor_lstm_dataset.npz`.
- `notebooks/03_lstm_training.ipynb` — writes configuration to `results/configs/` and saves the dataset path in POSIX form.
- `notebooks/04_reliability_analysis.ipynb` — reads config from `results/configs/`; writes raw diagnostics to `results/raw/`.
- `notebooks/04b_reliability_redesign.ipynb` — reads/writes dedicated config/raw locations.
- `notebooks/04c_reliability_finalization.ipynb` — reads/writes dedicated config/raw locations.
- `notebooks/05_mpc_experiments.ipynb` — reads/writes dedicated config/raw locations.
- `notebooks/06_final_evaluation.ipynb` — reads/writes dedicated config location and reproduces the final runtime plot.

### Moved without content duplication

From `results/metrics/` to `results/configs/`:

- `lstm_model_config.json`
- `reliability_config.json`
- `reliability_redesign_config.json`
- `reliability_final_config.json`
- `mpc_config.json`
- `final_mpc_config.json`

From `results/` to `results/raw/`:

- `reliability_diagnostics.npz`
- `reliability_redesign_diagnostics.npz`
- `reliability_final_diagnostics.npz`
- `mpc_closed_loop_traces.npz`

Notebook source paths and required-file assertions were updated consistently. A stale-path scan returned no old config/raw references.

## Files removed

Only reproducible local environments and caches were deleted:

| Target | Files | Bytes |
|---|---:|---:|
| `.venv/` | 32,036 | 895,647,119 |
| `.uv-cache/` | 29,944 | 861,510,987 |
| `.uv-python/` | 4,315 | 75,670,521 |
| `src/__pycache__/` | 5 | 76,894 |
| `notebooks/__pycache__/` | 1 | 53,974 |
| `scripts/__pycache__/` | 1 | 21,027 |
| **Total** | **66,302** | **1,832,980,522** |

Deletion is not directly recoverable, but every removed item is regenerated by Python/UV and contains no source or experiment evidence. No exploratory research notebook, ablation, raw diagnostic archive, saved metric, model weight, dataset, or plot was deleted.

## Execution and reproducibility issues found

| Finding | Resolution/status |
|---|---|
| README used the obsolete “Dual-Reliability Adaptive” title and only described an early workflow. | Replaced with final evidence-backed documentation. |
| `control` and `cvxpy` were listed but never imported. | Removed from `requirements.txt`. |
| Config JSONs were mixed into metrics. | Moved to `results/configs/`; notebook paths patched. |
| Raw diagnostic/trace NPZ files were loose in `results/`. | Moved to `results/raw/`; notebook paths patched. |
| Saved LSTM config contained a Windows path separator. | Stored as portable POSIX relative path. |
| Local environments/caches consumed about 1.83 GB. | Removed and ignored. |
| No dedicated runtime publication plot existed. | Added `final_runtime_comparison.png` and reproduction code. |
| Notebook outputs from Phase 5 contain transient `C:\Users\...\Temp\ipykernel...` warning paths. | These are historical stderr text only; source/config paths have no machine dependency. Outputs were preserved as experiment history. |
| Notebooks 01–04 have cleared execution counts/outputs. | Their code cells are self-contained, syntax-checked, and upstream artifacts exist. Full rerun remains the authoritative execution proof. |
| Final overall p95 was saved, but individual per-call final timing samples were not separately persisted. | Formula and source were audited; mean/max recompute from run telemetry; p95 is canonical in `final_summary.json`. Rerun notebook 06 for independent per-call reconstruction. |
| No project-scoped Git repository/remote exists in this workspace. | Content-ready; owner must commit/publish before external clone hosting. |

## Documentation inconsistencies found

- The old README title and “minimal starting point” description contradicted the completed project state. Corrected.
- Phase 4C explicitly recorded `ready_for_mpc=false` for the full three-state reliability proposal, while later notebooks implemented a narrower controller. The final documentation now states the actual simplification: MC Dropout and hard model-mismatch switching were removed; the sensor residual/CUSUM gate and a continuous monitoring/adaptation score were retained.
- Phase 3 recommended H=15 from recursive open-loop error, whereas the final MPC uses H=20. The documentation now explains that closed-loop stability/runtime screening superseded the initial prediction-only recommendation.
- Adaptive MPC did not win every disturbance scenario. The final docs report the actual 18.5827% aggregate degradation and do not claim universal improvement.
- The final timing does not support reliable 20 Hz execution. The README/status document explicitly report mean, p95, maximum, and the failed 50 ms criterion.
- A later notebook-06 execution selected `H=5` and overwrote the audited final artifacts. This was found during the 2026-09-02 consistency pass; no experimental file was edited or relabeled. Submission is blocked until the audited `H=20` artifacts are restored or regenerated.

## Final verified numerical results

### LSTM identification

- Test RMSE: `0.1408727 rad/s`.
- Test MAE: `0.1131055 rad/s`.
- Test R²: `0.9999062`.
- Persistence RMSE: `0.2897776 rad/s`.
- Recursive RMSE at H=5/10/15: `0.2109516 / 0.3364611 / 0.4877964 rad/s`.
- Recomputed train-only normalization matches the saved statistics to at least `5.41e-13` absolute difference.

### Reliability/controller claims

- Plain MPC sensor-fault interval RMSE: `7.5872084 rad/s`.
- Reliability-aware MPC sensor-fault interval RMSE: `1.4090244 rad/s`.
- Sensor-fault RMSE improvement: `81.4289478%`.
- Plain MPC disturbance/combined interval RMSE: `1.9093554 rad/s`.
- Adaptive MPC disturbance/combined interval RMSE: `2.2641661 rad/s`.
- Adaptive degradation: `18.5827461%`.
- Optimizer failures: `0`.
- Voltage violations: `0`.
- Slew violations: `0`.
- Total constraint violations: `0`.

### Final MPC runtime/configuration

- Configuration: H=`20`, Nc=`2`, warm start enabled, `maxiter=8`.
- Mean solve time: `69.4594309 ms`.
- Overall p95: `131.6933200 ms`.
- Maximum: `290.2360000 ms`.
- Previous Phase-5 mean: `199.6 ms`.
- Runtime reduction: `65.2007%`.
- Real-time at 50 ms/20 Hz: **No**.

## Remaining limitations

- MC Dropout is rejected from the core because it did not provide useful model-mismatch separation.
- Small bias and slow drift remain harder than large/noisy/dropout faults.
- Parameter-variation detection remains limited.
- Simultaneous sensor corruption and load disturbance is the main weak case: isolated sensor-fault robustness improved strongly, while the combined condition reduced the benefit of adaptive control.
- Dropout recovery may remain latched.
- Adaptive plant/model-mismatch handling does not consistently improve RMSE.
- The LSTM is open-loop trained and recursive error grows with horizon.
- Runtime is machine/load dependent and fails the strict 50 ms deadline.
- Evidence is simulation-only; hardware timing, calibration, and safety validation remain future work.
- External clone availability still requires a Git commit and remote publication outside this workspace.

## Final review gate

| Gate | Result |
|---|---|
| Final title/objective/architecture documented | Pass |
| Ordered notebooks and purpose documented | Pass |
| Source modules audited; no dead import candidates | Pass |
| Seeds, relative paths, splits, train-only normalization verified | Pass |
| Saved dataset/model/config artifacts present | Pass |
| Final claims recomputed from current saved evidence | **Fail; current artifacts are an overwritten H=5 rerun** |
| Canonical four-controller table complete | Structurally complete, but not the audited final H=20 table |
| Six concise final publication plots present | Pass |
| Negative findings and limitations preserved | Pass |
| Temporary environments/caches removed | Pass |
| Duplicate packaged files found | None |
| Project content ready for submission | **No; restore/regenerate audited H=20 artifacts** |
| Remotely cloneable from this workspace | **Not yet; no project Git remote** |
