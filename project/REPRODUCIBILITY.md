# Reproducibility

Use `requirements.txt` for routine development and saved-evidence reconstruction on Python 3.11. It is a supported convenience environment, not the environment that produced the frozen V4 results.

`requirements-frozen-v4.txt` records every package version captured by `results/v4/confirmatory/result_manifest.json`: Python 3.11.15 (Anaconda), NumPy 2.2.6, pandas 2.3.3, SciPy 1.13.1, and PyTorch 2.6.0+cu124. The CUDA 12.4 PyTorch build requires a compatible NVIDIA platform. The manifest did not record the full OS, driver, solver, BLAS, or transitive-package state, so byte-identical re-execution is not established.

For the published evidence path, verify the saved hashes, corrected CSV/JSON summaries, paper structure, and provenance records. Do not rerun dataset generation, training, calibration, or the V4 protocols merely to verify the repository. Large local-only inputs and outputs are listed in `ARCHIVAL_ARTIFACTS.md`.
