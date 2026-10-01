# Repository hygiene inventory

Inventory baseline: branch `main`, HEAD `2a6933d8ddaa7d45d9175f7ce0558fbd52a454e4`, clean working tree, 600 tracked files. No item was deleted during inventory.

| Classification | Paths | Reason / action |
|---|---|---|
| KEEP | `project/results/v4/confirmatory/`, preregistration, corrected summaries, paper tables/figures | frozen scientific authority and correction chain |
| KEEP | tracked `project/results/**/*.csv`, `*.json`, `*.npz`, `*.pt`, notebooks | historical evidence or artifacts referenced by frozen reports; uncertainty defaults to preservation |
| KEEP | `project/ekf_verify.log`, `c4_post_ekf_verify.log`, `v3_post_ekf_verify.log` | verifier evidence referenced by the research notebook and V4 phase-9 note; future logs remain ignored |
| KEEP / ARCHIVAL | `project/results/baseline_prefix/`, `baseline_v3_prefix/`, old audits and V1–V3 reports | deliberate historical snapshots; current authority is labeled elsewhere |
| KEEP | `project/results/v4/prereg/h1_h9_execution_plan.json` and other large tracked plan/CSV files | preregistration or exact frozen evidence, not disposable generated noise |
| MOVE/ARCHIVE | ignored V4 dataset, model binaries/histories, raw run matrices, events, checkpoints | local-only scientific artifacts; archive externally using `project/ARCHIVAL_ARTIFACTS.md` |
| IGNORE | `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.uv-cache/`, virtual environments, `.env*`, notebook checkpoints | reproducible local/cache state |
| IGNORE | V4 runtime status, SQLite sidecars, generated raw V4 CSVs, V4 dataset/model binaries | precise rules already exist in `project/.gitignore` |
| REMOVE FROM GIT ONLY | none | no tracked artifact was proven redundant enough to untrack safely |
| DELETE | ignored cache directories only | safe local clutter; no scientific content |
| NEEDS REVIEW | external publication bundle for ignored V4 artifacts | choose Zenodo/OSF/institutional storage and record a DOI/hash inventory later |

Documentation classification: root `README.md`, `project/README.md`, paper, and `context/` are CURRENT; V1–V3 phase reports are HISTORICAL; original H1/H2/H7 summaries and old audits are SUPERSEDED but retained; baseline-prefix copies are ARCHIVAL; no documentation was deleted as REDUNDANT because the copies preserve provenance.

After inventory, only `.pytest_cache/`, `project/.pytest_cache/`, `project/.uv-cache/`, and the `__pycache__/` directories under `project/scripts/`, `project/src/`, and `project/tests/` were removed. Tests may recreate ignored bytecode caches locally.
