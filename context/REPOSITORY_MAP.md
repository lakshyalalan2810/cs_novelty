# Repository map

The checkout has two layers:

~~~text
cs_novelty/
├── README.md                         outer orientation
├── project/                          research repository
│   ├── README.md                     project orientation
│   ├── src/                          plant, predictors, MPC, reliability, EKF
│   ├── scripts/                      dataset, training, protocol, analysis, audit
│   ├── data/                         raw and processed datasets
│   ├── models/                       model checkpoints and bundles
│   ├── results/                      metrics, configs, V4 run artifacts, paper data
│   ├── paper/                        main.tex, figures, tables, audit
│   ├── notebooks/                    historical and executed research notebooks
│   └── tests/                        read-only and protocol verification tests
└── context/                          this additive orientation package
~~~

## Important source directories

| Path | Role |
|---|---|
| project/src/motor_model.py | nonlinear PMDC equations and numerical plant integration |
| project/src/data_utils.py | trajectory generation, windows, splits, normalization |
| project/src/lstm_model.py | main residual LSTM |
| project/src/auxiliary_sensor_model.py | current/voltage-only auxiliary LSTM |
| project/src/ekf_observer.py | augmented current/speed/load-torque EKF |
| project/src/mpc.py | differentiable rollout, SLSQP MPC, PI fallback |
| project/src/reliability.py | V1/V3 residual monitor, arbitration, C4 helpers |
| project/src/reliability_v4.py | V4 witness-gated monitor |

## Important script families

| Path | Role |
|---|---|
| project/scripts/v4_closed_loop.py | actual V4 controller stack and cell execution |
| project/scripts/v4_protocol_common.py | shared seeds, configs, keys, checkpoint/output writing |
| project/scripts/v4_protocol_core.py | exact H1–H9 core planner/runner |
| project/scripts/v4_protocol_faultfree.py | fault-free protocol wrapper |
| project/scripts/v4_protocol_sweep.py | severity/load/detection sweep wrapper |
| project/scripts/v4_protocol_recovery.py | finite-fault recovery wrapper |
| project/scripts/v4_detector_configs.py | frozen V4 variant definitions |
| project/scripts/v4_train_models.py | V4 model training |
| project/scripts/v4_generate_dataset.py | V4 dataset generation |
| project/scripts/v4_calibrate.py | V4 residual and witness calibration |
| project/scripts/v4_confirm_analysis.py | historical confirmatory summary code |
| project/scripts/v4_statistics_audit.py | corrected hierarchical bootstrap audit |
| project/scripts/v4_confirm_verify.py | historical verifier; reproduces the clustering defect |
| project/scripts/v4_repair_ekf_witness_cells.py | targeted EKF integrity repair |
| project/scripts/evaluate_v3_closed_loop.py | legacy V3 evaluation path |
| project/scripts/verify_results.py | V1/final repository verifier |
| project/scripts/verify_c4_results.py | C4 development verifier |

## Important artifact directories

| Path | Evidence role |
|---|---|
| project/results/v4/confirmatory/ | current H1–H9 summaries, manifest, correction, integrity |
| project/results/v4/faultfree/ | 9,600-cell core subset |
| project/results/v4/sweep/ | 2,000-cell core subset |
| project/results/v4/recovery/ | 600-cell core subset |
| project/results/v4/calibration/ | per-seed calibration bundles |
| project/results/metrics/ | V1/V3/C4/EKF/training-seed metrics |
| project/results/configs/ | frozen controller and observer configurations |
| project/paper/ | manuscript, tables, figures, final audit |

## Directory lifecycle and ownership

| Directory | Who writes it | Who reads it | Current/historical | Frozen/large-artifact note |
|---|---|---|---|---|
| project/src/ | source edits and tests | scripts, notebooks, verifiers | current implementation plus historical helpers | small text; modify only with explicit scope |
| project/scripts/ | protocol/training/analysis utilities | researchers and verifiers | mixed current and historical entry points | some commands are expensive or mutating |
| project/tests/ | test authors | CI/manual audit | current checks, though some bind historical constants | read first; tests do not override current evidence |
| project/notebooks/ | notebook execution | historical research review | mostly V1–V3 historical | large outputs may be embedded; do not rerun casually |
| project/data/ | generators and preprocessing | training/evaluation | V4 dataset is current; raw data directory is empty | NPZ datasets are large local-only artifacts |
| project/models/ | training scripts | evaluators | historical model checkpoints | PT weights are large/local-only |
| project/results/configs/ | calibration/configuration scripts | runtime and verifiers | frozen configs plus historical configs | small configs; preserve hashes |
| project/results/metrics/ | evaluators and audits | paper and verifiers | V1–V3/C4/EKF/training-seed evidence | mixed size; some CSVs are large |
| project/results/v4/ | V4 protocol writers and audits | analysis, verifiers, paper | current core plus historical/superseded outputs | CSVs, SQLite, NPZ/PT artifacts are local-only |
| project/paper/ | table/figure generators and authors | reviewers | current manuscript and audit | PDFs are generated artifacts; LaTeX build unavailable |
| project/*.md | researchers and audits | humans and AIs | mixed; use authority hierarchy | old reports are preserved, not automatically current |
| context/ | this orientation build | future AIs/humans | current additive package | documentation only; no scientific artifacts copied |

## Orientation rule

Use [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md) before selecting a file from this map. A filename that sounds final is not automatically authoritative: historical summaries, old audits, old hypothesis tables, and pre-fix reports remain in the tree for provenance.

Related: [FILE_INDEX.md](FILE_INDEX.md), [CODE_CONNECTIONS.md](CODE_CONNECTIONS.md), [DO_NOT_TOUCH.md](DO_NOT_TOUCH.md).
