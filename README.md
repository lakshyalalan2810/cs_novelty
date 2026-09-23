# Disturbance-Aware Witness Gating for Reliability-Aware LSTM-MPC of a Nonlinear DC Motor

> Simulation research on reliability-aware learned MPC, virtual sensing, observer-based witness gating, and preregistered evaluation of speed-sensor fault handling in a nonlinear permanent-magnet DC motor.

This repository contains the complete research path from the original reliability-aware LSTM-MPC controller through auxiliary virtual sensing, dual-sensor arbitration, negative ablations, an EKF comparator, robustness studies, and the final preregistered V4 confirmatory evaluation.

The central result is intentionally narrow:

> **Under the preregistered 0.15 N·m load-disturbance condition, both V4 witness-gating variants significantly reduced load-induced false sensor-fault entries relative to the frozen V3 C3 controller after Holm correction. Broader improvements in fault detection, recovery, and tracking were not established.**

The project is **simulation-only**. It does not claim universal fault tolerance, formal fault isolation, certified stability, hardware/HIL validation, current-sensor fault tolerance, or reliable real-time execution at 20 Hz.

---

## Current status

| Item | Status |
|---|---|
| Nonlinear PMDC plant + LSTM-MPC baseline | Complete |
| V1/V2/V3 reliability architecture | Complete / frozen |
| C2 recovery-gate ablation | Closed negative result |
| C4 three-way attribution | **NO_GO** |
| Augmented-state EKF comparator | **BASELINE_ONLY** |
| Training-seed robustness | **TRAINING-SEED-SENSITIVE** |
| Final V3 robustness/severity study | Complete / frozen |
| V4 H1–H9 confirmatory core | **Complete and audited** |
| Final paper integration | **Ready for final human read-through** |
| Final PDF build | Not produced locally; LaTeX toolchain unavailable |

The final manuscript source is:

**[Disturbance-Aware Witness Gating for Reliability-Aware LSTM-MPC of a Nonlinear DC Motor](project/paper/main.tex)**

---

## Research question

Can a learned residual monitor detect a corrupted motor-speed sensor and substitute virtual feedback **without confusing genuine speed-sensor faults with plant transients, load disturbances, model mismatch, or observer/model disagreement?**

The project develops that question in stages:

1. simulate and validate a nonlinear permanent-magnet DC motor;
2. train a residual LSTM predictor from voltage and measured speed;
3. embed the learned model in constrained nonlinear MPC;
4. detect unreliable speed measurements using residual gates and two-sided CUSUM;
5. substitute virtual feedback when the physical speed sensor is considered unreliable;
6. add an independent current-informed auxiliary LSTM;
7. develop dual virtual-sensor arbitration;
8. test and reject a three-way scalar-consistency attribution extension;
9. compare against an augmented-state EKF and classical observer/plausibility baselines;
10. quantify training-seed, fault-severity, load-disturbance, and current-channel limits;
11. preregister and execute a V4 witness-gating study targeted at the V3 load-confounding failure mode.

---

## Final V4 confirmatory result

The original V4 umbrella contained **268,910** possible cells. The final confirmatory study executed the exact deduplicated **12,200-cell H1–H9 dependency core**:

```text
9,600 fault-free
2,000 sweep
  600 recovery
----------------
12,200 total
```

The remaining **257,260** umbrella cells were deferred and contribute to no confirmatory result.

Positive delta supports each preregistered hypothesis.

| H | Endpoint | Delta | Holm-adjusted p | Decision |
|---|---|---:|---:|---|
| H1 | false latch, C3 − V4-aux | +0.209167 | 0.2508 | Do not reject |
| H2 | false latch, C3 − V4-EKF | +0.209167 | 0.2508 | Do not reject |
| H3 | bias-2σ detection, V4-aux − C3 | −0.100775 | 0.0763 | Do not reject |
| H4 | bias-2σ detection, V4-EKF − C3 | −0.100775 | 0.0763 | Do not reject |
| H5 | bias-8σ recovery, V4-aux − C3 | 0.000000 | 1.0000 | Do not reject |
| H6 | bias-8σ RMSE, B − V4-aux | −2.557027 | 0.0763 | Do not reject |
| H7 | tracking-penalty contrast | −0.063573 | 0.2508 | Do not reject |
| H8 | load false entry, C3 − V4-aux | **+0.217742** | **<0.0009** | **Supported** |
| H9 | load false entry, C3 − V4-EKF | **+0.217742** | **0.0008** | **Supported** |

Only **H8 and H9** survive Holm correction at family (alpha=0.05).

### What that means

At the tested **0.15 N·m load step**, both:

- the auxiliary-LSTM witness variant, and
- the EKF witness variant

reduced false entry into sensor-fault handling relative to C3.

The result supports a specific mechanism: **witness gating can protect reliability-entry logic against this tested load-disturbance confound**.

It does **not** establish:

- general speed-sensor-fault superiority,
- improved low-bias detection,
- improved high-bias recovery,
- improved severe-bias tracking,
- universal disturbance robustness,
- formal diagnosis or fault isolation.

The complete corrected table, confidence intervals, exclusions, effect sizes, and p-values are in **[V4_CONFIRMATORY_RESULTS.md](project/V4_CONFIRMATORY_RESULTS.md)**.

---

## Why V4 was needed

The frozen V3 controller handled several abrupt speed-sensor faults well, but exposed an important failure mode:

> A genuine plant/load disturbance can cause enough residual/CUSUM evidence to trigger a false speed-sensor-fault latch even when the physical speed sensor is healthy.

Historical V3 evidence showed:

- strong abrupt bias/dropout protection in important tested cases;
- weak drift detection;
- load-induced false reliability entries;
- parameter-mismatch sensitivity;
- combined fault-plus-load results sensitive to trained-model realization;
- dependence on an assumed-healthy armature-current channel.

V4 therefore focused on **entry discrimination**, not on claiming a universal replacement for V3.

---

## System architecture

The plant advances at **10 ms**, while the controller updates every **50 ms**.

```text
                     +----------------------+
 reference ---------->| constrained LSTM-MPC |------ voltage ------+
                     +----------------------+                     |
                               ^                                  v
                               |                       +----------------------+
                               |                       | nonlinear PMDC motor |
                               |                       | + load disturbance   |
                               |                       +----------------------+
                               |                            |           |
                               |                          speed       current
                               |                            |           |
                               |                            v           v
                               |                    +-----------+  +-----------+
                               +--------------------| feedback  |<-| witness   |
                                                    | selector  |  | aux / EKF |
                                                    +-----------+  +-----------+
                                                          ^
                                                          |
                                                   +--------------+
                                                   | reliability  |
                                                   | monitor / V4 |
                                                   | witness gate |
                                                   +--------------+
                                                          ^
                                                          |
                                                     main LSTM
```

### Main components

| Component | Implementation | Role |
|---|---|---|
| Nonlinear motor | [`project/src/motor_model.py`](project/src/motor_model.py) | RK4 PMDC dynamics with viscous + smoothed Coulomb friction and load torque |
| Main learned model | [`project/src/lstm_model.py`](project/src/lstm_model.py) | 2-layer residual LSTM using a 20-sample `[voltage, speed]` history |
| MPC | [`project/src/mpc.py`](project/src/mpc.py) | Constrained nonlinear LSTM-MPC with SLSQP, warm start, voltage and slew limits |
| Frozen reliability logic | [`project/src/reliability.py`](project/src/reliability.py) | Residual gate + two-sided CUSUM + persistence/hysteresis |
| Auxiliary virtual sensor | [`project/src/auxiliary_sensor_model.py`](project/src/auxiliary_sensor_model.py) | 1-layer LSTM using only `[voltage, armature current]` |
| EKF observer | [`project/src/ekf_observer.py`](project/src/ekf_observer.py) | Augmented `[current, speed, load torque]` EKF |
| V4 reliability monitor | [`project/src/reliability_v4.py`](project/src/reliability_v4.py) | Startup gating, CUSUM clamping, witness agreement, bounded recovery |
| V4 closed loop | [`project/scripts/v4_closed_loop.py`](project/scripts/v4_closed_loop.py) | Confirmatory controller execution |
| Confirmatory analysis | [`project/scripts/v4_confirm_analysis.py`](project/scripts/v4_confirm_analysis.py) | H1–H9 inference |
| Independent verifier | [`project/scripts/v4_confirm_verify.py`](project/scripts/v4_confirm_verify.py) | Independent confirmatory reconstruction |

Armature current is treated as an additional physical channel correlated with electromagnetic torque and load-induced motor dynamics. It does **not** directly measure load.

---

## Historical research path

| Stage | Final status | Main finding |
|---|---|---|
| V1 | Complete / frozen | Established original reliability-aware LSTM-MPC sensor-fault handling |
| V2 | Complete | Added an independent current-informed auxiliary virtual sensor |
| V3 / C3 | Complete / frozen | Dual virtual-sensor arbitration; strong abrupt-fault results with important load/disturbance limitations |
| C2 | Negative ablation | Auxiliary recovery gate was never the active recovery constraint |
| C4-v1 | **NO_GO** | Three-way scalar consistency did not solve load false entries |
| EKF study | **BASELINE_ONLY** | Strong on some isolated faults, weaker under several mismatch/load conditions |
| Training-seed study | **TRAINING-SEED-SENSITIVE** | Combined fault+load behavior changed materially across learned-model realizations |
| Final V3 robustness | Complete / frozen | Severity, seed, load, and current-channel boundaries quantified |
| V4 | **Confirmatory study complete** | H8/H9 support reduced load-induced false entries at 0.15 N·m |

### C4 negative result

C4 tested pairwise distances among:

```text
physical sensor
main LSTM
auxiliary LSTM
```

with:

```text
d_sm = |sensor - main|
d_sa = |sensor - auxiliary|
d_ma = |main - auxiliary|
```

The intended plant/main-mismatch geometry did not appear in the preregistered development matrix, and the stage gate returned **NO_GO**. The project preserves this as a negative ablation rather than presenting it as a successful fault-isolation method.

### EKF role

The EKF uses the augmented state:

```text
[current, angular speed, load torque]
```

with voltage and armature current as online inputs. It is retained as a classical comparator and independent witness, not as an overall winner.

---

## Main LSTM evidence

The frozen seed-2026 main predictor uses two LSTM layers, hidden size 64, dropout 0.2, a 20-sample history, and residual prediction.

| Metric | Frozen result |
|---|---:|
| Held-out one-step RMSE | **0.140873 rad/s** |
| Held-out MAE | **0.113106 rad/s** |
| Held-out R² | **0.999906** |
| Persistence-baseline RMSE | **0.289778 rad/s** |
| Recursive endpoint RMSE, H=5 | **0.210952 rad/s** |
| Recursive endpoint RMSE, H=10 | **0.336461 rad/s** |
| Recursive endpoint RMSE, H=15 | **0.487796 rad/s** |

The project treats one-step prediction quality as necessary but not sufficient for reliable closed-loop substitution.

---

## Reproducibility and provenance

The repository follows a progressively stricter:

```text
freeze → evaluate → verify
```

workflow.

The V4 closeout includes several explicit provenance disclosures:

- an initial **400-cell** attempt was discarded before confirmatory analysis after a plan-file provenance defect;
- the scientific plan was independently reconstructed **12,200/12,200** and execution restarted from zero under an administratively re-frozen immutable plan;
- pre-analysis validation identified a deterministic initialization defect in **all and only 2,700 required V4_full_ekf cells**;
- those 2,700 cells were rerun by an outcome-independent mechanical predicate, while the other **9,500** cells were left untouched;
- a final audit found that historical H1/H2/H7 bootstrap code split simulation-seed reference clusters;
- corrected H1/H2/H7 summaries were recomputed from the unchanged frozen core while preserving the historical outputs;
- no H1–H9 reject/do-not-reject decision changed.

Authoritative final manuscript-facing values use the superseding result manifest:

```text
0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e
```

Key audit documents:

- [V4 confirmatory results](project/V4_CONFIRMATORY_RESULTS.md)
- [V4 statistical and repair audit](project/V4_STATISTICAL_AND_REPAIR_AUDIT.md)
- [V4 execution-plan re-freeze note](project/V4_EXECUTION_PLAN_REFREEZE_NOTE.md)
- [Final paper audit](project/PAPER_FINAL_AUDIT.md)
- [Final figure manifest](project/paper/FIGURE_MANIFEST.md)
- [Known limitations](project/LIMITATIONS.md)

---

## Final figures

The manuscript uses six deterministic vector figures generated only from frozen evidence:

1. research evolution and failure-driven design;
2. final reliability-gated architecture;
3. historical reliability-entry timeline;
4. conditional-detection context;
5. corrected H1–H9 effect plot;
6. load-induced false entries by training seed.

Full provenance and aggregation rules are in **[project/paper/FIGURE_MANIFEST.md](project/paper/FIGURE_MANIFEST.md)**.

---

## Repository structure

```text
cs_novelty/
├── README.md
└── project/
    ├── src/                         # plant, LSTM, MPC, reliability, EKF
    ├── scripts/                     # training, calibration, protocols, analysis, verification
    ├── tests/                       # regression, provenance, confirmatory and paper checks
    ├── notebooks/                   # original notebook-based research path
    ├── paper/
    │   ├── main.tex                 # final manuscript source
    │   ├── references.bib
    │   ├── FIGURE_MANIFEST.md
    │   ├── figures/
    │   └── tables/
    ├── results/
    │   ├── metrics/                 # frozen V1/V2/V3/C4/EKF evidence
    │   ├── training_seed_robustness/
    │   ├── final_robustness/
    │   └── v4/                      # V4 plans, calibration, summaries, provenance
    ├── V4_CONFIRMATORY_RESULTS.md
    ├── V4_STATISTICAL_AND_REPAIR_AUDIT.md
    ├── V4_EXECUTION_PLAN_REFREEZE_NOTE.md
    ├── PAPER_FINAL_AUDIT.md
    ├── LIMITATIONS.md
    └── requirements.txt
```

Large runtime checkpoints, generated datasets, model binaries, and raw V4 run matrices are intentionally kept out of Git. Reproducibility is preserved through source code, configurations, small summaries, manifests, hashes, and audit records.

---

## Quick start

```bash
git clone https://github.com/lakshyalalan2810/cs_novelty.git
cd cs_novelty/project

python -m venv .venv
```

### Windows / PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Verify saved evidence

Run non-destructive verifiers from `project/`:

```bash
python scripts/verify_results.py
python scripts/verify_v3_results.py
python scripts/verify_ekf_results.py
python scripts/verify_training_seed_robustness.py
python scripts/verify_final_robustness_study.py
```

The confirmatory and paper-specific verification logic is also covered by the V4 test and audit suite.

The final paper audit reports:

- **87 V4 tests passed**
- **13,746 subtests passed**
- historical verifiers passed
- Python `compileall` passed
- six deterministic vector figures regenerated successfully
- 0 missing figure/table includes
- 0 duplicate LaTeX labels
- 0 unresolved manuscript references
- 0 unresolved citation keys

A final PDF was not built in the audited environment because `latexmk`, `pdflatex`, and `xelatex` were unavailable.

---

## Scope and limitations

This repository is research code for a simulated nonlinear PMDC motor.

The final evidence remains bounded by:

- no hardware-in-the-loop or physical-motor validation;
- no formal stability guarantee;
- no formal fault-isolation proof;
- no supported current-sensor-fault tolerance;
- both current-informed witnesses assume a healthy current channel;
- V4 dropout, drift, combined-fault expansion, robustness OFAT, timing, other sweep severities, and non-bias recovery studies remain deferred;
- gradual drift and parameter mismatch remain important historical weaknesses;
- direct historical robustness comparisons include only three C3 training seeds and show training-seed sensitivity;
- the SLSQP MPC runtime does not support a strong 20 Hz real-time claim;
- V4 confirmatory inference has finite bootstrap resolution and only three top-level training-seed clusters for several hypotheses.

See **[LIMITATIONS.md](project/LIMITATIONS.md)** for the full mechanistic discussion.

---

## Important documents

For the shortest path through the final project:

- **[paper/main.tex](project/paper/main.tex)** — final manuscript source.
- **[PAPER_FINAL_AUDIT.md](project/PAPER_FINAL_AUDIT.md)** — paper-level claims, number, figure, and structure audit.
- **[V4_CONFIRMATORY_RESULTS.md](project/V4_CONFIRMATORY_RESULTS.md)** — corrected H1–H9 confirmatory closeout.
- **[V4_STATISTICAL_AND_REPAIR_AUDIT.md](project/V4_STATISTICAL_AND_REPAIR_AUDIT.md)** — bootstrap correction and EKF repair provenance.
- **[V4_EXECUTION_PLAN_REFREEZE_NOTE.md](project/V4_EXECUTION_PLAN_REFREEZE_NOTE.md)** — administrative re-freeze record.
- **[LIMITATIONS.md](project/LIMITATIONS.md)** — measured limitations and mechanistic explanations.
- **[paper/FIGURE_MANIFEST.md](project/paper/FIGURE_MANIFEST.md)** — figure provenance.
- **[V4_PREREGISTRATION.md](project/V4_PREREGISTRATION.md)** — preregistered V4 design.
- **[FINAL_ROBUSTNESS_AND_SEVERITY_STUDY.md](project/FINAL_ROBUSTNESS_AND_SEVERITY_STUDY.md)** — historical V3 robustness boundary study.
- **[TRAINING_SEED_ROBUSTNESS.md](project/TRAINING_SEED_ROBUSTNESS.md)** — training-seed sensitivity study.

---

## Publication status

The manuscript source has completed the repository-level scientific, statistical, provenance, figure, and structural audits and is marked:

> **PAPER READY FOR FINAL HUMAN READ-THROUGH**

The remaining publication tasks are external to the scientific experiment:

1. add final author/affiliation/contact metadata;
2. compile and visually inspect the manuscript PDF with a LaTeX toolchain;
3. adapt formatting to the selected journal/conference;
4. perform the final human submission review.

No additional experiment is required to support the current bounded conclusion.
