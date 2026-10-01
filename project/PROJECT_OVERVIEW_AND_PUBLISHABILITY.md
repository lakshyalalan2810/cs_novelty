# Project overview and journal publishability

> **HISTORICAL REVIEW DOCUMENT.** It predates the final V4 cleanup; use the root `README.md`, `V4_CONFIRMATORY_RESULTS.md`, and `context/START_HERE.md` for current claims.

This note consolidates two reviews of the repository: what the codebase is, and whether it is journal-publishable. It is a read/analysis document. It does not change frozen evidence, thresholds, seeds, or controllers.

---

## 1. What this codebase is

This is a **research codebase** for **sensor-fault-tolerant DC-motor control**: constrained LSTM model-predictive control (MPC) that swaps in a learned “virtual speed” when the physical speed sensor looks untrustworthy. Everything is **simulation-only**. The repo is careful about frozen seeds, independent verifiers, and what claims are actually supported.

### 1.1 Control loop

```text
reference → constrained LSTM-MPC → voltage → nonlinear DC motor → sensor
    ↑                                                          |
    └── measured speed or LSTM virtual feedback ← reliability monitor
```

When the monitor flags the measurement, the controller uses a virtual estimate (main LSTM, auxiliary current-only LSTM, or later an EKF/observer) instead of the corrupted sensor.

The **defensible result** is improved robustness to **abrupt sensor faults** (bias, dropout) in simulation — on the order of **~82% RMSE improvement** in V1, stronger on some V3 bias/dropout cases. It does **not** claim hardware readiness, real-time 20 Hz deployment, or that adaptive MPC is generally better. Load disturbance with a healthy sensor is a known failure mode: false CUSUM latches can **worsen** tracking (~2.7× vs plain MPC).

### 1.2 Repository layout

Almost everything lives under `project/`:

| Path | Role |
|---|---|
| `src/` | Plant, LSTM, reliability, MPC, EKF, V4 detector |
| `notebooks/` | Ordered V1 pipeline (`01` … `06`) |
| `scripts/` | V2–V4 evaluation, calibration, verifiers, paper tables |
| `tests/` | Regression tests for frozen contracts and V4 protocols |
| `results/` | Frozen configs, metrics, traces, V4 models/calibration |
| `paper/` | LaTeX paper, figures, tables |
| `*.md` | Preregistration, audits, limitations, architecture notes |

Core Python modules:

- `src/motor_model.py` — nonlinear PM DC motor (`i`, `ω`), Coulomb friction, 0–12 V
- `src/lstm_model.py` — two-layer residual LSTM (voltage + measured speed → next true speed)
- `src/auxiliary_sensor_model.py` — smaller current-only LSTM witness
- `src/reliability.py` — residual/CUSUM monitor + dual-virtual arbitration (V3)
- `src/reliability_v4.py` — V4 detector (warmup, CUSUM clamp, witness AND-rule)
- `src/mpc.py` — SLSQP LSTM-MPC, voltage/slew constraints, PI fallback
- `src/ekf_observer.py` / `src/baselines_v4.py` — classical comparators

Timing: plant/model **10 ms**; control **50 ms** (20 Hz). Frozen MPC: horizon **H=20**, two blocks **(5, 15)**, **8** SLSQP iterations. Runtime is often **not** real-time (many solves > 50 ms).

### 1.3 Research versions (how to read the repo)

The project is versioned as **frozen evidence layers**, not a typical product app:

1. **V1** — reliability-aware LSTM-MPC vs plain MPC. Strong sensor-fault win; combined fault+load and runtime are limitations. Notebooks 01–06.
2. **V2/V3** — auxiliary virtual sensor + dual-estimator arbitration (**C3**). Frozen 275-run matrix (5 controllers × 11 scenarios × 5 seeds). C2 (aux recovery gate) was a **negative ablation**. Drift is essentially undetectable because the autoregressive LSTM absorbs slow bias.
3. **C4** — three-way consistency attribution to fix load false-alarms. Development stage **NO-GO** (behaviorally same as C3). V3 remains the shipped architecture.
4. **V4** (current work) — preregistered confirmatory study: new data, 11 training seeds (2026–2036), detector fixes (warmup, CUSUM clamp, witness AND-rule), vs C3 and classical baselines (EKF CUSUM, median gate, back-EMF observer, PI).

### 1.4 How to run it

Pinned stack is in `project/requirements.txt` (Python 3.11+ recommended). Reproduce V1 via notebooks in order, then:

```bash
python scripts/verify_results.py
```

Later stages have their own evaluators/verifiers (`verify_v3_results.py`, `v4_confirm_analysis.py`, `run_all_verifiers.py`). The culture of the repo is: **freeze config → hold out seeds → verify hashes and reconstructed metrics**.

### 1.5 Honest boundaries (documented, not hidden)

- Simulation only; thresholds calibrated on clean validation may not transfer.
- Load vs sensor fault is **not uniquely identifiable** from one residual.
- Slow **drift** is structurally hard for an AR LSTM.
- Combined fault+load is **training-seed dependent** (bootstrap CI can cross zero).
- MPC solve time is not a reliable 20 Hz real-time loop.

Useful companion docs: `README.md`, `LIMITATIONS.md`, `V3_DUAL_VIRTUAL_SENSOR_EXTENSION.md`, `V4_PREREGISTRATION.md`, `NEXT_RESEARCH_ARCHITECTURE_PLAN.md`.

---

## 2. Frozen numerical anchors (from project docs)

These numbers are already recorded in the README / V3 reports. They are restated here for context, not as a new analysis.

### 2.1 Method timing

- Plant/model sample time: `0.01 s`.
- Control update interval: `0.05 s` (20 Hz); each command is held for five plant samples.
- Frozen final MPC: `H=20`, `Nc=2`, `blocks=(5, 15)`, `maxiter=8`, warm start enabled.
- Voltage constrained to `0–12 V`; low-mismatch maximum move `2 V` per control update.

### 2.2 V1 headline (seeds 12026–12030; 4 controllers × 11 scenarios × 5 seeds = 220 runs)

| Quantity | Value |
|---|---:|
| Plain MPC sensor-fault interval RMSE | `7.584044` rad/s |
| Reliability-aware MPC sensor-fault interval RMSE | `1.374104` rad/s |
| Sensor-fault RMSE improvement | `81.881647%` |
| Plain MPC disturbance/combined interval RMSE | `1.932806` rad/s |
| Adaptive MPC disturbance/combined interval RMSE | `4.709976` rad/s |
| Signed adaptive degradation `(adaptive−plain)/plain` | `143.685962%` (worsening) |
| Runtime mean / median / p95 | `67.96` / `60.55` / `135.78` ms |

All 220 runs: zero optimizer failures, voltage violations, slew violations, and non-finite events. `58.44%` of solves exceeded 50 ms; **not reliably real-time at 20 Hz**.

### 2.3 LSTM evidence

- Held-out one-step RMSE `0.140873` rad/s; MAE `0.113106`; R² `0.999906`.
- Persistence RMSE `0.289778` rad/s.
- Recursive endpoint RMSE at H=5/10/15: `0.210952` / `0.336461` / `0.487796` rad/s.
- Main LSTM: two layers, 64 hidden units, dropout 0.2, 50,753 parameters.
- Auxiliary LSTM: one layer, ~4,641 parameters, `[voltage, current]` only.

### 2.4 V3 frozen matrix (seeds 29026–29030; 5 controllers × 11 scenarios × 5 seeds = 275 runs)

- Abrupt-fault RMSE reductions vs plain MPC (C3): **87.2%** (5% bias), **95.4%** (15% bias), **98.2%** (dropout).
- Drift: **zero benefit**.
- Combined 5% bias + load: ~**8.75%** improvement (training-seed dependent across the later robustness study).
- Load disturbance: C3 mean RMSE `2.538714` vs plain MPC `0.937127` rad/s (**~2.7× / 171% degradation**).
- C2 identical to C1 (aux recovery gate uniquely binding **0** times).
- All 275 runs: zero optimizer failures, nonfinite events, voltage or slew violations.

### 2.5 Known mechanistic limitations (V3)

1. **Load false activation.** A load step moves true speed; the main AR LSTM mispredicts through the transient; residual/CUSUM latches even though the sensor is healthy. After latch, virtual history can make tracking much worse.
2. **Drift undetectable.** Slow ramp bias enters the LSTM history; predictions track the drifted signal; residual stays small. Not a threshold-tuning issue.
3. **Combined fault+load is training-initialization dependent.** Part A hierarchical bootstrap 95% CI for paired C3−B fault-window RMSE `[−1.52, +0.29]` rad/s, crossing zero. Seed 2026 unfavorable; 2027/2028 favorable. Training seed accounted for ~73.9% of centered variation in that study.
4. **C2 recovery-gate degeneracy.** Aux recovery gate sized from clean-data extremes never binds during speed-sensor faults.

### 2.6 C4 (three-way attribution) — frozen NO-GO

Intended load geometry (sensor ≈ auxiliary, main outlier) **did not occur** at load latches. C4 was numerically identical to C3 on 105 paired development runs. Stage gate **NO-GO**. Seeds 39026–39030 remain unused. Do not retune C4 into a journal “fix.”

---

## 3. V4 confirmatory study (current paper story)

Status of the manuscript (`paper/main.tex`): IEEE conference skeleton titled *Reliability-Aware LSTM-MPC With Dual Witnesses…*. Abstract and results still contain `[PENDING: V4 confirmatory]`. Related work and conclusion are TODOs.

### 3.1 Controllers

Implemented in `scripts/v4_closed_loop.py`. MPC law and PI fallback are frozen for every MPC-based controller.

- `B`: plain LSTM-MPC on measured speed (tracking anchor).
- `C3`: frozen V3 stack (legacy reference).
- `V4_full_aux` / `V4_full_ekf`: ReliabilityMonitorV4 with full flags (CUSUM clamp k=2.0, 150-sample warmup with sensor trust, witness AND-rule) and main-LSTM substitution; witness from aux LSTM or frozen EKF.
- `V4_frozen_baseline_aux`: frozen-structure monitor on V4-trained models (isolates new data/models from new detector).
- `E2`: EKF + own CUSUM + EKF substitution.
- `S1`: fixed gate on `|y_meas − EKF ω|` + EKF substitution.
- `S2`: model-free median/rate plausibility + median substitution.
- `S3`: back-EMF Luenberger + own CUSUM + observer substitution.
- `PI`: PI on measured speed (mechanics anchor).

Primary V4 controllers: `V4_full_aux`, `V4_full_ekf`. Remaining factorial cells are exploratory.

### 3.2 Preregistered hypotheses (family α = 0.05, Holm)

- H1: `V4_full_aux` false-latch probability < `C3` (fault-free).
- H2: `V4_full_ekf` false-latch probability < `C3` (fault-free).
- H3/H4: detection probability > `C3` at bias 2σ, clean-start pairs (aux / ekf).
- H5: `V4_full_aux` recovery probability > `C3` on finite bias 8σ.
- H6: `V4_full_aux` fault-window RMSE < `B` at bias 8σ (intention-to-treat).
- H7: `V4_full_aux` tracking penalty vs `B` < `C3` penalty vs `B`.
- H8/H9: load false-entry probability < `C3` at 0.15 N·m (aux / ekf).

Planned confirmatory scale: on the order of **~268,910** closed-loop runs across fault-free, sweep, recovery, robustness, and timing protocols.

Retained negatives (not to be overturned by design): drift hardness, C3 load false-entry, current-channel corruption of witnesses, combined-fault seed sensitivity, single-speed-sensor scope.

---

## 4. Is this journal-publishable?

**Not as a journal paper today.** It is a strong, honest **simulation study** that could become publishable after confirmatory results and a rewritten paper — most realistically as a **conference / workshop paper**, or a **mid-tier applied journal** if claims stay narrow. Top control journals are a different bar.

The current `paper/main.tex` being an incomplete IEEE **conference** skeleton is the accurate manuscript status.

### 4.1 Why it is close (and worth finishing)

Scientific hygiene is unusually good for this kind of research repo:

- Frozen configs, holdout seeds, independent verifiers, and documented negative results.
- A **bounded** claim already exists: abrupt speed-sensor faults (bias/dropout) get large RMSE reductions vs unprotected LSTM-MPC; drift and load false-latching are called out as failures.
- Classical comparators (EKF, Luenberger, median gate) are in the V4 design — that was the main reviewer vulnerability for a two-LSTM story.
- Training-seed robustness, severity envelope, and current-sensor sensitivity were already identified as journal-must-haves and are partly in V3/Part B / V4.

That is closer to a defensible paper than most LSTM-MPC demos.

### 4.2 Why it is not journal-ready yet

1. **The confirmatory study is unfinished.** V4 is the paper’s actual contribution story (detector fixes + preregistered H1–H9). Until those runs exist and the Holm family is reported **as preregistered**, there is no complete manuscript. Filling V4 after seeing results would also undermine the preregistration.

2. **Novelty is combination, not a new theory.** LSTM-MPC, CUSUM residuals, analytic redundancy, and virtual sensors are all standard. Reviewers will ask: *why two LSTMs if an EKF on voltage/current already exists?* In development, E2 already looked **better** than C3 on some abrupt faults and worse under parameter mismatch. If the EKF wins on confirmatory data, the honest paper is a **comparative evaluation**, not “learned dual witnesses are superior.”

3. **Simulation-only on one plant.** One nonlinear DC-motor simulator, nominal training distribution, assumed-healthy current. No HIL, no hardware, no unmodeled dynamics beyond scripted scenarios. Applied journals (`Control Engineering Practice`, `ISA Transactions`, `Engineering Applications of Artificial Intelligence`) typically want at least a bench motor or HIL.

4. **Structural failure modes remain.** Drift is absorbed by the AR LSTM (not a tuning bug). Load steps look like sensor faults. Combined fault+load is **training-seed dependent**. C4 attribution **failed**. Those are scientifically valuable *if* they stay in the paper; they are fatal if the abstract still sounds like general fault-tolerant control.

5. **No real-time or stability claim is supported.** Mean solve ~68 ms vs 50 ms period. No recursive feasibility / Lyapunov argument. Do not add those claims.

6. **The manuscript itself is not written.** Related work is TODOs; bibliography is four canonical papers. A journal will desk-reject that.

### 4.3 Venue snapshot

| Venue class | Today | After V4 + a real paper |
|---|---|---|
| Workshop / student paper | Almost, if you freeze V3 only and stay humble | Yes |
| IEEE/IFAC conference | No (incomplete V4 + manuscript) | **Plausible** |
| Applied journal (CEP, ISA, EAAI) | No | Possible **with HIL/hardware or a win on load false-entry + EKF comparison** |
| `Automatica` / `IEEE TAC` / `TCST` | No | Unlikely without theory, hardware, and a sharper contribution |

---

## 5. Changes that would make it publishable

Match the **venue**, then do the minimum extra science. Do not add architecture after seeing confirmatory results.

### 5.1 Path A — Conference (most realistic next step)

Target examples: IEEE CCTA / ACC / CDC workshop / IFAC motor-drive or fault-diagnosis session.

Required:

1. **Finish V4 confirmatory as frozen.** Report H1–H9 once, Holm-corrected, with exclusions and intention-to-treat as written. If several hypotheses fail, that is still a paper if you do not silently retune.
2. **Write the paper as a bounded evaluation**, not a new architecture. Title/abstract should say *preregistered simulation evaluation of residual-gated virtual feedback*, including negative envelope (drift, load, seed sensitivity).
3. **Related work that actually exists:** LSTM-MPC (Zarzycki & Ławryńczuk and related), analytic redundancy / observer FDI, residual CUSUM, virtual sensors, sensor-fault-tolerant drives. Position against EKF/Luenberger, not only against “plain MPC.”
4. **Lead with the comparative question:** learned witnesses vs physics observer vs unprotected MPC. Let the EKF win if it wins.
5. **Keep C2/C4 as negative ablations** (one table, not a sequel architecture).
6. **Explicit non-claims:** simulation-only, not 20 Hz real-time, no isolation proof, single speed-sensor fault, healthy current unless the robustness cells say otherwise.

That is enough for a **good conference paper** if V4 numbers are coherent.

### 5.2 Path B — Mid-tier applied journal

Same as Path A, plus at least one of:

- **Hardware or HIL** on a real DC motor with injected sensor bias/dropout (even cheap encoder-fault injection).
- **Real-time controller** that actually meets the sample period (linearized/QP LSTM-MPC or a simpler law), reported on named hardware.
- A **clearer methodological increment** that survives if the EKF beats the LSTMs — e.g. calibrated false-latch vs detection decomposition, or a detector that actually reduces load false-entry **without** killing detection (H8/H9 must work).

Without hardware, aim only at journals that accept simulation method papers, and still expect a literature fight plus uncertainty quantification. Do not aim at `Automatica` / `IEEE TAC` without theory and generality.

### 5.3 Must-have vs should-have (from the architecture plan)

**Must have before journal submission** (several of these are already in V3/Part B or V4 design; they must appear in the paper, not only in repo notes):

- One fair classical observer baseline (EKF on voltage/current).
- Multiple independent training seeds per main/auxiliary pair (V4 uses 11; V3 robustness used 3).
- Compact fault-severity envelope with frozen thresholds.
- Paired effect sizes, confidence intervals, predeclared primary endpoints.
- Current-sensor assumption treated (single-fault statement + sensitivity).
- Bound all claims: simulation-only, drift weakness, load false-entry, parameter-transfer limits, no formal stability proof, 20 Hz timing not met.

**Should have:**

- One-at-a-time plant parameter variations and calibration-transfer.
- Operating-point residual analysis.
- End-to-end timing on a specified CPU target.
- HIL or a small benchtop experiment if targeting an applied-control venue.

**Optional / future, not this paper:**

- Trajectory-linearized / QP LSTM-MPC (separate real-time contribution; heavy prior art).
- C4-v2 or a learned attribution classifier (present signals do not justify it).
- UKF / sliding-mode / observer catalog.
- Formal robust stability / recursive feasibility (only if claimed).
- Full hardware campaign (after simulation claims survive observer and training-seed baselines).

### 5.4 What not to do to “make it publishable”

- More detector variants after seeing confirmatory results.
- Claiming three-way attribution after C4 NO-GO.
- Claiming drift tolerance.
- Claiming real-time or hardware readiness from current SLSQP timings.
- Expanding to UKF/SMO/multi-observer catalogs unless the EKF exposes a specific gap.
- Inflating novelty around “dual witnesses” if they do not beat a current-fed EKF.
- Claiming the auxiliary LSTM is independent of all sensors; it assumes healthy current.

---

## 6. Recommended contribution story

The strongest defensible story (whether V4 “wins” or not):

> A reliability-aware constrained LSTM-MPC architecture uses input-diverse virtual speed estimates to maintain control under **abrupt** speed-sensor faults. A frozen, paired, multi-scenario evaluation shows large gains for bias and dropout, while explicitly identifying failure boundaries under load false alarms, drift, plant-parameter shift, training-seed sensitivity, and real-time computation. A classical current/voltage EKF comparator, multiple training seeds, and a fixed-threshold severity envelope establish whether the learned dual-sensor design adds value beyond a physics observer. Negative C2 and C4 ablations show that extra recovery and scalar three-way consistency gates do not solve causal fault attribution.

Novelty should **not** be “three models vote” or “LSTM-MPC is real time.” It should be the rigorously bounded reliability/feedback-selection result, including where it fails and why.

If the EKF wins, the honest contribution becomes a **comparative** result that motivates a simpler model-based or hybrid successor. The paper should not protect the two-LSTM story at the expense of the evidence.

If V4 H1/H2/H8 fail (false latch / load not fixed) and the EKF dominates abrupt faults, the still-publishable paper is:

> Learned residual substitution helps vs unprotected MPC on abrupt faults, fails on drift/load, and does not beat a current-based EKF.

A conference (and some journals) will take that more readily than a rewritten “new V5 detector.”

---

## 7. Shortest credible path

1. Finish the **preregistered V4** comparison (do not retune after seeing results).
2. Write a **tightly bounded evaluation paper** with real related work and explicit non-claims.
3. Lead with **EKF vs learned witnesses vs unprotected MPC**.
4. Keep C2/C4 as **negative ablations**.
5. Add **HIL/hardware** only if the selected venue is an applied journal.

Do not start trajectory-linearized MPC, explicit neural control, or a new attribution architecture in the current paper-hardening phase.
