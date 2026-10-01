# Data flow

## End-to-end V4 cell

~~~text
RunConfig
   |
   v
protocol wrapper -> ControllerStack -> run_v4_cell
                                      |
                                      v
             plant state x=[i,omega] and applied voltage u
                                      |
       +------------------------------+------------------------------+
       |                              |                              |
       v                              v                              v
  true speed/current            noisy/faulted speed              measured current
                                       |                              |
                                       v                              v
                          main history [u, speed]       aux/EKF history [u,current]
                                       |                              |
                                       v                              v
                                   y_main                  y_aux or omega_EKF
                                       |                              |
                                       +----------+-------------------+
                                                  v
                                      residuals and V4 monitor
                                                  |
                                                  v
                                            y_feedback
                                                  |
                                                  v
                                               MPC/PI
                                                  |
                                                  v
                                       applied voltage at next step
                                                  |
                                                  v
                                      rows, events, checkpoint
~~~

## Training data path

1. Dataset generation creates complete trajectories from the nonlinear plant under startup, MPC, PI, open multisine, and open random-step conditions.
2. Trajectory-level splitting prevents windows from crossing train/validation/test run boundaries.
3. Normalization is fitted on training data only.
4. A 20-sample window predicts the next true speed.
5. Main LSTM input uses voltage and measured/selected speed; auxiliary LSTM input uses voltage and current only.
6. V4 model bundles are trained for seeds 2026–2036. C3 bundles exist for seeds 2026–2028.

The training pipeline is implemented by project/scripts/v4_generate_dataset.py, project/src/data_utils.py, and project/scripts/v4_train_models.py. The dataset identity and split details are recorded in the V4 preregistration and result manifest; do not infer a new split from a local regenerated dataset.

## Runtime alignment

- Model step: 0.01 s.
- Controller update: every 5 model samples, 0.05 s.
- History length: 20 samples.
- The first inference uses the post-history state at approximately 0.20 s.
- Main residual is measured speed minus main prediction.
- Witness residual is measured speed minus auxiliary prediction or EKF speed.
- V4 debounced latching requires both main abnormality and witness abnormality for witness-gated variants.
- Substitution feeds the main prediction into MPC; witness estimates are not selected.
- Events record debounced entry and recovery transitions, not every instantaneous substitution.

## Fault injection

The V4 engine supports:

- additive speed bias over a half-open interval;
- speed dropout to zero for a configured duration;
- speed drift that ramps and saturates;
- persistent load torque changes;
- combined speed bias plus load;
- current noise and quantization;
- sample delays;
- plant parameter presets.

The confirmatory core uses only the exact subset required for H1–H9. The broad severity, robustness, DET, and timing protocols in the preregistration were not all executed.

## Logging path

Each protocol cell writes a canonical run row and event rows. Shared protocol code computes a SHA-256 key from the complete RunConfig, writes atomically through SQLite checkpoints, and exports:

    project/results/v4/{faultfree,sweep,recovery}/runs.csv
    project/results/v4/{faultfree,sweep,recovery}/events.csv
    project/results/v4/{faultfree,sweep,recovery}/checkpoint.sqlite3

The current V4 protocol writer does not persist full time-series traces from execute(); the confirmatory evidence is row/event based. Figure 6 uses the saved sweep rows rather than a selected trajectory.

## Reproducibility and paper pipeline

~~~text
frozen preregistration and plan
              |
              v
protocol core -> checkpoint.sqlite3 -> canonical runs/events CSV
                                      |
                                      v
                         confirm analysis and integrity checks
                                      |
                                      v
                           corrected statistics audit
                                      |
                                      +--> paper table
                                      +--> deterministic figures
                                      v
                              manuscript and final audit
~~~

The historical verifier checks plan, pairing, hashes, and result structure, but its recorded pass reproduced the old H1/H2/H7 clustering defect. The corrected statistics audit therefore has authority for manuscript-facing H1/H2/H7 values.

## What the data flow does not prove

- The frozen V4 dataset supplies an uncorrupted, noiseless simulated current channel with no quantization to the auxiliary and EKF witnesses.
- A witness disagreement is not a causal diagnosis of sensor versus plant failure.
- A debounced entry rate is not equivalent to conditional post-fault detection.
- The broad umbrella count is a plan scope, not an execution count.

Related: [ARCHITECTURE.md](ARCHITECTURE.md), [V4_CONFIRMATORY.md](V4_CONFIRMATORY.md), [STATISTICS.md](STATISTICS.md).
