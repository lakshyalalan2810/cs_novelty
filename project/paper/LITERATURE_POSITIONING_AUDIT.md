# Literature and novelty positioning audit

Date: 2026-10-08. Baseline supplied by the lead audit: `4b7c51dabff16a702d230f2b9f87c14f73ebf9fd` (Oct-1 cleanup). This audit changes bibliography metadata and scholarly positioning only. It does not change controllers, hypotheses, endpoints, exclusions, thresholds, seeds, statistics, or conclusions. `main.tex` integration belongs to the lead agent.

## Outcome

The original file contained ten entries but the manuscript cited only four. All ten have relevant roles in the replacement Related Work passage below; none is retained merely to enlarge the list. Six focused primary papers are added, bringing the intended cited set to sixteen. There are no intended unused or duplicate entries after insertion. A citation-key check against the final manuscript must run after integration.

Two material errors were corrected:

- `Bonassi2022` described a 2021 paper, and its DOI ended in `.411`; the actual paper uses `.417`. The corrected key is `Bonassi2021`.
- `Gustafsson2000` had the bibliographic shell of a 2001 book review in *Measurement Science and Technology*, attributed to the author of the book. It is replaced by the actual Wiley book, with book type, publisher, 2000 print/copyright year, and book DOI.

Also added the missing Mayne DOI and the verified Holm archival URL, protected NARX/PMSM/DC/MPC/ISS/GRU capitalization, and encoded compound surnames explicitly. Article identifiers 4330 and 111381 occupy `pages` because the current IEEE BibTeX style renders that field; they are article numbers, not page ranges.

## Entry-by-entry metadata verification

Every name, title, venue, year, and page range/article identifier below was compared against the linked publisher record or the original article hosted by its author/institution. Initials are retained where the original gives initials; full names are not guessed. Sources are direct article/book records, rather than search-result links.

| Key | Verified bibliographic record and source | Disposition / citation role |
|---|---|---|
| `Hochreiter1997` | Sepp Hochreiter; Jürgen Schmidhuber. *Long Short-Term Memory*. Neural Computation 9(8), 1735–1780 (1997). DOI 10.1162/neco.1997.9.8.1735. [MIT Press](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory). | Retain; original sequence-model architecture, not control/fault guarantees. |
| `Page1954` | E. S. Page. *Continuous Inspection Schemes*. Biometrika 41(1–2), 100–115 (1954). DOI 10.1093/biomet/41.1-2.100. [Oxford issue record](https://academic.oup.com/biomet/issue/41/1-2?browseBy=volume). | Retain; origin of sequential cumulative change monitoring. |
| `Holm1979` | Sture Holm. *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics 6(2), 65–70 (1979). [Wiley journal issue archived by JSTOR](https://www.jstor.org/stable/i412579), [article](https://www.jstor.org/stable/4615733). No DOI was established; none is invented. | Retain; multiplicity method in Statistics. |
| `Mayne2000` | D. Q. Mayne; J. B. Rawlings; C. V. Rao; P. O. M. Scokaert. *Constrained Model Predictive Control: Stability and Optimality*. Automatica 36(6), 789–814 (2000). DOI 10.1016/S0005-1098(99)00214-9. [Elsevier](https://www.sciencedirect.com/science/article/pii/S0005109899002149). | Retain; constrained MPC and conditional nature of formal guarantees. |
| `Bonassi2021` | Fabio Bonassi; Marcello Farina; Riccardo Scattolini. *Stability of Discrete-Time Feed-Forward Neural Networks in NARX Configuration*. IFAC-PapersOnLine 54(7), 547–552 (2021). DOI 10.1016/j.ifacol.2021.08.417. [Elsevier](https://www.sciencedirect.com/science/article/pii/S2405896321011915), [author's citation record](https://bonassifabio.github.io/publications/). | Correct key/DOI; NARX dynamics and model stability conditions. |
| `Terzi2021` | Enrico Terzi; Fabio Bonassi; Marcello Farina; Riccardo Scattolini. *Learning Model Predictive Control with Long Short-Term Memory Networks*. International Journal of Robust and Nonlinear Control 31(18), 8877–8896 (2021). DOI 10.1002/rnc.5519. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/rnc.5519), [author-hosted manuscript](https://re.public.polimi.it/retrieve/50431575-9467-4fdd-a321-10b2c7ee9d45/11311-1169896_Bonassi.pdf). | Retain; closest LSTM-as-plant-model MPC, convergent observer and asymptotic stability under derived conditions. |
| `Isermann2006` | Rolf Isermann. *Fault-Diagnosis Systems: An Introduction from Fault Detection to Fault Tolerance*. Springer (2006), XVIII + 475 pages. DOI 10.1007/3-540-30368-5. [Springer](https://link.springer.com/book/10.1007/3-540-30368-5?page=2). | Retain; diagnosis taxonomy, virtual/analytical redundancy, and DC-drive examples. Copyright/ebook year is 2006; publisher lists the softcover release in 2005. |
| `Gustafsson2000` | Fredrik Gustafsson. *Adaptive Filtering and Change Detection*. John Wiley & Sons (2000 print/copyright). DOI 10.1002/0470841613; print ISBN 9780471492870. [Wiley book](https://onlinelibrary.wiley.com/doi/book/10.1002/0470841613). | Replace review with book; sequential monitoring and filtering context. Wiley's online platform date is 16 October 2001, distinct from print/copyright year. |
| `Venkatasubramanian2003` | Venkat Venkatasubramanian; Raghunathan Rengaswamy; Kewen Yin; Surya N. Kavuri. *A Review of Process Fault Detection and Diagnosis: Part I: Quantitative Model-Based Methods*. Computers & Chemical Engineering 27(3), 293–311 (2003). DOI 10.1016/S0098-1354(02)00160-6. [Elsevier](https://www.sciencedirect.com/science/article/pii/S0098135402001606). | Retain; quantitative model-based diagnosis taxonomy, not evidence for this motor result. |
| `Zhao2019` | Rui Zhao; Ruqiang Yan; Zhenghua Chen; Kezhi Mao; Peng Wang; Robert X. Gao. *Deep Learning and Its Applications to Machine Health Monitoring*. Mechanical Systems and Signal Processing 115, 213–237 (2019). DOI 10.1016/j.ymssp.2018.05.050. [Elsevier](https://www.sciencedirect.com/science/article/pii/S0888327018303108). | Retain; deep monitoring context, explicitly separated from the present closed-loop evaluation. |
| `Chow1984` | Edward Y. Chow; Alan S. Willsky. *Analytical Redundancy and the Design of Robust Failure Detection Systems*. IEEE Transactions on Automatic Control 29(7), 603–614 (1984). DOI 10.1109/TAC.1984.1103593. [Original IEEE article hosted by MIT author](https://willsky.lids.mit.edu/publ_pdfs/43_pub_IEEE.pdf), [NASA identifier record](https://ntrs.nasa.gov/citations/19840056958). | Add; primary separation of residual generation from decision-making and robust redundancy. |
| `Aguilera2016` | F. Aguilera; P. M. de la Barrera; C. H. De Angelo; D. R. Espinoza Trejo. *Current-Sensor Fault Detection and Isolation for Induction-Motor Drives Using a Geometric Approach*. Control Engineering Practice 53, 35–46 (2016). DOI 10.1016/j.conengprac.2016.04.014. [Elsevier](https://www.sciencedirect.com/science/article/pii/S0967066116300843), [original article in CONICET repository](https://ri.conicet.gov.ar/bitstream/handle/11336/179867/CONICET_Digital_Nro.4e8a5895-87c2-4de9-9bc3-07dd0ef10624_B.pdf?isAllowed=y&sequence=2). | Add; geometric load-torque decoupling and sensor isolability, a stronger claim than witness agreement. |
| `Choi2021` | Kyunghwan Choi; Yonghun Kim; Seok-Kyoon Kim; Kyung-Soo Kim. *Current and Position Sensor Fault Diagnosis Algorithm for PMSM Drives Based on Robust State Observer*. IEEE Transactions on Industrial Electronics 68(6), 5227–5236 (2021). DOI 10.1109/TIE.2020.2992977. [IEEE](https://ieeexplore.ieee.org/document/9091899/), [authors' laboratory](https://kaist-mic-lab.github.io/publications/2021-current-position/). | Add; full-state/disturbance observer and normalized residual motor diagnosis with physical validation. The issue year is 2021; early online publication was 2020. |
| `Chu2023` | Kenny Sau Kang Chu; Kuewwai Chew; Yoong Choon Chang. *Fault-Diagnosis and Fault-Recovery System of Hall Sensors in Brushless DC Motor Based on Neural Networks*. Sensors 23(9), article 4330 (2023). DOI 10.3390/s23094330. [Publisher](https://www.mdpi.com/1424-8220/23/9/4330), [publisher PDF](https://mdpi-res.com/d_attachment/sensors/sensors-23-04330/article_deploy/sensors-23-04330.pdf). | Add; primary CNN-LSTM Hall-sensor classification and recovery. Chew's printed PDF name is “Kuewwai”; some indexes split it as Kuew Wai. Use printed spelling. |
| `Salazar2017` | Jean C. Salazar; Philippe Weber; Fatiha Nejjari; Ramon Sarrate; Didier Theilliol. *System Reliability Aware Model Predictive Control Framework*. Reliability Engineering & System Safety 167, 663–672 (2017). DOI 10.1016/j.ress.2017.04.012. [Elsevier](https://www.sciencedirect.com/science/article/pii/S0951832017304416). | Add; actuator-use/system-reliability tradeoff and Dynamic Bayesian Networks, not measurement-channel confidence. |
| `Bonassi2024` | Fabio Bonassi; Alessio La Bella; Marcello Farina; Riccardo Scattolini. *Nonlinear MPC Design for Incrementally ISS Systems with Application to GRU Networks*. Automatica 159, article 111381 (2024). DOI 10.1016/j.automatica.2023.111381. [Elsevier](https://www.sciencedirect.com/science/article/pii/S0005109823005484). | Add; recent recurrent-model MPC with explicit horizon and convergence conditions; no inherited guarantees here. |

## Required literature areas and novelty boundary

| Requested area | Sources | Sentence-level purpose |
|---|---|---|
| 1. Learned/neural MPC | Terzi2021, Bonassi2024, Mayne2000 | Learned prediction inside MPC is established; assumptions behind guarantees differ. |
| 2. LSTM/NARX dynamical prediction | Hochreiter1997, Bonassi2021, Terzi2021 | Sequence models and autoregressive dynamics are reused building blocks. |
| 3. Fault detection/diagnosis | Isermann2006, Venkatasubramanian2003 | Detection, diagnosis, accommodation are distinct tasks. |
| 4. Observer residual generation | Chow1984, Choi2021, Aguilera2016 | Residual robustness and disturbance rejection precede this work. |
| 5. CUSUM/change detection | Page1954, Gustafsson2000 | A change alarm does not identify its cause. |
| 6. Analytical redundancy/virtual sensing | Chow1984, Isermann2006, Chu2023 | Estimator substitution and cross-channel corroboration are established concepts. |
| 7. Motor-drive sensor fault handling | Choi2021, Aguilera2016, Chu2023 | Nearby drive diagnoses have different motor/sensor models and validation. |
| 8. ML/deep monitoring | Zhao2019, Chu2023 | Classification/sequence recovery is different from closed-loop reliability-entry behavior. |
| 9. Reliability/FTC MPC | Salazar2017; Terzi2021 for the prediction/control side | Distinguish actuator-health reliability from a sensor-monitor state. |
| 10. Novelty boundary | All above; frozen C2/C4/EKF/V4 evidence | Contribution is a specific failure-driven entry gate and preregistered evaluation, not invention of the ingredients or first formal isolation. |

The paper reuses MPC, LSTM prediction, virtual sensing, EKF/Luenberger estimation, residual thresholds, CUSUM, and hysteretic substitution. Cross-channel corroboration and disturbance-robust monitoring are also established. Avoid “first”, “unique”, or universal superiority claims: this search is focused scholarly positioning, not exhaustive proof of priority.

The operational question worth studying is whether suppressing entry when a speed-independent predictor agrees with the measured speed avoids a harmful false substitution caused by main-model transient error. “Speed-independent” describes excluded inputs; it does not imply statistical independence or independence from plant disturbances. Both witnesses still assume a healthy current channel. H8/H9 establish an effect of the tested full V4 variants relative to C3; their identical retained-pair result is not an isolated causal estimate of the AND gate or proof that the two estimator classes are interchangeable.

C2 supplies evidence that an added estimator condition can be inactive. C4 supplies evidence that scalar agreement geometry does not itself establish causal attribution. EKF remains BASELINE_ONLY. H3/H4/H6/H7 unfavorable observations and H5 null require separate detection, recovery, tracking, and entry outcomes; they cannot be turned into positive fault-tolerance evidence by citing prior work.

## Proposed exact replacement Related Work

```latex
Constrained MPC and learned dynamical prediction are established building
blocks~\cite{Mayne2000,Hochreiter1997}. Terzi et al. use LSTM plant models
with a convergent observer and asymptotic closed-loop stability under
derived network and controller conditions~\cite{Terzi2021}. Neural NARX
models likewise admit sufficient stability conditions~\cite{Bonassi2021},
and recent recurrent-model MPC relates a minimum prediction horizon to
closed-loop stability~\cite{Bonassi2024}. Our frozen networks and switched
feedback logic were not designed to satisfy those conditions; one-step
prediction accuracy and simulation constraint counters do not establish
such guarantees.

Model-based fault diagnosis distinguishes residual generation, decision
making, and fault accommodation~\cite{Chow1984,Venkatasubramanian2003}.
Analytical redundancy uses modeled relations among available signals
instead of requiring duplicate physical sensors; observers and virtual
sensing are standard tools, including in DC-drive diagnosis~\cite{Isermann2006}.
The main/auxiliary estimates and EKF comparator reuse this principle.
Excluding measured speed from a witness prevents direct contamination by
that input, but does not make its error independent of plant disturbances
or current-channel corruption.

CUSUM supplies sequential evidence for a change in a monitored
signal~\cite{Page1954,Gustafsson2000}; it does not identify the physical
cause of that change. A learned prediction residual can therefore respond
to both a sensor fault and model error during a genuine load transient.
The present question concerns entry into feedback substitution, whose
consequences must be evaluated separately from the residual alarm itself.

Disturbance-robust sensor diagnosis is already studied in electric drives.
Aguilera et al. construct induction-motor observer subsystems decoupled
from load torque to obtain current-sensor fault detectability and
isolability~\cite{Aguilera2016}. Choi et al. combine full-state and
disturbance observers with normalized residuals for current/position-sensor
diagnosis in a PMSM drive, including physical validation~\cite{Choi2021}.
These works establish a stronger diagnosis setting than our empirical
witness agreement. Their different plants, sensors, and protocols also
preclude a numerical superiority comparison with this PMDC study.

Deep learning is widely used for machine-health monitoring~\cite{Zhao2019}.
Closer to sensor accommodation, Chu et al. use CNN-LSTM models for BLDC
Hall-sensor fault classification and signal recovery~\cite{Chu2023}.
Thus neural sensor reconstruction is not itself the contribution here.
Our evaluation instead addresses how a monitor's entry decision changes
closed-loop behavior when a healthy speed sensor is exposed to a load
disturbance.

Reliability-aware MPC also has an established actuator-health meaning:
Salazar et al. include actuator usage and system/component reliability in
MPC to balance reliability with control performance~\cite{Salazar2017}.
Here ``reliability'' denotes the monitor's operational confidence in a
measurement channel; no component lifetime or failure probability is
optimized. The contribution is a failure-driven witness-entry design and
its preregistered, multiplicity-controlled evaluation within the frozen
PMDC architecture. Only H8/H9 support reduced false entry at the tested
0.15~N$\cdot$m load condition under clean-start pairing. The inactive C2
recovery gate, C4 NO\_GO attribution branch, EKF tradeoffs, and unsupported
detection/recovery/tracking hypotheses bound that result. Witness gating
is therefore evaluated as entry discrimination, not formal fault
isolation, general fault tolerance, or hardware readiness.
```

## Verification limits

- Primary records and original-paper abstracts/available manuscripts support the specific sentences above. This is not a complete systematic review, exhaustive priority search, or independent replication of prior papers.
- Several publisher pages were readable in indexed publisher results while direct opening returned 403/internal errors (Elsevier, MDPI, IEEE). Author-hosted originals/laboratory records and journal issue records corroborated metadata where available; this is stated rather than treating inaccessible full text as read.
- Choi author spellings are corroborated by the authors' laboratory; IEEE supplies volume/issue/year/pages/DOI and abstract. Chow metadata uses the actual MIT-hosted IEEE paper and NASA DOI record.
- The original book-review author/DOI was not reconstructed: the irrelevant review record was removed from the bibliography rather than repaired as a retained citation.
- Focused 2025–2026 searches also surfaced current-sensor LSTM reconstruction and monitoring articles. They were not added solely for recency when their detailed bibliographic/technical comparison was less well verified or redundant with the selected primary work. The selected recent control comparison is the 2024 Automatica paper; do not call this an exhaustive account of all literature through 2026.
- No retraction notice appeared in inspected records; a comprehensive Crossmark/retraction-database audit was not performed. The final author submission check should include it where the selected venue requests it.
- No DOI has been assigned to Holm here, and no publication/archive DOI for this project is invented.

## Validation performed

PowerShell assert-based checks found 16 unique BibTeX keys and 15 distinct citation keys in the proposed Related Work block, all resolved. The remaining key, `Holm1979`, is cited in Statistics. After the lead agent integrated the passage, an independent check of actual `main.tex` reported **16 cited keys, zero missing keys, zero unused entries**. `git diff --check -- project/paper/references.bib project/paper/LITERATURE_POSITIONING_AUDIT.md project/paper/VENUE_READINESS.md` passed. Git's LF-to-CRLF advisory is a checkout-format notice, not a bibliography error.

The lead/layout audit owns BibTeX compilation and final PDF reference rendering. No extra runtime tests are needed for these bibliographic/document changes; no scientific execution was run.

Verdict after passage integration and citation checks: **LITERATURE_COVERAGE = PASS_WITH_ACTIONS**. Remaining actions are final author review of the scholarly comparisons and venue-specific reference rendering; broader scientific claims remain unsupported.
