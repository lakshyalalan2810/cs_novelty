# Research timeline

The timeline below separates scientific decisions from later infrastructure or reporting corrections.

~~~text
V1 abrupt-fault benefit
        |
        v
V2 auxiliary recovery gate -> C2 identical, closed
        |
        v
V3 dual arbitration -> strongest frozen branch, load confounding found
        |
        +--> C4 attribution -> NO_GO
        |
        +--> EKF comparator -> BASELINE_ONLY
        |
        +--> seed study -> TRAINING-SEED-SENSITIVE
        |
        +--> severity study -> no supported operating point
        v
V4 preregistration -> preexecution pairing correction
        |
        v
12,200-cell core -> EKF integrity repair -> statistics correction
        |
        v
H8/H9 only -> bounded paper claim
~~~

## Stage ledger

| Stage | Purpose | Evidence | Decision |
|---|---|---|---|
| V1 | Establish reliability-aware response to abrupt sensor faults | 4 controllers x 11 scenarios x 5 seeds = 220 final runs | Keep as bounded isolated-fault result |
| V2/C2 | Test auxiliary agreement as recovery gate | 10,251 recovery opportunities; C1/C2 exact equality in 55 paired cases | Close negative ablation |
| V3/C3 | Use dual virtual-sensor arbitration during substitution | 5 controllers x 11 scenarios x 5 seeds = 275 runs | Freeze strongest architecture with limitations |
| C4 | Attribute sensor/main/aux mismatch using three distances | 105 development runs | NO_GO; no final holdout |
| EKF | Compare a classical current/voltage observer | 100 development closed-loop rows plus estimator studies | BASELINE_ONLY |
| Training-seed study | Test learned-weight sensitivity | 150 runs, seeds 2026–2028 | TRAINING-SEED-SENSITIVE |
| Severity study | Map bias/dropout/drift/load/current boundaries | Part A 150 rows; Part B 735 labeled rows | Complete; no supported operating point |
| V4 preregistration | Freeze witness-gating questions and multiplicity | Protocol, seeds, endpoint rules, umbrella scope | Frozen before confirmatory execution |
| Preexecution amendment | Repair valid pairing and add H6 B anchors | H1–H5/H7–H9 use C3 seeds 2026–2028; H6 uses all 11 V4 seeds | Administrative/pre-run correction |
| Administrative re-freeze | Remove generated timestamp drift | 400-cell partial attempt discarded; 12,200/12,200 reconstructed | No scientific effect |
| EKF integrity repair | Fix uninitialized V4 witness path | All and only 2,700 affected cells rerun | Outcome-independent infrastructure repair |
| V4 confirmatory core | Execute H1–H9 dependency set | 12,200 exact cells | H8/H9 only supported |
| Statistical correction | Restore simulation-seed cluster hierarchy | H1/H2/H7 corrected, no simulations rerun | Corrected summaries supersede historical |
| Paper integration | Align text, tables, figures, provenance | Final paper audit: 87 tests / 13,746 subtests | Ready for human read-through; compile unverified |

## Supersession rules

- Early V1 timing/configuration claims are superseded by the corrected H20/(5,15) evidence.
- Early positive C2 interpretation is superseded by the later forensic C1/C2 equality result.
- Single-seed V3 combined-fault evidence remains historical but is narrowed by the training-seed study.
- The old V4 H1/H2/H7 summary is historical; corrected hierarchical bootstrap values are authoritative.
- The 268,910-cell umbrella remains a preregistered scope, not a completed-run count.

Related: [EXPERIMENTS.md](EXPERIMENTS.md), [PROVENANCE.md](PROVENANCE.md), [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md).

