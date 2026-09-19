# C4 verifier evidence-infrastructure repair

The frozen C4-v1 development evidence remains scientifically unchanged and
retains its preregistered **NO-GO** result.

The repaired defect was a schema mismatch in `scripts/verify_c4_results.py`.
The saved C4 development matrix uses the evaluator's canonical columns
`attr_frac_likely_sensor_fault`,
`attr_frac_likely_plant_or_main_mismatch`, and
`attr_frac_likely_aux_mismatch`, while the verifier's state-fraction alias map
previously looked only for older forms that omitted `likely_`.

The verifier now resolves the canonical `likely_*` names first. The older
names remain accepted as backward-compatible aliases because they were
already part of the verifier's documented compatibility surface. No metric,
threshold, persistence rule, attribution logic, controller behavior, or
stage-gate arithmetic changed.

The repair necessarily changes the verifier source hash. The frozen
development configuration, development summary, and saved stage-gate record
correctly bind the verifier that existed when the development evidence was
created, SHA256
`294bfec07f45e5570e5a16401b5a372d4cf6793feca864f37d0ee8dc96f8424d`.
The repaired verifier therefore validates that historical self-hash as
provenance instead of rewriting any frozen C4 scientific artifact to contain
the post-evidence verifier hash.

Regression coverage loads the **actual saved development column set**, checks
that all canonical state fractions resolve, and independently recomputes the
saved development stage gate as `NO_GO`.

The canonical 105-row C4 development matrix remains byte-identical at SHA256
`108f7875ec8af675a4aa2dce5bec833735ff499be6ead588d3accce32efe3a15`.
The frozen V3 275-run matrix remains byte-identical at SHA256
`a919e0b16321765fb1be5f97c15dc61062ccea737cd4059aed887ca6aa647086`.

## Filtered-disagreement diagnostic contract repair

Full-verifier execution after the alias repair exposed a second, pre-existing
verifier mismatch. The saved development event artifact contains exactly 13
episodes where all three diagnostic fields `filtered_d_sm`, `filtered_d_sa`,
and `filtered_d_ma` are NaN. The deployed attributor intentionally clears this
diagnostic EWMA triplet whenever the auxiliary witness becomes unavailable;
raw pairwise disagreements continue to drive classification and veto logic.

All 13 historical rows have a complete raw disagreement triplet, persisted and
candidate attribution state `AMBIGUOUS`, zero auxiliary trusted fraction, and
an explicit unavailable-witness reason. Twelve record
`INSUFFICIENT_TRUSTED_HISTORY`; one records `AUX_PARAM_MISMATCH`. No saved row
contains a partially missing filtered triplet.

The verifier now accepts a missing filtered triplet only when all three values
are missing together and the same row proves the unavailable-witness state:
persisted and candidate state are `AMBIGUOUS`, auxiliary trusted fraction is
zero, and the reason is one of the evaluator's explicit unavailable-witness
states (`STARTUP_BLANKING`, `AUX_NONFINITE`, `AUX_OUT_OF_BOUNDS`,
`AUX_PARAM_MISMATCH`, `INSUFFICIENT_TRUSTED_HISTORY`, or
`AUX_EWMA_INCONSISTENT`). Present filtered values must still be numeric, finite,
and nonnegative. Raw `d_sm`, `d_sa`, and `d_ma`, event duration, flags, and the
rest of the decision-critical event contract remain strict.

Regression coverage loads the actual saved development event artifact, checks
the 13 historical unavailable-witness NaN triplets, verifies that the artifact
passes, and confirms that a NaN introduced into a READY row or a missing triplet
relabeled READY fails verification. This repair changes verifier behavior only;
the saved development evidence is unchanged and its independently recomputed
stage-gate result remains `NO_GO`.

Post-repair validation passed 40 C4 attribution/workflow regression tests. The
development-only verifier now completes with `C4 DEVELOPMENT VERIFIER RESULT:
PASS` while independently recomputing `C4 development stage gate: NO_GO`.
