# Provenance

## Execution chain

The V4 provenance chain is:

~~~text
historical plan
  7e409aef...
       |
       v
administrative re-freeze
  fb3e2e614...
       |
       v
core runner / preexecution manifest
       |
       v
400-cell partial attempt -> discarded
       |
       v
clean 12,200-cell restart
       |
       +--> 2,700 EKF cells repaired from deterministic init defect
       |
       +--> corrected H1/H2/H7 bootstrap, no simulations rerun
       v
corrected summary + artifact bindings
~~~

Scope accounting is recorded as 11,650 original-umbrella cells used by the core plus 550 supplemental H6 B-anchor cells, for 12,200 executed core cells; 257,260 original-umbrella cells remain deferred. This explains why the manifest’s deferred field is not 268,910 minus 12,200.

## Key hashes

| Object | SHA-256 |
|---|---|
| Historical original plan | 7e409aef24c1472382b19964927c878f1daf44f1564d23b6ef1e10f07c35b75f |
| Administratively re-frozen plan | fb3e2e614b2b35dd3ae66be3c821d389b59573291140ab2a2f906a2912166ae5 |
| Core runner | f7ee31ae6f266aacf6a7ecc28de9e6ef2cfaac8417bbdf4e1b909b9d1632a07f |
| Refreeze note | 170038598acdcce2a29e2d7b98da71e1e734090a9263386daac304ef83582006 |
| Preexecution manifest | d19370e61d36d70b9b94d6a196a423a7b1da9477fd21142b27ae08afe2b680a2 |
| EKF repair incident | 8766c62f2e8a8cd75db494de160086874f6e6b05274d5ea7bba5eb01019a9214 |
| Corrected H1–H9 CSV | d992473ce809f8736fa366e1e2b160b9f316f39d180391babcb1ad07f63d44f8 |
| Corrected final summary | 00dc0814722f6ecdc9b97ca9bbab7031c86dc831629c1be6fe6f5c5f383a01ba |
| Statistical correction record | 52cb4b03e03ff43b7e20ce13b766a3c29b2c24610263acc95629afdcfb89ffb2 |
| Archival Git-blob hash of result_manifest.json | 0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e |
| Historical Windows CRLF checkout hash | 58c2ece8410574de388b0933e17a0b8295e65834f74858edf7d21956e9ce3ff8 |

## Repair provenance

The old checkpoint provenance is 7979c98f08431d51c9828bc282f706893abee75de16dc9483d948c16e6a8eed1. The repaired checkpoint provenance is 39f1f08802baea34024ad3a323ae0b4a0c4095f13eb8a8e4752906073d2e6a50. The repaired source hash for project/scripts/v4_closed_loop.py is 9545883fc071ff420ed6667906ac31ebeba5c3fc1e337b61c7044f7df6b7ca20. The repair script hash is 3aa9c05ba4d0da3afe54f08d98daa86d9f16919d25517dd6165f4c9869d4595d.

The repair was all-and-only: 2,700 planned V4_full_ekf cells selected by identity, checkpoint provenance, and deterministic failure signature. It was not selected by result magnitude or statistical outcome.

## Statistical correction provenance

The historical source project/scripts/v4_confirm_analysis.py has hash f5a5bea644d7a5a5a865bc378783f826706f8ac7af8a4c09516f4b0dd7f7da64 and remains historical. The corrected audit source project/scripts/v4_statistics_audit.py has hash 40839b57c75f059f4b97bb081e0ffdbbe793a7db2237b4db6143fce8c5ab7756. It changed the hierarchical resampling implementation only; simulations rerun = 0.

## Environment and working state

The V4 manifest records Python 3.11.15 (Anaconda), NumPy 2.2.6, pandas 2.3.3, SciPy 1.13.1, and PyTorch 2.6.0+cu124. `project/requirements-frozen-v4.txt` preserves that partial archival record; `project/requirements.txt` is the supported reconstruction/development stack. A fresh environment must be treated as a reconstruction attempt, not assumed bitwise identical.

The final-cleanup baseline HEAD observed was `2a6933d8ddaa7d45d9175f7ce0558fbd52a454e4`. The tracked historical review `project/PROJECT_OVERVIEW_AND_PUBLISHABILITY.md` was preserved and labeled.

## Figure and paper provenance

project/paper/FIGURE_MANIFEST.md binds six deterministic vector PDFs to their source artifacts and records that the generator runs no scientific simulation. Figure 5 uses the corrected H1–H9 table; Figure 6 uses the load-0.15 sweep rows with the clean-start aggregation rule. No favorable representative trajectory was selected because the frozen core lacks a manuscript-ready common-condition trace.

project/PAPER_FINAL_AUDIT.md records the integrated manuscript audit, 87 tests / 13,746 subtests, figure/table/reference checks, and readiness for final human read-through. That Oct-1 record is preserved as historical. The 2026-10-08 finalization audit supersedes its build status with real Tectonic/XeTeX + BibTeX compilation, complete visual QA, and local archive preparation; no scientific result was rerun or changed.

The paper and test suite correctly cite `0591d54f…`, the committed LF-byte manifest hash. The former `58c2ece8…` filesystem hash was a CRLF checkout representation created by Windows `core.autocrlf=true`; `.gitattributes` now preserves the archival bytes. No scientific or metadata field differed.

Related: [V4_CONFIRMATORY.md](V4_CONFIRMATORY.md), [RESULTS_AUTHORITY.md](RESULTS_AUTHORITY.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
