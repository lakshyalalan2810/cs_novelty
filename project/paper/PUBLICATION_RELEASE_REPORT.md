> Packaging update, 2026-10-08: the unchanged evidence ZIP has been independently
> rechecked and is READY_FOR_DEPOSIT / LOCAL_ONLY. Current IEEE Access draft,
> author metadata gate and final verdict are in SUBMISSION_PACKAGE_REPORT.md;
> exact deposit metadata/procedure are in release/DEPOSIT_INSTRUCTIONS.md.
> The preceding release audit below remains historical evidence.

# Publication release and metadata report

Audit date: 2026-10-08. Scientific baseline: `4b7c51dabff16a702d230f2b9f87c14f73ebf9fd` (`Cleanup`). No scientific simulations, retraining, endpoint changes, or later research reconstruction were performed for this archive audit.

## Reproducibility verdict

**REPRODUCIBILITY_RELEASE: PASS_WITH_ACTIONS.** The original ignored artifacts have been located, restored locally, and packaged with exact baseline source and detached SHA-256 checksums. Public availability remains an action: no deposit, DOI, or public archive link exists. A clean source clone can build the saved paper assets and regenerate the tables; it cannot regenerate the load false-entry figure or reconstruct raw confirmatory inference without the separate artifact bundle. Independent bitwise scientific re-execution remains unestablished because the frozen environment record is partial.

The required ignored dependency map is in `../REPRODUCIBILITY.md`. The archive policy and restoration source are in `../ARCHIVAL_ARTIFACTS.md`; exact per-file availability, sizes, hashes, and integrity classifications are in `release/v4-artifact-inventory.json`. The ZIP preserves historical and corrected reports separately. It includes no later study, unavailable original-plan reconstruction, discarded initial attempt, or invented evidence.

## Archive checks and local outputs

- All nine canonical run/event/checkpoint files match `results/v4/confirmatory/result_manifest.json` exactly; the dataset matches its frozen preexecution SHA-256.
- All 22 model binaries and 22 histories are present. Associated model configurations, metrics, and training summary match baseline JSON semantics. Their new standalone hashes establish the prepared release inventory; the baseline did not contain prior standalone model/history hashes.
- The 507-byte final runtime snapshot matches its frozen hash and is preserved as provenance rather than an execution instruction.
- Frozen source/preexecution/result/corrected-summary bindings were checked. Seven working text files had only CRLF checkout differences; all exact baseline Git blobs match their expected hashes. The ZIP uses Git blobs and therefore preserves canonical LF bytes.
- The archive builder checks sorted membership, deterministic repeat bytes, CRC integrity, and every extracted entry's SHA-256. It uses only the Python standard library, Git, and the original saved files.
- Final independent verification found Windows text conversion in the initial local `git archive` source packaging. The corrected builder reads every baseline source file directly with `git show`; every source member in the rebuilt ZIP was independently checked against fresh baseline Git-blob reads. This correction changes packaging only and preserves scientific artifacts and inference.

Prepared outputs are `release/v4-frozen-evidence.zip` (local ignored binary), `release/v4-frozen-evidence.zip.sha256`, `release/v4-artifact-inventory.json`, and `release/SHA256SUMS.txt`. The scientific source inside the ZIP is the frozen baseline; the final edited manuscript/PDF must accompany the publication separately after review. Deposit the ZIP, detached checksum, inventory, source commit identifier, and final manuscript/supplement through the chosen archive. Verify the uploaded bytes before adding a persistent archive citation.

| Local bundle measure | Verified value |
|---|---:|
| Restored scientific artifacts | 54 files; 35,192,196 bytes |
| Preserved final runtime snapshot | 1 file; 507 bytes |
| Source files inside bundle | 608 exact baseline Git blobs |
| ZIP entries including internal inventory/checksums/README | 666 |
| ZIP size | 50,309,924 bytes (about 48.0 MiB) |

ZIP SHA-256: `96f390e72a72b11b1c6ca106f18b6ff463ca894c49c3a05cc9202ab70b626743`. The detached `.sha256` file is the copy-safe checksum authority for the local ZIP.

Rebuild/self-check command from the repository root:

```powershell
python project/paper/release/prepare_v4_archive.py 'C:\Users\Lakshya\OneDrive\Desktop\antenna\cs_novelty\project'
python -m py_compile project/paper/release/prepare_v4_archive.py
```

The local preparation used `C:\Users\Lakshya\anaconda3\python.exe` for these standard-library-only operations. Scientific tests/build/layout results are reported separately in `../PAPER_FINAL_AUDIT.md`; preparing an archive is not a claim that every scientific test, compilation, or visual check passed.

## Human-supplied publication metadata

Known names and affiliation supplied by the authors' task: **Lakshya Lalan**, **Shashwat Kansal**, **Vellore Institute of Technology, Vellore, India**. All remaining fields below require author information or a selected venue's requirements; none has been invented.

| Item | Required action |
|---|---|
| Author order and affiliation rendering | Authors confirm final order and the exact institutional form |
| Corresponding author | Identify the author responsible for correspondence |
| Email addresses | Supply author/corresponding-author emails if the venue requires them |
| ORCID IDs | Supply only actual IDs; check whether the target venue requires them |
| Funding / grants | Supply an accurate funding statement or author-confirmed absence of funding |
| Acknowledgments | Supply text and permission where applicable; do not create placeholder claims |
| Conflict / competing-interest declaration | Authors supply the actual declaration required by the venue |
| Author contributions | Authors supply real assignments if required; no role allocation has been inferred |
| Ethics statement | Determine venue applicability; do not invent approval or exemption |
| Data availability | Deposit the bundle and provide its real persistent identifier; until then state local preparation accurately |
| Code availability | Cite the source repository and approved immutable release commit; the scientific baseline is identified above |
| Artifact license | Confirm the license for data/models and deposit assets; the repository's code license alone does not document that decision |
| Supplementary-material statement | Specify the actual ZIP/inventory/source package and final accepted supplement allowed by the venue |
| AI assistance disclosure and author biographies | Authors verify the actual assistance disclosure and supply biographies required by the recommended IEEE Access target; see `VENUE_READINESS.md` |
| Venue/template and submission metadata | Apply the selected venue's requirements after venue assessment; no submission ID exists |

## Remaining release actions

1. Complete author declarations and venue-dependent metadata.
2. Review the final source/PDF/diff and approve the final publication revision before any commit or push.
3. Deposit the prepared scientific archive, complete deposit metadata/licensing, and verify the uploaded bytes.
4. Add only the actual archive DOI/URL and final approved source identifier to availability statements.

The archive audit skipped fresh scientific experiments and independent bitwise reruns. The remaining risk is reader access: the verified local bundle is not a public deposit until the authors publish it.
