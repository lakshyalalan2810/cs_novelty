# Independent LaTeX and visual QA â€” 2026-10-08

This report audits the preserved Oct-1 manuscript and the final revised PDF. The lead agent owns manuscript content, figures, and table generation. After a specific follow-up authorization, this reviewer added only two final typography controls: a 2-point double-float separator and the standard IEEE reference-column trigger at reference 6. No table value or scientific result was edited. Scientific values and schematic corrections were cross-checked against `SCIENCE_CONTENT_AUDIT.md` and `FIGURE_MANIFEST.md`. No simulation, retraining, or statistical resampling was run.

## Preserved baseline

Build input: `tmp/paper_source_before/main.tex`. Build output: `tmp/paper_build_before/main.pdf`; final-pass log: `tmp/paper_build_before/main.log`; BibTeX transcript: `tmp/paper_build_before/main.blg`. The lead agent compiled with portable Tectonic 0.17.0, whose build invokes real XeTeX, BibTeX, and xdvipdfmx and repeats passes. `pdfinfo` reports **8 letter-size pages**, PDF 1.5. The task's approximately seven-page estimate differs from this independent toolchain's measured output.

Rendering command from the repository root:

```powershell
pdftoppm -r 120 -png tmp/paper_build_before/main.pdf tmp/paper_render_before/page
pdfinfo tmp/paper_build_before/main.pdf
```

The reviewer opened **all eight full-page PNGs individually** using the image-view tool. Rendering success and compilation exit status alone were not treated as a layout pass.

| Baseline page | Visual inspection |
|---|---|
| 1 | Title, named authors and institution, abstract, introduction, contributions, start of Related Work. No collision or clipping. Body/title font substitution is visible; the configured IEEE Times shapes are unavailable under the baseline Unicode font encoding. |
| 2 | Figure 1, Related Work continuation, plant equations, control law, learned predictors, V3 detector. Equation labels and symbols are legible. Figure 1 has readable labels, modest color contrast, and ordered arrows that retain meaning in grayscale. No collision. |
| 3 | Figure 2, selector/V4 methods, Figure 3 entry-time strip plot. No typographic collision. Figure 2 incorrectly feeds a main residual into the witness in series and labels the main history bus as current-consuming; this is a scientific schematic defect flagged by the independent science reviewer. Figure 3 distinguishes crosses and hollow circles without depending on color. |
| 4 | Classical comparators, dataset/calibration/protocol, hypothesis definitions, inferential methodology and seed populations. No collision or missing glyph. Source descriptions need the separately documented content corrections. |
| 5 | **FAIL:** three complete SHA-256 strings intrude into the right-column results. **FAIL:** Table I is wider than its single-column wrapper, and its final conditional-entry columns are visibly clipped at the right-column boundary. Figure 4 itself has no collision, but its long two-column legend widens the source asset; scaling to 3.5 inches reduces nominal 8-point labels to approximately **6.37 points**. |
| 6 | Tables II/III remain legible across both columns. **FAIL:** Table IV crosses the gutter and overlays discussion text throughout multiple rows. It obscures both numbers and prose. |
| 7 | Figure 5 intervals, Figure 6 seed plot, limitations and conclusion. **FAIL:** Figure 5's rightmost tracking-penalty axis label is clipped inside the graphic, a defect not detectable from the LaTeX overfull warnings. Figure 6 has gray bars, hatched EKF series, and a legible legend; zero-valued V4 outcomes are described in its caption. No prose collision. |
| 8 | Four baseline references; generous unused space. Entries are readable and have no collision. The independent literature audit owns coverage and metadata verification. |

### Every baseline overfull occurrence

All **11** occurrences in the final baseline log are enumerated below; no occurrence was inferred only from an approximate earlier build.

| Log line | Source lines reported by TeX | Excess width (pt) | Diagnosis |
|---:|---|---:|---|
| 451 | 95â€“99 | 1.45346 | Figure 1 uses 7.16 inches rather than the exact IEEE text width; visually clean. |
| 473 | 153â€“159 | 1.45346 | Figure 2 full-width wrapper; visually clean. |
| 481 | 213â€“217 | 0.94499 | Figure 3 single-column wrapper; visually clean. |
| 531 | 380â€“392 | 56.07999 | Historical plan hash; visible cross-column overflow. |
| 542 | 380â€“392 | 68.57999 | Administrative re-freeze hash; visible cross-column overflow. |
| 558 | 380â€“392 | 65.51999 | Superseding result-manifest hash; visible cross-column overflow. |
| 584 | table line 4â€“manuscript line 407 | 64.15208 | Table I, per-seed post-hoc table; final columns clipped. |
| 591 | 411â€“414 | 0.94499 | Figure 4 wrapper; visually clean perimeter but undersized labels. |
| 599 | 452â€“458 | 1.45346 | Figure 5 wrapper; separate graphic-internal label clipping. |
| 606 | 468â€“473 | 0.94499 | Figure 6 wrapper; visually clean. |
| 611 | table line 4â€“manuscript line 487 | 144.39207 | Table IV, frozen Part B table; severe overlap with adjacent prose. |

The baseline final pass has **19 underfull warnings**: 17 horizontal (lines 112â€“119; three at 243â€“248; 271â€“277; 284â€“292; eleven at 380â€“392) and two vertical page-output warnings. The final pass reports **zero undefined citations/references**, no missing graphic, no PDF-generation error, and **28 font-warning records**, including Times bold/small-caps and Courier fallback plus the final aggregate warning. The repeated Times/Courier warnings are font substitutions, not unresolved bibliography entries. `main.blg` confirms BibTeX 0.99d and IEEEtran.bst 1.14.

## Locked final PDF review

Reviewed artifact: `project/paper/main.pdf`, **172,857 bytes**, SHA-256 **`15b22cb06d2cffc77f04b1cc2eff2015b6ca3e65d7219780920dd55511512e24`**. `pdfinfo` reports **9 letter-size pages**, PDF 1.5, produced by xdvipdfmx. The final source retains IEEEtran conference format. All manuscript edits, including first-use acronym expansions, precede this locked build and review.

The successful build command, run from `project/paper`, was:

```powershell
$env:FONTCONFIG_FILE=(Resolve-Path '../../tmp/fonts.conf').Path
& '../../tmp/paper_tools/tectonic.exe' --keep-logs --keep-intermediates --outdir '../../tmp/paper_build_final' main.tex 2>&1 | Tee-Object -FilePath '../../tmp/paper_build_final/final_console.txt'
```

Portable **Tectonic 0.17.0** invokes real XeTeX, **BibTeX 0.99d** with **IEEEtran.bst 1.14**, and xdvipdfmx. The final transcript records the initial TeX pass, BibTeX, and two further TeX passes until the auxiliary/reference state converges. Final-pass diagnostics are `tmp/paper_build_final/main.log`; bibliography diagnostics are `main.blg` beside it. The compiler output was copied to `project/paper/main.pdf` only after success.

Commands run from the repository root:

```powershell
Copy-Item -LiteralPath 'tmp/paper_build_final/main.pdf' -Destination 'project/paper/main.pdf'
pdftoppm -r 140 -png project/paper/main.pdf tmp/paper_render_final/page
pdftoppm -gray -r 110 -png project/paper/main.pdf tmp/paper_render_final_grayscale/page
pdfinfo project/paper/main.pdf
& 'C:/Users/Lakshya/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' tmp/final_pdf_qa.py
```

The reviewer individually opened **every locked final full-page color PNG (1–9)**, then separately opened grayscale pages **2, 4, 6, 7, and 8**, covering **all six figures**. Both render sets contain all nine pages. Individual revised architecture, conditional-detection, and hypothesis-effects assets were additionally rendered and inspected. Every final build candidate underwent renewed rendering; this locked verdict refers only to the exact hash above.

| Final page | Visual result and evidence |
|---|---|
| 1 | Times-compatible title and author/institution blocks, bold abstract, introduction, contributions and start of expanded Related Work. Long title is balanced over three lines. Expanded acronym definitions fit. No collision, missing glyph, or clipping. Mathematical load units render correctly. |
| 2 | Figure 1, expanded related work and start of Method. All six evolution boxes, arrowheads, branch verdicts, citations, plant equations and equation numbers are legible. Grayscale preserves meaning because boxes are named and ordered. |
| 3 | Learned predictors, detector, enabled V4 changes, healthy-current assumption and comparators. Mathematical residuals and fallback-source identifiers fit the columns. Expanded MAE definition is readable. No collision. |
| 4 | **Figure 2 schematic repaired:** selected speed feedback and voltage reach the main LSTM; voltage and healthy current reach the witness; separate main/witness residual paths reach the monitor, without the erroneous serial residual. No ground-truth input appears. Caption distinguishes debounced AND entry from instantaneous substitution. Figure 3's crosses/hollow circles and onset lines remain distinct in grayscale. Protocol text, caption and trailing comparator paragraph are readable; ragged-bottom spacing removes the earlier large gap. |
| 5 | Hypotheses, exact executed-core accounting, durations/noise scale, bootstrap methodology, seed populations, provenance and archive boundary. **Hash-overflow prose replaced by a reference to Table I**; neither column intrudes into the other. Source-only versus archived-artifact reproducibility is clear and legible. |
| 6 | **Table I:** all three complete 64-character hashes fit the full-width provenance table and extract unchanged. **Table II:** wrapped headers and all three complete seed rows fit a readable single column at approximately **7.97 points**, including conditional post-event entry; no scaling or clipping. Figure 4's revised legend fits at approximately **8.07-point** nominal legend/tick text. Hatched versus solid bars survive grayscale. Caption defines aggregation and denominators. Results/discussion text has no collision. |
| 7 | Table III's 16 condition rows and Table IV's complete nine-hypothesis family fit with clear captions and column boundaries. Table IV body is approximately **8.86 points**; Table III uses approximately **7.97 points**. **Figure 5 repaired:** all nine hypotheses and every full axis label, including tracking-penalty units, are visible. Hollow squares, filled circles and asterisks preserve the Holm distinction in grayscale. The page intentionally contains only these three floats. The 2-point double-float separator yields visible whitespace from captions/asset margins, with no collision or clipping and no overfull vertical box. |
| 8 | **Figure 6 now precedes the conclusion:** centered at the original readable single-column graphic width inside a double-column float. Seeds, probability scale and series legend are legible; caption defines the 124 pairs, denominators 43/34/47 and zero-entry V4 outcomes. **Table V repaired:** all 16 frozen rows and Wilson intervals fit a full-width wrapper at approximately **7.97 points**, without destructive scaling. Grayscale preserves series labels/hatching. Discussion/limitations continue below without collision. |
| 9 | Conclusion followed by all 16 references, balanced across both columns using IEEEtran's built-in trigger before reference 6. No remaining float occurs after the conclusion or in the bibliography. Both columns end at similar heights; the earlier empty final right column and heading-to-paragraph gap are repaired. Entries and final online source link are readable, with no missing characters or clipping. |

### Final diagnostics and independent checks

- **0 overfull horizontal boxes and 0 overfull vertical boxes**, versus 11 horizontal overflows in the preserved baseline. Every significant baseline overflow and graphic-internal clipping defect has a visually verified correction.
- **0 unresolved citations; 0 unresolved cross-references; 0 missing graphics/tables; 0 PDF-generation errors.** BibTeX completes without missing-entry diagnostics. Six figures and Tables I–V appear with complete captions. Prose references agree with table numbering after insertion of the provenance table.
- **13 underfull horizontal warnings; 0 underfull vertical warnings.** Source ranges: 74–76, 76–78, 80–82, 331–340, 365–370, 417–429, twice at 433–438, three at 460–469, and twice at 613–619. Every affected page was visually inspected. These are ordinary justified-column spacing warnings, with zero collision or clipped material.
- **0 font warnings.** Loading the T1 encoding before the IEEE class removes the earlier Unicode Times startup warnings. Actual body/title use embedded Times-compatible **Nimbus Roman**; hashes/path text use embedded Courier-compatible **Nimbus Mono**.
- The final log repeats the informational layout warning **"Text page 7 contains only floats"** twice, once per column output. Page 7 is intentionally a complete results table/figure page; it has been visually inspected and fits within the page with no overfull box. This is disclosed, not treated as a missing-text error.
- Independent recursive `pypdf` inspection of page/form resources found **19 unique embedded fonts**: 16 Type 1 plus three Type 0 wrappers with TrueType descendants, **zero Type 3**, **zero unembedded fonts**. Assert-based checks verify these conditions, no undefined citation/reference, no overfull box and no font warning. Machine-readable output: `tmp/paper_render_final/font_audit.json`; read-only check script: `tmp/final_pdf_qa.py`.
- `pdfplumber` extracts all three provenance hashes unchanged, including every character. Numeric table provenance and figure selection/aggregation were independently reconstructed by the science reviewer; typography changes retain those values and do not select a new trajectory.
- Figure text remains selectable vector content. Meaning does not depend solely on color: timeline markers, detection hatching, Holm squares/circles/asterisks, named evolution boxes and explicit legends remain interpretable in grayscale. The PDF is not structurally tagged; no PDF/UA certification is asserted.

## Locked verdicts

**LATEX_BUILD: PASS.** A real repeated-pass XeTeX/BibTeX build produces a valid nine-page PDF with resolved citations/references, all fonts embedded, complete assets, zero font warnings and zero overfull boxes.

**VISUAL_LAYOUT: PASS.** Every locked final color page and every figure in grayscale were inspected. There are **zero text/table collisions and zero clipped labels**. Full provenance hashes, both formerly oversized tables, small Figure 4 labels, the clipped Figure 5 axis label, Figure 6 placement and final-column balance are repaired. A later rebuilt PDF requires a renewed full-page visual pass before carrying forward these verdicts.

No science, endpoint, numeric result, or source trajectory was regenerated. No printed hardcopy or venue PDF-validation portal was checked; venue-specific page limits and human metadata remain separate readiness actions.
