# IEEE Access draft visual review — 2026-10-08

Final artifact: `submission_ieee_access/main.pdf`, ten pages. The PDF SHA-256
and page/font checks are recorded in IEEE_ACCESS_BUILD_REPORT.json.
All ten final pages were rendered with Poppler at 135 dpi and individually
inspected. The final images are `tmp/access_packaging/rendered/page-01.png`
through `page-10.png`; earlier unpadded renders are obsolete.

| Page | Inspected content | Result |
|---|---|---|
| 1 | Title, both authors and existing affiliation, abstract, seven index terms, introduction, contact/support placeholders | PASS; readable, no collision |
| 2 | Related work, motor equations and beginning of methods | PASS; headings, equations and running header clean |
| 3 | Evolution and architecture figures, method equations | PASS; both figures and captions readable, no clipping |
| 4 | Latch timeline, detector variants and classical baselines | PASS; labels, legend, caption and columns clean |
| 5 | Training/calibration, nine hypotheses, statistics and seed protocol | PASS; equations, list and paragraphs readable |
| 6 | Conditional detection plot, provenance prose, results and historical context | PASS; plot/caption clean; no filename overflow |
| 7 | Complete provenance hashes, hypothesis plot, load-entry plot, post-hoc seed table | PASS; all three 64-character hashes visible; axes, captions and table readable |
| 8 | Operating envelope, H1–H9 confirmatory table, frozen detection anchor | PASS; all wide tables contained within margins, values/captions readable |
| 9 | Discussion, limitations, conclusion, availability, supplementary material, AI acknowledgment, first references | PASS; floats precede Discussion; complete adverse/null findings and explicit human-action text readable |
| 10 | Remaining references, both biography placeholders, end marker | PASS; clean references and biographies; no overlap or blank page |

The final log retains 23 overfull hbox notices: two 9.2679 pt notices from the
official title/abstract/index layout and 21 505.12177 pt notices from the class's
zero-width footer overlays. The supplied class is unchanged. Inspection confirms
these do not clip or collide with content. Underfull notices describe spacing,
not missing content. No meaningful body/table/hash overflow, overfull vbox,
undefined reference/citation, missing asset, font substitution warning, or
unembedded/Type-3 font remains. Raw logs are retained; no warnings were hidden.

This is layout approval of a **draft**, not approval of missing personal facts.
Correspondence, funding, public DOI, earlier AI disclosure and biography fields
remain explicit. Authors must fill/approve them, rebuild matching source/PDF,
and repeat affected-page inspection before upload. IEEE-assigned publication
fields remain unassigned/pending. Photos and graphical abstract are acceptance
stage work.
