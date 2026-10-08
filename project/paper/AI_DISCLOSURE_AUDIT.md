# AI-use disclosure audit — 2026-10-08

Policy: [Access checklist](https://ieeeaccess.ieee.org/authors/submission-guidelines/),
[Access preparation](https://ieeeaccess.ieee.org/authors/preparing-your-article/),
[IEEE ethics](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/),
accessed 2026-10-08. Generated content requires disclosure identifying the system,
sections, and extent. The Access checklist also requires citations in affected
sections. Editing/grammar-only assistance has a recommended disclosure.

## Documented evidence and uncertainty

| Record | What is established | What is not established |
|---|---|---|
| LITERATURE_POSITIONING_AUDIT.md, "Proposed exact replacement Related Work" | An AI-workflow report contains substantial replacement Related Work prose integrated into the audited draft | Actual historical AI system/model and author editing extent |
| FINALIZATION_CHANGELOG.md; SCIENCE_CONTENT_AUDIT.md; FINAL_REVIEW_DIFF.patch | Methods, denominators, limitations, novelty wording and captions were revised in the preceding agent workflow; three figure assets/scripts received corrections | Passage-by-passage generation versus editing and full earlier project history |
| This packaging pass | OpenAI Codex converted the template and drafted front/back matter, availability, AI acknowledgment and metadata forms; retained scientific body/assets checked against pre-pass hashes | Prior tools, other contributors' use, author approval |

The earlier reports identify agent work, not a complete AI disclosure ledger.
No claim that all historical writing was human-only, that all AI use was merely
grammar, or that a particular earlier model was used is justified by these files.
No statement about AI executing the historical experiments or making scientific
decisions is inferred. This pass executed packaging checks and focused unit
tests, without new experiments, training or scientific simulations.

## Draft acknowledgment for author completion

> During manuscript preparation, [AUTHOR-CONFIRMED AI SYSTEM, version/date if
> known, stable system citation] was used to [generate/rewrite/edit] portions of
> [EXACT SECTIONS/PASSAGES], at the level of [SPECIFIC ASSISTANCE]. The authors
> [CONFIRM THEIR ACTUAL REVIEW/CORRECTION AND RESPONSIBILITY]. OpenAI Codex
> (https://openai.com/codex/) assisted with the IEEE Access template conversion
> and draft publication declarations. No experiments, model training, or
> scientific simulations were performed during this publication-packaging pass.

The completed statement belongs in Acknowledgment, before references. The
current TeX contains the documented current-pass statement and a conspicuous
human-action gate for prior use. It is not a complete historical disclosure.

HUMAN ACTION: identify the actual systems and materially generated/reworked
passages in Introduction, Related Work, Method, Experimental Protocol, Results,
Discussion, Limitations, Conclusion and captions as applicable. In each affected
section, add a concise scope note and a citation to the confirmed system (stable
URL or bibliographic system citation). The substantial documented Related Work
replacement especially requires resolution. Include figure/code assistance if
it falls within IEEE's generated-content policy. Do not cite Codex as the earlier
tool without confirming that history. A verified official Codex website entry
is added only to the submission bibliography for this documented current pass;
the original sixteen scholarly entries and review bibliography are unchanged.

The current-pass generated Data and Code Availability and Acknowledgment sections
identify and cite Codex, as does Supplementary Material. Editorial
templates/checklists are outside the scientific findings.
