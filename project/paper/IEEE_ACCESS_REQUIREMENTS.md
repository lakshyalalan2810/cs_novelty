# IEEE Access requirements checked 2026-10-08

Official sources, accessed 2026-10-08 (Asia/Calcutta):

- [S1: initial submission checklist](https://ieeeaccess.ieee.org/authors/submission-guidelines/)
- [S2: preparation and AI content policy](https://ieeeaccess.ieee.org/authors/preparing-your-article/)
- [S3: post-acceptance guide](https://ieeeaccess.ieee.org/authors/post-acceptance-guide/)
- [S4: IEEE author tools, ORCID, and data sharing](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/)
- [S5: IEEE publishing ethics](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/)
- [S6: supplementary materials](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/prepare-supplementary-materials/)
- [S7: current official LaTeX template](https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip)

## Initial submission

| Item | Requirement / guidance | Source | Package status |
|---|---|---|---|
| Format | Access template; two columns, single spaced | S1 | PASS |
| Files | Matching source and PDF; each <=40 MB | S1 | PASS |
| Length | No ordinary limit; recommend <20 pages; >20 needs inquiry unless exempt | S1 | PASS |
| Authors | All authors in source/PDF; verify portal extraction and consent/order | S1 | HUMAN_ACTION_REQUIRED |
| Correspondence | Designate contact in portal; template includes contact email | S1/S7 | HUMAN_ACTION_REQUIRED |
| ORCID | Submitting author's account: public, populated ORCID | S1 | HUMAN_ACTION_REQUIRED |
| Biographies | Every author; after references | S1 | HUMAN_ACTION_REQUIRED |
| Keywords | 3–10 at submission | S1 | PASS (7) |
| References | Accurate, relevant, unretracted | S1 | PASS within recorded check |
| Supplement | Applicable material ready for review | S1/S2 | HUMAN_ACTION_REQUIRED (deposit) |
| Article type | Research Article selected for this draft | S1 | PASS |
| Exclusivity | Authors confirm no concurrent submission | S1 | HUMAN_ACTION_REQUIRED |
| AI content | Acknowledgment, system identification, affected sections and extent; checklist also asks section citations | S1/S2 | HUMAN_ACTION_REQUIRED |
| Grammar/acronyms | Review English; define abbreviations at first use | S1 | PASS within preserved-content review |

The template specifies a one-paragraph 150–250-word abstract, alphabetized
index terms, and Access front matter (S7). The converted draft preserves the
audited abstract and uses seven terms; build evidence records its word count.

S4 additionally states an ORCID requirement for all IEEE journal authors.
Collect both authors' actual IDs; S1 specifically identifies the submitting
account requirement. A printed ORCID in the byline is not asserted mandatory.

S2/S5 distinguish content generation from grammar/editing assistance. Generated
content requires disclosure; grammar-only disclosure is recommended. The
draft must not assume the entire earlier workflow was grammar-only.

S2/S4 encourage data/code sharing and describe repository services. These pages
do not establish a universal mandatory public DOI or a mandatory separately
titled availability section. Deposit is nevertheless required for this package's
promised evidence release and the author's requested readiness gate.

Funding belongs in the template's support footnote when applicable. The checked
Access checklist does not mandate separate CRediT, competing-interest, or
general ethics sections. Authors must still answer any applicable portal questions
truthfully; declarations are prepared in SUBMISSION_DECLARATIONS.md.

## After acceptance only

S3 requests final source with biographies/photos, a PDF named `FINAL Article.pdf`,
a graphical abstract and its caption document, reviewed videos if any, and
separate graphics if not embedded. Copyright/licensing and APC settlement follow
acceptance. These are not initial-upload deliverables here. Publication DOI,
volume, issue, dates, and copyright are assigned by IEEE, not invented by authors.

## Template provenance and redistribution boundary

Downloaded S7 ZIP SHA-256:
`60c7efc9db8ac9e8bdb31c550ad4e03cb6f258a878ececc0bc690b6203e45a67`.
The example notes a 2026-05-13 byline revision. The supplied class internally
defaults to volume 11/year 2023; the draft explicitly labels both pending.
The real engine is pdfLaTeX, required by the supplied spotcolor package.

Official class/style/font/logo assets are unchanged and supplied only to compile
the submission. IEEEtran class/BibTeX files retain their LPPL notices. The
download supplies no separate blanket license for the Access fonts/logos;
do not relicense them under this repository's MIT license or include them in
a public evidence deposit without confirming redistribution rights. A public
source deposit can instruct readers to obtain S7 directly. The staging README
records this boundary; the frozen scientific ZIP contains no new template assets.
