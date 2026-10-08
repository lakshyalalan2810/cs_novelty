# Lightweight literature check — 2026-10-08

No scholarly entry or Related Work passage was changed. The existing sixteen
references all occur in the preserved manuscript; no unused or unresolved
scholarly key was found. Submission additionally cites the official Codex system
website for documented current-pass AI declarations; it is not a research citation.

## Metadata

The fifteen existing DOIs were retrieved from the current Crossref /works API.
Titles, author order, year, venue, volume/issue, pages/article number and DOI were
compared with the current bibliography and the publisher sources already listed
in LITERATURE_POSITIONING_AUDIT.md. Holm's author/title/1979/6(2)/65–70 record was
independently read in the [Wiley journal issue on JSTOR](https://www.jstor.org/stable/i412579).
No DOI was invented for Holm.

| Key | Current DOI / primary record | Result |
|---|---|---|
| Hochreiter1997 | https://doi.org/10.1162/neco.1997.9.8.1735 | Matches |
| Page1954 | https://doi.org/10.1093/biomet/41.1-2.100 | Matches |
| Holm1979 | https://www.jstor.org/stable/4615733 | Matches journal issue |
| Mayne2000 | https://doi.org/10.1016/S0005-1098(99)00214-9 | Matches |
| Bonassi2021 | https://doi.org/10.1016/j.ifacol.2021.08.417 | Matches |
| Terzi2021 | https://doi.org/10.1002/rnc.5519 | Matches |
| Isermann2006 | https://doi.org/10.1007/3-540-30368-5 | Matches; registry omits subtitle |
| Gustafsson2000 | https://doi.org/10.1002/0470841613 | 2000 print year retained; online date 2001 |
| Venkatasubramanian2003 | https://doi.org/10.1016/S0098-1354(02)00160-6 | Matches; registry title abbreviated |
| Chow1984 | https://doi.org/10.1109/TAC.1984.1103593 | Matches; printed author names corroborate registry initials |
| Aguilera2016 | https://doi.org/10.1016/j.conengprac.2016.04.014 | Matches |
| Choi2021 | https://doi.org/10.1109/TIE.2020.2992977 | Matches issue year 2021 |
| Chu2023 | https://doi.org/10.3390/s23094330 | Matches; printed Kuewwai spelling retained |
| Salazar2017 | https://doi.org/10.1016/j.ress.2017.04.012 | Matches |
| Bonassi2024 | https://doi.org/10.1016/j.automatica.2023.111381 | Matches issue year 2024 |
| Zhao2019 | https://doi.org/10.1016/j.ymssp.2018.05.050 | Matches issue year 2019 |

Publisher/registry differences above are already explained by the prior primary
records; they are not grounds to change valid metadata for presentation.
All sources accessed 2026-10-08. Current raw registry responses are retained in
ignored `tmp/access_packaging/crossref_*.json`, with a combined literature_check.json.

## Retraction check

The current [Crossref Retraction Watch production data](https://www.crossref.org/documentation/retrieve-metadata/retraction-watch/)
was downloaded from its documented
[CSV endpoint](https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv).
The file contains 73,005 records. A scan of original-paper DOI fields for all
fifteen DOIs, plus title matching for Holm, found zero matches. DOI responses
contained no retraction-type `updated-by` or `update-to` record. No cited paper
was identified as retracted in these current checks. Registry coverage/delay is
a limit; this is not a guarantee about future notices.

## Claim/source alignment

The preserved prose uses Mayne/Hochreiter for established MPC/LSTM ingredients;
Terzi/Bonassi for conditional network/control stability; Chow/Isermann/
Venkatasubramanian for redundancy and diagnosis taxonomy; Page/Gustafsson for
change monitoring; Aguilera/Choi for disturbance-aware motor-drive diagnosis;
Chu/Zhao for neural sensing/monitoring; Salazar for actuator/system reliability;
and Holm for multiplicity. The distinctions between these sources and the
paper's narrow simulated entry endpoint remain explicit. The current Terzi
registry abstract and [Bonassi author manuscript](https://re.public.polimi.it/bitstream/11311/1260618/9/Nonlinear_MPC_deltaISS.pdf)
corroborate the formal-guarantee discussion. Earlier primary originals support
the retained motor-drive comparisons; inaccessible publisher pages were not
treated as newly read full texts. No exhaustive closest-prior-work search or
independent replication of other papers was performed. No concrete missing
closest-prior-work citation or correctness problem was found in this final check.
