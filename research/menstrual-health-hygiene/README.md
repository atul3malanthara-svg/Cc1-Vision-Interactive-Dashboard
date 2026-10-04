# Menstrual Health and Hygiene — IEEE conference paper

A research paper on menstrual health and hygiene (MHH), laid out to match the
IEEE two-column conference template (US Letter, Times New Roman, 0.25-in column
gap, small-caps Roman-numeral headings, 8-pt captions and references).

| File | What it is |
|---|---|
| `Menstrual_Health_and_Hygiene_IEEE_Paper.docx` | The paper (editable Word file) |
| `Menstrual_Health_and_Hygiene_IEEE_Paper.pdf` | PDF preview of the same paper |
| `figures/` | The 7 charts used in the paper (300 dpi PNG) |
| `make_figures.py` | Generates the charts (matplotlib) |
| `build_paper.js` | Generates the Word file (docx-js) |
| `finalize_docx.py` | Post-processing step called by `build_paper.js` |

## Contents

6 pages, 7 figures (2 pie/donut charts, 5 bar charts), 4 tables, 2 equations,
26 IEEE-style references.

## Data sources

All plotted values come from published sources cited in the paper:

- WHO/UNICEF JMP 2023 — global household hygiene service ladder, 2022 (Fig. 1)
- WHO/UNICEF JMP 2024 — WASH and menstrual health provision in schools, 2023 (Fig. 2)
- NFHS-4 (2015–16) and NFHS-5 (2019–21), India — women aged 15–24 (Figs. 3–6)

Fig. 7 is a model estimate computed from Eqs. (1)–(2) with the stated
assumptions in Table III.

## Before submitting

- Replace the bracketed author names, affiliations and e-mails, and the
  funding footnote.
- Re-check each statistic against the original report if your venue requires it.

## Rebuild

```bash
pip install matplotlib
python make_figures.py
node build_paper.js        # needs the `docx` npm package
```
