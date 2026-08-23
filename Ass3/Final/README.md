# Project 3 — what to upload

**Upload these two files, nothing else:**

| File | Canvas submission point |
|---|---|
| `project3_report.pdf` | Project 3 — report |
| `project3_code.zip` | Project 3 — code |

Both submission points are group assignments, so **one of us uploading submits for the pair**.
Agree who does it and confirm it landed. `grade_group_students_individually` is set, so we are
still marked separately.

---

## ⛔ Do not upload `project3_report.docx`

It is kept here for reference only. Two independent reasons it cannot be submitted:

1. **It is 3 pages.** Word records this itself — `docProps/app.xml` inside the `.docx` says
   `<Pages>3</Pages>`. The unit's rule is that a report longer than 2 pages scores **0 for the
   whole report component**, which is 3 of the 5 marks.
2. The required filename is `project3_report.pdf`. A wrong filename is a zero on its own.

`project3_report.pdf` carries the same content at exactly 2 pages, plus four passages the `.docx`
had dropped (see below).

## Where the numbers come from

All of them are Karan's 23 Aug retrain — his `development.ipynb` run, his `best_model.pth`, his
`histories.pkl`. Nothing was re-trained and nothing was hand-typed.

**Average per-class accuracy 0.8148** on the 669 test images (target 0.75), overall 0.8146,
macro-F1 0.8152.

That run was checked rather than trusted: loading `best_model.pth` and scoring the test split on
**CPU** reproduces the CUDA numbers exactly — 0.8146 / 0.8148 / 0.8152, 7 of 20 classes above 0.87,
confusions 13 and 11. The previous bundle could not do that, which is what prompted the check.

## What changed from the bundle pushed at 17:33

- The report was regenerated through the LaTeX pipeline in `report_src/`, which is calibrated to
  exactly 2 pages. Every number in it is pulled from `main_report.ipynb`'s printed output by
  `build_report.py` — none is typed, so none can go stale.
- Restored four passages the `.docx` dropped: why Experiment 3 varying optimiser + weight decay +
  schedule counts as one axis rather than three; why the final `lr=0.0001` was never validated and
  is a design judgement; the "for the target application" assessment the brief asks for; and the
  calibration / per-class-support caveats. These are report marks, and they are also the two
  questions most likely to come up in the oral.
- Removed every em dash from the report prose. Each of the fourteen was replaced with the
  punctuation its own sentence needed rather than one substitute throughout: a colon where the dash
  introduced a definition or the evidence for the claim before it, paired commas for an appositive,
  parentheses for an aside inside a list, and a semicolon where the dashes had split a subject from
  its verb. One dash was not punctuation at all -- the baseline row of Table 1 used it for "no
  change against itself", which now reads `n/a`. The rebuilt PDF contains zero em dashes and zero
  en dashes, is still exactly 2 pages, and every number in it is unchanged.
- `main_report.ipynb`: dropped a trailing empty cell, added PDF figure output, and made the
  dataset-missing error say where the dataset should go.
- `development.ipynb`: **one source-only edit**, outputs untouched. Cell 27 had
  `print(f'{'Model':<28}...')`, which reuses the f-string's own quote inside its braces — that is
  Python 3.12+ syntax only. The Hub is 3.12.13 so it ran there, but it is a `SyntaxError` on 3.11
  and earlier, and the code scores 0 if it does not run. The two forms print identical characters.

## Checked before this was pushed

- `verify.py` → **ALL CHECKS PASSED** (33 checks, including the 2-page and filename cliffs)
- `project3_code.zip` unzipped into an **empty directory**, dataset placed beside it,
  `main_report.ipynb` run top to bottom: 8 cells, 0 errors, 0 stderr, output identical to the
  shipped notebook. "The files are in the zip" and "the zip works" are different checks; this is
  the second.
- Both PDF pages read visually.

## Rebuilding

```bash
cd report_src
python run_notebook.py main_report.ipynb   # ~100 s
python build_report.py                     # writes project3_report.tex from the notebook's output
tectonic project3_report.tex
python verify.py                           # must print ALL CHECKS PASSED
python build_zip.py
```
