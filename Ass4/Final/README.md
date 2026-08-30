# Project 4 (AI Auditing) — final pair, for review

**These two files are what gets submitted:**

| File | Size | MD5 |
|---|---:|---|
| `project4_report.pdf` | 153,516 B, **2 pages** | `118f6d9c4328d064e23d12f1d0665d0f` |
| `project4_code.zip` | 2,099,864 B, 7 flat entries | `d3ad064f80a8474752e9a9efab5618e4` |

`project4_code/` is the same archive unpacked, so the three notebooks can be read in the browser
without downloading anything. `report_src/` is everything the PDF is built from.

Built off your `Ass4/` draft. The structure, the framing and the metric coverage are yours and
survived unchanged. What follows is what moved and why, because the oral Q&A questions each of us
individually and neither of us can defend a conclusion we have not seen the argument for.

---

## Two of the three diagnoses changed

### Case 2 is a spurious correlation with the subject's sex, not frame style

Displaying all four class folders shows the internal set is built from **women without eyeglasses
and men wearing them**, and the external set holds **exactly the two combinations that never appear
internally**. Eyeglasses and sex are perfectly confounded in everything the model was validated
against, so a classifier that responded only to sex would have scored near the reported 0.987
without representing eyeglasses at all.

Stratifying the same predictions by combination:

| | no eyeglasses | eyeglasses |
|---|---:|---:|
| **women** | 150/150 = 1.000 | 93/150 = 0.620 |
| **men** | 120/150 = 0.800 | 146/150 = 0.973 |

Both cells of the pairing the model was shown are near-perfect; both cells it was never shown
collapse. The errors fall on **both sides** — 57 missed eyeglasses, 30 invented — which uniform
degradation cannot produce. That is also the reported symptom: it works for some users and not
others, on images of equal quality.

The draft's saliency sentence was also removed rather than reworded. It said the CAM maps show
attention drifting off the eye region on the missed images, but the CAM cell picked indices with
`rng.choice`, so it never selected a missed image. The notebook now compares **false negatives
against true positives** deliberately and prints the band shares, so the claim and the evidence are
the same thing.

### Case 3's recommendation is to disable the bypass, not re-tune it

External AUC is **0.564**. Temperature scaling is monotone: dividing a logit by a constant relabels
the confidence axis without reordering anything, so the set of achievable (coverage, accuracy) pairs
is fixed by the ranking. Recalibration therefore *cannot* produce a threshold at which the model is
90% accurate; it can only reveal that no such threshold exists. The draft's own numbers already
showed it — 0.558 at the 0.90 gate, 0.517 at 0.99 — so "re-derive the threshold so that 90% means
90%" was asking for something the data rules out.

The notebook now proves it two ways: a temperature fitted on half the field set and measured on the
other half (ECE 0.326 -> 0.070, **AUC unchanged at 0.544**, and afterwards no image reaches even
0.70 confidence, so the gate publishes nothing), and a risk-coverage sweep whose ceiling is 0.800.

### Case 1 kept its diagnosis and gained a mechanism

"Covariate shift" was right but asserted. It is now measured and named as the **texture dependency**
the brief gives as an example: the field images carry **9.9x less** high-frequency detail at
identical pixel dimensions and file format, with cases 2 and 3 as controls at 1.2x and 0.9x. The
decisive test blurs the *internal* images and changes nothing else, so subjects and labels are
identical throughout; at sigma 4.0 it lands on the field result, 0.620 against 0.619, with the same
one-sided shape, 310 false positives against 278. Brightness alone reaches only 0.918.

---

## What to check, in the order it matters

1. **Open `project4_report.pdf` and read it.** Two pages, and it is the whole submission for the
   report component. Every number in it is printed by one of the notebooks.
2. **Read the three Findings sections** (section 13 of each notebook). They are the long-form
   version of the report, and they are what the oral will follow.
3. **Read `project-notes.md`**, especially **Oral Q&A prep** near the end. It lists, per case, the
   alternative explanation the experiment was designed to rule out, and the questions to expect —
   including the one about whether Case 2's stratification adds anything beyond the split
   accuracies. Worth having pre-loaded rather than meeting it under questioning.

## If you want to rebuild any of it

```bash
python report_src/run_notebook.py case1.ipynb    # ~10 min CPU; case2/case3 ~2 min
python report_src/build_pdf.py                   # writes the .tex, compiles, fails on an overfull box
python report_src/verify.py                      # must print ALL CHECKS PASSED
python report_src/build_zip.py
python report_src/clean_room.py                  # unzip into an empty dir and run all three
```

`verify.py` is the gate: file names, the 2-page limit, no stderr anywhere in the stored outputs,
execution counts 1..N in order, every decimal in the report traced to a line some notebook printed,
and the zip's members byte-identical to the notebooks beside them. Paths inside these scripts point
at the machine they were written on, so adjust `BASE` before running them here.

## Verification already done

- **Clean room** — unzipped into an empty directory, datasets and checkpoints staged as the unit
  ships them, all three notebooks run top to bottom: 0 errors, 0 stderr, stdout byte-identical to
  the committed notebooks.
- **GPU node** — the same notebooks on the IFN680 GPU node (torch 2.13.0+cu126, `cuda:0`). Of the
  83 lines they print, the only differences are the four naming the environment and four values
  moving in the third decimal, where a borderline probability lands on the other side of 0.5 under
  a different kernel. **Every figure the report quotes is identical on both devices.**
- All three checkpoints are named `resnet_frozen_best.pth` and are three different models, so each
  notebook asserts the MD5 of the file it loaded. Unzipping all three into one folder now fails
  loudly instead of producing wrong numbers quietly.
