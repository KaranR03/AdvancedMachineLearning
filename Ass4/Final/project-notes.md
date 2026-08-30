# Project 4 — AI Auditing

- **Weight:** 5%
- **Mode:** Group — **Group 4** (Karan Rooprai n12498122, Nhu Hieu Nguyen n12194778)
- **Due:** Friday **28 Aug 2026** 11:59:59pm · hard lock Sunday **30 Aug 2026** 11:59:59pm
- **Submit:** `project4_report.pdf` + `project4_code.zip`, at two separate Canvas submission points
- **Status:** built and verified, **not yet uploaded**. Both files are in [`submission/`](submission/).

> ✅ **These times are confirmed, not extrapolated.** The Canvas submission points carry
> `due_at 2026-08-28T13:59:59Z` and `lock_at 2026-08-30T13:59:59Z`, which is 11:59:59pm Brisbane
> (UTC+10, no DST).

> ⚠️ **Renamed from `project-4-ml-doctor/`.** The Canvas assessment page is titled
> **"IFN680- Project 4 - AI Auditing"**; the unit outline calls it *ML doctor* and the supplied
> notebooks are headed `MLDoctor - Case1`. All three names refer to this project.

> **How the group submission works.** Both submission points are group assignments, so **one member
> uploading submits for the pair**, but `grade_group_students_individually` is true, so Canvas
> records a mark per member. Agree who uploads, and confirm it landed.

## Brief

You act as a **Lead AI Auditor**. Three pre-trained CelebA classifiers passed internal testing and
fail in the field. For each, identify the **root cause**, give the evidence, and recommend a fix.
**Retraining is explicitly out of scope.** Each case supplies a checkpoint, a start-up notebook, an
internal test set drawn from the development distribution, and an external sample of field data.

| | Scenario | Observed |
|---|---|---|
| **Case 1** | Grooming app: **clean-shaven** detector, studio to mobile | Near-perfect in the lab, *"significantly decreased"* in the field |
| **Case 2** | Retail kiosk: **eyeglasses** detector | Works for **some users**, fails for others, *in clear, well-lit images* |
| **Case 3** | Social tagging: **"appears young"** with a **high-confidence bypass** | Publishes wrong tags *while reporting near-certain confidence* |

The report must give **Diagnosis, Experiments and evidence, Recommendations** for each case, in at
most **2 pages** (over that is 0). The code zip must contain `case1.ipynb`, `case2.ipynb`,
`case3.ipynb`; wrong names score 0, and *"if your code does not work, you will score 0"*.

> ⚠️ **The bypass threshold is stated inconsistently.** The Canvas page says **90%**; the supplied
> `case3` notebook's own markdown says **99%**. Both are computed and both are reported.

## Findings

Each case gets the same diagnostic battery and then one experiment designed so that a negative
result would have falsified the diagnosis.

### Case 1 — texture dependency, exposed by a loss of image detail

The classifier reads "clean-shaven" from fine stubble texture on the chin and jaw. At identical
pixel dimensions and file format, the field images carry **9.9× less** high-frequency detail
(Laplacian variance 0.00109 against 0.01077) and are **0.262 brighter**. Cases 2 and 3 show ratios
of 1.2 and 0.9, so the gap belongs to this case rather than to the measurement.

**The controlled experiment.** Blurring the *internal* images, holding subjects and labels fixed,
walks accuracy monotonically down to the field result: at Gaussian σ 4.0 it reaches **0.620 against
the field's 0.619**, with 310 false positives against 278 and precision 0.570 against 0.568. The
brightness shift on its own reaches only 0.918. Loss of detail is therefore sufficient to produce
the failure, and a field set of different or harder subjects is not needed to explain it.

Saliency adds the mechanism: on images it gets right, 0.542 of the activation is on the mouth and
jaw; on images it gets wrong, 0.327, with the forehead and hair share rising from 0.042 to 0.242.

### Case 2 — a spurious correlation with the subject's sex

**This is the headline finding, and it is visible in the data.** The four class folders are:

| | `negative` (no eyeglasses) | `positive` (eyeglasses) |
|---|---|---|
| `test_internal` | all **women** | all **men** |
| `test_external` | all **men** | all **women** |

Eyeglasses and sex are **perfectly confounded internally**: a model that classified sex and never
looked at the glasses would have scored close to the reported 0.987. The internal set contains only
**two of the four** (sex × eyeglasses) combinations, and the external set is exactly the other two.

| | no eyeglasses | eyeglasses |
|---|---:|---:|
| **women** | 150/150 = **1.000** | 93/150 = **0.620** |
| **men** | 120/150 = **0.800** | 146/150 = **0.973** |

0.987 on the combinations it was shown, 0.710 on the ones it was not: an error rate **22× higher**.
Errors fall on both sides as a sex cue predicts — 57 women wearing eyeglasses missed, 30 men
without them reported as wearing some. Saliency measures the same thing: 0.552 of the activation is
on the eyes and nose when the model is right, 0.325 when it is wrong, where the largest share
(0.498) moves to the mouth and jaw.

### Case 3 — the model does not transfer, and its confidence hides that

The internal negatives are clearly elderly; the field negatives are middle-aged. The lab phase
measured a much easier problem. Internal AUC 0.9960 falls to **0.5640** externally, near chance,
while the softmax keeps emitting extreme probabilities, so ECE rises 0.036 → 0.296. Of field
predictions at ≥0.90 confidence (54.3% of the split) only **0.558** are correct; at ≥0.99, 0.517.

**Calibration cannot rescue it, and the notebook proves rather than asserts that.** A temperature
fitted on half the field set and measured on the other half drops ECE 0.326 → 0.070 while AUC stays
at 0.544, *unchanged*, because dividing a logit by a constant is monotone and cannot reorder
predictions. T = 12.16, and afterwards the most confident field image reaches only 0.687, so a gate
at either threshold publishes **0 of 150**. The risk-coverage curve gives the ceiling: the best
accuracy reachable at any coverage above 2% is **0.800**. So the recommendation is to **disable the
bypass**, not to re-tune it.

## What was built

```
notebook/            the working copy, and what the zip is built from
  case1.ipynb  case2.ipynb  case3.ipynb   generated by tools/build_notebooks.py
  case{N}_predictions.pkl                 labels and P(positive), cached per split
  case{N}_{confusion,evidence}.pdf        the two report figures each case contributes
  Case{1,2,3}/                            the supplied data and weights; never zipped
report/
  project4_report.tex   generated by tools/build_report.py, never hand-edited
  project4_report.pdf   2 pages
  figures/              copies of the notebooks' figures
submission/          ← the two files to upload
  project4_report.pdf  project4_code.zip
tools/
  build_notebooks.py           writes all three notebooks from one template
  run_notebook.py              executes one from a fresh kernel, reports what verify.py would fail
  rebuild_preserving_outputs.py  change prose without re-running; asserts the code is byte-identical
  build_report.py              reads the notebooks' printed output, writes the .tex
  build_pdf.py                 writes the .tex, compiles it, fails on an overfull box
  verify.py                    every pre-upload check
  build_zip.py                 writes submission/
  clean_room.py                unzips the archive into an empty directory and runs it
```

### Rebuilding

```bash
# from this folder, with ../../../.venv/Scripts/python.exe as `python`
python tools/build_notebooks.py
python tools/run_notebook.py case1.ipynb     # ~10 min on CPU, case2/case3 ~2 min each
python tools/build_pdf.py                    # build_report.py + tectonic + the box check
python tools/verify.py                       # must print ALL CHECKS PASSED
python tools/build_zip.py
python tools/clean_room.py                   # the grader's own test
```

`tectonic` is at `C:\Users\Admin\.conda\envs\tex\Library\bin\tectonic.exe`, not on PATH.

## Why the notebooks are generated rather than hand-written

The three audits run an identical battery, so a change to the battery would otherwise be three
edits that can drift apart. `build_notebooks.py` holds the shared cells once and parameterises them
per case, and it **compiles every generated cell before writing it** — the cell bodies are ordinary
Python strings, so an escape meant for the notebook (`\n` in a plot title) is otherwise consumed by
the generator and produces an unterminated literal that only surfaces minutes later during
execution.

`rebuild_preserving_outputs.py` exists because case 1 takes ten minutes to run and prose changes
far more often than code does. It regenerates and transplants the stored outputs back, but only
after asserting every code cell is byte-identical to the one that produced them.

## Traps this project has to survive

| | |
|---|---|
| **All three checkpoints are named `resnet_frozen_best.pth`** and are different models. A flat unzip would silently load the wrong one. Each notebook searches its case folder first **and asserts the MD5** of what it loaded |
| `np.trapezoid` needs numpy ≥ 2.0 and `np.trapz` is gone in ≥ 2.0. The measured GPU node has 2.5.1 but `ifn680-cpu` is unmeasured, and the assessment names the base environment. Both spellings are banned by `verify.py`; AUC comes from `sklearn` |
| `transforms.Resize` on a **tensor** only antialiases by default from torchvision 0.17. `antialias=True` is pinned so the numbers cannot depend on the marker's version |
| The supplied folders carry a `.DS_Store` inside every class directory, so a raw file count gives 842/742 where `ImageFolder` reports 840/740 |
| A `figure*` can only be placed at a page top **at or after its declaration**, so it is declared before the case sections; declared beside its own discussion it is deferred to a third page, which is a zero |
| **The report has about one line of spare space, and it was bought.** Relabelling three `Evidence.` headings to the brief's `Experiments and evidence.` costs 48 characters and once produced a three-page PDF, which scores 0. The room came from setting Table 1 in `\footnotesize` with `\tabcolsep` at 3pt and cutting a clause that restated the sentence above it. Recompile and page-count any wording change before keeping it; `verify.py` is what catches it |
| **Table 1 was 9.5pt wider than its column** and overhung the gutter on both sides. Nothing in `verify.py` saw it, because an overfull `\hbox` is a tectonic warning rather than a page count. `tools/build_pdf.py` now fails the build on one, and `verify.py` measures the rendered word boxes against the page margins |
| Figure 1 is hand-set inside `\twocolumn[...]` and is not a float, so `\setcounter{figure}{1}` is needed or the second caption also prints as "Figure 1" |

> ⚠️ **Do not trust the Case 1 scaffold's stored numbers.** Its saved output claims **1000/1000**
> images and **0.891/0.596** accuracy. The dataset Canvas serves today holds **840/740** and scores
> **0.937/0.619**: its assets were re-uploaded and the notebook was never re-run against them.
> Cases 2 and 3 still match their scaffolds. Run it, do not quote it.

## Verified on the Hub GPU node as well as locally

The brief asks for the GPU environment and for `.to(device)`, so `case1.ipynb` was also executed on
the Hub's `ifn680-gpu` node to confirm the code reaches the accelerator and that nothing in the
result depends on it.

```
device      : cuda:0            NVIDIA A16-4Q
torch       : 2.13.0+cu126      torchvision 0.28.0+cu126
python      : 3.12.14           numpy 2.5.2
20 code cells, 0 errors, 0 stderr, execution counts 1..20
```

Of the 83 printed lines, **four differ and all four name the environment** (device, torch,
torchvision, numpy). Four measured lines move in the third decimal, each a single image out of
370 or 740 changing side under a different floating-point kernel:

| line | CPU | GPU |
|---|---|---|
| coverage 50% accuracy | 0.778 | 0.781 |
| band share, correct, jaw | 0.542 | 0.541 |
| blur sigma 0.5 accuracy | 0.931 | 0.930 |
| blur sigma 1.0 accuracy | 0.887 | 0.886 |

Every number the report quotes is identical on both: internal 0.9369, external 0.619, the sigma 4.0
match at 0.620, brightness-only 0.918, the 9.9x sharpness ratio and the +0.262 brightness shift.

**What ships is the CPU run**, because the report is generated from whichever run ships and that
one is internally consistent, clean-room tested and already at exactly 2 pages. The report makes no
claim about hardware, so a stored `device: cpu` contradicts nothing in it. The GPU pass exists to
answer the question it was run to answer: the code uses the device it is given, and the diagnoses
do not depend on which one that is.

## The clean-room test

`project4_code.zip` was unzipped into an empty directory, the three supplied datasets staged beside
it as the archive's `README.txt` instructs, and all three notebooks executed from a fresh kernel:

```
case1.ipynb: 20 cells in 450 s, errors=0, stderr=0, output identical to the committed notebook
case2.ipynb: 19 cells in 113 s, errors=0, stderr=0, output identical to the committed notebook
case3.ipynb: 20 cells in 110 s, errors=0, stderr=0, output identical to the committed notebook
```

That is the check behind *"if your code does not work, you will score 0"*, and it also shows the
run is deterministic: a fresh kernel reproduces the stored output exactly.

## Oral Q&A prep

The likely questions for an audit are *why does this evidence rule out the other explanations?*
rather than *why this hyperparameter?* Each case therefore records what could have falsified it:

1. **Case 1.** The alternative is that the field photographs simply contain harder people. The blur
   sweep rules it out, because the subjects and labels are identical at every point on it. Be ready
   for: *why Gaussian blur when the mobile pipeline is not Gaussian?* The answer is that the sweep
   establishes sufficiency, not mechanism — removing detail alone reproduces the failure, so no
   further explanation is required.
2. **Case 2.** The alternative is uniform degradation on the positive class. The 2×2 table rules it
   out: errors track the attribute combination, not the class. Be ready for: *how do you know the
   folders differ by sex?* It was established by displaying all four, and the notebook shows them;
   the file names are **not** usable CelebA ids, since 37 collide between Case 1's two splits and
   show different faces, so the sets were renumbered and `list_attr_celeba.txt` cannot be joined.
   Also expect: *0.987 on the combinations the internal set contained and 0.710 on the ones
   it did not are just your two split accuracies, so what did stratifying add?* They are,
   and that identity **is** the finding: the internal split is exactly the two combinations,
   which is why an aggregate over it could never have exposed the confound. The stratification
   that carries new information is the four-cell breakdown, 1.000 and 0.973 on the pairing the
   model was shown against 0.800 and 0.620 on the pairing it was not, together with errors on
   both sides, 57 missed and 30 invented. Uniform degradation predicts neither.
3. **Case 3.** The alternative is that recalibration fixes it. Fitting a temperature out of sample
   rules it out: ECE falls and AUC does not move, which is what monotonicity requires.

## Source tutorial

[`weeks/week-05/IFN680_Week5_Tutorial.ipynb`](../../week-05/IFN680_Week5_Tutorial.ipynb) —
*Safety, Performance, and Ethics*. The Canvas page names the 5.3 tutorial as the reference for
evaluation metrics and saliency methods, and CAM comes straight from it.

## ⛔ The instructor solution to this exact assessment is in this repo

`reference/upstream/IFQ680_s1_2026/Week_4/Assesment/Case{1,2,3}/` holds *MLDoctor - Case1/2/3*. The
repo policy is **guideline only, never copy** ([`CLAUDE.md`](../../../CLAUDE.md)). None of it was
opened in building this project; every diagnosis above is derived from the supplied data.
