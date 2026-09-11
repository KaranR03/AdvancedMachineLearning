# Project 5 — final submission pair

`project5_report.pdf` (2 pages) and `project5_code.zip` are the two files to upload. They go to
two separate Canvas points. **Nothing has been uploaded yet.**

Built from the draft in `../project5_code/`. The analysis in that draft was sound and every
number in it reproduced, so the rebuild is about evidence and packaging, not substance.

## What is here

| | |
|---|---|
| `project5_report.pdf` | the report, 2 pages, generated from the notebook's own output |
| `project5_code.zip` | the archive, flat, 13 entries |
| `project5_code/` | the same files unpacked, so they can be read without unzipping |
| `report_src/` | the generator and the checks, not part of the submission |

## What changed from the draft

- The report is a **PDF named `project5_report.pdf`**. The draft was a `.docx`, and the unit
  states a wrong filename scores 0. It is generated from LaTeX, and every number in it is pulled
  out of `main_report.ipynb`'s printed output by regex rather than typed, so the two cannot drift.
- There is a **`project5_code.zip`**. The draft had loose files.
- Both training notebooks now **carry their outputs**. In the draft every `execution_count` was
  `None`, so nothing showed they ran.
- The **80k-vs-150k claim is now run** rather than asserted, in both directions, and its curves
  ship as JSON. It did not come back the way the draft's sentence predicted — see below.
- A **Limitations and recommendations** section was added.
- Per-position accuracy is now **conditioned on the position existing**. The draft zero-padded
  every answer to four digits, which scored a "thousands digit" as correct on the 7,510 answers
  that do not have one.
- Accuracies carry **Wilson intervals**, and the two directions are compared with an **exact
  McNemar test** rather than by eye. They are scored on identical examples, so the test applies.
- `tqdm` removed and `enable_nested_tensor=False` set, because both write to stderr once the
  notebooks actually run.

## What the run found

Re-trained from scratch with the same seed, it reproduces the draft's headline numbers exactly:
Forward 0.9993, Reverse 0.9956, McNemar p = 1.212e-07.

1. **The two directions fail at opposite ends of the problem, and the error sets do not overlap.**
   All 7 Forward errors are subtractions whose answer has a single digit, and all 7 are off by
   exactly one. 42 of the 44 Reverse errors are prompts whose first operand is two digits longer
   than the second, and 38 of those are wrong at the tens place.
2. **The data-efficiency gap is a unit-of-measurement artefact.** Forward reaches 0.95 at epoch 8
   on 150,000 examples and epoch 15 on 80,000 — but those are both 12,000 optimiser updates.
   Holding epochs fixed while shrinking the set also shortens the run. The report says so and
   recommends fixing updates instead.
3. **The reverse model crosses an optimisation barrier.** It sits near 0.10 until epoch 18, then
   passes 0.78 and 0.95 within one epoch. That is also why the reduced reverse run never converges:
   24,000 updates, against the 28,500 the full run needed.

## Reproducing the checks

From `report_src/`, with the paths at the top of each script pointed at wherever this sits:

```
python verify.py      # naming, 2 pages, notebook hygiene, the 0.78 threshold, traceability, zip
python clean_room.py  # unzips the archive and runs main_report.ipynb from it alone
```

`clean_room.py` is the one that matters: the brief says `main_report.ipynb` is run for grading and
any cell that fails scores 0 for the whole code component. It deletes the two figures first so
they have to regenerate, and compares the fresh output against the committed notebook line by
line. It passes in about 45 seconds on CPU.

## Provenance

Both training notebooks ran on the QUT JupyterHub GPU node (NVIDIA A16-4Q, torch 2.13.0+cu126) on
11 Sep 2026. `main_report.ipynb` ran on CPU and reproduces those numbers to the last digit,
because decoding is greedy. The notebooks are build artefacts of `report_src/build_notebooks.py`;
edit the generator, not the `.ipynb`, or the next build will overwrite the change.
