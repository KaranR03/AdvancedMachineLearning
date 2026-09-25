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

## Review pass before upload

Two further reads of the built report and the three notebooks against the marking criteria, after
the build was already verified. Nothing here changes a measured number: the two training notebooks
were **not** re-run, because every edit to them is a comment, a docstring, or the `_ =` that stops
a cell echoing an object, and none of those can alter an output. Their printed output is
byte-identical to the Hub run.

- **No marking vocabulary anywhere in the deliverables.** "a marker", "the rubric" and six
  phrasings of the form "asserted rather than assumed" were sitting in cell comments.
- **Digit positions are now split by operation as well as by mode**, which the task asks for and
  the report only did by mode. It turns up a fact worth a clause: no subtraction answer reaches
  the thousands column, because operands of at most three digits can only pass 999 by adding.
- **A carry and borrow sentence with its counts.** Forward is exact on both addition cases, so
  all 7 of its errors are subtractions and 6 of those need a borrow.
- **Every error is printed**, not the first twelve. Table 2's caption claimed it was drawn from
  the full list while the cell printed 12 of Reverse's 44.
- **0.78 is named** in the text, both captions and Table 1, where the report had said "the
  expected level" three times without the number.
- The closing section is now **Strengths, limitations and recommendations** and opens on the
  design strength, since the top tier asks for both halves. Each of the four limitations now
  states what it costs, the absolute-position one included.
- **Figure 1 redrawn.** Its labels were rendering at 4.7pt, because it was 10.4 in wide placed
  at 6.95 in. Labels raised to 10pt, height trimmed, and the legend cut to the colour key, which
  was wide enough to cover the curve it explained.
- Thousands separators throughout, the p-value typeset as proper math, and quantifiers written
  from the data, so "7 of Forward's 7 errors" now reads "All 7 of Forward's errors".
- **Nothing in a notebook points outside the archive.** Three comments named `build_report.py`,
  which is not in `project5_code.zip`, so anyone unzipping it met a reference to a file they had
  never been sent.

A third pass then went back to the brief and the two things it still did not match:

- **The report's parts now carry the brief's own names.** Submission Instructions ask for a
  "Method description" and a "Results and analysis", and neither phrase was anywhere in the
  report; it had three sections named after the tasks instead. Tasks 1 and 2 are now run-in bold
  heads inside Method description, which is what the four analysis sub-parts already were, and
  Task 3 became Results and analysis. Two of `main_report.ipynb`'s headings took the same
  treatment, so all four of the brief's analysis terms can now be found by scrolling it.
- **`main_report.ipynb` was re-run on the GPU node, from the shipped archive.** It is the notebook
  that gets run for grading and its stored outputs had only ever come from a CPU. Unzipping the
  archive on the Hub exactly as a marker would, with the two figure PDFs deleted first so they had
  to be regenerated, gave 0 errors, 0 stderr, execution counts 1 to 24, both figures back, and all
  9,745 characters of printed output identical to the CPU run to the same md5 — with torch, CUDA
  and numpy all different between the two. `report_src/adopt_hub_outputs.py` then took those
  outputs across, after checking cell by cell that the source bytes matched, so what ships is the
  verified notebook carrying the Hub's outputs rather than a file swapped in wholesale.

A fourth pass read the built PDF and all three notebooks the way a marker meets them,
rather than re-running the checks, and found four things:

- **The Method description now says how the models were trained.** It claimed that only the
  changes it described differed from the workshop, and then named none of the three settings that
  do differ: 30 epochs against the workshop's 15, a cosine schedule annealing the learning rate to
  zero, and 150,000 training examples against 50,000. Batch size appeared nowhere at all, so the
  report's own "12,000 optimiser updates" could not be checked by the person reading it.
- **A comment in both training notebooks contradicted the class beside it.** It read "dropout 0.1
  rather than the class default of 0.5" above a class that declares 0.1. The 0.5 is the workshop's
  default, and only for the positional encoding; its encoder layer hard-codes 0.1. One value is
  passed explicitly here so that training and evaluation build the same model, and the comment now
  says that.
- **`main_report.ipynb` uses the workshop's `evaluate()` instead of merely defining it.** It and
  `get_batch()` had no caller, which is what an earlier pass deleted `overall_counts()` for.
  Rather than delete them, the notebook now scores the same predictions both ways: over answer
  tokens, as the workshop did, and over the integer, as Task 3 asks. They agree to four decimals
  in both directions, which makes each a check on the other, and the token-level digit rate gives
  the report a digit-level number in the brief's own literal sense. The epoch count, the batch
  size and the parameter count are printed in the same run, so the report reads them out of stdout
  like every other number it carries instead of stating them from outside.
- **Wording.** "the figures reproduce exactly" on a page holding two figures now reads "the
  numbers"; a paragraph no longer opens with a numeral; and the sentence about the reverse curve's
  late jump states a bound the curve satisfies (below a quarter of final accuracy through epoch
  17) and names the epoch of each crossing, where it had said "near a tenth until epoch 18" of a
  curve that is at 0.21 by epoch 17.

A fifth pass read the compiled PDF once more, sentence by sentence, with the report as the
priority, and changed seven of them:

- **The reason Task 2 can safely emit the sign last was wrong.** It said the sign "is only
  determined once the magnitude is", which has the dependency backwards: subtracting by hand
  settles which operand is larger first, and the units digit depends on that. The reason that
  actually holds is that the sign follows from the prompt, which the model sees at every step.
- **The closing recommendation now earns its last two clauses.** "It trains as easily" sat two
  paragraphs after the curves that show 0.95 reached at epoch 8 against epoch 19, and "its failure
  mode is rarer" was doing more work than the evidence allows: both models concentrate their errors
  in a group of 92 examples, and they are different groups of 92. What separates them is the cost,
  7 errors against 42. Both clauses are now generated from the tables that printed those numbers,
  and the count comparison is only written at all when the two groups are the same size.
- **Three smaller things.** The two token-level rates now say which model each belongs to, because
  the sentence before them pairs its numbers as addition and subtraction. "The operand width table
  resolves it" named a table that lives in the notebook, not the report. And the barrier sentence
  no longer asserts a cause from a single seed that the limitations paragraph disowns a column
  later. Training data joined the list of settings the two models share, which is the strongest
  item on it: same seed, same split sizes, so literally the same 150,000 prompts.

Six more rails came with them, and four failed against the report before the fixes went in. The two
that passed are guards, named as such in the script. One of the six was wrong at first in a way
worth recording: it matched a phrase against the raw LaTeX and reported it absent while it sat in
front of it, because the line breaks in generated LaTeX fall wherever the generator's own string
wrapped and move whenever a sentence is edited. `verify.py` now keeps one whitespace-flattened copy
of the report for every rail that looks for a phrase.

A sixth pass read the compiled PDF once more and changed three things:

- **A sentence announced an agreement and then printed the same number twice.** The ablation
  paragraph said the two training-set sizes agree once measured in optimiser updates, and then:
  "Forward passes 0.95 after 12,000 updates on the reduced set against 12,000 on the full one."
  That is true, and it is the striking part of the ablation: epoch 8 on 150,000 examples and
  epoch 15 on 80,000 are the same 12,000 updates. Written as a contrast it reads as a slip. It
  now says "after 12,000 updates on either set", and the generator only reaches for that phrasing
  when the two counts are identical rather than merely close.
- **Figure 2 and Table 2 were never cited.** Both were numbered and captioned and nothing in the
  running text pointed at either, so a reader met them without being told why they were there.
  Figure 2 is now cited where its top panel's content is discussed and Table 2 on the sentence its
  caption restates.
- **The two ways this report counts a wrong digit are now tied together.** Figure 1's third panel
  shows 41 wrong digits at the tens for Reverse and the Error patterns paragraph says 38. Both are
  correct: the panel comes from the per-place accuracy, which marks a place wrong when a short
  prediction cannot supply it at all, and the paragraph counts only the errors that kept the right
  number of digits. The difference is exactly the three wrong-length predictions. Nothing tested
  that, because the traceability sweep matches decimals and both of these are integers.

Two of the four new rails failed against the report before the fixes, and one of the four had to
be written twice. The first version read the figure counter out of the stripped copy of the report
that the traceability sweep leaves behind, and `\setcounter` is one of the macros that sweep
removes, so it decided the first floating figure was Figure 1, found Figure 1 cited, and passed
without ever looking at Figure 2. It now reads the counter from the unmodified file.

A seventh pass went back to the unit's own deduction list, the one in the 11 Aug 2026
announcement, and checked the report against it item by item rather than against the rubric:

| named deduction | where we stand |
|---|---|
| report over 2 pages, 0 for the report | 2 pages, with 12 words of headroom before it becomes 3 |
| code does not run, 0 for the code | `clean_room.py` runs `main_report.ipynb` from the archive alone, 24 cells, 0 errors, 0 stderr |
| results not on the test set after fitting the best model on training+validation, -0.5 | **this was the gap** |
| code not commented | 16 to 17% comment lines, against the 9 to 10% that scored 2/2 on Project 4 |
| naming convention | `project5_report.pdf`, `project5_code.zip` |

That third row is the one that has already cost us half a mark once: Project 2's report came back
at 2.5 with "The confusion matrix and all results should be reported on the testing set after
training on train plus validation data."

Every result in this report was already measured on a held-out set of 10,000 that shares no prompt
with training or validation, so the first half was never in doubt. The second half describes a
workflow this project does not run, and the report never said so. There is no model selection
here at all: neither training notebook has early stopping, a patience counter or a
best-checkpoint rule, both train a fixed 30 epochs and save the final weights once, and the
validation split exists to draw the learning curves that the convergence finding rests on.
Folding it into training would delete that finding.

So the Set-up paragraph now reads "Validation only tracks convergence; no model is selected on
it." Ten words, paid for by trimming five phrases elsewhere, and a rail reads the claim back out
of both notebooks so it cannot outlive the code it describes.

Worth knowing for the next project: the first attempt at this spilled onto a third page. Ten
words added on page 1 cost more than ten words added at the end, because the probe measures slack
by appending filler to the last paragraph, which re-breaks one paragraph, while an insertion
re-breaks every paragraph after it.

An eighth pass re-read only the sentences the seventh pass had shortened, on the theory that a
trim is an edit like any other and the rails do not read for style. One was wrong: "held epochs
fixed instead of optimiser updates, so fixing updates instead would test data volume more directly"
used the same word twice in twelve. The second "instead" is gone.

Removing it moved the page headroom from 7 words to 12, which is the line-breaking asymmetry again
in the other direction: the word sat mid-document, so dropping it pulled a whole line back up.

`report_src/verify.py` now prints ALL CHECKS PASSED across 169 checks, and `clean_room.py` still
runs `main_report.ipynb` from the archive alone with 0 errors and 0 stderr — reproducing
GPU-stored outputs on a CPU, which is the device-independence claim the report makes, tested
rather than argued.

**The report has changed since the files that are on Canvas.** Attempt 1 of both points went up on
12 Sep and carries the build as it stood before the fourth pass, so none of the last three
passes is in it: not the training configuration in the Method section, not the token-level
scoring, not the GPU outputs in `main_report.ipynb`, and none of the ten sentence changes the
fifth and sixth passes made. `project5_code.zip` is unchanged by both of those passes and still
hashes to what the fourth pass produced; only the PDF moved.
Both points accept a second attempt until the Sunday lock.

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
