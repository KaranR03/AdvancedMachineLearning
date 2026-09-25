r"""Every check that must pass before Project 5 is uploaded.

Adapted from project 4's tools/verify.py, which passed on a submission that scored 5/5. The
cliff edges the unit states explicitly are asserted mechanically rather than eyeballed: a report
over 2 pages scores 0, wrong file names score 0, and code that does not run scores 0. This
project's brief adds one more, about main_report.ipynb: "if any cell fails, the code evaluation
will receive 0 marks".

The rubric for the report also prices a measured threshold, "the overall accuracy of both forward
and reverse models ... exceeds 0.78", so that is parsed out of the notebook's own printed output
and asserted here too.

Run:  python tools/verify.py
"""
import ast
import hashlib
import io
import json
import os
import re
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-08/project-5-extending-addition-llm")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
SUBMISSION = f"{BASE}/submission"

NOTEBOOKS = ("LLMForward.ipynb", "LLMReverse.ipynb", "main_report.ipynb")
SUPPORT = ("LLMForward.pth", "LLMReverse.pth", "project5_testset.pkl",
           "LLMForward_history.json", "LLMReverse_history.json",
           "LLMForward_history_80k.json", "LLMReverse_history_80k.json",
           "figure_1_overview.pdf", "figure_2_by_length.pdf")

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def load(name):
    return json.load(io.open(f"{NBDIR}/{name}", encoding="utf-8"))


def stream_text(notebook):
    return "\n".join("".join(o.get("text", [])) for c in notebook["cells"]
                     for o in c.get("outputs", []) if o.get("output_type") == "stream")


notebooks = {name: load(name) for name in NOTEBOOKS}
printed = {name: stream_text(nb) for name, nb in notebooks.items()}
main = printed["main_report.ipynb"]

# ---------------------------------------------------------------- 1. file names are pass/fail
print("\n1. File naming (wrong names score 0)")
check("report is named project5_report.pdf", os.path.exists(f"{REPORT}/project5_report.pdf"))
for name in NOTEBOOKS + SUPPORT:
    check(f"notebook/{name} present", os.path.exists(f"{NBDIR}/{name}"))

# ---------------------------------------------------------------- 2. the report
print("\n2. Report (over 2 pages scores 0)")
from pypdf import PdfReader  # noqa: E402

pdf = PdfReader(f"{REPORT}/project5_report.pdf")
check("report is at most 2 pages", len(pdf.pages) <= 2, f"{len(pdf.pages)} pages")
tex = io.open(f"{REPORT}/project5_report.tex", encoding="utf-8").read()
check("report names both group members",
      "Karan Rooprai" in tex and "Nhu Hieu Nguyen" in tex)
check("report names the group", re.search(r"Group\s+\d", tex) is not None)
check("no unfilled template placeholders", not re.search(r"__[A-Z0-9_]+__", tex))
# The two part names and the four analysis headings the brief asks for, in its own words, plus
# "recommendations by synthesising strengths and limitations", which is easy to omit. Tasks 1 and
# 2 are run-in heads rather than sections, so they are tested as \textbf and not as \section.
for heading in (r"\textbf{Operation robustness.}", r"\textbf{Digit-level performance.}",
                r"\textbf{Direction effects.}", r"\textbf{Error patterns.}",
                r"\section{Method description}", r"\section{Results and analysis",
                r"\textbf{Task 1", r"\textbf{Task 2",
                r"\section{Strengths, limitations and recommendations}"):
    check(f"report contains '{heading}'", heading in tex)

PDFTOTEXT = "C:/Users/Admin/.conda/envs/tex/Library/bin/pdftotext.exe"
if os.path.exists(PDFTOTEXT):
    import subprocess
    import tempfile
    LEFT, RIGHT = 42.52, 552.76      # 1.5cm margins on a 595.28pt A4 page
    SLACK = 3.0                      # microtype protrusion plus glyph side bearings
    with tempfile.TemporaryDirectory() as tmp:
        boxes = f"{tmp}/bbox.xml"
        subprocess.run([PDFTOTEXT, "-bbox", f"{REPORT}/project5_report.pdf", boxes],
                       check=True, capture_output=True)
        words = re.findall(r'<word xMin="([\d.]+)"[^>]*xMax="([\d.]+)"[^>]*>([^<]*)</word>',
                           io.open(boxes, encoding="utf-8").read())
    outside = [(w, float(a), float(b)) for a, b, w in words
               if float(a) < LEFT - SLACK or float(b) > RIGHT + SLACK]
    check("no text spills outside the page margins", not outside,
          f"{len(words)} words" if not outside
          else f"{len(outside)} outside, first is {outside[0][0]!r}")
else:
    # Reporting this as a skip would let the script finish with ALL CHECKS PASSED while one rail
    # had never run, so the absence of the tool is itself the failure.
    check("no text spills outside the page margins", False, f"pdftotext not at {PDFTOTEXT}")

rendered = "\n".join(page.extract_text() for page in pdf.pages)
for dash in ("\u2014", "\u2013"):
    label = "em" if dash == chr(0x2014) else "en"
    check(f"no {label} dash in the rendered PDF", dash not in rendered)

# ---------------------------------------------------------------- 3. notebook hygiene
print("\n3. Notebook hygiene (no warnings, no errors)")
for name, notebook in notebooks.items():
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    stderr = [o for c in code for o in c.get("outputs", [])
              if o.get("output_type") == "stream" and o.get("name") == "stderr"]
    counts = [c.get("execution_count") for c in code]
    source = "\n".join("".join(c["source"]) for c in code)
    lines = [line for line in source.split("\n") if line.strip()]
    comments = [line for line in lines if line.strip().startswith("#")]

    check(f"{name}: no error outputs", not errors)
    check(f"{name}: no stderr output", not stderr,
          "" if not stderr else "".join(stderr[0].get("text", ""))[:160])
    check(f"{name}: every code cell executed, counts 1..N in order",
          counts == list(range(1, len(code) + 1)), f"{len(code)} cells")
    check(f"{name}: every code cell carries a comment",
          all(any(line.strip().startswith("#") for line in c["source"]) for c in code))
    check(f"{name}: comment ratio at least 0.12",
          len(comments) / len(lines) >= 0.12,
          f"{len(comments)}/{len(lines)} = {len(comments) / len(lines):.0%}")
    # PyTorch warns about the nested tensor fast path on every TransformerEncoder built with
    # batch_first=False, and that warning lands in stderr in the stored output.
    check(f"{name}: disables the nested tensor fast path",
          "enable_nested_tensor=False" in source)
    check(f"{name}: no tqdm progress bars (they write to stderr)", "tqdm" not in source)
    for dash in ("\u2014", "\u2013"):
        label = "em" if dash == chr(0x2014) else "en"
        check(f"{name}: no {label} dash", dash not in json.dumps(notebook))
    # A notebook declaring nbformat 4.5 must give every cell an id, or the validator writes
    # MissingIDFieldWarning to stderr at load time, before any cell has run.
    if notebook["nbformat_minor"] >= 5:
        missing = [i for i, c in enumerate(notebook["cells"]) if not c.get("id")]
        check(f"{name}: every cell carries the id nbformat 4.5 requires", not missing,
              f"{len(notebook['cells'])} cells" if not missing else f"{len(missing)} without one")
    # nbformat keeps the newline on every source line but the last. Dropping them concatenates
    # the cell onto one physical line, which is a syntax error no text-level check would see.
    check(f"{name}: source lines keep their newlines",
          all(all(line.endswith("\n") for line in c["source"][:-1])
              for c in notebook["cells"] if len(c["source"]) > 1))

main_source = "\n".join("".join(c["source"])
                        for c in notebooks["main_report.ipynb"]["cells"]
                        if c["cell_type"] == "code")
check("main_report.ipynb contains no training loop",
      not re.search(r"\.backward\(\)|optimizer\.step\(\)|model\.train\(\)", main_source))
check("main_report.ipynb loads weights with weights_only", "weights_only=True" in main_source)
for name in ("LLMForward.ipynb", "LLMReverse.ipynb"):
    source = "\n".join("".join(c["source"]) for c in notebooks[name]["cells"]
                       if c["cell_type"] == "code")
    check(f"{name}: writes its learning curves to json", "_history.json" in source)
    check(f"{name}: runs the reduced-data ablation", "model_small" in source)
# The two training runs have to agree with each other, because one checkpoint is compared
# against the other. main_report is held to the same device for a different reason: it is the
# notebook that gets run in the Jupyter environment, so the outputs it ships should be the ones
# that environment produced. Greedy decoding is what lets the numbers survive the move, and the
# CPU run they were checked against is quoted in the archive's own README.
devices = {n: re.findall(r"^device : (\S+)", printed[n], re.M)[:1] for n in NOTEBOOKS}
ran_on = [devices[n][0] for n in NOTEBOOKS if devices[n]]
check("all three notebooks ran on the same device",
      len(ran_on) == len(NOTEBOOKS) and len(set(ran_on)) == 1, f"devices {devices}")
# The README inside the archive states this in words, and a claim in a shipped file needs a rail
# behind it; without one, rebuilding on a machine with no GPU would quietly make it false.
check("the stored outputs come from the GPU node", ran_on[:1] == ["cuda"], f"device {ran_on[:1]}")

# ---------------------------------------------------------------- 4. the measured threshold
print("\n4. Performance (the rubric prices 'exceeds 0.78')")
for mode in ("Forward", "Reverse"):
    row = re.search(rf"^\s*{mode}\s*\|\s*overall\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([\d.]+)",
                    main, re.M)
    check(f"{mode} overall accuracy parsed from the notebook", row is not None)
    if row:
        check(f"{mode} exceeds 0.78", float(row.group(1)) > 0.78, row.group(1))
check("the notebook asserts the threshold itself",
      "is below the target" in main_source)

# ---------------------------------------------------------------- 5. the analysis is present
print("\n5. main_report ran every analysis the brief asks for")
check("held-out set is at least 10k and balanced",
      re.search(r"test examples (\d+)", main) is not None
      and int(re.search(r"test examples (\d+)", main).group(1)) >= 10000,
      re.search(r"test examples (\d+)  subtraction fraction ([\d.]+)", main).group(0)
      if re.search(r"test examples (\d+)  subtraction fraction ([\d.]+)", main) else "")
check("test set regenerates from the seed and does not overlap training",
      "regenerates from SEED exactly: True" in main and "prompt overlap 0" in main)
check("per-operation accuracy reported", "addition" in main and "subtraction" in main)
check("digit-level accuracy reported with its denominators",
      re.search(r"^\s*thousands\s*\|\s*\d+\s*\|", main, re.M) is not None)
check("carry and borrow cases reported",
      "with_carry" in main and "with_borrow" in main)
check("direction effects tested, not asserted", "exact McNemar two-sided p" in main)
check("operand width analysis present", "len(a)-len(b)" in main)
check("errors listed with a failure kind", re.search(r"\|\s+(single digit slip|wrong length)",
                                                     main) is not None)
check("learning curves summarised from the training runs",
      re.search(r"^\s*Forward\s*\|\s*\d+\s*\|", main, re.M) is not None)
check("machine readable summary dumped", '"mcnemar"' in main)

# ---------------------------------------------------------------- 6. traceability
print("\n6. Traceability: every number in the report is printed by main_report.ipynb")
body = tex.split(r"\begin{document}", 1)[1]
body = re.sub(r"%.*", "", body)
body = re.sub(r"\\includegraphics\[[^]]*\]", "", body)
body = re.sub(r"\\(vspace|setcounter|documentclass|usepackage|hfill)\{?[^}]*\}?", "", body)
body = re.sub(r"\[\d+pt\]", "", body)
body = re.sub(r"0\.98\\textwidth", "", body)
# The same text with every run of whitespace collapsed. Checks that look for a phrase use this;
# checks that need paragraph boundaries, such as the numeral-opening sweep below, use `body`.
# A line break in the .tex is an artefact of where the generator's source string wrapped, and it
# moves whenever a sentence is edited, so a phrase rail reading the raw body reports "absent" on
# a sentence that is present and stops testing anything.
flat = re.sub(r"\s+", " ", body)
numbers = set(re.findall(r"\d+\.\d+", body))
# Values named by the task or the page geometry rather than measured by the notebook.
DESIGN = {"0.78", "0.95", "0.5", "1.5", "0.9", "0.8", "0.6", "0.06", "0.94", "0.75", "0.98"}
untraceable = sorted(n for n in numbers - DESIGN if n not in main)
check("no decimal appears only in the report", not untraceable,
      f"untraceable: {untraceable}" if untraceable else f"{len(numbers)} checked")
ids = set(re.findall(r"n\d{8}", tex))
check("student ids appear in the compiled PDF",
      all(i in rendered.replace(" ", "") for i in ids), f"{sorted(ids)}")

# ---------------------------------------------------------------- 7. the archive
print("\n7. The code archive")
archive = f"{SUBMISSION}/project5_code.zip"
if os.path.exists(archive):
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    for name in NOTEBOOKS + SUPPORT:
        check(f"zip contains {name}", name in names)
    check("zip has no .DS_Store", not any(".DS_Store" in n for n in names))
    check("zip has no .ipynb_checkpoints", not any(".ipynb_checkpoints" in n for n in names))
    check("zip is flat (no nested folder)", not any("/" in n.rstrip("/") for n in names),
          f"{len(names)} entries")
    # Every check above reads notebook/, and this one reads the archive. Without an equality
    # assertion a notebook edited after the last build_zip.py run still passes everything
    # while the file a marker actually opens is the older one.
    with zipfile.ZipFile(archive) as zf:
        for name in NOTEBOOKS + SUPPORT:
            if name not in names:
                continue
            same = (hashlib.md5(zf.read(name)).hexdigest()
                    == hashlib.md5(io.open(f"{NBDIR}/{name}", "rb").read()).hexdigest())
            check(f"zip's {name} is the current notebook/ file", same)
    submitted = f"{SUBMISSION}/project5_report.pdf"
    check("submission report present", os.path.exists(submitted))
    if os.path.exists(submitted):
        # Same reasoning as the archive entries above: the report is built in report/ and copied
        # here, so a rebuild that skips the copy leaves the older file as the one that gets
        # uploaded. Comparing sizes would not catch it, as a wording fix of equal length can
        # produce a PDF of identical length.
        check("submission report is the current build",
              hashlib.md5(io.open(submitted, "rb").read()).hexdigest()
              == hashlib.md5(io.open(f"{REPORT}/project5_report.pdf", "rb").read()).hexdigest(),
              "run tools/build_zip.py to refresh the copy")
    check("archive is under 50 MB", os.path.getsize(archive) < 50 * 1024 * 1024,
          f"{os.path.getsize(archive) / 1024 / 1024:.1f} MB")
else:
    check("project5_code.zip built", False, "run tools/build_zip.py")

# ------------------------------------------------- 8. what the deliverables must not contain
print("\n8. Deliverables explain the subject, not the submission")

# A deliverable that talks about how it will be assessed is writing about itself. Two shapes of
# that: the vocabulary of marking, and the self-congratulating comparative that says a line is
# better than a worse line nobody wrote. "marks" is deliberately absent from the first pattern,
# because a caption may legitimately say a line marks a threshold, and the second is anchored to
# a closed list of verbs, because "errors rather than accuracies" is an ordinary comparison.
ASSESSMENT_WORDS = re.compile(
    r"\b(marker|markers|rubric|graded|grades|grading|feedback|deduct\w*)\b", re.I)
SELF_GRADING = re.compile(
    r"rather than (asserted|assumed|eyeballed|claimed|intended|taken on trust)", re.I)

sources = {name: "\n".join("".join(c["source"]) for c in nb["cells"])
           for name, nb in notebooks.items()}
sources["project5_report.tex"] = tex
if os.path.exists(f"{SUBMISSION}/project5_code.zip"):
    with zipfile.ZipFile(f"{SUBMISSION}/project5_code.zip") as zf:
        if "README.txt" in zf.namelist():
            sources["README.txt"] = zf.read("README.txt").decode("utf-8")

for name, source in sources.items():
    hits = ASSESSMENT_WORDS.findall(source) + [m.group(0) for m in SELF_GRADING.finditer(source)]
    check(f"{name} does not talk about how it is marked", not hits, f"found {sorted(set(hits))}")

# Every file a notebook names has to be a file the reader was sent. The archive is the whole
# world the marker sees, so a path that is not in it is a dead reference, however true it is
# here. The match is deliberately wide, catching any bare filename with a known extension.
FILENAME = re.compile(r"[A-Za-z0-9_./*-]+\.(?:ipynb|pth|pkl|json|pdf|py|zip|txt|csv)\b")
if os.path.exists(f"{SUBMISSION}/project5_code.zip"):
    with zipfile.ZipFile(f"{SUBMISSION}/project5_code.zip") as zf:
        shipped = set(zf.namelist())
    for name in NOTEBOOKS:
        # A wildcard stands for the files it expands to; the README uses them as a group name.
        named = {n for n in FILENAME.findall(sources[name]) if "*" not in n}
        dangling = sorted(n for n in named if n not in shipped)
        check(f"{name} names only files the archive ships", not dangling, f"missing {dangling}")

# An execute_result is the repr of a cell's last expression. It is never an intended output
# here, and a bare torch.manual_seed(SEED) produced one whose memory address differed between
# the two notebooks, which made the only irreproducible line in the submission.
for name, nb in notebooks.items():
    echoes = [c["id"] for c in nb["cells"]
              for o in c.get("outputs", []) if o.get("output_type") == "execute_result"]
    check(f"{name} echoes no object repr", not echoes, f"cells {echoes}")

# The report's table of example failures says it is drawn from the full error list. Nothing
# tested that claim, and the cell printed the first twelve. A caption can be false about the
# code in a way no number check can see, so the claim is turned into an assertion here: the
# rows the notebook printed must equal the count it reported.
for mode in ("Forward", "Reverse"):
    rows = len(re.findall(rf"^\s*{mode}\s*\|\s*\S+=\s*\|", main, re.MULTILINE))
    stated = re.search(rf"^{mode}: (\d+) errors in", main, re.MULTILINE)
    check(f"{mode}: every error is printed, not a sample",
          stated is not None and rows == int(stated.group(1)),
          f"{rows} rows printed against {stated.group(1) if stated else '?'} reported")

# Formatting the report actually carries.
check("Figure 1 is labelled like the caption package labels the rest",
      r"\textbf{Figure 1:}" in tex)
check("the report names the 0.78 target rather than describing it", "0.78" in body)
check("no Python exponent notation survives into the report",
      not re.search(r"\de[+-]\d", body), "use \\times 10^{}")

# Grouped thousands. Tables are excluded: an arithmetic answer in the error table is a number,
# not a count, and 1005 grouped as 1,005 would be wrong. Student ids are not counts either.
prose = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", body, flags=re.S)
prose = re.sub(r"n\d{8}", "", prose)
ungrouped = re.findall(r"(?<![\d.,])\d{4,}(?![\d,])", prose)
check("every count of a thousand or more is grouped in prose", not ungrouped,
      f"ungrouped: {sorted(set(ungrouped))}")

# A vector figure carries no idea of how big it will be printed, so the size a label renders at
# is its authored size times placement width over native width. Figure 1's 7pt labels were
# landing at 4.7pt. The smallest size the figure cell authors is what this measures.
TEXTWIDTH = 18 / 2.54                             # 1.5cm margins on A4
PLACED = {"figure_1_overview.pdf": 0.98 * TEXTWIDTH,        # spans both columns
          "figure_2_by_length.pdf": (TEXTWIDTH - 0.6 / 2.54) / 2}   # one column
main_src = sources["main_report.ipynb"]
for figure, placed in PLACED.items():
    path = f"{REPORT}/figures/{figure}"
    if not os.path.exists(path):
        check(f"{figure} present", False)
        continue
    native = float(PdfReader(path).pages[0].mediabox.width) / 72
    cell = re.search(rf"savefig\(\"{figure}\"", main_src)
    start = main_src.rfind("plt.subplots", 0, cell.start()) if cell else -1
    block = main_src[start:cell.start()] if start >= 0 else ""
    authored = [float(s) for s in re.findall(r"fontsize=(\d+(?:\.\d+)?)", block)]
    smallest = min(authored) if authored else 10.0
    rendered_pt = smallest * placed / native
    check(f"{figure}: no label renders below 6pt beside 10pt body text",
          rendered_pt >= 6.0,
          f"{smallest:g}pt authored at {native:.2f}in placed at {placed:.2f}in "
          f"renders {rendered_pt:.1f}pt")

# ---------------------------------------------------------------- 9. the fourth pass
# The report says the workshop's training loop was inherited and "only the changes described
# below differ". Three settings do differ, and for three passes none of them was described: a
# sentence can be true of everything it mentions and still leave the reader unable to check the
# run. These rails make the description itself a requirement.
print("\n9. The training configuration the report describes")
method = body.split(r"\section{Method description}", 1)[1].split(r"\section{Results", 1)[0]
for token in ("epochs", "cosine", "Batch size"):
    check(f"the method section states the {token} used", token in method)

# Epochs is read back from the stored curves rather than trusted: the validation history has one
# entry per epoch, and it is the file main_report.ipynb plots, so a re-run with a different
# schedule changes the number the sentence has to carry.
epoch_counts = {}
for mode in ("Forward", "Reverse"):
    curve = json.load(io.open(f"{NBDIR}/LLM{mode}_history.json", encoding="utf-8"))["val_seq"]
    epoch_counts[mode] = len(curve)
stated_epochs = re.search(r"(\d+) epochs rather than", method)
check("the epoch count in the report is the number of epochs actually trained",
      stated_epochs is not None
      and {int(stated_epochs.group(1))} == set(epoch_counts.values()),
      f"report {stated_epochs.group(1) if stated_epochs else '?'}, history {epoch_counts}")

# Batch size and epochs are design constants, so the traceability sweep over decimals cannot
# reach them. main_report.ipynb prints both, which is what lets them be checked the same way
# every other number in the report is: against what the notebook said.
printed_training = re.search(r"training: (\d+) epochs, batch size (\d+)", main)
stated_batch = re.search(r"Batch size stays\s+at (\d+)", method)
check("the batch size the report names is the one main_report.ipynb printed",
      printed_training is not None and stated_batch is not None
      and printed_training.group(2) == stated_batch.group(1),
      f"report {stated_batch.group(1) if stated_batch else '?'}, "
      f"notebook {printed_training.group(2) if printed_training else '?'}")
check("the epoch count the report names is the one main_report.ipynb printed",
      printed_training is not None and stated_epochs is not None
      and printed_training.group(1) == stated_epochs.group(1),
      f"report {stated_epochs.group(1) if stated_epochs else '?'}, "
      f"notebook {printed_training.group(1) if printed_training else '?'}")

# A comment claimed dropout 0.1 was "rather than the class default of 0.5". 0.5 is the WORKSHOP
# class's default, and that class also hard-codes 0.1 inside its encoder layer; the class in
# these notebooks declares 0.1 and passes one value to both. Nothing tested the comment against
# the class three lines below it, so this tests the thing the comment was about instead: every
# construction agrees with the declaration, in every notebook that builds a model.
print("\n10. Dropout is one value, declared and passed consistently")
for name in NOTEBOOKS:
    src = sources[name]
    declared = re.search(r"def __init__\(self, ntoken[^)]*dropout=([\d.]+)\)", src)
    calls = re.findall(r"TransformerModel\([^)]*?dropout=([\d.]+)", src)
    check(f"{name}: every TransformerModel call passes the declared default",
          declared is not None and calls and all(c == declared.group(1) for c in calls),
          f"declared {declared.group(1) if declared else '?'}, "
          f"{len(calls)} call(s) {sorted(set(calls))}")
    check(f"{name}: no comment names a dropout default the class does not declare",
          "class default of 0.5" not in src)

# Prose that reads wrong without being wrong. "the figures reproduce exactly" sits on a page
# holding Figure 1 and Figure 2; a paragraph opening with a numeral reads as a list item. The
# figure paths have to come out first, or "figures/figure_1_overview.pdf" fails the first rail.
print("\n11. Prose that a number check cannot see")
clean = re.sub(r"figures/\S+", "", prose)
check('the report says "numbers" rather than "figures" for what reproduces',
      not re.search(r"\bfigures\b", clean), "figures/ paths excluded")

# Commands and inline maths go before sentence boundaries are found: \texttt{50-73=} would
# otherwise look like a sentence and $p = 1.212 \times 10^{-7}$ like several.
sentences = re.sub(r"\$[^$]*\$", " ", clean)
sentences = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?(?:\{[^{}]*\})?", " ", sentences)
opens_numeric = []
for para in re.split(r"\n\s*\n", sentences):
    para = re.sub(r"\s+", " ", para).strip()
    if not para:
        continue
    if re.match(r"\d", para):
        opens_numeric.append(para[:40])
    opens_numeric += [m.group(1) for m in re.finditer(r"[.!?]\s+(\d[\d.,]*)", para)]
check("no sentence in the report opens with a numeral", not opens_numeric, f"{opens_numeric}")

# Figure 1's centre and right panels plot error counts rather than rates, and the caption gives
# the reason: every accuracy they could have shown instead is above 0.98, so the bars would be
# indistinguishable. 0.98 is in the DESIGN set above, so the traceability sweep steps over it and
# nothing else reads it. This does: every accuracy those two panels cover, out of the notebook,
# smallest first. A guard rather than a finding -- the claim holds comfortably on this run -- kept
# so that a future run which drags one split under the stated floor has to restate the caption.
carry_rows = re.findall(r"^\s*(?:with_carry|no_carry|with_borrow|no_borrow)\s*\|\s*\d+\s*\|"
                        r"\s*\d+\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*$", main, re.M)
place_rows = re.findall(r"^\s*(?:thousands|hundreds|tens|units|sign)\s*\|\s*\d+\s*\|"
                        r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*$", main, re.M)
panel_acc = [float(v) for row in carry_rows + place_rows for v in row]
said_floor = re.search(r"every accuracy here exceeds ([\d.]+)", flat)
check("the accuracy floor the Figure 1 caption gives is true of every split it covers",
      said_floor is not None and panel_acc
      and min(panel_acc) > float(said_floor.group(1)),
      f"caption says above {said_floor.group(1) if said_floor else '?'}, "
      f"smallest of {len(panel_acc)} is {min(panel_acc) if panel_acc else '?'}")

# The barrier sentence explains the reduced reverse run's failure to converge by the barrier it
# has just described. One run of one seed cannot carry "which is also why": the limitations
# paragraph says as much two columns later, calling the convergence difference a single
# observation. The sentence is allowed to offer the explanation, not to assert it.
check("the barrier sentence offers its explanation rather than asserting it",
      "which is also why" not in flat, "hedged as 'which would also explain why'")

# The strengths paragraph lists what the two models hold in common, which is what makes direction
# the only variable. The training data belongs on that list -- same seed, same split sizes, so
# literally the same 150,000 prompts, with only the target string reversed -- and it is the
# strongest item on it. Written as an equivalence: the report claims it exactly when the two
# notebooks really do agree, so a future divergence deletes the claim rather than outliving it.
strengths = flat.split(r"\section{Strengths", 1)[-1]
shared = {}
for name in ("LLMForward.ipynb", "LLMReverse.ipynb"):
    seed = re.search(r"SEED\s*=\s*(\d+)", sources[name])
    split = re.search(r"train_size,\s*val_size,\s*test_size\s*=\s*([\d,\s]+)", sources[name])
    shared[name] = (seed.group(1) if seed else "?",
                    re.sub(r"\s+", " ", split.group(1)).strip() if split else "?")
same_data = len(set(shared.values())) == 1 and "?" not in shared["LLMForward.ipynb"]
check("the report names training data among the shared settings exactly when it is shared",
      ("training data" in strengths) == same_data,
      f"Forward {shared['LLMForward.ipynb']}, Reverse {shared['LLMReverse.ipynb']}")

# The barrier sentence names a flat phase, the epoch it ends and the two thresholds crossed
# after it. All four come out of the stored curve in build_report.py, so they are re-derived
# here from the same file, independently, and compared with what the sentence says. "below a
# quarter" is read strictly: the peak before the rise must really be under a quarter of final.
print("\n12. The barrier sentence against the curve it describes")
rev = json.load(io.open(f"{NBDIR}/LLMReverse_history.json", encoding="utf-8"))["val_seq"]
crossed = {t: next((i + 1 for i, v in enumerate(rev) if v >= t), None) for t in (0.78, 0.95)}
said_epochs = re.search(r"passes 0\.78 at epoch (\d+) and 0\.95 at epoch (\d+)", body)
check("the epochs the barrier sentence names are the reverse curve's own crossings",
      said_epochs is not None
      and [int(g) for g in said_epochs.groups()] == [crossed[0.78], crossed[0.95]],
      f"curve {crossed[0.78]}/{crossed[0.95]}, "
      f"report {list(said_epochs.groups()) if said_epochs else '?'}")
FLAT_WORDS = {"a tenth": 0.10, "a quarter": 0.25}
said_flat = re.search(r"stays below (a tenth|a quarter) of its final accuracy through epoch "
                      r"(\d+)", body)
flat_ratio = max(rev[:crossed[0.78] - 1]) / rev[-1] if crossed[0.78] else 1.0
check("the flat phase the barrier sentence describes is true of the curve",
      said_flat is not None
      and flat_ratio < FLAT_WORDS[said_flat.group(1)]
      and int(said_flat.group(2)) == crossed[0.78] - 1,
      f"peak before the rise is {flat_ratio:.3f} of final, "
      f"report says below {said_flat.group(1) if said_flat else '?'} "
      f"through epoch {said_flat.group(2) if said_flat else '?'}")

# The closing recommendation makes three claims about the direction it picks, and two of them are
# measurements sitting a column earlier in the same report: how fast it converged, and what its
# worst group of examples costs it. Both are re-read here from the notebook rather than trusted,
# because a sentence that recommends something on the strength of a number is the sentence a
# reader checks first.
full_95 = {}
for mode in ("Forward", "Reverse"):
    rows = re.findall(rf"^\s*{mode}\s*\|\s*(\d+)\s*\|\s*\S+\s*\|\s*(\S+)\s*\|\s*[\d.]+\s*$",
                      main, re.M)
    rows.sort(key=lambda r: -int(r[0]))          # the full training set is the larger one
    full_95[mode] = rows[0][1] if rows else "?"
picked = re.search(r"choose (Forward|Reverse):", flat)
other = {"Forward": "Reverse", "Reverse": "Forward"}.get(picked.group(1)) if picked else None
faster = (picked is not None and full_95[picked.group(1)].isdigit() and full_95[other].isdigit()
          and int(full_95[picked.group(1)]) < int(full_95[other]))
check("the recommendation claims faster convergence only when the curves show it",
      ("converges in fewer epochs" in flat) == faster,
      f"0.95 at epoch {full_95['Forward']} Forward, {full_95['Reverse']} Reverse; "
      f"report picks {picked.group(1) if picked else '?'}")

# Both models concentrate their errors in a group of 92 examples, and they are different groups of
# 92: Forward's single-digit answers, Reverse's two-digit width mismatch. Equal denominators mean
# the two error counts are the whole comparison, so the recommendation states them and this reads
# both back out of the tables that printed them.
len1_row = re.search(r"^\s*1 digit\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*\d+\s*$", main, re.M)
width = {}
for m in re.finditer(r"^\s*(-?\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*[\d.]+\s*\|\s*(\d+)\s*\|"
                     r"\s*[\d.]+\s*$", main, re.M):
    width[int(m.group(1))] = m.group(2)
hardest = re.search(r"its hardest case costs it (\d+) errors where (?:Forward|Reverse)'s "
                    r"costs (\d+)", flat)
expected = ([len1_row.group(1), width[max(width)]] if len1_row and width else None)
check("the two error counts the recommendation compares are the printed ones",
      hardest is not None and expected is not None and list(hardest.groups()) == expected,
      f"notebook {expected}, report {list(hardest.groups()) if hardest else 'sentence absent'}")

# ---------------------------------------------------------------- 13. nothing defined for show
# V14 deleted overall_counts() because it was defined and never called. evaluate() and
# get_batch() survived that pass for fidelity to the workshop, which is not a reason a reader of
# this notebook can act on: a function with no caller is a question the code does not answer.
# They are called now. This rail is what stops the next one from accumulating.
print("\n13. Every function in main_report.ipynb has a caller")
main_code = "\n".join("".join(c["source"]) for c in notebooks["main_report.ipynb"]["cells"]
                      if c["cell_type"] == "code")
# Over the syntax tree rather than by regex, because two of these functions are handed to a loop
# as values and never written with parentheses after them: searching for "name(" calls those dead
# when they are not. In the tree both uses are a Name node, and a method called on an object is
# an Attribute, which is how the tokenizer's own methods are reached.
tree = ast.parse(main_code)
CALLED_BY_TORCH = {"forward", "__init__"}   # invoked by nn.Module, never by name here
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)} - CALLED_BY_TORCH
referenced = ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
              | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})
dead = sorted(defined - referenced)
check("no function is defined without being used", not dead,
      f"{len(defined)} functions, dead: {dead}")

# ---------------------------------------------------------------- 14. two ways of scoring agree
# The notebook now scores the same predictions twice: over tokens, as the workshop did, and over
# integers, as Task 3 asks. They answer the same question, so a disagreement is a bug in one of
# them, and the report's claim that they match must not outlive the measurement of it.
print("\n14. Token-level scoring beside integer-level scoring")
agreements = []
token_rates = []
for mode in ("Forward", "Reverse"):
    token = re.search(rf"^token level \|\s*{mode}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
                      rf"\s*([\d.]+)\s*$", main, re.M)
    overall = re.search(rf"^\s*{mode}\s*\|\s*overall\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([\d.]+)",
                        main, re.M)
    check(f"{mode}: the workshop's token-level measure is reported", token is not None,
          f"digit {token.group(1)}, sequence {token.group(2)}" if token else "")
    token_rates.append(token.group(1) if token else "?")
    same = token is not None and overall is not None and token.group(2) == overall.group(1)
    agreements.append(same)
    check(f"{mode}: token-level sequence accuracy equals the integer-level accuracy", same,
          f"tokens {token.group(2) if token else '?'}, "
          f"integers {overall.group(1) if overall else '?'}")
check("the report claims the two scorings match only when they do",
      ("sequence accuracy reproduces the overall rates exactly" in body) == all(agreements),
      f"agree {agreements}")

# The sentence carrying these two rates follows one that pairs its numbers as addition and
# subtraction, so an unlabelled pair inherits that reading and is then wrong. Naming the models
# is the fix; this is what keeps the labels attached to the right numbers.
named = re.search(r"digit accuracy is ([\d.]+) for Forward and ([\d.]+) for Reverse", flat)
check("the token-level sentence names which model each rate belongs to",
      named is not None and list(named.groups()) == token_rates,
      f"notebook {token_rates}, report "
      f"{list(named.groups()) if named else 'the pair is unlabelled'}")

# --------------------------------------------- 15. floats, step counts, and two ways to count
print("\n15. Every float is cited, and the two place-value counts reconcile")

# A numbered float the prose never points at is a float the reader meets without being told why
# it is there. Two things make the numbering non-obvious. Figure 1 is hand-lettered inside the
# title block rather than floated, so the figure counter is pushed past it and the first real
# figure is Figure 2; and that counter has to be read from `tex`, because section 6 strips
# \setcounter out of `body` with the rest of the layout macros. A citation is required to carry
# the unbreakable space, which every genuine cross-reference here uses and the hand-lettered
# label does not, so the label cannot be mistaken for the prose citing itself.
prose = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
prose = re.sub(r"\s+", " ", prose)
offset = re.search(r"\\setcounter\{figure\}\{(\d+)\}", tex)
floats = []
for kind, first in (("figure", int(offset.group(1)) + 1 if offset else 1), ("table", 1)):
    for n, _ in enumerate(re.finditer(r"\\begin\{" + kind + r"\*?\}", body), start=first):
        floats.append((kind.capitalize(), n))
uncited = [f"{k} {n}" for k, n in floats if f"{k}~{n}" not in prose]
check("every numbered float is cited in the prose", not uncited,
      f"uncited: {uncited}" if uncited else "cited: " + ", ".join(f"{k} {n}" for k, n in floats))

# The ablation sentence says the two training-set sizes agree once measured in optimiser updates.
# When the two counts really are the same number, printing it twice with "against" between them
# reads as a slip rather than as the agreement the sentence is about. Written as an equivalence so
# that a run where the counts differ gets the two numbers back instead of a false "either set".
steps = re.search(r"passes 0\.95 after ([\d,]+) updates on (either set|the reduced set "
                  r"against ([\d,]+) on the full one)", flat)
if steps is None:
    check("the ablation sentence states the matched update count", False, "sentence not found")
else:
    equal_counts = steps.group(3) is None or steps.group(1) == steps.group(3)
    check("the sentence says 'either set' exactly when the two update counts are equal",
          (steps.group(2) == "either set") == equal_counts, f"reads {steps.group(0)[:72]!r}")

# Two counts of the same place value, both correct under their own definition, and the only thing
# holding them apart is a clause in the prose. Figure 1's third panel plots wrong digits from the
# per-place accuracy, which marks a place wrong when a short prediction cannot supply it at all.
# The sentence in Error patterns counts only errors that kept the right number of digits, which is
# why it reports a smaller number for the same place. The gap can never be negative and can never
# exceed the wrong-length errors; if it ever does, the qualifying clause has stopped being true
# and the figure and the sentence contradict each other in front of the reader.
places = re.search(r"Reverse wrong places \(0 = units\): \{([^}]*)\}", main)
rev_places = dict(re.findall(r"'([^']+)':\s*(\d+)", places.group(1))) if places else {}
tens_row = re.search(r"^\s*tens\s*\|\s*(\d+)\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*$", main, re.M)
said = re.search(r"right number of digits, (\d+) are wrong at the tens digit", flat)
saidlen = re.search(r"(\d+) predictions have the wrong length", flat)
if tens_row and said and saidlen and "wrong length" in rev_places:
    # Reconstructed from an accuracy printed to four places, so allow one for the rounding.
    panel = round((1 - float(tens_row.group(2))) * int(tens_row.group(1)))
    low = int(said.group(1))
    check("the figure's wrong-digit count at the tens reconciles with the sentence's",
          low - 1 <= panel <= low + int(saidlen.group(1)) + 1,
          f"figure {panel}, sentence {low}, wrong-length {saidlen.group(1)}")
else:
    check("the figure's wrong-digit count at the tens reconciles with the sentence's", False,
          "could not read one of the three numbers")

# The same sentence's subset cannot be larger than the population it is drawn from: the errors
# that kept the right number of digits are the total errors less the wrong-length ones.
total_rev = re.search(r"Of Reverse's (\d+) errors", flat)
if total_rev and said and saidlen:
    right_length = int(total_rev.group(1)) - int(saidlen.group(1))
    check("the tens-slip count does not exceed the right-length errors it is drawn from",
          int(said.group(1)) <= right_length,
          f"{said.group(1)} of {right_length} right-length errors")
else:
    check("the tens-slip count does not exceed the right-length errors it is drawn from", False,
          "could not read the error totals")

# The Set-up paragraph says the validation split only tracks convergence and that no model is
# selected on it. That is a claim about what the training notebooks do, and it is the claim a
# marker working from the unit's own deduction list checks when it asks whether results were
# reported after fitting the best model on training plus validation. Written as an equivalence:
# the sentence stands exactly while both notebooks still train for a fixed schedule and save the
# final weights unconditionally. Introduce early stopping or a best-checkpoint rule and the
# sentence has to go, rather than quietly becoming false.
selection = {}
for name in ("LLMForward.ipynb", "LLMReverse.ipynb"):
    code = sources[name]
    selection[name] = (
        re.search(r"\bbest[_a-z]*\s*=", code) is None          # no running best to compare against
        and "patience" not in code                             # no early-stopping counter
        and not re.search(r"early[_ ]?stop", code, re.I)
        and len(re.findall(r"torch\.save\(", code)) == 1       # one save, not one per improvement
        and re.search(r"if[^\n]*val[^\n]*:\s*\n\s*torch\.save", code) is None)  # and unconditional
no_selection = all(selection.values())
check("the report says nothing is selected on validation exactly when nothing is",
      ("no model is selected on it" in flat) == no_selection,
      f"Forward {selection['LLMForward.ipynb']}, Reverse {selection['LLMReverse.ipynb']}")

print("\n" + ("ALL CHECKS PASSED" if not failures
             else f"{len(failures)} CHECK(S) FAILED: " + "; ".join(failures)))
raise SystemExit(1 if failures else 0)
