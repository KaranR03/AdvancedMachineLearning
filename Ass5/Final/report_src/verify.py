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

print("\n" + ("ALL CHECKS PASSED" if not failures
             else f"{len(failures)} CHECK(S) FAILED: " + "; ".join(failures)))
raise SystemExit(1 if failures else 0)
