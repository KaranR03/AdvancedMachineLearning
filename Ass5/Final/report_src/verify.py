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
# The brief's own four analysis headings, plus the rubric's "recommendations by synthesising
# strengths and limitations", which is named in the full-marks tier and is easy to omit.
for heading in (r"\textbf{Operation robustness.}", r"\textbf{Digit-level performance.}",
                r"\textbf{Direction effects.}", r"\textbf{Error patterns.}",
                r"\section{Task 1", r"\section{Task 2", r"\section{Task 3",
                r"\section{Limitations and recommendations}"):
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
    print("  SKIP  no text spills outside the page margins   pdftotext not found")

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
# against the other. main_report is deliberately not held to the same device: it is executed
# wherever it is opened, and greedy decoding is what makes that produce the same numbers.
devices = {n: re.findall(r"^device : (\S+)", printed[n], re.M)[:1] for n in NOTEBOOKS}
trained_on = [devices[n][0] for n in ("LLMForward.ipynb", "LLMReverse.ipynb") if devices[n]]
check("both training notebooks ran on the same device", len(set(trained_on)) == 1,
      f"training {trained_on}, main_report {devices['main_report.ipynb']}")

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
    check("submission report present", os.path.exists(f"{SUBMISSION}/project5_report.pdf"))
    check("archive is under 50 MB", os.path.getsize(archive) < 50 * 1024 * 1024,
          f"{os.path.getsize(archive) / 1024 / 1024:.1f} MB")
else:
    check("project5_code.zip built", False, "run tools/build_zip.py")

print("\n" + ("ALL CHECKS PASSED" if not failures
             else f"{len(failures)} CHECK(S) FAILED: " + "; ".join(failures)))
raise SystemExit(1 if failures else 0)
