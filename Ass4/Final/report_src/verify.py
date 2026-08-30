r"""Every check that must pass before Project 4 is uploaded.

Adapted from project 3's tools/verify.py. The cliff edges the unit states explicitly are asserted
mechanically here rather than eyeballed: a report over 2 pages scores 0, wrong file names score 0,
and code that does not run scores 0.

Run:  python tools/verify.py
"""
import hashlib
import io
import json
import os
import re
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-06/project-4-ai-auditing")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
SUBMISSION = f"{BASE}/submission"

CASES = (1, 2, 3)
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


notebooks = {case: load(f"case{case}.ipynb") for case in CASES}
printed = {case: stream_text(nb) for case, nb in notebooks.items()}
all_printed = "\n".join(printed.values())

# ---------------------------------------------------------------- 1. file names are pass/fail
print("\n1. File naming (wrong names score 0)")
check("report is named project4_report.pdf", os.path.exists(f"{REPORT}/project4_report.pdf"))
for case in CASES:
    check(f"notebook/case{case}.ipynb present", os.path.exists(f"{NBDIR}/case{case}.ipynb"))
    check(f"notebook/case{case}_predictions.pkl present",
          os.path.exists(f"{NBDIR}/case{case}_predictions.pkl"))

# ---------------------------------------------------------------- 2. the report
print("\n2. Report (over 2 pages scores 0)")
from pypdf import PdfReader  # noqa: E402

pdf = PdfReader(f"{REPORT}/project4_report.pdf")
check("report is at most 2 pages", len(pdf.pages) <= 2, f"{len(pdf.pages)} pages")
tex = io.open(f"{REPORT}/project4_report.tex", encoding="utf-8").read()
check("report names both group members",
      "Karan Rooprai" in tex and "Nhu Hieu Nguyen" in tex)
check("report names the group", re.search(r"Group\s+\d", tex) is not None)
check("no unfilled template placeholders",
      not re.search(r"\b(?:CONE|CTWO|CTHREE)_[A-Z0-9]+", tex))
# The brief names the three things each case must cover, so the headings use its wording
# verbatim. Matching the bold label rather than a bare word means the check cannot be
# satisfied by the word appearing anywhere in a sentence.
for label in (r"\textbf{Diagnosis:", r"\textbf{Experiments and evidence.}",
              r"\textbf{Recommendations.}"):
    check(f"report heads every case with '{label}'", tex.count(label) >= 3,
          f"{tex.count(label)} occurrences")

# An overfull \hbox is a tectonic warning, not a page count, so nothing above could see Table 1
# sitting 9.5pt wider than its column and overhanging the gutter. Measuring the rendered word
# boxes against the geometry catches any such overflow whatever caused it. pdftotext ships with
# the same conda environment as tectonic; where it is absent the check reports itself skipped
# rather than passing silently.
PDFTOTEXT = "C:/Users/Admin/.conda/envs/tex/Library/bin/pdftotext.exe"
if os.path.exists(PDFTOTEXT):
    import subprocess
    import tempfile
    LEFT, RIGHT = 42.52, 552.76      # 1.5cm margins on a 595.28pt A4 page
    # microtype protrudes hyphens and full stops a little past the margin on purpose, and
    # pdftotext's word boxes include the glyph side bearings, so the measured overhang is
    # about 2pt on a page that is set correctly. The overfull table this check exists to
    # catch was 9.5pt wide of its column, 4.7pt each side once centred, so 3pt separates
    # the two cases with room on both sides of the line.
    SLACK = 3.0
    with tempfile.TemporaryDirectory() as tmp:
        boxes = f"{tmp}/bbox.xml"
        subprocess.run([PDFTOTEXT, "-bbox", f"{REPORT}/project4_report.pdf", boxes],
                       check=True, capture_output=True)
        words = re.findall(r'<word xMin="([\d.]+)"[^>]*xMax="([\d.]+)"[^>]*>([^<]*)</word>',
                           io.open(boxes, encoding="utf-8").read())
    outside = [(w, float(a), float(b)) for a, b, w in words
               if float(a) < LEFT - SLACK or float(b) > RIGHT + SLACK]
    check("no text spills outside the page margins", not outside,
          f"{len(words)} words" if not outside
          else f"{len(outside)} outside, first is {outside[0][0]!r}")
else:
    # Not a failure: this rail measures the compiled PDF with a binary from the conda
    # environment that also holds tectonic, which a machine that only reads the report
    # will not have. Say so rather than failing the whole gate on a missing tool.
    print("  SKIP  no text spills outside the page margins   pdftotext not found")

rendered = "\n".join(page.extract_text() for page in pdf.pages)
for dash in ("\u2014", "\u2013"):
    check(f"no {'em' if dash == chr(0x2014) else 'en'} dash in the rendered PDF", dash not in rendered)

# ---------------------------------------------------------------- 3. notebook hygiene
print("\n3. Notebook hygiene (no warnings, no errors)")
for case in CASES:
    notebook = notebooks[case]
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    stderr = [o for c in code for o in c.get("outputs", [])
              if o.get("output_type") == "stream" and o.get("name") == "stderr"]
    counts = [c.get("execution_count") for c in code]
    source = "\n".join("".join(c["source"]) for c in code)
    check(f"case{case}: no error outputs", not errors)
    check(f"case{case}: no stderr output", not stderr)
    check(f"case{case}: every code cell executed, counts 1..N in order",
          counts == list(range(1, len(code) + 1)), f"{len(code)} cells")
    check(f"case{case}: does not retrain",
          not re.search(r"\.backward\(\)|optimizer\.step\(\)|model\.train\(\)", source))
    check(f"case{case}: verifies which checkpoint it loaded", "EXPECTED_MD5" in source)
    # np.trapezoid does not exist before numpy 2.0. The assessment names the base Hub node, whose
    # numpy version is not published, so the safe choice is sklearn rather than either spelling.
    check(f"case{case}: no version-fragile numpy integration helper",
          "np.trapezoid" not in source and "np.trapz" not in source)
    check(f"case{case}: pins antialias on the tensor resize", "antialias=True" in source)
    for dash in ("\u2014", "\u2013"):
        check(f"case{case}: no {'em' if dash == chr(0x2014) else 'en'} dash",
              dash not in json.dumps(notebook))
    # A notebook declaring nbformat 4.5 must give every cell an id. When it does not, the
    # validator writes MissingIDFieldWarning to stderr at load time, before any cell has run, so
    # the stored-output checks above structurally cannot see it.
    if notebook["nbformat_minor"] >= 5:
        missing = [i for i, c in enumerate(notebook["cells"]) if not c.get("id")]
        check(f"case{case}: every cell carries the id nbformat 4.5 requires", not missing,
              f"{len(missing)} without one" if missing else f"{len(notebook['cells'])} cells")
    images = sum(1 for c in code for o in c.get("outputs", [])
                 if "image/png" in o.get("data", {}))
    check(f"case{case}: figures rendered", images >= 6, f"{images} images")

# ---------------------------------------------------------------- 4. the diagnosis is evidenced
print("\n4. Each case ran the experiment its diagnosis rests on")
check("case1 measured image statistics per folder", "sharpness" in printed[1])
check("case1 swept blur across the internal set and reached the field result",
      "blur sigma" in printed[1] and "closest match" in printed[1])
check("case1 ran the brightness control as well as the blur",
      "brightness alone reaches only" in printed[1])
check("case2 stratified accuracy by attribute combination",
      "seen internally" in printed[2] and "women, eyeglasses" in printed[2])
check("case3 fitted a temperature out of sample", "fitted temperature" in printed[3])
check("case3 swept the coverage frontier", "reachable at any coverage" in printed[3])
for case in CASES:
    check(f"case{case} printed AUC and ECE rather than only drawing them",
          "AUC" in printed[case] and "ECE" in printed[case])

# ---------------------------------------------------------------- 5. traceability
print("\n5. Traceability: every number in the report is printed by a notebook")
body = tex.split(r"\begin{document}", 1)[1]
body = re.sub(r"%.*", "", body)
body = re.sub(r"\\includegraphics\[[^]]*\]", "", body)
body = re.sub(r"\\(vspace|setcounter|documentclass|usepackage)\{?[^}]*\}?", "", body)
body = re.sub(r"\[\d+pt\]", "", body)
numbers = set(re.findall(r"\d+\.\d+", body))
# Thresholds and targets named by the task rather than measured by it.
DESIGN = {"0.90", "0.99", "0.5"}
untraceable = sorted(n for n in numbers - DESIGN if n not in all_printed)
check("no decimal appears only in the report", not untraceable,
      f"untraceable: {untraceable}" if untraceable else f"{len(numbers)} checked")

# ---------------------------------------------------------------- 6. the archive
print("\n6. The code archive")
archive = f"{SUBMISSION}/project4_code.zip"
if os.path.exists(archive):
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    for case in CASES:
        check(f"zip contains case{case}.ipynb", f"case{case}.ipynb" in names)
        check(f"zip contains case{case}_predictions.pkl",
              f"case{case}_predictions.pkl" in names)
    check("zip has no .DS_Store", not any(".DS_Store" in n for n in names))
    check("zip has no .ipynb_checkpoints", not any(".ipynb_checkpoints" in n for n in names))
    check("zip does not ship the supplied image sets",
          not any(n.lower().endswith((".jpg", ".jpeg")) for n in names), f"{len(names)} entries")
    check("zip is flat (no nested folder)", not any("/" in n.rstrip("/") for n in names))
    # Every check above reads notebook/, and this one reads the archive. Without an equality
    # assertion between them a notebook edited after the last build_zip.py run still passes
    # everything while the file a marker actually opens is the older one.
    with zipfile.ZipFile(archive) as zf:
        for name in [f"case{c}.ipynb" for c in CASES] + [f"case{c}_predictions.pkl" for c in CASES]:
            same = (hashlib.md5(zf.read(name)).hexdigest()
                    == hashlib.md5(io.open(f"{NBDIR}/{name}", "rb").read()).hexdigest())
            check(f"zip's {name} is the current notebook/ file", same)
    check("submission report present", os.path.exists(f"{SUBMISSION}/project4_report.pdf"))
    check("archive is under 50 MB", os.path.getsize(archive) < 50 * 1024 * 1024,
          f"{os.path.getsize(archive) / 1024 / 1024:.1f} MB")
else:
    check("project4_code.zip built", False, "run tools/build_zip.py")

print("\n" + ("ALL CHECKS PASSED" if not failures
             else f"{len(failures)} CHECK(S) FAILED: " + "; ".join(failures)))
raise SystemExit(1 if failures else 0)
