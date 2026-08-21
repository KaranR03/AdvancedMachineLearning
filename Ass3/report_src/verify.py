r"""Every check that must pass before Project 3 is uploaded.

Adapted from project 2's tools/verify.py. The two cliff edges the unit states explicitly -- a
report over 2 pages scores 0, and a code archive whose file names are wrong scores 0 -- are
asserted mechanically here rather than eyeballed.

Run:  python tools/verify.py
"""
import io
import json
import os
import re
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-05/project-3-aircraft-classification")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
SUBMISSION = f"{BASE}/submission"

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def load(name):
    return json.load(io.open(f"{NBDIR}/{name}", encoding="utf-8"))


def printed_text(notebook):
    return "\n".join("".join(o.get("text", "")) for c in notebook["cells"]
                     for o in c.get("outputs", []) if o.get("output_type") == "stream")


development = load("development.ipynb")
main_report = load("main_report.ipynb")
printed = printed_text(main_report)

# ---------------------------------------------------------------- 1. file names are pass/fail
print("\n1. File naming (wrong names score 0)")
check("report is named project3_report.pdf", os.path.exists(f"{REPORT}/project3_report.pdf"))
for name in ("development.ipynb", "main_report.ipynb", "best_model.pth", "histories.pkl"):
    check(f"notebook/{name} present", os.path.exists(f"{NBDIR}/{name}"))

# ---------------------------------------------------------------- 2. the report
print("\n2. Report (over 2 pages scores 0)")
from pypdf import PdfReader  # noqa: E402

pdf = PdfReader(f"{REPORT}/project3_report.pdf")
check("report is at most 2 pages", len(pdf.pages) <= 2, f"{len(pdf.pages)} pages")
tex = io.open(f"{REPORT}/project3_report.tex", encoding="utf-8").read()
check("report names both group members",
      "Karan Rooprai" in tex and "Nhu Hieu Nguyen" in tex)
check("report names the group", re.search(r"Group\s+\d", tex) is not None)
check("no unfilled template placeholders",
      not re.search(r"\[(X|Full Name|Student ID)\]", tex))
check("report states the refit on train+validation",
      "training plus" in tex and "trainval" in tex)

# ---------------------------------------------------------------- 3. notebook hygiene
print("\n3. Notebook hygiene (no warnings, no errors)")
for name, notebook in (("development.ipynb", development), ("main_report.ipynb", main_report)):
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    stderr = [o for c in code for o in c.get("outputs", [])
              if o.get("output_type") == "stream" and o.get("name") == "stderr"]
    check(f"{name}: no error outputs", not errors)
    check(f"{name}: no stderr output", not stderr)
    counts = [c.get("execution_count") for c in code if c.get("execution_count") is not None]
    check(f"{name}: execution counts strictly increasing",
          counts == sorted(counts) and len(counts) == len(set(counts)),
          f"{counts}")

# main_report must be fully executed: it is the notebook the grader runs.
main_code = [c for c in main_report["cells"] if c["cell_type"] == "code"]
check("main_report.ipynb: every code cell executed",
      all(c.get("execution_count") is not None for c in main_code),
      f"{len(main_code)} cells")
check("main_report.ipynb does not retrain",
      not re.search(r"\.backward\(\)|optimizer\.step\(\)|\.train\(\)",
                    "\n".join("".join(c["source"]) for c in main_code)))
check("main_report.ipynb loads the checkpoint rather than building one",
      "best_model.pth" in "\n".join("".join(c["source"]) for c in main_code))

# ---------------------------------------------------------------- 4. the target
print("\n4. The stated target")
avg = float(re.search(r"Average per-class accuracy\s+:\s+(\d\.\d+)", printed).group(1))
check("average per-class accuracy above 0.75", avg > 0.75, f"{avg:.4f}")
check("main_report printed PASS", "PASS" in printed)

# ---------------------------------------------------------------- 5. traceability
print("\n5. Traceability: every number in the report is printed by main_report.ipynb")
body = tex.split(r"\begin{document}", 1)[1]
body = re.sub(r"%.*", "", body)                       # drop LaTeX comments
body = re.sub(r"\\includegraphics\[[^]]*\]", "", body)  # drop figure sizing
body = re.sub(r"\\(vspace|setcounter|documentclass|usepackage)\{?[^}]*\}?", "", body)
body = re.sub(r"\[\d+pt\]", "", body)                 # drop \\[2pt] spacing
numbers = set(re.findall(r"\d+\.\d+", body))
# Numbers that are part of the method description rather than a measured result. They are stated
# in the notebooks' own markdown and in development.ipynb's code, not printed as output.
DESIGN = {"0.001", "0.0001", "0.75", "0.87", "0.2", "0.01", "18.0", "34.0"}
untraceable = sorted(n for n in numbers - DESIGN if n not in printed)
check("no unexplained decimal appears only in the report", not untraceable,
      f"untraceable: {untraceable}" if untraceable else "")
for token in ("1,331", "1,064", "267", "669"):
    plain = token.replace(",", "")
    if token in body:
        check(f"count {token} is printed by main_report", plain in printed)

# ---------------------------------------------------------------- 6. the archive
print("\n6. The code archive")
archive = f"{SUBMISSION}/project3_code.zip"
if os.path.exists(archive):
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    check("zip contains development.ipynb", "development.ipynb" in names)
    check("zip contains main_report.ipynb", "main_report.ipynb" in names)
    check("zip contains best_model.pth", "best_model.pth" in names)
    check("zip contains histories.pkl", "histories.pkl" in names)
    check("zip has no .DS_Store", not any(".DS_Store" in n for n in names))
    check("zip has no .ipynb_checkpoints", not any(".ipynb_checkpoints" in n for n in names))
    check("zip does not ship the dataset",
          not any(n.lower().endswith((".jpg", ".jpeg", ".png")) for n in names), f"{len(names)} entries")
    check("zip is flat (no nested folder)", not any("/" in n.rstrip("/") for n in names))
    check("submission report present", os.path.exists(f"{SUBMISSION}/project3_report.pdf"))
else:
    check("project3_code.zip built", False, "run tools/build_zip.py")

print("\n" + ("ALL CHECKS PASSED" if not failures
             else f"{len(failures)} CHECK(S) FAILED: " + "; ".join(failures)))
raise SystemExit(1 if failures else 0)
