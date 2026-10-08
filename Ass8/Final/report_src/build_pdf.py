r"""Compile the report and fail on the warnings a page count cannot see.

The page count in tools/verify.py catches the cliff edge the unit states, a report over two pages.
It cannot catch a line or a table wider than the column it sits in: TeX reports that as an
"Overfull \hbox" warning and still produces a two-page PDF, so every other check passes while the
table overhangs the gutter. Project 4's Table 1 sat 9.5pt wide of its column that way until it was
measured.

Underfull \vbox warnings are expected and are not failures. They are the ordinary consequence of
two-column output with \pagestyle{empty}: the last column is shorter than the text block, and TeX
reports the slack every time \output fires.

Run:  python tools/build_pdf.py
      P8_NBDIR=<dir> P8_REPORT=<dir> python tools/build_pdf.py    (a rehearsal; submission/ untouched)
"""
import os
import re
import shutil
import subprocess
import sys

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-11/project-8-3d-movie")
REPORT = os.environ.get("P8_REPORT", f"{BASE}/report")
REHEARSAL = "P8_REPORT" in os.environ
TEXBIN = "C:/Users/Admin/.conda/envs/tex/Library/bin"
TECTONIC = f"{TEXBIN}/tectonic.exe"
SOURCE = f"{REPORT}/project8_report.tex"

if not os.path.exists(TECTONIC):
    raise SystemExit(f"tectonic not found at {TECTONIC}")

subprocess.run([sys.executable, f"{BASE}/tools/build_report.py"], check=True)
result = subprocess.run([TECTONIC, "-X", "compile", SOURCE, "--outdir", REPORT],
                        capture_output=True, text=True)
log = result.stdout + result.stderr
# Kept beside the PDF so verify.py can re-check the overfull count without recompiling.
with open(f"{REPORT}/project8_report.log", "w", encoding="utf-8") as handle:
    handle.write(log)
if result.returncode != 0:
    print(log[-4000:])
    raise SystemExit("tectonic failed")

# Deduplicated because tectonic runs the engine twice to settle references, so every warning is
# reported once per pass.
overfull = sorted({line.strip() for line in log.splitlines()
                   if re.search(r"Overfull \\[hv]box", line)})
for line in overfull:
    print(line)

pages = subprocess.run([f"{TEXBIN}/pdfinfo.exe", f"{REPORT}/project8_report.pdf"],
                       capture_output=True, text=True).stdout
count = re.search(r"^Pages:\s+(\d+)", pages, re.M).group(1)
print(f"compiled {count} pages, {len(overfull)} overfull box warnings")
if count != "2":
    print(f"  the report must be exactly 2 pages; it is {count}")

# The copy the upload is taken from is refreshed here rather than only by build_zip.py, so that
# compiling the report and then checking it cannot report a stale one. It is written only once the
# report has passed its own checks, which keeps a three-page or overhanging build out of
# submission/ entirely.
if not overfull and count == "2" and not REHEARSAL:
    os.makedirs(f"{BASE}/submission", exist_ok=True)
    shutil.copyfile(f"{REPORT}/project8_report.pdf", f"{BASE}/submission/project8_report.pdf")
    print("  copied to submission/project8_report.pdf")
raise SystemExit(1 if overfull or count != "2" else 0)
