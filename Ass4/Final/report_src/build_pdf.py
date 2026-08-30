r"""Compile the report and fail on the warnings a page count cannot see.

The page count in tools/verify.py catches the cliff edge the unit states, a report over two pages.
It cannot catch a line or a table that is wider than the column it sits in: TeX reports that as an
"Overfull \hbox" warning and still produces a two-page PDF, so every other check passes while the
table overhangs the gutter. Table 1 sat 9.5pt wide of its column that way until it was measured.

Underfull \vbox warnings are expected here and are not failures. They are the ordinary consequence
of two-column output with \pagestyle{empty}: the last column of the document is shorter than the
text block, and TeX reports the slack every time \output fires.

Run:  python tools/build_pdf.py
"""
import os
import re
import subprocess
import sys

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-06/project-4-ai-auditing")
TECTONIC = "C:/Users/Admin/.conda/envs/tex/Library/bin/tectonic.exe"
SOURCE = f"{BASE}/report/project4_report.tex"

if not os.path.exists(TECTONIC):
    raise SystemExit(f"tectonic not found at {TECTONIC}")

subprocess.run([sys.executable, f"{BASE}/tools/build_report.py"], check=True)
result = subprocess.run([TECTONIC, "-X", "compile", SOURCE, "--outdir", f"{BASE}/report"],
                        capture_output=True, text=True)
log = result.stdout + result.stderr
if result.returncode != 0:
    print(log)
    raise SystemExit("tectonic failed")

# Deduplicated because tectonic runs the engine twice to settle references and cross-links, so
# every warning is reported once per pass.
overfull = sorted({line.strip() for line in log.splitlines()
                   if re.search(r"Overfull \\[hv]box", line)})
for line in overfull:
    print(line)

pages = subprocess.run(["C:/Users/Admin/.conda/envs/tex/Library/bin/pdfinfo.exe",
                        f"{BASE}/report/project4_report.pdf"],
                       capture_output=True, text=True).stdout
count = re.search(r"^Pages:\s+(\d+)", pages, re.M).group(1)
print(f"compiled {count} pages, {len(overfull)} overfull box warnings")
raise SystemExit(1 if overfull or count != "2" else 0)
