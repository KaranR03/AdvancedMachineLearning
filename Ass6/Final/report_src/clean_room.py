r"""Unzip project6_code.zip into an empty directory and run main_report.ipynb from a fresh kernel.

"The files are in the zip" and "the zip works" are different checks. This is the second one, and
for this project it is the graded path: the brief says of main_report.ipynb that "This notebook
will be run for grading, failure on any of the cell will results in 0 mark for the code
evaluation."

Only main_report.ipynb is executed. The three training notebooks take well over an hour on a GPU
and are not what is run for grading; what matters is that the notebook the brief names runs from
nothing but the archive's own contents.

The fresh output is compared with the stored one under the tolerance in tools/tolerance.py, so a
silent divergence is caught as well as an outright failure. The stored run is from the GPU node and
this one is on the local CPU, which is exactly the device change a marker's run may make.

Run:  python tools/clean_room.py
"""
import io
import json
import os
import sys
import time
import zipfile

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tolerance import compare  # noqa: E402

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-09/project-6-sharpness-quest")
NBDIR = f"{BASE}/notebook"
ARCHIVE = f"{BASE}/submission/project6_code.zip"
FIGURES = ("figure_1_grids.pdf", "figure_2_reconstructions.pdf")
CPU_LIMIT_SECONDS = 600


def stream_text(notebook):
    """The notebook's stdout, reassembled. Chunk boundaries are a timing detail, so the text is
    joined with nothing between chunks."""
    return "".join("".join(o.get("text", [])) for c in notebook["cells"]
                   for o in c.get("outputs", []) if o.get("output_type") == "stream")


scratch = os.environ.get("CLAUDE_JOB_DIR")
parent = f"{scratch}/tmp" if scratch else f"{BASE}/../../../scratch"
room = f"{parent}/p6_cleanroom_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(room)
print(f"clean room: {room}")
with zipfile.ZipFile(ARCHIVE) as archive:
    archive.extractall(room)
print(f"unzipped {len(os.listdir(room))} entries")

# The figures are deleted before the run. If they were left in place, a notebook whose plotting
# cells silently failed would still leave the files sitting there and look correct.
for name in FIGURES:
    if os.path.exists(f"{room}/{name}"):
        os.remove(f"{room}/{name}")
print("removed the two figures so the run has to regenerate them")

failures = []
name = "main_report.ipynb"
notebook = nbformat.read(io.open(f"{room}/{name}", encoding="utf-8"), as_version=4)
started = time.perf_counter()
try:
    NotebookClient(notebook, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": room}}).execute()
except Exception as error:
    failures.append(f"{name} raised {type(error).__name__}: {error}")
    print(f"  {name}: FAILED TO EXECUTE")
    print(f"    {error}"[:2000])
elapsed = time.perf_counter() - started

if not failures:
    code = [c for c in notebook.cells if c.cell_type == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    stderr = [o for c in code for o in c.get("outputs", [])
              if o.get("output_type") == "stream" and o.get("name") == "stderr"]
    fresh = stream_text(json.loads(nbformat.writes(notebook)))
    stored = stream_text(json.load(io.open(f"{NBDIR}/{name}", encoding="utf-8")))
    problems, notes = compare(stored, fresh)

    print(f"  {name}: {len(code)} cells in {elapsed:.0f} s, errors={len(errors)}, "
          f"stderr={len(stderr)}")
    print(f"  output matches the stored run within tolerance: {not problems}")
    for number, a, b, why in notes:
        print(f"    note, line {number}: {a.strip()!r} / {b.strip()!r} ({why})")
    if errors:
        failures.append(f"{name} produced {len(errors)} error outputs")
    if stderr:
        failures.append(f"{name} produced {len(stderr)} stderr blocks")
        for output in stderr:
            print(f"    STDERR: {''.join(output.get('text', []))[:300]}")
    if problems:
        failures.append(f"{name} printed {len(problems)} lines outside the tolerance")
        for number, a, b, why in problems[:8]:
            print(f"    line {number}: stored {a.strip()!r}")
            print(f"             fresh  {b.strip()!r}  ({why})")
    if elapsed > CPU_LIMIT_SECONDS:
        failures.append(f"{name} took {elapsed:.0f} s, over the {CPU_LIMIT_SECONDS} s limit")
    for figure in FIGURES:
        if not os.path.exists(f"{room}/{figure}"):
            failures.append(f"{figure} was not regenerated")
            print(f"    MISSING {figure}")

print()
if failures:
    print("CLEAN ROOM FAILED:")
    for failure in failures:
        print(f"  {failure}")
else:
    print("CLEAN ROOM PASSED: main_report.ipynb ran from the archive alone on this CPU, with no "
          "errors or warnings, and reproduced every number within tolerance")
print(f"(left in place for inspection: {room})")
sys.exit(1 if failures else 0)
