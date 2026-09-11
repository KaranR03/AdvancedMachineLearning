r"""Unzip project5_code.zip into an empty directory and run main_report.ipynb from a fresh kernel.

"The files are in the zip" and "the zip works" are different checks. This is the second one, and
for this project it is the graded path: the brief says of main_report.ipynb that "this notebook
will be run for grading; if any cell fails, the code evaluation will receive 0 marks".

Only main_report.ipynb is executed. The two training notebooks would take the better part of an
hour on a GPU and a marker will not run them; what matters is that the notebook the brief names
runs from nothing but the archive's own contents.

The fresh output is compared against the committed notebook, so a silent divergence is caught as
well as an outright failure.

Run:  python tools/clean_room.py
"""
import io
import json
import os
import sys
import tempfile
import time
import zipfile

import nbformat
from nbclient import NotebookClient

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-08/project-5-extending-addition-llm")
NBDIR = f"{BASE}/notebook"
ARCHIVE = f"{BASE}/submission/project5_code.zip"


def stream_text(notebook):
    """The notebook's stdout, reassembled.

    Joining the chunks with a newline would add a blank line wherever Jupyter happened to split
    a stream message, which is a timing detail and differs between two runs of the same code.
    """
    return "".join("".join(o.get("text", [])) for c in notebook["cells"]
                   for o in c.get("outputs", []) if o.get("output_type") == "stream")


room = tempfile.mkdtemp(prefix="ifn680_p5_cleanroom_")
print(f"clean room: {room}")
with zipfile.ZipFile(ARCHIVE) as archive:
    archive.extractall(room)
print(f"unzipped {len(os.listdir(room))} entries")

# The figures are deleted before the run. If they were left in place a notebook whose plotting
# cells silently failed would still leave the files sitting there and look correct.
for name in ("figure_1_overview.pdf", "figure_2_by_length.pdf"):
    path = f"{room}/{name}"
    if os.path.exists(path):
        os.remove(path)
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

if not failures:
    code = [c for c in notebook.cells if c.cell_type == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    stderr = [o for c in code for o in c.get("outputs", [])
              if o.get("output_type") == "stream" and o.get("name") == "stderr"]
    fresh = stream_text(json.loads(nbformat.writes(notebook)))
    committed = stream_text(json.load(io.open(f"{NBDIR}/{name}", encoding="utf-8")))

    # The device line legitimately differs: the committed run is from the GPU node and this one
    # is wherever the clean room runs. Everything else must match exactly, because decoding is
    # greedy and therefore deterministic across devices.
    def strip_env(text):
        return "\n".join(line for line in text.split("\n")
                         if not line.startswith(("device :", "torch  :", "numpy  :")))

    identical = strip_env(fresh).strip() == strip_env(committed).strip()

    print(f"  {name}: {len(code)} cells in {time.perf_counter() - started:.0f} s, "
          f"errors={len(errors)}, stderr={len(stderr)}")
    print(f"  output identical to the committed notebook (ignoring the environment lines): "
          f"{identical}")
    if errors:
        failures.append(f"{name} produced {len(errors)} error outputs")
    if stderr:
        failures.append(f"{name} produced {len(stderr)} stderr blocks")
        for output in stderr:
            print(f"    STDERR: {''.join(output.get('text', []))[:300]}")
    if not identical:
        failures.append(f"{name} printed different numbers from the committed run")
        fresh_lines, committed_lines = strip_env(fresh).splitlines(), \
            strip_env(committed).splitlines()
        differing = [(i, a, b) for i, (a, b) in enumerate(zip(committed_lines, fresh_lines))
                     if a != b]
        print(f"    {len(differing)} differing lines, first few:")
        for i, a, b in differing[:6]:
            print(f"      line {i}: committed {a.strip()!r}")
            print(f"                fresh     {b.strip()!r}")
    for figure in ("figure_1_overview.pdf", "figure_2_by_length.pdf"):
        if not os.path.exists(f"{room}/{figure}"):
            failures.append(f"{figure} was not regenerated")
            print(f"    MISSING {figure}")

print()
if failures:
    print("CLEAN ROOM FAILED:")
    for failure in failures:
        print(f"  {failure}")
else:
    print("CLEAN ROOM PASSED: main_report.ipynb ran from the archive alone, with no errors or "
          "warnings, and reproduced every number")
print(f"(left in place for inspection: {room})")
sys.exit(1 if failures else 0)
