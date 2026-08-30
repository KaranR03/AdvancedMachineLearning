r"""Unzip project4_code.zip into an empty directory, lay the supplied data out beside it as the
README instructs, and execute all three notebooks from a fresh kernel.

"The files are in the zip" and "the zip works" are different checks. This is the second one, and
it is the one the unit's "if your code does not work, you will score 0" refers to. The output is
compared against the committed notebooks so a silent divergence is caught too.

Run:  python tools/clean_room.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile

import nbformat
from nbclient import NotebookClient

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-06/project-4-ai-auditing")
NBDIR = f"{BASE}/notebook"
ARCHIVE = f"{BASE}/submission/project4_code.zip"


def stream_text(notebook):
    return "\n".join("".join(o.get("text", [])) for c in notebook["cells"]
                     for o in c.get("outputs", []) if o.get("output_type") == "stream")


room = tempfile.mkdtemp(prefix="ifn680_p4_cleanroom_")
print(f"clean room: {room}")
with zipfile.ZipFile(ARCHIVE) as archive:
    archive.extractall(room)
print(f"unzipped {len(os.listdir(room))} entries")

# The datasets and checkpoints ship with the assessment rather than in the archive, so they are
# staged here exactly as the archive's README tells a marker to stage them.
for case in (1, 2, 3):
    shutil.copytree(f"{NBDIR}/Case{case}", f"{room}/Case{case}")
    print(f"staged Case{case}/")

failures = []
for case in (1, 2, 3):
    name = f"case{case}.ipynb"
    notebook = nbformat.read(io.open(f"{room}/{name}", encoding="utf-8"), as_version=4)
    started = time.perf_counter()
    try:
        NotebookClient(notebook, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": room}}).execute()
    except Exception as error:
        failures.append(f"{name} raised {type(error).__name__}: {error}")
        print(f"  {name}: FAILED TO EXECUTE")
        continue

    code = [c for c in notebook.cells if c.cell_type == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    stderr = [o for c in code for o in c.get("outputs", [])
              if o.get("output_type") == "stream" and o.get("name") == "stderr"]
    fresh = stream_text(json.loads(nbformat.writes(notebook)))
    committed = stream_text(json.load(io.open(f"{NBDIR}/{name}", encoding="utf-8")))
    identical = fresh.strip() == committed.strip()

    print(f"  {name}: {len(code)} cells in {time.perf_counter() - started:.0f} s, "
          f"errors={len(errors)}, stderr={len(stderr)}, "
          f"output identical to the committed notebook: {identical}")
    if errors:
        failures.append(f"{name} produced {len(errors)} error outputs")
    if stderr:
        failures.append(f"{name} produced {len(stderr)} stderr blocks")
        for output in stderr:
            print(f"    STDERR: {''.join(output.get('text', []))[:300]}")
    if not identical:
        fresh_lines, committed_lines = fresh.splitlines(), committed.splitlines()
        differing = [(i, a, b) for i, (a, b) in enumerate(zip(committed_lines, fresh_lines))
                     if a != b]
        print(f"    {len(differing)} differing lines, first few:")
        for i, a, b in differing[:6]:
            print(f"      line {i}: committed {a.strip()!r}")
            print(f"                fresh     {b.strip()!r}")

print()
if failures:
    print("CLEAN ROOM FAILED:")
    for failure in failures:
        print(f"  {failure}")
else:
    print("CLEAN ROOM PASSED: all three notebooks ran from the archive with no errors or warnings")
print(f"(left in place for inspection: {room})")
sys.exit(1 if failures else 0)
