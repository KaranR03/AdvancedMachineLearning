r"""Run on the IFN680 GPU node: execute the shipped archive's main_report.ipynb the way it is graded.

The brief says of main_report.ipynb that "This notebook will be run for grading, failure on any of
the cell will results in 0 mark for the code evaluation", and asks for the IFN680 GPU environment.
So the final check runs the archive itself, not the working copy: project6_code.zip is reassembled
from its upload parts, its sha256 is checked against the local build, it is unzipped into a fresh
folder, the two figure PDFs are deleted so the run has to regenerate them, and main_report.ipynb is
executed from a fresh kernel. The executed notebook is copied to main_report_graded.ipynb for
tools/adopt_hub_outputs.py, and the outcome is written to graded_result.json.

Usage on the Hub, from the folder holding project6_code.zip or its .partNN pieces:
    python hub_graded_run.py <sha256 of the local project6_code.zip>
"""
import glob
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
EXPECTED = sys.argv[1]
FIGURES = ("figure_1_grids.pdf", "figure_2_reconstructions.pdf")

# The upload route caps a file at 10 MB, so the archive may arrive in parts.
parts = sorted(glob.glob("project6_code.zip.part*"))
if parts:
    with open("project6_code.zip", "wb") as whole:
        for part in parts:
            with open(part, "rb") as piece:
                whole.write(piece.read())
with open("project6_code.zip", "rb") as handle:
    digest = hashlib.sha256(handle.read()).hexdigest()

room = f"graded_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(room)
result = {"sha256": digest, "matches_local": digest == EXPECTED, "room": room, "parts": len(parts)}
if digest == EXPECTED:
    with zipfile.ZipFile("project6_code.zip") as archive:
        archive.extractall(room)
        result["entries"] = len(archive.namelist())
    for figure in FIGURES:
        os.remove(os.path.join(room, figure))
    path = os.path.join(room, "main_report.ipynb")
    notebook = nbformat.read(path, as_version=4)
    started = time.time()
    try:
        NotebookClient(notebook, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": os.path.abspath(room)}}).execute()
        result["raised"] = None
    except Exception as error:
        result["raised"] = f"{type(error).__name__}: {str(error)[-1500:]}"
    result["seconds"] = round(time.time() - started, 1)
    nbformat.write(notebook, path)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    outputs = [o for c in code for o in c.get("outputs", [])]
    result["cells"] = len(code)
    result["errors"] = sum(o.get("output_type") == "error" for o in outputs)
    result["stderr"] = sum(o.get("output_type") == "stream" and o.get("name") == "stderr"
                           for o in outputs)
    result["counts_1_to_n"] = [c.execution_count for c in code] == list(range(1, len(code) + 1))
    result["figures_regenerated"] = all(os.path.exists(os.path.join(room, f)) for f in FIGURES)
    shutil.copyfile(path, "main_report_graded.ipynb")
with open("graded_result.json", "w") as handle:
    json.dump(result, handle, indent=1)
print(json.dumps(result, indent=1))
