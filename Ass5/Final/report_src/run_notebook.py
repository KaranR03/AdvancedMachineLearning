r"""Execute one notebook in place from a fresh kernel and report what verify.py would fail on.

The kernel's working directory is set to notebook/, which is what lets each notebook reach
project5_testset.pkl, the checkpoints and the history files by the same bare names a marker will
use after unzipping the archive into one folder.

Run:  python tools/run_notebook.py main_report.ipynb
"""
import io
import sys
import time

import nbformat
from nbclient import NotebookClient

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-08/project-5-extending-addition-llm/notebook")
name = sys.argv[1]
path = f"{BASE}/{name}"

notebook = nbformat.read(io.open(path, encoding="utf-8"), as_version=4)
started = time.perf_counter()
NotebookClient(notebook, timeout=7200, kernel_name="python3",
               resources={"metadata": {"path": BASE}}).execute()
nbformat.write(notebook, io.open(path, "w", encoding="utf-8"))

code = [c for c in notebook.cells if c.cell_type == "code"]
errors, stderr, images, echoed = [], [], 0, []
for index, cell in enumerate(code):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            errors.append((index, output.get("ename"), output.get("evalue")))
        if output.get("output_type") == "stream" and output.get("name") == "stderr":
            stderr.append((index, "".join(output.get("text", ""))))
        if output.get("output_type") in ("display_data", "execute_result") and \
                "image/png" in output.get("data", {}):
            images += 1
    # A cell ending in a bare expression echoes the object it returns. Harmless, but it is noise
    # in a submitted notebook and no count of images or errors catches it.
    if any(o.get("output_type") == "execute_result" and "image/png" not in o.get("data", {})
           for o in cell.get("outputs", [])):
        echoed.append(index)

counts = [c.get("execution_count") for c in code]
print(f"\n{name}: {len(code)} code cells in {time.perf_counter() - started:.1f} s")
print(f"  execution counts 1..N in order: {counts == list(range(1, len(code) + 1))}")
print(f"  images embedded: {images}")
print(f"  errors: {len(errors)}   stderr blocks: {len(stderr)}   echoed reprs: {echoed}")
for index, ename, evalue in errors:
    print(f"    ERROR cell {index}: {ename}: {evalue}")
for index, text in stderr:
    print(f"    STDERR cell {index}: {text.strip()[:600]}")
sys.exit(1 if errors or stderr else 0)
