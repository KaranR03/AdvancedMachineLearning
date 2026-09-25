r"""Run on the IFN680 GPU node: execute the four notebooks in order, then pack the results.

Launched detached from a Hub kernel (start_new_session=True), so it survives the browser tab and
the kernel that started it. It writes a line to run_all.log after every cell, which is how the run
is watched from outside, and ends by writing p6_hub_outputs.zip with a sha256 manifest, so the
files that come back to the local build can be checked byte for byte against what the Hub wrote.

Usage on the Hub, from the folder holding the notebooks and mnist_custom.pt:
    python hub_run_all.py                       all four notebooks
    python hub_run_all.py main_report.ipynb     only the named ones
"""
import hashlib
import json
import os
import sys
import time
import zipfile

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
ORDER = sys.argv[1:] or ["cVAE_Baseline.ipynb", "cVAE_DiscriminatorLoss.ipynb",
                         "DigitClassifier.ipynb", "main_report.ipynb"]
ARTEFACTS = ["cVAE_Baseline.pth", "cVAE_DiscriminatorLoss.pth", "DigitClassifier.pth",
             "cVAE_Baseline_history.json", "cVAE_DiscriminatorLoss_history.json",
             "mnist_custom_test.pt", "figure_1_grids.pdf", "figure_2_reconstructions.pdf"]
LOG = open("run_all.log", "a", encoding="utf-8")


def log(message):
    LOG.write(f"{time.strftime('%H:%M:%S')} {message}\n")
    LOG.flush()


class LoggingClient(NotebookClient):
    """An nbclient that reports each printed line as it arrives, and each finished cell."""

    def process_message(self, msg, cell, cell_index):
        # Training cells run for most of an hour; their progress lines are the only live signal
        if msg.get("msg_type") == "stream":
            for line in msg["content"].get("text", "").splitlines():
                if line.strip():
                    log(f"    | {line[:160]}")
        return super().process_message(msg, cell, cell_index)

    def async_execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
        return self._run_and_log(cell, cell_index, execution_count, store_history)

    async def _run_and_log(self, cell, cell_index, execution_count, store_history):
        started = time.time()
        result = await super().async_execute_cell(cell, cell_index, execution_count,
                                                  store_history)
        if cell.cell_type == "code":
            text = "".join("".join(o.get("text", "")) for o in cell.get("outputs", [])
                           if o.get("output_type") == "stream")
            tail = text.strip().split("\n")[-1][:160] if text.strip() else ""
            log(f"  cell {cell.get('id')} {time.time() - started:.0f}s  {tail}")
        return result


failed = False
for name in ORDER:
    started = time.time()
    log(f"START {name}")
    notebook = nbformat.read(name, as_version=4)
    try:
        LoggingClient(notebook, timeout=7200, kernel_name="python3",
                      resources={"metadata": {"path": HERE}}).execute()
    except Exception as error:
        log(f"FAILED {name}: {type(error).__name__}: {str(error)[-2000:]}")
        failed = True
    nbformat.write(notebook, name)
    errors = sum(o.get("output_type") == "error" for c in notebook.cells
                 for o in c.get("outputs", []))
    stderr = sum(o.get("output_type") == "stream" and o.get("name") == "stderr"
                 for c in notebook.cells for o in c.get("outputs", []))
    log(f"END {name} {time.time() - started:.0f}s errors={errors} stderr={stderr}")
    if failed:
        break

packed = [name for name in ORDER + ARTEFACTS if os.path.exists(name)]
manifest = {name: hashlib.sha256(open(name, "rb").read()).hexdigest() for name in packed}
with open("p6_hub_manifest.json", "w") as handle:
    json.dump(manifest, handle, indent=1)
with zipfile.ZipFile("p6_hub_outputs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in packed + ["p6_hub_manifest.json", "run_all.log"]:
        archive.write(name)
log(f"{'STOPPED' if failed else 'DONE'} packed {len(packed)} files into p6_hub_outputs.zip")
