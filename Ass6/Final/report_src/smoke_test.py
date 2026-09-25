r"""Execute the reduced-size notebooks end to end on the local CPU.

Every code path runs before any GPU time is committed: the data split and test-file write, both
training loops, the lambda sweep and its selection rule, the per-seed refits, the checkpoint and
history formats, the classifier, and main_report.ipynb's dependence on all of those files. At 2,500
training images and 2 epochs the whole pipeline takes a few minutes.

It is a rehearsal, not a result: its numbers mean nothing and nothing it writes is submitted. It
works in the job's scratch room, never in notebook/, and copies the 112 MB dataset in once.

Run:  python tools/build_notebooks.py --smoke
      python tools/smoke_test.py
"""
import io
import os
import shutil
import sys
import time

import nbformat
import torch
from nbclient import NotebookClient

ROOT = "f:/document/IFN_680_Advanced_Machine_Learning_and_Applications"
ROOM = os.environ.get("CLAUDE_JOB_DIR", "C:/Users/Admin/.claude/jobs/8640b033") + "/tmp/smoke"
ORDER = ["cVAE_Baseline.ipynb", "cVAE_DiscriminatorLoss.ipynb", "DigitClassifier.ipynb",
         "main_report.ipynb"]
PRODUCED = ["mnist_custom_test.pt", "cVAE_Baseline.pth", "cVAE_DiscriminatorLoss.pth",
            "DigitClassifier.pth", "cVAE_Baseline_history.json",
            "cVAE_DiscriminatorLoss_history.json", "figure_1_grids.pdf",
            "figure_2_reconstructions.pdf"]

for name in ORDER:
    if not os.path.exists(f"{ROOM}/{name}"):
        raise SystemExit(f"{ROOM}/{name} missing; run tools/build_notebooks.py --smoke first")

if not os.path.exists(f"{ROOM}/mnist_custom.pt"):
    shutil.copyfile(f"{ROOT}/data/mnist_custom.pt", f"{ROOM}/mnist_custom.pt")
    print("copied mnist_custom.pt into the room")

# Everything the pipeline writes is cleared first, so the run proves each file is produced
# rather than left over from an earlier attempt. The dataset itself stays.
for stale in PRODUCED:
    if os.path.exists(f"{ROOM}/{stale}"):
        os.remove(f"{ROOM}/{stale}")

failures = []
for name in ORDER:
    path = f"{ROOM}/{name}"
    notebook = nbformat.read(io.open(path, encoding="utf-8"), as_version=4)
    started = time.perf_counter()
    try:
        NotebookClient(notebook, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": ROOM}}).execute()
    except Exception as error:
        failures.append(f"{name} raised {type(error).__name__}")
        print(f"  {name}: FAILED TO EXECUTE")
        print(f"    {error}"[:3000])
        nbformat.write(notebook, io.open(path, "w", encoding="utf-8"))
        break

    nbformat.write(notebook, io.open(path, "w", encoding="utf-8"))
    code = [c for c in notebook.cells if c.cell_type == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    stderr = [o for c in code for o in c.get("outputs", [])
              if o.get("output_type") == "stream" and o.get("name") == "stderr"]
    echoed = [c.get("id") for c in code
              if any(o.get("output_type") == "execute_result" for o in c.get("outputs", []))]
    images = sum(1 for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    counts = [c.get("execution_count") for c in code]
    print(f"  {name}: {len(code)} cells in {time.perf_counter() - started:.0f} s, "
          f"errors={len(errors)}, stderr={len(stderr)}, images={images}, "
          f"counts 1..N={counts == list(range(1, len(code) + 1))}, echoed={echoed}")
    for output in errors:
        print(f"    ERROR {output.get('ename')}: {''.join(output.get('evalue', ''))[:400]}")
        failures.append(f"{name}: {output.get('ename')}")
    for output in stderr:
        print(f"    STDERR: {''.join(output.get('text', []))[:400]}")
        failures.append(f"{name}: stderr output")
    if echoed:
        failures.append(f"{name}: cells echo a value: {echoed}")

print()
for required in PRODUCED:
    if not os.path.exists(f"{ROOM}/{required}"):
        failures.append(f"{required} was never written")
        print(f"  MISSING {required}")

# The checkpoint format main_report depends on: plain dicts of tensors, loadable with
# weights_only=True, holding one cVAE per seed and, for the discriminator run, one D per seed.
if not failures:
    for name, prefixes in [("cVAE_Baseline.pth", ["seed_"]),
                           ("cVAE_DiscriminatorLoss.pth", ["seed_", "discriminator_seed_"])]:
        keys = sorted(torch.load(f"{ROOM}/{name}", map_location="cpu", weights_only=True))
        print(f"  {name}: {keys}")
        for prefix in prefixes:
            if not any(key.startswith(prefix) for key in keys):
                failures.append(f"{name} holds no {prefix}* entry")

print()
if failures:
    print("SMOKE TEST FAILED:")
    for failure in failures:
        print(f"  {failure}")
else:
    print("SMOKE TEST PASSED: all four notebooks ran clean and produced every required file")
sys.exit(1 if failures else 0)
