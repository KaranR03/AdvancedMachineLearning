r"""Execute the reduced-size notebooks end to end on local CPU.

This exists because the two training notebooks have never been shown to run. The imported draft
shipped them with no stored outputs and every execution_count None, so "the training loop works"
was an assumption. Running the whole pipeline at 2,000 training examples for 2 epochs takes a few
minutes on a CPU and proves the loop, the checkpoint save, the history dump, the ablation and
main_report's dependency on all four files, before an hour of GPU time is committed to it.

It is a rehearsal, not a result: the numbers it produces are meaningless and nothing it writes is
ever submitted. It writes into the scratch directory, never into notebook/.

Run:  python tools/build_notebooks.py --smoke
      python tools/smoke_test.py
"""
import io
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOM = (os.environ.get("CLAUDE_JOB_DIR", "C:/Users/Admin/.claude/jobs/8640b033") + "/tmp/smoke")
ORDER = ["LLMForward.ipynb", "LLMReverse.ipynb", "main_report.ipynb"]

for name in ORDER:
    if not os.path.exists(f"{ROOM}/{name}"):
        raise SystemExit(f"{ROOM}/{name} missing; run tools/build_notebooks.py --smoke first")

# The training notebooks write project5_testset.pkl, the checkpoints and the history files into
# their working directory, and main_report.ipynb then reads all of them. Clearing those first
# means the run proves they are produced rather than left over from a previous attempt.
for stale in os.listdir(ROOM):
    if stale.endswith((".pkl", ".pth", ".json", ".pdf", ".png")):
        os.remove(f"{ROOM}/{stale}")

failures = []
for name in ORDER:
    path = f"{ROOM}/{name}"
    notebook = nbformat.read(io.open(path, encoding="utf-8"), as_version=4)
    started = time.perf_counter()
    try:
        NotebookClient(notebook, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": ROOM}}).execute()
    except Exception as error:
        failures.append(f"{name} raised {type(error).__name__}: {error}")
        print(f"  {name}: FAILED TO EXECUTE")
        print(f"    {error}"[:2000])
        continue

    nbformat.write(notebook, io.open(path, "w", encoding="utf-8"))
    code = [c for c in notebook.cells if c.cell_type == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    stderr = [o for c in code for o in c.get("outputs", [])
              if o.get("output_type") == "stream" and o.get("name") == "stderr"]
    images = sum(1 for c in code for o in c.get("outputs", [])
                 if "image/png" in o.get("data", {}))
    counts = [c.get("execution_count") for c in code]

    print(f"  {name}: {len(code)} cells in {time.perf_counter() - started:.0f} s, "
          f"errors={len(errors)}, stderr={len(stderr)}, images={images}, "
          f"counts 1..N in order={counts == list(range(1, len(code) + 1))}")
    for output in errors:
        print(f"    ERROR {output.get('ename')}: {''.join(output.get('evalue', ''))[:400]}")
        failures.append(f"{name}: {output.get('ename')}")
    for output in stderr:
        print(f"    STDERR: {''.join(output.get('text', []))[:400]}")
        failures.append(f"{name}: stderr output")

# main_report.ipynb is the notebook the marker runs, so what it needed to exist is listed here.
produced = sorted(f for f in os.listdir(ROOM) if f.endswith((".pkl", ".pth", ".json", ".pdf")))
print(f"\nartifacts produced: {produced}")
for required in ["project5_testset.pkl", "LLMForward.pth", "LLMReverse.pth",
                 "LLMForward_history.json", "LLMReverse_history.json",
                 "LLMForward_history_80k.json", "LLMReverse_history_80k.json",
                 "figure_1_overview.pdf", "figure_2_by_length.pdf"]:
    if required not in produced:
        failures.append(f"{required} was never written")
        print(f"  MISSING {required}")

print()
if failures:
    print("SMOKE TEST FAILED:")
    for failure in failures:
        print(f"  {failure}")
else:
    print("SMOKE TEST PASSED: all three notebooks ran clean and produced every required file")
sys.exit(1 if failures else 0)
