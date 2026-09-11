r"""Regenerate the notebooks from the template while keeping the outputs of the last real run.

Prose changes far more often than code does, and the two training notebooks take the better part
of an hour each on a GPU node, so re-running them to fix a sentence is waste. This regenerates
each notebook and transplants the stored outputs back, but only after asserting that every code
cell is byte-identical to the one that produced them. If any code changed, the notebook is left
alone and named, because its outputs would then describe something no longer in the file.

Outputs are transplanted by cell id, not by position, so inserting a markdown cell is safe and
renaming a code cell's id is not.

Run:  python tools/rebuild_preserving_outputs.py
"""
import io
import json
import shutil
import subprocess
import sys

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-08/project-5-extending-addition-llm")
NBDIR = f"{BASE}/notebook"
TOOLS = f"{BASE}/tools"
NOTEBOOKS = ("LLMForward.ipynb", "LLMReverse.ipynb", "main_report.ipynb")

executed = {}
for name in NOTEBOOKS:
    path = f"{NBDIR}/{name}"
    executed[name] = json.load(io.open(path, encoding="utf-8"))
    shutil.copyfile(path, f"{path}.backup")

subprocess.run([sys.executable, f"{TOOLS}/build_notebooks.py"], check=True)

stale = []
for name in NOTEBOOKS:
    path = f"{NBDIR}/{name}"
    fresh = json.load(io.open(path, encoding="utf-8"))
    old = {c["id"]: c for c in executed[name]["cells"] if c["cell_type"] == "code"}
    new = {c["id"]: c for c in fresh["cells"] if c["cell_type"] == "code"}

    changed = [cid for cid in new
               if cid not in old or "".join(old[cid]["source"]) != "".join(new[cid]["source"])]
    if changed or set(old) - set(new):
        stale.append((name, changed, sorted(set(old) - set(new))))
        shutil.copyfile(f"{path}.backup", path)
        continue

    for cid, cell in new.items():
        cell["outputs"] = old[cid]["outputs"]
        cell["execution_count"] = old[cid]["execution_count"]
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(fresh, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    kept = sum(len(c["outputs"]) for c in new.values())
    print(f"{name}: prose refreshed, {kept} stored outputs preserved")

for name, changed, removed in stale:
    print(f"{name}: CODE CHANGED (cells {changed}, removed {removed}); "
          f"left as it was, re-run it")
raise SystemExit(1 if stale else 0)
