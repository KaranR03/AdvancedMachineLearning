r"""Regenerate the notebooks from the template while keeping the outputs of the last real run.

Prose changes far more often than code does, and case1 takes a quarter of an hour to execute, so
re-running it to fix a sentence is waste. This regenerates each notebook and then transplants the
stored outputs back, but only after asserting that every code cell is byte-identical to the one
that produced them. If any code changed, the notebook is left alone and named, because its outputs
would then describe something that is no longer in the file.

Run:  python tools/rebuild_preserving_outputs.py
"""
import io
import json
import shutil
import subprocess
import sys

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-06/project-4-ai-auditing")
NBDIR = f"{BASE}/notebook"
TOOLS = f"{BASE}/tools"

executed = {}
for case in (1, 2, 3):
    path = f"{NBDIR}/case{case}.ipynb"
    executed[case] = json.load(io.open(path, encoding="utf-8"))
    shutil.copyfile(path, f"{path}.backup")

subprocess.run([sys.executable, f"{TOOLS}/build_notebooks.py"], check=True)

stale = []
for case in (1, 2, 3):
    path = f"{NBDIR}/case{case}.ipynb"
    fresh = json.load(io.open(path, encoding="utf-8"))
    old_code = [c for c in executed[case]["cells"] if c["cell_type"] == "code"]
    new_code = [c for c in fresh["cells"] if c["cell_type"] == "code"]

    if len(old_code) != len(new_code) or any(
            "".join(a["source"]) != "".join(b["source"]) for a, b in zip(old_code, new_code)):
        differing = [i for i, (a, b) in enumerate(zip(old_code, new_code))
                     if "".join(a["source"]) != "".join(b["source"])]
        stale.append((case, len(old_code), len(new_code), differing))
        shutil.copyfile(f"{path}.backup", path)
        continue

    for old, new in zip(old_code, new_code):
        new["outputs"] = old["outputs"]
        new["execution_count"] = old["execution_count"]
    with io.open(path, "w", encoding="utf-8") as handle:
        json.dump(fresh, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    kept = sum(len(c["outputs"]) for c in new_code)
    print(f"case{case}: prose refreshed, {kept} stored outputs preserved")

for case, before, after, differing in stale:
    print(f"case{case}: CODE CHANGED ({before} cells then, {after} now, "
          f"differing indices {differing}); left as it was, re-run it")
raise SystemExit(1 if stale else 0)
