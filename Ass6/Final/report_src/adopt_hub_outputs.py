r"""Take the stored outputs of a notebook from a run on the IFN680 GPU node.

Why this exists. main_report.ipynb is the notebook that gets run for grading, in the Jupyter
environment rather than here, so the outputs it ships should be that environment's. There is no GPU
on this machine, which means the only way to store a GPU run's outputs is to execute the archive on
the Hub and bring the executed file back. Wholesale replacement of the local file would be the
obvious way to do that and is the wrong one: the returned file is not the file the rest of the build
verified, and swapping it in quietly makes every later check a statement about a different artifact.

So this adopts the outputs and keeps the source. That is only sound if the two really are the same
notebook, which is asserted rather than assumed, and asserted on the bytes:

  * same cells, same ids, same order, same types;
  * every cell's source byte-identical, not merely equivalent. Comparing parsed syntax trees is
    right when the question is "can these outputs still be true" (see rebuild_preserving_outputs.py)
    and wrong here, where the question is "did the Hub run this exact file". A stale upload would
    pass an AST comparison and must not pass this one;
  * the incoming run is clean: no error outputs and nothing on stderr;
  * the incoming run agrees with the outputs being replaced, line for line, once the lines that
    name the machine are set aside. In Project 5 that was free, because greedy decoding visits the
    same tokens on any device. Project 6 samples, so every latent is drawn from a seeded CPU
    torch.Generator before moving to the device, every metric is printed at fixed precision, and
    what float noise remains is judged by the one tolerance stated in tools/tolerance.py.
    This is the rail that matters: it makes it impossible to adopt a run that would change a number
    the report quotes;
  * for a training notebook that has never been run here there is nothing to compare, so it is
    adopted on source identity and a clean run alone; the numbers it prints are its own.

Stream chunking is expected to differ and is not compared. A single nbconvert run coalesces
consecutive writes into one stream output where a cell-by-cell execution keeps them apart; the
concatenated text is what carries meaning, and that is what gets compared.

    python tools/adopt_hub_outputs.py <executed-notebook.ipynb>
"""
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tolerance import ENV, compare  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
NBDIR = f"{BASE}/notebook"


def spine(notebook):
    """The part of a notebook that has to match: what the cells are and what they say."""
    return [[c["id"], c["cell_type"], "".join(c["source"])] for c in notebook["cells"]]


def stdout_of(notebook):
    return "".join("".join(o["text"]) for c in notebook["cells"] for o in c.get("outputs", [])
                   if o.get("output_type") == "stream" and o.get("name") == "stdout")


def fingerprint(notebook):
    text = stdout_of(notebook)
    return "\n".join(ln for ln in text.split("\n") if not ENV.match(ln))


def env_lines(notebook):
    return [ln for ln in stdout_of(notebook).split("\n") if ENV.match(ln)]


if len(sys.argv) != 2:
    raise SystemExit(__doc__.strip().splitlines()[-1].strip())

incoming_path = sys.argv[1]
incoming = json.load(io.open(incoming_path, encoding="utf-8"))
TARGET = os.path.basename(incoming_path)
local_path = f"{NBDIR}/{TARGET}"
local = json.load(io.open(local_path, encoding="utf-8"))

# ------------------------------------------------------------------ 1. the same notebook, on bytes
if spine(incoming) != spine(local):
    mine, theirs = spine(local), spine(incoming)
    if len(mine) != len(theirs):
        raise SystemExit(f"cell counts differ: {len(mine)} here, {len(theirs)} in {incoming_path}")
    for i, (a, b) in enumerate(zip(mine, theirs)):
        if a != b:
            raise SystemExit(
                f"cell {i} differs, so {incoming_path} is not a run of this notebook.\n"
                f"  here   id={a[0]!r} type={a[1]} {len(a[2])} chars\n"
                f"  there  id={b[0]!r} type={b[1]} {len(b[2])} chars")
print(f"source identical: {len(spine(local))} cells, "
      f"sha256 {hashlib.sha256(json.dumps(spine(local), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]}")

# ------------------------------------------------------------------------- 2. the run itself is ok
problems = [(c["id"], o.get("ename") or "".join(o.get("text", ""))[:120])
            for c in incoming["cells"] for o in c.get("outputs", [])
            if o.get("output_type") == "error"
            or (o.get("output_type") == "stream" and o.get("name") == "stderr")]
if problems:
    raise SystemExit(f"the incoming run is not clean: {problems}")
print("incoming run clean: no error outputs, nothing on stderr")

# -------------------------------------------------------- 3. it says the same thing about the data
if not stdout_of(local).strip():
    print("no stored outputs here to compare with: adopted on source identity and a clean run")
else:
    mismatches, notes = compare(stdout_of(local), stdout_of(incoming))
    if mismatches:
        number, here, there, why = mismatches[0]
        raise SystemExit(
            "the incoming run prints something different, so adopting it would change a number "
            f"the report quotes.\n  {len(mismatches)} lines outside the tolerance; first at line "
            f"{number}: {why}\n  here  {here!r}\n  there {there!r}")
    for number, here, there, why in notes:
        print(f"  note, line {number}: {here.strip()!r} / {there.strip()!r} ({why})")
    print(f"printed output matches within tolerance: {len(fingerprint(local)):,} characters, "
          f"md5 here {hashlib.md5(fingerprint(local).encode()).hexdigest()}")
print(f"  environment here  {env_lines(local)}")
print(f"  environment there {env_lines(incoming)}")

# ------------------------------------------------------------------------------- 4. adopt them
theirs = {c["id"]: c for c in incoming["cells"] if c["cell_type"] == "code"}
codes = [c for c in local["cells"] if c["cell_type"] == "code"]
before = sum(len(c.get("outputs", [])) for c in codes)
for cell in codes:
    cell["outputs"] = theirs[cell["id"]].get("outputs", [])
    cell["execution_count"] = theirs[cell["id"]].get("execution_count")

counts = [c["execution_count"] for c in codes]
if counts != list(range(1, len(codes) + 1)):
    raise SystemExit(f"execution counts are not 1..{len(codes)}: {counts}")

with io.open(local_path, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(local, handle, indent=1, ensure_ascii=False)
    handle.write("\n")

after = sum(len(c["outputs"]) for c in codes)
print(f"adopted: {after} stored outputs replace {before}, execution counts 1..{len(codes)}")
print(f"wrote {local_path}")
