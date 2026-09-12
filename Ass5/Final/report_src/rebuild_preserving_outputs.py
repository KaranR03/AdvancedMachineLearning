r"""Regenerate the notebooks from the template while keeping the outputs of the last real run.

Prose changes far more often than code does, and the two training notebooks take the better part
of an hour each on a GPU node, so re-running them to fix a sentence is waste. This regenerates each
notebook and transplants the stored outputs back, but only for cells whose code still means what it
meant when those outputs were produced. A cell that now computes something else is left without
outputs and named, because its stored outputs would describe a file that no longer exists.

"Means the same" is decided on the parsed syntax tree rather than on the bytes. Three kinds of edit
provably cannot move a number, and all three vanish under parsing:

  * a comment, which never enters the tree at all;
  * a docstring, dropped here, since nothing in these notebooks reads __doc__;
  * wrapping a call as `_ = call()`, which suppresses the notebook's echo of the return value and
    changes nothing else.

Line numbers are excluded from the dump too, so rewrapping a long line is also free. Anything else,
including every edit to a printed string, still counts as a change.

The echo is the one case where an output does have to go. A cell ending in a bare expression stores
an execute_result holding the repr of that value; once the cell ends in `_ = ...` there is nothing
left to echo, so that output is dropped on the way in. The reverse case, a cell that gains an echo
it has no stored output for, cannot be repaired here and is reported instead.

Outputs are transplanted by cell id, not by position, so inserting a markdown cell is safe and
renaming a code cell's id is not.

main_report.ipynb is listed as re-executable: it runs on a CPU in about a minute, so when its code
changes it is simply left cleared for tools/run_notebook.py rather than treated as a loss.

Run:  python tools/rebuild_preserving_outputs.py
"""
import ast
import io
import json
import shutil
import subprocess
import sys

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-08/project-5-extending-addition-llm")
NBDIR = f"{BASE}/notebook"
TOOLS = f"{BASE}/tools"

# The training pair costs an hour each on the GPU node, so a code change there is a failure worth
# stopping for. main_report.ipynb is cheap to re-run, so a code change there is routine.
COSTLY = ("LLMForward.ipynb", "LLMReverse.ipynb")
REEXECUTABLE = ("main_report.ipynb",)
NOTEBOOKS = COSTLY + REEXECUTABLE

DOC_HOLDERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class DropEchoSuppression(ast.NodeTransformer):
    """Rewrite `_ = expr` as a bare `expr`, so adding the underscore reads as no change."""

    def visit_Assign(self, node):
        self.generic_visit(node)
        targets = node.targets
        if (len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id == "_"):
            return ast.Expr(value=node.value)
        return node


def normalised(source):
    """A form of the cell in which only meaning survives, or None if it does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, DOC_HOLDERS):
            body = node.body
            # A lone docstring in a module body is the cell's echoed value, so it stays.
            if (len(body) > 1 and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:]
    tree = DropEchoSuppression().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree)


def echoes(source):
    """True when running the cell would store an execute_result for its final value."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return bool(tree.body) and isinstance(tree.body[-1], ast.Expr)


executed = {}
for name in NOTEBOOKS:
    path = f"{NBDIR}/{name}"
    executed[name] = json.load(io.open(path, encoding="utf-8"))
    shutil.copyfile(path, f"{path}.backup")

# This script regenerates the notebooks itself, a few lines below. Running build_notebooks.py by
# hand beforehand therefore clears the very outputs this is here to rescue, and every later
# comparison then succeeds against an empty file: no drift is reported and nothing is carried
# across. Refusing to start is the only way to tell that apart from a genuinely clean run.
for name in COSTLY:
    stored = sum(len(c.get("outputs", [])) for c in executed[name]["cells"])
    if not stored:
        raise SystemExit(
            f"{name} has no stored outputs to preserve, so there is nothing for this script to "
            f"do.\nIt regenerates the notebooks itself; run it alone, not after "
            f"build_notebooks.py.\nTo recover the outputs of the last real run:\n"
            f"    python -c \"import io,json,zipfile; z=zipfile.ZipFile("
            f"'submission/project5_code.zip');\\\n"
            f"        [io.open('notebook/'+n,'w',encoding='utf-8',newline=chr(10)).write("
            f"json.dumps(json.loads(z.read(n)),indent=1,ensure_ascii=False)+chr(10))\\\n"
            f"         for n in z.namelist() if n.endswith('.ipynb')]\"")

subprocess.run([sys.executable, f"{TOOLS}/build_notebooks.py"], check=True)

lost = []
for name in NOTEBOOKS:
    path = f"{NBDIR}/{name}"
    fresh = json.load(io.open(path, encoding="utf-8"))
    old = {c["id"]: c for c in executed[name]["cells"] if c["cell_type"] == "code"}
    new = {c["id"]: c for c in fresh["cells"] if c["cell_type"] == "code"}

    changed, dropped_echo = [], 0
    for cid, cell in new.items():
        before = "".join(old[cid]["source"]) if cid in old else None
        after = "".join(cell["source"])
        if before is None or normalised(before) != normalised(after):
            changed.append(cid)
            continue
        outputs = old[cid]["outputs"]
        if echoes(before) and not echoes(after):
            kept = [o for o in outputs if o.get("output_type") != "execute_result"]
            dropped_echo += len(outputs) - len(kept)
            outputs = kept
        elif echoes(after) and not echoes(before):
            # The cell would now print a value no stored run ever produced.
            changed.append(cid)
            continue
        cell["outputs"] = outputs
        cell["execution_count"] = old[cid]["execution_count"]

    vanished = sorted(set(old) - set(new))
    if changed or vanished:
        lost.append((name, changed, vanished))
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(fresh, handle, indent=1, ensure_ascii=False)
        handle.write("\n")

    kept = sum(len(c["outputs"]) for c in new.values())
    echo_note = f", {dropped_echo} echo outputs dropped" if dropped_echo else ""
    print(f"{name}: rebuilt, {kept} stored outputs preserved{echo_note}")

blocking = False
for name, changed, vanished in lost:
    where = "re-run it on the GPU node" if name in COSTLY else "run tools/run_notebook.py on it"
    print(f"{name}: code changed (cells {changed}, removed {vanished}); "
          f"those cells have no outputs, {where}")
    blocking = blocking or name in COSTLY
raise SystemExit(1 if blocking else 0)
