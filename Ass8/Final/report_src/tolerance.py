r"""Compare two runs of main_report.ipynb line by line, with a stated numeric tolerance.

main_report.ipynb only evaluates trained networks: no training, no sampling noise, and the test
renders use no jitter. So two runs differ only where floating-point summation order differs by
device (a GPU and a CPU add the same products in a different order) and in how long rendering took.
The rule, stated once:

  * every non-numeric part of a line must be identical (words, model names, list shapes);
  * two numbers agree when they differ by at most one unit in the last printed place, or by 0.1%
    of their size, whichever is larger;
  * rendering times are not compared at all. They are a property of the machine, not of the
    models, so the time columns are masked before comparing (TIMED below);
  * the line that names the machine (device, GPU and torch version) is set aside.
"""
import re

ENV = re.compile(r"^(Using device|device|python|torch|torchvision|numpy)\s*:")
NUMBER = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?")
MODEL = r"(?:Tiny NeRF|Extended NeRF|No view dirs \(abl\. A\)|Uniform 96 \(abl\. B\))"
# The per-image render time: "123 ms per image" in a summary line, and the seconds column of the
# results table, which sits between the mean SSIM and the parameter count.
TIMED = [(re.compile(r"\d+ ms per image"), "<time> ms per image"),
         (re.compile(rf"^({MODEL}\s.*\s\d\.\d{{4}})\s+\d+\.\d{{3}}(\s+[\d,]+)$"), r"\1 <time>\2")]


def machine_free(text):
    lines = []
    for line in text.split("\n"):
        if ENV.match(line):
            continue
        for pattern, replacement in TIMED:
            line = pattern.sub(replacement, line)
        lines.append(line)
    return lines


def _decimals(token):
    return len(token.split(".")[1]) if "." in token and "e" not in token else 0


def _value(token):
    return float(token.replace(",", ""))


def compare(expected, actual):
    """Return (problems, notes). Each item is (line number, expected line, actual line, why)."""
    left, right = machine_free(expected.strip()), machine_free(actual.strip())
    problems, notes = [], []
    if len(left) != len(right):
        problems.append((0, f"{len(left)} lines", f"{len(right)} lines", "line counts differ"))
    for number, (a, b) in enumerate(zip(left, right), start=1):
        if a == b:
            continue
        if NUMBER.sub("#", a) != NUMBER.sub("#", b):
            problems.append((number, a, b, "the words differ"))
            continue
        worst, used_unit = None, False
        for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)):
            gap = abs(_value(x) - _value(y))
            size = max(abs(_value(x)), abs(_value(y)))
            unit = 10.0 ** -min(_decimals(x), _decimals(y))
            tolerance = max(unit, 1e-3 * size)
            if gap <= tolerance + 1e-12:
                used_unit = used_unit or gap > 0
                continue
            worst = f"{x} against {y}, beyond {tolerance:.2g}"
            break
        if worst:
            problems.append((number, a, b, worst))
        elif used_unit:
            notes.append((number, a, b, "within one unit in the last place"))
    return problems, notes
