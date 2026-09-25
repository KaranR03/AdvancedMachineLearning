r"""Compare two runs of main_report.ipynb line by line, with a stated numeric tolerance.

main_report.ipynb samples, so a run on the GPU node and a run on a CPU cannot be expected to print
byte-identical text: the same seeded latent codes go through convolutions whose floating-point
summation order differs by device. Every latent is drawn from a seeded CPU generator, which makes
the inputs identical, and every metric is printed at fixed precision, which hides most of the
remaining noise. What is left is compared here under one rule, stated once:

  * every non-numeric part of a line must be identical (words, True/False, list shapes);
  * two numbers agree when they differ by at most one unit in the last printed place, or by 0.1%
    of their size, whichever is larger;
  * a few lines report counts of classified images, where one image changing class between devices
    moves a rate by 1/n. Those lines get a count-based allowance, listed in COUNT_ALLOWANCE, and
    are reported as notes when they use it.

The three to five lines that describe the machine (device and library versions) are set aside.
"""
import re

ENV = re.compile(r"^(device|python|torch|torchvision|numpy)\s*:")
NUMBER = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?")

# Line prefix -> the largest difference allowed on it, from the counts behind the rate.
# Per-class rates are over about 1,000 images (892 to 1,135), so two flips move them by at most
# 0.0023; a grid rate is over 120 images, so one flip moves it by 0.0083.
COUNT_ALLOWANCE = {"class ": 0.0025, "grid accuracy": 0.0084}


def machine_free(text):
    return [line for line in text.split("\n") if not ENV.match(line)]


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
        allowance = next((value for prefix, value in COUNT_ALLOWANCE.items()
                          if a.startswith(prefix)), 0.0)
        worst, used_allowance = None, False
        for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)):
            gap = abs(_value(x) - _value(y))
            unit = 10.0 ** -min(_decimals(x), _decimals(y))
            tolerance = max(unit, 1e-3 * max(abs(_value(x)), abs(_value(y))))
            if gap <= tolerance + 1e-12:
                continue
            if gap <= allowance + 1e-12:
                used_allowance = True
                continue
            worst = f"{x} against {y}, beyond {tolerance:.2g}"
            break
        if worst:
            problems.append((number, a, b, worst))
        elif used_allowance:
            notes.append((number, a, b, "within the count allowance"))
    return problems, notes
