r"""Write report/project7_report.tex from the executed main_report.ipynb.

Every number in the report is pulled out of the notebook's printed output by regex, and every
sentence whose truth depends on the run (which schedule is better at a step count, whether an
interval excludes zero, whether the seeds agree, which way a loss moves) is generated from those
numbers. Nothing numeric is typed into the template except the settings the design fixes and the
findings quoted from prior work, which verify.py lists.

Run:  python tools/build_report.py
      P7_NBDIR=<dir> P7_REPORT=<dir> python tools/build_report.py    (against another run)
"""
import io
import json
import os
import re
import shutil

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-10/project-7-improving-ddpm")
NBDIR = os.environ.get("P7_NBDIR", f"{BASE}/notebook")
REPORT = os.environ.get("P7_REPORT", f"{BASE}/report")
FIGURES = f"{REPORT}/figures"
FIGURE_FILES = ("figure_1_grids.pdf", "figure_2_steps.pdf", "figure_3_few_steps.pdf",
                "figure_4_mechanism.pdf")

NUM = r"([\d.]+)"
SIGNED = r"([+-][\d.]+)"
COUNT = r"([\d,]+)"
SCHEDULES = ("linear", "cosine")
SAMPLERS = ("ddpm", "ddim")


def printed():
    """All stdout text of the executed main_report.ipynb, concatenated in cell order."""
    notebook = json.load(io.open(f"{NBDIR}/main_report.ipynb", encoding="utf-8"))
    chunks = ["".join(o.get("text", []))
              for cell in notebook["cells"] for o in cell.get("outputs", [])
              if o.get("output_type") == "stream"]
    text = "".join(chunks)
    assert text.strip(), "main_report.ipynb has no printed output; execute it first"
    return text


T = printed()


def grab(pattern, label):
    match = re.search(pattern, T, re.MULTILINE)
    assert match, f"could not find {label}"
    return match.groups() if len(match.groups()) > 1 else match.group(1)


def every(pattern, label, expected=None):
    rows = re.findall(pattern, T, re.MULTILINE)
    assert rows, f"could not find {label}"
    assert expected is None or len(rows) == expected, f"{label}: {len(rows)} rows, not {expected}"
    return rows


def f(text):
    return float(text.replace(",", ""))


V = {}

# ------------------------------------------------------------------------ setup and records
V["device"] = grab(r"^device\s*:\s*(\S+)", "device")
V["torch"] = grab(r"^torch\s*:\s*(\S+)", "torch version")
V["train_device"], V["gpu"], V["train_torch"] = grab(
    r"^trained on (\S+) \((.+)\), torch (\S+)$", "training device")
V["epochs"], V["batch"], V["lr"], seeds, V["n_train"] = grab(
    rf"^epochs (\d+) \| batch size (\d+) \| learning rate (\S+) \| seeds \[([\d, ]+)\] \| "
    rf"{COUNT} training images", "training settings")
SEEDS = [int(s) for s in seeds.split(",")]
RECORD = {}
for name, seconds, minutes, losses in every(
        rf"^record \| (linear|cosine) \| seconds per epoch {NUM} \| minutes per run (.+?) \| "
        rf"final training loss (.+)$", "training records", 2):
    RECORD[name] = dict(seconds=seconds, minutes=minutes.split(), losses=losses.split())
    assert len(RECORD[name]["losses"]) == len(SEEDS), f"{name}: one loss per seed expected"
V["n_unet"], V["n_clf"] = grab(rf"^parameters \| U-Net {COUNT} \| classifier {COUNT}",
                               "parameter counts")

GRID = dict(every(rf"^grid accuracy \| (linear|cosine) \| {NUM} of 120 images", "grid accuracy",
                  2))
V["n_a"], V["n_b"] = grab(rf"^sets \| A {COUNT} \| B {COUNT}", "evaluation sets")
V["clf_acc"] = grab(rf"^classifier \| all {COUNT} real test images \| accuracy {NUM}",
                    "classifier accuracy")[1]
V["floor"], V["floor_acc"], V["floor_spread"] = grab(
    rf"^floor \| real set A against set B \| FD {NUM} \| accuracy {NUM} \| spread {NUM}",
    "floor")

# ------------------------------------------------------------------ quality against K
QUALITY = {}
for seed, sampler, k, *values in every(
        rf"^quality \| seed (\d) \| (ddpm|ddim) \| K=\s*(\d+) \| linear FD\s+{NUM} accuracy {NUM} "
        rf"spread {NUM} \| cosine FD\s+{NUM} accuracy {NUM} spread {NUM}", "quality rows"):
    for i, name in enumerate(SCHEDULES):
        QUALITY[name, int(seed), sampler, int(k)] = dict(
            fd=values[3 * i], acc=values[3 * i + 1], spread=values[3 * i + 2])
REDUCED = sorted({k for (_, seed, _, k) in QUALITY if seed == SEEDS[1]})
FULL = max(k for (_, _, _, k) in QUALITY)
assert FULL == 1000 and REDUCED == [10, 20, 50, 100, 250], "unexpected step counts"

RULE = {}
for sampler, k, gap, low, high, others, winner in every(
        rf"^rule \| (ddpm|ddim) \| K=\s*(\d+) \| cosine minus linear FD {SIGNED} "
        rf"\[{SIGNED}, {SIGNED}\] \| other seeds (.+?) \| (cosine|linear|unclear)\s*$",
        "rule rows", 2 * len(REDUCED)):
    RULE[sampler, int(k)] = dict(gap=gap, low=low, high=high, others=others.split(),
                                 winner=winner)
VERDICT = {}
for sampler, cosine, linear, unclear in every(
        r"^verdict \| (ddpm|ddim) \| cosine better at K = (.+?) \| linear better at K = (.+?) \| "
        r"unclear at K = (.+?)\s*$", "verdicts", 2):
    VERDICT[sampler] = {"cosine": [] if cosine == "none" else cosine.split(", "),
                        "linear": [] if linear == "none" else linear.split(", "),
                        "unclear": [] if unclear == "none" else unclear.split(", ")}
V["full_gap"], V["full_low"], V["full_high"] = grab(
    rf"^full steps \| cosine minus linear FD {SIGNED} \[{SIGNED}, {SIGNED}\]", "full steps")
FEWEST = {}
STEPS_OR_NONE = r"(from K=\d+|at no reduced K)"
for sampler, own_lin, own_cos, reach in every(
        rf"^fewest steps \| (ddpm|ddim) \| within 10% of its own 1000-step FD: linear "
        rf"{STEPS_OR_NONE}, cosine {STEPS_OR_NONE} \| cosine at or below linear's 1000-step FD "
        rf"{STEPS_OR_NONE}\s*$", "fewest steps", 2):
    FEWEST[sampler] = {key: None if value.startswith("at no") else value.split("=")[1]
                       for key, value in (("linear", own_lin), ("cosine", own_cos),
                                          ("reach", reach))}

# ------------------------------------------------------------------ training time and loss
EPOCHS = every(rf"^epochs \| epoch\s+(\d+) \| linear FD\s+{NUM} accuracy {NUM} \| cosine "
               rf"FD\s+{NUM} accuracy {NUM}", "epoch rows")
(V["reach_epoch_lin"], V["reach_min_lin"], V["reach_epoch_cos"],
 V["reach_min_cos"]) = grab(rf"^epochs \| within 10% of the final FD from epoch: linear (\d+) "
                            rf"\({NUM} min\), cosine (\d+) \({NUM} min\)", "epochs to 10%")
LOSS = {}
for name, mean, n_noisy, noisy, elsewhere in every(
        rf"^loss \| (linear|cosine) \| mean over the 20 steps {NUM} \| (\d+) of 20 steps below "
        rf"0\.01 signal, mean MSE there {NUM}, elsewhere {NUM}", "loss summaries", 2):
    LOSS[name] = dict(mean=mean, n_noisy=n_noisy, noisy=noisy, elsewhere=elsewhere)
MATCHED = {}
for side, threshold, mean, low, high, top_lin, top_cos in every(
        rf"^loss \| matched log SNR (above|below) {SIGNED} \| cosine over linear error: mean "
        rf"{NUM}, from {NUM} to {NUM} \| largest error linear {NUM}, cosine {NUM}",
        "matched loss", 2):
    MATCHED[side] = dict(threshold=threshold, mean=mean, low=low, high=high, top_lin=top_lin,
                         top_cos=top_cos)
VARIANCE = {}
for k, *values in every(
        rf"^variance \| K=\s*(\d+) \| linear last-step noise {NUM} FD with beta\s+{NUM}, with "
        rf"beta_tilde\s+{NUM} \| cosine last-step noise {NUM} FD with beta\s+{NUM}, with "
        rf"beta_tilde\s+{NUM}", "variance follow-up", 5):
    VARIANCE[int(k)] = {name: dict(noise=values[3 * i], beta=values[3 * i + 1],
                                   tilde=values[3 * i + 2]) for i, name in enumerate(SCHEDULES)}
FACTS = {}
for name, share, n10, t2, abar2, weight, gain10, gain1000 in every(
        rf"^schedule \| (linear|cosine) \| below 0\.01: {NUM}% of steps, (\d+) of K=10's \| "
        rf"alpha_bar at t=(\d+) {NUM}, weight {NUM} \| first-step multiplier K=10 ([\d,.]+), "
        rf"K=1000 ([\d,.]+)", "schedule facts", 2):
    FACTS[name] = dict(share=share, n10=n10, t2=t2, abar2=abar2, weight=weight, gain10=gain10,
                       gain1000=gain1000)
V["short_end"] = grab(rf"^schedule \| linear over a 100-step process ends at alpha_bar {NUM}",
                      "short process")

# --------------------------------------------------------------------------------- figures
os.makedirs(FIGURES, exist_ok=True)
for name in FIGURE_FILES:
    shutil.copyfile(f"{NBDIR}/{name}", f"{FIGURES}/{name}")


# ------------------------------------------------------------------ helpers for the prose
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
SAMPLER_NAME = {"ddpm": "DDPM", "ddim": "DDIM"}


def listed(words):
    """a, b and c"""
    words = list(words)
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def ks(values):
    """['10', '20'] -> 'K = 10 and 20'. Each number is its own math group, so the commas and the
    word "and" stay in text mode, where they keep their spaces."""
    values = list(values)
    if not values:
        return ""
    return listed([f"$K = {values[0]}$"] + [f"${value}$" for value in values[1:]])


def q(name, sampler, k, key="fd", seed=None):
    return QUALITY[name, SEEDS[0] if seed is None else seed, sampler, k][key]


for name in SCHEDULES:
    V[f"grid_acc_{name}"] = GRID[name]
    V[f"seconds_{name}"] = RECORD[name]["seconds"]
    V[f"noisy_share_{name}"] = FACTS[name]["share"]
    V[f"noisy10_{name}"] = FACTS[name]["n10"]
    V[f"weight_{name}"] = FACTS[name]["weight"]
    V[f"abar2_{name}"] = FACTS[name]["abar2"]
    V[f"gain10_{name}"] = FACTS[name]["gain10"]
    V[f"gain1000_{name}"] = FACTS[name]["gain1000"]
    V[f"loss_final_{name}"] = RECORD[name]["losses"][0]
    V[f"loss_mean_{name}"] = LOSS[name]["mean"]
    V[f"loss_noisy_{name}"] = LOSS[name]["noisy"]
    V[f"loss_else_{name}"] = LOSS[name]["elsewhere"]
    V[f"n_noisy_{name}"] = LOSS[name]["n_noisy"]
    V[f"fd_full_{name}"] = q(name, "ddpm", FULL)
    V[f"acc_full_{name}"] = q(name, "ddpm", FULL, "acc")
    V[f"spread_full_{name}"] = q(name, "ddpm", FULL, "spread")
V["t2"] = FACTS["linear"]["t2"]
# Every printed FD, accuracy and spread by name, so the hand-written paragraphs cite a number as a
# placeholder (e.g. __FD_COSINE_DDPM_250_S1__) and never retype it.
for (name, seed, sampler, k), values in QUALITY.items():
    for key, value in values.items():
        V[f"{key}_{name}_{sampler}_{k}_s{seed}"] = value
for epoch, fd_lin, acc_lin, fd_cos, acc_cos in EPOCHS:
    V[f"epoch_fd_linear_{epoch}"], V[f"epoch_fd_cosine_{epoch}"] = fd_lin, fd_cos
for k, row in VARIANCE.items():
    for name in SCHEDULES:
        for key, value in row[name].items():
            V[f"var_{key}_{name}_{k}"] = value
for (sampler, k), row in RULE.items():
    V[f"gap_{sampler}_{k}"] = row["gap"]
    for seed, other in zip(SEEDS[1:], row["others"]):
        V[f"gap_{sampler}_{k}_s{seed}"] = other
V["n_seeds"] = NUMBER_WORDS[len(SEEDS)]


def verdict_sentence(sampler):
    """Which schedule the fixed rule favours at each reduced K, for one sampler."""
    groups = VERDICT[sampler]
    if len(groups["unclear"]) == len(REDUCED):
        return (f"With {SAMPLER_NAME[sampler]} sampling, neither schedule is clearly better at any "
                f"$K < {FULL}$.")
    parts = []
    if groups["cosine"]:
        parts.append(f"the cosine schedule is better at {ks(groups['cosine'])}")
    if groups["linear"]:
        parts.append(f"the linear schedule is better at {ks(groups['linear'])}")
    if groups["unclear"]:
        parts.append(f"neither is clearly better at {ks(groups['unclear'])}")
    return f"With {SAMPLER_NAME[sampler]} sampling, " + listed(parts) + "."


V["verdict_ddpm"] = verdict_sentence("ddpm")
V["verdict_ddim"] = verdict_sentence("ddim")

# The overall reading of the hypothesis, from the rule's outcomes over both samplers.
COS_WINS = sum(len(VERDICT[s]["cosine"]) for s in SAMPLERS)
LIN_WINS = sum(len(VERDICT[s]["linear"]) for s in SAMPLERS)
TOTAL = 2 * len(REDUCED)
# "Supports" is written only when the cosine schedule wins somewhere and loses nowhere, which is
# what verify.py checks; any case where each schedule wins somewhere is "mixed", whatever the count.
if COS_WINS == TOTAL:
    SUPPORT = "full"
elif COS_WINS > 0 and LIN_WINS == 0:
    SUPPORT = "partial"
elif COS_WINS > 0 and LIN_WINS > 0:
    SUPPORT = "mixed"
elif COS_WINS == 0 and LIN_WINS == 0:
    SUPPORT = "none-unclear"
else:
    SUPPORT = "against"
V["support"] = SUPPORT


def from_k(value):
    return "at no reduced $K$" if value is None else f"from $K = {value}$"


def fewest_clause(sampler, after_full_clause):
    """Where the three fewest-steps marks land for one sampler. A sampler that meets none of them
    is summarised in a few words, but only after a clause that has already named the three."""
    lin, cos, reach = (FEWEST[sampler][key] for key in ("linear", "cosine", "reach"))
    if after_full_clause and lin is None and cos is None and reach is None:
        return f"with {SAMPLER_NAME[sampler]} none of these happens at any reduced $K$"
    return (f"with {SAMPLER_NAME[sampler]} the linear model comes within 10\\% of its own "
            f"1000-step FD {from_k(lin)}, the cosine model {from_k(cos)}, and the cosine model "
            f"reaches the linear model's {from_k(reach)}")


FEWEST_CLAUSES = [fewest_clause(SAMPLERS[0], False)]
FEWEST_CLAUSES.append(fewest_clause(SAMPLERS[1], True))
V["fewest_sentence"] = "W" + "; ".join(FEWEST_CLAUSES)[1:] + "."

FULL_DECISIVE = f(V["full_high"]) < 0 or f(V["full_low"]) > 0
FULL_BETTER = "cosine" if f(V["full_gap"]) < 0 else "linear"
V["full_sentence"] = (
    f"At the full 1000 steps the FDs are {V['fd_full_linear']} (linear) and "
    f"{V['fd_full_cosine']} (cosine), a difference of {V['full_gap']} "
    f"[{V['full_low']}, {V['full_high']}]"
    + (f", so the {FULL_BETTER} model is better even without skipping steps."
       if FULL_DECISIVE else ", within evaluation noise."))

# Training time: the seconds per epoch, and how early each schedule reached its final quality.
SEC_GAP = abs(f(V["seconds_linear"]) - f(V["seconds_cosine"])) / f(V["seconds_linear"])
V["time_sentence"] = (
    f"An epoch took {V['seconds_linear']}~s (linear) and {V['seconds_cosine']}~s (cosine)"
    + (", equal within timing noise, as expected: the schedule changes which noise level a "
       "step uses, not the computation." if SEC_GAP < 0.05 else
       ", a difference the identical code path does not explain, so it reflects the machine.")
    + f" The seed-0 snapshots first come within 10\\% of their final FD at epoch "
    f"{V['reach_epoch_lin']} (linear, {V['reach_min_lin']}~min) and {V['reach_epoch_cos']} "
    f"(cosine, {V['reach_min_cos']}~min).")

# The loss mechanism: why the two training losses differ. Whether the models make the same error
# at the same noise level decides the wording, so it is read from the matched comparison.
ABOVE = MATCHED["above"]
SAME_ERROR = 0.9 <= f(ABOVE["mean"]) <= 1.1
V["match_mean"], V["match_min"], V["match_max"] = ABOVE["mean"], ABOVE["low"], ABOVE["high"]
V["match_top_lin"], V["match_top_cos"] = MATCHED["below"]["top_lin"], MATCHED["below"]["top_cos"]
V["loss_sentence"] = (
    f"The final training losses differ ({V['loss_final_linear']} linear, "
    f"{V['loss_final_cosine']} cosine at seed 0) "
    + ("mostly" if SAME_ERROR else "partly")
    + f" because the schedules weight noise levels differently: {V['n_noisy_linear']} of 20 "
    f"evenly spaced steps leave under 1\\% of the signal with the linear schedule and "
    f"{V['n_noisy_cosine']} with the cosine, where the noise is almost free to predict (MSE "
    f"{V['loss_noisy_linear']} against {V['loss_else_linear']} elsewhere). At equal noise "
    f"levels with over 1\\% of the signal left, the cosine model's error averages "
    f"{V['match_mean']} times the linear model's ({V['match_min']} to {V['match_max']})"
    + (", about the same." if SAME_ERROR else
       (", larger." if f(ABOVE["mean"]) > 1.1 else ", smaller.")))

# The follow-up on the stochastic sampler's last-step noise, at the step counts where the rule
# favoured the cosine schedule with DDPM (or at K = 10 if it favoured it nowhere).
FOLLOW_K = [int(k) for k in VERDICT["ddpm"]["cosine"]] or [10]
K0 = FOLLOW_K[0]
gap_beta = f(VARIANCE[K0]["cosine"]["beta"]) - f(VARIANCE[K0]["linear"]["beta"])
gap_tilde = f(VARIANCE[K0]["cosine"]["tilde"]) - f(VARIANCE[K0]["linear"]["tilde"])
CLOSES = abs(gap_tilde) < 0.5 * abs(gap_beta)
V["variance_sentence"] = (
    f"A follow-up at seed 0, run after the verdict, locates the DDPM gain. At $K = {K0}$ the "
    f"second-to-last step jumps to $t = 1$ and leaves noise of standard deviation "
    f"{VARIANCE[K0]['linear']['noise']} (linear) against {VARIANCE[K0]['cosine']['noise']} "
    f"(cosine), which the last step barely removes. With Ho et al.'s smaller posterior variance "
    f"$\\tilde\\beta$, the FDs at $K = {K0}$ go from {VARIANCE[K0]['linear']['beta']} to "
    f"{VARIANCE[K0]['linear']['tilde']} (linear) and from {VARIANCE[K0]['cosine']['beta']} to "
    f"{VARIANCE[K0]['cosine']['tilde']} (cosine)"
    + (", so most of the cosine schedule's advantage there came from that leftover noise."
       if CLOSES else ", so the leftover noise does not explain the cosine schedule's advantage "
       "there on its own."))


def table_rows():
    rows = []
    for k in REDUCED + [FULL]:
        cells = [str(k)]
        for sampler in SAMPLERS:
            if (SCHEDULES[0], SEEDS[0], sampler, k) not in QUALITY:
                cells.append("\\multicolumn{4}{c}{not run}")
                continue
            lin, cos = q("linear", sampler, k), q("cosine", sampler, k)
            if k == FULL:
                winner = (FULL_BETTER if FULL_DECISIVE else "unclear")
                gap = f"{V['full_gap']} [{V['full_low']}, {V['full_high']}]"
                others = "not run"
            else:
                rule = RULE[sampler, k]
                winner = rule["winner"]
                gap = f"{rule['gap']} [{rule['low']}, {rule['high']}]"
                others = ", ".join(rule["others"])
            lin = f"\\textbf{{{lin}}}" if winner == "linear" else lin
            cos = f"\\textbf{{{cos}}}" if winner == "cosine" else cos
            cells += [lin, cos, gap, others]
        rows.append(" & ".join(cells) + " \\\\")
    return "\n".join(rows)


V["table_rows"] = table_rows()

# Conclusion, generated from the rule's outcomes.
def cases(winner):
    return listed([f"{ks(VERDICT[s][winner])} with {SAMPLER_NAME[s]}"
                   for s in SAMPLERS if VERDICT[s][winner]])


# Each reading is written only when it applies, since some refer to cases that may not exist.
CONCLUSION = {
    "full": lambda: (
        "The evidence supports the hypothesis: by a rule fixed before the results, the cosine "
        "schedule gave better samples than the linear one at every reduced number of sampling "
        f"steps, with both samplers and at all {V['n_seeds']} training seeds."),
    "partial": lambda: (
        "The evidence supports the hypothesis in part: by a rule fixed before the results, the "
        f"cosine schedule gave better samples at {cases('cosine')}, and the linear schedule at no "
        "reduced step count."),
    "mixed": lambda: (
        "The evidence does not support the hypothesis in general: by a rule fixed before the "
        f"results, the cosine schedule gave better samples at {cases('cosine')}, but the linear "
        f"schedule did at {cases('linear')}."),
    "none-unclear": lambda: (
        "The evidence does not support the hypothesis: at no reduced number of steps did either "
        "schedule meet the rule fixed before the results."),
    "against": lambda: (
        "The evidence does not support the hypothesis: by a rule fixed before the results, the "
        f"linear schedule gave better samples at {cases('linear')}, and the cosine schedule at "
        "no reduced step count."),
}
# The verdict sentence is generated; what it means and what to do next are judgement, so the
# Conclusion's remaining one or two sentences are written by hand once the numbers exist, like the
# two Discussion paragraphs below. verify.py checks the whole stays at two or three sentences.
CONCLUSION_PATH = os.environ.get("P7_CONCLUSION", f"{BASE}/tools/conclusion.tex")
V["conclusion"] = CONCLUSION[SUPPORT]() + " " + (
    open(CONCLUSION_PATH, encoding="utf-8").read().strip() if os.path.exists(CONCLUSION_PATH)
    else "CONCLUSION SENTENCES PENDING.")

# The qualitative reading of Figure 1, written after viewing both grids at zoom. It is prose
# about images, so it carries no number; the claims it makes are re-checked by eye in review.
QUALITATIVE_PATH = os.environ.get("P7_QUALITATIVE", f"{BASE}/tools/qualitative.tex")
V["qualitative"] = (open(QUALITATIVE_PATH, encoding="utf-8").read().strip()
                    if os.path.exists(QUALITATIVE_PATH) else "QUALITATIVE PARAGRAPH PENDING.")
# The interpretation of the results, written once the numbers exist. Every number in it is a
# placeholder filled from V, so it cannot drift from the notebook.
INTERPRETATION_PATH = os.environ.get("P7_INTERPRETATION", f"{BASE}/tools/interpretation.tex")
V["interpretation"] = (open(INTERPRETATION_PATH, encoding="utf-8").read().strip()
                       if os.path.exists(INTERPRETATION_PATH)
                       else "INTERPRETATION PARAGRAPH PENDING.")
# Strengths, limitations and the recommendations drawn from them, after the training-time
# paragraph they draw on.
SYNTHESIS_PATH = os.environ.get("P7_SYNTHESIS", f"{BASE}/tools/synthesis.tex")
V["synthesis"] = (open(SYNTHESIS_PATH, encoding="utf-8").read().strip()
                  if os.path.exists(SYNTHESIS_PATH) else "SYNTHESIS PARAGRAPH PENDING.")

TEMPLATE = open(f"{BASE}/tools/report_template.tex", encoding="utf-8").read()
text = TEMPLATE
# Two passes, because the hand-written paragraphs carry placeholders of their own.
for _ in range(2):
    for key, value in V.items():
        text = text.replace(f"__{key.upper()}__", value)
# A sign in front of a number is typeset as a minus or plus, not a hyphen. Only signs after a
# space, "[" or "(" are touched, which leaves LaTeX lengths such as {-8pt} and exponents alone.
text = re.sub(r"(?<=[\s\[(])([+-])(?=\d)", r"$\1$", text)
leftover = re.findall(r"__[A-Z0-9_]+__", text)
assert not leftover, f"unfilled placeholders: {sorted(set(leftover))}"
for dash in ("\u2014", "\u2013"):
    assert dash not in text, "an em or en dash reached the report"

os.makedirs(REPORT, exist_ok=True)
io.open(f"{REPORT}/project7_report.tex", "w", encoding="utf-8", newline="\n").write(text)
print(f"wrote {REPORT}/project7_report.tex  ({len(text):,} characters)")
print(f"  values pulled from main_report.ipynb: {len(V)}")
print(f"  rule: cosine better in {COS_WINS} of {TOTAL} reduced cases, linear in {LIN_WINS}; "
      f"reading '{SUPPORT}'")
