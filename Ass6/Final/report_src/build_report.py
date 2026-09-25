r"""Write report/project6_report.tex from the executed main_report.ipynb.

Every number in the report is pulled out of the notebook's printed output by regex, and every
sentence whose truth depends on the run (a direction, whether an interval excludes zero, whether
the seeds agree) is generated from those numbers. Nothing numeric is typed into the template except
the settings the design fixes and the findings quoted from prior work, which verify.py lists.

Run:  python tools/build_report.py
      P6_NBDIR=<dir> P6_REPORT=<dir> python tools/build_report.py    (against another run)
"""
import io
import json
import os
import re
import shutil

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-09/project-6-sharpness-quest")
NBDIR = os.environ.get("P6_NBDIR", f"{BASE}/notebook")
REPORT = os.environ.get("P6_REPORT", f"{BASE}/report")
FIGURES = f"{REPORT}/figures"

NUM = r"([\d.]+)"
SIGNED = r"([+-][\d.]+)"
COUNT = r"([\d,]+)"


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


def values(text):
    """'+0.0123 -0.0045 +0.0100' -> the list of those strings."""
    return text.split()


V = {}

# ------------------------------------------------------------------------ setup and records
V["device"] = grab(r"^device\s*:\s*(\S+)", "device")
V["torch"] = grab(r"^torch\s*:\s*(\S+)", "torch version")
V["pixels_one"] = grab(rf"^pixels exactly 1\.0 : {NUM}", "share of pixels at 1.0")
V["n_cvae"], V["n_disc"], V["n_clf"] = grab(
    rf"^parameters\s*:\s*cVAE {COUNT} \| discriminator {COUNT} \| classifier {COUNT}",
    "parameter counts")
V["lambda"] = grab(rf"^lambda \(chosen on the development split\) : {NUM}", "lambda")
for name, key in (("baseline", "base"), ("discriminator loss", "adv")):
    epochs, seeds, n_refit, device, seconds = grab(
        rf"^{name}\s*\| epochs (\d+) \| seeds \[([\d, ]+)\] \| refit on {COUNT} images \| "
        rf"trained on (\S+) \| seconds per epoch {NUM}", f"{name} record")
    V[f"epochs_{key}"], V[f"seeds_{key}"] = epochs, seeds
    V[f"n_refit_{key}"], V[f"device_{key}"], V[f"seconds_{key}"] = n_refit, device, seconds
assert V["epochs_base"] == V["epochs_adv"], "the two models trained for different epoch counts"
assert V["seeds_base"] == V["seeds_adv"], "the two models trained with different seeds"
V["epochs"], V["n_refit"] = V["epochs_base"], V["n_refit_base"]
SEEDS = [int(s) for s in V["seeds_base"].split(",")]
V["n_seeds"] = str(len(SEEDS))

SWEEP = re.findall(rf"^sweep \| {NUM} \| {NUM} \| {NUM} \| {NUM} \| {NUM}\s*$", T, re.MULTILINE)
assert SWEEP, "could not find the lambda sweep"
LAMBDAS = [row[0] for row in SWEEP]
V["real_dev_sharp"] = SWEEP[0][3]

for name, key in (("baseline", "base"), ("discriminator loss", "adv")):
    last, previous, change, within = grab(
        rf"^convergence \| {name} \| last {NUM} \| previous {NUM} \| change {SIGNED} \| "
        rf"within 1% (True|False)", f"{name} convergence")
    V[f"conv_last_{key}"], V[f"conv_prev_{key}"] = last, previous
    V[f"conv_change_{key}"], V[f"conv_ok_{key}"] = change, within
V["window"], V["dacc_final"] = grab(
    rf"discriminator accuracy, mean of the last (\d+) dev epochs : {NUM}", "D accuracy")

GRID = re.findall(rf"^grid accuracy \| (.+?) \| {NUM} of (\d+) images", T, re.MULTILINE)
assert len(GRID) == 2, "expected two grid accuracy lines"
V["grid_acc_base"], V["grid_acc_adv"] = GRID[0][1], GRID[1][1]

# --------------------------------------------------------------------------------- metrics
ROWS = {}
for label in ("mse_recon", "sharpness_reconstructions", "midgrey_reconstructions",
              "sharpness_samples", "midgrey_samples"):
    real, base, adv, delta, low, high, seeds = grab(
        rf"^metric \| {label} \| real (\S+) \| baseline {NUM} \| discriminator {NUM} \| "
        rf"delta {SIGNED} \[{SIGNED}, {SIGNED}\] \| seeds (.+)$", f"metric {label}")
    ROWS[label] = dict(real=real, base=base, adv=adv, delta=delta, low=low, high=high,
                       seeds=values(seeds))
    assert len(ROWS[label]["seeds"]) == len(SEEDS), f"{label}: one delta per seed expected"

DECISION = {kind: grab(rf"^decision \| {kind} sharper with the discriminator loss: (True|False)",
                       f"decision {kind}") == "True"
            for kind in ("reconstructions", "samples")}

V["clf_real_acc"], V["clf_real_conf"] = grab(
    rf"^classifier \| real test images \| accuracy {NUM} \| confidence {NUM}", "classifier real")
CLF = {}
for name in ("baseline", "discriminator"):
    accuracy, confidence, seeds = grab(
        rf"^classifier \| samples {name} \| accuracy {NUM} \| confidence {NUM} \| seeds (.+)$",
        f"classifier {name}")
    CLF[name] = dict(acc=accuracy, conf=confidence, seeds=values(seeds))

PER_CLASS = re.findall(rf"^class (\d) \| {NUM} \| {NUM} \| {NUM}\s*$", T, re.MULTILINE)
assert len(PER_CLASS) == 10, "expected ten per-class rows"

V["half_a"], V["half_b"] = grab(rf"^halves \| A {COUNT} \| B {COUNT}", "halves")
V["fd_floor"] = grab(rf"^frechet \| floor, real A against real B \| {NUM}", "FD floor")
FD = {}
for kind in ("samples", "reconstructions"):
    base, adv, seeds_base, seeds_adv = grab(
        rf"^frechet \| {kind} \| baseline {NUM} \| discriminator {NUM} \| seeds baseline (.+?) "
        rf"\| seeds discriminator (.+)$", f"FD {kind}")
    FD[kind] = dict(base=base, adv=adv, seeds_base=values(seeds_base),
                    seeds_adv=values(seeds_adv))
base, adv, seeds_base, seeds_adv = grab(
    rf"^spread \| baseline {NUM} \| discriminator {NUM} \| seeds baseline (.+?) \| "
    rf"seeds discriminator (.+)$", "spread")
SPREAD = dict(base=base, adv=adv, seeds_base=values(seeds_base), seeds_adv=values(seeds_adv))

# --------------------------------------------------------------------------------- figures
os.makedirs(FIGURES, exist_ok=True)
for name in ("figure_1_grids.pdf", "figure_2_reconstructions.pdf"):
    shutil.copyfile(f"{NBDIR}/{name}", f"{FIGURES}/{name}")


# ------------------------------------------------------------------ helpers for the prose
def f(text):
    return float(text)


NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def listed(words):
    """a, b and c"""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def seeds_phrase(row):
    """The other seeds' differences, as printed: '+0.0101 and +0.0120 at seeds 1 and 2'."""
    others = row["seeds"][1:]
    return f"{listed(others)} at seeds {listed([str(s) for s in SEEDS[1:]])}"


def interval(row):
    return f"{row['delta']} [{row['low']}, {row['high']}]"


def ci_excludes_zero(row):
    return f(row["low"]) > 0 or f(row["high"]) < 0


def all_same_sign(row):
    signs = {f(value) > 0 for value in row["seeds"]}
    return len(signs) == 1 and all(f(value) != 0 for value in row["seeds"])


SHARP_R, GREY_R = ROWS["sharpness_reconstructions"], ROWS["midgrey_reconstructions"]
SHARP_S, GREY_S = ROWS["sharpness_samples"], ROWS["midgrey_samples"]
MSE = ROWS["mse_recon"]
REAL_SHARP, REAL_GREY = SHARP_R["real"], GREY_R["real"]
V["real_sharp"], V["real_grey"] = REAL_SHARP, REAL_GREY


def verdict(kind, sharp, grey, supported, cite=False, again=False):
    """One sentence per object, stating which parts of the fixed rule held. The interval itself
    is in Table 1, so the sentence says only whether it clears zero. Only the first verdict cites
    the table, on its first numbers."""
    moved = "rises" if f(sharp["delta"]) > 0 else "falls"
    held = (f"Laplacian variance {moved} from {sharp['base']} to {sharp['adv']}"
            + (" (Table~1; Figure~2a)" if cite else ""))
    if supported and again:
        # The rule was just spelled out for reconstructions, so the samples' sentence is short.
        return (f"The samples meet it too: Laplacian variance rises from {sharp['base']} to "
                f"{sharp['adv']} ({seeds_phrase(sharp)}), and the mid-grey fraction falls from "
                f"{grey['base']} to {grey['adv']}.")
    if supported:
        return (f"For {kind} the rule is met: {held}, with an interval above zero and differences "
                f"of {seeds_phrase(sharp)}, and the mid-grey fraction falls from {grey['base']} to "
                f"{grey['adv']}.")
    failed = []
    if not f(sharp["low"]) > 0:
        failed.append("its interval does not lie above zero")
    if not all(f(value) > 0 for value in sharp["seeds"]):
        failed.append(f"the seeds differ ({seeds_phrase(sharp)})")
    if not f(grey["delta"]) < 0:
        failed.append(f"the mid-grey fraction does not fall ({grey['base']} to {grey['adv']})")
    return f"For {kind} the rule is not met: {held}, but {listed(failed)}."


V["verdict_recon"] = verdict("reconstructions", SHARP_R, GREY_R, DECISION["reconstructions"],
                             cite=True)
V["verdict_samples"] = verdict("samples", SHARP_S, GREY_S, DECISION["samples"],
                               again=DECISION["reconstructions"])

# Where the discriminator's sharpness sits against the real images' own value: moving toward it
# is the claim; passing it would mean noise or ringing rather than a crisper stroke.
BELOW_REAL = f(SHARP_R["adv"]) < f(REAL_SHARP)
V["real_clause"] = (
    f"Both values remain below the real test images' {REAL_SHARP}, so the change moves "
    "reconstructions toward real sharpness, not past it."
    if BELOW_REAL and f(SHARP_R["base"]) < f(REAL_SHARP) else
    f"The discriminator's value exceeds the real test images' {REAL_SHARP}, which Laplacian "
    "variance alone would read as sharper than real; the mid-grey fraction is the check on "
    "whether that excess is crisper strokes or added noise.")

# Pixel fidelity: the discriminator is expected to trade a little MSE for realism.
MSE_UP = f(MSE["delta"]) > 0
V["mse_sentence"] = (
    f"The price is pixel fidelity: reconstruction MSE rises from {MSE['base']} to {MSE['adv']}, "
    "beyond evaluation noise. Pixel error punishes small shifts (Larsen et al.): a crisp stroke "
    "slightly misplaced costs more than a blur over both places."
    if MSE_UP and ci_excludes_zero(MSE) else
    f"Reconstruction MSE changes from {MSE['base']} to {MSE['adv']} ($\\Delta$ = "
    f"{interval(MSE)}), so the sharper strokes cost no measurable pixel fidelity."
    if not MSE_UP else
    f"Reconstruction MSE moves from {MSE['base']} to {MSE['adv']} ($\\Delta$ = "
    f"{interval(MSE)}), a difference within evaluation noise.")

# Training variance made concrete: when the per-seed MSE costs differ by more than a factor of
# two, the limitation sentence quotes their range, which Table 1 prints.
MSE_SEEDS = sorted(MSE["seeds"], key=f)
V["seed_spread_clause"] = (
    f" (the MSE cost ranges from {MSE_SEEDS[0]} to {MSE_SEEDS[-1]} across seeds)"
    if f(MSE_SEEDS[-1]) > 2 * f(MSE_SEEDS[0]) > 0 else "")

# Class consistency against the real images' accuracy: within one point counts as matching.
GAP_BASE = f(V["clf_real_acc"]) - f(CLF["baseline"]["acc"])
GAP_ADV = f(V["clf_real_acc"]) - f(CLF["discriminator"]["acc"])
V["clf_sentence"] = (
    f"Both models' samples are read as their intended class about as often as real digits "
    f"are ({CLF['baseline']['acc']} and {CLF['discriminator']['acc']} against "
    f"{V['clf_real_acc']})"
    if max(GAP_BASE, GAP_ADV) <= 0.01 else
    f"The classifier recognises {CLF['baseline']['acc']} of the baseline's samples and "
    f"{CLF['discriminator']['acc']} of the discriminator-loss samples as their intended class, "
    f"against {V['clf_real_acc']} for real test digits")

FD_S, FD_R = FD["samples"], FD["reconstructions"]
FD_S_BETTER = f(FD_S["adv"]) < f(FD_S["base"])
FD_S_SEEDS = all((f(a) < f(b)) == FD_S_BETTER for a, b in zip(FD_S["seeds_adv"],
                                                                FD_S["seeds_base"]))
V["fd_sentence"] = (
    f"In the classifier's feature space the samples' Fréchet distance to real digits "
    f"{'falls' if FD_S_BETTER else 'rises'} from {FD_S['base']} to {FD_S['adv']}"
    f"{', at every seed' if FD_S_SEEDS else ', though not at every seed'} "
    f"(the real-against-real floor is {V['fd_floor']}), and the reconstructions' from "
    f"{FD_R['base']} to {FD_R['adv']}.")
# Variety: a relative change within 5% counts as unchanged. The Conclusion uses the same bands,
# so the two sections cannot describe one number two ways.
SPREAD_CHANGE = f(SPREAD["adv"]) / f(SPREAD["base"]) - 1
SPREAD_BAND = "same" if abs(SPREAD_CHANGE) < 0.05 else ("down" if SPREAD_CHANGE < 0 else "up")
V["spread_sentence"] = (
    f"Within-class spread is {SPREAD['base']} for the baseline and {SPREAD['adv']} with the "
    "discriminator loss (1 matches real variety), so "
    + {"down": "samples with the discriminator loss are less varied, a step toward the mode "
               "collapse GANs are known for.",
       "same": "the discriminator loss left the samples' variety essentially unchanged.",
       "up": "samples with the discriminator loss are more varied, not less."}[SPREAD_BAND])

# The sweep: whether the chosen lambda sits at an edge of the range tried.
V["sweep_range"] = f"\\{{{', '.join(LAMBDAS)}\\}}"
AT_EDGE = V["lambda"] in (LAMBDAS[0], LAMBDAS[-1])
V["edge_clause"] = (
    " It lies at the edge of the range tried, so a wider sweep could move it." if AT_EDGE else "")
SWEEP_TEX = "\n".join(
    f"{lam} & {recon} & {sharp} & {dacc} \\\\" for lam, recon, sharp, _, dacc in SWEEP)

V["sweep_sentence"] = (
    f"In that order, the {NUMBER_WORDS[len(SWEEP)]} runs' reconstructions scored "
    f"{listed([row[2] for row in SWEEP])}, with $D$ accuracies of "
    f"{listed([row[4] for row in SWEEP])}, so $\\lambda = {V['lambda']}$ was chosen."
    + V["edge_clause"])

CONV_OK = V["conv_ok_base"] == "True" and V["conv_ok_adv"] == "True"
V["conv_sentence"] = (
    f"both runs meet it (relative changes of {V['conv_change_base']} and "
    f"{V['conv_change_adv']}; "
    "Figure~2b)."
    if CONV_OK else
    f"the changes are {V['conv_change_base']} for the baseline and {V['conv_change_adv']} with "
    "the discriminator loss (Figure~2b), so the rule is not met.")

# Conclusion, generated from the decision rule.
if DECISION["reconstructions"] and DECISION["samples"]:
    V["conclusion"] = (
        "The evidence supports the hypothesis: under a rule fixed before the results, adding a "
        "discriminator term made both the reconstructions and the samples of a cVAE measurably "
        f"sharper, consistently across {NUMBER_WORDS[len(SEEDS)]} training seeds.")
elif DECISION["reconstructions"]:
    V["conclusion"] = (
        "The evidence supports the hypothesis as stated, for reconstructions: adding a "
        f"discriminator term made them measurably sharper across {NUMBER_WORDS[len(SEEDS)]} "
        "training seeds, while the samples did not meet the same rule.")
elif DECISION["samples"]:
    V["conclusion"] = (
        "The evidence does not support the hypothesis as stated: the discriminator term sharpened "
        "the samples by the fixed rule but not the reconstructions the hypothesis names.")
else:
    V["conclusion"] = (
        "The evidence does not support the hypothesis: by a rule fixed before the results, the "
        "discriminator term did not make the reconstructions or the samples measurably sharper.")

# The Conclusion's second finding: what the discriminator cost or kept, from the numbers the
# Discussion reports and with the same thresholds it uses.
MSE_REL = f(MSE["delta"]) / f(MSE["base"])
FIDELITY = ("Pixel error did not rise measurably" if not (MSE_UP and ci_excludes_zero(MSE)) else
            "Pixel error rose slightly" if MSE_REL < 0.1 else "Pixel error rose")
CLF_DELTA = f(CLF["discriminator"]["acc"]) - f(CLF["baseline"]["acc"])
CONSISTENCY = ("class consistency was unchanged" if abs(CLF_DELTA) < 0.01 else
               "class consistency improved" if CLF_DELTA > 0 else "class consistency fell")
VARIETY = {"same": "variety was kept", "down": "variety fell", "up": "variety grew"}[SPREAD_BAND]
V["conclusion"] += f" {FIDELITY}, {CONSISTENCY} and {VARIETY}."

# The further experiment. While the discriminator's reconstructions stay below the real images'
# sharpness some blur remains, and a shift-tolerant loss tests whether pixel MSE causes it.
V["conclusion_next"] = (
    "Next, Larsen et al.'s shift-tolerant, feature-wise reconstruction loss would test whether "
    "the remaining blur comes from pixel MSE itself."
    if BELOW_REAL else
    "Next, Larsen et al.'s feature-space reconstruction loss would separate the adversarial "
    "term's effect from the pixel metric's.")


def table_row(name, row, digits_real=True):
    real = row["real"]    # "n/a" where no real value exists; "--" would typeset as an en dash
    others = ", ".join(row["seeds"][1:])    # one cell: the header names both seeds
    return (f"{name} & {real} & {row['base']} & {row['adv']} & {row['delta']} & "
            f"[{row['low']}, {row['high']}] & {others} \\\\")


V["table_rows"] = "\n".join([
    table_row("Reconstruction MSE", MSE),
    table_row("Laplacian variance, reconstructions", SHARP_R),
    table_row("Mid-grey fraction, reconstructions", GREY_R),
    table_row("Laplacian variance, samples", SHARP_S),
    table_row("Mid-grey fraction, samples", GREY_S),
])
V["table_rows_2"] = "\n".join([
    f"Classifier accuracy, samples & {V['clf_real_acc']} & {CLF['baseline']['acc']} & "
    f"{CLF['discriminator']['acc']} & \\multicolumn{{3}}{{l}}{{seeds: "
    f"{' '.join(CLF['baseline']['seeds'])} / {' '.join(CLF['discriminator']['seeds'])}}} \\\\",
    f"Fréchet distance, samples & {V['fd_floor']} & {FD_S['base']} & {FD_S['adv']} & "
    f"\\multicolumn{{3}}{{l}}{{seeds: {' '.join(FD_S['seeds_base'])} / "
    f"{' '.join(FD_S['seeds_adv'])}}} \\\\",
    f"Fréchet distance, reconstructions & {V['fd_floor']} & {FD_R['base']} & {FD_R['adv']} & "
    f"\\multicolumn{{3}}{{l}}{{seeds: {' '.join(FD_R['seeds_base'])} / "
    f"{' '.join(FD_R['seeds_adv'])}}} \\\\",
    f"Within-class spread, samples & 1 & {SPREAD['base']} & {SPREAD['adv']} & "
    f"\\multicolumn{{3}}{{l}}{{seeds: {' '.join(SPREAD['seeds_base'])} / "
    f"{' '.join(SPREAD['seeds_adv'])}}} \\\\",
])
V["sweep_rows"] = SWEEP_TEX

# The qualitative reading of Figure 1, written after viewing both grids at zoom. It is prose
# about images, so it carries no number; the claims it makes are re-checked by eye in review.
QUALITATIVE_PATH = os.environ.get("P6_QUALITATIVE", f"{BASE}/tools/qualitative.tex")
QUALITATIVE = open(QUALITATIVE_PATH, encoding="utf-8").read().strip() \
    if os.path.exists(QUALITATIVE_PATH) else "QUALITATIVE PARAGRAPH PENDING."
V["qualitative"] = QUALITATIVE

TEMPLATE = open(f"{BASE}/tools/report_template.tex", encoding="utf-8").read()
text = TEMPLATE
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
io.open(f"{REPORT}/project6_report.tex", "w", encoding="utf-8", newline="\n").write(text)
print(f"wrote {REPORT}/project6_report.tex  ({len(text):,} characters)")
print(f"  values pulled from main_report.ipynb: {len(V)}")
print(f"  decision: reconstructions {DECISION['reconstructions']}, samples {DECISION['samples']}")
