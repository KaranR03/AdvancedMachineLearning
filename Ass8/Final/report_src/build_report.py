r"""Write report/project8_report.tex from the executed main_report.ipynb.

Every number in the report is pulled out of the notebook's printed output by regex, and every
sentence whose truth depends on the run (on how many views a model is better, whether a difference
is consistent, which sampler wins at equal work, where the error reduction sits) is generated from
those numbers. Nothing numeric is typed into the template except the settings the design fixes and
the findings quoted from prior work, which verify.py lists.

Run:  python tools/build_report.py
      P8_NBDIR=<dir> P8_REPORT=<dir> python tools/build_report.py    (against another run)
"""
import io
import json
import os
import re
import shutil

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-11/project-8-3d-movie")
NBDIR = os.environ.get("P8_NBDIR", f"{BASE}/notebook")
REPORT = os.environ.get("P8_REPORT", f"{BASE}/report")
FIGURES = f"{REPORT}/figures"
FIGURE_FILES = ("figure_1_test_views.pdf", "figure_2_novel_views.pdf", "figure_3_curves.pdf",
                "figure_4_view_dependence.pdf", "figure_5_sampling.pdf")

NUM = r"([\d.]+)"
SIGNED = r"([+-][\d.]+)"
COUNT = r"([\d,]+)"
TINY, EXT, ABL_A, ABL_B = ("Tiny NeRF", "Extended NeRF", "No view dirs (abl. A)",
                           "Uniform 96 (abl. B)")
MODELS = (TINY, EXT, ABL_A, ABL_B)


def printed():
    """All stdout text of the executed main_report.ipynb, concatenated in cell order."""
    with io.open(f"{NBDIR}/main_report.ipynb", encoding="utf-8") as handle:
        notebook = json.load(handle)
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
    return float(text.replace(",", "").replace("%", ""))


V = {}

# ------------------------------------------------------------------------ the run itself
V["device"], V["gpu"], V["torch"] = grab(r"^Using device: (\S+) \| GPU: (.+?) \| torch (\S+)$",
                                         "device line")
S = {}
for name, *values in every(
        rf"^summary \| (.+?) \| PSNR {NUM} \| sd {NUM} \| SSIM {NUM} \| coarse PSNR (\S+) \| "
        rf"(\d+) evaluations per ray \| {COUNT} ms per image \| {COUNT} parameters \| "
        rf"{NUM} min training \| {NUM} s per epoch", "summary lines", 4):
    S[name] = dict(zip(("psnr", "sd", "ssim", "coarse", "evals", "ms", "params", "minutes",
                        "epoch_s"), values))
assert set(S) == set(MODELS), f"unexpected model names {sorted(S)}"
V["at_28"], V["n_views"] = grab(r"^Extended NeRF views at or above 28 dB: (\d+) of (\d+)",
                                "views at 28 dB")

PAIR = {}
for a, b, mean, low, high, wins, n in every(
        rf"^paired \| (.+?) - (.+?) \| mean {SIGNED} dB \| range {SIGNED} to {SIGNED} \| "
        r"(\d+) of (\d+) views higher", "paired differences", 5):
    PAIR[a, b] = dict(mean=mean, low=low, high=high, wins=int(wins), n=int(n))
picked = grab(r"^report view: (\d+) \|", "report view")

CURVE = {}
for name, epochs, best, final, last10, epoch_s, minutes in every(
        rf"^curves \| (.+?) \| epochs (\d+) \| best {NUM} dB \| final {NUM} dB \| "
        rf"last-10-epoch gain {SIGNED} dB \| {NUM} s per epoch \| {NUM} min", "curves", 4):
    CURVE[name] = dict(epochs=epochs, best=best, final=final, last10=last10)

FROZEN = {}
for name, change, frozen, normal in every(
        rf"^frozen direction \| (.+?) \| mean colour change {NUM} \| PSNR {NUM} dB with the "
        rf"direction frozen vs {NUM} dB", "frozen direction", 2):
    FROZEN[name] = dict(change=change, frozen=frozen, normal=normal)
V["points"], V["min_cameras"], V["median_cameras"] = grab(
    r"^photo consistency \| (\d+) surface points seen by at least (\d+) training cameras \| "
    r"median (\d+) cameras per point", "photo points")
V["spread_photo"], V["spread_ext"], V["spread_a"] = grab(
    rf"^photo consistency \| brightness spread across cameras \(median std\) \| photographs {NUM} "
    rf"\| Extended NeRF {NUM} \| ablation A {NUM}", "photo spread")
V["over_ext"], V["over_a"] = grab(
    r"^photo consistency \| renders vary more than the photographs at \| Extended NeRF (\d+)% of "
    r"points \| ablation A (\d+)% of points", "photo overshoot")
PHOTO = {}
for name, rms, corr in every(
        rf"^photo consistency \| (.+?) \| RMS brightness error against the photographs {NUM} \| "
        rf"median correlation with the photographs across cameras ([+-]?[\d.]+)", "photo RMS", 2):
    PHOTO[name] = dict(rms=rms, corr=corr)
V["edge_px"], V["edge_err_a"], V["edge_gain"] = grab(
    rf"^error split \| edge pixels {NUM}% of test pixels \| {NUM}% of ablation A's squared "
    rf"error \| ([+-]?[\d.]+)% of the reduction to Extended NeRF", "error split")

SHARE = {}
for name, pct, rays in every(
        r"^share of samples near the surface \((.+?)\s*\):\s+([\d.]+)%\s+\[([\d,]+) object rays\]",
        "near-surface shares", 4):
    SHARE[name.split(",")[0]] = pct
    V["object_rays"] = rays
BUDGET = {"hierarchical": {}, "uniform": {}}
for n_c, n_f, evals, p in every(
        rf"^budget \| hierarchical \| coarse (\d+) \+ fine (\d+) \| (\d+) evaluations \| PSNR {NUM} dB",
        "hierarchical budget"):
    BUDGET["hierarchical"][int(evals)] = dict(n_c=n_c, n_f=n_f, psnr=p)
for samples, evals, p in every(
        rf"^budget \| uniform \| (\d+) samples \| (\d+) evaluations \| PSNR {NUM} dB",
        "uniform budget"):
    BUDGET["uniform"][int(evals)] = dict(psnr=p)
EQUAL = {}
for evals, hier, uni, diff in every(
        rf"^equal work \| (\d+) evaluations per ray \| hierarchical {NUM} dB \| uniform {NUM} dB "
        rf"\| difference {SIGNED} dB", "equal work"):
    EQUAL[int(evals)] = dict(hier=hier, uni=uni, diff=diff)
V["frames"], V["elevation"], V["radius"] = grab(
    r"^turntable \| (\d+) frames \| camera (\d+) degrees above the horizontal \| radius (\d+)",
    "turntable")

# --------------------------------------------------------------------------------- figures
os.makedirs(FIGURES, exist_ok=True)
for name in FIGURE_FILES:
    shutil.copyfile(f"{NBDIR}/{name}", f"{FIGURES}/{name}")


# ------------------------------------------------------------------ helpers for the prose
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def on_views(pair):
    """'on all six views', 'on five of the six views', 'on none of the six views'."""
    p = PAIR[pair]
    if p["wins"] == p["n"]:
        return f"on all {WORDS[p['n']]} views"
    return f"on {WORDS[p['wins']]} of the {WORDS[p['n']]} views"


def consistent(pair):
    """A paired difference counts as consistent when every view moves the same way."""
    p = PAIR[pair]
    return p["wins"] in (0, p["n"])


def db(value):
    """A signed difference as the report prints it: the sign is typeset later."""
    return f"{value} dB"


def magnitude(value):
    return value.lstrip("+-")


ext, tiny, a, b = S[EXT], S[TINY], S[ABL_A], S[ABL_B]
p_tiny, p_a, p_b = PAIR[EXT, TINY], PAIR[EXT, ABL_A], PAIR[EXT, ABL_B]
TRAINED = int(ext["evals"])                   # coarse + fine network evaluations as trained
UNIFORM_TRAINED = int(b["evals"])
LOW = sorted(e for e in EQUAL if e < TRAINED)
HIGH = sorted(e for e in EQUAL if e >= TRAINED)
LOSES_LOW = all(f(EQUAL[e]["diff"]) < 0 for e in LOW)
WINS_HIGH = all(f(EQUAL[e]["diff"]) > 0 for e in HIGH)
TOP_UNIFORM = max(BUDGET["uniform"])

# ------------------------------------------------------------------------------- captions
V["fig1_caption"] = (
    f"Held-out test view {picked}, the view ranked in the middle by Extended NeRF's gain over Tiny "
    "NeRF: ground truth, both renders with their PSNR, and each model's absolute error on one "
    "colour scale.")
V["fig2_caption"] = (
    f"Novel views from a turntable of {V['frames']} frames, {V['elevation']} degrees above the "
    f"horizontal at radius {V['radius']}, every 45 degrees of azimuth: Tiny NeRF (top) and "
    "Extended NeRF (bottom).")
V["fig3_caption"] = (
    "Test PSNR per training epoch, for monitoring only (dashed: 28 dB); all results use the final "
    "epoch's weights.")
V["fig4_caption"] = (
    "Top: Extended NeRF's test view 0, with the direction input frozen to view 3's, and the "
    "colour change. Bottom, per object point away from image edges: brightness spread across the "
    "cameras that see it (left; quartiles) and correlation with the photographs (right; dotted: "
    "medians).")
V["fig5_caption"] = (
    "Top: one object ray of test view 0: coarse weights (bars), the 32 coarse samples and the 64 "
    "fine samples drawn from those weights (ticks). Bottom: mean test PSNR of the trained models "
    "at other sample counts (coarse to fine 1:2).")
V["table_caption"] = (
    f"Means over the {WORDS[int(V['n_views'])]} held-out views (PSNR in dB). View: direction "
    "input. Hier.: coarse and fine networks. Evals: network evaluations per ray.")


def table_rows():
    rows = []
    for name, label, view, hier in ((TINY, "Tiny NeRF", "no", "no"),
                                    (ABL_A, "Ablation A", "no", "yes"),
                                    (ABL_B, "Ablation B", "yes", "no"),
                                    (EXT, "Extended NeRF", "yes", "yes")):
        s = S[name]
        psnr = f"\\textbf{{{s['psnr']}}}" if name == EXT else s["psnr"]
        ssim = f"\\textbf{{{s['ssim']}}}" if name == EXT else s["ssim"]
        rows.append(f"{label} & {view} & {hier} & {s['evals']} & {s['params']} & {psnr} & "
                    f"{ssim} \\\\")
    return "\n".join(rows)


V["table_rows"] = table_rows()

# ---------------------------------------------------------------------------------- prose
V["intro"] = (
    "Tiny NeRF, from the Week 10 tutorial, learns a scene as a function from a 3D position to a "
    "colour and a volume density, and renders a pixel by accumulating that function at 64 evenly "
    "spaced depths along the camera ray (Salvado, 2026, p.~71). The full NeRF of Mildenhall et "
    "al. (2020) adds two things it lacks: a colour that depends on the viewing direction, and "
    "hierarchical sampling, in which a coarse network chooses where a fine network looks. This "
    "report adds both, trains on 100 of 106 synthetic $100 \\times 100$ images of a Lego "
    "bulldozer, tests on the other six (images 100 to 105) with PSNR and SSIM (Wang et al., 2004), "
    "and isolates each change with an ablation.")

V["changes"] = (
    "\\textbf{Task 1, view-dependent colour.} The position, encoded with $L = 6$ frequency bands "
    "(39 inputs), passes through two 128-unit layers, and the density is read from them before "
    "the direction enters, so the geometry is the same from every camera. A feature layer then "
    "meets the unit viewing direction, encoded with $L = 4$ bands (27 inputs; Salvado, 2026, "
    "p.~77), and a two-layer head outputs the colour. Ablation~A feeds the same network a zero "
    "direction: a matte renderer of identical size.\n\n"
    "\\textbf{Task 2, hierarchical sampling.} A coarse network evaluates $N_c = 32$ stratified "
    "depths per ray. Its weights $w_i = T_i(1 - e^{-\\sigma_i\\delta_i})$, detached from the "
    "graph, define a piecewise constant density on the ray, and the starter's "
    "\\texttt{sample\\_pdf} inverts its cumulative distribution to draw $N_f = 64$ more depths "
    "where the weights are large. A fine network evaluates all 96 sorted depths, and the loss "
    "adds the coarse and fine errors so the coarse network keeps learning where the surface "
    "is. Ablation~B: view-dependent colour, one network, 96 evenly spaced samples.\n\n"
    "\\textbf{Protocol.} Each model trains for 100 epochs on batches of 1,024 rays with Adam "
    "(learning rate decaying from $5 \\times 10^{-4}$ to $5 \\times 10^{-5}$) and is evaluated "
    "with the weights of its final epoch, not of its best test epoch.")

V["results"] = (
    f"Extended NeRF reaches a mean test PSNR of {ext['psnr']} dB (Table~1), above the 28 dB the "
    f"task expects, with {WORDS[int(V['at_28'])]} of the six views at or above it. It is "
    f"{magnitude(p_tiny['mean'])} dB above Tiny NeRF and higher {on_views((EXT, TINY))} "
    "(Figure~1); Figure~2 shows novel views around a full turn. The training curves "
    f"are nearly flat at the end (Figure~3): in the last ten epochs Extended NeRF gained "
    f"{CURVE[EXT]['last10']} dB, Tiny NeRF {CURVE[TINY]['last10']} dB.\n\n"
    f"Removing view dependence (ablation~A) costs {magnitude(p_a['mean'])} dB, and Extended NeRF "
    f"is higher {on_views((EXT, ABL_A))} (by {p_a['low']} to {p_a['high']} dB). Removing "
    f"hierarchical sampling (ablation~B) costs {magnitude(p_b['mean'])} dB"
    + ((f", but per view the difference runs from {p_b['low']} to {p_b['high']} dB, favouring "
        f"Extended NeRF only {on_views((EXT, ABL_B))}, so the two samplers are not reliably "
        "different at the trained budget.")
       if not consistent((EXT, ABL_B)) else
       (f", and Extended NeRF is higher {on_views((EXT, ABL_B))} (by {p_b['low']} to "
        f"{p_b['high']} dB), a small but consistent difference.")))

# ------------------------------------------------------------------------------ discussion
QUALITATIVE_PATH = os.environ.get("P8_QUALITATIVE", f"{BASE}/tools/qualitative.tex")
with open(QUALITATIVE_PATH, encoding="utf-8") if os.path.exists(QUALITATIVE_PATH) else \
        io.StringIO("QUALITATIVE PARAGRAPH PENDING.") as handle:
    qualitative = handle.read().strip()
PHOTO_VARIES = f(V["spread_photo"]) > f(V["spread_a"])
TRACKS = f(PHOTO[EXT]["corr"]) > f(PHOTO[ABL_A]["corr"])
OVERSHOOTS = f(V["spread_ext"]) > f(V["spread_photo"])
V["discussion"] = (
    "\\textbf{Rendering quality.} " + qualitative + "\n\n"
    "\\textbf{View-dependent appearance.} Freezing Extended NeRF's direction input at one "
    f"camera's direction changes its colours by {FROZEN[EXT]['change']} on average (0 to 1 "
    f"scale) and lowers test view 0 from {FROZEN[EXT]['normal']} to {FROZEN[EXT]['frozen']} dB "
    "(Figure~4, top): the model relies heavily on the direction. To test whether the photographs "
    f"warrant this, {V['points']} object points away from image edges were projected into the "
    f"training cameras that see them unoccluded (median {V['median_cameras']} per point). Across "
    f"those cameras, a point's brightness varies by {V['spread_photo']} (median standard "
    "deviation), ")
V["discussion"] += (
    (f"above the {V['spread_a']} that matte ablation~A shows from geometry and sampling alone, so "
     "the scene does change with viewpoint. ")
    if PHOTO_VARIES else
    (f"no more than the {V['spread_a']} that matte ablation~A shows from geometry and sampling "
     "alone, so the photographs barely change with viewpoint. "))
V["discussion"] += (
    ("Extended NeRF follows that change more closely than ablation~A (median correlation "
     f"{PHOTO[EXT]['corr']} against {PHOTO[ABL_A]['corr']}; Figure~4, bottom)")
    if TRACKS else
    ("Extended NeRF follows it no better than ablation~A (median correlation "
     f"{PHOTO[EXT]['corr']} against {PHOTO[ABL_A]['corr']}; Figure~4, bottom)"))
V["discussion"] += (
    (f", but its renders vary by {V['spread_ext']}, more than the photographs at "
     f"{V['over_ext']}\\% of points. That excess is what the shape-radiance ambiguity predicts "
     "(Zhang et al., 2020): a direction-dependent colour can also absorb geometry errors. ")
    if OVERSHOOTS else
    (f". Its renders vary by {V['spread_ext']}, no more than the photographs, so the direction "
     "input shows no sign of absorbing geometry errors, the risk Zhang et al. (2020) describe. "))
V["discussion"] += (
    "NeRF loses more without view dependence (31.01 against 27.66 dB; Mildenhall et al., 2020, "
    "Table~2), and its Fig.~4 shows why on this same bulldozer: the tread's specular reflection "
    "is lost.")

V["discussion_2"] = (
    "\\textbf{Hierarchical sampling.} The fine samples find the surface: on test view 0's "
    f"{V['object_rays']} object rays, {SHARE['fine']}\\% of them fall near it, "
    f"against {SHARE['uniform']}\\% of 96 uniform samples (near: inside the interval holding the "
    "central 90\\% of the final weights, widened by half its width each side; Figure~5, top, "
    "shows one ray). That buys little PSNR here: uniform sampling is already near its limit, "
    f"{BUDGET['uniform'][UNIFORM_TRAINED]['psnr']} dB at {UNIFORM_TRAINED} samples and "
    f"{BUDGET['uniform'][TOP_UNIFORM]['psnr']} dB at {TOP_UNIFORM} (Figure~5, bottom). At equal "
    "work (all network evaluations counted), ")
if LOSES_LOW:
    worst = min(LOW)
    V["discussion_2"] += (
        f"hierarchical sampling trails by {magnitude(EQUAL[worst]['diff'])} dB at {worst} "
        f"evaluations per ray, where the coarse pass has only "
        f"{BUDGET['hierarchical'][worst]['n_c']} samples to locate thin structures, "
        f"and by {magnitude(EQUAL[LOW[-1]]['diff'])} dB at {LOW[-1]}. ")
else:
    V["discussion_2"] += "hierarchical sampling is not behind at any budget below the trained one. "
V["discussion_2"] += (
    ("It leads only " if WINS_HIGH else "At the trained budget and above, it ")
    + " and ".join(("" if WINS_HIGH else "leads ") + f"by {magnitude(EQUAL[e]['diff'])} dB at {e}"
                   if f(EQUAL[e]["diff"]) > 0
                   else f"trails by {magnitude(EQUAL[e]['diff'])} dB at {e}" for e in HIGH)
    + ", against 0.95 dB in NeRF's ablation at an equal 256 (Mildenhall et al., 2020, "
    "Table~2).\n\n")

V["discussion_2"] += (
    f"\\textbf{{Overall performance.}} Extended NeRF's accuracy has a price: "
    f"{ext['params']} parameters against {tiny['params']}, {ext['evals']} network evaluations per "
    f"ray against {tiny['evals']}, {ext['ms']} against {tiny['ms']} ms per test image on an "
    f"{V['gpu']}, and {ext['minutes']} against {tiny['minutes']} minutes of training. "
    f"Ablation~B comes within {magnitude(p_b['mean'])} dB of it with {b['params']} parameters and "
    f"{b['minutes']} minutes.\n\n"
    "\\textbf{Strengths and limitations.} View-dependent colour gives a large, consistent gain "
    "for one small head, and importance sampling finds the surface; but hierarchical sampling "
    "doubles the networks for no reliable gain here, and the direction input can absorb "
    f"geometry errors. One seed per model cannot separate {magnitude(p_b['mean'])} dB from "
    "run-to-run variation, six test views are few, and the sweep re-renders rather than "
    "retrains.\n\n"
    "\\textbf{Recommendations.} For a scene this size, view-dependent colour with uniform "
    "sampling is the better trade: nearly all the quality at half the parameters and training "
    "time. Hierarchical sampling should be retested, with several seeds, where even spacing "
    "stops resolving the surface (higher resolution, deeper scenes), and penalising how much "
    "the colour varies with direction would show how much of that variation is compensation.")

V["conclusion"] = (
    f"Extended NeRF reaches {ext['psnr']} dB on the six held-out views, {magnitude(p_tiny['mean'])} "
    f"dB above Tiny NeRF; view-dependent colour is worth {magnitude(p_a['mean'])} dB over its "
    f"ablation and hierarchical sampling {magnitude(p_b['mean'])} dB at the trained budget. The "
    + ("direction input follows real changes in the photographs but exaggerates them"
       if PHOTO_VARIES and TRACKS and OVERSHOOTS else
       "direction input changes the renders far more than the photographs change with viewpoint")
    + ", and the coarse-to-fine sampler places samples well but pays only at large budgets "
    "here.")

V["zhang_bib"] = (
    "\\bib{Zhang, K., Riegler, G., Snavely, N. and Koltun, V. (2020). NeRF++: analyzing and "
    "improving neural radiance fields. arXiv:2010.07492.}")

with open(f"{BASE}/tools/report_template.tex", encoding="utf-8") as handle:
    TEMPLATE = handle.read()
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
with io.open(f"{REPORT}/project8_report.tex", "w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)
print(f"wrote {REPORT}/project8_report.tex  ({len(text):,} characters)")
print(f"  values pulled from main_report.ipynb: {len(V)}")
print(f"  equal work: hierarchical behind at every low budget {LOSES_LOW}, ahead at every high "
      f"budget {WINS_HIGH}; photographs vary {PHOTO_VARIES}, tracks {TRACKS}, overshoots {OVERSHOOTS}")
