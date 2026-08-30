r"""Write report/project4_report.tex from the executed case notebooks.

Every number in the report is pulled out of a notebook's printed output by regex here, so
"the report reports what the code produced" is mechanically true rather than merely intended.
Nothing numeric is typed into the template.

Run:  python tools/build_report.py
"""
import io
import json
import os
import re
import shutil

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-06/project-4-ai-auditing")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
FIGURES = f"{REPORT}/figures"

NUMBER = r"(-?\d+\.\d+)"
INT = r"(\d+)"


def printed(case):
    """All stdout text of one executed notebook, concatenated in cell order."""
    notebook = json.load(io.open(f"{NBDIR}/case{case}.ipynb", encoding="utf-8"))
    chunks = [("".join(o.get("text", [])))
              for cell in notebook["cells"] for o in cell.get("outputs", [])
              if o.get("output_type") == "stream"]
    text = "\n".join(chunks)
    assert text.strip(), f"case{case}.ipynb has no printed output; execute it first"
    return text


def grab(text, pattern, label):
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"could not find {label}"
    return match.groups() if len(match.groups()) > 1 else match.group(1)


values = {}
for case in (1, 2, 3):
    text = printed(case)
    values[case] = {"text": text}
    for split in ("internal", "external"):
        row = grab(text, rf"^{split}\s+{NUMBER}\s+{NUMBER}\s+{NUMBER}\s+{NUMBER}\s+"
                         rf"{NUMBER}\s+{NUMBER}\s+{INT}\s+{INT}\s+{INT}\s+{INT}\s*$",
                   f"case {case} {split} metric row")
        for key, raw in zip(("acc", "prec", "rec", "f1", "auc", "ece",
                             "tp", "fp", "tn", "fn"), row):
            values[case][f"{split}_{key}"] = raw
        for gate in ("0.90", "0.99"):
            published, accuracy = grab(
                text, rf"^{split}\s+{re.escape(gate)}\s+{NUMBER}%\s+{NUMBER}\s*$",
                f"case {case} {split} gate {gate}")
            values[case][f"{split}_gate{gate[-2:]}_published"] = published
            values[case][f"{split}_gate{gate[-2:]}_accuracy"] = accuracy
    values[case]["sharp_ratio"] = grab(text, rf"ratio={NUMBER}x", f"case {case} sharpness ratio")
    # The notebook prints the shift signed. The report reads "0.262 brighter", so the sign is
    # carried by the sentence rather than repeated in the number.
    values[case]["bright_shift"] = grab(text, r"shift=\+?(-?\d+\.\d+)",
                                        f"case {case} brightness shift")
    values[case]["sharp_internal"] = grab(text, rf"sharpness  internal={NUMBER}", "sharpness")
    values[case]["sharp_external"] = grab(
        text, rf"sharpness  internal=[\d.]+  external={NUMBER}", "external sharpness")
    bands = {}
    for label in ("misclassified", "correct"):
        bands[label] = grab(text, rf"^{label}\s+{NUMBER}\s+{NUMBER}\s+{NUMBER}\s*$",
                            f"case {case} saliency bands, {label}")
    values[case]["band"] = bands

one, two, three = values[1], values[2], values[3]

# ------------------------------------------------------------------- case 1: the blur sweep
one["match_sigma"] = grab(one["text"], r"closest match: blur sigma ([\d.]+)", "matched sigma")
one["match_acc"], one["match_fp"] = grab(
    one["text"], rf"gives accuracy {NUMBER} and {INT} false positives", "matched sweep row")
one["bright_only_acc"] = grab(one["text"], rf"brightness alone reaches only {NUMBER}",
                              "brightness control")

# ------------------------------------------------------------------- case 2: the stratification
for label, key in (("women, no eyeglasses", "wn"), ("men, eyeglasses", "mg"),
                   ("men, no eyeglasses", "mn"), ("women, eyeglasses", "wg")):
    right, total, accuracy = grab(
        two["text"], rf"^{label}\s+\w+\s+{INT}/{INT}\s+{NUMBER}\s*$", f"subgroup {key}")
    two[f"{key}_right"], two[f"{key}_total"], two[f"{key}_acc"] = right, total, accuracy
two["seen_acc"] = grab(two["text"], rf"present in the internal set : \d+/\d+ = {NUMBER}", "seen")
two["unseen_acc"] = grab(two["text"], rf"absent from it\s+: \d+/\d+ = {NUMBER}", "unseen")
two["multiplier"] = grab(two["text"], r"error rate multiplies by (\d+)x", "multiplier")

# ------------------------------------------------------------------- case 3: the recalibration
three["temperature"] = grab(three["text"], r"fitted temperature T = ([\d.]+)", "temperature")
for label, key in (("uncalibrated", "raw"), ("calibrated", "cal")):
    row = grab(three["text"], rf"^{label}\s+{NUMBER}\s+{NUMBER}\s+{NUMBER}\s+{NUMBER}\s*$",
               f"calibration row {key}")
    for name, raw in zip(("ece", "auc", "acc", "maxconf"), row):
        three[f"{key}_{name}"] = raw
three["reachable"] = grab(three["text"],
                          rf"reachable at any coverage above 2%\s+: {NUMBER}", "reachable")
three["cal_published"] = grab(three["text"], r"0\.90\s+\d+ of \d+\s+[\d.]+\s+(\d+) of \d+",
                              "calibrated publish count")
three["held_out"] = grab(three["text"], r"0\.90\s+\d+ of \d+\s+[\d.]+\s+\d+ of (\d+)",
                         "held-out size")

TEMPLATE = r"""% IFN680 Project 4 technical report.
% Generated by tools/build_report.py. Every number below is printed by one of the case notebooks.
% Build:  tectonic project4_report.tex
\documentclass[10pt,a4paper,twocolumn]{article}

\usepackage[margin=1.5cm,top=0.9cm,bottom=0.8cm]{geometry}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[font=small,labelfont=bf,skip=3pt]{caption}
\usepackage{titlesec}

\titlespacing*{\section}{0pt}{4pt}{0.5pt}
\titleformat{\section}{\normalfont\bfseries}{\thesection.}{0.5em}{}
\setlength{\parskip}{0pt}
\setlength{\parindent}{1em}
\setlength{\columnsep}{0.6cm}
\setlength{\textfloatsep}{3pt}
\setlength{\intextsep}{3pt}
\setlength{\floatsep}{3pt plus 1pt minus 1pt}
\setlength{\tabcolsep}{4pt}
\pagestyle{empty}

\renewcommand{\topfraction}{0.94}
\renewcommand{\bottomfraction}{0.6}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.75}

\begin{document}

% Figure 1 sits inside the \twocolumn optional argument rather than in a figure* float. In the
% article class \twocolumn[...] claims the page's double column top region, so a figure* declared
% afterwards can never be placed on page 1 and is silently deferred to page 2.
\twocolumn[{%
\begin{center}
{\large\bfseries Project 4: Auditing Three Pretrained CelebA Classifiers}\\[2pt]
{\bfseries Group 4}
\end{center}
\vspace{-2pt}
Student 1: Karan Rooprai \hfill Student 1 ID: n12498122 \\
Student 2: Nhu Hieu Nguyen \hfill Student 2 ID: n12194778 \\
\vspace{3pt}

\begin{center}
\includegraphics[width=0.315\textwidth]{figures/case1_confusion.pdf}\hfill
\includegraphics[width=0.315\textwidth]{figures/case2_confusion.pdf}\hfill
\includegraphics[width=0.315\textwidth]{figures/case3_confusion.pdf}
\end{center}
\vspace{-4pt}
\begin{quote}
\small\textbf{Figure 1.} Confusion matrices on both splits of each case, counts shown. The three
failures have different shapes: case 1 floods false positives (CONE_FP externally, CONE_FPI
internally), case 2 misses CTWO_FN positives while making only CTWO_FP false positives, and case 3
errs on both sides while staying confident.
\end{quote}
\vspace{2pt}
}]

Three ResNet-18 classifiers, each built by a different team for a different CelebA attribute,
passed internal testing and then failed in deployment. Each was reloaded unchanged and evaluated on
its supplied splits, with no retraining anywhere. Every case ran the same battery: the image
statistics of all four class folders before any prediction was made, then confusion matrices,
precision, recall and F1, ROC and AUC for discrimination, a ten-bin reliability diagram with
expected calibration error, the accuracy of the most confident predictions, and class activation
maps. Each case then ran one further experiment, designed so that a negative result would have
falsified its diagnosis. Aggregate accuracy cannot separate the three faults, since all three land
in a similar band externally (Table~1); the other axes are what distinguish them.

% Declared this early on purpose. A figure* can only be placed at the top of a page at or after
% its declaration point, so declaring it beside the text that discusses it leaves it no page top to
% claim and LaTeX defers it to a float page of its own, which would be a third page and a zero.
% Figure 1 is hand-set inside \twocolumn[...] and is not a float, so the float counter has never
% been advanced. Without this the caption below prints as "Figure 1" while every cross-reference
% in the text says Figure 2.
\setcounter{figure}{1}
\begin{figure*}[t]
\centering
\includegraphics[height=5.9cm]{figures/case1_evidence.pdf}\hfill
\includegraphics[height=5.9cm]{figures/case2_evidence.pdf}\hfill
\includegraphics[height=5.9cm]{figures/case3_evidence.pdf}
\caption{The experiment that identifies each fault. \textbf{Left, case 1:} field images beside
studio ones, and the accuracy of the studio set as blur is swept across it; blur alone reaches the
field result, the brightness shift does not. \textbf{Centre, case 2:} the four class folders, and
accuracy split by attribute combination, with the two combinations the internal set contained
outlined. \textbf{Right, case 3:} calibration measured on a held-out half of the field set, and
accuracy against the fraction published when predictions are ordered by confidence.}
\end{figure*}

\section{Case 1: clean-shaven, studio to mobile}
\textbf{Diagnosis: texture dependency, exposed by a loss of image detail between studio and mobile
capture.} The classifier reads ``clean-shaven'' from fine stubble texture on the chin and jaw. The
mobile images carry a fraction of the detail the studio images do, so that texture is gone and a
face that is not clean-shaven no longer looks different from one that is.

\textbf{Experiments and evidence.} At identical pixel dimensions and file format, the field images have a Laplacian
variance CONE_RATIO times lower than the studio ones, CONE_SHARPE against CONE_SHARPI, and are
CONE_SHIFT brighter. The same measurement on cases 2 and 3 gives ratios of CTWO_RATIO and
CTHREE_RATIO, so the gap belongs to this case rather than to the way it is measured. The errors are
one-sided in the direction that texture loss predicts: false positives rise from CONE_FPI to
CONE_FP while recall \emph{rises} to CONE_RECE.

The decisive test blurs the studio images and changes nothing else, so the subjects and the labels
are identical throughout. Accuracy falls monotonically with blur and passes through the field
result: at a Gaussian sigma of CONE_SIGMA pixels it reaches CONE_MATCHACC against the field's
CONE_ACCE, and arrives with the same shape, CONE_MATCHFP false positives against CONE_FP
(Figure~2, left). The brightness shift applied on its own reaches only CONE_BRIGHTACC, so exposure
differs between the splits but does not cause the failure. Gaussian blur is a stand-in for whatever
the mobile pipeline does rather than a claim about it; what the sweep establishes is
that removing detail is sufficient to produce the failure, so a field set of different or harder
subjects is not needed to explain it. Saliency agrees: splitting each activation map into three
bands of the face shows CONE_BAND_RIGHT_JAW of the attention on the mouth and jaw when the model is
right, falling to CONE_BAND_WRONG_JAW when it is wrong while the forehead and hair share rises from
CONE_BAND_RIGHT_BROW to CONE_BAND_WRONG_BROW. Once the texture is gone the region that should
decide the prediction holds nothing to decide it with.

\textbf{Recommendations.} Gate on sharpness at inference. Laplacian variance is one cheap statistic
per frame, and a frame below the studio range should prompt a retake rather than be scored. Match
the pre-processing to the capture path so the model is not asked for detail the camera never
recorded. The real fix is to extend the training data with mobile-quality captures and validate on
a held-out mobile sample, because a detector for a texture cue can only be trusted where the
texture survives.

\section{Case 2: eyeglasses at a retail kiosk}
\textbf{Diagnosis: a spurious correlation between eyeglasses and the subject's sex, learned because
the internal set contains only two of the four possible combinations of the two.}

\textbf{Experiments and evidence.} Displaying all four class folders (Figure~2, centre) shows the internal set is
built from women without eyeglasses and men wearing them, while the external set holds exactly the
two combinations that never appear internally. Eyeglasses and sex are therefore perfectly
confounded in everything the model was validated against, and a classifier responding only to sex
would have reached close to the reported internal accuracy of CTWO_ACCI without representing
eyeglasses at all.

Stratifying the same predictions by combination separates the two explanations that aggregate
accuracy conflates. It reaches CTWO_SEEN accuracy on the two combinations present
internally and CTWO_UNSEEN on the two absent from it, an error rate CTWO_MULT times higher. The errors fall
on both sides exactly as a sex cue predicts, missing eyeglasses on CTWO_WG_MISSED of the
CTWO_WG_TOTAL women wearing them and reporting eyeglasses on CTWO_FP men who are not. A model that
had merely degraded would err evenly across subjects; this one errs according to whether the
subject matches the pairing it was trained under, which is precisely the reported symptom of
working for some users and not others on images of equal quality.

Saliency measures the same thing directly. Splitting each activation map into three horizontal
bands of the face shows that on predictions the model gets right, CTWO_BAND_RIGHT_EYES of its
attention falls on the eyes and nose, where eyeglasses are. On the ones it gets wrong that share
drops to CTWO_BAND_WRONG_EYES and the largest share, CTWO_BAND_WRONG_JAW, moves to the mouth and
jaw. The model looks at the eyeglasses when it succeeds and elsewhere when it fails.

\textbf{Recommendations.} Recalibration cannot help, because the fault is in the representation
rather than in the confidence attached to it. Rebuild the validation set so eyeglasses and sex vary
independently, which means collecting the two missing combinations, and report performance per
combination rather than in aggregate, since an aggregate over an unbalanced set concealed this for
the whole lab phase. Until then the kiosk should confirm with the user rather than act on the
prediction, which costs one tap and avoids a recommendation resting on a demographic inference
nobody specified.

\begin{table}[b]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
Case & Split & Acc & P & R & F1 & AUC & ECE \\
\midrule
1 clean-shaven & internal & CONE_ACCI & CONE_PRECI & CONE_RECI & CONE_F1I & CONE_AUCI & CONE_ECEI \\
1 clean-shaven & external & CONE_ACCE & CONE_PRECE & CONE_RECE & CONE_F1E & CONE_AUCE & CONE_ECEE \\
2 eyeglasses & internal & CTWO_ACCI & CTWO_PRECI & CTWO_RECI & CTWO_F1I & CTWO_AUCI & CTWO_ECEI \\
2 eyeglasses & external & CTWO_ACCE & CTWO_PRECE & CTWO_RECE & CTWO_F1E & CTWO_AUCE & CTWO_ECEE \\
3 young & internal & CTHREE_ACCI & CTHREE_PRECI & CTHREE_RECI & CTHREE_F1I & CTHREE_AUCI & CTHREE_ECEI \\
3 young & external & CTHREE_ACCE & CTHREE_PRECE & CTHREE_RECE & CTHREE_F1E & CTHREE_AUCE & CTHREE_ECEE \\
\bottomrule
\end{tabular}
\caption{All three cases fall into a similar accuracy band externally. AUC, which measures
discrimination, and ECE, which measures whether the confidence can be believed, are what separate
them: case 1 keeps most of its ranking, case 3 loses nearly all of it, and only case 3 is badly
calibrated internally as well.}
\end{table}


\section{Case 3: ``appears young'' with an auto-publish bypass}
\textbf{Diagnosis: overconfidence under distribution shift.} The model does not transfer to the
deployed population and its confidence does not reveal it; validating on an unrepresentative
internal set is the cause. The internal negatives are clearly elderly faces while the field negatives are
middle-aged, so the lab phase measured a much easier problem than deployment poses.

\textbf{Experiments and evidence.} Internally the model reaches AUC CTHREE_AUCI and is reasonably calibrated at ECE
CTHREE_ECEI, so the bypass looked safe. Externally AUC falls to CTHREE_AUCE, close to the 0.5 of a
coin toss: the scores no longer rank the classes. The softmax nevertheless keeps emitting extreme
probabilities, so ECE rises to CTHREE_ECEE. The consequence for the bypass is direct. Of the field
predictions issued at 0.90 confidence or above, CTHREE_G90PUB percent of the split, only
CTHREE_G90ACC are correct. Raising the gate to 0.99 leaves CTHREE_G99PUB percent published at
CTHREE_G99ACC, so tightening it does not help: the confidence that selects them is no longer
informative about whether they are right. Both thresholds are reported because the assessment page
and the supplied notebook state different ones, and the conclusion is the same at either.

Fitting a temperature on half the field set and measuring on the other half confirms the mechanism.
Expected calibration error falls from CTHREE_RAWECE to CTHREE_CALECE while AUC stays at
CTHREE_CALAUC, unchanged, because dividing a logit by a constant is monotone and cannot reorder
predictions. The fitted temperature is CTHREE_TEMP, and after scaling the most confident field
image reaches only CTHREE_CALMAX, so a gate at either threshold would publish CTHREE_CALPUB of the
CTHREE_HELD held-out images. The risk-coverage curve states the ceiling directly: the best accuracy
reachable at any coverage above two percent of the field split is CTHREE_REACH (Figure~2, right).

\textbf{Recommendations.} Switch the bypass off rather than re-tune its threshold. No threshold
publishes at 0.90 accuracy on this population, so any gate setting trades volume against errors
instead of avoiding them. Recalibrate, but treat it as instrumentation rather than a repair: a
calibrated score is what lets monitoring detect this condition, and the fact that almost nothing
then clears the gate is the honest reading, not a defect of the calibration. Rebuild the internal
set so its negatives span the ages the system actually meets, and monitor expected calibration
error and high-confidence accuracy continuously, because both looked healthy in the lab and neither
stayed so.

\section{Summary}
The three models fail at similar accuracy for different reasons, which is why the remedies differ.
Case 1 is a property of the inputs, case 2 of the training distribution, and case 3 of the
validation design. In all three the aggregate metric that passed the lab phase is the same metric
that hid the fault, which is the general lesson: an audit has to compare distributions and stratify
results rather than summarise them.

\end{document}
"""

REPLACEMENTS = {
    "CONE_ACCI": one["internal_acc"], "CONE_PRECI": one["internal_prec"],
    "CONE_RECI": one["internal_rec"], "CONE_F1I": one["internal_f1"], "CONE_AUCI": one["internal_auc"],
    "CONE_ECEI": one["internal_ece"],
    "CONE_ACCE": one["external_acc"], "CONE_PRECE": one["external_prec"],
    "CONE_RECE": one["external_rec"], "CONE_F1E": one["external_f1"], "CONE_AUCE": one["external_auc"],
    "CONE_ECEE": one["external_ece"],
    "CTWO_ACCI": two["internal_acc"], "CTWO_PRECI": two["internal_prec"],
    "CTWO_RECI": two["internal_rec"], "CTWO_F1I": two["internal_f1"], "CTWO_AUCI": two["internal_auc"],
    "CTWO_ECEI": two["internal_ece"],
    "CTWO_ACCE": two["external_acc"], "CTWO_PRECE": two["external_prec"],
    "CTWO_RECE": two["external_rec"], "CTWO_F1E": two["external_f1"], "CTWO_AUCE": two["external_auc"],
    "CTWO_ECEE": two["external_ece"],
    "CTHREE_ACCI": three["internal_acc"], "CTHREE_PRECI": three["internal_prec"],
    "CTHREE_RECI": three["internal_rec"], "CTHREE_F1I": three["internal_f1"], "CTHREE_AUCI": three["internal_auc"],
    "CTHREE_ECEI": three["internal_ece"],
    "CTHREE_ACCE": three["external_acc"], "CTHREE_PRECE": three["external_prec"],
    "CTHREE_RECE": three["external_rec"], "CTHREE_F1E": three["external_f1"], "CTHREE_AUCE": three["external_auc"],
    "CTHREE_ECEE": three["external_ece"],
    # ------------------------------------------------------------------ case 1
    "CONE_RATIO": one["sharp_ratio"], "CONE_SHARPI": one["sharp_internal"],
    "CONE_SHARPE": one["sharp_external"], "CONE_SHIFT": one["bright_shift"],
    "CTWO_RATIO": two["sharp_ratio"], "CTHREE_RATIO": three["sharp_ratio"],
    "CONE_FPI": one["internal_fp"], "CONE_FP": one["external_fp"],
    "CONE_SIGMA": one["match_sigma"], "CONE_MATCHACC": one["match_acc"],
    "CONE_MATCHFP": one["match_fp"], "CONE_BRIGHTACC": one["bright_only_acc"],
    # ------------------------------------------------------------------ case 2
    "CTWO_FN": two["external_fn"], "CTWO_FP": two["external_fp"],
    "CTWO_SEEN": two["seen_acc"], "CTWO_UNSEEN": two["unseen_acc"],
    "CTWO_MULT": two["multiplier"],
    "CTWO_WG_MISSED": str(int(two["wg_total"]) - int(two["wg_right"])),
    "CTWO_WG_TOTAL": two["wg_total"],
    "CONE_BAND_RIGHT_JAW": one["band"]["correct"][2],
    "CONE_BAND_WRONG_JAW": one["band"]["misclassified"][2],
    "CONE_BAND_RIGHT_BROW": one["band"]["correct"][0],
    "CONE_BAND_WRONG_BROW": one["band"]["misclassified"][0],
    "CTWO_BAND_RIGHT_EYES": two["band"]["correct"][1],
    "CTWO_BAND_WRONG_EYES": two["band"]["misclassified"][1],
    "CTWO_BAND_WRONG_JAW": two["band"]["misclassified"][2],
    # ------------------------------------------------------------------ case 3
    "CTHREE_G90PUB": three["external_gate90_published"],
    "CTHREE_G90ACC": three["external_gate90_accuracy"],
    "CTHREE_G99PUB": three["external_gate99_published"],
    "CTHREE_G99ACC": three["external_gate99_accuracy"],
    "CTHREE_RAWECE": three["raw_ece"], "CTHREE_CALECE": three["cal_ece"],
    "CTHREE_CALAUC": three["cal_auc"], "CTHREE_CALMAX": three["cal_maxconf"],
    "CTHREE_TEMP": three["temperature"], "CTHREE_CALPUB": three["cal_published"],
    "CTHREE_HELD": three["held_out"], "CTHREE_REACH": three["reachable"],
}

body = TEMPLATE
# Longest first, so CTHREE_ACCI is never matched as CTHREE_ACC followed by a stray I.
for token in sorted(REPLACEMENTS, key=len, reverse=True):
    body = body.replace(token, str(REPLACEMENTS[token]))

leftover = sorted(set(re.findall(r"\b(?:CONE|CTWO|CTHREE)_[A-Z0-9_]+", body)))
assert not leftover, f"unsubstituted tokens: {leftover}"

# The house style bans em and en dashes anywhere in a deliverable.
for dash in ("\u2014", "\u2013"):
    assert dash not in body, f"dash {dash!r} in the report body"

os.makedirs(FIGURES, exist_ok=True)
for case in (1, 2, 3):
    for kind in ("confusion", "evidence"):
        source = f"{NBDIR}/case{case}_{kind}.pdf"
        assert os.path.exists(source), f"missing {source}; run the notebook first"
        shutil.copyfile(source, f"{FIGURES}/case{case}_{kind}.pdf")

io.open(f"{REPORT}/project4_report.tex", "w", encoding="utf-8").write(body)
print(f"wrote {REPORT}/project4_report.tex "
      f"({len(body.splitlines())} lines, {len(REPLACEMENTS)} substituted values)")
for case in (1, 2, 3):
    v = values[case]
    print(f"  case {case}: internal acc {v['internal_acc']} AUC {v['internal_auc']} | "
          f"external acc {v['external_acc']} AUC {v['external_auc']} ECE {v['external_ece']}")
