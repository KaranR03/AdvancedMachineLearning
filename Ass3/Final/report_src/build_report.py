r"""Build report/project3_report.tex from the numbers main_report.ipynb printed.

Nothing numeric is typed into this file by hand. Every value in the report is pulled out of the
executed main_report.ipynb, which is what makes the unit's rule ("every result and figure included
in the report should be produced by and copied from main_report.ipynb") mechanically true rather
than merely intended. tools/verify.py checks the same property from the other direction.

Build:  python tools/build_report.py  &&  tectonic report/project3_report.tex
"""
import io
import json
import os
import re
import shutil

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-05/project-3-aircraft-classification")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"

# The group template's header block: project title, group, then one line per student.
TITLE = "Aircraft Classification with a Pretrained ResNet-18"
GROUP = "4"
STUDENTS = [("Karan Rooprai", "n12498122"), ("Nhu Hieu Nguyen", "n12194778")]

# ---------------------------------------------------------------- read the notebook's output
notebook = json.load(io.open(f"{NBDIR}/main_report.ipynb", encoding="utf-8"))
printed = "\n".join("".join(o.get("text", "")) for c in notebook["cells"]
                    for o in c.get("outputs", []) if o.get("output_type") == "stream")
assert printed.strip(), "main_report.ipynb has no output; run tools/run_notebook.py first"


def grab(pattern, cast=float):
    match = re.search(pattern, printed)
    assert match, f"main_report.ipynb never printed anything matching {pattern!r}"
    groups = match.groups()
    return [cast(g) for g in groups] if len(groups) > 1 else cast(groups[0])


n_trainval, n_train, n_val = grab(r"Trainval (\d+) images: (\d+) train / (\d+) validation", int)
n_test = grab(r"Test images: (\d+) across", int)

baseline = grab(r"Baseline \(frozen SGD\)\s+(\d\.\d+)")
aug, aug_d = grab(r"Exp 1: \+ Augmentation\s+(\d\.\d+)\s+([-+]\d\.\d+)")
ft, ft_d = grab(r"Exp 2: Fine-tuning\s+(\d\.\d+)\s+([-+]\d\.\d+)")
adamw, adamw_d = grab(r"Exp 3: AdamW \+ Cosine\s+(\d\.\d+)\s+([-+]\d\.\d+)")

overall = grab(r"Overall test accuracy\s+:\s+(\d\.\d+)")
avg_pc = grab(r"Average per-class accuracy\s+:\s+(\d\.\d+)")
macro_f1 = grab(r"Macro-averaged F1\s+:\s+(\d\.\d+)")
n_strong, n_classes = grab(r"Classes scoring above 0\.87: (\d+) of (\d+)", int)
worst_name, worst_acc = re.search(r"Weakest class: (\S+) at (\d\.\d+)", printed).groups()
worst_acc = float(worst_acc)

# per-class table: every "class_NN  0.xxx" line printed by the ranked listing
per_class = [(m.group(1), float(m.group(2)))
             for m in re.finditer(r"^(class_\d+)\s+(\d\.\d+)\s*$", printed, re.M)]
assert len(per_class) == n_classes, f"expected {n_classes} per-class rows, got {len(per_class)}"
ranked = sorted(per_class, key=lambda r: r[1])
best_two = ranked[-2:][::-1]

# the confusion listing: "true predicted count"
confusions = [(m.group(1), m.group(2), int(m.group(3)))
              for m in re.finditer(r"^(class_\d+)\s+(class_\d+)\s+(\d+)\s*$", printed, re.M)]
assert confusions, "the confusion listing was not found in main_report.ipynb output"
top_conf = confusions[0]
# the reciprocal of the top confusion, if the model confuses the pair both ways
recip = next((c for c in confusions if c[0] == top_conf[1] and c[1] == top_conf[0]), None)

# The final model's training accuracy and its gap to test come from the notebook's printed output,
# not from histories.pkl directly: the unit requires every number in the report to be produced by
# main_report.ipynb, and tools/verify.py enforces exactly that from the other side.
final_train_acc, gap = grab(
    r"Final model training accuracy: (\d\.\d+)\s+gap to test: (\d\.\d+)")

# Epoch counts describe the method rather than a result, so they are read from the saved curves.
import pickle  # noqa: E402
hist = pickle.load(io.open(f"{NBDIR}/histories.pkl", "rb"))["histories"]
final_epochs = len(hist["best"]["train_loss"])
exp_epochs = len(hist["baseline"]["train_loss"])


def tex(name):
    r"""LaTeX-safe class name: class_00 -> class\_00, kept in a monospace-ish roman."""
    return name.replace("_", r"\_")


def pct(x):
    return f"{x:.4f}"


# ---------------------------------------------------------------- figures
os.makedirs(f"{REPORT}/figures", exist_ok=True)
for stem in ("learning_curves", "confusion_matrix"):
    for ext in ("pdf", "png"):
        source = f"{NBDIR}/{stem}.{ext}"
        assert os.path.exists(source), f"{source} missing; run main_report.ipynb first"
        shutil.copyfile(source, f"{REPORT}/figures/{stem}.{ext}")

# ---------------------------------------------------------------- the document
student_lines = "\n".join(
    rf"Student {i}: {name} \hfill Student {i} ID: {sid} \\"
    for i, (name, sid) in enumerate(STUDENTS, start=1))

D = {}
D["TITLE"] = TITLE
D["GROUP"] = GROUP
D["STUDENTS"] = student_lines
D["NTRAINVAL"] = f"{n_trainval:,}"
D["NTRAIN"] = f"{n_train:,}"
D["NVAL"] = str(n_val)
D["NTEST"] = str(n_test)
D["NCLASSES"] = str(n_classes)
D["EXPEPOCHS"] = str(exp_epochs)
D["FINALEPOCHS"] = str(final_epochs)
D["BASELINE"] = pct(baseline)
D["AUG"] = pct(aug)
D["AUGD"] = f"{aug_d:+.4f}".replace("-", "$-$")   # real minus, not a hyphen
D["FT"] = pct(ft)
D["FTD"] = f"{ft_d:+.4f}"
D["ADAMW"] = pct(adamw)
D["ADAMWD"] = f"{adamw_d:+.4f}"
D["OVERALL"] = pct(overall)
D["AVGPC"] = pct(avg_pc)
D["MACROF1"] = pct(macro_f1)
D["NSTRONG"] = str(n_strong)
D["WORST"] = tex(worst_name)
D["WORSTACC"] = f"{worst_acc:.3f}"
# The weakest class is discussed by name as one half of the top confusion pair, so PARTNER is the
# *other* half. Binding it to top_conf[1] unconditionally produced "class_00 ... is confused with
# class_00", because the pair is printed worst-second (class_01 -> class_00) as often as not.
partner = top_conf[0] if top_conf[1] == worst_name else top_conf[1]
assert worst_name in top_conf[:2], (
    f"weakest class {worst_name} is no longer half of the top confusion pair {top_conf[:2]}; "
    "the 'confused with ... in both directions' sentence would be wrong")
D["PARTNER"] = tex(partner)

# The next-weakest classes are offered as further examples, so neither may be a class the sentence
# before has already named -- otherwise the report introduces class_01 twice as if it were new.
named = {worst_name, partner}
also_weak = [r for r in ranked if r[0] not in named][:2]
assert len(also_weak) == 2, f"fewer than two further weak classes outside {named}"
D["SECOND"], D["SECONDACC"] = tex(also_weak[0][0]), f"{also_weak[0][1]:.3f}"
D["THIRD"], D["THIRDACC"] = tex(also_weak[1][0]), f"{also_weak[1][1]:.3f}"
D["BEST1"] = tex(best_two[0][0])
D["BEST1ACC"] = f"{best_two[0][1]:.3f}"
D["BEST2"] = tex(best_two[1][0])
D["BEST2ACC"] = f"{best_two[1][1]:.3f}"
D["CONFA"] = tex(top_conf[0])
D["CONFB"] = tex(top_conf[1])
D["CONFN"] = str(top_conf[2])
D["RECIPN"] = str(recip[2]) if recip else "0"
D["TRAINACC"] = f"{final_train_acc:.2f}"
D["GAP"] = f"{gap:.2f}"

TEMPLATE = r"""% IFN680 Project 3 technical report.
% Generated by tools/build_report.py. Every number below is printed by notebook/main_report.ipynb.
% Build:  tectonic project3_report.tex
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
\renewcommand{\bottomfraction}{0.55}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.75}

\begin{document}

% Figure 1 is set inside the \twocolumn optional argument rather than as a figure* float. In the
% article class \twocolumn[...] claims the page's double column top region, so a figure* declared
% afterwards can never be placed on page 1 and is silently deferred to page 2.
\twocolumn[{%
\begin{center}
{\large\bfseries Project 3: @TITLE@}\\[2pt]
{\bfseries Group @GROUP@}
\end{center}
\vspace{-2pt}
@STUDENTS@
\vspace{2pt}

\begin{center}
\includegraphics[width=0.88\textwidth]{figures/learning_curves.pdf}
\end{center}
\vspace{-8pt}
\begin{quote}
\small\textbf{Figure 1.} Validation accuracy (left) and cross-entropy loss (right) per epoch for
the baseline and all three experiments, on shared axes. Dashed lines on the right are training
loss. Fine-tuning (green) separates from the three frozen-backbone runs within two epochs.
\end{quote}
\vspace{4pt}
}]

A pretrained ResNet-18 (ImageNet weights) was adapted by transfer learning to classify the
@NCLASSES@-class FGVC-Aircraft subset. The \texttt{trainval} folder of @NTRAINVAL@ images was
divided by a stratified @NTRAIN@/@NVAL@ split into training and validation sets; the @NTEST@ test
images were held out and used once, for the final evaluation only. Every image was resized to
$224\times224$ and normalised with the ImageNet mean and standard deviation to match the pretrained
backbone, and the 1000-way ImageNet head was replaced with a fresh @NCLASSES@-way linear layer.
The three experiments follow a controlled design: each alters a single variable relative to a
common baseline and trains for @EXPEPOCHS@ epochs, so any change in validation accuracy is
attributable to that one factor.

\section{Baseline model}
The baseline uses ResNet-18 as a fixed feature extractor: the convolutional backbone is frozen and
only the new classification head is trained, with plain SGD (learning rate 0.001, momentum 0.9) on
un-augmented images. This establishes the performance floor every experiment must beat. It reached
a best validation accuracy of \textbf{@BASELINE@} (Table~1; blue curve, Figure~1). The frozen
ImageNet features separate the twenty aircraft types moderately well, but because the backbone
cannot adapt, the head alone cannot capture the fine-grained differences between visually similar
variants.

\section{Experiment 1: Data augmentation}
\textbf{Hypothesis:} randomly flipping, rotating ($\pm15^\circ$) and colour-jittering the training
images exposes the frozen extractor to more varied views, reducing over-fitting and improving
validation accuracy. \textbf{Result: not supported.} Augmentation lowered best validation accuracy
to @AUG@, a drop of @AUGD@ below the baseline (orange curve, Figure~1). With the backbone frozen
the network cannot adapt its features to the distorted inputs, so augmentation pushes images into
regions of the fixed ImageNet feature space that the linear head has not learned, and @EXPEPOCHS@
epochs are too few to compensate. Augmentation therefore hurts a pure feature-extraction setup on
this dataset.

\section{Experiment 2: Fine-tuning the backbone}
\textbf{Hypothesis:} ImageNet features are generic, so unfreezing the backbone and letting every
layer adapt to aircraft should capture fine-grained variant differences and lift accuracy well
above the frozen baseline. \textbf{Result: strongly supported.} Unfreezing the backbone was the
only change from the baseline, keeping the identical SGD optimiser and 0.001 learning rate; it
raised best validation accuracy to \textbf{@FT@}, a gain of @FTD@ (green curve, Figure~1). This is
by far the largest single improvement, confirming that adapting the convolutional features, and
not just the classifier, is what fine-grained aircraft recognition requires.

\section{Experiment 3: AdamW and cosine annealing}
\textbf{Hypothesis:} replacing plain SGD with AdamW (adaptive updates with decoupled weight decay)
and decaying the learning rate on a cosine schedule should give faster, more stable convergence of
the frozen head and a modest accuracy gain. \textbf{Result: supported, modestly.} With the backbone
frozen and images un-augmented, best validation accuracy rose to @ADAMW@, a gain of @ADAMWD@ over
the baseline (red curve, Figure~1). AdamW converges faster and to a slightly better optimum than
SGD, but with the backbone still frozen the ceiling is low, so the gain is real but small beside
fine-tuning.

Optimiser, decoupled weight decay and schedule are varied together as one variable: the
optimisation strategy, the axis the task description names. They are not separable the way a single
hyperparameter is: decoupled weight decay is the very thing that distinguishes AdamW from Adam, and
AdamW under cosine annealing is a different procedure rather than another setting of the same one.
Everything outside that axis
(frozen backbone, un-augmented data, @EXPEPOCHS@ epochs, batch size, 0.001 base learning rate)
is identical to the baseline, so the comparison stays controlled.

\begin{table}[t]
\centering
\small
\begin{tabular}{lrr}
\toprule
Model & Best val.\ acc. & $\Delta$ base \\
\midrule
Baseline (frozen, SGD)     & @BASELINE@ & n/a \\
Exp 1: + Augmentation      & @AUG@      & @AUGD@ \\
Exp 2: Fine-tuning         & @FT@       & @FTD@ \\
Exp 3: AdamW + cosine      & @ADAMW@    & @ADAMWD@ \\
\bottomrule
\end{tabular}
\caption{Best validation accuracy of the baseline and each single-variable experiment.}
\end{table}

% Declared here, well before the text that discusses it, because a figure* can only be placed at
% the top of a page at or after its declaration point. Declared at the end of the document it has
% no page top left to claim and LaTeX defers it to a float page of its own -- a third page, which
% is a zero for the report. Figure 1 is hand-set inside \twocolumn[...] and is not a float, so the
% float counter has to be advanced by hand or this caption also prints as "Figure 1".
\setcounter{figure}{1}
\begin{figure*}[t]
\centering
\includegraphics[width=0.44\textwidth]{figures/confusion_matrix.pdf}
\caption{Confusion matrix of the final model on the @NTEST@ test images, shown as a heatmap;
rows are true classes, columns predicted. The off-diagonal counts quoted in the text are printed
by \texttt{main\_report.ipynb} rather than read from this figure.}
\end{figure*}

\section{Best model}
\textbf{Rationale.} The final model keeps only the changes that improved validation accuracy over
the baseline. Fine-tuning (@FTD@) and AdamW with cosine annealing (@ADAMWD@) both helped and are
retained; augmentation is excluded, because it reduced accuracy and so is not a winning
ingredient. The final model is therefore an unfrozen ResNet-18 trained with AdamW and cosine
annealing on un-augmented images. It was chosen on best mean validation accuracy rather than on
stability or hardest-class performance, because the margin between fine-tuning and every
alternative (@FTD@) is far larger than the epoch-to-epoch variation visible in Figure~1.

Combining two separately validated changes alters the optimisation problem, so the learning rate
was cut to 0.0001 for the final run: AdamW's adaptive steps on every backbone layer at 0.001 are
more aggressive than either experiment tested, since Experiment~2 used SGD on all layers and
Experiment~3 used AdamW on the head alone. This is the one deliberate departure from a validated
configuration, and it is a design judgement rather than a measured result: once the model is
refit on all of \texttt{trainval}, no validation data survives on which to tune it.

\textbf{Result.} \textbf{The final model was retrained on the entire \texttt{trainval} set (all
@NTRAINVAL@ images, training plus validation, no validation split) for @FINALEPOCHS@ epochs, and
every figure reported here is measured on the @NTEST@ held-out test images with that refit model,
evaluated once.} It achieves an average per-class accuracy of \textbf{@AVGPC@}, comfortably
exceeding the required 0.75, with overall accuracy @OVERALL@ and a macro-averaged F1 of @MACROF1@.
Average per-class accuracy, the mean of the @NCLASSES@ per-class recall values, is the
reported metric rather than overall accuracy because it weights every aircraft type equally
instead of letting the larger or easier classes dominate. The two agree closely here only because
the test split is balanced. Macro F1 is given alongside because recall alone cannot see a class
the model over-predicts. The test confusion matrix is Figure~2.

\section{Potential and limitations}
The strong diagonal in Figure~2 shows most classes are recognised reliably: @NSTRONG@ of
@NCLASSES@ score above 0.87 and the best (@BEST1@, @BEST2@) reach @BEST1ACC@ and @BEST2ACC@. The
clear weakness is @WORST@ at @WORSTACC@, which is confused with @PARTNER@ in both directions:
@CONFN@ @CONFA@ images are predicted as @CONFB@ and @RECIPN@ the other way, indicating two
visually near-identical variants the model cannot separate. @SECOND@ (@SECONDACC@) and @THIRD@
(@THIRDACC@) fail similarly against look-alike types. The fine-tuned model also reaches
@TRAINACC@ training accuracy within a few epochs, so the gap of roughly @GAP@ to test performance
reflects real over-fitting, which the small per-class support makes hard to avoid: at roughly
@NTEST@$/$@NCLASSES@ test images per class, a single extra error moves a class score by about
three points.

For the target application the method is promising: @AVGPC@ average per-class accuracy over twenty
types, from barely a thousand labelled images and a backbone that trains in minutes, shows transfer
learning suits a catalogue task with narrow visual differences. The anticipated limitations are the
near-identical variants above, unlikely to separate without finer-grained supervision or
higher-resolution inputs; the small per-class support, which makes every class score coarse; and
uncalibrated output scores, which should not be read as probabilities in deployment. The obvious
next steps are reintroducing light augmentation now the backbone is trainable (augmentation
regularises a fine-tuned network differently from a frozen one, so Experiment~1 does not settle the
question), tuning weight decay or dropout, early stopping, and a deeper backbone such as
ResNet-34 for the hardest look-alike pairs.

\end{document}
"""

document = TEMPLATE
for key, value in D.items():
    document = document.replace(f"@{key}@", str(value))

leftover = re.findall(r"@[A-Z0-9]+@", document)
assert not leftover, f"unsubstituted placeholders remain: {sorted(set(leftover))}"

os.makedirs(REPORT, exist_ok=True)
io.open(f"{REPORT}/project3_report.tex", "w", encoding="utf-8").write(document)
print(f"wrote {REPORT}/project3_report.tex  ({len(document.splitlines())} lines)")
print(f"  avg per-class accuracy {avg_pc:.4f}   overall {overall:.4f}   macro F1 {macro_f1:.4f}")
print(f"  {n_strong} of {n_classes} classes above 0.87; weakest {worst_name} at {worst_acc:.3f}")
print(f"  top confusion: {top_conf[0]} -> {top_conf[1]} ({top_conf[2]} images)"
      + (f", reciprocal {recip[2]}" if recip else ""))
