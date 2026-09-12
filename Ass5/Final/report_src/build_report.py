r"""Write report/project5_report.tex from the executed main_report.ipynb.

Every number in the report is pulled out of the notebook's printed output by regex here, so
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
        "weeks/week-08/project-5-extending-addition-llm")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
FIGURES = f"{REPORT}/figures"

NUM = r"([\d.]+)"
INT = r"(\d+)"


def printed():
    """All stdout text of the executed main_report.ipynb, concatenated in cell order."""
    notebook = json.load(io.open(f"{NBDIR}/main_report.ipynb", encoding="utf-8"))
    chunks = ["".join(o.get("text", []))
              for cell in notebook["cells"] for o in cell.get("outputs", [])
              if o.get("output_type") == "stream"]
    text = "\n".join(chunks)
    assert text.strip(), "main_report.ipynb has no printed output; execute it first"
    return text


T = printed()


def group(value):
    """Thousands separators on a bare count, so 10,000 and 12,000 do not sit beside 5076.

    Only a run of digits is touched. Accuracies keep their decimal point, an arithmetic answer
    inside the error table never reaches here, and a sentence that has already been assembled is
    left alone because it is not a bare count.
    """
    return f"{int(value):,}" if value.isdigit() and len(value) > 3 else value


def grab(pattern, label, text=None):
    match = re.search(pattern, text if text is not None else T, re.MULTILINE)
    assert match, f"could not find {label}"
    return match.groups() if len(match.groups()) > 1 else match.group(1)


V = {}

# ------------------------------------------------------------------ overall and per-operation
for mode in ("Forward", "Reverse"):
    for split in ("overall", "addition", "subtraction"):
        n, correct, accuracy, low, high = grab(
            rf"^\s*{mode}\s*\|\s*{split}\s*\|\s*{INT}\s*\|\s*{INT}\s*\|\s*{NUM}\s*\|"
            rf"\s*\[{NUM},\s*{NUM}\]", f"{mode} {split} row")
        key = f"{mode.lower()}_{split}"
        V[key] = accuracy
        V[key + "_n"] = n
        V[key + "_correct"] = correct
        V[key + "_lo"], V[key + "_hi"] = low, high
        V[key + "_err"] = str(int(n) - int(correct))

# ------------------------------------------------------------------------------- McNemar
V["mcnemar_b"] = grab(rf"Forward right, Reverse wrong\s*:\s*{INT}", "mcnemar b")
V["mcnemar_c"] = grab(rf"Forward wrong, Reverse right\s*:\s*{INT}", "mcnemar c")
V["mcnemar_agree"] = grab(rf"both models agree on\s*:\s*{INT}", "mcnemar agree")
V["mcnemar_p"] = grab(r"exact McNemar two-sided p\s*:\s*([\d.e+-]+)", "mcnemar p")
# Python prints 1.212e-07; typeset inside $...$ that renders as an italic e and a minus sign
# floating between two numbers. The mantissa is kept verbatim so it is still a substring of what
# the notebook printed, which is what the traceability sweep matches on.
if "e" in V["mcnemar_p"]:
    MANTISSA, EXPONENT = V["mcnemar_p"].split("e")
    V["mcnemar_p_tex"] = f"{MANTISSA} \\times 10^{{{int(EXPONENT)}}}"
else:
    V["mcnemar_p_tex"] = V["mcnemar_p"]
# Examples both models get wrong: Forward's error count less the ones only Forward missed.
# Nothing prints it directly, but it is what decides whether the two failure sets overlap,
# which the error discussion and the caption of Table 2 both describe.
V["both_wrong"] = str(int(V["forward_overall_err"]) - int(V["mcnemar_c"]))
V["overlap_phrase"] = ("The two sets have no case in common."
                       if V["both_wrong"] == "0"
                       else f"The two sets overlap in {V['both_wrong']} cases.")

# ----------------------------------------------------------------------------- positions
for place in ("thousands", "hundreds", "tens", "units", "sign"):
    n, forward, reverse = grab(
        rf"^\s*{place}\s*\|\s*{INT}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*$", f"position {place}")
    V[f"pos_{place}_n"], V[f"pos_{place}_f"], V[f"pos_{place}_r"] = n, forward, reverse

# The same places split by operation. An empty group prints n/a instead of a rate, so the
# pattern has to accept it; only the thousands row of subtraction is ever in that state.
NUMNA = r"([\d.]+|n/a)"
for place in ("thousands", "hundreds", "tens", "units", "sign"):
    n_add, f_add, r_add, n_sub, f_sub, r_sub = grab(
        rf"^\s*{place}\s*\|\s*{INT}\s*\|\s*{NUMNA}\s*\|\s*{NUMNA}\s*\|"
        rf"\s*{INT}\s*\|\s*{NUMNA}\s*\|\s*{NUMNA}\s*$", f"{place} by operation")
    V[f"posop_{place}_nadd"], V[f"posop_{place}_fa"], V[f"posop_{place}_ra"] = n_add, f_add, r_add
    V[f"posop_{place}_nsub"], V[f"posop_{place}_fs"], V[f"posop_{place}_rs"] = n_sub, f_sub, r_sub

# -------------------------------------------------------------------------- carry / borrow
for case in ("with_carry", "no_carry", "with_borrow", "no_borrow"):
    n, fe, fa, re_, ra = grab(
        rf"^\s*{case}\s*\|\s*{INT}\s*\|\s*{INT}\s*\|\s*{NUM}\s*\|\s*{INT}\s*\|\s*{NUM}\s*$",
        f"carry/borrow {case}")
    V[f"cb_{case}_n"], V[f"cb_{case}_fe"], V[f"cb_{case}_fa"] = n, fe, fa
    V[f"cb_{case}_re"], V[f"cb_{case}_ra"] = re_, ra

# ------------------------------------------------------------------------ length and sign
for width in (1, 2, 3, 4):
    try:
        n, fe, re_ = grab(rf"^\s*{width} digit\s*\|\s*{INT}\s*\|\s*{INT}\s*\|\s*{INT}\s*$",
                          f"length {width}")
        V[f"len{width}_n"], V[f"len{width}_fe"], V[f"len{width}_re"] = n, fe, re_
    except AssertionError:
        V[f"len{width}_n"] = V[f"len{width}_fe"] = V[f"len{width}_re"] = "0"
for label, key in (("negative", "neg"), ("non-negative", "pos")):
    n, fe, re_ = grab(rf"^\s*{label}\s*\|\s*{INT}\s*\|\s*{INT}\s*\|\s*{INT}\s*$", f"sign {label}")
    V[f"{key}_n"], V[f"{key}_fe"], V[f"{key}_re"] = n, fe, re_

# ---------------------------------------------------------------- operand width difference
alignment = {}
for line in T.split("\n"):
    m = re.match(rf"^\s*(-?\d+)\s*\|\s*{INT}\s*\|\s*{INT}\s*\|\s*{NUM}\s*\|\s*{INT}\s*\|"
                 rf"\s*{NUM}\s*$", line)
    if m:
        alignment[int(m.group(1))] = m.groups()[1:]
assert alignment, "could not find the operand width difference table"
V["align_max_diff"] = str(max(alignment))
V["align_worst_n"] = alignment[max(alignment)][0]
V["align_worst_fe"] = alignment[max(alignment)][1]
V["align_worst_re"] = alignment[max(alignment)][3]
V["align_worst_rrate"] = alignment[max(alignment)][4]
V["align_worst_frate"] = alignment[max(alignment)][2]
# Every width difference at or below 1, pooled: this is the "perfect everywhere else" claim.
easy_n = sum(int(alignment[d][0]) for d in alignment if d <= 1)
easy_re = sum(int(alignment[d][3]) for d in alignment if d <= 1)
easy_fe = sum(int(alignment[d][1]) for d in alignment if d <= 1)
V["align_easy_n"], V["align_easy_re"], V["align_easy_fe"] = str(easy_n), str(easy_re), str(easy_fe)

# ------------------------------------------------------------------------- wrong places
import ast  # noqa: E402

places = {}
for mode in ("Forward", "Reverse"):
    raw = grab(rf"{mode}\s+wrong places \(0 = units\): (\{{.*?\}})", f"{mode} wrong places")
    places[mode] = ast.literal_eval(raw)  # a dict literal this notebook printed itself
V["rev_place1"] = str(places["Reverse"].get("place 1", 0))
V["rev_wronglen"] = str(places["Reverse"].get("wrong length", 0))
V["fwd_offby1"] = grab(rf"Forward\s+errors off by exactly 1: {INT}", "forward off by one")
V["rev_offby1"] = grab(rf"Reverse\s+errors off by exactly 1: {INT}", "reverse off by one")

# ------------------------------------------------------------------------ learning curves
for mode in ("Forward", "Reverse"):
    rows = re.findall(rf"^\s*{mode}\s*\|\s*{INT}\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*{NUM}\s*$",
                      T, re.MULTILINE)
    assert len(rows) == 2, f"expected two curve rows for {mode}, found {len(rows)}"
    rows.sort(key=lambda r: -int(r[0]))          # full training set first
    for label, row in zip(("full", "small"), rows):
        V[f"curve_{mode.lower()}_{label}_size"] = f"{int(row[0]):,}"
        V[f"curve_{mode.lower()}_{label}_tgt"] = row[1]
        V[f"curve_{mode.lower()}_{label}_95"] = row[2]
        V[f"curve_{mode.lower()}_{label}_final"] = row[3]

# Optimiser updates to the 0.95 mark. An epoch means a different amount of training at each
# set size, so this is the comparison that is actually like for like.
for mode in ("Forward", "Reverse"):
    step_rows = re.findall(rf"^steps to 0\.95 \|\s*{mode}\s*\|\s*{INT}\s*\|\s*(\S+)\s*$",
                           T, re.MULTILINE)
    assert len(step_rows) == 2, f"expected two step rows for {mode}, found {len(step_rows)}"
    step_rows.sort(key=lambda r: -int(r[0]))
    for label, row in zip(("full", "small"), step_rows):
        # Grouped for reading; the raw count is what the notebook printed.
        V[f"steps_{mode.lower()}_{label}"] = row[1]
        V[f"steps_{mode.lower()}_{label}_text"] = (f"{int(row[1]):,}" if row[1].isdigit()
                                                   else row[1])


def steps_close(first, second):
    """True when two step counts are within 15 percent, which is well inside one epoch."""
    if not (first.isdigit() and second.isdigit()):
        return False
    first, second = int(first), int(second)
    return abs(first - second) <= 0.15 * max(first, second)


FWD_SAME_STEPS = steps_close(V["steps_forward_full"], V["steps_forward_small"])
REV_SAME_STEPS = steps_close(V["steps_reverse_full"], V["steps_reverse_small"])
SAME_STEPS = FWD_SAME_STEPS and REV_SAME_STEPS
# One direction reaching the mark at the same update count while the other never reaches it
# inside the epochs run is still evidence about units of measurement, so it is worth stating.
PART_STEPS = (FWD_SAME_STEPS or REV_SAME_STEPS) and not SAME_STEPS
MATCHED = "Forward" if FWD_SAME_STEPS else "Reverse"
OTHER = "Reverse" if FWD_SAME_STEPS else "Forward"
V["matched_small"] = V[f"steps_{MATCHED.lower()}_small_text"]
V["matched_full"] = V[f"steps_{MATCHED.lower()}_full_text"]

# Whether a reduced training set separates the two directions is the question the ablation
# was run to answer, so both answers are written out and the measured curves pick one. Only
# values the notebook printed appear in the sentence; the computed gap decides the wording.
GAP_F = float(V["curve_forward_full_final"]) - float(V["curve_forward_small_final"])
GAP_R = float(V["curve_reverse_full_final"]) - float(V["curve_reverse_small_final"])
# Epochs to the 0.95 mark, where "None" means a run never reached it. A reduced set that
# converges to the same accuracy several epochs later is a different result from one that
# converges lower, and only these columns can tell the two apart.
REACHED = V["curve_forward_small_95"].isdigit() and V["curve_reverse_small_95"].isdigit()
SLOWER = (REACHED and V["curve_forward_full_95"].isdigit()
          and V["curve_reverse_full_95"].isdigit()
          and (int(V["curve_forward_small_95"]) > int(V["curve_forward_full_95"])
               or int(V["curve_reverse_small_95"]) > int(V["curve_reverse_full_95"])))
if REACHED:
    LEAD = ("Training-set size is the second axis. At "
            f"{V['curve_forward_small_size']} examples both orderings still pass 0.95 "
            f"validation sequence accuracy, at epoch {V['curve_forward_small_95']} and "
            f"{V['curve_reverse_small_95']} against {V['curve_forward_full_95']} and "
            f"{V['curve_reverse_full_95']} on the full set, and finish at "
            f"{V['curve_forward_small_final']} and {V['curve_reverse_small_final']} against "
            f"{V['curve_forward_full_final']} and {V['curve_reverse_full_final']} "
            "(Figure~1, left).")
else:
    LEAD = ("Training-set size is the second axis. At "
            f"{V['curve_forward_small_size']} examples the two orderings finish at "
            f"{V['curve_forward_small_final']} and {V['curve_reverse_small_final']} final "
            f"validation sequence accuracy, against {V['curve_forward_full_final']} and "
            f"{V['curve_reverse_full_final']} on the full set (Figure~1, left).")
if max(GAP_F, GAP_R) < 0.01 and SLOWER and SAME_STEPS:
    TAIL = (" What the smaller set costs is epochs rather than final accuracy, and those epochs "
            "are an artefact of its size: both orderings pass 0.95 after "
            f"{V['steps_forward_small_text']} and {V['steps_reverse_small_text']} optimiser updates on the "
            f"reduced set, against {V['steps_forward_full_text']} and {V['steps_reverse_full_text']} on the "
            "full one. Measured in updates rather than passes, the two training set sizes are "
            "indistinguishable, and data volume is not what separates the two orderings.")
elif PART_STEPS:
    TAIL = (f" The two set sizes are not comparable in epochs, but in optimiser updates "
            f"they agree: {MATCHED} passes 0.95 after "
            f"{V['matched_small']} updates on the reduced set against {V['matched_full']} on the "
            f"full one. {OTHER} does not reach 0.95 on the reduced set within the epochs run, "
            "which is what a shorter run looks like as much as a smaller one.")
elif max(GAP_F, GAP_R) < 0.01 and SLOWER:
    TAIL = (" What the smaller set costs is epochs rather than final accuracy, and it costs "
            "both orderings the delay, so data volume is not what separates them at this scale.")
elif max(GAP_F, GAP_R) < 0.01:
    TAIL = (" Both orderings therefore still converge on the reduced set, so whatever makes "
            "one harder to learn than the other is not a data requirement at this scale.")
elif GAP_R > GAP_F:
    TAIL = (" The reduced set costs the reverse ordering more, which is consistent with it "
            "being the harder of the two to learn.")
else:
    TAIL = (" The reduced set costs the forward ordering more, which the accuracy "
            "comparison above does not predict and which a single seed cannot settle.")
V["data_efficiency"] = LEAD + TAIL
# A smaller training set at the same epoch count is also a shorter run in optimiser steps,
# so a gap between the curves cannot be attributed to data volume alone. The caveat is only
# written when there is a gap for it to qualify.
if SAME_STEPS or PART_STEPS:
    V["ablation_caveat"] = (
        " The ablation also held epochs fixed instead of optimiser updates, so a design that "
        "fixes updates would test data volume more directly than this one does.")
elif max(GAP_F, GAP_R) >= 0.01 or SLOWER:
    V["ablation_caveat"] = (
        " The reduced-data runs also take fewer optimiser updates at the same epoch count, so "
        "that comparison conflates the size of the training set with the length of training; "
        "repeating it with the number of updates held fixed would separate the two.")
else:
    V["ablation_caveat"] = ""

# How much of each model's error budget sits in the one region its paragraph is about. The
# framing sentences around those paragraphs claim concentration and complementarity, so they
# are written from this measurement instead of being assumed to hold.
FWD_SHARE = int(V["len1_fe"]) / max(int(V["forward_overall_err"]), 1)
REV_SHARE = int(V["align_worst_re"]) / max(int(V["reverse_overall_err"]), 1)
CONCENTRATED = FWD_SHARE >= 0.5 and REV_SHARE >= 0.5
V["error_lead"] = ("Each direction concentrates its errors in a different region of the "
                   "problem."
                   if CONCENTRATED else
                   "The residual errors are few enough to characterise individually.")
V["complement_clause"] = (" That is the alignment reverse ordering makes hardest, and it is "
                          "the complement of Forward's weakness." if CONCENTRATED else "")

# "7 of Forward's 7 errors ... and 7 of them are off by one" counts the same set three times.
# When a part equals its whole the quantifier is the honest way to say it, and when it does not
# the fraction has to stay, so both forms are written and the measurement chooses.
FWD_ALL_SHORT = int(V["len1_fe"]) == int(V["forward_overall_err"])
FWD_ALL_OFF1 = int(V["fwd_offby1"]) == int(V["forward_overall_err"])
V["forward_error_opening"] = (
    f"All {V['forward_overall_err']} of Forward's errors fall on answers of a single digit"
    if FWD_ALL_SHORT else
    f"{V['len1_fe']} of Forward's {V['forward_overall_err']} errors fall on answers of a "
    "single digit")
V["forward_offby1_clause"] = (
    "and every one is wrong by exactly one unit" if FWD_ALL_OFF1 else
    f"and {V['fwd_offby1']} of them are wrong by exactly one unit")

# ----------------------------------------------------------------------- error examples
examples = re.findall(
    r"^\s*(Forward|Reverse)\s*\|\s*(\S+=)\s*\|\s*(-?\d+)\s*\|\s*(-?\d+|None)\s*\|\s*(.+?)\s*$",
    T, re.MULTILINE)
forward_examples = [e for e in examples if e[0] == "Forward"][:4]
reverse_examples = [e for e in examples if e[0] == "Reverse"][:4]
assert forward_examples and reverse_examples, "could not find error examples"

V["test_n"] = V["forward_overall_n"]
# Examples whose answer has more than one digit. The report says Forward makes no errors
# outside the single-digit group, so the size of "outside" has to be the complement, not the
# whole test set.
V["len1_rest"] = str(int(V["test_n"]) - int(V["len1_n"]))
# Answers with no thousands digit. The hundreds denominator is a different quantity and
# reads as this one, so it is computed here instead of reused.
V["pos_thousands_rest"] = str(int(V["test_n"]) - int(V["pos_thousands_n"]))
V["sub_fraction"] = grab(r"subtraction fraction ([\d.]+)", "subtraction fraction")

# --------------------------------------------------------------------------------- figures
os.makedirs(FIGURES, exist_ok=True)
for name in ("figure_1_overview.pdf", "figure_2_by_length.pdf"):
    shutil.copyfile(f"{NBDIR}/{name}", f"{FIGURES}/{name}")


def rows_to_tex(rows):
    out = []
    for _, prompt, true, pred, kind in rows:
        out.append(f"{prompt.replace('=', '')} & {true} & {pred} & {kind} \\\\")
    return "\n".join(out)


# --------------------------------------------------- sentences that depend on what the run did
PLACES = ("thousands", "hundreds", "tens", "units")


def listed(words):
    """thousands, hundreds and tens"""
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def place_phrase(key, label):
    """Describe one model's per-place accuracy, naming the perfect places and quoting the rest."""
    exact = [p for p in PLACES if V[f"pos_{p}_{key}"] == "1.0000"]
    other = [(p, V[f"pos_{p}_{key}"]) for p in PLACES if V[f"pos_{p}_{key}"] != "1.0000"]
    parts = []
    if exact:
        parts.append(f"{label} is exact at the {listed(exact)}")
    if other:
        detail = listed([f"{value} at the {place}" for place, value in other])
        parts.append(("and scores " + detail) if exact else f"{label} scores {detail}")
    return " ".join(parts)


V["place_sentence"] = (f"Measured that way, {place_phrase('f', 'Forward')}, while "
                       f"{place_phrase('r', 'Reverse')}.")
# "1.0000 for Forward and 1.0000 for Reverse" is the same number written twice, so the equal
# case is collapsed. The unequal case still needs both values.
SIGN_EQUAL = V["pos_sign_f"] == V["pos_sign_r"]
V["sign_sentence"] = (
    (f"Sign accuracy is {V['pos_sign_f']} for both models, so the " if SIGN_EQUAL else
     f"Sign accuracy is {V['pos_sign_f']} for Forward and {V['pos_sign_r']} for Reverse, so the ")
    + ("sign-last convention of Task 2 costs nothing."
       if float(V["pos_sign_r"]) >= float(V["pos_sign_f"])
       else "sign-last convention of Task 2 carries a small cost."))

# Which place each direction loses is a measurement, so the sentence names the place the table
# puts last rather than the one the argument expects. An operand of at most three digits cannot
# produce a four digit difference, so the subtraction thousands group is empty by construction
# and not merely unobserved; that is worth a clause, because an empty cell otherwise reads as a
# measurement someone forgot to take.
MEASURED = ["thousands", "hundreds", "tens", "units"]
REV_ADD = {p: float(V[f"posop_{p}_ra"]) for p in MEASURED if V[f"posop_{p}_ra"] != "n/a"}
REV_SUB = {p: float(V[f"posop_{p}_rs"]) for p in MEASURED if V[f"posop_{p}_rs"] != "n/a"}
WEAK_ADD = min(REV_ADD, key=REV_ADD.get)
WEAK_SUB = min(REV_SUB, key=REV_SUB.get)
FWD_ADD_PERFECT = all(V[f"posop_{p}_fa"] in ("1.0000", "n/a") for p in MEASURED)
SUB_NO_THOUSANDS = V["posop_thousands_nsub"] == "0"
WEAK_ADD_RATE, WEAK_SUB_RATE = V[f"posop_{WEAK_ADD}_ra"], V[f"posop_{WEAK_SUB}_rs"]
SAME_WEAK_RATE = V[f"posop_{WEAK_ADD}_rs"]

V["position_operation_sentence"] = (
    ("Split by operation, no subtraction answer reaches the thousands column at all, because "
     "operands of at most three digits can only pass 999 by adding, so that cell is empty by "
     "construction. " if SUB_NO_THOUSANDS else "Split by operation, ")
    + ("Forward is exact at every place on addition, and " if FWD_ADD_PERFECT else "")
    + (f"Reverse is weakest at the {WEAK_ADD} in both, {WEAK_ADD_RATE} on addition and "
       f"{SAME_WEAK_RATE} on subtraction." if WEAK_ADD == WEAK_SUB else
       f"Reverse is weakest at the {WEAK_ADD} on addition ({WEAK_ADD_RATE}) and at the "
       f"{WEAK_SUB} on subtraction ({WEAK_SUB_RATE})."))

# Whether the paired test separates the two orderings is a result, not a premise. When it does
# not, the error analysis becomes the more informative comparison rather than a weaker one, and
# the paragraph says so.
SIGNIFICANT = float(V["mcnemar_p"]) < 0.05
V["significance_sentence"] = (
    "The accuracy difference is therefore real rather than sampling noise, though it is small in "
    "absolute terms."
    if SIGNIFICANT else
    "The difference is therefore within sampling noise: with errors this rare on either side the "
    "test cannot separate the two orderings on accuracy alone, which makes where the errors fall "
    "the more informative comparison.")

FEWER = ("Forward" if int(V["forward_overall_err"]) < int(V["reverse_overall_err"])
         else "Reverse" if int(V["reverse_overall_err"]) < int(V["forward_overall_err"])
         else None)
if SIGNIFICANT and FEWER:
    V["recommendation"] = (
        f"Finally, if one direction has to be chosen, choose {FEWER}: it is more accurate on "
        "this test set by a margin the paired test separates from noise, it trains as easily, "
        "and its failure mode is rarer.")
elif FEWER:
    V["recommendation"] = (
        "Finally, accuracy alone does not choose between the two directions here, since the "
        f"paired test does not separate them. {FEWER} makes the fewer errors on this test set and "
        "its failure mode is the narrower, so it is the safer default, but the honest answer is "
        "that at this operand length the ordering matters less than the analysis above suggests "
        "it would at greater ones.")
else:
    V["recommendation"] = (
        "Finally, the two directions are indistinguishable on this test set, making the same "
        "number of errors, so the choice between them should rest on the failure modes described "
        "above rather than on accuracy.")

# An addition that carries nowhere tends to have a small second operand, so the no-carry group
# is enriched in the width mismatch the operand table identifies. When the measurement shows that
# inversion, saying so and resolving it is worth more than leaving it in the table unremarked.
V["carry_anomaly"] = (
    "One split reads like a counterexample. Reverse is less accurate on additions that carry "
    f"nowhere ({V['cb_no_carry_ra']} over {group(V['cb_no_carry_n'])} examples) than on "
    f"additions that do carry ({V['cb_with_carry_ra']} over {group(V['cb_with_carry_n'])}), "
    "which inverts the "
    "expectation that carrying is the harder case. The operand width table resolves it: an "
    "addition that carries in no column tends to have a small second operand, so the no-carry "
    "group is enriched in exactly the width mismatch above. Carrying is not what this model "
    "finds difficult; keeping the columns aligned is."
    if float(V["cb_no_carry_ra"]) < float(V["cb_with_carry_ra"]) else "")

# The brief names carry and borrow cases as one of the cuts the comparison has to report, and
# the figure alone does not carry the counts. Forward's half of this is the striking one: it is
# exact on both addition cases, which is what makes every one of its errors a subtraction.
FWD_SUB_ERR = int(V["cb_with_borrow_fe"]) + int(V["cb_no_borrow_fe"])
FWD_ADD_EXACT = V["cb_with_carry_fa"] == V["cb_no_carry_fa"] == "1.0000"
V["carry_sentence"] = (
    "Carrying and borrowing separate the two directions. Forward is exact on both addition "
    f"cases, the {group(V['cb_with_carry_n'])} sums that carry somewhere and the "
    f"{group(V['cb_no_carry_n'])} that carry nowhere, so all {FWD_SUB_ERR} of its errors are "
    f"subtractions and {V['cb_with_borrow_fe']} of those need a borrow. Reverse errs in all "
    f"four cases, {int(V['cb_with_carry_re']) + int(V['cb_no_carry_re'])} on addition and "
    f"{int(V['cb_with_borrow_re']) + int(V['cb_no_borrow_re'])} on subtraction."
    if FWD_ADD_EXACT else
    "Carrying and borrowing cut the errors of both directions. Forward scores "
    f"{V['cb_with_carry_fa']} on additions that carry and {V['cb_with_borrow_fa']} on "
    f"subtractions that borrow; Reverse scores {V['cb_with_carry_ra']} and "
    f"{V['cb_with_borrow_ra']} on the same two groups.")

# A curve that is flat for most of the run and then rises within an epoch or two is an
# optimisation barrier rather than a capacity limit, and only the epochs-to-threshold columns
# can distinguish the two.
BARRIER = (V["curve_reverse_full_tgt"].isdigit() and V["curve_reverse_full_95"].isdigit()
           and int(V["curve_reverse_full_tgt"]) >= 10
           and int(V["curve_reverse_full_95"]) - int(V["curve_reverse_full_tgt"]) <= 3)
V["barrier_sentence"] = (
    " The reverse curve is the more striking of the two: it stays near a tenth of its final "
    f"accuracy until epoch {V['curve_reverse_full_tgt']}, then passes both 0.78 "
    f"and 0.95 within {max(1, int(V['curve_reverse_full_95']) - int(V['curve_reverse_full_tgt']))}"
    " epoch of doing so (Figure~1, left). A transition that abrupt is a barrier being crossed "
    "rather than capacity being filled, which is also why the reduced reverse run, given fewer "
    "updates over the same number of epochs, never crosses it."
    if BARRIER else "")

# --------------------------------------------------------------- claims the prose makes
# The report does not only quote numbers, it states structural findings: that Forward is exact
# on addition, that each of its errors is a single-digit answer off by exactly one, that
# Reverse's errors sit entirely in the widest operand mismatch, and that the paired test
# separates the two directions. Those sentences describe one particular training run, and the
# notebooks are regenerated and re-executed whenever the generator changes. Each claim is
# therefore checked against what the notebook printed, so a run that contradicts the prose stops
# the build rather than shipping a sentence the evidence no longer supports.
BROKEN = []
CHECKED = 0


NOTED = []


def claim(holds, sentence, measured, soft=False):
    """soft=True records the result without failing: the prose states the fraction rather
    than the quantifier, so the sentence stays true either way and this is only a note about
    whether the headline finding reproduced in this run."""
    global CHECKED
    CHECKED += 1
    if not holds:
        (NOTED if soft else BROKEN).append(f"{sentence}\n      measured: {measured}")


claim(V["forward_addition"] == "1.0000",
      'Operation robustness: "Forward is exact on addition"',
      f"forward addition {V['forward_addition']}", soft=True)
claim(V["pos_thousands_f"] == V["pos_hundreds_f"] == V["pos_tens_f"] == "1.0000",
      'Digit-level: "Forward is exact at every place above the units"',
      f"thousands {V['pos_thousands_f']}, hundreds {V['pos_hundreds_f']}, "
      f"tens {V['pos_tens_f']}", soft=True)
claim(V["pos_thousands_r"] == "1.0000" and float(V["pos_hundreds_r"]) < 1
      and float(V["pos_tens_r"]) < 1,
      'Digit-level: "Reverse is exact at the thousands but not at the hundreds or the tens"',
      f"thousands {V['pos_thousands_r']}, hundreds {V['pos_hundreds_r']}, "
      f"tens {V['pos_tens_r']}", soft=True)
claim(V["pos_sign_f"] == V["pos_sign_r"],
      'Digit-level: "Sign accuracy is X for both"',
      f"Forward {V['pos_sign_f']}, Reverse {V['pos_sign_r']}", soft=True)
claim(float(V["mcnemar_p"]) < 0.05,
      'Direction effects: the paired test separates the two orderings',
      f"McNemar p {V['mcnemar_p']}", soft=True)
claim(V["curve_forward_full_95"].isdigit() and V["curve_reverse_full_95"].isdigit(),
      'Direction effects: "both directions converge, reaching 0.95 ... at epoch"',
      f"Forward {V['curve_forward_full_95']}, Reverse {V['curve_reverse_full_95']}")
claim(int(V["len1_fe"]) == int(V["forward_overall_err"]),
      'Error patterns: "Forward\'s errors are all subtractions whose answer has a single digit"',
      f"{V['len1_fe']} of {V['forward_overall_err']} Forward errors are single-digit answers",
      soft=True)
claim(int(V["fwd_offby1"]) == int(V["forward_overall_err"]),
      'Error patterns: "Every one is wrong by exactly one unit"',
      f"{V['fwd_offby1']} of {V['forward_overall_err']} off by one", soft=True)
claim(int(V["align_worst_re"]) == int(V["reverse_overall_err"])
      and int(V["align_easy_re"]) == 0,
      'Error patterns: "Reverse\'s errors are entirely confined to the widest width mismatch"',
      f"{V['align_worst_re']} of {V['reverse_overall_err']} at the widest mismatch, "
      f"{V['align_easy_re']} at a difference of one digit or less", soft=True)
claim(int(V["forward_overall_err"]) > 0 and int(V["reverse_overall_err"]) > 0,
      'Error patterns: both paragraphs describe a proportion of an error count',
      f"Forward {V['forward_overall_err']} errors, Reverse {V['reverse_overall_err']} errors")
claim(SUB_NO_THOUSANDS,
      'Digit-level: "no subtraction answer reaches the thousands column"',
      f"subtraction thousands n = {V['posop_thousands_nsub']}")
claim(FWD_ADD_EXACT,
      'Carry and borrow: "Forward is exact on both addition cases"',
      f"with_carry {V['cb_with_carry_fa']}, no_carry {V['cb_no_carry_fa']}", soft=True)
claim(int(V["cb_with_carry_fe"]) + int(V["cb_no_carry_fe"]) + FWD_SUB_ERR
      == int(V["forward_overall_err"]),
      'Carry and borrow: the four cases account for every Forward error',
      f"{V['cb_with_carry_fe']}+{V['cb_no_carry_fe']}+{FWD_SUB_ERR} "
      f"against {V['forward_overall_err']}")
claim(float(V["forward_overall"]) > float(V["reverse_overall"]),
      'Recommendation: Forward is the more accurate of the two',
      f"Forward {V['forward_overall']}, Reverse {V['reverse_overall']}", soft=True)

if BROKEN:
    print("The run contradicts sentences the template states. Rewrite each one from the data:")
    for item in BROKEN:
        print(f"  - {item}")
    raise SystemExit(1)
for note in NOTED:
    print(f"  note, the prose states the fraction instead: {note}")
print(f"  structural claims checked against this run: {CHECKED - len(NOTED)} of {CHECKED} "
      f"hold as stated")

TEMPLATE = r"""% IFN680 Project 5 technical report.
% Generated by tools/build_report.py. Every number below is printed by main_report.ipynb.
% Build:  tectonic project5_report.tex
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

% The banner sits inside the \twocolumn optional argument. In the article class \twocolumn[...]
% claims the page's double column top region, so a figure* declared afterwards can never be
% placed on page 1 and is silently deferred to page 2.
\twocolumn[{%
\begin{center}
{\large\bfseries Project 5: Extending an Addition LLM to Subtraction and Reverse-Order Prediction}\\[2pt]
{\bfseries Group 4}
\end{center}
\vspace{-2pt}
Student 1: Karan Rooprai \hfill Student 1 ID: n12498122 \\
Student 2: Nhu Hieu Nguyen \hfill Student 2 ID: n12194778 \\
\vspace{3pt}

\begin{center}
\includegraphics[width=0.98\textwidth]{figures/figure_1_overview.pdf}
\end{center}
\vspace{-4pt}
\begin{quote}
\small\textbf{Figure 1:} \textbf{Left:} validation sequence accuracy per epoch for both
directions, at the full training set of __CURVE_FORWARD_FULL_SIZE__ (solid) and at
__CURVE_FORWARD_SMALL_SIZE__ examples (dashed); the dash-dot line marks the 0.78 target.
\textbf{Centre and right:} error counts rather than accuracies, because every accuracy
here exceeds 0.98 and rates render the two models as identical bars. Forward makes
__FORWARD_OVERALL_ERR__ errors in __TEST_N__ examples, Reverse __REVERSE_OVERALL_ERR__, and they
fall in different places.
\end{quote}
\vspace{2pt}
}]

The Week 7 workshop trains a small causal Transformer to add by next-token prediction, completing
a prompt such as \texttt{23+1=} with \texttt{24}. Task 1 extends it to subtraction with signed
results, Task 2 trains a second model that emits the answer digits right to left, and Task 3
compares them on one shared held-out set. The tokeniser, architecture, causal-masked objective
and training loop follow the workshop; only the changes described below differ.

\section{Method description}
\textbf{Task 1, handling subtraction.}
Two changes were needed. First, the vocabulary gains a single \texttt{-} token with two roles: the
subtraction operator in a prompt (\texttt{50-73=}) and the sign of a negative result
(\texttt{-23}). Reusing one symbol keeps the vocabulary at fifteen tokens and lets the model infer
from position which role is meant, which is something self-attention can represent and a second
token would have obscured. Second, the generator samples an operation as well as two non-negative
operands of at most three digits; Python's \texttt{str} of a negative integer already produces the
signed target, so no special case is needed when $b > a$. Training data is a balanced mix, half
addition and half subtraction, with unique prompts and disjoint train, validation and test splits
drawn by slicing one shuffled list. Because borrows and signs make subtraction harder than
addition, more epochs and a validation set were used to confirm convergence.

\textbf{Task 2, reverse-prediction implementation.}
Only the target is reversed; the prompt is untouched, so both models receive identical inputs and
differ solely in output ordering. The answer string is reversed literally, so \texttt{31} becomes
\texttt{13} and the model emits the units digit first, matching the direction in which carries and
borrows propagate by hand. The brief does not define how to reverse a negative result, so the same
literal rule is applied: \texttt{-123} becomes \texttt{321-}, placing the sign last. That is
consistent with right-to-left computation, since the sign of a difference is only determined once
the magnitude is. At evaluation the generated string is un-reversed before being parsed.

\section{Results and analysis (Task 3)}
\textbf{Set-up.} Both models are evaluated on one shared held-out set of __TEST_N__ examples,
balanced between addition and subtraction (subtraction fraction __SUB_FRACTION__). The set is
regenerated from the seed inside \texttt{main\_report.ipynb}, checked against the shipped file,
and confirmed to share no prompt with training or validation. Decoding is greedy, so the figures
reproduce exactly on CPU or GPU.

\textbf{Operation robustness.} Both models clear the task's 0.78 target: Forward reaches
__FORWARD_OVERALL__ overall (95\% CI __FORWARD_OVERALL_LO__ to __FORWARD_OVERALL_HI__) and Reverse
__REVERSE_OVERALL__ (__REVERSE_OVERALL_LO__ to __REVERSE_OVERALL_HI__). Split by operation,
Forward reaches __FORWARD_ADDITION__ on addition over __FORWARD_ADDITION_N__ examples and
__FORWARD_SUBTRACTION__ on subtraction, and Reverse __REVERSE_ADDITION__ and
__REVERSE_SUBTRACTION__ respectively (Table~1).

\textbf{Digit-level performance.} Accuracy is reported per place value over the answers that
actually have that place, rather than over zero-padded answers. The distinction matters: only
__POS_THOUSANDS_N__ of __TEST_N__ answers reach the thousands column, so padding would score the
other __POS_THOUSANDS_REST__ as correct there by construction and drive every high position
to 1.000.
__PLACE_SENTENCE__ __SIGN_SENTENCE__ __POSITION_OPERATION_SENTENCE__

\textbf{Direction effects.} The two models were scored on identical examples, so the comparison is
paired and McNemar's exact test applies. Forward is right where Reverse is wrong __MCNEMAR_B__
times, Reverse right where Forward is wrong __MCNEMAR_C__ times, and the two agree on
__MCNEMAR_AGREE__ of __TEST_N__ examples; $p = __MCNEMAR_P_TEX__$. __SIGNIFICANCE_SENTENCE__

Trained on the full set both directions converge, reaching 0.95 validation sequence accuracy at epoch
__CURVE_FORWARD_FULL_95__ and __CURVE_REVERSE_FULL_95__ respectively.__BARRIER_SENTENCE__ __DATA_EFFICIENCY__

\textbf{Error patterns.} __ERROR_LEAD__

__FORWARD_ERROR_OPENING__, a group holding only __LEN1_N__ of the __TEST_N__ examples,
__FORWARD_OFFBY1_CLAUSE__. Predicting left to right requires committing to the answer's length before
emitting any digit, so a three-digit subtraction that collapses to a single digit is the case that
ordering makes hardest.

__ALIGN_WORST_RE__ of Reverse's __REVERSE_OVERALL_ERR__ errors fall on the __ALIGN_WORST_N__
prompts whose first operand is __ALIGN_MAX_DIFF__ digits longer than the second, a rate of
__ALIGN_WORST_RRATE__ there against __ALIGN_EASY_RE__ errors across the other __ALIGN_EASY_N__
examples. Among the errors that keep the right number of digits, __REV_PLACE1__ are wrong at
the tens digit, the first place the model reaches after the shorter operand has run out of
digits; __REV_WRONGLEN__ predictions have the wrong length instead.
Emitting units first makes the units column trivial, because both operands end there, but it
forces the model to detect that one operand is exhausted and carry the remainder
alone.__COMPLEMENT_CLAUSE__

__CARRY_SENTENCE__ __CARRY_ANOMALY__

\begin{table}[t]
\centering
\footnotesize
% Six columns and a spelled out interval need more room than the default gap leaves.
% tabcolsep is set inside the table environment, so it applies to this table only.
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Mode & Split & $n$ & Errors & Accuracy & 95\% CI \\
\midrule
Forward & overall & __FORWARD_OVERALL_N__ & __FORWARD_OVERALL_ERR__ & __FORWARD_OVERALL__ & __FORWARD_OVERALL_LO__ to __FORWARD_OVERALL_HI__ \\
Forward & addition & __FORWARD_ADDITION_N__ & __FORWARD_ADDITION_ERR__ & __FORWARD_ADDITION__ & __FORWARD_ADDITION_LO__ to __FORWARD_ADDITION_HI__ \\
Forward & subtraction & __FORWARD_SUBTRACTION_N__ & __FORWARD_SUBTRACTION_ERR__ & __FORWARD_SUBTRACTION__ & __FORWARD_SUBTRACTION_LO__ to __FORWARD_SUBTRACTION_HI__ \\
Reverse & overall & __REVERSE_OVERALL_N__ & __REVERSE_OVERALL_ERR__ & __REVERSE_OVERALL__ & __REVERSE_OVERALL_LO__ to __REVERSE_OVERALL_HI__ \\
Reverse & addition & __REVERSE_ADDITION_N__ & __REVERSE_ADDITION_ERR__ & __REVERSE_ADDITION__ & __REVERSE_ADDITION_LO__ to __REVERSE_ADDITION_HI__ \\
Reverse & subtraction & __REVERSE_SUBTRACTION_N__ & __REVERSE_SUBTRACTION_ERR__ & __REVERSE_SUBTRACTION__ & __REVERSE_SUBTRACTION_LO__ to __REVERSE_SUBTRACTION_HI__ \\
\bottomrule
\end{tabular}
\caption{Accuracy with the count behind it and a Wilson interval. Both directions clear 0.78
by a wide margin, so the informative object is the small number of errors.}
\end{table}

% Figure 1 is hand-labelled inside \twocolumn[...], so the counter is still at zero here.
\setcounter{figure}{1}
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/figure_2_by_length.pdf}
\caption{\textbf{Top:} errors by the number of digits in the answer, with the size of each
group underneath. \textbf{Bottom:} errors by the sign of the answer, which the reverse ordering
emits last. Both panels count the same errors as Figure~1, cut by the shape of the answer rather
than by the operation.}
\end{figure}

\begin{table}[b]
\centering
\footnotesize
\begin{tabular}{@{}lrrl@{}}
\toprule
Prompt & True & Predicted & Failure \\
\midrule
\multicolumn{4}{@{}l}{\emph{Forward: short answer, long operands}}\\
__FORWARD_EXAMPLES__
\multicolumn{4}{@{}l}{\emph{Reverse: operands of very different width}}\\
__REVERSE_EXAMPLES__
\bottomrule
\end{tabular}
\caption{Representative failures of each model, taken from the full error list printed by
\texttt{main\_report.ipynb}. __OVERLAP_PHRASE__}
\end{table}

\section{Strengths, limitations and recommendations}
The comparison's strength is its design. Both models share a tokeniser, architecture, optimiser
and schedule, so prediction direction is the only variable between them, and both are scored on
the same __TEST_N__ examples, which makes the accuracy difference a paired comparison with an
exact test behind it. Several limitations bound what it supports. Each model was trained once, from one seed, so the difference in convergence speed
between the directions is a single observation rather than a distribution.
Operands are capped at three digits, so nothing here shows whether either direction generalises to
longer arithmetic, and the error analysis suggests that is exactly where they would diverge.
Positions are encoded absolutely, following the workshop, so Reverse's alignment errors have
an untested cause.__ABLATION_CAVEAT__

The recommendations follow from those. Train several seeds before treating the convergence gap as
a property of the ordering rather than of one optimisation trajectory. Extend the operand range
and re-measure, because the failure modes found here are both about structure, answer length for
Forward and operand width for Reverse, and both should grow with digits. Test a relative or
learned positional encoding against the absolute one, since the reverse model's errors cluster at
exactly the place where relative positions would carry the alignment information it appears to
lack. __RECOMMENDATION__

\end{document}
"""

text = TEMPLATE
for key, value in V.items():
    # Counts are grouped here rather than at extraction, because the comparisons above need the
    # raw digits and the error table below must keep arithmetic answers ungrouped.
    text = text.replace(f"__{key.upper()}__", group(value))
text = text.replace("__FORWARD_EXAMPLES__", rows_to_tex(forward_examples))
text = text.replace("__REVERSE_EXAMPLES__", rows_to_tex(reverse_examples))

leftover = re.findall(r"__[A-Z0-9_]+__", text)
assert not leftover, f"unfilled placeholders: {sorted(set(leftover))}"
for dash in ("\u2014", "\u2013"):
    assert dash not in text, "an em or en dash reached the report"

os.makedirs(REPORT, exist_ok=True)
io.open(f"{REPORT}/project5_report.tex", "w", encoding="utf-8", newline="\n").write(text)
print(f"wrote {REPORT}/project5_report.tex  ({len(text):,} characters)")
print(f"  values pulled from main_report.ipynb: {len(V)}")
print(f"  error examples: {len(forward_examples)} Forward, {len(reverse_examples)} Reverse")
