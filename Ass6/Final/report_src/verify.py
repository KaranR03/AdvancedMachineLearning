r"""Every check that must pass before Project 6 is uploaded.

Adapted from Project 5's tools/verify.py. The cliff edges the unit states are asserted
mechanically: a report over 2 pages scores 0, wrong file names score 0, and the brief says of
main_report.ipynb that "This notebook will be run for grading, failure on any of the cell will
results in 0 mark for the code evaluation." A cliff check never skips: a missing tool is itself a
failure, so the script cannot finish with ALL CHECKS PASSED while a rail has not run.

Run:  python tools/verify.py
"""
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-09/project-6-sharpness-quest")
# P6_NBDIR / P6_REPORT / P6_SUBMISSION point the checks at another run, such as the smoke room.
NBDIR = os.environ.get("P6_NBDIR", f"{BASE}/notebook")
REPORT = os.environ.get("P6_REPORT", f"{BASE}/report")
SUBMISSION = os.environ.get("P6_SUBMISSION", f"{BASE}/submission")
TEXBIN = "C:/Users/Admin/.conda/envs/tex/Library/bin"

# Where the stored outputs must come from. "cuda" is the IFN680 GPU node; it would become "cpu"
# only if the Hub had been unreachable, and the archive's README would then have to say so.
EXPECTED_DEVICE = "cuda"

NOTEBOOKS = ("cVAE_Baseline.ipynb", "cVAE_DiscriminatorLoss.ipynb", "DigitClassifier.ipynb",
             "main_report.ipynb")
SUPPORT = ("cVAE_Baseline.pth", "cVAE_DiscriminatorLoss.pth", "DigitClassifier.pth",
           "cVAE_Baseline_history.json", "cVAE_DiscriminatorLoss_history.json",
           "mnist_custom_test.pt", "figure_1_grids.pdf", "figure_2_reconstructions.pdf")
HEADINGS = ("Introduction", "Methodology", "Experiments", "Discussion", "Conclusion")

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def load(name):
    return json.load(io.open(f"{NBDIR}/{name}", encoding="utf-8"))


def stream_text(notebook):
    return "".join("".join(o.get("text", [])) for c in notebook["cells"]
                   for o in c.get("outputs", []) if o.get("output_type") == "stream")


def code_of(notebook):
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")


notebooks = {name: load(name) for name in NOTEBOOKS}
printed = {name: stream_text(nb) for name, nb in notebooks.items()}
main = printed["main_report.ipynb"]
main_code = code_of(notebooks["main_report.ipynb"])
histories = {name: json.load(io.open(f"{NBDIR}/{name}_history.json", encoding="utf-8"))
             for name in ("cVAE_Baseline", "cVAE_DiscriminatorLoss")}

# ---------------------------------------------------------------- 1. names are pass/fail
print("\n1. File naming (wrong names score 0)")
check("report is named project6_report.pdf", os.path.exists(f"{REPORT}/project6_report.pdf"))
for name in NOTEBOOKS + SUPPORT:
    check(f"notebook/{name} present", os.path.exists(f"{NBDIR}/{name}"))

# ---------------------------------------------------------------- 2. the report
print("\n2. Report (over 2 pages scores 0)")
from pypdf import PdfReader  # noqa: E402

pdf = PdfReader(f"{REPORT}/project6_report.pdf")
check("report is exactly 2 pages", len(pdf.pages) == 2, f"{len(pdf.pages)} pages")
tex = io.open(f"{REPORT}/project6_report.tex", encoding="utf-8").read()
for person, sid in (("Karan Rooprai", "n12498122"), ("Nhu Hieu Nguyen", "n12194778")):
    check(f"report names {person} with {sid}", person in tex and sid in tex)
check("report names the group", re.search(r"Group\s+4", tex) is not None)
check("no unfilled template placeholders", not re.search(r"__[A-Z0-9_]+__", tex))
check("no pending qualitative paragraph", "PENDING" not in tex)
# The brief lists five sections; its own words are the headings, in its order.
positions = [tex.find(f"\\section{{{h}}}") for h in HEADINGS]
check("the five sections the brief names are headings, in its order",
      all(p >= 0 for p in positions) and positions == sorted(positions),
      ", ".join(f"{h} {'ok' if p >= 0 else 'MISSING'}" for h, p in zip(HEADINGS, positions)))
check("a reference list follows the Conclusion",
      tex.find(r"\section*{References}") > positions[-1] >= 0)

PDFTOTEXT = f"{TEXBIN}/pdftotext.exe"
if os.path.exists(PDFTOTEXT):
    LEFT, RIGHT = 42.52, 552.76      # 1.5cm margins on a 595.28pt A4 page
    SLACK = 3.0                      # microtype protrusion plus glyph side bearings
    with tempfile.TemporaryDirectory() as tmp:
        boxes = f"{tmp}/bbox.xml"
        subprocess.run([PDFTOTEXT, "-bbox", f"{REPORT}/project6_report.pdf", boxes],
                       check=True, capture_output=True)
        words = re.findall(r'<word xMin="([\d.]+)"[^>]*xMax="([\d.]+)"[^>]*>([^<]*)</word>',
                           io.open(boxes, encoding="utf-8").read())
    outside = [(w, float(a), float(b)) for a, b, w in words
               if float(a) < LEFT - SLACK or float(b) > RIGHT + SLACK]
    check("no text spills outside the page margins", not outside,
          f"{len(words)} words" if not outside
          else f"{len(outside)} outside, first is {outside[0][0]!r}")
else:
    check("no text spills outside the page margins", False, f"pdftotext not at {PDFTOTEXT}")

rendered = "\n".join(page.extract_text() for page in pdf.pages)
for dash, label in (("\u2014", "em"), ("\u2013", "en")):
    check(f"no {label} dash in the rendered PDF", dash not in rendered)
    check(f"no {label} dash in the .tex", dash not in tex)

# ---------------------------------------------------------------- 3. notebook hygiene
print("\n3. Notebook hygiene (no warnings, no errors)")
for name, notebook in notebooks.items():
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    stderr = [o for c in code for o in c.get("outputs", [])
              if o.get("output_type") == "stream" and o.get("name") == "stderr"]
    counts = [c.get("execution_count") for c in code]
    source = code_of(notebook)
    lines = [line for line in source.split("\n") if line.strip()]
    comments = [line for line in lines if line.strip().startswith("#")]
    check(f"{name}: no error outputs", not errors)
    check(f"{name}: no stderr output", not stderr,
          "" if not stderr else "".join(stderr[0].get("text", ""))[:160])
    check(f"{name}: every code cell executed, counts 1..N in order",
          counts == list(range(1, len(code) + 1)), f"{len(code)} cells")
    check(f"{name}: every code cell carries a comment",
          all(any(line.strip().startswith("#") for line in c["source"]) for c in code))
    check(f"{name}: comment ratio at least 0.12", len(comments) / len(lines) >= 0.12,
          f"{len(comments)}/{len(lines)} = {len(comments) / len(lines):.0%}")
    check(f"{name}: no tqdm progress bars (they write to stderr)", "tqdm" not in source)
    for dash, label in (("\u2014", "em"), ("\u2013", "en")):
        check(f"{name}: no {label} dash", dash not in json.dumps(notebook, ensure_ascii=False))
    missing = [i for i, c in enumerate(notebook["cells"]) if not c.get("id")]
    check(f"{name}: every cell carries an id", not missing)
    check(f"{name}: source lines keep their newlines",
          all(all(line.endswith("\n") for line in c["source"][:-1])
              for c in notebook["cells"] if len(c["source"]) > 1))
    echoes = [c["id"] for c in notebook["cells"]
              for o in c.get("outputs", []) if o.get("output_type") == "execute_result"]
    check(f"{name}: echoes no object repr", not echoes, f"cells {echoes}")
    check(f"{name}: no f-string needing Python 3.12 (PEP 701)",
          not re.search(r"""f"[^"\n]*\{[^}"\n]*"[^"\n]*"[^}\n]*\}""", source)
          and not re.search(r"""f'[^'\n]*\{[^}'\n]*'[^'\n]*'[^}\n]*\}""", source))

check("main_report.ipynb trains nothing",
      not re.search(r"\.backward\(|\.step\(\)|\.train\(\)|zero_grad", main_code))
# Comments are stripped first: one explains what forward() does without calling it.
main_uncommented = re.sub(r"#.*", "", main_code)
check("main_report.ipynb never calls a cVAE's forward pass",
      not re.search(r"\b(baseline|adversarial|models)\[[^]]*\]\(", main_uncommented)
      and not re.search(r"(?<!def )\bforward\(", main_uncommented))
check("every torch.randn in main_report.ipynb draws from a seeded generator",
      all("generator=" in line for line in main_code.split("\n") if "torch.randn(" in line))
check("main_report.ipynb asserts nothing about a metric value",
      all(re.search(r"shape|sorted\(", line) for line in main_code.split("\n")
          if line.strip().startswith("assert")))
check("main_report.ipynb loads every checkpoint with weights_only=True",
      main_code.count("weights_only=True") == 3, f"{main_code.count('weights_only=True')} uses")

devices = {n: (re.findall(r"^device\s*:\s*(\S+)", printed[n], re.M) or ["?"])[0]
           for n in NOTEBOOKS}
check("all four notebooks ran on one device", len(set(devices.values())) == 1, f"{devices}")
check(f"the stored outputs come from the {EXPECTED_DEVICE} device",
      all(d.startswith(EXPECTED_DEVICE) for d in devices.values()), f"{devices}")

# ---------------------------------------------------------------- 4. the protocol
print("\n4. Protocol: development split, refit, test split held back")
for name in ("cVAE_Baseline.ipynb", "cVAE_DiscriminatorLoss.ipynb", "DigitClassifier.ipynb"):
    source = code_of(notebooks[name])
    uses = [line for line in source.split("\n") if "test_images" in line]
    check(f"{name}: touches the test images only to save the split",
          len(uses) == 1 and "test_split" in uses[0], f"{len(uses)} line(s)")
check("main_report.ipynb reads the test split from its own file",
      'torch.load("mnist_custom_test.pt"' in main_code)
check("main_report.ipynb takes lambda from the history file",
      'adversarial_record["lambda"]' in main_code
      and not re.search(r"^\s*LAMBDA\s*=\s*[\d.]", main_code, re.M))
base_h, adv_h = histories["cVAE_Baseline"], histories["cVAE_DiscriminatorLoss"]
check("both cVAEs trained for the same number of epochs", base_h["epochs"] == adv_h["epochs"],
      f"{base_h['epochs']} and {adv_h['epochs']}")
check("both cVAEs trained with the same seeds", base_h["seeds"] == adv_h["seeds"] == [0, 1, 2],
      f"{base_h['seeds']} and {adv_h['seeds']}")
check("the final models were refit on all 60,000 training images",
      base_h["n_refit"] == adv_h["n_refit"] == 60000, f"{base_h['n_refit']}, {adv_h['n_refit']}")
check("each development curve holds one value per epoch",
      len(base_h["dev"]["dev_recon"]) == len(adv_h["dev"]["dev_recon"]) == base_h["epochs"])
check("each seed's refit curve holds one value per epoch",
      all(len(h["train_recon"]) == base_h["epochs"]
          for record in (base_h, adv_h) for h in record["refit"].values()))
for key, label in (("cVAE_Baseline", "baseline"), ("cVAE_DiscriminatorLoss",
                                                   "discriminator loss")):
    stored = histories[key]["convergence"]
    line = re.search(rf"^convergence \| {label} \| .*change ([+-][\d.]+) \| within 1% (\w+)",
                     main, re.M)
    notebook_line = re.search(r"relative change ([+-][\d.]+), within 1%: (\w+)",
                              printed[f"{key}.ipynb"])
    check(f"{label}: main_report re-derives the training notebook's convergence statistic",
          line is not None and notebook_line is not None
          and line.groups() == notebook_line.groups(),
          f"notebook {notebook_line.groups() if notebook_line else '?'}, "
          f"main_report {line.groups() if line else '?'}")
    check(f"{label}: converged by the 1% rule", stored["within_1pct"] is True,
          f"change {stored['relative_change']:+.4f}")
chosen = [row for row in adv_h["sweep"] if row["lambda"] == adv_h["lambda"]]
check("the chosen lambda is one the sweep ran", len(chosen) == 1, f"lambda {adv_h['lambda']}")
eligible = [row for row in adv_h["sweep"] if row["d_accuracy"] <= 0.98] or \
    [min(adv_h["sweep"], key=lambda row: row["d_accuracy"])]
rule = min(eligible, key=lambda row: (abs(row["dev_sharpness"] - row["real_dev_sharpness"]),
                                      row["lambda"]))["lambda"]
check("the chosen lambda is what the stated rule picks from the sweep", rule == adv_h["lambda"],
      f"rule {rule}, stored {adv_h['lambda']}")
check("the classifier notebook never scores the test split",
      "test_labels" not in code_of(notebooks["DigitClassifier.ipynb"]).replace(
          '("test_images", "test_labels")', ""))

# ---------------------------------------------------------------- 5. analysis present
print("\n5. main_report ran every analysis the report uses")
for pattern, label in (
        (r"^grid accuracy \| Baseline cVAE \|", "baseline 12x10 grid scored"),
        (r"^grid accuracy \| cVAE with discriminator loss", "discriminator 12x10 grid scored"),
        (r"^metric \| mse_recon \|", "reconstruction MSE"),
        (r"^metric \| sharpness_reconstructions \|", "Laplacian variance, reconstructions"),
        (r"^metric \| midgrey_reconstructions \|", "mid-grey fraction, reconstructions"),
        (r"^metric \| sharpness_samples \|", "Laplacian variance, samples"),
        (r"^metric \| midgrey_samples \|", "mid-grey fraction, samples"),
        (r"^decision \| reconstructions", "decision rule on reconstructions"),
        (r"^decision \| samples", "decision rule on samples"),
        (r"^classifier \| real test images", "classifier on real test digits"),
        (r"^classifier \| samples baseline .*\| seeds ", "classifier on baseline samples"),
        (r"^classifier \| samples discriminator .*\| seeds ", "classifier on discriminator samples"),
        (r"^frechet \| floor", "Frechet floor"),
        (r"^frechet \| samples \|", "Frechet distance, samples"),
        (r"^frechet \| reconstructions \|", "Frechet distance, reconstructions"),
        (r"^spread \|", "within-class spread"),
        (r'^\s*"decisions": \{', "machine-readable summary")):
    check(f"main_report printed the {label}", re.search(pattern, main, re.M) is not None)
check("per-class accuracies for all ten digits",
      len(re.findall(r"^class \d \|", main, re.M)) == 10)
check("every metric row carries one difference per seed",
      all(len(m.split()) == 3 for m in re.findall(r"^metric \|.*\| seeds (.+)$", main, re.M)))

# ---------------------------------------------------------------- 6. traceability
print("\n6. Traceability: every number in the report is printed by main_report.ipynb")
body = tex.split(r"\begin{document}", 1)[1].split(r"\section*{References}", 1)[0]
body = re.sub(r"(?<!\\)%.*", "", body)
body = re.sub(r"\\includegraphics\[[^]]*\]\{[^}]*\}", "", body)
body = re.sub(r"\\(vspace|setcounter|hfill)\*?\{[^}]*\}(\{[^}]*\})?", "", body)
body = re.sub(r"\[\d+pt\]", "", body)
body = re.sub(r"n\d{8}", "", body)                        # student ids
body = re.sub(r"pp?\.~[\d, ]+", "", body)                  # lecture page references
body = re.sub(r"\\S\d+", "", body)                         # section signs
body = re.sub(r"\b(19[9]\d|20[0-2]\d)\b(?=\)|;|,)", "", body)   # citation years
flat = re.sub(r"\s+", " ", body)
# Settings the design fixes, and nothing the run measured: the tutorial's loss weights, learning
# rates, batch and split sizes, image and kernel sizes, the rule's thresholds, the tutorial's own
# epoch count, the grid's shape, the bootstrap count and the Inception input size.
DESIGN = {"100", "0.1", "0.001", "1,000", "10,000", "60,000", "20", "3", "7", "8.3", "50",
          "0.9", "98", "95", "2,000", "12", "120", "128", "299", "10", "2", "4", "1"}
numbers = set(re.findall(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d])|(?<![\w.,])\d+\.\d+|"
                         r"(?<![\w.,])\d{2,}(?![\d.,])", flat))
untraceable = sorted(n for n in numbers - DESIGN if n not in main)
check("no number appears only in the report", not untraceable,
      f"untraceable: {untraceable}" if untraceable else f"{len(numbers)} checked")
check("no Python exponent notation survives into the report",
      not re.search(r"\de[+-]\d", body))
prose = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
ungrouped = re.findall(r"(?<![\d.,])\d{4,}(?![\d,])", prose)
check("every count of a thousand or more is grouped in prose", not ungrouped,
      f"ungrouped: {sorted(set(ungrouped))}")

# ---------------------------------------------------------------- 7. the archive
print("\n7. The code archive")
archive = f"{SUBMISSION}/project6_code.zip"
if os.path.exists(archive):
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for name in NOTEBOOKS + SUPPORT:
            check(f"zip contains {name}", name in names)
        check("zip has a README.txt", "README.txt" in names)
        check("zip has no .DS_Store or checkpoints",
              not any(".DS_Store" in n or ".ipynb_checkpoints" in n for n in names))
        check("zip is flat (no nested folder)", not any("/" in n for n in names),
              f"{len(names)} entries")
        for name in NOTEBOOKS + SUPPORT:
            if name in names:
                same = (hashlib.sha256(zf.read(name)).hexdigest()
                        == hashlib.sha256(io.open(f"{NBDIR}/{name}", "rb").read()).hexdigest())
                check(f"zip's {name} is the current notebook/ file", same)
        readme = zf.read("README.txt").decode("utf-8") if "README.txt" in names else ""
    submitted = f"{SUBMISSION}/project6_report.pdf"
    check("submission report is the current build",
          os.path.exists(submitted)
          and hashlib.sha256(io.open(submitted, "rb").read()).hexdigest()
          == hashlib.sha256(io.open(f"{REPORT}/project6_report.pdf", "rb").read()).hexdigest())
    check("archive is under 50 MB", os.path.getsize(archive) < 50 * 1024 * 1024,
          f"{os.path.getsize(archive) / 1024 / 1024:.1f} MB")
    FILENAME = re.compile(r"[A-Za-z0-9_.*-]+\.(?:ipynb|pth|pkl|json|pdf|pt|zip|txt|csv)\b")
    # mnist_custom.pt is the dataset the brief supplies on Canvas; the README says where it lives.
    SUPPLIED = {"mnist_custom.pt"}
    for name in NOTEBOOKS:
        text = "\n".join("".join(c["source"]) for c in notebooks[name]["cells"])
        named = {n for n in FILENAME.findall(text) if "*" not in n}
        dangling = sorted(n for n in named if n not in names and n not in SUPPLIED)
        check(f"{name} names only files the archive ships", not dangling, f"missing {dangling}")
else:
    readme = ""
    check("project6_code.zip built", False, "run tools/build_zip.py")

# ------------------------------------------------- 8. what the deliverables must not contain
print("\n8. Deliverables explain the subject, not the submission")
ASSESSMENT_WORDS = re.compile(
    r"\b(marker|markers|rubric|graded|grades|grading|feedback|deduct\w*)\b", re.I)
SELF_GRADING = re.compile(
    r"rather than (asserted|assumed|eyeballed|claimed|intended|taken on trust)", re.I)
sources = {name: "\n".join("".join(c["source"]) for c in nb["cells"])
           for name, nb in notebooks.items()}
sources["project6_report.tex"] = tex
sources["README.txt"] = readme
for name, source in sources.items():
    hits = ASSESSMENT_WORDS.findall(source) + [m.group(0) for m in SELF_GRADING.finditer(source)]
    # "grading time" names when the notebook is run, which is the brief's own framing of it.
    hits = [h for h in hits if h.lower() != "grading" or "grading time" not in source.lower()]
    check(f"{name} does not talk about how it is marked", not hits, f"found {sorted(set(hits))}")

# ---------------------------------------------------------------- 9. report-specific
print("\n9. What the report says, against what the run did")
offset = re.search(r"\\setcounter\{figure\}\{(\d+)\}", tex)
floats = []
for kind, first in (("figure", int(offset.group(1)) + 1 if offset else 1), ("table", 1)):
    for n, _ in enumerate(re.finditer(r"\\begin\{" + kind + r"\*?\}", body), start=first):
        floats.append((kind.capitalize(), n))
prose_flat = re.sub(r"\s+", " ", prose)
uncited = [f"{k} {n}" for k, n in floats if f"{k}~{n}" not in prose_flat]
check("every numbered float is cited in the prose", not uncited,
      f"uncited: {uncited}" if uncited else ", ".join(f"{k} {n}" for k, n in floats))
check("Figure 1 is cited in the prose", "Figure~1" in prose_flat)

sentences = re.sub(r"\$[^$]*\$", " x ", prose)
sentences = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?(?:\{[^{}]*\})?", " ", sentences)
opens_numeric = []
for para in re.split(r"\n\s*\n", sentences):
    para = re.sub(r"\s+", " ", para).strip()
    if para and re.match(r"\d", para):
        opens_numeric.append(para[:40])
    opens_numeric += [m.group(1) for m in re.finditer(r"[.!?]\s+(\d[\d.,]*)", para)]
check("no sentence in the report opens with a numeral", not opens_numeric, f"{opens_numeric}")

said_lambda = re.search(r"so \$\\lambda = ([\d.]+)\$ was chosen", flat)
check("the report's lambda is the stored one",
      said_lambda is not None and float(said_lambda.group(1)) == adv_h["lambda"],
      f"report {said_lambda.group(1) if said_lambda else '?'}, stored {adv_h['lambda']}")
said_epochs = re.search(r"from 0\.001 to 0 over (\d+) epochs", flat)
check("the report's epoch count is the stored one",
      said_epochs is not None and int(said_epochs.group(1)) == base_h["epochs"],
      f"report {said_epochs.group(1) if said_epochs else '?'}, stored {base_h['epochs']}")
check("the report names the seeds that were trained", "seeds 0, 1 and 2" in flat)
check("the report states the real = 1, fake = 0 convention",
      "real images are labelled 1 and fakes 0" in flat)
check("the report names the non-saturating generator loss", "non-saturating" in flat)
check("the report states the hypothesis verbatim",
      "adding a discriminator term to the loss function of a cVAE during training increases "
      "the sharpness of the reconstructed images" in flat)

decision_recon = re.search(r"^decision \| reconstructions sharper with the discriminator loss: "
                           r"(True|False)", main, re.M)
conclusion = flat.split(r"\section{Conclusion}", 1)[-1]
supports = "The evidence supports the hypothesis" in conclusion
check("the Conclusion's verdict is the decision rule's",
      decision_recon is not None and supports == (decision_recon.group(1) == "True"),
      f"rule {decision_recon.group(1) if decision_recon else '?'}, conclusion supports {supports}")
conclusion_text = re.sub(r"\\[A-Za-z]+\*?(\{[^}]*\})?", " ", conclusion)
conclusion_text = re.sub(r"(et al|e\.g|i\.e)\.", r"\1", conclusion_text)
n_sentences = len(re.findall(r"[.!?](\s|$)", conclusion_text.strip()))
check("the Conclusion is at most three sentences", n_sentences <= 3, f"{n_sentences}")
discussion = flat.split(r"\section{Discussion}", 1)[-1].split(r"\section{Conclusion}", 1)[0]
for word in ("quality", "sharp", "consisten", "variet"):
    check(f"the Discussion addresses {word}...", word in discussion.lower())
check("the Discussion makes recommendations from strengths and limitations",
      "Recommendations" in discussion and "Strengths and limitations" in discussion)

# Label sizes at placement: authored size times placement width over the figure's native width.
TEXTWIDTH = 18 / 2.54
PLACED = {"figure_1_grids.pdf": 0.84 * TEXTWIDTH,
          "figure_2_reconstructions.pdf": (TEXTWIDTH - 0.6 / 2.54) / 2}
for figure, placed in PLACED.items():
    path = f"{REPORT}/figures/{figure}"
    if not os.path.exists(path):
        check(f"{figure} present in report/figures", False)
        continue
    native = float(PdfReader(path).pages[0].mediabox.width) / 72
    cell = re.search(rf"savefig\(\"{figure}\"", main_code)
    start = max(main_code.rfind("plt.subplots", 0, cell.start()),
                main_code.rfind("Figure(figsize", 0, cell.start())) if cell else -1
    block = main_code[start:cell.start()] if start >= 0 else ""
    authored = [float(s) for s in re.findall(r"fontsize=(\d+(?:\.\d+)?)", block)]
    smallest = min(authored) if authored else 10.0
    check(f"{figure}: no label renders below 6pt", smallest * placed / native >= 6.0,
          f"{smallest:g}pt at {native:.2f}in placed at {placed:.2f}in renders "
          f"{smallest * placed / native:.1f}pt")

# Every function main_report defines is used.
tree = ast.parse(main_code)
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)} - {"forward",
                                                                                   "__init__"}
referenced = ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
              | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})
check("every function main_report.ipynb defines is used", not (defined - referenced),
      f"unused: {sorted(defined - referenced)}")

print("\n" + ("ALL CHECKS PASSED" if not failures
             else f"{len(failures)} CHECK(S) FAILED: " + "; ".join(failures)))
raise SystemExit(1 if failures else 0)
