r"""Every check that must pass before Project 7 is uploaded.

Adapted from Project 6's tools/verify.py. The cliff edges the unit states are asserted
mechanically: a report over 2 pages scores 0, wrong file names score 0, and the brief says of
main_report.ipynb that "This notebook will be run for grading; failure on any of the cells will
result in a 0 mark for the code evaluation." A cliff check never skips: a missing tool is itself a
failure, so the script cannot finish with ALL CHECKS PASSED while a rail has not run.

Run:  python tools/verify.py
"""
import ast
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
import zipfile

import numpy as np
import torch

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-10/project-7-improving-ddpm")
TUTORIAL = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
            "weeks/week-09/Week_9_solution.ipynb")
# P7_NBDIR / P7_REPORT / P7_SUBMISSION point the checks at another run, such as the smoke room.
NBDIR = os.environ.get("P7_NBDIR", f"{BASE}/notebook")
REPORT = os.environ.get("P7_REPORT", f"{BASE}/report")
SUBMISSION = os.environ.get("P7_SUBMISSION", f"{BASE}/submission")
TEXBIN = "C:/Users/Admin/.conda/envs/tex/Library/bin"
SMOKE = "P7_NBDIR" in os.environ

# Where the stored outputs must come from. "cuda" is the IFN680 GPU node; it would become "cpu"
# only if the Hub had been unreachable, and the archive's README would then have to say so.
EXPECTED_DEVICE = "cpu" if SMOKE else "cuda"

NOTEBOOKS = ("DDPM_CosineSchedule.ipynb", "DigitClassifier.ipynb", "main_report.ipynb")
SUPPORT = ("ddpm_linear.pth", "ddpm_cosine.pth", "DigitClassifier.pth", "DDPM_history.json",
           "mnist_custom_test.pt", "figure_1_grids.pdf", "figure_2_steps.pdf",
           "figure_3_few_steps.pdf", "figure_4_mechanism.pdf")
HEADINGS = ("Introduction", "Methodology", "Experiments", "Discussion", "Conclusion")
HYPOTHESIS = ("A cosine schedule allows a denoising diffusion probabilistic model to generate "
              "high-quality samples using fewer time steps than a linear schedule.")
CARRIED = ("extract", "linear_beta_schedule", "ConvBlock", "UNet_cond")
# The tutorial's own functions, in the order Tutorial 9.3 defines them.
TUTORIAL_ORDER = ("extract", "linear_beta_schedule", "q_sample", "ConvBlock", "UNet_cond",
                  "p_sample_cDDPM", "p_sample_loop_cDDPM")

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
ddpm_code = code_of(notebooks["DDPM_CosineSchedule.ipynb"])
history = json.load(io.open(f"{NBDIR}/DDPM_history.json", encoding="utf-8"))

# ---------------------------------------------------------------- 1. names are pass/fail
print("\n1. File naming (wrong names score 0)")
check("report is named project7_report.pdf", os.path.exists(f"{REPORT}/project7_report.pdf"))
for name in NOTEBOOKS + SUPPORT:
    check(f"notebook/{name} present", os.path.exists(f"{NBDIR}/{name}"))

# ---------------------------------------------------------------- 2. the report
print("\n2. Report (over 2 pages scores 0)")
from pypdf import PdfReader  # noqa: E402

pdf = PdfReader(f"{REPORT}/project7_report.pdf")
check("report is exactly 2 pages", len(pdf.pages) == 2, f"{len(pdf.pages)} pages")
tex = io.open(f"{REPORT}/project7_report.tex", encoding="utf-8").read()
for person, sid in (("Karan Rooprai", "n12498122"), ("Nhu Hieu Nguyen", "n12194778")):
    check(f"report names {person} with {sid}", person in tex and sid in tex)
check("report names the group", re.search(r"Group\s+4", tex) is not None)
check("no unfilled template placeholders", not re.search(r"__[A-Z0-9_]+__", tex))
check("no pending hand-written paragraph", "PENDING" not in tex)
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
        subprocess.run([PDFTOTEXT, "-bbox", f"{REPORT}/project7_report.pdf", boxes],
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
      not re.search(r"\.backward\(|\.step\(\)|\.train\(\)|zero_grad|optim\.", main_code))
randn_lines = [line.strip() for line in main_code.split("\n") if re.search(r"randn(_like)?\(",
                                                                               line)]
check("every random draw main_report.ipynb makes comes from a seeded generator",
      all("generator=" in line or line == "noise = torch.randn_like(x0).to(device)"
          for line in randn_lines) and "q_sample(x0, t, schedule, noise)" in main_code,
      f"{len(randn_lines)} draws")
asserts = [line.strip() for line in main_code.split("\n") if line.strip().startswith("assert")]
check("main_report.ipynb asserts nothing about a metric value",
      not any(re.search(r"fd|accuracy|score|loss|spread|floor", line) for line in asserts),
      f"{len(asserts)} asserts")
check("main_report.ipynb loads every checkpoint with weights_only=True",
      main_code.count("weights_only=True") == main_code.count("torch.load(") == 3,
      f"{main_code.count('weights_only=True')} uses")

devices = {n: (re.findall(r"^device\s*:\s*(\S+)", printed[n], re.M) or ["?"])[0]
           for n in NOTEBOOKS}
check("all three notebooks ran on one device", len(set(devices.values())) == 1, f"{devices}")
check(f"the stored outputs come from the {EXPECTED_DEVICE} device",
      all(d.startswith(EXPECTED_DEVICE) for d in devices.values()), f"{devices}")

# ---------------------------------------------------------------- 4. the tutorial and protocol
print("\n4. Tutorial 9.3 followed; the schedule the only difference; test split held back")
solution = json.load(io.open(TUTORIAL, encoding="utf-8"))["cells"]
definitions = {}
for cell in solution:
    if cell["cell_type"] != "code":
        continue
    text = "".join(cell["source"])
    try:
        tree = ast.parse(text)
    except SyntaxError:
        continue
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = ast.get_source_segment(text, node)
for name in CARRIED:
    check(f"{name} is byte-identical to Tutorial 9.3's solution in DDPM_CosineSchedule.ipynb",
          name in definitions and definitions[name] in ddpm_code)
    check(f"{name} is byte-identical to Tutorial 9.3's solution in main_report.ipynb",
          name in definitions and definitions[name] in main_code)
starts = [re.search(rf"^(def|class) {name}\(", ddpm_code, re.M) for name in TUTORIAL_ORDER]
check("the tutorial's functions appear in the tutorial's order",
      all(starts) and [m.start() for m in starts] == sorted(m.start() for m in starts),
      ", ".join(n for n, m in zip(TUTORIAL_ORDER, starts) if not m) or "all present")

# Run the two schedule functions exactly as the notebook defines them, then compare each with an
# independent computation: linspace for the tutorial's, eq. 17 of Nichol and Dhariwal in numpy
# for the cosine one.
namespace = {"torch": torch}
tree = ast.parse(ddpm_code)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("linear_beta_schedule",
                                                           "cosine_beta_schedule"):
        exec(ast.get_source_segment(ddpm_code, node), namespace)
linear = namespace["linear_beta_schedule"](1000).double().numpy()
check("linear_beta_schedule(1000) is linspace(0.0001, 0.02, 1000)",
      np.allclose(linear, np.linspace(1e-4, 0.02, 1000), atol=1e-9))
steps = np.arange(1001) / 1000
f_t = np.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
abar = f_t / f_t[0]
expected = np.minimum(1 - abar[1:] / abar[:-1], 0.999)
cosine = namespace["cosine_beta_schedule"](1000).double().numpy()
check("cosine_beta_schedule(1000) matches Nichol and Dhariwal's eq. 17, recomputed in numpy",
      np.allclose(cosine, expected, rtol=1e-5, atol=1e-7),
      f"largest gap {np.abs(cosine - expected).max():.1e}")
check("cosine betas: first below 1e-4, last clipped at 0.999, all increasing",
      cosine[0] < 1e-4 and abs(cosine[-1] - 0.999) < 1e-6 and np.all(np.diff(cosine) > 0))

expected_history = {"epochs": 5 if SMOKE else 150, "lr": 2e-4, "batch_size": 128,
                    "timesteps": 1000, "n_channels_unet": 32, "seeds": [0, 1, 2],
                    "n_train": 1000 if SMOKE else 60000}
for key, value in expected_history.items():
    check(f"history: {key} is {value}", history.get(key) == value, f"stored {history.get(key)}")
check(f"history: trained on {EXPECTED_DEVICE}", history["device"].startswith(EXPECTED_DEVICE),
      history["device"])
runs = history["runs"]
check("history: one run per schedule and seed, each with a loss and a time per epoch",
      sorted(runs) == sorted(f"{n}_seed_{s}" for n in ("linear", "cosine") for s in (0, 1, 2))
      and all(len(r["loss"]) == len(r["seconds"]) == history["epochs"] for r in runs.values()))
for name in ("linear", "cosine"):
    keys = sorted(torch.load(f"{NBDIR}/ddpm_{name}.pth", map_location="cpu",
                             weights_only=True))
    wanted = sorted([f"seed_{s}" for s in (0, 1, 2)]
                    + [f"epoch_{e}" for e in history["snapshot_epochs"]])
    check(f"ddpm_{name}.pth holds every seed's final model and seed 0's snapshots",
          keys == wanted, ", ".join(keys))
# Same seed, same initial weights: the schedule enters only through q_sample, so epoch 1 of a
# linear run and its cosine partner start from one network and see the same batches.
check("each training run resets the seed before building its model",
      re.search(r"def train_cddpm\(name, schedule, seed\):\n    torch\.manual_seed\(seed\)",
                ddpm_code) is not None)
for name in ("DDPM_CosineSchedule.ipynb", "DigitClassifier.ipynb"):
    source = code_of(notebooks[name])
    uses = [line for line in source.split("\n") if "test_images" in line]
    check(f"{name}: touches the test images only to save the split",
          len(uses) == 1 and "test_split" in uses[0], f"{len(uses)} line(s)")
check("main_report.ipynb reads the test split from its own file",
      'torch.load("mnist_custom_test.pt"' in main_code and '"mnist_custom.pt"' not in main_code)
check("main_report.ipynb takes its settings from the history file",
      'history["n_channels_unet"], history["seeds"]' in main_code)

# ---------------------------------------------------------------- 5. analysis present
print("\n5. main_report ran every analysis the report uses")
COUNTS = ((r"^grid accuracy \| (linear|cosine) \|", 2, "12x10 grid scores"),
          (r"^classifier \| all [\d,]+ real test images", 1, "classifier on real test digits"),
          (r"^floor \| real set A against set B", 1, "real-against-real floor"),
          (r"^quality \| seed \d \| (ddpm|ddim) \| K=", 31, "quality rows (1 + 3 x 2 x 5)"),
          (r"^rule \| (ddpm|ddim) \| K=", 10, "decision-rule rows"),
          (r"^verdict \| (ddpm|ddim) \|", 2, "verdicts"),
          (r"^full steps \|", 1, "full-step comparison"),
          (r"^fewest steps \|", 2, "fewest-steps summaries"),
          (r"^epochs \| epoch", len(history["snapshot_epochs"]) + 1, "epoch rows"),
          (r"^epochs \| within 10%", 1, "epochs to within 10%"),
          (r"^loss \| t=", 20, "per-step errors"),
          (r"^loss \| (linear|cosine) \| mean", 2, "loss summaries"),
          (r"^loss \| matched log SNR (above|below) ", 2, "matched-noise comparisons"),
          (r"^variance \| K=", 5, "posterior-variance follow-up rows"),
          (r"^schedule \| (linear|cosine) \|", 2, "schedule facts"),
          (r'^\s*"winners": \{', 1, "machine-readable summary"))
for pattern, expected, label in COUNTS:
    found = len(re.findall(pattern, main, re.M))
    check(f"main_report printed the {label}", found == expected, f"{found} of {expected}")
others = re.findall(r"^rule \|.*\| other seeds (.+?) \| \w+\s*$", main, re.M)
check("every rule row carries one difference per other seed",
      others and all(len(o.split()) == 2 for o in others))

# ---------------------------------------------------------------- 6. traceability
print("\n6. Traceability: every number in the report is printed by main_report.ipynb")
body = tex.split(r"\begin{document}", 1)[1].split(r"\section*{References}", 1)[0]
body = re.sub(r"(?<!\\)%.*", "", body)
body = re.sub(r"\\includegraphics\[[^]]*\]\{[^}]*\}", "", body)
body = re.sub(r"\\(vspace|setcounter|hfill)\*?\{[^}]*\}(\{[^}]*\})?", "", body)
body = re.sub(r"\\cmidrule\([lr]*\)\{[^}]*\}", "", body)
body = re.sub(r"\[\d+pt\]", "", body)
body = re.sub(r"n\d{8}", "", body)                        # student ids
body = re.sub(r"pp?\.~[\d, ]+", "", body)                  # lecture page references
body = re.sub(r"\\S\d+", "", body)                         # section signs
body = re.sub(r"eq\.~\d+", "", body)                       # equation numbers
body = re.sub(r"\b(19[9]\d|20[0-2]\d)\b(?=\)|;|,)", "", body)   # citation years
flat = re.sub(r"\s+", " ", body)
# Settings the design fixes, and nothing the run measured: the tutorial's T, schedule ends,
# learning rate, batch and channel counts, the cosine offset and clip, the step counts compared,
# the snapshot epochs, the grid's shape, the thresholds of the rule and of "high quality", the
# bootstrap count, the image size, the tutorial's number (9.3), and the prior-work figures quoted
# from Nichol and Dhariwal, Song et al. and Ho et al. (the EMA decay 0.9999, Appendix B), which the
# README records with their source.
DESIGN = {"9.3", "1000", "1,000", "0.02", "0.008", "0.999", "128", "32", "12", "120", "95", "10", "20",
          "50", "100", "250", "25", "150", "60,000", "2", "3", "4", "1", "0", "5",
          "3.37", "3.26", "2.90", "3.05", "17", "0.9999"}
numbers = set(re.findall(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d])|(?<![\w.,])\d+\.\d+|"
                         r"(?<![\w.,])\d{2,}(?![\d.,])", flat))
untraceable = sorted(n for n in numbers - DESIGN if n not in main)
check("no number appears only in the report", not untraceable,
      f"untraceable: {untraceable}" if untraceable else f"{len(numbers)} checked")
check("no Python exponent notation survives into the report",
      not re.search(r"\de[+-]\d", body))
prose = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
ungrouped = re.findall(r"(?<![\d.,])\d{4,}(?![\d,])", prose.replace("1000", ""))
check("every count of a thousand or more is grouped in prose (the step count 1000 aside)",
      not ungrouped, f"ungrouped: {sorted(set(ungrouped))}")

# ---------------------------------------------------------------- 7. the archive
print("\n7. The code archive")
archive = f"{SUBMISSION}/project7_code.zip"
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
    submitted = f"{SUBMISSION}/project7_report.pdf"
    check("submission report is the current build",
          os.path.exists(submitted)
          and hashlib.sha256(io.open(submitted, "rb").read()).hexdigest()
          == hashlib.sha256(io.open(f"{REPORT}/project7_report.pdf", "rb").read()).hexdigest())
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
    check("project7_code.zip built", False, "run tools/build_zip.py")

# ------------------------------------------------- 8. what the deliverables must not contain
print("\n8. Deliverables explain the subject, not the submission")
ASSESSMENT_WORDS = re.compile(
    r"\b(marker|markers|rubric|graded|grades|grading|feedback|deduct\w*)\b", re.I)
SELF_GRADING = re.compile(
    r"rather than (asserted|assumed|eyeballed|claimed|intended|taken on trust)", re.I)
sources = {name: "\n".join("".join(c["source"]) for c in nb["cells"])
           for name, nb in notebooks.items()}
sources["project7_report.tex"] = tex
sources["README.txt"] = readme
for name, source in sources.items():
    hits = ASSESSMENT_WORDS.findall(source) + [m.group(0) for m in SELF_GRADING.finditer(source)]
    check(f"{name} does not talk about how it is marked", not hits, f"found {sorted(set(hits))}")
# The brief's Introduction row names a discriminator, copied from Project 6. This project has
# none, so the word must not reach any deliverable.
for name, source in sources.items():
    check(f"{name} says nothing about a discriminator", "discriminator" not in source.lower())

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

check("the report states the hypothesis verbatim", HYPOTHESIS in flat)
said_epochs = re.search(r"batches of 128 for (\d+) epochs", flat)
check("the report's epoch count is the stored one",
      said_epochs is not None and int(said_epochs.group(1)) == history["epochs"],
      f"report {said_epochs.group(1) if said_epochs else '?'}, stored {history['epochs']}")
check("the report names the seeds that were trained",
      "seeds 0, 1 and 2" in flat.replace("~", " "))
cited = {name: name.split(",")[0] in flat for name in (
    "Ho et al", "Nichol and Dhariwal", "Song et al", "Lin et al", "Heusel et al", "Salvado",
    "Kingma et al")}
references = tex.split(r"\section*{References}", 1)[1]
listed_refs = re.findall(r"\\bib\{([A-Z][a-z]+),", references)
check("every reference listed is cited in the text",
      all(any(author in key for key, ok in cited.items() if ok) for author in listed_refs),
      f"listed {listed_refs}; cited {[k for k, ok in cited.items() if ok]}")

# The Conclusion's verdict must be the rule's: read the rule's outcomes back from main_report.
wins = {w: len(re.findall(rf"^rule \|.*\| {w}\s*$", main, re.M)) for w in ("cosine", "linear")}
conclusion = flat.split(r"\section{Conclusion}", 1)[-1]
supports = bool(re.search(r"The evidence supports the hypothesis", conclusion))
check("the Conclusion's verdict follows the decision rule",
      supports == (wins["cosine"] > 0 and wins["linear"] == 0),
      f"rule: cosine {wins['cosine']}, linear {wins['linear']}; conclusion supports {supports}")
conclusion_text = re.sub(r"\\[A-Za-z]+\*?(\{[^}]*\})?", " ", conclusion)
conclusion_text = re.sub(r"\$[^$]*\$", "x", conclusion_text)
conclusion_text = re.sub(r"(et al|e\.g|i\.e)\.", r"\1", conclusion_text)
n_sentences = len(re.findall(r"[.!?](\s|$)", conclusion_text.strip()))
check("the Conclusion is two or three sentences", 2 <= n_sentences <= 3, f"{n_sentences}")
discussion = flat.split(r"\section{Discussion}", 1)[-1].split(r"\section{Conclusion}", 1)[0]
for word in ("quality", "consisten", "variet", "training time"):
    check(f"the Discussion addresses {word}...", word in discussion.lower())
check("the Discussion makes recommendations from strengths and limitations",
      "Recommendations" in discussion and "Strengths and limitations" in discussion)

# Label sizes at placement: authored size times placement width over the figure's native width.
TEXTWIDTH = 18 / 2.54
COLUMN = (TEXTWIDTH - 0.6 / 2.54) / 2
# The banner's width is read from the report itself, so resizing it cannot slip past this check.
banner = re.search(r"\\includegraphics\[width=([\d.]+)\\textwidth\]\{figures/figure_1_grids\.pdf\}",
                   tex)
placements = {"figure_1_grids.pdf": float(banner.group(1)) * TEXTWIDTH if banner else TEXTWIDTH}
for figure in SUPPORT:
    if figure.startswith("figure_") and figure in tex and figure not in placements:
        placements[figure] = COLUMN
for figure, placed in placements.items():
    path = f"{REPORT}/figures/{figure}"
    if not os.path.exists(path):
        check(f"{figure} present in report/figures", False)
        continue
    native = float(PdfReader(path).pages[0].mediabox.width) / 72
    cell = re.search(rf"savefig\(\"{figure}\"", main_code)
    start = main_code.rfind("plt.subplots", 0, cell.start()) if cell else -1
    block = main_code[start:cell.start()] if start >= 0 else ""
    authored = [float(s) for s in re.findall(r"fontsize=(\d+(?:\.\d+)?)", block)] + [8.0]
    smallest = min(authored)
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
