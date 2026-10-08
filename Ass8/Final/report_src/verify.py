r"""Every check that must pass before Project 8 is uploaded.

Adapted from Project 7's tools/verify.py. The cliff edges the unit states are asserted
mechanically: a report over 2 pages scores 0, wrong file names score 0, and the brief says of
main_report.ipynb that "This notebook will be run for grading; if any cell fails, the code
evaluation will receive 0 marks." A cliff check never skips: a missing tool is itself a failure, so
the script cannot finish with ALL CHECKS PASSED while a rail has not run.

Run:  python tools/verify.py
      P8_NBDIR=<dir> P8_REPORT=<dir> P8_SUBMISSION=<dir> python tools/verify.py   (another run)
"""
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import warnings
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-11/project-8-3d-movie")
STARTER = f"{BASE}/data/IFN680_Week10_Assessment_StarterCode.ipynb"
DRAFT = f"{BASE}/draft/project8_code.zip"
NBDIR = os.environ.get("P8_NBDIR", f"{BASE}/notebook")
REPORT = os.environ.get("P8_REPORT", f"{BASE}/report")
SUBMISSION = os.environ.get("P8_SUBMISSION", f"{BASE}/submission")
TEXBIN = "C:/Users/Admin/.conda/envs/tex/Library/bin"
EXPECTED_DEVICE = os.environ.get("P8_DEVICE", "cuda")

NOTEBOOKS = ("TinyNeRF.ipynb", "ExtendedNeRF.ipynb", "main_report.ipynb")
WEIGHTS = ("tinynerf.pth", "extended_nerf.pth", "ablation_no_viewdirs.pth", "ablation_uniform96.pth")
SUPPORT = WEIGHTS + ("training_history.json", "tiny_nerf_data.npz")
FIGURES = ("figure_1_test_views.pdf", "figure_2_novel_views.pdf", "figure_4_view_dependence.pdf",
           "figure_5_sampling.pdf", "figure_3_curves.pdf")
# The starter's code that the brief calls provided, or that the pipeline must keep.
PROVIDED = ("get_rays", "NeRFDataset", "sample_pdf")
HEADINGS = ("Introduction", "Implemented changes", "Results", "Discussion", "Conclusion")

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


def definitions(cells):
    """Top-level functions and classes of a notebook, as syntax trees without docstrings."""
    found = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        text = "\n".join(line for line in "".join(cell["source"]).split("\n")
                         if not line.lstrip().startswith(("%", "!")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                for inner in ast.walk(node):
                    body = getattr(inner, "body", None)
                    if (isinstance(inner, (ast.FunctionDef, ast.ClassDef)) and body
                            and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)):
                        inner.body = body[1:] or [ast.Pass()]
                found[node.name] = ast.dump(node)
    return found


notebooks = {name: load(name) for name in NOTEBOOKS}
printed = {name: stream_text(nb) for name, nb in notebooks.items()}
main = printed["main_report.ipynb"]
main_code = code_of(notebooks["main_report.ipynb"])
ext_code = code_of(notebooks["ExtendedNeRF.ipynb"])

# ---------------------------------------------------------------- 1. names are pass/fail
print("\n1. File naming (wrong names score 0)")
check("report is named project8_report.pdf", os.path.exists(f"{REPORT}/project8_report.pdf"))
for name in NOTEBOOKS + SUPPORT:
    check(f"notebook/{name} present", os.path.exists(f"{NBDIR}/{name}"))

# ---------------------------------------------------------------- 2. the report
print("\n2. Report (over 2 pages scores 0)")
from pypdf import PdfReader  # noqa: E402

pdf = PdfReader(f"{REPORT}/project8_report.pdf")
check("report is exactly 2 pages", len(pdf.pages) == 2, f"{len(pdf.pages)} pages")
box = pdf.pages[0].mediabox
check("report is A4", abs(float(box.width) - 595.28) < 1 and abs(float(box.height) - 841.89) < 1)
tex = io.open(f"{REPORT}/project8_report.tex", encoding="utf-8").read()
for person, sid in (("Karan Rooprai", "n12498122"), ("Nhu Hieu Nguyen", "n12194778")):
    check(f"report names {person} with {sid}", person in tex and sid in tex)
check("report names the group", re.search(r"Group\s+4", tex) is not None)
check("no unfilled template placeholders", not re.search(r"__[A-Z0-9_]+__", tex))
positions = [tex.find(f"\\section{{{h}}}") for h in HEADINGS]
check("the report's sections are headings, in order",
      all(p >= 0 for p in positions) and positions == sorted(positions),
      ", ".join(f"{h} {'ok' if p >= 0 else 'MISSING'}" for h, p in zip(HEADINGS, positions)))
check("a reference list follows the Conclusion",
      tex.find(r"\section*{References}") > positions[-1] >= 0)
log = f"{REPORT}/project8_report.log"
overfull = len(re.findall(r"Overfull \\hbox", io.open(log, encoding="utf-8", errors="replace").read())) \
    if os.path.exists(log) else -1
check("the LaTeX log has no overfull box", overfull == 0, f"{overfull} overfull")

PDFTOTEXT = f"{TEXBIN}/pdftotext.exe"
if os.path.exists(PDFTOTEXT):
    LEFT, RIGHT = 42.52, 552.76      # 1.5cm margins on a 595.28pt A4 page
    SLACK = 3.0                      # microtype protrusion plus glyph side bearings
    with tempfile.TemporaryDirectory() as tmp:
        boxes = f"{tmp}/bbox.xml"
        subprocess.run([PDFTOTEXT, "-bbox", f"{REPORT}/project8_report.pdf", boxes],
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
devices = {}
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
    check(f"{name}: comment lines at least 10% of code lines", len(comments) / len(lines) >= 0.10,
          f"{len(comments)}/{len(lines)} = {len(comments) / len(lines):.0%}")
    check(f"{name}: no tqdm progress bar (it writes to stderr)", "tqdm(" not in source)
    check(f"{name}: no TODO left", "TODO" not in source)
    prev = None
    no_lead = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and prev == "code":
            no_lead.append(cell["id"])
        prev = cell["cell_type"]
    check(f"{name}: every code cell has a markdown lead-in", not no_lead, f"{no_lead}")
    for dash, label in (("\u2014", "em"), ("\u2013", "en")):
        check(f"{name}: no {label} dash", dash not in json.dumps(notebook, ensure_ascii=False))
    check(f"{name}: nbformat 4.5 with an id on every cell",
          (notebook["nbformat"], notebook["nbformat_minor"]) >= (4, 5)
          and all(c.get("id") for c in notebook["cells"]))
    echoes = [c["id"] for c in notebook["cells"]
              for o in c.get("outputs", []) if o.get("output_type") == "execute_result"
              and "interactive(" not in "".join(o.get("data", {}).get("text/plain", ""))]
    check(f"{name}: echoes no object repr", not echoes, f"cells {echoes}")
    check(f"{name}: no f-string needing Python 3.12 (PEP 701)",
          not re.search(r"""f"[^"\n]*\{[^}"\n]*"[^"\n]*"[^}\n]*\}""", source)
          and not re.search(r"""f'[^'\n]*\{[^}'\n]*'[^'\n]*'[^}\n]*\}""", source))
    device = re.search(r"Using device: (\w+)", printed[name])
    devices[name] = device.group(1) if device else "?"
check("all three notebooks ran on one device", len(set(devices.values())) == 1, f"{devices}")
check(f"the stored outputs come from the {EXPECTED_DEVICE} device",
      set(devices.values()) == {EXPECTED_DEVICE}, f"{devices}")
check("main_report.ipynb trains nothing",
      not re.search(r"\.backward\(|\.step\(\)|\.train\(\)|zero_grad|optim\.", main_code))
check("main_report.ipynb loads every checkpoint with weights_only=True",
      main_code.count("weights_only=True") == main_code.count("torch.load(") == 1,
      "one load() helper, used for the four checkpoints")
check("main_report.ipynb loads all four checkpoints", all(w in main_code for w in WEIGHTS))

# ---------------------------------------------------------------- 4. the starter and the tasks
print("\n4. The starter's pipeline is kept, and both tasks are implemented")
starter = definitions(json.load(io.open(STARTER, encoding="utf-8"))["cells"])
for name in NOTEBOOKS:
    ours = definitions(notebooks[name]["cells"])
    for fn in PROVIDED:
        if fn in ours:
            check(f"{name}: {fn} is the starter's, unchanged", ours[fn] == starter[fn])
check("ExtendedNeRF.ipynb defines get_rays, NeRFDataset and sample_pdf",
      all(fn in definitions(notebooks["ExtendedNeRF.ipynb"]["cells"]) for fn in PROVIDED))
for name in NOTEBOOKS:
    check(f"{name}: test split is images[100:], as the starter makes it",
          "test_images, test_poses = images[100:], poses[100:]" in code_of(notebooks[name]))
check("Task 1: NeRF.forward is implemented (no starter placeholder left)",
      "sigma,rgb = None, None" not in ext_code and "def forward(self, x, dir)" in ext_code)
check("Task 1: the density is computed before the direction enters",
      ext_code.find("sigma = F.relu(self.sigma_layer(h))") < ext_code.find("self.direction_encoding(dir)"))
check("Task 2: render_rays runs a coarse pass, sample_pdf and a fine pass",
      all(s in ext_code for s in ("def render_rays(coarse_network, fine_network",
                                  "sample_pdf(t_mid, weights_c[..., 1:-1], N_f", "fine_network, rays_o")))
check("Task 2: the fine depths are detached", "N_f, rand=rand).detach()" in ext_code)
ext_sanity = printed["ExtendedNeRF.ipynb"]
check("ExtendedNeRF's sanity check: density identical across directions, colour not",
      "identical for two directions: True" in ext_sanity and "differs between directions:   True" in ext_sanity)

# The training notebooks were edited without re-running: their code must match the draft's.
with zipfile.ZipFile(DRAFT) as draft_zip:
    for name in ("TinyNeRF.ipynb", "ExtendedNeRF.ipynb"):
        draft = json.loads(draft_zip.read(f"project8_code/{name}"))
        check(f"{name}: every function and class matches the trained draft",
              definitions(draft["cells"]) == definitions(notebooks[name]["cells"]))
        draft_out = [o for c in draft["cells"] if c["cell_type"] == "code"
                     for o in c.get("outputs", []) if o.get("name") != "stderr"]
        ours_out = [o for c in notebooks[name]["cells"] if c["cell_type"] == "code"
                    for o in c.get("outputs", [])]
        check(f"{name}: the stored outputs are the draft's GPU outputs", draft_out == ours_out,
              f"{len(ours_out)} outputs")

# ---------------------------------------------------------------- 5. the numbers that matter
print("\n5. The numbers the brief names")
summary = dict(re.findall(r"^summary \| ([^|]+?) \| PSNR ([\d.]+)", main, re.M))
check("main_report prints a summary line for each of the four models", len(summary) == 4,
      f"{sorted(summary)}")
ext = float(summary.get("Extended NeRF", 0))
check("Extended NeRF reaches an average test PSNR of at least 28 dB", ext >= 28, f"{ext:.2f} dB")
final = re.search(r"\[ExtendedNeRF\] epoch 100/100 \| train PSNR\s+[\d.]+ \| test PSNR\s+([\d.]+)",
                  printed["ExtendedNeRF.ipynb"])
check("main_report's Extended NeRF PSNR is the training notebook's final-epoch value",
      final is not None and abs(float(final.group(1)) - ext) < 0.01,
      f"training {final.group(1) if final else '?'}, main_report {ext:.2f}")
for label, pattern in (("paired view-by-view differences", r"^paired \|"),
                       ("the frozen-direction test", r"^frozen direction \|"),
                       ("the photo-consistency test", r"^photo consistency \|"),
                       ("the sample-budget sweep", r"^equal work \|"),
                       ("the turntable", r"^turntable \|")):
    check(f"main_report prints {label}", re.search(pattern, main, re.M) is not None)

# ---------------------------------------------------------------- 6. traceability
print("\n6. Traceability: every number in the report is printed by main_report.ipynb")
body = tex.split(r"\begin{document}", 1)[1].split(r"\section*{References}", 1)[0]
body = re.sub(r"(?<!\\)%.*", "", body)
body = re.sub(r"\\includegraphics\[[^]]*\]\{[^}]*\}", "", body)
body = re.sub(r"\\(vspace|setcounter|hfill|hspace)\*?\{[^}]*\}(\{[^}]*\})?", "", body)
body = re.sub(r"\\cmidrule\([lr]*\)\{[^}]*\}", "", body)
body = re.sub(r"\[-?[\d.]+pt\]", "", body)
body = re.sub(r"n\d{8}", "", body)                        # student ids
body = re.sub(r"pp?\.~[\d, ]+", "", body)                  # lecture page references
body = re.sub(r"\b(19[9]\d|20[0-2]\d)\b(?=\)|;|,)", "", body)   # citation years
flat = re.sub(r"\s+", " ", body)
# Settings the design fixes and the prior work quoted, nothing the run measured: the tutorial and
# starter settings (near 2, far 6, 64 samples, N_c 32, N_f 64, 96 and 128 samples, L = 6 and 4,
# 39 and 27 inputs, 128-unit layers, 1,024-ray batches, 100 epochs, the learning-rate ends, views
# 100 to 105 of 106 at 100 x 100), the 28 dB the brief expects, the budgets swept (24 to 256),
# the thresholds of the photo test (90th percentile, 10 cameras), the tutorial's number (10.4),
# and NeRF's Table 2 (31.01, 27.66, 30.06, the differences 3.35 and 0.95, 64 + 128 and 256
# samples), which the README records with their source.
DESIGN = {"10.4", "2", "6", "64", "32", "96", "128", "39", "27", "100", "1,024", "105", "106",
          "28", "25", "24", "48", "192", "256", "90", "10", "4", "3", "1", "0", "5", "8", "40", "45",
          "30", "12", "31.01", "27.66", "30.06", "3.35", "0.95", "0.5"}
numbers = set(re.findall(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d])|(?<![\w.,])\d+\.\d+|"
                         r"(?<![\w.,])\d{2,}(?![\d.,])", flat))
untraceable = sorted(n for n in numbers - DESIGN if n not in main)
check("no number appears only in the report", not untraceable,
      f"untraceable: {untraceable}" if untraceable else f"{len(numbers)} checked")
check("no Python exponent notation survives into the report", not re.search(r"\de[+-]\d", body))
prose = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
ungrouped = re.findall(r"(?<![\d.,])\d{5,}(?![\d,])", prose)
check("every count of ten thousand or more is grouped", not ungrouped, f"{sorted(set(ungrouped))}")

# ---------------------------------------------------------------- 7. the archive
print("\n7. The code archive")
archive = f"{SUBMISSION}/project8_code.zip"
readme = ""
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
        for name in FIGURES:
            check(f"zip contains {name}", name in names)
        for name in names:
            if name != "README.txt":
                same = (os.path.exists(f"{NBDIR}/{name}")
                        and hashlib.sha256(zf.read(name)).hexdigest()
                        == hashlib.sha256(io.open(f"{NBDIR}/{name}", "rb").read()).hexdigest())
                check(f"zip's {name} is the current notebook/ file", same)
        readme = zf.read("README.txt").decode("utf-8") if "README.txt" in names else ""
    submitted = f"{SUBMISSION}/project8_report.pdf"
    check("submission report is the current build",
          os.path.exists(submitted)
          and hashlib.sha256(io.open(submitted, "rb").read()).hexdigest()
          == hashlib.sha256(io.open(f"{REPORT}/project8_report.pdf", "rb").read()).hexdigest())
    check("archive is under 50 MB", os.path.getsize(archive) < 50 * 1024 * 1024,
          f"{os.path.getsize(archive) / 1024 / 1024:.1f} MB")
    FILENAME = re.compile(r"[A-Za-z0-9_.*-]+\.(?:ipynb|pth|json|npz|pdf|txt|gif)\b")
    MADE = set(FIGURES) | {"nerf_turntable.gif", "results_summary.json", "training_log.txt"}
    for name in NOTEBOOKS:
        text = "\n".join("".join(c["source"]) for c in notebooks[name]["cells"])
        named = {n for n in FILENAME.findall(text) if "*" not in n}
        dangling = sorted(n for n in named if n not in names and n not in MADE)
        check(f"{name} names only files the archive ships or makes", not dangling,
              f"missing {dangling}")
else:
    check("project8_code.zip built", False, "run tools/build_zip.py")

# ------------------------------------------------- 8. what the deliverables must not contain
print("\n8. Deliverables explain the subject, not the submission")
ASSESSMENT_WORDS = re.compile(
    r"\b(marker|markers|rubric|graded|grades|grading|feedback|deduct\w*)\b", re.I)
SELF_GRADING = re.compile(
    r"rather than (asserted|assumed|eyeballed|claimed|intended|taken on trust)", re.I)
sources = {name: "\n".join("".join(c["source"]) for c in nb["cells"])
           for name, nb in notebooks.items()}
sources["project8_report.tex"] = tex
sources["README.txt"] = readme
for name, source in sources.items():
    hits = ASSESSMENT_WORDS.findall(source) + [m.group(0) for m in SELF_GRADING.finditer(source)]
    check(f"{name} does not talk about how it is marked", not hits, f"found {sorted(set(hits))}")

# ---------------------------------------------------------------- 9. report-specific
print("\n9. What the report says, against what the run did")
floats = []
for kind in ("figure", "table"):
    for n, _ in enumerate(re.finditer(r"\\begin\{" + kind + r"\*?\}", tex), start=1):
        floats.append((kind.capitalize(), n))
if r"\twocolumn[" in tex and "Figure 1:" in tex:
    floats = [(k, n + 1 if k == "Figure" else n) for k, n in floats] + [("Figure", 1)]
prose_flat = re.sub(r"\s+", " ", prose)
uncited = [f"{k} {n}" for k, n in floats if f"{k}~{n}" not in prose_flat]
check("every numbered float is cited in the prose", not uncited,
      f"uncited: {uncited}" if uncited else ", ".join(f"{k} {n}" for k, n in sorted(floats)))
first_cite = {n: prose_flat.find(f"Figure~{n}") for k, n in floats if k == "Figure"}
check("figures are first cited in numerical order",
      [n for n, _ in sorted(first_cite.items(), key=lambda kv: kv[1])] == sorted(first_cite),
      f"{first_cite}")
sentences = re.sub(r"\$[^$]*\$", " x ", prose)
sentences = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?(?:\{[^{}]*\})?", " ", sentences)
opens_numeric = []
for para in re.split(r"\n\s*\n", sentences):
    para = re.sub(r"\s+", " ", para).strip()
    if para and re.match(r"\d", para):
        opens_numeric.append(para[:40])
    opens_numeric += [m.group(1) for m in re.finditer(r"[.!?]\s+(\d[\d.,]*)", para)]
check("no sentence in the report opens with a numeral", not opens_numeric, f"{opens_numeric}")
cited = {key: key in flat for key in ("Mildenhall et al", "Wang et al", "Zhang et al", "Salvado")}
references = tex.split(r"\section*{References}", 1)[1]
listed = re.findall(r"\\bib\{([A-Z][a-z]+),", references)
check("every reference listed is cited in the text",
      all(any(author in key for key, ok in cited.items() if ok) for author in listed),
      f"listed {listed}; cited {[k for k, ok in cited.items() if ok]}")
conclusion = flat.split(r"\section{Conclusion}", 1)[-1]
conclusion_text = re.sub(r"\\[A-Za-z]+\*?(\{[^}]*\})?", " ", conclusion)
conclusion_text = re.sub(r"\$[^$]*\$", "x", conclusion_text)
conclusion_text = re.sub(r"(et al|e\.g|i\.e)\.", r"\1", conclusion_text)
n_sentences = len(re.findall(r"[.!?](\s|$)", conclusion_text.strip()))
check("the Conclusion is two or three sentences", 2 <= n_sentences <= 3, f"{n_sentences}")
discussion = flat.split(r"\section{Discussion}", 1)[-1].split(r"\section{Conclusion}", 1)[0]
for phrase in ("rendering quality", "view-dependent appearance", "overall performance",
               "Strengths and limitations", "Recommendations"):
    check(f"the Discussion has a part on {phrase}", phrase.lower() in discussion.lower())
check("the report gives the 28 dB the brief expects and the measured mean",
      "28~dB" in flat.replace(" dB", "~dB") and f"{ext:.2f}" in flat)

# Label sizes at placement: authored size times placement width over the figure's native width.
TEXTWIDTH = 18 / 2.54
COLUMN = (TEXTWIDTH - 0.6 / 2.54) / 2
for figure in FIGURES:
    if figure not in tex:
        continue
    width = re.search(r"\\includegraphics\[width=([\d.]*)\\(textwidth|columnwidth|linewidth)\]\{figures/"
                      + re.escape(figure) + r"\}", tex)
    if not width:
        check(f"{figure}: placement width found in the .tex", False)
        continue
    scale = float(width.group(1) or 1)
    placed = scale * (TEXTWIDTH if width.group(2) == "textwidth" else COLUMN)
    path = f"{REPORT}/figures/{figure}"
    if not os.path.exists(path):
        check(f"{figure} present in report/figures", False)
        continue
    native = float(PdfReader(path).pages[0].mediabox.width) / 72
    smallest = 6.0                    # main_report sets every label to 6 pt or more (rcParams)
    check(f"{figure}: no label renders below 5.5 pt", smallest * placed / native >= 5.5,
          f"6 pt at {native:.2f} in placed at {placed:.2f} in renders {smallest * placed / native:.1f} pt")
    same = (hashlib.sha256(io.open(path, "rb").read()).hexdigest()
            == hashlib.sha256(io.open(f"{NBDIR}/{figure}", "rb").read()).hexdigest()) \
        if os.path.exists(f"{NBDIR}/{figure}") else False
    check(f"{figure} is the file main_report.ipynb wrote", same)

# Every function main_report defines is used.
tree = ast.parse(main_code)
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)} - {"forward", "__init__"}
referenced = ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
              | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})
check("every function main_report.ipynb defines is used", not (defined - referenced),
      f"unused: {sorted(defined - referenced)}")

print("\n" + ("ALL CHECKS PASSED" if not failures
             else f"{len(failures)} CHECK(S) FAILED: " + "; ".join(failures)))
raise SystemExit(1 if failures else 0)
