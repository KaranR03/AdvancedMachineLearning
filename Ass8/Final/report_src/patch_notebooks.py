r"""Build notebook/{TinyNeRF,ExtendedNeRF,main_report}.ipynb from the group draft.

The draft (draft/project8_code.zip, Karan Rooprai, 7 Oct 2026) holds the trained models and the
GPU outputs of the two training notebooks. Retraining would change every number, so the training
notebooks are edited, never re-run:

* markdown cells are added or rewritten (lead-ins, explanations, two explanatory figures);
* the only code change is the NeRF docstring becoming a raw string, which removes a
  SyntaxWarning. Python parses both spellings to the same string, so every code cell is checked
  to have an identical abstract syntax tree to the draft before its outputs are carried over,
  and the one stderr stream that the warning produced is dropped.

main_report.ipynb only evaluates saved models, so it is rebuilt here with its outputs cleared and
then executed (locally for development, on the Hub GPU node for the copy that ships).

Cell ids are kept from the draft; new cells get fixed ids starting with "p8", so a later
rebuild maps every cell to the same place.

Run:  python tools/patch_notebooks.py
"""
import ast
import base64
import copy
import io
import json
import warnings
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-11/project-8-3d-movie")
DRAFT = f"{BASE}/draft/project8_code.zip"
OUT = f"{BASE}/notebook"
FIGURES = f"{BASE}/tools/explanatory"


# --------------------------------------------------------------------------- helpers
def load(name):
    with zipfile.ZipFile(DRAFT) as archive:
        return json.loads(archive.read(f"project8_code/{name}.ipynb"))


def lines(text):
    """Notebook source format: a list of lines that keep their newline, except the last."""
    text = text.strip("\n")
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": lines(text)}


def code(cell_id, text):
    return {"cell_type": "code", "id": cell_id, "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines(text)}


def figure(name, alt):
    with open(f"{FIGURES}/{name}", "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"![{alt}](data:image/png;base64,{encoded})"


def index(nb, cell_id):
    for k, cell in enumerate(nb["cells"]):
        if cell["id"] == cell_id:
            return k
    raise KeyError(cell_id)


def set_markdown(nb, cell_id, text):
    cell = nb["cells"][index(nb, cell_id)]
    assert cell["cell_type"] == "markdown", cell_id
    cell["source"] = lines(text)


def set_code(nb, cell_id, text):
    cell = nb["cells"][index(nb, cell_id)]
    assert cell["cell_type"] == "code", cell_id
    cell["source"] = lines(text)


def insert_before(nb, cell_id, *cells):
    k = index(nb, cell_id)
    nb["cells"][k:k] = list(cells)


def insert_after(nb, cell_id, *cells):
    k = index(nb, cell_id) + 1
    nb["cells"][k:k] = list(cells)


def syntax(source):
    """Normalised syntax tree of a code cell; IPython magics are blanked first."""
    text = "".join(source)
    text = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                     for line in text.split("\n"))
    with warnings.catch_warnings():          # the draft's own source still has the bad escape
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.dump(ast.parse(text))


def check_code_unchanged(draft, patched):
    """Every code cell of a training notebook must parse to the draft's tree, cell for cell."""
    old = [c for c in draft["cells"] if c["cell_type"] == "code"]
    new = [c for c in patched["cells"] if c["cell_type"] == "code"]
    assert [c["id"] for c in old] == [c["id"] for c in new], "code cells added, removed or moved"
    for a, b in zip(old, new):
        assert syntax(a["source"]) == syntax(b["source"]), f"code changed in cell {a['id']}"
        assert a["outputs"] == b["outputs"] or a["id"] in WARNING_CELLS, a["id"]


def raw_docstring(nb, cell_id):
    """Turn the NeRF docstring into a raw string: same value, no invalid-escape warning."""
    cell = nb["cells"][index(nb, cell_id)]
    text = "".join(cell["source"])
    assert text.count('    """\n    TASK 1.') == 1, cell_id
    cell["source"] = lines(text.replace('    """\n    TASK 1.', '    r"""\n    TASK 1.', 1))


def drop_syntax_warning(nb, cell_id):
    cell = nb["cells"][index(nb, cell_id)]
    before = len(cell["outputs"])
    cell["outputs"] = [o for o in cell["outputs"]
                       if not (o.get("name") == "stderr"
                               and "SyntaxWarning" in "".join(o.get("text", [])))]
    assert len(cell["outputs"]) == before - 1, f"expected one warning stream in {cell_id}"


def save(nb, name):
    path = f"{OUT}/{name}.ipynb"
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(nb, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {path}")


WARNING_CELLS = {"b31a3dc0"}          # ExtendedNeRF's NeRF class, whose warning is dropped

# --------------------------------------------------------------------------- shared markdown
DATA_MD = """
# Data - Batching Rays
The synthetic scene `tiny_nerf_data.npz` holds 106 posed 100x100 RGB images of a Lego
bulldozer and the camera's focal length. Each pose is a 4x4 camera-to-world matrix: its first
three columns turn camera axes into world axes and its last column is the camera centre. As in
the starter code, views 0-99 train the model and views 100-105 are held out for testing, so every
reported metric comes from cameras the model never saw.
"""

VIEWS_MD = """
The six training views on top and the six held-out test views underneath, to show what the model
has to reproduce: the cameras circle the object at different heights.
"""

RAYS_MD = """
Each pixel defines a ray $\\mathbf{r}(t) = \\mathbf{o} + t\\mathbf{d}$ from the camera centre
$\\mathbf{o}$ through the pixel. `get_rays` builds the direction in camera coordinates from the
pinhole model, $((i - c_x)/f,\\; -(j - c_y)/f,\\; -1)$ (the camera looks down its own $-z$ axis
and image rows grow downwards), then rotates it into world coordinates with the pose.
"""

DATASET_MD = """
All 100 x 100 x 100 = 1,000,000 training rays are pooled and served in shuffled batches of 1,024,
so one gradient step mixes pixels from many cameras rather than one image at a time.
"""

ENCODING_MD = """
# Model - Neural Scene Representation
**Positional encoding.** A small MLP fed raw coordinates learns only smooth functions, so sharp
edges and fine texture come out blurred. Lifting each coordinate to
$\\gamma(p) = (p, \\sin(2^0\\pi p), \\cos(2^0\\pi p), \\dots, \\sin(2^{L-1}\\pi p), \\cos(2^{L-1}\\pi p))$
gives the network inputs that already vary quickly, so detail costs it far less. With $L = 6$ a 3D
point becomes $3 + 3 \\cdot 2 \\cdot 6 = 39$ numbers.
"""

RENDERING_MD = """
# Inference - Volume Rendering
A pixel's colour is the colour of the ray's samples, each weighted by how much light it sends to
the camera:
$C(\\mathbf{r}) \\approx \\sum_i w_i \\mathbf{c}_i$ with $w_i = T_i\\alpha_i$,
$\\alpha_i = 1 - e^{-\\sigma_i\\delta_i}$ (the chance the ray stops in sample $i$'s interval of
length $\\delta_i$) and $T_i = \\prod_{j<i}(1-\\alpha_j)$ (the chance it got that far). The weights
$w_i$ therefore show where along the ray the visible surface is, which Task 2 uses.
"""

HELPERS_MD = """
## Evaluation helpers
**PSNR** is $-10\\log_{10}(\\mathrm{MSE})$ for images in $[0, 1]$ (the tutorial's metric): every
halving of the squared error adds about 3 dB. **SSIM** (Wang et al., 2004) compares local means,
contrast and structure in 11 x 11 Gaussian windows, so it rewards sharp edges that PSNR barely
sees. `render_image` renders a whole test view in chunks of rays, which saves memory without
changing the result.
"""

TRAINING_MD = """
# NeRF Training Procedure
Each iteration (1) takes a batch of rays, (2) renders them, (3) measures the mean squared error to
the true pixel colours and (4) back-propagates. Every model in this project uses this same loop,
the same seed and batches, Adam with the learning rate decaying from $5\\times10^{-4}$ to
$5\\times10^{-5}$ (as in the original NeRF), and the same number of epochs, so differences between
models in `main_report.ipynb` come from the models and not from the training. After every epoch
the six test views are rendered and their PSNR is logged; the saved weights are those of the last
epoch, not the best one, so the test views never choose the model.
"""

BUDGET_MD = """
The training budget, shared by all four models in this project.
"""

SAVE_MD = """
Save the trained weights and this model's per-epoch history for `main_report.ipynb`, which reloads
them instead of training again.
"""

VIEWER_MD = """
The tutorial's interactive viewer: sliders move the camera over a sphere around the object, which
needs a live kernel with `ipywidgets` (a stored notebook shows a placeholder instead).
"""


# --------------------------------------------------------------------------- TinyNeRF.ipynb
def tiny():
    draft = load("TinyNeRF")
    nb = copy.deepcopy(draft)
    set_markdown(nb, "2fe77e0f", DATA_MD)
    insert_before(nb, "1d51ceba", md("p8t00001", VIEWS_MD))
    set_markdown(nb, "1195b29a", RAYS_MD)
    set_markdown(nb, "dc1ea685", DATASET_MD)
    set_markdown(nb, "ad199c28", ENCODING_MD)
    set_markdown(nb, "7b7c6e18", """
The Tiny NeRF MLP has **no view dependence**: colour is a function of position only, so every 3D
point has one colour whichever camera looks at it (a matte renderer). Three layers map the 39
encoded numbers to 128, 128 and finally 4 outputs: density $\\sigma$ (kept non-negative by a
ReLU) and colour (kept in $[0, 1]$ by a sigmoid).
""")
    set_markdown(nb, "d1e23a79", RENDERING_MD)
    set_markdown(nb, "7b683e41", """
Tiny NeRF places `N_SAMPLES_TINY = 64` samples evenly between `near = 2` and `far = 6` on every ray.
During training each sample is jittered inside its bin, so over many steps the network sees every
depth, not only 64 fixed ones.
""")
    set_markdown(nb, "129089a6", HELPERS_MD)
    set_markdown(nb, "d90b3870", TRAINING_MD)
    insert_before(nb, "31bda4e8", md("p8t00002", BUDGET_MD))
    insert_before(nb, "03970477", md("p8t00003", """
Build the tutorial's Tiny NeRF and train it with the shared loop. Its renderer is the tutorial's:
one network, 64 uniform samples per ray.
"""))
    insert_before(nb, "de21ce1d", md("p8t00004", SAVE_MD))
    insert_before(nb, "c1d60056", md("p8t00005", VIEWER_MD))
    check_code_unchanged(draft, nb)
    save(nb, "TinyNeRF")


# --------------------------------------------------------------------------- ExtendedNeRF.ipynb
def extended():
    draft = load("ExtendedNeRF")
    nb = copy.deepcopy(draft)
    set_markdown(nb, "57fd5f87", DATA_MD)
    insert_before(nb, "39a624df", md("p8e00001", VIEWS_MD))
    set_markdown(nb, "8d59950b", RAYS_MD)
    set_markdown(nb, "64cc3e8d", DATASET_MD)
    set_markdown(nb, "b8fccf15", ENCODING_MD)
    set_markdown(nb, "d2be44a6", f"""
## Task 1: view-dependent radiance field
Tiny NeRF gives each 3D point one colour, so it cannot show anything that changes with the camera,
such as a highlight that slides across a glossy surface. The upgraded network is
$F_\\theta: (\\gamma(\\mathbf{{x}}), \\gamma(\\mathbf{{d}})) \\mapsto (\\sigma, \\mathbf{{c}})$, wired so
that **only the colour can depend on the viewing direction** $\\mathbf{{d}}$:

{figure("nerf_architecture.png", "Task 1 network: the density head reads the position trunk only; the colour head also reads the encoded viewing direction")}

* The density $\\sigma$ is read from the position trunk **before** the direction enters, so the
  geometry is the same from every camera. If density could see the direction, the network could
  explain each photograph with a different shape and still fit it, and the scene would no longer
  be one consistent 3D object.
* `feature_layer` passes 128 position features to the colour head with no activation, as in the
  original NeRF, and the colour head concatenates them with $\\gamma(\\mathbf{{d}})$.
* The direction uses $L_{{dir}} = 4$ frequencies (27 numbers) against $L = 6$ for position, because
  colour changes slowly with the viewing angle compared with how fast it changes across the surface.
* `view_dependent=False` replaces the direction by zeros. It exists only for ablation A below: the
  same network, the same size, but matte again.
""")
    raw_docstring(nb, "b31a3dc0")
    drop_syntax_warning(nb, "b31a3dc0")
    set_markdown(nb, "27a28d91", RENDERING_MD)
    set_markdown(nb, "b1b5393f", f"""
## Task 2: hierarchical sampling with coarse and fine networks
With 64 evenly spaced samples most of the network's work lands in empty air in front of or behind
the object. Hierarchical sampling spends the samples where the surface is:

{figure("hierarchical_sampling.png", "Task 2 on one illustrative ray: coarse weights, the CDF that sample_pdf inverts, and where the fine samples land")}

1. **Coarse pass**: $N_c = 32$ stratified samples, evaluated by the coarse network. Their weights
   $w_i = T_i\\alpha_i$ show where the surface is likely to be.
2. **Fine pass**: the coarse weights between neighbouring samples become a probability
   distribution along the ray. `sample_pdf` (provided by the starter, unchanged) draws $N_f = 64$
   depths from it by inverse-transform sampling: a value $u$ in $[0, 1]$ is read across to the
   cumulative distribution and down to the depth where it is reached. The fine network then
   evaluates all $N_c + N_f = 96$ depths, sorted, so the coarse samples still cover the whole ray.
3. **Both passes are supervised**,
   $\\mathcal{{L}} = \\lVert C^c - I\\rVert^2 + \\lVert C^f - I\\rVert^2$: without the coarse term the
   coarse network would get no gradient (the sampling is detached) and would never learn where the
   surface is.

The two end weights are dropped because the PDF needs one value per bin between neighbouring
samples ($N_c - 1$ bins, $N_c - 2$ interior weights). In the deterministic branch the draws include
$u = 0$ and $u = 1$, which land at the two ends of the ray, as the figure shows.
""")
    insert_before(nb, "135b236d", md("p8e00002", """
`render_rays` is Task 2's implementation. `stratified_t_vals` is the tutorial's sampling (evenly
spaced, jittered during training), factored out so that the coarse pass and ablation B share it.
`query_and_render` evaluates one network on the points of a ray, giving it the unit ray direction
as the viewing direction, and volume-renders them. The fine depths are `detach`ed: they say where
to look, and no gradient should flow back through that choice.
"""))
    set_markdown(nb, "b42b3aa0", """
### Sanity checks
Before training: the output shapes are right, and the density really ignores the viewing direction
while the colour does not (the property Task 1 is built around).
""")
    set_markdown(nb, "341535be", HELPERS_MD)
    set_markdown(nb, "eb7e2c2c", TRAINING_MD)
    insert_before(nb, "8692ef00", md("p8e00003", BUDGET_MD))
    insert_before(nb, "f8a8e510", md("p8e00004", """
Build the coarse and the fine network (the same architecture, separate weights) and train them
together with one optimiser on the summed coarse and fine losses.
"""))
    insert_before(nb, "d2da4a3e", md("p8e00005", SAVE_MD))
    insert_before(nb, "35d4515b", md("p8e00006", VIEWER_MD))
    check_code_unchanged(draft, nb)
    save(nb, "ExtendedNeRF")


# --------------------------------------------------------------------------- main_report.ipynb
def main_report():
    nb = load("main_report")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    from main_report_cells import apply      # the evaluation code lives in its own file
    apply(nb, md=md, code=code, set_markdown=set_markdown, set_code=set_code,
          insert_before=insert_before, insert_after=insert_after, raw_docstring=raw_docstring)
    save(nb, "main_report")


if __name__ == "__main__":
    tiny()
    extended()
    main_report()
