r"""The cells of main_report.ipynb that differ from the group draft (used by patch_notebooks.py).

main_report.ipynb reproduces every number and figure of the report from the saved weights and
trains nothing. Against the draft it:

* prints the GPU name, so the render times in the report have a stated device;
* compares the models view by view (paired differences) instead of only by their means;
* tests whether the photographs themselves change colour with viewpoint, by projecting surface
  points into the 100 training cameras, and whether each model's renders follow that change;
* sweeps the number of samples per ray at render time, to measure when hierarchical sampling pays;
* draws every report figure at the size it is printed (7 pt text) and saves it as PDF;
* ends with one summary block that holds every number the report quotes.
"""

INTRO = """
# Project 8 main report: Tiny NeRF vs Extended NeRF (Task 3)
**Group 4**: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

Reproduces every number and figure in the report from the saved weights. **No training happens
here**: the four models are loaded from `tinynerf.pth`, `extended_nerf.pth`,
`ablation_no_viewdirs.pth` and `ablation_uniform96.pth`, and the training curves from
`training_history.json`. Every metric is computed on the 6 held-out test views (100-105), which no
model saw during training. Rendering at test time has no random jitter, so the numbers repeat
exactly from run to run; only the render times depend on the machine.

| Section | Question |
|---|---|
| 1 | How good is each model on the held-out views, overall and view by view? |
| 2 | What do the differences look like? |
| 3 | Did every model get the same training, and had it converged? |
| 4 | Does the scene really look different from different directions, and which model follows that? |
| 5 | Where do the samples land, and when does hierarchical sampling pay? |
| 6 | Novel views: a 360-degree turntable of generated camera poses (the "3D movie") |
| 7 | Every number the report quotes |
"""

IMPORTS = '''
# Libraries (the Tutorial 10.4 stack). This notebook only evaluates the saved models.
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Set device. Every tensor and network below is moved to it with .to(device).
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f"Using device: {device} | GPU: {gpu_name} | torch {torch.__version__}")
# Recent NVIDIA GPUs run convolutions in TF32 (a 10-bit mantissa) by default, which moves SSIM's
# Gaussian filtering in the third decimal; full float32 makes GPU and CPU runs agree.
torch.backends.cudnn.allow_tf32 = False

# Test-time rendering has no randomness; the seed only pins anything left random.
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# Scene bounds and sample budgets shared by every model in this project.
NEAR, FAR = 2.0, 6.0          # ray segment that contains the object (tutorial values)
N_SAMPLES_TINY = 64           # Tiny NeRF: uniform samples per ray (tutorial value)
N_C, N_F = 32, 64             # Extended NeRF: coarse samples, extra fine samples (starter values)

# Report figures are drawn at their printed size, so 7 pt here is 7 pt on the page.
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "pdf.fonttype": 42})
TEXT_WIDTH, COLUMN_WIDTH = 7.0, 3.4      # inches: the report's page width and column width
'''

DEFS_MD = """
## Data, rays and models
The definitions below are copied from `TinyNeRF.ipynb` and `ExtendedNeRF.ipynb` without change, so
that this notebook rebuilds exactly the models that were trained.
"""

LEADS = {
    "48b8adc5": "Load the 106 posed views and split them as the starter code does: 0-99 for "
                "training, 100-105 held out.",
    "bae555d4": "`get_rays`: the origin and direction of the ray through every pixel of a camera.",
    "e0dfb18d": "Positional encoding $\\gamma$, which lets a small MLP represent fine detail.",
    "b8c0efef": "The Tutorial 10.4 model (Tiny NeRF): position in, density and one colour out.",
    "2e3fc83f": "Task 1's model: density from position only, colour from position features and "
                "the viewing direction.",
    "4a620304": "Volume rendering: the pixel colour is the weighted sum of the samples' colours, "
                "with weights $w_i = T_i\\alpha_i$.",
    "3523d27e": "Tiny NeRF's renderer: 64 evenly spaced samples, one network.",
    "9ac211ce": "`sample_pdf`, provided by the starter code: inverse-transform sampling from the "
                "coarse weights.",
    "6fd89f93": "Task 2's renderer (coarse pass, `sample_pdf`, fine pass) and the uniform renderer "
                "of ablation B.",
    "a70a78d4": "Metrics (PSNR, SSIM) and a chunked full-image renderer.",
    "b69f145f": "Camera poses on a sphere around the object (from the tutorial), used for the "
                "turntable in section 6.",
}

QUANT_MD = """
## 1. Quantitative comparison on held-out views
PSNR and SSIM per test view, mean render time per 100 x 100 image and parameter count. The coarse
network's own PSNR shows what the fine pass adds on top of it. Ablation A is the Extended model
with its direction input set to zero (hierarchical, matte); ablation B is one view-dependent
network on 96 uniform samples (view-dependent, no hierarchy).
"""

QUANT_EXTRA = '''

at_28 = int(np.sum(np.array(results["Extended NeRF"]["psnr"]) >= 28))
print(f"Extended NeRF views at or above 28 dB: {at_28} of {len(test_images)}")
'''

PAIRED_MD = """
Every model renders the same six views, and some views are harder than others (thin parts seen
edge-on). Subtracting two models' PSNR view by view removes that shared difficulty, so a
difference that holds on every view is not an accident of which views were held out.
"""

PAIRED = '''
# Paired, view-by-view PSNR differences between models.
PAIRS = [("Extended NeRF", "Tiny NeRF"), ("Extended NeRF", "No view dirs (abl. A)"),
         ("Extended NeRF", "Uniform 96 (abl. B)"), ("No view dirs (abl. A)", "Tiny NeRF"),
         ("Uniform 96 (abl. B)", "Tiny NeRF")]
paired = {}
for a, b in PAIRS:
    d = np.array(results[a]["psnr"]) - np.array(results[b]["psnr"])
    paired[(a, b)] = d
    print(f"paired | {a} - {b} | mean {d.mean():+.2f} dB | range {d.min():+.2f} to {d.max():+.2f} | "
          f"{int((d > 0).sum())} of {len(d)} views higher")
'''

QUAL_MD = """
## 2. Qualitative comparison: held-out views and error maps
All six test views first (ground truth, both models, and the per-pixel error), then the report's
figure: the view ranked in the middle by Extended NeRF's gain over Tiny NeRF.
"""

QUAL = '''
# All six held-out views: ground truth, both models and their absolute error maps.
err = {name: [(renders[name][i] - test_images[i]).abs().mean(-1).cpu().numpy() for i in range(6)]
       for name in ("Tiny NeRF", "Extended NeRF")}
vmax = max(e.max() for name in err for e in err[name])        # one colour scale for every map
rows = [("ground truth", [im.cpu().numpy() for im in test_images], False),
        ("Tiny NeRF", [im.cpu().numpy() for im in renders["Tiny NeRF"]], False),
        ("Extended NeRF", [im.cpu().numpy() for im in renders["Extended NeRF"]], False),
        ("|error| Tiny", err["Tiny NeRF"], True), ("|error| Extended", err["Extended NeRF"], True)]
fig, axes = plt.subplots(5, 6, figsize=(TEXT_WIDTH, 6.0))
for r, (label, imgs, is_error) in enumerate(rows):
    for c in range(6):
        ax = axes[r, c]
        if is_error:
            im = ax.imshow(imgs[c], cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
        else:
            ax.imshow(imgs[c], interpolation="nearest")
        if label in results:
            ax.set_title(f"{results[label]['psnr'][c]:.2f} dB")
        elif r == 0:
            ax.set_title(f"test view {c}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[r, 0].set_ylabel(label)
fig.colorbar(im, ax=axes[3:, :], fraction=0.02, pad=0.01, label="mean absolute error")
plt.show()

# Report figure 1: the view ranked in the middle by Extended NeRF's gain over Tiny NeRF, as
# ground truth | Tiny NeRF | Extended NeRF | both error maps on the shared colour scale above.
gains = np.array(results["Extended NeRF"]["psnr"]) - np.array(results["Tiny NeRF"]["psnr"])
v = int(np.argsort(gains)[len(gains) // 2])
fig, axes = plt.subplots(1, 5, figsize=(TEXT_WIDTH, 1.75), gridspec_kw={"wspace": 0.04})
panels = [(test_images[v].cpu().numpy(), f"ground truth\\ntest view {v}", None),
          (renders["Tiny NeRF"][v].cpu().numpy(), f"Tiny NeRF\\n{results['Tiny NeRF']['psnr'][v]:.2f} dB", None),
          (renders["Extended NeRF"][v].cpu().numpy(),
           f"Extended NeRF\\n{results['Extended NeRF']['psnr'][v]:.2f} dB", None),
          (err["Tiny NeRF"][v], "|error|\\nTiny NeRF", "inferno"),
          (err["Extended NeRF"][v], "|error|\\nExtended NeRF", "inferno")]
for ax, (img, title, cmap) in zip(axes, panels):
    if cmap is None:
        ax.imshow(img, interpolation="nearest")
    else:
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, aspect=15)
plt.savefig("figure_1_test_views.pdf", bbox_inches="tight", dpi=300)
plt.show()
print("report view:", v, "| Extended - Tiny gain per view (dB):", np.round(gains, 2).tolist())
'''

CURVES_MD = """
## 3. Training curves (all four models, identical training budget)
Test PSNR after every epoch, from the histories saved by the training notebooks. The curves show
whether each model had converged by the last epoch; the weights used everywhere else are the
last epoch's.
"""

CURVES = '''
# Test PSNR during training for the four models, from the saved histories.
histories = json.load(open("training_history.json"))
key = {"Tiny NeRF": "TinyNeRF", "Extended NeRF": "ExtendedNeRF",
       "No view dirs (abl. A)": "Ablation_NoViewDirs", "Uniform 96 (abl. B)": "Ablation_Uniform96"}
colors = {"Tiny NeRF": "tab:gray", "Extended NeRF": "tab:red",
          "No view dirs (abl. A)": "tab:blue", "Uniform 96 (abl. B)": "tab:green"}

fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.35))
for name in MODEL_NAMES:
    h = histories[key[name]]
    ax.plot(np.arange(1, len(h["test_psnr"]) + 1), h["test_psnr"], color=colors[name], lw=1, label=name)
ax.axhline(28, ls="--", c="k", lw=0.6)
ax.set_xlabel("epoch"); ax.set_ylabel("mean test PSNR (dB)")
ax.set_ylim(20, 29.5); ax.set_xlim(0, len(h["test_psnr"])); ax.grid(alpha=0.3)
ax.legend(ncol=2, loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig("figure_3_curves.pdf", bbox_inches="tight")
plt.show()

for name in MODEL_NAMES:
    h = histories[key[name]]
    last10 = np.mean(h["test_psnr"][-10:]) - np.mean(h["test_psnr"][-20:-10])
    print(f"curves | {name} | epochs {len(h['test_psnr'])} | best {max(h['test_psnr']):.2f} dB | "
          f"final {h['test_psnr'][-1]:.2f} dB | last-10-epoch gain {last10:+.2f} dB | "
          f"{np.mean(h['epoch_seconds']):.1f} s per epoch | {np.sum(h['epoch_seconds']) / 60:.1f} min")
'''

VIEW_MD = """
## 4. View-dependent appearance
Two questions. **Does Extended NeRF use the direction?** Freezing its direction input at one fixed
direction (test view 3's) while keeping the geometry must change the render if it does, while the
matte ablation A cannot change at all. **Is the change real?** Take points on the object's surface
seen in test view 0, find every training camera that sees the same point unblocked, and read the
point's brightness in those photographs. If the photographs change with the camera, the scene is
view-dependent, and a model's renders of those same rays should follow the change.
"""

FROZEN = '''
# (1) Re-render test view 0 with every ray's direction input frozen to test view 3's direction.
def render_frozen_direction(coarse, fine, c2w, frozen_dir, chunk=4096):
    """Hierarchical render in which the colour head always receives frozen_dir."""
    class Frozen(torch.nn.Module):
        def __init__(self, net):
            super().__init__(); self.net = net
        def forward(self, x, d):
            return self.net(x, frozen_dir.expand_as(d))
    return render_image(lambda o, d: render_rays(Frozen(coarse), Frozen(fine), o, d,
                                                 NEAR, FAR, N_C, N_F)[1], c2w, chunk).clamp(0, 1)

view = 0
frozen_dir = F.normalize(-test_poses[3][:3, 2], dim=0)      # test view 3 looks along its -z axis
frozen = {"Extended NeRF": render_frozen_direction(ext_coarse, ext_fine, test_poses[view], frozen_dir),
          "No view dirs (abl. A)": render_frozen_direction(nvd_coarse, nvd_fine, test_poses[view], frozen_dir)}
change = {n: (frozen[n] - renders[n][view]).abs().mean(-1) for n in frozen}
frozen_psnr = {n: psnr(frozen[n], test_images[view]) for n in frozen}
for n in frozen:
    print(f"frozen direction | {n} | mean colour change {change[n].mean().item():.4f} | "
          f"PSNR {frozen_psnr[n]:.2f} dB with the direction frozen vs {results[n]['psnr'][view]:.2f} dB")
'''

PHOTO = '''
# (2) Do the photographs themselves change with viewpoint, and do the renders follow them?
def render_with_depth(coarse, fine, rays_o, rays_d, chunk=4096):
    """Final colour, expected depth sum(w t) and opacity sum(w) of the hierarchical renderer."""
    parts = []
    with torch.no_grad():
        for i in range(0, rays_o.shape[0], chunk):
            _, rgb, ex = render_rays(coarse, fine, rays_o[i:i + chunk], rays_d[i:i + chunk],
                                     NEAR, FAR, N_C, N_F, return_extras=True)
            parts.append((rgb.clamp(0, 1), (ex["weights_fine"] * ex["t_all"]).sum(-1),
                          ex["weights_fine"].sum(-1)))
    return [torch.cat(p, 0) for p in zip(*parts)]


def edge_strength(img):
    """Mean absolute difference between an image and its 3 x 3 local mean: large at edges."""
    x = img.permute(2, 0, 1)[None]
    local_mean = F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
    return (x - local_mean).abs().mean(1)[0]                                  # [H, W]


def read_pixels(img_chw, i, j):
    """Bilinear read of an image at fractional pixel coordinates (i = column, j = row)."""
    grid = torch.stack([i / (W - 1) * 2 - 1, j / (H - 1) * 2 - 1], -1)[None, None]
    return F.grid_sample(img_chw[None], grid, align_corners=True)[0, :, 0].T  # [N, channels]


LUMA = torch.tensor([0.299, 0.587, 0.114], device=device)    # brightness of an RGB colour
FLAT, DEPTH_TOL, MIN_CAMERAS = 0.02, 0.05, 10

# Surface points: pixels of test view 0 on the object (the background is black, and the models
# render it as opaque black too, so opacity alone would admit it), away from edges, placed at the
# rendered depth.
rays_o0, rays_d0 = (x.reshape(-1, 3) for x in get_rays(K, test_poses[view], H, W))
_, depth0, opacity0 = render_with_depth(ext_coarse, ext_fine, rays_o0, rays_d0)
flat0 = edge_strength(test_images[view]).reshape(-1) < FLAT
on_object = test_images[view].reshape(-1, 3).sum(-1) > 0.05          # not the black background
candidates = torch.nonzero((opacity0 > 0.99) & flat0 & on_object).squeeze(1)
chosen = candidates[torch.linspace(0, len(candidates) - 1, min(400, len(candidates)), device=device).long()]
points = rays_o0[chosen] + depth0[chosen, None] * rays_d0[chosen]

# Every training camera: project the points (get_rays in reverse), render the rays through those
# pixels, and keep a point only where the camera sees it unblocked and away from an edge.
photo, ext_lum, abl_lum = [], [], []
for c2w, img in zip(train_poses, train_images):
    R, o = c2w[:3, :3], c2w[:3, 3]
    p_cam = (points - o) @ R                              # camera coordinates R^T (p - o)
    z = (-p_cam[:, 2]).clamp_min(1e-3)                    # depth along the viewing axis
    i = K[0, 2] + K[0, 0] * p_cam[:, 0] / z
    j = K[1, 2] - K[1, 1] * p_cam[:, 1] / z
    in_frame = (-p_cam[:, 2] > NEAR) & (i >= 1) & (i <= W - 2) & (j >= 1) & (j <= H - 2)
    i, j = torch.where(in_frame, i, K[0, 2]), torch.where(in_frame, j, K[1, 2])
    rays_d = torch.stack([(i - K[0, 2]) / K[0, 0], -(j - K[1, 2]) / K[1, 1], -torch.ones_like(i)], -1) @ R.T
    rays_o = o.expand_as(rays_d)
    rgb_e, depth_e, opacity_e = render_with_depth(ext_coarse, ext_fine, rays_o, rays_d)
    rgb_a, _, _ = render_with_depth(nvd_coarse, nvd_fine, rays_o, rays_d)
    seen = (in_frame & (opacity_e > 0.99) & ((depth_e - z).abs() < DEPTH_TOL)
            & (read_pixels(edge_strength(img)[None], i, j)[:, 0] < FLAT))
    missing = torch.full_like(z, float("nan"))
    photo.append(torch.where(seen, read_pixels(img.permute(2, 0, 1), i, j) @ LUMA, missing))
    ext_lum.append(torch.where(seen, rgb_e @ LUMA, missing))
    abl_lum.append(torch.where(seen, rgb_a @ LUMA, missing))
photo, ext_lum, abl_lum = (torch.stack(x, 1).cpu().numpy() for x in (photo, ext_lum, abl_lum))

n_cameras = (~np.isnan(photo)).sum(1)
keep = n_cameras >= MIN_CAMERAS


def per_point(fn, *arrays):
    """Apply fn to the cameras that see each kept point."""
    out = []
    for rows in zip(*(a[keep] for a in arrays)):
        m = ~np.isnan(rows[0])
        out.append(fn(*(r[m] for r in rows)))
    return np.array(out)


def correlation(a, b):
    """Pearson correlation across cameras; undefined (nan) when either side does not vary."""
    a, b = a - a.mean(), b - b.mean()
    scale = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / scale) if scale > 1e-12 else float("nan")


spread = {n: per_point(np.std, x) for n, x in (("photographs", photo), ("Extended NeRF", ext_lum),
                                                 ("No view dirs (abl. A)", abl_lum))}
rms = {n: per_point(lambda p, x: np.sqrt(np.mean((x - p) ** 2)), photo, x)
       for n, x in (("Extended NeRF", ext_lum), ("No view dirs (abl. A)", abl_lum))}
corr = {n: per_point(correlation, photo, x) for n, x in (("Extended NeRF", ext_lum),
                                                          ("No view dirs (abl. A)", abl_lum))}
print(f"photo consistency | {int(keep.sum())} surface points seen by at least {MIN_CAMERAS} training "
      f"cameras | median {int(np.median(n_cameras[keep]))} cameras per point")
for stat, fn in (("median", np.median), ("mean", np.mean)):
    print(f"photo consistency | brightness spread across cameras ({stat} std) | photographs "
          f"{fn(spread['photographs']):.4f} | Extended NeRF {fn(spread['Extended NeRF']):.4f} | "
          f"ablation A {fn(spread['No view dirs (abl. A)']):.4f}")
for n in rms:
    print(f"photo consistency | {n} | RMS brightness error against the photographs {rms[n].mean():.4f} | "
          f"median correlation with the photographs across cameras {np.nanmedian(corr[n]):.2f}")
# Share of points where a model's renders vary more across cameras than the photographs do.
over = {n: float(np.mean(spread[n] > spread["photographs"])) for n in rms}
print(f"photo consistency | renders vary more than the photographs at | Extended NeRF "
      f"{100 * over['Extended NeRF']:.0f}% of points | ablation A {100 * over['No view dirs (abl. A)']:.0f}% of points")
'''

ERROR_SPLIT = '''
# (3) Where in the image does the direction input lower the error? Split every test pixel into
# edge pixels (edge strength at or above FLAT) and flat pixels, then share out the squared error.
edge_px, err_a, err_e = [], [], []
for i in range(len(test_images)):
    edge_px.append((edge_strength(test_images[i]) >= FLAT).reshape(-1))
    err_a.append(((renders["No view dirs (abl. A)"][i] - test_images[i]) ** 2).mean(-1).reshape(-1))
    err_e.append(((renders["Extended NeRF"][i] - test_images[i]) ** 2).mean(-1).reshape(-1))
edge_px, err_a, err_e = (torch.cat(x) for x in (edge_px, err_a, err_e))
reduction = err_a - err_e                          # positive where Extended NeRF is closer
print(f"error split | edge pixels {100 * edge_px.float().mean().item():.1f}% of test pixels | "
      f"{100 * (err_a[edge_px].sum() / err_a.sum()).item():.1f}% of ablation A's squared error | "
      f"{100 * (reduction[edge_px].sum() / reduction.sum()).item():.1f}% of the reduction to Extended NeRF")
'''

VIEW_FIG = '''
# Report figure 4: the frozen-direction experiment (top) and every kept surface point (bottom).
# All kept points are shown rather than a chosen one: left, how much a point's brightness varies
# across the cameras that see it; right, how closely each model's brightness follows the
# photographs from camera to camera.
fig = plt.figure(figsize=(COLUMN_WIDTH, 2.45))
outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.38)
top = outer[0].subgridspec(1, 3, wspace=0.05)
bottom = outer[1].subgridspec(1, 2, wspace=0.42)
panels = [(renders["Extended NeRF"][view].cpu().numpy(), "Extended NeRF"),
          (frozen["Extended NeRF"].cpu().numpy(), "direction frozen"),
          (change["Extended NeRF"].cpu().numpy(), "|colour change|")]
for c, (img, title) in enumerate(panels):
    ax = fig.add_subplot(top[0, c])
    ax.imshow(img, cmap="magma" if img.ndim == 2 else None, interpolation="nearest")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(bottom[0, 0])
ax.boxplot([spread["photographs"], spread["Extended NeRF"], spread["No view dirs (abl. A)"]],
           showfliers=False, widths=0.55, medianprops={"color": "k"})
ax.set_xticks([1, 2, 3], ["photos", "Extended", "abl. A"])
ax.set_ylabel("brightness spread"); ax.grid(alpha=0.3, axis="y")

ax = fig.add_subplot(bottom[0, 1])
bins = np.linspace(-1, 1, 21)
peak = 0
for n, colour, label in (("Extended NeRF", "tab:red", "Extended"),
                         ("No view dirs (abl. A)", "tab:blue", "abl. A")):
    counts, _, _ = ax.hist(corr[n][~np.isnan(corr[n])], bins=bins, histtype="step", color=colour,
                           lw=1, label=label)
    ax.axvline(np.nanmedian(corr[n]), color=colour, lw=0.7, ls=":")
    peak = max(peak, counts.max())
ax.set_ylim(0, 1.45 * peak)                        # headroom for the legend
ax.set_xlabel("correlation with photos"); ax.set_ylabel("surface points")
ax.legend(loc="upper left", frameon=False, handlelength=1.2, ncol=2, columnspacing=0.8)
plt.savefig("figure_4_view_dependence.pdf", bbox_inches="tight", dpi=300)
plt.show()
'''

SAMPLING_MD = """
## 5. Hierarchical sampling: where the samples go, and when it pays
First, for every ray of test view 0 that hits the object: the share of each sampler's points that
land near the surface, taken as the depth interval holding the central 90% of the final weights
(widened by half its width on each side, and at least one uniform bin wide). Then the sample
budget: a trained NeRF is a continuous function of position, so it can be rendered with more or
fewer samples per ray than it was trained with. Extended NeRF keeps its 1 : 2 split between coarse
and fine samples; ablation B uses evenly spaced samples. Comparing them at the same number of
**network evaluations per ray** (coarse plus fine for the hierarchical model) compares equal work.
"""

SWEEP = '''
# Sample budget at render time: mean test PSNR against the number of network evaluations per ray.
def mean_test_psnr(render_fn):
    # 1,024 rays per chunk: at 256 samples per ray a 4,096-ray chunk needs 512 MB per layer,
    # more than a 4 GB GPU slice has free at that point.
    return float(np.mean([psnr(render_image(render_fn, test_poses[i], chunk=1024).clamp(0, 1),
                               test_images[i]) for i in range(len(test_images))]))

BUDGETS = [24, 48, 96, 192]                        # samples the final network sees per ray
sweep = {"hierarchical": {}, "uniform": {}}        # keyed by network evaluations per ray
for S in BUDGETS:
    n_c = S // 3
    fn = lambda o, d, n_c=n_c, n_f=S - n_c: render_rays(ext_coarse, ext_fine, o, d, NEAR, FAR, n_c, n_f)[1]
    sweep["hierarchical"][n_c + S] = mean_test_psnr(fn)
    print(f"budget | hierarchical | coarse {n_c} + fine {S - n_c} | {n_c + S} evaluations | "
          f"PSNR {sweep['hierarchical'][n_c + S]:.2f} dB")
for E in sorted(set(BUDGETS) | set(sweep["hierarchical"])):
    sweep["uniform"][E] = mean_test_psnr(lambda o, d, E=E: render_rays_uniform(uni, o, d, NEAR, FAR, E))
    print(f"budget | uniform | {E} samples | {E} evaluations | PSNR {sweep['uniform'][E]:.2f} dB")
for E, p in sweep["hierarchical"].items():
    print(f"equal work | {E} evaluations per ray | hierarchical {p:.2f} dB | uniform "
          f"{sweep['uniform'][E]:.2f} dB | difference {p - sweep['uniform'][E]:+.2f} dB")
'''

SAMPLING_FIG = '''
# Report figure 5: one object ray of test view 0 (top) and the sample budget (bottom).
r = int(torch.nonzero(hit)[len(torch.nonzero(hit)) // 2])
fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 2.4), gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.62})
ax = axes[0]
tc, wc = ex["t_coarse"][r].cpu().numpy(), ex["weights_coarse"][r].cpu().numpy()
ta, wa = ex["t_all"][r].cpu().numpy(), ex["weights_fine"][r].cpu().numpy()
ax.bar(tc, wc, width=(FAR - NEAR) / N_C * 0.9, color="tab:blue", alpha=0.35, label=f"coarse weights ({N_C} samples)")
ax.plot(ta, wa, "-o", ms=1.5, color="tab:red", lw=0.8, label=f"fine weights ({N_C + N_F} samples)")
ax.plot(ex["t_fine"][r].cpu().numpy(), np.full(N_F, -0.05), "|", color="k", ms=5, label=f"{N_F} fine samples")
ax.set_xlim(NEAR, FAR); ax.set_xlabel("distance along the ray t"); ax.set_ylabel("weight $w_i$")
ax.set_ylim(-0.12, 2.3 * max(wc.max(), wa.max()))          # headroom so the legend clears the peak
ax.legend(loc="upper right", frameon=False); ax.grid(alpha=0.3)

ax = axes[1]
for name, colour, style in (("hierarchical", "tab:red", "-o"), ("uniform", "tab:green", "-s")):
    xs = sorted(sweep[name])
    label = "Extended NeRF (coarse + fine)" if name == "hierarchical" else "ablation B (uniform)"
    ax.plot(xs, [sweep[name][x] for x in xs], style, ms=3, lw=1, color=colour, label=label)
ax.axvline(N_C + N_C + N_F, color="tab:red", lw=0.6, ls=":")
ax.axvline(N_C + N_F, color="tab:green", lw=0.6, ls=":")
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(sweep["uniform"])); ax.set_xticklabels([str(x) for x in sorted(sweep["uniform"])])
ax.set_xlabel("network evaluations per ray (dotted: as trained)"); ax.set_ylabel("mean test PSNR (dB)")
ax.legend(loc="lower right", frameon=False); ax.grid(alpha=0.3)
plt.savefig("figure_5_sampling.pdf", bbox_inches="tight")
plt.show()
'''

MOVIE_MD = """
## 6. Novel-view synthesis: a 360-degree turntable (the "3D movie")
The camera circles the object 30 degrees above the horizontal at radius 4 (the tutorial's
`pose_spherical(theta, -30, 4)`). These 40 poses are generated, not taken from the dataset, so
every frame is a novel view. The strip below is the report's figure; `nerf_turntable.gif` is the
full 40-frame movie, Tiny NeRF on the left and Extended NeRF on the right.
"""

MOVIE = '''
# Novel views: a turntable of 40 frames, 9 degrees apart, rendered by both models.
thetas = np.linspace(0, 360, 40, endpoint=False)
frames = {name: [render_image(RENDERERS[name], pose_spherical(th, -30., 4.)).clamp(0, 1).cpu().numpy()
                 for th in thetas] for name in ("Tiny NeRF", "Extended NeRF")}
print(f"turntable | {len(thetas)} frames | camera 30 degrees above the horizontal | radius 4")

strip_idx = np.arange(0, 40, 5)                    # 8 frames, 45 degrees apart
fig, axes = plt.subplots(2, len(strip_idx), figsize=(TEXT_WIDTH, 1.95), gridspec_kw={"wspace": 0.03, "hspace": 0.05})
for r, name in enumerate(("Tiny NeRF", "Extended NeRF")):
    for c, k in enumerate(strip_idx):
        axes[r, c].imshow(frames[name][k], interpolation="nearest")
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            axes[r, c].set_title(f"azimuth {thetas[k]:.0f}°")
    axes[r, 0].set_ylabel(name)
plt.savefig("figure_2_novel_views.pdf", bbox_inches="tight", dpi=300)
plt.show()

# Side-by-side GIF: Tiny NeRF (left) | Extended NeRF (right)
gif = [Image.fromarray((np.concatenate([a, b], axis=1) * 255).astype(np.uint8)).resize((400, 200), Image.NEAREST)
       for a, b in zip(frames["Tiny NeRF"], frames["Extended NeRF"])]
gif[0].save("nerf_turntable.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)
print("saved figure_2_novel_views.pdf and nerf_turntable.gif (40 frames, Tiny NeRF | Extended NeRF)")
'''

SUMMARY_MD = """
## 7. Summary of all reported numbers
One line per model with every number in the report's table, then the saved summary file.
"""

SUMMARY = '''
# Every number quoted in the report: one summary line per model, and results_summary.json.
EVALUATIONS = {"Tiny NeRF": N_SAMPLES_TINY, "Extended NeRF": N_C + N_C + N_F,
               "No view dirs (abl. A)": N_C + N_C + N_F, "Uniform 96 (abl. B)": N_C + N_F}
summary = {}
for name in MODEL_NAMES:
    r, h = results[name], histories[key[name]]
    summary[name] = {"mean_psnr": float(np.mean(r["psnr"])), "std_psnr": float(np.std(r["psnr"])),
                     "mean_ssim": float(np.mean(r["ssim"])), "psnr_per_view": r["psnr"],
                     "ssim_per_view": r["ssim"], "ms_per_image": 1000 * r["render_seconds"],
                     "params": r["params"], "evaluations_per_ray": EVALUATIONS[name],
                     "train_minutes": float(np.sum(h["epoch_seconds"]) / 60),
                     "seconds_per_epoch": float(np.mean(h["epoch_seconds"]))}
    if "coarse_psnr" in r:
        summary[name]["coarse_mean_psnr"] = float(np.mean(r["coarse_psnr"]))
    s = summary[name]
    coarse = f"{s['coarse_mean_psnr']:.2f}" if "coarse_mean_psnr" in s else "none"
    print(f"summary | {name} | PSNR {s['mean_psnr']:.2f} | sd {s['std_psnr']:.2f} | SSIM {s['mean_ssim']:.3f} | "
          f"coarse PSNR {coarse} | {s['evaluations_per_ray']} evaluations per ray | {s['ms_per_image']:.0f} ms per image | "
          f"{s['params']:,} parameters | {s['train_minutes']:.1f} min training | {s['seconds_per_epoch']:.1f} s per epoch")
summary["paired"] = {f"{a} - {b}": d.tolist() for (a, b), d in paired.items()}
summary["frozen_direction"] = {"mean_change": {n: change[n].mean().item() for n in change},
                               "psnr": frozen_psnr}
summary["photo_consistency"] = {"points": int(keep.sum()),
                                "mean_spread": {n: float(v.mean()) for n, v in spread.items()},
                                "median_spread": {n: float(np.median(v)) for n, v in spread.items()},
                                "share_varying_more_than_photographs": over,
                                "rms": {n: float(v.mean()) for n, v in rms.items()},
                                "median_correlation": {n: float(np.nanmedian(v)) for n, v in corr.items()}}
summary["error_split"] = {"edge_pixel_share": edge_px.float().mean().item(),
                          "edge_share_of_ablation_a_error": (err_a[edge_px].sum() / err_a.sum()).item(),
                          "edge_share_of_reduction": (reduction[edge_px].sum() / reduction.sum()).item()}
summary["share_near_surface"] = share
summary["sample_budget"] = sweep
summary["device"] = gpu_name
json.dump(summary, open("results_summary.json", "w"), indent=1)
print("saved results_summary.json")
'''


def apply(nb, md, code, set_markdown, set_code, insert_before, insert_after, raw_docstring):
    set_markdown(nb, "49084702", INTRO)
    set_code(nb, "cd38ed18", IMPORTS)
    set_markdown(nb, "d8372ccc", DEFS_MD)
    for n, (cell_id, text) in enumerate(LEADS.items()):
        if cell_id != "48b8adc5":                   # the data cell follows the section heading
            insert_before(nb, cell_id, md(f"p8m{n:05d}", text))
        else:
            insert_after(nb, "d8372ccc", md(f"p8m{n:05d}", text))
    raw_docstring(nb, "2e3fc83f")
    set_markdown(nb, "c4cc94b0", """
## Load the trained models
Each checkpoint holds the weights saved at the last training epoch. The four renderers below are
deterministic: no jitter, and `sample_pdf` draws evenly spaced values of $u$.
""")
    loader = nb["cells"][[c["id"] for c in nb["cells"]].index("20924039")]
    text = "".join(loader["source"])
    assert "torch.load(path, map_location=device)" in text
    set_code(nb, "20924039", text.replace(
        "torch.load(path, map_location=device)",
        "torch.load(path, map_location=device, weights_only=True)   # tensors only, no pickled code"))
    set_markdown(nb, "5221e02d", QUANT_MD)
    nb_cell = nb["cells"][[c["id"] for c in nb["cells"]].index("718bcd2c")]
    nb_cell["source"] = (("".join(nb_cell["source"]).rstrip("\n") + QUANT_EXTRA).rstrip("\n")
                         .split("\n"))
    nb_cell["source"] = [s + "\n" for s in nb_cell["source"][:-1]] + [nb_cell["source"][-1]]
    insert_after(nb, "718bcd2c", md("p8m10001", PAIRED_MD), code("p8c10001", PAIRED))
    set_markdown(nb, "7a618c8c", QUAL_MD)
    set_code(nb, "90d9b6fa", QUAL)
    set_markdown(nb, "f165cd5f", CURVES_MD)
    set_code(nb, "23d116f3", CURVES)
    set_markdown(nb, "f6eacd23", VIEW_MD)
    set_code(nb, "d8d1c058", FROZEN)
    insert_after(nb, "d8d1c058",
                 md("p8m10002", "Now the photographs: project flat points on the object (not the "
                                "black background, which the models also render as opaque) into "
                                "every training camera and compare brightness across cameras."),
                 code("p8c10002", PHOTO),
                 md("p8m10006", "The flat surface points above leave out the edges. Next, where "
                                "in the image the direction input lowers the error: edge pixels "
                                "against flat pixels, over all six test views."),
                 code("p8c10006", ERROR_SPLIT),
                 md("p8m10003", "The report's figure: the frozen-direction maps, then every kept "
                                "surface point's brightness spread and its correlation with the "
                                "photographs, for both models."),
                 code("p8c10003", VIEW_FIG))
    set_markdown(nb, "35deabd4", SAMPLING_MD)
    src = "".join(nb["cells"][[c["id"] for c in nb["cells"]].index("4dc05a96")]["source"])
    cut = src.index("# Plot one object ray")
    assert "[{hit.sum().item()} object rays]" in src
    set_code(nb, "4dc05a96", src[:cut].replace("[{hit.sum().item()} object rays]",
                                               "[{hit.sum().item():,} object rays]"))
    insert_after(nb, "4dc05a96",
                 md("p8m10004", "The sample budget: Extended NeRF and ablation B rendered at "
                                "several numbers of samples per ray."),
                 code("p8c10004", SWEEP),
                 md("p8m10005", "The report's figure: where one ray's samples land, and the sample "
                                "budget against equal work."),
                 code("p8c10005", SAMPLING_FIG))
    set_markdown(nb, "9c492b5b", MOVIE_MD)
    set_code(nb, "e11be2c2", MOVIE)
    set_markdown(nb, "572a24fd", SUMMARY_MD)
    set_code(nb, "389df371", SUMMARY)
