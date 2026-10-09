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
for model, baseline in PAIRS:
    diff = np.array(results[model]["psnr"]) - np.array(results[baseline]["psnr"])
    paired[(model, baseline)] = diff
    print(f"paired | {model} - {baseline} | mean {diff.mean():+.2f} dB | range {diff.min():+.2f} to "
          f"{diff.max():+.2f} | {int((diff > 0).sum())} of {len(diff)} views higher")
'''

QUAL_MD = """
## 2. Qualitative comparison: held-out views and error maps
All six test views first (ground truth, both models, and the per-pixel error), then the report's
figure: the view ranked in the middle by Extended NeRF's gain over Tiny NeRF.
"""

QUAL = '''
# All six held-out views: ground truth, both models and their absolute error maps.
abs_error = {name: [(renders[name][i] - test_images[i]).abs().mean(-1).cpu().numpy() for i in range(6)]
             for name in ("Tiny NeRF", "Extended NeRF")}
vmax = max(error_map.max() for name in abs_error for error_map in abs_error[name])   # one colour scale
rows = [("ground truth", [image.cpu().numpy() for image in test_images], False),
        ("Tiny NeRF", [image.cpu().numpy() for image in renders["Tiny NeRF"]], False),
        ("Extended NeRF", [image.cpu().numpy() for image in renders["Extended NeRF"]], False),
        ("|error| Tiny", abs_error["Tiny NeRF"], True), ("|error| Extended", abs_error["Extended NeRF"], True)]
fig, axes = plt.subplots(5, 6, figsize=(TEXT_WIDTH, 6.0))
for row, (label, imgs, is_error) in enumerate(rows):
    for col in range(6):
        ax = axes[row, col]
        if is_error:
            error_image = ax.imshow(imgs[col], cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
        else:
            ax.imshow(imgs[col], interpolation="nearest")
        if label in results:
            ax.set_title(f"{results[label]['psnr'][col]:.2f} dB")
        elif row == 0:
            ax.set_title(f"test view {col}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[row, 0].set_ylabel(label)
fig.colorbar(error_image, ax=axes[3:, :], fraction=0.02, pad=0.01, label="mean absolute error")
plt.show()

# Report figure 1: the view ranked in the middle by Extended NeRF's gain over Tiny NeRF, as
# ground truth | Tiny NeRF | Extended NeRF | both error maps on the shared colour scale above.
gains = np.array(results["Extended NeRF"]["psnr"]) - np.array(results["Tiny NeRF"]["psnr"])
report_view = int(np.argsort(gains)[len(gains) // 2])
fig, axes = plt.subplots(1, 5, figsize=(TEXT_WIDTH, 1.75), gridspec_kw={"wspace": 0.04})
panels = [(test_images[report_view].cpu().numpy(), f"ground truth\\ntest view {report_view}", None),
          (renders["Tiny NeRF"][report_view].cpu().numpy(),
           f"Tiny NeRF\\n{results['Tiny NeRF']['psnr'][report_view]:.2f} dB", None),
          (renders["Extended NeRF"][report_view].cpu().numpy(),
           f"Extended NeRF\\n{results['Extended NeRF']['psnr'][report_view]:.2f} dB", None),
          (abs_error["Tiny NeRF"][report_view], "|error|\\nTiny NeRF", "inferno"),
          (abs_error["Extended NeRF"][report_view], "|error|\\nExtended NeRF", "inferno")]
for ax, (img, title, cmap) in zip(axes, panels):
    if cmap is None:
        ax.imshow(img, interpolation="nearest")
    else:
        error_image = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(error_image, ax=axes, fraction=0.012, pad=0.01, aspect=15)
plt.savefig("figure_1_test_views.pdf", bbox_inches="tight", dpi=300)
plt.show()
print("report view:", report_view, "| Extended - Tiny gain per view (dB):", np.round(gains, 2).tolist())
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
HISTORY_KEYS = {"Tiny NeRF": "TinyNeRF", "Extended NeRF": "ExtendedNeRF",
                "No view dirs (abl. A)": "Ablation_NoViewDirs", "Uniform 96 (abl. B)": "Ablation_Uniform96"}
colors = {"Tiny NeRF": "tab:gray", "Extended NeRF": "tab:red",
          "No view dirs (abl. A)": "tab:blue", "Uniform 96 (abl. B)": "tab:green"}

fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.35))
for name in MODEL_NAMES:
    history = histories[HISTORY_KEYS[name]]
    ax.plot(np.arange(1, len(history["test_psnr"]) + 1), history["test_psnr"], color=colors[name], lw=1,
            label=name)
ax.axhline(28, ls="--", c="k", lw=0.6)
ax.set_xlabel("epoch"); ax.set_ylabel("mean test PSNR (dB)")
ax.set_ylim(20, 29.5); ax.set_xlim(0, len(history["test_psnr"])); ax.grid(alpha=0.3)
ax.legend(ncol=2, loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig("figure_3_curves.pdf", bbox_inches="tight")
plt.show()

for name in MODEL_NAMES:
    history = histories[HISTORY_KEYS[name]]
    last10 = np.mean(history["test_psnr"][-10:]) - np.mean(history["test_psnr"][-20:-10])
    print(f"curves | {name} | epochs {len(history['test_psnr'])} | best {max(history['test_psnr']):.2f} dB | "
          f"final {history['test_psnr'][-1]:.2f} dB | last-10-epoch gain {last10:+.2f} dB | "
          f"{np.mean(history['epoch_seconds']):.1f} s per epoch | "
          f"{np.sum(history['epoch_seconds']) / 60:.1f} min")
'''

VIEW_MD = """
## 4. View-dependent appearance
Two questions. **Does Extended NeRF use the direction?** Freezing its direction input at one fixed
direction (test view 3's) while keeping the geometry must change the render if it does, while the
matte ablation A cannot change at all. **Is the change real?** Take points on the object's surface
seen in test view 0, find every training camera that sees the same point unoccluded, and read the
point's brightness in those photographs. If the photographs change with the camera, the scene is
view-dependent, and a model's renders of those same rays should follow the change.
"""

FROZEN = '''
# (1) Re-render test view 0 with every ray's direction input frozen to test view 3's direction.
def render_frozen_direction(coarse, fine, c2w, frozen_dir, chunk=4096):
    """Hierarchical render in which the colour head always receives frozen_dir."""
    class Frozen(torch.nn.Module):
        """Wraps a network so that its colour head sees frozen_dir instead of the ray's direction."""
        def __init__(self, net):
            super().__init__(); self.net = net
        def forward(self, x, d):
            return self.net(x, frozen_dir.expand_as(d))
    return render_image(lambda o, d: render_rays(Frozen(coarse), Frozen(fine), o, d,
                                                 NEAR, FAR, N_C, N_F)[1], c2w, chunk).clamp(0, 1)

view = 0
frozen_dir = F.normalize(-test_poses[3][:3, 2], dim=0)      # test view 3 looks along its -z axis
frozen = {"Extended NeRF": render_frozen_direction(ext_coarse, ext_fine, test_poses[view], frozen_dir),
          "No view dirs (abl. A)": render_frozen_direction(noview_coarse, noview_fine, test_poses[view],
                                                           frozen_dir)}
change = {name: (frozen[name] - renders[name][view]).abs().mean(-1) for name in frozen}
frozen_psnr = {name: psnr(frozen[name], test_images[view]) for name in frozen}
for name in frozen:
    print(f"frozen direction | {name} | mean colour change {change[name].mean().item():.4f} | "
          f"PSNR {frozen_psnr[name]:.2f} dB with the direction frozen vs {results[name]['psnr'][view]:.2f} dB")
'''

PHOTO = '''
# (2) Do the photographs themselves change with viewpoint, and do the renders follow them?
def render_with_depth(coarse, fine, rays_o, rays_d, chunk=4096):
    """Final colour, expected depth sum(w t) and opacity sum(w) of the hierarchical renderer."""
    parts = []
    with torch.no_grad():
        for i in range(0, rays_o.shape[0], chunk):
            _, rgb, extras = render_rays(coarse, fine, rays_o[i:i + chunk], rays_d[i:i + chunk],
                                         NEAR, FAR, N_C, N_F, return_extras=True)
            parts.append((rgb.clamp(0, 1), (extras["weights_fine"] * extras["t_all"]).sum(-1),
                          extras["weights_fine"].sum(-1)))
    return [torch.cat(column, 0) for column in zip(*parts)]


def edge_strength(img):
    """Mean absolute difference between an image and its 3 x 3 local mean: large at edges."""
    img_chw = img.permute(2, 0, 1)[None]
    local_mean = F.avg_pool2d(img_chw, 3, stride=1, padding=1, count_include_pad=False)
    return (img_chw - local_mean).abs().mean(1)[0]                            # [H, W]


def read_pixels(img_chw, i, j):
    """Bilinear read of an image at fractional pixel coordinates (i = column, j = row)."""
    grid = torch.stack([i / (W - 1) * 2 - 1, j / (H - 1) * 2 - 1], -1)[None, None]
    return F.grid_sample(img_chw[None], grid, align_corners=True)[0, :, 0].T  # [N, channels]


LUMA = torch.tensor([0.299, 0.587, 0.114], device=device)    # brightness of an RGB colour
FLAT, DEPTH_TOL, MIN_CAMERAS = 0.02, 0.05, 10

# Surface points: pixels of test view 0 on the object (the background is black, and the models
# render it as opaque black too, so opacity alone would admit it), away from edges, placed at the
# rendered depth.
rays_o0, rays_d0 = (rays.reshape(-1, 3) for rays in get_rays(K, test_poses[view], H, W))
_, depth0, opacity0 = render_with_depth(ext_coarse, ext_fine, rays_o0, rays_d0)
flat0 = edge_strength(test_images[view]).reshape(-1) < FLAT
on_object = test_images[view].reshape(-1, 3).sum(-1) > 0.05          # not the black background
candidates = torch.nonzero((opacity0 > 0.99) & flat0 & on_object).squeeze(1)
chosen = candidates[torch.linspace(0, len(candidates) - 1, min(400, len(candidates)), device=device).long()]
points = rays_o0[chosen] + depth0[chosen, None] * rays_d0[chosen]

# Every training camera: project the points (get_rays in reverse), render the rays through those
# pixels, and keep a point only where the camera sees it unoccluded and away from an edge.
photo, ext_lum, abl_lum = [], [], []
for c2w, img in zip(train_poses, train_images):
    rotation, origin = c2w[:3, :3], c2w[:3, 3]
    p_cam = (points - origin) @ rotation                  # camera coordinates R^T (p - o)
    depth = (-p_cam[:, 2]).clamp_min(1e-3)                # depth along the viewing axis
    px_col = K[0, 2] + K[0, 0] * p_cam[:, 0] / depth      # pixel coordinates of each point
    px_row = K[1, 2] - K[1, 1] * p_cam[:, 1] / depth
    in_frame = ((-p_cam[:, 2] > NEAR) & (px_col >= 1) & (px_col <= W - 2)
                & (px_row >= 1) & (px_row <= H - 2))
    px_col, px_row = torch.where(in_frame, px_col, K[0, 2]), torch.where(in_frame, px_row, K[1, 2])
    rays_d = torch.stack([(px_col - K[0, 2]) / K[0, 0], -(px_row - K[1, 2]) / K[1, 1],
                          -torch.ones_like(px_col)], -1) @ rotation.T
    rays_o = origin.expand_as(rays_d)
    ext_rgb, ext_depth, ext_opacity = render_with_depth(ext_coarse, ext_fine, rays_o, rays_d)
    abl_rgb, _, _ = render_with_depth(noview_coarse, noview_fine, rays_o, rays_d)
    seen = (in_frame & (ext_opacity > 0.99) & ((ext_depth - depth).abs() < DEPTH_TOL)
            & (read_pixels(edge_strength(img)[None], px_col, px_row)[:, 0] < FLAT))
    missing = torch.full_like(depth, float("nan"))
    photo.append(torch.where(seen, read_pixels(img.permute(2, 0, 1), px_col, px_row) @ LUMA, missing))
    ext_lum.append(torch.where(seen, ext_rgb @ LUMA, missing))
    abl_lum.append(torch.where(seen, abl_rgb @ LUMA, missing))
photo, ext_lum, abl_lum = (torch.stack(per_camera, 1).cpu().numpy()
                           for per_camera in (photo, ext_lum, abl_lum))

n_cameras = (~np.isnan(photo)).sum(1)
keep = n_cameras >= MIN_CAMERAS


def per_point(stat_fn, *arrays):
    """Apply stat_fn to the cameras that see each kept point."""
    out = []
    for point_rows in zip(*(array[keep] for array in arrays)):
        seen_by = ~np.isnan(point_rows[0])
        out.append(stat_fn(*(values[seen_by] for values in point_rows)))
    return np.array(out)


def correlation(a, b):
    """Pearson correlation across cameras; undefined (nan) when either side does not vary."""
    a, b = a - a.mean(), b - b.mean()
    scale = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / scale) if scale > 1e-12 else float("nan")


spread = {name: per_point(np.std, values)
          for name, values in (("photographs", photo), ("Extended NeRF", ext_lum),
                               ("No view dirs (abl. A)", abl_lum))}
rms = {name: per_point(lambda photo_values, render_values:
                       np.sqrt(np.mean((render_values - photo_values) ** 2)), photo, values)
       for name, values in (("Extended NeRF", ext_lum), ("No view dirs (abl. A)", abl_lum))}
corr = {name: per_point(correlation, photo, values)
        for name, values in (("Extended NeRF", ext_lum), ("No view dirs (abl. A)", abl_lum))}
print(f"photo consistency | {int(keep.sum())} surface points seen by at least {MIN_CAMERAS} training "
      f"cameras | median {int(np.median(n_cameras[keep]))} cameras per point")
for stat, stat_fn in (("median", np.median), ("mean", np.mean)):
    print(f"photo consistency | brightness spread across cameras ({stat} std) | photographs "
          f"{stat_fn(spread['photographs']):.4f} | Extended NeRF {stat_fn(spread['Extended NeRF']):.4f} | "
          f"ablation A {stat_fn(spread['No view dirs (abl. A)']):.4f}")
for name in rms:
    print(f"photo consistency | {name} | RMS brightness error against the photographs {rms[name].mean():.4f} | "
          f"median correlation with the photographs across cameras {np.nanmedian(corr[name]):.2f}")
# Share of points where a model's renders vary more across cameras than the photographs do.
over = {name: float(np.mean(spread[name] > spread["photographs"])) for name in rms}
print(f"photo consistency | renders vary more than the photographs at | Extended NeRF "
      f"{100 * over['Extended NeRF']:.0f}% of points | ablation A {100 * over['No view dirs (abl. A)']:.0f}% of points")
'''

ERROR_SPLIT = '''
# (3) Where in the image does the direction input lower the error? Split every test pixel into
# edge pixels (edge strength at or above FLAT) and flat pixels, then share out the squared error.
edge_px, sq_err_abl_a, sq_err_ext = [], [], []
for i in range(len(test_images)):
    edge_px.append((edge_strength(test_images[i]) >= FLAT).reshape(-1))
    sq_err_abl_a.append(((renders["No view dirs (abl. A)"][i] - test_images[i]) ** 2)
                        .mean(-1).reshape(-1))
    sq_err_ext.append(((renders["Extended NeRF"][i] - test_images[i]) ** 2).mean(-1).reshape(-1))
edge_px, sq_err_abl_a, sq_err_ext = (torch.cat(per_view)
                                     for per_view in (edge_px, sq_err_abl_a, sq_err_ext))
reduction = sq_err_abl_a - sq_err_ext              # positive where Extended NeRF is closer
print(f"error split | edge pixels {100 * edge_px.float().mean().item():.1f}% of test pixels | "
      f"{100 * (sq_err_abl_a[edge_px].sum() / sq_err_abl_a.sum()).item():.1f}% of ablation A's squared error | "
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
for col, (img, title) in enumerate(panels):
    ax = fig.add_subplot(top[0, col])
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
for name, colour, label in (("Extended NeRF", "tab:red", "Extended"),
                            ("No view dirs (abl. A)", "tab:blue", "abl. A")):
    counts, _, _ = ax.hist(corr[name][~np.isnan(corr[name])], bins=bins, histtype="step", color=colour,
                           lw=1, label=label)
    ax.axvline(np.nanmedian(corr[name]), color=colour, lw=0.7, ls=":")
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
fewer samples per ray than it was trained with. Extended NeRF keeps its 1:2 split between coarse
and fine samples; ablation B uses evenly spaced samples. Comparing them at the same number of
**network evaluations per ray** (coarse plus fine for the hierarchical model) compares equal work.
"""

SWEEP = '''
# Sample budget at render time: mean test PSNR against the number of network evaluations per ray.
def mean_test_psnr(render_fn):
    """Mean PSNR over the six test views of one renderer (rays -> colour)."""
    # 1,024 rays per chunk: at 256 samples per ray a 4,096-ray chunk needs 512 MB per layer,
    # more than a 4 GB GPU slice has free at that point.
    return float(np.mean([psnr(render_image(render_fn, test_poses[i], chunk=1024).clamp(0, 1),
                               test_images[i]) for i in range(len(test_images))]))

BUDGETS = [24, 48, 96, 192]                        # samples the final network sees per ray
sweep = {"hierarchical": {}, "uniform": {}}        # keyed by network evaluations per ray
for budget in BUDGETS:
    n_coarse = budget // 3                         # the trained 1:2 split of coarse to fine samples
    render_fn = lambda o, d, n_c=n_coarse, n_f=budget - n_coarse: render_rays(
        ext_coarse, ext_fine, o, d, NEAR, FAR, n_c, n_f)[1]
    sweep["hierarchical"][n_coarse + budget] = mean_test_psnr(render_fn)
    print(f"budget | hierarchical | coarse {n_coarse} + fine {budget - n_coarse} | "
          f"{n_coarse + budget} evaluations | PSNR {sweep['hierarchical'][n_coarse + budget]:.2f} dB")
for evaluations in sorted(set(BUDGETS) | set(sweep["hierarchical"])):
    sweep["uniform"][evaluations] = mean_test_psnr(
        lambda o, d, n=evaluations: render_rays_uniform(uniform_net, o, d, NEAR, FAR, n))
    print(f"budget | uniform | {evaluations} samples | {evaluations} evaluations | "
          f"PSNR {sweep['uniform'][evaluations]:.2f} dB")
for evaluations, hier_psnr in sweep["hierarchical"].items():
    uniform_psnr = sweep["uniform"][evaluations]
    print(f"equal work | {evaluations} evaluations per ray | hierarchical {hier_psnr:.2f} dB | uniform "
          f"{uniform_psnr:.2f} dB | difference {hier_psnr - uniform_psnr:+.2f} dB")
'''

SAMPLING_FIG = '''
# Report figure 5, top: one object ray of test view 0 (the middle one in pixel order). The bars are
# the coarse network's weights; below them, one row of ticks per sample set, so the evenly spaced
# coarse samples can be compared with the fine samples that sample_pdf drew from those weights.
object_rays = torch.nonzero(hit).squeeze(1)
ray = int(object_rays[len(object_rays) // 2])
t_coarse = extras["t_coarse"][ray].cpu().numpy()
w_coarse = extras["weights_coarse"][ray].cpu().numpy()
t_fine = extras["t_fine"][ray].cpu().numpy()
peak = float(w_coarse.max())

fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 2.35),
                         gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.62})
ax = axes[0]
ax.bar(t_coarse, w_coarse, width=(FAR - NEAR) / N_C * 0.9, color="tab:blue", alpha=0.45,
       label="coarse weights")
ax.plot(t_coarse, np.full(N_C, -0.15 * peak), "|", color="tab:blue", ms=4,
        label=f"{N_C} coarse samples")
ax.plot(t_fine, np.full(N_F, -0.36 * peak), "|", color="tab:red", ms=4,
        label=f"{N_F} fine samples")
ax.set_xlim(NEAR, FAR); ax.set_ylim(-0.48 * peak, 1.08 * peak)
ax.set_yticks([tick for tick in plt.MaxNLocator(3).tick_values(0, peak) if 0 <= tick <= 1.05 * peak])
ax.set_xlabel("distance along the ray t"); ax.set_ylabel("weight $w_i$")
ax.grid(alpha=0.3)
# The legend sits in its own row above the panel, where it cannot cover any data.
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, handlelength=0.8,
          columnspacing=0.8, borderaxespad=0.2)

# Bottom: mean test PSNR against network evaluations per ray, for both samplers.
ax = axes[1]
for name, colour, style in (("hierarchical", "tab:red", "-o"), ("uniform", "tab:green", "-s")):
    evaluations = sorted(sweep[name])
    label = "Extended NeRF (coarse + fine)" if name == "hierarchical" else "ablation B (uniform)"
    ax.plot(evaluations, [sweep[name][n] for n in evaluations], style, ms=3, lw=1, color=colour,
            label=label)
# A ring marks the budget each model was trained at (no full-height guide line to cross the legend).
for name, colour, trained in (("hierarchical", "tab:red", N_C + N_C + N_F),
                              ("uniform", "tab:green", N_C + N_F)):
    ax.plot(trained, sweep[name][trained], "o", ms=7, mfc="none", mec=colour, mew=0.8)
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(sweep["uniform"])); ax.set_xticklabels([str(n) for n in sorted(sweep["uniform"])])
ax.set_xlabel("network evaluations per ray (rings: as trained)"); ax.set_ylabel("mean test PSNR (dB)")
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
frames = {name: [render_image(RENDERERS[name], pose_spherical(theta, -30., 4.)).clamp(0, 1).cpu().numpy()
                 for theta in thetas] for name in ("Tiny NeRF", "Extended NeRF")}
print(f"turntable | {len(thetas)} frames | camera 30 degrees above the horizontal | radius 4")

strip_idx = np.arange(0, 40, 5)                    # 8 frames, 45 degrees apart
fig, axes = plt.subplots(2, len(strip_idx), figsize=(TEXT_WIDTH, 1.95),
                         gridspec_kw={"wspace": 0.03, "hspace": 0.05})
for row, name in enumerate(("Tiny NeRF", "Extended NeRF")):
    for col, frame in enumerate(strip_idx):
        axes[row, col].imshow(frames[name][frame], interpolation="nearest")
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(f"azimuth {thetas[frame]:.0f}°")
    axes[row, 0].set_ylabel(name)
plt.savefig("figure_2_novel_views.pdf", bbox_inches="tight", dpi=300)
plt.show()

# Side-by-side GIF: Tiny NeRF (left) | Extended NeRF (right)
gif_frames = [Image.fromarray((np.concatenate([tiny_frame, ext_frame], axis=1) * 255).astype(np.uint8))
              .resize((400, 200), Image.NEAREST)
              for tiny_frame, ext_frame in zip(frames["Tiny NeRF"], frames["Extended NeRF"])]
gif_frames[0].save("nerf_turntable.gif", save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
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
    result, history = results[name], histories[HISTORY_KEYS[name]]
    summary[name] = {"mean_psnr": float(np.mean(result["psnr"])), "std_psnr": float(np.std(result["psnr"])),
                     "mean_ssim": float(np.mean(result["ssim"])), "psnr_per_view": result["psnr"],
                     "ssim_per_view": result["ssim"], "ms_per_image": 1000 * result["render_seconds"],
                     "params": result["params"], "evaluations_per_ray": EVALUATIONS[name],
                     "train_minutes": float(np.sum(history["epoch_seconds"]) / 60),
                     "seconds_per_epoch": float(np.mean(history["epoch_seconds"]))}
    if "coarse_psnr" in result:
        summary[name]["coarse_mean_psnr"] = float(np.mean(result["coarse_psnr"]))
    line = summary[name]
    coarse = f"{line['coarse_mean_psnr']:.2f}" if "coarse_mean_psnr" in line else "none"
    print(f"summary | {name} | PSNR {line['mean_psnr']:.2f} | sd {line['std_psnr']:.2f} | "
          f"SSIM {line['mean_ssim']:.3f} | coarse PSNR {coarse} | {line['evaluations_per_ray']} evaluations "
          f"per ray | {line['ms_per_image']:.0f} ms per image | {line['params']:,} parameters | "
          f"{line['train_minutes']:.1f} min training | {line['seconds_per_epoch']:.1f} s per epoch")
summary["paired"] = {f"{model} - {baseline}": diff.tolist() for (model, baseline), diff in paired.items()}
summary["frozen_direction"] = {"mean_change": {name: change[name].mean().item() for name in change},
                               "psnr": frozen_psnr}
summary["photo_consistency"] = {
    "points": int(keep.sum()),
    "mean_spread": {name: float(values.mean()) for name, values in spread.items()},
    "median_spread": {name: float(np.median(values)) for name, values in spread.items()},
    "share_varying_more_than_photographs": over,
    "rms": {name: float(values.mean()) for name, values in rms.items()},
    "median_correlation": {name: float(np.nanmedian(values)) for name, values in corr.items()}}
summary["error_split"] = {"edge_pixel_share": edge_px.float().mean().item(),
                          "edge_share_of_ablation_a_error": (sq_err_abl_a[edge_px].sum()
                                                             / sq_err_abl_a.sum()).item(),
                          "edge_share_of_reduction": (reduction[edge_px].sum() / reduction.sum()).item()}
summary["share_near_surface"] = share
summary["sample_budget"] = sweep
summary["device"] = gpu_name
json.dump(summary, open("results_summary.json", "w"), indent=1)
print("saved results_summary.json")
'''


# Descriptive names for the short ones in the three draft cells this file edits. Renaming touches
# NAME tokens only (see rename), so strings, comments and attribute access are left as they were.
DRAFT_RENAMES = {
    "20924039": {"ckpt": "checkpoint", "uni": "uniform_net", "nvd_coarse": "noview_coarse",
                 "nvd_fine": "noview_fine", "m": "net", "k": "name", "v": "count"},
    "718bcd2c": {"fn": "render_fn", "imgs": "view_images", "im": "image", "r": "result",
                 "p": "view_psnr"},
    "4dc05a96": {"ex": "extras", "e": "chunk_extras", "k": "key", "v": "fraction", "t": "t_vals",
                 "w": "weights", "lo": "lower", "hi": "upper"},
}
DRAFT_DOCSTRINGS = {
    "def load(path):\n":
        '    """Read a checkpoint onto the current device (tensors only)."""\n',
    "def timed_render(render_fn, c2w):\n":
        '    """Render one view and return the image with its wall-clock time in seconds."""\n',
    "def surface_interval(t_vals, weights, lower=0.05, upper=0.95):\n":
        '    """Per ray, the depths at which the cumulative weight reaches lower and upper."""\n',
}


def rename(source, mapping):
    """Rename identifiers by token position, keeping every other character of the source."""
    import io
    import tokenize
    lines = source.splitlines(keepends=True)
    hits = [token for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.NAME and token.string in mapping]
    for token in reversed(hits):                    # right to left keeps earlier columns valid
        (row, start), (_, end) = token.start, token.end
        line = lines[row - 1]
        lines[row - 1] = line[:start] + mapping[token.string] + line[end:]
    return "".join(lines)


def add_docstrings(source):
    """Insert the one-line docstring of DRAFT_DOCSTRINGS under the one def line the cell holds."""
    headers = [header for header in DRAFT_DOCSTRINGS if header in source]
    assert len(headers) == 1, headers
    return source.replace(headers[0], headers[0] + DRAFT_DOCSTRINGS[headers[0]], 1)


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
    text = text.replace(
        "torch.load(path, map_location=device)",
        "torch.load(path, map_location=device, weights_only=True)   # tensors only, no pickled code")
    set_code(nb, "20924039", add_docstrings(rename(text, DRAFT_RENAMES["20924039"])))
    set_markdown(nb, "5221e02d", QUANT_MD)
    nb_cell = nb["cells"][[c["id"] for c in nb["cells"]].index("718bcd2c")]
    text = add_docstrings(rename("".join(nb_cell["source"]), DRAFT_RENAMES["718bcd2c"]))
    nb_cell["source"] = ((text.rstrip("\n") + QUANT_EXTRA).rstrip("\n")
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
                 md("p8m10002", "Now the photographs: project points on the object that lie away "
                                "from image edges (and off the black background, which the models "
                                "also render as opaque) into every training camera, and compare "
                                "brightness across cameras."),
                 code("p8c10002", PHOTO),
                 md("p8m10006", "The surface points above leave out the edges. Next, where "
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
    text = src[:cut].replace("[{hit.sum().item()} object rays]", "[{hit.sum().item():,} object rays]")
    set_code(nb, "4dc05a96", add_docstrings(rename(text, DRAFT_RENAMES["4dc05a96"])))
    insert_after(nb, "4dc05a96",
                 md("p8m10004", "The sample budget: Extended NeRF and ablation B rendered at "
                                "several numbers of samples per ray."),
                 code("p8c10004", SWEEP),
                 md("p8m10005", "The report's figure. Top: on one object ray, the coarse weights "
                                "and where each sampler's samples land (the evenly spaced coarse "
                                "samples, and the fine samples drawn from the coarse weights). "
                                "Bottom: the sample budget against equal work."),
                 code("p8c10005", SAMPLING_FIG))
    set_markdown(nb, "9c492b5b", MOVIE_MD)
    set_code(nb, "e11be2c2", MOVIE)
    set_markdown(nb, "572a24fd", SUMMARY_MD)
    set_code(nb, "389df371", SUMMARY)
