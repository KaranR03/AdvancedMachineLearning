r"""Render the explanatory figures that ExtendedNeRF.ipynb embeds in its markdown.

These figures explain mechanisms rather than report results, so the submission spec does not
require a notebook to generate them (writing guide, section 6, the two-tier rule). They are drawn
here, saved as PNG, and embedded by tools/patch_notebooks.py as base64 data URIs, so the training
notebook gains them without being re-executed and nothing extra ships in the zip.

    nerf_architecture.png       Task 1: where the viewing direction enters the network
    hierarchical_sampling.png   Task 2: coarse weights -> PDF -> fine samples, on one ray

The second figure is an illustration on a made-up ray (one surface at t = 4.1); its sampling
uses the starter's sample_pdf logic, copied below, so the picture follows the same rule.

Run:  python tools/build_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = "f:/document/IFN_680_Advanced_Machine_Learning_and_Applications"
BASE = f"{ROOT}/weeks/week-11/project-8-3d-movie"
OUT = f"{BASE}/tools/explanatory"

# House palette, writing guide section 6.
INK, ACCENT, WARM = "#1f2933", "#2f6f9f", "#c1553b"
MUTED, PALE, GREEN = "#7b8794", "#e8ecf1", "#3f7d58"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "font.family": "DejaVu Sans",
                     "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "dejavusans",
                     "axes.spines.top": False, "axes.spines.right": False})

UNIT = 0.5          # schematic units are half an inch
FONT = 7.2
NEAR, FAR = 2.0, 6.0


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white",
                pil_kwargs={"optimize": True})
    plt.close(fig)
    print(f"  {name:28s} {os.path.getsize(path) / 1024:6.1f} KB")


# --------------------------------------------------------------------------- schematic helpers
def width_of(text, size=FONT):
    """Box width, in schematic units, that fits the longest line of text at this font size."""
    longest = max(len(line) for line in text.split("\n"))
    return longest * size * 0.0178 + 0.5


def height_of(text, size=FONT):
    return len(text.split("\n")) * size * 0.036 + 0.45


def block(ax, x, y, w, h, text, face=PALE, edge=MUTED, colour=INK, size=FONT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=colour, linespacing=1.35)


def arrow(ax, start, end, colour=INK, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color=colour,
                                 linewidth=1.0, linestyle=(0, (3, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0))


def note(ax, x, y, text, colour=MUTED, size=FONT - 0.6, ha="left"):
    ax.text(x, y, text, fontsize=size, color=colour, ha=ha, va="center", linespacing=1.3)


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width * UNIT, height * UNIT))
    # The axes fill the figure, so one schematic unit really is UNIT inches.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- 1. Task 1 network
def architecture():
    """The NeRF class of ExtendedNeRF.ipynb, layer by layer, with the sizes for L = 6, L_dir = 4."""
    boxes = {
        "x": "3D point x\n(3 numbers)",
        "gx": "positional encoding\ngamma(x): 39 numbers",
        "l1": "layer1 + ReLU\n39 -> 128",
        "l2": "layer2 + ReLU\n128 -> 128",
        "sig": "sigma_layer + ReLU\n128 -> 1",
        "feat": "feature_layer\n128 -> 128\n(no activation)",
        "d": "viewing direction d\n(unit vector)",
        "gd": "positional encoding\ngamma(d): 27 numbers",
        "c1": "concatenate 128 + 27\ncolor_layer1 + ReLU\n155 -> 64",
        "c2": "color_layer2\n+ sigmoid\n64 -> 3",
    }
    w = {k: width_of(v) for k, v in boxes.items()}
    h = max(height_of(v) for v in boxes.values())
    gap = 0.7
    # columns: input | encoding | trunk 1 | trunk 2 | heads | colour 1 | colour 2
    x0 = 0.2
    # the position and the direction share the first two columns, so size those by the wider box
    w["x"] = w["d"] = max(w["x"], w["d"])
    w["gx"] = w["gd"] = max(w["gx"], w["gd"])
    w["sig"] = w["feat"] = max(w["sig"], w["feat"])
    xs = [x0]
    for k in ("x", "gx", "l1", "l2", "feat", "c1"):
        xs.append(xs[-1] + w[k] + gap)
    y_top, y_mid, y_low = 4.6, 2.55, 0.5
    fig, ax = canvas(xs[-1] + w["c2"] + 1.6, y_top + h + 1.2)
    ax.text(x0, y_top + h + 0.7, "Task 1: density sees only the position; colour also sees the "
            "viewing direction", fontsize=FONT + 0.6, color=ACCENT, weight="bold")
    block(ax, xs[0], y_top, w["x"], h, boxes["x"], face="white")
    block(ax, xs[1], y_top, w["gx"], h, boxes["gx"])
    block(ax, xs[2], y_top, w["l1"], h, boxes["l1"])
    block(ax, xs[3], y_top, w["l2"], h, boxes["l2"])
    block(ax, xs[4], y_top, w["sig"], h, boxes["sig"], edge=GREEN, colour=GREEN)
    block(ax, xs[4], y_mid, w["feat"], h, boxes["feat"])
    block(ax, xs[0], y_low, w["d"], h, boxes["d"], face="white", edge=WARM, colour=WARM)
    block(ax, xs[1], y_low, w["gd"], h, boxes["gd"], edge=WARM, colour=WARM)
    block(ax, xs[5], y_mid, w["c1"], h, boxes["c1"], edge=WARM)
    block(ax, xs[6], y_mid, w["c2"], h, boxes["c2"], edge=WARM)
    for a, b, k in ((0, 1, "x"), (1, 2, "gx"), (2, 3, "l1"), (3, 4, "l2")):
        arrow(ax, (xs[a] + w[k], y_top + h / 2), (xs[b], y_top + h / 2))
    # trunk output h feeds both the density head and the feature layer
    arrow(ax, (xs[4] - gap / 2, y_top + h / 2), (xs[4] - gap / 2, y_mid + h / 2))
    arrow(ax, (xs[4] - gap / 2, y_mid + h / 2), (xs[4], y_mid + h / 2))
    note(ax, xs[4] - gap / 2 - 0.1, y_top - 0.25, "h", colour=INK, ha="right")
    arrow(ax, (xs[4] + w["sig"], y_top + h / 2), (xs[4] + w["sig"] + 0.9, y_top + h / 2),
          colour=GREEN)
    note(ax, xs[4] + w["sig"] + 1.0, y_top + h / 2, "density sigma\n(geometry: the same\n"
         "from every camera)", colour=GREEN)
    arrow(ax, (xs[4] + w["feat"], y_mid + h / 2), (xs[5], y_mid + h / 2))
    arrow(ax, (xs[0] + w["d"], y_low + h / 2), (xs[1], y_low + h / 2), colour=WARM)
    arrow(ax, (xs[1] + w["gd"], y_low + h / 2), (xs[5] + w["c1"] / 2, y_low + h / 2), colour=WARM)
    arrow(ax, (xs[5] + w["c1"] / 2, y_low + h / 2), (xs[5] + w["c1"] / 2, y_mid), colour=WARM)
    arrow(ax, (xs[5] + w["c1"], y_mid + h / 2), (xs[6], y_mid + h / 2))
    arrow(ax, (xs[6] + w["c2"], y_mid + h / 2), (xs[6] + w["c2"] + 0.6, y_mid + h / 2),
          colour=WARM)
    note(ax, xs[6] + w["c2"] / 2, y_mid - 0.45, "colour rgb\n(may change\nwith the camera)",
         colour=WARM, ha="center")
    save(fig, "nerf_architecture.png")


# --------------------------------------------------------------------------- 2. Task 2 sampling
def volume_weights(t, surface=4.1, thickness=0.12, density=12.0):
    """Weights w_i = T_i alpha_i of a made-up ray that meets one opaque surface at t = surface."""
    sigma = density * np.exp(-0.5 * ((t - surface) / thickness) ** 2)
    delta = np.append(np.diff(t), 1e10)
    alpha = 1 - np.exp(-sigma * delta)
    transmittance = np.cumprod(np.append(1.0, 1 - alpha + 1e-10))[:-1]
    return alpha * transmittance


def sample_pdf(bins, weights, n_samples):
    """The starter's sample_pdf (deterministic branch), in NumPy, for one ray."""
    weights = weights + 1e-5
    pdf = weights / weights.sum()
    cdf = np.concatenate([[0.0], np.cumsum(pdf)])
    u = np.linspace(0.0, 1.0, n_samples)
    inds = np.searchsorted(cdf, u, side="right")
    below = np.maximum(inds - 1, 0)
    above = np.minimum(inds, len(cdf) - 1)
    denom = cdf[above] - cdf[below]
    denom = np.where(denom < 1e-5, 1.0, denom)
    t = (u - cdf[below]) / denom
    return bins[below] + t * (bins[above] - bins[below]), cdf, u


def hierarchical():
    n_c, n_f = 32, 64
    t_coarse = np.linspace(NEAR, FAR, n_c)
    w_coarse = volume_weights(t_coarse)
    t_mid = 0.5 * (t_coarse[1:] + t_coarse[:-1])
    t_fine, cdf, u = sample_pdf(t_mid, w_coarse[1:-1], n_f)
    t_all = np.sort(np.concatenate([t_coarse, t_fine]))

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.25, 0.6], "hspace": 0.45})
    ax = axes[0]
    ax.bar(t_coarse, w_coarse, width=(FAR - NEAR) / n_c * 0.8, color=ACCENT, alpha=0.75)
    ax.plot(t_coarse, np.full(n_c, -0.06), "|", color=ACCENT, ms=7)
    ax.set_ylabel("coarse weight")
    ax.set_title("1. Coarse pass: 32 evenly spaced samples; the weights w = T alpha peak where "
                 "the ray meets the surface", loc="left", color=INK)
    ax.set_ylim(-0.12, 1.05)

    ax = axes[1]
    # sample_pdf interpolates linearly between CDF knots placed at the bin midpoints
    ax.plot(t_mid, cdf, color=INK, lw=1.2, label="CDF of the coarse weights")
    picks = np.linspace(0, n_f - 1, 9).astype(int)
    for k in picks:
        ax.plot([NEAR, t_fine[k]], [u[k], u[k]], color=WARM, lw=0.6, alpha=0.8)
        ax.plot([t_fine[k], t_fine[k]], [u[k], 0], color=WARM, lw=0.6, alpha=0.8)
    ax.set_ylabel("cumulative\nprobability u")
    ax.set_title("2. sample_pdf (provided): evenly spaced u in [0, 1], read across to the CDF "
                 "and down to a depth t", loc="left", color=INK)
    ax.legend(loc="upper left", frameon=False)

    ax = axes[2]
    ax.plot(t_coarse, np.full(n_c, 1.0), "|", color=ACCENT, ms=8)
    ax.plot(t_fine, np.full(n_f, 0.0), "|", color=WARM, ms=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["64 fine", "32 coarse"])
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlabel("distance along the ray t (near = 2, far = 6)")
    ax.set_title(f"3. Fine pass: the fine network evaluates all {len(t_all)} depths, "
                 "sorted; most of them sit at the surface", loc="left", color=INK)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for a in axes:
        a.set_xlim(NEAR, FAR)
    fig.suptitle("Task 2 on one illustrative ray: the coarse network decides where the fine "
                 "network looks", x=0.02, ha="left", y=0.99, color=ACCENT, weight="bold",
                 fontsize=FONT + 1.2)
    save(fig, "hierarchical_sampling.png")


if __name__ == "__main__":
    architecture()
    hierarchical()
