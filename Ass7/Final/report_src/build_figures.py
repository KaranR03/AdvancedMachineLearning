r"""Render the explanatory figures that DDPM_CosineSchedule.ipynb embeds in its markdown.

These figures explain mechanisms rather than report results, so the submission spec does not
require a notebook to generate them (writing guide, section 6, the two-tier rule). They are drawn
here, saved as PNG, and embedded by tools/build_notebooks.py as base64 data URIs. A notebook can
therefore gain or change a figure without being re-executed, and nothing extra ships in the zip.

Every image of a digit is this project's own data, at a fixed index, noised with a seeded
generator, so the figures are deterministic. The schedules are computed here from the same
formulas the notebook uses. Schematic boxes are sized from their own text, so a longer label widens
its box instead of spilling over the border.

    architecture.png     the tutorial's conditional U-Net at 20 x 20 (section 4)
    training_step.png    one training step, and where the schedule enters it (section 5)
    sampling_steps.png   which steps a 10-step sampler visits, and one DDPM or DDIM step (section 6)

Run:  python tools/build_figures.py
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = "f:/document/IFN_680_Advanced_Machine_Learning_and_Applications"
BASE = f"{ROOT}/weeks/week-10/project-7-improving-ddpm"
OUT = f"{BASE}/tools/explanatory"
DATA = f"{ROOT}/data/mnist_custom.pt"

# House palette, writing guide section 6.
INK, ACCENT, WARM = "#1f2933", "#2f6f9f", "#c1553b"
MUTED, PALE, GREEN = "#7b8794", "#e8ecf1", "#3f7d58"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "font.family": "DejaVu Sans",
                     "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "dejavusans",
                     "axes.spines.top": False, "axes.spines.right": False})

# Schematics are drawn in units of half an inch, so a canvas W units wide is W / 2 inches.
UNIT = 0.5
FONT = 7.2
T = 1000


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white",
                pil_kwargs={"optimize": True})
    plt.close(fig)
    print(f"  {name:28s} {os.path.getsize(path) / 1024:6.1f} KB")


# --------------------------------------------------------------------------- the two schedules
def linear_alpha_bar():
    """The tutorial's linear_beta_schedule(1000), accumulated."""
    return torch.cumprod(1 - torch.linspace(0.0001, 0.02, T, dtype=torch.float64), 0).numpy()


def cosine_alpha_bar(s=0.008):
    """Nichol and Dhariwal's eq. 17, with beta clipped at 0.999, accumulated as the notebook does."""
    steps = torch.linspace(0, 1, T + 1, dtype=torch.float64)
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    betas = torch.clip(1 - (f[1:] / f[0]) / (f[:-1] / f[0]), max=0.999)
    return torch.cumprod(1 - betas, 0).numpy()


def sampling_steps(n_steps):
    return np.round(np.linspace(0, T - 1, n_steps)).astype(int)


# --------------------------------------------------------------------------- schematic helpers
def width_of(text, size=FONT):
    """Box width, in schematic units, that fits the longest line of text at this font size."""
    longest = max(len(line) for line in text.split("\n"))
    return longest * size * 0.0178 + 0.5


def height_of(text, size=FONT):
    return len(text.split("\n")) * size * 0.036 + 0.45


def block(ax, x, y, w, h, text, face=PALE, edge=MUTED, colour=INK, size=FONT, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=colour, weight=weight, linespacing=1.35)


def arrow(ax, start, end, colour=INK, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color=colour,
                                 linewidth=1.0, linestyle=(0, (3, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0))


def note(ax, x, y, text, colour=MUTED, size=FONT - 0.6, ha="left"):
    ax.text(x, y, text, fontsize=size, color=colour, ha=ha, va="center", linespacing=1.3)


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width * UNIT, height * UNIT))
    # The axes fill the figure, so one schematic unit really is UNIT inches and the text-width
    # estimate in width_of() holds; the default subplot margins would shrink every box by 23%.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def thumbnail(ax, image, x, y, size, title=None, colour=INK):
    """A 20 x 20 image drawn into schematic coordinates, with a thin frame and a caption."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, extent=(x, x + size, y, y + size),
              interpolation="nearest", zorder=2)
    ax.add_patch(FancyBboxPatch((x, y), size, size, boxstyle="square,pad=0", fill=False,
                                edgecolor=MUTED, linewidth=0.6, zorder=3))
    if title:
        ax.text(x + size / 2, y + size + 0.22, title, ha="center", va="bottom",
                fontsize=FONT - 0.6, color=colour, linespacing=1.25)


# --------------------------------------------------------------------------- 1. the U-Net
def architecture():
    enc = ["ConvBlock 1 -> 32\n32 x 20 x 20\n+ condition", "ConvBlock 32 -> 64\n64 x 10 x 10\n"
           "+ condition"]
    dec = ["concatenate: 32 + 32\nConvBlock 64 -> 32\n32 x 20 x 20\n+ condition",
           "concatenate: 64 + 64\nConvBlock 128 -> 32\n32 x 10 x 10\n+ condition"]
    bottleneck = "ConvBlock 64 -> 64\n64 x 5 x 5\n+ condition"
    source = "noisy image x_t\n1 x 20 x 20"
    output = "1 x 1 conv, 32 -> 1:\npredicted noise\n1 x 20 x 20"
    condition = ("step t / T -> small MLP -> 32 numbers\nclass c -> learned embedding -> 32 "
                 "numbers\nsum = the condition, added to every\nblock through its own "
                 "1 x 1 convolution")
    w = max(width_of(t) for t in enc + dec + [bottleneck])
    h = max(height_of(t) for t in enc + dec + [bottleneck])
    w_io = max(width_of(source), width_of(output))
    gap, margin = 0.75, 0.15
    x_in = margin
    x_enc = x_in + w_io + gap
    x_mid = x_enc + w + gap
    x_dec = x_mid + w + gap
    x_out = x_dec + w + gap
    total_w = x_out + w_io + margin
    rows = [5.7, 3.2, 0.7]
    fig, ax = canvas(total_w, rows[0] + h + 1.0)
    ax.text(margin, rows[0] + h + 0.55, "Tutorial 9.3's conditional U-Net at 20 x 20: two "
            "halvings down, two doublings up, a skip connection at each level", fontsize=FONT + 0.6,
            color=ACCENT, weight="bold")
    block(ax, x_in, rows[0] + (h - height_of(source)) / 2, w_io, height_of(source), source,
          face="white")
    block(ax, x_out, rows[0] + (h - height_of(output)) / 2, w_io, height_of(output), output,
          face="white", edge=WARM, colour=WARM)
    for level in (0, 1):
        block(ax, x_enc, rows[level], w, h, enc[level])
        block(ax, x_dec, rows[level], w, h, dec[level])
        # the skip connection hands the encoder's map straight across to the decoder
        arrow(ax, (x_enc + w, rows[level] + h * 0.72), (x_dec, rows[level] + h * 0.72),
              colour=GREEN, dashed=True)
        note(ax, x_mid + w / 2, rows[level] + h * 0.72 + 0.28, "skip connection",
             colour=GREEN, ha="center")
    block(ax, x_mid, rows[2], w, h, bottleneck, edge=ACCENT)
    arrow(ax, (x_in + w_io, rows[0] + h / 2), (x_enc, rows[0] + h / 2))
    arrow(ax, (x_dec + w, rows[0] + h / 2), (x_out, rows[0] + h / 2))
    # down the left side by average pooling, up the right side by nearest-neighbour upsampling
    arrow(ax, (x_enc + w * 0.3, rows[0]), (x_enc + w * 0.3, rows[1] + h))
    note(ax, x_enc + w * 0.36, (rows[0] + rows[1] + h) / 2, "average pool:\n20 -> 10")
    # the two diagonal arrows are labelled on the inside of the V, clear of every box
    arrow(ax, (x_enc + w * 0.5, rows[1]), (x_mid, rows[2] + h * 0.5))
    note(ax, x_enc + w * 0.5 + 0.9, rows[1] - 0.45, "average pool:\n10 -> 5", ha="left")
    arrow(ax, (x_mid + w, rows[2] + h * 0.5), (x_dec + w * 0.5, rows[1]))
    note(ax, x_dec + w * 0.5 - 0.9, rows[1] - 0.45, "upsample:\n5 -> 10", ha="right")
    arrow(ax, (x_dec + w * 0.7, rows[1] + h), (x_dec + w * 0.7, rows[0]))
    note(ax, x_dec + w * 0.64, (rows[0] + rows[1] + h) / 2, "upsample:\n10 -> 20", ha="right")
    small = FONT - 0.4
    block(ax, x_in, 0.25, width_of(condition, small), height_of(condition, small), condition,
          face="white", edge=ACCENT, colour=ACCENT, size=small)
    save(fig, "architecture.png")


# --------------------------------------------------------------------------- 2. one training step
def training_step(images, labels):
    digit = images[labels == 3][0, 0] * 2 - 1
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(digit.shape, generator=generator, dtype=torch.float64)
    step = 500
    noised = {}
    for name, alpha_bar in (("linear", linear_alpha_bar()), ("cosine", cosine_alpha_bar())):
        a = alpha_bar[step]
        noised[name] = (math.sqrt(a) * digit + math.sqrt(1 - a) * noise, a)

    boxes = ["a training digit x_0\nand its class c = 3",
             "draw a step t (here 500)\nand noise eps ~ N(0, I)",
             "q_sample: the schedule\nsets how much of x_0\nsurvives at step t",
             "UNet_cond(x_t, t, c)\npredicts the noise:\neps_hat",
             "loss = mean of\n(eps_hat - eps)^2;\nAdam updates the\nnetwork's weights"]
    widths = [width_of(text) for text in boxes]
    h = max(height_of(text) for text in boxes)
    gap, margin, size = 0.6, 0.15, 1.5
    xs = [margin]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + gap)
    total_w = xs[-1] + widths[-1] + margin
    y_box, y_img = 0.9, 0.9 + h + 0.55
    fig, ax = canvas(total_w, y_img + size + 1.6)
    ax.text(margin, y_img + size + 1.15, "One training step. The schedule enters in one place: "
            "how noisy x_t is at step t", fontsize=FONT + 0.6, color=ACCENT, weight="bold")
    for i, (x, w, text) in enumerate(zip(xs, widths, boxes)):
        highlight = i == 2
        block(ax, x, y_box, w, h, text, face="white" if highlight else PALE,
              edge=WARM if highlight else MUTED, colour=WARM if highlight else INK)
        if i:
            arrow(ax, (xs[i - 1] + widths[i - 1], y_box + h / 2), (x, y_box + h / 2))
    thumbnail(ax, ((digit + 1) / 2).numpy(), xs[0] + (widths[0] - size) / 2, y_img, size,
              "x_0")
    thumbnail(ax, ((noise.clamp(-2, 2) + 2) / 4).numpy(), xs[1] + (widths[1] - size) / 2, y_img,
              size, "eps")
    pair = xs[2] + (widths[2] - 2 * size - 0.25) / 2
    for k, (name, colour) in enumerate((("linear", WARM), ("cosine", ACCENT))):
        image, a = noised[name]
        thumbnail(ax, ((image.clamp(-1, 1) + 1) / 2).numpy(), pair + k * (size + 0.25), y_img,
                  size, f"x_t, {name}\nsignal left {a:.2f}", colour=colour)
    note(ax, margin, 0.35, "x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps.  At t = 500 "
         "the linear schedule has already removed most of the digit; the cosine schedule keeps "
         "more of it.", size=FONT - 0.2, colour=INK)
    save(fig, "training_step.png")


# --------------------------------------------------------------------------- 3. sampling
def sampling_steps_figure():
    fig = plt.figure(figsize=(7.8, 3.0))
    ax = fig.add_axes([0.07, 0.16, 0.36, 0.70])
    steps = np.arange(T)
    visited = sampling_steps(10)
    for name, alpha_bar, colour in (("linear", linear_alpha_bar(), WARM),
                                    ("cosine", cosine_alpha_bar(), ACCENT)):
        noisy = int((alpha_bar[visited] < 0.01).sum())
        ax.plot(steps, alpha_bar, color=colour, lw=1.2,
                label=f"{name}: {noisy} of 10 steps below 0.01")
        ax.plot(visited, alpha_bar[visited], "o", color=colour, ms=3.5)
    ax.set(xlabel="step t", ylabel="alpha_bar (signal left)", xlim=(-10, T + 10), ylim=(0, 1.02))
    ax.set_title("The 10 steps a K = 10 sampler visits", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    # The right half is a schematic in half-inch units, like the other figures.
    width, height = 0.55 * 7.8 / UNIT, 3.0 / UNIT
    sch = fig.add_axes([0.45, 0.0, 0.55, 1.0])
    sch.set_xlim(0, width)
    sch.set_ylim(0, height)
    sch.set_aspect("equal")
    sch.axis("off")
    sch.text(0.3, height - 0.35, "One reverse step, from step t to an earlier step t'",
             fontsize=FONT + 0.6, color=ACCENT, weight="bold", va="center")
    top = ["x_t", "U-Net: noise\nestimate eps_hat",
           "clean-image estimate\nx0_hat, clipped\nto [-1, 1]"]
    widths = [width_of(text) for text in top]
    h = max(height_of(text) for text in top)
    y_top = height - 1.1 - h
    x = 0.3
    ends = []
    for i, (text, w) in enumerate(zip(top, widths)):
        block(sch, x, y_top, w, h, text, face="white" if i == 0 else PALE)
        if i:
            arrow(sch, (ends[-1], y_top + h / 2), (x, y_top + h / 2))
        ends.append(x + w)
        x += w + 0.45
    ddpm = ("DDPM, the tutorial's step:\nweighted average of\nx0_hat and x_t, plus\nfresh "
            "noise sqrt(beta) z")
    ddim = ("DDIM: x0_hat taken to\nthe noise level of t'\nusing eps_hat; no\nfresh noise "
            "drawn")
    w_low = max(width_of(ddpm), width_of(ddim))
    h_low = max(height_of(ddpm), height_of(ddim))
    y_low = 0.3
    left, right = 0.3, width - 0.3 - w_low
    block(sch, left, y_low, w_low, h_low, ddpm, face="white", edge=WARM, colour=WARM)
    block(sch, right, y_low, w_low, h_low, ddim, face="white", edge=ACCENT, colour=ACCENT)
    source_x = ends[-1] - widths[-1] / 2
    arrow(sch, (source_x - 0.4, y_top), (left + w_low / 2, y_low + h_low))
    arrow(sch, (source_x + 0.2, y_top), (right + w_low / 2, y_low + h_low))
    save(fig, "sampling_steps.png")


if __name__ == "__main__":
    data = torch.load(DATA, weights_only=True)
    train_images, train_labels = data["train_images"].double(), data["train_labels"]
    print(f"writing to {OUT}")
    architecture()
    training_step(train_images, train_labels)
    sampling_steps_figure()
