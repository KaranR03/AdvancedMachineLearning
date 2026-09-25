r"""Render the explanatory figures that the training notebooks embed in their markdown.

These figures explain mechanisms rather than report results, so the submission spec does not
require a notebook to generate them (writing guide, section 6, the two-tier rule). They are drawn
here, saved as PNG, and embedded by tools/build_notebooks.py as base64 data URIs. A notebook can
therefore gain or change a figure without being re-executed, and nothing extra ships in the zip.

Every figure that shows data is drawn from this project's own dataset, at fixed indices, so it is
deterministic and matches what the notebooks train on. Schematic boxes are sized from their own
text, so a longer label widens its box instead of spilling over the border.

Run:  python tools/build_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = "f:/document/IFN_680_Advanced_Machine_Learning_and_Applications"
BASE = f"{ROOT}/weeks/week-09/project-6-sharpness-quest"
OUT = f"{BASE}/tools/explanatory"
DATA = f"{ROOT}/data/mnist_custom.pt"

# House palette, writing guide section 6.
INK, ACCENT, WARM = "#1f2933", "#2f6f9f", "#c1553b"
MUTED, PALE, GREEN = "#7b8794", "#e8ecf1", "#3f7d58"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "font.family": "DejaVu Sans",
                     "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})

LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)

# Schematics are drawn in units of half an inch, so a canvas W units wide is W / 2 inches.
UNIT = 0.5
FONT = 7.2


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white",
                pil_kwargs={"optimize": True})
    plt.close(fig)
    print(f"  {name:28s} {os.path.getsize(path) / 1024:6.1f} KB")


def laplacian(images):
    """Valid 3x3 Laplacian response, the same operator the notebooks use."""
    return F.conv2d(images, LAPLACE)


def gaussian_blur(images, sigma=1.0):
    radius = 3
    grid = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-grid ** 2 / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, 2 * radius + 1, -1)
    # Replicate padding keeps the white background white at the border.
    return F.conv2d(F.pad(images, (radius,) * 4, mode="replicate"), kernel)


# --------------------------------------------------------------------------- 1. why MSE blurs
def why_mse_blurs(images, labels):
    sevens = images[labels == 7]
    picks = [0, 1, 2]
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.1))
    for ax, index in zip(axes[:3], picks):
        ax.imshow(sevens[index, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"a real 7\n(number {index + 1})")
    axes[3].imshow(sevens[picks].mean(0)[0], cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("pixel mean of\nthose three", color=WARM)
    axes[4].imshow(sevens.mean(0)[0], cmap="gray", vmin=0, vmax=1)
    axes[4].set_title(f"pixel mean of all\n{len(sevens):,} training 7s", color=WARM)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("The pixel-wise average of several plausible digits is a blur, not a digit",
                 fontsize=9, y=1.07)
    save(fig, "why_mse_blurs.png")


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
            color=colour, weight=weight, linespacing=1.3)


def arrow(ax, start, end, colour=INK, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color=colour,
                                 linewidth=1.0, linestyle=(0, (3, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0))


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


# --------------------------------------------------------------------------- 2. the cVAE
def architecture():
    encoder = ["image x\n1 x 20 x 20", "conv 3x3, stride 2\n16 x 10 x 10",
               "conv 3x3, stride 2\n32 x 5 x 5", "conv 3x3, stride 2\n64 x 3 x 3",
               "conv 3x3 (was 4x4)\n256 x 1 x 1", "+ class c (10)\nlinear: mu, log var\n7 + 7"]
    decoder = ["sigmoid\nx_hat, 1 x 20 x 20", "convT 3x3, stride 2\n1 x 20 x 20",
               "convT 3x3, stride 2\n16 x 10 x 10", "convT 3x3, stride 2\n32 x 5 x 5",
               "convT 3x3 (was 4x4)\n64 x 3 x 3", "z (7) + class c (10)\nlinear: 256"]
    reparam = "z = mu + sigma * eps\neps ~ N(0, I)"
    width = max(width_of(t) for t in encoder + decoder + [reparam])
    height = max(height_of(t) for t in encoder + decoder)
    gap, margin = 0.45, 0.15
    xs = [margin + i * (width + gap) for i in range(6)]
    total_w = xs[-1] + width + margin
    fig, ax = canvas(total_w, 7.4)
    top, bottom = 5.2, 1.2
    ax.text(margin, top + height + 0.45, "Encoder: the tutorial's convolutions; only the last "
            "kernel changes, because 3 x 3 cannot fit a 4 x 4 kernel", fontsize=FONT,
            color=ACCENT)
    for i, (x, text) in enumerate(zip(xs, encoder)):
        block(ax, x, top, width, height, text, face="white" if i in (0, 5) else PALE,
              edge=ACCENT if i == 5 else MUTED, weight="bold" if i == 4 else "normal")
        if i:
            arrow(ax, (xs[i - 1] + width, top + height / 2), (x, top + height / 2))
    middle = (top + bottom + height) / 2 - 0.55
    block(ax, xs[5], middle, width, 1.1, "z = mu + sigma * eps\neps ~ N(0, I)", face="white",
          edge=WARM, colour=WARM)
    arrow(ax, (xs[5] + width / 2, top), (xs[5] + width / 2, middle + 1.1))
    arrow(ax, (xs[5] + width / 2, middle), (xs[5] + width / 2, bottom + height))
    for i, (x, text) in enumerate(zip(xs, decoder)):
        block(ax, x, bottom, width, height, text, face="white" if i in (0, 5) else PALE,
              edge=ACCENT if i == 5 else MUTED, weight="bold" if i == 4 else "normal")
        if i:
            arrow(ax, (x, bottom + height / 2), (xs[i - 1] + width, bottom + height / 2))
    ax.text(margin, 0.35, "Decoder: the mirror image, 1 -> 3 -> 5 -> 10 -> 20  (convT = "
            "transposed convolution, which enlarges a feature map)", fontsize=FONT, color=ACCENT)
    save(fig, "architecture.png")


# --------------------------------------------------------------------------- 3. the discriminator
def discriminator():
    inputs = ["image: real x,\nor a fake x_hat\n1 x 20 x 20", "class c as 10\nconstant planes\n10 x 20 x 20"]
    chain = ["stack\n11 x 20 x 20", "conv 3x3, stride 2\nLeakyReLU(0.2)\n32 x 10 x 10",
             "conv 3x3, stride 2\nLeakyReLU(0.2)\n64 x 5 x 5", "flatten + linear\n1600 -> 1",
             "one logit:\n> 0 reads 'real'\n< 0 reads 'fake'"]
    height = max(height_of(t) for t in inputs + chain)
    gap, margin = 0.55, 0.15
    in_w = max(width_of(t) for t in inputs)
    x = margin + in_w + gap
    widths = [width_of(t) for t in chain]
    fig, ax = canvas(x + sum(widths) + gap * (len(chain) - 1) + margin, 2 * height + 1.6)
    low, high = 0.3, 0.3 + height + 0.35
    mid = (low + high) / 2
    block(ax, margin, high, in_w, height, inputs[0], face="white")
    block(ax, margin, low, in_w, height, inputs[1], face="white", edge=ACCENT)
    for i, (text, w) in enumerate(zip(chain, widths)):
        last = i == len(chain) - 1
        block(ax, x, mid, w, height, text, face="white" if last else PALE,
              edge=WARM if last else MUTED, colour=WARM if last else INK)
        if i == 0:
            arrow(ax, (margin + in_w, high + height / 2), (x, mid + height * 0.7))
            arrow(ax, (margin + in_w, low + height / 2), (x, mid + height * 0.3))
        else:
            arrow(ax, (previous, mid + height / 2), (x, mid + height / 2))
        previous = x + w
        x += w + gap
    ax.text(margin, high + height + 0.3, "Conditional discriminator: it judges an image together "
            "with the class the image is supposed to show", fontsize=FONT, color=ACCENT)
    save(fig, "discriminator.png")


# --------------------------------------------------------------------------- 4. one training step
def training_step():
    batch = "batch:\nimages x,\nclasses c"
    recon = "encode x -> z\ndecode z -> x_hat"
    prior = "draw z_p ~ N(0, I)\ndecode z_p -> x_p"
    step1 = ("Step 1: discriminator\nreal x -> label 1\nx_hat, x_p -> label 0\n"
             "(cVAE outputs detached)")
    step2 = ("Step 2: cVAE\n100 MSE + 0.1 KL + lambda L_G\nL_G: x_hat, x_p -> label 1\n"
             "the decoder learns to fool D")
    update1 = "update D only\n(its own AdamW)"
    update2 = "update encoder + decoder\n(the tutorial's AdamW)"
    gap, margin = 0.7, 0.15
    w0, w1 = width_of(batch), max(width_of(recon), width_of(prior))
    w2, w3 = max(width_of(step1), width_of(step2)), max(width_of(update1), width_of(update2))
    h1, h2 = height_of(step1), height_of(step2)
    x0 = margin
    x1 = x0 + w0 + gap
    x2 = x1 + w1 + gap
    x3 = x2 + w2 + gap
    total_w = x3 + w3 + margin
    fig, ax = canvas(total_w, h1 + h2 + 3.2)
    y2 = 1.3
    y1 = y2 + h2 + 0.5
    hr = height_of(recon)
    yr, yp = y1 + h1 - hr, y2
    block(ax, x0, (y1 + y2 + h1) / 2 - height_of(batch) / 2, w0, height_of(batch), batch,
          face="white")
    block(ax, x1, yr, w1, hr, recon)
    block(ax, x1, yp, w1, hr, prior)
    batch_mid = (y1 + y2 + h1) / 2
    arrow(ax, (x0 + w0, batch_mid + 0.2), (x1, yr + hr / 2))
    arrow(ax, (x0 + w0, batch_mid - 0.2), (x1, yp + hr / 2))
    block(ax, x2, y1, w2, h1, step1, face="white", edge=WARM)
    block(ax, x2, y2, w2, h2, step2, face="white", edge=GREEN)
    arrow(ax, (x1 + w1, yr + hr * 0.6), (x2, y1 + h1 * 0.6), colour=WARM, dashed=True)
    arrow(ax, (x1 + w1, yp + hr * 0.8), (x2, y1 + h1 * 0.2), colour=WARM, dashed=True)
    arrow(ax, (x1 + w1, yr + hr * 0.2), (x2, y2 + h2 * 0.8), colour=GREEN)
    arrow(ax, (x1 + w1, yp + hr * 0.4), (x2, y2 + h2 * 0.4), colour=GREEN)
    block(ax, x3, y1 + h1 / 2 - 0.55, w3, 1.1, update1, edge=WARM)
    block(ax, x3, y2 + h2 / 2 - 0.55, w3, 1.1, update2, edge=GREEN)
    arrow(ax, (x2 + w2, y1 + h1 / 2), (x3, y1 + h1 / 2), colour=WARM)
    arrow(ax, (x2 + w2, y2 + h2 / 2), (x3, y2 + h2 / 2), colour=GREEN)
    ax.text(margin, y1 + h1 + 0.4, "One batch, in the lecture's order: the discriminator "
            "steps first, then the cVAE", fontsize=FONT + 0.6, color=ACCENT, weight="bold")
    ax.text(margin, 0.3, "Dashed: detached inputs, so step 1 cannot change the cVAE.  L_G is "
            "computed on decode(z.detach()),\nso the adversarial gradient trains the decoder "
            "only; the encoder learns from MSE and KL, as in the baseline.",
            fontsize=FONT, color=INK, linespacing=1.4)
    save(fig, "training_step.png")


# --------------------------------------------------------------------------- 5. Laplacian variance
def laplacian_figure(images, labels):
    digit = images[labels == 3][0:1]
    blurred = gaussian_blur(digit, sigma=1.0)
    panels = [(digit, "a real 3"), (laplacian(digit), "its Laplacian"),
              (blurred, "the same 3, blurred"), (laplacian(blurred), "its Laplacian")]
    variance_sharp = laplacian(digit).var().item()
    variance_blurred = laplacian(blurred).var().item()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
    limit = float(laplacian(digit).abs().max())
    for ax, (image, title) in zip(axes, panels):
        if "Laplacian" in title:
            ax.imshow(image[0, 0], cmap="RdBu_r", vmin=-limit, vmax=limit)
        else:
            ax.imshow(image[0, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].set_xlabel(f"variance {variance_sharp:.3f}", color=ACCENT)
    axes[3].set_xlabel(f"variance {variance_blurred:.3f}", color=WARM)
    fig.suptitle("Sharp edges give large positive and negative responses; blur flattens them, "
                 "and the variance falls", fontsize=9, y=1.05)
    save(fig, "laplacian.png")


if __name__ == "__main__":
    data = torch.load(DATA, weights_only=True)
    train_images, train_labels = data["train_images"], data["train_labels"]
    print(f"writing to {OUT}")
    why_mse_blurs(train_images, train_labels)
    architecture()
    discriminator()
    training_step()
    laplacian_figure(train_images, train_labels)
