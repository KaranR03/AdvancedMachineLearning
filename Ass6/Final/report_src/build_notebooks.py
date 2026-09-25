r"""Generate the four Project 6 notebooks from one template.

    notebook/cVAE_Baseline.ipynb            Task 1: the Tutorial 8.3 cVAE on mnist_custom.pt
    notebook/cVAE_DiscriminatorLoss.ipynb   Task 2: the same cVAE with a discriminator loss
    notebook/DigitClassifier.ipynb          the standalone classifier used for evaluation
    notebook/main_report.ipynb              Task 3: every number and figure in the report

The model, data and metric code is written once here and placed byte-identical in every notebook
that needs it, which is what makes the two cVAEs comparable by construction (project README,
decisions D8 and D10).

The generated notebooks carry no outputs. Execute them on the Hub GPU node (README, Hub route) or
rehearse the whole pipeline at tiny sizes with --smoke and tools/smoke_test.py.

Run:  python tools/build_notebooks.py            full sizes, written to notebook/
      python tools/build_notebooks.py --smoke    tiny sizes, written to the job's scratch room
"""
import base64
import io
import json
import os
import re
import sys

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-09/project-6-sharpness-quest")
SMOKE = "--smoke" in sys.argv
ROOM = os.environ.get("CLAUDE_JOB_DIR", "C:/Users/Admin/.claude/jobs/8640b033") + "/tmp/smoke"
NBDIR = ROOM if SMOKE else f"{BASE}/notebook"
FIGURES = f"{BASE}/tools/explanatory"

# Sizes. The smoke run exercises every code path in a few minutes on a CPU; its numbers mean
# nothing and it never writes into notebook/.
if SMOKE:
    SIZES = dict(N_TOTAL=2500, N_DEV=500, EPOCHS=2, WINDOW=1, PRINT_EVERY=1, SEEDS="(0, 1)",
                 LAMBDAS="(0.1, 1.0)", CLASSIFIER_EPOCHS=1,
                 # Six optimiser steps cannot beat these bars, so the smoke run relaxes them.
                 TRIVIAL_BAR="10 * trivial_loss", CHANCE_BAR="2 * np.log(NUM_CLASSES)")
else:
    SIZES = dict(N_TOTAL=60000, N_DEV=10000, EPOCHS=150, WINDOW=10, PRINT_EVERY=10,
                 SEEDS="(0, 1, 2)", LAMBDAS="(0.03, 0.1, 0.3, 1.0)", CLASSIFIER_EPOCHS=5,
                 TRIVIAL_BAR="trivial_loss", CHANCE_BAR="np.log(NUM_CLASSES) / 2")


def figure(name, alt):
    """An explanatory PNG from tools/build_figures.py, embedded as a base64 data URI."""
    with open(f"{FIGURES}/{name}", "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"![{alt}](data:image/png;base64,{encoded})"


# =========================================================================== shared code cells

CELL_IMPORTS = r'''# The Week 8 tutorial's stack. json carries results from the training notebooks to
# main_report.ipynb, and make_grid lays out digit grids the way the tutorial's figures do.
import json
__IMPORT_OS__import platform
__IMPORT_TIME__
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
__IMPORT_OPTIM__import torchvision
__IMPORT_DATA__from torchvision.utils import make_grid

# The brief asks for the IFN680 GPU environment. The same code runs on a CPU when no GPU is
# visible, so the notebook runs anywhere and every tensor is placed with .to(device).
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device      : {device}")
print(f"python      : {platform.python_version()}")
print(f"torch       : {torch.__version__}")
print(f"torchvision : {torchvision.__version__}")
print(f"numpy       : {np.__version__}")

# One seed for everything: the data split, the initial weights and every random draw.
SEED = 0
np.random.seed(SEED)
_ = torch.manual_seed(SEED)'''

CELL_STYLE = r'''# One palette for every figure in the project, so the figures read as a set.
INK, ACCENT, WARM = "#1f2933", "#2f6f9f", "#c1553b"
MUTED, PALE, GREEN = "#7b8794", "#e8ecf1", "#3f7d58"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.edgecolor": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})'''

CELL_DATA = r'''# mnist_custom.pt is the dataset supplied with the brief: MNIST resized to 20 x 20 and
# inverted. weights_only=True loads the tensors and refuses to run any code stored in the file.
data = torch.load("mnist_custom.pt", weights_only=True)
train_images, train_labels = data["train_images"], data["train_labels"]

# main_report.ipynb needs the test split and nothing else, so it is written to its own file
# here. No training notebook scores it or makes a prediction on it.
if not os.path.exists("mnist_custom_test.pt"):
    test_split = {key: data[key] for key in ("test_images", "test_labels")}
    torch.save(test_split, "mnist_custom_test.pt")
print(f"loaded mnist_custom.pt: {', '.join(data.keys())}")'''

CELL_AUDIT = r'''# What the file actually holds, measured before anything is trained on it.
border = torch.cat([train_images[..., 0, :], train_images[..., -1, :],
                    train_images[..., :, 0], train_images[..., :, -1]], dim=-1)
print(f"train_images : {tuple(train_images.shape)}, {train_images.dtype}")
print(f"train_labels : {tuple(train_labels.shape)}, {train_labels.dtype}")
print(f"pixel range  : {train_images.min():.3f} to {train_images.max():.3f}")
print(f"class counts : {torch.bincount(train_labels, minlength=10).tolist()}")
# The background is white (1.0) and the strokes dark: the reverse of the tutorial's MNIST.
print(f"pixels exactly 1.0 : {(train_images == 1).float().mean():.3f}")
print(f"mean border pixel  : {border.mean():.4f}")'''

CELL_SPLIT = r'''# The development split: a seeded random draw of N_DEV training images, held out to judge
# convergence and to choose settings. The final models retrain on all N_TOTAL images.
N_TOTAL, N_DEV = __N_TOTAL__, __N_DEV__
split_generator = torch.Generator().manual_seed(SEED)
order = torch.randperm(len(train_images), generator=split_generator)[:N_TOTAL]
dev_index, fit_index = order[:N_DEV], order[N_DEV:]
print(f"fit split         : {len(fit_index):,} images")
print(f"development split : {len(dev_index):,} images")
dev_counts = torch.bincount(train_labels[dev_index], minlength=10).tolist()
print(f"development class counts : {dev_counts}")
print(f"refit set (fit + development) : {len(order):,} images")'''

CELL_ENCODER = r'''# The tutorial's encoder with one change: the last convolution has a 3x3 kernel instead of
# 4x4, because on 20 x 20 images the map entering it is 3 x 3 (20 -> 10 -> 5 -> 3).
class cEncoder(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(cEncoder, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        # Define the encoder layers
        self.encoder_backbone = nn.Sequential(
            # Layer 1: 1x20x20 -> 16x10x10
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 2: 16x10x10 -> 32x5x5
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 3: 32x5x5 -> 64x3x3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 4: 64x3x3 -> 256x1x1 (the tutorial's kernel_size=4 fitted its 4x4 map)
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Flatten and project to z_dim
            nn.Flatten(),  # 256*1*1 = 256
        )
        self.encoder_head = nn.Sequential(
            nn.Linear(256 + num_classes, 2 * z_dim)
        )

    def forward(self, x, cond):
        # x shape: (batch_size, 1, 20, 20); cond: (batch_size, num_classes), one-hot
        x = self.encoder_backbone(x)
        x = torch.cat([x, cond], dim=1)
        x = self.encoder_head(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar'''

CELL_DECODER = r'''# The tutorial's decoder with the mirror-image change: the first transposed convolution has
# a 3x3 kernel, so the maps grow 1 -> 3 -> 5 -> 10 -> 20 and end at the image size.
class cDecoder(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(cDecoder, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.decoder_input = nn.Sequential(
            # Project latent vector to a feature map
            nn.Linear(z_dim + num_classes, 256),
            nn.ReLU(),
        )
        self.decoder_backbone = nn.Sequential(
            nn.Unflatten(1, (256, 1, 1)),  # Reshape to (256, 1, 1)
            # Layer 1: 256x1x1 -> 64x3x3 (the tutorial's kernel_size=4 gave 64x4x4)
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Layer 2: 64x3x3 -> 32x5x5
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            # Layer 3: 32x5x5 -> 16x10x10
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Layer 4: 16x10x10 -> 1x20x20
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1] for pixel values
        )

    def forward(self, z, cond):
        z = torch.cat([z, cond], dim=1)
        z = self.decoder_input(z)
        x_hat = self.decoder_backbone(z)
        return x_hat'''

CELL_LOSS = r'''# The tutorial's loss, unchanged: 100 x pixel MSE plus beta x KL, with beta = 0.1.
def compute_loss(x, x_hat, mu, logvar, beta=0.1):
    """
    Compute the reconstruction loss for the autoencoder.
    """

    # Reconstruction loss (MSE)
    recon_loss = 100 * F.mse_loss(x_hat, x, reduction='mean')

    # For standard autoencoder, KL divergence is 0
    if beta == 0:
        kl_loss = 0.0
    else:
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss'''

CELL_CVAE = r'''# The tutorial's cVAE, unchanged. forward() draws z with the reparameterisation trick, so the
# random draw stays outside the path the gradient has to flow through.
class cVAE(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(cVAE, self).__init__()
        self.encoder = cEncoder(z_dim, num_classes)
        self.decoder = cDecoder(z_dim, num_classes)
        self.z_dim = z_dim
        self.num_classes = num_classes

    def encode(self, x, cond):
        return self.encoder(x, cond)

    def decode(self, z, cond):
        return self.decoder(z, cond)

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        x_recon = self.decoder(z, cond)
        return x_recon, mu, logvar'''

CELL_SETTINGS = r'''# Settings shared by both cVAE notebooks, so the two runs are comparable by construction.
Z_DIM, NUM_CLASSES = 7, 10  # the brief fixes 7 latent dimensions (the tutorial used 8)
EPOCHS = __EPOCHS__  # the tutorial trains for 50, too few to settle on this data (4.4)
SEEDS = __SEEDS__  # the final models are retrained once per seed
# Progress is printed every PRINT_EVERY epochs; convergence compares windows of WINDOW epochs.
PRINT_EVERY, WINDOW = __PRINT_EVERY__, __WINDOW__'''

CELL_BATCHES = r'''# The tutorial's batching, over tensors instead of torchvision's MNIST class. A seeded
# generator fixes the shuffle order, so both cVAEs see the same batches in the same order.
BATCH_SIZE = 1000


def make_loader(index, shuffle, seed=SEED):
    dataset = TensorDataset(train_images[index], train_labels[index])
    order_generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
                      generator=order_generator)


def get_batch(batch, device, f_train=True, num_classes=10):
    # The tutorial's helper: move a batch to the device and one-hot encode its labels
    x = batch[0].to(device)
    cond = batch[1].to(device)
    cond = F.one_hot(cond, num_classes=num_classes).float()
    return x, cond


fit_dataloader = make_loader(fit_index, shuffle=True)
dev_dataloader = make_loader(dev_index, shuffle=False)
print(f"{len(fit_dataloader)} fit batches and {len(dev_dataloader)} development batches "
      f"of {BATCH_SIZE}")'''

CELL_MODEL_CHECK = r'''# One forward pass on a development batch confirms the adapted kernels restore the 20 x 20
# shape end to end, before any training time is spent.
model = cVAE(Z_DIM, num_classes=NUM_CLASSES).to(device)
x, cond = get_batch(next(iter(dev_dataloader)), device)
x_hat, mu, logvar = model(x, cond)
print(f"x {tuple(x.shape)} -> mu {tuple(mu.shape)} -> x_hat {tuple(x_hat.shape)}")
print(f"cVAE parameters: {sum(p.numel() for p in model.parameters()):,}")'''

CELL_EVALUATE = r'''# The tutorial's evaluation pass, pointed at the development split. As in the tutorial the
# image is rebuilt from mu, the centre of the encoder's distribution, so no random draw is
# involved and the score is repeatable.
def evaluate_loss(model, dataloader):
    model.eval()
    totals, count = np.zeros(3), 0
    with torch.no_grad():
        for batch in dataloader:
            x, cond = get_batch(batch, device, f_train=False)
            mu, logvar = model.encode(x, cond)
            x_hat = model.decode(mu, cond)
            losses = compute_loss(x, x_hat, mu, logvar)
            totals += len(x) * np.array([float(value) for value in losses])
            count += len(x)
    return totals / count  # total, reconstruction, KL'''

CELL_HISTORY = r'''# Bookkeeping for the training curves, which are saved to the history file for main_report.
def new_history(adversarial=False):
    keys = ["train_total", "train_recon", "train_kl", "dev_total", "dev_recon", "dev_kl",
            "seconds", "lr"]
    if adversarial:
        keys += ["train_d_loss", "train_g_loss", "dev_d_accuracy"]
    return {key: [] for key in keys}


def record(history, split, values):
    for name, value in zip(("total", "recon", "kl"), values):
        history[f"{split}_{name}"].append(float(value))


def report_epoch(epoch, nb_epoch, history):
    # One progress line every PRINT_EVERY epochs, plus the first and the last.
    if epoch == 0 or (epoch + 1) % PRINT_EVERY == 0 or epoch + 1 == nb_epoch:
        line = f"epoch {epoch + 1:3d}/{nb_epoch}  train loss {history['train_total'][-1]:.4f}"
        if history["dev_total"]:
            line += f"  dev recon {history['dev_recon'][-1]:.4f}"
        if history.get("dev_d_accuracy"):
            line += f"  D accuracy {history['dev_d_accuracy'][-1]:.3f}"
        print(line + f"  ({history['seconds'][-1]:.1f} s)")


def vae_optimizer(parameters):
    # The tutorial's optimiser settings, held in one place so both cVAEs use exactly these.
    return optim.AdamW(parameters, lr=0.001, eps=1e-5, fused=False, betas=(0.75, 0.99))


def cosine_schedule(optimizer, nb_epoch):
    # The learning rate falls from its starting value to 0 along half a cosine over the run,
    # so the last epochs take ever smaller steps and the weights settle instead of circling.
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epoch)'''

CELL_CONVERGENCE = r'''# A run has converged when its mean development reconstruction loss over the last WINDOW
# epochs is within 1% of its mean over the WINDOW epochs before them.
def convergence(curve, window):
    last, previous = np.mean(curve[-window:]), np.mean(curve[-2 * window:-window])
    change = (last - previous) / previous
    return {"last_mean": float(last), "previous_mean": float(previous),
            "relative_change": float(change), "within_1pct": bool(abs(change) < 0.01)}'''

CELL_SAMPLE = r'''# Decoding helpers shared by every notebook that draws digits. Reconstructions decode mu, as
# the tutorial's test pass does; samples decode codes drawn from N(0, I).
def reconstruct(model, images, labels, batch_size=1000):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = images[start:start + batch_size].to(device)
            cond = F.one_hot(labels[start:start + batch_size].to(device), NUM_CLASSES).float()
            mu, _ = model.encode(x, cond)
            outputs.append(model.decode(mu, cond).cpu())
    return torch.cat(outputs)


def decode_latents(model, z, labels, batch_size=1000):
    """Decode latent codes z under classes labels: conditional sampling when z ~ N(0, I)."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            cond = F.one_hot(labels[start:start + batch_size].to(device), NUM_CLASSES).float()
            outputs.append(model.decode(z[start:start + batch_size].to(device), cond).cpu())
    return torch.cat(outputs)


def sample_grid(model, z_grid):
    """Rows are the digits 0 to 9 and columns the codes in z_grid, as in the brief's example,
    so each column is one latent code decoded under all ten classes."""
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(len(z_grid))
    images = decode_latents(model, z_grid.repeat(NUM_CLASSES, 1), labels)
    return images, labels.cpu()'''

CELL_LAPLACIAN = r'''# Sharpness as the variance of the Laplacian (Pech-Pacheco et al., 2000). The 3x3 kernel
# responds to curvature in brightness: large at a crisp edge, near zero in a smooth blur.
LAPLACE_WEIGHTS = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
LAPLACE_KERNEL = torch.tensor(LAPLACE_WEIGHTS, device=device).view(1, 1, 3, 3)


def laplacian_variance(images, batch_size=2000):
    """Per-image variance of the Laplacian response (valid convolution, 18 x 18 values)."""
    values = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            response = F.conv2d(images[start:start + batch_size].to(device), LAPLACE_KERNEL)
            values.append(response.flatten(1).var(dim=1).cpu())
    return torch.cat(values).double().numpy()


# A second view of blur: the share of pixels that are neither background nor stroke.
MID_GREY = (0.1, 0.9)


def midgrey_fraction(images):
    grey = (images > MID_GREY[0]) & (images < MID_GREY[1])
    return grey.flatten(1).float().mean(dim=1).double().numpy()'''

CELL_DISCRIMINATOR = r'''# The conditional discriminator. The class enters as ten constant image planes, so every
# convolution sees the image together with the class it is supposed to show.
class Discriminator(nn.Module):
    def __init__(self, num_classes=10, image_size=20):
        super().__init__()
        self.image_size = image_size
        self.net = nn.Sequential(
            # 11x20x20 -> 32x10x10: the image plus one plane per class
            nn.Conv2d(1 + num_classes, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 32x10x10 -> 64x5x5
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 1600 features -> one logit: above 0 reads "real", below 0 reads "fake"
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 1),
        )

    def forward(self, x, cond):
        planes = cond[:, :, None, None].expand(-1, -1, self.image_size, self.image_size)
        return self.net(torch.cat([x, planes], dim=1))'''

CELL_CLASSIFIER = r'''# The evaluation classifier, in the style of lecture 9 p42's: two convolution blocks and a
# 128-unit layer. features() returns that layer, the space the Frechet distances use.
class DigitClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x10x10
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64x5x5
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128), nn.ReLU(),
        )
        self.output_layer = nn.Linear(128, num_classes)

    def features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        return self.output_layer(self.features(x))'''

CELL_CPU_STATE = r'''# Checkpoints are saved from the CPU, so they load on a machine without a GPU.
def cpu_state(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}'''

# =========================================================================== baseline cells

CELL_DATA_FIGURE = r'''# One training image per class, and every pixel value in the training set. Almost every pixel
# is pure white or near black, so a blurred digit, full of in-between greys, is easy to spot.
first_of_class = [int((train_labels == c).nonzero()[0]) for c in range(10)]
fig, (ax_digits, ax_hist) = plt.subplots(1, 2, figsize=(9, 2.2),
                                         gridspec_kw={"width_ratios": [3, 1.2]})
panel = make_grid(train_images[first_of_class], nrow=10, pad_value=1)
ax_digits.imshow(panel[0], cmap="gray", vmin=0, vmax=1)
ax_digits.set_title("the first training image of each class, 0 to 9")
ax_digits.axis("off")
ax_hist.hist(train_images.flatten().numpy(), bins=50, color=ACCENT, log=True)
ax_hist.set_xlabel("pixel value")
ax_hist.set_ylabel("count (log scale)")
plt.show()'''

CELL_TRAIN_FN = r'''# The tutorial's training loop. Changes from it: losses are summed with .item(), which keeps
# the running total out of the gradient graph; the evaluation split is the development split;
# the learning rate follows the cosine schedule; and the progress bars are dropped, because
# they write to stderr.
def training_loop(model, train_dataloader, dev_dataloader, nb_epoch):
    optimizer = vae_optimizer(model.parameters())
    scheduler = cosine_schedule(optimizer, nb_epoch)
    history = new_history()
    for epoch in range(nb_epoch):
        started = time.perf_counter()
        model.train()
        totals, count = np.zeros(3), 0
        for batch in train_dataloader:
            x, cond = get_batch(batch, device, f_train=True)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x, cond)
            total_loss, recon_loss, kl_loss = compute_loss(x, x_hat, mu, logvar)
            total_loss.backward()
            optimizer.step()
            totals += len(x) * np.array([total_loss.item(), recon_loss.item(), kl_loss.item()])
            count += len(x)
        record(history, "train", totals / count)
        if dev_dataloader is not None:
            record(history, "dev", evaluate_loss(model, dev_dataloader))
        history["seconds"].append(time.perf_counter() - started)
        # The rate this epoch used, then one step down the schedule for the next epoch
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        report_epoch(epoch, nb_epoch, history)
    return history'''

CELL_DEV_RUN = r'''# Development run: train on the fit split and score the development split after every epoch.
# Seeding before the model is built fixes its initial weights; the other notebook does the same.
_ = torch.manual_seed(SEED)
model = cVAE(Z_DIM, num_classes=NUM_CLASSES).to(device)
dev_history = training_loop(model, fit_dataloader, dev_dataloader, nb_epoch=EPOCHS)'''

CELL_DEV_CURVES = r'''# The convergence statistic of section 4.4, and the curves it summarises.
dev_convergence = convergence(dev_history["dev_recon"], WINDOW)
print(f"dev reconstruction loss, mean of the last {WINDOW} epochs : "
      f"{dev_convergence['last_mean']:.4f}")
print(f"mean of the {WINDOW} epochs before              : {dev_convergence['previous_mean']:.4f}")
print(f"relative change {dev_convergence['relative_change']:+.4f}, "
      f"within 1%: {dev_convergence['within_1pct']}")
epochs = np.arange(1, EPOCHS + 1)
fig, ax = plt.subplots(figsize=(5.5, 2.6))
ax.plot(epochs, dev_history["train_recon"], color=MUTED, label="fit split (z sampled)")
ax.plot(epochs, dev_history["dev_recon"], color=ACCENT, label="development split (from mu)")
ax.set_xlabel("epoch")
ax.set_ylabel("reconstruction loss, 100 x MSE")
ax.legend(frameon=False)
# The learning rate on a second axis: the curves flatten as the schedule approaches 0
twin = ax.twinx()
twin.plot(epochs, dev_history["lr"], color=WARM, linestyle="--")
twin.set_ylabel("learning rate (dashed)")
twin.spines["right"].set_visible(True)
plt.show()'''

CELL_DEV_RECON = r'''# Development images (top) and their reconstructions (bottom), one per class.
dev_images, dev_labels = train_images[dev_index], train_labels[dev_index]
picks = [int((dev_labels == c).nonzero()[0]) for c in range(10)]
rebuilt = reconstruct(model, dev_images[picks], dev_labels[picks])
panel = make_grid(torch.cat([dev_images[picks], rebuilt]), nrow=10, pad_value=1)
fig, ax = plt.subplots(figsize=(6, 1.5))
ax.imshow(panel[0], cmap="gray", vmin=0, vmax=1)
ax.axis("off")
plt.show()'''

CELL_REFIT = r'''# The final models: the same settings, retrained on all N_TOTAL training images, once per seed.
# Each seed also reshuffles the batches, so the spread across seeds shows how much the result
# depends on the luck of a single run.
refit_loaders = {seed: make_loader(order, shuffle=True, seed=seed) for seed in SEEDS}
checkpoint, refit_histories = {}, {}
for seed in SEEDS:
    print(f"--- seed {seed}")
    _ = torch.manual_seed(seed)
    model = cVAE(Z_DIM, num_classes=NUM_CLASSES).to(device)
    refit_histories[str(seed)] = training_loop(model, refit_loaders[seed], None, EPOCHS)
    checkpoint[f"seed_{seed}"] = cpu_state(model)'''

CELL_TRIVIAL = r'''# Every final model must beat the trivial predictor that answers every image with the mean
# training image. Its loss, 100 x MSE, is the bar each seed's last epoch has to clear.
refit_images = train_images[order]
mean_image = refit_images.mean(dim=0, keepdim=True)
trivial_loss = 100 * F.mse_loss(mean_image.expand_as(refit_images), refit_images).item()
print(f"trivial predictor (mean image): {trivial_loss:.4f}")
for seed, history in refit_histories.items():
    print(f"seed {seed}: last-epoch reconstruction loss {history['train_recon'][-1]:.4f}")
    assert history["train_recon"][-1] < __TRIVIAL_BAR__, f"seed {seed} did not learn"'''

CELL_SAVE_BASELINE = r'''# The weights (one set per seed) and every curve. main_report.ipynb reads both files.
torch.save(checkpoint, "cVAE_Baseline.pth")
history_record = {
    "model": "cVAE_Baseline", "lambda": 0.0, "epochs": EPOCHS, "seeds": list(SEEDS),
    "window": WINDOW, "n_fit": len(fit_index), "n_dev": len(dev_index), "n_refit": len(order),
    "dev": dev_history, "convergence": dev_convergence, "refit": refit_histories, "sweep": [],
    "seconds_per_epoch": float(np.mean([np.mean(h["seconds"]) for h in refit_histories.values()])),
    "device": str(device)}
with open("cVAE_Baseline_history.json", "w") as handle:
    json.dump(history_record, handle)
print(f"saved cVAE_Baseline.pth ({', '.join(checkpoint)}) and cVAE_Baseline_history.json")'''

CELL_LOOK_GRID = r'''# A first look at conditional sampling from the seed-0 model: rows are the digits 0 to 9 and
# each column is one latent code from N(0, I), decoded under every class.
_ = model.load_state_dict(checkpoint["seed_0"])
z_grid = torch.randn(12, Z_DIM, generator=torch.Generator().manual_seed(SEED)).to(device)
images, _ = sample_grid(model, z_grid)
fig, ax = plt.subplots(figsize=(4.4, 3.8))
ax.imshow(make_grid(images, nrow=12, pad_value=1)[0], cmap="gray", vmin=0, vmax=1)
ax.axis("off")
plt.show()'''

CELL_LATENT_NORMS = r'''# How far from the origin the encoder places real digits, against where sampling looks.
# Samples come from N(0, I); if the codes of real digits sit elsewhere, samples land in
# regions the decoder saw little of during training.
dev_cond = F.one_hot(train_labels[dev_index].to(device), NUM_CLASSES).float()
with torch.no_grad():
    dev_mu, _ = model.encode(train_images[dev_index].to(device), dev_cond)
norm_generator = torch.Generator().manual_seed(SEED)
prior_draws = torch.randn(len(dev_mu), Z_DIM, generator=norm_generator).to(device)
print(f"mean length of mu over development images : {dev_mu.norm(dim=1).mean():.3f}")
print(f"mean length of a draw from N(0, I)       : {prior_draws.norm(dim=1).mean():.3f}")'''

# =========================================================================== discriminator cells

CELL_ADV_LOSSES = r'''# Binary cross-entropy on logits, as lecture 9 p41 prescribes: real images carry the label 1
# and fakes the label 0. BCEWithLogitsLoss applies the sigmoid itself, which is more stable.
bce = nn.BCEWithLogitsLoss()


def discriminator_loss(discriminator, x_real, x_recon, x_prior, cond):
    """L_D. The two kinds of fake share the fake half of the loss equally."""
    real = discriminator(x_real, cond)
    fake_recon, fake_prior = discriminator(x_recon, cond), discriminator(x_prior, cond)
    loss_real = bce(real, torch.ones_like(real))
    loss_fake = (0.5 * bce(fake_recon, torch.zeros_like(fake_recon))
                 + 0.5 * bce(fake_prior, torch.zeros_like(fake_prior)))
    return loss_real + loss_fake


def generator_loss(discriminator, x_recon, x_prior, cond):
    """L_G, non-saturating: fakes are scored against the label 'real', so the loss is
    -log D(fake), whose gradient stays large while D is still rejecting fakes."""
    fake_recon, fake_prior = discriminator(x_recon, cond), discriminator(x_prior, cond)
    return (0.5 * bce(fake_recon, torch.ones_like(fake_recon))
            + 0.5 * bce(fake_prior, torch.ones_like(fake_prior)))'''

CELL_D_ACCURACY = r'''# How often the discriminator is right on development images: real ones should score above 0,
# their reconstructions and fresh samples below. The two halves are averaged, so 0.5 means it
# is fooled and 1.0 means it wins outright, when its gradient says little about how to improve.
def discriminator_accuracy(model, discriminator, dataloader, seed=SEED):
    model.eval()
    discriminator.eval()
    noise = torch.Generator().manual_seed(seed)
    real_right, fake_right, count = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            x, cond = get_batch(batch, device, f_train=False)
            mu, _ = model.encode(x, cond)
            x_prior = model.decode(torch.randn(len(x), Z_DIM, generator=noise).to(device), cond)
            real_right += (discriminator(x, cond) > 0).sum().item()
            fake_right += (discriminator(model.decode(mu, cond), cond) < 0).sum().item()
            fake_right += (discriminator(x_prior, cond) < 0).sum().item()
            count += len(x)
    return 0.5 * real_right / count + 0.5 * fake_right / (2 * count)'''

CELL_TRAIN_ADV = r'''# The tutorial's loop with a discriminator step placed before the cVAE step, the order of
# lecture 9 p41. Everything the baseline does is still here; the new lines are the D step,
# the prior samples, and lambda x L_G added to the cVAE's loss.
def training_loop_adversarial(model, discriminator, train_dataloader, dev_dataloader,
                              nb_epoch, lam):
    optimizer = vae_optimizer(model.parameters())
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    # Both players follow the same schedule, so the contest slows down together
    scheduler = cosine_schedule(optimizer, nb_epoch)
    d_scheduler = cosine_schedule(d_optimizer, nb_epoch)
    history = new_history(adversarial=True)
    for epoch in range(nb_epoch):
        started = time.perf_counter()
        model.train()
        discriminator.train()
        totals, count = np.zeros(5), 0
        for batch in train_dataloader:
            x, cond = get_batch(batch, device, f_train=True)
            # The cVAE's forward pass, written out so z is available: z = mu + sigma * eps
            mu, logvar = model.encode(x, cond)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            x_hat = model.decode(z, cond)
            x_prior = model.decode(torch.randn_like(mu), cond)
            # Step 1, the discriminator. Detached fakes: this step cannot change the cVAE
            d_optimizer.zero_grad()
            d_loss = discriminator_loss(discriminator, x, x_hat.detach(), x_prior.detach(), cond)
            d_loss.backward()
            d_optimizer.step()
            # Step 2, the cVAE. L_G sees decode(z.detach()): its gradient reaches the decoder only
            optimizer.zero_grad()
            total_loss, recon_loss, kl_loss = compute_loss(x, x_hat, mu, logvar)
            g_loss = generator_loss(discriminator, model.decode(z.detach(), cond), x_prior, cond)
            (total_loss + lam * g_loss).backward()
            optimizer.step()
            totals += len(x) * np.array([total_loss.item(), recon_loss.item(), kl_loss.item(),
                                         d_loss.item(), g_loss.item()])
            count += len(x)
        record(history, "train", totals[:3] / count)
        history["train_d_loss"].append(float(totals[3] / count))
        history["train_g_loss"].append(float(totals[4] / count))
        if dev_dataloader is not None:
            record(history, "dev", evaluate_loss(model, dev_dataloader))
            history["dev_d_accuracy"].append(
                discriminator_accuracy(model, discriminator, dev_dataloader))
        history["seconds"].append(time.perf_counter() - started)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        d_scheduler.step()
        report_epoch(epoch, nb_epoch, history)
    return history'''

CELL_D_CHECK = r'''# The discriminator's size, and one pass on a development batch: one logit per image.
discriminator = Discriminator(NUM_CLASSES).to(device)
x, cond = get_batch(next(iter(dev_dataloader)), device)
print(f"discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
print(f"D(x, c) for a batch of {len(x)}: {tuple(discriminator(x, cond).shape)}")'''

CELL_REAL_SHARPNESS = r'''# The development split's own sharpness is the reference a reconstruction is compared with.
dev_images, dev_labels = train_images[dev_index], train_labels[dev_index]
real_dev_sharpness = laplacian_variance(dev_images).mean()
real_dev_midgrey = midgrey_fraction(dev_images).mean()
print(f"development images: Laplacian variance {real_dev_sharpness:.4f}, "
      f"mid-grey fraction {real_dev_midgrey:.4f}")'''

CELL_SWEEP = r'''# One development run per lambda on the fit split, with everything else held fixed.
LAMBDAS = __LAMBDAS__
z_preview = torch.randn(4, Z_DIM, generator=torch.Generator().manual_seed(SEED)).to(device)
sweep, sweep_histories, previews = [], {}, {}
for lam in LAMBDAS:
    print(f"--- lambda = {lam}")
    _ = torch.manual_seed(SEED)
    model = cVAE(Z_DIM, num_classes=NUM_CLASSES).to(device)
    discriminator = Discriminator(NUM_CLASSES).to(device)
    history = training_loop_adversarial(model, discriminator, fit_dataloader, dev_dataloader,
                                        EPOCHS, lam)
    rebuilt = reconstruct(model, dev_images, dev_labels)
    sweep.append({"lambda": lam, "dev_recon_loss": float(np.mean(history["dev_recon"][-WINDOW:])),
                  "dev_sharpness": float(laplacian_variance(rebuilt).mean()),
                  "real_dev_sharpness": float(real_dev_sharpness),
                  "d_accuracy": float(np.mean(history["dev_d_accuracy"][-WINDOW:]))})
    sweep_histories[str(lam)] = history
    previews[lam] = sample_grid(model, z_preview)[0]'''

CELL_CHOOSE = r'''# The selection rule, fixed before the sweep ran. Runs whose discriminator won outright are
# set aside; of the rest, the reconstructions closest to real sharpness win, and a tie goes
# to the smaller lambda.
print("lambda   dev recon loss   dev sharpness   real dev sharpness   D accuracy")
for row in sweep:
    print(f"{row['lambda']:<8} {row['dev_recon_loss']:>14.4f} {row['dev_sharpness']:>15.4f} "
          f"{row['real_dev_sharpness']:>20.4f} {row['d_accuracy']:>12.4f}")
eligible = [row for row in sweep if row["d_accuracy"] <= 0.98]
if not eligible:
    eligible = [min(sweep, key=lambda row: row["d_accuracy"])]
chosen = min(eligible, key=lambda row: (abs(row["dev_sharpness"] - row["real_dev_sharpness"]),
                                        row["lambda"]))
LAMBDA = chosen["lambda"]
print(f"\nchosen lambda: {LAMBDA}")'''

CELL_PREVIEWS = r'''# Four latent codes decoded under every class, for each lambda: what the choice looks like.
fig, axes = plt.subplots(1, len(LAMBDAS), figsize=(2.0 * len(LAMBDAS), 4.2))
for ax, lam in zip(np.atleast_1d(axes), LAMBDAS):
    ax.imshow(make_grid(previews[lam], nrow=4, pad_value=1)[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"lambda = {lam}", color=WARM if lam == LAMBDA else INK)
    ax.axis("off")
plt.show()'''

CELL_ADV_CURVES = r'''# The chosen run's development curves, and its convergence statistic. The reconstruction
# loss is what has to settle; the discriminator's accuracy shows whether the contest stayed
# open, away from both 0.5 and 1.
dev_history = sweep_histories[str(LAMBDA)]
dev_convergence = convergence(dev_history["dev_recon"], WINDOW)
print(f"dev reconstruction loss, mean of the last {WINDOW} epochs : "
      f"{dev_convergence['last_mean']:.4f}")
print(f"mean of the {WINDOW} epochs before              : {dev_convergence['previous_mean']:.4f}")
print(f"relative change {dev_convergence['relative_change']:+.4f}, "
      f"within 1%: {dev_convergence['within_1pct']}")
epochs = np.arange(1, EPOCHS + 1)
fig, ax = plt.subplots(figsize=(5.5, 2.6))
ax.plot(epochs, dev_history["dev_recon"], color=WARM, label="dev reconstruction loss")
ax.set_xlabel("epoch")
ax.set_ylabel("100 x MSE")
twin = ax.twinx()
twin.plot(epochs, dev_history["dev_d_accuracy"], color=MUTED, linestyle="--")
twin.set_ylabel("D accuracy (dashed)")
twin.set_ylim(0.4, 1.0)
plt.show()'''

CELL_REFIT_ADV = r'''# The final models: lambda fixed, retrained on all N_TOTAL images once per seed, with the
# same seeds and batch orders as the baseline's final models.
refit_loaders = {seed: make_loader(order, shuffle=True, seed=seed) for seed in SEEDS}
checkpoint, refit_histories = {}, {}
for seed in SEEDS:
    print(f"--- seed {seed}")
    _ = torch.manual_seed(seed)
    model = cVAE(Z_DIM, num_classes=NUM_CLASSES).to(device)
    discriminator = Discriminator(NUM_CLASSES).to(device)
    refit_histories[str(seed)] = training_loop_adversarial(model, discriminator,
                                                           refit_loaders[seed], None, EPOCHS,
                                                           LAMBDA)
    checkpoint[f"seed_{seed}"] = cpu_state(model)
    checkpoint[f"discriminator_seed_{seed}"] = cpu_state(discriminator)'''

CELL_SAVE_ADV = r'''# The weights (the cVAE and its discriminator, one pair per seed) and every curve, including
# the whole lambda sweep. main_report.ipynb reads both files.
torch.save(checkpoint, "cVAE_DiscriminatorLoss.pth")
history_record = {
    "model": "cVAE_DiscriminatorLoss", "lambda": LAMBDA, "epochs": EPOCHS, "seeds": list(SEEDS),
    "window": WINDOW, "n_fit": len(fit_index), "n_dev": len(dev_index), "n_refit": len(order),
    "dev": dev_history, "convergence": dev_convergence, "refit": refit_histories,
    "sweep": sweep, "sweep_histories": sweep_histories,
    "seconds_per_epoch": float(np.mean([np.mean(h["seconds"]) for h in refit_histories.values()])),
    "device": str(device)}
with open("cVAE_DiscriminatorLoss_history.json", "w") as handle:
    json.dump(history_record, handle)
print(f"saved cVAE_DiscriminatorLoss.pth ({', '.join(checkpoint)})")
print("saved cVAE_DiscriminatorLoss_history.json")'''

# =========================================================================== classifier cells

CELL_CLF_TRAIN = r'''# Five fixed epochs on every training image, with no model selection: the classifier is a
# measuring instrument, so nothing about it is tuned and no development split is needed.
NUM_CLASSES, CLASSIFIER_EPOCHS = 10, __CLASSIFIER_EPOCHS__
_ = torch.manual_seed(SEED)
classifier = DigitClassifier(NUM_CLASSES).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(train_images[order], train_labels[order]), batch_size=256,
                    shuffle=True, generator=torch.Generator().manual_seed(SEED))
print(f"classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")'''

CELL_CLF_LOOP = r'''# A standard supervised loop: cross-entropy between the logits and the true class.
for epoch in range(CLASSIFIER_EPOCHS):
    classifier.train()
    loss_sum, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = classifier(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += len(images)
    print(f"epoch {epoch + 1}: training loss {loss_sum / count:.4f}, "
          f"training accuracy {correct / count:.4f}")
# Chance is a loss of ln 10 = 2.303; a working classifier ends far below it.
assert loss_sum / count < __CHANCE_BAR__, "the classifier did not learn"'''

CELL_CLF_SAVE = r'''# A plain state_dict saved from the CPU; main_report.ipynb rebuilds the class and loads it.
torch.save(cpu_state(classifier), "DigitClassifier.pth")
print("saved DigitClassifier.pth")'''

# =========================================================================== main_report cells

CELL_MR_DATA = r'''# The test split, written by the training notebooks. This notebook is the only place it is
# scored. Z_DIM and NUM_CLASSES are the values the checkpoints were trained with.
Z_DIM, NUM_CLASSES = 7, 10
test_data = torch.load("mnist_custom_test.pt", weights_only=True)
test_images, test_labels = test_data["test_images"], test_data["test_labels"]
assert tuple(test_images.shape[1:]) == (1, 20, 20)
print(f"test_images : {tuple(test_images.shape)}, {test_images.dtype}")
print(f"test class counts : {torch.bincount(test_labels, minlength=10).tolist()}")
print(f"pixels exactly 1.0 : {(test_images == 1).float().mean():.3f}")'''

CELL_MR_LOAD = r'''# The three checkpoints. Each holds plain state_dicts saved from the CPU; map_location puts
# them on this notebook's device. The cVAE files hold one trained model per seed.
def load_cvaes(path):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    models = {}
    for key in sorted(k for k in checkpoint if k.startswith("seed_")):
        network = cVAE(Z_DIM, num_classes=NUM_CLASSES).to(device)
        network.load_state_dict(checkpoint[key])
        models[int(key.split("_")[1])] = network.eval()
    return models, checkpoint


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


baseline, _ = load_cvaes("cVAE_Baseline.pth")
adversarial, adversarial_checkpoint = load_cvaes("cVAE_DiscriminatorLoss.pth")
assert sorted(baseline) == sorted(adversarial)
SEEDS = sorted(baseline)'''

CELL_MR_LOAD2 = r'''# The discriminator is not needed to evaluate the cVAE; it is loaded to show the file is
# complete and to count its parameters. The classifier is the measuring instrument.
discriminator = Discriminator(NUM_CLASSES).to(device)
_ = discriminator.load_state_dict(adversarial_checkpoint["discriminator_seed_0"])
classifier = DigitClassifier(NUM_CLASSES).to(device)
_ = classifier.load_state_dict(torch.load("DigitClassifier.pth", map_location=device,
                                          weights_only=True))
_ = classifier.eval()
print(f"seeds loaded : {SEEDS} for both cVAEs")
print(f"parameters   : cVAE {parameter_count(baseline[0]):,} | "
      f"discriminator {parameter_count(discriminator):,} | "
      f"classifier {parameter_count(classifier):,}")'''

CELL_MR_RECORDS = r'''# The training records: the chosen lambda, the epochs, where the models were trained, and
# how long an epoch took. Every value comes from the files, never retyped.
with open("cVAE_Baseline_history.json") as handle:
    baseline_record = json.load(handle)
with open("cVAE_DiscriminatorLoss_history.json") as handle:
    adversarial_record = json.load(handle)
LAMBDA = adversarial_record["lambda"]
records = {"baseline": baseline_record, "discriminator loss": adversarial_record}
print(f"lambda (chosen on the development split) : {LAMBDA}")
for name, record in records.items():
    print(f"{name:<18} | epochs {record['epochs']} | seeds {record['seeds']} | "
          f"refit on {record['n_refit']:,} images | trained on {record['device']} | "
          f"seconds per epoch {record['seconds_per_epoch']:.2f}")'''

CELL_MR_SWEEP = r'''# The lambda sweep, reprinted from the file, and the convergence statistic of each
# development run, re-derived here from its saved curve.
print("sweep | lambda | dev recon loss | dev sharpness | real dev sharpness | D accuracy")
for row in adversarial_record["sweep"]:
    print(f"sweep | {row['lambda']} | {row['dev_recon_loss']:.4f} | {row['dev_sharpness']:.4f} "
          f"| {row['real_dev_sharpness']:.4f} | {row['d_accuracy']:.4f}")
for name, record in records.items():
    check = convergence(record["dev"]["dev_recon"], record["window"])
    print(f"convergence | {name} | last {check['last_mean']:.4f} | "
          f"previous {check['previous_mean']:.4f} | change {check['relative_change']:+.4f} | "
          f"within 1% {check['within_1pct']}")
window = adversarial_record["window"]
final_accuracy = np.mean(adversarial_record["dev"]["dev_d_accuracy"][-window:])
print(f"discriminator accuracy, mean of the last {window} dev epochs : {final_accuracy:.4f}")'''

CELL_MR_LATENTS = r'''# Every latent code comes from a CPU generator seeded with SEED and is only then moved to the
# device, so the codes, and everything decoded from them, match on any machine.
latent_generator = torch.Generator().manual_seed(SEED)
z_grid = torch.randn(12, Z_DIM, generator=latent_generator).to(device)
z_eval = torch.randn(len(test_labels), Z_DIM, generator=latent_generator).to(device)
print(f"z_grid {tuple(z_grid.shape)} and z_eval {tuple(z_eval.shape)}, drawn with seed {SEED}")'''

CELL_MR_FIG1 = r'''# Figure 1: both 12 x 10 grids from the same twelve latent codes, so each column is one code
# decoded by the two models. Rows are the digits 0 to 9.
titles = {"baseline": "Baseline cVAE",
          "adversarial": f"cVAE with discriminator loss (\u03bb = {LAMBDA})"}
grid_images = {"baseline": sample_grid(baseline[0], z_grid),
               "adversarial": sample_grid(adversarial[0], z_grid)}
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), gridspec_kw={"wspace": 0.03})
for ax, (key, (images, labels)) in zip(axes, grid_images.items()):
    ax.imshow(make_grid(images, nrow=12, pad_value=1)[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(titles[key], fontsize=9)
    # The rows are the same digits in both grids, so only the left one is labelled.
    if ax is axes[0]:
        ax.set_yticks(12 + 22 * np.arange(10), [str(c) for c in range(10)], fontsize=8)
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(length=0)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
fig.savefig("figure_1_grids.pdf", bbox_inches="tight")
plt.show()'''

CELL_MR_CLASSIFY = r'''# The classifier's verdict on images that are meant to show given classes: whether its top
# choice is that class, and the probability it gives that class (its confidence).
def classify(images, labels, batch_size=2000):
    correct, confidence = [], []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            logits = classifier(images[start:start + batch_size].to(device))
            target = labels[start:start + batch_size].to(device)
            correct.append((logits.argmax(dim=1) == target).cpu())
            confidence.append(logits.softmax(dim=1).gather(1, target[:, None])[:, 0].cpu())
    return torch.cat(correct).double().numpy(), torch.cat(confidence).double().numpy()


for key, (images, labels) in grid_images.items():
    correct, _ = classify(images, labels)
    print(f"grid accuracy | {titles[key]} | {correct.mean():.4f} of {len(correct)} images")'''

CELL_MR_RECON = r'''# Every test image rebuilt from mu by every model (the objects the hypothesis is about), and
# one conditional sample per test image, under that image's own label, from the same code
# z_eval for both models. Every dictionary below is keyed by (model name, seed).
recon, samples = {}, {}
for name, models in (("baseline", baseline), ("adversarial", adversarial)):
    for seed in SEEDS:
        recon[name, seed] = reconstruct(models[seed], test_images, test_labels)
        samples[name, seed] = decode_latents(models[seed], z_eval, test_labels)
mse = {key: ((images - test_images) ** 2).flatten(1).mean(dim=1).double().numpy()
       for key, images in recon.items()}
print(f"{len(recon)} reconstruction sets and {len(samples)} sample sets of "
      f"{len(test_labels):,} images each")'''

CELL_MR_BOOTSTRAP = r'''# A paired bootstrap: both models are scored on the same images (or the same latent codes),
# so the per-pair differences are resampled, 2,000 times, and the middle 95% of their means
# is the interval. A fresh generator per call keeps every interval independent of cell order.
def paired_bootstrap(before, after, resamples=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    difference = after - before
    means = np.array([difference[rng.integers(0, len(difference), len(difference))].mean()
                      for _ in range(resamples)])
    low, high = np.percentile(means, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)'''

CELL_MR_METRIC_ROW = r'''# One printed line per metric: the real images' value where one exists, both models at seed 0,
# the paired difference (discriminator loss minus baseline) with its interval, and the same
# difference at every seed.
def metric_row(label, values, real=None, digits=4):
    before, after = values["baseline", 0], values["adversarial", 0]
    delta, low, high = paired_bootstrap(before, after)
    per_seed = [values["adversarial", s].mean() - values["baseline", s].mean() for s in SEEDS]
    real_text = f"{real.mean():.{digits}f}" if real is not None else "n/a"
    print(f"metric | {label} | real {real_text} | baseline {before.mean():.{digits}f} | "
          f"discriminator {after.mean():.{digits}f} | delta {delta:+.{digits}f} "
          f"[{low:+.{digits}f}, {high:+.{digits}f}] | seeds "
          + " ".join(f"{value:+.{digits}f}" for value in per_seed))
    return {"delta": delta, "low": low, "high": high, "seeds": per_seed}'''

CELL_MR_SHARPNESS = r'''# Pixel error, and the two views of sharpness, for reconstructions and for samples.
image_sets = {("recon",) + key: images for key, images in recon.items()}
image_sets.update({("sample",) + key: images for key, images in samples.items()})
sharpness = {key: laplacian_variance(images) for key, images in image_sets.items()}
grey = {key: midgrey_fraction(images) for key, images in image_sets.items()}
real_sharpness, real_grey = laplacian_variance(test_images), midgrey_fraction(test_images)


def by_kind(table, kind):
    return {key[1:]: value for key, value in table.items() if key[0] == kind}


rows = {"mse_recon": metric_row("mse_recon", mse, digits=5)}
for kind, label in (("recon", "reconstructions"), ("sample", "samples")):
    rows[f"sharpness_{kind}"] = metric_row(f"sharpness_{label}", by_kind(sharpness, kind),
                                           real=real_sharpness)
    rows[f"midgrey_{kind}"] = metric_row(f"midgrey_{label}", by_kind(grey, kind), real=real_grey)'''

CELL_MR_DECISION = r'''# The decision rule, fixed before any of these numbers existed. "Sharper" needs a higher
# Laplacian variance whose interval lies above 0, the same sign at every seed, and a lower
# mid-grey fraction agreeing with it.
def sharper(sharp_row, grey_row):
    return bool(sharp_row["low"] > 0 and all(value > 0 for value in sharp_row["seeds"])
                and grey_row["delta"] < 0)


decisions = {kind: sharper(rows[f"sharpness_{kind}"], rows[f"midgrey_{kind}"])
             for kind in ("recon", "sample")}
print(f"decision | reconstructions sharper with the discriminator loss: {decisions['recon']}")
print(f"decision | samples sharper with the discriminator loss: {decisions['sample']}")'''

CELL_MR_CLASSES = r'''# Class consistency (lecture 9 p42 and p87): the classifier on the real test images once,
# then on each model's samples, which were drawn under the test images' own labels.
real_correct, real_confidence = classify(test_images, test_labels)
print(f"classifier | real test images | accuracy {real_correct.mean():.4f} | "
      f"confidence {real_confidence.mean():.4f}")
verdicts = {key: classify(images, test_labels) for key, images in samples.items()}
for name, label in (("baseline", "baseline"), ("adversarial", "discriminator")):
    correct, confidence = verdicts[name, 0]
    seeds_text = " ".join(f"{verdicts[name, s][0].mean():.4f}" for s in SEEDS)
    print(f"classifier | samples {label} | accuracy {correct.mean():.4f} | "
          f"confidence {confidence.mean():.4f} | seeds {seeds_text}")'''

CELL_MR_PER_CLASS = r'''# The same accuracies per digit: a model can be strong overall and still fail one class.
labels_np = test_labels.numpy()
print("class | real   | baseline samples | discriminator samples")
for c in range(NUM_CLASSES):
    members = labels_np == c
    print(f"class {c} | {real_correct[members].mean():.4f} | "
          f"{verdicts['baseline', 0][0][members].mean():.4f} | "
          f"{verdicts['adversarial', 0][0][members].mean():.4f}")'''

CELL_MR_FEATURES = r'''# The classifier's 128-dimensional feature layer, for every image set, in float64.
def features(images, batch_size=2000):
    with torch.no_grad():
        parts = [classifier.features(images[start:start + batch_size].to(device)).cpu()
                 for start in range(0, len(images), batch_size)]
    return torch.cat(parts).double().numpy()


real_features = features(test_images)
sample_features = {key: features(images) for key, images in samples.items()}
recon_features = {key: features(images) for key, images in recon.items()}
print(f"feature vectors: {real_features.shape[1]} dimensions, "
      f"{1 + len(sample_features) + len(recon_features)} sets")'''

CELL_MR_FRECHET = r'''# The Frechet distance between two clouds of feature vectors, each summarised by its mean and
# covariance. tr sqrt(C_a C_b) comes from symmetric eigendecompositions, which stay real and
# exact where a general matrix square root can turn complex or print a warning.
def frechet_distance(a, b):
    mean_gap = ((a.mean(axis=0) - b.mean(axis=0)) ** 2).sum()
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    values, vectors = np.linalg.eigh(cov_a)
    root_a = (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T
    middle = np.linalg.eigvalsh(root_a @ cov_b @ root_a)
    cross = np.sqrt(np.clip(middle, 0, None)).sum()
    return float(mean_gap + np.trace(cov_a) + np.trace(cov_b) - 2 * cross)'''

CELL_MR_HALVES = r'''# Two class-stratified random halves of the test set. Comparing a generated set with half B,
# and half A with half B, keeps the size and the class mix of every comparison equal, so the
# real-against-real value is a fair floor.
rng = np.random.default_rng(SEED)
half_a, half_b = [], []
for c in range(NUM_CLASSES):
    members = rng.permutation(np.flatnonzero(labels_np == c))
    half_a.extend(members[:len(members) // 2])
    half_b.extend(members[len(members) // 2:])
half_a, half_b = np.sort(half_a), np.sort(half_b)
print(f"halves | A {len(half_a):,} | B {len(half_b):,}")'''

CELL_MR_FD_ROWS = r'''# Frechet distances: the floor, then each model's samples and reconstructions of half A
# (conditioned on half A's labels) against the real images of half B.
floor = frechet_distance(real_features[half_a], real_features[half_b])
print(f"frechet | floor, real A against real B | {floor:.2f}")
distances = {}
for kind, table in (("samples", sample_features), ("reconstructions", recon_features)):
    for key, values in table.items():
        distances[(kind,) + key] = frechet_distance(values[half_a], real_features[half_b])
    print(f"frechet | {kind} | baseline {distances[kind, 'baseline', 0]:.2f} | "
          f"discriminator {distances[kind, 'adversarial', 0]:.2f} | seeds baseline "
          + " ".join(f"{distances[kind, 'baseline', s]:.2f}" for s in SEEDS)
          + " | seeds discriminator "
          + " ".join(f"{distances[kind, 'adversarial', s]:.2f}" for s in SEEDS))'''

CELL_MR_SPREAD = r'''# Diversity: within each digit class, the total variance of the samples' features divided by
# that of the real images', averaged over classes. 1 matches real variety; well below 1 means
# the samples of a class look alike, the signature of mode collapse.
def spread_ratio(generated, real):
    ratios = [np.trace(np.cov(generated[labels_np == c], rowvar=False))
              / np.trace(np.cov(real[labels_np == c], rowvar=False)) for c in range(NUM_CLASSES)]
    return float(np.mean(ratios))


spread = {key: spread_ratio(values, real_features) for key, values in sample_features.items()}
print(f"spread | baseline {spread['baseline', 0]:.3f} | discriminator "
      f"{spread['adversarial', 0]:.3f} | seeds baseline "
      + " ".join(f"{spread['baseline', s]:.3f}" for s in SEEDS) + " | seeds discriminator "
      + " ".join(f"{spread['adversarial', s]:.3f}" for s in SEEDS))'''

CELL_MR_FIG2 = r'''# Figure 2: (a) one test digit per class, real and rebuilt by each seed-0 model; (b) the two
# development runs' reconstruction loss, with the discriminator's accuracy on a second axis.
# The figure is built over two cells, so it is a bare Figure, which Jupyter does not display
# until display() is called; a pyplot figure would appear half-drawn after this cell.
from matplotlib.figure import Figure
from IPython.display import display

first = [int(np.flatnonzero(labels_np == c)[0]) for c in range(NUM_CLASSES)]
strip = torch.cat([test_images[first], recon["baseline", 0][first],
                   recon["adversarial", 0][first]])
fig = Figure(figsize=(3.4, 2.3), layout="constrained")
ax_strip, ax_curve = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 1.4]})
ax_strip.imshow(make_grid(strip, nrow=10, pad_value=1)[0], cmap="gray", vmin=0, vmax=1)
ax_strip.set_yticks(12 + 22 * np.arange(3), ["real", "baseline", "disc. loss"], fontsize=7)
ax_strip.set_xticks([])
ax_strip.tick_params(length=0)
ax_strip.set_title("(a) test digits and their reconstructions", fontsize=7)
for side in ("left", "bottom"):
    ax_strip.spines[side].set_visible(False)'''

CELL_MR_FIG2B = r'''# Panel (b), then the figure is saved at the width of one report column.
epochs = np.arange(1, len(baseline_record["dev"]["dev_recon"]) + 1)
ax_curve.plot(epochs, baseline_record["dev"]["dev_recon"], color=ACCENT, label="baseline")
ax_curve.plot(epochs, adversarial_record["dev"]["dev_recon"], color=WARM,
              label=f"disc. loss, \u03bb = {LAMBDA}")
ax_curve.set_xlabel("epoch", fontsize=7)
ax_curve.set_ylabel("dev reconstruction loss", fontsize=7)
ax_curve.set_title("(b) development runs", fontsize=7)
twin = ax_curve.twinx()
twin.plot(epochs, adversarial_record["dev"]["dev_d_accuracy"], color=MUTED, linestyle="--",
          label="D accuracy")
twin.set_ylim(0.4, 1.0)
twin.set_ylabel("D accuracy (dashed)", fontsize=7)
twin.spines["right"].set_visible(True)
ax_curve.legend(frameon=False, fontsize=6.5, loc="upper right")
fig.savefig("figure_2_reconstructions.pdf", bbox_inches="tight")
display(fig)'''

CELL_MR_SUMMARY = r'''# Every headline number in one machine-readable block, rounded as printed above.
summary = {
    "lambda": LAMBDA, "seeds": SEEDS, "decisions": decisions,
    "metrics": {name: {k: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v])
                       for k, v in row.items()} for name, row in rows.items()},
    "classifier_real_accuracy": round(float(real_correct.mean()), 4),
    "classifier_sample_accuracy": {name: round(float(verdicts[name, 0][0].mean()), 4)
                                   for name in ("baseline", "adversarial")},
    "frechet_floor": round(floor, 2),
    "frechet_samples": {name: round(distances["samples", name, 0], 2)
                        for name in ("baseline", "adversarial")},
    "spread": {name: round(spread[name, 0], 3) for name in ("baseline", "adversarial")}}
print(json.dumps(summary, indent=1))'''


# =========================================================================== assembly

def as_source(text):
    """nbformat stores source as a list of lines that KEEP their newline characters."""
    lines = text.rstrip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def markdown(cell_id, source):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": as_source(source)}


def code(cell_id, source):
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {},
            "outputs": [], "source": as_source(source)}


IMPORT_SETS = {
    "training": {"__IMPORT_OS__": "import os\n", "__IMPORT_TIME__": "import time\n",
                 "__IMPORT_OPTIM__": "import torch.optim as optim\n",
                 "__IMPORT_DATA__": "from torch.utils.data import DataLoader, TensorDataset\n"},
    "report": {"__IMPORT_OS__": "", "__IMPORT_TIME__": "", "__IMPORT_OPTIM__": "",
               "__IMPORT_DATA__": ""},
}


def fill(text, **extra):
    values = {f"__{key}__": str(value) for key, value in SIZES.items()}
    values.update(extra)
    for key, value in values.items():
        text = text.replace(key, value)
    return text


def imports(kind):
    text = CELL_IMPORTS
    for key, value in IMPORT_SETS[kind].items():
        text = text.replace(key, value)
    return text


HEADER = r"""# IFN680 Project 6: The Sharpness Quest - __TITLE__

**Group 4** - Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

__INTRO__"""

HYPOTHESIS = ("> *Adding a discriminator term to the loss function of a cVAE during training increases "
              "the sharpness of the reconstructed images.*")

# --------------------------------------------------------------------------- markdown blocks

MD_SETUP = r"""## 1 Setup

The libraries, the device and the seed. The printed versions make the executed notebook its own
record of where it ran. The palette cell below it keeps every figure in the same colours."""

MD_DATA = r"""## 2 The data

`mnist_custom.pt` holds four tensors: 60,000 training images with their labels, and 10,000 test
images with theirs. The cell below loads it. Because `main_report.ipynb` needs the test split and
nothing else, the cell also writes that split to its own file, `mnist_custom_test.pt`, so the
archive does not have to carry the whole 112 MB dataset."""

MD_AUDIT = r"""### 2.1 What the file holds

The brief calls the data only "the modified MNIST dataset", so its properties are measured before
anything is trained on it. Two of them matter later. The images are **20 x 20**, not the
tutorial's 28 x 28, which is why the tutorial's model needs a change (section 3). And the
background is **white** (pixel value 1.0) with dark strokes, the reverse of the tutorial's MNIST."""

MD_SPLIT_TEMPLATE = r"""### __NUM__ A development split

Choosing settings and deciding when training has finished both need images the model is not
trained on. Using the test images for that would make them part of the training process, and
their score would stop being an honest measure of new data. So a **development split** is held
out from the training images, drawn at random with the seed:

| Split | Images | Used for |
|---|---|---|
| fit | 50,000 | training during development runs |
| development | 10,000 | curves, the convergence check, and choosing settings |
| all training images | 60,000 | the final models, retrained once the settings are fixed |
| test | 10,000 | `main_report.ipynb` only |

The draw is random because the MNIST files keep the order their images were assembled in. Its
documentation (LeCun, Cortes and Burges) says the test file's first 5,000 images are cleaner and
easier than its last 5,000, and that in one of its two source collections each writer's digits
appear as one consecutive block. A slice of consecutive images is therefore not guaranteed to
look like the rest.

| Call | What you give it | What you get back |
|---|---|---|
| `torch.Generator().manual_seed(s)` | a seed | a private random-number source, independent of every other draw |
| `torch.randperm(n, generator=g)` | a count and a generator | the numbers 0 to n-1 in a random order |
| `torch.bincount(labels, minlength=10)` | integer labels | how many times each of 0 to 9 occurs |"""

MD_MODEL_BASELINE = r"""## 3 The model

The tutorial's cVAE. An **encoder** turns an image and its class into a small cloud of possible
codes, described by a centre `mu` and a spread (`logvar`, the log of the variance), in a
7-dimensional **latent space**. A **decoder** turns a code and a class back into an image.
Because the class is given to both, the code only has to carry *how* the digit is written
(slant, thickness, size), and at generation time any class can be asked for.

__FIG_ARCHITECTURE__

**Why one kernel had to change.** A convolution with kernel size k, stride s and padding p turns
an n x n map into a map of side `floor((n + 2p - k) / s) + 1`. The tutorial's first three layers
(k = 3, s = 2, p = 1) take 20 to 10, 10 to 5, and 5 to `floor((5 + 2 - 3) / 2) + 1 = 3`. Its fourth
layer then has a 4 x 4 kernel and no padding, which is larger than the 3 x 3 map it is given, so
PyTorch stops with an error on the first batch. A 3 x 3 kernel covers the map exactly and gives
the 1 x 1 output the rest of the model expects. The decoder mirrors it.

Padding the images back to 28 x 28 would also run, but it changes the task. The border would add
384 constant white pixels to every image, 49% of the 784, and the decoder reproduces a constant
perfectly. The average pixel error would then roughly halve, and so would its weight against the
KL term, silently changing the balance the tutorial's loss was set up with.

| Call | What you give it | What you get back |
|---|---|---|
| `nn.Conv2d(c_in, c_out, kernel_size, stride, padding)` | maps of shape (batch, c_in, n, n) | (batch, c_out, n', n'): each output value is a weighted sum over a kernel-sized patch of every input map |
| `nn.ReLU()` | any tensor | the same tensor with negative values set to 0 |
| `nn.Flatten()` | (batch, 256, 1, 1) | (batch, 256) |
| `torch.cat([a, b], dim=1)` | two tensors with the same batch size | one tensor with their columns side by side |
| `torch.chunk(x, 2, dim=1)` | (batch, 14) | two tensors of (batch, 7): here `mu` and `logvar` |"""

MD_DECODER = r"""### 3.1 The decoder

A **transposed convolution** runs the arithmetic backwards: it spreads each input value over a
kernel-sized patch of a larger output. Its output side is
`(n - 1) s - 2p + k + output_padding`, which gives 1 to 3 (k = 3, s = 1, p = 0), then 3 to 5, 5 to
10 and 10 to 20 (k = 3, s = 2, p = 1, with output_padding 0, 1 and 1). The final sigmoid keeps every
pixel between 0 and 1, the range of the data.

| Call | What you give it | What you get back |
|---|---|---|
| `nn.Unflatten(1, (256, 1, 1))` | (batch, 256) | (batch, 256, 1, 1), a 1 x 1 map with 256 channels |
| `nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding, output_padding)` | (batch, c_in, n, n) | (batch, c_out, n', n') with n' as above |
| `nn.Sigmoid()` | any tensor | each value squashed into (0, 1) |"""

MD_LOSS = r"""### 3.2 The loss

`compute_loss` is the tutorial's, unchanged, and it is the loss in the brief's figure:

$$\mathcal{L}_{VAE} = 100 \cdot \mathrm{MSE}(x, \hat{x}) + 0.1 \cdot KL\big(q(z \mid x, c) \,\|\, \mathcal{N}(0, I)\big)$$

The first term, the **mean squared error** over all pixels, rewards a reconstruction that matches
the input pixel by pixel. The second, the **Kullback-Leibler divergence**, pulls each image's cloud
of codes toward a standard normal distribution. That is what makes generation possible later: codes
drawn from N(0, I) then land where the decoder has seen real codes during training. The weights 100
and 0.1 are the tutorial's."""

MD_CVAE = r"""### 3.3 The cVAE and the reparameterisation trick

`forward()` needs a random code `z` from the encoder's cloud, but a random draw has no gradient.
The trick is to draw the randomness separately, `eps ~ N(0, I)`, and compute
`z = mu + sigma * eps`. The randomness is then an input, and the gradient flows through `mu` and
`sigma` as through any other arithmetic. `encode()` and `decode()` expose the two halves, which the
evaluation code uses to rebuild an image from `mu` alone."""

MD_SETTINGS = r"""### 3.4 Settings, and a shape check

The settings every cVAE run in this project shares, then one forward pass. The printed shapes
confirm the adapted model takes a 1 x 20 x 20 image to a 7-number code and back to 1 x 20 x 20."""

MD_TRAINING = r"""## 4 Training

The tutorial trains in batches of 1,000 images drawn in a shuffled order. `get_batch` is the
tutorial's own helper: it moves a batch to the device and turns each label into a **one-hot**
vector, ten numbers that are all 0 except a 1 at the class.

| Call | What you give it | What you get back |
|---|---|---|
| `TensorDataset(images, labels)` | tensors with the same first dimension | a dataset whose item i is (images[i], labels[i]) |
| `DataLoader(dataset, batch_size, shuffle, generator)` | a dataset | an iterable of batches; with a seeded generator the shuffled order is the same every run |
| `F.one_hot(labels, num_classes=10)` | integer labels | (batch, 10) one-hot rows |"""

MD_EVALUATE = r"""### 4.1 Scoring the development split

After each epoch the model rebuilds the development images from `mu`, the centre of each image's
cloud of codes, and the loss is averaged over all of them. `model.eval()` and `torch.no_grad()` tell
PyTorch no training happens here, so it keeps no record for gradients and the pass is fast."""

MD_HISTORY = r"""### 4.2 Keeping the curves

Every epoch's losses go into a dictionary that is saved to disk at the end, so `main_report.ipynb`
can draw and check them without training anything. `vae_optimizer` holds the tutorial's AdamW
settings in one place, so both cVAE notebooks use literally the same ones, and
`cosine_schedule` holds the learning-rate schedule of section 4.4.

| Call | What you give it | What you get back |
|---|---|---|
| `optim.AdamW(parameters, lr, eps, betas)` | the model's weights and settings | an optimiser; `.step()` moves every weight a little against its gradient |
| `optimizer.zero_grad()` | nothing | clears the gradients left by the previous batch |
| `loss.backward()` | nothing | computes the gradient of `loss` for every weight |
| `CosineAnnealingLR(optimizer, T_max)` | an optimiser and a run length in epochs | a schedule; each `.step()` lowers the learning rate along half a cosine, reaching 0 after `T_max` steps |"""

MD_TRAIN_FN = r"""### 4.3 The training loop

For each batch: draw codes and rebuild the images (`model(x, cond)`), measure the loss, compute the
gradients, and take one optimiser step. After each epoch, score the development split and move
the learning rate one step down its schedule."""

MD_CONVERGENCE = r"""### 4.4 What "converged" means here

The brief asks for training "until convergence". Here that means the development reconstruction
loss has stopped improving: its mean over the last __WINDOW__ epochs is within 1% of its mean over
the __WINDOW__ epochs before. A mean over several epochs is used instead of a single epoch because
one epoch's value moves up and down a little by chance.

The tutorial keeps the learning rate constant for 50 epochs. On this data that is not enough:
at a constant rate the development loss was still falling by several percent every ten epochs
long after epoch 50, because every step has the same size and the weights keep moving instead of
settling. So both cVAEs use a **cosine schedule**: the learning rate starts at the tutorial's
0.001 and falls smoothly to 0 over the run, taking large steps early, when there is much to
learn, and ever smaller ones late, so the weights come to rest. Both models use the same
schedule and the same number of epochs, so it favours neither."""

MD_DEV_RUN = r"""## 5 Development run and convergence

The model trains on the fit split for __EPOCHS__ epochs under the cosine schedule, and the
development split is scored after every epoch."""

MD_DEV_CURVES = r"""### 5.1 Has it converged?

The statistic of section 4.4, then the two curves. The fit-split loss is measured on codes drawn at
random from each image's cloud, so it sits above the development loss, which rebuilds from `mu`.
The dashed line is the learning rate: the curves flatten as it approaches 0."""

MD_REFIT = r"""## 6 The final models

With the settings fixed, the development split has done its job, and the final models retrain on
all training images. Each is trained three times, with seeds 0, 1 and 2, because one run of a
neural network is one draw from many possible runs: a difference between two models is only
convincing if it survives a change of seed."""

MD_TRIVIAL = r"""### 6.1 A sanity check

A model that learned nothing useful could still return one average image for every input. Its
loss is the bar: each seed's final loss must be well below it."""

MD_SAVE = r"""### 6.2 Saving

The weights of all three seeds go into one file, and every curve into a JSON file. Both are read by
`main_report.ipynb`, which evaluates the models on the test split."""

MD_LOOK = r"""## 7 Samples

**Conditional sampling**: draw a code from N(0, I), choose a class, decode. The grid uses the brief's
layout, 12 samples per digit with the digits as rows, and the same 12 codes in every row, so a
column shows one handwriting style written as each of the ten digits. This is a first look. The
comparison with the other model is made in `main_report.ipynb`."""

MD_NORMS = r"""### 7.1 Where real digits sit in the latent space

Sampling assumes the codes of real digits look like draws from N(0, I). A draw from a
7-dimensional standard normal lies about 2.55 from the origin on average. If the encoder places real
digits much closer in or further out, samples come from regions the decoder rarely saw, which is one
reason samples can look worse than reconstructions."""


def baseline_notebook():
    intro = (
        "This notebook is **Task 1**. It trains the conditional variational autoencoder (cVAE) "
        "of Tutorial 8.3 on `mnist_custom.pt` with **7 latent dimensions**, and it is the "
        "baseline that `cVAE_DiscriminatorLoss.ipynb` is compared against.\n\n"
        "The model, the loss and the optimiser are the tutorial's. What differs:\n\n"
        "| | Tutorial 8.3 | This notebook | Why |\n|---|---|---|---|\n"
        "| Image size | 28 x 28 | 20 x 20 | `mnist_custom.pt` is MNIST resized to 20 x 20 and "
        "inverted |\n"
        "| Last encoder kernel, first decoder kernel | 4 x 4 | 3 x 3 | a 4 x 4 kernel does not "
        "fit the 3 x 3 map it meets at 20 x 20 (section 3) |\n"
        "| Latent size | 8 | 7 | the brief |\n"
        "| Learning rate | constant 0.001 for 50 epochs | from 0.001 down to 0 along a cosine, "
        "over __EPOCHS__ epochs | at a constant rate the loss was still falling well past 50 "
        "epochs, so training would not settle (section 4.4) |\n"
        "| Split scored every epoch | the test split | a development split of the training "
        "images | the test split is kept for `main_report.ipynb` (section 2.2) |\n"
        "| Final model | the model after one run | retrained on all 60,000 training images, "
        "once per seed 0, 1, 2 | section 6 |\n\n"
        "It writes `cVAE_Baseline.pth` (the weights of each seed), `cVAE_Baseline_history.json` "
        "(every training curve) and `mnist_custom_test.pt` (the test split).\n\n"
        "**Sections:** 1 Setup, 2 The data, 3 The model, 4 Training, 5 Development run and "
        "convergence, 6 The final models, 7 Samples.")
    return [
        markdown("md-title", fill(HEADER, __TITLE__="cVAE_Baseline", __INTRO__=fill(intro))),
        markdown("md-setup", MD_SETUP),
        code("code-imports", imports("training")),
        code("code-style", CELL_STYLE),
        markdown("md-data", MD_DATA),
        code("code-data", CELL_DATA),
        markdown("md-audit", MD_AUDIT),
        code("code-audit", CELL_AUDIT),
        markdown("md-look-data", "One training image per class, and a histogram of every pixel "
                                 "value in the training set. Almost every pixel is either pure "
                                 "white or near black. A blurred digit has many pixels in "
                                 "between, so blur is easy to see in this data, and easy to "
                                 "measure."),
        code("code-data-figure", CELL_DATA_FIGURE),
        markdown("md-split", MD_SPLIT_TEMPLATE.replace("__NUM__", "2.2")),
        code("code-split", fill(CELL_SPLIT)),
        markdown("md-model", MD_MODEL_BASELINE.replace(
            "__FIG_ARCHITECTURE__", figure("architecture.png", "The cVAE architecture"))),
        code("code-encoder", CELL_ENCODER),
        markdown("md-decoder", MD_DECODER),
        code("code-decoder", CELL_DECODER),
        markdown("md-loss", MD_LOSS),
        code("code-loss", CELL_LOSS),
        markdown("md-cvae", MD_CVAE),
        code("code-cvae", CELL_CVAE),
        markdown("md-settings", MD_SETTINGS),
        code("code-settings", fill(CELL_SETTINGS)),
        markdown("md-training", MD_TRAINING),
        code("code-batches", CELL_BATCHES),
        code("code-model-check", CELL_MODEL_CHECK),
        markdown("md-evaluate", MD_EVALUATE),
        code("code-evaluate", CELL_EVALUATE),
        markdown("md-history", MD_HISTORY),
        code("code-history", CELL_HISTORY),
        markdown("md-train-fn", MD_TRAIN_FN),
        code("code-train-fn", CELL_TRAIN_FN),
        markdown("md-convergence", fill(MD_CONVERGENCE)),
        code("code-convergence", CELL_CONVERGENCE),
        markdown("md-dev-run", fill(MD_DEV_RUN)),
        code("code-dev-run", CELL_DEV_RUN),
        markdown("md-dev-curves", MD_DEV_CURVES),
        code("code-dev-curves", CELL_DEV_CURVES),
        markdown("md-sample-helpers", "### 5.2 What the reconstructions look like\n\nThe "
                                      "helpers below decode images for this notebook and for "
                                      "`main_report.ipynb`: `reconstruct` rebuilds each image "
                                      "from `mu`, and `decode_latents` turns given codes into "
                                      "images of given classes."),
        code("code-sample", CELL_SAMPLE),
        markdown("md-dev-recon", "One development image per class (top row) and its "
                                 "reconstruction (bottom row). These are the images the "
                                 "hypothesis is about, and their blur is what it predicts a "
                                 "discriminator can reduce."),
        code("code-dev-recon", CELL_DEV_RECON),
        markdown("md-refit", MD_REFIT),
        code("code-cpu-state", CELL_CPU_STATE),
        code("code-refit-baseline", CELL_REFIT),
        markdown("md-trivial", MD_TRIVIAL),
        code("code-trivial", fill(CELL_TRIVIAL)),
        markdown("md-save", MD_SAVE),
        code("code-save-baseline", CELL_SAVE_BASELINE),
        markdown("md-look", MD_LOOK),
        code("code-look-grid", CELL_LOOK_GRID),
        markdown("md-norms", MD_NORMS),
        code("code-latent-norms", CELL_LATENT_NORMS),
    ]


# --------------------------------------------------------------------------- discriminator notebook

MD_WHY = r"""## 4 Why a discriminator could sharpen a cVAE

__FIG_BLUR__

**Why an MSE-trained decoder blurs.** Several different images are plausible for one code and
class: the same 7 can be written with the bar a little higher or lower. The decoder cannot know
which, and the single output that minimises the *average squared error* over all of them is their
pixel-wise **mean**. Averaging strokes in slightly different places gives a grey smear, as the
figure shows on this dataset's own 7s. Nothing in the MSE term objects to that smear, because it
is the best guess under that loss.

**What a discriminator adds.** A discriminator is a second network trained to tell real images
from the decoder's output. Its judgement becomes a loss for the decoder: output that the
discriminator can spot as fake is penalised. A grey smear does not look like any real digit, so a
decoder that also has to fool the discriminator is pushed toward outputs that look like one
plausible digit, with crisp strokes, instead of the average of many (lecture 9, pp. 45-46).

That is the hypothesis this notebook tests:

__HYPOTHESIS__

It is the idea behind the **VAE-GAN** of Larsen et al. (2016), who reported sharper samples than a
plain VAE on faces. The price is known too: an output that commits to one plausible digit can be
further, pixel by pixel, from the particular digit it was asked to reconstruct, so pixel error can
rise while sharpness does. `main_report.ipynb` measures both."""

MD_DISC = r"""## 5 The discriminator

__FIG_DISC__

The discriminator judges an image *together with its class* (lecture 9, p39: in a conditional GAN
"the class is given to both" networks). A well-drawn 3 labelled "8" should read as fake. The class
enters as ten extra image planes, each filled with that class's one-hot value, so every convolution
sees it. The network is deliberately small, two convolutions and one linear layer. A discriminator
much stronger than the decoder wins at once, and a discriminator that always wins gives the decoder
no usable signal about *how* to improve.

| Call | What you give it | What you get back |
|---|---|---|
| `nn.LeakyReLU(0.2)` | any tensor | negative values multiplied by 0.2 instead of set to 0, so small gradients survive |
| `cond[:, :, None, None].expand(-1, -1, 20, 20)` | one-hot rows (batch, 10) | (batch, 10, 20, 20): each class value copied across a whole plane |"""

MD_ADV_LOSSES = r"""## 6 The adversarial losses

Lecture 9 p41 fixes the recipe: the loss is binary cross-entropy, "real images are '1', fake
images are '0'", and there are two optimisers. The brief's figure writes the same two losses:

$$\mathcal{L}_D = \mathbb{E}_x[\log(1 - D(x))] + \mathbb{E}_z[\log D(G(z))] \qquad \mathcal{L}_G = \mathbb{E}_z[\log(1 - D(G(z)))]$$

where D(.) is the probability of "real". The discriminator minimises the first, pushing D(x) up to 1
on real images and D(G(z)) down to 0 on fakes. (Slides 37 and 45 label the classes the other way
round; the formula and p41 agree with each other, and this notebook follows them.)

**The generator's loss is used in its non-saturating form**, `-log D(G(z))`, instead of the
figure's `log(1 - D(G(z)))`. Both are smallest when the discriminator is fooled. They differ early
in training, when fakes are poor and D(G(z)) is near 0: there `log(1 - D)` is almost flat, so its
gradient nearly vanishes exactly when the decoder most needs a signal, while `-log D` is steep.
Goodfellow et al. (2014, section 3) recommend the same swap. In code it is simply the cross-entropy
of a fake against the label 1.

**Two kinds of fake.** The decoder produces two kinds of image, and the discriminator sees both, each
with half the weight:

- **reconstructions** `decode(z, c)` with z from an image's own cloud: the objects the hypothesis
  is about;
- **samples** `decode(z_p, c)` with `z_p ~ N(0, I)`: what conditional generation, Task 3, produces.

The cVAE's total loss becomes `100 MSE + 0.1 KL + lambda * L_G`. `lambda` is the only new setting,
and section 9 chooses it.

| Call | What you give it | What you get back |
|---|---|---|
| `nn.BCEWithLogitsLoss()(logits, targets)` | raw scores and 0/1 labels | the mean binary cross-entropy, with the sigmoid applied inside for numerical stability |
| `torch.ones_like(t)`, `torch.zeros_like(t)` | a tensor | a tensor of 1s or 0s with its shape and device |"""

MD_TRAIN_STEP = r"""## 7 One training step

__FIG_STEP__

Each batch takes two steps, in the order of lecture 9 p41, **discriminator first**:

1. **Discriminator step.** Real images are scored against 1, reconstructions and samples against 0,
   and only the discriminator's weights move. The fakes are passed through `.detach()`, which cuts
   them off from the cVAE, so this step cannot change the cVAE.
2. **cVAE step.** The baseline's loss plus `lambda * L_G`, scored by the *updated* discriminator,
   and only the cVAE's weights move.

**The adversarial gradient reaches the decoder only.** `L_G` is computed on `decode(z.detach(), c)`:
the same image, but with the path back into the encoder cut. The encoder therefore learns from MSE
and KL exactly as in the baseline, and only the decoder is asked to fool the discriminator. This
follows Larsen et al. (2016), who keep the GAN error out of the encoder: the encoder's job is to
describe the input, and making its codes fool a discriminator would work against that.

Everything else (seed, batch order, epochs, architecture, optimiser, learning-rate schedule) is
identical to `cVAE_Baseline.ipynb`, so the discriminator term is the only difference between the
two models. The discriminator's own optimiser follows the same cosine schedule, so both players
slow down together and the contest ends settled rather than with one side still learning.

| Call | What you give it | What you get back |
|---|---|---|
| `t.detach()` | a tensor | the same values with no link back to the computation that made them |
| `torch.randn_like(mu)` | a tensor | standard normal noise of the same shape, on the same device |"""

MD_D_CHECK = r"""### 7.1 The discriminator's size, and its accuracy

The discriminator's size, and one pass on a batch. `discriminator_accuracy` is the monitor used
during training: how often the discriminator calls development images, and the fakes made
from them, correctly, with real and fake weighted equally."""

MD_SHARPNESS = r"""## 8 Measuring sharpness

__FIG_LAPLACIAN__

The **Laplacian** of an image measures how sharply brightness bends at each pixel: the 3 x 3 kernel
`[[0, 1, 0], [1, -4, 1], [0, 1, 0]]` compares each pixel with its four neighbours. At a crisp stroke
edge the response swings strongly positive and negative, and in a smooth blur it stays near zero. The
**variance** of the response over the image is therefore one number that is high for a sharp image
and low for a blurred one (Pech-Pacheco et al., 2000), and the brief names it. Because the kernel's
weights sum to zero, flipping black and white only flips the response's sign, so the variance is the
same for this inverted data as for ordinary MNIST.

It has one known weakness: noise and ringing also raise it. So a reconstruction is compared with the
**real images' own value**, and above that level higher is not better. A second measure, the
**mid-grey fraction**, counts pixels that are neither background nor stroke (between 0.1 and 0.9).
Blur raises it, and so does noise, the opposite direction to the Laplacian's error, so the two
agreeing is stronger evidence than either alone.

| Call | What you give it | What you get back |
|---|---|---|
| `F.conv2d(images, kernel)` | images (n, 1, 20, 20) and a (1, 1, 3, 3) kernel | the 18 x 18 response of each image (no padding) |
| `t.flatten(1).var(dim=1)` | (n, 1, 18, 18) | the variance within each image |"""

MD_SWEEP = r"""## 9 Choosing lambda on the development split

`lambda` sets how much the decoder cares about fooling the discriminator against matching pixels.
Too small and nothing changes; too large and the decoder may chase the discriminator at the cost of
the digit. Four values are tried, each with a full development run on the fit split and the same
seed. The rule for choosing, fixed before the runs:

1. set aside any run whose discriminator won outright, accuracy above 0.98 on the development split
   over the last __WINDOW__ epochs, because its gradient no longer carries information (if every run
   is set aside, keep the one with the lowest accuracy);
2. of the rest, choose the lambda whose development reconstructions come **closest to the real
   development images' sharpness**, the hypothesis's own quantity;
3. break a tie toward the smaller lambda.

The test split plays no part: the choice uses the development split only."""

MD_CHOOSE = r"""### 9.1 The choice

The sweep's summary and the chosen value. It is written to the history file, and
`main_report.ipynb` reads it from there."""

MD_PREVIEWS = r"""### 9.2 What each lambda produces

Four latent codes decoded under every class by each development model (digits as rows). The chosen
lambda's title is in colour."""

MD_ADV_CURVES = r"""## 10 Convergence of the chosen run

The chosen development run's reconstruction loss, which has to settle, and the discriminator's
accuracy, which should stay away from 0.5 (no better than guessing) and from 1 (winning
outright)."""

MD_REFIT_ADV = r"""## 11 The final models

As for the baseline: lambda fixed, retrained on all training images, once per seed 0, 1 and 2, with
the same batch orders the baseline's final models used."""

MD_SAVE_ADV = r"""### 11.1 Sanity check and saving

The same bar as the baseline's: each seed's final reconstruction loss must be well below that of a
model that answers every input with the mean image. Then the weights and the curves are saved."""

MD_LOOK_ADV = r"""## 12 Samples

The same first look as in the baseline notebook, from the same 12 codes: rows are digits, and each
column is one code decoded under all ten classes. The side-by-side comparison, and every
measurement, is in `main_report.ipynb`."""


def adversarial_notebook():
    intro = (
        "This notebook is **Task 2**. It tests the hypothesis\n\n" + HYPOTHESIS + "\n\n"
        "by training the cVAE of `cVAE_Baseline.ipynb` with one addition: a **discriminator**, "
        "and a term in the loss that rewards the decoder for fooling it. Every other part (data, "
        "split, architecture, optimiser, learning-rate schedule, seeds, epochs) is identical, and the "
        "shared code cells "
        "are byte-for-byte the baseline's, so the discriminator term is the only difference "
        "between the two models.\n\n"
        "It writes `cVAE_DiscriminatorLoss.pth` (the cVAE and discriminator weights of each seed) "
        "and `cVAE_DiscriminatorLoss_history.json` (every curve, the lambda sweep and the chosen "
        "lambda).\n\n"
        "**Sections:** 1 Setup, 2 The data, 3 The cVAE, 4 Why a discriminator could sharpen a "
        "cVAE, 5 The discriminator, 6 The adversarial losses, 7 One training step, 8 Measuring "
        "sharpness, 9 Choosing lambda, 10 Convergence, 11 The final models, 12 Samples.")
    why = (MD_WHY.replace("__FIG_BLUR__", figure("why_mse_blurs.png", "Why MSE blurs"))
           .replace("__HYPOTHESIS__", HYPOTHESIS))
    return [
        markdown("md-title", fill(HEADER, __TITLE__="cVAE_DiscriminatorLoss", __INTRO__=intro)),
        markdown("md-setup", MD_SETUP),
        code("code-imports", imports("training")),
        code("code-style", CELL_STYLE),
        markdown("md-data", MD_DATA),
        code("code-data", CELL_DATA),
        markdown("md-audit", MD_AUDIT.replace("(section 3)", "(`cVAE_Baseline.ipynb`, section "
                                                             "3)")),
        code("code-audit", CELL_AUDIT),
        markdown("md-split", MD_SPLIT_TEMPLATE.replace("__NUM__", "2.2")),
        code("code-split", fill(CELL_SPLIT)),
        markdown("md-model", "## 3 The cVAE\n\nThe cVAE of `cVAE_Baseline.ipynb`, section 3, "
                             "copied byte for byte: the tutorial's encoder and decoder with one "
                             "3 x 3 kernel on each side, the tutorial's loss, and the "
                             "reparameterisation trick. The explanations and the architecture "
                             "figure are in that notebook."),
        code("code-encoder", CELL_ENCODER),
        code("code-decoder", CELL_DECODER),
        markdown("md-loss-cvae", "The tutorial's loss and cVAE, unchanged."),
        code("code-loss", CELL_LOSS),
        code("code-cvae", CELL_CVAE),
        markdown("md-settings", "### 3.1 Shared settings, batches and a shape check\n\nThe same "
                                "settings, loaders and helpers as the baseline (its sections 3.4 "
                                "to 4.4), so both models see identical batches in identical "
                                "order."),
        code("code-settings", fill(CELL_SETTINGS)),
        code("code-batches", CELL_BATCHES),
        markdown("md-model-check", "One forward pass through the cVAE, as in the baseline."),
        code("code-model-check", CELL_MODEL_CHECK),
        markdown("md-shared-helpers", "The baseline's evaluation pass, curve bookkeeping, "
                                      "optimiser and learning-rate schedule, unchanged."),
        code("code-evaluate", CELL_EVALUATE),
        code("code-history", CELL_HISTORY),
        markdown("md-convergence-helper", "The convergence statistic (the baseline's section "
                                          "4.4), and the decoding helpers."),
        code("code-convergence", CELL_CONVERGENCE),
        code("code-sample", CELL_SAMPLE),
        markdown("md-why", why),
        markdown("md-disc", MD_DISC.replace("__FIG_DISC__", figure("discriminator.png",
                                                                   "The discriminator"))),
        code("code-discriminator", CELL_DISCRIMINATOR),
        markdown("md-adv-losses", MD_ADV_LOSSES),
        code("code-adv-losses", CELL_ADV_LOSSES),
        markdown("md-train-step", MD_TRAIN_STEP.replace(
            "__FIG_STEP__", figure("training_step.png", "One training step"))),
        code("code-train-adv", CELL_TRAIN_ADV),
        markdown("md-d-check", MD_D_CHECK),
        code("code-d-check", CELL_D_CHECK),
        code("code-d-accuracy", CELL_D_ACCURACY),
        markdown("md-sharpness", MD_SHARPNESS.replace(
            "__FIG_LAPLACIAN__", figure("laplacian.png", "Laplacian variance"))),
        code("code-laplacian", CELL_LAPLACIAN),
        markdown("md-real-sharpness", "The development images' own sharpness, the reference "
                                      "every reconstruction is compared with."),
        code("code-real-sharpness", CELL_REAL_SHARPNESS),
        markdown("md-sweep", fill(MD_SWEEP)),
        code("code-cpu-state", CELL_CPU_STATE),
        code("code-sweep", fill(CELL_SWEEP)),
        markdown("md-choose", MD_CHOOSE),
        code("code-choose", CELL_CHOOSE),
        markdown("md-previews", MD_PREVIEWS),
        code("code-previews", CELL_PREVIEWS),
        markdown("md-adv-curves", MD_ADV_CURVES),
        code("code-adv-curves", CELL_ADV_CURVES),
        markdown("md-refit", MD_REFIT_ADV),
        code("code-refit-adv", CELL_REFIT_ADV),
        markdown("md-save", MD_SAVE_ADV),
        code("code-trivial", fill(CELL_TRIVIAL)),
        code("code-save-adv", CELL_SAVE_ADV),
        markdown("md-look", MD_LOOK_ADV),
        code("code-look-grid", CELL_LOOK_GRID),
    ]


# --------------------------------------------------------------------------- classifier notebook

def classifier_notebook():
    intro = (
        "This notebook trains the **standalone digit classifier** that `main_report.ipynb` uses "
        "to evaluate generated digits. It is an instrument, not one of the models being "
        "compared. Lecture 9 uses the same method: on p42 a classifier trained on real digits "
        "reaches 99% or more on them and 98% on a conditional GAN's samples, and p87 lists "
        "classifying generated data among the ways to measure it.\n\n"
        "It serves two purposes in `main_report.ipynb`:\n\n"
        "- **class consistency**: a sample drawn under class 3 should be recognised as a 3;\n"
        "- **a feature space**: its 128-unit layer describes a digit by 128 numbers, and "
        "comparing the average and spread of those numbers for real and generated digits gives a "
        "Frechet "
        "distance, the idea behind FID, in a space built for digits.\n\n"
        "It trains for a fixed five epochs on all 60,000 training images and nothing is tuned, "
        "so no development split is needed. It never makes a prediction on the test split. It "
        "writes `DigitClassifier.pth`.\n\n"
        "**Sections:** 1 Setup, 2 The data, 3 The model, 4 Training.")
    return [
        markdown("md-title", fill(HEADER, __TITLE__="DigitClassifier", __INTRO__=intro)),
        markdown("md-setup", "## 1 Setup\n\nThe libraries, the device and the seed, as in the "
                             "two cVAE notebooks."),
        code("code-imports", imports("training")),
        markdown("md-data", MD_DATA),
        code("code-data", CELL_DATA),
        markdown("md-audit", MD_AUDIT.replace("(section 3)", "(`cVAE_Baseline.ipynb`, section "
                                                             "3)")),
        code("code-audit", CELL_AUDIT),
        markdown("md-split", "### 2.2 The training images\n\nThe same seeded order as the cVAE "
                             "notebooks. The classifier trains on all of it, the fit and "
                             "development splits together, because nothing about it is chosen "
                             "on held-out data."),
        code("code-split", fill(CELL_SPLIT)),
        markdown("md-model", "## 3 The model\n\nTwo convolution blocks, each followed by "
                             "**max pooling**, which keeps the largest value in each 2 x 2 "
                             "square and so halves the map (20 to 10 to 5), then a 128-unit "
                             "layer and the 10 class scores. `features()` returns the 128-unit "
                             "layer.\n\n| Call | What you give it | What you get back |\n"
                             "|---|---|---|\n| `nn.MaxPool2d(2)` | maps (batch, c, n, n) | "
                             "(batch, c, n/2, n/2), the maximum of each 2 x 2 square |\n"
                             "| `F.cross_entropy(logits, labels)` | class scores and true "
                             "classes | the mean negative log-probability of the true class |"),
        code("code-classifier-model", CELL_CLASSIFIER),
        markdown("md-train", "## 4 Training\n\nAdam with a learning rate of 0.001, batches of "
                             "256, five epochs. Then a check that the loss ended far below the "
                             "ln 10 = 2.303 of guessing, and the weights are saved from the CPU."),
        code("code-cpu-state", CELL_CPU_STATE),
        code("code-clf-train", fill(CELL_CLF_TRAIN)),
        markdown("md-clf-loop", "The loop: score a batch, measure the cross-entropy, step."),
        code("code-clf-loop", fill(CELL_CLF_LOOP)),
        markdown("md-clf-save", "Saving the weights."),
        code("code-clf-save", CELL_CLF_SAVE),
    ]


# --------------------------------------------------------------------------- main_report notebook

def report_notebook():
    intro = (
        "This notebook reproduces **every number, table and figure in the report**. It trains "
        "nothing. It loads the three checkpoints (`cVAE_Baseline.pth`, "
        "`cVAE_DiscriminatorLoss.pth`, `DigitClassifier.pth`), the two training records "
        "(`*_history.json`) and the test split (`mnist_custom_test.pt`), and evaluates both "
        "cVAEs on the 10,000 test images. It runs top to bottom in a few minutes on a CPU.\n\n"
        "The hypothesis under test:\n\n" + HYPOTHESIS + "\n\n"
        "Every model and metric is explained where it is built: the cVAE in "
        "`cVAE_Baseline.ipynb` (section 3), the discriminator, the losses and the sharpness "
        "measures in `cVAE_DiscriminatorLoss.ipynb` (sections 4 to 8), and the classifier in "
        "`DigitClassifier.ipynb`. The lead-ins here say what each output is and how to read it."
        "\n\n**Sections:** 1 Setup, 2 Test data, 3 The models, 4 Checkpoints and training "
        "records, 5 Figure 1, 6 Reconstructions and samples, 7 Sharpness and pixel error, 8 Class "
        "consistency, 9 Frechet distances, 10 Diversity, 11 Figure 2, 12 Summary.")
    return [
        markdown("md-title", fill(HEADER, __TITLE__="main_report", __INTRO__=intro)),
        markdown("md-setup", MD_SETUP),
        code("code-imports-report", imports("report")),
        code("code-style", CELL_STYLE),
        markdown("md-test", "## 2 Test data\n\nThe 10,000 test images and their labels. The "
                            "class counts matter later: samples are drawn under exactly these "
                            "labels, so real and generated sets share one class mix."),
        code("code-test-data", CELL_MR_DATA),
        markdown("md-models", "## 3 The models\n\nThe class definitions, byte for byte the ones "
                              "the checkpoints were trained with: the cVAE "
                              "(`cVAE_Baseline.ipynb`, section 3), the discriminator "
                              "(`cVAE_DiscriminatorLoss.ipynb`, section 5) and the classifier "
                              "(`DigitClassifier.ipynb`, section 3)."),
        code("code-encoder", CELL_ENCODER),
        code("code-decoder", CELL_DECODER),
        markdown("md-models-2", "The cVAE, the discriminator and the classifier."),
        code("code-cvae", CELL_CVAE),
        code("code-discriminator", CELL_DISCRIMINATOR),
        markdown("md-models-3", "The classifier, and the helpers that decode images and measure "
                                "sharpness (`cVAE_DiscriminatorLoss.ipynb`, section 8)."),
        code("code-classifier-model", CELL_CLASSIFIER),
        code("code-sample", CELL_SAMPLE),
        markdown("md-models-4", "The sharpness measures and the convergence statistic."),
        code("code-laplacian", CELL_LAPLACIAN),
        code("code-convergence", CELL_CONVERGENCE),
        markdown("md-load", "## 4 Checkpoints and training records\n\nBoth cVAE files hold one "
                            "model per training seed. The printed parameter counts identify "
                            "the architectures."),
        code("code-load", CELL_MR_LOAD),
        code("code-load-2", CELL_MR_LOAD2),
        markdown("md-records", "### 4.1 How the models were trained\n\nFrom the history files: "
                               "the lambda chosen on the development split, the epochs and "
                               "seeds, where training ran, and the time per epoch."),
        code("code-records", CELL_MR_RECORDS),
        markdown("md-sweep", "### 4.2 The lambda sweep and the convergence check\n\nEach sweep "
                             "row is one development run: its reconstruction loss and sharpness "
                             "on the development split, the real development images' sharpness, "
                             "and the discriminator's accuracy. The chosen lambda is the one "
                             "whose sharpness comes closest to the real value without the "
                             "discriminator winning outright. A run has converged when the "
                             "change between its last two windows of epochs is within 1%."),
        code("code-sweep-table", CELL_MR_SWEEP),
        markdown("md-latents", "### 4.3 The latent codes\n\n`z_grid` holds the 12 codes for the "
                               "grids, and `z_eval` one code per test image for the "
                               "measurements. Both models decode the same codes, so every "
                               "sample comparison is paired."),
        code("code-latents", CELL_MR_LATENTS),
        markdown("md-fig1", "## 5 Figure 1: the two 12 x 10 grids\n\nTwelve samples per digit, "
                            "digits as rows, the brief's layout. Column k of both grids is the "
                            "same code `z_grid[k]`, so the two grids differ only by the model. "
                            "The classifier's accuracy on each grid's 120 images is printed "
                            "below it."),
        code("code-classify", CELL_MR_CLASSIFY.split("\n\n\nfor key")[0]),
        code("code-fig1", CELL_MR_FIG1),
        markdown("md-fig1-accuracy", "The classifier's accuracy on the 120 images of each grid."),
        code("code-grid-accuracy", "# The share of each grid's 120 images the classifier "
                                   "recognises as the digit they were drawn as.\nfor key" +
             CELL_MR_CLASSIFY.split("\n\n\nfor key")[1]),
        markdown("md-recon", "## 6 Reconstructions and samples\n\nEvery test image rebuilt "
                             "from `mu` by each model and seed, and one sample per test image, "
                             "drawn under that image's label from the shared code `z_eval`. "
                             "The reconstructions test the hypothesis as the brief states it; "
                             "the samples test the generation of Task 3."),
        code("code-recon", CELL_MR_RECON),
        markdown("md-bootstrap", "## 7 Sharpness and pixel error\n\nEach metric is printed with "
                                 "its **paired difference**, discriminator loss minus baseline, "
                                 "and a 95% bootstrap interval. If the interval excludes 0, the "
                                 "difference is not an accident of which test images were "
                                 "drawn. The per-seed differences show whether it survives "
                                 "retraining."),
        code("code-bootstrap", CELL_MR_BOOTSTRAP),
        code("code-metric-row", CELL_MR_METRIC_ROW),
        markdown("md-sharpness", "The rows: pixel error of the reconstructions (`mse_recon`), "
                                 "then Laplacian variance (`sharpness_*`, higher is sharper, up "
                                 "to the real value) and mid-grey fraction (`midgrey_*`, lower "
                                 "is sharper), each for reconstructions and for samples."),
        code("code-sharpness", CELL_MR_SHARPNESS),
        markdown("md-decision", "### 7.1 The decision rule\n\nThe rule was fixed before the "
                                "models were trained. The discriminator loss counts as "
                                "sharpening an image set only if three things hold together: "
                                "the Laplacian variance rises with an interval above 0, it "
                                "rises at every seed, and the mid-grey fraction falls. It is "
                                "applied to reconstructions and to samples separately."),
        code("code-decision", CELL_MR_DECISION),
        markdown("md-classes", "## 8 Class consistency\n\nThe classifier's accuracy on the "
                               "real test images is the reference point, not a ceiling: "
                               "generated digits can be easier to read than the most unusual "
                               "real handwriting. On the samples it measures "
                               "how often a digit drawn as class c is recognised as c. "
                               "Confidence is the probability it gives the intended class."),
        code("code-classes", CELL_MR_CLASSES),
        markdown("md-per-class", "Per digit, at seed 0."),
        code("code-per-class", CELL_MR_PER_CLASS),
        markdown("md-features", "## 9 Frechet distances\n\nThe Frechet distance compares two "
                                "clouds of feature vectors by their means and covariances: 0 "
                                "for identical clouds, larger as they drift apart. FID computes "
                                "it in the features of an ImageNet network at 299 x 299 pixels; "
                                "here it uses the digit classifier's 128-unit layer, built for "
                                "20 x 20 digits and available without a download."),
        code("code-features", CELL_MR_FEATURES),
        code("code-frechet", CELL_MR_FRECHET),
        markdown("md-halves", "The value depends on how many images are compared and in which "
                              "class mix, so every comparison uses halves of equal size and "
                              "mix, and the real-against-real value is the floor that no "
                              "model can beat."),
        code("code-halves", CELL_MR_HALVES),
        code("code-fd-rows", CELL_MR_FD_ROWS),
        markdown("md-spread", "## 10 Diversity\n\nMode collapse, a generator producing a few "
                              "images again and again, is the classic failure of adversarial "
                              "training. It shows as samples of one class that vary less than "
                              "real digits of that class."),
        code("code-spread", CELL_MR_SPREAD),
        markdown("md-fig2", "## 11 Figure 2: reconstructions and training curves\n\n(a) The "
                            "first test image of each class, rebuilt by each model at seed 0. "
                            "(b) The two development runs: the baseline and the chosen lambda."),
        code("code-fig2", CELL_MR_FIG2),
        code("code-fig2b", CELL_MR_FIG2B),
        markdown("md-summary", "## 12 Summary\n\nThe headline numbers in one machine-readable "
                               "block."),
        code("code-summary", CELL_MR_SUMMARY),
    ]


def write(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    os.makedirs(NBDIR, exist_ok=True)
    path = f"{NBDIR}/{name}"
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    return path, notebook


# =========================================================================== generator rails

FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


CONSTRUCTOR = re.compile(r"torch\.(zeros|ones|tensor|eye|arange|randn|full|empty)\(")
NESTED_FSTRING = re.compile(r"""\bf"[^"\n]*\{[^}\n]*"[^}\n]*\}|\bf'[^'\n]*\{[^}\n]*'[^}\n]*\}""")
BACKSLASH_IN_BRACES = re.compile(r"""\bf["'][^"'\n]*\{[^}\n]*\\[^}\n]*\}""")
EXPECTED_FIGURES = {"cVAE_Baseline.ipynb": 1, "cVAE_DiscriminatorLoss.ipynb": 4,
                    "DigitClassifier.ipynb": 0, "main_report.ipynb": 0}

built = {}
for name, cells in [("cVAE_Baseline.ipynb", baseline_notebook()),
                    ("cVAE_DiscriminatorLoss.ipynb", adversarial_notebook()),
                    ("DigitClassifier.ipynb", classifier_notebook()),
                    ("main_report.ipynb", report_notebook())]:
    path, notebook = write(name, cells)
    built[name] = notebook
    print(f"wrote {path}  ({len(cells)} cells)")

print("\ngenerator rails")
shared_sources = {}
for name, notebook in built.items():
    raw = json.dumps(notebook, ensure_ascii=False)
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    source = "\n".join("".join(c["source"]) for c in code_cells)
    lines = [line for line in source.split("\n") if line.strip()]
    comments = [line for line in lines if line.strip().startswith("#")]
    ratio = len(comments) / len(lines)
    ids = [c["id"] for c in notebook["cells"]]
    longest = max(len(line) for line in source.split("\n"))

    check(f"{name}: no em or en dash", "\u2014" not in raw and "\u2013" not in raw)
    check(f"{name}: no control characters in any cell",
          not any(ord(ch) < 32 and ch not in "\n\t"
                  for c in notebook["cells"] for ch in "".join(c["source"])))
    check(f"{name}: every cell has a unique id", len(ids) == len(set(ids)), f"{len(ids)} cells")
    check(f"{name}: every code cell carries a comment",
          all(any(line.strip().startswith("#") for line in c["source"]) for c in code_cells))
    check(f"{name}: source lines keep their newlines",
          all(all(line.endswith("\n") for line in c["source"][:-1])
              for c in notebook["cells"] if len(c["source"]) > 1))
    check(f"{name}: comment ratio at least 0.12", ratio >= 0.12,
          f"{len(comments)}/{len(lines)} = {ratio:.0%}")
    check(f"{name}: no code line longer than 100 characters", longest <= 100,
          f"longest {longest}")
    check(f"{name}: no tqdm", "tqdm" not in source)
    check(f"{name}: no placeholder left", "__" not in re.sub(r"__init__|__version__|__name__",
                                                             "", raw))
    check(f"{name}: no f-string needing Python 3.12 (PEP 701)",
          not NESTED_FSTRING.search(source) and not BACKSLASH_IN_BRACES.search(source))
    bare = [line.strip() for line in source.split("\n")
            if CONSTRUCTOR.search(line) and "device" not in line]
    check(f"{name}: every tensor constructor names the device", not bare, "; ".join(bare)[:200])
    run = 0
    worst = 0
    for cell in notebook["cells"]:
        run = run + 1 if cell["cell_type"] == "code" else 0
        worst = max(worst, run)
    check(f"{name}: at most 2 code cells in a row without markdown", worst <= 2, f"{worst}")
    check(f"{name}: every cell parses as Python",
          all(compile("".join(c["source"]), f"{name}:{c['id']}", "exec") or True
              for c in code_cells))
    embedded = sum("data:image/png;base64," in "".join(c["source"])
                   for c in notebook["cells"] if c["cell_type"] == "markdown")
    check(f"{name}: embeds {EXPECTED_FIGURES[name]} explanatory figures",
          embedded == EXPECTED_FIGURES[name], f"{embedded}")
    for cell in code_cells:
        shared_sources.setdefault(cell["id"], set()).add("".join(cell["source"]))

    if name == "main_report.ipynb":
        check(f"{name}: trains nothing",
              not re.search(r"\.backward\(|optim\.|optimizer|\.train\(\)|training_loop|"
                            r"train_epoch", source))
        check(f"{name}: never calls a cVAE's forward()",
              not re.search(r"\b(model|network)\(|(baseline|adversarial|models)\[[^\]]*\]\(",
                            source))
        check(f"{name}: every torch.randn has a generator",
              all("generator=" in line for line in source.split("\n") if "torch.randn(" in line))
        asserts = [line.strip() for line in source.split("\n") if line.strip().startswith("assert")]
        check(f"{name}: no assert on a metric value",
              not any(re.search(r"[<>]|\d\.\d", line) for line in asserts), "; ".join(asserts))
        check(f"{name}: loads with weights_only and map_location",
              source.count("weights_only=True") == 3
              and source.count("map_location=device") == 2)
        check(f"{name}: reads lambda from the history file, never a literal",
              'LAMBDA = adversarial_record["lambda"]' in source
              and not re.search(r"LAMBDA\s*=\s*\d", source))
    else:
        test_lines = [line.strip() for line in source.split("\n") if "test_images" in line]
        check(f"{name}: test_images appears only in the line that saves the split",
              test_lines == ['test_split = {key: data[key] for key in ("test_images", '
                             '"test_labels")}'], "; ".join(test_lines))
        if name != "DigitClassifier.ipynb":
            check(f"{name}: writes its history json", "_history.json" in source)

for cell_id, versions in shared_sources.items():
    if len(versions) > 1:
        check(f"shared cell {cell_id} is byte-identical in every notebook", False,
              f"{len(versions)} versions")

print("\n" + ("GENERATOR CHECKS PASSED" if not FAILURES
             else f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES)))
raise SystemExit(1 if FAILURES else 0)
