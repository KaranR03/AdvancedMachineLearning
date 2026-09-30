r"""Generate the three Project 7 notebooks from one template.

    notebook/DDPM_CosineSchedule.ipynb   Task 1: two conditional DDPMs, linear and cosine schedule
    notebook/DigitClassifier.ipynb       the standalone classifier used for evaluation
    notebook/main_report.ipynb           Task 2: every number and figure in the report

The model, schedule and sampler code is written once here and placed byte-identical in every
notebook that needs it, so the checkpoints main_report.ipynb loads are read by exactly the code
that trained them. The tutorial's own definitions (extract, linear_beta_schedule, ConvBlock,
UNet_cond) are not retyped: they are cut out of weeks/week-09/Week_9_solution.ipynb by the
syntax tree, so they stay byte-identical to Tutorial 9.3.

The generated notebooks carry no outputs. Execute them on the Hub GPU node (README, Hub route) or
rehearse the whole pipeline at tiny sizes with --smoke and tools/smoke_test.py.

Run:  python tools/build_notebooks.py            full sizes, written to notebook/
      python tools/build_notebooks.py --smoke    tiny sizes, written to the job's scratch room
      add --draft to skip the explanatory-figure count while the figures are being drawn
"""
import ast
import base64
import io
import json
import os
import re
import sys

REPO = "f:/document/IFN_680_Advanced_Machine_Learning_and_Applications"
BASE = f"{REPO}/weeks/week-10/project-7-improving-ddpm"
TUTORIAL = f"{REPO}/weeks/week-09/Week_9_solution.ipynb"
SMOKE = "--smoke" in sys.argv
DRAFT = "--draft" in sys.argv
ROOM = os.environ.get("CLAUDE_JOB_DIR", "C:/Users/Admin/.claude/jobs/8640b033") + "/tmp/p7/smoke"
NBDIR = ROOM if SMOKE else f"{BASE}/notebook"
FIGURES = f"{BASE}/tools/explanatory"

# Sizes. The smoke run exercises every code path in a few minutes on a CPU; its numbers mean
# nothing and it never writes into notebook/. It cuts epochs and data only: the seeds, the step
# counts and the snapshot list keep their full length, because generated text changes shape
# with the length of a list.
if SMOKE:
    SIZES = dict(N_TRAIN=1000, EPOCHS=5, SNAPSHOTS="(1, 2, 3, 4)", PRINT_EVERY=1,
                 SEEDS="(0, 1, 2)", CLASSIFIER_EPOCHS=1, TRIVIAL_BAR="2.0",
                 CHANCE_BAR="2 * np.log(NUM_CLASSES)", N_EVAL=120, RESAMPLES=50,
                 EPOCH_STEPS=20)
else:
    SIZES = dict(N_TRAIN=60000, EPOCHS=150, SNAPSHOTS="(10, 25, 50, 100)", PRINT_EVERY=10,
                 SEEDS="(0, 1, 2)", CLASSIFIER_EPOCHS=15, TRIVIAL_BAR="0.1",
                 CHANCE_BAR="np.log(NUM_CLASSES) / 2", N_EVAL=1000, RESAMPLES=1000,
                 EPOCH_STEPS=50)


def figure(name, alt):
    """An explanatory PNG from tools/build_figures.py, embedded as a base64 data URI."""
    path = f"{FIGURES}/{name}"
    if DRAFT and not os.path.exists(path):
        return f"*(figure {name} pending)*"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"![{alt}](data:image/png;base64,{encoded})"


# =========================================================================== the tutorial's code

def tutorial_definitions():
    """Every top-level def and class of the Week 9 solution, keyed by name, exactly as written."""
    with open(TUTORIAL, encoding="utf-8") as handle:
        cells = json.load(handle)["cells"]
    found = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                found[node.name] = ast.get_source_segment(source, node)
    return found


TUTORIAL_DEFS = tutorial_definitions()
# The four definitions carried across unchanged. verify.py re-reads the tutorial and compares.
CARRIED = ("extract", "linear_beta_schedule", "ConvBlock", "UNet_cond")

# =========================================================================== shared code cells

CELL_IMPORTS = r'''# The Week 9 tutorial's stack. json carries the training record to main_report.ipynb, and
# utils.make_grid lays out digit grids the way the tutorial's figures do.
import json
__IMPORT_OS__import platform
__IMPORT_TIME__
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
__IMPORT_DATA__from torchvision import utils

# The brief asks for the IFN680 GPU environment. The same code runs on a CPU when no GPU is
# visible, so the notebook runs anywhere and every tensor is placed with .to(device).
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device      : {device}")
print(f"python      : {platform.python_version()}")
print(f"torch       : {torch.__version__}")
print(f"torchvision : {torchvision.__version__}")
print(f"numpy       : {np.__version__}")

# One seed for everything that is not given its own: initial weights and random draws.
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

CELL_SCALE = r'''# The tutorial's transform, x*2-1, takes every pixel from [0, 1] to [-1, 1]. Batches of 128 as
# in the tutorial; each training run builds its own seeded DataLoader over this dataset.
N_TRAIN = __N_TRAIN__
train_dataset = TensorDataset(train_images[:N_TRAIN] * 2 - 1, train_labels[:N_TRAIN])
bs_train = 128
image_size = 20
n_class = 10
scaled = train_dataset.tensors[0]
print(f"training images: {len(train_dataset):,}, pixels from {scaled.min():.0f} to "
      f"{scaled.max():.0f}, batches of {bs_train}")'''


def tutorial_cell(comment, *names, preamble=""):
    """A code cell holding the named tutorial definitions, byte-identical, under a comment."""
    body = "\n\n".join(TUTORIAL_DEFS[name] for name in names)
    return comment + "\n" + preamble + body


CELL_TUTORIAL_FORWARD = (
    "# Supplied by Tutorial 9.3 and reproduced unchanged. extract() picks each image's schedule\n"
    "# value at its own step t, and linear_beta_schedule() is the baseline schedule under test.\n"
    + TUTORIAL_DEFS["extract"] + "\n\n\ntimesteps = 1000\n\n\n"
    + TUTORIAL_DEFS["linear_beta_schedule"])

CELL_COSINE = r'''# The schedule under test, written from Nichol and Dhariwal (2021), eq. 17: alpha_bar follows
# a squared cosine of t/T, offset by s = 0.008, and each beta is recovered as
# 1 - alpha_bar(t) / alpha_bar(t-1). Double precision, because near t = T this divides two very
# small numbers; beta is clipped at 0.999 because alpha_bar(T) is 0, which would make it 1.
def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
    f = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clip(betas, max=0.999).float()'''

CELL_SCHEDULES = r'''# One dictionary per schedule holds the three arrays the tutorial keeps as globals, so both
# schedules can be used side by side. alpha_bar is the running product of the alphas.
def make_schedule(betas):
    betas = betas.to(device)
    alphas = 1.0 - betas
    return {"betas": betas, "alphas": alphas, "alpha_bars": torch.cumprod(alphas, dim=0)}


SCHEDULE_FUNCTIONS = {"linear": linear_beta_schedule, "cosine": cosine_beta_schedule}
schedules = {name: make_schedule(build(timesteps)) for name, build in SCHEDULE_FUNCTIONS.items()}
for name, schedule in schedules.items():
    betas, alpha_bars = schedule["betas"], schedule["alpha_bars"]
    print(f"{name:6s} | beta from {betas[0]:.6f} to {betas[-1]:.4f} | alpha_bar from "
          f"{alpha_bars[0]:.6f} down to {alpha_bars[-1]:.2e}")
    # alpha_bar must fall at every step, or the process would un-noise the image somewhere
    assert bool((alpha_bars[1:] < alpha_bars[:-1]).all())'''

CELL_QSAMPLE = r'''# The tutorial's q_sample with one change: the schedule is an argument instead of a global.
# It jumps straight from a clean image x0 to step t:
# x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise.
def q_sample(x0, t, schedule, noise=None):
    if noise is None:
        noise = torch.randn_like(x0).to(device)
    alpha_bars = schedule["alpha_bars"]
    sqrt_alpha_bar = torch.sqrt(extract(alpha_bars,t,x0.shape))
    sqrt_one_minus_alpha_bar = torch.sqrt(extract(1-alpha_bars,t,x0.shape))
    return sqrt_alpha_bar*x0 + noise*sqrt_one_minus_alpha_bar'''

CELL_NOISING = r'''# The tutorial's picture of the forward process, drawn for both schedules from the same four
# training digits and the same noise, so every difference between the two rows is the schedule.
timesteps_investigated = [25, 50, 100, 250, 500, 999]
img = train_dataset.tensors[0][:4].to(device)
noise = torch.randn(img.shape, generator=torch.Generator().manual_seed(SEED)).to(device)
fig, axes = plt.subplots(2, len(timesteps_investigated), figsize=(7.5, 3.0))
for row, (name, schedule) in enumerate(schedules.items()):
    for col, step in enumerate(timesteps_investigated):
        t = torch.full((img.shape[0],), step, device=device, dtype=torch.long)
        noised = q_sample(img, t, schedule, noise).clamp(-1, 1)
        grid = utils.make_grid((noised + 1) / 2, nrow=2, pad_value=1)[0].cpu()
        axes[row, col].imshow(grid, cmap="gray", vmin=0, vmax=1)
        axes[row, col].set_title(f"{name}, t = {step}")
        axes[row, col].axis("off")
plt.tight_layout()
plt.show()'''

CELL_ALLOCATION = r'''# Where each schedule spends its 1000 steps. alpha_bar is the share of the clean image's signal
# left at step t; below 0.01 the image is almost pure noise. log SNR carries the same
# information on a scale that does not bunch up near 0 and 1.
fig, (ax_bar, ax_snr) = plt.subplots(1, 2, figsize=(7.5, 2.6))
steps = np.arange(timesteps)
for (name, schedule), colour in zip(schedules.items(), (WARM, ACCENT)):
    alpha_bars = schedule["alpha_bars"].double().cpu().numpy()
    ax_bar.plot(steps, alpha_bars, color=colour, label=name)
    ax_snr.plot(steps, np.log(alpha_bars / (1 - alpha_bars)), color=colour, label=name)
    print(f"{name:6s} | steps with alpha_bar below 0.01: {(alpha_bars < 0.01).mean():.1%} | "
          f"alpha_bar at the last step: {alpha_bars[-1]:.2e}")
ax_bar.axhline(0.01, color=MUTED, lw=0.8, ls="--")
ax_bar.set(xlabel="step t", ylabel="alpha_bar (signal left)")
ax_snr.set(xlabel="step t", ylabel="log SNR")
ax_bar.legend(frameon=False)
plt.tight_layout()
plt.show()'''

CELL_SHORT_T = r'''# The other reading of "fewer time steps": a shorter forward process. The tutorial's linear
# betas run from 0.0001 to 0.02 whatever T is, so a short process ends far from pure noise;
# the cosine schedule is defined on t / T, so it ends at noise for every T.
for short in (100, 250, 500):
    linear_end = torch.cumprod(1 - linear_beta_schedule(short), dim=0)[-1]
    cosine_end = torch.cumprod(1 - cosine_beta_schedule(short), dim=0)[-1]
    print(f"T = {short:3d} | alpha_bar at the last step: linear {linear_end:.3f}, "
          f"cosine {cosine_end:.2e}")'''

CELL_CONVBLOCK = tutorial_cell(
    "# Supplied by Tutorial 9.3 and reproduced unchanged: two 3 x 3 convolutions with ReLU, the\n"
    "# building block of every level of the U-Net.",
    "ConvBlock", preamble="channels = 1 # for MNIST\n\n")

CELL_UNET = tutorial_cell(
    "# Supplied by Tutorial 9.3 (its solution) and reproduced unchanged: the conditional U-Net.\n"
    "# The step t (divided by T) and the class each become 32 numbers, summed into cond_emb and\n"
    "# added at every level through a 1 x 1 convolution.",
    "UNet_cond")

CELL_SHAPE = r'''# One forward pass on eight training images at random steps. Two poolings take 20 to 10 to 5
# and the upsamplings bring it back, so the tutorial's U-Net accepts 20 x 20 images unchanged.
check_model = UNet_cond(ch=32, n_classes=n_class).to(device)
x_check = train_dataset.tensors[0][:8].to(device)
y_check = train_dataset.tensors[1][:8].to(device)
t_check = torch.randint(0, timesteps, (8,), device=device)
predicted = check_model(x_check, t_check, y_check)
print(f"input {tuple(x_check.shape)} -> predicted noise {tuple(predicted.shape)}")
print(f"parameters: {sum(p.numel() for p in check_model.parameters()):,}")'''

CELL_SETTINGS = r'''# The tutorial solution's settings for the conditional model, shared by every run so the
# schedule is the only difference between the two models.
epochs = __EPOCHS__
lr = 2e-4
n_channels_unet = 32
SEEDS = __SEEDS__
SNAPSHOT_EPOCHS = __SNAPSHOTS__  # seed 0 also keeps its weights at these epochs
PRINT_EVERY = __PRINT_EVERY__
STEP_COUNTS = (10, 20, 50, 100, 250, 1000)  # the sampling budgets K compared in section 6'''

CELL_CPU_STATE = r'''# Checkpoints are saved from the CPU, so they load on a machine without a GPU.
def cpu_state(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}'''

CELL_TRAIN_FN = r'''# The tutorial's training loop for UNet_cond, in a function so it runs once per schedule and
# seed. Changes: the schedule is an argument; the shuffle has its own seeded generator; the loss
# is averaged over each epoch on the device (one transfer per epoch, no progress bar);
# and seed 0 keeps copies of its weights at SNAPSHOT_EPOCHS.
def train_cddpm(name, schedule, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = UNet_cond(ch=n_channels_unet, n_classes=n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_dataloader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
    record = {"loss": [], "seconds": [], "snapshots": {}}
    for epoch in range(epochs):
        started = time.time()
        model.train()
        loss_sum = torch.zeros((), device=device)
        for imgs, labels in train_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            batch_size = imgs.shape[0]
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(imgs)
            x_t = q_sample(imgs, t, schedule, noise=noise)

            pred_noise = model(x_t,t,labels)
            loss = loss_fn(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * batch_size
        record["loss"].append(loss_sum.item() / len(train_dataset))
        record["seconds"].append(time.time() - started)
        if seed == SEEDS[0] and epoch + 1 in SNAPSHOT_EPOCHS:
            record["snapshots"][epoch + 1] = cpu_state(model)
        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            print(f"{name} seed {seed} | epoch {epoch + 1:3d}/{epochs} | "
                  f"loss {record['loss'][-1]:.4f} | {record['seconds'][-1]:.1f} s")
    return model, record'''

CELL_RUN = r'''# Six runs, seed by seed. A linear run and the cosine run of the same seed start from the same
# initial weights and draw the same batches, steps and noise, because each resets the seed.
# Each model's weights move to the CPU when it finishes, so the GPU holds one model at a time.
weights, records = {}, {}
for seed in SEEDS:
    for name, schedule in schedules.items():
        model, record = train_cddpm(name, schedule, seed)
        weights[name, seed] = cpu_state(model)
        records[name, seed] = record
        del model'''

CELL_TRIVIAL = r'''# Predicting zero noise everywhere scores a loss of 1, the variance of the noise. Every run must
# end far below it; a run that learned nothing would sit near it.
for (name, seed), record in records.items():
    print(f"{name:6s} seed {seed} | final loss {record['loss'][-1]:.4f} | "
          f"{np.mean(record['seconds']):.1f} s per epoch")
    assert record["loss"][-1] < __TRIVIAL_BAR__, f"{name} seed {seed} did not learn"'''

CELL_CURVES = r'''# Every run's loss per epoch, seed 0 solid. The statistic compares the mean loss over the last
# WINDOW epochs with the WINDOW before; a change near 0 means training has levelled off.
WINDOW = min(10, epochs // 2)
fig, ax = plt.subplots(figsize=(5.0, 2.8))
for (name, seed), record in records.items():
    colour = WARM if name == "linear" else ACCENT
    ax.plot(range(1, epochs + 1), record["loss"], color=colour, lw=1.2 if seed == 0 else 0.6,
            alpha=1.0 if seed == 0 else 0.5, label=f"{name}" if seed == 0 else None)
    last = np.mean(record["loss"][-WINDOW:])
    before = np.mean(record["loss"][-2 * WINDOW:-WINDOW])
    print(f"{name:6s} seed {seed} | last {WINDOW} epochs {last:.4f} | change on the "
          f"{WINDOW} before {(last - before) / before:+.1%}")
ax.set(xlabel="epoch", ylabel="training loss (MSE on the noise)", yscale="log")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()'''

CELL_SAVE = r'''# One file per schedule: a dictionary of CPU state_dicts keyed "seed_0", "seed_1", ... and, for
# seed 0, "epoch_10", "epoch_25", ...; the curves, durations and settings go into the JSON file.
for name in schedules:
    checkpoint = {f"seed_{seed}": weights[name, seed] for seed in SEEDS}
    for epoch, state in records[name, SEEDS[0]]["snapshots"].items():
        checkpoint[f"epoch_{epoch}"] = state
    torch.save(checkpoint, f"ddpm_{name}.pth")
history = {"epochs": epochs, "lr": lr, "batch_size": bs_train, "timesteps": timesteps,
           "n_channels_unet": n_channels_unet, "seeds": list(SEEDS),
           "snapshot_epochs": list(SNAPSHOT_EPOCHS), "n_train": len(train_dataset),
           "device": str(device), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
           "runs": {f"{name}_seed_{seed}": {"loss": r["loss"], "seconds": r["seconds"]}
                    for (name, seed), r in records.items()}}
with open("DDPM_history.json", "w") as handle:
    json.dump(history, handle, indent=1)
print(f"saved {', '.join(f'ddpm_{name}.pth' for name in schedules)} and DDPM_history.json")'''

CELL_STEPS = r'''# Which of the 1000 steps a K-step sampler visits: K evenly spaced values from the first step
# to the last, rounded to whole steps (Nichol and Dhariwal, section 4). K = 1000 visits every
# step, which is the tutorial's loop.
def sampling_steps(n_steps):
    return np.round(np.linspace(0, timesteps - 1, n_steps)).astype(int).tolist()


for n_steps in (10, 20):
    print(f"K = {n_steps} visits steps {sampling_steps(n_steps)}")'''

CELL_P_SAMPLE = r'''# The tutorial's reverse step (p_sample_cDDPM), able to jump from step t_index back to any
# earlier step prev_index, with beta recomputed from the two alpha_bars (the tutorial's beta_t
# when prev_index = t_index - 1). It is written in its x0 form: the noise estimate gives a
# clean-image estimate x0_hat, and the step is a weighted average of x0_hat and x_t. Unclipped,
# this is the tutorial's update (checked in DDPM_CosineSchedule.ipynb, section 6.4); clipping
# x0_hat to the data range stops one bad noise estimate from throwing the image off scale
# (DDPM_CosineSchedule.ipynb, section 6.5).
@torch.no_grad()
def p_sample_cDDPM(model, x_t, t_index, y, schedule, prev_index, z, clip=True):
    t = torch.full((x_t.shape[0],), t_index, device=x_t.device, dtype=torch.long)
    alpha_bar = schedule["alpha_bars"][t_index]
    alpha_bar_prev = (schedule["alpha_bars"][prev_index] if prev_index >= 0
                      else alpha_bar.new_ones(()))
    beta = 1 - alpha_bar / alpha_bar_prev
    alpha = 1 - beta
    pred_noise = model(x_t, t, y)
    x0_hat = (x_t - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
    if clip:
        x0_hat = x0_hat.clamp(-1, 1)
    mean = (alpha_bar_prev.sqrt() * beta / (1 - alpha_bar) * x0_hat
            + alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar) * x_t)
    # the last step adds no noise, as in the tutorial's z = 0 at t_index == 0
    return mean + beta.sqrt() * z if prev_index >= 0 else mean'''

CELL_DDIM = r'''# DDIM (Song et al., 2021): the same network and the same x0_hat, but a deterministic step that
# moves x0_hat to the earlier step's noise level and reuses the noise estimate instead of
# drawing fresh noise. The estimate is recomputed from the clipped x0_hat so the two agree.
@torch.no_grad()
def ddim_step(model, x_t, t_index, y, schedule, prev_index, clip=True):
    t = torch.full((x_t.shape[0],), t_index, device=x_t.device, dtype=torch.long)
    alpha_bar = schedule["alpha_bars"][t_index]
    alpha_bar_prev = (schedule["alpha_bars"][prev_index] if prev_index >= 0
                      else alpha_bar.new_ones(()))
    pred_noise = model(x_t, t, y)
    x0_hat = (x_t - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
    if clip:
        x0_hat = x0_hat.clamp(-1, 1)
        pred_noise = (x_t - alpha_bar.sqrt() * x0_hat) / (1 - alpha_bar).sqrt()
    return alpha_bar_prev.sqrt() * x0_hat + (1 - alpha_bar_prev).sqrt() * pred_noise'''

CELL_LOOP = r'''# The tutorial's sampling loop over the steps sampling_steps lists: start from pure noise and step
# down to 0. Every random draw comes from one seeded CPU generator and is then moved to the
# device, so a seed gives the same samples on any machine and both schedules start alike.
@torch.no_grad()
def p_sample_loop_cDDPM(model, y, schedule, n_steps=timesteps, sampler="ddpm", seed=1337,
                        clip=True):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(y.shape[0], channels, image_size, image_size, generator=generator).to(device)
    steps = sampling_steps(n_steps)
    for i in reversed(range(len(steps))):
        prev_index = steps[i - 1] if i > 0 else -1
        if sampler == "ddim":
            x = ddim_step(model, x, steps[i], y, schedule, prev_index, clip)
        else:
            z = torch.randn(x.shape, generator=generator).to(device)
            x = p_sample_cDDPM(model, x, steps[i], y, schedule, prev_index, z, clip)
    return x.clamp(-1,1)


# Large sets are drawn in chunks, chunk k with seed + k, and returned in [0, 1] on the CPU.
def generate(model, labels, schedule, n_steps, sampler, seed=1337, batch_size=1000):
    parts = []
    for number, start in enumerate(range(0, len(labels), batch_size)):
        y = labels[start:start + batch_size].to(device)
        x = p_sample_loop_cDDPM(model, y, schedule, n_steps, sampler, seed + number)
        parts.append(((x + 1) / 2).cpu())
    return torch.cat(parts)'''

CELL_EQUIVALENCE = r'''# Section 6.2's claim, checked: unclipped and with no step skipped, the x0 form equals the
# tutorial's update at every step. Double precision, and a stand-in network that returns one
# fixed noise estimate, so only the two formulas are compared.
def tutorial_update(schedule, x_t, t_index, pred_noise, z):
    t = torch.full((x_t.shape[0],), t_index, device=x_t.device, dtype=torch.long)
    beta = extract(schedule["betas"], t, x_t.shape)
    alpha = extract(schedule["alphas"], t, x_t.shape)
    alpha_bar = extract(schedule["alpha_bars"], t, x_t.shape)
    sigma = torch.sqrt(beta)
    z = 0 if t_index == 0 else z
    return 1.0 / alpha.sqrt() * (x_t - beta / ( (1 - alpha_bar).sqrt() ) * pred_noise) + sigma*z


check_generator = torch.Generator().manual_seed(SEED)
shape = (16, channels, image_size, image_size)
x_t, estimate, z = (torch.randn(shape, generator=check_generator, dtype=torch.float64).to(device)
                    for _ in range(3))
y_unused = torch.zeros(16, dtype=torch.long, device=device)
for name, build in SCHEDULE_FUNCTIONS.items():
    schedule64 = make_schedule(build(timesteps).double())
    worst = max((p_sample_cDDPM(lambda x, t, y: estimate, x_t, s, y_unused, schedule64, s - 1, z,
                                clip=False) - tutorial_update(schedule64, x_t, s, estimate, z))
                .abs().max().item() for s in range(timesteps))
    print(f"{name:6s} | largest difference from the tutorial's update over all "
          f"{timesteps} steps: {worst:.1e}")
    assert worst < 1e-6, "the x0 form must reproduce the tutorial's step"'''

CELL_GAIN = r'''# How much an error in the noise estimate is multiplied on the first reverse step, where the
# network sees pure noise, in the tutorial's form: beta / (sqrt(alpha) * sqrt(1 - alpha_bar)).
print("first reverse step: multiplier on an error in the noise estimate, by number of steps K")
for name, schedule in schedules.items():
    cells = []
    for n_steps in STEP_COUNTS:
        first, second = sampling_steps(n_steps)[-1], sampling_steps(n_steps)[-2]
        alpha_bar = schedule["alpha_bars"][first].double()
        alpha = alpha_bar / schedule["alpha_bars"][second].double()
        gain = (1 - alpha) / (alpha.sqrt() * (1 - alpha_bar).sqrt())
        cells.append(f"K={n_steps} {gain:,.2f}")
    print(f"{name:6s} | " + " | ".join(cells))'''

CELL_GRIDS = r'''# The tutorial's last step in the brief's layout: 12 samples per digit, digits as rows, from
# each schedule's seed-0 model with the full 1000-step sampler and the same starting noise.
sample_per_class = 12
labels = torch.arange(0,10).unsqueeze(1).repeat(1,sample_per_class).view(-1).to(device)
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
for ax, name in zip(axes, schedules):
    model = UNet_cond(ch=n_channels_unet, n_classes=n_class).to(device)
    _ = model.load_state_dict(weights[name, SEEDS[0]])
    samples = (p_sample_loop_cDDPM(model, labels, schedules[name]) + 1.0) / 2.0
    ax.imshow(utils.make_grid(samples, nrow=12, pad_value=1)[0].cpu(), cmap="gray",
              vmin=0, vmax=1)
    ax.set_title(f"{name} schedule, {timesteps} steps")
    ax.axis("off")
plt.tight_layout()
plt.show()'''

CELL_REAL = r'''# Twelve training images of each digit, in the same layout, for comparison.
first = torch.cat([torch.nonzero(train_labels == digit)[:12, 0] for digit in range(10)])
fig, ax = plt.subplots(figsize=(3.75, 3.6))
ax.imshow(utils.make_grid(train_images[first], nrow=12, pad_value=1)[0], cmap="gray",
          vmin=0, vmax=1)
ax.set_title("real training digits")
ax.axis("off")
plt.tight_layout()
plt.show()'''

# =========================================================================== classifier cells

CELL_CLASSIFIER = r'''# The evaluation classifier, in the style of lecture 9 p42's: two convolution blocks and a
# 128-unit layer. features() returns that layer, the space the Frechet distances use. It
# inverts each image first (1 - x), so strokes are bright on black as in the MNIST this design
# comes from, and the zero padding at the border matches the background.
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
        return self.feature_extractor(1 - x)

    def forward(self, x):
        return self.output_layer(self.features(x))'''

CELL_CLF_TRAIN = r'''# A fixed number of epochs on every training image, with no model selection: the classifier is
# a measuring instrument, so nothing about it is tuned and no development split is needed.
NUM_CLASSES, CLASSIFIER_EPOCHS = 10, __CLASSIFIER_EPOCHS__
_ = torch.manual_seed(SEED)
classifier = DigitClassifier(NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(train_images[:__N_TRAIN__], train_labels[:__N_TRAIN__]),
                    batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(SEED))
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
    print(f"epoch {epoch + 1:2d}: training loss {loss_sum / count:.4f}, "
          f"training accuracy {correct / count:.4f}")
# Chance is a loss of ln 10 = 2.303; a working classifier ends far below it.
assert loss_sum / count < __CHANCE_BAR__, "the classifier did not learn"'''

CELL_CLF_SAVE = r'''# A plain state_dict saved from the CPU; main_report.ipynb rebuilds the class and loads it.
torch.save(cpu_state(classifier), "DigitClassifier.pth")
print("saved DigitClassifier.pth")'''


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
                 "__IMPORT_DATA__": "from torch.utils.data import DataLoader, TensorDataset\n"},
    "report": {"__IMPORT_OS__": "", "__IMPORT_TIME__": "", "__IMPORT_DATA__": ""},
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


HEADER = r"""# IFN680 Project 7: Improving DDPM - __TITLE__

**Group 4** - Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

__INTRO__"""

HYPOTHESIS = ("> *A cosine schedule allows a denoising diffusion probabilistic model to generate "
              "high-quality samples using fewer time steps than a linear schedule.*")

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

The brief describes the file only as a custom MNIST, so its properties are measured before
anything is trained on it. The images are **20 x 20**, not the tutorial's 28 x 28, and the
background is **white** (pixel value 1.0) with dark strokes, the reverse of the tutorial's MNIST.
Neither needs a change to the diffusion model: it learns to remove noise from whatever images it
is given, and the U-Net's two halvings of the image size work at 20 x 20 (section 4)."""

MD_SCALE = r"""### 2.2 Scaling and batches

The tutorial's transform `x*2-1` maps every pixel from [0, 1] to [-1, 1]. The forward process of
section 3 ends in pure noise with mean 0 and variance 1, so the clean images are put on a
comparable scale, centred on 0. The tutorial trains in shuffled batches of 128; each training run
in section 5 builds its own `DataLoader` over this dataset.

| Call | What you give it | What you get back |
|---|---|---|
| `TensorDataset(images, labels)` | tensors with the same first dimension | a dataset whose item i is (images[i], labels[i]) |"""

MD_FORWARD = r"""## 3 The forward diffusion process

A diffusion model learns to generate images by learning to undo noise. The **forward process**
destroys an image in T = 1000 small steps. At step t it keeps most of the image and adds a little
Gaussian noise:

$$x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

The sequence $\beta_1, \dots, \beta_T$ is the **noise schedule**, the subject of this project.
Writing $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \alpha_1 \alpha_2 \cdots \alpha_t$, the
tutorial shows that the steps combine into one jump from the clean image:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$$

So $\bar\alpha_t$ is the share of the clean image's signal left at step t: 1 at the start and
close to 0 at the end, where $x_T$ is pure noise. **A schedule decides how fast
$\bar\alpha_t$ falls**, and so how many of the 1000 steps are spent at each noise level.

### 3.1 The linear schedule

The cell below is the tutorial's, unchanged. `extract` picks each image's value out of a
schedule array at that image's own step t, and `linear_beta_schedule` spaces the betas evenly from
0.0001 to 0.02.

| Call | What you give it | What you get back |
|---|---|---|
| `a.gather(-1, t)` | a schedule array and one step index per image | the array's value at each image's step |
| `torch.linspace(start, end, n)` | two ends and a count | n evenly spaced values from start to end |"""

MD_COSINE = r"""### 3.2 The cosine schedule

Nichol and Dhariwal (2021) observed that with the linear schedule the end of the forward process
is almost pure noise, so those steps contribute little. Their **cosine schedule** defines
$\bar\alpha_t$ directly, falling slowly at the start and the end and fastest in the middle:

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right), \qquad \beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}$$

The offset s = 0.008 keeps the first betas from being vanishingly small: they chose it so that
$\sqrt{\beta_0}$ is slightly smaller than one pixel step of the [-1, 1] range, 1/127.5. Each beta
is clipped at 0.999, because $\bar\alpha_T$ is 0 and the last beta would otherwise be exactly 1.

`cosine_beta_schedule` below is written from that formula. `make_schedule` then turns either
schedule's betas into the three arrays the tutorial keeps as globals, one dictionary per schedule,
and the printed lines show where each starts and ends.

| Call | What you give it | What you get back |
|---|---|---|
| `torch.cos(x)` | a tensor | the cosine of every value |
| `torch.clip(x, max=m)` | a tensor and a ceiling | the tensor with every value above m set to m |
| `torch.cumprod(a, dim=0)` | a 1-D tensor | the running product: a[0], a[0]a[1], a[0]a[1]a[2], ... |"""

MD_QSAMPLE = r"""### 3.3 Jumping to step t

The tutorial's `q_sample` computes the one-jump formula above. The only change is that the
schedule is passed in, so the same function serves both models.

| Call | What you give it | What you get back |
|---|---|---|
| `torch.randn_like(x)` | a tensor | standard normal noise of the same shape, on the same device |"""

MD_NOISING = r"""### 3.4 What each schedule does to a digit

The tutorial's picture of the forward process at steps 25, 50, 100, 250, 500 and 999, drawn for
both schedules from the same four training digits and the same noise. The linear row fades to
noise sooner than the cosine row; section 3.5 puts numbers on it."""

MD_ALLOCATION = r"""### 3.5 Where each schedule spends its steps

The left panel plots $\bar\alpha_t$; the right plots the same information as the **log
signal-to-noise ratio**, $\log(\bar\alpha_t / (1 - \bar\alpha_t))$, which spreads out the values
crowded near 0 and 1. The printed lines count the steps at which less than 1% of the signal is
left and the image is almost pure noise. A sampler walks through every one of those steps with
little left to do, which is the inefficiency the cosine schedule was designed to remove.

| Call | What you give it | What you get back |
|---|---|---|
| `tensor.double()` | a tensor | the same values in double precision |"""

MD_SHORT_T = r"""### 3.6 Which "time steps"?

"Fewer time steps" could also mean a shorter forward process, a smaller T. The cell below shows
what that does to the tutorial's linear schedule: its betas run from 0.0001 to 0.02 whatever T
is, so a short process ends far from pure noise, while sampling always starts from pure noise. The
cosine schedule is defined on t / T and ends at noise for any T. This project therefore trains both
models with the tutorial's T = 1000 and reads "fewer time steps" as fewer **sampling** steps
(section 6), the reading in which both schedules can be used as defined."""

MD_BACKWARD = r"""## 4 The backward process: a conditional U-Net

To generate an image the model runs the process backwards: from pure noise, remove a little noise
at a time. The network that makes this possible is trained to answer one question: given a noisy
image $x_t$, its step t and its class, **what noise was added?** Its answer has the image's shape,
one noise value per pixel.

__FIG_ARCHITECTURE__

The tutorial's network is a small **U-Net**. The encoder halves the image twice with average
pooling (20 x 20 to 10 x 10 to 5 x 5) and the decoder doubles it back. **Skip connections** hand
each decoder level the encoder's map of the same size, so fine detail does not have to squeeze
through the 5 x 5 bottleneck. The step t (divided by T) and the class label each become a vector
of 32 numbers; the two are added and injected at every level. The two cells below are the
tutorial's `ConvBlock` and `UNet_cond`, unchanged.

| Call | What you give it | What you get back |
|---|---|---|
| `nn.Conv2d(c_in, c_out, 3, padding=1)` | maps (batch, c_in, n, n) | (batch, c_out, n, n): each value a weighted sum over a 3 x 3 patch of every input map |
| `nn.AvgPool2d(2)` | (batch, c, n, n) | (batch, c, n/2, n/2), each value the mean of a 2 x 2 square |
| `nn.Upsample(scale_factor=2, mode='nearest')` | (batch, c, n, n) | (batch, c, 2n, 2n), each value copied into a 2 x 2 square |
| `nn.Embedding(n_classes, ch)` | integer class labels | one learned vector of ch numbers per label |
| `torch.cat([a, b], dim=1)` | two maps of the same size | one map holding the channels of both |"""

MD_SHAPE = r"""### 4.1 A shape check at 20 x 20

One forward pass on eight training images at random steps. The output must have the input's
shape, 1 x 20 x 20 per image: one predicted noise value per pixel."""

MD_TRAINING = r"""## 5 Training

__FIG_TRAINING__

The tutorial's training step: take a batch of clean images, draw a random step t for each, noise
each image to its step with `q_sample`, ask the network for the noise, and score the answer by the
**mean squared error** between the predicted and the true noise. The schedule enters in one place
only, `q_sample`, so one function trains both models.

### 5.1 Settings

The tutorial solution's settings for its conditional model: Adam at a learning rate of 0.0002,
batches of 128 and 150 epochs, with the tutorial's U-Net of 32 channels. `STEP_COUNTS` lists the
sampling budgets section 6 and `main_report.ipynb` compare."""

MD_TRAIN_FN = r"""### 5.2 The training loop

The tutorial's loop, inside a function so it can run once per schedule and seed. Each run first
resets the random seed. A linear run and the cosine run of the same seed therefore start from the
same initial weights and see the same batches, the same steps t and the same noise: **the schedule
is the only difference between them.** Seed 0 also keeps copies of its weights at a few earlier
epochs, which `main_report.ipynb` uses to measure how quickly each schedule's samples improve.

| Call | What you give it | What you get back |
|---|---|---|
| `DataLoader(dataset, batch_size, shuffle, generator)` | a dataset | an iterable of batches; with a seeded generator the shuffled order is the same every run |
| `torch.randint(0, timesteps, (n,), device=device)` | a range and a count | n random whole numbers from 0 to timesteps - 1 |
| `torch.optim.Adam(parameters, lr)` | the network's weights and a learning rate | an optimiser; `.step()` moves every weight a little against its gradient |
| `nn.MSELoss()` | nothing | a loss; called on two tensors it returns their mean squared difference |
| `loss.backward()` | nothing | computes the gradient of `loss` for every weight |"""

MD_RUN = r"""### 5.3 Six training runs

Three seeds, and for each seed one model per schedule. One run of a neural network is one draw
from many possible runs, so a difference between the schedules is only convincing if it survives
a change of seed. A line is printed every __PRINT_EVERY__ epochs: the mean loss of that epoch and
how long the epoch took."""

MD_TRIVIAL = r"""### 5.4 A sanity check

A network that learned nothing could predict zero noise everywhere. Its loss would be the variance
of the noise, 1. Every run must end far below that. The seconds per epoch show what the schedule
does to the cost of training: it changes which noise level each step uses, not how much
computation a step takes."""

MD_CURVES = r"""### 5.5 The loss curves

The loss of every run per epoch, seed 0 drawn solid. The printed change compares the mean loss of
the last 10 epochs with the 10 before: close to 0 means training has levelled off.

**The two schedules' losses cannot be
compared with each other.** Each is the average error over steps t drawn evenly from 1000, and the
two schedules put those steps at different noise levels, so each averages a different mix of easy
and hard questions. `main_report.ipynb` measures the error at each noise level instead
(section 9 there)."""

MD_SAVE = r"""### 5.6 Saving

One file per schedule. Each holds a plain dictionary of weights, one entry per seed and, for seed
0, one per snapshot epoch, saved from the CPU so the file loads on a machine without a GPU. Every
loss, duration and setting goes into `DDPM_history.json`."""

MD_SAMPLING = r"""## 6 Sampling

Generation starts from pure noise $x_T$ and applies a reverse step again and again. The tutorial's
sampler visits all 1000 steps. The hypothesis is about doing it in **fewer**, so this section
builds a sampler that visits only K of them, in two versions.

__FIG_SAMPLING__

### 6.1 Which steps a K-step sampler visits

K evenly spaced steps from the last to the first, rounded to whole steps: the choice Nichol and
Dhariwal make in their section 4. With K = 1000 it is every step, the tutorial's loop.

| Call | What you give it | What you get back |
|---|---|---|
| `np.linspace(0, 999, K)` | two ends and a count | K evenly spaced values |
| `np.round(v).astype(int)` | real values | the nearest whole numbers |"""

MD_P_SAMPLE = r"""### 6.2 One reverse step (DDPM)

The tutorial's step, Algorithm 2 of Ho et al. (2020), is

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \hat\epsilon\right) + \sqrt{\beta_t}\, z, \qquad z \sim \mathcal{N}(0, I)$$

where $\hat\epsilon$ is the network's noise estimate. Two changes let it skip steps safely.

**Skipping.** A step from t back to an earlier step t' uses the beta that takes
$\bar\alpha_{t'}$ to $\bar\alpha_t$ in one move, $\beta = 1 - \bar\alpha_t / \bar\alpha_{t'}$,
with $\alpha = 1 - \beta$. When t' = t - 1 this is the tutorial's $\beta_t$.

**The x0 form.** Solving the one-jump formula of section 3 for the clean image gives the
network's current guess of it, $\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$.
The same step can then be written as a weighted average of that guess and the current image:

$$x_{t'} = \frac{\sqrt{\bar\alpha_{t'}}\,\beta}{1 - \bar\alpha_t}\,\hat x_0 + \frac{\sqrt{\alpha}\,(1 - \bar\alpha_{t'})}{1 - \bar\alpha_t}\, x_t + \sqrt{\beta}\, z$$

The two forms are algebraically the same step (section 6.4 checks it at all 1000 steps). The x0
form has one advantage: $\hat x_0$ is an image, so it can be **clipped** to the data range
[-1, 1] before it is used. Section 6.5 shows why that matters for the cosine schedule."""

MD_DDIM = r"""### 6.3 DDIM

Song et al. (2021) showed that a network trained as in section 5 can also be sampled
**deterministically**. DDIM moves $\hat x_0$ to the earlier step's noise level and reuses the
current noise estimate as that level's noise, drawing no fresh noise:

$$x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat x_0 + \sqrt{1 - \bar\alpha_{t'}}\,\hat\epsilon$$

It was designed for sampling in few steps, which makes it the second sampler of the comparison.
It uses the same clipped $\hat x_0$, with $\hat\epsilon$ recomputed from it so the two agree."""

MD_LOOP = r"""### 6.4 The sampling loop, and a check against the tutorial

The tutorial's loop over the step list of section 6.1. All the randomness, the starting noise and
every z, comes from one seeded generator on the CPU and is then moved to the device. A seed
therefore gives the same samples on any machine, and the two schedules start from the same noise,
so their samples can be compared image by image. `generate` draws large sets in chunks of 1,000.

The check after it runs the x0 form, unclipped, against the tutorial's own update at every one of
the 1000 steps, in double precision, with a stand-in network that always returns the same noise
estimate, so only the formulas are compared.

| Call | What you give it | What you get back |
|---|---|---|
| `torch.Generator().manual_seed(s)` | a seed | a private random-number source on the CPU |
| `torch.randn(shape, generator=g)` | a shape and a generator | standard normal values drawn from that source |
| `x.clamp(-1, 1)` | a tensor | the tensor with every value limited to [-1, 1] |"""

MD_GAIN = r"""### 6.5 Why the clip

In the tutorial's form the noise estimate, and any error in it, is multiplied by
$\beta / (\sqrt{\alpha}\sqrt{1 - \bar\alpha_t})$. The table prints that multiplier for the first
reverse step, where the network sees pure noise, for each number of steps K. The cosine schedule's
last beta is 0.999, so its $\alpha$ there is 0.001 and the multiplier is large even when no step is
skipped; skipping steps makes it larger still. In the x0 form the same error lands in
$\hat x_0$, the clip bounds it by the data range, and the step keeps only a fraction of
$\hat x_0$."""

MD_GRIDS = r"""## 7 Generated digits

The tutorial's last step, in the brief's layout: 12 samples per digit with the digits as rows,
120 images per model, from each schedule's seed-0 model with the full 1000-step sampler and the
same starting noise. This is a first look; `main_report.ipynb` makes the comparison."""

MD_REAL = r"""### 7.1 Real digits

Twelve training images of each digit, for comparison."""


def ddpm_notebook():
    intro = (
        "This notebook is **Task 1**. It trains the conditional denoising diffusion "
        "probabilistic model (DDPM) of Tutorial 9.3 on `mnist_custom.pt` twice: once with the "
        "tutorial's `linear_beta_schedule`, and once with a `cosine_beta_schedule` implemented "
        "here. Everything else is shared, so the noise schedule is the only difference between "
        "the two models. `main_report.ipynb` compares them.\n\n"
        "The hypothesis under test:\n\n" + HYPOTHESIS + "\n\n"
        "The notebook follows the tutorial's order and keeps its code wherever it can. "
        "`extract`, `linear_beta_schedule`, `ConvBlock` and `UNet_cond` are the tutorial's, "
        "unchanged, and so are the optimiser, the learning rate, the batch size, the loss and "
        "the 1000 diffusion steps. What differs:\n\n"
        "| | Tutorial 9.3 | This notebook | Why |\n|---|---|---|---|\n"
        "| Data | torchvision's MNIST, 28 x 28 | `mnist_custom.pt`, 20 x 20, dark digits on "
        "white | the brief's dataset; the U-Net needs no change for it (section 4) |\n"
        "| Noise schedule | linear | linear and cosine, one model each | the hypothesis compares "
        "them (section 3) |\n"
        "| Schedule arrays | globals `betas`, `alphas`, `alpha_bars` | one dictionary per "
        "schedule, passed to `q_sample` and the sampler | two schedules exist side by side |\n"
        "| Training runs | one | three per schedule, seeds 0, 1 and 2 | a difference must "
        "survive retraining (section 5.3) |\n"
        "| Epochs | 100 in the exercise, 150 in the solution | __EPOCHS__ | the solution's "
        "conditional model |\n"
        "| Progress display | a tqdm bar per epoch | a printed line every __PRINT_EVERY__ epochs "
        "| tqdm writes to the error stream |\n"
        "| Sampler | all 1000 steps | the same update able to skip steps, in its x0 form with "
        "the clean-image estimate clipped, and DDIM | the hypothesis is about sampling in fewer "
        "steps (section 6) |\n\n"
        "It writes `ddpm_linear.pth` and `ddpm_cosine.pth` (the weights), `DDPM_history.json` "
        "(every training curve and setting) and `mnist_custom_test.pt` (the test split).\n\n"
        "**Sections:** 1 Setup, 2 The data, 3 The forward diffusion process, 4 The backward "
        "process, 5 Training, 6 Sampling, 7 Generated digits.")
    return [
        markdown("md-title", fill(HEADER, __TITLE__="DDPM_CosineSchedule", __INTRO__=fill(intro))),
        markdown("md-setup", MD_SETUP),
        code("code-imports", imports("training")),
        code("code-style", CELL_STYLE),
        markdown("md-data", MD_DATA),
        code("code-data", CELL_DATA),
        markdown("md-audit", MD_AUDIT),
        code("code-audit", CELL_AUDIT),
        markdown("md-scale", MD_SCALE),
        code("code-scale", fill(CELL_SCALE)),
        markdown("md-forward", MD_FORWARD),
        code("code-tutorial-forward", CELL_TUTORIAL_FORWARD),
        markdown("md-cosine", MD_COSINE),
        code("code-cosine", CELL_COSINE),
        code("code-schedules", CELL_SCHEDULES),
        markdown("md-qsample", MD_QSAMPLE),
        code("code-qsample", CELL_QSAMPLE),
        markdown("md-noising", MD_NOISING),
        code("code-noising", CELL_NOISING),
        markdown("md-allocation", MD_ALLOCATION),
        code("code-allocation", CELL_ALLOCATION),
        markdown("md-short-t", MD_SHORT_T),
        code("code-short-t", CELL_SHORT_T),
        markdown("md-backward", MD_BACKWARD.replace(
            "__FIG_ARCHITECTURE__", figure("architecture.png", "The conditional U-Net"))),
        code("code-convblock", CELL_CONVBLOCK),
        code("code-unet", CELL_UNET),
        markdown("md-shape", MD_SHAPE),
        code("code-shape", CELL_SHAPE),
        markdown("md-training", MD_TRAINING.replace(
            "__FIG_TRAINING__", figure("training_step.png", "One training step"))),
        code("code-settings", fill(CELL_SETTINGS)),
        code("code-cpu-state", CELL_CPU_STATE),
        markdown("md-train-fn", MD_TRAIN_FN),
        code("code-train-fn", CELL_TRAIN_FN),
        markdown("md-run", fill(MD_RUN)),
        code("code-run", CELL_RUN),
        markdown("md-trivial", MD_TRIVIAL),
        code("code-trivial", fill(CELL_TRIVIAL)),
        markdown("md-curves", MD_CURVES),
        code("code-curves", CELL_CURVES),
        markdown("md-save", MD_SAVE),
        code("code-save", CELL_SAVE),
        markdown("md-sampling", MD_SAMPLING.replace(
            "__FIG_SAMPLING__", figure("sampling_steps.png", "DDPM and DDIM steps"))),
        code("code-steps", CELL_STEPS),
        markdown("md-p-sample", MD_P_SAMPLE),
        code("code-p-sample", CELL_P_SAMPLE),
        markdown("md-ddim", MD_DDIM),
        code("code-ddim", CELL_DDIM),
        markdown("md-loop", MD_LOOP),
        code("code-loop", CELL_LOOP),
        code("code-equivalence", CELL_EQUIVALENCE),
        markdown("md-gain", MD_GAIN),
        code("code-gain", CELL_GAIN),
        markdown("md-grids", MD_GRIDS),
        code("code-grids", CELL_GRIDS),
        markdown("md-real", MD_REAL),
        code("code-real", CELL_REAL),
    ]


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
        "comparing the average and spread of those numbers for real and generated digits gives "
        "a Frechet distance, the idea behind FID, in a space built for digits.\n\n"
        "It trains for a fixed __CLASSIFIER_EPOCHS__ epochs on all 60,000 training images and "
        "nothing is tuned, so no development split is needed. It never makes a prediction on "
        "the test split. It writes `DigitClassifier.pth`.\n\n"
        "**Sections:** 1 Setup, 2 The data, 3 The model, 4 Training.")
    return [
        markdown("md-title", fill(HEADER, __TITLE__="DigitClassifier", __INTRO__=fill(intro))),
        markdown("md-setup", "## 1 Setup\n\nThe libraries, the device and the seed, as in "
                             "`DDPM_CosineSchedule.ipynb`."),
        code("code-imports", imports("training")),
        markdown("md-data", MD_DATA),
        code("code-data", CELL_DATA),
        markdown("md-audit", MD_AUDIT.replace("(section 4)", "(`DDPM_CosineSchedule.ipynb`, "
                                                             "section 4)")),
        code("code-audit", CELL_AUDIT),
        markdown("md-model", "## 3 The model\n\nTwo convolution blocks, each followed by "
                             "**max pooling**, which keeps the largest value in each 2 x 2 "
                             "square and so halves the map (20 to 10 to 5), then a 128-unit "
                             "layer and the 10 class scores. `features()` returns the 128-unit "
                             "layer. Each image is inverted first, so the digit is bright on a "
                             "black background: the zero padding a convolution adds at the "
                             "border then looks like background instead of a dark frame.\n\n"
                             "| Call | What you give it | What you get back |\n"
                             "|---|---|---|\n| `nn.MaxPool2d(2)` | maps (batch, c, n, n) | "
                             "(batch, c, n/2, n/2), the maximum of each 2 x 2 square |\n"
                             "| `F.cross_entropy(logits, labels)` | class scores and true "
                             "classes | the mean negative log-probability of the true class |"),
        code("code-classifier-model", CELL_CLASSIFIER),
        markdown("md-train", "## 4 Training\n\nAdam with a learning rate of 0.001, batches of "
                             "256, a fixed number of epochs. The weights are saved from the CPU."),
        code("code-cpu-state", CELL_CPU_STATE),
        code("code-clf-train", fill(CELL_CLF_TRAIN)),
        markdown("md-clf-loop", "The loop: score a batch, measure the cross-entropy, step. It "
                                "ends with a check that the loss fell far below the "
                                "ln 10 = 2.303 of guessing."),
        code("code-clf-loop", fill(CELL_CLF_LOOP)),
        markdown("md-clf-save", "Saving the weights."),
        code("code-clf-save", CELL_CLF_SAVE),
    ]


# main_report's cells live in their own module, so the training notebooks can be frozen and sent
# to the GPU while it is still being written.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import report_cells
except ImportError:
    report_cells = None


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


if __name__ == "__main__":
    CONSTRUCTOR = re.compile(r"torch\.(zeros|ones|tensor|eye|arange|randn|full|empty)\(")
    NESTED_FSTRING = re.compile(
        r"""\bf"[^"\n]*\{[^}\n]*"[^}\n]*\}|\bf'[^'\n]*\{[^}\n]*'[^}\n]*\}""")
    BACKSLASH_IN_BRACES = re.compile(r"""\bf["'][^"'\n]*\{[^}\n]*\\[^}\n]*\}""")
    EXPECTED_FIGURES = {"DDPM_CosineSchedule.ipynb": 3, "DigitClassifier.ipynb": 0,
                        "main_report.ipynb": 0}
    # Cells supplied by the tutorial are exempt from the line-width rule: they are byte-identical.
    SUPPLIED = {"code-tutorial-forward", "code-convblock", "code-unet"}

    plan = [("DDPM_CosineSchedule.ipynb", ddpm_notebook()),
            ("DigitClassifier.ipynb", classifier_notebook())]
    if report_cells is not None:
        plan.append(("main_report.ipynb", report_cells.report_notebook(sys.modules[__name__])))
    built = {}
    for name, cells in plan:
        path, notebook = write(name, cells)
        built[name] = notebook
        print(f"wrote {path}  ({len(cells)} cells)")

    print("\ngenerator rails")
    for definition in CARRIED:
        check(f"tutorial definition {definition} found", definition in TUTORIAL_DEFS)
    shared_sources = {}
    for name, notebook in built.items():
        raw = json.dumps(notebook, ensure_ascii=False)
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        source = "\n".join("".join(c["source"]) for c in code_cells)
        own = "\n".join("".join(c["source"]) for c in code_cells if c["id"] not in SUPPLIED)
        lines = [line for line in source.split("\n") if line.strip()]
        comments = [line for line in lines if line.strip().startswith("#")]
        ratio = len(comments) / len(lines)
        ids = [c["id"] for c in notebook["cells"]]
        longest = max(len(line) for line in own.split("\n"))

        check(f"{name}: no em or en dash", "\u2014" not in raw and "\u2013" not in raw)
        check(f"{name}: no control characters in any cell",
              not any(ord(ch) < 32 and ch not in "\n\t"
                      for c in notebook["cells"] for ch in "".join(c["source"])))
        check(f"{name}: every cell has a unique id", len(ids) == len(set(ids)),
              f"{len(ids)} cells")
        check(f"{name}: every code cell carries a comment",
              all(any(line.strip().startswith("#") for line in c["source"]) for c in code_cells))
        check(f"{name}: source lines keep their newlines",
              all(all(line.endswith("\n") for line in c["source"][:-1])
                  for c in notebook["cells"] if len(c["source"]) > 1))
        check(f"{name}: comment ratio at least 0.12", ratio >= 0.12,
              f"{len(comments)}/{len(lines)} = {ratio:.0%}")
        check(f"{name}: no code line of ours longer than 100 characters", longest <= 100,
              f"longest {longest}")
        check(f"{name}: no tqdm", "tqdm" not in source)
        check(f"{name}: no placeholder left",
              "__" not in re.sub(r"__init__|__version__|__name__", "", raw))
        check(f"{name}: no f-string needing Python 3.12 (PEP 701)",
              not NESTED_FSTRING.search(source) and not BACKSLASH_IN_BRACES.search(source))
        bare = [line.strip() for line in source.split("\n")
                if CONSTRUCTOR.search(line) and "device" not in line]
        check(f"{name}: every tensor constructor names the device", not bare,
              "; ".join(bare)[:200])
        run = 0
        worst = 0
        for cell in notebook["cells"]:
            run = run + 1 if cell["cell_type"] == "code" else 0
            worst = max(worst, run)
        check(f"{name}: at most 2 code cells in a row without markdown", worst <= 2, f"{worst}")
        check(f"{name}: every cell parses as Python",
              all(compile("".join(c["source"]), f"{name}:{c['id']}", "exec") or True
                  for c in code_cells))
        if not DRAFT:
            embedded = sum("data:image/png;base64," in "".join(c["source"])
                           for c in notebook["cells"] if c["cell_type"] == "markdown")
            check(f"{name}: embeds {EXPECTED_FIGURES[name]} explanatory figures",
                  embedded == EXPECTED_FIGURES[name], f"{embedded}")
        for cell in code_cells:
            shared_sources.setdefault(cell["id"], set()).add("".join(cell["source"]))
        if name == "main_report.ipynb":
            report_cells.rails(check, source)
        else:
            test_lines = [line.strip() for line in source.split("\n") if "test_images" in line]
            check(f"{name}: test_images appears only in the line that saves the split",
                  test_lines == ['test_split = {key: data[key] for key in ("test_images", '
                                 '"test_labels")}'], "; ".join(test_lines))
    for definition in CARRIED:
        check(f"{definition} is byte-identical to Tutorial 9.3 in DDPM_CosineSchedule.ipynb",
              TUTORIAL_DEFS[definition] in "\n".join(
                  "".join(c["source"]) for c in built["DDPM_CosineSchedule.ipynb"]["cells"]))
    for cell_id, versions in shared_sources.items():
        if len(versions) > 1:
            check(f"shared cell {cell_id} is byte-identical in every notebook", False,
                  f"{len(versions)} versions")

    print("\n" + ("GENERATOR CHECKS PASSED" if not FAILURES
                 else f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES)))
    raise SystemExit(1 if FAILURES else 0)
