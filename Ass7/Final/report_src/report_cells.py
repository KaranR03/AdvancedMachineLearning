r"""The cells of main_report.ipynb, imported by tools/build_notebooks.py.

Kept in their own module so the two training notebooks could be frozen and sent to the GPU while
this notebook was still being written. The model, schedule and sampler cells are not repeated here:
report_notebook() takes them from build_notebooks, so they are byte-identical to the cells that
trained the checkpoints (the generator's shared-cell rail checks it).

main_report.ipynb trains nothing. It loads the two checkpoints, the classifier, the training record
and the test split, samples both models, and prints every number the report quotes as a
pipe-delimited line that tools/build_report.py parses.
"""
import re

# --------------------------------------------------------------------------- code cells

PRECISION = r'''

# Full float32 in every convolution. PyTorch lets recent NVIDIA GPUs run convolutions in the
# shorter TF32 format by default; switching it off makes the numbers below depend less on which
# GPU, or CPU, runs this notebook. Sampling repeats the network up to 1000 times per image, so
# small rounding differences would otherwise have many steps in which to grow.
torch.backends.cudnn.allow_tf32 = False'''

CELL_MR_DATA = r'''# The test split, written by DDPM_CosineSchedule.ipynb (section 2). This notebook is the only
# place it is scored. The image size and the number of classes are read from it.
test_data = torch.load("mnist_custom_test.pt", weights_only=True)
test_images, test_labels = test_data["test_images"], test_data["test_labels"]
image_size, n_class = test_images.shape[-1], int(test_labels.max()) + 1
labels_np = test_labels.numpy()
print(f"test_images : {tuple(test_images.shape)}, {test_images.dtype}")
print(f"test class counts : {np.bincount(labels_np, minlength=n_class).tolist()}")
print(f"pixels exactly 1.0 : {(test_images == 1).float().mean():.3f}")'''

CELL_MR_RECORDS = r'''# How the models were trained, from the record DDPM_CosineSchedule.ipynb saved (section 5.6).
# Every setting below is read from the file, never retyped.
with open("DDPM_history.json") as handle:
    history = json.load(handle)
assert history["timesteps"] == timesteps
n_channels_unet, SEEDS = history["n_channels_unet"], history["seeds"]
print(f"trained on {history['device']} ({history['gpu']}), torch {history['torch']}")
print(f"epochs {history['epochs']} | batch size {history['batch_size']} | "
      f"learning rate {history['lr']} | seeds {SEEDS} | {history['n_train']:,} training images")
seconds_per_epoch = {}
for name in SCHEDULE_FUNCTIONS:
    runs = [history["runs"][f"{name}_seed_{seed}"] for seed in SEEDS]
    seconds_per_epoch[name] = float(np.mean([run["seconds"] for run in runs]))
    print(f"record | {name} | seconds per epoch {seconds_per_epoch[name]:.2f} | minutes per run "
          + " ".join(f"{sum(run['seconds']) / 60:.1f}" for run in runs)
          + " | final training loss " + " ".join(f"{run['loss'][-1]:.4f}" for run in runs))'''

CELL_MR_LOAD = r'''# The two checkpoints the brief asks for, one per schedule. Each holds plain CPU state_dicts:
# the final model of every seed ("seed_0", ...) and seed 0's earlier snapshots ("epoch_10", ...).
def load_unet(state):
    network = UNet_cond(ch=n_channels_unet, n_classes=n_class).to(device)
    network.load_state_dict(state)
    return network.eval()


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


checkpoints = {name: torch.load(f"ddpm_{name}.pth", map_location="cpu", weights_only=True)
               for name in SCHEDULE_FUNCTIONS}
models = {(name, seed): load_unet(checkpoints[name][f"seed_{seed}"])
          for name in SCHEDULE_FUNCTIONS for seed in SEEDS}
# The measuring instrument, trained in DigitClassifier.ipynb.
classifier = DigitClassifier(n_class).to(device)
_ = classifier.load_state_dict(torch.load("DigitClassifier.pth", map_location="cpu",
                                          weights_only=True))
_ = classifier.eval()
for name, checkpoint in checkpoints.items():
    print(f"ddpm_{name}.pth | {', '.join(checkpoint)}")
print(f"parameters | U-Net {parameter_count(models['linear', SEEDS[0]]):,} | "
      f"classifier {parameter_count(classifier):,}")'''

CELL_MR_CLASSIFY = r'''# The classifier's verdict on images drawn as given classes: whether its top choice is that
# class. Returned per image, so a rate can be split by class or resampled.
def classify(images, labels, batch_size=2000):
    correct = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            logits = classifier(images[start:start + batch_size].to(device))
            target = labels[start:start + batch_size].to(device)
            correct.append((logits.argmax(dim=1) == target).cpu())
    return torch.cat(correct).double().numpy()'''

CELL_MR_FIG1 = r'''# Figure 1: each schedule's seed-0 model with the tutorial's full 1000-step sampler. Both grids
# start from the same noise (seed 1337, as in DDPM_CosineSchedule.ipynb, section 7), so the two
# images in the same place differ only by the model. Rows are the digits 0 to 9, twelve each.
sample_per_class = 12
grid_labels = torch.arange(n_class, device=device).repeat_interleave(sample_per_class)
grids = {name: (p_sample_loop_cDDPM(models[name, SEEDS[0]], grid_labels, schedules[name]) + 1) / 2
         for name in SCHEDULE_FUNCTIONS}
titles = {"linear": "Linear schedule", "cosine": "Cosine schedule"}
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), gridspec_kw={"wspace": 0.03})
for ax, (name, images) in zip(axes, grids.items()):
    ax.imshow(utils.make_grid(images.cpu(), nrow=sample_per_class, pad_value=1)[0], cmap="gray",
              vmin=0, vmax=1)
    ax.set_title(f"{titles[name]}, {timesteps} steps", fontsize=9)
    # The rows are the same digits in both grids, so only the left one is labelled.
    if ax is axes[0]:
        ax.set_yticks(12 + 22 * np.arange(n_class), [str(c) for c in range(n_class)], fontsize=8)
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(length=0)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
fig.savefig("figure_1_grids.pdf", bbox_inches="tight")
plt.show()'''

CELL_MR_GRID_ACCURACY = r'''# The share of each grid's 120 images the classifier reads as the digit they were drawn as.
grid_accuracy = {name: float(classify(images, grid_labels).mean())
                 for name, images in grids.items()}
for name, value in grid_accuracy.items():
    print(f"grid accuracy | {name} | {value:.4f} of {len(grids[name])} images")'''

CELL_MR_SETS = r'''# Two disjoint class-stratified sets of N_EVAL test images with identical class counts. Samples
# are drawn under set A's labels and compared with set B's real images, so every comparison has
# the same size and class mix; set A's own real images against set B give the floor.
N_EVAL = __N_EVAL__
counts = np.bincount(labels_np, minlength=n_class)
quota = N_EVAL * counts / counts.sum()
per_class = np.floor(quota).astype(int)
# the classes with the largest remainders round up, so the counts add up to exactly N_EVAL
per_class[np.argsort(per_class - quota)[:N_EVAL - per_class.sum()]] += 1
rng = np.random.default_rng(SEED)
set_a, set_b = [], []
for c in range(n_class):
    members = rng.permutation(np.flatnonzero(labels_np == c))
    set_a.extend(members[:per_class[c]])
    set_b.extend(members[per_class[c]:2 * per_class[c]])
set_a, set_b = np.sort(set_a), np.sort(set_b)
eval_labels = test_labels[set_a]
labels_a, labels_b = labels_np[set_a], labels_np[set_b]
print(f"sets | A {len(set_a):,} | B {len(set_b):,} | per class {per_class.tolist()}")'''

CELL_MR_METRICS = r'''# The classifier's 128-unit feature layer, in float64, and the Frechet distance to one fixed
# reference set: the distance between two Gaussians fitted to the feature clouds, 0 when the
# means and covariances match. tr sqrt(C_ref C_x) comes from symmetric eigendecompositions, which
# stay real and exact; the reference's own square root is computed once and reused.
def features(images, batch_size=2000):
    with torch.no_grad():
        parts = [classifier.features(images[start:start + batch_size].to(device)).cpu()
                 for start in range(0, len(images), batch_size)]
    return torch.cat(parts).double().numpy()


def frechet_to(reference):
    mean_r, cov_r = reference.mean(axis=0), np.cov(reference, rowvar=False)
    values, vectors = np.linalg.eigh(cov_r)
    root_r = (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T

    def distance(x):
        mean_x, cov_x = x.mean(axis=0), np.cov(x, rowvar=False)
        cross = np.sqrt(np.clip(np.linalg.eigvalsh(root_r @ cov_x @ root_r), 0, None)).sum()
        return float(((mean_x - mean_r) ** 2).sum() + np.trace(cov_r) + np.trace(cov_x)
                     - 2 * cross)
    return distance


# Set B is the reference for every distance and every spread.
real_b = features(test_images[set_b])
fd_to_b = frechet_to(real_b)
real_variety = [np.trace(np.cov(real_b[labels_b == c], rowvar=False)) for c in range(n_class)]


# Diversity: within each class, the total variance of a set's features over that of set B's
# real images, averaged over the classes. 1 matches real variety; well below 1 means the samples
# of a class look alike, the signature of mode collapse.
def spread_ratio(values):
    return float(np.mean([np.trace(np.cov(values[labels_a == c], rowvar=False)) / real_variety[c]
                          for c in range(n_class)]))'''

CELL_MR_FLOOR = r'''# The reference values: the classifier on every real test image, and set A's real images
# scored exactly as a sample set will be. A model that drew real digits would score these.
real_accuracy = float(classify(test_images, test_labels).mean())
real_a = features(test_images[set_a])
floor = fd_to_b(real_a)
real_a_accuracy = float(classify(test_images[set_a], eval_labels).mean())
print(f"classifier | all {len(test_labels):,} real test images | accuracy {real_accuracy:.4f}")
print(f"floor | real set A against set B | FD {floor:.2f} | accuracy {real_a_accuracy:.4f} | "
      f"spread {spread_ratio(real_a):.3f}")'''

CELL_MR_BOOTSTRAP = r'''# A paired bootstrap for a difference in FD, cosine minus linear. Sample i of a linear set and
# sample i of the matching cosine set share their label and their starting noise, so both sets
# are resampled with the same indices, RESAMPLES times, and the middle 95% of the recomputed
# differences is the interval. A fresh generator per call keeps it independent of cell order.
RESAMPLES = __RESAMPLES__


def fd_gap_interval(linear_features, cosine_features, seed=SEED):
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(RESAMPLES):
        pick = rng.integers(0, len(linear_features), len(linear_features))
        gaps.append(fd_to_b(cosine_features[pick]) - fd_to_b(linear_features[pick]))
    low, high = np.percentile(gaps, [2.5, 97.5])
    return float(low), float(high)'''

CELL_MR_PLAN = r'''# The sampling budgets K and the two samplers. Seed 0's models are sampled at every K, and the
# tutorial's own 1000-step sampler is the full-quality reference; the models of the other seeds
# are sampled at every reduced K, the budgets the decision rule is about.
STEP_COUNTS = (10, 20, 50, 100, 250, 1000)
REDUCED = [n_steps for n_steps in STEP_COUNTS if n_steps < timesteps]
SAMPLERS = ("ddpm", "ddim")
plan = [(SEEDS[0], "ddpm", timesteps)] + [(seed, sampler, n_steps) for seed in SEEDS
                                          for sampler in SAMPLERS for n_steps in REDUCED]
print(f"{len(plan)} sample sets per schedule, {N_EVAL:,} images each")'''

CELL_MR_SWEEP = r'''# Every sample set, scored as it is drawn. The features are kept for the bootstrap, and seed 0's
# images for Figure 3. Both schedules use the same labels and the same starting noise.
scores, kept_features, kept_images = {}, {}, {}
for seed, sampler, n_steps in plan:
    cells = []
    for name in SCHEDULE_FUNCTIONS:
        images = generate(models[name, seed], eval_labels, schedules[name], n_steps, sampler)
        key = (name, seed, sampler, n_steps)
        kept_features[key] = features(images)
        scores[key] = {"fd": fd_to_b(kept_features[key]),
                       "accuracy": float(classify(images, eval_labels).mean()),
                       "spread": spread_ratio(kept_features[key])}
        if seed == SEEDS[0]:
            kept_images[key] = images
        cells.append(f"{name} FD {scores[key]['fd']:6.2f} accuracy "
                     f"{scores[key]['accuracy']:.4f} spread {scores[key]['spread']:.3f}")
    print(f"quality | seed {seed} | {sampler} | K={n_steps:4d} | " + " | ".join(cells))'''

CELL_MR_RULE = r'''# The decision rule, fixed before any sample was scored. At a reduced K, for one sampler, a
# schedule is "better" when its FD is lower at seed 0 with the paired interval excluding 0, and
# lower again at every other seed. Anything else is "unclear".
def compare(sampler, n_steps):
    gaps = [scores["cosine", seed, sampler, n_steps]["fd"]
            - scores["linear", seed, sampler, n_steps]["fd"] for seed in SEEDS]
    low, high = fd_gap_interval(kept_features["linear", SEEDS[0], sampler, n_steps],
                                kept_features["cosine", SEEDS[0], sampler, n_steps])
    if high < 0 and all(gap < 0 for gap in gaps):
        winner = "cosine"
    elif low > 0 and all(gap > 0 for gap in gaps):
        winner = "linear"
    else:
        winner = "unclear"
    return {"gaps": gaps, "low": low, "high": high, "winner": winner}


def listing(step_counts):
    return ", ".join(str(n_steps) for n_steps in step_counts) or "none"


decisions = {(sampler, n_steps): compare(sampler, n_steps)
             for sampler in SAMPLERS for n_steps in REDUCED}
for (sampler, n_steps), d in decisions.items():
    print(f"rule | {sampler} | K={n_steps:4d} | cosine minus linear FD {d['gaps'][0]:+.2f} "
          f"[{d['low']:+.2f}, {d['high']:+.2f}] | other seeds "
          + " ".join(f"{gap:+.2f}" for gap in d["gaps"][1:]) + f" | {d['winner']}")
for sampler in SAMPLERS:
    groups = {winner: [k for k in REDUCED if decisions[sampler, k]["winner"] == winner]
              for winner in ("cosine", "linear", "unclear")}
    print(f"verdict | {sampler} | cosine better at K = {listing(groups['cosine'])} | "
          f"linear better at K = {listing(groups['linear'])} | "
          f"unclear at K = {listing(groups['unclear'])}")
# At the full 1000 steps, seed 0 only: the hypothesis makes no claim here, but it is the
# baseline every reduced K is measured against.
full_fd = {name: scores[name, SEEDS[0], "ddpm", timesteps]["fd"] for name in SCHEDULE_FUNCTIONS}
full_low, full_high = fd_gap_interval(kept_features["linear", SEEDS[0], "ddpm", timesteps],
                                      kept_features["cosine", SEEDS[0], "ddpm", timesteps])
print(f"full steps | cosine minus linear FD {full_fd['cosine'] - full_fd['linear']:+.2f} "
      f"[{full_low:+.2f}, {full_high:+.2f}]")'''

CELL_MR_FEWEST = r'''# Two step counts that summarise each curve, at seed 0 and against the 1000-step reference: the
# fewest steps at which a schedule's FD is within 10% of its own full-step FD, and the fewest
# at which the cosine model's FD is at or below the linear model's full-step FD.
def fewest_steps(name, sampler, target):
    return next((k for k in REDUCED if scores[name, SEEDS[0], sampler, k]["fd"] <= target), None)


def as_steps(n_steps):
    return "at no reduced K" if n_steps is None else f"from K={n_steps}"


for sampler in SAMPLERS:
    own = {name: as_steps(fewest_steps(name, sampler, 1.1 * full_fd[name]))
           for name in SCHEDULE_FUNCTIONS}
    reach = as_steps(fewest_steps("cosine", sampler, full_fd["linear"]))
    print(f"fewest steps | {sampler} | within 10% of its own 1000-step FD: linear "
          f"{own['linear']}, cosine {own['cosine']} | cosine at or below linear's 1000-step "
          f"FD {reach}")'''

CELL_MR_VARIANCE = r'''# A follow-up, run after the rule's verdict and not part of it. At K steps the DDPM sampler's
# second-to-last step jumps from step sampling_steps(K)[1] to step 0 and adds noise of variance
# beta = 1 - alpha_bar[t] / alpha_bar[0]; the last step, told it is at step 0 where almost no
# noise is expected, removes almost none of it. last_noise gives that noise's standard deviation.
def last_noise(schedule, n_steps):
    alpha_bars = schedule["alpha_bars"].double().cpu()
    return float((1 - alpha_bars[sampling_steps(n_steps)[1]] / alpha_bars[0]).sqrt())


# The same sampler with Ho et al.'s smaller posterior variance, beta_tilde =
# (1 - alpha_bar_prev) / (1 - alpha_bar) * beta, in place of the tutorial's beta: each step's
# noise z is scaled by sqrt(beta_tilde / beta) before p_sample_cDDPM adds it. The generator is
# seeded and drawn exactly as in generate, so each image starts from the same noise as before.
@torch.no_grad()
def generate_posterior(model, labels, schedule, n_steps, seed=1337):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    y = labels.to(device)
    x = torch.randn(len(y), channels, image_size, image_size, generator=generator).to(device)
    alpha_bars = schedule["alpha_bars"]
    steps = sampling_steps(n_steps)
    for i in reversed(range(len(steps))):
        prev_index = steps[i - 1] if i > 0 else -1
        z = torch.randn(x.shape, generator=generator).to(device)
        if prev_index >= 0:
            z = z * ((1 - alpha_bars[prev_index]) / (1 - alpha_bars[steps[i]])).sqrt()
        x = p_sample_cDDPM(model, x, steps[i], y, schedule, prev_index, z)
    return ((x.clamp(-1, 1) + 1) / 2).cpu()


assert len(eval_labels) <= 1000, "generate draws one chunk per 1,000 images; so must this"
for n_steps in REDUCED:
    cells = []
    for name in SCHEDULE_FUNCTIONS:
        images = generate_posterior(models[name, SEEDS[0]], eval_labels, schedules[name], n_steps)
        cells.append(f"{name} last-step noise {last_noise(schedules[name], n_steps):.3f} FD with "
                     f"beta {scores[name, SEEDS[0], 'ddpm', n_steps]['fd']:6.2f}, with beta_tilde "
                     f"{fd_to_b(features(images)):6.2f}")
    print(f"variance | K={n_steps:4d} | " + " | ".join(cells))'''

CELL_MR_EPOCHS = r'''# Quality against training, seed 0: every saved snapshot sampled with DDIM at EPOCH_STEPS steps
# under set A's labels. The last epoch is the final seed-0 model, already scored in section 7.
EPOCH_STEPS = __EPOCH_STEPS__
assert EPOCH_STEPS in REDUCED
epoch_list = history["snapshot_epochs"] + [history["epochs"]]
epoch_scores = {}
for epoch in epoch_list:
    cells = []
    for name in SCHEDULE_FUNCTIONS:
        if epoch == history["epochs"]:
            epoch_scores[name, epoch] = scores[name, SEEDS[0], "ddim", EPOCH_STEPS]
        else:
            snapshot = load_unet(checkpoints[name][f"epoch_{epoch}"])
            images = generate(snapshot, eval_labels, schedules[name], EPOCH_STEPS, "ddim")
            epoch_scores[name, epoch] = {"fd": fd_to_b(features(images)),
                                         "accuracy": float(classify(images, eval_labels).mean())}
        cells.append(f"{name} FD {epoch_scores[name, epoch]['fd']:6.2f} accuracy "
                     f"{epoch_scores[name, epoch]['accuracy']:.4f}")
    print(f"epochs | epoch {epoch:3d} | " + " | ".join(cells))
# The first snapshot whose FD is at most 1.1 times the final model's, and the training time
# that took at the measured seconds per epoch.
cells = []
for name in SCHEDULE_FUNCTIONS:
    final = epoch_scores[name, epoch_list[-1]]["fd"]
    reached = next(e for e in epoch_list if epoch_scores[name, e]["fd"] <= 1.1 * final)
    cells.append(f"{name} {reached} ({reached * seconds_per_epoch[name] / 60:.1f} min)")
print("epochs | within 10% of the final FD from epoch: " + ", ".join(cells))'''

CELL_MR_LOSS = r'''# Each seed-0 model's noise-prediction error at 20 evenly spaced steps t, on set A's real images.
# At each step both schedules see the same images and the same noise; only the noise level that
# the schedule assigns to that step differs.
LOSS_STEPS = np.round(np.linspace(0, timesteps - 1, 20)).astype(int).tolist()
x0 = (test_images[set_a] * 2 - 1).to(device)
y0 = eval_labels.to(device)
step_loss, log_snr, signal_left = {name: [] for name in SCHEDULE_FUNCTIONS}, {}, {}
with torch.no_grad():
    for name, schedule in schedules.items():
        signal_left[name] = schedule["alpha_bars"].double().cpu().numpy()[LOSS_STEPS]
        log_snr[name] = np.log(signal_left[name] / (1 - signal_left[name]))
        for step in LOSS_STEPS:
            generator = torch.Generator().manual_seed(SEED + step)
            noise = torch.randn(x0.shape, generator=generator).to(device)
            t = torch.full((len(x0),), step, device=device, dtype=torch.long)
            predicted = models[name, SEEDS[0]](q_sample(x0, t, schedule, noise), t, y0)
            step_loss[name].append(float(((predicted - noise) ** 2).mean()))
for i, step in enumerate(LOSS_STEPS):
    print(f"loss | t={step:3d} | " + " | ".join(
        f"{name} log SNR {log_snr[name][i]:+6.2f} MSE {step_loss[name][i]:.4f}"
        for name in SCHEDULE_FUNCTIONS))'''

CELL_MR_LOSS_SUMMARY = r'''# Three summaries. The mean over the 20 steps estimates the training loss, which draws t evenly.
# The split separates steps with less than 1% of the signal left from the rest. The matched
# comparison interpolates both curves (in log error) onto one grid of log SNR values inside the
# range both schedules cover, and divides the cosine model's error by the linear model's. It is
# reported separately above and below 1% signal: below it both errors are tiny, and a ratio of two
# tiny numbers would dominate an average while saying little about the images.
for name in SCHEDULE_FUNCTIONS:
    errors, noisy = np.array(step_loss[name]), signal_left[name] < 0.01
    print(f"loss | {name} | mean over the 20 steps {errors.mean():.4f} | {noisy.sum()} of 20 "
          f"steps below 0.01 signal, mean MSE there {errors[noisy].mean():.4f}, elsewhere "
          f"{errors[~noisy].mean():.4f}")
low = max(log_snr[name].min() for name in SCHEDULE_FUNCTIONS)
high = min(log_snr[name].max() for name in SCHEDULE_FUNCTIONS)
grid = np.linspace(low, high, 50)
# log SNR falls as t rises, so each curve is reversed to give np.interp rising x values
matched = {name: np.exp(np.interp(grid, log_snr[name][::-1], np.log(step_loss[name])[::-1]))
           for name in SCHEDULE_FUNCTIONS}
ratio = matched["cosine"] / matched["linear"]
threshold = np.log(0.01 / 0.99)  # the log SNR at which 1% of the signal is left
for side, part in (("above", grid >= threshold), ("below", grid < threshold)):
    print(f"loss | matched log SNR {side} {threshold:+.2f} | cosine over linear error: mean "
          f"{ratio[part].mean():.2f}, from {ratio[part].min():.2f} to {ratio[part].max():.2f} | "
          f"largest error linear {matched['linear'][part].max():.4f}, cosine "
          f"{matched['cosine'][part].max():.4f}")'''

CELL_MR_SCHEDULE_FACTS = r'''# The schedule facts the report quotes (DDPM_CosineSchedule.ipynb, sections 3.5, 3.6 and 6.5).
# The weight is the share of the first clean-image estimate, made from pure noise, that the first
# step carries into the second state: sqrt(alpha_bar) there, since beta is close to 1.
steps_10 = sampling_steps(10)
for name, schedule in schedules.items():
    alpha_bars = schedule["alpha_bars"].double().cpu().numpy()
    gains = []
    for n_steps in (10, timesteps):
        first, second = sampling_steps(n_steps)[-1], sampling_steps(n_steps)[-2]
        alpha = alpha_bars[first] / alpha_bars[second]
        gains.append((1 - alpha) / (np.sqrt(alpha) * np.sqrt(1 - alpha_bars[first])))
    noisy_10 = int(sum(alpha_bars[step] < 0.01 for step in steps_10))
    print(f"schedule | {name} | below 0.01: {(alpha_bars < 0.01).mean():.1%} of steps, "
          f"{noisy_10} of K=10's | alpha_bar at t={steps_10[-2]} {alpha_bars[steps_10[-2]]:.5f}, "
          f"weight {np.sqrt(alpha_bars[steps_10[-2]]):.3f} | first-step multiplier K=10 "
          f"{gains[0]:,.2f}, K={timesteps} {gains[1]:,.2f}")
short_end = torch.cumprod(1 - linear_beta_schedule(100), dim=0)[-1]
print(f"schedule | linear over a 100-step process ends at alpha_bar {short_end:.3f}")'''

CELL_MR_FIG2 = r'''# Figure 2: FD (top) and class consistency (bottom) at seed 0 for each number of sampling steps
# K, solid for DDPM and dashed for DDIM. The other seeds' FDs at each reduced K are the small
# dots; the dotted lines are set A's real images, the floor of section 6. The step counts are
# evenly spaced, being the six budgets compared rather than a continuum, and FD is on a log
# scale so the K = 10 values do not flatten the differences at larger K.
COLOURS = {"linear": WARM, "cosine": ACCENT}
LINES = {"ddpm": "-", "ddim": "--"}
place = {n_steps: i for i, n_steps in enumerate(STEP_COUNTS)}
fig, (ax_fd, ax_acc) = plt.subplots(2, 1, figsize=(3.4, 2.2), sharex=True,
                                    gridspec_kw={"height_ratios": (1.1, 1)})
for name, colour in COLOURS.items():
    for sampler, line in LINES.items():
        used = [k for k in STEP_COUNTS if (name, SEEDS[0], sampler, k) in scores]
        x = [place[k] for k in used]
        ax_fd.plot(x, [scores[name, SEEDS[0], sampler, k]["fd"] for k in used], "o" + line,
                   color=colour, ms=2.5, lw=1.1, label=f"{name}, {sampler.upper()}")
        ax_acc.plot(x, [scores[name, SEEDS[0], sampler, k]["accuracy"] for k in used],
                    "o" + line, color=colour, ms=2.5, lw=1.1)
        for seed in SEEDS[1:]:
            ax_fd.plot([place[k] for k in REDUCED],
                       [scores[name, seed, sampler, k]["fd"] for k in REDUCED], ".",
                       color=colour, ms=2.5, alpha=0.5)
ax_fd.axhline(floor, color=MUTED, lw=0.8, ls=":")
ax_acc.axhline(real_a_accuracy, color=MUTED, lw=0.8, ls=":")
ax_fd.set_yscale("log")
ax_fd.set_yticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
ax_fd.set_ylabel("FD to\nreal digits", fontsize=7.5)
ax_acc.set_ylabel("class\nconsistency", fontsize=7.5)
# Room below the lowest consistency for the key, which sits under the curves from K = 50 on.
lowest = min(v["accuracy"] for v in scores.values())
ax_acc.set_ylim(lowest - 0.5 * (1 - lowest), 1.01)
fig.align_ylabels((ax_fd, ax_acc))
ax_acc.set_xticks(range(len(STEP_COUNTS)), [str(k) for k in STEP_COUNTS])
ax_acc.set_xlabel("sampling steps K", fontsize=7.5)
for ax in (ax_fd, ax_acc):
    ax.tick_params(labelsize=6.5)
    ax.minorticks_off()
# The key sits in the lower panel's bottom right, which the curves leave empty, so the FD panel
# needs no headroom for it.
ax_acc.legend(*ax_fd.get_legend_handles_labels(), frameon=False, fontsize=6.5, loc="lower right",
              ncol=2, handlelength=2.2, columnspacing=1.0, borderaxespad=0.2)
plt.tight_layout(h_pad=0.4)
fig.savefig("figure_2_steps.pdf", bbox_inches="tight")
plt.show()'''

CELL_MR_FIG3 = r'''# Figure 3: what fewer steps look like. For each K, one DDPM sample per digit from each seed-0
# model: the first image of each class in set A, so every row starts from the same noise.
SHOWN_STEPS = (10, 20, 50, timesteps)
first_of_class = [int(np.flatnonzero(labels_a == c)[0]) for c in range(n_class)]
rows, row_names = [], []
for n_steps in SHOWN_STEPS:
    for name in SCHEDULE_FUNCTIONS:
        rows.append(kept_images[name, SEEDS[0], "ddpm", n_steps][first_of_class])
        row_names.append(f"{name}, K={n_steps}")
fig, ax = plt.subplots(figsize=(3.4, 2.9))
ax.imshow(utils.make_grid(torch.cat(rows), nrow=n_class, pad_value=1)[0], cmap="gray",
          vmin=0, vmax=1)
ax.set_yticks(12 + 22 * np.arange(len(rows)), row_names, fontsize=7)
ax.set_xticks(12 + 22 * np.arange(n_class), [str(c) for c in range(n_class)], fontsize=7)
ax.tick_params(length=0)
for side in ("left", "bottom", "top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
fig.savefig("figure_3_few_steps.pdf", bbox_inches="tight")
plt.show()'''

CELL_MR_FIG4 = r'''# Figure 4: (a) each seed-0 model's noise-prediction error against the noise level, section 9;
# (b) FD against training epochs, section 8.
fig, (ax_loss, ax_epoch) = plt.subplots(1, 2, figsize=(3.4, 1.9))
for name, colour in COLOURS.items():
    ax_loss.plot(log_snr[name], step_loss[name], "o-", color=colour, ms=2.5, lw=1.0, label=name)
    ax_epoch.plot(epoch_list, [epoch_scores[name, e]["fd"] for e in epoch_list], "o-",
                  color=colour, ms=2.5, lw=1.0)
ax_loss.set(xlabel="log SNR", ylabel="noise MSE", yscale="log", title="(a) error by noise level")
ax_epoch.set(xlabel="epoch", ylabel=f"FD (DDIM, K={EPOCH_STEPS})", title="(b) FD by epoch")
ax_loss.legend(frameon=False, fontsize=6.5)
plt.tight_layout()
fig.savefig("figure_4_mechanism.pdf", bbox_inches="tight")
plt.show()'''

CELL_MR_SUMMARY = r'''# The headline numbers in one machine-readable block, rounded as printed above.
summary = {
    "n_eval": N_EVAL, "seeds": SEEDS, "floor": round(floor, 2),
    "real_accuracy": round(real_accuracy, 4),
    "grid_accuracy": {name: round(value, 4) for name, value in grid_accuracy.items()},
    "full_step_fd": {name: round(value, 2) for name, value in full_fd.items()},
    "winners": {sampler: {str(k): decisions[sampler, k]["winner"] for k in REDUCED}
                for sampler in SAMPLERS},
    "seconds_per_epoch": {name: round(value, 2) for name, value in seconds_per_epoch.items()}}
print(json.dumps(summary, indent=1))'''


# --------------------------------------------------------------------------- markdown

MD_SETUP = r"""## 1 Setup

The libraries, the device and the seed, as in `DDPM_CosineSchedule.ipynb`, with one addition: TF32
is switched off, so every convolution runs in full float32 and the numbers below depend as little
as possible on which GPU, or CPU, runs the notebook. The palette cell keeps every figure in the
project's colours."""

MD_TEST = r"""## 2 Test data

The 10,000 test images and their labels, written by `DDPM_CosineSchedule.ipynb` (section 2).
Neither model saw them in training. The class counts matter in section 6: samples are drawn under
test labels, so every real and generated set compared there has the same class mix."""

MD_MODELS = r"""## 3 The schedules, the network and the samplers

The definitions below are byte for byte those of `DDPM_CosineSchedule.ipynb`, so the checkpoints
are read by the code that trained them. First the two schedules: the tutorial's `extract` and
`linear_beta_schedule`, then `cosine_beta_schedule` (section 3 there)."""

MD_MODELS_2 = r"""One dictionary of arrays per schedule (section 3.2 there), and the tutorial's `q_sample`
(section 3.3 there), which section 9 uses to noise real test images."""

MD_MODELS_3 = r"""The tutorial's network, `ConvBlock` and `UNet_cond`, unchanged (section 4 there)."""

MD_MODELS_4 = r"""The samplers (section 6 there). `sampling_steps(K)` lists the K steps a sampler visits, and
`p_sample_cDDPM` is the tutorial's reverse step, able to skip steps, with its clean-image estimate
clipped to the data range."""

MD_MODELS_5 = r"""The deterministic DDIM step, and the loop that runs either sampler from seeded noise.
`generate` draws large sets in chunks of 1,000 and returns them in [0, 1]."""

MD_MODELS_6 = r"""The evaluation classifier (`DigitClassifier.ipynb`, section 3). `features()` returns its
128-unit layer."""

MD_LOAD = r"""## 4 Checkpoints and training records

From `DDPM_history.json`: where the models were trained, the settings both schedules share, and
for each schedule the seconds per epoch, the minutes per training run and each seed's final
training loss. Then the two checkpoints the brief asks for, one per schedule, each holding the
final model of every seed and seed 0's earlier snapshots, and the classifier. The printed keys
and parameter counts identify what was loaded."""

MD_FIG1 = r"""## 5 Figure 1: the two 12 x 10 grids

Twelve samples per digit with the digits as rows, the brief's layout, from each schedule's seed-0
model with the tutorial's full 1000-step sampler. Both grids start from the same noise, so the two
images in the same place differ only by the model. `classify` returns the classifier's verdict on each
image; it is used here and in every later section."""

MD_GRID_ACCURACY = r"""The share of each grid's 120 images the classifier reads as the digit they were drawn as."""

MD_SCORING = r"""## 6 How samples are scored

Three measures, each computed on a set of samples drawn under test labels:

- **class consistency**: the share of samples the classifier reads as the intended digit;
- **Frechet distance (FD)** to real test digits in the classifier's 128-unit feature layer: the
  distance between two Gaussians fitted to the two clouds of feature vectors. It is 0 when the
  means and covariances match, and grows as the samples look less like real digits or vary more
  or less than they do. FID is the same quantity in the features of an ImageNet network; this
  one uses a network built for 20 x 20 digits;
- **spread**: within each class, the variety of the samples' features relative to real digits'.
  Near 1 matches real variety; well below 1 means the samples of a class look alike, the
  signature of mode collapse.

FD depends on how many images are compared and in which class mix. The test set is therefore split
into two disjoint sets of the same size and identical class counts. Samples are drawn under set A's
labels and compared with set B's real images, and set A's own real images against set B give the
**floor**, what a model drawing real digits would score."""

MD_METRICS = r"""The feature layer, the FD against set B (whose mean, covariance and covariance square root
are computed once) and the spread."""

MD_FLOOR = r"""The reference values: the classifier's accuracy on all 10,000 real test images, and set A's real
images scored exactly as a sample set will be.

A **paired bootstrap** then turns one FD difference into an interval. Sample i of a linear set and
sample i of the matching cosine set share their label and their starting noise, so both sets are
resampled with the same indices and the difference recomputed each time. The middle 95% of the
resampled differences is the interval; if it excludes 0, the difference is not an accident of which
samples were drawn."""

MD_SWEEP = r"""## 7 Quality against the number of sampling steps

The experiment behind the hypothesis. Each model is sampled with K = 10, 20, 50, 100 and 250 steps by
both samplers, and with the tutorial's 1000-step sampler as the full-quality reference. DDIM is not
run at 1000 steps: it exists to take fewer, and the reference is the tutorial's own sampler. Seed
0's models are sampled at every K, the models of the other two seeds at every reduced K. Every set
uses set A's labels and the same starting noise, so each linear set is paired with a cosine set.

Each printed line gives the seed, the sampler, K, and for each schedule the FD (lower is better),
the class consistency and the spread."""

MD_RULE = r"""### 7.1 The decision rule

The rule was fixed before any sample was scored. At each reduced K and for each sampler, a schedule
counts as **better** only if three things hold together: its FD is lower at seed 0, the paired 95%
interval of the difference excludes 0, and its FD is lower again at every other seed. Otherwise
that step count is **unclear**. The `verdict` lines group the step counts by outcome. The last line
compares the two models at the full 1000 steps, where the hypothesis makes no claim but every
reduced K is measured against it."""

MD_FEWEST = r"""### 7.2 How few steps are enough

Two step counts summarise each curve, both at seed 0 and against the 1000-step reference: the
fewest steps at which a schedule's FD comes within 10% of its own full-step FD, and the fewest at
which the cosine model's FD reaches the linear model's full-step FD."""

MD_VARIANCE = r"""### 7.3 A follow-up: the noise the last step leaves

Run after the rule gave its verdict, and not part of it. Figure 3 shows the linear model's DDPM
samples at few steps on a speckled background, and the cosine model's on a clean one. With K steps
the sampler's second-to-last step jumps from a step well above 0 straight to step 0 and adds noise
of variance $\beta = 1 - \bar\alpha_t / \bar\alpha_0$. The last step is told it is at step 0, where
almost no noise is expected, so it removes almost none of it. The linear schedule's
$\bar\alpha_t$ falls faster near the start, so that jump, and the noise, is larger.

Each line gives, per schedule, the standard deviation of that noise, the seed-0 FD from section 7
with the tutorial's variance $\beta$, and the FD of the same sampler, from the same starting noise,
with the smaller posterior variance of Ho et al. (2020),
$\tilde\beta = \frac{1 - \bar\alpha_{t'}}{1 - \bar\alpha_t}\,\beta$, which adds almost no noise on that
jump."""

MD_EPOCHS = r"""## 8 Training time and quality against epochs

The schedule changes which noise level each training step uses, not the computation a step takes,
so the seconds per epoch of section 4 should match. What it can change is how many epochs a model
needs. Seed 0 kept snapshots of its weights during training, at the epochs listed in
`DDPM_history.json`. Each is sampled with DDIM at __EPOCH_STEPS__ steps under set A's labels and
scored as in section 7. The last line gives, for each schedule, the first epoch whose FD is within
10% of the final model's, and the training time that took."""

MD_LOSS = r"""## 9 The error at each noise level

A model's training loss is its error averaged over steps t drawn evenly from 1000. The two schedules
put those steps at different noise levels, so their training losses average different mixes of
easy and hard questions and cannot be compared directly. Here both seed-0 models answer the same
questions: set A's real images noised at 20 evenly spaced steps t, with the same noise for both.
Each line gives, per schedule, the noise level at that step as log SNR and the mean squared error
of the predicted noise. Where almost no signal is left, $x_t$ is nearly the noise itself and
predicting the noise is easy."""

MD_LOSS_SUMMARY = r"""Three summaries per schedule: the mean over the 20 steps, an estimate of the training loss; the
error at the steps with less than 1% of the signal left against the others; and the two models
compared at the same noise level, over the range of log SNR both schedules cover."""

MD_FACTS = r"""## 10 Where the schedules put their steps

The schedule facts the report quotes, recomputed from the schedules: the share of the 1000 steps
with less than 1% of the signal left, and how many of a 10-step sampler's steps fall there; the
signal left at a 10-step sampler's second step, and the weight $\sqrt{\bar\alpha}$ with which the
first clean-image estimate, made from pure noise, enters it; and the multiplier the tutorial's form
of the step puts on an error in the noise estimate at the first step (`DDPM_CosineSchedule.ipynb`,
section 6.5). The last line is the tutorial's linear schedule over a 100-step process (section 3.6
there)."""

MD_FIG2 = r"""## 11 Figures 2 to 4

Figures 1 and 2 are the report's. Figures 3 and 4 show the same results from other angles and
are not reproduced in the two-page report.

**Figure 2.** FD (top) and class consistency (bottom) against K at seed 0, solid for DDPM and
dashed for DDIM. The other seeds' FDs at each reduced K are the small dots. The dotted lines are
set A's real images, the floor of section 6."""

MD_FIG3 = r"""**Figure 3.** What fewer steps look like: one DDPM sample per digit from each seed-0 model at
K = 10, 20, 50 and 1000. Every row starts from the same noise."""

MD_FIG4 = r"""**Figure 4.** (a) The noise-prediction error of section 9 against log SNR. (b) FD against
training epochs, section 8."""

MD_SUMMARY = r"""## 12 Summary

The headline numbers in one machine-readable block."""


def report_notebook(nb):
    """The cells of main_report.ipynb, built from the shared cells of build_notebooks (nb)."""
    intro = (
        "This notebook reproduces **every number, table and figure in the report**. It trains "
        "nothing. It loads the two checkpoints `ddpm_linear.pth` and `ddpm_cosine.pth`, the "
        "evaluation classifier `DigitClassifier.pth`, the training record `DDPM_history.json` "
        "and the test split `mnist_custom_test.pt`, then samples both models and scores the "
        "samples. " + RUNTIME + "\n\n"
        "The hypothesis under test:\n\n" + nb.HYPOTHESIS + "\n\n"
        "Both models were trained at the tutorial's T = 1000 steps; \"fewer time steps\" is "
        "read as fewer **sampling** steps K (`DDPM_CosineSchedule.ipynb`, section 3.6 gives "
        "the reason). Every model, schedule and sampler is explained where it is built, in "
        "`DDPM_CosineSchedule.ipynb`, and the classifier in `DigitClassifier.ipynb`. The "
        "lead-ins here say what each output is and how to read it.\n\n"
        "**Sections:** 1 Setup, 2 Test data, 3 The schedules, the network and the samplers, "
        "4 Checkpoints and training records, 5 Figure 1, 6 How samples are scored, 7 Quality "
        "against the number of sampling steps, 8 Training time and quality against epochs, "
        "9 The error at each noise level, 10 Where the schedules put their steps, 11 Figures 2 "
        "to 4, 12 Summary.")
    markdown, code, fill = nb.markdown, nb.code, nb.fill
    return [
        markdown("md-title", fill(nb.HEADER, __TITLE__="main_report", __INTRO__=intro)),
        markdown("md-setup", MD_SETUP),
        code("code-imports-report", nb.imports("report") + PRECISION),
        code("code-style", nb.CELL_STYLE),
        markdown("md-test", MD_TEST),
        code("code-test-data", CELL_MR_DATA),
        markdown("md-models", MD_MODELS),
        code("code-tutorial-forward", nb.CELL_TUTORIAL_FORWARD),
        code("code-cosine", nb.CELL_COSINE),
        markdown("md-models-2", MD_MODELS_2),
        code("code-schedules", nb.CELL_SCHEDULES),
        code("code-qsample", nb.CELL_QSAMPLE),
        markdown("md-models-3", MD_MODELS_3),
        code("code-convblock", nb.CELL_CONVBLOCK),
        code("code-unet", nb.CELL_UNET),
        markdown("md-models-4", MD_MODELS_4),
        code("code-steps", nb.CELL_STEPS),
        code("code-p-sample", nb.CELL_P_SAMPLE),
        markdown("md-models-5", MD_MODELS_5),
        code("code-ddim", nb.CELL_DDIM),
        code("code-loop", nb.CELL_LOOP),
        markdown("md-models-6", MD_MODELS_6),
        code("code-classifier-model", nb.CELL_CLASSIFIER),
        markdown("md-load", MD_LOAD),
        code("code-records", CELL_MR_RECORDS),
        code("code-load", CELL_MR_LOAD),
        markdown("md-fig1", MD_FIG1),
        code("code-classify", CELL_MR_CLASSIFY),
        code("code-fig1", CELL_MR_FIG1),
        markdown("md-grid-accuracy", MD_GRID_ACCURACY),
        code("code-grid-accuracy", CELL_MR_GRID_ACCURACY),
        markdown("md-scoring", MD_SCORING),
        code("code-sets", fill(CELL_MR_SETS)),
        markdown("md-metrics", MD_METRICS),
        code("code-metrics", CELL_MR_METRICS),
        markdown("md-floor", MD_FLOOR),
        code("code-floor", CELL_MR_FLOOR),
        code("code-bootstrap", fill(CELL_MR_BOOTSTRAP)),
        markdown("md-sweep", MD_SWEEP),
        code("code-plan", CELL_MR_PLAN),
        code("code-sweep", CELL_MR_SWEEP),
        markdown("md-rule", MD_RULE),
        code("code-rule", CELL_MR_RULE),
        markdown("md-fewest", MD_FEWEST),
        code("code-fewest", CELL_MR_FEWEST),
        markdown("md-variance", MD_VARIANCE),
        code("code-variance", CELL_MR_VARIANCE),
        markdown("md-epochs", fill(MD_EPOCHS)),
        code("code-epochs", fill(CELL_MR_EPOCHS)),
        markdown("md-loss", MD_LOSS),
        code("code-loss", CELL_MR_LOSS),
        markdown("md-loss-summary", MD_LOSS_SUMMARY),
        code("code-loss-summary", CELL_MR_LOSS_SUMMARY),
        markdown("md-facts", MD_FACTS),
        code("code-schedule-facts", CELL_MR_SCHEDULE_FACTS),
        markdown("md-fig2", MD_FIG2),
        code("code-fig2", CELL_MR_FIG2),
        markdown("md-fig3", MD_FIG3),
        code("code-fig3", CELL_MR_FIG3),
        markdown("md-fig4", MD_FIG4),
        code("code-fig4", CELL_MR_FIG4),
        markdown("md-summary", MD_SUMMARY),
        code("code-summary", CELL_MR_SUMMARY),
    ]


# How long a full run takes, measured on the Hub GPU node (474 s and 523 s in two runs).
RUNTIME = ("It draws tens of thousands of samples, so it takes about 9 minutes on the IFN680 "
           "GPU server and far longer on a CPU.")

FIGURE_FILES = ("figure_1_grids.pdf", "figure_2_steps.pdf", "figure_3_few_steps.pdf",
                "figure_4_mechanism.pdf")


def rails(check, source):
    """Checks that apply to main_report.ipynb only. source is all of its code, joined."""
    lines = [line.strip() for line in source.split("\n")]
    check("main_report.ipynb: trains nothing (no backward pass, no optimiser)",
          ".backward(" not in source and "optim." not in source and ".step()" not in source)
    loads = [line for line in lines if "torch.load(" in line]
    check("main_report.ipynb: every torch.load uses weights_only=True",
          all("weights_only=True" in line or line.endswith(",") for line in loads)
          and source.count("weights_only=True") == source.count("torch.load("),
          f"{len(loads)} loads")
    check("main_report.ipynb: reads the test split file, never the full dataset",
          '"mnist_custom.pt"' not in source and "train_images" not in source)
    unseeded = [line for line in lines if re.search(r"\brandn(_like)?\(", line)
                and "generator=" not in line
                and line != "noise = torch.randn_like(x0).to(device)"]
    check("main_report.ipynb: every random draw it makes names a seeded generator", not unseeded,
          "; ".join(unseeded))
    check("main_report.ipynb: q_sample is always given its noise",
          "q_sample(x0, t, schedule, noise)" in source)
    saved = re.findall(r'savefig\("([^"]+)"', source)
    check("main_report.ipynb: saves exactly the report's figure files",
          tuple(saved) == FIGURE_FILES, ", ".join(saved))
