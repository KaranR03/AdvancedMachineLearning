r"""Generate notebook/LLMForward.ipynb, LLMReverse.ipynb and main_report.ipynb from one template.

The two training notebooks differ only in which direction the target answer is written, and both
share their tokeniser, data generator, model and evaluation code with main_report.ipynb. Writing
those cells once here keeps the three files consistent and makes a change to the shared code a
one-line edit instead of three.

The generated notebooks carry no outputs. Run tools/run_notebook.py on each one to execute them,
or tools/smoke_test.py for a reduced-size rehearsal on CPU.

Run:  python tools/build_notebooks.py            full sizes, for the real run
      python tools/build_notebooks.py --smoke    tiny sizes, written to the scratch directory
"""
import io
import json
import os
import re
import sys

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-08/project-5-extending-addition-llm")
SMOKE = "--smoke" in sys.argv
NBDIR = (os.environ.get("CLAUDE_JOB_DIR", "C:/Users/Admin/.claude/jobs/8640b033") + "/tmp/smoke"
         if SMOKE else f"{BASE}/notebook")

# Sizes. The real run uses the same 150k/10k/10k split the imported draft used, so the trained
# checkpoints stay comparable; the ablation re-trains on the first 80k of that same training set.
if SMOKE:
    TRAIN, VAL, TEST, ABLATION, EPOCHS = 2000, 500, 500, 1000, 2
else:
    TRAIN, VAL, TEST, ABLATION, EPOCHS = 150000, 10000, 10000, 80000, 30

# House palette from docs/assessment-writing-guide.md section 6, so the figures read as a set.
PALETTE = '''INK, ACCENT, WARM = "#1f2933", "#2f6f9f", "#c1553b"
MUTED, PALE, GREEN = "#7b8794", "#e8ecf1", "#3f7d58"'''

# --------------------------------------------------------------------------- shared code cells

CELL_IMPORTS = '''\
# Same stack as the Week 7 workshop, plus json for the machine readable result dumps.
import json
import math
__IMPORT_OS__import pickle
import random
import re
__IMPORT_TIME__

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# The brief asks for the GPU environment. The model still runs on CPU when no GPU is visible,
# so the notebook runs unchanged on a machine without one and gives the same answers.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")
print(f"torch  : {torch.__version__}")
print(f"numpy  : {np.__version__}")

# Every result quoted in the report is reproducible from this seed alone, including the
# train/validation/test split, which is regenerated and checked further down.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
# Bound to _ because manual_seed returns the Generator it just seeded, and a bare call as the
# last line of a cell would store that object's repr as an output.
_ = torch.manual_seed(SEED)
'''

CELL_TOKENIZER = '''\
# The workshop's vocabulary plus one "-" token. That token does double duty: it is the
# subtraction operator inside a prompt ("50-73=") and the negative sign inside a result ("-23").
# Reusing one symbol keeps the vocabulary at 15 and lets position carry the distinction, which
# is something the attention mechanism can learn and an extra token would have hidden.
pad_token = "[PAD]"
eos_token = "[EOS]"


class character_level_tokenizer:
    def __init__(self):
        self.vocab = [str(x) for x in range(10)] + ["+", "-", "="] + [pad_token, eos_token]
        self.token_to_id = {v: k for k, v in enumerate(self.vocab)}
        self.id_to_token = {k: v for k, v in enumerate(self.vocab)}
        self.ntokens = len(self.vocab)
        # Anything outside the vocabulary is stripped rather than mapped to an unknown token,
        # because every string this model ever sees is generated, not typed by a person.
        self.pattern = f"[^{re.escape(''.join(self.vocab))}]"

    def clean(self, text):
        return re.sub(self.pattern, "", text)

    def pre_tokenization(self, text):
        return [character for character in text]

    def encode(self, text):
        return [self.token_to_id[c] for c in self.pre_tokenization(self.clean(text))]

    def decode(self, token_list):
        return "".join([self.id_to_token[i] for i in token_list])


tokenizer = character_level_tokenizer()
ntokens = tokenizer.ntokens
print(f"ntokens: {ntokens} | vocab: {tokenizer.vocab}")
'''

CELL_DATA = '''\
def build_dataset(train_size, val_size, test_size, number_digits=3, seed=SEED):
    """Unique, non-overlapping splits with a balanced 50/50 addition and subtraction mix.

    Uniqueness is enforced per operation on the prompt string, then the two streams are
    interleaved and shuffled once before being sliced. Slicing a single shuffled list is what
    makes the three splits disjoint by construction rather than by a later check.
    """
    rng = random.Random(seed)
    total = train_size + val_size + test_size
    per_op = total // 2 + 1

    def unique_for(operation, n):
        seen, out, hi = set(), [], 10 ** number_digits
        while len(out) < n:
            a_int, b_int = rng.randint(0, hi - 1), rng.randint(0, hi - 1)
            prompt = f"{a_int}{operation}{b_int}="
            if prompt in seen:
                continue
            seen.add(prompt)
            # str() of a negative int already carries the leading "-", so the target needs no
            # special casing for b > a.
            result = a_int + b_int if operation == "+" else a_int - b_int
            out.append((prompt, str(result)))
        return out

    additions, subtractions = unique_for("+", per_op), unique_for("-", per_op)
    mixed = []
    for pair_add, pair_sub in zip(additions, subtractions):
        mixed += [pair_add, pair_sub]
    rng.shuffle(mixed)
    return (mixed[:train_size],
            mixed[train_size:train_size + val_size],
            mixed[train_size + val_size:train_size + val_size + test_size])
'''

CELL_REVERSE_HELPERS = '''\
# Reverse mode reverses the whole answer string literally, so "31" becomes "13" and the units
# digit is emitted first. For a negative result "-123" becomes "321-": the sign lands last,
# which is consistent with right-to-left arithmetic because the sign of a difference is only
# known once the magnitude has been worked out. The brief does not prescribe a rule here, so
# this is a choice, and reversing the string literally is the choice that needs no extra logic.
def to_reverse(text):
    return text[::-1]


def from_reverse(text):
    return text[::-1]


def parse_answer(answer_string, reverse=False):
    """Turn a decoded answer back into an int, un-reversing first when in reverse mode."""
    text = from_reverse(answer_string) if reverse else answer_string
    negative = text.startswith("-")
    digits = "".join(c for c in text if c.isdigit())
    if digits == "":
        return None  # the model emitted no digit at all; counted as an error, never as a zero
    return -int(digits) if negative else int(digits)


def operation_of(prompt):
    return "+" if "+" in prompt else "-"


def carry_positions(a_int, b_int):
    """Units-first flags marking each column of a + b that produces a carry."""
    width = max(len(str(a_int)), len(str(b_int)), len(str(a_int + b_int)))
    left, right = str(a_int).zfill(width), str(b_int).zfill(width)
    carry, flags = 0, []
    for i in reversed(range(width)):
        flags.append(carry)
        carry = 1 if int(left[i]) + int(right[i]) + carry >= 10 else 0
    return flags


def borrow_positions(a_int, b_int):
    """Units-first flags marking each column of a - b that requires a borrow.

    Borrowing is a property of the magnitudes, so the larger operand is put on top first. That
    makes 12-99 and 99-12 share a borrow structure, which is what the arithmetic actually does.
    """
    hi, lo = (a_int, b_int) if a_int >= b_int else (b_int, a_int)
    width = max(len(str(hi)), len(str(lo)))
    top, bottom = str(hi).zfill(width), str(lo).zfill(width)
    borrow, flags = 0, []
    for i in reversed(range(width)):
        upper, lower = int(top[i]) - borrow, int(bottom[i])
        flags.append(1 if upper < lower else 0)
        borrow = 1 if upper < lower else 0
    return flags
'''

CELL_POSENC = '''\
class PositionalEncoding(nn.Module):
    """Sinusoidal absolute positional encoding, exactly as the workshop defines it.

    Absolute rather than relative positions matter for this project: in reverse mode the model
    has to associate "the position 0 token" with the units digit of an answer whose length it
    has not yet committed to, and that is the comparison Task 2 is asking about.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2).float() / d_model) * (-math.log(1e4)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Stored as (max_len, 1, d_model) to match the (sequence, batch, feature) layout the
        # rest of the notebook uses, and registered as a buffer so it moves with .to(device).
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
'''

CELL_MODEL = '''\
class CustomEncoderLayer(nn.TransformerEncoderLayer):
    """The workshop's encoder layer, which keeps its attention weights for inspection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src2, attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
            need_weights=True, average_attn_weights=False)
        self.attn_weights = attn_weights
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1):
        super().__init__()
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = CustomEncoderLayer(ninp, nhead, nhid, dropout)
        # enable_nested_tensor is switched off explicitly. This model runs with
        # batch_first=False, so the nested-tensor fast path is unusable anyway and PyTorch
        # would otherwise emit a UserWarning on every construction.
        self.encoder = nn.TransformerEncoder(encoder_layers, nlayers, enable_nested_tensor=False)
        self.decoder = nn.Linear(ninp, ntoken)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, size):
        # log of a lower-triangular matrix of ones: 0 where attention is allowed, -inf above the
        # diagonal. Adding it to the attention scores is what makes the model causal.
        return torch.log(torch.tril(torch.ones(size, size)))

    def forward(self, src):
        mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output_enc = self.encoder(src, mask=mask)
        output_dec = self.decoder(output_enc)
        attention_maps = [layer.attn_weights for layer in self.encoder.layers]
        return F.log_softmax(output_dec, dim=-1), output_enc, attention_maps
'''

CELL_GENERATION = '''\
# Three utilities the workshop keeps separate: decoding, padding, and assembling a batch.
# Each takes the model or the data as an argument, so the same code serves training and
# evaluation and nothing here reads a global.
def generate(model, prompts, new_tokens=6):
    """Greedy autoregressive decoding: append the arg max token, then feed it back in.

    Greedy rather than sampled, so decoding is deterministic: re-running this notebook on
    different hardware gives the same predictions.
    """
    input_tensor = prompts.to(device)
    for _ in range(new_tokens):
        output, _, _ = model(input_tensor)
        token = torch.argmax(output[-1, :, :], -1).view((1, -1))
        input_tensor = torch.cat((input_tensor, token), 0)
    return input_tensor


def pad(token_list, type_list="prompts"):
    """Prompts are left-padded, answers are right-padded after an [EOS].

    Left-padding the prompts keeps the "=" in the same relative place as the generation start,
    so the first predicted token always follows the equals sign whatever the operand widths.
    """
    max_length = max(len(x) for x in token_list)
    out = []
    for x in token_list:
        if type_list == "prompts":
            out.append([tokenizer.token_to_id[pad_token]] * (max_length - len(x)) + x)
        else:
            out.append(x + [tokenizer.token_to_id[eos_token]] +
                       [tokenizer.token_to_id[pad_token]] * (max_length - len(x)))
    return out, max_length


def get_batch(data, i, batch_size, reverse):
    """One batch of (prompt, answer) pairs. `reverse` decides the target's digit order only."""
    j_end = min(i + batch_size, len(data))
    prompts = [tokenizer.encode(data[j][0]) for j in range(i, j_end)]
    padded_prompts, length_prompts = pad(prompts, "prompts")
    answers = [tokenizer.encode(to_reverse(data[j][1]) if reverse else data[j][1])
               for j in range(i, j_end)]
    padded_answers, length_answers = pad(answers, "answers")
    X = torch.stack([torch.tensor(x) for x in padded_prompts], 1)
    Y = torch.stack([torch.tensor(x) for x in padded_answers], 1)
    return X, Y, length_prompts, length_answers
'''

CELL_EVALUATE = '''\
# Two views of accuracy. evaluate() is the workshop's token-level measure, used to watch
# convergence while training; predict_numeric() decodes to integers, which is what Task 3
# scores, because whether the sum is right is a question about the number not the tokens.
def evaluate(model, input_data, reverse, batch_size=200):
    """Token-level and whole-sequence accuracy, as defined in the workshop.

    The model is passed in rather than read from the enclosing scope, so the same function can
    score the trained model and the ablation model without either of them being a global.
    """
    model.eval()
    correct_digit = total_digit = correct = total = 0.0
    with torch.no_grad():
        for i in range(0, len(input_data), batch_size):
            prompts, targets, length_prompts, length_answers = get_batch(
                input_data, i, batch_size, reverse)
            prompts, targets = prompts.to(device), targets.to(device)
            output = generate(model, prompts, length_answers + 1)
            answers_tokens = output[length_prompts:, :]
            target_mask_pad = targets != tokenizer.token_to_id[pad_token]
            equality_test = answers_tokens == targets
            correct_digit += torch.logical_and(target_mask_pad, equality_test).sum().item()
            total_digit += target_mask_pad.sum().item()
            correct += torch.all(torch.logical_or(~target_mask_pad, equality_test),
                                 axis=0).float().sum().item()
            total += targets.shape[-1]
    return correct_digit / total_digit, correct / total


def predict_numeric(model, input_data, reverse, batch_size=200, new_tokens=6):
    """Decode each prompt to an int. Returns (prompts, true ints, predicted ints).

    A prediction that cannot be parsed stays None rather than becoming 0, so an unparseable
    output is never silently scored as a correct answer to "a - a =".
    """
    model.eval()
    prompts_out, trues, preds = [], [], []
    with torch.no_grad():
        for i in range(0, len(input_data), batch_size):
            chunk = input_data[i:i + batch_size]
            encoded = [tokenizer.encode(p) for p, _ in chunk]
            padded, length_prompts = pad(encoded, "prompts")
            X = torch.stack([torch.tensor(x) for x in padded], 1).to(device)
            generated = generate(model, X, new_tokens)[length_prompts:, :]
            for b, (prompt, target) in enumerate(chunk):
                text = tokenizer.decode(generated[:, b].tolist())
                text = text.split(eos_token)[0].replace(pad_token, "")
                prompts_out.append(prompt)
                trues.append(int(target))
                preds.append(parse_answer(text, reverse=reverse))
    return prompts_out, trues, preds
'''

# ------------------------------------------------------------------- training notebook cells

CELL_TRAIN_DATA = '''\
# The same split feeds both models, so the comparison in main_report.ipynb is between two
# training directions rather than between two datasets.
train_size, val_size, test_size = __TRAIN__, __VAL__, __TEST__
data_train, data_val, data_test = build_dataset(train_size, val_size, test_size)

# The held-out set is written once and then reused. Whichever notebook runs first creates it;
# the other loads it, which is what guarantees both models are scored on identical examples.
if not os.path.exists("project5_testset.pkl"):
    with open("project5_testset.pkl", "wb") as handle:
        pickle.dump(data_test, handle)
    print("wrote project5_testset.pkl")
else:
    with open("project5_testset.pkl", "rb") as handle:
        data_test = pickle.load(handle)
    print("loaded the shared project5_testset.pkl")

# Slicing one shuffled list already guarantees the three splits are disjoint. The assertion
# below restates that guarantee as something the reader can watch pass, because a held-out
# score means nothing if a test prompt was also trained on.
train_prompts = {p for p, _ in data_train} | {p for p, _ in data_val}
overlap = sum(1 for p, _ in data_test if p in train_prompts)
assert overlap == 0, f"{overlap} test prompts also appear in train or validation"
assert len(data_test) >= __MIN_TEST__, "the brief asks for a held-out set of at least 10k samples"

n_sub = sum(1 for p, _ in data_test if "-" in p)
print(f"train {len(data_train)}  val {len(data_val)}  test {len(data_test)}")
print(f"test subtraction fraction {n_sub / len(data_test):.2f}")
print(f"train/test prompt overlap {overlap}")
print(f"examples: {data_train[:4]}")
'''

CELL_TRAIN_SETUP = '''\
learning_rate = 1e-3
epochs = __EPOCHS__
batch_size = 100

# dropout 0.1 rather than the class default of 0.5: the workshop's own encoder layer uses 0.1,
# and the add-and-subtract task needs the capacity. The cosine schedule anneals the learning
# rate to near zero by the last epoch, which is what settles the final digit accuracy.
torch.manual_seed(SEED)
model = TransformerModel(ntoken=ntokens, ninp=128, nhead=16, nhid=64, nlayers=6, dropout=0.1)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")


def train_epoch(model, optimizer, data, reverse, batch_size=100):
    """One pass of next-token prediction over `data`. Cross-entropy on the answer tokens only.

    Everything the function needs is an argument, so the identical code trains the main model
    and the reduced-data ablation model further down without either touching a global.
    """
    model.train()
    total_loss, n_batch = 0.0, 0
    for i in range(0, len(data) - 1, batch_size):
        n_batch += 1
        prompts, targets, length_prompts, _ = get_batch(data, i, batch_size, reverse)
        prompts, targets = prompts.to(device), targets.to(device)
        # Prompt and answer are concatenated into one causal sequence; the loss is taken only
        # over the positions that predict answer tokens, which is the slice below.
        input_tensor = torch.cat((prompts, targets), 0)
        optimizer.zero_grad()
        output, _, _ = model(input_tensor)
        output_answers = output[length_prompts - 1:-1, :, :].reshape(-1, ntokens)
        loss = F.cross_entropy(output_answers, targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / n_batch
'''

CELL_TRAIN_LOOP = '''\
# Validation runs every epoch, which is what turns training into a curve that can be read for
# convergence instead of a single final number. Training accuracy is measured on a fixed subset
# because scoring the full training set every epoch would cost more than the epoch itself.
history = {"loss": [], "val_digit": [], "val_seq": [], "train_seq": []}
started = time.time()
for epoch in range(1, epochs + 1):
    epoch_start = time.time()
    loss = train_epoch(model, optimizer, data_train, REVERSE, batch_size)
    scheduler.step()
    val_digit, val_seq = evaluate(model, data_val, REVERSE)
    _, train_seq = evaluate(model, data_train[:__TRAIN_PROBE__], REVERSE)
    history["loss"].append(loss)
    history["val_digit"].append(val_digit)
    history["val_seq"].append(val_seq)
    history["train_seq"].append(train_seq)
    print(f"epoch {epoch:2d} ({time.time() - epoch_start:5.1f}s)  loss {loss:.4f}  "
          f"val digit {val_digit:.4f}  val seq {val_seq:.4f}  train seq {train_seq:.4f}")
print(f"total training time: {(time.time() - started) / 60:.1f} min")
'''

CELL_TRAIN_SAVE = '''\
# The checkpoint is what main_report.ipynb loads; the history is what lets it draw the learning
# curves without containing a training loop, which the brief forbids there.
torch.save(model.state_dict(), "__CKPT__")
with open("__HIST__", "w") as handle:
    json.dump(history, handle)
print(f"saved __CKPT__ and __HIST__")
'''

CELL_ABLATION = '''\
# Does prediction direction change how much DATA the task needs, as opposed to how accurate the
# final model is? The only way to answer that is to train the same architecture on less data and
# compare the curves, so this repeats the run above on the first __ABLATION__ training examples
# and saves its history. It deliberately does not overwrite the checkpoint: this model exists to
# be plotted, not to be scored.
torch.manual_seed(SEED)
model_small = TransformerModel(ntoken=ntokens, ninp=128, nhead=16, nhid=64, nlayers=6, dropout=0.1)
model_small.to(device)
optimizer_small = torch.optim.AdamW(model_small.parameters(), lr=learning_rate)
scheduler_small = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_small, T_max=epochs)

data_small = data_train[:__ABLATION__]
history_small = {"loss": [], "val_digit": [], "val_seq": [], "train_seq": []}
started = time.time()
for epoch in range(1, epochs + 1):
    loss = train_epoch(model_small, optimizer_small, data_small, REVERSE, batch_size)
    scheduler_small.step()
    val_digit, val_seq = evaluate(model_small, data_val, REVERSE)
    history_small["loss"].append(loss)
    history_small["val_digit"].append(val_digit)
    history_small["val_seq"].append(val_seq)
    # The full run records training accuracy on a fixed probe subset; the ablation skips that
    # measurement to keep the two runs equal in cost per epoch. NaN keeps the four history
    # lists the same length without drawing a line matplotlib would otherwise interpolate.
    history_small["train_seq"].append(float("nan"))
    print(f"epoch {epoch:2d}  loss {loss:.4f}  val digit {val_digit:.4f}  val seq {val_seq:.4f}")
print(f"ablation on {len(data_small)} examples finished in {(time.time() - started) / 60:.1f} min")

with open("__HIST_SMALL__", "w") as handle:
    json.dump(history_small, handle)
print(f"saved __HIST_SMALL__")
'''

CELL_TRAIN_CURVE = '''\
__PALETTE__

# Both runs on one axis: the full training set solid, the reduced one dashed. The 0.78 line is
# the level the brief asks the finished model to clear.
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
epochs_axis = range(1, len(history["val_seq"]) + 1)
ax.plot(epochs_axis, history["val_seq"], color=ACCENT, lw=1.8,
        label=f"validation, {len(data_train):,} examples")
ax.plot(epochs_axis, history_small["val_seq"], color=WARM, lw=1.8, ls="--",
        label=f"validation, {len(data_small):,} examples")
ax.plot(epochs_axis, history["train_seq"], color=MUTED, lw=1.0, ls=":", label="training subset")
ax.axhline(__TARGET__, color=GREEN, lw=1.0, ls="-.", label="__TARGET__ target")
ax.set_xlabel("epoch")
ax.set_ylabel("sequence accuracy")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("__MODE__ mode: convergence at two training set sizes", fontsize=10, color=INK)
plt.tight_layout()
plt.show()
'''

CELL_TRAIN_SANITY = '''\
# A dozen worked examples, printed so the notebook shows what the model actually does rather
# than only what it scores. The held-out numbers here are the ones main_report.ipynb reproduces.
_, _, sample_preds = predict_numeric(model, data_test[:12], REVERSE)
print(f"{'prompt':>12} | {'true':>7} | {'predicted':>9}")
for (prompt, target), predicted in zip(data_test[:12], sample_preds):
    print(f"{prompt:>12} | {target:>7} | {str(predicted):>9}")

test_digit, test_seq = evaluate(model, data_test, REVERSE)
print(f"\\nheld-out test: digit accuracy {test_digit:.4f} | sequence accuracy {test_seq:.4f}")
'''

# ---------------------------------------------------------------------- main_report cells

CELL_MR_LOAD = '''\
# The held-out set ships with the archive, and is also regenerated from SEED below so a reader
# can confirm the file was not hand-picked after the fact.
with open("project5_testset.pkl", "rb") as handle:
    data_test = pickle.load(handle)

data_train, data_val, regenerated_test = build_dataset(__TRAIN__, __VAL__, __TEST__)
train_prompts = {p for p, _ in data_train} | {p for p, _ in data_val}
overlap = sum(1 for p, _ in data_test if p in train_prompts)
assert data_test == regenerated_test, "the shipped test set differs from the one SEED produces"
assert overlap == 0, f"{overlap} test prompts also appear in train or validation"
assert len(data_test) >= __MIN_TEST__, "the brief asks for a held-out set of at least 10k samples"

n_sub = sum(1 for p, _ in data_test if "-" in p)
print(f"test examples {len(data_test)}  subtraction fraction {n_sub / len(data_test):.2f}")
print(f"regenerates from SEED exactly: True  |  train/test prompt overlap {overlap}")
'''

CELL_MR_MODELS = '''\
# Both checkpoints hold a plain state_dict rather than a pickled module, so the architecture
# is rebuilt here and the weights are loaded into it.
def load_model(path):
    """Rebuild the architecture and load a checkpoint into it.

    dropout is passed explicitly so the loaded model matches the one that was trained; it makes
    no numerical difference under eval() but leaving it to a default would be a discrepancy a
    reader has to chase. weights_only=True refuses to execute anything inside the file.
    """
    model = TransformerModel(ntoken=ntokens, ninp=128, nhead=16, nhid=64, nlayers=6, dropout=0.1)
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


forward_model = load_model("LLMForward.pth")
reverse_model = load_model("LLMReverse.pth")
print("loaded LLMForward.pth and LLMReverse.pth")
'''

CELL_MR_PREDICT = '''\
# Both models are run over the same list in the same order, and the prompts are compared
# afterwards. The paired test further down is only valid if the two really did see identical
# examples, so that condition is confirmed here before anything depends on it.
# reverse= is passed explicitly rather than read from a global, so each call states which
# convention it is decoding under.
prompts, trues, preds_f = predict_numeric(forward_model, data_test, reverse=False)
prompts_r, trues_r, preds_r = predict_numeric(reverse_model, data_test, reverse=True)
assert prompts == prompts_r and trues == trues_r, "the two models saw different examples"
print(f"predicted {len(prompts)} examples with each model")
'''

CELL_MR_STATS = '''\
# Accuracy over 10,000 examples still carries sampling error, and both models were scored on
# the same examples. These functions turn a comparison of two point estimates into a
# statement about whether the difference between them is real.
def wilson_interval(successes, n, z=1.96):
    """95% confidence interval for a proportion.

    Wilson rather than the textbook normal interval because these accuracies sit against the
    boundary at 1.0, where the normal interval runs past it and stops being meaningful.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_exact(preds_a, preds_b):
    """Two-sided exact McNemar test on paired predictions.

    Both models answered the same examples, so the comparison is paired and only the cases
    where they disagree carry information. b and c below are those two disagreement counts.
    """
    b = sum(1 for t, x, y in zip(trues, preds_a, preds_b) if x == t and y != t)
    c = sum(1 for t, x, y in zip(trues, preds_a, preds_b) if x != t and y == t)
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) * (0.5 ** n)
    return b, c, min(1.0, 2 * tail)


def is_correct(prediction, truth):
    return prediction is not None and prediction == truth
'''

CELL_MR_OVERALL = '''\
# Overall and per-operation accuracy, with the count behind each rate and a Wilson interval.
# The interval is what shows whether the margin above the target is larger than sampling error.
print(f"{'mode':>8} | {'split':>12} | {'n':>6} | {'correct':>7} | {'accuracy':>8} | {'95% CI':>16}")
# Keyed (mode, split) so any single cell of this table can be looked up by name below.
results_table = {}
for name, preds in [("Forward", preds_f), ("Reverse", preds_r)]:
    for split in ["overall", "addition", "subtraction"]:
        if split == "overall":
            index = range(len(trues))
        else:
            symbol = "+" if split == "addition" else "-"
            index = [i for i in range(len(trues)) if operation_of(prompts[i]) == symbol]
        correct = sum(1 for i in index if is_correct(preds[i], trues[i]))
        n = len(index)
        low, high = wilson_interval(correct, n)
        results_table[(name, split)] = (correct, n, correct / n, low, high)
        print(f"{name:>8} | {split:>12} | {n:>6} | {correct:>7} | {correct / n:>8.4f} | "
              f"[{low:.4f}, {high:.4f}]")

# The task sets __TARGET__ as the level both models have to clear. Asserting it makes the
# notebook stop if a re-run ever falls below it, instead of quietly reporting a lower number.
for name in ["Forward", "Reverse"]:
    assert results_table[(name, "overall")][2] > __TARGET__, f"{name} is below the target"
print(f"\\nboth models exceed the __TARGET__ target on the held-out set")
'''

CELL_MR_MCNEMAR = '''\
# Does direction change accuracy? The two models answered identical examples, so the paired
# test is the right one and it needs only the two disagreement counts.
b, c, p_value = mcnemar_exact(preds_f, preds_r)
print(f"Forward right, Reverse wrong : {b}")
print(f"Forward wrong, Reverse right : {c}")
print(f"both models agree on         : {len(trues) - b - c} of {len(trues)} examples")
print(f"exact McNemar two-sided p    : {p_value:.3e}")
'''

CELL_MR_POSITION = '''\
def position_accuracy(preds, index=None):
    """Per-place accuracy, counted only over answers that HAVE that place.

    Zero-padding every answer to four digits would score the thousands place as correct for
    every two-digit answer, which inflates the high positions towards 1.0 by construction. Each
    position is therefore measured only on the answers long enough to reach it. The sign is a
    separate question and is measured over everything.

    index restricts the measurement to a subset of the test set, which is how the split by
    operation below reuses this function without duplicating the counting logic.
    """
    labels = ["thousands", "hundreds", "tens", "units"]
    correct = {k: 0 for k in labels}
    total = {k: 0 for k in labels}
    sign_correct = 0
    rows = list(range(len(trues))) if index is None else list(index)
    for row in rows:
        truth, prediction = trues[row], preds[row]
        digits_true = str(abs(truth))
        digits_pred = str(abs(prediction)) if prediction is not None else ""
        width = len(digits_true)
        for offset, label in enumerate(labels[4 - width:]):
            total[label] += 1
            # Compare the same place value, counting from the units end of each string.
            place = width - offset - 1
            got = digits_pred[len(digits_pred) - place - 1] if len(digits_pred) > place else None
            correct[label] += (got == digits_true[offset])
        sign_correct += (prediction is not None and (prediction < 0) == (truth < 0))
    accuracy = {k: (correct[k] / total[k] if total[k] else float("nan")) for k in labels}
    accuracy["sign"] = sign_correct / len(rows) if rows else float("nan")
    total["sign"] = len(rows)
    return accuracy, total


# n differs per position: only answers long enough to have a thousands digit are counted
# towards the thousands row, which is why the counts fall as the place value rises.
acc_f, n_f = position_accuracy(preds_f)
acc_r, n_r = position_accuracy(preds_r)
PLACES = ["thousands", "hundreds", "tens", "units", "sign"]
print(f"{'position':>10} | {'n':>6} | {'Forward':>8} | {'Reverse':>8}")
for label in PLACES:
    print(f"{label:>10} | {n_f[label]:>6} | {acc_f[label]:>8.4f} | {acc_r[label]:>8.4f}")

# The same measurement split by operation, which the task asks for alongside the split by mode.
# Note what the thousands row does here: both operands are at most three digits, so a difference
# can never exceed 999 and no subtraction answer has a thousands digit at all. That cell is empty
# by construction, and printing n/a says so rather than implying a rate of zero.
add_index = [i for i in range(len(trues)) if operation_of(prompts[i]) == "+"]
sub_index = [i for i in range(len(trues)) if operation_of(prompts[i]) == "-"]
acc_fa, n_add = position_accuracy(preds_f, add_index)
acc_ra, _ = position_accuracy(preds_r, add_index)
acc_fs, n_sub = position_accuracy(preds_f, sub_index)
acc_rs, _ = position_accuracy(preds_r, sub_index)


def rate(value, n):
    """Format a rate, or n/a when the group it would describe is empty."""
    return f"{value:8.4f}" if n else f"{'n/a':>8}"


print()
print(f"{'position':>10} | {'n add':>6} | {'Fwd add':>8} | {'Rev add':>8} | "
      f"{'n sub':>6} | {'Fwd sub':>8} | {'Rev sub':>8}")
for label in PLACES:
    print(f"{label:>10} | {n_add[label]:>6} | {rate(acc_fa[label], n_add[label])} | "
          f"{rate(acc_ra[label], n_add[label])} | {n_sub[label]:>6} | "
          f"{rate(acc_fs[label], n_sub[label])} | {rate(acc_rs[label], n_sub[label])}")
'''

CELL_MR_CARRY = '''\
# The columns the brief calls hard. Errors are printed beside accuracies because at these
# rates the counts are the readable quantity: 0.996 and 1.000 look alike, 18 and 0 do not.
def carry_borrow_accuracy(preds):
    """Accuracy split by whether a column carries (addition) or borrows (subtraction)."""
    result = {}
    for symbol, flag_function in [("+", carry_positions), ("-", borrow_positions)]:
        name = "carry" if symbol == "+" else "borrow"
        buckets = {f"with_{name}": [0, 0], f"no_{name}": [0, 0]}
        for i, prompt in enumerate(prompts):
            if operation_of(prompt) != symbol:
                continue
            a_int, b_int = (int(x) for x in prompt[:-1].split(symbol))
            key = f"with_{name}" if sum(flag_function(a_int, b_int)) > 0 else f"no_{name}"
            buckets[key][1] += 1
            buckets[key][0] += is_correct(preds[i], trues[i])
        result[name] = buckets
    return result


cb_f, cb_r = carry_borrow_accuracy(preds_f), carry_borrow_accuracy(preds_r)
header = f"{'case':>14} | {'n':>5} | {'Fwd err':>7} | {'Fwd acc':>8}"
print(header + f" | {'Rev err':>7} | {'Rev acc':>8}")
for group in ["carry", "borrow"]:
    for key in cb_f[group]:
        (fc, fn), (rc, rn) = cb_f[group][key], cb_r[group][key]
        print(f"{key:>14} | {fn:>5} | {fn - fc:>7} | {fc / fn:>8.4f} | "
              f"{rn - rc:>7} | {rc / rn:>8.4f}")
'''

CELL_MR_LENGTH = '''\
# The brief asks which cases fail, and result length is the other axis the errors could follow.
# Negative answers are separated out because they are the only ones carrying a sign token.
print(f"{'group':>16} | {'n':>6} | {'Fwd err':>8} | {'Rev err':>8}")
length_rows = {}
for width in [1, 2, 3, 4]:
    index = [i for i in range(len(trues)) if len(str(abs(trues[i]))) == width]
    if not index:
        continue
    fe = sum(1 for i in index if not is_correct(preds_f[i], trues[i]))
    re_ = sum(1 for i in index if not is_correct(preds_r[i], trues[i]))
    length_rows[f"{width} digit"] = (len(index), fe, re_)
    print(f"{str(width) + ' digit':>16} | {len(index):>6} | {fe:>8} | {re_:>8}")
for label, test in [("negative", lambda t: t < 0), ("non-negative", lambda t: t >= 0)]:
    index = [i for i in range(len(trues)) if test(trues[i])]
    fe = sum(1 for i in index if not is_correct(preds_f[i], trues[i]))
    re_ = sum(1 for i in index if not is_correct(preds_r[i], trues[i]))
    length_rows[label] = (len(index), fe, re_)
    print(f"{label:>16} | {len(index):>6} | {fe:>8} | {re_:>8}")
'''

CELL_MR_ALIGNMENT = '''# The single most informative cut of the errors. Each direction is hypothesised to fail at what
# its own ordering makes hard: left-to-right has to fix the answer's LENGTH before emitting any
# digit, while right-to-left has to notice when the shorter operand runs out of digits. Operand
# width difference tests the second hypothesis and answer length tests the first.
def width_difference(prompt):
    symbol = operation_of(prompt)
    left, right = (int(x) for x in prompt[:-1].split(symbol))
    return len(str(left)) - len(str(right))


print(f"{'len(a)-len(b)':>14} | {'n':>6} | {'Fwd err':>8} | {'Fwd rate':>9} | "
      f"{'Rev err':>8} | {'Rev rate':>9}")
alignment_rows = {}
for difference in sorted({width_difference(p) for p in prompts}):
    index = [i for i in range(len(trues)) if width_difference(prompts[i]) == difference]
    fe = sum(1 for i in index if not is_correct(preds_f[i], trues[i]))
    re_ = sum(1 for i in index if not is_correct(preds_r[i], trues[i]))
    alignment_rows[difference] = (len(index), fe, re_)
    print(f"{difference:>14} | {len(index):>6} | {fe:>8} | {fe / len(index):>9.4f} | "
          f"{re_:>8} | {re_ / len(index):>9.4f}")

# Which place value is actually wrong, counted from the units end. In reverse mode the units
# digit is emitted first, so a failure at place 1 means the model lost track one step in.
place_errors = {"Forward": {}, "Reverse": {}}
for name, preds in [("Forward", preds_f), ("Reverse", preds_r)]:
    for i in range(len(trues)):
        if is_correct(preds[i], trues[i]):
            continue
        if preds[i] is None:
            place_errors[name]["unparseable"] = place_errors[name].get("unparseable", 0) + 1
            continue
        true_digits, pred_digits = str(abs(trues[i])), str(abs(preds[i]))
        if len(true_digits) != len(pred_digits):
            place_errors[name]["wrong length"] = place_errors[name].get("wrong length", 0) + 1
            continue
        for k, (x, y) in enumerate(zip(true_digits, pred_digits)):
            if x != y:
                key = f"place {len(true_digits) - k - 1}"
                place_errors[name][key] = place_errors[name].get(key, 0) + 1
print()
for name in ["Forward", "Reverse"]:
    print(f"{name:>8} wrong places (0 = units): "
          f"{dict(sorted(place_errors[name].items()))}")

# How far off the wrong answers are. An error of exactly one unit is a different failure from
# an answer with the wrong number of digits.
for name, preds in [("Forward", preds_f), ("Reverse", preds_r)]:
    wrong = [i for i in range(len(trues)) if not is_correct(preds[i], trues[i])]
    off_by_one = sum(1 for i in wrong if preds[i] is not None and abs(preds[i] - trues[i]) == 1)
    print(f"{name:>8} errors off by exactly 1: {off_by_one} of {len(wrong)}")
'''

CELL_MR_ERRORS = '''\
def describe_error(index, prediction):
    """Classify one wrong answer so the failures can be counted by kind, not just listed."""
    truth = trues[index]
    if prediction is None:
        return "unparseable"
    if (prediction < 0) != (truth < 0):
        return "wrong sign"
    if len(str(abs(prediction))) != len(str(abs(truth))):
        return "wrong length"
    differing = sum(1 for x, y in zip(str(abs(truth)), str(abs(prediction))) if x != y)
    return "single digit slip" if differing == 1 else f"{differing} digit slips"


# Concrete mispredictions, not only rates: this is what the error patterns section of the
# report is written from, and it is the part a reader can check by hand.
print(f"{'mode':>8} | {'prompt':>12} | {'true':>7} | {'predicted':>10} | {'kind':>18}")
error_kinds = {"Forward": {}, "Reverse": {}}
for name, preds in [("Forward", preds_f), ("Reverse", preds_r)]:
    wrong = [i for i in range(len(trues)) if not is_correct(preds[i], trues[i])]
    for i in wrong:
        kind = describe_error(i, preds[i])
        error_kinds[name][kind] = error_kinds[name].get(kind, 0) + 1
    # Both models make few enough errors to print in full, and printing all of them is what
    # lets the report's table of example failures be checked against the complete list.
    for i in wrong:
        print(f"{name:>8} | {prompts[i]:>12} | {trues[i]:>7} | {str(preds[i]):>10} | "
              f"{describe_error(i, preds[i]):>18}")
    print(f"{name}: {len(wrong)} errors in {len(trues)} examples")

print()
for name in ["Forward", "Reverse"]:
    for kind, count in sorted(error_kinds[name].items(), key=lambda x: -x[1]):
        print(f"{name:>8} error kind {kind:>18} : {count}")
'''

CELL_MR_CURVES = '''\
# The learning curves come from the JSON the training notebooks wrote, which is how this
# notebook reports on training without containing a training loop.
# Four runs: two directions at two training set sizes.
histories = {}
for mode in ["Forward", "Reverse"]:
    for size in ["", "_80k"]:
        path = f"LLM{mode}_history{size}.json"
        with open(path) as handle:
            histories[(mode, size or "_full")] = json.load(handle)


def first_epoch_reaching(curve, level):
    """1-based epoch at which the curve first reaches `level`, or None if it never does."""
    for epoch, value in enumerate(curve, start=1):
        if value >= level:
            return epoch
    return None


# Epochs-to-threshold is the learnability measure; the final column is the accuracy the
# run settled at, so a slow start and a poor finish can be told apart.
print(f"{'mode':>8} | {'train size':>10} | {'ep>=tgt':>9} | {'ep>=0.95':>9} | {'final':>7}")
curve_rows = {}
for (mode, size), history in histories.items():
    curve = history["val_seq"]
    row = (first_epoch_reaching(curve, __TARGET__), first_epoch_reaching(curve, 0.95), curve[-1])
    curve_rows[(mode, size)] = row
    label = "__TRAIN__" if size == "_full" else "__ABLATION__"
    print(f"{mode:>8} | {label:>10} | {str(row[0]):>9} | {str(row[1]):>9} | {row[2]:>7.4f}")

# An epoch is not a fixed amount of training: a smaller training set is fewer batches, so the
# same number of epochs is a shorter run. The training notebooks used a batch size of __BATCH__,
# which makes the full set __TRAIN__ // __BATCH__ updates per epoch and the reduced set
# __ABLATION__ // __BATCH__. Converting epochs to updates is what separates a run that needed
# more training from one that only needed more passes over less data.
train_batch_size = __BATCH__
batches_per_epoch = {"_full": __TRAIN__ // train_batch_size,
                     "_80k": __ABLATION__ // train_batch_size}
for (mode, size), row in curve_rows.items():
    epoch_at_95 = row[1]
    steps = epoch_at_95 * batches_per_epoch[size] if epoch_at_95 is not None else None
    label = "__TRAIN__" if size == "_full" else "__ABLATION__"
    print(f"steps to 0.95 | {mode:>8} | {label:>7} | {str(steps):>8}")
'''

CELL_MR_FIGURE1 = '''\
__PALETTE__

# Figure 1 for the report. Counts rather than accuracies in the right two panels: every accuracy
# here is above 0.98, so a bar chart of rates renders the two models as identical rectangles,
# while the error counts differ by an order of magnitude and are legible.
# figsize sets how big the labels end up. The report places this at 6.95 in across both
# columns, so every size below is multiplied by roughly 0.67 on the page: 10 pt here renders
# near 6.7 pt, which is the smallest that stays comfortable beside 10 pt body text. The height
# is kept tight because this figure spans both columns, so every inch of it costs two inches
# of text.
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6), dpi=150)

# Left panel: convergence. Solid is the full training set, dashed the reduced one.
ax = axes[0]
for mode, colour in [("Forward", ACCENT), ("Reverse", WARM)]:
    for size, style in [("_full", "-"), ("_80k", "--")]:
        curve = histories[(mode, size)]["val_seq"]
        # Only the full-set curves are labelled. Colour is the direction and the line style is
        # the training set size, so naming all four would state the colour key twice and make
        # the box wider than the empty part of the panel, which put it over the curves.
        ax.plot(range(1, len(curve) + 1), curve, style, color=colour, lw=1.6,
                label=mode if size == "_full" else None)
ax.axhline(__TARGET__, color=GREEN, lw=1.0, ls="-.")
ax.set_xlabel("epoch")
ax.set_ylabel("validation sequence accuracy")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)
# Centre right is the only part of this panel no curve passes through: everything has either
# reached 1.0 by then or, for the reduced reverse run, stayed near 0.1 throughout.
ax.legend(fontsize=10, loc="center right", labelspacing=0.25, handlelength=1.6,
          borderpad=0.3, framealpha=0.95)
ax.set_title("Convergence by direction and data size", fontsize=11, color=INK)

ax = axes[1]
# Two line tick labels: at the width this figure is embedded, the carry and borrow
# names collide when set on one line.
categories = [("add\\nno carry", "carry", "no_carry"),
              ("add\\ncarry", "carry", "with_carry"),
              ("sub\\nno borrow", "borrow", "no_borrow"),
              ("sub\\nborrow", "borrow", "with_borrow")]
x = np.arange(len(categories))
errors_f = [cb_f[g][k][1] - cb_f[g][k][0] for _, g, k in categories]
errors_r = [cb_r[g][k][1] - cb_r[g][k][0] for _, g, k in categories]
ax.bar(x - 0.2, errors_f, 0.4, color=ACCENT, label="Forward")
ax.bar(x + 0.2, errors_r, 0.4, color=WARM, label="Reverse")
for xi, value in zip(x - 0.2, errors_f):
    ax.text(xi, value, str(value), ha="center", va="bottom", fontsize=10, color=INK)
for xi, value in zip(x + 0.2, errors_r):
    ax.text(xi, value, str(value), ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in categories], fontsize=10)
ax.set_ylabel("errors")
ax.set_ylim(0, max(errors_f + errors_r) * 1.25)
ax.grid(alpha=0.25, axis="y")
ax.legend(fontsize=10, labelspacing=0.25, handlelength=1.4, borderpad=0.3)
ax.set_title("Where the errors fall", fontsize=11, color=INK)

ax = axes[2]
labels = ["thousands", "hundreds", "tens", "units", "sign"]
# Abbreviated on the axis only. Set large enough to read once the figure is scaled into the
# report, the five full words overlap; the keys above stay spelled out.
tick_labels = ["1000s", "100s", "10s", "1s", "sign"]
wrong_f = [round((1 - acc_f[k]) * n_f[k]) for k in labels]
wrong_r = [round((1 - acc_r[k]) * n_r[k]) for k in labels]
x = np.arange(len(labels))
ax.bar(x - 0.2, wrong_f, 0.4, color=ACCENT, label="Forward")
ax.bar(x + 0.2, wrong_r, 0.4, color=WARM, label="Reverse")
for xi, value in zip(x - 0.2, wrong_f):
    ax.text(xi, value, str(value), ha="center", va="bottom", fontsize=10, color=INK)
for xi, value in zip(x + 0.2, wrong_r):
    ax.text(xi, value, str(value), ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=10)
ax.set_ylabel("wrong digits")
ax.set_ylim(0, max(wrong_f + wrong_r) * 1.25)
ax.grid(alpha=0.25, axis="y")
ax.legend(fontsize=10, labelspacing=0.25, handlelength=1.4, borderpad=0.3)
ax.set_title("Wrong digits by place value", fontsize=11, color=INK)

plt.tight_layout()
plt.savefig("figure_1_overview.pdf", bbox_inches="tight")
plt.show()
'''

CELL_MR_FIGURE2 = '''\
# Figure 2: the same errors cut by how long the answer is and whether it is negative, which is
# the axis the brief calls "longer strings".
# Placed in a single column at 3.47 in against a native 3.8, so the scale here is about 0.91
# and 7 pt renders near 6.4 pt. The height is trimmed to what two small bar panels need.
fig, axes = plt.subplots(2, 1, figsize=(3.9, 3.0), dpi=150)

# Left panel: how long the answer is. Right panel: whether it is negative, which is the
# only case carrying a sign token and the only one reverse mode emits last.
digit_rows = [k for k in length_rows if k.endswith("digit")]
x = np.arange(len(digit_rows))
axes[0].bar(x - 0.2, [length_rows[k][1] for k in digit_rows], 0.4, color=ACCENT, label="Forward")
axes[0].bar(x + 0.2, [length_rows[k][2] for k in digit_rows], 0.4, color=WARM, label="Reverse")
for xi, k in zip(x, digit_rows):
    axes[0].text(xi - 0.2, length_rows[k][1], str(length_rows[k][1]), ha="center", va="bottom",
                 fontsize=7, color=INK)
    axes[0].text(xi + 0.2, length_rows[k][2], str(length_rows[k][2]), ha="center", va="bottom",
                 fontsize=7, color=INK)
# The count above the tallest bar is drawn upwards from it, so the axis needs headroom of
# its own or that label lands on the frame.
axes[0].set_ylim(0, max(max(length_rows[k][1], length_rows[k][2]) for k in digit_rows) * 1.2)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{k}\\n(n={length_rows[k][0]:,})" for k in digit_rows], fontsize=7)
axes[0].set_ylabel("errors")
axes[0].grid(alpha=0.25, axis="y")
axes[0].legend(fontsize=7)
axes[0].set_title("Errors by answer length", fontsize=9, color=INK)

# Counts are labelled on every bar so the figure carries its own numbers.
sign_rows = ["non-negative", "negative"]
x = np.arange(len(sign_rows))
axes[1].bar(x - 0.2, [length_rows[k][1] for k in sign_rows], 0.4, color=ACCENT, label="Forward")
axes[1].bar(x + 0.2, [length_rows[k][2] for k in sign_rows], 0.4, color=WARM, label="Reverse")
for xi, k in zip(x, sign_rows):
    axes[1].text(xi - 0.2, length_rows[k][1], str(length_rows[k][1]), ha="center", va="bottom",
                 fontsize=7, color=INK)
    axes[1].text(xi + 0.2, length_rows[k][2], str(length_rows[k][2]), ha="center", va="bottom",
                 fontsize=7, color=INK)
axes[1].set_ylim(0, max(max(length_rows[k][1], length_rows[k][2]) for k in sign_rows) * 1.2)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{k}\\n(n={length_rows[k][0]:,})" for k in sign_rows], fontsize=7)
axes[1].grid(alpha=0.25, axis="y")
axes[1].legend(fontsize=7)
axes[1].set_title("Errors by sign of the answer", fontsize=9, color=INK)

plt.tight_layout()
plt.savefig("figure_2_by_length.pdf", bbox_inches="tight")
plt.show()
'''

CELL_MR_SUMMARY = '''\
# One machine-readable dump of everything the report quotes, keyed by name, so every number
# in the report can be traced back to the cell that produced it.
summary = {
    "overall": {name: {"correct": results_table[(name, "overall")][0],
                       "n": results_table[(name, "overall")][1],
                       "accuracy": results_table[(name, "overall")][2],
                       "ci": [results_table[(name, "overall")][3],
                              results_table[(name, "overall")][4]]}
                for name in ["Forward", "Reverse"]},
    "operation": {name: {split: results_table[(name, split)][2]
                         for split in ["addition", "subtraction"]}
                  for name in ["Forward", "Reverse"]},
    "positions": {"Forward": acc_f, "Reverse": acc_r, "n": n_f},
    "carry_borrow": {"Forward": cb_f, "Reverse": cb_r},
    "length": length_rows,
    "mcnemar": {"forward_only": b, "reverse_only": c, "p": p_value},
    "error_kinds": error_kinds,
    "curves": {f"{mode}{size}": curve_rows[(mode, size)] for mode, size in curve_rows},
}
print(json.dumps(summary, indent=1, default=lambda o: round(float(o), 6)))
'''

# --------------------------------------------------------------------------- notebook assembly


def as_source(text):
    """nbformat stores source as a list of lines that KEEP their newline characters.

    Splitting on "\n" and dropping them makes Jupyter concatenate the whole cell onto one
    physical line, which is a syntax error for anything with an indented block and is invisible
    to every check that runs on the text before this point.
    """
    lines = text.rstrip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def markdown(cell_id, source):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": as_source(source)}


def code(cell_id, source):
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {},
            "outputs": [], "source": as_source(source)}


def fill(text, **extra):
    # os and time are used by the training loop and by nothing in main_report, so the import
    # list is built per notebook instead of carrying a name one of the three never calls.
    values = {"__IMPORT_OS__": "import os\n", "__IMPORT_TIME__": "import time\n",
              "__TRAIN__": f"{TRAIN}", "__VAL__": f"{VAL}", "__TEST__": f"{TEST}",
              "__ABLATION__": f"{ABLATION}", "__EPOCHS__": f"{EPOCHS}",
              "__MIN_TEST__": f"{min(TEST, 10000)}",
              "__TRAIN_PROBE__": f"{min(5000, TRAIN)}",
              "__BATCH__": "100",
              "__TARGET__": "-1" if SMOKE else "0.78",
              "__PALETTE__": PALETTE}
    values.update(extra)
    for key, value in values.items():
        text = text.replace(key, value)
    return text


HEADER = """# IFN680 Project 5 - __TITLE__

**Group 4** - Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

__INTRO__
"""

SHARED = [
    ("md-step0", "## Setup"),
    ("code-imports", CELL_IMPORTS),
    ("md-step1", "## Step 1: Tokenizer"),
    ("code-tokenizer", CELL_TOKENIZER),
    ("md-step2", "## Step 2: Data for addition and subtraction\n\nOperands are non-negative and "
                 "at most three digits; a subtraction may be negative when `b > a`. Prompts of "
                 "the form `-2+3` are out of scope, so no operand ever carries a sign."),
    ("code-data", CELL_DATA),
    ("md-step2b", "### Reverse-order transform, parsing, and carry or borrow structure"),
    ("code-helpers", CELL_REVERSE_HELPERS),
    ("md-step3", "## Step 3: Positional encoding\n\nSelf-attention sees a set of tokens, "
                 "not a sequence, so position has to be added to the embedding before the "
                 "model can tell `12+3` from `21+3`. The fixed sinusoidal encoding of the "
                 "workshop is kept unchanged. Note what it encodes: where a digit sits in the "
                 "string, not what place value it holds. Those two coincide for the forward "
                 "ordering and not for the reverse one."),
    ("code-posenc", CELL_POSENC),
    ("md-step4", "## Step 4: Transformer model\n\nA causal decoder. Each position may "
                 "attend only to positions at or before it, which is what makes it safe to "
                 "train on every position of a sequence at once: no position can reach the "
                 "answer it is being asked to predict."),
    ("code-model", CELL_MODEL),
    ("md-step5", "## Step 5: Generation, padding and batching\n\nPrompts differ in length, "
                 "so they are padded to a common width before being stacked into a batch. "
                 "Decoding takes the arg max at every step instead of sampling, which makes a "
                 "prediction a function of the weights alone and the whole evaluation "
                 "repeatable."),
    ("code-generation", CELL_GENERATION),
    ("md-step6", "## Step 6: Accuracy metrics\n\nTwo levels, because they answer different "
                 "questions. Digit accuracy counts individual characters and shows how close a "
                 "wrong answer was; sequence accuracy demands the entire answer and is the one "
                 "the task is scored on. Sequence accuracy is always the lower of the two, and "
                 "the gap between them is a measure of how near the misses are."),
    ("code-evaluate", CELL_EVALUATE),
]


def build_training(mode):
    reverse = mode == "Reverse"
    direction = "right to left, units digit first" if reverse else "left to right"
    intro = (f"This notebook extends the Week 7 addition model to **subtraction**, including "
             f"negative results, and trains it in **{mode} mode**: the answer is predicted "
             f"{direction}.\n\nThe tokeniser, architecture, causal next-token objective and "
             f"training loop follow the Week 7 workshop. `REVERSE` below is the only switch "
             f"that separates this notebook from its counterpart, which is what makes output "
             f"direction the single variable in the comparison.")
    cells = [markdown("md-title", fill(HEADER, __TITLE__=f"LLM{mode}", __INTRO__=intro))]
    cells.append(code("code-mode", "# The one line that differs between the two training\n"
                                   "# notebooks. It selects the target digit order, and\n"
                                   "# nothing else in either file depends on the mode.\n"
                                   f"REVERSE = {reverse}"))
    for cell_id, source in SHARED:
        cells.append(markdown(cell_id, source) if cell_id.startswith("md-")
                     else code(cell_id, fill(source)))
    cells += [
        markdown("md-dataset", "## Build the split, and check it is disjoint"),
        code("code-dataset", fill(CELL_TRAIN_DATA)),
        markdown("md-step7", f"## Step 7: Train the {mode} model"),
        code("code-setup", fill(CELL_TRAIN_SETUP)),
        code("code-loop", fill(CELL_TRAIN_LOOP)),
        code("code-save", fill(CELL_TRAIN_SAVE, __CKPT__=f"LLM{mode}.pth",
                               __HIST__=f"LLM{mode}_history.json")),
        markdown("md-ablation", "### How much data does this direction need?\n\nFinal accuracy "
                                "and ease of training are different questions. The run below "
                                "repeats the training above on a smaller slice of the same data, "
                                "so the two curves can be compared in `main_report.ipynb`."),
        code("code-ablation", fill(CELL_ABLATION, __HIST_SMALL__=f"LLM{mode}_history_80k.json")),
        markdown("md-curves", "### Learning curves"),
        code("code-curves", fill(CELL_TRAIN_CURVE, __MODE__=mode)),
        markdown("md-sanity", "### Worked examples and the held-out score"),
        code("code-sanity", fill(CELL_TRAIN_SANITY)),
    ]
    return cells


def build_main_report():
    intro = ("This notebook reproduces **every number and figure the report carries** for Task 3. "
             "It does **not** train: it loads the two checkpoints, the shared held-out test set "
             "and the training histories, then compares the Forward and Reverse models across "
             "operations, digit positions, carry and borrow cases, answer length and error kind."
             "\n\nRun it top to bottom. It takes about a minute on CPU.")
    cells = [markdown("md-title", fill(HEADER, __TITLE__="main_report", __INTRO__=intro))]
    for cell_id, source in SHARED:
        cells.append(markdown(cell_id, source) if cell_id.startswith("md-")
                     else code(cell_id, fill(source, __IMPORT_OS__="", __IMPORT_TIME__="")))
    cells += [
        markdown("md-load", "## Load the shared test set, and verify it\n\nThe test set is "
                            "regenerated from `SEED` and compared against the file that ships "
                            "in the archive, so both its contents and its disjointness from "
                            "training are confirmed before any accuracy is computed."),
        code("code-load", fill(CELL_MR_LOAD)),
        markdown("md-models", "## Load both trained models"),
        code("code-models", fill(CELL_MR_MODELS)),
        markdown("md-predict", "## Predict with both models on the same examples"),
        code("code-predict", fill(CELL_MR_PREDICT)),
        markdown("md-stats", "## Statistical helpers\n\nTwo tools, for two different "
                             "questions. A Wilson interval puts a range around a single "
                             "accuracy; it is used instead of the textbook normal interval "
                             "because these rates sit hard against the boundary at 1.0, where "
                             "the normal interval runs past it and stops meaning anything. "
                             "McNemar's exact test compares the two models, and it is the "
                             "correct test here because both answered the same examples, so "
                             "only the cases where they disagree carry information."),
        code("code-stats", fill(CELL_MR_STATS)),
        markdown("md-overall", "## Operation robustness: overall and per-operation accuracy\n\nThe headline "
                               "number for each model, then the same number split by "
                               "operation. The split matters because subtraction is the "
                               "harder half: it borrows, and it can produce a negative "
                               "answer, so a model that looks strong overall can still be "
                               "carrying most of its errors on one side."),
        code("code-overall", fill(CELL_MR_OVERALL)),
        markdown("md-mcnemar", "## Direction effects: a paired test\n\nComparing two "
                               "accuracies measured on the same examples is not the same as "
                               "comparing two independent samples. The pairing removes the "
                               "variation caused by which examples happened to be drawn, "
                               "leaving only the cases the two models answer differently."),
        code("code-mcnemar", fill(CELL_MR_MCNEMAR)),
        markdown("md-position", "## Digit-level performance\n\nA sequence is either right "
                                "or wrong, which hides how close a wrong answer came. Scoring "
                                "each place value separately shows where a model loses a "
                                "digit. Each place is counted only over answers long enough "
                                "to have it: zero-padding every answer to four digits would "
                                "mark the thousands place correct for every two-digit answer "
                                "and drive the high positions to 1.0 by construction."),
        code("code-position", fill(CELL_MR_POSITION)),
        markdown("md-carry", "## Carry and borrow cases\n\nA carry or a borrow is a "
                             "column whose result depends on the column beside it, which is "
                             "exactly the dependency the two orderings treat differently. "
                             "Forward emits the most significant digit first, before the "
                             "carries underneath it have been worked out. Reverse emits the "
                             "units digit first, travelling in the same direction a carry "
                             "does by hand."),
        code("code-carry", fill(CELL_MR_CARRY)),
        markdown("md-length", "## Errors by answer length and sign\n\nAnswer length "
                              "stands in for how much a model has to commit to before it "
                              "writes anything: predicting left to right means choosing how "
                              "many digits the answer has first. The sign is the mirror case, "
                              "since the reverse ordering places it last, after the entire "
                              "magnitude has been emitted."),
        code("code-length", fill(CELL_MR_LENGTH)),
        markdown("md-alignment", "## Where each direction actually breaks\n\nThe two orderings "
                                 "make different things hard. Left to right has to settle the "
                                 "answer's length before emitting a digit; right to left has to "
                                 "keep track of the columns once the shorter operand has run "
                                 "out. The two cuts below test exactly those two predictions."),
        code("code-alignment", fill(CELL_MR_ALIGNMENT)),
        markdown("md-errors", "## Error patterns: every error, and what kind it is\n\nBoth models make "
                              "few enough mistakes to list in full. Each is classified by "
                              "what went wrong, so the failures can be counted by kind as "
                              "well as read individually, and so a claim about them can be "
                              "checked against the complete list rather than a sample."),
        code("code-errors", fill(CELL_MR_ERRORS)),
        markdown("md-curves", "## Learning curves from the training runs\n\nThe "
                              "histories saved by the two training notebooks, reloaded here "
                              "so that every figure the report carries is produced by this "
                              "one notebook. Each direction was trained twice, on the full "
                              "set and on a reduced one, which is what separates the effect "
                              "of data volume from the effect of ordering."),
        code("code-curves", fill(CELL_MR_CURVES)),
        markdown("md-fig1", "## Figure 1 for the report\n\nThree panels: convergence, "
                            "errors by carry and borrow case, and wrong digits by place "
                            "value. The right two panels plot counts instead of rates, "
                            "because every accuracy here is above 0.98 and bars of "
                            "near-identical rates would look identical."),
        code("code-fig1", fill(CELL_MR_FIGURE1)),
        markdown("md-fig2", "## Figure 2 for the report\n\nThe same errors cut by the "
                            "shape of the answer rather than by the operation: how many "
                            "digits it has, and whether it is negative."),
        code("code-fig2", fill(CELL_MR_FIGURE2)),
        markdown("md-summary", "## Machine-readable summary\n\nEvery number the report "
                               "quotes, dumped as JSON in one place. Collecting them here "
                               "means each value in the report traces back to the cell "
                               "that produced it, rather than being copied across by hand."),
        code("code-summary", fill(CELL_MR_SUMMARY)),
    ]
    return cells


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
    with io.open(path, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    return path, notebook


# --------------------------------------------------------------------------- generator rails

FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


built = {}
for name, cells in [("LLMForward.ipynb", build_training("Forward")),
                    ("LLMReverse.ipynb", build_training("Reverse")),
                    ("main_report.ipynb", build_main_report())]:
    path, notebook = write(name, cells)
    built[name] = notebook
    print(f"wrote {path}  ({len(cells)} cells)")

print("\ngenerator rails")
for name, notebook in built.items():
    raw = json.dumps(notebook, ensure_ascii=False)
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    source = "\n".join("".join(c["source"]) for c in code_cells)
    lines = [line for line in source.split("\n") if line.strip()]
    comments = [line for line in lines if line.strip().startswith("#")]
    ratio = len(comments) / len(lines)
    ids = [c["id"] for c in notebook["cells"]]

    check(f"{name}: no em or en dash", "\u2014" not in raw and "\u2013" not in raw)
    check(f"{name}: no control characters in any source",
          not any(ord(ch) < 32 and ch not in "\n\t" for ch in source))
    check(f"{name}: every cell has a unique id", len(ids) == len(set(ids)), f"{len(ids)} cells")
    check(f"{name}: every code cell carries a comment",
          all(any(line.strip().startswith("#") for line in c["source"]) for c in code_cells))
    # nbformat keeps the newline on every source line except the last. Dropping them silently
    # concatenates a cell onto one physical line, which is a syntax error for any indented
    # block and which every text-level check above still passes.
    check(f"{name}: source lines keep their newlines",
          all(all(line.endswith("\n") for line in c["source"][:-1])
              for c in notebook["cells"] if len(c["source"]) > 1))
    check(f"{name}: comment ratio at least 0.12", ratio >= 0.12,
          f"{len(comments)}/{len(lines)} = {ratio:.0%}")
    check(f"{name}: no line longer than 100 characters",
          all(len(line) <= 100 for line in source.split("\n")),
          f"longest {max(len(line) for line in source.split('\n'))}")
    if name == "main_report.ipynb":
        check(f"{name}: contains no training loop",
              not re.search(r"\.backward\(\)|optimizer\.step\(\)|model\.train\(\)", source))
    else:
        check(f"{name}: writes its history json", "_history.json" in source)
        check(f"{name}: contains the reduced-data ablation", "model_small" in source)
    check(f"{name}: no tqdm", "tqdm" not in source)
    check(f"{name}: disables nested tensor", "enable_nested_tensor=False" in source)

print("\n" + ("GENERATOR CHECKS PASSED" if not FAILURES
             else f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES)))
raise SystemExit(1 if FAILURES else 0)
