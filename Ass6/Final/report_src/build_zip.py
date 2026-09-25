r"""Assemble submission/project6_code.zip and submission/project6_report.pdf.

The brief names two notebooks the archive must include, cVAE_DiscriminatorLoss.ipynb and
main_report.ipynb, plus "any auxiliary files required", such as model weights and pickled losses.
The baseline and the evaluation classifier are trained in notebooks of their own (README, D10), so
their notebooks and checkpoints ship too, and main_report.ipynb loads all three checkpoints.

main_report.ipynb needs only the test split, so the archive carries that split alone rather than
the 112 MB mnist_custom.pt it was cut from.

The archive is flat. The Project 4 submission that scored 2/2 was nested inside a folder with
.DS_Store and __MACOSX entries and lost nothing for it, so this is a tidiness choice rather than
a rule, but a flat archive is also what makes the bare relative paths in the notebooks resolve.

Run:  python tools/build_zip.py
"""
import os
import shutil
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-09/project-6-sharpness-quest")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
SUBMISSION = f"{BASE}/submission"

# The unit states that a wrong file name scores 0, so both names are written out here rather
# than derived, and verify.py asserts them independently.
ARCHIVE = "project6_code.zip"
REPORT_PDF = "project6_report.pdf"

CONTENTS = [
    "cVAE_Baseline.ipynb", "cVAE_DiscriminatorLoss.ipynb", "DigitClassifier.ipynb",
    "main_report.ipynb",
    "cVAE_Baseline.pth", "cVAE_DiscriminatorLoss.pth", "DigitClassifier.pth",
    "cVAE_Baseline_history.json", "cVAE_DiscriminatorLoss_history.json",
    "mnist_custom_test.pt", "figure_1_grids.pdf", "figure_2_reconstructions.pdf",
]

README = """IFN680 Project 6 - The Sharpness Quest
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  cVAE_Baseline.ipynb            Task 1: trains the standard cVAE of Tutorial 8.3, 7 latent
                                 dimensions, adapted to the 20x20 images of mnist_custom.pt
  cVAE_DiscriminatorLoss.ipynb   Task 2: the same cVAE trained with an added discriminator loss,
                                 including the choice of the loss weight lambda
  DigitClassifier.ipynb          trains the standalone digit classifier used for evaluation
  main_report.ipynb              Task 3: loads the three checkpoints and reproduces every number
                                 and figure in the report, including the 12x10 grid of generated
                                 digits for both models. Contains no training loop.
  *.pth                          trained weights, plain state_dicts: one cVAE per training seed
                                 (0, 1, 2) in each cVAE file, plus each seed's discriminator
  *_history.json                 per-epoch training curves, the lambda sweep and the settings
  mnist_custom_test.pt           the 10,000-image test split of mnist_custom.pt, the only data
                                 main_report.ipynb reads
  figure_1_grids.pdf             the report's figures, as main_report.ipynb writes them
  figure_2_reconstructions.pdf

HOW TO RUN
  Unzip everything into one folder and run main_report.ipynb top to bottom. It needs no
  arguments and no data beyond what is in this archive, runs on a GPU or a CPU, and takes a
  few minutes on a CPU. Every random draw is seeded, so it prints the same numbers each time;
  on a different device the last printed digit can differ by float rounding.

  The stored outputs of all four notebooks come from the IFN680 GPU environment. The training
  notebooks need mnist_custom.pt, downloaded from this project's Canvas page into the same folder.
"""

os.makedirs(SUBMISSION, exist_ok=True)

for name in CONTENTS:
    assert os.path.exists(f"{NBDIR}/{name}"), \
        f"missing {name}; execute the notebooks before zipping"
assert os.path.exists(f"{REPORT}/{REPORT_PDF}"), "compile the report first"

shutil.copyfile(f"{REPORT}/{REPORT_PDF}", f"{SUBMISSION}/{REPORT_PDF}")

path = f"{SUBMISSION}/{ARCHIVE}"
if os.path.exists(path):
    os.remove(path)

with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in CONTENTS:
        archive.write(f"{NBDIR}/{name}", arcname=name)
    archive.writestr("README.txt", README)

with zipfile.ZipFile(path) as archive:
    names = archive.namelist()
    for required in ["cVAE_DiscriminatorLoss.ipynb", "main_report.ipynb"]:
        assert required in names, f"{required} must be present or the code scores 0"
    assert not any(".DS_Store" in n or ".ipynb_checkpoints" in n for n in names)
    total = sum(info.file_size for info in archive.infolist())

print(f"wrote {path}")
for info in sorted(zipfile.ZipFile(path).infolist(), key=lambda i: -i.file_size):
    print(f"  {info.file_size:>10,}  {info.filename}")
print(f"  {'-' * 10}")
print(f"  {total:>10,}  uncompressed   ({os.path.getsize(path):,} bytes on disk)")
print(f"wrote {SUBMISSION}/{REPORT_PDF}")
