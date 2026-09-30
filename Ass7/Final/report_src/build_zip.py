r"""Assemble submission/project7_code.zip and submission/project7_report.pdf.

The brief names two notebooks the archive must include, DDPM_CosineSchedule.ipynb and
main_report.ipynb, plus "any auxiliary files required for this process, such as model checkpoints,
pickle files with losses, or other necessary data". The evaluation classifier is trained in a
notebook of its own (README, D6), so its notebook and checkpoint ship too.

main_report.ipynb needs only the test split, so the archive carries that split alone rather than
the 112 MB mnist_custom.pt it was cut from (README, D13).

The archive is flat, which is what makes the bare relative paths in the notebooks resolve.

Run:  python tools/build_zip.py
"""
import os
import shutil
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-10/project-7-improving-ddpm")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
SUBMISSION = f"{BASE}/submission"

# The unit states that a wrong file name scores 0, so both names are written out here rather
# than derived, and verify.py asserts them independently.
ARCHIVE = "project7_code.zip"
REPORT_PDF = "project7_report.pdf"

CONTENTS = [
    "DDPM_CosineSchedule.ipynb", "DigitClassifier.ipynb", "main_report.ipynb",
    "ddpm_linear.pth", "ddpm_cosine.pth", "DigitClassifier.pth", "DDPM_history.json",
    "mnist_custom_test.pt", "figure_1_grids.pdf", "figure_2_steps.pdf", "figure_3_few_steps.pdf",
    "figure_4_mechanism.pdf",
]

# How long main_report.ipynb takes, measured on the runs recorded in the project README.
RUNTIME = ("about __GPU_MINUTES__ minutes on the IFN680 GPU server and about __CPU_MINUTES__ "
           "minutes on a CPU")
MEASURED = {"__GPU_MINUTES__": os.environ.get("P7_GPU_MINUTES", "10"),
            "__CPU_MINUTES__": os.environ.get("P7_CPU_MINUTES", "120")}
for key, value in MEASURED.items():
    RUNTIME = RUNTIME.replace(key, value)

README = f"""IFN680 Project 7 - Improving DDPM
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  DDPM_CosineSchedule.ipynb   Task 1: trains the conditional DDPM of Tutorial 9.3 twice, with the
                              tutorial's linear_beta_schedule and with a cosine_beta_schedule,
                              three training seeds each, and defines the samplers
  DigitClassifier.ipynb       trains the standalone digit classifier used for evaluation
  main_report.ipynb           Task 2: loads the checkpoints and reproduces every number and figure
                              in the report, including the 12x10 grid of generated digits for
                              both schedules. Contains no training loop.
  ddpm_linear.pth             trained U-Nets, plain state_dicts: the final model of each training
  ddpm_cosine.pth             seed (seed_0, seed_1, seed_2) and seed 0's earlier snapshots
                              (epoch_10, epoch_25, epoch_50, epoch_100)
  DigitClassifier.pth         the classifier's weights
  DDPM_history.json           every run's loss and seconds per epoch, and the training settings
  mnist_custom_test.pt        the 10,000-image test split of mnist_custom.pt, the only data
                              main_report.ipynb reads
  figure_*.pdf                the four figures main_report.ipynb writes; figures 1 and 2 are the
                              report's

HOW TO RUN
  Unzip everything into one folder and run main_report.ipynb top to bottom. It needs no
  arguments and no data beyond what is in this archive. It samples tens of thousands of digits,
  which takes {RUNTIME}.
  Every random draw is seeded, so it prints the same numbers each time; on a different device
  the last printed digit can differ by float rounding.

  The stored outputs of all three notebooks come from the IFN680 GPU environment. The training
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
    for required in ["DDPM_CosineSchedule.ipynb", "main_report.ipynb"]:
        assert required in names, f"{required} must be present or the code scores 0"
    assert not any(".DS_Store" in n or ".ipynb_checkpoints" in n for n in names)
    total = sum(info.file_size for info in archive.infolist())

print(f"wrote {path}")
for info in sorted(zipfile.ZipFile(path).infolist(), key=lambda i: -i.file_size):
    print(f"  {info.file_size:>10,}  {info.filename}")
print(f"  {'-' * 10}")
print(f"  {total:>10,}  uncompressed   ({os.path.getsize(path):,} bytes on disk)")
print(f"wrote {SUBMISSION}/{REPORT_PDF}")
