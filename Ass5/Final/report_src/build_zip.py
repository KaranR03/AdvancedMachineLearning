r"""Assemble submission/project5_code.zip and submission/project5_report.pdf.

The brief names what the archive must hold: the three notebooks, the held-out test set, the
trained weights and the evaluation code. The four history files are included as well, because
main_report.ipynb reads them to draw its learning curves and the archive has to be runnable on
its own; the two report figures are included so a marker sees the same PDFs the report embeds.

The archive is flat. The Project 4 submission that scored 2/2 was nested inside a folder with
.DS_Store and __MACOSX entries and lost nothing for it, so this is a tidiness choice rather than
a rule, but a flat archive is also what makes the bare relative paths in the notebooks resolve.

Run:  python tools/build_zip.py
"""
import os
import shutil
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-08/project-5-extending-addition-llm")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
SUBMISSION = f"{BASE}/submission"

# The unit states that a wrong file name scores 0, so both names are written out here rather
# than derived, and verify.py asserts them independently.
ARCHIVE = "project5_code.zip"
REPORT_PDF = "project5_report.pdf"

CONTENTS = [
    "LLMForward.ipynb", "LLMReverse.ipynb", "main_report.ipynb",
    "LLMForward.pth", "LLMReverse.pth",
    "project5_testset.pkl",
    "LLMForward_history.json", "LLMReverse_history.json",
    "LLMForward_history_80k.json", "LLMReverse_history_80k.json",
    "figure_1_overview.pdf", "figure_2_by_length.pdf",
]

README = """IFN680 Project 5 - Extending Addition LLM
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  LLMForward.ipynb          trains the left-to-right model on addition and subtraction
  LLMReverse.ipynb          trains the right-to-left model on the same data
  main_report.ipynb         Task 3: loads both models and reproduces every number and
                            figure the report carries. Contains no training loop.
  LLMForward.pth            trained weights, a plain state_dict
  LLMReverse.pth
  project5_testset.pkl      the shared held-out set, 10,000 examples, balanced between
                            addition and subtraction
  LLM*_history.json         validation curves from the two training runs
  LLM*_history_80k.json     validation curves from the same runs on 80,000 examples,
                            which is the data-efficiency comparison in the report
  figure_1_overview.pdf     the two figures the report embeds, as main_report.ipynb
  figure_2_by_length.pdf    writes them

HOW TO RUN
  Unzip everything into one folder and run main_report.ipynb top to bottom. It needs no
  arguments and no data beyond what is in this archive, and takes about a minute on CPU.

  The two training notebooks reproduce the checkpoints from scratch. Thirty epochs on the
  full training set took about seventeen minutes on the IFN680 GPU node, and each notebook
  then repeats the training on a reduced set for the data-efficiency comparison, so expect
  roughly half an hour per notebook in total.

  Every tensor and both models are moved with .to(device), where device is cuda when one is
  visible and cpu otherwise, so the notebooks run unchanged on either. The two training
  notebooks were run on the IFN680 GPU node and main_report.ipynb on a CPU, and the accuracies
  they report agree to the last digit: decoding is greedy, so the device changes the speed and
  not the result.

  The held-out test set is regenerated from SEED inside main_report.ipynb and compared
  against the shipped project5_testset.pkl, so the split can be audited without trusting
  the pickle.
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
    for required in ["LLMForward.ipynb", "LLMReverse.ipynb", "main_report.ipynb"]:
        assert required in names, f"{required} must be present or the code scores 0"
    assert not any(".DS_Store" in n or ".ipynb_checkpoints" in n for n in names)
    total = sum(info.file_size for info in archive.infolist())

print(f"wrote {path}")
for info in sorted(zipfile.ZipFile(path).infolist(), key=lambda i: -i.file_size):
    print(f"  {info.file_size:>10,}  {info.filename}")
print(f"  {'-' * 10}")
print(f"  {total:>10,}  uncompressed   ({os.path.getsize(path):,} bytes on disk)")
print(f"wrote {SUBMISSION}/{REPORT_PDF}")
