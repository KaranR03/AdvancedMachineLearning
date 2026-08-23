r"""Assemble submission/project3_code.zip and submission/project3_report.pdf.

The archive is flat and holds exactly what the grader needs to run main_report.ipynb: the two
notebooks, the trained weights and the saved learning curves, plus a README naming where the
dataset has to go. The dataset itself is 158 MB and is supplied by the unit, so it is not shipped.

Run:  python tools/build_zip.py
"""
import io
import os
import shutil
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-05/project-3-aircraft-classification")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
SUBMISSION = f"{BASE}/submission"

# The unit states that a wrong file name scores 0, so both names are asserted, not assumed.
ARCHIVE = "project3_code.zip"
REPORT_PDF = "project3_report.pdf"
CONTENTS = ["development.ipynb", "main_report.ipynb", "best_model.pth", "histories.pkl"]

README = """IFN680 Project 3 - Aircraft Classification
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  development.ipynb   trains the baseline, the three experiments and the final model,
                      and writes best_model.pth and histories.pkl
  main_report.ipynb   reproduces every figure and metric in the report. It retrains
                      nothing: it reloads the two files above.
  best_model.pth      weights of the final model, refit on the whole trainval split
  histories.pkl       learning curves and validation scores for all runs

HOW TO RUN
  Unzip FGVCAircraft_Subset20.zip into this same folder, so that the layout is:

      main_report.ipynb
      FGVCAircraft_Subset20/trainval/class_00 ... class_19
      FGVCAircraft_Subset20/test/class_00 ... class_19

  Then run main_report.ipynb top to bottom. It checks for the dataset in its first cell
  and stops with a readable message if it is somewhere else.

  The image set is not included here because it is 158 MB and is supplied with the task.
"""

os.makedirs(SUBMISSION, exist_ok=True)

for name in CONTENTS:
    assert os.path.exists(f"{NBDIR}/{name}"), f"missing {name}; build it before zipping"

assert os.path.exists(f"{REPORT}/{REPORT_PDF}"), "compile the report first"
shutil.copyfile(f"{REPORT}/{REPORT_PDF}", f"{SUBMISSION}/{REPORT_PDF}")

path = f"{SUBMISSION}/{ARCHIVE}"
if os.path.exists(path):
    os.remove(path)

with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in CONTENTS:
        zf.write(f"{NBDIR}/{name}", arcname=name)
    zf.writestr("README.txt", README)

with zipfile.ZipFile(path) as zf:
    names = zf.namelist()
    assert "development.ipynb" in names and "main_report.ipynb" in names, \
        "both notebooks must be in the archive or the code scores 0"
    assert not any(".DS_Store" in n or ".ipynb_checkpoints" in n for n in names)
    total = sum(i.file_size for i in zf.infolist())

print(f"wrote {path}")
for info in sorted(zipfile.ZipFile(path).infolist(), key=lambda i: -i.file_size):
    print(f"  {info.file_size:>10,}  {info.filename}")
print(f"  {'-' * 10}")
print(f"  {total:>10,}  uncompressed   ({os.path.getsize(path):,} bytes on disk)")
print(f"wrote {SUBMISSION}/{REPORT_PDF}")
