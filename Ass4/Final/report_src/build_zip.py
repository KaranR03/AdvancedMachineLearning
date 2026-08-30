r"""Assemble submission/project4_code.zip and submission/project4_report.pdf.

The archive is flat and holds exactly what a marker needs to reproduce the report: the three case
notebooks and the cached predictions each one writes, plus a README naming where the supplied
datasets and checkpoints have to go. Those are shipped with the assessment and are not duplicated
here; the notebooks look for them in several layouts and stop with a readable message otherwise.

Run:  python tools/build_zip.py
"""
import os
import shutil
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-06/project-4-ai-auditing")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
SUBMISSION = f"{BASE}/submission"

# The unit states that a wrong file name scores 0, so both names are asserted, not assumed.
ARCHIVE = "project4_code.zip"
REPORT_PDF = "project4_report.pdf"
CONTENTS = [f"case{case}{suffix}" for case in (1, 2, 3)
            for suffix in (".ipynb", "_predictions.pkl")]

README = """IFN680 Project 4 - AI Auditing
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  case1.ipynb   clean-shaven detector, studio to mobile
  case2.ipynb   eyeglasses detector for a retail kiosk
  case3.ipynb   'appears young' tagger with a high-confidence bypass
  caseN_predictions.pkl
                the labels and positive-class probabilities each notebook computes, cached
                so the metrics can be re-derived without a second pass over the images

HOW TO RUN
  Unzip the three supplied datasets and copy the three supplied checkpoints beside these
  notebooks, so that the layout is:

      case1.ipynb  case2.ipynb  case3.ipynb
      Case1/Case1Dataset/test_internal/{negative,positive}
      Case1/Case1Dataset/test_external/{negative,positive}
      Case1/resnet_frozen_best.pth
      Case2/...   Case3/...

  Then run each notebook top to bottom. The images and weights are supplied with the
  assessment and are not duplicated here.

  Each notebook also accepts CaseNDataset/ directly beside it, or the same layout under
  ~/Assessment4/. It searches those locations in order and stops with a message naming the
  expected layout if none of them holds the data.

  All three checkpoints are named resnet_frozen_best.pth but they are three different
  models. Each notebook checks the MD5 of the file it loaded against the one for its own
  case, so putting all three in one folder fails immediately instead of producing wrong
  numbers quietly.

  Every tensor and the model are moved with .to(device), where device is cuda:0 when one
  is visible and cpu otherwise, so the notebooks run unchanged on either. The stored
  outputs are from a CPU run. The same notebooks were also run on the IFN680 GPU node
  (torch 2.13.0+cu126, device cuda:0): of the 83 lines they print, the only differences
  are the four that name the environment and four values that move in the third decimal,
  where a borderline probability lands on the other side of 0.5 under a different kernel.
  Every figure quoted in the report is identical on both devices.
"""

os.makedirs(SUBMISSION, exist_ok=True)

for name in CONTENTS:
    assert os.path.exists(f"{NBDIR}/{name}"), f"missing {name}; execute the notebooks before zipping"

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
    for case in (1, 2, 3):
        assert f"case{case}.ipynb" in names, "all three notebooks must be present or the code scores 0"
    assert not any(".DS_Store" in n or ".ipynb_checkpoints" in n for n in names)
    total = sum(info.file_size for info in archive.infolist())

print(f"wrote {path}")
for info in sorted(zipfile.ZipFile(path).infolist(), key=lambda i: -i.file_size):
    print(f"  {info.file_size:>10,}  {info.filename}")
print(f"  {'-' * 10}")
print(f"  {total:>10,}  uncompressed   ({os.path.getsize(path):,} bytes on disk)")
print(f"wrote {SUBMISSION}/{REPORT_PDF}")
