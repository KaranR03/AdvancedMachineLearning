r"""Assemble submission/project8_code.zip and submission/project8_report.pdf.

The brief names three notebooks the archive must include, TinyNeRF.ipynb, ExtendedNeRF.ipynb and
main_report.ipynb, and asks for "all files and code required for reproducibility, including model
setup, trained weights (.pth) and evaluation code". main_report.ipynb reads the four checkpoints,
the training history and tiny_nerf_data.npz, so all of them ship; the training notebooks also
write training_log.txt, which ships as their per-epoch record.

The archive is flat, which is what makes the bare relative paths in the notebooks resolve.

Run:  python tools/build_zip.py
"""
import os
import shutil
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-11/project-8-3d-movie")
NBDIR = f"{BASE}/notebook"
REPORT = f"{BASE}/report"
SUBMISSION = f"{BASE}/submission"

# The unit states that a wrong file name scores 0, so both names are written out here rather
# than derived, and verify.py asserts them independently.
ARCHIVE = "project8_code.zip"
REPORT_PDF = "project8_report.pdf"
REQUIRED = ["TinyNeRF.ipynb", "ExtendedNeRF.ipynb", "main_report.ipynb"]

CONTENTS = REQUIRED + [
    "tinynerf.pth", "extended_nerf.pth", "ablation_no_viewdirs.pth", "ablation_uniform96.pth",
    "training_history.json", "training_log.txt", "tiny_nerf_data.npz", "results_summary.json",
    "figure_1_test_views.pdf", "figure_2_novel_views.pdf", "figure_3_curves.pdf",
    "figure_4_view_dependence.pdf", "figure_5_sampling.pdf", "nerf_turntable.gif",
]

# How long main_report.ipynb takes, measured on the runs recorded in the project README, as a
# phrase so a short run never reads "about 1 minutes".
GPU_TIME = os.environ.get("P8_GPU_TIME", "under a minute")
CPU_TIME = os.environ.get("P8_CPU_TIME", "about 10 minutes")

README = f"""IFN680 Project 8 - Extending Tiny NeRF
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  TinyNeRF.ipynb              the Tutorial 10.4 Tiny NeRF, trained for 100 epochs and evaluated
                              on the six held-out views (images 100 to 105)
  ExtendedNeRF.ipynb          Task 1 (view-dependent colour) and Task 2 (hierarchical sampling
                              with sample_pdf), plus the two ablations: A, no view direction,
                              and B, view-dependent colour with 96 uniform samples
  main_report.ipynb           Task 3: loads the four checkpoints and reproduces every number and
                              figure in the report. Contains no training loop.
  tinynerf.pth                trained weights, final epoch: Tiny NeRF; Extended NeRF (coarse and
  extended_nerf.pth           fine networks); ablation A (coarse and fine); ablation B (one
  ablation_no_viewdirs.pth    network)
  ablation_uniform96.pth
  training_history.json       every model's train and test PSNR and seconds per epoch
  training_log.txt            the training notebooks' per-epoch log
  tiny_nerf_data.npz          the tutorial's dataset: 106 images at 100 x 100, poses, focal length
  results_summary.json        every number main_report.ipynb prints, as JSON
  figure_*.pdf                the five report figures, written by main_report.ipynb
  nerf_turntable.gif          the 40-frame novel-view turntable, Tiny NeRF | Extended NeRF

HOW TO RUN
  Unzip everything into one folder and run main_report.ipynb top to bottom. It needs no
  arguments and no data beyond what is in this archive, and takes {GPU_TIME} on the IFN680
  GPU server and {CPU_TIME} on a CPU. Rendering is deterministic, so it prints the same
  numbers each time; on a different device the last printed digit can differ by float
  rounding, and the render times differ.

  The stored outputs of all three notebooks come from the IFN680 GPU environment. The training
  notebooks retrain every model from scratch (about 3.5 hours on that GPU).
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
    for required in REQUIRED:
        assert required in names, f"{required} must be present or the code scores 0"
    assert not any(".DS_Store" in n or ".ipynb_checkpoints" in n or "/" in n for n in names)
    total = sum(info.file_size for info in archive.infolist())

print(f"wrote {path}")
for info in sorted(zipfile.ZipFile(path).infolist(), key=lambda i: -i.file_size):
    print(f"  {info.file_size:>10,}  {info.filename}")
print(f"  {'-' * 10}")
print(f"  {total:>10,}  uncompressed   ({os.path.getsize(path):,} bytes on disk)")
print(f"wrote {SUBMISSION}/{REPORT_PDF}")
