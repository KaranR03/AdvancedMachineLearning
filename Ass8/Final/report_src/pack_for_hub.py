r"""Pack what the IFN680 GPU node needs to execute main_report.ipynb, in upload-sized parts.

The browser route that moves files onto the Hub caps one upload at about 10 MB, and
tiny_nerf_data.npz alone is 12.7 MB, so the inputs go up as one zip cut into parts of at most
9 MB. On the Hub the parts are joined, the sha256 printed here is checked, and the zip is unpacked
into a fresh folder.

Run:  python tools/pack_for_hub.py <output dir>
"""
import hashlib
import os
import sys
import zipfile

BASE = ("f:/document/IFN_680_Advanced_Machine_Learning_and_Applications/"
        "weeks/week-11/project-8-3d-movie")
FILES = [("notebook", "main_report.ipynb"), ("notebook", "tinynerf.pth"),
         ("notebook", "extended_nerf.pth"), ("notebook", "ablation_no_viewdirs.pth"),
         ("notebook", "ablation_uniform96.pth"), ("notebook", "training_history.json"),
         ("notebook", "tiny_nerf_data.npz"), ("tools", "hub_run_all.py")]
PART = 9 * 1024 * 1024

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
path = f"{out}/p8_hub_inputs.zip"
with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
    for folder, name in FILES:
        archive.write(f"{BASE}/{folder}/{name}", arcname=name)
data = open(path, "rb").read()
print(f"{path}: {len(data):,} bytes, sha256 {hashlib.sha256(data).hexdigest()}")
for i in range(0, len(data), PART):
    part = f"{path}.part{i // PART:02d}"
    with open(part, "wb") as handle:
        handle.write(data[i:i + PART])
    print(f"  {part}: {len(data[i:i + PART]):,} bytes")
for folder, name in FILES:
    digest = hashlib.sha256(open(f"{BASE}/{folder}/{name}", "rb").read()).hexdigest()
    print(f"  {name}: sha256 {digest[:16]}")
