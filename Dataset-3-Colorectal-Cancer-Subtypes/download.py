"""
Stage 1: download the THIRD HEST-1K subset (human colorectal, ZEN36..ZEN49: 14 slides, 7 patients, 40x)
into ./hest_data  (same folder name as for INT1-24, so the reader needs no path change).

Usage:
    export HF_TOKEN=hf_xxx        # your NEW token (PowerShell: $env:HF_TOKEN="hf_xxx")
    python download_crc.py

It also writes hest_data/splits_INT1to24.json (a patient-level split with
both rectal and colon cancer in test, seed 42). The file name is kept ONLY so that the unchanged reader and test script find it.
Delete that file if you prefer the reader's own random patient split.
"""
import os
import json
from huggingface_hub import login, snapshot_download

TOKEN = os.environ.get("HF_TOKEN") or "hf_***"
repo_id = "MahmoodLab/hest"
local_dir = "hest_data"

ids = [f"ZEN{i}" for i in range(36, 50)]               # ZEN36 ... ZEN49 (14 slides)
patterns = [f"*{i}[_.]**" for i in ids]               # 'ID_' or 'ID.' so ZEN4 can never match ZEN49

login(token=TOKEN)
snapshot_download(repo_id=repo_id, repo_type="dataset",
                  allow_patterns=patterns,
                  ignore_patterns=["wsis/*"],          # full-resolution .tif are not used by the reader
                  local_dir=local_dir)

# ---- patient-level split (seed 42). Each patient has 2 replicate slides and both stay together.
#      test: Patient 7 (rectal cancer) + Patient 3 (colon cancer); val: Patient 4 (colon cancer);
#      train: Patients 1, 2, 5 (cancer) and Patient 6 (the only healthy patient).
split = {
    "train": ["ZEN37", "ZEN38", "ZEN41", "ZEN42", "ZEN46", "ZEN47", "ZEN48", "ZEN49"],
    "val":   ["ZEN39", "ZEN43"],
    "test":  ["ZEN36", "ZEN40", "ZEN44", "ZEN45"],
}
assert sorted(sum(split.values(), [])) == sorted(ids)
sf = os.path.join(local_dir, "splits_INT1to24.json")
if not os.path.exists(sf):
    with open(sf, "w") as f:
        json.dump(split, f, indent=1)

# ---- file check: exactly the files the reader opens
print("\nfile check (st / patches / metadata / cellvit_seg):")
for i in ids:
    ok = [os.path.exists(f"{local_dir}/st/{i}.h5ad"),
          os.path.exists(f"{local_dir}/patches/{i}.h5"),
          os.path.exists(f"{local_dir}/metadata/{i}.json"),
          os.path.exists(f"{local_dir}/cellvit_seg/{i}_cellvit_seg.geojson.zip")]
    print(f"  {i}", "OK" if all(ok) else f"MISSING {ok}")