"""
Stage 1: download the second HEST-1K subset (human kidney, NCBI692..NCBI714: 23 slides, 22 patients)
into ./hest_data  (same folder name as for INT1-24, so the reader needs no path change).

Usage:
    export HF_TOKEN=hf_xxx        # your NEW token (PowerShell: $env:HF_TOKEN="hf_xxx")
    python download.py

It also writes hest_data/splits_INT1to24.json (a patient-level split stratified by
healthy / DKD / AKI). The file name is kept ONLY so that the unchanged reader and test script find it.
Delete that file if you prefer the reader's own random patient split.
"""
import os
import json
from huggingface_hub import login, snapshot_download

TOKEN = os.environ.get("HF_TOKEN") or "hf_***"
repo_id = "MahmoodLab/hest"
local_dir = "hest_data"

ids = [f"NCBI{i}" for i in range(692, 715)]            # NCBI692 ... NCBI714 (23 slides)
patterns = [f"*{i}[_.]**" for i in ids]               # 'ID_' or 'ID.' so NCBI71 can never match NCBI714

login(token=TOKEN)
snapshot_download(repo_id=repo_id, repo_type="dataset",
                  allow_patterns=patterns,
                  ignore_patterns=["wsis/*"],          # full-resolution .tif are not used by the reader
                  local_dir=local_dir)

# ---- patient-level split (seed 42), stratified by condition so that test/val contain
#      healthy, DKD and AKI patients. Patient 13 (NCBI701 + NCBI702) stays together in train.
split = {
    "train": ["NCBI693", "NCBI694", "NCBI695", "NCBI696", "NCBI698", "NCBI699", "NCBI700", "NCBI701",
              "NCBI702", "NCBI703", "NCBI704", "NCBI707", "NCBI709", "NCBI710", "NCBI711", "NCBI712"],
    "val":   ["NCBI692", "NCBI705", "NCBI713"],
    "test":  ["NCBI697", "NCBI706", "NCBI708", "NCBI714"],
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