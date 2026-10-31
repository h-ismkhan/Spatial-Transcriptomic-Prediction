# In the name of God
"""
main.py  --  multimodal gene-expression regression on HEST (INT1-INT24) with missing modalities.

Run from the folder that contains hest_data/ and Hest_Reg_Lymph_INT1to24.py:
    python main.py

Files: main.py (this, all settings) | data_utils.py | encoders.py | methods.py | reporting.py
MIST is NOT implemented: it models single-T-cell RNA + TCR sequences (VAE with an MMD term); it has no
counterpart for H&E / nuclei / context spots, so it is excluded from the competition.
"""
import os
import sys
import time
import json
from types import SimpleNamespace

import numpy as np
import torch

# =============================================================================================
#                                      SETTINGS (edit here)
# =============================================================================================
HF_TOKEN = 'hf_*'                       # Hugging Face token (needed for gated models such as H-optimus-0/1)
ROOT = "hest_data"
SPLIT_FILE = None                       # None -> <ROOT>/splits_INT1to24.json
SAMPLE_IDS = None                       # None -> INT1..INT24
N_GENES = 50
MIN_DETECT_FRAC = 0.1
EMPTY_POLICY = "keep"                   # 'keep' | 'restore_image'  (spots that lose every modality)
APPLY_NATURAL_MISSING = True            # nuclei of INT20/22/23 treated as missing

# missing-modality configurations ('-' or '_' accepted; also 'complex-30-30-5-5-20-5-5')
CONFIGS = ['100-image-20-nuclei-20-context', '20-image-100-nuclei-20-context',
           '20-image-20-nuclei-100-context', '100-image-100-nuclei-100-context']

N_RUNS = 3                              # run r uses seed BASE_SEED + r (missing pattern AND initialisation)
BASE_SEED = 42

# competitors: (METHOD, imputation, imputation, ...)  imputation in {"0", "mean"}
# METHOD in OURS, OURS-BRANCHED, BLEEP, GHIST, MCLSTEXP, STORM, FMH2ST
COMPETITORS = [
    ("OURS", "0")
    # ("OURS", "0", "mean"),
    # ("OURS-BRANCHED", "0", "mean"),
    # ("OURS-KVOTE", "0", "mean"),         # one entry per k in KVOTE_KS (OURS-3Vote, OURS-5Vote, ...)      # one copy of the OURS head per gene after the shared encoder
    # ("BLEEP", "0", "mean"),
    # ("GHIST", "0", "mean"),
    # ("MCLSTEXP", "0", "mean"),
    # ("STORM", "0", "mean"),
    # ("FMH2ST", "0", "mean"),
]
KVOTE_KS = (3, 5, 7)                    # k of the nearest-neighbour vote in the fused feature space
VARIANTS = ("default", "our_encoder")   # competitor encoders (OURS has only its own encoder)

# ---- encoders / caches
# frozen, cached encoders. OURS / OURS-BRANCHED / OURS-KVOTE are run once per encoder in this list
# (names like OURS-0-hoptimus1). The FIRST one is also the encoder of the competitors' 'our_encoder'
# variants / FMH2ST and the reference of the Wilcoxon tests (OURS-0-<first>).
# available: hoptimus0, hoptimus1, uni, gigapath (pathology, gated on HF) | resnet50, densenet121, vit_b16,
# dinov2 (ImageNet / generic) | dummy (debug). Aliases 'optimus0' / 'optimus1' are accepted.
# FOUNDATION_ENCODERS = ['hoptimus1', 'hoptimus0', 'uni', 'resnet50', 'densenet121', 'vit_b16']
FOUNDATION_ENCODERS = ['hoptimus1']
STORM_ENCODER = "resnet50"              # STORM's own frozen image encoder (ResNet50 in its code)
# fusion of the three modality vectors inside the OURS encoder. OURS-family entries run once per
# (encoder, fusion); the FIRST fusion is also used by the competitors' 'our_encoder' variants.
# available: 'concat' (original simple concatenation), 'sum', 'gate', 'attn', 'lmf'
FUSIONS = ['concat', 'gate', 'attn', 'lmf', 'sum']
PRETRAINED = True                       # pretrained weights for ResNet50/DenseNet121 pixel backbones
NATIVE_IMG_SIZE = 96                    # pixel patches of the 'default' competitors are down-scaled to this
DEFAULT_CTX_PATCHES = 3                 # neighbour patches per spot seen by the 'default' pixel encoders
EMBED_BATCH = 128
EMB_ON_DEVICE_MAX_GB = 3.0

# ---- training
ENSEMBLE_SIZE_CACHED = 5                # models averaged for OURS / our_encoder / STORM-default (cheap)
ENSEMBLE_SIZE_RAW = 1                   # models averaged for pixel-trained 'default' competitors (expensive)
RAW_EPOCHS = 20                         # epochs / patience / steps for pixel-trained 'default' competitors
RAW_PATIENCE = 5
RAW_MAX_STEPS_PER_EPOCH = None          # e.g. 100 to cap the cost of an epoch
HPARAMS = {
    "common": dict(lr=3e-4, weight_decay=0.05, batch_size=256, epochs=60, warmup=2, patience=10, enc_lr_mult=1.0),
    "OURS": {},
    "BLEEP": dict(lr=1e-3, enc_lr_mult=0.1, temperature=1.0, top_k=50),
    "MCLSTEXP": dict(enc_lr_mult=0.1, temperature=1.0, top_k=50, heads=8, head_dim=64, layers=2),
    "GHIST": dict(emb_dim=256, n_ref=8, w_nc=1.0),
    "FMH2ST": dict(lr=1e-4, window=256, k_spatial=4, k_feature=8),
    "STORM": dict(rank=15, lr=0.03, enc_lr=1e-3, epochs=250, patience=20, eval_every=5, knn=6, r1_batch=8192,
                  P=(1e-4, 0.30, 0.30, 0.05), smooth_alpha=0.5, smooth_steps=3),
}
OUT_DIR = "."                           # output-MSE.txt, output-MAE.txt, output-PCC.txt, output-summary.txt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE_DATASET = False
VERBOSE_TRAIN = False
# =============================================================================================

from data_utils import (build_base_datasets, set_missing, normalize_config, patch_offsets, SplitView,
                        build_embedding_table, build_pixel_store, RunData, canon_encoder)
from methods import build_method, needed_assets, METHOD_NAMES, RAW_PIXEL_METHODS
import reporting
from reporting import Results, evaluate, write_outputs, summary_text

CFG = SimpleNamespace(**{k: v for k, v in globals().items() if k.isupper()})
CFG.FOUNDATION_ENCODERS = list(dict.fromkeys(canon_encoder(e) for e in CFG.FOUNDATION_ENCODERS))   # unique, ordered
CFG.FOUNDATION_ENCODER = CFG.FOUNDATION_ENCODERS[0]         # primary encoder
CFG.STORM_ENCODER = canon_encoder(CFG.STORM_ENCODER)
CFG.FUSIONS = list(dict.fromkeys(CFG.FUSIONS))
CFG.FUSION = CFG.FUSIONS[0]


def expand_entries(competitors):
    """-> list of (method, variant, mode, display_name, encoder-or-None)"""
    out = []
    for item in competitors:
        item = (item,) if isinstance(item, str) else tuple(item)
        m = item[0].upper()
        modes = [str(x).lower() for x in item[1:]] or ["0", "mean"]
        for mode in modes:
            if mode not in ("0", "mean"):
                raise ValueError("imputation must be '0' or 'mean'")
        if m != "OURS-KVOTE" and m not in METHOD_NAMES:
            raise ValueError(f"unknown method {m}; choose from {METHOD_NAMES} or OURS-KVOTE")
        for mode in modes:
            tag = "0" if mode == "0" else "Mean"
            if m.startswith("OURS"):                           # OURS family: one entry per encoder (and per k)
                for enc in CFG.FOUNDATION_ENCODERS:
                    for fus in CFG.FUSIONS:
                        if m == "OURS-KVOTE":
                            for k in CFG.KVOTE_KS:
                                out.append((f"OURS-{k}VOTE", "our_encoder", mode, f"OURS-{k}Vote-{tag}-{enc}-{fus}", enc, fus))
                        else:
                            out.append((m, "our_encoder", mode, f"{m}-{tag}-{enc}-{fus}", enc, fus))
                continue
            for v in VARIANTS:
                out.append((m, v, mode, f"{m}-{tag}-{'Default' if v == 'default' else 'Our-Encoder'}", None, None))
    return out


def main():
    cfg, dev = CFG, torch.device(CFG.DEVICE)
    if cfg.HF_TOKEN and cfg.HF_TOKEN.strip("*"):          # skip the '****' placeholder
        from huggingface_hub import login
        login(token=cfg.HF_TOKEN)
    configs = [c.strip().strip("'\"") for c in cfg.CONFIGS]
    entries = expand_entries(cfg.COMPETITORS)
    reporting.REF_METHOD = f"OURS-0-{cfg.FOUNDATION_ENCODER}-{cfg.FUSION}"      # Wilcoxon reference
    print(f"device {dev} | configs {configs} | runs {cfg.N_RUNS} | {len(entries)} method variants")
    print("methods:", ", ".join(e[3] for e in entries))

    base = build_base_datasets(cfg)
    ids = base["train"].sample_ids
    offsets, _ = patch_offsets(cfg.ROOT, ids)
    emb_names, need_pix = needed_assets([(m, v, e[4]) for e in entries for m, v in [e[:2]]], cfg)
    stores = {"emb": {}}
    for n in emb_names:
        try:
            stores["emb"][n] = build_embedding_table(cfg, ids, n, dev)
        except Exception as ex:                      # e.g. gated Hugging Face repo without access
            if n == cfg.FOUNDATION_ENCODER or (n == cfg.STORM_ENCODER and any(m == "STORM" for m, *_ in entries)):
                raise
            print(f"[WARNING] encoder '{n}' skipped: {type(ex).__name__}: {str(ex).splitlines()[0][:150]}")
            entries = [e for e in entries if e[4] != n]
    if need_pix:
        stores["pix"] = build_pixel_store(cfg, ids, offsets)

    res = Results()
    t0 = time.time()
    for c in configs:
        for run in range(cfg.N_RUNS):
            seed = cfg.BASE_SEED + run
            for ds in base.values():
                set_missing(ds, c, seed, verbose=cfg.VERBOSE_DATASET and run == 0)
            views = {s: SplitView(ds, offsets) for s, ds in base.items()}
            rd = RunData(cfg, views, stores, dev)
            av = views["train"].avail.mean(0) * 100
            print(f"\n##### config '{c}' | run {run + 1}/{cfg.N_RUNS} | seed {seed} | train availability "
                  f"image {av[0]:.0f}% nuclei {av[1]:.0f}% context {av[2]:.0f}%")
            for m, v, mode, name, enc, fus in entries:
                t1 = time.time()
                raw = (v == "default" and m in RAW_PIXEL_METHODS)
                E = cfg.ENSEMBLE_SIZE_RAW if raw else cfg.ENSEMBLE_SIZE_CACHED
                preds = []
                for e in range(E):
                    meth = build_method(m, v, rd, mode, seed * 100 + e, cfg, dev, enc, fus)
                    meth.fit()
                    preds.append(meth.predict("test"))
                    del meth
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                P = np.mean(preds, 0)
                if not np.isfinite(P).all():
                    print(f"  [warn] {name}: non-finite ensemble prediction replaced by the train mean")
                    P = np.where(np.isfinite(P), P, rd.y_mu[None, :])
                r = evaluate(views["test"], P, rd.y_mu)
                res.add(c, name, r)
                print(f"  {name:32s} MSE {r['MSE']:.4f} MAE {r['MAE']:.4f} PCC {r['PCC']:.4f} "
                      f"({E} model(s), {time.time() - t1:.0f}s)")
                write_outputs(res, configs, cfg.OUT_DIR)          # incremental: results survive a crash
    print("\n" + summary_text(res, configs))
    write_outputs(res, configs, cfg.OUT_DIR)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Files: output-MSE/MAE/PCC/SCC/R2.txt and output-summary.txt")


if __name__ == "__main__":
    main()
