# In the name of God
"""
data_utils.py
=============
Everything that touches the HEST dataset reader:

  * building the train/val/test datasets once and re-simulating the missing modalities for every
    (config, run) with the reader's own simulator,
  * global patch indexing + on-disk caches (frozen-encoder embeddings, down-scaled pixel patches),
  * SplitView (plain numpy view of a split), RunData (train statistics used for 0/mean imputation),
  * Batcher (turns spot indices into imputed model inputs),
  * spatial tiling (windows of neighbouring spots, used by graph methods).

IMPUTATION RULES (identical for every method):
  mode '0'   : a missing modality is replaced by zeros
               image/context embedding -> zero vector ; pixels -> black patch (the reader's own
               placeholder) ; nuclei -> zero counts (then the usual feature transform + standardisation)
  mode 'mean': a missing modality is replaced by the mean of that modality over the TRAINING spots
               where it is available (embedding mean, mean image, mean standardised nuclei feature = 0)
Missing modalities are never "peeked": values of unavailable modalities are never read by a model.
"""
import os
import importlib
import numpy as np
import torch
import h5py

NUC_DIM = 10          # 5 log1p counts + 5 fractions
N_POS_BINS = 64


# ----------------------------------------------------------------------------- dataset reader
def import_dataset_class():
    for name in ("Hest_Reg_Lymph_INT1to24", "hest_reg_lymph_int1to24"):
        try:
            return importlib.import_module(name).Hest_Reg_Lymph_INT1to24
        except ImportError:
            continue
    raise ImportError("Cannot import Hest_Reg_Lymph_INT1to24 (put Hest_Reg_Lymph_INT1to24.py or "
                      "hest_reg_lymph_int1to24.py next to main.py)")


def normalize_config(c):
    """'100-image-20-nuclei-20-context' -> '100_image_20_nuclei_20_context' (reader format)."""
    return c.strip().strip("'\"").replace("-", "_")


def build_base_datasets(cfg):
    DS = import_dataset_class()
    split_file = cfg.SPLIT_FILE or os.path.join(cfg.ROOT, "splits_INT1to24.json")
    first = normalize_config(cfg.CONFIGS[0])
    common = dict(root=cfg.ROOT, sample_ids=cfg.SAMPLE_IDS, split_file=split_file, missing_config=first,
                  seed=cfg.BASE_SEED, n_genes=cfg.N_GENES, min_detect_frac=cfg.MIN_DETECT_FRAC,
                  empty_policy=cfg.EMPTY_POLICY, apply_natural_missing=cfg.APPLY_NATURAL_MISSING,
                  verbose=cfg.VERBOSE_DATASET)
    train = DS(split="train", genes=None, **common)
    val = DS(split="val", genes=train.genes, **common)
    test = DS(split="test", genes=train.genes, **common)
    return {"train": train, "val": val, "test": test}


def set_missing(ds, config, seed, verbose=False):
    """Re-run the reader's own missing-modality simulation (same code path as the constructor)."""
    ds.missing_config = normalize_config(config)
    ds.seed = seed
    ds._parse_missing_config()
    ds.avail_sim = ds._precompute_availability()
    ds.avail = ds._apply_physical_and_natural_missing(ds.avail_sim)
    if verbose:
        ds._print_dataset_info()


# ----------------------------------------------------------------------------- patch indexing
def patch_offsets(root, ids):
    counts = []
    for sid in ids:
        with h5py.File(os.path.join(root, "patches", f"{sid}.h5"), "r") as f:
            counts.append(int(f["img"].shape[0]))
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return {sid: int(off[i]) for i, sid in enumerate(ids)}, counts


class SplitView:
    """Plain numpy view of one split (no images)."""

    def __init__(self, ds, offsets):
        self.n = len(ds)
        self.Y = np.asarray(ds.Y, np.float32)
        self.nuc = np.asarray(ds.nuclei, np.float32)
        self.avail = np.asarray(ds.avail, bool).copy()
        self.slide_names = list(ds.slides)
        self.slide = np.asarray(ds.slide_of, np.int64)
        off = np.array([offsets[s] for s in self.slide_names], np.int64)[self.slide]
        self.row = off + np.asarray(ds.patch_row, np.int64)
        self.neigh = np.full((self.n, 6), -1, np.int64)
        for i, nb in enumerate(ds.neigh):
            if len(nb):
                self.neigh[i, :len(nb)] = off[i] + np.asarray(nb[:6], np.int64)
        self.xy = np.asarray(ds.xy, np.float32)
        self.pos = np.zeros((self.n, 2), np.int64)
        for s in range(len(self.slide_names)):
            m = self.slide == s
            lo, hi = self.xy[m].min(0), self.xy[m].max(0)
            p = (self.xy[m] - lo) / np.maximum(hi - lo, 1e-6) * (N_POS_BINS - 1)
            self.pos[m] = np.clip(np.round(p), 0, N_POS_BINS - 1).astype(np.int64)
        self.genes = list(ds.genes)


# ----------------------------------------------------------------------------- frozen encoders
ALIASES = {"optimus0": "hoptimus0", "optimus1": "hoptimus1", "h-optimus-0": "hoptimus0", "h-optimus-1": "hoptimus1",
           "vit_b_16": "vit_b16", "vitb16": "vit_b16", "dinov2_b": "dinov2"}
# name -> (timm model string, create kwargs). Normalisation comes from the model's own pretrained config.
TIMM_SPECS = {
    "hoptimus0": ("hf-hub:bioptimus/H-optimus-0", dict(init_values=1e-5, dynamic_img_size=False)),
    "hoptimus1": ("hf-hub:bioptimus/H-optimus-1", dict(init_values=1e-5, dynamic_img_size=False)),
    "uni": ("hf-hub:MahmoodLab/UNI", dict(init_values=1e-5, dynamic_img_size=True)),
    "gigapath": ("hf-hub:prov-gigapath/prov-gigapath", dict(dynamic_img_size=True)),
    "resnet50": ("resnet50", {}),
    "densenet121": ("densenet121", {}),
    "vit_b16": ("vit_base_patch16_224", {}),
    "dinov2": ("vit_base_patch14_dinov2.lvd142m", dict(img_size=224)),
}
HOPTIMUS_NORM = ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))


def canon_encoder(name):
    n = str(name).strip().lower()
    return ALIASES.get(n, n)


def load_embedder(name, device, pretrained=True):
    """Returns (callable float NCHW batch in [0,1] -> features, dim)."""
    import torch.nn.functional as F
    name = canon_encoder(name)
    if name == "dummy":                      # debugging only: fixed random conv features
        g = torch.Generator().manual_seed(0)
        W = torch.randn(8, 3, 4, 4, generator=g).to(device)

        def fn(x):                           # x float (B,3,224,224) in [0,1]
            z = F.conv2d(F.adaptive_avg_pool2d(x, 32), W, stride=4)
            return torch.cat([z.mean((2, 3)), z.std((2, 3))], 1).repeat(1, 4)   # (B, 64)
        return fn, 64
    if name not in TIMM_SPECS:
        raise ValueError(f"unknown frozen encoder '{name}'; available: {sorted(TIMM_SPECS)} + 'dummy'")
    import timm
    spec, kw = TIMM_SPECS[name]
    m = timm.create_model(spec, pretrained=pretrained, num_classes=0, **kw)
    if name.startswith("hoptimus"):
        mean, std = HOPTIMUS_NORM
    else:
        c = timm.data.resolve_model_data_config(m)
        mean, std = c["mean"], c["std"]
    m = m.eval().to(device)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    dim = m.num_features

    @torch.no_grad()
    def fn(x):
        x = (x - mean_t) / std_t
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = m(x)
        return out.float()
    return fn, dim


def _cache_dir(cfg):
    d = os.path.join(cfg.ROOT, "_cache_mm_bench")
    os.makedirs(d, exist_ok=True)
    return d


def build_embedding_table(cfg, ids, name, device):
    """(Ntotal, D) float16 table: one embedding per patch row of every slide, cached per slide."""
    fn, dim = None, None
    tabs = []
    for sid in ids:
        path = os.path.join(_cache_dir(cfg), f"emb_{name}_{sid}.npy")
        if os.path.exists(path):
            tabs.append(np.load(path))
            continue
        if fn is None:
            fn, dim = load_embedder(name, device, cfg.PRETRAINED)
        with h5py.File(os.path.join(cfg.ROOT, "patches", f"{sid}.h5"), "r") as f:
            ds, n = f["img"], f["img"].shape[0]
            out = []
            for a in range(0, n, cfg.EMBED_BATCH):
                x = torch.from_numpy(ds[a:a + cfg.EMBED_BATCH]).to(device).permute(0, 3, 1, 2).float() / 255.0
                out.append(fn(x).cpu().numpy().astype(np.float16))
        arr = np.concatenate(out)
        np.save(path, arr)
        print(f"[cache] embeddings {name} {sid}: {arr.shape}")
        tabs.append(arr)
    return np.concatenate(tabs)


class PixelStore:
    def __init__(self, arrays, offsets_list):
        self.arrays, self.off = arrays, np.asarray(offsets_list, np.int64)

    def get(self, rows):
        rows = np.asarray(rows, np.int64)
        s = np.searchsorted(self.off, rows, side="right") - 1
        out = np.empty((len(rows),) + self.arrays[0].shape[1:], np.uint8)
        for si in np.unique(s):
            m = s == si
            out[m] = self.arrays[si][rows[m] - self.off[si]]
        return out


def build_pixel_store(cfg, ids, offsets):
    import torch.nn.functional as F
    S = cfg.NATIVE_IMG_SIZE
    arrays = []
    for sid in ids:
        path = os.path.join(_cache_dir(cfg), f"px{S}_{sid}.npy")
        if not os.path.exists(path):
            with h5py.File(os.path.join(cfg.ROOT, "patches", f"{sid}.h5"), "r") as f:
                ds, n = f["img"], f["img"].shape[0]
                out = np.empty((n, 3, S, S), np.uint8)
                for a in range(0, n, 512):
                    x = torch.from_numpy(ds[a:a + 512]).permute(0, 3, 1, 2).float()
                    x = F.interpolate(x, size=(S, S), mode="bilinear", antialias=True, align_corners=False)
                    out[a:a + len(x)] = x.round().clamp(0, 255).byte().numpy()
            np.save(path, out)
            print(f"[cache] pixels {S}px {sid}: {out.shape}")
        arrays.append(np.load(path, mmap_mode="r"))
    return PixelStore(arrays, [offsets[s] for s in ids])


# ----------------------------------------------------------------------------- nuclei features
def nuc_features(counts):
    c = np.asarray(counts, np.float32)
    return np.concatenate([np.log1p(c), c / (1.0 + c.sum(1, keepdims=True))], 1).astype(np.float32)


# ----------------------------------------------------------------------------- run data
class RunData:
    """Train-set statistics + lazily built tables for one (config, run)."""

    def __init__(self, cfg, views, stores, device):
        self.cfg, self.views, self.stores, self.device = cfg, views, stores, device
        tr = views["train"]
        self.y_mu = tr.Y.mean(0).astype(np.float32)
        self.y_sd = (tr.Y.std(0) + 1e-6).astype(np.float32)
        m = tr.avail[:, 1]
        f = nuc_features(tr.nuc[m]) if m.any() else np.zeros((1, NUC_DIM), np.float32)
        self.nuc_mu, self.nuc_sd = f.mean(0), f.std(0) + 1e-6
        self.nuc0 = ((0.0 - self.nuc_mu) / self.nuc_sd).astype(np.float32)   # standardised "zero counts"
        self._nuc_std, self._tab, self._ctx, self._slice, self._stats = {}, {}, {}, {}, {}
        self._pix_mean = None
        self.tiles_cache = {}

    # ---- nuclei
    def nuc_std(self, split):
        if split not in self._nuc_std:
            v = self.views[split]
            raw = np.where(v.avail[:, 1:2], v.nuc, 0.0)           # unavailable rows are never read
            self._nuc_std[split] = ((nuc_features(raw) - self.nuc_mu) / self.nuc_sd).astype(np.float32)
        return self._nuc_std[split]

    # ---- embeddings
    def table(self, name):
        if name not in self._tab:
            t = torch.from_numpy(self.stores["emb"][name])
            gb = t.numel() * 2 / 1e9
            self._tab[name] = t.to(self.device) if gb <= self.cfg.EMB_ON_DEVICE_MAX_GB else t
        return self._tab[name]

    def _gather(self, name, rows):
        T = self.table(name)
        return T[torch.as_tensor(rows, device=T.device)].float()

    def ctx_emb(self, name, split):
        """(n, D) float16: mean embedding of the neighbouring patches (zeros if no neighbour)."""
        key = (name, split)
        if key not in self._ctx:
            v = self.views[split]
            D = self.table(name).shape[1]
            out = torch.zeros(v.n, D, dtype=torch.float16)
            for a in range(0, v.n, 4096):
                nb = v.neigh[a:a + 4096]
                valid = nb >= 0
                g = self._gather(name, np.where(valid, nb, 0).reshape(-1)).view(len(nb), 6, D)
                w = torch.from_numpy(valid.astype(np.float32)).to(g.device).unsqueeze(-1)
                out[a:a + len(nb)] = ((g * w).sum(1) / w.sum(1).clamp(min=1)).half().cpu()
            self._ctx[key] = out
        return self._ctx[key]

    def fill_stats(self, name):
        """Training means of the image / context embeddings (used by mode 'mean')."""
        if name not in self._stats:
            tr = self.views["train"]
            D = self.table(name).shape[1]
            m0, m2 = tr.avail[:, 0], tr.avail[:, 2]
            img = (self._gather(name, tr.row[m0]).mean(0).cpu() if m0.any() else torch.zeros(D))
            ctx = (self.ctx_emb(name, "train")[torch.from_numpy(m2)].float().mean(0) if m2.any()
                   else torch.zeros(D))
            self._stats[name] = (img, ctx)
        return self._stats[name]

    def slice_table(self, name, split):
        """Slide-level feature: mean embedding of the slide's spots whose image is AVAILABLE."""
        key = (name, split)
        if key not in self._slice:
            v = self.views[split]
            D = self.table(name).shape[1]
            out = torch.zeros(v.n, D)
            has = np.zeros(v.n, bool)
            for s in range(len(v.slide_names)):
                m = (v.slide == s) & v.avail[:, 0]
                if m.any():
                    mean = self._gather(name, v.row[m]).mean(0).cpu()
                    sel = v.slide == s
                    out[torch.from_numpy(sel)] = mean
                    has[sel] = True
            self._slice[key] = (out, torch.from_numpy(has))
        return self._slice[key]

    # ---- pixels
    def pix_mean(self):
        if self._pix_mean is None:
            tr = self.views["train"]
            rows = tr.row[tr.avail[:, 0]]
            if len(rows) == 0:
                S = self.cfg.NATIVE_IMG_SIZE
                self._pix_mean = np.zeros((3, S, S), np.float32)
            else:
                rng = np.random.RandomState(0)
                rows = np.sort(rng.choice(rows, min(len(rows), 1500), replace=False))
                self._pix_mean = self.stores["pix"].get(rows).astype(np.float32).mean(0)
        return self._pix_mean

    def batcher(self, split, mode):
        return Batcher(self, split, mode)


class Batcher:
    def __init__(self, rd, split, mode):
        assert mode in ("0", "mean")
        self.rd, self.split, self.mode, self.v = rd, split, mode, rd.views[split]
        self.dev = rd.device

    def make(self, idx, emb=None, pix=False, sl=False, train=False, pix_ctx=True):
        rd, v, dev, mode = self.rd, self.v, self.dev, self.mode
        idx = np.asarray(idx)
        av_np = v.avail[idx]
        av = torch.from_numpy(av_np).to(dev)
        b = {"idx": idx, "avail": av, "pos": torch.from_numpy(v.pos[idx]).to(dev)}
        b["y"] = torch.from_numpy((v.Y[idx] - rd.y_mu) / rd.y_sd).to(dev)
        # ---- nuclei (always provided)
        ns = torch.from_numpy(rd.nuc_std(self.split)[idx]).to(dev)
        fill = torch.from_numpy(rd.nuc0 if mode == "0" else np.zeros(NUC_DIM, np.float32)).to(dev)
        b["nuc"] = torch.where(av[:, 1:2], ns, fill.unsqueeze(0))
        raw = np.where(av_np[:, 1:2], v.nuc[idx], 0.0)
        tot = raw.sum(1, keepdims=True)
        b["q"] = torch.from_numpy((raw / np.maximum(tot, 1e-6)).astype(np.float32)).to(dev)
        b["qvalid"] = torch.from_numpy((av_np[:, 1] & (tot[:, 0] > 0))).to(dev)
        # ---- frozen embeddings
        if emb is not None:
            D = rd.table(emb).shape[1]
            m_img, m_ctx = rd.fill_stats(emb)
            f_img = torch.zeros(D, device=dev) if mode == "0" else m_img.to(dev)
            f_ctx = torch.zeros(D, device=dev) if mode == "0" else m_ctx.to(dev)
            img = rd._gather(emb, v.row[idx]).to(dev)
            b["img"] = torch.where(av[:, 0:1], img, f_img.unsqueeze(0))
            ctx = rd.ctx_emb(emb, self.split)[torch.from_numpy(idx)].float().to(dev)
            b["ctx"] = torch.where(av[:, 2:3], ctx, f_ctx.unsqueeze(0))
            if sl:
                tab, has = rd.slice_table(emb, self.split)
                t = tab[torch.from_numpy(idx)].to(dev)
                h = has[torch.from_numpy(idx)].to(dev).unsqueeze(1)
                b["slice"] = torch.where(h, t, f_img.unsqueeze(0))
        # ---- pixels
        if pix:
            S = rd.cfg.NATIVE_IMG_SIZE
            K = rd.cfg.DEFAULT_CTX_PATCHES
            fill_px = (np.zeros((3, S, S), np.float32) if mode == "0" else rd.pix_mean())
            fill_t = torch.from_numpy(fill_px).to(dev)
            px = torch.from_numpy(rd.stores["pix"].get(v.row[idx])).to(dev).float()
            b["pix"] = torch.where(av[:, 0].view(-1, 1, 1, 1), px, fill_t.unsqueeze(0)) / 255.0
            if not pix_ctx:
                return b
            rows = np.full((len(idx), K), -1, np.int64)
            for i, j in enumerate(idx):
                nb = v.neigh[j][v.neigh[j] >= 0]
                if train:
                    nb = np.random.permutation(nb)
                rows[i, :min(K, len(nb))] = nb[:K]
            valid = (rows >= 0) & av_np[:, 2:3]
            fillctx = ~av_np[:, 2]
            valid[fillctx, 0] = True                              # one imputed patch for missing context
            cp = torch.from_numpy(rd.stores["pix"].get(np.where(rows >= 0, rows, 0).reshape(-1))).to(dev).float()
            cp = cp.view(len(idx), K, 3, S, S)
            cp[torch.from_numpy(fillctx).to(dev), 0] = fill_t
            b["pix_ctx"] = cp / 255.0
            b["pix_valid"] = torch.from_numpy(valid).to(dev)
        return b


# ----------------------------------------------------------------------------- tiles / graphs
def make_tiles(view, size, rng=None):
    """Partition spots of each slide into spatial windows of <= size spots (recursive median split)."""
    tiles = []
    for s in range(len(view.slide_names)):
        ids = np.where(view.slide == s)[0]
        xy = view.xy[ids].astype(np.float64)
        if rng is not None:                                       # random rotation = different tiling every epoch
            a = rng.uniform(0, np.pi)
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            xy = xy @ R.T
        stack = [np.arange(len(ids))]
        while stack:
            cur = stack.pop()
            if len(cur) <= size:
                tiles.append(ids[cur])
                continue
            ext = np.ptp(xy[cur], axis=0)
            ax = int(np.argmax(ext))
            order = cur[np.argsort(xy[cur, ax], kind="stable")]
            h = len(order) // 2
            stack += [order[:h], order[h:]]
    return tiles


def spatial_adj(xy, k=4):
    """Dense symmetric kNN adjacency (+ self loops) from coordinates. xy: (n,2) torch."""
    n = xy.shape[0]
    d = torch.cdist(xy, xy)
    kk = min(k + 1, n)
    nn_idx = d.topk(kk, largest=False).indices
    A = torch.zeros(n, n, device=xy.device)
    A.scatter_(1, nn_idx, 1.0)
    A = ((A + A.t()) > 0).float()
    A.fill_diagonal_(1.0)
    return A
