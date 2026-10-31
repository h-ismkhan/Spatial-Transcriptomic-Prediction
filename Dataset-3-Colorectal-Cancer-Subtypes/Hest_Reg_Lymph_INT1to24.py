"""
Hest_Reg_Lymph_INT1to24
=======================
Regression reader for the HEST subset INT1-INT24 (Visium, ccRCC tumours, 24 patients),
with a missing-modality simulator that follows the CMU-MOSI reader (MOSIDatasetRegression).

NOTE ON THE NAME: the HEST metadata table lists these slides as KIDNEY (ccRCC); only the
per-sample JSON files say "Lymph node" (see Section 3, check C9 of the report). The class
name is kept as requested.

SAMPLE   : one Visium SPOT (not a slide).
MODALITIES (all derived from the downloaded files):
  image   : H&E patch of the spot                      <- patches/<ID>.h5   (like "text")
  nuclei  : counts of CellViT nuclei types in the spot <- cellvit_seg/<ID>_cellvit_seg.geojson.zip
  context : H&E patches of the (up to 6) neighbouring spots <- patches/<ID>.h5 (like "video" frames)
LABEL    : log1p(counts per 10,000) of the n_genes most variable genes <- st/<ID>.h5ad

EMPTY SPOTS: a spot can lose all modalities (simple mode by chance; complex mode when the drawn
subset contains only modalities that physically do not exist for that spot/slide). With
empty_policy='keep' (default) they stay empty; with 'restore_image' their image is restored.

MISSING-MODALITY CONFIG STRINGS (same logic as the MOSI reader, parts separated by '_'):
  simple  : '<X>_image_<Y>_nuclei_<Z>_context'   e.g. '100_image_20_nuclei_20_context'
            each modality is kept independently with probability X%, Y%, Z% (so, exactly as in
            the MOSI reader, a spot can end up with NO modality).
  complex : 'complex_<all>_<I>_<N>_<C>_<IN>_<IC>_<NC>'   (seven numbers, sum = 100)
            all three / image only / nuclei only / context only / image+nuclei /
            image+context / nuclei+context. Each spot gets exactly ONE of these subsets,
            drawn with one uniform random number per spot (cumulative thresholds), so every
            spot keeps at least one modality.

Requires: numpy, torch, h5py, anndata (0.10.x is enough, scanpy is NOT needed), scipy.
"""
import os
import json
import math
import zipfile
import warnings
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree

NUCLEI_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Epithelial", "Dead"]
DEFAULT_IDS = [f"ZEN{i}" for i in range(36, 50)]
# slides whose nuclei segmentation is incomplete (report, check C14); treated as "no nuclei modality"
DEFAULT_NATURAL_MISSING_NUCLEI = ("INT20", "INT22", "INT23")
# Visium array layout: neighbours of spot (row, col)
HEX_OFFSETS = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
# order of the seven subsets of the complex pattern (same order as MOSI: T, A, V, TA, TV, AV)
COMPLEX_SUBSETS = [(True, True, True), (True, False, False), (False, True, False),
                   (False, False, True), (True, True, False), (True, False, True),
                   (False, True, True)]


def _decode(b):
    return b.decode() if isinstance(b, (bytes, np.bytes_)) else str(b)


class Hest_Reg_Lymph_INT1to24(Dataset):
    def __init__(self, root="hest_data", split="train", sample_ids=None,
                 split_file=None, split_seed=42, test_frac=0.2, val_frac=0.1,
                 missing_config="100_image_100_nuclei_100_context", seed=42,
                 n_genes=50, genes=None, min_detect_frac=0.1, hvg_within_group=True,
                 apply_natural_missing=True, empty_policy="keep",
                 natural_missing_nuclei=DEFAULT_NATURAL_MISSING_NUCLEI,
                 cache_dir=None, verbose=True):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val' or 'test'")
        self.root = root
        self.split = split
        self.missing_config = missing_config
        self.seed = seed
        self.verbose = verbose
        self.apply_natural_missing = apply_natural_missing
        if empty_policy not in ("keep", "restore_image"):
            raise ValueError("empty_policy must be 'keep' or 'restore_image'")
        self.empty_policy = empty_policy
        self.natural_missing_nuclei = set(natural_missing_nuclei)
        self.cache_dir = cache_dir or os.path.join(root, "_cache_hest_reg")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.sample_ids = list(sample_ids) if sample_ids is not None else list(DEFAULT_IDS)
        self._h5 = {}
        self._h5_pid = os.getpid()

        self._check_files()
        self.meta = {sid: self._read_json(sid) for sid in self.sample_ids}
        self.slide_info = {sid: self._slide_info(sid) for sid in self.sample_ids}

        # ---- patient-level split (same idea as the MOSI split json) ----
        self.splits = self._get_splits(split_file, split_seed, test_frac, val_frac)
        self.slides = sorted(self.splits[split], key=self.sample_ids.index)
        if not self.slides:
            raise ValueError(f"split '{split}' contains no slides")

        # ---- genes: common to all slides, then selected on TRAIN slides only ----
        self.common_genes = self._common_genes()
        self._common_index = {g: i for i, g in enumerate(self.common_genes)}
        if genes is None:
            if split != "train":
                raise ValueError("val/test datasets need genes=train_dataset.genes "
                                 "(the gene panel must be selected on training slides only)")
            genes = self._select_genes(self.splits["train"], n_genes, min_detect_frac,
                                       hvg_within_group)
        missing = [g for g in genes if g not in self._common_index]
        if missing:
            raise ValueError(f"genes not present in all slides: {missing[:5]}")
        self.genes = list(genes)
        self._gene_idx = np.array([self._common_index[g] for g in self.genes])

        # ---- per-spot data of this split ----
        self._build_split_data()

        # ---- missing-modality simulation (MOSI logic) ----
        self._parse_missing_config()
        self.avail_sim = self._precompute_availability()
        self.avail = self._apply_physical_and_natural_missing(self.avail_sim)
        if verbose:
            self._print_dataset_info()

    # ------------------------------------------------------------------ files / metadata
    def _check_files(self):
        problems = []
        for sid in self.sample_ids:
            for sub, name in [("st", f"{sid}.h5ad"), ("patches", f"{sid}.h5"),
                              ("metadata", f"{sid}.json")]:
                if not os.path.exists(os.path.join(self.root, sub, name)):
                    problems.append(f"{sub}/{name}")
        if problems:
            raise FileNotFoundError("missing files under '%s': %s" % (self.root, problems[:10]))

    def _read_json(self, sid):
        with open(os.path.join(self.root, "metadata", f"{sid}.json")) as f:
            txt = f.read()
        return json.loads(txt.replace("NaN", "null"))   # the HEST json files contain bare NaN

    def _slide_info(self, sid):
        m = self.meta[sid]
        sub = str(m.get("subseries") or "").lower()
        group = "frozen" if "frozen" in sub else ("ffpe" if "ffpe" in sub else "unknown")
        patient = m.get("patient")
        if patient in (None, "", "None"):
            warnings.warn(f"{sid}: no patient label; the slide is treated as its own patient")
            key = f"slide:{sid}"
        else:
            key = f"{m.get('dataset_title')}|{patient}"
        return {"patient_key": key, "patient": patient, "group": group}

    def _get_splits(self, split_file, seed, test_frac, val_frac):
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                splits = json.load(f)
            used = set(sum((splits[k] for k in ("train", "val", "test")), []))
            unknown = used - set(self.sample_ids)
            if unknown:
                raise ValueError(f"split file contains slides not in sample_ids: {sorted(unknown)}")
            return splits
        splits = self.make_patient_splits(self.slide_info, test_frac, val_frac, seed)
        if split_file:
            with open(split_file, "w") as f:
                json.dump(splits, f, indent=1)
        return splits

    @staticmethod
    def make_patient_splits(slide_info, test_frac=0.2, val_frac=0.1, seed=42):
        """Split by PATIENT, stratified by preservation group (frozen / ffpe)."""
        rng = np.random.RandomState(seed)
        by_group = {}
        for sid, inf in slide_info.items():
            by_group.setdefault(inf["group"], {}).setdefault(inf["patient_key"], []).append(sid)
        splits = {"train": [], "val": [], "test": []}
        for group in sorted(by_group):
            patients = sorted(by_group[group])
            rng.shuffle(patients)
            n = len(patients)
            n_test = int(math.ceil(test_frac * n)) if n >= 3 else 0
            n_val = max(1, int(round(val_frac * n))) if (val_frac > 0 and n >= 4) else 0
            if n - n_test - n_val < 1:
                n_val = 0
            parts = {"test": patients[:n_test], "val": patients[n_test:n_test + n_val],
                     "train": patients[n_test + n_val:]}
            for name, plist in parts.items():
                for p in plist:
                    splits[name].extend(by_group[group][p])
        return splits

    # ------------------------------------------------------------------ genes / expression
    def _common_genes(self):
        common = None
        first = None
        for sid in self.sample_ids:
            a = ad.read_h5ad(os.path.join(self.root, "st", f"{sid}.h5ad"), backed="r")
            names = list(a.var_names)
            try:
                a.file.close()
            except Exception:
                pass
            if common is None:
                common, first = set(names), names
            else:
                common &= set(names)
        return [g for g in first if g in common]

    def _read_normalised(self, sid):
        """Expression of one slide restricted to the common genes, spots with a patch only.
        Normalisation: counts / (sum over COMMON genes) * 1e4, then log1p."""
        a = ad.read_h5ad(os.path.join(self.root, "st", f"{sid}.h5ad"))
        idx = a.var_names.get_indexer(self.common_genes)
        X = a.X
        X = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(np.asarray(X))
        X = X.astype(np.float32)[:, idx]
        d = X.data[:1_000_000]
        if d.size and not np.allclose(d, np.round(d)):
            warnings.warn(f"{sid}: expression values are not integers (not raw counts?)")
        tot = np.asarray(X.sum(axis=1)).ravel()
        scale = np.where(tot > 0, 1e4 / np.maximum(tot, 1e-12), 0.0).astype(np.float32)
        X = sparse.diags(scale) @ X
        X = X.tocsr()
        X.data = np.log1p(X.data)
        barcodes = np.asarray(a.obs_names)
        with h5py.File(os.path.join(self.root, "patches", f"{sid}.h5"), "r") as f:
            pb = np.array([_decode(x) for x in np.asarray(f["barcode"][:]).reshape(-1)])
        row_of = {b: i for i, b in enumerate(pb)}
        keep = np.array([b in row_of for b in barcodes])
        return {"obs": a.obs, "barcodes": barcodes, "keep": keep, "X": X, "row_of": row_of}

    def _select_genes(self, train_slides, n_genes, min_detect_frac, within_group):
        G = len(self.common_genes)
        acc = {}   # group -> [n, sum, sumsq, nnz]
        for sid in train_slides:
            d = self._read_normalised(sid)
            X = d["X"][d["keep"]]
            if X.shape[0] == 0:
                continue
            g = self.slide_info[sid]["group"] if within_group else "all"
            s = acc.setdefault(g, [0, np.zeros(G), np.zeros(G), np.zeros(G)])
            s[0] += X.shape[0]
            s[1] += np.asarray(X.sum(axis=0)).ravel()
            s[2] += np.asarray(X.multiply(X).sum(axis=0)).ravel()
            s[3] += X.getnnz(axis=0)
        N = sum(v[0] for v in acc.values())
        ss_within = sum(v[2] - v[1] ** 2 / v[0] for v in acc.values())   # within-group sum of squares
        var = ss_within / N
        detect = sum(v[3] for v in acc.values()) / N
        var = np.where(detect >= min_detect_frac, var, -np.inf)
        if np.isfinite(var).sum() < n_genes:
            raise ValueError("fewer genes than n_genes pass the detection filter")
        order = np.argsort(-var, kind="stable")[:n_genes]
        if self.verbose:
            print(f"[genes] selected {n_genes} of {G} common genes on {N} training spots "
                  f"(within-group variance: {within_group})")
        return [self.common_genes[i] for i in order]

    # ------------------------------------------------------------------ nuclei / context
    def _nuclei_counts(self, sid, xy_all, radius):
        path = os.path.join(self.root, "cellvit_seg", f"{sid}_cellvit_seg.geojson.zip")
        if not os.path.exists(path):
            return None
        cache = os.path.join(self.cache_dir, f"nuclei_{sid}_r{int(round(radius))}.npy")
        if os.path.exists(cache):
            c = np.load(cache)
            if c.shape == (len(xy_all), len(NUCLEI_TYPES)):
                return c
        with zipfile.ZipFile(path) as z:
            with z.open(z.namelist()[0]) as f:
                gj = json.load(f)
        feats = gj["features"] if isinstance(gj, dict) else gj
        pts, tid, unknown = [], [], Counter()
        for ft in feats:
            t = ft["properties"]["classification"]["name"]
            if t not in NUCLEI_TYPES:
                unknown[t] += 1
                continue
            g = ft["geometry"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            for poly in polys:
                pts.append(np.asarray(poly[0], float).mean(axis=0))   # (x, y) of the nucleus
                tid.append(NUCLEI_TYPES.index(t))
        if unknown:
            warnings.warn(f"{sid}: unknown nuclei types ignored: {dict(unknown)}")
        counts = np.zeros((len(xy_all), len(NUCLEI_TYPES)), dtype=np.int16)
        if pts:
            pts, tid = np.asarray(pts), np.asarray(tid)
            hits = cKDTree(pts).query_ball_point(xy_all, radius)
            for i, h in enumerate(hits):
                if h:
                    counts[i] = np.bincount(tid[h], minlength=len(NUCLEI_TYPES))
        np.save(cache, counts)
        return counts

    @staticmethod
    def _neighbours(rc, patch_rows):
        """Patch rows of the hexagonal neighbours (Visium array layout) of every spot."""
        pos = {(int(r), int(c)): i for i, (r, c) in enumerate(rc)}
        out = []
        for r, c in rc:
            idx = [pos[(int(r) + dr, int(c) + dc)] for dr, dc in HEX_OFFSETS
                   if (int(r) + dr, int(c) + dc) in pos]
            out.append(np.sort(patch_rows[idx]) if idx else np.zeros(0, dtype=np.int64))
        return out

    def _build_split_data(self):
        names, slide_of, rows, Y, NU, NEI, XY = [], [], [], [], [], [], []
        self.slide_stats, self.nuclei_file_ok = {}, {}
        for si, sid in enumerate(self.slides):
            d = self._read_normalised(sid)
            keep, obs = d["keep"], d["obs"]
            m = self.meta[sid]
            px = m.get("pixel_size_um_estimated") or m.get("pixel_size_um_embedded")
            radius = m["spot_diameter"] / 2.0 / px
            xy_all = obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)  # (x=col, y=row)
            rc_all = obs[["array_row", "array_col"]].to_numpy(int)
            n_keep = int(keep.sum())
            self.slide_stats[sid] = {"spots": len(keep), "kept": n_keep, "dropped_no_patch": int((~keep).sum())}
            bk = d["barcodes"][keep]
            prow = np.array([d["row_of"][b] for b in bk], dtype=np.int64)
            Xk = d["X"][keep][:, self._gene_idx]
            nuc = self._nuclei_counts(sid, xy_all, radius)
            self.nuclei_file_ok[sid] = nuc is not None
            nuc = (nuc[keep] if nuc is not None else np.zeros((n_keep, len(NUCLEI_TYPES)), np.int16))
            nei = self._neighbours(rc_all[keep], prow)
            self._check_hex_geometry(sid, rc_all[keep], xy_all[keep], m, px)
            names += [f"{sid}:{b}" for b in bk]
            slide_of += [si] * n_keep
            rows.append(prow)
            Y.append(Xk.toarray().astype(np.float32))
            NU.append(nuc.astype(np.float32))
            NEI += nei
            XY.append(xy_all[keep])
        self.names = names
        self.slide_of = np.asarray(slide_of, dtype=np.int64)
        self.patch_row = np.concatenate(rows)
        self.Y = np.concatenate(Y)
        self.nuclei = np.concatenate(NU)
        self.neigh = NEI
        self.xy = np.concatenate(XY)
        with h5py.File(os.path.join(self.root, "patches", f"{self.slides[0]}.h5"), "r") as f:
            self.patch_shape = tuple(f["img"].shape[1:])   # (H, W, 3)

    def _check_hex_geometry(self, sid, rc, xy, meta, px):
        """Sanity check of the assumed Visium neighbour layout (same-row neighbour at col+2)."""
        if getattr(self, "_geom_checked", False):
            return
        self._geom_checked = True
        pos = {(int(r), int(c)): i for i, (r, c) in enumerate(rc)}
        dist = [np.linalg.norm(xy[i] - xy[pos[(r, c + 2)]]) for (r, c), i in pos.items()
                if (r, c + 2) in pos][:2000]
        if not dist:
            warnings.warn(f"{sid}: no (row, col+2) neighbours found; check array_row/array_col")
            return
        expected = meta["inter_spot_dist"] / px
        ratio = float(np.median(dist)) / expected
        if self.verbose:
            print(f"[geometry] {sid}: median same-row neighbour distance = {np.median(dist):.0f}px "
                  f"(expected {expected:.0f}px, ratio {ratio:.2f})")
        if not 0.85 <= ratio <= 1.15:
            warnings.warn("neighbour distance differs from inter_spot_dist: the assumed Visium "
                          "array layout may be wrong for this slide")

    # ------------------------------------------------------------------ missing modalities (MOSI logic)
    def _parse_missing_config(self):
        parts = self.missing_config.split("_")
        if parts[0] == "complex":
            nums = parts[1:]
            if len(nums) != 7:
                raise ValueError("complex config needs 7 numbers: all, I, N, C, IN, IC, NC")
            self.complex_ratios = [int(x) / 100.0 for x in nums]
            if abs(sum(self.complex_ratios) - 1.0) > 0.01:
                raise ValueError(f"Complex ratios must sum to 1.0, got {sum(self.complex_ratios)}")
            self.complex_mode = True
        else:
            if len(parts) != 6 or parts[1] != "image" or parts[3] != "nuclei" or parts[5] != "context":
                raise ValueError("simple config must look like '100_image_20_nuclei_20_context'")
            self.complex_mode = False
            self.ratios = [int(parts[0]) / 100.0, int(parts[2]) / 100.0, int(parts[4]) / 100.0]

    def _precompute_availability(self):
        rng = np.random.RandomState(self.seed)
        n = len(self.names)
        avail = np.zeros((n, 3), dtype=bool)
        if self.complex_mode:
            cum = np.cumsum(self.complex_ratios)
            for i in range(n):
                r = rng.random_sample()
                k = int(np.searchsorted(cum, r, side="right"))
                avail[i] = COMPLEX_SUBSETS[min(k, 6)]
        else:
            for i in range(n):
                avail[i] = (rng.random_sample() < self.ratios[0],
                            rng.random_sample() < self.ratios[1],
                            rng.random_sample() < self.ratios[2])
        return avail

    def _apply_physical_and_natural_missing(self, avail_sim):
        """Beyond the simulation: a modality that does not exist cannot be 'available'.
        - nuclei: no nuclei file, or (if apply_natural_missing) a slide with incomplete segmentation
        - context: spot without any neighbouring spot that has a patch"""
        avail = avail_sim.copy()
        for i in range(len(self.names)):
            sid = self.slides[self.slide_of[i]]
            if (not self.nuclei_file_ok[sid]) or (self.apply_natural_missing and sid in self.natural_missing_nuclei):
                avail[i, 1] = False
            if len(self.neigh[i]) == 0:
                avail[i, 2] = False
        if self.empty_policy == "restore_image":
            # spots left with no modality at all (e.g. 'nuclei only' on a slide without nuclei)
            # get their image back; with 'keep' they stay empty, as in the MOSI simple mode
            avail[~avail.any(axis=1), 0] = True
        return avail

    # ------------------------------------------------------------------ info
    def _print_dataset_info(self):
        n = len(self.names)
        print("\n" + "=" * 80)
        print(f"Hest_Reg_Lymph_INT1to24: {self.split} split")
        print(f"  Config: {self.missing_config} | Seed: {self.seed}")
        print(f"  Slides ({len(self.slides)}): {self.slides}")
        pat = {self.slide_info[s]['patient_key'] for s in self.slides}
        grp = Counter(self.slide_info[s]['group'] for s in self.slides)
        print(f"  Patients: {len(pat)} | groups: {dict(grp)}")
        dropped = sum(v['dropped_no_patch'] for v in self.slide_stats.values())
        print(f"  Spots: {n} (dropped for missing patch: {dropped})")
        print(f"  Genes: {len(self.genes)} | label mean {self.Y.mean():.3f}, std {self.Y.std():.3f}")
        for title, av in (("Simulated availability (as configured)", self.avail_sim),
                          ("Final availability (after physical/natural missingness)", self.avail)):
            print(f"  {title}:")
            counts = Counter(map(tuple, av))
            for (i, nu, c), cnt in sorted(counts.items(), key=lambda x: -x[1]):
                mods = [m for m, f in zip(("Image", "Nuclei", "Context"), (i, nu, c)) if f] or ["None"]
                print(f"    {'+'.join(mods):<24s}: {cnt:>6d} ({100 * cnt / n:5.1f}%)")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------ Dataset API
    def __len__(self):
        return len(self.names)

    def _patch_file(self, sid):
        if self._h5_pid != os.getpid():          # new process (DataLoader worker): reopen
            self._h5, self._h5_pid = {}, os.getpid()
        if sid not in self._h5:
            self._h5[sid] = h5py.File(os.path.join(self.root, "patches", f"{sid}.h5"), "r")
        return self._h5[sid]

    def __getitem__(self, idx):
        sid = self.slides[self.slide_of[idx]]
        has_image, has_nuclei, has_context = (bool(v) for v in self.avail[idx])
        H, W, _ = self.patch_shape
        f = self._patch_file(sid)

        if has_image:
            img = np.ascontiguousarray(f["img"][int(self.patch_row[idx])].transpose(2, 0, 1))
            image = torch.from_numpy(img)
        else:
            image = torch.zeros((3, H, W), dtype=torch.uint8)

        nuclei = (torch.from_numpy(self.nuclei[idx]) if has_nuclei
                  else torch.zeros(len(NUCLEI_TYPES), dtype=torch.float32))

        if has_context:
            ctx = f["img"][[int(r) for r in self.neigh[idx]]]            # (k, H, W, 3), rows ascending
            context = torch.from_numpy(np.ascontiguousarray(ctx.transpose(0, 3, 1, 2)))
        else:
            context = torch.zeros((0, 3, H, W), dtype=torch.uint8)

        return {
            "name": self.names[idx],
            "slide": sid,
            "patient": self.slide_info[sid]["patient"],
            "group": self.slide_info[sid]["group"],
            "xy": torch.from_numpy(self.xy[idx].astype(np.float32)),
            "image": image, "nuclei": nuclei, "context": context,
            "label": torch.from_numpy(self.Y[idx]),
            "has_image": has_image, "has_nuclei": has_nuclei, "has_context": has_context,
        }

    @staticmethod
    def collate_fn(batch):
        """Context has a different number of patches per spot, so it stays a list."""
        out = {k: [b[k] for b in batch] for k in ("name", "slide", "patient", "group", "context")}
        for k in ("xy", "image", "nuclei", "label"):
            out[k] = torch.stack([b[k] for b in batch])
        for k in ("has_image", "has_nuclei", "has_context"):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.bool)
        return out


if __name__ == "__main__":
    root = "hest_data"
    split_file = os.path.join(root, "splits_INT1to24.json")   # saved once, then reused

    train = Hest_Reg_Lymph_INT1to24(root, split="train", split_file=split_file,
                                    missing_config="100_image_20_nuclei_20_context", seed=42)
    val = Hest_Reg_Lymph_INT1to24(root, split="val", split_file=split_file, genes=train.genes,
                                  missing_config="100_image_20_nuclei_20_context", seed=42)
    test = Hest_Reg_Lymph_INT1to24(root, split="test", split_file=split_file, genes=train.genes,
                                   missing_config="complex_20_30_20_10_15_5_0", seed=42)
    for i in range(3):
        s = train[i]
        print(s["name"], "| image", tuple(s["image"].shape), "| nuclei", s["nuclei"].tolist(),
              "| context", tuple(s["context"].shape), "| label", tuple(s["label"].shape),
              "| has:", s["has_image"], s["has_nuclei"], s["has_context"])