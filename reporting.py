# In the name of God
"""
reporting.py
============
Metrics (MSE, MAE, PCC), Wilcoxon signed-rank tests, winner tables and the text outputs.

METRICS (test split, original log1p(CP10k) scale; computed per run):
  MSE : mean squared error over all test spots x genes                (lower is better)
  MAE : mean absolute error over all test spots x genes               (lower is better)
  SCC : same as PCC but Spearman rank correlation (robust to the skewed, zero-inflated expression) (higher is better)
  R2  : 1 - SSE/SST per gene, SST taken around the TRAIN mean (R2 > 0 <=> better than predicting the training
        mean for every spot), averaged over genes                    (higher is better)
  PCC : per test slide, Pearson correlation over spots for every gene, averaged over genes, then over
        slides (HEST-benchmark style)                                  (higher is better)
Per-gene versions of the three metrics are kept for the Wilcoxon test.

WILCOXON: for every config and metric, OURS-0 vs every other method, two-sided signed-rank test on the
G per-gene values (each averaged over the runs). 'better' says which side has the better mean.
"""
import numpy as np
from scipy import stats

METRICS = ["MSE", "MAE", "PCC", "SCC", "R2"]
HIGHER_BETTER = {"MSE": False, "MAE": False, "PCC": True, "SCC": True, "R2": True}
REF_METHOD = "OURS-0"


def _slide_corr(view, pred, Y, rank=False):
    out = []
    for s in range(len(view.slide_names)):
        m = view.slide == s
        if m.sum() < 3:
            continue
        P, T = pred[m], Y[m]
        if rank:
            P, T = stats.rankdata(P, axis=0), stats.rankdata(T, axis=0)
        p, t = P - P.mean(0), T - T.mean(0)
        den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
        out.append(np.where(den > 1e-12, (p * t).sum(0) / np.maximum(den, 1e-12), np.nan))
    return np.array(out)


def _nanmean(a, axis=None):
    a = np.asarray(a, float)
    return np.nan if a.size == 0 or not np.isfinite(a).any() else float(np.nanmean(a)) if axis is None else np.nanmean(a, axis)


def evaluate(view, pred, train_mean):
    Y = view.Y
    err = pred - Y
    gene = {"MSE": (err ** 2).mean(0), "MAE": np.abs(err).mean(0)}
    for key, rank in (("PCC", False), ("SCC", True)):
        c = _slide_corr(view, pred, Y, rank)
        gene[key] = np.nanmean(c, 0) if c.size and np.isfinite(c).any() else np.full(Y.shape[1], np.nan)
        gene[key + "_slide"] = c
    sst = ((Y - train_mean[None, :]) ** 2).sum(0)               # R2 relative to the train-mean predictor
    gene["R2"] = np.where(sst > 1e-12, 1 - (err ** 2).sum(0) / np.maximum(sst, 1e-12), np.nan)
    res = {"MSE": float((err ** 2).mean()), "MAE": float(np.abs(err).mean()),
           "PCC": _nanmean(gene["PCC_slide"], None) if gene["PCC_slide"].size else np.nan,
           "SCC": _nanmean(gene["SCC_slide"], None) if gene["SCC_slide"].size else np.nan,
           "R2": _nanmean(gene["R2"])}
    res["gene"] = {m: gene[m] for m in METRICS}
    return res


class Results:
    """results[config][method] = {'MSE': [runs], 'MAE': [...], 'PCC': [...], 'gene': {metric: [arrays]}}"""

    def __init__(self):
        self.d = {}

    def add(self, config, method, r):
        e = self.d.setdefault(config, {}).setdefault(
            method, {m: [] for m in METRICS} | {"gene": {m: [] for m in METRICS}})
        for m in METRICS:
            e[m].append(r[m])
            e["gene"][m].append(r["gene"][m])

    def methods(self, config):
        return list(self.d[config].keys())


def wilcoxon_vs_ref(res, config, metric):
    out = {}
    if REF_METHOD not in res.d[config]:
        return out
    ref = np.nanmean(np.array(res.d[config][REF_METHOD]["gene"][metric]), 0)
    for m, e in res.d[config].items():
        if m == REF_METHOD:
            continue
        o = np.nanmean(np.array(e["gene"][metric]), 0)
        ok = np.isfinite(ref) & np.isfinite(o)
        try:
            p = float(stats.wilcoxon(ref[ok], o[ok]).pvalue) if (ref[ok] != o[ok]).any() else 1.0
        except ValueError:
            p = float("nan")
        mr, mo = np.mean(ref[ok]), np.mean(o[ok])
        better = REF_METHOD if (mr > mo) == HIGHER_BETTER[metric] else m
        out[m] = (p, better)
    return out


def summary_text(res, configs):
    L = []
    for c in configs:
        if c not in res.d:
            continue
        L.append("=" * 100)
        L.append(f"CONFIG '{c}'  (mean +- std over runs; * = winner)")
        L.append("=" * 100)
        names = res.methods(c)
        win = {}
        for m in METRICS:
            means = {n: float(np.nanmean(res.d[c][n][m])) for n in names}
            win[m] = (max if HIGHER_BETTER[m] else min)(means, key=means.get)
        L.append(f"{'method':34s}" + "".join(f"{m:>22s}" for m in METRICS))
        for n in names:
            row = f"{n:34s}"
            for m in METRICS:
                v = np.array(res.d[c][n][m], float)
                row += f"{('*' if win[m] == n else ' ') + f'{np.nanmean(v):.4f} +- {np.nanstd(v):.4f}':>22s}"
            L.append(row)
        L.append("Winner: " + " | ".join(f"{m}: {win[m]}" for m in METRICS))
        for m in METRICS:
            w = wilcoxon_vs_ref(res, c, m)
            if not w:
                L.append(f"[Wilcoxon {m}] {REF_METHOD} not among the selected methods")
                continue
            L.append(f"[Wilcoxon signed-rank on per-gene {m}, {REF_METHOD} vs method]")
            for n, (p, better) in w.items():
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                L.append(f"    vs {n:32s} p={p:.3g} {sig:3s} better mean: {better}")
        L.append("")
    # overall: average rank across configs and metrics
    ranks = {}
    for c in configs:
        if c not in res.d:
            continue
        for m in METRICS:
            names = res.methods(c)
            vals = np.array([np.nanmean(res.d[c][n][m]) for n in names])
            order = np.argsort(-vals if HIGHER_BETTER[m] else vals)
            for r, i in enumerate(order):
                ranks.setdefault(names[i], []).append(r + 1)
    if ranks:
        L.append("=" * 100)
        L.append("OVERALL (average rank over all configs and metrics, 1 = best)")
        for n, r in sorted(ranks.items(), key=lambda kv: np.mean(kv[1])):
            L.append(f"    {n:34s} {np.mean(r):.2f}")
        L.append(f"OVERALL WINNER: {min(ranks, key=lambda k: np.mean(ranks[k]))}")
    return "\n".join(L)


def write_outputs(res, configs, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    for m in METRICS:
        with open(os.path.join(out_dir, f"output-{m}.txt"), "w") as f:
            f.write("In the name of God\n")
            for c in configs:
                if c not in res.d:
                    continue
                f.write(f"'{c}'\n")
                for n in res.methods(c):
                    f.write(f"{n}\n")
                    for v in res.d[c][n][m]:
                        f.write(f"{v:.6g}\n")
    with open(os.path.join(out_dir, "output-summary.txt"), "w") as f:
        f.write("In the name of God\n" + summary_text(res, configs) + "\n")
