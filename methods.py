# In the name of God
"""
methods.py
==========
Self-contained re-implementations (nothing is imported from the competitors' repositories).

  OURS      : Method-1: OurEncoder -> LayerNorm-Linear(1536,1024)-GELU-Dropout-Linear(1024,G), MSE on z-scored targets.
  BLEEP     : CLIP-style contrastive image<->expression embedding (soft targets, temperature), prediction =
              mean expression of the top-k nearest TRAIN spots in the shared embedding space.
  MCLSTEXP  : contrastive image<->expression where the expression branch adds learned x/y position
              embeddings and runs self-attention blocks over the spots of the batch; same retrieval inference.
  GHIST     : (spot-level adaptation) multitask: expression = softmax-weighted learnable cell-type reference
              profiles + offset conditioned on the predicted neighbourhood composition; auxiliary KL loss on
              the nuclei-type composition (CellViT counts). Cell segmentation / single-cell resolution is NOT used.
  STORM     : (inductive adaptation) CP-style factorisation  y_i ~ C s_i  with free spot latents s_i on the
              training spots, regularised by R0 (magnitude), R1 (alignment of s_i with the image/multimodal
              features through U), R2 (spatial Laplacian smoothness), R3 (gene Laplacian, built from
              training co-expression because STRING/PPI is not available offline). Loss weights are set
              adaptively at start (lambda_i = P_i * L_mse / R_i). Test spots: s = smooth(features @ U).
  FMH2ST    : dual branch: foundation-feature branch (slice-aware features, spatial + learnable feature graph,
              GAT, attention fusion) and spot-specific branch (conv/CBAM or our encoder -> Transformer with
              position embeddings -> GAT), attention fusion, collaborative loss (L_U + L_F + L_S with distillation).
              Works on windows (tiles) of neighbouring spots.
Variants: 'default' = the competitor's own image encoder (pixels or its own frozen features),
          'our_encoder' = the encoder of OURS replaces the image encoder (everything else unchanged).
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from data_utils import make_tiles, spatial_adj, N_POS_BINS
from encoders import OurEncoder, NativeSource, RawConcatSource, PixelBackbone

METHOD_NAMES = ["OURS", "OURS-BRANCHED", "BLEEP", "GHIST", "MCLSTEXP", "STORM", "FMH2ST"]


# ----------------------------------------------------------------------------- helpers
def pcc_mean(pred, truth):
    """Mean over genes of the Pearson correlation computed over spots (nan-safe)."""
    p, t = pred - pred.mean(0), truth - truth.mean(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    r = np.where(den > 1e-12, (p * t).sum(0) / np.maximum(den, 1e-12), np.nan)
    return float(np.nanmean(r)) if np.isfinite(r).any() else -1.0


def finite_guard(a, where=""):
    bad = ~np.isfinite(a)
    if bad.any():
        print(f"      [warn] {int(bad.any(1).sum())} spot(s) with non-finite predictions ({where}) -> set to 0 (train mean)")
        a = np.where(bad, 0.0, a)
    return a


class ProjectionHead(nn.Module):           # BLEEP / mclSTExp projection head
    def __init__(self, d_in, d=256, dropout=0.1):
        super().__init__()
        self.p, self.fc, self.drop, self.ln = nn.Linear(d_in, d), nn.Linear(d, d), nn.Dropout(dropout), nn.LayerNorm(d)

    def forward(self, x):
        p = self.p(x)
        return self.ln(self.drop(self.fc(F.gelu(p))) + p)


def soft_ce(logits, targets):
    return (-targets * F.log_softmax(logits, -1)).sum(-1)


class DenseGAT(nn.Module):
    def __init__(self, d_in, d_out, heads=4, dropout=0.2):
        super().__init__()
        self.h, self.hd = heads, d_out // heads
        self.W = nn.Linear(d_in, heads * self.hd, bias=False)
        self.a_s = nn.Parameter(torch.randn(heads, self.hd) * 0.1)
        self.a_d = nn.Parameter(torch.randn(heads, self.hd) * 0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        N = x.shape[0]
        h = self.W(x).view(N, self.h, self.hd)
        e = F.leaky_relu((h * self.a_s).sum(-1).unsqueeze(1) + (h * self.a_d).sum(-1).unsqueeze(0), 0.2)  # (N,N,H)
        e = e.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        a = self.drop(torch.softmax(e, 1))
        return F.elu(torch.einsum("ijh,jhd->ihd", a, h).reshape(N, -1))


class AttnAgg(nn.Module):                  # FmH2ST attention aggregation of two representations
    def __init__(self, d):
        super().__init__()
        self.w, self.u = nn.Linear(d, d), nn.Linear(d, 1, bias=False)

    def forward(self, a, b):
        e = torch.stack([a, b], 1)
        al = torch.softmax(self.u(torch.tanh(self.w(e))), 1)
        return (al * e).sum(1)


def make_source(kind, native_kind, rd, cfg, enc=None, fusion=None):
    """Returns (source module, emb_name, use_pix)."""
    if kind == "our":
        name = enc or cfg.FOUNDATION_ENCODER
        return OurEncoder(rd.table(name).shape[1], fusion=fusion or cfg.FUSION), name, False
    return NativeSource(native_kind, cfg.PRETRAINED), None, True


# ----------------------------------------------------------------------------- base class
class TorchMethod:
    use_slice = False
    tile_mode = False
    need_pix_ctx = True

    def __init__(self, rd, mode, variant, seed, cfg, hp, device, enc=None, fusion=None):
        self.rd, self.mode, self.variant, self.cfg, self.hp, self.dev = rd, mode, variant, cfg, hp, device
        self.fusion = fusion                # fusion of the OURS encoder (None -> cfg.FUSION)
        self.enc = enc                      # frozen encoder of the OURS family (None -> cfg.FOUNDATION_ENCODER)
        self.G = rd.views["train"].Y.shape[1]
        self.bt = {s: rd.batcher(s, mode) for s in ("train", "val", "test")}
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.emb_name, self.use_pix = None, False
        self.build()
        self.net.to(device)
        self.amp = device.type == "cuda" and self.use_pix
        self.amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # --- to be implemented
    def build(self):
        raise NotImplementedError

    def loss(self, b):
        raise NotImplementedError

    def forward_z(self, b):
        raise NotImplementedError

    # --- batching
    def mk(self, split, idx, train=False):
        b = self.bt[split].make(idx, emb=self.emb_name, pix=self.use_pix, sl=self.use_slice, train=train,
                                pix_ctx=self.need_pix_ctx)
        if self.tile_mode:
            b["xy"] = torch.from_numpy(self.rd.views[split].xy[idx]).to(self.dev)
        return b

    def train_batches(self):
        n = self.rd.views["train"].n
        perm = self.rng.permutation(n)
        bs = self.hp["batch_size"]
        return [perm[i:i + bs] for i in range(0, n, bs) if len(perm[i:i + bs]) > 1]

    def eval_batches(self, split):
        n, bs = self.rd.views[split].n, self.hp["batch_size"]
        return [np.arange(i, min(i + bs, n)) for i in range(0, n, bs)]

    # --- optimisation
    def param_groups(self):
        lr, wd = self.hp["lr"], self.hp["weight_decay"]
        src = getattr(self, "src", None)
        if src is not None and getattr(src, "pretrained_backbone", False) and self.cfg.PRETRAINED:
            enc = list(src.encoder_params())
            ids = {id(p) for p in enc}
            rest = [p for p in self.net.parameters() if id(p) not in ids]
            return [{"params": enc, "lr": lr * self.hp["enc_lr_mult"], "weight_decay": wd},
                    {"params": rest, "lr": lr, "weight_decay": wd}]
        return [{"params": list(self.net.parameters()), "lr": lr, "weight_decay": wd}]

    def _step_loss(self, b):
        with torch.autocast(self.dev.type, dtype=self.amp_dtype, enabled=self.amp):
            return self.loss(b)

    def val_score(self):
        z = self.predict_z("val")
        return pcc_mean(z, (self.rd.views["val"].Y - self.rd.y_mu) / self.rd.y_sd)

    def fit(self):
        hp = self.hp
        opt = torch.optim.AdamW(self.param_groups())
        E, W = hp["epochs"], hp["warmup"]
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda e: (e + 1) / W if e < W else max(0.01, 0.5 * (1 + math.cos(math.pi * (e - W) / max(1, E - W)))))
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)
        best, wait, state = -1e9, 0, None
        for ep in range(E):
            self.net.train()
            batches = self.train_batches()
            if hp.get("max_steps"):
                batches = batches[:hp["max_steps"]]
            tot = 0.0
            for idx in batches:
                b = self.mk("train", idx, train=True)
                opt.zero_grad(set_to_none=True)
                loss = self._step_loss(b)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                tot += float(loss.detach())
            sched.step()
            sc = self.val_score()
            if self.cfg.VERBOSE_TRAIN:
                print(f"      ep {ep + 1:3d} loss {tot / max(1, len(batches)):.4f} val-PCC {sc:.4f}")
            if sc > best + 1e-6:
                best, wait, state = sc, 0, copy.deepcopy(self.net.state_dict())
            else:
                wait += 1
                if wait >= hp["patience"]:
                    break
        if state is not None:
            self.net.load_state_dict(state)
        return best

    # --- inference
    @torch.no_grad()
    def predict_z(self, split):
        self.net.eval()
        n = self.rd.views[split].n
        out = np.zeros((n, self.G), np.float32)
        for idx in self.eval_batches(split):
            b = self.mk(split, idx)
            with torch.autocast(self.dev.type, dtype=self.amp_dtype, enabled=self.amp):
                out[idx] = self.forward_z(b).float().cpu().numpy()
        return out

    def predict(self, split):
        return self.predict_z(split) * self.rd.y_sd + self.rd.y_mu


# ----------------------------------------------------------------------------- OURS
class OursMethod(TorchMethod):
    def build(self):
        self.src, self.emb_name, self.use_pix = make_source("our", None, self.rd, self.cfg, self.enc, self.fusion)
        d = self.src.out_dim
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, self.G))
        self.net = nn.ModuleDict({"src": self.src, "head": self.head})

    def forward_z(self, b):
        return self.head(self.src(b))

    def loss(self, b):
        return F.mse_loss(self.forward_z(b), b["y"])


class BranchedHead(nn.Module):
    """G independent copies of the OURS head (LayerNorm-Linear(d,1024)-GELU-Dropout-Linear(1024,1)), evaluated in parallel.
    Branch g only produces gene g; the encoder in front of it is shared and trained by all branches."""

    def __init__(self, d, G, hidden=1024, dropout=0.2):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.gam, self.bet = nn.Parameter(torch.ones(G, d)), nn.Parameter(torch.zeros(G, d))
        k1, k2 = 1 / math.sqrt(d), 1 / math.sqrt(hidden)
        self.W1 = nn.Parameter(torch.empty(G, d, hidden).uniform_(-k1, k1))
        self.b1 = nn.Parameter(torch.empty(G, hidden).uniform_(-k1, k1))
        self.W2 = nn.Parameter(torch.empty(G, hidden).uniform_(-k2, k2))
        self.b2 = nn.Parameter(torch.empty(G).uniform_(-k2, k2))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.ln(x)
        h = F.gelu(torch.einsum("bgd,gdh->bgh", z.unsqueeze(1) * self.gam + self.bet, self.W1) + self.b1)
        return (self.drop(h) * self.W2).sum(-1) + self.b2


class OursBranchedMethod(OursMethod):
    def build(self):
        self.src, self.emb_name, self.use_pix = make_source("our", None, self.rd, self.cfg, self.enc, self.fusion)
        self.head = BranchedHead(self.src.out_dim, self.G)
        self.net = nn.ModuleDict({"src": self.src, "head": self.head})


class OursKVoteMethod(OursMethod):
    """Trained exactly like OURS (encoder + head, MSE). For prediction the head is NOT used: the fused encoder
    output (1536-d) of the new spot is compared with the fused outputs of all TRAIN spots, and the prediction is
    the mean expression of its k nearest (Euclidean) training spots. Checkpoint selection (validation) is done
    on this k-NN prediction."""

    @torch.no_grad()
    def feats(self, split):
        self.net.eval()
        n = self.rd.views[split].n
        return torch.cat([self.src(self.mk(split, idx)).float() for idx in self.eval_batches(split)])

    @torch.no_grad()
    def predict_z(self, split):
        R = self.feats("train")
        Yref = torch.from_numpy((self.rd.views["train"].Y - self.rd.y_mu) / self.rd.y_sd).to(self.dev)
        Q = R if split == "train" else self.feats(split)
        k = min(self.hp["k"], R.shape[0])
        out = np.zeros((Q.shape[0], self.G), np.float32)
        for a in range(0, Q.shape[0], 1024):
            top = torch.cdist(Q[a:a + 1024], R).topk(k, dim=1, largest=False).indices
            out[a:a + 1024] = Yref[top].mean(1).cpu().numpy()
        return finite_guard(out, "kvote")


# ----------------------------------------------------------------------------- retrieval base
class RetrievalMethod(TorchMethod):
    """Contrastive training + top-k retrieval of training expression at inference."""

    def spot_embed_ref(self):
        raise NotImplementedError

    def img_embed(self, b):
        raise NotImplementedError

    @torch.no_grad()
    def predict_z(self, split):
        self.net.eval()
        R = F.normalize(self.spot_embed_ref(), dim=-1)                       # (n_train, d)
        Yref = torch.from_numpy((self.rd.views["train"].Y - self.rd.y_mu) / self.rd.y_sd).to(self.dev)
        k = min(self.hp["top_k"], R.shape[0])
        n = self.rd.views[split].n
        out = np.zeros((n, self.G), np.float32)
        for idx in self.eval_batches(split):
            b = self.mk(split, idx)
            with torch.autocast(self.dev.type, dtype=self.amp_dtype, enabled=self.amp):
                q = F.normalize(self.img_embed(b).float(), dim=-1)
            top = (q @ R.T).topk(k, dim=1).indices
            out[idx] = Yref[top].mean(1).cpu().numpy()
        return finite_guard(out, "retrieval")


class BleepMethod(RetrievalMethod):
    def build(self):
        kind = "our" if self.variant == "our_encoder" else "native"
        self.src, self.emb_name, self.use_pix = make_source(kind, "resnet50", self.rd, self.cfg, self.enc, self.fusion)
        self.ip = ProjectionHead(self.src.out_dim, 256, 0.1)
        self.sp = ProjectionHead(self.G, 256, 0.1)
        self.T = self.hp["temperature"]
        self.net = nn.ModuleDict({"src": self.src, "ip": self.ip, "sp": self.sp})

    def img_embed(self, b):
        return self.ip(self.src(b))

    def loss(self, b):
        ie, se = self.img_embed(b), self.sp(b["y"])
        logits = (se @ ie.T) / self.T
        tgt = F.softmax(((ie @ ie.T + se @ se.T) / 2) / self.T, -1)
        return ((soft_ce(logits, tgt) + soft_ce(logits.T, tgt.T)) / 2).mean()

    @torch.no_grad()
    def spot_embed_ref(self):
        tr = self.rd.views["train"]
        Y = torch.from_numpy((tr.Y - self.rd.y_mu) / self.rd.y_sd).to(self.dev)
        return torch.cat([self.sp(Y[i:i + 4096]) for i in range(0, tr.n, 4096)]).float()


class MclstexpMethod(RetrievalMethod):
    def build(self):
        kind = "our" if self.variant == "our_encoder" else "native"
        self.src, self.emb_name, self.use_pix = make_source(kind, "densenet121", self.rd, self.cfg, self.enc, self.fusion)
        G = self.G
        self.ip = ProjectionHead(self.src.out_dim, 256, 0.1)
        self.sp = ProjectionHead(G, 256, 0.1)
        self.xe, self.ye = nn.Embedding(N_POS_BINS, G), nn.Embedding(N_POS_BINS, G)
        self.blocks = nn.ModuleList([_AttnBlock(G, self.hp["heads"], self.hp["head_dim"]) for _ in range(self.hp["layers"])])
        self.T = self.hp["temperature"]
        self.net = nn.ModuleDict({"src": self.src, "ip": self.ip, "sp": self.sp, "xe": self.xe, "ye": self.ye,
                                  "blocks": self.blocks})

    def img_embed(self, b):
        return self.ip(self.src(b))

    def spot_embed(self, y, pos):
        x = (y + self.xe(pos[:, 0]) + self.ye(pos[:, 1])).unsqueeze(0)       # the batch is the sequence
        for blk in self.blocks:
            x = blk(x)
        return self.sp(x.squeeze(0))

    def loss(self, b):
        ie, se = self.img_embed(b), self.spot_embed(b["y"], b["pos"])
        cos = (se @ ie.T) / self.T
        lab = torch.arange(cos.shape[0], device=cos.device)
        return (F.cross_entropy(cos, lab) + F.cross_entropy(cos.T, lab)) / 2

    @torch.no_grad()
    def spot_embed_ref(self):
        tr = self.rd.views["train"]
        Y = torch.from_numpy((tr.Y - self.rd.y_mu) / self.rd.y_sd).to(self.dev)
        P = torch.from_numpy(tr.pos).to(self.dev)
        perm = np.random.RandomState(0).permutation(tr.n)
        out = torch.zeros(tr.n, 256, device=self.dev)
        bs = self.hp["batch_size"]
        for i in range(0, tr.n, bs):
            ix = torch.from_numpy(perm[i:i + bs]).to(self.dev)
            out[ix] = self.spot_embed(Y[ix], P[ix]).float()
        return out


class _AttnBlock(nn.Module):
    def __init__(self, d, heads, dh, ff=None):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.h, self.dh = heads, dh
        self.qkv = nn.Linear(d, 3 * heads * dh, bias=False)
        self.out = nn.Linear(heads * dh, d)
        self.ff = nn.Sequential(nn.Linear(d, ff or d), nn.GELU(), nn.Linear(ff or d, d))

    def forward(self, x):
        B, N, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, -1)
        x = x + self.out(a)
        return x + self.ff(self.n2(x))


# ----------------------------------------------------------------------------- GHIST (spot-level adaptation)
class GhistMethod(TorchMethod):
    def build(self):
        kind = "our" if self.variant == "our_encoder" else "native"
        self.src, self.emb_name, self.use_pix = make_source(kind, "unet", self.rd, self.cfg, self.enc, self.fusion)
        E, K, G, nct = self.hp["emb_dim"], self.hp["n_ref"], self.G, 5
        self.embed = nn.Sequential(nn.Linear(self.src.out_dim, E), nn.ReLU(), nn.Linear(E, E), nn.ReLU())
        self.w = nn.Sequential(nn.Linear(E, E), nn.ReLU(), nn.Linear(E, K))                  # weights over profiles
        self.R = nn.Parameter(torch.randn(K, G) * 0.1)                                       # learnable reference profiles
        self.comp = nn.Sequential(nn.Linear(E, E // 2), nn.ReLU(), nn.Linear(E // 2, nct))   # neighbourhood composition
        self.off = nn.Sequential(nn.Linear(E + nct, E), nn.ReLU(), nn.Linear(E, G))
        self.net = nn.ModuleDict({"src": self.src, "embed": self.embed, "w": self.w, "comp": self.comp, "off": self.off})
        self.net.R = self.R

    def run(self, b):
        h = self.embed(self.src(b))
        p = F.softmax(self.comp(h), -1)
        S = F.softmax(self.w(h), -1) @ self.R
        return S + self.off(torch.cat([h, p], 1)), p

    def forward_z(self, b):
        return self.run(b)[0]

    def loss(self, b):
        y, p = self.run(b)
        l_ge = F.mse_loss(y, b["y"])
        m = b["qvalid"]
        if m.any():
            q = (b["q"][m] + 1e-6)
            q = q / q.sum(1, keepdim=True)
            pe = p[m].float().clamp_min(1e-6)
            l_nc = (pe * (pe.log() - q.log())).sum(1).mean()                                 # KL(P_est || Q)
        else:
            l_nc = y.sum() * 0
        return l_ge + self.hp["w_nc"] * l_nc


# ----------------------------------------------------------------------------- FmH2ST
class FmH2STMethod(TorchMethod):
    use_slice = True
    tile_mode = True
    need_pix_ctx = False

    def build(self):
        self.emb_name = self.cfg.FOUNDATION_ENCODER
        D = self.rd.table(self.emb_name).shape[1]
        d, G = 512, self.G
        if self.variant == "our_encoder":
            self.src = OurEncoder(D, fusion=self.cfg.FUSION)
            self.use_pix = False
            fm_in, spot_in = self.src.out_dim + D, self.src.out_dim
            self.pix_enc = None
        else:
            self.src = None
            self.use_pix = True
            self.pix_enc = PixelBackbone("cbam_ms", False)
            fm_in, spot_in = 3 * D + 10, self.pix_enc.out_dim
        self.cemb = nn.Sequential(nn.Linear(fm_in, d), nn.SiLU(), nn.Linear(d, d))
        self.spot_proj = nn.Linear(spot_in, d)
        self.xe, self.ye = nn.Embedding(N_POS_BINS, d), nn.Embedding(N_POS_BINS, d)
        self.vit = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 2 * d, 0.1, batch_first=True, norm_first=True), 2)
        self.gat_spot = DenseGAT(d, d, 4)
        self.gat_fm = DenseGAT(d, d, 4)
        self.agg1, self.agg2 = AttnAgg(d), AttnAgg(d)
        self.dec = nn.Sequential(nn.Linear(d, G), nn.SiLU(), nn.Linear(G, G))
        self.fc1, self.fc2 = nn.Linear(d, G), nn.Linear(d, G)
        mods = {"cemb": self.cemb, "spot_proj": self.spot_proj, "xe": self.xe, "ye": self.ye, "vit": self.vit,
                "gat_spot": self.gat_spot, "gat_fm": self.gat_fm, "agg1": self.agg1, "agg2": self.agg2,
                "dec": self.dec, "fc1": self.fc1, "fc2": self.fc2}
        if self.src is not None:
            mods["src"] = self.src
        if self.pix_enc is not None:
            mods["pix_enc"] = self.pix_enc
        self.net = nn.ModuleDict(mods)

    def train_batches(self):
        tiles = make_tiles(self.rd.views["train"], self.hp["window"], self.rng)
        tiles = [t for t in tiles if len(t) > 8]
        return [tiles[i] for i in self.rng.permutation(len(tiles))]

    def eval_batches(self, split):
        key = (split, self.hp["window"])
        if key not in self.rd.tiles_cache:
            self.rd.tiles_cache[key] = make_tiles(self.rd.views[split], self.hp["window"], None)
        return self.rd.tiles_cache[key]

    def run(self, b):
        N = b["y"].shape[0]
        adj = spatial_adj(b["xy"], self.hp["k_spatial"])
        if self.src is not None:
            f = self.src(b)
            cemb_in, g = torch.cat([f, b["slice"]], 1), f
        else:
            cemb_in = torch.cat([b["img"], b["slice"], b["ctx"], b["nuc"]], 1)
            g = self.pix_enc(b["pix"])
        # spot-specific branch
        g = self.spot_proj(g) + self.xe(b["pos"][:, 0]) + self.ye(b["pos"][:, 1])
        f1 = self.gat_spot(self.vit(g.unsqueeze(0)).squeeze(0), adj)
        # foundation branch with spatial + learnable feature graph
        c = self.cemb(cemb_in)
        with torch.no_grad():
            cn = F.normalize(c.float(), dim=-1)
            sim = cn @ cn.T
            kf = min(self.hp["k_feature"] + 1, N)
            A2 = torch.zeros(N, N, device=c.device).scatter_(1, sim.topk(kf, dim=1).indices, 1.0)
            A2 = ((A2 + A2.T) > 0).float()
            A2.fill_diagonal_(1.0)
        f2 = self.agg1(self.gat_fm(c, adj), self.gat_fm(c, A2))
        fused = self.agg2(f1, f2)
        return self.dec(fused), self.fc1(f1), self.fc2(f2)

    def forward_z(self, b):
        return self.run(b)[0]

    def loss(self, b):
        o, o1, o2 = self.run(b)
        y = b["y"]
        L = F.mse_loss(o, y)
        for oo in (o1, o2):
            L = L + 0.7 * F.mse_loss(oo, y) + 0.3 * F.mse_loss(oo, o)
        return L

    @torch.no_grad()
    def predict_z(self, split):
        self.net.eval()
        n = self.rd.views[split].n
        out = np.zeros((n, self.G), np.float32)
        for idx in self.eval_batches(split):
            if len(idx) < 2:
                continue
            b = self.mk(split, idx)
            with torch.autocast(self.dev.type, dtype=self.amp_dtype, enabled=self.amp):
                out[idx] = self.forward_z(b).float().cpu().numpy()
        return finite_guard(out, "fmh2st")


# ----------------------------------------------------------------------------- STORM (inductive adaptation)
class StormMethod:
    def __init__(self, rd, mode, variant, seed, cfg, hp, device):
        self.rd, self.mode, self.variant, self.cfg, self.hp, self.dev = rd, mode, variant, cfg, hp, device
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.bt = {s: rd.batcher(s, mode) for s in ("train", "val", "test")}
        tr = rd.views["train"]
        self.G, self.n, self.R = tr.Y.shape[1], tr.n, hp["rank"]
        if variant == "our_encoder":
            self.emb_name = cfg.FOUNDATION_ENCODER
            self.src = OurEncoder(rd.table(self.emb_name).shape[1], fusion=cfg.FUSION).to(device)
        else:
            self.emb_name = cfg.STORM_ENCODER
            self.src = RawConcatSource(rd.table(self.emb_name).shape[1]).to(device)
        self.U = nn.Linear(self.src.out_dim, self.R, bias=False).to(device)
        self.S = nn.Parameter(torch.randn(self.n, self.R, device=device) * 0.3)
        self.C = nn.Parameter(torch.randn(self.G, self.R, device=device) * 0.3)
        self.Y = torch.from_numpy(tr.Y).to(device)
        var = torch.from_numpy(tr.Y.var(0)).to(device)
        gw = torch.log1p(var)
        self.gw = gw / (gw.max() + 1e-8) + 1e-3
        self.edges, self.deg = self._knn_edges(tr, hp["knn"])
        corr = np.nan_to_num(np.corrcoef(tr.Y.T), nan=0.0)
        np.fill_diagonal(corr, -1)
        A = np.zeros((self.G, self.G), np.float32)
        for g in range(self.G):
            A[g, np.argsort(-corr[g])[:5]] = 1
        A = np.maximum(A, A.T)
        d = np.maximum(A.sum(1), 1e-8)
        L = np.eye(self.G, dtype=np.float32) - A / np.sqrt(d)[:, None] / np.sqrt(d)[None, :]
        self.L_gene = torch.from_numpy(L).to(device)

    def _knn_edges(self, view, k):
        src, dst = [], []
        for s in range(len(view.slide_names)):
            ids = np.where(view.slide == s)[0]
            if len(ids) < 3:
                continue
            kk = min(k + 1, len(ids))
            _, nb = cKDTree(view.xy[ids]).query(view.xy[ids], k=kk)
            for j in range(1, kk):
                src.append(ids); dst.append(ids[nb[:, j]])
        src, dst = np.concatenate(src), np.concatenate(dst)
        pair = np.unique(np.stack([np.minimum(src, dst), np.maximum(src, dst)], 1), axis=0)
        e = torch.from_numpy(pair).to(self.dev)
        deg = torch.bincount(torch.cat([e[:, 0], e[:, 1]]), minlength=view.n).float().clamp(min=1)
        return e, deg

    def _feat(self, split, idx, train=False):
        return self.src(self.bt[split].make(idx, emb=self.emb_name, train=train))

    def _terms(self, first=False):
        yhat = self.S @ self.C.T
        w = torch.where(self.Y > 0, 1.0, 0.3) * self.gw
        mse = (w * (self.Y - yhat) ** 2).sum() / w.sum()
        r0 = (self.S ** 2).mean() + (self.C ** 2).mean()
        sub = np.sort(self.rng.choice(self.n, min(self.n, self.hp["r1_batch"]), replace=False))
        r1 = ((self.U(self._feat("train", sub, True)) - self.S[torch.from_numpy(sub).to(self.dev)]) ** 2).mean()
        sn = self.S / self.deg.sqrt().unsqueeze(1)
        r2 = ((sn[self.edges[:, 0]] - sn[self.edges[:, 1]]) ** 2).sum(1).mean()
        r3 = torch.trace(self.C.T @ self.L_gene @ self.C) / self.G
        return mse, r0, r1, r2, r3

    @torch.no_grad()
    def _smooth(self, split, Z):
        v = self.rd.views[split]
        e, _ = self._knn_edges(v, self.hp["knn"])
        src = torch.cat([e[:, 0], e[:, 1], torch.arange(v.n, device=self.dev)])
        dst = torch.cat([e[:, 1], e[:, 0], torch.arange(v.n, device=self.dev)])
        cnt = torch.bincount(src, minlength=v.n).float().unsqueeze(1)
        a, S = self.hp["smooth_alpha"], Z
        for _ in range(self.hp["smooth_steps"]):
            agg = torch.zeros_like(Z).index_add_(0, src, S[dst]) / cnt
            S = (1 - a) * Z + a * agg
        return S

    @torch.no_grad()
    def predict(self, split):
        self.src.eval()
        n = self.rd.views[split].n
        Z = torch.cat([self.U(self._feat(split, np.arange(i, min(i + 4096, n)))) for i in range(0, n, 4096)])
        return (self._smooth(split, Z) @ self.C.T).cpu().numpy()

    def fit(self):
        hp = self.hp
        enc_params = list(self.src.parameters()) + list(self.U.parameters())
        opt = torch.optim.Adam([{"params": [self.S, self.C], "lr": hp["lr"]},
                                {"params": enc_params, "lr": hp["enc_lr"]}])
        lam, best, wait, state = None, -1e9, 0, None
        for ep in range(hp["epochs"]):
            self.src.train()
            opt.zero_grad()
            mse, r0, r1, r2, r3 = self._terms()
            if lam is None:                                   # adaptive loss scaling: lambda_i = P_i * L_mse / R_i
                P = hp["P"]
                lam = [P[i] * mse.detach() / (r.detach() + 1e-8) for i, r in enumerate((r0, r1, r2, r3))]
            loss = mse + sum(l * r for l, r in zip(lam, (r0, r1, r2, r3)))
            loss.backward()
            opt.step()
            if (ep + 1) % hp["eval_every"] == 0 or ep + 1 == hp["epochs"]:
                self.src.eval()
                sc = pcc_mean(self.predict("val"), self.rd.views["val"].Y)
                if self.cfg.VERBOSE_TRAIN:
                    print(f"      ep {ep + 1:3d} loss {float(loss):.4f} val-PCC {sc:.4f}")
                if sc > best + 1e-6:
                    best, wait = sc, 0
                    state = [copy.deepcopy(m.state_dict()) for m in (self.src, self.U)] + [self.C.detach().clone()]
                else:
                    wait += hp["eval_every"]
                    if wait >= hp["patience"]:
                        break
        if state is not None:
            self.src.load_state_dict(state[0]); self.U.load_state_dict(state[1])
            with torch.no_grad():
                self.C.copy_(state[2])
        return best


# ----------------------------------------------------------------------------- factory
RAW_PIXEL_METHODS = {"BLEEP", "GHIST", "MCLSTEXP", "FMH2ST"}
CLASSES = {"OURS": OursMethod, "OURS-BRANCHED": OursBranchedMethod, "BLEEP": BleepMethod, "MCLSTEXP": MclstexpMethod, "GHIST": GhistMethod,
           "FMH2ST": FmH2STMethod, "STORM": StormMethod}


def get_hp(cfg, method, variant):
    hp = dict(cfg.HPARAMS["common"])
    hp.update(cfg.HPARAMS.get(method, {}))
    if variant == "default" and method in RAW_PIXEL_METHODS:      # raw-pixel training is expensive
        hp.update(epochs=cfg.RAW_EPOCHS, patience=cfg.RAW_PATIENCE, max_steps=cfg.RAW_MAX_STEPS_PER_EPOCH)
    return hp


def needed_assets(entries, cfg):
    """entries: (method, variant, encoder-or-None). Which caches are needed."""
    emb, pix = set(), False
    for mth, v, enc in entries:
        if mth.startswith("OURS"):
            emb.add(enc or cfg.FOUNDATION_ENCODER)
        elif v == "our_encoder" or mth == "FMH2ST":
            emb.add(cfg.FOUNDATION_ENCODER)
        if mth == "STORM" and v == "default":
            emb.add(cfg.STORM_ENCODER)
        if v == "default" and mth in RAW_PIXEL_METHODS:
            pix = True
    return sorted(emb), pix


def build_method(name, variant, rd, mode, seed, cfg, device, enc=None, fusion=None):
    import re
    mt = re.fullmatch(r"OURS-(\d+)VOTE", name)
    if mt:                                                     # e.g. OURS-5VOTE -> k = 5
        hp = get_hp(cfg, "OURS", variant)
        hp["k"] = int(mt.group(1))
        return OursKVoteMethod(rd, mode, variant, seed, cfg, hp, device, enc, fusion)
    if name.startswith("OURS"):
        return CLASSES[name](rd, mode, variant, seed, cfg, get_hp(cfg, name, variant), device, enc, fusion)
    return CLASSES[name](rd, mode, variant, seed, cfg, get_hp(cfg, name, variant), device)
