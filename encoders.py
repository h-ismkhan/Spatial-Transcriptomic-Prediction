# In the name of God
"""
encoders.py
===========
FEATURE SOURCES. A source maps a batch (see data_utils.Batcher) to one fused vector per spot.

  OurEncoder      : the encoder of the method note (Method-1): frozen-embedding image & context,
                    MLP(10->64->128) for nuclei, every modality projected to 512 with
                    Linear->LayerNorm->GELU, concatenated (1536-d).        [used by OURS and by every
                    competitor in its "Our-Encoder" variant]
  NativeSource    : the competitor's own image encoder trained on (down-scaled) pixels. The same
                    encoder processes the spot patch and up to K neighbour patches (context); nuclei go
                    through an MLP.  out = [f_img, f_ctx, f_nuc].        [competitor "Default" variants]
  RawConcatSource : no parameters: [image emb, context emb, nuclei features]   [STORM "Default"]

Pixel backbones: 'resnet50' (BLEEP), 'densenet121' (mclSTExp), 'unet' (GHIST, compact UNet with
skip connections), 'cbam_ms' (FmH2ST spot branch: multi-scale conv + CBAM + ConvMixer).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import NUC_DIM

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def proj_block(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.LayerNorm(o), nn.GELU())


FUSIONS = ("concat", "sum", "gate", "attn", "lmf")


class LowRankFusion(nn.Module):
    """LMF (Liu et al. 2018 / Zadeh): append 1 to every modality vector, project with rank-r factors, multiply
    the modality projections element-wise and sum over the rank."""

    def __init__(self, d, n_mod=3, rank=4):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.randn(rank, d + 1, d) / (d + 1) ** 0.5) for _ in range(n_mod)])
        self.rw = nn.Parameter(torch.ones(1, rank) / rank)
        self.b = nn.Parameter(torch.zeros(d))
        self.norm = nn.LayerNorm(d)

    def forward(self, xs):
        prod = None
        for x, W in zip(xs, self.W):
            z = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], 1)
            h = torch.einsum("bi,rio->bro", z, W)
            prod = h if prod is None else prod * h
        return F.gelu(self.norm((prod * self.rw.unsqueeze(-1)).sum(1) + self.b))


class OurEncoder(nn.Module):
    """Method-1 encoder. Every modality -> 512 (Linear-LayerNorm-GELU), then a FUSION:
       concat : [img, nuc, ctx] -> 1536-d (original)        sum  : mean of the three 512-d vectors
       gate   : per-spot softmax gate over modalities (Arevalo-style gated fusion) -> 512-d
       attn   : transformer layer over the 3 modality tokens (+ modality embeddings), mean pooled -> 512-d
       lmf    : low-rank multimodal (multiplicative) fusion, rank 4 -> 512-d"""

    def __init__(self, d_emb, d=512, fusion="concat"):
        super().__init__()
        assert fusion in FUSIONS, f"fusion must be one of {FUSIONS}"
        self.fusion = fusion
        self.p_img, self.p_ctx = proj_block(d_emb, d), proj_block(d_emb, d)
        self.nuc = nn.Sequential(nn.Linear(NUC_DIM, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU())
        self.p_nuc = proj_block(128, d)
        self.out_dim = 3 * d if fusion == "concat" else d
        if fusion == "gate":
            self.gate = nn.Linear(3 * d, 3)
        elif fusion == "attn":
            self.mod_emb = nn.Parameter(torch.zeros(3, d))
            self.tr = nn.TransformerEncoderLayer(d, 8, 2 * d, 0.1, batch_first=True, norm_first=True)
            self.norm = nn.LayerNorm(d)
        elif fusion == "lmf":
            self.lmf = LowRankFusion(d)

    def forward(self, b):
        xs = [self.p_img(b["img"]), self.p_nuc(self.nuc(b["nuc"])), self.p_ctx(b["ctx"])]
        if self.fusion == "concat":
            return torch.cat(xs, 1)
        if self.fusion == "sum":
            return sum(xs) / 3
        if self.fusion == "gate":
            g = torch.softmax(self.gate(torch.cat(xs, 1)), -1)
            return sum(g[:, i:i + 1] * x for i, x in enumerate(xs))
        if self.fusion == "attn":
            t = torch.stack(xs, 1) + self.mod_emb.unsqueeze(0)
            return self.norm(self.tr(t).mean(1))
        return self.lmf(xs)


class RawConcatSource(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.out_dim = 2 * d_emb + NUC_DIM

    def forward(self, b):
        return torch.cat([b["img"], b["ctx"], b["nuc"]], 1)


# ----------------------------------------------------------------------------- pixel backbones
def _cb(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    """Compact UNet (GHIST uses UNet 3+). Feature = [mean of first block, mean of last decoder block]."""

    def __init__(self, base=32):
        super().__init__()
        c = (base, base * 2, base * 4, base * 8)
        self.e1 = nn.Sequential(_cb(3, c[0]), _cb(c[0], c[0]))
        self.e2 = nn.Sequential(_cb(c[0], c[1]), _cb(c[1], c[1]))
        self.e3 = nn.Sequential(_cb(c[1], c[2]), _cb(c[2], c[2]))
        self.e4 = nn.Sequential(_cb(c[2], c[3]), _cb(c[3], c[3]))
        self.d3 = nn.Sequential(_cb(c[3] + c[2], c[2]), _cb(c[2], c[2]))
        self.d2 = nn.Sequential(_cb(c[2] + c[1], c[1]), _cb(c[1], c[1]))
        self.d1 = nn.Sequential(_cb(c[1] + c[0], c[0]), _cb(c[0], c[0]))
        self.out_dim = 2 * c[0]

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        x4 = self.e4(F.max_pool2d(x3, 2))
        up = lambda t, r: F.interpolate(t, size=r.shape[-2:], mode="bilinear", align_corners=False)
        y3 = self.d3(torch.cat([up(x4, x3), x3], 1))
        y2 = self.d2(torch.cat([up(y3, x2), x2], 1))
        y1 = self.d1(torch.cat([up(y2, x1), x1], 1))
        return torch.cat([x1.mean((2, 3)), y1.mean((2, 3))], 1)


class CBAM(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(), nn.Linear(max(c // r, 4), c))
        self.sp = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        a = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3)))).unsqueeze(-1).unsqueeze(-1)
        x = x * a
        s = torch.sigmoid(self.sp(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * s


class ConvMixerBlock(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.dw = nn.Sequential(nn.Conv2d(c, c, k, groups=c, padding=k // 2), nn.GELU(), nn.BatchNorm2d(c))
        self.pw = nn.Sequential(nn.Conv2d(c, c, 1), nn.GELU(), nn.BatchNorm2d(c))

    def forward(self, x):
        return self.pw(self.dw(x) + x)


class MultiScaleCBAM(nn.Module):
    """FmH2ST spot branch: kernels 3/5/7 + CBAM, 1x1 fusion + BN, ConvMixer blocks, down-sampling, flatten."""

    def __init__(self, c=32, n_mix=2, d_out=256):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, c, k, padding=k // 2) for k in (3, 5, 7)])
        self.cbams = nn.ModuleList([CBAM(c) for _ in range(3)])
        self.fuse = nn.Sequential(nn.Conv2d(3 * c, c, 1), nn.BatchNorm2d(c))
        self.mix = nn.Sequential(*[ConvMixerBlock(c) for _ in range(n_mix)])
        self.down = nn.Conv2d(c, c, 4, stride=4)
        self.fc = nn.Linear(c * 16, d_out)
        self.out_dim = d_out

    def forward(self, x):
        g = torch.cat([cb(cv(x)) for cv, cb in zip(self.convs, self.cbams)], 1)
        g = self.down(self.mix(self.fuse(g)))
        return self.fc(F.adaptive_avg_pool2d(g, 4).flatten(1))


class TimmBackbone(nn.Module):
    def __init__(self, name, pretrained):
        super().__init__()
        import timm
        try:
            self.m = timm.create_model(name, pretrained=pretrained, num_classes=0)
        except Exception as e:                                   # offline / no weights
            print(f"[warn] could not load pretrained '{name}' ({type(e).__name__}); using random init")
            self.m = timm.create_model(name, pretrained=False, num_classes=0)
        self.out_dim = self.m.num_features

    def forward(self, x):
        return self.m(x)


class PixelBackbone(nn.Module):
    def __init__(self, kind, pretrained=True):
        super().__init__()
        self.kind = kind
        if kind in ("resnet50", "densenet121"):
            self.net = TimmBackbone(kind, pretrained)
            self.norm = True
        elif kind == "unet":
            self.net, self.norm = TinyUNet(), False
        elif kind == "cbam_ms":
            self.net, self.norm = MultiScaleCBAM(), False
        else:
            raise ValueError(kind)
        self.register_buffer("mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMNET_STD).view(1, 3, 1, 1))
        self.out_dim = self.net.out_dim

    def forward(self, x):
        x = (x - self.mean) / self.std if self.norm else (x - 0.5) / 0.25
        return self.net(x)


class NativeSource(nn.Module):
    """Competitor's own image encoder on pixels; shared by the spot patch and the context patches."""

    def __init__(self, kind, pretrained=True):
        super().__init__()
        self.enc = PixelBackbone(kind, pretrained)
        self.nuc = nn.Sequential(nn.Linear(NUC_DIM, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU())
        self.d = self.enc.out_dim
        self.out_dim = 2 * self.d + 128
        self.pretrained_backbone = kind in ("resnet50", "densenet121")

    def forward(self, b):
        B, K = b["pix_ctx"].shape[:2]
        f_all = self.enc(torch.cat([b["pix"], b["pix_ctx"].flatten(0, 1)], 0))
        f_img, f_ctx = f_all[:B], f_all[B:].view(B, K, -1)
        w = b["pix_valid"].float().unsqueeze(-1)
        f_ctx = (f_ctx * w).sum(1) / w.sum(1).clamp(min=1)
        return torch.cat([f_img, f_ctx, self.nuc(b["nuc"])], 1)

    def encoder_params(self):
        return list(self.enc.parameters())

    def other_params(self):
        return list(self.nuc.parameters())
