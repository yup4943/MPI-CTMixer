"""MPI-CTMixer backbone (PSR block + dense-QKV LG-MHSA + multi-pixel head).

Code in this file is derived from the official CTMixer implementation (Convolution-Transformer Mixer
for hyperspectral image classification); the modifications described in the MPI-CTMixer paper are
(i) the wide plain-conv progressive spectral reduction (PSR) block, (ii) dense (groups=1) Q/K/V
projection in LG-MHSA, and (iii) a central KxK prediction head (`center_size`).

Module/attribute names are kept identical to the research code so that released checkpoints load
with `strict=True`.
"""
import math

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class PatchEmbeddings(nn.Module):
    """
    Module that extracts patches and projects them
    """
    def __init__(self, patch_size: int, patch_dim: int, emb_dim: int):
        super().__init__()
        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            # "b c (h p1) (w p2) -> b (p1 p2) c h w",
            p1=patch_size, p2=patch_size)

        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(in_features=patch_dim, out_features=emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rearrange into patches
        x = self.patchify(x)
        # Flatten patches into individual vectors
        x = self.flatten(x)
        # Project to higher dim
        x = self.proj(x)
        return x


class PositionalEmbeddings(nn.Module):
    """
    Learned positional embeddings
    """
    def __init__(self, num_pos: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_pos, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


class Attention_Conv(nn.Module):

    def __init__(self, dim: int, head_dim: int, num_heads: int, num_patch: int, patch_size: int):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(dim)
        self.qkv = nn.Conv2d(dim, self.inner_dim * 3, kernel_size=1, padding=0, groups=1, bias=False)
        self.avgpool=nn.AdaptiveAvgPool1d(dim)

        self.qs = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, bias=False)
        self.ks = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)

        qkv = self.qkv(self.act(self.bn(x)))
        qkv = qkv.contiguous().view(b, self.num_patch*self.num_patch, self.inner_dim * 3)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        q_1 = q[:, 0:self.num_heads//2, :, :]
        k_1 = k[:, 0:self.num_heads//2, :, :]
        v_1 = v[:, 0:self.num_heads//2, :, :]
        q_2 = q[:, self.num_heads//2:self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))
        k_2 = k[:, self.num_heads//2:self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))
        v_2 = v[:, self.num_heads//2:self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))

        q_2 = self.qs(q_2)
        k_2 = self.ks(k_2)
        res_2 = (q_2 + k_2 + v_2).reshape(b, n, -1)

        scores = torch.einsum("b h i d, b h j d -> b h i j", q_1, k_1)
        scores = scores * self.scale
        attn = self.attn(scores)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v_1)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        res = torch.cat([out, res_2], axis=2)
        out = self.avgpool(res)
        return out
    

class FeedForward_Conv(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(dim), nn.GELU(), 
            nn.Conv2d(dim, 64, kernel_size=1, padding=0, bias=False))

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64), nn.GELU(), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False))
        
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64), nn.GELU(), 
            nn.Conv2d(64, dim, kernel_size=1, padding=0, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, hw, dim = x.shape     # [bs, num_seq, dim]
        x_reshape = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        out1 = self.conv1(x_reshape)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2) + x_reshape
        result = out3.contiguous().view(b, self.num_patch * self.num_patch, self.dim)

        return result


class transformer(nn.Module):

    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), Attention_Conv(dim, head_dim, num_heads, num_patch, patch_size)),
                nn.Sequential(nn.LayerNorm(dim), FeedForward_Conv(dim, hidden_dim, num_patch, patch_size))
            ]
            self.layers.append(nn.ModuleList(layer))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PlainConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, width_mult=1.0):
        super().__init__()
        hidden = max(int(round(out_ch * width_mult)), out_ch)
        if isinstance(k, tuple):
            kh, kw = k
            ph, pw = (kh - 1) // 2, (kw - 1) // 2
        else:
            kh = kw = k
            ph = pw = (k - 1) // 2
        self.conv1 = nn.Conv2d(in_ch, hidden, (kh, kw), 1, (ph, pw), bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, out_ch, 1, bias=True)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class Pooling(nn.Module):
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")
        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x):
        return x.mean(dim=1)

    def cls_pool(self, x):
        return x[:, 0]

    def forward(self, x):
        return self.pool_fn(x)


class Classifier(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x):
        return self.model(x)


class MPICTMixer(nn.Module):

    def __init__(self, channels, num_classes, image_size, num_layers: int = 1, num_heads: int = 4,
                 center_size: int = 3, patch_size: int = 1, emb_dim: int = 128, head_dim=64,
                 hidden_dim: int = 64, pool: str = "mean", new_bands: int = 160, reduce_bands: int = 48,
                 plainconv_width_mult: float = 6.35):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patch = int(math.sqrt(self.num_patches))
        self.center_size = center_size
        self.act = nn.ReLU(inplace=True)

        def div_up(a, s):
            return (a + s - 1) // s

        def round_up(a, s):
            return div_up(a, s) * s

        s = 1
        nb = new_bands
        rb = reduce_bands
        t1_total = (2 * (nb - rb)) // 3 + rb
        t2_total = ((nb - rb) + 2) // 3 + rb
        t3_total = rb

        self.spec_expand = nn.Conv2d(channels, new_bands, kernel_size=1, bias=False)
        self.gprb_bn1 = nn.BatchNorm2d(nb)
        self.gprb_conv1 = PlainConvBlock(nb, t1_total, k=3, width_mult=plainconv_width_mult)
        self.gprb_bn2 = nn.BatchNorm2d(round_up(t1_total, s))
        self.gprb_conv2 = PlainConvBlock(t1_total, t2_total, k=3, width_mult=plainconv_width_mult)
        self.gprb_bn3 = nn.BatchNorm2d(round_up(t2_total, s))
        self.gprb_conv3 = PlainConvBlock(t2_total, t3_total, k=3, width_mult=plainconv_width_mult)

        patch_dim = rb * patch_size ** 2

        # Dual Residual Block (Ref-RDACN) -- IDENTICAL to CTMixer_original.py
        self.bn1 = nn.BatchNorm2d(emb_dim)
        self.conv1 = nn.Conv2d(emb_dim, 64, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, emb_dim, kernel_size=1, padding=0)

        # pixel-wise tokenization + LG-MHSA (ORIGINAL's local-global attention)
        self.patch_embeddings = PatchEmbeddings(patch_size=patch_size, patch_dim=patch_dim, emb_dim=emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=emb_dim)
        self.transformer = transformer(dim=emb_dim, num_layers=num_layers, num_heads=num_heads,
                                        head_dim=head_dim, hidden_dim=hidden_dim,
                                        num_patch=self.num_patch, patch_size=patch_size)
        self.dropout = nn.Dropout(0.5)

        self.pool = Pooling(pool=pool)
        self.classifier = Classifier(dim=emb_dim, num_classes=num_classes)
        self.center_pool = nn.AvgPool2d(
            kernel_size=self.image_size - self.center_size + 1,
            stride=1,
            padding=0
        )

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        B = x.size(0)
        x = x.squeeze(1)
        x = self.spec_expand(x)
        x = self.gprb_conv1(self.gprb_bn1(x))
        x = self.gprb_conv2(self.gprb_bn2(x))
        x = self.gprb_conv3(self.gprb_bn3(x))
        b, c, h, w = x.shape

        x4 = self.patch_embeddings(x)
        x5 = self.pos_embeddings(x4)
        x6 = self.transformer(x5)

        x4_c = x4.reshape(b, -1, h, w)
        x_c1 = self.conv1(self.act(self.bn1(x4_c)))
        x_c2 = self.conv2(self.act(self.bn2(x_c1)))
        x_c3 = self.conv3(self.act(self.bn3(x_c2)))

        tok = self.dropout(x6 + x_c3.reshape(B, h * w, -1))
        tok = tok.transpose(1, 2).reshape(B, self.emb_dim, h, w)

        center = self.center_pool(tok)
        center = center.permute(0, 2, 3, 1)
        center = center.reshape(B, self.center_size ** 2, self.emb_dim)

        logits = self.classifier(center)
        logits = logits.transpose(1, 2).reshape(B, -1, self.center_size, self.center_size)
        if return_logits:
            return logits
        return torch.softmax(logits, dim=1)



CTMixer = MPICTMixer  # alias used by the research code / checkpoints
