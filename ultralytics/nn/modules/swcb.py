# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
SWCB-YOLO custom modules.

This file implements the two learnable feature components of SWCB-YOLO described in
the paper "SWCB-YOLO: Toward Robust and Real-Time Wind Turbine Blade Defect Detection
in Unstructured Natural Environments":

  * ``ASSwin2``  - Asymmetric Strip Swin-TransformerV2 block (AS-Swin2). It augments the
                   standard square-window self-attention of Swin-TransformerV2 with two
                   orthogonal strip windows whose anisotropic receptive fields are aligned
                   with the dominant propagation axis of slender cracks. A reparameterized
                   one-dimensional directional Log-CPB supplies the positional bias for each
                   strip branch (Eq. (2)-(3) of the paper).

  * ``FS_DDA``   - Frequency-Spatial Dual-Domain Attention block. A 2D-DCT spectral
                   channel-weighting stage is cascaded into a large-receptive-field spatial
                   attention stage so that frequency reweighting and spatial geometric
                   refinement act in sequence on the same backbone feature (Eq. (4)-(7)).

The morphology-derived CA-Shape-IoU loss is implemented separately in
``ultralytics/utils/metrics.py`` (the differentiable IoU term) and
``ultralytics/utils/loss.py`` (the offline-field assembly), and is not part of this file.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("ASSwin2", "FS_DDA")


# ---------------------------------------------------------------------------------------
# Asymmetric Strip Swin-TransformerV2 (AS-Swin2)
# ---------------------------------------------------------------------------------------
def _to_windows(x, wh, ww):
    """Partition (B, H, W, C) into non-overlapping (wh, ww) windows -> (num_windows*B, wh*ww, C)."""
    b, h, w, c = x.shape
    x = x.view(b, h // wh, wh, w // ww, ww, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh * ww, c)
    return windows


def _from_windows(windows, wh, ww, h, w):
    """Inverse of :func:`_to_windows`: (num_windows*B, wh*ww, C) -> (B, H, W, C)."""
    c = windows.shape[-1]
    b = int(windows.shape[0] / (h * w / wh / ww))
    x = windows.view(b, h // wh, w // ww, wh, ww, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, c)
    return x


class _LogCPB(nn.Module):
    """Log-spaced Continuous Position Bias generator (SwinV2-style).

    A tiny MLP maps log-spaced relative coordinates to a per-head bias. ``ndim`` selects a
    full 2D bias (square window, ``ndim == 2``) or a 1D directional bias for a strip window
    (``ndim == 1``), realizing the reparameterized directional Log-CPB of Eq. (3).
    """

    def __init__(self, num_heads, hidden=256, ndim=2):
        super().__init__()
        self.ndim = ndim
        self.mlp = nn.Sequential(
            nn.Linear(ndim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_heads, bias=False),
        )

    def forward(self, rel_coords):
        """rel_coords: (L, L, ndim) raw relative offsets -> bias (num_heads, L, L)."""
        # log-spaced continuous transform: sign(d) * log2(1 + |d|)
        log_coords = torch.sign(rel_coords) * torch.log2(torch.abs(rel_coords) + 1.0)
        bias = self.mlp(log_coords)  # (L, L, num_heads)
        return bias.permute(2, 0, 1).contiguous()


class _StripWindowAttention(nn.Module):
    """Scaled-cosine windowed self-attention with a (directional) Log-CPB bias.

    Implements ASWA for a single partitioning scheme (Eq. (2)). ``mode`` is one of
    ``"square"`` (M x M, 2D bias), ``"h"`` (1 x M^2 horizontal strip, bias on dx) or
    ``"v"`` (M^2 x 1 vertical strip, bias on dy). All three branches share Q/K/V
    projections through the parent block but own separate, non-shared bias generators.
    """

    def __init__(self, dim, num_heads, window_size, mode="square"):
        super().__init__()
        assert mode in {"square", "h", "v"}
        self.dim = dim
        self.num_heads = num_heads
        self.mode = mode
        m = window_size
        if mode == "square":
            self.wh, self.ww = m, m
            ndim = 2
        elif mode == "h":
            self.wh, self.ww = 1, m * m
            ndim = 1
        else:  # "v"
            self.wh, self.ww = m * m, 1
            ndim = 1

        # learnable scaled-cosine temperature (clamped, SwinV2 convention)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads, 1, 1)))
        self.cpb = _LogCPB(num_heads, ndim=ndim)
        self._register_rel_coords(ndim)

    def _register_rel_coords(self, ndim):
        """Precompute the integer relative-coordinate grid for this window shape."""
        coords_h = torch.arange(self.wh)
        coords_w = torch.arange(self.ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, wh, ww)
        coords = torch.flatten(coords, 1)  # (2, L)
        rel = coords[:, :, None] - coords[:, None, :]  # (2, L, L) -> (dy, dx)
        rel = rel.permute(1, 2, 0).contiguous().float()  # (L, L, 2)
        if ndim == 1:
            # keep only the strip's active axis: dx for horizontal, dy for vertical
            axis = 1 if self.mode == "h" else 0
            rel = rel[:, :, axis:axis + 1]  # (L, L, 1)
        self.register_buffer("rel_coords", rel, persistent=False)

    def forward(self, qkv, h, w):
        """qkv: (B, H, W, 3*dim) already projected. Returns (B, H, W, dim)."""
        b = qkv.shape[0]
        # pad so H, W are divisible by the window shape, then window-partition
        pad_b = (self.wh - h % self.wh) % self.wh
        pad_r = (self.ww - w % self.ww) % self.ww
        if pad_b or pad_r:
            qkv = F.pad(qkv.permute(0, 3, 1, 2), (0, pad_r, 0, pad_b)).permute(0, 2, 3, 1)
        hp, wp = h + pad_b, w + pad_r

        win = _to_windows(qkv, self.wh, self.ww)  # (nW*B, L, 3*dim)
        nwb, length, _ = win.shape
        win = win.view(nwb, length, 3, self.num_heads, self.dim // self.num_heads)
        q, k, v = win[:, :, 0], win[:, :, 1], win[:, :, 2]
        q = q.permute(0, 2, 1, 3)  # (nW*B, heads, L, hd)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # scaled cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = attn * scale
        bias = self.cpb(self.rel_coords)  # (heads, L, L)
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(nwb, length, self.dim)
        out = _from_windows(out, self.wh, self.ww, hp, wp)  # (B, Hp, Wp, dim)
        if pad_b or pad_r:
            out = out[:, :h, :w, :].contiguous()
        return out


class _ASWA(nn.Module):
    """Asymmetric Strip Window Attention: square + horizontal + vertical branches summed.

    The three partitioning schemes share a single Q/K/V projection (weight sharing across
    branches) but use separate bias generators; their outputs are summed and projected, so
    the square branch contributes isotropic local context and the two strip branches inject
    axis-aligned long-range dependencies (paper, Sec. 3.3.1).
    """

    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.square = _StripWindowAttention(dim, num_heads, window_size, mode="square")
        self.horizontal = _StripWindowAttention(dim, num_heads, window_size, mode="h")
        self.vertical = _StripWindowAttention(dim, num_heads, window_size, mode="v")
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, h, w):
        """x: (B, H, W, dim) -> (B, H, W, dim)."""
        qkv = self.qkv(x)
        out = self.square(qkv, h, w) + self.horizontal(qkv, h, w) + self.vertical(qkv, h, w)
        return self.proj(out)


class _ASSwin2Block(nn.Module):
    """A single AS-Swin2 transformer block: W-MSA (square) then ASWA, each with LN + MLP residual.

    Feature maps first pass through the square-window branch for local context, then the
    asymmetric strip attention handles strongly anisotropic geometry; residual connections
    and LayerNorm combine the two stages (paper, Sec. 3.3.1, "Module Fusion").
    """

    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _ASWA(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, h, w):
        """x: (B, H*W, dim) -> (B, H*W, dim)."""
        b, n, c = x.shape
        shortcut = x
        y = self.norm1(x).view(b, h, w, c)
        y = self.attn(y, h, w).view(b, h * w, c)
        x = shortcut + y
        x = x + self.mlp(self.norm2(x))
        return x


class ASSwin2(nn.Module):
    """Asymmetric Strip Swin-TransformerV2 (AS-Swin2) module for the YOLO backbone.

    Wraps a stack of :class:`_ASSwin2Block` with a 1x1 input/output projection so it can be
    dropped into the YOLO graph as ``[c1, c2, depth, num_heads, window_size]``. Spatial
    resolution is preserved; only the channel count changes from ``c1`` to ``c2``.

    Args:
        c1 (int): input channels.
        c2 (int): output channels.
        depth (int): number of stacked AS-Swin2 blocks.
        num_heads (int): attention heads per block.
        window_size (int): base square-window edge length ``M``.
    """

    def __init__(self, c1, c2, depth=2, num_heads=4, window_size=7):
        super().__init__()
        self.proj_in = nn.Conv2d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()
        # keep heads compatible with the channel count
        num_heads = max(1, min(num_heads, c2 // 8))
        while c2 % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.blocks = nn.ModuleList(
            [_ASSwin2Block(c2, num_heads, window_size) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(c2)

    def forward(self, x):
        """x: (B, C1, H, W) -> (B, C2, H, W)."""
        x = self.proj_in(x)
        b, c, h, w = x.shape
        y = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        for blk in self.blocks:
            y = blk(y, h, w)
        y = self.norm(y)
        y = y.transpose(1, 2).reshape(b, c, h, w)
        return y


# ---------------------------------------------------------------------------------------
# Frequency-Spatial Dual-Domain Attention (FS-DDA)
# ---------------------------------------------------------------------------------------
def _dct_matrix(n, device, dtype):
    """Build the orthonormal type-II DCT matrix of size (n, n)."""
    i = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    j = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    d = torch.cos(math.pi * (2 * j + 1) * i / (2 * n))
    d[0, :] *= math.sqrt(1.0 / n)
    d[1:, :] *= math.sqrt(2.0 / n)
    return d


def _dct_2d(x, dct_h, dct_w):
    """Separable 2D-DCT applied per channel: x is (B, C, H, W)."""
    # rows then columns: D_h @ x @ D_w^T
    x = torch.einsum("hi,bcij->bchj", dct_h, x)
    x = torch.einsum("bchj,wj->bchw", x, dct_w)
    return x


class _SpectralChannelModulation(nn.Module):
    """2D-DCT spectral channel weighting (Eq. (4) inner term).

    Each ``H x W`` feature map is transformed by a separable 2D-DCT; the spectral maps are
    pooled into a per-channel multi-frequency descriptor and an MLP produces one weight per
    channel. The weights reweight channels by frequency-band energy and the inverse DCT
    reconstructs the modulated feature. DCT bases are cached per spatial size.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.act = nn.Sigmoid()
        self._cache = {}

    def _bases(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key not in self._cache:
            self._cache[key] = (
                _dct_matrix(h, device, dtype),
                _dct_matrix(w, device, dtype),
            )
        return self._cache[key]

    def forward(self, x):
        b, c, h, w = x.shape
        dct_h, dct_w = self._bases(h, w, x.device, x.dtype)
        spec = _dct_2d(x, dct_h, dct_w)  # (B, C, H, W) spectral
        # per-channel multi-frequency descriptor (energy over the spectrum)
        descriptor = spec.abs().mean(dim=(2, 3))  # (B, C)
        weights = self.act(self.mlp(descriptor)).view(b, c, 1, 1)  # (B, C, 1, 1)
        spec = spec * weights  # broadcast channel weights across the spectral map
        # inverse DCT via transposed bases (orthonormal: inverse == transpose)
        x_rec = torch.einsum("ih,bcij->bchj", dct_h, spec)
        x_rec = torch.einsum("bchj,jw->bchw", x_rec, dct_w)
        return x_rec


class _ChannelAttention(nn.Module):
    """No-reduction channel attention (CAM, Eq. (5)).

    Replaces CBAM's reduce-expand bottleneck with a single full-rank 1x1 mixing after global
    average pooling, preserving minor-crack channel information at low cost.
    """

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.fc(self.pool(x)))


class _SpatialAttention(nn.Module):
    """Large-kernel spatial attention (SAM, Eq. (7)).

    Concatenates channel-wise mean and max descriptors and applies a bias-free 7x7
    convolution to enforce local geometric continuity of micro-cracks.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.act(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class FS_DDA(nn.Module):
    """Frequency-Spatial Dual-Domain Attention module.

    Cascades, on a single backbone feature: (1) 2D-DCT spectral channel modulation, (2)
    full-rank channel attention (CAM), and (3) large-kernel spatial attention (SAM). Frequency
    reweighting and spatial refinement act in sequence to separate low-frequency background
    from high-frequency abnormal crack edges (paper, Sec. 3.3.2). Spatial size and channel
    count are preserved, so the module is a drop-in neck refinement taking ``[c1]``.

    Args:
        c1 (int): input/output channels.
        reduction (int): channel-MLP reduction ratio for the spectral stage.
        kernel_size (int): spatial-attention kernel size (7 in the paper).
    """

    def __init__(self, c1, reduction=16, kernel_size=7):
        super().__init__()
        self.spectral = _SpectralChannelModulation(c1, reduction=reduction)
        self.channel = _ChannelAttention(c1)
        self.spatial = _SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)."""
        x = self.spectral(x)
        x = self.channel(x)
        x = self.spatial(x)
        return x
