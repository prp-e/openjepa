"""
Pre-norm Vision Transformer backbone (Section 2), used identically for both the context
encoder and the target encoder (separate instances, separate/independently-updated params).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.proj(x)  # (B, D, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D); N flattened row-major (row*grid_w+col)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Pre-norm transformer block, reused by both the encoder(s) and the predictor
    (predictor instantiates its own Blocks at predictor_embed_dim)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


VIT_CONFIGS = {
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4),
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4),
    "vit_large": dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4),
    "vit_huge": dict(embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4),
}


class VisionTransformer(nn.Module):
    """
    No [CLS] token -- I-JEPA operates purely on patch tokens (Section 2, item 4).
    Ends with a LayerNorm (`self.norm`), which is what satisfies the spec's requirement that
    the TARGET encoder's output be "LayerNorm-normalized": since both encoders share this
    architecture, that normalization is built in for free and applies identically when this
    class is used as either the context or the target encoder.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.0, attn_drop_rate: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size, self.grid_size)  # (N, D)
        self.register_buffer("pos_embed", pos_embed, persistent=False)  # fixed, not learned

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        # Section 2, item 5: truncated normal std=0.02 for Linear/Conv2d weights, zero biases,
        # LayerNorm weight=1 / bias=0.
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, patch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        patch_indices: optional (N_sel,) LongTensor, SHARED across the batch (Section 3/4
            decision). If provided, only those patches are processed -- this is how the
            "context encoder receives visible context patches only" is implemented: full
            patch-embedding + positional-embedding happens first, then irrelevant tokens are
            simply gathered away before the transformer blocks run (cheap, exact, no padding).
            If None (target encoder case), ALL patches are processed.
        returns: (B, N or N_sel, D)
        """
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed.unsqueeze(0)  # (B, N, D) -- fixed positional embedding, added once
        if patch_indices is not None:
            x = x[:, patch_indices, :]  # (B, N_sel, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # final LayerNorm -> satisfies "then LayerNorm-normalized" for target encoder
        return x


def build_vit(name: str, img_size: int, patch_size: int, in_chans: int = 3) -> VisionTransformer:
    if name not in VIT_CONFIGS:
        raise ValueError(f"Unknown encoder_name '{name}'. Choose from {list(VIT_CONFIGS.keys())}.")
    cfg = VIT_CONFIGS[name]
    return VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, **cfg)
