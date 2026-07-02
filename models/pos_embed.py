"""
Fixed (non-learned) 2D sin-cos positional embeddings (Section 2, item 2).
These are registered as buffers, never as nn.Parameter -- gradients never flow into them.
"""
from __future__ import annotations

import numpy as np
import torch


def get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension per position (must be even).
    positions: 1D array, shape (M,), of position indices (row OR col indices).
    returns:   (M, embed_dim) array, interleaved as
               [sin(p*w0), cos(p*w0), sin(p*w1), cos(p*w1), ...] per the spec formula.
    """
    assert embed_dim % 2 == 0, "embed_dim must be even for sin-cos pos embed"
    omega = 1.0 / (10000 ** (np.arange(embed_dim // 2, dtype=np.float64) * 2 / embed_dim))  # (D/2,)
    positions = positions.reshape(-1).astype(np.float64)  # (M,)
    args = np.einsum("m,d->md", positions, omega)  # (M, D/2)
    emb = np.zeros((positions.shape[0], embed_dim), dtype=np.float64)
    emb[:, 0::2] = np.sin(args)
    emb[:, 1::2] = np.cos(args)
    return emb  # (M, embed_dim)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    Splits embed_dim into two equal halves: first half encodes row index, second half
    encodes column index (Section 2, item 2). Patches are flattened row-major
    (flat_index = row * grid_w + col), matching PatchEmbed's flatten order.

    returns: (grid_h * grid_w, embed_dim) float32 tensor.
    """
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    row_idx, col_idx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    row_idx = row_idx.reshape(-1)  # (N,) row-major flatten
    col_idx = col_idx.reshape(-1)
    pe_row = get_1d_sincos_pos_embed(half, row_idx)  # (N, half)
    pe_col = get_1d_sincos_pos_embed(half, col_idx)  # (N, half)
    pe = np.concatenate([pe_row, pe_col], axis=1)  # (N, embed_dim)
    return torch.from_numpy(pe).float()
