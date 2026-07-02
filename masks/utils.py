"""
Patch-index helper functions for multi-block masking (Section 3).
Patch grid positions (row, col) are flattened row-major: flat_index = row * grid_w + col,
matching the flattening order produced by `PatchEmbed` in models/vision_transformer.py.
"""
from __future__ import annotations

from typing import List, Set

import torch


def block_to_indices(top: int, left: int, h: int, w: int, grid_w: int) -> Set[int]:
    """Convert a rectangular block of patches (top-left corner + height/width) to a
    set of flat, row-major patch indices."""
    return {r * grid_w + c for r in range(top, top + h) for c in range(left, left + w)}


def indices_to_tensor(indices: Set[int]) -> torch.Tensor:
    """Deterministic (sorted) conversion of an index set to a LongTensor."""
    return torch.tensor(sorted(indices), dtype=torch.long)


def remove_overlap(context_indices: Set[int], target_indices_list: List[Set[int]]) -> Set[int]:
    """Set difference: remove every target-block patch from the context block (Section 3)."""
    out = set(context_indices)
    for t in target_indices_list:
        out -= t
    return out
