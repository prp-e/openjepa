"""
MaskCollator: implements the multi-block masking strategy of Section 3 as a DataLoader
`collate_fn`.

DESIGN DECISION (documented, see top-level design notes): masks are sampled ONCE PER BATCH
(shared across every image in the batch), not once per image. This is required for the
straightforward tensor-gather batching trick in Section 4 to be implementable without ragged
(variable-length) indexing, and matches the reference I-JEPA collator's behavior.

DESIGN DECISION #2: all `num_target_blocks` target blocks share the SAME sampled
(height, width) shape but have INDEPENDENT random positions. This guarantees identical
N_tgt across all target blocks, which is required for the repeat_interleave_batch trick
(models/predictor.py) to run as a single stacked forward pass instead of a Python loop.
"""
from __future__ import annotations

import math
import random
from typing import List, Set, Tuple

import torch

from masks.utils import block_to_indices, indices_to_tensor, remove_overlap


class MaskCollator:
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_target_blocks: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.20),
        context_scale: Tuple[float, float] = (0.85, 1.0),
        aspect_ratio_range: Tuple[float, float] = (0.75, 1.5),
        min_keep_patches: int = 10,
        allow_overlap: bool = False,
        max_retries: int = 20,
    ):
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_target_blocks = num_target_blocks
        self.target_scale = target_scale
        self.context_scale = context_scale
        self.aspect_ratio_range = aspect_ratio_range
        self.min_keep_patches = min_keep_patches
        self.allow_overlap = allow_overlap
        self.max_retries = max_retries

    def _sample_block_shape(self, scale_range, ar_range) -> Tuple[int, int]:
        area = random.uniform(*scale_range) * self.grid_h * self.grid_w
        ar = random.uniform(*ar_range)
        h = int(round(math.sqrt(area * ar)))
        w = int(round(math.sqrt(area / ar)))
        h = max(1, min(h, self.grid_h))
        w = max(1, min(w, self.grid_w))
        return h, w

    def _sample_position(self, h: int, w: int) -> Tuple[int, int]:
        top = random.randint(0, self.grid_h - h)
        left = random.randint(0, self.grid_w - w)
        return top, left

    def sample_masks(self) -> Tuple[Set[int], List[Set[int]]]:
        """Runs the Section 3 algorithm (with the two documented design decisions above).
        Returns (context_index_set, [target_index_set, ...])."""
        target_sets: List[Set[int]] = []
        for _ in range(self.max_retries):
            # --- target blocks: shape sampled once, positions sampled independently ---
            th, tw = self._sample_block_shape(self.target_scale, self.aspect_ratio_range)
            target_sets = []
            for _ in range(self.num_target_blocks):
                top, left = self._sample_position(th, tw)
                target_sets.append(block_to_indices(top, left, th, tw, self.grid_w))

            # --- context block: square-ish (aspect ratio fixed to 1.0 per Section 3) ---
            ch, cw = self._sample_block_shape(self.context_scale, (1.0, 1.0))
            ctop, cleft = self._sample_position(ch, cw)
            context_set = block_to_indices(ctop, cleft, ch, cw, self.grid_w)

            if not self.allow_overlap:
                context_set = remove_overlap(context_set, target_sets)

            if len(context_set) >= self.min_keep_patches:
                return context_set, target_sets

        # Fallback (Section 3): full grid minus the union of target blocks.
        full = set(range(self.grid_h * self.grid_w))
        context_set = remove_overlap(full, target_sets) if not self.allow_overlap else full
        return context_set, target_sets

    def __call__(self, batch: List[torch.Tensor]):
        """
        batch: list of (3, H, W) image tensors from the Dataset.

        Returns:
          images:        (B, 3, H, W)      float tensor
          context_mask:  (N_ctx,)          LongTensor, SHARED across the whole batch
          target_masks:  (T, N_tgt)        LongTensor, SHARED across the whole batch
        """
        images = torch.stack(batch, dim=0)  # (B, 3, H, W)
        context_set, target_sets = self.sample_masks()
        context_mask = indices_to_tensor(context_set)  # (N_ctx,)
        target_masks = torch.stack([indices_to_tensor(t) for t in target_sets], dim=0)  # (T, N_tgt)
        return images, context_mask, target_masks
