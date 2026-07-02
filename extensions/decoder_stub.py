"""
Abstract interface for a future latent-to-output decoder (Section 9.2).
NOT implemented here -- this is only the required extension point.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LatentToOutputDecoder(nn.Module):
    """
    Abstract base for decoding I-JEPA latents into a visible output modality.

    A future `VectorPathDecoder(LatentToOutputDecoder)` subclass would output path/coordinate
    primitives per region (e.g., SVG <path> control points, per-region fill/stroke parameters)
    rather than pixels, and should use `positions` to place each primitive correctly within
    the overall image composition.

    HYPOTHESIS (UNVERIFIED by this codebase): I-JEPA's texture-invariant, structure-focused
    representations may be well suited to vector/SVG generation, since both discard fine
    pixel-level texture in favor of geometric/structural content. This must be tested
    empirically once a concrete decoder subclass is trained -- nothing in this pretraining
    pipeline confirms or denies it.
    """

    def forward(self, latents: torch.Tensor, positions: torch.Tensor):
        """
        latents:   (B, N, D)  -- predicted or encoded per-patch latents
        positions: (B, N, 2)  -- (row, col) patch grid coordinates for each latent
        returns:   modality-specific output (e.g., pixels, or SVG path parameters)
        """
        raise NotImplementedError(
            "LatentToOutputDecoder is an abstract interface. Subclass it (e.g., "
            "VectorPathDecoder) to implement a concrete downstream decoder."
        )
