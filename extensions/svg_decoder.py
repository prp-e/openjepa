"""
extensions/svg_decoder.py

Concrete latent -> vector-graphics-parameters decoder.

--------------------------------------------------------------------------
v2 CHANGE: the original version mapped each patch's latent to a single
STROKED cubic Bezier curve only. That output space cannot represent solid,
flat-filled regions -- which is exactly what minimalistic vector icon art
is made of -- so reconstruction loss against real icon datasets stayed
high regardless of training, because the model was never able to draw the
shape it needed to.

This version adds a second, filled primitive per patch: a rotated ellipse
with its own color/opacity, composited UNDERNEATH the stroke. Each patch
can now paint a flat-colored blob (fill), a thin outline (stroke), both,
or effectively neither (near-zero learned opacity on one).

This changes the output dict's keys/shapes vs. the original version.
Previously-saved checkpoints/decoder.pt are NOT compatible -- retrain via
train_decoder.py.
--------------------------------------------------------------------------

INTEGRATION NOTE: extensions/decoder_stub.py is described elsewhere in this
project as defining an abstract interface for latent -> output decoders.
Its exact class name/method signature is not assumed here (guessing wrong
here would repeat the class-name mismatch already hit once with the
dataset loader) -- VectorPathDecoder below is a plain nn.Module. If
decoder_stub.py requires subclassing an ABC, swap nn.Module for that base
class and confirm the forward() signature matches.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class VectorPathDecoder(nn.Module):
    """
    Maps one latent vector per patch to two composited vector primitives:

      STROKE -- a cubic Bezier curve (4 control points) + color/width/opacity.
      FILL   -- a rotated ellipse (center, radii, rotation) + color/opacity,
                rendered underneath the stroke.

    All spatial outputs are normalized [0, 1] patch-local coordinates --
    extensions/rasterizer.py handles scaling to actual pixel coordinates.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # +2 for normalized (row, col) position -- explicit spatial context.
        in_dim = latent_dim + 2

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Stroke heads
        self.stroke_points_head = nn.Linear(hidden_dim, 8)   # 4 points x (x, y)
        self.stroke_color_head = nn.Linear(hidden_dim, 3)
        self.stroke_width_head = nn.Linear(hidden_dim, 1)
        self.stroke_opacity_head = nn.Linear(hidden_dim, 1)

        # Fill heads
        self.fill_center_head = nn.Linear(hidden_dim, 2)
        self.fill_radius_head = nn.Linear(hidden_dim, 2)
        self.fill_rotation_head = nn.Linear(hidden_dim, 1)
        self.fill_color_head = nn.Linear(hidden_dim, 3)
        self.fill_opacity_head = nn.Linear(hidden_dim, 1)

    def forward(self, latents: torch.Tensor, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        latents:   (N, latent_dim)
        positions: (N, 2) integer (row, col) grid coordinates.

        Returns a dict of per-patch primitive parameters:
            control_points  (N, 4, 2)  in [0, 1], patch-local
            stroke_color    (N, 3)     in [0, 1]
            stroke_width    (N, 1)     in pixels (> 0)
            stroke_opacity  (N, 1)     in [0, 1]
            fill_center     (N, 2)     in [0, 1], patch-local
            fill_radius     (N, 2)     in [0, 1], fraction of patch size
            fill_rotation   (N, 1)     in radians
            fill_color      (N, 3)     in [0, 1]
            fill_opacity    (N, 1)     in [0, 1]
        """
        n = latents.shape[0]
        device = latents.device

        grid_size = positions.max().item() + 1 if n > 0 else 1
        norm_positions = positions.float() / max(grid_size - 1, 1)  # -> [0, 1]

        x = torch.cat([latents, norm_positions.to(device)], dim=-1)
        h = self.trunk(x)

        control_points = torch.sigmoid(self.stroke_points_head(h)).view(n, 4, 2)
        stroke_color = torch.sigmoid(self.stroke_color_head(h))
        stroke_width = torch.nn.functional.softplus(self.stroke_width_head(h)) + 0.5
        stroke_opacity = torch.sigmoid(self.stroke_opacity_head(h))

        fill_center = torch.sigmoid(self.fill_center_head(h))
        fill_radius = torch.sigmoid(self.fill_radius_head(h)) * 0.75 + 0.05  # keep away from 0
        fill_rotation = torch.tanh(self.fill_rotation_head(h)) * math.pi
        fill_color = torch.sigmoid(self.fill_color_head(h))
        fill_opacity = torch.sigmoid(self.fill_opacity_head(h))

        return {
            "control_points": control_points,
            "stroke_color": stroke_color,
            "stroke_width": stroke_width,
            "stroke_opacity": stroke_opacity,
            "fill_center": fill_center,
            "fill_radius": fill_radius,
            "fill_rotation": fill_rotation,
            "fill_color": fill_color,
            "fill_opacity": fill_opacity,
        }