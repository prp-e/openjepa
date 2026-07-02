"""
Concrete decoder: latents -> SVG vector path parameters.

This is the first CONCRETE implementation of the abstract `LatentToOutputDecoder`
interface defined in `extensions/decoder_stub.py` (left untouched). It does NOT modify or
depend on any change to the core I-JEPA architecture (models/, engine/, masks/, data/,
train.py) -- it only CONSUMES the per-patch latents that pipeline already produces via
`train.py --dump_latents`.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from extensions.decoder_stub import LatentToOutputDecoder


class VectorPathDecoder(LatentToOutputDecoder):
    """
    Maps one latent vector per image patch to the parameters of a single cubic Bezier
    curve "belonging" to that patch's grid cell, plus basic stroke styling.

    Per-patch output (13 values total):
      - 4 control points (x, y), each in [0, 1], relative to the patch's own cell
        (P0..P3 of a cubic Bezier: "M P0 C P1 P2 P3")
      - stroke color (r, g, b), each in [0, 1]
      - stroke width (positive)
      - stroke opacity, in [0, 1]
    """

    CONTROL_POINTS = 4
    OUTPUT_DIM = CONTROL_POINTS * 2 + 3 + 1 + 1  # = 13

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.OUTPUT_DIM),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, latents: torch.Tensor, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        latents:   (N, latent_dim)  -- per-patch latents (e.g. loaded from latents.pt)
        positions: (N, 2)           -- (row, col) patch grid coordinates, passed through
                                        unchanged so the renderer knows where to place each curve

        returns: dict with control_points (N,4,2), color (N,3), stroke_width (N,1),
                 opacity (N,1), positions (N,2)
        """
        raw = self.net(latents)  # (N, 13)
        cp_raw, color_raw, width_raw, opacity_raw = torch.split(raw, [8, 3, 1, 1], dim=-1)

        control_points = torch.sigmoid(cp_raw).reshape(-1, self.CONTROL_POINTS, 2)  # in [0,1]
        color = torch.sigmoid(color_raw)  # in [0,1]
        stroke_width = torch.nn.functional.softplus(width_raw) + 0.5  # >= 0.5 px
        opacity = torch.sigmoid(opacity_raw)  # in [0,1]

        return {
            "control_points": control_points,
            "color": color,
            "stroke_width": stroke_width,
            "opacity": opacity,
            "positions": positions,
        }