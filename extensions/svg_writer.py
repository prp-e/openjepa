"""
extensions/svg_writer.py

Dependency-free SVG string builder. Assembles per-patch decoded primitives
(see extensions/svg_decoder.VectorPathDecoder) into a single, complete,
standalone SVG document -- no external XML library needed.

v2 CHANGE: each patch now emits TWO elements instead of one -- a filled
<ellipse> (drawn first, i.e. underneath) and a stroked cubic Bezier <path>
(drawn second, i.e. on top) -- mirroring the two-layer compositing done by
extensions/rasterizer.py, so the SVG matches the training-time raster
preview instead of showing only the stroke.
"""
from __future__ import annotations

import math
from typing import Dict

import torch


def _rgb_to_hex(color: torch.Tensor) -> str:
    r, g, b = [max(0, min(255, int(round(c * 255)))) for c in color.tolist()]
    return f"#{r:02x}{g:02x}{b:02x}"


def _patch_offset(row: int, col: int, patch_size: int) -> tuple:
    return col * patch_size, row * patch_size


def build_svg_document(
    decoded: Dict[str, torch.Tensor],
    positions: torch.Tensor,
    patch_size: int,
    grid_size: int,
) -> str:
    """
    decoded:   dict of per-patch tensors, exactly the output of
               VectorPathDecoder.forward().
    positions: (N, 2) integer (row, col) grid coordinates, same order as
               every tensor inside `decoded`.
    """
    canvas_size = grid_size * patch_size
    n = positions.shape[0]

    elements = []
    for i in range(n):
        row, col = int(positions[i, 0].item()), int(positions[i, 1].item())
        off_x, off_y = _patch_offset(row, col, patch_size)

        # --- fill ellipse (bottom layer) ---
        fcx = off_x + decoded["fill_center"][i, 0].item() * patch_size
        fcy = off_y + decoded["fill_center"][i, 1].item() * patch_size
        frx = decoded["fill_radius"][i, 0].item() * patch_size
        fry = decoded["fill_radius"][i, 1].item() * patch_size
        f_rot_deg = math.degrees(decoded["fill_rotation"][i, 0].item())
        fill_hex = _rgb_to_hex(decoded["fill_color"][i])
        fill_opacity = max(0.0, min(1.0, decoded["fill_opacity"][i, 0].item()))

        elements.append(
            f'<ellipse cx="{fcx:.2f}" cy="{fcy:.2f}" rx="{frx:.2f}" ry="{fry:.2f}" '
            f'transform="rotate({f_rot_deg:.2f} {fcx:.2f} {fcy:.2f})" '
            f'fill="{fill_hex}" fill-opacity="{fill_opacity:.3f}" />'
        )

        # --- stroke curve (top layer) ---
        pts = decoded["control_points"][i] * patch_size
        pts = pts + torch.tensor([off_x, off_y])
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts.tolist()
        stroke_hex = _rgb_to_hex(decoded["stroke_color"][i])
        stroke_width = decoded["stroke_width"][i, 0].item()
        stroke_opacity = max(0.0, min(1.0, decoded["stroke_opacity"][i, 0].item()))

        elements.append(
            f'<path d="M {x0:.2f} {y0:.2f} C {x1:.2f} {y1:.2f}, {x2:.2f} {y2:.2f}, {x3:.2f} {y3:.2f}" '
            f'fill="none" stroke="{stroke_hex}" stroke-width="{stroke_width:.2f}" '
            f'stroke-opacity="{stroke_opacity:.3f}" stroke-linecap="round" />'
        )

    body = "\n  ".join(elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" '
        f'viewBox="0 0 {canvas_size} {canvas_size}">\n'
        f'  <rect x="0" y="0" width="{canvas_size}" height="{canvas_size}" fill="white" />\n'
        f'  {body}\n'
        f'</svg>\n'
    )