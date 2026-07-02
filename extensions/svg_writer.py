"""
Minimal, dependency-free SVG string builder. Deliberately avoids a new third-party
dependency (e.g. svgwrite) -- requirements.txt stays exactly as-is.
"""
from __future__ import annotations

from typing import Dict

import torch


def _to_hex_color(rgb: torch.Tensor) -> str:
    """rgb: (3,) tensor in [0, 1] -> '#rrggbb' hex string."""
    r, g, b = (rgb.clamp(0, 1) * 255).round().long().tolist()
    return f"#{r:02x}{g:02x}{b:02x}"


def decoded_to_svg(decoded: Dict[str, torch.Tensor], patch_size: int, grid_size: int) -> str:
    """
    decoded: output dict from VectorPathDecoder.forward().
    patch_size: pixel size of one grid cell -- MUST match the patch_size used when the
                original latents were produced (pass it explicitly; latents.pt doesn't
                store it).
    grid_size: patches per side, used only to size the canvas.

    returns: a complete, self-contained SVG document as a string.
    """
    control_points = decoded["control_points"].detach().cpu()  # (N, 4, 2)
    color = decoded["color"].detach().cpu()                    # (N, 3)
    stroke_width = decoded["stroke_width"].detach().cpu()       # (N, 1)
    opacity = decoded["opacity"].detach().cpu()                 # (N, 1)
    positions = decoded["positions"].detach().cpu()             # (N, 2) -- (row, col)

    canvas_px = grid_size * patch_size
    paths = []
    for i in range(control_points.shape[0]):
        row, col = positions[i].tolist()
        top = row * patch_size
        left = col * patch_size

        pts = control_points[i] * patch_size  # local pixel offsets within the patch
        pts = pts + torch.tensor([left, top])  # shift into absolute canvas coordinates

        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts.tolist()
        hex_color = _to_hex_color(color[i])
        width = float(stroke_width[i].item())
        alpha = float(opacity[i].item())

        d = f"M {x0:.2f},{y0:.2f} C {x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f} {x3:.2f},{y3:.2f}"
        paths.append(
            f'<path d="{d}" fill="none" stroke="{hex_color}" '
            f'stroke-width="{width:.2f}" stroke-opacity="{alpha:.3f}" />'
        )

    body = "\n  ".join(paths)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_px}" height="{canvas_px}" '
        f'viewBox="0 0 {canvas_px} {canvas_px}">\n'
        f'  <rect width="100%" height="100%" fill="white" />\n'
        f"  {body}\n"
        f"</svg>\n"
    )
    return svg


def save_svg(svg_string: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_string)