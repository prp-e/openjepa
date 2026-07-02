"""
Minimal differentiable rasterizer: renders per-patch cubic Bezier stroke parameters
(as produced by extensions/svg_decoder.VectorPathDecoder) directly into a raster
image tensor, in pure PyTorch -- no new dependencies.

Each patch's curve renders into its own disjoint (patch_size x patch_size) tile, in
parallel across all patches. Tiles are reassembled into the full canvas via a
reshape/permute (patches are non-overlapping, so no scatter/compositing across
patches is needed) -- keeping the whole op vectorized and end-to-end differentiable.

Requires latents/curves to be in FULL-GRID row-major order (see
extensions/encoder_loader.encode_full_grid), not the masked-subset ordering used by
train.py --dump_latents.
"""
from __future__ import annotations

import torch


def full_grid_positions(grid_size: int, device=None) -> torch.Tensor:
    """Row-major (row, col) positions covering a complete grid_size x grid_size grid."""
    rows = torch.arange(grid_size, device=device).view(-1, 1).expand(grid_size, grid_size)
    cols = torch.arange(grid_size, device=device).view(1, -1).expand(grid_size, grid_size)
    return torch.stack([rows, cols], dim=-1).reshape(-1, 2)


def _sample_bezier(control_points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """control_points: (N, 4, 2) -> (N, num_samples, 2) points along each curve."""
    device = control_points.device
    t = torch.linspace(0.0, 1.0, num_samples, device=device)
    p0, p1, p2, p3 = control_points.unbind(dim=1)

    mt = 1.0 - t
    b0 = (mt ** 3).unsqueeze(-1)
    b1 = (3 * mt ** 2 * t).unsqueeze(-1)
    b2 = (3 * mt * t ** 2).unsqueeze(-1)
    b3 = (t ** 3).unsqueeze(-1)

    return (
        p0.unsqueeze(1) * b0 + p1.unsqueeze(1) * b1
        + p2.unsqueeze(1) * b2 + p3.unsqueeze(1) * b3
    )  # (N, K, 2)


def rasterize_patches(
    control_points: torch.Tensor,
    color: torch.Tensor,
    stroke_width: torch.Tensor,
    opacity: torch.Tensor,
    grid_size: int,
    patch_size: int,
    num_curve_samples: int = 20,
    sharpness: float = 3.0,
) -> torch.Tensor:
    """
    control_points: (N, 4, 2) normalized [0,1] patch-local coords (decoder output)
    color:          (N, 3) in [0, 1]
    stroke_width:   (N, 1) in pixels
    opacity:        (N, 1) in [0, 1]
    N must equal grid_size * grid_size (a FULL grid).

    Returns: (H, W, 3) image tensor in [0, 1], H = W = grid_size * patch_size,
    differentiable w.r.t. all four inputs.
    """
    device = control_points.device
    n = control_points.shape[0]
    assert n == grid_size * grid_size, (
        f"rasterize_patches expects a FULL grid ({grid_size * grid_size} patches), got {n}."
    )

    cp_px = control_points * patch_size
    curve_pts = _sample_bezier(cp_px, num_curve_samples)  # (N, K, 2)

    coords = torch.arange(patch_size, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    pixel_grid = torch.stack([grid_x, grid_y], dim=-1)  # (P, P, 2)

    diff = pixel_grid.view(1, patch_size, patch_size, 1, 2) - curve_pts.view(n, 1, 1, num_curve_samples, 2)
    dist = diff.pow(2).sum(-1).sqrt().min(dim=-1).values  # (N, P, P) - nearest sampled curve point

    half_width = (stroke_width / 2.0).view(n, 1, 1)
    coverage = torch.sigmoid((half_width - dist) * sharpness)  # soft anti-aliased edge

    alpha = (coverage * opacity.view(n, 1, 1)).unsqueeze(-1)     # (N, P, P, 1)
    color_map = color.view(n, 1, 1, 3)

    white = torch.ones(n, patch_size, patch_size, 3, device=device)
    tiles = white * (1 - alpha) + color_map * alpha  # (N, P, P, 3)

    tiles = tiles.view(grid_size, grid_size, patch_size, patch_size, 3)
    canvas = tiles.permute(0, 2, 1, 3, 4).reshape(grid_size * patch_size, grid_size * patch_size, 3)
    return canvas