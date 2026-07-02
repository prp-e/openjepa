"""
extensions/rasterizer.py

Pure-PyTorch, dependency-free DIFFERENTIABLE rasterizer.

v2 CHANGE: renders TWO composited primitives per patch instead of one -- a
filled rotated ellipse (bottom layer) plus a stroked cubic Bezier curve
(top layer) -- matching the two-primitive output of v2
extensions/svg_decoder.VectorPathDecoder. This is what lets the model
represent solid-colored, flat icon art instead of only thin outlines.

Each patch renders into its own disjoint (patch_size x patch_size) tile in
parallel; tiles are reassembled into the full canvas via reshape/permute
(patches are non-overlapping, so no cross-patch compositing is needed),
keeping the whole operation vectorized and end-to-end differentiable.
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


def _stroke_coverage(control_points_px, stroke_width, pixel_grid, num_curve_samples, sharpness):
    """Returns (N, P, P) soft coverage in [0, 1] for the stroke curve."""
    n, patch_size = control_points_px.shape[0], pixel_grid.shape[0]
    curve_pts = _sample_bezier(control_points_px, num_curve_samples)  # (N, K, 2)

    diff = pixel_grid.view(1, patch_size, patch_size, 1, 2) - curve_pts.view(n, 1, 1, num_curve_samples, 2)
    dist = diff.pow(2).sum(-1).sqrt().min(dim=-1).values  # (N, P, P)

    half_width = (stroke_width / 2.0).view(n, 1, 1)
    return torch.sigmoid((half_width - dist) * sharpness)


def _fill_coverage(center_px, radius_px, rotation, pixel_grid, sharpness):
    """
    Returns (N, P, P) soft coverage in [0, 1] for a rotated ellipse, via an
    implicit inside/outside test smoothed by a sigmoid -- analogous to the
    stroke's distance-to-curve sigmoid, but for filled area.

    center_px: (N, 2), radius_px: (N, 2) = (rx, ry), rotation: (N, 1) radians.
    """
    n, patch_size = center_px.shape[0], pixel_grid.shape[0]

    rel = pixel_grid.view(1, patch_size, patch_size, 2) - center_px.view(n, 1, 1, 2)

    cos_r = torch.cos(-rotation).view(n, 1, 1)
    sin_r = torch.sin(-rotation).view(n, 1, 1)
    x = rel[..., 0]
    y = rel[..., 1]
    x_rot = x * cos_r - y * sin_r
    y_rot = x * sin_r + y * cos_r

    rx = radius_px[:, 0].view(n, 1, 1).clamp(min=1e-3)
    ry = radius_px[:, 1].view(n, 1, 1).clamp(min=1e-3)

    # implicit ellipse equation: (x/rx)^2 + (y/ry)^2 - 1  (negative = inside)
    signed_dist = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 - 1.0
    return torch.sigmoid(-signed_dist * sharpness)


def rasterize_patches(
    control_points: torch.Tensor,
    stroke_color: torch.Tensor,
    stroke_width: torch.Tensor,
    stroke_opacity: torch.Tensor,
    fill_center: torch.Tensor,
    fill_radius: torch.Tensor,
    fill_rotation: torch.Tensor,
    fill_color: torch.Tensor,
    fill_opacity: torch.Tensor,
    grid_size: int,
    patch_size: int,
    num_curve_samples: int = 20,
    sharpness: float = 3.0,
) -> torch.Tensor:
    """
    Every positional input is normalized [0, 1] (or [0, 1]-fraction-of-
    patch-size for radius) patch-local coordinates, matching
    VectorPathDecoder's output. Scaling to pixels happens internally.

    N must equal grid_size * grid_size (a FULL grid).
    Returns: (H, W, 3) image tensor in [0, 1], H = W = grid_size * patch_size,
    differentiable w.r.t. every input above.
    """
    device = control_points.device
    n = control_points.shape[0]
    assert n == grid_size * grid_size, (
        f"rasterize_patches expects a FULL grid ({grid_size * grid_size} patches), got {n}."
    )

    coords = torch.arange(patch_size, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    pixel_grid = torch.stack([grid_x, grid_y], dim=-1)  # (P, P, 2)

    # --- fill layer (bottom) ---
    fill_center_px = fill_center * patch_size
    fill_radius_px = fill_radius * patch_size
    fill_cov = _fill_coverage(fill_center_px, fill_radius_px, fill_rotation, pixel_grid, sharpness)
    fill_alpha = (fill_cov * fill_opacity.view(n, 1, 1)).unsqueeze(-1)  # (N, P, P, 1)

    white = torch.ones(n, patch_size, patch_size, 3, device=device)
    base = white * (1 - fill_alpha) + fill_color.view(n, 1, 1, 3) * fill_alpha

    # --- stroke layer (top) ---
    control_points_px = control_points * patch_size
    stroke_cov = _stroke_coverage(control_points_px, stroke_width, pixel_grid, num_curve_samples, sharpness)
    stroke_alpha = (stroke_cov * stroke_opacity.view(n, 1, 1)).unsqueeze(-1)  # (N, P, P, 1)

    tiles = base * (1 - stroke_alpha) + stroke_color.view(n, 1, 1, 3) * stroke_alpha  # (N, P, P, 3)

    tiles = tiles.view(grid_size, grid_size, patch_size, patch_size, 3)
    canvas = tiles.permute(0, 2, 1, 3, 4).reshape(grid_size * patch_size, grid_size * patch_size, 3)
    return canvas