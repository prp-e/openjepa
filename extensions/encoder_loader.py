"""
Loads a FROZEN target encoder from a JEPA training checkpoint, for use by decoder
training. This intentionally bypasses masking/the predictor entirely -- it runs the
target encoder over the whole image to get one latent per patch, covering the full
grid (unlike train.py --dump_latents, which only exports the masked-subset latents
used during JEPA pretraining).

Does NOT modify models/vision_transformer.py -- only imports and instantiates it.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from models.vision_transformer import VisionTransformer  # not modified


def load_frozen_target_encoder(checkpoint_path: str, device: str = "cpu") -> Tuple[nn.Module, dict]:
    """
    Returns (target_encoder, full_config).

    NOTE: the VisionTransformer(...) call below assumes the same constructor
    signature (img_size, patch_size, embed_dim, depth, num_heads) used elsewhere in
    this project. If your models/vision_transformer.py uses different argument
    names, THIS is the one place to adjust.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model_cfg = cfg.get("model", {})

    encoder = VisionTransformer(
        img_size=cfg["data"]["img_size"],
        patch_size=cfg["data"]["patch_size"],
        embed_dim=model_cfg.get("embed_dim", 384),
        depth=model_cfg.get("depth", 12),
        num_heads=model_cfg.get("num_heads", 6),
    )
    encoder.load_state_dict(ckpt["target_encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.to(device)
    return encoder, cfg


@torch.no_grad()
def encode_full_grid(encoder: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """
    image: (1, 3, H, W) preprocessed tensor.
    Returns per-patch latents (num_patches, embed_dim) in row-major raster order --
    this ordering is required by extensions/rasterizer.py's tile-reassembly step.
    """
    latents = encoder(image)  # expected shape: (1, num_patches, embed_dim)
    if latents.dim() == 3:
        latents = latents.squeeze(0)
    return latents