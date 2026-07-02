"""
I-JEPA training / inference entry point.

USAGE:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --dump_latents \
        --checkpoint checkpoints/ckpt_epoch0.pt --image path/to/img.jpg --out latents.pt

DESIGN NOTE (Section 9.4): I-JEPA's representations are texture-invariant and
structure-focused because the target encoder is asked to predict abstract latents of masked
regions rather than reconstruct pixels, and no pixel-level augmentation is used. This is
HYPOTHESIZED to make these representations well suited to a future vector/SVG-primitive
decoder, since both discard fine pixel texture in favor of geometric/structural content --
but this hypothesis is UNVERIFIED by this codebase and must be tested empirically once a
downstream decoder (see extensions/decoder_stub.py) is actually trained.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import IMAGENET_MEAN, IMAGENET_STD, LocalImageFolderDataset, SyntheticDataset
from engine.ema import ema_momentum_schedule
from engine.train_one_epoch import train_one_epoch
from masks.multiblock import MaskCollator
from masks.utils import indices_to_tensor
from models.predictor import Predictor
from models.vision_transformer import build_vit
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.schedulers import lr_schedule, scale_lr_for_batch_size, weight_decay_schedule


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_models(cfg: Dict[str, Any], device: torch.device):
    m, d = cfg["model"], cfg["data"]
    context_encoder = build_vit(m["encoder_name"], d["img_size"], d["patch_size"]).to(device)

    # Target encoder starts as a deep copy of the context encoder (Section 1), then is frozen
    # w.r.t. autograd -- it is updated ONLY via the EMA rule (Section 6).
    target_encoder = copy.deepcopy(context_encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    grid_size = d["img_size"] // d["patch_size"]
    predictor = Predictor(
        encoder_embed_dim=context_encoder.embed_dim,
        grid_size=grid_size,
        predictor_embed_dim=m["predictor_embed_dim"],
        depth=m["predictor_depth"],
        num_heads=m["predictor_num_heads"],
        mlp_ratio=m["predictor_mlp_ratio"],
    ).to(device)
    return context_encoder, target_encoder, predictor


def build_mask_collator(cfg: Dict[str, Any]) -> MaskCollator:
    d, mk = cfg["data"], cfg["mask"]
    return MaskCollator(
        img_size=d["img_size"],
        patch_size=d["patch_size"],
        num_target_blocks=mk["num_target_blocks"],
        target_scale=tuple(mk["target_scale"]),
        context_scale=tuple(mk["context_scale"]),
        aspect_ratio_range=tuple(mk["aspect_ratio_range"]),
        min_keep_patches=mk["min_keep_patches"],
        allow_overlap=mk["allow_overlap"],
    )


def build_dataloader(cfg: Dict[str, Any]) -> DataLoader:
    d = cfg["data"]
    if d.get("use_synthetic", True) or not d.get("root"):
        dataset = SyntheticDataset(num_samples=d["synthetic_num_samples"], img_size=d["img_size"])
    else:
        dataset = LocalImageFolderDataset(root=d["root"], img_size=d["img_size"], train=True)

    collator = build_mask_collator(cfg)
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=d.get("num_workers", 0),
        collate_fn=collator,
        drop_last=True,
    )


def train(cfg: Dict[str, Any]) -> None:
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_bf16 = cfg["train"]["use_bf16"] and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if cfg["train"]["use_bf16"] and not use_bf16:
        print("[train] bfloat16 requested but unsupported on this device -- falling back to float32.")

    context_encoder, target_encoder, predictor = build_models(cfg, device)
    dataloader = build_dataloader(cfg)

    # Target encoder is intentionally excluded from the optimizer (Section 6: EMA-only).
    params = list(context_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["optim"]["base_lr"], betas=tuple(cfg["optim"]["betas"]))

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = int(cfg["optim"]["warmup_epoch_fraction"] * total_steps)
    scaled_lr = scale_lr_for_batch_size(
        cfg["optim"]["base_lr"], cfg["train"]["batch_size"], cfg["optim"]["reference_batch_size"]
    )

    lr_iter = lr_schedule(scaled_lr, warmup_steps, total_steps, cfg["optim"]["final_lr_fraction"])
    wd_iter = weight_decay_schedule(cfg["optim"]["wd_start"], cfg["optim"]["wd_end"], total_steps)
    m_iter = ema_momentum_schedule(cfg["ema"]["m_start"], cfg["ema"]["m_end"], total_steps)

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        stats = train_one_epoch(
            context_encoder, target_encoder, predictor, dataloader, optimizer,
            lr_iter, wd_iter, m_iter, device, use_bf16, epoch, global_step,
        )
        global_step = stats["global_step"]
        print(f"[epoch {epoch}] avg_loss={stats['avg_loss']:.4f}")

        if (epoch + 1) % cfg["train"]["checkpoint_every"] == 0:
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"ckpt_epoch{epoch}.pt")
            save_checkpoint(ckpt_path, context_encoder, target_encoder, predictor, optimizer, epoch, global_step, cfg)
            print(f"[epoch {epoch}] checkpoint saved to {ckpt_path}")


def dump_latents(cfg: Dict[str, Any], checkpoint_path: str, image_path: str, out_path: str) -> None:
    """
    Extension hook #1 (Section 9.1): given a trained checkpoint + an input image, output
    per-patch predicted latents AND their (row, col) grid positions -- NOT a single pooled
    vector -- so a future vector/SVG decoder can place primitives per-region.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_encoder, target_encoder, predictor = build_models(cfg, device)
    load_checkpoint(checkpoint_path, context_encoder, target_encoder, predictor, map_location=str(device))
    context_encoder.eval()
    predictor.eval()

    d = cfg["data"]
    transform = transforms.Compose(
        [
            transforms.Resize(d["img_size"]),
            transforms.CenterCrop(d["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)  # (1, 3, H, W)

    grid_size = d["img_size"] // d["patch_size"]
    collator = build_mask_collator(cfg)
    context_set, target_sets = collator.sample_masks()
    context_mask = indices_to_tensor(context_set).to(device)
    target_masks = torch.stack([indices_to_tensor(t) for t in target_sets], dim=0).to(device)

    with torch.no_grad():
        h_ctx = context_encoder(img, patch_indices=context_mask)  # (1, N_ctx, D)
        pred = predictor(h_ctx, context_mask, target_masks)  # (T, N_tgt, D)  -- B == 1 here

    T, N_tgt = target_masks.shape
    rows = (target_masks // grid_size).reshape(-1)  # (T*N_tgt,)
    cols = (target_masks % grid_size).reshape(-1)  # (T*N_tgt,)
    positions = torch.stack([rows, cols], dim=1)  # (T*N_tgt, 2)
    latents = pred.reshape(T * N_tgt, -1)  # (T*N_tgt, D)

    torch.save({"latents": latents.cpu(), "positions": positions.cpu()}, out_path)
    print(
        f"[dump_latents] saved {latents.shape[0]} per-patch latents (dim={latents.shape[1]}) "
        f"and grid positions to {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="I-JEPA training / inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--distributed", action="store_true",
                         help="Optional, config-gated DDP flag. NOT required for 1-GPU/CPU runs.")
    parser.add_argument("--dump_latents", action="store_true", help="Run extension hook #1 (Section 9.1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for --dump_latents")
    parser.add_argument("--image", type=str, default=None, help="Input image path for --dump_latents")
    parser.add_argument("--out", type=str, default="latents.pt", help="Output path for --dump_latents")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.dump_latents:
        assert args.checkpoint and args.image, "--dump_latents requires --checkpoint and --image"
        dump_latents(cfg, args.checkpoint, args.image, args.out)
        return

    if args.distributed:
        # DistributedDataParallel is intentionally config-gated but NOT implemented in this
        # version -- single-GPU / CPU training is the fully supported default.
        raise NotImplementedError(
            "Distributed training is config-gated but not implemented in this version. "
            "Remove --distributed to run single-device training."
        )

    train(cfg)


if __name__ == "__main__":
    main()
