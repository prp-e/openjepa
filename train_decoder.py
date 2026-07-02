"""
Trains extensions/svg_decoder.VectorPathDecoder against real images -- WITHOUT
touching configs/, data/, masks/, models/, engine/, utils/, or train.py.

Pipeline per image:
  real image -> FROZEN target_encoder -> full-grid per-patch latents
             -> VectorPathDecoder (TRAINABLE) -> per-patch Bezier curve params
             -> differentiable rasterizer -> reconstructed raster image
             -> pixel loss vs. original image -> backprop into decoder ONLY

USAGE:
    python train_decoder.py --config configs/default.yaml \
        --checkpoint checkpoints/ckpt_epoch137.pt \
        --data_root ./my_images \
        --epochs 100 --out checkpoints/decoder.pt
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import ImageFolderDataset  # NOTE: adjust class name/args if yours differs
from extensions.encoder_loader import load_frozen_target_encoder, encode_full_grid
from extensions.rasterizer import rasterize_patches, full_grid_positions
from extensions.svg_decoder import VectorPathDecoder  # untouched, imported as-is


def build_dataloader(data_root: str, img_size: int, batch_size: int) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolderDataset(root=data_root, transform=tfm)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VectorPathDecoder against real images.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="JEPA checkpoint (for frozen target_encoder)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--sharpness", type=float, default=3.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--out", type=str, default="checkpoints/decoder.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    img_size = cfg["data"]["img_size"]
    patch_size = cfg["data"]["patch_size"]
    grid_size = img_size // patch_size

    encoder, ckpt_cfg = load_frozen_target_encoder(args.checkpoint, device=args.device)
    embed_dim = ckpt_cfg.get("model", {}).get("embed_dim", 384)

    decoder = VectorPathDecoder(latent_dim=embed_dim, hidden_dim=args.hidden_dim).to(args.device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    positions = full_grid_positions(grid_size, device=args.device)

    loader = build_dataloader(args.data_root, img_size, args.batch_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        num_images = 0

        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch  # tolerate (img, label)
            images = images.to(args.device)

            for i in range(images.shape[0]):
                image = images[i : i + 1]  # (1, 3, H, W)

                latents = encode_full_grid(encoder, image)  # frozen, no grad
                decoded = decoder(latents, positions=positions)

                reconstructed = rasterize_patches(
                    control_points=decoded["control_points"],
                    color=decoded["color"],
                    stroke_width=decoded["stroke_width"],
                    opacity=decoded["opacity"],
                    grid_size=grid_size,
                    patch_size=patch_size,
                    sharpness=args.sharpness,
                )  # (H, W, 3)

                target = image.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
                loss = F.l1_loss(reconstructed, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_images += 1
                step += 1
                if step % args.log_every == 0:
                    print(f"[train_decoder] epoch {epoch} step {step} loss {loss.item():.4f}")

        print(f"[train_decoder] epoch {epoch} avg_loss {running_loss / max(1, num_images):.4f}")
        torch.save(decoder.state_dict(), args.out)

    print(f"[train_decoder] done. Decoder weights saved to {args.out}")


if __name__ == "__main__":
    main()