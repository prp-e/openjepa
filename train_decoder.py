"""
Trains extensions/svg_decoder.VectorPathDecoder against real images -- WITHOUT
touching configs/, data/, masks/, models/, engine/, utils/, or train.py.

Pipeline per image:
  real image -> FROZEN target_encoder -> full-grid per-patch latents
             -> VectorPathDecoder (TRAINABLE) -> per-patch fill + stroke params
             -> differentiable rasterizer -> reconstructed raster image
             -> pixel loss vs. original image -> backprop into decoder ONLY

There is no paired (image -> SVG) dataset anywhere in this project, so this
trains the decoder the way an autoencoder is trained: reconstruction loss
against the original image, with the JEPA encoder/predictor completely
frozen and untouched.

USAGE:
    python train_decoder.py --config configs/default.yaml \
        --checkpoint checkpoints/ckpt_epoch99.pt \
        --data_root ./my_images \
        --epochs 100 --out checkpoints/decoder.pt
"""
from __future__ import annotations

import argparse
import inspect
import os

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.dataset import LocalImageFolderDataset
from extensions.encoder_loader import load_frozen_target_encoder, encode_full_grid
from extensions.rasterizer import rasterize_patches, full_grid_positions
from extensions.svg_decoder import VectorPathDecoder


# ----------------------------------------------------------------------------
# Dataset plumbing -- introspects the constructor instead of hardcoding
# keyword arguments, and normalizes whatever __getitem__ returns into a
# plain (C, H, W) float tensor of the right size.
# ----------------------------------------------------------------------------

class _NormalizedImageDataset(Dataset):
    """Wraps LocalImageFolderDataset so every item is a resized (C, H, W) float tensor."""

    def __init__(self, base_dataset, img_size: int):
        self.base = base_dataset
        self.to_tensor_resized = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.base[idx]

        if isinstance(item, (tuple, list)):
            item = item[0]

        if isinstance(item, Image.Image):
            # Flatten any alpha channel onto a WHITE background -- must match
            # the rasterizer's white-background assumption, or transparent
            # icon backgrounds will silently mismatch the reconstruction.
            if item.mode in ("RGBA", "LA") or (item.mode == "P" and "transparency" in item.info):
                item = item.convert("RGBA")
                background = Image.new("RGB", item.size, (255, 255, 255))
                background.paste(item, mask=item.split()[-1])
                item = background
            else:
                item = item.convert("RGB")
            return self.to_tensor_resized(item)

        if torch.is_tensor(item):
            if item.dim() == 3 and item.shape[-1] in (1, 3) and item.shape[0] not in (1, 3):
                item = item.permute(2, 0, 1)
            return item.float()

        raise TypeError(
            f"Unrecognized item type returned by LocalImageFolderDataset: {type(item)}"
        )


def build_dataset(data_root: str, img_size: int) -> Dataset:
    sig = inspect.signature(LocalImageFolderDataset.__init__)
    kwargs = {}

    if "root" in sig.parameters:
        kwargs["root"] = data_root
    else:
        raise TypeError(
            "LocalImageFolderDataset.__init__ has no 'root' parameter -- "
            f"actual signature: {sig}"
        )

    if "img_size" in sig.parameters:
        kwargs["img_size"] = img_size
    if "transform" in sig.parameters:
        kwargs["transform"] = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    base_dataset = LocalImageFolderDataset(**kwargs)
    return _NormalizedImageDataset(base_dataset, img_size)


def build_dataloader(data_root: str, img_size: int, batch_size: int) -> DataLoader:
    dataset = build_dataset(data_root, img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_one_image(image, encoder, decoder, positions, grid_size, patch_size, sharpness):
    """image: (1, 3, H, W) tensor, already on the correct device."""
    with torch.no_grad():
        latents = encode_full_grid(encoder, image)  # frozen encoder, no grad tracked

    decoded = decoder(latents, positions=positions)

    reconstructed = rasterize_patches(
        control_points=decoded["control_points"],
        stroke_color=decoded["stroke_color"],
        stroke_width=decoded["stroke_width"],
        stroke_opacity=decoded["stroke_opacity"],
        fill_center=decoded["fill_center"],
        fill_radius=decoded["fill_radius"],
        fill_rotation=decoded["fill_rotation"],
        fill_color=decoded["fill_color"],
        fill_opacity=decoded["fill_opacity"],
        grid_size=grid_size,
        patch_size=patch_size,
        sharpness=sharpness,
    )  # (H, W, 3)

    target = image.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    return F.l1_loss(reconstructed, target)


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

    print(f"[train_decoder] loading frozen target encoder from {args.checkpoint}")
    encoder, ckpt_cfg = load_frozen_target_encoder(args.checkpoint, device=args.device)
    embed_dim = ckpt_cfg.get("model", {}).get("embed_dim", 384)

    decoder = VectorPathDecoder(latent_dim=embed_dim, hidden_dim=args.hidden_dim).to(args.device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    positions = full_grid_positions(grid_size, device=args.device)

    loader = build_dataloader(args.data_root, img_size, args.batch_size)
    print(f"[train_decoder] {len(loader.dataset)} images found, grid_size={grid_size}, "
          f"patch_size={patch_size}, embed_dim={embed_dim}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        num_images = 0

        for images in loader:
            images = images.to(args.device)

            for i in range(images.shape[0]):
                image = images[i : i + 1]

                loss = train_one_image(
                    image, encoder, decoder, positions, grid_size, patch_size, args.sharpness
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_images += 1
                step += 1
                if step % args.log_every == 0:
                    print(f"[train_decoder] epoch {epoch} step {step} loss {loss.item():.4f}")

        avg_loss = running_loss / max(1, num_images)
        print(f"[train_decoder] epoch {epoch} avg_loss {avg_loss:.4f}")
        torch.save(decoder.state_dict(), args.out)

    print(f"[train_decoder] done. Decoder weights saved to {args.out}")


if __name__ == "__main__":
    main()