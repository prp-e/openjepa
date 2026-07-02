"""
decode_to_svg.py

Command-line entry point: takes a latents.pt file (produced by
`train.py --dump_latents`) and an optional trained VectorPathDecoder
checkpoint, and writes out a complete, standalone SVG document.

Expected latents.pt contents (dict):
    "latents":    (N, latent_dim) tensor
    "positions":  (N, 2) tensor of (row, col) integer grid coordinates
    "patch_size": int (optional -- override via --patch_size if missing)

If --decoder_checkpoint is omitted, a freshly-initialized, UNTRAINED
decoder is used -- proves the code path runs end-to-end (valid SVG,
correct per-patch placement, correct fill+stroke layering) but produces
shapes/colors with no learned relationship to real image content.
"""
from __future__ import annotations

import argparse

import torch

from extensions.svg_decoder import VectorPathDecoder
from extensions.svg_writer import build_svg_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode latents.pt into an SVG file.")
    parser.add_argument("--latents", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--decoder_checkpoint", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=None,
                         help="Override patch size in pixels, if not stored inside latents.pt")
    args = parser.parse_args()

    payload = torch.load(args.latents, map_location="cpu")
    latents = payload["latents"]
    positions = payload["positions"]
    patch_size = args.patch_size or payload.get("patch_size")
    if patch_size is None:
        raise ValueError(
            "patch_size not found inside latents.pt and no --patch_size override given. "
            "Pass --patch_size explicitly (must match the value used during JEPA training)."
        )

    grid_size = int(positions.max().item()) + 1
    latent_dim = latents.shape[-1]

    decoder = VectorPathDecoder(latent_dim=latent_dim, hidden_dim=args.hidden_dim)
    if args.decoder_checkpoint:
        decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location="cpu"))
        decoder.eval()
        print(f"[decode_to_svg] loaded trained decoder from {args.decoder_checkpoint}")
    else:
        print(
            "[decode_to_svg] WARNING: no --decoder_checkpoint given -- using a randomly "
            "initialized, UNTRAINED decoder. Output SVG will be structurally valid but "
            "carries no learned relationship to real image content."
        )

    with torch.no_grad():
        decoded = decoder(latents, positions=positions)

    svg_doc = build_svg_document(decoded, positions=positions, patch_size=patch_size, grid_size=grid_size)
    with open(args.out, "w") as f:
        f.write(svg_doc)

    print(f"[decode_to_svg] wrote {args.out} ({grid_size * patch_size}x{grid_size * patch_size}px)")


if __name__ == "__main__":
    main()