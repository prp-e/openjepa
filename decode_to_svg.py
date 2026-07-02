"""
CLI: decode a `latents.pt` file (produced by `train.py --dump_latents`) into an SVG file.

This script does NOT modify, import, or depend on any change to the core I-JEPA training
architecture -- it only consumes the (latents, positions) tensors that pipeline already
exports, via the extension-point contract in extensions/decoder_stub.py.

USAGE:
    python decode_to_svg.py --latents latents.pt --out output.svg
    python decode_to_svg.py --latents latents.pt --out output.svg --decoder_checkpoint decoder.pt
"""
from __future__ import annotations

import argparse

import torch

from extensions.svg_decoder import VectorPathDecoder
from extensions.svg_writer import decoded_to_svg, save_svg


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode I-JEPA latents into an SVG file.")
    parser.add_argument("--latents", type=str, required=True, help="Path to latents.pt from --dump_latents")
    parser.add_argument("--out", type=str, default="output.svg", help="Output SVG path")
    parser.add_argument("--patch_size", type=int, default=16, help="Must match the patch_size used during training")
    parser.add_argument(
        "--decoder_checkpoint", type=str, default=None,
        help="Optional path to a trained VectorPathDecoder state_dict. If omitted, a "
             "freshly-initialized (UNTRAINED) decoder is used.",
    )
    parser.add_argument("--hidden_dim", type=int, default=128, help="Decoder MLP hidden size")
    args = parser.parse_args()

    data = torch.load(args.latents, map_location="cpu")
    latents = data["latents"]        # (N, D)
    positions = data["positions"]    # (N, 2)

    decoder = VectorPathDecoder(latent_dim=latents.shape[-1], hidden_dim=args.hidden_dim)
    if args.decoder_checkpoint is not None:
        decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location="cpu"))
        print(f"[decode_to_svg] loaded trained decoder weights from {args.decoder_checkpoint}")
    else:
        print(
            "[decode_to_svg] WARNING: no --decoder_checkpoint given -- using a randomly "
            "initialized, UNTRAINED decoder. Output SVG will be structurally valid but "
            "carries no learned relationship to the input image's content."
        )
    decoder.eval()

    grid_size = int(positions.max().item()) + 1  # inferred from observed (row, col) range
    with torch.no_grad():
        decoded = decoder(latents, positions)

    svg_string = decoded_to_svg(decoded, patch_size=args.patch_size, grid_size=grid_size)
    save_svg(svg_string, args.out)
    print(f"[decode_to_svg] saved {decoded['control_points'].shape[0]} path segments to {args.out}")


if __name__ == "__main__":
    main()