# openjepa

**A from-scratch PyTorch implementation of I-JEPA (Image-based Joint-Embedding Predictive Architecture)** — a self-supervised method for learning visual representations without labels, without pixel reconstruction, and without hand-crafted data augmentation.

---

## Why JEPA matters

Most self-supervised vision methods fall into two camps:

1. **Contrastive / augmentation-based methods** (SimCLR, BYOL, DINO, etc.) — these force the model to produce similar embeddings for two augmented views of the same image (crop, color-jitter, blur, etc.). The model's quality is heavily dependent on *how good your augmentations are*, which is a hand-engineered bottleneck.
2. **Generative / reconstruction-based methods** (MAE, pixel autoencoders, etc.) — these mask part of an image and train the model to reconstruct the missing *pixels*. This forces the model to spend capacity modeling low-level texture and noise, much of which is irrelevant to genuine visual understanding.

**JEPA takes a third path.** Instead of reconstructing pixels or relying on augmentation invariance, it:

- Masks out large regions of an image.
- Encodes the **visible** region with a context encoder.
- Encodes the **entire** image with a separate, slowly-updated target encoder (no gradients — updated only via exponential moving average).
- Trains a lightweight predictor to predict the **target encoder's latent representation** of the masked regions — not their pixels.

Because the prediction target is an abstract embedding rather than raw pixels, the model is naturally pushed toward learning **structural, semantic, texture-invariant representations** rather than memorizing low-level noise. This has made JEPA-style models a strong candidate backbone for downstream tasks where geometry and structure matter more than pixel-perfect detail — which is exactly why this repository exists: the long-term goal of `openjepa` is to serve as the encoder backbone for a vector graphics / SVG generation system, where discarding pixel texture in favor of structural understanding is a genuine advantage rather than a limitation.

This repo is a clean, dependency-minimal, fully offline-runnable implementation of that idea — and now includes a **complete, trainable pipeline from raw image to rendered SVG**: pretraining, latent extraction, a differentiable rasterizer, decoder training against real images, and final SVG export.

---

## Installation

```
pip install -r requirements.txt
```

Only `torch`, `torchvision`, `numpy`, `pyyaml`, and `tqdm` are required — no exotic dependencies, no internet access needed at any point. Everything added below (SVG decoder, rasterizer, decoder training) introduces **zero new dependencies** — the rasterizer is pure PyTorch and SVG files are built as plain XML strings by hand.

---

## Quick start: smoke test with synthetic data

Before training on anything real, verify the whole pipeline runs end-to-end using randomly generated images. This checks that masking, encoding, prediction, loss computation, and checkpointing all work — it does **not** check learning quality (there's no real structure in random noise to learn).

In `configs/default.yaml`, keep the defaults:

```
data:
  use_synthetic: true
```

Then run:

```
python train.py --config configs/default.yaml
```

You should see a progress bar, a printed loss per epoch, and checkpoint files appear under `checkpoints/`.

---

## Training on real images (encoder pretraining)

1. Gather your images into a single folder (subfolders are fine too — labels are ignored, since this is unsupervised):

```
openjepa/
└── my_images/
    ├── photo1.jpg
    ├── photo2.jpg
    └── ...
```

2. Edit `configs/default.yaml`:

```
data:
  root: ./my_images
  use_synthetic: false
```

3. Adjust batch size / epochs to fit your dataset size and hardware. For a small dataset (tens to hundreds of images), something like this is reasonable:

```
train:
  batch_size: 8
  epochs: 50
```

4. Run training:

```
python train.py --config configs/default.yaml
```

Watch the printed `avg_loss` — with real images, this should trend downward over epochs (unlike the synthetic smoke test, where it's expected to stay flat/noisy).

5. Checkpoints are saved to `checkpoints/ckpt_epochN.pt` after every epoch (configurable via `checkpoint_every`). Each checkpoint contains everything needed to resume or reuse the model: both encoders, the predictor, optimizer state, and the full training config. This is also the file the decoder training step below reads from, to obtain a frozen target encoder.

---

## Extracting latents from a trained model

Once you have a checkpoint, you can run the model on a single real image and export its per-patch latent representations (useful for inspection or downstream tasks):

```
python train.py --config configs/default.yaml --dump_latents \
    --checkpoint checkpoints/ckpt_epoch49.pt \
    --image path/to/some/image.jpg \
    --out latents.pt
```

This produces a file containing per-patch predicted latents and their `(row, col)` grid positions. Note: this only exports latents for the **masked target patches** sampled during JEPA's masking strategy — a subset of the grid, not the whole image. That's correct for verifying JEPA mechanics, but not what decoder training uses internally (see below).

---

## Training the SVG decoder against real images

Rather than requiring a paired (image → SVG) dataset — which doesn't exist anywhere — the decoder is trained the way an autoencoder is: a real image is encoded by the **frozen** target encoder into a full grid of per-patch latents, decoded into Bézier curve parameters, **rendered back into pixels by a differentiable rasterizer**, and compared against the original image with a pixel loss. Only the decoder's weights update; the JEPA encoder/predictor are untouched.

```
python train_decoder.py --config configs/default.yaml \
    --checkpoint checkpoints/ckpt_epoch137.pt \
    --data_root ./my_images \
    --epochs 100 \
    --out checkpoints/decoder.pt
```

Because this is self-supervised reconstruction (each image supervises itself), the `avg_loss` printed here should trend down more reliably and quickly than encoder pretraining — a few hundred images is enough to see it working, though more images/epochs will generalize better to new, unseen images.

This produces `checkpoints/decoder.pt` — a trained `VectorPathDecoder` state dict (distinct from a full JEPA checkpoint).

---

## Generating SVG output from latents

**SVG image generation is now fully trainable, not just structurally valid.** To decode a `latents.pt` file into an SVG using your trained decoder:

```
python decode_to_svg.py --latents latents.pt --out output.svg \
    --decoder_checkpoint checkpoints/decoder.pt
```

If `--decoder_checkpoint` is omitted, a freshly-initialized, **untrained** decoder is used instead — useful only for verifying the code path runs (valid XML, correct per-patch placement), not for meaningful output. Passing a trained decoder from the step above is what actually produces curves/colors shaped by real image content.

Open `output.svg` in any browser or image viewer — it's a complete, standalone SVG file either way.

---

## Project structure

- `configs/default.yaml` — all hyperparameters in one place: data, model size, masking, optimizer, EMA schedule.
- `data/dataset.py` — image loading from a local folder, plus a synthetic random-tensor dataset for offline testing.
- `masks/multiblock.py` — samples the context and target masks used each training batch.
- `masks/utils.py` — helper functions for converting mask blocks into patch indices.
- `models/pos_embed.py` — fixed sin-cos positional embeddings for patch locations.
- `models/vision_transformer.py` — the ViT encoder architecture (shared by both context and target encoders).
- `models/predictor.py` — the lightweight predictor that guesses masked-region latents.
- `engine/ema.py` — momentum schedule and update rule for the target encoder.
- `engine/loss.py` — the Smooth L1 loss comparing predicted vs. target latents.
- `engine/train_one_epoch.py` — the core training loop for a single epoch.
- `utils/schedulers.py` — learning rate and weight decay schedules.
- `utils/checkpoint.py` — saving/loading full training state.
- `extensions/decoder_stub.py` — the abstract interface that any latent-to-output decoder must implement.
- `extensions/svg_decoder.py` — concrete `VectorPathDecoder`: maps per-patch latents to Bézier curve + style parameters via a small MLP.
- `extensions/svg_writer.py` — dependency-free SVG string builder that assembles decoded curves into a complete SVG document.
- `extensions/encoder_loader.py` — loads a **frozen** target encoder from a JEPA checkpoint and runs it over a full, unmasked image grid (used only by decoder training).
- `extensions/rasterizer.py` — pure-PyTorch **differentiable rasterizer**: renders per-patch Bézier curves directly into a pixel tensor, enabling pixel-loss backpropagation into the decoder.
- `decode_to_svg.py` — command-line entry point: takes a `latents.pt` file (and optionally a trained decoder) and produces an `output.svg` file.
- `train_decoder.py` — command-line entry point: trains `VectorPathDecoder` against real images via frozen-encoder + differentiable-rasterizer reconstruction, saving `decoder.pt`.
- `train.py` — the command-line entry point tying encoder/predictor training and latent-dumping together.

---

## A note on scope

This repository now implements a **complete, working pipeline from raw image to rendered SVG**: JEPA pretraining → frozen-encoder latent extraction → a differentiable rasterizer → decoder training against real images via reconstruction loss → final SVG export. Every stage runs and produces genuine output.

What's still an open research question, not a code gap, is **output quality at scale**: how well this reconstruction-based training generalizes to unseen images, how the simple "one Bézier curve per patch" parametrization compares to richer vector representations (variable curve counts, layering, closed shapes/fills), and how results scale with more images/epochs. Those are tuning and modeling directions to explore now that the full loop is in place — not missing plumbing.