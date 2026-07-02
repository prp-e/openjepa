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

This repo is a clean, dependency-minimal, fully offline-runnable implementation of that idea — **and now includes a working first version of that SVG generation pipeline**, taking the project from "encoder only" to "encoder + latent-to-vector-graphics decoder."

---

## Installation

```
pip install -r requirements.txt
```

Only `torch`, `torchvision`, `numpy`, `pyyaml`, and `tqdm` are required — no exotic dependencies, no internet access needed at any point. The SVG decoder added below introduces **zero new dependencies** either — SVG files are built as plain XML strings by hand.

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

## Training on real images

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

5. Checkpoints are saved to `checkpoints/ckpt_epochN.pt` after every epoch (configurable via `checkpoint_every`). Each checkpoint contains everything needed to resume or reuse the model: both encoders, the predictor, optimizer state, and the full training config.

---

## Extracting latents from a trained model

Once you have a checkpoint, you can run the model on a single real image and export its per-patch latent representations (useful for downstream tasks, inspection, or feeding into the SVG decoder below):

```
python train.py --config configs/default.yaml --dump_latents \
    --checkpoint checkpoints/ckpt_epoch49.pt \
    --image path/to/some/image.jpg \
    --out latents.pt
```

This produces a file containing per-patch predicted latents and their `(row, col)` grid positions — not a single pooled vector — since the spatial decoder below needs to know *where* each latent came from.

---

## Generating SVG output from latents

**SVG image generation has now been achieved.** The abstract decoder interface originally left as a stub has a concrete, working implementation: a small MLP maps each patch's latent vector into the parameters of a cubic Bézier curve (control points, stroke color, stroke width, opacity), which are then assembled into a single valid SVG document.

To decode a `latents.pt` file into an SVG:

```
python decode_to_svg.py --latents latents.pt --out output.svg
```

Open `output.svg` in any browser or image viewer — it's a complete, standalone SVG file.

**Important honesty note:** by default this uses a freshly-initialized, *untrained* decoder (no `--decoder_checkpoint` given). That means the output SVG is **structurally correct** — valid XML, one curve correctly placed per patch, correct canvas sizing — but the curve shapes/colors currently carry **no learned relationship** to the actual image content, since there is no paired (image → SVG) training data or loss anywhere in this pipeline yet. This mirrors the same honest caveat used for the synthetic-data smoke test earlier in this README: it proves the *code path* works end-to-end, not that the *output* is visually meaningful yet. Training the decoder to produce meaningful vector art is a natural next step, requiring either paired ground-truth SVGs or a differentiable rasterization loss.

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
- `extensions/svg_decoder.py` — **concrete implementation**: maps per-patch latents to Bézier curve + style parameters via a small MLP.
- `extensions/svg_writer.py` — dependency-free SVG string builder that assembles decoded curves into a complete SVG document.
- `decode_to_svg.py` — command-line entry point: takes a `latents.pt` file and produces an `output.svg` file.
- `train.py` — the command-line entry point tying the encoder/predictor training and latent-dumping together.

---

## A note on scope

This repository implements **pretraining plus a first working decoder path**. It produces a trained encoder + predictor, exports their latents, and now converts those latents all the way into a rendered SVG file — closing the loop from raw image to vector graphics output for the first time in this project. What remains unverified is **decoder quality**: the current `VectorPathDecoder` is untrained by default, so while the encoder→latent→SVG pipeline is fully functional end-to-end, actually training that decoder against real ground-truth vector graphics (or a rasterization-based loss) is the next milestone, not yet part of this repository.