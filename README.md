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

Because the prediction target is an abstract embedding rather than raw pixels, the model is naturally pushed toward learning **structural, semantic, texture-invariant representations** rather than memorizing low-level noise. This has made JEPA-style models a strong candidate backbone for downstream tasks where geometry and structure matter more than pixel-perfect detail — which is exactly why this repository exists: the long-term goal of `openjepa` is to serve as the encoder backbone for a future **vector graphics / SVG generation system**, where discarding pixel texture in favor of structural understanding is a genuine advantage rather than a limitation.

This repo is a clean, dependency-minimal, fully offline-runnable implementation of that idea.

---

## Installation

```
pip install -r requirements.txt
```

Only `torch`, `torchvision`, `numpy`, `pyyaml`, and `tqdm` are required — no exotic dependencies, no internet access needed at any point.

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

Once you have a checkpoint, you can run the model on a single real image and export its per-patch latent representations (useful for downstream tasks, inspection, or feeding into a future decoder):

```
python train.py --config configs/default.yaml --dump_latents \
    --checkpoint checkpoints/ckpt_epoch49.pt \
    --image path/to/some/image.jpg \
    --out latents.pt
```

This produces a file containing per-patch predicted latents and their `(row, col)` grid positions — not a single pooled vector — since a future spatial decoder needs to know *where* each latent came from.

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
- `extensions/decoder_stub.py` — an abstract interface for a future decoder that would turn latents into a visible output (e.g., SVG paths) — not implemented yet, just the contract.
- `train.py` — the command-line entry point tying everything together.

---

## A note on scope

This repository implements **pretraining only**. It produces a trained encoder + predictor and a way to export their latents — it does not yet include any downstream decoder (SVG or otherwise). The hypothesis that JEPA-style latents are well-suited to vector graphics generation is stated as motivation above, but remains unverified until a decoder is actually built and tested against these latents.