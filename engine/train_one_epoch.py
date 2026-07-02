"""
Training loop implementing the Section 4 batched forward/backward pass.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator

import torch
import torch.nn as nn
from tqdm import tqdm

from engine.ema import update_target_encoder
from engine.loss import IJEPALoss


def train_one_epoch(
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    lr_iter: Iterator[float],
    wd_iter: Iterator[float],
    m_iter: Iterator[float],
    device: torch.device,
    use_bf16: bool,
    epoch: int,
    global_step: int,
) -> Dict[str, Any]:
    loss_fn = IJEPALoss(beta=1.0)
    context_encoder.train()
    predictor.train()
    target_encoder.eval()  # never trained by backprop -- EMA-only (Section 6)

    running_loss = 0.0
    num_batches = 0
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    pbar = tqdm(dataloader, desc=f"epoch {epoch}")
    for images, context_mask, target_masks in pbar:
        images = images.to(device, non_blocking=True)            # (B, 3, H, W)
        context_mask = context_mask.to(device, non_blocking=True)  # (N_ctx,)
        target_masks = target_masks.to(device, non_blocking=True)  # (T, N_tgt)
        B = images.shape[0]
        T, N_tgt = target_masks.shape

        lr = next(lr_iter)
        wd = next(wd_iter)
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = wd

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(device.type == "cuda")):
            # Section 4, step 2: context encoder on visible patches only.
            h_ctx = context_encoder(images, patch_indices=context_mask)  # (B, N_ctx, D_enc)

            # Section 4, steps 3-4: target encoder on ALL patches, no grad;
            # LayerNorm-normalization happens inside the encoder's final `self.norm`.
            with torch.no_grad():
                h_full = target_encoder(images, patch_indices=None)  # (B, N, D_enc)

            # Section 4, step 5: gather targets per block -> (T, B, N_tgt, D) -> (T*B, N_tgt, D)
            targets = torch.stack([h_full[:, target_masks[t], :] for t in range(T)], dim=0)
            targets = targets.reshape(T * B, N_tgt, -1).detach()  # sg[...] -- explicit stop-gradient

            # Section 4, steps 6-8: repeat_interleave_batch trick, single predictor forward.
            pred = predictor(h_ctx, context_mask, target_masks)  # (B*T, N_tgt, D_enc)

            # Section 4, step 9 / Section 5: Smooth L1 loss.
            loss = loss_fn(pred, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Section 6: EMA update, no_grad, in-place.
        m = next(m_iter)
        update_target_encoder(target_encoder, context_encoder, m)

        running_loss += loss.item()
        num_batches += 1
        global_step += 1
        pbar.set_postfix(loss=running_loss / num_batches, lr=lr, wd=wd, m=m)

    return {"avg_loss": running_loss / max(1, num_batches), "global_step": global_step}
