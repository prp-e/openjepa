"""
LR and weight-decay schedules (Section 7).
"""
from __future__ import annotations

import math
from typing import Iterator


def lr_schedule(base_lr: float, warmup_steps: int, total_steps: int,
                 final_lr_fraction: float = 0.0) -> Iterator[float]:
    """Linear warmup for `warmup_steps`, then cosine decay to
    `final_lr_fraction * base_lr` over the remaining steps."""
    final_lr = base_lr * final_lr_fraction
    for step in range(total_steps):
        if step < warmup_steps:
            yield base_lr * (step + 1) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            yield final_lr + (base_lr - final_lr) * cos
    while True:
        yield final_lr


def weight_decay_schedule(wd_start: float, wd_end: float, total_steps: int) -> Iterator[float]:
    """Cosine schedule increasing weight decay wd_start -> wd_end over training.
    Reasonable/configurable default, NOT a hard-verified paper constant (per Section 7)."""
    for step in range(total_steps):
        progress = step / max(1, total_steps - 1)
        cos = 0.5 * (1 - math.cos(math.pi * progress))  # 0 -> 1
        yield wd_start + (wd_end - wd_start) * cos
    while True:
        yield wd_end


def scale_lr_for_batch_size(base_lr: float, batch_size: int, reference_batch_size: int = 256) -> float:
    """lr = base_lr * batch_size / reference_batch_size -- common, configurable default (Section 7)."""
    return base_lr * batch_size / reference_batch_size
