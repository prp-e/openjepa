"""
EMA momentum scheduler + in-place target-encoder update (Section 6).
"""
from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn


def ema_momentum_schedule(m_start: float, m_end: float, total_steps: int) -> Iterator[float]:
    """Linear schedule for EMA momentum m, increasing m_start -> m_end over total_steps
    optimizer steps. Holds at m_end if queried beyond total_steps."""
    assert total_steps > 0
    for step in range(total_steps):
        frac = step / max(1, total_steps - 1)
        yield m_start + (m_end - m_start) * frac
    while True:
        yield m_end


@torch.no_grad()
def update_target_encoder(target_encoder: nn.Module, context_encoder: nn.Module, m: float) -> None:
    """
    theta_bar <- m * theta_bar + (1 - m) * theta   (Section 6).
    In-place, no autograd. `target_encoder` must already have requires_grad=False on all
    its parameters (enforced once at model construction in train.py).
    """
    for p_t, p_c in zip(target_encoder.parameters(), context_encoder.parameters()):
        p_t.data.mul_(m).add_(p_c.data, alpha=(1.0 - m))
    # Defensive: also EMA any floating-point buffers (this ViT has none besides the fixed,
    # non-trainable pos_embed, which is identical by construction, so this is a no-op here).
    for b_t, b_c in zip(target_encoder.buffers(), context_encoder.buffers()):
        if b_t.dtype.is_floating_point:
            b_t.data.mul_(m).add_(b_c.data, alpha=(1.0 - m))
