"""
Smooth L1 (Huber) loss wrapper (Section 5). Confirmed-standard choice per spec: NOT plain MSE.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IJEPALoss(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.criterion = nn.SmoothL1Loss(beta=beta, reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B*T, N_tgt, D). `target` MUST already be .detach()'d by the caller
        # (sg[.] in the Section 5 formula) -- enforced in engine/train_one_epoch.py.
        assert pred.shape == target.shape, f"shape mismatch: pred {pred.shape} vs target {target.shape}"
        return self.criterion(pred, target)
