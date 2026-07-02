"""
Checkpoint save/load (Section 9.3). A single file stores everything downstream
(SVG-decoder) training will need.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(path: str, context_encoder: nn.Module, target_encoder: nn.Module,
                     predictor: nn.Module, optimizer: optim.Optimizer, epoch: int,
                     step: int, config: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "context_encoder": context_encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, context_encoder: nn.Module, target_encoder: nn.Module,
                     predictor: nn.Module, optimizer: Optional[optim.Optimizer] = None,
                     map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    context_encoder.load_state_dict(ckpt["context_encoder"])
    target_encoder.load_state_dict(ckpt["target_encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
