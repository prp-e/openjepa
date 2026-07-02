"""
Predictor (Section 2, "Predictor spec" + Section 4 batching trick).
Narrower/shallower than the encoder; predicts masked target latents from context tokens
plus positional mask tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.pos_embed import get_2d_sincos_pos_embed
from models.vision_transformer import Block


class Predictor(nn.Module):
    def __init__(self, encoder_embed_dim: int, grid_size: int, predictor_embed_dim: int = 384,
                 depth: int = 6, num_heads: int = 12, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.embed_proj_in = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # Single learnable mask token, shape (1,1,Dp), broadcast to every masked position
        # (Section 2, "Predictor spec").
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        pos_embed = get_2d_sincos_pos_embed(predictor_embed_dim, grid_size, grid_size)  # (N, Dp)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

        self.blocks = nn.ModuleList(
            [Block(predictor_embed_dim, num_heads, mlp_ratio, qkv_bias) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.embed_proj_out = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ctx_tokens: torch.Tensor, ctx_indices: torch.Tensor,
                target_indices: torch.Tensor) -> torch.Tensor:
        """
        ctx_tokens:     (B, N_ctx, encoder_embed_dim)  -- context encoder output
        ctx_indices:    (N_ctx,)                       -- flat grid positions of context tokens
                                                            (shared across the batch)
        target_indices: (T, N_tgt)                     -- T target blocks, each with N_tgt flat
                                                            grid positions. N_tgt is IDENTICAL
                                                            across all T blocks by construction
                                                            (see masks/multiblock.py).
        returns: (B * T, N_tgt, encoder_embed_dim), row order = t * B + b
                 (matches the target-gathering order used in engine/train_one_epoch.py)
        """
        B, N_ctx, _ = ctx_tokens.shape
        T, N_tgt = target_indices.shape

        x_ctx = self.embed_proj_in(ctx_tokens)  # (B, N_ctx, Dp)
        # DESIGN DECISION (documented at top of this response): re-add positional embeddings
        # to context tokens here too, beyond the literal spec text (which only requires pos
        # embed on mask tokens). Matches the reference I-JEPA predictor and removes any
        # ambiguity about *where* each context token came from once inside the predictor.
        x_ctx = x_ctx + self.pos_embed[ctx_indices].unsqueeze(0)  # (B, N_ctx, Dp)

        # --- repeat_interleave_batch trick (Section 4, step 6): tile context along batch dim ---
        x_ctx_expanded = x_ctx.repeat(T, 1, 1)  # (B*T, N_ctx, Dp); row i = block (i//B), sample (i%B)

        pos = self.pos_embed[target_indices.reshape(-1)].reshape(T, N_tgt, -1)  # (T, N_tgt, Dp)
        mask_tok = self.mask_token.expand(T, N_tgt, -1) + pos  # (T, N_tgt, Dp)
        mask_tok = mask_tok.unsqueeze(1).expand(T, B, N_tgt, mask_tok.shape[-1]).reshape(T * B, N_tgt, -1)
        # mask_tok row i = block (i//B), sample (i%B) -- same ordering as x_ctx_expanded above.

        x = torch.cat([x_ctx_expanded, mask_tok], dim=1)  # (B*T, N_ctx + N_tgt, Dp)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        pred = x[:, N_ctx:, :]  # (B*T, N_tgt, Dp) -- keep only mask-token output positions
        pred = self.embed_proj_out(pred)  # (B*T, N_tgt, encoder_embed_dim)
        return pred
