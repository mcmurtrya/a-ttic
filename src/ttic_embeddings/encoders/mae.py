"""MAE ViT-L/16 — Meta, self-supervised (masked image reconstruction).

Two MAE-specific quirks worth knowing:

1. ViTMAEModel applies random masking by default. We override
   `mask_ratio = 0.0` so all 196 patches are encoded.

2. Even with mask_ratio = 0.0, ViTMAEModel still SHUFFLES the patch
   sequence (the random_masking step does an argsort over uniform
   noise; with len_keep == N this returns a full random permutation
   of patches rather than the identity). The model conveniently
   returns `ids_restore` so we can unshuffle. We do that in
   `_get_patches` so downstream code sees patches in canonical
   spatial order, just like the other three encoders.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTMAEConfig, ViTMAEModel

from .base import VisualEncoder


class MAEEncoder(VisualEncoder):
    name = "mae"
    hidden_dim = 1024
    expected_patch_count = 196
    image_size = 224
    default_checkpoint = "facebook/vit-mae-large"

    def __init__(self, checkpoint: str | None = None) -> None:
        self.checkpoint = checkpoint or self.default_checkpoint
        super().__init__()

    def _load_backbone(self) -> nn.Module:
        config = ViTMAEConfig.from_pretrained(self.checkpoint)
        config.mask_ratio = 0.0  # encode every patch
        return ViTMAEModel.from_pretrained(self.checkpoint, config=config)

    def _load_processor(self) -> Any:
        return ViTImageProcessor.from_pretrained(self.checkpoint)

    def _get_patches(self, outputs: Any) -> torch.Tensor:
        # last_hidden_state: [B, 1 + N_patch, D] — CLS at index 0,
        # patches in shuffled order at indices 1..N.
        # ids_restore: [B, N_patch] — gather indices that map shuffled
        # patches back to canonical spatial order.
        hidden = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        shuffled = hidden[:, 1:, :]
        d = shuffled.shape[-1]
        return torch.gather(
            shuffled,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, d),
        )
