"""DINOv2 ViT-L/14 — Meta, self-supervised (self-distillation, no labels)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model

from .base import VisualEncoder


class DINOv2Encoder(VisualEncoder):
    name = "dinov2"
    hidden_dim = 1024
    expected_patch_count = 256
    image_size = 224
    default_checkpoint = "facebook/dinov2-large"

    def __init__(self, checkpoint: str | None = None) -> None:
        self.checkpoint = checkpoint or self.default_checkpoint
        super().__init__()

    def _load_backbone(self) -> nn.Module:
        return Dinov2Model.from_pretrained(self.checkpoint)

    def _load_processor(self) -> Any:
        return AutoImageProcessor.from_pretrained(self.checkpoint)

    def _get_patches(self, outputs: Any) -> torch.Tensor:
        # DINOv2 (non-`-reg` variant): [CLS, patch_1, ..., patch_256]
        # If switching to `dinov2-with-registers-large`, drop indices 0..4
        # (CLS + 4 register tokens).
        return outputs.last_hidden_state[:, 1:, :]
