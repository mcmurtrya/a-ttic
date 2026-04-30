"""CLIP ViT-L/14 — OpenAI, language-supervised (softmax contrastive)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModel

from .base import VisualEncoder


class CLIPEncoder(VisualEncoder):
    name = "clip"
    hidden_dim = 1024
    expected_patch_count = 256
    image_size = 224
    default_checkpoint = "openai/clip-vit-large-patch14"

    def __init__(self, checkpoint: str | None = None) -> None:
        self.checkpoint = checkpoint or self.default_checkpoint
        super().__init__()

    def _load_backbone(self) -> nn.Module:
        return CLIPVisionModel.from_pretrained(self.checkpoint)

    def _load_processor(self) -> Any:
        return CLIPImageProcessor.from_pretrained(self.checkpoint)

    def _get_patches(self, outputs: Any) -> torch.Tensor:
        # CLIP layout: [CLS, patch_1, ..., patch_256] -> drop index 0
        return outputs.last_hidden_state[:, 1:, :]
