"""SigLIP ViT-L/16 at 256 res — Google, language-supervised (sigmoid contrastive).

256 resolution chosen to match CLIP's 256-patch input shape (see
encoder_selection.md, "Implementation notes").

SigLIP's vision tower does NOT prepend a CLS token: pooled output is
computed via multi-head attention pooling over the patch tokens. So
`last_hidden_state` is already patch-only and needs no slicing.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import SiglipImageProcessor, SiglipVisionModel

from .base import VisualEncoder


class SigLIPEncoder(VisualEncoder):
    name = "siglip"
    hidden_dim = 1024
    expected_patch_count = 256
    image_size = 256
    default_checkpoint = "google/siglip-large-patch16-256"

    def __init__(self, checkpoint: str | None = None) -> None:
        self.checkpoint = checkpoint or self.default_checkpoint
        super().__init__()

    def _load_backbone(self) -> nn.Module:
        return SiglipVisionModel.from_pretrained(self.checkpoint)

    def _load_processor(self) -> Any:
        return SiglipImageProcessor.from_pretrained(self.checkpoint)

    def _get_patches(self, outputs: Any) -> torch.Tensor:
        # SigLIP: no CLS token, last_hidden_state is patches only.
        return outputs.last_hidden_state
