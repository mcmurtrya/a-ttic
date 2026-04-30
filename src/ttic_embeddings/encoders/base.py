"""Common interface for visual encoders.

All four encoders subclass `VisualEncoder` and implement three methods:
  - `_load_backbone()`           — HuggingFace model load
  - `_load_processor()`          — HF preprocessor for that model
  - `_get_patches(outputs)`      — slice/unshuffle patch tokens from
                                   the backbone forward output

The base class handles freezing, shape verification, and the no-grad
forward pass. Patch-only output is mandated by the methods document:
the CLS token concentrates information differently across supervision
regimes and would systematically disadvantage the self-supervised
encoders if used.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
import torch.nn as nn
from PIL import Image


class VisualEncoder(nn.Module, ABC):
    name: ClassVar[str]
    hidden_dim: ClassVar[int]
    expected_patch_count: ClassVar[int]
    image_size: ClassVar[int]

    def __init__(self) -> None:
        super().__init__()
        self.backbone = self._load_backbone()
        self.processor = self._load_processor()
        self._freeze()

    @abstractmethod
    def _load_backbone(self) -> nn.Module:
        ...

    @abstractmethod
    def _load_processor(self) -> Any:
        ...

    @abstractmethod
    def _get_patches(self, outputs: Any) -> torch.Tensor:
        """Return [B, N_patch, D] from the backbone forward output.

        Each subclass owns its own layout: CLS removal, register-token
        removal, MAE patch unshuffling, etc.
        """

    def _freeze(self) -> None:
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        patches = self._get_patches(outputs)
        self._verify_shape(patches)
        return patches

    def _verify_shape(self, patches: torch.Tensor) -> None:
        _, n, d = patches.shape
        if n != self.expected_patch_count:
            raise RuntimeError(
                f"{self.name}: expected {self.expected_patch_count} patches, "
                f"got {n}. Check `_get_patches` against the HF model layout."
            )
        if d != self.hidden_dim:
            raise RuntimeError(
                f"{self.name}: expected hidden_dim {self.hidden_dim}, got {d}."
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"patches={self.expected_patch_count}, dim={self.hidden_dim}, "
            f"image_size={self.image_size})"
        )
