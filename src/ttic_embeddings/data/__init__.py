"""Dataset wrappers for COCO Captions and Visual Genome attributes."""
from __future__ import annotations

from .coco import (
    CocoCachedFeaturePairs,
    CocoCaptionPairs,
    CocoEvalImages,
    coco_root_from_env,
)

__all__ = [
    "CocoCachedFeaturePairs",
    "CocoCaptionPairs",
    "CocoEvalImages",
    "coco_root_from_env",
]
