"""Four visual encoders behind a common interface."""
from __future__ import annotations

from .base import VisualEncoder
from .clip import CLIPEncoder
from .dinov2 import DINOv2Encoder
from .mae import MAEEncoder
from .siglip import SigLIPEncoder

ENCODER_REGISTRY: dict[str, type[VisualEncoder]] = {
    "clip": CLIPEncoder,
    "siglip": SigLIPEncoder,
    "dinov2": DINOv2Encoder,
    "mae": MAEEncoder,
}


def build_encoder(name: str, **kwargs) -> VisualEncoder:
    name = name.lower()
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder {name!r}. Choose from: {sorted(ENCODER_REGISTRY)}"
        )
    return ENCODER_REGISTRY[name](**kwargs)


__all__ = [
    "ENCODER_REGISTRY",
    "CLIPEncoder",
    "DINOv2Encoder",
    "MAEEncoder",
    "SigLIPEncoder",
    "VisualEncoder",
    "build_encoder",
]
