"""Phase 0 smoke test — verify all encoders and decoder load cleanly.

Run:
    python scripts/00_smoke_test.py

Verifies:
    1. CLIP, SigLIP, DINOv2, MAE checkpoints all load from HuggingFace
    2. Each produces patch tokens with the expected shape
       (256 / 256 / 256 / 196 patches at hidden_dim 1024)
    3. All encoder parameters are frozen (no trainable params)
    4. GPT-2 medium loads and accepts a hand-constructed soft prefix
       through `inputs_embeds=`

Runs against random pixels — no data download required. First
invocation pulls HF checkpoints (~10 GB total) and is slow; subsequent
runs hit the local HF cache and finish in under a minute on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make `src/` importable when run as a script
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttic_embeddings.encoders import ENCODER_REGISTRY, build_encoder  # noqa: E402
from ttic_embeddings.utils import get_logger, set_seed  # noqa: E402

log = get_logger("smoke")


def random_image(size: int) -> Image.Image:
    arr = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def check_encoder(name: str) -> None:
    log.info(f"--- {name} ---")
    encoder = build_encoder(name)
    log.info(f"loaded: {encoder!r}")

    img = random_image(encoder.image_size)
    pixel_values = encoder.preprocess([img])
    log.info(f"preprocessed pixel_values: {tuple(pixel_values.shape)}")

    patches = encoder(pixel_values)
    log.info(f"patch tokens:              {tuple(patches.shape)}")

    expected = (1, encoder.expected_patch_count, encoder.hidden_dim)
    assert patches.shape == expected, (
        f"{name}: expected {expected}, got {tuple(patches.shape)}"
    )

    n_total = sum(p.numel() for p in encoder.parameters())
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert n_trainable == 0, f"{name}: {n_trainable} trainable params (expected 0)"
    log.info(f"{name}: OK ({n_total:,} params, all frozen)")


def check_decoder() -> None:
    log.info("--- gpt2-medium ---")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.eval()

    assert model.config.n_embd == 1024, (
        f"GPT-2 medium hidden_dim mismatch: {model.config.n_embd}"
    )
    log.info(f"hidden_dim: {model.config.n_embd}")

    # Hand-construct a 10-token soft prefix at the LM's hidden dim,
    # concatenate with text token embeddings, and verify the model
    # accepts the combined sequence via `inputs_embeds=`.
    soft_prefix = torch.randn(1, 10, model.config.n_embd)
    text_ids = tokenizer("a photo of", return_tensors="pt").input_ids
    text_embeds = model.transformer.wte(text_ids)
    full_embeds = torch.cat([soft_prefix, text_embeds], dim=1)

    with torch.no_grad():
        outputs = model(inputs_embeds=full_embeds)
    log.info(f"logits shape: {tuple(outputs.logits.shape)}")
    log.info("gpt2-medium: OK (accepts soft prefix via inputs_embeds)")


def main() -> int:
    set_seed(0)
    log.info(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    log.info(f"encoders to verify: {sorted(ENCODER_REGISTRY)}")

    failed: list[str] = []
    for name in ENCODER_REGISTRY:
        try:
            check_encoder(name)
        except Exception as e:
            log.error(f"{name}: FAILED — {type(e).__name__}: {e}")
            failed.append(name)

    try:
        check_decoder()
    except Exception as e:
        log.error(f"decoder: FAILED — {type(e).__name__}: {e}")
        failed.append("gpt2-medium")

    if failed:
        log.error(f"\nFailures: {failed}")
        return 1
    log.info("\nAll Phase 0 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
