"""Caption generation from a trained adaptor.

Given an image (or a batch of images), this module:
  1. Encodes through the frozen encoder.
  2. Projects to a soft prefix via the adaptor.
  3. Hands the prefix to GPT-2's `generate()` via inputs_embeds.
  4. Decodes the new token ids back to text.

Two decoding strategies, both delegated to HuggingFace's tested
`model.generate()`:

  beam search (default beam_size=5)
  nucleus sampling (default top_p=0.9)

Both are used in Phase 4 of the roadmap; the encoder swap may interact
with the decoding strategy (e.g., self-supervised encoders might produce
more interesting nucleus samples), so we run and report both.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_soft_prefix(
    encoder: nn.Module,
    adaptor: nn.Module,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Run encoder + adaptor end-to-end, return (B, k, D_dec) soft prefix."""
    patches = encoder(pixel_values)            # (B, N, D_enc)
    prefix = adaptor(patches)                  # (B, k, D_dec)
    return prefix


@torch.no_grad()
def generate_captions(
    encoder: nn.Module,
    adaptor: nn.Module,
    decoder: nn.Module,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    strategy: str = "beam",
    beam_size: int = 5,
    top_p: float = 0.9,
    max_new_tokens: int = 30,
) -> list[str]:
    """Generate one caption per image in the batch.

    Args:
        encoder: frozen visual encoder (VisualEncoder subclass).
        adaptor: trained PrefixAdaptor.
        decoder: frozen GPT-2 LM (GPT2LMHeadModel).
        tokenizer: GPT-2 tokenizer.
        pixel_values: (B, 3, H, W) — already preprocessed for the encoder.
        strategy: "beam" or "nucleus".
        beam_size: only used when strategy="beam".
        top_p: only used when strategy="nucleus".
        max_new_tokens: caption length budget (in tokens).

    Returns:
        List of B caption strings, EOS and special tokens removed.
    """
    if strategy not in {"beam", "nucleus"}:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose 'beam' or 'nucleus'"
        )

    # Make sure the tokenizer has a pad token — required by generate()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder.eval()
    decoder.eval()
    adaptor.eval()

    prefix = compute_soft_prefix(encoder, adaptor, pixel_values)  # (B, k, D)
    b, k, _ = prefix.shape

    # Match generate()'s expectation: attention_mask aligns to inputs_embeds
    attention_mask = torch.ones((b, k), dtype=torch.long, device=prefix.device)

    if strategy == "beam":
        gen_kwargs = dict(
            num_beams=beam_size,
            early_stopping=True,
            do_sample=False,
            length_penalty=1.0,
        )
    else:
        gen_kwargs = dict(
            do_sample=True,
            top_p=top_p,
            num_beams=1,
        )

    # When inputs_embeds is provided (no input_ids), HF returns ONLY the
    # newly generated tokens — not the input prefix — so we can decode
    # the result directly.
    output_ids = decoder.generate(
        inputs_embeds=prefix,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **gen_kwargs,
    )

    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # Strip leading whitespace and collapse internal whitespace
    return [c.strip() for c in captions]
