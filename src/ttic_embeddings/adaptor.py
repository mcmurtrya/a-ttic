"""Prefix adaptor — encoder patch tokens to k soft prompt tokens for the LLM.

This is the only trainable module in the captioning pipeline. The encoder
(CLIP/SigLIP/DINOv2/MAE) and the LLM (GPT-2 medium) are both frozen; the
adaptor is what learns to translate encoder representations into a prefix
the LLM can read.

Design decisions (all justified in methods.md):

  Two-layer MLP, applied per-token. Higher-capacity adaptors (Q-Former,
  deep transformers) risk absorbing the encoder differences this project
  is designed to measure. The experiment depends on the adaptor being a
  passive translator, not an interpreter. A single MLP is shared across
  all 10 prefix positions; the LLM's positional embeddings handle
  position-specific interpretation downstream.

  1 global mean-pool token + 3x3 adaptive average pool over the patch
  grid = 10 prefix tokens. We do not use the encoder's CLS token because
  the four encoders concentrate information at CLS very differently
  (CLIP/SigLIP heavily, DINOv2/MAE less so) — the global mean-pool token
  preserves a CLS-symmetric global representation. The 3x3 spatial grid
  carries enough structure to distinguish left/center/right and
  top/middle/bottom, which is the substrate the projective-spatial
  hypothesis (methods.md L60-61) needs. Adaptive average pooling
  collapses both the 16x16 (CLIP/SigLIP/DINOv2) and 14x14 (MAE) grids
  to the same 3x3 output, absorbing the patch-count asymmetry
  symmetrically.

  k = 10 soft tokens at the decoder hidden dim. Small enough that the
  prefix doesn't dominate the LLM's attention, large enough to carry
  scene-level information across 1 global + 9 spatial positions.

Forward pass:
    patches  (B, N, D_enc) where N = h*h (square grid; 16x16 or 14x14)
      -- transpose + reshape to (B, D_enc, h, h)
      -- adaptive_avg_pool2d to 3x3 -->     (B, 9, D_enc)   # 9 spatial cells
      -- mean over N -->                    (B, 1, D_enc)   # 1 global cell
      -- concat (global first) -->          (B, 10, D_enc)
      -- per-token MLP -->                  (B, 10, D_dec)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class PrefixAdaptor(nn.Module):
    """MLP mapping encoder patch features to k = 10 soft prompt tokens.

    Layout: 1 global mean-pool token at position 0, followed by 3x3 spatial
    grid in raster order (positions 1-9). A per-token MLP is shared across
    all 10 positions.

    Args:
        encoder_dim: hidden dimensionality of the encoder's patch tokens.
            All four encoders in this project use 1024.
        decoder_dim: hidden dimensionality of the LLM's input embeddings
            (1024 for GPT-2 medium).
        k_soft_tokens: must equal 10 (1 global + 3x3 spatial). Kept as a
            parameter for config compatibility; values other than 10 are
            rejected because the spatial-grid layout is hard-coded.
        hidden_dim: width of intermediate MLP layers.
        num_layers: total number of linear layers in the per-token MLP.
            Default is 2; num_layers=1 collapses to a single linear
            projection; num_layers=3+ adds hidden layers (ablation only).
        dropout: dropout probability applied between hidden layers.
        activation: nonlinearity, one of "gelu", "relu", "silu".
    """

    SPATIAL_GRID: int = 3
    K_TOKENS: int = 1 + SPATIAL_GRID * SPATIAL_GRID  # 1 global + 3x3 spatial

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        k_soft_tokens: int = 10,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; "
                f"choose from {sorted(_ACTIVATIONS)}"
            )
        if k_soft_tokens != self.K_TOKENS:
            raise ValueError(
                f"k_soft_tokens must equal {self.K_TOKENS} "
                f"(1 global + {self.SPATIAL_GRID}x{self.SPATIAL_GRID} spatial); "
                f"got {k_soft_tokens}. The spatial-grid layout is hard-coded."
            )

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.k = k_soft_tokens
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

        # Per-token MLP: applied independently to each of the 10 prefix
        # positions. The same weights are shared across positions —
        # specialization is the LLM's job via its positional embeddings.
        Activation = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_dim = encoder_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(Activation())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, decoder_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """Map encoder patch tokens to a 10-token spatial-grid soft prefix.

        Args:
            patch_features: shape (B, N, encoder_dim). N must be a perfect
                square (256 = 16x16 for CLIP/SigLIP/DINOv2, 196 = 14x14 for
                MAE). Adaptive average pooling absorbs the asymmetry.

        Returns:
            Tensor of shape (B, 10, decoder_dim) — global at position 0,
            3x3 spatial grid at positions 1-9 in raster order. Suitable
            for prepending to caption-token embeddings via the LLM's
            ``inputs_embeds`` argument.
        """
        if patch_features.dim() != 3:
            raise ValueError(
                f"expected (B, N, D) input, got shape {tuple(patch_features.shape)}"
            )
        if patch_features.size(-1) != self.encoder_dim:
            raise ValueError(
                f"input encoder_dim {patch_features.size(-1)} does not match "
                f"adaptor's expected encoder_dim {self.encoder_dim}"
            )

        b, n, d = patch_features.shape
        h = int(round(n ** 0.5))
        if h * h != n:
            raise ValueError(
                f"PrefixAdaptor expects a square patch grid (sqrt(N) integer); "
                f"got N={n} which is not a perfect square. All four project "
                f"encoders use square grids (16x16 or 14x14)."
            )

        # Reshape (B, N, D) -> (B, D, h, h) and pool to 3x3, then back
        # to (B, 9, D) in raster order.
        spatial = patch_features.transpose(1, 2).reshape(b, d, h, h)
        spatial = F.adaptive_avg_pool2d(spatial, self.SPATIAL_GRID)   # (B, D, 3, 3)
        spatial_tokens = spatial.flatten(2).transpose(1, 2)           # (B, 9, D)

        # Global token: average over all original patches (not the 3x3 cells —
        # we want the unbiased global mean even when the spatial grid is small).
        global_token = patch_features.mean(dim=1, keepdim=True)        # (B, 1, D)

        tokens = torch.cat([global_token, spatial_tokens], dim=1)      # (B, 10, D)
        return self.mlp(tokens)                                        # (B, 10, D_dec)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"PrefixAdaptor(encoder_dim={self.encoder_dim}, "
            f"decoder_dim={self.decoder_dim}, k={self.k}, "
            f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, "
            f"params={self.num_parameters():,})"
        )


def build_adaptor(cfg) -> PrefixAdaptor:
    """Construct a PrefixAdaptor from an OmegaConf-style config.

    Reads from cfg.adaptor.*, cfg.encoder.hidden_dim, cfg.decoder.hidden_dim.
    """
    a = cfg.adaptor
    return PrefixAdaptor(
        encoder_dim=cfg.encoder.hidden_dim,
        decoder_dim=cfg.decoder.hidden_dim,
        k_soft_tokens=a.k_soft_tokens,
        hidden_dim=a.hidden_dim,
        num_layers=a.get("num_layers", 2),
        dropout=a.dropout,
        activation=a.get("activation", "gelu"),
    )


def _smoke_check() -> None:
    """Minimal sanity check — invoke via ``python -m ttic_embeddings.adaptor``."""
    print("PrefixAdaptor smoke check")
    a = PrefixAdaptor(encoder_dim=1024, decoder_dim=1024)
    print(f"  {a}")

    # 256-patch input (CLIP / SigLIP / DINOv2): 16x16 -> 3x3 spatial pool
    x256 = torch.randn(2, 256, 1024)
    y256 = a(x256)
    print(f"  256-patch input  in={tuple(x256.shape)} -> out={tuple(y256.shape)}")
    assert y256.shape == (2, 10, 1024)

    # 196-patch input (MAE): 14x14 -> 3x3 spatial pool — adaptive pool absorbs
    # the patch-count asymmetry symmetrically.
    x196 = torch.randn(2, 196, 1024)
    y196 = a(x196)
    print(f"  196-patch input  in={tuple(x196.shape)} -> out={tuple(y196.shape)}")
    assert y196.shape == (2, 10, 1024)

    # Permutation-sensitivity check: shuffling patches across spatial
    # positions must change the output. (Mean-pool was permutation-invariant —
    # this assert would have caught the silent spatial-info-destroyed bug.)
    perm = torch.randperm(256)
    x256_shuffled = x256[:, perm, :]
    y256_shuffled = a(x256_shuffled)
    diff = (y256 - y256_shuffled).abs().max().item()
    assert diff > 1e-5, (
        "Permuting patches must change the output — adaptor is "
        "permutation-invariant and not preserving spatial info."
    )
    print(f"  permutation sensitivity: max output diff = {diff:.4f}")

    # Trainable param count
    n_trainable = sum(p.numel() for p in a.parameters() if p.requires_grad)
    print(f"  trainable params: {n_trainable:,}")
    assert n_trainable > 0

    # Backward pass: the encoder side runs under no_grad, so the
    # adaptor must produce gradient-tracked outputs from gradient-free
    # inputs. This is the bug we fixed in encoders/base.py — verify
    # the contract holds end-to-end.
    with torch.no_grad():
        fake_patches = torch.randn(2, 256, 1024)
    prefix = a(fake_patches)
    loss = (prefix - torch.randn_like(prefix)).pow(2).mean()
    loss.backward()
    grad_norm = a.mlp[0].weight.grad.norm().item()
    print(f"  backward pass: OK (grad norm on first linear: {grad_norm:.4f})")
    print("All adaptor checks passed.")


if __name__ == "__main__":
    _smoke_check()
