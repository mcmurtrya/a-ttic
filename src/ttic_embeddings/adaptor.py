"""Prefix adaptor — encoder patch tokens to k soft prompt tokens for the LLM.

This is the only trainable module in the captioning pipeline. The encoder
(CLIP/SigLIP/DINOv2/MAE) and the LLM (GPT-2 medium) are both frozen; the
adaptor is what learns to translate encoder representations into a prefix
the LLM can read.

Design decisions (all justified in methods.md):

  Two-layer MLP. Higher-capacity adaptors (Q-Former, deep transformers)
  risk absorbing the encoder differences this project is designed to
  measure. The experiment depends on the adaptor being a passive
  translator, not an interpreter.

  Mean-pool over patch tokens before the MLP. We do not use the CLS
  token because the four encoders concentrate information at CLS very
  differently (CLIP/SigLIP heavily, DINOv2/MAE less so). Mean-pooling
  treats them symmetrically and naturally handles MAE's 196-patch input
  against the other three encoders' 256-patch input — the MLP's input
  dimensionality is constant.

  k = 10 soft tokens at the decoder hidden dim. Small enough that the
  prefix doesn't dominate the LLM's attention, large enough to carry
  scene-level information.

Forward pass:
    patches  (B, N, D_enc)
      -- mean-pool over N -->  (B, D_enc)
      -- linear -->            (B, H)
      -- GELU + dropout -->
      -- linear -->            (B, k * D_dec)
      -- reshape -->           (B, k, D_dec)
"""
from __future__ import annotations

import torch
import torch.nn as nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class PrefixAdaptor(nn.Module):
    """MLP mapping encoder patch features to k soft prompt tokens.

    Args:
        encoder_dim: hidden dimensionality of the encoder's patch tokens.
            All four encoders in this project use 1024.
        decoder_dim: hidden dimensionality of the LLM's input embeddings
            (1024 for GPT-2 medium).
        k_soft_tokens: number of soft prefix tokens to produce.
        hidden_dim: width of intermediate MLP layers.
        num_layers: total number of linear layers in the MLP. Default
            is 2 (one hidden representation, two weight matrices) per
            the "two-layer MLP" commitment in methods.md. num_layers=1
            collapses to a single linear projection; num_layers=3+ adds
            hidden layers (available for ablation, not the headline run).
        dropout: dropout probability applied between hidden layers.
        activation: nonlinearity, one of "gelu", "relu", "silu".
    """

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

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.k = k_soft_tokens
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

        # Build the MLP: (num_layers - 1) hidden blocks, then a final
        # projection layer to k * decoder_dim.
        Activation = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_dim = encoder_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(Activation())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, k_soft_tokens * decoder_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """Map encoder patch tokens to a fixed-size soft prefix.

        Args:
            patch_features: shape (B, N, encoder_dim). N varies across
                encoders (256 for CLIP/SigLIP/DINOv2, 196 for MAE);
                the mean-pool below makes this transparent.

        Returns:
            Tensor of shape (B, k_soft_tokens, decoder_dim), suitable
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

        pooled = patch_features.mean(dim=1)        # (B, encoder_dim)
        out = self.mlp(pooled)                      # (B, k * decoder_dim)
        return out.view(-1, self.k, self.decoder_dim)

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

    # 256-patch input (CLIP / SigLIP / DINOv2)
    x256 = torch.randn(2, 256, 1024)
    y256 = a(x256)
    print(f"  256-patch input  in={tuple(x256.shape)} -> out={tuple(y256.shape)}")
    assert y256.shape == (2, 10, 1024)

    # 196-patch input (MAE) — mean-pool absorbs the asymmetry
    x196 = torch.randn(2, 196, 1024)
    y196 = a(x196)
    print(f"  196-patch input  in={tuple(x196.shape)} -> out={tuple(y196.shape)}")
    assert y196.shape == (2, 10, 1024)

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
