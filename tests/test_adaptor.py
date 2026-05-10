"""Tests for PrefixAdaptor.

The most important test in this file is the permutation-sensitivity check:
mean-pool would silently destroy spatial information before the LLM ever
saw it, undermining the methods.md L60-61 spatial-language hypothesis.
That bug was invisible from shape and gradient checks alone — a permutation
test catches it directly.
"""
from __future__ import annotations

import pytest
import torch

from ttic_embeddings.adaptor import PrefixAdaptor


@pytest.fixture(scope="module")
def adaptor():
    # Eval mode disables dropout so determinism-sensitive tests (permutation
    # invariance of the global token, spatial-swap deltas) aren't perturbed
    # by stochastic noise. Tests that need gradients still get them — eval
    # mode only affects dropout/batchnorm, not requires_grad.
    a = PrefixAdaptor(encoder_dim=1024, decoder_dim=1024)
    a.eval()
    return a


class TestOutputShape:
    def test_clip_siglip_dinov2_grid(self, adaptor):
        # 16x16 = 256 patches
        x = torch.randn(2, 256, 1024)
        y = adaptor(x)
        assert y.shape == (2, 10, 1024)

    def test_mae_grid(self, adaptor):
        # 14x14 = 196 patches — adaptive pool to 3x3 absorbs the asymmetry
        x = torch.randn(2, 196, 1024)
        y = adaptor(x)
        assert y.shape == (2, 10, 1024)

    def test_non_square_grid_rejected(self, adaptor):
        x = torch.randn(2, 250, 1024)  # not a perfect square
        with pytest.raises(ValueError, match="square patch grid"):
            adaptor(x)

    def test_wrong_encoder_dim_rejected(self, adaptor):
        x = torch.randn(2, 256, 768)  # wrong inner dim
        with pytest.raises(ValueError, match="encoder_dim"):
            adaptor(x)


class TestSpatialPreservation:
    """The whole point of the redesign: spatial structure must reach the LLM."""

    def test_patch_permutation_changes_output(self, adaptor):
        """Shuffling patches across positions must produce a different output.

        Under mean-pool this assertion fails (mean is permutation-invariant);
        under spatial-grid pooling, different patches land in different
        3x3 cells and the output changes. This is the assertion that would
        have caught the original silent bug.
        """
        torch.manual_seed(0)
        x = torch.randn(2, 256, 1024)
        perm = torch.randperm(256)
        x_shuffled = x[:, perm, :]

        with torch.no_grad():
            y = adaptor(x)
            y_shuffled = adaptor(x_shuffled)

        max_diff = (y - y_shuffled).abs().max().item()
        assert max_diff > 1e-4, (
            f"Output is permutation-invariant (max diff {max_diff:.2e}); "
            f"spatial information is being destroyed."
        )

    def test_global_token_is_permutation_invariant(self, adaptor):
        """Position 0 (the global mean-pool token) MUST be permutation-invariant.

        That is the role of the global token: it preserves the encoder-symmetric
        average representation. Only positions 1-9 (the spatial grid) should
        depend on patch order.
        """
        torch.manual_seed(0)
        x = torch.randn(2, 256, 1024)
        perm = torch.randperm(256)
        x_shuffled = x[:, perm, :]

        with torch.no_grad():
            y = adaptor(x)
            y_shuffled = adaptor(x_shuffled)

        global_diff = (y[:, 0, :] - y_shuffled[:, 0, :]).abs().max().item()
        spatial_diff = (y[:, 1:, :] - y_shuffled[:, 1:, :]).abs().max().item()
        assert global_diff < 1e-4, (
            f"Global token (position 0) should be permutation-invariant; "
            f"got max diff {global_diff:.2e}."
        )
        assert spatial_diff > 1e-4, (
            f"Spatial tokens (positions 1-9) should depend on patch order; "
            f"got max diff {spatial_diff:.2e}."
        )

    def test_spatial_swap_changes_output(self, adaptor):
        """Swapping the top-left and bottom-right patch quadrants must change
        the spatial tokens. This tests that the 3x3 grid actually maps to
        spatial position, not just permutation noise.
        """
        torch.manual_seed(0)
        # 16x16 grid: top-left 8x8 vs bottom-right 8x8
        x = torch.randn(1, 256, 1024)
        x_grid = x.view(1, 16, 16, 1024)
        x_swapped = x_grid.clone()
        x_swapped[:, :8, :8, :] = x_grid[:, 8:, 8:, :]
        x_swapped[:, 8:, 8:, :] = x_grid[:, :8, :8, :]
        x_swapped = x_swapped.view(1, 256, 1024)

        with torch.no_grad():
            y = adaptor(x)
            y_swapped = adaptor(x_swapped)

        # Spatial tokens differ; global token does not (same patches, just
        # rearranged — global mean is invariant).
        global_diff = (y[:, 0, :] - y_swapped[:, 0, :]).abs().max().item()
        spatial_diff = (y[:, 1:, :] - y_swapped[:, 1:, :]).abs().max().item()
        assert global_diff < 1e-4
        assert spatial_diff > 1e-4


class TestParamConstraint:
    def test_k_must_be_10(self):
        with pytest.raises(ValueError, match="k_soft_tokens must equal 10"):
            PrefixAdaptor(encoder_dim=1024, decoder_dim=1024, k_soft_tokens=8)

    def test_gradients_flow(self, adaptor):
        x = torch.randn(2, 256, 1024)
        y = adaptor(x)
        loss = y.pow(2).mean()
        loss.backward()
        # First linear should have non-zero gradient
        first_linear = adaptor.mlp[0]
        assert first_linear.weight.grad is not None
        assert first_linear.weight.grad.norm().item() > 0
