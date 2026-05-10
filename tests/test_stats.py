"""Statistical pipeline tests — synthetic data with planted effect sizes.

The point of these tests is to verify the stats module recovers what
we plant. We construct DataFrames where the supervision-category
contrast SHOULD be significant or null, then run the pipeline and
check that:

  - non-zero effects with reasonable N produce p < 0.001 and r > 0.10
  - null effects produce p > 0.05 and small r
  - the BH-FDR family-level correction does what it claims
  - within-category sanity-check tests are correctly flagged

These are the checks that defend against silent bugs in the analysis
spine — far more impactful than checking individual numerical edge
cases of scipy/statsmodels (which have their own test suites).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ttic_embeddings.stats import (
    LANG_SUP,
    SELF_SUP,
    bh_fdr,
    pairwise_contrast,
    primary_family,
    secondary_family,
    supervision_category_contrast,
    wilcoxon_paired,
)


# ---------------------------------------------------------------------
# Helpers — synthesize per-(image, encoder) score data
# ---------------------------------------------------------------------


def _synthesize(
    n_images: int,
    means: dict[str, float],
    sigma: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per (image_id, encoder), score drawn from N(mean, sigma).

    `means` maps encoder name to its true mean score.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for image_id in range(n_images):
        for enc, mu in means.items():
            rows.append(
                {
                    "image_id": image_id,
                    "encoder": enc,
                    "score": float(mu + rng.standard_normal() * sigma),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# wilcoxon_paired
# ---------------------------------------------------------------------


class TestWilcoxonPaired:
    def test_identical_inputs_no_test(self):
        result = wilcoxon_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        # All differences are zero -> n=0 -> we skip the test
        assert result["n"] == 0
        assert result["p_value"] == 1.0
        assert result["r"] == 0.0

    def test_recovers_clear_effect(self):
        # x systematically larger than y by ~1.0 over 100 pairs
        rng = np.random.default_rng(0)
        x = 1.0 + rng.standard_normal(100) * 0.2
        y = 0.0 + rng.standard_normal(100) * 0.2
        result = wilcoxon_paired(x, y)
        assert result["n"] == 100
        assert result["p_value"] < 1e-3
        assert result["r"] > 0.5

    def test_no_effect_high_p(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        result = wilcoxon_paired(x, y)
        # With independent N(0,1) samples, p should not reject at 0.01
        assert result["p_value"] > 0.01

    def test_drops_zero_diffs(self):
        # Half the pairs have zero diff; should drop to n=2
        result = wilcoxon_paired([1, 1, 1, 5], [1, 1, 1, 1])
        assert result["n"] == 1

    def test_r_not_clamped_at_underflow(self):
        """r is computed from the test statistic, not reverse-engineered from p.

        At our sample sizes (thousands of pairs) any real effect underflows
        p to 0. The previous reverse-engineered formula clamped r at 1.0 in
        that regime, making the pre-registered headline (r >= 0.10) unable
        to distinguish "huge effect" from "modest effect" once N got large.
        """
        rng = np.random.default_rng(0)
        n = 5000
        # Strong effect: every diff positive by ~1 sigma. r should be high
        # but well below the theoretical upper bound for Wilcoxon (~0.87).
        x = 1.0 + rng.standard_normal(n) * 0.2
        y = 0.0 + rng.standard_normal(n) * 0.2
        result_strong = wilcoxon_paired(x, y)
        assert result_strong["p_value"] == 0.0  # underflowed
        assert result_strong["r"] > 0.5
        assert result_strong["r"] < 0.95  # not pinned at 1.0

        # Smaller effect but same N — r should be smaller, not also clamped.
        # Cohen's d ~ 0.05 / 0.28 ≈ 0.18 — small effect.
        x2 = 1.05 + rng.standard_normal(n) * 0.2
        y2 = 1.00 + rng.standard_normal(n) * 0.2
        result_weak = wilcoxon_paired(x2, y2)
        assert result_weak["r"] < result_strong["r"] - 0.3


# ---------------------------------------------------------------------
# bh_fdr
# ---------------------------------------------------------------------


class TestBhFdr:
    def test_empty(self):
        p_adj, rej = bh_fdr([])
        assert p_adj == [] and rej == []

    def test_monotone_in_input(self):
        """Sort-stable: smallest raw p stays smallest after adjustment."""
        p_vals = [0.001, 0.01, 0.04, 0.5, 0.9]
        p_adj, _ = bh_fdr(p_vals)
        assert p_adj[0] == min(p_adj)

    def test_adjusted_at_least_raw(self):
        p_vals = [0.001, 0.01, 0.04, 0.5, 0.9]
        p_adj, _ = bh_fdr(p_vals)
        for raw, adj in zip(p_vals, p_adj):
            assert adj >= raw - 1e-12

    def test_alpha_threshold(self):
        p_vals = [0.001, 0.5, 0.5, 0.5]
        _, rej = bh_fdr(p_vals, alpha=0.05)
        assert rej[0] is True or rej[0] == True  # smallest passes
        assert all(not r for r in rej[1:])


# ---------------------------------------------------------------------
# supervision_category_contrast (single metric, single test)
# ---------------------------------------------------------------------


class TestSupervisionCategoryContrast:
    def test_recovers_planted_effect(self):
        # Lang-sup encoders systematically score 0.5 higher than self-sup
        df = _synthesize(
            n_images=200,
            means={"clip": 1.5, "siglip": 1.5, "dinov2": 1.0, "mae": 1.0},
            sigma=0.3,
            seed=0,
        )
        result = supervision_category_contrast(df, "score")
        assert result["p_value"] < 1e-3
        assert result["r"] > 0.3

    def test_null_effect(self):
        df = _synthesize(
            n_images=200,
            means={"clip": 1.0, "siglip": 1.0, "dinov2": 1.0, "mae": 1.0},
            sigma=0.3,
            seed=0,
        )
        result = supervision_category_contrast(df, "score")
        assert result["p_value"] > 0.05

    def test_small_effect_below_floor(self):
        # Effect ~0.05; large N produces small p but tiny r
        df = _synthesize(
            n_images=2000,
            means={"clip": 1.05, "siglip": 1.05, "dinov2": 1.0, "mae": 1.0},
            sigma=0.5,
            seed=0,
        )
        result = supervision_category_contrast(df, "score")
        # Effect should exist (p significant) but r should be small —
        # this is the case the r >= 0.10 floor was designed for.
        assert result["r"] < 0.20

    def test_missing_encoder_raises(self):
        df = _synthesize(
            n_images=10,
            means={"clip": 1.0, "siglip": 1.0, "dinov2": 1.0},
            seed=0,
        )
        with pytest.raises(ValueError, match="Missing encoder columns"):
            supervision_category_contrast(df, "score")


# ---------------------------------------------------------------------
# primary_family — multi-metric with FDR
# ---------------------------------------------------------------------


class TestPrimaryFamily:
    def _multi_metric_df(self, seed: int = 0, n_images: int = 1000) -> pd.DataFrame:
        """Three metrics with planted effects, one without.

        Uses N=1000 (not 200) for the "noise" metric reliability: at N=200,
        the r >= 0.10 floor only excludes p > 0.158, so under a true null
        ~5% of seeds spuriously pass the claimed_meaningful gate. At N=1000,
        r >= 0.10 requires p <= 0.0016, dropping the false-positive rate to
        ~0.16% per test.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for image_id in range(n_images):
            # lang-sup: higher specificity, lower spatial, higher abstraction
            for enc in ("clip", "siglip"):
                rows.append({
                    "image_id": image_id, "encoder": enc,
                    "specificity": 1.5 + rng.standard_normal() * 0.3,
                    "spatial": 0.05 + rng.standard_normal() * 0.02,
                    "abstraction": 8.0 + rng.standard_normal() * 0.5,
                    "noise": rng.standard_normal(),
                })
            for enc in ("dinov2", "mae"):
                rows.append({
                    "image_id": image_id, "encoder": enc,
                    "specificity": 1.0 + rng.standard_normal() * 0.3,
                    "spatial": 0.10 + rng.standard_normal() * 0.02,
                    "abstraction": 7.0 + rng.standard_normal() * 0.5,
                    "noise": rng.standard_normal(),
                })
        return pd.DataFrame(rows)

    def test_recovers_three_planted_one_null(self):
        df = self._multi_metric_df(seed=0)
        result = primary_family(
            df, ["specificity", "spatial", "abstraction", "noise"],
            alpha=0.05, effect_size_floor=0.10,
        )
        keyed = result.set_index("metric")
        # All three planted should be claimed_meaningful
        assert keyed.loc["specificity", "claimed_meaningful"]
        assert keyed.loc["spatial", "claimed_meaningful"]
        assert keyed.loc["abstraction", "claimed_meaningful"]
        # The noise metric should NOT be claimed meaningful
        assert not keyed.loc["noise", "claimed_meaningful"]

    def test_p_adj_present_and_monotone(self):
        df = self._multi_metric_df(seed=1)
        result = primary_family(df, ["specificity", "spatial", "abstraction", "noise"])
        assert "p_adj" in result.columns
        # p_adj >= p_value for each row
        assert (result["p_adj"] >= result["p_value"] - 1e-12).all()


# ---------------------------------------------------------------------
# secondary_family — pairwise contrasts with within-category flag
# ---------------------------------------------------------------------


class TestSecondaryFamily:
    def test_24_tests_for_4_metrics(self):
        df = _synthesize(
            n_images=200,
            means={"clip": 1.5, "siglip": 1.5, "dinov2": 1.0, "mae": 1.0},
            seed=0,
        )
        # 4 metrics with the same data shape
        for col in ["m1", "m2", "m3", "m4"]:
            df[col] = df["score"]
        result = secondary_family(df, ["m1", "m2", "m3", "m4"])
        # 6 encoder pairs * 4 metrics = 24 rows
        assert len(result) == 24

    def test_8_within_category_subset(self):
        df = _synthesize(
            n_images=200,
            means={"clip": 1.5, "siglip": 1.5, "dinov2": 1.0, "mae": 1.0},
            seed=0,
        )
        for col in ["m1", "m2", "m3", "m4"]:
            df[col] = df["score"]
        result = secondary_family(df, ["m1", "m2", "m3", "m4"])
        # within-category: (clip,siglip) and (dinov2,mae) for each of 4 metrics
        assert int(result["within_category"].sum()) == 8

    def test_within_category_no_planted_effect(self):
        # CLIP == SigLIP, DINOv2 == MAE — within-category contrasts should be null
        df = _synthesize(
            n_images=200,
            means={"clip": 1.5, "siglip": 1.5, "dinov2": 1.0, "mae": 1.0},
            seed=0,
        )
        result = secondary_family(df, ["score"])
        within = result[result["within_category"]]
        # No within-category test should be claimed meaningful
        assert not within["claimed_meaningful"].any()


# ---------------------------------------------------------------------
# pairwise_contrast — single (encoder_a, encoder_b, metric) test
# ---------------------------------------------------------------------


class TestPairwiseContrast:
    def test_basic(self):
        df = _synthesize(
            n_images=100,
            means={"clip": 2.0, "siglip": 2.0, "dinov2": 1.0, "mae": 1.0},
            seed=0,
        )
        result = pairwise_contrast(df, "score", "clip", "dinov2")
        assert result["encoder_a"] == "clip"
        assert result["encoder_b"] == "dinov2"
        assert result["p_value"] < 0.001
        assert result["r"] > 0.3
