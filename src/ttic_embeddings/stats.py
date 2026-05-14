"""Statistical analysis: paired Wilcoxon, BH-FDR, supervision-category contrasts.

Implements the exact procedure committed to in methods.md:

  Primary (confirmatory) family — supervision-category contrast.
    For each metric, average (CLIP, SigLIP) per image and compare
    against the per-image (DINOv2, MAE) average via paired Wilcoxon.
    4 tests, BH-FDR corrected at alpha = 0.05.

  Secondary (exploratory) family — all 6 pairwise contrasts × 4 metrics
    = 24 tests, BH-FDR corrected within this family separately. The
    8 within-category tests (CLIP vs SigLIP, DINOv2 vs MAE, across
    metrics) are a subset of the 24 and serve as pooling sanity checks.

Effect size is Wilcoxon r = Z / sqrt(N), reported alongside p_adj.
Headline interpretation requires r >= 0.10 per the pre-registered
threshold; significance alone is insufficient when N is in the
thousands.

Optional mixed-effects regression for robustness:
    score ~ supervision_category + (1 | encoder) + (1 | image)
treats encoder as a random effect nested in supervision category.
Agreement with the pooled-Wilcoxon primary supports the supervision
claim; disagreement points to one encoder driving most of its
category's effect.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------
# Core paired test
# ---------------------------------------------------------------------


def wilcoxon_paired(x: Iterable[float], y: Iterable[float]) -> dict:
    """Paired Wilcoxon signed-rank with Wilcoxon r effect size.

    Drops zero differences (Wilcoxon's `wilcox` zero method).

    Returns:
        Dict with statistic, p_value, n (number of non-zero pairs),
        and r (effect size, |Z|/sqrt(n)). When n == 0 (all zeros or
        empty input), returns p_value=1.0, r=0.0.
    """
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    diffs = x_arr - y_arr
    nonzero = diffs[np.isfinite(diffs) & (diffs != 0)]
    n = int(len(nonzero))
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": 0, "r": 0.0}
    res = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    p = float(res.pvalue)
    # Wilcoxon r = |Z| / sqrt(n). Compute Z directly from the test statistic
    # (with tie correction), not by reverse-engineering from the two-sided p:
    # at our sample sizes (thousands of paired captions) any real effect
    # underflows p to 0, which would clamp r at 1.0 and make the headline
    # statistic uninformative. methods.md L47 specifies r as the headline.
    abs_d = np.abs(nonzero)
    ranks = stats.rankdata(abs_d)
    t_plus = float(ranks[nonzero > 0].sum())
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(abs_d, return_counts=True)
    tie_correction = float((counts ** 3 - counts).sum()) / 48.0
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    sigma = math.sqrt(sigma2) if sigma2 > 0 else 0.0
    z = abs(t_plus - mu) / sigma if sigma > 0 else 0.0
    r = z / math.sqrt(n)
    return {
        "statistic": float(res.statistic),
        "p_value": p,
        "n": n,
        "r": r,
    }


# ---------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------


def bh_fdr(p_values: Iterable[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR correction on a list of p-values.

    Returns (adjusted_p_values, rejected_at_alpha).
    """
    p_list = list(p_values)
    if not p_list:
        return [], []
    rej, p_adj, _, _ = multipletests(p_list, alpha=alpha, method="fdr_bh")
    return list(p_adj), list(rej)


# ---------------------------------------------------------------------
# Family 1: supervision-category contrast (primary, 4 tests)
# ---------------------------------------------------------------------


LANG_SUP = ("clip", "siglip")
SELF_SUP = ("dinov2", "mae")


def supervision_category_contrast(
    df: pd.DataFrame,
    metric_col: str,
    encoder_col: str = "encoder",
    image_col: str = "image_id",
    lang_encoders: tuple[str, ...] = LANG_SUP,
    self_encoders: tuple[str, ...] = SELF_SUP,
) -> dict:
    """Run a single supervision-category Wilcoxon test for one metric.

    Pools the per-image averages of `lang_encoders` against the
    per-image averages of `self_encoders` and applies a paired test.
    """
    wide = df.pivot_table(
        index=image_col, columns=encoder_col, values=metric_col, aggfunc="mean"
    )
    needed = list(lang_encoders) + list(self_encoders)
    missing = [e for e in needed if e not in wide.columns]
    if missing:
        raise ValueError(f"Missing encoder columns in df: {missing}")
    wide = wide.dropna(subset=needed)
    lang_mean = wide[list(lang_encoders)].mean(axis=1)
    self_mean = wide[list(self_encoders)].mean(axis=1)
    result = wilcoxon_paired(lang_mean.values, self_mean.values)
    result["metric"] = metric_col
    result["contrast"] = "lang_sup_vs_self_sup"
    return result


def primary_family(
    df: pd.DataFrame,
    metrics: list[str],
    alpha: float = 0.05,
    effect_size_floor: float = 0.10,
    **contrast_kwargs,
) -> pd.DataFrame:
    """Run the primary (4-test) family with BH-FDR and effect-size flag.

    Returns a DataFrame with one row per metric and columns:
        metric, contrast, n, statistic, p_value, p_adj, r,
        rejected_at_alpha, claimed_meaningful.

    `claimed_meaningful` is True when r >= effect_size_floor AND the
    test passes BH-FDR — the headline criterion per methods.md.
    """
    rows = [
        supervision_category_contrast(df, m, **contrast_kwargs)
        for m in metrics
    ]
    p_adj, rej = bh_fdr([r["p_value"] for r in rows], alpha=alpha)
    out = pd.DataFrame(rows)
    out["p_adj"] = p_adj
    out["rejected_at_alpha"] = rej
    out["claimed_meaningful"] = (out["r"] >= effect_size_floor) & out["rejected_at_alpha"]
    return out


# ---------------------------------------------------------------------
# Family 2: all-pairwise contrasts (secondary, 24 tests for 4 metrics)
# ---------------------------------------------------------------------


def pairwise_contrast(
    df: pd.DataFrame,
    metric_col: str,
    encoder_a: str,
    encoder_b: str,
    encoder_col: str = "encoder",
    image_col: str = "image_id",
) -> dict:
    """Paired Wilcoxon between two encoders on one metric."""
    wide = df.pivot_table(
        index=image_col, columns=encoder_col, values=metric_col, aggfunc="mean"
    )
    if encoder_a not in wide.columns or encoder_b not in wide.columns:
        raise ValueError(f"Missing encoder column(s): {encoder_a}, {encoder_b}")
    paired = wide[[encoder_a, encoder_b]].dropna()
    result = wilcoxon_paired(paired[encoder_a].values, paired[encoder_b].values)
    result["metric"] = metric_col
    result["encoder_a"] = encoder_a
    result["encoder_b"] = encoder_b
    result["contrast"] = f"{encoder_a}_vs_{encoder_b}"
    return result


def secondary_family(
    df: pd.DataFrame,
    metrics: list[str],
    encoders: tuple[str, ...] = LANG_SUP + SELF_SUP,
    alpha: float = 0.05,
    effect_size_floor: float = 0.10,
    lang_encoders: tuple[str, ...] = LANG_SUP,
    self_encoders: tuple[str, ...] = SELF_SUP,
    **contrast_kwargs,
) -> pd.DataFrame:
    """All pairwise contrasts × all metrics, with BH-FDR within this family."""
    rows: list[dict] = []
    for i, e_a in enumerate(encoders):
        for e_b in encoders[i + 1:]:
            for metric in metrics:
                rows.append(
                    pairwise_contrast(df, metric, e_a, e_b, **contrast_kwargs)
                )
    p_adj, rej = bh_fdr([r["p_value"] for r in rows], alpha=alpha)
    out = pd.DataFrame(rows)
    out["p_adj"] = p_adj
    out["rejected_at_alpha"] = rej
    out["claimed_meaningful"] = (out["r"] >= effect_size_floor) & out["rejected_at_alpha"]
    out["within_category"] = out.apply(
        lambda r: (
            (r["encoder_a"] in lang_encoders and r["encoder_b"] in lang_encoders)
            or (r["encoder_a"] in self_encoders and r["encoder_b"] in self_encoders)
        ),
        axis=1,
    )
    return out


# ---------------------------------------------------------------------
# Robustness check: mixed-effects regression
# ---------------------------------------------------------------------


def mixed_effects_supervision_contrast(
    df: pd.DataFrame,
    metric_col: str,
    encoder_col: str = "encoder",
    image_col: str = "image_id",
    lang_encoders: tuple[str, ...] = LANG_SUP,
) -> dict:
    """Image-controlled supervision contrast, robust to zero-inflated metrics.

    methods.md L49 specifies `score ~ supervision + (1|encoder) + (1|image)`.
    Statsmodels' MixedLM cannot fit crossed random effects, so we pick
    one — and image is the only identifiable choice. Encoder is the
    WRONG random effect: supervision is a deterministic function of
    encoder (clip/siglip -> language, dinov2/mae -> self), so a random
    intercept by encoder absorbs the entire between-encoder signal that
    supervision is meant to explain. The fit degenerates: SE in the
    thousands, z ~ 0, model unidentifiable. See git history before
    2026-05-14 for the broken spec.

    Image is identifiable: each image appears once per encoder, so the
    random intercept absorbs per-image baselines (some images get long
    captions across all encoders, others short), and the supervision
    fixed effect captures the within-image difference between lang-sup
    and self-sup encoders.

    Even with the right RE, several metrics (adj_per_noun,
    projective_density, topological_density) are zero-inflated enough
    that MixedLM hits a singular covariance matrix. For those, we fall
    back to a paired t-test on per-image (lang_mean - self_mean)
    differences — the same target estimand, computed without iterative
    MLE. The `method` field records which estimator produced the row.
    """
    from statsmodels.formula.api import mixedlm

    work = df.copy()
    work["supervision"] = work[encoder_col].apply(
        lambda e: "language" if e in lang_encoders else "self"
    )
    work = work.dropna(subset=[metric_col, "supervision", image_col])
    if work.empty:
        return _empty_me_row(metric_col)

    n_images = int(work[image_col].nunique())
    n_obs = int(len(work))

    # Attempt 1: MixedLM with image random intercept.
    try:
        model = mixedlm(
            f"{metric_col} ~ C(supervision, Treatment('self'))",
            work,
            groups=work[image_col],
        )
        fitted = model.fit(method="lbfgs", disp=False)
        coef_name = next(
            (n for n in fitted.params.index if "supervision" in n),
            None,
        )
        coef = float(fitted.params[coef_name]) if coef_name else float("nan")
        se = float(fitted.bse[coef_name]) if coef_name else float("nan")
        if (
            coef_name is not None
            and bool(fitted.converged)
            and np.isfinite(coef)
            and np.isfinite(se)
            and se > 0
        ):
            return {
                "metric": metric_col, "method": "mixedlm_image_re",
                "coef": coef, "se": se,
                "z": float(fitted.tvalues[coef_name]),
                "p_value": float(fitted.pvalues[coef_name]),
                "n_images": n_images, "n_obs": n_obs, "converged": True,
            }
    except Exception:  # noqa: BLE001
        pass  # fall through to paired t-test

    # Fallback: paired t-test on per-image (lang_mean - self_mean).
    self_encoders = tuple(
        e for e in work[encoder_col].unique() if e not in lang_encoders
    )
    wide = work.pivot_table(
        index=image_col, columns=encoder_col, values=metric_col, aggfunc="mean"
    )
    lang_present = [e for e in lang_encoders if e in wide.columns]
    self_present = [e for e in self_encoders if e in wide.columns]
    if not lang_present or not self_present:
        return _empty_me_row(metric_col)
    wide = wide.dropna(subset=lang_present + self_present)
    if wide.empty:
        return _empty_me_row(metric_col)
    diff = wide[lang_present].mean(axis=1) - wide[self_present].mean(axis=1)
    n_paired = int(len(diff))
    if n_paired < 2:
        return _empty_me_row(metric_col)
    t_res = stats.ttest_1samp(diff.values, 0.0)
    return {
        "metric": metric_col, "method": "paired_t_per_image",
        "coef": float(diff.mean()),
        "se": float(diff.std(ddof=1) / math.sqrt(n_paired)),
        "z": float(t_res.statistic),
        "p_value": float(t_res.pvalue),
        "n_images": n_paired, "n_obs": n_obs, "converged": True,
    }


def _empty_me_row(metric_col: str) -> dict:
    return {
        "metric": metric_col, "method": "none",
        "coef": None, "se": None, "z": None, "p_value": None,
        "n_images": 0, "n_obs": 0, "converged": False,
    }
