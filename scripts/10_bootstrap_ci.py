"""Image-level paired bootstrap for the Wilcoxon r effect size.

For each (decoder, metric, seed), computes:
  - point estimate of Wilcoxon r for the lang_sup vs DINOv2 contrast
    (MAE excluded), matching scripts/06_analyze.py
  - bootstrap 95% CI via percentile method, resampling images with
    replacement (paired-by-image)

Also reports a bootstrap CI for the cross-seed mean r:
  - per iteration, independently resample images within each seed,
    compute r per seed, take the mean of the three rs

Writes captions/bootstrap_ci.csv.

Usage:
  python scripts/10_bootstrap_ci.py \
    --scores captions/scores_seed1.csv captions/scores_seed2.csv \
             captions/scores_seed3.csv \
    --metrics projective_density caption_length \
    --decoders beam nucleus \
    --n-boot 1000 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LANG_SUP = ("clip", "siglip")
SELF_SUP_HEADLINE = ("dinov2",)  # MAE excluded


def _wilcoxon_r(diff: np.ndarray) -> float:
    """Wilcoxon r = |Z| / sqrt(N_nonzero), matching stats.py."""
    nz = diff[diff != 0]
    if nz.size < 2:
        return float("nan")
    try:
        res = wilcoxon(nz, alternative="two-sided", zero_method="wilcox",
                       method="approx")
    except ValueError:
        return float("nan")
    # scipy returns statistic + pvalue; we want |Z| from approx
    # approx mode uses normal approximation: Z = (W - mean) / sd
    # Reconstruct from pvalue: |Z| = sqrt(2) * erfinv(1 - pvalue) when two-sided
    # Easier: use the normal-approx Z directly via wilcoxon's z attribute when
    # available; otherwise from pvalue.
    # scipy >= 1.11 exposes WilcoxonResult.zstatistic
    if hasattr(res, "zstatistic") and res.zstatistic is not None:
        z = abs(float(res.zstatistic))
    else:
        from scipy.special import ndtri
        # two-sided p -> |Z|
        p = max(float(res.pvalue), 1e-300)
        z = abs(ndtri(1 - p / 2))
    return z / np.sqrt(nz.size)


def per_image_diff(df: pd.DataFrame, metric: str, decoder: str) -> pd.DataFrame:
    """Return DataFrame [image_id, diff] of (lang_mean - dinov2)."""
    d = df[(df["metric"] == metric) & (df["decoder"] == decoder)]
    wide = d.pivot_table(index="image_id", columns="encoder",
                         values="score", aggfunc="mean")
    needed = list(LANG_SUP) + list(SELF_SUP_HEADLINE)
    wide = wide.dropna(subset=needed)
    lang_mean = wide[list(LANG_SUP)].mean(axis=1)
    self_mean = wide[list(SELF_SUP_HEADLINE)].mean(axis=1)
    return pd.DataFrame({
        "image_id": wide.index,
        "diff": (lang_mean - self_mean).values,
    })


def bootstrap_r(diff: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = diff.size
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rs[b] = _wilcoxon_r(diff[idx])
    return rs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", nargs="+", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+",
                        default=["projective_density", "caption_length"])
    parser.add_argument("--decoders", nargs="+", default=["beam", "nucleus"])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("captions/bootstrap_ci.csv"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load all seeds keyed by seed number
    by_seed: dict[int, pd.DataFrame] = {}
    for p in args.scores:
        df = pd.read_csv(p)
        seed_vals = df["seed"].unique()
        if len(seed_vals) != 1:
            raise ValueError(f"{p}: expected single seed, got {seed_vals}")
        by_seed[int(seed_vals[0])] = df

    rows: list[dict] = []
    for decoder in args.decoders:
        for metric in args.metrics:
            # Per-seed
            per_seed_diffs: dict[int, np.ndarray] = {}
            for seed, df in sorted(by_seed.items()):
                diff_df = per_image_diff(df, metric, decoder)
                diff = diff_df["diff"].values
                per_seed_diffs[seed] = diff
                r_point = _wilcoxon_r(diff)
                rs_boot = bootstrap_r(diff, args.n_boot, rng)
                lo, hi = np.nanpercentile(rs_boot, [2.5, 97.5])
                rows.append({
                    "decoder": decoder, "metric": metric, "seed": seed,
                    "n_pairs": int(diff.size),
                    "r_point": r_point,
                    "r_boot_mean": float(np.nanmean(rs_boot)),
                    "r_boot_lo": float(lo), "r_boot_hi": float(hi),
                })

            # Cross-seed mean r bootstrap
            mean_rs = np.empty(args.n_boot)
            for b in range(args.n_boot):
                per_seed_rs = []
                for seed in sorted(per_seed_diffs):
                    d = per_seed_diffs[seed]
                    idx = rng.integers(0, d.size, size=d.size)
                    per_seed_rs.append(_wilcoxon_r(d[idx]))
                mean_rs[b] = float(np.nanmean(per_seed_rs))
            mean_r_point = float(np.nanmean(
                [_wilcoxon_r(per_seed_diffs[s]) for s in sorted(per_seed_diffs)]
            ))
            lo, hi = np.nanpercentile(mean_rs, [2.5, 97.5])
            rows.append({
                "decoder": decoder, "metric": metric, "seed": "aggregate",
                "n_pairs": int(sum(d.size for d in per_seed_diffs.values())),
                "r_point": mean_r_point,
                "r_boot_mean": float(np.nanmean(mean_rs)),
                "r_boot_lo": float(lo), "r_boot_hi": float(hi),
            })
            print(f"{decoder:8s} {metric:24s} aggregate "
                  f"r={mean_r_point:.3f} CI=[{lo:.3f}, {hi:.3f}]")

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
