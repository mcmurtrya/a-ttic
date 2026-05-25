"""Paired sign-flip permutation test under H0 of no encoder effect.

Under H0, the sign of each per-image (lang_mean - dinov2) difference is
equally likely to be + or -. We randomly flip signs B times, recompute
Wilcoxon r each time, and report:

  p_perm = (1 + #|r_perm| >= |r_observed|) / (B + 1)

This is a confirmatory check on the asymptotic Wilcoxon p-values from
scripts/06_analyze.py — for N >= 1000 and the effect sizes observed,
the two should agree, but with zero-inflated metrics it's worth
confirming the asymptotic approximation isn't biased.

Writes captions/permutation_test.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LANG_SUP = ("clip", "siglip")
SELF_SUP_HEADLINE = ("dinov2",)


def wilcoxon_r(diff: np.ndarray) -> float:
    nz = diff[diff != 0]
    if nz.size < 2:
        return float("nan")
    try:
        res = wilcoxon(nz, alternative="two-sided", zero_method="wilcox",
                       method="approx")
    except ValueError:
        return float("nan")
    if hasattr(res, "zstatistic") and res.zstatistic is not None:
        z = abs(float(res.zstatistic))
    else:
        from scipy.special import ndtri
        p = max(float(res.pvalue), 1e-300)
        z = abs(ndtri(1 - p / 2))
    return z / np.sqrt(nz.size)


def per_image_diff(df: pd.DataFrame, metric: str, decoder: str) -> np.ndarray:
    d = df[(df["metric"] == metric) & (df["decoder"] == decoder)]
    wide = d.pivot_table(index="image_id", columns="encoder",
                         values="score", aggfunc="mean")
    needed = list(LANG_SUP) + list(SELF_SUP_HEADLINE)
    wide = wide.dropna(subset=needed)
    lang_mean = wide[list(LANG_SUP)].mean(axis=1)
    self_mean = wide[list(SELF_SUP_HEADLINE)].mean(axis=1)
    return (lang_mean - self_mean).values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", nargs="+", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+",
                        default=["projective_density", "caption_length"])
    parser.add_argument("--decoders", nargs="+", default=["beam", "nucleus"])
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("captions/permutation_test.csv"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    by_seed: dict[int, pd.DataFrame] = {}
    for p in args.scores:
        df = pd.read_csv(p)
        s = int(df["seed"].iloc[0])
        by_seed[s] = df

    rows: list[dict] = []
    for decoder in args.decoders:
        for metric in args.metrics:
            for seed, df in sorted(by_seed.items()):
                diff = per_image_diff(df, metric, decoder)
                r_obs = wilcoxon_r(diff)
                # Sign-flip permutation: flip sign of each diff independently
                ge_count = 0
                for _ in range(args.n_perm):
                    signs = rng.choice([-1.0, 1.0], size=diff.size)
                    r_perm = wilcoxon_r(diff * signs)
                    if abs(r_perm) >= abs(r_obs):
                        ge_count += 1
                p_perm = (1 + ge_count) / (args.n_perm + 1)
                rows.append({
                    "decoder": decoder, "metric": metric, "seed": seed,
                    "r_obs": r_obs, "n_perm": args.n_perm,
                    "n_perm_ge": ge_count, "p_perm": p_perm,
                })
                print(f"{decoder:8s} {metric:20s} seed={seed} "
                      f"r={r_obs:.3f}  p_perm={p_perm:.5f}  "
                      f"(#ge={ge_count}/{args.n_perm})")

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
