"""Phase 5 — statistical analysis on metric scores.

Reads:
    {scores_path}                                    long-format CSV from script 05

Writes:
    {scores_path with stem scores->analysis}_primary.csv
        4 supervision-category contrasts with BH-FDR (the headline test).
    {scores_path with stem scores->analysis}_secondary.csv
        24 pairwise contrasts with BH-FDR within this family separately.
        Includes a `within_category` boolean for the 8 sanity-check tests.
    {scores_path with stem scores->analysis}_summary.txt
        Human-readable rendering of both families plus a within-category
        diagnostic block.

Per methods.md, the headline criterion is `claimed_meaningful` in the
output: True when both `p_adj <= alpha` AND Wilcoxon `r >= 0.10`.

Usage:
    uv run python scripts/06_analyze.py --scores captions/scores_seed1.csv
    uv run python scripts/06_analyze.py --scores captions/scores_seed1.csv \\
        --decoder nucleus --metrics adj_per_noun,projective_density
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttic_embeddings.stats import primary_family, secondary_family   # noqa: E402
from ttic_embeddings.utils import get_logger                         # noqa: E402

log = get_logger("analyze")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", required=True, type=Path,
                        help="Long-format scores CSV (output of script 05).")
    parser.add_argument("--decoder", default="beam", choices=["beam", "nucleus"],
                        help="Which decoding strategy to analyze (default: beam).")
    parser.add_argument("--metrics", default=None,
                        help="Comma-separated metric names. Default: all in the file.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--effect-size-floor", type=float, default=0.10,
                        help="Pre-registered Wilcoxon r threshold for "
                             "claimed_meaningful (default: 0.10).")
    args = parser.parse_args()

    df = pd.read_csv(args.scores)
    log.info("Loaded %d score rows from %s", len(df), args.scores)
    log.info("Decoders present: %s", sorted(df["decoder"].unique()))
    log.info("Encoders present: %s", sorted(df["encoder"].unique()))
    log.info("Metrics present:  %s", sorted(df["metric"].unique()))

    df_dec = df[df["decoder"] == args.decoder].copy()
    log.info("Filtered to decoder=%s: %d rows", args.decoder, len(df_dec))
    if df_dec.empty:
        log.error("No rows for decoder=%s; check the input file.", args.decoder)
        return 1

    metrics = (
        [m.strip() for m in args.metrics.split(",")]
        if args.metrics else sorted(df_dec["metric"].unique())
    )
    log.info("Analyzing %d metrics: %s", len(metrics), metrics)

    # The stats functions expect (image_id, encoder) -> per-metric column shape.
    # Pivot from long to mixed: image_id and encoder as index, metric as columns.
    wide = df_dec.pivot_table(
        index=["image_id", "encoder"],
        columns="metric",
        values="score",
        aggfunc="mean",
    ).reset_index()

    output_dir = args.output_dir or args.scores.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base = args.scores.stem.replace("scores", "analysis")
    suffix = f"_{args.decoder}"

    # --- Primary family -----------------------------------------------
    log.info("\nRunning primary family (supervision-category contrast, %d tests)...",
             len(metrics))
    primary = primary_family(
        wide, metrics,
        alpha=args.alpha, effect_size_floor=args.effect_size_floor,
    )
    primary_path = output_dir / f"{base}{suffix}_primary.csv"
    primary.to_csv(primary_path, index=False)
    log.info("Primary results:\n%s", primary.to_string(index=False))
    log.info("Wrote -> %s", primary_path)

    # --- Secondary family ---------------------------------------------
    log.info("\nRunning secondary family (pairwise contrasts, %d tests)...",
             6 * len(metrics))
    secondary = secondary_family(
        wide, metrics,
        alpha=args.alpha, effect_size_floor=args.effect_size_floor,
    )
    secondary_path = output_dir / f"{base}{suffix}_secondary.csv"
    secondary.to_csv(secondary_path, index=False)
    log.info("Wrote -> %s", secondary_path)

    # --- Summary text -------------------------------------------------
    summary_path = output_dir / f"{base}{suffix}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Analysis — {args.scores.name}, decoder={args.decoder}\n\n")
        f.write(f"alpha = {args.alpha}\n")
        f.write(f"effect_size_floor (Wilcoxon r) = {args.effect_size_floor}\n\n")
        f.write("## Primary family (supervision-category contrast)\n\n")
        f.write("Pools (CLIP, SigLIP) vs (DINOv2, MAE) per image, per metric.\n")
        f.write("BH-FDR corrected within this 4-test family.\n\n")
        f.write(primary.to_string(index=False))
        f.write("\n\n## Secondary family (all pairwise contrasts)\n\n")
        f.write("6 encoder pairs * %d metrics = %d tests, "
                "BH-FDR corrected within this family.\n\n"
                % (len(metrics), 6 * len(metrics)))
        f.write(secondary.to_string(index=False))
        f.write("\n\n## Within-category sanity checks (subset of secondary)\n\n")
        f.write("CLIP vs SigLIP and DINOv2 vs MAE per metric. Should be null\n")
        f.write("under the supervision-category claim; large within-category\n")
        f.write("differences mean one encoder drives that category's effect.\n\n")
        within = secondary[secondary["within_category"]]
        f.write(within.to_string(index=False))
    log.info("Wrote -> %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
