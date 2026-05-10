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

from ttic_embeddings.stats import (                                  # noqa: E402
    mixed_effects_supervision_contrast,
    primary_family,
    secondary_family,
)
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

    # --- Mixed-effects robustness check (methods.md L49) --------------
    log.info(
        "\nRunning mixed-effects robustness check on primary contrasts (%d metrics)...",
        len(metrics),
    )
    mixed_rows: list[dict] = []
    for metric in metrics:
        per_metric_long = df_dec[df_dec["metric"] == metric].rename(
            columns={"score": metric}
        )
        try:
            row = mixed_effects_supervision_contrast(per_metric_long, metric)
        except Exception as exc:  # noqa: BLE001
            log.warning("mixed-effects failed for %s: %s", metric, exc)
            row = {"metric": metric, "coef": None, "se": None,
                   "z": None, "p_value": None, "error": str(exc)}
        mixed_rows.append(row)
    mixed = pd.DataFrame(mixed_rows)
    mixed_path = output_dir / f"{base}{suffix}_mixed_effects.csv"
    mixed.to_csv(mixed_path, index=False)
    log.info("Mixed-effects results:\n%s", mixed.to_string(index=False))
    log.info("Wrote -> %s", mixed_path)

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
        f.write("\n\n## Mixed-effects robustness (methods.md L49)\n\n")
        f.write(
            "score ~ supervision + (1 | encoder). Agreement with the pooled\n"
            "Wilcoxon primary supports the supervision claim; disagreement\n"
            "points to one encoder driving most of its category's effect.\n\n"
        )
        f.write(mixed.to_string(index=False))
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

        # Per-cell MTLD: by design (methods.md L37) MTLD is computed per
        # (encoder × decoder) cell rather than per caption, since per-caption
        # MTLD is unreliable below ~50 tokens. Cross-encoder comparison is via
        # bootstrap CI overlap rather than paired Wilcoxon.
        diversity_path = args.scores.with_name(
            args.scores.stem + "_diversity.csv"
        )
        if diversity_path.exists():
            diversity = pd.read_csv(diversity_path)
            diversity_dec = diversity[diversity["decoder"] == args.decoder]
            f.write("\n\n## Lexical diversity (MTLD, per-cell with bootstrap 95% CI)\n\n")
            f.write(
                "MTLD pooled over all captions in each (encoder, decoder, seed) "
                "cell. Two cells differ meaningfully when their bootstrap CIs "
                "do not overlap.\n\n"
            )
            f.write(diversity_dec.to_string(index=False))
            log.info("MTLD per-cell:\n%s", diversity_dec.to_string(index=False))
        else:
            log.warning(
                "MTLD per-cell file not found at %s — re-run script 05 to "
                "produce it.", diversity_path,
            )
    log.info("Wrote -> %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
