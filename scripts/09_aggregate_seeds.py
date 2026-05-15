"""Phase 5b — aggregate per-seed analysis outputs into a cross-seed summary.

Inputs:
    Per-seed analysis files written by scripts/06_analyze.py, e.g.
        captions/analysis_seed1_beam_primary.csv
        captions/analysis_seed2_beam_primary.csv
        captions/analysis_seed3_beam_primary.csv
    (and the parallel _secondary.csv and _mixed_effects.csv files).

Outputs:
    For each family (primary, secondary, mixed_effects), one CSV with
    one row per (metric[, encoder pair]) carrying:
        - per-seed values of the headline statistic (r for primary/
          secondary, coef for mixed_effects),
        - mean and SD across seeds,
        - n_meaningful_seeds (how many seeds called the row meaningful),
        - claimed_robust (True iff every seed called it meaningful).

    Plus an aggregate_<decoder><suffix>_summary.txt with a human-readable
    rendering and a flag block for any metric whose verdict is unstable
    across seeds (i.e. meaningful in some seeds but not others).

Usage:
    uv run python scripts/09_aggregate_seeds.py \\
        --analysis-dir captions/ --decoder beam --seeds 1,2,3
    uv run python scripts/09_aggregate_seeds.py \\
        --analysis-dir captions/ --decoder beam --suffix _no_mae
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttic_embeddings.utils import get_logger  # noqa: E402

log = get_logger("aggregate_seeds")

# Family-specific keys: which columns identify a row, and which is the
# headline statistic to pool across seeds.
FAMILIES = {
    "primary": {
        "key": ["metric", "contrast"],
        "stat": "r",
        "meaningful": "claimed_meaningful",
    },
    "secondary": {
        "key": ["metric", "encoder_a", "encoder_b", "contrast"],
        "stat": "r",
        "meaningful": "claimed_meaningful",
    },
    "mixed_effects": {
        "key": ["metric"],
        "stat": "coef",
        "meaningful": None,  # ME has no pre-registered meaningful flag
    },
}


def discover_seeds(
    analysis_dir: Path, decoder: str, suffix: str, family: str,
) -> dict[int, Path]:
    """Find all analysis_seedN_<decoder><suffix>_<family>.csv in the dir."""
    pattern = f"analysis_seed*_{decoder}{suffix}_{family}.csv"
    matches: dict[int, Path] = {}
    for path in sorted(analysis_dir.glob(pattern)):
        # Extract the seed number from the filename.
        stem = path.stem
        try:
            seed_token = stem.split("_")[1]  # "seed1"
            seed = int(seed_token.removeprefix("seed"))
        except (IndexError, ValueError):
            log.warning("Could not parse seed from %s; skipping", path.name)
            continue
        matches[seed] = path
    return matches


def aggregate_family(
    family: str, seed_paths: dict[int, Path],
) -> pd.DataFrame:
    """Pool one family's per-seed CSVs into a cross-seed summary."""
    spec = FAMILIES[family]
    key, stat, meaningful_col = spec["key"], spec["stat"], spec["meaningful"]

    per_seed: list[pd.DataFrame] = []
    for seed, path in sorted(seed_paths.items()):
        df = pd.read_csv(path)
        df = df.assign(seed=seed)
        per_seed.append(df)
    long = pd.concat(per_seed, ignore_index=True)

    seeds = sorted(seed_paths.keys())
    pivoted = long.pivot_table(
        index=key, columns="seed", values=stat, aggfunc="first",
    )
    pivoted.columns = [f"{stat}_seed{s}" for s in pivoted.columns]
    pivoted = pivoted.reset_index()

    seed_cols = [f"{stat}_seed{s}" for s in seeds]
    pivoted[f"mean_{stat}"] = pivoted[seed_cols].mean(axis=1)
    pivoted[f"sd_{stat}"] = pivoted[seed_cols].std(axis=1, ddof=1)

    if meaningful_col is not None:
        meaningful_pivot = long.pivot_table(
            index=key, columns="seed", values=meaningful_col, aggfunc="first",
        )
        n_meaningful = meaningful_pivot.fillna(False).astype(bool).sum(axis=1)
        pivoted["n_meaningful_seeds"] = n_meaningful.reindex(
            pivoted.set_index(key).index
        ).values
        pivoted["claimed_robust"] = pivoted["n_meaningful_seeds"] == len(seeds)
        pivoted["verdict_unstable"] = (
            (pivoted["n_meaningful_seeds"] > 0)
            & (pivoted["n_meaningful_seeds"] < len(seeds))
        )
    return pivoted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--analysis-dir", type=Path, default=Path("captions"))
    parser.add_argument("--decoder", default="beam", choices=["beam", "nucleus"])
    parser.add_argument("--suffix", default="",
                        help="Match the --output-suffix used in 06_analyze.py "
                             "(e.g. '_no_mae').")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds to require (default: use "
                             "whatever per-seed files are present).")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.analysis_dir
    requested_seeds = (
        {int(s.strip()) for s in args.seeds.split(",")}
        if args.seeds else None
    )

    summary_lines: list[str] = [
        f"# Cross-seed aggregate — decoder={args.decoder}, suffix={args.suffix or '(none)'}",
        "",
    ]

    for family in FAMILIES:
        seed_paths = discover_seeds(
            args.analysis_dir, args.decoder, args.suffix, family,
        )
        if requested_seeds is not None:
            missing = requested_seeds - set(seed_paths)
            if missing:
                log.warning(
                    "Family %s: requested seeds %s missing — found only %s. "
                    "Aggregating what's available.",
                    family, sorted(missing), sorted(seed_paths),
                )
            seed_paths = {s: p for s, p in seed_paths.items() if s in requested_seeds}
        if not seed_paths:
            log.warning("Family %s: no per-seed files found; skipping.", family)
            summary_lines.append(f"## {family}\n\n  (no per-seed inputs found)\n")
            continue
        log.info("Family %s: aggregating seeds %s",
                 family, sorted(seed_paths.keys()))
        agg = aggregate_family(family, seed_paths)

        out_path = output_dir / (
            f"analysis_aggregate_{args.decoder}{args.suffix}_{family}.csv"
        )
        agg.to_csv(out_path, index=False)
        log.info("Wrote -> %s", out_path)

        summary_lines.append(f"## {family} (seeds {sorted(seed_paths)})\n")
        summary_lines.append(agg.to_string(index=False))
        summary_lines.append("")
        if "verdict_unstable" in agg.columns:
            unstable = agg[agg["verdict_unstable"]]
            if not unstable.empty:
                summary_lines.append(
                    f"### Unstable verdicts in {family} "
                    f"(meaningful in some seeds but not others)\n"
                )
                summary_lines.append(unstable.to_string(index=False))
                summary_lines.append("")

    summary_path = output_dir / (
        f"analysis_aggregate_{args.decoder}{args.suffix}_summary.txt"
    )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    log.info("Wrote -> %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
