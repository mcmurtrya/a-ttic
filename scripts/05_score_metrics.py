"""Phase 5 — score the four caption-style metrics on generated captions.

Reads:
    {captions_path}                               JSONL from script 04

Writes:
    {captions_path with stem captions->scores}.csv
        long-format DataFrame ready for the stats module:
        image_id, encoder, decoder, seed, metric, score

Caches parsed spaCy Docs alongside the input JSONL via DocBin so
subsequent re-scoring (e.g. with a different VG annotation file)
doesn't re-parse from scratch.

Optional: --vg-attributes path/to/attributes.json enables
vg_attr_precision and vg_attr_recall metrics (per methods.md).
Without it, those metrics are simply not computed and downstream
analysis just sees fewer metric columns.

Usage:
    uv run python scripts/05_score_metrics.py \\
        --captions captions/captions_seed1.jsonl
    uv run python scripts/05_score_metrics.py \\
        --captions captions/captions_seed1.jsonl \\
        --vg-attributes data/vg/attributes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttic_embeddings.metrics import (                              # noqa: E402
    PROJECTIVE_LEXICON,
    TOPOLOGICAL_LEXICON,
    adj_per_noun,
    build_phrase_matcher,
    get_nlp,
    head_noun_min_depth,
    parse_and_cache,
    projective_density,
    scene_object_ratio,
    topological_density,
    vg_attribute_precision_recall,
)
from ttic_embeddings.metrics.diversity import caption_length        # noqa: E402
from ttic_embeddings.metrics.specificity import load_vg_attributes  # noqa: E402
from ttic_embeddings.utils import get_logger                        # noqa: E402

log = get_logger("score_metrics")


def load_captions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info("Loaded %d caption rows from %s", len(rows), path)
    return rows


def score_all(
    rows: list[dict],
    docs: list,
    topological_matcher,
    projective_matcher,
    vg_attrs: dict | None,
) -> pd.DataFrame:
    """One score row per (caption × metric). Long format."""
    out: list[dict] = []
    for row, doc in zip(rows, docs):
        base = {
            "image_id": row["image_id"],
            "encoder": row["encoder"],
            "decoder": row["decoder"],
            "seed": row.get("seed", 0),
        }
        out.extend([
            {**base, "metric": "adj_per_noun",
             "score": adj_per_noun(doc)},
            {**base, "metric": "topological_density",
             "score": topological_density(doc, topological_matcher)},
            {**base, "metric": "projective_density",
             "score": projective_density(doc, projective_matcher)},
            {**base, "metric": "head_noun_min_depth",
             "score": head_noun_min_depth(doc)},
            {**base, "metric": "scene_object_ratio",
             "score": scene_object_ratio(doc)},
            {**base, "metric": "caption_length",
             "score": caption_length(doc)},
        ])
        if vg_attrs is not None:
            p, r = vg_attribute_precision_recall(doc, vg_attrs, row["image_id"])
            out.append({**base, "metric": "vg_attr_precision", "score": p})
            out.append({**base, "metric": "vg_attr_recall", "score": r})
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--captions", required=True, type=Path,
                        help="Path to captions JSONL (output of script 04).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path. Default: alongside the input.")
    parser.add_argument("--vg-attributes", type=Path, default=None,
                        help="Optional VG attributes.json — enables vg_attr_* metrics.")
    parser.add_argument("--n-process", type=int, default=1,
                        help="spaCy multiprocessing workers (default: 1).")
    args = parser.parse_args()

    rows = load_captions(args.captions)
    if not rows:
        log.error("No caption rows. Did script 04 finish?")
        return 1

    captions_text = [r["caption"] for r in rows]
    parse_cache = args.captions.with_suffix(".spacy")
    docs = parse_and_cache(captions_text, parse_cache, n_process=args.n_process)
    log.info("Parsed %d captions (cache: %s)", len(docs), parse_cache)

    nlp = get_nlp()
    topological_matcher = build_phrase_matcher(nlp, TOPOLOGICAL_LEXICON)
    projective_matcher = build_phrase_matcher(nlp, PROJECTIVE_LEXICON)

    vg_attrs = None
    if args.vg_attributes is not None:
        if not args.vg_attributes.exists():
            log.error("VG attributes file not found: %s", args.vg_attributes)
            return 1
        log.info("Loading VG attributes from %s ...", args.vg_attributes)
        vg_attrs = load_vg_attributes(args.vg_attributes)
        log.info("Loaded VG attributes for %d images", len(vg_attrs))

    df = score_all(rows, docs, topological_matcher, projective_matcher, vg_attrs)

    if args.output is None:
        output_path = args.captions.with_name(
            args.captions.stem.replace("captions", "scores") + ".csv"
        )
    else:
        output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Wrote %d score rows to %s", len(df), output_path)

    summary = (
        df.dropna(subset=["score"])
          .groupby(["encoder", "metric"])["score"]
          .agg(["count", "mean", "std"])
    )
    log.info("Per-encoder/metric summary:\n%s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
