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
from ttic_embeddings.metrics.diversity import (                    # noqa: E402
    caption_length,
    mtld_from_docs,
)
from ttic_embeddings.metrics.specificity import (                   # noqa: E402
    load_vg_attributes,
    load_vg_to_coco_remap,
)
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


def score_diversity_cells(
    rows: list[dict],
    docs: list,
    n_bootstrap: int = 1000,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """One MTLD score per (encoder, decoder, seed) cell, with bootstrap 95% CI.

    Per methods.md L37, MTLD is unreliable below ~50 tokens, so we
    pool all captions in a cell before measuring. Bootstrap CI is
    over caption-resampling within the cell.
    """
    import numpy as np
    by_cell: dict[tuple, list] = {}
    for row, doc in zip(rows, docs):
        key = (row["encoder"], row["decoder"], row.get("seed", 0))
        by_cell.setdefault(key, []).append(doc)

    rng = np.random.default_rng(rng_seed)
    out: list[dict] = []
    for (encoder, decoder, seed), cell_docs in by_cell.items():
        n = len(cell_docs)
        point = mtld_from_docs(cell_docs)
        if n_bootstrap > 0 and n > 0:
            samples = np.empty(n_bootstrap, dtype=float)
            for b in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                samples[b] = mtld_from_docs([cell_docs[i] for i in idx])
            ci_lo, ci_hi = np.quantile(samples, [0.025, 0.975])
        else:
            ci_lo = ci_hi = float("nan")
        out.append({
            "encoder": encoder,
            "decoder": decoder,
            "seed": seed,
            "metric": "mtld",
            "score": point,
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "n_captions": n,
        })
    return pd.DataFrame(out)


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
    parser.add_argument("--vg-image-data", type=Path, default=None,
                        help="VG image_data.json (for VG↔COCO id remap). "
                             "Defaults to image_data.json beside --vg-attributes.")
    parser.add_argument("--n-process", type=int, default=1,
                        help="spaCy multiprocessing workers (default: 1).")
    parser.add_argument("--mtld-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for per-cell MTLD CI "
                             "(default: 1000; set to 0 to skip).")
    parser.add_argument("--mtld-bootstrap-seed", type=int, default=0,
                        help="Seed for MTLD bootstrap RNG (default: 0).")
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
        image_data_path = args.vg_image_data or (
            args.vg_attributes.parent / "image_data.json"
        )
        if not image_data_path.exists():
            log.error(
                "VG image_data.json not found at %s. "
                "VG attributes are keyed by VG image-ids; without "
                "image_data.json we cannot remap to COCO ids and every "
                "lookup would silently miss. Re-run scripts/01_download_data.py "
                "or pass --vg-image-data.",
                image_data_path,
            )
            return 1
        log.info("Building VG→COCO id remap from %s ...", image_data_path)
        remap = load_vg_to_coco_remap(image_data_path)
        log.info("VG→COCO remap covers %d images", len(remap))
        log.info("Loading VG attributes from %s ...", args.vg_attributes)
        vg_attrs = load_vg_attributes(args.vg_attributes, image_id_remap=remap)
        coco_ids_in_captions = {r["image_id"] for r in rows}
        overlap = len(coco_ids_in_captions & vg_attrs.keys())
        log.info(
            "Loaded VG attributes for %d images (%d overlap with caption rows)",
            len(vg_attrs), overlap,
        )
        if overlap == 0:
            log.error(
                "Zero overlap between caption image_ids and VG attribute keys "
                "after remap. The remap is not working — caption ids look like "
                "%s, VG keys look like %s.",
                list(coco_ids_in_captions)[:3],
                list(vg_attrs.keys())[:3],
            )
            return 1

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

    log.info(
        "Computing per-cell MTLD with %d bootstrap resamples...",
        args.mtld_bootstrap,
    )
    diversity_df = score_diversity_cells(
        rows, docs,
        n_bootstrap=args.mtld_bootstrap,
        rng_seed=args.mtld_bootstrap_seed,
    )
    diversity_path = output_path.with_name(
        output_path.stem + "_diversity.csv"
    )
    diversity_df.to_csv(diversity_path, index=False)
    log.info(
        "Wrote %d per-cell MTLD rows to %s",
        len(diversity_df), diversity_path,
    )

    summary = (
        df.dropna(subset=["score"])
          .groupby(["encoder", "metric"])["score"]
          .agg(["count", "mean", "std"])
    )
    log.info("Per-encoder/metric summary:\n%s", summary)
    log.info("Per-cell MTLD:\n%s", diversity_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
