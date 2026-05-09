"""Phase 5 — captioning-quality precondition (CIDEr, optionally SPICE).

For each (encoder, decoder), compute CIDEr against the COCO reference
captions. Per methods.md, this is a precondition gate: all four
encoders must produce captions whose CIDEr scores fall within ~10% of
one another before we claim style differences from the four caption-style
metrics. If one encoder is dramatically worse on reference-based quality,
the style differences are confounded with quality — that encoder might
just be a worse captioner, not a stylistically different one.

CIDEr is order-invariant n-gram TF-IDF overlap, well-supported on every
platform. SPICE additionally measures scene-graph overlap and is more
diagnostic of what the caption actually says about the image, but it
has a Java dependency (Stanford Scene Graph Parser); we attempt it and
fall back gracefully if Java/JDK isn't installed.

Tokenization: pycocoevalcap's PTBTokenizer is also Java-based. We try
it first for reproducibility with literature CIDEr scores, and fall
back to a simple regex tokenizer otherwise. The fallback is sufficient
for cross-encoder COMPARISON within this project (which is what the
precondition check needs) but not for absolute reproducibility with
external CIDEr numbers.

Reads:
    {captions_path}                                    JSONL from script 04
    $COCO_ROOT/annotations/captions_val2017.json      reference captions

Writes:
    {output_dir}/quality_results.csv
        long-format: encoder, decoder, metric, score, n_images
    {output_dir}/quality_summary.txt
        per-encoder table + 10% precondition check (pass/fail)

Usage:
    uv run python scripts/08_caption_quality.py \\
        --captions captions/captions_seed1.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttic_embeddings.utils import get_logger  # noqa: E402

log = get_logger("caption_quality")


# ---------------------------------------------------------------------
# Lazy imports — fail with a clear message if extras aren't installed
# ---------------------------------------------------------------------


def _import_cider():
    try:
        from pycocoevalcap.cider.cider import Cider
        return Cider
    except ImportError as e:
        raise SystemExit(
            "pycocoevalcap not installed. Run `make install-dev` to install\n"
            "the caption-quality extras (pycocoevalcap + pycocotools).\n"
            f"Original error: {e}"
        )


def _try_import_spice():
    """Return Spice class or None if SPICE/Java is unavailable."""
    try:
        from pycocoevalcap.spice.spice import Spice
        return Spice
    except Exception as e:
        log.warning("SPICE unavailable (%s); skipping.", type(e).__name__)
        return None


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_generated_captions(path: Path) -> dict[tuple[str, str], dict[int, list[str]]]:
    """Load JSONL into {(encoder, decoder): {image_id: [caption]}}.

    pycocoevalcap expects each image's hypothesis to be a single-element
    list — preserves the same shape as the multi-reference ground truth.
    """
    out: dict[tuple[str, str], dict[int, list[str]]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["encoder"], row["decoder"])
            out[key][int(row["image_id"])] = [row["caption"]]
    return dict(out)


def load_reference_captions(coco_root: Path, split: str = "val") -> dict[int, list[str]]:
    """Load 5-per-image reference captions from COCO."""
    captions_path = coco_root / "annotations" / f"captions_{split}2017.json"
    if not captions_path.exists():
        raise FileNotFoundError(
            f"COCO captions JSON not found at {captions_path}. "
            f"Run `make data` first."
        )
    with open(captions_path) as f:
        data = json.load(f)
    refs: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        refs[int(ann["image_id"])].append(ann["caption"].strip())
    return dict(refs)


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------


_REGEX_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def _regex_tokenize(captions_dict: dict[int, list[str]]) -> dict[int, list[str]]:
    """Lowercased word-level tokenization. Java-free fallback."""
    return {
        k: [" ".join(_REGEX_TOKEN.findall(c.lower())) for c in caps]
        for k, caps in captions_dict.items()
    }


def tokenize(captions_dict: dict[int, list[str]], try_java: bool = True) -> dict[int, list[str]]:
    """Tokenize captions for CIDEr/SPICE.

    Tries pycocoevalcap's PTBTokenizer (Java-based, canonical) and falls
    back to a regex tokenizer if Java is not available.
    """
    if try_java:
        try:
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
            wrapped = {
                k: [{"caption": c} for c in caps]
                for k, caps in captions_dict.items()
            }
            return PTBTokenizer().tokenize(wrapped)
        except Exception as e:
            log.warning(
                "PTBTokenizer failed (%s); falling back to regex tokenizer. "
                "Cross-encoder comparisons remain valid; absolute scores will "
                "differ slightly from literature CIDEr numbers.",
                type(e).__name__,
            )
    return _regex_tokenize(captions_dict)


# ---------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------


def evaluate_condition(
    hypotheses: dict[int, list[str]],
    references: dict[int, list[str]],
    spice_cls: Any | None,
    use_java_tokenizer: bool,
) -> dict[str, Any]:
    """Compute CIDEr (and SPICE if available) for one (encoder, decoder)."""
    Cider = _import_cider()

    # Restrict to the intersection of image ids present in both
    common = sorted(set(hypotheses) & set(references))
    if not common:
        return {"cider": float("nan"), "spice": None, "n_images": 0}

    gts_raw = {iid: references[iid] for iid in common}
    res_raw = {iid: hypotheses[iid] for iid in common}

    gts = tokenize(gts_raw, try_java=use_java_tokenizer)
    res = tokenize(res_raw, try_java=use_java_tokenizer)

    cider_score, _ = Cider().compute_score(gts, res)

    spice_score: float | None = None
    if spice_cls is not None:
        try:
            spice_score, _ = spice_cls().compute_score(gts, res)
            spice_score = float(spice_score)
        except Exception as e:
            log.warning("SPICE compute failed: %s: %s", type(e).__name__, e)

    return {
        "cider": float(cider_score),
        "spice": spice_score,
        "n_images": len(common),
    }


# ---------------------------------------------------------------------
# Precondition check
# ---------------------------------------------------------------------


def check_cider_precondition(
    df: pd.DataFrame,
    threshold: float = 0.10,
    decoder: str = "beam",
) -> dict:
    """Verify all encoders are within `threshold` (fraction) of the max CIDEr."""
    cider = df[(df["metric"] == "cider") & (df["decoder"] == decoder)]
    cider = cider.dropna(subset=["score"])
    if cider.empty or len(cider) < 2:
        return {
            "passed": None,
            "reason": f"need >=2 encoders for decoder={decoder}, "
                      f"got {len(cider)}",
            "n_encoders": len(cider),
        }
    max_score = float(cider["score"].max())
    min_score = float(cider["score"].min())
    if max_score <= 0:
        return {"passed": None, "reason": "max CIDEr <= 0", "max": max_score, "min": min_score}
    spread = (max_score - min_score) / max_score
    return {
        "passed": bool(spread <= threshold),
        "max": max_score,
        "min": min_score,
        "spread_fraction": float(spread),
        "threshold": float(threshold),
        "decoder": decoder,
        "encoders": sorted(cider["encoder"].unique().tolist()),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--captions", required=True, type=Path,
                        help="JSONL of generated captions (output of script 04).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: alongside the captions file.")
    parser.add_argument("--coco-root", type=Path, default=None,
                        help="Override $COCO_ROOT (where annotations/ lives).")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Precondition spread threshold (default 0.10 = 10%%).")
    parser.add_argument("--no-spice", action="store_true",
                        help="Skip SPICE even if available.")
    parser.add_argument("--no-java-tokenizer", action="store_true",
                        help="Skip PTBTokenizer attempt; go straight to regex.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = OmegaConf.load(repo_root / "configs" / "base.yaml")
    coco_root = (
        Path(args.coco_root)
        if args.coco_root is not None
        else Path(base_cfg.paths.coco_root)
    )

    log.info("Loading generated captions: %s", args.captions)
    by_condition = load_generated_captions(args.captions)
    log.info("Loaded %d (encoder, decoder) conditions", len(by_condition))
    for k, v in by_condition.items():
        log.info("  %s: %d images", k, len(v))

    log.info("Loading reference captions from %s ...", coco_root)
    references = load_reference_captions(coco_root, split="val")
    log.info("Loaded references for %d images", len(references))

    # Try to load SPICE once (not per-condition)
    spice_cls = None if args.no_spice else _try_import_spice()
    use_java_tokenizer = not args.no_java_tokenizer

    rows: list[dict] = []
    for (encoder, decoder), hypotheses in by_condition.items():
        log.info("--- %s / %s ---", encoder, decoder)
        result = evaluate_condition(
            hypotheses, references,
            spice_cls=spice_cls,
            use_java_tokenizer=use_java_tokenizer,
        )
        log.info("  CIDEr=%.4f  SPICE=%s  n_images=%d",
                 result["cider"],
                 f"{result['spice']:.4f}" if result["spice"] is not None else "n/a",
                 result["n_images"])
        rows.append({
            "encoder": encoder, "decoder": decoder,
            "metric": "cider", "score": result["cider"],
            "n_images": result["n_images"],
        })
        if result["spice"] is not None:
            rows.append({
                "encoder": encoder, "decoder": decoder,
                "metric": "spice", "score": result["spice"],
                "n_images": result["n_images"],
            })

    if not rows:
        log.error("No conditions evaluated.")
        return 1

    df = pd.DataFrame(rows)
    output_dir = args.output_dir or args.captions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quality_results.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %d quality rows to %s", len(df), csv_path)

    # Precondition check on the headline decoder (beam)
    precondition = check_cider_precondition(
        df, threshold=args.threshold, decoder="beam",
    )

    # Summary text
    summary_path = output_dir / "quality_summary.txt"
    pivot = df.pivot_table(index=["encoder", "decoder"], columns="metric", values="score")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Caption-quality precondition — {args.captions.name}\n\n")
        f.write("## Per-condition scores\n\n")
        f.write(pivot.to_string())
        f.write("\n\n## Precondition check (CIDEr, beam decoder)\n\n")
        f.write(f"threshold: {args.threshold:.2%} of max CIDEr\n\n")
        for k, v in precondition.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        if precondition.get("passed") is True:
            f.write("PRECONDITION PASSED — encoders are within tolerance on CIDEr.\n")
            f.write("Style differences in scripts 05/06 can be claimed.\n")
        elif precondition.get("passed") is False:
            f.write("PRECONDITION FAILED — encoders differ on CIDEr by more than the\n")
            f.write("threshold. Style claims should be scoped to 'comparable-quality'\n")
            f.write("encoders only, or the worst encoder excluded from the headline.\n")
        else:
            f.write("PRECONDITION INDETERMINATE — too few encoders to evaluate.\n")
            f.write(f"Reason: {precondition.get('reason')}\n")

    log.info("Wrote summary to %s", summary_path)
    log.info("Per-condition scores:\n%s", pivot.to_string())
    log.info("Precondition: %s", precondition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
