"""Semantic specificity metrics: adjectives-per-noun and VG attribute P/R.

Methods.md predicts that language-supervised encoders produce
captions with higher adjective density and higher attribute precision
against Visual Genome ground truth — i.e. they bias generation toward
"a small red car next to a tall blue building" over "a vehicle near a
structure."

Two metrics:

  adj_per_noun(doc) — count of `amod` adjective dependents per noun.
    Restricted to attributive `amod` to exclude predicative
    adjectives ("the dog is brown") which are syntactically distinct
    and carry different information.

  vg_attribute_precision_recall(doc, vg_attrs, image_id) — set
    overlap of caption adjectives against VG ground-truth attributes.
    Stronger than adj/noun because it's grounded in actual visual
    properties; weaker in that VG coverage is partial (~3.5K of
    5K val images).
"""
from __future__ import annotations

import json
from pathlib import Path

from spacy.tokens import Doc


def adj_per_noun(doc: Doc) -> float:
    """Average attributive adjectives per noun in the caption.

    Returns 0.0 for captions with no nouns (rare, but possible —
    "Beautiful." with no other content has no noun head).
    """
    nouns = [t for t in doc if t.pos_ == "NOUN"]
    if not nouns:
        return 0.0
    n_amod = sum(
        1
        for noun in nouns
        for child in noun.children
        if child.dep_ == "amod" and child.pos_ == "ADJ"
    )
    return n_amod / len(nouns)


def caption_adjectives(doc: Doc) -> set[str]:
    """All adjective lemmas in a caption, lowercased."""
    return {tok.lemma_.lower() for tok in doc if tok.pos_ == "ADJ"}


def vg_attribute_precision_recall(
    doc: Doc,
    vg_attrs: dict[int, set[str]],
    image_id: int,
) -> tuple[float | None, float | None]:
    """Set-overlap precision and recall against VG ground-truth attributes.

    Args:
        doc: parsed caption.
        vg_attrs: image_id -> set of ground-truth attribute strings
            (already lowercased and lemmatized by the loader).
        image_id: COCO image id; lookup key into vg_attrs.

    Returns:
        (precision, recall). Either may be None when the metric is
        undefined for this caption/image combination — e.g. precision
        is undefined when the caption contains no adjectives, recall
        is undefined when the VG image has no attributes. Both None
        when the image has no VG entry at all.
    """
    if image_id not in vg_attrs:
        return None, None
    cap_attrs = caption_adjectives(doc)
    gt_attrs = vg_attrs[image_id]

    precision: float | None
    recall: float | None
    inter = cap_attrs & gt_attrs

    precision = len(inter) / len(cap_attrs) if cap_attrs else None
    recall = len(inter) / len(gt_attrs) if gt_attrs else None
    return precision, recall


def load_vg_attributes(
    attributes_json_path: Path,
    image_id_remap: dict[int, int] | None = None,
) -> dict[int, set[str]]:
    """Load Visual Genome attribute annotations into a dict.

    Args:
        attributes_json_path: path to VG's attributes.json (unzipped
            output of scripts/01_download_data.py).
        image_id_remap: optional VG-id -> COCO-id mapping. VG and
            COCO use different image-id namespaces; the caller is
            responsible for joining via image_data.json. If None, the
            returned dict uses VG image ids as keys.

    Returns:
        Dict mapping image_id -> set[str] of ground-truth attributes,
        lowercased.
    """
    with open(attributes_json_path) as f:
        records = json.load(f)
    out: dict[int, set[str]] = {}
    for rec in records:
        img_id = rec.get("image_id")
        if img_id is None:
            continue
        attrs: set[str] = set()
        for obj in rec.get("attributes", []):
            for a in obj.get("attributes", []) or []:
                attrs.add(a.lower())
        if image_id_remap is not None:
            img_id = image_id_remap.get(img_id, img_id)
        if attrs:
            out[img_id] = out.get(img_id, set()) | attrs
    return out
