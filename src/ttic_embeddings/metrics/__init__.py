"""Caption-style metrics, organized by methods.md axis.

Four metric modules, one per axis from methods.md:

  specificity   — adjectives/noun, VG attribute precision/recall
  spatial       — topological + projective lexicon density
  abstraction   — WordNet hypernym depth, scene/object ratio
  diversity     — MTLD, mean caption length

Plus parse.py with shared spaCy infrastructure (cached parsing via DocBin).

All metric functions take a parsed spaCy `Doc` and return a per-caption
score (or None if the metric is undefined for that caption — e.g.
hypernym depth on a caption with no recognizable head noun).
"""
from __future__ import annotations

from .abstraction import head_noun_min_depth, scene_object_ratio
from .diversity import mean_caption_length, mtld
from .parse import get_nlp, load_parsed_docs, parse_and_cache, parse_captions
from .spatial import (
    PROJECTIVE_LEXICON,
    TOPOLOGICAL_LEXICON,
    build_phrase_matcher,
    projective_density,
    topological_density,
)
from .specificity import adj_per_noun, vg_attribute_precision_recall

__all__ = [
    "PROJECTIVE_LEXICON",
    "TOPOLOGICAL_LEXICON",
    "adj_per_noun",
    "build_phrase_matcher",
    "get_nlp",
    "head_noun_min_depth",
    "load_parsed_docs",
    "mean_caption_length",
    "mtld",
    "parse_and_cache",
    "parse_captions",
    "projective_density",
    "scene_object_ratio",
    "topological_density",
    "vg_attribute_precision_recall",
]
