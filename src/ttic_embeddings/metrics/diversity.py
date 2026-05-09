"""Lexical diversity and length metrics.

Methods.md commits to MTLD over raw type-token ratio because TTR is
heavily length-confounded (short documents look more "diverse" by
construction). MTLD is length-robust within the size regime where
it was validated; below ~50 tokens it becomes unreliable.

Implication: per-caption MTLD is noise. We compute MTLD over
*concatenated* tokens within an (encoder × decoder) cell — pooling
all 5K val captions for one condition gives a stable diversity
estimate for that condition. Cross-encoder comparison is then via
bootstrap resampling at the cell level rather than paired Wilcoxon
across images (since MTLD is no longer per-image).

Caption length is straightforward and is reported as a per-caption
score for paired analysis.
"""
from __future__ import annotations

from typing import Iterable

from lexical_diversity import lex_div as ld
from spacy.tokens import Doc


def mtld(text_or_tokens: str | list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity.

    Accepts either a string (split on whitespace) or a list of tokens.
    The default threshold of 0.72 matches the original McCarthy & Jarvis
    formulation and is what the `lexical-diversity` package uses by
    default.
    """
    if isinstance(text_or_tokens, str):
        tokens = text_or_tokens.split()
    else:
        tokens = list(text_or_tokens)
    if not tokens:
        return 0.0
    return ld.mtld(tokens)


def mtld_from_docs(docs: Iterable[Doc]) -> float:
    """MTLD over the concatenation of all tokens across multiple Docs.

    Use this to compute one MTLD score per (encoder × decoder) cell
    over all captions in that cell. Lemmas are lowercased to count
    morphological variants together.
    """
    tokens: list[str] = []
    for doc in docs:
        tokens.extend(tok.lemma_.lower() for tok in doc if not tok.is_punct)
    return mtld(tokens)


def caption_length(doc: Doc) -> int:
    """Token count of a single caption (excluding punctuation)."""
    return sum(1 for tok in doc if not tok.is_punct)


def mean_caption_length(docs: Iterable[Doc]) -> float:
    docs = list(docs)
    if not docs:
        return 0.0
    return sum(caption_length(d) for d in docs) / len(docs)
