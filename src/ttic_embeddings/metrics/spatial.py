"""Spatial-language metrics: topological vs projective term density.

Two lexicons, distinguished per the linguistics of spatial cognition:

  Topological — containment/contact prepositions. Nearly unavoidable
    English filler ("on the table", "in the bag"); occurs even when
    the speaker isn't reasoning spatially. We expect topological
    density to be roughly comparable across encoders.

  Projective — directional / metric / frame-of-reference language.
    Requires the speaker to encode geometric relations between
    objects ("left of", "behind", "next to"). The hypothesis lives
    here: self-supervised encoders should bias generation toward
    higher projective density because they preserve more of the
    spatial layout of the scene.

Both metrics: count phrase occurrences in the caption, normalized by
total token count (not caption count) to control for length effects.
Multi-word phrases ("in front of") are matched via spaCy's
PhraseMatcher on the LEMMA attribute.
"""
from __future__ import annotations

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

# Topological prepositions (containment/contact). Pre-registered in
# methods.md L33 — do not silently expand without updating the doc,
# since the test statistic is sensitive to the lexicon.
TOPOLOGICAL_LEXICON: list[str] = [
    "in",
    "on",
    "at",
    "inside",
    "outside",
    "with",
]

# Projective phrases (directional/metric). Pre-registered in methods.md
# L33. Multi-word phrases ("left of") match via PhraseMatcher on lemma;
# the matcher catches "to the left of" via the embedded "left of" span,
# so spelled-out variants do not need separate entries.
PROJECTIVE_LEXICON: list[str] = [
    "left of",
    "right of",
    "behind",
    "in front of",
    "above",
    "below",
    "next to",
    "between",
    "near",
]


def build_phrase_matcher(nlp: Language, phrases: list[str]) -> PhraseMatcher:
    """Construct a PhraseMatcher matching on lemma.

    Uses `nlp.pipe()` (full pipeline including lemmatizer) rather than
    `nlp.make_doc()` (tokenizer only). The PhraseMatcher with
    attr="LEMMA" requires its patterns to have lemma annotations, and
    those are only produced by the lemmatizer component.
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    patterns = list(nlp.pipe(phrases))
    matcher.add("SPATIAL", patterns)
    return matcher


def _term_count(doc: Doc, matcher: PhraseMatcher) -> int:
    return len(matcher(doc))


def _density(doc: Doc, matcher: PhraseMatcher) -> float:
    """Term count divided by total token count (excluding zero-length docs)."""
    if len(doc) == 0:
        return 0.0
    return _term_count(doc, matcher) / len(doc)


def topological_density(doc: Doc, matcher: PhraseMatcher) -> float:
    """Topological-term density (matches per token).

    The matcher must have been built from TOPOLOGICAL_LEXICON.
    Defensive: this function does not enforce that — it just counts
    matches from whatever matcher you passed in.
    """
    return _density(doc, matcher)


def projective_density(doc: Doc, matcher: PhraseMatcher) -> float:
    """Projective-term density (matches per token).

    The matcher must have been built from PROJECTIVE_LEXICON.
    """
    return _density(doc, matcher)
