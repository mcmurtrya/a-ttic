"""Shared pytest fixtures.

Loads expensive resources (spaCy pipeline, WordNet, PhraseMatchers)
once per test session. Tests that depend on optional assets skip
gracefully if those assets aren't installed, so `make test` runs
on a fresh checkout without requiring `make install-dev` first.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def nlp():
    """The shared spaCy pipeline (en_core_web_lg)."""
    try:
        from ttic_embeddings.metrics.parse import get_nlp
        return get_nlp()
    except (OSError, ImportError) as e:
        pytest.skip(f"spaCy en_core_web_lg not installed: {e}")


@pytest.fixture(scope="session")
def topological_matcher(nlp):
    from ttic_embeddings.metrics.spatial import (
        TOPOLOGICAL_LEXICON,
        build_phrase_matcher,
    )
    return build_phrase_matcher(nlp, TOPOLOGICAL_LEXICON)


@pytest.fixture(scope="session")
def projective_matcher(nlp):
    from ttic_embeddings.metrics.spatial import (
        PROJECTIVE_LEXICON,
        build_phrase_matcher,
    )
    return build_phrase_matcher(nlp, PROJECTIVE_LEXICON)


@pytest.fixture(scope="session")
def wordnet():
    """NLTK's WordNet corpus, as a smoke-checked module reference."""
    try:
        from nltk.corpus import wordnet as wn
        # Trigger the actual data load — ImportError-free imports do not
        # guarantee the corpus is downloaded.
        wn.synsets("dog")
        return wn
    except (ImportError, LookupError) as e:
        pytest.skip(f"NLTK wordnet corpus not available: {e}")
