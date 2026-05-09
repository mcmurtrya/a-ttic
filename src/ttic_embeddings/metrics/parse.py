"""spaCy parsing infrastructure with on-disk caching.

Parsing dominates evaluation runtime — parsing the full ~40K caption
sweep takes a few minutes; the four metric scorers afterward take
seconds. We cache parsed Docs to a DocBin (spaCy's purpose-built
serialization format) so we parse exactly once per caption set.

Defaults to en_core_web_lg per methods.md. The smaller `_sm` model
has less reliable dependency parses, which matters because
`adj_per_noun` and `head_noun_min_depth` rely on `amod` and `ROOT`
relations being correct.
"""
from __future__ import annotations

import logging
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.tokens import Doc, DocBin

logger = logging.getLogger(__name__)

_NLP: Language | None = None
_MODEL_NAME = "en_core_web_lg"


def get_nlp() -> Language:
    """Lazy-load the global spaCy pipeline."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(_MODEL_NAME)
    return _NLP


def parse_captions(
    captions: list[str],
    n_process: int = 1,
    batch_size: int = 256,
) -> list[Doc]:
    """Parse a list of captions through spaCy.

    n_process default is 1 because spaCy's multiprocessing has known
    rough edges on Windows + CPython 3.11+. Bumping to 4 cuts wall
    time roughly 3x on Linux/macOS; safe to try and fall back.
    """
    nlp = get_nlp()
    return list(nlp.pipe(captions, n_process=n_process, batch_size=batch_size))


def cache_parsed_docs(docs: list[Doc], path: Path) -> None:
    """Persist parsed Docs to a DocBin file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = DocBin(docs=docs, store_user_data=True)
    db.to_disk(path)


def load_parsed_docs(path: Path) -> list[Doc]:
    """Re-hydrate parsed Docs from a DocBin file."""
    nlp = get_nlp()
    db = DocBin().from_disk(path)
    return list(db.get_docs(nlp.vocab))


def parse_and_cache(
    captions: list[str],
    cache_path: Path,
    n_process: int = 1,
    force: bool = False,
) -> list[Doc]:
    """Parse + cache, or load existing cache if present.

    Set force=True to invalidate the cache and re-parse from scratch.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        logger.info("Loading cached parses from %s", cache_path)
        docs = load_parsed_docs(cache_path)
        if len(docs) == len(captions):
            return docs
        logger.warning(
            "Cache size mismatch (%d cached vs %d captions); re-parsing.",
            len(docs), len(captions),
        )
    logger.info("Parsing %d captions with spaCy...", len(captions))
    docs = parse_captions(captions, n_process=n_process)
    cache_parsed_docs(docs, cache_path)
    logger.info("Cached parses to %s", cache_path)
    return docs
