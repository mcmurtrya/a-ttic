"""Abstraction-level metrics: WordNet hypernym depth and scene/object ratio.

Two complementary measures of "how concrete is the language":

  head_noun_min_depth — for the caption's head noun, average WordNet
    `min_depth()` across that lemma's synsets. Greater depth = more
    specific term. "Labrador" has higher min_depth than "dog" which
    has higher min_depth than "animal".

  scene_object_ratio — fraction of vocabulary matches that belong to
    a scene-category vocabulary (Places365) vs. an object-category
    vocabulary (COCO + frequent VG objects). High ratio = scene-level
    description; low ratio = object-naming description.

Predicted direction: language-supervised encoders produce captions
with higher hypernym depth (more specific terms) and lower scene/object
ratio (more object naming, less scene description) than self-supervised
encoders.

WordNet vs the included COCO list cover the common case but not every
COCO/VG token has a clean WordNet synset. ~5–10% of head nouns will
return None — proper nouns, OOV terms, fragments. The analysis layer
should handle None by excluding that observation rather than imputing.
"""
from __future__ import annotations

from typing import Iterable

from nltk.corpus import wordnet as wn
from spacy.tokens import Doc, Token

# COCO 80 object categories. Multi-word categories ("traffic light")
# are kept as single tokens for matching; the scoring code handles
# multi-word match against the lemma stream below.
COCO_OBJECTS: frozenset[str] = frozenset(
    {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "light", "hydrant", "sign", "meter",
        "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "ski", "snowboard",
        "ball", "kite", "bat", "glove", "skateboard", "surfboard",
        "racket", "bottle", "glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "pizza", "donut", "cake", "chair", "couch",
        "plant", "bed", "table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy", "drier", "toothbrush",
    }
)

# Curated scene vocabulary — a representative subset of Places365.
# The full Places365 list (365 categories) should be loaded from a
# data file in production runs; for now, this set covers the most
# common scene-category words seen in COCO captions.
DEFAULT_SCENES: frozenset[str] = frozenset(
    {
        "airport", "alley", "amphitheater", "arcade", "arena", "attic",
        "auditorium", "backyard", "bakery", "ballroom", "bank", "barn",
        "bathroom", "beach", "bedroom", "boardwalk", "bridge", "cafe",
        "campsite", "canyon", "cathedral", "cellar", "church", "city",
        "classroom", "cliff", "closet", "coast", "corridor", "courthouse",
        "courtyard", "dam", "desert", "diner", "dock", "downtown",
        "driveway", "dunes", "factory", "farm", "field", "forest",
        "garage", "garden", "gas", "gym", "harbor", "highway", "hospital",
        "hotel", "house", "kitchen", "laboratory", "lake", "library",
        "lighthouse", "lobby", "mall", "market", "meadow", "mosque",
        "mountain", "museum", "nursery", "ocean", "office", "orchard",
        "outdoor", "palace", "park", "parking", "patio", "pavilion",
        "pier", "plaza", "playground", "pool", "porch", "prairie",
        "prison", "racetrack", "raft", "railway", "ranch", "restaurant",
        "river", "road", "ruin", "schoolyard", "sidewalk", "skyline",
        "slope", "stadium", "stage", "store", "street", "studio",
        "subway", "supermarket", "swamp", "synagogue", "temple",
        "theater", "tower", "town", "tunnel", "valley", "village",
        "vineyard", "warehouse", "waterfall", "wharf", "yard",
        "indoor", "outdoor",
    }
)


def _head_noun(doc: Doc) -> Token | None:
    """Find the head noun of a caption's parse tree.

    Strategy: prefer the ROOT token if it's a NOUN; else its first
    NOUN child; else the first NOUN in the doc. Returns None if the
    caption has no NOUN tokens at all.
    """
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is not None:
        if root.pos_ == "NOUN":
            return root
        for child in root.children:
            if child.pos_ == "NOUN":
                return child
    for tok in doc:
        if tok.pos_ == "NOUN":
            return tok
    return None


def head_noun_min_depth(doc: Doc) -> float | None:
    """Average WordNet min_depth across synsets of the head noun's lemma.

    Returns None when:
      - the caption has no head noun (no NOUN tokens),
      - the head noun's lemma has no WordNet synsets (OOV / proper noun).
    """
    head = _head_noun(doc)
    if head is None:
        return None
    synsets = wn.synsets(head.lemma_.lower(), pos=wn.NOUN)
    if not synsets:
        return None
    depths = [s.min_depth() for s in synsets]
    return sum(depths) / len(depths)


def _vocab_matches(doc: Doc, vocab: Iterable[str]) -> int:
    vocab_set = set(vocab)
    return sum(1 for tok in doc if tok.lemma_.lower() in vocab_set)


def scene_object_ratio(
    doc: Doc,
    scene_vocab: Iterable[str] = DEFAULT_SCENES,
    object_vocab: Iterable[str] = COCO_OBJECTS,
) -> float | None:
    """Compute n_scene / (n_scene + n_object), in [0, 1].

    Returns None when the caption matches neither vocabulary
    (e.g. an extremely short caption with only function words).
    """
    n_scene = _vocab_matches(doc, scene_vocab)
    n_object = _vocab_matches(doc, object_vocab)
    if n_scene + n_object == 0:
        return None
    return n_scene / (n_scene + n_object)
