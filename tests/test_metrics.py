"""Hand-crafted caption fixtures with known scores.

These tests are the cheapest defense against silent metric bugs. Each
test uses a caption whose expected score we computed by hand (or whose
relative ordering against another caption is unambiguous).

Where exact integer counts are robust across spaCy versions, we assert
exact values. Where parser behavior could shift slightly, we assert
relative ordering or sign instead.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------
# Specificity: adj_per_noun, caption_adjectives
# ---------------------------------------------------------------------


class TestAdjPerNoun:
    def test_one_adj_one_noun(self, nlp):
        from ttic_embeddings.metrics.specificity import adj_per_noun
        doc = nlp("A red car.")
        # nouns: car (1); amod children of car: red (1)
        assert adj_per_noun(doc) == 1.0

    def test_no_adjectives(self, nlp):
        from ttic_embeddings.metrics.specificity import adj_per_noun
        doc = nlp("A car.")
        assert adj_per_noun(doc) == 0.0

    def test_two_pairs(self, nlp):
        from ttic_embeddings.metrics.specificity import adj_per_noun
        doc = nlp("A red car and a blue truck.")
        # nouns: car, truck (2); amods: red on car, blue on truck (2)
        assert adj_per_noun(doc) == 1.0

    def test_predicative_adjective_excluded(self, nlp):
        """Predicative `is brown` is `acomp`, not `amod`, and must not count."""
        from ttic_embeddings.metrics.specificity import adj_per_noun
        doc = nlp("The dog is brown.")
        assert adj_per_noun(doc) == 0.0

    def test_no_nouns(self, nlp):
        from ttic_embeddings.metrics.specificity import adj_per_noun
        doc = nlp("Beautiful.")  # adjective only, no noun head
        assert adj_per_noun(doc) == 0.0

    def test_empty_caption(self, nlp):
        from ttic_embeddings.metrics.specificity import adj_per_noun
        doc = nlp("")
        assert adj_per_noun(doc) == 0.0


class TestCaptionAdjectives:
    def test_extracts_lemmas(self, nlp):
        from ttic_embeddings.metrics.specificity import caption_adjectives
        doc = nlp("A small red car.")
        assert caption_adjectives(doc) == {"small", "red"}

    def test_lowercased(self, nlp):
        from ttic_embeddings.metrics.specificity import caption_adjectives
        doc = nlp("RED car.")  # parsed as ADJ noun
        assert "red" in caption_adjectives(doc)


class TestVgAttributePR:
    def test_returns_none_when_image_missing(self, nlp):
        from ttic_embeddings.metrics.specificity import (
            vg_attribute_precision_recall,
        )
        doc = nlp("A red car.")
        p, r = vg_attribute_precision_recall(doc, vg_attrs={}, image_id=42)
        assert p is None and r is None

    def test_perfect_match(self, nlp):
        from ttic_embeddings.metrics.specificity import (
            vg_attribute_precision_recall,
        )
        doc = nlp("A red car.")
        p, r = vg_attribute_precision_recall(
            doc, vg_attrs={42: {"red"}}, image_id=42,
        )
        assert p == 1.0
        assert r == 1.0

    def test_partial_match(self, nlp):
        from ttic_embeddings.metrics.specificity import (
            vg_attribute_precision_recall,
        )
        doc = nlp("A red shiny car.")
        # caption attrs: {red, shiny}; gt: {red, blue}
        # intersection: {red}; precision 1/2 = 0.5; recall 1/2 = 0.5
        p, r = vg_attribute_precision_recall(
            doc, vg_attrs={42: {"red", "blue"}}, image_id=42,
        )
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(0.5)


# ---------------------------------------------------------------------
# Spatial: topological_density, projective_density
# ---------------------------------------------------------------------


class TestTopologicalDensity:
    def test_on_matches(self, nlp, topological_matcher):
        from ttic_embeddings.metrics.spatial import topological_density
        doc = nlp("A book on a shelf.")
        density = topological_density(doc, topological_matcher)
        # one match in the doc, density = 1 / len(doc)
        assert density == pytest.approx(1 / len(doc))

    def test_no_topological(self, nlp, topological_matcher):
        from ttic_embeddings.metrics.spatial import topological_density
        doc = nlp("A cat.")
        assert topological_density(doc, topological_matcher) == 0.0

    def test_in_matches(self, nlp, topological_matcher):
        from ttic_embeddings.metrics.spatial import topological_density
        doc = nlp("A bird in a tree.")
        assert topological_density(doc, topological_matcher) > 0


class TestProjectiveDensity:
    def test_next_to_matches(self, nlp, projective_matcher):
        """`next to` is a multi-word phrase; matcher must handle it."""
        from ttic_embeddings.metrics.spatial import projective_density
        doc = nlp("A book next to a lamp.")
        assert projective_density(doc, projective_matcher) > 0

    def test_behind_matches(self, nlp, projective_matcher):
        from ttic_embeddings.metrics.spatial import projective_density
        doc = nlp("A cat behind the chair.")
        assert projective_density(doc, projective_matcher) > 0

    def test_no_projective_in_topological_only(self, nlp, projective_matcher):
        """Topological-only captions should score zero on projective."""
        from ttic_embeddings.metrics.spatial import projective_density
        doc = nlp("A book on a shelf.")
        assert projective_density(doc, projective_matcher) == 0.0

    def test_in_front_of_multiword(self, nlp, projective_matcher):
        from ttic_embeddings.metrics.spatial import projective_density
        doc = nlp("A car in front of a building.")
        assert projective_density(doc, projective_matcher) > 0


# ---------------------------------------------------------------------
# Abstraction: head_noun_min_depth, scene_object_ratio
# ---------------------------------------------------------------------


class TestHeadNounMinDepth:
    def test_dog_has_depth(self, nlp, wordnet):
        from ttic_embeddings.metrics.abstraction import head_noun_min_depth
        doc = nlp("A dog runs.")
        depth = head_noun_min_depth(doc)
        assert depth is not None
        assert depth > 0

    def test_no_noun_returns_none(self, nlp, wordnet):
        from ttic_embeddings.metrics.abstraction import head_noun_min_depth
        doc = nlp("Beautiful.")
        assert head_noun_min_depth(doc) is None

    def test_specific_deeper_than_general(self, nlp, wordnet):
        """Hyponym (more specific) should have greater min_depth than its hypernym."""
        from ttic_embeddings.metrics.abstraction import head_noun_min_depth
        # mammal -> animal -> organism (animal is shallower than mammal)
        d_mammal = head_noun_min_depth(nlp("A mammal sleeps."))
        d_animal = head_noun_min_depth(nlp("An animal sleeps."))
        assert d_mammal is not None and d_animal is not None
        assert d_mammal > d_animal


class TestSceneObjectRatio:
    def test_only_objects(self, nlp):
        from ttic_embeddings.metrics.abstraction import scene_object_ratio
        doc = nlp("A car and a person.")
        # car, person -> COCO_OBJECTS. No scene matches.
        assert scene_object_ratio(doc) == 0.0

    def test_only_scene(self, nlp):
        from ttic_embeddings.metrics.abstraction import scene_object_ratio
        doc = nlp("A kitchen.")
        # kitchen -> DEFAULT_SCENES. No object matches.
        assert scene_object_ratio(doc) == 1.0

    def test_no_match_returns_none(self, nlp):
        from ttic_embeddings.metrics.abstraction import scene_object_ratio
        doc = nlp("Hello world.")
        assert scene_object_ratio(doc) is None

    def test_mixed(self, nlp):
        from ttic_embeddings.metrics.abstraction import scene_object_ratio
        doc = nlp("A car in a kitchen.")
        # 1 object (car), 1 scene (kitchen) -> ratio 0.5
        ratio = scene_object_ratio(doc)
        assert ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------
# Diversity: mtld, mean_caption_length, caption_length
# ---------------------------------------------------------------------


class TestDiversity:
    def test_mtld_diverse_higher_than_repetitive(self):
        """Diverse realistic text should have higher MTLD than repetitive text.

        Both inputs need to have TTR drop below the 0.72 threshold at some
        point — the MTLD algorithm returns 0 for all-unique input because
        TTR stays at 1.0 and the factor count never increments.
        """
        from ttic_embeddings.metrics.diversity import mtld
        # Heavily repeated short pattern — TTR drops fast, MTLD is small.
        repetitive = mtld(["the", "cat", "sat", "the", "cat", "sat"] * 25)
        # Varied text with realistic repetition — TTR drops slower, MTLD higher.
        varied = (
            "the quick brown fox jumps over the lazy dog and runs through "
            "the park to find its friends near the wooden bench by the river"
        ).split()
        diverse = mtld(varied * 4)
        assert diverse > repetitive

    def test_mtld_empty_input(self):
        from ttic_embeddings.metrics.diversity import mtld
        assert mtld([]) == 0.0
        assert mtld("") == 0.0

    def test_mtld_string_split(self):
        """String input should be split on whitespace and produce a finite score."""
        from ttic_embeddings.metrics.diversity import mtld
        result = mtld("the cat sat on the mat the cat sat on the mat")
        assert result > 0

    def test_caption_length_excludes_punctuation(self, nlp):
        from ttic_embeddings.metrics.diversity import caption_length
        doc = nlp("A red car.")
        # tokens: A, red, car, . — punct is excluded
        assert caption_length(doc) == 3

    def test_mean_caption_length(self, nlp):
        from ttic_embeddings.metrics.diversity import mean_caption_length
        docs = [nlp("A car."), nlp("A small red car drives by.")]
        # lengths: 2, 6 -> mean 4
        assert mean_caption_length(docs) == pytest.approx(4.0)

    def test_mean_caption_length_empty(self):
        from ttic_embeddings.metrics.diversity import mean_caption_length
        assert mean_caption_length([]) == 0.0
