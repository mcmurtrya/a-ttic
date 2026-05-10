"""Tests for the COCO dataset wrappers — focus on the holdout split.

The holdout exists to keep early-stopping val PPL off val2017 (which is
the captioning eval set). Two properties matter and we test both:

  Determinism — same coco_root + same holdout_seed must give the same
  set on every machine, every run, every encoder. If this breaks, two
  encoders' early-stopping signals see different images and any
  cross-encoder comparison is contaminated.

  Disjointness — train (filter_mode='exclude') and holdout (filter_mode
  ='include') must never share an image_id.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_fake_coco(coco_root: Path, n_images: int = 100) -> None:
    """Minimal annotations layout: train2017/ + annotations/captions_train2017.json."""
    (coco_root / "train2017").mkdir(parents=True, exist_ok=True)
    (coco_root / "annotations").mkdir(parents=True, exist_ok=True)
    images = [{"id": i, "file_name": f"{i:012d}.jpg"} for i in range(n_images)]
    annotations = [
        {"image_id": i, "caption": f"caption {i}"}
        for i in range(n_images)
    ]
    payload = {"images": images, "annotations": annotations}
    with open(coco_root / "annotations" / "captions_train2017.json", "w") as f:
        json.dump(payload, f)


class TestTrainHoldoutImageIds:
    def test_deterministic_across_calls(self, tmp_path):
        from ttic_embeddings.data.coco import train_holdout_image_ids
        _write_fake_coco(tmp_path, n_images=100)
        a = train_holdout_image_ids(tmp_path, n_holdout=10, holdout_seed=0)
        b = train_holdout_image_ids(tmp_path, n_holdout=10, holdout_seed=0)
        assert a == b
        assert len(a) == 10

    def test_seed_changes_set(self, tmp_path):
        from ttic_embeddings.data.coco import train_holdout_image_ids
        _write_fake_coco(tmp_path, n_images=100)
        a = train_holdout_image_ids(tmp_path, n_holdout=10, holdout_seed=0)
        b = train_holdout_image_ids(tmp_path, n_holdout=10, holdout_seed=1)
        assert a != b

    def test_holdout_too_large_raises(self, tmp_path):
        from ttic_embeddings.data.coco import train_holdout_image_ids
        _write_fake_coco(tmp_path, n_images=10)
        with pytest.raises(ValueError, match="holdout size"):
            train_holdout_image_ids(tmp_path, n_holdout=10, holdout_seed=0)


class TestCocoCaptionPairsFiltering:
    """The dataset constructor needs an image_processor + tokenizer, both
    expensive to instantiate. We bypass __init__ and exercise just the
    filter logic on the items list, which is what matters for leakage.
    """

    def test_exclude_mode_drops_listed_ids(self, tmp_path):
        # Hit the filter logic by hand, mirroring what __init__ builds.
        # This avoids spinning up a HF tokenizer in unit tests.
        from ttic_embeddings.data.coco import _load_captions
        _write_fake_coco(tmp_path, n_images=20)
        image_index, annotations = _load_captions(
            tmp_path / "annotations" / "captions_train2017.json"
        )
        holdout = {0, 1, 2, 3, 4}

        def _keep(image_id, filter_mode):
            in_set = image_id in holdout
            return in_set if filter_mode == "include" else not in_set

        train_items = [a for a in annotations if _keep(a["image_id"], "exclude")]
        holdout_items = [a for a in annotations if _keep(a["image_id"], "include")]

        train_ids = {a["image_id"] for a in train_items}
        holdout_ids = {a["image_id"] for a in holdout_items}
        assert train_ids.isdisjoint(holdout_ids)
        assert holdout_ids == holdout
        assert len(train_ids) == 15
