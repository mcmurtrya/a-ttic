"""Tests for four new behaviors:
  1. parse_and_cache — content-hash gating via .meta sidecar
  2. set_seed / seed_worker — deterministic RNG seeding
  3. _load_captions — lru_cache prevents repeated JSON reads
  4. maybe_resume — hparam-drift warning on lr mismatch
"""
from __future__ import annotations

import json
import logging
import random
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import torch
import torch.nn as nn


# =============================================================================
# 1. parse_and_cache — content-hash gating
# =============================================================================


class TestCaptionsHash:
    def test_deterministic(self):
        from ttic_embeddings.metrics.parse import _captions_hash
        captions = ["A red car.", "A dog runs.", "Two cats sleep."]
        assert _captions_hash(captions) == _captions_hash(captions)

    def test_sensitive_to_single_char_change(self):
        from ttic_embeddings.metrics.parse import _captions_hash
        captions_a = ["A red car."]
        captions_b = ["A red car!"]  # one char different
        assert _captions_hash(captions_a) != _captions_hash(captions_b)

    def test_order_matters(self):
        from ttic_embeddings.metrics.parse import _captions_hash
        a = ["first", "second"]
        b = ["second", "first"]
        assert _captions_hash(a) != _captions_hash(b)

    def test_empty_list(self):
        from ttic_embeddings.metrics.parse import _captions_hash
        h = _captions_hash([])
        assert isinstance(h, str) and len(h) == 64  # sha256 hex digest


# Helper: build a fake Doc list that parse_captions would return
def _fake_docs(captions):
    """Return opaque sentinel objects — one per caption, distinguishable by index."""
    return [object() for _ in captions]


class TestParseAndCache:
    """Mock parse_captions + cache_parsed_docs + load_parsed_docs so spaCy is never loaded."""

    def _make_mocks(self, captions):
        """Return a tuple (mock_parse, mock_cache, mock_load, fake_result)."""
        fake_result = _fake_docs(captions)
        mock_parse = MagicMock(return_value=fake_result)
        mock_cache = MagicMock()
        mock_load = MagicMock(return_value=fake_result)
        return mock_parse, mock_cache, mock_load, fake_result

    def test_same_captions_hits_cache(self, tmp_path):
        """Second call with matching hash must NOT re-invoke parse_captions."""
        from ttic_embeddings.metrics.parse import _captions_hash

        captions = ["A cat.", "A dog."]
        cache_path = tmp_path / "docs.spacy"
        meta_path = cache_path.with_suffix(cache_path.suffix + ".meta")
        expected_hash = _captions_hash(captions)

        mock_parse, mock_cache, mock_load, fake_result = self._make_mocks(captions)

        with (
            patch("ttic_embeddings.metrics.parse.parse_captions", mock_parse),
            patch("ttic_embeddings.metrics.parse.cache_parsed_docs", mock_cache),
            patch("ttic_embeddings.metrics.parse.load_parsed_docs", mock_load),
        ):
            # First call: cache absent → parse + write
            from ttic_embeddings.metrics.parse import parse_and_cache
            parse_and_cache(captions, cache_path)

            # Simulate what the real code writes after caching
            cache_path.touch()
            meta_path.write_text(expected_hash)

            mock_parse.reset_mock()
            mock_load.reset_mock()

            # Second call: cache + meta present with matching hash → load, no parse
            result = parse_and_cache(captions, cache_path)

        mock_parse.assert_not_called()
        mock_load.assert_called_once()

    def test_changed_caption_same_length_invalidates_cache(self, tmp_path):
        """Edited captions (same count) must trigger re-parse."""
        from ttic_embeddings.metrics.parse import _captions_hash

        original = ["A cat.", "A dog."]
        edited = ["A cat.", "A pig."]  # same length, different content
        cache_path = tmp_path / "docs.spacy"
        meta_path = cache_path.with_suffix(cache_path.suffix + ".meta")

        mock_parse, mock_cache, mock_load, _ = self._make_mocks(original)

        with (
            patch("ttic_embeddings.metrics.parse.parse_captions", mock_parse),
            patch("ttic_embeddings.metrics.parse.cache_parsed_docs", mock_cache),
            patch("ttic_embeddings.metrics.parse.load_parsed_docs", mock_load),
        ):
            from ttic_embeddings.metrics.parse import parse_and_cache

            # Plant stale meta (original hash) but request edited captions
            cache_path.touch()
            meta_path.write_text(_captions_hash(original))

            mock_parse.reset_mock()
            parse_and_cache(edited, cache_path)

        # parse_captions must have been called once for the edited captions
        mock_parse.assert_called_once_with(edited, n_process=1)

    def test_missing_meta_invalidates_cache(self, tmp_path):
        """Cache file present but no .meta → treated as stale → re-parses."""
        captions = ["A cat.", "A dog."]
        cache_path = tmp_path / "docs.spacy"
        cache_path.touch()  # cache file exists, no .meta

        mock_parse, mock_cache, mock_load, _ = self._make_mocks(captions)

        with (
            patch("ttic_embeddings.metrics.parse.parse_captions", mock_parse),
            patch("ttic_embeddings.metrics.parse.cache_parsed_docs", mock_cache),
            patch("ttic_embeddings.metrics.parse.load_parsed_docs", mock_load),
        ):
            from ttic_embeddings.metrics.parse import parse_and_cache
            parse_and_cache(captions, cache_path)

        mock_parse.assert_called_once()
        mock_load.assert_not_called()

    def test_force_true_always_reparses(self, tmp_path):
        """force=True must re-parse even when cache + meta are valid."""
        from ttic_embeddings.metrics.parse import _captions_hash

        captions = ["A cat.", "A dog."]
        cache_path = tmp_path / "docs.spacy"
        meta_path = cache_path.with_suffix(cache_path.suffix + ".meta")
        cache_path.touch()
        meta_path.write_text(_captions_hash(captions))  # valid meta

        mock_parse, mock_cache, mock_load, _ = self._make_mocks(captions)

        with (
            patch("ttic_embeddings.metrics.parse.parse_captions", mock_parse),
            patch("ttic_embeddings.metrics.parse.cache_parsed_docs", mock_cache),
            patch("ttic_embeddings.metrics.parse.load_parsed_docs", mock_load),
        ):
            from ttic_embeddings.metrics.parse import parse_and_cache
            parse_and_cache(captions, cache_path, force=True)

        mock_parse.assert_called_once()
        mock_load.assert_not_called()

    def test_new_hash_written_after_reparse(self, tmp_path):
        """After a stale-cache re-parse the new hash must land in the .meta file."""
        from ttic_embeddings.metrics.parse import _captions_hash

        captions = ["Brand new caption."]
        cache_path = tmp_path / "docs.spacy"
        meta_path = cache_path.with_suffix(cache_path.suffix + ".meta")
        cache_path.touch()
        meta_path.write_text("definitely_not_the_right_hash")

        mock_parse, mock_cache, mock_load, _ = self._make_mocks(captions)

        with (
            patch("ttic_embeddings.metrics.parse.parse_captions", mock_parse),
            patch("ttic_embeddings.metrics.parse.cache_parsed_docs", mock_cache),
            patch("ttic_embeddings.metrics.parse.load_parsed_docs", mock_load),
        ):
            from ttic_embeddings.metrics.parse import parse_and_cache
            parse_and_cache(captions, cache_path)

        assert meta_path.read_text().strip() == _captions_hash(captions)


# =============================================================================
# 2. set_seed / seed_worker
# =============================================================================


class TestSetSeed:
    def test_reproducible_random(self):
        from ttic_embeddings.utils import set_seed
        set_seed(42)
        a = [random.random() for _ in range(5)]
        set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_reproducible_numpy(self):
        from ttic_embeddings.utils import set_seed
        set_seed(7)
        a = np.random.rand(10).tolist()
        set_seed(7)
        b = np.random.rand(10).tolist()
        assert a == b

    def test_reproducible_torch(self):
        from ttic_embeddings.utils import set_seed
        set_seed(99)
        a = torch.rand(5).tolist()
        set_seed(99)
        b = torch.rand(5).tolist()
        assert a == b

    def test_different_seeds_differ(self):
        from ttic_embeddings.utils import set_seed
        set_seed(1)
        a = torch.rand(5).tolist()
        set_seed(2)
        b = torch.rand(5).tolist()
        assert a != b

    def test_sets_pythonhashseed_env(self):
        from ttic_embeddings.utils import set_seed
        set_seed(123)
        assert os.environ.get("PYTHONHASHSEED") == "123"

    def test_sets_cublas_workspace_config(self):
        from ttic_embeddings.utils import set_seed
        # Remove the var so setdefault fires, then verify it's set afterwards
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        set_seed(0)
        assert "CUBLAS_WORKSPACE_CONFIG" in os.environ


class TestSeedWorker:
    def test_deterministic_numpy_state(self):
        """Same torch global seed → same numpy state after seed_worker."""
        from ttic_embeddings.utils import seed_worker

        torch.manual_seed(42)
        seed_worker(0)
        a = np.random.rand(5).tolist()

        torch.manual_seed(42)
        seed_worker(0)
        b = np.random.rand(5).tolist()

        assert a == b

    def test_deterministic_python_random_state(self):
        from ttic_embeddings.utils import seed_worker

        torch.manual_seed(77)
        seed_worker(0)
        a = [random.random() for _ in range(5)]

        torch.manual_seed(77)
        seed_worker(0)
        b = [random.random() for _ in range(5)]

        assert a == b

    def test_different_initial_seeds_differ(self):
        """Different torch global seeds must produce different worker states."""
        from ttic_embeddings.utils import seed_worker

        torch.manual_seed(1)
        seed_worker(0)
        a = np.random.rand(5).tolist()

        torch.manual_seed(2)
        seed_worker(0)
        b = np.random.rand(5).tolist()

        assert a != b


# =============================================================================
# 3. _load_captions — lru_cache
# =============================================================================


def _write_minimal_coco(path: Path) -> None:
    payload = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [{"image_id": 1, "caption": "x"}],
    }
    path.write_text(json.dumps(payload))


class TestLoadCaptionsCache:
    def test_second_call_does_not_reread(self, tmp_path):
        """lru_cache must absorb the second call — json.load invoked only once."""
        from ttic_embeddings.data.coco import _load_captions
        _load_captions.cache_clear()

        ann_path = tmp_path / "captions.json"
        _write_minimal_coco(ann_path)

        read_count = 0
        real_open = open  # capture before patching

        def counting_open(file, *args, **kwargs):
            nonlocal read_count
            if Path(file) == ann_path:
                read_count += 1
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=counting_open):
            _load_captions(ann_path)
            _load_captions(ann_path)

        assert read_count == 1, (
            f"Expected 1 file open for JSON, got {read_count}. "
            "lru_cache may not be working."
        )

    def test_returns_correct_structure(self, tmp_path):
        from ttic_embeddings.data.coco import _load_captions
        _load_captions.cache_clear()

        ann_path = tmp_path / "captions.json"
        _write_minimal_coco(ann_path)

        image_index, annotations = _load_captions(ann_path)
        assert 1 in image_index
        assert image_index[1]["file_name"] == "a.jpg"
        assert len(annotations) == 1
        assert annotations[0]["caption"] == "x"

    def test_different_paths_are_independent(self, tmp_path):
        from ttic_embeddings.data.coco import _load_captions
        _load_captions.cache_clear()

        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        payload_a = {"images": [{"id": 10, "file_name": "x.jpg"}], "annotations": []}
        payload_b = {"images": [{"id": 20, "file_name": "y.jpg"}], "annotations": []}
        path_a.write_text(json.dumps(payload_a))
        path_b.write_text(json.dumps(payload_b))

        idx_a, _ = _load_captions(path_a)
        idx_b, _ = _load_captions(path_b)
        assert 10 in idx_a and 10 not in idx_b
        assert 20 in idx_b and 20 not in idx_a


# =============================================================================
# 4. maybe_resume — hparam-drift warning
# =============================================================================


def _tiny_model_and_optimizer(lr: float):
    """Return (nn.Linear(2,2), AdamW) — minimal but real PyTorch objects."""
    model = nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    return model, opt


def _dummy_scheduler(opt):
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)


class TestMaybeResume:
    def test_no_checkpoint_returns_zero_inf(self, tmp_path):
        from ttic_embeddings.train import maybe_resume
        model, opt = _tiny_model_and_optimizer(1e-3)
        sched = _dummy_scheduler(opt)
        ckpt = tmp_path / "nonexistent.pt"
        step, ppl = maybe_resume(model, opt, sched, ckpt)
        assert step == 0
        assert ppl == float("inf")

    def test_no_drift_no_warning(self, tmp_path, caplog):
        """Checkpoint saved with lr=1e-3, resumed with lr=1e-3 → no WARNING."""
        from ttic_embeddings.train import save_checkpoint, maybe_resume

        model, opt = _tiny_model_and_optimizer(1e-3)
        sched = _dummy_scheduler(opt)
        ckpt = tmp_path / "ckpt.pt"
        save_checkpoint(model, opt, sched, step=5, val_ppl=3.14, path=ckpt)

        # Fresh model + optimizer with the SAME lr
        model2, opt2 = _tiny_model_and_optimizer(1e-3)
        sched2 = _dummy_scheduler(opt2)

        with caplog.at_level(logging.WARNING, logger="ttic_embeddings.train"):
            maybe_resume(model2, opt2, sched2, ckpt)

        drift_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "lr" in r.message.lower()
            and "differ" in r.message.lower()
        ]
        assert len(drift_warnings) == 0, (
            f"Unexpected lr-drift warning on matching lr: {drift_warnings}"
        )

    def test_lr_drift_emits_warning(self, tmp_path, caplog):
        """Checkpoint saved with lr=1e-3, resumed with lr=5e-4 → WARNING about lr."""
        from ttic_embeddings.train import save_checkpoint, maybe_resume

        model, opt = _tiny_model_and_optimizer(1e-3)
        sched = _dummy_scheduler(opt)
        ckpt = tmp_path / "ckpt.pt"
        save_checkpoint(model, opt, sched, step=10, val_ppl=2.5, path=ckpt)

        # Fresh model + optimizer with DIFFERENT lr
        model2, opt2 = _tiny_model_and_optimizer(5e-4)
        sched2 = _dummy_scheduler(opt2)

        with caplog.at_level(logging.WARNING, logger="ttic_embeddings.train"):
            step, ppl = maybe_resume(model2, opt2, sched2, ckpt)

        drift_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "lr" in r.message.lower()
        ]
        assert len(drift_warnings) >= 1, (
            "Expected at least one WARNING mentioning 'lr' for a drift from "
            "lr=1e-3 to lr=5e-4, but none were emitted."
        )

    def test_resume_restores_step_and_ppl(self, tmp_path):
        """maybe_resume must return the saved step and val_ppl."""
        from ttic_embeddings.train import save_checkpoint, maybe_resume

        model, opt = _tiny_model_and_optimizer(1e-3)
        sched = _dummy_scheduler(opt)
        ckpt = tmp_path / "ckpt.pt"
        save_checkpoint(model, opt, sched, step=42, val_ppl=1.23, path=ckpt)

        model2, opt2 = _tiny_model_and_optimizer(1e-3)
        sched2 = _dummy_scheduler(opt2)
        step, ppl = maybe_resume(model2, opt2, sched2, ckpt)

        assert step == 42
        assert ppl == pytest.approx(1.23)
