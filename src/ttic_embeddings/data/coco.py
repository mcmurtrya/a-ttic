"""COCO Captions dataset wrappers.

Two PyTorch Datasets, one per use case:

  CocoCaptionPairs — yields one (image, caption) pair per annotation.
    Used for adaptor training. With train2017's 5 captions per image,
    iterating this dataset gives ~590K examples per epoch.

  CocoEvalImages — yields one entry per unique image. The 5 reference
    captions per image are stored separately and accessible via the
    `references(image_id)` method. We split images and references like
    this so the dataset returns only tensor-friendly fields, keeping
    PyTorch's default_collate happy for batched eval; references are
    looked up by image_id in the eval loop after generation.

Both expect $COCO_ROOT to point at a directory laid out by
`scripts/01_download_data.py`:

    $COCO_ROOT/
      annotations/captions_{train,val}2017.json
      {train,val}2017/*.jpg

Both are parameterized by the encoder's image processor (small,
picklable by torch's dataloader workers — unlike the full encoder,
which is large and shouldn't be sent to workers).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def coco_root_from_env(default: str = "./data/coco") -> Path:
    """Resolve $COCO_ROOT or fall back to a project-relative default."""
    return Path(os.environ.get("COCO_ROOT", default))


def train_holdout_image_ids(
    coco_root: Path | str | None = None,
    n_holdout: int = 5000,
    holdout_seed: int = 0,
) -> set[int]:
    """Deterministic held-out subset of train2017 image ids for early-stopping val.

    Captioning eval reports metrics on val2017 (CocoEvalImages). If we
    also use val2017 for the early-stopping criterion, the model selects
    a checkpoint specifically tuned to the very images we then evaluate
    on — train/eval leakage. Methods.md commits to a held-out subset of
    train2017 for the stopping signal; this function produces it.

    Choice is deterministic across encoders/seeds — the holdout must be
    identical for every condition we compare. `holdout_seed` is intentionally
    separate from `cfg.seed`; the same holdout is used by every run.
    """
    coco_root = Path(coco_root) if coco_root else coco_root_from_env()
    annotations_path = coco_root / "annotations" / "captions_train2017.json"
    image_index, _ = _load_captions(annotations_path)
    image_ids = sorted(image_index.keys())
    if n_holdout >= len(image_ids):
        raise ValueError(
            f"holdout size {n_holdout} >= train2017 image count {len(image_ids)}"
        )
    rng = np.random.default_rng(holdout_seed)
    selected = rng.choice(len(image_ids), size=n_holdout, replace=False)
    return {image_ids[i] for i in selected}


def _load_captions(annotations_path: Path) -> tuple[dict[int, dict], list[dict]]:
    """Read a COCO captions JSON. Returns (image_index, annotations)."""
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"COCO captions JSON not found at {annotations_path}. "
            f"Run `make data` (or scripts/01_download_data.py) first."
        )
    with open(annotations_path) as f:
        data = json.load(f)
    image_index: dict[int, dict] = {img["id"]: img for img in data["images"]}
    return image_index, data["annotations"]


def _preprocess_image(image_processor: Any, image: Image.Image):
    """Run a HuggingFace image processor on a single PIL image, return (3, H, W)."""
    out = image_processor(images=[image], return_tensors="pt")
    return out["pixel_values"][0]


class CocoCaptionPairs(Dataset):
    """Yields one (image, caption) pair per COCO annotation.

    Each item is a dict with fields:
        image_id        int
        pixel_values    torch.Tensor (3, H, W) — already preprocessed for the encoder
        input_ids       torch.LongTensor (L,)  — caption tokens + EOS, padded to L
        attention_mask  torch.LongTensor (L,)  — 1 on real tokens, 0 on padding
        caption         str — raw caption text, useful for debugging

    The pad_token of the GPT-2 tokenizer is set to its eos_token if not
    already configured; the attention_mask lets the training loop replace
    the corresponding label positions with -100 so loss isn't computed
    on padding.
    """

    def __init__(
        self,
        coco_root: Path | str | None = None,
        split: str = "train",
        image_processor: Any = None,
        tokenizer: Any = None,
        max_caption_tokens: int = 32,
        image_ids_filter: set[int] | None = None,
        filter_mode: str = "include",
    ) -> None:
        if image_processor is None:
            raise ValueError("image_processor is required (e.g. encoder.processor)")
        if tokenizer is None:
            raise ValueError("tokenizer is required (e.g. GPT2Tokenizer.from_pretrained(...))")
        if filter_mode not in ("include", "exclude"):
            raise ValueError(
                f"filter_mode must be 'include' or 'exclude', got {filter_mode!r}"
            )

        self.coco_root = Path(coco_root) if coco_root else coco_root_from_env()
        self.split = split
        self.image_dir = self.coco_root / f"{split}2017"
        annotations_path = self.coco_root / "annotations" / f"captions_{split}2017.json"

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory {self.image_dir} not found. "
                f"Run `make data` to populate it."
            )

        # GPT-2 has no pad token by default — repurpose EOS for right-padding.
        # The attention_mask will tell the training loop where the real tokens end.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_caption_tokens = max_caption_tokens

        image_index, annotations = _load_captions(annotations_path)

        def _keep(image_id: int) -> bool:
            if image_id not in image_index:
                return False
            if image_ids_filter is None:
                return True
            in_set = image_id in image_ids_filter
            return in_set if filter_mode == "include" else not in_set

        self.items: list[tuple[int, str, str]] = [
            (
                ann["image_id"],
                image_index[ann["image_id"]]["file_name"],
                ann["caption"].strip(),
            )
            for ann in annotations
            if _keep(ann["image_id"])
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        image_id, file_name, caption = self.items[idx]

        image = Image.open(self.image_dir / file_name).convert("RGB")
        pixel_values = _preprocess_image(self.image_processor, image)

        # Append EOS so the model learns when to stop
        text = caption + self.tokenizer.eos_token
        enc = self.tokenizer(
            text,
            max_length=self.max_caption_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return {
            "image_id": image_id,
            "pixel_values": pixel_values,
            "input_ids": enc.input_ids[0],
            "attention_mask": enc.attention_mask[0],
            "caption": caption,
        }


class CocoEvalImages(Dataset):
    """Yields one entry per unique image in the requested split.

    Each item is a dict with:
        image_id        int
        pixel_values    torch.Tensor (3, H, W) — already preprocessed for the encoder

    Reference captions are stored separately and accessible via
    `references(image_id) -> list[str]`. Splitting like this keeps
    default_collate happy for batched eval; the eval loop pulls
    references by id after generation.
    """

    def __init__(
        self,
        coco_root: Path | str | None = None,
        split: str = "val",
        image_processor: Any = None,
    ) -> None:
        if image_processor is None:
            raise ValueError("image_processor is required (e.g. encoder.processor)")

        self.coco_root = Path(coco_root) if coco_root else coco_root_from_env()
        self.split = split
        self.image_dir = self.coco_root / f"{split}2017"
        annotations_path = self.coco_root / "annotations" / f"captions_{split}2017.json"

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory {self.image_dir} not found. "
                f"Run `make data` to populate it."
            )

        self.image_processor = image_processor

        image_index, annotations = _load_captions(annotations_path)
        refs_by_image: dict[int, list[str]] = defaultdict(list)
        for ann in annotations:
            refs_by_image[ann["image_id"]].append(ann["caption"].strip())

        # Stable order: iterate image_index by id so eval is deterministic
        self._refs_by_image: dict[int, list[str]] = dict(refs_by_image)
        self.items: list[tuple[int, str]] = [
            (image_id, image_index[image_id]["file_name"])
            for image_id in sorted(image_index)
            if image_id in refs_by_image
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        image_id, file_name = self.items[idx]
        image = Image.open(self.image_dir / file_name).convert("RGB")
        pixel_values = _preprocess_image(self.image_processor, image)
        return {"image_id": image_id, "pixel_values": pixel_values}

    def references(self, image_id: int) -> list[str]:
        """Return the list of reference captions for one image (typically 5)."""
        return self._refs_by_image[image_id]


def _smoke_check() -> None:
    """Quick check — needs COCO data on disk plus HF tokenizer/processor downloads.

    Run via:
        uv run python -m ttic_embeddings.data.coco
    """
    print("CocoCaptionPairs / CocoEvalImages smoke check")

    coco_root = coco_root_from_env()
    print(f"  coco_root: {coco_root}")
    if not (coco_root / "annotations" / "captions_train2017.json").exists():
        raise SystemExit(
            "Annotations not found. Run `make data` first to populate $COCO_ROOT."
        )

    from transformers import CLIPImageProcessor, GPT2Tokenizer
    image_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

    # --- training pairs -----------------------------------------------
    train_ds = CocoCaptionPairs(
        coco_root=coco_root,
        split="train",
        image_processor=image_proc,
        tokenizer=tokenizer,
        max_caption_tokens=32,
    )
    print(f"\n  CocoCaptionPairs(split=train) length: {len(train_ds):,}")
    item = train_ds[0]
    print(f"    image_id            = {item['image_id']}")
    print(f"    pixel_values.shape  = {tuple(item['pixel_values'].shape)}")
    print(f"    input_ids.shape     = {tuple(item['input_ids'].shape)}")
    print(f"    attention_mask.sum  = {int(item['attention_mask'].sum())} of {len(item['attention_mask'])}")
    print(f"    caption             = {item['caption']!r}")

    # --- eval images --------------------------------------------------
    val_ds = CocoEvalImages(
        coco_root=coco_root,
        split="val",
        image_processor=image_proc,
    )
    print(f"\n  CocoEvalImages(split=val) length: {len(val_ds):,}")
    item = val_ds[0]
    print(f"    image_id            = {item['image_id']}")
    print(f"    pixel_values.shape  = {tuple(item['pixel_values'].shape)}")
    refs = val_ds.references(item["image_id"])
    print(f"    references          = {len(refs)} captions")
    print(f"    refs[0]             = {refs[0]!r}")

    print("\nAll dataset checks passed.")


if __name__ == "__main__":
    _smoke_check()
