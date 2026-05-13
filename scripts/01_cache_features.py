"""Phase 1.5 — precompute and cache encoder features for COCO train2017.

The encoder is frozen across the entire training pipeline; recomputing
its forward 22 epochs in a row was the dominant cost per training step.
This script runs the encoder once and writes patch tokens to a memmap.
Training then loads tensors instead of running the encoder.

Storage: fp16 memmap of shape (N_images, patch_count, dim) in column-major
contiguous order. A sidecar JSON records the row-by-row image_id order
and shape metadata so the training-side dataset can index by image_id.

Usage:
    uv run python scripts/01_cache_features.py --encoder clip --gpu 0
    uv run python scripts/01_cache_features.py --encoder siglip --gpu 1
    uv run python scripts/01_cache_features.py --encoder dinov2 --gpu 0
    uv run python scripts/01_cache_features.py --encoder mae --gpu 1

Resume-safe: if the meta JSON exists and its n/p/d match expectations,
the script exits without recomputing.

Sizing for COCO train2017 (118,287 images), fp16:
    CLIP-L:   256 patches x 1024 dim = 59 GiB
    SigLIP-L: 256 patches x 1024 dim = 59 GiB
    DINOv2-L: 256 patches x 1024 dim = 59 GiB
    MAE-L:    196 patches x 1024 dim = 47 GiB

Default $FEATURE_CACHE_ROOT is ./features. Set to a path with enough
free disk before running.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttic_embeddings.data.coco import (                            # noqa: E402
    _load_captions,
    _preprocess_image,
    coco_root_from_env,
)
from ttic_embeddings.encoders import build_encoder                  # noqa: E402
from ttic_embeddings.utils import get_logger                        # noqa: E402


class _ImageOnly(Dataset):
    """Minimal Dataset for cache building: returns preprocessed pixels by row."""

    def __init__(self, image_dir: Path, image_index: dict, image_ids: list[int],
                 image_processor) -> None:
        self.image_dir = image_dir
        self.image_index = image_index
        self.image_ids = image_ids
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        file_name = self.image_index[image_id]["file_name"]
        with Image.open(self.image_dir / file_name) as im:
            im = im.convert("RGB")
            px = _preprocess_image(self.image_processor, im)
        return idx, px

log = get_logger("cache_features")


def load_encoder_config(encoder_name: str, configs_dir: Path):
    base = OmegaConf.load(configs_dir / "base.yaml")
    enc = OmegaConf.load(configs_dir / f"{encoder_name}.yaml")
    if "defaults" in enc:
        del enc["defaults"]
    return OmegaConf.merge(base, enc)


def existing_cache_is_valid(meta_path: Path, expected_n: int, expected_p: int, expected_d: int) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    if (meta.get("n") != expected_n
            or meta.get("patch_count") != expected_p
            or meta.get("dim") != expected_d):
        return False
    bin_path = meta_path.with_suffix("").with_suffix(".fp16.bin")
    if not bin_path.exists():
        return False
    expected_bytes = expected_n * expected_p * expected_d * 2
    return bin_path.stat().st_size == expected_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encoder", required=True,
                        choices=["clip", "siglip", "dinov2", "mae"])
    parser.add_argument("--split", default="train",
                        choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cache-root", type=str, default=None,
                        help="Override $FEATURE_CACHE_ROOT.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap the number of images cached (for smoke tests).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_encoder_config(args.encoder, repo_root / "configs")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    cache_root = Path(args.cache_root) if args.cache_root else Path(cfg.paths.cache_root)
    out_dir = cache_root / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{args.split}2017_features.fp16.bin"
    meta_path = out_dir / f"{args.split}2017_features.meta.json"

    # ----- Resolve image set ----------------------------------------
    coco_root = Path(cfg.paths.coco_root) if cfg.paths.coco_root else coco_root_from_env()
    annotations_path = coco_root / "annotations" / f"captions_{args.split}2017.json"
    image_index, _ = _load_captions(annotations_path)
    image_ids = sorted(image_index.keys())
    if args.max_images is not None:
        image_ids = image_ids[:args.max_images]
    n = len(image_ids)
    p = int(cfg.encoder.patch_count)
    d = int(cfg.encoder.hidden_dim)
    expected_bytes = n * p * d * 2

    log.info("Encoder: %s (%s)", args.encoder, cfg.encoder.checkpoint)
    log.info("Images:  %d  |  patches: %d  |  dim: %d  |  fp16 size: %.1f GiB",
             n, p, d, expected_bytes / (1024 ** 3))
    log.info("Output:  %s", bin_path)

    if existing_cache_is_valid(meta_path, n, p, d):
        log.info("Cache already valid at %s — nothing to do.", meta_path)
        return 0

    # ----- Build encoder + image processor --------------------------
    encoder = build_encoder(args.encoder, checkpoint=cfg.encoder.checkpoint).to(device)
    encoder.eval()

    image_dir = coco_root / f"{args.split}2017"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    # ----- Allocate memmap ------------------------------------------
    log.info("Allocating memmap (%.1f GiB)...", expected_bytes / (1024 ** 3))
    mm = np.memmap(bin_path, dtype=np.float16, mode="w+", shape=(n, p, d))

    # ----- Encode in batches with multi-worker dataloader -----------
    ds = _ImageOnly(image_dir, image_index, image_ids, encoder.processor)
    nw = args.num_workers
    extras = {"persistent_workers": True, "prefetch_factor": 4} if nw > 0 else {}
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        **extras,
    )

    t0 = time.time()
    last_report = t0
    done = 0

    for indices, pixel_values in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            patches = encoder(pixel_values)  # (B, P, D)

        # fp16 storage: rounding loss is well below the downstream
        # training noise floor for normalized transformer activations.
        out_np = patches.to(torch.float16).cpu().numpy()
        # indices come from the Dataset and may be a tensor; coerce.
        idx_arr = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        # Contiguous batches => write as a slice if ids are sequential,
        # else fancy-indexed. With shuffle=False above this is always sequential.
        first, last = idx_arr[0], idx_arr[-1]
        if last - first + 1 == len(idx_arr):
            mm[first:last + 1] = out_np
        else:
            for i, row in zip(idx_arr, out_np):
                mm[i] = row

        done += len(idx_arr)
        now = time.time()
        if now - last_report > 30:
            ips = done / max(1.0, now - t0)
            eta_s = (n - done) / max(1e-6, ips)
            log.info("  %d / %d images  |  %.1f img/s  |  ETA %.1f min",
                     done, n, ips, eta_s / 60)
            last_report = now

    mm.flush()
    del mm

    # ----- Sidecar metadata -----------------------------------------
    meta = {
        "encoder": args.encoder,
        "encoder_checkpoint": cfg.encoder.checkpoint,
        "split": args.split,
        "n": n,
        "patch_count": p,
        "dim": d,
        "dtype": "float16",
        "shape": [n, p, d],
        "image_ids": image_ids,
        "bin_filename": bin_path.name,
    }
    meta_path.write_text(json.dumps(meta))
    elapsed = time.time() - t0
    log.info("Done. %d images in %.1f min (%.1f img/s).", n, elapsed / 60, n / elapsed)
    log.info("Wrote %s (%.1f GiB) and %s",
             bin_path, expected_bytes / (1024 ** 3), meta_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
