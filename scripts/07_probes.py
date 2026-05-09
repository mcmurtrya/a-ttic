"""Phase 5 — linear probes on frozen encoder features.

For each encoder, train a linear classifier on COCO 80-class object
multi-label classification, using mean-pooled patch features as the
probe input. The probe accuracy gives independent evidence of what
each encoder's representations preserve, complementing the
caption-style metrics in scripts 05/06.

Methodologically: if language-supervised encoders score higher on the
object probe AND produce more object-naming captions, that is converging
evidence for the supervision-objective claim. If probe accuracies are
similar across encoders but caption styles differ, the difference cannot
be attributed to "what the encoder represents" — it is in how the
adaptor reads the representation. Either outcome is informative.

Independent of training: this script uses the raw frozen encoders and
does not require trained adaptors. You can run it anytime after Phase 0.

Visual Genome spatial-relation probes are a TODO. They need
relationships.json, which the data download script does not currently
fetch. Adding that probe is one extra read in 01_download_data.py and
one extra task block here.

Reads:
    $COCO_ROOT/annotations/instances_val2017.json    object detection labels
    $COCO_ROOT/val2017/                              val images

Writes:
    {output_dir}/probe_results.csv
        long-format: encoder, task, metric, score, n_classes, n_train, n_test
    {cache_root}/probe_features_{encoder}.npz
        cached mean-pooled features per encoder (re-run with no recompute)

Compute note:
    Feature extraction is ~5 sec/image per encoder on CPU. The full 5K
    val set per encoder takes ~7 hours on CPU, ~5 minutes on A100. Use
    --max-images 100 for a script smoke test on CPU.

Usage:
    uv run python scripts/07_probes.py
    uv run python scripts/07_probes.py --encoders clip,siglip
    uv run python scripts/07_probes.py --max-images 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttic_embeddings.data.coco import CocoEvalImages   # noqa: E402
from ttic_embeddings.encoders import build_encoder      # noqa: E402
from ttic_embeddings.utils import get_logger, set_seed  # noqa: E402

log = get_logger("probes")

ALL_ENCODERS = ("clip", "siglip", "dinov2", "mae")


def load_config(encoder_name: str, configs_dir: Path) -> DictConfig:
    base = OmegaConf.load(configs_dir / "base.yaml")
    enc = OmegaConf.load(configs_dir / f"{encoder_name}.yaml")
    if "defaults" in enc:
        del enc["defaults"]
    return OmegaConf.merge(base, enc)


def load_coco_object_labels(
    instances_path: Path,
) -> tuple[dict[int, np.ndarray], list[str]]:
    """Load COCO 80-class multi-label binary vectors per image.

    COCO category IDs are sparse (some integers skipped). We remap to
    contiguous 0..79 indexing aligned with the alphabetical category list.
    """
    if not instances_path.exists():
        raise FileNotFoundError(
            f"COCO instances annotations not found at {instances_path}. "
            f"Run `make data` to populate $COCO_ROOT/annotations/. The "
            f"annotations_trainval2017.zip already includes instances JSONs."
        )
    with open(instances_path) as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda c: c["id"])
    cat_names = [c["name"] for c in categories]
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    n_classes = len(categories)

    image_ids = {img["id"] for img in data["images"]}
    labels = {img_id: np.zeros(n_classes, dtype=np.float32) for img_id in image_ids}

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_idx = cat_id_to_idx[ann["category_id"]]
        labels[img_id][cat_idx] = 1.0

    return labels, cat_names


def extract_features(
    encoder: torch.nn.Module,
    dataset: CocoEvalImages,
    device: torch.device,
    cache_path: Path | None,
    max_images: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean-pooled patch features per image. Cached to .npz on disk."""
    if cache_path is not None and cache_path.exists():
        log.info("Loading cached features from %s", cache_path)
        data = np.load(cache_path)
        feats = data["features"]
        ids = data["image_ids"]
        if max_images is not None and len(feats) > max_images:
            feats = feats[:max_images]
            ids = ids[:max_images]
        return feats, ids

    encoder.to(device)
    encoder.eval()
    n = len(dataset) if max_images is None else min(max_images, len(dataset))

    features: list[np.ndarray] = []
    image_ids: list[int] = []
    for i in tqdm(range(n), desc="extracting", unit="img"):
        item = dataset[i]
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        with torch.no_grad():
            patches = encoder(pixel_values)
        feat = patches.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        features.append(feat)
        image_ids.append(int(item["image_id"]))

    features_arr = np.stack(features, axis=0)
    image_ids_arr = np.array(image_ids, dtype=np.int64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, features=features_arr, image_ids=image_ids_arr)
        log.info("Cached features to %s (shape=%s)", cache_path, features_arr.shape)

    return features_arr, image_ids_arr


def train_and_evaluate_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_train: int,
    seed: int = 0,
) -> dict:
    """Random-shuffle 80/20 split, train OneVsRest LR probe, return metrics."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(features))
    rng.shuffle(indices)
    n_train_actual = min(n_train, len(features) - max(1, len(features) // 5))
    train_idx = indices[:n_train_actual]
    test_idx = indices[n_train_actual:]

    X_train, Y_train = features[train_idx], labels[train_idx]
    X_test, Y_test = features[test_idx], labels[test_idx]

    log.info(
        "Probe: train=%d, test=%d, classes=%d",
        len(X_train), len(X_test), Y_train.shape[1],
    )
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        n_jobs=1,  # set to -1 for parallelism on Linux/Mac; can hang on Windows
    )
    clf.fit(X_train, Y_train)

    proba = np.column_stack([
        est.predict_proba(X_test)[:, 1] for est in clf.estimators_
    ])
    pred = (proba > 0.5).astype(np.int32)

    aps: list[float] = []
    f1s: list[float] = []
    for c in range(Y_train.shape[1]):
        # Skip classes with no positives in test (AP / F1 undefined)
        if Y_test[:, c].sum() == 0:
            continue
        aps.append(float(average_precision_score(Y_test[:, c], proba[:, c])))
        f1s.append(float(f1_score(Y_test[:, c], pred[:, c], zero_division=0)))

    return {
        "mAP": float(np.mean(aps)) if aps else float("nan"),
        "mean_F1": float(np.mean(f1s)) if f1s else float("nan"),
        "n_classes_evaluated": len(aps),
        "n_classes_total": int(Y_train.shape[1]),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def probe_one_encoder(
    encoder_name: str,
    cfg: DictConfig,
    image_id_to_label: dict[int, np.ndarray],
    device: torch.device,
    cache_root: Path,
    n_train: int,
    max_images: int | None,
    seed: int,
) -> list[dict]:
    """Run feature extraction + probe training for one encoder."""
    log.info("=" * 60)
    log.info("Encoder: %s", encoder_name)
    log.info("=" * 60)

    encoder = build_encoder(encoder_name, checkpoint=cfg.encoder.checkpoint)
    val_ds = CocoEvalImages(
        coco_root=cfg.paths.coco_root,
        split="val",
        image_processor=encoder.processor,
    )

    cache_path = cache_root / f"probe_features_{encoder_name}.npz"
    features, image_ids = extract_features(
        encoder, val_ds, device, cache_path, max_images=max_images,
    )

    # Build label matrix aligned to the order of image_ids
    missing = [i for i in image_ids if int(i) not in image_id_to_label]
    if missing:
        log.warning("%d images have no instances annotation; dropping them",
                    len(missing))
    keep = np.array([int(i) in image_id_to_label for i in image_ids])
    features = features[keep]
    image_ids = image_ids[keep]
    Y = np.stack([image_id_to_label[int(i)] for i in image_ids], axis=0)
    log.info("Features %s, Labels %s", features.shape, Y.shape)

    # Free encoder before fitting the probe — encoder isn't needed past this
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics = train_and_evaluate_probe(features, Y, n_train=n_train, seed=seed)
    log.info("%s: mAP=%.3f  mean_F1=%.3f  (%d/%d classes evaluated)",
             encoder_name, metrics["mAP"], metrics["mean_F1"],
             metrics["n_classes_evaluated"], metrics["n_classes_total"])

    common = dict(
        encoder=encoder_name, task="coco_objects",
        n_classes=metrics["n_classes_evaluated"],
        n_train=metrics["n_train"], n_test=metrics["n_test"],
    )
    return [
        {**common, "metric": "mAP", "score": metrics["mAP"]},
        {**common, "metric": "mean_F1", "score": metrics["mean_F1"]},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encoders", default=",".join(ALL_ENCODERS))
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit val set size per encoder (default: full 5K).")
    parser.add_argument("--n-train", type=int, default=4000,
                        help="Probe training subset size (default: 4000).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV. Default: $CAPTION_ROOT/probe_results.csv.")
    parser.add_argument("--instances-path", type=Path, default=None,
                        help="Override path to COCO instances_val2017.json.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    encoder_names = [e.strip() for e in args.encoders.split(",") if e.strip()]
    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs"
    base_cfg = OmegaConf.load(configs_dir / "base.yaml")

    instances_path = (
        args.instances_path
        if args.instances_path is not None
        else Path(base_cfg.paths.coco_root) / "annotations" / "instances_val2017.json"
    )
    log.info("Loading COCO object labels from %s ...", instances_path)
    image_id_to_label, cat_names = load_coco_object_labels(instances_path)
    log.info("Loaded labels for %d images, %d categories",
             len(image_id_to_label), len(cat_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    cache_root = Path(base_cfg.paths.cache_root)

    all_rows: list[dict] = []
    for encoder_name in encoder_names:
        cfg = load_config(encoder_name, configs_dir)
        try:
            rows = probe_one_encoder(
                encoder_name, cfg, image_id_to_label,
                device=device, cache_root=cache_root,
                n_train=args.n_train, max_images=args.max_images,
                seed=args.seed,
            )
        except Exception as e:
            log.error("Probe failed for %s: %s: %s",
                      encoder_name, type(e).__name__, e)
            continue
        all_rows.extend(rows)

    if not all_rows:
        log.error("No probe results — all encoders failed.")
        return 1

    df = pd.DataFrame(all_rows)
    if args.output is None:
        output_path = Path(base_cfg.paths.caption_root) / "probe_results.csv"
    else:
        output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Wrote %d probe rows to %s", len(df), output_path)

    pivot = df.pivot_table(index="encoder", columns="metric", values="score")
    log.info("Probe summary:\n%s", pivot.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
