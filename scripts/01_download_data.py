"""Download COCO Captions + Visual Genome attribute annotations.

Resulting layout:

  $COCO_ROOT/
    annotations/
      captions_train2017.json
      captions_val2017.json
    train2017/   (118,287 .jpg files, ~18 GB)
    val2017/     (5,000 .jpg files, ~778 MB)

  $VG_ROOT/
    attributes.json
    image_data.json

Skips downloads that are already present. Set COCO_ROOT and VG_ROOT
in your environment, or pass --coco-root / --vg-root.

Total size on disk: ~26 GB (COCO 25 GB, VG attributes ~1 GB).
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# COCO 2017 splits and annotations
COCO_DOWNLOADS = [
    ("train2017.zip", "http://images.cocodataset.org/zips/train2017.zip"),
    ("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip"),
    (
        "annotations_trainval2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    ),
]

# Visual Genome attribute annotations + image metadata
VG_DOWNLOADS = [
    (
        "attributes.json.zip",
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip",
    ),
    (
        "image_data.json.zip",
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
    ),
]


def _progress(blocks: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    pct = min(100.0, 100.0 * blocks * block_size / total_size)
    mb = blocks * block_size / 1e6
    total_mb = total_size / 1e6
    sys.stdout.write(f"\r  {pct:5.1f}%  ({mb:7.1f}/{total_mb:.1f} MB)")
    sys.stdout.flush()


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip ] {dest.name} already present")
        return
    print(f"  [fetch] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urlretrieve(url, tmp, _progress)
        sys.stdout.write("\n")
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    print(f"  [unzip] {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def fetch_coco(coco_root: Path) -> None:
    print(f"\nCOCO -> {coco_root}")
    coco_root.mkdir(parents=True, exist_ok=True)
    for filename, url in COCO_DOWNLOADS:
        zip_path = coco_root / filename
        _download(url, zip_path)
        marker = coco_root / (filename.replace(".zip", "") + ".unzipped")
        if not marker.exists():
            _unzip(zip_path, coco_root)
            marker.touch()

    expected = [
        coco_root / "annotations" / "captions_train2017.json",
        coco_root / "annotations" / "captions_val2017.json",
        coco_root / "train2017",
        coco_root / "val2017",
    ]
    for p in expected:
        if not p.exists():
            raise FileNotFoundError(f"Expected {p} after COCO download")
    n_train = sum(1 for _ in (coco_root / "train2017").glob("*.jpg"))
    n_val = sum(1 for _ in (coco_root / "val2017").glob("*.jpg"))
    print(f"  COCO train2017 images: {n_train}")
    print(f"  COCO val2017 images:   {n_val}")
    if n_train < 118_000 or n_val < 4_900:
        print(
            "  WARNING: image counts lower than expected "
            "(118,287 train / 5,000 val). Did extraction finish?"
        )


def fetch_vg(vg_root: Path) -> None:
    print(f"\nVG -> {vg_root}")
    vg_root.mkdir(parents=True, exist_ok=True)
    for filename, url in VG_DOWNLOADS:
        zip_path = vg_root / filename
        _download(url, zip_path)
        marker = vg_root / (filename.replace(".zip", "") + ".unzipped")
        if not marker.exists():
            _unzip(zip_path, vg_root)
            marker.touch()

    expected = [
        vg_root / "attributes.json",
        vg_root / "image_data.json",
    ]
    for p in expected:
        if not p.exists():
            raise FileNotFoundError(f"Expected {p} after VG download")
    print("  VG attributes.json and image_data.json present")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download COCO Captions and Visual Genome attribute annotations."
    )
    parser.add_argument(
        "--coco-root",
        default=os.environ.get("COCO_ROOT", "./data/coco"),
        help="Destination for COCO Captions (default: $COCO_ROOT or ./data/coco)",
    )
    parser.add_argument(
        "--vg-root",
        default=os.environ.get("VG_ROOT", "./data/vg"),
        help="Destination for Visual Genome attributes (default: $VG_ROOT or ./data/vg)",
    )
    parser.add_argument("--skip-coco", action="store_true")
    parser.add_argument("--skip-vg", action="store_true")
    args = parser.parse_args()

    if not args.skip_coco:
        fetch_coco(Path(args.coco_root))
    if not args.skip_vg:
        fetch_vg(Path(args.vg_root))

    print("\nDone. Set these in your shell so downstream scripts find the data:")
    print(f"  export COCO_ROOT={args.coco_root}")
    print(f"  export VG_ROOT={args.vg_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
