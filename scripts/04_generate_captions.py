"""Phase 4 — generate captions for all (encoder, decoder, image) combinations.

Reads:
    configs/{encoder}.yaml                              encoder configs
    {checkpoint_root}/{encoder}_seed{N}/adaptor_best.pt trained adaptor
    $COCO_ROOT/val2017/                                 val images

Writes:
    {caption_root}/captions_seed{N}.jsonl
        one row per (image_id, encoder, decoder), with shape:
        {"image_id": 139, "encoder": "clip", "decoder": "beam",
         "seed": 1, "caption": "A dog sits on a chair."}

Encoders are loaded one at a time and freed before loading the next, to
keep peak GPU memory bounded. The decoder (GPT-2) is loaded once and
reused across encoders.

Usage:
    uv run python scripts/04_generate_captions.py
    uv run python scripts/04_generate_captions.py --encoders clip,siglip
    uv run python scripts/04_generate_captions.py --max-images 100  # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttic_embeddings.adaptor import build_adaptor                  # noqa: E402
from ttic_embeddings.data.coco import CocoEvalImages               # noqa: E402
from ttic_embeddings.encoders import build_encoder                 # noqa: E402
from ttic_embeddings.generate import generate_captions             # noqa: E402
from ttic_embeddings.utils import get_logger                       # noqa: E402

log = get_logger("generate_captions")

ALL_ENCODERS = ("clip", "siglip", "dinov2", "mae")
ALL_DECODERS = ("beam", "nucleus")


def load_config(encoder_name: str, configs_dir: Path) -> DictConfig:
    base = OmegaConf.load(configs_dir / "base.yaml")
    enc = OmegaConf.load(configs_dir / f"{encoder_name}.yaml")
    if "defaults" in enc:
        del enc["defaults"]
    return OmegaConf.merge(base, enc)


def find_checkpoint(checkpoint_root: Path, encoder_name: str, seed: int) -> Path | None:
    """Search for the trained adaptor checkpoint in a few likely locations."""
    candidates = [
        checkpoint_root / f"{encoder_name}_seed{seed}" / "adaptor_best.pt",
        checkpoint_root / encoder_name / f"seed{seed}" / "adaptor_best.pt",
        checkpoint_root / encoder_name / "adaptor_best.pt",  # single-seed runs
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def generate_for_encoder(
    encoder_name: str,
    cfg: DictConfig,
    decoder: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    seed: int,
    device: torch.device,
    decoders: tuple[str, ...],
    max_images: int | None,
) -> list[dict]:
    """Generate captions for one encoder. Returns list of result dicts."""
    log.info("=" * 60)
    log.info("Encoder: %s", encoder_name)
    log.info("=" * 60)

    checkpoint_root = Path(cfg.paths.checkpoint_root)
    ckpt_path = find_checkpoint(checkpoint_root, encoder_name, seed)
    if ckpt_path is None:
        log.warning(
            "No trained adaptor found for %s (seed %d). Searched under %s. "
            "Skipping this encoder.",
            encoder_name, seed, checkpoint_root,
        )
        return []

    encoder = build_encoder(encoder_name, checkpoint=cfg.encoder.checkpoint)
    encoder.to(device)
    log.info("Loaded encoder: %r", encoder)

    adaptor = build_adaptor(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    adaptor.load_state_dict(ckpt["adaptor_state_dict"])
    adaptor.to(device)
    adaptor.eval()
    log.info(
        "Loaded adaptor from %s (val_ppl=%.3f, training step=%d)",
        ckpt_path, ckpt.get("val_ppl", float("nan")), ckpt.get("step", -1),
    )

    val_ds = CocoEvalImages(
        coco_root=cfg.paths.coco_root,
        split="val",
        image_processor=encoder.processor,
    )
    n = len(val_ds) if max_images is None else min(max_images, len(val_ds))
    log.info("Generating on %d val images, decoders: %s", n, ", ".join(decoders))

    rows: list[dict] = []
    for i in tqdm(range(n), desc=f"{encoder_name}", unit="img"):
        item = val_ds[i]
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        for strategy in decoders:
            kwargs = {}
            if strategy == "beam":
                kwargs["beam_size"] = cfg.generate.beam_size
            elif strategy == "nucleus":
                kwargs["top_p"] = cfg.generate.nucleus_p
            caption = generate_captions(
                encoder, adaptor, decoder, tokenizer,
                pixel_values,
                strategy=strategy,
                max_new_tokens=cfg.generate.max_new_tokens,
                **kwargs,
            )[0]
            rows.append({
                "image_id": int(item["image_id"]),
                "encoder": encoder_name,
                "decoder": strategy,
                "seed": seed,
                "caption": caption,
            })

    # Free GPU memory before loading next encoder
    del encoder, adaptor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log.info("Generated %d caption rows for %s", len(rows), encoder_name)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--encoders", default=",".join(ALL_ENCODERS),
        help="Comma-separated encoder names (default: all four).",
    )
    parser.add_argument(
        "--decoders", default=",".join(ALL_DECODERS),
        help="Comma-separated decoding strategies (default: beam,nucleus).",
    )
    parser.add_argument("--seed", type=int, default=1,
                        help="Adaptor training seed to use (default: 1).")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit per encoder (default: full val set).")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSONL output path. Default: $CAPTION_ROOT/captions_seed{N}.jsonl.")
    args = parser.parse_args()

    encoder_names = [e.strip() for e in args.encoders.split(",") if e.strip()]
    decoders = tuple(d.strip() for d in args.decoders.split(",") if d.strip())
    for d in decoders:
        if d not in ALL_DECODERS:
            raise SystemExit(f"Unknown decoder strategy: {d!r}")

    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs"
    base_cfg = OmegaConf.load(configs_dir / "base.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s (CUDA: %s)", device, torch.cuda.is_available())
    log.info("Encoders: %s", encoder_names)
    log.info("Decoders: %s", decoders)
    log.info("Seed: %d", args.seed)
    log.info("Max images per encoder: %s", args.max_images or "ALL")

    log.info("Loading shared decoder: %s", base_cfg.decoder.name)
    decoder = GPT2LMHeadModel.from_pretrained(base_cfg.decoder.name).to(device)
    decoder.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(base_cfg.decoder.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.output is None:
        caption_root = Path(base_cfg.paths.caption_root)
        caption_root.mkdir(parents=True, exist_ok=True)
        output_path = caption_root / f"captions_seed{args.seed}.jsonl"
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for encoder_name in encoder_names:
        cfg = load_config(encoder_name, configs_dir)
        rows = generate_for_encoder(
            encoder_name, cfg, decoder, tokenizer,
            seed=args.seed, device=device,
            decoders=decoders, max_images=args.max_images,
        )
        all_rows.extend(rows)

    if not all_rows:
        log.error("No captions generated — likely no trained adaptors found. Train first.")
        return 1

    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d caption rows to %s", len(all_rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
