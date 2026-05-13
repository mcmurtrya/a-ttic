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
from typing import Any

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
from ttic_embeddings.utils import get_logger, set_seed             # noqa: E402

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
    output_handle: Any | None = None,
    completed: set | None = None,
) -> list[dict]:
    """Generate captions for one encoder. Returns list of result dicts.

    If `output_handle` is provided, each row is also written and flushed
    to the file as soon as it's generated (spot-reclaim safe). If
    `completed` is provided, (image_id, encoder, decoder) tuples already
    in the set are skipped — used for resume after a partial run.
    """
    completed = completed if completed is not None else set()
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    n_skipped = 0
    for i in tqdm(range(n), desc=f"{encoder_name}", unit="img"):
        item = val_ds[i]
        image_id = int(item["image_id"])

        # Skip the encoder forward entirely if every decoder for this
        # image is already in the completed set.
        pending = [d for d in decoders if (image_id, encoder_name, d) not in completed]
        if not pending:
            n_skipped += 1
            continue

        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        for strategy in pending:
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
            row = {
                "image_id": image_id,
                "encoder": encoder_name,
                "decoder": strategy,
                "seed": seed,
                "caption": caption,
            }
            rows.append(row)
            if output_handle is not None:
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
            completed.add((image_id, encoder_name, strategy))

    if n_skipped:
        log.info("Skipped %d already-complete images for %s",
                 n_skipped, encoder_name)

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
                        help="Adaptor training seed AND sampling seed (default: 1). "
                             "Drives nucleus-sampling RNG so generated captions are "
                             "reproducible across runs.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit per encoder (default: full val set).")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSONL output path. Default: $CAPTION_ROOT/captions_seed{N}.jsonl.")
    parser.add_argument("--restart", action="store_true",
                        help="Delete any existing output JSONL and start fresh "
                             "(default: append + skip-completed for spot resume).")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (e.g. 0 or 1). Calls "
                             "torch.cuda.set_device(N). Use this instead of "
                             "CUDA_VISIBLE_DEVICES when the environment's "
                             "PyTorch+CUDA combo doesn't honor that variable.")
    args = parser.parse_args()

    encoder_names = [e.strip() for e in args.encoders.split(",") if e.strip()]
    decoders = tuple(d.strip() for d in args.decoders.split(",") if d.strip())
    for d in decoders:
        if d not in ALL_DECODERS:
            raise SystemExit(f"Unknown decoder strategy: {d!r}")

    # Seed RNGs so nucleus sampling is reproducible across runs. Without
    # this, the seed flag only locates the checkpoint — sampling itself
    # was non-deterministic, and the `seed` field stamped per row was
    # only the training seed, not the sampling seed.
    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs"
    base_cfg = OmegaConf.load(configs_dir / "base.yaml")

    if torch.cuda.is_available():
        gpu_idx = args.gpu if args.gpu is not None else 0
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")
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

    # Resume support: scan any existing rows in the output JSONL and skip
    # those (image_id, encoder, decoder) tuples on regeneration. This makes
    # the script spot-reclaim safe — restart re-runs only what's missing.
    completed: set[tuple[int, str, str]] = set()
    if args.restart and output_path.exists():
        log.info("--restart: removing existing %s", output_path)
        output_path.unlink()
    elif output_path.exists():
        n_bad = 0
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Tolerate a partial trailing line from a hard-killed
                # previous run: incomplete JSON gets dropped, the rest
                # of the file is still resumable.
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                completed.add((int(row["image_id"]), row["encoder"], row["decoder"]))
        if n_bad:
            log.warning(
                "Skipped %d malformed JSONL line(s) in %s (likely from a "
                "hard-killed previous run); resume continues with the rest.",
                n_bad, output_path,
            )
        log.info("Found %d already-completed rows in %s; will skip them",
                 len(completed), output_path)

    all_rows: list[dict] = []
    with open(output_path, "a", encoding="utf-8") as f:
        for encoder_name in encoder_names:
            cfg = load_config(encoder_name, configs_dir)
            rows = generate_for_encoder(
                encoder_name, cfg, decoder, tokenizer,
                seed=args.seed, device=device,
                decoders=decoders, max_images=args.max_images,
                output_handle=f, completed=completed,
            )
            all_rows.extend(rows)

    if not all_rows and not completed:
        log.error("No captions generated — likely no trained adaptors found. Train first.")
        return 1

    log.info("Wrote %d new caption rows to %s (%d total including resumed)",
             len(all_rows), output_path, len(all_rows) + len(completed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
