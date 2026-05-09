"""Phase 1 — orchestration script for the CLIP vertical slice.

Wires together:
  - configs/base.yaml + configs/clip.yaml (manually merged; OmegaConf
    doesn't honor Hydra's `defaults:` key)
  - encoder (CLIP-L/14 from src/ttic_embeddings/encoders)
  - adaptor (PrefixAdaptor from src/ttic_embeddings/adaptor)
  - frozen GPT-2 medium decoder
  - CocoCaptionPairs train + val DataLoaders
  - the training loop in src/ttic_embeddings/train.py

After training (or when --smoke is passed for a tiny end-to-end run),
loads the best adaptor checkpoint and generates 5 sample captions on
held-out val images so you can eyeball whether captions are coherent —
the Phase 1 success criterion in the roadmap.

Usage (full run):
    uv run python scripts/02_train_clip.py

Usage (tiny end-to-end smoke run, ~5–10 min on CPU):
    uv run python scripts/02_train_clip.py --smoke

Usage with overrides:
    uv run python scripts/02_train_clip.py --max-steps 2000 --batch-size 64
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Make `src/` importable when run as a script
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttic_embeddings.adaptor import build_adaptor                    # noqa: E402
from ttic_embeddings.data.coco import CocoCaptionPairs, CocoEvalImages  # noqa: E402
from ttic_embeddings.encoders import build_encoder                   # noqa: E402
from ttic_embeddings.generate import generate_captions               # noqa: E402
from ttic_embeddings.train import CaptioningModel, train             # noqa: E402
from ttic_embeddings.utils import get_logger, set_seed                # noqa: E402

log = get_logger("train_clip")


# ---------------------------------------------------------------------
# Config loading (manual base-merge; we don't run Hydra)
# ---------------------------------------------------------------------


def load_config(encoder_name: str, configs_dir: Path) -> DictConfig:
    base = OmegaConf.load(configs_dir / "base.yaml")
    enc = OmegaConf.load(configs_dir / f"{encoder_name}.yaml")
    # Strip the Hydra-flavored `defaults:` key — we already loaded base.
    if "defaults" in enc:
        del enc["defaults"]
    return OmegaConf.merge(base, enc)


def apply_smoke_overrides(cfg: DictConfig) -> DictConfig:
    """Tiny end-to-end run: ~50 train steps, small batches, fast val cycle."""
    cfg.train.max_steps = 50
    cfg.train.warmup_steps = 5
    cfg.train.batch_size = 8
    cfg.train.val_every_steps = 25
    cfg.train.val_max_batches = 4
    cfg.train.patience_evals = 100   # don't early-stop a smoke run
    cfg.train.num_workers = 0        # avoid worker spawn cost on tiny batches
    cfg.logging.log_every_steps = 5
    return cfg


def apply_cli_overrides(cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encoder", default="clip",
                        choices=["clip", "siglip", "dinov2", "mae"])
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny end-to-end run for sanity (50 steps, batch=8, "
                             "small data subset).")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging even if WANDB_API_KEY is set.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start training from step 0 even if "
                             "adaptor_latest.pt exists (default: auto-resume "
                             "for spot-reclaim safety).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs"
    cfg = load_config(args.encoder, configs_dir)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    cfg = apply_cli_overrides(cfg, args)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s (CUDA available: %s)",
             device, torch.cuda.is_available())
    log.info("Encoder: %s (%s)", cfg.encoder.name, cfg.encoder.checkpoint)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # ----- Build models ---------------------------------------------
    encoder = build_encoder(cfg.encoder.name, checkpoint=cfg.encoder.checkpoint)
    adaptor = build_adaptor(cfg)
    decoder = GPT2LMHeadModel.from_pretrained(cfg.decoder.name)
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.decoder.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CaptioningModel(encoder, adaptor, decoder, cfg.adaptor.k_soft_tokens)
    log.info("Model assembled. Adaptor: %s", adaptor)

    # ----- Datasets + loaders ---------------------------------------
    train_ds = CocoCaptionPairs(
        coco_root=cfg.paths.coco_root,
        split="train",
        image_processor=encoder.processor,
        tokenizer=tokenizer,
        max_caption_tokens=32,
    )
    val_ds = CocoCaptionPairs(
        coco_root=cfg.paths.coco_root,
        split="val",
        image_processor=encoder.processor,
        tokenizer=tokenizer,
        max_caption_tokens=32,
    )
    if args.smoke:
        # Trim to a few hundred for fast iteration. Subset preserves
        # the dataset's __getitem__ contract for the DataLoader.
        train_ds = Subset(train_ds, range(min(200, len(train_ds))))
        val_ds = Subset(val_ds, range(min(64, len(val_ds))))
        log.info("Smoke run: subset to %d train / %d val items",
                 len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    log.info("Loaders built. train batches: %d, val batches: %d",
             len(train_loader), len(val_loader))

    # ----- Optional wandb ------------------------------------------
    wandb_run = None
    if not args.no_wandb:
        import os
        if "WANDB_API_KEY" in os.environ:
            try:
                import wandb  # noqa: WPS433
                wandb_run = wandb.init(
                    project=cfg.logging.wandb_project,
                    entity=cfg.logging.wandb_entity,
                    name=f"{cfg.encoder.name}_seed{cfg.seed}"
                         + ("_smoke" if args.smoke else ""),
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
                log.info("wandb run started: %s", wandb_run.name)
            except Exception as e:
                log.warning("wandb init failed (%s); continuing without wandb", e)

    # ----- Train ----------------------------------------------------
    output_dir = (
        Path(cfg.paths.checkpoint_root) / f"{cfg.encoder.name}_seed{cfg.seed}"
        if not args.smoke
        else Path(cfg.paths.checkpoint_root) / f"{cfg.encoder.name}_smoke"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoints will be saved under: %s", output_dir)

    summary = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=output_dir,
        device=device,
        wandb_run=wandb_run,
        resume=not args.no_resume,
    )
    log.info("Training summary: %s", summary)

    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()

    # ----- Sample captions on held-out val images ------------------
    log.info("\nGenerating sample captions from the trained adaptor...")
    best_ckpt = output_dir / "adaptor_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.adaptor.load_state_dict(ckpt["adaptor_state_dict"])
        log.info("Loaded best checkpoint (val_ppl=%.3f)", ckpt["val_ppl"])

    val_eval_ds = CocoEvalImages(
        coco_root=cfg.paths.coco_root,
        split="val",
        image_processor=encoder.processor,
    )

    n_samples = 5
    log.info("Sampling %d val images for qualitative inspection:", n_samples)
    for i in range(min(n_samples, len(val_eval_ds))):
        item = val_eval_ds[i]
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        beam_caption = generate_captions(
            model.encoder, model.adaptor, model.decoder, tokenizer,
            pixel_values,
            strategy="beam",
            beam_size=cfg.generate.beam_size,
            max_new_tokens=cfg.generate.max_new_tokens,
        )[0]
        nucleus_caption = generate_captions(
            model.encoder, model.adaptor, model.decoder, tokenizer,
            pixel_values,
            strategy="nucleus",
            top_p=cfg.generate.nucleus_p,
            max_new_tokens=cfg.generate.max_new_tokens,
        )[0]
        refs = val_eval_ds.references(item["image_id"])
        log.info("  image_id %d", item["image_id"])
        log.info("    beam     : %r", beam_caption)
        log.info("    nucleus  : %r", nucleus_caption)
        log.info("    reference: %r", refs[0])

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
