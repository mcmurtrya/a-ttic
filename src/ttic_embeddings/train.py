"""Adaptor training loop.

Composes the three pieces of the captioning system:

    frozen encoder -> trainable adaptor -> frozen GPT-2

into a single nn.Module (CaptioningModel) and runs the language-modeling
loss against COCO captions. Only the adaptor receives gradients; the
encoder is wrapped in @torch.no_grad and the GPT-2 weights have
requires_grad=False.

Stopping criterion: matched validation perplexity (methods.md). Train
until val PPL plateaus for `patience_evals` consecutive validations.
The motivation is in methods.md — fixed-step training would penalize
encoders whose representations are simply harder to align with the
LLM, conflating "harder to adapt" with "produces different captions."

The label-construction logic deserves a careful read: the soft prefix
is concatenated to caption-token embeddings, and the corresponding
label positions are set to -100 so the prefix doesn't contribute to
the LM loss. With the standard left-shift inside HF's loss, the last
prefix position correctly predicts the first caption token, so no
information is wasted.
"""
from __future__ import annotations

import logging
import math
import random
import signal
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class CaptioningModel(nn.Module):
    """Frozen encoder + trainable adaptor + frozen GPT-2 LM head.

    forward(pixel_values, input_ids, attention_mask) -> (loss, logits)

    The model expects already-preprocessed pixel values (the dataset
    handles preprocessing per encoder) and tokenized caption text with
    EOS appended. Padding is handled by the attention_mask.
    """

    def __init__(
        self,
        encoder: nn.Module,
        adaptor: nn.Module,
        decoder: nn.Module,
        k_soft_tokens: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adaptor = adaptor
        self.decoder = decoder
        self.k = k_soft_tokens

        # Freeze encoder + decoder. Encoder is already frozen in
        # VisualEncoder.__init__, but we re-assert defensively so
        # CaptioningModel's contract is self-contained.
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.decoder.eval()

    def train(self, mode: bool = True) -> "CaptioningModel":
        """Set train/eval mode without un-freezing encoder/decoder.

        nn.Module.train() recursively calls .train() on all children.
        We override here so that encoder + decoder stay in eval mode
        (relevant for any dropout/LayerNorm-stats behavior in the
        backbones) regardless of the parent's mode.
        """
        super().train(mode)
        self.encoder.eval()
        self.decoder.eval()
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder forward is @torch.no_grad() in VisualEncoder, so
        # patch_features is detached from the autograd graph. The
        # adaptor's params have requires_grad=True, so its output is
        # gradient-tracked.
        patch_features = self.encoder(pixel_values)
        prefix_embeds = self.adaptor(patch_features)            # (B, k, D_dec)

        text_embeds = self.decoder.transformer.wte(input_ids)    # (B, L, D_dec)
        full_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

        b = input_ids.size(0)
        prefix_attn = torch.ones(
            (b, self.k),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attn = torch.cat([prefix_attn, attention_mask], dim=1)

        # Labels:
        #   prefix positions -> -100 (no loss; HF skips these)
        #   text positions   -> token id, except padding -> -100
        prefix_labels = torch.full(
            (b, self.k),
            fill_value=-100,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        text_labels = input_ids.clone()
        text_labels[attention_mask == 0] = -100
        full_labels = torch.cat([prefix_labels, text_labels], dim=1)

        outputs = self.decoder(
            inputs_embeds=full_embeds,
            attention_mask=full_attn,
            labels=full_labels,
        )
        return outputs.loss, outputs.logits


# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------


class EarlyStopper:
    """Patience-based early stop on a minimization metric (e.g. val PPL)."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.evals_since_best = 0

    def update(self, value: float) -> bool:
        """Return True if training should stop."""
        if value < self.best:
            self.best = value
            self.evals_since_best = 0
            return False
        self.evals_since_best += 1
        return self.evals_since_best >= self.patience


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup from 0 to base lr, then cosine decay to 0."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@contextmanager
def maybe_autocast(use_amp: bool, device_type: str = "cuda"):
    """Yield a torch.autocast context for bf16 if enabled, else a no-op."""
    if use_amp:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            yield
    else:
        yield


@torch.no_grad()
def validate(
    model: CaptioningModel,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None = None,
) -> float:
    """Token-weighted average perplexity over (a subset of) the val set.

    Token-weighted because the per-batch loss is a mean over real
    (non-pad) tokens, not over batches. Multiplying each batch's loss
    by its real-token count and summing gives total log-loss; dividing
    by total tokens at the end gives the correct average per-token
    loss whose exp is perplexity.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with maybe_autocast(use_amp, device.type):
            loss, _ = model(pixel_values, input_ids, attention_mask)
        n_tokens = int(attention_mask.sum().item())
        if n_tokens == 0:
            continue
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    if was_training:
        model.train()

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def _capture_rng_state() -> dict:
    """Snapshot every RNG that could affect training (Python, NumPy, Torch).

    Saved into the checkpoint so that on spot-reclaim resume the next
    sample is the one we would have drawn in the never-killed run —
    not a fresh draw from a re-seeded RNG.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: dict) -> None:
    """Inverse of _capture_rng_state. Tolerates missing CUDA on the resume host."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    adaptor: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    val_ppl: float,
    path: Path,
) -> None:
    """Save adaptor + optimizer + scheduler + RNG state. Atomic via temp + rename."""
    payload = {
        "adaptor_state_dict": adaptor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "val_ppl": val_ppl,
        "rng_state": _capture_rng_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on POSIX, atomic-enough on NTFS


def maybe_resume(
    adaptor: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: Path,
) -> tuple[int, float]:
    """Restore training state from `ckpt_path` if it exists.

    Returns (start_step, last_val_ppl). When the checkpoint is absent,
    returns (0, inf) — the same state as a fresh run.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return 0, float("inf")
    logger.info("Resuming from %s ...", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    adaptor.load_state_dict(ckpt["adaptor_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_step = int(ckpt.get("step", 0))
    last_val_ppl = float(ckpt.get("val_ppl", float("inf")))
    rng_state = ckpt.get("rng_state")
    if rng_state is not None:
        _restore_rng_state(rng_state)
        logger.info("Restored RNG state from checkpoint")
    else:
        logger.warning(
            "Checkpoint at %s has no rng_state; resuming with current RNGs. "
            "Sample order will diverge from the never-killed run.",
            ckpt_path,
        )
    logger.info(
        "Resumed at step %d (last val_ppl=%.3f)", start_step, last_val_ppl
    )
    return start_step, last_val_ppl


class _GracefulExit:
    """Trap SIGTERM/SIGINT so the train loop can save and exit cleanly.

    Spot reclaims send SIGTERM with ~2 minutes warning. We set
    `received = True` from the handler and the train loop checks it
    after each step, saves a checkpoint, and exits with
    stopped_reason='interrupted'.
    """

    def __init__(self) -> None:
        self.received = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                # signal.signal() can only be installed from the main
                # thread; if we're not in main, just no-op.
                pass

    def _handle(self, signum: int, frame: Any) -> None:
        logger.warning(
            "Received signal %d; will save checkpoint and exit at next safe step.",
            signum,
        )
        self.received = True


# ---------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------


def train(
    model: CaptioningModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Any,
    output_dir: Path | str,
    device: torch.device,
    wandb_run: Any | None = None,
    resume: bool = True,
) -> dict:
    """Train the adaptor until val PPL plateaus or max_steps is reached.

    Args:
        model: assembled CaptioningModel (encoder + adaptor + decoder).
        train_loader: yields batches with pixel_values, input_ids,
            attention_mask. Built from CocoCaptionPairs.
        val_loader: same shape as train_loader, on the val split.
        cfg: OmegaConf config with .train, .logging, .adaptor sections.
        output_dir: where to save checkpoints (adaptor_best.pt,
            adaptor_latest.pt) and any per-encoder logs.
        device: torch.device to move model + batches onto.
        wandb_run: optional wandb run to log metrics into; pass None
            to skip wandb logging.

    Returns:
        dict with best_val_ppl, last_val_ppl, total_steps, stopped_reason.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.train()  # respects override: encoder/decoder stay in eval

    optimizer = torch.optim.AdamW(
        model.adaptor.parameters(),
        lr=cfg.train.lr,
        betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = make_lr_scheduler(
        optimizer, cfg.train.warmup_steps, cfg.train.max_steps
    )

    use_amp = (
        cfg.train.precision == "bf16"
        and torch.cuda.is_available()
        and device.type == "cuda"
    )

    early_stopper = EarlyStopper(patience=cfg.train.patience_evals)

    log_every = cfg.logging.log_every_steps
    val_every = cfg.train.val_every_steps
    max_steps = cfg.train.max_steps
    grad_clip = float(getattr(cfg.train, "grad_clip", 1.0))
    val_max_batches = getattr(cfg.train, "val_max_batches", None)

    n_trainable = sum(
        p.numel() for p in model.adaptor.parameters() if p.requires_grad
    )
    logger.info(
        "Starting training. max_steps=%d, val_every=%d, patience=%d, "
        "batch_size=%d, device=%s, amp=%s",
        max_steps, val_every, cfg.train.patience_evals,
        cfg.train.batch_size, device, use_amp,
    )
    logger.info("Trainable params (adaptor only): %s", f"{n_trainable:,}")

    # Resume from `adaptor_latest.pt` if present (spot-reclaim safe).
    latest_ckpt = output_dir / "adaptor_latest.pt"
    best_ckpt = output_dir / "adaptor_best.pt"
    start_step = 0
    last_val_ppl = float("inf")
    best_val_ppl = float("inf")
    if resume:
        start_step, last_val_ppl = maybe_resume(
            model.adaptor, optimizer, scheduler, latest_ckpt
        )
        if best_ckpt.exists():
            try:
                best_payload = torch.load(best_ckpt, map_location="cpu")
                best_val_ppl = float(best_payload.get("val_ppl", float("inf")))
                logger.info("Restored best_val_ppl=%.3f from %s",
                            best_val_ppl, best_ckpt)
            except Exception as e:
                logger.warning("Could not read best_val_ppl from %s: %s",
                               best_ckpt, e)

    graceful = _GracefulExit()

    step = start_step
    train_loss_running = 0.0
    train_loss_count = 0
    t0 = time.time()
    stopped_reason = "max_steps"

    while step < max_steps:
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(use_amp, device.type):
                loss, _ = model(pixel_values, input_ids, attention_mask)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.adaptor.parameters(), max_norm=grad_clip
            )
            optimizer.step()
            scheduler.step()

            train_loss_running += loss.item()
            train_loss_count += 1
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t0
                steps_per_sec = step / max(1.0, elapsed)
                avg_loss = train_loss_running / max(1, train_loss_count)
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "step %6d | loss %.4f | lr %.2e | gnorm %.2f | %.2f step/s",
                    step, avg_loss, lr, float(grad_norm), steps_per_sec,
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        "train/steps_per_sec": steps_per_sec,
                        "step": step,
                    })
                train_loss_running = 0.0
                train_loss_count = 0

            if step % val_every == 0:
                val_ppl = validate(
                    model, val_loader, device, use_amp,
                    max_batches=val_max_batches,
                )
                last_val_ppl = val_ppl
                logger.info("step %6d | val/ppl %.3f", step, val_ppl)
                if wandb_run is not None:
                    wandb_run.log({"val/perplexity": val_ppl, "step": step})

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    save_checkpoint(
                        model.adaptor, optimizer, scheduler, step, val_ppl,
                        output_dir / "adaptor_best.pt",
                    )
                    logger.info(
                        "  best val/ppl so far (%.3f) -> saved %s",
                        val_ppl, output_dir / "adaptor_best.pt",
                    )

                save_checkpoint(
                    model.adaptor, optimizer, scheduler, step, val_ppl,
                    latest_ckpt,
                )

                if early_stopper.update(val_ppl):
                    logger.info(
                        "Early stop: no val/ppl improvement for %d evals. "
                        "Best: %.3f",
                        early_stopper.patience, early_stopper.best,
                    )
                    stopped_reason = "early_stop"
                    break

            if graceful.received:
                logger.warning(
                    "Graceful exit: saving checkpoint at step %d and exiting.",
                    step,
                )
                save_checkpoint(
                    model.adaptor, optimizer, scheduler,
                    step, last_val_ppl,
                    latest_ckpt,
                )
                stopped_reason = "interrupted"
                break

            if step >= max_steps:
                break

        if stopped_reason in ("early_stop", "interrupted"):
            break

    return {
        "best_val_ppl": best_val_ppl,
        "last_val_ppl": last_val_ppl,
        "total_steps": step,
        "stopped_reason": stopped_reason,
    }
