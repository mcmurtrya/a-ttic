# Phase A — Seed Aggregation Run Brief

Generate captions, score, and analyze seeds 2 and 3 for the four existing
encoders, so the supervision-category contrast has the seed-variance
characterization `methods.md` promises. Seed 1 is already done and committed.

This is a follow-up to the main pipeline run (`RUN_PIPELINE.md`); it does
**not** train or run probes — the 12 adaptors already exist and probes are
seed-independent.

## What it does

`scripts/run_phase_a.sh`, single-GPU, for each of seeds 2 and 3:

1. Generate captions — 4 encoders × 2 decoders × 5K val images
2. Merge per-encoder JSONL → `captions/captions_seed{N}.jsonl`
3. Score the four caption-style metric families → `scores_seed{N}.csv`
4. CIDEr precondition → `quality_results.csv` / `quality_summary.txt`
5. Statistical analysis — beam + nucleus, full 4-encoder and MAE-excluded

Then a cross-seed aggregation (`09_aggregate_seeds.py`) pools seeds 1–3
into `analysis_aggregate_*` files with per-metric mean/SD of the effect
size and a `claimed_robust` flag (meaningful in every seed, not just one).

## Hardware and cost

- 1× A6000 (48 GB) is the reference target. Any ≥6 GB CUDA GPU works.
- Estimated wall-clock on A6000: **~12 hours** (≈$6–10 cloud).
  Per-encoder, per-seed: CLIP ~90 min, SigLIP ~90 min, DINOv2 ~20 min,
  MAE ~140 min. CPU scoring (spaCy) adds ~45 min total.
- VRAM footprint is ~1.5 GB — the run is memory-bandwidth-bound, not
  VRAM-bound, so a larger card mainly helps via bandwidth.

## Setup

```bash
cd ttic_embeddings
make install-dev                      # uv sync + spaCy model + NLTK assets

# Data: Phase A needs COCO val2017 + annotations + VG attributes ONLY.
# COCO train2017 (19 GB) is NOT needed — it is training-only.
export COCO_ROOT=$HOME/data/coco
export VG_ROOT=$HOME/data/vg
mkdir -p "$COCO_ROOT" "$VG_ROOT"
# Fetch val2017, annotations_trainval2017, and VG attributes.json +
# image_data.json into those roots. (See the Makefile `data` target;
# you can skip the train2017 download.)
```

The 8 required checkpoints — `checkpoints/{clip,siglip,dinov2,mae}_seed{2,3}/adaptor_best.pt`
— must be present. They were committed by the main pipeline run; if the
rental is a fresh clone, make sure they came down with it.

## Launch

```bash
source .env.pipeline    # or export COCO_ROOT / VG_ROOT manually
bash scripts/run_phase_a.sh
```

The script runs a pre-flight check first — it verifies all 8 checkpoints
and the dataset paths and aborts before using any GPU time if something
is missing. Resume-safe: if interrupted, just re-run; `04_generate_captions.py`
skips already-completed rows.

Detached run (recommended for the ~12 h duration):

```bash
nohup bash -c 'source .env.pipeline && bash scripts/run_phase_a.sh' \
    > phase_a.out 2>&1 &
```

Monitor with `tail -f phase_a.out` or `tail -f logs/phaseA_*/gen_*.log`.

### Useful overrides

- `SEEDS="2"` — process one seed only
- `ENCODERS="dinov2"` — single encoder (debugging)
- `SKIP_GENERATE=1` — re-run scoring/analysis on existing captions
- `GPU=1` — use a different GPU index

## On completion

Verify, then persist:

- `captions/scores_seed{2,3}.csv` exist and have ~320K rows each
- `captions/analysis_aggregate_*_summary.txt` — check the `verdict_unstable`
  block: any metric meaningful in some seeds but not others is a finding
  about seed sensitivity, not a bug
- `captions/quality_summary.txt` per seed — confirm whether the CIDEr
  precondition passes or fails per seed (it failed at 4 encoders for seed 1)

`captions/` is gitignored — commit results with `git add -f` (the
`captions/captions_*.jsonl`, `scores_*.csv`, `analysis_*.csv`,
`analysis_aggregate_*`, `quality_*` files, and `logs/phaseA_*/`).
Push to a `phase-a-results-YYYYMMDD` branch, not directly to `main`.
