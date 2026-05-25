# Phase A — Seed Aggregation Run Brief

> **How to use this file**: To run Phase A autonomously, point an agent at
> the `AGENT BRIEF` section below. To run it yourself, follow `Setup` →
> `Launch` and read the operator sections.

Generate captions, score, and analyze seeds 2 and 3 for the four existing
encoders, so the supervision-category contrast has the seed-variance
characterization `methods.md` promises. Seed 1 is already done and committed.

This is a follow-up to the main pipeline run (`RUN_PIPELINE.md`); it does
**not** train or run probes — the 12 adaptors already exist and probes are
seed-independent.

---

## AGENT BRIEF

You are running and overseeing Phase A on a rented single-GPU Linux
instance, autonomously. The user (amcmurtry@uchicago.edu) is offline. When
Phase A finishes you commit results to a branch and write a summary.

The runnable script is `scripts/run_phase_a.sh` — **read it first**. It is
single-GPU, resume-safe, and runs its own pre-flight. Your job is to set up
the environment, launch it detached, monitor it, and persist results.

### Environment — verify, then trust

- Repo `ttic_embeddings`, branch `main`. You are in the repo root.
- 1 GPU (A6000 reference; any ≥6 GB CUDA GPU works). VRAM footprint is
  ~1.5 GB — Phase A is memory-bandwidth-bound, not VRAM-bound.
- `uv` installed. `make install-dev` may not have run yet.
- The 8 adaptor checkpoints `checkpoints/{clip,siglip,dinov2,mae}_seed{2,3}/adaptor_best.pt`
  must be present — they ride along in the repo from the main pipeline run.
  If a fresh clone lacks them, **STOP and report** — do NOT retrain.
- Needs COCO val2017 + `annotations/` + VG `attributes.json` + `image_data.json`.
  COCO train2017 (19 GB) is **not** needed — do not download it.
- Estimated wall time: **~12 h on an A6000**.

### Plan

1. **Setup.** `cd` to repo. `make install-dev`. Fetch data into
   `$HOME/data/coco` and `$HOME/data/vg` (val2017 + annotations + VG only).
   Export `COCO_ROOT` / `VG_ROOT`, persist them to `.env.pipeline`, do NOT
   touch `~/.bashrc`.
2. **Pre-flight.** `run_phase_a.sh` checks all 8 checkpoints and dataset
   paths and aborts before using GPU time if anything is missing. If it
   aborts, fix the `MISS` items; do not bypass the check.
3. **Launch detached:**
   ```
   nohup bash -c 'source .env.pipeline && bash scripts/run_phase_a.sh' \
       > phase_a.out 2>&1 &
   ```
   Save the PID. Launch via `Bash` with `run_in_background=true`.
4. **Monitor** every 15–30 min via `tail` of `phase_a.out` and
   `logs/phaseA_*/gen_*.log`. Do not poll faster than necessary.
5. On `Phase A complete.` → commit results (below) and report.

### Monitoring and intervention

- **Stuck job** (no log progress >30 min, `nvidia-smi` util 0%): kill the
  python PID, inspect the per-encoder log, re-run `run_phase_a.sh` — it
  resumes via `04_generate_captions.py`'s skip-completed behavior.
- **OOM / CUDA error**: unlikely at a 1.5 GB footprint; check for a
  competing GPU process first, then re-run (resume-safe).
- **Disk**: `df -h` periodically. Caption JSONLs are small; the `.spacy`
  parse caches are the main growth.
- **CIDEr precondition failing for a seed is NOT an error.** Seed 1 failed
  it at 4 encoders (MAE drags the spread past 10%). Note the per-seed
  pass/fail and continue.
- **Do not edit code** unless a real bug surfaces. If you must, make a
  single focused commit on a fix branch (NOT `main`) explaining the bug.

### Success criteria

- `captions/scores_seed2.csv` and `scores_seed3.csv` exist, ~320K rows each.
- `captions/analysis_seed{2,3}_{beam,nucleus}{,_no_mae}_*.csv` exist.
- `captions/analysis_aggregate_{beam,nucleus}{,_no_mae}_*.csv` exist.
- In each `analysis_aggregate_*_summary.txt`, inspect the `verdict_unstable`
  block — any metric meaningful in some seeds but not others is a finding
  about seed sensitivity and must be called out in your report.

### Commit results (on success only)

`captions/` is gitignored — `git add -f` is required. Create a branch
`phase-a-results-YYYYMMDD` from `main` (do NOT commit to `main`), force-add:
`captions/captions_seed{2,3}.jsonl`, `scores_seed{2,3}.csv`,
`scores_seed{2,3}_diversity.csv`, `analysis_seed{2,3}_*`,
`analysis_aggregate_*`, `quality_results.csv`, `quality_summary.txt`, and
`logs/phaseA_*/`. One commit, descriptive message, **no `Co-Authored-By`
trailer** (project preference). Push the branch. Report branch + SHA.

### Failure handling

`run_phase_a.sh` non-zero exit → re-run (resume-safe). Up to 3 retries for
transient failures. After 3, stop, write a failure report with log paths
and the last commands tried, and do NOT shut the instance down.

### Final report must include

(a) success/failure, (b) branch + commit SHA if pushed, (c) per-seed CIDEr
precondition pass/fail, (d) any `verdict_unstable` metrics from the
aggregates, (e) anomalies and whether the instance was shut down.

### Don't

- Don't retrain adaptors or re-cache features — Phase A is generation only.
- Don't push to `main`; don't `--force`, `--no-verify`, amend, `git clean`,
  or `reset --hard`.
- Don't shut the instance down on failure.
- Don't modify `~/.bashrc`, `~/.gitconfig`, `~/.ssh/`, `~/.claude/`.

---

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
