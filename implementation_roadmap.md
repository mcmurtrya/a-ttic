# Python implementation roadmap

> **Status: pre-execution design document.** This roadmap was written before the codebase existed, and describes the *intended* phased plan. The actual build diverged in several places — script names were consolidated (`02_extract_features.py` and `03_train_adaptors.py` from the roadmap became `01_cache_features.py` and `02_train_adaptor.py` in the repo), Phase numbering in the roadmap (0–3) is distinct from the "Phase A" cross-seed-analysis follow-up referenced in `RUN_PHASE_A.md` and the report, and several appendix steps (`09_aggregate_seeds.py`, `10_bootstrap_ci.py`, `11_permutation_test.py`) were added after this doc was frozen. For the as-built state of the code, see the *Layout* section of [`README.md`](README.md) and the *Reproducibility Pointers* table in [`encoder_pretraining_caption_report.tex`](encoder_pretraining_caption_report.tex). This file is preserved as the original design rationale.

## Scope

This document describes how to build the codebase that implements the experiment specified in `methods.md`. It covers project structure, dependencies, a phased implementation plan, and the key technical decisions that should be made up front. It does not contain the code itself — those modules are written against this plan in subsequent passes.

The guiding principle is *vertical slice first*: implement the entire pipeline end-to-end for a single encoder (CLIP) before parallelizing across the other three. A working end-to-end pipeline at 5% scale catches integration bugs while they are cheap; scaling a pipeline that works is mechanical, scaling a pipeline that doesn't is debugging four broken pipelines at once.

## Project structure

```
ttic_embeddings/
├── README.md
├── pyproject.toml
├── configs/
│   ├── base.yaml              # shared training, generation, metric config
│   ├── clip.yaml              # encoder-specific overrides
│   ├── siglip.yaml
│   ├── dinov2.yaml
│   └── mae.yaml
├── src/ttic_embeddings/
│   ├── __init__.py
│   ├── data/
│   │   ├── coco.py            # COCO Captions train/val loading
│   │   ├── vg.py              # Visual Genome attribute loading + alignment
│   │   └── cache.py           # disk-backed feature cache
│   ├── encoders/
│   │   ├── base.py            # common encoder interface (load, forward, patches)
│   │   ├── clip.py
│   │   ├── siglip.py
│   │   ├── dinov2.py
│   │   └── mae.py
│   ├── adaptor.py             # 2-layer MLP prefix projector
│   ├── decoder.py             # GPT-2 medium wrapper, frozen
│   ├── train.py               # adaptor training loop with matched-perplexity stop
│   ├── generate.py            # beam + nucleus caption generation
│   ├── metrics/
│   │   ├── parse.py           # cached spaCy parsing
│   │   ├── specificity.py     # adj/noun, VG attribute precision/recall
│   │   ├── spatial.py         # topological + projective lexicon matching
│   │   ├── abstraction.py     # WordNet hypernym depth, scene/object ratio
│   │   └── diversity.py       # MTLD, mean length
│   ├── stats.py               # paired Wilcoxon, BH-FDR, mixed-effects robustness
│   ├── probes.py              # linear probes (object cls, spatial relation cls)
│   ├── caption_quality.py     # CIDEr, SPICE precondition
│   └── utils.py               # seeding, logging, checkpointing
├── scripts/
│   ├── 01_download_data.py    # COCO, VG annotations
│   ├── 02_extract_features.py # cache encoder features per split
│   ├── 03_train_adaptors.py   # 12-run sweep (4 encoders × 3 seeds)
│   ├── 04_generate_captions.py
│   ├── 05_score_metrics.py
│   ├── 06_analyze.py          # primary, secondary, mixed-effects, effect sizes
│   ├── 07_probes.py
│   └── 08_caption_quality.py
└── tests/
    ├── test_metrics.py        # hand-written captions with known scores
    ├── test_stats.py          # synthetic data with known effect sizes
    └── test_adaptor.py        # forward pass, parameter count
```

The split between `src/` modules (importable, testable) and `scripts/` (orchestration, glue) is deliberate. Library code stays small and unit-testable; scripts are end-to-end and can be run in order to reproduce the full experiment.

## Environment

Python 3.11. Pin dependency versions in `pyproject.toml`:

```
torch >= 2.1
transformers >= 4.40        # CLIP, SigLIP, DINOv2, MAE all available
spacy >= 3.7                # en_core_web_lg, NOT _sm
nltk >= 3.8                 # WordNet
lexical-diversity >= 0.1.1  # MTLD
scipy >= 1.11               # Wilcoxon
statsmodels >= 0.14         # multipletests for BH-FDR, mixed-effects
pingouin >= 0.5             # Wilcoxon r effect size out of the box
pandas >= 2.0
numpy < 2.0                 # avoid the 2.x breakage with older deps
pyyaml
omegaconf                   # config composition without Hydra's overhead
tqdm
wandb                       # training logs across the 12 runs
pycocoevalcap               # CIDEr, SPICE
```

Hardware target: single A100 (40 or 80 GB) per training run, with all 12 runs serializable on a single GPU or parallelizable across multiple. UChicago RCC Midway is the default; cloud A100 spot is the fallback. The full project fits in 15–30 A100-hours.

## Phased plan

### Phase 0 — setup and verification (3–5 days)

Goal: every dependency works, every checkpoint loads, every dataset is present, and a single image goes through every encoder cleanly.

- Create the repo, environment, and config skeleton.
- Run `01_download_data.py` to pull COCO Captions train2017 + val2017 and Visual Genome attribute annotations. Verify image counts (118,287 train / 5,000 val) and check that the VG-COCO image-ID alignment file resolves cleanly.
- Write the four encoder modules against a common interface: `load() -> nn.Module`, `forward(images) -> Tensor[B, N_patches, D]`. Verify on five COCO val images per encoder that you get the patch counts you expect (256 / 256 / 256 / 196).
- Smoke-test GPT-2 medium loads and generates from a hand-constructed soft prefix.

This phase is short and unsexy but eliminates the "but I thought SigLIP at 256 res was…" class of bug before it costs you a training run.

### Phase 1 — CLIP vertical slice (5–7 days)

Goal: a complete, end-to-end pipeline that takes COCO images, trains a CLIP adaptor on a small subset, generates captions, and produces a (small) caption file. No metrics or statistics yet — just proof that the spine of the system works.

- Implement `adaptor.py`: 2-layer MLP, GELU activation, output = `k=10` soft prompt tokens at GPT-2's 1024 hidden dim. Verify parameter count is in the ~5M range.
- Implement `train.py` with the language-modeling loss on caption tokens conditioned on the soft prefix. Use AdamW, lr=1e-4, batch size 256, linear warmup + cosine decay. Use matched validation perplexity as stopping criterion (plateau detection over a rolling window).
- Run on a 5,000-image COCO subset for 2,000–5,000 steps. Watch the validation loss curve. Confirm it converges to something reasonable (perplexity well below GPT-2's prior on captions).
- Implement `generate.py`: beam search (size 5) and nucleus sampling (p=0.9). Generate 100 captions from the trained adaptor and read them. They should be coherent English describing image content. If they're nonsense, debug here, not in Phase 3.

The success criterion for this phase is human-readable captions from one encoder. If captions are coherent, the architectural assumptions are validated and scaling is mechanical. If they're not, every later phase is wasted work, so do not move past this checkpoint until the slice works.

### Phase 2 — metric and statistical infrastructure (4–6 days)

Goal: every metric is implemented and validated against hand-constructed cases; the statistical pipeline runs end-to-end on synthetic data.

- Implement `metrics/parse.py` first. Load `en_core_web_lg`, run `nlp.pipe(captions, batch_size=256, n_process=4)`, pickle the parsed `Doc` objects to disk. Parsing dominates evaluation runtime, so caching is essential.
- Implement the four metric scorers. Each takes a list of cached `Doc` objects and returns a per-caption score:
  - `specificity.py` — count attributive `amod` dependents per noun. VG attribute precision/recall is a separate function on the VG-aligned subset.
  - `spatial.py` — `PhraseMatcher` over lemmatized tokens, two lexicons (topological, projective), normalized by token count.
  - `abstraction.py` — head noun extraction, WordNet `min_depth()`, scene/object ratio against Places365 + COCO/VG vocabularies.
  - `diversity.py` — MTLD on concatenated tokens per (encoder × decoder) cell, mean length per caption.
- Write `tests/test_metrics.py` with 10–20 hand-constructed captions whose expected scores you compute by hand. This is the cheapest defense against silent metric bugs.
- Implement `stats.py`: paired Wilcoxon, BH-FDR within family, Wilcoxon r effect size, mixed-effects regression with statsmodels.
- Write `tests/test_stats.py` with synthetic data drawn from known distributions (e.g., paired samples with mean shift = 0.2) and verify the pipeline recovers the right effect size and rejects/accepts at the expected rate.

By the end of Phase 2 the pipeline is fully implemented but only one encoder is trained. Everything that follows is scaling and orchestration.

### Phase 3 — scale to four encoders × three seeds (7–10 days)

Goal: 12 trained adaptors, all checkpointed, with training curves logged.

- Decide feature caching strategy (see "Feature caching" under Key technical decisions). The default plan: cache patch features per (encoder × split) at float16 to disk, processing one encoder at a time to bound peak storage. Total cache footprint ~240 GB at float16 across all four encoders, which fits on a single Midway scratch volume.
- Run `02_extract_features.py` four times (once per encoder). Each run takes 30–60 minutes on an A100 and produces ~50 GB of cached features per encoder.
- Run `03_train_adaptors.py` for the 12-run sweep. With cached features, each adaptor trains in 20–40 minutes on A100. Total wall-clock: 4–8 hours serially, much less if parallelized across GPUs. Log everything to W&B: training loss, validation perplexity, learning rate, gradient norms.
- Verify final validation perplexities are within a tight band across encoders. If one encoder is dramatically worse on perplexity, that's where the matched-perplexity stopping criterion earns its keep — it doesn't penalize encoders that train slower, but it also flags encoders that genuinely cannot reach competitive perplexity, which is itself a finding to report.

This phase is embarrassingly parallel: 12 independent training runs. If you have access to multiple GPUs, run them concurrently.

### Phase 4 — caption generation and metric scoring (3–5 days)

Goal: ~40K captions scored on all four metrics, in a long-format DataFrame ready for analysis.

- Run `04_generate_captions.py`: 5K val × 4 encoders × 2 decoders = 40K captions. Beam search is the slow component (2–5 hours total). For the 3 seeds, use the seed-1 adaptor for the headline run and report seed variability separately so generation cost stays at 4 × 2 × 5K rather than 12 × 2 × 5K.
- Run `05_score_metrics.py`. Output schema: `image_id, encoder, decoder, seed, metric, score`. Long format, pivot to wide on demand for paired tests.
- Validate the schema by spot-checking a handful of (image, encoder, metric) tuples against the captions and the metric definitions.

### Phase 5 — analysis, probes, quality preconditions (4–6 days)

Goal: every test in the `methods.md` statistical analysis section runs to completion and produces tables and figures.

- Run `06_analyze.py`:
  - Primary family: 4 supervision-category contrasts, BH-FDR corrected.
  - Secondary family: 24 pairwise contrasts, BH-FDR corrected.
  - Effect sizes (Wilcoxon r) for every test.
  - Mixed-effects robustness check on the supervision-category contrast.
  - Output: one table per family with columns `metric, contrast, n, r, p, p_adj, claimed_meaningful`.
- Run `07_probes.py`: linear probes on each frozen encoder for COCO object classification and VG spatial relation classification. Report accuracy and connect to caption-metric findings.
- Run `08_caption_quality.py`: CIDEr and SPICE per encoder. Verify all four are within 10% of one another; if not, scope the style claims to "conditional on captioning quality."
- Build the predicted-vs-actual figure: same axes as the predicted-results figure, with actual scores overlaid.
- Curate 20 qualitative examples for the side-by-side figure.

### Phase 6 — integration and writing (ongoing through phases 3–5)

Treat writing as a parallel track, not a final phase. Each completed analysis output should immediately feed into the corresponding section of the proposal/report, so by the end of Phase 5 the document is mostly drafted.

## Key technical decisions

### Feature caching strategy

The largest single decision. Patch features at float32 are: 118K train images × 256 patches × 1024 dims × 4 bytes ≈ 124 GB per encoder. Float16 halves this. Across four encoders at float16: ~250 GB train + ~10 GB val ≈ 260 GB total.

Three viable strategies:

1. *Cache all features for all encoders to disk at float16.* Best wall-clock for adaptor training. Needs ~260 GB scratch space.
2. *Cache one encoder at a time.* Train all three seeds for that encoder, then move to the next. Bounds peak storage at ~70 GB per encoder. This is the recommended default unless storage is plentiful.
3. *Recompute features on the fly during training.* Saves storage entirely but adds a forward pass through a frozen ViT-L per training step. Roughly 2–3× slower training.

Recommended: strategy 2 unless Midway scratch easily holds 300 GB, in which case strategy 1 saves coordination overhead.

### Training framework

Plain PyTorch, no Lightning or HF Trainer. The adaptor training loop is small enough (a few dozen lines) that the abstraction overhead of a framework is not worth the loss of legibility. W&B for logging covers the observability that frameworks usually justify.

### Configuration

OmegaConf with YAML files. One `base.yaml` with shared training/generation/metric config, four encoder-specific YAMLs that override `encoder.name`, `encoder.checkpoint`, `encoder.patch_count`, `encoder.hidden_dim`. Hydra is more powerful but introduces config-composition complexity that this project does not need.

### Reproducibility

Set seeds at the top of every script: `random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False`. Pin all dependency versions. Save the resolved config alongside each checkpoint.

### Logging

W&B project per encoder, run name = `{encoder}_seed{n}`. Log validation perplexity every N steps so the matched-perplexity stopping criterion can be applied programmatically and inspected visually. Save final perplexities to a single CSV that the proposal references directly.

## Risk register

| Risk | Phase that addresses it | Mitigation if risk fires |
|---|---|---|
| Encoder feature shapes don't match expectations | Phase 0 | Catch in smoke test before any training; either adapt the adaptor input layer or downsample patches |
| Adaptor doesn't converge for some encoder | Phase 1 (for CLIP), Phase 3 (others) | Matched-perplexity stop already handles this; report the gap explicitly as a result rather than treating it as a bug |
| GPT-2 medium produces incoherent captions | Phase 1 | Switch to GPT-2 large, increase `k`, or revisit the soft-prefix design before scaling |
| Feature cache exceeds storage | Phase 3 | Fall back to strategy 2 or 3 for caching |
| spaCy parsing too slow at 40K captions | Phase 2 | Already mitigated by `n_process=4` and disk caching; if still slow, parse once and never again |
| Metrics return unexpected zeros (e.g., spatial) | Phase 2 | Hand-written tests catch zero-inflation issues before scaling; the Wilcoxon `zero_method="wilcox"` handles tied zeros correctly |
| CIDEr scores diverge across encoders by >10% | Phase 5 | Triggers the pre-registered "scope style claims to comparable-quality" caveat; not a bug, a finding |
| One encoder's HF checkpoint missing or moved | Phase 0 | Pin checkpoint hashes; have a fallback list (e.g., `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` as CLIP backup) |

## Parallelization opportunities

- Phase 3's 12 training runs are independent. Run as many concurrently as GPU count allows.
- Phase 4's 8 generation runs (4 encoders × 2 decoders) are independent.
- Phase 5's metric scoring is per-caption and trivially parallelizable across CPU cores.
- Statistical analysis is fast and serial — do not bother parallelizing it.

## Minimum viable version

If compute or time is tight, the defensible reduced-scope version is:

- 1 seed instead of 3 (3× saving in Phase 3).
- 1K val images instead of 5K (5× saving in Phase 4 and 5).
- Beam search only, drop nucleus sampling (2× saving in Phase 4).
- Skip linear probes and report only caption-metric results.

This brings the project to ~5–8 GPU-hours total and still defends a primary supervision-category contrast with sufficient power for r ≥ 0.10 detection. The seed-variance argument weakens, but the structural finding remains.
