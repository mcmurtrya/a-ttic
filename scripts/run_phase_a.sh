#!/usr/bin/env bash
# Phase A — seed aggregation: generate + score + analyze the remaining seeds.
#
# The 12 adaptors (4 encoders x 3 seeds) are already trained and seed-1
# captions/scores/analysis already exist in the repo. This script fills
# in the remaining seeds (default: 2 and 3) so the supervision-category
# contrast has the seed-variance characterization methods.md promises.
#
# Single-GPU by design — tuned for one rented A6000. It does NOT train,
# cache features, or run linear probes: training is done, and probes are
# seed-independent (already in captions/probe_results.csv).
#
# Phases per seed: generate captions (4 encoders, sequential) -> merge ->
# score metrics -> CIDEr precondition -> statistical analysis (beam +
# nucleus, full 4-encoder and MAE-excluded). After all seeds: cross-seed
# aggregation via 09_aggregate_seeds.py.
#
# Everything is resume-safe: 04_generate_captions.py appends and skips
# completed (encoder, decoder, image) rows, so an interrupted run
# continues where it left off on re-launch.
#
# Prerequisites:
#   - 1 GPU visible
#   - COCO_ROOT / VG_ROOT exported (source .env.pipeline)
#   - checkpoints/{enc}_seed{N}/adaptor_best.pt present for every
#     (encoder, seed) pair this script processes
#   - `make install-dev` already run; you are in the repo root
#   - COCO val2017 + annotations present (train2017 NOT needed)
#
# Usage:
#   source .env.pipeline && bash scripts/run_phase_a.sh
#
# Environment overrides:
#   SEEDS="2 3"                       seeds to process (default: 2 3)
#   ENCODERS="clip siglip dinov2 mae" encoders (default: all four)
#   GPU=0                             GPU index for --gpu
#   SKIP_GENERATE=1                   skip caption generation
#   SKIP_SCORE=1                      skip scoring + quality
#   SKIP_ANALYZE=1                    skip analysis + aggregation

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from repo root

SEEDS=${SEEDS:-"2 3"}
ENCODERS=${ENCODERS:-"clip siglip dinov2 mae"}
GPU=${GPU:-0}
LOG_DIR="logs/phaseA_$(date +%Y%m%d_%H%M%S)"
mkdir -p captions

read -r -a ENC_ARR <<< "$ENCODERS"
read -r -a SEED_ARR <<< "$SEEDS"

echo "===================================================="
echo "TTIC Embeddings — Phase A (seed aggregation)"
echo "Started:  $(date)"
echo "Seeds:    $SEEDS"
echo "Encoders: $ENCODERS"
echo "GPU:      $GPU"
echo "Logs:     $LOG_DIR"
echo "===================================================="

# -------- Pre-flight checks --------
# Verify every checkpoint and dataset path BEFORE burning GPU hours.
echo ""
echo "=== Pre-flight ==="
preflight_ok=1

for seed in "${SEED_ARR[@]}"; do
    for enc in "${ENC_ARR[@]}"; do
        ckpt="checkpoints/${enc}_seed${seed}/adaptor_best.pt"
        if [[ -f "$ckpt" ]]; then
            echo "  ok    $ckpt"
        else
            echo "  MISS  $ckpt"
            preflight_ok=0
        fi
    done
done

if [[ -z "${COCO_ROOT:-}" ]]; then
    echo "  MISS  \$COCO_ROOT not set — run: source .env.pipeline"
    preflight_ok=0
elif [[ ! -d "$COCO_ROOT/val2017" ]]; then
    echo "  MISS  \$COCO_ROOT/val2017 not found ($COCO_ROOT/val2017)"
    preflight_ok=0
else
    echo "  ok    \$COCO_ROOT/val2017"
fi

VG_FLAG=""
if [[ -n "${VG_ROOT:-}" && -f "$VG_ROOT/attributes.json" ]]; then
    VG_FLAG="--vg-attributes $VG_ROOT/attributes.json"
    echo "  ok    \$VG_ROOT/attributes.json (VG attribute metrics enabled)"
else
    echo "  warn  VG attributes not found — VG metrics will be skipped"
fi

if [[ "$preflight_ok" -ne 1 ]]; then
    echo ""
    echo "Pre-flight FAILED. Fix the items marked MISS and re-run."
    exit 1
fi
echo "Pre-flight OK."
mkdir -p "$LOG_DIR"

# -------- Per-seed: generate -> merge -> score -> quality -> analyze ----
for seed in "${SEED_ARR[@]}"; do
    echo ""
    echo "===== Seed $seed ====="
    MERGED="captions/captions_seed${seed}.jsonl"
    SCORES="captions/scores_seed${seed}.csv"

    if [[ -z "${SKIP_GENERATE:-}" ]]; then
        for enc in "${ENC_ARR[@]}"; do
            echo "[$(date +%H:%M:%S)] GEN     $enc seed $seed"
            uv run python scripts/04_generate_captions.py \
                --encoders "$enc" --seed "$seed" --gpu "$GPU" \
                --output "captions/captions_${enc}_seed${seed}.jsonl" \
                2>&1 | tee "$LOG_DIR/gen_${enc}_seed${seed}.log"
        done
        echo "[$(date +%H:%M:%S)] MERGE   -> $MERGED"
        : > "$MERGED"
        for enc in "${ENC_ARR[@]}"; do
            cat "captions/captions_${enc}_seed${seed}.jsonl" >> "$MERGED"
        done
        echo "[$(date +%H:%M:%S)] MERGED  ($(wc -l < "$MERGED") rows)"
    fi

    if [[ -z "${SKIP_SCORE:-}" ]]; then
        echo "[$(date +%H:%M:%S)] SCORE   seed $seed"
        uv run python scripts/05_score_metrics.py \
            --captions "$MERGED" $VG_FLAG \
            2>&1 | tee "$LOG_DIR/score_seed${seed}.log"
        echo "[$(date +%H:%M:%S)] QUALITY seed $seed (CIDEr precondition)"
        uv run python scripts/08_caption_quality.py \
            --captions "$MERGED" \
            2>&1 | tee "$LOG_DIR/quality_seed${seed}.log"
    fi

    if [[ -z "${SKIP_ANALYZE:-}" ]]; then
        if [[ ! -f "$SCORES" ]]; then
            echo "  WARN  $SCORES missing — skipping analysis for seed $seed"
            continue
        fi
        for DEC in beam nucleus; do
            echo "[$(date +%H:%M:%S)] ANALYZE seed $seed $DEC (4-encoder)"
            uv run python scripts/06_analyze.py \
                --scores "$SCORES" --decoder "$DEC" \
                2>&1 | tee "$LOG_DIR/analyze_seed${seed}_${DEC}.log"
            echo "[$(date +%H:%M:%S)] ANALYZE seed $seed $DEC (MAE excluded)"
            uv run python scripts/06_analyze.py \
                --scores "$SCORES" --decoder "$DEC" \
                --self-encoders dinov2 --output-suffix _no_mae \
                2>&1 | tee "$LOG_DIR/analyze_seed${seed}_${DEC}_no_mae.log"
        done
    fi
done

# -------- Cross-seed aggregation (picks up seed 1 already in repo) ------
if [[ -z "${SKIP_ANALYZE:-}" ]]; then
    echo ""
    echo "=== Cross-seed aggregation ==="
    for DEC in beam nucleus; do
        for SUFFIX in "" "_no_mae"; do
            echo "[$(date +%H:%M:%S)] AGGREGATE decoder=$DEC suffix=${SUFFIX:-(none)}"
            uv run python scripts/09_aggregate_seeds.py \
                --analysis-dir captions --decoder "$DEC" --suffix "$SUFFIX" \
                2>&1 | tee -a "$LOG_DIR/aggregate.log"
        done
    done
fi

echo ""
echo "===================================================="
echo "Phase A complete."
echo "Finished: $(date)"
echo "Logs:     $LOG_DIR"
echo ""
echo "Key outputs:"
for seed in "${SEED_ARR[@]}"; do
    echo "  Seed $seed:  captions/captions_seed${seed}.jsonl  scores_seed${seed}.csv"
done
echo "  Per-seed analysis:    captions/analysis_seed{N}_{beam,nucleus}{,_no_mae}_*.csv"
echo "  Cross-seed aggregate: captions/analysis_aggregate_{beam,nucleus}{,_no_mae}_*.csv"
echo "  Aggregate summaries:  captions/analysis_aggregate_*_summary.txt"
echo "===================================================="
