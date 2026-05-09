#!/usr/bin/env bash
# End-to-end pipeline driver, optimized for a 2-GPU instance.
#
# Trains all 4 encoders × 3 seeds = 12 adaptors via paired GPU dispatch.
# Generates captions only from seed 1 (the headline configuration per
# implementation_roadmap.md) to bound generation cost. Then runs scoring,
# stats analysis, linear probes, and the CIDEr quality precondition.
#
# All training and generation steps auto-resume on re-run, so you can
# kill this script and restart it without losing meaningful progress.
#
# Prerequisites:
#   - 2 GPUs visible at CUDA_VISIBLE_DEVICES=0 and 1
#   - `make install-dev` already run
#   - `make data` already run (COCO + VG present)
#   - You're in the repo root
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#
# Environment overrides (optional):
#   SEEDS="1 2 3"            seeds to train (default: 1 2 3)
#   GENERATE_SEED="1"        seed used for caption generation (default: 1)
#   ENCODERS="clip siglip dinov2 mae"
#   SKIP_TRAIN=1             skip training phase
#   SKIP_GENERATE=1          skip caption generation
#   SKIP_PROBES=1            skip linear probes
#   SKIP_QUALITY=1           skip CIDEr precondition

set -euo pipefail

# -------- Configuration --------
SEEDS=${SEEDS:-"1 2 3"}
GENERATE_SEED=${GENERATE_SEED:-1}
ENCODERS=${ENCODERS:-"clip siglip dinov2 mae"}
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p captions

cd "$(dirname "$0")/.."   # always run from repo root

echo "===================================================="
echo "TTIC Embeddings — full pipeline"
echo "Started:  $(date)"
echo "Logs:     $LOG_DIR"
echo "Seeds:    $SEEDS"
echo "Encoders: $ENCODERS"
echo "===================================================="

# -------- Helpers --------

# run two training jobs in parallel, one per GPU, wait for both
run_train_pair() {
    local enc0="$1" seed0="$2" enc1="$3" seed1="$4"
    local tag0="${enc0}_seed${seed0}" tag1="${enc1}_seed${seed1}"
    echo ""
    echo "[$(date +%H:%M:%S)] TRAIN  GPU0=$tag0   GPU1=$tag1"

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_train_clip.py \
        --encoder "$enc0" --seed "$seed0" \
        > "$LOG_DIR/train_${tag0}.log" 2>&1 &
    local pid0=$!

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_train_clip.py \
        --encoder "$enc1" --seed "$seed1" \
        > "$LOG_DIR/train_${tag1}.log" 2>&1 &
    local pid1=$!

    wait "$pid0" "$pid1"
    echo "[$(date +%H:%M:%S)] DONE   $tag0 + $tag1"
}

# run two caption-generation jobs in parallel, one per GPU
run_gen_pair() {
    local enc0="$1" enc1="$2" seed="$3"
    echo ""
    echo "[$(date +%H:%M:%S)] GEN    GPU0=$enc0   GPU1=$enc1   (seed $seed)"

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/04_generate_captions.py \
        --encoders "$enc0" --seed "$seed" \
        --output "captions/captions_${enc0}_seed${seed}.jsonl" \
        > "$LOG_DIR/gen_${enc0}_seed${seed}.log" 2>&1 &
    local pid0=$!

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/04_generate_captions.py \
        --encoders "$enc1" --seed "$seed" \
        --output "captions/captions_${enc1}_seed${seed}.jsonl" \
        > "$LOG_DIR/gen_${enc1}_seed${seed}.log" 2>&1 &
    local pid1=$!

    wait "$pid0" "$pid1"
    echo "[$(date +%H:%M:%S)] DONE   $enc0 + $enc1 generation"
}

# run two probes in parallel, one per GPU, with separate output files
run_probe_pair() {
    local enc0="$1" enc1="$2"
    echo ""
    echo "[$(date +%H:%M:%S)] PROBE  GPU0=$enc0   GPU1=$enc1"

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/07_probes.py \
        --encoders "$enc0" \
        --output "captions/probe_${enc0}.csv" \
        > "$LOG_DIR/probe_${enc0}.log" 2>&1 &
    local pid0=$!

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/07_probes.py \
        --encoders "$enc1" \
        --output "captions/probe_${enc1}.csv" \
        > "$LOG_DIR/probe_${enc1}.log" 2>&1 &
    local pid1=$!

    wait "$pid0" "$pid1"
    echo "[$(date +%H:%M:%S)] DONE   $enc0 + $enc1 probe"
}

# -------- Phase 0: smoke test --------

echo ""
echo "=== Phase 0: smoke test ==="
uv run python scripts/00_smoke_test.py 2>&1 | tee "$LOG_DIR/smoke.log"

# -------- Phase 1-3: train all 12 (encoder, seed) combinations --------

if [[ -z "${SKIP_TRAIN:-}" ]]; then
    echo ""
    echo "=== Phases 1-3: train 12 adaptors (4 encoders × 3 seeds) on 2 GPUs ==="
    # Convert space-separated strings to arrays
    read -r -a ENC_ARR <<< "$ENCODERS"
    read -r -a SEED_ARR <<< "$SEEDS"

    # Pair encoders: (enc[0], enc[1]) and (enc[2], enc[3]) per seed.
    # For 4 encoders this is 2 pairs per seed × 3 seeds = 6 pairs total.
    for seed in "${SEED_ARR[@]}"; do
        for ((i = 0; i < ${#ENC_ARR[@]}; i += 2)); do
            run_train_pair \
                "${ENC_ARR[$i]}" "$seed" \
                "${ENC_ARR[$((i + 1))]}" "$seed"
        done
    done
else
    echo "Skipping training phase (SKIP_TRAIN set)."
fi

# -------- Phase 4: caption generation (seed 1 only, headline) --------

if [[ -z "${SKIP_GENERATE:-}" ]]; then
    echo ""
    echo "=== Phase 4: caption generation, seed $GENERATE_SEED ==="
    read -r -a ENC_ARR <<< "$ENCODERS"
    for ((i = 0; i < ${#ENC_ARR[@]}; i += 2)); do
        run_gen_pair \
            "${ENC_ARR[$i]}" \
            "${ENC_ARR[$((i + 1))]}" \
            "$GENERATE_SEED"
    done

    # Merge per-encoder JSONL files into one
    echo ""
    echo "[$(date +%H:%M:%S)] Merging per-encoder JSONL files..."
    MERGED="captions/captions_seed${GENERATE_SEED}.jsonl"
    : > "$MERGED"
    for enc in "${ENC_ARR[@]}"; do
        cat "captions/captions_${enc}_seed${GENERATE_SEED}.jsonl" >> "$MERGED"
    done
    echo "[$(date +%H:%M:%S)] Merged into $MERGED ($(wc -l < "$MERGED") rows)"
else
    echo "Skipping caption generation (SKIP_GENERATE set)."
fi

# -------- Phase 5a: score metrics --------

echo ""
echo "=== Phase 5a: score caption metrics ==="
SCORE_INPUT="captions/captions_seed${GENERATE_SEED}.jsonl"
if [[ -f "$SCORE_INPUT" ]]; then
    VG_FLAG=""
    if [[ -n "${VG_ROOT:-}" && -f "$VG_ROOT/attributes.json" ]]; then
        VG_FLAG="--vg-attributes $VG_ROOT/attributes.json"
    fi
    uv run python scripts/05_score_metrics.py \
        --captions "$SCORE_INPUT" $VG_FLAG \
        2>&1 | tee "$LOG_DIR/score_seed${GENERATE_SEED}.log"
else
    echo "No captions JSONL found at $SCORE_INPUT — skipping scoring."
fi

# -------- Phase 5b: statistical analysis --------

echo ""
echo "=== Phase 5b: statistical analysis ==="
SCORES_CSV="captions/scores_seed${GENERATE_SEED}.csv"
if [[ -f "$SCORES_CSV" ]]; then
    for DEC in beam nucleus; do
        echo ""
        echo "[$(date +%H:%M:%S)] Analyzing decoder=$DEC"
        uv run python scripts/06_analyze.py \
            --scores "$SCORES_CSV" \
            --decoder "$DEC" \
            2>&1 | tee "$LOG_DIR/analyze_${DEC}.log"
    done
else
    echo "No scores CSV at $SCORES_CSV — skipping analysis."
fi

# -------- Phase 5c: linear probes (independent of training) --------

if [[ -z "${SKIP_PROBES:-}" ]]; then
    echo ""
    echo "=== Phase 5c: linear probes ==="
    read -r -a ENC_ARR <<< "$ENCODERS"
    for ((i = 0; i < ${#ENC_ARR[@]}; i += 2)); do
        run_probe_pair \
            "${ENC_ARR[$i]}" \
            "${ENC_ARR[$((i + 1))]}"
    done

    # Merge per-encoder probe results
    echo ""
    echo "[$(date +%H:%M:%S)] Merging probe CSVs..."
    uv run python -c "
import pandas as pd, glob
dfs = [pd.read_csv(f) for f in sorted(glob.glob('captions/probe_*.csv'))
       if 'results' not in f]
if dfs:
    pd.concat(dfs, ignore_index=True).to_csv('captions/probe_results.csv', index=False)
    print('Merged into captions/probe_results.csv')
"
else
    echo "Skipping linear probes (SKIP_PROBES set)."
fi

# -------- Phase 5d: caption quality precondition --------

if [[ -z "${SKIP_QUALITY:-}" ]]; then
    echo ""
    echo "=== Phase 5d: caption quality precondition (CIDEr / SPICE) ==="
    if [[ -f "$SCORE_INPUT" ]]; then
        uv run python scripts/08_caption_quality.py \
            --captions "$SCORE_INPUT" \
            2>&1 | tee "$LOG_DIR/quality_seed${GENERATE_SEED}.log"
    else
        echo "No captions JSONL — skipping quality check."
    fi
else
    echo "Skipping caption quality (SKIP_QUALITY set)."
fi

# -------- Summary --------

echo ""
echo "===================================================="
echo "Pipeline complete."
echo "Finished: $(date)"
echo "Logs:     $LOG_DIR"
echo ""
echo "Key outputs:"
echo "  Trained adaptors: checkpoints/{encoder}_seed{N}/adaptor_best.pt"
echo "  Captions:         $SCORE_INPUT"
echo "  Scores (long):    $SCORES_CSV"
echo "  Analysis:         captions/analysis_seed${GENERATE_SEED}_{beam,nucleus}_*.csv"
echo "  Probe results:    captions/probe_results.csv"
echo "  Quality results:  captions/quality_results.csv"
echo "  Quality summary:  captions/quality_summary.txt"
echo "===================================================="
