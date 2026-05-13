#!/usr/bin/env bash
# Eval-chain smoke test on a snapshot of in-progress clip_seed1 captions.
# Purpose: catch bugs in 05/06/08 scripts BEFORE the full pipeline gets there.
# CPU-only; runs in parallel with everything else.

set -uo pipefail
SIDE_DIR=/home/ubuntu/sidejobs
SMOKE_DIR=$SIDE_DIR/eval_smoke
mkdir -p "$SMOKE_DIR"
exec > "$SMOKE_DIR/eval_smoke.log" 2>&1

source /home/ubuntu/ttic_embeddings/.env.pipeline
cd /home/ubuntu/ttic_embeddings

echo "=== eval_smoke start $(date -u) ==="
SNAP="$SIDE_DIR/clip_seed1_snapshot.jsonl"
wc -l "$SNAP"

# --- 05: score metrics (uses VG attributes for vg_attr_* metrics) ---
echo ""
echo "=== 05_score_metrics ==="
uv run python scripts/05_score_metrics.py \
    --captions "$SNAP" \
    --output "$SMOKE_DIR/scores.csv" \
    --vg-attributes /home/ubuntu/data/vg/attributes.json \
    --vg-image-data /home/ubuntu/data/vg/image_data.json
echo "[$(date -u +%H:%M:%S)] 05 done (rc=$?). scores.csv:"
ls -la "$SMOKE_DIR/scores.csv" 2>/dev/null || echo "NO scores.csv"

# --- 06: analyze (will fail on single-encoder data, but exposes arg/IO bugs) ---
echo ""
echo "=== 06_analyze ==="
uv run python scripts/06_analyze.py \
    --scores "$SMOKE_DIR/scores.csv" \
    --output-dir "$SMOKE_DIR/analysis" || \
    echo "[$(date -u +%H:%M:%S)] 06 returned non-zero (expected: single-encoder data can't run paired tests)"

# --- 08: caption quality (CIDEr) ---
echo ""
echo "=== 08_caption_quality ==="
uv run python scripts/08_caption_quality.py \
    --captions "$SNAP" \
    --output-dir "$SMOKE_DIR/quality" \
    --coco-root /home/ubuntu/data/coco
echo "[$(date -u +%H:%M:%S)] 08 done (rc=$?)"

echo ""
echo "=== eval_smoke end $(date -u) ==="
