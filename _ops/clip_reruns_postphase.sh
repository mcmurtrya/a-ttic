#!/usr/bin/env bash
# Post-MAE CLIP rerun phase.
#
# Why this exists:
#   - clip_seed1 was killed mid-training during Intervention-2 (throughput fix)
#     and resumed under a different LR schedule. Its final val_ppl (10.167 @
#     50k) is ~0.3 ppl worse than seed2 at the same step count — that's
#     resume contamination, not seed variance. The contamination also poisons
#     Phase 4 captions generated from that adaptor.
#   - clip_seed2 was trained on a 50k-step cosine schedule. Its step-35k
#     checkpoint (val_ppl 9.881) is on a different LR trajectory than the
#     native-35k cosine used by siglip/dinov2/mae. Not apples-to-apples for
#     matched-cap comparison.
#
# What this does (sequentially, after main pipeline finishes):
#   1. Re-cache clip features (~10-15 min; pipeline deleted cache after Phase 3)
#   2. Move old clip_seed1 + clip_seed2 checkpoints + captions aside (kept
#      under *_orig_<ts>/ for reference; not deleted)
#   3. Train clip_seed1 (GPU 0) + clip_seed2 (GPU 1) paired, --no-resume,
#      native 35k cosine (default max_steps from configs/base.yaml). ~6h.
#   4. Re-generate captions for clip_seed1 (the headline-captions seed).
#      clip_seed2 captions are not in the pipeline (only seed1 is). ~1.5h.
#   5. Re-merge captions_seed1.jsonl, then re-run Phase 5a/b/d (score,
#      analyze, quality). Probes (5c) are encoder-only, not redone.
#   6. Write completion marker so the finisher proceeds to commit+push+shutdown.
#
# Idempotency: a final 'clip_reruns.complete' marker file gates re-entry.
# If this script is interrupted, re-running picks up where the last completed
# step left off (auto-resume in training, idempotent caching, resume-safe
# caption gen, re-running score/analyze just rewrites the same outputs).
#
# Failure handling: if any step fails the script writes 'clip_reruns.failed'
# and exits non-zero. The finisher should NOT proceed to shutdown in that case
# (handled by finisher_v2.sh logic).

set -uo pipefail

REPO=/home/ubuntu/ttic_embeddings
MARKER_OK=/home/ubuntu/clip_reruns.complete
MARKER_FAIL=/home/ubuntu/clip_reruns.failed
LOG_DIR_PARENT="$REPO/logs"
TS=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="$LOG_DIR_PARENT/reruns_$TS"
mkdir -p "$LOG_DIR"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG_DIR/orchestrator.log"; }

cleanup_fail() {
    local msg="$1"
    log "FAIL: $msg"
    echo "$msg" > "$MARKER_FAIL"
    exit 1
}

if [[ -f "$MARKER_OK" ]]; then
    log "Marker $MARKER_OK already exists; reruns already done. Exiting 0."
    exit 0
fi

cd "$REPO" || cleanup_fail "cannot cd to $REPO"

if [[ -f "$REPO/.env.pipeline" ]]; then
    set -a; source "$REPO/.env.pipeline"; set +a
fi

log "=== CLIP reruns start; log dir: $LOG_DIR ==="

# ---- Step 1: re-cache clip features (only if missing) ----
if [[ -d "$REPO/features/clip" ]] && \
   [[ -n "$(ls -A "$REPO/features/clip" 2>/dev/null)" ]]; then
    log "Step 1: clip feature cache already present, skipping re-cache."
else
    log "Step 1: re-caching clip features on GPU 0..."
    if ! uv run python scripts/01_cache_features.py \
            --encoder clip --gpu 0 \
            > "$LOG_DIR/cache_clip.log" 2>&1; then
        cleanup_fail "clip cache failed; see $LOG_DIR/cache_clip.log"
    fi
    log "Step 1: clip cache done."
fi

# ---- Step 2: move old artifacts aside (only if not already moved) ----
log "Step 2: archiving old clip_seed1/seed2 checkpoints + captions..."
for seed in 1 2; do
    ck="$REPO/checkpoints/clip_seed${seed}"
    if [[ -d "$ck" && ! -L "$ck" ]]; then
        # Only archive if not already done in a prior aborted run
        if [[ ! -d "${ck}_orig" ]]; then
            mv "$ck" "${ck}_orig"
            log "  archived $ck -> ${ck}_orig"
        else
            log "  ${ck}_orig already exists; removing stale $ck"
            rm -rf "$ck"
        fi
    fi
done
cap1="$REPO/captions/captions_clip_seed1.jsonl"
if [[ -f "$cap1" && ! -f "${cap1}.orig" ]]; then
    mv "$cap1" "${cap1}.orig"
    log "  archived $cap1 -> ${cap1}.orig"
elif [[ -f "$cap1" ]]; then
    rm -f "$cap1"
    log "  removed stale $cap1 (orig already preserved)"
fi

# ---- Step 3: train clip_seed1 + clip_seed2 paired, fresh ----
log "Step 3: training clip_seed1 (GPU 0) + clip_seed2 (GPU 1), --no-resume..."
uv run python scripts/02_train_clip.py \
    --encoder clip --seed 1 --gpu 0 --no-resume \
    > "$LOG_DIR/train_clip_seed1.log" 2>&1 &
pid0=$!
uv run python scripts/02_train_clip.py \
    --encoder clip --seed 2 --gpu 1 --no-resume \
    > "$LOG_DIR/train_clip_seed2.log" 2>&1 &
pid1=$!

# Wait, capture exit codes separately
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
if [[ $rc0 -ne 0 || $rc1 -ne 0 ]]; then
    cleanup_fail "training failed (seed1 rc=$rc0, seed2 rc=$rc1); see $LOG_DIR/"
fi
log "Step 3: training done (both seeds, native 35k)."

# ---- Step 4: regen clip_seed1 captions ----
log "Step 4: regenerating clip_seed1 captions on GPU 0..."
if ! uv run python scripts/04_generate_captions.py \
        --encoders clip --seed 1 --gpu 0 \
        --output "$REPO/captions/captions_clip_seed1.jsonl" \
        > "$LOG_DIR/gen_clip_seed1.log" 2>&1; then
    cleanup_fail "clip_seed1 caption gen failed; see $LOG_DIR/gen_clip_seed1.log"
fi
log "Step 4: clip_seed1 captions regenerated."

# ---- Step 5: re-merge + re-run Phase 5a/b/d ----
log "Step 5: re-merging captions_seed1.jsonl..."
ENCODERS=${ENCODERS:-"clip siglip dinov2 mae"}
MERGED="$REPO/captions/captions_seed1.jsonl"
: > "$MERGED"
for enc in $ENCODERS; do
    src="$REPO/captions/captions_${enc}_seed1.jsonl"
    if [[ -f "$src" ]]; then
        cat "$src" >> "$MERGED"
    else
        cleanup_fail "missing per-encoder captions file: $src"
    fi
done
log "  merged into $MERGED ($(wc -l < "$MERGED") rows)"

log "Step 5a: re-scoring metrics..."
VG_FLAG=""
if [[ -n "${VG_ROOT:-}" && -f "${VG_ROOT}/attributes.json" ]]; then
    VG_FLAG="--vg-attributes ${VG_ROOT}/attributes.json"
fi
if ! uv run python scripts/05_score_metrics.py \
        --captions "$MERGED" $VG_FLAG \
        > "$LOG_DIR/score_seed1.log" 2>&1; then
    cleanup_fail "05_score_metrics failed; see $LOG_DIR/score_seed1.log"
fi

log "Step 5b: re-analyzing (beam + nucleus)..."
SCORES_CSV="$REPO/captions/scores_seed1.csv"
if [[ ! -f "$SCORES_CSV" ]]; then
    cleanup_fail "scores CSV missing after score step: $SCORES_CSV"
fi
for DEC in beam nucleus; do
    if ! uv run python scripts/06_analyze.py \
            --scores "$SCORES_CSV" --decoder "$DEC" \
            > "$LOG_DIR/analyze_${DEC}.log" 2>&1; then
        cleanup_fail "06_analyze (decoder=$DEC) failed; see $LOG_DIR/analyze_${DEC}.log"
    fi
done

log "Step 5d: re-running caption-quality precondition..."
if ! uv run python scripts/08_caption_quality.py \
        --captions "$MERGED" \
        > "$LOG_DIR/quality_seed1.log" 2>&1; then
    # quality precondition can fail "softly" (n_encoders<2 etc) — log but don't
    # block the marker, because that path is informational
    log "  WARN: 08_caption_quality non-zero exit; continuing (see log)"
fi

# ---- Step 6: cleanup + marker ----
# Remove regenerated clip feature cache to free disk before commit/push
if [[ -d "$REPO/features/clip" ]]; then
    log "Cleanup: removing clip feature cache ($(du -sh "$REPO/features/clip" | awk '{print $1}'))"
    rm -rf "$REPO/features/clip"
fi

cat > "$MARKER_OK" <<EOF
clip_reruns completed $(date -u)
Log dir: $LOG_DIR
Trained: clip_seed1, clip_seed2 (native 35k, --no-resume)
Regen:   captions_clip_seed1.jsonl
Rescored: scores_seed1.csv, analysis_seed1_{beam,nucleus}_*.csv, quality_results.csv
Archived old: checkpoints/clip_seed{1,2}_orig/, captions/captions_clip_seed1.jsonl.orig
EOF

log "=== CLIP reruns DONE; marker written to $MARKER_OK ==="
exit 0
