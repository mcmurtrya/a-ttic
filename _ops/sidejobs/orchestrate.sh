#!/usr/bin/env bash
# Sidejobs orchestrator — uses GPU 1 idle window during siglip_seed3 solo phase.
# Started: 2026-05-11 14:30 UTC.
#
# Sequence (after siglip_seed2 finishes, all on GPU 1):
#   1. MAE pre-flight: cache MAE features (~22 min, ~47 GiB) then
#      train mae_seed99 for 5000 steps (~48 min). Sacrificial: cache +
#      checkpoint are deleted at the end.
#   2. siglip_seed1 caption gen (~83 min). Output is resume-safe so
#      the pipeline's later Phase 4 will skip these rows.
#   3. Linear probes for clip + siglip (~10 min).
#
# Disk gate: aborts before stage 1's cache if df shows < 50 GiB free.
# Must complete before ~20:22 UTC (dinov2 paired phase reclaims GPU 1).

set -uo pipefail

SIDE_DIR=/home/ubuntu/sidejobs
mkdir -p "$SIDE_DIR"
exec > "$SIDE_DIR/orchestrate.log" 2>&1

source /home/ubuntu/ttic_embeddings/.env.pipeline
cd /home/ubuntu/ttic_embeddings

echo "=== orchestrate.sh start $(date -u) pid=$$ ==="

# --- Wait for GPU 1 training to finish ---
echo "[$(date -u +%H:%M:%S)] Waiting for GPU 1 training to finish..."
while pgrep -f "02_train_clip.py.*--gpu 1" >/dev/null 2>&1; do
    sleep 20
done
echo "[$(date -u +%H:%M:%S)] GPU 1 is free."
sleep 10

# Sanity: pipeline still running
if ! kill -0 950372 2>/dev/null; then
    echo "[$(date -u +%H:%M:%S)] WARNING: pipeline PID 950372 not running."
fi

# --- Stage 1: MAE pre-flight ---
echo ""
echo "=== Stage 1: MAE pre-flight ==="

# Disk gate
free_gb=$(df --output=avail -BG /home/ubuntu | tail -1 | tr -d 'G ')
echo "[$(date -u +%H:%M:%S)] Disk free: ${free_gb} GiB (need >= 50)"
if (( free_gb < 50 )); then
    echo "[$(date -u +%H:%M:%S)] ABORTING stage 1: not enough disk."
    SKIP_MAE=1
fi

if [[ -d features/mae ]]; then
    echo "[$(date -u +%H:%M:%S)] WARNING: features/mae already exists. Aborting MAE pre-flight."
    SKIP_MAE=1
fi

if [[ -z "${SKIP_MAE:-}" ]]; then
    echo "[$(date -u +%H:%M:%S)] MAE cache start..."
    uv run python scripts/01_cache_features.py \
        --encoder mae --gpu 1 \
        > "$SIDE_DIR/mae_cache.log" 2>&1
    cache_rc=$?
    echo "[$(date -u +%H:%M:%S)] MAE cache done (rc=$cache_rc)"

    if (( cache_rc == 0 )); then
        echo "[$(date -u +%H:%M:%S)] MAE pre-flight train start (seed=99, max-steps=5000)..."
        uv run python scripts/02_train_clip.py \
            --encoder mae \
            --seed 99 \
            --gpu 1 \
            --max-steps 5000 \
            --no-resume \
            --no-wandb \
            > "$SIDE_DIR/mae_preflight.log" 2>&1
        train_rc=$?
        echo "[$(date -u +%H:%M:%S)] MAE pre-flight train done (rc=$train_rc)"

        # Extract val_ppl trajectory and final loss lines
        {
            echo "MAE pre-flight (seed=99, max_steps=5000, cached features)"
            echo "started: see mae_preflight.log timestamps"
            echo ""
            echo "val/ppl trajectory:"
            grep -E "val/ppl|best val" "$SIDE_DIR/mae_preflight.log" || true
            echo ""
            echo "loss every 50 steps (sampled):"
            grep "| INFO" "$SIDE_DIR/mae_preflight.log" | grep "step " | awk 'NR%4==1' | tail -30 || true
            echo ""
            echo "Training summary:"
            grep -E "Training summary|stopped_reason" "$SIDE_DIR/mae_preflight.log" || true
        } > "$SIDE_DIR/mae_preflight_curve.txt"
    fi

    # --- Cleanup MAE pre-flight artifacts ---
    echo "[$(date -u +%H:%M:%S)] Cleaning up MAE pre-flight artifacts..."
    rm -rf features/mae
    rm -rf checkpoints/mae_seed99
    echo "[$(date -u +%H:%M:%S)] Cleanup done. Disk free: $(df --output=avail -BG /home/ubuntu | tail -1 | tr -d 'G ') GiB"
fi

# --- Stage 2: siglip_seed1 caption gen ---
echo ""
echo "=== Stage 2: siglip_seed1 caption gen ==="
if pgrep -f "02_train_clip.py.*--gpu 1" >/dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%S)] GPU 1 busy. Skipping caption gen."
else
    echo "[$(date -u +%H:%M:%S)] siglip_seed1 caption gen start (resume-safe)..."
    uv run python scripts/04_generate_captions.py \
        --encoders siglip --seed 1 --gpu 1 \
        --output captions/captions_siglip_seed1.jsonl \
        > "$SIDE_DIR/siglip_seed1_gen.log" 2>&1
    rc=$?
    echo "[$(date -u +%H:%M:%S)] siglip_seed1 caption gen done (rc=$rc)"
    wc -l captions/captions_siglip_seed1.jsonl 2>/dev/null || true
fi

# --- Stage 3: linear probes ---
echo ""
echo "=== Stage 3: linear probes (clip + siglip) ==="
if pgrep -f "02_train_clip.py.*--gpu 1" >/dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%S)] GPU 1 busy. Skipping probes."
else
    echo "[$(date -u +%H:%M:%S)] Probes start..."
    uv run python scripts/07_probes.py \
        --encoders clip,siglip --gpu 1 \
        --output captions/probe_clip_siglip.csv \
        > "$SIDE_DIR/probes.log" 2>&1
    rc=$?
    echo "[$(date -u +%H:%M:%S)] Probes done (rc=$rc)"
fi

echo ""
echo "=== orchestrate.sh end $(date -u) ==="
