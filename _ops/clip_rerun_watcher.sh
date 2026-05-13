#!/usr/bin/env bash
# CLIP seed 1+2 rerun watcher.
#
# Waits for the full pipeline to finish (marker "Pipeline complete." in
# pipeline.out), then re-trains clip_seed1 and clip_seed2 at the current
# 35k-step budget from configs/base.yaml, so the encoder comparison is
# budget-matched across all CLIP and SigLIP runs.
#
# - Polls every 5 min for the completion marker.
# - Existing 50k checkpoints are moved to checkpoints/clip_seed{N}_50k_backup/
#   before launching, so the 9.853-PPL adaptor is preserved if the new
#   35k run ends up worse.
# - Reruns are launched in parallel: seed 1 on GPU 0, seed 2 on GPU 1,
#   with --no-resume so each starts from step 0.
# - Exits without triggering if run_full_pipeline.sh dies before the
#   completion marker appears, or after a 48 h safety cutoff.
#
# Single-instance via flock. Launch:
#   nohup bash /home/ubuntu/clip_rerun_watcher.sh >/dev/null 2>&1 &

set -u

LOG=/home/ubuntu/clip_rerun_watcher.log
LOCK=/home/ubuntu/clip_rerun_watcher.lock
REPO=/home/ubuntu/ttic_embeddings
PIPELINE_OUT=$REPO/pipeline.out
RERUN_LOG_DIR=$REPO/logs/20260510_162813
SAFETY_CUTOFF_S=$((48 * 3600))

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -u) another watcher already running; exiting" >>"$LOG"
  exit 0
fi

note() { echo "$(date -u) $*" >>"$LOG"; }

START_TS=$(date -u +%s)
note "=== watcher started (pid $$) ==="

if [ ! -f "$PIPELINE_OUT" ]; then
  note "FATAL: $PIPELINE_OUT not found; nothing to watch"
  exit 1
fi
if [ ! -d "$RERUN_LOG_DIR" ]; then
  note "FATAL: $RERUN_LOG_DIR not found; cannot log reruns there"
  exit 1
fi

# Refuse to start if a marker is already present — pipeline.out is
# append-only across runs, and we don't want to trigger off a stale one.
if grep -q "Pipeline complete\." "$PIPELINE_OUT" 2>/dev/null; then
  note "FATAL: 'Pipeline complete.' already present in pipeline.out at start; refusing to trigger"
  exit 1
fi

while true; do
  TS=$(date -u +%s)
  ELAPSED=$((TS - START_TS))

  if [ "$ELAPSED" -ge "$SAFETY_CUTOFF_S" ]; then
    note "EXIT: 48 h safety cutoff reached without completion marker"
    exit 0
  fi

  if grep -q "Pipeline complete\." "$PIPELINE_OUT" 2>/dev/null; then
    note "TRIGGER: pipeline completion marker detected"
    break
  fi

  PIPELINE_PIDS=$(pgrep -f 'run_full_pipeline.sh' || true)
  if [ -z "$PIPELINE_PIDS" ]; then
    note "EXIT: run_full_pipeline.sh not running and no completion marker; not triggering reruns"
    exit 0
  fi

  sleep 300
done

cd "$REPO" || { note "FATAL: cannot cd to $REPO"; exit 1; }

for seed in 1 2; do
  src="checkpoints/clip_seed${seed}"
  dst="checkpoints/clip_seed${seed}_50k_backup"
  if [ -d "$src" ]; then
    if [ -e "$dst" ]; then
      note "WARN: $dst already exists; skipping backup of $src"
    else
      mv "$src" "$dst"
      note "Backed up $src -> $dst"
    fi
  else
    note "WARN: $src not found; no backup needed"
  fi
done

note "Launching clip_seed1 (GPU 0) and clip_seed2 (GPU 1) reruns at 35k-step budget"

uv run python scripts/02_train_clip.py \
    --encoder clip --seed 1 --gpu 0 --no-resume \
    > "$RERUN_LOG_DIR/train_clip_seed1_rerun.log" 2>&1 &
pid0=$!
note "  clip_seed1 rerun pid=$pid0 -> train_clip_seed1_rerun.log"

uv run python scripts/02_train_clip.py \
    --encoder clip --seed 2 --gpu 1 --no-resume \
    > "$RERUN_LOG_DIR/train_clip_seed2_rerun.log" 2>&1 &
pid1=$!
note "  clip_seed2 rerun pid=$pid1 -> train_clip_seed2_rerun.log"

wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?

note "clip_seed1 rerun exited rc=$rc0"
note "clip_seed2 rerun exited rc=$rc1"

if [ $rc0 -eq 0 ] && [ $rc1 -eq 0 ]; then
  note "=== reruns complete OK ==="
else
  note "=== reruns finished with errors (rc0=$rc0 rc1=$rc1) ==="
fi
