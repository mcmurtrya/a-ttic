#!/usr/bin/env bash
# Autonomous pipeline finisher: waits for run_full_pipeline.sh, retries on
# failure (up to 3x), commits + pushes results on success, shuts down.
# Launched detached via nohup; not interactive.

set -u
LOG=/home/ubuntu/finisher.log
exec >> "$LOG" 2>&1

# Single-instance guard: refuse to run if another finisher already holds the lock.
LOCK=/home/ubuntu/finisher.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -u) another finisher already running (lock held); exiting"
  exit 0
fi

REPO=/home/ubuntu/ttic_embeddings
SUMMARY=/home/ubuntu/pipeline_summary.txt
START_TIME="2026-05-10 06:35 UTC"

echo ""
echo "=== finisher started $(date -u) ==="
cd "$REPO"

wait_no_pipeline() {
  # Wait until no run_full_pipeline.sh process exists. Robust to PID changes.
  while pgrep -f 'run_full_pipeline.sh' >/dev/null 2>&1; do
    sleep 60
  done
}

is_complete() {
  grep -q "Pipeline complete\." "$REPO/pipeline.out" 2>/dev/null
}

echo "Waiting for any active run_full_pipeline.sh..."
wait_no_pipeline
echo "=== no pipeline running $(date -u) ==="

ATTEMPTS=0
MAX_RETRIES=3

while ! is_complete && [ $ATTEMPTS -lt $MAX_RETRIES ]; do
  ATTEMPTS=$((ATTEMPTS+1))
  echo "=== retry $ATTEMPTS started $(date -u) ==="
  if [ -f "$REPO/.env.pipeline" ]; then
    set -a; source "$REPO/.env.pipeline"; set +a
  else
    export COCO_ROOT=$HOME/data/coco
    export VG_ROOT=$HOME/data/vg
  fi
  bash "$REPO/scripts/run_full_pipeline.sh" >> "$REPO/pipeline.out" 2>&1
  RC=$?
  echo "=== retry $ATTEMPTS exited rc=$RC at $(date -u) ==="
done

if is_complete; then
  echo "=== success path: preparing commit $(date -u) ==="

  git config user.name "Adam McMurtry"
  git config user.email "amcmurtry@uchicago.edu"

  BRANCH="pipeline-results-$(date -u +%Y%m%d)"
  if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    git checkout "$BRANCH"
  else
    git checkout -b "$BRANCH"
  fi

  LOGDIR=$(ls -dt logs/*/ 2>/dev/null | head -1)
  echo "Latest log dir: $LOGDIR"

  # Force-add gitignored result paths.
  [ -n "$LOGDIR" ] && git add -f "$LOGDIR" || true
  shopt -s nullglob
  for f in captions/*.jsonl captions/*.csv captions/quality_summary.txt; do
    git add -f "$f" || true
  done
  shopt -u nullglob

  CKPT_TOTAL_KB=$(du -sk checkpoints 2>/dev/null | awk '{print $1}')
  CKPT_TOTAL_KB=${CKPT_TOTAL_KB:-0}
  CKPT_MAX_BYTES=$(find checkpoints -name 'adaptor_best.pt' -printf '%s\n' 2>/dev/null | sort -n | tail -1)
  CKPT_MAX_BYTES=${CKPT_MAX_BYTES:-0}
  # Rule: total < 500MB AND no single file > 90MB.
  if [ "$CKPT_TOTAL_KB" -gt 0 ] && [ "$CKPT_TOTAL_KB" -lt 512000 ] && [ "$CKPT_MAX_BYTES" -lt 94371840 ]; then
    shopt -s nullglob
    for ck in checkpoints/*/adaptor_best.pt; do
      git add -f "$ck" || true
    done
    shopt -u nullglob
    CKPT_NOTE="checkpoints included (total ${CKPT_TOTAL_KB} KB, max single $((CKPT_MAX_BYTES/1024)) KB)"
  else
    CKPT_NOTE="checkpoints skipped (total ${CKPT_TOTAL_KB} KB, max single $((CKPT_MAX_BYTES/1024)) KB; rule: total<500MB AND max<90MB)"
  fi
  echo "Checkpoint decision: $CKPT_NOTE"

  if git diff --cached --quiet; then
    echo "Nothing staged — aborting commit"
    cat > "$SUMMARY" <<EOF
TTIC Embeddings pipeline — COMPLETE BUT NO RESULTS STAGED
Started: $START_TIME
Ended:   $(date -u)
Branch attempted: $BRANCH
$CKPT_NOTE
Inspect: $LOG, $REPO/pipeline.out, $REPO/logs/
EOF
    echo "=== finisher done (no commit) $(date -u) ==="
    exit 0
  fi

  COMMIT_MSG=$(cat <<COMMITEOF
Pipeline results $(date -u +%Y-%m-%d)

Full TTIC Embeddings pipeline run: 4 encoders x 3 seeds = 12 adaptors,
captions, metrics, statistical analysis, linear probes, CIDEr/SPICE quality.

Log dir: $LOGDIR
$CKPT_NOTE
Retries needed: $ATTEMPTS
COMMITEOF
)
  git commit -m "$COMMIT_MSG"
  COMMIT_RC=$?

  if [ $COMMIT_RC -ne 0 ]; then
    cat > "$SUMMARY" <<EOF
TTIC Embeddings pipeline — COMMIT FAILED
Pipeline completed but git commit returned $COMMIT_RC.
Branch attempted: $BRANCH
Inspect: $LOG and $REPO/pipeline.out
EOF
    echo "=== finisher done (commit failed) $(date -u) ==="
    exit 0
  fi

  SHA=$(git rev-parse HEAD)
  git push -u origin "$BRANCH"
  PUSH_RC=$?

  if [ $PUSH_RC -ne 0 ]; then
    cat > "$SUMMARY" <<EOF
TTIC Embeddings pipeline — PUSH FAILED
Pipeline + commit OK, but push returned $PUSH_RC.
Branch: $BRANCH (local only)
Commit SHA: $SHA
$CKPT_NOTE
Retries to success: $ATTEMPTS
Inspect: $LOG
EOF
    echo "=== finisher done (push failed; not shutting down) $(date -u) ==="
    exit 0
  fi

  cat > "$SUMMARY" <<EOF
TTIC Embeddings pipeline — SUCCESS
Started:  $START_TIME
Finished: $(date -u)
Log dir:  $LOGDIR
Branch:   $BRANCH
Commit:   $SHA
$CKPT_NOTE
Retries to success: $ATTEMPTS
EOF
  echo "=== success — shutting down in 60s $(date -u) ==="
  sleep 60
  sudo shutdown -h now "TTIC pipeline complete; instance shutting down"
else
  cat > "$SUMMARY" <<EOF
TTIC Embeddings pipeline — UNRECOVERABLE FAILURE
Started: $START_TIME
Ended:   $(date -u)
Attempts: $((ATTEMPTS + 1)) (1 initial + $ATTEMPTS retries; cap = $MAX_RETRIES retries)

Last 100 lines of $REPO/pipeline.out:
----
$(tail -100 "$REPO/pipeline.out" 2>/dev/null)
----

Inspect: $LOG, $REPO/pipeline.out, $REPO/logs/
Instance was NOT shut down.
EOF
  echo "=== unrecoverable failure — not shutting down $(date -u) ==="
fi

echo "=== finisher done $(date -u) ==="
