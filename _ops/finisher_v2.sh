#!/usr/bin/env bash
# Pipeline finisher (v2): waits for run_full_pipeline.sh, runs CLIP reruns,
# commits + pushes results on success, shuts down.
#
# Difference from v1 (finisher.sh): after detecting "Pipeline complete.",
# this version launches /home/ubuntu/clip_reruns_postphase.sh and waits for
# the clip_reruns.complete marker before doing git work. If the reruns fail
# (clip_reruns.failed marker), the finisher writes the failure summary and
# exits WITHOUT shutting down or committing — preserves disk state for
# investigation.

set -u
LOG=/home/ubuntu/finisher.log
exec >> "$LOG" 2>&1

# Single-instance guard. Note: we deliberately DO NOT reuse
# /home/ubuntu/finisher.lock — that file's lock is still held by the orphaned
# pipeline tree (PID 950372 + python training children), because they inherited
# fd 9 from the original finisher.sh when it forked them. The inherited fd will
# only release when the pipeline tree dies, which is precisely when v2 needs
# to be alive and working. So v2 uses its own lock file.
LOCK=/home/ubuntu/finisher_v2.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -u) another finisher_v2 already running (lock held); exiting"
  exit 0
fi

REPO=/home/ubuntu/ttic_embeddings
SUMMARY=/home/ubuntu/pipeline_summary.txt
START_TIME="2026-05-10 06:35 UTC"
RERUNS_SCRIPT=/home/ubuntu/clip_reruns_postphase.sh
RERUNS_OK=/home/ubuntu/clip_reruns.complete
RERUNS_FAIL=/home/ubuntu/clip_reruns.failed
RERUNS_LOG=/home/ubuntu/clip_reruns.log

echo ""
echo "=== finisher_v2 started $(date -u) ==="
cd "$REPO"

wait_no_pipeline() {
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

if ! is_complete; then
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
  echo "=== unrecoverable pipeline failure — not shutting down $(date -u) ==="
  exit 0
fi

# ====================== NEW: CLIP reruns phase ======================
echo "=== Pipeline complete; launching CLIP reruns $(date -u) ==="
# Clear any stale failure marker from a previous attempt
rm -f "$RERUNS_FAIL"

if [[ ! -x "$RERUNS_SCRIPT" ]]; then
  cat > "$SUMMARY" <<EOF
TTIC Embeddings pipeline — RERUN SCRIPT MISSING
Pipeline complete but $RERUNS_SCRIPT not found / not executable.
Not committing, not shutting down. Inspect: $LOG
EOF
  echo "=== rerun script missing — bail $(date -u) ==="
  exit 0
fi

# Run reruns; their internal logging goes to logs/reruns_*/ inside the repo
bash "$RERUNS_SCRIPT" > "$RERUNS_LOG" 2>&1
RERUN_RC=$?
echo "=== CLIP reruns exited rc=$RERUN_RC at $(date -u) ==="

if [[ $RERUN_RC -ne 0 || -f "$RERUNS_FAIL" || ! -f "$RERUNS_OK" ]]; then
  cat > "$SUMMARY" <<EOF
TTIC Embeddings pipeline — RERUNS FAILED
Main pipeline completed but CLIP reruns failed (rc=$RERUN_RC).
Not committing, not shutting down. Disk state preserved for investigation.
Failure marker: $RERUNS_FAIL
Rerun log:      $RERUNS_LOG
Inspect:        $LOG, $REPO/logs/reruns_*/
EOF
  echo "=== rerun failure — bail without commit/shutdown $(date -u) ==="
  exit 0
fi
echo "=== CLIP reruns complete; proceeding to commit/push $(date -u) ==="
# ====================================================================

echo "=== success path: preparing commit $(date -u) ==="

git config user.name "Adam McMurtry"
git config user.email "amcmurtry@uchicago.edu"

BRANCH="pipeline-results-$(date -u +%Y%m%d)"
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH"
fi

# Copy the freshest agent_notes.md (and finisher.log + pipeline_summary.txt
# while we're at it) into the repo so they survive instance shutdown.
# agent_notes.md is also independently snapshot-pushed to the orphan branch
# ops/agent-notes, but this is the canonical co-located copy.
echo "Copying operational artifacts into _ops/..."
mkdir -p _ops
[ -f /home/ubuntu/agent_notes.md ]            && cp -f /home/ubuntu/agent_notes.md            _ops/agent_notes.md
[ -f /home/ubuntu/finisher.log ]              && cp -f /home/ubuntu/finisher.log              _ops/finisher.log
[ -f /home/ubuntu/pipeline_summary.txt ]      && cp -f /home/ubuntu/pipeline_summary.txt      _ops/pipeline_summary.txt
# Orchestration scripts — persist so the rerun + commit logic survives shutdown
[ -f /home/ubuntu/finisher_v2.sh ]            && cp -f /home/ubuntu/finisher_v2.sh            _ops/finisher_v2.sh
[ -f /home/ubuntu/clip_reruns_postphase.sh ]  && cp -f /home/ubuntu/clip_reruns_postphase.sh  _ops/clip_reruns_postphase.sh
[ -f /home/ubuntu/clip_rerun_watcher.sh ]     && cp -f /home/ubuntu/clip_rerun_watcher.sh     _ops/clip_rerun_watcher.sh
[ -f /home/ubuntu/clip_reruns.log ]           && cp -f /home/ubuntu/clip_reruns.log           _ops/clip_reruns.log
[ -f /home/ubuntu/clip_reruns.complete ]      && cp -f /home/ubuntu/clip_reruns.complete      _ops/clip_reruns.complete
git add -f _ops/ || true

# Force-add ALL log dirs (main pipeline + reruns + any abandoned).
# Total size <1 MB; cheap insurance against the prior `ls -dt | head -1`
# bug that silently skipped the main pipeline log once reruns existed.
LOGDIR=$(ls -dt logs/*/ 2>/dev/null | grep -v reruns | head -1)
echo "Latest main pipeline log dir (for commit message): $LOGDIR"
echo "Adding logs/ recursively..."
git add -f logs/ || true

shopt -s nullglob
for f in captions/*.jsonl captions/*.jsonl.orig captions/*.csv captions/quality_summary.txt; do
  git add -f "$f" || true
done
shopt -u nullglob

# Stage modified tracked source files (configs, scripts, src/) and any
# untracked code/config files left over from the run. Critical for
# reproducibility: the commit should reflect the code that actually ran.
echo "Staging source modifications..."
git add -u || true   # all modifications to tracked files
for f in pipeline.out .env.pipeline RUN_PIPELINE.md scripts/01_cache_features.py; do
  [ -e "$f" ] && git add -f "$f" || true
done

CKPT_TOTAL_KB=$(du -sk checkpoints 2>/dev/null | awk '{print $1}')
CKPT_TOTAL_KB=${CKPT_TOTAL_KB:-0}
CKPT_MAX_BYTES=$(find checkpoints -name 'adaptor_best.pt' -printf '%s\n' 2>/dev/null | sort -n | tail -1)
CKPT_MAX_BYTES=${CKPT_MAX_BYTES:-0}
# Rule: total < 800MB AND no single file > 90MB.
# Raised from 500MB: with 12 seeds and 49 MB per checkpoint dir
# (adaptor_best.pt + adaptor_latest.pt), total reaches ~580 MB. Only the
# adaptor_best.pt files (~288 MB total) actually get committed, but the
# check guards the whole dir as a conservative cap. 800 MB gives headroom.
if [ "$CKPT_TOTAL_KB" -gt 0 ] && [ "$CKPT_TOTAL_KB" -lt 819200 ] && [ "$CKPT_MAX_BYTES" -lt 94371840 ]; then
  shopt -s nullglob
  for ck in checkpoints/*/adaptor_best.pt; do
    case "$ck" in *_orig/*) continue ;; esac
    git add -f "$ck" || true
  done
  shopt -u nullglob
  CKPT_NOTE="checkpoints included (total ${CKPT_TOTAL_KB} KB, max single $((CKPT_MAX_BYTES/1024)) KB; *_orig/ excluded)"
else
  CKPT_NOTE="checkpoints skipped (total ${CKPT_TOTAL_KB} KB, max single $((CKPT_MAX_BYTES/1024)) KB; rule: total<800MB AND max<90MB)"
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
  echo "=== finisher_v2 done (no commit) $(date -u) ==="
  exit 0
fi

COMMIT_MSG=$(cat <<COMMITEOF
Pipeline results $(date -u +%Y-%m-%d)

Full TTIC Embeddings pipeline run: 4 encoders x 3 seeds = 12 adaptors,
captions, metrics, statistical analysis, linear probes, CIDEr/SPICE quality.

Includes CLIP seed1+seed2 reruns at native 35k cosine (replacing
resume-contaminated seed1 and schedule-mismatched seed2).

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
  echo "=== finisher_v2 done (commit failed) $(date -u) ==="
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
  echo "=== finisher_v2 done (push failed; not shutting down) $(date -u) ==="
  exit 0
fi

cat > "$SUMMARY" <<EOF
TTIC Embeddings pipeline — SUCCESS (with CLIP reruns)
Started:  $START_TIME
Finished: $(date -u)
Log dir:  $LOGDIR
Branch:   $BRANCH
Commit:   $SHA
$CKPT_NOTE
Retries to success: $ATTEMPTS
CLIP reruns: see $RERUNS_OK
EOF
echo "=== success — shutting down in 60s $(date -u) ==="
sleep 60
sudo shutdown -h now "TTIC pipeline + CLIP reruns complete; instance shutting down"
