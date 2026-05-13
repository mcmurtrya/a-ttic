#!/usr/bin/env bash
# Pipeline anomaly watcher.
#
# Read-only: never signals processes, edits files, or modifies state. Only
# appends observations to /home/ubuntu/pipeline_watcher.log.
#
# Polls every 5 min. Detects: completion, process death, cache stall,
# training stall, disk pressure, OOM / Python tracebacks. Logs an OK
# summary every 6 polls (~30 min) so the log has a continuous timeline.
#
# Single-instance via flock. Exits cleanly when "Pipeline complete." marker
# appears in pipeline.out or after a 36-hour safety cutoff.
#
# Launch: nohup bash /home/ubuntu/pipeline_watcher.sh >/dev/null 2>&1 &

set -u

LOG=/home/ubuntu/pipeline_watcher.log
LOCK=/home/ubuntu/pipeline_watcher.lock
REPO=/home/ubuntu/ttic_embeddings
PIPELINE_OUT=$REPO/pipeline.out
SAFETY_CUTOFF_S=$((36 * 3600))

# Single-instance guard.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -u) another watcher already running; exiting" >>"$LOG"
  exit 0
fi

note() {
  echo "$(date -u) $*" >>"$LOG"
}

START_TS=$(date -u +%s)
note "=== watcher started (pid $$) ==="

# Cross-poll state
prev_cache_log_mtime=0
cache_stall_polls=0
prev_cache_ips=""
declare -A prev_ckpt_mtime
declare -A ckpt_stall_polls
declare -A reported_errors    # avoid re-reporting same traceback
poll_count=0

phase_for() {
  # Best-effort phase name from the most recent log file
  local log_dir="$1"
  local latest
  latest=$(ls -t "$log_dir"/*.log 2>/dev/null | head -1)
  [ -z "$latest" ] && { echo "unknown"; return; }
  local base
  base=$(basename "$latest")
  case "$base" in
    smoke.log)    echo "smoke" ;;
    cache_*.log)  echo "cache:${base#cache_}" ;;
    train_*.log)  echo "train:${base#train_}" ;;
    gen_*.log)    echo "generate:${base#gen_}" ;;
    score_*.log)  echo "score" ;;
    analyze_*)    echo "analyze" ;;
    probe_*)      echo "probe" ;;
    quality_*)    echo "quality" ;;
    *)            echo "$base" ;;
  esac
}

while true; do
  TS=$(date -u +%s)
  poll_count=$((poll_count + 1))
  ELAPSED=$((TS - START_TS))

  # ---- Safety cutoff ----------------------------------------------
  if [ "$ELAPSED" -ge "$SAFETY_CUTOFF_S" ]; then
    note "EXIT: 36h safety cutoff reached"
    exit 0
  fi

  # ---- Completion -------------------------------------------------
  if grep -q "Pipeline complete\." "$PIPELINE_OUT" 2>/dev/null; then
    note "DONE: 'Pipeline complete.' marker found in pipeline.out"
    exit 0
  fi

  # ---- Process presence -------------------------------------------
  PIPELINE_PIDS=$(pgrep -f 'run_full_pipeline.sh' || true)
  FINISHER_PIDS=$(pgrep -f 'finisher.sh' || true)
  if [ -z "$PIPELINE_PIDS" ] && [ -z "$FINISHER_PIDS" ]; then
    note "ALERT: no run_full_pipeline.sh AND no finisher.sh — pipeline likely terminated, finisher gave up"
    # Keep polling in case something resurrects; user may want to know if it stays dead
  elif [ -z "$PIPELINE_PIDS" ] && [ -n "$FINISHER_PIDS" ]; then
    : # finisher likely between retries; harmless transient
  fi

  # ---- Disk -------------------------------------------------------
  USE=$(df -P /home/ubuntu | awk 'NR==2{print $5}' | tr -d '%')
  if [ "${USE:-0}" -ge 95 ]; then
    note "ALERT: disk at ${USE}% full on /home/ubuntu"
  elif [ "${USE:-0}" -ge 85 ]; then
    note "WARN: disk at ${USE}% (>85%)"
  fi

  # ---- Latest log dir + phase -------------------------------------
  LATEST_LOG_DIR=$(ls -dt "$REPO"/logs/*/ 2>/dev/null | head -1)
  LATEST_LOG_DIR=${LATEST_LOG_DIR%/}
  PHASE="unknown"
  LATEST_FILE=""
  if [ -n "$LATEST_LOG_DIR" ]; then
    PHASE=$(phase_for "$LATEST_LOG_DIR")
    LATEST_FILE=$(ls -t "$LATEST_LOG_DIR"/*.log 2>/dev/null | head -1)
  fi

  # ---- Cache phase: stall + slow img/s ----------------------------
  if [[ "$PHASE" == cache:* ]] && [ -n "$LATEST_FILE" ]; then
    mtime=$(stat -c '%Y' "$LATEST_FILE")
    if [ "$mtime" -le "$prev_cache_log_mtime" ]; then
      cache_stall_polls=$((cache_stall_polls + 1))
    else
      cache_stall_polls=0
    fi
    prev_cache_log_mtime=$mtime
    if [ "$cache_stall_polls" -ge 5 ]; then
      note "ALERT: $PHASE log unchanged for 5 polls (~25 min). file=$LATEST_FILE"
      cache_stall_polls=0
    fi

    IPS=$(grep -oE '[0-9]+\.[0-9]+ img/s' "$LATEST_FILE" | tail -1 | awk '{print $1}')
    if [ -n "$IPS" ] && awk -v x="$IPS" 'BEGIN{exit !(x<50)}'; then
      note "WARN: cache img/s low: $IPS  ($PHASE)"
    fi
    prev_cache_ips=$IPS
  fi

  # ---- Training phase: errors + checkpoint stalls -----------------
  if [ -n "$LATEST_LOG_DIR" ]; then
    for tlog in "$LATEST_LOG_DIR"/train_*.log; do
      [ -f "$tlog" ] || continue
      if grep -qE 'Traceback|out of memory|CUDA error|RuntimeError' "$tlog"; then
        last_err=$(grep -E 'Traceback|out of memory|CUDA error|RuntimeError' "$tlog" | tail -1)
        key="$tlog::$last_err"
        if [ -z "${reported_errors[$key]:-}" ]; then
          note "ALERT: error in $(basename "$tlog"): $last_err"
          reported_errors[$key]=1
        fi
      fi
    done
  fi

  if [[ "$PHASE" == train:* ]]; then
    for ck in "$REPO"/checkpoints/*/adaptor_latest.pt; do
      [ -f "$ck" ] || continue
      mt=$(stat -c '%Y' "$ck")
      prev=${prev_ckpt_mtime[$ck]:-0}
      age=$((TS - mt))
      if [ "$mt" -le "$prev" ] && [ "$age" -gt 300 ]; then
        ckpt_stall_polls[$ck]=$((${ckpt_stall_polls[$ck]:-0} + 1))
      else
        ckpt_stall_polls[$ck]=0
      fi
      prev_ckpt_mtime[$ck]=$mt
      if [ "${ckpt_stall_polls[$ck]:-0}" -ge 4 ]; then
        note "ALERT: $ck mtime stuck for 4 polls (~20 min); age=${age}s"
        ckpt_stall_polls[$ck]=0
      fi
    done
  fi

  # ---- Periodic OK summary every 6 polls (~30 min) ----------------
  if [ $((poll_count % 6)) -eq 1 ]; then
    GPU_LINE=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd '; ')
    DISK_LINE=$(df -h /home/ubuntu | awk 'NR==2{printf "disk %s/%s (%s)", $3, $2, $5}')
    note "OK: phase=$PHASE pids=pipe[${PIPELINE_PIDS:-none}] fin[${FINISHER_PIDS:-none}] gpu=[$GPU_LINE] $DISK_LINE"
  fi

  sleep 300
done
