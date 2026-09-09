#!/usr/bin/env bash
# Poll GPUs; print GPU_FREE when a card is empty enough for three-source.
set -euo pipefail
ROOT="/data/hzy/lxt/CollaborativePerceptionDefense"
LOG="$ROOT/logs/gpu_watch.log"
INTERVAL="${1:-60}"
# Need ~12GB free for Ego+UAV+AttFuse, or nearly empty card.
FREE_MIN_MIB="${FREE_MIN_MIB:-12000}"
USED_EMPTY_MIB="${USED_EMPTY_MIB:-1500}"
MATCHER_PID="${MATCHER_PID:-3884337}"

mkdir -p "$(dirname "$LOG")"
echo "[watch] start $(date) interval=${INTERVAL}s free_min=${FREE_MIN_MIB} empty_used<${USED_EMPTY_MIB}" | tee -a "$LOG"

notified_free=0
notified_done=0

while true; do
  ts="$(date '+%F %T')"
  mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
  free_ids=()
  summary=""
  for line in "${rows[@]}"; do
    IFS=',' read -r idx used free util <<<"$line"
    idx="${idx// /}"; used="${used// /}"; free="${free// /}"; util="${util// /}"
    summary+=" gpu${idx}:used=${used}MiB free=${free}MiB util=${util}%"
    if [[ "$used" -le "$USED_EMPTY_MIB" ]] || [[ "$free" -ge "$FREE_MIN_MIB" ]]; then
      free_ids+=("$idx")
    fi
  done

  if [[ ${#free_ids[@]} -gt 0 ]]; then
    echo "[$ts] GPU_FREE ids=${free_ids[*]}${summary}" | tee -a "$LOG"
    if [[ "$notified_free" -eq 0 ]]; then
      notified_free=1
    fi
  else
    echo "[$ts] busy${summary}" >> "$LOG"
  fi

  if [[ -n "$MATCHER_PID" ]] && ! kill -0 "$MATCHER_PID" 2>/dev/null; then
    if [[ "$notified_done" -eq 0 ]]; then
      echo "[$ts] MATCHER_DONE pid=$MATCHER_PID" | tee -a "$LOG"
      notified_done=1
    fi
  fi

  # Stop watching if matcher done AND at least one free GPU (user can restart watch).
  if [[ "$notified_done" -eq 1 && ${#free_ids[@]} -gt 0 ]]; then
    echo "[$ts] watch exit: matcher done and GPU free" | tee -a "$LOG"
    exit 0
  fi

  sleep "$INTERVAL"
done
