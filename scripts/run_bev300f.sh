#!/usr/bin/env bash
# 300f scene 2025-07-17-16-12_1 (cases 0-29) with BEV matcher.
# Clean then remove_early; det via DET_SCORE_THRES (default 0.1); tau=0.7.
# Does not kill other jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=/home/hzy/miniconda3/envs/v2u4/bin/python
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DET="${DET_SCORE_THRES:-0.1}"
# tag suffix e.g. det010 / det015
DET_TAG="$("$PY" -c "print('det{:03d}'.format(int(round(float('$DET')*100))))")"
OUT="${OUT_DIR:-logs/bev300f}"
mkdir -p "$OUT"
MATCHER="$ROOT/logs/bev_matcher/matcher_last.pth"
STATUS="$OUT/status_${STAMP}.txt"
LOG="$OUT/orchestrator_${STAMP}.log"

pick_gpu() {
  # Prefer max free; print only index on stdout.
  "$PY" - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    text=True,
)
rows = []
for line in out.strip().splitlines():
    i, free = line.split(",")
    rows.append((int(free.strip()), int(i.strip())))
rows.sort(reverse=True)
best_free, best_i = rows[0]
print(f"[gpu] pick {best_i} free={best_free}MiB", file=sys.stderr, flush=True)
print(best_i)
PY
}

run_one() {
  local mode="$1" level="$2" tag="$3" gpu="$4"
  local save="$OUT/bev300f_${tag}_${STAMP}.json"
  local runlog="$OUT/bev300f_${tag}_${STAMP}.log"
  echo "[run] $(date '+%F %T') mode=$mode level=$level gpu=$gpu -> $save" | tee -a "$LOG"
  echo "$tag RUNNING gpu=$gpu $(date '+%F %T')" >>"$STATUS"
  set +e
  "$PY" -W ignore scripts/run_three_source.py \
    --gpu "$gpu" --model attfuse --cases 0-29 --quiet \
    --score_thres 0.3 --theta_soft 0.05 --det_score_thres "$DET" \
    --gate_q_mode p --buf_q_mode p --theta_confirm 0.7 \
    --bev_matcher "$MATCHER" --bev_tau 0.7 --bev_c_min 0.2 \
    --mode "$mode" --level "$level" --early_remove_mode box \
    --defense ours --asr_dist_thres 0 \
    --save "$save" >"$runlog" 2>&1
  local rc=$?
  set -e
  echo "[done] $(date '+%F %T') rc=$rc tag=$tag log=$runlog" | tee -a "$LOG"
  echo "$tag DONE rc=$rc $(date '+%F %T')" >>"$STATUS"
  return "$rc"
}

{
  echo "[start] $(date '+%F %T') bev300f stamp=$STAMP det=$DET tag=$DET_TAG out=$OUT"
  echo "status=$STATUS"
  GPU="$(pick_gpu | head -1)"
  echo "[gpu] using $GPU"
  # Wait until free >= 5500 MiB (retry every 60s, max ~6h)
  for n in $(seq 1 360); do
    free="$("$PY" - <<PY
import subprocess
out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True)
print(int(out.strip().splitlines()[$GPU].split()[0]))
PY
)"
    echo "[wait] gpu$GPU free=${free}MiB try=$n $(date '+%F %T')"
    if [[ "$free" -ge 5500 ]]; then
      break
    fi
    sleep 60
  done
  run_one none early "clean_${DET_TAG}" "$GPU" || true
  GPU="$(pick_gpu | head -1)"
  run_one remove early "remove_early_${DET_TAG}" "$GPU" || true
  echo "[all-done] $(date '+%F %T')"
} 2>&1 | tee -a "$LOG"
