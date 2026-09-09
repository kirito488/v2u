#!/usr/bin/env bash
# Re-run all attack scenarios with locked hyperparams for comparison vs
# logs/compare_*_20260905_182559 (full val, all scenes).
#
# Locked:
#   theta_soft=0.2  score/theta_p=0.3  kappa=1.4/1.0  confirm=0.7
#   T=10  K_atk=10  pool_k=3  beta=0.5  gamma=0.5  tau=0.1
#   V_min = UAV S P5 (from clean dump calib; default 0.80)
#   det=0.2  BEV on (matcher_last, bev_tau=0.6)
#   remove_select=prefer  defense=ours  all_scenes
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=/home/hzy/miniconda3/envs/v2u4/bin/python
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT_DIR:-logs/attack_suite_soft02}"
NEED_MIB="${NEED_FREE_MIB:-12000}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
VMIN="${VMIN:-0.80}"
DET="${DET_SCORE_THRES:-0.2}"
REMOVE_SELECT="${REMOVE_SELECT:-prefer}"
MATCHER="${MATCHER:-$ROOT/logs/bev_matcher/matcher_last.pth}"
BEV_TAU="${BEV_TAU:-0.6}"
BEV_CMIN="${BEV_CMIN:-0.2}"

mkdir -p "$OUT"
STATUS="$OUT/status_${STAMP}.txt"
LOG="$OUT/orchestrator_${STAMP}.log"
QUEUE="$OUT/queue_${STAMP}.txt"
GPU_LOCK_DIR="$OUT/gpu_locks_${STAMP}"
mkdir -p "$GPU_LOCK_DIR"
: >"$STATUS"
: >"$QUEUE"

enq() {
  local tag="$1" mode="$2" level="$3"
  echo "${tag}|${mode}|${level}" >>"$QUEUE"
}

# Same matrix as compare_*_20260905_182559
enq clean_none none early
enq spoof_early spoof early
enq spoof_intermediate spoof intermediate
enq spoof_late spoof late
enq remove_early remove early
enq remove_intermediate remove intermediate
enq remove_late remove late
enq multi_spoof_intermediate multi_spoof intermediate
enq mass_remove_intermediate mass_remove intermediate

NJOBS=$(wc -l <"$QUEUE")
{
  echo "[start] $(date '+%F %T') stamp=$STAMP jobs=$NJOBS"
  echo "out=$OUT all_scenes=1 det=$DET soft=0.2 score=0.3 kappa=1.4/1.0 confirm=0.7"
  echo "T=10 Katk=10 K=3 beta=0.5 gamma=0.5 tau=0.1 vmin=$VMIN remove_select=$REMOVE_SELECT"
  echo "BEV=on matcher=$MATCHER bev_tau=$BEV_TAU bev_c_min=$BEV_CMIN"
  echo "compare_vs=logs/compare_*_20260905_182559 + clean_none_20260905_182559"
  echo "parallel=$MAX_PARALLEL need_mib=$NEED_MIB"
} | tee -a "$LOG" | tee -a "$STATUS"

pick_gpu() {
  NEED_MIB="$NEED_MIB" GPU_LOCK_DIR="$GPU_LOCK_DIR" "$PY" - <<'PY'
import os, subprocess, time
need = int(os.environ["NEED_MIB"])
lock_root = os.environ["GPU_LOCK_DIR"]
os.makedirs(lock_root, exist_ok=True)
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    text=True,
)
rows = []
for line in out.strip().splitlines():
    i, free = line.split(",")
    rows.append((int(free.strip()), int(i.strip())))
rows.sort(reverse=True)
now = time.time()
for free, idx in rows:
    if free < need:
        continue
    lock = os.path.join(lock_root, "gpu_{}.lock".format(idx))
    if os.path.exists(lock):
        try:
            if now - os.path.getmtime(lock) < 7200:
                continue
        except OSError:
            pass
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        print(idx)
        break
    except FileExistsError:
        continue
PY
}

release_gpu() {
  rm -f "${GPU_LOCK_DIR}/gpu_${1}.lock" 2>/dev/null || true
}

run_job() {
  local line="$1"
  IFS='|' read -r tag mode level <<<"$line"
  local save="$OUT/${tag}_${STAMP}.json"
  local runlog="$OUT/${tag}_${STAMP}.log"
  if [[ -f "$save" ]]; then
    echo "[skip] $tag exists" | tee -a "$STATUS"
    return 0
  fi
  local gpu=""
  for _ in $(seq 1 720); do
    gpu="$(pick_gpu)"
    [[ -n "$gpu" ]] && break
    sleep 20
  done
  if [[ -z "$gpu" ]]; then
    echo "[fail] no gpu for $tag" | tee -a "$STATUS"
    return 1
  fi
  echo "[run] $(date '+%F %T') gpu=$gpu tag=$tag mode=$mode level=$level" | tee -a "$STATUS" | tee -a "$LOG"
  set +e
  local -a cmd=(
    "$PY" -W ignore scripts/run_three_source.py
    --gpu "$gpu" --model attfuse --all_scenes --quiet
    --score_thres 0.3 --theta_p 0.3 --theta_soft 0.2
    --det_score_thres "$DET"
    --gate_q_mode p --buf_q_mode p
    --q_h 0.3 --q_l 0.2
    --theta_confirm 0.7 --theta_reject 0.1
    --t_timeout 10 --k_atk 10 --pool_k 3
    --kappa_pos 1.4 --kappa_neg 1.0
    --v_min "$VMIN"
    --beta_v 0.5 --gamma_r 0.5 --tau 0.1
    --bev_matcher "$MATCHER" --bev_tau "$BEV_TAU" --bev_c_min "$BEV_CMIN"
    --mode "$mode" --level "$level"
    --early_remove_mode box --remove_select "$REMOVE_SELECT"
    --n_objects 3
    --defense ours --asr_dist_thres 0
    --save "$save"
  )
  "${cmd[@]}" >"$runlog" 2>&1
  local rc=$?
  set -e
  release_gpu "$gpu"
  echo "[done] $(date '+%F %T') rc=$rc tag=$tag gpu=$gpu" | tee -a "$STATUS" | tee -a "$LOG"
  return 0
}

export -f run_job pick_gpu release_gpu
export PY ROOT OUT STAMP STATUS LOG NEED_MIB GPU_LOCK_DIR VMIN DET REMOVE_SELECT MATCHER BEV_TAU BEV_CMIN

running=0
pids=()
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  while [[ "$running" -ge "$MAX_PARALLEL" ]]; do
    new_pids=(); running=0
    for pid in "${pids[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid"); running=$((running+1))
      fi
    done
    pids=("${new_pids[@]:-}")
    [[ "$running" -ge "$MAX_PARALLEL" ]] && sleep 20
  done
  run_job "$line" &
  pids+=("$!")
  running=$((running+1))
  sleep 5
done <"$QUEUE"

for pid in "${pids[@]:-}"; do
  wait "$pid" || true
done
echo "[all-done] $(date '+%F %T') stamp=$STAMP" | tee -a "$STATUS" | tee -a "$LOG"
