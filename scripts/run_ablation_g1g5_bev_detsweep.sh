#!/usr/bin/env bash
# Full ablation from docs/ABLATION_PLAN.md + BEV + det∈{0.05,0.1,0.15,0.2}.
# One-at-a-time from baseline; clean + remove_early prefer; 4 parallel GPUs.
# Does NOT kill other jobs; waits for free memory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=/home/hzy/miniconda3/envs/v2u4/bin/python
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT_DIR:-logs/ablation_bev_detsweep}"
MATCHER="${MATCHER:-$ROOT/logs/bev_matcher/matcher_last.pth}"
NEED_MIB="${NEED_FREE_MIB:-12000}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
REMOVE_SELECT="${REMOVE_SELECT:-prefer}"
BEV_TAU="${BEV_TAU:-0.7}"
BEV_CMIN="${BEV_CMIN:-0.2}"

mkdir -p "$OUT"
STATUS="$OUT/status_${STAMP}.txt"
LOG="$OUT/orchestrator_${STAMP}.log"
QUEUE="$OUT/queue_${STAMP}.txt"
GPU_LOCK_DIR="$OUT/gpu_locks_${STAMP}"
mkdir -p "$GPU_LOCK_DIR"
: >"$STATUS"
: >"$QUEUE"

# Baseline locked to ABLATION_PLAN.md (theta_soft=0.1 main default)
BASE_SOFT=0.1
BASE_CONFIRM=0.7
BASE_REJECT=0.1
BASE_T=10
BASE_KATK=10
BASE_POOLK=3
BASE_KPOS=1.4
BASE_KNEG=1.0
BASE_VMIN=0.25
BASE_BETA=0.5
BASE_GAMMA=0.5
BASE_TAU=0.1

det_tag() {
  case "$1" in
    0.05) echo 005 ;;
    0.1|0.10) echo 010 ;;
    0.15) echo 015 ;;
    0.2|0.20) echo 020 ;;
    *) echo "$1" | tr -d '.' ;;
  esac
}

# Append job lines: tag|det|mode|extra_args
enq() {
  local tag="$1" det="$2" mode="$3"
  shift 3
  local extra=("$@")
  local ex=""
  if ((${#extra[@]})); then
    ex=$(printf '%q ' "${extra[@]}")
  fi
  echo "${tag}|${det}|${mode}|${ex}" >>"$QUEUE"
}

# ---- Phase A: baseline × det sweep (main table first) ----
for det in 0.05 0.1 0.15 0.2; do
  dt=$(det_tag "$det")
  enq "base_det${dt}_clean" "$det" none
  enq "base_det${dt}_remove" "$det" remove
done

# ---- Phase B: G1–G5 one-at-a-time for each det (plan order G4→G1→G3→G5→G2) ----
for det in 0.1 0.2 0.15 0.05; do
  dt=$(det_tag "$det")
  # G4 lifetime
  for t in 5 15; do
    enq "g4_T${t}_det${dt}_clean" "$det" none --t_timeout "$t"
    enq "g4_T${t}_det${dt}_remove" "$det" remove --t_timeout "$t"
  done
  for k in 5 15; do
    enq "g4_Katk${k}_det${dt}_clean" "$det" none --k_atk "$k"
    enq "g4_Katk${k}_det${dt}_remove" "$det" remove --k_atk "$k"
  done
  for k in 1 5; do
    enq "g4_K${k}_det${dt}_clean" "$det" none --pool_k "$k"
    enq "g4_K${k}_det${dt}_remove" "$det" remove --pool_k "$k"
  done
  # G1 soft
  for soft in 0.15 0.2; do
    st=$(echo "$soft" | tr -d '.')
    enq "g1_soft${st}_det${dt}_clean" "$det" none --theta_soft "$soft"
    enq "g1_soft${st}_det${dt}_remove" "$det" remove --theta_soft "$soft"
  done
  # G3 confirm
  for c in 0.6 0.8; do
    ct=$(echo "$c" | tr -d '.')
    enq "g3_c${ct}_det${dt}_clean" "$det" none --theta_confirm "$c"
    enq "g3_c${ct}_det${dt}_remove" "$det" remove --theta_confirm "$c"
  done
  # G5 beta / gamma / tau
  for b in 0 1; do
    enq "g5_beta${b}_det${dt}_clean" "$det" none --beta_v "$b"
    enq "g5_beta${b}_det${dt}_remove" "$det" remove --beta_v "$b"
  done
  for g in 0 1; do
    enq "g5_gamma${g}_det${dt}_clean" "$det" none --gamma_r "$g"
    enq "g5_gamma${g}_det${dt}_remove" "$det" remove --gamma_r "$g"
  done
  for tau in 0.05 0.2; do
    tt=$(echo "$tau" | tr -d '.')
    enq "g5_tau${tt}_det${dt}_clean" "$det" none --tau "$tau"
    enq "g5_tau${tt}_det${dt}_remove" "$det" remove --tau "$tau"
  done
  # G2 kappa pairs
  for pair in "1.0,1.0" "1.4,1.4" "2.0,1.0" "2.0,1.4"; do
    kp="${pair%,*}"; kn="${pair#*,}"
    kt=$(echo "${kp}_${kn}" | tr -d '.')
    enq "g2_k${kt}_det${dt}_clean" "$det" none --kappa_pos "$kp" --kappa_neg "$kn"
    enq "g2_k${kt}_det${dt}_remove" "$det" remove --kappa_pos "$kp" --kappa_neg "$kn"
  done
done

NJOBS=$(wc -l <"$QUEUE")
{
  echo "[start] $(date '+%F %T') stamp=$STAMP jobs=$NJOBS"
  echo "out=$OUT matcher=$MATCHER remove_select=$REMOVE_SELECT bev_tau=$BEV_TAU"
  echo "baseline soft=$BASE_SOFT confirm=$BASE_CONFIRM T=$BASE_T Katk=$BASE_KATK K=$BASE_POOLK"
  echo "parallel=$MAX_PARALLEL need_mib=$NEED_MIB"
} | tee -a "$LOG" | tee -a "$STATUS"

pick_gpu() {
  "$PY" - <<'PY'
import os, subprocess, time
need = int(os.environ.get("NEED_MIB", "12000"))
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
    # stale lock > 2h
    if os.path.exists(lock):
        try:
            age = now - os.path.getmtime(lock)
            if age < 7200:
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
  local g="$1"
  rm -f "${GPU_LOCK_DIR}/gpu_${g}.lock" 2>/dev/null || true
}

run_job() {
  local line="$1"
  IFS='|' read -r tag det mode extra <<<"$line"
  local save="$OUT/ablation_${tag}_${STAMP}.json"
  local runlog="$OUT/ablation_${tag}_${STAMP}.log"
  if [[ -f "$save" ]]; then
    echo "[skip] $tag exists" | tee -a "$STATUS"
    return 0
  fi
  local gpu=""
  for _ in $(seq 1 720); do
    gpu="$(pick_gpu)"
    if [[ -n "$gpu" ]]; then
      break
    fi
    sleep 20
  done
  if [[ -z "$gpu" ]]; then
    echo "[fail] no gpu for $tag" | tee -a "$STATUS"
    return 1
  fi
  local extra_arr=()
  # printf %%q on empty used to write literal '' into the queue
  if [[ -n "${extra}" && "${extra}" != "''" && "${extra}" != "\"\"" ]]; then
    eval "extra_arr=( $extra )"
  fi
  echo "[run] $(date '+%F %T') gpu=$gpu tag=$tag det=$det mode=$mode ${extra_arr[*]:-}" | tee -a "$STATUS" | tee -a "$LOG"
  set +e
  local -a cmd=(
    "$PY" -W ignore scripts/run_three_source.py
    --gpu "$gpu" --model attfuse --cases 0-29 --quiet
    --score_thres 0.3 --theta_p 0.3
    --theta_soft "$BASE_SOFT"
    --det_score_thres "$det"
    --gate_q_mode p --buf_q_mode p
    --theta_confirm "$BASE_CONFIRM"
    --theta_reject "$BASE_REJECT"
    --t_timeout "$BASE_T"
    --kappa_pos "$BASE_KPOS" --kappa_neg "$BASE_KNEG"
    --k_atk "$BASE_KATK" --pool_k "$BASE_POOLK"
    --v_min "$BASE_VMIN"
    --beta_v "$BASE_BETA" --gamma_r "$BASE_GAMMA" --tau "$BASE_TAU"
    --bev_matcher "$MATCHER" --bev_tau "$BEV_TAU" --bev_c_min "$BEV_CMIN"
    --mode "$mode" --level early --early_remove_mode box
    --remove_select "$REMOVE_SELECT"
    --defense ours --asr_dist_thres 0
  )
  if ((${#extra_arr[@]})); then
    cmd+=("${extra_arr[@]}")
  fi
  cmd+=(--save "$save")
  "${cmd[@]}" >"$runlog" 2>&1
  local rc=$?
  set -e
  release_gpu "$gpu"
  echo "[done] $(date '+%F %T') rc=$rc tag=$tag gpu=$gpu" | tee -a "$STATUS" | tee -a "$LOG"
  return 0
}

# Parallel worker pool over queue
export -f run_job pick_gpu release_gpu det_tag
export PY ROOT OUT STAMP STATUS LOG MATCHER NEED_MIB REMOVE_SELECT BEV_TAU BEV_CMIN GPU_LOCK_DIR
export BASE_SOFT BASE_CONFIRM BASE_REJECT BASE_T BASE_KATK BASE_POOLK
export BASE_KPOS BASE_KNEG BASE_VMIN BASE_BETA BASE_GAMMA BASE_TAU

# Simple bash parallel: feed queue to background jobs
running=0
declare -a pids=()
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  while [[ "$running" -ge "$MAX_PARALLEL" ]]; do
    new_pids=()
    running=0
    for pid in "${pids[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
        running=$((running + 1))
      fi
    done
    pids=("${new_pids[@]:-}")
    if [[ "$running" -ge "$MAX_PARALLEL" ]]; then
      sleep 20
    fi
  done
  run_job "$line" &
  pids+=("$!")
  running=$((running + 1))
  sleep 5  # stagger GPU pick
done <"$QUEUE"

for pid in "${pids[@]:-}"; do
  wait "$pid" || true
done

echo "[all-done] $(date '+%F %T') stamp=$STAMP" | tee -a "$STATUS" | tee -a "$LOG"
