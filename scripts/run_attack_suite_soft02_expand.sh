#!/usr/bin/env bash
# Expand remaining soft02 suite jobs onto free GPUs (same STAMP).
# Skips tags whose JSON already exists. Locks busy GPUs so they are not stolen.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=/home/hzy/miniconda3/envs/v2u4/bin/python
export PYTHONUNBUFFERED=1

STAMP="${STAMP:?STAMP required}"
OUT="${OUT_DIR:-logs/attack_suite_soft02}"
NEED_MIB="${NEED_FREE_MIB:-12000}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
VMIN="${VMIN:-0.80}"
DET="${DET_SCORE_THRES:-0.2}"
REMOVE_SELECT="${REMOVE_SELECT:-prefer}"
MATCHER="${MATCHER:-$ROOT/logs/bev_matcher/matcher_last.pth}"
BEV_TAU="${BEV_TAU:-0.6}"
BEV_CMIN="${BEV_CMIN:-0.2}"
# Comma-separated GPU ids already running other suite jobs (do not steal)
BUSY_GPUS="${BUSY_GPUS:-}"

mkdir -p "$OUT"
STATUS="$OUT/status_expand_${STAMP}.txt"
LOG="$OUT/orchestrator_expand_${STAMP}.log"
QUEUE="$OUT/queue_expand_${STAMP}.txt"
GPU_LOCK_DIR="$OUT/gpu_locks_expand_${STAMP}"
mkdir -p "$GPU_LOCK_DIR"
: >"$STATUS"

# Remaining matrix (caller may override via QUEUE_FILE)
if [[ -n "${QUEUE_FILE:-}" && -f "$QUEUE_FILE" ]]; then
  if [[ "$(readlink -f "$QUEUE_FILE")" != "$(readlink -f "$QUEUE")" ]]; then
    cp "$QUEUE_FILE" "$QUEUE"
  fi
  # else: QUEUE_FILE already is QUEUE; keep as-is
else
  cat >"$QUEUE" <<'EOF'
spoof_early|spoof|early
spoof_intermediate|spoof|intermediate
spoof_late|spoof|late
remove_early|remove|early
remove_late|remove|late
multi_spoof_intermediate|multi_spoof|intermediate
mass_remove_intermediate|mass_remove|intermediate
EOF
fi
if [[ ! -s "$QUEUE" ]]; then
  echo "[fail] empty queue $QUEUE" | tee -a "$STATUS"
  exit 1
fi

# Pre-lock busy GPUs
IFS=',' read -ra _BUSY <<<"$BUSY_GPUS"
for g in "${_BUSY[@]:-}"; do
  [[ -z "$g" ]] && continue
  echo "busy" >"$GPU_LOCK_DIR/gpu_${g}.lock"
done

NJOBS=$(grep -c . "$QUEUE" || true)
{
  echo "[start] $(date '+%F %T') stamp=$STAMP expand remaining=$NJOBS"
  echo "parallel=$MAX_PARALLEL need_mib=$NEED_MIB busy_gpus=$BUSY_GPUS"
  echo "soft=0.2 score=0.3 kappa=1.4/1.0 confirm=0.7 vmin=$VMIN det=$DET bev_tau=$BEV_TAU"
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
  # never release pre-locked busy GPUs
  local g="$1"
  for b in "${_BUSY[@]:-}"; do
    [[ "$g" == "$b" ]] && return 0
  done
  rm -f "${GPU_LOCK_DIR}/gpu_${g}.lock" 2>/dev/null || true
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
  # also skip if a live process is already writing this tag
  if pgrep -af "run_three_source.py" | grep -q -- "--save ${save}\|/${tag}_${STAMP}"; then
    echo "[skip] $tag already running" | tee -a "$STATUS"
    return 0
  fi
  local gpu=""
  for _ in $(seq 1 720); do
    gpu="$(pick_gpu || true)"
    [[ -n "$gpu" ]] && break
    sleep 20
  done
  if [[ -z "$gpu" ]]; then
    echo "[fail] no gpu for $tag" | tee -a "$STATUS"
    return 1
  fi
  echo "[run] $(date '+%F %T') gpu=$gpu tag=$tag mode=$mode/$level" | tee -a "$STATUS" | tee -a "$LOG"
  set +e
  "$PY" -W ignore scripts/run_three_source.py \
    --gpu "$gpu" --model attfuse --all_scenes --quiet \
    --score_thres 0.3 --theta_p 0.3 --theta_soft 0.2 \
    --det_score_thres "$DET" \
    --gate_q_mode p --buf_q_mode p \
    --q_h 0.3 --q_l 0.2 \
    --theta_confirm 0.7 --theta_reject 0.1 \
    --t_timeout 10 --k_atk 10 --pool_k 3 \
    --kappa_pos 1.4 --kappa_neg 1.0 \
    --v_min "$VMIN" \
    --beta_v 0.5 --gamma_r 0.5 --tau 0.1 \
    --bev_matcher "$MATCHER" --bev_tau "$BEV_TAU" --bev_c_min "$BEV_CMIN" \
    --mode "$mode" --level "$level" \
    --early_remove_mode box --remove_select "$REMOVE_SELECT" \
    --n_objects 3 \
    --defense ours --asr_dist_thres 0 \
    --save "$save" >"$runlog" 2>&1
  local rc=$?
  set -e
  release_gpu "$gpu"
  echo "[done] $(date '+%F %T') rc=$rc tag=$tag gpu=$gpu" | tee -a "$STATUS" | tee -a "$LOG"
  return 0
}

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
    [[ "$running" -ge "$MAX_PARALLEL" ]] && sleep 15
  done
  run_job "$line" &
  pids+=("$!")
  running=$((running+1))
  sleep 3
done <"$QUEUE"

for pid in "${pids[@]:-}"; do
  wait "$pid" || true
done
echo "[all-done] $(date '+%F %T') stamp=$STAMP expand" | tee -a "$STATUS" | tee -a "$LOG"
