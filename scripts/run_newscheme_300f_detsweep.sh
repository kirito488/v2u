#!/usr/bin/env bash
# New scheme: P-gate + soft Ego + always-weight P0; continuous 300f (cases 0-29)
# Sweep OpenCOOD det_score_thres in {0.05, 0.1, 0.2}; clean + remove_early.
set -euo pipefail
ROOT=/data/hzy/lxt/CollaborativePerceptionDefense
PY=/home/hzy/miniconda3/envs/v2u4/bin/python
cd "$ROOT"
export PYTHONUNBUFFERED=1
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/newscheme_300f
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/status_${STAMP}.txt"
exec > >(tee -a "$LOGDIR/orchestrator_${STAMP}.log") 2>&1

echo "[start] $(date) newscheme 300f det sweep"
echo "status=$STATUS"

need_mib=${NEED_FREE_MIB:-18000}

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v need="$need_mib" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0>=need) print $1, $2}' \
    | sort -k2 -nr | head -1 | awk '{print $1}'
}

wait_gpu() {
  while true; do
    g=$(pick_gpu || true)
    if [[ -n "${g:-}" ]]; then
      echo "[gpu] picked $g free>=${need_mib}MiB at $(date)" >&2
      echo "$g"
      return 0
    fi
    echo "[wait] no GPU with >=${need_mib}MiB free @ $(date); sleep 60" >&2
    sleep 60
  done
}

run_one() {
  local det="$1" mode="$2" level="$3"
  local tag_d
  case "$det" in
    0.05) tag_d=005 ;;
    0.1|0.10) tag_d=010 ;;
    0.2|0.20) tag_d=020 ;;
    *) tag_d=$(echo "$det" | tr -d '.') ;;
  esac
  local tag
  if [[ "$mode" == "none" ]]; then
    tag="clean_det${tag_d}"
  else
    tag="remove_${level}_det${tag_d}"
  fi
  local out="$LOGDIR/newscheme_300f_${tag}_${STAMP}.json"
  local log="$LOGDIR/newscheme_300f_${tag}_${STAMP}.log"
  if [[ -f "$out" ]]; then
    echo "[skip] exists $out"
    return 0
  fi
  local gpu
  gpu=$(wait_gpu)
  echo "[run] det=$det mode=$mode level=$level gpu=$gpu -> $out" | tee -a "$STATUS"
  set +e
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/run_three_source.py \
    --gpu "$gpu" \
    --model attfuse \
    --cases 0-29 \
    --det_score_thres "$det" \
    --score_thres 0.3 \
    --theta_soft 0.05 \
    --theta_p 0.3 \
    --q_h 0.3 \
    --q_l 0.05 \
    --gate_q_mode p \
    --buf_q_mode p \
    --n_ref 40 \
    --n_ref_uav 25 \
    --mode "$mode" \
    --level "$level" \
    --defense ours \
    -q \
    --save "$out" \
    >"$log" 2>&1
  local rc=$?
  set -e
  echo "[done] rc=$rc det=$det mode=$mode $(date) log=$log" | tee -a "$STATUS"
  if [[ $rc -ne 0 ]]; then
    echo "[warn] non-zero rc; continue next job" | tee -a "$STATUS"
  fi
  return 0
}

for det in 0.2 0.1 0.05; do
  run_one "$det" none early
  run_one "$det" remove early
done

echo "[all-done] $(date)"
