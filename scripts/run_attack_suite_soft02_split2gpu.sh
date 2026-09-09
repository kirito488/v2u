#!/usr/bin/env bash
# Split all-attack+clean suite across TWO pinned GPUs (sequential jobs each).
# Config: soft=0.2 score=0.3 kappa=1.4/1.0 confirm=0.7
#         T=10 Katk=10 K=3 beta=0.5 gamma=0.5 tau=0.1 Vmin=P5(0.80)
#         det=0.2 BEV tau=0.6 remove_select=prefer all_scenes defense=ours
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=/home/hzy/miniconda3/envs/v2u4/bin/python
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT_DIR:-logs/attack_suite_soft02}"
GPU_A="${GPU_A:-5}"
GPU_B="${GPU_B:-6}"
VMIN="${VMIN:-0.80}"
DET="${DET_SCORE_THRES:-0.2}"
MATCHER="${MATCHER:-$ROOT/logs/bev_matcher/matcher_last.pth}"
BEV_TAU="${BEV_TAU:-0.6}"
BEV_CMIN="${BEV_CMIN:-0.2}"
REMOVE_SELECT="${REMOVE_SELECT:-prefer}"

mkdir -p "$OUT"
STATUS="$OUT/status_split_${STAMP}.txt"
LOG="$OUT/orchestrator_split_${STAMP}.log"

# Part A (GPU_A): clean + all spoof + remove_early
PART_A=(
  "clean_none|none|early"
  "spoof_early|spoof|early"
  "spoof_intermediate|spoof|intermediate"
  "spoof_late|spoof|late"
  "remove_early|remove|early"
)
# Part B (GPU_B): remaining remove + multi/mass
PART_B=(
  "remove_intermediate|remove|intermediate"
  "remove_late|remove|late"
  "multi_spoof_intermediate|multi_spoof|intermediate"
  "mass_remove_intermediate|mass_remove|intermediate"
)

{
  echo "[start] $(date '+%F %T') stamp=$STAMP"
  echo "GPU_A=$GPU_A jobs=${#PART_A[@]} | GPU_B=$GPU_B jobs=${#PART_B[@]}"
  echo "soft=0.2 score=0.3 kappa=1.4/1.0 confirm=0.7 T/Katk/K=10/10/3"
  echo "beta/gamma/tau=0.5/0.5/0.1 vmin=$VMIN det=$DET bev_tau=$BEV_TAU remove_select=$REMOVE_SELECT"
  echo "compare_vs=logs/compare_*_20260905_182559 + clean_none_20260905_182559"
} | tee "$STATUS" | tee "$LOG"

run_one() {
  local gpu="$1" tag="$2" mode="$3" level="$4"
  local save="$OUT/${tag}_${STAMP}.json"
  local runlog="$OUT/${tag}_${STAMP}.log"
  if [[ -f "$save" ]]; then
    echo "[skip] $tag" | tee -a "$STATUS"
    return 0
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
  echo "[done] $(date '+%F %T') rc=$rc tag=$tag gpu=$gpu" | tee -a "$STATUS" | tee -a "$LOG"
  return 0
}

run_part() {
  local gpu="$1"; shift
  local spec
  for spec in "$@"; do
    IFS='|' read -r tag mode level <<<"$spec"
    run_one "$gpu" "$tag" "$mode" "$level"
  done
  echo "[part-done] gpu=$gpu $(date '+%F %T')" | tee -a "$STATUS" | tee -a "$LOG"
}

run_part "$GPU_A" "${PART_A[@]}" &
PID_A=$!
run_part "$GPU_B" "${PART_B[@]}" &
PID_B=$!
echo "[pids] A=$PID_A (gpu$GPU_A) B=$PID_B (gpu$GPU_B)" | tee -a "$STATUS"
wait "$PID_A" || true
wait "$PID_B" || true
echo "[all-done] $(date '+%F %T') stamp=$STAMP" | tee -a "$STATUS" | tee -a "$LOG"
