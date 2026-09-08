#!/usr/bin/env bash
# Q_h/Q_l ablation under C=C_abs*V, n_ref=200, 75-ray V.
# Usage: bash scripts/run_q_ablation_cabs_v.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2u4

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
CASES="${CASES:-0-19}"   # 200 consecutive frames
N_REF="${N_REF:-200}"
SCORE="${SCORE:-0.3}"
OUTDIR="${OUTDIR:-logs}"
mkdir -p "$OUTDIR"

# pairs: tag q_h q_l gpu
PAIRS=(
  "q035_025 0.35 0.25 0"
  "q040_030 0.40 0.30 1"
  "q042_032 0.42 0.32 2"
  "q038_030 0.38 0.30 3"
)

MODES=(
  "clean none"
  "remove_early remove"
)

echo "[qablate] TS=$TS cases=$CASES n_ref=$N_REF"
echo "[qablate] pairs: ${PAIRS[*]}"

pids=()
for pair in "${PAIRS[@]}"; do
  read -r TAG QH QL GPU <<<"$pair"
  for mode_spec in "${MODES[@]}"; do
    read -r KIND MODE <<<"$mode_spec"
    LEVEL_ARGS=()
    if [[ "$MODE" != "none" ]]; then
      LEVEL_ARGS=(--level early)
    fi
    SAVE="$OUTDIR/qablate_${KIND}_${TAG}_${TS}.json"
    LOG="$OUTDIR/qablate_${KIND}_${TAG}_${TS}.log"
    echo "[launch] $KIND $TAG Qh=$QH Ql=$QL gpu=$GPU -> $SAVE"
    CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_three_source.py \
      --model attfuse \
      --score_thres "$SCORE" \
      --cases "$CASES" \
      --mode "$MODE" \
      "${LEVEL_ARGS[@]}" \
      --defense ours \
      -q \
      --n_ref "$N_REF" --n_ref_uav "$N_REF" \
      --q_h "$QH" --q_l "$QL" \
      --asr_dist_thres 0 \
      --save "$SAVE" \
      >"$LOG" 2>&1 &
    pids+=($!)
    # stagger model loads slightly
    sleep 8
  done
done

echo "[qablate] waiting ${#pids[@]} jobs: ${pids[*]}"
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[qablate] job $pid failed"
    fail=1
  fi
done

python -u scripts/summarize_q_ablation_cabs_v.py --ts "$TS" --outdir "$OUTDIR" \
  || true

if [[ "$fail" -ne 0 ]]; then
  echo "[qablate] finished with failures"
  exit 1
fi
echo "[qablate] all done TS=$TS"
