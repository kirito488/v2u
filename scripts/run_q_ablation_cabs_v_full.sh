#!/usr/bin/env bash
# Full-val Q_h/Q_l ablation: clean + remove_early, all scenes.
# C=C_abs*V, n_ref=200, 75-ray polar V.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2u4

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
N_REF="${N_REF:-200}"
SCORE="${SCORE:-0.3}"
OUTDIR="${OUTDIR:-logs}"
mkdir -p "$OUTDIR" "$OUTDIR/experiment"

# tag  q_h  q_l  gpu_clean  gpu_remove
PAIRS=(
  "q035_025 0.35 0.25 0 1"
  "q040_030 0.40 0.30 2 3"
  "q042_032 0.42 0.32 5 6"
  "q038_030 0.38 0.30 7 0"
)

echo "[qablate-full] TS=$TS all_scenes n_ref=$N_REF"
echo "$TS" > "$OUTDIR/q_ablation_cabs_v_full_${TS}.ts"

pids=()
launch() {
  local KIND="$1" MODE="$2" LEVEL="$3" TAG="$4" QH="$5" QL="$6" GPU="$7"
  local SAVE LOG EXTRA=()
  SAVE="$OUTDIR/qablate_full_${KIND}_${TAG}_${TS}.json"
  LOG="$OUTDIR/qablate_full_${KIND}_${TAG}_${TS}.log"
  if [[ -n "$LEVEL" ]]; then EXTRA=(--level "$LEVEL"); fi
  echo "[launch] gpu=$GPU $KIND $TAG Qh=$QH Ql=$QL"
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_three_source.py \
    --model attfuse \
    --score_thres "$SCORE" \
    --all_scenes \
    --mode "$MODE" \
    "${EXTRA[@]}" \
    --defense ours \
    -q \
    --n_ref "$N_REF" --n_ref_uav "$N_REF" \
    --q_h "$QH" --q_l "$QL" \
    --asr_dist_thres 0 \
    --save "$SAVE" \
    >"$LOG" 2>&1 &
  pids+=($!)
  sleep 12
}

# Wave A: all cleans (4 GPUs) — avoid colliding remove on GPU0 until wave B
for pair in "${PAIRS[@]}"; do
  read -r TAG QH QL GC GR <<<"$pair"
  launch clean none "" "$TAG" "$QH" "$QL" "$GC"
done

echo "[qablate-full] waiting clean jobs ${pids[*]}"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
pids=()

# Wave B: all remove_early
for pair in "${PAIRS[@]}"; do
  read -r TAG QH QL GC GR <<<"$pair"
  launch remove_early remove early "$TAG" "$QH" "$QL" "$GR"
done

echo "[qablate-full] waiting remove jobs ${pids[*]}"
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

python -u scripts/summarize_q_ablation_cabs_v.py --ts "$TS" --outdir "$OUTDIR" --full \
  || true

if [[ "$fail" -ne 0 ]]; then
  echo "[qablate-full] finished with failures TS=$TS"
  exit 1
fi
echo "[qablate-full] all done TS=$TS"
