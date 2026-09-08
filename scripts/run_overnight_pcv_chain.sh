#!/usr/bin/env bash
# Overnight chain (do not relaunch the broken Q-ablation wait loop):
#   0) wait for current qablate_full_*_20260907_204018 PIDs
#   1) summarize Q=P·C
#   2) P-only Ego partition (--gate_q_mode p), same Q_h/Q_l pairs
#   3) Tentative P pairs (--buf_q_mode p, several theta_confirm × q_l)
#   4) write logs/experiment/MORNING_PCV_REPORT.md
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2u4

OUTDIR="${OUTDIR:-logs}"
mkdir -p "$OUTDIR" "$OUTDIR/experiment"
STATUS="$OUTDIR/overnight_pcv_chain_status.txt"
N_REF="${N_REF:-200}"
SCORE="${SCORE:-0.3}"
Q_TS="${Q_TS:-20260907_204018}"
# PIDs of the currently running qablate_full jobs (as of 2026-09-07 22:20)
WAIT_PIDS="${WAIT_PIDS:-3418309 3419579 3419977 3421239 3421672 3422027 3422380}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$STATUS" >&2
}

json_ok() {
  local f="$1"
  python - "$f" <<'PY'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception:
    raise SystemExit(1)
print(len(d.get("frames") or []))
if len(d.get("frames") or []) < 50:
    raise SystemExit(2)
PY
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 2500) print $1}'
}

wait_pids() {
  local left
  while true; do
    left=()
    for pid in $WAIT_PIDS; do
      if kill -0 "$pid" 2>/dev/null; then
        left+=("$pid")
      fi
    done
    if [[ ${#left[@]} -eq 0 ]]; then
      log "qablate PIDs all exited"
      break
    fi
    log "waiting qablate PIDs: ${left[*]}"
    sleep 60
  done
}

wait_jobs() {
  local fail=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      fail=1
      log "job pid=$pid failed"
    fi
  done
  return "$fail"
}

launch_one() {
  local GPU="$1" KIND="$2" MODE="$3" LEVEL="$4" TAG="$5" SAVE="$6"
  shift 6
  local LOG="${SAVE%.json}.log"
  local EXTRA=()
  if [[ -n "$LEVEL" ]]; then EXTRA+=(--level "$LEVEL"); fi
  log "launch gpu=$GPU $KIND $TAG -> $SAVE extras=$*"
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_three_source.py \
    --model attfuse \
    --score_thres "$SCORE" \
    --all_scenes \
    --mode "$MODE" \
    "${EXTRA[@]}" \
    --defense ours \
    -q \
    --n_ref "$N_REF" --n_ref_uav "$N_REF" \
    --asr_dist_thres 0 \
    --save "$SAVE" \
    "$@" \
    >"$LOG" 2>&1 &
  echo $!
  sleep 8
}

assign_gpus() {
  local need="$1"
  local g=()
  local tries=0
  while [[ ${#g[@]} -lt $need && $tries -lt 90 ]]; do
    mapfile -t g < <(free_gpus)
    if [[ ${#g[@]} -ge $need ]]; then
      echo "${g[@]:0:$need}"
      return 0
    fi
    log "need $need free GPUs, have ${#g[@]} (${g[*]:-none}); sleep 30"
    sleep 30
    tries=$((tries + 1))
  done
  echo "${g[@]}"
}

P_PAIRS=(
  "p035_025 0.35 0.25"
  "p040_030 0.40 0.30"
  "p042_032 0.42 0.32"
  "p038_030 0.38 0.30"
)

# Tentative P pairs: (tag, theta_confirm, q_l) ; gate P-only, q_h=0.40, buf_q_mode=p
TENT_PAIRS=(
  "c050_ql030 0.50 0.30"
  "c070_ql030 0.70 0.30"
  "c090_ql030 0.90 0.30"
  "c070_ql025 0.70 0.25"
)

log "===== overnight PCV chain start Q_TS=$Q_TS ====="
wait_pids

log "summarize Q ablation $Q_TS"
python -u scripts/summarize_q_ablation_cabs_v.py --ts "$Q_TS" --outdir "$OUTDIR" --full \
  >"$OUTDIR/experiment/q_ablation_cabs_v_full_${Q_TS}.stdout" 2>&1 || log "Q summarize had errors"

P_TS="${P_TS:-$(date +%Y%m%d_%H%M%S)}"
echo "$P_TS" > "$OUTDIR/pgate_full_${P_TS}.ts"
log "===== wave P-only gate TS=$P_TS ====="

mapfile -t GPUS < <(assign_gpus 8 | tr ' ' '\n')
if [[ ${#GPUS[@]} -lt 8 ]]; then
  log "WARN only ${#GPUS[@]} GPUs; launching what we can"
fi
pids=()
gi=0
for pair in "${P_PAIRS[@]}"; do
  read -r TAG QH QL <<<"$pair"
  for spec in "clean none " "remove_early remove early"; do
    read -r KIND MODE LEVEL <<<"$spec"
    GPU="${GPUS[$((gi % ${#GPUS[@]}))]:-0}"
    gi=$((gi + 1))
    SAVE="$OUTDIR/pgate_full_${KIND}_${TAG}_${P_TS}.json"
    pids+=("$(launch_one "$GPU" "$KIND" "$MODE" "$LEVEL" "$TAG" "$SAVE" \
      --q_h "$QH" --q_l "$QL" --gate_q_mode p)")
  done
done
log "waiting P-gate jobs ${pids[*]}"
wait_jobs "${pids[@]}" || log "some P-gate jobs failed"

TENT_TS="${TENT_TS:-$(date +%Y%m%d_%H%M%S)}"
echo "$TENT_TS" > "$OUTDIR/tentp_full_${TENT_TS}.ts"
log "===== wave Tentative P TS=$TENT_TS ====="

mapfile -t GPUS < <(assign_gpus 8 | tr ' ' '\n')
pids=()
gi=0
QH_TENT="${QH_TENT:-0.40}"
for pair in "${TENT_PAIRS[@]}"; do
  read -r TAG TC QL <<<"$pair"
  for spec in "clean none " "remove_early remove early"; do
    read -r KIND MODE LEVEL <<<"$spec"
    GPU="${GPUS[$((gi % ${#GPUS[@]}))]:-0}"
    gi=$((gi + 1))
    SAVE="$OUTDIR/tentp_full_${KIND}_${TAG}_${TENT_TS}.json"
    pids+=("$(launch_one "$GPU" "$KIND" "$MODE" "$LEVEL" "$TAG" "$SAVE" \
      --q_h "$QH_TENT" --q_l "$QL" --gate_q_mode p \
      --buf_q_mode p --theta_confirm "$TC")")
  done
done
log "waiting Tentative jobs ${pids[*]}"
wait_jobs "${pids[@]}" || log "some Tentative jobs failed"

log "===== write morning report ====="
python -u scripts/summarize_overnight_pcv.py \
  --q_ts "$Q_TS" --p_ts "$P_TS" --tent_ts "$TENT_TS" --outdir "$OUTDIR" \
  >"$OUTDIR/experiment/MORNING_PCV_REPORT.stdout" 2>&1 || log "morning report script failed"

# sanity: list missing jsons
python - "$OUTDIR" "$Q_TS" "$P_TS" "$TENT_TS" <<'PY' | tee -a "$STATUS"
import sys
from pathlib import Path
outdir, qts, pts, tts = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
need = []
for tag in ("q035_025","q040_030","q042_032","q038_030"):
    for k in ("clean","remove_early"):
        need.append(outdir / f"qablate_full_{k}_{tag}_{qts}.json")
for tag in ("p035_025","p040_030","p042_032","p038_030"):
    for k in ("clean","remove_early"):
        need.append(outdir / f"pgate_full_{k}_{tag}_{pts}.json")
for tag in ("c050_ql030","c070_ql030","c090_ql030","c070_ql025"):
    for k in ("clean","remove_early"):
        need.append(outdir / f"tentp_full_{k}_{tag}_{tts}.json")
miss=[]
ok=[]
for p in need:
    if p.is_file() and p.stat().st_size > 10000:
        ok.append(p.name)
    else:
        miss.append(str(p))
print(f"json ok={len(ok)} missing={len(miss)}")
for m in miss:
    print("MISSING", m)
PY

log "===== overnight PCV chain done P_TS=$P_TS TENT_TS=$TENT_TS ====="
log "report: $OUTDIR/experiment/MORNING_PCV_REPORT.md"
