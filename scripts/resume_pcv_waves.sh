#!/usr/bin/env bash
# Resume unfinished Q / P-gate / Tentative full-val jobs from ckpt.
# Does NOT use $(launch) — that ran jobs in a subshell so wait() returned immediately.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2u4

OUTDIR="${OUTDIR:-logs}"
mkdir -p "$OUTDIR" "$OUTDIR/experiment"
STATUS="$OUTDIR/resume_pcv_waves_status.txt"
N_REF="${N_REF:-200}"
SCORE="${SCORE:-0.3}"
Q_TS="${Q_TS:-20260907_204018}"
P_TS="${P_TS:-20260908_010227}"
TENT_TS="${TENT_TS:-20260908_010401}"
# GPU 2 is occupied by DiMo-VAD
SKIP_GPUS="${SKIP_GPUS:-2}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS" >&2; }

json_ok() {
  python - "$1" <<'PY'
import json, os, sys
p = sys.argv[1]
if not os.path.isfile(p) or os.path.getsize(p) < 10000:
    raise SystemExit(1)
d = json.load(open(p))
if len(d.get("frames") or []) < 50:
    raise SystemExit(2)
PY
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v skip=" $SKIP_GPUS " '{
        gsub(/ /,"",$1); gsub(/ /,"",$2);
        if (index(skip, " "$1" ")==0 && $2+0 < 2500) print $1
      }'
}

wait_alive() {
  local left
  while true; do
    left=()
    for pid in "$@"; do
      if kill -0 "$pid" 2>/dev/null; then
        left+=("$pid")
      fi
    done
    if [[ ${#left[@]} -eq 0 ]]; then
      log "wave PIDs all exited"
      return 0
    fi
    log "waiting ${#left[@]} job(s): ${left[*]}"
    sleep 60
  done
}

# Launch in THIS shell. Never wrap this in $(...) — that orphans the job
# and makes wait() return immediately.
launch_job() {
  local GPU="$1" KIND="$2" MODE="$3" LEVEL="$4" TAG="$5" SAVE="$6"
  shift 6
  local LOG="${SAVE%.json}.log"
  if json_ok "$SAVE"; then
    log "SKIP already-ok $SAVE"
    return 0
  fi
  local EXTRA=()
  if [[ -n "$LEVEL" && "$LEVEL" != "-" && "$LEVEL" != '""' ]]; then
    EXTRA+=(--level "$LEVEL")
  fi
  log "launch gpu=$GPU $KIND $TAG -> $SAVE $*"
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
    >>"$LOG" 2>&1 &
  WAVE_PIDS+=($!)
  log "  pid=$!"
}

next_gpu() {
  local g cand
  while true; do
    mapfile -t g < <(free_gpus)
    for cand in "${g[@]}"; do
      # Prefer a GPU not reserved this wave; reuse if reservation is idle
      # (memory already <2500 means the previous job on it is gone).
      echo "$cand"
      return 0
    done
    log "no free GPU (skip=$SKIP_GPUS); sleep 20"
    sleep 20
  done
}

run_wave() {
  local name="$1"
  local line KIND MODE LEVEL TAG SAVE GPU extras
  WAVE_PIDS=()
  WAVE_USED=""
  log "===== $name ====="
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    read -r KIND MODE LEVEL TAG SAVE extras <<<"$line"
    if json_ok "$SAVE"; then
      log "SKIP $SAVE"
      continue
    fi
    if pgrep -af -- "--save $SAVE" | grep -v grep >/dev/null; then
      local live
      live="$(pgrep -af -- "--save $SAVE" | grep -v grep | awk '{print $1}' | head -1)"
      log "ATTACH already-running pid=$live $SAVE"
      WAVE_PIDS+=("$live")
      continue
    fi
    GPU="$(next_gpu)"
    WAVE_USED+=" $GPU"
    # extras unquoted: --q_h 0.40 ...
    # shellcheck disable=SC2086
    launch_job "$GPU" "$KIND" "$MODE" "$LEVEL" "$TAG" "$SAVE" $extras
    sleep 6
  done
  if [[ ${#WAVE_PIDS[@]} -eq 0 ]]; then
    log "$name: nothing to run"
    return 0
  fi
  log "$name waiting ${WAVE_PIDS[*]}"
  wait_alive "${WAVE_PIDS[@]}"
}

log "===== resume PCV waves start ====="

# Wave 1 leftovers (Q=P·C)
run_wave "wave1 Q=P·C" <<EOF
clean none - q040_030 $OUTDIR/qablate_full_clean_q040_030_${Q_TS}.json --q_h 0.40 --q_l 0.30
remove_early remove early q042_032 $OUTDIR/qablate_full_remove_early_q042_032_${Q_TS}.json --q_h 0.42 --q_l 0.32
clean none - q042_032 $OUTDIR/qablate_full_clean_q042_032_${Q_TS}.json --q_h 0.42 --q_l 0.32
remove_early remove early q040_030 $OUTDIR/qablate_full_remove_early_q040_030_${Q_TS}.json --q_h 0.40 --q_l 0.30
clean none - q035_025 $OUTDIR/qablate_full_clean_q035_025_${Q_TS}.json --q_h 0.35 --q_l 0.25
remove_early remove early q038_030 $OUTDIR/qablate_full_remove_early_q038_030_${Q_TS}.json --q_h 0.38 --q_l 0.30
EOF

python -u scripts/summarize_q_ablation_cabs_v.py --ts "$Q_TS" --outdir "$OUTDIR" --full \
  >>"$OUTDIR/experiment/q_ablation_cabs_v_full_${Q_TS}.stdout" 2>&1 || log "Q summarize errors"

# Wave 2 P-only gate
run_wave "wave2 P-only gate" <<EOF
clean none - p035_025 $OUTDIR/pgate_full_clean_p035_025_${P_TS}.json --q_h 0.35 --q_l 0.25 --gate_q_mode p
remove_early remove early p035_025 $OUTDIR/pgate_full_remove_early_p035_025_${P_TS}.json --q_h 0.35 --q_l 0.25 --gate_q_mode p
clean none - p040_030 $OUTDIR/pgate_full_clean_p040_030_${P_TS}.json --q_h 0.40 --q_l 0.30 --gate_q_mode p
remove_early remove early p040_030 $OUTDIR/pgate_full_remove_early_p040_030_${P_TS}.json --q_h 0.40 --q_l 0.30 --gate_q_mode p
clean none - p042_032 $OUTDIR/pgate_full_clean_p042_032_${P_TS}.json --q_h 0.42 --q_l 0.32 --gate_q_mode p
remove_early remove early p042_032 $OUTDIR/pgate_full_remove_early_p042_032_${P_TS}.json --q_h 0.42 --q_l 0.32 --gate_q_mode p
clean none - p038_030 $OUTDIR/pgate_full_clean_p038_030_${P_TS}.json --q_h 0.38 --q_l 0.30 --gate_q_mode p
remove_early remove early p038_030 $OUTDIR/pgate_full_remove_early_p038_030_${P_TS}.json --q_h 0.38 --q_l 0.30 --gate_q_mode p
EOF

# Wave 3 Tentative P
run_wave "wave3 Tentative P" <<EOF
clean none - c050_ql030 $OUTDIR/tentp_full_clean_c050_ql030_${TENT_TS}.json --q_h 0.40 --q_l 0.30 --gate_q_mode p --buf_q_mode p --theta_confirm 0.50
remove_early remove early c050_ql030 $OUTDIR/tentp_full_remove_early_c050_ql030_${TENT_TS}.json --q_h 0.40 --q_l 0.30 --gate_q_mode p --buf_q_mode p --theta_confirm 0.50
clean none - c070_ql030 $OUTDIR/tentp_full_clean_c070_ql030_${TENT_TS}.json --q_h 0.40 --q_l 0.30 --gate_q_mode p --buf_q_mode p --theta_confirm 0.70
remove_early remove early c070_ql030 $OUTDIR/tentp_full_remove_early_c070_ql030_${TENT_TS}.json --q_h 0.40 --q_l 0.30 --gate_q_mode p --buf_q_mode p --theta_confirm 0.70
clean none - c090_ql030 $OUTDIR/tentp_full_clean_c090_ql030_${TENT_TS}.json --q_h 0.40 --q_l 0.30 --gate_q_mode p --buf_q_mode p --theta_confirm 0.90
remove_early remove early c090_ql030 $OUTDIR/tentp_full_remove_early_c090_ql030_${TENT_TS}.json --q_h 0.40 --q_l 0.30 --gate_q_mode p --buf_q_mode p --theta_confirm 0.90
clean none - c070_ql025 $OUTDIR/tentp_full_clean_c070_ql025_${TENT_TS}.json --q_h 0.40 --q_l 0.25 --gate_q_mode p --buf_q_mode p --theta_confirm 0.70
remove_early remove early c070_ql025 $OUTDIR/tentp_full_remove_early_c070_ql025_${TENT_TS}.json --q_h 0.40 --q_l 0.25 --gate_q_mode p --buf_q_mode p --theta_confirm 0.70
EOF

log "===== write report ====="
python -u scripts/summarize_overnight_pcv.py \
  --q_ts "$Q_TS" --p_ts "$P_TS" --tent_ts "$TENT_TS" --outdir "$OUTDIR" \
  >"$OUTDIR/experiment/MORNING_PCV_REPORT.stdout" 2>&1 || log "report failed"
log "===== resume done ====="
log "report: $OUTDIR/experiment/MORNING_PCV_REPORT.md"
