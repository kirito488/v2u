#!/bin/bash
# Soft02 low-concurrency local queue: at most 1 GPU job OR 1 offline agg.
# Never overlap agg with GPU runs. Resume via ckpt/--save.
set -u
ROOT=/data/hzy/lxt/CollaborativePerceptionDefense
cd "$ROOT"
STAMP=20260909_112324
OUT=$ROOT/logs/attack_suite_soft02
PY=/home/hzy/miniconda3/envs/v2u4/bin/python
MATCHER=$ROOT/logs/bev_matcher/matcher_last.pth
STATUS=$OUT/status_serial_${STAMP}.txt
LOCK=$OUT/serial_queue.lock
GPU_NEED_MB=18000
AGG_NEED_GB=18
MAX_GPU=1   # hard cap

mkdir -p "$OUT"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[serial] another queue holds $LOCK — exit" >&2
  exit 0
fi
exec >>"$STATUS" 2>&1
echo "[serial] start $(date '+%F %T') MAX_GPU=$MAX_GPU AGG_NEED_GB=$AGG_NEED_GB pid=$$"

COMMON=(--model attfuse --all_scenes --quiet
  --score_thres 0.3 --theta_p 0.3 --theta_soft 0.2 --det_score_thres 0.2
  --gate_q_mode p --buf_q_mode p --q_h 0.3 --q_l 0.2
  --theta_confirm 0.7 --theta_reject 0.1 --t_timeout 10 --k_atk 10 --pool_k 3
  --kappa_pos 1.4 --kappa_neg 1.0 --v_min 0.80 --beta_v 0.5 --gamma_r 0.5 --tau 0.1
  --bev_matcher "$MATCHER" --bev_tau 0.6 --bev_c_min 0.2
  --early_remove_mode box --remove_select prefer --n_objects 3
  --defense ours --asr_dist_thres 0)

# tag|mode|level
AGG_QUEUE=(
  "spoof_early|spoof|early"
  "remove_early|remove|early"
  "multi_spoof_intermediate|multi_spoof|intermediate"
)
GPU_QUEUE=(
  "multi_spoof_intermediate|multi_spoof|intermediate"
  "spoof_intermediate|spoof|intermediate"
  "remove_intermediate|remove|intermediate"
  "mass_remove_intermediate|mass_remove|intermediate"
)

has_json(){ [[ -f "$OUT/${1}_${STAMP}.json" ]]; }
ckpt_done(){
  local m="$OUT/${1}_${STAMP}_ckpt/manifest.json"
  [[ -f "$m" ]] || { echo 0; return; }
  python3 -c "import json; print(len(json.load(open('$m')).get('done') or []))" 2>/dev/null || echo 0
}
is_run(){ pgrep -af "run_three_source.py" | grep -E -- "--save ([^ ]*/)?${1}_${STAMP}\.json" >/dev/null; }
is_agg(){ pgrep -af "aggregate_ckpt_metrics.py" | grep -F "${1}_${STAMP}_ckpt" >/dev/null; }
n_gpu_jobs(){ pgrep -af "run_three_source.py" | grep attack_suite_soft02 | grep -v pgrep | wc -l; }
n_agg_jobs(){ pgrep -af "aggregate_ckpt_metrics.py" | grep attack_suite_soft02 | grep -v pgrep | wc -l; }

mem_avail_gb(){
  awk '/MemAvailable:/ {printf "%.1f", $2/1024/1024}' /proc/meminfo
}

pick_gpu(){
  local best="" bf=0 idx free
  while IFS=, read -r idx free; do
    idx=$(echo "$idx" | tr -d ' ')
    free=$(echo "$free" | tr -d ' ')
    case "$idx" in 3|5|6|7) ;; *) continue;; esac
    [[ "$free" -lt "$GPU_NEED_MB" ]] && continue
    pgrep -af "run_three_source.py --gpu ${idx} " | grep attack_suite_soft02 >/dev/null && continue
    if [[ "$free" -gt "$bf" ]]; then bf=$free; best=$idx; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  echo "$best"
}

run_agg(){
  local tag="$1" mode="$2" level="$3"
  has_json "$tag" && return 0
  local nd
  nd=$(ckpt_done "$tag")
  [[ "$nd" -lt 10 ]] && return 1
  echo "[agg] $(date '+%F %T') $tag (wait RAM>=${AGG_NEED_GB}G)"
  while true; do
    local av; av=$(mem_avail_gb)
    # no GPU jobs while aggregating
    if [[ $(n_gpu_jobs) -gt 0 ]]; then
      echo "  wait: gpu jobs still alive av=${av}G"; sleep 30; continue
    fi
    if awk -v a="$av" -v n="$AGG_NEED_GB" 'BEGIN{exit !(a+0>=n+0)}'; then
      echo "  start agg av=${av}G"
      break
    fi
    echo "  wait RAM av=${av}G < ${AGG_NEED_GB}G"; sleep 30
  done
  $PY -u scripts/aggregate_ckpt_metrics.py \
    --ckpt_dir "$OUT/${tag}_${STAMP}_ckpt" \
    --save "$OUT/${tag}_${STAMP}.json" \
    --mode "$mode" --level "$level" --q_h 0.3 --q_l 0.2 \
    >"$OUT/${tag}_${STAMP}_agg.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 && -f "$OUT/${tag}_${STAMP}.json" ]]; then
    echo "[agg-ok] $tag rc=$rc"
  else
    echo "[agg-fail] $tag rc=$rc — will retry later"
  fi
  return $rc
}

run_gpu(){
  local tag="$1" mode="$2" level="$3"
  has_json "$tag" && return 0
  local nd; nd=$(ckpt_done "$tag")
  [[ "$nd" -ge 10 ]] && return 0  # leave to agg
  # only one at a time
  while [[ $(n_gpu_jobs) -ge $MAX_GPU ]] || [[ $(n_agg_jobs) -gt 0 ]]; do
    echo "[gpu-wait] $(date '+%H:%M:%S') n_gpu=$(n_gpu_jobs) n_agg=$(n_agg_jobs) for $tag"
    sleep 45
  done
  # need headroom so job + OS + DiMo don't OOM
  while true; do
    local av; av=$(mem_avail_gb)
    awk -v a="$av" 'BEGIN{exit !(a+0>=22)}' && break
    echo "[gpu-wait-ram] $tag av=${av}G < 22G"; sleep 40
  done
  local g; g=$(pick_gpu)
  if [[ -z "$g" ]]; then
    echo "[gpu-skip] no free GPU for $tag"; return 1
  fi
  echo "[gpu] $(date '+%F %T') gpu=$g resume $tag ckpt=${nd}/10"
  $PY -W ignore scripts/run_three_source.py --gpu "$g" "${COMMON[@]}" \
    --mode "$mode" --level "$level" \
    --save "$OUT/${tag}_${STAMP}.json" \
    >>"$OUT/${tag}_${STAMP}.log" 2>&1
  local rc=$?
  echo "[gpu-done] $tag rc=$rc ckpt=$(ckpt_done "$tag")/10"
  return 0
}

# --- phase 1: finish JSON for any 10/10 tags (serial agg) ---
echo "[phase1] offline agg for completed ckpts"
for item in "${AGG_QUEUE[@]}"; do
  IFS='|' read -r tag mode level <<<"$item"
  has_json "$tag" && { echo "[skip-json] $tag"; continue; }
  nd=$(ckpt_done "$tag")
  if [[ "$nd" -ge 10 ]]; then
    run_agg "$tag" "$mode" "$level" || true
  else
    echo "[agg-defer] $tag ckpt=${nd}/10"
  fi
done

# --- phase 2: GPU one-by-one, agg immediately when a tag hits 10/10 ---
echo "[phase2] GPU serial + immediate agg"
for item in "${GPU_QUEUE[@]}"; do
  IFS='|' read -r tag mode level <<<"$item"
  has_json "$tag" && { echo "[skip-json] $tag"; continue; }
  # loop until json exists (run -> maybe agg)
  while ! has_json "$tag"; do
    nd=$(ckpt_done "$tag")
    if [[ "$nd" -ge 10 ]]; then
      run_agg "$tag" "$mode" "$level" || sleep 60
      continue
    fi
    run_gpu "$tag" "$mode" "$level" || sleep 60
    # if process exited early, loop resumes
  done
  echo "[done] $tag"
done

echo "[serial] ALL DONE $(date '+%F %T')"
