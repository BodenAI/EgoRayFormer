#!/usr/bin/env bash
# Training launcher: stage-1 heatmap pre-training, EgoRayFormer and the baselines (TSR-like, v1-like).
#
#   bash scripts/run_baselines.sh heatmap          # stage 1: 2D heatmap pre-training of the R18 encoder @256 (protocol independent)
#   bash scripts/run_baselines.sh v1like           # EgoPoseFormer-v1-like single-frame model, root + device protocol
#   bash scripts/run_baselines.sh tsr              # TSR-Ego-like (TFM + single-stage + absolute heads + age emb), T = 8
#   bash scripts/run_baselines.sh egorayformer          # EgoRayFormer full (P2 + P3 + P4), R18 @256, T = 8
#   bash scripts/run_baselines.sh all
#
# Heatmap stage of another encoder / resolution: HM_NAME names the run, HM_EXTRA is appended after stage_heatmap.yaml
# (res_*.yaml also set the e2e batch size, so restate the heatmap batch; heatmap_r18_256_rw ran 4 GPUs x 64 clips), e.g.
#   HM_NAME=heatmap_r18_384_rw HM_EXTRA="--config configs/res_384.yaml --data.batch_size 256 --trainer.strategy auto" \
#     TRAINSET=rw GPUS=7 bash scripts/run_baselines.sh heatmap
#   HM_NAME=heatmap_sfvit_s_256_rw HM_EXTRA="--config configs/enc_sfvit_s.yaml --data.batch_size 256 --trainer.strategy auto" \
#     TRAINSET=rw GPUS=7 bash scripts/run_baselines.sh heatmap          # P1 stage 1; the e2e runs then take HM_NAME too
# Environment: GPUS (default 4,5,6,7), DRY=1 prints the commands only, TRAINSET in {ue2rw (default), rw, ue2},
# PROTOS (root and/or device), EXTRA (extra CLI overrides appended to every fit / test command of the model runs) and
# RUN_SUFFIX (appended to the run names, so re-runs with other overrides do not overwrite earlier ones), e.g. the
# TSR-Ego training budget (1 GPU, batch 16, stride-4 windows):
#   GPUS=4 TRAINSET=rw PROTOS=root RUN_SUFFIX=_1gpu \
#     EXTRA="--trainer.strategy auto --data.batch_size 16 --data.clips_per_epoch_scale 2" bash scripts/run_baselines.sh tsr
# Equal step budget on any number of GPUs: GLOBAL_BATCH=16 [CLIPS_SCALE=2] splits the global batch over the
# devices.
# Every run reports device-/root-protocol metrics (mpjpe vs mpjpe_root); the *training* target frame is set by the data
# config: data_root_relative.yaml (TSR-Ego protocol) or data_device_relative.yaml + res_256.yaml (official leaderboard).
# Test split: a run is trained and validated only, then work_dirs/<run>/fit_done.json
# records completion and the best-validation checkpoint (scripts/ckpt_select.py). ALLOW_TEST=1 additionally tests the
# last and the best checkpoint - for a frozen configuration only (test numbers are reported after the freeze).
# Every fit is a new training attempt: `ckpt_select.py start` before it invalidates an earlier
# completion record of the same run name, so a forced re-run (SKIP_DONE=0) is never reported finished from the old one.
# Weights-only fine-tuning: INIT_FROM_RUN=<run> waits until that run finished, then initialises the
# model from its best-validation checkpoint (--model.init_from: weights only - optimizer, scheduler, epoch and step
# start fresh, the budget is this run's own; --ckpt_path would resume the source training instead).
# Dependent jobs: HM_WAIT=1 (stage-1 run HM_NAME) and INIT_FROM_RUN start from the source's finished
# current attempt only; a source whose attempt stopped without a completion record (failed, killed) is an error. The
# exit status of this script is the one of the failing step (set -e, pipefail), and no completion record is written
# for a failed fit.
# PRECHECK=1: every fit's arguments are built as for the real launch (heatmap initialisation,
# INIT_FROM_RUN, EXTRA) and only parsed - ``run.py fit ... --print_config`` into work_dirs/precheck/<run>.yaml, then
# scripts/precheck_config.py reports the derived quantities (also <run>.json). No training, no attempt, no waiting:
# HM_WAIT / INIT_FROM_RUN sources that have not finished are reported as pending.
# Run inside tmux, with the Python environment activated, from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${ROOT:-/data/unrealego/processed}
GPUS=${GPUS:-4,5,6,7}
NDEV=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
TRAINSET=${TRAINSET:-ue2rw}
STAGES=("${@:-all}")   # one or more of: heatmap v1like tsr egorayformer all
for s in "${STAGES[@]}"; do
  case "$s" in heatmap|v1like|tsr|egorayformer|all) ;; *) echo "unknown stage '$s' (heatmap | v1like | tsr | egorayformer | all)" >&2; exit 1 ;; esac
done
want() { for s in "${STAGES[@]}"; do [ "$s" = "$1" ] || [ "$s" = all ] && return 0; done; return 1; }
export CUDA_VISIBLE_DEVICES=$GPUS

case "$TRAINSET" in
  ue2rw) TS_CFG="" ; TAG="ue2rw" ; NEED="unrealego2 unrealego_rw" ;;
  rw)    TS_CFG="--config configs/data_rw_only.yaml" ; TAG="rw" ; NEED="unrealego_rw" ;;
  ue2)   TS_CFG="--config configs/data_ue2_only.yaml" ; TAG="ue2" ; NEED="unrealego2" ;;
  *) echo "unknown TRAINSET $TRAINSET" >&2; exit 1 ;;
esac
for ds in $NEED; do
  if [ -z "${DRY:-}" ] && [ ! -f "$ROOT/$ds/train/index.json" ]; then
    echo "processed $ds records not found under $ROOT - run scripts/stage_unrealego2_stream.sh / scripts/stage_unrealego_rw.sh first" >&2
    exit 1
  fi
done
HM_NAME=${HM_NAME:-heatmap_r18_256_$TAG}
PROTOS=${PROTOS:-root device}   # protocols to train: root (TSR-Ego) and/or device (leaderboard); lets two GPU groups split the work
latest_ckpt() {  # latest_ckpt <run name> -> newest last.ckpt written under work_dirs/<name> (empty if none)
  { ls -t work_dirs/"$1"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null || true; } | head -1
}
best_ckpt() {  # best_ckpt <run name> -> best-validation checkpoint (scripts/ckpt_select.py); empty if it is unknown
  python scripts/ckpt_select.py best "$1" 2>/dev/null || true
}
running() { pgrep -f -- "default_root_dir work_dirs/$1 " >/dev/null; }
wait_finished() {  # wait_finished <run> <max seconds>: until the current training attempt of <run> has its completion
  # record. A run that has not started (no fit_attempt.json) or is training is waited for; an attempt that stopped
  # without a record (failed or killed) is an error, so a dependent job never starts from a failed stage.
  # Re-queue the dependent job after re-running the stopped one.
  local run=$1 max=$2 waited=0 poll=${WAIT_POLL:-60}
  until python scripts/ckpt_select.py finished "$run"; do
    if [ -f "work_dirs/$run/fit_attempt.json" ] && ! running "$run"; then
      sleep "$poll"   # the record follows the end of the fit within seconds; the fit may also be starting up
      python scripts/ckpt_select.py finished "$run" && return 0
      if ! running "$run"; then
        echo "$run: its current training attempt stopped without a completion record (failed or killed)" >&2; return 1
      fi
    fi
    [ "$waited" -ge "$max" ] && { echo "$run: not finished after $max s" >&2; return 1; }
    sleep "$poll"; waited=$((waited + poll))
  done
}
logger_json() {  # every run logs (CSV) and checkpoints under work_dirs/<name>/lightning_logs/version_*/
  printf '[{"class_path":"pytorch_lightning.loggers.CSVLogger","init_args":{"save_dir":"work_dirs/%s"}}]' "$1"
}
test_ckpt() {  # test_ckpt <name> <ckpt> <suffix> <args...> (ALLOW_TEST=1 only)
  local name=$1 ck=$2 suffix=$3; shift 3
  echo "=== test $name $suffix ($ck)"
  EGORAYFORMER_ALLOW_TEST=1 python run.py test "$@" --trainer.devices 1 --trainer.default_root_dir "work_dirs/$name" \
    --trainer.logger "$(logger_json "$name")" --ckpt_path "$ck" 2>&1 | tee "work_dirs/logs/${name}_test${suffix}.log"
}
precheck() {  # precheck <name> <args...>: the fit of run(), parsed by LightningCLI only (PRECHECK=1)
  local name=$1; shift
  mkdir -p work_dirs/precheck
  local out="work_dirs/precheck/$name.yaml"
  echo "=== precheck $name"
  python run.py fit "$@" --trainer.devices "$NDEV" --trainer.default_root_dir "work_dirs/$name" \
    --trainer.logger "$(logger_json "$name")" --print_config > "$out"
  python scripts/precheck_config.py "$out" --name "$name" --json "work_dirs/precheck/$name.json" --pending "${PENDING:-}"
  if python scripts/ckpt_select.py finished "$name"; then echo "  note: the current training attempt of $name has finished"; fi
}
run() {  # run <name> <args...>: a new training attempt, fit + validation, then the completion record
  # work_dirs/<name>/fit_done.json of that attempt; the last and the best-validation checkpoints are tested only with
  # ALLOW_TEST=1. A run whose current attempt finished is skipped (SKIP_DONE=0 to force a new attempt), so pipelines
  # can be relaunched; a test log counts only for runs of the old flow without attempts (an attempt
  # that failed after an earlier successful test is trained again).
  local name=$1; shift
  mkdir -p work_dirs/logs
  if [ -n "${PRECHECK:-}" ]; then precheck "$name" "$@"; return; fi
  if [ "${SKIP_DONE:-1}" = 1 ] && [ -z "${DRY:-}" ] && { python scripts/ckpt_select.py finished "$name" || \
       { [ ! -f "work_dirs/$name/fit_attempt.json" ] && grep -qs "test/" "work_dirs/logs/${name}_test.log"; }; }; then
    echo "=== $name: already trained (current attempt finished, work_dirs/$name/fit_done.json) - skipping"; return 0
  fi
  echo "=== $name"; echo "python run.py fit $* --trainer.devices $NDEV --trainer.default_root_dir work_dirs/$name"
  if [ -z "${DRY:-}" ]; then
    local att
    att=$(python scripts/ckpt_select.py start "$name") || { echo "!! $name: no new training attempt - not training" >&2; return 1; }
    echo "training attempt $att"
    python run.py fit "$@" --trainer.devices "$NDEV" --trainer.default_root_dir "work_dirs/$name" \
      --trainer.logger "$(logger_json "$name")" 2>&1 | tee "work_dirs/logs/$name.log"
    python scripts/ckpt_select.py record "$name" > /dev/null || echo "!! $name: completion record without last.ckpt" >&2
    if [ "${ALLOW_TEST:-0}" = 1 ] && [ -z "${NO_TEST:-}" ] && ! echo "$*" | grep -q stage_heatmap; then
      local ck best; ck=$(latest_ckpt "$name"); best=$(best_ckpt "$name")
      [ -n "$best" ] || echo "!! $name: best-validation checkpoint unknown (python scripts/ckpt_select.py best $name)" >&2
      [ -n "$ck" ] && test_ckpt "$name" "$ck" "" "$@"
      if [ -n "$best" ] && [ "$(readlink -f "$best")" != "$(readlink -f "$ck")" ]; then test_ckpt "$name" "$best" "_best" "$@"; fi
    fi
  fi
}
PROTO_ROOT="--config configs/base_unrealego.yaml --config configs/data_root_relative.yaml $TS_CFG"
PROTO_DEV="--config configs/base_unrealego.yaml --config configs/data_device_relative.yaml --config configs/res_256.yaml $TS_CFG"
proto() { for p in $PROTOS; do [ "$p" = "$1" ] && return 0; done; return 1; }

if want heatmap; then
  # shellcheck disable=SC2086
  run "$HM_NAME" $PROTO_ROOT --config configs/stage_heatmap.yaml ${HM_EXTRA:-}
fi
EXTRA=${EXTRA:-}
PENDING=""
if [ -n "${INIT_FROM_RUN:-}" ] && ! want heatmap; then  # weights-only fine-tuning from another run's best checkpoint
  if [ -n "${PRECHECK:-}" ]; then  # resolved, never waited for
    if python scripts/ckpt_select.py finished "$INIT_FROM_RUN" && INIT_CKPT=$(python scripts/ckpt_select.py best "$INIT_FROM_RUN"); then :
    else
      INIT_CKPT="<pending: best checkpoint of $INIT_FROM_RUN>"; PENDING="INIT_FROM_RUN $INIT_FROM_RUN has not finished"
    fi
  elif [ -z "${DRY:-}" ]; then
    wait_finished "$INIT_FROM_RUN" 172800 || { echo "INIT_FROM_RUN: no finished $INIT_FROM_RUN to start from" >&2; exit 1; }
    INIT_CKPT=$(python scripts/ckpt_select.py best "$INIT_FROM_RUN") \
      || { echo "INIT_FROM_RUN: no best-validation checkpoint of $INIT_FROM_RUN" >&2; exit 1; }
  else
    INIT_CKPT="<best checkpoint of $INIT_FROM_RUN>"
  fi
  echo "weights-only initialisation from $INIT_CKPT"
  INIT_ARGS=(--model.init_from "$INIT_CKPT")
else
  if [ -n "${HM_WAIT:-}" ] && ! want heatmap && [ -z "${DRY:-}" ] && [ -z "${PRECHECK:-}" ]; then  # queued behind HM_NAME
    wait_finished "$HM_NAME" 43200 || { echo "HM_WAIT: no finished $HM_NAME to start from" >&2; exit 1; }
  fi
  # the last.ckpt of the stage-1 run's finished current attempt - never the checkpoint of a stopped or failed one
  HM_CKPT=${HM_CKPT:-$(python scripts/ckpt_select.py last "$HM_NAME" 2>/dev/null || true)}
  if [ -n "${PRECHECK:-}" ] && [ -z "$HM_CKPT" ]; then
    HM_CKPT="<pending: last.ckpt of $HM_NAME>"; PENDING="HM_NAME $HM_NAME has not finished"
  fi
  if [ -z "${DRY:-}" ] && [ -z "$HM_CKPT" ]; then
    echo "no heatmap checkpoint found under work_dirs/$HM_NAME - run the heatmap stage first (or set HM_CKPT)" >&2; exit 1
  fi
  if echo "$EXTRA" | grep -q "enc_sfvit" && echo "${HM_CKPT:-}" | grep -q "heatmap_r18"; then
    echo "EXTRA selects a StereoFisheyeViT encoder but HM_CKPT is an R18 heatmap checkpoint ($HM_CKPT):" \
         "set HM_NAME (or HM_CKPT) to the ViT stage-1 run, e.g. HM_NAME=heatmap_sfvit_s_256_rw" >&2
    exit 1
  fi
  INIT_ARGS=(--model.encoder_pretrained "$HM_CKPT")
fi
# Equal step budgets across GPU counts - GLOBAL_BATCH clips per optimizer step split over the devices and a
# fixed clips_per_epoch_scale (CLIPS_SCALE, default 2), so a 4-GPU run sees the same number of steps as a 1-GPU run.
if [ -n "${GLOBAL_BATCH:-}" ]; then
  PER_DEV=$(( GLOBAL_BATCH / NDEV ))
  [ "$PER_DEV" -ge 1 ] || { echo "GLOBAL_BATCH=$GLOBAL_BATCH smaller than the number of GPUs ($NDEV)" >&2; exit 1; }
  EXTRA="$EXTRA --data.batch_size $PER_DEV --data.clips_per_epoch_scale ${CLIPS_SCALE:-2}"
  echo "step budget: global batch $GLOBAL_BATCH = $NDEV x $PER_DEV clips, clips_per_epoch_scale ${CLIPS_SCALE:-2}"
fi
S=${RUN_SUFFIX:-}
if want v1like; then
  proto root   && run v1like_root_$TAG$S   $PROTO_ROOT --config configs/ablation_v1_like.yaml "${INIT_ARGS[@]}" $EXTRA
  proto device && run v1like_device_$TAG$S $PROTO_DEV  --config configs/ablation_v1_like.yaml "${INIT_ARGS[@]}" $EXTRA
fi
if want tsr; then
  proto root   && run tsr_root_$TAG$S   $PROTO_ROOT --config configs/baseline_tsr_ego.yaml "${INIT_ARGS[@]}" $EXTRA
  proto device && run tsr_device_$TAG$S $PROTO_DEV  --config configs/baseline_tsr_ego.yaml "${INIT_ARGS[@]}" $EXTRA
fi
if want egorayformer; then
  proto root   && run egorayformer_root_$TAG$S   $PROTO_ROOT --config configs/egorayformer_full.yaml "${INIT_ARGS[@]}" $EXTRA
  proto device && run egorayformer_device_$TAG$S $PROTO_DEV  --config configs/egorayformer_full.yaml "${INIT_ARGS[@]}" $EXTRA
fi
echo "done (${STAGES[*]}, trainset $TRAINSET, protocols $PROTOS)"
