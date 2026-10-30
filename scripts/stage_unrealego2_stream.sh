#!/usr/bin/env bash
# UnrealEgo2 staging that keeps temporary PNGs on disk (under $STAGE_DIR), one sequence at a time.
#
# For each environment: download the pose part and the rgb parts (resumable), extract the pose json,
# then stream the rgb tar. PNGs of at most MAX_INFLIGHT sequences sit under
# $STAGE_DIR/_stream_work/png and are deleted as soon as they have been turned into records.
# The downloaded .tar.gz parts are never deleted. GPUs are hidden from this process and it asks the kernel to be killed first if
# the machine runs out of memory.
#
#   bash scripts/fetch_unrealego_meta.sh
#   bash scripts/stage_unrealego2_stream.sh train_val
#   bash scripts/stage_unrealego2_stream.sh test
#   ENVS="Lab_Scene" bash scripts/stage_unrealego2_stream.sh train_val
#
# Environment: DATA_ROOT, STAGE_DIR, OUT_ROOT, SPLIT_TXT_DIR, IMG_SIZE (default 384),
# WORKERS / MAX_INFLIGHT (default 2; temporary PNGs are a few GB), MIN_FREE_GB (default 80),
# PART_GB (default 10, nominal size of one part; the free-space check counts only bytes not yet downloaded),
# PARALLEL (concurrent part downloads, default 1), ENVS (default: all, largest first).
set -euo pipefail

SPLIT="${1:-train_val}"
DATA_ROOT="${DATA_ROOT:-/data/unrealego}"
SERVER="https://unrealego.mpi-inf.mpg.de/data"
STAGE_DIR="${STAGE_DIR:-$DATA_ROOT/_staging_ue2}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/processed}"
SPLIT_TXT_DIR="${SPLIT_TXT_DIR:-$DATA_ROOT/UnrealEgoData2_splits}"
WORKERS="${WORKERS:-2}"
MAX_INFLIGHT="${MAX_INFLIGHT:-2}"
IMG_SIZE="${IMG_SIZE:-384}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"
PART_GB="${PART_GB:-10}"
PARALLEL="${PARALLEL:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$HERE/download_utils.sh"
# Hide the training GPUs. oom_score_adj makes the kernel reclaim this job before the trainers.
export CUDA_VISIBLE_DEVICES=""
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
trap 'echo "[stream] interrupted" >&2; trap - INT TERM; kill -TERM 0 2>/dev/null; exit 130' INT TERM

for sp in train validation test; do
  [ -f "$SPLIT_TXT_DIR/$sp.txt" ] || { echo "missing $SPLIT_TXT_DIR/$sp.txt - run scripts/fetch_unrealego_meta.sh first" >&2; exit 1; }
done
case "$SPLIT" in
  train_val) RECORD_SPLITS=(train validation) ;;
  test) RECORD_SPLITS=(test) ;;
  *) echo "split must be train_val or test, got $SPLIT" >&2; exit 2 ;;
esac

RGB_NAME="UnrealEgoData2_${SPLIT}_rgb"
POSE_NAME="UnrealEgoData2_${SPLIT}_pose"
mkdir -p "$STAGE_DIR"
# fd 9 is used inside poser_curl_resume; keep the run lock on fd 8.
exec 8>"$STAGE_DIR/.stream.lock"
flock -n 8 || { echo "[stream] another staging run holds $STAGE_DIR/.stream.lock" >&2; exit 1; }

free_gb() { df -BG --output=avail "${1:-$STAGE_DIR}" | tail -1 | tr -dc '0-9'; }

fetch_env_counts() {
  curl -sSL --http1.1 --retry 5 --retry-all-errors --retry-delay 2 \
    "$SERVER/download_unrealego2_${SPLIT}_$1.sh" | sed -n 's/^ *\["\([^"]*\)"\]=\([0-9]*\).*/\1 \2/p'
}

declare -A RGB_N POSE_N
while read -r env n; do [ -n "$env" ] && RGB_N["$env"]=$n; done < <(fetch_env_counts rgb)
while read -r env n; do [ -n "$env" ] && POSE_N["$env"]=$n; done < <(fetch_env_counts pose)
[ "${#RGB_N[@]}" -gt 0 ] || { echo "[stream] could not read the environment list" >&2; exit 1; }

download_parts() {  # $1 = archive name, $2 = env, $3 = part count. Does not extract.
  local name=$1 env=$2 n=$3 i part running=0
  mkdir -p "$STAGE_DIR/$name"
  for ((i = 0; i < n; i++)); do
    part=$(printf "%s.tar.gz%03d" "$env" "$i")
    [ -f "$STAGE_DIR/$name/.ok_$part" ] && continue
    echo "[stream] GET $name/$env/$part (have $(stat -c%s "$STAGE_DIR/$name/$part" 2>/dev/null || echo 0) bytes)"
    if [ "$PARALLEL" -le 1 ]; then
      poser_curl_resume "$SERVER/$name/$env/$part" "$STAGE_DIR/$name/$part" \
        || { echo "[stream] download of $name/$env/$part failed" >&2; exit 1; }
      touch "$STAGE_DIR/$name/.ok_$part"
    else
      (poser_curl_resume "$SERVER/$name/$env/$part" "$STAGE_DIR/$name/$part" && touch "$STAGE_DIR/$name/.ok_$part") &
      running=$((running + 1))
      if [ "$running" -ge "$PARALLEL" ]; then wait -n || true; running=$((running - 1)); fi
    fi
  done
  wait
  for ((i = 0; i < n; i++)); do
    part=$(printf "%s.tar.gz%03d" "$env" "$i")
    [ -f "$STAGE_DIR/$name/.ok_$part" ] || { echo "[stream] download of $name/$env/$part failed" >&2; exit 1; }
  done
}

if [ -z "${ENVS:-}" ]; then  # largest first: the biggest compressed parts land while the records are still small
  ENVS=$(for env in "${!RGB_N[@]}"; do echo "${RGB_N[$env]} $env"; done | sort -nr | awk '{print $2}' | tr '\n' ' ')
fi

echo "[stream] split=$SPLIT img=${IMG_SIZE} inflight=${MAX_INFLIGHT} (PNGs on $STAGE_DIR/_stream_work, a few GB). environments: $ENVS"
df -h "$STAGE_DIR" | tail -1
mkdir -p "$OUT_ROOT"
out_real=$(readlink -f "$OUT_ROOT")

for env in $ENVS; do
  [ -n "${RGB_N[$env]:-}" ] || { echo "[stream] unknown environment: $env" >&2; exit 1; }
  [ -n "${POSE_N[$env]:-}" ] || { echo "[stream] no pose part count for $env" >&2; exit 1; }
  done_mark="$STAGE_DIR/.done_${SPLIT}_${env}"
  # skip only when this OUT_ROOT already has the records. The marker remembers the path and the resolution.
  decision=$(python "$HERE/stage_unrealego2_stream.py" --resume-decision \
    --marker "$done_mark" --out-root "$out_real" --img-size "$IMG_SIZE" \
    --split-dir "$SPLIT_TXT_DIR" --env "$env" --splits "${RECORD_SPLITS[@]}")
  case "$decision" in
    skip)
      echo "[stream] skip $env (done at $out_real)"
      continue ;;
    size_mismatch)
      echo "[stream] $done_mark is a different resolution than IMG_SIZE=${IMG_SIZE} for $out_real" >&2
      exit 1 ;;
    reprocess) ;;
    *) echo "[stream] unexpected resume decision: $decision" >&2; exit 1 ;;
  esac
  pending=$(python "$HERE/stage_unrealego2_stream.py" --pending-gb \
    --rgb-dir "$STAGE_DIR/$RGB_NAME" --pose-dir "$STAGE_DIR/$POSE_NAME" \
    --env "$env" --rgb-count "${RGB_N[$env]}" --pose-count "${POSE_N[$env]}" --part-gb "$PART_GB")
  need=$(( pending + MIN_FREE_GB ))
  have=$(free_gb "$STAGE_DIR")
  if [ "$have" -lt "$need" ]; then
    echo "[stream] $env still needs about ${pending}G of downloads plus ${MIN_FREE_GB}G margin (${need}G); $STAGE_DIR has ${have}G. Stopping before this environment." >&2
    exit 2
  fi
  echo "[stream] =================== $env: rgb ${RGB_N[$env]} parts, pose ${POSE_N[$env]} parts, ${have}G free"
  download_parts "$POSE_NAME" "$env" "${POSE_N[$env]}"
  download_parts "$RGB_NAME" "$env" "${RGB_N[$env]}"

  if python "$HERE/stage_unrealego2_stream.py" \
      --env "$env" \
      --rgb-dir "$STAGE_DIR/$RGB_NAME" \
      --pose-dir "$STAGE_DIR/$POSE_NAME" \
      --rgb-count "${RGB_N[$env]}" \
      --pose-count "${POSE_N[$env]}" \
      --split-dir "$SPLIT_TXT_DIR" \
      --splits "${RECORD_SPLITS[@]}" \
      --out-root "$OUT_ROOT" \
      --work-root "$STAGE_DIR/_stream_work" \
      --img-size "$IMG_SIZE" \
      --workers "$WORKERS" \
      --max-inflight "$MAX_INFLIGHT" \
      --min-free-gb "$MIN_FREE_GB"
  then
    rm -rf "$STAGE_DIR/_stream_work/png" "$STAGE_DIR/_stream_work/pose"
    printf '%s\n%s\n' "$IMG_SIZE" "$out_real" > "$done_mark"
    echo "[stream] $env done; compressed parts kept under $STAGE_DIR"
    df -h "$STAGE_DIR" | tail -1
  else
    rc=$?
    if [ "$rc" -eq 2 ]; then
      echo "[stream] stopped on low disk during $env. Parts were kept; rerun the same command to resume." >&2
    else
      echo "[stream] $env failed (exit $rc). Parts were kept." >&2
    fi
    exit "$rc"
  fi
done
echo "[stream] all environments staged"
