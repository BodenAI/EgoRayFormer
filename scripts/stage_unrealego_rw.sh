#!/usr/bin/env bash
# Disk-aware staging of UnrealEgo-RW (real-world stereo, 872^2, 16 joints, per-camera Scaramuzza calibration):
# download (rgb + pose parts, one at a time) -> extract -> preprocess into the unified format -> delete raw.
# Raw data is ~180 GB (train_val rgb 14 x 10 GB parts, test rgb 4 parts, pose parts small); the processed 512^2
# uint8 records are ~0.2 TB.
#
#   bash scripts/fetch_unrealego_meta.sh                                        # split files + calibrations (once)
#   bash scripts/stage_unrealego_rw.sh                                          # train_val + test, one part at a time
#   PARALLEL=4 bash scripts/stage_unrealego_rw.sh test                          # 4 concurrent parts, only one group
#
# Environment variables: DATA_ROOT (default /data/unrealego), STAGE_DIR (scratch; point it at an existing
# download of the official tar parts to skip the download - parts with an ``.ok_<part>`` marker are not fetched
# again), OUT_ROOT, SPLIT_TXT_DIR, CALIB_DIR, WORKERS (preprocess processes), IMG_SIZE (stored resolution, default 512),
# PARALLEL (concurrent part downloads, default 1), KEEP_RAW=1 to keep the extracted pngs, KEEP_PARTS=1 to keep the tar
# parts after extraction (the default when STAGE_DIR is given, i.e. an existing download is never deleted).
# Parts are extracted into $STAGE_DIR/_extracted/, and only that directory is removed after preprocessing.
# Long connections may drop every few 10-100 MB; poser_curl_resume (scripts/download_utils.sh) resumes as long as
# the file keeps growing. Ctrl+C kills the whole process group, including background curls.
set -euo pipefail

GROUPS_TO_DO="${1:-train_val test}"   # keep train_val first: the dataset rig extrinsics are fitted from the first processed
                                       # split and preprocess_unrealego_raw.py only accepts the train split for that
DATA_ROOT="${DATA_ROOT:-/data/unrealego}"
SERVER="https://unrealego.mpi-inf.mpg.de/data"
[ -n "${STAGE_DIR:-}" ] && KEEP_PARTS="${KEEP_PARTS:-1}"
STAGE_DIR="${STAGE_DIR:-$DATA_ROOT/_staging_rw}"
EXTRACT_DIR="$STAGE_DIR/_extracted"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/processed}"
SPLIT_TXT_DIR="${SPLIT_TXT_DIR:-$DATA_ROOT/UnrealEgoData_rw_splits}"
CALIB_DIR="${CALIB_DIR:-$DATA_ROOT/calib}"
WORKERS="${WORKERS:-32}"
IMG_SIZE="${IMG_SIZE:-512}"
PARALLEL="${PARALLEL:-1}"
ORIG_SIZE=872
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$HERE/download_utils.sh"
# background jobs of a non-interactive shell ignore SIGINT, so Ctrl+C must be forwarded as SIGTERM to the whole group
trap 'echo "[stage] interrupted" >&2; trap - INT TERM; kill -TERM 0 2>/dev/null; exit 130' INT TERM

CALIB_L="$CALIB_DIR/fisheye_calib_unrealego_rw_left_scaramuzza.json"
CALIB_R="$CALIB_DIR/fisheye_calib_unrealego_rw_right_scaramuzza.json"
for f in "$CALIB_L" "$CALIB_R" "$SPLIT_TXT_DIR/train.txt" "$SPLIT_TXT_DIR/validation.txt" "$SPLIT_TXT_DIR/test.txt"; do
  [ -f "$f" ] || { echo "missing $f - run scripts/fetch_unrealego_meta.sh first" >&2; exit 1; }
done
mkdir -p "$STAGE_DIR"
cd "$STAGE_DIR"

fetch_env_counts() {  # $1 = official script name -> prints "env count" lines
  curl -sSL --http1.1 --retry 5 --retry-all-errors --retry-delay 2 "$SERVER/$1" \
    | sed -n 's/^ *\["\([^"]*\)"\]=\([0-9]*\).*/\1 \2/p'
}

download_parts() {  # $1 = archive name (UnrealEgoData_rw_<group>_<rgb|pose>), $2 = env, $3 = num parts
  local name=$1 env=$2 n=$3 i part running=0
  mkdir -p "$name" && cd "$name"
  for ((i = 0; i < n; i++)); do
    part=$(printf "%s.tar.gz%03d" "$env" "$i")
    [ -f ".ok_$part" ] && continue
    echo "GET $name/$env/$part (have $(stat -c%s "$part" 2>/dev/null || echo 0) bytes)"
    if [ "$PARALLEL" -le 1 ]; then
      poser_curl_resume "$SERVER/$name/$env/$part" "$part" || { echo "download of $name/$env/$part failed" >&2; exit 1; }
      touch ".ok_$part"
    else
      (poser_curl_resume "$SERVER/$name/$env/$part" "$part" && touch ".ok_$part") &
      running=$((running + 1))
      if [ "$running" -ge "$PARALLEL" ]; then wait -n || true; running=$((running - 1)); fi
    fi
  done
  wait
  for ((i = 0; i < n; i++)); do
    part=$(printf "%s.tar.gz%03d" "$env" "$i")
    [ -f ".ok_$part" ] || { echo "download of $name/$env/$part failed" >&2; exit 1; }
  done
  echo "extracting $name/$env ($n parts) -> $EXTRACT_DIR/$name"
  mkdir -p "$EXTRACT_DIR/$name"
  # numbered parts only (ignores *.temp leftovers of other downloaders)
  cat "$env".tar.gz[0-9][0-9][0-9] | tar xzf - -C "$EXTRACT_DIR/$name"
  if [ -z "${KEEP_PARTS:-}" ]; then rm -f "$env".tar.gz[0-9][0-9][0-9] ".ok_$env".tar.gz* ".lock_$env".tar.gz*; fi
  cd ..
}

for group in $GROUPS_TO_DO; do
  RGB_NAME="UnrealEgoData_rw_${group}_rgb"
  POSE_NAME="UnrealEgoData_rw_${group}_pose"
  if [ -f "$STAGE_DIR/.done_${group}" ]; then echo "skip $group (done)"; continue; fi
  declare -A RGB_N=() POSE_N=()
  while read -r env n; do [ -n "$env" ] && RGB_N["$env"]=$n; done < <(fetch_env_counts "download_unrealego_rw_${group}_rgb.sh")
  while read -r env n; do [ -n "$env" ] && POSE_N["$env"]=$n; done < <(fetch_env_counts "download_unrealego_rw_${group}_pose.sh")
  echo "=================== $group: environments ${!RGB_N[*]}"
  for env in "${!RGB_N[@]}"; do
    [ -f "$STAGE_DIR/.dl_${group}_${env}" ] && { echo "skip download $group/$env (done)"; continue; }
    download_parts "$RGB_NAME" "$env" "${RGB_N[$env]}"
    [ -n "${POSE_N[$env]:-}" ] && download_parts "$POSE_NAME" "$env" "${POSE_N[$env]}"
    touch "$STAGE_DIR/.dl_${group}_${env}"
  done

  RGB_ROOT="$EXTRACT_DIR/$RGB_NAME/$RGB_NAME"; [ -d "$RGB_ROOT" ] || RGB_ROOT="$EXTRACT_DIR/$RGB_NAME"
  POSE_ROOT="$EXTRACT_DIR/$POSE_NAME/$POSE_NAME"; [ -d "$POSE_ROOT" ] || POSE_ROOT="$EXTRACT_DIR/$POSE_NAME"
  if [ "$group" = "train_val" ]; then splits="train validation"; else splits="test"; fi
  for sp in $splits; do
    python "$HERE/preprocess_unrealego_raw.py" --dataset unrealego_rw --rgb_root "$RGB_ROOT" --pose_root "$POSE_ROOT" \
      --split_txt "$SPLIT_TXT_DIR/$sp.txt" --split_name "$sp" --out_root "$OUT_ROOT" --workers "$WORKERS" \
      --img_size "$IMG_SIZE" --orig_size "$ORIG_SIZE" --calib_left "$CALIB_L" --calib_right "$CALIB_R"
  done
  if [ -z "${KEEP_RAW:-}" ]; then rm -rf "${EXTRACT_DIR:?}/$RGB_NAME" "${EXTRACT_DIR:?}/$POSE_NAME"; fi
  touch "$STAGE_DIR/.done_${group}"
  df -h "$OUT_ROOT" | tail -1
done
# the camera mount differs between capture sessions by up to a few cm: one rig per session, see fit_session_rigs.py
# (fits on the train split only; a test-only session takes the nearest fitted session's rig)
python "$HERE/fit_session_rigs.py" --root "$OUT_ROOT" --dataset unrealego_rw --fit_splits train --fallback nearest
echo "UnrealEgo-RW staged -> $OUT_ROOT/unrealego_rw"
