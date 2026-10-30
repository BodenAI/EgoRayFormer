#!/usr/bin/env bash
# Fetch the small public metadata of the UnrealEgo benchmark (no registration needed as of 2026-09):
#   * official train / validation / test split files of UnrealEgo2 and UnrealEgo-RW
#   * fisheye calibrations (Scaramuzza + OpenCV) of the UnrealEgo(2) and UnrealEgo-RW cameras
#   * the official download scripts (kept for reference; stage_unrealego{2,_rw}.sh parse them)
#
#   bash scripts/fetch_unrealego_meta.sh                        # -> $DATA_ROOT (default /data/unrealego)
#
# Direct links (all under https://unrealego.mpi-inf.mpg.de/data/):
#   data_split/UnrealEgoData2.zip            train.txt (12139) / validation.txt (1545) / test.txt (1523) motions
#   data_split/UnrealEgoData_rw.zip          train.txt (402) / validation.txt (96) / test.txt (93) sequences
#   fisheye_calib_unrealego.zip              UnrealEgo / UnrealEgo2 camera (left = right), 1024^2
#   fisheye_calib_unrealego_rw.zip           UnrealEgo-RW left / right cameras, 872^2
#   download_unrealego2_{train_val,test}_{rgb,pose,depth}.sh
#   download_unrealego_rw_{train_val,test}_{rgb,pose}.sh
#   UnrealEgoData2_test_pose_example.zip     leaderboard submission example
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/unrealego}"
SERVER="https://unrealego.mpi-inf.mpg.de/data"
CURL="curl -sS -L --http1.1 --retry 5 --retry-all-errors --retry-delay 2"

mkdir -p "$DATA_ROOT/UnrealEgoData2_splits" "$DATA_ROOT/UnrealEgoData_rw_splits" "$DATA_ROOT/calib" "$DATA_ROOT/official_scripts"
cd "$DATA_ROOT"

$CURL -o /tmp/UnrealEgoData2.zip "$SERVER/data_split/UnrealEgoData2.zip"
$CURL -o /tmp/UnrealEgoData_rw.zip "$SERVER/data_split/UnrealEgoData_rw.zip"
unzip -o -q -j /tmp/UnrealEgoData2.zip -d "$DATA_ROOT/UnrealEgoData2_splits"
unzip -o -q -j /tmp/UnrealEgoData_rw.zip -d "$DATA_ROOT/UnrealEgoData_rw_splits"

$CURL -o /tmp/fisheye_calib_unrealego.zip "$SERVER/fisheye_calib_unrealego.zip"
$CURL -o /tmp/fisheye_calib_unrealego_rw.zip "$SERVER/fisheye_calib_unrealego_rw.zip"
unzip -o -q -j /tmp/fisheye_calib_unrealego.zip -d "$DATA_ROOT/calib"
unzip -o -q -j /tmp/fisheye_calib_unrealego_rw.zip -d "$DATA_ROOT/calib"

for s in download_unrealego2_train_val_rgb.sh download_unrealego2_train_val_pose.sh download_unrealego2_test_rgb.sh \
         download_unrealego2_test_pose.sh download_unrealego_rw_train_val_rgb.sh download_unrealego_rw_train_val_pose.sh \
         download_unrealego_rw_test_rgb.sh download_unrealego_rw_test_pose.sh; do
  $CURL -o "$DATA_ROOT/official_scripts/$s" "$SERVER/$s"
done
$CURL -o "$DATA_ROOT/official_scripts/UnrealEgoData2_test_pose_example.zip" "$SERVER/UnrealEgoData2_test_pose_example.zip"

echo "split files:"; wc -l "$DATA_ROOT"/UnrealEgoData2_splits/*.txt "$DATA_ROOT"/UnrealEgoData_rw_splits/*.txt
echo "calibrations:"; ls "$DATA_ROOT/calib"
