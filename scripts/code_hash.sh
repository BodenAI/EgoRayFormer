#!/usr/bin/env bash
# Code version of a run: sha1 over the sources of the egorayformer package (sorted relative paths + contents), identical
# for identical code in any copy of the repository. Code changes alter the random-number stream, so runs with the same
# seed are comparable only within one hash.
#
#   bash scripts/code_hash.sh                    # print the 12-character hash
#   bash scripts/code_hash.sh work_dirs/<run>    # also write work_dirs/<run>/code_hash.txt
set -euo pipefail
cd "$(dirname "$0")/.."
h=$(find egorayformer -name '*.py' -not -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-12)
if [ $# -ge 1 ]; then
  mkdir -p "$1"
  printf 'hash %s\ncode_dir %s\ndate %s\n' "$h" "$(pwd)" "$(date '+%F %T')" > "$1/code_hash.txt"
fi
echo "$h"
