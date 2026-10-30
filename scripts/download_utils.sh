#!/usr/bin/env bash
# Resumable HTTPS downloads for the dataset staging scripts (fetch_unrealego_meta.sh, stage_unrealego*.sh), which
# source this file. Linux only (GNU stat, util-linux flock). Proxies are taken from the usual environment variables
# (http_proxy / https_proxy); activate the Python environment before running a staging script.
#
# poser_curl_resume <url> <dest>: proxies and data servers drop long connections (curl 56 / 18 / 28 ..., every few
# tens to hundreds of MB). After every drop the download resumes from the saved bytes; it keeps resuming as long as the
# file grows and gives up after POSER_CURL_STALL_RETRIES (default 10) attempts without progress. HTTP 4xx / 5xx and disk
# write errors fail immediately; Ctrl+C / SIGTERM return at once without retrying.

poser_remote_size() {  # $1=url -> Content-Length (following redirects); empty on failure
  curl -sSIL --http1.1 --retry 5 --retry-all-errors --retry-delay 2 --connect-timeout 30 "$1" 2>/dev/null \
    | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}'
}

poser_curl_resume() {  # $1=url $2=dest; one download process per destination file (concurrent appends corrupt it)
  local dest=$2 rc lock
  lock="$(dirname "$dest")/.lock_$(basename "$dest")"
  exec 9>"$lock"
  if ! flock -n 9; then
    echo "[poser_curl] $dest is being written by another process (lock $lock); refusing to download concurrently" >&2
    exec 9>&-
    return 1
  fi
  _poser_curl_resume_impl "$1" "$dest"
  rc=$?
  exec 9>&-
  return "$rc"
}

_poser_curl_resume_impl() {  # $1=url $2=dest
  local url=$1 dest=$2 total have=0 before=0 stall=0 drops=0 rc err
  local max_stall="${POSER_CURL_STALL_RETRIES:-10}" delay="${POSER_CURL_RETRY_DELAY:-2}"
  total=$(poser_remote_size "$url")
  [ -n "$total" ] || { echo "[poser_curl] cannot get size of $url" >&2; return 1; }
  while :; do
    have=0; [ -f "$dest" ] && have=$(stat -c%s "$dest")
    if [ "$have" -eq "$total" ]; then
      echo "[poser_curl] done $(basename "$dest"): $((total / 1048576)) MB, $drops reconnects"
      return 0
    elif [ "$have" -gt "$total" ]; then
      echo "[poser_curl] $dest is larger than remote ($have > $total bytes): delete it and rerun" >&2
      return 1
    fi
    err=$(curl -fsSL --http1.1 --retry 0 --connect-timeout 30 --speed-limit 10240 --speed-time 60 \
            -C - -o "$dest" "$url" 2>&1)
    rc=$?
    before=$have; have=0; [ -f "$dest" ] && have=$(stat -c%s "$dest")
    case $rc in
      0) # curl "succeeded" but the file is still short (empty 206 body, 416 on a short resume): count it as a stall
         # instead of looping back immediately - the old ``continue`` here could spin without sleeping
         if [ "$have" -gt "$before" ]; then continue; fi ;;
      130|143) echo "[poser_curl] interrupted" >&2; return "$rc" ;;
      3|22|23|33) echo "[poser_curl] FAILED $dest: $err" >&2; return 1 ;;
    esac
    drops=$((drops + 1))
    if [ "$have" -gt "$before" ]; then stall=0; else stall=$((stall + 1)); fi
    if [ "$stall" -ge "$max_stall" ]; then
      echo "[poser_curl] FAILED $dest: no progress in $stall attempts ($have/$total bytes): $err" >&2
      return 1
    fi
    printf '[poser_curl] %s: drop #%d (curl %d) at %d/%d MB (%d%%), resuming\n' \
      "$(basename "$dest")" "$drops" "$rc" $((have / 1048576)) $((total / 1048576)) $((have * 100 / total)) >&2
    sleep "$delay" || return 130
  done
}
