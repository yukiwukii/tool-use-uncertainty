#!/usr/bin/env bash
# Submit work to the Merlin GPU job (see toolmech/merlin/job_entry.sh).
#
#   toolmech/gpu.sh run  NAME -- CMD...   sync code, queue task, wait, print log tail; exits with task rc
#   toolmech/gpu.sh submit NAME -- CMD... same but return immediately (check with `status`/`wait`)
#   toolmech/gpu.sh wait NAME             block until NAME finishes; exits with its rc
#   toolmech/gpu.sh status                queue / running / recent finished, job log tail
#   toolmech/gpu.sh log NAME              full task log
#
# Each task gets one GPU and runs from a fresh copy of the code synced at
# submit time, with `results/` shared. NAME must be unique ([A-Za-z0-9_.-]).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H=/mnt/hdfs/nadya.wangsajaya/toolmech

sync_code() {
  ( flock 9
    rsync -a --delete \
      --exclude .venv --exclude .hf_cache --exclude results --exclude .git \
      --exclude __pycache__ --exclude 'when2tool/outputs' --exclude 'when2tool/probe_data' \
      "$ROOT/" "$H/code/"
  ) 9>"$ROOT/.gpu_sync.lock"
}

queue_task() {
  local name=$1; shift
  [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad NAME: $name" >&2; exit 2; }
  for d in queue running done failed; do
    [ -e "$H/$d/$name.sh" ] && { echo "NAME $name already used ($d)" >&2; exit 2; }
  done
  [ "${1:-}" = "--" ] && shift
  local cmd; printf -v cmd '%q ' "$@"
  sync_code
  printf 'set -eo pipefail\necho "[task] $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date)"\n%s\n' "$cmd" \
    > "$H/queue/.$name.tmp"
  mv "$H/queue/.$name.tmp" "$H/queue/$name.sh"
  echo "queued $name"
}

wait_task() {
  local name=$1
  until [ -e "$H/done/$name.rc" ] || [ -e "$H/failed/$name.rc" ]; do sleep 20; done
  local rc; rc=$(cat "$H"/{done,failed}/"$name.rc" 2>/dev/null | head -1)
  echo "---- $name rc=$rc (log: $H/logs/$name.log)"
  tail -40 "$H/logs/$name.log"
  return "$rc"
}

case "${1:-status}" in
  run)    shift; name=$1; shift; queue_task "$name" "$@"; wait_task "$name" ;;
  submit) shift; name=$1; shift; queue_task "$name" "$@" ;;
  wait)   wait_task "$2" ;;
  log)    cat "$H/logs/$2.log" ;;
  status)
    echo "queue:";   ls "$H/queue"   | grep '\.sh$' || true
    echo "running:"; for c in "$H"/running/*.claim; do [ -e "$c" ] && echo "  $(basename "$c" .claim)  $(cat "$c")"; done
    echo "done (recent):";   ls -t "$H/done"   | grep '\.rc$' | head -10 || true
    echo "failed (recent):"; ls -t "$H/failed" | grep '\.rc$' | head -10 || true
    echo "job log:"; tail -5 "$H/logs/_job.log" 2>/dev/null || echo "  (no job yet)"
    ;;
  *) sed -n 2,12p "$0"; exit 2 ;;
esac
