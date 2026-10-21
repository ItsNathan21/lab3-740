#!/system/bin/sh
# HDK8750 teaching-power calibration WORKLOAD WRAPPER (on-device).
#
# Owns the workload-intent signals the read-only sampler trusts:
#   <outdir>/WL_PIDS    the worker PIDs, one per line (written before the flag)
#   <outdir>/WL_ACTIVE  present while the workload is running
# Both are created up front and REMOVED in a trap on any exit (normal, killed,
# or crash), and the workers are killed too. So if the workload dies early the
# files vanish and the sampler's EXPECT_ACTIVE check fails the window rather
# than mistaking it for idle.
#
# Usage (run for a whole active window; each arg is ONE fully-formed pinned
# command, e.g. a hex-masked taskset invocation):
#   sh power_workload_wrapper.sh <outdir> <window_id> \
#       "taskset 1 <cmd> --iters N" "taskset 2 <cmd> --iters N" ...
# For Gold-6 pass 6 commands (taskset 1/2/4/8/10/20); Prime-2 pass 2 (taskset 40/80);
# GPU pass one persistent command. Size --iters so it outlasts the sampler duration.
# The wrapper records every command plus combined worker stdout/stderr and a
# terminal =WORKLOAD_EXIT status in <outdir>/<window_id>.workload.log.

set -u
OUT="${1:?outdir required}"; shift
WINDOW_ID="${1:?window_id required}"; shift
case "$WINDOW_ID" in
  ''|*[!A-Za-z0-9_.-]*) echo "window_id contains unsafe characters" >&2; exit 2 ;;
esac
[ "$#" -ge 1 ] || { echo "need at least one worker command"; exit 2; }
FLAG="$OUT/WL_ACTIVE"
PIDS="$OUT/WL_PIDS"
LOG="$OUT/$WINDOW_ID.workload.log"
WORKERS=""
FINAL_STATUS_WRITTEN=0

mkdir -p "$OUT" || { echo "cannot create output directory: $OUT" >&2; exit 3; }
[ ! -e "$FLAG" ] && [ ! -e "$PIDS" ] && [ ! -e "$LOG" ] || {
  echo "refusing to start: stale/active intent files or workload log exists for $WINDOW_ID in $OUT" >&2
  exit 3
}
cleanup() {
  rc=$1
  trap - EXIT INT TERM HUP
  rm -f "$FLAG" "$PIDS"
  [ -n "$WORKERS" ] && kill $WORKERS 2>/dev/null
  if [ "$FINAL_STATUS_WRITTEN" = "0" ]; then
    printf '=WORKLOAD_EXIT window_id=%s status=%s\n' "$WINDOW_ID" "$rc" >> "$LOG" 2>/dev/null
  fi
  exit "$rc"
}
trap 'cleanup $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

{
  printf '=WORKLOAD window_id=%s worker_count=%s\n' "$WINDOW_ID" "$#"
  command_index=0
  for cmd in "$@"; do
    printf '=COMMAND_BEGIN index=%s\n%s\n=COMMAND_END index=%s\n' \
      "$command_index" "$cmd" "$command_index"
    command_index=$((command_index + 1))
  done
  echo '=OUTPUT_BEGIN'
} > "$LOG" || exit 3

: > "$PIDS" || exit 3
worker_index=0
for cmd in "$@"; do
  printf '=WORKER_OUTPUT_BEGIN index=%s\n' "$worker_index" >> "$LOG" || exit 3
  sh -c "$cmd" >> "$LOG" 2>&1 &
  p=$!
  WORKERS="$WORKERS $p"
  echo "$p" >> "$PIDS" || exit 3
  worker_index=$((worker_index + 1))
done
touch "$FLAG" || exit 3

# Stay alive until all workers finish (or we are killed -> trap cleans up).
# Wait individually so one failed worker cannot be hidden by a later success.
wrapper_rc=0
for p in $WORKERS; do
  wait "$p"
  worker_rc=$?
  printf '=WORKER_EXIT pid=%s status=%s\n' "$p" "$worker_rc" >> "$LOG" || wrapper_rc=1
  [ "$worker_rc" -eq 0 ] || wrapper_rc=1
done
WORKERS=""
printf '=WORKLOAD_EXIT window_id=%s status=%s\n' "$WINDOW_ID" "$wrapper_rc" >> "$LOG" || exit 3
FINAL_STATUS_WRITTEN=1
exit "$wrapper_rc"
