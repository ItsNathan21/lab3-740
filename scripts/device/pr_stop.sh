#!/system/bin/sh
# Stop every power_run.py load. Exists as a FILE for a specific reason.
#
# `adb shell pkill -f pr_gpu_loop` does not reliably work here: the shell that
# runs the command has the pattern in its own command line, so pkill matches and
# kills that shell - the target can survive and the rest of a `a; b; c` chain
# never executes. A loop script that survived this way kept relaunching its GPU
# worker through two later measurement windows before it was noticed.
#
# Keeping the patterns inside the file means the invoking command line is only
# "sh /data/local/tmp/pr_stop.sh", which matches nothing below. $$ is excluded
# anyway, belt and braces, and we kill parents before children so a loop cannot
# relaunch its worker after we have killed it.
SELF=$$
for pat in pr_cpu_loop pr_gpu_loop power_workload_wrapper \
           gpu_sol_microbench matmul_neon matmul_opencl matmul_qnn matmul_dsp; do
  for p in $(pgrep -f "$pat" 2>/dev/null); do
    [ "$p" = "$SELF" ] && continue
    kill -9 "$p" 2>/dev/null
  done
done
sleep 1
left=$(pgrep -f 'pr_cpu_loop|pr_gpu_loop|gpu_sol_microbench|matmul_' 2>/dev/null | wc -l)
echo "pr_stop: $left remaining"
