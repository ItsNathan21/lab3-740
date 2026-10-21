#!/system/bin/sh
# Pinned NEON matmul load for power_run.py. Loops until killed.
# Arg 1: taskset hex mask (one core). Exists as a file so power_run.py can pass
# a worker command with no nested quoting through `adb shell`.
MASK="${1:?taskset mask required}"
K=/data/local/tmp/profiler-kernels/matmul_neon_handtuned
while :; do
  taskset "$MASK" "$K" --size 512 --iterations 200 >/dev/null 2>&1
done
