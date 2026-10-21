#!/system/bin/sh
# Continuous GPU load for power_run.py. Loops until killed.
# Same stream microbench and parameters the v1 GPU calibration windows used, so
# a modeled run is driven by the load the coefficients were fitted against.
# Deliberately NOT matmul with a large --inner-reps: that repeats the math
# inside a single dispatch, which turns into one multi-minute kernel launch.
G=/data/local/tmp/profiler-kernels/gpu/gpu_sol_microbench
while :; do
  "$G" --mode stream --reps 4000 --global 262144 --local 256 \
       --bytes-mb 64 --passes 4 >/dev/null 2>&1
done
