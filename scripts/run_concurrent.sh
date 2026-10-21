#!/usr/bin/env bash
# run_concurrent.sh - Lab 3 Task 10.6. Run the CPU and the DSP at the same time.
#
#   bash scripts/run_concurrent.sh
#   bash scripts/run_concurrent.sh --cpu-iters 6 --dsp-iters 2 --cin 64
#
# Runs three measurements and prints one table:
#   1. CPU alone   all eight cores, conv_neon
#   2. DSP alone   all HVX contexts, conv_dsp
#   3. both together, started at the same moment
#
# The point is not that concurrency is faster - it is what concurrency costs
# the engine that was already running. Pick iteration counts (below) so the two
# solo runs take roughly the same wall time, or the overlap window is mostly
# one engine running alone and the comparison means nothing.
#
# The defaults are balanced for the default shape on an HDK8850: measured
# ~65 ms per CPU iteration and ~167 ms per DSP iteration, so 8 and 3 give
# roughly 530 ms and 510 ms. Change the shape and you must rebalance; run the
# two solo measurements first and pick the pair that matches.
#
# Two reasons not to lower these. conv_neon spawns and joins its threads INSIDE
# the timed region, a fixed ~8 ms that a short run charges entirely to itself.
# And the DSP's first iteration is slow (165 us/iter against 152 us/iter once
# warm), so a single-iteration DSP run reports warmup as steady state.
#
# Everything runs on the board; this script only orchestrates and parses.

set -euo pipefail

DEV=/data/local/tmp/profiler-kernels
H=256 W=256 CIN=64 COUT=64 K=3
CPU_ITERS=8 DSP_ITERS=3 CPU_THREADS=8

while [ $# -gt 0 ]; do
    case "$1" in
        --H) H="$2"; shift 2 ;;
        --W) W="$2"; shift 2 ;;
        --cin) CIN="$2"; shift 2 ;;
        --cout) COUT="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --cpu-iters) CPU_ITERS="$2"; shift 2 ;;
        --dsp-iters) DSP_ITERS="$2"; shift 2 ;;
        --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

adb devices | grep -q "device$" || { echo "no board attached" >&2; exit 1; }

SHAPE="--H $H --W $W --cin $CIN --cout $COUT --k $K"
CPU_CMD="taskset ff $DEV/convolution/conv_neon $SHAPE --iterations $CPU_ITERS --threads $CPU_THREADS"
DSP_CMD="cd $DEV/dsp && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH='.;\$PWD' ./conv_dsp $SHAPE --iterations $DSP_ITERS --variant hvx"

# 2*H*W*Cin*Cout*K*K flop per iteration.
FLOP=$(python -c "print(2*$H*$W*$CIN*$COUT*$K*$K)")

# `|| true` so a missing field yields "" rather than aborting the script under
# `set -e`; require() then says which run failed and shows what it printed,
# which is far more use than a bare exit code.
field() { echo "$1" | grep -o "$2=[0-9.e+-]*" | head -1 | cut -d= -f2 || true; }
require() {
    if [ -z "$2" ]; then
        echo "error: $1 produced no '$3'. It printed:" >&2
        echo "  ${4:-(nothing)}" >&2
        echo "Check the binaries are executable:" >&2
        echo "  adb shell chmod 755 $DEV/dsp/conv_dsp $DEV/convolution/conv_neon" >&2
        exit 1
    fi
}
gflops() { python -c "
us=float('${2:-0}') or 1.0
print(f'{$1*int(${3:-1})/us/1000.0:.2f}')"; }

echo "shape ${H}x${W} cin=$CIN cout=$COUT k=$K   cpu_iters=$CPU_ITERS threads=$CPU_THREADS   dsp_iters=$DSP_ITERS"
echo

echo "[1/3] CPU alone"
cpu_solo=$(adb shell "$CPU_CMD" 2>&1 | tail -1)
cpu_solo_us=$(field "$cpu_solo" wall_us)
require "CPU solo" "$cpu_solo_us" wall_us "$cpu_solo"

echo "[2/3] DSP alone"
dsp_solo=$(adb shell "$DSP_CMD" 2>&1 | tail -1)
dsp_solo_us=$(field "$dsp_solo" wall_us)
require "DSP solo" "$dsp_solo_us" wall_us "$dsp_solo"

echo "[3/3] both at once"
# Launch both inside ONE adb shell so they start together; two adb invocations
# would stagger by the round-trip and the overlap would be partly fictional.
both=$(adb shell "
    ( $CPU_CMD > /data/local/tmp/_cc_cpu.txt 2>&1 ) &
    cpid=\$!
    ( $DSP_CMD > /data/local/tmp/_cc_dsp.txt 2>&1 ) &
    dpid=\$!
    wait \$cpid; wait \$dpid
    echo '---CPU---'; tail -1 /data/local/tmp/_cc_cpu.txt
    echo '---DSP---'; tail -1 /data/local/tmp/_cc_dsp.txt
    rm -f /data/local/tmp/_cc_cpu.txt /data/local/tmp/_cc_dsp.txt
" 2>&1)

cpu_con=$(echo "$both" | sed -n '/---CPU---/,$p' | sed -n 2p)
dsp_con=$(echo "$both" | sed -n '/---DSP---/,$p' | sed -n 2p)
cpu_con_us=$(field "$cpu_con" wall_us)
dsp_con_us=$(field "$dsp_con" wall_us)
require "CPU concurrent" "$cpu_con_us" wall_us "$cpu_con"
require "DSP concurrent" "$dsp_con_us" wall_us "$dsp_con"

echo
printf "%-22s %12s %12s %10s\n" "" "solo GFLOP/s" "conc GFLOP/s" "slowdown"
printf "%-22s %12s %12s %10s\n" \
    "CPU (${CPU_THREADS} cores)" \
    "$(gflops "$FLOP" "$cpu_solo_us" "$CPU_ITERS")" \
    "$(gflops "$FLOP" "$cpu_con_us" "$CPU_ITERS")" \
    "$(python -c "print(f'{float(${cpu_con_us:-1})/float(${cpu_solo_us:-1}):.2f}x')")"
printf "%-22s %12s %12s %10s\n" \
    "DSP (HVX)" \
    "$(gflops "$FLOP" "$dsp_solo_us" "$DSP_ITERS")" \
    "$(gflops "$FLOP" "$dsp_con_us" "$DSP_ITERS")" \
    "$(python -c "print(f'{float(${dsp_con_us:-1})/float(${dsp_solo_us:-1}):.2f}x')")"
echo
echo "aggregate concurrent: $(python -c "
c=$FLOP*$CPU_ITERS/float(${cpu_con_us:-1})/1000.0
d=$FLOP*$DSP_ITERS/float(${dsp_con_us:-1})/1000.0
print(f'{c+d:.2f} GFLOP/s   (CPU alone was {$FLOP*$CPU_ITERS/float(${cpu_solo_us:-1})/1000.0:.2f})')")"
echo
echo "Fingerprints (must be unchanged by concurrency):"
echo "  CPU solo $(field "$cpu_solo" fingerprint)   conc $(field "$cpu_con" fingerprint)"
echo "  DSP solo $(field "$dsp_solo" fingerprint)   conc $(field "$dsp_con" fingerprint)"
