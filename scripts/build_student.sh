#!/usr/bin/env bash
# build_student.sh - build your Lab 3 DSP code and push it to the board.
#
#   bash scripts/build_student.sh conv     # Task 10.1  kernels/lab3/conv/student_conv.c
#   bash scripts/build_student.sh morph    # Task 10.10 kernels/lab3/morph/student_morph.c
#   bash scripts/build_student.sh vtcm     # Task 11    kernels/lab3/vtcm/student_vtcm.c
#   bash scripts/build_student.sh memshare # Task 12    kernels/lab3/memshare/memshare.cpp
#
# Your file replaces the reference implementation of the same entry point, so
# the rest of the kernel - the FastRPC plumbing, the host binary already on the
# board - is untouched and you only ever rebuild the one .so.
#
# Run it in the lab container, or from your own machine as:
#   python scripts/lab3.py run bash scripts/build_student.sh conv
#
# The freshly built library is pushed to the board automatically (forgetting
# that push means silently measuring the OLD code). Add --no-push to skip it.
# Other options are passed straight through to build_dsp_skel.sh, so --arch,
# --sdk and --threads all work here too.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHICH="${1:-}"; [ $# -gt 0 ] && shift
PUSH=1
ARGS=()
for a in "$@"; do
    if [ "$a" = "--no-push" ]; then PUSH=0; else ARGS+=("$a"); fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# push FILE DEVICE_DIR: copy to the board, or print the command if we can't.
push() {
    local f="$1" dir="$2"
    if [ "$PUSH" = 1 ] && command -v adb >/dev/null 2>&1 \
       && adb get-state >/dev/null 2>&1 && adb push "$f" "$dir/" >/dev/null; then
        echo "Pushed to the board: $dir/$(basename "$f")"
    else
        [ "$PUSH" = 1 ] && echo "Could not reach the board. Push it yourself when it is connected:"
        [ "$PUSH" = 0 ] && echo "Not pushed (--no-push). To push it:"
        echo "  adb push $f $dir/"
    fi
}

# Task 12 is not a DSP build: memshare.cpp becomes an aarch64 .so loaded by the
# NPU harness on the board, so it uses the NDK directly and returns here.
if [ "$WHICH" = "memshare" ]; then
    SRC="$ROOT/kernels/lab3/memshare/memshare.cpp"
    OUT="$ROOT/kernels/lab3/memshare/libmemshare.so"
    [ -f "$SRC" ] || { echo "error: your file is missing: $SRC" >&2; exit 1; }

    CXX=""
    for c in "${NDK_ROOT:-}" "$ROOT/.toolchain/android-ndk-r27d"; do
        [ -n "$c" ] || continue
        for f in "$c"/toolchains/llvm/prebuilt/*/bin/clang++; do
            [ -x "$f" ] && CXX="$f" && break
        done
        [ -n "$CXX" ] && break
    done
    if [ -z "$CXX" ]; then
        command -v clang++ >/dev/null 2>&1 && CXX=clang++
    fi
    [ -n "$CXX" ] || { echo "error: no NDK clang++ found. Run: python install.py --role dev --fix" >&2; exit 1; }

    echo "building memshare.cpp with $CXX"
    "$CXX" --target=aarch64-linux-android33 -O2 -shared -fPIC \
        -static-libstdc++ -o "$OUT" "$SRC" -ldl "$@"

    echo
    echo "Built: $OUT"
    echo
    push "$OUT" /data/local/tmp/profiler-kernels/npu
    echo
    echo "Run it (baseline, then shared):"
    echo "  python profiler.py matmul npu --variant spike -p size=1024"
    echo "  python profiler.py matmul npu --variant spike -p size=1024 -p memshare=./libmemshare.so"
    echo
    echo "The shared run must show shared_impl=memshare and the same fp as"
    echo "the baseline. shared_impl=builtin_rpcmem means your allocator was never called."
    exit 0
fi

case "$WHICH" in
    conv)
        SRC="$ROOT/kernels/lab3/conv/student_conv.c"
        KDIR="$ROOT/kernels/convolution/dsp"
        DROP="dsp_conv_hvx_imp.c"
        OUT="$KDIR/prebuilt/libdsp_conv_skel.so"
        RUN='./conv_dsp --H 256 --W 256 --cin 64 --cout 64 --k 3 --iterations 5 --variant hvx'
        ;;
    morph)
        SRC="$ROOT/kernels/lab3/morph/student_morph.c"
        KDIR="$ROOT/kernels/convolution/dsp"
        DROP="dsp_conv_morph_imp.c"
        OUT="$KDIR/prebuilt/libdsp_conv_skel.so"
        RUN='PROFILER:python profiler.py morph dsp -p op=erode -p k=3 -p write_pgm=true'
        ;;
    vtcm)
        SRC="$ROOT/kernels/lab3/vtcm/student_vtcm.c"
        KDIR="$ROOT/kernels/convolution/dsp"
        DROP="dsp_conv_stage_imp.c"
        OUT="$KDIR/prebuilt/libdsp_conv_skel.so"
        RUN='PROFILER:python profiler.py stage dsp -p impl=staged -p p=1'
        ;;
    *)
        echo "usage: $0 conv|morph|vtcm|memshare [--arch v81] [--sdk PATH]" >&2
        exit 2 ;;
esac

[ -f "$SRC" ] || { echo "error: your file is missing: $SRC" >&2; exit 1; }

echo "building $(basename "$SRC")  (replacing $DROP)"
echo
# build_dsp_skel.sh ends with its own "Push it with: adb push ..." hint; drop it,
# since this script pushes for you. pipefail keeps a failed build fatal.
bash "$ROOT/scripts/build_dsp_skel.sh" "$KDIR" \
    --exclude "$DROP" --add "$SRC" "$@" \
    | awk '/^Push it with:/ {skip = 2} skip > 0 {skip--; next} {print}'

echo
push "$OUT" /data/local/tmp/profiler-kernels/dsp
echo
echo "Run it:"
if [ "${RUN#PROFILER:}" != "$RUN" ]; then
    echo "  ${RUN#PROFILER:}"
else
    echo "  adb shell \"cd /data/local/tmp/profiler-kernels/dsp && LD_LIBRARY_PATH=. \\"
    echo "      ADSP_LIBRARY_PATH='.;\\\$PWD' $RUN\""
fi
echo
echo "Check the fingerprint against your baseline. If it moved, the speedup is not real."
