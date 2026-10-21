#!/usr/bin/env bash
# Cross-compile reduction_opencl_fp32 for arm64-android (roofline anchor #2, GPU).
#
# Produces: kernels/reduction/gpu/opencl_fp32/reduction_opencl_fp32
#   A standalone arm64 ELF that dlopen's /vendor/lib64/libOpenCL.so, compiles an
#   embedded multi-pass tree-reduction kernel via the Adreno on-device compiler,
#   runs it with CL_QUEUE_PROFILING_ENABLE, and prints a GPU_DISPATCH_RESULT
#   block (duration_ns = summed passes; readback_ns ~= 0 since the result is one
#   float). OpenCL is loaded at runtime — no compile-time -lOpenCL.
#
# Run from inside the control container OR from the host:
#
#   # Inside the container:
#   bash kernels/reduction/gpu/opencl_fp32/build.sh
#
#   # From the host (amd64 control image; android-clang is a cross-compiler):
#   docker run --rm -v "$(pwd):/workspace" snapdragon-profiler-control:dev \
#     bash /workspace/kernels/reduction/gpu/opencl_fp32/build.sh

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/reduction_opencl_fp32.c"
OUT="$DIR/reduction_opencl_fp32"

if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH. Run from inside the control container." >&2
    exit 1
fi

echo "Building reduction_opencl_fp32 (reduction GPU, roofline memory-bound anchor #2)..."
android-clang -std=c11 -O2 -fno-omit-frame-pointer \
    -Wall -Wextra \
    "$SRC" \
    -ldl -lm \
    -o "$OUT"

file_out=$(file "$OUT")
echo "$file_out"
echo "$file_out" | grep -q 'ELF 64-bit LSB.*ARM aarch64' \
    || { echo "ERROR: $OUT is not an aarch64 ELF — wrong compiler?" >&2; exit 1; }

SIZE=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")
echo "Built: $OUT ($SIZE bytes)"
