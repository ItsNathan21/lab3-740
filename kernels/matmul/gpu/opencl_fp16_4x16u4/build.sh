#!/usr/bin/env bash
# Cross-compile matmul_opencl_fp16_4x16u4 for arm64-android.
#
# True-FP16 register-blocked 4x16 matmul (half16 loads/accumulators). See the
# FP32 sibling kernels/matmul/gpu/opencl_fp32_4x16u4/build.sh for the canonical
# pattern. Uses _Float16 host buffers, so the binary uploads half-precision A/B.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/matmul_opencl_fp16_4x16u4.c"
OUT="$DIR/matmul_opencl_fp16_4x16u4"

if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH." >&2
    echo "       Run this script from inside the control container." >&2
    exit 1
fi

echo "Building matmul_opencl_fp16_4x16u4 (true-FP16 4x16 GPU candidate)..."
echo "  Source: $SRC"
echo "  Output: $OUT"
echo ""

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
echo ""
echo "Built: $OUT ($SIZE bytes)"
