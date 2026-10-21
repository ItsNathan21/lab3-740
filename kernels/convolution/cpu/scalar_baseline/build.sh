#!/usr/bin/env bash
# Cross-compile the scalar_baseline convolution variant for arm64-android.
#   docker run --rm -v "$(pwd):/workspace" snapdragon-profiler-control:dev #     bash /workspace/kernels/convolution/cpu/scalar_baseline/build.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/conv_scalar.c"; OUT="$DIR/conv_scalar"
if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH. Run inside the control container." >&2; exit 1
fi
android-clang -std=c11 -O2 -fno-omit-frame-pointer "$SRC" -lm -o "$OUT"
file_out=$(file "$OUT"); echo "$file_out"
echo "$file_out" | grep -q 'ELF 64-bit LSB.*ARM aarch64' || { echo "ERROR: not aarch64 ELF" >&2; exit 1; }
echo "Built: $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"
