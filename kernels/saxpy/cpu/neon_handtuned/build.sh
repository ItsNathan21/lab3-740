#!/usr/bin/env bash
# Cross-compile the neon_handtuned SAXPY variant for arm64-android.
#
# Pure C with arm_neon.h intrinsics (NEON enabled automatically for aarch64).
# Run from inside the control container OR from the host:
#
#   # Inside the container:
#   bash kernels/saxpy/cpu/neon_handtuned/build.sh
#
#   # From the host (amd64 control image; android-clang is a cross-compiler):
#   docker run --rm -v "$(pwd):/workspace" snapdragon-profiler-control:dev \
#     bash /workspace/kernels/saxpy/cpu/neon_handtuned/build.sh

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/saxpy_neon.c"
OUT="$DIR/saxpy_neon"

if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH. Run from inside the control container." >&2
    exit 1
fi

android-clang -std=c11 -O2 -fno-omit-frame-pointer \
    "$SRC" -lm \
    -o "$OUT"

file_out=$(file "$OUT")
echo "$file_out"
echo "$file_out" | grep -q 'ELF 64-bit LSB.*ARM aarch64' \
    || { echo "ERROR: $OUT is not an aarch64 ELF" >&2; exit 1; }

echo "Built: $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"
