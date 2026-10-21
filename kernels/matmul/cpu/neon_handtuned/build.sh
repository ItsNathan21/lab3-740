#!/usr/bin/env bash
# Cross-compile the neon_handtuned matmul variant for arm64-android.
#
# Pure C (uses arm_neon.h intrinsics). No C++ runtime needed; no
# -lc++_static / -lc++abi flags required. NEON intrinsics are
# enabled automatically by the aarch64 target.
#
# Run from inside the control container OR from the host:
#
#   # Inside the container:
#   bash kernels/matmul/cpu/neon_handtuned/build.sh
#
#   # From the host:
#   docker run --rm --platform linux/arm64 \
#     -v "$(pwd):/workspace" snapdragon-profiler-control:dev \
#     bash /workspace/kernels/matmul/cpu/neon_handtuned/build.sh

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/matmul_neon.c"
OUT="$DIR/matmul_neon"

if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH. Run from inside the control container." >&2
    exit 1
fi

# -O2: realistic optimization, matches diagnostic + eigen variants.
# -fno-omit-frame-pointer: callstack-friendly for future
#   simpleperf record use.
# -lm: needed for sqrt() in the fingerprint computation.
android-clang -std=c11 -O2 -fno-omit-frame-pointer \
    "$SRC" -lm \
    -o "$OUT"

# Verify it's an Android arm64 binary.
file_out=$(file "$OUT")
echo "$file_out"
echo "$file_out" | grep -q 'ELF 64-bit LSB.*ARM aarch64' \
    || { echo "ERROR: $OUT is not an aarch64 ELF" >&2; exit 1; }

echo "Built: $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"
