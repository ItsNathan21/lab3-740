#!/usr/bin/env bash
# Cross-compile gpu_sol_microbench for arm64 Android.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/gpu_sol_microbench.c"
OUT="$DIR/gpu_sol_microbench"

if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH. Run from the control container." >&2
    exit 1
fi

echo "Building gpu_sol_microbench..."
echo "  Source: $SRC"
echo "  Output: $OUT"

android-clang -std=c11 -O2 -fno-omit-frame-pointer \
    -Wall -Wextra \
    "$SRC" \
    -ldl \
    -o "$OUT"

file_out="$(file "$OUT")"
echo "$file_out"
echo "$file_out" | grep -q 'ELF 64-bit LSB.*ARM aarch64' \
    || { echo "ERROR: $OUT is not an aarch64 ELF" >&2; exit 1; }

size="$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")"
echo "Built: $OUT ($size bytes)"
