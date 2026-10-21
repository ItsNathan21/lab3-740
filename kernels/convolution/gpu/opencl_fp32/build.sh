#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/conv_opencl_fp32.c"; OUT="$DIR/conv_opencl_fp32"
command -v android-clang >/dev/null 2>&1 || { echo "ERROR: android-clang not on PATH" >&2; exit 1; }
android-clang -std=c11 -O2 -fno-omit-frame-pointer -Wall -Wextra "$SRC" -ldl -lm -o "$OUT"
file_out=$(file "$OUT"); echo "$file_out"
echo "$file_out" | grep -q 'ELF 64-bit LSB.*ARM aarch64' || { echo "ERROR: not aarch64 ELF" >&2; exit 1; }
echo "Built: $OUT ($(stat -c%s "$OUT") bytes)"
