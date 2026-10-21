#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/matmul_opencl_int8_dot_8x8.c"; OUT="$DIR/matmul_opencl_int8_dot_8x8"
command -v android-clang >/dev/null 2>&1 || { echo "ERROR: android-clang" >&2; exit 1; }
android-clang -std=c11 -O2 -fno-omit-frame-pointer -Wall -Wextra "$SRC" -ldl -o "$OUT"
file "$OUT" | grep -q "ELF 64-bit LSB.*ARM aarch64" || { echo "ERROR: not aarch64" >&2; exit 1; }
echo "Built: $OUT ($(stat -c%s "$OUT") bytes)"
