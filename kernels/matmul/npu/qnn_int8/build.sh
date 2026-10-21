#!/usr/bin/env bash
# build.sh — cross-compile matmul_qnn_int8 for arm64-android
#
# Run from inside the control container (has android-clang wrapper).
# Shares QNN headers with the qnn_fp16 sibling directory.
#
# Usage (from project root in control container):
#   bash kernels/matmul/npu/qnn_int8/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$SCRIPT_DIR/matmul_qnn_int8"
HDR="$SCRIPT_DIR/../qnn_fp16/include"

CLANG=android-clang

if ! command -v "$CLANG" >/dev/null 2>&1; then
    echo "ERROR: android-clang not found. Are you inside the control container?"
    exit 1
fi

echo "Using clang: $CLANG"

"$CLANG" \
    -std=c11 \
    -O2 \
    -fno-omit-frame-pointer \
    -Wall -Wextra \
    -I "$HDR" \
    -I "$HDR/QNN" \
    -I "$HDR/QNN/HTP" \
    -o "$OUT" \
    "$SCRIPT_DIR/matmul_qnn_int8.c" \
    -ldl -lm

echo "Built: $OUT"
file "$OUT"
