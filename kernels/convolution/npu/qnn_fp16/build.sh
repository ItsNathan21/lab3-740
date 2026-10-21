#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; OUT="$SCRIPT_DIR/conv_qnn_fp16"
command -v android-clang >/dev/null 2>&1 || { echo "ERROR: android-clang not found"; exit 1; }
android-clang -std=c11 -O2 -fno-omit-frame-pointer -Wall -Wextra \
    -I "$SCRIPT_DIR/include" -I "$SCRIPT_DIR/include/QNN" -I "$SCRIPT_DIR/include/QNN/HTP" \
    -o "$OUT" "$SCRIPT_DIR/conv_qnn_fp16.c" -ldl -lm
echo "Built: $OUT"; file "$OUT"
