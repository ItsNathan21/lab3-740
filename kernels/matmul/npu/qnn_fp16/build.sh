#!/usr/bin/env bash
# build.sh — cross-compile matmul_qnn_fp16 for arm64-android
#
# Run from inside the control container (has android-clang wrapper).
# No link-time QNN dependency: libQnnHtp.so is dlopen'd at runtime.
# The QNN headers in include/QNN/ are vendored from QAIRT 2.40.0.
#
# Usage (from project root in control container):
#   bash kernels/matmul/npu/qnn_fp16/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$SCRIPT_DIR/matmul_qnn_fp16"
SPIKE="$SCRIPT_DIR/matmul_qnn_shared_spike"

# The control container Dockerfile installs the NDK sysroot at
# /opt/android-sysroot and creates an android-clang wrapper at
# /usr/local/bin/android-clang that bakes --target, --sysroot,
# -resource-dir, -rtlib=compiler-rt, -unwindlib=libunwind.
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
    -I "$SCRIPT_DIR/include" \
    -I "$SCRIPT_DIR/include/QNN" \
    -I "$SCRIPT_DIR/include/QNN/HTP" \
    -o "$OUT" \
    "$SCRIPT_DIR/matmul_qnn_fp16.c" \
    -ldl -lm

echo "Built: $OUT"
file "$OUT"

# Task 12 Part B harness: same flags, same headers, different main. Provisioning
# preflights this binary too, so building only the first one leaves Part B
# unrunnable on the board -- which is exactly what happened before 2026-09-10.
"$CLANG"     -std=c11     -O2     -fno-omit-frame-pointer     -Wall -Wextra     -I "$SCRIPT_DIR/include"     -I "$SCRIPT_DIR/include/QNN"     -I "$SCRIPT_DIR/include/QNN/HTP"     -o "$SPIKE"     "$SCRIPT_DIR/matmul_qnn_shared_spike.c"     -ldl -lm

echo "Built: $SPIKE"
file "$SPIKE"
