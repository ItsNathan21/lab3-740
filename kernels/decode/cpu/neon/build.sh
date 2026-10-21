#!/usr/bin/env bash
# Cross-compile the weight-stationary GEMM (prefill vs decode) for arm64-android.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH. Add tools/ to PATH or use the container." >&2
    exit 1
fi
android-clang -std=c11 -O3 -ffp-contract=fast -o "$DIR/decode_neon" "$DIR/decode_neon.c" -lm
file "$DIR/decode_neon"
