#!/usr/bin/env bash
# build_stub.sh — compile dsp_reduction ARM64 stub.so + main exe (control container)
# Usage: bash kernels/reduction/dsp/build_stub.sh
# Requires the generated/ dir next to this script (it ships with the bundle).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if command -v android-clang >/dev/null 2>&1; then
    CC="android-clang"
else
    NDK_ROOT="${NDK_ROOT:-/opt/android-ndk-r27c}"
    # The NDK ships one prebuilt dir per host OS; pick whichever is present so
    # this works from the container (linux) and from a host NDK (windows/darwin).
    NDK_PREBUILT=""
    for host in linux-x86_64 windows-x86_64 darwin-x86_64; do
        if [ -d "$NDK_ROOT/toolchains/llvm/prebuilt/$host" ]; then
            NDK_PREBUILT="$NDK_ROOT/toolchains/llvm/prebuilt/$host"; break
        fi
    done
    [ -n "$NDK_PREBUILT" ] || { echo "ERROR: no NDK prebuilt toolchain under $NDK_ROOT" >&2; exit 1; }
    CC="$NDK_PREBUILT/bin/aarch64-linux-android31-clang"
    [ -x "$CC" ] || CC="$CC.cmd"            # Windows NDK wraps clang in .cmd
fi

SDK="${SDK_ROOT:-/sdk/hexagon}"
SDK_INCS="$SDK/incs"
RPCMEM_A="$SDK/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a"
CDSPRPC_SO="$SDK/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so"
CDSPRPC_DIR="$(dirname "$CDSPRPC_SO")"

GEN_DIR="$SCRIPT_DIR/generated"
SRC_DIR="$SCRIPT_DIR/src"
OUT_DIR="$SCRIPT_DIR/prebuilt"

mkdir -p "$OUT_DIR"

echo "=== Step 1: libdsp_reduction_stub.so (DT_NEEDED: libcdsprpc.so) ==="
"$CC" \
    -shared -fPIC \
    -I "$SDK_INCS" -I "$SDK_INCS/stddef" -I "$GEN_DIR" \
    "$GEN_DIR/dsp_reduction_stub.c" \
    "$RPCMEM_A" \
    -L "$CDSPRPC_DIR" -lcdsprpc \
    -Wl,-rpath-link,"$CDSPRPC_DIR" \
    -o "$OUT_DIR/libdsp_reduction_stub.so"
echo "stub built"; file "$OUT_DIR/libdsp_reduction_stub.so"

echo ""
echo "=== Step 2: reduction_dsp (single main, --variant scalar|hvx) ==="
"$CC" \
    -I "$SDK_INCS" -I "$SDK_INCS/stddef" -I "$GEN_DIR" \
    "$SRC_DIR/dsp_reduction_main.c" \
    -lm -ldl \
    -o "$OUT_DIR/reduction_dsp"
echo "main built"; file "$OUT_DIR/reduction_dsp"

echo ""
echo "=== DT_NEEDED check ==="
echo "stub (expect libcdsprpc.so):"
readelf -d "$OUT_DIR/libdsp_reduction_stub.so" 2>/dev/null | grep NEEDED || true
echo "reduction_dsp (expect libm + libdl + libc):"
readelf -d "$OUT_DIR/reduction_dsp" 2>/dev/null | grep NEEDED || true

echo ""
echo "=== Done ==="
ls -lh "$OUT_DIR/libdsp_reduction_stub.so" "$OUT_DIR/reduction_dsp"
