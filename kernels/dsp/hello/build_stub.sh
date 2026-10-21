#!/usr/bin/env bash
# build_stub.sh — compile dsp_hello ARM64 stub.so + main exe (run inside control container)
# Usage: bash kernels/dsp/hello/build_stub.sh
#
# Two separate outputs:
#   libdsp_hello_stub.so  — ARM64 shared lib, DT_NEEDED: libcdsprpc.so
#   dsp_hello_test        — ARM64 exe, DT_NEEDED: libdl + libc only (no libcdsprpc)
#
# The test binary uses the QNN-validator pattern:
#   1. dlopen("libcdsprpc.so", RTLD_GLOBAL)   <- by name, linker finds /vendor/lib64,
#                                                loads in vendor namespace
#   2. dlopen("libdsp_hello_stub.so", ...)    <- stub's DT_NEEDED on cdsprpc already satisfied
#
# Expects:
#   - Android NDK at /opt/android-ndk-r27c (or NDK_ROOT env var)
#   - Hexagon SDK mounted at /sdk/hexagon
#     (docker run -v "C:/Qualcomm/Hexagon_SDK/<version>:/sdk/hexagon")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Android cross-compiler — prefer wrapper if available
if command -v android-clang >/dev/null 2>&1; then
    CC="android-clang"
else
    NDK_ROOT="${NDK_ROOT:-/opt/android-ndk-r27c}"
    CC="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang"
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

echo "=== Build step 1: libdsp_hello_stub.so (ARM64 shared lib, DT_NEEDED: libcdsprpc.so) ==="
echo "CC: $CC"
echo "SDK: $SDK"

"$CC" \
    -shared -fPIC \
    -I "$SDK_INCS" \
    -I "$SDK_INCS/stddef" \
    -I "$GEN_DIR" \
    "$GEN_DIR/dsp_hello_stub.c" \
    "$RPCMEM_A" \
    -L "$CDSPRPC_DIR" \
    -lcdsprpc \
    -Wl,-rpath-link,"$CDSPRPC_DIR" \
    -o "$OUT_DIR/libdsp_hello_stub.so"

echo "libdsp_hello_stub.so built"
file "$OUT_DIR/libdsp_hello_stub.so"

echo ""
echo "=== Build step 2: dsp_hello_test (ARM64 exe, DT_NEEDED: libdl + libc only) ==="

# main.c includes dsp_hello.h (for remote_handle64 typedef) — pull from generated dir.
# Link ONLY libdl (for dlopen). No -lcdsprpc; the preload happens at runtime.
"$CC" \
    -I "$SDK_INCS" \
    -I "$SDK_INCS/stddef" \
    -I "$GEN_DIR" \
    "$SRC_DIR/dsp_hello_main.c" \
    -ldl \
    -o "$OUT_DIR/dsp_hello_test"

echo "dsp_hello_test built"
file "$OUT_DIR/dsp_hello_test"

echo ""
echo "=== Verifying DT_NEEDED ==="
echo "libdsp_hello_stub.so NEEDED (expect libcdsprpc.so):"
readelf -d "$OUT_DIR/libdsp_hello_stub.so" 2>/dev/null | grep NEEDED || \
    llvm-readelf -d "$OUT_DIR/libdsp_hello_stub.so" | grep NEEDED || true

echo "dsp_hello_test NEEDED (expect libdl + libc only, NO libcdsprpc):"
readelf -d "$OUT_DIR/dsp_hello_test" 2>/dev/null | grep NEEDED || \
    llvm-readelf -d "$OUT_DIR/dsp_hello_test" | grep NEEDED || true

echo ""
echo "=== Build complete ==="
echo "Outputs:"
ls -lh "$OUT_DIR/libdsp_hello_stub.so" "$OUT_DIR/dsp_hello_test"
