#!/usr/bin/env bash
# build_stub.sh — compile dsp_conv ARM64 stub.so + main exe (control container)
# Usage: bash kernels/reduction/dsp/build_stub.sh
# Requires the generated/ dir next to this script (it ships with the bundle).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Same compiler wrapper new_kernel.py uses (it finds the NDK from install.py's
# state), so this works in the lab container without NDK_ROOT set by hand.
REPO_TOOLS="$SCRIPT_DIR/../../../tools"
[ -x "$REPO_TOOLS/android-clang" ] && PATH="$(cd "$REPO_TOOLS" && pwd):$PATH"
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

SDK="${SDK_ROOT:-${HEXAGON_SDK_ROOT:-}}"
if [ -z "$SDK" ]; then
    # /sdk/hexagon was the old image's layout; the lab container has the SDK
    # under Qualcomm/Hexagon_SDK/<version>. Take the newest one.
    # (SDK 6.7 no longer ships rpcmem.a; the stub never needed it.)
    for cand in /sdk/hexagon $(ls -d /local/mnt/workspace/Qualcomm/Hexagon_SDK/*/ \
                /opt/Qualcomm/Hexagon_SDK/*/ "$HOME"/Qualcomm/Hexagon_SDK/*/ 2>/dev/null | sort -rV); do
        if [ -d "$cand/incs" ]; then SDK="${cand%/}"; break; fi
    done
fi
[ -n "$SDK" ] || { echo "ERROR: Hexagon SDK not found; set SDK_ROOT" >&2; exit 1; }
SDK_INCS="$SDK/incs"
RPCMEM_A="$SDK/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a"
CDSPRPC_SO="$SDK/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so"
CDSPRPC_DIR="$(dirname "$CDSPRPC_SO")"

GEN_DIR="$SCRIPT_DIR/generated"
SRC_DIR="$SCRIPT_DIR/src"
OUT_DIR="$SCRIPT_DIR/prebuilt"

mkdir -p "$OUT_DIR"

echo "=== Step 1: libdsp_conv_stub.so (DT_NEEDED: libcdsprpc.so) ==="
"$CC" \
    -shared -fPIC \
    -I "$SDK_INCS" -I "$SDK_INCS/stddef" -I "$GEN_DIR" \
    "$GEN_DIR/dsp_conv_stub.c" \
    $( [ -f "$RPCMEM_A" ] && echo "$RPCMEM_A" ) \
    -L "$CDSPRPC_DIR" -lcdsprpc \
    -Wl,-rpath-link,"$CDSPRPC_DIR" \
    -o "$OUT_DIR/libdsp_conv_stub.so"
echo "stub built"; file "$OUT_DIR/libdsp_conv_stub.so"

echo ""
echo "=== Step 2: conv_dsp (single main, --variant scalar|hvx) ==="
"$CC" \
    -I "$SDK_INCS" -I "$SDK_INCS/stddef" -I "$GEN_DIR" \
    "$SRC_DIR/dsp_conv_main.c" \
    -lm -ldl \
    -o "$OUT_DIR/conv_dsp"
echo "main built"; file "$OUT_DIR/conv_dsp"

echo ""
echo "=== Step 3: conv_dsp_int8 (int8 in / int32 out, HVX vrmpy) ==="
"$CC" \
    -I "$SDK_INCS" -I "$SDK_INCS/stddef" -I "$GEN_DIR" \
    "$SRC_DIR/dsp_conv_int8_main.c" \
    -lm -ldl \
    -o "$OUT_DIR/conv_dsp_int8"
echo "int8 main built"; file "$OUT_DIR/conv_dsp_int8"

echo ""
echo "=== Step 4: conv_morph (Lab 3 Task 10 morphology) ==="
"$CC"     -I "$SDK_INCS" -I "$SDK_INCS/stddef" -I "$GEN_DIR"     "$SRC_DIR/dsp_conv_morph_main.c"     -lm -ldl     -o "$OUT_DIR/conv_morph"
echo "morph main built"; file "$OUT_DIR/conv_morph"

echo ""
echo "=== Step 5: conv_stage (Lab 3 Task 11 VTCM staging) ==="
"$CC"     -I "$SDK_INCS" -I "$SDK_INCS/stddef" -I "$GEN_DIR"     "$SRC_DIR/dsp_conv_stage_main.c"     -lm -ldl     -o "$OUT_DIR/conv_stage"
echo "stage main built"; file "$OUT_DIR/conv_stage"

echo ""
echo "=== Step 6: conv_vtcm_probe (Lab 3 Task 11.1-11.4) ==="
"$CC"     -I "$SDK_INCS" -I "$SDK_INCS/stddef" -I "$GEN_DIR"     "$SRC_DIR/dsp_conv_vtcm_probe_main.c"     -lm -ldl     -o "$OUT_DIR/conv_vtcm_probe"
echo "vtcm probe built"; file "$OUT_DIR/conv_vtcm_probe"

echo ""
echo "=== DT_NEEDED check ==="
echo "stub (expect libcdsprpc.so):"
readelf -d "$OUT_DIR/libdsp_conv_stub.so" 2>/dev/null | grep NEEDED || true
echo "conv_dsp (expect libm + libdl + libc):"
readelf -d "$OUT_DIR/conv_dsp" 2>/dev/null | grep NEEDED || true

echo ""
echo "=== Done ==="
ls -lh "$OUT_DIR/libdsp_conv_stub.so" "$OUT_DIR/conv_dsp"
