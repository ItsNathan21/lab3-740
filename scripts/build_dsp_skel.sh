#!/usr/bin/env bash
# build_dsp_skel.sh - compile a Hexagon DSP skel .so, on any host.
#
# The bash counterpart to kernels/<k>/dsp/build_skel.ps1. Same output, same
# flags, but it runs anywhere bash does:
#
#   Linux / the qualcomm container   native Hexagon SDK for linux-x86_64
#   macOS (Apple Silicon or Intel)   inside the linux/amd64 container
#   Windows (Git Bash / MSYS)        against the native SDK, .exe tools
#
# The SDK has no macOS build, which is why a Mac goes through the container.
# Everything else about the invocation is identical on all three.
#
#   bash scripts/build_dsp_skel.sh kernels/convolution/dsp
#   bash scripts/build_dsp_skel.sh kernels/convolution/dsp --arch v81 --threads 8
#
# Options
#   --arch vNN      Hexagon architecture. Default: read ro.soc.model off an
#                   attached board via adb; falls back to nothing (hard error),
#                   so a build host with no board must pass this explicitly.
#   --threads N     HVX worker contexts baked in as CONV_MT_THREADS. Default 8
#                   (SM8850P reports 8 x 128B contexts). Pass 6 for an SM8750.
#   --sdk PATH      Hexagon SDK root. Default: $HEXAGON_SDK_ROOT, then the
#                   usual install locations.
#   --regen         Re-run qaic even though generated/ is already committed.
#   -h, --help

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------- arguments
KDIR="" ARCH="" THREADS=8 SDK="" REGEN=0
EXCLUDE=() EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --arch)    ARCH="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --sdk)     SDK="$2"; shift 2 ;;
        --exclude) EXCLUDE+=("$2"); shift 2 ;;
        --add)     EXTRA+=("$2"); shift 2 ;;
        --regen)   REGEN=1; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        -*)        die "unknown option: $1" ;;
        *)         [ -z "$KDIR" ] || die "give exactly one kernel dir"; KDIR="$1"; shift ;;
    esac
done
[ -n "$KDIR" ] || die "usage: $0 <kernel dsp dir> [--arch vNN] [--threads N]"
[ -d "$KDIR" ] || die "no such directory: $KDIR"
KDIR="$(cd "$KDIR" && pwd)"

# ------------------------------------------------------------------ the SDK
#
# Any 6.x will do, and we take the NEWEST installed rather than a pinned
# version. Qualcomm stopped reliably serving 6.6.0.0 through QPM, so a hardcoded
# version is a dependency on them keeping an old release available -- and when
# that broke, this script found nothing while check_setup.py (which globs) said
# SETUP OK. A gate that passes and then a build that fails is the worst
# arrangement of the two.
#
# HEXAGON_SDK_MIN is a floor, not a pin: 6.6.0.0 is where HAP_compute_res.h
# arrived, which the VTCM kernels need (Discovery 44). Below that the build
# fails later and less clearly, so refuse it here.
HEXAGON_SDK_MIN="${HEXAGON_SDK_MIN:-6.6.0.0}"

# Sortable key for a dotted version: 6.10.0.0 must rank above 6.9.0.0, which
# is exactly what a plain string sort gets wrong.
vkey() { printf '%s' "$1" | awk -F. '{printf "%05d%05d%05d%05d", $1,$2,$3,$4}'; }

if [ -z "$SDK" ]; then
    best=""; best_k=""
    for base in \
        /opt/Qualcomm/Hexagon_SDK \
        "$HOME/Qualcomm/Hexagon_SDK" \
        /local/mnt/workspace/Qualcomm/Hexagon_SDK \
        /opt/qcom/Hexagon_SDK \
        /c/Qualcomm/Hexagon_SDK \
        "C:/Qualcomm/Hexagon_SDK"
    do
        [ -d "$base" ] || continue
        for cand in "$base"/*; do
            [ -d "$cand/incs" ] || continue
            v="$(basename "$cand")"
            case "$v" in [0-9]*) ;; *) continue ;; esac
            k="$(vkey "$v")"
            [ "$k" -lt "$(vkey "$HEXAGON_SDK_MIN")" ] && continue
            if [ -z "$best_k" ] || [ "$k" -gt "$best_k" ]; then best="$cand"; best_k="$k"; fi
        done
    done
    # An explicit HEXAGON_SDK_ROOT outranks discovery: if someone went to the
    # trouble of setting it, honour it even when it is older.
    if [ -n "${HEXAGON_SDK_ROOT:-}" ] && [ -d "${HEXAGON_SDK_ROOT}/incs" ]; then
        SDK="$HEXAGON_SDK_ROOT"
    else
        SDK="$best"
    fi
fi
[ -n "$SDK" ] || die "Hexagon SDK not found (looking for $HEXAGON_SDK_MIN or newer).
  Set HEXAGON_SDK_ROOT or pass --sdk.
  In the lab container qpm-cli installs it under
  /local/mnt/workspace/Qualcomm/Hexagon_SDK/<version>, which is searched above,
  so seeing this there means the install did not finish or it was sent
  elsewhere with --path. Check with: hexagon-doctor"
[ -d "$SDK/incs" ] || die "not a Hexagon SDK root (no incs/): $SDK"
echo "Hexagon SDK: $SDK"

# Tool binaries carry .exe on Windows and nothing everywhere else.
#
# .exe is only accepted when we are actually on Windows. A bind-mounted Windows
# SDK inside a Linux container shows its .exe files as executable, so without
# this guard we would pick one up and fail with "cannot execute binary file"
# instead of saying what is really wrong.
case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW*|MSYS*|CYGWIN*) HOST_IS_WINDOWS=1 ;;
    *)                    HOST_IS_WINDOWS=0 ;;
esac

find_tool() {
    local base="$1"
    if [ -x "$base" ] && [ ! -d "$base" ]; then echo "$base"; return 0; fi
    if [ "$HOST_IS_WINDOWS" = 1 ] && [ -f "$base.exe" ]; then echo "$base.exe"; return 0; fi
    if [ "$HOST_IS_WINDOWS" = 0 ] && [ -f "$base.exe" ]; then
        echo "error: found $(basename "$base").exe but this host is not Windows.
  You are pointing at a Windows Hexagon SDK from a non-Windows host. The Linux
  SDK is a separate download from the Qualcomm Package Manager." >&2
        return 1
    fi
    return 1
}

TOOLBIN=""
for t in "$SDK"/tools/HEXAGON_Tools/*/Tools/bin; do
    if [ -d "$t" ]; then TOOLBIN="$t"; fi
done
[ -n "$TOOLBIN" ] || die "no HEXAGON_Tools/*/Tools/bin under $SDK"
CLANG="$(find_tool "$TOOLBIN/hexagon-clang")" \
    || die "hexagon-clang not found in $TOOLBIN"

# qaic ships one binary per host OS; Ubuntu covers the container and Linux.
QAIC=""
for q in "$SDK/ipc/fastrpc/qaic/Ubuntu/qaic" "$SDK/ipc/fastrpc/qaic/WinNT/qaic"; do
    QAIC="$(find_tool "$q" 2>/dev/null)" && break || QAIC=""
done

# ------------------------------------------------------------ architecture
if [ -z "$ARCH" ]; then
    soc=""
    command -v adb >/dev/null 2>&1 && \
        soc="$(adb shell getprop ro.soc.model 2>/dev/null | tr -d '\r\n ' || true)"
    case "$soc" in
        SM8750)          ARCH=v79 ;;
        SM8850|SM8850P)  ARCH=v81 ;;
        *)  die "cannot auto-detect the Hexagon architecture (ro.soc.model='$soc').
  Pass --arch v81 for an HDK8850, or --arch v79 for an HDK8750.
  A build host with no board attached always needs this." ;;
    esac
    echo "Auto-detected SoC $soc -> Hexagon $ARCH"
fi
case "$ARCH" in v[0-9][0-9]) ;; *) die "--arch must look like v81, got: $ARCH" ;; esac
[ -d "$SDK/rtos/qurt/compute$ARCH" ] \
    || die "this SDK has no compute$ARCH. Re-run the Qualcomm Package Manager and
  select the $ARCH runtime; without it nothing you build will run on the board."

# ------------------------------------------------------------------- layout
GEN="$KDIR/generated"; SRC="$KDIR/src"; INC="$KDIR/inc"; OUT="$KDIR/prebuilt"
mkdir -p "$GEN" "$OUT"
IDL="$(ls "$INC"/*.idl 2>/dev/null | head -1)" || true
[ -n "${IDL:-}" ] || die "no .idl under $INC"
NAME="$(basename "$IDL" .idl)"

# ------------------------------------------- step 1: qaic (usually a no-op)
# generated/ is committed, so this only runs on a fresh IDL or with --regen.
if [ "$REGEN" = 1 ] || [ ! -f "$GEN/${NAME}_skel.c" ]; then
    [ -n "$QAIC" ] || die "generated/${NAME}_skel.c is missing and qaic was not
  found under $SDK/ipc/fastrpc/qaic/. It is normally committed to git - check
  you have a complete checkout before reaching for the SDK."
    echo "=== Step 1: qaic - IDL to stub/skel/header ==="
    "$QAIC" "$IDL" -o "$GEN" -I "$SDK/incs" -I "$SDK/incs/stddef"
else
    echo "=== Step 1: qaic - skipped, generated/${NAME}_skel.c is present ==="
fi

# ------------------------------------------------- step 2: compile the skel
# Everything in src/ is DSP-side except the *_main.c files, which are the ARM
# hosts and are built separately by build_stub.sh.
#
# --exclude drops one by basename and --add substitutes a file from elsewhere,
# which is how a student file replaces the reference implementation of the same
# entry point without either copy being edited.
SOURCES=()
for f in "$SRC"/*.c; do
    b="$(basename "$f")"
    case "$b" in *_main.c) continue ;; esac
    skip=0
    for x in ${EXCLUDE[@]+"${EXCLUDE[@]}"}; do
        if [ "$b" = "$(basename "$x")" ]; then skip=1; fi
    done
    if [ "$skip" = 1 ]; then continue; fi
    SOURCES+=("$f")
done
for f in ${EXTRA[@]+"${EXTRA[@]}"}; do
    [ -f "$f" ] || die "--add: no such file: $f"
    SOURCES+=("$(cd "$(dirname "$f")" && pwd)/$(basename "$f")")
done
[ ${#SOURCES[@]} -gt 0 ] || die "no DSP sources under $SRC"

# The SDK's DMA helper is only linked in when a source actually asks for it.
# Adding it unconditionally still builds, but it bloats every other kernel's
# skel by ~5 KB of dead code and stops the output matching build_skel.ps1.
DMA="$SDK/addons/compute/libs/userdma/utils_lib/src/dma_utils.c"
if [ -f "$DMA" ] && grep -qs 'dma_utils\.h' "$SRC"/*.c; then
    SOURCES+=("$DMA")
fi

echo ""
echo "=== Step 2: hexagon-clang -m$ARCH - compile skel ==="
echo "  sdk     $SDK"
echo "  arch    $ARCH   threads $THREADS"
echo "  sources $(for s in "${SOURCES[@]}"; do basename "$s"; done | tr '\n' ' ')"

INCLUDES=(
    -I "$SDK/incs"
    -I "$SDK/incs/stddef"
    -I "$SDK/rtos/qurt/compute$ARCH/include/qurt"
    -I "$SDK/rtos/qurt/compute$ARCH/include/posix"
    -I "$SDK/addons/compute/libs/userdma/utils_lib/api"
    -I "$SDK/addons/compute/libs/userdma/utils_lib/inc"
    -I "$SDK/libs/qhl_hvx/inc/qhdsp_hvx"
    -I "$SDK/libs/qhl_hvx/inc/qhblas_hvx"
    -I "$SDK/libs/qhl_hvx/inc/qhmath_hvx"
    -I "$SDK/libs/qhl/inc/qhcomplex"
    -I "$SDK/libs/qhl/inc/qhmath"
    -I "$GEN"
)

# Qualcomm's HVX math libraries, linked only for kernels that call into them.
# Convolution does; matmul, saxpy and reduction do not, and linking them anyway
# grows those skels by ~160 bytes and stops the output matching build_skel.ps1.
QHL="$SDK/libs/qhl_hvx/prebuilt/hexagon_toolv19_$ARCH"
LIBS=()
if grep -qs '#include *["<]qh' "$SRC"/*.c; then
    for a in libqhdsp_hvx.a libqhmath_hvx.a; do
        if [ -f "$QHL/$a" ]; then LIBS+=("$QHL/$a"); fi
    done
fi

# Each kernel names its own thread-count macro (CONV_MT_THREADS, MM_MT_THREADS);
# saxpy and reduction have none. Define whichever this kernel actually uses, so
# --threads works everywhere and no kernel gets a define it never reads.
DEFINES=()
# The `|| true` matters: saxpy and reduction have no such macro, and under
# `set -o pipefail` a grep that finds nothing would otherwise abort the build.
MACRO="$(grep -rho '[A-Z][A-Z0-9_]*_MT_THREADS' "$SRC"/*.c 2>/dev/null | sort -u | head -1 || true)"
if [ -n "$MACRO" ]; then DEFINES+=("-D$MACRO=$THREADS"); fi

"$CLANG" \
    "-m$ARCH" -mhvx -mhvx-length=128B \
    -fPIC -G0 -O2 -shared \
    "${DEFINES[@]}" \
    "${INCLUDES[@]}" \
    "$GEN/${NAME}_skel.c" \
    "${SOURCES[@]}" \
    "${LIBS[@]}" \
    -o "$OUT/lib${NAME}_skel.so"

echo ""
echo "=== Build complete ==="
if READELF="$(find_tool "$TOOLBIN/hexagon-readelf" 2>/dev/null)"; then
    "$READELF" -h "$OUT/lib${NAME}_skel.so" | grep -E "Machine|Class|Type" || true
fi
ls -l "$OUT/lib${NAME}_skel.so"
echo ""
echo "Push it with:"
echo "  adb push $OUT/lib${NAME}_skel.so /data/local/tmp/profiler-kernels/dsp/"
