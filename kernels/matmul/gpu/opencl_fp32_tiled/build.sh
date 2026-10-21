#!/usr/bin/env bash
# Cross-compile matmul_opencl_fp32_tiled for arm64-android (Phase 4 Step 8).
#
# Produces: kernels/matmul/gpu/opencl_fp32_tiled/matmul_opencl_fp32_tiled
#   A standalone arm64 ELF that:
#     - dlopen's /vendor/lib64/libOpenCL.so at runtime
#     - compiles an embedded tiled OpenCL C kernel via the Adreno on-device compiler
#     - runs a 512×512 FP32 tiled matmul (TILE=16) using __local SRAM
#     - prints a GPU_DISPATCH_RESULT protocol block to stdout
#
# Key difference from the naive kernel (opencl_fp32/build.sh):
#   The embedded kernel uses __local float tileA[TILE][TILE] and tileB[TILE][TILE]
#   to load TILE×TILE blocks of A and B into on-chip SRAM before computing.
#   Each element is reused TILE=16 times vs once in the naive version.
#   Two barriers per tile iteration (CLK_LOCAL_MEM_FENCE) synchronize all
#   workitems in the workgroup after loading and after accumulating.
#
# Requires android-clang (installed inside the control container by the
# Phase 1 Dockerfile). Does NOT require static OpenCL linkage — the ICD
# loader is opened at runtime via dlopen (same pattern as opencl_enum and
# the naive matmul_opencl_fp32).
#
# Compiler flags:
#   -std=c11              Pure C11, no C++ runtime needed.
#   -O2                   Production-level optimization; matches CPU variants.
#   -fno-omit-frame-pointer  Preserve call frames for future simpleperf record.
#   -Wall -Wextra         Treat most warnings as-is; source is warning-clean.
#   -ldl                  Link libdl for dlopen/dlsym (Android linker helpers).
#   -lm                   Link libm for sqrt() in the fingerprint computation.
#   (no -lOpenCL)         OpenCL is loaded at runtime; no compile-time stub needed.
#
# Run from inside the control container OR from the host (one-shot):
#
#   # Inside the container (preferred — android-clang is on PATH):
#   bash kernels/matmul/gpu/opencl_fp32_tiled/build.sh
#
#   # From the host (bind-mounts the project root):
#   docker run --rm --platform linux/amd64 \
#     -v "$(pwd):/workspace" snapdragon-profiler-control:dev \
#     bash /workspace/kernels/matmul/gpu/opencl_fp32_tiled/build.sh

set -euo pipefail

# ── Locate source and output paths ────────────────────────────────────────────
# DIR = directory containing this script (kernels/matmul/gpu/opencl_fp32_tiled/)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source file: the tiled matmul C wrapper (Phase 4 Step 8)
SRC="$DIR/matmul_opencl_fp32_tiled.c"

# Output binary: _tiled suffix distinguishes it from the naive binary on the device
OUT="$DIR/matmul_opencl_fp32_tiled"

# ── Sanity-check: android-clang must be available ─────────────────────────────
# android-clang is the NDK wrapper installed in the control container.
# It sets --target=aarch64-linux-android24, the correct sysroot, and
# the Android-specific runtime library paths automatically.
if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH." >&2
    echo "       Run this script from inside the control container." >&2
    echo "       (The NDK wrapper is installed by the Phase 1 Dockerfile.)" >&2
    exit 1
fi

echo "Building matmul_opencl_fp32_tiled (Phase 4 Step 8 tiled GPU kernel)..."
echo "  Source: $SRC"
echo "  Output: $OUT"
echo "  Tiled:  TILE=16, __local SRAM, 2 barriers/tile iteration"
echo ""

# ── Compile ───────────────────────────────────────────────────────────────────
# Flags are identical to the naive build — the tiling is inside the OpenCL C
# kernel string, which is compiled on-device by the Adreno online compiler.
# The host-side C wrapper is architecturally the same as the naive version.
android-clang -std=c11 -O2 -fno-omit-frame-pointer \
    -Wall -Wextra \
    "$SRC" \
    -ldl -lm \
    -o "$OUT"

# ── Verify the output is an arm64 Android ELF ─────────────────────────────────
# `file` reads the ELF header. Grep for the two tokens that confirm:
#   - ELF 64-bit LSB: 64-bit little-endian (aarch64 is always LSB)
#   - ARM aarch64: correct target architecture
# A wrong compiler or accidentally cross-compiling to x86_64 would pass the
# link step but produce a binary that segfaults immediately on the HDK.
file_out=$(file "$OUT")
echo "$file_out"
echo "$file_out" | grep -q 'ELF 64-bit LSB.*ARM aarch64' \
    || { echo "ERROR: $OUT is not an aarch64 ELF — wrong compiler?" >&2; exit 1; }

# ── Report size ───────────────────────────────────────────────────────────────
# stat -c%s is GNU coreutils (Linux/container); stat -f%z is BSD/macOS.
SIZE=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Built: $OUT ($SIZE bytes)"
echo ""
echo "Phase 4 Step 8 — next steps:"
echo ""
echo "  1. Push and run on the HDK:"
echo "     adb push $OUT /data/local/tmp/profiler-kernels/gpu/matmul_opencl_fp32_tiled"
echo "     adb shell chmod +x /data/local/tmp/profiler-kernels/gpu/matmul_opencl_fp32_tiled"
echo "     adb shell /data/local/tmp/profiler-kernels/gpu/matmul_opencl_fp32_tiled"
echo ""
echo "  2. Or use execute_gpu_dispatch() from the orchestrator:"
echo "     from orchestrator.backends.gpu import execute_gpu_dispatch, get_kernel"
echo "     outcome = execute_gpu_dispatch(connection, kernel=get_kernel('opencl_matmul_fp32_tiled'))"
echo ""
echo "  Expected stdout: GPU_DISPATCH_RESULT block with status: OK"
echo "  and cl_start_ns / cl_end_ns / duration_ns populated."
echo ""
echo "  Compare against naive (5.277 ms) — expect 3-10x speedup from local SRAM."
echo "  Fingerprint must match naive within FP32 tolerance (~1e-3 relative error)"
echo "  to confirm the tiled kernel produces correct results."
echo "════════════════════════════════════════════════════════════"
