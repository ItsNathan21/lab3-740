#!/usr/bin/env bash
# Cross-compile matmul_opencl_fp16 for arm64-android (Phase 4 Step 9).
#
# Produces: kernels/matmul/gpu/opencl_fp16/matmul_opencl_fp16
#   A standalone arm64 ELF that:
#     - dlopen's /vendor/lib64/libOpenCL.so at runtime
#     - compiles an embedded FP16 OpenCL C kernel via the Adreno on-device compiler
#     - runs a 512×512 FP16 matmul (half inputs/output, float accumulator)
#     - prints a GPU_DISPATCH_RESULT protocol block to stdout
#
# Key differences from the FP32 naive kernel (opencl_fp32/build.sh):
#   The C wrapper uses __fp16 (ARM native 16-bit float type) to fill the host
#   matrices with FP16 values converted from the same splitmix64 float stream as
#   the FP32 kernel. The embedded OpenCL C kernel requires:
#       #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#   confirmed available on Adreno 830 in discovery 27.
#
#   __fp16 is an ARM C extension natively supported on aarch64 — no extra flags
#   are needed beyond the standard android-clang invocation. The aarch64 ABI
#   treats __fp16 as a first-class type with hardware FCVT support.
#
# Requires android-clang (installed inside the control container by the
# Phase 1 Dockerfile). Does NOT require static OpenCL linkage — the ICD
# loader is opened at runtime via dlopen (same pattern as all GPU kernels).
#
# Compiler flags: identical to FP32 naive and tiled variants.
#   -std=c11              Pure C11, no C++ runtime needed.
#   -O2                   Production-level optimization; matches CPU variants.
#   -fno-omit-frame-pointer  Preserve call frames for future simpleperf record.
#   -Wall -Wextra         Treat most warnings as-is; source is warning-clean.
#   -ldl                  Link libdl for dlopen/dlsym (Android linker helpers).
#   -lm                   Link libm for sqrt() in the fingerprint computation.
#   (no -lOpenCL)         OpenCL is loaded at runtime; no compile-time stub needed.
#   (no -mfpu=neon-fp16)  __fp16 is part of the base aarch64 ABI; this flag is
#                         for arm32 only and must NOT be passed to aarch64 clang.
#
# Run from inside the control container OR from the host (one-shot):
#
#   # Inside the container (preferred — android-clang is on PATH):
#   bash kernels/matmul/gpu/opencl_fp16/build.sh
#
#   # From the host (bind-mounts the project root):
#   docker run --rm --platform linux/amd64 \
#     -v "$(pwd):/workspace" snapdragon-profiler-control:dev \
#     bash /workspace/kernels/matmul/gpu/opencl_fp16/build.sh

set -euo pipefail

# ── Locate source and output paths ────────────────────────────────────────────
# DIR = directory containing this script (kernels/matmul/gpu/opencl_fp16/)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="$DIR/matmul_opencl_fp16.c"
OUT="$DIR/matmul_opencl_fp16"

# ── Sanity-check: android-clang must be available ─────────────────────────────
if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH." >&2
    echo "       Run this script from inside the control container." >&2
    echo "       (The NDK wrapper is installed by the Phase 1 Dockerfile.)" >&2
    exit 1
fi

echo "Building matmul_opencl_fp16 (Phase 4 Step 9 FP16 GPU kernel)..."
echo "  Source: $SRC"
echo "  Output: $OUT"
echo "  Precision: FP16 inputs/output, float accumulator (mixed-precision)"
echo "  Extension: cl_khr_fp16 (confirmed on Adreno 830, discovery 27)"
echo ""

# ── Compile ───────────────────────────────────────────────────────────────────
# Flags identical to FP32 naive and tiled variants.
# __fp16 is natively supported on aarch64 — no extra FP16 flags needed.
android-clang -std=c11 -O2 -fno-omit-frame-pointer \
    -Wall -Wextra \
    "$SRC" \
    -ldl -lm \
    -o "$OUT"

# ── Verify the output is an arm64 Android ELF ─────────────────────────────────
file_out=$(file "$OUT")
echo "$file_out"
echo "$file_out" | grep -q 'ELF 64-bit LSB.*ARM aarch64' \
    || { echo "ERROR: $OUT is not an aarch64 ELF — wrong compiler?" >&2; exit 1; }

# ── Report size ───────────────────────────────────────────────────────────────
SIZE=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Built: $OUT ($SIZE bytes)"
echo ""
echo "Phase 4 Step 9 — next steps:"
echo ""
echo "  1. Push and run on the HDK:"
echo "     adb push $OUT /data/local/tmp/profiler-kernels/gpu/matmul_opencl_fp16"
echo "     adb shell chmod +x /data/local/tmp/profiler-kernels/gpu/matmul_opencl_fp16"
echo "     adb shell /data/local/tmp/profiler-kernels/gpu/matmul_opencl_fp16"
echo ""
echo "  2. Or use execute_gpu_dispatch() from the orchestrator:"
echo "     from orchestrator.backends.gpu import execute_gpu_dispatch, get_kernel"
echo "     outcome = execute_gpu_dispatch(connection, kernel=get_kernel('opencl_matmul_fp16'))"
echo ""
echo "  Expected stdout: GPU_DISPATCH_RESULT block with status: OK"
echo "  Expected timing: ~2.5-3 ms (expected ~2x speedup vs FP32 naive 5.28 ms)"
echo "  Fingerprint will DIFFER from FP32 reference (33585655.900771) due to"
echo "  FP16 rounding of both inputs and output — this is expected and correct."
echo "════════════════════════════════════════════════════════════"
