#!/usr/bin/env bash
# Cross-compile the eigen_baseline matmul variant for arm64-android
# using the control container's `android-clang` wrapper.
#
# C++ runtime link note: `android-clang` invokes `clang` (not
# `clang++`), so the C++ runtime is NOT auto-linked. We pass
# `-lc++_static -lc++abi` explicitly. See
# `discoveries/24-matmul-build-pipeline-scoping.md` for the
# empirical finding.
#
# Run from inside the control container OR from the host:
#
#   # Inside the container:
#   bash kernels/matmul/cpu/eigen_baseline/build.sh
#
#   # From the host:
#   docker run --rm --platform linux/arm64 \
#     -v "$(pwd):/workspace" snapdragon-profiler-control:dev \
#     bash /workspace/kernels/matmul/cpu/eigen_baseline/build.sh

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_DIR="$(cd "$DIR/.." && pwd)"
SRC="$DIR/matmul_eigen.cpp"
OUT="$DIR/matmul_eigen"
# ORIGINAL LINE (Mac / fully-vendored setup):
# Points to the Eigen 3.4.0 headers that were committed under third_party/.
# On the original Mac setup these headers were complete (all of Eigen/src/Core/
# was present). Left here so you can see what the original intent was.
#EIGEN_INC="$CPU_DIR/third_party/eigen-3.4.0"

# WINDOWS / PRUNED-VENDOR FIX:
# The vendored Eigen headers in this repo were committed with the entire
# Eigen/src/Core/ directory missing (it was pruned to save space). The
# top-level Eigen/Core module header (which Eigen/Dense includes first)
# therefore cannot be found, and the build fails with:
#   fatal error: 'Core' file not found
#
# Fix: start by still pointing at the vendored copy (so the original path
# is tried first). Then, if Eigen/Core is not actually there, automatically
# fall back to the system-installed Eigen provided by the libeigen3-dev
# apt package (installed at /usr/include/eigen3/ inside the container).
#
# To install the system Eigen inside the container before building, run:
#   apt-get update && apt-get install -y libeigen3-dev
# (Ubuntu 24.04 ships Eigen 3.4.0, so the version matches.)
EIGEN_INC="$CPU_DIR/third_party/eigen-3.4.0"   # try vendored path first

# Check whether the vendored Eigen/Core header actually exists.
# -f tests that the path is a regular file (not a directory or symlink).
if [ ! -f "$EIGEN_INC/Eigen/Core" ]; then
    # Vendored Core is missing — try the system Eigen installed by libeigen3-dev.
    # /usr/include/eigen3 is the standard install location on Debian/Ubuntu.
    if [ -d "/usr/include/eigen3" ] && [ -f "/usr/include/eigen3/Eigen/Core" ]; then
        # System Eigen found — override EIGEN_INC to point there instead.
        EIGEN_INC="/usr/include/eigen3"
        echo "INFO: vendored Eigen/Core missing; using system Eigen at $EIGEN_INC"
    else
        # Neither vendored nor system Eigen has Core — we cannot build.
        # Tell the user exactly what to do to fix it.
        echo "ERROR: Eigen/Core not found in vendored headers ($EIGEN_INC/Eigen/Core)" >&2
        echo "       and no system Eigen detected at /usr/include/eigen3." >&2
        echo "       Fix: run   apt-get update && apt-get install -y libeigen3-dev" >&2
        echo "       inside the control container, then re-run this script." >&2
        exit 1
    fi
fi

if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH. Run from inside the control container." >&2
    exit 1
fi

# -O2: realistic optimization, matches diagnostic kernel default.
# -DEIGEN_MPL2_ONLY: pin to MPL2-clean Eigen paths (no LGPL).
# -fno-omit-frame-pointer: keep callstacks profilable for future
#   simpleperf record use.
# -lc++_static -lc++abi: explicit C++ runtime link, see header
#   comment.
android-clang -x c++ -std=c++17 -O2 -fno-omit-frame-pointer \
    -DEIGEN_MPL2_ONLY \
    -I"$EIGEN_INC" \
    "$SRC" \
    -lc++_static -lc++abi \
    -o "$OUT"

# Verify it's an Android arm64 binary.
file_out=$(file "$OUT")
echo "$file_out"
echo "$file_out" | grep -q 'ELF 64-bit LSB.*ARM aarch64' \
    || { echo "ERROR: $OUT is not an aarch64 ELF" >&2; exit 1; }

echo "Built: $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"
