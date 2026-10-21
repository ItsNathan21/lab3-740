"""NumPy reference for SAXPY correctness validation (Phase 10 / roofline).

SAXPY is `y <- alpha * x + y`, single-precision. It is the memory-bound
anchor of the roofline (arithmetic intensity ~0.17 FLOP/byte) — the
opposite corner from matmul. See `kernels/saxpy/DESIGN.md`.

Inputs `x` and `y` are generated with the SAME splitmix64 PRNG + uint64->fp32
conversion as the matmul kernels (`kernels/matmul/cpu/reference.py`), so a
given (size, seed) produces bit-identical `x` and `y` in C and Python. That is
the basis for device-variant <-> reference comparison.

## Idempotency note

SAXPY is NOT idempotent: applying it twice gives `alpha*x + (alpha*x + y)`.
The reference therefore models exactly ONE pass, and the device binaries only
emit a reference-comparable fingerprint under `--validate-only` (one pass).
Timing runs apply K accumulating passes — identical memory traffic per pass,
which is what a bandwidth measurement wants — and their fingerprint is not
expected to match this single-pass reference.

Validation is done at a modest size (the pure-Python splitmix64 generator is
sequential and slow at 2^24); correctness does not depend on the vector being
DRAM-sized, so the host validates at e.g. 2^16 and measures bandwidth at large
sizes separately.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Splitmix64 constants — identical to matmul/cpu/reference.py and the C kernels
# (https://prng.di.unimi.it/splitmix64.c).
_SPLITMIX_INC = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_U64_MASK = 0xFFFFFFFFFFFFFFFF

DEFAULT_ALPHA = 2.0


def splitmix64_next(state: int) -> Tuple[int, int]:
    """Return `(new_state, output)` per splitmix64."""
    state = (state + _SPLITMIX_INC) & _U64_MASK
    z = state
    z = ((z ^ (z >> 30)) * _SPLITMIX_M1) & _U64_MASK
    z = ((z ^ (z >> 27)) * _SPLITMIX_M2) & _U64_MASK
    z = z ^ (z >> 31)
    return state, z


def u64_to_unit_double(x: int) -> float:
    """uint64 -> double in [0, 1) using the top 53 bits.

    Matches the C `(x >> 11) * (1.0 / (1ULL << 53))`.
    """
    return (x >> 11) * (1.0 / (1 << 53))


def generate_inputs(size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """The same (x, y) fp32 vectors the device kernels see for (size, seed).

    First `size` draws are `x`, next `size` are `y` — the same "A then B"
    ordering the matmul kernels use. Each element is `2*unit_double - 1` in
    [-1, 1), cast to float32.
    """
    state = seed & _U64_MASK
    out = np.empty(2 * size, dtype=np.float64)
    for i in range(2 * size):
        state, val = splitmix64_next(state)
        out[i] = 2.0 * u64_to_unit_double(val) - 1.0
    x = out[:size].astype(np.float32, copy=False)
    y = out[size:].astype(np.float32, copy=False)
    return x, y


def fingerprint(y: np.ndarray) -> float:
    """`sum(y) + L2_norm(y)` of the updated vector, in fp64.

    The device variants compute the same quantity. Match within 1e-3 relative
    tolerance (fp32 reorder-tolerant), same policy as matmul M4.
    """
    return float(y.astype(np.float64).sum()) + float(
        np.linalg.norm(y.astype(np.float64))
    )


def reference_fingerprint(size: int, seed: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Canonical single-pass fingerprint for (size, seed, alpha)."""
    x, y = generate_inputs(size, seed)
    y_out = (np.float32(alpha) * x + y).astype(np.float32)
    return fingerprint(y_out)


if __name__ == "__main__":
    import time

    for n in (1 << 16, 1 << 18):
        t0 = time.monotonic()
        fp = reference_fingerprint(n, 42)
        dt = (time.monotonic() - t0) * 1000
        print(f"reference_fingerprint({n}, 42, alpha=2.0) = {fp:.6e}  ({dt:.0f} ms)")
