"""NumPy reference for matmul correctness validation (Phase 3.5).

Per `kernels/matmul/cpu/DESIGN.md` M4 (architect-resolved with
tolerance asymmetry, Session 25 + 26): each variant's output is
validated against this reference within 1e-3 relative tolerance
to absorb legitimate fp32 reorder-dependent accumulation
differences. Same-variant-same-seed reproducibility is checked
separately at bit-exact / 1e-7 absolute (Q2 in DESIGN.md).

The variant kernels (eigen_baseline, neon_handtuned) use the
same splitmix64 PRNG + uint64→fp32 conversion documented here,
so seed=42 produces the SAME A and B matrices in C/C++ as in
this Python reference. That's the basis for variant ↔
reference comparison.

## Why splitmix64

- Trivial to implement bit-identically in C and Python (no
  language-runtime divergence).
- Stateless after the first seed-mix: each call produces a
  fresh uint64, no buffer-management or global state to track.
- Fast enough for the matrix sizes we care about (512×512
  doubles = 524288 elements; ~5ms in pure Python).

NumPy's default RNG is convenient but its bit-level output
isn't trivially reproducible from C; sticking to splitmix64
keeps the reference and the variants in lockstep.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Splitmix64 constants per the canonical reference
# (https://prng.di.unimi.it/splitmix64.c).
_SPLITMIX_INC = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_U64_MASK = 0xFFFFFFFFFFFFFFFF


def splitmix64_next(state: int) -> Tuple[int, int]:
    """Return `(new_state, output)` per splitmix64.

    `output` is the uint64 to consume; `new_state` is what the
    caller should pass on the next invocation.
    """
    state = (state + _SPLITMIX_INC) & _U64_MASK
    z = state
    z = ((z ^ (z >> 30)) * _SPLITMIX_M1) & _U64_MASK
    z = ((z ^ (z >> 27)) * _SPLITMIX_M2) & _U64_MASK
    z = z ^ (z >> 31)
    return state, z


def u64_to_unit_double(x: int) -> float:
    """Convert a uint64 to a double in [0, 1) using the top 53 bits.

    Standard IEEE 754 conversion — matches the C
    `(x >> 11) * (1.0 / (1ULL << 53))` pattern used by the
    variant kernels.
    """
    return (x >> 11) * (1.0 / (1 << 53))


def generate_inputs(size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the same (A, B) fp32 matrices the device-side
    variants will see for this `(size, seed)`.

    Each element is in [-1, 1) by 2 * unit_double - 1, then cast
    to float32 (single-precision round-to-nearest, which the
    hardware path matches).
    """
    n = size * size
    state = seed & _U64_MASK
    out = np.empty(2 * n, dtype=np.float64)
    # 2*n elements: first n for A, next n for B (consistent with
    # the device-side ordering documented in the variant kernels).
    for i in range(2 * n):
        state, val = splitmix64_next(state)
        out[i] = 2.0 * u64_to_unit_double(val) - 1.0
    A = out[:n].reshape(size, size).astype(np.float32, copy=False)
    B = out[n:].reshape(size, size).astype(np.float32, copy=False)
    return A, B


def fingerprint(C: np.ndarray) -> float:
    """`sum(C) + frobenius_norm(C)` of the result matrix.

    The variants compute the same quantity on their device-side
    output. Match within 1e-3 relative tolerance per DESIGN.md
    M4 / Q1.
    """
    fro = float(np.linalg.norm(C, ord="fro"))
    s = float(C.sum())
    return s + fro


def reference_fingerprint(size: int, seed: int) -> float:
    """The canonical fingerprint for `(size, seed)`. Variants
    must agree within 1e-3 relative tolerance."""
    A, B = generate_inputs(size, seed)
    C = A @ B
    return fingerprint(C)


# Module-level cache for the common `(512, 42)` case so unit tests
# that reuse it don't recompute (~80 ms on a modern Mac).
_FINGERPRINT_CACHE: dict[tuple[int, int], float] = {}


def cached_reference_fingerprint(size: int, seed: int) -> float:
    key = (size, seed)
    if key not in _FINGERPRINT_CACHE:
        _FINGERPRINT_CACHE[key] = reference_fingerprint(size, seed)
    return _FINGERPRINT_CACHE[key]


if __name__ == "__main__":
    # Diagnostic when run directly: print the v1 reference
    # fingerprint for the canonical (512, 42).
    import time
    t0 = time.monotonic()
    fp = reference_fingerprint(512, 42)
    elapsed = time.monotonic() - t0
    print(f"reference_fingerprint(512, 42) = {fp:.6e}  ({elapsed*1000:.1f} ms)")
