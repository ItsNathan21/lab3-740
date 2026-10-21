"""NumPy reference for sum-reduction correctness validation (roofline, 3rd workload).

Reduction is `s <- sum(x_i)`, single-precision input. It is the *second*
memory-bound anchor of the roofline (arithmetic intensity ~0.25 FLOP/byte,
read-only) — distinct from SAXPY (read+write stream) because it collapses N
elements to a single scalar, so the output-transport tax that punished the GPU/
DSP/NPU on SAXPY vanishes here. See `kernels/reduction/DESIGN.md`.

Input `x` is generated with the SAME splitmix64 PRNG + uint64->fp32 conversion
as the SAXPY and matmul kernels (`kernels/saxpy/cpu/reference.py`), so a given
(size, seed) produces a bit-identical `x` in C and Python. `x` takes the first
`size` draws — the same leading-draw convention every kernel uses.

## Reduction-order / associativity note (decision R6)

Floating-point addition is NOT associative, so sequential, pairwise/tree, and
HVX-lane-parallel summations of the SAME `x` produce DIFFERENT roundings of the
same mathematical sum. Unlike SAXPY, there is no single bit-exact answer to
match. The policy pinned here (and inherited by every future reducing kernel —
dot, L2 norm, mean, variance, softmax-denominator):

  * The reference computes the sum in **fp64** and treats that as truth.
  * A device fp32 result is accepted iff it agrees to a **relative** tolerance
    of 1e-3 (the project's standard fingerprint tolerance), with an absolute
    floor so a near-zero true sum (random cancellation) doesn't make the
    relative test explode: |dev - ref| <= 1e-3 * max(|ref|, 1.0).

`x` is drawn from [-1, 1), so E[sum]=0 and the sum is a random-walk magnitude
~sqrt(N/3); partial sums never grow large, so fp32 lane-accumulation stays well
inside 1e-3 relative at every sweep size.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Splitmix64 constants — identical to saxpy/cpu/reference.py and the C kernels
# (https://prng.di.unimi.it/splitmix64.c).
_SPLITMIX_INC = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_U64_MASK = 0xFFFFFFFFFFFFFFFF

REL_TOL = 1e-3


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


def generate_input(size: int, seed: int) -> np.ndarray:
    """The same fp32 vector `x` the device kernels see for (size, seed).

    First `size` draws are `x` — the same leading-draw convention SAXPY/matmul
    use. Each element is `2*unit_double - 1` in [-1, 1), cast to float32.
    """
    state = seed & _U64_MASK
    out = np.empty(size, dtype=np.float64)
    for i in range(size):
        state, val = splitmix64_next(state)
        out[i] = 2.0 * u64_to_unit_double(val) - 1.0
    return out.astype(np.float32, copy=False)


def reference_sum(size: int, seed: int) -> float:
    """Canonical fp64 'truth' sum of `x(size, seed)`.

    fp64 accumulation over fp32 inputs — the order-independent reference every
    backend's fp32 reduction is compared against.
    """
    x = generate_input(size, seed)
    return float(x.astype(np.float64).sum())


def validate(device_value: float, size: int, seed: int, rel_tol: float = REL_TOL) -> bool:
    """True iff a device fp32 reduction agrees with the fp64 truth (R6 policy).

    Combined relative/absolute test: |dev - ref| <= rel_tol * max(|ref|, 1.0).
    The absolute floor guards against a near-zero true sum from random
    cancellation.
    """
    ref = reference_sum(size, seed)
    return abs(device_value - ref) <= rel_tol * max(abs(ref), 1.0)


if __name__ == "__main__":
    import time

    for n in (1 << 16, 1 << 18):
        t0 = time.monotonic()
        s = reference_sum(n, 42)
        dt = (time.monotonic() - t0) * 1000
        print(f"reference_sum({n}, 42) = {s:.6e}  ({dt:.0f} ms)")
