"""NumPy reference for direct 2D convolution (roofline, 4th workload / mid-AI point).

Convolution is `out = conv2d(in, W)`, dense, K×K, stride 1, "same" (zero) padding,
NHWC layout. It is the **middle** of the roofline: its arithmetic intensity is set
by the channel count (`AI ≈ (K²/8)·C` for Cin=Cout=C), so the default Cin=Cout=8,
K=3 lands at AI≈18 — between the memory-bound anchors (SAXPY 0.17, reduction 0.25)
and compute-bound matmul (85). See `kernels/convolution/DESIGN.md` (decision C1).

Layout conventions (decision C4), fixed here so all four backends + this reference
agree bit-for-bit on element order:
  * input   NHWC (single batch): in[h, w, ic]     flat = (h*W + w)*Cin + ic
  * weights OHWI:                W[oc, kh, kw, ic] flat = ((oc*K + kh)*K + kw)*Cin + ic
  * output  NHWC:                out[h, w, oc]     flat = (h*W + w)*Cout + oc
No bias in v1.

splitmix64 draw order (decision C6): the input's H·W·Cin values first (HWC flat
order), then the weights' Cout·K²·Cin values (OHWI flat order), one advancing
state — bit-identical to the C/OpenCL/HVX kernels. Each value is 2·unit−1 in
[-1,1), cast to float32.

Reduction-order tolerance (decision C6 / reused R6): convolution accumulates
K²·Cin products per output; fp32 vs fp64 orders differ, so the fp64 result is
truth and a device fp32 result passes iff |dev−ref| ≤ 1e-3·max(|ref|,1).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_SPLITMIX_INC = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_U64_MASK = 0xFFFFFFFFFFFFFFFF

REL_TOL = 1e-3


def _splitmix_stream(n: int, seed: int) -> np.ndarray:
    """n splitmix64 outputs mapped to fp32 in [-1,1).

    splitmix64 state after the i-th call is seed + i·INC (mod 2^64), so the whole
    stream is vectorizable (closed-form state) — matches the sequential C loop
    bit-for-bit.
    """
    i = np.arange(1, n + 1, dtype=object)
    z = (seed + i * _SPLITMIX_INC) & _U64_MASK
    z = ((z ^ (z >> 30)) * _SPLITMIX_M1) & _U64_MASK
    z = ((z ^ (z >> 27)) * _SPLITMIX_M2) & _U64_MASK
    z = z ^ (z >> 31)
    u = (z >> 11).astype(np.float64) * (1.0 / (1 << 53))
    return (2.0 * u - 1.0).astype(np.float32)


def generate_inputs(H: int, W: int, Cin: int, Cout: int, K: int, seed: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """(input NHWC [H,W,Cin], weights OHWI [Cout,K,K,Cin]) fp32 for (dims, seed)."""
    n_in = H * W * Cin
    n_w = Cout * K * K * Cin
    stream = _splitmix_stream(n_in + n_w, seed)
    x = stream[:n_in].reshape(H, W, Cin)
    w = stream[n_in:].reshape(Cout, K, K, Cin)
    return x, w


def conv2d_ref(H: int, W: int, Cin: int, Cout: int, K: int, seed: int) -> np.ndarray:
    """fp64 'truth' direct convolution, stride 1, same (zero) padding, NHWC/OHWI.

    Accumulates in fp64 — the order-independent reference the device fp32 kernels
    are compared against.
    """
    x, w = generate_inputs(H, W, Cin, Cout, K, seed)
    x = x.astype(np.float64)
    w = w.astype(np.float64)
    pad = K // 2
    xp = np.zeros((H + 2 * pad, W + 2 * pad, Cin), dtype=np.float64)
    xp[pad:pad + H, pad:pad + W, :] = x
    out = np.zeros((H, W, Cout), dtype=np.float64)
    for kh in range(K):
        for kw in range(K):
            # shifted input window [H,W,Cin] for this (kh,kw) tap
            win = xp[kh:kh + H, kw:kw + W, :]
            # out[h,w,oc] += sum_ic win[h,w,ic] * w[oc,kh,kw,ic]
            out += np.einsum("hwi,oi->hwo", win, w[:, kh, kw, :], optimize=True)
    return out


def fingerprint(out: np.ndarray) -> float:
    """sum(out) + L2_norm(out), fp64 — matches the device kernels."""
    o = out.astype(np.float64)
    return float(o.sum()) + float(np.linalg.norm(o.reshape(-1)))


def reference_fingerprint(H: int, W: int, Cin: int, Cout: int, K: int, seed: int) -> float:
    return fingerprint(conv2d_ref(H, W, Cin, Cout, K, seed))


def validate(device_value: float, H: int, W: int, Cin: int, Cout: int, K: int,
             seed: int, rel_tol: float = REL_TOL) -> bool:
    ref = reference_fingerprint(H, W, Cin, Cout, K, seed)
    return abs(device_value - ref) <= rel_tol * max(abs(ref), 1.0)


def arithmetic_intensity(H: int, W: int, Cin: int, Cout: int, K: int) -> float:
    """FLOP / bytes (fp32), standard convention (full MACs, ignores border)."""
    flop = 2.0 * H * W * Cout * K * K * Cin
    bytes_ = 4.0 * (H * W * Cin + Cout * K * K * Cin + H * W * Cout)
    return flop / bytes_


if __name__ == "__main__":
    import time
    for C in (8, 16):
        t0 = time.monotonic()
        fp = reference_fingerprint(256, 256, C, C, 3, 42)
        ai = arithmetic_intensity(256, 256, C, C, 3)
        dt = (time.monotonic() - t0) * 1000
        print(f"conv 256x256 Cin=Cout={C} K3: AI={ai:.1f}  fp={fp:.6e}  ({dt:.0f} ms)")
