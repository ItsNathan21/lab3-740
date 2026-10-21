// dsp_reduction_hvx_imp.c — HVX-vectorized FP32 sum-reduction, cDSP v79.
// sum = sum(x_i), 32 float32 accumulated per 128B HVX vector into a lane-wise
// accumulator, then a horizontal fold of the 32 lanes at the very end.
//
// The 32 independent lane accumulators break the dependent add chain that
// bottlenecks the scalar baseline (same principle as the CPU NEON multi-
// accumulator variant) — this is why the DSP HVX reduction is expected to run
// far above the DSP scalar path, before the FastRPC input copy is layered on.
//
// Intrinsics (hvx_hexagon_protos.h, -mhvx -mhvx-length=128B -mv79):
//   Q6_Vqf32_vadd_VsfVsf(a, b) — IEEE float + IEEE float -> QF32
//   Q6_Vsf_equals_Vqf32(qf)    — QF32 -> IEEE float32
//   Q6_V_vzero()               — all-zero vector (= +0.0f in every sf lane)
//
// open/close/scalar already provided by dsp_reduction_imp.c — do NOT redeclare.

#include "dsp_reduction.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "qurt_hvx.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>

#define VFLOATS 32   // 128B HVX vector / 4 bytes per float

int dsp_reduction_reduce_hvx_fp32(remote_handle64 h,
                                  const float* x, int xLen,
                                  float* sum,
                                  int n,
                                  int* dsp_us) {
    uint64_t t0 = HAP_perf_get_time_us();

    int n_vec = n / VFLOATS;   // full 32-wide strips

    qurt_hvx_lock(QURT_HVX_MODE_128B);

    // Lane-wise accumulator: 32 independent fp32 partial sums.
    HVX_Vector acc = Q6_V_vzero();   // +0.0f in every lane
    for (int iv = 0; iv < n_vec; iv++) {
        int i = iv * VFLOATS;
        // Compiler emits VMEMU for potentially-unaligned FastRPC buffers.
        HVX_Vector vx = *((HVX_Vector*)(x + i));
        // acc += vx  (IEEE+IEEE -> QF32, converted back to IEEE sf each step).
        acc = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc, vx));
    }

    qurt_hvx_unlock();

    // Horizontal fold: store the 32 lane partials and sum them on the scalar
    // unit (done once, negligible vs the streaming loop).
    union { HVX_Vector v; float f[VFLOATS]; } lanes __attribute__((aligned(128)));
    lanes.v = acc;
    float s = 0.0f;
    for (int k = 0; k < VFLOATS; k++) s += lanes.f[k];

    // Scalar tail when n % 32 != 0.
    for (int i = n_vec * VFLOATS; i < n; i++) s += x[i];

    *sum = s;

    *dsp_us = (int)(HAP_perf_get_time_us() - t0);
    FARF(RUNTIME_HIGH, "dsp_reduction HVX n=%d dsp_us=%d", n, *dsp_us);
    return 0;
}
