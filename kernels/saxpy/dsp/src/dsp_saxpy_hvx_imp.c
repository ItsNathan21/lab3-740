// dsp_saxpy_hvx_imp.c — HVX-vectorized FP32 SAXPY, cDSP v79.
// out = alpha*x + y, 32 float32 per 128B HVX vector.
//
// Intrinsics (hvx_hexagon_protos.h, -mhvx -mhvx-length=128B -mv79):
//   Q6_Vqf32_vmpy_VsfRsf(vec, bits) — IEEE-float-vec × scalar-reg → QF32
//   Q6_Vqf32_vadd_VsfVsf(a, b)      — IEEE float + IEEE float     → QF32
//   Q6_Vsf_equals_Vqf32(qf)         — QF32 → IEEE float32
//
// open/close/scalar already provided by dsp_saxpy_imp.c — do NOT redeclare here.

#include "dsp_saxpy.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "qurt_hvx.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>

#define VFLOATS 32   // 128B HVX vector / 4 bytes per float

int dsp_saxpy_saxpy_hvx_fp32(remote_handle64 h,
                             const float* x, int xLen,
                             const float* y, int yLen,
                             float alpha,
                             float* out, int outLen,
                             int n,
                             int* dsp_us) {
    uint64_t t0 = HAP_perf_get_time_us();

    int n_vec = n / VFLOATS;      // full 32-wide strips
    int alpha_bits = *(int*)&alpha;

    qurt_hvx_lock(QURT_HVX_MODE_128B);

    for (int iv = 0; iv < n_vec; iv++) {
        int i = iv * VFLOATS;
        // Compiler emits VMEMU for potentially-unaligned FastRPC buffers.
        HVX_Vector vx = *((HVX_Vector*)(x + i));
        HVX_Vector vy = *((HVX_Vector*)(y + i));
        // alpha*x in QF32, convert to IEEE, then add y (IEEE+IEEE → QF32), store IEEE.
        HVX_Vector prod_sf = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfRsf(vx, alpha_bits));
        HVX_Vector sum_qf  = Q6_Vqf32_vadd_VsfVsf(prod_sf, vy);
        *((HVX_Vector*)(out + i)) = Q6_Vsf_equals_Vqf32(sum_qf);
    }

    qurt_hvx_unlock();

    // Scalar tail when n % 32 != 0.
    for (int i = n_vec * VFLOATS; i < n; i++) {
        out[i] = alpha * x[i] + y[i];
    }

    *dsp_us = (int)(HAP_perf_get_time_us() - t0);
    FARF(RUNTIME_HIGH, "dsp_saxpy HVX n=%d dsp_us=%d", n, *dsp_us);
    return 0;
}
