// dsp_matmul_vtcm_imp.c — HVX FP32 matmul with B in VTCM scratchpad.
// Phase 6 Step 6.
//
// Improvement over HVX variant (Step 5): B matrix is copied into VTCM before
// the compute loop. VTCM provides 128B/cycle zero-miss reads vs the L2 cache
// misses caused by B's 2048-byte stride access pattern (N=512, row-major).
//
// With B in VTCM and N=512 (j always a 128B-aligned multiple):
//   - VTCM pointer is 128B-aligned (HAP_compute_res guarantees this)
//   - b_vtcm + k*512 + j: k*2048 (128B multiple) + j*4 (128B multiple) → aligned
//   - Compiler emits VMEM (aligned) instead of VMEMU (unaligned)
//   - VTCM throughput: 128B/cycle (1 HVX vector/cycle, no cache miss)
//
// open/close provided by dsp_matmul_imp.c.

#include "dsp_matmul.h"
#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "qurt_hvx.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <string.h>

#define VFLOATS 32

int dsp_matmul_matmul_hvx_vtcm_fp32(remote_handle64 h,
                                     const float* a_data, int a_dataLen,
                                     const float* b_data, int b_dataLen,
                                     float* c_data,       int c_dataLen,
                                     int n, int* dsp_us) {
    uint64_t t0 = HAP_perf_get_time_us();

    int b_bytes = n * n * (int)sizeof(float);
    const float* b_work = b_data;  // default: read from regular memory

    // --- Acquire VTCM and copy B into it ---
    compute_res_attr_t res_attr;
    unsigned int ctx_id = 0;
    if (HAP_compute_res_attr_init(&res_attr) == 0) {
        if (HAP_compute_res_attr_set_vtcm_param(&res_attr,
                                                (unsigned int)b_bytes, 1) == 0) {
            ctx_id = HAP_compute_res_acquire(&res_attr, 100000); // 100 ms timeout
        }
    }
    float* b_vtcm = (ctx_id != 0)
                    ? (float*)HAP_compute_res_attr_get_vtcm_ptr(&res_attr)
                    : 0;
    if (b_vtcm) {
        memcpy(b_vtcm, b_data, (unsigned int)b_bytes);
        // Tell the compiler b_vtcm is 128B-aligned so it can emit VMEM, not VMEMU.
        b_work = (const float*)__builtin_assume_aligned(b_vtcm, 128);
    } else if (ctx_id != 0) {
        HAP_compute_res_release(ctx_id);
        ctx_id = 0;
    }

    uint64_t t_setup = HAP_perf_get_time_us();

    // --- HVX compute loop (same structure as hvx variant) ---
    int n_vec = n / VFLOATS;

    qurt_hvx_lock(QURT_HVX_MODE_128B);

    for (int i = 0; i < n; i++) {
        const float* a_row = a_data + i * n;

        for (int jv = 0; jv < n_vec; jv++) {
            int j = jv * VFLOATS;

            HVX_Vector b0 = *((HVX_Vector*)(b_work + j));
            HVX_Vector acc_qf = Q6_Vqf32_vmpy_VsfRsf(b0, *(int*)&a_row[0]);

            for (int k = 1; k < n; k++) {
                HVX_Vector bk = *((HVX_Vector*)(b_work + k * n + j));
                acc_qf = Q6_Vqf32_vadd_Vqf32Vqf32(
                             acc_qf,
                             Q6_Vqf32_vmpy_VsfRsf(bk, *(int*)&a_row[k]));
            }

            *((HVX_Vector*)(c_data + i * n + j)) = Q6_Vsf_equals_Vqf32(acc_qf);
        }

        // Scalar tail for N % 32 != 0 (not triggered for N=512).
        for (int j = n_vec * VFLOATS; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < n; k++)
                acc += a_row[k] * b_work[k * n + j];
            c_data[i * n + j] = acc;
        }
    }

    qurt_hvx_unlock();

    // --- Release VTCM ---
    if (ctx_id != 0)
        HAP_compute_res_release(ctx_id);

    uint64_t t_done = HAP_perf_get_time_us();
    *dsp_us = (int)(t_done - t0);
    FARF(RUNTIME_HIGH, "dsp_matmul HVX+VTCM n=%d setup_us=%u compute_us=%u total=%d vtcm=%s",
         n,
         (unsigned int)(t_setup - t0),
         (unsigned int)(t_done - t_setup),
         *dsp_us,
         b_vtcm ? "yes" : "no");
    return 0;
}
