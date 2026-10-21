// dsp_conv_stage_imp.c — moving a large buffer through VTCM, in blocks.
//
// REFERENCE IMPLEMENTATION. Lab 3 Task 11 ships a scaffold
// (kernels/lab3/vtcm/student_vtcm.c) that replaces this file at build time with
// calc_staged() hollowed out. Both define dsp_conv_stage_calc.
//
// The working set is mb megabytes, generated here rather than marshalled over
// FastRPC — transporting 32 MB each way would swamp the thing being measured.
// It is processed in blocks that fit VTCM. P is the number of passes over each
// block: P=1 means no reuse at all, and larger P means more arithmetic per byte
// brought in, which is the knob Task 11.10 sweeps.
//
//   impl 0  calc_ddr           compute straight out of ordinary memory
//   impl 1  calc_staged_naive  memcpy each block into VTCM, then compute
//   impl 2  calc_staged        the student's version
//
// checksum is the correctness check: it must be identical across all three.
// copy_us and compute_us split dsp_us, so the split itself is a result
// (Task 11.6).

#include "dsp_conv.h"
#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "qurt_hvx.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define STAGE_BLOCK_BYTES (2 * 1024 * 1024)   /* 2 MB: four fit in 8 MB VTCM */

// The kernel applied to a block, P times. Deliberately cheap per element, so
// that at low P the cost is dominated by getting the bytes here at all.
//
// It reads through HVX rather than scalar loads, and that is not a detail: VTCM
// is a vector scratchpad, and a scalar walk over it is roughly 8x SLOWER than
// the same walk over cached DDR, because it gives up both the vector width and
// the L2 prefetcher without gaining anything. Measured on this board before the
// loop below was vectorised. Compare like with like or the staging comparison
// inverts and tells you the opposite of the truth.
#define STAGE_VF 32   /* 128-byte HVX vector / 4 bytes per float */

static uint32_t stage_kernel(const float *buf, int n, int P) {
    union { HVX_Vector v; float f[STAGE_VF]; } lanes __attribute__((aligned(128)));
    const HVX_Vector *vb = (const HVX_Vector *)buf;
    const int nvec = n / STAGE_VF;
    HVX_Vector vhalf;
    {   /* splat 0.5f across all 32 lanes */
        union { HVX_Vector v; float f[STAGE_VF]; } s __attribute__((aligned(128)));
        for (int i = 0; i < STAGE_VF; i++) s.f[i] = 0.5f;
        vhalf = s.v;
    }

    HVX_Vector vacc = Q6_V_vzero();
    for (int p = 0; p < P; p++) {
        for (int i = 0; i < nvec; i++) {
            HVX_Vector prod = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vb[i], vhalf));
            vacc = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(vacc, prod));
        }
    }
    lanes.v = vacc;
    float acc = 0.0f;
    for (int i = 0; i < STAGE_VF; i++) acc += lanes.f[i];
    /* scalar tail, if the block is not a whole number of vectors */
    for (int p = 0; p < P; p++)
        for (int i = nvec * STAGE_VF; i < n; i++) acc += buf[i] * 0.5f;
    return (uint32_t)acc;
}

// 128-byte vector copy. The core's own load path is not the widest way to move
// bytes on this part, which is the subject of Task 11.7 and 11.9.
static void hvx_copy(void *dst, const void *src, int bytes) {
    HVX_Vector *d = (HVX_Vector *)dst;
    const HVX_Vector *s = (const HVX_Vector *)src;
    int nvec = bytes / 128;
    for (int i = 0; i < nvec; i++) d[i] = s[i];
    int done = nvec * 128;
    if (done < bytes)
        memcpy((uint8_t *)dst + done, (const uint8_t *)src + done, bytes - done);
}

int dsp_conv_stage_calc(remote_handle64 h,
                        int mb, int P, int impl,
                        int *dsp_us, int *copy_us, int *compute_us,
                        int *checksum) {
    (void)h;
    if (mb <= 0 || mb > 128 || P <= 0) return -1;

    const long total_bytes = (long)mb * 1024 * 1024;
    const int  block_bytes = STAGE_BLOCK_BYTES;
    const int  block_n = block_bytes / (int)sizeof(float);
    const int  nblocks = (int)(total_bytes / block_bytes);
    if (nblocks < 1) return -1;

    float *src = (float *)memalign(128, (size_t)total_bytes);
    if (!src) return -1;
    // Deterministic fill, so the checksum is reproducible run to run.
    for (long i = 0; i < total_bytes / (long)sizeof(float); i++)
        src[i] = (float)((i * 1103515245u + 12345u) & 0xFFFF) * (1.0f / 65536.0f);

    // VTCM, if we are staging. Two blocks so Task 11.10 can double-buffer.
    unsigned int ctx_id = 0;
    float *vtcm = 0;
    if (impl != 0) {
        compute_res_attr_t attr;
        if (HAP_compute_res_attr_init(&attr) == 0 &&
            HAP_compute_res_attr_set_vtcm_param(&attr, (unsigned int)(2 * block_bytes), 1) == 0) {
            ctx_id = HAP_compute_res_acquire(&attr, 100000);
            if (ctx_id) vtcm = (float *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
        }
        if (!vtcm) {
            if (ctx_id) { HAP_compute_res_release(ctx_id); ctx_id = 0; }
            free(src);
            FARF(ERROR, "stage_calc: VTCM unavailable");
            return -2;
        }
    }

    uint64_t t_copy = 0, t_comp = 0, t0 = HAP_perf_get_time_us();
    uint32_t sum = 0;

    qurt_hvx_lock(QURT_HVX_MODE_128B);

    if (impl == 0) {
        /* calc_ddr — compute straight out of ordinary memory. */
        for (int b = 0; b < nblocks; b++) {
            uint64_t a = HAP_perf_get_time_us();
            sum += stage_kernel(src + (long)b * block_n, block_n, P);
            t_comp += HAP_perf_get_time_us() - a;
        }
    } else if (impl == 1) {
        /* calc_staged_naive — memcpy into VTCM, then compute there. */
        for (int b = 0; b < nblocks; b++) {
            uint64_t a = HAP_perf_get_time_us();
            memcpy(vtcm, src + (long)b * block_n, block_bytes);
            uint64_t m = HAP_perf_get_time_us();
            sum += stage_kernel(vtcm, block_n, P);
            uint64_t e = HAP_perf_get_time_us();
            t_copy += m - a;
            t_comp += e - m;
        }
    } else {
        /* calc_staged — reference answer: HVX copy, double-buffered so the
         * next block moves while the current one is being computed. */
        float *buf[2] = { vtcm, vtcm + block_n };
        uint64_t a = HAP_perf_get_time_us();
        hvx_copy(buf[0], src, block_bytes);
        t_copy += HAP_perf_get_time_us() - a;

        for (int b = 0; b < nblocks; b++) {
            int cur = b & 1, nxt = cur ^ 1;
            if (b + 1 < nblocks) {
                uint64_t c0 = HAP_perf_get_time_us();
                hvx_copy(buf[nxt], src + (long)(b + 1) * block_n, block_bytes);
                t_copy += HAP_perf_get_time_us() - c0;
            }
            uint64_t k0 = HAP_perf_get_time_us();
            sum += stage_kernel(buf[cur], block_n, P);
            t_comp += HAP_perf_get_time_us() - k0;
        }
    }

    qurt_hvx_unlock();

    uint64_t t1 = HAP_perf_get_time_us();
    *dsp_us     = (int)(t1 - t0);
    *copy_us    = (int)t_copy;
    *compute_us = (int)t_comp;
    *checksum   = (int)sum;

    if (ctx_id) HAP_compute_res_release(ctx_id);
    free(src);

    FARF(RUNTIME_HIGH, "stage_calc impl=%d mb=%d P=%d blocks=%d dsp_us=%d "
         "copy_us=%d compute_us=%d cksum=%d",
         impl, mb, P, nblocks, *dsp_us, *copy_us, *compute_us, *checksum);
    return 0;
}
