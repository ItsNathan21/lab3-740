// student_vtcm.c - Lab 3, Task 11. DDR-to-VTCM staging.
//
// A buffer of `mb` megabytes has to be processed in blocks that fit VTCM.
// P is the number of passes over each block, so it is the arithmetic intensity
// knob: P=1 means no reuse at all.
//
//   impl 0  calc_ddr           GIVEN. Computes straight out of ordinary memory.
//   impl 1  calc_staged_naive  GIVEN. memcpy each block into VTCM, then compute.
//   impl 2  calc_staged        HVX copy, serial DMA, or double-buffered DMA.
//
// Select the experiment at build time using STAGE_COPY_MODE:
//   0 = HVX (11.6), 1 = serial DMA (11.7), 2 = overlapped DMA (11.8, default).
// For the cache experiment only, also set STAGE_DMA_FLUSH=0 when building.
//   STAGE_COPY_MODE=1 bash scripts/build_student.sh vtcm
//   python profiler.py stage dsp -p impl=staged -p p=1
//
// `checksum` must be identical for all three implementations at a given P. If
// yours differs you are skipping or double-counting a block.
//
// A warning that will cost you an afternoon otherwise: read through HVX, not
// scalar loads. VTCM is a vector scratchpad, and a scalar walk over it is about
// 8x SLOWER than the same walk over cached DDR - it gives up the vector width
// AND the L2 prefetcher and gains nothing. stage_kernel() below already does
// this correctly; the lesson is for the copy you are about to write.

#include "dsp_conv.h"
#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "qurt_hvx.h"
#include "qurt_memory.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define STAGE_BLOCK_BYTES (2 * 1024 * 1024)   /* 2 MB per block */

#ifndef STAGE_COPY_MODE
#define STAGE_COPY_MODE 2
#endif
#ifndef STAGE_DMA_FLUSH
#define STAGE_DMA_FLUSH 1
#endif
#if STAGE_COPY_MODE < 0 || STAGE_COPY_MODE > 2
#error "STAGE_COPY_MODE must be 0, 1, or 2"
#endif

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

#if STAGE_COPY_MODE == 0
static void my_copy(void *dst, const void *src, int bytes) {
    HVX_Vector *out = dst;
    const HVX_Vector *in = src;
    const int vectors = bytes / (int)sizeof(*out);
    for (int i = 0; i < vectors; ++i) out[i] = in[i];
    const int tail = bytes % (int)sizeof(*out);
    if (tail) memcpy(out + vectors, in + vectors, (size_t)tail);
}
#else
// One 16-byte Hexagon type-0 descriptor. This transfer is ordered and reads
// DDR directly (source-cache bypass), with VTCM as the destination.
typedef struct __attribute__((aligned(16))) {
    uint32_t next, control, src, dst;
} stage_dma_desc;

static int start_copy(stage_dma_desc *desc, void *dst, const void *src, int bytes) {
    if (STAGE_DMA_FLUSH &&
        qurt_mem_cache_clean((qurt_addr_t)src, (qurt_size_t)bytes,
                             QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE) != QURT_EOK)
        return -1;

    if ((Q6_R_dmpoll() & 3) != 0) return -1;
    *desc = (stage_dma_desc){0, (uint32_t)bytes | (1u << 29) | (1u << 30),
                             (uint32_t)(uintptr_t)src, (uint32_t)(uintptr_t)dst};
    __asm__ volatile("release(%0):at" :: "r"(desc) : "memory");
    Q6_dmstart_A(desc);
    return 0;
}

static int wait_for_copy(void) {
    const int status = Q6_R_dmwait() & 3;
    __asm__ volatile("" ::: "memory");
    return status == 0 ? 0 : -1;
}
#endif

int dsp_conv_stage_calc(remote_handle64 h,
                        int mb, int P, int impl,
                        int *dsp_us, int *copy_us, int *compute_us,
                        int *checksum) {
    (void)h;
    if (mb <= 0 || mb > 128 || P <= 0) return -1;

    const long total_bytes = (long)mb * 1024 * 1024;
    const int  block_bytes = STAGE_BLOCK_BYTES;
    const int  block_n = block_bytes / (int)sizeof(float);
    const int  nblocks = (int)((total_bytes + block_bytes - 1) / block_bytes);
    const int  last_bytes = (int)(total_bytes - (long)(nblocks - 1) * block_bytes);

    float *src = (float *)memalign(128, (size_t)total_bytes);
    if (!src) return -1;
    // Deterministic fill, so the checksum is reproducible run to run.
    for (long i = 0; i < total_bytes / (long)sizeof(float); i++)
        src[i] = (float)((i * 1103515245u + 12345u) & 0xFFFF) * (1.0f / 65536.0f);

    // The overlapped path reserves a second block for the pending DMA.
    unsigned int ctx_id = 0;
    float *vtcm = 0;
    if (impl != 0) {
        const unsigned int vtcm_bytes = block_bytes * (impl == 2 && STAGE_COPY_MODE == 2 ? 2 : 1);
        compute_res_attr_t attr;
        if (HAP_compute_res_attr_init(&attr) == 0 &&
            HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_bytes, 1) == 0) {
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
    int rc = 0;

    qurt_hvx_lock(QURT_HVX_MODE_128B);

#if STAGE_COPY_MODE == 2
    if (impl == 2) {
        float *buf[2] = { vtcm, vtcm + block_n };
        stage_dma_desc desc;
        const int first_bytes = nblocks == 1 ? last_bytes : block_bytes;
        uint64_t started = HAP_perf_get_time_us();
        rc = start_copy(&desc, buf[0], src, first_bytes);
        if (rc == 0) rc = wait_for_copy();
        t_copy += HAP_perf_get_time_us() - started;

        for (int b = 0; b < nblocks && rc == 0; ++b) {
            const int current = b % 2;
            const int has_next = b + 1 < nblocks;
            if (has_next) {
                const int next_bytes = b + 2 == nblocks ? last_bytes : block_bytes;
                started = HAP_perf_get_time_us();
                rc = start_copy(&desc, buf[1 - current], src + (long)(b + 1) * block_n,
                                next_bytes);
                t_copy += HAP_perf_get_time_us() - started;
                if (rc != 0) break;
            }

            const int bytes = has_next ? block_bytes : last_bytes;
            started = HAP_perf_get_time_us();
            sum += stage_kernel(buf[current], bytes / (int)sizeof(float), P);
            t_comp += HAP_perf_get_time_us() - started;

            // Count only the exposed DMA time in copy_us. The rest overlaps.
            if (has_next) {
                started = HAP_perf_get_time_us();
                rc = wait_for_copy();
                t_copy += HAP_perf_get_time_us() - started;
            }
        }
    } else
#endif
    {
        for (int b = 0; b < nblocks; b++) {
            const int bytes = b + 1 == nblocks ? last_bytes : block_bytes;
            const float *block = src + (long)b * block_n;
            if (impl != 0) {
                const uint64_t started = HAP_perf_get_time_us();
                if (impl == 1) memcpy(vtcm, block, (size_t)bytes);
#if STAGE_COPY_MODE == 0
                else my_copy(vtcm, block, bytes);
#elif STAGE_COPY_MODE == 1
                else {
                    stage_dma_desc desc;
                    rc = start_copy(&desc, vtcm, block, bytes);
                    if (rc == 0) rc = wait_for_copy();
                    if (rc != 0) break;
                }
#endif
                block = vtcm;
                t_copy += HAP_perf_get_time_us() - started;
            }

            const uint64_t started = HAP_perf_get_time_us();
            sum += stage_kernel(block, bytes / (int)sizeof(float), P);
            t_comp += HAP_perf_get_time_us() - started;
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
    if (rc != 0) {
        FARF(ERROR, "stage_calc: DMA transfer failed");
        return -3;
    }

    FARF(RUNTIME_HIGH, "stage_calc impl=%d mb=%d P=%d blocks=%d dsp_us=%d "
         "copy_us=%d compute_us=%d cksum=%d",
         impl, mb, P, nblocks, *dsp_us, *copy_us, *compute_us, *checksum);
    return 0;
}
