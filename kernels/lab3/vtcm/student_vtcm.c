// student_vtcm.c - Lab 3, Task 11. Your file. Edit it.
//
// A buffer of `mb` megabytes has to be processed in blocks that fit VTCM.
// P is the number of passes over each block, so it is the arithmetic intensity
// knob: P=1 means no reuse at all.
//
//   impl 0  calc_ddr           GIVEN. Computes straight out of ordinary memory.
//   impl 1  calc_staged_naive  GIVEN. memcpy each block into VTCM, then compute.
//   impl 2  calc_staged        YOURS. Beat calc_ddr, using no more VTCM than
//                              you are given.
//
// Two things to fix, in this order (Task 11 Part B):
//   1. my_copy() is memcpy, and memcpy is the bottleneck. Measure it first,
//      then replace it with an HVX vector copy.
//   2. calc_staged() copies a block, computes it, copies the next.
//      Split VTCM in two and overlap those.
//
//   bash scripts/build_student.sh vtcm                  # builds and pushes
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
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define STAGE_BLOCK_BYTES (2 * 1024 * 1024)   /* 2 MB per block */

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

// ===========================================================================
// TASK 11.7 - my_copy(): move `bytes` from DDR into VTCM.
//
// It is memcpy right now, and Task 11.6 asks you to turn its measured rate into
// a bandwidth figure and compare that against what you measured in Task 11.2
// and 11.4. The gap is the whole point.
//
// Replace it with a copy through 128-byte HVX vectors. HVX_Vector is a 128-byte
// type: a load and a store of one moves 128 bytes. Handle a tail that is not a
// whole number of vectors.
// ===========================================================================
static void my_copy(void *dst, const void *src, int bytes) {
    // ---- TODO(Task 11.7): replace this with an HVX vector copy -------------
    memcpy(dst, src, (size_t)bytes);
    // ------------------------------------------------------------------------
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
        /* ===================================================================
         * TASK 11.10 - calc_staged(): your version.
         *
         * What is here now is the simplest correct thing: copy a block in,
         * compute it, copy the next one in, compute that. Nothing overlaps, so
         * the DSP is idle during every copy and the copy path is idle during
         * every compute.
         *
         * You have room for TWO blocks in VTCM - buf[0] and buf[1] below. Start
         * the next block's copy before computing the current one, and alternate.
         * The timings you report should show copy_us falling relative to
         * dsp_us as the copies hide behind compute.
         * =================================================================== */
        float *buf[2] = { vtcm, vtcm + block_n };
        (void)buf;

        for (int b = 0; b < nblocks; b++) {
            uint64_t c0 = HAP_perf_get_time_us();
            my_copy(buf[0], src + (long)b * block_n, block_bytes);
            uint64_t c1 = HAP_perf_get_time_us();
            sum += stage_kernel(buf[0], block_n, P);
            uint64_t c2 = HAP_perf_get_time_us();
            t_copy += c1 - c0;
            t_comp += c2 - c1;
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
