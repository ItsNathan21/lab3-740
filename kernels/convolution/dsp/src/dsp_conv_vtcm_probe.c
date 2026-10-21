// dsp_conv_vtcm_probe.c — VTCM characterization probe, cDSP (v81 on the HDK8850).
//
// Measures how much VTCM a single
// process can actually acquire, whether two acquires can share it, and read/write
// bandwidth for VTCM against ordinary (malloc'd) memory.
//
// NOTE ON THE API. Hexagon SDK 6.6 has **no declaration** of HAP_request_VTCM /
// HAP_release_VTCM anywhere in incs/ — the symbol survives only inside prebuilt
// simulator binaries. Last year's TCM.c cannot compile against this SDK. The
// supported path is HAP_compute_res_attr_* + HAP_compute_res_acquire, which is
// what this probe (and the Task 11 starter) use.
//
// Bandwidth is measured with HVX vector accesses, which is what VTCM exists to
// feed. Reads accumulate into a vector and the result is returned, so the loads
// cannot be optimised away.

#include "dsp_conv.h"
#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "HAP_vtcm_mgr.h"
#include "qurt_hvx.h"
#include "qurt_thread.h"
#include "qurt.h"
#include "dma_utils.h"
#include "qhdsp_hvx.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define VB 128   /* bytes per HVX vector */

/* Write `bytes` of vector stores across buf, `iters` times. Nothing guards this
 * loop and nothing needs to: buf came from outside this function and is used
 * after it, so the compiler cannot prove the stores are dead. */
static void bw_write(void *buf, int bytes, int iters) {
    HVX_Vector v = Q6_V_vsplat_R(0x5A5A5A5A);
    int nvec = bytes / VB;
    for (int it = 0; it < iters; it++) {
        HVX_Vector *p = (HVX_Vector *)buf;
        for (int i = 0; i < nvec; i++) p[i] = v;
    }
}

/* Read `bytes` of vector loads across buf, `iters` times. Returns a checksum so
 * the compiler cannot delete the loop. */
/* FOUR independent accumulators, not one. A single accumulator makes every add
 * depend on the previous one, and that dependency chain, not memory, becomes
 * the limit (the same trap as a naive reduction). */
static int bw_read(const void *buf, int bytes, int iters) {
    HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
    HVX_Vector a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
    int nvec = bytes / VB;
    int n4 = nvec & ~3;
    for (int it = 0; it < iters; it++) {
        const HVX_Vector *p = (const HVX_Vector *)buf;
        for (int i = 0; i < n4; i += 4) {
            a0 = Q6_Vw_vadd_VwVw(a0, p[i + 0]);
            a1 = Q6_Vw_vadd_VwVw(a1, p[i + 1]);
            a2 = Q6_Vw_vadd_VwVw(a2, p[i + 2]);
            a3 = Q6_Vw_vadd_VwVw(a3, p[i + 3]);
        }
        for (int i = n4; i < nvec; i++) a0 = Q6_Vw_vadd_VwVw(a0, p[i]);
    }
    HVX_Vector acc = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a0, a1),
                                     Q6_Vw_vadd_VwVw(a2, a3));
    union { HVX_Vector v; int w[32]; } u __attribute__((aligned(128)));
    u.v = acc;
    int s = 0;
    for (int i = 0; i < 32; i++) s += u.w[i];
    return s;
}

/* ---- Task 5 reference implementations -------------------------------------
 * A working set larger than VTCM, processed with `passes` passes over each
 * block. Staging only pays when there is reuse: with one pass you have paid the
 * DDR read anyway and the copy is pure overhead.
 *   mode 10 : DDR only      — passes over the whole set, in place
 *   mode 11 : naive staged  — copy block to VTCM, passes, next block (serial)
 *   mode 12 : overlapped    — two half-buffers; copy next while computing current
 * size_bytes = total working set; iters = passes per block.
 */
#define T5_VTCM_BUDGET (8 * 1024 * 1024)

/* noinline + a memory clobber at every call site: t5_sum resets its accumulator
 * internally, so P calls with identical arguments are a pure function the
 * compiler may legally evaluate once and multiply. Without this the P column
 * measures one pass, not P. */
__attribute__((noinline))
static int t5_sum(const void *buf, int bytes) {
    HVX_Vector acc = Q6_V_vzero();
    const HVX_Vector *p = (const HVX_Vector *)buf;
    int nvec = bytes / VB;
    for (int i = 0; i < nvec; i++) acc = Q6_Vw_vadd_VwVw(acc, p[i]);
    union { HVX_Vector v; int w[32]; } u __attribute__((aligned(128)));
    u.v = acc;
    int s = 0;
    for (int i = 0; i < 32; i++) s += u.w[i];
    return s;
}

/* Copy strategies, DDR -> VTCM. Which one you pick decides whether staging is
 * worth doing at all:  memcpy ~1 GB/s, HVX vector ~12 GB/s, user DMA = ?
 *   0 = memcpy   1 = HVX vector   2 = user DMA engine                        */
static void t5_copy(void *dst, const void *src, int bytes, int how) {
    if (how == 0) { memcpy(dst, src, (size_t)bytes); return; }

    if (how == 2) {
        /* The DMA engine reads DDR directly, so the source must be flushed out
         * of the cache first or it may transfer stale data. */
        qurt_mem_cache_clean((qurt_addr_t)src, (qurt_size_t)bytes,
                             QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        void *desc = 0;
        if (posix_memalign(&desc, 16, DMA_DESC_SIZE_1D) == 0 && desc) {
            memset(desc, 0, DMA_DESC_SIZE_1D);
            dma_desc_1d_params_t p;
            memset(&p, 0, sizeof(p));
            p.src_address = (uint32_t)(uintptr_t)src;
            p.dst_address = (uint32_t)(uintptr_t)dst;
            p.length      = (uint32_t)bytes;
            p.order       = 1;
            p.src_bypass  = 1;   /* DDR source: bypass cache */
            p.dst_bypass  = 0;   /* VTCM destination */
            if (dma_desc_init(desc, &p, DMA_DESC_TYPE_1D) == DMA_SUCCESS &&
                dma_desc_submit(&desc, 1) == DMA_SUCCESS) {
                dma_wait_for_idle();
                free(desc);
                return;
            }
            free(desc);
        }
        memcpy(dst, src, (size_t)bytes);   /* fallback if DMA setup failed */
        return;
    }

    const HVX_Vector *s = (const HVX_Vector *)src;
    HVX_Vector *d = (HVX_Vector *)dst;
    int nv = bytes / VB;
    for (int i = 0; i < nv; i++) d[i] = s[i];
    int tail = nv * VB;
    if (tail < bytes) memcpy((unsigned char *)dst + tail,
                             (const unsigned char *)src + tail, (size_t)(bytes - tail));
}

typedef struct {
    const unsigned char *src;
    unsigned char *dst;
    int bytes;
    int use_hvx;
    volatile int go, done;
} t5_copy_job_t;

static void t5_copier(void *arg) {
    t5_copy_job_t *j = (t5_copy_job_t *)arg;
    for (;;) {
        while (!j->go) { /* spin */ }
        if (j->bytes < 0) break;
        t5_copy(j->dst, j->src, j->bytes, j->use_hvx);
        j->go = 0;
        j->done = 1;
    }
    qurt_thread_exit(0);
}

/* ---- Task 6: morphology, hand-written vs Qualcomm's QHL library ------------
 * Square 8-bit image, width a multiple of 128 (the library requires it).
 *   mode 20/21 : erode  KxK — scalar reference / qhdsp_hvx
 *   mode 22/23 : dilate KxK — scalar reference / qhdsp_hvx
 * size_bytes = image width (= height); iters = K (3, 5 or 7).
 * Only the interior is checksummed: the library does not document its border
 * convention, so comparing edges would test the wrong thing.
 */
static int t6_interior_sum(const uint8_t *img, int w, int pad) {
    int s = 0;
    for (int y = pad; y < w - pad; y++)
        for (int x = pad; x < w - pad; x++)
            s += (int)img[(long)y * w + x] * ((x & 7) + 1);
    return s;
}

static void t6_morph_scalar(const uint8_t *in, uint8_t *out, int w, int K, int is_dilate) {
    int pad = K / 2;
    for (int y = pad; y < w - pad; y++) {
        for (int x = pad; x < w - pad; x++) {
            uint8_t acc = is_dilate ? 0 : 255;
            for (int ky = -pad; ky <= pad; ky++) {
                const uint8_t *row = in + (long)(y + ky) * w + x;
                for (int kx = -pad; kx <= pad; kx++) {
                    uint8_t v = row[kx];
                    if (is_dilate) { if (v > acc) acc = v; }
                    else           { if (v < acc) acc = v; }
                }
            }
            out[(long)y * w + x] = acc;
        }
    }
}

int dsp_conv_vtcm_probe(remote_handle64 h, int mode, int size_bytes, int iters,
                        int *us, int *ok, int *a, int *b) {
    *us = 0; *ok = 0; *a = 0; *b = 0;

    if (mode >= 20 && mode <= 23) {
        int w = size_bytes, K = iters, pad = iters / 2;
        int is_dilate = (mode == 22 || mode == 23);
        int use_qhl   = (mode == 21 || mode == 23);
        uint8_t *in = 0, *out = 0;
        if (posix_memalign((void **)&in, 128, (size_t)w * w) ||
            posix_memalign((void **)&out, 128, (size_t)w * w)) { *ok = 0; return 0; }
        /* deterministic salt-and-pepper-ish content */
        for (long i = 0; i < (long)w * w; i++) {
            unsigned v = (unsigned)(i * 2654435761u);
            in[i] = (uint8_t)((v >> 24) ^ (v >> 11));
        }
        memset(out, 0, (size_t)w * w);

        qurt_hvx_lock(QURT_HVX_MODE_128B);
        uint64_t t0 = HAP_perf_get_time_us();
        if (!use_qhl) {
            t6_morph_scalar(in, out, w, K, is_dilate);
        } else if (is_dilate) {
            if (K == 3)      qhdsp_hvx_dilate3x3_ab(in, w, w, w, out, w);
            else if (K == 5) qhdsp_hvx_dilate5x5_ab(in, w, w, w, out, w);
            else             qhdsp_hvx_dilate7x7_ab(in, w, w, w, out, w);
        } else {
            if (K == 3)      qhdsp_hvx_erode3x3_ab(in, w, w, w, out, w);
            else if (K == 5) qhdsp_hvx_erode5x5_ab(in, w, w, w, out, w);
            else             qhdsp_hvx_erode7x7_ab(in, w, w, w, out, w);
        }
        *us = (int)(HAP_perf_get_time_us() - t0);
        qurt_hvx_unlock();

        *a = t6_interior_sum(out, w, pad + 1);   /* +1 slack on the border */
        *b = w;
        *ok = 1;
        free(in); free(out);
        return 0;
    }

    /* 11/12 stage with memcpy; 15/16 HVX copy; 17/18 user DMA. */
    if (mode >= 10 && mode <= 19) {
        int use_hvx_copy = 0;
        if (mode == 15 || mode == 16) use_hvx_copy = 1;
        if (mode == 17 || mode == 18) use_hvx_copy = 2;
        if (mode == 15 || mode == 17) mode = 11;
        if (mode == 16 || mode == 18) mode = 12;
        int total  = size_bytes;
        int passes = iters < 1 ? 1 : iters;
        unsigned char *ddr = (unsigned char *)memalign(128, (size_t)total);
        if (!ddr) { *ok = 0; return 0; }
        for (int i = 0; i < total; i++) ddr[i] = (unsigned char)(i * 31u);

        int budget = T5_VTCM_BUDGET;
        if (budget > total) budget = total;
        int checksum = 0;

        if (mode == 13 || mode == 14 || mode == 19) { /* ---- copy cost alone ---- */
            int budget2 = T5_VTCM_BUDGET;
            if (budget2 > total) budget2 = total;
            compute_res_attr_t at2;
            unsigned int id2 = 0;
            if (HAP_compute_res_attr_init(&at2) == 0 &&
                HAP_compute_res_attr_set_vtcm_param(&at2, (unsigned int)budget2, 1) == 0)
                id2 = HAP_compute_res_acquire(&at2, 100000);
            unsigned char *vt2 = id2 ? (unsigned char *)HAP_compute_res_attr_get_vtcm_ptr(&at2) : 0;
            if (!vt2) { if (id2) HAP_compute_res_release(id2); free(ddr); *ok = 0; return 0; }

            qurt_hvx_lock(QURT_HVX_MODE_128B);
            uint64_t t0 = HAP_perf_get_time_us();
            for (int off = 0; off < total; off += budget2) {
                int n = (total - off) < budget2 ? (total - off) : budget2;
                t5_copy(vt2, ddr + off, n, mode == 13 ? 0 : (mode == 14 ? 1 : 2));
            }
            *us = (int)(HAP_perf_get_time_us() - t0);
            qurt_hvx_unlock();
            HAP_compute_res_release(id2);
            *a = 0; *b = total; *ok = 1;
            free(ddr);
            return 0;
        }

        if (mode == 10) {                       /* ---- DDR only ---- */
            qurt_hvx_lock(QURT_HVX_MODE_128B);
            uint64_t t0 = HAP_perf_get_time_us();
            for (int p = 0; p < passes; p++) {
                checksum += t5_sum(ddr, total);
                __asm__ __volatile__("" ::: "memory");
            }
            *us = (int)(HAP_perf_get_time_us() - t0);
            qurt_hvx_unlock();
            *a = checksum; *b = 0; *ok = 1;
            free(ddr);
            return 0;
        }

        compute_res_attr_t at;
        unsigned int id = 0;
        if (HAP_compute_res_attr_init(&at) == 0 &&
            HAP_compute_res_attr_set_vtcm_param(&at, (unsigned int)budget, 1) == 0)
            id = HAP_compute_res_acquire(&at, 100000);
        unsigned char *vt = id ? (unsigned char *)HAP_compute_res_attr_get_vtcm_ptr(&at) : 0;
        if (!vt) { if (id) HAP_compute_res_release(id); free(ddr); *ok = 0; return 0; }

        if (mode == 11) {                       /* ---- naive staged ---- */
            qurt_hvx_lock(QURT_HVX_MODE_128B);
            uint64_t t0 = HAP_perf_get_time_us();
            for (int off = 0; off < total; off += budget) {
                int n = (total - off) < budget ? (total - off) : budget;
                t5_copy(vt, ddr + off, n, use_hvx_copy);
                for (int p = 0; p < passes; p++) {
                    checksum += t5_sum(vt, n);
                    __asm__ __volatile__("" ::: "memory");
                }
            }
            *us = (int)(HAP_perf_get_time_us() - t0);
            qurt_hvx_unlock();
        } else {                                /* ---- overlapped ---- */
            int half = budget / 2;
            unsigned char *bufs[2] = { vt, vt + half };
            t5_copy_job_t job = { 0, 0, 0, 0, 0 };
            qurt_thread_t tid;
            void *stack = memalign(128, 64 * 1024);
            qurt_thread_attr_t attr;
            qurt_thread_attr_init(&attr);
            qurt_thread_attr_set_name(&attr, "t5_copy");
            qurt_thread_attr_set_stack_addr(&attr, stack);
            qurt_thread_attr_set_stack_size(&attr, 64 * 1024);
            qurt_thread_attr_set_priority(&attr,
                (unsigned short)qurt_thread_get_priority(qurt_thread_get_id()));
            int spawned = (qurt_thread_create(&tid, &attr, t5_copier, &job) == QURT_EOK);

            qurt_hvx_lock(QURT_HVX_MODE_128B);
            uint64_t t0 = HAP_perf_get_time_us();
            int cur = 0, off = 0;
            int n0 = (total - off) < half ? (total - off) : half;
            t5_copy(bufs[cur], ddr + off, n0, use_hvx_copy);   /* prime */
            int cur_n = n0;
            off += n0;
            while (cur_n > 0) {
                int nxt = (total - off) < half ? (total - off) : half;
                if (nxt > 0 && spawned) {                   /* start next copy */
                    job.src = ddr + off; job.dst = bufs[cur ^ 1];
                    job.bytes = nxt; job.use_hvx = use_hvx_copy; job.done = 0; job.go = 1;
                }
                for (int p = 0; p < passes; p++) {
                    checksum += t5_sum(bufs[cur], cur_n);
                    __asm__ __volatile__("" ::: "memory");
                }
                if (nxt > 0) {
                    if (spawned) { while (!job.done) { /* wait */ } }
                    else t5_copy(bufs[cur ^ 1], ddr + off, nxt, use_hvx_copy);
                    off += nxt;
                }
                cur ^= 1; cur_n = nxt;
            }
            *us = (int)(HAP_perf_get_time_us() - t0);
            qurt_hvx_unlock();

            if (spawned) { job.bytes = -1; job.go = 1; int st; qurt_thread_join(tid, &st); }
            if (stack) free(stack);
        }

        HAP_compute_res_release(id);
        *a = checksum; *b = budget; *ok = 1;
        free(ddr);
        return 0;
    }

    /* ---- mode 0: what does the device claim? ---- */
    if (mode == 0) {
        unsigned int page_size = 0, page_count = 0;
        int rc = HAP_query_total_VTCM(&page_size, &page_count);
        *a = (int)page_size;
        *b = (int)page_count;
        *ok = (rc == 0);
        return 0;
    }

    /* ---- modes 1,2: can we actually acquire it? ---- */
    if (mode == 1 || mode == 2) {
        compute_res_attr_t at1, at2;
        unsigned int id1 = 0, id2 = 0;
        uint64_t t0 = HAP_perf_get_time_us();

        if (HAP_compute_res_attr_init(&at1) == 0 &&
            HAP_compute_res_attr_set_vtcm_param(&at1, (unsigned int)size_bytes, 1) == 0)
            id1 = HAP_compute_res_acquire(&at1, 100000);
        *a = (id1 != 0);

        if (mode == 2) {
            if (HAP_compute_res_attr_init(&at2) == 0 &&
                HAP_compute_res_attr_set_vtcm_param(&at2, (unsigned int)size_bytes, 1) == 0)
                id2 = HAP_compute_res_acquire(&at2, 100000);
            *b = (id2 != 0);
        }

        *us = (int)(HAP_perf_get_time_us() - t0);
        *ok = (mode == 1) ? (id1 != 0) : (id1 != 0 && id2 != 0);

        if (id2) HAP_compute_res_release(id2);
        if (id1) HAP_compute_res_release(id1);
        return 0;
    }

    /* ---- modes 3,4: VTCM bandwidth ---- */
    if (mode == 3 || mode == 4) {
        compute_res_attr_t at;
        unsigned int id = 0;
        if (HAP_compute_res_attr_init(&at) == 0 &&
            HAP_compute_res_attr_set_vtcm_param(&at, (unsigned int)size_bytes, 1) == 0)
            id = HAP_compute_res_acquire(&at, 100000);
        if (!id) { *ok = 0; return 0; }
        void *buf = HAP_compute_res_attr_get_vtcm_ptr(&at);
        if (!buf) { HAP_compute_res_release(id); *ok = 0; return 0; }

        /* Timed: only the `iters` passes below. Not timed: the acquire, the
         * HVX lock, and one untimed write pass that populates the buffer. */
        qurt_hvx_lock(QURT_HVX_MODE_128B);
        bw_write(buf, size_bytes, 1);                  /* warm / populate */
        uint64_t t0 = HAP_perf_get_time_us();
        if (mode == 3) bw_write(buf, size_bytes, iters);
        else           *a = bw_read(buf, size_bytes, iters);
        *us = (int)(HAP_perf_get_time_us() - t0);
        qurt_hvx_unlock();

        HAP_compute_res_release(id);
        *ok = 1;
        return 0;
    }

    /* ---- modes 5,6: ordinary memory bandwidth, same code path ---- */
    if (mode == 5 || mode == 6) {
        void *buf = memalign(128, (size_t)size_bytes);
        if (!buf) { *ok = 0; return 0; }

        qurt_hvx_lock(QURT_HVX_MODE_128B);
        bw_write(buf, size_bytes, 1);
        uint64_t t0 = HAP_perf_get_time_us();
        if (mode == 5) bw_write(buf, size_bytes, iters);
        else           *a = bw_read(buf, size_bytes, iters);
        *us = (int)(HAP_perf_get_time_us() - t0);
        qurt_hvx_unlock();

        free(buf);
        *ok = 1;
        return 0;
    }

    return 0;
}
