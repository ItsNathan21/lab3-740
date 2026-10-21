// dsp_conv_hvx_imp.c — HVX-vectorized FP32 direct convolution, cDSP v79.
// The input-channel (ic) reduction is accumulated in 32-lane HVX vectors, one
// tap at a time, then horizontally folded per output element. Efficient when
// Cin >= 32 (a full HVX vector per tap); at small Cin the vector is partly empty
// and a scalar tail dominates — so, like the CPU NEON conv, HVX's benefit grows
// with the channel count (visible in the AI sweep).
//
// MULTITHREADED: the chip has 6 x 128B HVX contexts (qurt_hvx_get_units()
// reports 0x600) and the single-context version used exactly one of them. The
// output-row range is partitioned into CONV_MT_THREADS contiguous stripes, each
// worker locking its own HVX context. Output rows are independent — a stripe
// reads input rows and read-only weights and writes only its own output rows —
// so no synchronization is needed inside the loop and the result is bit-identical
// to the single-threaded path (same per-element operation order).
//
// Thread count is a compile-time constant so the A/B is a skel swap only: the
// IDL signature is unchanged, so the host binary and stub are untouched.
//
// open/close/scalar provided by dsp_conv_imp.c — do NOT redeclare.

#include "dsp_conv.h"
#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "qurt_hvx.h"
#include "qurt_thread.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define VFLOATS 32   // 128B HVX vector / 4 bytes per float

// 6 = the HVX context count on SM8750 (qurt_hvx_get_units() -> 0x600). Measured
// peak: T=6 is 4.85x T=1 at C=64; T=8 regresses to 3.50x as 8 threads contend
// for 6 units.
#ifndef CONV_MT_THREADS
#define CONV_MT_THREADS 6
#endif

#define CONV_MT_STACK (64 * 1024)

typedef struct {
    const float *in;
    const float *w;
    float       *out;
    int H, W, Cin, Cout, K;
    int oh_begin, oh_end;
} conv_stripe_t;

// One contiguous band of output rows. Identical arithmetic to the original
// single-threaded loop; only the oh bounds differ.
static void conv_hvx_stripe(const conv_stripe_t *s) {
    const int H = s->H, W = s->W, Cin = s->Cin, Cout = s->Cout, K = s->K;
    const int pad = K / 2;
    const int ic_vec = Cin / VFLOATS;   // full 32-wide strips over channels
    const int ic_tail = ic_vec * VFLOATS;
    const float *in = s->in;
    const float *w = s->w;
    float *out = s->out;

    union { HVX_Vector v; float f[VFLOATS]; } lanes __attribute__((aligned(128)));

    for (int oh = s->oh_begin; oh < s->oh_end; oh++) {
        for (int ow = 0; ow < W; ow++) {
            float* out_px = &out[((long)oh * W + ow) * Cout];
            for (int oc = 0; oc < Cout; oc++) {
                HVX_Vector vacc = Q6_V_vzero();
                float tail = 0.0f;
                for (int kh = 0; kh < K; kh++) {
                    int ih = oh + kh - pad;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow + kw - pad;
                        if (iw < 0 || iw >= W) continue;
                        const float* in_px = &in[((long)ih * W + iw) * Cin];
                        const float* w_px  = &w[(((long)oc * K + kh) * K + kw) * Cin];
                        for (int iv = 0; iv < ic_vec; iv++) {
                            int ic = iv * VFLOATS;
                            HVX_Vector vin = *((HVX_Vector*)(in_px + ic));
                            HVX_Vector vw  = *((HVX_Vector*)(w_px + ic));
                            // vacc += vin * vw   (mpy -> qf32, add to sf acc, back to sf)
                            HVX_Vector prod = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vin, vw));
                            vacc = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(vacc, prod));
                        }
                        for (int ic = ic_tail; ic < Cin; ic++) tail += in_px[ic] * w_px[ic];
                    }
                }
                lanes.v = vacc;
                float acc = tail;
                for (int lane = 0; lane < VFLOATS; lane++) acc += lanes.f[lane];
                out_px[oc] = acc;
            }
        }
    }
}

static void conv_worker(void *arg) {
    conv_stripe_t *s = (conv_stripe_t *)arg;
    qurt_hvx_lock(QURT_HVX_MODE_128B);
    conv_hvx_stripe(s);
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

int dsp_conv_conv_hvx_fp32(remote_handle64 h,
                           const float* in, int inLen,
                           const float* w, int wLen,
                           float* out, int outLen,
                           int H, int W, int Cin, int Cout, int K,
                           int* dsp_us) {
    uint64_t t0 = HAP_perf_get_time_us();

    /* --- Stage the weights in VTCM ------------------------------------------
     * The inner loop walks the ENTIRE weight tensor for every output pixel:
     * Cout*K*K*Cin floats, re-read H*W times with no reuse distance to exploit.
     * The input, by contrast, is shared between neighbouring output pixels and
     * behaves well in cache. So the weights are what belongs in the scratchpad.
     *
     * Cout*K*K*Cin*4 is 144 KB at C=64 and 576 KB at C=128 — comfortably inside
     * the 8 MB VTCM. Acquired ONCE here, before the workers spawn; it is
     * read-only during compute, so all six threads share one copy with no
     * synchronization. Falls back to ordinary memory if VTCM is unavailable. */
    int w_bytes = Cout * K * K * Cin * (int)sizeof(float);
    const float* w_work = w;

    compute_res_attr_t res_attr;
    unsigned int ctx_id = 0;
    if (HAP_compute_res_attr_init(&res_attr) == 0) {
        if (HAP_compute_res_attr_set_vtcm_param(&res_attr,
                                                (unsigned int)w_bytes, 1) == 0) {
            ctx_id = HAP_compute_res_acquire(&res_attr, 100000); /* 100 ms */
        }
    }
    float* w_vtcm = (ctx_id != 0)
                    ? (float*)HAP_compute_res_attr_get_vtcm_ptr(&res_attr)
                    : 0;
    if (w_vtcm) {
        memcpy(w_vtcm, w, (unsigned int)w_bytes);
        /* Promise 128B alignment so the compiler emits VMEM, not VMEMU. */
        w_work = (const float*)__builtin_assume_aligned(w_vtcm, 128);
    } else if (ctx_id != 0) {
        HAP_compute_res_release(ctx_id);
        ctx_id = 0;
    }

    uint64_t t_setup = HAP_perf_get_time_us();

    int nthreads = CONV_MT_THREADS;
    if (nthreads > H) nthreads = H;
    if (nthreads < 1) nthreads = 1;

    conv_stripe_t stripes[CONV_MT_THREADS];
    qurt_thread_t tids[CONV_MT_THREADS];
    void         *stacks[CONV_MT_THREADS];
    int           spawned[CONV_MT_THREADS];

    // Contiguous row partition; the remainder is spread over the first stripes.
    int base = H / nthreads, rem = H % nthreads, next = 0;
    for (int t = 0; t < nthreads; t++) {
        int n = base + (t < rem ? 1 : 0);
        stripes[t].in = in;   stripes[t].w = w_work; stripes[t].out = out;
        stripes[t].H  = H;    stripes[t].W = W;      stripes[t].Cin = Cin;
        stripes[t].Cout = Cout; stripes[t].K = K;
        stripes[t].oh_begin = next;
        stripes[t].oh_end   = next + n;
        next += n;
        stacks[t]  = 0;
        spawned[t] = 0;
    }

    int prio = qurt_thread_get_priority(qurt_thread_get_id());

    // Stripes 1..n-1 go to workers; stripe 0 runs on the calling thread so we
    // use n contexts total rather than n+1 threads for n stripes.
    for (int t = 1; t < nthreads; t++) {
        stacks[t] = memalign(128, CONV_MT_STACK);
        if (!stacks[t]) continue;

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "conv_mt");
        qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
        qurt_thread_attr_set_stack_size(&attr, CONV_MT_STACK);
        qurt_thread_attr_set_priority(&attr, (unsigned short)prio);

        if (qurt_thread_create(&tids[t], &attr, conv_worker, &stripes[t]) == QURT_EOK) {
            spawned[t] = 1;
        } else {
            free(stacks[t]);
            stacks[t] = 0;
            FARF(ERROR, "dsp_conv MT: thread %d create failed, running inline", t);
        }
    }

    // Calling thread takes stripe 0.
    qurt_hvx_lock(QURT_HVX_MODE_128B);
    conv_hvx_stripe(&stripes[0]);
    qurt_hvx_unlock();

    for (int t = 1; t < nthreads; t++) {
        if (spawned[t]) {
            int status = 0;
            qurt_thread_join(tids[t], &status);
        }
    }

    // Any stripe whose thread could not be created still has to be computed.
    for (int t = 1; t < nthreads; t++) {
        if (!spawned[t]) {
            qurt_hvx_lock(QURT_HVX_MODE_128B);
            conv_hvx_stripe(&stripes[t]);
            qurt_hvx_unlock();
        }
        if (stacks[t]) free(stacks[t]);
    }

    if (ctx_id != 0) HAP_compute_res_release(ctx_id);

    uint64_t t_end = HAP_perf_get_time_us();
    *dsp_us = (int)(t_end - t0);
    FARF(RUNTIME_HIGH,
         "dsp_conv HVX-MT(%d) HxW=%dx%d C=%d/%d vtcm=%s w_kb=%d setup_us=%u "
         "compute_us=%u dsp_us=%d",
         nthreads, H, W, Cin, Cout, w_vtcm ? "yes" : "no", w_bytes / 1024,
         (unsigned int)(t_setup - t0), (unsigned int)(t_end - t_setup), *dsp_us);
    return 0;
}
