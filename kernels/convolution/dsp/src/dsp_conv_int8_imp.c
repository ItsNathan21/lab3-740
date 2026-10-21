// dsp_conv_int8_imp.c — INT8 direct convolution on cDSP v79, HVX + VTCM + MT.
//
// This is HVX's native regime. One 128-byte vector holds 128 int8 lanes against
// only 32 fp32, and `vrmpy` performs a 4-way byte dot product per word lane:
// one instruction consumes 128 activation bytes and 128 weight bytes and
// produces 32 int32 partial sums. The fp32 path needs a separate multiply and
// add per 32 lanes to do the same work.
//
// Accumulation is int32 and exact. Worst case is |127 * -128| = 16256 per
// product over K*K*Cin terms; at K=3, Cin=256 that is 9*256*16256 ≈ 3.7e7,
// comfortably inside int32. So this kernel and the CPU int8 kernel must agree
// **bit for bit** — a far stronger correctness check than fp32 allows.
//
// Same structure as the fp32 path: weights staged in VTCM once, output rows
// partitioned across CONV_MT_THREADS HVX contexts.
//
// open/close/scalar/hvx provided elsewhere — do NOT redeclare.

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

#define VBYTES 128   // one HVX vector, in int8 lanes
#define VWORDS 32    // the same vector viewed as int32 lanes

#ifndef CONV_MT_THREADS
#define CONV_MT_THREADS 6
#endif

#define CONV_MT_STACK (64 * 1024)

typedef struct {
    const int8_t *in;
    const int8_t *w;
    int          *out;   /* qaic maps `rout sequence<long>` to int* on Hexagon */
    int H, W, Cin, Cout, K;
    int oh_begin, oh_end;
} conv8_stripe_t;

static void conv_int8_stripe(const conv8_stripe_t *s) {
    const int H = s->H, W = s->W, Cin = s->Cin, Cout = s->Cout, K = s->K;
    const int pad = K / 2;
    const int ic_vec  = Cin / VBYTES;          // whole 128-byte channel blocks
    const int ic_tail = ic_vec * VBYTES;       // where the scalar tail starts
    const int8_t *in = s->in, *w = s->w;
    int *out = s->out;

    union { HVX_Vector v; int32_t word[VWORDS]; } lanes __attribute__((aligned(128)));

    for (int oh = s->oh_begin; oh < s->oh_end; oh++) {
        for (int ow = 0; ow < W; ow++) {
            int *out_px = &out[((long)oh * W + ow) * Cout];
            for (int oc = 0; oc < Cout; oc++) {
                HVX_Vector vacc = Q6_V_vzero();
                int32_t tail = 0;
                for (int kh = 0; kh < K; kh++) {
                    int ih = oh + kh - pad;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow + kw - pad;
                        if (iw < 0 || iw >= W) continue;
                        const int8_t *in_px = &in[((long)ih * W + iw) * Cin];
                        const int8_t *w_px  = &w[(((long)oc * K + kh) * K + kw) * Cin];
                        for (int iv = 0; iv < ic_vec; iv++) {
                            int ic = iv * VBYTES;
                            HVX_Vector vin = *((HVX_Vector*)(in_px + ic));
                            HVX_Vector vw  = *((HVX_Vector*)(w_px + ic));
                            // 32 word lanes, each += 4 signed byte products.
                            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, vin, vw);
                        }
                        for (int ic = ic_tail; ic < Cin; ic++)
                            tail += (int32_t)in_px[ic] * (int32_t)w_px[ic];
                    }
                }
                lanes.v = vacc;
                int32_t acc = tail;
                for (int l = 0; l < VWORDS; l++) acc += lanes.word[l];
                out_px[oc] = acc;
            }
        }
    }
}

static void conv_int8_worker(void *arg) {
    conv8_stripe_t *s = (conv8_stripe_t *)arg;
    qurt_hvx_lock(QURT_HVX_MODE_128B);
    conv_int8_stripe(s);
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

int dsp_conv_conv_int8(remote_handle64 h,
                       const unsigned char* in, int inLen,
                       const unsigned char* w, int wLen,
                       int* out, int outLen,
                       int H, int W, int Cin, int Cout, int K,
                       int* dsp_us) {
    uint64_t t0 = HAP_perf_get_time_us();

    const int8_t* in_s = (const int8_t*)in;
    const int8_t* w_s  = (const int8_t*)w;

    /* Weights into VTCM, once, shared read-only by every worker. At int8 the
     * tensor is a quarter the size of the fp32 one — 576 KB at Cout=Cin=256. */
    int w_bytes = Cout * K * K * Cin;
    const int8_t* w_work = w_s;

    compute_res_attr_t res_attr;
    unsigned int ctx_id = 0;
    if (HAP_compute_res_attr_init(&res_attr) == 0) {
        if (HAP_compute_res_attr_set_vtcm_param(&res_attr,
                                                (unsigned int)w_bytes, 1) == 0) {
            ctx_id = HAP_compute_res_acquire(&res_attr, 100000);
        }
    }
    int8_t* w_vtcm = (ctx_id != 0)
                     ? (int8_t*)HAP_compute_res_attr_get_vtcm_ptr(&res_attr)
                     : 0;
    if (w_vtcm) {
        memcpy(w_vtcm, w_s, (unsigned int)w_bytes);
        w_work = (const int8_t*)__builtin_assume_aligned(w_vtcm, 128);
    } else if (ctx_id != 0) {
        HAP_compute_res_release(ctx_id);
        ctx_id = 0;
    }

    uint64_t t_setup = HAP_perf_get_time_us();

    int nthreads = CONV_MT_THREADS;
    if (nthreads > H) nthreads = H;
    if (nthreads < 1) nthreads = 1;

    conv8_stripe_t stripes[CONV_MT_THREADS];
    qurt_thread_t  tids[CONV_MT_THREADS];
    void          *stacks[CONV_MT_THREADS];
    int            spawned[CONV_MT_THREADS];

    int base = H / nthreads, rem = H % nthreads, next = 0;
    for (int t = 0; t < nthreads; t++) {
        int rows = base + (t < rem ? 1 : 0);
        stripes[t].in = in_s;  stripes[t].w = w_work;  stripes[t].out = out;
        stripes[t].H = H;      stripes[t].W = W;       stripes[t].Cin = Cin;
        stripes[t].Cout = Cout; stripes[t].K = K;
        stripes[t].oh_begin = next;
        stripes[t].oh_end   = next + rows;
        next += rows;
        stacks[t] = 0; spawned[t] = 0;
    }

    int prio = qurt_thread_get_priority(qurt_thread_get_id());

    for (int t = 1; t < nthreads; t++) {
        stacks[t] = memalign(128, CONV_MT_STACK);
        if (!stacks[t]) continue;
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "conv8_mt");
        qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
        qurt_thread_attr_set_stack_size(&attr, CONV_MT_STACK);
        qurt_thread_attr_set_priority(&attr, (unsigned short)prio);
        if (qurt_thread_create(&tids[t], &attr, conv_int8_worker, &stripes[t]) == QURT_EOK)
            spawned[t] = 1;
        else { free(stacks[t]); stacks[t] = 0; }
    }

    qurt_hvx_lock(QURT_HVX_MODE_128B);
    conv_int8_stripe(&stripes[0]);
    qurt_hvx_unlock();

    for (int t = 1; t < nthreads; t++)
        if (spawned[t]) { int st = 0; qurt_thread_join(tids[t], &st); }

    for (int t = 1; t < nthreads; t++) {
        if (!spawned[t]) {
            qurt_hvx_lock(QURT_HVX_MODE_128B);
            conv_int8_stripe(&stripes[t]);
            qurt_hvx_unlock();
        }
        if (stacks[t]) free(stacks[t]);
    }

    if (ctx_id != 0) HAP_compute_res_release(ctx_id);

    uint64_t t_end = HAP_perf_get_time_us();
    *dsp_us = (int)(t_end - t0);
    FARF(RUNTIME_HIGH,
         "dsp_conv INT8-MT(%d) HxW=%dx%d C=%d/%d vtcm=%s setup_us=%u dsp_us=%d",
         nthreads, H, W, Cin, Cout, w_vtcm ? "yes" : "no",
         (unsigned int)(t_setup - t0), *dsp_us);
    return 0;
}
