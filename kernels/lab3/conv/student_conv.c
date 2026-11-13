// student_conv.c — Lab 3, Task 10.1, multithreaded HVX convolution.
//
// Build with --threads 1 for the baseline and --threads 8 for the HDK8850:
//
//   bash scripts/build_student.sh conv
//   adb push kernels/convolution/dsp/prebuilt/libdsp_conv_skel.so \
//            /data/local/tmp/profiler-kernels/dsp/
//   adb shell "cd /data/local/tmp/profiler-kernels/dsp && LD_LIBRARY_PATH=. \
//       ADSP_LIBRARY_PATH='.;$PWD' ./conv_dsp --H 256 --W 256 --cin 64 \
//       --cout 64 --k 3 --iterations 5 --variant hvx"
//
// The `fingerprint=` in the output is a hash of the result. Partitioning work
// across contexts must not change it. If yours moves, your rows overlap or you
// have left a gap — and a faster wrong answer is still a wrong answer.

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

#define VFLOATS 32          // one 128-byte HVX vector holds 32 fp32 lanes
#define MY_STACK (64 * 1024)

// The build script sets CONV_MT_THREADS. The HDK8850 has eight 128B contexts.
#ifndef CONV_MT_THREADS
#define CONV_MT_THREADS 8
#endif
#define MY_THREADS CONV_MT_THREADS

typedef struct {
    const float *in;
    const float *w;
    float       *out;
    int H, W, Cin, Cout, K;
    int oh_begin, oh_end;   // this worker computes output rows [begin, end)
    int id, nthreads;       // which worker am I, out of how many
} my_stripe_t;

// ---------------------------------------------------------------------------
// The input-channel reduction accumulates in 32-lane HVX vectors, one tap at a
// time, then folds horizontally per output element. Full vectors are used while
// Cin >= 32; anything left over goes down the scalar tail. That asymmetry is
// the subject of Task 10.5, so it is worth reading even though you do not edit
// it.
// ---------------------------------------------------------------------------
static inline HVX_Vector load_vector(const float *p) {
    if ((uintptr_t)p % sizeof(HVX_Vector) == 0)
        return *(const HVX_Vector *)p;
    // The Task 10.5 sweep includes channel counts that misalign taps.
    return *(const HVX_UVector *)p;
}

static void conv_hvx_stripe(const my_stripe_t *s) {
    const int H = s->H, W = s->W, Cin = s->Cin, Cout = s->Cout, K = s->K;
    const int pad = K / 2;
    const int ic_vec = Cin / VFLOATS;
    const int ic_tail = ic_vec * VFLOATS;
    const float *in = s->in;
    const float *w = s->w;
    float *out = s->out;

    union { HVX_Vector v; float f[VFLOATS]; } lanes __attribute__((aligned(128)));

    for (int oh = s->oh_begin; oh < s->oh_end; oh++) {
        for (int ow = 0; ow < W; ow++) {
            float *out_px = &out[((long)oh * W + ow) * Cout];
            for (int oc = 0; oc < Cout; oc++) {
                HVX_Vector vacc = Q6_V_vzero();
                float tail = 0.0f;
                for (int kh = 0; kh < K; kh++) {
                    int ih = oh + kh - pad;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow + kw - pad;
                        if (iw < 0 || iw >= W) continue;
                        const float *in_px = &in[((long)ih * W + iw) * Cin];
                        const float *w_px  = &w[(((long)oc * K + kh) * K + kw) * Cin];
                        for (int iv = 0; iv < ic_vec; iv++) {
                            int ic = iv * VFLOATS;
                            HVX_Vector vin = load_vector(in_px + ic);
                            HVX_Vector vw  = load_vector(w_px + ic);
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

// Give each of the first H % nthreads workers one extra row.
static void my_range(int id, int nthreads, int H, int *oh_begin, int *oh_end) {
    const int rows = H / nthreads;
    const int extra = H % nthreads;
    *oh_begin = id * rows + (id < extra ? id : extra);
    *oh_end = *oh_begin + rows + (id < extra);
}

// This is the whole worker. It asks my_range() which rows are its own, claims
// an HVX context, computes them, and releases the context. The calling thread
// runs the same two steps for worker 0 without spawning.
static void my_worker(void *arg) {
    my_stripe_t *s = (my_stripe_t *)arg;
    my_range(s->id, s->nthreads, s->H, &s->oh_begin, &s->oh_end);

    qurt_hvx_lock(QURT_HVX_MODE_128B);
    conv_hvx_stripe(s);
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

// ---------------------------------------------------------------------------
// GIVEN. Entry point: stage weights in VTCM, spawn MY_THREADS-1 workers, run
// one stripe on the calling thread, join. You should not need to edit this,
// but you should read it — Task 11.5 asks about the VTCM block.
// ---------------------------------------------------------------------------
int dsp_conv_conv_hvx_fp32(remote_handle64 h,
                           const float *in, int inLen,
                           const float *w, int wLen,
                           float *out, int outLen,
                           int H, int W, int Cin, int Cout, int K,
                           int *dsp_us) {
    (void)h; (void)inLen; (void)wLen; (void)outLen;
    uint64_t t0 = HAP_perf_get_time_us();

    /* --- Stage the weights in VTCM -----------------------------------------
     * The inner loop walks the entire weight tensor for every output pixel:
     * Cout*K*K*Cin floats, re-read H*W times. The input, by contrast, is shared
     * between neighbouring output pixels and behaves well in cache. So the
     * weights are what belongs in the scratchpad.
     *
     * This happens unconditionally, at every size. Task 11.5 asks whether that
     * was a good idea. Leave it alone until then. */
    int w_bytes = Cout * K * K * Cin * (int)sizeof(float);
    const float *w_work = w;

    compute_res_attr_t res_attr;
    unsigned int ctx_id = 0;
    if (HAP_compute_res_attr_init(&res_attr) == 0) {
        if (HAP_compute_res_attr_set_vtcm_param(&res_attr,
                                                (unsigned int)w_bytes, 1) == 0) {
            ctx_id = HAP_compute_res_acquire(&res_attr, 100000); /* 100 ms */
        }
    }
    float *w_vtcm = (ctx_id != 0)
                    ? (float *)HAP_compute_res_attr_get_vtcm_ptr(&res_attr)
                    : 0;
    if (w_vtcm) {
        memcpy(w_vtcm, w, (unsigned int)w_bytes);
        w_work = (const float *)__builtin_assume_aligned(w_vtcm, 128);
    } else if (ctx_id != 0) {
        HAP_compute_res_release(ctx_id);
        ctx_id = 0;
    }

    int nthreads = MY_THREADS;
    if (nthreads > H) nthreads = H;
    if (nthreads < 1) nthreads = 1;

    my_stripe_t   stripes[MY_THREADS];
    qurt_thread_t tids[MY_THREADS];
    void         *stacks[MY_THREADS];
    int           spawned[MY_THREADS];

    for (int t = 0; t < nthreads; t++) {
        stripes[t].in   = in;   stripes[t].w    = w_work; stripes[t].out = out;
        stripes[t].H    = H;    stripes[t].W    = W;      stripes[t].Cin = Cin;
        stripes[t].Cout = Cout; stripes[t].K    = K;
        stripes[t].id   = t;    stripes[t].nthreads = nthreads;
        stripes[t].oh_begin = 0; stripes[t].oh_end = H;   /* my_worker overrides */
        stacks[t]  = 0;
        spawned[t] = 0;
    }

    int prio = qurt_thread_get_priority(qurt_thread_get_id());

    for (int t = 1; t < nthreads; t++) {
        stacks[t] = memalign(128, MY_STACK);
        if (!stacks[t]) continue;

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "student_conv");
        qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
        qurt_thread_attr_set_stack_size(&attr, MY_STACK);
        qurt_thread_attr_set_priority(&attr, (unsigned short)prio);

        if (qurt_thread_create(&tids[t], &attr, my_worker, &stripes[t]) == QURT_EOK) {
            spawned[t] = 1;
        } else {
            free(stacks[t]);
            stacks[t] = 0;
            FARF(ERROR, "student_conv: thread %d create failed, running inline", t);
        }
    }

    /* The calling thread is worker 0 — n stripes across n contexts, not n+1
     * threads for n stripes. Same two steps as my_worker(), minus the spawn and
     * the thread_exit, so it picks up your my_range() automatically. */
    {
        my_stripe_t *s = &stripes[0];
        my_range(s->id, s->nthreads, s->H, &s->oh_begin, &s->oh_end);
        qurt_hvx_lock(QURT_HVX_MODE_128B);
        conv_hvx_stripe(s);
        qurt_hvx_unlock();
    }

    for (int t = 1; t < nthreads; t++) {
        if (spawned[t]) {
            int status = 0;
            qurt_thread_join(tids[t], &status);
        }
    }

    /* A stripe whose thread could not be created still has to be computed. */
    for (int t = 1; t < nthreads; t++) {
        if (!spawned[t]) {
            my_stripe_t *s = &stripes[t];
            my_range(s->id, s->nthreads, s->H, &s->oh_begin, &s->oh_end);
            qurt_hvx_lock(QURT_HVX_MODE_128B);
            conv_hvx_stripe(s);
            qurt_hvx_unlock();
        }
        if (stacks[t]) free(stacks[t]);
    }

    if (ctx_id != 0) HAP_compute_res_release(ctx_id);

    uint64_t t_end = HAP_perf_get_time_us();
    *dsp_us = (int)(t_end - t0);
    FARF(RUNTIME_HIGH, "student_conv T=%d HxW=%dx%d C=%d/%d vtcm=%s dsp_us=%d",
         nthreads, H, W, Cin, Cout, w_vtcm ? "yes" : "no", *dsp_us);
    return 0;
}
