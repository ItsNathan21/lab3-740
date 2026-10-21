// dsp_matmul_hvx_imp.c — HVX-vectorized FP32 matmul, cDSP v79.
// Phase 6 Step 5.
//
// Strategy: vectorize over output columns (j) in chunks of 32 floats
//   (1 HVX vector = 128 bytes = 32 × float32).  Inner loop accumulates
//   A[i,k] * B[k, j:j+32] in QF32 space over all k.
//
// MULTITHREADED: the chip has 6 × 128B HVX contexts (qurt_hvx_get_units() ->
// 0x600) and the single-context version used exactly one. The output rows of C
// are partitioned into MM_MT_THREADS contiguous stripes, each worker locking its
// own HVX context. Row i of C depends only on row i of A and all of B (both
// read-only), and writes only row i of C — so no synchronization is needed and
// the result is bit-identical to the single-threaded path.
//
// Thread count is a compile-time constant so the A/B is a skel swap only: the
// IDL signature is unchanged, so the host binary and stub are untouched.
//
// Key intrinsics (hvx_hexagon_protos.h, -mhvx -mhvx-length=128B -mv79):
//   Q6_Vqf32_vmpy_VsfRsf(vec, bits) — IEEE-float-vec × scalar-reg → QF32
//   Q6_Vqf32_vadd_Vqf32Vqf32(a, b) — QF32 + QF32 → QF32
//   Q6_Vsf_equals_Vqf32(qf)        — QF32 → IEEE float32
//
// open/close already provided by dsp_matmul_imp.c — do NOT redeclare here.

#include "dsp_matmul.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "qurt_hvx.h"
#include "qurt_thread.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <stdlib.h>

// 128B HVX vector / 4 bytes per float = 32 float32 values per register.
#define VFLOATS 32

// 6 = the HVX context count on SM8750. Measured on the convolution kernel:
// T=6 is 4.85x T=1; T=8 regresses as 8 threads contend for 6 units.
#ifndef MM_MT_THREADS
#define MM_MT_THREADS 8   /* SM8850P has 8 HVX contexts (0x800); SM8750 had 6 */
#endif

#define MM_MT_STACK (64 * 1024)

typedef struct {
    const float *a_data;
    const float *b_data;
    float       *c_data;
    int n;
    int i_begin, i_end;
} mm_stripe_t;

// One contiguous band of output rows. Identical arithmetic to the original
// single-threaded loop; only the i bounds differ.
static void matmul_hvx_stripe(const mm_stripe_t *s) {
    const int n = s->n;
    const int n_vec = n / VFLOATS;   // full 32-float-wide strips per row
    const float *a_data = s->a_data;
    const float *b_data = s->b_data;
    float *c_data = s->c_data;

    for (int i = s->i_begin; i < s->i_end; i++) {
        const float* a_row = a_data + i * n;

        // --- HVX path: process 32 output columns at a time ---
        for (int jv = 0; jv < n_vec; jv++) {
            int j = jv * VFLOATS;

            // Initialize accumulator with the k=0 product to avoid QF32 zero init.
            HVX_Vector b0 = *((HVX_Vector*)(b_data + j));
            HVX_Vector acc_qf = Q6_Vqf32_vmpy_VsfRsf(b0, *(int*)&a_row[0]);

            for (int k = 1; k < n; k++) {
                // Load 32 consecutive IEEE-float32 from B[k, j:j+32].
                // Compiler emits VMEMU when b_data is not 128B-aligned.
                HVX_Vector bk = *((HVX_Vector*)(b_data + k * n + j));

                // Multiply-accumulate: Q6_Vqf32_vmpy_VsfRsf broadcasts the scalar.
                // *(int*)&a_row[k] reinterprets float bits as Word32 (safe type-pun).
                acc_qf = Q6_Vqf32_vadd_Vqf32Vqf32(
                             acc_qf,
                             Q6_Vqf32_vmpy_VsfRsf(bk, *(int*)&a_row[k]));
            }

            // Convert QF32 accumulator → IEEE float32 and store.
            *((HVX_Vector*)(c_data + i * n + j)) = Q6_Vsf_equals_Vqf32(acc_qf);
        }

        // --- Scalar tail when N % 32 != 0 (not triggered for N=512) ---
        for (int j = n_vec * VFLOATS; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < n; k++)
                acc += a_row[k] * b_data[k * n + j];
            c_data[i * n + j] = acc;
        }
    }
}

static void matmul_worker(void *arg) {
    mm_stripe_t *s = (mm_stripe_t *)arg;
    qurt_hvx_lock(QURT_HVX_MODE_128B);
    matmul_hvx_stripe(s);
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

int dsp_matmul_matmul_hvx_fp32(remote_handle64 h,
                                const float* a_data, int a_dataLen,
                                const float* b_data, int b_dataLen,
                                float* c_data,       int c_dataLen,
                                int n, int* dsp_us) {
    uint64_t t0 = HAP_perf_get_time_us();

    int nthreads = MM_MT_THREADS;
    if (nthreads > n) nthreads = n;
    if (nthreads < 1) nthreads = 1;

    mm_stripe_t   stripes[MM_MT_THREADS];
    qurt_thread_t tids[MM_MT_THREADS];
    void         *stacks[MM_MT_THREADS];
    int           spawned[MM_MT_THREADS];

    // Contiguous row partition; the remainder is spread over the first stripes.
    int base = n / nthreads, rem = n % nthreads, next = 0;
    for (int t = 0; t < nthreads; t++) {
        int rows = base + (t < rem ? 1 : 0);
        stripes[t].a_data = a_data;
        stripes[t].b_data = b_data;
        stripes[t].c_data = c_data;
        stripes[t].n = n;
        stripes[t].i_begin = next;
        stripes[t].i_end   = next + rows;
        next += rows;
        stacks[t]  = 0;
        spawned[t] = 0;
    }

    int prio = qurt_thread_get_priority(qurt_thread_get_id());

    // Stripes 1..n-1 go to workers; stripe 0 runs on the calling thread so we
    // use nthreads contexts total rather than nthreads+1 threads.
    for (int t = 1; t < nthreads; t++) {
        stacks[t] = memalign(128, MM_MT_STACK);
        if (!stacks[t]) continue;

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "mm_mt");
        qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
        qurt_thread_attr_set_stack_size(&attr, MM_MT_STACK);
        qurt_thread_attr_set_priority(&attr, (unsigned short)prio);

        if (qurt_thread_create(&tids[t], &attr, matmul_worker, &stripes[t]) == QURT_EOK) {
            spawned[t] = 1;
        } else {
            free(stacks[t]);
            stacks[t] = 0;
            FARF(ERROR, "dsp_matmul MT: thread %d create failed, running inline", t);
        }
    }

    // Calling thread takes stripe 0.
    qurt_hvx_lock(QURT_HVX_MODE_128B);
    matmul_hvx_stripe(&stripes[0]);
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
            matmul_hvx_stripe(&stripes[t]);
            qurt_hvx_unlock();
        }
        if (stacks[t]) free(stacks[t]);
    }

    *dsp_us = (int)(HAP_perf_get_time_us() - t0);
    FARF(RUNTIME_HIGH, "dsp_matmul HVX-MT(%d) n=%d dsp_us=%d", nthreads, n, *dsp_us);
    return 0;
}
