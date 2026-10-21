/*
 * nbody_neon.c — direct-summation N-body force kernel, NEON, threaded.
 *
 * WHY THIS KERNEL EXISTS
 * ----------------------
 * Every other workload in this profiler sits at one end of the roofline:
 * saxpy/reduction are memory-bound, matmul/convolution are dense tensor ops
 * that the NPU wins outright. Nothing occupied the compute-bound-but-not-a-
 * tensor-op region — which is exactly where a GPU is supposed to win.
 *
 * Direct N-body is that workload. It does O(N^2) arithmetic on O(N) data, so
 * arithmetic intensity grows linearly with N:
 *
 *     AI = 20*N^2 flop / (16*N bytes) = 1.25*N flop/byte
 *
 * At N=8192 that is ~10,000 flop/byte — three orders of magnitude past the
 * ridge point. It is also trivially parallel, and QNN cannot express it, so
 * the NPU cannot take it away.
 *
 * LAYOUT
 * ------
 * Structure-of-arrays (x[], y[], z[], m[]) rather than array-of-structs. This
 * lets the inner loop load four j-bodies per vector on NEON, and gives the
 * OpenCL version perfectly coalesced reads. Both backends use the same layout
 * so the two sources read as the same algorithm.
 *
 * PRECISION
 * ---------
 * Uses IEEE sqrt and divide (vsqrtq_f32 / vdivq_f32), NOT the fast reciprocal
 * estimates. vrsqrteq_f32 + Newton would be measurably faster but would not
 * agree with the GPU's sqrt(), and a cross-backend fingerprint that cannot be
 * compared is worth less than the cycles it saves.
 *
 * DETERMINISM
 * -----------
 * Thread t owns a contiguous band of i-bodies and accumulates over ALL j in
 * ascending order. The summation order for any given body is therefore
 * independent of thread count, so the fingerprint is bit-identical from 1 to 8
 * threads — the same property the other threaded kernels here guarantee.
 *
 * Output line:
 *   nbody-neon_handtuned-end fingerprint=<float> wall_us=<int> \
 *     iterations=<int> threads=<int> pin=<0|1> prime_weight=<float> \
 *     bodies=<int> flop=<int> bytes=<int>
 */

/* _GNU_SOURCE comes from the build flag (-D_GNU_SOURCE), not from here — it
 * must be defined before any system header, and cpu_threads.h is included
 * below system headers. Same convention as the other threaded kernels. */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../../common/cpu_threads.h"

/* Softening factor. Prevents a singularity when two bodies coincide; every
 * production N-body code has one. eps2 is added to r^2 before the sqrt. */
#define EPS2 1e-6f

/* Conventional N-body flop count per pairwise interaction. The literature
 * standard is 20; this loop issues 18-20 depending on how the divide is
 * counted, and we report 20 so numbers are comparable with published figures. */
#define FLOP_PER_INTERACTION 20

typedef struct {
    const float *x, *y, *z, *m;
    float *ax, *ay, *az;
    int n;
    int reps;
} nbody_ctx;

/* Horizontal add of a 4-lane vector. */
static inline float hsum(float32x4_t v) { return vaddvq_f32(v); }

static void nbody_worker(void *vctx, int tid, int nthreads) {
    nbody_ctx *c = (nbody_ctx *)vctx;
    const int n = c->n;

    cpu_pin_self(tid);

    /* Each thread owns a contiguous band of i-bodies. grain=1: the i-loop is
     * scalar in i (vectorisation is over j), so no alignment constraint. */
    int lo, hi;
    cpu_band(n, 1, tid, nthreads, &lo, &hi);

    const float32x4_t eps2 = vdupq_n_f32(EPS2);

    for (int r = 0; r < c->reps; r++) {
        for (int i = lo; i < hi; i++) {
            const float32x4_t xi = vdupq_n_f32(c->x[i]);
            const float32x4_t yi = vdupq_n_f32(c->y[i]);
            const float32x4_t zi = vdupq_n_f32(c->z[i]);

            float32x4_t axv = vdupq_n_f32(0.0f);
            float32x4_t ayv = vdupq_n_f32(0.0f);
            float32x4_t azv = vdupq_n_f32(0.0f);

            /* n is forced to a multiple of 4 in main(), so no tail loop. */
            for (int j = 0; j < n; j += 4) {
                float32x4_t dx = vsubq_f32(vld1q_f32(&c->x[j]), xi);
                float32x4_t dy = vsubq_f32(vld1q_f32(&c->y[j]), yi);
                float32x4_t dz = vsubq_f32(vld1q_f32(&c->z[j]), zi);

                /* r2 = dx^2 + dy^2 + dz^2 + eps2 */
                float32x4_t r2 = vfmaq_f32(eps2, dx, dx);
                r2 = vfmaq_f32(r2, dy, dy);
                r2 = vfmaq_f32(r2, dz, dz);

                /* s = m_j / r^3, with r^3 = r2 * sqrt(r2). IEEE ops so the
                 * result matches the OpenCL kernel's sqrt()/divide. */
                float32x4_t r1 = vsqrtq_f32(r2);
                float32x4_t r3 = vmulq_f32(r2, r1);
                float32x4_t s  = vdivq_f32(vld1q_f32(&c->m[j]), r3);

                axv = vfmaq_f32(axv, dx, s);
                ayv = vfmaq_f32(ayv, dy, s);
                azv = vfmaq_f32(azv, dz, s);
            }

            /* Self-interaction contributed dx=dy=dz=0, so it added exactly
             * zero — no need to special-case j==i. */
            c->ax[i] = hsum(axv);
            c->ay[i] = hsum(ayv);
            c->az[i] = hsum(azv);
        }
    }
}

/* splitmix64 — identical generator to every other kernel in this tree, so the
 * GPU version can reproduce the same bodies from the same seed. */
static uint64_t sm_state;
static void sm_seed(uint64_t s) { sm_state = s; }
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static float sm_float(void) {
    return (float)(sm_next() >> 11) * (1.0f / (float)(1ULL << 53));
}

static double fingerprint(const float *ax, const float *ay, const float *az, int n) {
    double sum = 0.0, sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)ax[i] + (double)ay[i] + (double)az[i];
        sq  += (double)ax[i] * ax[i] + (double)ay[i] * ay[i] + (double)az[i] * az[i];
    }
    return sum + sqrt(sq);
}

int main(int argc, char **argv) {
    int bodies  = 4096;
    int reps    = 1;
    int threads = 1;
    int seed    = 42;
    int pin     = 0;
    double prime_weight = 1.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bodies") == 0 && i + 1 < argc) {
            bodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            bodies = atoi(argv[++i]);            /* alias, matches other kernels */
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pin") == 0) {
            pin = 1;
        } else if (strcmp(argv[i], "--prime-weight") == 0 && i + 1 < argc) {
            prime_weight = atof(argv[++i]);
        }
    }

    if (threads < 1) threads = 1;
    if (threads > CPU_MAX_THREADS) threads = CPU_MAX_THREADS;
    /* Multiple of 64, not 4: the OpenCL tiled variant stages 64 bodies per
     * workgroup, and keeping both backends on the same constraint means they
     * simulate exactly the same system for a given --bodies. */
    if (bodies < 64) bodies = 64;
    bodies &= ~63;

    cpu_split_config(pin, prime_weight);
    cpu_prepare_split(threads);

    float *x  = aligned_alloc(64, (size_t)bodies * sizeof(float));
    float *y  = aligned_alloc(64, (size_t)bodies * sizeof(float));
    float *z  = aligned_alloc(64, (size_t)bodies * sizeof(float));
    float *m  = aligned_alloc(64, (size_t)bodies * sizeof(float));
    float *ax = aligned_alloc(64, (size_t)bodies * sizeof(float));
    float *ay = aligned_alloc(64, (size_t)bodies * sizeof(float));
    float *az = aligned_alloc(64, (size_t)bodies * sizeof(float));
    if (!x || !y || !z || !m || !ax || !ay || !az) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    /* Bodies in a unit cube, masses in [0,1). Four independent streams so the
     * GPU kernel can regenerate them identically. */
    sm_seed((uint64_t)seed);
    for (int i = 0; i < bodies; i++) x[i] = sm_float();
    sm_seed((uint64_t)seed + 1);
    for (int i = 0; i < bodies; i++) y[i] = sm_float();
    sm_seed((uint64_t)seed + 2);
    for (int i = 0; i < bodies; i++) z[i] = sm_float();
    sm_seed((uint64_t)seed + 3);
    for (int i = 0; i < bodies; i++) m[i] = sm_float();

    memset(ax, 0, (size_t)bodies * sizeof(float));
    memset(ay, 0, (size_t)bodies * sizeof(float));
    memset(az, 0, (size_t)bodies * sizeof(float));

    nbody_ctx ctx = { x, y, z, m, ax, ay, az, bodies, reps };

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    cpu_parallel(threads, nbody_worker, &ctx);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    int64_t wall_us = (int64_t)((t1.tv_sec - t0.tv_sec) * 1000000LL +
                                (t1.tv_nsec - t0.tv_nsec) / 1000LL);

    /* flop is the TOTAL across all reps (flop_is_total=True in profiler.py).
     * bytes counts the j-array traffic actually streamed: 4 arrays * 4 B, read
     * once per i per rep. The caches absorb nearly all of it, which is the
     * whole point — this kernel is compute-bound by construction. */
    int64_t interactions = (int64_t)bodies * bodies * reps;
    int64_t flop  = interactions * FLOP_PER_INTERACTION;
    int64_t bytes = (int64_t)bodies * bodies * 4 * (int64_t)sizeof(float) * reps;

    double fp = fingerprint(ax, ay, az, bodies);
    fprintf(stdout,
            "nbody-neon_handtuned-end fingerprint=%.6e wall_us=%lld "
            "iterations=%d threads=%d pin=%d prime_weight=%.3f "
            "bodies=%d flop=%lld bytes=%lld\n",
            fp, (long long)wall_us, reps, threads, pin, prime_weight,
            bodies, (long long)flop, (long long)bytes);
    fflush(stdout);

    free(x); free(y); free(z); free(m);
    free(ax); free(ay); free(az);
    return 0;
}
