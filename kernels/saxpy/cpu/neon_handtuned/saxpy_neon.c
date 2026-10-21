/*
 * Snapdragon Profiler — saxpy / cpu / neon_handtuned (roofline, memory-bound anchor).
 *
 * y <- alpha * x + y, single-precision, hand-vectorized with NEON:
 * 4 floats per iteration via vfmaq_n_f32(y, x, alpha) = y + x*alpha.
 * SAXPY is bandwidth-bound (arithmetic intensity ~0.17 FLOP/byte), so this
 * is expected to land at essentially the same throughput as the scalar
 * baseline once both saturate DRAM — that convergence is itself a result
 * (there is no compute headroom for vectorization to recover). See
 * kernels/saxpy/DESIGN.md.
 *
 * Inputs x and y are deterministic via splitmix64 from the seed
 * (bit-identical to kernels/saxpy/cpu/reference.py and the scalar variant).
 *
 * Argv contract:
 *   --size N          vector length (default 4194304 = 2^22); multiple of 4
 *   --seed N          splitmix64 seed for inputs (default 42)
 *   --alpha F         scalar multiplier (default 2.0)
 *   --validate-only   compute exactly ONE pass, print fingerprint, exit
 *   --iterations N    accumulating passes before printing (default 1)
 *   --threads N       worker threads over disjoint element bands (default 1)
 *
 * Stdout:
 *   saxpy-neon_handtuned-start size=<N> seed=<S> alpha=<F> threads=<T>
 *   saxpy-neon_handtuned-end fingerprint=<float> wall_us=<int> \
 *       iterations=<K> threads=<T> elems=<N> bytes=<int>
 *
 * bytes = 3 * N * 4 * K. Fingerprint = sum(y) + L2_norm(y), fp64. Validated
 * vs the NumPy reference within 1e-3 relative tolerance.
 *
 * Threading: y[i] depends only on x[i] and y[i], so bands are fully independent
 * and the fingerprint is BIT-IDENTICAL at any --threads value. SAXPY is the
 * memory-bound corner of the roofline (~0.17 FLOP/byte), so more cores are
 * expected to buy little once DRAM saturates — that flatness is the result.
 */

#include <arm_neon.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cpu_threads.h"

/* ── splitmix64 (bit-identical to reference.py + matmul kernels) ── */

static const uint64_t SPLITMIX_INC = 0x9E3779B97F4A7C15ULL;
static const uint64_t SPLITMIX_M1 = 0xBF58476D1CE4E5B9ULL;
static const uint64_t SPLITMIX_M2 = 0x94D049BB133111EBULL;

static inline uint64_t splitmix64_next(uint64_t* state) {
    *state += SPLITMIX_INC;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * SPLITMIX_M1;
    z = (z ^ (z >> 27)) * SPLITMIX_M2;
    z = z ^ (z >> 31);
    return z;
}

static inline double u64_to_unit_double(uint64_t x) {
    return (double)(x >> 11) * (1.0 / (double)(1ULL << 53));
}

static void generate_inputs(int size, uint64_t seed, float* x, float* y) {
    uint64_t state = seed;
    for (int i = 0; i < size; i++) {
        x[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(&state)) - 1.0);
    }
    for (int i = 0; i < size; i++) {
        y[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(&state)) - 1.0);
    }
}

/* ── NEON SAXPY over the band [begin, end): y <- alpha*x + y, 4 lanes/iter ──
 * Bounds are multiples of 4 (v1 size contract + grain-4 banding), no tail. */
static void saxpy_neon(float alpha, const float* x, float* y,
                       int begin, int end) {
    for (int i = begin; i < end; i += 4) {
        float32x4_t vx = vld1q_f32(&x[i]);
        float32x4_t vy = vld1q_f32(&y[i]);
        /* vfmaq_n_f32(acc, vec, scalar) = acc + vec*scalar = y + x*alpha */
        vy = vfmaq_n_f32(vy, vx, alpha);
        vst1q_f32(&y[i], vy);
    }
}

/* Each worker owns one band for ALL iterations — one spawn per run, not one
 * per pass. Bands are disjoint and there is no cross-iteration dependency. */
typedef struct {
    float alpha;
    const float* x;
    float* y;
    int size;
    int reps;
} saxpy_ctx;

static void saxpy_worker(void* vctx, int tid, int nthreads) {
    saxpy_ctx* c = (saxpy_ctx*)vctx;
    int begin, end;
    cpu_band(c->size, 4, tid, nthreads, &begin, &end);
    for (int it = 0; it < c->reps; it++) {
        saxpy_neon(c->alpha, c->x, c->y, begin, end);
    }
}

static double compute_fingerprint(const float* y, int n) {
    double s = 0.0, sq = 0.0;
    for (int i = 0; i < n; i++) {
        double v = (double)y[i];
        s += v;
        sq += v * v;
    }
    return s + sqrt(sq);
}

static int parse_args(int argc, char** argv, int* size, uint64_t* seed,
                      float* alpha, bool* validate_only, int* iterations,
                      int* threads) {
    *size = 4194304; /* 2^22 */
    *seed = 42;
    *alpha = 2.0f;
    *validate_only = false;
    *iterations = 1;
    *threads = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            *size = atoi(argv[++i]);
            if (*size <= 0 || (*size % 4) != 0) {
                fprintf(stderr, "ERROR: --size must be > 0 and a multiple of 4\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            *seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            *alpha = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--validate-only") == 0) {
            *validate_only = true;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            *iterations = atoi(argv[++i]);
            if (*iterations <= 0) {
                fprintf(stderr, "ERROR: --iterations must be > 0\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            *threads = atoi(argv[++i]);
            if (*threads < 1 || *threads > CPU_MAX_THREADS) {
                fprintf(stderr, "ERROR: --threads must be 1..%d\n", CPU_MAX_THREADS);
                return 1;
            }
        } else {
            fprintf(stderr, "ERROR: unknown arg: %s\n", argv[i]);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    int size, iterations, threads;
    uint64_t seed;
    float alpha;
    bool validate_only;
    if (parse_args(argc, argv, &size, &seed, &alpha, &validate_only, &iterations,
                   &threads)) {
        return 1;
    }
    if (threads > size / 4) threads = size / 4;   /* never fewer than 4 elems/band */

    fprintf(stdout,
            "saxpy-neon_handtuned-start size=%d seed=%llu alpha=%.6g threads=%d\n",
            size, (unsigned long long)seed, (double)alpha, threads);
    fflush(stdout);

    float *x = NULL, *y = NULL;
    if (posix_memalign((void**)&x, 64, (size_t)size * sizeof(float)) != 0 ||
        posix_memalign((void**)&y, 64, (size_t)size * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: posix_memalign failed\n");
        return 1;
    }
    generate_inputs(size, seed, x, y);

    int reps = validate_only ? 1 : iterations;

    saxpy_ctx ctx = { alpha, x, y, size, reps };
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    cpu_parallel(threads, saxpy_worker, &ctx);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    int64_t wall_us = (int64_t)((t1.tv_sec - t0.tv_sec) * 1000000LL +
                                (t1.tv_nsec - t0.tv_nsec) / 1000LL);
    int64_t bytes = (int64_t)3 * size * (int64_t)sizeof(float) * reps;

    double fp = compute_fingerprint(y, size);
    fprintf(stdout,
            "saxpy-neon_handtuned-end fingerprint=%.6e wall_us=%lld "
            "iterations=%d threads=%d elems=%d bytes=%lld\n",
            fp, (long long)wall_us, reps, threads, size, (long long)bytes);
    fflush(stdout);

    free(x);
    free(y);
    return 0;
}
