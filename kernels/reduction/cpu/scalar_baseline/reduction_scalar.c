/*
 * Snapdragon Profiler — reduction / cpu / scalar_baseline (roofline, memory-bound anchor #2).
 *
 * s <- sum(x_i), single-precision, naive single-accumulator scalar loop.
 * Sum-reduction is read-only and bandwidth-bound (arithmetic intensity
 * ~0.25 FLOP/byte). This baseline uses one fp32 accumulator, so its inner loop
 * is a dependent add chain — latency-bound at small N — which is exactly the
 * foil for the multi-accumulator NEON variant. At DRAM-sized N both converge to
 * the read-bandwidth roof (no compute headroom to recover), and that
 * convergence is itself a result. See kernels/reduction/DESIGN.md.
 *
 * Input x is deterministic via splitmix64 from the seed (bit-identical to
 * kernels/reduction/reference.py and the NEON variant).
 *
 * Argv contract:
 *   --size N          vector length (default 4194304 = 2^22); multiple of 16
 *   --seed N          splitmix64 seed for input (default 42)
 *   --validate-only   compute exactly ONE pass, print fingerprint, exit
 *   --iterations N    timed passes over x (default 1)
 *   --threads N       worker threads over disjoint chunk ranges (default 1); reduction is idempotent,
 *                     so every pass re-reads the same N elems (bandwidth), and
 *                     the fingerprint equals the single-pass sum regardless of K
 *
 * Stdout:
 *   reduction-scalar_baseline-start size=<N> seed=<S> threads=<T>
 *   reduction-scalar_baseline-end fingerprint=<sum> wall_us=<int> \
 *       iterations=<K> threads=<T> elems=<N> bytes=<int>
 *
 * bytes = N * 4 * K (read-only). Fingerprint = the fp32-computed sum, fp64
 * printed. Validated vs the fp64 NumPy reference within 1e-3 relative tolerance
 * (reduction-order tolerant; decision R6).
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cpu_threads.h"

/* ── splitmix64 (bit-identical to reference.py + saxpy/matmul kernels) ── */

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

static void generate_input(int size, uint64_t seed, float* x) {
    uint64_t state = seed;
    for (int i = 0; i < size; i++) {
        x[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(&state)) - 1.0);
    }
}

/* ── Scalar reduction: single fp32 accumulator (dependent add chain) ── */
static float reduce_scalar(const float* x, int begin, int end) {
    float s = 0.0f;
    for (int i = begin; i < end; i++) {
        s += x[i];
    }
    return s;
}

/* ── Deterministic parallel reduction ──
 *
 * fp32 addition is NOT associative, so if the split depended on --threads the
 * fingerprint would too, and a 1-thread run could not be compared against a
 * 6-thread run. The array is therefore always cut into RED_CHUNKS chunks whose
 * boundaries depend only on `size`; workers take contiguous RANGES of chunks,
 * and the partials are folded in fixed chunk order below. Result is
 * bit-identical at every --threads value.
 *
 * Each worker folds into a stack-local array and publishes once at the end, so
 * the shared partials[] is not written in the inner loop (false sharing). */
#define RED_CHUNKS 64   /* upper bound; the live count comes from red_chunks() */

/* Chunk count derived from `size` ALONE — never from --threads, so the fold
 * order and therefore the fingerprint stay reproducible across thread counts.
 * One chunk per 64 KB keeps chunks large enough that per-chunk call overhead
 * stays negligible even when the working set fits in L1. */
static int red_chunks(int size) {
    int c = size / 16384;            /* 16384 floats = 64 KB */
    if (c > RED_CHUNKS) c = RED_CHUNKS;
    if (c < 1) c = 1;
    return c;
}

typedef struct {
    const float* x;
    int size;
    int reps;
    float partials[RED_CHUNKS];
} reduce_ctx;

static void reduce_worker(void* vctx, int tid, int nthreads) {
    reduce_ctx* c = (reduce_ctx*)vctx;
    int nchunks = red_chunks(c->size);
    int first, last;
    cpu_band(nchunks, 1, tid, nthreads, &first, &last);

    float local[RED_CHUNKS];
    for (int k = first; k < last; k++) local[k] = 0.0f;

    for (int it = 0; it < c->reps; it++) {
        for (int k = first; k < last; k++) {
            int begin, end;
            cpu_band(c->size, 16, k, nchunks, &begin, &end);
            local[k] = reduce_scalar(c->x, begin, end);
        }
        /* Keep every pass a real pass: without this the compiler may hoist the
         * whole computation out of the iteration loop. */
        __asm__ __volatile__("" ::: "memory");
    }
    for (int k = first; k < last; k++) c->partials[k] = local[k];
}

/* Fold in fixed chunk order — the step that makes the result reproducible. */
static float reduce_combine(const reduce_ctx* c) {
    int nchunks = red_chunks(c->size);
    float s = 0.0f;
    for (int k = 0; k < nchunks; k++) s += c->partials[k];
    return s;
}

static int parse_args(int argc, char** argv, int* size, uint64_t* seed,
                      bool* validate_only, int* iterations, int* threads) {
    *size = 4194304; /* 2^22 */
    *seed = 42;
    *validate_only = false;
    *iterations = 1;
    *threads = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            *size = atoi(argv[++i]);
            if (*size <= 0 || (*size % 16) != 0) {
                fprintf(stderr, "ERROR: --size must be > 0 and a multiple of 16\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            *seed = strtoull(argv[++i], NULL, 10);
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
    bool validate_only;
    if (parse_args(argc, argv, &size, &seed, &validate_only, &iterations,
                   &threads)) {
        return 1;
    }
    if (threads > red_chunks(size)) threads = red_chunks(size);

    fprintf(stdout, "reduction-scalar_baseline-start size=%d seed=%llu threads=%d\n",
            size, (unsigned long long)seed, threads);
    fflush(stdout);

    float* x = NULL;
    if (posix_memalign((void**)&x, 64, (size_t)size * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: posix_memalign failed\n");
        return 1;
    }
    generate_input(size, seed, x);

    int reps = validate_only ? 1 : iterations;

    reduce_ctx ctx;
    ctx.x = x;
    ctx.size = size;
    ctx.reps = reps;
    for (int k = 0; k < RED_CHUNKS; k++) ctx.partials[k] = 0.0f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    cpu_parallel(threads, reduce_worker, &ctx);
    float s = reduce_combine(&ctx);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    int64_t wall_us = (int64_t)((t1.tv_sec - t0.tv_sec) * 1000000LL +
                                (t1.tv_nsec - t0.tv_nsec) / 1000LL);
    int64_t bytes = (int64_t)size * (int64_t)sizeof(float) * reps;

    fprintf(stdout,
            "reduction-scalar_baseline-end fingerprint=%.6e wall_us=%lld "
            "iterations=%d threads=%d elems=%d bytes=%lld\n",
            (double)s, (long long)wall_us, reps, threads, size,
            (long long)bytes);
    fflush(stdout);

    free(x);
    return 0;
}
