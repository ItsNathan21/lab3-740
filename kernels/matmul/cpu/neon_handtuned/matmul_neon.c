/*
 * Snapdragon Profiler — matmul / cpu / neon_handtuned (Phase 3.5).
 *
 * Hand-tuned NEON 4×4 outer-product blocked fp32 matmul.
 * Inputs A and B are row-major and deterministic via splitmix64
 * from the user-supplied seed (bit-identical to
 * `kernels/matmul/cpu/reference.py` and the Eigen variant —
 * same seed → same A, B).
 *
 * Kernel structure: outer two loops walk 4×4 output tiles.
 * Inner loop walks K, computing 4 NEON FMAs per K step (one
 * per output row's 4-wide vector). Tile size is fixed at 4
 * because the v1 size 512 is divisible by 4 and we want a
 * trivially-correct kernel — no edge-case handling per
 * `DESIGN.md` M2.
 *
 * Argv contract (per `kernels/matmul/cpu/DESIGN.md` M3):
 *   --size N              square matrix size (default 512); must be
 *                         a multiple of 4 in v1
 *   --seed N              splitmix64 seed for inputs (default 42)
 *   --validate-only       compute once, print fingerprint, exit
 *   --iterations N        compute N times before printing (default 1)
 *   --threads N           worker threads over output-row bands (default 1)
 *   --pin                 bind worker t to the t-th CPU in the affinity mask
 *   --prime-weight W      give prime cores (6-7) a W-times larger band.
 *                         Requires --pin to be meaningful. 1.0 = even split.
 *
 * Stdout (per DESIGN.md M3):
 *   matmul-neon_handtuned-start size=512 seed=42 threads=<T>
 *   matmul-neon_handtuned-end fingerprint=<float> wall_us=<int> \
 *       iterations=<K> threads=<T> size=<N> flop=<int> bytes=<int>
 *
 * flop = 2*N^3*K and bytes = 3*N^2*4*K are TOTALS across iterations.
 * Threading splits the i loop into 4-row-aligned bands; each C tile is
 * computed by one worker in the same k order, so the fingerprint is
 * bit-identical at any --threads value.
 *
 * Fingerprint = sum(C) + frobenius_norm(C). Validated against the
 * NumPy reference within 1e-3 relative tolerance per M4/Q1 and
 * against the eigen_baseline variant within the same tolerance.
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

/* ── splitmix64 (bit-identical to reference.py + matmul_eigen) ── */

static const uint64_t SPLITMIX_INC = 0x9E3779B97F4A7C15ULL;
static const uint64_t SPLITMIX_M1  = 0xBF58476D1CE4E5B9ULL;
static const uint64_t SPLITMIX_M2  = 0x94D049BB133111EBULL;

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

static void generate_inputs(int size, uint64_t seed,
                            float* A, float* B) {
    uint64_t state = seed;
    int n = size * size;
    for (int i = 0; i < n; i++) {
        uint64_t v = splitmix64_next(&state);
        A[i] = (float)(2.0 * u64_to_unit_double(v) - 1.0);
    }
    for (int i = 0; i < n; i++) {
        uint64_t v = splitmix64_next(&state);
        B[i] = (float)(2.0 * u64_to_unit_double(v) - 1.0);
    }
}

/* ── 4×4 NEON-blocked matmul ──
 *
 * Computes C = A * B, all matrices row-major, all dimensions =
 * `size` (must be multiple of 4 — v1 contract).
 *
 * Inner loop: at each K step, load b_kj = (B[k,j..j+3]) (4 floats
 * contiguous in row-major B), then FMA into 4 accumulator
 * vectors c0..c3, scaled by A[i+0..3, k] respectively.
 *
 * Register pressure: 4 acc vectors + 1 b vector + 4 a scalars per
 * tile. Well within ARMv8's 32 NEON registers.
 */
static void matmul_neon(const float* A, const float* B, float* C,
                         int size, int i_begin, int i_end) {
    for (int i = i_begin; i < i_end; i += 4) {
        for (int j = 0; j < size; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);
            float32x4_t c2 = vdupq_n_f32(0.0f);
            float32x4_t c3 = vdupq_n_f32(0.0f);
            for (int k = 0; k < size; k++) {
                float32x4_t b_kj = vld1q_f32(&B[k * size + j]);
                /* vfmaq_n_f32(acc, vec, scalar): acc + vec * scalar */
                c0 = vfmaq_n_f32(c0, b_kj, A[(i + 0) * size + k]);
                c1 = vfmaq_n_f32(c1, b_kj, A[(i + 1) * size + k]);
                c2 = vfmaq_n_f32(c2, b_kj, A[(i + 2) * size + k]);
                c3 = vfmaq_n_f32(c3, b_kj, A[(i + 3) * size + k]);
            }
            vst1q_f32(&C[(i + 0) * size + j], c0);
            vst1q_f32(&C[(i + 1) * size + j], c1);
            vst1q_f32(&C[(i + 2) * size + j], c2);
            vst1q_f32(&C[(i + 3) * size + j], c3);
        }
    }
}

/* One row band per worker for ALL iterations: a single spawn per run.
 * Bands write disjoint rows of C and only read A and B. */
typedef struct {
    const float* A;
    const float* B;
    float* C;
    int size;
    int reps;
} matmul_ctx;

static void matmul_worker(void* vctx, int tid, int nthreads) {
    matmul_ctx* c = (matmul_ctx*)vctx;
    int i_begin, i_end;
    cpu_band(c->size, 4, tid, nthreads, &i_begin, &i_end);
    for (int it = 0; it < c->reps; it++) {
        matmul_neon(c->A, c->B, c->C, c->size, i_begin, i_end);
    }
}

static double compute_fingerprint(const float* C, int n) {
    /* sum(C) + frobenius_norm(C). Both done in fp64 to limit
     * accumulation error in the fingerprint itself (so two
     * variants with bit-different C matrices still compare
     * meaningfully). */
    double s = 0.0;
    double sq = 0.0;
    for (int i = 0; i < n; i++) {
        double x = (double)C[i];
        s += x;
        sq += x * x;
    }
    return s + sqrt(sq);
}

static int parse_args(int argc, char** argv,
                       int* size, uint64_t* seed,
                       bool* validate_only, int* iterations, int* threads,
                       int* pin, double* prime_weight) {
    *size = 512;
    *seed = 42;
    *validate_only = false;
    *iterations = 1;
    *threads = 1;
    *pin = 0;
    *prime_weight = 1.0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            *size = atoi(argv[++i]);
            if (*size <= 0 || (*size % 4) != 0) {
                fprintf(stderr, "ERROR: --size must be > 0 and a multiple of 4 (v1)\n");
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
        } else if (strcmp(argv[i], "--pin") == 0) {
            *pin = 1;
        } else if (strcmp(argv[i], "--prime-weight") == 0 && i + 1 < argc) {
            *prime_weight = strtod(argv[++i], NULL);
            if (*prime_weight <= 0.0) {
                fprintf(stderr, "ERROR: --prime-weight must be > 0\n");
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
    int size, iterations, threads, pin;
    double prime_weight;
    uint64_t seed;
    bool validate_only;
    if (parse_args(argc, argv, &size, &seed, &validate_only, &iterations,
                   &threads, &pin, &prime_weight)) {
        return 1;
    }
    if (threads > size / 4) threads = size / 4;   /* >= one 4-row tile each */

    cpu_split_config(pin, prime_weight);
    fprintf(stdout, "matmul-neon_handtuned-start size=%d seed=%llu threads=%d "
            "pin=%d prime_weight=%.3f\n",
            size, (unsigned long long)seed, threads, pin, prime_weight);
    fflush(stdout);

    int n = size * size;
    /* 64-byte aligned alloc for NEON-friendly access (16-byte
     * vectors fit cleanly). aligned_alloc was added to Android
     * Bionic at API 28; we target 24, so use posix_memalign
     * which has been there since the beginning. */
    float *A = NULL, *B = NULL, *C = NULL;
    if (posix_memalign((void**)&A, 64, n * sizeof(float)) != 0 ||
        posix_memalign((void**)&B, 64, n * sizeof(float)) != 0 ||
        posix_memalign((void**)&C, 64, n * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: posix_memalign failed\n");
        return 1;
    }
    generate_inputs(size, seed, A, B);

    int reps = validate_only ? 1 : iterations;
    matmul_ctx ctx = { A, B, C, size, reps };

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    cpu_parallel(threads, matmul_worker, &ctx);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    int64_t wall_us = (int64_t)((t1.tv_sec - t0.tv_sec) * 1000000LL
                                 + (t1.tv_nsec - t0.tv_nsec) / 1000LL);

    /* Totals across all iterations, matching the wrapper's flop_is_total. */
    int64_t flop = (int64_t)2 * size * size * size * reps;
    int64_t bytes = (int64_t)3 * size * size * (int64_t)sizeof(float) * reps;

    double fp = compute_fingerprint(C, n);
    fprintf(stdout,
            "matmul-neon_handtuned-end fingerprint=%.6e wall_us=%lld "
            "iterations=%d threads=%d pin=%d prime_weight=%.3f size=%d "
            "flop=%lld bytes=%lld\n",
            fp, (long long)wall_us, reps, threads, pin, prime_weight, size,
            (long long)flop, (long long)bytes);
    fflush(stdout);

    free(A);
    free(B);
    free(C);
    return 0;
}
