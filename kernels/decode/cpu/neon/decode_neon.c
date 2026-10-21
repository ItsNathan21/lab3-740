/* decode_neon.c — weight-stationary GEMM: C[S x N] = A[S x K] * W[K x N]
 *
 * The transformer projection/FFN shape with the token count S exposed. That one
 * parameter walks the workload across the roofline ridge:
 *
 *   S = 1     decode.  Each weight is read once and used once (GEMV).
 *             AI = S/2 = 0.5 FLOP/byte of weight traffic. DRAM-bound.
 *   S = 128   prefill. Each weight is reused across the token batch.
 *             AI = 64. Compute-bound.
 *
 * Same weights, same kernel. Only the batch changes, and the limiting resource
 * changes with it -- which is why single-stream LLM decode is bandwidth-bound
 * on hardware with enormous TOPS, and why batching is the lever.
 *
 *   predicted decode ceiling:  tokens/s <= DRAM bandwidth / weight bytes
 *
 * Blocking. A 4x16 register tile of C is held in 16 NEON accumulators across
 * the whole k loop, so C is written to memory exactly once. Two earlier
 * versions got this wrong and are worth recording:
 *   - token loop outside the k loop re-streamed W once per token, so the reuse
 *     the batch was supposed to buy never happened;
 *   - C accumulated in memory meant a load+store per FMA, and the kernel hit
 *     its own load/store ceiling (~43% of CPU peak) at every batch size, giving
 *     a flat curve that looked like a chip property and was not.
 * The loop order and where the accumulators live are the whole experiment.
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SB 4        /* tokens per register tile */
#define NB 16       /* output columns per register tile (4 NEON vectors) */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static unsigned long long rng_state;
static float next_f(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((rng_state >> 40) / 16777216.0) - 0.5f;
}

int main(int argc, char **argv) {
    int S = 1, K = 1024, N = 1024, iters = 0;
    unsigned long long seed = 42;

    for (int i = 1; i < argc - 1; i++) {
        if      (!strcmp(argv[i], "--tokens"))     S     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k"))          K     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n"))          N     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iterations")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed"))       seed  = strtoull(argv[++i], NULL, 10);
    }
    if (S < 1 || K < 1 || N < 1) {
        printf("status: FAILED\nerror: --tokens/--k/--n must be >= 1\n");
        return 1;
    }
    /* Default the pass count so every batch size runs for roughly the same
       wall time, ~150 ms. A shorter run at small S finishes before the
       governor has ramped, and then the number measures DVFS state rather than
       the kernel: at S=1 a 5 ms run reported 11 GFLOP/s against 32 with the
       clock pinned. Pin the clock for anything you intend to quote. */
    if (iters <= 0) { iters = 2048 / S; if (iters < 1) iters = 1; }

    rng_state = seed ? seed : 1;
    float *A = malloc((size_t)S * K * sizeof(float));
    float *W = malloc((size_t)K * N * sizeof(float));
    float *C = malloc((size_t)S * N * sizeof(float));
    if (!A || !W || !C) { printf("status: FAILED\nerror: allocation\n"); return 1; }
    for (size_t i = 0; i < (size_t)S * K; i++) A[i] = next_f();
    for (size_t i = 0; i < (size_t)K * N; i++) W[i] = next_f();

    double t0 = now_us();
    for (int it = 0; it < iters; it++) {
        for (int s0 = 0; s0 < S; s0 += SB) {
            const int sb = (S - s0 < SB) ? (S - s0) : SB;
            for (int n0 = 0; n0 < N; n0 += NB) {
                const int nb = (N - n0 < NB) ? (N - n0) : NB;

                if (nb == NB) {
                    /* Fast path: the whole 4x16 tile lives in registers for the
                       entire k loop. W streams past once; C is stored once. */
                    float32x4_t acc[SB][4];
                    for (int s = 0; s < sb; s++)
                        for (int v = 0; v < 4; v++) acc[s][v] = vdupq_n_f32(0.0f);

                    for (int k = 0; k < K; k++) {
                        const float *w = W + (size_t)k * N + n0;
                        const float32x4_t w0 = vld1q_f32(w);
                        const float32x4_t w1 = vld1q_f32(w + 4);
                        const float32x4_t w2 = vld1q_f32(w + 8);
                        const float32x4_t w3 = vld1q_f32(w + 12);
                        for (int s = 0; s < sb; s++) {
                            const float32x4_t a = vdupq_n_f32(A[(size_t)(s0 + s) * K + k]);
                            acc[s][0] = vfmaq_f32(acc[s][0], a, w0);
                            acc[s][1] = vfmaq_f32(acc[s][1], a, w1);
                            acc[s][2] = vfmaq_f32(acc[s][2], a, w2);
                            acc[s][3] = vfmaq_f32(acc[s][3], a, w3);
                        }
                    }
                    for (int s = 0; s < sb; s++) {
                        float *c = C + (size_t)(s0 + s) * N + n0;
                        vst1q_f32(c,      acc[s][0]); vst1q_f32(c + 4,  acc[s][1]);
                        vst1q_f32(c + 8,  acc[s][2]); vst1q_f32(c + 12, acc[s][3]);
                    }
                } else {
                    /* Ragged tail. */
                    for (int s = 0; s < sb; s++)
                        for (int n = 0; n < nb; n++) {
                            float sum = 0.0f;
                            for (int k = 0; k < K; k++)
                                sum += A[(size_t)(s0 + s) * K + k] * W[(size_t)k * N + n0 + n];
                            C[(size_t)(s0 + s) * N + n0 + n] = sum;
                        }
                }
            }
        }
    }
    double t1 = now_us();

    double fp = 0.0;
    for (size_t i = 0; i < (size_t)S * N; i++) fp += C[i];

    const double us     = t1 - t0;
    const double flop   = 2.0 * S * K * N * iters;
    const double wbytes = (double)K * N * sizeof(float) * iters;   /* W read once per pass */

    printf("status: OK\n");
    printf("tokens: %d\nk: %d\nn: %d\niterations: %d\nseed: %llu\n",
           S, K, N, iters, (unsigned long long)seed);
    printf("mean_us: %.1f\nwall_us: %.1f\n", us / iters, us);
    /* Per pass, to match mean_us. The harness takes mean_us as compute_us and
       divides the printed flop by it, so totals here would inflate by iters. */
    printf("flop: %.0f\nbytes: %.0f\n", flop / iters, wbytes / iters);
    printf("weight_mb: %.2f\n", (double)K * N * sizeof(float) / (1024.0 * 1024.0));
    printf("arithmetic_intensity: %.3f\n", flop / wbytes);
    printf("gflops: %.2f\n", flop / (us * 1e-6) / 1e9);
    printf("weight_gbps: %.2f\n", wbytes / (us * 1e-6) / 1e9);
    printf("tokens_per_s: %.1f\n", (double)S * iters / (us * 1e-6));
    printf("fingerprint: %.6f\n", fp);

    free(A); free(W); free(C);
    return 0;
}
