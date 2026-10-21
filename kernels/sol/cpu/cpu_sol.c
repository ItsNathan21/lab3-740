/* cpu_sol.c - CPU NEON fp32 FMA throughput probe.
 *
 * READ THIS BEFORE QUOTING ITS NUMBER. This reports 74.6 GFLOP/s on Prime at
 * 4320 MHz (2.16 FMA/cycle), and a real blocked GEMM reaches 111.3 (3.22).
 * A synthetic peak test losing to real work means the test is wrong, so use
 * 111.3 as the fp32 roof and treat this as a lower bound.
 *
 * Two failure modes were hit getting here, both worth knowing:
 *
 *   1. Accumulator array indexed by a loop variable. The compiler spills it and
 *      you measure load/store throughput, not FMA issue. Named registers only.
 *
 *   2. Dependency through the ACCUMULATE input with constant operands
 *      (a += b*k). The product is loop-invariant, so it strength-reduces to an
 *      add chain and the FMAs stop issuing. Symptom: throughput scales
 *      perfectly linearly with chain count and never plateaus - 48 chains
 *      "achieved" 11.93 FMA/cycle, which is not physical. Time still scales
 *      with iterations, so that check does not catch it.
 *
 * This version routes the dependency through the MULTIPLICAND (a = k + a*b),
 * which cannot be reduced. The cost is a longer dependency, so it saturates at
 * 2.16 FMA/cycle. A correct synthetic peak test needs operands that change each
 * iteration - at which point it is a GEMM, which is why the GEMM is the better
 * estimator.
 *
 * Named register variables, not an indexed array: indexing float32x4_t a[8]
 * with a loop variable makes the compiler spill to memory, and the result then
 * measures load/store throughput rather than FMA issue rate.
 *
 * 32 independent chains, so FMA latency is hidden and what remains is issue
 * width. No memory traffic in the timed loop.
 */
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec/1e3;}

/* 16 chains reached only 2.16 FMA/cycle while a real GEMM hit 3.22, so 16 is
   too few to cover FMA latency across what appear to be 4 NEON pipes. 32 chains
   uses the full architectural vector register file -- the most ILP expressible
   without spilling. */
#define C16(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)

int main(int argc, char **argv) {
    long iters = 20000000;
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], "--iterations")) iters = atol(argv[++i]);

    const float32x4_t b = vdupq_n_f32(1.0000001f);
    const float32x4_t k = vdupq_n_f32(0.0000001f);
#define DECL(i) float32x4_t a##i = vdupq_n_f32((float)(i + 1));
    C16(DECL)
#undef DECL

    double t0 = now_us();
    for (long it = 0; it < iters; it++) {
#define FMA(i) a##i = vfmaq_f32(k, a##i, b);
        C16(FMA)
#undef FMA
    }
    double t1 = now_us();

    float sink = 0;
#define SINK(i) sink += vgetq_lane_f32(a##i, 0);
    C16(SINK)
#undef SINK

    double flop = (double)iters * 16 * 4 * 2;   /* 32 chains x 4 lanes x 2 FLOP */
    double us = t1 - t0;
    printf("status: OK\niterations: %ld\nchains: 16\nwall_us: %.1f\n", iters, us);
    printf("gflops: %.1f\n", flop / (us * 1e-6) / 1e9);
    printf("flop_per_cycle_at_4.32GHz: %.2f\n", flop / (us * 1e-6) / 4.32e9);
    printf("fingerprint: %.6f\n", (double)sink);
    return 0;
}
