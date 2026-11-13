/* bp_stress.c - Lab 3, Task 5.1: make the branch predictor fail as often as
 * you can.
 *
 * Build and push (inside the lab container, from the repository root):
 *     python scripts/new_kernel.py bp_stress --push
 * Measure:
 *     python profiler.py custom cpu --binary bp_stress --core prime \
 *         --counters raw-br-pred,raw-br-mis-pred
 *
 * The score is raw-br-mis-pred / raw-br-pred.
 *
 * Profiler parameters reach this program as flags: `-p size=N` arrives as
 * `--size N`. Add your own flags the same way.
 *
 * OUTPUT. The profiler reads the "key: value" lines printed at the end. Keep
 * mean_us (time per iteration) and fingerprint (a value computed from your
 * work, so the compiler cannot throw that work away).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
    long size = 1L << 20;     /* work per iteration */
    int iters = 300;          /* repeats; aim for a run of 200 ms or more */
    unsigned long long seed = 42;

    for (int i = 1; i < argc - 1; i++) {
        if      (!strcmp(argv[i], "--size"))       size  = atol(argv[++i]);
        else if (!strcmp(argv[i], "--iterations")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed"))       seed  = strtoull(argv[++i], NULL, 10);
    }
    if (size < 1 || iters < 1) {
        printf("status: FAILED\nerror: need --size >= 1 and --iterations >= 1\n");
        return 1;
    }
    uint64_t state = seed;
    uint64_t taken_count = 0;
    double t0 = now_us();
    for (int it = 0; it < iters; it++) {
#ifdef __clang__
        /* Amortize the predictable loop branch across four decisions. */
#pragma clang loop unroll_count(4)
#endif
        for (long i = 0; i < size; i++) {
            state = state * 6364136223846793005ULL + 1;
#if defined(__aarch64__)
            /* Skip the increment with one branch. Synchronize the instruction
             * stream at the join to limit speculation into later decisions. */
            __asm__ volatile(
                "tbz %[state], #63, 1f\n\t"
                "add %[count], %[count], #1\n"
                "1: isb"
                : [count] "+r" (taken_count)
                : [state] "r" (state));
#else
            if (state >> 63)
                taken_count++;
#endif
        }
    }
    double us = now_us() - t0;

    printf("status: OK\n");
    printf("size: %ld\niterations: %d\n", size, iters);
    printf("mean_us: %.1f\n", us / iters);     /* per iteration */
    printf("wall_us: %.1f\n", us);
    printf("flop: 0\n");
    printf("fingerprint: %llu\n", (unsigned long long)taken_count);
    return 0;
}
