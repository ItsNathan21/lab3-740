/* rob_stress.c - Lab 3, Task 5.3: keep the backend stalled for as many cycles
 * as you can.
 *
 * Build and push (inside the lab container, from the repository root):
 *     python scripts/new_kernel.py rob_stress --push
 * Measure:
 *     python profiler.py custom cpu --binary rob_stress --core prime --counters stalls
 *
 * Your score is backend_stall_pct. As shipped, this file sums an array that
 * fits in cache, which keeps the backend busy rather than stalled. Replace
 * the marked block with your own code. Read the compiler PSA first: clang
 * will happily delete code whose result nobody uses.
 *
 * Profiler parameters reach this program as flags: `-p mb=N` arrives as
 * `--mb N`. Add your own flags the same way.
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
    long mb = 1;              /* buffer size in MB */
    int iters = 15000;        /* repeats; aim for a run of 200 ms or more */
    unsigned long long seed = 42;

    for (int i = 1; i < argc - 1; i++) {
        if      (!strcmp(argv[i], "--mb"))         mb    = atol(argv[++i]);
        else if (!strcmp(argv[i], "--iterations")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed"))       seed  = strtoull(argv[++i], NULL, 10);
    }
    if (mb < 1 || iters < 1) {
        printf("status: FAILED\nerror: need --mb >= 1 and --iterations >= 1\n");
        return 1;
    }
    (void)seed;   /* use it if your program needs random numbers */

    size_t n = (size_t)mb << 20 >> 3;        /* 8-byte elements */
    uint64_t *buf = malloc(n * sizeof *buf);
    if (!buf) { printf("status: FAILED\nerror: allocation\n"); return 1; }
    for (size_t i = 0; i < n; i++) buf[i] = i;

    uint64_t acc = 0;
    double t0 = now_us();
    for (int it = 0; it < iters; it++) {
        /* ------------------------------------------------------------------
         * YOUR CODE GOES HERE. The placeholder below is a plain sum: loads
         * that do not depend on each other, from a buffer that fits in cache.
         * ---------------------------------------------------------------- */
        for (size_t i = 0; i < n; i++)
            acc += buf[i];
    }
    double us = now_us() - t0;

    printf("status: OK\n");
    printf("mb: %ld\niterations: %d\n", mb, iters);
    printf("mean_us: %.1f\n", us / iters);     /* per iteration */
    printf("wall_us: %.1f\n", us);
    printf("flop: 0\n");
    printf("fingerprint: %llu\n", (unsigned long long)acc);
    free(buf);
    return 0;
}
