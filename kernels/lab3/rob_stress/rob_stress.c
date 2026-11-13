/* rob_stress.c - Lab 3, Task 5.3: keep the backend stalled for as many cycles
 * as you can.
 *
 * Build and push (inside the lab container, from the repository root):
 *     python scripts/new_kernel.py rob_stress --push
 * Measure:
 *     python profiler.py custom cpu --binary rob_stress --core prime --counters stalls
 *
 * The score is backend_stall_pct. Each load in the pointer chase supplies
 * the index for the next load.
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
    long mb = 32;             /* buffer size in MiB */
    int iters = 4;            /* near-full traversals of the buffer */
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
    const size_t bytes_per_mib = 1u << 20;
    if ((uintmax_t)mb > SIZE_MAX / bytes_per_mib) {
        printf("status: FAILED\nerror: buffer size too large\n");
        return 1;
    }

    size_t bytes = (size_t)mb * bytes_per_mib;
    size_t *next = malloc(bytes);
    if (!next) { printf("status: FAILED\nerror: allocation\n"); return 1; }
    size_t n = bytes / sizeof *next;

    /* Splice each new index after a pseudorandom member of the same ring. */
    uint64_t state = seed;
    next[0] = 0;
    for (size_t i = 1; i < n; i++) {
        state = state * 6364136223846793005ULL + 1;
        size_t after = (state ^ (state >> 32)) % i;
        next[i] = next[after];
        next[after] = i;
    }

    size_t index = 0;
    double t0 = now_us();
    for (int it = 0; it < iters; it++) {
        /* Stop one link short of a full lap so the fingerprint advances. */
        for (size_t i = 1; i < n; i++)
            index = next[index];
    }
    double us = now_us() - t0;

    printf("status: OK\n");
    printf("mb: %ld\niterations: %d\n", mb, iters);
    printf("mean_us: %.1f\n", us / iters);     /* per iteration */
    printf("wall_us: %.1f\n", us);
    printf("flop: 0\n");
    printf("fingerprint: %zu\n", index);
    free(next);
    return 0;
}
