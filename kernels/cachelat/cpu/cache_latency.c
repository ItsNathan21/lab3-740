/*
 * Snapdragon Profiler — cachelat / cpu (cache hierarchy + prefetcher probe).
 *
 * Measures LOAD-TO-USE LATENCY, not bandwidth, by chasing a dependent pointer
 * chain: every access must complete before the next address is known, so the
 * core cannot overlap them. Latency exposes cache capacity far more sharply
 * than bandwidth does — a streaming read stays flat across the L1 boundary
 * because the hardware prefetcher hides the miss, whereas a chase cannot be
 * prefetched at all and steps cleanly at every level of the hierarchy.
 *
 * Two access patterns over the SAME buffer and the SAME number of accesses:
 *
 *   --pattern random      a random permutation cycle over cache lines. The next
 *                         address depends on the loaded value, so the prefetcher
 *                         has nothing to predict. This is the ruler used to
 *                         measure cache capacity.
 *   --pattern sequential  a cycle that walks lines in address order. Same
 *                         dependent chain, same instruction count, but now the
 *                         address stream is trivially predictable.
 *
 * The gap between the two IS the prefetcher's contribution, and the hwprf
 * counters attribute it.
 *
 * Both patterns visit every line exactly once per lap, so the working set and
 * the access count are identical — only the order differs.
 *
 * Argv contract:
 *   --kb N            working-set size in KiB (default 256)
 *   --accesses N      dependent loads to time (default 20000000)
 *   --pattern P       random | sequential (default random)
 *   --stride N        bytes between visited elements (default 64 = one line)
 *   --seed N          splitmix64 seed for the permutation (default 42)
 *   --hugepages       back the buffer with 2 MB pages via madvise(MADV_HUGEPAGE)
 *   --smallpages      force 4 KB pages via madvise(MADV_NOHUGEPAGE)
 *
 * `--hugepages` is the TLB experiment. Same working set, same access pattern,
 * same number of loads — but 512× fewer page-table entries to cover it. Any
 * latency difference is the TLB and nothing else, because the cache hierarchy
 * cannot tell what size page a line arrived under. It is the only clean way to
 * separate "the data does not fit in cache" from "the translations do not fit
 * in the TLB", and on this SoC those two boundaries are easy to confuse.
 *
 * `--smallpages` is the other half of that experiment, and on this board the
 * one that matters: the kernel runs transparent huge pages in "always" mode,
 * so a large buffer already gets 2 MB pages with no flag at all, and
 * --hugepages changes nothing. --smallpages opts the buffer out, so
 * `--smallpages` against the default is the 4 KB vs 2 MB comparison. The two
 * flags are mutually exclusive.
 *
 * Stdout:
 *   cachelat-start kb=<N> pattern=<P> stride=<N> lines=<N> seed=<S>
 *   cachelat-end fingerprint=<int> wall_us=<int> accesses=<N> kb=<N> \
 *       ns_per_access=<float> pattern=<P> lines=<N> bytes=<int>
 *
 * `bytes` counts one cache line per access (what the memory system actually
 * moves), so the wrapper's GB/s column reports achieved line-fill rate.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif
#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE 15
#endif
#define HUGE_PAGE_BYTES (2u * 1024u * 1024u)

/* ── splitmix64 (bit-identical to the other kernels) ── */

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

/* Build a single cycle visiting all `n` slots exactly once.
 *
 * A cycle (not just a shuffle) matters: it guarantees the chase never falls
 * into a short sub-loop that would sit in cache regardless of buffer size.
 * Sattolo's algorithm produces a permutation that is one n-cycle by
 * construction. */
static void build_cycle(uint32_t* next, uint32_t n, uint64_t seed, bool random) {
    uint32_t* order = (uint32_t*)malloc((size_t)n * sizeof(uint32_t));
    if (!order) {
        fprintf(stderr, "ERROR: malloc failed\n");
        exit(1);
    }
    for (uint32_t i = 0; i < n; i++) order[i] = i;

    if (random) {
        /* Sattolo: for i from n-1 down to 1, swap with j uniform in [0, i). */
        uint64_t st = seed;
        for (uint32_t i = n - 1; i >= 1; i--) {
            uint32_t j = (uint32_t)(splitmix64_next(&st) % i);
            uint32_t t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
    }
    /* order[] is the visit sequence; link each slot to the next one visited. */
    for (uint32_t i = 0; i < n; i++) next[order[i]] = order[(i + 1) % n];
    free(order);
}

int main(int argc, char** argv) {
    int kb = 256, stride = 64, hugepages = 0, smallpages = 0;
    long long accesses = 20000000;
    uint64_t seed = 42;
    const char* pattern = "random";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--kb") && i + 1 < argc) {
            kb = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--accesses") && i + 1 < argc) {
            accesses = atoll(argv[++i]);
        } else if (!strcmp(argv[i], "--pattern") && i + 1 < argc) {
            pattern = argv[++i];
        } else if (!strcmp(argv[i], "--stride") && i + 1 < argc) {
            stride = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--hugepages")) {
            hugepages = 1;
        } else if (!strcmp(argv[i], "--smallpages")) {
            smallpages = 1;
        } else {
            fprintf(stderr, "ERROR: unknown arg: %s\n", argv[i]);
            return 1;
        }
    }
    if (hugepages && smallpages) {
        fprintf(stderr, "ERROR: --hugepages and --smallpages are mutually exclusive\n");
        return 1;
    }
    bool random = !strcmp(pattern, "random");
    if (!random && strcmp(pattern, "sequential")) {
        fprintf(stderr, "ERROR: --pattern must be random or sequential\n");
        return 1;
    }
    if (kb <= 0 || stride < 4 || accesses <= 0) {
        fprintf(stderr, "ERROR: --kb > 0, --stride >= 4, --accesses > 0\n");
        return 1;
    }

    size_t bytes = (size_t)kb * 1024;
    uint32_t lines = (uint32_t)(bytes / (size_t)stride);
    if (lines < 2) {
        fprintf(stderr, "ERROR: working set too small for this stride\n");
        return 1;
    }
    uint32_t slots = (uint32_t)(bytes / sizeof(uint32_t));
    uint32_t step = (uint32_t)((size_t)stride / sizeof(uint32_t));

    printf("cachelat-start kb=%d pattern=%s stride=%d lines=%u hugepages=%d "
           "smallpages=%d seed=%llu\n",
           kb, pattern, stride, lines, hugepages, smallpages,
           (unsigned long long)seed);
    fflush(stdout);

    /* Align to the huge-page boundary when asked: a merely 4 KB-aligned region
     * usually cannot be promoted to 2 MB pages no matter what madvise says. */
    uint32_t* buf = NULL;
    size_t align = hugepages ? HUGE_PAGE_BYTES : 4096;
    size_t alloc = (size_t)slots * sizeof(uint32_t);
    if (hugepages) {
        alloc = (alloc + HUGE_PAGE_BYTES - 1) & ~(size_t)(HUGE_PAGE_BYTES - 1);
    }
    if (posix_memalign((void**)&buf, align, alloc) != 0) {
        fprintf(stderr, "ERROR: posix_memalign failed\n");
        return 1;
    }
    if (hugepages && madvise(buf, alloc, MADV_HUGEPAGE) != 0) {
        fprintf(stderr, "WARN: madvise(MADV_HUGEPAGE) failed; pages stay 4 KB\n");
    }
    /* Must happen before the memset below: once a range has been faulted in as
     * huge pages, opting out later does not split them. */
    if (smallpages && madvise(buf, alloc, MADV_NOHUGEPAGE) != 0) {
        fprintf(stderr, "WARN: madvise(MADV_NOHUGEPAGE) failed; THP may still apply\n");
    }
    memset(buf, 0, alloc);   /* fault every page in so promotion can happen */

    /* Build the cycle over line indices, then scatter it into the buffer so
     * that buf[line * step] holds the SLOT index of the next line. */
    uint32_t* next = (uint32_t*)malloc((size_t)lines * sizeof(uint32_t));
    if (!next) {
        fprintf(stderr, "ERROR: malloc failed\n");
        return 1;
    }
    build_cycle(next, lines, seed, random);
    for (uint32_t l = 0; l < lines; l++) buf[l * step] = next[l] * step;
    free(next);

    /* Warm the TLB and page tables so the timed loop measures cache, not
     * first-touch page faults. */
    volatile uint32_t warm = 0;
    for (uint32_t l = 0; l < lines; l++) warm += buf[l * step];
    (void)warm;

    uint32_t idx = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    for (long long i = 0; i < accesses; i++) {
        idx = buf[idx];              /* the whole benchmark: load-to-use chain */
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    int64_t wall_us = (int64_t)((t1.tv_sec - t0.tv_sec) * 1000000LL +
                                (t1.tv_nsec - t0.tv_nsec) / 1000LL);
    double ns = wall_us * 1000.0 / (double)accesses;

    /* One line moved per access is what the memory system sees. */
    long long moved = accesses * (long long)stride;

    printf("cachelat-end fingerprint=%u wall_us=%lld accesses=%lld kb=%d "
           "ns_per_access=%.4f pattern=%s lines=%u hugepages=%d smallpages=%d "
           "bytes=%lld\n",
           idx, (long long)wall_us, accesses, kb, ns, pattern, lines, hugepages,
           smallpages, moved);
    fflush(stdout);

    free(buf);
    return 0;
}
