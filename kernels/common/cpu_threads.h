/*
 * Snapdragon Profiler — fork/join helper shared by the threaded CPU kernels.
 *
 * Threading model: spawn ONCE, outside the iteration loop. Each worker runs the
 * full `--iterations` loop over its own slice. The alternative — fork/join per
 * iteration — puts a pthread_create (~20-50 us) in front of every pass, which at
 * SAXPY's ~660 us/pass would show up as 25-40% overhead and confound exactly the
 * core-scaling measurement these kernels exist to support.
 *
 * This is only safe because the workloads have no cross-iteration dependency:
 * every pass recomputes the same thing from the same inputs, and workers touch
 * disjoint slices. A workload with a loop-carried dependency would need a
 * barrier per iteration instead.
 *
 * Determinism: slice boundaries depend only on (total, grain, nthreads), never
 * on scheduling. Combined with a fixed reduction order (see reduction kernels),
 * results are bit-identical at any --threads value.
 */

#ifndef SNAPDRAGON_PROFILER_CPU_THREADS_H
#define SNAPDRAGON_PROFILER_CPU_THREADS_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE     /* pthread_setaffinity_np, sched_getaffinity */
#endif

#include <pthread.h>
#include <sched.h>

#define CPU_MAX_THREADS 8   /* SM8750 has 8 cores: 0-5 gold, 6-7 prime */
#define CPU_PRIME_FIRST 6   /* cores 6-7 are the prime cluster (variant 0x3) */

typedef void (*cpu_worker_fn)(void* ctx, int tid, int nthreads);

typedef struct {
    cpu_worker_fn fn;
    void* ctx;
    int tid;
    int nthreads;
} cpu_task_t;

/* ── Placement and work-split policy ───────────────────────────────────────
 *
 * Default: the OS places threads anywhere in the process affinity mask, and
 * every thread gets an equal band. That is the right default and it is what
 * every measurement in the lab uses unless a task says otherwise.
 *
 * With pinning ON, worker t is bound to the t-th CPU in the mask, so the
 * mapping from thread to core is known rather than whatever the scheduler
 * chose. That is what makes a WEIGHTED split meaningful: on a mask spanning
 * both clusters, equal bands give the faster prime cores less work than they
 * could do, and they idle waiting for gold to finish. Setting a prime weight
 * above 1.0 hands them a proportionally larger band.
 *
 * Both are opt-in and both are recorded in the kernel's stdout line. */

static int cpu_pin_enabled = 0;
static double cpu_prime_weight = 1.0;
static double cpu_thread_weight[CPU_MAX_THREADS];
static int cpu_thread_cpu[CPU_MAX_THREADS];
static int cpu_weights_ready = 0;

static inline void cpu_split_config(int pin, double prime_weight) {
    cpu_pin_enabled = pin;
    cpu_prime_weight = prime_weight > 0.0 ? prime_weight : 1.0;
}

/* CPUs this process is allowed to run on, in ascending order. */
static inline int cpu_affinity_list(int* out, int max) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return 0;
    int n = 0;
    for (int c = 0; c < CPU_MAX_THREADS && n < max; c++) {
        if (CPU_ISSET(c, &set)) out[n++] = c;
    }
    return n;
}

/* Decide thread -> cpu and thread -> weight before any worker starts. */
static inline void cpu_prepare_split(int nthreads) {
    int allowed[CPU_MAX_THREADS];
    int n = cpu_affinity_list(allowed, CPU_MAX_THREADS);
    for (int t = 0; t < nthreads; t++) {
        int cpu = (n > 0) ? allowed[t % n] : -1;
        cpu_thread_cpu[t] = cpu;
        cpu_thread_weight[t] =
            (cpu >= CPU_PRIME_FIRST) ? cpu_prime_weight : 1.0;
    }
    cpu_weights_ready = 1;
}

/* Bind the CALLING thread to one CPU.
 *
 * Uses sched_setaffinity with pid 0 rather than pthread_setaffinity_np: on
 * Linux, pid 0 means "the calling thread", and Bionic did not export
 * pthread_setaffinity_np until API 36 — linking against it fails on this NDK
 * with `undefined symbol`. sched_setaffinity has been there since the start. */
static inline void cpu_pin_self(int tid) {
    if (!cpu_pin_enabled || cpu_thread_cpu[tid] < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_thread_cpu[tid], &set);
    sched_setaffinity(0, sizeof(set), &set);
}

static void* cpu_task_entry(void* arg) {
    cpu_task_t* t = (cpu_task_t*)arg;
    cpu_pin_self(t->tid);
    t->fn(t->ctx, t->tid, t->nthreads);
    return NULL;
}

/* Run fn(ctx, t, nthreads) for t in [0, nthreads).
 *
 * Worker 0 runs on the CALLING thread, so nthreads workers occupy nthreads
 * cores rather than nthreads+1. nthreads==1 calls fn directly with no pthread
 * involvement at all, so the single-threaded path costs exactly what it did
 * before threading was added. A failed pthread_create runs that worker inline —
 * slower, but still correct. */
static inline void cpu_parallel(int nthreads, cpu_worker_fn fn, void* ctx) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > CPU_MAX_THREADS) nthreads = CPU_MAX_THREADS;
    cpu_prepare_split(nthreads);
    if (nthreads == 1) {
        cpu_pin_self(0);
        fn(ctx, 0, 1);
        return;
    }

    cpu_task_t tasks[CPU_MAX_THREADS];
    pthread_t tids[CPU_MAX_THREADS];
    int spawned[CPU_MAX_THREADS];

    for (int t = 0; t < nthreads; t++) {
        tasks[t].fn = fn;
        tasks[t].ctx = ctx;
        tasks[t].tid = t;
        tasks[t].nthreads = nthreads;
        spawned[t] = 0;
    }
    for (int t = 1; t < nthreads; t++) {
        spawned[t] = (pthread_create(&tids[t], NULL, cpu_task_entry, &tasks[t]) == 0);
    }
    cpu_task_entry(&tasks[0]);
    for (int t = 1; t < nthreads; t++) {
        if (spawned[t]) {
            pthread_join(tids[t], NULL);
        } else {
            cpu_task_entry(&tasks[t]);
        }
    }
}

/* Split [0, total) into nthreads contiguous bands, each a multiple of `grain`.
 *
 * `grain` keeps SIMD alignment intact: 4 for a 4-lane NEON body, 16 for the
 * 4-accumulator reduction body, 1 for scalar. The remainder is spread one unit
 * per band across the low-numbered bands, so no band is ever more than one grain
 * unit larger than another. */
static inline void cpu_band(int total, int grain, int tid, int nthreads,
                            int* begin, int* end) {
    int units = total / grain;

    /* Weighted path: band size proportional to the weight of the core this
     * thread runs on. Boundaries are computed by prefix sum over ALL threads,
     * so the bands still tile [0, total) exactly with no gap or overlap, and
     * every thread computes the same boundaries independently. */
    if (cpu_weights_ready && cpu_prime_weight != 1.0 && nthreads > 1) {
        double sum = 0.0;
        for (int t = 0; t < nthreads; t++) sum += cpu_thread_weight[t];
        double before = 0.0;
        for (int t = 0; t < tid; t++) before += cpu_thread_weight[t];
        int first = (int)((before / sum) * units + 0.5);
        int last = (tid == nthreads - 1)
                       ? units
                       : (int)(((before + cpu_thread_weight[tid]) / sum) * units + 0.5);
        if (first < 0) first = 0;
        if (last > units) last = units;
        if (last < first) last = first;
        *begin = first * grain;
        *end = last * grain;
        return;
    }

    int base = units / nthreads;
    int rem = units % nthreads;
    int first = tid * base + (tid < rem ? tid : rem);
    int count = base + (tid < rem ? 1 : 0);
    *begin = first * grain;
    *end = (first + count) * grain;
}

#endif /* SNAPDRAGON_PROFILER_CPU_THREADS_H */
