/*
 * Snapdragon Profiler — convolution / cpu / int8_neon.
 *
 * INT8 direct convolution, int32 accumulate. The CPU counterpart to the cDSP
 * int8 kernel, and the fair baseline for it: both use their machine's 4-way
 * byte dot-product instruction, so neither side is handicapped.
 *
 *   NEON  vdotq_s32 : 16 bytes/instruction ->  4 int32 lanes
 *   HVX   vrmpy     : 128 bytes/instruction -> 32 int32 lanes
 *
 * That 8x width ratio is the thing being measured.
 *
 * Operand generation is byte-identical to dsp_conv_int8_main.c — the same
 * splitmix64 stream, drawn input-then-weights, each value reduced to int8 via
 * (u >> 24) & 0xFF reinterpreted as signed. Integer arithmetic is exact, so the
 * two checksums must agree EXACTLY. Any difference is a bug, not rounding.
 *
 * Requires -march=...+dotprod (SDOT is ARMv8.4; Oryon-2 is ARMv9).
 * Cin should be a multiple of 16; a scalar tail covers the remainder.
 */

#include <arm_neon.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_THREADS 16

static const uint64_t SM_INC = 0x9E3779B97F4A7C15ULL;
static const uint64_t SM1 = 0xBF58476D1CE4E5B9ULL;
static const uint64_t SM2 = 0x94D049BB133111EBULL;

static inline uint64_t sm_next(uint64_t* s) {
    *s += SM_INC;
    uint64_t z = *s;
    z = (z ^ (z >> 30)) * SM1;
    z = (z ^ (z >> 27)) * SM2;
    return z ^ (z >> 31);
}
static void fill_i8(int8_t* d, int64_t n, uint64_t* s) {
    for (int64_t i = 0; i < n; i++) d[i] = (int8_t)((sm_next(s) >> 24) & 0xFF);
}
static int64_t checksum_i32(const int32_t* o, int64_t n) {
    int64_t acc = 0;
    for (int64_t i = 0; i < n; i++) acc += (int64_t)o[i] * (int64_t)((i & 7) + 1);
    return acc;
}

/* One contiguous band of output rows. */
static void conv_int8_band(const int8_t* in, const int8_t* w, int32_t* out,
                           int H, int W, int Cin, int Cout, int K,
                           int oh_begin, int oh_end) {
    int pad = K / 2;
    int ic16 = Cin & ~15;   /* largest multiple of 16 <= Cin */
    for (int oh = oh_begin; oh < oh_end; oh++) {
        for (int ow = 0; ow < W; ow++) {
            int32_t* out_px = &out[((int64_t)oh * W + ow) * Cout];
            for (int oc = 0; oc < Cout; oc++) {
                int32x4_t vacc = vdupq_n_s32(0);
                int32_t tail = 0;
                for (int kh = 0; kh < K; kh++) {
                    int ih = oh + kh - pad;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow + kw - pad;
                        if (iw < 0 || iw >= W) continue;
                        const int8_t* in_px = &in[((int64_t)ih * W + iw) * Cin];
                        const int8_t* w_px  = &w[(((int64_t)oc * K + kh) * K + kw) * Cin];
                        int ic = 0;
                        for (; ic < ic16; ic += 16) {
                            vacc = vdotq_s32(vacc, vld1q_s8(&in_px[ic]), vld1q_s8(&w_px[ic]));
                        }
                        for (; ic < Cin; ic++)
                            tail += (int32_t)in_px[ic] * (int32_t)w_px[ic];
                    }
                }
                out_px[oc] = vaddvq_s32(vacc) + tail;
            }
        }
    }
}

typedef struct {
    const int8_t *in, *w;
    int32_t* out;
    int H, W, Cin, Cout, K;
    int oh_begin, oh_end;
} band_t;

static void* band_entry(void* arg) {
    band_t* b = (band_t*)arg;
    conv_int8_band(b->in, b->w, b->out, b->H, b->W, b->Cin, b->Cout, b->K,
                   b->oh_begin, b->oh_end);
    return NULL;
}

static void conv_int8(const int8_t* in, const int8_t* w, int32_t* out,
                      int H, int W, int Cin, int Cout, int K, int nthreads) {
    if (nthreads > H) nthreads = H;
    if (nthreads < 1) nthreads = 1;
    if (nthreads == 1) {
        conv_int8_band(in, w, out, H, W, Cin, Cout, K, 0, H);
        return;
    }

    band_t    bands[MAX_THREADS];
    pthread_t tids[MAX_THREADS];
    int       spawned[MAX_THREADS];

    int base = H / nthreads, rem = H % nthreads, next = 0;
    for (int t = 0; t < nthreads; t++) {
        int rows = base + (t < rem ? 1 : 0);
        bands[t] = (band_t){ in, w, out, H, W, Cin, Cout, K, next, next + rows };
        next += rows;
        spawned[t] = 0;
    }
    for (int t = 1; t < nthreads; t++)
        spawned[t] = (pthread_create(&tids[t], NULL, band_entry, &bands[t]) == 0);
    band_entry(&bands[0]);
    for (int t = 1; t < nthreads; t++) {
        if (spawned[t]) pthread_join(tids[t], NULL);
        else band_entry(&bands[t]);
    }
}

int main(int argc, char** argv) {
    int H = 128, W = 128, Cin = 256, Cout = 256, K = 3, iterations = 1, threads = 1;
    uint64_t seed = 42;
    bool validate_only = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--H") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--W") && i + 1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cin") && i + 1 < argc) Cin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cout") && i + 1 < argc) Cout = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--validate-only")) validate_only = true;
        else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else { fprintf(stderr, "ERROR: unknown arg: %s\n", argv[i]); return 1; }
    }
    if (threads < 1 || threads > MAX_THREADS) {
        fprintf(stderr, "ERROR: --threads must be 1..%d\n", MAX_THREADS); return 1;
    }

    printf("conv-int8_neon-start H=%d W=%d cin=%d cout=%d k=%d seed=%llu threads=%d\n",
           H, W, Cin, Cout, K, (unsigned long long)seed, threads);
    fflush(stdout);

    int64_t n_in  = (int64_t)H * W * Cin;
    int64_t n_w   = (int64_t)Cout * K * K * Cin;
    int64_t n_out = (int64_t)H * W * Cout;
    int8_t *in = NULL, *w = NULL;
    int32_t* out = NULL;
    if (posix_memalign((void**)&in, 64, n_in) ||
        posix_memalign((void**)&w, 64, n_w) ||
        posix_memalign((void**)&out, 64, n_out * sizeof(int32_t))) {
        fprintf(stderr, "ERROR: posix_memalign failed\n"); return 1;
    }
    uint64_t state = seed;
    fill_i8(in, n_in, &state);
    fill_i8(w, n_w, &state);

    int reps = validate_only ? 1 : iterations;

    volatile int64_t sink = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    for (int it = 0; it < reps; it++) {
        conv_int8(in, w, out, H, W, Cin, Cout, K, threads);
        sink += out[0];
        __asm__ __volatile__("" ::: "memory");
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    (void)sink;

    int64_t wall_us = (int64_t)((t1.tv_sec - t0.tv_sec) * 1000000LL +
                                (t1.tv_nsec - t0.tv_nsec) / 1000LL);
    int64_t ops   = (int64_t)2 * H * W * Cout * K * K * Cin * reps;
    int64_t bytes = (int64_t)(n_in + n_w + 4 * n_out) * reps;

    printf("conv-int8_neon-end checksum=%lld wall_us=%lld iterations=%d threads=%d "
           "H=%d W=%d cin=%d cout=%d k=%d ops=%lld bytes=%lld\n",
           (long long)checksum_i32(out, n_out), (long long)wall_us, reps, threads,
           H, W, Cin, Cout, K, (long long)ops, (long long)bytes);
    fflush(stdout);

    free(in); free(w); free(out);
    return 0;
}
