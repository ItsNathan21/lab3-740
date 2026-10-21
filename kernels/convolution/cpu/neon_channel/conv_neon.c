/*
 * Snapdragon Profiler — convolution / cpu / neon_channel (roofline mid-AI point).
 *
 * out = conv2d(in, W), dense KxK, stride 1, "same" zero padding, NHWC in/out,
 * OHWI weights. Same math as the scalar baseline; the inner reduction over the
 * input-channel dim (ic — contiguous in NHWC/OHWI) is vectorized with NEON FMA
 * (vfmaq_f32) into a 4-lane accumulator, horizontally reduced per output element.
 * AI ≈ (K²/8)·C for Cin=Cout=C; default Cin=Cout=8, K=3, 256² → AI≈18.
 *
 * Inputs + weights are deterministic via splitmix64 (bit-identical to
 * kernels/convolution/reference.py and the scalar variant). See that file / the
 * scalar variant header for the layout + draw-order + argv contract (identical).
 *
 * v1 requires Cin a multiple of 4 (the sweep uses C ∈ {4,8,16,32,64}); a scalar
 * tail handles any remainder for safety.
 */

#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_THREADS 16

static const uint64_t SPLITMIX_INC = 0x9E3779B97F4A7C15ULL;
static const uint64_t SPLITMIX_M1 = 0xBF58476D1CE4E5B9ULL;
static const uint64_t SPLITMIX_M2 = 0x94D049BB133111EBULL;

static inline uint64_t splitmix64_next(uint64_t* state) {
    *state += SPLITMIX_INC;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * SPLITMIX_M1;
    z = (z ^ (z >> 27)) * SPLITMIX_M2;
    return z ^ (z >> 31);
}
static inline double u64_to_unit_double(uint64_t x) {
    return (double)(x >> 11) * (1.0 / (double)(1ULL << 53));
}
static void fill_stream(float* dst, int64_t n, uint64_t* state) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(state)) - 1.0);
    }
}

/* NEON conv over a contiguous band of output rows [oh_begin, oh_end).
 * Output rows are independent — a band reads input rows and read-only weights
 * and writes only its own output rows — so bands may run concurrently with no
 * synchronization, and the result is bit-identical to the serial loop. */
static void conv_neon_band(const float* in, const float* w, float* out,
                           int H, int W, int Cin, int Cout, int K,
                           int oh_begin, int oh_end) {
    int pad = K / 2;
    int ic4 = Cin & ~3;   /* largest multiple of 4 ≤ Cin */
    for (int oh = oh_begin; oh < oh_end; oh++) {
        for (int ow = 0; ow < W; ow++) {
            float* out_px = &out[((int64_t)oh * W + ow) * Cout];
            for (int oc = 0; oc < Cout; oc++) {
                float32x4_t vacc = vdupq_n_f32(0.0f);
                float tail = 0.0f;
                for (int kh = 0; kh < K; kh++) {
                    int ih = oh + kh - pad;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow + kw - pad;
                        if (iw < 0 || iw >= W) continue;
                        const float* in_px = &in[((int64_t)ih * W + iw) * Cin];
                        const float* w_px  = &w[(((int64_t)oc * K + kh) * K + kw) * Cin];
                        int ic = 0;
                        for (; ic < ic4; ic += 4) {
                            vacc = vfmaq_f32(vacc, vld1q_f32(&in_px[ic]), vld1q_f32(&w_px[ic]));
                        }
                        for (; ic < Cin; ic++) tail += in_px[ic] * w_px[ic];
                    }
                }
                out_px[oc] = vaddvq_f32(vacc) + tail;
            }
        }
    }
}

/* --- threading ---------------------------------------------------------- */

typedef struct {
    const float *in, *w;
    float* out;
    int H, W, Cin, Cout, K;
    int oh_begin, oh_end;
} conv_band_t;

static void* conv_band_entry(void* arg) {
    conv_band_t* b = (conv_band_t*)arg;
    conv_neon_band(b->in, b->w, b->out, b->H, b->W, b->Cin, b->Cout, b->K,
                   b->oh_begin, b->oh_end);
    return NULL;
}

/* Split the output rows across nthreads workers. nthreads==1 runs inline, so
 * the single-threaded path is exactly the original code with no pthread cost. */
static void conv_neon(const float* in, const float* w, float* out,
                      int H, int W, int Cin, int Cout, int K, int nthreads) {
    if (nthreads > H) nthreads = H;
    if (nthreads < 1) nthreads = 1;

    if (nthreads == 1) {
        conv_neon_band(in, w, out, H, W, Cin, Cout, K, 0, H);
        return;
    }

    conv_band_t bands[MAX_THREADS];
    pthread_t   tids[MAX_THREADS];
    int         spawned[MAX_THREADS];

    int base = H / nthreads, rem = H % nthreads, next = 0;
    for (int t = 0; t < nthreads; t++) {
        int rows = base + (t < rem ? 1 : 0);
        bands[t] = (conv_band_t){ in, w, out, H, W, Cin, Cout, K, next, next + rows };
        next += rows;
        spawned[t] = 0;
    }

    /* Worker 0 runs on the calling thread so nthreads bands use nthreads cores. */
    for (int t = 1; t < nthreads; t++) {
        spawned[t] = (pthread_create(&tids[t], NULL, conv_band_entry, &bands[t]) == 0);
    }
    conv_band_entry(&bands[0]);
    for (int t = 1; t < nthreads; t++) {
        if (spawned[t]) pthread_join(tids[t], NULL);
        else conv_band_entry(&bands[t]);   /* create failed: run it inline */
    }
}

static double compute_fingerprint(const float* out, int64_t n) {
    double s = 0.0, sq = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double v = (double)out[i];
        s += v; sq += v * v;
    }
    return s + sqrt(sq);
}

int main(int argc, char** argv) {
    int H = 256, W = 256, Cin = 8, Cout = 8, K = 3, iterations = 1, threads = 1;
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
    if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || (K % 2) == 0) {
        fprintf(stderr, "ERROR: H,W,cin,cout>0 and k>0 odd\n"); return 1;
    }
    if (threads < 1 || threads > MAX_THREADS) {
        fprintf(stderr, "ERROR: --threads must be 1..%d\n", MAX_THREADS); return 1;
    }

    printf("conv-neon_channel-start H=%d W=%d cin=%d cout=%d k=%d seed=%llu threads=%d\n",
           H, W, Cin, Cout, K, (unsigned long long)seed, threads);
    fflush(stdout);

    int64_t n_in  = (int64_t)H * W * Cin;
    int64_t n_w   = (int64_t)Cout * K * K * Cin;
    int64_t n_out = (int64_t)H * W * Cout;
    float *in = NULL, *w = NULL, *out = NULL;
    if (posix_memalign((void**)&in, 64, n_in * sizeof(float)) ||
        posix_memalign((void**)&w, 64, n_w * sizeof(float)) ||
        posix_memalign((void**)&out, 64, n_out * sizeof(float))) {
        fprintf(stderr, "ERROR: posix_memalign failed\n"); return 1;
    }
    uint64_t state = seed;
    fill_stream(in, n_in, &state);
    fill_stream(w, n_w, &state);

    int reps = validate_only ? 1 : iterations;

    volatile double sink = 0.0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    for (int it = 0; it < reps; it++) {
        conv_neon(in, w, out, H, W, Cin, Cout, K, threads);
        sink += (double)out[0];
        __asm__ __volatile__("" ::: "memory");
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    (void)sink;

    int64_t wall_us = (int64_t)((t1.tv_sec - t0.tv_sec) * 1000000LL +
                                (t1.tv_nsec - t0.tv_nsec) / 1000LL);
    int64_t flop  = (int64_t)2 * H * W * Cout * K * K * Cin * reps;
    int64_t bytes = (int64_t)4 * (n_in + n_w + n_out) * reps;
    double fp = compute_fingerprint(out, n_out);

    printf("conv-neon_channel-end fingerprint=%.6e wall_us=%lld iterations=%d "
           "threads=%d H=%d W=%d cin=%d cout=%d k=%d flop=%lld bytes=%lld\n",
           fp, (long long)wall_us, reps, threads, H, W, Cin, Cout, K,
           (long long)flop, (long long)bytes);
    fflush(stdout);

    free(in); free(w); free(out);
    return 0;
}
