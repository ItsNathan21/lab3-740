/*
 * Snapdragon Profiler — convolution / cpu / scalar_baseline (roofline mid-AI point).
 *
 * out = conv2d(in, W), dense, KxK, stride 1, "same" (zero) padding, NHWC input /
 * OHWI weights / NHWC output. Direct convolution (no im2col, no Winograd) so the
 * arithmetic intensity is transparent: AI ≈ (K²/8)·C for Cin=Cout=C. The default
 * Cin=Cout=8, K=3, 256×256 lands at AI≈18 — the middle of the roofline. Channels
 * are a parameter so C can be swept to trace the roofline diagonal.
 *
 * Naive scalar accumulation (foil for the NEON variant). Inputs + weights are
 * deterministic via splitmix64 (bit-identical to kernels/convolution/reference.py
 * and the NEON variant): input's H·W·Cin values first, then weights' Cout·K²·Cin.
 *
 * Argv:
 *   --H N --W N       spatial dims (default 256)
 *   --cin N --cout N  channels (default 8); AI ≈ (K²/8)·(Cin·Cout/(Cin+Cout))
 *   --k N             kernel size (default 3, odd)
 *   --seed N          splitmix64 seed (default 42)
 *   --validate-only   one pass, print fingerprint, exit
 *   --iterations N    timed passes (default 1)
 *
 * Stdout:
 *   conv-scalar_baseline-start H=.. W=.. cin=.. cout=.. k=.. seed=..
 *   conv-scalar_baseline-end fingerprint=<%.6e> wall_us=<int> iterations=<K> \
 *       H=.. W=.. cin=.. cout=.. k=.. flop=<int> bytes=<int>
 *
 * flop  = 2·H·W·Cout·K²·Cin·K_iters (full MACs). bytes = 4·(in+weights+out)·K_iters.
 * Fingerprint = sum(out)+L2(out), fp64. Validated vs fp64 reference, 1e-3 rel.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
/* Fill n fp32 values in [-1,1) from one advancing state. */
static void fill_stream(float* dst, int64_t n, uint64_t* state) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(state)) - 1.0);
    }
}

/* Direct conv: out[oh,ow,oc] = sum_{kh,kw,ic} in[oh+kh-pad, ow+kw-pad, ic] * w[oc,kh,kw,ic]
 * NHWC in/out, OHWI weights, zero pad. */
static void conv_scalar(const float* in, const float* w, float* out,
                        int H, int W, int Cin, int Cout, int K) {
    int pad = K / 2;
    for (int oh = 0; oh < H; oh++) {
        for (int ow = 0; ow < W; ow++) {
            float* out_px = &out[((int64_t)oh * W + ow) * Cout];
            for (int oc = 0; oc < Cout; oc++) {
                float acc = 0.0f;
                for (int kh = 0; kh < K; kh++) {
                    int ih = oh + kh - pad;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow + kw - pad;
                        if (iw < 0 || iw >= W) continue;
                        const float* in_px = &in[((int64_t)ih * W + iw) * Cin];
                        const float* w_px  = &w[(((int64_t)oc * K + kh) * K + kw) * Cin];
                        for (int ic = 0; ic < Cin; ic++) {
                            acc += in_px[ic] * w_px[ic];
                        }
                    }
                }
                out_px[oc] = acc;
            }
        }
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
    int H = 256, W = 256, Cin = 8, Cout = 8, K = 3, iterations = 1;
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
        else { fprintf(stderr, "ERROR: unknown arg: %s\n", argv[i]); return 1; }
    }
    if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || (K % 2) == 0) {
        fprintf(stderr, "ERROR: H,W,cin,cout>0 and k>0 odd\n"); return 1;
    }

    printf("conv-scalar_baseline-start H=%d W=%d cin=%d cout=%d k=%d seed=%llu\n",
           H, W, Cin, Cout, K, (unsigned long long)seed);
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
        conv_scalar(in, w, out, H, W, Cin, Cout, K);
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

    printf("conv-scalar_baseline-end fingerprint=%.6e wall_us=%lld iterations=%d "
           "H=%d W=%d cin=%d cout=%d k=%d flop=%lld bytes=%lld\n",
           fp, (long long)wall_us, reps, H, W, Cin, Cout, K,
           (long long)flop, (long long)bytes);
    fflush(stdout);

    free(in); free(w); free(out);
    return 0;
}
