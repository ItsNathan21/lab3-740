// dsp_conv_main.c — ARM64 driver for cDSP direct convolution (scalar + HVX).
// out = conv2d(in, weights), fp32. Roofline mid-AI point.
//
// Argv: --H --W --cin --cout --k --seed --variant scalar|hvx --power baseline|lab
//       --validate-only --iterations N   (defaults 256/256/8/8/3/42/hvx/baseline/1)
//
// Stdout:
//   conv-dsp_<variant>-start H=.. W=.. cin=.. cout=.. k=.. seed=..
//   conv-dsp_<variant>-end fingerprint=<%.6e> wall_us=<int> dsp_us=<int> \
//       iterations=<K> H=.. W=.. cin=.. cout=.. k=.. flop=<int> bytes=<int>
//
// wall_us: FastRPC round-trip (input+weights IN, HxWxCout output OUT). dsp_us:
// compute-only. Convolution is idempotent, so the fingerprint matches the fp64
// reference (kernels/convolution/reference.py) within 1e-3 relative.

#include "dsp_conv.h"
#include "remote.h"
#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const uint64_t SM_INC = 0x9E3779B97F4A7C15ULL, SM1 = 0xBF58476D1CE4E5B9ULL, SM2 = 0x94D049BB133111EBULL;
static inline uint64_t sm_next(uint64_t* s) { *s += SM_INC; uint64_t z = *s; z = (z ^ (z >> 30)) * SM1; z = (z ^ (z >> 27)) * SM2; return z ^ (z >> 31); }
static inline double u2d(uint64_t x) { return (double)(x >> 11) * (1.0 / (double)(1ULL << 53)); }
static void fill_stream(float* d, int64_t n, uint64_t* s) { for (int64_t i = 0; i < n; i++) d[i] = (float)(2.0 * u2d(sm_next(s)) - 1.0); }
static double fingerprint(const float* o, int64_t n) { double s = 0, sq = 0; for (int64_t i = 0; i < n; i++) { double v = o[i]; s += v; sq += v * v; } return s + sqrt(sq); }

typedef int (*fn_ctrl)(uint32_t, void*, uint32_t);
typedef int (*fn_open)(const char*, remote_handle64*);
typedef int (*fn_close)(remote_handle64);
typedef int (*fn_conv)(remote_handle64, const float*, int, const float*, int, float*, int, int, int, int, int, int, int*);
typedef int (*fn_power)(remote_handle64, int);

#define DSP_CONV_URI  dsp_conv_URI CDSP_DOMAIN "&_unsigned_module=1"

static long long now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000; }

int main(int argc, char** argv) {
    int H = 256, W = 256, Cin = 8, Cout = 8, K = 3, iterations = 1;
    uint64_t seed = 42;
    bool validate_only = false;
    const char *variant = "hvx", *power = "baseline";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--H") && i+1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--W") && i+1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cin") && i+1 < argc) Cin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cout") && i+1 < argc) Cout = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--variant") && i+1 < argc) variant = argv[++i];
        else if (!strcmp(argv[i], "--power") && i+1 < argc) power = argv[++i];
        else if (!strcmp(argv[i], "--validate-only")) { validate_only = true; iterations = 1; }
        else if (!strcmp(argv[i], "--iterations") && i+1 < argc) iterations = atoi(argv[++i]);
    }
    bool use_hvx = !strcmp(variant, "hvx");

    printf("conv-dsp_%s-start H=%d W=%d cin=%d cout=%d k=%d seed=%llu\n",
           variant, H, W, Cin, Cout, K, (unsigned long long)seed);

    int64_t n_in = (int64_t)H * W * Cin, n_w = (int64_t)Cout * K * K * Cin, n_out = (int64_t)H * W * Cout;
    float *in = malloc(n_in * sizeof(float)), *w = malloc(n_w * sizeof(float)), *out = malloc(n_out * sizeof(float));
    if (!in || !w || !out) { fprintf(stderr, "malloc failed\n"); return 1; }
    uint64_t st = seed; fill_stream(in, n_in, &st); fill_stream(w, n_w, &st);
    memset(out, 0, n_out * sizeof(float));

    void* cdsprpc = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!cdsprpc) { fprintf(stderr, "dlopen libcdsprpc.so failed: %s\n", dlerror()); return 1; }
    fn_ctrl my_ctrl = (fn_ctrl)dlsym(cdsprpc, "remote_session_control");
    if (my_ctrl) { struct remote_rpc_control_unsigned_module u = { .domain = CDSP_DOMAIN_ID, .enable = 1 }; my_ctrl(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof(u)); }

    void* stub = dlopen("./libdsp_conv_stub.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!stub) { fprintf(stderr, "dlopen stub failed: %s\n", dlerror()); dlclose(cdsprpc); return 1; }
    fn_open my_open = (fn_open)dlsym(stub, "dsp_conv_open");
    fn_close my_close = (fn_close)dlsym(stub, "dsp_conv_close");
    fn_conv my_scalar = (fn_conv)dlsym(stub, "dsp_conv_conv_fp32");
    fn_conv my_hvx = (fn_conv)dlsym(stub, "dsp_conv_conv_hvx_fp32");
    fn_power my_power = (fn_power)dlsym(stub, "dsp_conv_set_power_mode");
    if (!my_open || !my_close || !my_scalar || !my_hvx) { fprintf(stderr, "dlsym failed: %s\n", dlerror()); dlclose(stub); dlclose(cdsprpc); return 1; }
    fn_conv my_conv = use_hvx ? my_hvx : my_scalar;

    remote_handle64 h = -1;
    if (my_open(DSP_CONV_URI, &h) != 0) { fprintf(stderr, "dsp_conv_open failed\n"); dlclose(stub); dlclose(cdsprpc); return 1; }
    if (my_power) my_power(h, !strcmp(power, "lab") ? 1 : 0);

    int dsp_warm = 0;
    my_conv(h, in, (int)n_in, w, (int)n_w, out, (int)n_out, H, W, Cin, Cout, K, &dsp_warm);

    long long wall_start = now_us();
    int dsp_us_last = 0;
    for (int it = 0; it < iterations; it++) {
        int ret = my_conv(h, in, (int)n_in, w, (int)n_w, out, (int)n_out, H, W, Cin, Cout, K, &dsp_us_last);
        if (ret != 0) { fprintf(stderr, "dsp_conv dispatch failed: 0x%x\n", ret); my_close(h); dlclose(stub); dlclose(cdsprpc); return 1; }
    }
    long long wall_us = now_us() - wall_start;

    int64_t flop = (int64_t)2 * H * W * Cout * K * K * Cin * iterations;
    int64_t bytes = (int64_t)4 * (n_in + n_w + n_out) * iterations;
    printf("conv-dsp_%s-end fingerprint=%.6e wall_us=%lld dsp_us=%d iterations=%d "
           "H=%d W=%d cin=%d cout=%d k=%d flop=%lld bytes=%lld\n",
           variant, fingerprint(out, n_out), wall_us, dsp_us_last, iterations,
           H, W, Cin, Cout, K, (long long)flop, (long long)bytes);

    my_close(h); dlclose(stub); dlclose(cdsprpc);
    free(in); free(w); free(out);
    return 0;
}
