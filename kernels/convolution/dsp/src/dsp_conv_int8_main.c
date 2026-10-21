// dsp_conv_int8_main.c — ARM64 driver for cDSP INT8 direct convolution.
// out = conv2d(in, weights), int8 in / int32 out, HVX vrmpy + VTCM + 6 threads.
//
// Argv: --H --W --cin --cout --k --seed --power baseline|lab
//       --validate-only --iterations N   (defaults 128/128/256/256/3/42/baseline/1)
//
// Stdout:
//   conv-dsp_int8-start H=.. W=.. cin=.. cout=.. k=.. seed=..
//   conv-dsp_int8-end checksum=<int64> wall_us=.. dsp_us=.. iterations=.. \
//       H=.. W=.. cin=.. cout=.. k=.. ops=.. bytes=..
//
// Integer math is exact, so `checksum` must match the CPU int8 kernel EXACTLY —
// not to some tolerance. Any difference is a bug, not rounding.
//
// Operand generation is byte-identical to the CPU kernel: the same splitmix64
// stream, drawn input-then-weights, each value reduced to int8 via (u>>24)&0xFF
// reinterpreted as signed.

#include "dsp_conv.h"
#include "remote.h"
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const uint64_t SM_INC = 0x9E3779B97F4A7C15ULL, SM1 = 0xBF58476D1CE4E5B9ULL, SM2 = 0x94D049BB133111EBULL;
static inline uint64_t sm_next(uint64_t* s) { *s += SM_INC; uint64_t z = *s; z = (z ^ (z >> 30)) * SM1; z = (z ^ (z >> 27)) * SM2; return z ^ (z >> 31); }
static void fill_i8(int8_t* d, int64_t n, uint64_t* s) {
    for (int64_t i = 0; i < n; i++) d[i] = (int8_t)((sm_next(s) >> 24) & 0xFF);
}
static int64_t checksum_i32(const int32_t* o, int64_t n) {
    int64_t acc = 0;
    for (int64_t i = 0; i < n; i++) acc += (int64_t)o[i] * (int64_t)((i & 7) + 1);
    return acc;
}

typedef int (*fn_ctrl)(uint32_t, void*, uint32_t);
typedef int (*fn_open)(const char*, remote_handle64*);
typedef int (*fn_close)(remote_handle64);
typedef int (*fn_conv8)(remote_handle64, const unsigned char*, int, const unsigned char*, int,
                        int*, int, int, int, int, int, int, int*);
typedef int (*fn_power)(remote_handle64, int);

#define DSP_CONV_URI  dsp_conv_URI CDSP_DOMAIN "&_unsigned_module=1"

static long long now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000; }

int main(int argc, char** argv) {
    int H = 128, W = 128, Cin = 256, Cout = 256, K = 3, iterations = 1;
    uint64_t seed = 42;
    bool validate_only = false;
    const char *power = "baseline";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--H") && i+1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--W") && i+1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cin") && i+1 < argc) Cin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cout") && i+1 < argc) Cout = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--power") && i+1 < argc) power = argv[++i];
        else if (!strcmp(argv[i], "--validate-only")) { validate_only = true; iterations = 1; }
        else if (!strcmp(argv[i], "--iterations") && i+1 < argc) iterations = atoi(argv[++i]);
    }
    (void)validate_only;

    printf("conv-dsp_int8-start H=%d W=%d cin=%d cout=%d k=%d seed=%llu\n",
           H, W, Cin, Cout, K, (unsigned long long)seed);
    fflush(stdout);

    int64_t n_in = (int64_t)H * W * Cin, n_w = (int64_t)Cout * K * K * Cin, n_out = (int64_t)H * W * Cout;
    int8_t *in = malloc(n_in), *w = malloc(n_w);
    int32_t *out = malloc(n_out * sizeof(int32_t));
    if (!in || !w || !out) { fprintf(stderr, "malloc failed\n"); return 1; }
    uint64_t st = seed; fill_i8(in, n_in, &st); fill_i8(w, n_w, &st);
    memset(out, 0, n_out * sizeof(int32_t));

    void* cdsprpc = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!cdsprpc) { fprintf(stderr, "dlopen libcdsprpc.so failed: %s\n", dlerror()); return 1; }
    fn_ctrl my_ctrl = (fn_ctrl)dlsym(cdsprpc, "remote_session_control");
    if (my_ctrl) { struct remote_rpc_control_unsigned_module u = { .domain = CDSP_DOMAIN_ID, .enable = 1 }; my_ctrl(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof(u)); }

    void* stub = dlopen("./libdsp_conv_stub.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!stub) { fprintf(stderr, "dlopen stub failed: %s\n", dlerror()); dlclose(cdsprpc); return 1; }
    fn_open  my_open  = (fn_open)dlsym(stub, "dsp_conv_open");
    fn_close my_close = (fn_close)dlsym(stub, "dsp_conv_close");
    fn_conv8 my_conv  = (fn_conv8)dlsym(stub, "dsp_conv_conv_int8");
    fn_power my_power = (fn_power)dlsym(stub, "dsp_conv_set_power_mode");
    if (!my_open || !my_close || !my_conv) { fprintf(stderr, "dlsym failed: %s\n", dlerror()); dlclose(stub); dlclose(cdsprpc); return 1; }

    remote_handle64 h = -1;
    if (my_open(DSP_CONV_URI, &h) != 0) { fprintf(stderr, "dsp_conv_open failed\n"); dlclose(stub); dlclose(cdsprpc); return 1; }
    if (my_power) my_power(h, !strcmp(power, "lab") ? 1 : 0);

    int dsp_warm = 0;
    my_conv(h, (const unsigned char*)in, (int)n_in, (const unsigned char*)w, (int)n_w,
            out, (int)n_out, H, W, Cin, Cout, K, &dsp_warm);

    long long wall_start = now_us();
    int dsp_us_last = 0;
    for (int it = 0; it < iterations; it++) {
        int ret = my_conv(h, (const unsigned char*)in, (int)n_in, (const unsigned char*)w, (int)n_w,
                          out, (int)n_out, H, W, Cin, Cout, K, &dsp_us_last);
        if (ret != 0) { fprintf(stderr, "dsp_conv_int8 dispatch failed: 0x%x\n", ret); my_close(h); dlclose(stub); dlclose(cdsprpc); return 1; }
    }
    long long wall_us = now_us() - wall_start;

    int64_t ops   = (int64_t)2 * H * W * Cout * K * K * Cin * iterations;
    int64_t bytes = (int64_t)(n_in + n_w + 4 * n_out) * iterations;
    printf("conv-dsp_int8-end checksum=%lld wall_us=%lld dsp_us=%d iterations=%d "
           "H=%d W=%d cin=%d cout=%d k=%d ops=%lld bytes=%lld\n",
           (long long)checksum_i32(out, n_out), wall_us, dsp_us_last, iterations,
           H, W, Cin, Cout, K, (long long)ops, (long long)bytes);
    fflush(stdout);

    my_close(h); dlclose(stub); dlclose(cdsprpc);
    free(in); free(w); free(out);
    return 0;
}
