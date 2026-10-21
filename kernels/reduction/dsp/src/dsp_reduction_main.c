// dsp_reduction_main.c — ARM64 driver for cDSP sum-reduction (scalar + HVX).
// sum = sum(x_i), fp32. Read-only memory-bound roofline anchor #2.
//
// Argv:
//   --size N          vector length (default 4194304 = 2^22)
//   --seed N          splitmix64 seed (default 42)
//   --variant V       scalar | hvx  (default hvx)
//   --power P         baseline | lab (default baseline)
//   --validate-only   one dispatch, print fingerprint, exit 0
//   --iterations N    reps inside a single timed dispatch (default 1)
//
// Stdout:
//   reduction-dsp_<variant>-start size=N seed=S
//   reduction-dsp_<variant>-end fingerprint=<%.6e> wall_us=<int> dsp_us=<int> \
//       iterations=<K> elems=<N> bytes=<int>
//
// wall_us: ARM64 total FastRPC round-trip (includes the N-float INPUT copy — the
//          transport tax that dominates a memory-bound kernel; the OUTPUT is one
//          scalar, so unlike SAXPY there is no result-copy tax). dsp_us: DSP
//          compute-only. bytes: N * 4 * K (read-only DSP-side traffic).
// The reduction is idempotent, so repeated dispatches give the same sum and the
// fingerprint matches the fp64 NumPy reference (kernels/reduction/reference.py)
// within 1e-3 relative tolerance (decision R6).

#include "dsp_reduction.h"
#include "remote.h"
#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const uint64_t SPLITMIX_INC = 0x9E3779B97F4A7C15ULL;
static const uint64_t SPLITMIX_M1  = 0xBF58476D1CE4E5B9ULL;
static const uint64_t SPLITMIX_M2  = 0x94D049BB133111EBULL;

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
static void generate_input(int n, uint64_t seed, float* x) {
    uint64_t state = seed;
    for (int i = 0; i < n; i++)
        x[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(&state)) - 1.0);
}

typedef int (*fn_session_ctrl)(uint32_t req, void* data, uint32_t datalen);
typedef int (*fn_open)(const char*, remote_handle64*);
typedef int (*fn_close)(remote_handle64);
typedef int (*fn_reduce)(remote_handle64,
                         const float*, int, float*, int, int*);
typedef int (*fn_power)(remote_handle64, int);

#define DSP_REDUCTION_URI  dsp_reduction_URI CDSP_DOMAIN "&_unsigned_module=1"

static long long now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

int main(int argc, char** argv) {
    int size = 4194304;
    uint64_t seed = 42;
    bool validate_only = false;
    int iterations = 1;
    const char* variant = "hvx";
    const char* power = "baseline";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) variant = argv[++i];
        else if (strcmp(argv[i], "--power") == 0 && i + 1 < argc) power = argv[++i];
        else if (strcmp(argv[i], "--validate-only") == 0) { validate_only = true; iterations = 1; }
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
    }
    bool use_hvx = (strcmp(variant, "hvx") == 0);

    printf("reduction-dsp_%s-start size=%d seed=%llu\n",
           variant, size, (unsigned long long)seed);

    float* x = (float*)malloc((size_t)size * sizeof(float));
    if (!x) { fprintf(stderr, "malloc failed\n"); return 1; }
    generate_input(size, seed, x);

    void* cdsprpc = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!cdsprpc) { fprintf(stderr, "dlopen libcdsprpc.so failed: %s\n", dlerror()); return 1; }

    fn_session_ctrl my_session_ctrl =
        (fn_session_ctrl)dlsym(cdsprpc, "remote_session_control");
    if (my_session_ctrl) {
        struct remote_rpc_control_unsigned_module unsig =
            { .domain = CDSP_DOMAIN_ID, .enable = 1 };
        my_session_ctrl(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsig, sizeof(unsig));
    }

    void* stub = dlopen("./libdsp_reduction_stub.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!stub) { fprintf(stderr, "dlopen stub failed: %s\n", dlerror()); dlclose(cdsprpc); return 1; }

    fn_open   my_open   = (fn_open)  dlsym(stub, "dsp_reduction_open");
    fn_close  my_close  = (fn_close) dlsym(stub, "dsp_reduction_close");
    fn_reduce my_scalar = (fn_reduce)dlsym(stub, "dsp_reduction_reduce_fp32");
    fn_reduce my_hvx    = (fn_reduce)dlsym(stub, "dsp_reduction_reduce_hvx_fp32");
    fn_power  my_power  = (fn_power) dlsym(stub, "dsp_reduction_set_power_mode");
    if (!my_open || !my_close || !my_scalar || !my_hvx) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        dlclose(stub); dlclose(cdsprpc); return 1;
    }
    fn_reduce my_reduce = use_hvx ? my_hvx : my_scalar;

    remote_handle64 h = -1;
    if (my_open(DSP_REDUCTION_URI, &h) != 0) {
        fprintf(stderr, "dsp_reduction_open failed\n");
        dlclose(stub); dlclose(cdsprpc); return 1;
    }

    if (my_power) my_power(h, (strcmp(power, "lab") == 0) ? 1 : 0);

    float sum_warmup = 0.0f;
    int dsp_us_warmup = 0;
    my_reduce(h, x, size, &sum_warmup, size, &dsp_us_warmup);

    long long wall_start = now_us();
    int dsp_us_last = 0;
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        int ret = my_reduce(h, x, size, &sum, size, &dsp_us_last);
        if (ret != 0) {
            fprintf(stderr, "dsp_reduction dispatch failed: 0x%x\n", ret);
            my_close(h); dlclose(stub); dlclose(cdsprpc); return 1;
        }
    }
    long long wall_us = now_us() - wall_start;

    int64_t bytes = (int64_t)size * (int64_t)sizeof(float) * iterations;

    printf("reduction-dsp_%s-end fingerprint=%.6e wall_us=%lld dsp_us=%d "
           "iterations=%d elems=%d bytes=%lld\n",
           variant, (double)sum, wall_us, dsp_us_last, iterations, size, (long long)bytes);

    my_close(h);
    dlclose(stub);
    dlclose(cdsprpc);
    free(x);
    return 0;
}
