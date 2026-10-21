// dsp_matmul_main.c — ARM64 driver for DSP scalar matmul.
// Phase 6 Step 4.
//
// Argv contract (matches CPU variants):
//   --size N         square matrix size, default 512
//   --seed N         splitmix64 seed, default 42
//   --validate-only  one dispatch, print fingerprint, exit 0
//   --iterations N   reps inside a single timed dispatch, default 1
//
// Stdout:
//   matmul-dsp_scalar-start size=N seed=N
//   matmul-dsp_scalar-end fingerprint=<%.6e> wall_us=<int> dsp_us=<int>
//
// wall_us: ARM64-measured total FastRPC round-trip (includes buffer copy).
// dsp_us:  DSP-side compute-only time (HAP_perf_get_time_us inside skel).
//
// FastRPC init (3 steps, Android 15 workarounds):
//   1. dlopen("libcdsprpc.so") by name → vendor namespace
//   2. remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)
//   3. dlopen("./libdsp_matmul_stub.so") from CWD

#include "dsp_matmul.h"
#include "remote.h"
#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── splitmix64 (bit-identical to CPU variants) ── */
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
static void generate_inputs(int n, uint64_t seed, float* A, float* B) {
    uint64_t state = seed;
    int nn = n * n;
    for (int i = 0; i < nn; i++) {
        A[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(&state)) - 1.0);
    }
    for (int i = 0; i < nn; i++) {
        B[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(&state)) - 1.0);
    }
}
static double compute_fingerprint(const float* C, int nn) {
    double s = 0.0, sq = 0.0;
    for (int i = 0; i < nn; i++) {
        double x = (double)C[i];
        s += x; sq += x * x;
    }
    return s + sqrt(sq);
}

/* ── FastRPC fn-pointer types ── */
typedef int (*fn_session_ctrl)(uint32_t req, void* data, uint32_t datalen);
typedef int (*fn_open)(const char*, remote_handle64*);
typedef int (*fn_close)(remote_handle64);
typedef int (*fn_matmul)(remote_handle64,
                         const float*, int,
                         const float*, int,
                         float*, int,
                         int, int*);

#define DSP_MATMUL_URI  dsp_matmul_URI CDSP_DOMAIN "&_unsigned_module=1"

static long long now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

int main(int argc, char** argv) {
    /* ── parse args ── */
    int size = 512;
    uint64_t seed = 42;
    bool validate_only = false;
    int iterations = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--validate-only") == 0) {
            validate_only = true; iterations = 1;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        }
    }

    printf("matmul-dsp_scalar-start size=%d seed=%llu\n",
           size, (unsigned long long)seed);

    int nn = size * size;
    float* A = (float*)malloc(nn * sizeof(float));
    float* B = (float*)malloc(nn * sizeof(float));
    float* C = (float*)malloc(nn * sizeof(float));
    if (!A || !B || !C) { fprintf(stderr, "malloc failed\n"); return 1; }
    generate_inputs(size, seed, A, B);
    memset(C, 0, nn * sizeof(float));

    /* ── FastRPC init: step 1 — preload cdsprpc in vendor namespace ── */
    void* cdsprpc = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!cdsprpc) {
        fprintf(stderr, "dlopen libcdsprpc.so failed: %s\n", dlerror());
        return 1;
    }

    /* ── FastRPC init: step 2 — enable unsigned PD on cDSP ── */
    fn_session_ctrl my_session_ctrl =
        (fn_session_ctrl)dlsym(cdsprpc, "remote_session_control");
    if (my_session_ctrl) {
        struct remote_rpc_control_unsigned_module unsig =
            { .domain = CDSP_DOMAIN_ID, .enable = 1 };
        my_session_ctrl(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsig, sizeof(unsig));
    }

    /* ── FastRPC init: step 3 — load stub (cdsprpc already in global table) ── */
    void* stub = dlopen("./libdsp_matmul_stub.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!stub) {
        fprintf(stderr, "dlopen stub failed: %s\n", dlerror());
        dlclose(cdsprpc); return 1;
    }

    fn_open     my_open     = (fn_open)    dlsym(stub, "dsp_matmul_open");
    fn_close    my_close    = (fn_close)   dlsym(stub, "dsp_matmul_close");
    fn_matmul   my_matmul   = (fn_matmul)  dlsym(stub, "dsp_matmul_matmul_fp32");
    if (!my_open || !my_close || !my_matmul) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        dlclose(stub); dlclose(cdsprpc); return 1;
    }

    remote_handle64 h = -1;
    if (my_open(DSP_MATMUL_URI, &h) != 0) {
        fprintf(stderr, "dsp_matmul_open failed\n");
        dlclose(stub); dlclose(cdsprpc); return 1;
    }

    /* ── warm-up (one call before timing) ── */
    int dsp_us_warmup = 0;
    my_matmul(h, A, nn, B, nn, C, nn, size, &dsp_us_warmup);

    /* ── timed dispatch ── */
    long long wall_start = now_us();
    int dsp_us_last = 0;
    for (int iter = 0; iter < iterations; iter++) {
        int ret = my_matmul(h, A, nn, B, nn, C, nn, size, &dsp_us_last);
        if (ret != 0) {
            fprintf(stderr, "dsp_matmul_matmul_fp32 failed: 0x%x\n", ret);
            my_close(h); dlclose(stub); dlclose(cdsprpc); return 1;
        }
    }
    long long wall_us = now_us() - wall_start;

    double fp = compute_fingerprint(C, nn);

    printf("matmul-dsp_scalar-end fingerprint=%.6e wall_us=%lld dsp_us=%d\n",
           fp, wall_us, dsp_us_last);

    my_close(h);
    dlclose(stub);
    dlclose(cdsprpc);
    free(A); free(B); free(C);
    return 0;
}
