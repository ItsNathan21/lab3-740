// matmul_dsp_sweep_main.c — DSP variance characterization sweep driver.
// Phase 6 variance characterization.
//
// Argv:
//   --variant  hvx|vtcm        which DSP kernel (default: hvx)
//   --power    baseline|lab    DSP power mode (default: baseline)
//   --warmup   N               warm-up dispatches before measuring (default: 10)
//   --measure  N               measured dispatches (default: 50)
//   --size     N               matrix size (default: 512)
//   --seed     N               splitmix64 seed (default: 42)
//   --quiet                    suppress per-iteration lines for long plateaus
//
// Stdout (one status line per measured dispatch, then a summary):
//   dsp-sweep-start variant=hvx power=baseline warmup=10 measure=50 size=512 seed=42
//   dsp-sweep-iter iter=0 dsp_us=XXXXX
//   ...
//   dsp-sweep-end mean_us=XXXXX std_us=XXXXX cv_pct=X.XX min_us=XXXXX max_us=XXXXX
//
// wall_us is not reported: this sweep is about DSP-side timing stability.
// Each dispatch sends the same A, B matrices (no seed rotation) so cache
// effects are constant and the measured variance reflects clock stability.

#include "dsp_matmul.h"
#include "remote.h"
#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    for (int i = 0; i < nn; i++)
        A[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(&state)) - 1.0);
    for (int i = 0; i < nn; i++)
        B[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(&state)) - 1.0);
}

typedef int (*fn_session_ctrl)(uint32_t req, void* data, uint32_t datalen);
typedef int (*fn_open)(const char*, remote_handle64*);
typedef int (*fn_close)(remote_handle64);
typedef int (*fn_matmul)(remote_handle64,
                         const float*, int,
                         const float*, int,
                         float*, int,
                         int, int*);
typedef int (*fn_set_power)(remote_handle64, int);

#define DSP_MATMUL_URI  dsp_matmul_URI CDSP_DOMAIN "&_unsigned_module=1"

int main(int argc, char** argv) {
    const char* variant  = "hvx";
    const char* power    = "baseline";
    int n_warmup  = 10;
    int n_measure = 50;
    int size      = 512;
    uint64_t seed = 42;
    bool quiet    = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--quiet")) {
            quiet = true;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "%s requires a value\n", argv[i]);
            return 2;
        }
        if      (!strcmp(argv[i],"--variant")) variant  = argv[++i];
        else if (!strcmp(argv[i],"--power"))   power    = argv[++i];
        else if (!strcmp(argv[i],"--warmup"))  n_warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--measure")) n_measure= atoi(argv[++i]);
        else if (!strcmp(argv[i],"--size"))    size     = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--seed"))    seed     = (uint64_t)atoll(argv[++i]);
        else {
            fprintf(stderr, "unsupported argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (strcmp(variant, "hvx") != 0 && strcmp(variant, "vtcm") != 0) {
        fprintf(stderr, "--variant must be hvx|vtcm\n");
        return 2;
    }
    if (strcmp(power, "baseline") != 0 && strcmp(power, "lab") != 0) {
        fprintf(stderr, "--power must be baseline|lab\n");
        return 2;
    }
    if (n_warmup < 0 || n_measure <= 0 || size <= 0 || (size % 32) != 0) {
        fprintf(stderr,
                "--warmup must be >=0, --measure >0, and --size a positive multiple of 32\n");
        return 2;
    }

    int power_mode_id = (!strcmp(power, "lab")) ? 1 : 0;
    const char* kernel_sym = (!strcmp(variant, "vtcm"))
                             ? "dsp_matmul_matmul_hvx_vtcm_fp32"
                             : "dsp_matmul_matmul_hvx_fp32";

    printf("dsp-sweep-start variant=%s power=%s warmup=%d measure=%d size=%d seed=%llu\n",
           variant, power, n_warmup, n_measure, size, (unsigned long long)seed);

    int nn = size * size;
    float* A = (float*)malloc(nn * sizeof(float));
    float* B = (float*)malloc(nn * sizeof(float));
    float* C = (float*)malloc(nn * sizeof(float));
    int*   samples = (int*)malloc(n_measure * sizeof(int));
    if (!A || !B || !C || !samples) { fprintf(stderr, "malloc failed\n"); return 1; }
    generate_inputs(size, seed, A, B);

    /* ── FastRPC init ── */
    void* cdsprpc = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!cdsprpc) { fprintf(stderr, "dlopen libcdsprpc.so: %s\n", dlerror()); return 1; }

    fn_session_ctrl my_session_ctrl =
        (fn_session_ctrl)dlsym(cdsprpc, "remote_session_control");
    if (!my_session_ctrl) {
        fprintf(stderr, "dlsym remote_session_control: %s\n", dlerror());
        dlclose(cdsprpc);
        free(A); free(B); free(C); free(samples);
        return 1;
    }
    struct remote_rpc_control_unsigned_module unsig =
        { .domain = CDSP_DOMAIN_ID, .enable = 1 };
    int session_rc =
        my_session_ctrl(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsig, sizeof(unsig));
    if (session_rc != 0) {
        fprintf(stderr, "enable unsigned cDSP PD failed: 0x%x\n", session_rc);
        dlclose(cdsprpc);
        free(A); free(B); free(C); free(samples);
        return 1;
    }

    void* stub = dlopen("./libdsp_matmul_stub.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!stub) { fprintf(stderr, "dlopen stub: %s\n", dlerror()); dlclose(cdsprpc); return 1; }

    fn_open      my_open    = (fn_open)      dlsym(stub, "dsp_matmul_open");
    fn_close     my_close   = (fn_close)     dlsym(stub, "dsp_matmul_close");
    fn_matmul    my_matmul  = (fn_matmul)    dlsym(stub, kernel_sym);
    fn_set_power my_power   = (fn_set_power) dlsym(stub, "dsp_matmul_set_power_mode");
    if (!my_open || !my_close || !my_matmul || !my_power) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        dlclose(stub); dlclose(cdsprpc); return 1;
    }

    remote_handle64 h = -1;
    if (my_open(DSP_MATMUL_URI, &h) != 0) {
        fprintf(stderr, "dsp_matmul_open failed\n");
        dlclose(stub); dlclose(cdsprpc); return 1;
    }

    /* ── Set power mode ── */
    int power_rc = my_power(h, power_mode_id);
    if (power_rc != 0) {
        fprintf(stderr,
                "set_power_mode(%s) rejected: 0x%x; refusing unpinned sweep\n",
                power, power_rc);
        if (power_mode_id != 0) {
            int restore_rc = my_power(h, 0);
            if (restore_rc != 0)
                fprintf(stderr, "baseline restore after LAB failure also failed: 0x%x\n",
                        restore_rc);
        }
        my_close(h);
        dlclose(stub); dlclose(cdsprpc);
        free(A); free(B); free(C); free(samples);
        return 1;
    }
    /* API acceptance is not a physical-clock readback; HAP exposes none. */
    printf("dsp-power-mode-applied mode=%s api_status=accepted\n", power);
    fflush(stdout);

    /* ── Warm-up ── */
    int dsp_us_tmp = 0;
    bool run_failed = false;
    for (int i = 0; i < n_warmup; i++) {
        memset(C, 0, nn * sizeof(float));
        int ret = my_matmul(h, A, nn, B, nn, C, nn, size, &dsp_us_tmp);
        if (ret != 0 || dsp_us_tmp <= 0) {
            fprintf(stderr,
                    "warmup failed at iter %d: rc=0x%x dsp_us=%d\n",
                    i, ret, dsp_us_tmp);
            run_failed = true;
            break;
        }
    }

    /* ── Measurement loop ── */
    for (int i = 0; i < n_measure && !run_failed; i++) {
        memset(C, 0, nn * sizeof(float));
        int ret = my_matmul(h, A, nn, B, nn, C, nn, size, &samples[i]);
        if (ret != 0 || samples[i] <= 0) {
            fprintf(stderr,
                    "matmul failed at iter %d: rc=0x%x dsp_us=%d\n",
                    i, ret, samples[i]);
            run_failed = true;
            break;
        }
        if (!quiet) {
            printf("dsp-sweep-iter iter=%d dsp_us=%d\n", i, samples[i]);
            fflush(stdout);
        }
    }

    /* ── Restore baseline power before closing ── */
    if (power_mode_id != 0) {
        int restore_rc = my_power(h, 0);
        if (restore_rc != 0) {
            fprintf(stderr, "baseline restore after LAB sweep failed: 0x%x\n",
                    restore_rc);
            run_failed = true;
        }
    }

    int close_rc = my_close(h);
    if (close_rc != 0) {
        fprintf(stderr, "dsp_matmul_close failed: 0x%x\n", close_rc);
        run_failed = true;
    }
    dlclose(stub);
    dlclose(cdsprpc);

    if (run_failed) {
        free(A); free(B); free(C); free(samples);
        return 1;
    }

    /* ── On-device statistics ── */
    double sum = 0.0, sum2 = 0.0;
    int valid = 0, min_us = 0x7fffffff, max_us = 0;
    for (int i = 0; i < n_measure; i++) {
        if (samples[i] <= 0) continue;
        sum  += samples[i];
        sum2 += (double)samples[i] * samples[i];
        if (samples[i] < min_us) min_us = samples[i];
        if (samples[i] > max_us) max_us = samples[i];
        valid++;
    }
    double mean = (valid > 0) ? sum / valid : 0.0;
    double var  = (valid > 1) ? (sum2 / valid - mean * mean) : 0.0;
    double std  = (var > 0) ? sqrt(var) : 0.0;
    double cv   = (mean > 0) ? 100.0 * std / mean : 0.0;

    if (valid != n_measure) {
        fprintf(stderr, "incomplete sweep: valid=%d expected=%d\n", valid, n_measure);
        free(A); free(B); free(C); free(samples);
        return 1;
    }

    printf("dsp-sweep-end mean_us=%.1f std_us=%.1f cv_pct=%.3f min_us=%d max_us=%d valid=%d\n",
           mean, std, cv, min_us, max_us, valid);

    free(A); free(B); free(C); free(samples);
    return 0;
}
