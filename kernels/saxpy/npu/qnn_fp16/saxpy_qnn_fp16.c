/*
 * matmul_qnn_fp16.c — QNN HTP FP16 matmul harness
 *
 * Phase 7 Step 5 NPU kernel. Mirrors the OpenCL harness design:
 *   - dlopen libQnnHtp.so at runtime (no link-time QNN dependency)
 *   - Constructs a 512×512 FP16 MatMul graph via QNN C API
 *   - Profiles with QNN_PROFILE_LEVEL_DETAILED → per-op HTP timing
 *   - Fills inputs with splitmix64 PRNG (--seed N flag)
 *   - Prints NPU_DISPATCH_RESULT block for orchestrator parsing
 *
 * Build (see build.sh):
 *   aarch64-linux-android-clang -std=c11 -O2 -fno-omit-frame-pointer
 *     -I include -ldl -lm -o matmul_qnn_fp16 matmul_qnn_fp16.c
 *
 * Run on device:
 *   LD_LIBRARY_PATH=/data/local/tmp/profiler-kernels/npu \
 *   ADSP_LIBRARY_PATH=/data/local/tmp/profiler-kernels/npu \
 *   ./matmul_qnn_fp16 [--seed 42] [--n 512] [--reps 1]
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "QNN/QnnInterface.h"
#include "QNN/QnnProfile.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnOpDef.h"
#include "QNN/QnnDevice.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"

/* ── wall-clock helper ────────────────────────────────────────────────────── */

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* ── splitmix64 PRNG (same as CPU/GPU kernels) ────────────────────────────── */

static uint64_t sm_state;

static void sm_seed(uint64_t s) { sm_state = s; }

static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* float in [0,1) */
static float sm_float(void) {
    return (float)(sm_next() >> 11) * (1.0f / (float)(1ULL << 53));
}

/* ── __fp16 helpers ───────────────────────────────────────────────────────── */

typedef uint16_t fp16_t;

static fp16_t float_to_fp16(float f) {
    /* Use compiler's __fp16 if available (arm64 ABI), else bit-cast via union */
#if defined(__aarch64__)
    __fp16 h = (__fp16)f;
    fp16_t bits;
    memcpy(&bits, &h, 2);
    return bits;
#else
    /* Software conversion fallback */
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign     = (bits >> 31) & 0x1;
    uint32_t exp      = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (exp == 0xFF) return (fp16_t)((sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0));
    int new_exp = (int)exp - 127 + 15;
    if (new_exp >= 31) return (fp16_t)((sign << 15) | 0x7C00);
    if (new_exp <= 0) return (fp16_t)(sign << 15);
    return (fp16_t)((sign << 15) | ((uint32_t)new_exp << 10) | (mantissa >> 13));
#endif
}

static float fp16_to_float(fp16_t h) {
#if defined(__aarch64__)
    __fp16 val;
    memcpy(&val, &h, 2);
    return (float)val;
#else
    uint32_t sign     = (h >> 15) & 0x1;
    uint32_t exp      = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    if (exp == 0x1F) return sign ? -__builtin_inff() : __builtin_inff();
    if (exp == 0)    return (sign ? -1.0f : 1.0f) * (float)mantissa / (float)(1 << 24);
    float f;
    uint32_t bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mantissa << 13);
    memcpy(&f, &bits, 4);
    return f;
#endif
}

/* ── QNN function pointer table (resolved via QnnInterface_getProviders) ──── */

typedef Qnn_ErrorHandle_t (*QnnInterface_GetProvidersFn_t)(
    const QnnInterface_t ***providers, uint32_t *num_providers);

/* Resolved once at startup; used everywhere */
static QnnInterface_t g_iface;   /* copy of the vtable */

#define QNN(fn, ...) do {                                           \
    Qnn_ErrorHandle_t _rc = g_iface.QNN_INTERFACE_VER_NAME.fn(__VA_ARGS__); \
    if (_rc != QNN_SUCCESS) {                                       \
        fprintf(stderr, "[qnn] " #fn " failed: 0x%x\n", (unsigned)_rc); \
        goto fail;                                                  \
    }                                                               \
} while(0)

/* ── HTP power mode (perf infrastructure) ────────────────────────────────── */

static Qnn_DeviceHandle_t            g_device_handle = NULL;
static QnnHtpDevice_Infrastructure_t *g_htp_infra    = NULL;
static uint32_t                       g_power_cfg_id  = 0;

/*
 * Apply DCVS-off + TURBO voltage corners + sleep disabled + max RPC polling.
 * "sustained_high_performance" preset: fixed cDSP clock, no power-gating.
 *
 * Requires deviceCreate first — perf infra is device-level; without a device
 * handle, createPowerConfigId(0,0) returns INVALID_INPUT because device 0
 * doesn't exist from the backend's perspective.
 *
 * Call after backendCreate, before graphFinalize.
 */
static int htp_perf_begin(void) {
    /* 1. Create a device so the HTP backend registers hardware device 0 */
    Qnn_ErrorHandle_t rc =
        g_iface.QNN_INTERFACE_VER_NAME.deviceCreate(NULL /*logger*/, NULL /*config*/,
                                                     &g_device_handle);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[perf] deviceCreate failed: 0x%x\n", (unsigned)rc);
        return -1;
    }

    /* 2. Get infra pointer — backend sets *deviceInfra to its own struct */
    QnnDevice_Infrastructure_t raw_infra = NULL;
    rc = g_iface.QNN_INTERFACE_VER_NAME.deviceGetInfrastructure(&raw_infra);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[perf] deviceGetInfrastructure failed: 0x%x\n", (unsigned)rc);
        return -1;
    }
    g_htp_infra = (QnnHtpDevice_Infrastructure_t *)raw_infra;
    if (!g_htp_infra || !g_htp_infra->perfInfra.createPowerConfigId) {
        fprintf(stderr, "[perf] infra ptr or createPowerConfigId is NULL\n");
        return -1;
    }

    rc = g_htp_infra->perfInfra.createPowerConfigId(0 /*deviceId*/, 0 /*coreId*/,
                                                     &g_power_cfg_id);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[perf] createPowerConfigId(0,0) failed: 0x%x\n", (unsigned)rc);
        return -1;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
        .dcvsV3Config = {
            .contextId             = g_power_cfg_id,
            .setDcvsEnable         = 1,
            .dcvsEnable            = 0,   /* DCVS off — fixed frequency */
            .powerMode             = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
            .setSleepLatency       = 1,
            .sleepLatency          = 40,  /* µs — minimal wake latency */
            .setSleepDisable       = 1,
            .sleepDisable          = 1,   /* no sleep between dispatches */
            .setBusParams          = 1,
            .busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO,
            .setCoreParams         = 1,
            .coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO,
        }
    };

    /* RPC polling: cDSP stays awake polling for the next call (no sleep jitter) */
    QnnHtpPerfInfrastructure_PowerConfig_t rpc = {
        .option             = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME,
        .rpcPollingTimeConfig = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME,
    };

    const QnnHtpPerfInfrastructure_PowerConfig_t *cfgs[] = { &dcvs, &rpc, NULL };
    rc = g_htp_infra->perfInfra.setPowerConfig(g_power_cfg_id, cfgs);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[perf] setPowerConfig failed: 0x%x\n", (unsigned)rc);
        return -1;
    }
    fprintf(stderr, "[perf] sustained_high_performance active (config_id=%u)\n",
            g_power_cfg_id);
    return 0;
}

static void htp_perf_end(void) {
    if (g_htp_infra && g_power_cfg_id) {
        g_htp_infra->perfInfra.destroyPowerConfigId(g_power_cfg_id);
        g_power_cfg_id = 0;
    }
    if (g_device_handle) {
        g_iface.QNN_INTERFACE_VER_NAME.deviceFree(g_device_handle);
        g_device_handle = NULL;
    }
}

/* ── arg parsing ──────────────────────────────────────────────────────────── */

static int    g_size      = 1048576; /* vector length (default 2^20) */
static int    g_seed      = 42;
static int    g_reps      = 1;
static float  g_alpha     = 2.0f;
static int    g_sustained = 0;  /* 0=default power, 1=sustained_high_performance */

static void parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i+1 < argc) { g_size  = atoi(argv[++i]); }
        if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) { g_seed  = atoi(argv[++i]); }
        if (strcmp(argv[i], "--reps") == 0 && i+1 < argc) { g_reps  = atoi(argv[++i]); }
        if (strcmp(argv[i], "--alpha") == 0 && i+1 < argc) { g_alpha = strtof(argv[++i], NULL); }
        if (strcmp(argv[i], "--perf-mode") == 0 && i+1 < argc) {
            const char *m = argv[++i];
            g_sustained = (strcmp(m, "sustained") == 0 ||
                           strcmp(m, "sustained_high_performance") == 0) ? 1 : 0;
        }
    }
}

/* ── Qnn_Tensor_t initialiser helper ──────────────────────────────────────── */
/*
 * dims must be caller-owned with lifetime >= the graph (i.e., on main() stack
 * or heap). Compound literals would go out of scope — use explicit arrays.
 */
static void make_fp16_tensor(Qnn_Tensor_t *t, const char *name,
                              Qnn_TensorType_t type,
                              uint32_t rank, uint32_t *dims) { /* dims caller-owned */
    memset(t, 0, sizeof(*t));
    t->version = QNN_TENSOR_VERSION_1;
    Qnn_TensorV1_t *v = &t->v1;
    v->name            = name;
    v->type            = type;
    v->dataFormat      = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    v->dataType        = QNN_DATATYPE_FLOAT_16;
    v->quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    v->quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    v->rank            = rank;
    v->dimensions      = dims;   /* pointer must outlive the graph */
    v->memType         = QNN_TENSORMEMTYPE_RAW;
    /* clientBuf left as NULL/0 by default; STATIC tensors set it before create */
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    parse_args(argc, argv);

    int N = g_size;                 /* vector length (kept as N for reuse below) */
    size_t elem  = (size_t)N;
    size_t bytes = elem * sizeof(fp16_t);

    /* ── 1. Allocate and fill input buffers ─────────────────────────────── */
    /* out = alpha*x + y. x, y in [-1,1), x drawn first then y from one advancing
     * splitmix64 state — bit-identical value stream to the CPU/GPU/DSP SAXPY
     * (then cast to fp16), so the fingerprint cross-validates (within fp16). */

    fp16_t *A = malloc(bytes);   /* x */
    fp16_t *B = malloc(bytes);   /* y */
    fp16_t *C = malloc(bytes);   /* out */
    if (!A || !B || !C) { fprintf(stderr, "malloc failed\n"); return 1; }

    sm_seed((uint64_t)g_seed);
    for (size_t i = 0; i < elem; i++) A[i] = float_to_fp16(2.0f * sm_float() - 1.0f);
    for (size_t i = 0; i < elem; i++) B[i] = float_to_fp16(2.0f * sm_float() - 1.0f);

    fp16_t alpha_h = float_to_fp16(g_alpha);
    memset(C, 0, bytes);

    /* ── 2. Load QNN HTP backend via dlopen ─────────────────────────────── */

    void *htp_lib = dlopen("libQnnHtp.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!htp_lib) {
        fprintf(stderr, "[qnn] dlopen libQnnHtp.so failed: %s\n", dlerror());
        return 1;
    }

    QnnInterface_GetProvidersFn_t get_providers =
        (QnnInterface_GetProvidersFn_t)dlsym(htp_lib, "QnnInterface_getProviders");
    if (!get_providers) {
        fprintf(stderr, "[qnn] dlsym QnnInterface_getProviders failed: %s\n", dlerror());
        return 1;
    }

    const QnnInterface_t **providers = NULL;
    uint32_t num_providers = 0;
    if (get_providers(&providers, &num_providers) != QNN_SUCCESS || num_providers == 0) {
        fprintf(stderr, "[qnn] QnnInterface_getProviders returned no providers\n");
        return 1;
    }

    /* Copy the first provider's interface (the HTP backend) */
    g_iface = providers[0][0];

    /* ── 3. Backend, Context, Graph ─────────────────────────────────────── */

    Qnn_BackendHandle_t backend  = NULL;
    Qnn_ContextHandle_t context  = NULL;
    Qnn_GraphHandle_t   graph    = NULL;
    Qnn_ProfileHandle_t profile_finalize = NULL;
    Qnn_ProfileHandle_t profile_execute  = NULL;
    int64_t t0, t1;
    int64_t finalize_us = 0;
    int64_t *exec_times = NULL;

    QNN(backendCreate, NULL /*logger*/, NULL /*config*/, &backend);

    /* Optional: sustained_high_performance power mode (--perf-mode sustained) */
    if (g_sustained) {
        if (htp_perf_begin() != 0) goto fail;
    }

    QNN(contextCreate, backend, NULL /*device*/, NULL /*config*/, &context);
    QNN(graphCreate,   context, "saxpy_fp16", NULL /*config*/, &graph);

    /* ── 4. Define tensors (SAXPY: out = alpha*x + y) ───────────────────── */
    /* Rank-1 vectors; alpha is a rank-1 [1] STATIC constant broadcast over the
     * multiply. tmp is a NATIVE intermediate (alpha*x). Dimension arrays must
     * outlive the graph — keep on main() stack. */
    uint32_t dims_vec[1]   = {(uint32_t)N};
    uint32_t dims_alpha[1] = {1};

    Qnn_Tensor_t tX, tY, tAlpha, tTmp, tOut;
    make_fp16_tensor(&tX,     "input_x",  QNN_TENSOR_TYPE_APP_WRITE, 1, dims_vec);
    make_fp16_tensor(&tY,     "input_y",  QNN_TENSOR_TYPE_APP_WRITE, 1, dims_vec);
    make_fp16_tensor(&tAlpha, "alpha",    QNN_TENSOR_TYPE_STATIC,    1, dims_alpha);
    make_fp16_tensor(&tTmp,   "tmp_ax",   QNN_TENSOR_TYPE_NATIVE,    1, dims_vec);
    make_fp16_tensor(&tOut,   "output",   QNN_TENSOR_TYPE_APP_READ,  1, dims_vec);

    /* STATIC alpha carries its data at creation time. */
    tAlpha.v1.clientBuf.data     = &alpha_h;
    tAlpha.v1.clientBuf.dataSize = (uint32_t)sizeof(fp16_t);

    /* Register tensors — this assigns each an id used by the op nodes below. */
    QNN(tensorCreateGraphTensor, graph, &tX);
    QNN(tensorCreateGraphTensor, graph, &tY);
    QNN(tensorCreateGraphTensor, graph, &tAlpha);
    QNN(tensorCreateGraphTensor, graph, &tTmp);
    QNN(tensorCreateGraphTensor, graph, &tOut);

    /* ── 5. Add nodes: tmp = x * alpha ; out = tmp + y ──────────────────── */

    Qnn_Tensor_t mul_in[2]  = {tX, tAlpha};
    Qnn_Tensor_t mul_out[1] = {tTmp};
    Qnn_OpConfig_t op_mul = {
        .version = QNN_OPCONFIG_VERSION_1,
        .v1 = {
            .name        = "saxpy_mul",
            .packageName = QNN_OP_PACKAGE_NAME_QTI_AISW,
            .typeName    = QNN_OP_ELEMENT_WISE_MULTIPLY,
            .numOfParams   = 0, .params = NULL,
            .numOfInputs   = 2, .inputTensors  = mul_in,
            .numOfOutputs  = 1, .outputTensors = mul_out,
        }
    };
    QNN(graphAddNode, graph, op_mul);

    Qnn_Tensor_t add_in[2]  = {tTmp, tY};
    Qnn_Tensor_t add_out[1] = {tOut};
    Qnn_OpConfig_t op_add = {
        .version = QNN_OPCONFIG_VERSION_1,
        .v1 = {
            .name        = "saxpy_add",
            .packageName = QNN_OP_PACKAGE_NAME_QTI_AISW,
            .typeName    = QNN_OP_ELEMENT_WISE_ADD,
            .numOfParams   = 0, .params = NULL,
            .numOfInputs   = 2, .inputTensors  = add_in,
            .numOfOutputs  = 1, .outputTensors = add_out,
        }
    };
    QNN(graphAddNode, graph, op_add);

    /* ── 6. Profile handles (separate for finalize vs execute) ─────────── */

    QNN(profileCreate, backend, QNN_PROFILE_LEVEL_DETAILED, &profile_finalize);
    QNN(profileCreate, backend, QNN_PROFILE_LEVEL_DETAILED, &profile_execute);

    /* ── 7. Finalize (JIT compile graph to HTP microcode) ───────────────── */

    t0 = now_ns();
    QNN(graphFinalize, graph, profile_finalize, NULL /*signal*/);
    t1 = now_ns();
    finalize_us = (t1 - t0) / 1000;

    /* ── 8. Bind data buffers and execute ───────────────────────────────── */

    Qnn_Tensor_t exec_inputs[2]  = {tX, tY};
    Qnn_Tensor_t exec_outputs[1] = {tOut};

    /* Bind data buffers — only set at execute time, not at tensor registration */
    exec_inputs[0].v1.clientBuf.data     = A;   /* x   */
    exec_inputs[0].v1.clientBuf.dataSize = (uint32_t)bytes;
    exec_inputs[1].v1.clientBuf.data     = B;   /* y   */
    exec_inputs[1].v1.clientBuf.dataSize = (uint32_t)bytes;
    exec_outputs[0].v1.clientBuf.data    = C;   /* out */
    exec_outputs[0].v1.clientBuf.dataSize = (uint32_t)bytes;

    /* ── 9. Warm-up execute (not timed) ────────────────────────────────── */
    /*
     * HTP v79 backend does not support profiling at graphExecute time —
     * passing a non-NULL profile handle returns QNN_GRAPH_ERROR_SET_PROFILE.
     * Use NULL always; timing is captured via wall-clock (now_ns).
     */
    {
        Qnn_ErrorHandle_t wrc = g_iface.QNN_INTERFACE_VER_NAME.graphExecute(
            graph, exec_inputs, 2, exec_outputs, 1, NULL, NULL);
        if (wrc != QNN_SUCCESS)
            fprintf(stderr, "[qnn] warm-up execute failed: 0x%x (continuing)\n",
                    (unsigned)wrc);
    }

    /* ── 10. Rep loop ───────────────────────────────────────────────────── */

    exec_times = malloc((size_t)g_reps * sizeof(int64_t));
    if (!exec_times) { fprintf(stderr, "malloc exec_times failed\n"); goto fail; }

    for (int r = 0; r < g_reps; r++) {
        t0 = now_ns();
        Qnn_ErrorHandle_t rc = g_iface.QNN_INTERFACE_VER_NAME.graphExecute(
            graph, exec_inputs, 2, exec_outputs, 1, NULL, NULL);
        t1 = now_ns();
        if (rc != QNN_SUCCESS) {
            fprintf(stderr, "[qnn] graphExecute rep %d failed: 0x%x\n", r, (unsigned)rc);
            goto fail;
        }
        exec_times[r] = (t1 - t0) / 1000;
    }
    /* ── 11. Execute timing stats ───────────────────────────────────────── */

    double sum_t = 0.0;
    int64_t min_us = exec_times[0], max_us = exec_times[0];
    for (int r = 0; r < g_reps; r++) {
        sum_t += (double)exec_times[r];
        if (exec_times[r] < min_us) min_us = exec_times[r];
        if (exec_times[r] > max_us) max_us = exec_times[r];
    }
    double mean_us = sum_t / (double)g_reps;
    double var = 0.0;
    for (int r = 0; r < g_reps; r++) {
        double d = (double)exec_times[r] - mean_us;
        var += d * d;
    }
    double stddev_us = sqrt(var / (double)g_reps);
    double cov_pct   = (mean_us > 0.0) ? (stddev_us / mean_us) * 100.0 : 0.0;

    /* ── 12. Read profiling events from finalize profile ────────────────── */
    /*
     * Finalize-time profiling works correctly; execute-time profiling does not.
     */
    const QnnProfile_EventId_t *events = NULL;
    uint32_t num_events = 0;
    g_iface.QNN_INTERFACE_VER_NAME.profileGetEvents(profile_finalize, &events, &num_events);

    int64_t finalize_profile_us = 0;
    for (uint32_t i = 0; i < num_events; i++) {
        QnnProfile_EventData_t evdata;
        if (g_iface.QNN_INTERFACE_VER_NAME.profileGetEventData(events[i], &evdata)
                != QNN_SUCCESS) continue;
        if (evdata.type == QNN_PROFILE_EVENTTYPE_FINALIZE)
            finalize_profile_us = (int64_t)evdata.value;
    }

    /* ── 13. Fingerprint (cross-validation vs CPU/GPU) ──────────────────── */

    double sum_c = 0.0, frob_c = 0.0;
    for (size_t i = 0; i < elem; i++) {
        float v = fp16_to_float(C[i]);
        sum_c  += v;
        frob_c += (double)v * v;
    }
    frob_c = sqrt(frob_c);
    double fingerprint = sum_c + frob_c;   /* combined; matches CPU/GPU/DSP (fp16) */

    /* ── 14. Output block ───────────────────────────────────────────────── */

    printf("NPU_DISPATCH_RESULT\n");
    printf("status: SUCCESS\n");
    printf("power_mode: %s\n",             g_sustained ? "sustained_high_performance" : "default");
    printf("reps: %d\n",                   g_reps);
    printf("execute_us_mean: %.1f\n",      mean_us);
    printf("execute_us_stddev: %.1f\n",    stddev_us);
    printf("execute_us_min: %lld\n",       (long long)min_us);
    printf("execute_us_max: %lld\n",       (long long)max_us);
    printf("execute_us_cov_pct: %.3f\n",   cov_pct);
    printf("finalize_us: %lld\n",          (long long)finalize_us);
    printf("finalize_profile_us: %lld\n",  (long long)finalize_profile_us);
    printf("num_profile_events: %u\n",     num_events);
    printf("precision: fp16\n");
    printf("elems: %d\n",                  N);
    printf("bytes: %lld\n",                (long long)((int64_t)3 * N * (int64_t)sizeof(fp16_t)));
    printf("alpha: %.6g\n",                (double)g_alpha);
    printf("fingerprint: %.6e\n",          fingerprint);
    printf("fingerprint_sum: %.6f\n",      sum_c);
    printf("fingerprint_frob: %.6f\n",     frob_c);
    for (int r = 0; r < g_reps; r++)
        printf("rep_%d_us: %lld\n", r, (long long)exec_times[r]);
    printf("NPU_DISPATCH_END\n");

    free(exec_times);

    /* ── cleanup ────────────────────────────────────────────────────────── */

    htp_perf_end();
    if (profile_execute)  g_iface.QNN_INTERFACE_VER_NAME.profileFree(profile_execute);
    if (profile_finalize) g_iface.QNN_INTERFACE_VER_NAME.profileFree(profile_finalize);
    /* graph is owned by context; freed via contextFree — no graphFree in QNN API */
    if (context) g_iface.QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    if (backend) g_iface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    dlclose(htp_lib);
    free(A); free(B); free(C);
    return 0;

fail:
    printf("NPU_DISPATCH_RESULT\n");
    printf("status: FAILED\n");
    printf("NPU_DISPATCH_END\n");
    free(exec_times);
    htp_perf_end();
    if (profile_execute)  g_iface.QNN_INTERFACE_VER_NAME.profileFree(profile_execute);
    if (profile_finalize) g_iface.QNN_INTERFACE_VER_NAME.profileFree(profile_finalize);
    if (context) g_iface.QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    if (backend) g_iface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    if (htp_lib) dlclose(htp_lib);
    free(A); free(B); free(C);
    return 1;
}
