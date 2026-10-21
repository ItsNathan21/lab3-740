/*
 * matmul_qnn_int8.c — QNN HTP INT8 matmul harness
 *
 * Phase 7 Step 6 NPU kernel. Mirrors matmul_qnn_fp16 but uses
 * QNN_DATATYPE_SFIXED_POINT_8 with scale-offset quantization.
 * INT8 compute runs on the HMX (Hexagon Matrix eXtension) — the same
 * hardware that achieves the 45 TOPS peak rating on SM8750.
 *
 * Quantization:
 *   inputs A,B: scale = 1/127, offset = 0   (symmetric, real range [-1, 1])
 *   output  C:  scale = 4/127, offset = 0   (real range [-4, 4]; covers the
 *               ~±4 expected magnitude for 512×512 uniform random INT8 inputs)
 *
 * Build (see build.sh):
 *   android-clang -std=c11 -O2 -I include -ldl -lm -o matmul_qnn_int8 matmul_qnn_int8.c
 *
 * Run on device:
 *   LD_LIBRARY_PATH=/data/local/tmp/profiler-kernels/npu \
 *   ADSP_LIBRARY_PATH=/data/local/tmp/profiler-kernels/npu \
 *   ./matmul_qnn_int8 [--seed 42] [--n 512] [--reps 1] [--perf-mode default|sustained]
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
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

/* ── splitmix64 PRNG ──────────────────────────────────────────────────────── */

static uint64_t sm_state;
static void     sm_seed(uint64_t s) { sm_state = s; }
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* ── QNN interface ────────────────────────────────────────────────────────── */

typedef Qnn_ErrorHandle_t (*QnnInterface_GetProvidersFn_t)(
    const QnnInterface_t ***providers, uint32_t *num_providers);

static QnnInterface_t g_iface;
static Qnn_Version_t  g_provider_core_version = QNN_VERSION_INIT;

/*
 * This client is built against QNN core API 2.30. Newer QAIRT releases may
 * append functions to the same major-version interface. Use our compiled
 * interface prefix only after the provider reports major 2 and minor >= 30.
 */
static int select_compatible_provider(const QnnInterface_t **providers,
                                      uint32_t num_providers) {
    for (uint32_t i = 0; i < num_providers; i++) {
        const QnnInterface_t *provider = providers[i];
        if (!provider) continue;

        const Qnn_Version_t version = provider->apiVersion.coreApiVersion;
        fprintf(stderr,
                "[qnn] provider[%u] name=%s core_api=%u.%u.%u backend_api=%u.%u.%u\n",
                i,
                provider->providerName ? provider->providerName : "<unnamed>",
                version.major, version.minor, version.patch,
                provider->apiVersion.backendApiVersion.major,
                provider->apiVersion.backendApiVersion.minor,
                provider->apiVersion.backendApiVersion.patch);

        if (version.major == QNN_API_VERSION_MAJOR &&
            version.minor >= QNN_API_VERSION_MINOR) {
            g_iface = *provider;  /* copy only the compiled 2.30 prefix */
            g_provider_core_version = version;
            return 0;
        }
    }

    fprintf(stderr,
            "[qnn] no compatible provider: require core API %u.%u+ with major %u\n",
            QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR,
            QNN_API_VERSION_MAJOR);
    return -1;
}

#define QNN(fn, ...) do {                                                    \
    Qnn_ErrorHandle_t _rc = g_iface.QNN_INTERFACE_VER_NAME.fn(__VA_ARGS__); \
    if (_rc != QNN_SUCCESS) {                                                \
        fprintf(stderr, "[qnn] " #fn " failed: 0x%x\n", (unsigned)_rc);     \
        goto fail;                                                           \
    }                                                                        \
} while(0)

/* ── HTP power mode (perf infrastructure) ────────────────────────────────── */

static Qnn_DeviceHandle_t             g_device_handle = NULL;
static QnnHtpDevice_Infrastructure_t *g_htp_infra     = NULL;
static uint32_t                       g_power_cfg_id  = 0;

static int htp_perf_begin(void) {
    Qnn_ErrorHandle_t rc =
        g_iface.QNN_INTERFACE_VER_NAME.deviceCreate(NULL, NULL, &g_device_handle);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[perf] deviceCreate failed: 0x%x\n", (unsigned)rc);
        return -1;
    }

    QnnDevice_Infrastructure_t raw_infra = NULL;
    rc = g_iface.QNN_INTERFACE_VER_NAME.deviceGetInfrastructure(&raw_infra);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[perf] deviceGetInfrastructure failed: 0x%x\n", (unsigned)rc);
        return -1;
    }
    g_htp_infra = (QnnHtpDevice_Infrastructure_t *)raw_infra;
    if (!g_htp_infra ||
        g_htp_infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF ||
        !g_htp_infra->perfInfra.createPowerConfigId ||
        !g_htp_infra->perfInfra.destroyPowerConfigId ||
        !g_htp_infra->perfInfra.setPowerConfig) {
        fprintf(stderr, "[perf] incompatible HTP perf infrastructure table\n");
        return -1;
    }

    rc = g_htp_infra->perfInfra.createPowerConfigId(0, 0, &g_power_cfg_id);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[perf] createPowerConfigId failed: 0x%x\n", (unsigned)rc);
        return -1;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
        .dcvsV3Config = {
            .contextId               = g_power_cfg_id,
            .setDcvsEnable           = 1,
            .dcvsEnable              = 0,
            .powerMode               = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
            .setSleepLatency         = 1,
            .sleepLatency            = 40,
            .setSleepDisable         = 1,
            .sleepDisable            = 1,
            .setBusParams            = 1,
            .busVoltageCornerMin     = DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerTarget  = DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerMax     = DCVS_VOLTAGE_VCORNER_TURBO,
            .setCoreParams           = 1,
            .coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO,
        }
    };
    QnnHtpPerfInfrastructure_PowerConfig_t rpc = {
        .option               = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME,
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

static int g_n         = 512;
static int g_seed      = 42;
static int g_reps      = 1;
static int g_sustained = 0;

static int parse_int_option(int argc, char **argv, int *index,
                            const char *option, int *value_out) {
    if (*index + 1 >= argc) {
        fprintf(stderr, "%s requires an integer value\n", option);
        return -1;
    }
    const char *text = argv[++(*index)];
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < INT_MIN || value > INT_MAX) {
        fprintf(stderr, "invalid %s value: %s\n", option, text);
        return -1;
    }
    *value_out = (int)value;
    return 0;
}

static int parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0) {
            if (parse_int_option(argc, argv, &i, "--n", &g_n) != 0) return -1;
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (parse_int_option(argc, argv, &i, "--seed", &g_seed) != 0) return -1;
        } else if (strcmp(argv[i], "--reps") == 0) {
            if (parse_int_option(argc, argv, &i, "--reps", &g_reps) != 0) return -1;
        } else if (strcmp(argv[i], "--perf-mode") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--perf-mode requires default|sustained\n");
                return -1;
            }
            const char *m = argv[++i];
            if (strcmp(m, "sustained") == 0 ||
                strcmp(m, "sustained_high_performance") == 0) {
                g_sustained = 1;
            } else if (strcmp(m, "default") == 0) {
                g_sustained = 0;
            } else {
                fprintf(stderr,
                        "unsupported --perf-mode %s; expected default|sustained\n",
                        m);
                return -1;
            }
        } else {
            fprintf(stderr, "unsupported argument: %s\n", argv[i]);
            return -1;
        }
    }
    if (g_n <= 0 || g_reps <= 0) {
        fprintf(stderr, "--n and --reps must be positive\n");
        return -1;
    }
    return 0;
}

/* ── INT8 tensor helper ───────────────────────────────────────────────────── */

static void make_int8_tensor(Qnn_Tensor_t *t, const char *name,
                              Qnn_TensorType_t type,
                              uint32_t *dims,
                              float scale) {
    memset(t, 0, sizeof(*t));
    t->version = QNN_TENSOR_VERSION_1;
    Qnn_TensorV1_t *v = &t->v1;
    v->name       = name;
    v->type       = type;
    v->dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    v->dataType   = QNN_DATATYPE_SFIXED_POINT_8;
    v->quantizeParams.encodingDefinition    = QNN_DEFINITION_DEFINED;
    v->quantizeParams.quantizationEncoding  = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    v->quantizeParams.scaleOffsetEncoding.scale  = scale;
    v->quantizeParams.scaleOffsetEncoding.offset = 0;
    v->rank       = 2;
    v->dimensions = dims;
    v->memType    = QNN_TENSORMEMTYPE_RAW;
    /* clientBuf left NULL/0 — bound at execute time only */
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (parse_args(argc, argv) != 0) return 2;

    int    N     = g_n;
    size_t elem  = (size_t)N * N;
    size_t bytes = elem * sizeof(int8_t);

    /* Quantization scales:
     *   inputs:  scale_in  = 1/127  →  1 INT8 unit = 1/127 real  →  range [-1.0, 1.0]
     *   output:  scale_out = 4.0    →  1 INT8 unit = 4.0 real    →  range [-512, 508]
     *
     * scale_out = 4.0 covers the theoretical maximum for 512×512 INT8 dot products
     * with inputs in [-1, 1].  Typical output magnitude is ±7.5 (3σ ≈ ±22), which
     * maps to INT8 values of ±5 — coarse but unsaturated.
     *
     * Inputs are filled with the full signed INT8 range (0xFF mask → [-128, 127]) so
     * that dot products have zero mean and the output distribution is symmetric. */
    const float scale_in  = 1.0f / 127.0f;
    const float scale_out = 4.0f;

    /* ── 1. Allocate and fill input buffers ─────────────────────────────── */

    int8_t *A = malloc(bytes);
    int8_t *B = malloc(bytes);
    int8_t *C = malloc(bytes);
    if (!A || !B || !C) { fprintf(stderr, "malloc failed\n"); return 1; }

    sm_seed((uint64_t)g_seed);
    for (size_t i = 0; i < elem; i++) A[i] = (int8_t)(sm_next() & 0xFF);

    sm_seed((uint64_t)g_seed + 1);
    for (size_t i = 0; i < elem; i++) B[i] = (int8_t)(sm_next() & 0xFF);

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
        fprintf(stderr, "[qnn] dlsym QnnInterface_getProviders failed\n");
        return 1;
    }

    const QnnInterface_t **providers = NULL;
    uint32_t num_providers = 0;
    if (get_providers(&providers, &num_providers) != QNN_SUCCESS || num_providers == 0) {
        fprintf(stderr, "[qnn] QnnInterface_getProviders returned no providers\n");
        return 1;
    }
    if (select_compatible_provider(providers, num_providers) != 0) {
        dlclose(htp_lib);
        free(A); free(B); free(C);
        return 1;
    }

    /* ── 3. Backend, perf mode, context, graph ──────────────────────────── */

    Qnn_BackendHandle_t backend          = NULL;
    Qnn_ContextHandle_t context          = NULL;
    Qnn_GraphHandle_t   graph            = NULL;
    Qnn_ProfileHandle_t profile_finalize = NULL;
    int64_t             t0, t1;
    int64_t             finalize_us      = 0;
    int64_t            *exec_times       = NULL;

    QNN(backendCreate, NULL, NULL, &backend);

    if (g_sustained) {
        if (htp_perf_begin() != 0) goto fail;
    }

    QNN(contextCreate, backend, NULL, NULL, &context);
    QNN(graphCreate,   context, "matmul_int8", NULL, &graph);

    /* ── 4. Define tensors ──────────────────────────────────────────────── */

    uint32_t dims_A[2] = {(uint32_t)N, (uint32_t)N};
    uint32_t dims_B[2] = {(uint32_t)N, (uint32_t)N};
    uint32_t dims_C[2] = {(uint32_t)N, (uint32_t)N};

    Qnn_Tensor_t tA, tB, tC;
    make_int8_tensor(&tA, "input_a",  QNN_TENSOR_TYPE_APP_WRITE, dims_A, scale_in);
    make_int8_tensor(&tB, "input_b",  QNN_TENSOR_TYPE_APP_WRITE, dims_B, scale_in);
    make_int8_tensor(&tC, "output_c", QNN_TENSOR_TYPE_APP_READ,  dims_C, scale_out);

    QNN(tensorCreateGraphTensor, graph, &tA);
    QNN(tensorCreateGraphTensor, graph, &tB);
    QNN(tensorCreateGraphTensor, graph, &tC);

    /* ── 5. Add MatMul node ─────────────────────────────────────────────── */

    Qnn_Tensor_t in_tensors[2]  = {tA, tB};
    Qnn_Tensor_t out_tensors[1] = {tC};

    Qnn_OpConfig_t op = {
        .version = QNN_OPCONFIG_VERSION_1,
        .v1 = {
            .name          = "matmul_int8_0",
            .packageName   = QNN_OP_PACKAGE_NAME_QTI_AISW,
            .typeName      = QNN_OP_MAT_MUL,
            .numOfParams   = 0,
            .params        = NULL,
            .numOfInputs   = 2,
            .inputTensors  = in_tensors,
            .numOfOutputs  = 1,
            .outputTensors = out_tensors,
        }
    };

    QNN(graphAddNode, graph, op);

    /* ── 6. Profile + finalize ──────────────────────────────────────────── */

    QNN(profileCreate, backend, QNN_PROFILE_LEVEL_DETAILED, &profile_finalize);

    t0 = now_ns();
    QNN(graphFinalize, graph, profile_finalize, NULL);
    t1 = now_ns();
    finalize_us = (t1 - t0) / 1000;

    /* ── 7. Bind data buffers ───────────────────────────────────────────── */

    Qnn_Tensor_t exec_inputs[2]  = {tA, tB};
    Qnn_Tensor_t exec_outputs[1] = {tC};

    exec_inputs[0].v1.clientBuf.data     = A;
    exec_inputs[0].v1.clientBuf.dataSize = (uint32_t)bytes;
    exec_inputs[1].v1.clientBuf.data     = B;
    exec_inputs[1].v1.clientBuf.dataSize = (uint32_t)bytes;
    exec_outputs[0].v1.clientBuf.data    = C;
    exec_outputs[0].v1.clientBuf.dataSize = (uint32_t)bytes;

    /* ── 8. Warm-up execute ─────────────────────────────────────────────── */

    {
        Qnn_ErrorHandle_t wrc = g_iface.QNN_INTERFACE_VER_NAME.graphExecute(
            graph, exec_inputs, 2, exec_outputs, 1, NULL, NULL);
        if (wrc != QNN_SUCCESS)
            fprintf(stderr, "[qnn] warm-up execute failed: 0x%x (continuing)\n",
                    (unsigned)wrc);
    }

    /* ── 9. Rep loop ────────────────────────────────────────────────────── */

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

    /* ── 10. Stats ──────────────────────────────────────────────────────── */

    double  sum_t = 0.0;
    int64_t min_us = exec_times[0], max_us = exec_times[0];
    for (int r = 0; r < g_reps; r++) {
        sum_t += (double)exec_times[r];
        if (exec_times[r] < min_us) min_us = exec_times[r];
        if (exec_times[r] > max_us) max_us = exec_times[r];
    }
    double mean_us   = sum_t / (double)g_reps;
    double var = 0.0;
    for (int r = 0; r < g_reps; r++) {
        double d = (double)exec_times[r] - mean_us;
        var += d * d;
    }
    double stddev_us = sqrt(var / (double)g_reps);
    double cov_pct   = (mean_us > 0.0) ? (stddev_us / mean_us) * 100.0 : 0.0;

    /* ── 11. Finalize profile events ────────────────────────────────────── */

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

    /* ── 12. Fingerprint (dequantize → float) ───────────────────────────── */

    double sum_c = 0.0, frob_c = 0.0;
    for (size_t i = 0; i < elem; i++) {
        float v = (float)C[i] * scale_out;
        sum_c  += v;
        frob_c += (double)v * v;
    }
    frob_c = sqrt(frob_c);

    /* ── 13. Output block ───────────────────────────────────────────────── */

    printf("NPU_DISPATCH_RESULT\n");
    printf("status: SUCCESS\n");
    printf("precision: int8\n");
    printf("power_mode: %s\n",             g_sustained ? "sustained_high_performance" : "default");
    printf("qnn_core_api: %u.%u.%u\n",     g_provider_core_version.major,
                                              g_provider_core_version.minor,
                                              g_provider_core_version.patch);
    printf("reps: %d\n",                   g_reps);
    printf("execute_us_mean: %.1f\n",      mean_us);
    printf("execute_us_stddev: %.1f\n",    stddev_us);
    printf("execute_us_min: %lld\n",       (long long)min_us);
    printf("execute_us_max: %lld\n",       (long long)max_us);
    printf("execute_us_cov_pct: %.3f\n",   cov_pct);
    printf("finalize_us: %lld\n",          (long long)finalize_us);
    printf("finalize_profile_us: %lld\n",  (long long)finalize_profile_us);
    printf("num_profile_events: %u\n",     num_events);
    printf("matrix_n: %d\n",               N);
    printf("scale_in: %.6f\n",             scale_in);
    printf("scale_out: %.6f\n",            scale_out);
    printf("fingerprint_sum: %.6f\n",      sum_c);
    printf("fingerprint_frob: %.6f\n",     frob_c);
    for (int r = 0; r < g_reps; r++)
        printf("rep_%d_us: %lld\n", r, (long long)exec_times[r]);
    printf("NPU_DISPATCH_END\n");

    free(exec_times);

    /* ── cleanup ────────────────────────────────────────────────────────── */

    htp_perf_end();
    if (profile_finalize) g_iface.QNN_INTERFACE_VER_NAME.profileFree(profile_finalize);
    if (context) g_iface.QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    if (backend) g_iface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    dlclose(htp_lib);
    free(A); free(B); free(C);
    return 0;

fail:
    printf("NPU_DISPATCH_RESULT\n");
    printf("status: FAILED\n");
    printf("precision: int8\n");
    printf("NPU_DISPATCH_END\n");
    free(exec_times);
    htp_perf_end();
    if (profile_finalize) g_iface.QNN_INTERFACE_VER_NAME.profileFree(profile_finalize);
    if (context) g_iface.QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    if (backend) g_iface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    if (htp_lib) dlclose(htp_lib);
    free(A); free(B); free(C);
    return 1;
}
