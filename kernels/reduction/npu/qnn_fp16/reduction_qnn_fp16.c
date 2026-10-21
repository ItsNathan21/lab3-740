/*
 * reduction_qnn_fp16.c — QNN HTP FP16 sum-reduction harness (roofline anchor #2, NPU).
 *
 * sum = sum(x_i) via a SINGLE native QNN ReduceSum op — the controlled contrast
 * to the SAXPY NPU kernel, which had NO native op and decomposed into thousands
 * of ConvLayer tiles with a 50-110 s finalize (Discovery 57, the S8 catastrophe).
 * Reduction *does* have a native primitive, so `finalize_us` here is the readout
 * for prediction R-ii (kernels/reduction/DESIGN.md): does a native op escape the
 * finalize explosion, or is the HTP path unusable for ALL memory-bound work?
 *
 * Numerical caveat (decision R5): the graph is fp16 in/out. An fp16 ACCUMULATOR
 * over N elements of magnitude ~1 saturates once the partial sum exceeds fp16's
 * integer-exact range (~2048), so at large N the returned sum can be numerically
 * degenerate regardless of whether the op is "correct." We validate at small N
 * against the fp64 reference and report the large-N accumulation behavior as a
 * finding, not a pass/fail gate.
 *
 * Graph: input x [N] fp16 (APP_WRITE) --ReduceSum(axes=[0], keep_dims=1)--> [1] fp16 (APP_READ).
 *
 * Run on device:
 *   LD_LIBRARY_PATH=<dir> ADSP_LIBRARY_PATH=<dir> \
 *   ./reduction_qnn_fp16 [--seed 42] [--size 65536] [--reps 5] [--perf-mode sustained]
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

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* ── splitmix64 PRNG (bit-identical to reduction/reference.py + CPU/GPU/DSP) ── */
static uint64_t sm_state;
static void sm_seed(uint64_t s) { sm_state = s; }
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static float sm_float(void) { return (float)(sm_next() >> 11) * (1.0f / (float)(1ULL << 53)); }

/* ── __fp16 helpers ── */
typedef uint16_t fp16_t;
static fp16_t float_to_fp16(float f) {
#if defined(__aarch64__)
    __fp16 h = (__fp16)f; fp16_t bits; memcpy(&bits, &h, 2); return bits;
#else
    uint32_t bits; memcpy(&bits, &f, 4);
    uint32_t sign=(bits>>31)&1, exp=(bits>>23)&0xFF, mant=bits&0x7FFFFF;
    if (exp==0xFF) return (fp16_t)((sign<<15)|0x7C00|(mant?0x200:0));
    int ne=(int)exp-127+15;
    if (ne>=31) return (fp16_t)((sign<<15)|0x7C00);
    if (ne<=0)  return (fp16_t)(sign<<15);
    return (fp16_t)((sign<<15)|((uint32_t)ne<<10)|(mant>>13));
#endif
}
static float fp16_to_float(fp16_t h) {
#if defined(__aarch64__)
    __fp16 val; memcpy(&val, &h, 2); return (float)val;
#else
    uint32_t sign=(h>>15)&1, exp=(h>>10)&0x1F, mant=h&0x3FF;
    if (exp==0x1F) return sign?-__builtin_inff():__builtin_inff();
    if (exp==0)    return (sign?-1.0f:1.0f)*(float)mant/(float)(1<<24);
    float f; uint32_t bits=(sign<<31)|((exp-15+127)<<23)|(mant<<13); memcpy(&f,&bits,4); return f;
#endif
}

/* ── QNN interface plumbing ── */
typedef Qnn_ErrorHandle_t (*QnnInterface_GetProvidersFn_t)(const QnnInterface_t ***, uint32_t *);
static QnnInterface_t g_iface;

#define QNN(fn, ...) do {                                              \
    Qnn_ErrorHandle_t _rc = g_iface.QNN_INTERFACE_VER_NAME.fn(__VA_ARGS__); \
    if (_rc != QNN_SUCCESS) {                                          \
        fprintf(stderr, "[qnn] " #fn " failed: 0x%x\n", (unsigned)_rc);\
        goto fail;                                                     \
    }                                                                  \
} while(0)

/* ── HTP power mode (identical to saxpy/matmul NPU harness) ── */
static Qnn_DeviceHandle_t            g_device_handle = NULL;
static QnnHtpDevice_Infrastructure_t *g_htp_infra    = NULL;
static uint32_t                       g_power_cfg_id  = 0;

static int htp_perf_begin(void) {
    Qnn_ErrorHandle_t rc =
        g_iface.QNN_INTERFACE_VER_NAME.deviceCreate(NULL, NULL, &g_device_handle);
    if (rc != QNN_SUCCESS) { fprintf(stderr, "[perf] deviceCreate failed: 0x%x\n", (unsigned)rc); return -1; }
    QnnDevice_Infrastructure_t raw_infra = NULL;
    rc = g_iface.QNN_INTERFACE_VER_NAME.deviceGetInfrastructure(&raw_infra);
    if (rc != QNN_SUCCESS) { fprintf(stderr, "[perf] deviceGetInfrastructure failed: 0x%x\n", (unsigned)rc); return -1; }
    g_htp_infra = (QnnHtpDevice_Infrastructure_t *)raw_infra;
    if (!g_htp_infra || !g_htp_infra->perfInfra.createPowerConfigId) {
        fprintf(stderr, "[perf] infra ptr or createPowerConfigId is NULL\n"); return -1;
    }
    rc = g_htp_infra->perfInfra.createPowerConfigId(0, 0, &g_power_cfg_id);
    if (rc != QNN_SUCCESS) { fprintf(stderr, "[perf] createPowerConfigId(0,0) failed: 0x%x\n", (unsigned)rc); return -1; }

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
        .dcvsV3Config = {
            .contextId=g_power_cfg_id, .setDcvsEnable=1, .dcvsEnable=0,
            .powerMode=QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
            .setSleepLatency=1, .sleepLatency=40, .setSleepDisable=1, .sleepDisable=1,
            .setBusParams=1,
            .busVoltageCornerMin=DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerTarget=DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerMax=DCVS_VOLTAGE_VCORNER_TURBO,
            .setCoreParams=1,
            .coreVoltageCornerMin=DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerTarget=DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerMax=DCVS_VOLTAGE_VCORNER_TURBO,
        }
    };
    QnnHtpPerfInfrastructure_PowerConfig_t rpc = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME,
        .rpcPollingTimeConfig = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME,
    };
    const QnnHtpPerfInfrastructure_PowerConfig_t *cfgs[] = { &dcvs, &rpc, NULL };
    rc = g_htp_infra->perfInfra.setPowerConfig(g_power_cfg_id, cfgs);
    if (rc != QNN_SUCCESS) { fprintf(stderr, "[perf] setPowerConfig failed: 0x%x\n", (unsigned)rc); return -1; }
    fprintf(stderr, "[perf] sustained_high_performance active (config_id=%u)\n", g_power_cfg_id);
    return 0;
}
static void htp_perf_end(void) {
    if (g_htp_infra && g_power_cfg_id) { g_htp_infra->perfInfra.destroyPowerConfigId(g_power_cfg_id); g_power_cfg_id = 0; }
    if (g_device_handle) { g_iface.QNN_INTERFACE_VER_NAME.deviceFree(g_device_handle); g_device_handle = NULL; }
}

/* ── arg parsing ── */
static int   g_size      = 65536;  /* small default: fp16 accumulate degrades at large N */
static int   g_seed      = 42;
static int   g_reps      = 5;
static int   g_sustained = 0;
static void parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i+1 < argc) g_size = atoi(argv[++i]);
        if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) g_seed = atoi(argv[++i]);
        if (strcmp(argv[i], "--reps") == 0 && i+1 < argc) g_reps = atoi(argv[++i]);
        if (strcmp(argv[i], "--perf-mode") == 0 && i+1 < argc) {
            const char *m = argv[++i];
            g_sustained = (strcmp(m,"sustained")==0 || strcmp(m,"sustained_high_performance")==0) ? 1 : 0;
        }
    }
}

/* ── tensor initialiser (dtype-parametric; dims caller-owned) ── */
static void make_tensor(Qnn_Tensor_t *t, const char *name, Qnn_TensorType_t type,
                        Qnn_DataType_t dtype, uint32_t rank, uint32_t *dims) {
    memset(t, 0, sizeof(*t));
    t->version = QNN_TENSOR_VERSION_1;
    Qnn_TensorV1_t *v = &t->v1;
    v->name       = name;
    v->type       = type;
    v->dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    v->dataType   = dtype;
    v->quantizeParams.encodingDefinition   = QNN_DEFINITION_UNDEFINED;
    v->quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    v->rank       = rank;
    v->dimensions = dims;
    v->memType    = QNN_TENSORMEMTYPE_RAW;
}

int main(int argc, char **argv) {
    parse_args(argc, argv);
    int N = g_size;
    size_t elem = (size_t)N;
    size_t in_bytes  = elem * sizeof(fp16_t);
    size_t out_bytes = sizeof(fp16_t);

    /* ── 1. Input x (fp16), first N draws — bit-identical value stream to CPU/GPU/DSP ── */
    fp16_t *X = malloc(in_bytes);
    fp16_t  C = 0;   /* single scalar output */
    if (!X) { fprintf(stderr, "malloc failed\n"); return 1; }
    sm_seed((uint64_t)g_seed);
    for (size_t i = 0; i < elem; i++) X[i] = float_to_fp16(2.0f * sm_float() - 1.0f);

    /* axes = [0] (reduce the single axis); STATIC uint32 param tensor. */
    uint32_t axes_data[1] = { 0 };

    /* ── 2. Load QNN HTP backend ── */
    void *htp_lib = dlopen("libQnnHtp.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!htp_lib) { fprintf(stderr, "[qnn] dlopen libQnnHtp.so failed: %s\n", dlerror()); return 1; }
    QnnInterface_GetProvidersFn_t get_providers =
        (QnnInterface_GetProvidersFn_t)dlsym(htp_lib, "QnnInterface_getProviders");
    if (!get_providers) { fprintf(stderr, "[qnn] dlsym failed: %s\n", dlerror()); return 1; }
    const QnnInterface_t **providers = NULL; uint32_t num_providers = 0;
    if (get_providers(&providers, &num_providers) != QNN_SUCCESS || num_providers == 0) {
        fprintf(stderr, "[qnn] no providers\n"); return 1;
    }
    g_iface = providers[0][0];

    /* ── 3. Backend, Context, Graph ── */
    Qnn_BackendHandle_t backend = NULL;
    Qnn_ContextHandle_t context = NULL;
    Qnn_GraphHandle_t   graph   = NULL;
    Qnn_ProfileHandle_t profile_finalize = NULL;
    int64_t t0, t1, finalize_us = 0;
    int64_t *exec_times = NULL;

    QNN(backendCreate, NULL, NULL, &backend);
    if (g_sustained) { if (htp_perf_begin() != 0) goto fail; }
    QNN(contextCreate, backend, NULL, NULL, &context);
    QNN(graphCreate,   context, "reduce_sum_fp16", NULL, &graph);

    /* ── 4. Tensors ── */
    uint32_t dims_in[1]   = {(uint32_t)N};
    uint32_t dims_out[1]  = {1};
    uint32_t dims_axes[1] = {1};

    Qnn_Tensor_t tX, tOut, tAxes;
    make_tensor(&tX,   "input_x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 1, dims_in);
    make_tensor(&tOut, "output",  QNN_TENSOR_TYPE_APP_READ,  QNN_DATATYPE_FLOAT_16, 1, dims_out);
    /* axes param tensor: STATIC uint32, carries its data at creation. */
    make_tensor(&tAxes, "axes", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1, dims_axes);
    tAxes.v1.clientBuf.data     = axes_data;
    tAxes.v1.clientBuf.dataSize = (uint32_t)sizeof(axes_data);

    QNN(tensorCreateGraphTensor, graph, &tX);
    QNN(tensorCreateGraphTensor, graph, &tOut);
    /* The axes param is a STATIC tensor — like a weight, it must be registered
     * so it receives a valid id before being referenced in the op param. */
    QNN(tensorCreateGraphTensor, graph, &tAxes);

    /* ── 5. Single ReduceSum node ── */
    Qnn_Param_t params[2];
    memset(params, 0, sizeof(params));
    params[0].paramType = QNN_PARAMTYPE_TENSOR;
    params[0].name      = QNN_OP_REDUCE_SUM_PARAM_AXES;
    params[0].tensorParam = tAxes;
    params[1].paramType = QNN_PARAMTYPE_SCALAR;
    params[1].name      = QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS;
    params[1].scalarParam.dataType   = QNN_DATATYPE_BOOL_8;
    params[1].scalarParam.bool8Value = 1;   /* keep dims -> output shape [1] */

    Qnn_Tensor_t rs_in[1]  = { tX };
    Qnn_Tensor_t rs_out[1] = { tOut };
    Qnn_OpConfig_t op_reduce = {
        .version = QNN_OPCONFIG_VERSION_1,
        .v1 = {
            .name        = "reduce_sum",
            .packageName = QNN_OP_PACKAGE_NAME_QTI_AISW,
            .typeName    = QNN_OP_REDUCE_SUM,
            .numOfParams   = 2, .params = params,
            .numOfInputs   = 1, .inputTensors  = rs_in,
            .numOfOutputs  = 1, .outputTensors = rs_out,
        }
    };
    QNN(graphAddNode, graph, op_reduce);

    /* ── 6. Profile + finalize (the R-ii readout) ── */
    QNN(profileCreate, backend, QNN_PROFILE_LEVEL_DETAILED, &profile_finalize);
    t0 = now_ns();
    QNN(graphFinalize, graph, profile_finalize, NULL);
    t1 = now_ns();
    finalize_us = (t1 - t0) / 1000;

    /* ── 7. Bind buffers + warm-up execute ── */
    Qnn_Tensor_t exec_in[1]  = { tX };
    Qnn_Tensor_t exec_out[1] = { tOut };
    exec_in[0].v1.clientBuf.data      = X;  exec_in[0].v1.clientBuf.dataSize  = (uint32_t)in_bytes;
    exec_out[0].v1.clientBuf.data     = &C; exec_out[0].v1.clientBuf.dataSize = (uint32_t)out_bytes;
    {
        Qnn_ErrorHandle_t wrc = g_iface.QNN_INTERFACE_VER_NAME.graphExecute(
            graph, exec_in, 1, exec_out, 1, NULL, NULL);
        if (wrc != QNN_SUCCESS)
            fprintf(stderr, "[qnn] warm-up execute failed: 0x%x (continuing)\n", (unsigned)wrc);
    }

    /* ── 8. Timed rep loop ── */
    exec_times = malloc((size_t)g_reps * sizeof(int64_t));
    if (!exec_times) { fprintf(stderr, "malloc exec_times failed\n"); goto fail; }
    for (int r = 0; r < g_reps; r++) {
        t0 = now_ns();
        Qnn_ErrorHandle_t rc = g_iface.QNN_INTERFACE_VER_NAME.graphExecute(
            graph, exec_in, 1, exec_out, 1, NULL, NULL);
        t1 = now_ns();
        if (rc != QNN_SUCCESS) { fprintf(stderr, "[qnn] execute rep %d failed: 0x%x\n", r, (unsigned)rc); goto fail; }
        exec_times[r] = (t1 - t0) / 1000;
    }

    double sum_t = 0.0; int64_t min_us = exec_times[0], max_us = exec_times[0];
    for (int r = 0; r < g_reps; r++) {
        sum_t += (double)exec_times[r];
        if (exec_times[r] < min_us) min_us = exec_times[r];
        if (exec_times[r] > max_us) max_us = exec_times[r];
    }
    double mean_us = sum_t / (double)g_reps;
    double var = 0.0;
    for (int r = 0; r < g_reps; r++) { double d=(double)exec_times[r]-mean_us; var += d*d; }
    double stddev_us = sqrt(var / (double)g_reps);
    double cov_pct   = (mean_us > 0.0) ? (stddev_us/mean_us)*100.0 : 0.0;

    /* finalize profile event (matches saxpy harness) */
    const QnnProfile_EventId_t *events = NULL; uint32_t num_events = 0;
    g_iface.QNN_INTERFACE_VER_NAME.profileGetEvents(profile_finalize, &events, &num_events);
    int64_t finalize_profile_us = 0;
    for (uint32_t i = 0; i < num_events; i++) {
        QnnProfile_EventData_t ev;
        if (g_iface.QNN_INTERFACE_VER_NAME.profileGetEventData(events[i], &ev) != QNN_SUCCESS) continue;
        if (ev.type == QNN_PROFILE_EVENTTYPE_FINALIZE) finalize_profile_us = (int64_t)ev.value;
    }

    /* ── 9. Fingerprint = the reduced sum (fp16). Compared to fp64 reference. ── */
    double result = (double)fp16_to_float(C);

    printf("NPU_DISPATCH_RESULT\n");
    printf("status: SUCCESS\n");
    printf("power_mode: %s\n", g_sustained ? "sustained_high_performance" : "default");
    printf("reps: %d\n", g_reps);
    printf("execute_us_mean: %.1f\n", mean_us);
    printf("execute_us_stddev: %.1f\n", stddev_us);
    printf("execute_us_min: %lld\n", (long long)min_us);
    printf("execute_us_max: %lld\n", (long long)max_us);
    printf("execute_us_cov_pct: %.3f\n", cov_pct);
    printf("finalize_us: %lld\n", (long long)finalize_us);
    printf("finalize_profile_us: %lld\n", (long long)finalize_profile_us);
    printf("num_profile_events: %u\n", num_events);
    printf("precision: fp16\n");
    printf("elems: %d\n", N);
    printf("bytes: %lld\n", (long long)((int64_t)N * (int64_t)sizeof(fp16_t))); /* read-only */
    printf("fingerprint: %.6e\n", result);
    for (int r = 0; r < g_reps; r++) printf("rep_%d_us: %lld\n", r, (long long)exec_times[r]);
    printf("NPU_DISPATCH_END\n");

    free(exec_times);
    htp_perf_end();
    if (profile_finalize) g_iface.QNN_INTERFACE_VER_NAME.profileFree(profile_finalize);
    if (context) g_iface.QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    if (backend) g_iface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    dlclose(htp_lib);
    free(X);
    return 0;

fail:
    printf("NPU_DISPATCH_RESULT\n");
    printf("status: FAILED\n");
    printf("NPU_DISPATCH_END\n");
    free(exec_times);
    htp_perf_end();
    if (profile_finalize) g_iface.QNN_INTERFACE_VER_NAME.profileFree(profile_finalize);
    if (context) g_iface.QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    if (backend) g_iface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    if (htp_lib) dlclose(htp_lib);
    free(X);
    return 1;
}
