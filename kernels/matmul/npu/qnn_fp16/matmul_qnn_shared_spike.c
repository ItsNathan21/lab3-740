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
#include "QNN/QnnMem.h"
#include "QNN/HTP/QnnHtpMem.h"
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

static int    g_n         = 512;
static int    g_seed      = 42;
static int    g_reps      = 1;
static int    g_sustained = 0;  /* 0=default power, 1=sustained_high_performance */
static int    g_shared    = 0;  /* 0=clientBuf (baseline), 1=rpcmem + memhandle */

/* ── rpcmem, resolved via dlsym (no link-time dependency) ─────────────────── */
static void  *g_rpc_so   = NULL;
static void *(*p_rpcmem_alloc)(int, uint32_t, int)  = NULL;
static void  (*p_rpcmem_free)(void *)               = NULL;
static int   (*p_rpcmem_to_fd)(void *)              = NULL;
static void  (*p_rpcmem_init)(void)                 = NULL;

#define RPCMEM_HEAP_ID_SYSTEM  25
#define RPCMEM_DEFAULT_FLAGS    1

static int rpcmem_bring_up(void) {
    const char *cands[] = { "libcdsprpc.so", "librpcmem.so" };
    for (unsigned i = 0; i < 2; i++) {
        g_rpc_so = dlopen(cands[i], RTLD_NOW | RTLD_GLOBAL);
        if (!g_rpc_so) continue;
        p_rpcmem_alloc = (void *(*)(int, uint32_t, int))dlsym(g_rpc_so, "rpcmem_alloc");
        p_rpcmem_free  = (void (*)(void *))dlsym(g_rpc_so, "rpcmem_free");
        p_rpcmem_to_fd = (int (*)(void *))dlsym(g_rpc_so, "rpcmem_to_fd");
        p_rpcmem_init  = (void (*)(void))dlsym(g_rpc_so, "rpcmem_init");
        if (p_rpcmem_alloc && p_rpcmem_free && p_rpcmem_to_fd) {
            if (p_rpcmem_init) p_rpcmem_init();
            fprintf(stderr, "[spike] rpcmem via %s\n", cands[i]);
            return 1;
        }
        dlclose(g_rpc_so); g_rpc_so = NULL;
    }
    fprintf(stderr, "[spike] rpcmem symbols not found\n");
    return 0;
}

/* ── Student memshare library (Lab 3 task N2) ─────────────────────────────── */
/*
 * With --memshare <path>, the shared allocation is delegated to a library the
 * student writes. It must export exactly three C symbols:
 *
 *     void* AllocateSharedBuffer(size_t size);   // NULL on failure
 *     void  FreeSharedBuffer(void* ptr);
 *     int   GetSharedBufferFd(void* ptr);        // -1 if not shareable
 *
 * Without --memshare, --shared uses the built-in rpcmem path above, which is
 * the staff reference. Both routes hit identical QNN memRegister code below,
 * so a student's numbers are directly comparable to the reference numbers.
 */
static void       *g_stu_so   = NULL;
static const char *g_memshare = NULL;
static void *(*p_stu_alloc)(size_t) = NULL;
static void  (*p_stu_free)(void *)  = NULL;
static int   (*p_stu_fd)(void *)    = NULL;

static int memshare_bring_up(const char *path) {
    g_stu_so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!g_stu_so) {
        fprintf(stderr, "[memshare] dlopen %s failed: %s\n", path, dlerror());
        return 0;
    }
    p_stu_alloc = (void *(*)(size_t))dlsym(g_stu_so, "AllocateSharedBuffer");
    p_stu_free  = (void  (*)(void *))dlsym(g_stu_so, "FreeSharedBuffer");
    p_stu_fd    = (int   (*)(void *))dlsym(g_stu_so, "GetSharedBufferFd");
    if (!p_stu_alloc || !p_stu_free || !p_stu_fd) {
        fprintf(stderr, "[memshare] %s is missing:%s%s%s\n", path,
                p_stu_alloc ? "" : " AllocateSharedBuffer",
                p_stu_free  ? "" : " FreeSharedBuffer",
                p_stu_fd    ? "" : " GetSharedBufferFd");
        fprintf(stderr, "[memshare] declare them extern \"C\" if building as C++.\n");
        dlclose(g_stu_so); g_stu_so = NULL;
        return 0;
    }
    fprintf(stderr, "[memshare] using %s\n", path);
    return 1;
}

/* Frees through whichever allocator produced the buffer. */
static void shared_release(void *p) {
    if (!p) return;
    if (p_stu_free)         p_stu_free(p);
    else if (p_rpcmem_free) p_rpcmem_free(p);
}

static void parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--shared") == 0) { g_shared = 1; }
        if (strcmp(argv[i], "--memshare") == 0 && i+1 < argc) {
            g_memshare = argv[++i];
            g_shared   = 1;   /* --memshare implies --shared */
        }
        if (strcmp(argv[i], "--n")    == 0 && i+1 < argc) { g_n    = atoi(argv[++i]); }
        if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) { g_seed = atoi(argv[++i]); }
        if (strcmp(argv[i], "--reps") == 0 && i+1 < argc) { g_reps = atoi(argv[++i]); }
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
                              uint32_t *dims) { /* caller-owned [2] array */
    memset(t, 0, sizeof(*t));
    t->version = QNN_TENSOR_VERSION_1;
    Qnn_TensorV1_t *v = &t->v1;
    v->name            = name;
    v->type            = type;
    v->dataFormat      = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    v->dataType        = QNN_DATATYPE_FLOAT_16;
    v->quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    v->quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    v->rank            = 2;
    v->dimensions      = dims;   /* pointer must outlive the graph */
    v->memType         = QNN_TENSORMEMTYPE_RAW;
    /* clientBuf left as NULL/0 — data is bound at graphExecute time only */
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    parse_args(argc, argv);

    int N = g_n;
    size_t elem  = (size_t)N * N;
    size_t bytes = elem * sizeof(fp16_t);

    /* ── 1. Allocate and fill input buffers ─────────────────────────────── */

    /* Baseline: three ordinary heap buffers — QNN must copy them across the
     * process boundary on every execute.
     * Shared: ONE rpcmem allocation holding all three tensors at known offsets.
     * rpcmem pages come from a heap that can be exported as a file descriptor,
     * which is the whole reason they can be mapped rather than copied. */
    fp16_t *A = NULL, *B = NULL, *C = NULL;
    void   *shared_base = NULL;
    int     shared_fd   = -1;
    size_t  off_A = 0, off_B = bytes, off_C = 2 * bytes;
    /* Declared here, not at the binding site: several `goto fail` paths above
     * that point would otherwise jump past the initialiser. */
    Qnn_MemHandle_t mem_handles[3] = { NULL, NULL, NULL };

    if (g_shared && g_memshare) {
        /* Student-supplied allocator (task N2). */
        if (!memshare_bring_up(g_memshare)) return 1;
        shared_base = p_stu_alloc((size_t)(3 * bytes));
        if (!shared_base) {
            fprintf(stderr, "[memshare] AllocateSharedBuffer(%zu) returned NULL\n",
                    (size_t)(3 * bytes));
            return 1;
        }
        shared_fd = p_stu_fd(shared_base);
        fprintf(stderr, "[memshare] base=%p fd=%d total=%zu\n",
                shared_base, shared_fd, (size_t)(3 * bytes));
        if (shared_fd < 0) {
            fprintf(stderr, "[memshare] GetSharedBufferFd returned %d — the buffer "
                            "is not exportable, so it cannot be shared.\n", shared_fd);
            shared_release(shared_base);
            return 1;
        }
        A = (fp16_t *)((char *)shared_base + off_A);
        B = (fp16_t *)((char *)shared_base + off_B);
        C = (fp16_t *)((char *)shared_base + off_C);
    } else if (g_shared) {
        if (!rpcmem_bring_up()) return 1;
        shared_base = p_rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                                     (int)(3 * bytes));
        if (!shared_base) { fprintf(stderr, "[spike] rpcmem_alloc failed\n"); return 1; }
        shared_fd = p_rpcmem_to_fd(shared_base);
        fprintf(stderr, "[spike] rpcmem base=%p fd=%d total=%zu\n",
                shared_base, shared_fd, 3 * bytes);
        if (shared_fd < 0) { fprintf(stderr, "[spike] no FD — cannot share\n"); return 1; }
        A = (fp16_t *)((char *)shared_base + off_A);
        B = (fp16_t *)((char *)shared_base + off_B);
        C = (fp16_t *)((char *)shared_base + off_C);
    } else {
        A = malloc(bytes);
        B = malloc(bytes);
        C = malloc(bytes);
        if (!A || !B || !C) { fprintf(stderr, "malloc failed\n"); return 1; }
    }

    sm_seed((uint64_t)g_seed);
    for (size_t i = 0; i < elem; i++) A[i] = float_to_fp16(sm_float());

    sm_seed((uint64_t)g_seed + 1);
    for (size_t i = 0; i < elem; i++) B[i] = float_to_fp16(sm_float());

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
    QNN(graphCreate,   context, "matmul_fp16", NULL /*config*/, &graph);

    /* ── 4. Define tensors ──────────────────────────────────────────────── */

    /* Dimension arrays must outlive the graph — keep on main() stack */
    uint32_t dims_A[2] = {(uint32_t)N, (uint32_t)N};
    uint32_t dims_B[2] = {(uint32_t)N, (uint32_t)N};
    uint32_t dims_C[2] = {(uint32_t)N, (uint32_t)N};

    Qnn_Tensor_t tA, tB, tC;
    make_fp16_tensor(&tA, "input_a",  QNN_TENSOR_TYPE_APP_WRITE, dims_A);
    make_fp16_tensor(&tB, "input_b",  QNN_TENSOR_TYPE_APP_WRITE, dims_B);
    make_fp16_tensor(&tC, "output_c", QNN_TENSOR_TYPE_APP_READ,  dims_C);

    /* Register tensors with the graph */
    QNN(tensorCreateGraphTensor, graph, &tA);
    QNN(tensorCreateGraphTensor, graph, &tB);
    QNN(tensorCreateGraphTensor, graph, &tC);

    /* ── 5. Add MatMul node ─────────────────────────────────────────────── */

    Qnn_Tensor_t in_tensors[2]  = {tA, tB};
    Qnn_Tensor_t out_tensors[1] = {tC};

    Qnn_OpConfig_t op = {
        .version = QNN_OPCONFIG_VERSION_1,
        .v1 = {
            .name       = "matmul_0",
            .packageName = QNN_OP_PACKAGE_NAME_QTI_AISW,
            .typeName   = QNN_OP_MAT_MUL,
            .numOfParams   = 0,
            .params        = NULL,
            .numOfInputs   = 2,
            .inputTensors  = in_tensors,
            .numOfOutputs  = 1,
            .outputTensors = out_tensors,
        }
    };

    QNN(graphAddNode, graph, op);

    /* ── 6. Profile handles (separate for finalize vs execute) ─────────── */

    QNN(profileCreate, backend, QNN_PROFILE_LEVEL_DETAILED, &profile_finalize);
    QNN(profileCreate, backend, QNN_PROFILE_LEVEL_DETAILED, &profile_execute);

    /* ── 7. Finalize (JIT compile graph to HTP microcode) ───────────────── */

    t0 = now_ns();
    QNN(graphFinalize, graph, profile_finalize, NULL /*signal*/);
    t1 = now_ns();
    finalize_us = (t1 - t0) / 1000;

    /* ── 8. Bind data buffers and execute ───────────────────────────────── */

    Qnn_Tensor_t exec_inputs[2]  = {tA, tB};
    Qnn_Tensor_t exec_outputs[1] = {tC};

    if (!g_shared) {
        /* Bind data buffers — only set at execute time, not at tensor registration */
        exec_inputs[0].v1.clientBuf.data     = A;
        exec_inputs[0].v1.clientBuf.dataSize = (uint32_t)bytes;
        exec_inputs[1].v1.clientBuf.data     = B;
        exec_inputs[1].v1.clientBuf.dataSize = (uint32_t)bytes;
        exec_outputs[0].v1.clientBuf.data    = C;
        exec_outputs[0].v1.clientBuf.dataSize = (uint32_t)bytes;
    } else {
        /* One FD, three registrations at different offsets — the HTP header
         * states each register call with a different offset yields its own
         * handle. Note there is no `version`/`v1` in these structs; last year's
         * harness assumed there was. */
        uint32_t mdims[2] = { (uint32_t)N, (uint32_t)N };
        size_t offs[3] = { off_A, off_B, off_C };
        for (int i = 0; i < 3; i++) {
            QnnMemHtp_Descriptor_t htp;
            memset(&htp, 0, sizeof(htp));
            htp.type = QNN_HTP_MEM_SHARED_BUFFER;
            htp.size = (uint64_t)(3 * bytes);
            htp.sharedBufferConfig.fd     = shared_fd;
            htp.sharedBufferConfig.offset = (uint64_t)offs[i];

            Qnn_MemDescriptor_t md;
            memset(&md, 0, sizeof(md));
            md.memShape.numDim      = 2;
            md.memShape.dimSize     = mdims;
            md.memShape.shapeConfig = NULL;
            md.dataType             = QNN_DATATYPE_FLOAT_16;
            md.memType              = QNN_MEM_TYPE_CUSTOM;
            md.customInfo           = &htp;

            Qnn_ErrorHandle_t mrc = g_iface.QNN_INTERFACE_VER_NAME.memRegister(
                context, &md, 1u, &mem_handles[i]);
            if (mrc != QNN_SUCCESS || !mem_handles[i]) {
                fprintf(stderr, "[spike] memRegister[%d] failed: 0x%x\n",
                        i, (unsigned)mrc);
                return 1;
            }
        }
        fprintf(stderr, "[spike] registered 3 memhandles OK\n");

        exec_inputs[0].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
        exec_inputs[0].v1.memHandle = mem_handles[0];
        exec_inputs[1].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
        exec_inputs[1].v1.memHandle = mem_handles[1];
        exec_outputs[0].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
        exec_outputs[0].v1.memHandle = mem_handles[2];
    }

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

    /* ── 14. Output block ───────────────────────────────────────────────── */

    printf("NPU_DISPATCH_RESULT\n");
    printf("status: SUCCESS\n");
    printf("power_mode: %s\n",             g_sustained ? "sustained_high_performance" : "default");
    printf("shared_impl: %s\n",            !g_shared ? "none_clientbuf"
                                            : (g_memshare ? "memshare" : "builtin_rpcmem"));
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
    printf("matrix_n: %d\n",               N);
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
    if (g_shared) {
        for (int i = 0; i < 3; i++)
            if (mem_handles[i])
                g_iface.QNN_INTERFACE_VER_NAME.memDeRegister(&mem_handles[i], 1u);
    }
    if (context) g_iface.QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    if (backend) g_iface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    dlclose(htp_lib);
    /* A/B/C are offsets into ONE rpcmem allocation — never free() them
     * individually. Release the single buffer through rpcmem instead. */
    if (g_shared) {
        shared_release(shared_base);
        if (g_rpc_so) dlclose(g_rpc_so);
    } else {
        free(A); free(B); free(C);
    }
    return 0;

fail:
    printf("NPU_DISPATCH_RESULT\n");
    printf("status: FAILED\n");
    printf("NPU_DISPATCH_END\n");
    free(exec_times);
    htp_perf_end();
    if (profile_execute)  g_iface.QNN_INTERFACE_VER_NAME.profileFree(profile_execute);
    if (profile_finalize) g_iface.QNN_INTERFACE_VER_NAME.profileFree(profile_finalize);
    if (g_shared) {
        for (int i = 0; i < 3; i++)
            if (mem_handles[i])
                g_iface.QNN_INTERFACE_VER_NAME.memDeRegister(&mem_handles[i], 1u);
    }
    if (context) g_iface.QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    if (backend) g_iface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    if (htp_lib) dlclose(htp_lib);
    if (g_shared) {
        shared_release(shared_base);
        if (g_rpc_so) dlclose(g_rpc_so);
    } else {
        free(A); free(B); free(C);
    }
    return 1;
}
