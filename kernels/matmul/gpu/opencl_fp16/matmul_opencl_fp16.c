/*
 * matmul_opencl_fp16.c — Phase 4 Step 9 FP16 GPU kernel.
 *
 * 512×512 matrix multiply using OpenCL half-precision (FP16) buffers.
 * Inputs A and B are stored as half (16-bit IEEE 754), output C is half.
 * The inner accumulator is float (32-bit) for numerical stability — FP16
 * accumulation over 512 iterations would overflow or lose too much precision.
 *
 * Why FP16:
 *   Adreno 830 has 2× FP32 throughput for FP16 arithmetic (confirmed via
 *   cl_khr_fp16 in discovery 27). FP16 buffers also use half the memory
 *   bandwidth (1 MB per matrix vs 2 MB for FP32), so both the ALU and
 *   bandwidth bottlenecks benefit. Expected speedup: ~2× vs FP32 naive.
 *
 * Design (same as matmul_opencl_fp32.c unless noted):
 *   - dlopen /vendor/lib64/libOpenCL.so — same Qualcomm loader path
 *   - On-device online compiler — cl_khr_fp16 available (discovery 27)
 *   - CL_QUEUE_PROFILING_ENABLE — CL_PROFILING_COMMAND_START/END timing
 *   - HOST_UNIFIED_MEMORY=YES — CL_MEM_COPY_HOST_PTR for A, B
 *   - Host buffers: uint16_t (raw FP16 bits), converted from float via
 *     ARM __fp16 native type (arm64 compiler extension)
 *   - Same splitmix64 inputs as FP32 (seed=42 → A, seed=43 → B) so
 *     fingerprints can be compared — FP16 rounding will cause a small
 *     deviation (~100–10000 units) vs the FP32 value of 33585655.900771
 *
 * Argv (same as FP32 variant):
 *   --size  N     matrix dimension (default 512; must be > 0)
 *   --seed  N     splitmix64 seed for A/B (default 42)
 *   --wg-x  N     workgroup X dimension (default 16)
 *   --wg-y  N     workgroup Y dimension (default 16)
 *   --wg-z  N     workgroup Z dimension (default 1; must be 1)
 *
 * Stdout protocol: same GPU_DISPATCH_RESULT v1 format as FP32.
 * Build: bash kernels/matmul/gpu/opencl_fp16/build.sh
 */

#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* ── Section 1: OpenCL type definitions ─────────────────────────────────────
 * Identical to matmul_opencl_fp32.c — inline to avoid needing CL headers.
 */
typedef void *cl_platform_id;
typedef void *cl_device_id;
typedef void *cl_context;
typedef void *cl_command_queue;
typedef void *cl_mem;
typedef void *cl_program;
typedef void *cl_kernel;
typedef void *cl_event;

typedef int32_t   cl_int;
typedef uint32_t  cl_uint;
typedef uint64_t  cl_ulong;
typedef uint64_t  cl_bitfield;
typedef uint32_t  cl_bool;
typedef uint64_t  cl_device_type;
typedef uintptr_t cl_device_info;
typedef uintptr_t cl_platform_info;
typedef uintptr_t cl_context_properties;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_queue_properties;

#define CL_SUCCESS              0
#define CL_BUILD_PROGRAM_FAILURE (-11)
#define CL_INVALID_VALUE        (-30)
#define CL_DEVICE_TYPE_GPU      ((cl_device_type)(1 << 2))
#define CL_CONTEXT_PLATFORM     ((cl_context_properties)0x1084)
#define CL_MEM_READ_WRITE       ((cl_mem_flags)(1 << 0))
#define CL_MEM_COPY_HOST_PTR    ((cl_mem_flags)(1 << 5))
#define CL_QUEUE_PROPERTIES     ((cl_queue_properties)0x1093)
#define CL_QUEUE_PROFILING_ENABLE ((cl_queue_properties)(1 << 1))
#define CL_PROFILING_COMMAND_START ((cl_uint)0x1282)
#define CL_PROFILING_COMMAND_END   ((cl_uint)0x1283)
#define CL_PROGRAM_BUILD_LOG    ((cl_uint)0x1183)
#define CL_TRUE                 ((cl_bool)1)


/* ── Section 2: FP16 OpenCL C kernel source ──────────────────────────────────
 *
 * Key design decisions vs FP32:
 *
 * 1. #pragma OPENCL EXTENSION cl_khr_fp16 : enable
 *    Required to use the `half` type. Confirmed available on Adreno 830
 *    (discovery 27 listed cl_khr_fp16 in the 57-extension set).
 *
 * 2. Inputs/outputs use `half` (16-bit IEEE 754).
 *    The GPU loads 2 bytes per element instead of 4, halving DRAM bandwidth.
 *
 * 3. Accumulator is `float` (32-bit).
 *    FP16 has only 10 mantissa bits. Over 512 iterations of mad(), the
 *    partial sums would be catastrophically rounded if kept in half. A
 *    float accumulator gives 23 bits of mantissa throughout the inner loop;
 *    only the final store back to half loses precision (rounding to 10 bits).
 *    This is the standard mixed-precision approach used in deep learning.
 *
 * 4. The `(float)A[...] * (float)B[...]` casts promote each half load to
 *    float before multiplication. Adreno's VFPU handles float natively.
 *    mad(float, float, float) maps to the FMA instruction on Adreno 830.
 *
 * 5. C[...] = (half)acc converts the float accumulator back to half for
 *    storage — this is where the FP16 rounding error enters relative to
 *    the FP32 result.
 */
static const char *KERNEL_SOURCE =
    /* Enable the cl_khr_fp16 extension — required before any 'half' usage. */
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "\n"
    "__kernel void matmul_fp16(\n"
    "    __global const half* A,\n"   /* FP16 input A, row-major */
    "    __global const half* B,\n"   /* FP16 input B, row-major */
    "    __global       half* C,\n"   /* FP16 output C, row-major */
    "    int N)\n"
    "{\n"
    "    int row = (int)get_global_id(0);\n"
    "    int col = (int)get_global_id(1);\n"
    "    /* Bounds check: global size may be rounded up to workgroup multiple. */\n"
    "    if (row >= N || col >= N) return;\n"
    "    /* float accumulator for numerical stability — avoids FP16 overflow\n"
    "     * and rounding over 512 inner-loop iterations. */\n"
    "    float acc = 0.0f;\n"
    "    for (int k = 0; k < N; k++) {\n"
    "        /* Promote half inputs to float before multiply, then FMA. */\n"
    "        acc = mad((float)A[row * N + k], (float)B[k * N + col], acc);\n"
    "    }\n"
    "    /* Store result back as half — only source of FP16 rounding error. */\n"
    "    C[row * N + col] = (half)acc;\n"
    "}\n";


/* ── Section 3: Function pointer types and globals ───────────────────────────
 * Same set as matmul_opencl_fp32.c — all loaded by load_opencl().
 */
typedef cl_int (*pfn_clGetPlatformIDs)(
    cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*pfn_clGetDeviceIDs)(
    cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_context (*pfn_clCreateContext)(
    const cl_context_properties *, cl_uint, const cl_device_id *,
    void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
typedef cl_command_queue (*pfn_clCreateCommandQueueWithProperties)(
    cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
typedef cl_mem (*pfn_clCreateBuffer)(
    cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_program (*pfn_clCreateProgramWithSource)(
    cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_int (*pfn_clBuildProgram)(
    cl_program, cl_uint, const cl_device_id *, const char *,
    void (*)(cl_program, void *), void *);
typedef cl_int (*pfn_clGetProgramBuildInfo)(
    cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
typedef cl_kernel (*pfn_clCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_int (*pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(
    cl_command_queue, cl_kernel, cl_uint, const size_t *,
    const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*pfn_clEnqueueReadBuffer)(
    cl_command_queue, cl_mem, cl_bool, size_t, size_t,
    void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*pfn_clFinish)(cl_command_queue);
typedef cl_int (*pfn_clGetEventProfilingInfo)(
    cl_event, cl_uint, size_t, void *, size_t *);
typedef cl_int (*pfn_clReleaseEvent)(cl_event);
typedef cl_int (*pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (*pfn_clReleaseProgram)(cl_program);
typedef cl_int (*pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (*pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*pfn_clReleaseContext)(cl_context);

static pfn_clGetPlatformIDs                   clGetPlatformIDs_fn                   = NULL;
static pfn_clGetDeviceIDs                     clGetDeviceIDs_fn                     = NULL;
static pfn_clCreateContext                    clCreateContext_fn                    = NULL;
static pfn_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties_fn = NULL;
static pfn_clCreateBuffer                     clCreateBuffer_fn                     = NULL;
static pfn_clCreateProgramWithSource          clCreateProgramWithSource_fn          = NULL;
static pfn_clBuildProgram                     clBuildProgram_fn                     = NULL;
static pfn_clGetProgramBuildInfo              clGetProgramBuildInfo_fn              = NULL;
static pfn_clCreateKernel                     clCreateKernel_fn                     = NULL;
static pfn_clSetKernelArg                     clSetKernelArg_fn                     = NULL;
static pfn_clEnqueueNDRangeKernel             clEnqueueNDRangeKernel_fn             = NULL;
static pfn_clEnqueueReadBuffer                clEnqueueReadBuffer_fn                = NULL;
static pfn_clFinish                           clFinish_fn                           = NULL;
static pfn_clGetEventProfilingInfo            clGetEventProfilingInfo_fn            = NULL;
static pfn_clReleaseEvent                     clReleaseEvent_fn                     = NULL;
static pfn_clReleaseKernel                    clReleaseKernel_fn                    = NULL;
static pfn_clReleaseProgram                   clReleaseProgram_fn                   = NULL;
static pfn_clReleaseMemObject                 clReleaseMemObject_fn                 = NULL;
static pfn_clReleaseCommandQueue              clReleaseCommandQueue_fn              = NULL;
static pfn_clReleaseContext                   clReleaseContext_fn                   = NULL;


/* ── Section 4: Helpers ──────────────────────────────────────────────────────*/

static void fail_dispatch(const char *msg) {
    printf("GPU_DISPATCH_RESULT\n");
    printf("status: FAILED\n");
    printf("error: %s\n", msg);
    printf("GPU_DISPATCH_END\n");
    fflush(stdout);
    exit(1);
}

static void fail_dispatch_fmt(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fail_dispatch(buf);
}

static void *load_sym(void *handle, const char *name) {
    void *sym = dlsym(handle, name);
    if (!sym) {
        char buf[256];
        snprintf(buf, sizeof(buf), "dlsym(%s) failed: %s", name, dlerror());
        fail_dispatch(buf);
    }
    return sym;
}

/* load_opencl: identical to FP32 variant — same library, same function set. */
static void load_opencl(void) {
    const char *path = "/vendor/lib64/libOpenCL.so";
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        char buf[256];
        snprintf(buf, sizeof(buf), "dlopen(%s) failed: %s", path, dlerror());
        fail_dispatch(buf);
    }
    clGetPlatformIDs_fn                   = (pfn_clGetPlatformIDs)                   load_sym(h, "clGetPlatformIDs");
    clGetDeviceIDs_fn                     = (pfn_clGetDeviceIDs)                     load_sym(h, "clGetDeviceIDs");
    clCreateContext_fn                    = (pfn_clCreateContext)                    load_sym(h, "clCreateContext");
    clCreateCommandQueueWithProperties_fn = (pfn_clCreateCommandQueueWithProperties) load_sym(h, "clCreateCommandQueueWithProperties");
    clCreateBuffer_fn                     = (pfn_clCreateBuffer)                     load_sym(h, "clCreateBuffer");
    clCreateProgramWithSource_fn          = (pfn_clCreateProgramWithSource)          load_sym(h, "clCreateProgramWithSource");
    clBuildProgram_fn                     = (pfn_clBuildProgram)                     load_sym(h, "clBuildProgram");
    clGetProgramBuildInfo_fn              = (pfn_clGetProgramBuildInfo)              load_sym(h, "clGetProgramBuildInfo");
    clCreateKernel_fn                     = (pfn_clCreateKernel)                     load_sym(h, "clCreateKernel");
    clSetKernelArg_fn                     = (pfn_clSetKernelArg)                     load_sym(h, "clSetKernelArg");
    clEnqueueNDRangeKernel_fn             = (pfn_clEnqueueNDRangeKernel)             load_sym(h, "clEnqueueNDRangeKernel");
    clEnqueueReadBuffer_fn                = (pfn_clEnqueueReadBuffer)                load_sym(h, "clEnqueueReadBuffer");
    clFinish_fn                           = (pfn_clFinish)                           load_sym(h, "clFinish");
    clGetEventProfilingInfo_fn            = (pfn_clGetEventProfilingInfo)            load_sym(h, "clGetEventProfilingInfo");
    clReleaseEvent_fn                     = (pfn_clReleaseEvent)                     load_sym(h, "clReleaseEvent");
    clReleaseKernel_fn                    = (pfn_clReleaseKernel)                    load_sym(h, "clReleaseKernel");
    clReleaseProgram_fn                   = (pfn_clReleaseProgram)                   load_sym(h, "clReleaseProgram");
    clReleaseMemObject_fn                 = (pfn_clReleaseMemObject)                 load_sym(h, "clReleaseMemObject");
    clReleaseCommandQueue_fn              = (pfn_clReleaseCommandQueue)              load_sym(h, "clReleaseCommandQueue");
    clReleaseContext_fn                   = (pfn_clReleaseContext)                   load_sym(h, "clReleaseContext");
}

/* splitmix64_next: identical to FP32 variant. Same seed → same float values
 * before the float→half conversion, so inputs are a deterministic subset
 * of the FP32 inputs (rounded to the nearest representable half). */
static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

/* f32_to_f16: convert a float to a uint16_t containing the FP16 bit pattern.
 *
 * Uses ARM's __fp16 native 16-bit float type, which the arm64 ABI supports
 * natively. The cast `(__fp16)f` invokes the hardware float-to-half conversion
 * instruction (FCVT). __builtin_memcpy avoids strict-aliasing UB when reading
 * the resulting bits as uint16_t.
 *
 * This is the canonical approach on arm64 — no software bit-twiddling needed.
 */
static uint16_t f32_to_f16(float f) {
    __fp16 h = (__fp16)f;
    uint16_t bits;
    __builtin_memcpy(&bits, &h, sizeof(bits));
    return bits;
}

/* f16_to_f32: convert a uint16_t containing FP16 bits back to float.
 * Used for fingerprint computation after reading the half output buffer. */
static float f16_to_f32(uint16_t bits) {
    __fp16 h;
    __builtin_memcpy(&h, &bits, sizeof(h));
    return (float)h;
}

/* fill_matrix_fp16: fill an N×N buffer of uint16_t with FP16 values.
 *
 * Algorithm:
 * 1. Generate float values in [0, 1) using the same splitmix64 mapping as
 *    the FP32 kernel (same seed → same float values).
 * 2. Convert each float to FP16 (f32_to_f16). Values near 1.0 round to
 *    the nearest representable half (~0.001 step near 1.0 for FP16).
 * 3. Store the uint16_t bit pattern — this is what gets uploaded to the GPU.
 *
 * The FP16 inputs are a rounded version of the FP32 inputs. The fingerprint
 * will differ from the FP32 reference (33585655.900771) because:
 *   (a) input rounding: float → half loses ~7 bits of mantissa per element
 *   (b) output rounding: float accumulator → half stores a rounded result
 */
static void fill_matrix_fp16(uint16_t *mat, int N, uint64_t seed) {
    uint64_t state = seed;
    int total = N * N;
    for (int i = 0; i < total; i++) {
        uint64_t v = splitmix64_next(&state);
        /* Same float mapping as fill_matrix() in FP32 kernel. */
        float f = (float)((double)v / 18446744073709551616.0);
        mat[i] = f32_to_f16(f);
    }
}

static size_t round_up(int n, int m) {
    return (size_t)(((n + m - 1) / m) * m);
}


/* ── Section 5: main ─────────────────────────────────────────────────────────*/

int main(int argc, char **argv) {
    /* ── Parse arguments — same as FP32 ─────────────────────────────────── */
    int      size = 512;
    uint64_t seed = 42;
    int      wg_x = 16;
    int      wg_y = 16;
    int      wg_z = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--wg-x") == 0 && i + 1 < argc) {
            wg_x = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--wg-y") == 0 && i + 1 < argc) {
            wg_y = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--wg-z") == 0 && i + 1 < argc) {
            wg_z = atoi(argv[++i]);
        }
    }

    if (size <= 0 || wg_x <= 0 || wg_y <= 0 || wg_z != 1) {
        fail_dispatch("invalid arguments: size and wg-x/y must be >0; wg-z must be 1");
    }

    load_opencl();

    /* ── Get platform + GPU device — identical to FP32 ──────────────────── */
    cl_uint nplat = 0;
    cl_int rc = clGetPlatformIDs_fn(0, NULL, &nplat);
    if (rc != CL_SUCCESS || nplat == 0) {
        fail_dispatch_fmt("clGetPlatformIDs failed: rc=%d, nplat=%u", rc, nplat);
    }
    cl_platform_id platform = NULL;
    clGetPlatformIDs_fn(1, &platform, NULL);

    cl_uint ndev = 0;
    rc = clGetDeviceIDs_fn(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &ndev);
    if (rc != CL_SUCCESS || ndev == 0) {
        fail_dispatch_fmt("clGetDeviceIDs(GPU) failed: rc=%d, ndev=%u", rc, ndev);
    }
    cl_device_id device = NULL;
    clGetDeviceIDs_fn(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context_properties ctx_props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        (cl_context_properties)0
    };
    cl_context ctx = clCreateContext_fn(ctx_props, 1, &device, NULL, NULL, &rc);
    if (rc != CL_SUCCESS || !ctx) {
        fail_dispatch_fmt("clCreateContext failed: rc=%d", rc);
    }

    cl_queue_properties q_props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        (cl_queue_properties)0
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties_fn(
        ctx, device, q_props, &rc);
    if (rc != CL_SUCCESS || !queue) {
        clReleaseContext_fn(ctx);
        fail_dispatch_fmt("clCreateCommandQueueWithProperties failed: rc=%d", rc);
    }

    /* ── Allocate host matrices — FP16: uint16_t, 2 bytes/element ───────── */
    /* FP16 buffers are half the size of FP32. 512×512×2 = 524288 bytes = 0.5 MB
     * per matrix vs 1 MB for FP32. Total: 1.5 MB (A + B + C) vs 3 MB for FP32. */
    size_t nbytes = (size_t)size * (size_t)size * sizeof(uint16_t);

    uint16_t *h_A = (uint16_t *)malloc(nbytes);
    uint16_t *h_B = (uint16_t *)malloc(nbytes);
    uint16_t *h_C = (uint16_t *)malloc(nbytes);

    if (!h_A || !h_B || !h_C) {
        fail_dispatch("malloc failed for host FP16 matrices (out of memory)");
    }

    /* Fill A and B with FP16 values derived from the same splitmix64 seeds
     * as the FP32 kernel. seed=42 for A, seed=43 (seed+1) for B. */
    fill_matrix_fp16(h_A, size, seed);
    fill_matrix_fp16(h_B, size, seed + 1);
    memset(h_C, 0, nbytes);

    /* ── Create OpenCL buffers — same flags, different size ─────────────── */
    /* The GPU sees these as `__global half*` buffers (via the kernel source).
     * CL_MEM_COPY_HOST_PTR uploads h_A and h_B to device memory at creation. */
    cl_mem buf_A = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     nbytes, h_A, &rc);
    if (rc != CL_SUCCESS || !buf_A) {
        fail_dispatch_fmt("clCreateBuffer(A fp16) failed: rc=%d", rc);
    }
    cl_mem buf_B = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     nbytes, h_B, &rc);
    if (rc != CL_SUCCESS || !buf_B) {
        clReleaseMemObject_fn(buf_A);
        fail_dispatch_fmt("clCreateBuffer(B fp16) failed: rc=%d", rc);
    }
    cl_mem buf_C = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE,
                                     nbytes, NULL, &rc);
    if (rc != CL_SUCCESS || !buf_C) {
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        fail_dispatch_fmt("clCreateBuffer(C fp16) failed: rc=%d", rc);
    }

    /* ── Build the FP16 kernel program ───────────────────────────────────── */
    /* The pragma in KERNEL_SOURCE enables cl_khr_fp16.
     * -cl-fast-relaxed-math allows the compiler to use native FMA units.
     * First compile may take 2–5 s (on-device Adreno online compiler). */
    cl_program program = clCreateProgramWithSource_fn(
        ctx, 1, &KERNEL_SOURCE, NULL, &rc);
    if (rc != CL_SUCCESS || !program) {
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clCreateProgramWithSource failed: rc=%d", rc);
    }

    rc = clBuildProgram_fn(program, 1, &device,
                           "-cl-fast-relaxed-math -cl-std=CL3.0",
                           NULL, NULL);
    if (rc != CL_SUCCESS) {
        char build_log[4096];
        memset(build_log, 0, sizeof(build_log));
        clGetProgramBuildInfo_fn(program, device, CL_PROGRAM_BUILD_LOG,
                                 sizeof(build_log) - 1, build_log, NULL);
        char err_buf[512];
        snprintf(err_buf, sizeof(err_buf),
                 "clBuildProgram(fp16) failed: rc=%d | log: %.200s", rc, build_log);
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch(err_buf);
    }

    /* ── Create kernel "matmul_fp16" and set arguments ───────────────────── */
    cl_kernel kernel = clCreateKernel_fn(program, "matmul_fp16", &rc);
    if (rc != CL_SUCCESS || !kernel) {
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clCreateKernel(matmul_fp16) failed: rc=%d", rc);
    }

    rc  = clSetKernelArg_fn(kernel, 0, sizeof(cl_mem), &buf_A);
    rc |= clSetKernelArg_fn(kernel, 1, sizeof(cl_mem), &buf_B);
    rc |= clSetKernelArg_fn(kernel, 2, sizeof(cl_mem), &buf_C);
    rc |= clSetKernelArg_fn(kernel, 3, sizeof(cl_int), &size);
    if (rc != CL_SUCCESS) {
        clReleaseKernel_fn(kernel);
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clSetKernelArg failed: rc=%d", rc);
    }

    /* ── Enqueue NDRange — same workgroup geometry as FP32 ──────────────── */
    size_t global[2] = { round_up(size, wg_x), round_up(size, wg_y) };
    size_t local[2]  = { (size_t)wg_x, (size_t)wg_y };
    cl_event kernel_event = NULL;

    rc = clEnqueueNDRangeKernel_fn(
        queue, kernel, 2, NULL, global, local, 0, NULL, &kernel_event);
    if (rc != CL_SUCCESS) {
        clReleaseKernel_fn(kernel);
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clEnqueueNDRangeKernel failed: rc=%d", rc);
    }

    rc = clFinish_fn(queue);
    if (rc != CL_SUCCESS) {
        clReleaseEvent_fn(kernel_event);
        clReleaseKernel_fn(kernel);
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clFinish failed: rc=%d", rc);
    }

    /* ── Read profiling timestamps — identical to FP32 ───────────────────── */
    cl_ulong cl_start_ns = 0, cl_end_ns = 0;
    rc  = clGetEventProfilingInfo_fn(kernel_event, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &cl_start_ns, NULL);
    rc |= clGetEventProfilingInfo_fn(kernel_event, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &cl_end_ns, NULL);
    if (rc != CL_SUCCESS) {
        clReleaseEvent_fn(kernel_event);
        clReleaseKernel_fn(kernel);
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clGetEventProfilingInfo failed: rc=%d", rc);
    }
    cl_ulong duration_ns = cl_end_ns - cl_start_ns;
    clReleaseEvent_fn(kernel_event);

    /* ── Read back C and compute fingerprint ──────────────────────────────── */
    /* Read back the uint16_t FP16 buffer, then convert each element to float
     * for fingerprint computation. Using double accumulation for accuracy. */
    rc = clEnqueueReadBuffer_fn(queue, buf_C, CL_TRUE, 0, nbytes,
                                h_C, 0, NULL, NULL);
    if (rc != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer warning: rc=%d; fingerprint unavailable\n", rc);
    }

    /* Fingerprint: sum(C) + frobenius_norm(C), same formula as FP32.
     * Each uint16_t element is converted to float via f16_to_f32 before
     * accumulation. The resulting fingerprint will differ from the FP32
     * value (33585655.900771) by the accumulated FP16 rounding error. */
    double sum_c   = 0.0;
    double frob_sq = 0.0;
    int total = size * size;
    for (int i = 0; i < total; i++) {
        float val = f16_to_f32(h_C[i]);
        sum_c   += (double)val;
        frob_sq += (double)val * (double)val;
    }
    double fingerprint = sum_c + sqrt(frob_sq);

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    clReleaseKernel_fn(kernel);
    clReleaseProgram_fn(program);
    clReleaseMemObject_fn(buf_A);
    clReleaseMemObject_fn(buf_B);
    clReleaseMemObject_fn(buf_C);
    clReleaseCommandQueue_fn(queue);
    clReleaseContext_fn(ctx);

    free(h_A);
    free(h_B);
    free(h_C);

    /* ── Print GPU_DISPATCH_RESULT protocol block ─────────────────────────── */
    printf("GPU_DISPATCH_RESULT\n");
    printf("status: OK\n");
    printf("cl_start_ns: %llu\n",  (unsigned long long)cl_start_ns);
    printf("cl_end_ns: %llu\n",    (unsigned long long)cl_end_ns);
    printf("duration_ns: %llu\n",  (unsigned long long)duration_ns);
    printf("workgroup_x: %d\n",    wg_x);
    printf("workgroup_y: %d\n",    wg_y);
    printf("workgroup_z: %d\n",    wg_z);
    printf("global_x: %zu\n",      global[0]);
    printf("global_y: %zu\n",      global[1]);
    printf("global_z: 1\n");
    printf("fingerprint: %.6f\n",  fingerprint);
    printf("GPU_DISPATCH_END\n");
    fflush(stdout);

    return 0;
}
