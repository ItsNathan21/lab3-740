/*
 * matmul_opencl_fp32_vec4.c — FP32 GPU vectorized candidate.
 *
 * Standalone arm64 ELF that runs a 512x512 FP32 matrix multiply on the
 * Adreno 830 GPU via OpenCL and prints the GPU_DISPATCH_RESULT
 * protocol block to stdout (parsed by host-side opencl_timing.py).
 *
 * Design: GPU analogue of kernels/matmul/cpu/neon_handtuned/matmul_neon.c.
 * Same splitmix64 GENERATOR as the CPU variants, but NOT the same inputs:
 * this fills [0.0, 1.0) from a fresh stream per matrix, while the CPU/DSP
 * kernels fill [-1.0, 1.0) from one continuous stream. Both are correct --
 * each matches an independent host reference -- but their fingerprints are
 * NOT comparable with each other. GPU and NPU agree; CPU and DSP agree.
 *
 * Kernel shape:
 *   - one work-item computes four adjacent C columns for one output row
 *   - B is loaded with vload4() and C is stored with vstore4()
 *   - A remains a scalar broadcast into a float4 accumulator
 *   - no __local memory and no barriers
 *
 * This is intended to test Qualcomm's documented 128-bit vector-load guidance
 * without paying the barrier/occupancy cost that made the tiled variant slower.
 *
 * OpenCL approach (confirmed from discovery 27):
 *   - dlopen /vendor/lib64/libOpenCL.so  (Qualcomm loader stub)
 *   - On-device online compiler          (CL_DEVICE_COMPILER_AVAILABLE=YES)
 *   - CL_QUEUE_PROFILING_ENABLE          (CL_PROFILING_TIMER_RESOLUTION=1000 ns)
 *   - HOST_UNIFIED_MEMORY = YES          (zero-copy possible; we use standard
 *                                         host buffers + CL_MEM_COPY_HOST_PTR)
 *
 * Argv:
 *   --size  N     matrix dimension (default 512; must be > 0)
 *   --seed  N     splitmix64 seed for A/B (default 42)
 *   --wg-x  N     workgroup X dimension (default 16)
 *   --wg-y  N     workgroup Y dimension (default 16)
 *   --wg-z  N     workgroup Z dimension (default 1; must be 1 for 2D NDRange)
 *
 * Stdout on success:
 *   GPU_DISPATCH_RESULT
 *   status: OK
 *   cl_start_ns: <uint64>
 *   cl_end_ns: <uint64>
 *   duration_ns: <uint64>
 *   workgroup_x: <uint>
 *   workgroup_y: <uint>
 *   workgroup_z: <uint>
 *   global_x: <uint>
 *   global_y: <uint>
 *   global_z: <uint>
 *   fingerprint: <float>
 *   GPU_DISPATCH_END
 *
 * Stdout on failure:
 *   GPU_DISPATCH_RESULT
 *   status: FAILED
 *   error: <message>
 *   GPU_DISPATCH_END
 *
 * Exit: 0 on success, 1 on failure.
 *
 * Build (inside control container from /workspace):
 *   bash kernels/matmul/gpu/opencl_fp32/build.sh
 */

#include <dlfcn.h>    /* dlopen, dlsym, dlerror */
#include <math.h>     /* sqrt, fabs — fingerprint computation */
#include <stdarg.h>   /* va_list, va_start, va_end — fail_dispatch_fmt */
#include <stddef.h>   /* size_t, NULL */
#include <stdint.h>   /* uint32_t, uint64_t, intptr_t */
#include <stdio.h>    /* printf, fprintf, snprintf, vsnprintf */
#include <stdlib.h>   /* malloc, free, exit, atoi, atoll */
#include <string.h>   /* strcmp, memset */


/* ── Section 1: Minimal OpenCL type definitions ──────────────────────────────
 * All types exactly mirror CL/cl.h and CL/cl_platform.h.
 * Defined inline so no OpenCL headers are needed at compile time — the NDK
 * sysroot we extracted does not ship CL headers (same rationale as
 * opencl_enum.c from Phase 4 Step 1).
 */

/* Opaque handles — all are void pointers on Adreno and every other real CL impl. */
typedef void *cl_platform_id;
typedef void *cl_device_id;
typedef void *cl_context;
typedef void *cl_command_queue;
typedef void *cl_mem;
typedef void *cl_program;
typedef void *cl_kernel;
typedef void *cl_event;

/* Integer types matching cl_platform.h */
typedef int32_t   cl_int;
typedef uint32_t  cl_uint;
typedef uint64_t  cl_ulong;
typedef uint64_t  cl_bitfield;
typedef uint32_t  cl_bool;
typedef uint64_t  cl_device_type;
typedef uintptr_t cl_device_info;
typedef uintptr_t cl_platform_info;
typedef uintptr_t cl_context_properties; /* used in clCreateContext props array */
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_queue_properties; /* used in clCreateCommandQueueWithProperties */

/* Return codes */
#define CL_SUCCESS              0
#define CL_BUILD_PROGRAM_FAILURE (-11)
#define CL_INVALID_VALUE        (-30)

/* Device type query token */
#define CL_DEVICE_TYPE_GPU   ((cl_device_type)(1 << 2))
#define CL_DEVICE_TYPE_ALL   ((cl_device_type)0xFFFFFFFF)

/* Context property key */
#define CL_CONTEXT_PLATFORM  ((cl_context_properties)0x1084)

/* Memory flags (bitmask) */
#define CL_MEM_READ_WRITE     ((cl_mem_flags)(1 << 0))
#define CL_MEM_COPY_HOST_PTR  ((cl_mem_flags)(1 << 5))

/* Command-queue property keys (for clCreateCommandQueueWithProperties) */
#define CL_QUEUE_PROPERTIES           ((cl_queue_properties)0x1093)
/* Command-queue property flag bits (same values in both old and new API) */
#define CL_QUEUE_PROFILING_ENABLE     ((cl_queue_properties)(1 << 1))

/* Profiling info tokens (passed to clGetEventProfilingInfo) */
#define CL_PROFILING_COMMAND_START    ((cl_uint)0x1282)
#define CL_PROFILING_COMMAND_END      ((cl_uint)0x1283)

/* Program build info token */
#define CL_PROGRAM_BUILD_LOG          ((cl_uint)0x1183)

/* CL_BOOL convenience */
#define CL_FALSE ((cl_bool)0)
#define CL_TRUE  ((cl_bool)1)


/* ── Section 2: Embedded OpenCL C kernel source ──────────────────────────────
 * Naive global-memory FP32 matmul. Each work item computes one C[row][col].
 * Bounds check supports global sizes rounded up to workgroup multiples.
 * mad(a, b, c) = a*b+c — maps to the GPU's FMA unit on Adreno 830.
 */
static const char *KERNEL_SOURCE =
    "__kernel void matmul_fp32_vec4(\n"
    "    __global const float* A,\n"
    "    __global const float* B,\n"
    "    __global       float* C,\n"
    "    int N)\n"
    "{\n"
    "    const int row = (int)get_global_id(0);\n"
    "    const int col4 = ((int)get_global_id(1)) << 2;\n"
    "    if (row >= N || col4 >= N) return;\n"
    "\n"
    "    float4 acc = (float4)(0.0f);\n"
    "    for (int k = 0; k < N; k++) {\n"
    "        const float a = A[row * N + k];\n"
    "        if (col4 + 3 < N) {\n"
    "            const float4 b = vload4(0, B + k * N + col4);\n"
    "            acc = mad((float4)(a), b, acc);\n"
    "        } else {\n"
    "            acc.s0 = mad(a, B[k * N + col4 + 0], acc.s0);\n"
    "            if (col4 + 1 < N) acc.s1 = mad(a, B[k * N + col4 + 1], acc.s1);\n"
    "            if (col4 + 2 < N) acc.s2 = mad(a, B[k * N + col4 + 2], acc.s2);\n"
    "        }\n"
    "    }\n"
    "\n"
    "    if (col4 + 3 < N) {\n"
    "        vstore4(acc, 0, C + row * N + col4);\n"
    "    } else {\n"
    "        C[row * N + col4 + 0] = acc.s0;\n"
    "        if (col4 + 1 < N) C[row * N + col4 + 1] = acc.s1;\n"
    "        if (col4 + 2 < N) C[row * N + col4 + 2] = acc.s2;\n"
    "    }\n"
    "}\n";


/* ── Section 3: Function pointer types and globals ───────────────────────────
 * One typedef + one global for every CL function called in main().
 * Populated by load_opencl() below.
 */

typedef cl_int (*pfn_clGetPlatformIDs)(
    cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
typedef cl_int (*pfn_clGetDeviceIDs)(
    cl_platform_id platform, cl_device_type device_type,
    cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);
typedef cl_context (*pfn_clCreateContext)(
    const cl_context_properties *properties,
    cl_uint num_devices, const cl_device_id *devices,
    void (*pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data, cl_int *errcode_ret);
typedef cl_command_queue (*pfn_clCreateCommandQueueWithProperties)(
    cl_context context, cl_device_id device,
    const cl_queue_properties *properties, cl_int *errcode_ret);
typedef cl_mem (*pfn_clCreateBuffer)(
    cl_context context, cl_mem_flags flags,
    size_t size, void *host_ptr, cl_int *errcode_ret);
typedef cl_program (*pfn_clCreateProgramWithSource)(
    cl_context context, cl_uint count,
    const char **strings, const size_t *lengths, cl_int *errcode_ret);
typedef cl_int (*pfn_clBuildProgram)(
    cl_program program, cl_uint num_devices,
    const cl_device_id *device_list, const char *options,
    void (*pfn_notify)(cl_program, void *), void *user_data);
typedef cl_int (*pfn_clGetProgramBuildInfo)(
    cl_program program, cl_device_id device,
    cl_uint param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret);
typedef cl_kernel (*pfn_clCreateKernel)(
    cl_program program, const char *kernel_name, cl_int *errcode_ret);
typedef cl_int (*pfn_clSetKernelArg)(
    cl_kernel kernel, cl_uint arg_index,
    size_t arg_size, const void *arg_value);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(
    cl_command_queue command_queue, cl_kernel kernel,
    cl_uint work_dim, const size_t *global_work_offset,
    const size_t *global_work_size, const size_t *local_work_size,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event);
typedef cl_int (*pfn_clEnqueueReadBuffer)(
    cl_command_queue command_queue, cl_mem buffer,
    cl_bool blocking_read, size_t offset, size_t size,
    void *ptr, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);
typedef cl_int (*pfn_clEnqueueWriteBuffer)(
    cl_command_queue command_queue, cl_mem buffer,
    cl_bool blocking_write, size_t offset, size_t size,
    const void *ptr, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);
typedef cl_int (*pfn_clFinish)(cl_command_queue command_queue);
typedef cl_int (*pfn_clGetEventProfilingInfo)(
    cl_event event, cl_uint param_name,
    size_t param_value_size, void *param_value,
    size_t *param_value_size_ret);
typedef cl_int (*pfn_clReleaseEvent)(cl_event event);
typedef cl_int (*pfn_clReleaseKernel)(cl_kernel kernel);
typedef cl_int (*pfn_clReleaseProgram)(cl_program program);
typedef cl_int (*pfn_clReleaseMemObject)(cl_mem memobj);
typedef cl_int (*pfn_clReleaseCommandQueue)(cl_command_queue command_queue);
typedef cl_int (*pfn_clReleaseContext)(cl_context context);

/* Global function pointer instances — set by load_opencl(). */
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
static pfn_clEnqueueWriteBuffer               clEnqueueWriteBuffer_fn               = NULL;
static pfn_clFinish                           clFinish_fn                           = NULL;
static pfn_clGetEventProfilingInfo            clGetEventProfilingInfo_fn            = NULL;
static pfn_clReleaseEvent                     clReleaseEvent_fn                     = NULL;
static pfn_clReleaseKernel                    clReleaseKernel_fn                    = NULL;
static pfn_clReleaseProgram                   clReleaseProgram_fn                   = NULL;
static pfn_clReleaseMemObject                 clReleaseMemObject_fn                 = NULL;
static pfn_clReleaseCommandQueue              clReleaseCommandQueue_fn              = NULL;
static pfn_clReleaseContext                   clReleaseContext_fn                   = NULL;


/* ── Section 4: Helpers ──────────────────────────────────────────────────────*/

/* fail_dispatch: print GPU_DISPATCH_RESULT FAILED block to stdout and exit(1).
 * All error paths call this so the host parser always gets a parseable block. */
static void fail_dispatch(const char *msg) {
    printf("GPU_DISPATCH_RESULT\n");
    printf("status: FAILED\n");
    printf("error: %s\n", msg);
    printf("GPU_DISPATCH_END\n");
    fflush(stdout);
    exit(1);
}

/* fail_dispatch_fmt: formatted variant of fail_dispatch (vsnprintf into buffer). */
static void fail_dispatch_fmt(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fail_dispatch(buf);
}

/* load_sym: dlsym wrapper that exits on failure — same pattern as opencl_enum.c. */
static void *load_sym(void *handle, const char *name) {
    void *sym = dlsym(handle, name);
    if (!sym) {
        char buf[256];
        snprintf(buf, sizeof(buf), "dlsym(%s) failed: %s", name, dlerror());
        fail_dispatch(buf);
    }
    return sym;
}

/* load_opencl: dlopen /vendor/lib64/libOpenCL.so and load all function pointers.
 * Same path and RTLD flags as opencl_enum.c — confirmed working in discovery 27. */
static void load_opencl(void) {
    const char *path = "/vendor/lib64/libOpenCL.so";
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        char buf[256];
        snprintf(buf, sizeof(buf), "dlopen(%s) failed: %s", path, dlerror());
        fail_dispatch(buf);
    }
    /* Load every function we need. load_sym calls fail_dispatch on any miss. */
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
    clEnqueueWriteBuffer_fn               = (pfn_clEnqueueWriteBuffer)               load_sym(h, "clEnqueueWriteBuffer");
    clFinish_fn                           = (pfn_clFinish)                           load_sym(h, "clFinish");
    clGetEventProfilingInfo_fn            = (pfn_clGetEventProfilingInfo)            load_sym(h, "clGetEventProfilingInfo");
    clReleaseEvent_fn                     = (pfn_clReleaseEvent)                     load_sym(h, "clReleaseEvent");
    clReleaseKernel_fn                    = (pfn_clReleaseKernel)                    load_sym(h, "clReleaseKernel");
    clReleaseProgram_fn                   = (pfn_clReleaseProgram)                   load_sym(h, "clReleaseProgram");
    clReleaseMemObject_fn                 = (pfn_clReleaseMemObject)                 load_sym(h, "clReleaseMemObject");
    clReleaseCommandQueue_fn              = (pfn_clReleaseCommandQueue)              load_sym(h, "clReleaseCommandQueue");
    clReleaseContext_fn                   = (pfn_clReleaseContext)                   load_sym(h, "clReleaseContext");
    /* Do NOT dlclose(h) — Adreno's atexit handlers may crash on close. */
}

/* splitmix64_next: advance PRNG state, return next uint64.
 * Identical algorithm to the CPU matmul variants (Session 21).
 * Same seed → same A and B → same C. Cross-checks against the GPU/NPU group. */
static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

/* fill_matrix: fill an N×N float matrix with splitmix64 values in [0.0, 1.0).
 * NOT the CPU mapping: CPU/DSP use 2*u-1 over [-1,1) from one stream. */
static void fill_matrix(float *mat, int N, uint64_t seed) {
    uint64_t state = seed;
    int total = N * N;
    for (int i = 0; i < total; i++) {
        uint64_t v = splitmix64_next(&state);
        /* Map uint64 → [0,1): divide by 2^64 using double precision. */
        mat[i] = (float)((double)v / 18446744073709551616.0);
    }
}

/* round_up: round n up to the nearest multiple of m (for global work size). */
static size_t round_up(int n, int m) {
    return (size_t)(((n + m - 1) / m) * m);
}


/* ── Section 5: main ─────────────────────────────────────────────────────────*/

int main(int argc, char **argv) {
    /* ── Parse arguments ─────────────────────────────────────────────────── */
    int    size = 512;   /* matrix dimension; default matches CPU variants */
    uint64_t seed = 42;  /* splitmix64 seed; matches CPU variants */
    int    wg_x  = 16;   /* workgroup X; 16×16 = 256 threads, fits 1024 max */
    int    wg_y  = 16;   /* workgroup Y */
    int    wg_z  = 1;    /* workgroup Z; must be 1 for a 2D NDRange */

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

    /* Basic sanity checks on arguments. */
    if (size <= 0 || wg_x <= 0 || wg_y <= 0 || wg_z != 1) {
        fail_dispatch("invalid arguments: size and wg-x/y must be >0; wg-z must be 1");
    }

    /* ── Load OpenCL library ─────────────────────────────────────────────── */
    load_opencl();

    /* ── Get platform and GPU device ─────────────────────────────────────── */
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

    /* ── Create context ──────────────────────────────────────────────────── */
    /* Explicitly specify the platform in the context properties so the driver
     * doesn't have to guess when there are multiple platforms. */
    cl_context_properties ctx_props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        (cl_context_properties)0
    };
    cl_context ctx = clCreateContext_fn(ctx_props, 1, &device, NULL, NULL, &rc);
    if (rc != CL_SUCCESS || !ctx) {
        fail_dispatch_fmt("clCreateContext failed: rc=%d", rc);
    }

    /* ── Create command queue with profiling enabled ─────────────────────── */
    /* CL_QUEUE_PROFILING_ENABLE is required for CL_PROFILING_COMMAND_START/END.
     * Using clCreateCommandQueueWithProperties (OpenCL 2.0+ API, per discovery 27
     * which confirmed OpenCL 3.0 full profile on the Adreno 830). */
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

    /* ── Allocate host matrices and fill A, B ────────────────────────────── */
    size_t nbytes = (size_t)size * (size_t)size * sizeof(float);

    float *h_A = (float *)malloc(nbytes);
    float *h_B = (float *)malloc(nbytes);
    float *h_C = (float *)malloc(nbytes);

    if (!h_A || !h_B || !h_C) {
        fail_dispatch("malloc failed for host matrices (out of memory)");
    }

    /* Fill A with seed 42, B with seed 43 — matches the CPU kernel convention.
     * splitmix64_next advances the state deterministically, so the same seed
     * always produces the same matrix. */
    fill_matrix(h_A, size, seed);
    fill_matrix(h_B, size, seed + 1);
    memset(h_C, 0, nbytes);  /* zero C before dispatch */

    /* ── Create OpenCL buffers (empty) ──────────────────────────────────────
     * Allocate empty buffers without CL_MEM_COPY_HOST_PTR. Data is uploaded
     * via explicit clEnqueueWriteBuffer below so the transfer time is
     * separately profiled (CL_PROFILING_COMMAND_START/END on the write event).
     * CL_MEM_COPY_HOST_PTR would do an implicit transfer at creation time with
     * no profiling event — transfer_ns would be unmeasurable. */
    cl_mem buf_A = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE,
                                     nbytes, NULL, &rc);
    if (rc != CL_SUCCESS || !buf_A) {
        fail_dispatch_fmt("clCreateBuffer(A) failed: rc=%d", rc);
    }

    cl_mem buf_B = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE,
                                     nbytes, NULL, &rc);
    if (rc != CL_SUCCESS || !buf_B) {
        clReleaseMemObject_fn(buf_A);
        fail_dispatch_fmt("clCreateBuffer(B) failed: rc=%d", rc);
    }

    cl_mem buf_C = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE,
                                     nbytes, NULL, &rc);
    if (rc != CL_SUCCESS || !buf_C) {
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        fail_dispatch_fmt("clCreateBuffer(C) failed: rc=%d", rc);
    }

    /* ── Enqueue CPU→GPU data transfer with profiling ────────────────────────
     * Non-blocking writes (CL_FALSE): enqueued but not yet complete.
     * clFinish below waits for both before the kernel is dispatched.
     * The in-order queue ensures the kernel sees the uploaded data, but
     * the explicit clFinish makes the transfer→compute boundary visible. */
    cl_event write_a_event = NULL, write_b_event = NULL;

    rc = clEnqueueWriteBuffer_fn(queue, buf_A, CL_FALSE, 0, nbytes,
                                 h_A, 0, NULL, &write_a_event);
    if (rc != CL_SUCCESS) {
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clEnqueueWriteBuffer(A) failed: rc=%d", rc);
    }

    rc = clEnqueueWriteBuffer_fn(queue, buf_B, CL_FALSE, 0, nbytes,
                                 h_B, 0, NULL, &write_b_event);
    if (rc != CL_SUCCESS) {
        clReleaseEvent_fn(write_a_event);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clEnqueueWriteBuffer(B) failed: rc=%d", rc);
    }

    /* Wait for both writes before enqueuing the kernel. */
    rc = clFinish_fn(queue);
    if (rc != CL_SUCCESS) {
        clReleaseEvent_fn(write_a_event);
        clReleaseEvent_fn(write_b_event);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clFinish (after writes) failed: rc=%d", rc);
    }

    /* Read CL_PROFILING timestamps for both write events. */
    cl_ulong write_a_start = 0, write_a_end = 0;
    cl_ulong write_b_start = 0, write_b_end = 0;
    rc  = clGetEventProfilingInfo_fn(write_a_event, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &write_a_start, NULL);
    rc |= clGetEventProfilingInfo_fn(write_a_event, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &write_a_end,   NULL);
    rc |= clGetEventProfilingInfo_fn(write_b_event, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &write_b_start, NULL);
    rc |= clGetEventProfilingInfo_fn(write_b_event, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &write_b_end,   NULL);
    clReleaseEvent_fn(write_a_event);
    clReleaseEvent_fn(write_b_event);
    if (rc != CL_SUCCESS) {
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clGetEventProfilingInfo (write events) failed: rc=%d", rc);
    }
    cl_ulong transfer_a_ns = write_a_end - write_a_start;
    cl_ulong transfer_b_ns = write_b_end - write_b_start;
    cl_ulong transfer_ns   = transfer_a_ns + transfer_b_ns;

    /* ── Build the OpenCL kernel program ─────────────────────────────────── */
    /* clCreateProgramWithSource takes the kernel source string. The online
     * compiler (CL_DEVICE_COMPILER_AVAILABLE=YES) compiles it on-device.
     * First compile on a cold cache may take 2–5 s; subsequent runs may be
     * faster if the driver caches the compiled binary. */
    cl_program program = clCreateProgramWithSource_fn(
        ctx, 1, &KERNEL_SOURCE, NULL, &rc);
    if (rc != CL_SUCCESS || !program) {
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clCreateProgramWithSource failed: rc=%d", rc);
    }

    /* Compile with -cl-fast-relaxed-math for FMA generation on Adreno 830.
     * -cl-std=CL3.0 is explicit; the driver defaults to a recent version
     * anyway, but being explicit avoids any "newest available" ambiguity. */
    rc = clBuildProgram_fn(program, 1, &device,
                           "-cl-fast-relaxed-math -cl-std=CL3.0",
                           NULL, NULL);
    if (rc != CL_SUCCESS) {
        /* Retrieve and surface the compiler error log for debugging.
         * CL_PROGRAM_BUILD_LOG contains the full compiler output. */
        char build_log[4096];
        memset(build_log, 0, sizeof(build_log));
        clGetProgramBuildInfo_fn(program, device, CL_PROGRAM_BUILD_LOG,
                                 sizeof(build_log) - 1, build_log, NULL);
        /* Truncate log at first newline for the single-line error: field,
         * but include first 200 chars so the problem is identifiable. */
        char err_buf[512];
        snprintf(err_buf, sizeof(err_buf),
                 "clBuildProgram failed: rc=%d | log: %.200s", rc, build_log);
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch(err_buf);
    }

    /* ── Create kernel and set arguments ─────────────────────────────────── */
    cl_kernel kernel = clCreateKernel_fn(program, "matmul_fp32_vec4", &rc);
    if (rc != CL_SUCCESS || !kernel) {
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clCreateKernel failed: rc=%d", rc);
    }

    /* Set the four kernel arguments: A, B, C, N */
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

    /* ── Enqueue NDRange kernel ───────────────────────────────────────────── */
    /* Global Y is in groups of four output columns. */
    int vec_cols = (size + 3) / 4;
    size_t global[2] = { round_up(size, wg_x), round_up(vec_cols, wg_y) };
    size_t local[2]  = { (size_t)wg_x, (size_t)wg_y };

    /* The kernel event captures CL_PROFILING_COMMAND_START and _END. */
    cl_event kernel_event = NULL;

    rc = clEnqueueNDRangeKernel_fn(
        queue, kernel,
        2,       /* work_dim: 2D NDRange (row × col) */
        NULL,    /* global_work_offset: no offset */
        global,  /* global_work_size: {round_up(N,wg_x), round_up(N,wg_y)} */
        local,   /* local_work_size: {wg_x, wg_y} */
        0, NULL, /* no wait-list */
        &kernel_event);

    if (rc != CL_SUCCESS) {
        clReleaseKernel_fn(kernel);
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clEnqueueNDRangeKernel failed: rc=%d", rc);
    }

    /* Wait for the kernel to finish before reading back C and profiling data. */
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

    /* ── Read profiling timestamps ────────────────────────────────────────── */
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
    /* Blocking read: copies the device buffer back to h_C on the host.
     * Used for validation — fingerprint = sum(C) + frobenius_norm(C),
     * matching the CPU kernel fingerprint convention (Session 21). */
    rc = clEnqueueReadBuffer_fn(queue, buf_C, CL_TRUE, 0, nbytes,
                                h_C, 0, NULL, NULL);
    if (rc != CL_SUCCESS) {
        /* Non-fatal — dispatch succeeded; just skip fingerprint. */
        fprintf(stderr, "clEnqueueReadBuffer warning: rc=%d; fingerprint unavailable\n", rc);
    }

    /* Compute fingerprint: sum(C) + frobenius_norm(C). */
    double sum_c   = 0.0;
    double frob_sq = 0.0;
    int total = size * size;
    for (int i = 0; i < total; i++) {
        sum_c   += (double)h_C[i];
        frob_sq += (double)h_C[i] * (double)h_C[i];
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
    /* Protocol v1 as defined in src/host/orchestrator/backends/gpu/opencl_timing.py.
     * Fields cl_start_ns / cl_end_ns / duration_ns are required.
     * workgroup_* and global_* are optional but provided for full characterization. */
    printf("GPU_DISPATCH_RESULT\n");
    printf("status: OK\n");
    printf("cl_start_ns: %llu\n",   (unsigned long long)cl_start_ns);
    printf("cl_end_ns: %llu\n",     (unsigned long long)cl_end_ns);
    printf("duration_ns: %llu\n",   (unsigned long long)duration_ns);
    printf("transfer_ns: %llu\n",   (unsigned long long)transfer_ns);
    printf("workgroup_x: %d\n",     wg_x);
    printf("workgroup_y: %d\n",     wg_y);
    printf("workgroup_z: %d\n",     wg_z);
    printf("global_x: %zu\n",       global[0]);
    printf("global_y: %zu\n",       global[1]);
    printf("global_z: 1\n");
    printf("fingerprint: %.6f\n",   fingerprint);
    printf("GPU_DISPATCH_END\n");
    fflush(stdout);

    return 0;
}
