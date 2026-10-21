/*
 * matmul_opencl_fp16_4x16u4.c — FP16 GPU register-blocked matmul candidate.
 *
 * True half-precision analogue of opencl_fp32_4x16u4: each work-item computes a
 * 4-row x 16-column C micro-tile, but A, B, C and the accumulators are all
 * `half`. B is loaded as half16 (one vector per K step) and reused across four
 * rows; A is a scalar broadcast; the K loop is unrolled by 4. No __local
 * memory, no barriers.
 *
 * Rationale: sol_microbench `alu_half` measured the Adreno 830 FP16 ALU roof at
 * 6290 GFLOP/s, exactly 2x the 3145 GFLOP/s FP32 roof (packed 2:1). FP16 also
 * halves accumulator register bytes, and register/occupancy pressure is the
 * measured FP32 GEMM limiter (3x16 > 4x16 > 6x16). So an FP16 register-tiled
 * GEMM is the highest-value throughput lever beyond the 802 GFLOP/s FP32 best.
 *
 * Numeric contract: FP16 accumulation over 512 K terms is lossy, so this kernel
 * does NOT match the FP32 CPU fingerprint exactly. It is validated by relative
 * error against the FP32 reference fingerprint (host-side gate in the benchmark
 * harness), not by exact match. The fingerprint here is computed the same way
 * (sum + frobenius norm) over the float-promoted FP16 result for comparability.
 *
 * Argv: --size N (must be 512) --seed N --wg-x N --wg-y N --wg-z 1 --inner-reps N
 *
 * Stdout: GPU_DISPATCH_RESULT protocol block (same as the FP32 variants).
 *
 * Build: bash kernels/matmul/gpu/opencl_fp16_4x16u4/build.sh
 */

#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Minimal OpenCL type definitions (mirror CL/cl.h) ──────────────────────── */
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
typedef uintptr_t cl_context_properties;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_queue_properties;

#define CL_SUCCESS              0
#define CL_DEVICE_TYPE_GPU   ((cl_device_type)(1 << 2))
#define CL_CONTEXT_PLATFORM  ((cl_context_properties)0x1084)
#define CL_MEM_READ_WRITE     ((cl_mem_flags)(1 << 0))
#define CL_QUEUE_PROPERTIES           ((cl_queue_properties)0x1093)
#define CL_QUEUE_PROFILING_ENABLE     ((cl_queue_properties)(1 << 1))
#define CL_PROFILING_COMMAND_START    ((cl_uint)0x1282)
#define CL_PROFILING_COMMAND_END      ((cl_uint)0x1283)
#define CL_PROGRAM_BUILD_LOG          ((cl_uint)0x1183)
#define CL_TRUE  ((cl_bool)1)
#define CL_FALSE ((cl_bool)0)

/* ── Embedded OpenCL C kernel: half16 register-tiled 4x16 matmul ───────────── */
static const char *KERNEL_SOURCE =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void matmul_fp16_4x16u4(\n"
    "    __global const half* restrict A,\n"
    "    __global const half* restrict B,\n"
    "    __global       half* restrict C,\n"
    "    int N,\n"
    "    int inner_reps)\n"
    "{\n"
    "    const int row0 = ((int)get_global_id(0)) << 2;\n"
    "    const int row1 = row0 + 1;\n"
    "    const int row2 = row0 + 2;\n"
    "    const int row3 = row0 + 3;\n"
    "    const int col16 = ((int)get_global_id(1)) << 4;\n"
    "    if (row0 >= 512 || col16 >= 512) return;\n"
    "\n"
    "    half16 acc = (half16)((half)0.0h);\n"
    "    half16 acc1 = (half16)((half)0.0h);\n"
    "    half16 acc2 = (half16)((half)0.0h);\n"
    "    half16 acc3 = (half16)((half)0.0h);\n"
    "    const half inv_reps = (half)(1.0f / (float)inner_reps);\n"
    "    for (int rep = 0; rep < inner_reps; rep++) {\n"
    "        #pragma unroll 2\n"
    "        for (int k = 0; k < 512; k += 4) {\n"
    "            const int a0base = (row0 << 9) + k;\n"
    "            const int a1base = (row1 << 9) + k;\n"
    "            const int a2base = (row2 << 9) + k;\n"
    "            const int a3base = (row3 << 9) + k;\n"
    "            const half a0 = A[a0base + 0];\n"
    "            const half a1 = A[a0base + 1];\n"
    "            const half a2 = A[a0base + 2];\n"
    "            const half a3 = A[a0base + 3];\n"
    "            const half r0 = A[a1base + 0];\n"
    "            const half r1 = A[a1base + 1];\n"
    "            const half r2 = A[a1base + 2];\n"
    "            const half r3 = A[a1base + 3];\n"
    "            const half s0 = A[a2base + 0];\n"
    "            const half s1 = A[a2base + 1];\n"
    "            const half s2 = A[a2base + 2];\n"
    "            const half s3 = A[a2base + 3];\n"
    "            const half t0 = A[a3base + 0];\n"
    "            const half t1 = A[a3base + 1];\n"
    "            const half t2 = A[a3base + 2];\n"
    "            const half t3 = A[a3base + 3];\n"
    "            const half16 b0 = vload16(0, B + ((k + 0) << 9) + col16);\n"
    "            const half16 b1 = vload16(0, B + ((k + 1) << 9) + col16);\n"
    "            const half16 b2 = vload16(0, B + ((k + 2) << 9) + col16);\n"
    "            const half16 b3 = vload16(0, B + ((k + 3) << 9) + col16);\n"
    "            acc = mad((half16)(a0), b0, acc);\n"
    "            acc = mad((half16)(a1), b1, acc);\n"
    "            acc = mad((half16)(a2), b2, acc);\n"
    "            acc = mad((half16)(a3), b3, acc);\n"
    "            acc1 = mad((half16)(r0), b0, acc1);\n"
    "            acc1 = mad((half16)(r1), b1, acc1);\n"
    "            acc1 = mad((half16)(r2), b2, acc1);\n"
    "            acc1 = mad((half16)(r3), b3, acc1);\n"
    "            acc2 = mad((half16)(s0), b0, acc2);\n"
    "            acc2 = mad((half16)(s1), b1, acc2);\n"
    "            acc2 = mad((half16)(s2), b2, acc2);\n"
    "            acc2 = mad((half16)(s3), b3, acc2);\n"
    "            acc3 = mad((half16)(t0), b0, acc3);\n"
    "            acc3 = mad((half16)(t1), b1, acc3);\n"
    "            acc3 = mad((half16)(t2), b2, acc3);\n"
    "            acc3 = mad((half16)(t3), b3, acc3);\n"
    "        }\n"
    "    }\n"
    "    vstore16(acc * (half16)(inv_reps), 0, C + (row0 << 9) + col16);\n"
    "    vstore16(acc1 * (half16)(inv_reps), 0, C + (row1 << 9) + col16);\n"
    "    vstore16(acc2 * (half16)(inv_reps), 0, C + (row2 << 9) + col16);\n"
    "    vstore16(acc3 * (half16)(inv_reps), 0, C + (row3 << 9) + col16);\n"
    "}\n";

/* ── Function pointer types and globals ────────────────────────────────────── */
typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_context (*pfn_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                          void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
typedef cl_command_queue (*pfn_clCreateCommandQueueWithProperties)(cl_context, cl_device_id,
                                                                   const cl_queue_properties *, cl_int *);
typedef cl_mem (*pfn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_program (*pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_int (*pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *,
                                     void (*)(cl_program, void *), void *);
typedef cl_int (*pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
typedef cl_kernel (*pfn_clCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_int (*pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
                                             const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *,
                                          cl_uint, const cl_event *, cl_event *);
typedef cl_int (*pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *,
                                           cl_uint, const cl_event *, cl_event *);
typedef cl_int (*pfn_clFinish)(cl_command_queue);
typedef cl_int (*pfn_clGetEventProfilingInfo)(cl_event, cl_uint, size_t, void *, size_t *);
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
static pfn_clEnqueueWriteBuffer               clEnqueueWriteBuffer_fn               = NULL;
static pfn_clFinish                           clFinish_fn                           = NULL;
static pfn_clGetEventProfilingInfo            clGetEventProfilingInfo_fn            = NULL;
static pfn_clReleaseEvent                     clReleaseEvent_fn                     = NULL;
static pfn_clReleaseKernel                    clReleaseKernel_fn                    = NULL;
static pfn_clReleaseProgram                   clReleaseProgram_fn                   = NULL;
static pfn_clReleaseMemObject                 clReleaseMemObject_fn                 = NULL;
static pfn_clReleaseCommandQueue              clReleaseCommandQueue_fn              = NULL;
static pfn_clReleaseContext                   clReleaseContext_fn                   = NULL;

/* ── Helpers ───────────────────────────────────────────────────────────────── */
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
    clEnqueueWriteBuffer_fn               = (pfn_clEnqueueWriteBuffer)               load_sym(h, "clEnqueueWriteBuffer");
    clFinish_fn                           = (pfn_clFinish)                           load_sym(h, "clFinish");
    clGetEventProfilingInfo_fn            = (pfn_clGetEventProfilingInfo)            load_sym(h, "clGetEventProfilingInfo");
    clReleaseEvent_fn                     = (pfn_clReleaseEvent)                     load_sym(h, "clReleaseEvent");
    clReleaseKernel_fn                    = (pfn_clReleaseKernel)                    load_sym(h, "clReleaseKernel");
    clReleaseProgram_fn                   = (pfn_clReleaseProgram)                   load_sym(h, "clReleaseProgram");
    clReleaseMemObject_fn                 = (pfn_clReleaseMemObject)                 load_sym(h, "clReleaseMemObject");
    clReleaseCommandQueue_fn              = (pfn_clReleaseCommandQueue)              load_sym(h, "clReleaseCommandQueue");
    clReleaseContext_fn                   = (pfn_clReleaseContext)                   load_sym(h, "clReleaseContext");
}

/* splitmix64 + fill: identical mapping to the FP32 variants, but stored as
 * _Float16 so the device buffers are half-precision. */
static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void fill_matrix_half(_Float16 *mat, int N, uint64_t seed) {
    uint64_t state = seed;
    int total = N * N;
    for (int i = 0; i < total; i++) {
        uint64_t v = splitmix64_next(&state);
        mat[i] = (_Float16)((double)v / 18446744073709551616.0);
    }
}

static size_t round_up(int n, int m) {
    return (size_t)(((n + m - 1) / m) * m);
}

/* ── main ──────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    int    size = 512;
    uint64_t seed = 42;
    int    wg_x  = 4;
    int    wg_y  = 32;
    int    wg_z  = 1;
    int    inner_reps = 1;

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
        } else if (strcmp(argv[i], "--inner-reps") == 0 && i + 1 < argc) {
            inner_reps = atoi(argv[++i]);
        }
    }

    if (size <= 0 || wg_x <= 0 || wg_y <= 0 || wg_z != 1 || inner_reps <= 0) {
        fail_dispatch("invalid arguments: size, wg-x/y, and inner-reps must be >0; wg-z must be 1");
    }
    if (size != 512) {
        fail_dispatch("opencl_fp16_4x16u4 is specialized for --size 512");
    }

    load_opencl();

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
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, (cl_context_properties)0
    };
    cl_context ctx = clCreateContext_fn(ctx_props, 1, &device, NULL, NULL, &rc);
    if (rc != CL_SUCCESS || !ctx) {
        fail_dispatch_fmt("clCreateContext failed: rc=%d", rc);
    }

    cl_queue_properties q_props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, (cl_queue_properties)0
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties_fn(ctx, device, q_props, &rc);
    if (rc != CL_SUCCESS || !queue) {
        fail_dispatch_fmt("clCreateCommandQueueWithProperties failed: rc=%d", rc);
    }

    size_t nbytes = (size_t)size * (size_t)size * sizeof(_Float16);

    _Float16 *h_A = (_Float16 *)malloc(nbytes);
    _Float16 *h_B = (_Float16 *)malloc(nbytes);
    _Float16 *h_C = (_Float16 *)malloc(nbytes);
    if (!h_A || !h_B || !h_C) {
        fail_dispatch("malloc failed for host matrices (out of memory)");
    }
    fill_matrix_half(h_A, size, seed);
    fill_matrix_half(h_B, size, seed + 1);
    memset(h_C, 0, nbytes);

    cl_mem buf_A = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, nbytes, NULL, &rc);
    if (rc != CL_SUCCESS || !buf_A) fail_dispatch_fmt("clCreateBuffer(A) failed: rc=%d", rc);
    cl_mem buf_B = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, nbytes, NULL, &rc);
    if (rc != CL_SUCCESS || !buf_B) fail_dispatch_fmt("clCreateBuffer(B) failed: rc=%d", rc);
    cl_mem buf_C = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, nbytes, NULL, &rc);
    if (rc != CL_SUCCESS || !buf_C) fail_dispatch_fmt("clCreateBuffer(C) failed: rc=%d", rc);

    cl_event write_a_event = NULL, write_b_event = NULL;
    rc = clEnqueueWriteBuffer_fn(queue, buf_A, CL_FALSE, 0, nbytes, h_A, 0, NULL, &write_a_event);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clEnqueueWriteBuffer(A) failed: rc=%d", rc);
    rc = clEnqueueWriteBuffer_fn(queue, buf_B, CL_FALSE, 0, nbytes, h_B, 0, NULL, &write_b_event);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clEnqueueWriteBuffer(B) failed: rc=%d", rc);
    rc = clFinish_fn(queue);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clFinish (after writes) failed: rc=%d", rc);

    cl_ulong write_a_start = 0, write_a_end = 0, write_b_start = 0, write_b_end = 0;
    rc  = clGetEventProfilingInfo_fn(write_a_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write_a_start, NULL);
    rc |= clGetEventProfilingInfo_fn(write_a_event, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &write_a_end,   NULL);
    rc |= clGetEventProfilingInfo_fn(write_b_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write_b_start, NULL);
    rc |= clGetEventProfilingInfo_fn(write_b_event, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &write_b_end,   NULL);
    clReleaseEvent_fn(write_a_event);
    clReleaseEvent_fn(write_b_event);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clGetEventProfilingInfo (write events) failed: rc=%d", rc);
    cl_ulong transfer_ns = (write_a_end - write_a_start) + (write_b_end - write_b_start);

    cl_program program = clCreateProgramWithSource_fn(ctx, 1, &KERNEL_SOURCE, NULL, &rc);
    if (rc != CL_SUCCESS || !program) fail_dispatch_fmt("clCreateProgramWithSource failed: rc=%d", rc);

    rc = clBuildProgram_fn(program, 1, &device, "-cl-fast-relaxed-math -cl-std=CL3.0", NULL, NULL);
    if (rc != CL_SUCCESS) {
        char build_log[4096];
        memset(build_log, 0, sizeof(build_log));
        clGetProgramBuildInfo_fn(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log) - 1, build_log, NULL);
        fail_dispatch_fmt("clBuildProgram failed: rc=%d | log: %.200s", rc, build_log);
    }

    cl_kernel kernel = clCreateKernel_fn(program, "matmul_fp16_4x16u4", &rc);
    if (rc != CL_SUCCESS || !kernel) fail_dispatch_fmt("clCreateKernel failed: rc=%d", rc);

    rc  = clSetKernelArg_fn(kernel, 0, sizeof(cl_mem), &buf_A);
    rc |= clSetKernelArg_fn(kernel, 1, sizeof(cl_mem), &buf_B);
    rc |= clSetKernelArg_fn(kernel, 2, sizeof(cl_mem), &buf_C);
    rc |= clSetKernelArg_fn(kernel, 3, sizeof(cl_int), &size);
    rc |= clSetKernelArg_fn(kernel, 4, sizeof(cl_int), &inner_reps);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clSetKernelArg failed: rc=%d", rc);

    int row_blocks = (size + 3) / 4;
    int vec_cols = (size + 15) / 16;
    size_t global[2] = { round_up(row_blocks, wg_x), round_up(vec_cols, wg_y) };
    size_t local[2]  = { (size_t)wg_x, (size_t)wg_y };

    cl_event kernel_event = NULL;
    rc = clEnqueueNDRangeKernel_fn(queue, kernel, 2, NULL, global, local, 0, NULL, &kernel_event);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clEnqueueNDRangeKernel failed: rc=%d", rc);

    rc = clFinish_fn(queue);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clFinish failed: rc=%d", rc);

    cl_ulong cl_start_ns = 0, cl_end_ns = 0;
    rc  = clGetEventProfilingInfo_fn(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &cl_start_ns, NULL);
    rc |= clGetEventProfilingInfo_fn(kernel_event, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &cl_end_ns, NULL);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clGetEventProfilingInfo failed: rc=%d", rc);
    cl_ulong duration_ns = cl_end_ns - cl_start_ns;
    clReleaseEvent_fn(kernel_event);

    rc = clEnqueueReadBuffer_fn(queue, buf_C, CL_TRUE, 0, nbytes, h_C, 0, NULL, NULL);
    if (rc != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer warning: rc=%d; fingerprint unavailable\n", rc);
    }

    /* Fingerprint over the float-promoted FP16 result, same formula as FP32. */
    double sum_c = 0.0, frob_sq = 0.0;
    int total = size * size;
    for (int i = 0; i < total; i++) {
        double v = (double)(float)h_C[i];
        sum_c   += v;
        frob_sq += v * v;
    }
    double fingerprint = sum_c + sqrt(frob_sq);

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

    printf("GPU_DISPATCH_RESULT\n");
    printf("status: OK\n");
    printf("cl_start_ns: %llu\n",   (unsigned long long)cl_start_ns);
    printf("cl_end_ns: %llu\n",     (unsigned long long)cl_end_ns);
    printf("duration_ns: %llu\n",   (unsigned long long)duration_ns);
    printf("transfer_ns: %llu\n",   (unsigned long long)transfer_ns);
    printf("inner_reps: %d\n",      inner_reps);
    printf("effective_flops: %.0f\n",
           2.0 * (double)size * (double)size * (double)size * (double)inner_reps);
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
