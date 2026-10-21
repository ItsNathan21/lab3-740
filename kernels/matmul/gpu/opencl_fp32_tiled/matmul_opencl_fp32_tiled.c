/*
 * matmul_opencl_fp32_tiled.c — Phase 4 Step 8 GPU tiled matmul.
 *
 * Standalone arm64 ELF that runs a TILED FP32 512×512 matrix multiply
 * on the Adreno 830 GPU via OpenCL and prints the GPU_DISPATCH_RESULT
 * protocol block to stdout (parsed by host-side opencl_timing.py).
 *
 * How tiling improves on the naive kernel (matmul_opencl_fp32.c):
 *   The naive kernel (Phase 4 Step 6) has each work item read its entire
 *   row of A and column of B from global memory (DRAM) — N reads per work
 *   item, all potentially cache-miss. For N=512, that is 512 floats × 2
 *   matrices × 262144 work items = 268 million global memory reads.
 *
 *   The tiled kernel loads TILE×TILE blocks ("tiles") of A and B into
 *   __local memory (on-chip SRAM, 32 KiB per workgroup on Adreno 830 per
 *   discovery 27) before the inner multiply loop. Once in __local memory,
 *   each float is reused TILE times instead of once. Global memory traffic
 *   drops by ~TILE (16×), from O(N³) to O(N³/TILE) global reads.
 *
 * Tile size: TILE = 16.
 *   Matches the default workgroup size (16×16 = 256 threads).
 *   __local memory used: 2 × 16×16 × 4 bytes = 2 KiB per workgroup,
 *   well within the 32 KiB limit.
 *
 * CONSTRAINT: wg_x and wg_y MUST both equal TILE (16). Passing a different
 *   workgroup size via --wg-x/--wg-y will fail with an error. The tile size
 *   is embedded as a #define in the kernel source; changing it requires
 *   recompiling the binary (or a future --tile argument + dynamic source).
 *
 * Expected speedup over naive: 3–10× for 512×512 FP32 on Adreno 830,
 *   depending on the ratio of on-chip bandwidth to global memory bandwidth.
 *   Exact numbers in discovery 32.
 *
 * OpenCL approach: identical to matmul_opencl_fp32.c (Phase 4 Step 6).
 *   - dlopen /vendor/lib64/libOpenCL.so
 *   - On-device online compiler (CL_DEVICE_COMPILER_AVAILABLE=YES)
 *   - CL_QUEUE_PROFILING_ENABLE for duration_ns
 *   - Same GPU_DISPATCH_RESULT protocol v1
 *   - Same splitmix64 seeding (seed=42 for A, seed=43 for B)
 *   - Same fingerprint = sum(C) + frobenius_norm(C) for cross-validation
 *
 * Argv (identical to naive binary):
 *   --size  N   matrix dimension (default 512; must be > 0)
 *   --seed  N   splitmix64 seed for A/B (default 42)
 *   --wg-x  N   workgroup X (default 16; MUST equal TILE=16)
 *   --wg-y  N   workgroup Y (default 16; MUST equal TILE=16)
 *   --wg-z  N   workgroup Z (default 1; must be 1)
 *
 * Build (inside control container from /workspace):
 *   bash kernels/matmul/gpu/opencl_fp32_tiled/build.sh
 */

#include <dlfcn.h>    /* dlopen, dlsym, dlerror */
#include <math.h>     /* sqrt — fingerprint computation */
#include <stdarg.h>   /* va_list, va_start, va_end — fail_dispatch_fmt */
#include <stddef.h>   /* size_t, NULL */
#include <stdint.h>   /* uint32_t, uint64_t, intptr_t */
#include <stdio.h>    /* printf, fprintf, snprintf, vsnprintf */
#include <stdlib.h>   /* malloc, free, exit, atoi, atoll */
#include <string.h>   /* strcmp, memset */


/* ── Section 1: Minimal OpenCL type definitions ──────────────────────────────
 * Identical to matmul_opencl_fp32.c — all types mirror CL/cl.h exactly.
 * Defined inline so no OpenCL headers are needed at compile time.
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

#define CL_DEVICE_TYPE_GPU   ((cl_device_type)(1 << 2))
#define CL_DEVICE_TYPE_ALL   ((cl_device_type)0xFFFFFFFF)

#define CL_CONTEXT_PLATFORM  ((cl_context_properties)0x1084)

#define CL_MEM_READ_WRITE     ((cl_mem_flags)(1 << 0))
#define CL_MEM_COPY_HOST_PTR  ((cl_mem_flags)(1 << 5))

#define CL_QUEUE_PROPERTIES           ((cl_queue_properties)0x1093)
#define CL_QUEUE_PROFILING_ENABLE     ((cl_queue_properties)(1 << 1))

#define CL_PROFILING_COMMAND_START    ((cl_uint)0x1282)
#define CL_PROFILING_COMMAND_END      ((cl_uint)0x1283)

#define CL_PROGRAM_BUILD_LOG          ((cl_uint)0x1183)

#define CL_TRUE  ((cl_bool)1)


/* ── Section 2: Embedded OpenCL C kernel source (TILED VERSION) ──────────────
 *
 * Key differences from the naive kernel:
 *
 *   1. __local tile buffers tileA and tileB hold a TILE×TILE sub-block of A
 *      and B respectively. Each float loaded from global memory is then reused
 *      TILE times from __local memory instead of re-fetching from DRAM.
 *
 *   2. Outer loop: "for t in 0..n_tiles" steps through TILE-wide strips of
 *      the K (shared) dimension.
 *
 *   3. Two barriers per tile iteration:
 *      - After loading (CLK_LOCAL_MEM_FENCE): ensures all work items have
 *        finished writing to tileA/tileB before any work item reads them.
 *      - After the inner multiply loop (CLK_LOCAL_MEM_FENCE): ensures all
 *        work items have finished reading tileA/tileB before the next
 *        iteration overwrites them.
 *
 *   4. Boundary checks in the tile load guards against N not being a multiple
 *      of TILE. For N=512 and TILE=16 (512 = 32 × 16) there are no boundary
 *      tiles — all loads are in-bounds. The checks cost nothing in that case.
 *
 *   5. TILE = 16 is defined as a #define inside the kernel source string so
 *      the Adreno online compiler sees it as a compile-time constant and can
 *      unroll the inner k-loop and optimize tile indexing.
 *
 *   Memory layout note: A and B are both row-major (standard C layout).
 *      A[row][col] = A[row * N + col].
 *      Tile loading:
 *        tileA[lrow][lcol] ← A[global_row][tile_k_start + lcol]  (row of A)
 *        tileB[lrow][lcol] ← B[tile_k_start + lrow][global_col]  (col of B)
 *      Loading tileB accesses B with stride N (column-major access of row-major
 *      B) — this is the pattern that the naive kernel pays per-element; tiling
 *      amortizes it by loading once per tile into on-chip memory.
 */
static const char *KERNEL_SOURCE =
    /* TILE must match the workgroup size (both wg-x and wg-y). */
    "#define TILE 16\n"
    "__kernel void matmul_tiled(\n"
    "    __global const float* A,\n"
    "    __global const float* B,\n"
    "    __global       float* C,\n"
    "    int N)\n"
    "{\n"
    "    /* Global indices: which element of C this work item computes. */\n"
    "    const int row  = (int)get_global_id(0);\n"
    "    const int col  = (int)get_global_id(1);\n"
    "    /* Local indices: position within the TILE×TILE workgroup. */\n"
    "    const int lrow = (int)get_local_id(0);\n"
    "    const int lcol = (int)get_local_id(1);\n"
    "\n"
    "    /* On-chip tile buffers in __local (SRAM) memory.\n"
    "     * 2 × TILE×TILE × 4 bytes = 2 KiB per workgroup.\n"
    "     * Adreno 830 provides 32 KiB __local per workgroup (discovery 27). */\n"
    "    __local float tileA[TILE][TILE];\n"
    "    __local float tileB[TILE][TILE];\n"
    "\n"
    "    float acc = 0.0f;\n"
    "\n"
    "    /* Number of TILE-wide strips to iterate over the K dimension. */\n"
    "    const int n_tiles = (N + TILE - 1) / TILE;\n"
    "\n"
    "    for (int t = 0; t < n_tiles; t++) {\n"
    "        /* Each work item loads one element into each tile.\n"
    "         * Boundary check handles N not divisible by TILE. */\n"
    "        const int a_col = t * TILE + lcol;\n"
    "        const int b_row = t * TILE + lrow;\n"
    "        tileA[lrow][lcol] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;\n"
    "        tileB[lrow][lcol] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;\n"
    "\n"
    "        /* Synchronize: all work items must finish loading before any reads. */\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "\n"
    "        /* Inner product over this tile — all reads hit __local SRAM. */\n"
    "        for (int k = 0; k < TILE; k++) {\n"
    "            acc = mad(tileA[lrow][k], tileB[k][lcol], acc);\n"
    "        }\n"
    "\n"
    "        /* Synchronize: all work items must finish reading before next load. */\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "\n"
    "    /* Write result for in-bounds work items only. */\n"
    "    if (row < N && col < N) {\n"
    "        C[row * N + col] = acc;\n"
    "    }\n"
    "}\n";


/* ── Section 3: Function pointer types and globals ───────────────────────────
 * Identical to matmul_opencl_fp32.c — one typedef + one global per CL call.
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


/* ── Section 4: Helpers ──────────────────────────────────────────────────────
 * Identical to matmul_opencl_fp32.c.
 */

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
    clFinish_fn                           = (pfn_clFinish)                           load_sym(h, "clFinish");
    clGetEventProfilingInfo_fn            = (pfn_clGetEventProfilingInfo)            load_sym(h, "clGetEventProfilingInfo");
    clReleaseEvent_fn                     = (pfn_clReleaseEvent)                     load_sym(h, "clReleaseEvent");
    clReleaseKernel_fn                    = (pfn_clReleaseKernel)                    load_sym(h, "clReleaseKernel");
    clReleaseProgram_fn                   = (pfn_clReleaseProgram)                   load_sym(h, "clReleaseProgram");
    clReleaseMemObject_fn                 = (pfn_clReleaseMemObject)                 load_sym(h, "clReleaseMemObject");
    clReleaseCommandQueue_fn              = (pfn_clReleaseCommandQueue)              load_sym(h, "clReleaseCommandQueue");
    clReleaseContext_fn                   = (pfn_clReleaseContext)                   load_sym(h, "clReleaseContext");
}

/* splitmix64 and fill_matrix: identical to matmul_opencl_fp32.c.
 * Same seed → same A, B → same C → same fingerprint for cross-validation. */
static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void fill_matrix(float *mat, int N, uint64_t seed) {
    uint64_t state = seed;
    int total = N * N;
    for (int i = 0; i < total; i++) {
        uint64_t v = splitmix64_next(&state);
        mat[i] = (float)((double)v / 18446744073709551616.0);
    }
}

static size_t round_up(int n, int m) {
    return (size_t)(((n + m - 1) / m) * m);
}


/* ── Section 5: main ─────────────────────────────────────────────────────────*/

/* The compile-time TILE constant embedded in KERNEL_SOURCE above. Used here
 * to validate that the user's --wg-x and --wg-y match it. */
#define TILE_SIZE 16

int main(int argc, char **argv) {
    /* ── Parse arguments ─────────────────────────────────────────────────── */
    int      size = 512;
    uint64_t seed = 42;
    int      wg_x = TILE_SIZE;
    int      wg_y = TILE_SIZE;
    int      wg_z = 1;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--wg-x") == 0 && i + 1 < argc) wg_x = atoi(argv[++i]);
        else if (strcmp(argv[i], "--wg-y") == 0 && i + 1 < argc) wg_y = atoi(argv[++i]);
        else if (strcmp(argv[i], "--wg-z") == 0 && i + 1 < argc) wg_z = atoi(argv[++i]);
    }

    if (size <= 0 || wg_z != 1) {
        fail_dispatch("invalid arguments: size must be >0; wg-z must be 1");
    }

    /* Tiled kernel constraint: workgroup dimensions must equal TILE_SIZE.
     * The __local tile arrays are declared as [TILE][TILE] = [16][16] in the
     * kernel source. A different workgroup size would cause each work item to
     * load/read the wrong tile elements, producing incorrect results silently.
     * Reject early with a clear message rather than producing wrong output. */
    if (wg_x != TILE_SIZE || wg_y != TILE_SIZE) {
        fail_dispatch_fmt(
            "tiled kernel requires wg-x=%d and wg-y=%d (TILE size); "
            "got wg-x=%d wg-y=%d. "
            "The tile dimensions are fixed at compile time in the kernel source.",
            TILE_SIZE, TILE_SIZE, wg_x, wg_y);
    }

    /* ── Load OpenCL and get GPU device (identical to naive binary) ───────── */
    load_opencl();

    cl_uint nplat = 0;
    cl_int rc = clGetPlatformIDs_fn(0, NULL, &nplat);
    if (rc != CL_SUCCESS || nplat == 0)
        fail_dispatch_fmt("clGetPlatformIDs failed: rc=%d nplat=%u", rc, nplat);

    cl_platform_id platform = NULL;
    clGetPlatformIDs_fn(1, &platform, NULL);

    cl_uint ndev = 0;
    rc = clGetDeviceIDs_fn(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &ndev);
    if (rc != CL_SUCCESS || ndev == 0)
        fail_dispatch_fmt("clGetDeviceIDs(GPU) failed: rc=%d ndev=%u", rc, ndev);

    cl_device_id device = NULL;
    clGetDeviceIDs_fn(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    /* ── Create context and profiling-enabled command queue ───────────────── */
    cl_context_properties ctx_props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        (cl_context_properties)0
    };
    cl_context ctx = clCreateContext_fn(ctx_props, 1, &device, NULL, NULL, &rc);
    if (rc != CL_SUCCESS || !ctx)
        fail_dispatch_fmt("clCreateContext failed: rc=%d", rc);

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

    /* ── Allocate host matrices and fill A (seed), B (seed+1) ────────────── */
    size_t nbytes = (size_t)size * (size_t)size * sizeof(float);

    float *h_A = (float *)malloc(nbytes);
    float *h_B = (float *)malloc(nbytes);
    float *h_C = (float *)malloc(nbytes);
    if (!h_A || !h_B || !h_C)
        fail_dispatch("malloc failed for host matrices (out of memory)");

    fill_matrix(h_A, size, seed);        /* seed=42 → same A as naive + CPU */
    fill_matrix(h_B, size, seed + 1);    /* seed=43 → same B as naive + CPU */
    memset(h_C, 0, nbytes);

    /* ── Create OpenCL buffers ───────────────────────────────────────────── */
    cl_mem buf_A = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     nbytes, h_A, &rc);
    if (rc != CL_SUCCESS || !buf_A)
        fail_dispatch_fmt("clCreateBuffer(A) failed: rc=%d", rc);

    cl_mem buf_B = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     nbytes, h_B, &rc);
    if (rc != CL_SUCCESS || !buf_B) {
        clReleaseMemObject_fn(buf_A);
        fail_dispatch_fmt("clCreateBuffer(B) failed: rc=%d", rc);
    }

    cl_mem buf_C = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, nbytes, NULL, &rc);
    if (rc != CL_SUCCESS || !buf_C) {
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        fail_dispatch_fmt("clCreateBuffer(C) failed: rc=%d", rc);
    }

    /* ── Build the tiled kernel program ──────────────────────────────────── */
    /* The online compiler sees TILE=16 as a compile-time constant and should
     * unroll the inner k-loop (16 iterations) and optimize tile indexing. */
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
                 "clBuildProgram failed: rc=%d | log: %.200s", rc, build_log);
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch(err_buf);
    }

    /* ── Create kernel (name "matmul_tiled" matches KERNEL_SOURCE) ────────── */
    cl_kernel kernel = clCreateKernel_fn(program, "matmul_tiled", &rc);
    if (rc != CL_SUCCESS || !kernel) {
        clReleaseProgram_fn(program);
        clReleaseMemObject_fn(buf_A);
        clReleaseMemObject_fn(buf_B);
        clReleaseMemObject_fn(buf_C);
        fail_dispatch_fmt("clCreateKernel(matmul_tiled) failed: rc=%d", rc);
    }

    /* Set kernel args: A, B, C, N (same signature as naive kernel). */
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

    /* ── Enqueue tiled NDRange ────────────────────────────────────────────── */
    /* Global work size must be a multiple of the workgroup size (TILE_SIZE).
     * round_up ensures this. For N=512, TILE=16: global = {512, 512} exactly. */
    size_t global[2] = { round_up(size, wg_x), round_up(size, wg_y) };
    size_t local[2]  = { (size_t)wg_x, (size_t)wg_y };

    cl_event kernel_event = NULL;
    rc = clEnqueueNDRangeKernel_fn(
        queue, kernel,
        2, NULL,       /* 2D NDRange, no global offset */
        global, local, /* global = {512,512}, local = {16,16} */
        0, NULL, &kernel_event);
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

    /* ── Read CL_PROFILING timestamps ────────────────────────────────────── */
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

    /* ── Read back C and compute fingerprint ─────────────────────────────── */
    /* Same fingerprint formula as CPU and naive GPU variants:
     * fingerprint = sum(C) + frobenius_norm(C).
     * With matching seeds (42/43), tiled and naive should produce identical
     * fingerprints (FP32 arithmetic order may differ slightly, but the
     * accumulated sums should agree within floating-point tolerance). */
    rc = clEnqueueReadBuffer_fn(queue, buf_C, CL_TRUE, 0, nbytes,
                                h_C, 0, NULL, NULL);
    if (rc != CL_SUCCESS)
        fprintf(stderr, "clEnqueueReadBuffer warning: rc=%d; fingerprint unavailable\n", rc);

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

    /* ── Print GPU_DISPATCH_RESULT protocol block ────────────────────────── */
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
