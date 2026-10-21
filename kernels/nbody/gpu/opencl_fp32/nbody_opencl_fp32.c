/*
 * nbody_opencl_fp32.c - direct-summation N-body, OpenCL FP32, Adreno 830.
 *
 * The compute-bound anchor of the roofline. Every other workload in this tree
 * is either memory-bound (saxpy, reduction) or a dense tensor op that the NPU
 * wins outright (matmul, convolution). N-body does O(N^2) arithmetic on O(N)
 * data, so arithmetic intensity is 1.25*N flop/byte and grows with the
 * problem, and QNN cannot express it - which makes it the one workload where
 * the GPU gets a fair fight.
 *
 * Two variants, selected with --variant:
 *   naive - straight global-memory reads
 *   tiled - 64-body __local staging, each global load reused 64x  (default)
 *
 * Shipping both is deliberate. Benchmarking the naive kernel against another
 * backend measures your code, not the silicon; that trap is the subject of the
 * GPU section task P5, so the tuned variant has to exist alongside it.
 *
 * Sections 1, 3 and 4 (CL typedefs, function-pointer table, dlopen helpers)
 * are lifted verbatim from saxpy_opencl_fp32.c.
 *
 * NB: run at shell uid. Under root, clGetPlatformIDs returns -1001.
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


/* Buffer access flags. The SAXPY header this file borrows from only defines
 * CL_MEM_READ_WRITE, because that is all it used. */
#ifndef CL_MEM_READ_ONLY
#define CL_MEM_READ_ONLY  ((cl_mem_flags)(1 << 2))
#endif
#ifndef CL_MEM_WRITE_ONLY
#define CL_MEM_WRITE_ONLY ((cl_mem_flags)(1 << 1))
#endif
#ifndef CL_PROGRAM_BUILD_LOG
#define CL_PROGRAM_BUILD_LOG 0x1183
#endif


/* ── Section 2: Embedded OpenCL C kernel source ──────────────────────────────
 * Two variants of direct-summation N-body, selected with --variant.
 *
 *   naive — one work-item per body, reading every other body straight from
 *           global memory. Correct, and the obvious first thing to write.
 *
 *   tiled — the same arithmetic, but each workgroup cooperatively stages a
 *           64-body tile into __local memory and every lane reads the tile
 *           from there. Each global load is then reused 64 times instead of
 *           once. This is the standard N-body GPU optimisation.
 *
 * Both accumulate j in ascending order, matching the NEON kernel exactly, so
 * all three implementations produce the same fingerprint.
 *
 * Structure-of-arrays, so lane i and lane i+1 read adjacent floats — the
 * coalesced access pattern that kernels/memlab/gpu measures directly.
 *
 * Softening EPS2 = 1e-6f must match nbody_neon.c. IEEE sqrt() and divide, not
 * native_rsqrt(), so the result is comparable across backends.
 */
#define NB_TILE 64

static const char *KERNEL_SOURCE =
    "#define EPS2 1e-6f\n"
    "#define TILE 64\n"
    "\n"
    "__kernel void nbody_naive(\n"
    "    __global const float* px, __global const float* py,\n"
    "    __global const float* pz, __global const float* pm,\n"
    "    __global       float* ax, __global       float* ay,\n"
    "    __global       float* az, const int N)\n"
    "{\n"
    "    int i = (int)get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "    float xi = px[i], yi = py[i], zi = pz[i];\n"
    "    float sx = 0.0f, sy = 0.0f, sz = 0.0f;\n"
    "    for (int j = 0; j < N; j++) {\n"
    "        float dx = px[j] - xi;\n"
    "        float dy = py[j] - yi;\n"
    "        float dz = pz[j] - zi;\n"
    "        float r2 = dx*dx + dy*dy + dz*dz + EPS2;\n"
    "        float r1 = sqrt(r2);\n"
    "        float s  = pm[j] / (r2 * r1);\n"
    "        sx += dx * s; sy += dy * s; sz += dz * s;\n"
    "    }\n"
    "    ax[i] = sx; ay[i] = sy; az[i] = sz;\n"
    "}\n"
    "\n"
    "__attribute__((reqd_work_group_size(TILE, 1, 1)))\n"
    "__kernel void nbody_tiled(\n"
    "    __global const float* px, __global const float* py,\n"
    "    __global const float* pz, __global const float* pm,\n"
    "    __global       float* ax, __global       float* ay,\n"
    "    __global       float* az, const int N)\n"
    "{\n"
    "    int i   = (int)get_global_id(0);\n"
    "    int lid = (int)get_local_id(0);\n"
    "    __local float lx[TILE], ly[TILE], lz[TILE], lm[TILE];\n"
    "    float xi = 0.0f, yi = 0.0f, zi = 0.0f;\n"
    "    if (i < N) { xi = px[i]; yi = py[i]; zi = pz[i]; }\n"
    "    float sx = 0.0f, sy = 0.0f, sz = 0.0f;\n"
    "    for (int base = 0; base < N; base += TILE) {\n"
    "        int j = base + lid;\n"
    "        lx[lid] = px[j]; ly[lid] = py[j];\n"
    "        lz[lid] = pz[j]; lm[lid] = pm[j];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int k = 0; k < TILE; k++) {\n"
    "            float dx = lx[k] - xi;\n"
    "            float dy = ly[k] - yi;\n"
    "            float dz = lz[k] - zi;\n"
    "            float r2 = dx*dx + dy*dy + dz*dz + EPS2;\n"
    "            float r1 = sqrt(r2);\n"
    "            float s  = lm[k] / (r2 * r1);\n"
    "            sx += dx * s; sy += dy * s; sz += dz * s;\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (i < N) { ax[i] = sx; ay[i] = sy; az[i] = sz; }\n"
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
 * Same seed → same A and B → same C → same fingerprint for cross-validation. */
static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

/* u64_to_unit_double: top-53-bit mapping to [0,1), matching the CPU SAXPY
 * reference (kernels/saxpy/cpu/reference.py) and variants exactly. */
static double u64_to_unit_double(uint64_t x) {
    return (double)(x >> 11) * (1.0 / (double)(1ULL << 53));
}

/* fill_xy: fill x[0..n) then y[0..n) from one advancing splitmix64 state,
 * each element in [-1,1). Bit-identical to the CPU SAXPY reference (x drawn
 * first, then y), so the GPU fingerprint cross-validates against CPU. */

/* round_up: round n up to the nearest multiple of m (for global work size). */
static size_t round_up(int n, int m) {
    return (size_t)(((n + m - 1) / m) * m);
}


/* ── Section 5: main ─────────────────────────────────────────────────────────*/

/* splitmix64 stream, seeded independently per array — must match the four
 * streams in nbody_neon.c (seed, seed+1, seed+2, seed+3) exactly, or the two
 * backends simulate different systems and the fingerprints cannot be compared. */
static void fill_stream(float *dst, int n, uint64_t seed) {
    uint64_t st = seed;
    for (int i = 0; i < n; i++) {
        dst[i] = (float)u64_to_unit_double(splitmix64_next(&st));
    }
}

int main(int argc, char **argv) {
    int      bodies  = 4096;
    uint64_t seed    = 42;
    int      reps    = 1;
    int      wg_x    = NB_TILE;
    const char *variant = "tiled";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bodies") == 0 && i + 1 < argc) {
            bodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            bodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--workgroup") == 0 && i + 1 < argc) {
            wg_x = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            variant = argv[++i];
        }
    }

    /* Same constraint as the NEON kernel: multiple of 64. */
    if (bodies < NB_TILE) bodies = NB_TILE;
    bodies &= ~(NB_TILE - 1);
    if (reps < 1) reps = 1;

    int tiled = (strcmp(variant, "tiled") == 0);
    /* The tiled kernel carries reqd_work_group_size(64,1,1); handing it any
     * other local size is a CL_INVALID_WORK_GROUP_SIZE at enqueue, so pin it
     * rather than letting --workgroup produce a confusing driver error. */
    if (tiled) wg_x = NB_TILE;
    if (wg_x < 1) wg_x = NB_TILE;

    load_opencl();

    cl_int rc = CL_SUCCESS;

    cl_uint nplat = 0;
    rc = clGetPlatformIDs_fn(0, NULL, &nplat);
    if (rc != CL_SUCCESS || nplat == 0) {
        fail_dispatch_fmt("clGetPlatformIDs failed: rc=%d, nplat=%u "
                          "(note: `adb root` breaks OpenCL — run as shell)",
                          rc, nplat);
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
    if (rc != CL_SUCCESS || !ctx) fail_dispatch_fmt("clCreateContext failed: rc=%d", rc);

    cl_queue_properties q_props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, (cl_queue_properties)0
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties_fn(ctx, device, q_props, &rc);
    if (rc != CL_SUCCESS || !queue) fail_dispatch_fmt("clCreateCommandQueue failed: rc=%d", rc);

    /* ── Host arrays: structure-of-arrays, four independent streams ──────── */
    size_t nbytes = (size_t)bodies * sizeof(float);
    float *h_x  = (float *)malloc(nbytes);
    float *h_y  = (float *)malloc(nbytes);
    float *h_z  = (float *)malloc(nbytes);
    float *h_m  = (float *)malloc(nbytes);
    float *h_ax = (float *)malloc(nbytes);
    float *h_ay = (float *)malloc(nbytes);
    float *h_az = (float *)malloc(nbytes);
    if (!h_x || !h_y || !h_z || !h_m || !h_ax || !h_ay || !h_az) {
        fail_dispatch("malloc failed for host body arrays");
    }

    fill_stream(h_x, bodies, seed);
    fill_stream(h_y, bodies, seed + 1);
    fill_stream(h_z, bodies, seed + 2);
    fill_stream(h_m, bodies, seed + 3);

    /* ── Device buffers ─────────────────────────────────────────────────── */
    cl_mem b_x  = clCreateBuffer_fn(ctx, CL_MEM_READ_ONLY,  nbytes, NULL, &rc);
    cl_mem b_y  = clCreateBuffer_fn(ctx, CL_MEM_READ_ONLY,  nbytes, NULL, &rc);
    cl_mem b_z  = clCreateBuffer_fn(ctx, CL_MEM_READ_ONLY,  nbytes, NULL, &rc);
    cl_mem b_m  = clCreateBuffer_fn(ctx, CL_MEM_READ_ONLY,  nbytes, NULL, &rc);
    cl_mem b_ax = clCreateBuffer_fn(ctx, CL_MEM_WRITE_ONLY, nbytes, NULL, &rc);
    cl_mem b_ay = clCreateBuffer_fn(ctx, CL_MEM_WRITE_ONLY, nbytes, NULL, &rc);
    cl_mem b_az = clCreateBuffer_fn(ctx, CL_MEM_WRITE_ONLY, nbytes, NULL, &rc);
    if (!b_x || !b_y || !b_z || !b_m || !b_ax || !b_ay || !b_az) {
        fail_dispatch_fmt("clCreateBuffer failed: rc=%d", rc);
    }

    /* Upload. Only the first write is profiled — the four are identical in
     * size, so transfer_ns * 4 is the honest total and one event is enough. */
    cl_event wr_event = NULL;
    clEnqueueWriteBuffer_fn(queue, b_x, CL_FALSE, 0, nbytes, h_x, 0, NULL, &wr_event);
    clEnqueueWriteBuffer_fn(queue, b_y, CL_FALSE, 0, nbytes, h_y, 0, NULL, NULL);
    clEnqueueWriteBuffer_fn(queue, b_z, CL_FALSE, 0, nbytes, h_z, 0, NULL, NULL);
    clEnqueueWriteBuffer_fn(queue, b_m, CL_FALSE, 0, nbytes, h_m, 0, NULL, NULL);
    clFinish_fn(queue);

    /* ── Build ───────────────────────────────────────────────────────────── */
    cl_program program = clCreateProgramWithSource_fn(ctx, 1, &KERNEL_SOURCE, NULL, &rc);
    if (rc != CL_SUCCESS || !program) fail_dispatch_fmt("clCreateProgramWithSource: rc=%d", rc);

    rc = clBuildProgram_fn(program, 1, &device, "", NULL, NULL);
    if (rc != CL_SUCCESS) {
        char log[8192]; size_t log_len = 0;
        clGetProgramBuildInfo_fn(program, device, CL_PROGRAM_BUILD_LOG,
                                 sizeof(log) - 1, log, &log_len);
        log[log_len < sizeof(log) ? log_len : sizeof(log) - 1] = '\0';
        fail_dispatch_fmt("clBuildProgram failed: rc=%d\n%s", rc, log);
    }

    cl_kernel kernel = clCreateKernel_fn(program, tiled ? "nbody_tiled" : "nbody_naive", &rc);
    if (rc != CL_SUCCESS || !kernel) fail_dispatch_fmt("clCreateKernel(%s): rc=%d", variant, rc);

    int argi = 0;
    clSetKernelArg_fn(kernel, argi++, sizeof(cl_mem), &b_x);
    clSetKernelArg_fn(kernel, argi++, sizeof(cl_mem), &b_y);
    clSetKernelArg_fn(kernel, argi++, sizeof(cl_mem), &b_z);
    clSetKernelArg_fn(kernel, argi++, sizeof(cl_mem), &b_m);
    clSetKernelArg_fn(kernel, argi++, sizeof(cl_mem), &b_ax);
    clSetKernelArg_fn(kernel, argi++, sizeof(cl_mem), &b_ay);
    clSetKernelArg_fn(kernel, argi++, sizeof(cl_mem), &b_az);
    rc = clSetKernelArg_fn(kernel, argi++, sizeof(cl_int), &bodies);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clSetKernelArg failed: rc=%d", rc);

    /* ── Dispatch ────────────────────────────────────────────────────────── */
    size_t global[1] = { round_up(bodies, wg_x) };
    size_t local[1]  = { (size_t)wg_x };

    cl_ulong best_ns = 0, cl_start_ns = 0, cl_end_ns = 0;

    for (int r = 0; r < reps; r++) {
        cl_event k_event = NULL;
        rc = clEnqueueNDRangeKernel_fn(queue, kernel, 1, NULL, global, local,
                                       0, NULL, &k_event);
        if (rc != CL_SUCCESS) {
            fail_dispatch_fmt("clEnqueueNDRangeKernel failed: rc=%d "
                              "(global=%zu local=%zu)", rc, global[0], local[0]);
        }
        clFinish_fn(queue);

        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo_fn(k_event, CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &t0, NULL);
        clGetEventProfilingInfo_fn(k_event, CL_PROFILING_COMMAND_END,
                                   sizeof(cl_ulong), &t1, NULL);
        clReleaseEvent_fn(k_event);

        /* Best-of-N. The GPU clock cannot be pinned while OpenCL is usable
         * (root kills the ICD), so the minimum is the least DVFS-contaminated
         * sample available. */
        cl_ulong d = t1 - t0;
        if (best_ns == 0 || d < best_ns) { best_ns = d; cl_start_ns = t0; cl_end_ns = t1; }
    }

    /* ── Read back ───────────────────────────────────────────────────────── */
    cl_event rd_event = NULL;
    clEnqueueReadBuffer_fn(queue, b_ax, CL_FALSE, 0, nbytes, h_ax, 0, NULL, &rd_event);
    clEnqueueReadBuffer_fn(queue, b_ay, CL_FALSE, 0, nbytes, h_ay, 0, NULL, NULL);
    clEnqueueReadBuffer_fn(queue, b_az, CL_FALSE, 0, nbytes, h_az, 0, NULL, NULL);
    clFinish_fn(queue);

    cl_ulong transfer_ns = 0, readback_ns = 0;
    if (wr_event) {
        cl_ulong a = 0, b = 0;
        clGetEventProfilingInfo_fn(wr_event, CL_PROFILING_COMMAND_START, sizeof(a), &a, NULL);
        clGetEventProfilingInfo_fn(wr_event, CL_PROFILING_COMMAND_END,   sizeof(b), &b, NULL);
        transfer_ns = (b - a) * 4;      /* four identically sized uploads */
        clReleaseEvent_fn(wr_event);
    }
    if (rd_event) {
        cl_ulong a = 0, b = 0;
        clGetEventProfilingInfo_fn(rd_event, CL_PROFILING_COMMAND_START, sizeof(a), &a, NULL);
        clGetEventProfilingInfo_fn(rd_event, CL_PROFILING_COMMAND_END,   sizeof(b), &b, NULL);
        readback_ns = (b - a) * 3;      /* three identically sized downloads */
        clReleaseEvent_fn(rd_event);
    }

    /* Fingerprint: identical formula to nbody_neon.c. */
    double sum = 0.0, sq = 0.0;
    for (int i = 0; i < bodies; i++) {
        sum += (double)h_ax[i] + (double)h_ay[i] + (double)h_az[i];
        sq  += (double)h_ax[i] * h_ax[i] + (double)h_ay[i] * h_ay[i]
             + (double)h_az[i] * h_az[i];
    }
    double fingerprint = sum + sqrt(sq);

    clReleaseKernel_fn(kernel);
    clReleaseProgram_fn(program);
    clReleaseMemObject_fn(b_x);  clReleaseMemObject_fn(b_y);
    clReleaseMemObject_fn(b_z);  clReleaseMemObject_fn(b_m);
    clReleaseMemObject_fn(b_ax); clReleaseMemObject_fn(b_ay);
    clReleaseMemObject_fn(b_az);
    clReleaseCommandQueue_fn(queue);
    clReleaseContext_fn(ctx);
    free(h_x); free(h_y); free(h_z); free(h_m);
    free(h_ax); free(h_ay); free(h_az);

    /* flop/bytes are PER DISPATCH — duration_ns is also per dispatch, so the
     * ratio profiler.py computes is correct without an iteration divisor.
     * 20 flop per interaction is the N-body literature convention. */
    long long flop  = 20LL * bodies * bodies;
    long long bytes = 4LL * bodies * bodies * (long long)sizeof(float);

    printf("GPU_DISPATCH_RESULT\n");
    printf("status: OK\n");
    printf("cl_start_ns: %llu\n", (unsigned long long)cl_start_ns);
    printf("cl_end_ns: %llu\n",   (unsigned long long)cl_end_ns);
    printf("duration_ns: %llu\n", (unsigned long long)best_ns);
    printf("transfer_ns: %llu\n", (unsigned long long)transfer_ns);
    printf("readback_ns: %llu\n", (unsigned long long)readback_ns);
    printf("workgroup_x: %d\n",   wg_x);
    printf("workgroup_y: 1\n");
    printf("workgroup_z: 1\n");
    printf("global_x: %zu\n",     global[0]);
    printf("global_y: 1\n");
    printf("global_z: 1\n");
    printf("variant: %s\n",       tiled ? "tiled" : "naive");
    printf("bodies: %d\n",        bodies);
    printf("iterations: %d\n",    reps);
    printf("flop: %lld\n",        flop);
    printf("bytes: %lld\n",       bytes);
    printf("fingerprint: %.6e\n", fingerprint);
    printf("GPU_DISPATCH_END\n");
    fflush(stdout);

    return 0;
}
