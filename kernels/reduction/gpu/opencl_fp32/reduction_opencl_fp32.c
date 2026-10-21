/*
 * reduction_opencl_fp32.c — sum-reduction on the Adreno 830 (roofline anchor #2, GPU).
 *
 * s <- sum(x_i), fp32, via a multi-pass local-memory tree reduction:
 *   - grid-stride load: each work-item sums several input elements into a
 *     private accumulator (breaks the N->partials ratio from workgroup count);
 *   - local-memory tree: the workgroup folds its accumulators to one partial;
 *   - relaunch on the partials until a single scalar remains.
 * "duration_ns" is the SUM of every pass's kernel time (the honest device-compute
 * figure, decision R7a). The result is one float, so "readback_ns" is a 4-byte
 * copy ~= 0 — the whole reason reduction is not a SAXPY rerun: the output
 * transport tax that punished the GPU on SAXPY (readback of N elements) vanishes
 * here, which is prediction R-i in kernels/reduction/DESIGN.md.
 *
 * Same splitmix64 input as the CPU reduction variants (same seed -> same x ->
 * same sum), so the GPU fingerprint cross-validates against reference.py within
 * 1e-3 relative tolerance (decision R6; fp32 reduction order differs by design).
 *
 * Argv:
 *   --size N     vector length (default 4194304 = 2^22; must be > 0)
 *   --seed N     splitmix64 seed for x (default 42)
 *   --wg-x N     workgroup size (default 256; must be a power of two, <=1024)
 *
 * Stdout (GPU_DISPATCH_RESULT block): status, duration_ns (summed passes),
 * transfer_ns (upload of x), readback_ns (4-byte result), num_passes, elems,
 * bytes (= N*4, read-only), fingerprint (= the sum).
 *
 * Build (inside control container from /workspace):
 *   bash kernels/reduction/gpu/opencl_fp32/build.sh
 */

#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* ── Section 1: Minimal OpenCL type definitions (mirror CL/cl.h) ── */

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

#define CL_FALSE ((cl_bool)0)
#define CL_TRUE  ((cl_bool)1)


/* ── Section 2: Embedded OpenCL C kernel ──
 * Grid-stride load into a private accumulator, then a local-memory tree fold.
 * One partial per workgroup is written to out[group_id]. Relaunched by the host
 * on the partials until a single element remains. */
static const char *KERNEL_SOURCE =
    "__kernel void reduce_sum(\n"
    "    __global const float* in,\n"
    "    __global       float* out,\n"
    "    const int N,\n"
    "    __local        float* scratch)\n"
    "{\n"
    "    int lid    = (int)get_local_id(0);\n"
    "    int lsize  = (int)get_local_size(0);\n"
    "    int gid    = (int)get_global_id(0);\n"
    "    int stride = (int)get_global_size(0);\n"
    "    /* grid-stride: each item accumulates several inputs into a private reg */\n"
    "    float acc = 0.0f;\n"
    "    for (int i = gid; i < N; i += stride) acc += in[i];\n"
    "    scratch[lid] = acc;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    /* tree reduction in local memory (lsize is a power of two) */\n"
    "    for (int s = lsize >> 1; s > 0; s >>= 1) {\n"
    "        if (lid < s) scratch[lid] += scratch[lid + s];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (lid == 0) out[get_group_id(0)] = scratch[0];\n"
    "}\n";


/* ── Section 3: Function pointer types and globals ── */

typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_context (*pfn_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *,
    void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
typedef cl_command_queue (*pfn_clCreateCommandQueueWithProperties)(cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
typedef cl_mem (*pfn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_program (*pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_int (*pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void (*)(cl_program, void *), void *);
typedef cl_int (*pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
typedef cl_kernel (*pfn_clCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_int (*pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
    const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
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


/* ── Section 4: Helpers ── */

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
    /* Do NOT dlclose(h) — Adreno atexit handlers may crash on close. */
}

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static double u64_to_unit_double(uint64_t x) {
    return (double)(x >> 11) * (1.0 / (double)(1ULL << 53));
}

/* fill_x: x[0..n) from an advancing splitmix64 state, each in [-1,1).
 * Bit-identical to kernels/reduction/reference.py (x takes the first n draws). */
static void fill_x(float *x, int n, uint64_t seed) {
    uint64_t state = seed;
    for (int i = 0; i < n; i++) {
        x[i] = (float)(2.0 * u64_to_unit_double(splitmix64_next(&state)) - 1.0);
    }
}

static int is_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }


/* ── Section 5: main ── */

int main(int argc, char **argv) {
    int      size = 4194304; /* 2^22 */
    uint64_t seed = 42;
    int      wg   = 256;     /* workgroup size; power of two for the tree */
    const int MAX_GROUPS = 1024; /* cap partials per pass; grid-stride covers the rest */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--wg-x") == 0 && i + 1 < argc) {
            wg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--wg-y") == 0 && i + 1 < argc) {
            ++i; /* accepted for CLI compatibility; ignored (1D reduction) */
        } else if (strcmp(argv[i], "--wg-z") == 0 && i + 1 < argc) {
            ++i;
        }
    }

    if (size <= 0) fail_dispatch("invalid arguments: size must be > 0");
    if (!is_pow2(wg) || wg > 1024) fail_dispatch("invalid --wg-x: must be a power of two <= 1024");

    load_opencl();

    cl_uint nplat = 0;
    cl_int rc = clGetPlatformIDs_fn(0, NULL, &nplat);
    if (rc != CL_SUCCESS || nplat == 0) fail_dispatch_fmt("clGetPlatformIDs failed: rc=%d, nplat=%u", rc, nplat);
    cl_platform_id platform = NULL;
    clGetPlatformIDs_fn(1, &platform, NULL);

    cl_uint ndev = 0;
    rc = clGetDeviceIDs_fn(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &ndev);
    if (rc != CL_SUCCESS || ndev == 0) fail_dispatch_fmt("clGetDeviceIDs(GPU) failed: rc=%d, ndev=%u", rc, ndev);
    cl_device_id device = NULL;
    clGetDeviceIDs_fn(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context_properties ctx_props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, (cl_context_properties)0
    };
    cl_context ctx = clCreateContext_fn(ctx_props, 1, &device, NULL, NULL, &rc);
    if (rc != CL_SUCCESS || !ctx) fail_dispatch_fmt("clCreateContext failed: rc=%d", rc);

    cl_queue_properties q_props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, (cl_queue_properties)0
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties_fn(ctx, device, q_props, &rc);
    if (rc != CL_SUCCESS || !queue) fail_dispatch_fmt("clCreateCommandQueueWithProperties failed: rc=%d", rc);

    /* ── Host input ── */
    size_t nbytes = (size_t)size * sizeof(float);
    float *h_x = (float *)malloc(nbytes);
    if (!h_x) fail_dispatch("malloc failed for host vector (out of memory)");
    fill_x(h_x, size, seed);

    /* ── Device buffers: input x, and two ping-pong partial buffers ──
     * The largest partial count is MAX_GROUPS, so the ping-pong buffers hold
     * MAX_GROUPS floats each. Pass 1 reads buf_x; later passes ping-pong. */
    cl_mem buf_x = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, nbytes, NULL, &rc);
    if (rc != CL_SUCCESS || !buf_x) fail_dispatch_fmt("clCreateBuffer(x) failed: rc=%d", rc);
    cl_mem buf_p0 = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, (size_t)MAX_GROUPS * sizeof(float), NULL, &rc);
    if (rc != CL_SUCCESS || !buf_p0) fail_dispatch_fmt("clCreateBuffer(p0) failed: rc=%d", rc);
    cl_mem buf_p1 = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, (size_t)MAX_GROUPS * sizeof(float), NULL, &rc);
    if (rc != CL_SUCCESS || !buf_p1) fail_dispatch_fmt("clCreateBuffer(p1) failed: rc=%d", rc);

    /* ── Upload x (profiled: this is the one real transport for reduction) ── */
    cl_event write_event = NULL;
    rc = clEnqueueWriteBuffer_fn(queue, buf_x, CL_FALSE, 0, nbytes, h_x, 0, NULL, &write_event);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clEnqueueWriteBuffer(x) failed: rc=%d", rc);
    rc = clFinish_fn(queue);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clFinish (after write) failed: rc=%d", rc);
    cl_ulong w0 = 0, w1 = 0;
    clGetEventProfilingInfo_fn(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &w0, NULL);
    clGetEventProfilingInfo_fn(write_event, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &w1, NULL);
    clReleaseEvent_fn(write_event);
    cl_ulong transfer_ns = w1 - w0;

    /* ── Build program + kernel ── */
    cl_program program = clCreateProgramWithSource_fn(ctx, 1, &KERNEL_SOURCE, NULL, &rc);
    if (rc != CL_SUCCESS || !program) fail_dispatch_fmt("clCreateProgramWithSource failed: rc=%d", rc);
    rc = clBuildProgram_fn(program, 1, &device, "-cl-fast-relaxed-math -cl-std=CL3.0", NULL, NULL);
    if (rc != CL_SUCCESS) {
        char build_log[4096]; memset(build_log, 0, sizeof(build_log));
        clGetProgramBuildInfo_fn(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log) - 1, build_log, NULL);
        char err_buf[512];
        snprintf(err_buf, sizeof(err_buf), "clBuildProgram failed: rc=%d | log: %.200s", rc, build_log);
        fail_dispatch(err_buf);
    }
    cl_kernel kernel = clCreateKernel_fn(program, "reduce_sum", &rc);
    if (rc != CL_SUCCESS || !kernel) fail_dispatch_fmt("clCreateKernel failed: rc=%d", rc);

    /* ── Multi-pass reduction loop ──
     * cur = live element count. Each pass reduces cur -> num_groups. Ping-pong
     * the partial buffers. duration_ns accumulates every pass's kernel time. */
    cl_mem  in_buf   = buf_x;
    cl_mem  out_bufs[2] = { buf_p0, buf_p1 };
    int     cur      = size;
    int     pass     = 0;
    int     toggle   = 0;
    cl_ulong duration_ns = 0;
    cl_mem  last_out = NULL;

    while (cur > 1) {
        /* number of workgroups this pass: enough to cover cur, capped so the
         * grid-stride loop does the heavy lifting rather than launching millions
         * of groups. */
        int num_groups = (cur + wg - 1) / wg;
        if (num_groups > MAX_GROUPS) num_groups = MAX_GROUPS;
        cl_mem out_buf = out_bufs[toggle];

        rc  = clSetKernelArg_fn(kernel, 0, sizeof(cl_mem), &in_buf);
        rc |= clSetKernelArg_fn(kernel, 1, sizeof(cl_mem), &out_buf);
        rc |= clSetKernelArg_fn(kernel, 2, sizeof(cl_int), &cur);
        rc |= clSetKernelArg_fn(kernel, 3, (size_t)wg * sizeof(float), NULL); /* __local scratch */
        if (rc != CL_SUCCESS) fail_dispatch_fmt("clSetKernelArg (pass %d) failed: rc=%d", pass, rc);

        size_t global[1] = { (size_t)num_groups * (size_t)wg };
        size_t local[1]  = { (size_t)wg };
        cl_event kev = NULL;
        rc = clEnqueueNDRangeKernel_fn(queue, kernel, 1, NULL, global, local, 0, NULL, &kev);
        if (rc != CL_SUCCESS) fail_dispatch_fmt("clEnqueueNDRangeKernel (pass %d) failed: rc=%d", pass, rc);
        rc = clFinish_fn(queue);
        if (rc != CL_SUCCESS) fail_dispatch_fmt("clFinish (pass %d) failed: rc=%d", pass, rc);

        cl_ulong s0 = 0, s1 = 0;
        clGetEventProfilingInfo_fn(kev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s0, NULL);
        clGetEventProfilingInfo_fn(kev, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &s1, NULL);
        clReleaseEvent_fn(kev);
        duration_ns += (s1 - s0);

        last_out = out_buf;
        in_buf   = out_buf;      /* next pass reduces this pass's partials */
        cur      = num_groups;   /* one partial per group */
        toggle  ^= 1;            /* ping-pong so we never read+write the same buf */
        pass++;
    }

    /* ── Read back the single scalar result (4 bytes ~= 0 transport: R-i) ── */
    float result = 0.0f;
    cl_ulong readback_ns = 0;
    if (last_out == NULL) {
        /* size == 1 edge case: no pass ran; just copy the one input element. */
        result = h_x[0];
    } else {
        cl_event rev = NULL;
        rc = clEnqueueReadBuffer_fn(queue, last_out, CL_TRUE, 0, sizeof(float), &result, 0, NULL, &rev);
        if (rc != CL_SUCCESS) fprintf(stderr, "clEnqueueReadBuffer warning: rc=%d\n", rc);
        if (rev) {
            cl_ulong r0 = 0, r1 = 0;
            if (clGetEventProfilingInfo_fn(rev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &r0, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo_fn(rev, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &r1, NULL) == CL_SUCCESS) {
                readback_ns = r1 - r0;
            }
            clReleaseEvent_fn(rev);
        }
    }

    clReleaseKernel_fn(kernel);
    clReleaseProgram_fn(program);
    clReleaseMemObject_fn(buf_x);
    clReleaseMemObject_fn(buf_p0);
    clReleaseMemObject_fn(buf_p1);
    clReleaseCommandQueue_fn(queue);
    clReleaseContext_fn(ctx);
    free(h_x);

    /* fingerprint = the sum itself (compared to reference.py fp64 truth, R6). */
    int64_t bytes = (int64_t)size * (int64_t)sizeof(float); /* read-only */
    printf("GPU_DISPATCH_RESULT\n");
    printf("status: OK\n");
    printf("duration_ns: %llu\n",   (unsigned long long)duration_ns);
    printf("transfer_ns: %llu\n",   (unsigned long long)transfer_ns);
    printf("readback_ns: %llu\n",   (unsigned long long)readback_ns);
    printf("num_passes: %d\n",      pass);
    printf("workgroup_x: %d\n",     wg);
    printf("elems: %d\n",           size);
    printf("bytes: %lld\n",         (long long)bytes);
    printf("fingerprint: %.6e\n",   (double)result);
    printf("GPU_DISPATCH_END\n");
    fflush(stdout);
    return 0;
}
