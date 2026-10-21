/*
 * conv_opencl_fp32.c — direct 2D convolution on the Adreno 830 (roofline mid-AI, GPU).
 *
 * out = conv2d(in, W), dense KxK, stride 1, same zero padding, NHWC in/out, OHWI
 * weights. One work-item computes one output element out[oh,ow,oc] via the
 * K²·Cin MAC loop. AI ≈ 2.25·C for Cin=Cout=C (default C=8, K=3, 256² → AI≈18),
 * channels parameterized for the roofline AI sweep.
 *
 * Unlike reduction, the output is H·W·Cout (not a scalar), so host<->device
 * transport is back in play: transfer_ns = upload(input+weights),
 * readback_ns = download(output). Same splitmix64 input+weights as the CPU
 * variants (input first, then weights) — fingerprint cross-validates within 1e-3.
 *
 * Argv: --H --W --cin --cout --k --seed --wg-x (default 256/256/8/8/3/42/64)
 * Stdout: GPU_DISPATCH_RESULT block (duration_ns, transfer_ns, readback_ns,
 *         H,W,cin,cout,k, flop, bytes, fingerprint).
 */

#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *cl_platform_id; typedef void *cl_device_id; typedef void *cl_context;
typedef void *cl_command_queue; typedef void *cl_mem; typedef void *cl_program;
typedef void *cl_kernel; typedef void *cl_event;
typedef int32_t cl_int; typedef uint32_t cl_uint; typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield; typedef uint32_t cl_bool; typedef uint64_t cl_device_type;
typedef uintptr_t cl_device_info; typedef uintptr_t cl_platform_info;
typedef uintptr_t cl_context_properties; typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_queue_properties;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU ((cl_device_type)(1 << 2))
#define CL_CONTEXT_PLATFORM ((cl_context_properties)0x1084)
#define CL_MEM_READ_WRITE ((cl_mem_flags)(1 << 0))
#define CL_QUEUE_PROPERTIES ((cl_queue_properties)0x1093)
#define CL_QUEUE_PROFILING_ENABLE ((cl_queue_properties)(1 << 1))
#define CL_PROFILING_COMMAND_START ((cl_uint)0x1282)
#define CL_PROFILING_COMMAND_END ((cl_uint)0x1283)
#define CL_PROGRAM_BUILD_LOG ((cl_uint)0x1183)
#define CL_FALSE ((cl_bool)0)
#define CL_TRUE ((cl_bool)1)

static const char *KERNEL_SOURCE =
    "__kernel void conv2d(\n"
    "    __global const float* in, __global const float* w, __global float* out,\n"
    "    const int H, const int W, const int Cin, const int Cout, const int K)\n"
    "{\n"
    "    int gid = (int)get_global_id(0);\n"
    "    int total = H * W * Cout;\n"
    "    if (gid >= total) return;\n"
    "    int oc = gid % Cout;\n"
    "    int ow = (gid / Cout) % W;\n"
    "    int oh = (gid / Cout) / W;\n"
    "    int pad = K >> 1;\n"
    "    float acc = 0.0f;\n"
    "    for (int kh = 0; kh < K; kh++) {\n"
    "        int ih = oh + kh - pad; if (ih < 0 || ih >= H) continue;\n"
    "        for (int kw = 0; kw < K; kw++) {\n"
    "            int iw = ow + kw - pad; if (iw < 0 || iw >= W) continue;\n"
    "            __global const float* in_px = in + ((long)ih * W + iw) * Cin;\n"
    "            __global const float* w_px  = w + (((long)oc * K + kh) * K + kw) * Cin;\n"
    "            for (int ic = 0; ic < Cin; ic++) acc = mad(in_px[ic], w_px[ic], acc);\n"
    "        }\n"
    "    }\n"
    "    out[((long)oh * W + ow) * Cout + oc] = acc;\n"
    "}\n";

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
typedef cl_int (*pfn_clReleaseMemObject)(cl_mem);

static pfn_clGetPlatformIDs clGetPlatformIDs_fn; static pfn_clGetDeviceIDs clGetDeviceIDs_fn;
static pfn_clCreateContext clCreateContext_fn; static pfn_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties_fn;
static pfn_clCreateBuffer clCreateBuffer_fn; static pfn_clCreateProgramWithSource clCreateProgramWithSource_fn;
static pfn_clBuildProgram clBuildProgram_fn; static pfn_clGetProgramBuildInfo clGetProgramBuildInfo_fn;
static pfn_clCreateKernel clCreateKernel_fn; static pfn_clSetKernelArg clSetKernelArg_fn;
static pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel_fn; static pfn_clEnqueueReadBuffer clEnqueueReadBuffer_fn;
static pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer_fn; static pfn_clFinish clFinish_fn;
static pfn_clGetEventProfilingInfo clGetEventProfilingInfo_fn; static pfn_clReleaseEvent clReleaseEvent_fn;
static pfn_clReleaseMemObject clReleaseMemObject_fn;

static void fail_dispatch(const char *msg) {
    printf("GPU_DISPATCH_RESULT\nstatus: FAILED\nerror: %s\nGPU_DISPATCH_END\n", msg);
    fflush(stdout); exit(1);
}
static void fail_dispatch_fmt(const char *fmt, ...) {
    char buf[512]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
    fail_dispatch(buf);
}
static void *load_sym(void *h, const char *name) {
    void *s = dlsym(h, name);
    if (!s) { char b[256]; snprintf(b, sizeof(b), "dlsym(%s) failed: %s", name, dlerror()); fail_dispatch(b); }
    return s;
}
static void load_opencl(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) { char b[256]; snprintf(b, sizeof(b), "dlopen libOpenCL.so failed: %s", dlerror()); fail_dispatch(b); }
    clGetPlatformIDs_fn = (pfn_clGetPlatformIDs)load_sym(h, "clGetPlatformIDs");
    clGetDeviceIDs_fn = (pfn_clGetDeviceIDs)load_sym(h, "clGetDeviceIDs");
    clCreateContext_fn = (pfn_clCreateContext)load_sym(h, "clCreateContext");
    clCreateCommandQueueWithProperties_fn = (pfn_clCreateCommandQueueWithProperties)load_sym(h, "clCreateCommandQueueWithProperties");
    clCreateBuffer_fn = (pfn_clCreateBuffer)load_sym(h, "clCreateBuffer");
    clCreateProgramWithSource_fn = (pfn_clCreateProgramWithSource)load_sym(h, "clCreateProgramWithSource");
    clBuildProgram_fn = (pfn_clBuildProgram)load_sym(h, "clBuildProgram");
    clGetProgramBuildInfo_fn = (pfn_clGetProgramBuildInfo)load_sym(h, "clGetProgramBuildInfo");
    clCreateKernel_fn = (pfn_clCreateKernel)load_sym(h, "clCreateKernel");
    clSetKernelArg_fn = (pfn_clSetKernelArg)load_sym(h, "clSetKernelArg");
    clEnqueueNDRangeKernel_fn = (pfn_clEnqueueNDRangeKernel)load_sym(h, "clEnqueueNDRangeKernel");
    clEnqueueReadBuffer_fn = (pfn_clEnqueueReadBuffer)load_sym(h, "clEnqueueReadBuffer");
    clEnqueueWriteBuffer_fn = (pfn_clEnqueueWriteBuffer)load_sym(h, "clEnqueueWriteBuffer");
    clFinish_fn = (pfn_clFinish)load_sym(h, "clFinish");
    clGetEventProfilingInfo_fn = (pfn_clGetEventProfilingInfo)load_sym(h, "clGetEventProfilingInfo");
    clReleaseEvent_fn = (pfn_clReleaseEvent)load_sym(h, "clReleaseEvent");
    clReleaseMemObject_fn = (pfn_clReleaseMemObject)load_sym(h, "clReleaseMemObject");
}

static uint64_t splitmix64_next(uint64_t *s) {
    uint64_t z = (*s += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static double u2d(uint64_t x) { return (double)(x >> 11) * (1.0 / (double)(1ULL << 53)); }
static void fill_stream(float *dst, int64_t n, uint64_t *st) {
    for (int64_t i = 0; i < n; i++) dst[i] = (float)(2.0 * u2d(splitmix64_next(st)) - 1.0);
}
static size_t round_up(int64_t n, int m) { return (size_t)(((n + m - 1) / m) * m); }
static cl_ulong prof(cl_event e) {
    cl_ulong s = 0, en = 0;
    clGetEventProfilingInfo_fn(e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL);
    clGetEventProfilingInfo_fn(e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &en, NULL);
    return en - s;
}

int main(int argc, char **argv) {
    int H = 256, W = 256, Cin = 8, Cout = 8, K = 3, wg = 64;
    uint64_t seed = 42;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--H") && i+1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--W") && i+1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cin") && i+1 < argc) Cin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cout") && i+1 < argc) Cout = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--wg-x") && i+1 < argc) wg = atoi(argv[++i]);
    }
    if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || wg <= 0) fail_dispatch("invalid args");

    load_opencl();
    cl_uint nplat = 0; cl_int rc = clGetPlatformIDs_fn(0, NULL, &nplat);
    if (rc != CL_SUCCESS || nplat == 0) fail_dispatch_fmt("clGetPlatformIDs rc=%d", rc);
    cl_platform_id plat = NULL; clGetPlatformIDs_fn(1, &plat, NULL);
    cl_uint ndev = 0; rc = clGetDeviceIDs_fn(plat, CL_DEVICE_TYPE_GPU, 0, NULL, &ndev);
    if (rc != CL_SUCCESS || ndev == 0) fail_dispatch_fmt("clGetDeviceIDs rc=%d", rc);
    cl_device_id dev = NULL; clGetDeviceIDs_fn(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    cl_context_properties cp[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0 };
    cl_context ctx = clCreateContext_fn(cp, 1, &dev, NULL, NULL, &rc);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clCreateContext rc=%d", rc);
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties_fn(ctx, dev, qp, &rc);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clCreateCommandQueue rc=%d", rc);

    int64_t n_in = (int64_t)H * W * Cin, n_w = (int64_t)Cout * K * K * Cin, n_out = (int64_t)H * W * Cout;
    float *h_in = malloc(n_in * sizeof(float)), *h_w = malloc(n_w * sizeof(float)), *h_out = malloc(n_out * sizeof(float));
    if (!h_in || !h_w || !h_out) fail_dispatch("host malloc failed");
    uint64_t st = seed; fill_stream(h_in, n_in, &st); fill_stream(h_w, n_w, &st);

    cl_mem b_in = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, n_in * sizeof(float), NULL, &rc);
    cl_mem b_w  = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, n_w * sizeof(float), NULL, &rc);
    cl_mem b_out = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, n_out * sizeof(float), NULL, &rc);
    if (!b_in || !b_w || !b_out) fail_dispatch("clCreateBuffer failed");

    cl_event ei = NULL, ew = NULL;
    rc = clEnqueueWriteBuffer_fn(q, b_in, CL_FALSE, 0, n_in * sizeof(float), h_in, 0, NULL, &ei);
    rc |= clEnqueueWriteBuffer_fn(q, b_w, CL_FALSE, 0, n_w * sizeof(float), h_w, 0, NULL, &ew);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clEnqueueWriteBuffer rc=%d", rc);
    clFinish_fn(q);
    cl_ulong transfer_ns = prof(ei) + prof(ew);
    clReleaseEvent_fn(ei); clReleaseEvent_fn(ew);

    cl_program prog = clCreateProgramWithSource_fn(ctx, 1, &KERNEL_SOURCE, NULL, &rc);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clCreateProgramWithSource rc=%d", rc);
    rc = clBuildProgram_fn(prog, 1, &dev, "-cl-fast-relaxed-math -cl-std=CL3.0", NULL, NULL);
    if (rc != CL_SUCCESS) {
        char log[4096]; memset(log, 0, sizeof(log));
        clGetProgramBuildInfo_fn(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        fail_dispatch_fmt("clBuildProgram rc=%d | %.200s", rc, log);
    }
    cl_kernel k = clCreateKernel_fn(prog, "conv2d", &rc);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clCreateKernel rc=%d", rc);
    rc  = clSetKernelArg_fn(k, 0, sizeof(cl_mem), &b_in);
    rc |= clSetKernelArg_fn(k, 1, sizeof(cl_mem), &b_w);
    rc |= clSetKernelArg_fn(k, 2, sizeof(cl_mem), &b_out);
    rc |= clSetKernelArg_fn(k, 3, sizeof(cl_int), &H);
    rc |= clSetKernelArg_fn(k, 4, sizeof(cl_int), &W);
    rc |= clSetKernelArg_fn(k, 5, sizeof(cl_int), &Cin);
    rc |= clSetKernelArg_fn(k, 6, sizeof(cl_int), &Cout);
    rc |= clSetKernelArg_fn(k, 7, sizeof(cl_int), &K);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clSetKernelArg rc=%d", rc);

    size_t global[1] = { round_up(n_out, wg) };
    size_t local[1] = { (size_t)wg };
    cl_event kev = NULL;
    rc = clEnqueueNDRangeKernel_fn(q, k, 1, NULL, global, local, 0, NULL, &kev);
    if (rc != CL_SUCCESS) fail_dispatch_fmt("clEnqueueNDRangeKernel rc=%d", rc);
    clFinish_fn(q);
    cl_ulong duration_ns = prof(kev);
    clReleaseEvent_fn(kev);

    cl_event rev = NULL;
    rc = clEnqueueReadBuffer_fn(q, b_out, CL_TRUE, 0, n_out * sizeof(float), h_out, 0, NULL, &rev);
    cl_ulong readback_ns = (rc == CL_SUCCESS && rev) ? prof(rev) : 0;
    if (rev) clReleaseEvent_fn(rev);

    double sum = 0.0, sq = 0.0;
    for (int64_t i = 0; i < n_out; i++) { double v = h_out[i]; sum += v; sq += v * v; }
    double fingerprint = sum + sqrt(sq);

    clReleaseMemObject_fn(b_in); clReleaseMemObject_fn(b_w); clReleaseMemObject_fn(b_out);
    free(h_in); free(h_w); free(h_out);

    int64_t flop = (int64_t)2 * H * W * Cout * K * K * Cin;
    int64_t bytes = (int64_t)4 * (n_in + n_w + n_out);
    printf("GPU_DISPATCH_RESULT\nstatus: OK\n");
    printf("duration_ns: %llu\ntransfer_ns: %llu\nreadback_ns: %llu\n",
           (unsigned long long)duration_ns, (unsigned long long)transfer_ns, (unsigned long long)readback_ns);
    printf("H: %d\nW: %d\ncin: %d\ncout: %d\nk: %d\n", H, W, Cin, Cout, K);
    printf("flop: %lld\nbytes: %lld\nfingerprint: %.6e\nGPU_DISPATCH_END\n",
           (long long)flop, (long long)bytes, fingerprint);
    fflush(stdout);
    return 0;
}
