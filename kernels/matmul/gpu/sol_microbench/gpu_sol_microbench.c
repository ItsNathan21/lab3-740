/*
 * gpu_sol_microbench.c
 *
 * Standalone Android arm64 OpenCL microbenchmarks for Adreno speed-of-light
 * checks. This is intentionally separate from the matmul GPU_DISPATCH_RESULT
 * protocol: these kernels are calibration workloads, not model kernels.
 *
 * Modes:
 *   --mode alu     register-heavy FP32 FMA loop, reports GFLOP/s
 *   --mode reg     register/ALU dependency proxy, reports GFLOP/s and GB/s proxy
 *   --mode stream  large global-memory copy/update loop, reports GB/s
 *   --mode global  global-buffer read working-set sweep, reports GB/s
 *   --mode local   local-memory read loop, reports GB/s
 *
 * The binary builds the OpenCL program once, allocates buffers once, enqueues
 * warmup kernels, then enqueues the timed kernels back-to-back before waiting.
 * That gives qprof a sustained GPU window instead of a series of short process
 * launches and online-compile gaps.
 */

#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
typedef uintptr_t cl_context_properties;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_queue_properties;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU ((cl_device_type)(1 << 2))
#define CL_CONTEXT_PLATFORM ((cl_context_properties)0x1084)
#define CL_MEM_READ_WRITE ((cl_mem_flags)(1 << 0))
#define CL_QUEUE_PROPERTIES ((cl_queue_properties)0x1093)
#define CL_QUEUE_PROFILING_ENABLE ((cl_queue_properties)(1 << 1))
#define CL_TRUE ((cl_bool)1)
#define CL_FALSE ((cl_bool)0)
#define CL_PROFILING_COMMAND_START ((cl_uint)0x1282)
#define CL_PROFILING_COMMAND_END ((cl_uint)0x1283)
#define CL_PROGRAM_BUILD_LOG ((cl_uint)0x1183)

typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_context (*pfn_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                          void (*)(const char *, const void *, size_t, void *),
                                          void *, cl_int *);
typedef cl_command_queue (*pfn_clCreateCommandQueueWithProperties)(cl_context, cl_device_id,
                                                                   const cl_queue_properties *, cl_int *);
typedef cl_mem (*pfn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_program (*pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char **,
                                                    const size_t *, cl_int *);
typedef cl_int (*pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *,
                                     void (*)(cl_program, void *), void *);
typedef cl_int (*pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
typedef cl_kernel (*pfn_clCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_int (*pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
                                             const size_t *, const size_t *, cl_uint,
                                             const cl_event *, cl_event *);
typedef cl_int (*pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                           const void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                          void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*pfn_clFinish)(cl_command_queue);
typedef cl_int (*pfn_clGetEventProfilingInfo)(cl_event, cl_uint, size_t, void *, size_t *);
typedef cl_int (*pfn_clReleaseEvent)(cl_event);
typedef cl_int (*pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (*pfn_clReleaseProgram)(cl_program);
typedef cl_int (*pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (*pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*pfn_clReleaseContext)(cl_context);

static pfn_clGetPlatformIDs clGetPlatformIDs_fn;
static pfn_clGetDeviceIDs clGetDeviceIDs_fn;
static pfn_clCreateContext clCreateContext_fn;
static pfn_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties_fn;
static pfn_clCreateBuffer clCreateBuffer_fn;
static pfn_clCreateProgramWithSource clCreateProgramWithSource_fn;
static pfn_clBuildProgram clBuildProgram_fn;
static pfn_clGetProgramBuildInfo clGetProgramBuildInfo_fn;
static pfn_clCreateKernel clCreateKernel_fn;
static pfn_clSetKernelArg clSetKernelArg_fn;
static pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel_fn;
static pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer_fn;
static pfn_clEnqueueReadBuffer clEnqueueReadBuffer_fn;
static pfn_clFinish clFinish_fn;
static pfn_clGetEventProfilingInfo clGetEventProfilingInfo_fn;
static pfn_clReleaseEvent clReleaseEvent_fn;
static pfn_clReleaseKernel clReleaseKernel_fn;
static pfn_clReleaseProgram clReleaseProgram_fn;
static pfn_clReleaseMemObject clReleaseMemObject_fn;
static pfn_clReleaseCommandQueue clReleaseCommandQueue_fn;
static pfn_clReleaseContext clReleaseContext_fn;

static const char *KERNEL_SOURCE =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void alu_fma(__global float4 *out, int iters) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    float x = 1.0f + (float)(gid & 1023) * 0.00012207f;\n"
    "    float4 a0 = (float4)(1.001f, 1.003f, 1.005f, 1.007f) * x;\n"
    "    float4 a1 = (float4)(1.011f, 1.013f, 1.017f, 1.019f) * x;\n"
    "    float4 b0 = (float4)(0.997f, 0.991f, 0.983f, 0.977f);\n"
    "    float4 b1 = (float4)(0.971f, 0.967f, 0.953f, 0.947f);\n"
    "    float4 c0 = (float4)(0.101f, 0.103f, 0.107f, 0.109f) + x;\n"
    "    float4 c1 = (float4)(0.113f, 0.127f, 0.131f, 0.137f) + x;\n"
    "    float4 c2 = (float4)(0.139f, 0.149f, 0.151f, 0.157f) + x;\n"
    "    float4 c3 = (float4)(0.163f, 0.167f, 0.173f, 0.179f) + x;\n"
    "    float4 c4 = (float4)(0.181f, 0.191f, 0.193f, 0.197f) + x;\n"
    "    float4 c5 = (float4)(0.199f, 0.211f, 0.223f, 0.227f) + x;\n"
    "    float4 c6 = (float4)(0.229f, 0.233f, 0.239f, 0.241f) + x;\n"
    "    float4 c7 = (float4)(0.251f, 0.257f, 0.263f, 0.269f) + x;\n"
    "    for (int i = 0; i < iters; i++) {\n"
    "        c0 = mad(a0, b0, c0);\n"
    "        c1 = mad(a1, b0, c1);\n"
    "        c2 = mad(a0, b1, c2);\n"
    "        c3 = mad(a1, b1, c3);\n"
    "        c4 = mad(c0, b0, c4);\n"
    "        c5 = mad(c1, b1, c5);\n"
    "        c6 = mad(c2, b0, c6);\n"
    "        c7 = mad(c3, b1, c7);\n"
    "    }\n"
    "    out[gid] = (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7) * 0.125f;\n"
    "}\n"
    /* FP16 analogue of alu_fma: same 8 independent FMA streams, but half8
     * vectors instead of float4. If the Adreno ALU packs fp16 2:1, this should
     * report ~2x the GFLOP/s of alu_fma. counted at 8 mads * 8 lanes * 2. */
    "__kernel void alu_fma_half(__global half8 *out, int iters) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    half x = (half)(1.0f + (float)(gid & 1023) * 0.00012207f);\n"
    "    half8 a0 = (half8)((half)1.001f) * x;\n"
    "    half8 a1 = (half8)((half)1.011f) * x;\n"
    "    half8 b0 = (half8)((half)0.997f);\n"
    "    half8 b1 = (half8)((half)0.971f);\n"
    "    half8 c0 = (half8)((half)0.101f) + x;\n"
    "    half8 c1 = (half8)((half)0.113f) + x;\n"
    "    half8 c2 = (half8)((half)0.139f) + x;\n"
    "    half8 c3 = (half8)((half)0.163f) + x;\n"
    "    half8 c4 = (half8)((half)0.181f) + x;\n"
    "    half8 c5 = (half8)((half)0.199f) + x;\n"
    "    half8 c6 = (half8)((half)0.229f) + x;\n"
    "    half8 c7 = (half8)((half)0.251f) + x;\n"
    "    for (int i = 0; i < iters; i++) {\n"
    "        c0 = mad(a0, b0, c0);\n"
    "        c1 = mad(a1, b0, c1);\n"
    "        c2 = mad(a0, b1, c2);\n"
    "        c3 = mad(a1, b1, c3);\n"
    "        c4 = mad(c0, b0, c4);\n"
    "        c5 = mad(c1, b1, c5);\n"
    "        c6 = mad(c2, b0, c6);\n"
    "        c7 = mad(c3, b1, c7);\n"
    "    }\n"
    "    out[gid] = (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7) * (half8)((half)0.125f);\n"
    "}\n"
    "__kernel void stream_copy(__global const float4 *in,\n"
    "                          __global float4 *out,\n"
    "                          int elems4,\n"
    "                          int passes) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    int stride = (int)get_global_size(0);\n"
    "    float4 bias = (float4)(1.0f, 2.0f, 3.0f, 4.0f);\n"
    "    for (int p = 0; p < passes; p++) {\n"
    "        for (int i = gid; i < elems4; i += stride) {\n"
    "            float4 v = in[i];\n"
    "            out[i] = v + bias;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "__kernel void global_read(__global const float4 *in,\n"
    "                          __global float4 *out,\n"
    "                          int elems4,\n"
    "                          int iters,\n"
    "                          int lanes) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    int stride = (int)get_global_size(0);\n"
    "    int base = gid % elems4;\n"
    "    float4 v0 = (float4)(0.25f);\n"
    "    float4 v1 = (float4)(0.50f);\n"
    "    float4 v2 = (float4)(0.75f);\n"
    "    float4 v3 = (float4)(1.00f);\n"
    "    for (int i = 0; i < iters; i++) {\n"
    "        int j0 = (base + i * stride) % elems4;\n"
    "        if (lanes > 0) v0 += in[j0];\n"
    "        if (lanes > 1) v1 += in[(j0 + stride) % elems4];\n"
    "        if (lanes > 2) v2 += in[(j0 + 2 * stride) % elems4];\n"
    "        if (lanes > 3) v3 += in[(j0 + 3 * stride) % elems4];\n"
    "    }\n"
    "    out[gid] = v0 + v1 + v2 + v3;\n"
    "}\n"
    "__kernel void reg_proxy(__global float4 *out, int iters) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    float x = 1.0f + (float)(gid & 1023) * 0.00012207f;\n"
    "    float4 r0 = (float4)(1.001f, 1.003f, 1.005f, 1.007f) * x;\n"
    "    float4 r1 = (float4)(1.011f, 1.013f, 1.017f, 1.019f) * x;\n"
    "    float4 r2 = (float4)(1.021f, 1.023f, 1.029f, 1.031f) * x;\n"
    "    float4 r3 = (float4)(1.037f, 1.041f, 1.043f, 1.047f) * x;\n"
    "    float4 r4 = (float4)(0.997f, 0.991f, 0.983f, 0.977f);\n"
    "    float4 r5 = (float4)(0.971f, 0.967f, 0.953f, 0.947f);\n"
    "    float4 r6 = (float4)(0.941f, 0.937f, 0.929f, 0.919f);\n"
    "    float4 r7 = (float4)(0.911f, 0.907f, 0.887f, 0.883f);\n"
    "    for (int i = 0; i < iters; i++) {\n"
    "        r0 = r0 + r4;\n"
    "        r1 = r1 + r5;\n"
    "        r2 = r2 + r6;\n"
    "        r3 = r3 + r7;\n"
    "        r4 = r4 + r1;\n"
    "        r5 = r5 + r2;\n"
    "        r6 = r6 + r3;\n"
    "        r7 = r7 + r0;\n"
    "    }\n"
    "    out[gid] = (r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7) * 0.125f;\n"
    "}\n"
    "__kernel void regsweep_1(__global float4 *out, int iters) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    float x = 1.0f + (float)(gid & 1023) * 0.00012207f;\n"
    "    float4 a = (float4)(1.001f, 1.003f, 1.005f, 1.007f) * x;\n"
    "    float4 b = (float4)(0.997f, 0.991f, 0.983f, 0.977f);\n"
    "    float4 c0 = (float4)(0.101f, 0.103f, 0.107f, 0.109f) + x;\n"
    "    for (int i = 0; i < iters; i++) { c0 = mad(a, b, c0); }\n"
    "    out[gid] = c0;\n"
    "}\n"
    "__kernel void regsweep_2(__global float4 *out, int iters) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    float x = 1.0f + (float)(gid & 1023) * 0.00012207f;\n"
    "    float4 a0 = (float4)(1.001f, 1.003f, 1.005f, 1.007f) * x;\n"
    "    float4 a1 = (float4)(1.011f, 1.013f, 1.017f, 1.019f) * x;\n"
    "    float4 b = (float4)(0.997f, 0.991f, 0.983f, 0.977f);\n"
    "    float4 c0 = (float4)(0.101f, 0.103f, 0.107f, 0.109f) + x;\n"
    "    float4 c1 = (float4)(0.113f, 0.127f, 0.131f, 0.137f) + x;\n"
    "    for (int i = 0; i < iters; i++) { c0 = mad(a0, b, c0); c1 = mad(a1, b, c1); }\n"
    "    out[gid] = c0 + c1;\n"
    "}\n"
    "__kernel void regsweep_4(__global float4 *out, int iters) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    float x = 1.0f + (float)(gid & 1023) * 0.00012207f;\n"
    "    float4 a0 = (float4)(1.001f, 1.003f, 1.005f, 1.007f) * x;\n"
    "    float4 a1 = (float4)(1.011f, 1.013f, 1.017f, 1.019f) * x;\n"
    "    float4 b0 = (float4)(0.997f, 0.991f, 0.983f, 0.977f);\n"
    "    float4 b1 = (float4)(0.971f, 0.967f, 0.953f, 0.947f);\n"
    "    float4 c0 = (float4)(0.101f, 0.103f, 0.107f, 0.109f) + x;\n"
    "    float4 c1 = (float4)(0.113f, 0.127f, 0.131f, 0.137f) + x;\n"
    "    float4 c2 = (float4)(0.139f, 0.149f, 0.151f, 0.157f) + x;\n"
    "    float4 c3 = (float4)(0.163f, 0.167f, 0.173f, 0.179f) + x;\n"
    "    for (int i = 0; i < iters; i++) { c0 = mad(a0,b0,c0); c1 = mad(a1,b0,c1); c2 = mad(a0,b1,c2); c3 = mad(a1,b1,c3); }\n"
    "    out[gid] = c0 + c1 + c2 + c3;\n"
    "}\n"
    "__kernel void regsweep_8(__global float4 *out, int iters) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    float x = 1.0f + (float)(gid & 1023) * 0.00012207f;\n"
    "    float4 a0 = (float4)(1.001f, 1.003f, 1.005f, 1.007f) * x;\n"
    "    float4 a1 = (float4)(1.011f, 1.013f, 1.017f, 1.019f) * x;\n"
    "    float4 b0 = (float4)(0.997f, 0.991f, 0.983f, 0.977f);\n"
    "    float4 b1 = (float4)(0.971f, 0.967f, 0.953f, 0.947f);\n"
    "    float4 c0 = (float4)(0.101f, 0.103f, 0.107f, 0.109f) + x;\n"
    "    float4 c1 = (float4)(0.113f, 0.127f, 0.131f, 0.137f) + x;\n"
    "    float4 c2 = (float4)(0.139f, 0.149f, 0.151f, 0.157f) + x;\n"
    "    float4 c3 = (float4)(0.163f, 0.167f, 0.173f, 0.179f) + x;\n"
    "    float4 c4 = (float4)(0.181f, 0.191f, 0.193f, 0.197f) + x;\n"
    "    float4 c5 = (float4)(0.199f, 0.211f, 0.223f, 0.227f) + x;\n"
    "    float4 c6 = (float4)(0.229f, 0.233f, 0.239f, 0.241f) + x;\n"
    "    float4 c7 = (float4)(0.251f, 0.257f, 0.263f, 0.269f) + x;\n"
    "    for (int i = 0; i < iters; i++) { c0=mad(a0,b0,c0); c1=mad(a1,b0,c1); c2=mad(a0,b1,c2); c3=mad(a1,b1,c3); c4=mad(c0,b0,c4); c5=mad(c1,b1,c5); c6=mad(c2,b0,c6); c7=mad(c3,b1,c7); }\n"
    "    out[gid] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;\n"
    "}\n"
    "__kernel void regsweep_12(__global float4 *out, int iters) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    float x = 1.0f + (float)(gid & 1023) * 0.00012207f;\n"
    "    float4 a0=(float4)(1.001f,1.003f,1.005f,1.007f)*x, a1=(float4)(1.011f,1.013f,1.017f,1.019f)*x, a2=(float4)(1.021f,1.023f,1.029f,1.031f)*x;\n"
    "    float4 b0=(float4)(0.997f,0.991f,0.983f,0.977f), b1=(float4)(0.971f,0.967f,0.953f,0.947f), b2=(float4)(0.941f,0.937f,0.929f,0.919f);\n"
    "    float4 c0=(float4)(0.101f)+x,c1=(float4)(0.113f)+x,c2=(float4)(0.139f)+x,c3=(float4)(0.163f)+x,c4=(float4)(0.181f)+x,c5=(float4)(0.199f)+x,c6=(float4)(0.229f)+x,c7=(float4)(0.251f)+x,c8=(float4)(0.271f)+x,c9=(float4)(0.281f)+x,c10=(float4)(0.307f)+x,c11=(float4)(0.311f)+x;\n"
    "    for (int i=0;i<iters;i++){ c0=mad(a0,b0,c0); c1=mad(a1,b0,c1); c2=mad(a2,b0,c2); c3=mad(a0,b1,c3); c4=mad(a1,b1,c4); c5=mad(a2,b1,c5); c6=mad(a0,b2,c6); c7=mad(a1,b2,c7); c8=mad(a2,b2,c8); c9=mad(c0,b0,c9); c10=mad(c1,b1,c10); c11=mad(c2,b2,c11); }\n"
    "    out[gid]=c0+c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11;\n"
    "}\n"
    "__kernel void regsweep_16(__global float4 *out, int iters) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    float x = 1.0f + (float)(gid & 1023) * 0.00012207f;\n"
    "    float4 a0=(float4)(1.001f,1.003f,1.005f,1.007f)*x,a1=(float4)(1.011f,1.013f,1.017f,1.019f)*x,a2=(float4)(1.021f,1.023f,1.029f,1.031f)*x,a3=(float4)(1.037f,1.041f,1.043f,1.047f)*x;\n"
    "    float4 b0=(float4)(0.997f,0.991f,0.983f,0.977f),b1=(float4)(0.971f,0.967f,0.953f,0.947f),b2=(float4)(0.941f,0.937f,0.929f,0.919f),b3=(float4)(0.911f,0.907f,0.887f,0.883f);\n"
    "    float4 c0=(float4)(0.101f)+x,c1=(float4)(0.113f)+x,c2=(float4)(0.139f)+x,c3=(float4)(0.163f)+x,c4=(float4)(0.181f)+x,c5=(float4)(0.199f)+x,c6=(float4)(0.229f)+x,c7=(float4)(0.251f)+x,c8=(float4)(0.271f)+x,c9=(float4)(0.281f)+x,c10=(float4)(0.307f)+x,c11=(float4)(0.311f)+x,c12=(float4)(0.313f)+x,c13=(float4)(0.317f)+x,c14=(float4)(0.331f)+x,c15=(float4)(0.337f)+x;\n"
    "    for (int i=0;i<iters;i++){ c0=mad(a0,b0,c0); c1=mad(a1,b0,c1); c2=mad(a2,b0,c2); c3=mad(a3,b0,c3); c4=mad(a0,b1,c4); c5=mad(a1,b1,c5); c6=mad(a2,b1,c6); c7=mad(a3,b1,c7); c8=mad(a0,b2,c8); c9=mad(a1,b2,c9); c10=mad(a2,b2,c10); c11=mad(a3,b2,c11); c12=mad(a0,b3,c12); c13=mad(a1,b3,c13); c14=mad(a2,b3,c14); c15=mad(a3,b3,c15); }\n"
    "    out[gid]=c0+c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11+c12+c13+c14+c15;\n"
    "}\n"
    "__kernel void local_read(__global float4 *out,\n"
    "                         __local float4 *scratch,\n"
    "                         int local_elems4,\n"
    "                         int iters,\n"
    "                         int lanes) {\n"
    "    int gid = (int)get_global_id(0);\n"
    "    int lid = (int)get_local_id(0);\n"
    "    int lsz = (int)get_local_size(0);\n"
    "    for (int i = lid; i < local_elems4; i += lsz) {\n"
    "        scratch[i] = (float4)((float)i, (float)(i + 1), (float)(i + 2), (float)(i + 3));\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    int idx0 = lid;\n"
    "    int idx1 = lid + lsz;\n"
    "    int idx2 = lid + 2 * lsz;\n"
    "    int idx3 = lid + 3 * lsz;\n"
    "    float4 v0 = (float4)(0.25f);\n"
    "    float4 v1 = (float4)(0.50f);\n"
    "    float4 v2 = (float4)(0.75f);\n"
    "    float4 v3 = (float4)(1.00f);\n"
    "    for (int i = 0; i < iters; i++) {\n"
    "        if (lanes > 0) { v0 += scratch[idx0]; }\n"
    "        if (lanes > 1) { v1 += scratch[idx1]; }\n"
    "        if (lanes > 2) { v2 += scratch[idx2]; }\n"
    "        if (lanes > 3) { v3 += scratch[idx3]; }\n"
    "    }\n"
    "    out[gid] = v0 + v1 + v2 + v3;\n"
    "}\n";

typedef struct {
    float s[4];
} host_float4;

static void fail_result(const char *msg) {
    printf("GPU_SOL_RESULT\n");
    printf("status: FAILED\n");
    printf("error: %s\n", msg);
    printf("GPU_SOL_END\n");
    fflush(stdout);
    exit(1);
}

static void fail_result_fmt(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fail_result(buf);
}

static uint64_t parse_u64(const char *s, uint64_t fallback) {
    char *end = NULL;
    uint64_t v = strtoull(s, &end, 10);
    if (end == s || v == 0) {
        return fallback;
    }
    return v;
}

static size_t round_up_size(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

static void load_opencl(void) {
    void *lib = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!lib) {
        fail_result_fmt("dlopen libOpenCL failed: %s", dlerror());
    }

#define LOAD_CL(name) do { \
    name##_fn = (pfn_##name)dlsym(lib, #name); \
    if (!name##_fn) fail_result_fmt("dlsym %s failed", #name); \
} while (0)

    LOAD_CL(clGetPlatformIDs);
    LOAD_CL(clGetDeviceIDs);
    LOAD_CL(clCreateContext);
    LOAD_CL(clCreateCommandQueueWithProperties);
    LOAD_CL(clCreateBuffer);
    LOAD_CL(clCreateProgramWithSource);
    LOAD_CL(clBuildProgram);
    LOAD_CL(clGetProgramBuildInfo);
    LOAD_CL(clCreateKernel);
    LOAD_CL(clSetKernelArg);
    LOAD_CL(clEnqueueNDRangeKernel);
    LOAD_CL(clEnqueueWriteBuffer);
    LOAD_CL(clEnqueueReadBuffer);
    LOAD_CL(clFinish);
    LOAD_CL(clGetEventProfilingInfo);
    LOAD_CL(clReleaseEvent);
    LOAD_CL(clReleaseKernel);
    LOAD_CL(clReleaseProgram);
    LOAD_CL(clReleaseMemObject);
    LOAD_CL(clReleaseCommandQueue);
    LOAD_CL(clReleaseContext);

#undef LOAD_CL
}

static cl_ulong event_start(cl_event ev) {
    cl_ulong t = 0;
    cl_int rc = clGetEventProfilingInfo_fn(ev, CL_PROFILING_COMMAND_START,
                                           sizeof(t), &t, NULL);
    if (rc != CL_SUCCESS) {
        fail_result_fmt("clGetEventProfilingInfo START failed: rc=%d", rc);
    }
    return t;
}

static cl_ulong event_end(cl_event ev) {
    cl_ulong t = 0;
    cl_int rc = clGetEventProfilingInfo_fn(ev, CL_PROFILING_COMMAND_END,
                                           sizeof(t), &t, NULL);
    if (rc != CL_SUCCESS) {
        fail_result_fmt("clGetEventProfilingInfo END failed: rc=%d", rc);
    }
    return t;
}

int main(int argc, char **argv) {
    const char *mode = "alu";
    uint64_t global_items_arg = 262144;
    uint64_t local_items_arg = 256;
    uint64_t iters_arg = 4096;
    uint64_t reps_arg = 64;
    uint64_t warmup_arg = 4;
    uint64_t bytes_mb_arg = 64;
    uint64_t bytes_kb_arg = 0;
    uint64_t passes_arg = 4;
    uint64_t local_kb_arg = 32;
    uint64_t lanes_arg = 4;
    uint64_t accums_arg = 8;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--global") == 0 && i + 1 < argc) {
            global_items_arg = parse_u64(argv[++i], global_items_arg);
        } else if (strcmp(argv[i], "--local") == 0 && i + 1 < argc) {
            local_items_arg = parse_u64(argv[++i], local_items_arg);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters_arg = parse_u64(argv[++i], iters_arg);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps_arg = parse_u64(argv[++i], reps_arg);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_arg = parse_u64(argv[++i], warmup_arg);
        } else if (strcmp(argv[i], "--bytes-mb") == 0 && i + 1 < argc) {
            bytes_mb_arg = parse_u64(argv[++i], bytes_mb_arg);
        } else if (strcmp(argv[i], "--bytes-kb") == 0 && i + 1 < argc) {
            bytes_kb_arg = parse_u64(argv[++i], bytes_kb_arg);
        } else if (strcmp(argv[i], "--passes") == 0 && i + 1 < argc) {
            passes_arg = parse_u64(argv[++i], passes_arg);
        } else if (strcmp(argv[i], "--local-kb") == 0 && i + 1 < argc) {
            local_kb_arg = parse_u64(argv[++i], local_kb_arg);
        } else if (strcmp(argv[i], "--lanes") == 0 && i + 1 < argc) {
            lanes_arg = parse_u64(argv[++i], lanes_arg);
        } else if (strcmp(argv[i], "--accums") == 0 && i + 1 < argc) {
            accums_arg = parse_u64(argv[++i], accums_arg);
        } else {
            fail_result_fmt("unknown or incomplete argument: %s", argv[i]);
        }
    }

    if (strcmp(mode, "alu") != 0 && strcmp(mode, "alu_half") != 0 &&
        strcmp(mode, "reg") != 0 && strcmp(mode, "regsweep") != 0 &&
        strcmp(mode, "stream") != 0 && strcmp(mode, "global") != 0 &&
        strcmp(mode, "local") != 0) {
        fail_result("mode must be alu, alu_half, reg, regsweep, stream, global, or local");
    }
    if (local_items_arg == 0 || local_items_arg > 1024) {
        fail_result("local size must be in 1..1024");
    }
    if (reps_arg == 0 || reps_arg > 100000) {
        fail_result("reps must be in 1..100000");
    }

    size_t local_items = (size_t)local_items_arg;
    size_t global_items = round_up_size((size_t)global_items_arg, local_items);
    int iters = (int)iters_arg;
    int passes = (int)passes_arg;
    int reps = (int)reps_arg;
    int warmup = (int)warmup_arg;
    int accums = (int)accums_arg;
    if (!(accums == 1 || accums == 2 || accums == 4 || accums == 8 || accums == 12 || accums == 16)) {
        fail_result("accums must be one of 1,2,4,8,12,16");
    }
    int lanes = (int)lanes_arg;
    if (lanes < 1) lanes = 1;
    if (lanes > 4) lanes = 4;
    size_t local_bytes = (size_t)local_kb_arg * 1024u;
    int local_elems4 = (int)(local_bytes / sizeof(host_float4));
    int local_lanes = (int)(local_elems4 / (int)local_items);
    if (local_lanes > lanes) local_lanes = lanes;
    if (local_lanes > 4) local_lanes = 4;
    if (local_lanes < 1) local_lanes = 1;
    if (local_bytes == 0 || local_bytes > 32768u) {
        fail_result("local-kb must be in 1..32 for this Adreno OpenCL device");
    }
    if (local_elems4 < (int)local_items) {
        fail_result("local-kb is too small for local workgroup size");
    }

    load_opencl();

    cl_int rc = CL_SUCCESS;
    cl_platform_id platform = NULL;
    cl_uint platform_count = 0;
    rc = clGetPlatformIDs_fn(1, &platform, &platform_count);
    if (rc != CL_SUCCESS || !platform || platform_count == 0) {
        fail_result_fmt("clGetPlatformIDs failed: rc=%d count=%u", rc, platform_count);
    }

    cl_device_id device = NULL;
    cl_uint device_count = 0;
    rc = clGetDeviceIDs_fn(platform, CL_DEVICE_TYPE_GPU, 1, &device, &device_count);
    if (rc != CL_SUCCESS || !device || device_count == 0) {
        fail_result_fmt("clGetDeviceIDs GPU failed: rc=%d count=%u", rc, device_count);
    }

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
    };
    cl_context ctx = clCreateContext_fn(props, 1, &device, NULL, NULL, &rc);
    if (rc != CL_SUCCESS || !ctx) {
        fail_result_fmt("clCreateContext failed: rc=%d", rc);
    }

    cl_queue_properties qprops[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties_fn(ctx, device, qprops, &rc);
    if (rc != CL_SUCCESS || !queue) {
        clReleaseContext_fn(ctx);
        fail_result_fmt("clCreateCommandQueueWithProperties failed: rc=%d", rc);
    }

    cl_mem buf_in = NULL;
    cl_mem buf_out = NULL;
    size_t stream_bytes = (bytes_kb_arg > 0)
        ? (size_t)bytes_kb_arg * 1024u
        : (size_t)bytes_mb_arg * 1024u * 1024u;
    int elems4 = (int)(stream_bytes / sizeof(host_float4));
    if (elems4 <= 0) {
        clReleaseCommandQueue_fn(queue);
        clReleaseContext_fn(ctx);
        fail_result("stream byte size is too small");
    }

    if (strcmp(mode, "alu") == 0 || strcmp(mode, "alu_half") == 0 ||
        strcmp(mode, "reg") == 0 ||
        strcmp(mode, "regsweep") == 0 || strcmp(mode, "local") == 0) {
        size_t out_bytes = global_items * sizeof(host_float4);
        buf_out = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, out_bytes, NULL, &rc);
        if (rc != CL_SUCCESS || !buf_out) {
            clReleaseCommandQueue_fn(queue);
            clReleaseContext_fn(ctx);
            fail_result_fmt("clCreateBuffer alu output failed: rc=%d", rc);
        }
    } else {
        stream_bytes = (size_t)elems4 * sizeof(host_float4);
        host_float4 *host_in = (host_float4 *)malloc(stream_bytes);
        if (!host_in) {
            clReleaseCommandQueue_fn(queue);
            clReleaseContext_fn(ctx);
            fail_result("malloc stream input failed");
        }
        for (int i = 0; i < elems4; i++) {
            host_in[i].s[0] = (float)(i & 255);
            host_in[i].s[1] = (float)((i + 1) & 255);
            host_in[i].s[2] = (float)((i + 2) & 255);
            host_in[i].s[3] = (float)((i + 3) & 255);
        }
        buf_in = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, stream_bytes, NULL, &rc);
        if (rc != CL_SUCCESS || !buf_in) {
            free(host_in);
            clReleaseCommandQueue_fn(queue);
            clReleaseContext_fn(ctx);
            fail_result_fmt("clCreateBuffer stream input failed: rc=%d", rc);
        }
        buf_out = clCreateBuffer_fn(ctx, CL_MEM_READ_WRITE, stream_bytes, NULL, &rc);
        if (rc != CL_SUCCESS || !buf_out) {
            free(host_in);
            clReleaseMemObject_fn(buf_in);
            clReleaseCommandQueue_fn(queue);
            clReleaseContext_fn(ctx);
            fail_result_fmt("clCreateBuffer stream output failed: rc=%d", rc);
        }
        rc = clEnqueueWriteBuffer_fn(queue, buf_in, CL_TRUE, 0, stream_bytes,
                                     host_in, 0, NULL, NULL);
        free(host_in);
        if (rc != CL_SUCCESS) {
            clReleaseMemObject_fn(buf_in);
            clReleaseMemObject_fn(buf_out);
            clReleaseCommandQueue_fn(queue);
            clReleaseContext_fn(ctx);
            fail_result_fmt("clEnqueueWriteBuffer stream input failed: rc=%d", rc);
        }
    }

    cl_program program = clCreateProgramWithSource_fn(ctx, 1, &KERNEL_SOURCE, NULL, &rc);
    if (rc != CL_SUCCESS || !program) {
        fail_result_fmt("clCreateProgramWithSource failed: rc=%d", rc);
    }
    rc = clBuildProgram_fn(program, 1, &device, "-cl-fast-relaxed-math -cl-std=CL3.0",
                           NULL, NULL);
    if (rc != CL_SUCCESS) {
        char log[4096];
        memset(log, 0, sizeof(log));
        clGetProgramBuildInfo_fn(program, device, CL_PROGRAM_BUILD_LOG,
                                 sizeof(log) - 1, log, NULL);
        fail_result_fmt("clBuildProgram failed: rc=%d log=%.240s", rc, log);
    }

    const char *kernel_name = "stream_copy";
    if (strcmp(mode, "alu") == 0) {
        kernel_name = "alu_fma";
    } else if (strcmp(mode, "alu_half") == 0) {
        kernel_name = "alu_fma_half";
    } else if (strcmp(mode, "reg") == 0) {
        kernel_name = "reg_proxy";
    } else if (strcmp(mode, "regsweep") == 0) {
        if (accums == 1) kernel_name = "regsweep_1";
        else if (accums == 2) kernel_name = "regsweep_2";
        else if (accums == 4) kernel_name = "regsweep_4";
        else if (accums == 8) kernel_name = "regsweep_8";
        else if (accums == 12) kernel_name = "regsweep_12";
        else kernel_name = "regsweep_16";
    } else if (strcmp(mode, "global") == 0) {
        kernel_name = "global_read";
    } else if (strcmp(mode, "local") == 0) {
        kernel_name = "local_read";
    }
    cl_kernel kernel = clCreateKernel_fn(program, kernel_name, &rc);
    if (rc != CL_SUCCESS || !kernel) {
        fail_result_fmt("clCreateKernel %s failed: rc=%d", kernel_name, rc);
    }

    if (strcmp(mode, "alu") == 0 || strcmp(mode, "alu_half") == 0 ||
        strcmp(mode, "reg") == 0 || strcmp(mode, "regsweep") == 0) {
        rc = clSetKernelArg_fn(kernel, 0, sizeof(cl_mem), &buf_out);
        rc |= clSetKernelArg_fn(kernel, 1, sizeof(cl_int), &iters);
    } else if (strcmp(mode, "local") == 0) {
        rc = clSetKernelArg_fn(kernel, 0, sizeof(cl_mem), &buf_out);
        rc |= clSetKernelArg_fn(kernel, 1, local_bytes, NULL);
        rc |= clSetKernelArg_fn(kernel, 2, sizeof(cl_int), &local_elems4);
        rc |= clSetKernelArg_fn(kernel, 3, sizeof(cl_int), &iters);
        rc |= clSetKernelArg_fn(kernel, 4, sizeof(cl_int), &local_lanes);
    } else if (strcmp(mode, "global") == 0) {
        rc = clSetKernelArg_fn(kernel, 0, sizeof(cl_mem), &buf_in);
        rc |= clSetKernelArg_fn(kernel, 1, sizeof(cl_mem), &buf_out);
        rc |= clSetKernelArg_fn(kernel, 2, sizeof(cl_int), &elems4);
        rc |= clSetKernelArg_fn(kernel, 3, sizeof(cl_int), &iters);
        rc |= clSetKernelArg_fn(kernel, 4, sizeof(cl_int), &lanes);
    } else {
        rc = clSetKernelArg_fn(kernel, 0, sizeof(cl_mem), &buf_in);
        rc |= clSetKernelArg_fn(kernel, 1, sizeof(cl_mem), &buf_out);
        rc |= clSetKernelArg_fn(kernel, 2, sizeof(cl_int), &elems4);
        rc |= clSetKernelArg_fn(kernel, 3, sizeof(cl_int), &passes);
    }
    if (rc != CL_SUCCESS) {
        fail_result_fmt("clSetKernelArg failed: rc=%d", rc);
    }

    size_t global[1] = { global_items };
    size_t local[1] = { local_items };

    for (int i = 0; i < warmup; i++) {
        cl_event ev = NULL;
        rc = clEnqueueNDRangeKernel_fn(queue, kernel, 1, NULL, global, local, 0, NULL, &ev);
        if (rc != CL_SUCCESS) {
            fail_result_fmt("warmup clEnqueueNDRangeKernel failed: rc=%d", rc);
        }
        rc = clFinish_fn(queue);
        if (rc != CL_SUCCESS) {
            fail_result_fmt("warmup clFinish failed: rc=%d", rc);
        }
        clReleaseEvent_fn(ev);
    }

    cl_event *events = (cl_event *)calloc((size_t)reps, sizeof(cl_event));
    if (!events) {
        fail_result("calloc events failed");
    }

    for (int i = 0; i < reps; i++) {
        rc = clEnqueueNDRangeKernel_fn(queue, kernel, 1, NULL, global, local, 0, NULL, &events[i]);
        if (rc != CL_SUCCESS) {
            fail_result_fmt("timed clEnqueueNDRangeKernel failed at rep %d: rc=%d", i, rc);
        }
    }
    rc = clFinish_fn(queue);
    if (rc != CL_SUCCESS) {
        fail_result_fmt("timed clFinish failed: rc=%d", rc);
    }

    cl_ulong first_start = event_start(events[0]);
    cl_ulong last_end = event_end(events[reps - 1]);
    cl_ulong kernel_sum_ns = 0;
    cl_ulong min_ns = UINT64_MAX;
    cl_ulong max_ns = 0;
    for (int i = 0; i < reps; i++) {
        cl_ulong s = event_start(events[i]);
        cl_ulong e = event_end(events[i]);
        cl_ulong d = e - s;
        kernel_sum_ns += d;
        if (d < min_ns) min_ns = d;
        if (d > max_ns) max_ns = d;
        clReleaseEvent_fn(events[i]);
    }
    free(events);

    host_float4 check;
    memset(&check, 0, sizeof(check));
    rc = clEnqueueReadBuffer_fn(queue, buf_out, CL_TRUE, 0, sizeof(check),
                                &check, 0, NULL, NULL);
    if (rc != CL_SUCCESS) {
        fail_result_fmt("clEnqueueReadBuffer checksum failed: rc=%d", rc);
    }
    double checksum = (double)check.s[0] + (double)check.s[1] +
                      (double)check.s[2] + (double)check.s[3];

    const double window_ns = (double)(last_end - first_start);
    const double sum_ns = (double)kernel_sum_ns;
    double gflops = 0.0;
    double bandwidth_gbps = 0.0;
    double counted_flops = 0.0;
    double counted_bytes = 0.0;

    if (strcmp(mode, "alu") == 0) {
        const double flops_per_iter_per_work_item = 64.0;
        counted_flops = (double)global_items * (double)iters *
                        flops_per_iter_per_work_item * (double)reps;
        gflops = counted_flops / sum_ns;
    } else if (strcmp(mode, "alu_half") == 0) {
        /* 8 half8 FMAs per iter = 8 mads * 8 lanes * 2 flops = 128 fp16-flops. */
        const double flops_per_iter_per_work_item = 128.0;
        counted_flops = (double)global_items * (double)iters *
                        flops_per_iter_per_work_item * (double)reps;
        gflops = counted_flops / sum_ns;
    } else if (strcmp(mode, "reg") == 0) {
        const double flops_per_iter_per_work_item = 32.0;
        const double proxy_bytes_per_iter_per_work_item = 8.0 * 3.0 * (double)sizeof(host_float4);
        counted_flops = (double)global_items * (double)iters *
                        flops_per_iter_per_work_item * (double)reps;
        counted_bytes = (double)global_items * (double)iters *
                        proxy_bytes_per_iter_per_work_item * (double)reps;
        gflops = counted_flops / sum_ns;
        bandwidth_gbps = counted_bytes / sum_ns;
    } else if (strcmp(mode, "regsweep") == 0) {
        const double flops_per_iter_per_work_item = (double)accums * 8.0;
        counted_flops = (double)global_items * (double)iters *
                        flops_per_iter_per_work_item * (double)reps;
        gflops = counted_flops / sum_ns;
    } else if (strcmp(mode, "stream") == 0) {
        counted_bytes = (double)stream_bytes * 2.0 * (double)passes * (double)reps;
        bandwidth_gbps = counted_bytes / sum_ns;
    } else if (strcmp(mode, "global") == 0) {
        counted_bytes = (double)global_items * (double)iters *
                        (double)lanes * (double)sizeof(host_float4) *
                        (double)reps;
        bandwidth_gbps = counted_bytes / sum_ns;
    } else {
        counted_bytes = (double)global_items * (double)iters *
                        (double)local_lanes * (double)sizeof(host_float4) *
                        (double)reps;
        bandwidth_gbps = counted_bytes / sum_ns;
    }

    printf("GPU_SOL_RESULT\n");
    printf("status: OK\n");
    printf("mode: %s\n", mode);
    printf("cl_start_ns: %llu\n", (unsigned long long)first_start);
    printf("cl_end_ns: %llu\n", (unsigned long long)last_end);
    printf("kernel_time_ns_sum: %llu\n", (unsigned long long)kernel_sum_ns);
    printf("kernel_window_ns: %.0f\n", window_ns);
    printf("kernel_min_ns: %llu\n", (unsigned long long)min_ns);
    printf("kernel_max_ns: %llu\n", (unsigned long long)max_ns);
    printf("reps: %d\n", reps);
    printf("warmup: %d\n", warmup);
    printf("global_items: %zu\n", global_items);
    printf("local_items: %zu\n", local_items);
    printf("iters: %d\n", iters);
    printf("passes: %d\n", passes);
    printf("lanes: %d\n", lanes);
    printf("accums: %d\n", accums);
    printf("local_kb: %zu\n", local_bytes / 1024u);
    printf("local_elems4: %d\n", local_elems4);
    printf("local_lanes: %d\n", local_lanes);
    printf("stream_bytes_per_buffer: %zu\n", stream_bytes);
    printf("counted_flops: %.0f\n", counted_flops);
    printf("counted_bytes: %.0f\n", counted_bytes);
    printf("gflops: %.3f\n", gflops);
    printf("bandwidth_gbps: %.3f\n", bandwidth_gbps);
    printf("checksum: %.6f\n", checksum);
    printf("GPU_SOL_END\n");
    fflush(stdout);

    clReleaseKernel_fn(kernel);
    clReleaseProgram_fn(program);
    if (buf_in) clReleaseMemObject_fn(buf_in);
    if (buf_out) clReleaseMemObject_fn(buf_out);
    clReleaseCommandQueue_fn(queue);
    clReleaseContext_fn(ctx);

    return 0;
}
