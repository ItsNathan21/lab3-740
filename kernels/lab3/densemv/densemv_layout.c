/* Task 9.8: compare dense GPU matvec with different physical row strides.
 * From the lab container's workspace:
 *   bash tools/android-clang -std=c11 -O2 -Wall -Wextra \
 *     kernels/lab3/densemv/densemv_layout.c -ldl -lm -o /tmp/densemv_layout
 *   adb push /tmp/densemv_layout /data/local/tmp/densemv_layout
 *   adb shell chmod 755 /data/local/tmp/densemv_layout
 *   adb shell /data/local/tmp/densemv_layout
 * The optional positional argument is the number of timed runs per layout.
 * The output reports both the mean and the coefficient of variation.
 */

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* OpenCL is loaded on the board. The Android NDK has no OpenCL headers. */
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_event *cl_event;

enum {
    CL_DEVICE_TYPE_GPU = 1 << 2,
    CL_QUEUE_PROFILING_ENABLE = 1 << 1,
    CL_MEM_WRITE_ONLY = 1 << 1,
    CL_MEM_READ_ONLY = 1 << 2,
    CL_MEM_COPY_HOST_PTR = 1 << 5,
    CL_PROGRAM_BUILD_LOG = 0x1183,
    CL_PROFILING_COMMAND_START = 0x1282,
    CL_PROFILING_COMMAND_END = 0x1283,
};

static cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
static cl_int (*clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id *, cl_uint *);
static cl_context (*clCreateContext)(const intptr_t *, cl_uint, const cl_device_id *,
    void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
static cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int *);
static cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char **,
    const size_t *, cl_int *);
static cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *,
    void (*)(cl_program, void *), void *);
static cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
static cl_kernel (*clCreateKernel)(cl_program, const char *, cl_int *);
static cl_mem (*clCreateBuffer)(cl_context, cl_ulong, size_t, void *, cl_int *);
static cl_int (*clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
static cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint,
    const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
static cl_int (*clWaitForEvents)(cl_uint, const cl_event *);
static cl_int (*clGetEventProfilingInfo)(cl_event, cl_uint, size_t, void *, size_t *);
static cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t,
    void *, cl_uint, const cl_event *, cl_event *);
static cl_int (*clReleaseEvent)(cl_event);
static cl_int (*clReleaseMemObject)(cl_mem);
static cl_int (*clReleaseKernel)(cl_kernel);
static cl_int (*clReleaseProgram)(cl_program);
static cl_int (*clReleaseCommandQueue)(cl_command_queue);
static cl_int (*clReleaseContext)(cl_context);

static const char *kernel_source =
    "__kernel void densemv(__global const float *A, __global const float *x,\n"
    "                      __global float *y, int rows, int cols, int stride) {\n"
    "    int row = get_global_id(0);\n"
    "    if (row >= rows) return;\n"
    "    __global const float *arow = A + (long)row * stride;\n"
    "    float sum = 0.0f;\n"
    "    for (int col = 0; col < cols; ++col) sum += arow[col] * x[col];\n"
    "    y[row] = sum;\n"
    "}\n";

static void die(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void check(cl_int error, const char *operation) {
    if (error == 0) return;
    fprintf(stderr, "%s failed (%d)\n", operation, error);
    exit(EXIT_FAILURE);
}

#define CHECK(call) check((call), #call)
#define LOAD(name) do { name = dlsym(opencl, #name); if (!name) die(dlerror()); } while (0)
#define SET_ARG(index, value) CHECK(clSetKernelArg(kernel, index, sizeof(value), &(value)))

static double measure(cl_context context, cl_command_queue queue, cl_kernel kernel,
                      cl_int size, cl_int padding, int reps, double baseline) {
    const cl_int stride = size + padding;
    const size_t matrix_bytes = (size_t)size * stride * sizeof(float);
    const size_t vector_bytes = (size_t)size * sizeof(float);
    float *matrix = malloc(matrix_bytes);
    float *input = malloc(vector_bytes);
    float *output = malloc(vector_bytes);
    float *expected = malloc(vector_bytes);
    if (!matrix || !input || !output || !expected) die("host allocation failed");

    // Small integers make all intermediate FP32 sums exact across layouts.
    for (int col = 0; col < size; ++col) input[col] = (float)(col % 13 - 6);
    for (int row = 0; row < size; ++row) {
        float sum = 0.0f;
        float *data = matrix + (size_t)row * stride;
        for (int col = 0; col < size; ++col) {
            data[col] = (float)((row * 13 + col * 5 + (row ^ col)) % 31 - 15);
            sum += data[col] * input[col];
        }
        for (int col = size; col < stride; ++col) data[col] = 1000.0f;
        expected[row] = sum;
    }

    cl_int error;
    cl_mem a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              matrix_bytes, matrix, &error);
    check(error, "clCreateBuffer(A)");
    cl_mem x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              vector_bytes, input, &error);
    check(error, "clCreateBuffer(x)");
    cl_mem y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vector_bytes, NULL, &error);
    check(error, "clCreateBuffer(y)");
    free(matrix);
    free(input);

    SET_ARG(0, a);
    SET_ARG(1, x);
    SET_ARG(2, y);
    SET_ARG(3, size);
    SET_ARG(4, size);
    SET_ARG(5, stride);

    const size_t local = 64;
    const size_t global = ((size_t)size + local - 1) / local * local;
    double mean = 0.0;
    double sum_sq = 0.0;
    for (int rep = 0; rep <= reps; ++rep) {
        cl_event event;
        CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &event));
        CHECK(clWaitForEvents(1, &event));
        if (rep > 0) {
            cl_ulong start, end;
            CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL));
            CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL));
            if (end <= start) die("OpenCL returned an invalid event duration");
            const double sample = (double)(end - start) / 1000.0;
            const double delta = sample - mean;
            mean += delta / rep;
            sum_sq += delta * (sample - mean);
        }
        CHECK(clReleaseEvent(event));
    }

    CHECK(clEnqueueReadBuffer(queue, y, 1, 0, vector_bytes, output, 0, NULL, NULL));
    for (int row = 0; row < size; ++row) {
        if (output[row] != expected[row]) {
            fprintf(stderr, "output mismatch at row %d: got %g, expected %g\n",
                    row, output[row], expected[row]);
            exit(EXIT_FAILURE);
        }
    }
    free(output);
    free(expected);
    CHECK(clReleaseMemObject(y));
    CHECK(clReleaseMemObject(x));
    CHECK(clReleaseMemObject(a));

    const double cov = 100.0 * sqrt(sum_sq / (reps - 1)) / mean;
    const double ratio = baseline > 0.0 ? mean / baseline : 1.0;
    printf("%d,%d,%d,%.3f,%.3f,%.3f,%s\n", size, stride, reps, mean, cov, ratio,
           size == 4096 && ratio <= 1.5 ? "yes" : "no");
    fflush(stdout);
    return mean;
}

int main(int argc, char **argv) {
    int reps = 12;
    if (argc == 2) {
        char *end;
        long count = strtol(argv[1], &end, 10);
        if (*end || count < 2 || count > 10000) die("reps must be between 2 and 10000");
        reps = (int)count;
    } else if (argc != 1) {
        die("usage: densemv_layout [reps]");
    }

    void *opencl = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!opencl) die(dlerror());
    LOAD(clGetPlatformIDs);
    LOAD(clGetDeviceIDs);
    LOAD(clCreateContext);
    LOAD(clCreateCommandQueue);
    LOAD(clCreateProgramWithSource);
    LOAD(clBuildProgram);
    LOAD(clGetProgramBuildInfo);
    LOAD(clCreateKernel);
    LOAD(clCreateBuffer);
    LOAD(clSetKernelArg);
    LOAD(clEnqueueNDRangeKernel);
    LOAD(clWaitForEvents);
    LOAD(clGetEventProfilingInfo);
    LOAD(clEnqueueReadBuffer);
    LOAD(clReleaseEvent);
    LOAD(clReleaseMemObject);
    LOAD(clReleaseKernel);
    LOAD(clReleaseProgram);
    LOAD(clReleaseCommandQueue);
    LOAD(clReleaseContext);

    cl_platform_id platform;
    cl_device_id device;
    cl_int error;
    CHECK(clGetPlatformIDs(1, &platform, NULL));
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    check(error, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
    check(error, "clCreateCommandQueue");
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &error);
    check(error, "clCreateProgramWithSource");
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (error != 0) {
        char log[4096] = {0};
        if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL) == 0)
            fprintf(stderr, "%s\n", log);
        check(error, "clBuildProgram");
    }
    cl_kernel kernel = clCreateKernel(program, "densemv", &error);
    check(error, "clCreateKernel");

    puts("logical_size,row_stride_floats,reps,mean_us,cov_pct,ratio_vs_4000,meets_1_5x");
    const double baseline = measure(context, queue, kernel, 4000, 0, reps, 0.0);
    const cl_int padding[] = {0, 8, 16, 32, 64, 96, 128};
    for (size_t i = 0; i < sizeof(padding) / sizeof(padding[0]); ++i)
        measure(context, queue, kernel, 4096, padding[i], reps, baseline);

    CHECK(clReleaseKernel(kernel));
    CHECK(clReleaseProgram(program));
    CHECK(clReleaseCommandQueue(queue));
    CHECK(clReleaseContext(context));
    dlclose(opencl);
    return 0;
}
