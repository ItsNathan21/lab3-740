// dsp_conv_stage_main.c — ARM64 driver for VTCM block staging (Lab 3 Task 11).
//
// Argv: --mb --p --impl ddr|naive|staged --iterations   (defaults 32/1/ddr/1)
//
// Output:
//   stage-<impl>-start mb=.. P=..
//   stage-<impl>-end checksum=<int> wall_us=<int> dsp_us=<int> copy_us=<int> \
//                    compute_us=<int> copy_gbps=<%.2f> iterations=..
//
// checksum must be identical across all three implementations. copy_gbps is
// derived from copy_us and the bytes actually moved, so Task 11.6 can compare
// it directly against the VTCM and DDR bandwidths measured in Task 11.2/11.4.

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "dsp_conv.h"

#define CDSP_DOMAIN   "&_dom=cdsp"
#define DSP_CONV_URI  dsp_conv_URI CDSP_DOMAIN "&_unsigned_module=1"

typedef int (*fn_open)(const char*, remote_handle64*);
typedef int (*fn_close)(remote_handle64);
typedef int (*fn_stage)(remote_handle64, int, int, int, int*, int*, int*, int*);

static uint64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, 0);
    return (uint64_t)tv.tv_sec * 1000000ull + tv.tv_usec;
}

int main(int argc, char** argv) {
    int mb = 32, P = 1, iterations = 1;
    const char* impl_s = "ddr";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--mb") && i+1 < argc) mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--p")  && i+1 < argc) P = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--impl") && i+1 < argc) impl_s = argv[++i];
        else if (!strcmp(argv[i], "--iterations") && i+1 < argc) iterations = atoi(argv[++i]);
    }

    int impl = 0;
    if (!strcmp(impl_s, "naive"))  impl = 1;
    else if (!strcmp(impl_s, "staged")) impl = 2;
    else if (strcmp(impl_s, "ddr")) {
        fprintf(stderr, "--impl must be ddr, naive or staged\n"); return 1;
    }

    void* cdsprpc = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!cdsprpc) { fprintf(stderr, "libcdsprpc.so: %s\n", dlerror()); return 1; }
    void* stub = dlopen("./libdsp_conv_stub.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!stub) { fprintf(stderr, "stub: %s\n", dlerror()); return 1; }

    fn_open  my_open  = (fn_open)dlsym(stub, "dsp_conv_open");
    fn_close my_close = (fn_close)dlsym(stub, "dsp_conv_close");
    fn_stage my_stage = (fn_stage)dlsym(stub, "dsp_conv_stage_calc");
    if (!my_open || !my_close || !my_stage) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror()); return 1;
    }

    printf("stage-%s-start mb=%d P=%d\n", impl_s, mb, P);
    fflush(stdout);

    remote_handle64 h = 0;
    if (my_open(DSP_CONV_URI, &h) != 0) { fprintf(stderr, "open failed\n"); return 1; }

    int dsp_us = 0, copy_us = 0, compute_us = 0, checksum = 0;
    uint64_t t0 = now_us();
    for (int it = 0; it < iterations; it++) {
        int rc = my_stage(h, mb, P, impl, &dsp_us, &copy_us, &compute_us, &checksum);
        if (rc != 0) {
            fprintf(stderr, "stage_calc failed rc=%d%s\n", rc,
                    rc == -2 ? " (VTCM unavailable)" : "");
            my_close(h); return 1;
        }
    }
    uint64_t wall_us = (now_us() - t0) / (uint64_t)(iterations > 0 ? iterations : 1);
    my_close(h);

    double copy_gbps = 0.0;
    if (copy_us > 0)
        copy_gbps = ((double)mb * 1024.0 * 1024.0) / ((double)copy_us * 1e-6) / 1e9;

    printf("stage-%s-end checksum=%d wall_us=%llu dsp_us=%d copy_us=%d "
           "compute_us=%d copy_gbps=%.2f iterations=%d mb=%d P=%d\n",
           impl_s, checksum, (unsigned long long)wall_us, dsp_us, copy_us,
           compute_us, copy_gbps, iterations, mb, P);

    dlclose(stub); dlclose(cdsprpc);
    return 0;
}
