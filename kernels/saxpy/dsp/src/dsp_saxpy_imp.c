// dsp_saxpy_imp.c — scalar FP32 SAXPY, runs on cDSP v79.
// out = alpha*x + y. Baseline (no HVX) to establish DSP compute cost for the
// memory-bound roofline anchor. dsp_us is compute-only (excludes FastRPC).

#include "dsp_saxpy.h"
#include "HAP_farf.h"
#include "HAP_perf.h"

int dsp_saxpy_open(const char* uri, remote_handle64* h) {
    *h = 1;
    return 0;
}
int dsp_saxpy_close(remote_handle64 h) { return 0; }

int dsp_saxpy_saxpy_fp32(remote_handle64 h,
                         const float* x, int xLen,
                         const float* y, int yLen,
                         float alpha,
                         float* out, int outLen,
                         int n,
                         int* dsp_us) {
    uint64_t t0 = HAP_perf_get_time_us();

    for (int i = 0; i < n; i++) {
        out[i] = alpha * x[i] + y[i];
    }

    *dsp_us = (int)(HAP_perf_get_time_us() - t0);
    FARF(RUNTIME_HIGH, "dsp_saxpy scalar n=%d dsp_us=%d", n, *dsp_us);
    return 0;
}
