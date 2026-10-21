// dsp_reduction_imp.c — scalar FP32 sum-reduction, runs on cDSP v79.
// sum = sum(x_i). Baseline (no HVX) to establish DSP compute cost for the
// read-only memory-bound roofline anchor. dsp_us is compute-only (excludes
// FastRPC). Like the CPU scalar baseline, this is a single dependent add chain
// — latency-bound, the foil for the HVX variant.

#include "dsp_reduction.h"
#include "HAP_farf.h"
#include "HAP_perf.h"

int dsp_reduction_open(const char* uri, remote_handle64* h) {
    *h = 1;
    return 0;
}
int dsp_reduction_close(remote_handle64 h) { return 0; }

int dsp_reduction_reduce_fp32(remote_handle64 h,
                              const float* x, int xLen,
                              float* sum,
                              int n,
                              int* dsp_us) {
    uint64_t t0 = HAP_perf_get_time_us();

    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += x[i];
    }
    *sum = s;

    *dsp_us = (int)(HAP_perf_get_time_us() - t0);
    FARF(RUNTIME_HIGH, "dsp_reduction scalar n=%d dsp_us=%d", n, *dsp_us);
    return 0;
}
