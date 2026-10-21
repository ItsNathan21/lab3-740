// dsp_matmul_imp.c — scalar FP32 matmul, runs on cDSP v79.
// Phase 6 Step 4: baseline (no HVX, no VTCM) to establish DSP compute cost.

#include "dsp_matmul.h"
#include "HAP_farf.h"
#include "HAP_perf.h"

int dsp_matmul_open(const char* uri, remote_handle64* h) {
    *h = 1;
    return 0;
}
int dsp_matmul_close(remote_handle64 h) { return 0; }

int dsp_matmul_matmul_fp32(remote_handle64 h,
                            const float* a_data, int a_dataLen,
                            const float* b_data, int b_dataLen,
                            float* c_data, int c_dataLen,
                            int n,
                            int* dsp_us) {
    uint64_t t0 = HAP_perf_get_time_us();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            const float* a_row = a_data + i * n;
            for (int k = 0; k < n; k++) {
                acc += a_row[k] * b_data[k * n + j];
            }
            c_data[i * n + j] = acc;
        }
    }

    *dsp_us = (int)(HAP_perf_get_time_us() - t0);
    FARF(RUNTIME_HIGH, "dsp_matmul scalar n=%d dsp_us=%d", n, *dsp_us);
    return 0;
}
