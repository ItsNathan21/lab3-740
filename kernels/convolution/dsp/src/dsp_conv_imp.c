// dsp_conv_imp.c — scalar FP32 direct 2D convolution, cDSP v79.
// out = conv2d(in, weights), NHWC in/out, OHWI weights, stride 1, same zero pad.
// Baseline (no HVX). dsp_us is compute-only (excludes FastRPC).

#include "dsp_conv.h"
#include "HAP_farf.h"
#include "HAP_perf.h"

int dsp_conv_open(const char* uri, remote_handle64* h) { *h = 1; return 0; }
int dsp_conv_close(remote_handle64 h) { return 0; }

int dsp_conv_conv_fp32(remote_handle64 h,
                       const float* in, int inLen,
                       const float* w, int wLen,
                       float* out, int outLen,
                       int H, int W, int Cin, int Cout, int K,
                       int* dsp_us) {
    uint64_t t0 = HAP_perf_get_time_us();
    int pad = K / 2;
    for (int oh = 0; oh < H; oh++) {
        for (int ow = 0; ow < W; ow++) {
            float* out_px = &out[((long)oh * W + ow) * Cout];
            for (int oc = 0; oc < Cout; oc++) {
                float acc = 0.0f;
                for (int kh = 0; kh < K; kh++) {
                    int ih = oh + kh - pad;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow + kw - pad;
                        if (iw < 0 || iw >= W) continue;
                        const float* in_px = &in[((long)ih * W + iw) * Cin];
                        const float* w_px  = &w[(((long)oc * K + kh) * K + kw) * Cin];
                        for (int ic = 0; ic < Cin; ic++) acc += in_px[ic] * w_px[ic];
                    }
                }
                out_px[oc] = acc;
            }
        }
    }
    *dsp_us = (int)(HAP_perf_get_time_us() - t0);
    FARF(RUNTIME_HIGH, "dsp_conv scalar HxW=%dx%d C=%d/%d dsp_us=%d", H, W, Cin, Cout, *dsp_us);
    return 0;
}
