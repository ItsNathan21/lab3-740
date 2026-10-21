// dsp_conv_morph_imp.c — morphological erode/dilate on the cDSP.
//
// REFERENCE IMPLEMENTATION. Lab 3 Task 10.10 ships a scaffold
// (kernels/lab3/morph/student_morph.c) that replaces this file at build time
// with the scalar path hollowed out. Both define dsp_conv_morph_u8, so the
// build swaps one for the other and nothing else moves.
//
// impl 0 — scalar. A plain KxK neighbourhood min/max per pixel. Reloads each
//          pixel up to KxK times, which is the point: its cost grows with the
//          window area (Task 10.12).
// impl 1 — QHL. Qualcomm's hand-tuned HVX routines. Same answer in the
//          interior, and nearly independent of K, because a sliding-window
//          implementation keeps neighbours in vector registers instead of
//          going back to memory for them.
//
// Border policy: out-of-image neighbours are skipped, so edge pixels take the
// min/max over the part of the window that exists. QHL makes its own choice at
// the border, so the host compares the interior only.

#include "dsp_conv.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "qurt_hvx.h"
#include "qhdsp_hvx.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- scalar
static void morph_scalar(const uint8_t *in, uint8_t *out,
                         int W, int H, int K, int op) {
    const int pad = K / 2;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int acc = (op == 0) ? 255 : 0;
            for (int ky = -pad; ky <= pad; ky++) {
                int yy = y + ky;
                if (yy < 0 || yy >= H) continue;
                for (int kx = -pad; kx <= pad; kx++) {
                    int xx = x + kx;
                    if (xx < 0 || xx >= W) continue;
                    int v = in[(long)yy * W + xx];
                    if (op == 0) { if (v < acc) acc = v; }
                    else         { if (v > acc) acc = v; }
                }
            }
            out[(long)y * W + x] = (uint8_t)acc;
        }
    }
}

// ------------------------------------------------------------------- QHL
// The library wants 128-byte-aligned buffers and a stride that is a multiple
// of 128; the caller guarantees W % 128 == 0 and hands us aligned copies.
static int morph_qhl(uint8_t *in, uint8_t *out, int W, int H, int K, int op) {
    if (op == 0) {
        if (K == 3) return qhdsp_hvx_erode3x3_ab(in, W, (uint32_t)W, (uint32_t)H, out, W);
        if (K == 5) return qhdsp_hvx_erode5x5_ab(in, W, (uint32_t)W, (uint32_t)H, out, W);
        if (K == 7) return qhdsp_hvx_erode7x7_ab(in, W, (uint32_t)W, (uint32_t)H, out, W);
    } else {
        if (K == 3) return qhdsp_hvx_dilate3x3_ab(in, W, (uint32_t)W, (uint32_t)H, out, W);
        if (K == 5) return qhdsp_hvx_dilate5x5_ab(in, W, (uint32_t)W, (uint32_t)H, out, W);
        if (K == 7) return qhdsp_hvx_dilate7x7_ab(in, W, (uint32_t)W, (uint32_t)H, out, W);
    }
    return -1;
}

int dsp_conv_morph_u8(remote_handle64 h,
                      const unsigned char *in, int inLen,
                      unsigned char *out, int outLen,
                      int W, int H, int K, int op, int impl,
                      int *dsp_us) {
    (void)h;
    if (W <= 0 || H <= 0 || (W % 128) != 0) return -1;
    if (K != 3 && K != 5 && K != 7) return -1;
    if (inLen < W * H || outLen < W * H) return -1;

    int rc = 0;
    uint64_t t0, t1;

    if (impl == 0) {
        t0 = HAP_perf_get_time_us();
        morph_scalar(in, out, W, H, K, op);
        t1 = HAP_perf_get_time_us();
    } else {
        // Aligned working copies; the FastRPC buffers carry no alignment
        // guarantee and QHL will not accept them otherwise.
        uint8_t *ain = (uint8_t *)memalign(128, (size_t)W * H);
        uint8_t *aout = (uint8_t *)memalign(128, (size_t)W * H);
        if (!ain || !aout) { free(ain); free(aout); return -1; }
        memcpy(ain, in, (size_t)W * H);

        qurt_hvx_lock(QURT_HVX_MODE_128B);
        t0 = HAP_perf_get_time_us();
        rc = morph_qhl(ain, aout, W, H, K, op);
        t1 = HAP_perf_get_time_us();
        qurt_hvx_unlock();

        memcpy(out, aout, (size_t)W * H);
        free(ain);
        free(aout);
    }

    *dsp_us = (int)(t1 - t0);
    FARF(RUNTIME_HIGH, "morph impl=%d op=%d K=%d %dx%d dsp_us=%d rc=%d",
         impl, op, K, W, H, *dsp_us, rc);
    return rc;
}
