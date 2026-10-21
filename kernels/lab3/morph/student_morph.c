// student_morph.c — Lab 3, Task 10.10. Your file. Edit it.
//
// Erosion replaces each pixel with the MINIMUM over its KxK neighbourhood;
// dilation replaces it with the MAXIMUM. One of them removes isolated bright
// specks, the other fills isolated dark holes. Task 10.9 asks you which is
// which, and why applying the wrong one makes things worse.
//
// Write morph_scalar(). The QHL path below is Qualcomm's tuned HVX library and
// is already wired up — you are not implementing that, you are competing with
// it (Tasks 10.11-10.12).
//
//   bash scripts/build_student.sh morph      # builds and pushes to the board
//   python profiler.py morph dsp -p op=erode -p k=3 -p write_pgm=true
//
// The clean image is dog.pgm (1024x768), next to the binary on the board.
// To see what you are repairing before you run anything, open
// kernels/convolution/dsp/data/dog_corrupted.png: it is exactly the damaged
// input the board builds for the default settings (damage=salt, seed 42).
// The client reports `residual=`, the number of interior pixels where your
// output differs from the same operator applied to the undamaged image, i.e.
// damage that survived. The right operator leaves a small residual; the wrong
// one leaves an order of magnitude more. -p write_pgm=true saves before and
// after PNGs to morph_images/ in the container.
//
// Border policy: any consistent choice is fine as long as you document it. The
// client compares the interior only, because QHL makes its own choice at the
// edge and that is not what this task is about.

#include "dsp_conv.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "qurt_hvx.h"
#include "qhdsp_hvx.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ===========================================================================
// TASK 10.10 — implement erosion and dilation.
//
//   in   : W*H bytes, row-major, one byte per pixel
//   out  : W*H bytes, same layout, yours to fill completely
//   K    : window size, 3, 5 or 7. The window is KxK centred on the pixel.
//   op   : 0 = erode (minimum), 1 = dilate (maximum)
//
// A straightforward nested loop is the right answer here. It will be slow, and
// slow is the point: Task 10.12 asks you to explain why its cost grows with K
// while the library's barely moves.
//
//   for each output pixel (x, y):
//       acc = 255 for erosion, 0 for dilation
//       for ky in -pad..pad, kx in -pad..pad:
//           v = in[(y+ky)*W + (x+kx)]        // skip if outside the image
//           acc = min(acc, v)  for erosion
//           acc = max(acc, v)  for dilation
//       out[y*W + x] = acc
// ===========================================================================
static void morph_scalar(const uint8_t *in, uint8_t *out,
                         int W, int H, int K, int op) {
    // ---- TODO(Task 10.10) ---------------------------------------------------
    // Remove this line once you write the real thing. It copies the input
    // through unchanged, so the damage survives and residual stays high.
    (void)K; (void)op;
    memcpy(out, in, (size_t)W * H);
    // -------------------------------------------------------------------------
}

// ---------------------------------------------------------------------------
// GIVEN. Qualcomm's QHL HVX library. Do not change this.
//
// It wants 128-byte-aligned buffers and a stride that is a multiple of 128,
// which is why the client insists W % 128 == 0 and why the caller below hands
// it aligned copies rather than the FastRPC buffers directly.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// GIVEN. Entry point. Dispatches to yours or to QHL, and times only the work.
// ---------------------------------------------------------------------------
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
