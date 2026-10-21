// dsp_conv_vtcm_probe_main.c — ARM64 driver for the VTCM characterization probe.
// The measurement tool for Lab 3 Task 11 Part A.
// Usage: ./conv_vtcm_probe [--power lab] [--all]
//   default : sections 1-4 (capacity, acquisition limits, bandwidth)
//   --all   : also 5-7, staff reference numbers for Task 11 Part B and
//             Task 10 Part D. Students do not need them; they are answers.

#include "dsp_conv.h"
#include "remote.h"
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*fn_ctrl)(uint32_t, void*, uint32_t);
typedef int (*fn_open)(const char*, remote_handle64*);
typedef int (*fn_close)(remote_handle64);
typedef int (*fn_probe)(remote_handle64, int, int, int, int*, int*, int*, int*);
typedef int (*fn_power)(remote_handle64, int);

#define DSP_CONV_URI  dsp_conv_URI CDSP_DOMAIN "&_unsigned_module=1"

static fn_probe g_probe;
static remote_handle64 g_h;

static int P(int mode, int size, int iters, int *us, int *a, int *b) {
    int ok = 0, aa = 0, bb = 0, uu = 0;
    g_probe(g_h, mode, size, iters, &uu, &ok, &aa, &bb);
    if (us) *us = uu; if (a) *a = aa; if (b) *b = bb;
    return ok;
}

static double gbps(long long bytes, int us) {
    if (us <= 0) return 0.0;
    return (double)bytes / 1e9 / ((double)us / 1e6);
}

int main(int argc, char** argv) {
    const char *power = "lab";
    int all = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--power") && i + 1 < argc) power = argv[++i];
        else if (!strcmp(argv[i], "--all")) all = 1;
    }

    void* cdsprpc = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!cdsprpc) { fprintf(stderr, "dlopen libcdsprpc: %s\n", dlerror()); return 1; }
    fn_ctrl my_ctrl = (fn_ctrl)dlsym(cdsprpc, "remote_session_control");
    if (my_ctrl) { struct remote_rpc_control_unsigned_module u = { .domain = CDSP_DOMAIN_ID, .enable = 1 }; my_ctrl(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof(u)); }

    void* stub = dlopen("./libdsp_conv_stub.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!stub) { fprintf(stderr, "dlopen stub: %s\n", dlerror()); return 1; }
    fn_open  my_open  = (fn_open)dlsym(stub, "dsp_conv_open");
    fn_close my_close = (fn_close)dlsym(stub, "dsp_conv_close");
    fn_power my_power = (fn_power)dlsym(stub, "dsp_conv_set_power_mode");
    g_probe = (fn_probe)dlsym(stub, "dsp_conv_vtcm_probe");
    if (!my_open || !g_probe) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }

    g_h = -1;
    if (my_open(DSP_CONV_URI, &g_h) != 0) { fprintf(stderr, "open failed\n"); return 1; }
    if (my_power) my_power(g_h, !strcmp(power, "lab") ? 1 : 0);

    int us, a, b;

    printf("=== 1. What the device claims ===\n");
    P(0, 0, 0, &us, &a, &b);
    printf("  HAP_query_total_VTCM: page_size=%d page_count=%d  -> total %d KB\n\n",
           a, b, (int)(((long long)a * b) / 1024));

    printf("=== 2. Single acquire: how much can one process actually get? ===\n");
    const int KB = 1024;
    int sizes[] = { 1*KB*KB, 2*KB*KB, 4*KB*KB, 6*KB*KB, 7*KB*KB, 8*KB*KB,
                    8*KB*KB + 1, 8*KB*KB + 4096, 12*KB*KB, 16*KB*KB };
    for (unsigned i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        int ok = P(1, sizes[i], 0, &us, &a, &b);
        printf("  %8.3f MB (%8d bytes) : %-4s (acquire %d us)\n",
               sizes[i] / 1048576.0, sizes[i], ok ? "OK" : "FAIL", us);
    }

    printf("\n=== 3. Two simultaneous acquires of the same size ===\n");
    int two[] = { 1*KB*KB, 2*KB*KB, 4*KB*KB, 5*KB*KB };
    for (unsigned i = 0; i < sizeof(two)/sizeof(two[0]); i++) {
        P(2, two[i], 0, &us, &a, &b);
        printf("  2 x %5.1f MB (= %5.1f MB) : first=%s second=%s\n",
               two[i] / 1048576.0, 2.0 * two[i] / 1048576.0,
               a ? "OK" : "FAIL", b ? "OK" : "FAIL");
    }

    printf("\n=== 4. Bandwidth: VTCM vs ordinary memory (HVX vector accesses) ===\n");
    printf("  %10s  %12s %12s   %12s %12s\n",
           "size", "VTCM write", "VTCM read", "DDR write", "DDR read");
    double ddr_read_big = 0.0;      /* DDR read at the largest size, reused in section 6 */
    int bw_sizes[] = { 64*KB, 256*KB, 1*KB*KB, 2*KB*KB, 4*KB*KB, 8*KB*KB };
    for (unsigned i = 0; i < sizeof(bw_sizes)/sizeof(bw_sizes[0]); i++) {
        int sz = bw_sizes[i];
        int iters = (int)(268435456LL / sz);      /* ~256 MB of traffic per point */
        if (iters < 4) iters = 4;
        long long bytes = (long long)sz * iters;
        double r[4] = {0,0,0,0};
        int modes[4] = { 3, 4, 5, 6 };
        for (int m = 0; m < 4; m++) {
            if (P(modes[m], sz, iters, &us, &a, &b)) r[m] = gbps(bytes, us);
        }
        printf("  %7d KB  %9.1f GB/s %9.1f GB/s   %9.1f GB/s %9.1f GB/s\n",
               sz / 1024, r[0], r[1], r[2], r[3]);
        ddr_read_big = r[3];
    }

    if (!all) {
        printf("\n");
        my_close(g_h);
        return 0;
    }

    printf("\n=== 5. Task 5 reference: 32 MB working set, 8 MB VTCM ===\n");
    printf("  Staging pays only with reuse. P = passes over each block.\n\n");
    printf("  %3s %10s %10s %10s %10s %10s %10s   %8s\n",
           "P", "DDR only", "memcpy", "mcpy+ovl", "HVXcopy", "HVX+ovl", "DMA+ovl", "best/DDR");
    int total = 32 * KB * KB;
    int passes[] = { 1, 2, 4, 8, 16 };
    int modes5[6] = { 10, 11, 12, 15, 16, 18 };
    for (unsigned i = 0; i < sizeof(passes)/sizeof(passes[0]); i++) {
        int P_ = passes[i];
        int t[6] = {0,0,0,0,0,0}, cs[6] = {0,0,0,0,0,0};
        int bad = 0;
        for (int m = 0; m < 6; m++) {
            int uu, aa, bb;
            if (P(modes5[m], total, P_, &uu, &aa, &bb)) { t[m] = uu; cs[m] = aa; }
            if (m && t[m] && cs[m] != cs[0]) bad = 1;
        }
        int best = 0;
        for (int m = 3; m < 6; m++) if (t[m] && (!best || t[m] < best)) best = t[m];
        printf("  %3d %7.1f ms %7.1f ms %7.1f ms %7.1f ms %7.1f ms %7.1f ms   %7.2fx%s\n",
               P_, t[0]/1000.0, t[1]/1000.0, t[2]/1000.0, t[3]/1000.0, t[4]/1000.0,
               t[5]/1000.0, best ? (double)t[0]/best : 0.0,
               bad ? "  <-- CHECKSUM MISMATCH" : "");
    }

    printf("\n=== 6. How fast can we even get data INTO VTCM? (32 MB, DDR -> VTCM) ===\n");
    {
        int uu, aa, bb;
        if (P(13, total, 0, &uu, &aa, &bb))
            printf("  memcpy()      : %8.1f ms  = %5.2f GB/s\n",
                   uu / 1000.0, gbps(total, uu));
        if (P(14, total, 0, &uu, &aa, &bb))
            printf("  HVX vector    : %8.1f ms  = %5.2f GB/s\n",
                   uu / 1000.0, gbps(total, uu));
        if (P(19, total, 0, &uu, &aa, &bb))
            printf("  user DMA      : %8.1f ms  = %5.2f GB/s\n",
                   uu / 1000.0, gbps(total, uu));
        printf("  (compare: DDR read %5.2f GB/s at 8 MB, measured in section 4)\n", ddr_read_big);
    }

    printf("\n=== 7. Task 6: morphology — your code vs Qualcomm's QHL library ===\n");
    printf("  1024x1024 uint8 image, interior-only checksum\n\n");
    printf("  %-8s %4s %12s %12s   %8s  %s\n",
           "op", "K", "scalar", "QHL hvx", "speedup", "match");
    {
        int w = 1024;
        int ks[3] = { 3, 5, 7 };
        const char *names[2] = { "erode", "dilate" };
        for (int op = 0; op < 2; op++) {
            for (int ki = 0; ki < 3; ki++) {
                int uu1 = 0, uu2 = 0, a1 = 0, a2 = 0, bb;
                int m_s = op ? 22 : 20, m_q = op ? 23 : 21;
                int ok1 = P(m_s, w, ks[ki], &uu1, &a1, &bb);
                int ok2 = P(m_q, w, ks[ki], &uu2, &a2, &bb);
                printf("  %-8s %4d %9.2f ms %9.2f ms   %7.1fx  %s\n",
                       names[op], ks[ki], uu1/1000.0, uu2/1000.0,
                       uu2 ? (double)uu1/uu2 : 0.0,
                       (ok1 && ok2) ? (a1 == a2 ? "yes" : "NO") : "err");
            }
        }
    }

    printf("\n");
    my_close(g_h);
    return 0;
}
