// dsp_conv_morph_main.c — ARM64 driver for cDSP morphology (Lab 3 Task 10.9-10.12).
//
// Loads a clean image, damages a copy, repairs it with erode or dilate on the
// DSP, and reports timing plus a fingerprint. Two damage modes exist so Task
// 10.9 has something to diagnose:
//
//   --damage salt    isolated bright specks  -> erosion (min filter) removes them
//   --damage pepper  isolated dark holes     -> dilation (max filter) fills them
//
// The clean image is a P5 PGM: --input <file>, default ./dog.pgm (shipped next
// to this binary; 1024x768). Its size sets W and H. If no --input is given and
// dog.pgm is missing, a synthetic blob image of --W x --H is generated instead.
// W must be a multiple of 128 (QHL alignment).
//
// Argv: --input --W --H --k --op erode|dilate --impl student|qhl
//       --damage salt|pepper --seed --iterations --write-pgm
//       (defaults dog.pgm/3/erode/student/salt/42/1; synthetic 1024x1024)
//
// Output:
//   morph-<impl>-start W=.. H=.. k=.. op=.. damage=..
//   morph-<impl>-end fingerprint=<%.6e> residual=<int> wall_us=<int> dsp_us=<int> ...
//
// `residual` counts interior pixels where the repaired image differs from the
// same operator applied to the CLEAN image, i.e. damage the operator failed to
// remove. Comparing against the clean image itself would not work on a photo:
// a min or max filter changes nearly every textured pixel, so right and wrong
// operators would both score ~100%. With this reference the right operator
// leaves a small residual and the wrong one grows every speck into a KxK block.
// --write-pgm dumps clean/damaged/repaired PGMs to look at.

#include <dlfcn.h>
#include <stdbool.h>
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
typedef int (*fn_morph)(remote_handle64, const unsigned char*, int,
                        unsigned char*, int, int, int, int, int, int, int*);

static uint64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, 0);
    return (uint64_t)tv.tv_sec * 1000000ull + tv.tv_usec;
}

static double fingerprint(const unsigned char* p, long n) {
    double acc = 0.0;
    for (long i = 0; i < n; i++) acc += (double)p[i] * ((i % 7) + 1);
    return acc / (double)n;
}

// Minimal binary PGM (P5, maxval 255) reader; '#' comments allowed in the header.
static int pgm_token(FILE* f) {
    int c, v = 0;
    do { c = fgetc(f); if (c == '#') while (c != '\n' && c != EOF) c = fgetc(f); }
    while (c == ' ' || c == '\t' || c == '\n' || c == '\r');
    if (c < '0' || c > '9') return -1;
    while (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); c = fgetc(f); }
    return v;   /* the single whitespace after the token has been consumed */
}

static unsigned char* read_pgm(const char* path, int* W, int* H) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char* img = NULL;
    if (fgetc(f) == 'P' && fgetc(f) == '5') {
        int w = pgm_token(f), h = pgm_token(f), mx = pgm_token(f);
        if (w > 0 && h > 0 && mx == 255) {
            img = (unsigned char*)malloc((size_t)w * h);
            if (img && fread(img, 1, (size_t)w * h, f) == (size_t)w * h) { *W = w; *H = h; }
            else { free(img); img = NULL; }
        }
    }
    fclose(f);
    return img;
}

// Host-side KxK min/max with the same skip-out-of-image border policy as the
// reference DSP code. Only used to build the residual reference.
static void morph_host(const unsigned char* in, unsigned char* out,
                       int W, int H, int K, int op) {
    int r = K / 2;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int acc = op ? 0 : 255;
            for (int dy = -r; dy <= r; dy++) {
                int yy = y + dy; if (yy < 0 || yy >= H) continue;
                for (int dx = -r; dx <= r; dx++) {
                    int xx = x + dx; if (xx < 0 || xx >= W) continue;
                    int v = in[(long)yy * W + xx];
                    acc = op ? (v > acc ? v : acc) : (v < acc ? v : acc);
                }
            }
            out[(long)y * W + x] = (unsigned char)acc;
        }
}

static void write_pgm(const char* path, const unsigned char* img, int W, int H) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    fwrite(img, 1, (size_t)W * H, f);
    fclose(f);
}

int main(int argc, char** argv) {
    int W = 1024, H = 1024, K = 3, iterations = 1;
    unsigned long long seed = 42;
    const char *op_s = "erode", *impl_s = "student", *damage_s = "salt";
    const char *input = NULL;
    bool write_img = false;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--input") && i+1 < argc) input = argv[++i];
        else if (!strcmp(argv[i], "--W") && i+1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--H") && i+1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--op") && i+1 < argc) op_s = argv[++i];
        else if (!strcmp(argv[i], "--impl") && i+1 < argc) impl_s = argv[++i];
        else if (!strcmp(argv[i], "--damage") && i+1 < argc) damage_s = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--iterations") && i+1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--write-pgm")) write_img = true;
    }

    int op = strcmp(op_s, "dilate") ? 0 : 1;
    int impl = strcmp(impl_s, "qhl") ? 0 : 1;
    int salt = strcmp(damage_s, "pepper") ? 1 : 0;

    // Clean image: the PGM if there is one, else the synthetic blob.
    const char* image_s = "synthetic";
    unsigned char* clean = read_pgm(input ? input : "dog.pgm", &W, &H);
    if (clean) image_s = input ? input : "dog.pgm";
    else if (input) { fprintf(stderr, "cannot read %s (need binary PGM, P5, maxval 255)\n", input); return 1; }

    if (W % 128 != 0) {
        fprintf(stderr, "W must be a multiple of 128 (QHL alignment): got %d\n", W);
        return 1;
    }
    long n = (long)W * H;
    if (!clean) {
        clean = (unsigned char*)malloc(n);
        if (!clean) { fprintf(stderr, "oom\n"); return 1; }
        // A smooth subject: bright blob on dark ground, or its inverse.
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int dx = x - W/2, dy = y - H/2;
                int r2 = dx*dx + dy*dy;
                int v = (r2 < (W/3)*(W/3)) ? 200 : 40;
                clean[(long)y*W + x] = (unsigned char)(salt ? v : 255 - v);
            }
        }
    }
    unsigned char* damaged  = (unsigned char*)malloc(n);
    unsigned char* repaired = (unsigned char*)malloc(n);
    unsigned char* expect   = (unsigned char*)malloc(n);
    if (!damaged || !repaired || !expect) { fprintf(stderr, "oom\n"); return 1; }
    // Damage: salt = isolated bright specks (erosion removes them);
    //         pepper = isolated dark holes (dilation fills them).
    memcpy(damaged, clean, n);
    unsigned long long st = seed ? seed : 1;
    // n/64 specks: dense enough to see at a glance, sparse enough that the
    // right operator can still remove almost all of them.
    long nspecks = n / 64;
    for (long i = 0; i < nspecks; i++) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        long idx = (long)((st >> 16) % (unsigned long long)n);
        damaged[idx] = salt ? 255 : 0;
    }

    void* cdsprpc = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!cdsprpc) { fprintf(stderr, "libcdsprpc.so: %s\n", dlerror()); return 1; }
    void* stub = dlopen("./libdsp_conv_stub.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!stub) { fprintf(stderr, "stub: %s\n", dlerror()); return 1; }

    fn_open  my_open  = (fn_open)dlsym(stub, "dsp_conv_open");
    fn_close my_close = (fn_close)dlsym(stub, "dsp_conv_close");
    fn_morph my_morph = (fn_morph)dlsym(stub, "dsp_conv_morph_u8");
    if (!my_open || !my_close || !my_morph) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror()); return 1;
    }

    printf("morph-%s-start image=%s W=%d H=%d k=%d op=%s damage=%s seed=%llu\n",
           impl_s, image_s, W, H, K, op_s, damage_s, seed);
    fflush(stdout);

    remote_handle64 h = 0;
    if (my_open(DSP_CONV_URI, &h) != 0) { fprintf(stderr, "open failed\n"); return 1; }

    int dsp_us = 0, dsp_us_last = 0;
    uint64_t t0 = now_us();
    for (int it = 0; it < iterations; it++) {
        int rc = my_morph(h, damaged, (int)n, repaired, (int)n, W, H, K, op, impl, &dsp_us);
        if (rc != 0) { fprintf(stderr, "morph failed rc=%d\n", rc); my_close(h); return 1; }
        dsp_us_last = dsp_us;
    }
    uint64_t wall_us = now_us() - t0;
    my_close(h);

    // Residual over the interior only: QHL chooses its own border policy, and
    // the border is not what this task is about. Reference = same operator on
    // the clean image (see the header).
    morph_host(clean, expect, W, H, K, op);
    int pad = K / 2;
    long residual = 0;
    for (int y = pad; y < H - pad; y++)
        for (int x = pad; x < W - pad; x++)
            if (repaired[(long)y*W + x] != expect[(long)y*W + x]) residual++;

    if (write_img) {
        write_pgm("morph_clean.pgm", clean, W, H);
        write_pgm("morph_damaged.pgm", damaged, W, H);
        write_pgm("morph_repaired.pgm", repaired, W, H);
    }

    printf("morph-%s-end fingerprint=%.6e residual=%ld wall_us=%llu dsp_us=%d "
           "iterations=%d W=%d H=%d k=%d op=%s damage=%s\n",
           impl_s, fingerprint(repaired, n), residual,
           (unsigned long long)(wall_us / (iterations > 0 ? iterations : 1)),
           dsp_us_last, iterations, W, H, K, op_s, damage_s);

    free(clean); free(damaged); free(repaired); free(expect);
    dlclose(stub); dlclose(cdsprpc);
    return 0;
}
