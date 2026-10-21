// dsp_hello_main.c — ARM64 test driver, runs on device CPU.
//
// Two-phase init to bypass Android 15 linker namespace restrictions:
//   1. dlopen("libcdsprpc.so") by name — linker finds /vendor/lib64/libcdsprpc.so
//      and loads it in the vendor namespace (can resolve HIDL/vendor symbols).
//   2. Call remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE) to enable
//      unsigned PD before opening any handle.  Without this, the FastRPC runtime
//      connects via fastrpc_shell_3 (signed) and rejects unsigned module URIs.
//   3. dlopen stub.so — its DT_NEEDED on libcdsprpc.so is already satisfied.
//
// Executable DT_NEEDED: libdl + libc only (no libcdsprpc.so).

#include "dsp_hello.h"
#include "remote.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define STUB_LIB "/data/local/tmp/profiler-kernels/dsp/libdsp_hello_stub.so"

// DSPRPC_CONTROL_UNSIGNED_MODULE = 2, FASTRPC_MANAGE_DSP_LIBRARY_PATH = 20
// (from session_control_req_id enum in remote.h).
// remote_rpc_control_unsigned_module, fastrpc_manage_dsp_lib_path,
// and CDSP_DOMAIN_ID are all defined in remote.h.

#define SKEL_DIR  "/data/local/tmp/profiler-kernels/dsp"

typedef int (*fn_session_ctrl)(uint32_t req, void* data, uint32_t datalen);
typedef int (*fn_open)(const char*, remote_handle64*);
typedef int (*fn_close)(remote_handle64);
typedef int (*fn_ping)(remote_handle64, int, int*);
typedef int (*fn_get_caps)(remote_handle64, int*, int*, int*);

// Use the SDK-generated URI (filename only, no path).
// FastRPC resolves it via the library search path we set below.
#define DSP_HELLO_URI  dsp_hello_URI CDSP_DOMAIN "&_unsigned_module=1"

int main(void) {
    // Step 1: preload libcdsprpc.so by name — vendor namespace, avoids namespace failure.
    void* cdsprpc = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!cdsprpc) {
        fprintf(stderr, "[ARM64] dlopen libcdsprpc.so failed: %s\n", dlerror());
        return 1;
    }
    printf("[ARM64] libcdsprpc.so preloaded\n");

    // Step 2: enable unsigned module loading on cDSP (domain 3) BEFORE any open.
    fn_session_ctrl my_session_ctrl =
        (fn_session_ctrl)dlsym(cdsprpc, "remote_session_control");
    if (!my_session_ctrl) {
        fprintf(stderr, "[ARM64] dlsym remote_session_control failed: %s\n", dlerror());
        dlclose(cdsprpc);
        return 1;
    }
    struct remote_rpc_control_unsigned_module unsig = { .domain = CDSP_DOMAIN_ID, .enable = 1 };
    int rc = my_session_ctrl(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsig, sizeof(unsig));
    if (rc != 0) {
        fprintf(stderr, "[ARM64] remote_session_control(UNSIGNED_MODULE) failed: 0x%x\n", rc);
        dlclose(cdsprpc);
        return 1;
    }
    printf("[ARM64] unsigned PD enabled on domain %d\n", CDSP_DOMAIN_ID);

    // Add skel dir to FastRPC library search path so the filename-only URI resolves.
    fastrpc_manage_dsp_lib_path lpath = {
        .effec_domain_id = CDSP_DOMAIN_ID,
        .type            = FASTRPC_SET_PATH,
        .path            = SKEL_DIR,
        .pathlen         = sizeof(SKEL_DIR),
    };
    rc = my_session_ctrl(FASTRPC_MANAGE_DSP_LIBRARY_PATH, &lpath, sizeof(lpath));
    if (rc != 0) {
        // Non-fatal on older kernels; log and continue.
        fprintf(stderr, "[ARM64] FASTRPC_MANAGE_DSP_LIBRARY_PATH returned 0x%x (continuing)\n", rc);
    } else {
        printf("[ARM64] skel search path set to %s\n", SKEL_DIR);
    }

    // Step 3: load stub — its DT_NEEDED on libcdsprpc.so is already satisfied.
    void* stub = dlopen(STUB_LIB, RTLD_LAZY | RTLD_GLOBAL);
    if (!stub) {
        fprintf(stderr, "[ARM64] dlopen %s failed: %s\n", STUB_LIB, dlerror());
        dlclose(cdsprpc);
        return 1;
    }

    fn_open     my_open     = (fn_open)    dlsym(stub, "dsp_hello_open");
    fn_close    my_close    = (fn_close)   dlsym(stub, "dsp_hello_close");
    fn_ping     my_ping     = (fn_ping)    dlsym(stub, "dsp_hello_ping");
    fn_get_caps my_get_caps = (fn_get_caps)dlsym(stub, "dsp_hello_get_caps");

    if (!my_open || !my_close || !my_ping || !my_get_caps) {
        fprintf(stderr, "[ARM64] dlsym failed: %s\n", dlerror());
        dlclose(stub);
        dlclose(cdsprpc);
        return 1;
    }

    printf("[ARM64] stub loaded OK, opening cDSP handle...\n");
    remote_handle64 h = -1;
    int ret = my_open(DSP_HELLO_URI, &h);
    if (ret != 0) {
        fprintf(stderr, "[ARM64] dsp_hello_open failed: 0x%x\n", ret);
        dlclose(stub);
        dlclose(cdsprpc);
        return 1;
    }
    printf("[ARM64] handle opened: 0x%llx\n", (unsigned long long)h);

    // --- ping ---
    int result = 0;
    ret = my_ping(h, 42, &result);
    if (ret != 0) {
        fprintf(stderr, "[ARM64] ping failed: 0x%x\n", ret);
        my_close(h);
        dlclose(stub);
        dlclose(cdsprpc);
        return 1;
    }
    printf("[ARM64] ping(42) = %d  (expected 43, %s)\n",
           result, result == 43 ? "PASS" : "FAIL");

    // --- capability probe ---
    int hvx_units = 0, vtcm_total_kb = 0, vtcm_avail_kb = 0;
    ret = my_get_caps(h, &hvx_units, &vtcm_total_kb, &vtcm_avail_kb);
    if (ret != 0) {
        fprintf(stderr, "[ARM64] get_caps failed: 0x%x\n", ret);
    } else {
        int units_128b = (hvx_units >> 8) & 0xFF;
        int units_64b  = (hvx_units >> 0) & 0xFF;
        printf("[ARM64] HVX units: 0x%x  (%d x 128B = %d x 64B)\n",
               hvx_units, units_128b, units_64b);
        printf("[ARM64] VTCM total:     %d KB\n", vtcm_total_kb);
        printf("[ARM64] VTCM available: %d KB\n", vtcm_avail_kb);
    }

    my_close(h);
    dlclose(stub);
    dlclose(cdsprpc);
    printf("[ARM64] done.\n");
    return (result == 43) ? 0 : 1;
}
