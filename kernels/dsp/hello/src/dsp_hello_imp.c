// dsp_hello_imp.c — skel implementation, runs on cDSP v79
// Compiled with hexagon-clang -mv79 on Windows host.

#include "dsp_hello.h"
#include "HAP_farf.h"
#include "HAP_vtcm_mgr.h"
#include "qurt_hvx.h"

int dsp_hello_open(const char* uri, remote_handle64* h) {
    *h = 1;
    FARF(RUNTIME_HIGH, "dsp_hello_open");
    return 0;
}

int dsp_hello_close(remote_handle64 h) {
    return 0;
}

int dsp_hello_ping(remote_handle64 h, int val, int* result) {
    *result = val + 1;
    FARF(RUNTIME_HIGH, "dsp_hello_ping: %d -> %d", val, *result);
    return 0;
}

int dsp_hello_get_caps(remote_handle64 h,
                       int* hvx_units,
                       int* vtcm_total_kb,
                       int* vtcm_avail_kb) {
    // HVX unit encoding: upper byte = num 128B units, lower byte = num 64B units.
    // QURT_HVX_HW_UNITS_2X128B_4X64B = 0x00000204 means 2x128B = 4x64B.
    *hvx_units = qurt_hvx_get_units();

    // Total VTCM: page_size * page_count bytes.
    unsigned int page_size = 0, page_count = 0;
    if (HAP_query_total_VTCM(&page_size, &page_count) == 0) {
        *vtcm_total_kb = (int)((page_size * page_count) / 1024);
    } else {
        *vtcm_total_kb = -1;
    }

    // Available VTCM: largest contiguous free block.
    unsigned int avail_block = 0, max_page = 0, num_pages = 0;
    if (HAP_query_avail_VTCM(&avail_block, &max_page, &num_pages) == 0) {
        *vtcm_avail_kb = (int)(avail_block / 1024);
    } else {
        *vtcm_avail_kb = -1;
    }

    FARF(RUNTIME_HIGH, "dsp_hello_get_caps: hvx_units=0x%x total_vtcm=%dKB avail_vtcm=%dKB",
         *hvx_units, *vtcm_total_kb, *vtcm_avail_kb);
    return 0;
}
