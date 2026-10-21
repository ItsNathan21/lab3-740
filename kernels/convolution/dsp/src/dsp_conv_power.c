// dsp_conv_power.c — DSP-side power mode control (mirrors reduction/dsp).
// mode 0 = BASELINE, mode 1 = LAB (TURBO, no sleep). Errors non-fatal.

#include "dsp_conv.h"
#include "HAP_power.h"
#include "HAP_farf.h"

int dsp_conv_set_power_mode(remote_handle64 h, int mode) {
    int rc;
    if (mode == 1) {
        rc = HAP_power_request_abs(1000, 5000, 1);
        if (rc) FARF(ERROR, "HAP_power_request_abs failed rc=%d", rc);
        rc = HAP_power_set_core_corner(NULL,
                 HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO);
        if (rc) FARF(ERROR, "HAP_power_set_core_corner TURBO failed rc=%d", rc);
        rc = HAP_power_set_sleep_mode(NULL, HAP_DCVS_LPM_LEVEL1);
        if (rc) FARF(ERROR, "HAP_power_set_sleep_mode failed rc=%d", rc);
        rc = HAP_power_set_sleep_latency(NULL, 1);
        if (rc) FARF(ERROR, "HAP_power_set_sleep_latency failed rc=%d", rc);
        FARF(RUNTIME_HIGH, "set_power_mode LAB: TURBO+no-sleep applied");
    } else {
        HAP_power_request(0, 0, 0);
        HAP_power_set_sleep_latency(NULL, 65535);
        HAP_power_set_sleep_mode(NULL, HAP_DCVS_LPM_ENABLE_ALL);
        FARF(RUNTIME_HIGH, "set_power_mode BASELINE: defaults restored");
    }
    return 0;
}
