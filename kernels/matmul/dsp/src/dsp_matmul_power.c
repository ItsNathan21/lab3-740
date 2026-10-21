// dsp_matmul_power.c — DSP-side power mode control for variance sweep.
// Phase 6 variance characterization.
//
// mode 0 = BASELINE : release explicit core-corner, throughput, and sleep votes
// mode 1 = LAB      : request TURBO and no-sleep behavior from the unsigned PD
//
// HAP_power calls run on the cDSP and submit their votes for this PD.
// Any rejected LAB vote is fatal to the caller: a calibration window must not
// silently run at baseline/default power. A zero return only proves that the
// unsigned-PD APIs accepted the requests; HAP exposes no clock/corner readback,
// so it still cannot prove that the physical cDSP clock stayed fixed.

#include "dsp_matmul.h"
#include "HAP_power.h"
#include "HAP_farf.h"

/* Provisioning rejects a skel built before fail-closed LAB handling existed. */
__attribute__((used, visibility("default")))
const char dsp_power_fail_closed_marker[] = "dsp-power-fail-closed-v2";

int dsp_matmul_set_power_mode(remote_handle64 h, int mode) {
    int rc;
    int first_error = 0;

    (void)h;
    if (mode != 0 && mode != 1) {
        FARF(ERROR, "set_power_mode invalid mode=%d", mode);
        return -1;
    }

    if (mode == 1) {
        // LAB: request high MIPS + bus BW + minimal latency
        rc = HAP_power_request_abs(1000, 5000, 1); // MIPS, MB/s bus, 1µs latency
        if (rc) {
            FARF(ERROR, "HAP_power_request_abs failed rc=%d", rc);
            if (!first_error) first_error = rc;
        }

        // Request core TURBO for min/target/max. There is no readback API.
        rc = HAP_power_set_core_corner(NULL,
                 HAP_DCVS_VCORNER_TURBO,
                 HAP_DCVS_VCORNER_TURBO,
                 HAP_DCVS_VCORNER_TURBO);
        if (rc) {
            FARF(ERROR, "HAP_power_set_core_corner TURBO failed rc=%d", rc);
            if (!first_error) first_error = rc;
        }

        rc = HAP_power_set_sleep_mode(NULL, HAP_DCVS_LPM_LEVEL1);
        if (rc) {
            FARF(ERROR, "HAP_power_set_sleep_mode LPM_LEVEL1 failed rc=%d", rc);
            if (!first_error) first_error = rc;
        }

        rc = HAP_power_set_sleep_latency(NULL, 1);
        if (rc) {
            FARF(ERROR, "HAP_power_set_sleep_latency 1us failed rc=%d", rc);
            if (!first_error) first_error = rc;
        }

        if (first_error) {
            FARF(ERROR, "set_power_mode LAB rejected; first rc=%d", first_error);
        } else {
            FARF(RUNTIME_HIGH, "set_power_mode LAB: requests accepted");
        }
    } else {
        // BASELINE: release power votes and restore defaults
        rc = HAP_power_request(0, 0, 0);
        if (rc) {
            FARF(ERROR, "HAP_power_request baseline failed rc=%d", rc);
            if (!first_error) first_error = rc;
        }
        // DISABLE is the documented default/no-explicit-corner setting.
        rc = HAP_power_set_core_corner(NULL,
                 HAP_DCVS_VCORNER_DISABLE,
                 HAP_DCVS_VCORNER_DISABLE,
                 HAP_DCVS_VCORNER_DISABLE);
        if (rc) {
            FARF(ERROR, "HAP_power_set_core_corner baseline failed rc=%d", rc);
            if (!first_error) first_error = rc;
        }
        rc = HAP_power_set_sleep_latency(NULL, 65535);
        if (rc) {
            FARF(ERROR, "HAP_power_set_sleep_latency baseline failed rc=%d", rc);
            if (!first_error) first_error = rc;
        }
        rc = HAP_power_set_sleep_mode(NULL, HAP_DCVS_LPM_ENABLE_ALL);
        if (rc) {
            FARF(ERROR, "HAP_power_set_sleep_mode baseline failed rc=%d", rc);
            if (!first_error) first_error = rc;
        }

        if (first_error) {
            FARF(ERROR, "set_power_mode BASELINE restore rejected; first rc=%d",
                 first_error);
        } else {
            FARF(RUNTIME_HIGH, "set_power_mode BASELINE: defaults restored");
        }
    }
    return first_error;
}
