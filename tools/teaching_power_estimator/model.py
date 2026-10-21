"""HDK8750 teaching power estimator — model core (PURE STDLIB, no numpy).

This module is the *portable* half of the estimator: it defines the model
artifact, the feature layout, the GPU ramp basis, the runtime estimate, and
canonical (de)serialization. It is deliberately dependency-free so the estimate
path can run anywhere (including on-device) — NumPy lives only in ``fit.py``.

The model predicts TOTAL 12 V supply-input power for the HDK8750, as an
opt-in, clearly-labeled *ballpark* teaching estimate. It is NOT a measurement
and NOT a physical per-engine decomposition. See
``decisions/0019-teaching-power-estimate.md`` and
``Rutwik/power-infrastructure/software-power-model-plan.md``.

Model (exactly 8 nonnegative fitted coefficients; quiescent power is an
observed constant, not fitted):

    P_hat        = P_q + P_hat_dynamic
    P_hat_dynamic = F*a + g_gold*X_gold + g_prime*X_prime
                    + d0*G0 + d1*G1 + d2*G2
                    + d_npu*N + d_dsp*D                            (all >= 0)

where per calibration/estimation window:
    a       = activity fraction (workload-intent, union for concurrent)
    X_c     = mean_t[ EAS_power_W(cluster, f_c(t)) * sum_i util_i(t) ]   (W scale)
    G_j     = mean_t[ I_GPU(t) * busy(t) * phi_j(f_gpu(t)) ]             (busy in [0,1])
    N, D    = duration-weighted NPU/DSP path-active fractions.  The two
              paths hardware-serialize on the shared Hexagon cDSP and are
              therefore required to be mutually exclusive.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

# --- versions & fixed identity -------------------------------------------------

ARTIFACT_SCHEMA_VERSION = 2
MODEL_EQUATION_VERSION = "teaching-power-v2"
TELEMETRY_AGGREGATION_VERSION = "agg-v2"

# The measurement boundary is the board's 12 V supply input on every HDK we
# calibrate; the board itself is identified by calibration_board_serial, so this
# names the boundary rather than the board.
BOUNDARY = "HDK_12V_supply_input"
SCOPE = "whole_system"
QUALITY = "modeled_ballpark"

# Feature columns for the dynamic fit, and the coefficient each multiplies.
# Order is load-bearing: fit, estimate, and serialization all rely on it.
FEATURE_ORDER = [
    "activity_fraction",
    "x_gold",
    "x_prime",
    "gpu_g0",
    "gpu_g1",
    "gpu_g2",
    "npu_active",
    "dsp_active",
]
COEFFICIENT_NAMES = [
    "floor_F",
    "gain_gold",
    "gain_prime",
    "gpu_d0",
    "gpu_d1",
    "gpu_d2",
    "dP_npu",
    "dP_dsp",
]
COEFFICIENT_UNITS = {
    "floor_F": "W_per_activity_fraction",
    "gain_gold": "W_per_(EAS_W*util)",
    "gain_prime": "W_per_(EAS_W*util)",
    "gpu_d0": "W_per_(busy*phi)",
    "gpu_d1": "W_per_(busy*phi)",
    "gpu_d2": "W_per_(busy*phi)",
    "dP_npu": "W_per_npu_active_fraction",
    "dP_dsp": "W_per_dsp_active_fraction",
    "gold_cluster": "W_per_GHz",
    "gold_percore": "W_per_(core*GHz)",
    "prime_cluster": "W_per_GHz",
    "prime_percore": "W_per_(core*GHz)",
}

ACTIVE_FRACTION_FEATURES = {"activity_fraction", "npu_active", "dsp_active"}
SUPPORTED_PERF_MODES = ("sustained",)
SUPPORTED_DSP_POWER_MODES = ("lab",)
MODEL_PROFILE_NPU_DSP = "npu_dsp"
MODEL_PROFILE_NPU_ONLY = "npu_only"
MODEL_PROFILE_CPU_GPU = "cpu_gpu_only"
MODEL_PROFILE_CPU_GPU_FREQ = "cpu_gpu_freq"
# Fourth rung: CPU/GPU fitted as usual, plus an NPU term that was DECLARED from
# operator meter readings rather than fitted from a validated window. It exists
# because the NPU windows are not capturable on this board (discovery 73) and a
# labelled approximate number was judged more useful to students than a hole.
# Its quality tag is deliberately NOT `modeled_ballpark`, so nothing can mistake
# it for the calibrated model.
MODEL_PROFILE_NPU_DECLARED = "npu_declared"
QUALITY_DECLARED = "declared_unvalidated"
MODEL_PROFILES = (MODEL_PROFILE_NPU_DSP, MODEL_PROFILE_NPU_ONLY,
                  MODEL_PROFILE_CPU_GPU, MODEL_PROFILE_NPU_DECLARED,
                  MODEL_PROFILE_CPU_GPU_FREQ)
NPU_ONLY_FEATURE_ORDER = FEATURE_ORDER[:-1]
NPU_ONLY_COEFFICIENT_NAMES = COEFFICIENT_NAMES[:-1]
# Third rung: CPU+GPU only, no Hexagon term at all. This is the v1 fit's exact
# design, re-expressed under the v2 schema so there is ONE estimator code path
# instead of a v1 branch. It exists because the NPU windows are not capturable
# on this board without cooling (discovery 73) and the DSP path cannot be pinned
# at all (discovery 64) -- so shipping a usable CPU/GPU estimator that reports
# the Hexagon as `unavailable` beats shipping nothing. Nothing here is fitted:
# a cpu_gpu_only artifact carries the v1 coefficients unchanged.
CPU_GPU_FEATURE_ORDER = FEATURE_ORDER[:-2]
CPU_GPU_COEFFICIENT_NAMES = COEFFICIENT_NAMES[:-2]
# Fourth rung: CPU+GPU with the CPU expressed in frequency rather than the
# kernel's EAS table. Discovery 80 measured `actual ~ EAS_W^0.66` on SM8850 --
# the vendor table is the SM8750 curve resampled onto the new OPP ladder and does
# not describe this silicon -- so an EAS-based fit cannot pass its own gate here.
#
# Each cluster gets TWO terms rather than one, because power is strongly
# sublinear in core count (six Gold cores cost ~2.2x one, not 6x): a fixed cost
# for having the cluster clocked at f, plus a per-busy-core cost. At n saturated
# cores that is (cluster + percore*n) * f. A single `cores x GHz` product was
# tried first and falsified by the 1-core windows. See ADR 0023.
CPU_GPU_FREQ_FEATURE_ORDER = [
    "activity_fraction",
    "xf_gold_cluster",
    "xf_gold_percore",
    "xf_prime_cluster",
    "xf_prime_percore",
    "gpu_g0",
    "gpu_g1",
    "gpu_g2",
]
CPU_GPU_FREQ_COEFFICIENT_NAMES = [
    "floor_F",
    "gold_cluster",
    "gold_percore",
    "prime_cluster",
    "prime_percore",
    "gpu_d0",
    "gpu_d1",
    "gpu_d2",
]
CPU_IDLE_BASIS_SEMANTICS = "max_zero_absolute_eas_util_minus_median_idle_anchor"
SOURCE_SESSION_HASH_ALGORITHM = "sha256-project-canonical-json-v1"
QNN_REQUIRED_RUNTIME_FILES = {
    "libQnnHtp.so",
    "libQnnHtpPrepare.so",
    "libQnnHtpV79Stub.so",
    "libQnnHtpV79CalculatorStub.so",
    "libQnnSystem.so",
    "libQnnHtpV79Skel.so",
    "libCalculator_skel.so",
    "qnn-platform-validator",
    "qnn-net-run",
}
MINIMUM_VALIDATED_INTERVAL_S = 30.0
MANDATORY_KNOWN_UNMODELED_TERMS = {
    "memory/DDR/LLCC",
    "NPU/DSP internal frequency and utilization",
    "NPU/DSP workload-class variation within a path",
    "precision",
    "thermal-leakage",
    "PMIC-efficiency",
    "cross-board transfer",
}

REQUIRED_GATE_CHECKS = {
    MODEL_PROFILE_NPU_DSP: {
        "holdout_ids_match",
        "has_cpu_holdout",
        "has_gpu_holdout",
        "has_npu_holdout",
        "has_dsp_holdout",
        "npu_coefficient_positive",
        "dsp_coefficient_positive",
        "beats_constant",
        "beats_idle_plus_active",
        "cpu_beats_eas_only",
        "gpu_beats_active_constant",
        "npu_within_25pct_observed",
        "dsp_within_25pct_observed",
        "hexagon_beats_shared_active_constant",
        "cpu_gpu_mae_no_material_regression",
        "cpu_gpu_max_rel_no_material_regression",
    },
    MODEL_PROFILE_NPU_ONLY: {
        "holdout_ids_match",
        "has_cpu_holdout",
        "has_gpu_holdout",
        "has_npu_holdout",
        "npu_coefficient_positive",
        "beats_constant",
        "beats_idle_plus_active",
        "cpu_beats_eas_only",
        "gpu_beats_active_constant",
        "npu_within_25pct_observed",
        "cpu_gpu_mae_no_material_regression",
        "cpu_gpu_max_rel_no_material_regression",
    },
    # The v1 design's own gates. No Hexagon checks, because no Hexagon term is
    # fitted; and no cpu_gpu regression checks, because this profile *is* the
    # baseline those compare against -- asking it not to regress against itself
    # would be vacuous.
    MODEL_PROFILE_NPU_DECLARED: {
        "holdout_ids_match",
        "has_cpu_holdout",
        "has_gpu_holdout",
        "beats_constant",
        "beats_idle_plus_active",
        "cpu_beats_eas_only",
        "gpu_beats_active_constant",
    },
    MODEL_PROFILE_CPU_GPU: {
        "holdout_ids_match",
        "has_cpu_holdout",
        "has_gpu_holdout",
        "beats_constant",
        "beats_idle_plus_active",
        "cpu_beats_eas_only",
        "gpu_beats_active_constant",
    },
    # Same gates as cpu_gpu_only plus the two absolute-accuracy ceilings. The
    # EAS-only baseline is still required: cpu_gpu_freq does not use the EAS
    # table, so beating a model that does is exactly the comparison that
    # justifies dropping it (discovery 80).
    MODEL_PROFILE_CPU_GPU_FREQ: {
        "holdout_ids_match",
        "has_cpu_holdout",
        "has_gpu_holdout",
        "beats_constant",
        "beats_idle_plus_active",
        "cpu_beats_eas_only",
        "gpu_beats_active_constant",
        "cpu_gpu_mae_no_material_regression",
        "cpu_gpu_max_rel_no_material_regression",
    },
}

# Fixed, load-validated statement of what the two Hexagon coefficients mean.
# There is deliberately no separately fitted "shared-domain" term: because the
# paths serialize, hexagon_active == npu_active + dsp_active and such a term
# would make the design matrix rank-deficient.  Each coefficient is therefore
# an honest path-total attribution that absorbs the shared-domain wake cost.
HEXAGON_SUBSYSTEM = {
    "hardware_domain": "Hexagon_cDSP",
    "paths": ["QNN_HTP_v79", "FastRPC_user_PD"],
    "hardware_serialized": True,
    "path_term_semantics": "mutually_exclusive_effective_path_step_attributions",
    "feature_semantics": "duration_weighted_workload_intent_fraction",
    "coefficient_scope": "whole_system_increment_after_common_floor_and_other_joint_features",
    "frequency_scaled": False,
    "physical_shared_path_decomposition": False,
    "concurrent_npu_dsp_supported": False,
    "concurrent_path_terms_summable": False,
}

HEXAGON_SUBSYSTEM_NPU_DECLARED = {
    "hardware_domain": "Hexagon_cDSP",
    "paths": ["QNN_HTP_v79"],
    "uncalibrated_peer_path": "FastRPC_user_PD",
    "hardware_serialized": True,
    "path_term_semantics": "declared_operator_meter_single_point",
    "feature_semantics": "duration_weighted_workload_intent_fraction",
    "coefficient_scope": "whole_system_increment_after_common_floor_and_other_joint_features",
    "frequency_scaled": False,
    "physical_shared_path_decomposition": False,
    "concurrent_npu_dsp_supported": False,
    "concurrent_path_terms_summable": False,
    "npu_calibrated": False,
    "npu_declared": True,
    "npu_validated": False,
    "dsp_calibrated": False,
    "npu_declared_reason": "operator meter readings at the 12 V input; no "
                           "stationary window is capturable without active "
                           "cooling (discovery 73)",
    "dsp_uncalibrated_reason": "unsigned PD refuses the HAP lab power vote "
                               "0x80000414 (discovery 64)",
}

HEXAGON_SUBSYSTEM_CPU_GPU = {
    "hardware_domain": "Hexagon_cDSP",
    "paths": [],
    "uncalibrated_paths": ["QNN_HTP_v79", "FastRPC_user_PD"],
    "hardware_serialized": True,
    "path_term_semantics": "no_hexagon_term_fitted",
    "feature_semantics": "not_applicable",
    "coefficient_scope": "not_applicable",
    "frequency_scaled": False,
    "physical_shared_path_decomposition": False,
    "concurrent_npu_dsp_supported": False,
    "concurrent_path_terms_summable": False,
    "npu_calibrated": False,
    "dsp_calibrated": False,
    # Why each is uncalibrated, carried in the artifact so a reader never has to
    # guess whether it was an oversight.
    "npu_uncalibrated_reason": "no capturable stationary window without active "
                              "cooling (discovery 73)",
    "dsp_uncalibrated_reason": "unsigned PD refuses the HAP lab power vote "
                              "0x80000414 (discovery 64)",
}

HEXAGON_SUBSYSTEM_NPU_ONLY = {
    "hardware_domain": "Hexagon_cDSP",
    "paths": ["QNN_HTP_v79"],
    "uncalibrated_peer_path": "FastRPC_user_PD",
    "hardware_serialized": True,
    "path_term_semantics": "effective_npu_path_step_attribution",
    "feature_semantics": "duration_weighted_workload_intent_fraction",
    "coefficient_scope": "whole_system_increment_after_common_floor_and_other_joint_features",
    "frequency_scaled": False,
    "physical_shared_path_decomposition": False,
    "concurrent_npu_dsp_supported": False,
    "concurrent_path_terms_summable": False,
    "dsp_calibrated": False,
}


def feature_order_for_profile(model_profile: str) -> list[str]:
    if model_profile == MODEL_PROFILE_NPU_DSP:
        return list(FEATURE_ORDER)
    if model_profile == MODEL_PROFILE_NPU_ONLY:
        return list(NPU_ONLY_FEATURE_ORDER)
    if model_profile == MODEL_PROFILE_NPU_DECLARED:
        return list(NPU_ONLY_FEATURE_ORDER)
    if model_profile == MODEL_PROFILE_CPU_GPU:
        return list(CPU_GPU_FEATURE_ORDER)
    if model_profile == MODEL_PROFILE_CPU_GPU_FREQ:
        return list(CPU_GPU_FREQ_FEATURE_ORDER)
    raise ModelError(f"unsupported model_profile {model_profile!r}")


def coefficient_names_for_profile(model_profile: str) -> list[str]:
    if model_profile == MODEL_PROFILE_NPU_DSP:
        return list(COEFFICIENT_NAMES)
    if model_profile == MODEL_PROFILE_NPU_ONLY:
        return list(NPU_ONLY_COEFFICIENT_NAMES)
    if model_profile == MODEL_PROFILE_NPU_DECLARED:
        return list(NPU_ONLY_COEFFICIENT_NAMES)
    if model_profile == MODEL_PROFILE_CPU_GPU:
        return list(CPU_GPU_COEFFICIENT_NAMES)
    if model_profile == MODEL_PROFILE_CPU_GPU_FREQ:
        return list(CPU_GPU_FREQ_COEFFICIENT_NAMES)
    raise ModelError(f"unsupported model_profile {model_profile!r}")


def hexagon_subsystem_for_profile(model_profile: str) -> dict[str, Any]:
    if model_profile == MODEL_PROFILE_NPU_DSP:
        return dict(HEXAGON_SUBSYSTEM)
    if model_profile == MODEL_PROFILE_NPU_ONLY:
        return dict(HEXAGON_SUBSYSTEM_NPU_ONLY)
    if model_profile == MODEL_PROFILE_NPU_DECLARED:
        return dict(HEXAGON_SUBSYSTEM_NPU_DECLARED)
    if model_profile in (MODEL_PROFILE_CPU_GPU, MODEL_PROFILE_CPU_GPU_FREQ):
        # Same Hexagon story either way: nothing on it is calibrated. The
        # profiles differ only in how the CPU term is expressed.
        return dict(HEXAGON_SUBSYSTEM_CPU_GPU)
    raise ModelError(f"unsupported model_profile {model_profile!r}")


class ModelError(ValueError):
    """Raised on malformed artifacts, features, or unsupported versions."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_digest_map(name: str, value: Any, required_keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != required_keys:
        raise ModelError(f"{name} must contain exactly {sorted(required_keys)!r}")
    for key, digest in value.items():
        if not isinstance(key, str) or not key or not _is_sha256(digest):
            raise ModelError(f"{name}[{key!r}] must be a lowercase SHA-256 digest")


def _validate_hexagon_runtime_provenance(value: Any, model_profile: str) -> None:
    # cpu_gpu_only calibrates no Hexagon path, so there is no runtime to attest.
    # It must still say so explicitly rather than omit the field, so that
    # "no provenance" is a recorded state and not a missing one.
    if model_profile in (MODEL_PROFILE_CPU_GPU, MODEL_PROFILE_CPU_GPU_FREQ,
                         MODEL_PROFILE_NPU_DECLARED):
        if not isinstance(value, dict) or set(value) != {"declared", "reason"}:
            raise ModelError(
                "cpu_gpu_only hexagon_runtime_provenance must contain exactly "
                "['declared', 'reason']"
            )
        if value["declared"] is not False:
            raise ModelError(
                f"profile {model_profile!r} attests no Hexagon runtime; "
                "hexagon_runtime_provenance.declared must be False"
            )
        if not isinstance(value["reason"], str) or not value["reason"].strip():
            raise ModelError(
                "cpu_gpu_only hexagon_runtime_provenance.reason must be a "
                "non-empty explanation"
            )
        return
    expected_paths = {"npu", "dsp"} if model_profile == MODEL_PROFILE_NPU_DSP else {"npu"}
    if not isinstance(value, dict) or set(value) != expected_paths:
        raise ModelError(
            f"hexagon_runtime_provenance must contain exactly {sorted(expected_paths)!r}"
        )
    npu = value.get("npu")
    if not isinstance(npu, dict):
        raise ModelError("hexagon_runtime_provenance.npu must be an object")
    expected_npu_keys = {
        "perf_mode",
        "qairt_version",
        "qairt_runtime_hashes",
        "workload_binary_sha256_by_variant",
        "qnn_core_api",
        "mode_proof",
    }
    if set(npu) != expected_npu_keys:
        raise ModelError(
            "hexagon_runtime_provenance.npu has an unsupported/missing field set"
        )
    if npu.get("perf_mode") != "sustained":
        raise ModelError("NPU runtime provenance perf_mode must be 'sustained'")
    if not isinstance(npu.get("qairt_version"), str) or not npu["qairt_version"].strip():
        raise ModelError("NPU runtime provenance qairt_version must be non-empty")
    if not re.fullmatch(r"2\.(?:3[0-9]|[4-9][0-9]|[1-9][0-9]{2,})\.\d+", str(npu.get("qnn_core_api", ""))):
        raise ModelError("NPU runtime provenance qnn_core_api must be compatible 2.30+")
    if npu.get("mode_proof") != "client_log_vote_accepted_and_dispatch_success":
        raise ModelError("NPU runtime provenance lacks the required live mode/dispatch proof")
    _validate_digest_map(
        "hexagon_runtime_provenance.npu.qairt_runtime_hashes",
        npu.get("qairt_runtime_hashes"),
        set(QNN_REQUIRED_RUNTIME_FILES),
    )
    _validate_digest_map(
        "hexagon_runtime_provenance.npu.workload_binary_sha256_by_variant",
        npu.get("workload_binary_sha256_by_variant"),
        {"qnn_fp16", "qnn_int8"},
    )

    if model_profile == MODEL_PROFILE_NPU_DSP:
        dsp = value.get("dsp")
        if not isinstance(dsp, dict):
            raise ModelError("hexagon_runtime_provenance.dsp must be an object")
        if set(dsp) != {
            "power_mode",
            "workload_binary_sha256_by_variant",
            "dsp_stub_sha256",
            "dsp_skel_sha256",
            "mode_proof",
        }:
            raise ModelError(
                "hexagon_runtime_provenance.dsp has an unsupported/missing field set"
            )
        if dsp.get("power_mode") != "lab":
            raise ModelError("DSP runtime provenance power_mode must be 'lab'")
        if dsp.get("mode_proof") != "client_log_hap_vote_accepted_and_dispatch_success":
            raise ModelError("DSP runtime provenance lacks the required live mode/dispatch proof")
        _validate_digest_map(
            "hexagon_runtime_provenance.dsp.workload_binary_sha256_by_variant",
            dsp.get("workload_binary_sha256_by_variant"),
            {"dsp_vtcm", "dsp_hvx"},
        )
        for key in ("dsp_stub_sha256", "dsp_skel_sha256"):
            if not _is_sha256(dsp.get(key)):
                raise ModelError(f"hexagon_runtime_provenance.dsp.{key} must be SHA-256")


# --- GPU ramp basis (piecewise-linear, monotone by cumulative increments) ------

def gpu_ramp_basis(freq_hz: float, knots_hz: list[float]) -> tuple[float, float, float]:
    """Return (phi0, phi1, phi2) for a GPU frequency given 3 ascending knots.

    phi0(f) = 1
    phi1(f) = clip((f - k0) / (k1 - k0), 0, 1)
    phi2(f) = clip((f - k1) / (k2 - k1), 0, 1)

    With nonnegative coefficients d0,d1,d2 the fixed-busy power at the three
    knots is d0, d0+d1, d0+d1+d2 -> monotone by construction.
    """
    if len(knots_hz) != 3:
        raise ModelError(f"gpu knots must be 3 ascending values, got {knots_hz!r}")
    try:
        freq_hz = float(freq_hz)
        k0, k1, k2 = (float(value) for value in knots_hz)
    except (TypeError, ValueError) as exc:
        raise ModelError("gpu frequency and knots must be numeric") from exc
    if not all(math.isfinite(value) for value in (freq_hz, k0, k1, k2)):
        raise ModelError("gpu frequency and knots must be finite")
    if not (k0 < k1 < k2):
        raise ModelError(f"gpu knots must be strictly ascending, got {knots_hz!r}")

    def clip01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    phi0 = 1.0
    phi1 = clip01((freq_hz - k0) / (k1 - k0))
    phi2 = clip01((freq_hz - k1) / (k2 - k1))
    return phi0, phi1, phi2


def gpu_freq_in_range(freq_hz: float, knots_hz: list[float]) -> bool:
    return knots_hz[0] <= freq_hz <= knots_hz[2]


# --- artifact ------------------------------------------------------------------

@dataclass
class PowerModelArtifact:
    """Fitted, self-describing teaching-power model.

    Coefficients are stored by name in ``named_coefficients`` as
    ``{name: {"value": float, "units": str}}``. Use ``theta()`` for the ordered
    vector aligned with ``FEATURE_ORDER``.
    """

    model_version: str
    model_profile: str
    quiescent_power_w: float
    idle_anchor_power_w: list[float]
    named_coefficients: dict[str, dict[str, Any]]
    gpu_ramp: dict[str, Any]  # {"knots_hz": [k0,k1,k2], "units": "Hz"}
    calibration_control_config: dict[str, Any]
    train_window_ids: list[str]
    holdout_window_ids: list[str]
    validation_observed: dict[str, Any]
    known_unmodeled_terms: list[str]
    minimum_validated_interval_s: float
    calibration_board_serial: str
    android_build_fingerprint: str
    kernel_release: str
    eas_table_hash: str
    supported_cpu_gpu_opps: dict[str, Any]
    calibration_temperature_range: dict[str, Any]
    calibration_dataset_hash: str
    fit_input_hash: str
    source_session_hashes: dict[str, str]
    source_session_hash_algorithm: str
    cpu_idle_basis: dict[str, Any]
    hexagon_runtime_provenance: dict[str, Any]
    supported_perf_modes: list[str]
    supported_dsp_power_modes: list[str]
    hexagon_subsystem: dict[str, Any]
    # The calibration board's EAS table, {cluster: [[freq_khz, power_w], ...]}.
    # Defaulted because artifacts predating it must still load; when present it
    # lets the estimator's gate distinguish "different silicon" (the frequency
    # ladder changed) from "same board, leakage redetermined at boot" (the power
    # values moved a fraction of a percent). eas_table_hash alone cannot.
    eas_table: dict[str, Any] = field(default_factory=dict)
    # fixed identity (defaulted; validated on load)
    artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION
    model_equation_version: str = MODEL_EQUATION_VERSION
    telemetry_aggregation_version: str = TELEMETRY_AGGREGATION_VERSION
    boundary: str = BOUNDARY
    scope: str = SCOPE
    quality: str = QUALITY

    # -- coefficient access --
    def theta(self) -> list[float]:
        out = []
        for name in self.coefficient_names():
            entry = self.named_coefficients.get(name)
            if entry is None or "value" not in entry:
                raise ModelError(f"artifact missing coefficient {name!r}")
            out.append(float(entry["value"]))
        return out

    def feature_order(self) -> list[str]:
        return feature_order_for_profile(self.model_profile)

    def coefficient_names(self) -> list[str]:
        return coefficient_names_for_profile(self.model_profile)

    def knots_hz(self) -> list[float]:
        return [float(k) for k in self.gpu_ramp["knots_hz"]]

    # -- (de)serialization --
    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "model_version": self.model_version,
            "model_profile": self.model_profile,
            "model_equation_version": self.model_equation_version,
            "telemetry_aggregation_version": self.telemetry_aggregation_version,
            "boundary": self.boundary,
            "scope": self.scope,
            "quality": self.quality,
            "quiescent_power_w": self.quiescent_power_w,
            "idle_anchor_power_w": list(self.idle_anchor_power_w),
            "named_coefficients": self.named_coefficients,
            "feature_order": self.feature_order(),
            "gpu_ramp": self.gpu_ramp,
            "calibration_control_config": self.calibration_control_config,
            "train_window_ids": list(self.train_window_ids),
            "holdout_window_ids": list(self.holdout_window_ids),
            "validation_observed": self.validation_observed,
            "known_unmodeled_terms": list(self.known_unmodeled_terms),
            "minimum_validated_interval_s": self.minimum_validated_interval_s,
            "calibration_board_serial": self.calibration_board_serial,
            "android_build_fingerprint": self.android_build_fingerprint,
            "kernel_release": self.kernel_release,
            "eas_table_hash": self.eas_table_hash,
            "eas_table": self.eas_table,
            "supported_cpu_gpu_opps": self.supported_cpu_gpu_opps,
            "calibration_temperature_range": self.calibration_temperature_range,
            "calibration_dataset_hash": self.calibration_dataset_hash,
            "fit_input_hash": self.fit_input_hash,
            "source_session_hashes": self.source_session_hashes,
            "source_session_hash_algorithm": self.source_session_hash_algorithm,
            "cpu_idle_basis": self.cpu_idle_basis,
            "hexagon_runtime_provenance": self.hexagon_runtime_provenance,
            "supported_perf_modes": list(self.supported_perf_modes),
            "supported_dsp_power_modes": list(self.supported_dsp_power_modes),
            "hexagon_subsystem": self.hexagon_subsystem,
        }

    def to_json(self) -> str:
        """Canonical JSON: sorted keys, compact, no NaN/Inf."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PowerModelArtifact":
        if not isinstance(d, dict):
            raise ModelError(f"artifact root must be an object, got {type(d).__name__}")
        eqv = d.get("model_equation_version")
        if eqv != MODEL_EQUATION_VERSION:
            raise ModelError(
                f"unsupported model_equation_version {eqv!r} (this build understands "
                f"{MODEL_EQUATION_VERSION!r})"
            )
        if d.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ModelError(
                f"unsupported artifact_schema_version {d.get('artifact_schema_version')!r}"
            )
        model_profile = d.get("model_profile")
        if model_profile not in MODEL_PROFILES:
            raise ModelError(f"unsupported model_profile {model_profile!r}")
        expected_feature_order = feature_order_for_profile(model_profile)
        expected_coefficients = coefficient_names_for_profile(model_profile)
        if d.get("feature_order") != expected_feature_order:
            raise ModelError(f"feature_order mismatch: {d.get('feature_order')!r}")
        model_version = d.get("model_version")
        if (
            not isinstance(model_version, str)
            or not re.fullmatch(r"sm8[0-9]{3}-teaching-v2[A-Za-z0-9_.-]*", model_version)
        ):
            # Board-agnostic: the model is per-SoC and the artifact is board-gated
            # by serial anyway, so pinning this to one SoC only blocked SM8850.
            raise ModelError("artifact model_version must be a safe smXXXX-teaching-v2 slug")
        # fixed identity must not be tampered with
        want_quality = (QUALITY_DECLARED
                        if model_profile == MODEL_PROFILE_NPU_DECLARED else QUALITY)
        for key, want in (("boundary", BOUNDARY), ("scope", SCOPE),
                          ("quality", want_quality),
                          ("telemetry_aggregation_version", TELEMETRY_AGGREGATION_VERSION)):
            if d.get(key) != want:
                raise ModelError(f"artifact {key}={d.get(key)!r} != expected {want!r}")
        # quiescent power must be finite and positive
        try:
            pq = float(d.get("quiescent_power_w"))
        except (TypeError, ValueError) as exc:
            raise ModelError(
                f"invalid quiescent_power_w {d.get('quiescent_power_w')!r}"
            ) from exc
        if not math.isfinite(pq) or pq <= 0.0:
            raise ModelError(f"invalid quiescent_power_w {pq!r}")
        # coefficients: present, finite, nonnegative, correct units
        nc = d.get("named_coefficients", {})
        if not isinstance(nc, dict) or set(nc) != set(expected_coefficients):
            raise ModelError(
                f"named_coefficients must contain exactly the {model_profile} coefficients "
                f"{expected_coefficients!r}; got "
                f"{sorted(nc) if isinstance(nc, dict) else nc!r}"
            )
        for name in expected_coefficients:
            entry = nc.get(name)
            if not isinstance(entry, dict) or "value" not in entry:
                raise ModelError(f"artifact missing coefficient {name!r}")
            try:
                v = float(entry["value"])
            except (TypeError, ValueError) as exc:
                raise ModelError(f"coefficient {name!r} value is not numeric") from exc
            if not math.isfinite(v) or v < 0.0:
                raise ModelError(f"coefficient {name!r} invalid (must be finite, >=0): {v!r}")
            if entry.get("units") != COEFFICIENT_UNITS[name]:
                raise ModelError(f"coefficient {name!r} wrong units {entry.get('units')!r}")
        # GPU knots: 3, finite, strictly ascending
        gpu_ramp = d.get("gpu_ramp")
        if not isinstance(gpu_ramp, dict):
            raise ModelError("artifact gpu_ramp must be an object")
        knots = gpu_ramp.get("knots_hz")
        try:
            valid_knots = (
                isinstance(knots, list)
                and len(knots) == 3
                and all(math.isfinite(float(k)) for k in knots)
            )
        except (TypeError, ValueError):
            valid_knots = False
        if not valid_knots:
            raise ModelError(f"invalid gpu_ramp.knots_hz {knots!r}")
        if not (0.0 < float(knots[0]) < float(knots[1]) < float(knots[2])):
            raise ModelError(f"gpu knots not strictly ascending: {knots!r}")
        # required provenance / identity fields must be present and non-empty
        for key in ("model_version", "calibration_board_serial", "android_build_fingerprint",
                    "kernel_release", "eas_table_hash", "calibration_dataset_hash", "fit_input_hash",
                    "source_session_hashes", "train_window_ids", "holdout_window_ids"):
            if not d.get(key):
                raise ModelError(f"artifact missing required field {key!r}")
        for key in (
            "calibration_board_serial",
            "android_build_fingerprint",
            "kernel_release",
        ):
            if not isinstance(d.get(key), str) or not d[key].strip():
                raise ModelError(f"artifact {key} must be a non-empty string")
        for key in ("eas_table_hash", "calibration_dataset_hash", "fit_input_hash"):
            if not _is_sha256(d.get(key)):
                raise ModelError(f"artifact {key} must be lowercase SHA-256")
        # idle anchors: exactly 3, finite, positive
        anchors = d.get("idle_anchor_power_w") or []
        try:
            valid_anchors = (
                isinstance(anchors, list)
                and len(anchors) == 3
                and all(math.isfinite(float(a)) and float(a) > 0 for a in anchors)
            )
        except (TypeError, ValueError):
            valid_anchors = False
        if not valid_anchors:
            raise ModelError(f"idle_anchor_power_w must be 3 finite positive values: {anchors!r}")
        if not math.isclose(pq, float(statistics.median(float(a) for a in anchors)), rel_tol=0.0, abs_tol=1e-9):
            raise ModelError("quiescent_power_w must equal the median of the three idle anchors")
        # minimum validated interval: finite, positive
        try:
            mvi = float(d.get("minimum_validated_interval_s", 0.0))
        except (TypeError, ValueError) as exc:
            raise ModelError("minimum_validated_interval_s must be numeric") from exc
        if not math.isfinite(mvi) or mvi < MINIMUM_VALIDATED_INTERVAL_S:
            raise ModelError(
                "minimum_validated_interval_s must be finite and at least "
                f"{MINIMUM_VALIDATED_INTERVAL_S:g}: {mvi!r}"
            )
        # gpu ramp units
        if d.get("gpu_ramp", {}).get("units") != "Hz":
            raise ModelError(f"gpu_ramp.units must be 'Hz', got {d.get('gpu_ramp', {}).get('units')!r}")
        # NPU is calibrated only for explicitly enumerated requested perf modes.
        # cpu_gpu_only fits no Hexagon term, so it enumerates none: a perf mode
        # it never calibrated would be a claim it cannot support.
        modes = d.get("supported_perf_modes")
        expected_modes = (
            [] if model_profile in (MODEL_PROFILE_CPU_GPU,
                                    MODEL_PROFILE_CPU_GPU_FREQ,
                                    MODEL_PROFILE_NPU_DECLARED)
            else list(SUPPORTED_PERF_MODES)
        )
        if modes != expected_modes:
            raise ModelError(
                f"teaching-power-v2 profile {model_profile!r} requires "
                f"supported_perf_modes={expected_modes!r}; "
                f"got supported_perf_modes={modes!r}"
            )
        dsp_modes = d.get("supported_dsp_power_modes")
        expected_dsp_modes = (
            list(SUPPORTED_DSP_POWER_MODES)
            if model_profile == MODEL_PROFILE_NPU_DSP
            else []
        )
        if dsp_modes != expected_dsp_modes:
            raise ModelError(
                f"teaching-power-v2 profile {model_profile!r} requires "
                f"supported_dsp_power_modes={expected_dsp_modes!r}; "
                f"got supported_dsp_power_modes={dsp_modes!r}"
            )
        source_hashes = d.get("source_session_hashes")
        # A merged Hexagon artifact has two sources (shipped base + extension).
        # cpu_gpu_only has one: the v1 dataset it was translated from, with no
        # extension session in existence to name.
        # cpu_gpu_freq is a standalone campaign on its own board: there is no
        # shipped base session to inherit from, so it names the session it was
        # actually fitted from instead of borrowing another board's provenance.
        if model_profile == MODEL_PROFILE_CPU_GPU_FREQ:
            expected_source_keys = {"calibration_session_sha256"}
        elif model_profile in (MODEL_PROFILE_CPU_GPU, MODEL_PROFILE_NPU_DECLARED):
            expected_source_keys = {"v1_dataset"}
        else:
            expected_source_keys = {"base_session_sha256", "extension_session_sha256"}
        if not isinstance(source_hashes, dict) or set(source_hashes) != expected_source_keys:
            raise ModelError(
                f"profile {model_profile!r} requires source_session_hashes keys "
                f"{sorted(expected_source_keys)}"
            )
        for key, digest in source_hashes.items():
            if not isinstance(digest, str) or len(digest) != 64 or any(
                c not in "0123456789abcdef" for c in digest
            ):
                raise ModelError(f"source_session_hashes.{key} is not lowercase SHA-256")
        if d.get("source_session_hash_algorithm") != SOURCE_SESSION_HASH_ALGORITHM:
            raise ModelError(
                "artifact source_session_hash_algorithm must be "
                f"{SOURCE_SESSION_HASH_ALGORITHM!r}"
            )

        cpu_basis = d.get("cpu_idle_basis")
        if not isinstance(cpu_basis, dict) or cpu_basis.get("semantics") != CPU_IDLE_BASIS_SEMANTICS:
            raise ModelError("artifact cpu_idle_basis has unsupported/missing semantics")
        for key in ("x_gold", "x_prime"):
            try:
                value = float(cpu_basis[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelError(f"artifact cpu_idle_basis.{key} must be numeric") from exc
            if not math.isfinite(value) or value < 0.0:
                raise ModelError(
                    f"artifact cpu_idle_basis.{key} must be finite and nonnegative"
                )

        runtime_provenance = d.get("hexagon_runtime_provenance")
        _validate_hexagon_runtime_provenance(runtime_provenance, model_profile)
        temperature = d.get("calibration_temperature_range")
        if (
            not isinstance(temperature, dict)
            or temperature.get("source") != "observed_raw_telemetry"
            or not isinstance(temperature.get("by_sensor"), dict)
            or not temperature["by_sensor"]
        ):
            raise ModelError(
                "calibration_temperature_range must contain a non-empty observed sensor envelope"
            )
        for sensor, bounds in temperature["by_sensor"].items():
            if not isinstance(sensor, str) or not sensor or not isinstance(bounds, dict):
                raise ModelError("calibration temperature sensor entries are malformed")
            if set(bounds) != {"min_c", "max_c"}:
                raise ModelError(
                    f"calibration temperature sensor {sensor!r} must contain min_c/max_c"
                )
            try:
                lower = float(bounds["min_c"])
                upper = float(bounds["max_c"])
            except (TypeError, ValueError) as exc:
                raise ModelError(
                    f"calibration temperature sensor {sensor!r} bounds must be numeric"
                ) from exc
            if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
                raise ModelError(
                    f"calibration temperature sensor {sensor!r} has invalid bounds"
                )
        known_terms = d.get("known_unmodeled_terms")
        if (
            not isinstance(known_terms, list)
            or any(not isinstance(term, str) or not term for term in known_terms)
            or not MANDATORY_KNOWN_UNMODELED_TERMS.issubset(known_terms)
        ):
            raise ModelError(
                "known_unmodeled_terms must retain every mandatory v2 limitation"
            )
        # Fixed shared-Hexagon semantics must survive serialization unchanged.
        if d.get("hexagon_subsystem") != hexagon_subsystem_for_profile(model_profile):
            raise ModelError(
                "artifact hexagon_subsystem does not declare the required shared, "
                "hardware-serialized, mutually-exclusive path semantics"
            )
        # train/holdout ids: unique and disjoint
        tr = d.get("train_window_ids") or []
        ho = d.get("holdout_window_ids") or []
        if (
            not isinstance(tr, list)
            or not isinstance(ho, list)
            or any(not isinstance(value, str) or not value for value in tr + ho)
        ):
            raise ModelError("train/holdout window ids must be non-empty strings")
        if len(set(tr)) != len(tr) or len(set(ho)) != len(ho) or (set(tr) & set(ho)):
            raise ModelError("train/holdout window ids must be unique and disjoint")
        # the model must have PASSED its acceptance gate to be loadable
        validation = d.get("validation_observed")
        if not isinstance(validation, dict):
            raise ModelError("artifact validation_observed must be an object")
        gate = validation.get("gate")
        checks = gate.get("checks") if isinstance(gate, dict) else None
        required_checks = REQUIRED_GATE_CHECKS[model_profile]
        if (
            not isinstance(checks, dict)
            or set(checks) != required_checks
            or any(type(value) is not bool or not value for value in checks.values())
            or gate.get("passed") is not True
            or gate.get("passed") is not all(checks.values())
        ):
            raise ModelError("artifact validation_observed.gate.passed must be true (gate not passed / report missing)")
        design_hash = validation.get("design_hash")
        if not _is_sha256(design_hash):
            raise ModelError("artifact validation_observed.design_hash must be lowercase SHA-256")
        if validation.get("source_session_hashes") != source_hashes:
            raise ModelError(
                "artifact validation source_session_hashes must match top-level provenance"
            )
        if validation.get("source_session_hash_algorithm") != SOURCE_SESSION_HASH_ALGORITHM:
            raise ModelError("artifact validation source hash algorithm mismatch")
        for metric in ("idle_subtracted_mae_w", "max_total_relative_error"):
            try:
                value = float(validation[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelError(f"artifact validation metric {metric!r} is missing/invalid") from exc
            if not math.isfinite(value) or value < 0.0:
                raise ModelError(f"artifact validation metric {metric!r} must be finite/nonnegative")
        return cls(
            model_version=d["model_version"],
            model_profile=model_profile,
            quiescent_power_w=float(d["quiescent_power_w"]),
            idle_anchor_power_w=[float(x) for x in d["idle_anchor_power_w"]],
            named_coefficients=d["named_coefficients"],
            gpu_ramp=d["gpu_ramp"],
            calibration_control_config=d.get("calibration_control_config", {}),
            train_window_ids=list(d.get("train_window_ids", [])),
            holdout_window_ids=list(d.get("holdout_window_ids", [])),
            validation_observed=d.get("validation_observed", {}),
            known_unmodeled_terms=list(d.get("known_unmodeled_terms", [])),
            minimum_validated_interval_s=float(d.get("minimum_validated_interval_s", 0.0)),
            calibration_board_serial=d.get("calibration_board_serial", ""),
            android_build_fingerprint=d.get("android_build_fingerprint", ""),
            kernel_release=d.get("kernel_release", ""),
            eas_table_hash=d.get("eas_table_hash", ""),
            eas_table=dict(d.get("eas_table", {}) or {}),
            supported_cpu_gpu_opps=d.get("supported_cpu_gpu_opps", {}),
            calibration_temperature_range=d.get("calibration_temperature_range", {}),
            calibration_dataset_hash=d.get("calibration_dataset_hash", ""),
            fit_input_hash=d.get("fit_input_hash", ""),
            source_session_hashes=dict(d.get("source_session_hashes", {})),
            source_session_hash_algorithm=d.get("source_session_hash_algorithm", ""),
            cpu_idle_basis=dict(d.get("cpu_idle_basis", {})),
            hexagon_runtime_provenance=dict(d.get("hexagon_runtime_provenance", {})),
            supported_perf_modes=list(d.get("supported_perf_modes", [])),
            supported_dsp_power_modes=list(d.get("supported_dsp_power_modes", [])),
            hexagon_subsystem=d.get("hexagon_subsystem", {}),
            artifact_schema_version=d["artifact_schema_version"],
            model_equation_version=eqv,
            telemetry_aggregation_version=d.get(
                "telemetry_aggregation_version", TELEMETRY_AGGREGATION_VERSION
            ),
            boundary=d.get("boundary", BOUNDARY),
            scope=d.get("scope", SCOPE),
            quality=d.get("quality", QUALITY),
        )

    @classmethod
    def from_json(cls, text: str) -> "PowerModelArtifact":
        def _reject_const(tok: str) -> float:
            raise ModelError(f"artifact JSON contains non-finite constant {tok!r}")

        d = json.loads(text, parse_constant=_reject_const)
        return cls.from_dict(d)


# --- feature vector & estimate -------------------------------------------------

def feature_vector(
    aggregate: dict[str, Any],
    *,
    model_profile: str = MODEL_PROFILE_NPU_DSP,
    cpu_idle_basis: dict[str, Any] | None = None,
) -> list[float]:
    """Pull the ordered dynamic-feature vector from a host-computed aggregate.

    ``aggregate`` must contain every key in ``FEATURE_ORDER`` (produced by
    ``aggregate.py``). Values must be finite and nonnegative.
    """
    if not isinstance(aggregate, dict):
        raise ModelError(f"aggregate must be an object, got {type(aggregate).__name__}")
    if model_profile == MODEL_PROFILE_NPU_ONLY and "dsp_active" in aggregate:
        try:
            unmodeled_dsp = float(aggregate["dsp_active"])
        except (TypeError, ValueError) as exc:
            raise ModelError("npu-only profile received malformed dsp_active") from exc
        if not math.isfinite(unmodeled_dsp) or unmodeled_dsp < 0.0:
            raise ModelError("npu-only profile received invalid dsp_active")
        if unmodeled_dsp > 0.0:
            raise ModelError(
                "DSP power is unavailable: this artifact was calibrated with the npu-only profile"
            )
    order = feature_order_for_profile(model_profile)
    vec = []
    for key in order:
        if key not in aggregate:
            raise ModelError(f"aggregate missing feature {key!r}")
        try:
            v = float(aggregate[key])
        except (TypeError, ValueError) as exc:
            raise ModelError(f"feature {key!r} is not numeric: {aggregate[key]!r}") from exc
        if not math.isfinite(v):
            raise ModelError(f"feature {key!r} is not finite: {v!r}")
        if v < 0.0:
            raise ModelError(f"feature {key!r} is negative: {v!r}")
        if key in ("x_gold", "x_prime") and cpu_idle_basis is not None:
            if cpu_idle_basis.get("semantics") != CPU_IDLE_BASIS_SEMANTICS:
                raise ModelError("unsupported cpu_idle_basis semantics")
            try:
                idle_value = float(cpu_idle_basis[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelError(f"cpu_idle_basis.{key} is missing/invalid") from exc
            if not math.isfinite(idle_value) or idle_value < 0.0:
                raise ModelError(f"cpu_idle_basis.{key} must be finite/nonnegative")
            v = max(0.0, v - idle_value)
        if key in ACTIVE_FRACTION_FEATURES and v > 1.0 + 1e-9:
            raise ModelError(f"feature {key!r} is not an active fraction in [0,1]: {v!r}")
        vec.append(v)
    activity = vec[order.index("activity_fraction")]
    # cpu_gpu_only carries neither Hexagon column, so both are structurally zero
    # and the mutual-exclusion / union checks below are vacuously satisfied.
    npu = vec[order.index("npu_active")] if "npu_active" in order else 0.0
    dsp = vec[order.index("dsp_active")] if "dsp_active" in order else 0.0
    if npu > 0.0 and dsp > 0.0:
        raise ModelError(
            "npu_active and dsp_active cannot both be nonzero: both paths share and "
            "hardware-serialize on the Hexagon cDSP"
        )
    if npu > activity + 1e-9 or dsp > activity + 1e-9:
        raise ModelError(
            "npu_active and dsp_active cannot exceed the union workload "
            "activity_fraction"
        )
    return vec


def estimate_power(aggregate: dict[str, Any], artifact: PowerModelArtifact) -> dict[str, Any]:
    """Estimate total board power (W) for one aggregated window.

    Returns a dict with ``power_w``, ``dynamic_above_quiescent_w`` and any
    ``warnings`` (out-of-range GPU freq, dynamic-frequency approximation).
    Pure arithmetic — matches the numpy fit's predictions to fp tolerance.
    """
    theta = artifact.theta()
    feats = feature_vector(
        aggregate,
        model_profile=artifact.model_profile,
        cpu_idle_basis=artifact.cpu_idle_basis,
    )
    dynamic = sum(t * f for t, f in zip(theta, feats))
    if dynamic < 0.0:  # guard; nonneg coeffs*features should preclude this
        dynamic = 0.0
    power = artifact.quiescent_power_w + dynamic

    warnings: list[str] = []
    gpu_freq = aggregate.get("gpu_freq_hz")
    if aggregate.get("gpu_active") and gpu_freq is not None:
        if not gpu_freq_in_range(float(gpu_freq), artifact.knots_hz()):
            warnings.append(
                f"gpu_freq_out_of_calibrated_range ({gpu_freq} Hz outside "
                f"{artifact.knots_hz()})"
            )
    if aggregate.get("gpu_dynamic_frequency_approximation"):
        warnings.append("gpu_dynamic_frequency_approximation=true")
    return {
        "power_w": power,
        "dynamic_above_quiescent_w": dynamic,
        "warnings": warnings,
    }


def integrate_energy(window_aggregates: list[dict[str, Any]], artifact: PowerModelArtifact) -> float:
    """Energy (J) = sum_k P(features_k) * dt_k over DISJOINT stationary windows.

    Each aggregate must carry ``window_s`` and monotonic ``t_start``/``t_end``.
    Windows must be non-overlapping and each at least ``minimum_validated_interval_s``
    long (review P0-16/finding 8+9). Empty input is an error, not zero energy.
    """
    if not window_aggregates:
        raise ModelError("no windows to integrate — refusing to emit zero energy")
    min_s = float(artifact.minimum_validated_interval_s or 0.0)
    spans = []
    for agg in window_aggregates:
        for k in ("window_s", "t_start", "t_end"):
            if k not in agg:
                raise ModelError(f"aggregate missing {k!r} for energy integration")
        wsec = float(agg["window_s"])
        t0, t1 = float(agg["t_start"]), float(agg["t_end"])
        if not (math.isfinite(wsec) and math.isfinite(t0) and math.isfinite(t1)):
            raise ModelError("non-finite window timing")
        span = t1 - t0
        if span <= 0.0:
            raise ModelError(f"non-positive window span {t0}->{t1}")
        # dt is DERIVED FROM THE SPAN, not the independent window_s; the two must
        # agree within a small tolerance or the aggregate is inconsistent (finding).
        if abs(span - wsec) > max(1.0, 0.1 * span):
            raise ModelError(f"window_s ({wsec}) disagrees with span ({span:.3f}) — inconsistent aggregate")
        if span < min_s:
            raise ModelError(f"window {span:.1f}s < minimum_validated_interval_s {min_s:.1f}s — out of validated scope")
        spans.append((t0, t1, agg))
    spans.sort(key=lambda s: s[0])
    for (a0, a1, _), (b0, _b1, _b) in zip(spans, spans[1:]):
        if b0 < a1:
            raise ModelError(f"overlapping windows [{a0},{a1}] and starting {b0} — energy would double-count")
    return sum(estimate_power(agg, artifact)["power_w"] * (t1 - t0) for t0, t1, agg in spans)
