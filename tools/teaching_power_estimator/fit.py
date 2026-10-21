"""HDK8750 teaching power estimator — fitting & evaluation (NumPy; OFFLINE only).

This is the only module that imports NumPy. Keep it out of the estimate path
(``model.py`` / ``aggregate.py`` stay stdlib so the estimate can run anywhere).

Responsibilities:
  * ``exhaustive_nnls`` — exact nonnegative least squares by active-face
    enumeration (8 coefficients -> 2**8 subsets). Deterministic.
  * ``fit_power_model`` — quiescent constant + 8 nonnegative dynamic coeffs;
    requires full column rank or fails (no silent arbitrary coefficients).
    Holdout windows never enter the fit.
  * ``evaluate_power_model`` — full-model vs executable baselines on the
    preselected holdout; returns a report with the acceptance gate.

Baselines and the gate follow the approved plan / review P0-12.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from typing import Any

import numpy as np

from .model import (
    MODEL_PROFILE_CPU_GPU_FREQ,
    COEFFICIENT_NAMES,
    COEFFICIENT_UNITS,
    FEATURE_ORDER,
    CPU_IDLE_BASIS_SEMANTICS,
    MANDATORY_KNOWN_UNMODELED_TERMS,
    MODEL_PROFILE_CPU_GPU,
    MODEL_PROFILE_NPU_DSP,
    MODEL_PROFILE_NPU_ONLY,
    MODEL_EQUATION_VERSION,
    SOURCE_SESSION_HASH_ALGORITHM,
    TELEMETRY_AGGREGATION_VERSION,
    ModelError,
    PowerModelArtifact,
    coefficient_names_for_profile,
    feature_vector,
    feature_order_for_profile,
    hexagon_subsystem_for_profile,
)

# Max fractional spread across the 3 idle anchors before we refuse to fit.
IDLE_DRIFT_REJECT_FRAC = 0.20

# Locked before the v2 fit is observed.  The v1 observations are retained for
# provenance; the acceptance limits are the predeclared rounded bounds from the
# v2 calibration plan (0.402 W and 8.3%).
V1_CPU_GPU_HOLDOUT_MAE_W = 0.3016397082237301
V1_CPU_GPU_MAX_REL_ERROR = 0.06311514417987642
V1_ARTIFACT_SHA256 = "12a0928f5609c31986cc264076ac9852d94867151218a869a0a7983c997a0db1"
V1_DATASET_SHA256 = "041b4710acc7bc10e412465adf9866387c426188d76a81c80de5082aee51c0dc"
MAX_CPU_GPU_HOLDOUT_MAE_W = 0.402
MAX_CPU_GPU_HOLDOUT_REL_ERROR = 0.083
MAX_HEXAGON_HOLDOUT_REL_ERROR = 0.25


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_experiment_design(
    idle: list[dict[str, Any]],
    train: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    design: dict[str, Any] | None,
) -> None:
    """Fail-closed check that the collected split matches the predeclared design.

    Always enforces disjoint window-id sets. When ``design`` is provided (from
    the locked manifest) also enforces idle-anchor count, the exact predeclared
    holdout id set, the CPU/GPU/NPU/DSP holdout composition, and any
    predeclared NPU/DSP training counts.
    """
    groups = [("idle", idle), ("train", train), ("holdout", holdout)]
    seen: dict[str, str] = {}
    for gname, grp in groups:
        for w in grp:
            wid = w["window_id"]
            if wid in seen:
                raise ModelError(f"window_id {wid!r} appears in both {seen[wid]} and {gname} — sets must be disjoint")
            seen[wid] = gname
    if not design:
        return
    if "idle_anchors" in design and len(idle) != design["idle_anchors"]:
        raise ModelError(f"expected {design['idle_anchors']} idle anchors, got {len(idle)}")
    want = set(design.get("holdout_window_ids", []))
    if want and {w["window_id"] for w in holdout} != want:
        raise ModelError(f"holdout set {{{', '.join(sorted(w['window_id'] for w in holdout))}}} != predeclared {sorted(want)}")
    hcpu = sum(1 for w in holdout if w.get("engine_class") == "cpu")
    hgpu = sum(1 for w in holdout if w.get("engine_class") == "gpu")
    hnpu = sum(1 for w in holdout if w.get("engine_class") == "npu")
    hdsp = sum(1 for w in holdout if w.get("engine_class") == "dsp")
    if "holdout_cpu" in design and hcpu != design["holdout_cpu"]:
        raise ModelError(f"expected {design['holdout_cpu']} CPU holdouts, got {hcpu}")
    if "holdout_gpu" in design and hgpu != design["holdout_gpu"]:
        raise ModelError(f"expected {design['holdout_gpu']} GPU holdouts, got {hgpu}")
    if "holdout_npu" in design and hnpu != design["holdout_npu"]:
        raise ModelError(f"expected {design['holdout_npu']} NPU holdouts, got {hnpu}")
    if "holdout_dsp" in design and hdsp != design["holdout_dsp"]:
        raise ModelError(f"expected {design['holdout_dsp']} DSP holdouts, got {hdsp}")
    tnpu = sum(1 for w in train if w.get("engine_class") == "npu")
    tdsp = sum(1 for w in train if w.get("engine_class") == "dsp")
    if "train_npu" in design and tnpu != design["train_npu"]:
        raise ModelError(f"expected {design['train_npu']} NPU training windows, got {tnpu}")
    if "train_dsp" in design and tdsp != design["train_dsp"]:
        raise ModelError(f"expected {design['train_dsp']} DSP training windows, got {tdsp}")


# --- exact nonnegative least squares ------------------------------------------

def exhaustive_nnls(X: np.ndarray, y: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, float]:
    """Exact NNLS for a small number of columns via active-face enumeration.

    Enumerate every subset of active columns; solve each with ``lstsq``; keep
    subsets whose solved coefficients are all >= -tol (clip tiny negatives to
    0); return the minimum-residual feasible solution. Deterministic
    tie-break: fewer active columns first, then lexicographically smallest
    subset. The all-zero solution is the fallback.
    """
    n, p = X.shape
    best_theta = np.zeros(p)
    best_res = float(np.sum((X @ best_theta - y) ** 2))
    best_key: tuple[int, tuple[int, ...]] = (0, ())
    for r in range(1, p + 1):
        for subset in itertools.combinations(range(p), r):
            cols = list(subset)
            sol, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
            if np.any(sol < -tol):
                continue
            theta = np.zeros(p)
            theta[cols] = np.clip(sol, 0.0, None)
            res = float(np.sum((X @ theta - y) ** 2))
            key = (r, subset)
            if res < best_res - 1e-12 or (abs(res - best_res) <= 1e-12 and key < best_key):
                best_res, best_theta, best_key = res, theta, key
    return best_theta, best_res


# --- design matrix helpers -----------------------------------------------------

def _design_matrix(
    windows: list[dict[str, Any]],
    model_profile: str,
    cpu_idle_basis: dict[str, Any],
) -> np.ndarray:
    return np.array(
        [
            feature_vector(
                w["aggregate"],
                model_profile=model_profile,
                cpu_idle_basis=cpu_idle_basis,
            )
            for w in windows
        ],
        dtype=float,
    )


def _measured(windows: list[dict[str, Any]]) -> np.ndarray:
    try:
        values = np.array([float(w["measured_power_w"]) for w in windows], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelError("every fitted/evaluated window needs numeric measured_power_w") from exc
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ModelError("measured_power_w values must be finite and positive")
    return values


# --- fit -----------------------------------------------------------------------

def fit_power_model(
    train_windows: list[dict[str, Any]],
    idle_windows: list[dict[str, Any]],
    session_meta: dict[str, Any],
    model_spec: dict[str, Any],
) -> PowerModelArtifact:
    """Fit P_q (observed constant) + 8 nonnegative dynamic coefficients.

    ``train_windows`` / ``idle_windows`` each: {window_id, aggregate, measured_power_w[, engine_class]}.
    ``session_meta`` carries board/build identity + hashes + EAS hash + temp/OPP ranges.
    ``model_spec`` carries knots_hz, holdout_window_ids, minimum_validated_interval_s, known_unmodeled_terms.
    """
    if len(idle_windows) < 1:
        raise ModelError("need at least one idle anchor to set quiescent power")
    idle_p = sorted(float(w["measured_power_w"]) for w in idle_windows)
    p_q = float(statistics.median(idle_p))
    if p_q <= 0.0:
        raise ModelError(f"nonpositive quiescent power {p_q!r}")
    spread = (max(idle_p) - min(idle_p)) / p_q
    if spread > IDLE_DRIFT_REJECT_FRAC:
        raise ModelError(
            f"idle anchors drift {spread:.1%} (> {IDLE_DRIFT_REJECT_FRAC:.0%}) — "
            f"session unstable, refuse to fit: {idle_p}"
        )

    model_profile = model_spec.get("model_profile", MODEL_PROFILE_NPU_DSP)
    coefficient_names = coefficient_names_for_profile(model_profile)
    feature_order = feature_order_for_profile(model_profile)
    extra_terms = model_spec.get("known_unmodeled_terms") or []
    if not isinstance(extra_terms, list) or any(
        not isinstance(term, str) or not term for term in extra_terms
    ):
        raise ModelError("known_unmodeled_terms additions must be non-empty strings")
    cpu_idle_basis = {
        "semantics": CPU_IDLE_BASIS_SEMANTICS,
        "x_gold": float(statistics.median(float(w["aggregate"]["x_gold"]) for w in idle_windows)),
        "x_prime": float(statistics.median(float(w["aggregate"]["x_prime"]) for w in idle_windows)),
    }

    if len(train_windows) < len(coefficient_names):
        raise ModelError(
            f"need >= {len(coefficient_names)} training windows, got {len(train_windows)}"
        )
    X = _design_matrix(train_windows, model_profile, cpu_idle_basis)
    y = _measured(train_windows) - p_q
    rank = int(np.linalg.matrix_rank(X))
    if rank < len(coefficient_names):
        raise ModelError(
            f"design matrix rank-deficient (rank {rank} < {len(coefficient_names)}); "
            f"calibration lacks independent contrast — fit fails rather than guess"
        )
    theta, _res = exhaustive_nnls(X, y)

    named = {
        name: {"value": float(theta[i]), "units": COEFFICIENT_UNITS[name]}
        for i, name in enumerate(coefficient_names)
    }
    knots = [float(k) for k in model_spec["knots_hz"]]

    fit_input_blob = json.dumps(
        {
            "calibration_dataset_hash": session_meta.get("calibration_dataset_hash", ""),
            "train_window_ids": [w["window_id"] for w in train_windows],
            "knots_hz": knots,
            "eas_table_hash": session_meta.get("eas_table_hash", ""),
            "telemetry_aggregation_version": TELEMETRY_AGGREGATION_VERSION,
            "model_equation_version": MODEL_EQUATION_VERSION,
            "design_profile": model_spec.get("design_profile"),
            "supported_perf_modes": list(model_spec.get("supported_perf_modes", [])),
            "supported_dsp_power_modes": list(
                model_spec.get("supported_dsp_power_modes", [])
            ),
            "source_session_hashes": session_meta.get("source_session_hashes", {}),
            "design_hash": model_spec.get("design_hash"),
            "model_profile": model_profile,
            "feature_order": feature_order,
            "cpu_idle_basis": cpu_idle_basis,
            "legacy_cpu_gpu_acceptance": {
                "v1_artifact_sha256": V1_ARTIFACT_SHA256,
                "v1_dataset_sha256": V1_DATASET_SHA256,
                "maximum_holdout_mae_w": MAX_CPU_GPU_HOLDOUT_MAE_W,
                "maximum_holdout_relative_error": MAX_CPU_GPU_HOLDOUT_REL_ERROR,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    return PowerModelArtifact(
        model_version=model_spec.get("model_version", "sm8750-teaching-v2"),
        model_profile=model_profile,
        quiescent_power_w=p_q,
        idle_anchor_power_w=idle_p,
        named_coefficients=named,
        gpu_ramp={"knots_hz": knots, "units": "Hz"},
        calibration_control_config=session_meta.get("calibration_control_config", {}),
        train_window_ids=[w["window_id"] for w in train_windows],
        holdout_window_ids=list(model_spec.get("holdout_window_ids", [])),
        validation_observed={},  # filled by evaluate_power_model
        known_unmodeled_terms=sorted(
            MANDATORY_KNOWN_UNMODELED_TERMS
            | set(extra_terms)
        ),
        minimum_validated_interval_s=float(model_spec.get("minimum_validated_interval_s", 0.0)),
        calibration_board_serial=session_meta.get("calibration_board_serial", ""),
        android_build_fingerprint=session_meta.get("android_build_fingerprint", ""),
        kernel_release=session_meta.get("kernel_release", ""),
        eas_table_hash=session_meta.get("eas_table_hash", ""),
        supported_cpu_gpu_opps=session_meta.get("supported_cpu_gpu_opps", {}),
        calibration_temperature_range=session_meta.get("calibration_temperature_range", {}),
        calibration_dataset_hash=session_meta.get("calibration_dataset_hash", ""),
        fit_input_hash=_sha256_hex(fit_input_blob),
        source_session_hashes=dict(session_meta.get("source_session_hashes", {})),
        source_session_hash_algorithm=session_meta.get(
            "source_session_hash_algorithm", SOURCE_SESSION_HASH_ALGORITHM
        ),
        cpu_idle_basis=cpu_idle_basis,
        hexagon_runtime_provenance=dict(
            session_meta.get("hexagon_runtime_provenance", {})
        ),
        supported_perf_modes=list(model_spec.get("supported_perf_modes", [])),
        supported_dsp_power_modes=list(
            model_spec.get("supported_dsp_power_modes", [])
        ),
        hexagon_subsystem=hexagon_subsystem_for_profile(model_profile),
    )


# --- baselines (fit from TRAIN only) ------------------------------------------

def _fit_idle_active(train: list[dict[str, Any]], p_q: float) -> float:
    a = np.array([float(w["aggregate"]["activity_fraction"]) for w in train])
    dyn = _measured(train) - p_q
    denom = float(np.sum(a * a))
    return max(0.0, float(np.sum(a * dyn) / denom)) if denom > 0 else 0.0


def _fit_cpu_eas_only(
    train_cpu: list[dict[str, Any]],
    p_q: float,
    cpu_idle_basis: dict[str, Any],
) -> tuple[float, float]:
    """Reduced CPU model P_q + F*a + g*(x_gold+x_prime), fitted on train CPU."""
    if not train_cpu:
        return 0.0, 0.0
    A = np.array(
        [
            [
                float(w["aggregate"]["activity_fraction"]),
                max(
                    0.0,
                    float(w["aggregate"]["x_gold"])
                    - float(cpu_idle_basis["x_gold"]),
                )
                + max(
                    0.0,
                    float(w["aggregate"]["x_prime"])
                    - float(cpu_idle_basis["x_prime"]),
                ),
            ]
            for w in train_cpu
        ]
    )
    dyn = _measured(train_cpu) - p_q
    theta, _ = exhaustive_nnls(A, dyn)
    return float(theta[0]), float(theta[1])


def _fit_gpu_active_const(train_gpu: list[dict[str, Any]], p_q: float) -> float:
    if not train_gpu:
        return 0.0
    return max(0.0, float(np.mean(_measured(train_gpu) - p_q)))


def _fit_path_active_const(
    train_path: list[dict[str, Any]], p_q: float, feature_name: str
) -> float:
    """Fit nonnegative ``P_q + c*path_active`` from one Hexagon path's train rows."""
    if not train_path:
        return 0.0
    active = np.array([float(w["aggregate"][feature_name]) for w in train_path])
    dyn = _measured(train_path) - p_q
    denom = float(np.sum(active * active))
    if denom <= 0.0:
        return 0.0
    return max(0.0, float(np.sum(active * dyn) / denom))


def _shared_hexagon_design(
    windows: list[dict[str, Any]], artifact: PowerModelArtifact
) -> np.ndarray:
    """Nested seven-column model: v1 columns plus one pooled ``N + D`` step."""
    rows = []
    for window in windows:
        aggregate = window["aggregate"]
        vector = feature_vector(
            aggregate,
            model_profile=artifact.model_profile,
            cpu_idle_basis=artifact.cpu_idle_basis,
        )
        rows.append(vector[:6] + [vector[6] + vector[7]])
    return np.array(rows, dtype=float)


def _fit_shared_hexagon_model(
    train: list[dict[str, Any]], artifact: PowerModelArtifact
) -> np.ndarray:
    """Fit the fair reduced alternative to two separate Hexagon path steps.

    This keeps the activity floor, CPU EAS shapes, and GPU ramp, replacing only
    the two path columns with one pooled ``hexagon_active = N + D`` column. A
    path-only ``P_q + c*(N+D)`` comparison would let the full model win merely
    through inherited CPU/GPU/floor terms and would not test whether two path
    coefficients add value.
    """
    if artifact.model_profile != MODEL_PROFILE_NPU_DSP:
        raise ModelError("pooled NPU+DSP baseline is defined only for npu_dsp profile")
    matrix = _shared_hexagon_design(train, artifact)
    if int(np.linalg.matrix_rank(matrix)) != 7:
        raise ModelError("pooled Hexagon baseline matrix is not full rank (expected 7)")
    theta, _ = exhaustive_nnls(matrix, _measured(train) - artifact.quiescent_power_w)
    return theta


def _mae(pred: np.ndarray, meas: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - meas))) if len(meas) else float("nan")


def _predict_full(artifact: PowerModelArtifact, windows: list[dict[str, Any]]) -> np.ndarray:
    theta = np.array(artifact.theta())
    X = _design_matrix(windows, artifact.model_profile, artifact.cpu_idle_basis)
    return artifact.quiescent_power_w + X @ theta


def evaluate_power_model(
    artifact: PowerModelArtifact,
    holdout_windows: list[dict[str, Any]],
    train_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Full model vs executable baselines on the preselected holdout.

    Baselines are fit from TRAIN only. Returns a report incl. the gate.
    """
    p_q = artifact.quiescent_power_w
    meas_h = _measured(holdout_windows)
    pred_full = _predict_full(artifact, holdout_windows)

    # baselines (train-fit)
    const_total = float(np.mean(_measured(train_windows)))
    c_active = _fit_idle_active(train_windows, p_q)
    train_cpu = [w for w in train_windows if w.get("engine_class") == "cpu"]
    train_gpu = [w for w in train_windows if w.get("engine_class") == "gpu"]
    train_npu = [w for w in train_windows if w.get("engine_class") == "npu"]
    train_dsp = [w for w in train_windows if w.get("engine_class") == "dsp"]
    eas_F, eas_g = _fit_cpu_eas_only(train_cpu, p_q, artifact.cpu_idle_basis)
    c_gpu = _fit_gpu_active_const(train_gpu, p_q)
    c_npu = _fit_path_active_const(train_npu, p_q, "npu_active")
    c_dsp = _fit_path_active_const(train_dsp, p_q, "dsp_active")
    shared_hexagon_theta = (
        _fit_shared_hexagon_model(train_windows, artifact)
        if artifact.model_profile == MODEL_PROFILE_NPU_DSP
        else None
    )

    # holdout subsets by engine class
    hold_cpu = [w for w in holdout_windows if w.get("engine_class") == "cpu"]
    hold_gpu = [w for w in holdout_windows if w.get("engine_class") == "gpu"]
    hold_npu = [w for w in holdout_windows if w.get("engine_class") == "npu"]
    hold_dsp = [w for w in holdout_windows if w.get("engine_class") == "dsp"]

    def const_pred(ws):
        return np.full(len(ws), const_total)

    def idle_active_pred(ws):
        a = np.array([float(w["aggregate"]["activity_fraction"]) for w in ws])
        return p_q + c_active * a

    def eas_only_pred(ws):
        activity = np.array(
            [float(w["aggregate"]["activity_fraction"]) for w in ws]
        )
        xs = np.array(
            [
                max(
                    0.0,
                    float(w["aggregate"]["x_gold"])
                    - float(artifact.cpu_idle_basis["x_gold"]),
                )
                + max(
                    0.0,
                    float(w["aggregate"]["x_prime"])
                    - float(artifact.cpu_idle_basis["x_prime"]),
                )
                for w in ws
            ]
        )
        return p_q + eas_F * activity + eas_g * xs

    def gpu_const_pred(ws):
        ind = np.array([1.0 if w["aggregate"].get("gpu_active") else 0.0 for w in ws])
        return p_q + c_gpu * ind

    def path_const_pred(ws, feature_name, coefficient):
        active = np.array([float(w["aggregate"][feature_name]) for w in ws])
        return p_q + coefficient * active

    def shared_hexagon_pred(ws):
        if shared_hexagon_theta is None:
            raise ModelError("shared Hexagon baseline unavailable for npu-only profile")
        return p_q + _shared_hexagon_design(ws, artifact) @ shared_hexagon_theta

    if not holdout_windows:
        raise ModelError("no holdout windows to evaluate")
    evaluated_ids = {w["window_id"] for w in holdout_windows}
    ids_match = (not artifact.holdout_window_ids) or (set(artifact.holdout_window_ids) == evaluated_ids)

    full_mae = _mae(pred_full, meas_h)
    report_baselines = {
        "constant_total": {"value_w": const_total, "holdout_mae_w": _mae(const_pred(holdout_windows), meas_h)},
        "idle_plus_active": {"c_w": c_active, "holdout_mae_w": _mae(idle_active_pred(holdout_windows), meas_h)},
    }
    # v2 scope is CPU+GPU+NPU+DSP: every holdout class must be present.  CPU and
    # GPU retain their established class gates.  NPU/DSP path constants are
    # descriptive only: on pure steady-state path rows they are functionally
    # equivalent to the corresponding v2 step.  The meaningful reduced Hexagon
    # baseline is one pooled shared-active constant across both paths.
    gates: dict[str, bool] = {
        "holdout_ids_match": bool(ids_match),
        "has_cpu_holdout": bool(hold_cpu),
        "has_gpu_holdout": bool(hold_gpu),
        "beats_constant": full_mae < report_baselines["constant_total"]["holdout_mae_w"],
        "beats_idle_plus_active": full_mae < report_baselines["idle_plus_active"]["holdout_mae_w"],
    }
    # The cpu_gpu_only profile has no Hexagon terms at all (ADR 0022), so the
    # NPU gates do not apply to it -- asserting them would demand a coefficient
    # the profile never defines.
    if artifact.model_profile not in (MODEL_PROFILE_CPU_GPU, MODEL_PROFILE_CPU_GPU_FREQ):
        gates["has_npu_holdout"] = bool(hold_npu)
        gates["npu_coefficient_positive"] = (
            artifact.named_coefficients["dP_npu"]["value"] > 1e-12
        )
    if artifact.model_profile == MODEL_PROFILE_NPU_DSP:
        gates["has_dsp_holdout"] = bool(hold_dsp)
        gates["dsp_coefficient_positive"] = (
            artifact.named_coefficients["dP_dsp"]["value"] > 1e-12
        )
    elif artifact.model_profile not in (MODEL_PROFILE_CPU_GPU, MODEL_PROFILE_CPU_GPU_FREQ) and (hold_dsp or train_dsp):
        raise ModelError("npu-only profile cannot contain DSP train/holdout windows")
    engine_metrics: dict[str, dict[str, Any]] = {}
    if hold_cpu:
        meas_cpu = _measured(hold_cpu)
        pred_cpu = _predict_full(artifact, hold_cpu)
        full_cpu = _mae(pred_cpu, meas_cpu)
        cpu_rel = float(np.max(np.abs(pred_cpu - meas_cpu) / meas_cpu))
        eas_cpu = _mae(eas_only_pred(hold_cpu), _measured(hold_cpu))
        report_baselines["cpu_eas_only"] = {"F_w": eas_F, "g": eas_g, "holdout_mae_w": eas_cpu}
        engine_metrics["cpu"] = {
            "n_holdout": len(hold_cpu),
            "holdout_mae_w": full_cpu,
            "max_total_relative_error": cpu_rel,
            "holdout_window_ids": [w["window_id"] for w in hold_cpu],
        }
        gates["cpu_beats_eas_only"] = full_cpu < eas_cpu
    if hold_gpu:
        meas_gpu = _measured(hold_gpu)
        pred_gpu = _predict_full(artifact, hold_gpu)
        full_gpu = _mae(pred_gpu, meas_gpu)
        gpu_rel = float(np.max(np.abs(pred_gpu - meas_gpu) / meas_gpu))
        gpu_c = _mae(gpu_const_pred(hold_gpu), meas_gpu)
        report_baselines["gpu_active_constant"] = {"c_w": c_gpu, "holdout_mae_w": gpu_c}
        engine_metrics["gpu"] = {
            "n_holdout": len(hold_gpu),
            "holdout_mae_w": full_gpu,
            "max_total_relative_error": gpu_rel,
            "holdout_window_ids": [w["window_id"] for w in hold_gpu],
        }
        gates["gpu_beats_active_constant"] = full_gpu < gpu_c
    if hold_npu:
        meas_npu = _measured(hold_npu)
        pred_npu = _predict_full(artifact, hold_npu)
        pred_npu_c = path_const_pred(hold_npu, "npu_active", c_npu)
        full_npu = _mae(pred_npu, meas_npu)
        npu_c = _mae(pred_npu_c, meas_npu)
        npu_rel = float(np.max(np.abs(pred_npu - meas_npu) / meas_npu))
        report_baselines["npu_active_constant"] = {
            "c_w": c_npu,
            "train_window_ids": [w["window_id"] for w in train_npu],
            "holdout_window_ids": [w["window_id"] for w in hold_npu],
            "full_prediction_w": [float(v) for v in pred_npu],
            "baseline_prediction_w": [float(v) for v in pred_npu_c],
            "measured_w": [float(v) for v in meas_npu],
            "full_holdout_mae_w": full_npu,
            "holdout_mae_w": npu_c,
            "full_max_relative_error": npu_rel,
        }
        engine_metrics["npu"] = {
            "n_holdout": len(hold_npu),
            "holdout_mae_w": full_npu,
            "max_total_relative_error": npu_rel,
            "holdout_window_ids": [w["window_id"] for w in hold_npu],
        }
        gates["npu_within_25pct_observed"] = npu_rel <= MAX_HEXAGON_HOLDOUT_REL_ERROR
    if hold_dsp and artifact.model_profile == MODEL_PROFILE_NPU_DSP:
        meas_dsp = _measured(hold_dsp)
        pred_dsp = _predict_full(artifact, hold_dsp)
        pred_dsp_c = path_const_pred(hold_dsp, "dsp_active", c_dsp)
        full_dsp = _mae(pred_dsp, meas_dsp)
        dsp_c = _mae(pred_dsp_c, meas_dsp)
        dsp_rel = float(np.max(np.abs(pred_dsp - meas_dsp) / meas_dsp))
        report_baselines["dsp_active_constant"] = {
            "c_w": c_dsp,
            "train_window_ids": [w["window_id"] for w in train_dsp],
            "holdout_window_ids": [w["window_id"] for w in hold_dsp],
            "full_prediction_w": [float(v) for v in pred_dsp],
            "baseline_prediction_w": [float(v) for v in pred_dsp_c],
            "measured_w": [float(v) for v in meas_dsp],
            "full_holdout_mae_w": full_dsp,
            "holdout_mae_w": dsp_c,
            "full_max_relative_error": dsp_rel,
        }
        engine_metrics["dsp"] = {
            "n_holdout": len(hold_dsp),
            "holdout_mae_w": full_dsp,
            "max_total_relative_error": dsp_rel,
            "holdout_window_ids": [w["window_id"] for w in hold_dsp],
        }
        gates["dsp_within_25pct_observed"] = dsp_rel <= MAX_HEXAGON_HOLDOUT_REL_ERROR

    hold_hexagon = hold_npu + hold_dsp
    if hold_hexagon and artifact.model_profile == MODEL_PROFILE_NPU_DSP:
        meas_hex = _measured(hold_hexagon)
        pred_hex = shared_hexagon_pred(hold_hexagon)
        shared_mae = _mae(pred_hex, meas_hex)
        full_hex_mae = _mae(_predict_full(artifact, hold_hexagon), meas_hex)
        report_baselines["hexagon_active_constant"] = {
            "model": "v1_dynamic_terms_plus_one_pooled_hexagon_active_step",
            "pooled_hexagon_c_w": float(shared_hexagon_theta[-1]),
            "theta": [float(value) for value in shared_hexagon_theta],
            "train_window_ids": [w["window_id"] for w in train_windows],
            "holdout_window_ids": [w["window_id"] for w in hold_hexagon],
            "holdout_mae_w": shared_mae,
            "full_holdout_mae_w": full_hex_mae,
        }
        gates["hexagon_beats_shared_active_constant"] = full_hex_mae < shared_mae

    legacy_holdout = hold_cpu + hold_gpu
    legacy_mae = float("nan")
    legacy_max_rel = float("nan")
    if legacy_holdout:
        legacy_meas = _measured(legacy_holdout)
        legacy_pred = _predict_full(artifact, legacy_holdout)
        legacy_mae = _mae(legacy_pred, legacy_meas)
        legacy_max_rel = float(np.max(np.abs(legacy_pred - legacy_meas) / legacy_meas))
    gates["cpu_gpu_mae_no_material_regression"] = bool(
        legacy_holdout
        and legacy_mae <= MAX_CPU_GPU_HOLDOUT_MAE_W
    )
    gates["cpu_gpu_max_rel_no_material_regression"] = bool(
        legacy_holdout
        and legacy_max_rel <= MAX_CPU_GPU_HOLDOUT_REL_ERROR
    )

    max_rel = float(np.max(np.abs(pred_full - meas_h) / meas_h)) if len(meas_h) else float("nan")
    passed = all(gates.values())

    return {
        "n_holdout": len(holdout_windows),
        "idle_subtracted_mae_w": full_mae,
        "max_total_relative_error": max_rel,
        "is_prediction_interval": False,
        "note": "Descriptive calibration result; no coverage guarantee.",
        "legacy_cpu_gpu_regression": {
            "holdout_mae_w": legacy_mae,
            "max_total_relative_error": legacy_max_rel,
            "v1_reference_mae_w": V1_CPU_GPU_HOLDOUT_MAE_W,
            "v1_reference_max_total_relative_error": V1_CPU_GPU_MAX_REL_ERROR,
            "v1_artifact_sha256": V1_ARTIFACT_SHA256,
            "v1_dataset_sha256": V1_DATASET_SHA256,
            "maximum_holdout_mae_w": MAX_CPU_GPU_HOLDOUT_MAE_W,
            "maximum_holdout_relative_error": MAX_CPU_GPU_HOLDOUT_REL_ERROR,
        },
        "engine_metrics": engine_metrics,
        "baselines": report_baselines,
        "gate": {"passed": passed, "checks": gates},
    }
