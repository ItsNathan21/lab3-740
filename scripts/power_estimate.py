"""Ballpark total-board power/energy for a run, from the teaching model.

Run from project root:
    python scripts/power_estimate.py <artifact.json> <run_telemetry.json>

STDLIB ONLY (no numpy) -- the estimate path is portable. Emits the
output-contract JSON: a modeled, opt-in, **board-specific**, whole-system 12 V
supply-input power estimate. NOT a measurement, NOT a physical per-engine split.

Fail-closed guards (review findings 9/10):
  * NPU/DSP input requires the exact calibrated runtime/client hashes and live mode proof.
  * simultaneous NPU+DSP intent -> `unavailable` (shared cDSP serializes).
  * throttled or temperature-out-of-envelope Hexagon windows -> `unavailable`.
  * EAS-hash / board-serial / build / kernel mismatch -> out-of-distribution warnings.
  * windows shorter than the validated minimum, or missing device identity, -> warnings.
  * empty input -> error (never a spurious zero).
See decisions/0019-teaching-power-estimate.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.teaching_power_estimator import aggregate as A  # noqa: E402
from tools.teaching_power_estimator.model import (  # noqa: E402
    ModelError,
    MODEL_PROFILE_NPU_ONLY,
    PowerModelArtifact,
    estimate_power,
    integrate_energy,
    )


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


_HEXAGON_COOLING_TYPES = {"cdsp_sw_hvx", "cdsp_sw_hmx"}
_HEXAGON_TEMP_PREFIXES = ("nsphvx-", "nsphmx-")


def _unavailable(
    artifact: PowerModelArtifact,
    reason: str,
    offending_window_id: str | None = None,
    ) -> dict:
    """Return the same honest boundary/provenance contract even without a number."""
    out = {
        "kind": "unavailable",
        "quality": artifact.quality,
        "reason": reason,
        "board_specific": True,
        "calibration_board_serial": artifact.calibration_board_serial,
        "boundary": artifact.boundary,
        "scope": artifact.scope,
        "model_version": artifact.model_version,
        "model_profile": artifact.model_profile,
        "physical_engine_attribution": False,
        "concurrency_validated": False,
        "supported_perf_modes": artifact.supported_perf_modes,
        "supported_dsp_power_modes": artifact.supported_dsp_power_modes,
        "hexagon_subsystem": artifact.hexagon_subsystem,
        "known_unmodeled_terms": artifact.known_unmodeled_terms,
        "provenance": {
            "calibration_dataset_hash": artifact.calibration_dataset_hash,
            "fit_input_hash": artifact.fit_input_hash,
            "design_hash": artifact.validation_observed.get("design_hash"),
            "source_session_hashes": artifact.source_session_hashes,
        },
    }
    if offending_window_id is not None:
        out["offending_window_id"] = offending_window_id
    return out


def _hexagon_runtime_reason(
    artifact: PowerModelArtifact,
    condition: dict,
    *,
    npu_active: bool,
    dsp_active: bool,
        ) -> str | None:
    """Return why this exact runtime population is outside calibration, if any."""
    calibrated = artifact.hexagon_runtime_provenance

    if dsp_active and artifact.model_profile == MODEL_PROFILE_NPU_ONLY:
        return "DSP is unavailable: this artifact was fit with model_profile='npu_only'"

    if npu_active:
        expected = calibrated.get("npu")
        if not isinstance(expected, dict):
            return "artifact lacks calibrated NPU runtime provenance"
        perf_mode = condition.get("perf_mode")
        if perf_mode not in artifact.supported_perf_modes:
            return (
                f"NPU perf_mode {perf_mode!r} is not calibrated; "
                f"supported={artifact.supported_perf_modes!r}"
            )
        comparisons = (
            ("perf_mode", perf_mode, expected.get("perf_mode")),
            ("QAIRT build ID", condition.get("qairt_version"), expected.get("qairt_version")),
            ("QNN core API", condition.get("qnn_core_api"), expected.get("qnn_core_api")),
            (
                "QAIRT deployed runtime hashes",
                condition.get("qairt_runtime_hashes"),
                expected.get("qairt_runtime_hashes"),
            ),
            ("NPU live mode proof", condition.get("mode_proof"), expected.get("mode_proof")),
        )
        for label, observed, reference in comparisons:
            if observed != reference:
                return f"{label} does not exactly match the calibrated NPU runtime"

        variant = condition.get("workload_variant")
        binaries = expected.get("workload_binary_sha256_by_variant")
        if not isinstance(variant, str) or not isinstance(binaries, dict) or variant not in binaries:
            return f"NPU workload_variant {variant!r} was not calibrated"
        if condition.get("workload_binary_sha256") != binaries[variant]:
            return f"NPU client hash for workload_variant {variant!r} does not match calibration"

    if dsp_active:
        expected = calibrated.get("dsp")
        if not isinstance(expected, dict):
            return "artifact lacks calibrated DSP runtime provenance"
        power_mode = condition.get("dsp_power_mode")
        if power_mode not in artifact.supported_dsp_power_modes:
            return (
                f"DSP power mode {power_mode!r} is not calibrated; "
                f"supported={artifact.supported_dsp_power_modes!r}"
            )
        comparisons = (
            ("DSP power mode", power_mode, expected.get("power_mode")),
            ("DSP stub hash", condition.get("dsp_stub_sha256"), expected.get("dsp_stub_sha256")),
            ("DSP skel hash", condition.get("dsp_skel_sha256"), expected.get("dsp_skel_sha256")),
            ("DSP live mode proof", condition.get("mode_proof"), expected.get("mode_proof")),
        )
        for label, observed, reference in comparisons:
            if observed != reference:
                return f"{label} does not exactly match the calibrated DSP runtime"

        variant = condition.get("workload_variant")
        binaries = expected.get("workload_binary_sha256_by_variant")
        if not isinstance(variant, str) or not isinstance(binaries, dict) or variant not in binaries:
            return f"DSP workload_variant {variant!r} was not calibrated"
        if condition.get("workload_binary_sha256") != binaries[variant]:
            return f"DSP driver hash for workload_variant {variant!r} does not match calibration"

    return None


def _finite_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result)else None


def _hexagon_thermal_reason(artifact: PowerModelArtifact, aggregate: dict) -> str | None:
    """Reject unknown/throttled or thermally out-of-envelope Hexagon windows."""
    n_samples = aggregate.get("n_samples")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        return "Hexagon telemetry has an invalid sample count"

    cooling = aggregate.get("cooling_state_ranges")
    if not isinstance(cooling, dict):
        return "Hexagon cooling telemetry is missing"
    missing_cooling = sorted(_HEXAGON_COOLING_TYPES - set(cooling))
    if missing_cooling:
        return f"Hexagon cooling telemetry is incomplete; missing={missing_cooling!r}"
    for name in sorted(_HEXAGON_COOLING_TYPES):
        observed = cooling.get(name)
        if not isinstance(observed, dict) or observed.get("n_samples") != n_samples:
            return f"{name} cooling telemetry does not cover the complete window"
        minimum = _finite_number(observed.get("min"))
        maximum = _finite_number(observed.get("max"))
        if minimum is None or maximum is None:
            return f"{name} cooling telemetry is malformed"
        if minimum != 0.0 or maximum != 0.0:
            return f"Hexagon cooling/throttling detected: {name} range={minimum:g}..{maximum:g}"

    envelope = artifact.calibration_temperature_range
    by_sensor = envelope.get("by_sensor")if isinstance(envelope, dict)else None
    if not isinstance(by_sensor, dict):
        return "artifact lacks a per-sensor Hexagon calibration temperature envelope"
    calibrated = {
        name: bounds
        for name, bounds in by_sensor.items()
        if isinstance(name, str) and name.startswith(_HEXAGON_TEMP_PREFIXES)
    }
    if not calibrated or any(
        not any(name.startswith(prefix)for name in calibrated)
        for prefix in _HEXAGON_TEMP_PREFIXES
    ):
        return "artifact Hexagon temperature envelope lacks both HVX and HMX sensor families"

    observed_temps = aggregate.get("temperature_ranges_c")
    if not isinstance(observed_temps, dict):
        return "Hexagon temperature telemetry is missing"
    observed_names = {
        name
        for name in observed_temps
        if isinstance(name, str) and name.startswith(_HEXAGON_TEMP_PREFIXES)
    }
    if observed_names != set(calibrated):
        missing = sorted(set(calibrated) - observed_names)
        extra = sorted(observed_names - set(calibrated))
        return (
            "Hexagon temperature sensor population differs from calibration; "
            f"missing={missing!r} extra={extra!r}"
        )

    for name in sorted(calibrated):
        bounds = calibrated[name]
        observed = observed_temps.get(name)
        if not isinstance(bounds, dict) or not isinstance(observed, dict):
            return f"temperature envelope/telemetry for {name} is malformed"
        cal_min = _finite_number(bounds.get("min_c"))
        cal_max = _finite_number(bounds.get("max_c"))
        run_min = _finite_number(observed.get("min_c"))
        run_max = _finite_number(observed.get("max_c"))
        if None in (cal_min, cal_max, run_min, run_max) or cal_min > cal_max or run_min > run_max:
            return f"temperature envelope/telemetry for {name} is malformed"
        if observed.get("n_samples") != n_samples:
            return f"{name} temperature telemetry does not cover the complete window"
        if run_min < cal_min or run_max > cal_max:
            return (
                f"{name} temperature {run_min:g}..{run_max:g} C is outside the "
                f"calibrated envelope {cal_min:g}..{cal_max:g} C"
            )
        if run_max - run_min > A.MAX_HEXAGON_PLATEAU_TEMP_SPAN_C:
            return (
                f"{name} temperature span {run_max - run_min:.1f} C exceeds the "
                f"validated plateau limit {A.MAX_HEXAGON_PLATEAU_TEMP_SPAN_C:.1f} C"
            )
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python scripts/power_estimate.py <artifact.json> <run_telemetry.json>")
        return 2
    try:
        artifact = PowerModelArtifact.from_json(pathlib.Path(argv[1]).read_text())
        run = json.loads(pathlib.Path(argv[2]).read_text())
    except (OSError, ModelError, TypeError, ValueError)as e:
        print(f"load failed: {e}")
        return 1

    if not isinstance(run, dict):
        print("load failed: run telemetry root must be a JSON object")
        return 1
    run_windows = run.get("windows")
    if not isinstance(run_windows, list) or not run_windows:
        print("run has no windows -- refusing to emit a spurious zero estimate")
        return 1

        # Hexagon paths are discrete workload-class steps, valid only for the exact
        # runtime population and live proof recorded by calibration. Missing fields
        # never default. Keep the IDs so aggregate-level thermal checks below can
        # identify the precise offending window.
    hexagon_window_ids: set[str] = set()
    for window_index, w in enumerate(run_windows):
        if not isinstance(w, dict):
            print(
                "estimate failed (fail-closed): "
                f"run window at index {window_index} must be an object"
            )
            return 1
        wid = w.get("window_id")
        if not isinstance(wid, str) or not wid:
            print(
                "estimate failed (fail-closed): "
                f"run window at index {window_index} has invalid/missing window_id {wid!r}"
            )
            return 1
        cond = w.get("condition", {})
        if not isinstance(cond, dict):
            print(
                "estimate failed (fail-closed): "
                f"window {wid!r} condition must be an object"
            )
            return 1
        raw_engine = cond.get("engine_class")
        if not isinstance(raw_engine, str):
            print(
                "estimate failed (fail-closed): "
                f"window {wid!r} condition.engine_class must be a string"
            )
            return 1
        ec = raw_engine.lower()
        for key in ("npu_active", "dsp_active"):
            if key in cond and type(cond[key]) is not bool:
                print(
                    f"estimate failed (fail-closed): window {wid!r} "
                    f"condition.{key} must be JSON boolean, got {cond[key]!r}"
                )
                return 1
        npu_declared = cond.get("npu_active", False)
        dsp_declared = cond.get("dsp_active", False)
        if ec == "npu" and "npu_active" in cond and not npu_declared:
            print(
                "estimate failed (fail-closed): "
                f"window {wid!r} engine_class='npu' contradicts npu_active=false"
            )
            return 1
        if ec == "dsp" and "dsp_active" in cond and not dsp_declared:
            print(
                "estimate failed (fail-closed): "
                f"window {wid!r} engine_class='dsp' contradicts dsp_active=false"
            )
            return 1
        npu_active = ec == "npu" or npu_declared
        dsp_active = ec == "dsp" or dsp_declared
        if npu_active and dsp_active:
            print(json.dumps(_unavailable(
                artifact,
                "simultaneous NPU+DSP intent is unsupported: both paths share and "
                "hardware-serialize on the Hexagon cDSP; use disjoint windows",
                wid,
            ), indent=2))
            return 0
        if npu_active or dsp_active:
            reason = _hexagon_runtime_reason(
                artifact,
                cond,
                npu_active=npu_active,
                dsp_active=dsp_active,
            )
            if reason is not None:
                print(json.dumps(_unavailable(artifact, reason, wid), indent=2))
                return 0
            hexagon_window_ids.add(wid)

    # How far the kernel's EAS power values may move from calibration before
    # the fit is considered to be describing different hardware. Measured
    # boot-to-boot drift on the calibration board is 0.89-1.56% (Prime cluster
    # only, frequency ladder unchanged); 5% leaves headroom for that without
    # admitting a genuinely different part, which in practice also fails the
    # ladder check.
    EAS_VALUE_TOL_PCT = 5.0

    warnings: list[str] = []
    # Hard gates.  The coefficients are fitted per board against a bench supply
    # (ADR 0019), so on any other board the number is not "less accurate", it is
    # meaningless.  These conditions therefore return `unavailable` rather than
    # a warned-but-emitted wattage.  Moving the HDK8750 model onto an HDK8850
    # (SM8850, Adreno 840, different EAS tables and GPU frequency ladder) trips
    # both the EAS-hash and board-serial gates.
    blocking: list[str] = []

    # OOD: EAS table hash (board/kernel proxy)
    try:
        run_eas_hash = hashlib.sha256(
            _canonical(run.get("eas_tables", {})).encode()
        ).hexdigest()
    except (TypeError, ValueError)as exc:
        print(f"estimate failed (fail-closed): invalid EAS table JSON: {exc}")
        return 1
    # OOD: EAS table. An exact-hash gate cannot pass on this board.
    #
    # Measured 2026-09-09 on 9d261d4e: the kernel's energy model is byte-stable
    # WITHIN a boot (5 consecutive reads identical) and completely
    # temperature-independent (identical across a 53.5 C swing, 56.2 -> 109.7 C),
    # but it re-derives ~0.9-1.6% different power values for the Prime cluster
    # ACROSS boots. 47 of 57 shared OPPs were byte-identical; the 10 that moved
    # were all cpu6 and all in the same direction. The frequency LADDER -- the
    # part that actually identifies the silicon -- was identical.
    #
    # So the hash answers "different" every boot on the very board the model was
    # fitted on, and power was permanently unavailable. What the gate must
    # actually distinguish is different SILICON from redetermined leakage:
    #
    #   * ladder mismatch  -> blocking. This is what catches a different part.
    #     The SM8750 board (cead6f2e) has 16/16 OPPs topping out at 3532800 /
    #     4320000 kHz against this board's 28/29 at 3628800 / 4608000, so the
    #     ladder separates them unambiguously.
    #   * values beyond tolerance -> blocking. Same ladder but materially
    #     different power means the fit's basis moved.
    #   * values within tolerance -> warning naming the worst OPP, not a block.
    #
    # The hash is still recorded for provenance; it is simply no longer the
    # thing that decides. Requires `eas_table` in the artifact; if it is absent
    # we fall back to the old exact-hash behaviour rather than passing blindly.
    # Set only when this board's kernel energy model was compared against the
    # calibration board's and matched: same ladder, values within tolerance.
    # It is the physical evidence that the serial number only stood in for.
    eas_verified = False
    run_eas = run.get("eas_tables", {}) or {}
    cal_eas = getattr(artifact, "eas_table", None) or {}
    if cal_eas and run_eas:
        def _ladder(t):
            return {k: sorted(int(f) for f, _ in v) for k, v in t.items()}

        def _values(t):
            return {k: {int(f): float(p) for f, p in v} for k, v in t.items()}

        lad_run, lad_cal = _ladder(run_eas), _ladder(cal_eas)
        if lad_run != lad_cal:
            only_run = sorted(set(lad_run) - set(lad_cal))
            only_cal = sorted(set(lad_cal) - set(lad_run))
            detail = []
            if only_run or only_cal:
                detail.append(f"clusters run={sorted(lad_run)} calibration={sorted(lad_cal)}")
            for cl in sorted(set(lad_run) & set(lad_cal)):
                if lad_run[cl] != lad_cal[cl]:
                    detail.append(
                        f"{cl}: run has {len(lad_run[cl])} OPPs topping at "
                        f"{lad_run[cl][-1]} kHz, calibration {len(lad_cal[cl])} "
                        f"topping at {lad_cal[cl][-1]} kHz")
            blocking.append(
                "eas_ladder_mismatch: the CPU frequency ladder differs from the "
                "calibration board, so this is not the hardware the coefficients "
                "describe -- " + "; ".join(detail))
        else:
            v_run, v_cal = _values(run_eas), _values(cal_eas)
            worst_pct, worst_at = 0.0, None
            for cl in sorted(set(v_run) & set(v_cal)):
                for f in sorted(set(v_run[cl]) & set(v_cal[cl])):
                    base = v_cal[cl][f]
                    if base <= 0:
                        continue
                    pct = abs(100.0 * (v_run[cl][f] - base) / base)
                    if pct > worst_pct:
                        worst_pct, worst_at = pct, (cl, f, base, v_run[cl][f])
            if worst_pct == 0.0:
                eas_verified = True
            if worst_pct > EAS_VALUE_TOL_PCT:
                cl, f, b, r = worst_at
                blocking.append(
                    f"eas_value_drift: {cl} at {f} kHz reports {r:.6f} W against "
                    f"{b:.6f} W at calibration ({worst_pct:.2f}% > "
                    f"{EAS_VALUE_TOL_PCT:.1f}% tolerance) -- the energy model's "
                    "basis has moved too far for the fit to describe it")
            elif worst_pct > 0.0:
                eas_verified = True
                cl, f, b, r = worst_at
                warnings.append(
                    f"eas_values_redetermined: same frequency ladder, power values "
                    f"differ by up to {worst_pct:.2f}% (worst {cl} at {f} kHz, "
                    f"{b:.6f} -> {r:.6f} W); within the {EAS_VALUE_TOL_PCT:.1f}% "
                    "tolerance, treated as the same board across boots")
    elif artifact.eas_table_hash and run_eas_hash != artifact.eas_table_hash:
        # No embedded calibration table to compare against: fail closed exactly
        # as before rather than assume the hardware matches.
        blocking.append(
            "eas_table_hash_mismatch: run hardware/kernel differs from the "
            "calibration board and the artifact carries no eas_table to "
            "diagnose it further -- the fitted coefficients do not describe "
            "this hardware")

        # OOD: explicit device identity (model is board-specific)
    dev = run.get("device", {})
    if isinstance(dev, dict) and dev:
        missing_identity = [
            key
            for key in ("board_serial", "android_build_fingerprint", "kernel_release")
            if not dev.get(key)
        ]
        if missing_identity:
            warnings.append(
                "device_identity_incomplete: missing " + ",".join(missing_identity)
            )
        if dev.get("board_serial") and artifact.calibration_board_serial and dev["board_serial"] != artifact.calibration_board_serial:
            if eas_verified:
                # Different unit, same hardware by measurement. The CPU term is
                # confirmed against this board's own energy model; the GPU and
                # board-level terms have no equivalent table and are inferred.
                warnings.append(
                    f"board_transferred: run {dev['board_serial']!r} is not the calibration board "
                    f"{artifact.calibration_board_serial!r}, but its CPU energy model matches "
                    "within tolerance, so the fitted coefficients are applied. The CPU "
                    "term is verified against this board; the GPU and board-level terms "
                    "are not independently checkable without a bench supply. Treat the "
                    "absolute watts as carrying a few percent more uncertainty than on "
                    "the calibration board; comparisons between two runs on THIS board "
                    "are unaffected.")
            else:
                blocking.append(
                    f"board_serial_mismatch: run {dev['board_serial']!r} vs calibration "
                    f"{artifact.calibration_board_serial!r}, and this board's energy model "
                    "could not be verified against the calibration board's -- nothing "
                    "establishes that the fitted coefficients describe this hardware")
        for k, af in (("android_build_fingerprint", artifact.android_build_fingerprint), ("kernel_release", artifact.kernel_release)):
            if dev.get(k) and af and dev[k] != af:
                warnings.append(f"{k}_mismatch: run {dev[k]!r} vs calibration {af!r}")
    else:
        blocking.append("device_identity_unknown: run does not declare board/build/kernel -- cannot confirm this is the calibration board (the model is board-specific)")

    if blocking:
        print(json.dumps(
            _unavailable(artifact, "out_of_distribution: " + "; ".join(blocking)),
            indent=2))
        return 0

    min_s = float(artifact.minimum_validated_interval_s or 0.0)
    try:
        records = A.records_from_session(run, artifact.knots_hz())
        for record in records:
            wid = record["window_id"]
            if wid not in hexagon_window_ids:
                continue
            reason = _hexagon_thermal_reason(artifact, record["aggregate"])
            if reason is not None:
                print(json.dumps(_unavailable(artifact, reason, wid), indent=2))
                return 0
        windows_out = []
        valid = []  # (power_w, window_s) for windows within the validated interval
        for r in records:
            agg = r["aggregate"]
            if float(agg["window_s"]) < min_s:
                # short/transient -> power itself is UNAVAILABLE (not just energy)
                warnings.append(f"window {r['window_id']!r} {float(agg['window_s']):.1f}s < validated minimum {min_s:.1f}s -- power unavailable")
                windows_out.append({
                    "window_id": r["window_id"], "power_w": None, "dynamic_above_quiescent_w": None,
                    "window_s": agg["window_s"], "unavailable": "below_minimum_validated_interval", "warnings": [],
                })
                continue
            est = estimate_power(agg, artifact)
            windows_out.append({
                "window_id": r["window_id"],
                "power_w": round(est["power_w"], 3),
                "dynamic_above_quiescent_w": round(est["dynamic_above_quiescent_w"], 3),
                "window_s": agg["window_s"],
                "warnings": est["warnings"],
            })
            warnings.extend(est["warnings"])
            valid.append((est["power_w"], float(agg["window_s"])))
    except (KeyError, ModelError, TypeError, ValueError)as e:
        print(f"estimate failed (fail-closed): {e}")
        return 1

    try:
        total_energy = round(integrate_energy([r["aggregate"]for r in records], artifact), 3)
    except ModelError as e:
        total_energy = None
        warnings.append(f"energy_unavailable: {e}")

    if valid:
        tot_s = sum(s for _, s in valid) or 1.0
        mean_power = sum(p * s for p, s in valid) / tot_s
        power_w = round(mean_power, 3)
        dynamic_w = round(mean_power - artifact.quiescent_power_w, 3)
    else:
        power_w = None
        dynamic_w = None
        warnings.append("power_unavailable: no window met the validated minimum interval")

    out = {
        "kind": "heuristic_estimate",
        "quality": artifact.quality,
        "board_specific": True,
        "board_match": ("calibration_board"
                        if run.get("device", {}).get("board_serial")
                           == artifact.calibration_board_serial
                        else "transferred_eas_verified"),
        "calibration_board_serial": artifact.calibration_board_serial,
        "power_w": power_w,
        "dynamic_above_quiescent_w": dynamic_w,
        "total_energy_j": total_energy,
        "boundary": artifact.boundary,
        "scope": artifact.scope,
        "model_version": artifact.model_version,
        "model_profile": artifact.model_profile,
        "provenance": {
            "calibration_dataset_hash": artifact.calibration_dataset_hash,
            "fit_input_hash": artifact.fit_input_hash,
            "design_hash": artifact.validation_observed.get("design_hash"),
            "source_session_hashes": artifact.source_session_hashes,
        },
        "supported_perf_modes": artifact.supported_perf_modes,
        "supported_dsp_power_modes": artifact.supported_dsp_power_modes,
        "hexagon_subsystem": artifact.hexagon_subsystem,
        "concurrency_validated": False,
        "physical_engine_attribution": False,
        "known_unmodeled_terms": artifact.known_unmodeled_terms,
        "validation_observed": artifact.validation_observed,
        "windows": windows_out,
        "warnings": sorted(set(warnings)),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
