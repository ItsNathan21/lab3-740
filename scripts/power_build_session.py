"""Assemble + VALIDATE a calibration session (or a student run) from raw sampler
output and an operator manifest.

Run from project root:
    python scripts/power_build_session.py <manifest.json> <sampler_outdir> <session_out.json>

FAIL-CLOSED. For every window declared in the manifest this tool requires:
  * a raw file whose =WINDOW header id matches the manifest id (no swapped files);
  * a `.done` sentinel and no `.fail`;
  * a strictly-parseable, complete, sequential sample stream that `aggregate_window`
    accepts (integrity checked, not just present);
  * enough duration (>= minimum_validated_interval_s);
  * for calibration splits (idle_anchor/train/holdout): exactly 5 finite/positive CV
    meter reads (validated via the real meter validator), and -- where the manifest
    declares them -- the observed CPU OPP / GPU pwrlevel actually pinned to the
    requested value (control-state validation).
Any problem fails the whole build; nothing is silently skipped. A `run` split needs
no meter/control (the student-run path). STDLIB only.
"""

from __future__ import annotations

import json
import hashlib
import math
import pathlib
import re
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.teaching_power_estimator import aggregate as A  # noqa: E402
from tools.teaching_power_estimator.model import (  # noqa: E402
    MINIMUM_VALIDATED_INTERVAL_S,
    ModelError,
    )

_RESERVED = {
    "split",
    "engine_class",
    "gpu_active",
    "npu_active",
    "dsp_active",
    "meter_samples",
    }
_NEEDS_POWER = {"idle_anchor", "train", "holdout"}
_ALLOWED_SPLITS = _NEEDS_POWER | {"run"}
_ALLOWED_ENGINES = {"idle", "floor", "cpu", "gpu", "npu", "dsp", "combined"}
EXPECTED_METER = 5
CPU_KHZ_TOL = 20_000
_WINDOW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_NPU_RUNTIME_FILES = {
    "libQnnHtp.so",
    "libQnnHtpPrepare.so",
    "libQnnHtpV79Stub.so",
    "libQnnSystem.so",
    "libQnnHtpV79Skel.so",
    "libQnnHtpV79CalculatorStub.so",
    "libCalculator_skel.so",
    "qnn-platform-validator",
    "qnn-net-run",
    }
_NPU_CAPTURE_FILES = _NPU_RUNTIME_FILES | {"matmul_qnn_fp16", "matmul_qnn_int8"}
_DSP_CAPTURE_FILES = {
    "matmul_dsp_sweep",
    "libdsp_matmul_stub.so",
    "libdsp_matmul_skel.so",
    }
_HEXAGON_COOLING_TYPES = {"cdsp_sw_hvx", "cdsp_sw_hmx"}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef"for char in value)
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_runtime_hash_snapshot(wid: str, text: str) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line.strip())
        if not match:
            raise ModelError(f"{wid}: malformed runtime hash line {line!r}")
        digest, device_path = match.groups()
        name = pathlib.PurePosixPath(device_path).name
        if name in records:
            raise ModelError(f"{wid}: duplicate runtime hash basename {name!r}")
        records[name] = {"sha256": digest, "device_path": device_path}
    return records


def _validate_workload_log(
    wid: str,
    engine: str,
    spec: dict,
    text: str,
            ) -> dict[str, str]:
    if text.count(f"=WORKLOAD window_id={wid} worker_count=1") != 1:
        raise ModelError(f"{wid}: workload log must identify exactly one worker")
    if text.count(f"=WORKLOAD_EXIT window_id={wid} status=0") != 1:
        raise ModelError(f"{wid}: workload log lacks one successful terminal status")
    command = spec.get("workload_cmd")
    if not isinstance(command, str) or not command.strip():
        raise ModelError(f"{wid}: workload_cmd must be a non-empty string")
    command_block = re.search(
        r"(?ms)^=COMMAND_BEGIN index=0\n(.*?)\n=COMMAND_END index=0$", text
    )
    if not command_block or command_block.group(1) != command:
        raise ModelError(f"{wid}: workload log command does not match immutable manifest")

    if engine == "npu":
        required = (
            "status: SUCCESS",
            "power_mode: sustained_high_performance",
            "[perf] sustained_high_performance active",
            "NPU_DISPATCH_END",
        )
        missing = [marker for marker in required if marker not in text]
        if missing:
            raise ModelError(f"{wid}: NPU workload log missing success/mode proof {missing}")
        api_match = re.search(r"(?m)^qnn_core_api:\s*(2\.\d+\.\d+)\s*$", text)
        precision_match = re.search(r"(?m)^precision:\s*(fp16|int8)\s*$", text)
        size_match = re.search(r"(?m)^matrix_n:\s*(\d+)\s*$", text)
        if not api_match or not precision_match or not size_match:
            raise ModelError(f"{wid}: incomplete NPU result block")
        major, minor, _patch = (int(part)for part in api_match.group(1).split("."))
        if major != 2 or minor < 30:
            raise ModelError(f"{wid}: incompatible live QNN core API {api_match.group(1)}")
        expected_variant = "qnn_fp16"if precision_match.group(1) == "fp16"else "qnn_int8"
        if spec.get("workload_variant") != expected_variant:
            raise ModelError(
                f"{wid}: workload log precision does not match workload_variant"
            )
        if int(size_match.group(1)) != int(spec.get("matrix_size", -1)):
            raise ModelError(f"{wid}: workload log matrix size does not match manifest")
        return {
            "qnn_core_api": api_match.group(1),
            "mode_proof": "client_log_vote_accepted_and_dispatch_success",
        }

    if engine == "dsp":
        start_match = re.search(
            r"(?m)^dsp-sweep-start variant=(vtcm|hvx) power=lab .* size=(\d+) ", text
        )
        if not start_match:
            raise ModelError(f"{wid}: DSP workload log lacks exact LAB start record")
        if "dsp-power-mode-applied mode=lab api_status=accepted"not in text:
            raise ModelError(f"{wid}: DSP workload log lacks accepted HAP LAB vote")
        end_match = re.search(r"(?m)^dsp-sweep-end .* valid=(\d+)\s*$", text)
        if not end_match or int(end_match.group(1)) <= 0:
            raise ModelError(f"{wid}: DSP workload log lacks a successful dispatch summary")
        expected_variant = "dsp_vtcm"if start_match.group(1) == "vtcm"else "dsp_hvx"
        if spec.get("workload_variant") != expected_variant:
            raise ModelError(f"{wid}: DSP log variant does not match manifest")
        if int(start_match.group(2)) != int(spec.get("matrix_size", -1)):
            raise ModelError(f"{wid}: DSP log matrix size does not match manifest")
        return {
            "mode_proof": "client_log_hap_vote_accepted_and_dispatch_success"
        }
    raise ModelError(f"{wid}: runtime log proof requested for unsupported engine {engine!r}")


def _load_runtime_capture(
    wid: str,
    engine: str,
    spec: dict,
    outdir: pathlib.Path,
    session_meta: dict,
            ) -> tuple[dict, dict[str, dict[str, str]], str, dict[str, str]]:
    manifest_path = outdir / f"{wid}.runtime-provenance.json"
    hashes_path = outdir / f"{wid}.runtime-artifacts.sha256"
    log_path = outdir / f"{wid}.workload.log"
    try:
        manifest_text = manifest_path.read_text()
        manifest = json.loads(manifest_text)
        hash_records = _parse_runtime_hash_snapshot(wid, hashes_path.read_text())
        workload_log = log_path.read_text()
    except (OSError, TypeError, ValueError, ModelError)as exc:
        raise ModelError(f"{wid}: cannot load runtime proof sidecars: {exc}")from exc
    if not isinstance(manifest, dict):
        raise ModelError(f"{wid}: runtime provenance root must be an object")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("manifest_kind") != "hdk8750-runtime-provenance"
        or manifest.get("hash_algorithm") != "sha256"
    ):
        raise ModelError(f"{wid}: unsupported runtime provenance manifest")
    device = manifest.get("device")
    expected_device = {
        "serial": session_meta.get("calibration_board_serial"),
        "android_build_fingerprint": session_meta.get("android_build_fingerprint"),
        "kernel_release": session_meta.get("kernel_release"),
    }
    if device != expected_device:
        raise ModelError(
            f"{wid}: runtime provenance device {device!r} != session {expected_device!r}"
        )
    validation = manifest.get("validation")
    required_gates = (
        {
            "npu_requested",
            "npu_platform_validator_passed",
            "npu_fp16_sustained_graph_passed",
            "npu_int8_sustained_graph_passed",
        }
        if engine == "npu"
        else {"dsp_requested", "dsp_lab_dispatch_passed"}
    )
    if not isinstance(validation, dict) or any(
        validation.get(gate) is not True for gate in required_gates
    ):
        raise ModelError(f"{wid}: runtime provenance live gates did not all pass")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ModelError(f"{wid}: runtime provenance files must be a list")
    manifest_files: dict[str, dict] = {}
    for entry in files:
        if not isinstance(entry, dict) or entry.get("engine_class")not in {"npu", "dsp"}:
            raise ModelError(f"{wid}: malformed runtime provenance file entry")
        name = entry.get("name")
        if not isinstance(name, str) or name in manifest_files or not _is_sha256(entry.get("sha256")):
            raise ModelError(f"{wid}: invalid/duplicate runtime provenance file {name!r}")
        manifest_files[name] = entry

    expected_names = _NPU_CAPTURE_FILES if engine == "npu"else _DSP_CAPTURE_FILES
    if {name for name, entry in manifest_files.items()if entry["engine_class"] == engine} != expected_names:
        raise ModelError(
            f"{wid}: runtime manifest {engine} file set does not match the fixed deployment"
        )
        # One extra hash binds the exact manifest snapshot; all expected deployed
        # files must then match both that snapshot and the sampler's fresh hashes.
    if "runtime-provenance.json"not in hash_records:
        raise ModelError(f"{wid}: runtime hash snapshot does not bind its manifest")
    if hash_records["runtime-provenance.json"]["sha256"] != _sha256_text(manifest_text):
        raise ModelError(f"{wid}: runtime provenance manifest hash mismatch")
    observed_engine_names = set(hash_records) - {"runtime-provenance.json"}
    if observed_engine_names != expected_names:
        raise ModelError(
            f"{wid}: observed runtime file set {sorted(observed_engine_names)} != "
            f"{sorted(expected_names)}"
        )
    for name in expected_names:
        observed = hash_records[name]
        declared = manifest_files[name]
        if (
            observed["sha256"] != declared.get("sha256")
            or observed["device_path"] != declared.get("device_path")
        ):
            raise ModelError(f"{wid}: observed runtime bytes for {name!r} differ from provision manifest")

    log_proof = _validate_workload_log(wid, engine, spec, workload_log)
    return manifest, hash_records, workload_log, log_proof


def _declared_bool(wid: str, spec: dict, key: str, errors: list[str]) -> bool:
    value = spec.get(key, False)
    if type(value) is not bool:
        errors.append(f"{wid}: {key} must be JSON boolean, got {value!r}")
        return False
    return value


def _hash_issues(wid: str, spec: dict, keys: tuple[str, ...]) -> list[str]:
    return [
        f"{wid}: {key} must be a lowercase SHA-256 digest"
        for key in keys
        if not _is_sha256(spec.get(key))
    ]


def _control_issues(wid: str, spec: dict, samples: list[dict]) -> list[str]:
    """Verify the observed control state was actually pinned to the requested OPP."""
    issues: list[str] = []
    req_cpu = spec.get("requested_cpu_khz")
    if req_cpu is not None:
        try:
            req_cpu = int(req_cpu)
        except (TypeError, ValueError):
            return [f"{wid}: requested_cpu_khz must be an integer"]
        policy = spec.get("requested_cpu_policy")
        if policy not in {"policy0", "policy6"}:
            return [
                f"{wid}: requested_cpu_policy must be 'policy0' or 'policy6' "
                "when requested_cpu_khz is set"
            ]
        for i, s in enumerate(samples):
            try:
                cur = int(s.get(f"{policy}_khz", -1))
                mn = int(s.get(f"{policy}_min_khz", -1))
                mx = int(s.get(f"{policy}_max_khz", -1))
            except (TypeError, ValueError):
                cur = mn = mx = -1
            pinned = (
                abs(cur - req_cpu) <= CPU_KHZ_TOL
                and abs(mn - req_cpu) <= CPU_KHZ_TOL
                and abs(mx - req_cpu) <= CPU_KHZ_TOL
            )
            if not pinned:
                issues.append(
                    f"{wid} sample {i}: {policy} not pinned (min=max=cur) at "
                    f"requested {req_cpu} kHz"
                )
                break
    req_pl = spec.get("requested_gpu_pwrlevel")
    if req_pl is not None:
        try:
            req_pl = int(req_pl)
        except (TypeError, ValueError):
            return issues + [f"{wid}: requested_gpu_pwrlevel must be an integer"]
        for i, s in enumerate(samples):
            try:
                mn = int(s["gpu_min_pwrlevel"]); mx = int(s["gpu_max_pwrlevel"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{wid} sample {i}: gpu pwrlevel unreadable")
                break
            if not (mn == mx == req_pl):
                issues.append(f"{wid} sample {i}: gpu pwrlevel not pinned to {req_pl} (min={mn} max={mx})")
                break
    return issues


def _hexagon_thermal_issues(wid: str, aggregate: dict) -> list[str]:
    """Reject throttled or thermally non-stationary Hexagon plateaus."""
    issues: list[str] = []
    cooling = aggregate.get("cooling_state_ranges") or {}
    missing = sorted(_HEXAGON_COOLING_TYPES - set(cooling))
    if missing:
        issues.append(f"{wid}: missing required Hexagon cooling telemetry {missing}")
    for name in sorted(_HEXAGON_COOLING_TYPES & set(cooling)):
        state_range = cooling[name]
        if state_range.get("n_samples") != aggregate.get("n_samples"):
            issues.append(
                f"{wid}: {name} cooling telemetry covers "
                f"{state_range.get('n_samples')}/{aggregate.get('n_samples')} samples"
            )
        if state_range.get("min") != 0 or state_range.get("max") != 0:
            issues.append(
                f"{wid}: {name} cooling state is {state_range!r}; calibrated/run "
                "Hexagon estimate requires a stable unthrottled state 0"
            )

    temps = aggregate.get("temperature_ranges_c") or {}
    hexagon_temps = {
        name: value
        for name, value in temps.items()
        if name.startswith("nsphvx-") or name.startswith("nsphmx-")
    }
    for family in ("nsphvx-", "nsphmx-"):
        if not any(name.startswith(family)for name in hexagon_temps):
            issues.append(f"{wid}: no {family[:-1]} temperature telemetry")
    for name, temp_range in sorted(hexagon_temps.items()):
        if temp_range.get("n_samples") != aggregate.get("n_samples"):
            issues.append(
                f"{wid}: {name} temperature telemetry covers "
                f"{temp_range.get('n_samples')}/{aggregate.get('n_samples')} samples"
            )
        span = float(temp_range["max_c"]) - float(temp_range["min_c"])
        if span > A.MAX_HEXAGON_PLATEAU_TEMP_SPAN_C:
            issues.append(
                f"{wid}: {name} temperature span {span:.1f} C exceeds "
                f"{A.MAX_HEXAGON_PLATEAU_TEMP_SPAN_C:.1f} C; plateau is not stationary"
            )
    return issues


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: python scripts/power_build_session.py <manifest.json> <sampler_outdir> <session_out.json>")
        return 2
    try:
        manifest = json.loads(pathlib.Path(argv[1]).read_text())
    except (OSError, ValueError)as exc:
        print(f"FATAL: cannot load manifest: {exc}")
        return 1
    if not isinstance(manifest, dict):
        print("FATAL: manifest root must be a JSON object")
        return 1
    outdir = pathlib.Path(argv[2])
    session_out = pathlib.Path(argv[3])
    if session_out.exists():
        print(f"refusing to overwrite existing {session_out} -- sessions are immutable")
        return 1

    try:
        eas_parsed = A.parse_eas_dump((outdir / "eas.txt").read_text())
    except (OSError, ValueError, ModelError)as exc:
        print(f"FATAL: cannot parse sampler EAS table: {exc}")
        return 1
    eas_tables = {k: [[f, p]for (f, p) in rows]for k, rows in eas_parsed.items()}
    if not eas_tables:
        print("FATAL: eas.txt produced no tables -- sampler must run as root")
        return 1
    eas_for_agg = {k: [(int(f), float(p))for f, p in rows]for k, rows in eas_tables.items()}

    spec_top = manifest.get("model_spec", {})
    if not isinstance(spec_top, dict):
        print("FATAL: manifest.model_spec must be an object")
        return 1
    try:
        knots = [float(k)for k in spec_top.get("knots_hz", [])]
        min_s = float(spec_top.get("minimum_validated_interval_s", 0.0))
    except (TypeError, ValueError)as exc:
        print(f"FATAL: malformed numeric model_spec value: {exc}")
        return 1
    if len(knots) != 3:
        print("FATAL: manifest model_spec.knots_hz must be 3 values")
        return 1
    if not math.isfinite(min_s) or min_s < MINIMUM_VALIDATED_INTERVAL_S:
        print(
            "FATAL: manifest model_spec.minimum_validated_interval_s must be finite "
            f"and >= {MINIMUM_VALIDATED_INTERVAL_S:g}"
        )
        return 1
    manifest_windows = manifest.get("windows")
    if not isinstance(manifest_windows, dict) or not manifest_windows:
        print("FATAL: manifest.windows must be a non-empty object")
        return 1
    sess_meta = manifest.get("session", {})
    if not isinstance(sess_meta, dict):
        print("FATAL: manifest.session must be an object")
        return 1

    extension_idle_checks = manifest.get("extension_idle_checks")
    if spec_top.get("design_profile") in {"npu_v2", "npu_only_v2"}:
        try:
            A.validate_extension_idle_checks(extension_idle_checks)
        except ModelError as exc:
            print(f"FATAL: invalid extension idle bridge: {exc}")
            return 1

    errors: list[str] = []
    windows = []
    for wid, spec in manifest_windows.items():
        if not isinstance(wid, str) or not _WINDOW_ID.fullmatch(wid):
            errors.append(f"unsafe/invalid window id {wid!r}")
            continue
        if not isinstance(spec, dict):
            errors.append(f"{wid}: window declaration must be an object")
            continue
        split = spec.get("split")
        engine = spec.get("engine_class")
        if split not in _ALLOWED_SPLITS:
            errors.append(f"{wid}: split must be one of {sorted(_ALLOWED_SPLITS)}, got {split!r}")
            continue
        if engine not in _ALLOWED_ENGINES:
            errors.append(
                f"{wid}: engine_class must be one of {sorted(_ALLOWED_ENGINES)}, got {engine!r}"
            )
            continue
        raw_path = outdir / f"{wid}.raw"
        if not raw_path.exists():
            errors.append(f"{wid}: missing {wid}.raw"); continue
        if (outdir / f"{wid}.fail").exists():
            errors.append(f"{wid}: has a .fail sentinel (sampler detected bad reads)"); continue
        done_path = outdir / f"{wid}.done"
        if not done_path.exists():
            errors.append(f"{wid}: no .done sentinel (sampler did not complete)"); continue
        try:
            done_text = done_path.read_text().strip()
        except OSError as exc:
            errors.append(f"{wid}: cannot read .done sentinel: {exc}"); continue
        done_match = re.fullmatch(r"OK samples=(\d+)(?:\s+.+)?", done_text)
        if not done_match:
            errors.append(f"{wid}: malformed .done sentinel {done_text!r}"); continue
        try:
            header, samples = A.parse_raw_window(raw_path.read_text())
        except Exception as e:
            errors.append(f"{wid}: raw parse/integrity failed: {e}"); continue
        if header.get("window_id") != wid:
            errors.append(f"{wid}: raw header id {header.get('window_id')!r} != manifest id (swapped file?)"); continue
        if header.get("engine_class") != engine:
            errors.append(
                f"{wid}: raw header engine_class={header.get('engine_class')!r} != "
                f"manifest {engine!r}"
            )
            continue
        if int(done_match.group(1)) != len(samples):
            errors.append(
                f"{wid}: .done sample count {done_match.group(1)} != raw sample count {len(samples)}"
            )
            continue
        expected_active = 0 if engine == "idle"else 1
        if header.get("expect_active") != str(expected_active):
            errors.append(
                f"{wid}: raw expect_active={header.get('expect_active')!r}, expected "
                f"{expected_active} for engine_class={engine!r}"
            )
            continue
        declared_duration = float(header["duration_s"])
        declared_hz = float(header["hz"])
        dur = float(samples[-1]["t_before"]) - float(samples[0]["t_before"])
        if dur < min_s:
            errors.append(f"{wid}: window {dur:.1f}s < minimum_validated_interval_s {min_s:.1f}s"); continue
        if dur < 0.90 * declared_duration:
            errors.append(
                f"{wid}: observed span {dur:.2f}s is <90% of declared "
                f"duration {declared_duration:.2f}s"
            )
            continue
        expected_samples = declared_duration * declared_hz + 1.0
        minimum_samples = max(3, math.floor(0.80 * expected_samples))
        if len(samples) < minimum_samples:
            errors.append(
                f"{wid}: sparse capture has {len(samples)} samples; expected about "
                f"{expected_samples:.0f} and require at least {minimum_samples}"
            )
            continue
        gaps = [
            float(right["t_before"]) - float(left["t_before"])
            for left, right in zip(samples, samples[1:])
        ]
        max_allowed_gap = max(1.0, 3.0 / declared_hz)
        if max(gaps) > max_allowed_gap:
            errors.append(
                f"{wid}: maximum sample gap {max(gaps):.3f}s exceeds "
                f"{max_allowed_gap:.3f}s at declared {declared_hz:g} Hz"
            )
            continue

        bool_error_count = len(errors)
        gpu_declared = _declared_bool(wid, spec, "gpu_active", errors)
        npu_declared = _declared_bool(wid, spec, "npu_active", errors)
        dsp_declared = _declared_bool(wid, spec, "dsp_active", errors)
        if len(errors) != bool_error_count:
            continue
        if engine in ("gpu", "combined") and "gpu_active" in spec and not gpu_declared:
            errors.append(f"{wid}: engine_class={engine!r} contradicts gpu_active=false")
            continue
        if engine == "npu" and "npu_active" in spec and not npu_declared:
            errors.append(f"{wid}: engine_class='npu' contradicts npu_active=false")
            continue
        if engine == "dsp" and "dsp_active" in spec and not dsp_declared:
            errors.append(f"{wid}: engine_class='dsp' contradicts dsp_active=false")
            continue
        if engine in {"idle", "floor", "cpu"} and (
            gpu_declared or npu_declared or dsp_declared
        ):
            errors.append(
                f"{wid}: engine_class={engine!r} cannot declare an accelerator path active"
            )
            continue
        if engine == "gpu" and (npu_declared or dsp_declared):
            errors.append(
                f"{wid}: engine_class='gpu' cannot also declare a Hexagon path; "
                "use engine_class='combined'"
            )
            continue
        if engine == "npu" and dsp_declared:
            errors.append(f"{wid}: engine_class='npu' contradicts dsp_active=true")
            continue
        if engine == "dsp" and npu_declared:
            errors.append(f"{wid}: engine_class='dsp' contradicts npu_active=true")
            continue
        gpu_active = engine in ("gpu", "combined") or gpu_declared
        npu_active = engine == "npu" or npu_declared
        dsp_active = engine == "dsp" or dsp_declared
        if npu_active and dsp_active:
            errors.append(f"{wid}: NPU and DSP cannot be declared active in the same window")
            continue
        try:  # aggregate now, not just at fit time — catches integrity issues early
            aggregate = A.aggregate_window(
                samples,
                eas_for_agg,
                knots,
                gpu_active=gpu_active,
                npu_active=npu_active,
                dsp_active=dsp_active,
            )
        except ModelError as e:
            errors.append(f"{wid}: aggregation rejected: {e}"); continue
        if npu_active or dsp_active:
            thermal_issues = _hexagon_thermal_issues(wid, aggregate)
            if thermal_issues:
                errors.extend(thermal_issues)
                continue

        runtime_manifest = None
        runtime_hash_records = None
        workload_log = None
        captured_provenance: dict = {}
        if engine in {"npu", "dsp"}:
            try:
                (
                    runtime_manifest,
                    runtime_hash_records,
                    workload_log,
                    log_proof,
                ) = _load_runtime_capture(wid, engine, spec, outdir, sess_meta)
                manifest_files = {
                    entry["name"]: entry["sha256"]
                    for entry in runtime_manifest["files"]
                    if entry["engine_class"] == engine
                }
                if engine == "npu":
                    variant = spec.get("workload_variant")
                    binary_name = {
                        "qnn_fp16": "matmul_qnn_fp16",
                        "qnn_int8": "matmul_qnn_int8",
                    }.get(variant)
                    if binary_name is None:
                        raise ModelError(
                            f"{wid}: NPU workload_variant must be qnn_fp16 or qnn_int8"
                        )
                    captured_provenance = {
                        "workload_binary_sha256": manifest_files[binary_name],
                        "qairt_version": runtime_manifest["qairt"]["sdk_build_id"],
                        "qairt_runtime_hashes": {
                            name: manifest_files[name]
                            for name in sorted(_NPU_RUNTIME_FILES)
                        },
                        **log_proof,
                    }
                else:
                    if spec.get("workload_variant")not in {"dsp_vtcm", "dsp_hvx"}:
                        raise ModelError(
                            f"{wid}: DSP workload_variant must be dsp_vtcm or dsp_hvx"
                        )
                    captured_provenance = {
                        "workload_binary_sha256": manifest_files["matmul_dsp_sweep"],
                        "dsp_stub_sha256": manifest_files["libdsp_matmul_stub.so"],
                        "dsp_skel_sha256": manifest_files["libdsp_matmul_skel.so"],
                        **log_proof,
                    }
                for key, captured_value in captured_provenance.items():
                    if key in spec and spec[key] != captured_value:
                        raise ModelError(
                            f"{wid}: declared {key} differs from captured device/runtime proof"
                        )
            except (KeyError, ModelError, TypeError, ValueError)as exc:
                errors.append(f"{wid}: runtime/workload proof invalid: {exc}")
                continue

        if split in _NEEDS_POWER:
            # aggregate.window_measured_power_w already accepts
            # MIN_METER_SAMPLES..EXPECTED_METER_SAMPLES and records the actual
            # count plus the spread, so the builder matches that range rather
            # than demanding five. Fewer reads mean a weaker steady-state
            # estimate -- with one there is no spread at all -- so prefer more
            # where the panel is stable enough to read them.
            n_meter = len(spec.get("meter_samples", []))
            if not (A.MIN_METER_SAMPLES <= n_meter <= EXPECTED_METER):
                errors.append(
                    f"{wid}: calibration needs {A.MIN_METER_SAMPLES}-{EXPECTED_METER} "
                    f"meter samples, got {n_meter}"
                )
                continue
            try:
                A.window_measured_power_w({"window_id": wid, "meter_samples": spec.get("meter_samples", [])})
            except ModelError as e:
                errors.append(f"{wid}: meter invalid: {e}"); continue
            errors.extend(_control_issues(wid, spec, samples))
            if npu_active:
                supported = spec_top.get("supported_perf_modes") or []
                if spec.get("perf_mode") != "sustained" or supported != ["sustained"]:
                    errors.append(
                        f"{wid}: NPU calibration requires perf_mode='sustained' and "
                        "model_spec.supported_perf_modes=['sustained']"
                    )
                    continue
                required = ("workload_cmd", "workload_binary_sha256", "qairt_version", "qairt_runtime_hashes")
                missing = [key for key in required if not spec.get(key)]
                if missing:
                    errors.append(f"{wid}: NPU calibration missing provenance {missing}")
                    continue
                if not isinstance(spec.get("workload_cmd"), str) or not spec["workload_cmd"].strip():
                    errors.append(f"{wid}: workload_cmd must be a non-empty string")
                if not isinstance(spec.get("qairt_version"), str) or not spec["qairt_version"].strip():
                    errors.append(f"{wid}: qairt_version must be a non-empty string")
                errors.extend(_hash_issues(wid, spec, ("workload_binary_sha256",)))
                runtime_hashes = spec.get("qairt_runtime_hashes")
                if not isinstance(runtime_hashes, dict):
                    errors.append(f"{wid}: qairt_runtime_hashes must be an object")
                    continue
                absent_runtime = sorted(_NPU_RUNTIME_FILES - set(runtime_hashes))
                if absent_runtime:
                    errors.append(
                        f"{wid}: qairt_runtime_hashes missing required files {absent_runtime}"
                    )
                for filename, digest in runtime_hashes.items():
                    if not isinstance(filename, str) or not filename or not _is_sha256(digest):
                        errors.append(
                            f"{wid}: qairt_runtime_hashes[{filename!r}] must be lowercase SHA-256"
                        )
            if dsp_active:
                supported = spec_top.get("supported_dsp_power_modes") or []
                if spec.get("dsp_power_mode") != "lab" or supported != ["lab"]:
                    errors.append(
                        f"{wid}: DSP calibration requires dsp_power_mode='lab' and "
                        "model_spec.supported_dsp_power_modes=['lab']"
                    )
                    continue
                required = (
                    "workload_cmd",
                    "workload_binary_sha256",
                    "dsp_stub_sha256",
                    "dsp_skel_sha256",
                )
                missing = [key for key in required if not spec.get(key)]
                if missing:
                    errors.append(f"{wid}: DSP calibration missing provenance {missing}")
                    continue
                if not isinstance(spec.get("workload_cmd"), str) or not spec["workload_cmd"].strip():
                    errors.append(f"{wid}: workload_cmd must be a non-empty string")
                errors.extend(
                    _hash_issues(
                        wid,
                        spec,
                        (
                            "workload_binary_sha256",
                            "dsp_stub_sha256",
                            "dsp_skel_sha256",
                        ),
                    )
                )

        clock_path = outdir / f"{wid}.clockstats"
        provenance = {k: v for k, v in spec.items()if k not in _RESERVED}
        provenance.update(captured_provenance)
        windows.append({
            "window_id": wid,
            "split": split,
            "condition": {
                "engine_class": engine,
                "gpu_active": gpu_active,
                "npu_active": npu_active,
                "dsp_active": dsp_active,
                **provenance,
            },
            "meter_samples": spec.get("meter_samples", []),
            "raw_header": header,
            # Preserve the exact completion claim that was checked against the
            # parsed sample stream.  The merged-session validator rechecks this
            # instead of trusting that a JSON file necessarily came through
            # this builder.
            "sampler_completion": {
                "sentinel": done_text,
                "reported_samples": int(done_match.group(1)),
                "parsed_samples": len(samples),
            },
            "gpu_clock_stats": clock_path.read_text()if clock_path.exists()else "",
            "runtime_provenance": runtime_manifest,
            "runtime_provenance_canonical_sha256": (
                hashlib.sha256(
                    json.dumps(
                        runtime_manifest,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                if runtime_manifest is not None
                else None
            ),
            "runtime_artifact_hashes": runtime_hash_records,
            "workload_log": workload_log,
            "workload_log_sha256": (
                _sha256_text(workload_log)if workload_log is not None else None
            ),
            "telemetry_samples": samples,
        })

    if errors:
        print("SESSION BUILD FAILED (fail-closed) -- not writing session:")
        for e in errors:
            print(f"  - {e}")
        return 1

    session = {
        "schema_version": 2,
        "session": sess_meta,
        "device": {
            "board_serial": sess_meta.get("calibration_board_serial", ""),
            "android_build_fingerprint": sess_meta.get("android_build_fingerprint", ""),
            "kernel_release": sess_meta.get("kernel_release", ""),
        },
        "calibration_control_config": manifest.get("calibration_control_config", {}),
        "eas_tables": eas_tables,
        "model_spec": spec_top,
        "extension_idle_checks": extension_idle_checks,
        "windows": windows,
    }
    try:
        session_out.parent.mkdir(parents=True, exist_ok=True)
        session_out.write_text(json.dumps(session, indent=2))
    except OSError as exc:
        print(f"FATAL: cannot write session: {exc}")
        return 1
    print(f"wrote {session_out} with {len(windows)} windows (all validated: parse+aggregate+meter+control)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
