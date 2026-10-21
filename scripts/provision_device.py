#!/usr/bin/env python3
"""provision_device.py - one-time HDK8850 provisioning. Any OS, no PowerShell.

    python scripts/provision_device.py \
        --qairt-root  <path>/qairt/2.40.0.251030 \
        --dsp-skel    kernels/matmul/dsp/prebuilt/libdsp_matmul_skel.so

A faithful port of provision_device.ps1. Same pushes, same SHA-256 verification,
same preflight and dispatch gates, same provenance manifest. It exists because
the PowerShell original made every student install `pwsh` to run one script once
-- a 71 MB download on macOS -- for a job whose only external tool is `adb`.

Standard library only, like install.py, so it runs wherever the profiler does.

CPU/GPU need no SDK; their binaries are committed prebuilt. NPU needs the QAIRT
runtime and DSP needs a Hexagon-built skel, neither of which may be
redistributed, so their paths are supplied by the operator. Without them the
script still provisions what it can and reports the rest UNAVAILABLE rather than
half-working.

Idempotent - safe to re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV_BASE = "/data/local/tmp/profiler-kernels"
DEV_NPU = f"{DEV_BASE}/npu"
DEV_DSP = f"{DEV_BASE}/dsp"
DEV_GPU = f"{DEV_BASE}/gpu"

# Hexagon/HTP architecture per SoC. NPU and DSP artifacts are built per
# architecture, so provisioning the wrong one fails at backendCreate rather than
# at push time. Extend this table when a new board arrives.
SOC_HTP_ARCH = {
    "SM8750": "v79",    # Snapdragon 8 Elite,       HDK8750
    "SM8850": "v81",    # Snapdragon 8 Elite Gen 5, HDK8850
    "SM8850P": "v81",
}

summary: dict[str, str] = {}
device_serial: str | None = None
deployed_runtime_files: list[dict] = []


class Fail(Exception):
    """A provisioning gate refused. The message is the whole diagnosis."""


def say(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------- host helpers

def require_file(path, description: str) -> pathlib.Path:
    p = pathlib.Path(path)
    if not p.is_file():
        raise Fail(f"{description} missing: {p}")
    return p


def _read_api(header, prefix: str, kind: str) -> tuple[int, int, int]:
    raw = require_file(header, f"QNN {kind} API-version header").read_text(
        encoding="utf-8", errors="replace")
    out = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"^\s*#define\s+{prefix}_{part}\s+(\d+)\s*$", raw, re.M)
        if not m:
            raise Fail(f"cannot parse QNN {kind} API version from {header}")
        out.append(int(m.group(1)))
    return tuple(out)  # type: ignore[return-value]


def read_qnn_core_api(header):
    return _read_api(header, "QNN_API_VERSION", "core")


def read_qnn_htp_api(header):
    return _read_api(header, "QNN_HTP_API_VERSION", "HTP")


def require_binary_marker(path, marker: str, build_command: str) -> None:
    p = require_file(path, "client binary")
    if marker.encode("ascii") not in p.read_bytes():
        raise Fail(f"stale client binary lacks {marker!r}: {p}\n"
                   f"Rebuild with: {build_command}")


def host_sha256(path) -> str:
    p = require_file(path, "file to hash")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------- adb plumbing

def _adb(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    # argv is passed through directly, so no shell and no MSYS path rewriting -
    # which is exactly the bug (discovery 72) that makes a bash port unsafe on
    # Windows, where `adb push x /data/local/tmp/y` silently pushes nothing.
    return subprocess.run(["adb", *args], capture_output=True, text=True,
                          timeout=timeout)


def select_device() -> str:
    global device_serial
    if device_serial:
        return device_serial
    try:
        r = _adb(["devices", "-l"], timeout=120)
    except FileNotFoundError:
        raise Fail("adb not found on PATH")
    if r.returncode != 0:
        raise Fail(f"adb devices failed: {r.stdout}{r.stderr}")
    ready = [ln for ln in r.stdout.splitlines()
             if re.match(r"^\S+\s+device(\s|$)", ln)]
    if len(ready) != 1:
        raise Fail(f"expected exactly one authorized adb device, found "
                   f"{len(ready)}:\n{r.stdout}{r.stderr}")
    device_serial = ready[0].split()[0]
    return device_serial


def dev(command: str, description: str = "adb shell") -> str:
    r = _adb(["-s", select_device(), "shell", command])
    if r.returncode != 0:
        raise Fail(f"{description} failed (exit {r.returncode}):\n"
                   f"{r.stdout}{r.stderr}")
    return r.stdout


def push_file(src, dst: str) -> None:
    r = _adb(["-s", select_device(), "push", str(src), dst])
    if r.returncode != 0:
        raise Fail(f"adb push FAILED: {src} -> {dst} (exit {r.returncode}):\n"
                   f"{r.stdout}{r.stderr}")


def device_sha256(path: str) -> str:
    # Callers pass fixed provisioner-owned paths. Require exactly one
    # conventional sha256sum record and reject noise rather than accidentally
    # accepting a partial read.
    lines = [ln for ln in dev(f"sha256sum '{path}'",
                              f"sha256sum {path}").splitlines() if ln.strip()]
    if len(lines) != 1:
        raise Fail(f"device sha256sum returned {len(lines)} nonempty lines "
                   f"for {path}:\n" + "\n".join(lines))
    m = re.match(r"^\s*([0-9A-Fa-f]{64})\s+\*?\S+\s*$", lines[0])
    if not m:
        raise Fail(f"cannot parse device SHA-256 for {path}: {lines[0]}")
    return m.group(1).lower()


def push_verified_runtime_file(engine_class: str, source, destination: str) -> None:
    if engine_class not in ("npu", "dsp"):
        raise Fail(f"internal error: unsupported runtime engine class {engine_class}")
    push_file(source, destination)
    hh = host_sha256(source)
    dh = device_sha256(destination)
    if dh != hh:
        raise Fail(f"SHA-256 mismatch after push: {source} -> {destination} "
                   f"(host={hh} device={dh})")
    deployed_runtime_files.append({
        "engine_class": engine_class,
        "name": destination.rsplit("/", 1)[-1],
        "device_path": destination,
        "size_bytes": pathlib.Path(source).stat().st_size,
        "sha256": hh,
    })
    say(f"  SHA-256 verified {destination} = {hh}")


def dev_root_script(script: str, description: str = "root device script") -> str:
    """Stage a script as the shell user, then have root execute that path.

    Never pipe a script to `adb shell`: a CRLF host would send CR into Android
    sh, which treats it as part of the last token. Writing exact LF bytes also
    sidesteps the board's fragile `su ... -c` quoting.
    """
    body = script.replace("\r\n", "\n").replace("\r", "\n")
    if not body.endswith("\n"):
        body += "\n"
    remote = f"/data/local/tmp/profiler-provision-{os.getpid()}-{uuid.uuid4().hex}.sh"
    fd, local = tempfile.mkstemp(suffix=".sh")
    os.close(fd)
    try:
        pathlib.Path(local).write_bytes(body.encode("utf-8"))
        push_file(local, remote)
        r = _adb(["-s", select_device(), "shell", "su", "0", "sh", remote])
        if r.returncode != 0:
            raise Fail(f"{description} failed (exit {r.returncode}):\n"
                       f"{r.stdout}{r.stderr}")
        return r.stdout
    finally:
        if device_serial:
            _adb(["-s", device_serial, "shell", "rm", "-f", remote], timeout=120)
        pathlib.Path(local).unlink(missing_ok=True)


def resolve_htp_arch(override: str | None) -> str:
    if override:
        a = override.lower()
        if not re.fullmatch(r"v\d+", a):
            raise Fail(f"--htp-arch must look like v79 or v81, got: {override}")
        return a
    soc = dev("getprop ro.soc.model", "read ro.soc.model").strip()
    if not soc:
        raise Fail("cannot read ro.soc.model to determine Hexagon architecture; "
                   "pass --htp-arch explicitly")
    if soc not in SOC_HTP_ARCH:
        raise Fail(f"unknown SoC {soc!r}: no Hexagon architecture mapping. Pass "
                   f"--htp-arch (e.g. --htp-arch v81) and add the SoC to "
                   f"SOC_HTP_ARCH in this script.")
    return SOC_HTP_ARCH[soc]


# ------------------------------------------------------------------ the work

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Provision an HDK8850 for Lab 3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="PowerShell-style flags (-QairtRoot, -DspSkel, -DspStub, "
               "-HtpArch) are also accepted, so notes written for the old "
               "script still work.")
    ap.add_argument("--qairt-root", default="",
                    help="extracted QAIRT SDK, e.g. ~/qairt/2.40.0.251030. "
                         "REQUIRED for the NPU (Tasks 12, 13).")
    ap.add_argument("--dsp-skel", default="",
                    help="Hexagon-compiled DSP skel .so. REQUIRED for the DSP "
                         "(Tasks 10, 11). Ships prebuilt in the repo.")
    ap.add_argument("--dsp-stub", default="",
                    help="ARM64 FastRPC stub .so; defaults beside --dsp-skel.")
    ap.add_argument("--htp-arch", default="",
                    help="v79 or v81; default: detected from ro.soc.model.")

    # accept the PowerShell spellings too
    alias = {"-qairtroot": "--qairt-root", "-dspskel": "--dsp-skel",
             "-dspstub": "--dsp-stub", "-htparch": "--htp-arch"}
    argv = [alias.get(a.lower(), a) for a in argv]
    a = ap.parse_args(argv[1:])

    qairt_root = a.qairt_root.strip()
    dsp_skel = a.dsp_skel.strip()
    dsp_stub = a.dsp_stub.strip()

    npu_requested = bool(qairt_root)
    dsp_requested = bool(dsp_skel or dsp_stub)

    npu_fp16 = ROOT / "kernels/matmul/npu/qnn_fp16/matmul_qnn_fp16"
    npu_int8 = ROOT / "kernels/matmul/npu/qnn_int8/matmul_qnn_int8"
    npu_spike = ROOT / "kernels/matmul/npu/qnn_fp16/matmul_qnn_shared_spike"
    npu_manifest: list[tuple[pathlib.Path, str]] = []
    runtime_build = None
    runtime_api = client_api = runtime_htp_api = client_htp_api = None
    htp_arch = None

    # ------------------------------------------------ local prerequisites
    if npu_requested:
        qroot = pathlib.Path(qairt_root).expanduser()
        if not qroot.is_dir():
            raise Fail(f"--qairt-root not found: {qroot}")
        qroot = qroot.resolve()
        htp_arch = resolve_htp_arch(a.htp_arch)
        up = htp_arch.upper()
        say(f"QAIRT preflight: Hexagon architecture = {htp_arch}")
        arm = qroot / "lib/aarch64-android"
        hexd = qroot / f"lib/hexagon-{htp_arch}/unsigned"
        binp = qroot / "bin/aarch64-android"
        npu_manifest = [
            (arm / "libQnnHtp.so", "libQnnHtp.so"),
            # Required by these clients' graphAddNode/graphFinalize path. The
            # calculator validator does not exercise it and can pass without it.
            (arm / "libQnnHtpPrepare.so", "libQnnHtpPrepare.so"),
            (arm / f"libQnnHtp{up}Stub.so", f"libQnnHtp{up}Stub.so"),
            (arm / f"libQnnHtp{up}CalculatorStub.so", f"libQnnHtp{up}CalculatorStub.so"),
            (arm / "libQnnSystem.so", "libQnnSystem.so"),
            (hexd / f"libQnnHtp{up}Skel.so", f"libQnnHtp{up}Skel.so"),
            (hexd / "libCalculator_skel.so", "libCalculator_skel.so"),
            (binp / "qnn-platform-validator", "qnn-platform-validator"),
            (binp / "qnn-net-run", "qnn-net-run"),
        ]
        for src, _ in npu_manifest:
            require_file(src, "required QAIRT file")

        runtime_api = read_qnn_core_api(qroot / "include/QNN/QnnCommon.h")
        client_api = read_qnn_core_api(
            ROOT / "kernels/matmul/npu/qnn_fp16/include/QNN/QnnCommon.h")
        if runtime_api[0] != client_api[0] or runtime_api[1] < client_api[1]:
            raise Fail(
                f"incompatible QNN core APIs: client {client_api[0]}.{client_api[1]}, "
                f"runtime {runtime_api[0]}.{runtime_api[1]}; require same major and "
                f"runtime minor >= client minor")
        runtime_htp_api = read_qnn_htp_api(qroot / "include/QNN/HTP/QnnHtpCommon.h")
        client_htp_api = read_qnn_htp_api(
            ROOT / "kernels/matmul/npu/qnn_fp16/include/QNN/HTP/QnnHtpCommon.h")
        if runtime_htp_api[0] != client_htp_api[0] or runtime_htp_api[1] < client_htp_api[1]:
            raise Fail(
                f"incompatible QNN HTP APIs: client {client_htp_api[0]}.{client_htp_api[1]}, "
                f"runtime {runtime_htp_api[0]}.{runtime_htp_api[1]}; require same major "
                f"and runtime minor >= client minor")

        build_header = require_file(qroot / "include/QNN/QnnSdkBuildId.h",
                                    "QAIRT build-id header")
        m = re.search(r'#define\s+QNN_SDK_BUILD_ID\s+"([^"]+)"',
                      build_header.read_text(encoding="utf-8", errors="replace"))
        if not m:
            raise Fail(f"cannot parse QAIRT build ID from {build_header}")
        runtime_build = m.group(1)

        require_binary_marker(npu_fp16, "qnn_core_api:",
                              "bash kernels/matmul/npu/qnn_fp16/build.sh")
        require_binary_marker(npu_int8, "qnn_core_api:",
                              "bash kernels/matmul/npu/qnn_int8/build.sh")
        # The spike prints no qnn_core_api: line - it is a Task 12 harness, not
        # a dispatch client - so it is checked on a marker it actually carries.
        require_binary_marker(npu_spike, "shared_impl:",
                              "bash kernels/matmul/npu/qnn_fp16/build.sh")
        say(f"QAIRT preflight: build={runtime_build} "
            f"core_api={'.'.join(map(str, runtime_api))}, "
            f"client_core_api={'.'.join(map(str, client_api))}, "
            f"htp_api={'.'.join(map(str, runtime_htp_api))}, "
            f"client_htp_api={'.'.join(map(str, client_htp_api))}")

    dsp_sweep = ROOT / "kernels/matmul/dsp/prebuilt/matmul_dsp_sweep"
    if dsp_requested:
        dsp_skel = dsp_skel or str(ROOT / "kernels/matmul/dsp/prebuilt/libdsp_matmul_skel.so")
        dsp_stub = dsp_stub or str(ROOT / "kernels/matmul/dsp/prebuilt/libdsp_matmul_stub.so")
        dsp_skel = str(require_file(pathlib.Path(dsp_skel).expanduser(),
                                    "DSP Hexagon skel").resolve())
        dsp_stub = str(require_file(pathlib.Path(dsp_stub).expanduser(),
                                    "DSP ARM64 FastRPC stub").resolve())
        require_binary_marker(dsp_skel, "dsp-power-fail-closed-v2",
                              "bash scripts/build_dsp_skel.sh kernels/matmul/dsp --arch v81")
        require_binary_marker(dsp_sweep, "dsp-power-mode-applied",
                              "bash kernels/matmul/dsp/build_stub.sh")

    # ------------------------------------------------------- preconditions
    say("=== device check ===")
    select_device()
    serial = dev("getprop ro.serialno", "read device serial").strip()
    fingerprint = dev("getprop ro.build.fingerprint", "read build fingerprint").strip()
    kernel = dev("uname -r", "read kernel release").strip()
    if not (serial and fingerprint and kernel):
        raise Fail("device identity read returned empty data")
    say(f"  serial={serial}")
    say(f"  build ={fingerprint}")
    say(f"  kernel={kernel}")
    if "uid=0" not in dev_root_script("id", "root availability check"):
        raise Fail("root unavailable (need su 0)")
    say("  root  = OK")

    dev_root_script(f"""set -eu
mkdir -p {DEV_GPU} {DEV_NPU} {DEV_DSP}
# A failed reprovision must not leave a previously valid manifest available to a
# calibration sampler. Publish a fresh manifest only after every requested file
# hash and live dispatch gate succeeds.
rm -f {DEV_BASE}/runtime-provenance.json
# adb push runs as AID_SHELL (2000) even though install/validation use su.
# Repair ownership as well as creating the directories, so reruns work after an
# older provisioner left root-owned files behind.
chown -R 2000:2000 {DEV_BASE}
chmod 0755 {DEV_BASE} {DEV_GPU} {DEV_NPU} {DEV_DSP}
""", "create shell-writable device directories")

    # ---------------------------------------------------------- CPU kernels
    say("\n=== CPU kernels ===")
    n = 0
    for src, dst in (
        # build_cpu_threaded.ps1 emits matmul_neon_handtuned; matmul_neon is a
        # stale June build with no --threads support, and pushing that one left
        # matmul unthreadable on every board, which Task 6 needs.
        ("kernels/matmul/cpu/neon_handtuned/matmul_neon_handtuned", f"{DEV_BASE}/matmul_neon_handtuned"),
        ("kernels/matmul/cpu/eigen_baseline/matmul_eigen", f"{DEV_BASE}/matmul_eigen_baseline"),
    ):
        p = ROOT / src
        if p.exists():
            push_file(p, dst)
            dev(f"chmod 755 {dst}", "chmod CPU kernel")
            n += 1
            say(f"  pushed {dst.rsplit('/', 1)[-1]}")
        else:
            say(f"  MISSING (not built): {src}")
    summary["cpu_kernels"] = f"{n} pushed"

    # ------------------------------- saxpy / reduction / convolution kernels
    say("\n=== saxpy / reduction / convolution kernels ===")
    extra = [
        ("kernels/saxpy/cpu/scalar_baseline/saxpy_scalar", f"{DEV_BASE}/saxpy/saxpy_scalar"),
        ("kernels/saxpy/cpu/neon_handtuned/saxpy_neon", f"{DEV_BASE}/saxpy/saxpy_neon"),
        ("kernels/saxpy/gpu/opencl_fp32/saxpy_opencl_fp32", f"{DEV_BASE}/saxpy/saxpy_opencl_fp32"),
        ("kernels/reduction/cpu/scalar_baseline/reduction_scalar", f"{DEV_BASE}/reduction/reduction_scalar"),
        ("kernels/reduction/cpu/neon_multiacc/reduction_neon", f"{DEV_BASE}/reduction/reduction_neon"),
        ("kernels/reduction/gpu/opencl_fp32/reduction_opencl_fp32", f"{DEV_BASE}/reduction/reduction_opencl_fp32"),
        ("kernels/convolution/cpu/scalar_baseline/conv_scalar", f"{DEV_BASE}/convolution/conv_scalar"),
        ("kernels/convolution/cpu/neon_channel/conv_neon", f"{DEV_BASE}/convolution/conv_neon"),
        ("kernels/convolution/gpu/opencl_fp32/conv_opencl_fp32", f"{DEV_BASE}/convolution/conv_opencl_fp32"),
        ("kernels/cachelat/cpu/cache_latency", f"{DEV_BASE}/cachelat/cache_latency"),
        ("kernels/memlab/gpu/gpu_memlab", f"{DEV_BASE}/memlab/gpu_memlab"),
        # Lab 3 Tasks 10 and 11 run entirely on these, and none was ever
        # provisioned - they had been pushed by hand on the first board.
        ("kernels/dsp/hello/prebuilt/dsp_hello_test", f"{DEV_DSP}/dsp_hello_test"),
        # dsp_hello_test dlopens this pair by absolute path; without them it
        # exits 0 having printed only a dlopen failure, which reads like a run.
        ("kernels/dsp/hello/prebuilt/libdsp_hello_stub.so", f"{DEV_DSP}/libdsp_hello_stub.so"),
        ("kernels/dsp/hello/prebuilt/libdsp_hello_skel.so", f"{DEV_DSP}/libdsp_hello_skel.so"),
        ("kernels/convolution/dsp/prebuilt/conv_dsp_int8", f"{DEV_DSP}/conv_dsp_int8"),
        ("kernels/convolution/dsp/prebuilt/conv_morph", f"{DEV_DSP}/conv_morph"),
        # conv_morph loads this from its cwd as the default clean image (Task 10 Part D).
        ("kernels/convolution/dsp/data/dog.pgm", f"{DEV_DSP}/dog.pgm"),
        ("kernels/convolution/dsp/prebuilt/conv_stage", f"{DEV_DSP}/conv_stage"),
        ("kernels/convolution/dsp/prebuilt/conv_vtcm_probe", f"{DEV_DSP}/conv_vtcm_probe"),
        ("kernels/saxpy/npu/qnn_fp16/saxpy_qnn_fp16", f"{DEV_NPU}/saxpy_qnn_fp16"),
        ("kernels/reduction/npu/qnn_fp16/reduction_qnn_fp16", f"{DEV_NPU}/reduction_qnn_fp16"),
        ("kernels/convolution/npu/qnn_fp16/conv_qnn_fp16", f"{DEV_NPU}/conv_qnn_fp16"),
        ("kernels/decode/cpu/neon/decode_neon", f"{DEV_BASE}/decode/decode_neon"),
        ("kernels/decode/npu/qnn_fp16/decode_qnn_fp16", f"{DEV_NPU}/decode_qnn_fp16"),
        ("kernels/sol/cpu/cpu_sol", f"{DEV_BASE}/cpu_sol"),
        # Audited against the registry 2026-09-22: these seven were built in
        # the repo and never in the manifest, so every workload below raised a
        # bare "No such file or directory" from the device. Same failure as
        # discovery 85 -- the manifest lagging WORKLOADS.
        ("kernels/convolution/cpu/int8_neon/conv_int8_neon",
         f"{DEV_BASE}/convolution/conv_int8_neon"),
        # nbody is the compute-bound anchor of the roofline and the one
        # workload where the GPU competes on equal terms; both variants are
        # --variant flags on this single binary.
        ("kernels/nbody/cpu/neon_handtuned/nbody_neon", f"{DEV_BASE}/nbody/nbody_neon"),
        ("kernels/nbody/gpu/opencl_fp32/nbody_opencl_fp32",
         f"{DEV_BASE}/nbody/nbody_opencl_fp32"),
        # kernels/stress/ (the Task 5 staff solutions) is deliberately NOT
        # pushed: students write bp_stress/rob_stress themselves under custom/.
        # decode's GPU pair: `gemv` is the 6x kernel the decode ladder exists
        # to show, so without it the workload only demonstrates the naive end.
        ("kernels/decode/gpu/opencl/decode_opencl_fp32",
         f"{DEV_BASE}/decode/decode_opencl_fp32"),
        ("kernels/decode/gpu/opencl_gemv/decode_opencl_gemv",
         f"{DEV_BASE}/decode/decode_opencl_gemv"),
    ]
    # Created as root, but adb push writes as `shell`, so they must be chowned
    # or every push into them fails with "remote couldn't create file".
    dirs = " ".join(f"{DEV_BASE}/{d}" for d in
                    ("saxpy", "reduction", "convolution", "decode",
                     "nbody"))
    dev_root_script(f"set -eu\nmkdir -p {dirs}\nchmod 0755 {dirs}\n"
                    f"chown shell:shell {dirs}\n", "mkdir workload dirs")
    n = 0
    for src, dst in extra:
        p = ROOT / src
        if p.exists():
            push_file(p, dst)
            n += 1
        else:
            say(f"  MISSING (not built): {src}")
    # chmod by search, not by a hand-maintained directory list: adb push from a
    # Windows host creates files 0666, and dirs created implicitly by the push -
    # cachelat, memlab - were never in the old list, so Task 3 and Task 9
    # shipped non-executable.
    dev_root_script(
        f"set -eu\nfind {DEV_BASE} -maxdepth 2 -type f ! -name '*.so' "
        f"! -name '*.json' -exec chmod 755 {{}} +\nchmod -R 755 {DEV_NPU}\n",
        "chmod workload kernels")
    say(f"  pushed {n} CPU/GPU kernels for saxpy, reduction, convolution")
    summary["workload_kernels"] = f"{n} pushed"

    # ---------------------------------------------------------- GPU kernels
    say("\n=== GPU kernels ===")
    n = 0
    gpu_dir = ROOT / "kernels/matmul/gpu"
    if gpu_dir.is_dir():
        for variant in sorted(p for p in gpu_dir.iterdir() if p.is_dir()):
            for f in sorted(variant.iterdir()):
                # each variant dir ships a prebuilt arm64 binary beside its .c
                if f.is_file() and not f.suffix and f.stat().st_size > 1000:
                    push_file(f, f"{DEV_GPU}/{f.name}")
                    n += 1
    dev_root_script(f"set -eu\nchmod -R 755 {DEV_GPU}\n", "chmod GPU kernels")
    say(f"  pushed {n} GPU binaries")
    summary["gpu_kernels"] = f"{n} pushed"

    # ----------------------------------------------------------------- NPU
    say(f"\n=== NPU (QNN HTP {htp_arch if npu_requested else 'not requested'}) ===")
    if not npu_requested:
        say("  SKIPPED: --qairt-root not supplied.")
        say("  NPU dispatch will FAIL without a compatible QAIRT runtime. The")
        say("  device's /vendor/lib64/hw/audio copy is v2.23.1 and is NOT a")
        say("  valid substitute.")
        summary["npu"] = "UNAVAILABLE (no QAIRT supplied)"
    else:
        for src, name in npu_manifest:
            push_verified_runtime_file("npu", src, f"{DEV_NPU}/{name}")
        push_verified_runtime_file("npu", npu_fp16, f"{DEV_NPU}/matmul_qnn_fp16")
        push_verified_runtime_file("npu", npu_int8, f"{DEV_NPU}/matmul_qnn_int8")
        push_verified_runtime_file("npu", npu_spike, f"{DEV_NPU}/matmul_qnn_shared_spike")
        dev_root_script(f"set -eu\nchmod -R 755 {DEV_NPU}\n",
                        "chmod NPU runtime and clients")
        say(f"  pushed {len(npu_manifest)} required QNN files + 2 clients")
        summary["npu"] = f"{runtime_build}; {len(npu_manifest)} QNN files + 2 clients pushed"

    # ----------------------------------------------------------------- DSP
    say("\n=== DSP (FastRPC / Hexagon) ===")
    dsp_pinned = False
    if not dsp_requested:
        say("  SKIPPED: neither --dsp-skel nor --dsp-stub supplied.")
        summary["dsp"] = "UNAVAILABLE (DSP artifacts not requested)"
    else:
        push_verified_runtime_file("dsp", dsp_sweep, f"{DEV_DSP}/matmul_dsp_sweep")
        push_verified_runtime_file("dsp", dsp_stub, f"{DEV_DSP}/libdsp_matmul_stub.so")
        # The FastRPC URI names this exact file; never preserve an arbitrary basename.
        push_verified_runtime_file("dsp", dsp_skel, f"{DEV_DSP}/libdsp_matmul_skel.so")
        # Per discovery 41: FASTRPC_MANAGE_DSP_LIBRARY_PATH is EUNSUPPORTED here
        # and ADSP_LIBRARY_PATH is NOT honored for skel lookup -> the skel must
        # be in CWD (dispatch cds into DEV_DSP) or a vendor RPCFS path.
        dev_root_script(
            f"set -eu\nchmod 755 {DEV_DSP}/matmul_dsp_sweep "
            f"{DEV_DSP}/libdsp_matmul_stub.so {DEV_DSP}/libdsp_matmul_skel.so\n",
            "chmod DSP artifacts")
        say("  pushed sweep + mandatory ARM64 stub + Hexagon skel")
        summary["dsp"] = "sweep + stub + skel pushed (CWD-resolved)"

        # The sweep client pins DSP power and fail-closes when refused, and HAP
        # power voting is rejected from an unsigned PD (0x80000414, discovery
        # 64). These direct clients do not vote, so they are what the registry
        # dispatches; they need their own stub/skel pairs beside them.
        say("\n=== DSP direct clients (unpinned; see discovery 64) ===")
        n = 0
        for rel in (
            "kernels/matmul/dsp/prebuilt/matmul_dsp_scalar",
            "kernels/matmul/dsp/prebuilt/matmul_dsp_hvx",
            "kernels/matmul/dsp/prebuilt/matmul_dsp_vtcm",
            "kernels/saxpy/dsp/prebuilt/saxpy_dsp",
            "kernels/saxpy/dsp/prebuilt/libdsp_saxpy_stub.so",
            "kernels/saxpy/dsp/prebuilt/libdsp_saxpy_skel.so",
            "kernels/reduction/dsp/prebuilt/reduction_dsp",
            "kernels/reduction/dsp/prebuilt/libdsp_reduction_stub.so",
            "kernels/reduction/dsp/prebuilt/libdsp_reduction_skel.so",
            "kernels/convolution/dsp/prebuilt/conv_dsp",
            "kernels/convolution/dsp/prebuilt/libdsp_conv_stub.so",
            "kernels/convolution/dsp/prebuilt/libdsp_conv_skel.so",
        ):
            p = ROOT / rel
            if p.exists():
                push_file(p, f"{DEV_DSP}/{p.name}")
                n += 1
            else:
                say(f"  MISSING (build it: bash scripts/build_dsp_skel.sh "
                    f"kernels/<k>/dsp --arch v81, then build_stub.sh): {rel}")
        dev_root_script(f"set -eu\nchmod -R 755 {DEV_DSP}\n", "chmod DSP direct clients")
        say(f"  pushed {n} direct clients + their stub/skel pairs")
        summary["dsp_direct"] = f"{n} pushed"

    # -------------------------------------------------------------- verify
    if npu_requested:
        say("\n=== verify: qnn-platform-validator ===")
        env = (f"cd {DEV_NPU}\nexport LD_LIBRARY_PATH={DEV_NPU}\n"
               f"export ADSP_LIBRARY_PATH={DEV_NPU}\n")
        out = dev_root_script("set -eu\n" + env +
                              "./qnn-platform-validator --backend dsp --testBackend\n",
                              "qnn-platform-validator")
        say(out)
        if not re.search(r"Unit Test[^\r\n]*Passed", out, re.I):
            raise Fail("qnn-platform-validator exited zero without a passing "
                       "unit-test result")
        summary["npu_validator"] = "PASS"

        say("\n=== verify: real FP16 HTP graph in sustained mode ===")
        out = dev_root_script("set -eu\n" + env +
                              "./matmul_qnn_fp16 --n 512 --reps 2 --perf-mode sustained\n",
                              "sustained FP16 NPU client")
        say(out)
        api_ok = r"^qnn_core_api:\s*2\.(?:3[0-9]|[4-9][0-9]|[1-9][0-9]{2,})\."
        if (not re.search(r"^status:\s*SUCCESS\s*$", out, re.M)
                or not re.search(r"^power_mode:\s*sustained_high_performance\s*$", out, re.M)
                or not re.search(api_ok, out, re.M)):
            raise Fail("FP16 client did not prove SUCCESS + sustained mode + "
                       "compatible QNN core API")
        summary["npu_fp16_client"] = "PASS (real sustained FP16 graph)"

        say("\n=== verify: real INT8 HTP graph in sustained mode ===")
        out = dev_root_script("set -eu\n" + env +
                              "./matmul_qnn_int8 --n 512 --reps 2 --perf-mode sustained\n",
                              "sustained INT8 NPU client")
        say(out)
        if (not re.search(r"^status:\s*SUCCESS\s*$", out, re.M)
                or not re.search(r"^precision:\s*int8\s*$", out, re.M)
                or not re.search(r"^power_mode:\s*sustained_high_performance\s*$", out, re.M)
                or not re.search(api_ok, out, re.M)):
            raise Fail("INT8 client did not prove SUCCESS + INT8 + sustained "
                       "mode + compatible QNN core API")
        summary["npu_int8_client"] = "PASS (real sustained INT8 graph)"

    if dsp_requested:
        # What is gated here is a real FastRPC dispatch returning the known-good
        # fingerprint. Clock *pinning* is reported, not required: this board
        # family runs compute in an unsigned PD where HAP_power_request_abs is
        # refused with 0x80000414 (discovery 64, ADR 0022). Gating on an
        # accepted LAB vote would fail-closed on every board we have.
        say("\n=== verify: DSP VTCM dispatch (clock pinning reported, not required) ===")
        pin_out = dev_root_script(
            f"cd {DEV_DSP}\nexport LD_LIBRARY_PATH=.\n"
            f"./matmul_dsp_sweep --variant vtcm --power lab --warmup 1 "
            f"--measure 1 --size 128 --quiet 2>&1 || true\n",
            "DSP LAB power-vote probe")
        dsp_pinned = bool(re.search(
            r"^dsp-power-mode-applied mode=lab api_status=accepted\s*$", pin_out, re.M))
        if dsp_pinned:
            say("  DSP clock pinning: AVAILABLE (LAB vote accepted)")
        else:
            say("  DSP clock pinning: UNAVAILABLE (unsigned PD refuses HAP power "
                "votes, 0x80000414)")
            say("  -> DSP timings are unpinned and vary more than the other backends.")

        out = dev_root_script(f"set -eu\ncd {DEV_DSP}\nexport LD_LIBRARY_PATH=.\n"
                              f"./matmul_dsp_hvx --size 256\n", "DSP HVX dispatch")
        say(out)
        m = re.search(r"^matmul-dsp_hvx-end fingerprint=([0-9.eE+-]+)", out, re.M)
        if not m:
            raise Fail("DSP validation did not produce a completed HVX dispatch")
        # N=256, seed=42 reference; stable across SM8750 (v79) and SM8850 (v81).
        fp_val = float(m.group(1))
        if abs(fp_val - 1358.729) > 0.01:
            raise Fail(f"DSP dispatch produced fingerprint {fp_val}, expected "
                       f"1358.729 (matmul N=256 seed=42)")
        say(f"  fingerprint {fp_val} matches reference 1358.729")
        pin_note = "pinned" if dsp_pinned else "unpinned (0x80000414)"
        summary["dsp_verify"] = f"PASS (HVX dispatch, fingerprint verified; clocks {pin_note})"

    # -------------------------------------------------- provenance manifest
    # Identifiers and hashes only, never licensed Qualcomm payloads or host
    # paths. Published last: samplers fail closed while a provision attempt is
    # incomplete or any live verification above has failed.
    qairt_provenance = None
    if npu_requested:
        qairt_provenance = {
            "sdk_build_id": runtime_build,
            "provider_core_api": ".".join(map(str, runtime_api)),
            "provider_htp_api": ".".join(map(str, runtime_htp_api)),
            "client_core_api": ".".join(map(str, client_api)),
            "client_htp_api": ".".join(map(str, client_htp_api)),
        }
    manifest = {
        "schema_version": 1,
        # Name kept for the power-calibration consumers that match on it.
        "manifest_kind": "hdk8750-runtime-provenance",
        "hash_algorithm": "sha256",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device": {
            "serial": serial,
            "android_build_fingerprint": fingerprint,
            "kernel_release": kernel,
        },
        "qairt": qairt_provenance,
        "validation": {
            "npu_requested": npu_requested,
            "npu_platform_validator_passed": npu_requested,
            "npu_fp16_sustained_graph_passed": npu_requested,
            "npu_int8_sustained_graph_passed": npu_requested,
            "dsp_requested": dsp_requested,
            # Kept under its original name for the power-calibration consumers.
            # It now means "the DSP dispatch gate passed", which no longer
            # requires an accepted LAB power vote -- see dsp_clocks_pinned.
            "dsp_lab_dispatch_passed": dsp_requested,
            # False on every board without a testsig; timings are then unpinned.
            "dsp_clocks_pinned": dsp_pinned,
        },
        "files": deployed_runtime_files,
    }
    text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    json.loads(text)  # never publish something we cannot read back

    device_manifest = f"{DEV_BASE}/runtime-provenance.json"
    pending = f"{DEV_BASE}/.runtime-provenance.{os.getpid()}.pending"
    fd, local = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    published = False
    try:
        pathlib.Path(local).write_bytes(text.encode("utf-8"))
        # Stage under a name the sampler never reads. Only an exact-hash-verified
        # file is renamed to the fixed manifest path.
        push_file(local, pending)
        hh = host_sha256(local)
        if device_sha256(pending) != hh:
            raise Fail("runtime provenance manifest SHA-256 mismatch after push")
        dev(f"chmod 0444 {pending}", "chmod pending runtime provenance manifest")
        dev(f"mv '{pending}' '{device_manifest}'", "publish runtime provenance manifest")
        if device_sha256(device_manifest) != hh:
            raise Fail("runtime provenance manifest SHA-256 changed during publication")
        published = True
        say("\n=== runtime provenance ===")
        say(f"  manifest={device_manifest}")
        say(f"  SHA-256={hh}")
        summary["provenance"] = (f"PASS ({len(deployed_runtime_files)} NPU/DSP files; "
                                 f"manifest {hh})")
    finally:
        if not published and device_serial:
            try:
                dev_root_script(f"set -eu\nrm -f '{pending}' '{device_manifest}'\n",
                                "remove incomplete runtime provenance")
            except Fail as exc:
                say(f"warning: could not remove incomplete device provenance: {exc}")
        pathlib.Path(local).unlink(missing_ok=True)

    say("\n=== provisioning summary ===")
    for k, v in summary.items():
        say(f"  {k:<16} {v}")
    say(f"\ndevice dirs: {DEV_BASE} (plus /gpu, /npu, /dsp)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Fail as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        sys.exit(1)
