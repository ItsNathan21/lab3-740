"""verify_board.py - is this HDK8850 healthy enough to measure on?

The compute counterpart to `check_display.ps1`, and cross-platform: pure stdlib,
runs the same on macOS, Windows and Linux.

    python scripts/verify_board.py                 # human-readable
    python scripts/verify_board.py --json out.json # machine-readable too
    python scripts/verify_board.py --quick         # skip the dispatch tests

Exit code 0 if nothing FAILed, 1 otherwise. WARN and SKIP do not fail the run:
NPU and DSP legitimately need SDK runtimes that are not in this repository, so a
board without them is *incomplete*, not *broken*.

Status meanings
  PASS  checked and correct
  FAIL  checked and wrong -> exit 1
  WARN  suspicious but not disqualifying
  SKIP  could not check (missing prerequisite, e.g. no root)
  INFO  recorded for context, never judged
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVICE_BASE = "/data/local/tmp/profiler-kernels"

PASS, FAIL, WARN, SKIP, INFO = "PASS", "FAIL", "WARN", "SKIP", "INFO"
_COLOR = {PASS: "\033[32m", FAIL: "\033[31m", WARN: "\033[33m", SKIP: "\033[90m", INFO: "\033[36m"}
_RESET = "\033[0m"

checks: list[dict] = []
_serial: str | None = None


def add(group: str, name: str, status: str, detail: str = "", hint: str = "") -> None:
    checks.append({"group": group, "name": name, "status": status, "detail": detail, "hint": hint})
    color = _COLOR.get(status, "") if sys.stdout.isatty() else ""
    reset = _RESET if color else ""
    line = f"  [{color}{status:4s}{reset}] {name}"
    if detail:
        line += f" - {detail}"
    print(line)
    if hint and status in (FAIL, WARN):
        print(f"         hint: {hint}")


def adb(*args: str, timeout: int = 30) -> tuple[int, str]:
    """Run an adb command. Returns (rc, combined output)."""
    cmd = ["adb"]
    if _serial:
        cmd += ["-s", _serial]
    cmd += list(args)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, "adb not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"


def sh(cmd: str, timeout: int = 30) -> tuple[int, str]:
    return adb("shell", cmd, timeout=timeout)


# ----------------------------------------------------------------- host + link
def check_host_and_device(want_serial: str | None) -> bool:
    global _serial
    rc, out = adb("version")
    if rc == 127:
        add("host", "adb on PATH", FAIL, "not found",
            "install platform-tools and add it to PATH")
        return False
    add("host", "adb on PATH", PASS, out.splitlines()[0] if out else "")

    rc, out = adb("devices")
    if rc != 0:
        add("host", "adb server", FAIL, out)
        return False
    devices = [l.split()[0] for l in out.splitlines()[1:] if l.strip().endswith("device")]
    unauth = [l for l in out.splitlines()[1:] if "unauthorized" in l]
    if unauth:
        add("device", "authorization", FAIL, "device unauthorized",
            "accept the USB-debugging prompt on the board, then re-run")
        return False
    if not devices:
        add("device", "attached", FAIL, "no device in 'device' state",
            "check the USB-C cable and that the board is powered")
        return False
    if want_serial:
        if want_serial not in devices:
            add("device", "attached", FAIL, f"{want_serial} not among {devices}")
            return False
        _serial = want_serial
    elif len(devices) > 1:
        add("device", "attached", FAIL, f"{len(devices)} devices: {devices}",
            "pass --serial <id> to choose one")
        return False
    else:
        _serial = devices[0]
    add("device", "attached", PASS, _serial)
    return True


def check_identity() -> None:
    for label, prop in (("serial", "ro.serialno"),
                        ("build", "ro.build.fingerprint"),
                        ("android", "ro.build.version.release")):
        rc, out = sh(f"getprop {prop}")
        add("device", label, INFO if rc == 0 else SKIP, out if rc == 0 else "unreadable")
    rc, out = sh("uname -r")
    add("device", "kernel", INFO if rc == 0 else SKIP, out)


def check_root() -> bool:
    rc, out = sh("su 0 id -u")
    rooted = rc == 0 and out.strip().endswith("0")
    if rooted:
        add("device", "root", PASS, "su 0 available")
    else:
        add("device", "root", WARN, "not available",
            "frequency pinning and GPU/EAS nodes need root; measurements will be noisier")
    return rooted


# --------------------------------------------------------------- provisioning
def check_kernels() -> dict[str, bool]:
    rc, out = sh(f"ls {DEVICE_BASE} 2>/dev/null")
    if rc != 0 or not out:
        add("provision", "kernel tree", FAIL, f"{DEVICE_BASE} missing or empty",
            "run scripts/provision_device.py (setup guide: Provision the board)")
        return {}
    entries = set(out.split())
    add("provision", "kernel tree", PASS, f"{len(entries)} entries")

    present = {}
    for backend, probe in (("cpu", "matmul_neon_handtuned"), ("gpu", "gpu"),
                           ("npu", "npu"), ("dsp", "dsp")):
        ok = probe in entries
        present[backend] = ok
        if ok:
            add("provision", f"{backend} artifacts", PASS, probe)
        elif backend in ("npu", "dsp"):
            add("provision", f"{backend} artifacts", SKIP, f"{probe}/ absent",
                "needs QAIRT (NPU) / Hexagon skel (DSP) - not in this repo")
        else:
            add("provision", f"{backend} artifacts", FAIL, f"{probe} absent",
                "re-run provisioning")
    return present


# ------------------------------------------------------------ control surface
def check_frequency_control(rooted: bool) -> None:
    for pol, name in (("policy0", "CPU Gold"), ("policy6", "CPU Prime")):
        rc, out = sh(f"cat /sys/devices/system/cpu/cpufreq/{pol}/scaling_cur_freq")
        if rc == 0 and out.strip().isdigit():
            add("control", f"{name} freq readable", PASS, f"{int(out)//1000} MHz")
        else:
            add("control", f"{name} freq readable", FAIL, out or "unreadable")

    if not rooted:
        add("control", "GPU pwrlevel", SKIP, "needs root")
        add("control", "EAS energy model", SKIP, "needs root")
        return

    # Adreno 830 (SM8750) exposed the current clock under kgsl's devfreq/
    # symlink; Adreno 840 (SM8850) dropped that subdirectory and keeps only
    # `gpuclk`.  `gpuclk` is present on both, so try it first.
    for node in ("gpuclk", "devfreq/cur_freq"):
        rc, out = sh(f"su 0 cat /sys/class/kgsl/kgsl-3d0/{node}")
        if rc == 0 and out.strip().isdigit():
            add("control", "GPU clock readable", PASS,
                f"{int(out)//1_000_000} MHz (via {node})")
            break
    else:
        add("control", "GPU clock readable", WARN, out or "unreadable")

    # The power model needs this; absence is not fatal for dispatch.
    # SM8850 boards ship with debugfs unmounted, which hides the energy model
    # (and /sys/kernel/debug/regulator).  debugfs is a virtual filesystem, so
    # mounting it is harmless and reversible -- but it does not survive a
    # reboot, hence the attempt here rather than in one-time provisioning.
    rc, out = sh("su 0 ls /sys/kernel/debug/energy_model/")
    if rc != 0 or not out:
        mrc, _ = sh("su 0 mount -t debugfs none /sys/kernel/debug")
        if mrc == 0:
            rc, out = sh("su 0 ls /sys/kernel/debug/energy_model/")
    add("control", "EAS energy model", PASS if rc == 0 and out else WARN,
        out.replace(chr(10), " ") if rc == 0 else "unreadable",
        "" if rc == 0 else
        "power model needs this; dispatch does not. Try: "
        "adb shell su 0 mount -t debugfs none /sys/kernel/debug")


def check_thermal() -> None:
    """A hot board measures differently. Report, and warn if it starts warm.

    The zone whitelist lives in profiler.py and is shared, because this check
    previously carried its own copy that had drifted: it filtered on `*cpuss*`,
    a name this SoC does not use, and so read the GPU and DDR while being blind
    to all 16 CPU sensors and all 8 Hexagon ones. The whitelist also excludes
    `cpu-hw-trip-0/1`, which are trip-point constants reading 105000 rather
    than sensors.
    """
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        from profiler import _THERMAL_ZONE_RE
    except Exception as exc:
        add("thermal", "zones", SKIP, f"cannot import the zone list: {exc}")
        return

    rc, out = sh('for z in /sys/class/thermal/thermal_zone*; do '
                 'echo "${z##*/} $(cat $z/type 2>/dev/null)"; done')
    if rc != 0 or not out:
        add("thermal", "zones", SKIP, "unreadable")
        return
    paths, types = [], []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and _THERMAL_ZONE_RE.match(parts[1]):
            paths.append(f"/sys/class/thermal/{parts[0]}/temp")
            types.append(parts[1])
    if not paths:
        add("thermal", "zones", SKIP, "no recognised zones")
        return

    rc, out = sh("cat " + " ".join(paths))
    temps = {}
    for zone, raw in zip(types, out.split()):
        raw = raw.strip()
        if raw.lstrip("-").isdigit():
            c = int(raw) / 1000.0
            if -40.0 < c < 200.0:
                temps[zone] = c
    if not temps:
        add("thermal", "zones", SKIP, "no parseable zones")
        return
    zone, hi = max(temps.items(), key=lambda kv: kv[1])
    if hi >= 70:
        add("thermal", "starting temperature", WARN, f"{hi:.0f} C on {zone}",
            "let the board cool; a hot start distorts the first measurements")
    else:
        add("thermal", "starting temperature", PASS,
            f"{hi:.0f} C max ({zone}, {len(temps)} zones)")


# ------------------------------------------------------------------- dispatch
def check_dispatch(present: dict[str, bool]) -> None:
    """The real test: can each backend actually run something?"""
    try:
        from profiler import Backend, Profiler, matmul
    except Exception as e:  # pragma: no cover - import guard
        add("dispatch", "import profiler.py", FAIL, str(e))
        return

    p = Profiler()
    for backend in (Backend.CPU, Backend.GPU, Backend.DSP, Backend.NPU):
        key = str(backend)
        if present and not present.get(key, True):
            add("dispatch", f"{key} dispatch", SKIP, "artifacts not provisioned")
            continue
        kw = {"size": 256}
        if key == "gpu":
            # Most register-blocked GPU variants are compiled for one shape;
            # 'best' asserts --size 512. Use its size rather than the default.
            kw = {"size": 512, "variant": "best"}
        t0 = time.time()
        try:
            r = p.run(matmul(backend, **kw))
        except Exception as e:
            add("dispatch", f"{key} dispatch", FAIL, f"raised {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        if r.ok and r.compute_us:
            add("dispatch", f"{key} dispatch", PASS,
                f"{r.compute_us:.0f} us compute, {dt:.1f} s wall")
        elif r.ok:
            add("dispatch", f"{key} dispatch", WARN, "ran but produced no compute time")
        else:
            hint = ("needs QAIRT / Hexagon skel - see scripts/provision_device.py"
                    if key in ("npu", "dsp") else "")
            add("dispatch", f"{key} dispatch", FAIL, (r.error or "unknown")[:120], hint)


# ----------------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Verify an HDK8850 is healthy enough to measure on.")
    ap.add_argument("--serial", help="adb serial, if more than one device is attached")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    ap.add_argument("--quick", action="store_true", help="skip the dispatch tests (no kernels run)")
    args = ap.parse_args(argv[1:])

    print("HDK8850 board verification\n")

    print("Host and link:")
    if not check_host_and_device(args.serial):
        return _finish(args.json)

    print("\nDevice:")
    check_identity()
    rooted = check_root()

    print("\nProvisioning:")
    present = check_kernels()

    print("\nControl surface:")
    check_frequency_control(rooted)

    print("\nThermal:")
    check_thermal()

    if args.quick:
        print("\nDispatch:  skipped (--quick)")
    else:
        print("\nDispatch (this runs real kernels):")
        check_dispatch(present)

    return _finish(args.json)


def _finish(json_path: str | None) -> int:
    n = {s: sum(1 for c in checks if c["status"] == s) for s in (PASS, FAIL, WARN, SKIP, INFO)}
    print(f"\n{n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail, {n[SKIP]} skipped")

    if json_path:
        pathlib.Path(json_path).write_text(json.dumps(
            {"serial": _serial, "checks": checks, "summary": n}, indent=2), encoding="utf-8")
        print(f"wrote {json_path}")

    if n[FAIL]:
        print("\nBOARD CHECK FAILED - fix the FAIL lines above before measuring.")
        return 1
    if n[SKIP]:
        print("\nBOARD OK (some checks skipped - see SKIP lines; NPU/DSP need SDK artifacts).")
    else:
        print("\nBOARD OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
