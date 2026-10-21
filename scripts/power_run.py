"""power_run.py - run a workload and get a modeled power/energy figure. No meter.

    python scripts/power_run.py cpu
    python scripts/power_run.py cpu gpu npu cpu+gpu      # several, in sequence

Automates section G of the calibration run-sheet: run the workload under the
intent wrapper plus the sampler, pull the telemetry, build a `run`-split session,
and estimate against a fitted artifact. The output is MODELED whole-board power,
never a measurement and never a per-engine split.

Engines the artifact was not fitted for come back `unavailable` with a reason.
That is the correct answer, not a failure - with the shipped v1 artifact the NPU
and DSP are exactly that until the v2 bench session lands.

NPU runs carry the same two costs calibration does (discovery 67): the client
must exit on its own for the runtime proof, and the window needs a thermal
plateau first, because the fit rejects a Hexagon span over 10 C whatever the
split.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "/data/local/tmp/powerrun"
KER = "/data/local/tmp/profiler-kernels"
NPU = f"{KER}/npu"
GPU = f"{KER}/gpu"
# v2, not v1: the shipped v1 artifact declares model_equation_version
# 'teaching-power-v1' and artifact_schema_version 1, and the current estimator
# understands only 'teaching-power-v2' / schema 2, so it is refused on load.
# v2 does not exist until the bench session is fitted - hence the check below.
def _default_artifact() -> pathlib.Path:
    """First artifact that exists, most-capable first. Keeps callers stable when a
    Hexagon calibration lands: nothing has to be re-pointed."""
    models = ROOT / "references" / "phase-5-energy" / "models"
    # SM8850 first. Its artifact is board-gated (power_estimate.py refuses a
    # serial mismatch), so listing both boards is safe and the attached one
    # wins. Without the SM8850 entry this returned the SM8750 model on an
    # SM8850 board, which then gated itself off and produced no watts at all --
    # a silent empty result rather than an error. profiler.py already resolved
    # this correctly; only direct power_run.py callers were affected.
    for name in ("sm8850-teaching-v2-cpu-gpu.json",
                 "sm8750-teaching-v2-npu.json",
                 "sm8750-teaching-v2-cpu-gpu.json"):
        if (models / name).exists():
            return models / name
    return models / "sm8850-teaching-v2-cpu-gpu.json"


DEFAULT_ARTIFACT = _default_artifact()

DURATION_S = 40            # > the 30 s minimum_validated_interval_s floor
GOLD_MASKS = ("1", "2", "4", "8", "10", "20")     # Gold cores 0-5

# CPU/GPU loads are loops in pushed device scripts: they outlast any window and
# are stopped afterwards. Keeping them in files avoids nested quoting through
# `adb shell`, which is the usual way these worker strings break.
DEVICE_SCRIPTS = ("pr_cpu_loop.sh", "pr_gpu_loop.sh", "pr_stop.sh")
# Never `adb shell pkill -f <pattern>`: the invoking shell carries the pattern
# in its own command line, so pkill kills that shell and the target can
# survive. pr_stop.sh keeps the patterns in a file. See its header.
STOP = "sh /data/local/tmp/pr_stop.sh"
CPU_LOOP = "sh /data/local/tmp/pr_cpu_loop.sh"
GPU_CMD = "sh /data/local/tmp/pr_gpu_loop.sh"
NPU_REPS = 1_811_320       # ~240 s at the measured 132.5 us/rep
NPU_CMD = (f"cd {NPU} && LD_LIBRARY_PATH={NPU} ADSP_LIBRARY_PATH={NPU} "
           f"./matmul_qnn_fp16 --n 512 --perf-mode sustained --reps {NPU_REPS}")

PRESETS: dict[str, dict] = {
    "cpu":     {"engine": "cpu",      "workers": [f"{CPU_LOOP} {m}" for m in GOLD_MASKS]},
    "gpu":     {"engine": "gpu",      "workers": [GPU_CMD]},
    "npu":     {"engine": "npu",      "workers": [NPU_CMD]},
    # Labelled "gpu", not "combined": power_build_session.py accepts a
    # `combined` engine_class but the sampler does not emit one, and the build
    # requires the raw header and the manifest to agree - so `combined` is
    # unreachable. The label costs nothing here. The model's CPU terms
    # (X_gold, X_prime) are driven by per-cluster frequency and utilisation
    # telemetry, not by this field; only the GPU/NPU/DSP indicator terms are
    # gated by it. A CPU+GPU window tagged "gpu" therefore still charges for
    # both loads.
    "cpu+gpu": {"engine": "gpu", "workers": [f"{CPU_LOOP} {m}" for m in GOLD_MASKS]
                                            + [GPU_CMD]},
}

PLATEAU_POLL_S, PLATEAU_HISTORY_S, PLATEAU_SPAN_C, PLATEAU_CAP_S = 5, 30, 3.0, 150
HEX_ZONES = (48, 49, 50, 51, 52, 53, 54)


def sh(*a: str, timeout: int = 300) -> tuple[int, str]:
    p = subprocess.run(["adb", *a], capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def wait_for_plateau() -> bool:
    hist: list[list[float]] = []
    need = PLATEAU_HISTORY_S // PLATEAU_POLL_S
    for waited in range(0, PLATEAU_CAP_S, PLATEAU_POLL_S):
        _, out = sh("shell", " ".join(
            f"cat /sys/class/thermal/thermal_zone{z}/temp;" for z in HEX_ZONES))
        vals = [int(v) / 1000.0 for v in out.split() if v.strip().lstrip("-").isdigit()]
        hist = (hist + [vals])[-need:]
        if len(hist) == need and all(len(s) == len(HEX_ZONES) for s in hist):
            span = max(max(z) - min(z) for z in zip(*hist))   # per zone, over time
            if span <= PLATEAU_SPAN_C:
                print(f"    plateau after {waited}s (hottest {max(hist[-1]):.0f} C, "
                      f"drift {span:.1f} C)")
                return True
        time.sleep(PLATEAU_POLL_S)
    print("    never reached a thermal plateau")
    return False


def run_one(name: str, out: pathlib.Path, artifact: pathlib.Path,
            custom: dict | None = None) -> dict | None:
    preset = custom or PRESETS[name]
    engine, workers = preset["engine"], preset["workers"]
    rid = name.replace("+", "_")
    print(f"\n=== {name}  (engine_class={engine}, {len(workers)} worker(s)) ===")

    # The wrapper refuses to start over stale intent files, so clear them first.
    sh("shell", f"rm -f {DEV}/{rid}.* {DEV}/WL_ACTIVE {DEV}/WL_PIDS 2>/dev/null")
    sh("shell", f"mkdir -p {DEV}")
    quoted = " ".join(f"'{w}'" for w in workers)
    sh("shell", f"nohup sh /data/local/tmp/power_workload_wrapper.sh {DEV} {rid} "
                f"{quoted} >/dev/null 2>&1 &")

    if engine == "npu":
        print("    warming up under load ...")
        if not wait_for_plateau():
            sh("shell", STOP)
            return None
    else:
        time.sleep(8)      # let clocks ramp and the workers actually start

    _, alive = sh("shell", f"cat {DEV}/WL_PIDS 2>/dev/null | wc -l")
    print(f"    {alive.strip()} worker PID(s) registered; sampling {DURATION_S}s")
    # Zero PIDs means every worker died before the window opened, and the window
    # is already lost -- the sampler's liveness check will fail each sample and
    # report BAD_READS with no hint as to why. The reason is sitting in the
    # workload log, so surface it here instead of making the caller go and find
    # it. The usual cause is an unset execute bit on a freshly pushed binary:
    # `adb push` does not preserve it, and the worker exits 126 instantly.
    if alive.strip() in ("0", ""):
        _, why = sh("shell", f"tail -6 {DEV}/{rid}.workload.log 2>/dev/null")
        print("    NO WORKERS STARTED -- the window will fail. Last log lines:")
        for line in (why or "(workload log is empty)").splitlines():
            print(f"      {line}")
        print("    If this says 'Permission denied' (exit 126), the binary is "
              "not executable:")
        print("      adb shell chmod 755 <path to the binary>")
    sh("shell", f"su 0 nohup sh /data/local/tmp/power_calib_sampler.sh "
                f"{rid} {DURATION_S} 1 {DEV} 1 {engine} >/dev/null 2>&1 &")

    for _ in range(DURATION_S + 60):
        _, ls = sh("shell", f"ls {DEV}/{rid}.done {DEV}/{rid}.fail 2>/dev/null")
        if ".fail" in ls:
            _, why = sh("shell", f"cat {DEV}/{rid}.fail")
            print(f"    SAMPLER FAILED: {why[:160]}")
            sh("shell", STOP)
            return None
        if ".done" in ls:
            break
        time.sleep(1)
    else:
        print("    sampler timed out")
        sh("shell", STOP)
        return None

    if engine == "npu":
        print("    waiting for the NPU client to exit cleanly ...")
        for _ in range(180):
            _, log = sh("shell", f"cat {DEV}/{rid}.workload.log 2>/dev/null")
            if f"=WORKLOAD_EXIT window_id={rid} status=0" in log:
                break
            time.sleep(2)
    _, stopped = sh("shell", STOP)
    print(f"    {stopped}")
    time.sleep(2)

    for ext in ("raw", "done", "workload.log",
                "runtime-provenance.json", "runtime-artifacts.sha256"):
        sh("pull", f"{DEV}/{rid}.{ext}", str(out / f"{rid}.{ext}"))
    sh("pull", f"{DEV}/eas.txt", str(out / "eas.txt"))
    if not (out / f"{rid}.raw").exists():
        print("    no telemetry pulled")
        return None

    window: dict = {"split": "run", "engine_class": engine}
    if engine in ("gpu", "combined"):
        window["gpu_active"] = True
    if engine == "npu":
        window.update({"npu_active": True, "perf_mode": "sustained",
                       "workload_variant": "qnn_fp16", "matrix_size": 512,
                       "workload_cmd": NPU_CMD})
    _, serial = sh("shell", "getprop ro.serialno")
    _, fp = sh("shell", "getprop ro.build.fingerprint")
    _, kr = sh("shell", "uname -r")
    manifest = {
        "session": {"calibration_board_serial": serial,
                    "android_build_fingerprint": fp, "kernel_release": kr,
                    "supply": "none - modeled run, no meter",
                    "notes": f"power_run.py preset {name}"},
        "model_spec": {"knots_hz": [222000000.0, 660000000.0, 1100000000.0],
                       "model_version": "run-telemetry",
                       "minimum_validated_interval_s": 30.0},
        "windows": {rid: window},
    }
    mpath, tpath = out / f"{rid}.manifest.json", out / f"{rid}.telemetry.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tpath.unlink(missing_ok=True)     # sessions are immutable; allow a re-run
    r = subprocess.run([sys.executable, "scripts/power_build_session.py",
                        str(mpath), str(out), str(tpath)],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print("    BUILD FAILED:")
        for ln in (r.stdout + r.stderr).strip().splitlines()[:6]:
            print(f"      {ln}")
        return None

    summarize_features(tpath)
    r = subprocess.run([sys.executable, "scripts/power_estimate.py",
                        str(artifact), str(tpath)],
                       capture_output=True, text=True, cwd=ROOT)
    print("    -- estimate --")
    for ln in ((r.stdout + r.stderr).strip() or "(no output)").splitlines()[:12]:
        print(f"    {ln}")
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {"raw_output": (r.stdout + r.stderr).strip()}


def summarize_features(tpath: pathlib.Path) -> None:
    """Print the model's INPUTS for this window.

    Worth showing separately from the estimate: these are measured telemetry and
    are meaningful even when no artifact can turn them into watts. They are also
    where an implausible power number would come from - a GPU window with
    gpu_busy near zero, or a CPU window with x_gold near zero, means the load
    never landed, whatever the estimator then prints.
    """
    sys.path.insert(0, str(ROOT))
    from tools.teaching_power_estimator import aggregate as A
    s = json.loads(tpath.read_text())
    eas = {k: [(int(f), float(p)) for f, p in rows] for k, rows in s["eas_tables"].items()}
    knots = [float(k) for k in s["model_spec"]["knots_hz"]]
    for w in s["windows"]:
        # The build step nests the active flags under "condition"; reading them
        # from the top level silently yields all-False, which zeroes the GPU
        # basis terms and the NPU fraction while gpu_busy still reads 0.99.
        c = w.get("condition") or {}
        a = A.aggregate_window(w["telemetry_samples"], eas, knots,
                               gpu_active=bool(c.get("gpu_active")),
                               npu_active=bool(c.get("npu_active")),
                               dsp_active=bool(c.get("dsp_active")))
        g = a.get("gpu_busy_mean")
        f = a.get("gpu_freq_hz")
        print(f"    -- model inputs ({a['n_samples']} samples over {a['window_s']:.1f}s) --")
        print(f"      activity={a['activity_fraction']:.3f}  "
              f"X_gold={a['x_gold']:.3f} W  X_prime={a['x_prime']:.3f} W")
        print(f"      gpu_busy={'n/a' if g is None else f'{g:.3f}'}  "
              f"gpu_freq={'n/a' if f is None else f'{f/1e6:.0f} MHz'}  "
              f"G=({a['gpu_g0']:.3f},{a['gpu_g1']:.3f},{a['gpu_g2']:.3f})  "
              f"npu={a['npu_active']:.2f}")
        hx = [v for k, v in a.get("temperature_ranges_c", {}).items()
              if k.startswith(("nsphvx", "nsphmx"))]
        if hx:
            print(f"      hexagon {min(v['min_c'] for v in hx):.0f}-"
                  f"{max(v['max_c'] for v in hx):.0f} C")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # No `choices=` here: with nargs="*" argparse validates the DEFAULT list
    # against choices and rejects the empty list, breaking --cmd-only runs.
    ap.add_argument("presets", nargs="*", default=None, metavar="PRESET",
                    help=f"built-in workloads to run, in order: {', '.join(sorted(PRESETS))}")
    ap.add_argument("--cmd", help="measure YOUR OWN workload: a device-side command. "
                                  "It must outlast the window; loop it if it is short.")
    ap.add_argument("--engine", choices=("cpu", "gpu", "npu"), default="cpu",
                    help="which engine --cmd drives (default cpu)")
    ap.add_argument("--label", default="custom", help="name for the --cmd window")
    ap.add_argument("--out", default="powerrun", help="host directory for telemetry")
    ap.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    ap.add_argument("--seconds", type=int, default=None,
                    help="sampling window length; must clear the artifact's "
                         "minimum_validated_interval_s")
    ap.add_argument("--json", action="store_true",
                    help="print ONLY the estimate as JSON on stdout, for callers")
    a = ap.parse_args(argv[1:])
    if a.seconds is not None:
        if a.seconds < 30:
            print("--seconds must be >= 30: the model is only validated over "
                  "intervals of minimum_validated_interval_s")
            return 2
        globals()["DURATION_S"] = a.seconds
    a.presets = a.presets or []
    bad = [p for p in a.presets if p not in PRESETS]
    if bad:
        print(f"unknown preset(s) {bad}; choose from {sorted(PRESETS)}")
        return 2
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rc, serial = sh("shell", "getprop ro.serialno")
    if rc != 0 or not serial:
        print("no board")
        return 1
    for s in DEVICE_SCRIPTS:
        sh("push", str(ROOT / "scripts" / "device" / s), f"/data/local/tmp/{s}")
    # provision_device.ps1 does not push these, so a freshly provisioned board
    # has kernels but no sampler. Push them here rather than making it a step
    # the reader has to know about.
    for s in ("power_calib_sampler.sh", "power_workload_wrapper.sh"):
        sh("push", str(ROOT / "scripts" / s), f"/data/local/tmp/{s}")
    sh("shell", "chmod 755 " + " ".join(
        f"/data/local/tmp/{s}" for s in
        DEVICE_SCRIPTS + ("power_calib_sampler.sh", "power_workload_wrapper.sh")))
    rc_root, _ = sh("shell", "su 0 id -u")
    if rc_root != 0:
        print("REFUSING: the sampler needs root (su 0) to read the EAS tables.")
        return 1
    # The EAS tables live in debugfs, which SM8850 builds leave unmounted and
    # which does not survive a reboot. Without it the sampler writes an empty
    # eas.txt and the failure only surfaces later as "eas.txt produced no
    # tables" -- confusing for a student who just called Profiler.power().
    # debugfs is a virtual filesystem, so mounting it is harmless and reversible.
    rc_eas, _ = sh("shell", "su 0 ls /sys/kernel/debug/energy_model/")
    if rc_eas != 0:
        sh("shell", "su 0 mount -t debugfs none /sys/kernel/debug")
        rc_eas, _ = sh("shell", "su 0 ls /sys/kernel/debug/energy_model/")
        if rc_eas != 0:
            print("REFUSING: /sys/kernel/debug/energy_model is unreadable and "
                  "debugfs will not mount; the model needs the EAS tables.")
            return 1
    _, ps = sh("shell", "ps -A -o ARGS 2>/dev/null")
    if any(k in ps for k in ("matmul_qnn", "matmul_neon", "gpu_sol_microbench")):
        print("REFUSING: compute clients are already running; they would inflate "
              "every reading.\n  adb shell sh /data/local/tmp/pr_stop.sh")
        return 1
    art = pathlib.Path(a.artifact)
    if not art.exists():
        print(f"note: {art.name} does not exist yet, so no watts will be produced.\n"
              "  Telemetry capture and the model's input features still run and are\n"
              "  reported below; re-run the estimate once the artifact is fitted:\n"
              "    python scripts/power_capture_npu.py --out powercal   (needs the DP831A)\n")
    # Under --json this banner would land on stdout ahead of the JSON object and
    # break the parse, so it joins the rest of the narration on stderr.
    print(f"board {serial} | artifact {art.name}",
          file=sys.stderr if a.json else sys.stdout)

    if not a.presets and not a.cmd:
        print("nothing to run: give a preset (cpu/gpu/npu/cpu+gpu) or --cmd")
        return 2
    if a.cmd and a.engine == "npu":
        # An NPU window must declare a workload_cmd that the build step matches
        # byte-for-byte against the log, plus the binary hash and precision of a
        # recognised QNN client. An arbitrary command cannot satisfy that, so it
        # would fail at the build rather than here.
        print("--cmd --engine npu is not supported: NPU windows require the "
              "runtime proof of a known QNN client.\n"
              "  Use the built-in `npu` preset, or run your workload on cpu/gpu.")
        return 2

    results = {}
    # With --json the progress narration goes to stderr, so stdout carries
    # exactly one JSON object. Otherwise a caller parsing stdout also gets the
    # banner and the per-window commentary, and has to guess where JSON starts.
    narrate = (contextlib.redirect_stdout(sys.stderr) if a.json
               else contextlib.nullcontext())
    with narrate:
        for name in a.presets:
            results[name] = run_one(name, out, art)
            time.sleep(10)      # let the board settle between runs
        if a.cmd:
            results[a.label] = run_one(
                a.label, out, art,
                custom={"engine": a.engine, "workers": [a.cmd]})
        print(f"\ntelemetry and manifests in {out}")
    if a.json:
        one = results.get(a.label) if a.cmd else next(iter(results.values()), None)
        print(json.dumps(one if isinstance(one, dict) else {}, indent=2))
    return 0 if any(v for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
