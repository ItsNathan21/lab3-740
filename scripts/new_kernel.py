"""new_kernel.py - scaffold, build and push a kernel of your own.

    python scripts/new_kernel.py mykernel              # create kernels/custom/mykernel/
    python scripts/new_kernel.py mykernel --build      # + cross-compile it
    python scripts/new_kernel.py mykernel --push       # + build and push to the board

Lab 3 Task 5 programs live with the rest of your lab code, in
kernels/lab3/<name>/<name>.c. For those, --push builds that file and pushes it:

    python scripts/new_kernel.py bp_stress --push      # kernels/lab3/bp_stress/bp_stress.c
    python scripts/new_kernel.py rob_stress --push     # kernels/lab3/rob_stress/rob_stress.c

Then run it through the same harness as everything else:

    from profiler import Profiler, Backend, custom
    Profiler().run(custom(Backend.CPU, "mykernel", size=1024))

The generated kernel already satisfies the output contract, so `Result` comes
back populated - timing, GFLOP/s, GB/s and a fingerprint - with no registry
edit anywhere.

Needs the Android NDK: `python install.py --role dev --fix`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEVICE_CUSTOM = "/data/local/tmp/profiler-kernels/custom"

TEMPLATE = r'''/* {name}.c - a kernel of your own, on the CPU.
 *
 * THE OUTPUT CONTRACT. profiler.py scrapes every "key: value" line and every
 * "key=value" token, so all you must do is print. Required:
 *
 *   status: OK          optional. Only OK/SUCCESS/PASS/PASSED/DONE pass;
 *                       any other word fails the run. Omit it and the
 *                       exit code decides, which is what most kernels do.
 *   mean_us: <float>    REQUIRED per-iteration time. Alternatives, in order:
 *                       harness prefers them: execute_us_mean, dsp_us,
 *                       mean_us, duration_ns, wall_us.
 *
 * Optional but worth it:
 *   flop:  <n>   per iteration -> Result.gflops is computed for you
 *   bytes: <n>   per iteration -> Result.gbps
 *   fingerprint: <float>   a checksum of your output. Nothing verifies it for
 *                          you; comparing it across variants is how you catch
 *                          a kernel that got faster by computing the wrong
 *                          thing.
 *
 * Report flop and bytes PER ITERATION, matching mean_us. Reporting totals
 * while timing one pass is a real bug we shipped once: it inflated a result
 * by 25x and looked plausible.
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}}

/* Same generator the shipped kernels use, so a given --seed reproduces
   bit-identically and fingerprints are comparable across backends. */
static unsigned long long rng;
static float next_f(void) {{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) / 16777216.0) - 0.5f;
}}

int main(int argc, char **argv) {{
    int n = 1048576, iters = 0;
    unsigned long long seed = 42;

    for (int i = 1; i < argc - 1; i++) {{
        if      (!strcmp(argv[i], "--size"))       n     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iterations")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed"))       seed  = strtoull(argv[++i], NULL, 10);
    }}
    if (n < 1) {{ printf("status: FAILED\nerror: --size must be >= 1\n"); return 1; }}
    /* Aim for ~150 ms total. A run that finishes before the governor ramps
       measures DVFS state, not your kernel. */
    if (iters <= 0) {{ iters = (int)(2e9 / n); if (iters < 1) iters = 1; }}

    rng = seed ? seed : 1;
    float *x = malloc((size_t)n * sizeof(float));
    float *y = malloc((size_t)n * sizeof(float));
    if (!x || !y) {{ printf("status: FAILED\nerror: allocation\n"); return 1; }}
    for (int i = 0; i < n; i++) {{ x[i] = next_f(); y[i] = next_f(); }}

    /* ---------------------------------------------------------------------
     * YOUR KERNEL GOES HERE. This placeholder is y = 2*x + y (SAXPY), which
     * is memory-bound at 0.17 FLOP/byte -- work out what that predicts
     * before you measure it.
     * ------------------------------------------------------------------- */
    const float32x4_t a = vdupq_n_f32(2.0f);
    double t0 = now_us();
    for (int it = 0; it < iters; it++) {{
        int i = 0;
        for (; i + 4 <= n; i += 4)
            vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), a, vld1q_f32(x + i)));
        for (; i < n; i++) y[i] += 2.0f * x[i];
    }}
    double us = now_us() - t0;

    double fp = 0.0;
    for (int i = 0; i < n; i++) fp += y[i];

    printf("status: OK\n");
    printf("size: %d\niterations: %d\nseed: %llu\n", n, iters, seed);
    printf("mean_us: %.1f\n", us / iters);            /* per iteration */
    printf("wall_us: %.1f\n", us);
    printf("flop: %.0f\n",  2.0 * n);                 /* per iteration */
    printf("bytes: %.0f\n", 12.0 * n);                /* 2 reads + 1 write */
    printf("fingerprint: %.6f\n", fp);

    free(x); free(y);
    return 0;
}}
'''

BUILD = '''#!/usr/bin/env bash
# Cross-compile {name} for arm64-android.
#   python install.py --role dev --fix     # once, to get the NDK
#   . .toolchain/activate.sh               # or add tools/ to PATH yourself
set -euo pipefail
DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
if ! command -v android-clang >/dev/null 2>&1; then
    echo "ERROR: android-clang not on PATH." >&2
    echo "  python install.py --role dev --fix && . .toolchain/activate.sh" >&2
    exit 1
fi
android-clang -std=c11 -O3 -o "$DIR/{name}" "$DIR/{name}.c" -lm
file "$DIR/{name}"
'''

README = '''# {name}

Your own kernel. Build, push, run:

```bash
python scripts/new_kernel.py {name} --push
python -c "from profiler import *; print(Profiler().run(custom(Backend.CPU, '{name}', size=1048576)).summary())"
```

`{name}.c` starts as a NEON SAXPY (0.17 FLOP/byte, memory-bound). Replace the
marked block with your own work.

Before you trust any number from it:

- report `flop` and `bytes` **per iteration**, matching `mean_us`
- check the `fingerprint` against a reference - a kernel that got faster by
  computing the wrong thing still reports a fast time
- pin the clock (`scripts/power_pin.py`) and use `--repeat` before quoting
  anything
'''


def _env() -> dict:
    """PATH with tools/ (android-clang) and the bundled NDK, without asking."""
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(ROOT / "tools"), env.get("PATH", "")])
    state = ROOT / ".toolchain" / "paths.json"
    if state.exists() and "NDK_ROOT" not in env:
        try:
            ndk = json.loads(state.read_text(encoding="utf-8")).get("ndk")
            if ndk:
                env["NDK_ROOT"] = ndk
        except Exception:
            pass
    return env


def _push(binary: pathlib.Path, name: str) -> bool:
    adb = shutil.which("adb") or str(ROOT / ".toolchain/platform-tools/adb")
    subprocess.run([adb, "shell", f"mkdir -p {DEVICE_CUSTOM}"], capture_output=True)
    p = subprocess.run([adb, "push", str(binary), f"{DEVICE_CUSTOM}/{name}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print("  push failed:", (p.stdout + p.stderr).strip()[:200])
        return False
    subprocess.run([adb, "shell", f"chmod 755 {DEVICE_CUSTOM}/{name}"], capture_output=True)
    print(f"  pushed to {DEVICE_CUSTOM}/{name}")
    return True


MEASURE = {
    "bp_stress":  "--counters raw-br-pred,raw-br-mis-pred",
    "rob_stress": "--counters stalls",
}


def build_lab3(name: str, d: pathlib.Path, push: bool) -> int:
    src, out = d / f"{name}.c", d / name
    env = _env()
    clang = shutil.which("android-clang", path=env["PATH"])
    if not clang:
        print("  tools/android-clang is missing. Run: python install.py --role dev --fix")
        return 1
    bash = shutil.which("bash")
    cmd = [clang, "-std=c11", "-O3", "-Wall", "-o", str(out), str(src), "-lm"]
    if bash:   # android-clang is a bash script; run it through bash on every OS
        cmd = [bash] + cmd
    print(f"  building {src.relative_to(ROOT)}")
    if subprocess.run(cmd, env=env).returncode != 0:
        return 1
    if not push:
        print(f"  built {out.relative_to(ROOT)}\n\n  next: python scripts/new_kernel.py {name} --push")
        return 0
    if not _push(out, name):
        return 1
    extra = MEASURE.get(name, "")
    print(f"\n  measure it:\n    python profiler.py custom cpu --binary {name} --core prime {extra}".rstrip())
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="kernel name (also the binary name)")
    ap.add_argument("--build", action="store_true", help="cross-compile it")
    ap.add_argument("--push", action="store_true", help="build, then push to the board")
    ap.add_argument("--force", action="store_true", help="overwrite an existing scaffold")
    a = ap.parse_args(argv[1:])

    if not a.name.replace("_", "").isalnum():
        raise SystemExit(f"name must be alphanumeric/underscore, got {a.name!r}")

    # A Lab 3 program (kernels/lab3/<name>/<name>.c, e.g. bp_stress) is built
    # straight from that file: no scaffold, no build.sh. It lands in the same
    # custom/ folder on the board, so `profiler.py custom --binary <name>` runs it.
    lab = ROOT / "kernels" / "lab3" / a.name
    if (lab / f"{a.name}.c").is_file():
        return build_lab3(a.name, lab, push=a.push)

    d = ROOT / "kernels" / "custom" / a.name
    src = d / f"{a.name}.c"
    if src.exists() and not a.force:
        print(f"  {src.relative_to(ROOT)} exists (use --force to overwrite)")
    else:
        d.mkdir(parents=True, exist_ok=True)
        src.write_text(TEMPLATE.format(name=a.name), encoding="utf-8", newline="\n")
        bs = d / "build.sh"
        bs.write_text(BUILD.format(name=a.name), encoding="utf-8", newline="\n")
        bs.chmod(bs.stat().st_mode | 0o111)
        (d / "README.md").write_text(README.format(name=a.name), encoding="utf-8", newline="\n")
        print(f"  created {d.relative_to(ROOT)}/  ({a.name}.c, build.sh, README.md)")

    if not (a.build or a.push):
        print(f"\n  next: python scripts/new_kernel.py {a.name} --push")
        return 0

    # Make tools/ and any bundled NDK visible without asking the user to.
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(ROOT / "tools"), env.get("PATH", "")])
    state = ROOT / ".toolchain" / "paths.json"
    if state.exists() and "NDK_ROOT" not in env:
        try:
            ndk = json.loads(state.read_text(encoding="utf-8")).get("ndk")
            if ndk:
                env["NDK_ROOT"] = ndk
        except Exception:
            pass

    bash = shutil.which("bash")
    if not bash:
        print("  bash not found; run kernels/custom/%s/build.sh yourself" % a.name)
        return 1
    r = subprocess.run([bash, str(d / "build.sh")], env=env)
    if r.returncode != 0:
        return r.returncode
    if not a.push:
        return 0

    adb = shutil.which("adb") or str(ROOT / ".toolchain/platform-tools/adb")
    subprocess.run([adb, "shell", f"mkdir -p {DEVICE_CUSTOM}"], capture_output=True)
    p = subprocess.run([adb, "push", str(d / a.name), f"{DEVICE_CUSTOM}/{a.name}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print("  push failed:", (p.stdout + p.stderr).strip()[:200])
        return 1
    subprocess.run([adb, "shell", f"chmod 755 {DEVICE_CUSTOM}/{a.name}"], capture_output=True)
    print(f"  pushed to {DEVICE_CUSTOM}/{a.name}")
    print(f"\n  run it:\n"
          f'    python -c "from profiler import *; '
          f"print(Profiler().run(custom(Backend.CPU, '{a.name}', size=1048576)).summary())\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
