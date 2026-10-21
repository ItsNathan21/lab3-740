#!/usr/bin/env python3
"""Pin CPU/GPU operating points for a power-calibration window, with read-back.

Run-sheet section B says the control state must be *set and confirmed* -- a
silent clamp (the kernel accepting a write and applying something else) is the
failure this guards against. Every pin here is read back and compared; a
mismatch is a hard error, never a warning.

    python scripts/power_pin.py show
    python scripts/power_pin.py gold 787200        # kHz, from the OPP ladder
    python scripts/power_pin.py prime 4070400
    python scripts/power_pin.py gpu 222            # MHz, or a pwrlevel index
    python scripts/power_pin.py restore            # deployment defaults back
    python scripts/power_pin.py eas                # kernel power-vs-frequency table

`restore` uses values saved on the first pin of this session
(scripts/.power_pin_state.json). Take the idle anchors AFTER a restore, per
run-sheet section B.

Frequencies must be present in scaling_available_frequencies -- the SM8850P
Prime policy advertises scaling_max_freq 4608000 while the highest enumerated
OPP is 4396800, and pinning to the former silently misbehaves.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

STATE = pathlib.Path(__file__).resolve().parent / ".power_pin_state.json"
POLICY = {"gold": "policy0", "prime": "policy6"}
KGSL = "/sys/class/kgsl/kgsl-3d0"


def sh(cmd: str, root: bool = False) -> tuple[int, str]:
    full = f"su 0 sh -c '{cmd}'" if root else cmd
    try:
        p = subprocess.run(["adb", "shell", full],
                           capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 1, str(exc)
    return p.returncode, (p.stdout + p.stderr).strip()


def read(path: str) -> str:
    # KGSL nodes are root-only on this board; CPU cpufreq is not. Reading
    # everything as root keeps one code path and never silently returns "".
    rc, out = sh(f"cat {path}", root=True)
    if rc != 0 or not out.strip() or "Permission denied" in out or "No such file" in out:
        rc, out = sh(f"cat {path}")
    if rc != 0 or "Permission denied" in out or "No such file" in out:
        return ""
    return out.strip()


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------ state
def save_defaults() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text())
    d = {}
    for name, pol in POLICY.items():
        d[name] = {
            "scaling_min_freq": read(f"/sys/devices/system/cpu/cpufreq/{pol}/scaling_min_freq"),
            "scaling_max_freq": read(f"/sys/devices/system/cpu/cpufreq/{pol}/scaling_max_freq"),
            "scaling_governor": read(f"/sys/devices/system/cpu/cpufreq/{pol}/scaling_governor"),
        }
    d["gpu"] = {
        "min_pwrlevel": read(f"{KGSL}/min_pwrlevel"),
        "max_pwrlevel": read(f"{KGSL}/max_pwrlevel"),
    }
    STATE.write_text(json.dumps(d, indent=2))
    print(f"saved deployment defaults -> {STATE.name}")
    return d


# ------------------------------------------------------------------- CPU
def cpu_opps(pol: str) -> list[int]:
    raw = read(f"/sys/devices/system/cpu/cpufreq/{pol}/scaling_available_frequencies")
    return sorted(int(v) for v in raw.split() if v.isdigit())


def pin_cpu(cluster: str, khz: int) -> None:
    pol = POLICY[cluster]
    opps = cpu_opps(pol)
    if not opps:
        die(f"cannot read the {cluster} OPP ladder")
    if khz not in opps:
        near = min(opps, key=lambda v: abs(v - khz))
        die(f"{khz} kHz is not an enumerated {cluster} OPP.\n"
            f"       nearest is {near} ({near/1000:.1f} MHz)\n"
            f"       ladder: {' '.join(str(v) for v in opps)}")
    save_defaults()
    base = f"/sys/devices/system/cpu/cpufreq/{pol}"
    # Drop min to the floor first so raising max can never be rejected for
    # crossing it, then clamp both onto the target.
    rc, out = sh(f"echo {opps[0]} > {base}/scaling_min_freq; "
                 f"echo {khz} > {base}/scaling_max_freq; "
                 f"echo {khz} > {base}/scaling_min_freq", root=True)
    if rc != 0:
        die(f"pin write failed: {out}")
    got_min = read(f"{base}/scaling_min_freq")
    got_max = read(f"{base}/scaling_max_freq")
    if got_min != str(khz) or got_max != str(khz):
        die(f"SILENT CLAMP: asked {khz}/{khz}, read back {got_min}/{got_max}")
    print(f"{cluster} pinned: min=max={khz} kHz ({khz/1000:.1f} MHz)  [read-back OK]")


# ------------------------------------------------------------------- GPU
def gpu_table() -> list[int]:
    raw = read(f"{KGSL}/freq_table_mhz")
    return [int(v) for v in raw.split() if v.isdigit()]


def pin_gpu(value: int) -> None:
    tbl = gpu_table()
    if not tbl:
        die("cannot read the GPU frequency table")
    # A value inside the index range that is not also a listed MHz is a level.
    if value in tbl:
        lvl, mhz = tbl.index(value), value
    elif 0 <= value < len(tbl):
        lvl, mhz = value, tbl[value]
    else:
        die(f"{value} is neither a listed MHz nor a valid pwrlevel 0-{len(tbl)-1}.\n"
            f"       table: " + " ".join(f"{i}:{m}" for i, m in enumerate(tbl)))
    save_defaults()
    # min_pwrlevel is the numerically larger (slower) bound; widen it before
    # narrowing so neither write is rejected for crossing the other.
    rc, out = sh(f"echo {len(tbl)-1} > {KGSL}/min_pwrlevel; "
                 f"echo {lvl} > {KGSL}/max_pwrlevel; "
                 f"echo {lvl} > {KGSL}/min_pwrlevel", root=True)
    if rc != 0:
        die(f"pin write failed: {out}")
    got_min = read(f"{KGSL}/min_pwrlevel")
    got_max = read(f"{KGSL}/max_pwrlevel")
    if got_min != str(lvl) or got_max != str(lvl):
        die(f"SILENT CLAMP: asked level {lvl}/{lvl}, read back {got_min}/{got_max}")
    clk = read(f"{KGSL}/gpuclk")
    note = ""
    if clk.isdigit():
        actual = int(clk) // 1_000_000
        note = f", gpuclk={actual} MHz"
        if actual != mhz:
            note += "  <- idle; confirm again under load"
    print(f"GPU pinned: pwrlevel {lvl} = {mhz} MHz  [read-back OK{note}]")


# --------------------------------------------------------------- restore
def restore() -> None:
    if not STATE.is_file():
        die("no saved defaults; nothing to restore")
    d = json.loads(STATE.read_text())
    for name, pol in POLICY.items():
        v = d.get(name, {})
        if not v.get("scaling_min_freq"):
            continue
        base = f"/sys/devices/system/cpu/cpufreq/{pol}"
        sh(f"echo {cpu_opps(pol)[0]} > {base}/scaling_min_freq; "
           f"echo {v['scaling_max_freq']} > {base}/scaling_max_freq; "
           f"echo {v['scaling_min_freq']} > {base}/scaling_min_freq", root=True)
        print(f"{name} restored: min={read(base + '/scaling_min_freq')} "
              f"max={read(base + '/scaling_max_freq')}")
    g = d.get("gpu", {})
    if g.get("min_pwrlevel"):
        sh(f"echo {len(gpu_table())-1} > {KGSL}/min_pwrlevel; "
           f"echo {g['max_pwrlevel']} > {KGSL}/max_pwrlevel; "
           f"echo {g['min_pwrlevel']} > {KGSL}/min_pwrlevel", root=True)
        print(f"GPU restored: min_pwrlevel={read(KGSL + '/min_pwrlevel')} "
              f"max_pwrlevel={read(KGSL + '/max_pwrlevel')}")
    STATE.unlink()
    print("\nDeployment power management is back. Take idle anchors now, not before.")


def show() -> None:
    for name, pol in POLICY.items():
        base = f"/sys/devices/system/cpu/cpufreq/{pol}"
        cur = read(f"{base}/scaling_cur_freq")
        mn, mx = read(f"{base}/scaling_min_freq"), read(f"{base}/scaling_max_freq")
        pinned = "PINNED" if mn == mx and mn else "free"
        print(f"{name:<6} cur={cur or '?':>8}  min={mn or '?':>8}  max={mx or '?':>8}  [{pinned}]")
    mn, mx = read(f"{KGSL}/min_pwrlevel"), read(f"{KGSL}/max_pwrlevel")
    clk = read(f"{KGSL}/gpuclk")
    mhz = f"{int(clk)//1_000_000} MHz" if clk.isdigit() else "?"
    pinned = "PINNED" if mn == mx and mn else "free"
    print(f"gpu    clk={mhz:>8}  min_lvl={mn or '?':>3}  max_lvl={mx or '?':>3}  [{pinned}]")
    tbl = gpu_table()
    if tbl:
        print("       levels: " + "  ".join(f"{i}:{m}" for i, m in enumerate(tbl)))


def eas() -> None:
    """Dump the kernel's own energy model: power at each OPP, per cluster.

    This is the table the scheduler uses to decide where to put work, and it is
    the only power-versus-frequency curve on this board that does not come from
    our own fitted model. debugfs is not mounted by default on this build, so
    mount it first; the mount is harmless and persists until reboot.

    Columns are `cluster freq_mhz power_w`, ready to paste into a fit.
    """
    rc, out = sh(
        "mount -t debugfs none /sys/kernel/debug 2>/dev/null; "
        "for d in /sys/kernel/debug/energy_model/*/ps:*/; do "
        "  c=$(basename $(dirname $d)); "
        "  f=$(cat $d/frequency 2>/dev/null); "
        "  p=$(cat $d/power 2>/dev/null); "
        "  [ -n \"$f\" ] && echo $c $f $p; "
        "done", root=True)
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            rows.append((parts[0], int(parts[1]), int(parts[2])))
    if not rows:
        die("no energy-model rows read. Needs root (su 0) and debugfs; "
            f"adb said: {out[:200]!r}")
    print(f"{'cluster':8} {'freq_mhz':>9} {'power_w':>10}")
    for cluster, khz, uw in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f"{cluster:8} {khz / 1000.0:9.1f} {uw / 1e6:10.6f}")
    per = {}
    for cluster, _, _ in rows:
        per[cluster] = per.get(cluster, 0) + 1
    print()
    print(", ".join(f"{c}: {k} operating points" for c, k in sorted(per.items())))
    print("cpu0 is the Gold cluster, cpu6 the Prime cluster. Power is per core.")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "show":
        show()
    elif cmd == "restore":
        restore()
    elif cmd in ("gold", "prime"):
        if len(sys.argv) != 3 or not sys.argv[2].isdigit():
            die(f"usage: power_pin.py {cmd} <kHz>")
        pin_cpu(cmd, int(sys.argv[2]))
    elif cmd == "eas":
        eas()
    elif cmd == "gpu":
        if len(sys.argv) != 3 or not sys.argv[2].isdigit():
            die("usage: power_pin.py gpu <MHz|pwrlevel>")
        pin_gpu(int(sys.argv[2]))
    else:
        die(f"unknown command {cmd!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
