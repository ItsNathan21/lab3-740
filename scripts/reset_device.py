#!/usr/bin/env python3
"""Put the board back to how it arrived, so provisioning has to be done again.

Why this exists: the only way to know the setup guide works is to follow it on
an unprovisioned board, and after the first time nobody has one. This makes
one.

WHY NOT JUST REBOOT. It is the obvious guess and it does nothing. On this HDK
/data is f2fs on a real block device:

    /dev/block/dm-57 on /data type f2fs (rw,lazytime,...)

so /data/local/tmp survives a reboot, a bootloader cycle and a battery pull.
The kernels are still there afterwards and provisioning still reports
everything already in place. Files have to be deleted explicitly, which is what
this does. `--reboot` is offered anyway, because clearing the DVFS and thermal
state between experiments is a real thing to want -- it just is not what
removes the kernels.

Nothing here touches the bootloader, the system partition, or anything outside
/data/local/tmp. It cannot brick the board, and everything it removes is
restored by re-running provision_device.py.

    python scripts/reset_device.py                 # show what would go
    python scripts/reset_device.py --yes           # do it
    python scripts/reset_device.py --yes --models  # also the staged networks
    python scripts/reset_device.py --yes --reboot  # and restart the board
"""
from __future__ import annotations

import argparse
import subprocess
import sys

DEVICE_BASE = "/data/local/tmp/profiler-kernels"
MODEL_DIR = "/data/local/tmp"
# Staged separately from provisioning and measured in gigabytes -- the Llama
# DLC alone is 6 GB. Re-pushing them is a long job over USB, so they are opt-in
# rather than part of a default reset.
MODEL_DIRS = ("mnv2", "nlp", "qwen", "llama", "sd15")
# Left behind by Profiler.power() if a run was interrupted before it reaped it.
STRAY_FILES = ("/data/local/tmp/profiler_power_loop.sh",)


def sh(cmd: str, serial: str | None = None, timeout: int = 120) -> tuple[int, str]:
    argv = ["adb"] + (["-s", serial] if serial else []) + ["shell", cmd]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "adb not found on PATH"


def size_of(path: str, serial: str | None) -> str | None:
    """Human-readable size, or None when the path is not there."""
    rc, out = sh(f"du -sh {path} 2>/dev/null", serial)
    if rc != 0 or not out.strip():
        return None
    return out.split()[0]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Delete the provisioned kernels so setup must be redone.")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete; without this, only report")
    ap.add_argument("--models", action="store_true",
                    help="also remove the staged networks (gigabytes, slow to "
                         "re-push)")
    ap.add_argument("--reboot", action="store_true",
                    help="reboot afterwards. Note: the reboot is NOT what "
                         "deletes anything; it clears DVFS and thermal state.")
    ap.add_argument("--serial", default=None,
                    help="target this device when several are attached")
    a = ap.parse_args(argv[1:])

    rc, out = sh("echo ok", a.serial, timeout=20)
    if rc != 0 or "ok" not in out:
        print(f"error: no board reachable ({out.strip()[:120]})", file=sys.stderr)
        print("  check `adb devices`; pass --serial if more than one is attached",
              file=sys.stderr)
        return 1

    targets: list[tuple[str, str]] = []
    sz = size_of(DEVICE_BASE, a.serial)
    if sz:
        targets.append((DEVICE_BASE, sz))
    else:
        print(f"  {DEVICE_BASE} is already gone")

    if a.models:
        for d in MODEL_DIRS:
            path = f"{MODEL_DIR}/{d}"
            msz = size_of(path, a.serial)
            if msz:
                targets.append((path, msz))

    strays = []
    for f in STRAY_FILES:
        rc, _ = sh(f"test -e {f}", a.serial)
        if rc == 0:
            strays.append(f)

    if not targets and not strays:
        print("Nothing to remove. The board is already unprovisioned.")
        return 0

    print("Would remove:" if not a.yes else "Removing:")
    for path, s in targets:
        print(f"  {path}   ({s})")
    for f in strays:
        print(f"  {f}   (leftover power-measurement loop)")

    if not a.yes:
        print()
        print("Nothing was deleted. Re-run with --yes to do it.")
        if not a.models:
            print("Staged networks are left alone unless you pass --models.")
        return 0

    failed = []
    for path, _ in targets:
        # Created by root during provisioning, so removal needs root too.
        rc, out = sh(f"su 0 rm -rf {path}", a.serial, timeout=300)
        if rc != 0:
            failed.append((path, out.strip()[:120]))
    for f in strays:
        sh(f"su 0 rm -f {f}", a.serial)

    # Confirm rather than trust the exit code: `rm -rf` on a path it cannot
    # touch can still exit 0 in some shells, and a reset that silently did
    # nothing is worse than one that failed loudly.
    still_there = [p for p, _ in targets if size_of(p, a.serial)]
    if failed or still_there:
        print()
        for path, err in failed:
            print(f"error: could not remove {path}: {err}", file=sys.stderr)
        for path in still_there:
            print(f"error: {path} is still present after the delete",
                  file=sys.stderr)
        print("  is `su 0` available on this build?", file=sys.stderr)
        return 1

    print()
    print("Board reset.")

    if a.reboot:
        print("Rebooting (this clears DVFS and thermal state; it is not what")
        print("deleted the kernels above). Wait ~30 s, then `adb devices`.")
        sh("reboot", a.serial, timeout=30)

    print()
    print("To put it back:")
    print("  python scripts/provision_device.py --qairt-root <path> \\")
    print("      --dsp-skel kernels/matmul/dsp/prebuilt/libdsp_matmul_skel.so")
    print()
    print("`python scripts/verify_board.py` should FAIL until you do --")
    print("that failure is the proof this worked.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit("interrupted")
