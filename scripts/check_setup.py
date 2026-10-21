#!/usr/bin/env python3
"""check_setup.py - is this machine ready, before the board arrives?

    python scripts/check_setup.py

Everything here is checked WITHOUT a board attached, so you can do the whole
setup early and know it worked. It does not push anything, does not need a
device, and changes nothing.

What it cannot tell you: whether your USB cable carries data, and whether the
board itself is healthy. Those need the board. Everything else - Python, adb,
Docker, and above all whether your QAIRT install is the right one - it can.

Exit code is 0 when you are ready, 1 when something needs fixing.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"
_TAG = {PASS: "  [PASS] ", FAIL: "  [FAIL] ", WARN: "  [warn] ", INFO: "  [info] "}
_fails = 0
_warns = 0


def say(status: str, name: str, detail: str = "", fix: str = "") -> None:
    global _fails, _warns
    if status == FAIL:
        _fails += 1
    if status == WARN:
        _warns += 1
    print(f"{_TAG[status]}{name}" + (f" - {detail}" if detail else ""))
    for line in (fix or "").strip().splitlines():
        if line:
            print(f"           {line}")


def section(title: str) -> None:
    print(f"\n{title}")


def qnn_api(header: pathlib.Path, prefix: str) -> tuple[int, int, int] | None:
    try:
        raw = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    out = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"^\s*#define\s+{prefix}_{part}\s+(\d+)\s*$", raw, re.M)
        if not m:
            return None
        out.append(int(m.group(1)))
    return tuple(out)  # type: ignore[return-value]


def find_qairt() -> pathlib.Path | None:
    """Wherever QPM, or a hand-extracted archive, might have put it."""
    saved = None
    try:
        saved = json.loads((ROOT / ".toolchain/paths.json").read_text()).get("qairt")
    except Exception:
        pass
    cands: list[pathlib.Path] = [pathlib.Path(saved)] if saved else []
    home = pathlib.Path.home()
    for pat in ("Downloads/*/qairt/*", "Qualcomm/*/qairt/*", "qairt/*"):
        cands += sorted(home.glob(pat))
    for base in ("/opt/qcom/aistack/qairt", "/opt/qcom/aistack", "/opt/qairt",
                 "/opt/qualcomm/qairt", "C:/Qualcomm/AIStack/qairt",
                 "C:/Qualcomm/qairt"):
        cands += sorted(pathlib.Path(base).glob("*"))
    for c in cands:
        if (c / "lib/aarch64-android/libQnnHtp.so").exists():
            return c
    return None


def main() -> int:
    print("Lab 3 setup check - no board required\n"
          f"host: {platform.system()} {platform.machine()}")

    # ---------------------------------------------------------------- python
    section("Python")
    v = sys.version_info
    if v[:2] >= (3, 10):
        say(PASS, "version", f"{v.major}.{v.minor}.{v.micro}")
    else:
        say(FAIL, "version", f"{v.major}.{v.minor}, need 3.10+",
            "Install from python.org. On Windows tick 'Add python.exe to PATH'.")

    # ------------------------------------------------------------------- adb
    section("adb")
    local = ROOT / ".toolchain/platform-tools" / ("adb.exe" if IS_WIN else "adb")
    adb = str(local) if local.exists() else shutil.which("adb")
    if not adb:
        say(FAIL, "adb", "not found",
            "python install.py --fix     # downloads it here, no admin needed")
    else:
        out = subprocess.run([adb, "version"], capture_output=True, text=True).stdout
        m = re.search(r"Version (\d+)\.(\d+)\.(\d+)", out)
        ver = ".".join(m.groups()) if m else "unknown"
        # A very old adb and a modern one disagree about the protocol, and the
        # loser gets its server killed. 30 is roughly Android 11 era.
        if m and int(m.group(1)) < 30:
            say(WARN, "version", f"{ver} at {adb}",
                "Old enough to fight a newer adb server. Prefer:\n"
                "  python install.py --fix")
        else:
            say(PASS, "version", f"{ver} at {adb}")

    # ---------------------------------------------------------------- QAIRT
    section("QAIRT (the NPU runtime - Tasks 12 and 13)")
    qairt = find_qairt()
    if not qairt:
        say(FAIL, "installed", "not found",
            "Install it with the Qualcomm Package Manager, in the lab container:\n"
            "  qpm-cli --license-activate qualcomm_ai_runtime_sdk\n"
            "  qpm-cli --extract qualcomm_ai_runtime_sdk -v 2.40.0.251030\n"
            "It goes quiet for up to 10 minutes, then asks [y/n]: answer y.\n"
            "If a name is rejected: qpm-cli --product-list gives names, and\n"
            "qpm-cli --info <product> gives that product's versions.\n"
            "Then re-run this check. If it is installed somewhere unusual:\n"
            "  python install.py --qairt <path>/qairt/<version>")
    else:
        say(PASS, "installed", f"{qairt.name} at {qairt}")

        client = qnn_api(ROOT / "kernels/matmul/npu/qnn_fp16/include/QNN/QnnCommon.h",
                         "QNN_API_VERSION")
        sdk = qnn_api(qairt / "include/QNN/QnnCommon.h", "QNN_API_VERSION")
        if sdk is None:
            say(FAIL, "core API", "cannot read include/QNN/QnnCommon.h",
                "That path should exist inside a QAIRT root. Wrong directory?")
        elif client is None:
            say(WARN, "core API", f"{sdk[0]}.{sdk[1]}.{sdk[2]} (nothing to compare with)")
        elif sdk[0] != client[0] or sdk[1] < client[1]:
            say(FAIL, "core API",
                f"SDK {sdk[0]}.{sdk[1]} cannot drive our clients (need {client[0]}.{client[1]}+)",
                "Install the lab's pinned QAIRT:\n"
                "  qpm-cli --extract qualcomm_ai_runtime_sdk -v 2.40.0.251030")
        else:
            say(PASS, "core API", f"{sdk[0]}.{sdk[1]}.{sdk[2]} >= client {client[0]}.{client[1]}")

        # Prove this is a real SDK before trusting anything else about it.
        # The checks around this one can all be satisfied by copying the
        # runtime off a provisioned board and dropping our own client
        # headers into include/QNN -- at which point the version check
        # above compares a header against a copy of itself and passes.
        # bin/x86_64-linux-clang holds the SDK's HOST tools: x86-64 Linux
        # binaries that never ship to an aarch64 board and are not among
        # the 9 files provisioning pushes.
        host_tools = qairt / "bin" / "x86_64-linux-clang"
        n_tools = 0
        try:
            n_tools = sum(1 for _ in host_tools.iterdir())
        except OSError:
            pass
        if n_tools == 0:
            say(FAIL, "host tools", "bin/x86_64-linux-clang missing or empty",
                "This holds the SDK's own x86 tools, so its absence means this is\n"
                "not a QAIRT installation -- most likely a tree copied off a board,\n"
                "or an install that did not finish. Re-install it with QPM.")
        elif not (qairt / "sdk.yaml").is_file():
            say(WARN, "sdk.yaml", "missing",
                "Every QAIRT install carries one. If you assembled this directory\n"
                "by hand, re-install it with QPM instead.")
        else:
            say(PASS, "host tools", f"bin/x86_64-linux-clang, {n_tools} tools")

        # the exact files provisioning will copy, so a missing component is
        # found now rather than in front of a board
        need = [
            "lib/aarch64-android/libQnnHtp.so",
            "lib/aarch64-android/libQnnHtpPrepare.so",
            "lib/aarch64-android/libQnnHtpV81Stub.so",
            "lib/aarch64-android/libQnnHtpV81CalculatorStub.so",
            "lib/aarch64-android/libQnnSystem.so",
            "lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so",
            "lib/hexagon-v81/unsigned/libCalculator_skel.so",
            "bin/aarch64-android/qnn-platform-validator",
            "bin/aarch64-android/qnn-net-run",
        ]
        missing = [p for p in need if not (qairt / p).exists()]
        if missing:
            say(FAIL, "v81 components", f"{len(missing)} of {len(need)} files missing",
                "This board is Hexagon v81. Missing:\n  " + "\n  ".join(missing[:4])
                + ("\n  ..." if len(missing) > 4 else "")
                + "\nRe-install QAIRT and include the v81 / Hexagon components.")
        else:
            total = sum((qairt / p).stat().st_size for p in need)
            say(PASS, "v81 components", f"all {len(need)} present, {total/1e6:.0f} MB")

    # ------------------------------------------------------------- Android NDK
    # Fetched by `install.py --role dev --fix` in step 02. Checked here because
    # it is 840 MB and nothing else tells you whether it arrived until you try
    # to compile Task 5, which is the worst moment to find out.
    section("Android NDK (Tasks 5 and 12 - code for the board's CPU)")
    ndk = None
    ndk_cands = [ROOT / ".toolchain", pathlib.Path.home() / "Downloads"]
    try:
        saved = json.loads((ROOT / ".toolchain/paths.json").read_text()).get("ndk")
        if saved:
            ndk_cands.insert(0, pathlib.Path(saved).parent)
    except Exception:
        pass
    for base in ndk_cands:
        try:
            for c in sorted(base.glob("android-ndk-*")):
                if (c / "toolchains/llvm/prebuilt").is_dir():
                    ndk = c
                    break
        except OSError:
            pass
        if ndk:
            break
    if ndk:
        say(PASS, "installed", ndk.name)
    else:
        say(WARN, "installed", "not found",
            "Tasks 5 and 12 need it, and both are graded. Free, no account:\n"
            "  python install.py --role dev --fix        # about 840 MB")

    # ----------------------------------------------------------- Hexagon SDK
    section("Hexagon SDK (Tasks 10 and 11 - code for the DSP)")
    hexagon = None
    # The registered path wins. `install.py --hexagon` recorded where the SDK
    # actually is, and guessing alongside a registry means two sources of truth:
    # on a machine with both a Windows SDK in C:/Qualcomm and the Linux one in
    # ~/Qualcomm, the guess order found the Windows install and reported a
    # correct setup as broken.
    try:
        reg = json.loads((ROOT / ".toolchain/paths.json").read_text()).get("hexagon")
        if reg and (pathlib.Path(reg) / "incs").is_dir():
            hexagon = pathlib.Path(reg)
    except Exception:
        pass
    # Collect every candidate, then prefer one that can actually build: a
    # machine with an old native Windows SDK in C:/Qualcomm AND the Linux one
    # in ~/Qualcomm would otherwise report the unusable one and fail a correct
    # setup. First-found is only the fallback.
    # 6.6.0.0 is a FLOOR, not a pin. HAP_compute_res.h arrived in 6.6 and the
    # VTCM kernels need it (Discovery 44); on an older SDK the build fails
    # pointing at our source rather than at the SDK. Newer is fine and is what
    # QPM now serves, so take the newest installed.
    HEX_MIN = (6, 6, 0, 0)

    def vkey(name: str) -> tuple:
        # A string sort puts 6.9.0.0 above 6.10.0.0, which would have this gate
        # check one SDK while the build used another.
        parts = []
        for piece in name.split("."):
            parts.append(int(piece) if piece.isdigit() else -1)
        return tuple(parts + [0] * (4 - len(parts)))[:4]

    found = []
    for base in ([] if hexagon else
                 ["C:/Qualcomm/Hexagon_SDK", str(pathlib.Path.home() / "Qualcomm/Hexagon_SDK"),
                  "/opt/qcom/Hexagon_SDK", "/opt/Qualcomm/Hexagon_SDK",
                  "/local/mnt/workspace/Qualcomm/Hexagon_SDK"]):
        try:
            for c in pathlib.Path(base).glob("*"):
                if (c / "incs").is_dir() and vkey(c.name) >= HEX_MIN:
                    found.append(c)
        except OSError:
            pass
    found.sort(key=lambda c: vkey(c.name), reverse=True)
    if not hexagon and found:
        usable = [c for c in found
                  if any(c.glob("tools/HEXAGON_Tools/*/Tools/bin/hexagon-clang"))]
        hexagon = usable[0] if usable else found[0]
    if not hexagon:
        say(FAIL, "installed", "not found",
            "Tasks 10 and 11 are graded and need this to compile. It is the\n"
            "slowest thing in the setup, so get it now rather than then:\n"
            "  qpm-cli --license-activate hexagonsdk6.x\n"
            "  qpm-cli --install hexagonsdk6.x\n"
            "Any 6.6.0.0 or newer works -- do not pin a version, and do not pass\n"
            "--path: the default install location is one of the places searched.\n"
            "(Measuring with the PREBUILT DSP kernels works without it.)")
    elif not (hexagon / "rtos/qurt/computev81").is_dir():
        say(FAIL, "computev81", f"missing in {hexagon.name}",
            "This board is v81. An SDK without it compiles fine and produces a\n"
            "skeleton the board cannot load. Re-install including computev81.")
    else:
        # Directories prove nothing -- a tree copied off the board has the same
        # shape. The compiler is what makes it an SDK, so look for it, and check
        # it is the LINUX build: a Windows install has every directory, the v81
        # runtime, and hexagon-clang.exe, which no container can execute.
        cc = sorted(hexagon.glob("tools/HEXAGON_Tools/*/Tools/bin/hexagon-clang"))
        exe = sorted(hexagon.glob("tools/HEXAGON_Tools/*/Tools/bin/hexagon-clang.exe"))
        if cc:
            # The discriminator that matters is the file NAME: a Linux SDK has
            # `hexagon-clang`, a Windows one has `hexagon-clang.exe`. The ELF
            # header is a hint, not a gate -- in a Linux SDK hexagon-clang is a
            # symlink to clang-19, which a Windows host cannot follow, so a strict
            # header check failed on a good SDK that works fine in the container
            # where the compiler is actually run.
            head = b""
            try:
                head = cc[-1].open("rb").read(4)
            except OSError:
                pass
            note = "Linux hexagon-clang"
            if head and head != b"\x7fELF":
                note += " (symlink; header unread from this host, which is normal)"
            say(PASS, "installed", f"{hexagon.name}, computev81, {note}")
        elif exe:
            say(FAIL, "hexagon-clang", "this is the WINDOWS SDK (hexagon-clang.exe)",
                "The lab compiles DSP code in a Linux container, which cannot run\n"
                ".exe files. Install the LINUX Hexagon SDK -- a separate download,\n"
                "not a copy of a Windows install. Run QPM inside the container.")
        else:
            say(FAIL, "hexagon-clang", f"not found under {hexagon.name}/tools/HEXAGON_Tools",
                "The directories are there but the compiler is not, so this is not\n"
                "a working SDK. Re-install it with QPM.")

    # ---------------------------------------------------------------- Docker
    # If we are already IN the lab container, there is nothing to check and
    # every word of the advice below is wrong -- it would tell someone standing
    # inside the container to go and install it. Note it and move on.
    in_container = os.path.exists("/.dockerenv")
    if not in_container:
        try:
            with open("/proc/1/cgroup", "rb") as fh:
                in_container = b"docker" in fh.read() or b"containerd" in fh.read()
        except OSError:
            pass

    section("Docker")
    if in_container:
        say(PASS, "environment", "running inside the lab container",
            "Nothing to install. Note that Docker itself is deliberately not\n"
            "available in here -- start and stop the container from your own\n"
            "machine with scripts/lab3.py.")
        docker = None
    else:
        docker = shutil.which("docker")
    if not in_container and not docker:
        status = FAIL if IS_MAC else INFO
        say(status, "installed", "not found",
            "Docker Desktop: https://www.docker.com/products/docker-desktop/\n"
            + ("macOS has no Hexagon toolchain, so DSP code is compiled in a\n"
               "container. On Apple Silicon also enable Settings -> General ->\n"
               "'Use Rosetta for x86/amd64 emulation'."
               if IS_MAC else
               "The lab runs in a container on every platform so that everyone's\n"
               "environment matches. You can work natively instead if you\n"
               "prefer - nothing here requires it off macOS."))
    elif docker:
        r = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout.strip():
            say(WARN, "daemon", "installed but not running",
                "Start Docker Desktop, then re-run this check.")
        else:
            say(PASS, "daemon", f"server {r.stdout.strip()}")

    # ------------------------------------------------------------------ done
    print()
    if _fails:
        print(f"{_fails} problem(s) to fix" + (f", {_warns} warning(s)" if _warns else ""))
        print("Fix the [FAIL] lines above and run this again.")
        return 1
    print("SETUP OK" + (f" ({_warns} warning(s) worth reading)" if _warns else ""))
    print("Nothing here needed the board. When yours arrives, continue with\n"
          "  python install.py            # should report everything [ok]\n"
          "  python scripts/verify_board.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
