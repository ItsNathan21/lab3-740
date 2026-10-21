#!/usr/bin/env python3
"""install.py - set this machine up to drive the HDK8850. Windows, macOS, Linux.

    python install.py                  # report what is present and what is not
    python install.py --fix            # download and install what is free to
    python install.py --role dev --fix # also fetch the NDK for kernel builds
    python install.py --qairt <path>   # register a Qualcomm SDK you installed

Standard library only, so it runs on a bare Python before anything exists.

Everything it installs goes in .toolchain/ inside the repo. Nothing touches the
system, nothing needs admin, and deleting that directory undoes it.

Two of the four dependencies cannot be automated: QAIRT and the Hexagon SDK sit
behind a Qualcomm account and a licence click-through, and may not be
redistributed. Only a TA provisioning a board needs them - students do not.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent
TOOLCHAIN = ROOT / ".toolchain"
STATE = TOOLCHAIN / "paths.json"

SYS = platform.system()
IS_WIN, IS_MAC = SYS == "Windows", SYS == "Darwin"
GOOG = {"Windows": "windows", "Darwin": "darwin", "Linux": "linux"}.get(SYS, "linux")

NDK_VER = "r27d"
DL = {
    "platform-tools": (
        f"https://dl.google.com/android/repository/platform-tools-latest-{GOOG}.zip",
        "platform-tools", 16),
    "ndk": (
        f"https://dl.google.com/android/repository/android-ndk-{NDK_VER}-{GOOG}.zip",
        f"android-ndk-{NDK_VER}", 840),
}

OK, MISS, WARN, SKIP, FIXED = "OK", "MISSING", "WARN", "SKIP", "INSTALLED"
_TAG = {OK: "  [ok]      ", MISS: "  [MISSING] ", WARN: "  [warn]    ",
        SKIP: "  [skip]    ", FIXED: "  [got it]  "}
_missing: list[str] = []


def say(status: str, name: str, detail: str = "", remedy: str = "") -> None:
    if status == MISS:
        _missing.append(name)
    print(f"{_TAG[status]}{name}" + (f" - {detail}" if detail else ""))
    for line in (remedy or "").strip().splitlines():
        if line:
            print(f"              {line}")


def state_load() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def state_set(**kw) -> None:
    s = state_load()
    s.update({k: str(v) for k, v in kw.items() if v})
    TOOLCHAIN.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def download(url: str, dest: pathlib.Path, approx_mb: int) -> bool:
    print(f"              downloading {url.rsplit('/', 1)[1]} (~{approx_mb} MB)")
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r              {done * 100 // total:3d}%  "
                          f"{done/1e6:6.0f} / {total/1e6:.0f} MB", end="", flush=True)
        print()
        tmp.replace(dest)
        return True
    except Exception as e:
        print(f"\n              download failed: {e}")
        # A python.org Python on macOS ships with no trusted CA certificates
        # until its own installer is run, so every HTTPS download fails this
        # way. Re-running the same command -- the obvious next move, and what
        # this script used to imply -- fails identically. Name the one command
        # that fixes it.
        if "CERTIFICATE_VERIFY_FAILED" in str(e) or "SSL" in type(e).__name__:
            print("              This is the macOS certificate problem, not a network fault.")
            print("              Python from python.org trusts nothing until you run:")
            print(f"                open \"/Applications/Python {sys.version_info.major}."
                  f"{sys.version_info.minor}/Install Certificates.command\"")
            print("              Then re-run this command.")
        tmp.unlink(missing_ok=True)
        return False


def unzip(archive: pathlib.Path, into: pathlib.Path) -> None:
    print("              extracting ...")
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            # zipfile.extractall writes a symlink entry as a small text file
            # holding the target's name. The Linux NDK's clang, clang++ and
            # friends ARE symlinks, so extractall leaves a clang++ that runs
            # its own name as a shell command and fails with "clang: error:
            # no input files". Recreate them as links.
            if not IS_WIN and (info.external_attr >> 16) & 0o170000 == 0o120000:
                dest = into / info.filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_symlink() or dest.exists():
                    dest.unlink()
                os.symlink(z.read(info).decode(), dest)
            else:
                z.extract(info, into)
    repair_zip_symlinks(into)
    # zipfile drops the executable bit; restore it for anything meant to run.
    if not IS_WIN:
        for p in into.rglob("*"):
            if p.is_symlink():
                continue
            if p.is_file() and (p.parent.name == "bin" or p.suffix in ("", ".sh")):
                try:
                    p.chmod(p.stat().st_mode | 0o111)
                except OSError:
                    pass
    archive.unlink(missing_ok=True)


def fetch(key: str) -> pathlib.Path | None:
    url, dirname, mb = DL[key]
    target = TOOLCHAIN / dirname
    if target.is_dir():
        return target
    TOOLCHAIN.mkdir(parents=True, exist_ok=True)
    arc = TOOLCHAIN / f"{key}.zip"
    if not download(url, arc, mb):
        return None
    unzip(arc, TOOLCHAIN)
    return target if target.is_dir() else None


# ---------------------------------------------------------------------- checks

def check_python() -> None:
    v = sys.version_info
    if v[:2] >= (3, 10):
        say(OK, "Python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        say(MISS, "Python", f"{v.major}.{v.minor}, need >= 3.10",
            "Install Python 3.10+ from python.org and re-run with it.")


def check_stdlib_only() -> None:
    """Prove, not assert, that measuring needs no third-party packages."""
    import ast
    tgts = [ROOT / "profiler.py", ROOT / "scripts/verify_board.py", ROOT / "scripts/sweep.py"]
    tgts += sorted((ROOT / "examples").rglob("*.py"))
    std, ext = set(sys.stdlib_module_names), set()
    for f in tgts:
        if not f.exists():
            continue
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                ext |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                ext.add(n.module.split(".")[0])
    ext -= std | {"profiler"}
    if ext:
        say(WARN, "third-party imports", ", ".join(sorted(ext)),
            "The measurement path should be stdlib-only. Please report this.")
    else:
        say(OK, "dependencies", "none - measuring is stdlib-only")


def check_adb(fix: bool) -> str | None:
    local = TOOLCHAIN / "platform-tools" / ("adb.exe" if IS_WIN else "adb")
    adb = str(local) if local.exists() else shutil.which("adb")
    if adb:
        out = subprocess.run([adb, "version"], capture_output=True, text=True).stdout
        say(OK, "adb", out.splitlines()[0] if out else adb)
        state_set(adb=adb)
        return adb
    if fix:
        d = fetch("platform-tools")
        if d:
            adb = str(d / ("adb.exe" if IS_WIN else "adb"))
            say(FIXED, "adb", adb)
            state_set(adb=adb)
            return adb
    say(MISS, "adb", "not on PATH",
        "python install.py --fix        # downloads it here, no admin needed\n"
        + ("or: brew install --cask android-platform-tools" if IS_MAC else
           "or: sudo apt install android-tools-adb" if not IS_WIN else
           "or install Android platform-tools and add it to PATH"))
    return None


def check_board(adb: str | None) -> bool:
    if not adb:
        say(SKIP, "board", "needs adb")
        return False
    out = subprocess.run([adb, "devices"], capture_output=True, text=True).stdout
    lines = [l for l in out.splitlines()[1:] if l.strip()]
    ready = [l.split()[0] for l in lines if l.endswith("device")]
    if any("unauthorized" in l for l in lines):
        say(MISS, "board", "unauthorized",
            "Accept the USB-debugging prompt on the board's screen, then re-run.")
        return False
    if not ready:
        say(MISS, "board", "none attached",
            "Connect the HDK8850 over USB-C. A charge-only cable is the usual cause.")
        return False
    say(OK if len(ready) == 1 else WARN, "board",
        ready[0] if len(ready) == 1 else f"{len(ready)} attached: {ready}",
        "" if len(ready) == 1 else "Unplug the others, or pass --serial to the tools.")
    return True


def check_kernels(adb: str | None, board: bool) -> None:
    if not board:
        say(SKIP, "kernels on device", "needs a board")
        return
    out = subprocess.run([adb, "shell", "ls /data/local/tmp/profiler-kernels 2>/dev/null"],
                         capture_output=True, text=True).stdout
    entries = set(out.split())
    if not entries:
        # Students provision from the lab container, as the setup guide says.
        # --dsp-skel is not optional in practice: without it the script
        # provisions CPU/GPU/NPU and reports DSP unavailable, which silently
        # removes Tasks 10 and 11.
        fix = ("Provision it from the lab container (about 10 minutes, once per board):\n"
               "  python scripts/provision_device.py \\\n"
               "      --qairt-root /opt/qcom/aistack/qairt/2.40.0.251030 \\\n"
               "      --dsp-skel kernels/matmul/dsp/prebuilt/libdsp_matmul_skel.so")
        say(MISS, "kernels on device", "board not provisioned", fix)
        return
    want = {"gpu": "GPU", "npu": "NPU", "dsp": "DSP", "decode": "decode"}
    miss = [v for k, v in want.items() if k not in entries]
    say(OK if not miss else WARN, "kernels on device",
        f"{len(entries)} entries" + (f"; missing {', '.join(miss)}" if miss else ""),
        "Re-run provisioning to add the missing engines." if miss else "")


def repair_zip_symlinks(root: pathlib.Path) -> int:
    """Turn symlinks that an older install.py extracted as text files back
    into links: a tiny file in a bin/ directory whose whole content is the
    name of a sibling. Returns how many were repaired. No-op on Windows, whose
    NDK has no symlinks."""
    if IS_WIN or not root.is_dir():
        return 0
    fixed = 0
    for bindir in root.rglob("bin"):
        if not bindir.is_dir():
            continue
        for p in bindir.iterdir():
            if p.is_symlink() or not p.is_file() or p.stat().st_size > 128:
                continue
            try:
                target = p.read_text(encoding="ascii").strip()
            except (UnicodeDecodeError, OSError):
                continue
            if target and "/" not in target and "\n" not in target and target != p.name \
                    and (bindir / target).exists():
                p.unlink()
                os.symlink(target, p)
                fixed += 1
    return fixed


def check_ndk(fix: bool, register: str | None = None) -> None:
    # An explicitly registered path wins over everything, and a wrong one is
    # reported rather than silently falling through to a 840 MB re-download.
    if register:
        p = pathlib.Path(register).expanduser()
        if not (p / "toolchains/llvm/prebuilt").is_dir():
            say(MISS, "Android NDK", f"{p} is not an NDK root",
                "point --ndk at the directory containing toolchains/llvm/prebuilt")
            return
        state_set(ndk=p)
        say(OK, "Android NDK", f"{p}  (registered)")
        say(OK, "android-clang", "tools/android-clang - all 38 kernel builds work here")
        return
    saved = state_load().get("ndk")
    cands = [pathlib.Path(p) for p in (os.environ.get("NDK_ROOT"), saved) if p]
    cands += [TOOLCHAIN / f"android-ndk-{NDK_VER}"]
    cands += sorted(pathlib.Path.home().glob("Downloads/android-ndk-*"))
    cands += sorted(pathlib.Path.home().glob("Downloads/*/android-ndk-*"))
    found = next((c for c in cands if (c / "toolchains/llvm/prebuilt").is_dir()), None)
    if not found and fix:
        found = fetch("ndk")
        if found:
            say(FIXED, "Android NDK", str(found))
    if found:
        state_set(ndk=found)
        if not str(found).startswith(str(TOOLCHAIN)):
            say(OK, "Android NDK", str(found))
        n = repair_zip_symlinks(found)
        if n:
            say(FIXED, "NDK symlinks", f"{n} compiler links were plain files; relinked")
        if (ROOT / "tools" / "android-clang").is_file():
            say(OK, "android-clang", "tools/android-clang - all 38 kernel builds work here")
        else:
            say(MISS, "android-clang", "tools/android-clang is missing from this checkout",
                "every kernels/*/build.sh needs it; get it from the course repository")
    else:
        say(MISS, "Android NDK", "not found (needed only to build kernels)",
            "python install.py --role dev --fix     # ~840 MB, free, no account")


def _qnn_core_api(header: pathlib.Path) -> tuple[int, int] | None:
    """(major, minor) from a QnnCommon.h, or None if it cannot be read."""
    try:
        raw = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    out = []
    for part in ("MAJOR", "MINOR"):
        m = re.search(r"^\s*#define\s+QNN_API_VERSION_" + part + r"\s+(\d+)\s*$",
                      raw, re.M)
        if not m:
            return None
        out.append(int(m.group(1)))
    return out[0], out[1]


def qairt_api_note(root: pathlib.Path) -> str:
    """Empty when this SDK can drive our clients; otherwise why not.

    The directory name says nothing useful - 2.40 and 2.44 both work, and the
    device's own bundled 2.23.1 does not. What decides it is the QNN core API in
    the SDK's header against the one the clients were compiled with, which is
    the same comparison provisioning makes before it copies anything.
    """
    sdk = _qnn_core_api(root / "include/QNN/QnnCommon.h")
    client = _qnn_core_api(ROOT / "kernels/matmul/npu/qnn_fp16/include/QNN/QnnCommon.h")
    if sdk is None:
        return "cannot read include/QNN/QnnCommon.h - is this a QAIRT root?"
    if client is None:
        return ""          # nothing to compare against; do not cry wolf
    if sdk[0] != client[0] or sdk[1] < client[1]:
        return (f"core API {sdk[0]}.{sdk[1]} cannot drive our clients "
                f"(need {client[0]}.{client[1]}+). Install a newer QAIRT.")
    return ""


def check_qualcomm_sdk(name: str, key: str, probe: str, why: str,
                       explicit: str | None) -> None:
    """QAIRT and Hexagon: licensed, so they cannot be downloaded automatically."""
    saved = state_load().get(key)
    cands = [pathlib.Path(p) for p in
             (explicit, os.environ.get(key.upper() + "_ROOT"), saved) if p]
    if key == "qairt":
        # Where QPM actually puts it, on each platform, plus the places a
        # hand-extracted zip tends to land.
        for pat in ("Downloads/*/qairt/*", "Qualcomm/*/qairt/*"):
            cands += sorted(pathlib.Path.home().glob(pat))
        for base in ("/opt/qcom/aistack/qairt", "/opt/qcom/aistack",
                     "/opt/qairt", "/opt/qualcomm/qairt",
                     "C:/Qualcomm/AIStack/qairt", "C:/Qualcomm/qairt"):
            cands += sorted(pathlib.Path(base).glob("*"))
    else:
        if IS_WIN:
            cands += sorted(pathlib.Path("C:/Qualcomm/Hexagon_SDK").glob("*"))
        cands += sorted(pathlib.Path.home().glob("Qualcomm/Hexagon_SDK/*"))
        # QPM's own default inside the lab container. Without this, a student
        # who installed the SDK correctly two steps earlier gets
        # "[MISSING] Hexagon SDK" here and "[PASS] installed - 6.7.0.0" from
        # check_setup.py one screen later -- two official scripts disagreeing
        # about whether the setup is done. $HOME is /root in the container, so
        # the glob above looks in /root/Qualcomm/... and finds nothing.
        # The QAIRT branch above already lists /opt/qcom/aistack, and
        # check_setup.py already lists /local/mnt/workspace; only this branch
        # was missed when the other two were updated.
        for base in ("/local/mnt/workspace/Qualcomm/Hexagon_SDK",
                     "/opt/Qualcomm/Hexagon_SDK", "/opt/qcom/Hexagon_SDK"):
            cands += sorted(pathlib.Path(base).glob("*"))
    found = next((c for c in cands if (c / probe).exists()), None)
    if found:
        state_set(**{key: found})
        note = ""
        if key == "qairt":
            note = qairt_api_note(found)
        if key == "hexagon" and not (found / "rtos/qurt/computev81").is_dir():
            note = "No computev81 - this board is Hexagon v81, its skel cannot be built."
        say(WARN if note else OK, name, f"{found.name} at {found}", note)
        return
    if explicit:
        say(MISS, name, f"{explicit} does not contain {probe}")
        return
    say(MISS, name, "not installed", why)


# -------------------------------------------------------------------- activate

def write_activate() -> None:
    ndk = state_load().get("ndk")
    pt = TOOLCHAIN / "platform-tools"
    dirs = [ROOT / "tools"] + ([pt] if pt.is_dir() else [])
    entries = [str(d) for d in dirs]
    # A POSIX shell on Windows (Git Bash, MSYS) only honours PATH entries in
    # /c/... form; "C:/..." is silently ignored and the system adb wins.
    def posix(d: pathlib.Path) -> str:
        s = d.as_posix()
        if IS_WIN and len(s) > 1 and s[1] == ":":
            s = f"/{s[0].lower()}{s[2:]}"
        return s
    sh_entries = [posix(d) for d in dirs]
    TOOLCHAIN.mkdir(parents=True, exist_ok=True)
    (TOOLCHAIN / "activate.sh").write_text(
        "# source this:  . .toolchain/activate.sh\n"
        f'export PATH="{":".join(sh_entries)}:$PATH"\n'
        + (f'export NDK_ROOT="{pathlib.Path(ndk).as_posix()}"\n' if ndk else ""),
        encoding="utf-8", newline="\n")
    (TOOLCHAIN / "activate.ps1").write_text(
        "# dot-source this:  . .toolchain\\activate.ps1\n"
        f'$env:PATH = "{";".join(entries)};$env:PATH"\n'
        + (f'$env:NDK_ROOT = "{ndk}"\n' if ndk else ""), encoding="utf-8")
    print("\n  wrote .toolchain/activate.sh and .toolchain/activate.ps1")
    print("    sh/zsh :  . .toolchain/activate.sh")
    print("    pwsh   :  . .toolchain\\activate.ps1")


# ------------------------------------------------------------------------ main

QAIRT_WHY = ("Qualcomm-licensed and not redistributable, so this cannot fetch it.\n"
             "Every student needs it: it provisions the board and Tasks 12 and 13\n"
             "run on it. Install it in the lab container:\n"
             "  qpm-cli --license-activate qualcomm_ai_runtime_sdk\n"
             "  qpm-cli --extract qualcomm_ai_runtime_sdk -v 2.40.0.251030\n"
             "  It goes quiet for up to 10 minutes, then asks [y/n]: answer y.\n"
             "  Do not pass --path; QPM's default location is searched. If it\n"
             "  is somewhere else:  python install.py --qairt <that path>")
HEX_WHY = ("Qualcomm-licensed and not redistributable.\n"
           "Needed to COMPILE the DSP tasks, which are graded. Measuring with the\n"
           "prebuilt skels works without it. Install it in the lab container:\n"
           "  qpm-cli --license-activate hexagonsdk6.x\n"
           "  qpm-cli --install hexagonsdk6.x\n"
           "  Any 6.6.0.0 or newer with computev81. Do not pass --path; QPM's\n"
           "  default location is searched. Run hexagon-doctor to see what is wrong.")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=("student", "ta", "dev"), default="student")
    ap.add_argument("--fix", action="store_true", help="download what is freely available")
    ap.add_argument("--qairt", metavar="PATH", help="register an installed QAIRT")
    ap.add_argument("--hexagon", metavar="PATH", help="register an installed Hexagon SDK")
    ap.add_argument("--ndk", metavar="PATH",
                    help="register an already-downloaded Android NDK")
    a = ap.parse_args(argv[1:])
    role = a.role
    if (a.qairt or a.hexagon or a.ndk) and role == "student":
        role = "dev" if (a.hexagon or a.ndk) else "ta"

    print(f"HDK8850 profiler setup - {SYS}, role: {role}"
          + ("   (--fix: will download)" if a.fix else "") + "\n")

    print("Measuring on the board:")
    check_python()
    check_stdlib_only()
    adb = check_adb(a.fix)
    check_kernels(adb, check_board(adb))

    if role in ("ta", "dev"):
        print("\nProvisioning a board (TA):")
        check_qualcomm_sdk("QAIRT (NPU runtime)", "qairt",
                           "lib/aarch64-android/libQnnHtp.so", QAIRT_WHY, a.qairt)

    if role == "dev":
        print("\nBuilding kernels:")
        check_ndk(a.fix, a.ndk)
        check_qualcomm_sdk("Hexagon SDK (DSP skels)", "hexagon", "incs", HEX_WHY, a.hexagon)

    write_activate()

    print("\n" + "-" * 64)
    if not _missing:
        print("Ready.\n"
              "  python scripts/verify_board.py                          # 17 checks, real dispatches\n"
              "  python profiler.py matmul cpu -p size=512 --core prime  # first measurement")
        return 0
    print("Missing: " + ", ".join(_missing))
    free = [m for m in _missing if m in ("adb", "Android NDK")]
    if free and not a.fix:
        print("\nFree to download: " + ", ".join(free) + "\n  python install.py"
              + (" --role dev" if "Android NDK" in free else "") + " --fix")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
