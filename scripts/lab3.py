#!/usr/bin/env python3
"""lab3.py - the one command that runs everything for Lab 3.

  python scripts/lab3.py                      open a shell in your lab container
  python scripts/lab3.py qpm <file.deb>       install Qualcomm Package Manager in it
  python scripts/lab3.py clone <your-repo>    clone your repository into it (once)
  python scripts/lab3.py pull                 pull your latest code in        <- daily
  python scripts/lab3.py run <command...>     run one command in it
  python scripts/lab3.py save                 copy your submission zip out
  python scripts/lab3.py backup               copy out anything git does not have
  python scripts/lab3.py status               what is installed, is the board visible
  python scripts/lab3.py reset                DESTROY the container and start over

On macOS and Linux, type python3 instead of python.

This drives the lab container (named lab3). It runs in ANY terminal - PowerShell,
Windows Terminal, cmd, macOS Terminal, a Linux shell - because the only thing
it needs is the Python you already installed for the lab. No Git Bash.

HOW THE LAB FITS TOGETHER

  your machine                     this container
  ------------                     --------------
  edit code in your editor   -->   git pull       -->  build, run, measure
  git push to YOUR repo            both Qualcomm SDKs    against the board

You write code where you always write code, with your own editor, and push it
to your own GitHub repository. The container pulls it and runs it. It is a
build-and-measure box, not a place you live in.

WHY A CONTAINER AT ALL: the Hexagon and QAIRT SDKs are x86-64 Linux only --
there is no macOS build. The container is the Linux machine they need. It
also means everyone in the course runs an identical toolchain, so a number
you measure is comparable to a number your classmate measures.

WHAT IS IN IT, AND WHAT LOSING IT COSTS YOU

The container is created once and KEPT. `exit` stops it, it does not delete
it, and the next `python scripts/lab3.py` picks up where you left off. Inside
are the two SDKs (about 9 GB, and QAIRT makes you log in to Qualcomm again)
and your clone.

Deleting it does NOT cost you your code -- that is on your own disk and in
your own repository, which is the point of working this way. It costs you the
SDK install. So do not run `docker system prune` or Docker Desktop's
"Clean / Purge data" while you still need it; but if it happens you are an
afternoon of downloading away from fine, not a rewrite away.

The one thing that exists ONLY in here is whatever you measured and have not
pushed: result files, logs, traces. `python scripts/lab3.py backup` copies
exactly those out -- it asks git what it does not have, and takes that.
"""
from __future__ import annotations

import getpass
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

IMAGE = os.environ.get("PROFILER_IMAGE", "ghcr.io/abarande/snapdragon-profiler-qualcomm:v3")
NAME = os.environ.get("LAB_CONTAINER", "lab3")
WORKDIR = "/workspace"

WINDOWS = os.name == "nt"
# Git Bash sets MSYSTEM. Its window (mintty) is not a Windows console, which is
# the one case below that needs special handling.
GIT_BASH = WINDOWS and bool(os.environ.get("MSYSTEM"))

# How this script was invoked, so every hint below is a line the reader can
# actually paste: `python scripts/lab3.py` from the repository root, or
# `python lab3.py` from inside scripts/.
_PY = "python" if WINDOWS else "python3"
ME = f"{_PY} {(sys.argv[0] or 'scripts/lab3.py').replace(os.sep, '/')}"


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def say(msg: str = "") -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ docker
def docker(*args: str, capture: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run one docker command. Never raises on a non-zero exit; callers check."""
    try:
        return subprocess.run(
            ["docker", *args],
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
    except FileNotFoundError:
        die("docker not found. Install Docker Desktop, start it, and try again.")
        raise  # unreachable; keeps type checkers quiet


def out(*args: str) -> str:
    """stdout of a docker command, CRs stripped, or '' if it failed."""
    r = docker(*args)
    return r.stdout.replace("\r", "").strip() if r.returncode == 0 else ""


def ok(*args: str) -> bool:
    return docker(*args).returncode == 0


# -------------------------------------------------------- the keyboard
def _mintty_pty(fd: int) -> bool:
    """True if fd is Git Bash's terminal (a named pipe, not a console).

    A Windows program such as python.exe or docker.exe started from mintty
    sees a pipe called msys-...-pty0-from-master instead of a console. That is
    why `docker exec -it` from Git Bash gives a blank screen with no prompt and
    every line typed arrives with a stray carriage return.
    """
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class FileNameInfo(ctypes.Structure):
            _fields_ = [("FileNameLength", wintypes.DWORD),
                        ("FileName", wintypes.WCHAR * 1024)]

        info = FileNameInfo()
        handle = msvcrt.get_osfhandle(fd)
        if not ctypes.windll.kernel32.GetFileInformationByHandleEx(
                handle, 2, ctypes.byref(info), ctypes.sizeof(info)):  # 2 = FileNameInfo
            return False
        name = info.FileName[: info.FileNameLength // 2]
        return ("msys-" in name or "cygwin-" in name) and "-pty" in name
    except Exception:
        return False


def keyboard() -> tuple[list[str], list[str], dict | None]:
    """How to hand the container a real keyboard: (prefix, exec flags, env).

    A real terminal (PowerShell, Windows Terminal, macOS Terminal, a Linux
    shell) gets `docker exec -it`. Git Bash gets the same through winpty, which
    ships with Git for Windows and gives docker.exe a real console. Anything
    else - output piped into a file, a script feeding stdin - gets `-i`, since
    `-t` would fail there.
    """
    if sys.stdin.isatty() and sys.stdout.isatty():
        return [], ["-it"], None
    if GIT_BASH and _mintty_pty(0) and _mintty_pty(1):
        if shutil.which("winpty"):
            # winpty is itself a Git Bash program, so without these it would
            # rewrite /workspace into C:/Program Files/Git/workspace.
            env = dict(os.environ, MSYS_NO_PATHCONV="1", MSYS2_ARG_CONV_EXCL="*")
            return ["winpty"], ["-it"], env
        say("warning: Git Bash's window cannot give the container a keyboard, and")
        say("         winpty was not found. Run this from PowerShell instead.")
    return [], ["-i"], None


def exec_in(args: list[str]) -> int:
    """docker exec with your keyboard attached; returns the exit code."""
    prefix, flags, env = keyboard()
    try:
        return subprocess.call([*prefix, "docker", "exec", *flags, "-w", WORKDIR, NAME, *args], env=env)
    except KeyboardInterrupt:
        return 130


def _unpaste(s: str) -> str:
    """Strip bracketed-paste markers (ESC[200~ / ESC[201~) and surrounding space."""
    return re.sub(r"(\x1b|\^\[)\[20[01]~", "", s).strip()


# ------------------------------------------------------------ host paths
def host_path(p: str) -> str:
    """A path on YOUR machine, in the form this OS's docker can open.

    Git Bash expands ~/Downloads to /c/Users/you/Downloads before Python sees
    it; Windows reads that as C:\\c\\Users\\..., so turn it back into C:/...
    PowerShell does not expand ~ for programs at all, so do that here too.
    """
    p = os.path.expanduser(p)
    if WINDOWS:
        m = re.match(r"^/([a-zA-Z])(/.*)?$", p)
        if m:
            p = f"{m.group(1).upper()}:{m.group(2) or '/'}"
    return p


# ------------------------------------------------------------ the container
def lab_network() -> str:
    """How to wire the container to the adb server on this machine.

    adb is a client and a server; the server owns the USB cable and binds
    127.0.0.1:5037 -- loopback only. Getting the container's client to that
    server is the one piece of plumbing that differs by Docker flavour.

      Docker Desktop (Windows, macOS, and Desktop on Linux)
          host.docker.internal reaches the host through a proxy the adb
          server accepts as local. Verified working; do not change.

      native Docker Engine on Linux
          host.docker.internal resolves to the docker0 bridge gateway, which
          a loopback-only server never sees, so `adb devices` comes back EMPTY
          with no error. --network host puts the container in the host's
          network namespace and it just works -- without starting adb with -a,
          which would expose the server to the network.

    Override with LAB_NETWORK=host or LAB_NETWORK=bridge if the guess is wrong.
    """
    if os.environ.get("LAB_NETWORK"):
        return os.environ["LAB_NETWORK"]
    if "Docker Desktop" in out("info", "--format", "{{.OperatingSystem}}"):
        return "bridge"
    return "host" if out("info", "--format", "{{.OSType}}") == "linux" else "bridge"


def exists() -> bool:
    return NAME in out("ps", "-a", "--format", "{{.Names}}").splitlines()


def running() -> bool:
    return NAME in out("ps", "--format", "{{.Names}}").splitlines()


def cloned() -> bool:
    return ok("exec", NAME, "test", "-d", f"{WORKDIR}/.git")


def empty_workspace() -> bool:
    return out("exec", NAME, "ls", "-A", WORKDIR) == ""


def start() -> None:
    if not running() and not ok("start", NAME):
        die(f"could not start container '{NAME}'")


def create() -> None:
    say("=== creating your lab container (once) ===")
    # --platform: Qualcomm's tools are x86-only; on Apple Silicon Rosetta
    #   translates them.
    # ANDROID_ADB_SERVER_ADDRESS: a container owns no USB. The adb server on
    #   your machine owns the cable, and this points the container's client at
    #   it. Without it the board is invisible in here even though it is
    #   plugged in.
    # No bind mounts and no --rm, both deliberately. Nothing on your disk is
    #   exposed to this container, and it is meant to be kept.
    net = lab_network()
    if net == "host":
        say("  (native Linux Docker Engine: using host networking so the")
        say("   container can reach the adb server on 127.0.0.1)")
        wiring = ["--network", "host",
                  "-e", "ANDROID_ADB_SERVER_ADDRESS=127.0.0.1"]
    else:
        wiring = ["-e", "ANDROID_ADB_SERVER_ADDRESS=host.docker.internal",
                  "--add-host", "host.docker.internal:host-gateway"]
    if not ok("create", "-it", "--name", NAME, "--platform", "linux/amd64",
              *wiring, "-e", "ANDROID_ADB_SERVER_PORT=5037",
              "-w", WORKDIR, IMAGE, "bash"):
        die("could not create the container")
    if not ok("start", NAME):
        die("could not start the container")
    say(f"created and started: {NAME}")
    say()
    say("Three things to do next, in this order:")
    say(f"  1. {ME} qpm <your QualcommPackageManager3 .deb>")
    say(f"  2. {ME}                 # then install both SDKs with qpm-cli")
    say(f"  3. {ME} clone <your repository URL>")
    say()


def ensure() -> None:
    if not exists():
        if not ok("image", "inspect", IMAGE):
            say(f"=== fetching {IMAGE} (once, about 430 MB) ===")
            if docker("pull", "--platform", "linux/amd64", IMAGE, capture=False).returncode != 0:
                die("could not download the lab image, and it is not on this machine.\n"
                    "  Check your connection and try again.")
            say()
        create()
    else:
        start()


def need_container() -> None:
    if not exists():
        die(f"no container named '{NAME}' yet. Run: {ME}")
    start()


# ------------------------------------------------------------ commands
def cmd_shell(args: list[str]) -> int:
    ensure()
    return exec_in(["bash", *args])


def cmd_run(args: list[str]) -> int:
    if not args:
        die(f"usage: {ME} run <command...>")
    ensure()
    # Running a LAB command before the repository is in there produces a bare
    # "can't open file '/workspace/scripts/...'", which reads like a broken
    # bundle rather than a step not done yet. But only lab commands:
    # `run qpm-cli --login ...` happens BEFORE the clone and is fine.
    first = args[0]
    if first in ("python", "python3", "bash", "sh", "make", "make_final") or first.startswith("./"):
        if not cloned() and empty_workspace():
            die(f"{WORKDIR} is empty - your code is not in the container yet.\n"
                f"      {ME} clone <your repository URL>")
    return exec_in(args)


def cmd_qpm(args: list[str]) -> int:
    # The .deb is downloaded on your machine because it sits behind a Qualcomm
    # login and a container has no browser. It is COPIED in, not mounted, so it
    # behaves identically on every host.
    if not args:
        die(f"usage: {ME} qpm <path to QualcommPackageManager3*.deb>\n"
            "  Download it from   https://qpm.qualcomm.com/   (a free Qualcomm account).")
    pattern = host_path(args[0])
    matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else \
        ([pattern] if os.path.isfile(pattern) else [])
    if not matches:
        die(f"no such file: {args[0]}")
    deb = max(matches, key=os.path.getmtime)
    if len(matches) > 1:
        say(f"({len(matches)} files match; using the newest: {os.path.basename(deb)})")
    if not deb.endswith(".deb"):
        die(f"that is not a .deb file: {deb}")
    ensure()
    say(f"=== copying {os.path.basename(deb)} into the container ===")
    if not ok("cp", deb, f"{NAME}:/tmp/qpm.deb"):
        die("could not copy it in")
    say("=== installing ===")
    # The image already carries sudo, xdg-utils and policykit-1 -- three
    # packages QPM declares and does not ship -- so a bare dpkg -i configures
    # cleanly and there is no apt step to get wrong.
    if docker("exec", NAME, "dpkg", "-i", "/tmp/qpm.deb", capture=False).returncode != 0:
        die("dpkg refused the package")
    docker("exec", NAME, "rm", "-f", "/tmp/qpm.deb")
    if not ok("exec", NAME, "sh", "-c", "command -v qpm-cli >/dev/null"):
        die("it installed, but qpm-cli is not on PATH. Tell a TA.")
    say()
    say(f"qpm-cli is installed. Now open a shell ({ME}) and run:")
    say("  qpm-cli --login <whatever you sign in to qpm.qualcomm.com with>")
    say()
    say("  qpm-cli --license-activate qualcomm_ai_runtime_sdk")
    say("  qpm-cli --extract qualcomm_ai_runtime_sdk -v 2.40.0.251030")
    say()
    say("  qpm-cli --license-activate hexagonsdk6.x")
    say("  qpm-cli --install hexagonsdk6.x")
    say()
    say("Both SDKs ask you to accept a licence. Answer y.")
    say()
    say("QAIRT goes quiet at ~99% CPU for up to 10 minutes after")
    say('"Downloading ...." and BEFORE it asks. That is it verifying a')
    say("1.4 GB archive, not a hang. Do not kill it.")
    say()
    say("If a product name is rejected: --product-list gives NAMES, and")
    say("qpm-cli --info <product> gives that product's VERSIONS. You need")
    say("both -- the older name qualcomm_ai_engine_direct is still listed")
    say("but stops at 2.31, which is why it has to be the runtime_sdk one.")
    say()
    say("Let both land wherever QPM puts them by default, and do not pin a")
    say("Hexagon version -- any 6.6.0.0 or newer works, and Qualcomm no")
    say("longer reliably serves 6.6. QAIRT IS pinned, deliberately: it")
    say("ships to the board and our clients compile against its API.")
    return 0


def cmd_clone(args: list[str]) -> int:
    if not args:
        die(f"usage: {ME} clone <your repository URL>\n"
            "  YOUR repository -- the one you push your lab code to, not the course's.\n"
            "  Copy the URL from the green Code button on GitHub.")
    url = args[0]
    ensure()
    if cloned():
        die(f"{WORKDIR} is already a clone. To get your latest code:\n      {ME} pull")
    if not empty_workspace():
        die(f"{WORKDIR} is not empty and is not a clone. Look at what is in\n"
            f"  there before overwriting it:   {ME} run ls -la")
    say(f"=== cloning {url} ===")
    say()
    # Unauthenticated first: if the repository is public this just works and
    # nobody gets asked for anything.
    if ok("exec", "-e", "GIT_TERMINAL_PROMPT=0", NAME, "git", "clone", "-q", url, WORKDIR):
        say(f"cloned into {WORKDIR}")
        return 0
    host = urlparse(url).hostname if url.startswith("https://") else None
    if not host:
        die("clone failed, and only an https:// URL can be cloned with a token.\n"
            "  Use the HTTPS URL from the green Code button on GitHub.")
    say("That repository is private, so git needs to know who you are.")
    say()
    say("  Username: your GitHub username")
    say("  Token:    a personal access token, NOT your GitHub password.")
    say("            GitHub stopped accepting passwords over HTTPS in 2021.")
    say()
    say("  Make one at  https://github.com/settings/tokens  -- classic")
    say("  token, one scope: repo. Copy it before you close the page.")
    say("  Pasting the token shows nothing. Paste and press Enter.")
    say()
    # Asked for HERE rather than by git inside the container, for two reasons.
    # Terminals wrap pasted text in invisible markers (ESC[200~ ... ESC[201~,
    # "bracketed paste"); git's hidden prompt keeps them, so a correctly pasted
    # token fails with no clue why. And git's own prompt depends on the
    # container getting a real terminal, which Git Bash cannot provide.
    try:
        user = _unpaste(input("  Username: "))
        token = _unpaste(getpass.getpass("  Token:    "))
    except (EOFError, KeyboardInterrupt):
        say()
        die("cancelled.")
    if not user or not token:
        die("a username and a token are both needed.")
    # Store first, then clone without prompting. The helper has to be in place
    # before git uses the token: git only hands a token to whatever helper is
    # configured at that moment, so `pull` can reuse it later.
    cred = f"protocol=https\nhost={host}\nusername={user}\npassword={token}\n\n"
    docker("exec", NAME, "git", "config", "--global", "credential.helper", "store")
    subprocess.run(["docker", "exec", "-i", NAME, "git", "credential", "approve"],
                   input=cred, text=True, capture_output=True)
    r = docker("exec", "-e", "GIT_TERMINAL_PROMPT=0", NAME, "git", "clone", "-q", url, WORKDIR)
    if r.returncode == 0:
        say()
        say(f"cloned into {WORKDIR}")
        say()
        say("git will remember that token inside this container so pull")
        say("does not ask every time. It is kept in plain text, in here.")
        say("Do not share this container, and revoke the token when the")
        say("course ends.")
        return 0
    # Forget a token that did not work, so a retry starts clean.
    subprocess.run(["docker", "exec", "-i", NAME, "git", "credential", "reject"],
                   input=cred, text=True, capture_output=True)
    detail = (r.stderr or "").strip().splitlines()
    die("clone failed. Check the URL, that the token has the repo scope, and\n"
        "  that this GitHub account can see the repository."
        + (f"\n  git said: {detail[-1]}" if detail else ""))
    return 1


def cmd_pull(_: list[str]) -> int:
    need_container()
    if not cloned():
        die(f"{WORKDIR} is not a clone yet. Run:\n      {ME} clone <your repository URL>")
    return exec_in(["git", "pull"])


def cmd_save(_: list[str]) -> int:
    need_container()
    src = f"{WORKDIR}/final_lab3.zip"
    if not ok("exec", NAME, "test", "-f", src):
        die(f"no final_lab3.zip in the container yet.\n  Build it first:  {ME} run python make_final")
    if not ok("cp", f"{NAME}:{src}", "final_lab3.zip"):
        die("could not copy the submission out")
    say("copied final_lab3.zip to this folder. That is the file you submit.")
    return 0


def cmd_backup(_: list[str]) -> int:
    need_container()
    if not cloned():
        die(f"{WORKDIR} is not a clone, so there is nothing to compare it\n"
            f"  against. Copy things out by hand:   docker cp {NAME}:{WORKDIR}/<file> .")
    # Ask git what it does not have. Anything modified or untracked exists
    # only in this container, and that is exactly what is worth copying:
    # results, traces, logs, and any edit made in here rather than on your own
    # machine. -z keeps names with spaces and renames unambiguous.
    say("=== asking git what exists only in the container ===")
    raw = docker("exec", "-w", WORKDIR, NAME, "git", "status", "--porcelain=v1", "-z",
                 "--untracked-files=all").stdout or ""
    entries = raw.split("\0")
    files: list[str] = []
    i = 0
    while i < len(entries):
        e = entries[i]
        i += 1
        if len(e) < 4:
            continue
        status, path = e[:2], e[3:]
        if "R" in status or "C" in status:
            i += 1  # the next entry is the original name
        if "D" in status:
            continue  # deleted in here: nothing to copy
        files.append(path)
    if not files:
        say("nothing to back up - your repository already has everything.")
        return 0
    dest = time.strftime("lab3-backup-%Y%m%d-%H%M%S")
    n = 0
    for f in files:
        target = os.path.join(dest, *f.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if ok("cp", f"{NAME}:{WORKDIR}/{f}", target):
            n += 1
    if n == 0:
        shutil.rmtree(dest, ignore_errors=True)
        say("nothing could be copied out.")
    else:
        say(f"copied {n} item(s) to ./{dest}")
        say("If any of that is code, it belongs in your repository - put it")
        say("there rather than relying on this folder.")
    return 0


def cmd_status(_: list[str]) -> int:
    if not exists():
        say(f"container '{NAME}': does not exist yet (run: {ME})")
        return 0
    say(f"container '{NAME}': {out('inspect', NAME, '--format', '{{.State.Status}}')}")
    img = out("inspect", NAME, "--format", "{{.Config.Image}}")
    say(f"  image:      {img}")
    if img != IMAGE:
        say(f"              (current lab image is {IMAGE}; yours is older.")
        say("               That is fine and it keeps working. Rebuilding to")
        say("               pick up a newer one destroys the SDKs inside, so")
        say("               only do it on a TA's say-so.)")
    start()

    def q(cmd: str) -> str:
        return out("exec", NAME, "sh", "-c", cmd)

    def first(pattern: str) -> str:
        return q(f'd=$(ls -d {pattern} 2>/dev/null | head -1); echo "${{d:-not installed yet}}"')

    qpm = q('command -v qpm-cli >/dev/null && echo installed || echo "not installed yet"')
    say(f"  qpm-cli:    {qpm}")
    say(f"  QAIRT:      {first('/opt/qcom/aistack/qairt/*')}")
    say(f"  Hexagon:    {first('/local/mnt/workspace/Qualcomm/Hexagon_SDK/*')}")
    if cloned():
        url = q(f"cd {WORKDIR} && git config --get remote.origin.url")
        branch = q(f"cd {WORKDIR} && git rev-parse --abbrev-ref HEAD")
        say(f"  your repo:  {url}")
        say(f"  branch:     {branch}")
    else:
        say(f"  your repo:  not cloned yet ({ME} clone <url>)")
    net = out("inspect", NAME, "--format", "{{.HostConfig.NetworkMode}}")
    say(f"  network:    {net}")
    lines = out("exec", NAME, "adb", "devices").splitlines()
    board = lines[1].strip() if len(lines) > 1 else ""
    if board:
        say(f"  board:      {board}")
    else:
        say("  board:      none visible")
        say("              Does `adb devices` on your OWN machine show it?")
        if net == "host":
            say("              If it does, check the cable and that the board is")
            say("              authorised. This container is already on host")
            say("              networking, so the wiring is not the problem.")
        else:
            say("              If it does and this does not, the container cannot")
            say("              reach your adb server. On native Linux Docker Engine")
            say("              that is the known case -- recreate it with host")
            say("              networking:")
            say(f"                {ME} backup && {ME} reset")
            say(f"                LAB_NETWORK=host {ME}")
    return 0


def cmd_reset(_: list[str]) -> int:
    if not exists():
        die(f"no container named '{NAME}' to reset.")
    say(f"This DELETES the container '{NAME}' and everything inside it:")
    say("  - both Qualcomm SDKs (about 9 GB, and QAIRT needs your login again)")
    say("  - anything you measured in here and have not copied out")
    say()
    say("Your code is safe: it is on your own disk and in your repository.")
    say(f"Run '{ME} backup' first if there are results you want.")
    say()
    try:
        answer = input("Type the word DELETE to confirm: ").strip()
    except EOFError:
        answer = ""
    if answer != "DELETE":
        say("cancelled, nothing was removed.")
        return 0
    if ok("rm", "-f", NAME):
        say(f"removed. Run '{ME}' to start over.")
        return 0
    die("could not remove it")
    return 1


COMMANDS = {
    "shell": cmd_shell, "run": cmd_run, "qpm": cmd_qpm, "clone": cmd_clone,
    "pull": cmd_pull, "save": cmd_save, "backup": cmd_backup,
    "status": cmd_status, "reset": cmd_reset,
}


def main(argv: list[str]) -> int:
    cmd, args = (argv[0], argv[1:]) if argv else ("shell", [])
    if cmd in ("-h", "--help", "help"):
        say(__doc__.replace("python scripts/lab3.py", ME))
        return 0
    if cmd not in COMMANDS:
        die(f"unknown command '{cmd}'.\n"
            "  Try: shell | qpm | clone | pull | run | save | backup | status | reset")
    if shutil.which("docker") is None:
        die("docker not found. Install Docker Desktop, start it, and try again.")
    if not ok("info"):
        die("Docker is installed but not running. Start Docker Desktop and try again.")
    return COMMANDS[cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
