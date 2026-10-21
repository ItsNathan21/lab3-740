"""sweep.py - run a workload across backends and sizes, report mean and spread.

Cross-platform replacement for saxpy_sweep.ps1, reduction_sweep.ps1,
convolution_sweep.ps1 and cross_backend_sweep.ps1. One implementation on top of
profiler.py, so macOS, Windows and Linux stay in step and there is no dialect
drift between shell variants.

    python scripts/sweep.py saxpy  --sizes 65536,262144,1048576,4194304
    python scripts/sweep.py matmul --sizes 256,512,1024 --backends gpu --variant best
    python scripts/sweep.py convolution --sizes 4,8,16,32,64 --skip npu
    python scripts/sweep.py matmul --sizes 512 --reps 20 --csv out.csv

--sizes maps to each workload's natural size knob: `size` for matmul, saxpy and
reduction, and cin=cout=<value> for convolution (its channel sweep is what moves
arithmetic intensity). Anything else goes through --param.

Run 0 is always discarded as warm-up; it is a different population (cold caches,
and ~45 ms of OpenCL context creation on the GPU's first dispatch). Reported CoV
is over the timed runs only. Defaults are deliberately low so a sweep finishes;
for a number you intend to quote, use --reps 20 or more.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profiler import WORKLOADS, Backend, Profiler, Task  # noqa: E402

# The knob each workload sweeps over. Convolution moves arithmetic intensity
# through channel count rather than a single "size".
SIZE_PARAM = {
    "matmul": ("size",),
    "saxpy": ("size",),
    "reduction": ("size",),
    "convolution": ("cin", "cout"),
}


def parse_params(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param wants key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            out[key] = int(raw)
        except ValueError:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out


def measure(p: Profiler, task: Task, reps: int) -> dict:
    """One configuration, reps timed dispatches after a discarded warm-up."""
    times, fps, last = [], set(), None
    for i in range(max(1, reps) + 1):
        r = p.run(task)
        if not r.ok:
            return {"error": (r.error or "failed")[:70]}
        if r.compute_us is None:
            return {"error": "ran but reported no compute time"}
        if i == 0:
            continue                      # warm-up: different population
        times.append(r.compute_us)
        if r.fingerprint is not None:
            fps.add(round(r.fingerprint, 4))
        last = r
    if not times or last is None:
        return {"error": "no timed runs completed"}
    mean = statistics.fmean(times)
    return {
        "mean_us": mean,
        "cov_pct": (statistics.stdev(times) / mean * 100) if len(times) > 1 else None,
        "min_us": min(times),
        "max_us": max(times),
        "n": len(times),
        "gflops": last.gflops,
        "gbps": last.gbps,
        "fingerprint": next(iter(fps)) if len(fps) == 1 else None,
        "fingerprint_stable": len(fps) <= 1,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workload", choices=sorted(WORKLOADS))
    ap.add_argument("--sizes", default="", help="comma-separated, e.g. 256,512,1024")
    ap.add_argument("--backends", default="cpu,gpu,dsp,npu")
    ap.add_argument("--skip", default="", help="backends to omit, e.g. npu,dsp")
    ap.add_argument("--reps", type=int, default=3, help="timed reps per point (default 3)")
    ap.add_argument("--variant", default=None, help="kernel variant, applied to every backend that has it")
    ap.add_argument("--param", action="append", default=[], metavar="K=V",
                    help="extra kernel parameter, repeatable")
    ap.add_argument("--csv", metavar="PATH")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv[1:])

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    names = [b.strip() for b in args.backends.split(",") if b.strip() and b.strip() not in skip]
    try:
        backends = [Backend(n) for n in names]
    except ValueError as e:
        raise SystemExit(f"unknown backend: {e}")

    wl = WORKLOADS[args.workload]
    backends = [b for b in backends if b in wl.kernels] or list(wl.kernels)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()] or [None]
    fixed = parse_params(args.param)
    keys = SIZE_PARAM[args.workload]

    # The single most common way to produce a wrong cross-backend chart.
    if args.workload == "matmul" and Backend.GPU in backends and args.variant is None:
        print("NOTE: GPU matmul defaults to the naive kernel, ~16x slower than\n"
              "      --variant best. That gap will dominate this sweep.\n", file=sys.stderr)

    print(f"{args.workload} sweep - {len(backends)} backend(s) x {len(sizes)} size(s), "
          f"{args.reps} timed rep(s) after warm-up")
    hdr = f"  {'backend':<8} {'size':>9} {'mean us':>11} {'CoV%':>7} {'GFLOP/s':>9} {'GB/s':>8}  notes"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))

    rows = []
    t0 = time.time()
    for size in sizes:
        for backend in backends:
            params = dict(fixed)
            if size is not None:
                params.update({k: size for k in keys})
            if args.variant:
                params["variant"] = args.variant
            try:
                task = Task(args.workload, backend, params=dict(params),
                            variant=params.pop("variant", None))
            except Exception as e:
                print(f"  {str(backend):<8} {size!s:>9}  skipped: {e}")
                rows.append({"workload": args.workload, "backend": str(backend),
                             "size": size, "error": f"skipped: {e}"})
                continue

            res = measure(Profiler(), task, args.reps)
            row = {"workload": args.workload, "backend": str(backend), "size": size, **res}
            rows.append(row)

            if res.get("error"):
                print(f"  {str(backend):<8} {size!s:>9} {'-':>11} {'-':>7} {'-':>9} {'-':>8}  {res['error']}")
                continue
            note = "" if res["fingerprint_stable"] else "FINGERPRINT VARIED"
            cov = f"{res['cov_pct']:.2f}" if res["cov_pct"] is not None else "-"
            # "-" not 0.0: the kernel did not report bytes, which is not a zero.
            gf = f"{res['gflops']:.1f}" if res["gflops"] else "-"
            gb = f"{res['gbps']:.1f}" if res["gbps"] else "-"
            print(f"  {str(backend):<8} {size!s:>9} {res['mean_us']:>11.1f} {cov:>7} "
                  f"{gf:>9} {gb:>8}  {note}")

    ok = [r for r in rows if not r.get("error")]
    print(f"\n  {len(ok)}/{len(rows)} points, {time.time() - t0:.0f} s")
    if ok and args.reps < 20:
        print("  reps < 20: fine for a shape, not for a number you quote.")
    if any(not r.get("fingerprint_stable", True) for r in ok):
        print("  WARNING: a fingerprint varied across reps - results are not trustworthy.")

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {args.csv}")
    if args.json and rows:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
