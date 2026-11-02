"""Eight single-Prime matmul roofline points; run inside the HDK container.

Run: python task_scripts/script4_2.py
Preview without accessing the board: python task_scripts/script4_2.py --dry-run
AI assumes 2*N^3 FLOPs and 12*N^2 bytes per multiply (ideal cache reuse).
Modeled GB/s is NOT measured DRAM bandwidth. Repeated passes can reuse cache.
Outputs go to a new timestamped directory inside ignored task_output/.
The Prime clock stays pinned after completion, like the other lab sweeps.
"""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SIZES = (64, 128, 192, 256, 384, 512, 1024, 2048)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=int, default=4396800, help="Prime kHz")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # Roughly >= 1 billion FLOPs per process, with at least 10 passes.
    # Check measured durations against your Task 2.1 trust threshold.
    plan = [(n, max(10, math.ceil(1e9 / (2 * n**3)))) for n in SIZES]
    for n, iterations in plan:
        print(f"N={n:4d} AI={n / 6:9.4f} FLOP/byte iterations={iterations} repeats=3")
    if args.dry_run:
        return

    sys.path.insert(0, str(ROOT))
    from profiler import Backend, Profiler, matmul

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/power_pin.py"), "prime", str(args.frequency)],
        cwd=ROOT, check=True,
    )
    out = ROOT / "task_output" / datetime.now().strftime("task4_2_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print(f"Saving results to {out}", flush=True)
    profiler = Profiler()
    columns = ["size", "ai_flop_per_byte", "iterations", "repeat", "pinned_khz",
               "total_timed_ms", "us_per_matmul", "gflops", "modeled_gbps"]
    summary_columns = ["size", "ai_flop_per_byte", "iterations", "pinned_khz",
                       "mean_gflops", "min_gflops", "max_gflops", "cov_pct",
                       "mean_modeled_gbps", "mean_us_per_matmul", "min_total_timed_ms"]
    with (out / "runs.csv").open("w", newline="") as runs_file, \
         (out / "summary.csv").open("w", newline="") as summary_file, \
         (out / "raw.txt").open("w") as raw:
        runs = csv.DictWriter(runs_file, fieldnames=columns)
        summary = csv.DictWriter(summary_file, fieldnames=summary_columns)
        runs.writeheader()
        summary.writeheader()
        for n, iterations in plan:
            rows = []
            for repeat in range(1, 4):
                result = profiler.run(matmul(
                    Backend.CPU, size=n, iterations=iterations, core="prime",
                    threads=1, variant="neon", counters="basic", timeout=600,
                ))
                raw.write(f"\n=== N={n} iterations={iterations} repeat={repeat} ===\n")
                raw.write(result.raw + "\n")
                raw.write(f"derived={result.derived}\n")
                raw.flush()
                if not result.ok or not result.compute_us_per_iter or not result.compute_us:
                    raise RuntimeError(f"Failed N={n}: {result.error}; inspect {out / 'raw.txt'}")
                seconds = result.compute_us_per_iter * 1e-6
                row = dict(size=n, ai_flop_per_byte=n / 6, iterations=iterations,
                           repeat=repeat, pinned_khz=args.frequency,
                           total_timed_ms=result.compute_us / 1000,
                           us_per_matmul=result.compute_us_per_iter,
                           gflops=2 * n**3 / seconds / 1e9,
                           modeled_gbps=12 * n**2 / seconds / 1e9)
                rows.append(row)
                runs.writerow(row)
                runs_file.flush()
                print(f"N={n} repeat={repeat}: {result.summary()}", flush=True)
            rates = [r["gflops"] for r in rows]
            mean = statistics.mean(rates)
            summary.writerow(dict(
                size=n, ai_flop_per_byte=n / 6, iterations=iterations,
                pinned_khz=args.frequency, mean_gflops=mean,
                min_gflops=min(rates), max_gflops=max(rates),
                cov_pct=100 * statistics.stdev(rates) / mean,
                mean_modeled_gbps=statistics.mean(r["modeled_gbps"] for r in rows),
                mean_us_per_matmul=statistics.mean(r["us_per_matmul"] for r in rows),
                min_total_timed_ms=min(r["total_timed_ms"] for r in rows),
            ))
            summary_file.flush()
    print(f"Plot summary.csv: ai_flop_per_byte (x) vs mean_gflops (y).\n{out}")


if __name__ == "__main__":
    main()
