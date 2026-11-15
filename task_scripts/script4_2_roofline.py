"""Collect empirical roofs and four default-dimension workload points.

python task_scripts/script4_2.py --roofline
python task_scripts/script4_2.py --roofline --dry-run

Uses one Prime core, FP32, three measured processes per configuration.
Pilot runs choose enough internal iterations for >=0.25 s of timed work.
Only iteration counts change for the four default-dimension points.
Compute ceiling: highest mean GFLOP/s among tested configurations.
Memory ceiling: highest mean useful GB/s among large streaming probes.
These are empirical estimates, not guaranteed architectural peaks. AI is
ideal-reuse algorithmic AI, not counter-measured DRAM AI. Repeated passes
can hit caches. Reduction's single output scalar is negligible and omitted.
"""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def counts(workload, params):
    n = params.get("size", 4194304)
    if workload == "matmul":
        n = params.get("size", 512)
        return 2 * n**3, 12 * n**2
    if workload == "saxpy":
        return 2 * n, 12 * n
    if workload == "reduction":
        return n, 4 * n
    h, w, ci, co, k = (params.get(key, default) for key, default in
                         [("H", 256), ("W", 256), ("cin", 8), ("cout", 8), ("k", 3)])
    return 2*h*w*co*k*k*ci, 4*(h*w*ci+k*k*ci*co+h*w*co)


def configurations():
    configs = []
    for name in ("saxpy", "reduction", "convolution", "matmul"):
        configs.append(("default_" + name, "default", name, "neon", {}))
    for variant in ("neon", "eigen"):
        for n in (64, 128, 256, 512):
            configs.append((f"compute_matmul_{variant}_{n}", "compute", "matmul", variant, {"size": n}))
    for c in (16, 32):
        configs.append((f"compute_conv_{c}", "compute", "convolution", "neon", {"cin": c, "cout": c}))
    for name in ("saxpy", "reduction"):
        for n in (16777216, 33554432):
            configs.append((f"memory_{name}_{n}", "memory", name, "neon", {"size": n}))
    return configs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frequency", type=int, default=4396800)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--only-prefix", help="Collect only matching configuration names")
    args = ap.parse_args()
    configs = configurations()
    if args.only_prefix:
        configs = [c for c in configs if c[0].startswith(args.only_prefix)]
        if not configs:
            ap.error('No configurations match --only-prefix')
    if args.dry_run:
        for label, role, name, variant, params in configs:
            f, b = counts(name, params)
            print(f"{label}: role={role}, AI={f/b:.6f}, params={params}, repeats=3")
        return
    sys.path.insert(0, str(ROOT))
    import profiler as p
    subprocess.run([sys.executable, str(ROOT/'scripts/power_pin.py'), 'prime', str(args.frequency)],
                   cwd=ROOT, check=True)
    out = args.output or ROOT/'task_output'/datetime.now().strftime('task4_2_roofline_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={out}", flush=True)
    engine = p.Profiler()
    columns = ['configuration','role','workload','variant','params','ai','iterations',
               'repeat','pinned_khz','timed_ms','gflops','useful_gbps','temp_c','effective_ghz']
    summaries = []
    with (out/'runs.csv').open('w', newline='') as runfile, (out/'raw.txt').open('w') as raw:
        writer = csv.DictWriter(runfile, fieldnames=columns)
        writer.writeheader()
        for label, role, name, variant, params in configs:
            f, b = counts(name, params)
            iterations = 10 if name in ('matmul','convolution') else 100
            def execute(iters, tag):
                kwargs = dict(params, iterations=iters, core='prime', variant=variant,
                              counters='basic', timeout=600)
                if variant != 'eigen':
                    kwargs['threads'] = 1
                result = engine.run(getattr(p, name)(p.Backend.CPU, **kwargs))
                raw.write(f"\n=== {label} {tag} iterations={iters} ===\n{result.raw}\n")
                raw.write(f"derived={result.derived}\ntemp_c={result.temp_c}\n")
                raw.flush()
                if not result.ok or not result.compute_us or result.compute_key != 'wall_us':
                    raise RuntimeError(f"{label}: {result.error}; inspect raw.txt")
                return result
            pilot = execute(iterations, 'pilot')
            iterations = max(iterations, math.ceil(250000 / (pilot.compute_us / iterations)))
            samples = []
            for rep in range(1,4):
                result = execute(iterations, f'repeat={rep}')
                seconds = result.compute_us / iterations / 1e6
                row = dict(configuration=label, role=role, workload=name, variant=variant,
                           params=json.dumps(params,sort_keys=True), ai=f/b, iterations=iterations,
                           repeat=rep, pinned_khz=args.frequency,
                           timed_ms=result.compute_us/1000,
                           gflops=f/seconds/1e9, useful_gbps=b/seconds/1e9,
                           temp_c=result.temp_c, effective_ghz=result.derived.get('effective_ghz'))
                writer.writerow(row)
                runfile.flush()
                samples.append(row)
                print(f"{label} [{rep}/3]: {row['gflops']:.3f} GFLOP/s, {row['useful_gbps']:.3f} useful GB/s", flush=True)
                time.sleep(1)
            summary = {key:samples[0][key] for key in ['configuration','role','workload','variant','params','ai','iterations','pinned_khz']}
            for metric in ['gflops','useful_gbps']:
                values = [r[metric] for r in samples]
                for prefix, fn in [('mean',statistics.mean),('min',min),('max',max)]:
                    summary[f'{prefix}_{metric}'] = fn(values)
            summaries.append(summary)
            with (out/'summary.csv').open('w',newline='') as sf:
                sw = csv.DictWriter(sf, fieldnames=list(summaries[0]))
                sw.writeheader()
                sw.writerows(summaries)
    if args.only_prefix:
        print(f'Partial measurements saved to {out}', flush=True)
        return
    compute = max(summaries,key=lambda s:s['mean_gflops'])
    memory = max((s for s in summaries if s['role']=='memory'),key=lambda s:s['mean_useful_gbps'])
    model = {'compute_gflops':compute['mean_gflops'], 'memory_gbps':memory['mean_useful_gbps'],
             'compute_source':compute['configuration'],'memory_source':memory['configuration'],
             'ridge_ai':compute['mean_gflops']/memory['mean_useful_gbps'],
             'frequency_khz':args.frequency,
             'note':'Empirical maxima of configuration means; useful-byte streaming bandwidth, ideal-reuse AI.'}
    (out/'roofline.json').write_text(json.dumps(model,indent=2)+'\n')
    print(json.dumps(model,indent=2),flush=True)


if __name__ == '__main__':
    main()
