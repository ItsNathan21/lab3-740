# Snapdragon Profiler

A profiler for the Qualcomm **HDK8850 (SM8850P, "Snapdragon 8 Elite Gen 5")**. It
dispatches benchmark kernels to the board's four compute engines -- **CPU, GPU,
DSP, NPU** -- and returns comparable, provenance-carrying measurements for each.

Everything is reachable through one file: **`profiler.py`**.

```python
from profiler import Profiler, Backend, convolution

p = Profiler()
for b in Backend:
    print(p.run(convolution(b, cin=8, cout=8)).summary())
```

## Start here

Follow the **Lab 3 handout**. Task 1 sets up your machine, the container and
the board; every task after that says exactly what to run.

## Check the board first

```bash
python install.py                # what is missing, and the fix for each
python scripts/verify_board.py   # link, root, kernels, clocks, thermal, dispatch
```

`verify_board.py` runs real dispatches on all four engines. If it reports
`BOARD OK`, the lab will work.

## The API

| Call | Meaning |
| --- | --- |
| `p.run(task)` | one dispatch -> `Result` |
| `p.run([t1, t2])` | a batch, strictly serial |
| `p.run([t1, t2], parallel=True)` | genuinely concurrent across engines |
| `p.run_dag([...])` | named tasks in dependency order |
| `p.power(task)` | modelled whole-board watts and joules |

Four workloads (`matmul`, `saxpy`, `reduction`, `convolution`) x four backends,
each with variants, kernel parameters and CPU affinity control.

```bash
python profiler.py --catalogue   # every kernel and parameter, generated from the registry
```

**Reference manual:** [`docs/profiler.tex`](docs/profiler.tex) -- execution model,
the full API surface, the kernel and counter catalogues, worked recipes, and the
measurement caveats that decide whether a number is trustworthy. Build it with
`pdflatex docs/profiler.tex` (run it twice, for the table of contents), or paste
it into Overleaf -- it uses only standard packages.

## What this bundle is

The student-facing subset of a larger repository: the profiler, the kernels,
and the scripts the handout tells you to run.

## Reading a number honestly

Three things worth knowing before you trust any result:

- **`p.power()` is a model, not a measurement.** This board has no readable power
  sensor, so power is estimated from telemetry through coefficients fitted once
  against a bench supply. It is whole-board watts, never per-engine, and only
  over intervals of 30 s or more.
- **DSP and NPU power report `unavailable`.** That is the correct answer on this
  hardware, not a gap to work around.
- **Fingerprints are how you know two engines computed the same thing.** If they
  disagree, the comparison is meaningless no matter how good the timings look.
