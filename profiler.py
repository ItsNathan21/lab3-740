#!/usr/bin/env python3
"""SM8850P heterogeneous profiler wrapper: four workloads x four backends over adb.

    p = Profiler()
    p.run(convolution(Backend.NPU, cin=8, cout=8))       -> Result
    p.run([t1, t2])                                      -> [Result, ...]  serial
    p.run([t1, t2], parallel=True)                       -> [Result, ...]  concurrent
    p.run_dag([...])                                     -> {name: Result}

CPU hardware counters ride along with any dispatch:

    r = p.run(convolution(Backend.CPU, cin=32, cout=32,
                          threads=6, core="gold",       # where, and how many
                          counters="cache", per_core=True))
    r.counters      # {'raw-l1d-cache': 1.2e8, 'raw-l1d-cache-refill': 3.4e5, ...}
    r.derived       # {'ipc': 2.91, 'l1d_miss_pct': 0.28, ...}
    r.per_core      # {0: {...}, 1: {...}, ...}

`counters=` takes a preset (basic|branch|cache|tlb|stalls|mix|memory) or any
list of events; `python profiler.py --list-counters` prints the catalogue of
what this silicon actually implements.

Full reference — installation, execution model, API, kernel/parameter catalogue,
recipes, caveats: docs/profiler.tex.  Live tour: `python profiler.py --demo`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

__all__ = [
    # dispatch
    "Profiler", "Device", "Backend", "Task", "Result",
    # workload constructors
    "matmul", "saxpy", "reduction", "convolution", "nbody", "decode",
    "model", "custom",
    # registries, for callers that enumerate what exists
    "WORKLOADS", "MODELS",
]

DEVICE_BASE = "/data/local/tmp/profiler-kernels"
_HERE = pathlib.Path(__file__).resolve().parent


def _default_power_artifact() -> pathlib.Path:
    """Prefer a real Hexagon-calibrated artifact; fall back to CPU/GPU-only.

    `npu_only` appears here the moment a Hexagon calibration lands, without any
    caller changing. Until then the cpu_gpu_only artifact is what exists, and it
    reports NPU/DSP as unavailable rather than guessing.
    """
    models = _HERE / "references" / "phase-5-energy" / "models"
    # SM8850 first: its artifact is board-gated anyway (power_estimate.py refuses
    # a serial mismatch), so listing both boards here is safe and the right one
    # wins on whichever board is attached.
    for name in ("sm8850-teaching-v2-cpu-gpu.json",        # SM8850, cpu_gpu_freq
                 "sm8750-teaching-v2-npu.json",           # fitted, if it exists
                 "sm8750-teaching-v2-npu-declared.json",  # declared NPU term
                 "sm8750-teaching-v2-cpu-gpu.json"):      # CPU/GPU only
        cand = models / name
        if cand.exists():
            return cand
    return models / "sm8750-teaching-v2-cpu-gpu.json"

# Converted networks, run through QNN's own backends rather than our kernels.
# One DLC executes on any engine by swapping --backend, so the ENGINE is the
# variable and the model is fixed -- which is the whole point of the case study.
# Built by scripts/model_partition.py; see docs/student/07-model-case-study.md.
MODEL_DIR = "/data/local/tmp"
MODELS: dict[str, dict] = {
    "mobilenetv2": {"dir": "mnv2", "dlc": "mnv2.dlc",       "list": "input_list.txt",
                    "macs": 301e6,  "kind": "cnn",  "note": "depthwise-separable, linear"},
    "mnv2_s1":     {"dir": "mnv2", "dlc": "mnv2_s1.dlc",    "list": "input_list.txt",
                    "macs": 120.8e6, "kind": "cnn", "note": "stage 1 of 3 (40% MACs)"},
    "mnv2_s2":     {"dir": "mnv2", "dlc": "mnv2_s2.dlc",    "list": "l2.txt",
                    "macs": 104.6e6, "kind": "cnn", "note": "stage 2 of 3 (35%)"},
    "mnv2_s3":     {"dir": "mnv2", "dlc": "mnv2_s3.dlc",    "list": "l3.txt",
                    "macs": 75.3e6,  "kind": "cnn", "note": "stage 3 of 3 (25%)"},
    "minilm":      {"dir": "nlp",  "dlc": "minilm.dlc",     "list": "l_minilm.txt",
                    "macs": 15e6,   "kind": "encoder",
                    "note": "6-layer BERT encoder, seq=128"},
    "distilgpt2":  {"dir": "nlp",  "dlc": "distilgpt2.dlc", "list": "l_gpt.txt",
                    "macs": 88.4e6, "kind": "decoder",
                    "note": "6-layer decoder PREFILL, seq=128, no KV cache"},
    # Decode steps at bucket 32, fp32, exported by scripts/export_llm_mask4d.py.
    # These are the size knob for the LLM case study: same mechanism, 15x the
    # parameters. fp32 is not a choice -- the CPU backend refuses fp16 DLCs
    # (discovery 71) -- so the DLCs are large and decode is weight-bound.
    "qwen05_dec32": {"dir": "qwen", "dlc": "qwen_m4d_fp32.dlc", "list": "l_qm4d.txt",
                     "macs": 494e6, "kind": "decoder",
                     "note": "Qwen2.5-0.5B decode, 494M params, 2.52 GB DLC"},
    "llama1b_dec32": {"dir": "llama", "dlc": "llama_m4d_fp32.dlc", "list": "l_ll.txt",
                      "macs": 1236e6, "kind": "decoder",
                      "note": "Llama-3.2-1B decode, 1.24B params, 6.00 GB DLC"},
    # Diffusion: three stages of one pipeline, built by scripts/export_diffusion.py.
    # The stage boundary is a 64 KB latent -- cheap enough that a genuine
    # multi-engine pipeline pays, which the LLM's 4.7 MB KV handoff never allowed.
    # Every denoising step has identical shapes, so no bucketing is needed.
    # Takes token EMBEDDINGS, not ids: the converter folds integer inputs away
    # (discovery 77), so the 77-row lookup happens on the host and the 12
    # transformer layers -- all of the arithmetic -- run here.
    "sd_text_encoder": {"dir": "sd15", "dlc": "te.dlc", "list": "l_te.txt",
                        "macs": 6.7e9, "kind": "clip",
                        "note": "CLIP text encoder, 85M params on device, once per image"},
    "sd_unet": {"dir": "sd15", "dlc": "unet.dlc", "list": "l_unet.txt",
                "macs": 68e9, "kind": "unet",
                "note": "distilled SD UNet, 323M params, ONCE PER DENOISING STEP"},
    "sd_vae_decoder": {"dir": "sd15", "dlc": "vae.dlc", "list": "l_vae.txt",
                       "macs": 1.3e12, "kind": "vae",
                       "note": "VAE decoder, 49M params, once per image; the most "
                               "engine-sensitive stage on this board (8x spread)"},
}
# QNN backend library per engine. The DSP has a QNN backend too, but our device
# only carries the CPU/GPU/HTP ones -- provisioning pushes those three.
_QNN_BACKEND = {"cpu": "libQnnCpu.so", "gpu": "libQnnGpu.so", "npu": "libQnnHtp.so"}


# ─────────────────────────────────────────────────────────────────────────────
# CPU hardware counters (simpleperf / ARMv9 PMU on Oryon-2)
# ─────────────────────────────────────────────────────────────────────────────
#
# Every event below was probed on this silicon against a
# memory-streaming workload on BOTH clusters and returned a real, non-zero
# count. 12 hardware counters are available per core, so up to 12 events can be
# collected simultaneously without time-multiplexing.
#
# The catalogue is a whitelist of what WORKS, not of what exists: `simpleperf
# list raw` advertises 742 events and none of them error out. The ones that are
# not implemented return a silent, confident ZERO — see UNSUPPORTED below.
#
# Re-verified on SM8850P (board 9d261d4e) 2026-09-08: every catalogued name was
# run against a memory-heavy workload in batches. All 92 below returned
# non-zero; all 43 in UNSUPPORTED returned exactly zero -- an exact set match
# with the SM8750 catalogue, so no edits were needed. Note this checks that a
# counter COUNTS, not that its value is correct.

COUNTER_GROUPS: dict[str, dict[str, str]] = {
    "core": {
        "cpu-cycles":               "core clock cycles",
        "instructions":             "retired instructions",
        "bus-cycles":               "bus clock cycles",
    },
    "branch": {
        "branch-instructions":      "retired branches",
        "branch-misses":            "mispredicted branches",
        "raw-br-pred":              "branches that reached the predictor",
        "raw-br-mis-pred":          "predictor misses (speculative)",
        "raw-br-retired":           "retired branches (raw encoding)",
        "raw-br-mis-pred-retired":  "mispredicts that actually retired",
        "raw-br-immed-spec":        "direct/immediate branches",
        "raw-br-indirect-spec":     "indirect branches (the expensive kind)",
        "raw-br-return-spec":       "returns (exercises the return stack)",
    },
    "cache": {
        "raw-l1d-cache":            "L1 data cache accesses",
        "raw-l1d-cache-refill":     "L1 data cache refills (misses)",
        "raw-l1d-cache-rd":         "L1D read accesses",
        "raw-l1d-cache-wr":         "L1D write accesses",
        "raw-l1d-cache-wb":         "L1D writebacks (dirty evictions)",
        "raw-l1d-cache-lmiss-rd":   "L1D long-latency read misses",
        "raw-l1d-cache-miss":       "L1D misses",
        "raw-l1i-cache":            "L1 instruction cache accesses",
        "raw-l1i-cache-refill":     "L1I refills (misses)",
        "raw-l1i-cache-lmiss":      "L1I long-latency misses",
        "raw-l2d-cache":            "L2 accesses (L2 is shared 0-5 / 6-7)",
        "raw-l2d-cache-refill":     "L2 refills (misses -> system cache/DRAM)",
        "raw-l2d-cache-rd":         "L2 read accesses",
        "raw-l2d-cache-wr":         "L2 write accesses",
        "raw-l2d-cache-wb":         "L2 writebacks",
        "raw-l2d-cache-allocate":   "L2 allocations without a refill",
        "raw-l2d-cache-lmiss-rd":   "L2 long-latency read misses",
        "raw-l2d-cache-miss":       "L2 misses",
        "raw-l1d-cache-refill-inner": "L1D refills served from inner (on-cluster)",
        "raw-l1d-cache-refill-outer": "L1D refills served from outer (off-cluster)",
    },
    # The hardware prefetcher is visible on this silicon (Lecture 19 territory).
    # hwprf counts accesses the PREFETCHER generated; prfm counts software PRFM
    # instructions. A sequential stream drives hwprf hard; a pointer chase does
    # not — that contrast is the whole experiment.
    "prefetch": {
        "raw-l1d-cache-hwprf":      "L1D accesses generated by the HW prefetcher",
        "raw-l2d-cache-hwprf":      "L2 accesses generated by the HW prefetcher",
        "raw-l1i-cache-hwprf":      "L1I accesses generated by the HW prefetcher",
        "raw-l1d-cache-prfm":       "L1D accesses from software PRFM instructions",
        "raw-l2d-cache-prfm":       "L2 accesses from software PRFM instructions",
        "raw-l1d-cache-refill-prfm": "L1D refills caused by software PRFM",
        "raw-l2d-cache-refill-prfm": "L2 refills caused by software PRFM",
    },
    "tlb": {
        "raw-l1d-tlb":              "L1 data TLB accesses",
        "raw-l1d-tlb-refill":       "L1 dTLB refills (misses)",
        "raw-l1d-tlb-rd":           "L1 dTLB read accesses",
        "raw-l1d-tlb-wr":           "L1 dTLB write accesses",
        "raw-l1d-tlb-refill-rd":    "L1 dTLB read refills",
        "raw-l1d-tlb-refill-wr":    "L1 dTLB write refills",
        "raw-l1i-tlb":              "L1 instruction TLB accesses",
        "raw-l1i-tlb-refill":       "L1 iTLB refills",
        "raw-l2d-tlb":              "L2 unified TLB accesses",
        "raw-l2d-tlb-refill":       "L2 TLB refills (-> hardware page walk)",
        "raw-dtlb-walk":            "data page-table walks",
        "raw-itlb-walk":            "instruction page-table walks",
        "raw-dtlb-walk-large":      "walks that resolved to a LARGE page",
        "raw-dtlb-walk-small":      "walks that resolved to a small (4K) page",
        "raw-dtlb-walk-rw":         "walks from a read or write (not prefetch)",
        "raw-dtlb-walk-prfm":       "walks caused by software PRFM",
        "raw-dtlb-step":            "data page-table walk steps (depth of walk)",
        "raw-itlb-step":            "instruction page-table walk steps",
    },
    "stalls": {
        "stalled-cycles-frontend":  "cycles with no op issued, frontend's fault",
        "stalled-cycles-backend":   "cycles with no op issued, backend's fault",
        "raw-stall":                "cycles with no op issued at all",
        "raw-stall-backend":        "backend stall cycles (raw encoding)",
        "raw-stall-frontend":       "frontend stall cycles (raw encoding)",
        "raw-stall-slot":           "issue SLOTS lost (finer than cycles)",
        "raw-stall-slot-backend":   "issue slots lost to the backend",
        "raw-stall-slot-frontend":  "issue slots lost to the frontend",
        "raw-stall-backend-mem":    "backend stalls attributed to memory",
    },
    "mix": {
        "raw-inst-retired":         "retired instructions (raw encoding)",
        "raw-inst-spec":            "speculatively executed instructions",
        "raw-op-retired":           "retired micro-ops",
        "raw-op-spec":              "speculatively executed micro-ops",
        "raw-ld-spec":              "loads executed",
        "raw-st-spec":              "stores executed",
        "raw-dp-spec":              "integer data-processing ops",
        "raw-vfp-spec":             "FP/SIMD ops (NEON counts HERE, not ase-*)",
    },
    "memory": {
        "raw-mem-access":           "memory accesses issued",
        "raw-mem-access-rd":        "memory reads",
        "raw-mem-access-wr":        "memory writes",
        "raw-bus-access":           "bus (interconnect) accesses",
    },
    "exceptions": {
        "raw-exc-taken":            "exceptions taken",
        "raw-exc-return":           "exception returns",
    },
    # simpleperf's hw-cache aliases. Same silicon events under friendlier names.
    "aliases": {
        "cache-references":         "alias: L1D accesses",
        "cache-misses":             "alias: last-level-reachable misses",
        "branch-loads":             "alias: branches",
        "branch-load-misses":       "alias: branch mispredicts",
        "L1-dcache-loads":          "alias: raw-l1d-cache",
        "L1-dcache-load-misses":    "alias: raw-l1d-cache-refill",
        "L1-icache-loads":          "alias: raw-l1i-cache",
        "L1-icache-load-misses":    "alias: raw-l1i-cache-refill",
        "dTLB-loads":               "alias: raw-l1d-tlb",
        "dTLB-load-misses":         "alias: raw-l1d-tlb-refill",
        "iTLB-loads":               "alias: raw-l1i-tlb",
        "iTLB-load-misses":         "alias: raw-l1i-tlb-refill",
    },
}

CPU_COUNTERS: dict[str, str] = {e: d for g in COUNTER_GROUPS.values() for e, d in g.items()}

# Events `simpleperf list` advertises that this silicon does NOT implement.
# They do not error — they return 0. Blocked so nobody builds an argument on a
# counter that was never counting.
UNSUPPORTED_COUNTERS: dict[str, str] = {
    **{e: "no L3/system-cache PMU: the SLC is invisible to the core PMU on "
          "this SoC (sysfs exposes only L1D/L1I/L2)"
       for e in ("raw-l3d-cache", "raw-l3d-cache-refill", "raw-l3d-cache-rd",
                 "raw-l3d-cache-allocate", "raw-l3d-cache-hit", "raw-l3d-cache-miss",
                 "raw-ll-cache", "raw-ll-cache-miss", "raw-ll-cache-rd",
                 "raw-ll-cache-miss-rd")},
    **{e: "per-structure stall attribution not implemented; use "
          "raw-stall-backend-mem instead"
       for e in ("raw-stall-backend-l1d", "raw-stall-backend-l2d",
                 "raw-stall-backend-tlb", "raw-stall-frontend-l1i",
                 "raw-stall-frontend-l2i", "raw-stall-frontend-tlb")},
    **{e: "cache HIT counters are not implemented; derive hits as "
          "(accesses - refills)"
       for e in ("raw-l1d-cache-hit", "raw-l2d-cache-hit", "raw-l1d-cache-allocate",
                 "raw-l1d-cache-prf")},
    "raw-ase-inst-spec": "reads 0 even on NEON code; use raw-vfp-spec for SIMD",
    "raw-l2i-tlb": "not implemented",
    "raw-l2i-tlb-refill": "not implemented",
    "raw-cid-write-retired": "not implemented",
    # Prefetch attribution: the *-hwprf ACCESS counters work, but the refill and
    # hit-attribution variants do not. Count prefetcher traffic, not its hit rate.
    **{e: "prefetch refill/hit attribution not implemented; raw-l1d-cache-hwprf "
          "and raw-l2d-cache-hwprf (accesses) do work"
       for e in ("raw-l1d-cache-refill-hwprf", "raw-l2d-cache-refill-hwprf",
                 "raw-l1i-cache-refill-hwprf", "raw-l1d-cache-refill-prf",
                 "raw-l2d-cache-refill-prf", "raw-l2d-cache-prf",
                 "raw-l1i-cache-prf", "raw-l1d-cache-hit-rd-fhwprf",
                 "raw-l1d-cache-hit-rd-fprf", "raw-l1d-cache-hit-rd-fprfm",
                 "raw-l1d-lfb-hit-rd", "raw-l1d-lfb-hit-rd-fhwprf",
                 "raw-l2d-cache-hit-rd-fhwprf", "raw-l2d-lfb-hit-rd-fhwprf")},
    **{e: "not implemented" for e in ("raw-ldst-align-lat", "raw-ld-align-lat",
                                      "raw-st-align-lat", "raw-dtlb-walk-block",
                                      "raw-dtlb-walk-page")},
}

# Named bundles, each <= 12 events so they never time-multiplex.
COUNTER_PRESETS: dict[str, list[str]] = {
    "basic":  ["cpu-cycles", "instructions"],
    "branch": ["cpu-cycles", "instructions", "branch-instructions", "branch-misses",
               "raw-br-immed-spec", "raw-br-indirect-spec", "raw-br-return-spec"],
    "cache":  ["cpu-cycles", "instructions", "raw-l1d-cache", "raw-l1d-cache-refill",
               "raw-l1d-cache-wb", "raw-l1i-cache", "raw-l1i-cache-refill",
               "raw-l2d-cache", "raw-l2d-cache-refill", "raw-l2d-cache-wb"],
    "tlb":    ["cpu-cycles", "instructions", "raw-l1d-tlb", "raw-l1d-tlb-refill",
               "raw-l1i-tlb", "raw-l1i-tlb-refill", "raw-l2d-tlb",
               "raw-l2d-tlb-refill", "raw-dtlb-walk", "raw-itlb-walk"],
    "stalls": ["cpu-cycles", "instructions", "stalled-cycles-frontend",
               "stalled-cycles-backend", "raw-stall-slot", "raw-stall-slot-backend",
               "raw-stall-slot-frontend", "raw-stall-backend-mem"],
    "mix":    ["cpu-cycles", "instructions", "raw-inst-spec", "raw-op-retired",
               "raw-op-spec", "raw-ld-spec", "raw-st-spec", "raw-dp-spec",
               "raw-vfp-spec"],
    "memory": ["cpu-cycles", "instructions", "raw-mem-access", "raw-mem-access-rd",
               "raw-mem-access-wr", "raw-bus-access", "raw-l1d-cache-refill",
               "raw-l2d-cache-refill", "raw-stall-backend-mem"],
    "prefetch": ["cpu-cycles", "instructions", "raw-l1d-cache", "raw-l1d-cache-refill",
                 "raw-l1d-cache-hwprf", "raw-l2d-cache", "raw-l2d-cache-refill",
                 "raw-l2d-cache-hwprf", "raw-stall-backend-mem"],
}

MAX_HW_COUNTERS = 12   # per core on Oryon-2; beyond this simpleperf multiplexes


def resolve_counters(spec) -> list[str]:
    """None | preset name | event | list of either -> a flat, deduped event list.

    Raises on events this silicon does not implement, naming the reason.
    """
    if spec is None:
        return []
    items = [spec] if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for item in items:
        for ev in COUNTER_PRESETS.get(item, [item]):
            if ev in UNSUPPORTED_COUNTERS:
                raise ValueError(
                    f"counter {ev!r} reads a constant 0 on this board — "
                    f"{UNSUPPORTED_COUNTERS[ev]}. It would not error on device, "
                    f"it would silently return 0.")
            if ev not in out:
                out.append(ev)
    return out


# Derived metrics: (name, numerator candidates, denominator candidates, scale).
# Computed only when both sides are present; first candidate found wins.
_DERIVED: tuple[tuple[str, tuple[str, ...], tuple[str, ...], float], ...] = (
    ("ipc",                ("instructions",), ("cpu-cycles",), 1.0),
    ("cpi",                ("cpu-cycles",), ("instructions",), 1.0),
    ("branch_miss_pct",    ("branch-misses", "raw-br-mis-pred", "branch-load-misses"),
                           ("branch-instructions", "raw-br-pred", "branch-loads"), 100.0),
    ("l1d_miss_pct",       ("raw-l1d-cache-refill", "L1-dcache-load-misses"),
                           ("raw-l1d-cache", "L1-dcache-loads"), 100.0),
    ("l1i_miss_pct",       ("raw-l1i-cache-refill", "L1-icache-load-misses"),
                           ("raw-l1i-cache", "L1-icache-loads"), 100.0),
    ("l2d_miss_pct",       ("raw-l2d-cache-refill",), ("raw-l2d-cache",), 100.0),
    ("l1dtlb_miss_pct",    ("raw-l1d-tlb-refill", "dTLB-load-misses"),
                           ("raw-l1d-tlb", "dTLB-loads"), 100.0),
    ("l2dtlb_miss_pct",    ("raw-l2d-tlb-refill",), ("raw-l2d-tlb",), 100.0),
    ("pagewalks_per_kinst", ("raw-dtlb-walk",), ("instructions",), 1000.0),
    ("frontend_stall_pct", ("stalled-cycles-frontend", "raw-stall-frontend"),
                           ("cpu-cycles",), 100.0),
    ("backend_stall_pct",  ("stalled-cycles-backend", "raw-stall-backend"),
                           ("cpu-cycles",), 100.0),
    ("mem_stall_pct",      ("raw-stall-backend-mem",), ("cpu-cycles",), 100.0),
    ("slot_backend_pct",   ("raw-stall-slot-backend",), ("raw-stall-slot",), 100.0),
    ("l1d_wb_per_kinst",   ("raw-l1d-cache-wb",), ("instructions",), 1000.0),
    # Share of cache traffic the hardware prefetcher generated, not the program.
    ("l1d_hwprf_pct",      ("raw-l1d-cache-hwprf",), ("raw-l1d-cache",), 100.0),
    ("l2d_hwprf_pct",      ("raw-l2d-cache-hwprf",), ("raw-l2d-cache",), 100.0),
)


def derive(counters: dict[str, float]) -> dict[str, float]:
    """Ratios the raw counts imply. Only what the present events support."""
    def pick(names: tuple[str, ...]) -> float | None:
        for n in names:
            if counters.get(n):
                return float(counters[n])
        return None

    out: dict[str, float] = {}
    for name, nums, dens, scale in _DERIVED:
        n, d = pick(nums), pick(dens)
        if n is not None and d:
            out[name] = n / d * scale

    # Speculation waste needs both a retired and a speculative count.
    retired = counters.get("raw-inst-retired") or counters.get("instructions")
    spec = counters.get("raw-inst-spec")
    if retired and spec and spec > 0:
        out["spec_waste_pct"] = (1.0 - float(retired) / float(spec)) * 100.0
    return out


def list_counters() -> str:
    """The full catalogue: presets, then every verified event by group."""
    lines = [f"PRESETS ({len(COUNTER_PRESETS)}) - pass as counters='name', or --counters name", ""]
    for name, events in COUNTER_PRESETS.items():
        lines.append(f"  {name:<8} {len(events):>2} events: {', '.join(events)}")
    lines += ["", f"EVENTS ({len(CPU_COUNTERS)} verified on SM8850P Oryon-2, "
                  f"max {MAX_HW_COUNTERS} at once)", ""]
    for group, events in COUNTER_GROUPS.items():
        lines.append(f"  [{group}]")
        w = max(len(e) for e in events)
        lines += [f"    {e:<{w}}  {d}" for e, d in events.items()]
        lines.append("")
    lines.append(f"BLOCKED ({len(UNSUPPORTED_COUNTERS)}) - exist in `simpleperf list`, "
                 f"return a silent 0 on this silicon")
    lines.append("")
    w = max(len(e) for e in UNSUPPORTED_COUNTERS)
    lines += [f"  {e:<{w}}  {r}" for e, r in UNSUPPORTED_COUNTERS.items()]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Backends and workloads
# ─────────────────────────────────────────────────────────────────────────────

class Backend(Enum):
    """The four engines. Iterating `Backend` gives all four, which is how the
    one-workload-everywhere comparisons are written."""
    CPU = "cpu"   # Oryon-2, taskset affinity, wall_us
    GPU = "gpu"   # Adreno 840, OpenCL, duration_ns
    DSP = "dsp"   # Hexagon cDSP v81, FastRPC + HVX, dsp_us
    NPU = "npu"   # Hexagon HTP / QNN, execute_us + finalize_us

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class KernelEntry:
    """One (workload, backend): device subdir, variant->binary, always-on flags."""
    subdir: str
    variants: dict[str, str]
    default: str
    ldpath: str = "/vendor/lib64"
    variant_is_flag: bool = False       # DSP selects the variant with --variant
    defaults: dict[str, object] = field(default_factory=dict)
    # variant -> the only --size it accepts. Register-blocked GPU kernels are
    # compiled for one shape and assert on anything else; catching it here beats
    # a device-side failure halfway through a sweep.
    fixed_size: dict[str, int] = field(default_factory=dict)
    # Rename a parameter on the way to the CLI, where a kernel spells it
    # differently. Kernels ignore unknown flags silently, so a mismatch here
    # does not error -- it quietly measures the kernel's default instead.
    param_map: dict[str, str] = field(default_factory=dict)
    # Which VARIANTS accept `--threads N`. Threading is per-binary, not per-entry:
    # conv_neon is threaded but conv_scalar sits in the same entry and hard-errors
    # on the flag. Empty means the whole entry is single-threaded.
    threaded_variants: frozenset[str] = frozenset()
    # This kernel prints flop=/bytes= covering ALL iterations while its timing
    # field measures ONE. Only the DSP does this: `dsp_us` comes from inside the
    # DSP and is per-iteration, but the harness sums the work. Without the flag
    # throughput reads high by exactly the iteration count.
    totals_with_per_iteration_time: bool = False


@dataclass(frozen=True)
class Workload:
    """A workload family. flop_fn/bytes_fn derive throughput for kernels that
    don't print flop=/bytes= themselves; printed values always win."""
    name: str
    kernels: dict[Backend, KernelEntry]
    flop_fn: Callable[[dict], float] | None = None
    bytes_fn: Callable[[dict], float] | None = None


# The registry: single source of truth for the on-device layout. Paths and flags
# match scripts/*_sweep.ps1, so a single run() reproduces a published sweep point.

def _matmul_flop(p: dict) -> float:
    n = int(p.get("size", 512))
    return 2.0 * n * n * n

def _conv_flop(p: dict) -> float:
    H, W = int(p.get("H", 256)), int(p.get("W", 256))
    cin, cout, k = int(p.get("cin", 8)), int(p.get("cout", 8)), int(p.get("k", 3))
    return 2.0 * H * W * cout * k * k * cin

def _nbody_flop(p: dict) -> float:
    # 20 flop per pairwise interaction is the N-body literature convention.
    # Both kernels round --bodies down to a multiple of 64, so mirror that here
    # or the reported GFLOP/s is computed from a body count nobody simulated.
    n = int(p.get("bodies", p.get("size", 4096)))
    n = max(64, n) & ~63
    return 20.0 * n * n
def _decode_flop(p: dict) -> float:
    s = int(p.get("tokens", 1))
    k = int(p.get("k", 1024))
    n = int(p.get("n", 1024))
    return 2.0 * s * k * n

def _decode_bytes(p: dict) -> float:
    """Weight traffic only. Weights dominate while S << K, which is the regime
    the workload exists to show; the accounting stops being right for large S."""
    return float(int(p.get("k", 1024)) * int(p.get("n", 1024)) * 4)


WORKLOADS: dict[str, Workload] = {
    "matmul": Workload("matmul", flop_fn=_matmul_flop, kernels={
        # Eigen beats the hand-tuned NEON kernel on both clusters (disc 25/26) —
        # the CPU analogue of "tiled is slower than naive" on the GPU.
        # iterations=10 by default. cpu_parallel() spawns and joins INSIDE the
        # timed region, so a single-iteration run charges the whole thread
        # spawn to a short measurement -- and the shorter the region gets, the
        # worse it looks, which is exactly what happens as threads are added.
        # Measured 2026-09-09 on Gold: matmul N=512 scales 2.55x at one
        # iteration and 4.11x at thirty. Every other CPU kernel already loops
        # (saxpy/reduction 200, convolution 10), so matmul was the only one
        # reporting spawn overhead as poor parallel efficiency.
        Backend.CPU: KernelEntry("", {"neon": "matmul_neon_handtuned",
                                       "eigen": "matmul_eigen_baseline"}, "neon",
                                 defaults={"iterations": 10},
                                 threaded_variants=frozenset({"neon"})),
        # All five share one CLI (--size/--seed) and one output contract
        # (duration_ns, fingerprint, status:); provisioning pushes every binary
        # under kernels/matmul/gpu/, so all five are on-device.
        #   naive -> tiled  is 2.54x SLOWER  (discovery 32) — see examples/04
        #   naive -> fp16   is ~1.0x         (discovery 33) — see examples/05
        #   naive -> best   is ~16x faster   (discovery 51)
        # fp32_4x16u4 and fp16_best share the 4x16 tile shape, so that pair
        # isolates precision alone; 'best' is a different (3x16) shape.
        # More variants exist unregistered under kernels/matmul/gpu/.
        # The register-tile ladder: vec4 -> 3x16 -> 4x16 -> 8x16 sweeps tile
        # shape, which is the knob that actually moved GFLOP/s on this GPU.
        Backend.GPU: KernelEntry("gpu", {"naive": "matmul_opencl_fp32",
                                          "tiled": "matmul_opencl_fp32_tiled",
                                          "fp16": "matmul_opencl_fp16",
                                          "vec4": "matmul_opencl_fp32_vec4",
                                          "fp32_3x16u4": "matmul_opencl_fp32_3x16u4",
                                          "fp32_3x16u4_db": "matmul_opencl_fp32_3x16u4_db",
                                          "fp32_4x16u4": "matmul_opencl_fp32_4x16u4",
                                          "fp32_8x16u4": "matmul_opencl_fp32_8x16u4",
                                          "anysize": "matmul_opencl_fp32_3x16_anysize",
                                          "fp16_best": "matmul_opencl_fp16_4x16u4",
                                          "best": "matmul_opencl_fp32_3x16u4_sg_half"},
                                 "naive", ldpath=".",
                                 fixed_size={"best": 512, "fp16_best": 512,
                                             "fp32_3x16u4": 512, "fp32_3x16u4_db": 512,
                                             "fp32_4x16u4": 512, "fp32_8x16u4": 512}),
        # Direct clients, not matmul_dsp_sweep. The sweep client pins DSP power
        # first and fail-closes if that is refused -- and HAP power voting is
        # rejected (0x80000414) from the unsigned PD we run in, with no testsig
        # installed. Compute over FastRPC works fine; only the power vote does
        # not. Consequence: DSP numbers here are unpinned, so expect higher
        # variance than the other backends.
        Backend.DSP: KernelEntry("dsp", {"scalar": "matmul_dsp_scalar",
                                          "hvx": "matmul_dsp_hvx",
                                          "vtcm": "matmul_dsp_vtcm"},
                                 "hvx"),
        # `spike` is the Task 12 Part B harness: same fp16 graph, plus a shared-
        # buffer path. Baseline = no extra flag; shared = -p memshare=./libmemshare.so
        # (which implies --shared). Its summary shows shared_impl so a run that
        # silently fell back to copies is visible.
        Backend.NPU: KernelEntry("npu", {"fp16": "matmul_qnn_fp16", "int8": "matmul_qnn_int8",
                                         "spike": "matmul_qnn_shared_spike"},
                                 "fp16", defaults={"perf_mode": "sustained", "reps": 16},
                                 # client takes --n, and spells repetitions --reps.
                                 # Without the second mapping this was the only
                                 # cell of the 4x4 matrix where `-p iterations N`
                                 # hard-failed, so a loop over all four backends
                                 # died on one engine.
                                 param_map={"size": "n", "iterations": "reps"}),
    }),
    "saxpy": Workload("saxpy", kernels={
        Backend.CPU: KernelEntry("saxpy", {"scalar": "saxpy_scalar", "neon": "saxpy_neon"},
                                 "neon", defaults={"iterations": 200},
                                 threaded_variants=frozenset({"scalar", "neon"})),
        Backend.GPU: KernelEntry("saxpy", {"fp32": "saxpy_opencl_fp32"}, "fp32"),
        Backend.DSP: KernelEntry("dsp", {"scalar": "saxpy_dsp", "hvx": "saxpy_dsp"},
                                 "hvx", variant_is_flag=True,
                                 totals_with_per_iteration_time=True, defaults={"power": "lab", "iterations": 50}),
        Backend.NPU: KernelEntry("npu", {"fp16": "saxpy_qnn_fp16"}, "fp16",
                                 defaults={"perf_mode": "sustained", "reps": 5}),
    }),
    "reduction": Workload("reduction", kernels={
        Backend.CPU: KernelEntry("reduction", {"scalar": "reduction_scalar", "neon": "reduction_neon"},
                                 "neon", defaults={"iterations": 200},
                                 threaded_variants=frozenset({"scalar", "neon"})),
        Backend.GPU: KernelEntry("reduction", {"fp32": "reduction_opencl_fp32"}, "fp32"),
        Backend.DSP: KernelEntry("dsp", {"scalar": "reduction_dsp", "hvx": "reduction_dsp"},
                                 "hvx", variant_is_flag=True,
                                 totals_with_per_iteration_time=True,
                                 defaults={"power": "lab", "iterations": 50}),
        Backend.NPU: KernelEntry("npu", {"fp16": "reduction_qnn_fp16"}, "fp16",
                                 defaults={"perf_mode": "sustained", "reps": 5}),
    }),
    "convolution": Workload("convolution", flop_fn=_conv_flop, kernels={
        # conv_neon/conv_int8_neon are the only threaded CPU kernels on the device
        # today; conv_scalar rejects --threads.
        Backend.CPU: KernelEntry("convolution",
                                 {"scalar": "conv_scalar", "neon": "conv_neon",
                                  "int8": "conv_int8_neon"},
                                 "neon", defaults={"iterations": 10},
                                 threaded_variants=frozenset({"neon", "int8"})),
        Backend.GPU: KernelEntry("convolution", {"fp32": "conv_opencl_fp32"}, "fp32"),
        # int8 is its own binary (int8 in, int32 out, HVX vrmpy). It ignores the
        # --variant flag the other two need, prints ops=/checksum= where they
        # print flop=/fingerprint=, and like them reports totals over all
        # iterations against a per-iteration dsp_us; _parse maps both names.
        Backend.DSP: KernelEntry("dsp", {"scalar": "conv_dsp", "hvx": "conv_dsp",
                                         "int8": "conv_dsp_int8"},
                                 "hvx", variant_is_flag=True,
                                 totals_with_per_iteration_time=True,
                                 defaults={"power": "lab", "iterations": 2}),
        Backend.NPU: KernelEntry("npu", {"fp16": "conv_qnn_fp16"}, "fp16",
                                 defaults={"perf_mode": "sustained", "reps": 5}),
    }),
    # Compute-bound anchor of the roofline. saxpy/reduction sit at the
    # memory-bound end; matmul/convolution are dense tensor ops the NPU wins
    # outright. N-body is O(N^2) arithmetic on O(N) data — AI = 1.25*N
    # flop/byte, growing with the problem — and QNN cannot express it, so this
    # is the one workload where the GPU competes on equal terms. CPU and GPU
    # only: there is no DSP or NPU variant, by design rather than omission.
    "nbody": Workload("nbody", flop_fn=_nbody_flop, kernels={
        Backend.CPU: KernelEntry("nbody", {"neon": "nbody_neon"}, "neon",
                                 defaults={"iterations": 4},
                                 threaded_variants=frozenset({"neon"})),
        # One binary, two kernels: --variant naive|tiled. Both must exist —
        # comparing a naive GPU kernel against another backend measures the
        # code, not the silicon (see the GPU section, task P5).
        Backend.GPU: KernelEntry("nbody", {"naive": "nbody_opencl_fp32",
                                           "tiled": "nbody_opencl_fp32"},
                                 "tiled", variant_is_flag=True,
                                 defaults={"iterations": 5}),
    }),
    # bp_stress / rob_stress are NOT workloads: students write them for Task 5
    # (kernels/lab3/) and run them with `profiler.py custom --binary <name>`.
    # Adreno memory-subsystem lab. NB: run at SHELL uid — `adb root` makes
    # clGetPlatformIDs return -1001 (no platform), because the OpenCL ICD is not
    # reachable from the su domain. Every other GPU kernel here has the same
    # constraint, and CPU governor control needs the opposite. Do not hold both.
    "gpu_memlab": Workload("gpu_memlab", kernels={
        Backend.GPU: KernelEntry("memlab", {"memlab": "gpu_memlab"}, "memlab",
                                 defaults={"mode": "coalesce"}),
    }),
    # Weight-stationary GEMM with the token count exposed: C[S x n] = A[S x k] * W[k x n].
    # `tokens` is the only knob, and it walks the workload across the roofline
    # ridge -- S=1 is LLM decode (every weight read once, DRAM-bound), large S is
    # prefill (weights reused across the batch, compute-bound). The ranking
    # inverts across it: at S=1 one CPU core beats the NPU; at S=256 the NPU wins
    # 29x (discovery 66).
    # Your own kernels. Nothing is registered here -- pass the device binary
    # name to custom(), so writing a kernel never means editing this file.
    # Contract: print ONE timing field (mean_us / wall_us / duration_ns /
    # dsp_us / execute_us_mean) -- that one is required, a run without a time is
    # reported as a failure. Optionally `flop:`, `bytes:`, `fingerprint:`, and
    # `status:` (only OK/SUCCESS/PASS/PASSED/DONE count as success; most shipped
    # kernels print none and rely on the exit code).
    # See docs/student/05-your-own-kernel.md.
    # Lab 3 Task 3's cache-hierarchy ruler. Registered so it goes through the
    # same wrapper as everything else -- counters (which need `su 0` on this
    # board), --repeat, --core. Driving it by hand through `custom` does not
    # work: `custom` resolves binaries under custom/, not cachelat/.
    "cachelat": Workload("cachelat", kernels={
        Backend.CPU: KernelEntry("cachelat", {"latency": "cache_latency"}, "latency",
                                 defaults={"accesses": 40000000, "pattern": "random"}),
    }),
    # Lab 3 Task 10 Part D: erosion/dilation on the DSP. One binary; the
    # implementation (student / qhl) is a parameter, not a variant, because both
    # live in the same skel. `residual` (pixels still differing from the clean
    # image) is the correctness check and is printed on the summary line.
    "morph": Workload("morph", kernels={
        Backend.DSP: KernelEntry("dsp", {"morph": "conv_morph"}, "morph"),
    }),
    # Lab 3 Task 11 Part B: stream a buffer through VTCM in blocks. One binary;
    # `impl` (ddr / naive / staged) picks the code path and `p` the passes per
    # block. The summary shows the copy/compute split and the exact checksum,
    # which must match across the three impls at a given p.
    "stage": Workload("stage", kernels={
        Backend.DSP: KernelEntry("dsp", {"stage": "conv_stage"}, "stage",
                                 defaults={"mb": 32}),
    }),
    "custom": Workload("custom", kernels={
        Backend.CPU: KernelEntry("custom", {"_": "_"}, "_"),
        Backend.GPU: KernelEntry("custom", {"_": "_"}, "_", ldpath="."),
        Backend.DSP: KernelEntry("custom", {"_": "_"}, "_"),
        Backend.NPU: KernelEntry("custom", {"_": "_"}, "_"),
    }),
    # Networks, not kernels: dispatched through qnn-net-run with a DLC. The
    # KernelEntry is a formality -- build_command handles this workload itself.
    # CPU/GPU/NPU only: QAIRT ships a QNN DSP backend but provisioning does not
    # push it, so listing DSP here would advertise a dispatch that cannot run.
    "model": Workload("model", kernels={b: KernelEntry("", {"qnn": "qnn-net-run"}, "qnn")
                                        for b in (Backend.CPU, Backend.GPU, Backend.NPU)}),
    "decode": Workload("decode", flop_fn=_decode_flop, bytes_fn=_decode_bytes, kernels={
        Backend.CPU: KernelEntry("decode", {"neon": "decode_neon"}, "neon"),
        # naive: one work item per output element. At S=1 the row>=S guard
        # masks 94% of the launch, leaving 1024 live items each running a
        # K-deep dependent mad chain -- 13% of this GPU's 78.5 GB/s roof.
        # gemv: split-K, float4 columns, two columns per item, local reduction.
        # 6x the naive kernel and 86% of the measured 74.4 GB/s stream roof once
        # the weights exceed cache. Default stays naive so the ladder is
        # visible; see disc 66.
        Backend.GPU: KernelEntry("decode", {"naive": "decode_opencl_fp32",
                                            "gemv": "decode_opencl_gemv"},
                                 "naive", ldpath=".", param_map={"n": "ncols"},
                                 defaults={"wg-x": 64, "wg-y": 8}),
        Backend.NPU: KernelEntry("npu", {"fp16": "decode_qnn_fp16"}, "fp16",
                                 defaults={"perf_mode": "sustained", "reps": 20},
                                 param_map={"n": "ncols"}),  # client spells it --ncols
    }),
}

# Per-cluster maximum clock, for spotting thermal throttling after the fact.
# Top of each cluster's cpufreq ladder, read from
# /sys/devices/system/cpu/cpufreq/policy{0,6}/scaling_available_frequencies
# on the SM8850P. These were the SM8750's values (3.5328 / 4.32) until
# 2026-09-10, which overstated clock_of_max_pct by 1.7-2.6 points and let a
# healthy Gold run report OVER 100% of maximum. The Prime ceiling is the
# ceiling is the top of the published ladder (4.3968), not cpuinfo_max_freq
# (4.608): 4608000 is settable and scaling_cur_freq will report it, but it
# is absent from scaling_available_frequencies, effective_ghz only reaches
# ~3.8-3.9 there, and under sustained load the governor settles to 4.34.
# 4.3968 is the fastest setpoint a student can actually select and hold.
CLUSTER_MAX_GHZ = {"gold": 3.6288, "prime": 4.3968}
CLOCK_FLAG_RATIO = 0.80  # below this fraction of cluster max, flag the run


# ─────────────────────────────────────────────────────────────────────────────
# Task — one dispatch
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    """A workload on a backend with parameters. params become --key value flags.

    `core` picks WHERE the work may run (taskset affinity); `threads` picks HOW
    MANY worker threads the kernel spawns. They are independent knobs and both
    are recorded — 6 threads on a 1-core mask is a legal, measurable experiment.
    """
    workload: str
    backend: Backend
    params: dict = field(default_factory=dict)
    variant: str | None = None
    name: str | None = None
    timeout: float = 180.0
    depends_on: list[str] = field(default_factory=list)
    core: object = None  # CPU affinity (int | list[int] | "prime"/"gold"/"all" | hex mask)
    threads: int | None = None      # -> --threads N, threaded variants only
    counters: object = None         # None | preset | event | list of either
    per_core: bool = False          # break counter totals down per CPU
    counter_cpus: object = None     # count per-CPU system-wide instead of per-process
    # Dispatch this device binary instead of looking one up in the registry.
    # Lets you run a kernel you just wrote without editing this file.
    binary: str | None = None

    def __post_init__(self) -> None:
        if self.workload not in WORKLOADS:
            raise ValueError(f"unknown workload {self.workload!r}; known: {list(WORKLOADS)}")
        if self.backend not in WORKLOADS[self.workload].kernels:
            have = ", ".join(sorted(str(b) for b in WORKLOADS[self.workload].kernels))
            raise ValueError(
                f"{self.workload} has no {self.backend} kernel; "
                f"it runs on: {have}")
        entry = WORKLOADS[self.workload].kernels[self.backend]
        if self.binary:
            if self.name is None:
                self.name = f"custom.{self.backend}.{self.binary}"
            resolve_counters(self.counters)   # validate now, resolve on access
            return
        # A mistyped variant used to surface as a bare KeyError from deep inside
        # build_command, with no list of what was valid. Variant names like
        # `fp32_3x16u4_db` get typed wrong often enough that this needs to be a
        # sentence, not a traceback.
        if self.variant is not None and self.variant not in entry.variants:
            near = [v for v in entry.variants
                    if v.startswith(self.variant[:3]) or self.variant in v]
            msg = (f"{self.workload}.{self.backend} has no variant "
                   f"{self.variant!r}; available: {', '.join(entry.variants)}"
                   f"  (default: {entry.default})")
            if near:
                msg += "\n  did you mean: " + ", ".join(near) + "?"
            raise ValueError(msg)
        # Fail here, not on the device three dispatches into a sweep.
        need = entry.fixed_size.get(self.variant or entry.default)
        got = self.params.get("size")
        if got is not None:
            try:
                got = int(got)          # accept "512" as well as 512
            except (TypeError, ValueError):
                pass
        if need is not None and got is not None and got != need:
            raise ValueError(
                f"variant {self.variant or entry.default!r} is compiled for size={need} "
                f"only, got size={got}. Use an any-size variant "
                f"(naive, tiled, vec4, fp16, anysize) to sweep sizes."
            )
        if self.threads is not None:
            if int(self.threads) < 1:
                raise ValueError(f"threads must be >= 1, got {self.threads}")
            if self.resolved_variant not in entry.threaded_variants:
                threaded = sorted(entry.threaded_variants)
                raise ValueError(
                    f"{self.workload}.{self.backend} variant "
                    f"{self.resolved_variant!r} does not accept --threads "
                    f"(the binary hard-errors on it). Threaded variants here: "
                    f"{threaded or 'none'}.")
        # Validate the selection here so a typo fails at construction rather
        # than three dispatches into a sweep; the resolved list itself is a
        # property, so assigning `task.counters` later actually takes effect.
        resolve_counters(self.counters)
        if self.name is None:
            self.name = f"{self.workload}.{self.backend}" + (f".{self.variant}" if self.variant else "")

    @property
    def counter_events(self) -> list:
        """Resolved event list for `counters`, recomputed on every access.

        This used to be cached in __post_init__, which meant

            t = reduction(Backend.CPU, size=N)
            t.counters = "stalls"        # silently did nothing

        dispatched with no counters at all and returned an empty `derived`
        dict, with no error anywhere. Building the Task with `counters=` always
        worked (the kernel helpers forward it, along with core/threads/per_core,
        via _task) -- but mutating one afterwards is an equally obvious thing to
        do, and silently doing nothing is the wrong answer to it. Now both work."""
        return resolve_counters(self.counters)

    @property
    def entry(self) -> KernelEntry:
        """The registry entry this Task dispatches: device dir, variants, flags."""
        return WORKLOADS[self.workload].kernels[self.backend]

    @property
    def resolved_variant(self) -> str:
        """The variant that will actually run -- `variant` if given, else the
        registry default. Worth checking: defaults are rarely the fastest."""
        return self.variant or self.entry.default

    @property
    def cores(self) -> list[int]:
        """The CPU ids the affinity mask actually permits."""
        mask = int(_affinity_mask(self.core), 16)
        return [i for i in range(8) if mask >> i & 1]

    @property
    def oversubscribed(self) -> bool:
        """More worker threads than cores they are allowed to run on."""
        return bool(self.threads and self.threads > len(self.cores))


# Constructors: Task-level kwargs are peeled off, every other kwarg is a kernel param.
def _task(workload: str, backend: Backend, params: dict) -> Task:
    name = params.pop("name", None)
    variant = params.pop("variant", None)
    depends_on = params.pop("depends_on", [])
    timeout = params.pop("timeout", 180.0)
    core = params.pop("core", None)
    threads = params.pop("threads", None)
    counters = params.pop("counters", None)
    per_core = params.pop("per_core", False)
    counter_cpus = params.pop("counter_cpus", None)
    return Task(workload, backend, params, variant, name, timeout, depends_on, core,
                threads, counters, per_core, counter_cpus)

def matmul(backend: Backend, **params) -> Task:
    """Dense square matmul. `size=N` (default 512); AI ~ N/6, so it is the
    compute-bound end of the workload set. Variants differ enormously: GPU
    `best` is ~16x `naive`, CPU `eigen` ~3x `neon`."""
    return _task("matmul", backend, params)

def saxpy(backend: Backend, **params) -> Task:
    """`y = a*x + y` over `size` floats (default 1M). Arithmetic intensity 0.17 --
    the memory-bound end, and the workload where the CPU beats the NPU 50x."""
    return _task("saxpy", backend, params)

def reduction(backend: Backend, **params) -> Task:
    """Sum of `size` floats (default 4M). Memory-bound like saxpy, but the NPU has
    a native ReduceSum, so it compiles in ms rather than seconds -- the pair is a
    controlled experiment in op support."""
    return _task("reduction", backend, params)

def convolution(backend: Backend, **params) -> Task:
    """2D convolution, `cin`/`cout`/`H`/`W`/`k`. AI ~18 at C=8, which is enough to
    feed the NPU's tensor cores: it wins by ~68x over one CPU cluster here."""
    return _task("convolution", backend, params)

def nbody(backend: Backend, **params) -> Task:
    """Direct-summation N-body. CPU and GPU only — the compute-bound anchor."""
    return _task("nbody", backend, params)
def custom(backend: Backend, binary: str, **params) -> Task:
    """Run your own kernel: custom(Backend.CPU, "my_kernel", size=1024).

    `binary` is a filename under /data/local/tmp/profiler-kernels/custom/.
    Build and push it with `python scripts/new_kernel.py --push <name>`."""
    params = dict(params)
    return Task("custom", backend, params=params, binary=binary,
                name=params.pop("name", None), timeout=params.pop("timeout", 180.0),
                depends_on=params.pop("depends_on", []), core=params.pop("core", None))


def model(backend: Backend, name: str, **params) -> Task:
    """Run a converted network on one engine.

    The same DLC runs on any backend, so comparing engines is a one-word change:

        p.run(model(Backend.NPU, "minilm"))
        p.run(model(Backend.GPU, "minilm"))

    `reps` sets how many inferences one dispatch performs. Timing a single
    inference means running twice and subtracting, because graph preparation is
    paid once and differs enormously by backend -- see `Profiler.time_model`.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    if str(backend) not in _QNN_BACKEND:
        raise ValueError(f"{backend} has no QNN backend on this device; "
                         f"use {sorted(_QNN_BACKEND)}")
    params = dict(params)
    t = Task("model", backend, params=params,
             name=params.pop("name", None) or f"model.{name}.{backend}")
    t.model = name
    return t


def decode(backend: Backend, **params) -> Task:
    """Weight-stationary GEMM; `tokens` walks it across the roofline ridge."""
    return _task("decode", backend, params)


# ─────────────────────────────────────────────────────────────────────────────
# Result — the parsed outcome of a dispatch
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    """Outcome of one Task. compute_us is backend-native compute time; host
    transport stays separate in `fields`. Every parsed key is in `fields`."""
    task: Task
    ok: bool
    compute_us: float | None = None
    wall_us: float | None = None
    compute_key: str | None = None    # which kernel field compute_us came from
    gflops: float | None = None
    gbps: float | None = None
    fingerprint: float | None = None
    fields: dict = field(default_factory=dict)   # every parsed key -> value
    raw: str = ""
    error: str | None = None
    counters: dict = field(default_factory=dict)      # event -> count
    per_core: dict = field(default_factory=dict)      # cpu id -> {event: count}
    derived: dict = field(default_factory=dict)       # ipc, miss rates, stall %
    multiplexed: bool = False   # > MAX_HW_COUNTERS events -> counts are estimates
    # Ran below CLOCK_FLAG_RATIO of the cluster's max clock. This is NOT
    # necessarily thermal throttling -- under the shipping `walt` governor a
    # low clock is normal DVFS. It means "this did not run at full speed", and
    # whether that was intended is the caller's to judge. Needs cpu-cycles in
    # the counter selection and an affinity mask inside one cluster.
    below_max_clock: bool = False
    # Thermal context, read AFTER the dispatch. `temp_c` is the hottest zone;
    # `temps` is every zone read, so a caller can see which part is hot -- a
    # 78 C core next to a 30 C GPU is a different situation from both at 78.
    # None when thermal reporting is off or the sensors were unreadable.
    temp_c: float | None = None
    temps: dict = field(default_factory=dict)
    too_hot: bool = False

    @property
    def iterations(self) -> int:
        """How many times the kernel repeated the work inside one dispatch.

        Matters because `compute_us` is the TOTAL across them: a 119 ms
        convolution at iterations=10 is ten 11.9 ms convolutions, not one slow
        one. `gflops` already accounts for it.
        """
        if self.compute_key not in _TOTAL_COMPUTE_FIELDS:
            return 1        # already per-dispatch; nothing to divide out
        for key in ("iterations", "reps", "inner-reps", "inner_reps"):
            if key in self.fields:
                try:
                    return max(1, int(self.fields[key]))
                except (TypeError, ValueError):
                    return 1
        return 1

    @property
    def compute_us_per_iter(self) -> float | None:
        """`compute_us` divided by the repeat count -- the per-dispatch latency."""
        if self.compute_us is None:
            return None
        return self.compute_us / self.iterations

    def summary(self) -> str:
        """One aligned line: name, time, throughput, and every extra field the
        kernel printed. `us/iter xN` means the kernel repeated the work N times
        and the time shown is per repetition."""
        if not self.ok:
            return f"[FAIL] {self.task.name}: {self.error}"
        bits = [f"{self.task.name:28s}"]
        if self.compute_us is not None:
            n = self.iterations
            if n > 1:
                # Show the per-dispatch number, because that is what reads as a
                # latency, and say the repeat count out loud so the total is
                # still recoverable. Without this, compute_us looks like one
                # dispatch and is silently n times too big.
                bits.append(f"{self.compute_us_per_iter:10.1f} us/iter x{n}")
            else:
                bits.append(f"{self.compute_us:10.1f} us")
        if self.gflops is not None:
            bits.append(f"{self.gflops:8.1f} GFLOP/s")
        if self.gbps is not None:
            bits.append(f"{self.gbps:6.1f} GB/s")
        # ns_per_access is cachelat's headline number and the ruler Task 3
        # measures the hierarchy with; without it the summary shows only GB/s,
        # which is the one metric a latency probe should not be read as.
        for extra in ("shared_impl", "copy_us", "compute_us", "copy_gbps", "checksum",
                      "residual", "ns_per_access", "execute_us_cov_pct", "finalize_us", "transfer_ns", "readback_ns"):
            if extra in self.fields:
                bits.append(f"{extra}={self.fields[extra]}")
        if self.fingerprint is not None and "checksum" not in self.fields:
            bits.append(f"fp={self.fingerprint:.4e}")
        if "ipc" in self.derived:
            bits.append(f"IPC={self.derived['ipc']:.2f}")
        if self.multiplexed:
            bits.append("[multiplexed]")
        if self.below_max_clock:
            bits.append(f"[clock {self.derived.get('clock_of_max_pct', 0):.0f}% of max]")
        # Only when hot: a temperature on every line would be noise, and the
        # number is always in `temp_c` for anyone who wants it.
        if self.too_hot and self.temp_c is not None:
            bits.append(f"[HOT {self.temp_c:.0f}C]")
        return "  ".join(bits)

    def counter_table(self) -> str:
        """Counts, then the ratios they imply. Empty when no counters ran."""
        if not self.counters:
            return "(no counters collected)"
        w = max(len(e) for e in self.counters)
        lines = [f"{e:<{w}}  {int(v):>16,}" for e, v in self.counters.items()]
        if self.per_core:
            lines.append("")
            events = list(next(iter(self.per_core.values())))
            lines.append("per-core:  cpu  " + "  ".join(f"{e:>18}" for e in events))
            for cpu in sorted(self.per_core):
                row = self.per_core[cpu]
                lines.append(f"{'':10} {cpu:>4}  "
                             + "  ".join(f"{int(row.get(e, 0)):>18,}" for e in events))
        if self.derived:
            lines.append("")
            dw = max(len(k) for k in self.derived)
            lines += [f"{k:<{dw}}  {v:>16.3f}" for k, v in self.derived.items()]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Result({self.task.name}, ok={self.ok}, compute_us={self.compute_us})"


# Priority order for "the compute time" across the kernel output formats.
_COMPUTE_FIELDS = ("execute_us_mean", "dsp_us", "mean_us", "duration_ns", "wall_us")
# Of those, only wall_us is a TOTAL across the repetitions the kernel was asked
# for. The rest are already per-dispatch (a mean, or one timed region), so they
# must never be divided by the repeat count -- doing so reported an NPU
# convolution as 32 us/iter when the per-inference time was 172 us.
_TOTAL_COMPUTE_FIELDS = ("wall_us",)


def _parse(stdout: str) -> dict:
    """Scrape every `key=value` token and `key: value` line into one flat dict."""
    fields: dict[str, object] = {}

    def store(k: str, v: str) -> None:
        v = v.strip()
        try:
            fields[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            fields[k] = v

    for line in stdout.splitlines():
        m = re.match(r"\s*([A-Za-z_]\w*)\s*:\s*([-\d.eE+]+)\s*$", line)  # "key: value"
        if m:
            store(m.group(1), m.group(2))
        m = re.match(r"\s*(shared_impl)\s*:\s*(\w+)\s*$", line)       # Task 12 spike: a word
        if m:
            fields[m.group(1)] = m.group(2)
        for tok in line.split():                                        # "key=value"
            if "=" in tok:
                k, _, v = tok.partition("=")
                if re.fullmatch(r"[A-Za-z_]\w*", k) and v:
                    store(k, v)
    return fields


def _parse_counters(
    stdout: str,
) -> tuple[dict[str, float], dict[int, dict[str, float]], float | None]:
    """Pull simpleperf's --csv block out of the kernel's own stdout.

    Rows are `count,event,rate,unit,` — or `cpu,count,event,rate,unit,` under
    --per-core. Neither shape collides with the kernel's `key=value` output, so
    both parsers can read the same stream.

    Also returns simpleperf's own `Total test time` in seconds. That is the
    window the COUNTS cover — the whole process, not just the kernel's
    self-timed region — and it is the only correct denominator for turning
    cycles into a clock rate.
    """
    totals: dict[str, float] = {}
    per_core: dict[int, dict[str, float]] = {}
    seconds: float | None = None
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        if parts[0] == "Total test time":
            try:
                seconds = float(parts[1])
            except ValueError:
                pass
            continue
        # per-core: leading integer cpu id, then count, then event name
        if parts[0].isdigit() and parts[1].isdigit() and parts[2] in CPU_COUNTERS:
            per_core.setdefault(int(parts[0]), {})[parts[2]] = float(parts[1])
        elif parts[0].isdigit() and parts[1] in CPU_COUNTERS:
            totals[parts[1]] = totals.get(parts[1], 0.0) + float(parts[0])
    # Under --per-core simpleperf prints no aggregate row; sum the cores.
    if per_core and not totals:
        for row in per_core.values():
            for event, count in row.items():
                totals[event] = totals.get(event, 0.0) + count
    return totals, per_core, seconds


def _build_result(task: Task, stdout: str, stderr: str, rc: int) -> Result:
    fields = _parse(stdout)
    counters, per_core, counter_seconds = _parse_counters(stdout)

    # Any status token that is not an success word counts as failure. The old
    # test looked for the literal "FAILED", so a kernel printing `status: FAIL`
    # or `status: ERROR` was reported ok=True -- and `fields.get("status")` was
    # dead code besides, because _parse keeps only numeric values.
    _status = re.search(r"(?im)^\s*status\s*[:=]\s*(\S+)", stdout)
    _tok = _status.group(1).strip().upper() if _status else None
    failed = (rc != 0) or (_tok is not None and
                           _tok not in ("OK", "SUCCESS", "PASS", "PASSED", "DONE"))
    if failed and not fields:
        err = (stderr.strip() or stdout.strip() or f"exit {rc}")[:300]
        # A network that was never pushed fails as a bare shell error --
        # "cd: /data/local/tmp/mnv2: No such file or directory" -- which tells a
        # student nothing. Case studies 6-8 all need models staged, so name the
        # command that stages them.
        if task.workload == "model" and "No such file" in err:
            spec = MODELS.get(task.model, {})
            err = (f"model {task.model!r} is not on the device "
                   f"({MODEL_DIR}/{spec.get('dir','?')}/{spec.get('dlc','?')}). "
                   f"It has to be staged on the board before it can run; ask a TA.")
        return Result(task, ok=False, raw=stdout, error=err)

    # First present of the priority fields (duration_ns is ns -> us).
    compute_us = None
    compute_key = None
    for key in _COMPUTE_FIELDS:
        if key in fields and isinstance(fields[key], (int, float)):
            compute_us = fields[key] / 1000.0 if key == "duration_ns" else float(fields[key])
            compute_key = key
            break

    wall_us = float(fields["wall_us"]) if "wall_us" in fields else None
    fp = float(fields["fingerprint"]) if "fingerprint" in fields else None
    if fp is None and isinstance(fields.get("checksum"), (int, float)):
        fp = float(fields["checksum"])          # conv_dsp_int8's name for it
    # The QNN matmul clients print the two halves (sum, frobenius) but never the
    # combined line every other kernel prints, so their output checksum read as
    # absent -- the one cell of the 4x4 matrix with nothing to cross-check
    # against. The halves are right there; combine them rather than rebuild and
    # re-push a device binary for a print statement.
    if fp is None and {"fingerprint_sum", "fingerprint_frob"} <= fields.keys():
        try:
            fp = float(fields["fingerprint_sum"]) + float(fields["fingerprint_frob"])
        except (TypeError, ValueError):
            fp = None

    # Prefer flop=/bytes= printed by the kernel; else derive from the workload.
    # Integer kernels (conv_dsp_int8) say ops= for the same quantity.
    wl = WORKLOADS[task.workload]
    flop = fields.get("flop", fields.get("ops"))
    if flop is None and wl.flop_fn is not None:
        flop = wl.flop_fn(task.params)          # always per-iteration
    nbytes = fields.get("bytes")
    # flop_fn describes ONE dispatch. When the kernel only reports wall_us -- a
    # total across every repetition it was asked for -- dividing one dispatch's
    # work by that total understates throughput by exactly the repeat count.
    # The CPU NEON matmul does this: `iterations=20` reported 1.37 GFLOP/s
    # instead of ~29. The other timing fields (execute_us_mean, dsp_us,
    # mean_us) are already per-dispatch, so they must not be scaled.
    reps = 1
    if flop is not None and "flop" not in fields and "ops" not in fields and compute_key == "wall_us":
        for key in ("iterations", "reps", "inner-reps", "inner_reps"):
            if key in _merged_params(task):
                try:
                    reps = max(1, int(_merged_params(task)[key]))
                except (TypeError, ValueError):
                    reps = 1
                break
    # The mirror-image mismatch, which the `reps` rule above does not cover.
    # `reps` handles "flop_fn gives ONE dispatch, wall_us is a TOTAL" by scaling
    # the work up. The DSP kernels have the opposite problem: they PRINT
    # flop=/bytes= as totals across every iteration while `dsp_us` is measured
    # PER iteration, so the work has to be scaled back down instead. Left
    # unhandled this inflates every DSP figure by the iteration count -- 50x on
    # saxpy/reduction and 2x on convolution at the registry defaults.
    work_div = 1
    if task.entry.totals_with_per_iteration_time and compute_key != "wall_us":
        it = fields.get("iterations", fields.get("reps"))
        if isinstance(it, (int, float)) and int(it) >= 1:
            work_div = int(it)

    gflops = gbps = None
    if compute_us and compute_us > 0:
        if flop:
            gflops = float(flop) * reps / work_div / (compute_us * 1e-6) / 1e9
        if nbytes:
            gbps = float(nbytes) * reps / work_div / (compute_us * 1e-6) / 1e9

    # Some kernels compute their own rate and print it, because only they know
    # what counts as USEFUL. gpu_memlab is the case in point: at stride 16 the
    # memory system moves 16x the bytes it delivers, and the host has no way to
    # derive "useful" from the outside -- there is no `bytes` field to divide.
    # Without this the summary line showed a bare wall time and `.gbps` was
    # None, so the whole GPU section looked like it needed raw adb.
    if gbps is None:
        v = fields.get("useful_gbps")
        if isinstance(v, (int, float)):
            gbps = float(v)
    if gflops is None:
        v = fields.get("gflops")
        if isinstance(v, (int, float)):
            gflops = float(v)

    derived = derive(counters)

    # Effective clock, from cycles actually retired over simpleperf's own window.
    # `scaling_cur_freq` reports the governor's REQUEST; under thermal throttling
    # the silicon runs slower and sysfs keeps claiming the requested value. A
    # sweep that silently drops from 4.40 to 1.37 GHz mid-run looks like a
    # workload effect unless the real clock is recorded alongside it.
    #
    # The denominator MUST be simpleperf's total test time, not the kernel's
    # self-timed wall_us: the counts cover the whole process (startup, input
    # generation, teardown) while wall_us covers only the timed region. Dividing
    # whole-process cycles by region-only time reported 17 GHz on a 4.4 GHz part.
    # The governor's request, in the same unit as effective_ghz so the two are
    # directly comparable. Present whenever the CPU branch ran, counters or not.
    # Reported only when the workload itself ran long enough to be sampled.
    # The sampler polls every 5 ms and spans the whole shell invocation, so a
    # 7 ms workload yields one or two in-run samples out of ~30 -- the rest are
    # idle before and after, and the maximum then reads LOWER than the effective
    # clock, teaching the exact opposite of the truth. Gating on sample count
    # does not catch this (the count is high); gating on runtime does.
    # At a 2 ms poll, 15 ms of workload is ~7 in-run samples: the floor for a
    # maximum that means anything.
    _scf = fields.get("scaling_cur_freq_khz_max")
    _nsc = fields.get("scaling_cur_freq_samples")
    _wall = fields.get("wall_us")
    if (isinstance(_scf, (int, float)) and _scf > 0
            and isinstance(_nsc, (int, float)) and _nsc >= 5
            and isinstance(_wall, (int, float)) and _wall >= 15000):
        derived["scaling_cur_freq_ghz"] = float(_scf) / 1e6
        derived["scaling_cur_freq_samples"] = float(_nsc)

    cycles = counters.get("cpu-cycles")
    below_max = False
    if cycles and counter_seconds and counter_seconds > 0:
        nthreads = task.threads or 1
        ghz = cycles / nthreads / counter_seconds / 1e9
        derived["effective_ghz"] = ghz
        # Which cluster did this run on? Only a mask entirely inside one cluster
        # gives an unambiguous ceiling to compare against.
        cores = set(task.cores)
        ceiling = None
        if cores and cores <= {6, 7}:
            ceiling = CLUSTER_MAX_GHZ["prime"]
        elif cores and cores <= {0, 1, 2, 3, 4, 5}:
            ceiling = CLUSTER_MAX_GHZ["gold"]
        if ceiling:
            derived["clock_of_max_pct"] = ghz / ceiling * 100.0
            below_max = ghz < ceiling * CLOCK_FLAG_RATIO

    # A run that reported no time at all is a contract violation, not a success.
    # Left as ok=True it produced a blank summary line and no error -- the least
    # helpful thing a first custom kernel can do. `model` dispatches legitimately
    # have no compute_us (qnn-net-run prints none) but do carry wall_us, so
    # requiring one OR the other covers both without special-casing.
    # Scoped to kernels: `model` dispatches run qnn-net-run, which prints no
    # timing field we parse -- their wall_us is filled in by _run_one afterwards,
    # so it is still None here and a naive check would fail every model.
    contract_error = None
    if not failed and compute_us is None and task.workload != "model":
        failed = True
        contract_error = (
            "kernel printed no timing field: expected one of "
            + ", ".join(_COMPUTE_FIELDS)
            + ". Start from `python scripts/new_kernel.py <name>`, whose kernel"
            " already prints every required field.")

    return Result(task, ok=not failed, compute_us=compute_us, wall_us=wall_us,
                  compute_key=compute_key,
                  gflops=gflops, gbps=gbps, fingerprint=fp, fields=fields, raw=stdout,
                  error=None if not failed else (contract_error or "kernel reported failure"),
                  counters=counters, per_core=per_core, derived=derived,
                  multiplexed=len(task.counter_events) > MAX_HW_COUNTERS,
                  below_max_clock=below_max)


# ─────────────────────────────────────────────────────────────────────────────
# Thermal
# ─────────────────────────────────────────────────────────────────────────────
#
# This HDK runs hot and has fallen over before: a sustained NPU graph reached
# 98 C with mitigation engaged and the board dropped off USB (discovery 73).
# Measured here 2026-09-22: idle ~30 C, one heavy 6-thread CPU dispatch 62 C,
# four back to back plateau at 77-79 C. 2026-10-09: Task 7's all-core
# interference runs read 88-92 C right after the dispatch (it drops within
# seconds), so 85 C warned on every one of them. 90 C still catches a runaway
# well before 98.

THERMAL_LIMIT_C = 90.0

# A WHITELIST, not a filter, and deliberately so. /sys/class/thermal also
# carries `cpu-hw-trip-0` and `cpu-hw-trip-1`, which read a constant 105000 --
# they are trip-point settings, not sensors. Taking max() over every zone
# reports 105 C forever, and this project published that false claim once
# already. Anything not matched here is ignored.
#
# `cpu-N-M-K` are the 16 core sensors, `cpullc-*` the cluster caches, `gpuss-*`
# the 11 Adreno zones, `nsphvx-*`/`nsphmx-*` the 8 Hexagon ones (HVX and HMX --
# the DSP and NPU), and `ddr` the memory. Note there is NO `cpuss` zone on
# SM8850P; that was the SM8750's name.
_THERMAL_ZONE_RE = re.compile(r"^(?:cpu-\d+-\d+-\d+|cpullc-\d+-\d+|gpuss-\d+"
                              r"|nsphvx-\d+|nsphmx-\d+|ddr)$")


class ThermalReader:
    """Reads the die sensors that matter, cheaply enough to do per dispatch.

    Scanning all 135 zones costs about a second, because it is two `cat`s per
    zone over adb. Resolving the zone indices once and then reading only the
    whitelisted `temp` files in a single `cat` costs about 60 ms.
    """

    def __init__(self, device: "Device"):
        self.device = device
        self._paths: list[str] | None = None   # resolved lazily, once
        self._types: list[str] = []

    def _resolve(self) -> None:
        """Map zone index -> type once, and keep the ones we care about."""
        out, _, rc = self.device.shell(
            'for z in /sys/class/thermal/thermal_zone*; do '
            'echo "${z##*/} $(cat $z/type 2>/dev/null)"; done', timeout=30)
        paths, types = [], []
        if rc == 0:
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 2 and _THERMAL_ZONE_RE.match(parts[1]):
                    paths.append(f"/sys/class/thermal/{parts[0]}/temp")
                    types.append(parts[1])
        self._paths, self._types = paths, types

    def read(self) -> dict[str, float]:
        """zone type -> degrees C. Empty dict when nothing is readable."""
        if self._paths is None:
            self._resolve()
        if not self._paths:
            return {}
        out, _, rc = self.device.shell("cat " + " ".join(self._paths), timeout=30)
        if rc != 0:
            return {}
        temps: dict[str, float] = {}
        for zone, raw in zip(self._types, out.split()):
            raw = raw.strip()
            if raw.lstrip("-").isdigit():
                # millidegrees. Zones report -273000 when a sensor is disabled;
                # averaging that in would mask a genuinely hot neighbour.
                c = int(raw) / 1000.0
                if -40.0 < c < 200.0:
                    temps[zone] = c
        return temps


# ─────────────────────────────────────────────────────────────────────────────
# Device — the thin adb layer
# ─────────────────────────────────────────────────────────────────────────────

class Device:
    """Shell commands on the HDK over adb. serial=None uses the only device."""

    def __init__(self, serial: str | None = None, adb: str = "adb", base: str = DEVICE_BASE):
        self.serial = serial
        self.adb = adb
        self.base = base

    def shell(self, command: str, timeout: float = 180.0) -> tuple[str, str, int]:
        """Run one adb shell command. Returns `(stdout, stderr, returncode)`."""
        argv = [self.adb] + (["-s", self.serial] if self.serial else []) + ["shell", command]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            return proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            return "", f"timeout after {timeout}s", 124

    def is_connected(self) -> bool:
        """True if the board answers an echo within 10 s."""
        out, _, rc = self.shell("echo ok", timeout=10)
        return rc == 0 and "ok" in out


# ─────────────────────────────────────────────────────────────────────────────
# Backend command builders
# ─────────────────────────────────────────────────────────────────────────────

# SM8850P clusters: Prime = cores 6-7, Gold = cores 0-5. Same split as the
# SM8750 this began on (Discovery 07/26), but the ladders differ: 28 Gold OPPs
# to 3.6288 GHz and 27 Prime to 4.3968 GHz, sharing only 4 setpoints.
_CLUSTERS = {"prime": 0xC0, "gold": 0x3F, "all": 0xFF}


def _affinity_mask(core) -> str:
    """core spec -> taskset hex mask. None -> "f0" (the sweeps' default)."""
    if core is None:
        return "f0"
    if isinstance(core, int):
        return format(1 << core, "x")
    if isinstance(core, (list, tuple)):
        m = 0
        for c in core:
            m |= 1 << int(c)
        return format(m, "x")
    s = str(core).lower()
    if s in _CLUSTERS:
        return format(_CLUSTERS[s], "x")
    return s[2:] if s.startswith("0x") else s  # raw hex mask


def _flags(params: dict) -> str:
    """kwargs -> CLI flags: perf_mode='x' -> '--perf-mode x', True -> bare flag,
    None/False dropped."""
    out: list[str] = []
    for k, v in params.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        out.append(flag if v is True else f"{flag} {shlex.quote(str(v))}")
    return " ".join(out)


def _merged_params(task: Task) -> dict:
    """Entry defaults overlaid with the Task's params (Task wins)."""
    p = dict(task.entry.defaults)
    p.update(task.params)
    for src, dst in task.entry.param_map.items():
        if src in p:
            p[dst] = p.pop(src)
    if task.entry.variant_is_flag:
        p["variant"] = task.resolved_variant
    if task.threads is not None:
        p["threads"] = int(task.threads)     # __post_init__ proved the variant takes it
    return p


def _simpleperf_prefix(task: Task) -> str:
    """`simpleperf stat ...` wrapper, or "" when no counters were requested.

    Default is PROCESS-scoped: count only this workload, wherever it runs. Pass
    counter_cpus to switch to per-CPU counting, which is system-wide on those
    CPUs and therefore also catches whatever else lands there.
    """
    if not task.counter_events:
        return ""
    bits = ["simpleperf", "stat", "--csv"]
    if task.per_core:
        bits.append("--per-core")
    if task.counter_cpus is not None:
        cpus = (task.counter_cpus if isinstance(task.counter_cpus, str)
                else ",".join(str(c) for c in
                              ([task.counter_cpus] if isinstance(task.counter_cpus, int)
                               else task.counter_cpus)))
        bits += ["--cpu", cpus]
    bits += ["-e", ",".join(task.counter_events)]
    return " ".join(bits) + " "


def build_command(task: Task, base: str) -> str:
    """The exact device shell command for a Task. Callable without hardware.

    When counters are requested the whole command is re-run under `su 0`. On
    this board (SM8850P, Android 16) the shell domain cannot open the PMU:
    simpleperf answers "Event type 'cpu-cycles' is not supported on the device"
    for EVERY event, including ones `simpleperf list` advertises, while the
    identical command under `su 0` counts normally. `perf_event_paranoid` is
    already -1, so this is an SELinux domain restriction, not a paranoid level.

    This is `su 0` from an unrooted adbd, which is not the same thing as
    `adb root`. `adb root` restarts adbd in a domain that cannot reach the
    OpenCL driver, so it breaks every GPU dispatch; that trap is Lab 3 Task 1
    and we must not spring it here just to read a counter.
    """
    cmd = _build_command_raw(task, base)
    if task.counter_events:
        # shlex.quote, not a hand-rolled escape. The hand-rolled one here used
        # "'\''", which in Python source is three quotes rather than the POSIX
        # escape it resembles -- and the DSP command is the only one carrying a
        # single quote (ADSP_LIBRARY_PATH='.;<dir>'), so DSP counter runs died
        # with the shell trying to execute the working directory while every
        # other backend was fine.
        cmd = f"su 0 sh -c {shlex.quote(cmd)}"
    return cmd


def _build_command_raw(task: Task, base: str) -> str:
    """The command itself, before any privilege wrapper."""
    if task.workload == "model":
        # Networks go through qnn-net-run, not our kernels: one DLC, and the
        # engine is chosen by which QNN backend library is loaded.
        spec = MODELS[task.model]
        wdir = f"{MODEL_DIR}/{spec['dir']}"
        reps = int(task.params.get("reps", 1))
        out = task.params.get("output_dir") or f"run_{task.model}_{task.backend}"
        return (f"cd {wdir} && rm -rf {out} && LD_LIBRARY_PATH={wdir} "
                f"ADSP_LIBRARY_PATH={wdir} ./qnn-net-run "
                f"--dlc_path {spec['dlc']} --backend {_QNN_BACKEND[str(task.backend)]} "
                f"--input_list {spec['list']} --output_dir {out} "
                f"--num_inferences {reps}")

    entry = task.entry
    binary = task.binary or entry.variants[task.resolved_variant]
    wdir = f"{base}/{entry.subdir}" if entry.subdir else base
    args = _flags(_merged_params(task))
    perf = _simpleperf_prefix(task)

    if task.backend is Backend.CPU:
        path = f"{wdir}/{binary}"
        cmd = f"{perf}taskset {_affinity_mask(task.core)} {path} {args}".strip()
        # Single-threaded runs only. The sampler forks `cat` and `sleep` every
        # 2 ms, which is free when there are idle cores and expensive when there
        # are not: measured +31% wall time on a 6-thread convolution across the
        # whole Gold cluster, against ~1.5% on one Prime core with the rest of
        # the machine idle. That perturbation lands squarely on the multi-core
        # scaling tasks, where it would be read as poor parallel efficiency.
        # simpleperf itself costs nothing measurable either way.
        threads = task.threads or 1
        if not task.counter_events or threads > 1:
            return cmd
        # With counters on, also record what the GOVERNOR was asking for while
        # the workload ran, so it can be set beside effective_ghz (which is
        # derived from cycles actually retired). The two disagree constantly on
        # this board: sysfs will happily report the 4.608 GHz boost point for a
        # run whose real average clock was under 2 GHz.
        #
        # Sampled during the run, not after: the governor decays to idle within
        # milliseconds of the workload exiting, so a single read afterwards
        # reports the idle clock and characterises nothing.
        #
        # The sampler stops on a sentinel file rather than being killed. An
        # earlier version used `kill $!` and took the whole adb shell down with
        # it (exit 143). Only runs when counters were explicitly requested --
        # the profiler does not start background work it was not asked for.
        #
        # Cost, measured: ~1.5% on one Prime core with the rest of the machine
        # idle (matmul eigen N=1024, 107.2 -> 105.5 GFLOP/s), but +31% when the
        # workload already occupies every core in the cluster. Hence the
        # single-thread guard above -- the cheap case is the only one it runs in.
        pol = 6 if (task.cores and min(int(c) for c in task.cores) >= 6) else 0
        f, d = "/data/local/tmp/_scf.$$", "/data/local/tmp/_scf.$$.done"
        sysfs = f"/sys/devices/system/cpu/cpufreq/policy{pol}/scaling_cur_freq"
        return (
            f"rm -f {f} {d}; "
            f"( while [ ! -f {d} ]; do cat {sysfs} 2>/dev/null; sleep 0.01; done > {f} ) & "
            f"{cmd}; __rc=$?; "
            f"touch {d}; sleep 0.05; "
            f"echo scaling_cur_freq_khz_max=$(sort -rn {f} 2>/dev/null | head -1); "
            f"echo scaling_cur_freq_samples=$(wc -l < {f} 2>/dev/null); "
            f"rm -f {f} {d}; exit $__rc")

    if task.backend is Backend.GPU:
        return f"cd {wdir} && LD_LIBRARY_PATH={entry.ldpath} {perf}./{binary} {args}".strip()

    if task.backend is Backend.DSP:
        # FastRPC: stub beside cwd, skel on ADSP_LIBRARY_PATH.
        return (f"cd {wdir} && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH='.;{wdir}' "
                f"{perf}./{binary} {args}").strip()

    if task.backend is Backend.NPU:
        return (f"cd {wdir} && LD_LIBRARY_PATH={wdir} ADSP_LIBRARY_PATH={wdir} "
                f"{perf}./{binary} {args}").strip()

    raise ValueError(f"unhandled backend {task.backend}")


# ─────────────────────────────────────────────────────────────────────────────
# Profiler — the front door
# ─────────────────────────────────────────────────────────────────────────────

class Profiler:
    """Dispatches Tasks to the device and returns Results."""

    def __init__(self, device: Device | None = None, *, verbose: bool = False,
                 thermal: bool = True, thermal_limit_c: float = THERMAL_LIMIT_C):
        """`thermal` reads the die sensors after each dispatch (about 60 ms) and
        warns once the hottest zone passes `thermal_limit_c`.

        It REPORTS and never acts: it will not sleep, throttle, retry or refuse.
        Whether a hot board is a problem depends on what you are measuring, and
        that judgement stays with you. Pass `thermal=False` to switch the read
        off entirely -- worth doing for a long sweep of short dispatches, where
        60 ms per point is the dominant cost.
        """
        self.device = device or Device()
        self.verbose = verbose
        self.thermal = thermal
        self.thermal_limit_c = thermal_limit_c
        self._warned_at = 0.0
        self._thermal = ThermalReader(self.device)
        self._warned_hot = False

    def run(self, tasks, *, parallel: bool = False):
        """One Task -> Result; a list -> [Result], serial unless parallel=True."""
        if isinstance(tasks, Task):
            return self._run_one(tasks)
        return self.run_parallel(tasks) if parallel else self.run_serial(tasks)

    def _run_one(self, task: Task) -> Result:
        cmd = build_command(task, self.device.base)
        if self.verbose:
            print(f"  $ {cmd}")
        t0 = time.monotonic()
        out, err, rc = self.device.shell(cmd, timeout=task.timeout)
        host_us = (time.monotonic() - t0) * 1e6
        r = _build_result(task, out, err, rc)
        # Host-observed round trip, always recorded. For `model` tasks it is the
        # ONLY timing available -- qnn-net-run prints none -- and it includes adb
        # latency and graph preparation, so it is never a per-inference figure.
        r.fields["host_wall_us"] = round(host_us, 1)
        self._check_thermal(r)
        if task.workload == "model":
            r.ok = r.ok and "Finished Executing" in out
            r.wall_us = round(host_us, 1)
            r.error = None if r.ok else (r.error or "qnn-net-run did not finish")
        return r

    def run_serial(self, tasks: list[Task]) -> list[Result]:
        """Strictly one after another — isolates each measurement."""
        return [self._run_one(t) for t in tasks]

    def run_parallel(self, tasks: list[Task]) -> list[Result]:
        """One host thread per Task. Different backends are independent hardware
        units, so those Tasks genuinely execute at the same time."""
        if not tasks:
            return []
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            return list(pool.map(self._run_one, tasks))

    def time_model(self, backend: Backend, name: str, reps: int = 100) -> dict:
        """Per-inference time for a network, with graph preparation removed.

        Runs the model twice -- once and then reps+1 times -- and subtracts.
        Preparation is paid once per dispatch and differs a lot by backend (CPU
        0.16 s, GPU 0.52 s, HTP 0.59 s on MobileNetV2), so a single timed run
        measures mostly compilation. The subtraction cancels it, along with the
        constant adb round trip.

        Returns per-inference ms, the prepare cost, and inferences/second.
        """
        a = self._run_one(model(backend, name, reps=1))
        b = self._run_one(model(backend, name, reps=reps + 1))
        if not (a.ok and b.ok):
            return {"model": name, "backend": str(backend), "ok": False,
                    "error": (a.error or b.error or "dispatch failed")}
        per_ms = (b.wall_us - a.wall_us) / reps / 1000.0
        # The subtraction cancels graph preparation only if preparation costs the
        # same in both runs. It often does not: a cold DLC read from storage can
        # dominate the first run, and then the reps+1 run comes back FASTER in
        # wall time and per_ms goes negative. Reporting -0.75 ms/inference as a
        # measurement is worse than reporting nothing, so refuse and say what to
        # do -- more reps push the per-inference signal above the load noise.
        if per_ms <= 0:
            return {"model": name, "backend": str(backend), "ok": False,
                    "error": (f"load-time noise exceeded the per-inference signal "
                              f"(reps={reps} gave {per_ms:.3f} ms/inference). "
                              f"Raise reps: this model needs enough inferences "
                              f"that they outweigh the one-off graph load."),
                    "per_inference_ms": None,
                    "prepare_plus_one_ms": round(a.wall_us / 1000.0, 1),
                    "reps": reps,
                    "macs": MODELS[name]["macs"]}
        return {"model": name, "backend": str(backend), "ok": True,
                "per_inference_ms": round(per_ms, 3),
                "prepare_plus_one_ms": round(a.wall_us / 1000.0, 1),
                "inferences_per_s": round(1000.0 / per_ms, 1),
                "reps": reps,
                "macs": MODELS[name]["macs"]}

    def power(self, task: Task, *, seconds: int = 35,
              artifact: str | None = None) -> dict:
        """Estimated TOTAL BOARD power and energy while `task` runs in a loop.

            p.power(matmul(Backend.CPU, size=1024))
            # {'available': True, 'power_w': 6.59, 'energy_j': 230.6, ...}

        Three things this is not, all of which matter more than the number:

        * **Not a measurement.** It is a calibrated model of 12 V supply-input
          power, fitted against a bench supply. `quality` says
          `modeled_ballpark` for that reason.
        * **Not per-engine.** It predicts what the whole board draws. There is no
          readable per-rail telemetry on this HDK (discovery 57), so "GPU watts"
          is not a thing this can produce.
        * **Not per-dispatch.** The model is only validated over intervals of
          `minimum_validated_interval_s` (30 s), so the task is looped to fill
          `seconds` and the answer describes that window. A single 200 us
          dispatch has no modeled power.

        Returns `available: False` with a `reason` when the loaded artifact has
        nothing for the task's engine. With the shipped artifact that is **DSP**:
        an unsigned PD cannot cast the HAP power vote the calibration needs
        (discovery 64). CPU and GPU are fitted; the NPU carries declared operator
        readings (5 W fp16, 4 W int8) and is flagged `validated: False` so a
        reader can tell a fitted number from a hand-entered one. Reporting
        `unavailable` is the correct answer, not a fallback number: see
        `decisions/0019`, `0020` and `0022`.

        One task, one engine. The model's concurrent-path terms are not summable
        (`concurrent_path_terms_summable: false`), so there is no call that gives
        the power of a multi-engine pipeline. Measure each stage and add the
        ENERGIES -- see `docs/heterogeneous-workloads.md`.
        """
        import subprocess as _sp

        art = pathlib.Path(artifact) if artifact else _default_power_artifact()
        if not art.exists():
            return {"available": False, "engine": str(task.backend),
                    "reason": f"no power artifact at {art}"}
        spec = json.loads(art.read_text(encoding="utf-8-sig"))
        hexagon = spec.get("hexagon_subsystem", {})
        engine = str(task.backend)
        declared = spec.get("npu_declared")
        if engine in ("npu", "dsp"):
            # `declared` counts as available but NOT as calibrated: the returned
            # record carries validated=False and the caveats, so a reader can
            # tell a measured-and-fitted number from a hand-entered one.
            ok = (hexagon.get(f"{engine}_calibrated", False)
                  or (engine == "npu" and hexagon.get("npu_declared", False)))
            if not ok:
                return {
                    "available": False, "engine": engine,
                    "reason": hexagon.get(f"{engine}_uncalibrated_reason",
                                          f"{engine} is not calibrated in "
                                          f"profile {spec.get('model_profile')!r}"),
                    "model_profile": spec.get("model_profile"),
                }
        # A DECLARED NPU number is not computed from telemetry, so there is no
        # window to sample. Returning it directly is both honest -- running a
        # 35 s NPU window and then ignoring its features would be theatre -- and
        # safer, since that is exactly the run that drives the die to 98 C
        # (discovery 73).
        if engine == "npu" and declared and not hexagon.get("npu_calibrated", False):
            prec = task.variant or declared.get("default_precision", "fp16")
            total = declared.get("totals_w", {}).get(prec)
            if total is None:
                return {"available": False, "engine": engine,
                        "reason": f"no declared NPU total for precision {prec!r}; "
                                  f"have {sorted(declared.get('totals_w', {}))}"}
            return {
                "available": True,
                "engine": engine,
                "precision": prec,
                "power_w": total,
                "energy_j": round(total * seconds, 1),
                "window_s": seconds,
                "boundary": spec.get("boundary"),
                "scope": "whole_board_not_per_engine",
            }
        if seconds < spec.get("minimum_validated_interval_s", 30.0):
            return {"available": False, "engine": engine,
                    "reason": f"seconds={seconds} is below the artifact's "
                              f"validated interval "
                              f"{spec['minimum_validated_interval_s']}s"}

        # A single dispatch is microseconds and the sampler marks a window BAD
        # the moment the workload PID dies, so the task has to be looped past the
        # end of the window and then exit 0 -- which is also what the session
        # build requires (`WORKLOAD_EXIT status=0`). The margin covers the 8 s
        # clock-ramp settle power_run.py takes before it opens the window.
        #
        # The loop lives in a device SCRIPT rather than an inline `sh -c '...'`:
        # power_run.py single-quotes each worker, so any quote of ours collides
        # with its quoting and the wrapper silently registers 0 PIDs.
        out = pathlib.Path(tempfile.mkdtemp(prefix="profiler-power-"))
        inner = build_command(task, DEVICE_BASE)
        budget = seconds + 25
        loop = out / "pp_loop.sh"
        loop.write_text(
            "#!/system/bin/sh\n"
            f"end=$(( $(date +%s) + {budget} ))\n"
            "while [ $(date +%s) -lt $end ]; do\n"
            f"  {inner} >/dev/null 2>&1\n"
            "done\n"
            "exit 0\n", encoding="utf-8", newline="\n")
        dev_loop = "/data/local/tmp/profiler_power_loop.sh"
        _sp.run(["adb", "push", str(loop), dev_loop],
                capture_output=True, text=True)
        _sp.run(["adb", "shell", f"chmod 755 {dev_loop}"],
                capture_output=True, text=True)
        cmd = f"sh {dev_loop}"
        # A 35 s window plus setup is about a minute of silence. Without a line
        # here the call looks hung, and the first thing a student does about a
        # hung command is Ctrl-C it. stderr, so stdout stays parseable.
        print(f"  measuring power: looping {task.name or task.workload} for "
              f"{seconds} s ...", file=sys.stderr, flush=True)
        r = _sp.run([sys.executable, str(_HERE / "scripts" / "power_run.py"),
                     "--cmd", cmd, "--engine", engine,
                     "--label", task.name or f"{task.workload}.{engine}",
                     "--seconds", str(seconds), "--artifact", str(art),
                     "--out", str(out), "--json"],
                    capture_output=True, text=True, cwd=str(_HERE))
        # The loop is given `seconds + 25` so it cannot end mid-window, which
        # means it OUTLIVES the measurement by up to 25 s. Left alone, a second
        # power() call lands while the first loop is still running and
        # power_run.py refuses it ("compute clients are already running").
        # Tasks that measure several configurations in a row hit this every
        # time, so reap the loop here rather than leaving it to expire.
        _sp.run(["adb", "shell",
                 f"pkill -f {dev_loop} 2>/dev/null; rm -f {dev_loop}"],
                capture_output=True, text=True)
        try:
            est = json.loads(r.stdout)
        except ValueError:
            return {"available": False, "engine": engine,
                    "reason": "power_run.py produced no estimate",
                    "output": (r.stdout + r.stderr).strip()[:400]}
        watts = est.get("power_w")
        out_rec = {
            "available": watts is not None,
            "engine": engine,
            "power_w": watts,
            "energy_j": round(watts * seconds, 1) if watts is not None else None,
            "window_s": seconds,
            "boundary": spec.get("boundary"),
            "scope": "whole_board_not_per_engine",
        }
        # Carry the estimator's own verdict through. Without this an
        # `available: False` arrives with no explanation and the only way to
        # find out why is to re-run power_run.py by hand -- which is exactly
        # how the sparse-capture and EAS-gate bugs stayed hidden. `reason` says
        # why there is no number; `warnings` says what to distrust about one
        # that IS present.
        if est.get("reason"):
            out_rec["reason"] = est["reason"]
        if est.get("warnings"):
            out_rec["warnings"] = est["warnings"]
        # "calibration_board" means the coefficients were fitted on this exact
        # board; "transferred_eas_verified" means they were fitted elsewhere and
        # this board's kernel energy model was checked against the calibration
        # board's before they were applied. The second carries more uncertainty
        # in the absolute watts and none in a comparison between two runs here.
        if est.get("board_match"):
            out_rec["board_match"] = est["board_match"]
        if watts is None and not est.get("reason"):
            out_rec["reason"] = f"estimator returned kind={est.get('kind')!r} with no power_w"
        return out_rec

    def _check_thermal(self, r: Result) -> None:
        """Record the die temperature after a dispatch, and say if it is high.

        Deliberately after, not during: reading sysfs mid-run would perturb the
        measurement, and the question being asked here is "is the board too hot
        to keep going", which is a question about now.
        """
        if not self.thermal:
            return
        temps = self._thermal.read()
        if not temps:
            return
        r.temps = temps
        zone, hottest = max(temps.items(), key=lambda kv: kv[1])
        r.temp_c = hottest
        r.too_hot = hottest >= self.thermal_limit_c
        if r.too_hot and (not self._warned_hot or hottest >= self._warned_at + 5.0):
            # stderr, so stdout stays parseable for a sweep piping to a file.
            # Once per hot spell (plus if it climbs another 5 C), not once per
            # task: run_parallel checks after every task and used to repeat it.
            self._warned_at = hottest
            print(f"  TOO HOT: {hottest:.0f} C on {zone} "
                  f"(limit {self.thermal_limit_c:.0f} C). Let the board cool "
                  f"for a minute before the next run -- measurements taken now "
                  f"are throttled, and this board has dropped off USB at 98 C.",
                  file=sys.stderr, flush=True)
            self._warned_hot = True
        elif self._warned_hot and hottest < self.thermal_limit_c - 5.0:
            print(f"  cooled to {hottest:.0f} C.", file=sys.stderr, flush=True)
            self._warned_hot = False

    def run_dag(self, tasks: list[Task]) -> dict[str, Result]:
        """Named Tasks in dependency order; each level runs in parallel."""
        by_name = {t.name: t for t in tasks}
        if len(by_name) != len(tasks):
            raise ValueError("run_dag requires unique Task.name values")
        for t in tasks:
            for dep in t.depends_on:
                if dep not in by_name:
                    raise ValueError(f"{t.name} depends on unknown task {dep!r}")

        results: dict[str, Result] = {}
        remaining = set(by_name)
        while remaining:
            ready = [by_name[n] for n in remaining
                     if all(d in results for d in by_name[n].depends_on)]
            if not ready:
                raise ValueError(f"dependency cycle among: {sorted(remaining)}")
            for t, r in zip(ready, self.run_parallel(ready)):
                results[t.name] = r
                remaining.discard(t.name)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI / demo
# ─────────────────────────────────────────────────────────────────────────────

def _catalogue() -> None:
    """Print the kernel catalogue straight from WORKLOADS.

    Generated rather than written, so it cannot drift from what actually
    dispatches. The hand-maintained catalogue in docs/profiler.tex already did:
    it predates the decode workload, custom(), and six GPU matmul variants."""
    print("Kernel catalogue - generated from the registry, so it is current.\n")
    for wname, wl in WORKLOADS.items():
        if wname == "custom":
            print("custom")
            print("    kernels of your own; scaffold one with: python scripts/new_kernel.py <name>")
            print("    python profiler.py custom cpu --binary <name> -p size=1048576\n")
            continue
        print(wname)
        for b, e in wl.kernels.items():
            marks = [v + (f" (size={e.fixed_size[v]} only)" if v in e.fixed_size else "")
                     + ("*" if v == e.default else "") for v in e.variants]
            print(f"    {str(b):4} {', '.join(marks)}")
            if e.defaults:
                always = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in e.defaults.items())
                print(f"         always sent: {always}")
            if e.param_map:
                ren = ", ".join(f"-p {k} -> --{v}" for k, v in e.param_map.items())
                print(f"         renamed:     {ren}")
        print()
    print("* marks the default variant. Parameters go through -p, e.g. -p size=512.")


def _demo(profiler: Profiler) -> None:
    print("== connectivity ==")
    if not profiler.device.is_connected():
        # ASCII on purpose: this is the first error a new student sees, and a
        # Windows console renders an em-dash as a replacement character.
        print("  no device - is the HDK attached? (adb devices)")
        return
    print("  device OK\n")

    print("== convolution AI=18 (C=8), every backend IN PARALLEL ==")
    tasks = [convolution(b, cin=8, cout=8) for b in Backend]
    for r in profiler.run_parallel(tasks):
        print("  " + r.summary())
    # Without this note the NPU looks broken: it reads ~2 GFLOP/s here and ~460
    # two sections down, for the same workload. The cause is the shared Hexagon,
    # and leaving a 200x discrepancy unexplained in a first-run demo is worse
    # than spending three lines on it.
    print("   note the NPU: ~200x slower here than it is serially, below. It")
    print("   shares the Hexagon with the DSP and queues behind the DSP's 160 ms")
    print("   of convolution. Four engines, three independent.")

    print("\n== the roofline inversion: NPU on SAXPY vs convolution (serial) ==")
    print("   this pauses ~50 s and is not hung: QNN has no native SAXPY op, so the")
    print("   graph decomposes into thousands of tiles. The wait IS the finding -")
    print("   watch finalize_us below, then compare it against convolution's ~12 ms.")
    for r in profiler.run_serial([saxpy(Backend.NPU, size=65536),
                                  convolution(Backend.NPU, cin=8, cout=8)]):
        print("  " + r.summary())

    print("\n== CPU core affinity: same reduction on Prime (6-7) vs Gold (0-5), serial ==")
    core_tasks = [
        reduction(Backend.CPU, size=4194304, core="prime", name="prime_6-7"),
        reduction(Backend.CPU, size=4194304, core="gold", name="gold_0-5"),
        reduction(Backend.CPU, size=4194304, core=7, name="single_core_7"),
    ]
    # parallel=True instead makes them contend for DRAM — same call, one flag.
    for r in profiler.run(core_tasks, parallel=False):
        print("  " + r.summary())

    print("\n== hardware counters: same conv, 1 thread vs 6, with cache stats ==")
    for n in (1, 6):
        r = profiler.run(convolution(Backend.CPU, cin=32, cout=32, iterations=2,
                                     threads=n, core="gold", counters="cache",
                                     name=f"conv_{n}t"))
        d = r.derived
        print(f"  {r.task.name:10s} {r.compute_us:9.0f} us  IPC={d.get('ipc', 0):.2f}"
              f"  L1D miss={d.get('l1d_miss_pct', 0):.2f}%"
              f"  L2 miss={d.get('l2d_miss_pct', 0):.2f}%")

    print("\n== a tiny DAG: CPU reduction, then GPU+NPU convolution after it ==")
    dag = profiler.run_dag([
        reduction(Backend.CPU, size=1048576, name="warmup"),
        convolution(Backend.GPU, cin=16, cout=16, name="gpu_conv", depends_on=["warmup"]),
        convolution(Backend.NPU, cin=16, cout=16, name="npu_conv", depends_on=["warmup"]),
    ])
    for name, r in dag.items():
        print(f"  {name:10s} {r.summary()}")


def _png(path: pathlib.Path, pix: bytes, w: int, h: int) -> None:
    """Grayscale 8-bit PNG with the standard library only (no Pillow needed)."""
    import struct, zlib
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    raw = b"".join(b"\x00" + pix[y * w:(y + 1) * w] for y in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _read_pgm(path: pathlib.Path) -> tuple[bytes, int, int]:
    data = path.read_bytes()
    tok, i = [], 2
    while len(tok) < 3:                      # width, height, maxval
        while data[i:i + 1].isspace(): i += 1
        if data[i:i + 1] == b"#":
            while data[i:i + 1] not in (b"\n", b""): i += 1
            continue
        j = i
        while data[j:j + 1].isdigit(): j += 1
        tok.append(int(data[i:j])); i = j
    w, h = tok[0], tok[1]
    return data[i + 1:i + 1 + w * h], w, h


def _morph_images(task: "Task", device: "Device") -> None:
    """After `morph ... -p write_pgm=true`: copy the three images off the board
    into ./morph_images/ as PNGs, plus a side-by-side clean | damaged | repaired.
    The board writes PGMs, which Windows and macOS do not open."""
    out = pathlib.Path("morph_images")
    out.mkdir(exist_ok=True)
    tag = "{}_{}_{}_k{}".format(task.params.get("impl", "student"), task.params.get("op", "erode"),
                                task.params.get("damage", "salt"), task.params.get("k", 3))
    imgs = []
    for name in ("clean", "damaged", "repaired"):
        local = out / f"morph_{name}.pgm"
        argv = [device.adb] + (["-s", device.serial] if device.serial else []) + \
               ["pull", f"{device.base}/dsp/morph_{name}.pgm", str(local)]
        if subprocess.run(argv, capture_output=True).returncode != 0:
            print(f"  could not pull morph_{name}.pgm from the board")
            return
        pix, w, h = _read_pgm(local)
        local.unlink()
        imgs.append(pix)
    gap = 16
    W = 3 * w + 2 * gap
    rows = []
    for y in range(h):
        sep = b"\xff" * gap
        rows.append(imgs[0][y*w:(y+1)*w] + sep + imgs[1][y*w:(y+1)*w] + sep + imgs[2][y*w:(y+1)*w])
    _png(out / f"{tag}_damaged.png", imgs[1], w, h)
    _png(out / f"{tag}_repaired.png", imgs[2], w, h)
    _png(out / f"{tag}_compare.png", b"".join(rows), W, h)
    print(f"  images: {out}/{tag}_compare.png (clean | damaged | repaired)")
    print(f"  to view them, from PowerShell or a terminal on your laptop:\n"
          f"      docker cp lab3:/workspace/morph_images .")


def main() -> None:
    ap = argparse.ArgumentParser(description="SM8850P heterogeneous profiler wrapper")
    ap.add_argument("--demo", action="store_true", help="run a live tour on the attached device")
    ap.add_argument("--catalogue", "--catalog", action="store_true", dest="catalogue",
                    help="print every workload/backend/variant, generated from the registry")
    ap.add_argument("--serial", default=None, help="adb device serial (default: only device)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each device command")
    ap.add_argument("workload", nargs="?", choices=list(WORKLOADS), help="run one workload")
    ap.add_argument("backend", nargs="?", type=Backend, help="cpu|gpu|dsp|npu")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--core", default=None,
                    help="CPU affinity: a core (7), a mask (0xc0), or prime|gold|all")
    ap.add_argument("--threads", type=int, default=None,
                    help="worker threads (threaded kernel variants only)")
    ap.add_argument("--counters", default=None,
                    help="preset (basic|branch|cache|tlb|stalls|mix|memory) or a "
                         "comma-separated event list")
    ap.add_argument("--per-core", action="store_true",
                    help="break counter totals down per CPU")
    ap.add_argument("--counter-cpus", default=None,
                    help="count per-CPU system-wide on these CPUs instead of "
                         "per-process (e.g. 0-5)")
    ap.add_argument("--list-counters", action="store_true",
                    help="print the verified counter catalogue and exit")
    ap.add_argument("--param", "-p", action="append", default=[],
                    metavar="KEY=VAL", help="kernel param, e.g. -p cin=16 -p cout=16")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                    help="run N times and report mean, range and coefficient of "
                         "variation. One run is not a measurement.")
    ap.add_argument("--binary", default=None, metavar="NAME",
                    help="with workload 'custom': a binary in the device custom/ dir")
    args = ap.parse_args()

    if args.list_counters:
        print(list_counters())
        return

    profiler = Profiler(Device(serial=args.serial), verbose=args.verbose)

    if args.catalogue:
        _catalogue()
        return

    if args.demo or not args.workload:
        _demo(profiler)
        return

    params = {}
    for kv in args.param:
        k, _, v = kv.partition("=")
        # Coerce numbers. Left as strings, "512" != 512 and the fixed_size guard
        # rejects `--variant best -p size=512` with "compiled for size=512 only,
        # got size=512" -- the first GPU command the student docs give.
        # Bare-flag parameters. `_flags` turns True into a flag with no value,
        # which is what kernels like cache_latency's --hugepages expect; without
        # this the CLI can only produce `--hugepages 1` and the kernel rejects
        # the stray argument. Spelled out rather than treating 1 as true, since
        # plenty of parameters are legitimately the number 1.
        if v.lower() in ("true", "false"):
            params[k] = (v.lower() == "true")
            continue
        try:
            params[k] = int(v)
        except ValueError:
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v
    core: object = args.core
    if core is not None and core.isdigit():
        core = int(core)
    if args.workload == "custom" and not args.binary:
        raise SystemExit(
            "workload 'custom' runs a kernel of your own, so it needs --binary:\n"
            "  python profiler.py custom cpu --binary mykernel -p size=1048576\n"
            "Create one with: python scripts/new_kernel.py mykernel --push")
    if args.binary and args.workload != "custom":
        raise SystemExit("--binary applies only to workload 'custom'")
    counters = args.counters.split(",") if args.counters else None
    task = Task(args.workload, args.backend, params, variant=args.variant, core=core,
                binary=args.binary, threads=args.threads, counters=counters,
                per_core=args.per_core, counter_cpus=args.counter_cpus)
    if args.repeat < 1:
        raise SystemExit("error: --repeat must be at least 1")

    if args.repeat == 1:
        result = profiler.run(task)
        print(result.summary())
        if args.verbose and "host_wall_us" in result.fields:
            # The whole process as seen from the host: setup, every run, exit.
            print(f"  host_wall_us={result.fields['host_wall_us']:.0f}")
        if task.workload == "morph" and task.params.get("write_pgm") and result.ok:
            _morph_images(task, profiler.device)
        if result.counters:
            print()
            print(result.counter_table())
        return

    # Repeated runs are separate PROCESSES, not repetitions inside one dispatch.
    # That is the point: process-level repeats re-pay startup and let the
    # governor settle differently each time, so the spread they expose is the
    # spread a real caller would see. A kernel's own `--iterations` cannot show
    # you that -- it amortises it away.
    results = [profiler.run(task) for _ in range(args.repeat)]
    for i, r in enumerate(results, 1):
        print(f"  {i:>3}  {r.summary()}")

    ok = [r for r in results if r.ok]
    if not ok:
        raise SystemExit("error: every run failed")

    # Summarise whichever headline metric this workload actually reports.
    for name, get, unit in (("GFLOP/s", lambda r: r.gflops, ""),
                            ("GB/s", lambda r: r.gbps, ""),
                            # No rate: summarise the same time the per-run lines show
                            # (compute_us, e.g. dsp_us), not the host-side wall_us.
                            (f"time ({ok[0].compute_key or 'us'})", lambda r: r.compute_us, " us")):
        vals = [get(r) for r in ok]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        cov = (sd / mean * 100.0) if mean else 0.0
        print()
        print(f"  n={len(vals)}  {name}: mean {mean:.1f}  "
              f"range {min(vals):.1f}-{max(vals):.1f}  CoV {cov:.1f}%")
        if len(ok) != len(results):
            print(f"  ({len(results) - len(ok)} of {len(results)} runs failed "
                  f"and are excluded)")
        break

    fps = {round(r.fingerprint, 6) for r in ok if r.fingerprint is not None}
    if len(fps) > 1:
        print(f"  WARNING: {len(fps)} distinct fingerprints across these runs -- "
              f"they did not all compute the same answer")

    # Counters too. A single run prints its counter table; repeated runs used to
    # print only IPC, so --repeat silently dropped effective_ghz, the stall
    # percentages and every raw count a task asked for. Show the mean of each
    # across the runs, with the range for the derived ratios, so one command
    # gives both the spread and the counters.
    with_ctr = [r for r in ok if r.counters]
    if with_ctr:
        n = len(with_ctr)
        print()
        print(f"  counters, mean of {n} run(s):")
        events = [e for e in with_ctr[0].counters
                  if all(e in r.counters for r in with_ctr)]
        if events:
            w = max(len(e) for e in events)
            for e in events:
                mean = sum(r.counters[e] for r in with_ctr) / n
                print(f"  {e:<{w}}  {mean:>16,.0f}")
        keys = [k for k in with_ctr[0].derived
                if all(isinstance(r.derived.get(k), (int, float)) for r in with_ctr)]
        if keys:
            print()
            w = max(len(k) for k in keys)
            for k in keys:
                vals = [float(r.derived[k]) for r in with_ctr]
                mean = sum(vals) / n
                rng = f"   range {min(vals):.3f}-{max(vals):.3f}" if n > 1 else ""
                print(f"  {k:<{w}}  {mean:>16.3f}{rng}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        # Task validation raises ValueError with a message written to be read --
        # available variants, which backends a workload has, the size a variant
        # is compiled for. A traceback above it just buries the sentence that
        # tells you what to type instead.
        raise SystemExit(f"error: {exc}")
    except KeyboardInterrupt:
        raise SystemExit("interrupted")
