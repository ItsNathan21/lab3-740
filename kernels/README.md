# Kernels

Source code and pre-built variants for every kernel in the library.

## Layout

```
kernels/
└── <kernel-name>/
    ├── README.md       ← what this kernel does, parameter space
    ├── cpu/            ← CPU sources + pre-built .so per variant
    ├── gpu/            ← OpenCL / Vulkan sources + binaries
    ├── dsp/            ← Hexagon sources + .so per HVX-thread-count variant
    └── npu/            ← QNN graph sources + compiled .bin per precision
```


## Adding a new kernel

(to be written once Phase 3 produces the first kernel)

## Rebuilding variants

(to be written once the build pipeline lands in Phase 3)
