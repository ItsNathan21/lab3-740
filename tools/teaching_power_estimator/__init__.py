"""HDK8750 teaching power estimator (external, opt-in, offline).

A coarse, clearly-labeled *ballpark* model of total 12 V supply-input power for
the SM8750 HDK, for CMU 18-740. NOT a measurement, NOT a physical per-engine
decomposition. Deliberately lives OUTSIDE the profiler library (`src/host/`)
because the library forbids inference/prediction; see
`decisions/0019-teaching-power-estimate.md` and its Hexagon extension ADR 0020.

Public surface:
  - model.py   (stdlib): PowerModelArtifact, estimate_power, integrate_energy
  - aggregate.py (stdlib): aggregate_window + raw parsers
  - fit.py     (numpy, offline only): fit_power_model, evaluate_power_model
"""

from __future__ import annotations

from .model import (
    ARTIFACT_SCHEMA_VERSION,
    BOUNDARY,
    FEATURE_ORDER,
    MODEL_EQUATION_VERSION,
    QUALITY,
    SCOPE,
    ModelError,
    PowerModelArtifact,
    estimate_power,
    feature_vector,
    gpu_ramp_basis,
    integrate_energy,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BOUNDARY",
    "FEATURE_ORDER",
    "MODEL_EQUATION_VERSION",
    "QUALITY",
    "SCOPE",
    "ModelError",
    "PowerModelArtifact",
    "estimate_power",
    "feature_vector",
    "gpu_ramp_basis",
    "integrate_energy",
]
