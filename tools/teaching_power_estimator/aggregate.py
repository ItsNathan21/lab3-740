"""HDK8750 teaching power estimator — host-side aggregation (PURE STDLIB).

Turns *raw* device telemetry samples into the validated per-window aggregate
features the fit/estimate consume. All arithmetic (jiffie deltas, EAS lookup,
GPU basis, duration weighting) happens HERE, not on the Android shell — the
calibration session is the scarce resource, so raw samples are preserved and
aggregation is repeatable/inspectable off-device (review P0-6).

Feature definitions (per window, duration-weighted over sample intervals):
    activity_fraction = mean_t[ workload_active(t) ]      (workload intent, union for concurrent)
    X_c   = mean_t[ EAS_power_W(cluster, f_c(t)) * sum_i util_i(t) ]   (util from /proc/stat deltas)
    G_j   = mean_t[ I_GPU(t) * busy(t) * phi_j(f_gpu(t)) ]            (busy in [0,1])
    N     = mean_t[ I_NPU_path(t) * workload_active(t) ]
    D     = mean_t[ I_DSP_path(t) * workload_active(t) ]

NPU and DSP are two mutually-exclusive software paths onto the same Hexagon
cDSP.  The path identity is declared by immutable window provenance; the
duration comes from the same sampled workload-intent signal as the floor term.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from .model import ModelError, gpu_ramp_basis

# CPU scaling freq must land within this of a real EAS OPP, else the read is
# treated as failed/blank and the window is rejected (fail-closed).
EAS_OPP_TOL_KHZ = 20_000  # 20 MHz

# Cluster -> (EAS key, cpu ids). Gold = policy0 (cpu0-5), Prime = policy6 (cpu6-7).
DEFAULT_CLUSTER_CPUS = {"gold": (0, 1, 2, 3, 4, 5), "prime": (6, 7)}
CLUSTER_EAS_KEY = {"gold": "cpu0", "prime": "cpu6"}

# GPU frequency treated as "held" if its spread across samples is under this.
GPU_FREQ_HELD_TOL_HZ = 5_000_000.0  # 5 MHz

# A second lab block must demonstrate that its idle supply boundary still
# matches the immutable v1 block.  These are cheap meter-only bridge checks,
# not additional fitted windows.
EXTENSION_IDLE_POSITIONS = ("start", "middle", "end")
IDLE_BRIDGE_REJECT_FRAC = 0.20

# Hexagon calibration plateaus are rejected if a relevant sensor traverses a
# large thermal range.  This is a scope guard, not a power-model feature.
MAX_HEXAGON_PLATEAU_TEMP_SPAN_C = 10.0


# --- /proc/stat ---------------------------------------------------------------

def parse_proc_stat_cpu_lines(text: str) -> dict[int, list[int]]:
    """Parse per-core lines ('cpu0 u n s idle iowait irq softirq steal ...').

    Returns {cpu_id: [jiffie fields]}. Ignores the aggregate 'cpu ' line.
    """
    out: dict[int, list[int]] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("cpu"):
            continue
        tag = parts[0]
        if tag == "cpu":  # aggregate line, skip
            continue
        try:
            cpu_id = int(tag[3:])
        except ValueError:
            continue
        try:
            fields = [int(x) for x in parts[1:]]
        except ValueError as e:
            raise ModelError(f"bad /proc/stat line {line!r}: {e}")
        if len(fields) < 7:
            raise ModelError(f"/proc/stat line {tag} has too few fields: {line!r}")
        out[cpu_id] = fields
    return out


def proc_stat_of(sample: dict[str, Any]) -> dict[int, list[int]]:
    """Return int-keyed per-cpu jiffies for a sample.

    Accepts either ``proc_stat_raw`` (raw /proc/stat text, as the device shell
    emits) or a pre-parsed ``proc_stat`` dict (keys may be int or str after a
    JSON round-trip).
    """
    if "proc_stat_raw" in sample:
        return parse_proc_stat_cpu_lines(sample["proc_stat_raw"])
    ps = sample.get("proc_stat")
    if ps is None:
        raise ModelError(f"sample {sample.get('sample_index')!r} missing proc_stat/proc_stat_raw")
    return {int(k): [int(x) for x in v] for k, v in ps.items()}


def busy_total(jiffies: list[int]) -> tuple[int, int]:
    """busy = user+nice+system+irq+softirq+steal; total = busy+idle+iowait.

    Guest / guest_nice are deliberately excluded (already counted in user/nice).
    """
    user, nice, system, idle, iowait, irq, softirq = jiffies[:7]
    steal = jiffies[7] if len(jiffies) > 7 else 0
    busy = user + nice + system + irq + softirq + steal
    total = busy + idle + iowait
    return busy, total


# --- EAS table ----------------------------------------------------------------

def parse_eas_dump(text: str) -> dict[str, list[tuple[int, float]]]:
    """Parse raw EM dump lines 'EM <cluster> freq=<kHz> power_uW=<uW>'.

    Returns {cluster_key: [(freq_khz, power_w), ...]} sorted by freq. µW->W here.
    """
    out: dict[str, list[tuple[int, float]]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] != "EM":
            continue
        cluster = parts[1]
        freq_khz = power_uw = None
        for tok in parts[2:]:
            if tok.startswith("freq="):
                freq_khz = int(tok.split("=", 1)[1])
            elif tok.startswith("power_uW="):
                power_uw = int(tok.split("=", 1)[1])
        if freq_khz is None or power_uw is None:
            continue
        out.setdefault(cluster, []).append((freq_khz, power_uw / 1e6))
    for k in out:
        out[k].sort(key=lambda t: t[0])
    return out


def eas_power_w(table: dict[str, list[tuple[int, float]]], eas_key: str, freq_khz: int) -> float:
    """Nearest-OPP EAS power (W) for a cluster at a given frequency (kHz).

    Fail-closed: if the frequency is not within ``EAS_OPP_TOL_KHZ`` of a real
    OPP (e.g. a blank read parsed as 0, or an unexpected freq), reject rather
    than silently snapping to the lowest OPP.
    """
    rows = table.get(eas_key)
    if not rows:
        raise ModelError(f"no EAS table for {eas_key!r}")
    best = min(rows, key=lambda t: abs(t[0] - freq_khz))
    if abs(best[0] - freq_khz) > EAS_OPP_TOL_KHZ:
        raise ModelError(
            f"{eas_key} freq {freq_khz} kHz not within {EAS_OPP_TOL_KHZ} kHz of any OPP "
            f"(nearest {best[0]}) — likely a failed/blank read; window rejected"
        )
    return best[1]


# --- window aggregation -------------------------------------------------------

def aggregate_window(
    samples: list[dict[str, Any]],
    eas_table: dict[str, list[tuple[int, float]]],
    knots_hz: list[float],
    *,
    gpu_active: bool,
    npu_active: bool = False,
    dsp_active: bool = False,
    cluster_cpus: dict[str, tuple[int, ...]] = DEFAULT_CLUSTER_CPUS,
) -> dict[str, Any]:
    """Reduce raw samples for one window to aggregate features.

    Each sample dict must have: sample_index, t_before, boot_id, cpu_online_mask,
    proc_stat {cpu_id: [jiffies]}, policy0_khz, policy6_khz, gpuclk_hz,
    gpu_busy_pct, workload_active (0/1), and optionally temps_c.

    Raises ModelError on any integrity problem (review P0-6 reject rules).
    """
    if len(samples) < 2:
        raise ModelError(f"window needs >= 2 samples for jiffie deltas, got {len(samples)}")
    for name, flag in (
        ("gpu_active", gpu_active),
        ("npu_active", npu_active),
        ("dsp_active", dsp_active),
    ):
        if type(flag) is not bool:
            raise ModelError(f"{name} path declaration must be boolean, got {flag!r}")
    if npu_active and dsp_active:
        raise ModelError(
            "window cannot declare both NPU and DSP active: both paths share and "
            "hardware-serialize on the Hexagon cDSP"
        )

    boot_ids = {s.get("boot_id") for s in samples}
    if len(boot_ids) != 1 or next(iter(boot_ids), None) in (None, ""):
        raise ModelError(f"boot_id changed within window: {boot_ids!r}")
    masks = {s.get("cpu_online_mask") for s in samples}
    if len(masks) != 1 or next(iter(masks), None) in (None, ""):
        raise ModelError(f"cpu_online_mask changed within window: {masks!r}")

    # workload liveness: if the controlled workload was marked active in any
    # sample but its process was not alive, the window is contaminated (finding 4).
    for s in samples:
        if "workload_active" not in s or int(s["workload_active"]) not in (0, 1):
            raise ModelError(
                f"sample {s.get('sample_index')!r} missing/non-binary workload_active"
            )
        if "pid_alive" in s and int(s["pid_alive"]) not in (0, 1):
            raise ModelError(f"sample {s.get('sample_index')!r} has non-binary pid_alive")
        t_before = float(s.get("t_before", float("nan")))
        if not math.isfinite(t_before):
            raise ModelError(f"sample {s.get('sample_index')!r} has invalid t_before")
        if "t_after" in s:
            t_after = float(s["t_after"])
            if not math.isfinite(t_after) or t_after < t_before:
                raise ModelError(f"sample {s.get('sample_index')!r} has invalid t_after")
        if int(s.get("workload_active", 0)) == 1 and int(s.get("pid_alive", 1)) == 0:
            raise ModelError(
                f"workload_active=1 but pid_alive=0 at sample {s.get('sample_index')!r} "
                f"— workload died during window; rejected"
            )

    expected_cpus = sorted(c for cpus in cluster_cpus.values() for c in cpus)

    x_acc = {"gold": 0.0, "prime": 0.0}
    # Frequency-weighted CPU accumulators for the cpu_gpu_freq profile (ADR 0023).
    # `cluster` is max-utilisation x GHz -- the cost of having the cluster clocked
    # at all; `percore` is summed-utilisation x GHz. Both are accumulated
    # unconditionally so one aggregate serves either profile.
    xf_cluster = {"gold": 0.0, "prime": 0.0}
    xf_percore = {"gold": 0.0, "prime": 0.0}
    g_acc = [0.0, 0.0, 0.0]
    a_acc = 0.0
    npu_acc = 0.0
    dsp_acc = 0.0
    w_sum = 0.0
    gpu_freqs: list[float] = []
    gpu_busy: list[float] = []
    temperature_samples: dict[str, list[float]] = {}
    cooling_samples: dict[str, list[int]] = {}

    for sample in samples:
        temps = sample.get("temps_c", {})
        if temps is not None and not isinstance(temps, dict):
            raise ModelError(
                f"sample {sample.get('sample_index')!r} temps_c must be an object"
            )
        for name, raw_value in (temps or {}).items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ModelError(
                    f"sample {sample.get('sample_index')!r} temperature {name!r} "
                    f"is not numeric"
                ) from exc
            if not math.isfinite(value):
                raise ModelError(
                    f"sample {sample.get('sample_index')!r} temperature {name!r} "
                    "is non-finite"
                )
            temperature_samples.setdefault(str(name), []).append(value)

        cooling = sample.get("cooling_states", {})
        if cooling is not None and not isinstance(cooling, dict):
            raise ModelError(
                f"sample {sample.get('sample_index')!r} cooling_states must be an object"
            )
        for name, raw_value in (cooling or {}).items():
            if isinstance(raw_value, bool):
                raise ModelError(
                    f"sample {sample.get('sample_index')!r} cooling state {name!r} "
                    "must be an integer"
                )
            try:
                value = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ModelError(
                    f"sample {sample.get('sample_index')!r} cooling state {name!r} "
                    "is not an integer"
                ) from exc
            if value < 0:
                raise ModelError(
                    f"sample {sample.get('sample_index')!r} cooling state {name!r} "
                    "is negative"
                )
            cooling_samples.setdefault(str(name), []).append(value)

    for k in range(len(samples) - 1):
        s0, s1 = samples[k], samples[k + 1]
        dt = float(s1["t_before"]) - float(s0["t_before"])
        if not math.isfinite(dt) or dt <= 0.0:
            raise ModelError(f"nonpositive interval duration dt={dt!r} at sample {k}")

        ps0 = proc_stat_of(s0)
        ps1 = proc_stat_of(s1)
        for cpu in expected_cpus:
            if cpu not in ps0 or cpu not in ps1:
                raise ModelError(f"cpu{cpu} missing from /proc/stat at interval {k}")

        # per-cluster EAS-weighted summed utilization for this interval
        for cluster, cpus in cluster_cpus.items():
            freq_khz = int(s0["policy0_khz"] if cluster == "gold" else s0["policy6_khz"])
            p_w = eas_power_w(eas_table, CLUSTER_EAS_KEY[cluster], freq_khz)
            sum_util = 0.0
            max_util = 0.0
            for cpu in cpus:
                b0, t0 = busy_total(ps0[cpu])
                b1, t1 = busy_total(ps1[cpu])
                db, dtot = b1 - b0, t1 - t0
                if dtot <= 0:
                    raise ModelError(f"nonpositive total jiffie delta cpu{cpu} interval {k}")
                if db < 0:
                    raise ModelError(f"negative busy jiffie delta cpu{cpu} interval {k}")
                u = db / dtot
                if u < 0.0 or u > 1.0 + 1e-6:
                    raise ModelError(f"utilization out of [0,1]: cpu{cpu}={u!r}")
                sum_util += min(u, 1.0)
                max_util = max(max_util, min(u, 1.0))
            x_acc[cluster] += dt * (p_w * sum_util)
            f_ghz = freq_khz / 1e6
            xf_cluster[cluster] += dt * (max_util * f_ghz)
            xf_percore[cluster] += dt * (sum_util * f_ghz)

        # GPU basis for this interval
        f_gpu = float(s0["gpuclk_hz"])
        busy_pct = float(s0["gpu_busy_pct"])
        if not math.isfinite(f_gpu) or f_gpu <= 0.0:
            raise ModelError(f"invalid gpuclk_hz {f_gpu!r} at interval {k}")
        if not math.isfinite(busy_pct) or not (-0.001 <= busy_pct <= 100.001):
            raise ModelError(f"gpu_busy_pct out of [0,100]: {busy_pct!r} at interval {k} — rejected")
        busy = min(max(busy_pct / 100.0, 0.0), 1.0)
        ind = 1.0 if gpu_active else 0.0
        phi0, phi1, phi2 = gpu_ramp_basis(f_gpu, knots_hz)
        g_acc[0] += dt * (ind * busy * phi0)
        g_acc[1] += dt * (ind * busy * phi1)
        g_acc[2] += dt * (ind * busy * phi2)
        gpu_freqs.append(f_gpu)
        gpu_busy.append(busy)

        workload_intent = float(s0["workload_active"])
        if workload_intent not in (0.0, 1.0):
            raise ModelError(
                f"workload_active must be binary, got {workload_intent!r} at interval {k}"
            )
        a_acc += dt * workload_intent
        npu_acc += dt * workload_intent * (1.0 if npu_active else 0.0)
        dsp_acc += dt * workload_intent * (1.0 if dsp_active else 0.0)
        w_sum += dt

    if w_sum <= 0.0:
        raise ModelError("zero total window duration")

    gpu_freq_spread = (max(gpu_freqs) - min(gpu_freqs)) if gpu_freqs else 0.0
    agg = {
        "activity_fraction": a_acc / w_sum,
        "x_gold": x_acc["gold"] / w_sum,
        "x_prime": x_acc["prime"] / w_sum,
        "xf_gold_cluster": xf_cluster["gold"] / w_sum,
        "xf_gold_percore": xf_percore["gold"] / w_sum,
        "xf_prime_cluster": xf_cluster["prime"] / w_sum,
        "xf_prime_percore": xf_percore["prime"] / w_sum,
        "gpu_g0": g_acc[0] / w_sum,
        "gpu_g1": g_acc[1] / w_sum,
        "gpu_g2": g_acc[2] / w_sum,
        "npu_active": npu_acc / w_sum,
        "dsp_active": dsp_acc / w_sum,
        "gpu_active": bool(gpu_active),
        "gpu_freq_hz": (sum(gpu_freqs) / len(gpu_freqs)) if gpu_freqs else None,
        "gpu_busy_mean": (sum(gpu_busy) / len(gpu_busy)) if gpu_busy else None,
        "gpu_dynamic_frequency_approximation": bool(gpu_active and gpu_freq_spread > GPU_FREQ_HELD_TOL_HZ),
        "window_s": w_sum,
        "t_start": float(samples[0]["t_before"]),
        "t_end": float(samples[-1].get("t_after", samples[-1]["t_before"])),
        "n_samples": len(samples),
        "n_intervals": len(samples) - 1,
        "temperature_ranges_c": {
            name: {
                "min_c": min(values),
                "max_c": max(values),
                "n_samples": len(values),
            }
            for name, values in sorted(temperature_samples.items())
        },
        "cooling_state_ranges": {
            name: {"min": min(values), "max": max(values), "n_samples": len(values)}
            for name, values in sorted(cooling_samples.items())
        },
    }
    return agg


# --- session loading (shared by both CLIs; stdlib) ----------------------------

def eas_tables_from_session(session: dict[str, Any]) -> dict[str, list[tuple[int, float]]]:
    """session['eas_tables'] = {'cpu0': [[freq_khz, power_w], ...], 'cpu6': [...]}."""
    raw_tables = session.get("eas_tables")
    if not isinstance(raw_tables, dict) or not raw_tables:
        raise ModelError("session.eas_tables must be a non-empty object")
    out: dict[str, list[tuple[int, float]]] = {}
    try:
        for key, rows in raw_tables.items():
            if not isinstance(key, str) or not isinstance(rows, list) or not rows:
                raise ModelError(f"invalid EAS table entry {key!r}")
            parsed = [(int(f), float(p)) for f, p in rows]
            if any(freq <= 0 or not math.isfinite(power) or power <= 0.0 for freq, power in parsed):
                raise ModelError(f"EAS table {key!r} contains invalid frequency/power")
            out[key] = parsed
    except (TypeError, ValueError) as exc:
        raise ModelError(f"malformed session.eas_tables: {exc}") from exc
    return out


EXPECTED_METER_SAMPLES = 5
# Video/panel-read reality: 1-5 accepted. The actual count and spread are recorded
# per window; a single read means jitter is UNKNOWN for that window (not zero).
MIN_METER_SAMPLES = 1


def window_measured_power_w(window: dict[str, Any]) -> tuple[float, float, float]:
    """P_window = mean_r(V_r * I_r) from the DP831A reads. Fail-closed protocol.

    Requires ``MIN_METER_SAMPLES``..``EXPECTED_METER_SAMPLES`` samples (reading a
    jittering panel off video realistically yields 3-5; the actual count and the
    spread are recorded rather than silently assumed), each with explicit
    ``mode == "CV"`` and finite positive V/I (finding 6). Returns
    ``(mean_w, min_w, max_w)`` so callers can surface the spread cheaply.
    """
    wid = window.get("window_id")
    ms = window.get("meter_samples") or []
    if not (MIN_METER_SAMPLES <= len(ms) <= EXPECTED_METER_SAMPLES):
        raise ModelError(
            f"window {wid!r} needs {MIN_METER_SAMPLES}-{EXPECTED_METER_SAMPLES} meter_samples, got {len(ms)}")
    powers = []
    for s in ms:
        if str(s.get("mode")) != "CV":
            raise ModelError(f"window {wid!r} meter sample mode {s.get('mode')!r} != 'CV' — rejected")
        v = float(s["voltage_v"])
        i = float(s["current_a"])
        if not (math.isfinite(v) and math.isfinite(i) and v > 0.0 and i > 0.0):
            raise ModelError(f"window {wid!r} non-finite/non-positive meter V/I: V={v!r} I={i!r}")
        powers.append(v * i)
    return sum(powers) / len(powers), min(powers), max(powers)


def validate_extension_idle_checks(
    checks: Any,
    *,
    reference_power_w: float | None = None,
    reject_fraction: float = IDLE_BRIDGE_REJECT_FRAC,
) -> dict[str, Any]:
    """Validate start/middle/end meter-only bridge readings for a new lab block.

    The checks are deliberately not fitted rows.  They detect the otherwise
    unidentifiable case where a lab-day quiescent offset is absorbed into every
    newly fitted Hexagon path step.
    """
    if not isinstance(checks, list) or len(checks) != len(EXTENSION_IDLE_POSITIONS):
        raise ModelError(
            "extension_idle_checks must contain exactly start/middle/end entries"
        )
    means: list[float] = []
    ranges: list[list[float]] = []
    for expected_position, check in zip(EXTENSION_IDLE_POSITIONS, checks):
        if not isinstance(check, dict):
            raise ModelError("every extension idle check must be an object")
        if check.get("position") != expected_position:
            raise ModelError(
                "extension_idle_checks positions must be ordered exactly "
                f"{list(EXTENSION_IDLE_POSITIONS)!r}"
            )
        samples = check.get("meter_samples")
        if not isinstance(samples, list) or len(samples) != EXPECTED_METER_SAMPLES:
            raise ModelError(
                f"extension idle check {expected_position!r} requires exactly "
                f"{EXPECTED_METER_SAMPLES} meter samples"
            )
        mean_w, min_w, max_w = window_measured_power_w(
            {"window_id": f"extension_idle_{expected_position}", "meter_samples": samples}
        )
        means.append(mean_w)
        ranges.append([min_w, max_w])

    median_w = float(statistics.median(means))
    drift_fraction = (max(means) - min(means)) / median_w
    if drift_fraction > reject_fraction:
        raise ModelError(
            f"extension idle bridge drift {drift_fraction:.1%} exceeds "
            f"{reject_fraction:.0%}: {means!r}"
        )

    max_reference_deviation = None
    if reference_power_w is not None:
        try:
            reference = float(reference_power_w)
        except (TypeError, ValueError) as exc:
            raise ModelError("idle bridge reference power must be numeric") from exc
        if not math.isfinite(reference) or reference <= 0.0:
            raise ModelError("idle bridge reference power must be finite and positive")
        max_reference_deviation = max(abs(value - reference) / reference for value in means)
        if max_reference_deviation > reject_fraction:
            raise ModelError(
                f"extension idle bridge differs from immutable base P_q by up to "
                f"{max_reference_deviation:.1%} (> {reject_fraction:.0%}); "
                "lab-block offset would be confounded with Hexagon path power"
            )

    return {
        "positions": list(EXTENSION_IDLE_POSITIONS),
        "mean_power_w": means,
        "meter_power_ranges_w": ranges,
        "median_power_w": median_w,
        "within_block_drift_fraction": drift_fraction,
        "max_reference_deviation_fraction": max_reference_deviation,
        "reject_fraction": reject_fraction,
        "passed": True,
        "used_as_fit_rows": False,
    }


_SCALAR_INT = {
    "policy0_khz", "policy6_khz",
    "policy0_min_khz", "policy0_max_khz", "policy6_min_khz", "policy6_max_khz",
    "gpuclk_hz",
    "gpu_min_pwrlevel", "gpu_max_pwrlevel", "gpu_thermal_pwrlevel",
    "workload_active", "pid_alive",
}


def parse_raw_window(text: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Parse a device sampler ``<window>.raw`` file into (header, samples).

    Block format emitted by scripts/power_calib_sampler.sh:
        =WINDOW <id> hz=.. duration_s=.. boot_id=..
        =SAMPLE <idx> <t_before> <t_after>
        =SCALARS k=v ...
        =TEMPS name=millideg ...
        =STAT
        <raw /proc/stat lines>
        =ENDSTAT
    Each sample dict is directly consumable by ``aggregate_window``.
    """
    header: dict[str, str] = {}
    samples: list[dict[str, Any]] = []
    boot_id = None
    header_seen = False
    cur: dict[str, Any] | None = None
    stat: list[str] | None = None
    seen_blocks: set[str] = set()
    for line in text.splitlines():
        if line.startswith("=WINDOW"):
            if header_seen or cur is not None or samples:
                raise ModelError("raw file must contain exactly one leading =WINDOW header")
            header_seen = True
            parts = line.split()
            if len(parts) >= 2 and "=" not in parts[1]:
                header["window_id"] = parts[1]
            for tok in parts[2:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k in header:
                        raise ModelError(f"duplicate raw header key {k!r}")
                    header[k] = v
            boot_id = header.get("boot_id")
        elif line.startswith("=SAMPLE"):
            if stat is not None:
                raise ModelError("new =SAMPLE encountered before =ENDSTAT")
            if not header_seen:
                raise ModelError("=SAMPLE before =WINDOW")
            if cur is not None:
                samples.append(cur)
            parts = line.split()
            if len(parts) != 4:
                raise ModelError(f"malformed =SAMPLE line {line!r}")
            cur = {
                "sample_index": int(parts[1]),
                "t_before": float(parts[2]),
                "t_after": float(parts[3]),
                "boot_id": boot_id,
            }
            stat = None
            seen_blocks = set()
        elif line.startswith("=SCALARS"):
            if cur is None:
                raise ModelError("=SCALARS before =SAMPLE")
            if stat is not None:
                raise ModelError("=SCALARS encountered inside =STAT block")
            if "SCALARS" in seen_blocks:
                raise ModelError(f"duplicate =SCALARS block in sample {cur.get('sample_index')!r}")
            seen_blocks.add("SCALARS")
            for tok in line.split()[1:]:
                if "=" not in tok:
                    continue
                k, v = tok.split("=", 1)
                normalized_key = "cpu_online_mask" if k == "online" else k
                if normalized_key in cur:
                    raise ModelError(
                        f"duplicate scalar key {k!r} in sample {cur.get('sample_index')!r}"
                    )
                blank = v in ("", "None")
                if k == "online":
                    if blank:
                        raise ModelError(f"blank online mask in sample {cur.get('sample_index')!r}")
                    cur["cpu_online_mask"] = v
                elif k == "gpu_busy_pct":
                    if blank:
                        raise ModelError(f"blank gpu_busy_pct in sample {cur.get('sample_index')!r}")
                    cur["gpu_busy_pct"] = float(v)
                elif k in _SCALAR_INT:
                    if blank:
                        raise ModelError(f"blank {k} in sample {cur.get('sample_index')!r} — failed read, window rejected")
                    cur[k] = int(v)
                else:
                    if k in ("policy0_gov", "policy6_gov") and blank:
                        raise ModelError(
                            f"blank {k} in sample {cur.get('sample_index')!r} — failed read"
                        )
                    cur[k] = v
        elif line.startswith("=TEMPS"):
            if cur is None:
                raise ModelError("=TEMPS before =SAMPLE")
            if stat is not None:
                raise ModelError("=TEMPS encountered inside =STAT block")
            if "TEMPS" in seen_blocks:
                raise ModelError(f"duplicate =TEMPS block in sample {cur.get('sample_index')!r}")
            seen_blocks.add("TEMPS")
            temps: dict[str, float] = {}
            for tok in line.split()[1:]:
                if "=" not in tok:
                    continue
                k, v = tok.split("=", 1)
                if k in temps:
                    raise ModelError(f"duplicate temperature key {k!r}")
                try:
                    temps[k] = int(v) / 1000.0
                except ValueError as exc:
                    raise ModelError(
                        f"invalid temperature token {tok!r} in sample "
                        f"{cur.get('sample_index')!r}"
                    ) from exc
            cur["temps_c"] = temps
        elif line.startswith("=COOLING"):
            if cur is None:
                raise ModelError("=COOLING before =SAMPLE")
            if stat is not None:
                raise ModelError("=COOLING encountered inside =STAT block")
            if "COOLING" in seen_blocks:
                raise ModelError(f"duplicate =COOLING block in sample {cur.get('sample_index')!r}")
            seen_blocks.add("COOLING")
            cooling: dict[str, int] = {}
            for tok in line.split()[1:]:
                if "=" not in tok:
                    continue
                k, v = tok.split("=", 1)
                if k in cooling:
                    raise ModelError(f"duplicate cooling key {k!r}")
                try:
                    cooling[k] = int(v)
                except ValueError as exc:
                    raise ModelError(
                        f"invalid cooling token {tok!r} in sample "
                        f"{cur.get('sample_index')!r}"
                    ) from exc
            cur["cooling_states"] = cooling
        elif line == "=STAT":
            if cur is None:
                raise ModelError("=STAT before =SAMPLE")
            if stat is not None or "STAT" in seen_blocks:
                raise ModelError(f"duplicate/nested =STAT block in sample {cur.get('sample_index')!r}")
            seen_blocks.add("STAT")
            stat = []
        elif line == "=ENDSTAT":
            if cur is None or stat is None:
                raise ModelError("=ENDSTAT without an open =STAT block")
            cur["proc_stat_raw"] = "\n".join(stat)
            stat = None
        elif stat is not None:
            stat.append(line)
        elif line.strip():
            raise ModelError(f"unrecognized raw line {line!r}")
    if stat is not None:
        raise ModelError("raw file ended before =ENDSTAT")
    if cur is not None:
        samples.append(cur)

    # completeness / integrity of the parsed window (fail-closed, finding 3)
    if not header.get("window_id"):
        raise ModelError("raw file missing a =WINDOW header with a window id")
    if not header.get("boot_id"):
        raise ModelError("raw header missing boot_id")
    if header.get("expect_active") not in ("0", "1"):
        raise ModelError("raw header expect_active must be explicitly 0 or 1")
    try:
        header_hz = float(header["hz"])
        header_duration = float(header["duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelError("raw header requires numeric hz and duration_s") from exc
    if not (
        math.isfinite(header_hz)
        and header_hz > 0.0
        and math.isfinite(header_duration)
        and header_duration > 0.0
    ):
        raise ModelError("raw header hz and duration_s must be finite and positive")
    if len(samples) < 2:
        raise ModelError(f"raw window needs at least two complete samples, got {len(samples)}")
    expected_active = int(header["expect_active"])
    _required = (
        "proc_stat_raw", "policy0_khz", "policy6_khz", "gpuclk_hz", "gpu_busy_pct",
        "policy0_min_khz", "policy0_max_khz", "policy0_gov",
        "policy6_min_khz", "policy6_max_khz", "policy6_gov",
        "cpu_online_mask", "workload_active", "pid_alive",
        "gpu_min_pwrlevel", "gpu_max_pwrlevel", "gpu_thermal_pwrlevel",
        "temps_c", "cooling_states",
    )
    for i, s in enumerate(samples):
        if s.get("sample_index") != i:
            raise ModelError(f"non-sequential sample_index at position {i}: {s.get('sample_index')!r}")
        missing = [k for k in _required if k not in s]
        if missing:
            raise ModelError(f"sample {i} missing required fields {missing} (incomplete block)")
        if not s.get("cpu_online_mask"):
            raise ModelError(f"sample {i} empty cpu_online_mask")
        if int(s["workload_active"]) not in (0, 1) or int(s["pid_alive"]) not in (0, 1):
            raise ModelError(f"sample {i} non-binary workload_active/pid_alive")
        if int(s["workload_active"]) != expected_active:
            raise ModelError(
                f"sample {i} workload_active={s['workload_active']} does not match "
                f"raw header expect_active={expected_active}"
            )
        if expected_active == 1 and int(s["pid_alive"]) != 1:
            raise ModelError(f"sample {i} expected-active workload PID is not alive")
        if not (math.isfinite(float(s["t_before"])) and math.isfinite(float(s["t_after"]))):
            raise ModelError(f"sample {i} has non-finite timestamps")
        if float(s["t_after"]) < float(s["t_before"]):
            raise ModelError(f"sample {i} t_after < t_before")
    return header, samples


def records_from_session(session: dict[str, Any], knots_hz: list[float]) -> list[dict[str, Any]]:
    """Turn a row-oriented session/run dict into fit/estimate window records.

    Each record: {window_id, split, engine_class, aggregate[, measured_power_w]}.
    ``measured_power_w`` is included only when the window carries meter_samples.
    """
    if not isinstance(session, dict):
        raise ModelError("session root must be an object")
    windows = session.get("windows")
    if not isinstance(windows, list):
        raise ModelError("session.windows must be a list")
    eas = eas_tables_from_session(session)
    recs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    allowed_engines = {"idle", "floor", "cpu", "gpu", "npu", "dsp", "combined"}
    allowed_splits = {"idle_anchor", "train", "holdout", "run"}
    for w in windows:
        if not isinstance(w, dict):
            raise ModelError("every session window must be an object")
        window_id = w.get("window_id")
        if not isinstance(window_id, str) or not window_id:
            raise ModelError(f"window has invalid/missing window_id {window_id!r}")
        if window_id in seen_ids:
            raise ModelError(f"duplicate window_id {window_id!r}")
        seen_ids.add(window_id)
        if not isinstance(w.get("telemetry_samples"), list):
            raise ModelError(f"window {window_id!r} telemetry_samples must be a list")
        cond = w.get("condition", {})
        if not isinstance(cond, dict):
            raise ModelError(f"window {w.get('window_id')!r} condition must be an object")
        engine = cond.get("engine_class")
        split = w.get("split")
        if engine not in allowed_engines:
            raise ModelError(
                f"window {w.get('window_id')!r} has unsupported/missing engine_class {engine!r}"
            )
        if split not in allowed_splits:
            raise ModelError(
                f"window {w.get('window_id')!r} has unsupported/missing split {split!r}"
            )

        def declared_flag(key: str) -> bool:
            value = cond.get(key, False)
            if type(value) is not bool:
                raise ModelError(
                    f"window {w.get('window_id')!r} condition.{key} must be boolean, "
                    f"got {value!r}"
                )
            return value

        gpu_declared = declared_flag("gpu_active")
        npu_declared = declared_flag("npu_active")
        dsp_declared = declared_flag("dsp_active")
        if engine in {"idle", "floor", "cpu"} and (
            gpu_declared or npu_declared or dsp_declared
        ):
            raise ModelError(
                f"window {window_id!r} engine_class={engine!r} cannot declare an "
                "accelerator path active"
            )
        if engine == "gpu" and (npu_declared or dsp_declared):
            raise ModelError(
                f"window {window_id!r} engine_class='gpu' cannot also declare a "
                "Hexagon path; use engine_class='combined'"
            )
        if engine == "npu" and dsp_declared:
            raise ModelError(f"window {window_id!r} NPU path contradicts dsp_active=true")
        if engine == "dsp" and npu_declared:
            raise ModelError(f"window {window_id!r} DSP path contradicts npu_active=true")
        if engine in ("gpu", "combined") and "gpu_active" in cond and not gpu_declared:
            raise ModelError(
                f"window {w.get('window_id')!r} engine_class={engine!r} contradicts "
                "condition.gpu_active=false"
            )
        gpu_active = engine in ("gpu", "combined") or gpu_declared
        if engine == "npu" and "npu_active" in cond and not npu_declared:
            raise ModelError(
                f"window {w.get('window_id')!r} engine_class='npu' contradicts "
                "condition.npu_active=false"
            )
        if engine == "dsp" and "dsp_active" in cond and not dsp_declared:
            raise ModelError(
                f"window {w.get('window_id')!r} engine_class='dsp' contradicts "
                "condition.dsp_active=false"
            )
        npu_active = engine == "npu" or npu_declared
        dsp_active = engine == "dsp" or dsp_declared
        agg = aggregate_window(
            w["telemetry_samples"],
            eas,
            knots_hz,
            gpu_active=gpu_active,
            npu_active=npu_active,
            dsp_active=dsp_active,
        )
        rec: dict[str, Any] = {
            "window_id": window_id,
            "split": split,
            "engine_class": engine,
            "perf_mode": cond.get("perf_mode"),
            "dsp_power_mode": cond.get("dsp_power_mode"),
            "aggregate": agg,
        }
        if split in {"idle_anchor", "train", "holdout"} and not w.get("meter_samples"):
            raise ModelError(
                f"calibration window {window_id!r} is missing meter_samples"
            )
        if w.get("meter_samples"):
            mean_w, min_w, max_w = window_measured_power_w(w)
            rec["measured_power_w"] = mean_w
            rec["measured_power_range_w"] = [min_w, max_w]
        recs.append(rec)
    return recs
