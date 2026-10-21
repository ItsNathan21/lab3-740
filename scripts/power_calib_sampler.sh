#!/system/bin/sh
# HDK8750 teaching-power calibration RAW sampler (on-device).
#
# READ-ONLY by design: it mutates no sysfs, so there is nothing to restore and
# it cannot brick the board. Frequency/pwrlevel PINNING is done host-side by the
# ReproducibilityController (src/host/orchestrator/control) per the run-sheet;
# this sampler just records the ACTUAL control state it observes, at ~1 Hz, as
# raw blocks. All arithmetic (jiffie deltas, EAS lookup, GPU basis) happens on
# the host in tools/teaching_power_estimator/aggregate.py (review P0-6).
#
# Usage (run detached as root; see scripts/power_build_session.py to assemble):
#   su 0 nohup sh /data/local/tmp/power_calib_sampler.sh \
#       <window_id> <duration_s> [hz] [outdir] <expect_active:0|1> \
#       <engine_class:idle|floor|cpu|gpu|npu|dsp> &
#
# workload_active is read from a flag file the workload wrapper creates while the
# controlled workload is running: <outdir>/WL_ACTIVE  (present => 1).  This is
# workload INTENT, not inferred from utilization/busy (review P0-3).
#
# Emits (raw, no arithmetic):
#   <outdir>/eas.txt                 EAS energy_model table (dumped once)
#   <outdir>/<window_id>.raw         per-sample blocks (see format below)
#   <outdir>/<window_id>.clockstats  gpu_clock_stats before + after the window
#   <outdir>/<window_id>.runtime-provenance.json  NPU/DSP provision snapshot
#   <outdir>/<window_id>.runtime-artifacts.sha256 NPU/DSP observed hashes
#   <outdir>/<window_id>.done|.fail  completion / failure sentinel

set -u
WINDOW_ID="${1:?window_id required}"
DURATION_S="${2:?duration_s required}"
HZ="${3:-1}"
OUTDIR="${4:-/data/local/tmp/powercal}"
EXPECT_ACTIVE="${5:?expect_active (0 idle, 1 workload) required}"
ENGINE_CLASS="${6:?engine_class required}"
case "$EXPECT_ACTIVE" in 0|1) ;; *) echo "expect_active must be 0 or 1" >&2; exit 2 ;; esac
case "$ENGINE_CLASS" in
  idle|floor|cpu|gpu|npu|dsp) ;;
  *) echo "engine_class must be one of idle, floor, cpu, gpu, npu, dsp" >&2; exit 2 ;;
esac
case "$ENGINE_CLASS:$EXPECT_ACTIVE" in
  idle:0|floor:1|cpu:1|gpu:1|npu:1|dsp:1) ;;
  *) echo "engine_class=$ENGINE_CLASS contradicts expect_active=$EXPECT_ACTIVE" >&2; exit 2 ;;
esac
case "$WINDOW_ID" in
  ''|*[!A-Za-z0-9_.-]*) echo "window_id contains unsafe characters" >&2; exit 2 ;;
esac
case "$DURATION_S" in ''|*[!0-9]*) echo "duration_s must be a positive integer" >&2; exit 2 ;; esac
case "$HZ" in ''|*[!0-9]*) echo "hz must be a positive integer" >&2; exit 2 ;; esac
[ "$DURATION_S" -gt 0 ] || { echo "duration_s must be positive" >&2; exit 2; }
[ "$HZ" -gt 0 ] || { echo "hz must be positive" >&2; exit 2; }

KGSL=/sys/class/kgsl/kgsl-3d0
EM=/sys/kernel/debug/energy_model
P0DIR=/sys/devices/system/cpu/cpufreq/policy0
P6DIR=/sys/devices/system/cpu/cpufreq/policy6
P0="$P0DIR/scaling_cur_freq"
P6="$P6DIR/scaling_cur_freq"
ONLINE=/sys/devices/system/cpu/online
RAW="$OUTDIR/$WINDOW_ID.raw"
DONE="$OUTDIR/$WINDOW_ID.done"
FAIL="$OUTDIR/$WINDOW_ID.fail"
PROVISION_BASE=/data/local/tmp/profiler-kernels
PROVISION_MANIFEST="$PROVISION_BASE/runtime-provenance.json"
PROVISION_SNAPSHOT="$OUTDIR/$WINDOW_ID.runtime-provenance.json"
RUNTIME_HASHES="$OUTDIR/$WINDOW_ID.runtime-artifacts.sha256"
FLAG="$OUTDIR/WL_ACTIVE"
WL_PIDS="$OUTDIR/WL_PIDS"   # workload wrapper writes its process PIDs here (one per line)
BAD=0                        # set if any required read is empty or workload dies -> .fail not .done

mkdir -p "$OUTDIR"
rm -f "$DONE" "$FAIL"

emit_fail() { echo "FAILED $(date '+%Y-%m-%d %H:%M:%S')" > "$FAIL"; }
cleanup() { [ -f "$DONE" ] || [ -f "$FAIL" ] || emit_fail; }
trap cleanup EXIT
trap 'exit 3' INT TERM HUP

# Snapshot the provisioner-issued JSON, then independently hash the exact fixed
# deployed file set used by this engine.  The provisioner already compared each
# file to its host source; this second observation binds a window to the bytes
# that were actually present when it ran.  Any missing/unreadable input
# invalidates the window before telemetry collection starts.
capture_runtime_provenance() {
  manifest_tmp="$PROVISION_SNAPSHOT.tmp.$$"
  hashes_tmp="$RUNTIME_HASHES.tmp.$$"
  rm -f "$manifest_tmp" "$hashes_tmp" "$PROVISION_SNAPSHOT" "$RUNTIME_HASHES"
  command -v sha256sum >/dev/null 2>&1 || return 1
  [ -s "$PROVISION_MANIFEST" ] || return 1
  cp "$PROVISION_MANIFEST" "$manifest_tmp" || return 1
  # Cheap corruption/truncation guard without assuming jq exists on Android.
  grep -q '"manifest_kind"[[:space:]]*:[[:space:]]*"hdk8750-runtime-provenance"' "$manifest_tmp" || return 1
  grep -q '"hash_algorithm"[[:space:]]*:[[:space:]]*"sha256"' "$manifest_tmp" || return 1
  grep -q '"device"[[:space:]]*:' "$manifest_tmp" || return 1
  grep -q '"files"[[:space:]]*:' "$manifest_tmp" || return 1
  sha256sum "$PROVISION_MANIFEST" > "$hashes_tmp" || return 1

  case "$ENGINE_CLASS" in
    npu)
      grep -q '"npu_requested"[[:space:]]*:[[:space:]]*true' "$manifest_tmp" || return 1
      grep -q '"npu_platform_validator_passed"[[:space:]]*:[[:space:]]*true' "$manifest_tmp" || return 1
      grep -q '"npu_fp16_sustained_graph_passed"[[:space:]]*:[[:space:]]*true' "$manifest_tmp" || return 1
      grep -q '"npu_int8_sustained_graph_passed"[[:space:]]*:[[:space:]]*true' "$manifest_tmp" || return 1
      grep -q '"sdk_build_id"[[:space:]]*:' "$manifest_tmp" || return 1
      runtime_files="
$PROVISION_BASE/npu/libQnnHtp.so
$PROVISION_BASE/npu/libQnnHtpPrepare.so
$PROVISION_BASE/npu/libQnnHtpV79Stub.so
$PROVISION_BASE/npu/libQnnHtpV79CalculatorStub.so
$PROVISION_BASE/npu/libQnnSystem.so
$PROVISION_BASE/npu/libQnnHtpV79Skel.so
$PROVISION_BASE/npu/libCalculator_skel.so
$PROVISION_BASE/npu/qnn-platform-validator
$PROVISION_BASE/npu/qnn-net-run
$PROVISION_BASE/npu/matmul_qnn_fp16
$PROVISION_BASE/npu/matmul_qnn_int8"
      ;;
    dsp)
      grep -q '"dsp_requested"[[:space:]]*:[[:space:]]*true' "$manifest_tmp" || return 1
      grep -q '"dsp_lab_dispatch_passed"[[:space:]]*:[[:space:]]*true' "$manifest_tmp" || return 1
      runtime_files="
$PROVISION_BASE/dsp/matmul_dsp_sweep
$PROVISION_BASE/dsp/libdsp_matmul_stub.so
$PROVISION_BASE/dsp/libdsp_matmul_skel.so"
      ;;
    *) runtime_files="" ;;
  esac

  for runtime_file in $runtime_files; do
    grep -F -q "$runtime_file" "$manifest_tmp" || return 1
    [ -f "$runtime_file" ] && [ -r "$runtime_file" ] || return 1
    hash_record=$(sha256sum "$runtime_file") || return 1
    observed_hash=${hash_record%% *}
    [ "${#observed_hash}" -eq 64 ] || return 1
    grep -q '"sha256"[[:space:]]*:[[:space:]]*"'"$observed_hash"'"' "$manifest_tmp" || return 1
    echo "$hash_record" >> "$hashes_tmp" || return 1
  done
  [ -s "$hashes_tmp" ] || return 1
  mv "$manifest_tmp" "$PROVISION_SNAPSHOT" || return 1
  mv "$hashes_tmp" "$RUNTIME_HASHES" || return 1
}

case "$ENGINE_CLASS" in
  npu|dsp)
    if ! capture_runtime_provenance; then
      rm -f "$PROVISION_SNAPSHOT.tmp.$$" "$RUNTIME_HASHES.tmp.$$" \
            "$PROVISION_SNAPSHOT" "$RUNTIME_HASHES"
      echo "PROVENANCE_CAPTURE_FAILED engine_class=$ENGINE_CLASS $(date '+%Y-%m-%d %H:%M:%S')" > "$FAIL"
      exit 1
    fi
    ;;
esac

# thermal zones of interest, resolved once (type -> zone path)
THERMAL_ZONES=""
for z in /sys/class/thermal/thermal_zone*; do
  t=$(cat "$z/type" 2>/dev/null)
  case "$t" in
    cpu-0-5-0|cpu-1-0-0|cpuss-0|gpuss-0|ddr|socd|nsphvx-*|nsphmx-*) THERMAL_ZONES="$THERMAL_ZONES $t:$z" ;;
  esac
done

# Diagnostic-only Hexagon throttle state.  These are NOT activity features.
COOLING_DEVICES=""
for c in /sys/class/thermal/cooling_device*; do
  t=$(cat "$c/type" 2>/dev/null)
  case "$t" in
    cdsp_sw_hvx|cdsp_sw_hmx) COOLING_DEVICES="$COOLING_DEVICES $t:$c" ;;
  esac
done

read_gpu_busy() {
  # gpu_busy_percentage may be "NN %" or "NN"; emit the integer only
  v=$(cat "$KGSL/gpu_busy_percentage" 2>/dev/null)
  echo "$v" | tr -dc '0-9'
}

# pid_alive=1 iff every wrapper-declared workload PID is alive (finding 4).
# Idle windows have no WL_PIDS file -> not-applicable -> 1.
check_pids() {
  [ -f "$WL_PIDS" ] || { echo 1; return; }
  while read -r pid; do
    [ -n "$pid" ] || continue
    kill -0 "$pid" 2>/dev/null || { echo 0; return; }
  done < "$WL_PIDS"
  echo 1
}

# EAS table (once per session; safe to overwrite identically)
: > "$OUTDIR/eas.txt"
for c in cpu0 cpu6; do
  for d in "$EM/$c"/ps:*; do
    [ -d "$d" ] || continue
    f=$(cat "$d/frequency" 2>/dev/null); p=$(cat "$d/power" 2>/dev/null)
    [ -n "$f" ] && echo "EM $c freq=$f power_uW=$p" >> "$OUTDIR/eas.txt"
  done
done

# gpu clock-stats BEFORE (host verifies the expected pwrlevel slot advances)
{ echo "before $(cut -d. -f1 /proc/uptime)"; cat "$KGSL/gpu_clock_stats" 2>/dev/null; } > "$OUTDIR/$WINDOW_ID.clockstats"

# One sample block appended to $RAW. Sets BAD=1 on any empty required read or
# on a workload marked active whose process group has died (fail-closed).
emit_sample() {
  read tb _ < /proc/uptime
  p0=$(cat "$P0" 2>/dev/null); p6=$(cat "$P6" 2>/dev/null)
  p0min=$(cat "$P0DIR/scaling_min_freq" 2>/dev/null); p0max=$(cat "$P0DIR/scaling_max_freq" 2>/dev/null); p0gov=$(cat "$P0DIR/scaling_governor" 2>/dev/null)
  p6min=$(cat "$P6DIR/scaling_min_freq" 2>/dev/null); p6max=$(cat "$P6DIR/scaling_max_freq" 2>/dev/null); p6gov=$(cat "$P6DIR/scaling_governor" 2>/dev/null)
  # Adreno 830 (SM8750) exposes the live clock under kgsl's devfreq/ symlink;
  # Adreno 840 (SM8850) dropped that subdirectory and keeps only gpuclk. Both
  # report Hz, so either satisfies the host-side aggregator. Empty here means a
  # BAD_READS failure for the whole window, so try both.
  gclk=$(cat "$KGSL/devfreq/cur_freq" 2>/dev/null)
  [ -n "$gclk" ] || gclk=$(cat "$KGSL/gpuclk" 2>/dev/null)
  gbusy=$(read_gpu_busy)
  gmin=$(cat "$KGSL/min_pwrlevel" 2>/dev/null); gmax=$(cat "$KGSL/max_pwrlevel" 2>/dev/null)
  gth=$(cat "$KGSL/thermal_pwrlevel" 2>/dev/null)
  mask=$(cat "$ONLINE" 2>/dev/null)
  [ -f "$FLAG" ] && wl=1 || wl=0
  pidalive=$(check_pids)
  stat=$(cat /proc/stat)
  temps=""
  for tz in $THERMAL_ZONES; do
    ty=${tz%%:*}; zp=${tz#*:}; tv=$(cat "$zp/temp" 2>/dev/null)
    temps="$temps $ty=$tv"
  done
  cooling=""
  for cd in $COOLING_DEVICES; do
    ty=${cd%%:*}; cp=${cd#*:}; cv=$(cat "$cp/cur_state" 2>/dev/null)
    cooling="$cooling $ty=$cv"
  done
  read ta _ < /proc/uptime
  # required-field integrity + workload liveness
  for v in "$p0" "$p6" "$p0min" "$p0max" "$p0gov" "$p6min" "$p6max" "$p6gov" \
           "$gclk" "$gbusy" "$gmin" "$gmax" "$gth" "$mask"; do
    [ -z "$v" ] && BAD=1
  done
  [ "$wl" = "1" ] && [ "$pidalive" = "0" ] && BAD=1
  # for declared-active windows: the workload must be provably running EVERY sample
  # (nonempty WL_PIDS, all alive, flag set) — an early death that removes the files
  # is caught here rather than passing as idle.
  if [ "$EXPECT_ACTIVE" = "1" ]; then
    { [ "$wl" = "1" ] && [ "$pidalive" = "1" ] && [ -s "$WL_PIDS" ]; } || BAD=1
  else
    { [ "$wl" = "0" ] && [ ! -e "$WL_PIDS" ]; } || BAD=1
  fi
  {
    echo "=SAMPLE $idx $tb $ta"
    echo "=SCALARS policy0_khz=$p0 policy6_khz=$p6 policy0_min_khz=$p0min policy0_max_khz=$p0max policy0_gov=$p0gov policy6_min_khz=$p6min policy6_max_khz=$p6max policy6_gov=$p6gov gpuclk_hz=$gclk gpu_busy_pct=$gbusy gpu_min_pwrlevel=$gmin gpu_max_pwrlevel=$gmax gpu_thermal_pwrlevel=$gth online=$mask workload_active=$wl pid_alive=$pidalive"
    echo "=TEMPS$temps"
    echo "=COOLING$cooling"
    echo "=STAT"
    echo "$stat"
    echo "=ENDSTAT"
  } >> "$RAW"
  idx=$((idx + 1))
}

# --- sampling loop ---
: > "$RAW"
echo "=WINDOW $WINDOW_ID hz=$HZ duration_s=$DURATION_S expect_active=$EXPECT_ACTIVE engine_class=$ENGINE_CLASS boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)" >> "$RAW"
END=$(( $(date +%s) + DURATION_S ))
idx=0
usec=$(( 1000000 / HZ ))
PERIOD_NS=$(( 1000000000 / HZ ))
LAG=0
# Deadline-compensated pacing (finding 82). emit_sample is NOT free: it reads
# every thermal zone plus /proc/stat, which costs ~0.3 s on a busy board and
# more on a saturated one. Sleeping a fixed 1/HZ *after* the emit therefore
# gives a period of (emit + 1/HZ), not 1/HZ. Measured: a 35 s cpu window under
# an 8-thread load produced 27 samples at 1.366 s apart instead of ~36, and the
# host rejected the whole window as a sparse capture -- so power was
# unavailable for exactly the saturated configurations that most need it.
# Sleep only the REMAINDER of each period so the rate holds under load.
NEXT_NS=$(date +%s%N)
while [ "$(date +%s)" -lt "$END" ]; do
  emit_sample
  NEXT_NS=$(( NEXT_NS + PERIOD_NS ))
  REM_NS=$(( NEXT_NS - $(date +%s%N) ))
  if [ "$REM_NS" -gt 0 ]; then
    usleep $(( REM_NS / 1000 )) 2>/dev/null || sleep 1
  else
    # emit alone now exceeds the period. Resync rather than let the lag
    # compound, and count it so the host can see the window was paced badly
    # instead of inferring it from a short sample count.
    LAG=$(( LAG + 1 ))
    NEXT_NS=$(date +%s%N)
  fi
done
emit_sample   # final endpoint snapshot so N samples cover the full duration (finding 7)

# gpu clock-stats AFTER
{ echo "after $(cut -d. -f1 /proc/uptime)"; cat "$KGSL/gpu_clock_stats" 2>/dev/null; } >> "$OUTDIR/$WINDOW_ID.clockstats"

if [ "$BAD" = "0" ]; then
  echo "OK samples=$idx pacing_lag=$LAG $(date '+%Y-%m-%d %H:%M:%S')" > "$DONE"
else
  echo "BAD_READS samples=$idx pacing_lag=$LAG $(date '+%Y-%m-%d %H:%M:%S')" > "$FAIL"
  exit 1
fi
