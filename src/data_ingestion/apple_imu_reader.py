#!/usr/bin/env python3
"""
Parse a .imubin file and plot all 6 IMU channels.
Supports:
- Float-precision workout time alignment (via CMAbsoluteTime ↔ workoutTime mapping)
- Workout JSON overlay (original OR aligned-export format)
- Rep markers from aligned export
- Optional Butterworth low-pass filter (zero-phase via filtfilt)

Usage:
    python plot_imu.py run.imubin
    python plot_imu.py run.imubin --workout result.aligned.json
    python plot_imu.py run.imubin --workout result.aligned.json --lowpass 10
    python plot_imu.py run.imubin --workout result.aligned.json --lowpass 10 --x workout
"""

import argparse
import colorsys
import json
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

HEADER_SIZE = 48
SAMPLE_SIZE = 36
MAGIC = b"IMU\x00"


# ---------------------------------------------------------------------------
# Binary parsing
# ---------------------------------------------------------------------------
def parse_header(buf: bytes) -> dict:
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"File too small ({len(buf)} bytes).")
    if buf[:4] != MAGIC:
        raise ValueError(f"Bad magic {buf[:4]!r}; expected {MAGIC!r}.")
    return {
        "version":    buf[4],
        "wrist":      "Right" if buf[5] == 1 else "Left",
        "crown":      "Right" if buf[6] == 1 else "Left",
        "run_id":     buf[8:44].split(b"\x00", 1)[0].decode("ascii", errors="replace"),
        "workout_id": struct.unpack_from("<i", buf, 44)[0],
    }


def parse_samples(buf: bytes) -> np.ndarray:
    payload = buf[HEADER_SIZE:]
    n, rem = divmod(len(payload), SAMPLE_SIZE)
    if rem:
        print(f"Warning: trailing {rem} bytes — file may be truncated.", file=sys.stderr)
        payload = payload[: n * SAMPLE_SIZE]
    dtype = np.dtype([
        ("timestamp",   "<f8"),
        ("workoutTime", "<i4"),
        ("ax", "<f4"), ("ay", "<f4"), ("az", "<f4"),
        ("gx", "<f4"), ("gy", "<f4"), ("gz", "<f4"),
    ])
    assert dtype.itemsize == SAMPLE_SIZE
    return np.frombuffer(payload, dtype=dtype)


# ---------------------------------------------------------------------------
# Workout-time ↔ CMAbsoluteTime mapping
# ---------------------------------------------------------------------------
def build_workout_time_mapper(samples: np.ndarray):
    """
    workoutTime in the binary is integer seconds. To resolve fractional workout
    times (from the aligned JSON export), we fit a linear map:
        cmAbsoluteTime = a * workoutTime + b
    using the actual sample timestamps. This recovers subsecond precision.

    Strategy: for each integer second value present, take the first sample with
    that value (it's the one closest to that integer boundary). Then fit a line
    through those (workoutTime, timestamp) anchor points using least squares.
    """
    wt = samples["workoutTime"]
    ts = samples["timestamp"]

    # Find the first sample index for each unique workoutTime value
    _, first_idx = np.unique(wt, return_index=True)
    anchor_wt = wt[first_idx].astype(np.float64)
    anchor_ts = ts[first_idx]

    if len(anchor_wt) < 2:
        # Fallback: single anchor, slope = 1 (assume 1:1)
        b = anchor_ts[0] - anchor_wt[0]
        return (lambda w: w + b), (lambda t: t - b)

    # Linear least squares fit
    A = np.vstack([anchor_wt, np.ones_like(anchor_wt)]).T
    (a, b), *_ = np.linalg.lstsq(A, anchor_ts, rcond=None)

    def workout_to_abs(w):  # accepts scalar or array
        return a * np.asarray(w) + b

    def abs_to_workout(t):
        return (np.asarray(t) - b) / a

    return workout_to_abs, abs_to_workout


# ---------------------------------------------------------------------------
# Workout JSON parsing — handles both original and aligned-export formats
# ---------------------------------------------------------------------------
def parse_workout(path: Path) -> tuple[list[dict], list[dict]]:
    """
    Returns (segments, reps).
    - segments: [{name, start, end, is_rest}, ...]  (floats supported)
    - reps:     [{workout_time, segment_name}, ...]
    """
    data = json.loads(path.read_text())

    # The aligned export wraps the workout under "workout"
    workout = data.get("workout", data)

    segments: list[dict] = []
    reps: list[dict] = []

    for round_ in workout.get("roundResults", []):
        for ex in round_.get("exerciseResults", []):
            seg = {
                "name":    ex["name"],
                "start":   float(ex["startTime"]),
                "end":     float(ex["endTime"]),
                "is_rest": ex["name"].upper() == "REST",
            }
            segments.append(seg)
            for r in ex.get("reps", []) or []:
                reps.append({
                    "workout_time": float(r["workoutTime"]),
                    "segment_name": ex["name"],
                })
    return segments, reps


def _make_palette(segments: list[dict]) -> dict[str, str]:
    exercise_names = [s["name"] for s in segments if not s["is_rest"]]
    unique = list(dict.fromkeys(exercise_names))
    palette = {"REST": "#cccccc"}
    for i, name in enumerate(unique):
        hue = i / max(len(unique), 1)
        r, g, b = colorsys.hls_to_rgb(hue, 0.75, 0.65)
        palette[name] = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
    return palette


# ---------------------------------------------------------------------------
# Butterworth low-pass filter
# ---------------------------------------------------------------------------
def lowpass_filter(samples: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass to all 6 channels."""
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        raise SystemExit("scipy required for --lowpass. Install with: pip install scipy")

    # Estimate sample rate from timestamps
    ts = samples["timestamp"]
    if len(ts) < 2:
        return samples
    dt = np.median(np.diff(ts))
    fs = 1.0 / dt

    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        print(f"Warning: cutoff {cutoff_hz} Hz >= Nyquist {nyq:.1f} Hz; skipping filter.",
              file=sys.stderr)
        return samples

    b, a = butter(order, cutoff_hz / nyq, btype="low")

    out = samples.copy()
    for ch in ("ax", "ay", "az", "gx", "gy", "gz"):
        out[ch] = filtfilt(b, a, samples[ch]).astype(np.float32)

    print(f"Applied Butterworth low-pass: order={order}, cutoff={cutoff_hz} Hz, "
          f"sample rate≈{fs:.1f} Hz")
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot(
    samples:  np.ndarray,
    header:   dict,
    out_path: Path,
    x_mode:   str,
    segments: list[dict] | None = None,
    reps:     list[dict] | None = None,
    lowpass_hz: float | None = None,
) -> None:
    n = len(samples)
    if n == 0:
        raise ValueError("No samples to plot.")

    workout_to_abs, abs_to_workout = build_workout_time_mapper(samples)

    # --- x-axis: array of x values for the samples themselves ---
    if x_mode == "time":
        x = samples["timestamp"] - samples["timestamp"][0]
        x_label = "Elapsed time (s)"
        # Convert workout time (float) → elapsed seconds
        def wt_to_x(w):
            return float(workout_to_abs(w) - samples["timestamp"][0])
    elif x_mode == "workout":
        # Use float workout time derived from timestamp, not the integer field
        x = abs_to_workout(samples["timestamp"]).astype(np.float64)
        x_label = "Workout time (s)"
        def wt_to_x(w):
            return float(w)
    else:
        x = np.arange(n)
        x_label = "Sample index"
        def wt_to_x(w):
            # find nearest sample by absolute time
            target_abs = workout_to_abs(w)
            return float(np.searchsorted(samples["timestamp"], target_abs))

    channels = [
        ("ax", "Accel X", "g"),
        ("ay", "Accel Y", "g"),
        ("az", "Accel Z", "g"),
        ("gx", "Gyro X",  "rad/s"),
        ("gy", "Gyro Y",  "rad/s"),
        ("gz", "Gyro Z",  "rad/s"),
    ]

    palette = _make_palette(segments) if segments else {}

    fig, axes = plt.subplots(6, 1, figsize=(28, 22), sharex=True)
    accel_color = "#1f77b4"
    gyro_color  = "#d62728"

    for ax, (key, label, unit) in zip(axes, channels):
        # Segment overlays (drawn behind the trace)
        if segments:
            for seg in segments:
                color = palette.get(seg["name"], "#eeeeee")
                alpha = 0.18 if seg["is_rest"] else 0.28
                x0 = wt_to_x(seg["start"])
                x1 = wt_to_x(seg["end"])
                ax.axvspan(x0, x1, color=color, alpha=alpha, linewidth=0)

                if ax is axes[0] and not seg["is_rest"]:
                    mid = (x0 + x1) / 2
                    ax.text(
                        mid, 1.01, seg["name"],
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom",
                        fontsize=6.5, rotation=45, clip_on=False,
                        color="#333333",
                    )

        line_color = accel_color if key.startswith("a") else gyro_color
        ax.plot(x, samples[key], linewidth=0.7, color=line_color)

        # Rep tick marks on each subplot
        if reps:
            for rep in reps:
                rx = wt_to_x(rep["workout_time"])
                rep_color = palette.get(rep["segment_name"], "#ffffff")
                ax.axvline(rx, color=rep_color, linewidth=0.8, alpha=0.6,
                           linestyle="-", zorder=1.5)

        ax.set_ylabel(f"{label}\n({unit})", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.grid(True, which="minor", alpha=0.1, linestyle=":")
        ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)

    # Granular x-axis ticks
    if x_mode in ("time", "workout"):
        duration = x[-1] - x[0]
        if duration > 0:
            # Aim for ~20-40 major ticks
            # Common intervals: 1, 2, 5, 10, 30, 60, 120, 300, 600...
            potential_intervals = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
            interval = 5
            for pi in potential_intervals:
                if duration / pi <= 40:
                    interval = pi
                    break
            
            axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(interval))
            minor_interval = interval / 5
            if minor_interval >= 1:
                axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(minor_interval))
            else:
                axes[-1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    elif x_mode == "index":
        # For sample index, maybe every 1000 or 5000 depending on duration
        # but let's stick to 1000 if it's not too crowded
        axes[-1].xaxis.set_major_locator(ticker.AutoLocator())
        axes[-1].xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Rep number labels on the top subplot
    if reps:
        # Group reps by segment so numbering restarts per exercise
        from collections import defaultdict
        per_seg = defaultdict(list)
        for r in reps:
            per_seg[r["segment_name"]].append(r["workout_time"])
        for seg_name, times in per_seg.items():
            for j, t in enumerate(sorted(times)):
                rx = wt_to_x(t)
                axes[0].text(
                    rx, 1.08, str(j + 1),
                    transform=axes[0].get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=6, color=palette.get(seg_name, "#666"),
                    clip_on=False,
                )

    axes[-1].set_xlabel(x_label)

    # --- legend ---
    if segments:
        handles = [
            mpatches.Patch(color=color, alpha=0.6, label=name)
            for name, color in palette.items() if name != "REST"
        ]
        if reps:
            handles.append(mpatches.Patch(color="white", label=f"reps (n={len(reps)})"))
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 7),
            fontsize=7.5,
            framealpha=0.8,
            bbox_to_anchor=(0.5, 0.0),
        )
        bottom_margin = 0.06 + 0.015 * ((len(handles) - 1) // 7)
    else:
        bottom_margin = 0.02

    # --- title ---
    duration = samples["timestamp"][-1] - samples["timestamp"][0] if n > 1 else 0.0
    rate     = (n - 1) / duration if duration > 0 else float("nan")
    filter_note = f"  •  LP {lowpass_hz} Hz" if lowpass_hz else ""
    fig.suptitle(
        f"IMU run {header['run_id']}  •  workout #{header['workout_id']}  •  "
        f"wrist={header['wrist']}, crown={header['crown']}\n"
        f"{n:,} samples  •  {duration:.2f} s  •  ~{rate:.1f} Hz{filter_note}",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, bottom_margin, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Plot 6 IMU channels from a .imubin file.")
    p.add_argument("path",        type=Path, help="Path to .imubin file")
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--x",         choices=("time", "index", "workout"), default="workout",
                   help="X-axis mode (default: workout)")
    p.add_argument("--workout",   type=Path, default=None,
                   help="Workout JSON (original or aligned export) for segments + reps")
    p.add_argument("--lowpass",   type=float, default=None, metavar="HZ",
                   help="Apply Butterworth low-pass at this cutoff (e.g. 10)")
    p.add_argument("--filter-order", type=int, default=4,
                   help="Butterworth filter order (default: 4)")
    args = p.parse_args()

    buf     = args.path.read_bytes()
    header  = parse_header(buf)
    samples = parse_samples(buf)

    print(f"Run ID:     {header['run_id']}")
    print(f"Workout ID: {header['workout_id']}")
    print(f"Wrist:      {header['wrist']}")
    print(f"Crown:      {header['crown']}")
    print(f"Samples:    {len(samples):,}")

    if args.lowpass is not None:
        samples = lowpass_filter(samples, args.lowpass, args.filter_order)

    segments, reps = None, None
    if args.workout:
        segments, reps = parse_workout(args.workout)
        print(f"Segments:   {len(segments)}  •  Reps: {len(reps)}  (from {args.workout.name})")

    out_path = args.out or args.path.with_suffix(".png")
    plot(samples, header, out_path, args.x, segments, reps,
         lowpass_hz=args.lowpass)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())