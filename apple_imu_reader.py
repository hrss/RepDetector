#!/usr/bin/env python3
"""
Parse a .imubin file and plot all 6 IMU channels.
Optionally overlay labeled exercise segments from a workout result JSON.

Usage:
    python plot_imu.py run.imubin
    python plot_imu.py run.imubin --workout result.json
    python plot_imu.py run.imubin --workout result.json -o out.png --x workout
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

HEADER_SIZE = 48
SAMPLE_SIZE = 36
MAGIC = b"IMU\x00"

SAMPLE_STRUCT = struct.Struct("<di6f")
assert SAMPLE_STRUCT.size == SAMPLE_SIZE


def parse_header(buf: bytes) -> dict:
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"File too small ({len(buf)} bytes) to contain a header.")
    if buf[:4] != MAGIC:
        raise ValueError(f"Bad magic {buf[:4]!r}; expected {MAGIC!r}.")
    version = buf[4]
    wrist   = buf[5]
    crown   = buf[6]
    run_id  = buf[8:44].split(b"\x00", 1)[0].decode("ascii", errors="replace")
    (workout_id,) = struct.unpack_from("<i", buf, 44)
    return {
        "version":    version,
        "wrist":      "Right" if wrist == 1 else "Left",
        "crown":      "Right" if crown == 1 else "Left",
        "run_id":     run_id,
        "workout_id": workout_id,
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


def parse_workout_segments(path: Path) -> list:
    """Extract exercise segments. Returns [{name, start, end, is_rest}, ...]"""
    data = json.loads(path.read_text())
    segments = []
    for round_ in data.get("roundResults", []):
        for ex in round_.get("exerciseResults", []):
            segments.append({
                "name":    ex["name"],
                "start":   ex["startTime"],
                "end":     ex["endTime"],
                "is_rest": ex["name"].upper() == "REST",
            })
    return segments


def _make_palette(segments: list) -> dict:
    exercise_names = [s["name"] for s in segments if not s["is_rest"]]
    unique = list(dict.fromkeys(exercise_names))
    palette = {"REST": "#cccccc"}
    for i, name in enumerate(unique):
        hue = i / max(len(unique), 1)
        r, g, b = colorsys.hls_to_rgb(hue, 0.75, 0.65)
        palette[name] = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
    return palette


def _workout_to_x(samples: np.ndarray, workout_time: int, x: np.ndarray) -> float:
    idx = np.searchsorted(samples["workoutTime"], workout_time)
    idx = min(idx, len(x) - 1)
    return float(x[idx])


def _workout_to_index(samples: np.ndarray, workout_time: int) -> int:
    return int(np.searchsorted(samples["workoutTime"], workout_time))


TARGET_HZ = 200


def plot(samples, header, out_path, x_mode, segments=None):
    n = len(samples)
    if n == 0:
        raise ValueError("No samples to plot.")

    if x_mode == "time":
        x = samples["timestamp"] - samples["timestamp"][0]
        x_label = "Elapsed time (s)"
    elif x_mode == "workout":
        x = samples["workoutTime"].astype(float)
        x_label = "Workout time (s)"
    else:
        x = np.arange(n)
        x_label = "Sample index"

    channels = [
        ("ax", "Accel X", "g"),
        ("ay", "Accel Y", "g"),
        ("az", "Accel Z", "g"),
        ("gx", "Gyro X",  "rad/s"),
        ("gy", "Gyro Y",  "rad/s"),
        ("gz", "Gyro Z",  "rad/s"),
    ]

    palette = _make_palette(segments) if segments else {}

    fig, axes = plt.subplots(6, 1, figsize=(14, 11), sharex=True)
    accel_color = "#1f77b4"
    gyro_color  = "#d62728"

    for ax, (key, label, unit) in zip(axes, channels):
        # Segment overlays
        if segments:
            for seg in segments:
                color = palette.get(seg["name"], "#eeeeee")
                alpha = 0.18 if seg["is_rest"] else 0.28

                if x_mode == "workout":
                    x0, x1 = float(seg["start"]), float(seg["end"])
                elif x_mode == "time":
                    x0 = _workout_to_x(samples, seg["start"], x)
                    x1 = _workout_to_x(samples, seg["end"],   x)
                else:
                    x0 = float(_workout_to_index(samples, seg["start"]))
                    x1 = float(_workout_to_index(samples, seg["end"]))

                ax.axvspan(x0, x1, color=color, alpha=alpha, linewidth=0)

                # Label exercises on the top subplot only
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
        ax.set_ylabel(f"{label}\n({unit})", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel(x_label)

    # Legend
    if segments:
        handles = [
            mpatches.Patch(color=color, alpha=0.6, label=name)
            for name, color in palette.items()
            if name != "REST"
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 6),
            fontsize=7.5,
            framealpha=0.8,
            bbox_to_anchor=(0.5, 0.0),
        )
        bottom_margin = 0.06 + 0.015 * ((len(handles) - 1) // 6)
    else:
        bottom_margin = 0.02

    duration = samples["timestamp"][-1] - samples["timestamp"][0] if n > 1 else 0.0
    rate     = (n - 1) / duration if duration > 0 else float("nan")
    fig.suptitle(
        f"IMU run {header['run_id']}  •  workout #{header['workout_id']}  •  "
        f"wrist={header['wrist']}, crown={header['crown']}\n"
        f"{n:,} samples  •  {duration:.2f} s  •  "
        f"target {TARGET_HZ} Hz  •  measured ~{rate:.1f} Hz",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, bottom_margin, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Plot 6 IMU channels from a .imubin file.")
    p.add_argument("path",        type=Path, help="Path to .imubin file")
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--x",         choices=("time", "index", "workout"), default="workout")
    p.add_argument("--workout",   type=Path, default=None,
                   help="Workout result JSON for segment overlay")
    args = p.parse_args()

    buf     = args.path.read_bytes()
    header  = parse_header(buf)
    samples = parse_samples(buf)

    print(f"Run ID:     {header['run_id']}")
    print(f"Workout ID: {header['workout_id']}")
    print(f"Wrist:      {header['wrist']}")
    print(f"Crown:      {header['crown']}")
    print(f"Samples:    {len(samples):,}")

    segments = None
    if args.workout:
        segments = parse_workout_segments(args.workout)
        print(f"Segments:   {len(segments)} (from {args.workout.name})")

    out_path = args.out or args.path.with_suffix(".png")
    plot(samples, header, out_path, args.x, segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())