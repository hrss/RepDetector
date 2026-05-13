#!/usr/bin/env python3
"""
Parse a .imubin file (single-file binary IMU format) and plot all 6 channels.

Binary layout (little-endian throughout):

  Header (48 bytes):
    0..4    magic    b"IMU\0"
    4       version  uint8  (expected: 2)
    5       wrist    uint8  (0=Left, 1=Right)
    6       crown    uint8  (0=Left, 1=Right)
    7       pad      uint8
    8..44   runId    36 ASCII bytes (UUID, zero-padded)
    44..48  workout  int32

  Sample (36 bytes, repeated):
    0..8    timestamp    float64
    8..12   workoutTime  int32
    12..16  ax           float32  (g)
    16..20  ay           float32  (g)
    20..24  az           float32  (g)
    24..28  gx           float32  (rad/s)
    28..32  gy           float32  (rad/s)
    32..36  gz           float32  (rad/s)

Usage:
    python plot_imu.py path/to/run.imubin [-o out.png] [--x time|index|workout]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HEADER_SIZE = 48
SAMPLE_SIZE = 36
MAGIC = b"IMU\x00"

# struct format for one sample: little-endian, double + int32 + 6 floats
SAMPLE_STRUCT = struct.Struct("<dl6f")  # 'l' is 4 bytes on the wire here; safer to be explicit:
SAMPLE_STRUCT = struct.Struct("<di6f")  # d=float64, i=int32, 6f=six float32  -> 8+4+24 = 36
assert SAMPLE_STRUCT.size == SAMPLE_SIZE, SAMPLE_STRUCT.size


def parse_header(buf: bytes) -> dict:
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"File too small ({len(buf)} bytes) to contain a header.")
    if buf[:4] != MAGIC:
        raise ValueError(f"Bad magic {buf[:4]!r}; expected {MAGIC!r}.")

    version = buf[4]
    wrist = buf[5]
    crown = buf[6]
    # buf[7] is padding
    run_id = buf[8:44].split(b"\x00", 1)[0].decode("ascii", errors="replace")
    (workout_id,) = struct.unpack_from("<i", buf, 44)

    return {
        "version": version,
        "wrist": "Right" if wrist == 1 else "Left",
        "crown": "Right" if crown == 1 else "Left",
        "run_id": run_id,
        "workout_id": workout_id,
    }


def parse_samples(buf: bytes) -> np.ndarray:
    """Return a structured array with named columns."""
    payload = buf[HEADER_SIZE:]
    n, rem = divmod(len(payload), SAMPLE_SIZE)
    if rem != 0:
        print(
            f"Warning: trailing {rem} bytes after {n} samples — file may be truncated.",
            file=sys.stderr,
        )
        payload = payload[: n * SAMPLE_SIZE]

    dtype = np.dtype(
        [
            ("timestamp", "<f8"),
            ("workoutTime", "<i4"),
            ("ax", "<f4"),
            ("ay", "<f4"),
            ("az", "<f4"),
            ("gx", "<f4"),
            ("gy", "<f4"),
            ("gz", "<f4"),
        ]
    )
    assert dtype.itemsize == SAMPLE_SIZE, dtype.itemsize
    return np.frombuffer(payload, dtype=dtype)


TARGET_HZ = 50  # configured samplingRate in IMUCollector


def plot(samples: np.ndarray, header: dict, out_path: Path, x_mode: str) -> None:
    n = len(samples)
    if n == 0:
        raise ValueError("No samples to plot.")

    if x_mode == "time":
        x = samples["timestamp"] - samples["timestamp"][0]
        x_label = "Elapsed time (s, from first sample)"
    elif x_mode == "workout":
        x = samples["workoutTime"]
        x_label = "Workout time (s)"
    else:  # index
        x = np.arange(n)
        x_label = "Sample index"

    channels = [
        ("ax", "Accel X", "g"),
        ("ay", "Accel Y", "g"),
        ("az", "Accel Z", "g"),
        ("gx", "Gyro X", "rad/s"),
        ("gy", "Gyro Y", "rad/s"),
        ("gz", "Gyro Z", "rad/s"),
    ]

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    accel_color = "#1f77b4"
    gyro_color = "#d62728"

    for ax, (key, label, unit) in zip(axes, channels):
        color = accel_color if key.startswith("a") else gyro_color
        ax.plot(x, samples[key], linewidth=0.8, color=color)
        ax.set_ylabel(f"{label}\n({unit})")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel(x_label)

    duration = samples["timestamp"][-1] - samples["timestamp"][0] if n > 1 else 0.0
    rate = (n - 1) / duration if duration > 0 else float("nan")
    fig.suptitle(
        f"IMU run {header['run_id']}  •  workout #{header['workout_id']}  •  "
        f"wrist={header['wrist']}, crown={header['crown']}\n"
        f"{n:,} samples  •  {duration:.2f} s  •  "
        f"target {TARGET_HZ} Hz  •  measured ~{rate:.1f} Hz",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Plot 6 IMU channels from a .imubin file.")
    p.add_argument("path", type=Path, help="Path to .imubin file")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <input>.png)",
    )
    p.add_argument(
        "--x",
        choices=("time", "index", "workout"),
        default="time",
        help="What to use for the x-axis (default: time)",
    )
    args = p.parse_args()

    buf = args.path.read_bytes()
    header = parse_header(buf)
    samples = parse_samples(buf)

    print(f"Run ID:     {header['run_id']}")
    print(f"Workout ID: {header['workout_id']}")
    print(f"Version:    {header['version']}")
    print(f"Wrist:      {header['wrist']}")
    print(f"Crown:      {header['crown']}")
    print(f"Samples:    {len(samples):,}")

    out_path = args.out or args.path.with_suffix(".png")
    plot(samples, header, out_path, args.x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())