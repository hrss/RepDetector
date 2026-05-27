"""
Apple .imubin -> unified canonical session format.

Pipeline:
    .imubin file + (optional) workout JSON
        -> parse binary into raw arrays
        -> canonicalize (rotate + unit-check)
        -> attach segments + reps from workout JSON
        -> write_session(...)

Apple .imubin binary layout (from plot_imu.py):
    Header (48 bytes):
        4   magic       b"IMU\x00"
        1   version     u8
        1   wrist       u8 (0=left, 1=right)
        1   crown       u8 (0=left, 1=right)
        1   <padding>
        36  run_id      ascii (null-padded)
        4   workout_id  i32 LE
    Sample (36 bytes each, packed):
        8   timestamp   f64  (CMAbsoluteTime, seconds)
        4   workoutTime i32  (integer seconds from workout start)
        4*6 ax,ay,az,gx,gy,gz  f32 each

Units in .imubin (from CMDeviceMotion):
    accel: g     (specific force, already in g)
    gyro:  rad/s (CMDeviceMotion uses rad/s by default)

So Apple data needs NO unit conversion before canonicalize — only rotation.
"""

from __future__ import annotations

import argparse
import struct
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from canonicalize import (
    Wrist,
    Crown,
    apple_imubin_spec,
    canonicalize_dataframe,
)
from session_io import Rep, Segment, SessionMeta, write_session

# ---------------------------------------------------------------------------
# Binary parsing constants (mirrored from plot_imu.py)
# ---------------------------------------------------------------------------

HEADER_SIZE = 48
SAMPLE_SIZE = 36
MAGIC = b"IMU\x00"

_SAMPLE_DTYPE = np.dtype(
    [
        ("timestamp", "<f8"),
        ("workoutTime", "<i4"),
        ("ax", "<f4"), ("ay", "<f4"), ("az", "<f4"),
        ("gx", "<f4"), ("gy", "<f4"), ("gz", "<f4"),
    ]
)
assert _SAMPLE_DTYPE.itemsize == SAMPLE_SIZE

NOMINAL_RATE_HZ = 200  # Apple Watch HKWorkoutSession motion stream nominal rate

# Rate sanity bounds: warn if the empirical rate falls outside this range.
# Apple's stream is supposed to be 200Hz; anything wildly different probably
# indicates a corrupt or wrong-format file rather than a legitimate session.
MIN_REASONABLE_RATE = 150.0
MAX_REASONABLE_RATE = 250.0


# ---------------------------------------------------------------------------
# Header + samples
# ---------------------------------------------------------------------------

@dataclass
class ImuBinHeader:
    version: int
    wrist: Wrist
    crown: Crown
    run_id: str
    workout_id: int


def parse_header(buf: bytes) -> ImuBinHeader:
    if len(buf) < HEADER_SIZE:
        raise ValueError(f".imubin too small ({len(buf)} bytes); expected >= {HEADER_SIZE}")
    if buf[:4] != MAGIC:
        raise ValueError(f"Bad magic {buf[:4]!r}; expected {MAGIC!r}")

    version = buf[4]
    wrist_byte = buf[5]
    crown_byte = buf[6]
    # buf[7] is padding/reserved
    run_id = buf[8:44].split(b"\x00", 1)[0].decode("ascii", errors="replace")
    workout_id = struct.unpack_from("<i", buf, 44)[0]

    if wrist_byte not in (0, 1):
        raise ValueError(f"Invalid wrist byte: {wrist_byte}")
    if crown_byte not in (0, 1):
        raise ValueError(f"Invalid crown byte: {crown_byte}")

    return ImuBinHeader(
        version=version,
        wrist="right" if wrist_byte == 1 else "left",
        crown="right" if crown_byte == 1 else "left",
        run_id=run_id,
        workout_id=workout_id,
    )


def parse_samples(buf: bytes) -> np.ndarray:
    """Return a structured array of all samples. Drops trailing partial sample."""
    payload = buf[HEADER_SIZE:]
    n, rem = divmod(len(payload), SAMPLE_SIZE)
    if rem:
        warnings.warn(
            f".imubin has {rem} trailing bytes (truncated last sample); dropping them",
            stacklevel=2,
        )
        payload = payload[: n * SAMPLE_SIZE]
    return np.frombuffer(payload, dtype=_SAMPLE_DTYPE)


def empirical_rate_hz(samples: np.ndarray) -> float:
    """Estimate sample rate from the median inter-sample interval."""
    if len(samples) < 2:
        return float("nan")
    dt = np.median(np.diff(samples["timestamp"]))
    if dt <= 0:
        return float("nan")
    return 1.0 / dt


# ---------------------------------------------------------------------------
# Workout JSON parsing (segments + reps)
# ---------------------------------------------------------------------------

def _build_workout_time_mapper(samples: np.ndarray):
    """Recover the fractional workout_time <-> sample-timestamp relationship.

    The integer `workoutTime` field rolls over once per second. Each
    distinct integer value's FIRST occurrence in the stream is closest to
    the true integer boundary. A least-squares line through those anchor
    points (workoutTime, sample-timestamp) gives us a sub-second-accurate
    map. Mirrors plot_imu.py.

    Returns: workout_to_abs(w) -> absolute sample-timestamp
             abs_to_workout(t) -> fractional workout_time
    """
    wt = samples["workoutTime"]
    ts = samples["timestamp"]
    _, first_idx = np.unique(wt, return_index=True)
    anchor_wt = wt[first_idx].astype(np.float64)
    anchor_ts = ts[first_idx]

    if len(anchor_wt) < 2:
        # Degenerate: only one workoutTime value present. Assume slope = 1.
        b = anchor_ts[0] - anchor_wt[0]
        return (lambda w: np.asarray(w) + b), (lambda t: np.asarray(t) - b)

    A = np.vstack([anchor_wt, np.ones_like(anchor_wt)]).T
    (a, b), *_ = np.linalg.lstsq(A, anchor_ts, rcond=None)

    def workout_to_abs(w):
        return a * np.asarray(w) + b

    def abs_to_workout(t):
        return (np.asarray(t) - b) / a

    return workout_to_abs, abs_to_workout


def parse_workout_json(
    workout_path: Path,
    samples: np.ndarray,
    *,
    skip_invalid_segments: bool = True,
) -> list[Segment]:
    """Extract segments + reps from a workout JSON, converted to canonical
    session-relative seconds (matching the t_sec convention in session_io).

    Accepts both the original workout shape and the "aligned export" wrapper
    (where the workout is nested under "workout"), matching plot_imu.py.
    """
    import json
    data = json.loads(workout_path.read_text())
    workout = data.get("workout", data)

    if len(samples) == 0:
        raise ValueError("Can't align workout JSON to empty sample stream")

    # workoutTime is relative to workout start; session-relative t_sec is
    # relative to the FIRST sample. We need the offset between them.
    workout_to_abs, _ = _build_workout_time_mapper(samples)
    first_sample_abs = float(samples["timestamp"][0])
    last_sample_abs = float(samples["timestamp"][-1])

    def wt_to_session_sec(w: float) -> float:
        return float(workout_to_abs(w)) - first_sample_abs

    session_end_sec = last_sample_abs - first_sample_abs

    segments: list[Segment] = []
    dropped = 0
    for round_ in workout.get("roundResults", []):
        for ex in round_.get("exerciseResults", []):
            name = ex.get("name", "Unknown")
            start_wt = ex.get("startTime")
            end_wt = ex.get("endTime")

            if start_wt is None or end_wt is None:
                if skip_invalid_segments:
                    dropped += 1
                    continue
                raise ValueError(f"Segment {name!r}: missing startTime or endTime")

            start_sec = wt_to_session_sec(float(start_wt))
            end_sec = wt_to_session_sec(float(end_wt))

            # Drop segments that are invalid or completely outside the sample
            # stream. Clip ones that partially overlap.
            if end_sec <= start_sec:
                if skip_invalid_segments:
                    dropped += 1
                    continue
                raise ValueError(f"Segment {name!r}: end {end_sec} <= start {start_sec}")
            if end_sec < 0 or start_sec > session_end_sec:
                dropped += 1
                continue

            start_sec = max(start_sec, 0.0)
            end_sec = min(end_sec, session_end_sec)

            reps: list[Rep] = []
            for r in (ex.get("reps") or []):
                rt = r.get("workoutTime")
                if rt is None:
                    continue
                t = wt_to_session_sec(float(rt))
                # Drop reps outside the segment (shouldn't happen, but defensive)
                if start_sec <= t <= end_sec:
                    reps.append(Rep(t=t))

            segments.append(Segment(name=name, start=start_sec, end=end_sec, reps=reps))

    if dropped > 0:
        warnings.warn(
            f"{workout_path.name}: dropped {dropped} invalid/out-of-range segment(s)",
            stacklevel=2,
        )
    return segments


# ---------------------------------------------------------------------------
# Top-level ingestion
# ---------------------------------------------------------------------------

def ingest_imubin(
    imubin_path: str | Path,
    out_dir: str | Path,
    *,
    workout_path: str | Path | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    device_model: str | None = None,
    override_wrist: Wrist | None = None,
    override_crown: Crown | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """End-to-end: read .imubin, canonicalize, attach segments, write session.

    Args:
        imubin_path: input .imubin file.
        out_dir: directory for the output .parquet + .meta.json.
        workout_path: optional workout JSON for segment + rep annotations.
        session_id: override the session ID (default: derived from header).
        user_id: optional user identifier to store in metadata.
        device_model: e.g. "Apple Watch Ultra 2"; stored in metadata.
        override_wrist: override wrist orientation from the .imubin header.
        override_crown: override crown orientation from the .imubin header.
        overwrite: pass through to write_session.

    Returns:
        (parquet_path, meta_path)
    """
    imubin_path = Path(imubin_path)
    buf = imubin_path.read_bytes()

    header = parse_header(buf)
    wrist = override_wrist or header.wrist
    crown = override_crown or header.crown
    samples = parse_samples(buf)
    if len(samples) == 0:
        raise ValueError(f"{imubin_path}: contains no samples")

    rate = empirical_rate_hz(samples)
    if not (MIN_REASONABLE_RATE <= rate <= MAX_REASONABLE_RATE):
        warnings.warn(
            f"{imubin_path.name}: empirical sample rate {rate:.1f} Hz is outside "
            f"the expected range [{MIN_REASONABLE_RATE}, {MAX_REASONABLE_RATE}] Hz. "
            f"File may be corrupt or in an unexpected format.",
            stacklevel=2,
        )

    # Build a DataFrame in the column shape canonicalize_dataframe expects.
    df_raw = pd.DataFrame(
        {
            "acc_x": samples["ax"].astype(np.float32),
            "acc_y": samples["ay"].astype(np.float32),
            "acc_z": samples["az"].astype(np.float32),
            "gyro_x": samples["gx"].astype(np.float32),
            "gyro_y": samples["gy"].astype(np.float32),
            "gyro_z": samples["gz"].astype(np.float32),
        }
    )

    spec = apple_imubin_spec(wrist=wrist, crown=crown)
    df_canon = canonicalize_dataframe(df_raw, spec)

    # Rename to the storage schema (ax/ay/az/gx/gy/gz) before write_session.
    df_canon = df_canon.rename(
        columns={
            "acc_x": "ax", "acc_y": "ay", "acc_z": "az",
            "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz",
        }
    )

    segments: list[Segment] = []
    if workout_path is not None:
        segments = parse_workout_json(Path(workout_path), samples)

    meta = SessionMeta(
        session_id=session_id or header.run_id,
        device="apple_watch",
        device_model=device_model,
        wrist=wrist,
        crown=crown,
        native_sample_rate_hz=NOMINAL_RATE_HZ,
        user_id=user_id,
        workout_id=str(header.workout_id) if header.workout_id else None,
        source_format=f"imubin_v{header.version}",
        source_file=str(imubin_path.resolve()),
        segments=segments,
    )

    return write_session(out_dir, df_canon, meta, overwrite=overwrite)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Ingest an Apple Watch .imubin into the unified session format.",
    )
    p.add_argument("imubin", type=Path, help="Input .imubin file")
    p.add_argument("out_dir", type=Path, help="Output directory for .parquet + .meta.json")
    p.add_argument("--workout", type=Path, default=None, help="Optional workout JSON")
    p.add_argument("--session-id", type=str, default=None,
                   help="Override session ID (default: run_id from header)")
    p.add_argument("--user-id", type=str, default=None)
    p.add_argument("--device-model", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    try:
        pq_path, meta_path = ingest_imubin(
            args.imubin,
            args.out_dir,
            workout_path=args.workout,
            session_id=args.session_id,
            user_id=args.user_id,
            device_model=args.device_model,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {pq_path}")
    print(f"      {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())