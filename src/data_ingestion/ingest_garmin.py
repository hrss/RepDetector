"""
Garmin FIT -> unified canonical session format.

Pipeline:
    .fit file + (optional) section JSON paths
        -> fitslicer.extract_raw_fit_data (already g + rad/s, uniform 25Hz grid)
        -> canonicalize (rotate)
        -> attach segments + reps from section JSONs
        -> write_session(...)

Assumptions (per fitslicer.py):
    - Input FIT uses the 25 Hz / 10-bit-gyro Connect IQ wire format.
    - extract_raw_fit_data returns a DataFrame with columns:
        rel_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
      with rel_time in seconds (uniform 25 Hz grid), acc in g, gyro in rad/s,
      and NaN gaps already interpolated.

Section JSON shape (matches Apple workout JSON):
    {
      "sectionId": ...,
      "startTime": float,         # section bounds in seconds (FIT-relative)
      "endTime":   float,
      "roundResults": [
        {
          "exerciseResults": [
            {"name": ..., "startTime": float, "endTime": float,
             "reps": [{"workoutTime": float}, ...]}
          ]
        }
      ]
    }
Multiple section JSONs are accepted per ingestion; all exercises from all
sections are flattened into the session's segments list.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from canonicalize import (
    Wrist,
    canonicalize_dataframe,
    garmin_fit_spec,
)
from session_io import Rep, Segment, SessionMeta, write_session
from fit_slicer import extract_raw_fit_data, SAMPLE_RATE as FIT_SAMPLE_RATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOMINAL_RATE_HZ = FIT_SAMPLE_RATE  # 25 Hz, from fitslicer

# Tighter bounds than Apple because the Garmin grid is exact-by-construction
# in fitslicer (no jitter). If the empirical median ever deviates, the FIT
# format has changed.
MIN_REASONABLE_RATE = 24.0
MAX_REASONABLE_RATE = 26.0


# ---------------------------------------------------------------------------
# Section JSON parsing
# ---------------------------------------------------------------------------

def parse_section_jsons(
    json_paths: list[Path],
    session_duration_sec: float,
    *,
    skip_invalid_segments: bool = True,
) -> list[Segment]:
    """Parse one or more section JSON files into a flat list of Segments.

    For Garmin, exercise startTime/endTime are already in seconds relative to
    the FIT start, which IS the session start — so no time mapping is needed
    (unlike the Apple path).
    """
    segments: list[Segment] = []
    dropped = 0

    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            warnings.warn(f"Skipping {json_path}: {e}", stacklevel=2)
            continue

        for round_ in data.get("roundResults", []):
            for ex in round_.get("exerciseResults", []):
                name = ex.get("name", "Unknown")
                start = ex.get("startTime")
                end = ex.get("endTime")

                if start is None or end is None:
                    if skip_invalid_segments:
                        dropped += 1
                        continue
                    raise ValueError(
                        f"{json_path.name}: segment {name!r} missing startTime/endTime"
                    )

                start = float(start)
                end = float(end)

                if end <= start:
                    if skip_invalid_segments:
                        dropped += 1
                        continue
                    raise ValueError(
                        f"{json_path.name}: segment {name!r} has end <= start "
                        f"({end} <= {start})"
                    )

                # Drop segments entirely outside the FIT timeline; clip ones
                # that partially overlap.
                if end < 0 or start > session_duration_sec:
                    dropped += 1
                    continue

                start = max(start, 0.0)
                end = min(end, session_duration_sec)

                reps: list[Rep] = []
                for r in (ex.get("reps") or []):
                    rt = r.get("workoutTime")
                    if rt is None:
                        continue
                    t = float(rt)
                    if start <= t <= end:
                        reps.append(Rep(t=t))

                segments.append(Segment(name=name, start=start, end=end, reps=reps))

    if dropped > 0:
        warnings.warn(
            f"Dropped {dropped} invalid/out-of-range segment(s) across "
            f"{len(json_paths)} section JSON(s)",
            stacklevel=2,
        )
    return segments


# ---------------------------------------------------------------------------
# Top-level ingestion
# ---------------------------------------------------------------------------

def ingest_fit(
    fit_path: str | Path,
    out_dir: str | Path,
    *,
    wrist: Wrist,
    section_json_paths: list[str | Path] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    workout_id: str | None = None,
    device_model: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """End-to-end: read .fit, canonicalize, attach segments, write session.

    Args:
        fit_path: input .fit file (25 Hz / 10-bit-gyro Connect IQ format).
        out_dir: directory for the output .parquet + .meta.json.
        wrist: which wrist this watch was worn on. Required (Garmin FIT does
               not encode this).
        section_json_paths: optional list of section JSON files providing
                            exercise segment + rep annotations.
        session_id: override session ID (default: FIT filename stem).
        user_id: optional user identifier to store in metadata.
        workout_id: optional workout identifier (not encoded in FIT).
        device_model: e.g. "Fenix 7"; stored in metadata.
        overwrite: pass through to write_session.

    Returns:
        (parquet_path, meta_path)
    """
    fit_path = Path(fit_path)
    if not fit_path.exists():
        raise FileNotFoundError(fit_path)

    df_raw = extract_raw_fit_data(str(fit_path))
    if df_raw.empty:
        raise ValueError(f"{fit_path}: FIT extraction returned no data")

    # Sanity-check the empirical sample rate.
    if len(df_raw) >= 2:
        dt = float(np.median(np.diff(df_raw["rel_time"].to_numpy())))
        rate = 1.0 / dt if dt > 0 else float("nan")
        if not (MIN_REASONABLE_RATE <= rate <= MAX_REASONABLE_RATE):
            warnings.warn(
                f"{fit_path.name}: empirical sample rate {rate:.2f} Hz is outside "
                f"the expected range [{MIN_REASONABLE_RATE}, {MAX_REASONABLE_RATE}] Hz. "
                f"FIT format may have changed.",
                stacklevel=2,
            )

    # Canonicalize. Input is already in canonical units (g, rad/s) from
    # fitslicer; this rotation is the only transform left.
    spec = garmin_fit_spec(wrist=wrist)
    df_canon = canonicalize_dataframe(df_raw, spec)

    # Rename to the storage schema (ax/ay/az/gx/gy/gz) before write_session.
    df_canon = df_canon.rename(
        columns={
            "acc_x": "ax", "acc_y": "ay", "acc_z": "az",
            "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz",
        }
    )

    # Segment parsing. Use the LAST rel_time as the upper bound for clipping.
    session_duration_sec = float(df_canon["rel_time"].iloc[-1])
    segments: list[Segment] = []
    if section_json_paths:
        segments = parse_section_jsons(
            [Path(p) for p in section_json_paths],
            session_duration_sec,
        )

    # Drop rel_time before writing — write_session regenerates t_sec from
    # sample_idx anyway. This also avoids the validation tripping on rel_time
    # vs t_sec mismatch.
    df_canon = df_canon.drop(columns=["rel_time"])

    meta = SessionMeta(
        session_id=session_id or fit_path.stem,
        device="garmin",
        device_model=device_model,
        wrist=wrist,
        crown=None,
        native_sample_rate_hz=NOMINAL_RATE_HZ,
        user_id=user_id,
        workout_id=workout_id,
        source_format="fit",
        source_file=str(fit_path.resolve()),
        segments=segments,
    )

    return write_session(out_dir, df_canon, meta, overwrite=overwrite)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Ingest a Garmin .fit into the unified session format.",
    )
    p.add_argument("fit", type=Path, help="Input .fit file")
    p.add_argument("out_dir", type=Path, help="Output directory for .parquet + .meta.json")
    p.add_argument("--wrist", choices=("left", "right"), required=True,
                   help="Wrist the watch was worn on (FIT doesn't encode this)")
    p.add_argument("--section-json", type=Path, action="append", default=[],
                   help="Section JSON file with exercise annotations. May be "
                        "repeated for multi-section workouts.")
    p.add_argument("--session-id", type=str, default=None,
                   help="Override session ID (default: FIT filename stem)")
    p.add_argument("--user-id", type=str, default=None)
    p.add_argument("--workout-id", type=str, default=None)
    p.add_argument("--device-model", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    try:
        pq_path, meta_path = ingest_fit(
            args.fit,
            args.out_dir,
            wrist=args.wrist,
            section_json_paths=args.section_json or None,
            session_id=args.session_id,
            user_id=args.user_id,
            workout_id=args.workout_id,
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