"""
Unified session I/O: one canonicalized recording -> one Parquet + one JSON sidecar.

ON-DISK LAYOUT
==============
    <out_dir>/<session_id>.parquet     # IMU samples, canonical frame + units
    <out_dir>/<session_id>.meta.json   # everything else

PARQUET SCHEMA (strict, enforced via pyarrow)
---------------------------------------------
    sample_idx : int32        # 0, 1, 2, ... — authoritative time reference
    t_sec      : float64      # seconds from session start; = sample_idx / native_sample_rate_hz
    ax, ay, az : float32      # canonical accel, units = g
    gx, gy, gz : float32      # canonical gyro,  units = rad/s

`sample_idx` and `t_sec` are redundant by construction. The writer enforces
their consistency; readers can trust either.

METADATA SCHEMA (JSON)
----------------------
    {
      "schema_version": "v1",
      "session_id":      string,
      "device":          "apple_watch" | "garmin",
      "device_model":    string | null,
      "wrist":           "left" | "right",
      "crown":           "left" | "right" | null,
      "native_sample_rate_hz": int,
      "n_samples":       int,
      "duration_sec":    float,
      "canonical_frame_version": "v1",     # bump if canonicalize.py changes
      "user_id":         string | null,
      "workout_id":      string | null,
      "source_format":   "imubin_v1" | "fit",
      "source_file":     string | null,
      "ingested_at":     ISO-8601 UTC string,
      "segments": [
        {
          "name":  string,
          "start": float,   # seconds, matches t_sec
          "end":   float,
          "reps":  [ {"t": float}, ... ]   # may be empty
        }
      ]
    }

DESIGN NOTES
------------
- Writes are atomic: temp file + rename, so a crashed writer leaves no half-file.
- The reader returns a NamedTuple (df, meta) so call sites can't accidentally
  swap argument order, which they would do with a plain (df, meta) tuple.
- We do NOT inline labels into the dataframe. That's a derived view — apply
  `attach_labels()` at training time, the same way your existing data_loader does.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "v1"
CANONICAL_FRAME_VERSION = "v1"  # bump when canonicalize.py's matrices change

# The Parquet schema as a pyarrow Schema. We pin field types AND order.
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("sample_idx", pa.int32(), nullable=False),
        pa.field("t_sec", pa.float64(), nullable=False),
        pa.field("ax", pa.float32(), nullable=False),
        pa.field("ay", pa.float32(), nullable=False),
        pa.field("az", pa.float32(), nullable=False),
        pa.field("gx", pa.float32(), nullable=False),
        pa.field("gy", pa.float32(), nullable=False),
        pa.field("gz", pa.float32(), nullable=False),
    ]
)

IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")

Device = Literal["apple_watch", "garmin"]
Wrist = Literal["left", "right"]
Crown = Literal["left", "right"]


# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Rep:
    t: float  # seconds from session start


@dataclass
class Segment:
    name: str
    start: float
    end: float
    reps: list[Rep] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Segment {self.name!r}: end {self.end} < start {self.start}")
        # Coerce dict-shaped reps (from JSON) into Rep instances.
        coerced: list[Rep] = []
        for r in self.reps:
            if isinstance(r, Rep):
                coerced.append(r)
            elif isinstance(r, dict):
                coerced.append(Rep(t=float(r["t"])))
            else:
                raise TypeError(f"Rep must be Rep or dict, got {type(r)}")
        self.reps = coerced


@dataclass
class SessionMeta:
    session_id: str
    device: Device
    wrist: Wrist
    native_sample_rate_hz: int
    source_format: str

    # Derived / optional
    device_model: str | None = None
    crown: Crown | None = None
    user_id: str | None = None
    workout_id: str | None = None
    source_file: str | None = None
    segments: list[Segment] = field(default_factory=list)

    # Filled in by the writer
    n_samples: int = 0
    duration_sec: float = 0.0
    schema_version: str = SCHEMA_VERSION
    canonical_frame_version: str = CANONICAL_FRAME_VERSION
    ingested_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "device": self.device,
            "device_model": self.device_model,
            "wrist": self.wrist,
            "crown": self.crown,
            "native_sample_rate_hz": self.native_sample_rate_hz,
            "n_samples": self.n_samples,
            "duration_sec": self.duration_sec,
            "canonical_frame_version": self.canonical_frame_version,
            "user_id": self.user_id,
            "workout_id": self.workout_id,
            "source_format": self.source_format,
            "source_file": self.source_file,
            "ingested_at": self.ingested_at,
            "segments": [
                {
                    "name": s.name,
                    "start": s.start,
                    "end": s.end,
                    "reps": [{"t": r.t} for r in s.reps],
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionMeta":
        # Forward-compatibility: warn on unknown schema versions but still load.
        sv = d.get("schema_version")
        if sv != SCHEMA_VERSION:
            import warnings
            warnings.warn(
                f"Session {d.get('session_id')!r}: schema_version={sv!r} "
                f"differs from current {SCHEMA_VERSION!r}. Proceeding anyway.",
                stacklevel=2,
            )
        segs = [
            Segment(
                name=s["name"],
                start=float(s["start"]),
                end=float(s["end"]),
                reps=[Rep(t=float(r["t"])) for r in s.get("reps", [])],
            )
            for s in d.get("segments", [])
        ]
        return cls(
            session_id=d["session_id"],
            device=d["device"],
            wrist=d["wrist"],
            native_sample_rate_hz=int(d["native_sample_rate_hz"]),
            source_format=d["source_format"],
            device_model=d.get("device_model"),
            crown=d.get("crown"),
            user_id=d.get("user_id"),
            workout_id=d.get("workout_id"),
            source_file=d.get("source_file"),
            segments=segs,
            n_samples=int(d.get("n_samples", 0)),
            duration_sec=float(d.get("duration_sec", 0.0)),
            schema_version=sv or SCHEMA_VERSION,
            canonical_frame_version=d.get("canonical_frame_version", "unknown"),
            ingested_at=d.get("ingested_at", ""),
        )


class LoadedSession(NamedTuple):
    df: pd.DataFrame
    meta: SessionMeta


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _atomic_write(target: Path, write_fn) -> None:
    """Write to a temp file in the same dir, then rename. Same-dir rename is
    atomic on POSIX and on Windows (since 3.3+) for files that don't already
    exist at the target; we tolerate overwrite by using os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write_fn(tmp)
        os.replace(tmp, target)
    except Exception:
        # Best-effort cleanup; don't mask the original exception.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _validate_dataframe(df: pd.DataFrame, expected_rate_hz: int) -> None:
    """Catch the failures-of-imagination before they become bad Parquet files."""
    missing = [c for c in IMU_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if len(df) == 0:
        raise ValueError("DataFrame is empty.")

    # Non-finite values are the #1 cause of mysterious NaN losses in training.
    for col in IMU_COLUMNS:
        arr = df[col].to_numpy()
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"Column {col!r} contains {n_bad} non-finite value(s) "
                f"(NaN or inf). Fix upstream — don't save bad data."
            )

    # If t_sec is present from upstream, sanity-check monotonicity. We'll
    # regenerate it from sample_idx anyway, but a non-monotonic input is a
    # red flag worth surfacing.
    if "t_sec" in df.columns:
        t = df["t_sec"].to_numpy()
        if not np.all(np.diff(t) >= 0):
            n_back = int((np.diff(t) < 0).sum())
            raise ValueError(
                f"t_sec is not monotonically non-decreasing ({n_back} backward steps). "
                f"This usually means timestamps got mangled upstream."
            )


def write_session(
    out_dir: str | Path,
    df: pd.DataFrame,
    meta: SessionMeta,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """
    Write a canonicalized session to <out_dir>/<session_id>.parquet
    and <out_dir>/<session_id>.meta.json. Atomic.

    The DataFrame must already be in the canonical frame and canonical units
    (run it through `canonicalize.canonicalize_dataframe()` first, with the
    accel/gyro columns renamed to ax/ay/az/gx/gy/gz).

    Args:
        df: must have columns ax, ay, az, gx, gy, gz (and may have others, which
            are dropped). Index is ignored. Row order is taken as sample order.
        meta: SessionMeta — n_samples, duration_sec, ingested_at will be filled
              in by this function (any existing values overwritten).
        overwrite: if False, raises FileExistsError when target files exist.

    Returns:
        (parquet_path, meta_path)
    """
    out_dir = Path(out_dir)
    parquet_path = out_dir / f"{meta.session_id}.parquet"
    meta_path = out_dir / f"{meta.session_id}.meta.json"

    if not overwrite:
        for p in (parquet_path, meta_path):
            if p.exists():
                raise FileExistsError(f"{p} exists (pass overwrite=True to replace)")

    _validate_dataframe(df, meta.native_sample_rate_hz)

    n = len(df)
    rate = meta.native_sample_rate_hz
    sample_idx = np.arange(n, dtype=np.int32)
    t_sec = sample_idx.astype(np.float64) / float(rate)

    # Build the pyarrow Table directly so we control types precisely. Doing it
    # via DataFrame.to_parquet leaves room for pandas to silently promote
    # float32 -> float64, which is a subtle disk-bloat trap.
    table = pa.Table.from_arrays(
        [
            pa.array(sample_idx, type=pa.int32()),
            pa.array(t_sec, type=pa.float64()),
            pa.array(df["ax"].to_numpy(dtype=np.float32), type=pa.float32()),
            pa.array(df["ay"].to_numpy(dtype=np.float32), type=pa.float32()),
            pa.array(df["az"].to_numpy(dtype=np.float32), type=pa.float32()),
            pa.array(df["gx"].to_numpy(dtype=np.float32), type=pa.float32()),
            pa.array(df["gy"].to_numpy(dtype=np.float32), type=pa.float32()),
            pa.array(df["gz"].to_numpy(dtype=np.float32), type=pa.float32()),
        ],
        schema=PARQUET_SCHEMA,
    )

    # Fill in derived metadata fields.
    meta.n_samples = n
    meta.duration_sec = float(t_sec[-1]) if n > 0 else 0.0
    meta.ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta.schema_version = SCHEMA_VERSION
    meta.canonical_frame_version = CANONICAL_FRAME_VERSION

    _atomic_write(
        parquet_path,
        lambda p: pq.write_table(
            table,
            p,
            compression="zstd",  # better ratio than snappy at trivial CPU cost
            compression_level=3,
            use_dictionary=False,  # no string cols; dict would only add overhead
        ),
    )
    _atomic_write(
        meta_path,
        lambda p: p.write_text(json.dumps(meta.to_dict(), indent=2)),
    )

    return parquet_path, meta_path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_session(parquet_path: str | Path) -> LoadedSession:
    """
    Load a session by Parquet path. Locates the sidecar JSON automatically.
    Returns LoadedSession(df, meta) where df has the strict Parquet schema and
    meta is a SessionMeta.
    """
    parquet_path = Path(parquet_path)
    meta_path = parquet_path.with_suffix("").with_suffix(".meta.json")
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Sidecar missing: {meta_path}")

    table = pq.read_table(parquet_path)

    # Defensive: confirm the file matches our expected schema. Anything weird
    # (an old file written with the wrong types, a partial overwrite, etc.)
    # surfaces here as a loud error rather than as silent training-time bugs.
    actual_fields = {f.name: f.type for f in table.schema}
    expected_fields = {f.name: f.type for f in PARQUET_SCHEMA}
    for name, expected_type in expected_fields.items():
        if name not in actual_fields:
            raise ValueError(f"{parquet_path}: missing column {name!r}")
        if actual_fields[name] != expected_type:
            raise ValueError(
                f"{parquet_path}: column {name!r} has type {actual_fields[name]}, "
                f"expected {expected_type}"
            )

    df = table.to_pandas()
    meta = SessionMeta.from_dict(json.loads(meta_path.read_text()))

    # Cross-check: row count in Parquet must match meta. If these disagree, one
    # of the two files is stale (e.g. user clobbered the parquet but not the
    # sidecar, or vice versa).
    if len(df) != meta.n_samples:
        raise ValueError(
            f"{parquet_path}: row count {len(df)} != meta.n_samples {meta.n_samples}. "
            f"Sidecar/Parquet mismatch — one was overwritten."
        )

    return LoadedSession(df=df, meta=meta)


def list_sessions(dir_path: str | Path) -> Iterator[Path]:
    """Yield every <id>.parquet under dir_path that has a matching sidecar."""
    dir_path = Path(dir_path)
    for p in sorted(dir_path.rglob("*.parquet")):
        sidecar = p.with_suffix("").with_suffix(".meta.json")
        if sidecar.exists():
            yield p


# ---------------------------------------------------------------------------
# Convenience: attach labels at load time (replaces the inline labeling in
# your existing data_loader.load_raw_section_data)
# ---------------------------------------------------------------------------

def attach_labels(
    df: pd.DataFrame,
    meta: SessionMeta,
    rest_label: str = "Rest",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Add a string label column to df by mapping each sample's t_sec against
    meta.segments. Samples outside any segment get `rest_label`.

    Does NOT modify df in place; returns a new DataFrame.
    """
    if "t_sec" not in df.columns:
        raise ValueError("df must have a t_sec column")

    t = df["t_sec"].to_numpy()
    labels = np.full(len(df), rest_label, dtype=object)

    for seg in meta.segments:
        if seg.name == rest_label:
            continue  # already the default
        mask = (t >= seg.start) & (t <= seg.end)
        labels[mask] = seg.name

    out = df.copy()
    out[label_col] = labels
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Round-trip a synthetic session through write + read."""
    import shutil

    tmpdir = Path(tempfile.mkdtemp(prefix="session_io_test_"))
    try:
        n = 4000  # 200 seconds at 20Hz
        rate = 20
        t = np.arange(n) / rate
        df = pd.DataFrame(
            {
                "ax": np.sin(2 * np.pi * 0.5 * t).astype(np.float32),
                "ay": np.cos(2 * np.pi * 0.5 * t).astype(np.float32),
                "az": np.ones(n, dtype=np.float32),
                "gx": np.zeros(n, dtype=np.float32),
                "gy": np.zeros(n, dtype=np.float32),
                "gz": np.sin(2 * np.pi * 0.1 * t).astype(np.float32),
            }
        )
        meta = SessionMeta(
            session_id="test_session_001",
            device="garmin",
            wrist="left",
            native_sample_rate_hz=rate,
            source_format="fit",
            device_model="Fenix 7",
            user_id="u_42",
            workout_id="w_99",
            source_file="/data/2053/raw.fit",
            segments=[
                Segment(
                    name="Air Squat",
                    start=10.0,
                    end=40.5,
                    reps=[Rep(t=12.3), Rep(t=15.1), Rep(t=18.0)],
                ),
                Segment(name="Shoulder Press", start=50.0, end=80.0, reps=[]),
            ],
        )

        pq_path, meta_path = write_session(tmpdir, df, meta)
        assert pq_path.exists() and meta_path.exists()

        loaded = read_session(pq_path)
        assert loaded.df.shape == (n, 8)  # sample_idx, t_sec, + 6 IMU
        assert loaded.df["sample_idx"].iloc[0] == 0
        assert loaded.df["sample_idx"].iloc[-1] == n - 1
        np.testing.assert_allclose(loaded.df["t_sec"].iloc[-1], (n - 1) / rate)
        np.testing.assert_allclose(loaded.df["ax"].to_numpy(), df["ax"].to_numpy())
        assert loaded.meta.session_id == "test_session_001"
        assert loaded.meta.n_samples == n
        assert len(loaded.meta.segments) == 2
        assert len(loaded.meta.segments[0].reps) == 3
        assert loaded.meta.segments[0].reps[0].t == 12.3

        # attach_labels
        labeled = attach_labels(loaded.df, loaded.meta)
        # t=0 -> Rest, t=10..40.5 -> Air Squat, t=50..80 -> Shoulder Press
        assert labeled.loc[labeled["t_sec"] == 0.0, "label"].iloc[0] == "Rest"
        assert labeled.loc[labeled["t_sec"] == 20.0, "label"].iloc[0] == "Air Squat"
        assert labeled.loc[labeled["t_sec"] == 60.0, "label"].iloc[0] == "Shoulder Press"

        # overwrite guard
        try:
            write_session(tmpdir, df, meta)
        except FileExistsError:
            pass
        else:
            raise AssertionError("overwrite guard didn't fire")

        # overwrite=True works
        write_session(tmpdir, df, meta, overwrite=True)

        # validation: non-finite values rejected
        bad = df.copy()
        bad.loc[100, "ax"] = np.nan
        bad_meta = SessionMeta(
            session_id="bad_session",
            device="garmin",
            wrist="left",
            native_sample_rate_hz=rate,
            source_format="fit",
        )
        try:
            write_session(tmpdir, bad, bad_meta)
        except ValueError as e:
            assert "non-finite" in str(e)
        else:
            raise AssertionError("NaN validation didn't fire")

        print("session_io._selftest: OK")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    _selftest()
