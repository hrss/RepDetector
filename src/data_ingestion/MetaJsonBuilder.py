#!/usr/bin/env python3

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def session_id_from_events_path(events_path: Path) -> str:
    name = events_path.name

    if name.endswith(".events.json"):
        return name.removesuffix(".events.json")

    if name.endswith(".json"):
        return name.removesuffix(".json")

    return events_path.stem


def extract_segments(events_data: dict[str, Any], unknown_label: str = "UNKNOWN") -> list[dict[str, Any]]:
    workout = events_data.get("workout", {})
    round_results = workout.get("roundResults", [])

    segments: list[dict[str, Any]] = []

    for round_result in round_results:
        exercise_results = round_result.get("exerciseResults", [])

        for exercise in exercise_results:
            name = exercise.get("name")
            if name is None:
                name = unknown_label

            segment = {
                "name": name,
                "start": exercise.get("startTime"),
                "end": exercise.get("endTime"),
                "reps": exercise.get("reps", []),
            }

            segments.append(segment)

    return segments


def build_meta(
    events_path: Path,
    events_data: dict[str, Any],
    *,
    device: str,
    device_model: str | None,
    wrist: str | None,
    crown: str | None,
    native_sample_rate_hz: int | None,
    canonical_frame_version: str,
    user_id: str | None,
    workout_id: str | None,
    source_format: str,
    source_file: str | None,
    unknown_label: str,
) -> dict[str, Any]:
    session_id = session_id_from_events_path(events_path)

    workout = events_data.get("workout", {})
    imu = events_data.get("imu", {})

    t_start = imu.get("t_start", workout.get("startTime"))
    t_end = imu.get("t_end", workout.get("endTime"))

    duration_sec = None
    if isinstance(t_start, (int, float)) and isinstance(t_end, (int, float)):
        duration_sec = round(float(t_end) - float(t_start), 6)

    n_samples = imu.get("samples")

    if source_file is None:
        source_file = imu.get("file")

    meta = {
        "schema_version": "v1",
        "session_id": session_id,
        "device": device,
        "device_model": device_model,
        "wrist": wrist,
        "crown": crown,
        "native_sample_rate_hz": native_sample_rate_hz,
        "n_samples": n_samples,
        "duration_sec": duration_sec,
        "canonical_frame_version": canonical_frame_version,
        "user_id": user_id,
        "workout_id": workout_id,
        "source_format": source_format,
        "source_file": source_file,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segments": extract_segments(events_data, unknown_label=unknown_label),
    }

    return meta


def default_output_path(events_path: Path) -> Path:
    session_id = session_id_from_events_path(events_path)
    return events_path.with_name(f"{session_id}.meta.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a .events.json workout annotation file into a .meta.json file."
    )

    parser.add_argument(
        "events_json",
        type=Path,
        help="Path to the input .events.json file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the output .meta.json file. Defaults to same folder/session_id.meta.json.",
    )

    parser.add_argument("--device", default="apple_watch")
    parser.add_argument("--device-model", default=None)
    parser.add_argument("--wrist", default=None, choices=["left", "right", None])
    parser.add_argument("--crown", default=None, choices=["left", "right", None])
    parser.add_argument("--native-sample-rate-hz", type=int, default=200)
    parser.add_argument("--canonical-frame-version", default="v1")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--workout-id", default=None)
    parser.add_argument("--source-format", default="events_json")
    parser.add_argument("--source-file", default=None)
    parser.add_argument("--unknown-label", default="UNKNOWN")

    args = parser.parse_args()

    events_path: Path = args.events_json
    output_path: Path = args.output or default_output_path(events_path)

    events_data = load_json(events_path)

    meta = build_meta(
        events_path,
        events_data,
        device=args.device,
        device_model=args.device_model,
        wrist=args.wrist,
        crown=args.crown,
        native_sample_rate_hz=args.native_sample_rate_hz,
        canonical_frame_version=args.canonical_frame_version,
        user_id=args.user_id,
        workout_id=args.workout_id,
        source_format=args.source_format,
        source_file=args.source_file,
        unknown_label=args.unknown_label,
    )

    write_json(output_path, meta)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
