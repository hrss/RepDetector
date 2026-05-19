"""
FIT extractor for the Garmin 25 Hz / 10-bit-gyro wire format.

Wire format (per RECORD frame, one per second):
  accel_x/y/z: SINT16 × 25, units = mg (divide by 1000 -> g)
  gyro_x_lo:   UINT8  × 25, low 8 bits of biased SINT10
  gyro_y_lo:   UINT8  × 25
  gyro_z_lo:   UINT8  × 25
  gyro_hi:     UINT8  × 19, high 2 bits of each axis sample, packed sequentially
                            bits 6N..6N+5 = (Z<<4)|(Y<<2)|X for sample N
  Scaler = 1.0, so stored SINT10 IS deg/s.

Output CSV schema:
  rel_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
  - rel_time: seconds since first sample (float)
  - acc_*:    g
  - gyro_*:   deg/s
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import fitdecode

# --- CONFIGURATION ---
SAMPLE_RATE = 25
ACCEL_DIVISOR = 1000.0   # mg -> g
GYRO_HI_BYTES = 19       # ceil(25 * 6 / 8)


# ---------------------------------------------------------------------------
# 10-bit gyro unpacking (mirror of Monkey C encoder)
# ---------------------------------------------------------------------------

def _unpack_gyro_hi(hi_bytes):
    """
    Unpack the 19-byte gyro_hi field into three arrays of 25 high-2-bit values.
    Returns (hiX, hiY, hiZ), each a uint8 numpy array of length 25.

    None values in the input (from FIT-layer "invalid" sentinels for unwritten
    bytes) are treated as 0. The corresponding samples will still be dropped
    downstream because their lo bytes will also be None.
    """
    if len(hi_bytes) != GYRO_HI_BYTES:
        raise ValueError(
            f"Expected {GYRO_HI_BYTES} bytes for gyro_hi, got {len(hi_bytes)}"
        )

    # Materialize as a list, replacing None with 0.
    hi_clean = [(b if b is not None else 0) for b in hi_bytes]
    hi = np.asarray(hi_clean, dtype=np.uint8)

    hiX = np.zeros(SAMPLE_RATE, dtype=np.uint8)
    hiY = np.zeros(SAMPLE_RATE, dtype=np.uint8)
    hiZ = np.zeros(SAMPLE_RATE, dtype=np.uint8)

    for n in range(SAMPLE_RATE):
        six = 0
        start_bit = n * 6
        for b in range(6):
            abs_bit = start_bit + b
            bit_val = (int(hi[abs_bit // 8]) >> (abs_bit % 8)) & 1
            six |= bit_val << b
        hiX[n] = six & 0x3
        hiY[n] = (six >> 2) & 0x3
        hiZ[n] = (six >> 4) & 0x3

    return hiX, hiY, hiZ


def _decode_gyro_sint10(lo, hi2):
    """Reconstruct SINT10 -> deg/s (scaler 1.0)."""
    lo = np.asarray(lo, dtype=np.uint16)
    hi2 = np.asarray(hi2, dtype=np.uint16)
    biased = (hi2 << 8) | lo                  # uint10 in [0, 1023]
    return (biased.astype(np.int16) - 512).astype(np.float32)


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else [value]


# ---------------------------------------------------------------------------
# Frame decoder
# ---------------------------------------------------------------------------

def _decode_frame(frame, t_offset_seconds):
    """Decode one RECORD frame into a list of 25 sample dicts (or fewer if
    the whole frame is empty).

    Within a frame, samples where any of the 6 source bytes is None are emitted
    with NaN values. They get filled by linear interpolation in the post-pass.
    Frames where the entire payload is None (e.g. the first record before the
    sensor stream starts) are dropped entirely.
    """
    acc_x = _as_list(frame.get_value('accel_x'))
    acc_y = _as_list(frame.get_value('accel_y'))
    acc_z = _as_list(frame.get_value('accel_z'))

    lo_x = _as_list(frame.get_value('gyro_x_lo'))
    lo_y = _as_list(frame.get_value('gyro_y_lo'))
    lo_z = _as_list(frame.get_value('gyro_z_lo'))
    hi   = frame.get_value('gyro_hi')

    if hi is None:
        return []
    hi_bytes = list(hi) if isinstance(hi, (list, tuple)) else [hi]

    # All-None frame (first record before sensor stream starts).
    if all(b is None for b in hi_bytes):
        return []

    try:
        hiX, hiY, hiZ = _unpack_gyro_hi(hi_bytes)
    except ValueError as e:
        print(f"Warning: malformed gyro_hi in frame ({e}); skipping frame.")
        return []

    # Replace None lo-bytes with 0 so the SINT10 decoder runs cleanly. The
    # decoded gyro values at those positions are meaningless and will be
    # overwritten with NaN below.
    lo_x_clean = [(v if v is not None else 0) for v in lo_x[:SAMPLE_RATE]]
    lo_y_clean = [(v if v is not None else 0) for v in lo_y[:SAMPLE_RATE]]
    lo_z_clean = [(v if v is not None else 0) for v in lo_z[:SAMPLE_RATE]]

    gx_all = _decode_gyro_sint10(lo_x_clean, hiX)
    gy_all = _decode_gyro_sint10(lo_y_clean, hiY)
    gz_all = _decode_gyro_sint10(lo_z_clean, hiZ)

    out = []
    for i in range(SAMPLE_RATE):
        # Check if THIS sample is recoverable (all 6 source bytes present).
        valid = (i < len(acc_x) and acc_x[i] is not None and
                 i < len(acc_y) and acc_y[i] is not None and
                 i < len(acc_z) and acc_z[i] is not None and
                 i < len(lo_x) and lo_x[i] is not None and
                 i < len(lo_y) and lo_y[i] is not None and
                 i < len(lo_z) and lo_z[i] is not None)

        if valid:
            out.append({
                'rel_time': t_offset_seconds + (i / SAMPLE_RATE),
                'acc_x': acc_x[i] / ACCEL_DIVISOR,
                'acc_y': acc_y[i] / ACCEL_DIVISOR,
                'acc_z': acc_z[i] / ACCEL_DIVISOR,
                'gyro_x': float(gx_all[i]),
                'gyro_y': float(gy_all[i]),
                'gyro_z': float(gz_all[i]),
            })
        else:
            out.append({
                'rel_time': t_offset_seconds + (i / SAMPLE_RATE),
                'acc_x': np.nan, 'acc_y': np.nan, 'acc_z': np.nan,
                'gyro_x': np.nan, 'gyro_y': np.nan, 'gyro_z': np.nan,
            })
    return out


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------

def extract_raw_fit_data(fit_path):
    """
    Read a FIT file and return a DataFrame of raw IMU samples on a uniform
    25 Hz grid:
        rel_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    Units: acc in g, gyro in deg/s, time in seconds from first sample.

    Gaps in the source data (None bytes from FIT-layer "invalid" sentinels)
    are filled by linear interpolation; boundary gaps use nearest-neighbor.
    Records that are entirely None (e.g. a placeholder first record) are
    dropped before interpolation, so the timeline starts at the first record
    with any valid data.
    """
    print(f"Extracting raw FIT data from: {fit_path}...")
    data_points = []
    start_time = None

    with fitdecode.FitReader(fit_path) as reader:
        for frame in reader:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA or frame.name != 'record':
                continue

            ts = frame.get_value('timestamp')
            if ts is None:
                continue
            if start_time is None:
                start_time = ts

            t_offset = (ts - start_time).total_seconds()
            data_points.extend(_decode_frame(frame, t_offset))

    df = pd.DataFrame(data_points)
    if df.empty:
        return df

    return _fill_gaps(df)


def _fill_gaps(df):
    """
    Fill NaN gaps in the sensor columns by linear interpolation, with
    nearest-neighbor fill at the boundaries. The rel_time column is left
    alone (it's already a uniform 25 Hz grid by construction).

    Reports how many samples were filled, so a noisy input is visible.
    """
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    total_nans = int(df[sensor_cols].isna().any(axis=1).sum())
    if total_nans == 0:
        return df

    pct = 100.0 * total_nans / len(df)
    print(f"  Interpolating {total_nans} gap samples ({pct:.1f}% of {len(df)} rows)")

    # 'linear' interpolates between valid endpoints; 'limit_direction=both'
    # combined with bfill/ffill handles leading/trailing NaNs by extending
    # the nearest valid value.
    df = df.copy()
    df[sensor_cols] = df[sensor_cols].interpolate(method='linear', limit_direction='both')
    # If a sensor column was 100% NaN (shouldn't happen, but defensive),
    # interpolate leaves it as NaN. Drop those rows.
    df = df.dropna(subset=sensor_cols)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CSV writing & section slicing
# ---------------------------------------------------------------------------

def save_fit_as_csv(fit_path, output_csv_path=None):
    """Extracts raw IMU data from a single FIT file and saves it as CSV."""
    if not os.path.exists(fit_path):
        print(f"Error: FIT file does not exist: {fit_path}")
        return False
    if not fit_path.lower().endswith(".fit"):
        print(f"Error: Expected a .fit file, got: {fit_path}")
        return False

    if output_csv_path is None:
        base_path, _ = os.path.splitext(fit_path)
        output_csv_path = f"{base_path}.csv"

    df = extract_raw_fit_data(fit_path)
    if df.empty:
        print("Error: FIT extraction yielded no data.")
        return False

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df)} raw rows to: {output_csv_path}")
    return True


def slice_and_save_raw(fit_df, section_json, section_dir):
    """Cuts the continuous dataframe and saves the isolated raw unit."""
    section_id = section_json.get("sectionId", "unknown_id")
    start_time = section_json.get("startTime")
    end_time = section_json.get("endTime")

    if start_time is None or end_time is None:
        print(f"Skipping section {section_id}: Missing start or end time.")
        return False

    section_df = fit_df[
        (fit_df['rel_time'] >= start_time) & (fit_df['rel_time'] <= end_time)
    ].copy()

    if section_df.empty:
        print(f"Warning: No data found for section {section_id} ({start_time}s - {end_time}s)")
        return False

    csv_path = os.path.join(section_dir, "imu.csv")
    section_df.to_csv(csv_path, index=False)
    print(f"  -> Saved {len(section_df)} raw rows to: {csv_path}")
    return True


def process_workout_directory(workout_dir):
    """
    Processes one workout_result directory:
      workout_dir/
        *.fit
        section_<id>/
          data.json
    """
    print(f"\nScanning workout directory: {workout_dir}")

    fit_files = glob.glob(os.path.join(workout_dir, "*.fit"))
    if not fit_files:
        print(f"Error: No .fit file found in {workout_dir}")
        return
    if len(fit_files) > 1:
        print(f"Warning: Multiple .fit files found. Using the first one: {fit_files[0]}")

    target_fit = fit_files[0]

    section_dirs = [
        path for path in glob.glob(os.path.join(workout_dir, "section_*"))
        if os.path.isdir(path)
    ]
    if not section_dirs:
        print(f"Error: No section subdirectories found in {workout_dir}")
        return

    full_df = extract_raw_fit_data(target_fit)
    if full_df.empty:
        print("Error: FIT extraction yielded no data.")
        return

    success_count = 0
    for section_dir in section_dirs:
        data_json_path = os.path.join(section_dir, "data.json")
        if not os.path.exists(data_json_path):
            print(f"Skipping {section_dir}: missing data.json")
            continue
        try:
            with open(data_json_path, 'r', encoding="utf-8") as f:
                section_data = json.load(f)
            if "sectionId" in section_data and "startTime" in section_data:
                if slice_and_save_raw(full_df, section_data, section_dir):
                    success_count += 1
            else:
                print(f"Skipping {section_dir}: invalid data.json")
        except Exception as e:
            print(f"Failed to process {section_dir}: {e}")

    print(f"Workout processing complete. Successfully sliced {success_count}/{len(section_dirs)} sections.")


def process_data_root(data_root):
    if not os.path.isdir(data_root):
        print(f"Error: data root does not exist or is not a directory: {data_root}")
        return
    for dir_name in os.listdir(data_root):
        workout_path = os.path.join(data_root, dir_name)
        if os.path.isdir(workout_path):
            process_workout_directory(workout_dir=workout_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw IMU data from Garmin FIT files (25 Hz, 10-bit gyro format)."
    )
    parser.add_argument(
        "fit_file",
        nargs="?",
        help="Optional path to a single .fit file. If provided, a CSV is generated."
    )
    parser.add_argument(
        "-o", "--output",
        help="Optional output CSV path when processing a single .fit file."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root for the workout directory slicing flow. Default: data"
    )
    args = parser.parse_args()

    if args.fit_file:
        save_fit_as_csv(args.fit_file, args.output)
    else:
        process_data_root(args.data_root)