import os
import glob
import json
import pandas as pd
import fitdecode

# --- CONFIGURATION ---
SAMPLE_RATE = 20
ACCEL_DIVISOR = 102.0
GYRO_DIVISOR = 100.0


def extract_raw_fit_data(fit_path):
    """Unrolls the 20Hz arrays and returns pure raw data."""
    print(f"Extracting raw 20Hz data from: {fit_path}...")
    data_points = []
    start_time = None

    with fitdecode.FitReader(fit_path) as reader:
        for frame in reader:
            if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == 'record':
                ts = frame.get_value('timestamp')
                if start_time is None:
                    start_time = ts

                def get_val(frame, field):
                    val = frame.get_value(field)
                    return val if isinstance(val, (list, tuple)) else [val]

                acc_x = get_val(frame, 'accel_x')
                acc_y = get_val(frame, 'accel_y')
                acc_z = get_val(frame, 'accel_z')
                gyr_x = get_val(frame, 'gyro_x')
                gyr_y = get_val(frame, 'gyro_y')
                gyr_z = get_val(frame, 'gyro_z')

                for i, val in enumerate(acc_x):
                    if val is None: continue

                    t_offset = (ts - start_time).total_seconds() + (i / SAMPLE_RATE)

                    data_points.append({
                        'rel_time': t_offset,
                        'acc_x': val / ACCEL_DIVISOR,
                        'acc_y': acc_y[i] / ACCEL_DIVISOR,
                        'acc_z': acc_z[i] / ACCEL_DIVISOR,
                        'gyro_x': gyr_x[i] / GYRO_DIVISOR,
                        'gyro_y': gyr_y[i] / GYRO_DIVISOR,
                        'gyro_z': gyr_z[i] / GYRO_DIVISOR
                    })

    return pd.DataFrame(data_points)


def slice_and_save_raw(fit_df, section_json, section_dir):
    """Cuts the continuous dataframe and saves the isolated raw unit."""
    section_id = section_json.get("sectionId", "unknown_id")
    start_time = section_json.get("startTime")
    end_time = section_json.get("endTime")

    if start_time is None or end_time is None:
        print(f"Skipping section {section_id}: Missing start or end time.")
        return False

    # -1 usually means unfinished; skip those
    # if start_time < 0 or end_time < 0 or end_time <= start_time:
    #     print(f"Skipping section {section_id}: invalid time range ({start_time}, {end_time}).")
    #     return False

    section_df = fit_df[(fit_df['rel_time'] >= start_time) & (fit_df['rel_time'] <= end_time)].copy()

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


if __name__ == "__main__":
    process_workout_directory(workout_dir="data/1797")
    pass