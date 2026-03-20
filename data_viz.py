import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, spectrogram

# --- CONFIGURATION ---
SAMPLE_RATE = 20


def apply_butterworth(data, cutoff, fs, order=4):
    """Applies a zero-phase low-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # filtfilt applies the filter forward and backward to ensure zero phase shift
    return filtfilt(b, a, data)


def calculate_kinematics(df):
    """Calculates derived features on the fly from X/Y/Z data."""
    df['smv_acc'] = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2)
    df['smv_gyro'] = np.sqrt(df['gyro_x'] ** 2 + df['gyro_y'] ** 2 + df['gyro_z'] ** 2)

    # Gravity vector approximation for orientation
    df['pitch'] = np.degrees(np.arctan2(df['acc_x'], np.sqrt(df['acc_y'] ** 2 + df['acc_z'] ** 2)))
    df['roll'] = np.degrees(np.arctan2(df['acc_y'], df['acc_z']))

    return df


def generate_dashboard(section_dir, apply_filter=False, cutoff_hz=3.0):
    """Reads raw CSV, optionally filters, calculates kinematics, and generates UI."""
    csv_path = os.path.join(section_dir, "imu.csv")
    json_path = os.path.join(section_dir, "data.json")

    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        print(f"Error: Missing CSV or JSON in {section_dir}")
        return

    # 1. Load raw data and metadata
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    section_id = metadata["sectionId"]
    start_time = metadata["startTime"]

    # 2. Apply Butterworth Filter (If flagged)
    if apply_filter:
        cols_to_filter = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        for col in cols_to_filter:
            df[col] = apply_butterworth(df[col], cutoff=cutoff_hz, fs=SAMPLE_RATE)

    # 3. Calculate physics just in time for plotting
    df = calculate_kinematics(df)

    # 4. Initialize UI
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Energy (SMV)", "Rhythm (Spectrogram)", "Orientation (Pitch & Roll)")
    )

    # Track 1: Energy
    fig.add_trace(
        go.Scatter(x=df['rel_time'], y=df['smv_acc'], mode='lines', name='Accel SMV', line=dict(color='#00F0FF')),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df['rel_time'], y=df['smv_gyro'], mode='lines', name='Gyro SMV', line=dict(color='#FF0055')),
        row=1, col=1)

    # Track 2: Rhythm (Spectrogram strictly on 20Hz)
    f, t, Sxx = spectrogram(df['smv_acc'].values, fs=SAMPLE_RATE, nperseg=32, noverlap=16)
    t_shifted = t + start_time
    fig.add_trace(go.Heatmap(x=t_shifted, y=f, z=10 * np.log10(Sxx + 1e-10), colorscale='Viridis', showscale=False),
                  row=2, col=1)

    # Track 3: Orientation
    fig.add_trace(go.Scatter(x=df['rel_time'], y=df['pitch'], mode='lines', name='Pitch', line=dict(color='#00FF00')),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df['rel_time'], y=df['roll'], mode='lines', name='Roll', line=dict(color='#FFAA00')),
                  row=3, col=1)

    # 5. Overlay Labels
    for round_data in metadata.get("roundResults", []):
        for exercise in round_data.get("exerciseResults", []):
            fig.add_vrect(
                x0=exercise["startTime"], x1=exercise["endTime"],
                fillcolor="green", opacity=0.15, layer="below", line_width=0,
                annotation_text=f"{exercise['name']} ({exercise['result']} reps)",
                annotation_position="top left", row="all", col=1
            )

        fig.add_vrect(
            x0=round_data["restingTimeStart"], x1=round_data["endTime"],
            fillcolor="gray", opacity=0.2, layer="below", line_width=0,
            annotation_text="Rest", annotation_position="top left", row="all", col=1
        )

    # 6. UI Polish (Update Title based on filter flag)
    filter_status = f" | Filtered: {cutoff_hz}Hz" if apply_filter else " | RAW"
    fig.update_layout(
        title=f"WodBuddy Analysis: Section {section_id[:8]}{filter_status}",
        height=900, template="plotly_dark", hovermode="x unified"
    )

    # Save output with a name reflecting the filter state
    file_name = "visualizer_filtered.html" if apply_filter else "visualizer_raw.html"
    html_path = os.path.join(section_dir, file_name)
    fig.write_html(html_path)
    print(f"Visualizer saved to: {html_path}")


if __name__ == "__main__":
    # Example usage:
    directory = "dataset/section_c11b7a06-0606-444f-b598-8343665bc5a5"

    # Generate the Raw Dashboard
    # generate_dashboard(directory, apply_filter=False)

    # Generate the Filtered Dashboard
    generate_dashboard(directory, apply_filter=True, cutoff_hz=3.0)
    pass