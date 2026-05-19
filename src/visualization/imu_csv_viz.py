import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt
import argparse

# --- CONFIGURATION ---
SAMPLE_RATE = 20


def apply_butterworth(data, cutoff, fs, order=4):
    """Applies a zero-phase low-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # filtfilt applies the filter forward and backward to ensure zero phase shift
    return filtfilt(b, a, data)


def visualize_imu_csv(csv_path, apply_filter=False, cutoff_hz=3.0, output_path=None):
    """
    Reads a CSV file with IMU data and generates a 6-axis visualization.
    Expected columns: rel_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        return

    # 1. Load data
    df = pd.read_csv(csv_path)

    # 2. Apply Butterworth Filter (If flagged)
    if apply_filter:
        cols_to_filter = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        for col in cols_to_filter:
            if col in df.columns:
                df[col] = apply_butterworth(df[col], cutoff=cutoff_hz, fs=SAMPLE_RATE)

    # 3. Initialize UI
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Accelerometer (m/s² or raw)", "Gyroscope (deg/s or raw)")
    )

    # Track 1: Accelerometer
    acc_colors = {'acc_x': '#FF4B4B', 'acc_y': '#2ECC71', 'acc_z': '#3498DB'}
    for col in ['acc_x', 'acc_y', 'acc_z']:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df['rel_time'], y=df[col], mode='lines', name=col, line=dict(color=acc_colors[col])),
                row=1, col=1)

    # Track 2: Gyroscope
    gyro_colors = {'gyro_x': '#E67E22', 'gyro_y': '#9B59B6', 'gyro_z': '#F1C40F'}
    for col in ['gyro_x', 'gyro_y', 'gyro_z']:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df['rel_time'], y=df[col], mode='lines', name=col, line=dict(color=gyro_colors[col])),
                row=2, col=1)

    # 4. UI Polish
    filter_status = f" | Filtered: {cutoff_hz}Hz" if apply_filter else " | RAW"
    file_name = os.path.basename(csv_path)
    fig.update_layout(
        title=f"IMU 6-Axis Visualization: {file_name}{filter_status}",
        height=800, template="plotly_dark", hovermode="x unified"
    )

    # 5. Save output
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        suffix = "_filtered.html" if apply_filter else "_raw.html"
        output_path = base_name + suffix

    fig.write_html(output_path)
    print(f"Visualizer saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 6-axis IMU data from a CSV file.")
    parser.add_argument("csv_path", help="Path to the IMU CSV file")
    parser.add_argument("--filter", action="store_true", help="Apply Butterworth low-pass filter")
    parser.add_argument("--cutoff", type=float, default=3.0, help="Cutoff frequency for the filter (default: 3.0Hz)")
    parser.add_argument("--output", help="Path to save the output HTML file")

    args = parser.parse_args()

    visualize_imu_csv(args.csv_path, apply_filter=args.filter, cutoff_hz=args.cutoff, output_path=args.output)
