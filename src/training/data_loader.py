"""
Data Loader Module for Exercise Recognition Training

This module provides utilities for loading, processing, and windowing IMU sensor data
from workout sessions for machine learning model training. It supports both PyTorch
neural networks and traditional ML models (e.g., Decision Trees).

Key Components:
---------------
1. load_raw_section_data(): Loads and preprocesses a single workout section
   - Reads IMU CSV and metadata JSON files
   - Applies Butterworth low-pass filtering to sensor signals
   - Assigns exercise labels based on temporal annotations

2. WodDataset: PyTorch Dataset for neural network training
   - Loads data from directory structure with section_* subdirectories
   - Creates sliding windows from continuous sensor streams
   - Returns tensors shaped [Channels=6, Sequence Length]
   - Automatically encodes labels to integers

3. load_and_window_data_for_dt(): Prepares data for Decision Tree models
   - Similar windowing approach as WodDataset
   - Extracts statistical features from each window
   - Supports class filtering via 'allowed_classes' config
   - Returns feature vectors and encoded labels

Usage Example:
--------------
    # For PyTorch models (LSTM, CNN, etc.):
    config = {
        'window_size_sec': 1.0,
        'step_size_sec': 0.05,
        'sample_rate': 20,
        'lowpass_cutoff': 5.0
    }
    dataset = WodDataset('data/training', config)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # For Decision Tree models:
    X, y, label_encoder = load_and_window_data_for_dt('data/training', config)
    model = DecisionTreeClassifier()
    model.fit(X, y)

Expected Directory Structure:
----------------------------
    data/
      workout_result_xxx/
        section_abc123/
          imu.csv          # Sensor data with columns: rel_time, acc_x/y/z, gyro_x/y/z
          data.json        # Metadata with roundResults and exerciseResults

Configuration Parameters:
------------------------
    - window_size_sec: Duration of each analysis window (typically 1.0 second)
    - step_size_sec: Sliding window step size (typically 0.05 for 95% overlap)
    - sample_rate: IMU sampling frequency in Hz (typically 20 Hz)
    - lowpass_cutoff: Butterworth filter cutoff frequency in Hz (typically 5.0 Hz)
    - allowed_classes: Optional list of exercise names to include (for filtering)
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from src.core.exercises import canonicalize_label
from src.core.filtering import apply_butterworth
from src.core.data_utils import extract_features
from collections import Counter


def load_raw_section_data(section_path, config):
    """
    Loads IMU data and metadata from either legacy directory format or new Parquet/Meta.json format.

    Legacy: section_path is a directory containing imu.csv and data.json
    New: section_path is a file prefix (e.g. 'data/processed/uuid') where .parquet and .meta.json exist
    """
    df = None
    metadata = None

    # 1. Try New Format (.parquet + .meta.json)
    parquet_path = section_path + ".parquet"
    # Support both .meta.json, .json and .aligned.json for metadata
    meta_json_path = section_path + ".meta.json"
    if not os.path.exists(meta_json_path):
        meta_json_path = section_path + ".json"
    if not os.path.exists(meta_json_path):
        meta_json_path = section_path + ".aligned.json"

    if os.path.exists(parquet_path) and os.path.exists(meta_json_path):
        df = pd.read_parquet(parquet_path)
        with open(meta_json_path, 'r', encoding="utf-8") as f:
            metadata = json.load(f)

        # Map Parquet columns to canonical names used in the pipeline
        rename_map = {
            't_sec': 'rel_time',
            'ax': 'acc_x', 'ay': 'acc_y', 'az': 'acc_z',
            'gx': 'gyro_x', 'gy': 'gyro_y', 'gz': 'gyro_z'
        }
        df = df.rename(columns=rename_map)

    # 2. Try Legacy Format (Directory with imu.csv + data.json)
    elif os.path.isdir(section_path):
        csv_path = os.path.join(section_path, "imu.csv")
        json_path = os.path.join(section_path, "data.json")

        if os.path.exists(csv_path) and os.path.exists(json_path):
            df = pd.read_csv(csv_path)
            with open(json_path, 'r', encoding="utf-8") as f:
                metadata = json.load(f)

    if df is None or metadata is None:
        return None, None

    # Apply Filtering
    sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in sensors:
        df[f'{col}_filt'] = apply_butterworth(df[col], config['lowpass_cutoff'], config['sample_rate'])

    # Assign Labels (Handle multiple metadata schemas)
    df['label'] = 'REST'

    # Schema A: New 'segments' list
    if "segments" in metadata:
        for seg in metadata["segments"]:
            mask = (df['rel_time'] >= seg['start']) & (df['rel_time'] <= seg['end'])
            df.loc[mask, 'label'] = canonicalize_label(seg['name'])

    # Schema B: Legacy 'roundResults' (direct or nested in 'workout')
    round_results = []
    if "roundResults" in metadata:
        round_results = metadata["roundResults"]
    elif "workout" in metadata and isinstance(metadata["workout"], dict):
        round_results = metadata["workout"].get("roundResults", [])

    for round_data in round_results:
        for ex in round_data.get("exerciseResults", []):
            start = ex.get('startTime') or ex.get('start')
            end = ex.get('endTime') or ex.get('end')
            if start is not None and end is not None:
                mask = (df['rel_time'] >= start) & (df['rel_time'] <= end)
                df.loc[mask, 'label'] = canonicalize_label(ex['name'])

    return df, metadata


class WodDataset(Dataset):
    def __init__(self, data_dir, config):
        self.cfg = config
        self.X = []
        self.y = []

        self._load_and_window_data(data_dir)

        # Encode labels to integers
        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(self.y)

    def _load_and_window_data(self, data_dir):
        print(f"Loading training data from {data_dir}...")

        # 1. Collect Legacy Section Directories
        legacy_dirs = glob.glob(os.path.join(data_dir, "**", "section_*"), recursive=True)

        # 2. Collect New Processed Parquet Files
        parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
        new_prefixes = [p.replace(".parquet", "") for p in parquet_files]

        all_sources = legacy_dirs + new_prefixes
        print(f"Found {len(legacy_dirs)} legacy sections and {len(new_prefixes)} new parquet sessions.")

        window_pts = int(self.cfg['window_size_sec'] * self.cfg['sample_rate'])
        step_pts = int(self.cfg['step_size_sec'] * self.cfg['sample_rate'])

        sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        filt_cols = [f'{s}_filt' for s in sensors]

        for source in all_sources:
            df, _ = load_raw_section_data(source, self.cfg)
            if df is None:
                continue

            signals = df[filt_cols].values
            labels = df['label'].values

            for i in range(0, len(df) - window_pts, step_pts):
                window_data = signals[i: i + window_pts]
                mid_idx = i + window_pts // 2

                self.X.append(window_data)
                self.y.append(labels[mid_idx])

        self.X = np.array(self.X)
        print(f"Created {len(self.X)} windows across {len(set(self.y))} classes.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Shape: [Channels (6), Sequence Length (window_pts)]
        x_tensor = torch.FloatTensor(self.X[idx]).transpose(0, 1)
        y_tensor = torch.tensor(self.y_encoded[idx], dtype=torch.long)
        return x_tensor, y_tensor


def load_and_window_data_for_dt(data_dir, config):
    print(f"Loading training data from {data_dir}...")

    # 1. Collect Legacy Section Directories
    legacy_dirs = glob.glob(os.path.join(data_dir, "**", "section_*"), recursive=True)

    # 2. Collect New Processed Parquet Files
    parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    new_prefixes = [p.replace(".parquet", "") for p in parquet_files]

    all_sources = legacy_dirs + new_prefixes
    print(f"Found {len(legacy_dirs)} legacy sections and {len(new_prefixes)} new parquet sessions.")

    window_pts = int(config['window_size_sec'] * config['sample_rate'])
    step_pts = int(config['step_size_sec'] * config['sample_rate'])

    X_list = []
    y_list = []
    class_counts = Counter()

    sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    filt_cols = [f'{s}_filt' for s in sensors]

    for source in all_sources:
        df, _ = load_raw_section_data(source, config)
        if df is None:
            continue

        signals = df[filt_cols].values
        labels = df['label'].values

        for i in range(0, len(df) - window_pts, step_pts):
            mid_idx = i + window_pts // 2
            label = labels[mid_idx]

            if config.get('allowed_classes') is not None and label not in config['allowed_classes']:
                continue

            window_data = signals[i: i + window_pts]
            stat_features = extract_features(window_data)
            X_list.append(stat_features)
            y_list.append(label)
            class_counts[label] += 1

    X_np = np.array(X_list)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_list)

    print(f"Created {len(X_np)} windows. Feature Vector Shape: {X_np.shape} across {len(set(y_encoded))} classes.")
    print(f"Class Distribution: {dict(class_counts)}")
    return X_np, y_encoded, le
