import os
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from collections import Counter

from src.core.exercises import canonicalize_label
from src.training.data_loader import load_raw_section_data
from src.core.data_utils import extract_features

# --- CONFIGURATION ---
CONFIG = {
    'sample_rate': 25,
    'window_size_sec': 2.0,
    'step_size_sec': 0.4,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
    'max_depth': 15,
    'min_samples_split': 5,
}

TARGET_EXERCISES = ["AIR_SQUAT", "DOUBLE_UNDER", "PUSH_UP", "REST"]

def get_session_data(data_dir):
    parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    sessions = {}
    
    for p in parquet_files:
        session_id = os.path.basename(p).split('.')[0]
        prefix = p.replace(".parquet", "")
        df, meta = load_raw_section_data(prefix, CONFIG)
        if df is None:
            continue
            
        # Handle sessions with metadata-only labels
        if (df['label'] == 'Rest').all() or df['label'].isnull().all() or (df['label'] == 'REST').all():
            all_segments = []
            if "exercises" in meta:
                all_segments = meta["exercises"]
            elif "workout" in meta and isinstance(meta["workout"], dict):
                w = meta["workout"]
                if "exercises" in w:
                    all_segments = w["exercises"]
                elif "roundResults" in w:
                    for r in w.get("roundResults", []):
                        all_segments.extend(r.get("exerciseResults", []))
            
            current_time = 0.0
            for seg in all_segments:
                name = seg.get('name') or seg.get('canonicalName')
                duration = 0.0
                if "endConditionValues" in seg and seg["endConditionValues"]:
                    duration = float(seg["endConditionValues"][0])
                elif "duration" in seg:
                    duration = float(seg["duration"])
                
                start = current_time
                end = current_time + duration
                current_time = end
                
                if name:
                    mask = (df['rel_time'] >= start) & (df['rel_time'] <= end)
                    df.loc[mask, 'label'] = canonicalize_label(name)
            
        df['norm_label'] = df['label'].apply(canonicalize_label)
        df = df.dropna(subset=['norm_label']).reset_index(drop=True)
        if not df.empty:
            sessions[session_id] = df
        
    return sessions

def create_windows_dt(df, config):
    window_pts = int(config['window_size_sec'] * config['sample_rate'])
    step_pts = int(config['step_size_sec'] * config['sample_rate'])
    
    sensors = ['acc_x_filt', 'acc_y_filt', 'acc_z_filt', 'gyro_x_filt', 'gyro_y_filt', 'gyro_z_filt']
    signals = df[sensors].values
    labels = df['norm_label'].values
    
    X_features = []
    window_labels = []
    
    for i in range(0, len(df) - window_pts, step_pts):
        window_data = signals[i : i + window_pts]
        mid_idx = i + window_pts // 2
        
        feat = extract_features(window_data)
        X_features.append(feat)
        window_labels.append(labels[mid_idx])
        
    return np.array(X_features), np.array(window_labels)

def train_model():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed" / "apple"
    output_dir = project_root / "models" / "decision_tree"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}...")
    sessions = get_session_data(str(data_dir))
    
    if not sessions:
        print("No sessions found with target labels.")
        return

    all_X = []
    all_y = []
    
    print("Extracting features...")
    for session_id, df in sessions.items():
        X, y = create_windows_dt(df, CONFIG)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
    
    X_train = np.concatenate(all_X)
    y_train_raw = np.concatenate(all_y)
    
    # Label Encoding
    le = LabelEncoder()
    le.fit(TARGET_EXERCISES)
    y_train = le.transform(y_train_raw)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"Training on {len(X_train_scaled)} windows.")
    print(f"Class distribution: {dict(Counter(y_train_raw))}")
    
    # Train Decision Tree
    model = DecisionTreeClassifier(
        max_depth=CONFIG['max_depth'],
        min_samples_split=CONFIG['min_samples_split'],
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    # Save artifacts
    model_path = output_dir / "dt_model_subset.joblib"
    scaler_path = output_dir / "scaler_subset.joblib"
    le_path = output_dir / "label_encoder_subset.joblib"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)
    
    print(f"\nTraining complete.")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Label Encoder saved to: {le_path}")

if __name__ == "__main__":
    train_model()
