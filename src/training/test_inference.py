import os
import json
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import joblib
from collections import Counter

from src.core.exercises import canonicalize_label
from src.training.data_loader import load_raw_section_data
from src.models.cnn.cnn_model import SixAxisCNN
from src.core.data_utils import extract_features
from src.training.decoders import WodDecoder, RestPolicy

# --- CONFIGURATIONS ---
CNN_CONFIG = {
    'sample_rate': 20,
    'window_size_sec': 2.5,
    'step_size_sec': 0.5,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001
}

DT_CONFIG = {
    'sample_rate': 20,
    'window_size_sec': 2.5,
    'step_size_sec': 0.5,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
    'max_depth': 15,
    'min_samples_split': 5
}

def load_workout_plan(plan_path):
    """Loads a workout plan from JSON and returns a list of exercise labels."""
    if not os.path.exists(plan_path):
        return None
    with open(plan_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plan_segments = []
    rounds = data.get("rounds", 1)
    
    if "exercises" in data:
        plan_segments = data["exercises"]
    elif "roundResults" in data:
        for r in data["roundResults"]:
            plan_segments.extend(r.get("exerciseResults", []))
    elif "workout" in data and isinstance(data["workout"], dict):
        w = data["workout"]
        rounds = w.get("rounds", rounds)
        if "exercises" in w:
            plan_segments = w["exercises"]
        elif "roundResults" in w:
            for r in w.get("roundResults", []):
                plan_segments.extend(r.get("exerciseResults", []))
                
    base_sequence = []
    for seg in plan_segments:
        name = seg.get('name')
        canon = seg.get('canonicalName')
        
        norm_name = None
        if canon:
             norm_name = canon
        elif name:
            norm_name = canonicalize_label(name)
        
        if not norm_name and (canon or name):
            raise ValueError(f"LOUD FAIL: Unmapped exercise name='{name}', canon='{canon}'")
        
        if norm_name:
            base_sequence.append(norm_name)
    
    return base_sequence * rounds

def guide_predictions(probs, le, workout_plan, decoder_type='viterbi'):
    """Adjusts predictions based on a workout plan using revisable decoders."""
    if not workout_plan:
        return np.argmax(probs, axis=1)
    
    decoder = WodDecoder(
        workout_sequence=workout_plan,
        label_encoder=le,
        confidence_threshold=0.8,
        dwell_seconds=5.0,
        step_size_sec=0.5
    )
    
    if decoder_type == 'viterbi':
        return decoder.decode_viterbi(probs)
    elif decoder_type == 'rollback':
        preds, _ = decoder.decode_greedy_wod(probs, use_rollback=True)
        return preds
    else:
        return decoder.decode_greedy_baseline(probs)

def get_session_data(data_dir, config):
    parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    sessions = {}
    
    for p in parquet_files:
        session_id = os.path.basename(p).split('.')[0]
        prefix = p.replace(".parquet", "")
        df, meta = load_raw_section_data(prefix, config)
        if df is None:
            continue
            
        # Robust label extraction
        if (df['label'] == 'Rest').all() or df['label'].isnull().all():
            all_segments = []
            if "segments" in meta:
                all_segments = meta["segments"]
            elif "exercises" in meta:
                all_segments = meta["exercises"]
            
            for seg in all_segments:
                name = seg.get('name')
                start = seg.get('start')
                end = seg.get('end')
                if name and start is not None and end is not None:
                    mask = (df['rel_time'] >= start) & (df['rel_time'] <= end)
                    df.loc[mask, 'label'] = name
        
        df['norm_label'] = df['label'].apply(canonicalize_label)
        df = df.dropna(subset=['norm_label']).reset_index(drop=True)
        sessions[session_id] = df
        
    return sessions

def create_windows_cnn(df, config):
    window_pts = int(config['window_size_sec'] * config['sample_rate'])
    step_pts = int(config['step_size_sec'] * config['sample_rate'])
    sensors = ['acc_x_filt', 'acc_y_filt', 'acc_z_filt', 'gyro_x_filt', 'gyro_y_filt', 'gyro_z_filt']
    signals = df[sensors].values
    labels = df['norm_label'].values
    times = df['rel_time'].values
    
    windows, window_labels, window_times = [], [], []
    for i in range(0, len(df) - window_pts, step_pts):
        windows.append(signals[i : i + window_pts])
        mid_idx = i + window_pts // 2
        window_labels.append(labels[mid_idx])
        window_times.append(times[mid_idx])
    return np.array(windows), np.array(window_labels), np.array(window_times)

def create_windows_dt(df, config):
    window_pts = int(config['window_size_sec'] * config['sample_rate'])
    step_pts = int(config['step_size_sec'] * config['sample_rate'])
    sensors = ['acc_x_filt', 'acc_y_filt', 'acc_z_filt', 'gyro_x_filt', 'gyro_y_filt', 'gyro_z_filt']
    signals = df[sensors].values
    labels = df['norm_label'].values
    times = df['rel_time'].values
    
    X_features, window_labels, window_times = [], [], []
    for i in range(0, len(df) - window_pts, step_pts):
        window_data = signals[i : i + window_pts]
        mid_idx = i + window_pts // 2
        X_features.append(extract_features(window_data))
        window_labels.append(labels[mid_idx])
        window_times.append(times[mid_idx])
    return np.array(X_features), np.array(window_labels), np.array(window_times)

class LabeledWindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.X = torch.FloatTensor(windows).transpose(1, 2)
        self.y = torch.LongTensor(labels)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def plot_results(session_id, df, window_times, expected_labels, predicted_labels, metrics, output_dir, model_name, decoder_name="viterbi"):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_classes = sorted(set(expected_labels) | set(predicted_labels))
    cmap = plt.get_cmap("tab20")
    color_map = {cls: cmap(i % cmap.N) for i, cls in enumerate(all_classes)}
    for cls in all_classes:
        if "REST" in cls.upper(): color_map[cls] = "#d9d9d9"

    fig, (ax_signal, ax_ribbon) = plt.subplots(2, 1, figsize=(18, 7), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    if "acc_z_filt" in df.columns:
        ax_signal.plot(df["rel_time"], df["acc_z_filt"], color="black", linewidth=0.8, alpha=0.75)
    
    ax_signal.set_title(f"Model: {model_name} | Session: {session_id} | Decoder: {decoder_name} | Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f}")
    ax_signal.set_ylabel("Accel Z")
    ax_signal.grid(True, alpha=0.25)

    def draw_ribbon(ax, y_pos, labels, times, title):
        for i in range(len(labels)):
            start = times[i] - 0.25
            end = times[i] + 0.25
            ax.add_patch(plt.Rectangle((start, y_pos), end - start, 0.5, color=color_map[labels[i]], alpha=0.8))
        ax.text(times[0], y_pos + 0.55, title, va="bottom", ha="left", fontsize=10, fontweight="bold")

    draw_ribbon(ax_ribbon, 1.0, expected_labels, window_times, "Expected")
    draw_ribbon(ax_ribbon, 0.0, predicted_labels, window_times, "Predicted")

    ax_ribbon.set_ylim(-0.2, 1.8)
    ax_ribbon.set_yticks([])
    ax_ribbon.set_xlabel("Time (s)")
    handles = [plt.Line2D([0], [0], color=color_map[cls], lw=8, label=cls) for cls in all_classes]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(output_dir / f"{model_name}_{decoder_name}_{session_id}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def run_test_inference(use_guided=True, mini_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Guided inference: {use_guided}")
    print(f"Mini-model training: {mini_model}")

    # Load Workout Plan
    workout_plan = load_workout_plan("workout.json")
    if workout_plan:
        print(f"Loaded workout plan with {len(workout_plan)} segments.")
    elif use_guided or mini_model:
        print("Warning: workout.json not found or empty. Guidance and mini-model will have no effect.")

    # 1. Load Training Data
    print("Loading training data...")
    train_sessions = get_session_data("data/processed/apple", CNN_CONFIG)
    
    # 2. Prepare Data for training
    all_X_cnn, all_y_cnn = [], []
    all_X_dt, all_y_dt = [], []
    
    for df in train_sessions.values():
        X, y, _ = create_windows_cnn(df, CNN_CONFIG)
        all_X_cnn.append(X)
        all_y_cnn.append(y)
        
        X_dt, y_dt, _ = create_windows_dt(df, DT_CONFIG)
        all_X_dt.append(X_dt)
        all_y_dt.append(y_dt)

    X_train_cnn = np.concatenate(all_X_cnn)
    y_train_cnn_str = np.concatenate(all_y_cnn)
    X_train_dt = np.concatenate(all_X_dt)
    y_train_dt_str = np.concatenate(all_y_dt)

    # Filter training data if mini_model is True
    if mini_model and workout_plan:
        allowed_classes = set(workout_plan)
        allowed_classes.add("REST")
        print(f"Mini-model training enabled. Allowed classes: {allowed_classes}")
        
        mask_cnn = np.array([label in allowed_classes for label in y_train_cnn_str])
        X_train_cnn = X_train_cnn[mask_cnn]
        y_train_cnn_str = y_train_cnn_str[mask_cnn]
        
        mask_dt = np.array([label in allowed_classes for label in y_train_dt_str])
        X_train_dt = X_train_dt[mask_dt]
        y_train_dt_str = y_train_dt_str[mask_dt]
        
        if len(y_train_cnn_str) == 0:
            print("Warning: No training data left after mini-model filtering! Reverting to all data.")
            # Re-concatenate to restore (or just don't filter)
            X_train_cnn = np.concatenate(all_X_cnn)
            y_train_cnn_str = np.concatenate(all_y_cnn)
            X_train_dt = np.concatenate(all_X_dt)
            y_train_dt_str = np.concatenate(all_y_dt)

    le = LabelEncoder()
    y_train_cnn = le.fit_transform(y_train_cnn_str)
    y_train_dt = le.transform(y_train_dt_str) # Use same encoder
    
    scaler = StandardScaler()
    X_train_dt_scaled = scaler.fit_transform(X_train_dt)

    # 3. Train Models
    print("Training CNN...")
    num_classes = len(le.classes_)
    cnn_model = SixAxisCNN(num_classes=num_classes).to(device)
    dataset = LabeledWindowDataset(X_train_cnn, y_train_cnn)
    loader = DataLoader(dataset, batch_size=CNN_CONFIG['batch_size'], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=CNN_CONFIG['lr'])
    
    for epoch in range(CNN_CONFIG['epochs']):
        cnn_model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(cnn_model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0: print(f"CNN Epoch {epoch+1} complete")

    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=DT_CONFIG['max_depth'], min_samples_split=DT_CONFIG['min_samples_split'])
    dt_model.fit(X_train_dt_scaled, y_train_dt)

    # 4. Load Test Data
    print("Loading test data...")
    test_sessions = get_session_data("data/data/test", CNN_CONFIG)
    
    output_dir = Path("src/training/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []

    for sid, df in test_sessions.items():
        print(f"Evaluating session {sid}...")
        
        # 1. CNN Inference
        X_test_cnn, y_test_str, times = create_windows_cnn(df, CNN_CONFIG)
        cnn_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test_cnn).transpose(1, 2).to(device)
            logits = cnn_model(X_tensor)
            probs_cnn = torch.softmax(logits, dim=1).cpu().numpy()

        # 2. DT Inference
        X_test_dt, _, _ = create_windows_dt(df, DT_CONFIG)
        X_test_dt_scaled = scaler.transform(X_test_dt)
        probs_dt = dt_model.predict_proba(X_test_dt_scaled)

        # Iterate over decoder variants
        variants = ["baseline", "rollback", "viterbi"] if use_guided and workout_plan else ["unconstrained"]
        
        session_results = {'session_id': sid}

        for v in variants:
            print(f"  Decoder: {v}")
            # CNN
            if v == "unconstrained":
                preds_cnn = np.argmax(probs_cnn, axis=1)
            else:
                preds_cnn = guide_predictions(probs_cnn, le, workout_plan, decoder_type=v)
            
            labels_cnn = le.inverse_transform(preds_cnn)
            acc_cnn = accuracy_score(y_test_str, labels_cnn)
            f1_cnn = f1_score(y_test_str, labels_cnn, average='macro')
            plot_results(sid, df, times, y_test_str, labels_cnn, {'acc': acc_cnn, 'f1': f1_cnn}, output_dir, "CNN", v)
            
            session_results[f'cnn_{v}_acc'] = acc_cnn
            session_results[f'cnn_{v}_f1'] = f1_cnn

            # DT
            if v == "unconstrained":
                preds_dt = np.argmax(probs_dt, axis=1)
            else:
                preds_dt = guide_predictions(probs_dt, le, workout_plan, decoder_type=v)
                
            labels_dt = le.inverse_transform(preds_dt)
            acc_dt = accuracy_score(y_test_str, labels_dt)
            f1_dt = f1_score(y_test_str, labels_dt, average='macro')
            plot_results(sid, df, times, y_test_str, labels_dt, {'acc': acc_dt, 'f1': f1_dt}, output_dir, "DT", v)
            
            session_results[f'dt_{v}_acc'] = acc_dt
            session_results[f'dt_{v}_f1'] = f1_dt

        results.append(session_results)

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test inference with optional workout plan guidance.")
    parser.add_argument("--guided", type=str, default="true", help="Use workout.json to constrain predictions (true/false, default: true)")
    parser.add_argument("--mini", type=str, default="true", help="Train models only with workout classes + REST (true/false, default: true)")
    parser.add_argument("--decoder", type=str, default="viterbi", choices=["baseline", "rollback", "viterbi"], help="Decoder type (default: viterbi)")
    args = parser.parse_args()
    
    use_guided = args.guided.lower() == "true"
    mini_model = args.mini.lower() == "true"
    run_test_inference(use_guided=use_guided, mini_model=mini_model)
