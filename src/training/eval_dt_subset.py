import os
import json
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import joblib
from collections import Counter

from src.core.exercises import canonicalize_label
from src.training.data_loader import load_raw_section_data
from src.core.data_utils import extract_features
from src.training.decoders import WodDecoder, RestPolicy

# --- CONFIGURATION (Matches train_dt_subset.py) ---
CONFIG = {
    'sample_rate': 25,
    'window_size_sec': 2.0,
    'step_size_sec': 0.4,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
}

TARGET_EXERCISES = ["AIR_SQUAT", "DOUBLE_UNDER", "PUSH_UP", "REST"]

def load_workout_plan(plan_path):
    if not os.path.exists(plan_path):
        return None
    with open(plan_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rounds = data.get('rounds', 1)
    exercises = data.get('exercises', [])
    base_sequence = []
    
    for ex in exercises:
        name = ex.get('name')
        canon = ex.get('canonicalName')
        
        norm_name = None
        if canon:
             norm_name = canon
        
        if not norm_name and name:
            norm_name = canonicalize_label(name)
        
        if norm_name:
            base_sequence.append(norm_name)
    
    return base_sequence * rounds

def get_session_data(data_dir, config):
    parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    sessions = {}
    
    for p in parquet_files:
        session_id = os.path.basename(p).split('.')[0]
        prefix = p.replace(".parquet", "")
        df, meta = load_raw_section_data(prefix, config)
        if df is None:
            continue
            
        # Handle sessions with metadata-only labels
        if (df['label'] == 'Rest').all() or df['label'].isnull().all():
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
                    df.loc[mask, 'label'] = name
            
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
    times = df['rel_time'].values
    
    X_features = []
    window_labels = []
    window_times = []
    
    for i in range(0, len(df) - window_pts, step_pts):
        window_data = signals[i : i + window_pts]
        mid_idx = i + window_pts // 2
        
        feat = extract_features(window_data)
        X_features.append(feat)
        window_labels.append(labels[mid_idx])
        window_times.append(times[mid_idx])
        
    return np.array(X_features), np.array(window_labels), np.array(window_times)

def plot_results(session_id, df, window_times, expected_labels, predicted_labels, metrics, output_dir, model_name, decoder_name="viterbi"):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_classes = sorted(set(expected_labels) | set(predicted_labels))
    cmap = plt.get_cmap("tab20")
    color_map = {cls: cmap(i % cmap.N) for i, cls in enumerate(all_classes)}
    for cls in all_classes:
        if "REST" in cls.upper(): color_map[cls] = "#d9d9d9"

    fig, (ax_signal, ax_ribbon) = plt.subplots(2, 1, figsize=(18, 7), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    
    # Use Accel Z if available, otherwise SMV or something
    if "acc_z_filt" in df.columns:
        ax_signal.plot(df["rel_time"], df["acc_z_filt"], color="black", linewidth=0.8, alpha=0.75)
    
    ax_signal.set_title(f"Model: {model_name} | Session: {session_id} | Decoder: {decoder_name} | Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f}")
    ax_signal.set_ylabel("Accel Z")
    ax_signal.grid(True, alpha=0.25)

    def draw_ribbon(ax, y_pos, labels, times, title):
        for i in range(len(labels)):
            start = times[i] - CONFIG['step_size_sec']/2
            end = times[i] + CONFIG['step_size_sec']/2
            ax.add_patch(plt.Rectangle((start, y_pos - 0.4), end - start, 0.8, color=color_map[labels[i]]))
        ax.text(-0.01, y_pos, title, transform=ax.get_yaxis_transform(), ha='right', va='center', fontweight='bold')

    draw_ribbon(ax_ribbon, 1, expected_labels, window_times, "Ground Truth")
    draw_ribbon(ax_ribbon, 0, predicted_labels, window_times, "Predicted")

    ax_ribbon.set_ylim(-0.2, 1.8)
    ax_ribbon.set_yticks([])
    ax_ribbon.set_xlabel("Time (s)")
    handles = [plt.Line2D([0], [0], color=color_map[cls], lw=8, label=cls) for cls in all_classes]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(output_dir / f"DT_subset_{decoder_name}_{session_id}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def run_evaluation():
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "models" / "decision_tree"
    data_dir = project_root / "data" / "processed" / "apple"
    output_dir = project_root / "src" / "training" / "test_results_dt_subset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    print("Loading model artifacts...")
    model = joblib.load(model_dir / "dt_model_subset.joblib")
    scaler = joblib.load(model_dir / "scaler_subset.joblib")
    le = joblib.load(model_dir / "label_encoder_subset.joblib")

    # Load workout plan
    workout_plan = load_workout_plan("workout.json")
    if workout_plan:
        print(f"Loaded workout plan: {workout_plan}")
    else:
        print("No workout.json found, skipping guided decoders.")

    # Load sessions
    print(f"Loading sessions from {data_dir}...")
    sessions = get_session_data(str(data_dir), CONFIG)

    results = []

    for sid, df in sessions.items():
        print(f"Evaluating session {sid}...")
        
        # Windowing and feature extraction
        X, y_true_str, times = create_windows_dt(df, CONFIG)
        if len(X) == 0:
            continue
            
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)

        # Decoder variants
        variants = ["baseline"]
        if workout_plan:
            variants.extend(["viterbi"]) # User specifically asked for viterbi

        session_results = {'session_id': sid}

        for v in variants:
            print(f"  Decoder: {v}")
            
            decoder = WodDecoder(
                workout_sequence=workout_plan if workout_plan else [],
                label_encoder=le,
                confidence_threshold=0.8,
                dwell_seconds=5.0,
                step_size_sec=CONFIG['step_size_sec']
            )

            if v == "baseline":
                preds_idx = np.argmax(probs, axis=1)
            elif v == "viterbi":
                preds_idx = decoder.decode_viterbi(probs)
            
            labels_pred = le.inverse_transform(preds_idx)
            
            acc = accuracy_score(y_true_str, labels_pred)
            f1 = f1_score(y_true_str, labels_pred, average='macro')
            
            plot_results(sid, df, times, y_true_str, labels_pred, {'acc': acc, 'f1': f1}, output_dir, "DT_subset", v)
            
            session_results[f'{v}_acc'] = acc
            session_results[f'{v}_f1'] = f1
            print(f"    Acc: {acc:.4f}, F1: {f1:.4f}")

        results.append(session_results)

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation complete. Results and plots saved to {output_dir}")

if __name__ == "__main__":
    run_evaluation()
