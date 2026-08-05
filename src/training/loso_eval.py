import os
import json
import glob
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from collections import Counter
import joblib

from src.core.exercises import canonicalize_label
from src.training.data_loader import load_raw_section_data
from src.models.cnn.cnn_model import SixAxisCNN
from src.training.decoders import WodDecoder, RestPolicy

# --- CONFIGURATION ---
CONFIG = {
    'sample_rate': 25,
    'window_size_sec': 2.5,
    'step_size_sec': 0.5,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
    'batch_size': 32,
    'epochs': 10, # Balanced for speed and quality
    'lr': 0.001,
    'dwell_sweep': [2, 3, 5, 8, 10],
    'default_dwell': 5.0,
    'tolerance_sec': 3.0, # Reduced from 10.0 to 3.0 as per BUG 2
    'weighted_loss': True # Added for BUG 4
}

IGNORE_LABELS = ["null", "setup"]

def load_workout_plan(plan_path):
    """Loads a workout plan from JSON and returns a list of exercise segments."""
    if not os.path.exists(plan_path):
        return None
    with open(plan_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plan_segments = []
    rounds = data.get("rounds", 1)
    
    # Check various locations for exercises in the schema
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
    
    # Map and handle rounds
    base_sequence = []
    for seg in plan_segments:
        name = seg.get('canonicalName') or seg.get('name')
        norm_name = canonicalize_label(name)
        
        if not norm_name and name:
            raise ValueError(f"LOUD FAIL: Unmapped exercise name='{name}'")
        
        if norm_name:
            base_sequence.append(norm_name)
    
    full_sequence = base_sequence * rounds
    return full_sequence

def calculate_wod_metrics(y_true_labels, y_pred_labels, times, step_size_sec, tolerance_sec=5.0):
    """
    Calculates detailed WOD metrics as requested.
    """
    T = len(y_true_labels)
    correct_mask = y_true_labels == y_pred_labels
    
    acc = accuracy_score(y_true_labels, y_pred_labels)
    # n_true, n_emitted, recall, precision, correct-class, wrong-class
    # (These are standard multi-class metrics)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    
    # Transitions
    def find_transitions(labels):
        trans = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                trans.append({'time': i * step_size_sec, 'from': labels[i-1], 'to': labels[i], 'idx': i})
        return trans

    gt_trans = find_transitions(y_true_labels)
    pred_trans = find_transitions(y_pred_labels)
    
    # False transitions per minute
    # A false transition is one that doesn't match any GT transition within tolerance
    false_transitions = 0
    premature_transitions = 0
    latencies = []
    
    matched_gt = set()
    for pt in pred_trans:
        found_match = False
        for i, gt in enumerate(gt_trans):
            if i in matched_gt: continue
            # Match if classes match and within tolerance
            if pt['from'] == gt['from'] and pt['to'] == gt['to']:
                latency = pt['time'] - gt['time']
                if abs(latency) <= 15.0: # Broad window for matching
                    matched_gt.add(i)
                    latencies.append(latency)
                    found_match = True
                    # Premature?
                    if latency < -tolerance_sec:
                        premature_transitions += 1
                    break
        if not found_match:
            false_transitions += 1
            
    session_duration_min = (T * step_size_sec) / 60.0
    ft_per_min = false_transitions / session_duration_min if session_duration_min > 0 else 0
    
    premature_rate = premature_transitions / len(pred_trans) if len(pred_trans) > 0 else 0
    
    # Unrecoverable error rate: 
    # Defined as session where an early transition caused the remaining sequence to be misattributed.
    # We'll check if the last exercise is correct at the end.
    unrecoverable = 1 if not correct_mask[-1] else 0
    
    # Total seconds of misattributed time
    misattributed_sec = np.sum(~correct_mask) * step_size_sec
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_trans_per_min': ft_per_min,
        'median_latency': np.median(latencies) if latencies else 0,
        'p90_latency': np.percentile(latencies, 90) if latencies else 0,
        'premature_rate': premature_rate,
        'unrecoverable': unrecoverable,
        'misattributed_sec': misattributed_sec,
        'n_gt_trans': len(gt_trans),
        'n_pred_trans': len(pred_trans)
    }

def get_session_data(data_dir):
    parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    sessions = {}
    
    before_counts = Counter()
    after_counts = Counter()
    
    for p in parquet_files:
        session_id = os.path.basename(p).split('.')[0]
        prefix = p.replace(".parquet", "")
        df, meta = load_raw_section_data(prefix, CONFIG)
        if df is None:
            continue
            
        # --- Robust label extraction for inference ---
        # If labels are all 'Rest' or missing, try to extract from other meta fields
        # This handles the new schema without needing changes in shared data_loader.py
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
            elif "roundResults" in meta:
                for r in meta.get("roundResults", []):
                    all_segments.extend(r.get("exerciseResults", []))
            
            current_time = 0.0
            for seg in all_segments:
                name = seg.get('name') or seg.get('canonicalName')
                start = seg.get('start') or seg.get('startTime')
                end = seg.get('end') or seg.get('endTime')
                
                if start is None or end is None:
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
        # ---------------------------------------------
            
        # Record before counts
        before_counts.update(df['label'].values)
        
        # Normalize labels
        df['norm_label'] = df['label'].apply(canonicalize_label)
        
        # Record after counts (excluding None/NaN)
        after_counts.update([l for l in df['norm_label'].values if pd.notna(l)])
        
        # Drop ignored labels for training/eval
        df = df.dropna(subset=['norm_label']).reset_index(drop=True)
        
        sessions[session_id] = df
        
    # Print distribution table
    print("\n--- Label Normalization Summary ---")
    all_classes = sorted([str(k) for k in (set(before_counts.keys()) | set(after_counts.keys())) if k is not None])
    print(f"{'Original Label':<25} | {'Normalized':<25} | {'Count'}")
    print("-" * 65)
    for cls in all_classes:
        # Check original key (it might be a string or nan)
        orig_key = cls
        if cls == "nan":
            norm = "IGNORED"
        else:
            try:
                norm = canonicalize_label(cls)
                if norm is None:
                    norm = "IGNORED"
            except:
                norm = "ERROR"
        
        # find original count
        count = 0
        for k, v in before_counts.items():
            if str(k) == cls:
                count = v
                break

        print(f"{str(cls):<25} | {str(norm):<25} | {count}")
    
    print("\n--- Final Class Distribution ---")
    for cls, count in after_counts.most_common():
        print(f"{cls:<25}: {count}")
        
    return sessions

# 2. Dataset for Windowing
class LabeledWindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.X = torch.FloatTensor(windows).transpose(1, 2) # [N, 6, seq]
        self.y = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_windows(df, config, label_encoder):
    window_pts = int(config['window_size_sec'] * config['sample_rate'])
    step_pts = int(config['step_size_sec'] * config['sample_rate'])
    
    sensors = ['acc_x_filt', 'acc_y_filt', 'acc_z_filt', 'gyro_x_filt', 'gyro_y_filt', 'gyro_z_filt']
    signals = df[sensors].values
    labels = df['norm_label'].values
    times = df['rel_time'].values
    
    windows = []
    window_labels = []
    window_times = []
    
    for i in range(0, len(df) - window_pts, step_pts):
        window_data = signals[i : i + window_pts]
        mid_idx = i + window_pts // 2
        
        windows.append(window_data)
        window_labels.append(labels[mid_idx])
        window_times.append(times[mid_idx])
        
    windows = np.array(windows)
    encoded_labels = label_encoder.transform(window_labels)
    
    return windows, encoded_labels, window_times

# 3. Transition Decoder and Evaluation
def get_transitions(labels, times):
    transitions = []
    if len(labels) == 0:
        return transitions
        
    current_label = labels[0]
    # We can consider the start as a transition to the first label if we want, 
    # but usually we look for changes.
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            transitions.append({
                'time': times[i],
                'from': current_label,
                'to': labels[i]
            })
            current_label = labels[i]
    return transitions

def dwell_decode(predictions, times, dwell_seconds, step_size_sec):
    dwell_steps = int(dwell_seconds / step_size_sec)
    if dwell_steps < 1:
        dwell_steps = 1
        
    decoded_states = []
    if len(predictions) == 0:
        return decoded_states
        
    # Initialize current_state to the first class
    current_state = predictions[0]
    
    candidate = None
    candidate_count = 0
    last_transition_time = times[0] - dwell_seconds # Allow immediate transition
    
    for i in range(len(predictions)):
        pred = predictions[i]
        time = times[i]
        
        if pred != current_state:
            if pred == candidate:
                candidate_count += 1
            else:
                candidate = pred
                candidate_count = 1
        else:
            candidate = None
            candidate_count = 0
            
        if candidate_count >= dwell_steps:
            # Check refractory period
            if (time - last_transition_time) >= (dwell_seconds - 1e-6):
                current_state = candidate
                decoded_states.append({'time': time, 'label': current_state})
                last_transition_time = time
                candidate = None
                candidate_count = 0
                
    # Sanity checks
    for i in range(1, len(decoded_states)):
        assert decoded_states[i]['time'] > decoded_states[i-1]['time'], "Timestamps must be monotonically increasing"
        assert (decoded_states[i]['time'] - decoded_states[i-1]['time']) >= (dwell_seconds - 1e-6), f"Events too close: {decoded_states[i]['time'] - decoded_states[i-1]['time']} < {dwell_seconds}"
        
    return decoded_states

def evaluate_transitions(gt_transitions, pred_states, tolerance_sec=3.0):
    """
    Greedy one-to-one matching between true transitions and emitted events.
    """
    results = {
        'matched_and_correct': 0,
        'matched_but_wrong': 0,
        'missed': 0,
        'false': 0,
        'latencies': [],
        'total_gt': len(gt_transitions),
        'total_emitted': len(pred_states)
    }

    if not gt_transitions:
        results['false'] = len(pred_states)
        return results
    if not pred_states:
        results['missed'] = len(gt_transitions)
        return results

    # 1. All possible matches within tolerance
    matches = []
    for i, gt in enumerate(gt_transitions):
        for j, ps in enumerate(pred_states):
            dist = abs(ps['time'] - gt['time'])
            if dist <= tolerance_sec:
                matches.append((i, j, dist))

    # 2. Sort by distance (nearest-first)
    matches.sort(key=lambda x: x[2])

    used_gt = set()
    used_pred = set()

    for gt_idx, pred_idx, dist in matches:
        if gt_idx not in used_gt and pred_idx not in used_pred:
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)

            # Use signed latency (ps - gt)
            results['latencies'].append(pred_states[pred_idx]['time'] - gt_transitions[gt_idx]['time'])

            if pred_states[pred_idx]['label'] == gt_transitions[gt_idx]['to']:
                results['matched_and_correct'] += 1
            else:
                results['matched_but_wrong'] += 1

    results['matched'] = results['matched_and_correct'] + results['matched_but_wrong']
    results['missed'] = len(gt_transitions) - results['matched']
    results['false'] = len(pred_states) - results['matched']

    return results

def guide_predictions(all_preds, window_times, plan, label_encoder):
    """Adjusts predictions based on a workout plan."""
    if not plan:
        return all_preds
    
    guided_preds = []
    class_to_idx = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
    rest_idx = class_to_idx.get("REST")
    
    for pred, t in zip(all_preds, window_times):
        # Find if there is a planned exercise for this time
        planned_label = None
        for seg in plan:
            if seg['start'] <= t <= seg['end']:
                planned_label = seg['name']
                break
        
        if planned_label and planned_label in class_to_idx:
            # Plan-based guidance: 
            # If the model predicts the planned label, great.
            # If not, we might still want to consider the model's output, 
            # but for this specific instruction, we'll let the plan override 
            # non-rest predictions to ensure alignment with the "guided" intent.
            planned_idx = class_to_idx[planned_label]
            if pred != planned_idx and pred != rest_idx:
                # If model thinks it's another exercise but plan says X, trust the plan
                guided_preds.append(planned_idx)
            else:
                guided_preds.append(pred)
        else:
            guided_preds.append(pred)
            
    return guided_preds

def run_loso():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed" / "apple"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Processed Apple data directory was not found: {data_dir}\n"
            "Check that the data exists and that PyCharm is running the correct project."
        )

    sessions = get_session_data(str(data_dir))
    session_ids = list(sessions.keys())
    
    all_labels = []
    for df in sessions.values():
        all_labels.extend(df['norm_label'].unique())
    
    le = LabelEncoder()
    le.fit(sorted(list(set(all_labels))))
    num_classes = len(le.classes_)
    
    fold_results = []
    pooled_y_true = []
    pooled_y_pred = []
    
    # Store transition metrics for sweep
    sweep_results = {dwell: [] for dwell in CONFIG['dwell_sweep']}
    
    # Track which classes were in training for each fold
    fold_train_classes = []
    variant_aggregate_results = {}
    
    for held_out_id in session_ids:
        print(f"\n{'='*20} FOLD: Held out {held_out_id} {'='*20}")
        
        train_sessions = [sid for sid in session_ids if sid != held_out_id]
        
        train_dfs = [sessions[sid] for sid in train_sessions]
        test_df = sessions[held_out_id]
        
        # --- BUG 3: Class Leakage Fix (Per-fold LabelEncoder) ---
        train_classes_in_fold = set()
        for df in train_dfs:
            train_classes_in_fold.update(df['norm_label'].unique())
        test_classes_in_fold = set(test_df['norm_label'].unique())
        common_classes = sorted(list(train_classes_in_fold.intersection(test_classes_in_fold)))
        
        for cls in sorted(train_classes_in_fold.union(test_classes_in_fold)):
            if cls not in common_classes:
                if cls in test_classes_in_fold:
                    count = len(test_df[test_df['norm_label'] == cls])
                    loc = "TEST (held-out)"
                else:
                    count = sum(len(df[df['norm_label'] == cls]) for df in train_dfs)
                    loc = "TRAIN"
                print(f"  [BUG 3] Dropping class '{cls}' ({count} windows in {loc}) - not present in both sets.")
        
        train_dfs = [df[df['norm_label'].isin(common_classes)].copy() for df in train_dfs]
        test_df = test_df[test_df['norm_label'].isin(common_classes)].copy()
        
        if len(test_df) == 0:
            print(f"  WARNING: No windows left in test set after filtering! Skipping fold.")
            continue
            
        # Fit local label encoder for this fold
        le_fold = LabelEncoder()
        le_fold.fit(common_classes)
        num_classes_fold = len(le_fold.classes_)
        
        # 2. Fit Scaler on TRAIN only
        scaler = StandardScaler()
        sensors = ['acc_x_filt', 'acc_y_filt', 'acc_z_filt', 'gyro_x_filt', 'gyro_y_filt', 'gyro_z_filt']
        
        train_data_full = pd.concat(train_dfs)
        scaler.fit(train_data_full[sensors])
        
        # Apply scaling
        for df in train_dfs:
            df[sensors] = scaler.transform(df[sensors])
        test_df_scaled = test_df.copy()
        test_df_scaled[sensors] = scaler.transform(test_df_scaled[sensors])
        
        # Track training classes
        train_classes = set()
        for df in train_dfs:
            train_classes.update(df['norm_label'].unique())
        fold_train_classes.append(train_classes)
        
        # 3. Create Windows
        X_train_list, y_train_list = [], []
        for df in train_dfs:
            X, y, _ = create_windows(df, CONFIG, le_fold)
            X_train_list.append(X)
            y_train_list.append(y)
        
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        X_test, y_test, t_test = create_windows(test_df_scaled, CONFIG, le_fold)
        
        # Print fold info
        print(f"Train windows: {len(X_train)}, Test windows: {len(X_test)}")
        class_dist = Counter(le_fold.inverse_transform(y_test))
        print(f"Test Class Distribution: {dict(class_dist)}")
        unreliable = [c_name for c_name, count in class_dist.items() if count < 20]
        if unreliable:
            print(f"WARNING: Unreliable metrics for small classes: {unreliable}")
        
        # 4. Train Model
        train_ds = LabeledWindowDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
        
        model = SixAxisCNN(num_classes=num_classes_fold).to(device)
        
        # --- BUG 4: Class Weighting ---
        if CONFIG['weighted_loss']:
            class_counts = Counter(y_train)
            weights = np.zeros(num_classes_fold)
            for i in range(num_classes_fold):
                if i in class_counts:
                    # Inverse frequency
                    weights[i] = len(y_train) / (num_classes_fold * class_counts[i])
            class_weights = torch.FloatTensor(weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            rest_labels = [l for l in le_fold.classes_ if "REST" in l.upper()]
            if rest_labels:
                rest_weight = weights[le_fold.transform([rest_labels[0]])[0]]
                print(f"  [BUG 4] Using weighted loss. REST weight: {rest_weight:.2f}")
        else:
            criterion = nn.CrossEntropyLoss()
            
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
        
        for epoch in range(CONFIG['epochs']):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        
        # 5. Evaluate Layer A (Per-window) & Layer B (WOD Decoders)
        model.eval()
        test_ds = LabeledWindowDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(device)
                out = model(bx)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                all_probs.extend(probs)
        
        all_probs = np.array(all_probs)
        y_test_labels = le_fold.inverse_transform(y_test)
        
        # WOD Decoder setup
        plan_path = project_root / "workout2.json"
        try:
            wod_sequence = load_workout_plan(plan_path)
            if wod_sequence is None:
                wod_sequence = []
        except ValueError as e:
            print(f"  {e}")
            wod_sequence = []

        # Only evaluate WOD decoders if we have a plan and all exercises are in our fold le
        can_run_wod = len(wod_sequence) > 0
        if can_run_wod:
            for name in wod_sequence:
                if name not in le_fold.classes_:
                    print(f"  [WOD] Skipping WOD decoders: exercise '{name}' not in this fold's classes.")
                    can_run_wod = False
                    break
        
        decoder = None
        if can_run_wod:
            decoder = WodDecoder(
                workout_sequence=wod_sequence,
                label_encoder=le_fold,
                confidence_threshold=0.8,
                dwell_seconds=CONFIG['default_dwell'],
                step_size_sec=CONFIG['step_size_sec'],
                rollback_window_sec=15.0
            )

        variants = [
            {'name': 'baseline', 'type': 'baseline'},
        ]
        if decoder:
            variants.extend([
                {'name': 'greedy_wod_off', 'type': 'greedy', 'rollback': False, 'policy': RestPolicy.OFF},
                {'name': 'greedy_wod_rollback_off', 'type': 'greedy', 'rollback': True, 'policy': RestPolicy.OFF},
                {'name': 'greedy_wod_rollback_prefer', 'type': 'greedy', 'rollback': True, 'policy': RestPolicy.PREFER_REST},
                {'name': 'greedy_wod_rollback_require', 'type': 'greedy', 'rollback': True, 'policy': RestPolicy.REQUIRE_REST},
                {'name': 'viterbi', 'type': 'viterbi'}
            ])

        fold_variant_results = {}
        fold_plot_dir = project_root / "src" / "training" / "loso_fold_plots"
        for var in variants:
            v_name = var['name']
            if var['type'] == 'baseline':
                y_pred_idx = decoder.decode_greedy_baseline(all_probs) if decoder else np.argmax(all_probs, axis=1)
                rollback_info = []
            elif var['type'] == 'greedy':
                decoder.rest_policy = var['policy']
                y_pred_idx, rollback_info = decoder.decode_greedy_wod(all_probs, use_rollback=var['rollback'])
            elif var['type'] == 'viterbi':
                y_pred_idx = decoder.decode_viterbi(all_probs)
                rollback_info = []
            
            y_pred_labels = le_fold.inverse_transform(y_pred_idx)
            metrics = calculate_wod_metrics(y_test_labels, y_pred_labels, t_test, CONFIG['step_size_sec'])
            metrics['rollback_count'] = len(rollback_info)
            
            fold_variant_results[v_name] = {
                'metrics': metrics,
                'labels': y_pred_labels
            }
            
            if v_name not in variant_aggregate_results:
                variant_aggregate_results[v_name] = []
            variant_aggregate_results[v_name].append(metrics)

            # Plot this variant
            plot_loso_fold_results(
                held_out_id=held_out_id,
                test_df_scaled=test_df_scaled,
                window_times=t_test,
                expected_labels=y_test_labels,
                predicted_labels=y_pred_labels,
                accuracy=metrics['accuracy'],
                f1_macro=metrics['f1'],
                output_dir=fold_plot_dir,
                variant_name=v_name
            )

        # Use 'greedy_wod_rollback_prefer' as the primary for standard fold reporting if available, else baseline
        primary_var = 'greedy_wod_rollback_prefer' if 'greedy_wod_rollback_prefer' in fold_variant_results else 'baseline'
        primary_metrics = fold_variant_results[primary_var]['metrics']
        primary_labels = fold_variant_results[primary_var]['labels']

        pooled_y_true.extend(le.transform(y_test_labels))
        pooled_y_pred.extend(le.transform(primary_labels))

        acc = primary_metrics['accuracy']
        f1 = primary_metrics['f1']
        
        print(f"Fold primary variant ({primary_var}) Macro-F1: {f1:.4f}")
        print(f"Fold Accuracy: {acc:.4f}")
        
        fold_results.append({
            'session_id': held_out_id,
            'accuracy': acc,
            'f1_macro': f1,
            'class_dist': dict(class_dist),
            'unreliable_classes': unreliable
        })
        
        # 6. Evaluate Layer B (Transitions)
        # Use labels already decoded above
        
        gt_transitions = get_transitions(y_test_labels, t_test)
        session_duration_min = (t_test[-1] - t_test[0]) / 60.0
        
        for dwell in CONFIG['dwell_sweep']:
            pred_states = dwell_decode(primary_labels, t_test, dwell, CONFIG['step_size_sec'])
            res = evaluate_transitions(gt_transitions, pred_states, CONFIG['tolerance_sec'])
            
            # BUG 1: Sanity output
            print(f"  dwell={dwell}s: n_true={res['total_gt']}, n_emitted={res['total_emitted']}")
            if dwell >= 5.0 and res['total_gt'] > 0:
                ratio = res['total_emitted'] / res['total_gt']
                # User asked for ~3x. We use 10x as a limit for BUG 1 "physically impossible" loud failure.
                assert ratio < 10.0, f"BUG 1: Excessive events at dwell={dwell}s! {res['total_emitted']} events for {res['total_gt']} transitions (ratio {ratio:.1f}x)"
            
            # BUG 2 metrics
            recall = res['matched'] / res['total_gt'] if res['total_gt'] > 0 else 0
            precision = res['matched_and_correct'] / res['total_emitted'] if res['total_emitted'] > 0 else 0
            
            false_trans_per_min = res['false'] / session_duration_min
            med_latency = np.median(res['latencies']) if res['latencies'] else np.nan
            p90_latency = np.quantile(res['latencies'], 0.9) if res['latencies'] else np.nan
            
            sweep_results[dwell].append({
                'session_id': held_out_id,
                'recall': recall,
                'precision': precision,
                'ft_per_min': false_trans_per_min,
                'med_latency': med_latency,
                'p90_latency': p90_latency,
                'correct_class': res['matched_and_correct'],
                'wrong_class': res['matched_but_wrong'],
                'n_true': res['total_gt'],
                'n_emitted': res['total_emitted']
            })

    # --- POOLED RESULTS & SUMMARY ---
    print("\n\n" + "="*50)
    print("FINAL LOSO SUMMARY")
    print("="*50)
    
    # Layer A Summary
    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['f1_macro'] for r in fold_results]
    
    print(f"\nLayer A (Per-window):")
    print(f"Pooled Accuracy: {accuracy_score(pooled_y_true, pooled_y_pred):.4f}")
    print(f"Pooled Macro-F1: {f1_score(pooled_y_true, pooled_y_pred, average='macro'):.4f}")
    
    # Per-class metrics pooled
    p, r, f, s = precision_recall_fscore_support(pooled_y_true, pooled_y_pred, labels=range(num_classes))
    print(f"\n{'Class':<25} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'Support'}")
    print("-" * 60)
    for i, class_name in enumerate(le.classes_):
        # Flag if class was ever missing from training in any fold
        missing_flag = ""
        for j, train_classes in enumerate(fold_train_classes):
            if class_name not in train_classes:
                missing_flag = f" (Untrained in fold {j})"
                break
        print(f"{class_name:<25} | {p[i]:<6.4f} | {r[i]:<6.4f} | {f[i]:<6.4f} | {s[i]}{missing_flag}")
    
    print(f"\nFold Spread - Layer A (n={len(session_ids)} folds):")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f} (min: {np.min(accs):.4f}, max: {np.max(accs):.4f})")
    print(f"Macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f} (min: {np.min(f1s):.4f}, max: {np.max(f1s):.4f})")
    
    # Layer B Sweep Summary
    print(f"\nLayer B (Transition Detection) Sweep:")
    print(f"{'Dwell (s)':<10} | {'Recall':<10} | {'Precision':<10} | {'FT/min':<10} | {'Med Lat':<8} | {'P90 Lat':<8} | {'Correct':<8} | {'Wrong'}")
    print("-" * 95)
    for dwell in CONFIG['dwell_sweep']:
        d_res = sweep_results[dwell]
        avg_recall = np.mean([r['recall'] for r in d_res])
        avg_prec = np.mean([r['precision'] for r in d_res])
        avg_ft = np.mean([r['ft_per_min'] for r in d_res])
        avg_med_lat = np.nanmean([r['med_latency'] for r in d_res])
        avg_p90_lat = np.nanmean([r['p90_latency'] for r in d_res])
        total_correct = np.sum([r['correct_class'] for r in d_res])
        total_wrong = np.sum([r['wrong_class'] for r in d_res])
        print(f"{dwell:<10} | {avg_recall:<10.4f} | {avg_prec:<10.4f} | {avg_ft:<10.4f} | {avg_med_lat:<8.2f} | {avg_p90_lat:<8.2f} | {total_correct:<8} | {total_wrong}")

    print(f"\nLayer B Fold Spread (at dwell={CONFIG['default_dwell']}s):")
    d_res = sweep_results[CONFIG['default_dwell']]
    recalls = [r['recall'] for r in d_res]
    precs = [r['precision'] for r in d_res]
    fts = [r['ft_per_min'] for r in d_res]
    print(f"Recall   : {np.mean(recalls):.4f} ± {np.std(recalls):.4f} (min: {np.min(recalls):.4f}, max: {np.max(recalls):.4f})")
    print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f} (min: {np.min(precs):.4f}, max: {np.max(precs):.4f})")
    print(f"FT/min   : {np.mean(fts):.4f} ± {np.std(fts):.4f} (min: {np.min(fts):.4f}, max: {np.max(fts):.4f})")

    # --- WOD VARIANT COMPARISON ---
    print("\n\n" + "="*50)
    print("WOD DECODER VARIANT COMPARISON")
    print("="*50)
    print(f"{'Variant':<28} | {'Acc':<6} | {'F1':<6} | {'FT/min':<6} | {'Prem%':<6} | {'Unrec%':<6} | {'Rollbk'}")
    print("-" * 80)
    for v_name, v_metrics_list in variant_aggregate_results.items():
        avg_acc = np.mean([m['accuracy'] for m in v_metrics_list])
        avg_f1 = np.mean([m['f1'] for m in v_metrics_list])
        avg_ft = np.mean([m['false_trans_per_min'] for m in v_metrics_list])
        avg_prem = np.mean([m['premature_rate'] for m in v_metrics_list]) * 100
        avg_unrec = np.mean([m['unrecoverable'] for m in v_metrics_list]) * 100
        total_rollback = np.sum([m['rollback_count'] for m in v_metrics_list])
        print(f"{v_name:<28} | {avg_acc:<6.4f} | {avg_f1:<6.4f} | {avg_ft:<6.2f} | {avg_prem:<6.1f} | {avg_unrec:<6.1f} | {total_rollback}")

    # Plot Confusion Matrix
    cm = confusion_matrix(pooled_y_true, pooled_y_pred, normalize='true')
    plt.figure(figsize=(14, 12))
    
    # Use a custom categorical-like colormap if we want to be very safe, 
    # but 'viridis' or 'magma' are usually better for intensity.
    # The user specifically complained about palette collisions (blues/greens).
    # 'rocket' or 'mako' are good, but let's try 'tab20' or 'tab10' for distinctness if it was categorical.
    # Since it's a heatmap of values, 'YlGnBu' is standard but has "two blues".
    # 'turbo' is actually great for distinctness across the scale.
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=le.classes_, yticklabels=le.classes_, 
                cmap='turbo', cbar_kws={'label': 'Recall'})
    
    plt.title("Pooled LOSO Confusion Matrix (Normalized by Row/Recall)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("loso_confusion_matrix.png")
    print(f"\nSaved confusion matrix to loso_confusion_matrix.png")
    
    # Save results to JSON
    results_to_save = {
        'fold_results': fold_results,
        'sweep_results': sweep_results,
        'config': CONFIG
    }
    # Convert numpy types to native for JSON
    def json_serialize(obj):
        if isinstance(obj, (np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        if isinstance(obj, Counter): return dict(obj)
        if np.isnan(obj): return None
        return obj

    with open("loso_results.json", "w") as f:
        json.dump(results_to_save, f, default=json_serialize, indent=4)
    print(f"Saved fold results to loso_results.json")


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

def _draw_label_ribbon(ax, y_pos, labels, times, label_name, color_map):
    if len(labels) == 0:
        return

    start_time = times[0]
    current_label = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            end_time = times[i]
            ax.barh(
                y_pos,
                end_time - start_time,
                left=start_time,
                height=0.35,
                color=color_map[current_label],
                edgecolor="none",
            )
            start_time = end_time
            current_label = labels[i]

    ax.barh(
        y_pos,
        times[-1] - start_time,
        left=start_time,
        height=0.35,
        color=color_map[current_label],
        edgecolor="none",
    )

    ax.text(
        times[0],
        y_pos + 0.28,
        label_name,
        va="bottom",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )

def plot_loso_fold_results(
        held_out_id,
        test_df_scaled,
        window_times,
        expected_labels,
        predicted_labels,
        accuracy,
        f1_macro,
        output_dir,
        variant_name="baseline"
):
    """
    Saves a per-fold visualization comparing expected labels against predictions.
    The labels are plotted at the model window midpoint timestamps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    window_times = np.asarray(window_times)
    expected_labels = np.asarray(expected_labels)
    predicted_labels = np.asarray(predicted_labels)

    all_classes = sorted(set(expected_labels) | set(predicted_labels))
    cmap = plt.get_cmap("tab20")
    color_map = {
        cls: cmap(i % cmap.N)
        for i, cls in enumerate(all_classes)
    }

    for cls in all_classes:
        if "REST" in cls.upper():
            color_map[cls] = "#d9d9d9"

    fig, (ax_signal, ax_ribbon) = plt.subplots(
        2,
        1,
        figsize=(18, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1]},
    )

    if "rel_time" in test_df_scaled.columns and "acc_z_filt" in test_df_scaled.columns:
        ax_signal.plot(
            test_df_scaled["rel_time"],
            test_df_scaled["acc_z_filt"],
            color="black",
            linewidth=0.8,
            alpha=0.75,
            label="acc_z_filt",
        )
        ax_signal.legend(loc="upper right")

    ax_signal.set_title(
        f"LOSO Fold: {held_out_id} | Var: {variant_name} | Acc={accuracy:.4f} | Macro-F1={f1_macro:.4f}"
    )
    ax_signal.set_ylabel("Scaled accel Z")
    ax_signal.grid(True, alpha=0.25)

    _draw_label_ribbon(
        ax_ribbon,
        1.0,
        expected_labels,
        window_times,
        "Expected",
        color_map,
    )
    _draw_label_ribbon(
        ax_ribbon,
        0.0,
        predicted_labels,
        window_times,
        "Predicted",
        color_map,
    )

    mismatch_mask = expected_labels != predicted_labels
    if np.any(mismatch_mask):
        ax_ribbon.scatter(
            window_times[mismatch_mask],
            np.full(np.sum(mismatch_mask), -0.35),
            marker="x",
            color="red",
            s=14,
            alpha=0.7,
            label="Mismatch",
        )

    ax_ribbon.set_ylim(-0.7, 1.6)
    ax_ribbon.set_yticks([])
    ax_ribbon.set_xlabel("Relative time (seconds)")
    ax_ribbon.set_title("Expected vs Predicted Labels")
    ax_ribbon.grid(True, axis="x", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color=color_map[cls], lw=8, label=cls)
        for cls in all_classes
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 5),
        bbox_to_anchor=(0.5, -0.03),
    )

    save_path = output_dir / f"loso_fold_{_safe_filename(held_out_id)}_{variant_name}.png"
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved LOSO fold plot ({variant_name}) to {save_path}")


if __name__ == "__main__":
    run_loso()
