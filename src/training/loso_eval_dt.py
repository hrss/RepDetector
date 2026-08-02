import os
import json
import glob
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from collections import Counter
import joblib

from src.training.data_loader import load_raw_section_data
from src.core.data_utils import extract_features

# --- CONFIGURATION ---
# Decision Tree usually works better with the features at 20Hz as per decision_tree.py
CONFIG = {
    'sample_rate': 20,
    'window_size_sec': 2.5,
    'step_size_sec': 0.5,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
    'max_depth': 15,
    'min_samples_split': 5,
    'dwell_sweep': [2, 3, 5, 8, 10],
    'default_dwell': 5.0,
    'tolerance_sec': 3.0,
}

# 1. Label Normalization
class ExerciseLabel(Enum):
    AIR_SQUAT = "Air Squat"
    BURPEE = "Burpee"
    KB_SWING = "KB swing"
    WALL_BALL = "Wall ball"
    PUSH_UP = "Push-up"
    REST = "REST"
    SIT_UP = "Sit-up"
    WALKING_LUNGE = "Walking lunge"
    BOX_JUMP = "Box jump"
    DOUBLE_UNDER = "Double-under"
    RUN = "Run"
    RUN_ALL_OUT = "Run All Out"

LABEL_MAP = {
    "rest": ExerciseLabel.REST,
    "air squat": ExerciseLabel.AIR_SQUAT,
    "air_squat": ExerciseLabel.AIR_SQUAT,
    "burpee": ExerciseLabel.BURPEE,
    "kb swing": ExerciseLabel.KB_SWING,
    "kb_swing": ExerciseLabel.KB_SWING,
    "kb swing (russian)": ExerciseLabel.KB_SWING,
    "kb swing (american)": ExerciseLabel.KB_SWING,
    "wall ball": ExerciseLabel.WALL_BALL,
    "wall_ball_shot": ExerciseLabel.WALL_BALL,
    "push-up": ExerciseLabel.PUSH_UP,
    "chest_to_wall_hspu": ExerciseLabel.PUSH_UP,
    "sit-up": ExerciseLabel.SIT_UP,
    "sit_up": ExerciseLabel.SIT_UP,
    "walking lunge": ExerciseLabel.WALKING_LUNGE,
    "sandbag_lunges": ExerciseLabel.WALKING_LUNGE,
    "box jump": ExerciseLabel.BOX_JUMP,
    "box_jump": ExerciseLabel.BOX_JUMP,
    "double-under": ExerciseLabel.DOUBLE_UNDER,
    "double_under": ExerciseLabel.DOUBLE_UNDER,
    "run": ExerciseLabel.RUN,
    "run all out": ExerciseLabel.RUN_ALL_OUT,
}

IGNORE_LABELS = ["null", "setup"]

def normalize_label(label_str):
    if not label_str or pd.isna(label_str):
        return None
    ls = label_str.strip().lower()
    if ls in IGNORE_LABELS:
        return None
    
    # Direct match
    if ls in LABEL_MAP:
        return LABEL_MAP[ls].value
    
    # Substring matches for variants
    if "kb swing" in ls:
        return ExerciseLabel.KB_SWING.value
    if "rest" in ls:
        return ExerciseLabel.REST.value
    if "wall ball" in ls:
        return ExerciseLabel.WALL_BALL.value
    if "lunge" in ls:
        return ExerciseLabel.WALKING_LUNGE.value
    if "box jump" in ls:
        return ExerciseLabel.BOX_JUMP.value
    if "double under" in ls or "double-under" in ls:
        return ExerciseLabel.DOUBLE_UNDER.value
    if "push up" in ls or "push-up" in ls:
        return ExerciseLabel.PUSH_UP.value
    if "sit up" in ls or "sit-up" in ls:
        return ExerciseLabel.SIT_UP.value
    if "air squat" in ls:
        return ExerciseLabel.AIR_SQUAT.value
    if "burpee" in ls:
        return ExerciseLabel.BURPEE.value
    if "run all out" in ls:
        return ExerciseLabel.RUN_ALL_OUT.value
    if "run" in ls:
        return ExerciseLabel.RUN.value
        
    raise ValueError(f"Unmapped label: {label_str}")

def load_workout_plan(plan_path):
    """Loads a workout plan from JSON and returns a list of exercise segments."""
    if not os.path.exists(plan_path):
        return None
    with open(plan_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plan_segments = []
    # Check various locations for exercises in the schema
    if "exercises" in data:
        plan_segments = data["exercises"]
    elif "roundResults" in data:
        for r in data["roundResults"]:
            plan_segments.extend(r.get("exerciseResults", []))
    elif "workout" in data and isinstance(data["workout"], dict):
        w = data["workout"]
        if "exercises" in w:
            plan_segments = w["exercises"]
        elif "roundResults" in w:
            for r in w.get("roundResults", []):
                plan_segments.extend(r.get("exerciseResults", []))
                
    # Normalize segments to have name, start, end
    normalized_plan = []
    current_time = 0.0
    for seg in plan_segments:
        name = seg.get('name') or seg.get('canonicalName')
        
        # Try to get start/end directly
        start = seg.get('start') or seg.get('startTime')
        end = seg.get('end') or seg.get('endTime')
        
        # If not present, try to derive from duration
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
            norm_name = normalize_label(name)
            if norm_name: # might be None for 'Setup'
                normalized_plan.append({
                    'name': norm_name,
                    'start': start,
                    'end': end
                })
    return normalized_plan

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
            planned_idx = class_to_idx[planned_label]
            if pred != planned_idx and pred != rest_idx:
                guided_preds.append(planned_idx)
            else:
                guided_preds.append(pred)
        else:
            guided_preds.append(pred)
            
    return guided_preds

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
            
        before_counts.update(df['label'].values)
        df['norm_label'] = df['label'].apply(normalize_label)
        after_counts.update([l for l in df['norm_label'].values if pd.notna(l)])
        df = df.dropna(subset=['norm_label']).reset_index(drop=True)
        sessions[session_id] = df
        
    print("\n--- Label Normalization Summary ---")
    all_classes = sorted([str(k) for k in (set(before_counts.keys()) | set(after_counts.keys())) if k is not None])
    print(f"{'Original Label':<25} | {'Normalized':<25} | {'Count'}")
    print("-" * 65)
    for cls in all_classes:
        orig_key = cls
        if cls == "nan":
            norm = "IGNORED"
        else:
            try:
                norm = normalize_label(cls)
                if norm is None:
                    norm = "IGNORED"
            except:
                norm = "ERROR"
        
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

def create_windows_dt(df, config, label_encoder):
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
        
        # Extract features for Decision Tree
        feat = extract_features(window_data)
        X_features.append(feat)
        window_labels.append(labels[mid_idx])
        window_times.append(times[mid_idx])
        
    X_features = np.array(X_features)
    encoded_labels = label_encoder.transform(window_labels)
    
    return X_features, encoded_labels, window_times

def get_transitions(labels, times):
    transitions = []
    if len(labels) == 0:
        return transitions
        
    current_label = labels[0]
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
        
    current_state = predictions[0]
    candidate = None
    candidate_count = 0
    last_transition_time = times[0] - dwell_seconds 
    
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
            if (time - last_transition_time) >= (dwell_seconds - 1e-6):
                current_state = candidate
                decoded_states.append({'time': time, 'label': current_state})
                last_transition_time = time
                candidate = None
                candidate_count = 0
                
    return decoded_states

def evaluate_transitions(gt_transitions, pred_states, tolerance_sec=3.0):
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

    matches = []
    for i, gt in enumerate(gt_transitions):
        for j, ps in enumerate(pred_states):
            dist = abs(ps['time'] - gt['time'])
            if dist <= tolerance_sec:
                matches.append((i, j, dist))

    matches.sort(key=lambda x: x[2])
    used_gt = set()
    used_pred = set()

    for gt_idx, pred_idx, dist in matches:
        if gt_idx not in used_gt and pred_idx not in used_pred:
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
            results['latencies'].append(pred_states[pred_idx]['time'] - gt_transitions[gt_idx]['time'])

            if pred_states[pred_idx]['label'] == gt_transitions[gt_idx]['to']:
                results['matched_and_correct'] += 1
            else:
                results['matched_but_wrong'] += 1

    results['matched'] = results['matched_and_correct'] + results['matched_but_wrong']
    results['missed'] = len(gt_transitions) - results['matched']
    results['false'] = len(pred_states) - results['matched']

    return results

def run_loso_dt():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed" / "apple"

    if not data_dir.exists():
        raise FileNotFoundError(f"Processed Apple data directory not found: {data_dir}")

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
    sweep_results = {dwell: [] for dwell in CONFIG['dwell_sweep']}
    fold_train_classes = []
    
    for held_out_id in session_ids:
        print(f"\n{'='*20} FOLD: Held out {held_out_id} {'='*20}")
        
        train_sessions = [sid for sid in session_ids if sid != held_out_id]
        train_dfs = [sessions[sid] for sid in train_sessions]
        test_df = sessions[held_out_id]
        
        train_classes_in_fold = set()
        for df in train_dfs:
            train_classes_in_fold.update(df['norm_label'].unique())
        test_classes_in_fold = set(test_df['norm_label'].unique())
        common_classes = sorted(list(train_classes_in_fold.intersection(test_classes_in_fold)))
        
        train_dfs = [df[df['norm_label'].isin(common_classes)].copy() for df in train_dfs]
        test_df = test_df[test_df['norm_label'].isin(common_classes)].copy()
        
        if len(test_df) == 0:
            print(f"  WARNING: No windows left in test set after filtering! Skipping fold.")
            continue
            
        le_fold = LabelEncoder()
        le_fold.fit(common_classes)
        
        # Scaling is less critical for Trees but extract_features output might benefit if using distance-based parts
        scaler = StandardScaler()
        sensors = ['acc_x_filt', 'acc_y_filt', 'acc_z_filt', 'gyro_x_filt', 'gyro_y_filt', 'gyro_z_filt']
        
        train_data_full = pd.concat(train_dfs)
        scaler.fit(train_data_full[sensors])
        for df in train_dfs:
            df[sensors] = scaler.transform(df[sensors])
        test_df_scaled = test_df.copy()
        test_df_scaled[sensors] = scaler.transform(test_df_scaled[sensors])
        
        train_classes = set()
        for df in train_dfs:
            train_classes.update(df['norm_label'].unique())
        fold_train_classes.append(train_classes)
        
        X_train_list, y_train_list = [], []
        for df in train_dfs:
            X, y, _ = create_windows_dt(df, CONFIG, le_fold)
            X_train_list.append(X)
            y_train_list.append(y)
        
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test, y_test, t_test = create_windows_dt(test_df_scaled, CONFIG, le_fold)
        
        print(f"Train windows: {len(X_train)}, Test windows: {len(X_test)}")
        class_dist = Counter(le_fold.inverse_transform(y_test))
        print(f"Test Class Distribution: {dict(class_dist)}")
        
        # Train Decision Tree
        model = DecisionTreeClassifier(
            max_depth=CONFIG['max_depth'],
            min_samples_split=CONFIG['min_samples_split'],
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # Evaluate Layer A
        y_pred_fold = model.predict(X_test)
        
        # [NEW] Plan-based Guidance for Inference
        project_root = Path(__file__).resolve().parents[2]
        plan_path = project_root / "workout.json"
        plan = load_workout_plan(plan_path)
        if plan:
            print(f"  [GUIDANCE] Applying workout plan to guide inference for session {held_out_id}")
            y_pred_fold = guide_predictions(y_pred_fold, t_test, plan, le_fold)
        
        y_test_labels = le_fold.inverse_transform(y_test)
        y_pred_labels = le_fold.inverse_transform(y_pred_fold)
        
        pooled_y_true.extend(le.transform(y_test_labels))
        pooled_y_pred.extend(le.transform(y_pred_labels))
        
        acc = accuracy_score(y_test, y_pred_fold)
        f1 = f1_score(y_test, y_pred_fold, average='macro')

        fold_plot_dir = project_root / "src" / "training" / "loso_fold_plots_dt"
        plot_loso_fold_results(
            held_out_id=held_out_id,
            test_df_scaled=test_df_scaled,
            window_times=t_test,
            expected_labels=y_test_labels,
            predicted_labels=y_pred_labels,
            accuracy=acc,
            f1_macro=f1,
            output_dir=fold_plot_dir,
        )
        
        rest_labels = [l for l in le_fold.classes_ if "REST" in l.upper()]
        if rest_labels:
            rest_idx = le_fold.transform([rest_labels[0]])[0]
            y_baseline = np.full_like(y_test, rest_idx)
            baseline_acc = accuracy_score(y_test, y_baseline)
            delta = acc - baseline_acc
            print(f"Fold Macro-F1: {f1:.4f}")
            print(f"Fold Accuracy: {acc:.4f} (Baseline: {baseline_acc:.4f}, Delta: {delta:+.4f})")
        
        fold_results.append({
            'session_id': held_out_id,
            'accuracy': acc,
            'f1_macro': f1,
            'class_dist': dict(class_dist)
        })
        
        # Evaluate Layer B
        gt_transitions = get_transitions(y_test_labels, t_test)
        session_duration_min = (t_test[-1] - t_test[0]) / 60.0
        
        for dwell in CONFIG['dwell_sweep']:
            pred_states = dwell_decode(y_pred_labels, t_test, dwell, CONFIG['step_size_sec'])
            res = evaluate_transitions(gt_transitions, pred_states, CONFIG['tolerance_sec'])
            
            recall = res['matched'] / res['total_gt'] if res['total_gt'] > 0 else 0
            precision = res['matched_and_correct'] / res['total_emitted'] if res['total_emitted'] > 0 else 0
            false_trans_per_min = res['false'] / session_duration_min
            med_latency = np.median(res['latencies']) if res['latencies'] else np.nan
            
            sweep_results[dwell].append({
                'session_id': held_out_id,
                'recall': recall,
                'precision': precision,
                'ft_per_min': false_trans_per_min,
                'med_latency': med_latency,
                'correct_class': res['matched_and_correct'],
                'wrong_class': res['matched_but_wrong'],
                'n_true': res['total_gt'],
                'n_emitted': res['total_emitted']
            })

    # Summary
    print("\n\n" + "="*50)
    print("FINAL LOSO SUMMARY (Decision Tree)")
    print("="*50)
    
    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['f1_macro'] for r in fold_results]
    
    print(f"\nLayer A (Per-window):")
    print(f"Pooled Accuracy: {accuracy_score(pooled_y_true, pooled_y_pred):.4f}")
    print(f"Pooled Macro-F1: {f1_score(pooled_y_true, pooled_y_pred, average='macro'):.4f}")
    
    p, r, f, s = precision_recall_fscore_support(pooled_y_true, pooled_y_pred, labels=range(num_classes))
    print(f"\n{'Class':<25} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'Support'}")
    print("-" * 60)
    for i, class_name in enumerate(le.classes_):
        print(f"{class_name:<25} | {p[i]:<6.4f} | {r[i]:<6.4f} | {f[i]:<6.4f} | {s[i]}")
    
    print(f"\nLayer B (Transition Detection) Sweep:")
    print(f"{'Dwell (s)':<10} | {'Recall':<10} | {'Precision':<10} | {'FT/min':<10} | {'Med Lat':<8}")
    print("-" * 60)
    for dwell in CONFIG['dwell_sweep']:
        d_res = sweep_results[dwell]
        avg_recall = np.mean([r['recall'] for r in d_res])
        avg_prec = np.mean([r['precision'] for r in d_res])
        avg_ft = np.mean([r['ft_per_min'] for r in d_res])
        avg_med_lat = np.nanmean([r['med_latency'] for r in d_res])
        print(f"{dwell:<10} | {avg_recall:<10.4f} | {avg_prec:<10.4f} | {avg_ft:<10.4f} | {avg_med_lat:<8.2f}")

    # Confusion Matrix
    cm = confusion_matrix(pooled_y_true, pooled_y_pred, normalize='true')
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=le.classes_, yticklabels=le.classes_, cmap='turbo')
    plt.title("Pooled LOSO Confusion Matrix - DT")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("loso_confusion_matrix_dt.png")
    
    # Save JSON
    results_to_save = {
        'fold_results': fold_results,
        'sweep_results': sweep_results,
        'config': CONFIG
    }
    def json_serialize(obj):
        if isinstance(obj, (np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        if isinstance(obj, Counter): return dict(obj)
        if np.isnan(obj): return None
        return obj

    with open("loso_results_dt.json", "w") as f:
        json.dump(results_to_save, f, default=json_serialize, indent=4)

def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

def _draw_label_ribbon(ax, y_pos, labels, times, label_name, color_map):
    if len(labels) == 0: return
    start_time = times[0]
    current_label = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            ax.barh(y_pos, times[i] - start_time, left=start_time, height=0.35, color=color_map[current_label])
            start_time = times[i]
            current_label = labels[i]
    ax.barh(y_pos, times[-1] - start_time, left=start_time, height=0.35, color=color_map[current_label])
    ax.text(times[0], y_pos + 0.28, label_name, va="bottom", ha="left", fontsize=10, fontweight="bold")

def plot_loso_fold_results(held_out_id, test_df_scaled, window_times, expected_labels, predicted_labels, accuracy, f1_macro, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_classes = sorted(set(expected_labels) | set(predicted_labels))
    cmap = plt.get_cmap("tab20")
    color_map = {cls: cmap(i % cmap.N) for i, cls in enumerate(all_classes)}
    for cls in all_classes:
        if "REST" in cls.upper(): color_map[cls] = "#d9d9d9"
    fig, (ax_signal, ax_ribbon) = plt.subplots(2, 1, figsize=(18, 7), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    if "rel_time" in test_df_scaled.columns and "acc_z_filt" in test_df_scaled.columns:
        ax_signal.plot(test_df_scaled["rel_time"], test_df_scaled["acc_z_filt"], color="black", linewidth=0.8, alpha=0.75)
    ax_signal.set_title(f"LOSO DT Fold: {held_out_id} | Acc={accuracy:.4f} | F1={f1_macro:.4f}")
    ax_signal.set_ylabel("Scaled accel Z")
    _draw_label_ribbon(ax_ribbon, 1.0, expected_labels, window_times, "Expected", color_map)
    _draw_label_ribbon(ax_ribbon, 0.0, predicted_labels, window_times, "Predicted", color_map)
    ax_ribbon.set_ylim(-0.7, 1.6)
    ax_ribbon.set_yticks([])
    ax_ribbon.set_xlabel("Relative time (seconds)")
    plt.tight_layout()
    plt.savefig(output_dir / f"loso_fold_{_safe_filename(held_out_id)}.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    run_loso_dt()
