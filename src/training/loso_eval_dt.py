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

from src.core.exercises import canonicalize_label
from src.training.data_loader import load_raw_section_data
from src.core.data_utils import extract_features
from src.training.decoders import WodDecoder, RestPolicy

# --- CONFIGURATION ---
# Decision Tree usually works better with the features at 25Hz as per device requirement
CONFIG = {
    'sample_rate': 25,
    'window_size_sec': 2.5,   # was 2.0  -> 62 samples at 25 Hz, floor back to 0.4 Hz
    'step_size_sec': 0.5,     # was 0.4  -> 12 samples, keeps ~4 windows/sec
    'lowpass_cutoff': 3,
    'filter_order': 4,
    'max_depth': 15,
    'min_samples_split': 5,
    'dwell_sweep': [2, 3, 5, 8, 10],
    'default_dwell': 5.0,
    'tolerance_sec': 3.0,
}

IGNORE_LABELS = ["null", "setup"]

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
            norm_name = canonicalize_label(name)
            if norm_name: # might be None for 'Setup'
                normalized_plan.append({
                    'name': norm_name,
                    'start': start,
                    'end': end
                })
    return normalized_plan

def calculate_wod_metrics(y_true_labels, y_pred_labels, times, step_size_sec, tolerance_sec=5.0):
    """
    Calculates detailed WOD metrics as requested.
    """
    T = len(y_true_labels)
    correct_mask = y_true_labels == y_pred_labels

    acc = accuracy_score(y_true_labels, y_pred_labels)
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

    false_transitions = 0
    premature_transitions = 0
    latencies = []

    matched_gt = set()
    for pt in pred_trans:
        found_match = False
        for i, gt in enumerate(gt_trans):
            if i in matched_gt: continue
            if pt['from'] == gt['from'] and pt['to'] == gt['to']:
                latency = pt['time'] - gt['time']
                if abs(latency) <= 15.0:
                    matched_gt.add(i)
                    latencies.append(latency)
                    found_match = True
                    if latency < -tolerance_sec:
                        premature_transitions += 1
                    break
        if not found_match:
            false_transitions += 1

    session_duration_min = (T * step_size_sec) / 60.0
    ft_per_min = false_transitions / session_duration_min if session_duration_min > 0 else 0
    premature_rate = premature_transitions / len(pred_trans) if len(pred_trans) > 0 else 0
    unrecoverable = 1 if not correct_mask[-1] else 0
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
        df = apply_labels(df)
        after_counts.update([l for l in df['norm_label'].values if pd.notna(l)])
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
                norm = canonicalize_label(cls)
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


def apply_labels(df, strict=True):
    df = df.copy()
    df['norm_label'] = df['label'].apply(
        lambda x: canonicalize_label(x, strict=strict, display=True)  # noqa: F821
    )
    # canonicalize_label returns None for setup/null/transition -> DROP them.
    # Under the broken version these became REST and were used as training data.
    n_before = len(df)
    df = df.dropna(subset=['norm_label']).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"    dropped {n_dropped} rows with ignore labels (setup/null/etc)")
    return df


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
        'matched': 0,
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

def build_wod_sequence(plan_data, le_fold):
    sequence = []
    missing = []
    for ex in plan_data.get('exercises', []):
        raw = ex.get('canonicalName') or ex.get('name')
        name = canonicalize_label(raw, strict=False, display=True)  # noqa: F821
        if name is None:
            continue                      # Setup/null entries in the plan
        if name in le_fold.classes_:
            sequence.append(name)
        else:
            missing.append((raw, name))

    if missing:
        print(f"  [WOD] WARNING: {len(missing)} planned exercise(s) not in this "
              f"fold's classes -> decoders DISABLED: {missing}")
        print(f"        fold classes: {list(le_fold.classes_)}")
        return []

    return sequence * plan_data.get('rounds', 1)


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
    variant_aggregate_results = {}

    for held_out_id in session_ids:
        print(f"\n{'='*20} FOLD: Held out {held_out_id} {'='*20}")

        train_sessions = [sid for sid in session_ids if sid != held_out_id]
        train_dfs = [sessions[sid] for sid in train_sessions]
        test_df = sessions[held_out_id]

        # We need ALL classes for the global pooled metrics later
        # But we only train on classes present in the train set.
        train_classes_in_fold = set()
        for df in train_dfs:
            train_classes_in_fold.update(df['norm_label'].unique())

        # Ensure we have common classes or at least all train classes
        le_fold = LabelEncoder()
        le_fold.fit(sorted(list(train_classes_in_fold)))

        X_train_list, y_train_list = [], []
        for df in train_dfs:
            X, y, _ = create_windows_dt(df, CONFIG, le_fold)
            X_train_list.append(X)
            y_train_list.append(y)

        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Pre-filter test_df to only include classes model knows about for initial predict
        test_df_valid = test_df[test_df['norm_label'].isin(le_fold.classes_)].copy()
        if len(test_df_valid) == 0:
            print(f"  [FOLD] Skipping: No test data with labels known to train set.")
            continue

        X_test, y_test, t_test = create_windows_dt(test_df_valid, CONFIG, le_fold)
        y_test_labels = le_fold.inverse_transform(y_test)

        print(f"Train windows: {len(X_train)}, Test windows: {len(X_test)}")
        class_dist = Counter(y_test_labels)
        print(f"Test Class Distribution: {dict(class_dist)}")

        # Fit on TRAIN FEATURES only, apply to both. This is what gets exported
        # as the 's' block in model_data.json.
        feat_scaler = StandardScaler()
        X_train_s = feat_scaler.fit_transform(X_train)
        X_test_s = feat_scaler.transform(X_test)

        # Train Decision Tree
        model = DecisionTreeClassifier(
            max_depth=CONFIG['max_depth'],
            min_samples_split=CONFIG['min_samples_split'],
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_s, y_train)

        # Probabilities for WOD decoders
        all_probs = model.predict_proba(X_test_s)

        # 5. [NEW] Revisable decoders
        # Load workout plan
        plan_path = project_root / "workout2.json"
        wod_sequence = []
        if os.path.exists(plan_path):
            try:
                with open(plan_path, 'r', encoding='utf-8') as f:
                    plan_data = json.load(f)
                    wod_sequence = build_wod_sequence(plan_data, le_fold)
            except Exception as e:
                print(f"  [WOD] Could not load workout plan: {e}")
                wod_sequence = []

        decoder = None
        if wod_sequence:
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
                {'name': 'viterbi', 'type': 'viterbi'}
            ])

        fold_variant_results = {}
        fold_plot_dir = project_root / "src" / "training" / "loso_fold_plots_dt"

        for var in variants:
            v_name = var['name']
            y_pred_idx = []
            rollback_info = []
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
                test_df_scaled=test_df_valid,
                window_times=t_test,
                expected_labels=y_test_labels,
                predicted_labels=y_pred_labels,
                accuracy=metrics['accuracy'],
                f1_macro=metrics['f1'],
                output_dir=fold_plot_dir,
                variant_name=v_name
            )

        # Use 'greedy_wod_rollback_prefer' as primary
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
            'class_dist': dict(class_dist)
        })

        # Evaluate Layer B
        gt_transitions = get_transitions(y_test_labels, t_test)
        session_duration_min = (t_test[-1] - t_test[0]) / 60.0

        for dwell in CONFIG['dwell_sweep']:
            pred_states = dwell_decode(primary_labels, t_test, dwell, CONFIG['step_size_sec'])
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

    # Variant Summary
    print("\nDECODER VARIANT COMPARISON (Averaged over folds):")
    print(f"{'Variant':<30} | {'Acc':<6} | {'Macro-F1':<8} | {'FT/min':<6} | {'Misattr(s)':<10} | {'Rollbacks'}")
    print("-" * 80)
    for v_name, v_metrics in variant_aggregate_results.items():
        avg_acc = np.mean([m['accuracy'] for m in v_metrics])
        avg_f1 = np.mean([m['f1'] for m in v_metrics])
        avg_ft = np.mean([m['false_trans_per_min'] for m in v_metrics])
        avg_mis = np.mean([m['misattributed_sec'] for m in v_metrics])
        total_rollbacks = sum([m.get('rollback_count', 0) for m in v_metrics])
        print(f"{v_name:<30} | {avg_acc:<6.4f} | {avg_f1:<8.4f} | {avg_ft:<6.2f} | {avg_mis:<10.1f} | {total_rollbacks}")

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
    """
    Draws a horizontal colored ribbon representing label sequences.
    """
    if len(labels) == 0:
        return

    # Convert to numpy for faster processing if not already
    labels = np.asarray(labels)
    times = np.asarray(times)

    # Calculate step size
    if len(times) > 1:
        dt = times[1] - times[0]
    else:
        dt = 0.5

    # Find contiguous blocks of the same label
    n = len(labels)
    if n == 0:
        return

    starts = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = np.concatenate(([0], starts))
    ends = np.concatenate((starts[1:], [n]))

    for s, e in zip(starts, ends):
        label = labels[s]
        t_start = times[s] - dt / 2
        t_end = times[min(e, n - 1)] + dt / 2
        color = color_map.get(label, "#333333")

        ax.barh(
            y_pos,
            t_end - t_start,
            left=t_start,
            height=0.6,
            color=color,
            alpha=0.8,
            edgecolor="none",
        )

    ax.text(
        times[0],
        y_pos + 0.35,
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
        f"LOSO DT Fold: {held_out_id} | Var: {variant_name} | Acc={accuracy:.4f} | Macro-F1={f1_macro:.4f}"
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

    print(f"Saved LOSO DT fold plot ({variant_name}) to {save_path}")

if __name__ == "__main__":
    run_loso_dt()
