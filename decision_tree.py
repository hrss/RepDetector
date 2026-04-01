import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt
from collections import deque, Counter
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# --- CONFIGURATION ---
CONFIG = {
    'sample_rate': 20,
    'window_size_sec': 2.5,
    'step_size_sec': 0.5,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
    'max_depth': 15,
    'min_samples_split': 5
}


# --- 1. SIGNAL PROCESSING & FEATURE EXTRACTION ---
def apply_butterworth(data, cutoff, fs, order=4):
    """Zero-phase low-pass filter to clean the 20Hz jitter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# def extract_features(window):
#     """
#     Translates a (50, 6) raw time-series window into 34 statistical features.
#     Axes: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
#     """
#     features = []
#
#     # 1. Per-Axis Statistics (30 features)
#     for i in range(6):
#         axis_data = window[:, i]
#         features.extend([
#             np.mean(axis_data),
#             np.std(axis_data),
#             np.min(axis_data),
#             np.max(axis_data),
#             np.max(axis_data) - np.min(axis_data)  # Peak-to-Peak Range
#         ])
#
#     # 2. Cross-Axis Statistics (Signal Magnitude Vectors) (4 features)
#     # This gives the tree an understanding of "Total Energy" regardless of watch orientation
#     acc_smv = np.sqrt(window[:, 0] ** 2 + window[:, 1] ** 2 + window[:, 2] ** 2)
#     features.append(np.mean(acc_smv))
#     features.append(np.std(acc_smv))
#
#     gyro_smv = np.sqrt(window[:, 3] ** 2 + window[:, 4] ** 2 + window[:, 5] ** 2)
#     features.append(np.mean(gyro_smv))
#     features.append(np.std(gyro_smv))
#
#     return np.array(features)


#20 features (Added FFT Dominant Frequencies)
def extract_features(window):
    """
    20 carefully selected features that force the tree to learn
    universal physics (Accel), technique nuances (Gyro), and 
    repetition cadence (FFT).
    """
    # --- 8 ACCELEROMETER FEATURES (The Universal Physics) ---

    # 1. Total Kinetic Energy
    acc_smv = np.sqrt(window[:, 0] ** 2 + window[:, 1] ** 2 + window[:, 2] ** 2)
    acc_smv_mean = np.mean(acc_smv)

    # 2. Vertical Displacement & Orientation (Z-Axis) -> Crucial for Squats/Deadlifts
    acc_z = window[:, 2]
    acc_z_mean = np.mean(acc_z)
    acc_z_range = np.max(acc_z) - np.min(acc_z)
    acc_z_std = np.std(acc_z)

    # 3. Forward/Back Displacement & Orientation (Y-Axis) -> Crucial for arm hanging vs rack
    acc_y = window[:, 1]
    acc_y_mean = np.mean(acc_y)
    acc_y_range = np.max(acc_y) - np.min(acc_y)

    # 4. Side-to-Side Displacement & Orientation (X-Axis)
    acc_x = window[:, 0]
    acc_x_mean = np.mean(acc_x)
    acc_x_range = np.max(acc_x) - np.min(acc_x)

    # --- 8 GYROSCOPE FEATURES (The Technique Nuances) ---

    # 5. Total Rotational Energy
    gyro_smv = np.sqrt(window[:, 3] ** 2 + window[:, 4] ** 2 + window[:, 5] ** 2)
    gyro_smv_mean = np.mean(gyro_smv)
    gyro_smv_std = np.std(gyro_smv)

    # 6. Specific Wrist Rotations (Peaks and Valleys of technique)
    gyro_x = window[:, 3]
    gyro_x_max = np.max(gyro_x)
    gyro_x_min = np.min(gyro_x)

    # 7. Specific Wrist Rotations (Peaks and Valleys of technique)
    gyro_y = window[:, 4]
    gyro_y_max = np.max(gyro_y)
    gyro_y_min = np.min(gyro_y)

    # 8. Specific Wrist Rotations (Peaks and Valleys of technique)
    gyro_z = window[:, 5]
    gyro_z_max = np.max(gyro_z)
    gyro_z_min = np.min(gyro_z)

    # --- 4 FFT FEATURES (The Repetition Cadence) ---
    # These help distinguish between fast/slow movements and identify periodic patterns
    
    # FFT for Accel SMV
    acc_fft = np.abs(np.fft.rfft(acc_smv - acc_smv_mean))
    acc_dom_freq_idx = np.argmax(acc_fft[1:]) + 1  # Skip DC component
    acc_max_power = acc_fft[acc_dom_freq_idx]

    # FFT for Gyro SMV
    gyro_fft = np.abs(np.fft.rfft(gyro_smv - gyro_smv_mean))
    gyro_dom_freq_idx = np.argmax(gyro_fft[1:]) + 1  # Skip DC component
    gyro_max_power = gyro_fft[gyro_dom_freq_idx]

    return np.array([
        # The 8 Accel Features
        acc_smv_mean, acc_z_mean, acc_z_range, acc_z_std,
        acc_y_mean, acc_y_range, acc_x_mean, acc_x_range,
        # The 8 Gyro Features
        gyro_smv_mean, gyro_smv_std,
        gyro_x_max, gyro_x_min, gyro_y_max, gyro_y_min, gyro_z_max, gyro_z_min,
        # The 4 FFT Features
        acc_dom_freq_idx, acc_max_power,
        gyro_dom_freq_idx, gyro_max_power
    ])

def smooth_predictions(raw_preds, window_size=12):
    """Majority vote smoothing to prevent flickering predictions."""
    smoothed = []
    history = deque([raw_preds[0]] * window_size, maxlen=window_size)
    for p in raw_preds:
        history.append(p)
        vote = Counter(history).most_common(1)[0][0]
        smoothed.append(vote)
    return smoothed


# --- 2. DATASET LOADER ---
def load_and_window_data(data_dir, config):
    print(f"Loading training data from {data_dir}...")
    section_dirs = glob.glob(os.path.join(data_dir, "**", "section_*"), recursive=True)

    window_pts = int(config['window_size_sec'] * config['sample_rate'])
    step_pts = int(config['step_size_sec'] * config['sample_rate'])

    X_list = []
    y_list = []

    for s_dir in section_dirs:
        csv_path = os.path.join(s_dir, "imu.csv")
        json_path = os.path.join(s_dir, "data.json")

        if not os.path.exists(csv_path) or not os.path.exists(json_path):
            continue

        df = pd.read_csv(csv_path)
        with open(json_path, 'r', encoding="utf-8") as f:
            metadata = json.load(f)

        sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        for col in sensors:
            df[f'{col}_filt'] = apply_butterworth(df[col], config['lowpass_cutoff'], config['sample_rate'])

        df['label'] = 'Rest'
        for round_data in metadata.get("roundResults", []):
            for ex in round_data.get("exerciseResults", []):
                mask = (df['rel_time'] >= ex['startTime']) & (df['rel_time'] <= ex['endTime'])
                df.loc[mask, 'label'] = ex['name']

        filt_cols = [f'{s}_filt' for s in sensors]
        signals = df[filt_cols].values
        labels = df['label'].values

        for i in range(0, len(df) - window_pts, step_pts):
            window_data = signals[i: i + window_pts]
            mid_idx = i + window_pts // 2

            # --- APPLY FEATURE ENGINEERING ---
            stat_features = extract_features(window_data)
            X_list.append(stat_features)
            y_list.append(labels[mid_idx])

    X_np = np.array(X_list)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_list)

    print(f"Created {len(X_np)} windows. Feature Vector Shape: {X_np.shape} across {len(set(y_encoded))} classes.")
    return X_np, y_encoded, le


# --- 3. TEST AND VISUALIZE INFERENCE ---
def test_and_plot_section(section_dir, model, label_encoder, config):
    print(f"\nTesting Inference on Section: {os.path.basename(section_dir)}")

    csv_path = os.path.join(section_dir, "imu.csv")
    json_path = os.path.join(section_dir, "data.json")

    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        return

    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in sensors:
        df[f'{col}_filt'] = apply_butterworth(df[col], config['lowpass_cutoff'], config['sample_rate'])

    df['truth'] = 'Rest'
    for round_data in metadata.get("roundResults", []):
        for ex in round_data.get("exerciseResults", []):
            mask = (df['rel_time'] >= ex['startTime']) & (df['rel_time'] <= ex['endTime'])
            df.loc[mask, 'truth'] = ex['name']

    window_pts = int(config['window_size_sec'] * config['sample_rate'])
    predictions_raw = []
    time_tracker = []

    filt_cols = [f'{s}_filt' for s in sensors]
    signals = df[filt_cols].values

    for i in range(len(df)):
        if i < window_pts:
            predictions_raw.append("Rest")
            time_tracker.append(df['rel_time'].iloc[i])
            continue

        window_data = signals[i - window_pts: i]

        # --- APPLY FEATURE ENGINEERING TO INFERENCE WINDOW ---
        x_features = extract_features(window_data).reshape(1, -1)

        pred_idx = model.predict(x_features)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        predictions_raw.append(pred_label)
        time_tracker.append(df['rel_time'].iloc[i])

    predictions_smooth = smooth_predictions(predictions_raw, window_size=10)

    res_df = pd.DataFrame({
        'time': time_tracker,
        'truth': df['truth'].values,
        'pred_raw': predictions_raw,
        'pred_smooth': predictions_smooth,
        'acc_z': df['acc_z_filt'].values
    })

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    unique_classes = sorted(list(set([str(c) for c in label_encoder.classes_] + ['Rest'])))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_classes))]
    color_map = {cls: col for cls, col in zip(unique_classes, colors)}
    color_map['Rest'] = '#e0e0e0'

    ax1.plot(res_df['time'], res_df['acc_z'], color='black', alpha=0.7, linewidth=1, label='Accel Z (Filtered)')
    section_id = metadata.get("sectionId", "Unknown")[:8]
    ax1.set_title(f"Signal Data vs Ground Truth (Section: {section_id}...)")
    ax1.set_ylabel("G-Force")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    def draw_ribbons(ax, y_pos, labels, times, title):
        if len(times) < 2: return
        times = np.array(times)
        labels = np.array(labels)
        step = times[1] - times[0]
        for t, lab in zip(times, labels):
            rect = mpatches.Rectangle((t - step / 2, y_pos), step, 0.8, color=color_map.get(lab, 'gray'))
            ax.add_patch(rect)
        offset = (times[-1] - times[0]) * 0.02
        ax.text(times[0] - offset, y_pos + 0.4, title, va='center', ha='right', fontweight='bold')

    draw_ribbons(ax2, 0.0, res_df['pred_smooth'], res_df['time'], "Smoothed AI")
    draw_ribbons(ax2, 1.0, res_df['pred_raw'], res_df['time'], "Raw Decision Tree")
    draw_ribbons(ax2, 2.0, res_df['truth'], res_df['time'], "Ground Truth")

    ax2.set_ylim(-0.5, 3.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Relative Time (Seconds)")
    ax2.set_title("Decision Tree Pipeline Results")

    handles = [mpatches.Patch(color=color_map[c], label=c) for c in unique_classes]
    fig.legend(handles=handles, loc='lower center', ncol=len(unique_classes), bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    save_name = f"dt_inference_{section_id}.png"
    plt.savefig(save_name, bbox_inches="tight")
    print(f"Saved plot to {save_name}")
    plt.show()


# --- 4. EXPORT AND SAVE ---
def save_model(model, le, num_features):
    # 1. Save standard Python assets (for retraining/analysis)
    joblib.dump(le, "wodbuddy_dt_label_encoder.pkl")
    joblib.dump(model, "wodbuddy_dt_model.pkl")

    # 2. Export ONNX for the Go Backend
    initial_type = [('input_sensors', FloatTensorType([None, num_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open("wodbuddy_dt_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Exported Go Backend model to wodbuddy_dt_model.onnx")

    # 3. Export JSON for the Garmin Watch
    export_tree_for_garmin(model, le, "wodbuddy_dt.json")


def print_feature_importances(model):
    """Maps the 34 array indices back to human-readable names and ranks them."""

    # 1. Recreate the exact naming structure of our extract_features() array
    axes = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    stats = ['mean', 'std', 'min', 'max', 'range']

    feature_names = []
    for axis in axes:
        for stat in stats:
            feature_names.append(f"{axis}_{stat}")

    feature_names.extend(['acc_smv_mean', 'acc_smv_std', 'gyro_smv_mean', 'gyro_smv_std'])

    # 2. Extract the Gini importance scores from the trained tree
    importances = model.feature_importances_

    # 3. Zip them together and sort highest to lowest
    feature_importance_pairs = list(zip(feature_names, importances))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

    # 4. Print the lean math required for the Garmin
    print("\n--- DECISION TREE FEATURE IMPORTANCES ---")
    print("These are the ONLY features the Garmin watch actually needs to calculate:\n")

    cumulative_importance = 0.0
    features_to_keep = 0

    for name, importance in feature_importance_pairs:
        # Ignore anything that contributes less than 1% to the model's decisions
        if importance > 0.01:
            print(f"{name: <15} | Weight: {importance:.4f}")
            cumulative_importance += importance
            features_to_keep += 1

    print("-" * 40)
    print(f"Total features to calculate in Monkey C: {features_to_keep} out of 34")
    print(f"Cumulative predictive power retained: {cumulative_importance * 100:.1f}%")


def print_lean_feature_importances(model):
    """Accurately maps the 20 features to their weights."""
    feature_names = [
        # The 8 Accel Features
        "acc_smv_mean", "acc_z_mean", "acc_z_range", "acc_z_std",
        "acc_y_mean", "acc_y_range", "acc_x_mean", "acc_x_range",
        # The 8 Gyro Features
        "gyro_smv_mean", "gyro_smv_std",
        "gyro_x_max", "gyro_x_min", "gyro_y_max", "gyro_y_min", "gyro_z_max", "gyro_z_min",
        # The 4 FFT Features
        "acc_dom_freq", "acc_max_power",
        "gyro_dom_freq", "gyro_max_power"
    ]

    importances = model.feature_importances_

    feature_importance_pairs = list(zip(feature_names, importances))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

    print("\n--- ACTUAL DECISION TREE WEIGHTS ---")
    for name, importance in feature_importance_pairs:
        if importance > 0.001:
            print(f"{name: <15} | Weight: {importance:.4f}")

def export_tree_for_garmin(clf, label_encoder, filename="wodbuddy_dt.json"):
    """Extracts the tree into a flat JSON array for Monkey C."""
    tree = clf.tree_

    ciq_payload = {
        "meta": {
            "nodes": tree.node_count,
            "classes": label_encoder.classes_.tolist()
        },
        "f": tree.feature.tolist(),
        "t": tree.threshold.tolist(),
        "l": tree.children_left.tolist(),
        "r": tree.children_right.tolist(),
        "c": np.argmax(tree.value[:, 0, :], axis=1).tolist()
    }

    with open(filename, 'w') as f:
        # Use separators to minify the JSON and save Garmin RAM
        json.dump(ciq_payload, f, separators=(',', ':'))

    print(f"Exported Garmin Watch model to {filename} ({tree.node_count} nodes)")

# --- 5. EXECUTION ---
if __name__ == "__main__":
    X, y, label_encoder = load_and_window_data(data_dir="data", config=CONFIG)
    num_features = X.shape[1]

    print("\n--- Training Decision Tree ---")
    model = DecisionTreeClassifier(
        max_depth=CONFIG['max_depth'],
        min_samples_split=CONFIG['min_samples_split'],
        random_state=42
    )
    model.fit(X, y)
    print_lean_feature_importances(model)
    print(f"Training Complete. Model Depth: {model.get_depth()}")

    save_model(model, label_encoder, num_features)

    # Pick one of your sliced sections to test against
    test_target_dir = "data/1797/section_3109"
    test_and_plot_section(test_target_dir, model, label_encoder, CONFIG)




