import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from src.training.data_loader import load_and_window_data_for_dt, load_raw_section_data
from src.core.data_utils import smooth_predictions, extract_features
from src.visualization.inference_viz import plot_classification_results

# --- CONFIGURATION ---
CONFIG = {
    'sample_rate': 20,
    'window_size_sec': 2.5,
    'step_size_sec': 0.5,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
    'max_depth': 15,
    'min_samples_split': 5,
    'allowed_classes': ['Air Squat', 'Shoulder Press', 'Shuttle Run', 'Rest'],  # List of strings or None for all
}




def test_and_plot_section(section_dir, model, label_encoder, config):
    print(f"\nTesting Inference on Section: {os.path.basename(section_dir)}")

    df, metadata = load_raw_section_data(section_dir, config)
    if df is None:
        return

    df['truth'] = 'Rest'
    for round_data in metadata.get("roundResults", []):
        for ex in round_data.get("exerciseResults", []):
            mask = (df['rel_time'] >= ex['startTime']) & (df['rel_time'] <= ex['endTime'])
            df.loc[mask, 'truth'] = ex['name']

    window_pts = int(config['window_size_sec'] * config['sample_rate'])
    predictions_raw = []
    time_tracker = []

    sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
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

    plot_classification_results(res_df, metadata, label_encoder, f"dt_inference_{metadata.get('sectionId', 'unknown')[:8]}.png")


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


def load_model_and_test_section():
    """Load the saved Decision Tree model and test it on a section."""
    print("Loading saved model and label encoder...")

    # Load the saved model and label encoder
    model = joblib.load("wodbuddy_dt_model.pkl")
    label_encoder = joblib.load("wodbuddy_dt_label_encoder.pkl")

    print(f"Model loaded successfully. Tree depth: {model.get_depth()}")
    print(f"Classes: {label_encoder.classes_}")

    # Test on a specific section
    test_target_dir = "data/2053/section_3708"
    test_and_plot_section(test_target_dir, model, label_encoder, CONFIG)
    """Load the saved Decision Tree model and test it on a section."""
    print("Loading saved model and label encoder...")

    # Load the saved model and label encoder
    model = joblib.load("wodbuddy_dt_model.pkl")
    label_encoder = joblib.load("wodbuddy_dt_label_encoder.pkl")

    print(f"Model loaded successfully. Tree depth: {model.get_depth()}")
    print(f"Classes: {label_encoder.classes_}")

    # Test on a specific section
    test_target_dir = "data/2053/section_3708"
    test_and_plot_section(test_target_dir, model, label_encoder, CONFIG)

# --- 5. EXECUTION ---
if __name__ == "__main__":
    X, y, label_encoder = load_and_window_data_for_dt(data_dir="data", config=CONFIG)
    # num_features = X.shape[1]
    #
    # print(f"\nOriginal Class Distribution: {Counter(y)}")
    #
    # # 1. Find the integer label for "Rest"
    # try:
    #     rest_label = label_encoder.transform(['Rest'])[0]
    # except ValueError:
    #     print("Warning: 'Rest' not found in this dataset slice.")
    #     rest_label = None
    #
    # if rest_label is not None:
    #     # 2. Find the size of the next biggest class
    #     counts = Counter(y)
    #     non_rest_counts = {k: v for k, v in counts.items() if k != rest_label}
    #
    #     if non_rest_counts:
    #         next_biggest_size = max(non_rest_counts.values())
    #
    #         # --- THE FIX: Add a multiplier to keep Rest dominant ---
    #         rest_multiplier = 10.0  # Try 1.5x or 2.0x
    #         rest_target_size = int(next_biggest_size * rest_multiplier)
    #
    #         print(f"\nNext biggest exercise size: {next_biggest_size}")
    #         print(f"New target size for 'Rest': {rest_target_size}")
    #
    #         # 3. Undersample ONLY the "Rest" class to the new multiplier
    #         if counts[rest_label] > rest_target_size:
    #             print("Undersampling 'Rest' data...")
    #             rus = RandomUnderSampler(
    #                 sampling_strategy={rest_label: rest_target_size},
    #                 random_state=42
    #             )
    #             X, y = rus.fit_resample(X, y)
    #             print(f"Post-Undersampling Distribution: {Counter(y)}")
    #
    #         # 4. Oversample minority classes using SMOTE
    #         print("\nOversampling minority classes with SMOTE...")
    #
    #         # We explicitly define the SMOTE strategy so it only inflates
    #         # the exercises up to 'next_biggest_size', NOT all the way up to 'Rest'.
    #         current_counts = Counter(y)
    #         smote_strategy = {
    #             cls: next_biggest_size
    #             for cls in current_counts.keys()
    #             if cls != rest_label and current_counts[cls] < next_biggest_size
    #         }
    #
    #         if smote_strategy:
    #             smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
    #             X, y = smote.fit_resample(X, y)
    #
    #         print(f"Final Balanced Distribution: {Counter(y)}\n")
    #
    # print("--- Training Decision Tree ---")
    # model = DecisionTreeClassifier(
    #     max_depth=CONFIG['max_depth'],
    #     min_samples_split=CONFIG['min_samples_split'],
    #     random_state=42
    # )
    # model.fit(X, y)
    # print_lean_feature_importances(model)
    # print(f"Training Complete. Model Depth: {model.get_depth()}")
    #
    # save_model(model, label_encoder, num_features)
    #
    # Pick one of your sliced sections to test against

    # test_target_dir = "data/2053/section_3708"
    # test_and_plot_section(test_target_dir, model, label_encoder, CONFIG)

    # Or use the new function to load and test
    load_model_and_test_section()


