import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.signal import butter, filtfilt
from collections import deque, Counter
import torch.onnx
import joblib
from sklearn.preprocessing import StandardScaler

# Import the extractor from your first script!
# from fit_slicer import extract_raw_fit_data

# --- CONFIGURATION ---
CONFIG = {
    'sample_rate': 20,
    'window_size_sec': 2.5,  # 20Hz * 2.5s = 50 timesteps
    'step_size_sec': 0.5,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
    'batch_size': 32,
    'epochs': 60,  # Micro networks often need a few more epochs to converge
    'lr': 0.001
}


# --- 1. SIGNAL PROCESSING ---
def apply_butterworth(data, cutoff, fs, order=4):
    """Zero-phase low-pass filter to clean the jitter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


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
        section_dirs = glob.glob(os.path.join(data_dir, "**", "section_*"), recursive=True)

        window_pts = int(self.cfg['window_size_sec'] * self.cfg['sample_rate'])
        step_pts = int(self.cfg['step_size_sec'] * self.cfg['sample_rate'])

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
                df[f'{col}_filt'] = apply_butterworth(df[col], self.cfg['lowpass_cutoff'], self.cfg['sample_rate'])

            df['label'] = 'Rest'
            for round_data in metadata.get("roundResults", []):
                for ex in round_data.get("exerciseResults", []):
                    mask = (df['rel_time'] >= ex['startTime']) & (df['rel_time'] <= ex['endTime'])
                    df.loc[mask, 'label'] = ex['name']

            filt_cols = [f'{s}_filt' for s in sensors]

            # --- NEW: Apply Standard Scaling ---
            # If the scaler doesn't exist yet (first file), fit it. Otherwise, transform.
            if not hasattr(self, 'scaler'):
                self.scaler = StandardScaler()
                df[filt_cols] = self.scaler.fit_transform(df[filt_cols])
            else:
                df[filt_cols] = self.scaler.transform(df[filt_cols])
            # -----------------------------------

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
        x_tensor = torch.FloatTensor(self.X[idx]).transpose(0, 1)
        y_tensor = torch.tensor(self.y_encoded[idx], dtype=torch.long)
        return x_tensor, y_tensor


# --- 3. THE GARMIN MICRO-CNN ARCHITECTURE ---
class GarminMicroCNN(nn.Module):
    def __init__(self, num_classes):
        super(GarminMicroCNN, self).__init__()
        # Input: [Batch, 6 Channels, 50 Timesteps]

        # Layer 1: Learn basic local shapes (16 filters instead of 8)
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Length drops from 48 to 24

        # Layer 2: Learn complex multi-phase movements
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # Length drops from 22 to 11

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)

        # Layer 3: Classification
        # 11 timesteps * 16 filters = 176 flat features
        self.fc = nn.Linear(176, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout(self.flatten(x))
        return self.fc(x)


# --- 4. MONKEY C EXPORTER ---
def export_to_monkey_c(model, label_encoder, filename="wodbuddy_micro_weights.json"):
    """
    Extracts weights, quantizes them to 16-bit integers, flattens the arrays
    to prevent CIQ object-overhead, and exports to JSON.
    """
    print("\n--- Exporting to Monkey C ---")

    # Scale factor for quantization (1024 allows fast bit-shift division in Monkey C: value >> 10)
    SCALE = 1024.0

    # Extract weights and move to CPU numpy arrays
    w_conv = model.conv1.weight.detach().cpu().numpy()
    b_conv = model.conv1.bias.detach().cpu().numpy()
    w_fc = model.fc.weight.detach().cpu().numpy()
    b_fc = model.fc.bias.detach().cpu().numpy()

    # Create the CIQ payload
    ciq_payload = {
        "metadata": {
            "scale_factor": int(SCALE),
            "num_classes": len(label_encoder.classes_),
            "classes": label_encoder.classes_.tolist()
        },
        "conv1": {
            # Shape: [out_channels, in_channels, kernel_size] -> Flattened
            "weights": (w_conv * SCALE).astype(np.int16).flatten().tolist(),
            "bias": (b_conv * SCALE).astype(np.int16).flatten().tolist()
        },
        "fc": {
            # Shape: [out_features, in_features] -> Flattened
            "weights": (w_fc * SCALE).astype(np.int16).flatten().tolist(),
            "bias": (b_fc * SCALE).astype(np.int16).flatten().tolist()
        }
    }

    with open(filename, 'w') as f:
        json.dump(ciq_payload, f)

    print(
        f"Successfully packed {len(ciq_payload['conv1']['weights']) + len(ciq_payload['fc']['weights'])} quantized weights.")
    print(f"Saved optimized CIQ model to {filename}")


# --- 5. TEST AND VISUALIZE INFERENCE ---
def test_and_plot_section(section_dir, model, label_encoder, config, device):
    print(f"\nTesting Inference on Section: {os.path.basename(section_dir)}")

    csv_path = os.path.join(section_dir, "imu.csv")
    json_path = os.path.join(section_dir, "data.json")

    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        print(f"Error: Missing files in {section_dir}")
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

    model.eval()
    with torch.no_grad():
        filt_cols = [f'{s}_filt' for s in sensors]
        signals = df[filt_cols].values

        for i in range(len(df)):
            if i < window_pts:
                predictions_raw.append("Rest")
                time_tracker.append(df['rel_time'].iloc[i])
                continue

            window_data = signals[i - window_pts: i]
            x_tensor = torch.FloatTensor(window_data).transpose(0, 1).unsqueeze(0).to(device)

            logits = model(x_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
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
    draw_ribbons(ax2, 1.0, res_df['pred_raw'], res_df['time'], "Raw AI Output")
    draw_ribbons(ax2, 2.0, res_df['truth'], res_df['time'], "Ground Truth")

    ax2.set_ylim(-0.5, 3.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Relative Time (Seconds)")
    ax2.set_title("Classification Pipeline Results")

    handles = [mpatches.Patch(color=color_map[c], label=c) for c in unique_classes]
    fig.legend(handles=handles, loc='lower center', ncol=len(unique_classes), bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    save_name = f"inference_{section_id}.png"
    plt.savefig(save_name, bbox_inches="tight")
    print(f"Saved plot to {save_name}")
    plt.show()


# --- 6. EXECUTION ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Load Data
    dataset = WodDataset(data_dir="data", config=CONFIG)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    num_classes = len(dataset.le.classes_)

    # 2. Initialize Model
    model = GarminMicroCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # 3. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{CONFIG['epochs']} - Loss: {total_loss / len(dataloader):.4f}")

    print("Training Complete.")

    # 4. Save and Export
    joblib.dump(dataset.le, "wodbuddy_label_encoder.pkl")
    torch.save(model.state_dict(), "wodbuddy_micro_weights.pth")

    # Export the specialized JSON for Monkey C!
    export_to_monkey_c(model, dataset.le, filename="wodbuddy_micro_weights.json")

    # Export to ONNX for your Go backend if needed
    dummy_input = torch.randn(1, 6, 50, device=device)  # Note: 50 timesteps now!
    torch.onnx.export(
        model, dummy_input, "wodbuddy_model.onnx",
        export_params=True, opset_version=14,
        input_names=['input_sensors'], output_names=['class_logits'],
        dynamic_axes={'input_sensors': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}}
    )

    # 5. Test against a sliced section
    test_target_dir = "data/1797/section_3109"
    test_and_plot_section(test_target_dir, model, dataset.le, CONFIG, device)