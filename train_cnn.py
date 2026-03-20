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

# Import the extractor from your first script!
from fit_slicer import extract_raw_fit_data

# --- CONFIGURATION ---
CONFIG = {
    'sample_rate': 20,
    'window_size_sec': 2.5,
    'step_size_sec': 0.5,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
    'batch_size': 32,
    'epochs': 20,
    'lr': 0.001
}


# --- 1. SIGNAL PROCESSING ---
def apply_butterworth(data, cutoff, fs, order=4):
    """Zero-phase low-pass filter to clean the 20Hz jitter."""
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
        section_dirs = glob.glob(os.path.join(data_dir, "section_*"))

        window_pts = int(self.cfg['window_size_sec'] * self.cfg['sample_rate'])
        step_pts = int(self.cfg['step_size_sec'] * self.cfg['sample_rate'])

        for s_dir in section_dirs:
            csv_path = os.path.join(s_dir, "imu_data.csv")
            json_path = os.path.join(s_dir, "metadata.json")

            if not os.path.exists(csv_path) or not os.path.exists(json_path):
                continue

            df = pd.read_csv(csv_path)
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            # 1. Apply Filtering to all 6 axes
            sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            for col in sensors:
                df[f'{col}_filt'] = apply_butterworth(df[col], self.cfg['lowpass_cutoff'], self.cfg['sample_rate'])

            # 2. Assign Ground Truth Labels based on JSON
            df['label'] = 'Rest'  # Default state
            for round_data in metadata.get("roundResults", []):
                for ex in round_data.get("exerciseResults", []):
                    mask = (df['rel_time'] >= ex['startTime']) & (df['rel_time'] <= ex['endTime'])
                    df.loc[mask, 'label'] = ex['name']

            # 3. Slide the Window
            filt_cols = [f'{s}_filt' for s in sensors]
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
        # Shape: [Channels (6), Sequence Length (20)]
        x_tensor = torch.FloatTensor(self.X[idx]).transpose(0, 1)
        y_tensor = torch.tensor(self.y_encoded[idx], dtype=torch.long)
        return x_tensor, y_tensor


# --- 3. THE 1D-CNN ARCHITECTURE ---
class SixAxisCNN(nn.Module):
    def __init__(self, num_classes):
        super(SixAxisCNN, self).__init__()
        # Input shape: (Batch, 6 Channels, 20 Timesteps)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # output: 10 timesteps

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # output: 5 timesteps

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # output: 1 timestep (Global Average Pooling)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --- 4. TEST AND VISUALIZE INFERENCE ---
def test_and_plot_section(section_dir, model, label_encoder, config):
    """Runs inference on a pre-sliced section directory and plots the results."""
    print(f"\nTesting Inference on Section: {os.path.basename(section_dir)}")

    csv_path = os.path.join(section_dir, "imu_data.csv")
    json_path = os.path.join(section_dir, "metadata.json")

    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        print(f"Error: Missing files in {section_dir}")
        return

    # 1. Load the raw sliced data
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    # 2. Apply Filtering to match training conditions
    sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in sensors:
        df[f'{col}_filt'] = apply_butterworth(df[col], config['lowpass_cutoff'], config['sample_rate'])

    # 3. Get Ground Truth for comparison (from the section JSON structure)
    df['truth'] = 'Rest'
    for round_data in metadata.get("roundResults", []):
        for ex in round_data.get("exerciseResults", []):
            mask = (df['rel_time'] >= ex['startTime']) & (df['rel_time'] <= ex['endTime'])
            df.loc[mask, 'truth'] = ex['name']

    # 4. Slide Inference Window
    window_pts = int(config['window_size_sec'] * config['sample_rate'])
    predictions_raw = []
    time_tracker = []

    model.eval()
    with torch.no_grad():
        filt_cols = [f'{s}_filt' for s in sensors]
        signals = df[filt_cols].values

        # Step by 1 frame for a dense, high-resolution prediction map
        for i in range(len(df)):
            if i < window_pts:
                predictions_raw.append("Rest")  # Padding for the first 1-second window
                time_tracker.append(df['rel_time'].iloc[i])
                continue

            window_data = signals[i - window_pts: i]
            # Format for model: [Batch=1, Channels=6, SeqLen=20]
            x_tensor = torch.FloatTensor(window_data).transpose(0, 1).unsqueeze(0).to(device)

            logits = model(x_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

            predictions_raw.append(pred_label)
            time_tracker.append(df['rel_time'].iloc[i])

    # 5. Apply your smoothing logic
    predictions_smooth = smooth_predictions(predictions_raw, window_size=10)

    res_df = pd.DataFrame({
        'time': time_tracker,
        'truth': df['truth'].values,
        'pred_raw': predictions_raw,
        'pred_smooth': predictions_smooth,
        'acc_z': df['acc_z_filt'].values  # For plotting the context wave
    })

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # 1. Force all classes to standard Python strings
    unique_classes = sorted(list(set([str(c) for c in label_encoder.classes_] + ['Rest'])))

    # 2. Map colors properly using zip
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_classes))]
    color_map = {cls: col for cls, col in zip(unique_classes, colors)}
    color_map['Rest'] = '#e0e0e0'  # Override 'Rest' to be gray

    # Top Plot: The Signal Context
    ax1.plot(res_df['time'], res_df['acc_z'], color='black', alpha=0.7, linewidth=1, label='Accel Z (Filtered)')
    section_id = metadata.get("sectionId", "Unknown")[:8]
    ax1.set_title(f"Signal Data vs Ground Truth (Section: {section_id}...)")
    ax1.set_ylabel("G-Force")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom Plot: Ribbon Diagram
    # Bottom Plot: Ribbon Diagram
    def draw_ribbons(ax, y_pos, labels, times, title):
        if len(times) < 2: return

        # Convert Pandas Series to Numpy arrays to prevent indexing errors
        times = np.array(times)
        labels = np.array(labels)

        step = times[1] - times[0]
        for t, lab in zip(times, labels):
            rect = mpatches.Rectangle((t - step / 2, y_pos), step, 0.8, color=color_map.get(lab, 'gray'))
            ax.add_patch(rect)

        # Now times[-1] works perfectly to get the last element
        offset = (times[-1] - times[0]) * 0.02
        ax.text(times[0] - offset, y_pos + 0.4, title, va='center', ha='right', fontweight='bold')

    draw_ribbons(ax2, 0.0, res_df['pred_smooth'], res_df['time'], "Smoothed AI")
    draw_ribbons(ax2, 1.0, res_df['pred_raw'], res_df['time'], "Raw AI Output")
    draw_ribbons(ax2, 2.0, res_df['truth'], res_df['time'], "Ground Truth")

    ax2.set_ylim(-0.5, 3.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Relative Time (Seconds)")
    ax2.set_title("Classification Pipeline Results")

    # Global Legend
    handles = [mpatches.Patch(color=color_map[c], label=c) for c in unique_classes]
    fig.legend(handles=handles, loc='lower center', ncol=len(unique_classes), bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    save_name = f"inference_{section_id}.png"
    plt.savefig(save_name, bbox_inches="tight")
    print(f"Saved plot to {save_name}")
    plt.show()

def save_model():
    import joblib

    # 1. Save the scikit-learn Label Encoder
    joblib.dump(dataset.le, "wodbuddy_label_encoder.pkl")
    print("Saved Label Encoder to wodbuddy_label_encoder.pkl")

    # 2. Save the PyTorch State Dict (Weights only - best practice)
    torch.save(model.state_dict(), "wodbuddy_cnn_weights.pth")
    print("Saved PyTorch weights to wodbuddy_cnn_weights.pth")

    # 3. Export to ONNX for the Go Backend
    # ONNX requires a "dummy input" to trace the graph's mathematical operations
    # Our shape is [Batch_Size=1, Channels=6, Sequence_Length=20]
    dummy_input = torch.randn(1, 6, 20, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        "wodbuddy_model.onnx",
        export_params=True,
        opset_version=14,
        input_names=['input_sensors'],
        output_names=['class_logits'],
        dynamic_axes={'input_sensors': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}}
    )
    print("Exported pro")

# --- 5. EXECUTION ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Load Data
    dataset = WodDataset(data_dir="dataset", config=CONFIG)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    num_classes = len(dataset.le.classes_)

    # 2. Initialize Model
    model = SixAxisCNN(num_classes=num_classes).to(device)
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

    save_model()

    # Pick one of your sliced sections to test against
    test_target_dir = "dataset/section_c11b7a06-0606-444f-b598-8343665bc5a5"

    # Make sure to pass model, not model.cpu() if x_tensor is sent to device
    test_and_plot_section(test_target_dir, model, dataset.le, CONFIG)
