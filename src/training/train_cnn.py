import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.onnx

from src.training.data_loader import WodDataset, load_raw_section_data
from src.models.cnn_model import SixAxisCNN
from src.core.data_utils import smooth_predictions
from src.visualization.inference_viz import plot_classification_results

# --- CONFIGURATION ---
CONFIG = {
    'sample_rate': 20,
    'window_size_sec': 2.5,
    'step_size_sec': 0.5,
    'lowpass_cutoff': 3.0,
    'filter_order': 4,
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001
}




# --- 4. TEST AND VISUALIZE INFERENCE ---
def test_and_plot_section(section_dir, model, label_encoder, config):
    """Runs inference on a pre-sliced section directory and plots the results."""
    print(f"\nTesting Inference on Section: {os.path.basename(section_dir)}")

    df, metadata = load_raw_section_data(section_dir, config)
    if df is None:
        print(f"Error: Missing files in {section_dir}")
        return

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
    sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
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

    plot_classification_results(res_df, metadata, label_encoder, f"inference_{metadata.get('sectionId', 'unknown')[:8]}.png")

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
    dataset = WodDataset(data_dir="data", config=CONFIG)
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
    test_target_dir = "data/1797/section_3109"

    # Make sure to pass model, not model.cpu() if x_tensor is sent to device
    test_and_plot_section(test_target_dir, model, dataset.le, CONFIG)
