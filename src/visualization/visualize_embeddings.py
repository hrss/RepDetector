import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

# Import your existing architecture and dataset loader
from train_cnn import SixAxisCNN, WodDataset, CONFIG


def extract_embeddings(model, dataloader, device):
    """Runs the dataset through the CNN but stops before the final classification."""
    model.eval()

    all_embeddings = []
    all_labels = []

    print("Extracting 128-D embeddings from the CNN...")
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)

            # 1. Run data strictly through the Convolutional/Pooling layers
            # This outputs the rich, learned spatial features (Shape: [Batch, 128, 1])
            features = model.features(batch_x)

            # 2. Flatten the output to 1D arrays [Batch, 128]
            flattened_features = torch.flatten(features, 1)

            all_embeddings.append(flattened_features.cpu().numpy())
            all_labels.append(batch_y.numpy())

    # Concatenate all batches into single massive arrays
    return np.vstack(all_embeddings), np.concatenate(all_labels)


def plot_latent_space():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the Label Encoder and initialize the blank model
    try:
        le = joblib.load("wodbuddy_label_encoder.pkl")
    except FileNotFoundError:
        print("Error: Could not find wodbuddy_label_encoder.pkl. Train the CNN first.")
        return

    num_classes = len(le.classes_)
    model = SixAxisCNN(num_classes=num_classes)

    # 2. Inject your trained weights
    try:
        model.load_state_dict(torch.load("wodbuddy_cnn_weights.pth", map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Error: Could not find wodbuddy_cnn_weights.pth. Train the CNN first.")
        return

    model.to(device)

    # 3. Load the dataset
    # We use a large batch size because we just want to push everything through quickly
    dataset = WodDataset(data_dir="data", config=CONFIG)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # 4. Extract the 128-Dimensional Embeddings
    embeddings_128d, encoded_labels = extract_embeddings(model, dataloader, device)

    # Translate the integer labels back to strings (e.g., 'Sit Ups', 'Rest')
    string_labels = le.inverse_transform(encoded_labels)

    # Filter out 'Rest' class
    mask = string_labels != 'Rest'
    embeddings_128d = embeddings_128d[mask]
    string_labels = string_labels[mask]

    # 5. Dimensionality Reduction (Squash 128D -> 2D)
    print("Running t-SNE to reduce 128 dimensions down to 2D. This may take a minute...")
    # Perplexity controls how to balance local vs global aspects of the data.
    # 30-50 is standard. Lower it if your dataset is very small.
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_128d)

    # 6. Build a DataFrame for easy plotting
    df_plot = pd.DataFrame({
        'TSNE_X': embeddings_2d[:, 0],
        'TSNE_Y': embeddings_2d[:, 1],
        'Movement': string_labels
    })

    # 7. Plotting the Clusters
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # Use seaborn to automatically color-code the different CrossFit movements
    scatter = sns.scatterplot(
        data=df_plot,
        x='TSNE_X',
        y='TSNE_Y',
        hue='Movement',
        palette='tab10',
        alpha=0.6,
        s=40,  # Marker size
        edgecolor=None
    )

    # Label one point per movement directly on the plot
    labeled_movements = set()
    for _, row in df_plot.iterrows():
        movement = row["Movement"]
        if movement in labeled_movements:
            continue

        plt.annotate(
            movement,
            (row["TSNE_X"], row["TSNE_Y"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
        labeled_movements.add(movement)

    plt.title("WodBuddy Latent Space: t-SNE Projection of CNN Embeddings", fontweight='bold', fontsize=14)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")

    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    plt.savefig("latent_space_map.png", dpi=300)
    print("Saved 2D mapping to latent_space_map.png")
    plt.show()

if __name__ == "__main__":
    plot_latent_space()