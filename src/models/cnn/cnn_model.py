import torch.nn as nn

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
