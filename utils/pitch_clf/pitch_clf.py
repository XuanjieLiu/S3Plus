import torch
import torch.nn as nn


class MelCNNPitchClassifier(nn.Module):
    def __init__(self, n_mels=128, n_frames=10, n_class=12):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3, 5), padding=(1, 2)),  # (batch, 1, n_frames, n_mels)
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, n_mels)),  # 聚合时域
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * n_mels, 128),
            nn.ReLU(),
            nn.Linear(128, n_class),
        )

    def forward(self, x):
        # x: (batch, n_frames, n_mels)
        x = x.unsqueeze(1)  # (batch, 1, n_frames, n_mels)
        x = self.cnn(x)
        x = self.fc(x)
        return x


# TODO: validate the model with a simple test case
