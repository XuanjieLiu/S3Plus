import torch
import torch.nn as nn


class MelCNNPitchClassifier(nn.Module):
    def __init__(self, n_mels=128, n_frames=32, n_class=12):
        super().__init__()
        if n_mels == 128:
            n_mels = 64  # deliberately ignore higher frequency bands
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (5, 3), padding=(2, 1)),  # (batch, 1, n_mels, n_frames)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, (5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveMaxPool2d((1, n_mels)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * n_mels, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_class),
        )

    def forward(self, x):
        # x: (batch, n_mels, n_frames)
        x = x.unsqueeze(1)  # (batch, 1, n_mels, n_frames)
        # deliberately ignore higher freq
        x = x[:, :, :64, :]

        x = self.cnn(x)
        x = self.fc(x)

        return x


# TODO: validate the model with a simple test case
if __name__ == "__main__":
    model = MelCNNPitchClassifier()
    x = torch.randn(8, 128, 32)  # batch size of 8, 128 mel bands, 32 frames
    output = model(x)
    print(output.shape)  # should be (8, 12) for 12 classes
