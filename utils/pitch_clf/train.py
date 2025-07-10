import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


import dataloader.insnotes_dataloader as dataloader_module
from utils.pitch_clf.pitch_clf import MelCNNPitchClassifier


S_LIST = dataloader_module.S_LIST
C_LIST = dataloader_module.C_LIST

val_data_dir = "../data/insnotes_major_val"

train_loader = dataloader_module.get_dataloader(batch_size=128, num_workers=4)
val_loader = dataloader_module.get_dataloader(
    batch_size=128, num_workers=4, test=True, data_dir=val_data_dir
)

model = MelCNNPitchClassifier(n_mels=128, n_frames=32, n_class=len(C_LIST))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

steps = 10000

for step in tqdm(range(steps)):
    model.train()
    for batch in train_loader:
        audio, contents, styles = batch
        audio = audio.to(device)
        contents = contents.to(device)
        styles = styles.to(device)

        # Forward pass
        outputs = model(audio.squeeze())
        loss = criterion(outputs, contents)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            for val_batch in val_loader:
                val_audio, val_contents, _ = val_batch
                val_audio = val_audio.to(device)
                val_contents = val_contents.to(device)

                val_outputs = model(val_audio)
                val_loss += criterion(val_outputs, val_contents).item()

                _, predicted = torch.max(val_outputs, 1)
                val_acc += (predicted == val_contents).sum().item()

            print(f"Validation Loss at Step {step}: {val_loss / len(val_loader)}")
            print(
                f"Validation Accuracy at Step {step}: {val_acc / len(val_loader.dataset)}"
            )

# Save the model
model_save_path = "mel_cnn_pitch_classifier.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# TODO: finish and test this
