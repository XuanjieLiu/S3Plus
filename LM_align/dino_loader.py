import torch
from torchvision import transforms
from timm.models.vision_transformer import VisionTransformer
from shared import DEVICE
import os

# Path to your DINO checkpoint
checkpoint_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/checkpoints/')
checkpoint_name = "dino_deitsmall8_pretrain.pth"
checkpoint_path = os.path.join(checkpoint_root, checkpoint_name)

# Define a function to load the DINO model
def load_dino_vit_s8(checkpoint_path):
    # Load the ViT-S/8 model structure
    model = VisionTransformer(
        img_size=224,  # DINO typically trains on 224x224 images
        patch_size=8,  # S/8 means patch size is 8
        embed_dim=384,  # ViT-S has an embedding dimension of 384
        depth=12,  # Number of transformer blocks
        num_heads=6,  # Number of attention heads
        num_classes=0,  # No classification head
    )

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("teacher", checkpoint)  # DINO checkpoints usually store the teacher's state dict

    # Remove `module.` prefix from keys if it exists
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    # Set the model to evaluation mode
    model.eval()

    return model

# Load your DINO model
model = load_dino_vit_s8(checkpoint_path)
print("DINO model loaded successfully!")

# Define a transform pipeline for your images
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
])


# Assume you have a dataloader named `dataloader`
def evaluate_images(model, dataloader):
    features = []
    with torch.no_grad():  # Disable gradient computation for inference
        for images, _ in dataloader:
            images = images.to("cuda" if torch.cuda.is_available() else "cpu")
            outputs = model(images)  # Extract features
            features.append(outputs.cpu())  # Store features on CPU

    # Concatenate all features into a single tensor
    features = torch.cat(features, dim=0)
    return features