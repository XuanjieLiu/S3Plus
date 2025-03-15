import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import random


DATA_ROOT = '{}{}'.format(
        os.path.dirname(os.path.abspath(__file__)), 
        '/../fruit_recognition_dataset/chrisfilo/fruit-recognition/versions/1'
    )

SUBSET_LIST = [
    'Apple/Apple A',
    'Apple/Apple D',
]


# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert image to Tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])


# Define the custom dataset
class FruitRecognitionDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, subset_list=None, transform=transform):
        if subset_list is None:
            subset_list = SUBSET_LIST
        self.data_root = data_root
        self.subset_list = subset_list
        self.transform = transform
        self.data = []

        for subset in subset_list:
            subset_path = os.path.join(data_root, subset)
            for img_file in os.listdir(subset_path):
                img_path = os.path.join(subset_path, img_file)
                self.data.append((img_path, subset))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
    

def visulize_one_img(img):
    img = img.permute(1, 2, 0)  # Change the order of dimensions for plotting
    # img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Unnormalize
    img = img.numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def random_mask_batch(batch_images):
    mask_1 = torch.ones_like(batch_images)
    mask_2 = torch.ones_like(batch_images)
    batch_size, _, height, width = batch_images.shape
    split_ratio = random.uniform(0.2, 0.8)
    
    is_vertical_split = random.random() > 0.5
    if is_vertical_split:
        mask_1[:, :, :, :int(width * split_ratio)] = 0
        mask_2[:, :, :, int(width * split_ratio):] = 0
    else:
        mask_1[:, :, :int(height * split_ratio), :] = 0
        mask_2[:, :, int(height * split_ratio):, :] = 0

    masked_images_1 = batch_images * mask_1
    masked_images_2 = batch_images * mask_2
    return masked_images_1, masked_images_2


if __name__ == "__main__":
    # Create the dataset
    dataset = FruitRecognitionDataset(DATA_ROOT, SUBSET_LIST, transform=transform)
    print("Number of samples in the dataset:", len(dataset))

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate over the DataLoader
    for images, labels in dataloader:
        print(images.shape, labels)
        masked_1, masked_2 = random_mask_batch(images)
        # Visualize one image
        visulize_one_img(images[0])
        visulize_one_img(masked_1[0])
        visulize_one_img(masked_2[0])
        break