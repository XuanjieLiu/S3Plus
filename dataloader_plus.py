import torch
import torch.utils.data
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from shared import *

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, dataset_path,
            cache_all=True,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.f_list = os.listdir(dataset_path)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.data_list = []
        self.cache_all = cache_all
        if cache_all:
            self.cacheAll()

    def cacheAll(self):
        for name in self.f_list:
            data = self.read_a_data_from_disk(name)
            self.data_list.append(data)

    def __len__(self):
        return len(self.f_list)

    def __getitem__(self, index):
        if self.cache_all:
            return self.data_list[index]
        else:
            return self.read_a_data_from_disk(self.f_list[index])

    def read_a_data_from_disk(self, data_name):
        data_path = os.path.join(self.dataset_path, data_name)
        img_list = os.listdir(data_path)
        img_list.sort()
        img_tensors = [self.transform(Image.open(os.path.join(data_path, name)).convert('RGB')).to(DEVICE) for name in img_list]
        return img_tensors, img_list


if __name__ == "__main__":
    dataset = Dataset("dataset/PlusPair-(1,8)-FixedPos/train")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    aaa = []
    for batch_ndx, sample in enumerate(loader):
        aaa.append(sample[1])
        if batch_ndx > 20:
            break
    print(aaa)
    print('break')
