import torch
import torch.utils.data
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from shared import DEVICE


def load_enc_eval_data(loader, encode_func):
    num_labels = []
    num_z = None
    for batch_ndx, sample in enumerate(loader):
        data, labels = sample
        data = data.to(DEVICE)
        num = [int(label.split('-')[0]) for label in labels]
        z = encode_func(data)
        num_labels.extend(num)
        if num_z is None:
            num_z = z
        else:
            num_z = torch.cat((num_z, z), dim=0)
    return num_z, num_labels


def load_enc_eval_data_with_style(loader, encode_func):
    num_labels = []
    colors = []
    shapes = []
    num_z = None
    for batch_ndx, sample in enumerate(loader):
        data, labels = sample
        data = data.to(DEVICE)
        num = [int(label.split('-')[0]) for label in labels]
        shape = [label.split('-')[1] for label in labels]
        color = [label.split('-')[2].split('.')[0] for label in labels]
        z = encode_func(data)
        num_labels.extend(num)
        colors.extend(color)
        shapes.extend(shape)
        if num_z is None:
            num_z = z
        else:
            num_z = torch.cat((num_z, z), dim=0)
    return num_z, num_labels, colors, shapes


class SingleImgDataset(torch.utils.data.Dataset):
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
        img = Image.open(data_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, data_name


if __name__ == "__main__":
    dataset = SingleImgDataset("dataset/(1,16)-8-['circle', 'triangle_down', 'star', 'thin_diamond']")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    aaa = []
    for batch_ndx, sample in enumerate(loader):
        aaa.append(sample[1])
        if batch_ndx > 20:
            break
    print(aaa)
    print('break')
