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
        self,
        dataset_path,
        transform=None,              # 支持自定义 transforms.Compose([...])
        cache_all=True,
        augment_times=1              # 每张图片重复采样多少次（使用不同随机 transform）
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.f_list = sorted(os.listdir(dataset_path))
        self.cache_all = cache_all
        self.augment_times = augment_times

        # 使用默认 transform，如果没有传入
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.data_list = []

        if cache_all:
            self.cacheAll()

    def cacheAll(self):
        for name in self.f_list:
            data = self.read_a_data_from_disk(name, apply_transform=False)
            self.data_list.append(data)

    def __len__(self):
        return len(self.f_list) * self.augment_times

    def __getitem__(self, index):
        base_index = index % len(self.f_list)

        if self.cache_all:
            # 从缓存读取 tensor，再施加 transform
            raw_tensor, name = self.data_list[base_index]
            img = transforms.ToPILImage()(raw_tensor)
            transformed_tensor = self.transform(img)
            return transformed_tensor, name
        else:
            return self.read_a_data_from_disk(self.f_list[base_index], apply_transform=True)

    def read_a_data_from_disk(self, data_name, apply_transform=True):
        data_path = os.path.join(self.dataset_path, data_name)
        img = Image.open(data_path).convert('RGB')

        if apply_transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

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
