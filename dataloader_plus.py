import os
import random
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from shared import *

# class Dataset(torch.utils.data.Dataset):
#     def __init__(
#             self, dataset_path,
#             cache_all=True,
#     ):
#         super().__init__()
#         self.dataset_path = dataset_path
#         self.f_list = os.listdir(dataset_path)
#         self.transform = transforms.Compose([transforms.ToTensor()])
#         self.data_list = []
#         self.cache_all = cache_all
#         if cache_all:
#             self.cacheAll()
#
#     def cacheAll(self):
#         for name in self.f_list:
#             data = self.read_a_data_from_disk(name)
#             self.data_list.append(data)
#
#     def __len__(self):
#         return len(self.f_list)
#
#     def __getitem__(self, index):
#         if self.cache_all:
#             return self.data_list[index]
#         else:
#             return self.read_a_data_from_disk(self.f_list[index])
#
#     def read_a_data_from_disk(self, data_name):
#         data_path = os.path.join(self.dataset_path, data_name)
#         img_list = os.listdir(data_path)
#         img_list.sort()
#         img_tensors = [self.transform(Image.open(os.path.join(data_path, name)).convert('RGB')).to(DEVICE) for name in img_list]
#         return img_tensors, img_list

class MultiImgDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        cache_all=True,
        transform=None,                 # 传入一个或多个transform（Compose）
        augment_times=1                 # 每个样本重复几次（不同随机transform）
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.f_list = sorted(os.listdir(dataset_path))
        self.cache_all = cache_all
        self.augment_times = augment_times

        self.transform = transform

        # 缓存原始图像数据
        self.data_list = []
        if cache_all:
            self.cacheAll()

    def cacheAll(self):
        for name in self.f_list:
            data = self.read_a_data_from_disk(name)
            self.data_list.append(data)

    def __len__(self):
        return len(self.f_list) * self.augment_times

    def __getitem__(self, index):
        base_index = index % len(self.f_list)

        if self.cache_all:
            # 从缓存中读取原始图像 tensor，再应用 transform
            raw_imgs, img_names = self.data_list[base_index]
            if self.transform is None:
                # 如果没有 transform，则直接返回原始图像 tensor
                return raw_imgs, img_names
            transformed_imgs = [
                self.transform(img_tensor) for img_tensor in raw_imgs
            ]
            return transformed_imgs, img_names
        else:
            return self.read_a_data_from_disk(self.f_list[base_index], apply_transform=True)

    def read_a_data_from_disk(self, data_name, apply_transform=False):
        data_path = os.path.join(self.dataset_path, data_name)
        img_list = sorted(os.listdir(data_path))

        img_tensors = []
        for name in img_list:
            img = Image.open(os.path.join(data_path, name)).convert('RGB')
            img_tensor = transforms.ToTensor()(img).to(DEVICE)
            if apply_transform and self.transform is not None:
                img_tensor = self.transform(img_tensor)
            img_tensors.append(img_tensor)

        return img_tensors, img_list


if __name__ == "__main__":
    dataset = MultiImgDataset("dataset/PlusPair-(1,8)-FixedPos/train")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    aaa = []
    for batch_ndx, sample in enumerate(loader):
        aaa.append(sample[1])
        if batch_ndx > 20:
            break
    print(aaa)
    print('break')
