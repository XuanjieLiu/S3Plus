from torchvision.datasets import VOCDetection
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from PIL import Image
from collections import defaultdict

from tqdm import tqdm


def load_Pascal_VOC():
    # 数据路径
    data_dir = './VOCdevkit/'
    year = '2012'  # VOC2007 或 VOC2012
    image_set = 'train'  # 'train', 'val', 'test'

    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 加载数据集
    dataset = VOCDetection(
        root=data_dir,
        year=year,
        image_set=image_set,
        # transform=transform,  # 图像预处理
        target_transform=None,  # 可选：标注转换
        # download=True  # 如果需要下载数据集，设为 True
    )
    return dataset

def statistic_one_object_img(dataset):
    single_class_images = []  # 保存只包含一种类别的图片
    object_counts = defaultdict(list)  # 保存每类图片的目标数量

    # 遍历数据集
    for idx in tqdm(range(len(dataset))):
        _, target = dataset[idx]
        objects = target['annotation']['object']

        # 获取类别
        if not isinstance(objects, list):
            # 如果只有一个目标，转为列表
            objects = [objects]
        classes = [obj['name'] for obj in objects]

        # 判断是否是单类目标
        unique_classes = set(classes)
        if len(unique_classes) == 1:
            single_class_images.append(idx)
            object_counts[next(iter(unique_classes))].append(len(classes))  # 保存目标个数

    # 统计结果
    total_single_class_images = len(single_class_images)
    print(f"Number of images with only one object class: {total_single_class_images}")
    for cls, counts in object_counts.items():
        counts_count = defaultdict(int)
        for count in counts:
            counts_count[count] += 1
        print(f"Class '{cls}' has {len(counts)} images, object counts: {counts[:10]} (showing first 10 counts)")
        print(f"Object counts distribution: {counts_count}")

    # Total number of images with only one object class
    total_one_object_images = len(single_class_images)
    print(f"Number of images with only one object class: {total_one_object_images}")


def statistic_img_sizes(dataset):
    sizes = []  # 保存图片尺寸

    # 遍历数据集
    for idx in tqdm(range(len(dataset))):
        image, _ = dataset[idx]
        sizes.append(image.size)

    # 统计结果
    sizes = set(sizes)
    print(f"Number of unique image sizes: {len(sizes)}")
    print(f"Unique image sizes: {sizes}")


if __name__ == "__main__":

    # 加载数据集
    dataset = load_Pascal_VOC()
    statistic_one_object_img(dataset)

    # # DataLoader
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

