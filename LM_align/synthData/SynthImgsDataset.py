
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import random
from synthData.SynthCommon_win import TriplePlan, draw_objects_on_image, gen_recurrent_data, OBJ_LIST, OBJ_LIST_2, gen_recurrent_plan
from typing import List, Tuple
from pathlib import Path
from torch.utils.data import DataLoader

# ----------------------------
# Dataset: 从本地预生成的数据读取
# ----------------------------
class PreGeneratedDataset(Dataset):
    """
    从预生成的数据文件夹中加载数据，每个样本包含 image_a, image_b, image_c 以及 label。
    文件夹结构示例：
      output_dir/
         sample_0000/
             image_a.png
             image_b.png
             image_c.png
             label.json
         sample_0001/
             ...
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # 每个子文件夹即为一个样本
        self.sample_dirs = sorted([os.path.join(root_dir, d)
                                   for d in os.listdir(root_dir)
                                   if os.path.isdir(os.path.join(root_dir, d))])
        self.transform = transform

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        image_a = Image.open(os.path.join(sample_dir, "image_a.png")).convert("RGB")
        image_b = Image.open(os.path.join(sample_dir, "image_b.png")).convert("RGB")
        image_c = Image.open(os.path.join(sample_dir, "image_c.png")).convert("RGB")
        with open(os.path.join(sample_dir, "label.json"), "r", encoding="utf-8") as f:
            label = json.load(f)
        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
            image_c = self.transform(image_c)
        return {"image_a": image_a, "image_b": image_b, "image_c": image_c, "label": label}
    

class onlineGenDataset(Dataset):
    def __init__(self, num_samples, reuse_times, transform=None, obj_list=OBJ_LIST, auto_update=True):
        """
        Parameters
        ----------
        num_samples : int
            生成的样本数量
        reuse_times : int
            每个样本重复的次数
        """
        self.num_samples = num_samples
        self.reuse_times = reuse_times
        self.chunk = gen_recurrent_data(num_samples, obj_list=obj_list)
        self.data_size = num_samples * reuse_times
        self.transform = transform
        self.obj_list = obj_list
        self.auto_update = auto_update

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        if index + 1 == self.data_size and self.auto_update:
            self.chunk = gen_recurrent_data(self.num_samples, obj_list=self.obj_list)
        if index % self.num_samples == 0:
            random.shuffle(self.chunk)
        image_a, image_b, image_c, label = self.chunk[index % self.num_samples]
        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
            image_c = self.transform(image_c)
        return {"image_a": image_a, "image_b": image_b, "image_c": image_c, "label": label}
    


# ----------------------------用 plan + obj_list 渲染出 Dataset（与原字段保持一致）----------------------------
class PlanRenderDataset(Dataset):
    """给定同一份 plan，用不同 obj_list 渲染出两套数据。"""
    def __init__(self, plans: List[TriplePlan], obj_list: List[str],
                 transform=None, canvas_size=(224,224)):
        assert len(obj_list) > 0
        self.plans = plans
        self.obj_list = obj_list
        self.transform = transform
        self.canvas_size = canvas_size

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, idx):
        plan = self.plans[idx]
        # 为了可重复且平均分配，不依赖随机数：循环使用 obj_list
        obj_label = self.obj_list[idx % len(self.obj_list)]

        img_a = draw_objects_on_image(obj_label, plan.boxes_a, self.canvas_size)
        img_b = draw_objects_on_image(obj_label, plan.boxes_b, self.canvas_size)
        img_c = draw_objects_on_image(obj_label, plan.boxes_c, self.canvas_size)

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
            img_c = self.transform(img_c)

        label = {"obj": obj_label, "a": plan.a, "b": plan.b, "c": plan.c}
        return {"image_a": img_a, "image_b": img_b, "image_c": img_c, "label": label}
#----------------------------用 plan + obj_list 渲染出 Dataset（与原字段保持一致）----------------------------

def export_dataloader_to_dir(dataloader, dataset_dir: str):
    """
    将 dataloader 中的所有 (image_a, image_b, image_c, label) 数据
    导出为文件结构化的数据集。

    输出结构:
      dataset_dir/
        00000/
          image_a.png
          image_b.png
          image_c.png
          label.json
        00001/
          image_a.png
          image_b.png
          image_c.png
          label.json
        ...

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        含有键 "image_a"、"image_b"、"image_c"、"label" 的 batch。
    dataset_dir : str
        输出根目录路径。
    """
    os.makedirs(dataset_dir, exist_ok=True)

    sample_idx = 0
    for batch in dataloader:
        # batch 可能是张量批，也可能是 list[dict]
        batch_size = None

        # PyTorch 默认 collate，把同名字段堆叠为张量
        batch_size = batch["image_a"].size(0)
        for i in range(batch_size):
            # 单个样本
            image_a = batch["image_a"][i]
            image_b = batch["image_b"][i]
            image_c = batch["image_c"][i]
            label = {k: (v[i].item() if hasattr(v, "size") else v[i])
                     for k, v in batch["label"].items()}

            _save_sample(dataset_dir, sample_idx, image_a, image_b, image_c, label)
            sample_idx += 1

    print(f"✅ Export complete. {sample_idx} samples saved to {dataset_dir}.")


def _save_sample(root_dir: str, index: int, image_a, image_b, image_c, label: dict):
    """
    保存单个样本到 root_dir/index 下。
    image_a/b/c 可能是 tensor 或 PIL.Image。
    """
    import torch
    from torchvision.transforms.functional import to_pil_image

    sample_dir = Path(root_dir) / f"{index:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    def _save_image(img, name):
        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)
        img.save(sample_dir / name)

    _save_image(image_a, "image_a.png")
    _save_image(image_b, "image_b.png")
    _save_image(image_c, "image_c.png")

    with open(sample_dir / "label.json", "w", encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False, indent=2)



# ----------------------------一次性构建两个随机的 val / ood EVAL DataLoader----------------------------
def init_test_online_synth_dataloaders(num_samples):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    val_dataset = onlineGenDataset(num_samples=num_samples, reuse_times=1, transform=transform, auto_update=False)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    ood_dataset = onlineGenDataset(num_samples=num_samples, reuse_times=1, transform=transform, obj_list=OBJ_LIST_2, auto_update=False)
    ood_dataloader = DataLoader(ood_dataset, batch_size=128, shuffle=False)

    return val_dataloader, ood_dataloader


# ----------------------------一次性构建“成对”的 val / ood EVAL DataLoader（完全同分布）----------------------------
def init_test_online_synth_dataloaders_paired(num_samples,
                                              obj_list_val=OBJ_LIST,
                                              obj_list_ood=OBJ_LIST_2,
                                              canvas_size=(224,224)):
    """
    产生两套 dataloader：完全相同的 (a,b,c) 与 盒子分布/顺序，
    仅 icon 来自不同 obj_list。
    """
    transform = transforms.Compose([
        transforms.Resize(canvas_size),
        transforms.ToTensor()
    ])

    # 共享同一份 plan
    plans = gen_recurrent_plan(num_samples, canvas_size=canvas_size)

    val_dataset = PlanRenderDataset(plans, obj_list=obj_list_val,
                                    transform=transform, canvas_size=canvas_size)
    ood_dataset = PlanRenderDataset(plans, obj_list=obj_list_ood,
                                    transform=transform, canvas_size=canvas_size)

    # 注意：为了严格一一对应，建议 shuffle=False
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    ood_loader = DataLoader(ood_dataset, batch_size=128, shuffle=False)
    return val_loader, ood_loader
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    n_samples = 16
    dataset_dir_1 = "/l/users/xuanjie.liu/S3Plus/LM_align/synthData/test_export_dataset_1"
    dataset_dir_2 = "/l/users/xuanjie.liu/S3Plus/LM_align/synthData/test_export_dataset_2"
    os.makedirs(dataset_dir_1, exist_ok=True)
    os.makedirs(dataset_dir_2, exist_ok=True)
    val_loader_1, ood_loader_1 = init_test_online_synth_dataloaders(n_samples)
    val_loader_2, ood_loader_2 = init_test_online_synth_dataloaders_paired(n_samples)
    export_dataloader_to_dir(val_loader_1, os.path.join(dataset_dir_1, "val"))
    export_dataloader_to_dir(ood_loader_1, os.path.join(dataset_dir_1, "ood"))
    export_dataloader_to_dir(val_loader_2, os.path.join(dataset_dir_2, "val"))
    export_dataloader_to_dir(ood_loader_2, os.path.join(dataset_dir_2, "ood"))