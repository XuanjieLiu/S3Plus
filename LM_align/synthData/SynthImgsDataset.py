
from matplotlib import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import random
from LM_align.synthData.SynthCommon import TriplePlan, draw_objects_on_image, gen_recurrent_data, OBJ_LIST, OBJ_LIST_2, gen_recurrent_plan
from typing import List, Tuple

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
