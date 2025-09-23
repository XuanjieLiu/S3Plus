from SynthCommon import *


import numpy as np
import random
from tqdm import tqdm

def gen_one_data_from_boxes_seq(boxes, a, b, obj_label, canvas_size):
    c = a + b
    n_idx = np.array(list(range(len(boxes))))
    a_idx = random.sample(list(n_idx), a)
    b_idx = random.sample(list(n_idx), b)
    c_idx = random.sample(list(n_idx), c)
    boxes_a = [boxes[i] for i in a_idx]
    boxes_b = [boxes[i] for i in b_idx]
    boxes_c = [boxes[i] for i in c_idx]

    image_a = draw_objects_on_image(obj_label, boxes_a, canvas_size)
    image_b = draw_objects_on_image(obj_label, boxes_b, canvas_size)
    image_c = draw_objects_on_image(obj_label, boxes_c, canvas_size)
    label = {"obj": obj_label, "a": a, "b": b, "c": c}
    return image_a, image_b, image_c, label


# ----------------------------
# 数据预生成函数：保存数据到本地
# ----------------------------
def pre_generate_oneSeqBox_dataset(num_samples, box_seq, output_dir, canvas_size=(224, 224), is_plot_hist=True):
    """
    预生成 num_samples 个数据点并保存到 output_dir 目录下。
    每个数据点保存在单独的文件夹中，包含 image_a.png, image_b.png, image_c.png 以及 label.json。
    如果某个数据点生成失败（如无法生成足够的 boxes），则跳过，并记录下来。
    """
    num_box = len(box_seq)
    os.makedirs(output_dir, exist_ok=True)
    generated = 0
    sample_index = 0
    seq_a = []
    seq_b = []
    for i in tqdm(range(num_samples)):
        obj_label = random.choice(OBJ_LIST)
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        seq_a.append(a)
        seq_b.append(b)
        assert a + b <= num_box, f"Error: a + b = {a} + {b} = {a+b} > num_box = {num_box}"
        datapoints = recurrent_generate_data(box_seq, a+b, obj_label, canvas_size)
        for datapoint in datapoints:
            image_a, image_b, image_c, label = datapoint
            # 保存到 sample_{index:04d} 文件夹下
            sample_dir = os.path.join(output_dir, f"sample_{sample_index:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            image_a.save(os.path.join(sample_dir, "image_a.png"))
            image_b.save(os.path.join(sample_dir, "image_b.png"))
            image_c.save(os.path.join(sample_dir, "image_c.png"))
            # 保存 label 为 JSON 格式
            with open(os.path.join(sample_dir, "label.json"), "w", encoding="utf-8") as f:
                json.dump(label, f, ensure_ascii=False, indent=2)
            # print(f"Saved sample {sample_index}: {label}")
            generated += 1
            sample_index += 1

    print(f"Pre-generation complete. Generated {generated} samples.")
    if is_plot_hist:
        plot_hist(seq_a, seq_b)


num_samples_train = 8 # 根据需要调整样本数量
num_samples_val = 4
output_dir_train = "new_icon_train"
output_dir_val = "new_icon_val"
num_box = 12
canvas_size = (224, 224)
min_size = 20
max_size = 40

box_seq = generate_non_overlapping_boxes(num_box, canvas_size, min_size, max_size, max_attempts=1000, max_retry=5)
pre_generate_oneSeqBox_dataset(num_samples_train, box_seq, output_dir_train, canvas_size=canvas_size)
pre_generate_oneSeqBox_dataset(num_samples_val, box_seq, output_dir_val, canvas_size=canvas_size, is_plot_hist=False)