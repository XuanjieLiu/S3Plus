import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
plt.rcParams['font.family'] = ['Noto Color Emoji', 'DejaVu Sans']
# ----------------------------
# 全局配置与 Emoji 映射
# ----------------------------
OBJ_LIST = ['white_king', 'white_rook', 'white_knight', 'sun', 'cloud',
            'telephone', 'crossed_swords', 'radioactive', 'heart', 'skull']

OBJ_LIST_2 = ['white_bishop', 'white_pawn', 'flag', 'sparkle', 'umbrella']
# OBJ_LIST_2 = ['umbrella']


EMOJI_MAP = {
    # ♔–♟ Chess symbols
    "white_king": "♔",
    "white_queen": "♕",
    "white_rook": "♖",
    "white_bishop": "♗",
    "white_knight": "♘",
    "white_pawn": "♙",
    "black_king": "♚",
    "black_queen": "♛",
    "black_rook": "♜",
    "black_bishop": "♝",
    "black_knight": "♞",
    "black_pawn": "♟",

    # ♠ ♥ ♦ ♣ Playing card suits
    "spade": "♠",
    "heart": "♥",
    "diamond": "♦",
    "club": "♣",

    # Misc symbols (U+2600–U+26FF)
    "sun": "☀",
    "cloud": "☁",
    "umbrella": "☂",
    "snowman": "☃",
    "comet": "☄",
    "telephone": "☎",
    "skull": "☠",
    "radioactive": "☢",
    "biohazard": "☣",
    "flag": "⚑",
    "crossed_swords": "⚔",
    "gear": "⚙",
    "scales": "⚖",
    "church": "⛪",
    "fountain": "⛲",
    "tent": "⛺",
    "sailboat": "⛵",

    # Dingbats (U+2700–U+27BF)
    "scissors": "✂",
    "envelope": "✉",
    "pencil": "✏",
    "heavy_check_mark": "✔",
    "cross_mark": "✘",
    "star": "✦",
    "sparkle": "✧",
    "flower1": "✿",
    "flower2": "❀",
}


# ----------------------------
# 工具函数：判断两个矩形是否重叠
# ----------------------------
def boxes_overlap(box1, box2):
    """
    判断两个矩形是否有重叠。box 格式：(x, y, w, h)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    return True


# ----------------------------
# 功能函数：生成不重叠的随机矩形框列表
# ----------------------------
def generate_non_overlapping_boxes(num_boxes, canvas_size, min_size, max_size,
                                   max_attempts=1000, max_retry=5):
    """
    在给定画布上生成 num_boxes 个不重叠的随机矩形框。
    如果在 max_attempts 内未生成，则重试 max_retry 次；
    如果仍然不成功，则返回 None，表示跳过本次数据点生成。
    每个 box 格式为 (x, y, w, h)
    """
    canvas_w, canvas_h = canvas_size
    for retry in range(max_retry):
        boxes = []
        attempts = 0
        while len(boxes) < num_boxes and attempts < max_attempts:
            w = random.randint(min_size, max_size)
            h = random.randint(min_size, max_size)
            x = random.randint(0, canvas_w - w)
            y = random.randint(0, canvas_h - h)
            new_box = (x, y, w, h)
            if any(boxes_overlap(new_box, box) for box in boxes):
                attempts += 1
                continue
            boxes.append(new_box)
        if len(boxes) == num_boxes:
            return boxes
        else:
            print(f"Retry {retry + 1}/{max_retry}: only generated {len(boxes)} boxes out of {num_boxes}. Retrying.")
    # 重试结束后仍未成功
    print(f"Skipping data point: failed to generate {num_boxes} non-overlapping boxes after {max_retry} retries.")
    return None

def _load_font(font_size: int):
    """
    1) 优先用用户级 Noto Color Emoji（~/.local/share/fonts）
    2) 退回系统常见路径
    3) 再退回 DejaVu Sans（Linux 常见的单色大覆盖字体）
    4) 最后退回 PIL 默认字体（一定不报错）
    """
    candidates = [
        os.getenv("EMOJI_FONT_PATH"),  # 允许你通过环境变量指定：export EMOJI_FONT_PATH=~/.local/share/fonts/NotoColorEmoji.ttf
        os.path.expanduser("~/.local/share/fonts/NotoColorEmoji.ttf"),
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/NotoColorEmoji/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if not p:
            continue
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size=font_size)
        except Exception:
            pass
    # 某些机器可直接用族名（fontconfig 能解析时）
    for family in ["Noto Color Emoji", "DejaVu Sans"]:
        try:
            return ImageFont.truetype(family, size=font_size)
        except Exception:
            pass
    # 兜底：永不报错
    return ImageFont.load_default()

# ----------------------------
# 功能函数：在图像上绘制 emoji 表示的对象
# ----------------------------
def draw_object(draw, obj_label, box):
    """
    在给定的 ImageDraw 对象上绘制一个 emoji 表示的对象。
    参数：
      - obj_label: 对象名称（例如 'dog'）
      - box: (x, y, w, h) 指定放置位置和尺寸
    """
    # 获取对应的 emoji 字符，若没有则直接使用对象名称
    emoji_char = EMOJI_MAP.get(obj_label, obj_label)
    x, y, w, h = box
    # 根据 box 尺寸设置字体大小，这里取最小边长作为字体大小
    font_size = int(min(w, h))
    try:
        font_size = int(max(10, min(w, h)))
        emoji_font = _load_font(font_size)
    except Exception as e:
        print("Warning: load seguiemj.ttf failed, use default font. Error:", e)
        emoji_font = ImageFont.load_default()

    # 使用 textbbox 计算 emoji 文本的尺寸，便于居中
    bbox = draw.textbbox((0, 0), emoji_char, font=emoji_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = x + (w - text_w) / 2
    text_y = y + (h - text_h) / 2
    draw.text((text_x, text_y), emoji_char, font=emoji_font, fill="black")


# ----------------------------
# 功能函数：根据给定框列表生成一张图像
# ----------------------------
def draw_objects_on_image(obj_label, boxes, canvas_size, bg_color="white"):
    """
    在指定画布上绘制多个对象（使用 emoji），返回一张 PIL Image。
    """
    image = Image.new("RGB", canvas_size, color=bg_color)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw_object(draw, obj_label, box)
    return image


# ----------------------------
# 功能函数：生成一个数据点（三张图像：图a, 图b, 图c）
# ----------------------------
def generate_datapoint(obj_label, canvas_size=(256, 256), min_size=20, max_size=40,
                       max_attempts=1000, max_retry=5):
    """
    生成一个数据点，包含 3 张图像：
      - 图c：包含 c 个 obj 的图像（c 在 1~20 随机选择）
      - 随机将 c 拆分为 a + b（0 <= a <= c, b = c - a）
      - 图a：从图c中删除 b 个对象，仅保留 a 个
      - 图b：从图c中删除 a 个对象，仅保留 b 个
    Label 中返回 {"obj": obj, "a": a, "b": b, "c": c}。
    如果生成 boxes 失败，则返回 None 表示跳过本次数据点。
    """
    c = random.randint(1, 20)
    # c = 1
    a = random.randint(0, c)
    b = c - a

    boxes_c = generate_non_overlapping_boxes(c, canvas_size, min_size, max_size,
                                             max_attempts=max_attempts, max_retry=max_retry)
    if boxes_c is None:
        # 返回 None，表示本次数据点跳过
        return None

    image_c = draw_objects_on_image(obj_label, boxes_c, canvas_size)

    indices = list(range(c))
    random.shuffle(indices)
    indices_a = sorted(indices[:a])
    indices_b = sorted(indices[a:])
    boxes_a = [boxes_c[i] for i in indices_a]
    boxes_b = [boxes_c[i] for i in indices_b]

    image_a = draw_objects_on_image(obj_label, boxes_a, canvas_size)
    image_b = draw_objects_on_image(obj_label, boxes_b, canvas_size)

    label = {"obj": obj_label, "a": a, "b": b, "c": c}
    return image_a, image_b, image_c, label


# ----------------------------
# 数据预生成函数：保存数据到本地
# ----------------------------
def pre_generate_dataset(num_samples, output_dir,
                         canvas_size=(256, 256), min_size=20, max_size=40,
                         max_attempts=1000, max_retry=5):
    """
    预生成 num_samples 个数据点并保存到 output_dir 目录下。
    每个数据点保存在单独的文件夹中，包含 image_a.png, image_b.png, image_c.png 以及 label.json。
    如果某个数据点生成失败（如无法生成足够的 boxes），则跳过，并记录下来。
    """
    os.makedirs(output_dir, exist_ok=True)
    skipped = []  # 用于记录跳过的数据点（记录样本索引及 c 值）
    generated = 0
    sample_index = 0

    while generated < num_samples:
        obj_label = random.choice(OBJ_LIST)
        datapoint = generate_datapoint(obj_label, canvas_size, min_size, max_size,
                                       max_attempts=max_attempts, max_retry=max_retry)
        if datapoint is None:
            # 记录跳过情况，记录下本次尝试的 c 值
            # 这里我们用 -1 表示该数据点因 boxes 生成失败而跳过
            skipped.append({"sample_index": sample_index, "reason": "box generation failed"})
            sample_index += 1
            continue

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
        print(f"Saved sample {sample_index}: {label}")
        generated += 1
        sample_index += 1

    print(f"Pre-generation complete. Generated {generated} samples.")
    if skipped:
        print("Skipped samples info:")
        for info in skipped:
            print(info)
    return skipped


def plot_hist(X, Y):
    Z = [x + y for x, y in zip(X, Y)]
    All = np.concatenate([X, Y, Z])
    # 重新绘制直方图，确保 x 轴刻度为所有整数
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    ele_X = sorted(list(set(X)))
    plt.hist(X, bins=np.arange(ele_X[0] - 0.5, ele_X[-1] + 0.6, 1), alpha=0.7, edgecolor='black', density=True)
    plt.xticks(ele_X)  # 设置 x 轴刻度为所有整数
    plt.xlabel("A")
    plt.ylabel("Density")
    plt.title("Distribution of A")

    plt.subplot(2, 2, 2)
    ele_Y = sorted(list(set(Y)))
    plt.hist(Y, bins=np.arange(ele_Y[0] - 0.5, ele_Y[-1] + 0.6, 1), alpha=0.7, edgecolor='black', density=True)
    plt.xticks(ele_Y)  # 设置 x 轴刻度为所有整数
    plt.xlabel("B")
    plt.ylabel("Density")
    plt.title("Distribution of B")

    plt.subplot(2, 2, 3)
    ele_Z = sorted(list(set(Z)))
    plt.hist(Z, bins=np.arange(ele_Z[0] - 0.5, ele_Z[-1] + 0.6, 1), alpha=0.7, edgecolor='black', density=True)
    plt.xticks(ele_Z)  # 设置 x 轴刻度为所有整数
    plt.xlabel("C")
    plt.ylabel("Density")
    plt.title("Distribution of C")

    plt.subplot(2, 2, 4)
    ele_All = sorted(list(set(All)))
    plt.hist(All, bins=np.arange(ele_All[0] - 0.5, ele_All[-1] + 0.6, 1), alpha=0.7, edgecolor='black', density=True)
    plt.xticks(ele_All)  # 设置 x 轴刻度为所有整数
    plt.xlabel("All")
    plt.ylabel("Density")
    plt.title("Distribution of All")

    plt.tight_layout()
    plt.show()


"""
Recurrent Dataset Generation
"""


def recurrent_generate_data(boxes, c, obj_label, canvas_size, max_iter=1000):
    data = []
    max_num = c
    rest_iter = max_iter
    box_idx = list(range(len(boxes)))
    while rest_iter > 0 and max_num >= 2:
        a = random.randint(1, max_num - 1)
        b = max_num - a
        if b < 0:
            print(f"Error: a = {a}, b = {b}, max_num = {max_num}")
            break
        c_idx = random.sample(box_idx, max_num)
        a_idx = random.sample(c_idx, a)
        b_idx = [i for i in c_idx if i not in a_idx]
        boxes_a = [boxes[i] for i in a_idx]
        boxes_b = [boxes[i] for i in b_idx]
        boxes_c = [boxes[i] for i in c_idx]
        image_a = draw_objects_on_image(obj_label, boxes_a, canvas_size)
        image_b = draw_objects_on_image(obj_label, boxes_b, canvas_size)
        image_c = draw_objects_on_image(obj_label, boxes_c, canvas_size)
        label = {"obj": obj_label, "a": len(boxes_a), "b": len(boxes_b), "c": len(boxes_c)}
        data.append((image_a, image_b, image_c, label))
        rest_iter -= 1
        if a >= b:
            max_num = a
            box_idx = a_idx
        else:
            max_num = b
            box_idx = b_idx
    return data


def gen_recurrent_data(num_samples, canvas_size=(224, 224), obj_list=OBJ_LIST):
    data = []
    while len(data) < num_samples:
        obj_label = random.choice(obj_list)
        c = random.randint(8, 10)
        boxes = generate_non_overlapping_boxes(c, canvas_size, 30, 40)
        if boxes is None:
            continue
        data += recurrent_generate_data(boxes, c, obj_label, canvas_size)
    data = data[:num_samples]
    print(f"Generated {len(data)} samples.")
    return data


#----------------------------生成“计划”：不画图，只决定 a/b/c 和三个视图的盒子----------------------------
@dataclass
class TriplePlan:
    # 三张图用到的盒子（位置完全固定），以及 a,b,c
    boxes_a: List[Tuple[int,int,int,int]]
    boxes_b: List[Tuple[int,int,int,int]]
    boxes_c: List[Tuple[int,int,int,int]]
    a: int
    b: int
    c: int

def recurrent_generate_plan(boxes, c, max_iter=1000) -> List[TriplePlan]:
    """与 recurrent_generate_data 对齐，但不渲染，只返回 plan。"""
    plans = []
    max_num = c
    rest_iter = max_iter
    box_idx = list(range(len(boxes)))
    while rest_iter > 0 and max_num >= 2:
        a = random.randint(1, max_num - 1)
        b = max_num - a
        if b < 0:
            break
        c_idx = random.sample(box_idx, max_num)
        a_idx = random.sample(c_idx, a)
        b_idx = [i for i in c_idx if i not in a_idx]

        boxes_a = [boxes[i] for i in a_idx]
        boxes_b = [boxes[i] for i in b_idx]
        boxes_c = [boxes[i] for i in c_idx]

        plans.append(TriplePlan(
            boxes_a=boxes_a, boxes_b=boxes_b, boxes_c=boxes_c,
            a=a, b=b, c=max_num
        ))

        rest_iter -= 1
        if a >= b:
            max_num = a
            box_idx = a_idx
        else:
            max_num = b
            box_idx = b_idx
    return plans

def gen_recurrent_plan(num_samples, canvas_size=(224,224), min_size=30, max_size=40) -> List[TriplePlan]:
    """直到收集到 num_samples 条 plan 为止。"""
    plans: List[TriplePlan] = []
    while len(plans) < num_samples:
        c = random.randint(8, 10)
        boxes = generate_non_overlapping_boxes(c, canvas_size, min_size, max_size)
        if boxes is None:
            continue
        plans.extend(recurrent_generate_plan(boxes, c, max_iter=1000))
    return plans[:num_samples]
#----------------------------在线生成数据集的 DataLoader 初始化----------------------------