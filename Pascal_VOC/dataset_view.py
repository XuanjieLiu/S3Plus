import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from torchvision.datasets import VOCDetection
import os

# 加载 VOC 数据集
data_dir = "./VOCdevkit/"
year = "2012"
image_set = "train"
dataset = VOCDetection(root=data_dir, year=year, image_set=image_set)

# 筛选出只包含一种类别的图片
filtered_data = []
for idx, (_, target) in enumerate(dataset):
    objects = target['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]
    classes = [obj['name'] for obj in objects]
    if len(set(classes)) == 1:
        filtered_data.append((idx, classes[0], len(classes)))  # (图片索引, 类别, 目标数量)

# 初始化主窗口
root = tk.Tk()
root.title("VOCDetection Viewer")
root.geometry("800x600")

# 筛选条件变量
selected_class = tk.StringVar(value="All")
selected_count = tk.StringVar(value="All")

# 显示图片用的变量
current_image = None

# 过滤后的图片列表
filtered_images = []

# 筛选函数
def apply_filters():
    global filtered_images
    filtered_images = []

    # 获取筛选条件
    class_filter = selected_class.get()
    count_filter = selected_count.get()

    # 应用筛选条件
    for idx, obj_class, obj_count in filtered_data:
        if (class_filter == "All" or obj_class == class_filter) and \
           (count_filter == "All" or int(count_filter) == obj_count):
            filtered_images.append((idx, obj_class, obj_count))

    # 更新图片列表下拉菜单
    update_image_list()

# 更新下拉菜单
def update_image_list():
    image_list_menu['values'] = [f"Image {i}: {cls} ({cnt})" for i, (idx, cls, cnt) in enumerate(filtered_images)]
    if filtered_images:
        image_list_menu.current(0)
    else:
        image_list_menu.set("No Images Found")
    display_image()

# 显示图片
def display_image(*args):
    global current_image
    if not filtered_images:
        return
    selected_idx = image_list_menu.current()
    if selected_idx < 0:
        return

    img_idx, _, _ = filtered_images[selected_idx]
    image_path = dataset.images[img_idx]
    img = Image.open(image_path)
    img.thumbnail((400, 400))  # 缩放以适应窗口
    current_image = ImageTk.PhotoImage(img)
    image_label.config(image=current_image)

# 类别和目标数量选项
all_classes = sorted(set(obj_class for _, obj_class, _ in filtered_data))
all_counts = sorted(set(obj_count for _, _, obj_count in filtered_data))

# 类别筛选器
tk.Label(root, text="Filter by Class:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
class_filter_menu = ttk.Combobox(root, textvariable=selected_class, values=["All"] + all_classes)
class_filter_menu.grid(row=0, column=1, padx=10, pady=10)
class_filter_menu.bind("<<ComboboxSelected>>", lambda e: apply_filters())

# 数量筛选器
tk.Label(root, text="Filter by Count:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
count_filter_menu = ttk.Combobox(root, textvariable=selected_count, values=["All"] + list(map(str, all_counts)))
count_filter_menu.grid(row=1, column=1, padx=10, pady=10)
count_filter_menu.bind("<<ComboboxSelected>>", lambda e: apply_filters())

# 图片列表下拉菜单
tk.Label(root, text="Filtered Images:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
image_list_menu = ttk.Combobox(root)
image_list_menu.grid(row=2, column=1, padx=10, pady=10)
image_list_menu.bind("<<ComboboxSelected>>", display_image)

# 图片显示区域
image_label = tk.Label(root)
image_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# 初始化界面
apply_filters()

# 运行主循环
root.mainloop()
