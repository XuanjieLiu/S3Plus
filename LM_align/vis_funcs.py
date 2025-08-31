import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
import math
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use('Agg')

def visualize_alignment(z, labels, imgs, embs, zs, e_ab, z_ab, save_path=None):
    assert len(imgs) == len(embs) == len(zs) == 3
    def plot_sps_labels(z, labels, ax):
        COLOR_LIST = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'hotpink', 'gray', 'steelblue', 'olive']
        sorted_label = sorted(labels)
        sorted_indices = [i[0] for i in sorted(enumerate(labels), key=lambda x: x[1])]
        sorted_num_z = [z[i] for i in sorted_indices]
        X = [item[0] for item in sorted_num_z]
        Y = [item[1] for item in sorted_num_z]
        for i in range(0, len(z)):
            ax.scatter(X[i], Y[i],
                        marker=f'${sorted_label[i]}$',
                        s=200,
                        alpha=0.5,
                        c=COLOR_LIST[sorted_label[i] % len(COLOR_LIST)],
                        )

        ax.grid(True, which="both", linestyle="--")
        ax.plot(X, Y, linestyle='dashed', linewidth=0.5)
    
    img_labels = ['a', 'b', 'c']
    colors = ['g', 'b', 'r']
    # 创建画布
    fig = plt.figure(figsize=(10, 6))  # 画布尺寸可以根据需求调整

    # 创建网格布局
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.1)  # 设置宽度比例3:1，调整间距wspace

    # 左边大图
    ax1 = plt.subplot(gs[0, 0])  # 左边第一列的大图
    plot_sps_labels(z, labels, ax1)
    ax1.set_aspect('equal')  # 确保比例

    # 大图补充 embs，zs，e_ab，z_ab
    for i in range(3):
        ax1.scatter(embs[i][0], embs[i][1], 
                    marker=f's', 
                    alpha=1, 
                    label=f'e_{img_labels[i]}',
                    c='none', edgecolors=colors[i],
                    facecolors='none', 
                    s=40+30*i,
                    )
        ax1.scatter(zs[i][0], zs[i][1], 
                    marker=f'o', 
                    alpha=1, 
                    label=f'z_{img_labels[i]}',
                    c='none', edgecolors=colors[i],
                    facecolors='none', 
                    s=40+30*i,
                    )
        # ax1.scatter(zs[i][0], zs[i][1], marker=f'o', s=20, alpha=0.7, c=colors[i], labels=f'z_{img_labels[i]}')
    ax1.scatter(e_ab[0], e_ab[1], marker=f's', s=130, alpha=1, edgecolors='purple', c='none', facecolors='none', label='e_ab')
    ax1.scatter(z_ab[0], z_ab[1], marker=f'o', s=130, alpha=1, edgecolors='purple', c='none', facecolors='none', label='z_ab')
    # ax1.scatter(z_ab[0], z_ab[1], marker=f'x', s=20, alpha=0.5, c='orange', labels='z_ab')
    ax1.legend()

    # 右边三张小图，纵向排列，分别显示 img_a, img_b, img_c
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1], hspace=0.3)  # 嵌套子网格，设置间距hspace
    for i in range(3):
        ax = plt.subplot(gs_right[i])
        img_to_show = np.transpose(imgs[i], (1, 2, 0))  # 从 (3, 224, 224) 到 (224, 224, 3)
        ax.imshow(img_to_show)
        ax.set_title(f"{img_labels[i]}", color=colors[i])
        # ax.axis('off')
        ax.spines['top'].set_visible(True)     # 显示顶部轴
        ax.spines['right'].set_visible(True)   # 显示右侧轴
        ax.set_xticks([])  # 隐藏 x 轴刻度
        ax.set_yticks([])  # 隐藏 y 轴刻度
        ax.tick_params(axis='both', which='both', length=0)

    # 保存图片
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)  # 关闭画布，释放内存


def plot_confusion_matrix(ax, pred_label, label_all, objs_all, obj_type,
                          normalize_rows=True, show_counts=False):
    """
    在给定的坐标轴 ax 上绘制混淆矩阵热力图，并添加对角辅助线，
    同时在标题中显示该类别名称及准确率（保留三位小数）。

    当 obj_type=="Overall" 时，不做类别过滤，使用所有数据。

    参数：
    - normalize_rows: True 时逐行归一化（每行和为 1）；False 时显示计数。
    - show_counts:    仅在 normalize_rows=True 时生效；为 True 则在单元格中显示 `计数(比例)`。
    """
    # 转换为 numpy 数组
    pred_label = np.array(pred_label)
    label_all  = np.array(label_all)
    objs_all   = np.array(objs_all)

    # 如果是整体统计，则直接使用所有数据；否则过滤指定类别
    if obj_type == "Overall":
        pred_sub = pred_label
        label_sub = label_all
    else:
        mask = (objs_all == obj_type)
        pred_sub = pred_label[mask]
        label_sub = label_all[mask]

    # 计算准确率（防止除0错误）
    accuracy = np.mean(pred_sub == label_sub) if len(label_sub) > 0 else 0.0

    # 动态识别实际与预测计数的最小值和最大值
    if len(label_sub) > 0:
        min_actual = int(np.min(label_sub))
        max_actual = int(np.max(label_sub))
    else:
        min_actual, max_actual = 0, 0
    if len(pred_sub) > 0:
        min_pred = int(np.min(pred_sub))
        max_pred = int(np.max(pred_sub))
    else:
        min_pred, max_pred = 0, 0

    # 构建混淆矩阵（计数）
    n_actual = max_actual - min_actual + 1
    n_pred   = max_pred   - min_pred   + 1
    confusion = np.zeros((n_actual, n_pred), dtype=int)

    for pred, actual in zip(pred_sub, label_sub):
        if min_actual <= actual <= max_actual and min_pred <= pred <= max_pred:
            confusion[int(actual) - min_actual, int(pred) - min_pred] += 1

    # === 逐行归一化 ===
    if normalize_rows:
        row_sums = confusion.sum(axis=1, keepdims=True)
        # 安全除法：行和为 0 的行保持为 0
        confusion_norm = np.divide(
            confusion.astype(float),
            row_sums,
            out=np.zeros_like(confusion, dtype=float),
            where=(row_sums != 0)
        )
        data_to_show = confusion_norm
        vmin, vmax = 0.0, 1.0  # 固定颜色范围，便于不同子图可比
    else:
        data_to_show = confusion
        vmin, vmax = None, None  # 计数模式使用自适应范围

    # 自定义颜色映射：从 green 到 yellow（保留你的风格）
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green', 'yellow'])

    im = ax.imshow(data_to_show, interpolation='nearest', cmap=custom_cmap,
                   origin='lower', vmin=vmin, vmax=vmax)

    # 在每个格子内添加数字注释
    for i in range(n_actual):
        for j in range(n_pred):
            if normalize_rows:
                if show_counts:
                    ax.text(j, i, f'{confusion[i, j]} ({data_to_show[i, j]:.2f})',
                            ha='center', va='center', color='black', fontsize=9)
                else:
                    ax.text(j, i, f'{data_to_show[i, j]:.2f}',
                            ha='center', va='center', color='black', fontsize=9)
            else:
                ax.text(j, i, str(confusion[i, j]),
                        ha='center', va='center', color='black', fontsize=9)

    # 坐标轴刻度与标签
    ax.set_xticks(np.arange(n_pred))
    ax.set_xticklabels(np.arange(min_pred, max_pred + 1))
    ax.set_yticks(np.arange(n_actual))
    ax.set_yticklabels(np.arange(min_actual, max_actual + 1))
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Actual Count')

    # 对角辅助线（实际=预测）
    lower_bound = max(min_actual, min_pred)
    upper_bound = min(max_actual, max_pred)
    if lower_bound <= upper_bound:
        a_values   = np.arange(lower_bound, upper_bound + 1)
        row_coords = a_values - min_actual
        col_coords = a_values - min_pred
        ax.plot(col_coords, row_coords, color='red', linestyle='--', linewidth=2)

    # 标题
    ax.set_title(f'{obj_type} (Acc: {accuracy:.3f})')

    # 颜色条
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_accuracy_bar(ax, categories, accuracies):
    """
    在给定的坐标轴 ax 上绘制柱状图，展示每个类别（包含 Overall）的准确率。
    """
    bars = ax.bar(categories, accuracies, color='skyblue', edgecolor='black')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    ax.set_title('Accuracy for Each Object Type (Including Overall)')
    # 在每个柱子上添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{acc:.3f}', ha='center', va='bottom')

def vis_all_confusion_and_accuracy(pred_label, label_all, objs_all, save_path=None):
    """
    综合绘制图形：
    1. 第一行：左侧为整体（Overall）的混淆矩阵，右侧为包含 Overall 在内的各类别准确率柱状图。
    2. 从第二行开始：绘制各个具体类别（不含 Overall）的混淆矩阵。
    """
    pred_label = np.array(pred_label)
    label_all = np.array(label_all)
    objs_all = np.array(objs_all)
    
    # 计算整体准确率
    overall_acc = np.mean(pred_label == label_all) if len(label_all) > 0 else 0.0
    
    # 获取各类别
    unique_objs = np.unique(objs_all)
    
    # 构造柱状图数据：第一项为 "Overall"，后续为各个 obj_type
    categories = np.concatenate((np.array(["Overall"]), unique_objs))
    bar_accuracies = [overall_acc]
    for obj in unique_objs:
        mask = objs_all == obj
        pred_sub = pred_label[mask]
        label_sub = label_all[mask]
        if len(label_sub) > 0:
            acc = np.mean(pred_sub == label_sub)
        else:
            acc = 0.0
        bar_accuracies.append(acc)
    
    # 计算个别类别混淆矩阵的数量（不包括 Overall）
    n_obj = len(unique_objs)
    n_cols = min(4, n_obj) if n_obj > 0 else 1
    n_rows = math.ceil(n_obj / n_cols)
    
    # 创建大图。这里将大图分为两部分：
    # 第一部分（第一行）：一个左右两列的子区域，左侧显示 Overall 的混淆矩阵，右侧显示柱状图（较宽）。
    # 第二部分（从第二行开始）：显示各个具体类别的混淆矩阵。
    fig = plt.figure(constrained_layout=True, figsize=(18, 6 + 4*n_rows))
    # 主区域分为两行
    gs = fig.add_gridspec(2, 1, height_ratios=[1, n_rows])
    
    # 第一行：分为左右两列，宽度比设置为 1:3（左侧显示 Overall 混淆矩阵，右侧显示柱状图）
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1,3])
    # 左侧：Overall 混淆矩阵
    ax_overall = fig.add_subplot(gs_top[0, 0])
    plot_confusion_matrix(ax_overall, pred_label, label_all, objs_all, "Overall")
    # 右侧：柱状图
    ax_bar = fig.add_subplot(gs_top[0, 1])
    plot_accuracy_bar(ax_bar, categories, bar_accuracies)
    
    # 第二部分：为每个具体类别绘制混淆矩阵
    gs_bottom = gs[1].subgridspec(n_rows, n_cols)
    for idx, obj in enumerate(unique_objs):
        row = idx // n_cols
        col = idx % n_cols
        ax_cm = fig.add_subplot(gs_bottom[row, col])
        plot_confusion_matrix(ax_cm, pred_label, label_all, objs_all, obj)
    
    # 隐藏多余的子图（如果有）
    total_plots = n_rows * n_cols
    if total_plots > n_obj:
        for idx in range(n_obj, total_plots):
            row = idx // n_cols
            col = idx % n_cols
            ax_dummy = fig.add_subplot(gs_bottom[row, col])
            ax_dummy.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)  # 关闭画布，释放内存

# 示例调用：
if __name__ == "__main__":
    # 示例数据（包含多个类别以及可能的 0 值）
    pred_label = [3, 4, 4, 5, 2, 7, 8, 9, 10, 4, 6, 0, 2, 3, 5, 4, 4]
    label_all  = [3, 4, 6, 5, 2, 6, 8, 9, 10, 3, 6, 0, 2, 2, 5, 4, 5]
    objs_all   = ['apple', 'car', 'apple', 'flower', 'apple', 'car', 'apple', 'flower', 
                  'apple', 'car', 'apple', 'apple', 'banana', 'banana', 'banana', 'dog', 'dog']
    
    vis_all_confusion_and_accuracy(pred_label, label_all, objs_all)





