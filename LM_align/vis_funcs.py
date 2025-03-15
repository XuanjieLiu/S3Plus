import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
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