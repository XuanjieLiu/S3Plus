import random
import numpy as np
from eval_common import CommonEvaler, draw_scatter_point_line
from dataMaker_commonFunc import MARK_NAME_SPACE, MARK_NAME_SPACE_MAHJONG
from common_func import EXP_ROOT, KMMatcher
from shared import *
from importlib import reload
import sys
import os
from common_func import keep_max_in_matrix_colum, record_num_list, DATASET_ROOT, EXP_ROOT, dict_switch_key_value
from dataloader import SingleImgDataset, load_enc_eval_data_with_style
from torch.utils.data import DataLoader
import matplotlib.markers
import matplotlib.pyplot as plt
matplotlib.use('AGG')


def plot_plusZ_against_label(
        num_emb_idx,
        num_labels,
        num_colors,
        num_shapes,
        eval_path,
        is_scatter_lines=False,
        is_gird=False,
        title: str = '',
        y_label: str = '',
        reordered_embs=None,
        is_shift=True
):
    fig, ax = plt.subplots(figsize=(21, 14))
    if reordered_embs is not None:
        X = []
        Y = []
        for i in range(len(num_emb_idx)):
            if num_emb_idx[i] not in reordered_embs:
                continue
            X.append(num_labels[i])
            Y.append(reordered_embs.index(num_emb_idx[i]) + 1)
    else:
        X = num_labels
        Y = num_emb_idx
    for i in range(len(Y)):
        if is_shift:
            x_shift = random.uniform(-0.25, 0.25)
            if reordered_embs is not None:
                y_shift = random.uniform(-0.4, 0.4)
            else:
                y_shift = random.uniform(-1.5, 1.5)
        else:
            x_shift = 0
            y_shift = 0
        ax.scatter(X[i]+x_shift, Y[i]+y_shift, c='none', edgecolors=num_colors[i],
                   facecolors='none', s=60, marker=num_shapes[i])
        ax.scatter(X[i], Y[i], c='navy', s=30, marker='o')
    ax.set(xlabel='Num of Points on the Card', xticks=range(0, max(num_labels) + 1))
    ax.set(ylabel=y_label)
    ax.set_title(f"{title}")
    ax.grid(is_gird)
    if is_scatter_lines:
        draw_scatter_point_line(ax, [*X], [*Y])
    ax.label_outer()
    # plt.legend()
    plt.savefig(f'{eval_path}.png')
    plt.cla()
    plt.clf()
    plt.close()


class MultiStyleZcEvaler(CommonEvaler):
    def __init__(self, config, model_path=None, loaded_model=None):
        super(MultiStyleZcEvaler, self).__init__(config, model_path, loaded_model)

    def eval(self, data_loader, save_path, figure_title='', is_shift=True, is_plot_graph=True):
        num_z, num_labels, colors, shapes = load_enc_eval_data_with_style(
            data_loader,
            lambda x: self.model.find_indices(
                self.model.batch_encode_to_z(x)[0],
                True, False
            )
        )
        num_emb_idx = [x[0] for x in num_z.detach().cpu().numpy()]
        emb_efficiency = self.calc_emb_matching_score(num_emb_idx, num_labels)
        reordered_embs = self.reorder_emb(num_emb_idx, num_labels)
        if is_plot_graph:
            shape_dict = dict_switch_key_value(MARK_NAME_SPACE)
            if 'mahjong' in self.config['single_img_eval_set_path']:
                shape_dict = dict_switch_key_value(MARK_NAME_SPACE_MAHJONG)
            shape_marks = [shape_dict[shape] for shape in shapes]
            plot_plusZ_against_label(num_emb_idx, num_labels, colors, shape_marks,
                                     eval_path=f'{save_path}_{round(emb_efficiency, 2)}',
                                     is_scatter_lines=True, is_gird=True,
                                     title=f'{figure_title} (match rate: {round(emb_efficiency, 2)})',
                                     y_label='Used Embedding',
                                     reordered_embs=reordered_embs,
                                     is_shift=is_shift
                                     )
        return emb_efficiency

    def assemble_label_emb_matrix(self, num_emb_idx, num_labels):
        """
        Assemble a label embedding matrix based on the provided embedding indices and labels.
        :param num_emb_idx:
        :param num_labels:
        :return: label_emb_matrix: A matrix where each row corresponds to a unique label and each column corresponds to an embedding index.
        """
        book_size = self.config['embeddings_num']
        total_emb = pow(book_size, self.latent_embedding_1)
        all_labels = [int(n) for n in list(set(num_labels))]
        label_emb_matrix = np.zeros((len(all_labels), total_emb))
        for i in range(len(num_labels)):
            label = num_labels[i]
            emb_idx = int(num_emb_idx[i])
            label_emb_matrix[all_labels.index(label), emb_idx] += 1
        return label_emb_matrix

    def calc_KM(self, num_emb_idx, num_labels):
        label_emb_matrix = self.assemble_label_emb_matrix(num_emb_idx, num_labels)
        KM_value = KMMatcher(label_emb_matrix).solve(verbose=False)
        full_score = len(num_labels)
        return KM_value / full_score

    def calc_emb_matching_score(self, num_emb_idx, num_labels):
        label_emb_matrix = self.assemble_label_emb_matrix(num_emb_idx, num_labels)
        max_row_matrix = keep_max_in_matrix_colum(label_emb_matrix)
        score = np.sum(max_row_matrix)
        full_score = len(num_labels)
        return score / full_score

    def reorder_emb(self, num_emb_idx, num_labels):
        label_emb_matrix = self.assemble_label_emb_matrix(num_emb_idx, num_labels)
        max_row_matrix = keep_max_in_matrix_colum(label_emb_matrix)
        all_used_embs = []
        for i in range(len(max_row_matrix)):
            all_used_embs.extend(find_emb_idx_by_efficiency(max_row_matrix[i]))
        return all_used_embs


def find_emb_idx_by_efficiency(array_1d):
    indices = np.argsort(array_1d)
    # 翻转索引以实现从大到小的排序
    reversed_indices = np.flip(indices)
    embs = []
    for i in reversed_indices:
        if array_1d[i] > 0:
            embs.append(i)
        else:
            break
    return embs


if __name__ == "__main__":
    EVAL_SETS = [
        {
            'name': 'Train_style',
            'path': os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_TrainStyle')
        },
    ]
    single_img_eval_set = SingleImgDataset(EVAL_SETS[0]['path'])
    single_img_eval_loader = DataLoader(single_img_eval_set, batch_size=256)
    EVAL_SETS[0]['loader'] = single_img_eval_loader

    EXP_NAME = '2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing'
    SUB_EXP = 6
    CHECK_POINT = 'checkpoint_9500.pt'
    exp_path = os.path.join(EXP_ROOT, EXP_NAME)
    check_point_path = os.path.join(exp_path, str(SUB_EXP), CHECK_POINT)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)

    evaler = MultiStyleZcEvaler(t_config.CONFIG, check_point_path)
    save_path = os.path.join(exp_path, str(SUB_EXP), 'eval_zc')
    emb_efficiency = evaler.eval(single_img_eval_loader, save_path)
    print(f'Emb efficiency: {emb_efficiency}')