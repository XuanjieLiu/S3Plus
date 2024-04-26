import random

from eval_common import CommonEvaler, draw_scatter_point_line
from dataMaker_commonFunc import MARK_NAME_SPACE
from common_func import EXP_ROOT
from shared import *
from importlib import reload
import sys
import os
from common_func import load_config_from_exp_name, record_num_list, DATASET_ROOT, EXP_ROOT, dict_switch_key_value
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
):
    fig, ax = plt.subplots(figsize=(21, 14))
    for i in range(len(num_emb_idx)):
        x_shift = random.uniform(-0.25, 0.25)
        y_shift = random.uniform(-1.5, 1.5)
        ax.scatter(num_labels[i]+x_shift, num_emb_idx[i]+y_shift, c='none', edgecolors=num_colors[i],
                   facecolors='none', s=60, marker=num_shapes[i])
    ax.set(xlabel='Num of Points on the Card', xticks=range(0, max(num_labels) + 1))
    ax.set(ylabel=y_label)
    ax.set_title(f"{title}")
    ax.grid(is_gird)
    if is_scatter_lines:
        draw_scatter_point_line(ax, [*num_labels], [*num_emb_idx])
    ax.label_outer()
    # plt.legend()
    plt.savefig(f'{eval_path}.png')
    plt.cla()
    plt.clf()
    plt.close()


class MultiStyleZcEvaler(CommonEvaler):
    def __init__(self, config, model_path=None):
        super(MultiStyleZcEvaler, self).__init__(config, model_path)

    def eval(self, data_loader, save_path, figure_title=''):
        num_z, num_labels, colors, shapes = load_enc_eval_data_with_style(
            data_loader,
            lambda x: self.model.find_indices(
                self.model.batch_encode_to_z(x)[0],
                True, False
            )
        )
        num_emb_idx = num_z.detach().cpu().numpy()
        shape_dict = dict_switch_key_value(MARK_NAME_SPACE)
        shape_marks = [shape_dict[shape] for shape in shapes]
        plot_plusZ_against_label(num_emb_idx, num_labels, colors, shape_marks, eval_path=save_path,
                                 is_scatter_lines=True, is_gird=True, title=figure_title, y_label='Content Emb Idx')


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
    SUB_EXP = 1
    CHECK_POINT = 'checkpoint_10000.pt'
    exp_path = os.path.join(EXP_ROOT, EXP_NAME)
    check_point_path = os.path.join(exp_path, str(SUB_EXP), CHECK_POINT)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)

    evaler = MultiStyleZcEvaler(t_config.CONFIG, check_point_path)
    save_path = os.path.join(exp_path, str(SUB_EXP), 'eval_zc')
    evaler.eval(single_img_eval_loader, save_path)