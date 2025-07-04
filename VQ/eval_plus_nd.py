import os
import random
from typing import List

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader_plus import Dataset
from loss_counter import LossCounter
from VQVAE import VQVAE
from shared import *
from eval_common import draw_scatter_point_line
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from scipy import stats
from common_func import parse_label

matplotlib.use('AGG')
MODEL_PATH = 'curr_model.pt'
EVAL_ROOT = 'eval_z'

MIN_KS_NUM = 1


class EncZ:
    def __init__(self, label, z):
        self.label = label
        self.z = z


class PlusZ:
    def __init__(self, label_a, label_b, plus_c_z):
        self.plus_c_z = plus_c_z
        self.label_c = label_a + label_b




# def plot_plusZ_against_label(all_enc_z, all_plus_z, eval_path, eval_helper: EvalHelper = None):
#     ks_mean, _ = calc_ks_enc_plus_z(all_enc_z, all_plus_z)
#     dim_z = all_enc_z[0].z.size(0)
#     fig, axs = plt.subplots(1, dim_z, sharey='none', figsize=(dim_z * 5, 5))
#     if dim_z == 1:
#         axs = [axs]
#     enc_x = [ob.label for ob in all_enc_z]
#     plus_x = [ob.label_c for ob in all_plus_z]
#     for i in range(0, dim_z):
#         enc_y = [ob.z.cpu()[i].item() for ob in all_enc_z]
#         plus_y = [ob.plus_c_z.cpu()[i].item() for ob in all_plus_z]
#         axs[i].scatter(enc_x, enc_y, edgecolors='blue', label='Encoder output', facecolors='none', s=60)
#         axs[i].scatter(plus_x, plus_y, edgecolors='red', label='Plus output', facecolors='none', s=20)
#         axs[i].set(xlabel='Num of Points on the card', xticks=range(0, max(enc_x) + 1))
#         if eval_helper is not None:
#             eval_helper.draw_scatter_point_line(axs[i], i, [*enc_x, *plus_x], [*enc_y, *plus_y])
#             eval_helper.set_axis(axs[i], i)
#         else:
#             axs[i].grid(True)
#             axs[i].set_title(f"z_c ({i + 1})")
#     # for ax in axs.flat:
#     #     ax.label_outer()
#     plt.legend()
#     plt.savefig(f'{eval_path}_ks_{ks_mean}.png')
#     plt.cla()
#     plt.clf()
#     plt.close()


def plot_plusZ_against_label(
        all_enc_z,
        all_plus_z,
        eval_path,
        n_rows: int = 1,
        n_cols: int = -1,
        is_scatter_lines=False,
        is_gird=False,
        sub_title: str = '',
        y_label: str = '',
        sub_fig_size=5
):
    dim_z = all_enc_z[0].z.size(0)
    n_row = n_rows
    n_col = dim_z if n_cols == -1 else n_cols
    fig, axs = plt.subplots(n_row, n_col, sharey='all', sharex='all',
                            figsize=(n_col * sub_fig_size, n_row * sub_fig_size), squeeze=False)
    enc_x = [ob.label for ob in all_enc_z]
    plus_x = [ob.label_c for ob in all_plus_z]
    for i in range(0, dim_z):
        enc_y = [ob.z.cpu()[i].item() for ob in all_enc_z]
        plus_y = [ob.plus_c_z.cpu()[i].item() for ob in all_plus_z]
        axs.flat[i].scatter(enc_x, enc_y, edgecolors='blue', label='Encoder output', facecolors='none', s=60)
        axs.flat[i].scatter(plus_x, plus_y, edgecolors='red', label='Plus output', facecolors='none', s=20)
        axs.flat[i].set(xlabel='Num of Points on the card', xticks=range(0, max(enc_x) + 1))
        axs.flat[i].set(ylabel=y_label)
        axs.flat[i].set_title(f"{sub_title} ({i + 1})")
        axs.flat[i].grid(is_gird)
        if is_scatter_lines:
            draw_scatter_point_line(axs.flat[i], [*enc_x, *plus_x], [*enc_y, *plus_y])
    for ax in axs.flat:
        ax.label_outer()
    plt.legend()
    plt.savefig(f'{eval_path}.png')
    plt.cla()
    plt.clf()
    plt.close()


class LabelKs:
    def __init__(self, label: int):
        self.label = label
        self.enc_z_list = []
        self.plus_z_list = []

    def calc_min_len(self):
        return min(len(self.enc_z_list), len(self.plus_z_list))

    def calc_ks(self):
        min_len = self.calc_min_len()
        sample1 = random.sample(self.enc_z_list, min_len)
        sample2 = random.sample(self.plus_z_list, min_len)
        ks, p_value = stats.ks_2samp(sample1, sample2)
        return ks, p_value

    def calc_accu(self):
        total = len(self.plus_z_list)
        assert total != 0, f"plus_z_list of card {self.label} is empty."
        assert len(self.enc_z_list) != 0, f"enc_z_list of card {self.label} is empty."
        mode_enc = stats.mode(self.enc_z_list, keepdims=False)[0]
        correct = len(list(filter(lambda x: x == mode_enc, self.plus_z_list)))
        accu = correct / total
        return accu


def calc_ks_enc_plus_z(enc_z: List[EncZ], plus_z: List[PlusZ]):
    min_label = min(enc_z, key=lambda x: x.label).label
    max_label = max(enc_z, key=lambda x: x.label).label
    label_dict = {}
    for i in range(min_label, max_label + 1):
        label_dict[i] = LabelKs(i)
    for item in enc_z:
        label_dict[item.label].enc_z_list.append(item.z.cpu().item())
    for item in plus_z:
        label_dict[item.label_c].plus_z_list.append(item.plus_c_z.cpu().item())
    label_ks_list = list(label_dict.values())
    ks_all = 0
    accu_all = 0
    item_num = 0
    for item in label_ks_list:
        min_len = item.calc_min_len()
        if min_len >= MIN_KS_NUM:
            ks_all += item.calc_ks()[0]
            accu_all += item.calc_accu()
            item_num += 1
    ks_mean = ks_all / item_num
    accu_mean = accu_all / item_num
    return round(ks_mean, 4), round(accu_mean, 4)


def calc_multi_emb_plus_accu(enc_z: List[EncZ], plus_z: List[PlusZ]):
    """
    Calculate the accuracy of plus operation based. A label can be represented by multiple embeddings.
    An embedding can only represent one label, determined by the mode of the label it represents.
    :param enc_z:
    :param plus_z:
    :return: accu: float, the accuracy of plus operation.
    """
    # 先确认每个 emb 对应的 labels
    emb_dict = {}
    for item in enc_z:
        if int(item.z.item()) not in emb_dict:
            emb_dict[int(item.z.item())] = []
        emb_dict[int(item.z.item())].append(item.label)
    # 根据每个 emb 表示的 label 的众数确定其唯一对应的 label
    emb_label_dict = {}
    for emb, labels in emb_dict.items():
        mode_label = stats.mode(labels, keepdims=False)[0]
        emb_label_dict[emb] = mode_label
    # 根据 emb_label_dict 计算 plus_z 的准确率
    n_correct = 0
    n_total = len(plus_z)
    assert n_total != 0, "plus_z is empty."
    for item in plus_z:
        label_z = emb_label_dict.get(int(item.plus_c_z.item()))
        if label_z is not None and int(label_z) == int(item.label_c):
            n_correct += 1
    return n_correct / n_total


class VQvaePlusEval:
    def __init__(self, config, model_path=None, loaded_model: VQVAE = None):
        self.config = config
        self.zc_dim = config['latent_embedding_1'] * config['embedding_dim']
        if loaded_model is not None:
            self.model = loaded_model
        elif model_path is not None:
            self.model = VQVAE(config).to(DEVICE)
            self.reload_model(model_path)
            print(f"Model is loaded from {model_path}")
        else:
            self.model = VQVAE(config).to(DEVICE)
            print("No model is loaded")
        self.model.eval()

    def reload_model(self, model_path):
        self.model.load_state_dict(self.model.load_tensor(model_path))

    def load_plusZ_eval_data(self, data_loader: DataLoader, is_find_index=True):
        all_enc_z = []
        all_plus_z = []
        for batch_ndx, sample in enumerate(data_loader):
            enc_z_list = []
            plus_z_list = []
            data, labels = sample
            za = self.model.batch_encode_to_z(data[0])[0]
            zb = self.model.batch_encode_to_z(data[1])[0]
            zc = self.model.batch_encode_to_z(data[2])[0]
            za = za[..., 0:self.zc_dim]
            zb = zb[..., 0:self.zc_dim]
            zc = zc[..., 0:self.zc_dim]
            label_a = [parse_label(x) for x in labels[0]]
            label_b = [parse_label(x) for x in labels[1]]
            label_c = [parse_label(x) for x in labels[2]]
            plus_c = self.model.plus(za, zb)[0]
            if is_find_index:
                idx_z_a = self.model.find_indices(za, False)
                idx_z_b = self.model.find_indices(zb, False)
                idx_z_c = self.model.find_indices(zc, False)
                idx_plus_c = self.model.find_indices(plus_c, False)
            else:
                idx_z_a = za
                idx_z_b = zb
                idx_z_c = zc
                idx_plus_c = plus_c
            for i in range(0, za.size(0)):
                enc_z_list.append(EncZ(label_a[i], idx_z_a[i]))
                enc_z_list.append(EncZ(label_b[i], idx_z_b[i]))
                enc_z_list.append(EncZ(label_c[i], idx_z_c[i]))
                plus_z_list.append(PlusZ(label_a[i], label_b[i], idx_plus_c[i]))
            all_enc_z.extend(enc_z_list)
            all_plus_z.extend(plus_z_list)
        return all_enc_z, all_plus_z

    def eval(self, eval_path, dataloader: DataLoader):
        all_enc_z, all_plus_z = self.load_plusZ_eval_data(dataloader)
        plot_plusZ_against_label(all_enc_z, all_plus_z, eval_path)

    def eval_multi_emb_accu(self, eval_data_loader: DataLoader):
        all_enc_z, all_plus_z = self.load_plusZ_eval_data(eval_data_loader, is_find_index=True)
        accu = calc_multi_emb_plus_accu(all_enc_z, all_plus_z)
        return accu

# if __name__ == "__main__":
#     os.makedirs(EVAL_ROOT, exist_ok=True)
#     dataset = Dataset(CONFIG['train_data_path'])
#     loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
#     evaler = VQvaePlusEval(CONFIG, loader, model_path=MODEL_PATH)
#     result_path = os.path.join(EVAL_ROOT, f'plus_eval_{MODEL_PATH}.png')
#     evaler.eval(result_path)
