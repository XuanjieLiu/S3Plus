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
from eval_common import EvalHelper
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.use('AGG')
MODEL_PATH = 'curr_model.pt'
EVAL_ROOT = 'eval_z'


MIN_KS_NUM = 3

class EncZ:
    def __init__(self, label, z):
        self.label = label
        self.z = z


class PlusZ:
    def __init__(self, label_a, label_b, plus_c_z):
        self.plus_c_z = plus_c_z
        self.label_c = label_a + label_b


def parse_label(label):
    return int(label.split('.')[0].split('-')[1])


def plot_plusZ_against_label(all_enc_z, all_plus_z, eval_path, eval_helper: EvalHelper = None):
    ks_mean = calc_ks_enc_plus_z(all_enc_z, all_plus_z)
    dim_z = all_enc_z[0].z.size(0)
    fig, axs = plt.subplots(1, dim_z, sharey='none', figsize=(dim_z * 5, 5))
    if dim_z == 1:
        axs = [axs]
    enc_x = [ob.label for ob in all_enc_z]
    plus_x = [ob.label_c for ob in all_plus_z]
    for i in range(0, dim_z):
        enc_y = [ob.z.cpu()[i].item() for ob in all_enc_z]
        plus_y = [ob.plus_c_z.cpu()[i].item() for ob in all_plus_z]
        axs[i].scatter(enc_x, enc_y, edgecolors='blue', label='Encoder output', facecolors='none', s=60)
        axs[i].scatter(plus_x, plus_y, edgecolors='red', label='Plus output', facecolors='none', s=20)
        axs[i].set(xlabel='Num of Points on the card', xticks=range(0, max(enc_x) + 1))
        if eval_helper is not None:
            eval_helper.draw_scatter_point_line(axs[i], i, [*enc_x, *plus_x], [*enc_y, *plus_y])
            eval_helper.set_axis(axs[i], i)
        else:
            axs[i].grid(True)
            axs[i].set_title(f"z_c ({i + 1})")
    # for ax in axs.flat:
    #     ax.label_outer()
    plt.legend()
    plt.savefig(f'{eval_path}_ks_{ks_mean}.png')
    plt.cla()
    plt.clf()
    plt.close()


class LabelKs:
    def __init__(self, label: int):
        self.label = label
        self.enc_z_list = []
        self.plus_z_list = []
        self.min_len = -1
        self.ks = -1

    def calc_min_len(self):
        self.min_len = min(len(self.enc_z_list), len(self.plus_z_list))
        return self.min_len

    def calc_ks(self):
        min_len = self.calc_min_len()
        sample1 = random.sample(self.enc_z_list, min_len)
        sample2 = random.sample(self.plus_z_list, min_len)
        ks, p_value = stats.ks_2samp(sample1, sample2)
        self.ks = ks
        return ks, p_value


def calc_ks_enc_plus_z(enc_z: List[EncZ], plus_z: List[PlusZ]):
    min_label = min(enc_z, key=lambda x: x.label).label
    max_label = max(enc_z, key=lambda x: x.label).label
    label_dict = {}
    for i in range(min_label, max_label+1):
        label_dict[i] = LabelKs(i)
    for item in enc_z:
        label_dict[item.label].enc_z_list.append(item.z.cpu().item())
    for item in plus_z:
        label_dict[item.label_c].plus_z_list.append(item.plus_c_z.cpu().item())
    label_ks_list = list(label_dict.values())
    ks_all = 0
    item_num = 0
    for item in label_ks_list:
        item.calc_min_len()
        if item.min_len >= MIN_KS_NUM:
            item.calc_ks()
            ks_all += item.ks
            item_num += 1
    ks_mean = ks_all / item_num
    return round(ks_mean, 4)


class VQvaePlusEval:
    def __init__(self, config, data_loader: DataLoader, model_path=None, loaded_model: VQVAE = None):
        self.loader = data_loader
        self.zc_dim = config['latent_embedding_1'] * config['embedding_dim']
        assert model_path is not None or loaded_model is not None, "model_path and loaded_model cannot both be None"
        if loaded_model is not None:
            self.model = loaded_model
        else:
            self.model = VQVAE(config).to(DEVICE)
            self.model.load_state_dict(self.model.load_tensor(model_path))
        self.isVAE = config['kld_loss_scalar'] > 0.00001
        self.model.eval()

    def load_plusZ_eval_data(self):
        all_enc_z = []
        all_plus_z = []
        for batch_ndx, sample in enumerate(self.loader):
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
            idx_z_a = self.model.find_indices(za, False)
            idx_z_b = self.model.find_indices(zb, False)
            idx_z_c = self.model.find_indices(zc, False)
            idx_plus_c = self.model.find_indices(plus_c, False)
            for i in range(0, za.size(0)):
                enc_z_list.append(EncZ(label_a[i], idx_z_a[i]))
                enc_z_list.append(EncZ(label_b[i], idx_z_b[i]))
                enc_z_list.append(EncZ(label_c[i], idx_z_c[i]))
                plus_z_list.append(PlusZ(label_a[i], label_b[i], idx_plus_c[i]))
            all_enc_z.extend(enc_z_list)
            all_plus_z.extend(plus_z_list)
        return all_enc_z, all_plus_z

    def eval(self, eval_path):
        all_enc_z, all_plus_z = self.load_plusZ_eval_data()
        plot_plusZ_against_label(all_enc_z, all_plus_z, eval_path)


# if __name__ == "__main__":
#     os.makedirs(EVAL_ROOT, exist_ok=True)
#     dataset = Dataset(CONFIG['train_data_path'])
#     loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
#     evaler = VQvaePlusEval(CONFIG, loader, model_path=MODEL_PATH)
#     result_path = os.path.join(EVAL_ROOT, f'plus_eval_{MODEL_PATH}.png')
#     evaler.eval(result_path)